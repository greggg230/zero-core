//! A self-play job: the games `gid_base .. gid_base + games` of a run's self-play driver (the Python
//! remote worker's: rolling, queue_only, its seed; `settings::selfplay_driver_cfg`) played with one
//! network, their samples written as shards (`shard`). The loop is the remote worker's
//! (`remoteplay.Worker._phase` + `infer.drive`): `resume`, then `pump` (take the finished games'
//! results and samples, write them, queue a setup of the run's spec for every game it may start -
//! at most `conc` in flight, `games` in all), the drive loop with up to two network batches in flight
//! whose answers are submitted in launch order (`infer._drive_pipelined`) and a `pump` after every
//! submitted batch. The evaluator runs on a thread of its own, so the next batch is searched while
//! one is evaluated.
//!
//! Control (`Control`, from the agent's lines; zero-farm PROTOCOL.md 5, `aw/1`): `yield` - no batch is
//! started any more, the ones in flight finish, the evaluator drops its session; `resume` - it is
//! rebuilt and the same games go on (with a deterministic batch order the rows are those of a job
//! never yielded); `finish` - no new setup, the games in flight finish; `quit` - the driver's phase
//! ends now (games in flight are lost); `checkpoint` - the remote worker's phase change: the
//! driver's phase ends, the new network is loaded and the driver resumes with its id (games in
//! flight go on; their searches restart with the new network, each row keeps the id of the network
//! that searched it).
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use serde_json::{json, Map, Value};
use zero_search::driver::{Batch, NnSeat, SeatCfg};
use zero_search::dynapi::DynDriver;

use crate::eval::{Answer, Evaluator, Expect, OrtCfg, OrtEvaluator, Uniform};
use crate::settings::{selfplay_driver_cfg, spec_of, Settings};
use crate::shard::{GameSamples, Npy, ShardWriter, Written};
use crate::GameEntry;

/// A rolling driver's quota when its phase ends by `end_phase` (zero-core `remoteplay.BIG`).
pub const BIG: u64 = 1 << 62;
/// Network batches in flight (`infer.drive`'s ZERO_INFLIGHT default).
pub const INFLIGHT: usize = 2;

#[derive(Clone, Debug)]
pub enum EvalKind {
    /// A `zero-onnx/1` graph through ONNX Runtime; `reference`: its checkpoint's reference set, answered
    /// before the first real batch (the per-machine self-test, `selftest`).
    Onnx { model: PathBuf, ort: OrtCfg, sha256: Option<String>, reference: Option<PathBuf> },
    /// `infer.UniformEvaluator` (tests and the plumbing gate).
    Uniform,
}

#[derive(Clone, Debug)]
pub struct Job {
    pub lease_id: String,
    pub settings: Settings,
    pub eval: EvalKind,
    pub games: u64,
    pub gid_base: u32,
    pub seed: u64,
    /// The network id the rows carry (the checkpoint's generation; the remote worker's it - 1).
    pub net: u32,
    /// Search threads (the driver's); the settings' `threads` when None.
    pub search_threads: Option<usize>,
    /// Overrides of the settings' `conc` / `groups` (a farm job's own; None: the run's).
    pub conc: Option<usize>,
    pub groups: Option<usize>,
    pub out_dir: PathBuf,
    /// Shard file names: `<prefix>_SSS.npz`.
    pub prefix: String,
    /// Extra shard meta (data.py's keys besides game / features / actions / rows).
    pub meta: Map<String, Value>,
    /// Keep the first n network batches as sent to the evaluator (onnx_record.py's format) in this file.
    pub record_batches: Option<(PathBuf, usize)>,
    pub status_s: f64,
}

/// A checkpoint to switch to.
#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub gen: u32,
    pub model: PathBuf,
    pub sha256: Option<String>,
    /// Its reference set (the self-test runs before the new network answers a batch).
    pub reference: Option<PathBuf>,
}

#[derive(Default)]
pub(crate) struct CtlState {
    pub(crate) yielded: bool,
    pub(crate) quit: bool,
    pub(crate) finish: bool,
    pub(crate) checkpoint: Option<Checkpoint>,
}

/// The job's control state, shared by the agent's line reader and the job.
#[derive(Default)]
pub struct Control {
    pub(crate) s: Mutex<CtlState>,
    pub(crate) cv: Condvar,
}

impl Control {
    pub fn quit(&self) {
        let mut s = self.s.lock().unwrap();
        s.quit = true;
        s.yielded = false;
        self.cv.notify_all();
    }
    pub fn finish(&self) {
        self.s.lock().unwrap().finish = true;
    }
    pub fn checkpoint(&self, c: Checkpoint) {
        self.s.lock().unwrap().checkpoint = Some(c);
    }
    pub fn is_yielded(&self) -> bool {
        self.s.lock().unwrap().yielded
    }
    fn set_yielded(&self, on: bool) -> bool {
        let mut s = self.s.lock().unwrap();
        let was = s.yielded;
        s.yielded = on;
        self.cv.notify_all();
        was != on
    }
}

pub(crate) enum Req {
    Eval(Batch),
    Release(Sender<()>),
    Restore(Sender<Result<(), String>>),
    Swap(Box<dyn Evaluator>, Sender<()>),
    /// The job is over (the agent's handle may outlive it: the thread ends on this, not on the
    /// channel's close).
    Stop,
}

/// The evaluator thread's handle (the agent's yield / resume reach it too).
#[derive(Clone)]
pub struct EvalLink {
    pub(crate) tx: Sender<Req>,
}

impl EvalLink {
    /// After the batches sent so far: drop the session.
    pub fn release(&self) {
        let (t, r) = channel();
        if self.tx.send(Req::Release(t)).is_ok() {
            let _ = r.recv();
        }
    }
    pub fn restore(&self) -> Result<(), String> {
        let (t, r) = channel();
        self.tx.send(Req::Restore(t)).map_err(|_| "the evaluator thread is gone".to_string())?;
        r.recv().map_err(|_| "the evaluator thread is gone".to_string())?
    }
}

/// Throughput counters (zero-farm PROTOCOL.md 3 `Stats`, the ones a self-play job has).
#[derive(Clone, Debug, Default)]
pub struct Progress {
    pub games: u64,
    pub started: u64,
    pub rows: u64,
    pub shards: u64,
    pub leaves: u64,
    pub sims: u64,
    pub decisions: u64,
    pub infer_batches: u64,
    pub infer_rows: u64,
    pub infer_ms: f64,
    pub wait_ms: f64,
    /// The search threads' busy time (the driver's `busy_s`).
    pub search_ms: f64,
    pub batch_ms: Vec<f32>,
    pub yields: u64,
    pub yield_ms_max: f64,
    pub concurrency: usize,
    pub threads: usize,
    pub wall_s: f64,
    pub provider: String,
    pub finished_gids: Vec<u32>,
    /// The last self-test's largest difference.
    pub selftest_max_abs: Option<f64>,
}

impl Progress {
    pub fn stats(&self) -> Value {
        let mut ms = self.batch_ms.clone();
        ms.sort_by(|a, b| a.total_cmp(b));
        let q = |p: f64| if ms.is_empty() { Value::Null } else { json!((ms[((ms.len() - 1) as f64 * p).round() as usize] as f64 * 1000.0).round() / 1000.0) };
        json!({
            "games": self.games, "decisions": self.decisions, "leaves": self.leaves, "infer_batches": self.infer_batches, "search_ms": self.search_ms.round(),
            "infer_ms_p50": q(0.5), "infer_ms_p95": q(0.95), "infer_ms": self.infer_ms.round(), "wait_ms": self.wait_ms.round(),
            "rss_mb": rss_mb(), "threads": self.threads, "concurrency": self.concurrency, "provider": self.provider,
            "yields": self.yields, "yield_ms_max": (self.yield_ms_max * 10.0).round() / 10.0,
            "rows": self.rows, "shards": self.shards, "sims": self.sims, "games_started": self.started,
            "infer_rows": self.infer_rows, "wall_s": (self.wall_s * 10.0).round() / 10.0,
            "selftest_max_abs": self.selftest_max_abs.map(|m| if m.is_finite() { json!(m) } else { json!("inf") }),
            "leaves_per_s": if self.wall_s > 0.0 { ((self.leaves as f64 / self.wall_s) * 10.0).round() / 10.0 } else { 0.0 },
        })
    }
}

#[cfg(windows)]
mod mem {
    #[repr(C)]
    pub struct Pmc {
        pub cb: u32,
        pub page_faults: u32,
        pub peak_working_set: usize,
        pub working_set: usize,
        pub rest: [usize; 6],
    }
    #[link(name = "psapi")]
    extern "system" {
        pub fn GetProcessMemoryInfo(h: *mut core::ffi::c_void, pmc: *mut Pmc, cb: u32) -> i32;
    }
    extern "system" {
        pub fn GetCurrentProcess() -> *mut core::ffi::c_void;
    }
    /// (working set, peak working set) in bytes.
    pub fn now() -> Option<(usize, usize)> {
        let mut p: Pmc = unsafe { std::mem::zeroed() };
        p.cb = std::mem::size_of::<Pmc>() as u32;
        (unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut p, p.cb) } != 0).then_some((p.working_set, p.peak_working_set))
    }
}

/// The process's resident set (MB; Windows: the working set; elsewhere /proc's VmRSS), 0 if unknown.
pub fn rss_mb() -> u64 {
    #[cfg(windows)]
    {
        mem::now().map_or(0, |m| (m.0 >> 20) as u64)
    }
    #[cfg(not(windows))]
    {
        std::fs::read_to_string("/proc/self/status").ok().and_then(|s| s.lines().find(|l| l.starts_with("VmRSS:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())).map_or(0, |kb| kb / 1024)
    }
}

/// The peak working set (MB; Windows only, else 0).
pub fn peak_rss_mb() -> u64 {
    #[cfg(windows)]
    {
        mem::now().map_or(0, |m| (m.1 >> 20) as u64)
    }
    #[cfg(not(windows))]
    {
        0
    }
}

/// Events out (`aw/1` lines).
pub trait Sink: Send + Sync {
    fn event(&self, v: Value);
    fn log(&self, msg: &str) {
        eprintln!("{msg}");
    }
}

/// How a job ended.
#[derive(Clone, Debug)]
pub struct Outcome {
    /// 'done' | 'finish' | 'quit'
    pub reason: &'static str,
    pub stats: Value,
    /// Game ids of the job that did not finish (half-open ranges).
    pub unfinished: Vec<(u32, u32)>,
    pub shards: Vec<Written>,
}

/// Ranges [a, b) of the ids in `lo..hi` not in `done`.
pub fn missing_ranges(lo: u32, hi: u32, done: &[u32]) -> Vec<(u32, u32)> {
    let mut d: Vec<u32> = done.iter().copied().filter(|&g| g >= lo && g < hi).collect();
    d.sort_unstable();
    d.dedup();
    let mut out = Vec::new();
    let mut at = lo;
    for g in d {
        if g > at {
            out.push((at, g));
        }
        at = g + 1;
    }
    if at < hi {
        out.push((at, hi));
    }
    out
}

/// Ranges [a, b) covering `gids`.
pub fn ranges(gids: &[u32]) -> Vec<(u32, u32)> {
    let mut d = gids.to_vec();
    d.sort_unstable();
    d.dedup();
    let mut out: Vec<(u32, u32)> = Vec::new();
    for g in d {
        match out.last_mut() {
            Some(r) if r.1 == g => r.1 = g + 1,
            _ => out.push((g, g + 1)),
        }
    }
    out
}

/// The job's evaluator; a network with a reference set first passes the self-test (`selftest`: the
/// event, and an error - the job's end - if it fails: a worker never plays with an evaluator that
/// doesn't answer as torch does). `gen`: the network's generation in an eval job (the event names it).
pub(crate) fn build_eval(kind: &EvalKind, want: &Expect, sink: &dyn Sink, progress: &Mutex<Progress>, gen: Option<u32>) -> Result<Box<dyn Evaluator>, String> {
    Ok(match kind {
        EvalKind::Uniform => Box::new(Uniform),
        EvalKind::Onnx { model, ort, sha256, reference } => {
            let mut w = want.clone();
            w.sha256 = sha256.clone();
            let mut ev = OrtEvaluator::load(model, ort.clone(), &w)?;
            match reference {
                Some(r) => {
                    let t0 = Instant::now();
                    let set = crate::selftest::load(r)?;
                    crate::selftest::check_meta(&set, &w, &ev.meta)?;
                    let st = crate::selftest::run(&mut ev, &set)?;
                    // (the self-test's batches are not the job's)
                    ev.calls = 0;
                    ev.rows = 0;
                    ev.run_s = 0.0;
                    ev.batch_ms.clear();
                    let max_abs = if st.max_abs.is_finite() { json!(st.max_abs) } else { json!("inf") };
                    let mut ev_line = json!({"ev": "selftest", "provider": "cpu", "max_abs": max_abs, "ok": st.ok, "rows": st.rows,
                                      "items": {"priors": st.items[0], "value": st.items[1], "margin": st.items[2], "aux": st.items[3]},
                                      "seconds": (t0.elapsed().as_secs_f64() * 1000.0).round() / 1000.0});
                    if let Some(g) = gen {
                        ev_line["gen"] = json!(g);
                    }
                    sink.event(ev_line);
                    progress.lock().unwrap().selftest_max_abs = Some(st.max_abs);
                    if !st.ok {
                        return Err(format!(
                            "selftest failed{}: the network's answers on this machine differ from torch's by {:e} (> {:e}; priors {:e}, value {:e}, margin {:e}, aux {:e}) on {} reference rows ({}): not playing with this evaluator",
                            gen.map_or(String::new(), |g| format!(" (gen {g})")),
                            st.max_abs,
                            crate::selftest::TOL,
                            st.items[0],
                            st.items[1],
                            st.items[2],
                            st.items[3],
                            st.rows,
                            r.display()
                        ));
                    }
                    sink.log(&format!("selftest passed{}: max abs {:e} on {} rows ({})", gen.map_or(String::new(), |g| format!(" (gen {g})")), st.max_abs, st.rows, r.display()));
                }
                None => sink.log("no reference set: the self-test is skipped"),
            }
            Box::new(ev)
        }
    })
}

/// The evaluator thread: answers in the order asked.
#[allow(clippy::type_complexity)]
pub(crate) fn spawn_eval(mut ev: Box<dyn Evaluator>) -> (EvalLink, Receiver<(u64, Result<Answer, String>, f64)>, std::thread::JoinHandle<()>) {
    let (tx, rx) = channel::<Req>();
    let (atx, arx) = channel();
    let h = std::thread::Builder::new()
        .name("zero-worker-eval".into())
        .spawn(move || {
            for r in rx {
                match r {
                    Req::Eval(b) => {
                        let t0 = Instant::now();
                        let a = ev.evaluate(&b);
                        if atx.send((b.id, a, t0.elapsed().as_secs_f64())).is_err() {
                            return;
                        }
                    }
                    Req::Release(done) => {
                        ev.release();
                        let _ = done.send(());
                    }
                    Req::Restore(done) => {
                        let _ = done.send(ev.restore());
                    }
                    Req::Swap(new, done) => {
                        ev = new;
                        let _ = done.send(());
                    }
                    Req::Stop => return,
                }
            }
        })
        .expect("spawn the evaluator thread");
    (EvalLink { tx }, arx, h)
}

/// The recorded network batches (`tools/onnx_record.py`'s npz).
pub(crate) struct Recorder {
    path: PathBuf,
    max: usize,
    batches: usize,
    xs_off: Vec<i64>,
    xs_idx: Vec<i32>,
    xs_val: Vec<f32>,
    n_players: Vec<i64>,
    player: Vec<i32>,
    gid: Vec<i64>,
    extra: Vec<u8>,
    opt_off: Vec<i64>,
    opt_action: Vec<i32>,
    batch_off: Vec<i64>,
    ids: Vec<i64>,
}

impl Recorder {
    pub(crate) fn new(path: PathBuf, max: usize) -> Recorder {
        Recorder { path, max, batches: 0, xs_off: vec![0], xs_idx: vec![], xs_val: vec![], n_players: vec![], player: vec![], gid: vec![], extra: vec![], opt_off: vec![0], opt_action: vec![], batch_off: vec![0], ids: vec![] }
    }
    pub(crate) fn add(&mut self, b: &Batch) {
        if self.batches >= self.max {
            return;
        }
        let base = *self.xs_off.last().unwrap();
        for r in 0..b.b {
            self.xs_off.push(base + b.xs_off[r + 1]);
        }
        self.xs_idx.extend(b.xs_idx.iter().map(|&i| i as i32));
        self.xs_val.extend_from_slice(&b.xs_val);
        self.n_players.extend(b.n_players.iter().map(|&x| x as i64));
        self.player.extend_from_slice(&b.player);
        self.gid.extend_from_slice(&b.gid);
        self.extra.extend_from_slice(&b.extra);
        let ob = *self.opt_off.last().unwrap();
        for r in 0..b.b {
            self.opt_off.push(ob + b.opt_off[r + 1] as i64);
        }
        self.opt_action.extend_from_slice(&b.opt_action);
        self.batch_off.push(self.batch_off.last().unwrap() + b.b as i64);
        self.ids.push(b.id as i64);
        self.batches += 1;
    }
    pub(crate) fn write(&self, meta: &Value) -> Result<(), String> {
        let f32n = |v: &[f32]| Npy { descr: "<f4".into(), shape: vec![v.len()], data: v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let u8n = |v: &[u8]| Npy { descr: "|u1".into(), shape: vec![v.len()], data: v.to_vec() };
        let members = vec![
            ("meta", Npy::unicode(&meta.to_string())),
            ("xs_off", Npy::i64(&self.xs_off, vec![self.xs_off.len()])),
            ("xs_idx", Npy::i32(&self.xs_idx, vec![self.xs_idx.len()])),
            ("xs_val", f32n(&self.xs_val)),
            ("n_players", Npy::i64(&self.n_players, vec![self.n_players.len()])),
            ("player", Npy::i32(&self.player, vec![self.player.len()])),
            ("gid", Npy::i64(&self.gid, vec![self.gid.len()])),
            ("extra", u8n(&self.extra)),
            ("opt_off", Npy::i64(&self.opt_off, vec![self.opt_off.len()])),
            ("opt_action", Npy::i32(&self.opt_action, vec![self.opt_action.len()])),
            ("batch_off", Npy::i64(&self.batch_off, vec![self.batch_off.len()])),
            ("batch_id", Npy::i64(&self.ids, vec![self.ids.len()])),
        ];
        crate::shard::write_npz(&self.path, &members, 1).map(|_| ())
    }
}

/// The job's driver settings: the Python remote worker's driver for the run's settings
/// (`selfplay_driver_cfg`), then the native worker's own: sparse rows (the graph reads CSR; the same
/// rows), the job's search threads, conc and groups.
pub fn job_driver_cfg(game: &GameEntry, job: &Job) -> Result<zero_search::driver::DriverCfg, String> {
    let info = (game.info)();
    let mut cfg = selfplay_driver_cfg(&job.settings, &info, job.seed, job.net)?;
    cfg.sparse = true;
    if let Some(t) = job.search_threads {
        cfg.threads = t;
    }
    if let Some(c) = job.conc {
        cfg.concurrency = c;
    }
    if let Some(g) = job.groups {
        cfg.groups = g;
    }
    Ok(cfg)
}

struct Run<'a> {
    job: &'a Job,
    sink: &'a dyn Sink,
    ctl: &'a Control,
    drv: Box<dyn DynDriver>,
    link: EvalLink,
    answers: Receiver<(u64, Result<Answer, String>, f64)>,
    pending: VecDeque<u64>,
    writer: ShardWriter,
    spec: String,
    cap: u64,
    enq: u64,
    finished: u64,
    stop_topup: bool,
    ended: bool,
    switching: Option<Checkpoint>,
    progress: Arc<Mutex<Progress>>,
    recorder: Option<Recorder>,
    want: Expect,
    t0: Instant,
    t_status: Instant,
    fin_unsent: u64,
}

impl Run<'_> {
    fn snapshot(&mut self) {
        let st = self.drv.stats();
        let mut p = self.progress.lock().unwrap();
        p.games = self.finished;
        p.started = self.enq - self.drv.queued() as u64;
        p.leaves = st.leaves;
        p.sims = st.sims;
        p.wait_ms = st.wait_s * 1e3;
        p.search_ms = st.busy_s * 1e3;
        p.wall_s = self.t0.elapsed().as_secs_f64();
    }

    fn emit_shard(&mut self, w: Written) {
        {
            let mut p = self.progress.lock().unwrap();
            p.rows += w.rows as u64;
            p.shards += 1;
        }
        let rel = w.path.strip_prefix(&self.job.out_dir).map(|p| p.to_path_buf()).unwrap_or_else(|_| w.path.clone());
        let gids: Vec<[u32; 2]> = ranges(&w.gids).into_iter().map(|r| [r.0, r.1]).collect();
        self.sink.event(json!({"ev": "shard", "seq": w.seq, "path": rel.to_string_lossy().replace('\\', "/"), "games": w.games, "rows": w.rows, "bytes": w.bytes, "gids": gids, "gen": w.gen}));
    }

    /// The remote worker's `pump`: control, finished games into the shards, setups queued, status.
    fn pump(&mut self) -> Result<(), String> {
        let (quit, finish, ck) = {
            let mut s = self.ctl.s.lock().unwrap();
            (s.quit, s.finish, s.checkpoint.take())
        };
        if quit && !self.ended {
            self.drv.end_phase();
            self.ended = true;
            self.stop_topup = true;
        }
        if finish {
            self.stop_topup = true;
        }
        if let Some(c) = ck {
            if !self.ended {
                self.drv.end_phase();
                self.switching = Some(c);
            }
        }
        let rows = self.drv.take_results();
        let recs = self.drv.take_samples();
        if rows.is_empty() && !recs.is_empty() {
            return Err("samples without finished games".into());
        }
        let mut it = recs.into_iter();
        let n = rows.len() as u64;
        for r in rows {
            let k = r.samples as usize;
            let records: Vec<_> = it.by_ref().take(k).collect();
            if records.len() != k {
                return Err(format!("game {} has {k} samples, {} left", r.gid, records.len()));
            }
            {
                // (Stats: the finished games' searched decisions)
                let mut p = self.progress.lock().unwrap();
                p.finished_gids.push(r.gid);
                p.decisions += r.decisions as u64;
            }
            if let Some(w) = self.writer.add(GameSamples { gid: r.gid, records })? {
                self.emit_shard(w);
            }
        }
        if it.next().is_some() {
            return Err("samples left over after the finished games".into());
        }
        self.finished += n;
        self.fin_unsent += n;
        if self.fin_unsent > 0 {
            self.sink.event(json!({"ev": "fin", "n": self.fin_unsent}));
            self.fin_unsent = 0;
        }
        if !self.stop_topup && self.switching.is_none() {
            let outstanding = self.enq - self.finished;
            let room = self.cap.saturating_sub(outstanding).min(self.job.games - self.enq);
            for _ in 0..room {
                let tag = format!("{}:{}", self.job.lease_id, self.enq);
                self.drv.enqueue(&tag, &self.spec)?;
                self.enq += 1;
            }
        }
        self.snapshot();
        if self.t_status.elapsed().as_secs_f64() >= self.job.status_s {
            self.t_status = Instant::now();
            let s = self.progress.lock().unwrap().stats();
            self.sink.event(json!({"ev": "status", "stats": s}));
        }
        Ok(())
    }

    fn send(&mut self, b: Batch) -> Result<(), String> {
        if let Some(r) = &mut self.recorder {
            r.add(&b);
        }
        let mut s = self.ctl.s.lock().unwrap();
        while s.yielded && !s.quit {
            s = self.ctl.cv.wait(s).unwrap();
        }
        self.pending.push_back(b.id);
        self.link.tx.send(Req::Eval(b)).map_err(|_| "the evaluator thread is gone".to_string())
    }

    fn finish_oldest(&mut self) -> Result<(), String> {
        let want = self.pending.pop_front().expect("a batch in flight");
        let (id, a, secs) = self.answers.recv().map_err(|_| "the evaluator thread is gone".to_string())?;
        if id != want {
            return Err(format!("internal: answer of batch {id}, expected {want}"));
        }
        let a = a?;
        {
            let mut p = self.progress.lock().unwrap();
            p.infer_batches += 1;
            p.infer_ms += secs * 1e3;
            if p.batch_ms.len() < 1 << 20 {
                p.batch_ms.push((secs * 1e3) as f32);
            }
        }
        self.drv.submit(id, &a.priors, &a.value, &a.margin, &a.aux)?;
        self.pump()
    }

    /// `infer._drive_pipelined` until the driver has no batch (every game done, or the phase ended).
    fn drive(&mut self) -> Result<(), String> {
        loop {
            let b = if self.pending.is_empty() { self.drv.next_batch(true)? } else { self.drv.next_batch(false)? };
            match b {
                None => {
                    if self.pending.is_empty() {
                        return Ok(());
                    }
                    self.finish_oldest()?;
                }
                Some(b) => {
                    if b.external {
                        return Err("an external batch: the native worker has no external players".into());
                    }
                    self.progress.lock().unwrap().infer_rows += b.b as u64;
                    self.send(b)?;
                    if self.pending.len() >= INFLIGHT {
                        self.finish_oldest()?;
                    }
                }
            }
        }
    }
}

/// Run `job` to its end. `link_out` receives the evaluator link (the agent's yield / resume) once the
/// evaluator is loaded.
pub fn run_job(game: &GameEntry, job: &Job, ctl: &Control, sink: &dyn Sink, progress: Arc<Mutex<Progress>>, link_out: &dyn Fn(EvalLink)) -> Result<Outcome, String> {
    let t0 = Instant::now();
    let info = (game.info)();
    let cfg = job_driver_cfg(game, job)?;
    let belief = cfg.seats.iter().any(|s| matches!(s, SeatCfg::Nn(n) if NnSeat::belief_on(n)));
    let want = Expect { game: info.name.to_string(), features: info.num_features, actions: info.num_actions, aux_width: info.aux.iter().map(|a| a.1).sum(), belief, sha256: None };
    let ev = build_eval(&job.eval, &want, sink, &progress, None)?;
    sink.log(&format!("job {}: {} games from gid {}, seed {}, net {}; driver: conc {}, threads {}, groups {}, sparse; evaluator: {}", job.lease_id, job.games, job.gid_base, job.seed, job.net, cfg.concurrency, cfg.threads, cfg.groups, ev.describe()));
    {
        let mut p = progress.lock().unwrap();
        p.concurrency = cfg.concurrency;
        p.threads = cfg.threads;
        p.provider = match &job.eval {
            EvalKind::Uniform => "uniform".into(),
            EvalKind::Onnx { .. } => "cpu".into(),
        };
    }
    if cfg.concurrency == 0 {
        return Err("conc 0: no game would ever start (the setup cap is the games in flight)".into());
    }
    let cap = cfg.concurrency as u64;
    let spec = spec_of(&job.settings)?;
    let mut drv = (game.new_driver)(cfg)?;
    if job.gid_base != 0 {
        drv.set_first_gid(job.gid_base)?;
    }
    drv.resume(BIG, job.net)?;
    let mut meta = Map::new();
    meta.insert("game".into(), json!(info.name));
    meta.insert("features".into(), json!(info.num_features));
    meta.insert("actions".into(), json!(info.num_actions));
    meta.extend(job.meta.clone());
    let sparse = crate::settings::truthy(job.settings.get("shard_sparse").unwrap_or(&crate::pyjson::PyJson::Null));
    let shard_rows = crate::settings::py_int("shard_rows", job.settings.req("shard_rows")?)?;
    let writer = ShardWriter::new(&job.out_dir, &job.prefix, shard_rows.max(1) as usize, meta, sparse, info.num_features, 1)?;
    let (link, answers, eval_thread) = spawn_eval(ev);
    link_out(link.clone());
    let mut r = Run { job, sink, ctl, drv, link, answers, pending: VecDeque::new(), writer, spec, cap, enq: 0, finished: 0, stop_topup: false, ended: false, switching: None, progress: progress.clone(), recorder: job.record_batches.as_ref().map(|(p, n)| Recorder::new(p.clone(), *n)), want, t0, t_status: Instant::now(), fin_unsent: 0 };
    let res = (|| -> Result<(), String> {
        r.pump()?;
        let mut idle = 0;
        loop {
            r.drive()?;
            if let Some(c) = r.switching.take() {
                // the remote worker's phase change: the new network, the driver resumed with its id
                let new = build_eval(&EvalKind::Onnx { model: c.model.clone(), ort: match &r.job.eval {
                    EvalKind::Onnx { ort, .. } => ort.clone(),
                    EvalKind::Uniform => return Err("checkpoint: a uniform-evaluator job has no network to switch".into()),
                }, sha256: c.sha256.clone(), reference: c.reference.clone() }, &r.want, r.sink, &r.progress, None)?;
                let (t, rx) = channel();
                r.link.tx.send(Req::Swap(new, t)).map_err(|_| "the evaluator thread is gone".to_string())?;
                let _ = rx.recv();
                r.drv.resume(BIG, c.gen)?;
                r.sink.log(&format!("checkpoint {} ({}): the driver resumed with it", c.gen, c.model.display()));
                r.pump()?;
                continue;
            }
            if r.ended || r.finished >= r.job.games || (r.stop_topup && r.enq == r.finished) {
                break;
            }
            // nothing to play now (a queue_only driver with an empty queue): wait a little for control
            idle += 1;
            if idle > 1 {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            r.pump()?;
        }
        r.pump()?;
        if let Some(w) = r.writer.flush()? {
            r.emit_shard(w);
        }
        Ok(())
    })();
    let rec_res = match (&r.recorder, &res) {
        (Some(rc), Ok(())) => rc.write(&json!({"game": info.name, "features": info.num_features, "actions": info.num_actions, "seed": job.seed, "worker": "zero_worker"})),
        _ => Ok(()),
    };
    r.snapshot();
    let Run { link, writer, .. } = r;
    let _ = link.tx.send(Req::Stop);
    drop(link);
    let _ = eval_thread.join();
    res?;
    rec_res?;
    let p = progress.lock().unwrap();
    let reason = if p.games >= job.games {
        "done"
    } else if ctl.s.lock().unwrap().quit {
        "quit"
    } else {
        "finish"
    };
    let lo = job.gid_base;
    let hi = job.gid_base + job.games as u32;
    let mut stats = p.stats();
    stats["peak_rss_mb"] = json!(peak_rss_mb());
    // (games after the last shard that had no samples - an aborted game -: in no shard's `games`; the remote
    // worker's 'unshipped')
    stats["unshipped"] = json!(writer.pending_games());
    stats["seconds"] = json!((t0.elapsed().as_secs_f64() * 100.0).round() / 100.0);
    Ok(Outcome { reason, stats, unfinished: missing_ranges(lo, hi, &p.finished_gids), shards: writer.written })
}

/// The yield (`aw/1`): no more batches started, the ones in flight answered, the session dropped.
/// -> the time it took (s); false: already yielded.
pub fn do_yield(ctl: &Control, link: Option<&EvalLink>, progress: &Mutex<Progress>) -> Option<f64> {
    let t0 = Instant::now();
    if !ctl.set_yielded(true) {
        return None;
    }
    if let Some(l) = link {
        l.release();
    }
    let dt = t0.elapsed().as_secs_f64();
    let mut p = progress.lock().unwrap();
    p.yields += 1;
    p.yield_ms_max = p.yield_ms_max.max(dt * 1e3);
    Some(dt)
}

pub fn do_resume(ctl: &Control, link: Option<&EvalLink>) -> Result<bool, String> {
    if let Some(l) = link {
        l.restore()?;
    }
    Ok(ctl.set_yielded(false))
}
