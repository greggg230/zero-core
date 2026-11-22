//! A self-play job: the games `gid_base .. gid_base + games` of a run's self-play driver (the Python
//! remote worker's: rolling, queue_only, its seed; `settings::selfplay_driver_cfg`) played with one
//! network, written as game records (`records`: what replays each game and its search's outputs; zero-farm
//! phase 3) and / or their samples as shards (`shard`; `Job::output`). The loop is the remote worker's
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
//!
//! A rolling job (`Job::rolling`, zero-farm PROTOCOL.md 10.5, zero-core docs/az.md 54) does not end when its `games`
//! are done: it keeps up to `conc` games in flight while ids are left, then idles (the control thread still answers
//! `ping` and sends `status`) until `extend` raises its games or `finish` / `quit` ends it; `low` asks for more ids
//! once per games value when fewer than `conc` are left; a records file (and a shard) is also written `flush_s`
//! seconds after the first game in it finished; and a `checkpoint` builds and self-tests the new network on a
//! thread of its own while the games go on with the old one, then does the phase change above. A job without
//! `rolling` runs exactly as before.
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use serde_json::{json, Map, Value};
use zero_search::driver::{Batch, NnSeat, SeatCfg};
use zero_search::dynapi::DynDriver;

use crate::eval::{Answer, Evaluator, Expect, OrtCfg, Uniform};
use crate::settings::{selfplay_driver_cfg, spec_of, Settings};
use crate::records::{Output, RecordWriter};
use crate::shard::{GameSamples, Npy, ShardWriter, Written};
use crate::GameEntry;

/// A rolling driver's quota when its phase ends by `end_phase` (zero-core `remoteplay.BIG`).
pub const BIG: u64 = 1 << 62;
/// Network batches in flight (`infer.drive`'s ZERO_INFLIGHT default).
pub const INFLIGHT: usize = 2;

#[derive(Clone, Debug)]
pub enum EvalKind {
    /// A `zero-onnx/1` or `/2` graph through ONNX Runtime; `sha256`: the model file's (checked before it is loaded);
    /// `reference`: its checkpoint's reference set, answered before the first real batch (the per-machine
    /// self-test, `selftest`).
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
    /// Game records, shards or both (`records`).
    pub output: Output,
    /// A rolling job (zero-farm PROTOCOL.md 10.5; None: a classic job, exactly as before).
    pub rolling: Option<Rolling>,
}

/// A rolling job's settings (the job line's `rolling: true`, `flush_s`).
#[derive(Clone, Debug)]
pub struct Rolling {
    /// The longest a finished game waits in the worker before its file is written (s).
    pub flush_s: f64,
}

impl Default for Rolling {
    fn default() -> Rolling {
        Rolling { flush_s: 120.0 }
    }
}

/// A checkpoint to switch to.
#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub gen: u32,
    pub model: PathBuf,
    /// The model file's sha256 (checked before it is loaded).
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
    /// A rolling job's games as `extend` raised them (the largest asked; None: never extended).
    pub(crate) extend: Option<u64>,
}

/// The evaluators (one per evaluator thread) a rolling job's checkpoint built that the job has not swapped in yet: a
/// `yield` releases them too (no GPU memory held while yielded; each is rebuilt on its first batch).
type Staged = Arc<Mutex<Option<Vec<Box<dyn Evaluator>>>>>;

/// The job's control state, shared by the agent's line reader and the job.
#[derive(Default)]
pub struct Control {
    pub(crate) s: Mutex<CtlState>,
    pub(crate) cv: Condvar,
    /// A rolling job's next evaluator (`Staged`) and the yielded state as its builder thread reads it.
    staged: Staged,
    yielded_flag: Arc<AtomicBool>,
}

impl Control {
    pub fn quit(&self) {
        let mut s = self.s.lock().unwrap();
        s.quit = true;
        s.yielded = false;
        self.yielded_flag.store(false, Ordering::SeqCst);
        self.cv.notify_all();
    }
    pub fn finish(&self) {
        self.s.lock().unwrap().finish = true;
    }
    pub fn checkpoint(&self, c: Checkpoint) {
        self.s.lock().unwrap().checkpoint = Some(c);
    }
    /// A rolling job's `extend`: its games become `total` (never lower: a lower or equal total than one asked
    /// before is ignored here, and the job ignores one not above its games) -> whether it was taken.
    pub fn extend(&self, total: u64) -> bool {
        let mut s = self.s.lock().unwrap();
        if s.extend.is_some_and(|g| g >= total) {
            return false;
        }
        s.extend = Some(total);
        true
    }
    pub fn is_yielded(&self) -> bool {
        self.s.lock().unwrap().yielded
    }
    fn set_yielded(&self, on: bool) -> bool {
        let mut s = self.s.lock().unwrap();
        let was = s.yielded;
        s.yielded = on;
        self.yielded_flag.store(on, Ordering::SeqCst);
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

/// The evaluator threads' handle (the agent's yield / resume reach them too): one thread per session (`OrtCfg::workers`,
/// one unless the job asks for more), batches sent to them in turn.
#[derive(Clone)]
pub struct EvalLink {
    txs: Vec<Sender<Req>>,
    next: Arc<std::sync::atomic::AtomicUsize>,
}

const GONE: &str = "the evaluator thread is gone";

impl EvalLink {
    /// The evaluator threads.
    pub fn workers(&self) -> usize {
        self.txs.len()
    }
    /// A batch to the next evaluator thread (in turn).
    pub(crate) fn eval(&self, b: Batch) -> Result<(), String> {
        let i = self.next.fetch_add(1, Ordering::SeqCst) % self.txs.len();
        self.txs[i].send(Req::Eval(b)).map_err(|_| GONE.to_string())
    }
    /// A batch to evaluator thread `i` (an eval job's lane: the thread of the batch's key, docs/az.md 71).
    pub(crate) fn eval_to(&self, i: usize, b: Batch) -> Result<(), String> {
        let tx = self.txs.get(i).ok_or_else(|| format!("internal: evaluator thread {i} of {}", self.txs.len()))?;
        tx.send(Req::Eval(b)).map_err(|_| GONE.to_string())
    }
    /// After the batches sent so far: drop the sessions.
    pub fn release(&self) {
        let rs: Vec<_> = self.txs.iter().filter_map(|tx| {
            let (t, r) = channel();
            tx.send(Req::Release(t)).is_ok().then_some(r)
        }).collect();
        for r in rs {
            let _ = r.recv();
        }
    }
    pub fn restore(&self) -> Result<(), String> {
        let mut rs = Vec::new();
        for tx in &self.txs {
            let (t, r) = channel();
            tx.send(Req::Restore(t)).map_err(|_| GONE.to_string())?;
            rs.push(r);
        }
        let mut out = Ok(());
        for r in rs {
            let x = r.recv().map_err(|_| GONE.to_string()).and_then(|x| x);
            if out.is_ok() {
                out = x;
            }
        }
        out
    }
    /// New evaluators, one per thread (after the batches sent so far).
    pub(crate) fn swap(&self, evs: Vec<Box<dyn Evaluator>>) -> Result<(), String> {
        if evs.len() != self.txs.len() {
            return Err(format!("internal: {} evaluators for {} evaluator threads", evs.len(), self.txs.len()));
        }
        let mut rs = Vec::new();
        for (tx, ev) in self.txs.iter().zip(evs) {
            let (t, r) = channel();
            tx.send(Req::Swap(ev, t)).map_err(|_| GONE.to_string())?;
            rs.push(r);
        }
        for r in rs {
            let _ = r.recv();
        }
        Ok(())
    }
    /// The job is over.
    pub(crate) fn stop(&self) {
        for tx in &self.txs {
            let _ = tx.send(Req::Stop);
        }
    }
}

/// A sink that drops everything (the second and further sessions of a job: the first's events stand for them).
pub(crate) struct NoSink;
impl Sink for NoSink {
    fn event(&self, _: Value) {}
    fn log(&self, _: &str) {}
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
    /// Network batches by rows, in powers of two (bucket k: 2^k <= rows < 2^(k+1)): `status` and `end` show them as
    /// `batch_rows_hist` only when the environment sets ZERO_WORKER_BATCH_HIST=1 (tools/worker_bench.py).
    pub batch_rows_log2: [u64; 24],
    pub yields: u64,
    pub yield_ms_max: f64,
    pub concurrency: usize,
    pub threads: usize,
    pub wall_s: f64,
    pub provider: String,
    pub finished_gids: Vec<u32>,
    /// Branched games (zero-core docs/az.md 24, 72; all 0 without): branch rows finished, the near-tie branches among
    /// them that merged back into their game, the near-tie rule's eligible decisions of the finished games.
    pub branches: u64,
    pub merges: u64,
    pub neartie: u64,
    /// The last self-test's largest difference.
    pub selftest_max_abs: Option<f64>,
}

impl Progress {
    pub fn stats(&self) -> Value {
        let mut ms = self.batch_ms.clone();
        ms.sort_by(|a, b| a.total_cmp(b));
        let q = |p: f64| if ms.is_empty() { Value::Null } else { json!((ms[((ms.len() - 1) as f64 * p).round() as usize] as f64 * 1000.0).round() / 1000.0) };
        let mut v = json!({
            "games": self.games, "decisions": self.decisions, "leaves": self.leaves, "infer_batches": self.infer_batches, "search_ms": self.search_ms.round(),
            "infer_ms_p50": q(0.5), "infer_ms_p95": q(0.95), "infer_ms": self.infer_ms.round(), "wait_ms": self.wait_ms.round(),
            "rss_mb": rss_mb(), "threads": self.threads, "concurrency": self.concurrency, "provider": self.provider,
            "yields": self.yields, "yield_ms_max": (self.yield_ms_max * 10.0).round() / 10.0,
            "rows": self.rows, "shards": self.shards, "sims": self.sims, "games_started": self.started,
            "infer_rows": self.infer_rows, "wall_s": (self.wall_s * 10.0).round() / 10.0,
            "selftest_max_abs": self.selftest_max_abs.map(|m| if m.is_finite() { json!(m) } else { json!("inf") }),
            "leaves_per_s": if self.wall_s > 0.0 { ((self.leaves as f64 / self.wall_s) * 10.0).round() / 10.0 } else { 0.0 },
        });
        if self.branches > 0 || self.neartie > 0 {
            // (branched games: only a job that plays them - every other job's stats as before)
            v["branches"] = json!(self.branches);
            v["merges"] = json!(self.merges);
            v["neartie"] = json!(self.neartie);
        }
        if batch_hist_on() {
            let h: serde_json::Map<String, Value> =
                self.batch_rows_log2.iter().enumerate().filter(|x| *x.1 > 0).map(|(k, n)| ((1u64 << k).to_string(), json!(n))).collect();
            v["batch_rows_hist"] = Value::Object(h);
        }
        v
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

/// The process's resident set (MB; Windows: the working set; macOS: the physical footprint, what the agent's memory
/// cap reads; elsewhere /proc's VmRSS), 0 if unknown.
pub fn rss_mb() -> u64 {
    #[cfg(windows)]
    {
        mem::now().map_or(0, |m| (m.0 >> 20) as u64)
    }
    #[cfg(target_os = "macos")]
    {
        crate::platform::darwin::usage(std::process::id()).map_or(0, |u| u.footprint >> 20)
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        std::fs::read_to_string("/proc/self/status").ok().and_then(|s| s.lines().find(|l| l.starts_with("VmRSS:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())).map_or(0, |kb| kb / 1024)
    }
}

/// The peak working set (MB; Windows: the peak working set; macOS: the largest physical footprint; else 0).
pub fn peak_rss_mb() -> u64 {
    #[cfg(windows)]
    {
        mem::now().map_or(0, |m| (m.1 >> 20) as u64)
    }
    #[cfg(target_os = "macos")]
    {
        crate::platform::darwin::usage(std::process::id()).map_or(0, |u| u.max_footprint >> 20)
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        0
    }
}

/// ZERO_WORKER_BATCH_HIST=1: the stats show the batches' rows (`Progress::batch_rows_log2`).
fn batch_hist_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ZERO_WORKER_BATCH_HIST").as_deref() == Ok("1"))
}

/// The power-of-two bucket of a batch of `rows` rows (`Progress::batch_rows_log2`).
pub fn rows_bucket(rows: usize) -> usize {
    (usize::BITS - 1 - rows.max(1).leading_zeros()).min(23) as usize
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
    /// The records files written (`Output::Records` / `Both`).
    pub records: Vec<Written>,
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

/// A model file's sha256 against the job's (`sha256` on a job, `checkpoint` or `models[gen]` line: always the
/// FILE's sha256 - the lease's - for self-play and eval alike; the graph's `checkpoint_sha256` metadata, the
/// torch checkpoint's, is never compared with it). The error names the file and the job's hash, never the
/// file's own (a job must not learn the hash of a file it named; zero-farm security review H3).
pub(crate) fn check_file_sha256(model: &std::path::Path, want: &str, gen: Option<u32>) -> Result<(), String> {
    let b = std::fs::read(model).map_err(|e| format!("{}: {e}", model.display()))?;
    let got = crate::proto::sha256_hex(&b);
    if got.eq_ignore_ascii_case(want) {
        return Ok(());
    }
    let what = gen.map_or(String::new(), |g| format!("gen {g}: "));
    Err(format!("{what}{}: the file's sha256 is not the job's ({want})", model.display()))
}

/// The job's evaluator; a network with a reference set first passes the self-test (`selftest`: the
/// event, and an error - the job's end - if it fails: a worker never plays with an evaluator that
/// doesn't answer as torch does). `gen`: the network's generation in an eval job (the event names it).
/// A model with a `sha256` is checked against it first (`check_file_sha256`).
pub(crate) fn build_eval(kind: &EvalKind, want: &Expect, sink: &dyn Sink, progress: &Mutex<Progress>, gen: Option<u32>) -> Result<Box<dyn Evaluator>, String> {
    Ok(match kind {
        EvalKind::Uniform => Box::new(Uniform),
        EvalKind::Onnx { model, ort, sha256, reference } => crate::session::build_evaluator(model, ort, sha256.as_deref(), reference.as_deref(), want, sink, progress, gen)?,
    })
}

/// The job's evaluators: `kind`'s network once per evaluator thread (`OrtCfg::workers`; the uniform evaluator once),
/// the first reporting to `sink` (its self-test event), the others quietly - each self-tested all the same.
pub(crate) fn build_evals(kind: &EvalKind, want: &Expect, sink: &dyn Sink, progress: &Mutex<Progress>, gen: Option<u32>) -> Result<Vec<Box<dyn Evaluator>>, String> {
    let n = match kind {
        EvalKind::Onnx { ort, .. } => ort.workers.max(1),
        EvalKind::Uniform => 1,
    };
    let mut v = vec![build_eval(kind, want, sink, progress, gen)?];
    for _ in 1..n {
        v.push(build_eval(kind, want, &NoSink, progress, gen)?);
    }
    Ok(v)
}

/// The evaluator threads (one per evaluator): each answers its batches in the order asked; the answers of all of them
/// on one channel, by batch id (the job submits them in launch order).
#[allow(clippy::type_complexity)]
pub(crate) fn spawn_eval(evs: Vec<Box<dyn Evaluator>>) -> (EvalLink, Receiver<(u64, Result<Answer, String>, f64)>, Vec<std::thread::JoinHandle<()>>) {
    let (atx, arx) = channel();
    let mut txs = Vec::new();
    let mut hs = Vec::new();
    for (i, ev) in evs.into_iter().enumerate() {
        let (tx, rx) = channel::<Req>();
        txs.push(tx);
        hs.push(spawn_eval_thread(ev, rx, atx.clone(), i));
    }
    (EvalLink { txs, next: Arc::new(std::sync::atomic::AtomicUsize::new(0)) }, arx, hs)
}

#[allow(clippy::type_complexity)]
fn spawn_eval_thread(mut ev: Box<dyn Evaluator>, rx: Receiver<Req>, atx: Sender<(u64, Result<Answer, String>, f64)>, i: usize) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name(if i == 0 { "zero-worker-eval".into() } else { format!("zero-worker-eval-{i}") })
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
        .expect("spawn the evaluator thread")
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
    /// Variable-length rows (zero-core docs/search.md 2.15; tools/onnx_record.py's arrays): the tail tokens and the
    /// overrides, kept when the batches carry them.
    t_off: Vec<i64>,
    t_seg: Vec<u16>,
    t_slot: Vec<u32>,
    t_val: Vec<f32>,
    ov_off: Vec<i64>,
    ov_opt: Vec<i32>,
    ov_tok: Vec<i64>,
    tails: bool,
    pointers: bool,
}

impl Recorder {
    pub(crate) fn new(path: PathBuf, max: usize) -> Recorder {
        Recorder { path, max, batches: 0, xs_off: vec![0], xs_idx: vec![], xs_val: vec![], n_players: vec![], player: vec![], gid: vec![], extra: vec![], opt_off: vec![0], opt_action: vec![], batch_off: vec![0], ids: vec![], t_off: vec![0], t_seg: vec![], t_slot: vec![], t_val: vec![], ov_off: vec![0], ov_opt: vec![], ov_tok: vec![], tails: false, pointers: false }
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
        if !b.t_off.is_empty() {
            self.tails = true;
            let tb = *self.t_off.last().unwrap();
            for r in 0..b.b {
                self.t_off.push(tb + b.t_off[r + 1]);
            }
            self.t_seg.extend_from_slice(&b.t_seg);
            self.t_slot.extend_from_slice(&b.t_slot);
            self.t_val.extend_from_slice(&b.t_val);
        }
        if !b.ov_off.is_empty() {
            self.pointers = true;
            let vb = *self.ov_off.last().unwrap();
            for r in 0..b.b {
                self.ov_off.push(vb + b.ov_off[r + 1]);
            }
            self.ov_opt.extend_from_slice(&b.ov_opt);
            self.ov_tok.extend_from_slice(&b.ov_tok);
        }
        self.batch_off.push(self.batch_off.last().unwrap() + b.b as i64);
        self.ids.push(b.id as i64);
        self.batches += 1;
    }
    /// The file (`meta`: the job's; with variable-length rows `info`'s `t_segs` / `option_pointers` added).
    pub(crate) fn write(&self, meta: &Value, info: &zero_search::dynapi::GameInfo) -> Result<(), String> {
        let f32n = |v: &[f32]| Npy { descr: "<f4".into(), shape: vec![v.len()], data: v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let u8n = |v: &[u8]| Npy { descr: "|u1".into(), shape: vec![v.len()], data: v.to_vec() };
        let mut meta = meta.clone();
        if self.tails || self.pointers {
            if let Some(fmt) = crate::shard::RowsFmt::of(info) {
                for (k, v) in fmt.meta() {
                    meta[k] = v;
                }
            }
        }
        let mut members = vec![
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
        if self.tails {
            members.push(("t_off", Npy::i64(&self.t_off, vec![self.t_off.len()])));
            members.push(("t_seg", Npy::u16(&self.t_seg, vec![self.t_seg.len()])));
            members.push(("t_slot", Npy::u32(&self.t_slot, vec![self.t_slot.len()])));
            members.push(("t_val", f32n(&self.t_val)));
        }
        if self.pointers {
            members.push(("ov_off", Npy::i64(&self.ov_off, vec![self.ov_off.len()])));
            members.push(("ov_opt", Npy::i32(&self.ov_opt, vec![self.ov_opt.len()])));
            members.push(("ov_tok", Npy::i64(&self.ov_tok, vec![self.ov_tok.len()])));
        }
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
    /// Answers of later batches that came before the oldest's (several evaluator threads).
    early: std::collections::HashMap<u64, (Result<Answer, String>, f64)>,
    /// Batches in flight at most: INFLIGHT, or one per evaluator thread (`infer._drive_pipelined`'s rule).
    inflight: usize,
    writer: ShardWriter,
    /// The records files (`Job::output` with records).
    rec: Option<RecordWriter>,
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
    /// The job's game ids: `job.games`, raised by a rolling job's `extend`.
    games: u64,
    /// A rolling job: the games value `low` was sent for.
    low_sent: Option<u64>,
    /// A rolling job: when the first game the writers hold (in no file yet) finished.
    held_since: Option<Instant>,
    /// A rolling job's checkpoint whose network is being built, and a newer one asked meanwhile.
    building: Option<Building>,
    next_ck: Option<Checkpoint>,
}

/// A rolling job's next network, built on a thread of its own while the games go on (`Run::start_build`).
struct Building {
    ck: Checkpoint,
    thread: std::thread::JoinHandle<Result<(), String>>,
    /// The build's events and log lines, sent on by the job when it collects the build.
    out: Arc<Buffered>,
}

/// A sink that keeps events and log lines (a background build's; the job's own sink sends them on, in order).
#[derive(Default)]
struct Buffered(Mutex<Vec<Result<Value, String>>>);

impl Sink for Buffered {
    fn event(&self, v: Value) {
        self.0.lock().unwrap().push(Ok(v));
    }
    fn log(&self, msg: &str) {
        self.0.lock().unwrap().push(Err(msg.to_string()));
    }
}

/// A rolling job's next evaluators: the checkpoint's network on the job's runtime and providers (its file's sha256
/// checked, its self-test passed: `selftest` with `gen`), one per evaluator thread, or the uniform evaluator in a
/// uniform job (tests, the plumbing gate).
fn checkpoint_eval(kind: &EvalKind, c: &Checkpoint, want: &Expect, sink: &dyn Sink, progress: &Mutex<Progress>) -> Result<Vec<Box<dyn Evaluator>>, String> {
    match kind {
        EvalKind::Uniform => Ok(vec![Box::new(Uniform)]),
        EvalKind::Onnx { ort, .. } => build_evals(&EvalKind::Onnx { model: c.model.clone(), ort: ort.clone(), sha256: c.sha256.clone(), reference: c.reference.clone() }, want, sink, progress, Some(c.gen)),
    }
}

impl Run<'_> {
    /// Games the writers hold that are in no file yet.
    fn held(&self) -> usize {
        match &self.rec {
            Some(rw) => rw.pending_games(),
            None => self.writer.pending_games(),
        }
    }

    /// A rolling job: write what the writers hold (`flush_s` passed since the first of it finished).
    fn flush_held(&mut self) -> Result<(), String> {
        if let Some(w) = self.rec.as_mut().map(|rw| rw.flush()).transpose()?.flatten() {
            self.emit_file("records", w);
        }
        if self.job.output.shards() {
            if let Some(w) = self.writer.flush()? {
                self.emit_file("shard", w);
            }
        }
        self.held_since = None;
        Ok(())
    }

    /// A rolling job's checkpoint: its network built and self-tested on a thread of its own (into `ctl.staged`).
    fn start_build(&mut self, c: Checkpoint) {
        let out = Arc::new(Buffered::default());
        let (kind, want, progress, staged, flag, o2, c2) = (self.job.eval.clone(), self.want.clone(), self.progress.clone(), self.ctl.staged.clone(), self.ctl.yielded_flag.clone(), out.clone(), c.clone());
        let thread = std::thread::Builder::new()
            .name("zero-worker-checkpoint".into())
            .spawn(move || {
                let mut evs = checkpoint_eval(&kind, &c2, &want, &*o2, &progress)?;
                let mut st = staged.lock().unwrap();
                // (built during a yield: released now, rebuilt on its first batch - do_yield)
                if flag.load(Ordering::SeqCst) {
                    for ev in evs.iter_mut() {
                        ev.release();
                    }
                }
                *st = Some(evs);
                Ok(())
            })
            .expect("spawn the checkpoint thread");
        self.sink.log(&format!("checkpoint {} ({}): building its evaluator while the games go on", c.gen, c.model.display()));
        self.building = Some(Building { ck: c, thread, out });
    }

    /// A rolling job's checkpoints: one asked starts its build (or waits for the one being built: the newest
    /// wins); a finished build ends the driver's phase (`switching`: the evaluator is swapped in once the phase
    /// ended, `run_job`).
    fn rolling_checkpoint(&mut self, ck: Option<Checkpoint>) -> Result<(), String> {
        if let Some(c) = ck {
            if self.building.is_some() || self.switching.is_some() {
                self.next_ck = Some(c);
            } else if !self.ended {
                self.start_build(c);
            }
        }
        if self.building.as_ref().is_some_and(|b| b.thread.is_finished()) {
            let b = self.building.take().unwrap();
            let res = b.thread.join().map_err(|_| "the checkpoint thread panicked".to_string())?;
            for x in std::mem::take(&mut *b.out.0.lock().unwrap()) {
                match x {
                    Ok(v) => self.sink.event(v),
                    Err(m) => self.sink.log(&m),
                }
            }
            // (a failed self-test is fatal, as at the start)
            res?;
            if let Some(n) = self.next_ck.take() {
                // (built, but a newer one came meanwhile: that one instead)
                self.ctl.staged.lock().unwrap().take();
                self.sink.log(&format!("checkpoint {}: superseded by {} before the switch", b.ck.gen, n.gen));
                if !self.ended {
                    self.start_build(n);
                }
            } else if !self.ended {
                self.drv.end_phase();
                self.switching = Some(b.ck);
            }
        }
        Ok(())
    }

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

    /// A result file is ready: `ev` 'shard' or 'records' (PROTOCOL.md 5). Its rows count once (the records' when
    /// both are written).
    fn emit_file(&mut self, ev: &str, w: Written) {
        {
            let mut p = self.progress.lock().unwrap();
            if ev == "records" || !self.job.output.records() {
                p.rows += w.rows as u64;
            }
            p.shards += 1;
        }
        let rel = w.path.strip_prefix(&self.job.out_dir).map(|p| p.to_path_buf()).unwrap_or_else(|_| w.path.clone());
        let gids: Vec<[u32; 2]> = ranges(&w.gids).into_iter().map(|r| [r.0, r.1]).collect();
        self.sink.event(json!({"ev": ev, "seq": w.seq, "path": rel.to_string_lossy().replace('\\', "/"), "games": w.games, "rows": w.rows, "bytes": w.bytes, "gids": gids, "gen": w.gen}));
    }

    /// The remote worker's `pump`: control, finished games into the shards, setups queued, status.
    fn pump(&mut self) -> Result<(), String> {
        let (quit, finish, ck, extend) = {
            let mut s = self.ctl.s.lock().unwrap();
            (s.quit, s.finish, s.checkpoint.take(), s.extend)
        };
        if quit && !self.ended {
            self.drv.end_phase();
            self.ended = true;
            self.stop_topup = true;
        }
        if finish {
            self.stop_topup = true;
        }
        let rolling = self.job.rolling.clone();
        if rolling.is_some() {
            // (a rolling job: `extend` raises the games, never lowers them; the checkpoint's network is built in
            // the background)
            if let Some(g) = extend.filter(|&g| g > self.games) {
                self.games = g;
            }
            self.rolling_checkpoint(ck)?;
        } else if let Some(c) = ck {
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
        // games dropped after a panic (zero-core docs/search.md 2.18): each failure's replay record into the log and into
        // the next records file's header (`failures`, records.rs); their rows carry no samples and no trace, so they go
        // into neither the records nor the shards
        for f in self.drv.take_failures() {
            self.sink.log(&format!("game failure (dropped after a panic): {f}"));
            if let Some(rw) = self.rec.as_mut() {
                rw.add_failure(serde_json::from_str(&f).unwrap_or_else(|_| serde_json::Value::String(f.clone())));
            }
        }
        let mut it = recs.into_iter();
        // (a branch game is its game's second row - zero-core docs/az.md 24: the job's games are its setups, the game
        // ids it started; a branch is counted in the stats, never as one of them)
        let n = rows.iter().filter(|r| r.branch.is_none()).count() as u64;
        for r in rows {
            let k = r.samples as usize;
            let records: Vec<_> = it.by_ref().take(k).collect();
            if records.len() != k {
                return Err(format!("game {} has {k} samples, {} left", r.gid, records.len()));
            }
            {
                // (Stats: the finished games' searched decisions)
                let mut p = self.progress.lock().unwrap();
                match &r.branch {
                    None => p.finished_gids.push(r.gid),
                    Some(b) => {
                        p.branches += 1;
                        p.merges += b.merged.is_some() as u64;
                    }
                }
                p.neartie += r.neartie.unwrap_or(0) as u64;
                p.decisions += r.decisions as u64;
            }
            if r.abort_reason == Some("panic") {
                continue;
            }
            if let Some(rw) = self.rec.as_mut() {
                if let Some(w) = rw.add(&r, &records)? {
                    self.emit_file("records", w);
                }
            }
            if self.job.output.shards() {
                if let Some(w) = self.writer.add(GameSamples { gid: r.gid, records })? {
                    self.emit_file("shard", w);
                }
            }
            if rolling.is_some() {
                // (the first finished game the writers hold since their last file: flush_s counts from it)
                if self.held() == 0 {
                    self.held_since = None;
                } else if self.held_since.is_none() {
                    self.held_since = Some(Instant::now());
                }
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
            let room = self.cap.saturating_sub(outstanding).min(self.games - self.enq);
            for _ in 0..room {
                let tag = format!("{}:{}", self.job.lease_id, self.enq);
                self.drv.enqueue(&tag, &self.spec)?;
                self.enq += 1;
            }
        }
        if let Some(rl) = &rolling {
            // fewer than conc ids left to start: ask for more, once per games value (PROTOCOL.md 10.5)
            if !self.stop_topup && self.games - self.enq < self.cap && self.low_sent != Some(self.games) {
                self.low_sent = Some(self.games);
                self.sink.event(json!({"ev": "low", "started": self.enq, "games": self.games}));
            }
            if self.held_since.is_some_and(|t| t.elapsed().as_secs_f64() >= rl.flush_s) {
                self.flush_held()?;
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
        self.link.eval(b)
    }

    fn finish_oldest(&mut self) -> Result<(), String> {
        let want = self.pending.pop_front().expect("a batch in flight");
        // (several evaluator threads: a later batch's answer may come first - kept until its turn)
        let (a, secs) = loop {
            if let Some(x) = self.early.remove(&want) {
                break x;
            }
            let (id, a, secs) = self.answers.recv().map_err(|_| GONE.to_string())?;
            if id == want {
                break (a, secs);
            }
            if !self.pending.contains(&id) {
                return Err(format!("internal: answer of batch {id}, expected {want}"));
            }
            self.early.insert(id, (a, secs));
        };
        let id = want;
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
                    {
                        let mut p = self.progress.lock().unwrap();
                        p.infer_rows += b.b as u64;
                        p.batch_rows_log2[rows_bucket(b.b)] += 1;
                    }
                    self.send(b)?;
                    if self.pending.len() >= self.inflight {
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
    if cfg.branch.is_some() && job.output.records() {
        // (a game record replays a game from its seed: records/1 has no branch prefix nor a merged branch's outcome -
        // zero-core docs/az.md 72.7: the farm pool's records path is the open part; shards jobs play branches)
        return Err("selfplay_branch: branched games are played into shards only - a game record (output 'records' / 'both') cannot replay a branch game yet (zero-core docs/az.md 72.7)".into());
    }
    let belief = cfg.seats.iter().any(|s| matches!(s, SeatCfg::Nn(n) if NnSeat::belief_on(n)));
    let want = Expect::of(&info, belief);
    let evs = build_evals(&job.eval, &want, sink, &progress, None)?;
    let workers = if evs.len() > 1 { format!(" x {} evaluator threads", evs.len()) } else { String::new() };
    sink.log(&format!("job {}: {} games from gid {}, seed {}, net {}; driver: conc {}, threads {}, groups {}, sparse; evaluator: {}{workers}", job.lease_id, job.games, job.gid_base, job.seed, job.net, cfg.concurrency, cfg.threads, cfg.groups, evs[0].describe()));
    {
        let mut p = progress.lock().unwrap();
        p.concurrency = cfg.concurrency;
        p.threads = cfg.threads;
        p.provider = evs[0].provider();
    }
    if cfg.concurrency == 0 {
        return Err("conc 0: no game would ever start (the setup cap is the games in flight)".into());
    }
    let cap = cfg.concurrency as u64;
    let spec = spec_of(&job.settings)?;
    let aux = cfg.aux;
    let mut drv = (game.new_driver)(cfg)?;
    if job.gid_base != 0 {
        drv.set_first_gid(job.gid_base)?;
    }
    if job.output.records() {
        // (every game's seed, real moves and setup: what its record replays from; zero-core docs/search.md 2.17)
        drv.set_trace(true)?;
    }
    drv.resume(BIG, job.net)?;
    let mut meta = Map::new();
    meta.insert("game".into(), json!(info.name));
    meta.insert("features".into(), json!(info.num_features));
    meta.insert("actions".into(), json!(info.num_actions));
    meta.extend(job.meta.clone());
    let sparse = crate::settings::truthy(job.settings.get("shard_sparse").unwrap_or(&crate::pyjson::PyJson::Null));
    let shard_rows = crate::settings::py_int("shard_rows", job.settings.req("shard_rows")?)?;
    let rec = if job.output.records() {
        let mut h = Map::new();
        h.insert("game".into(), json!(info.name));
        h.insert("spec".into(), json!(spec));
        h.insert("aux".into(), json!(aux));
        h.insert("features".into(), json!(info.num_features));
        h.insert("actions".into(), json!(info.num_actions));
        h.insert("aux_width".into(), json!(info.aux.iter().map(|a| a.1).sum::<usize>()));
        h.insert("seed".into(), json!(job.seed));
        h.insert("lease".into(), json!(job.lease_id));
        h.insert("build_id".into(), job.meta.get("build_id").cloned().unwrap_or(Value::Null));
        Some(RecordWriter::new(&job.out_dir, &job.prefix, shard_rows.max(1) as usize, h, 6)?)
    } else {
        None
    };
    let writer = ShardWriter::new(&job.out_dir, &job.prefix, shard_rows.max(1) as usize, meta, sparse, info.num_features, 1, crate::shard::RowsFmt::of(&info))?;
    let (link, answers, eval_threads) = spawn_eval(evs);
    link_out(link.clone());
    let inflight = INFLIGHT.max(link.workers());
    let mut r = Run { job, sink, ctl, drv, link, answers, pending: VecDeque::new(), early: Default::default(), inflight, writer, rec, spec, cap, enq: 0, finished: 0, stop_topup: false, ended: false, switching: None, progress: progress.clone(), recorder: job.record_batches.as_ref().map(|(p, n)| Recorder::new(p.clone(), *n)), want, t0, t_status: Instant::now(), fin_unsent: 0, games: job.games, low_sent: None, held_since: None, building: None, next_ck: None };
    let res = (|| -> Result<(), String> {
        r.pump()?;
        let mut idle = 0;
        loop {
            r.drive()?;
            if r.job.rolling.is_some() {
                if r.ended {
                    // (quit while a switch was armed: no swap, the driver is never resumed after its end)
                    r.switching = None;
                }
                if let Some(c) = r.switching.take() {
                    // a rolling job's phase change: the evaluator its checkpoint built (and self-tested) in the
                    // background swapped in, the driver resumed with its id
                    let new = r.ctl.staged.lock().unwrap().take().ok_or("internal: a checkpoint switch without its evaluator")?;
                    r.link.swap(new)?;
                    r.drv.resume(BIG, c.gen)?;
                    r.sink.log(&format!("checkpoint {} ({}): the driver resumed with it", c.gen, c.model.display()));
                    if let Some(n) = r.next_ck.take() {
                        r.start_build(n);
                    }
                    idle = 0;
                    r.pump()?;
                    continue;
                }
                if r.ended || (r.stop_topup && r.enq == r.finished) {
                    break;
                }
                // every released id played (or a queue_only driver with an empty queue): wait for extend, finish,
                // quit or a checkpoint's build (a pump every 20 ms: flush_s and the build are checked there too)
                idle += 1;
                if idle > 1 {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
                r.pump()?;
                continue;
            }
            if let Some(c) = r.switching.take() {
                // the remote worker's phase change: the new network, the driver resumed with its id
                let new = build_evals(&EvalKind::Onnx { model: c.model.clone(), ort: match &r.job.eval {
                    EvalKind::Onnx { ort, .. } => ort.clone(),
                    EvalKind::Uniform => return Err("checkpoint: a uniform-evaluator job has no network to switch".into()),
                }, sha256: c.sha256.clone(), reference: c.reference.clone() }, &r.want, r.sink, &r.progress, None)?;
                r.link.swap(new)?;
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
        if let Some(w) = r.rec.as_mut().map(|rw| rw.flush()).transpose()?.flatten() {
            r.emit_file("records", w);
        }
        if let Some(n) = r.rec.as_ref().map(|rw| rw.pending_failures()).filter(|&n| n > 0) {
            // (a file is written only with a game in it: failures after the job's last finished game stay in the log)
            r.sink.log(&format!("{n} game failure(s) after the last finished game: in this log only, in no records file"));
        }
        if let Some(w) = r.writer.flush()? {
            r.emit_file("shard", w);
        }
        Ok(())
    })();
    let rec_res = match (&r.recorder, &res) {
        (Some(rc), Ok(())) => rc.write(&json!({"game": info.name, "features": info.num_features, "actions": info.num_actions, "seed": job.seed, "worker": "zero_worker"}), &info),
        _ => Ok(()),
    };
    r.snapshot();
    let games = r.games;
    let Run { link, writer, rec, .. } = r;
    link.stop();
    drop(link);
    for h in eval_threads {
        let _ = h.join();
    }
    res?;
    rec_res?;
    let p = progress.lock().unwrap();
    // (a rolling job never ends 'done': it waits for more ids until finish or quit)
    let reason = if job.rolling.is_none() && p.games >= job.games {
        "done"
    } else if ctl.s.lock().unwrap().quit {
        "quit"
    } else {
        "finish"
    };
    // (the job's ids: a rolling job's released ones, its games as extend left them)
    let lo = job.gid_base;
    let hi = job.gid_base + games as u32;
    let mut stats = p.stats();
    stats["peak_rss_mb"] = json!(peak_rss_mb());
    // (games after the last shard that had no samples - an aborted game -: in no shard's `games`; the remote
    // worker's 'unshipped')
    stats["unshipped"] = json!(rec.as_ref().map_or(writer.pending_games(), |rw| rw.pending_games()));
    stats["seconds"] = json!((t0.elapsed().as_secs_f64() * 100.0).round() / 100.0);
    Ok(Outcome { reason, stats, unfinished: missing_ranges(lo, hi, &p.finished_gids), shards: writer.written, records: rec.map(|rw| rw.written).unwrap_or_default() })
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
    // (a rolling job's next network, built but not swapped in yet; set_yielded came first: one staged after this
    // sees the flag and is released by its builder)
    if let Some(evs) = ctl.staged.lock().unwrap().as_mut() {
        for e in evs.iter_mut() {
            e.release();
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    struct Quiet;
    impl Sink for Quiet {
        fn event(&self, _: Value) {}
        fn log(&self, _: &str) {}
    }

    #[test]
    fn a_models_sha256_is_its_files_checked_before_it_is_loaded() {
        let d = std::env::temp_dir().join(format!("zero-worker-sha-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("model.onnx");
        std::fs::write(&p, b"not a graph").unwrap();
        let good = crate::proto::sha256_hex(b"not a graph");
        assert!(check_file_sha256(&p, &good, None).is_ok());
        assert!(check_file_sha256(&p, &good.to_uppercase(), None).is_ok());
        let bad = "0".repeat(64);
        let e = check_file_sha256(&p, &bad, Some(2)).unwrap_err();
        assert!(e.starts_with("gen 2: ") && e.contains("not the job's") && !e.contains(&good), "{e}");
        assert!(check_file_sha256(&d.join("missing.onnx"), &good, None).is_err());
        // a wrong hash ends the job before ONNX Runtime is touched (there is no runtime here at all; the right hash
        // reaching the loader is the worker's integration tests': MTG Zero tests/test_worker_farm.py)
        let want = Expect { game: "toy".into(), features: 1, actions: 1, aux_width: 0, belief: false, tails: vec![], option_pointers: false };
        let ort = OrtCfg::cpu(d.join("no-onnxruntime.dll"), 1, false);
        let progress = Mutex::new(Progress::default());
        let kind = EvalKind::Onnx { model: p.clone(), ort, sha256: Some(bad.clone()), reference: None };
        match build_eval(&kind, &want, &Quiet, &progress, Some(1)) {
            Err(e) => assert!(e.starts_with("gen 1: ") && e.contains("not the job's"), "{e}"),
            Ok(_) => panic!("a model whose file is not the job's sha256 was loaded"),
        }
        let _ = std::fs::remove_dir_all(&d);
    }
}
