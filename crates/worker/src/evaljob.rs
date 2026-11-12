//! An eval job (zero-farm PROTOCOL.md 5.2, docs/az.md 42.2): games `gid_base .. gid_base + games` of an
//! evaluation match between two seats - networks (one ONNX Runtime session per generation of the job's
//! `models`, each self-tested first) or built-in bots - played as Python's `Run.play_match` plays them
//! (`Run.nn_seat(key, eval_sims)` seats, `Run.driver(table, seed, conc, paired)`: `settings::
//! eval_driver_cfg`), as a rolling queue_only driver (`settings::eval_lease_cfg`) with every game's setup
//! queued before it starts and its first game id the lease's, so a lease plays exactly those games of the
//! whole match: game g has the seats in `seats` order for even g and swapped for odd g, and with `paired`
//! games 2k and 2k + 1 share the deal (every random stream from k).
//!
//! The loop is the self-play job's (`job`, `infer._drive_pipelined`): up to two network batches in flight,
//! answered in launch order on the evaluator thread (here one evaluator per generation, chosen by the
//! batch's key), a `pump` after every answer: control, finished games into result rows, status.
//!
//! Results: one JSON line per game, `{"gid", "pair", "seats", "result", "margin", "rounds", "over"}` -
//! `pair` = gid / 2 (null unpaired), `seats` this game's seats in its order, `result` per seat in that
//! order (1 win, 0.5 draw, 0 loss), `margin` seat 0's score minus the best other seat's (`Run.play_match`'s
//! margin from seat 0), as `json.dumps(row, separators=(',', ':'))` writes it - zero.farm_eval's `row_json`
//! bit for bit. A file `<prefix>_SSS.jsonl` (rows by gid) is written every `flush_games` complete games
//! (a pair is complete when both its games are in) or `flush_s` seconds, and at the end; the event `evals`
//! announces it. `finish`: no new game starts (the queue is dropped), the games in flight finish; `quit`:
//! the driver's phase ends now, finished games are flushed (a pair's lone game too), the rest is
//! `unfinished`.
use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde_json::json;
use zero_search::driver::{Batch, GameResult, KeyKind, NnSeat, SeatCfg};
use zero_search::dynapi::DynDriver;

use crate::eval::{Answer, Evaluator, Expect, OrtCfg, Uniform};
use crate::job::{build_eval, missing_ranges, peak_rss_mb, ranges, spawn_eval, Control, EvalKind, EvalLink, Outcome, Progress, Recorder, Req, Sink, BIG, INFLIGHT};
use crate::pyjson::PyJson;
use crate::settings::{eval_driver_cfg, eval_lease_cfg, eval_spec_of, EvalSeat, Settings};
use crate::shard::Written;
use crate::GameEntry;

/// A network of the job (`models[gen]`): its graph, the file's sha256 (checked before it is loaded) and its
/// reference set (the self-test).
#[derive(Clone, Debug)]
pub struct ModelSpec {
    pub model: PathBuf,
    pub sha256: Option<String>,
    pub reference: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct EvalJob {
    pub lease_id: String,
    pub settings: Settings,
    pub seats: Vec<EvalSeat>,
    pub models: BTreeMap<u32, ModelSpec>,
    /// ONNX Runtime's settings; None: the uniform evaluator answers every network (tests).
    pub ort: Option<OrtCfg>,
    pub games: u64,
    pub gid_base: u32,
    pub seed: u64,
    pub paired: bool,
    /// Search threads (the settings' `threads` when None), and a job's own conc / groups.
    pub search_threads: Option<usize>,
    pub conc: Option<usize>,
    pub groups: Option<usize>,
    pub out_dir: PathBuf,
    /// Result files: `<prefix>_SSS.jsonl`.
    pub prefix: String,
    /// A result file every this many complete games, or `flush_s` seconds after the oldest unwritten one.
    pub flush_games: usize,
    pub flush_s: f64,
    pub record_batches: Option<(PathBuf, usize)>,
    pub status_s: f64,
}

impl EvalJob {
    /// The job's checks before anything is loaded: whole pairs, seats naming the job's networks.
    pub fn check(&self) -> Result<(), String> {
        if self.games == 0 {
            return Err("eval job: games 0".into());
        }
        if self.paired && (self.gid_base % 2 != 0 || self.games % 2 != 0) {
            return Err(format!("eval job: paired games come in pairs (2k, 2k + 1): gid_base {} and games {} must be even", self.gid_base, self.games));
        }
        if self.seats.len() != 2 {
            return Err(format!("eval job: 2 seats (got {})", self.seats.len()));
        }
        for s in &self.seats {
            if let EvalSeat::Net(g) = s {
                if !self.models.contains_key(g) {
                    return Err(format!("eval job: seat {{\"net\": {g}}} but models has {:?}", self.models.keys().collect::<Vec<_>>()));
                }
            }
        }
        if (self.gid_base as u64) + self.games > u32::MAX as u64 {
            return Err("eval job: game ids beyond 2^32".into());
        }
        Ok(())
    }

    /// The seats of game `gid` in its order.
    pub fn seats_of(&self, gid: u32) -> [&EvalSeat; 2] {
        if gid % 2 == 0 {
            [&self.seats[0], &self.seats[1]]
        } else {
            [&self.seats[1], &self.seats[0]]
        }
    }
}

/// One finished game's result row (module docs; zero.farm_eval `row_json`).
pub fn row_json(gid: u32, paired: bool, seats: [&EvalSeat; 2], r: &GameResult) -> String {
    let sc: Vec<f64> = r.scores.iter().map(|&x| x as f64).collect();
    // (Python: sc[0] - max(sc[j] for j != 0), floats)
    let best_other = sc.iter().skip(1).copied().fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a });
    let margin = if sc.len() > 1 { sc[0] - best_other } else { 0.0 };
    PyJson::Dict(vec![
        ("gid".into(), PyJson::Int(gid as i128)),
        ("pair".into(), if paired { PyJson::Int((gid / 2) as i128) } else { PyJson::Null }),
        ("seats".into(), PyJson::List(seats.iter().map(|s| s.pyjson()).collect())),
        ("result".into(), PyJson::List(r.result.iter().map(|&x| PyJson::Float(x as f64)).collect())),
        ("margin".into(), PyJson::Float(margin)),
        ("rounds".into(), PyJson::Int(r.rounds as i128)),
        ("over".into(), PyJson::Bool(r.over)),
    ])
    .dumps_compact()
}

/// The evaluators of the driver's keys: one per generation, a key ('m': seat 0's network, 'o': seat 1's)
/// answered by its generation's.
struct ByKey {
    evs: Vec<(u32, Box<dyn Evaluator>)>,
    key_ev: Vec<Option<usize>>,
}

impl Evaluator for ByKey {
    fn evaluate(&mut self, b: &Batch) -> Result<Answer, String> {
        let i = self.key_ev.get(b.key as usize).copied().flatten().ok_or_else(|| format!("a batch of key {} that no network answers", b.key))?;
        self.evs[i].1.evaluate(b)
    }
    fn release(&mut self) {
        for e in &mut self.evs {
            e.1.release();
        }
    }
    fn restore(&mut self) -> Result<(), String> {
        for e in &mut self.evs {
            e.1.restore()?;
        }
        Ok(())
    }
    fn describe(&self) -> String {
        self.evs.iter().map(|(g, e)| format!("gen {g}: {}", e.describe())).collect::<Vec<_>>().join("; ")
    }
}

struct EvalRun<'a> {
    job: &'a EvalJob,
    sink: &'a dyn Sink,
    ctl: &'a Control,
    drv: Box<dyn DynDriver>,
    link: EvalLink,
    answers: Receiver<(u64, Result<Answer, String>, f64)>,
    pending: VecDeque<u64>,
    progress: Arc<Mutex<Progress>>,
    recorder: Option<Recorder>,
    t0: Instant,
    t_status: Instant,
    /// Games started or queued (all of them, less the queue dropped by `finish`).
    planned: u64,
    finished: u64,
    ended: bool,
    finishing: bool,
    fin_unsent: u64,
    /// Finished games' rows by gid, not written yet; `since`: when the oldest complete one came.
    rows: BTreeMap<u32, String>,
    since: Option<Instant>,
    seq: u64,
    written: Vec<Written>,
}

impl EvalRun<'_> {
    fn snapshot(&mut self) {
        let st = self.drv.stats();
        let mut p = self.progress.lock().unwrap();
        p.games = self.finished;
        p.started = self.planned - self.drv.queued() as u64;
        p.leaves = st.leaves;
        p.sims = st.sims;
        p.wait_ms = st.wait_s * 1e3;
        p.search_ms = st.busy_s * 1e3;
        p.wall_s = self.t0.elapsed().as_secs_f64();
    }

    /// The rows ready to write: every finished game whose pair is complete (unpaired: every one); `all`:
    /// every finished game (the job's end).
    fn ready(&self, all: bool) -> Vec<u32> {
        self.rows
            .keys()
            .copied()
            .filter(|&g| all || !self.job.paired || self.rows.contains_key(&(g ^ 1)))
            .collect()
    }

    fn write_file(&mut self, gids: Vec<u32>) -> Result<(), String> {
        if gids.is_empty() {
            return Ok(());
        }
        let mut text = String::new();
        for g in &gids {
            text.push_str(&self.rows.remove(g).expect("a row"));
            text.push('\n');
        }
        std::fs::create_dir_all(&self.job.out_dir).map_err(|e| format!("{}: {e}", self.job.out_dir.display()))?;
        let name = format!("{}_{:03}.jsonl", self.job.prefix, self.seq);
        let path = self.job.out_dir.join(&name);
        let tmp = self.job.out_dir.join(format!("{name}.tmp"));
        std::fs::write(&tmp, text.as_bytes()).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
        let bytes = text.len() as u64;
        {
            let mut p = self.progress.lock().unwrap();
            p.rows += gids.len() as u64;
            p.shards += 1;
        }
        let rs: Vec<[u32; 2]> = ranges(&gids).into_iter().map(|r| [r.0, r.1]).collect();
        self.sink.event(json!({"ev": "evals", "seq": self.seq, "path": name, "games": gids.len(), "rows": gids.len(), "bytes": bytes, "gids": rs}));
        self.written.push(Written { seq: self.seq as usize, path, games: gids.len(), rows: gids.len(), bytes, gids: gids.clone(), gen: None });
        self.seq += 1;
        self.since = None;
        Ok(())
    }

    /// Control, finished games into rows (and files), status.
    fn pump(&mut self) -> Result<(), String> {
        let (quit, finish, ck) = {
            let mut s = self.ctl.s.lock().unwrap();
            (s.quit, s.finish, s.checkpoint.take())
        };
        if ck.is_some() {
            self.sink.event(json!({"ev": "error", "message": "checkpoint: an eval job's networks are fixed (its models)", "fatal": false}));
        }
        if quit && !self.ended {
            self.drv.end_phase();
            self.ended = true;
        }
        if finish && !self.finishing {
            self.finishing = true;
            let n = self.drv.clear_queue() as u64;
            self.planned -= n;
            self.sink.log(&format!("finish: {n} queued games dropped, the {} in flight play on", self.planned - self.finished));
        }
        let res = self.drv.take_results();
        if !self.drv.take_samples().is_empty() {
            return Err("internal: an eval driver recorded samples".into());
        }
        let n = res.len() as u64;
        for r in res {
            {
                let mut p = self.progress.lock().unwrap();
                p.finished_gids.push(r.gid);
                p.decisions += r.decisions as u64;
            }
            let row = row_json(r.gid, self.job.paired, self.job.seats_of(r.gid), &r);
            self.rows.insert(r.gid, row);
        }
        self.finished += n;
        self.fin_unsent += n;
        if self.fin_unsent > 0 {
            self.sink.event(json!({"ev": "fin", "n": self.fin_unsent}));
            self.fin_unsent = 0;
        }
        let ready = self.ready(false);
        if !ready.is_empty() && self.since.is_none() {
            self.since = Some(Instant::now());
        }
        if ready.len() >= self.job.flush_games.max(1) || (!ready.is_empty() && self.since.is_some_and(|t| t.elapsed().as_secs_f64() >= self.job.flush_s)) {
            self.write_file(ready)?;
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

/// The evaluators of an eval job: every model of `models` loaded and self-tested (the event `selftest`
/// with its `gen`; a failure ends the job before any game), keyed to the driver's keys.
fn build_evaluators(game: &GameEntry, job: &EvalJob, keys: &[(String, KeyKind)], belief: bool, sink: &dyn Sink, progress: &Mutex<Progress>) -> Result<ByKey, String> {
    let info = (game.info)();
    let want = Expect { game: info.name.to_string(), features: info.num_features, actions: info.num_actions, aux_width: info.aux.iter().map(|a| a.1).sum(), belief };
    let mut evs: Vec<(u32, Box<dyn Evaluator>)> = Vec::new();
    let mut worst: Option<f64> = None;
    for (gen, m) in &job.models {
        let ev: Box<dyn Evaluator> = match &job.ort {
            None => Box::new(Uniform),
            Some(ort) => {
                // (the file's sha256, checked by build_eval before the graph is loaded)
                let ev = build_eval(&EvalKind::Onnx { model: m.model.clone(), ort: ort.clone(), sha256: m.sha256.clone(), reference: m.reference.clone() }, &want, sink, progress, Some(*gen))?;
                // (Stats' selftest_max_abs: the largest over the job's networks)
                if let Some(x) = progress.lock().unwrap().selftest_max_abs {
                    worst = Some(worst.map_or(x, |w| if x.is_nan() || x > w { x } else { w }));
                }
                ev
            }
        };
        evs.push((*gen, ev));
    }
    progress.lock().unwrap().selftest_max_abs = worst;
    let mut key_ev = Vec::with_capacity(keys.len());
    for (k, kind) in keys {
        let seat = match k.as_str() {
            "m" => &job.seats[0],
            "o" => &job.seats[1],
            other => return Err(format!("internal: an eval driver key {other:?}")),
        };
        key_ev.push(match (seat, kind) {
            (EvalSeat::Net(g), KeyKind::Nn) => Some(evs.iter().position(|e| e.0 == *g).expect("checked: the seat's model")),
            _ => None,
        });
    }
    Ok(ByKey { evs, key_ev })
}

/// Run an eval job to its end (`link_out`: the evaluator link, for the agent's yield / resume).
pub fn run_eval_job(game: &GameEntry, job: &EvalJob, ctl: &Control, sink: &dyn Sink, progress: Arc<Mutex<Progress>>, link_out: &dyn Fn(EvalLink)) -> Result<Outcome, String> {
    let t0 = Instant::now();
    job.check()?;
    let info = (game.info)();
    let m = eval_driver_cfg(&job.settings, &info, &job.seats, job.games, job.seed, job.paired, job.conc, job.search_threads, job.groups)?;
    let cfg = eval_lease_cfg(m);
    let belief = cfg.seats.iter().any(|s| matches!(s, SeatCfg::Nn(n) if NnSeat::belief_on(n)));
    let spec = eval_spec_of(&job.settings)?;
    let ev = build_evaluators(game, job, &cfg.keys, belief, sink, &progress)?;
    sink.log(&format!(
        "eval job {}: games {}..{} ({}), seed {}, seats {:?}; driver: conc {}, threads {}, groups {}; evaluators: {}",
        job.lease_id,
        job.gid_base,
        job.gid_base as u64 + job.games,
        if job.paired { "paired" } else { "unpaired" },
        job.seed,
        job.seats,
        cfg.concurrency,
        cfg.threads,
        cfg.groups,
        ev.describe()
    ));
    {
        let mut p = progress.lock().unwrap();
        p.concurrency = cfg.concurrency;
        p.threads = cfg.threads;
        p.provider = if job.ort.is_some() { "cpu".into() } else { "uniform".into() };
    }
    let mut drv = (game.new_driver)(cfg)?;
    drv.set_first_gid(job.gid_base)?;
    for k in 0..job.games {
        drv.enqueue(&format!("{}:{}", job.lease_id, job.gid_base as u64 + k), &spec)?;
    }
    drv.resume(BIG, 0)?;
    let (link, answers, eval_thread) = spawn_eval(Box::new(ev));
    link_out(link.clone());
    let mut r = EvalRun {
        job,
        sink,
        ctl,
        drv,
        link,
        answers,
        pending: VecDeque::new(),
        progress: progress.clone(),
        recorder: job.record_batches.as_ref().map(|(p, n)| Recorder::new(p.clone(), *n)),
        t0,
        t_status: Instant::now(),
        planned: job.games,
        finished: 0,
        ended: false,
        finishing: false,
        fin_unsent: 0,
        rows: BTreeMap::new(),
        since: None,
        seq: 0,
        written: Vec::new(),
    };
    let res = (|| -> Result<(), String> {
        r.pump()?;
        r.drive()?;
        r.pump()?;
        // the rest: every complete pair, then (quit / finish) the lone games too
        let rest = r.ready(true);
        r.write_file(rest)
    })();
    let rec_res = match (&r.recorder, &res) {
        (Some(rc), Ok(())) => rc.write(&json!({"game": info.name, "features": info.num_features, "actions": info.num_actions, "seed": job.seed, "worker": "zero_worker eval"})),
        _ => Ok(()),
    };
    r.snapshot();
    let EvalRun { link, written, .. } = r;
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
    stats["seconds"] = json!((t0.elapsed().as_secs_f64() * 100.0).round() / 100.0);
    Ok(Outcome { reason, stats, unfinished: missing_ranges(lo, hi, &p.finished_gids), shards: written, records: Vec::new() })
}
