//! An eval job (zero-farm PROTOCOL.md 5.2, docs/az.md 44.2): games `gid_base .. gid_base + games` of an
//! evaluation match between two seats - networks (one ONNX Runtime session per generation of the job's
//! `models`, each self-tested first) or built-in bots - played as Python's `Run.play_match` plays them
//! (`Run.nn_seat(key, eval_sims)` seats, `Run.driver(table, seed, conc, paired)`: `settings::
//! eval_driver_cfg`), as a rolling queue_only driver (`settings::eval_lease_cfg`) with every game's setup
//! queued before it starts and its first game id the lease's, so a lease plays exactly those games of the
//! whole match: game g has the seats in `seats` order for even g and swapped for odd g, and with `paired`
//! games 2k and 2k + 1 share the deal (every random stream from k).
//!
//! The loop is the self-play job's (`job`, `infer._drive_pipelined`): network batches in flight (two, or one per
//! evaluator lane), their answers submitted in launch order, a `pump` after every answer: control, finished games into
//! result rows, status.
//!
//! **The layout** (docs/az.md 71, [`eval_layout`]): each batch goes to an **evaluator lane** of its key, an evaluator
//! thread with sessions of its own, so the seats' batches - and on the CPU several groups' - run side by side. On the
//! CPU a job line without `groups` gets groups from its threads, never the settings' `groups` (the run's own self-play
//! batching made CPU leases ~2 rows an ONNX Runtime call), each key lanes of its own, each lane's session a few
//! threads ([`EVAL_SESSION_THREADS`]); on CUDA a lane per generation; on any other GPU provider one lane for every
//! network (as before); `LaneMode::One` is the layout before. Only the batches change - which games' leaves a batch
//! holds and how many run at once -, never a search: one leaf per game per batch, no virtual loss; a game's moves are
//! the same when its network answers are (the farm's CPU graphs answer a row the same in any batch and with any thread
//! count - measured, docs/az.md 71).
//!
//! Results: one JSON line per game, `{"gid", "pair", "seats", "result", "margin", "rounds", "over"}` -
//! `pair` = gid / 2 (null unpaired), `seats` this game's seats in its order, `result` per seat in that
//! order (1 win, 0.5 draw, 0 loss), `margin` seat 0's score minus the best other seat's (`Run.play_match`'s
//! margin from seat 0), as `json.dumps(row, separators=(',', ':'))` writes it - zero.farm_eval's `row_json`
//! bit for bit.
//!
//! Traced rows (the eval setting `eval_trace`, docs/az.md 50; zero-farm analysis campaigns, PROTOCOL.md 5.3): with it
//! on the driver traces its games (`Driver::set_trace`, docs/search.md 2.17: it changes no game) and each row ends with
//! two more keys, `"digest"` - the first 16 hex of the SHA-256 of the record's compact JSON with its keys sorted
//! (serde_json's text here; Python's `json.dumps(rec, sort_keys=True, separators=(',', ':'), ensure_ascii=False)`,
//! zero.farm_eval `trace_digest`) - and `"rec"`, the game's record (`zero_search::record::game_json` without samples:
//! gid, players, seed, every real move, the result, margins, rounds, setup) - what the desktop replays
//! (`zero_search::record::replay`, the binding's `replay_rows`). Off (the default) the rows are what they were.
//!
//! A file `<prefix>_SSS.jsonl` (rows by gid) is written every `flush_games` complete games
//! (a pair is complete when both its games are in) or `flush_s` seconds, and at the end; the event `evals`
//! announces it. `finish`: no new game starts (the queue is dropped), the games in flight finish; `quit`:
//! the driver's phase ends now, finished games are flushed (a pair's lone game too), the rest is
//! `unfinished`.
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde_json::json;
use zero_search::driver::{Batch, GameResult, KeyKind, NnSeat, SeatCfg};
use zero_search::dynapi::DynDriver;
use zero_search::record::game_json;

use crate::eval::{Answer, Evaluator, Expect, OrtCfg, Uniform};
use crate::job::{build_eval, missing_ranges, peak_rss_mb, ranges, rows_bucket, spawn_eval, Control, EvalKind, EvalLink, NoSink, Outcome, Progress, Recorder, Sink, BIG, INFLIGHT};
use crate::session::ProviderKind;
use crate::pyjson::PyJson;
use crate::settings::{eval_driver_cfg, eval_lease_cfg, eval_spec_of, eval_trace_of, EvalSeat, Settings};
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
    /// The job named its inference threads (the job line's `infer_threads`, the command line's `--infer-threads`):
    /// every lane's sessions get exactly `ort`'s; false: the layout's (`eval_layout`).
    pub own_infer_threads: bool,
    /// The evaluator lanes (module docs; `LaneMode::Auto` unless a test or a measurement asks for the layout of before).
    pub lanes: LaneMode,
    pub games: u64,
    pub gid_base: u32,
    pub seed: u64,
    pub paired: bool,
    /// Search threads (the settings' `threads` when None), and a job's own conc / groups (groups None on the CPU:
    /// `eval_layout`'s, not the settings').
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
            if let Some(g) = s.net() {
                if !self.models.contains_key(&g) {
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
    let mut d = vec![
        ("gid".into(), PyJson::Int(gid as i128)),
        ("pair".into(), if paired { PyJson::Int((gid / 2) as i128) } else { PyJson::Null }),
        ("seats".into(), PyJson::List(seats.iter().map(|s| s.pyjson()).collect())),
        ("result".into(), PyJson::List(r.result.iter().map(|&x| PyJson::Float(x as f64)).collect())),
        ("margin".into(), PyJson::Float(margin)),
        ("rounds".into(), PyJson::Int(r.rounds as i128)),
        ("over".into(), PyJson::Bool(r.over)),
    ];
    if r.abort_reason == Some("panic") {
        // a game dropped after a panic inside it (zero-core docs/search.md 2.18): not a result (the row of before for
        // every other game)
        d.push(("abort_reason".into(), PyJson::Str("panic".into())));
    }
    PyJson::Dict(d).dumps_compact()
}

/// A traced game's two extra keys (module docs): `,"digest":"<16 hex>","rec":<record>` inserted before the row's
/// closing brace.
pub fn traced_row_json(row: &str, r: &GameResult) -> Result<String, String> {
    let rec = serde_json::to_string(&game_json(r, &[])?).map_err(|e| e.to_string())?;
    let digest = &crate::proto::sha256_hex(rec.as_bytes())[..16];
    let body = row.strip_suffix('}').ok_or("internal: a row that is not a JSON object")?;
    Ok(format!("{body},\"digest\":\"{digest}\",\"rec\":{rec}}}"))
}

/// The most intra-op threads of an eval job's CPU session when the job names none (docs/az.md 71): ONNX Runtime runs the
/// farm's graphs (~1,000 small nodes) poorly on many threads - on an M4 3 threads 2,530 rows/s, 6 threads 2,650, 12
/// slower than 4 (docs/az.md 62); the desktop's i9 fastest at 3-4 (63) -, so a CPU's threads serve the job as several
/// sessions side by side, not one wide one.
pub const EVAL_SESSION_THREADS: usize = 4;

/// The driver's groups of a CPU eval job whose job line names none (`eval_layout`): one - a step's batch per seat holds
/// every game in flight, the two seats' batches on two lanes at once. Measured on the farm (docs/az.md 71, 16-thread
/// PCs): 32-game leases 645-658 games/h in one group against 524-651 in two (four lanes of half the rows), 64-game
/// leases 709-838 against 656 - a second group splits a lease's last games into still smaller batches.
pub const EVAL_GROUPS: usize = 1;

/// How an eval job's network batches are spread over evaluator threads (module docs, docs/az.md 71).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneMode {
    /// `eval_layout`'s lanes: on the CPU lanes per network seat key, on CUDA / TensorRT-RTX one per generation, on any
    /// other GPU provider one for every network.
    Auto,
    /// One evaluator thread for every network and the settings' groups (the layout before docs/az.md 71; tests and
    /// measurements).
    One,
}

impl LaneMode {
    pub fn parse(s: &str) -> Result<LaneMode, String> {
        match s {
            "auto" => Ok(LaneMode::Auto),
            "one" => Ok(LaneMode::One),
            _ => Err(format!("lanes {s:?}: 'auto' or 'one'")),
        }
    }
}

/// An eval job's layout (module docs, docs/az.md 71): the driver's groups, the lanes (evaluator threads) each of the
/// driver's keys sends its batches to (in turn), the keys each lane answers, each lane's sessions' intra-op threads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub groups: usize,
    pub key_lanes: Vec<Vec<usize>>,
    pub lane_keys: Vec<Vec<usize>>,
    pub intra: usize,
}

/// The layout of an eval job. `key_gens`: each driver key's network generation (None: a bot's key); `provider`: the
/// first the job asks for; `job_groups`: the job line's `groups`; `settings_groups`: the settings'; `intra` /
/// `workers`: the job's sessions' threads and sessions (`proto::job_ort`: its `infer_threads` or search threads T, one
/// session; macOS's CPU 3-thread sessions, T / 3 of them); `own`: the job named its `infer_threads`.
///
/// * `LaneMode::One`: the layout before - the job's groups or the settings', one lane for every network, `intra`.
/// * A GPU provider: the job's groups or the settings' (the agent's GPU rule leaves out a `groups` equal to the
///   template's); CUDA / TensorRT-RTX a lane per generation, any other provider one lane; the job's threads shared by
///   the lanes unless its own.
/// * The CPU: sessions of S = min(`intra`, `EVAL_SESSION_THREADS`) threads (`intra` when the job named its threads); n
///   network keys; the threads `intra` x `workers` = B. Groups: the job's, else `EVAL_GROUPS` (never the settings': the
///   run's own self-play batching made CPU leases ~2 rows a call). Each key gets W = B / (n S) lanes within 1..groups (its
///   batches go to them in turn: with the job's groups G > 1, a group's batch runs beside the others'), each lane its own
///   session (one network on both seats: sessions of it per key), of B / (n W) threads within 1..S (the job's own
///   `intra` when named). Two networks, no job groups: 16 threads 1 group, 2 lanes of 4; 6 threads 2 lanes of 3; a Mac's
///   3 x 5 2 lanes of 3; a job's groups 2 on 16 threads: 4 lanes of 4.
#[allow(clippy::too_many_arguments)]
pub fn eval_layout(key_gens: &[Option<u32>], mode: LaneMode, provider: ProviderKind, job_groups: Option<usize>, settings_groups: usize, intra: usize, workers: usize, own: bool) -> Layout {
    let intra = intra.max(1);
    let budget = intra * workers.max(1);
    let nn: Vec<usize> = (0..key_gens.len()).filter(|&k| key_gens[k].is_some()).collect();
    let mut key_lanes = vec![Vec::new(); key_gens.len()];
    let mut lane_keys: Vec<Vec<usize>> = Vec::new();
    let mut lane_gen: Vec<u32> = Vec::new();
    // (one lane per generation, or one for all)
    let mut shared = |all: bool, key_lanes: &mut Vec<Vec<usize>>, lane_keys: &mut Vec<Vec<usize>>| {
        for &k in &nn {
            let g = key_gens[k].unwrap();
            let i = match (all, lane_gen.iter().position(|&x| x == g)) {
                (true, _) if !lane_keys.is_empty() => 0,
                (false, Some(i)) => i,
                _ => {
                    lane_keys.push(Vec::new());
                    lane_gen.push(g);
                    lane_keys.len() - 1
                }
            };
            lane_keys[i].push(k);
            key_lanes[k].push(i);
        }
    };
    let cpu = provider == ProviderKind::Cpu;
    let (groups, intra) = match (mode, cpu) {
        (LaneMode::One, _) => {
            shared(true, &mut key_lanes, &mut lane_keys);
            (job_groups.unwrap_or(settings_groups).max(1), intra)
        }
        (LaneMode::Auto, false) => {
            let per_gen = matches!(provider, ProviderKind::Cuda | ProviderKind::TensorRtRtx);
            shared(!per_gen, &mut key_lanes, &mut lane_keys);
            let t = if own { intra } else { (budget / lane_keys.len().max(1)).clamp(1, intra) };
            (job_groups.unwrap_or(settings_groups).max(1), t)
        }
        (LaneMode::Auto, true) => {
            let s = if own { intra } else { intra.min(EVAL_SESSION_THREADS) };
            let n = nn.len().max(1);
            let fit = budget / (n * s);
            let groups = job_groups.unwrap_or(EVAL_GROUPS).max(1);
            let w = fit.clamp(1, groups);
            for &k in &nn {
                for _ in 0..w {
                    lane_keys.push(vec![k]);
                    key_lanes[k].push(lane_keys.len() - 1);
                }
            }
            let t = if own { intra } else { (budget / (n * w)).clamp(1, s) };
            (groups, t)
        }
    };
    Layout { groups, key_lanes, lane_keys, intra }
}

/// The lanes' evaluators (`Layout::lane_keys`): each lane a session of every generation its keys play (`build` makes
/// one), its keys mapped to them; every other key of the driver None in that lane.
fn lane_evaluators(key_gens: &[Option<u32>], lane_keys: &[Vec<usize>], build: &mut dyn FnMut(u32) -> Result<Box<dyn Evaluator>, String>) -> Result<Vec<ByKey>, String> {
    let mut lanes = Vec::with_capacity(lane_keys.len());
    for ks in lane_keys {
        let mut evs: Vec<(u32, Box<dyn Evaluator>)> = Vec::new();
        for &k in ks {
            let g = key_gens[k].ok_or("internal: a lane's key without a network")?;
            if !evs.iter().any(|e| e.0 == g) {
                evs.push((g, build(g)?));
            }
        }
        let key_ev = key_gens.iter().enumerate().map(|(k, g)| g.filter(|_| ks.contains(&k)).and_then(|g| evs.iter().position(|e| e.0 == g))).collect();
        lanes.push(ByKey { evs, key_ev });
    }
    Ok(lanes)
}

/// The evaluators of the driver's keys a lane answers: one per generation, a key ('m': seat 0's network, 'o': seat
/// 1's) answered by its generation's (a lane per key: one network, one key).
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
    fn provider(&self) -> String {
        // (the networks' providers; one name when they agree, as they do on one PC)
        let mut p: Vec<String> = self.evs.iter().map(|(_, e)| e.provider()).collect();
        p.dedup();
        if p.is_empty() {
            "uniform".into()
        } else {
            p.join("+")
        }
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
    /// Answers of later batches that came before the oldest's (several lanes).
    early: HashMap<u64, (Result<Answer, String>, f64)>,
    /// Each driver key's lanes (evaluator threads; its batches go to them in turn) and its next one.
    key_lanes: Vec<Vec<usize>>,
    rr: Vec<usize>,
    /// Batches in flight at most: INFLIGHT, or one per lane.
    inflight: usize,
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
    /// Rows carry their game's record (`eval_trace`).
    trace: bool,
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
        for f in self.drv.take_failures() {
            self.sink.log(&format!("game failure (dropped after a panic): {f}"));
        }
        let n = res.len() as u64;
        for r in res {
            {
                let mut p = self.progress.lock().unwrap();
                p.finished_gids.push(r.gid);
                p.decisions += r.decisions as u64;
            }
            let mut row = row_json(r.gid, self.job.paired, self.job.seats_of(r.gid), &r);
            // (a game dropped after a panic has no trace: its row says so - `"abort_reason":"panic"` - without a record;
            // docs/search.md 2.18)
            if self.trace && r.abort_reason != Some("panic") {
                row = traced_row_json(&row, &r)?;
            }
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
        let k = b.key as usize;
        let lanes = self.key_lanes.get(k).filter(|l| !l.is_empty()).ok_or_else(|| format!("a batch of key {} that no network answers", b.key))?;
        let lane = lanes[self.rr[k] % lanes.len()];
        self.rr[k] += 1;
        let mut s = self.ctl.s.lock().unwrap();
        while s.yielded && !s.quit {
            s = self.ctl.cv.wait(s).unwrap();
        }
        self.pending.push_back(b.id);
        self.link.eval_to(lane, b)
    }

    fn finish_oldest(&mut self) -> Result<(), String> {
        let want = self.pending.pop_front().expect("a batch in flight");
        // (several lanes: a later batch's answer may come first - kept until its turn; the answers are submitted in
        // launch order, so the driver sees what one lane would have given it)
        let (a, secs) = loop {
            if let Some(x) = self.early.remove(&want) {
                break x;
            }
            let (id, a, secs) = self.answers.recv().map_err(|_| "the evaluator thread is gone".to_string())?;
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

/// Each of the driver's keys' network generation (None: a bot's key).
fn key_gens_of(job: &EvalJob, keys: &[(String, KeyKind)]) -> Result<Vec<Option<u32>>, String> {
    let mut key_gens = Vec::with_capacity(keys.len());
    for (k, kind) in keys {
        let seat = match k.as_str() {
            "m" => &job.seats[0],
            "o" => &job.seats[1],
            other => return Err(format!("internal: an eval driver key {other:?}")),
        };
        key_gens.push(match (seat.net(), kind) {
            (Some(g), KeyKind::Nn) => Some(g),
            _ => None,
        });
    }
    Ok(key_gens)
}

/// An eval job's evaluator lanes (`layout`; module docs): every model of `models` loaded and self-tested (the event
/// `selftest` with its `gen`, once per generation - a generation's further sessions are tested quietly; a failure ends
/// the job before any game), each lane's sessions with the layout's intra-op threads.
fn build_lanes(game: &GameEntry, job: &EvalJob, key_gens: &[Option<u32>], layout: &Layout, belief: bool, sink: &dyn Sink, progress: &Mutex<Progress>) -> Result<Vec<ByKey>, String> {
    let want = Expect::of_entry(game, belief);
    let ort = job.ort.as_ref().map(|o| {
        let mut o = o.clone();
        o.intra_threads = layout.intra;
        o.workers = 1;
        o
    });
    let mut worst: Option<f64> = None;
    let mut tested: Vec<u32> = Vec::new();
    let mut build = |gen: u32| -> Result<Box<dyn Evaluator>, String> {
        let m = job.models.get(&gen).ok_or_else(|| format!("internal: no model of gen {gen}"))?;
        Ok(match &ort {
            None => Box::new(Uniform),
            Some(o) => {
                let first_of_gen = !tested.contains(&gen);
                tested.push(gen);
                // (the file's sha256, checked by build_eval before the graph is loaded)
                let kind = EvalKind::Onnx { model: m.model.clone(), ort: o.clone(), sha256: m.sha256.clone(), reference: m.reference.clone() };
                let ev = build_eval(&kind, &want, if first_of_gen { sink } else { &NoSink }, progress, Some(gen))?;
                // (Stats' selftest_max_abs: the largest over the job's sessions)
                if let Some(x) = progress.lock().unwrap().selftest_max_abs {
                    worst = Some(worst.map_or(x, |w| if x.is_nan() || x > w { x } else { w }));
                }
                ev
            }
        })
    };
    let lanes = lane_evaluators(key_gens, &layout.lane_keys, &mut build)?;
    // every model of the job checked and self-tested, a network no seat plays too (as before)
    for &g in job.models.keys() {
        if job.ort.is_some() && !key_gens.contains(&Some(g)) {
            drop(build(g)?);
        }
    }
    progress.lock().unwrap().selftest_max_abs = worst;
    Ok(lanes)
}

/// Run an eval job to its end (`link_out`: the evaluator link, for the agent's yield / resume).
pub fn run_eval_job(game: &GameEntry, job: &EvalJob, ctl: &Control, sink: &dyn Sink, progress: Arc<Mutex<Progress>>, link_out: &dyn Fn(EvalLink)) -> Result<Outcome, String> {
    let t0 = Instant::now();
    job.check()?;
    let info = (game.info)();
    let m = eval_driver_cfg(&job.settings, &info, &job.seats, job.games, job.seed, job.paired, job.conc, job.search_threads, job.groups)?;
    let mut cfg = eval_lease_cfg(m);
    let belief = cfg.seats.iter().any(|s| matches!(s, SeatCfg::Nn(n) if NnSeat::belief_on(n)));
    let spec = eval_spec_of(&job.settings)?;
    let trace = eval_trace_of(&job.settings)?;
    // the layout (module docs): the driver's groups (cfg.groups is the job's or the settings'), the lanes, their threads
    let key_gens = key_gens_of(job, &cfg.keys)?;
    // (the provider the job asks for first decides; a session that falls back to the CPU keeps its lane)
    let first = job.ort.as_ref().and_then(|o| o.providers.first().copied()).unwrap_or(ProviderKind::Cpu);
    let (intra0, workers0) = job.ort.as_ref().map_or((cfg.threads, 1), |o| (o.intra_threads, o.workers));
    let layout = eval_layout(&key_gens, job.lanes, first, job.groups, cfg.groups, intra0, workers0, job.own_infer_threads);
    cfg.groups = layout.groups;
    let lanes = build_lanes(game, job, &key_gens, &layout, belief, sink, &progress)?;
    let intra = layout.intra;
    let lane_text: Vec<String> = lanes
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let ks: Vec<&str> = layout.lane_keys[i].iter().map(|&k| cfg.keys[k].0.as_str()).collect();
            format!("lane {i} (keys {}): {}", ks.join(", "), l.describe())
        })
        .collect();
    sink.log(&format!(
        "eval job {}: games {}..{} ({}{}), seed {}, seats {:?}; driver: conc {}, threads {}, groups {}; {} evaluator lane(s) ({:?}, {} intra-op threads each): {}",
        job.lease_id,
        job.gid_base,
        job.gid_base as u64 + job.games,
        if job.paired { "paired" } else { "unpaired" },
        if trace { ", traced" } else { "" },
        job.seed,
        job.seats,
        cfg.concurrency,
        cfg.threads,
        cfg.groups,
        lanes.len(),
        job.lanes,
        intra,
        lane_text.join("; ")
    ));
    {
        let mut p = progress.lock().unwrap();
        p.concurrency = cfg.concurrency;
        p.threads = cfg.threads;
        // (the lanes' providers; one name when they agree, as they do on one PC)
        let mut pv: Vec<String> = lanes.iter().map(|l| l.provider()).collect();
        pv.sort();
        pv.dedup();
        p.provider = if pv.is_empty() { "uniform".into() } else { pv.join("+") };
    }
    let mut drv = (game.new_driver)(cfg)?;
    drv.set_first_gid(job.gid_base)?;
    if trace {
        // (every game's seed, real moves and setup: its record, replayed on the desktop; docs/search.md 2.17)
        drv.set_trace(true)?;
    }
    for k in 0..job.games {
        drv.enqueue(&format!("{}:{}", job.lease_id, job.gid_base as u64 + k), &spec)?;
    }
    drv.resume(BIG, 0)?;
    // (an evaluator thread per lane; a bot-only match has no lane: one idle thread keeps the link's yield / resume)
    let n_lanes = lanes.len();
    let evs: Vec<Box<dyn Evaluator>> = if lanes.is_empty() { vec![Box::new(Uniform)] } else { lanes.into_iter().map(|l| Box::new(l) as Box<dyn Evaluator>).collect() };
    let (link, answers, eval_threads) = spawn_eval(evs);
    link_out(link.clone());
    let mut r = EvalRun {
        job,
        sink,
        ctl,
        drv,
        link,
        answers,
        pending: VecDeque::new(),
        early: HashMap::new(),
        rr: vec![0; layout.key_lanes.len()],
        key_lanes: layout.key_lanes,
        inflight: INFLIGHT.max(n_lanes),
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
        trace,
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
        (Some(rc), Ok(())) => rc.write(&crate::job::stamped(game, json!({"game": info.name, "features": info.num_features, "actions": info.num_actions, "seed": job.seed, "worker": "zero_worker eval"})), &info),
        _ => Ok(()),
    };
    r.snapshot();
    let EvalRun { link, written, .. } = r;
    link.stop();
    drop(link);
    for h in eval_threads {
        let _ = h.join();
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn lay(gens: &[Option<u32>], mode: LaneMode, p: ProviderKind, job_groups: Option<usize>, settings_groups: usize, intra: usize, workers: usize, own: bool) -> (usize, Vec<Vec<usize>>, Vec<Vec<usize>>, usize) {
        let l = eval_layout(gens, mode, p, job_groups, settings_groups, intra, workers, own);
        (l.groups, l.key_lanes, l.lane_keys, l.intra)
    }

    #[test]
    fn eval_layouts() {
        use ProviderKind::{Cpu, Cuda, DirectMl};
        let two = [Some(1), Some(2)];
        // the layout before: the settings' groups (or the job's), one lane, the job's threads
        assert_eq!(lay(&two, LaneMode::One, Cpu, None, 4, 16, 1, false), (4, vec![vec![0], vec![0]], vec![vec![0, 1]], 16));
        assert_eq!(lay(&two, LaneMode::One, Cpu, Some(2), 4, 16, 1, false), (2, vec![vec![0], vec![0]], vec![vec![0, 1]], 16));
        // the CPU: 16 threads -> one group (never the settings' 4), a lane per key of 4 threads
        let one = (1, vec![vec![0], vec![1]], vec![vec![0], vec![1]], 4);
        assert_eq!(lay(&two, LaneMode::Auto, Cpu, None, 4, 16, 1, false), one);
        // 20 or 32 threads: the same (sessions of 4); 6 threads: 3 each; 1 thread: 1 each
        assert_eq!(lay(&two, LaneMode::Auto, Cpu, None, 4, 20, 1, false), one);
        assert_eq!(lay(&two, LaneMode::Auto, Cpu, None, 4, 32, 1, false), one);
        assert_eq!(lay(&two, LaneMode::Auto, Cpu, None, 4, 6, 1, false), (1, vec![vec![0], vec![1]], vec![vec![0], vec![1]], 3));
        assert_eq!(lay(&two, LaneMode::Auto, Cpu, None, 4, 1, 1, false), (1, vec![vec![0], vec![1]], vec![vec![0], vec![1]], 1));
        // macOS's 3-thread sessions (16 threads: 3 x 5): a lane per key of 3
        assert_eq!(lay(&two, LaneMode::Auto, Cpu, None, 4, 3, 5, false), (1, vec![vec![0], vec![1]], vec![vec![0], vec![1]], 3));
        // the job's own groups: kept; its batches over as many lanes per key as the threads hold (16: 2 of 4 each)
        assert_eq!(lay(&two, LaneMode::Auto, Cpu, Some(2), 4, 16, 1, false), (2, vec![vec![0, 1], vec![2, 3]], vec![vec![0], vec![0], vec![1], vec![1]], 4));
        assert_eq!(lay(&two, LaneMode::Auto, Cpu, Some(4), 4, 32, 1, false).2.len(), 8);
        assert_eq!(lay(&two, LaneMode::Auto, Cpu, Some(1), 4, 16, 1, false), one);
        // the job's own threads (a gate's --infer-threads 4): sessions of exactly 4, one lane per key on 4 threads
        assert_eq!(lay(&two, LaneMode::Auto, Cpu, None, 4, 4, 1, true), (1, vec![vec![0], vec![1]], vec![vec![0], vec![1]], 4));
        // one network on both seats: lanes per key all the same (sessions of it per key)
        assert_eq!(lay(&[Some(1), Some(1)], LaneMode::Auto, Cpu, None, 4, 6, 1, false), (1, vec![vec![0], vec![1]], vec![vec![0], vec![1]], 3));
        // a bot's key has no lane; one network key: one group, its lane of 4
        assert_eq!(lay(&[None, Some(2)], LaneMode::Auto, Cpu, None, 4, 16, 1, false), (1, vec![vec![], vec![0]], vec![vec![1]], 4));
        assert_eq!(lay(&[None, None], LaneMode::Auto, Cpu, None, 4, 16, 1, false).2, Vec::<Vec<usize>>::new());
        // CUDA: the job's or the settings' groups, a lane per generation, the threads shared
        assert_eq!(lay(&two, LaneMode::Auto, Cuda, None, 2, 16, 1, false), (2, vec![vec![0], vec![1]], vec![vec![0], vec![1]], 8));
        assert_eq!(lay(&[Some(1), Some(1)], LaneMode::Auto, Cuda, Some(1), 4, 16, 1, false), (1, vec![vec![0], vec![0]], vec![vec![0, 1]], 16));
        // other GPU providers: one lane, as before
        assert_eq!(lay(&two, LaneMode::Auto, DirectMl, None, 2, 16, 1, false), (2, vec![vec![0], vec![0]], vec![vec![0, 1]], 16));
    }

    /// An evaluator whose answer names its network: the value of every row is the generation.
    struct Tagged(u32);
    impl Evaluator for Tagged {
        fn evaluate(&mut self, b: &Batch) -> Result<Answer, String> {
            Ok(Answer { value: vec![self.0 as f32; b.b * zero_search::MAX_SEATS], ..Default::default() })
        }
        fn describe(&self) -> String {
            format!("tagged {}", self.0)
        }
    }

    #[test]
    fn each_key_reaches_its_network() {
        use ProviderKind::{Cpu, Cuda, DirectMl};
        for (gens, p, threads) in [([Some(1), Some(2)], Cpu, 16), ([Some(1), Some(2)], Cpu, 6), ([Some(1), Some(2)], Cuda, 16), ([Some(1), Some(2)], DirectMl, 16), ([Some(2), Some(2)], Cpu, 16), ([Some(2), Some(2)], Cuda, 16), ([None, Some(2)], Cpu, 16)] {
            let l = eval_layout(&gens, LaneMode::Auto, p, Some(2).filter(|_| threads > 8), 2, threads, 1, false);
            let mut built = Vec::new();
            let mut lanes = lane_evaluators(&gens, &l.lane_keys, &mut |g| {
                built.push(g);
                Ok(Box::new(Tagged(g)))
            })
            .unwrap();
            // one session per (lane, generation)
            let want_sessions: usize = l.lane_keys.iter().map(|ks| {
                let mut g: Vec<u32> = ks.iter().map(|&k| gens[k].unwrap()).collect();
                g.sort();
                g.dedup();
                g.len()
            }).sum();
            assert_eq!(built.len(), want_sessions, "{gens:?} {p:?}");
            for (k, g) in gens.iter().enumerate() {
                let mut b = Batch::default();
                (b.key, b.b) = (k as u16, 1);
                match g {
                    Some(g) => {
                        assert!(!l.key_lanes[k].is_empty());
                        for (j, lane) in lanes.iter_mut().enumerate() {
                            if l.key_lanes[k].contains(&j) {
                                assert_eq!(lane.evaluate(&b).unwrap().value[0], *g as f32, "key {k} of {gens:?}: its network");
                            } else {
                                assert!(lane.evaluate(&b).is_err(), "lane {j} answers only its keys");
                            }
                        }
                    }
                    None => assert!(l.key_lanes[k].is_empty()),
                }
            }
        }
    }
}
