//! Batched multi-threaded self-play / evaluation driver (docs/search.md 2), generic over
//! `SearchGame`. A port of Dominion Zero's `engine/src/search/driver.rs` without the
//! Dominion-only parts (token observations, staging buffers, exploration games, loop guard),
//! plus **external seats**.
//!
//! The driver runs `concurrency` games at once, split into `groups` groups; each group is
//! split into chunks of slots that the worker threads advance in parallel. Advancing a slot =
//! play its game (bots and single-option decisions inline, network seats through their search
//! tree) until its search needs one network evaluation - or an external seat is to move -,
//! exactly one leaf / request per game per batch (no virtual loss, no in-tree batching).
//! When every chunk of a group is done, its leaves are assembled into one batch per evaluator
//! key (network keys: a [B, F] feature matrix + legal-action lists; external keys: the
//! games' exported states). The caller takes batches with `next_batch` (groups in fixed
//! round-robin order, so a run is reproducible from its seed and the evaluator outputs) and
//! answers with `submit` (priors / values / margins) or `submit_external` (one payload per
//! request, applied to the game by `SearchGame::apply_external`); once all batches of a group
//! are answered the group is dispatched again (finished games are replaced then). With
//! `groups >= 2` the CPU searches group B while the caller evaluates group A.
//!
//! Per decision (docs/search.md 3.4): playout-cap randomisation, root Dirichlet noise,
//! temperature sampling for the first `temp_moves` searched decisions, training records of
//! full searches (policy target: visits or completed Q, `PolicyTarget`), tree reuse.
//!
//! **Rolling mode** (`DriverCfg::rolling`): games never run out - game `gid` takes the seats
//! `games[gid % games.len()]` - and play is divided into phases. `resume(quota, net)` runs
//! until `quota` more games have finished, then the driver pauses (no group is dispatched
//! again; `next_batch` returns None once every group is paused). The games in progress keep
//! their state. A `resume` with a new network id starts a new search epoch: every slot drops
//! its trees and pending leaf and searches its current decision again from an empty tree
//! (keeping that decision's playout-cap draw), so no search mixes two networks.
use std::collections::{HashMap, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::Instant;

use crate::game::{make_bot, Bot, SearchGame, MAX_SEATS};
use crate::rng::{split64, Rng};
use crate::tree::Tree;
use crate::xbuf::{XBuf, XDtype, XPool, XPtr};

mod worker;
use worker::worker;

const NO_GAME: u32 = u32::MAX;

/// Policy target of a recorded decision (docs/search.md 3.1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PolicyTarget {
    /// The root visit distribution N / sum N (P0 if nothing was visited).
    Visits,
    /// Gumbel AlphaZero's completed-Q target `softmax(ln P0 + (c_visit + max N) c_scale
    /// completedQ)` (`tree::completed_q_policy`; Danihelka et al. 2022: c_visit 50, c_scale 1).
    CompletedQ { c_visit: f64, c_scale: f64 },
}

impl PolicyTarget {
    /// From the seat settings `policy_target` ('visits' | 'cq'), `cq_c_visit`, `cq_c_scale`.
    pub fn parse(name: &str, c_visit: f64, c_scale: f64) -> Result<PolicyTarget, String> {
        match name {
            "visits" => Ok(PolicyTarget::Visits),
            "cq" => {
                if !(c_visit.is_finite() && c_visit >= 0.0 && c_scale.is_finite() && c_scale >= 0.0) {
                    return Err(format!("policy_target 'cq' needs finite cq_c_visit >= 0 and cq_c_scale >= 0 (got {c_visit}, {c_scale})"));
                }
                Ok(PolicyTarget::CompletedQ { c_visit, c_scale })
            }
            _ => Err(format!("policy_target {name:?}: 'visits' or 'cq'")),
        }
    }
}

/// Search settings of a network seat.
#[derive(Clone, Debug, PartialEq)]
pub struct NnSeat {
    /// Evaluator key index (`DriverCfg::keys`, a network key).
    pub key: u16,
    pub sims: u32,
    pub c_puct: f64,
    pub selfplay: bool,
    pub alpha: f64,
    pub eps: f64,
    pub temp_moves: u32,
    pub sims_fast: Option<u32>,
    pub p_full: f64,
    pub chance_k: u32,
    pub margin_w: f64,
    pub redet: bool,
    /// Training target of the seat's recorded decisions (self-play, full searches).
    pub policy_target: PolicyTarget,
}

impl NnSeat {
    /// Evaluation seat defaults.
    pub fn eval(key: u16, sims: u32) -> NnSeat {
        NnSeat { key, sims, c_puct: 1.0, selfplay: false, alpha: 0.3, eps: 0.25, temp_moves: 30, sims_fast: None, p_full: 1.0, chance_k: 0, margin_w: 0.0, redet: true, policy_target: PolicyTarget::Visits }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SeatCfg {
    /// A built-in bot by name (`random`, or one of the game's `bot_names`).
    Bot(String),
    /// A seat played from outside (evaluator key index of an external key).
    External(u16),
    Nn(NnSeat),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyKind {
    Nn,
    External,
}

#[derive(Clone, Debug)]
pub struct DriverCfg {
    /// The game's spec (JSON, `SearchGame::parse_spec`).
    pub spec: String,
    /// Distinct seats; `games[g][p]` indexes this table.
    pub seats: Vec<SeatCfg>,
    /// Seats per game (players = its length). Rolling mode: a cyclic table.
    pub games: Vec<Vec<u16>>,
    /// Evaluator keys ('m', 'o', 'ext:jsbot', ...) and their kinds.
    pub keys: Vec<(String, KeyKind)>,
    pub concurrency: usize,
    pub seed: u64,
    pub record: bool,
    pub threads: usize,
    pub groups: usize,
    /// Real moves after which a game is abandoned (result: a draw; `over` false).
    pub max_steps: u64,
    pub rolling: bool,
    /// Network id stored with the samples until a `resume` sets another.
    pub net: u32,
    /// Element type of the network batches' features (`Batch::x`): F32 = the game's encoding
    /// as is; F16 = rounded to half precision by the fill jobs (half the bytes to copy and
    /// upload; docs/search.md 2.2).
    pub x_dtype: XDtype,
}

/// A finished game (docs/search.md 2.5).
#[derive(Clone, Debug)]
pub struct GameResult {
    pub gid: u32,
    pub n: usize,
    pub over: bool,
    pub result: Vec<f32>,
    pub scores: Vec<f32>,
    pub steps: u64,
    /// Searched decisions (network seats).
    pub decisions: u32,
    /// Answers applied for external seats.
    pub ext_moves: u32,
    pub rounds: u32,
    /// Training samples this game emitted.
    pub samples: u32,
}

/// One recorded root decision with its targets.
#[derive(Clone, Debug)]
pub struct Record {
    pub x: Vec<f32>,
    pub player: u8,
    pub n_players: u8,
    pub opts: Vec<u16>,
    /// Policy target over `opts` (the seat's `PolicyTarget`).
    pub pi: Vec<f32>,
    pub z: [f32; MAX_SEATS],
    pub q: [f32; MAX_SEATS],
    pub sc: [f32; MAX_SEATS],
    pub net: u32,
}

/// A batch for one evaluator key (docs/search.md 2.2).
#[derive(Clone, Debug, Default)]
pub struct Batch {
    pub id: u64,
    pub key: u16,
    pub external: bool,
    group: usize,
    pub b: usize,
    /// Network batches: features [b * F] (`DriverCfg::x_dtype`; a pooled buffer: dropping the
    /// batch recycles it), legal actions as segments of `opt_action`.
    pub x: XBuf,
    pub opt_off: Vec<i32>,
    pub opt_action: Vec<i32>,
    /// Deciding seat (absolute), players, game id per row.
    pub player: Vec<i32>,
    pub n_players: Vec<i32>,
    pub gid: Vec<i64>,
    /// External batches: the exported state per row.
    pub states: Vec<String>,
}

/// Rows of one chunk for one key.
#[derive(Default)]
struct LeafBuf {
    x: Vec<f32>,
    player: Vec<i32>,
    n_players: Vec<i32>,
    gid: Vec<i64>,
    nopt: Vec<i32>,
    opt_action: Vec<i32>,
    states: Vec<String>,
}

impl LeafBuf {
    fn clear(&mut self) {
        self.x.clear();
        self.player.clear();
        self.n_players.clear();
        self.gid.clear();
        self.nopt.clear();
        self.opt_action.clear();
        self.states.clear();
    }
    fn len(&self) -> usize {
        self.player.len()
    }
}

enum Eval {
    Nn { opt_off: Vec<i32>, priors: Vec<f32>, value: Vec<f32>, margin: Vec<f32>, bases: Vec<u32> },
    Ext { payloads: Vec<String>, bases: Vec<u32> },
}

enum Pending {
    Leaf { key: u16, local: u32, seat: usize, node: u32 },
    Ext { key: u16, local: u32 },
}

struct Slot<G: SearchGame> {
    gid: u32,
    start: Option<u32>,
    game: Option<G>,
    n: usize,
    rng: Rng,
    noise_rng: Rng,
    search_rng: Rng,
    bots: Vec<Option<Box<dyn Bot<G>>>>,
    trees: [Option<Box<Tree<G>>>; MAX_SEATS],
    spare: Vec<Box<Tree<G>>>,
    sims: u32,
    target: Option<u32>,
    full: bool,
    decisions: u32,
    ext_moves: u32,
    steps: u64,
    records: Vec<Record>,
    rq: Vec<Option<[f64; MAX_SEATS]>>,
    pending: Option<Pending>,
    ext_payload: Option<String>,
    /// Path of the pending leaf's simulation (kept until its evaluation is backed up).
    path: Vec<(u32, u32)>,
    /// Legal actions of the live game's pending decision (valid iff `legal_ok`).
    legal: Vec<u16>,
    legal_ok: bool,
    epoch: u32,
    net: u32,
}

impl<G: SearchGame> Slot<G> {
    fn empty() -> Slot<G> {
        Slot {
            gid: NO_GAME,
            start: None,
            game: None,
            n: 0,
            rng: Rng::new(0),
            noise_rng: Rng::new(0),
            search_rng: Rng::new(0),
            bots: Vec::new(),
            trees: Default::default(),
            spare: Vec::new(),
            sims: 0,
            target: None,
            full: true,
            decisions: 0,
            ext_moves: 0,
            steps: 0,
            records: Vec::new(),
            rq: Vec::new(),
            pending: None,
            ext_payload: None,
            path: Vec::new(),
            legal: Vec::new(),
            legal_ok: false,
            epoch: 0,
            net: 0,
        }
    }
    fn drop_tree(&mut self, q: usize) {
        if let Some(t) = self.trees[q].take() {
            self.spare.push(t);
        }
    }
    fn drop_trees(&mut self) {
        for q in 0..MAX_SEATS {
            self.drop_tree(q);
        }
    }
    fn ensure_legal(&mut self) {
        if !self.legal_ok {
            self.legal.clear();
            self.game.as_ref().unwrap().legal_actions(&mut self.legal);
            self.legal_ok = true;
        }
    }
}

struct Chunk<G: SearchGame> {
    slots: Vec<Slot<G>>,
    bufs: Vec<LeafBuf>,
    results: Vec<GameResult>,
    samples: Vec<Record>,
}

#[derive(Default)]
struct GroupOut {
    batches: Vec<Batch>,
    results: Vec<GameResult>,
    samples: Vec<Record>,
}

/// Raw destination of one network batch (its Vecs' heap buffers), written by the group's
/// fill jobs, each its own rows / options.
#[derive(Clone, Copy)]
struct Dest {
    x: XPtr,
    player: *mut i32,
    n_players: *mut i32,
    gid: *mut i64,
    opt_off: *mut i32,
    opt_action: *mut i32,
}

// SAFETY: a Dest points into the heap buffers of a batch owned by the group's `Asm`, which
// keeps them alive and unresized until the last fill job finished; fill jobs write disjoint
// rows / option ranges.
unsafe impl Send for Dest {}
unsafe impl Sync for Dest {}

/// A network batch being filled by the group's fill jobs.
struct AsmKey {
    key: usize,
    dest: Dest,
    row_base: Vec<usize>,
    opt_base: Vec<usize>,
}

#[derive(Default)]
struct Asm {
    batches: Vec<Batch>,
    keys: Vec<AsmKey>,
}

struct Group<G: SearchGame> {
    chunks: Vec<Mutex<Chunk<G>>>,
    remaining: AtomicUsize,
    asm: Mutex<Option<Asm>>,
    fill_remaining: AtomicUsize,
    evals: RwLock<Vec<Option<Eval>>>,
    out: Mutex<Option<GroupOut>>,
    /// Per key: row offset of every chunk in the last assembled batch.
    bases: Mutex<Vec<Vec<u32>>>,
}

#[derive(Default)]
struct Counters {
    sims: AtomicU64,
    leaves: AtomicU64,
    ext_requests: AtomicU64,
    busy_ns: AtomicU64,
    assemble_ns: AtomicU64,
    restart_sims: AtomicU64,
    restarts: AtomicU64,
    recorded: AtomicU64,
}

struct Shared<G: SearchGame> {
    cfg: DriverCfg,
    spec: G::Spec,
    groups: Vec<Group<G>>,
    queue: Mutex<VecDeque<(u32, u32)>>,
    queue_cv: Condvar,
    done: Mutex<Vec<bool>>,
    done_cv: Condvar,
    error: Mutex<Option<String>>,
    stop: AtomicBool,
    counters: Counters,
    epoch: AtomicU32,
    net: AtomicU32,
    /// Spare feature buffers (network batches).
    x_pool: Arc<XPool>,
}

impl<G: SearchGame> Shared<G> {
    fn fail(&self, msg: String) {
        let mut e = self.error.lock().unwrap();
        if e.is_none() {
            *e = Some(msg);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GState {
    Working,
    Outstanding(usize),
    Idle,
    Paused,
}

/// Throughput counters (`SearchDriver.stats()`).
#[derive(Clone, Debug, Default)]
pub struct DriverStats {
    pub sims: u64,
    pub leaves: u64,
    pub ext_requests: u64,
    pub batches: u64,
    pub busy_s: f64,
    pub assemble_s: f64,
    pub wait_s: f64,
    pub wall_s: f64,
    pub threads: usize,
    pub groups: usize,
    pub games_started: u64,
    pub games_finished: u64,
    pub phase_games: u64,
    pub quota: u64,
    pub dropped_leaves: u64,
    pub restart_sims: u64,
    pub restarts: u64,
    pub recorded: u64,
    pub emitted: u64,
    pub epoch: u32,
    pub net: u32,
}

/// The games in progress.
#[derive(Clone, Debug, Default)]
pub struct InFlight {
    pub games: u64,
    pub oldest_age_games: u64,
    pub max_steps: u64,
    pub max_rounds: u32,
}

pub struct Driver<G: SearchGame> {
    sh: Arc<Shared<G>>,
    workers: Vec<JoinHandle<()>>,
    gstate: Vec<GState>,
    cursor: usize,
    ready: VecDeque<Batch>,
    /// batch id -> (group, key, rows, opt_off of a network batch)
    outstanding: HashMap<u64, (usize, u16, usize, Vec<i32>)>,
    next_bid: u64,
    next_gid: u32,
    results: Vec<GameResult>,
    samples: Vec<Record>,
    wait_ns: u64,
    batches: u64,
    t0: Instant,
    finished_games: u64,
    phase_games: u64,
    quota: u64,
    dropped_leaves: u64,
    emitted: u64,
}

impl<G: SearchGame> Driver<G> {
    pub fn new(cfg: DriverCfg) -> Result<Driver<G>, String> {
        let spec = G::parse_spec(&cfg.spec)?;
        for (g, s) in cfg.games.iter().enumerate() {
            if !(G::MIN_PLAYERS..=G::MAX_PLAYERS.min(MAX_SEATS)).contains(&s.len()) {
                return Err(format!("game {g}: {} seats ({} supports {}..{})", s.len(), G::NAME, G::MIN_PLAYERS, G::MAX_PLAYERS));
            }
            if s.iter().any(|&i| i as usize >= cfg.seats.len()) {
                return Err(format!("game {g}: bad seat index"));
            }
        }
        for s in &cfg.seats {
            match s {
                SeatCfg::Nn(n) => {
                    if n.sims == 0 || n.sims_fast == Some(0) {
                        return Err("network seats need sims >= 1".into());
                    }
                    if cfg.keys.get(n.key as usize).map(|k| k.1) != Some(KeyKind::Nn) {
                        return Err("network seat: bad evaluator key".into());
                    }
                    if let PolicyTarget::CompletedQ { c_visit, c_scale } = n.policy_target {
                        PolicyTarget::parse("cq", c_visit, c_scale)?;
                    }
                }
                SeatCfg::External(k) => {
                    if cfg.keys.get(*k as usize).map(|k| k.1) != Some(KeyKind::External) {
                        return Err("external seat: bad evaluator key".into());
                    }
                }
                SeatCfg::Bot(name) => {
                    if make_bot::<G>(&spec, name, 0).is_none() {
                        return Err(format!("unknown bot {name:?} for {} ({:?})", G::NAME, crate::game::bot_names::<G>()));
                    }
                }
            }
        }
        if cfg.rolling && cfg.games.is_empty() {
            return Err("rolling mode needs at least one game's seats".into());
        }
        let conc = if cfg.rolling { cfg.concurrency.max(1) } else { cfg.concurrency.max(1).min(cfg.games.len().max(1)) };
        let groups = cfg.groups.max(1).min(conc);
        let threads = cfg.threads.max(1);
        let nkeys = cfg.keys.len();
        let mut gv = Vec::new();
        let mut max_ns = 0;
        for g in 0..groups {
            let ns = (g + 1) * conc / groups - g * conc / groups;
            max_ns = max_ns.max(ns);
            let nchunks = ns.min(threads * 4).max(1);
            let mut chunks = Vec::new();
            for c in 0..nchunks {
                let a = c * ns / nchunks;
                let b = (c + 1) * ns / nchunks;
                chunks.push(Mutex::new(Chunk { slots: (a..b).map(|_| Slot::empty()).collect(), bufs: (0..nkeys).map(|_| LeafBuf::default()).collect(), results: Vec::new(), samples: Vec::new() }));
            }
            gv.push(Group {
                chunks,
                remaining: AtomicUsize::new(0),
                asm: Mutex::new(None),
                fill_remaining: AtomicUsize::new(0),
                evals: RwLock::new((0..nkeys).map(|_| None).collect()),
                out: Mutex::new(None),
                bases: Mutex::new(vec![Vec::new(); nkeys]),
            });
        }
        let rolling = cfg.rolling;
        let net = cfg.net;
        // a batch has at most one row per slot of its group; at most one batch per network key
        // and group is outstanding (+2 spare: buffers dropped late)
        let nn_keys = cfg.keys.iter().filter(|k| k.1 == KeyKind::Nn).count();
        let x_pool = XPool::new(cfg.x_dtype, max_ns * G::NUM_FEATURES, groups * nn_keys + 2);
        let cfg = DriverCfg { threads, groups, ..cfg };
        let sh = Arc::new(Shared {
            cfg,
            spec,
            groups: gv,
            queue: Mutex::new(VecDeque::new()),
            queue_cv: Condvar::new(),
            done: Mutex::new(vec![false; groups]),
            done_cv: Condvar::new(),
            error: Mutex::new(None),
            stop: AtomicBool::new(false),
            counters: Counters::default(),
            epoch: AtomicU32::new(0),
            net: AtomicU32::new(net),
            x_pool,
        });
        let mut workers = Vec::new();
        for k in 0..threads {
            let sh2 = sh.clone();
            let h = std::thread::Builder::new().name(format!("zero-search-{k}")).spawn(move || worker(sh2)).map_err(|e| format!("spawn: {e}"))?;
            workers.push(h);
        }
        let mut d = Driver {
            sh,
            workers,
            gstate: vec![GState::Idle; groups],
            cursor: 0,
            ready: VecDeque::new(),
            outstanding: HashMap::new(),
            next_bid: 0,
            next_gid: 0,
            results: Vec::new(),
            samples: Vec::new(),
            wait_ns: 0,
            batches: 0,
            t0: Instant::now(),
            finished_games: 0,
            phase_games: 0,
            quota: 0,
            dropped_leaves: 0,
            emitted: 0,
        };
        for g in 0..groups {
            if rolling {
                d.gstate[g] = GState::Paused;
            } else {
                d.dispatch(g);
            }
        }
        Ok(d)
    }

    pub fn cfg(&self) -> &DriverCfg {
        &self.sh.cfg
    }

    fn pausing(&self) -> bool {
        self.sh.cfg.rolling && self.phase_games >= self.quota
    }

    fn dispatch(&mut self, g: usize) {
        let grp = &self.sh.groups[g];
        let n_games = if self.sh.cfg.rolling { u32::MAX } else { self.sh.cfg.games.len() as u32 };
        let mut busy = false;
        for ch in &grp.chunks {
            let mut ch = ch.lock().unwrap();
            for s in ch.slots.iter_mut() {
                if s.game.is_none() && s.start.is_none() && self.next_gid < n_games {
                    s.start = Some(self.next_gid);
                    self.next_gid += 1;
                }
                busy |= s.game.is_some() || s.start.is_some();
            }
        }
        if !busy {
            self.gstate[g] = GState::Idle;
            return;
        }
        self.gstate[g] = GState::Working;
        self.sh.done.lock().unwrap()[g] = false;
        grp.remaining.store(grp.chunks.len(), Ordering::SeqCst);
        let mut q = self.sh.queue.lock().unwrap();
        for c in 0..grp.chunks.len() {
            q.push_back((g as u32, c as u32));
        }
        drop(q);
        self.sh.queue_cv.notify_all();
    }

    fn check_error(&self) -> Result<(), String> {
        match &*self.sh.error.lock().unwrap() {
            Some(e) => Err(e.clone()),
            None => Ok(()),
        }
    }

    /// Rolling mode: end the current phase now (as if its quota had been reached).
    pub fn end_phase(&mut self) {
        if self.sh.cfg.rolling {
            self.quota = self.phase_games;
        }
    }

    /// Every game finished and every batch answered (never for a rolling driver).
    pub fn finished(&self) -> bool {
        self.ready.is_empty() && self.outstanding.is_empty() && self.gstate.iter().all(|s| *s == GState::Idle)
    }

    /// Rolling mode: the phase ended (or none began).
    pub fn paused(&self) -> bool {
        self.sh.cfg.rolling && self.ready.is_empty() && self.outstanding.is_empty() && self.gstate.iter().all(|s| matches!(s, GState::Paused | GState::Idle))
    }

    /// Rolling mode: play until `quota` more games have finished; `net` = the network id of
    /// the searches from now on (another id restarts every search in progress).
    pub fn resume(&mut self, quota: u64, net: u32) -> Result<(), String> {
        self.check_error()?;
        if !self.sh.cfg.rolling {
            return Err("resume: not a rolling driver".into());
        }
        if !self.paused() {
            return Err("resume: the current phase has not ended (take and answer its batches until next_batch returns None)".into());
        }
        if quota == 0 {
            return Err("resume: quota must be >= 1".into());
        }
        self.quota = quota;
        self.phase_games = 0;
        if self.sh.net.swap(net, Ordering::SeqCst) != net {
            self.sh.epoch.fetch_add(1, Ordering::SeqCst);
        }
        for g in 0..self.gstate.len() {
            self.dispatch(g);
        }
        Ok(())
    }

    /// The next batch, or None when every game has finished (rolling: the phase ended).
    /// `block = false`: None also when the next batch is not ready yet.
    pub fn next_batch(&mut self, block: bool) -> Result<Option<Batch>, String> {
        loop {
            self.check_error()?;
            if let Some(b) = self.ready.pop_front() {
                self.outstanding.insert(b.id, (b.group, b.key, b.b, b.opt_off.clone()));
                self.batches += 1;
                return Ok(Some(b));
            }
            if self.gstate.iter().all(|s| matches!(s, GState::Idle | GState::Paused)) {
                return Ok(None);
            }
            let ng = self.gstate.len();
            let Some(g) = (0..ng).map(|k| (self.cursor + k) % ng).find(|&g| self.gstate[g] == GState::Working) else {
                if !block {
                    return Ok(None);
                }
                return Err("submit the outstanding batches before asking for more".into());
            };
            let t = Instant::now();
            {
                let mut done = self.sh.done.lock().unwrap();
                if !block && !done[g] {
                    return Ok(None);
                }
                while !done[g] {
                    done = self.sh.done_cv.wait(done).unwrap();
                }
            }
            self.wait_ns += t.elapsed().as_nanos() as u64;
            self.check_error()?;
            let out = self.sh.groups[g].out.lock().unwrap().take().unwrap_or_default();
            let nfin = out.results.len() as u64;
            self.finished_games += nfin;
            self.phase_games += nfin;
            self.results.extend(out.results);
            self.samples.extend(out.samples);
            self.cursor = (g + 1) % ng;
            if self.pausing() {
                // the phase is over: this group's leaves are not evaluated (nothing was backed
                // up yet, the next phase walks again)
                for b in &out.batches {
                    self.dropped_leaves += b.b as u64;
                }
                if !out.batches.is_empty() {
                    for ch in &self.sh.groups[g].chunks {
                        for s in ch.lock().unwrap().slots.iter_mut() {
                            s.pending = None;
                        }
                    }
                }
                self.gstate[g] = GState::Paused;
                continue;
            }
            if out.batches.is_empty() {
                self.dispatch(g);
                continue;
            }
            self.gstate[g] = GState::Outstanding(out.batches.len());
            for mut b in out.batches {
                b.id = self.next_bid;
                self.next_bid += 1;
                self.ready.push_back(b);
            }
        }
    }

    fn answered(&mut self, g: usize) -> Result<(), String> {
        match self.gstate[g] {
            GState::Outstanding(1) if self.pausing() => self.gstate[g] = GState::Paused,
            GState::Outstanding(1) => self.dispatch(g),
            GState::Outstanding(k) => self.gstate[g] = GState::Outstanding(k - 1),
            s => return Err(format!("internal: group {g} in state {s:?} got an answer")),
        }
        Ok(())
    }

    /// Network evaluation of batch `id`: priors [options], value / margin [B * MAX_SEATS]
    /// (relative to the decider: column j = seat (player + j) % n_players).
    pub fn submit(&mut self, id: u64, priors: &[f32], value: &[f32], margin: &[f32]) -> Result<(), String> {
        let Some((g, key, b, opt_off)) = self.outstanding.get(&id).map(|o| (o.0, o.1, o.2, &o.3)) else {
            return Err(format!("unknown or already answered batch id {id}"));
        };
        if self.sh.cfg.keys[key as usize].1 != KeyKind::Nn {
            return Err(format!("batch {id} is an external batch: use submit_external"));
        }
        let nopt = *opt_off.last().unwrap_or(&0) as usize;
        if priors.len() != nopt || value.len() != b * MAX_SEATS || margin.len() != b * MAX_SEATS {
            return Err(format!("evaluation shapes: priors {} (want {nopt}), value {} / margin {} (want {})", priors.len(), value.len(), margin.len(), b * MAX_SEATS));
        }
        let (_, _, _, opt_off) = self.outstanding.remove(&id).unwrap();
        let grp = &self.sh.groups[g];
        let bases = std::mem::take(&mut grp.bases.lock().unwrap()[key as usize]);
        grp.evals.write().unwrap()[key as usize] = Some(Eval::Nn { opt_off, priors: priors.to_vec(), value: value.to_vec(), margin: margin.to_vec(), bases });
        self.answered(g)
    }

    /// Answers of external batch `id`: one payload per request, in order.
    pub fn submit_external(&mut self, id: u64, payloads: Vec<String>) -> Result<(), String> {
        let Some((g, key, b)) = self.outstanding.get(&id).map(|o| (o.0, o.1, o.2)) else {
            return Err(format!("unknown or already answered batch id {id}"));
        };
        if self.sh.cfg.keys[key as usize].1 != KeyKind::External {
            return Err(format!("batch {id} is a network batch: use submit"));
        }
        if payloads.len() != b {
            return Err(format!("external answers: {} payloads for {b} requests", payloads.len()));
        }
        self.outstanding.remove(&id);
        let grp = &self.sh.groups[g];
        let bases = std::mem::take(&mut grp.bases.lock().unwrap()[key as usize]);
        grp.evals.write().unwrap()[key as usize] = Some(Eval::Ext { payloads, bases });
        self.answered(g)
    }

    pub fn take_results(&mut self) -> Vec<GameResult> {
        std::mem::take(&mut self.results)
    }

    pub fn take_samples(&mut self) -> Vec<Record> {
        self.emitted += self.samples.len() as u64;
        std::mem::take(&mut self.samples)
    }

    pub fn stats(&self) -> DriverStats {
        let c = &self.sh.counters;
        DriverStats {
            sims: c.sims.load(Ordering::Relaxed),
            leaves: c.leaves.load(Ordering::Relaxed),
            ext_requests: c.ext_requests.load(Ordering::Relaxed),
            batches: self.batches,
            busy_s: c.busy_ns.load(Ordering::Relaxed) as f64 * 1e-9,
            assemble_s: c.assemble_ns.load(Ordering::Relaxed) as f64 * 1e-9,
            wait_s: self.wait_ns as f64 * 1e-9,
            wall_s: self.t0.elapsed().as_secs_f64(),
            threads: self.sh.cfg.threads,
            groups: self.sh.cfg.groups,
            games_started: self.next_gid as u64,
            games_finished: self.finished_games,
            phase_games: self.phase_games,
            quota: self.quota,
            dropped_leaves: self.dropped_leaves,
            restart_sims: c.restart_sims.load(Ordering::Relaxed),
            restarts: c.restarts.load(Ordering::Relaxed),
            recorded: c.recorded.load(Ordering::Relaxed),
            emitted: self.emitted,
            epoch: self.sh.epoch.load(Ordering::Relaxed),
            net: self.sh.net.load(Ordering::Relaxed),
        }
    }

    pub fn in_flight(&self) -> InFlight {
        let mut r = InFlight::default();
        let mut oldest = u64::MAX;
        for grp in &self.sh.groups {
            for ch in &grp.chunks {
                let ch = ch.lock().unwrap();
                for s in &ch.slots {
                    let Some(g) = s.game.as_ref() else { continue };
                    r.games += 1;
                    oldest = oldest.min(s.gid as u64);
                    r.max_steps = r.max_steps.max(s.steps);
                    r.max_rounds = r.max_rounds.max(g.rounds());
                }
            }
        }
        if r.games > 0 {
            r.oldest_age_games = (self.next_gid as u64).saturating_sub(oldest);
        }
        r
    }
}

impl<G: SearchGame> Drop for Driver<G> {
    fn drop(&mut self) {
        self.sh.stop.store(true, Ordering::SeqCst);
        self.sh.queue_cv.notify_all();
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}

/// Training sample arrays (docs/search.md 2.4) of `records`.
pub struct SampleArrays {
    pub n: usize,
    pub f: usize,
    pub x: Vec<f32>,
    pub player: Vec<i32>,
    pub n_players: Vec<i32>,
    pub opt_off: Vec<i64>,
    pub opt_action: Vec<i32>,
    pub pi: Vec<f32>,
    pub z: Vec<f32>,
    pub q: Vec<f32>,
    pub sc: Vec<f32>,
    pub net: Vec<u32>,
}

pub fn sample_arrays(recs: &[Record], f: usize) -> SampleArrays {
    let n = recs.len();
    let mut s = SampleArrays {
        n,
        f,
        x: Vec::with_capacity(n * f),
        player: Vec::with_capacity(n),
        n_players: Vec::with_capacity(n),
        opt_off: vec![0],
        opt_action: Vec::new(),
        pi: Vec::new(),
        z: Vec::with_capacity(n * MAX_SEATS),
        q: Vec::with_capacity(n * MAX_SEATS),
        sc: Vec::with_capacity(n * MAX_SEATS),
        net: Vec::with_capacity(n),
    };
    for r in recs {
        debug_assert_eq!(r.x.len(), f);
        s.x.extend_from_slice(&r.x);
        s.player.push(r.player as i32);
        s.n_players.push(r.n_players as i32);
        s.opt_action.extend(r.opts.iter().map(|&a| a as i32));
        s.opt_off.push(s.opt_action.len() as i64);
        s.pi.extend_from_slice(&r.pi);
        s.z.extend_from_slice(&r.z);
        s.q.extend_from_slice(&r.q);
        s.sc.extend_from_slice(&r.sc);
        s.net.push(r.net);
    }
    s
}

#[cfg(test)]
mod tests;
