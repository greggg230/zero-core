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
//! round-robin order, so a run is reproducible from its seed and the evaluator outputs; a
//! `queue_only` driver serves whichever group is ready first, so a slow game holds up its own
//! group only) and
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
//!
//! **Deferred results** (`SearchGame::deferred`, docs/search.md 2.8): a game that ends at a
//! deferred terminal is **parked** with its training records (features as float16) and root
//! values; its slot starts the next game. `take_deferred()` hands out `(gid, payload)` once per
//! parked game, `resolve(gid, results)` gives the game its outcome and emits its result row and
//! samples exactly as a finished game's (`z` from `result()`, `q` from the root values, `sc`
//! from `margin()`); `pending()` counts parked games. Rolling quotas count finished and
//! resolved games, never parked ones.
//!
//! **Queued setups** (docs/search.md 2.9): `enqueue(tag, spec)` makes the next game a free slot
//! starts use that spec instead of the driver's (the seats stay the driver's); with
//! `DriverCfg::queue_only` (rolling drivers) slots only start queued games and stay idle while
//! the queue is empty (`next_batch` returns None when nothing is left to play; `enqueue` wakes
//! the idle groups). A game started from the queue carries its tag into its result row. The
//! groups share the queued setups (`queue_share`): each starts its part of everything in flight
//! or queued, never leaving a free slot idle while the other groups are full.
use std::collections::{HashMap, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::thread::JoinHandle;
use std::time::Instant;

use crate::game::{make_bot, Bot, SearchGame, MAX_SEATS};
use crate::rng::{split64, Rng};
use crate::tree::Tree;

mod worker;
use worker::worker;

const NO_GAME: u32 = u32::MAX;

// The driver's locks are taken poison-tolerant: a panic is recorded as the driver's error
// (`Shared::fail`) and the group it hit must still be wound up (`finish_assembly`), so a
// poisoned lock must neither kill a worker thread nor the caller.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn read<T>(m: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    m.read().unwrap_or_else(PoisonError::into_inner)
}

fn write<T>(m: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    m.write().unwrap_or_else(PoisonError::into_inner)
}

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

/// How a network seat chooses a move it does not sample (docs/search.md 3.3): every move of an
/// evaluation seat, and a self-play seat's after its first `temp_moves` searched decisions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Select {
    /// argmax (N, P0), the first maximum on exact ties: AlphaZero's rule, the default.
    Visits,
    /// argmax of the completed-Q improved policy's logits `ln max(P0, PRIOR_FLOOR) + (c_visit +
    /// max N) c_scale completedQ` (`tree::completed_q_logits`, the logits of the 'cq' training
    /// target; Gumbel AlphaZero's final-move rule) among the options with at least `min_visits`
    /// visits (0: every option, an unvisited one at v_mix; when no option has that many, the
    /// most visited ones), the first maximum on exact ties.
    Cq { c_visit: f64, c_scale: f64, min_visits: u32 },
}

impl Select {
    /// From the seat settings `select` ('visits' | 'cq'), `select_min_visits` (of 'cq' only) and
    /// the completed-Q transform `select_c_visit`, `select_c_scale` (absent: the seat's
    /// `cq_c_visit`, `cq_c_scale`, the 'cq' target's; the target never reads these two).
    pub fn parse(name: &str, min_visits: u32, c_visit: f64, c_scale: f64) -> Result<Select, String> {
        match name {
            "visits" if min_visits == 0 => Ok(Select::Visits),
            "visits" => Err(format!("select_min_visits {min_visits}: a guard of select 'cq' (select 'visits' has none)")),
            "cq" => {
                if !(c_visit.is_finite() && c_visit >= 0.0 && c_scale.is_finite() && c_scale >= 0.0) {
                    return Err(format!("select 'cq' needs a finite completed-Q transform c_visit >= 0, c_scale >= 0 (select_c_visit / select_c_scale, default cq_c_visit / cq_c_scale; got {c_visit}, {c_scale})"));
                }
                Ok(Select::Cq { c_visit, c_scale, min_visits })
            }
            _ => Err(format!("select {name:?}: 'visits' or 'cq'")),
        }
    }

    /// The chosen option of a searched decision from its root statistics: priors before noise
    /// `p0`, visits `n`, value sums `w` (the decider's utility), the network's utility `v` of
    /// the root for its decider. 0 for a decision without options. The driver's moves and
    /// `SearchTree.root_select` (the Python parity tests) both come from here.
    pub fn choose(&self, p0: &[f32], n: &[u32], w: &[f64], v: f64) -> usize {
        match *self {
            Select::Visits => argmax_visits(n, p0),
            Select::Cq { c_visit, c_scale, min_visits } => {
                if n.is_empty() {
                    return 0;
                }
                let lg = crate::tree::completed_q_logits(p0, n, w, v, c_visit, c_scale);
                let m = min_visits.min(n.iter().copied().max().unwrap_or(0));
                let mut best: Option<usize> = None;
                for k in 0..lg.len() {
                    if n[k] >= m && best.is_none_or(|b| lg[k] > lg[b]) {
                        best = Some(k);
                    }
                }
                best.unwrap_or(0)
            }
        }
    }
}

/// max by (N, P0), the first maximum on exact ties (`Select::Visits`); 0 without options.
pub fn argmax_visits(n: &[u32], p0: &[f32]) -> usize {
    let mut best = 0;
    for k in 1..n.len() {
        if (n[k], p0[k]) > (n[best], p0[best]) {
            best = k;
        }
    }
    best
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
    /// Simulations of a minor decision (`SearchGame::minor_decision`; never recorded). None:
    /// minor decisions are searched like the others.
    pub sims_minor: Option<u32>,
    pub chance_k: u32,
    pub margin_w: f64,
    pub redet: bool,
    /// Training target of the seat's recorded decisions (self-play, full searches).
    pub policy_target: PolicyTarget,
    /// How a move that is not sampled is chosen (`Select`; default `Select::Visits`).
    pub select: Select,
}

impl NnSeat {
    /// Evaluation seat defaults.
    pub fn eval(key: u16, sims: u32) -> NnSeat {
        NnSeat { key, sims, c_puct: 1.0, selfplay: false, alpha: 0.3, eps: 0.25, temp_moves: 30, sims_fast: None, p_full: 1.0, sims_minor: None, chance_k: 0, margin_w: 0.0, redet: true, policy_target: PolicyTarget::Visits, select: Select::Visits }
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
    /// Rolling mode: slots start only queued setups (`Driver::enqueue`), idle while the queue
    /// is empty.
    pub queue_only: bool,
    /// Log the first `move_log` real moves of every game (seat, action, legal actions) into
    /// its result row (`GameResult::moves`, a diagnostic: zero.h2h's `--move-log`); 0 = off
    /// (nothing changes).
    pub move_log: u32,
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
    /// The tag of the queued setup the game was started from (`Driver::enqueue`).
    pub tag: Option<String>,
    /// `DriverCfg::move_log`: the game's first real moves (seat, action, legal
    /// actions), the external seats' answers excluded; empty when off.
    pub moves: Vec<(u8, u16, Vec<u16>)>,
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
    /// Network batches: features [b * F], legal actions as segments of `opt_action`.
    pub x: Vec<f32>,
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

/// A queued game setup (`Driver::enqueue`): the spec of the next game a free slot starts.
struct Setup<G: SearchGame> {
    tag: String,
    spec: G::Spec,
}

/// f32 -> IEEE half precision bits, rounding to nearest even (numpy's `astype(float16)`); NaN
/// stays NaN, overflow becomes inf.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 | (man >> 13) as u16 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1u32 << shift) - 1);
        let mut r = m >> shift;
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1;
    }
    sign | r as u16
}

/// IEEE half precision bits -> f32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// A parked game's training record: the features as float16 (what the sample shards store
/// anyway: the round trip changes nothing written).
struct ParkedRecord {
    x: Vec<u16>,
    player: u8,
    n_players: u8,
    opts: Vec<u16>,
    pi: Vec<f32>,
    net: u32,
}

impl ParkedRecord {
    fn park(r: Record) -> ParkedRecord {
        ParkedRecord { x: r.x.iter().map(|&v| f32_to_f16(v)).collect(), player: r.player, n_players: r.n_players, opts: r.opts, pi: r.pi, net: r.net }
    }
    fn unpark(self) -> Record {
        Record { x: self.x.iter().map(|&h| f16_to_f32(h)).collect(), player: self.player, n_players: self.n_players, opts: self.opts, pi: self.pi, z: [0.0; MAX_SEATS], q: [0.0; MAX_SEATS], sc: [0.0; MAX_SEATS], net: self.net }
    }
    fn bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.x.len() * 2 + self.opts.len() * 2 + self.pi.len() * 4
    }
}

/// A game waiting at its deferred terminal for `Driver::resolve`.
struct Parked<G: SearchGame> {
    gid: u32,
    tag: Option<String>,
    game: G,
    n: usize,
    records: Vec<ParkedRecord>,
    rq: Vec<Option<[f64; MAX_SEATS]>>,
    steps: u64,
    decisions: u32,
    ext_moves: u32,
    payload: String,
    /// Memory held by the records (`DriverStats::parked_bytes`).
    bytes: u64,
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
    /// The queued setup of the game `start` begins.
    setup: Option<Setup<G>>,
    /// The tag of the game's queued setup.
    tag: Option<String>,
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
    /// The current decision is minor (searched with `sims_minor`).
    minor: bool,
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
    /// The logged real moves (`DriverCfg::move_log`).
    moves: Vec<(u8, u16, Vec<u16>)>,
}

impl<G: SearchGame> Slot<G> {
    fn empty() -> Slot<G> {
        Slot {
            gid: NO_GAME,
            start: None,
            setup: None,
            tag: None,
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
            minor: false,
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
            moves: Vec::new(),
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
    parked: Vec<Parked<G>>,
}

struct GroupOut<G: SearchGame> {
    batches: Vec<Batch>,
    results: Vec<GameResult>,
    samples: Vec<Record>,
    parked: Vec<Parked<G>>,
}

impl<G: SearchGame> Default for GroupOut<G> {
    fn default() -> Self {
        GroupOut { batches: Vec::new(), results: Vec::new(), samples: Vec::new(), parked: Vec::new() }
    }
}

/// Raw destination of one network batch (its Vecs' heap buffers), written by the group's
/// fill jobs, each its own rows / options.
#[derive(Clone, Copy)]
struct Dest {
    x: *mut f32,
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
    out: Mutex<Option<GroupOut<G>>>,
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
    minor: AtomicU64,
    /// The longest a game's step took (one slot's advance between two of its leaves: its search
    /// and moves, ns), that game's gid, and the steps over SLOW_STEP_NS.
    step_max_ns: AtomicU64,
    step_max_gid: AtomicU64,
    slow_steps: AtomicU64,
}

/// A game's step longer than this counts as slow (`DriverStats::slow_steps`): an engine's slow
/// path holds up every game of its group (and the caller) that long.
pub const SLOW_STEP_NS: u64 = 10_000_000_000;

struct Shared<G: SearchGame> {
    cfg: DriverCfg,
    spec: G::Spec,
    groups: Vec<Group<G>>,
    queue: Mutex<VecDeque<(u32, u32)>>,
    queue_cv: Condvar,
    done: Mutex<Vec<bool>>,
    done_cv: Condvar,
    error: Mutex<Option<String>>,
    /// `error` is set (read under the `done` lock by `next_batch`'s wait).
    failed: AtomicBool,
    stop: AtomicBool,
    counters: Counters,
    epoch: AtomicU32,
    net: AtomicU32,
}

impl<G: SearchGame> Shared<G> {
    /// Record the driver's error (the first one sticks) and wake a caller waiting in
    /// `next_batch`: it returns the error at once, whatever the state of the groups.
    fn fail(&self, msg: String) {
        {
            let mut e = lock(&self.error);
            if e.is_none() {
                *e = Some(msg);
            }
        }
        self.failed.store(true, Ordering::SeqCst);
        // the waiter checks `failed` under the `done` lock: taking it here orders this notify
        // after its check (no lost wakeup)
        drop(lock(&self.done));
        self.done_cv.notify_all();
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
    /// Minor decisions decided by network seats with `sims_minor`.
    pub minor: u64,
    pub emitted: u64,
    pub epoch: u32,
    pub net: u32,
    /// Games parked at a deferred terminal now, the memory their records hold, games parked
    /// so far, games resolved so far, setups queued now.
    pub parked: u64,
    pub parked_bytes: u64,
    pub deferred: u64,
    pub resolved: u64,
    pub queued: u64,
    /// The longest step of a game so far (s) and that game's gid; steps over `SLOW_STEP_NS`.
    pub step_max_s: f64,
    pub step_max_gid: u64,
    pub slow_steps: u64,
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
    /// Games at a deferred terminal waiting for `resolve`, by gid.
    parked: HashMap<u32, Parked<G>>,
    parked_bytes: u64,
    /// (gid, payload) of games parked since the last `take_deferred`.
    deferred_out: Vec<(u32, String)>,
    deferred: u64,
    resolved: u64,
    queue: VecDeque<Setup<G>>,
    /// Per group: its slots, and its slots with a game as of its last dispatch (`queue_share`);
    /// per group and chunk: the latter (`in_flight` of a chunk being searched).
    gslots: Vec<usize>,
    gbusy: Vec<usize>,
    cbusy: Vec<Vec<usize>>,
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
                    if n.sims == 0 || n.sims_fast == Some(0) || n.sims_minor == Some(0) {
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
        if cfg.queue_only && !cfg.rolling {
            return Err("queue_only needs a rolling driver".into());
        }
        let conc = if cfg.rolling { cfg.concurrency.max(1) } else { cfg.concurrency.max(1).min(cfg.games.len().max(1)) };
        let groups = cfg.groups.max(1).min(conc);
        let threads = cfg.threads.max(1);
        let nkeys = cfg.keys.len();
        let mut gv = Vec::new();
        for g in 0..groups {
            let ns = (g + 1) * conc / groups - g * conc / groups;
            let nchunks = ns.min(threads * 4).max(1);
            let mut chunks = Vec::new();
            for c in 0..nchunks {
                let a = c * ns / nchunks;
                let b = (c + 1) * ns / nchunks;
                chunks.push(Mutex::new(Chunk { slots: (a..b).map(|_| Slot::empty()).collect(), bufs: (0..nkeys).map(|_| LeafBuf::default()).collect(), results: Vec::new(), samples: Vec::new(), parked: Vec::new() }));
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
            failed: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            counters: Counters::default(),
            epoch: AtomicU32::new(0),
            net: AtomicU32::new(net),
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
            parked: HashMap::new(),
            parked_bytes: 0,
            deferred_out: Vec::new(),
            deferred: 0,
            resolved: 0,
            queue: VecDeque::new(),
            gslots: Vec::new(),
            gbusy: vec![0; groups],
            cbusy: Vec::new(),
        };
        d.cbusy = d.sh.groups.iter().map(|grp| vec![0; grp.chunks.len()]).collect();
        d.gslots = d.sh.groups.iter().map(|grp| grp.chunks.iter().map(|c| lock(c).slots.len()).sum()).collect();
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
        let queue_only = self.sh.cfg.queue_only;
        // queue_only: this group's share of the queued setups (`queue_share`), so that the groups'
        // batches pipeline; any other driver fills every free slot
        let mut take = if queue_only { self.queue_share(g) } else { usize::MAX };
        let mut busy = false;
        let mut nbusy = 0;
        for (c, ch) in grp.chunks.iter().enumerate() {
            let mut ch = lock(ch);
            let mut cb = 0;
            for s in ch.slots.iter_mut() {
                if s.game.is_none() && s.start.is_none() && self.next_gid < n_games && !(queue_only && (self.queue.is_empty() || take == 0)) {
                    s.start = Some(self.next_gid);
                    s.setup = self.queue.pop_front();
                    self.next_gid += 1;
                    take = take.saturating_sub(1);
                }
                let b = s.game.is_some() || s.start.is_some();
                busy |= b;
                cb += b as usize;
            }
            self.cbusy[g][c] = cb;
            nbusy += cb;
        }
        self.gbusy[g] = nbusy;
        if !busy {
            self.gstate[g] = GState::Idle;
            return;
        }
        self.gstate[g] = GState::Working;
        lock(&self.sh.done)[g] = false;
        grp.remaining.store(grp.chunks.len(), Ordering::SeqCst);
        let mut q = lock(&self.sh.queue);
        for c in 0..grp.chunks.len() {
            q.push_back((g as u32, c as u32));
        }
        drop(q);
        self.sh.queue_cv.notify_all();
    }

    /// How many queued setups group `g` starts now (a `queue_only` driver): its share of every
    /// game in flight or queued (the groups' games as of their last dispatch), but never fewer than
    /// the queue holds beyond the other groups' free slots - so the groups stay balanced (each has
    /// games to search while the caller evaluates another's batch) and no free slot idles while
    /// the others are full. Without it the first group re-dispatched took the whole queue: an
    /// evaluation's 384 matches started 383 / 1, one group's search and inference in turn.
    fn queue_share(&self, g: usize) -> usize {
        let grp = &self.sh.groups[g];
        let mut busy = 0;
        for ch in &grp.chunks {
            busy += lock(ch).slots.iter().filter(|s| s.game.is_some() || s.start.is_some()).count();
        }
        let ng = self.gstate.len();
        let others_busy: usize = (0..ng).filter(|&h| h != g).map(|h| self.gbusy[h]).sum();
        let others_room: usize = (0..ng).filter(|&h| h != g).map(|h| self.gslots[h].saturating_sub(self.gbusy[h])).sum();
        let q = self.queue.len();
        let target = (others_busy + busy + q).div_ceil(ng);
        target.saturating_sub(busy).max(q.saturating_sub(others_room))
    }

    fn check_error(&self) -> Result<(), String> {
        match &*lock(&self.sh.error) {
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

    /// Every game finished (or resolved) and every batch answered (never for a rolling driver).
    pub fn finished(&self) -> bool {
        self.ready.is_empty() && self.outstanding.is_empty() && self.gstate.iter().all(|s| *s == GState::Idle) && self.parked.is_empty()
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
            let order: Vec<usize> = (0..ng).map(|k| (self.cursor + k) % ng).filter(|&g| self.gstate[g] == GState::Working).collect();
            if order.is_empty() {
                if !block {
                    return Ok(None);
                }
                return Err("submit the outstanding batches before asking for more".into());
            }
            // the group served next: the next one in round-robin order (a run reproducible from
            // its seed); a queue_only driver serves the first of its working groups that is ready,
            // so a game whose step takes long (an engine's slow path) holds up its own group only,
            // never every game of the driver (docs/search.md 2.3)
            let any = self.sh.cfg.queue_only;
            let t = Instant::now();
            let g = {
                let failed = || self.sh.failed.load(Ordering::SeqCst);
                let mut done = lock(&self.sh.done);
                loop {
                    let ready = if any { order.iter().copied().find(|&g| done[g]) } else { Some(order[0]).filter(|&g| done[g]) };
                    if let Some(g) = ready {
                        break g;
                    }
                    if failed() {
                        break order[0];
                    }
                    if !block {
                        return Ok(None);
                    }
                    done = self.sh.done_cv.wait(done).unwrap_or_else(PoisonError::into_inner);
                }
            };
            self.wait_ns += t.elapsed().as_nanos() as u64;
            self.check_error()?;
            let out = lock(&self.sh.groups[g].out).take().unwrap_or_default();
            let nfin = out.results.len() as u64;
            self.finished_games += nfin;
            self.phase_games += nfin;
            self.results.extend(out.results);
            self.samples.extend(out.samples);
            for mut p in out.parked {
                self.deferred += 1;
                self.parked_bytes += p.bytes;
                self.deferred_out.push((p.gid, std::mem::take(&mut p.payload)));
                self.parked.insert(p.gid, p);
            }
            self.cursor = (g + 1) % ng;
            if self.pausing() {
                // the phase is over: this group's leaves are not evaluated (nothing was backed
                // up yet, the next phase walks again)
                for b in &out.batches {
                    self.dropped_leaves += b.b as u64;
                }
                if !out.batches.is_empty() {
                    for ch in &self.sh.groups[g].chunks {
                        for s in lock(ch).slots.iter_mut() {
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
        let bases = std::mem::take(&mut lock(&grp.bases)[key as usize]);
        write(&grp.evals)[key as usize] = Some(Eval::Nn { opt_off, priors: priors.to_vec(), value: value.to_vec(), margin: margin.to_vec(), bases });
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
        let bases = std::mem::take(&mut lock(&grp.bases)[key as usize]);
        write(&grp.evals)[key as usize] = Some(Eval::Ext { payloads, bases });
        self.answered(g)
    }

    pub fn take_results(&mut self) -> Vec<GameResult> {
        std::mem::take(&mut self.results)
    }

    /// Games parked at a deferred terminal since the last call: (gid, `deferred_payload()`).
    pub fn take_deferred(&mut self) -> Vec<(u32, String)> {
        std::mem::take(&mut self.deferred_out)
    }

    /// Games parked at a deferred terminal, waiting for `resolve`.
    pub fn pending(&self) -> usize {
        self.parked.len()
    }

    /// A parked game (tests, analysis).
    pub fn parked_game(&self, gid: u32) -> Option<&G> {
        self.parked.get(&gid).map(|p| &p.game)
    }

    /// The outcome of parked game `gid` (`SearchGame::resolve(results)`, JSON): its result row
    /// and training samples are emitted as a finished game's, and it counts toward the rolling
    /// quota. An unknown game, a result the game refuses or a game still deferred afterwards is
    /// an error (the game stays parked when it refused the result).
    pub fn resolve(&mut self, gid: u32, results: &str) -> Result<(), String> {
        self.check_error()?;
        let seed = self.sh.cfg.seed;
        let Some(mut p) = self.parked.remove(&gid) else {
            return Err(format!("resolve: game {gid} is not waiting for a deferred result ({} parked)", self.parked.len()));
        };
        let short = |s: &str| if s.len() > 300 { format!("{}...", &s[..s.char_indices().nth(300).map_or(s.len(), |(i, _)| i)]) } else { s.to_string() };
        match catch_unwind(AssertUnwindSafe(|| p.game.resolve(results))) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                let msg = format!("game {gid} (driver seed {seed}): resolve refused {}: {e}", short(results));
                self.parked.insert(gid, p);
                return Err(msg);
            }
            Err(pn) => {
                self.parked_bytes -= p.bytes;
                return Err(format!("game {gid} (driver seed {seed}): resolve panicked: {}", worker::panic_msg(pn)));
            }
        }
        self.parked_bytes -= p.bytes;
        if p.game.deferred() || p.game.to_move().is_some() {
            return Err(format!("game {gid} (driver seed {seed}): still deferred (or not over) after resolve({})", short(results)));
        }
        let done = worker::Done { gid, tag: p.tag.take(), n: p.n, steps: p.steps, decisions: p.decisions, ext_moves: p.ext_moves, moves: Vec::new() };
        let records: Vec<Record> = p.records.into_iter().map(ParkedRecord::unpark).collect();
        let cfg = &self.sh.cfg;
        let mut samples = Vec::new();
        let row = catch_unwind(AssertUnwindSafe(|| worker::emit(cfg.record, &p.game, true, done, records, p.rq, &mut samples)))
            .map_err(|pn| format!("game {gid} (driver seed {seed}): its outcome after resolve panicked: {}", worker::panic_msg(pn)))?;
        self.results.push(row);
        self.samples.append(&mut samples);
        self.finished_games += 1;
        self.phase_games += 1;
        self.resolved += 1;
        Ok(())
    }

    /// Queue a game setup: the next game a free slot starts uses `spec` (JSON, the game's
    /// `parse_spec`; an error here) instead of the driver's, and its result row carries `tag`.
    /// Idle groups (a `queue_only` driver whose queue ran dry) start again unless the phase
    /// ended.
    pub fn enqueue(&mut self, tag: &str, spec: &str) -> Result<(), String> {
        self.check_error()?;
        let s = G::parse_spec(spec).map_err(|e| format!("enqueue {tag:?}: {e}"))?;
        self.queue.push_back(Setup { tag: tag.to_string(), spec: s });
        if !self.pausing() {
            for g in 0..self.gstate.len() {
                if self.gstate[g] == GState::Idle {
                    self.dispatch(g);
                }
            }
        }
        Ok(())
    }

    /// Setups queued and not started yet.
    pub fn queued(&self) -> usize {
        self.queue.len()
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
            minor: c.minor.load(Ordering::Relaxed),
            emitted: self.emitted,
            epoch: self.sh.epoch.load(Ordering::Relaxed),
            net: self.sh.net.load(Ordering::Relaxed),
            parked: self.parked.len() as u64,
            parked_bytes: self.parked_bytes,
            deferred: self.deferred,
            resolved: self.resolved,
            queued: self.queue.len() as u64,
            step_max_s: c.step_max_ns.load(Ordering::Relaxed) as f64 * 1e-9,
            step_max_gid: c.step_max_gid.load(Ordering::Relaxed),
            slow_steps: c.slow_steps.load(Ordering::Relaxed),
        }
    }

    /// The games in progress. Never waits for a search thread: a chunk being searched right now
    /// counts its games as of its group's last dispatch and is left out of the other fields (a
    /// game whose step takes minutes must not hold the caller - and with it Python's GIL - that
    /// long).
    pub fn in_flight(&self) -> InFlight {
        let mut r = InFlight::default();
        let mut oldest = u64::MAX;
        for (g, grp) in self.sh.groups.iter().enumerate() {
            for (c, ch) in grp.chunks.iter().enumerate() {
                let ch = match ch.try_lock() {
                    Ok(ch) => ch,
                    Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
                    Err(std::sync::TryLockError::WouldBlock) => {
                        r.games += self.cbusy[g][c] as u64;
                        continue;
                    }
                };
                for s in &ch.slots {
                    let Some(game) = s.game.as_ref() else { continue };
                    r.games += 1;
                    oldest = oldest.min(s.gid as u64);
                    r.max_steps = r.max_steps.max(s.steps);
                    r.max_rounds = r.max_rounds.max(game.rounds());
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
