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
//! full searches (policy target: visits or completed Q, `PolicyTarget`), tree reuse, and the
//! opt-in root minimum visits of full searches (`NnSeat::root_min_visits`, `root_floor`,
//! docs/search.md 3.5), the opt-in root's belief (`NnSeat::belief_tilt`, docs/search.md 3.8: one
//! flagged `ROW_BELIEF` row per searched decision, whose auxiliary output `submit_aux` returns; the
//! search's worlds are then dealt by it).
//! docs/search.md 3.5), and the opt-in Gumbel root search (`NnSeat::root_search`, docs/search.md
//! 3.7: Gumbel noise and sequential halving at the root instead of PUCT, Dirichlet noise and
//! temperature; its move and root value from the tree's `gumbel_choose` / `gumbel_q`).
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

use crate::game::{make_bot, Belief, Bot, SearchGame, MAX_SEATS};
use crate::lookahead::{Look, LookCfg};
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

/// How a network seat's searches choose their root walks and their move (docs/search.md 3.7, seat
/// key `root_search`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RootSearch {
    /// PUCT at the root, root Dirichlet noise on full self-play searches, temperature sampling for
    /// the first `temp_moves` decisions, then the seat's move rule (`select`): the default, the
    /// search of before bit for bit.
    Puct,
    /// Gumbel AlphaZero's root (Danihelka et al. 2022; `Tree::set_gumbel`): Gumbel noise of scale
    /// `scale` per root option (a self-play seat's: on its first `temp_moves` searched decisions
    /// only, later ones without; an evaluation seat's: on every decision), sequential halving over
    /// `min(m, options)` options, sigma's transform `c_visit`, `c_scale`; the move is the halving's
    /// winner (`Tree::gumbel_choose`: no Dirichlet noise, no temperature, `select` unused), the root
    /// value `Tree::gumbel_q`. Every search of the seat - full, fast and minor - is one. PUCT below
    /// the root.
    Gumbel { m: u32, scale: f64, c_visit: f64, c_scale: f64 },
}

impl RootSearch {
    /// From the seat settings `root_search` ('puct' | 'gumbel'), `gumbel_m`, `gumbel_scale`,
    /// `gumbel_c_visit`, `gumbel_c_scale` (the last four: Gumbel's only).
    pub fn parse(name: &str, m: u32, scale: f64, c_visit: f64, c_scale: f64) -> Result<RootSearch, String> {
        match name {
            "puct" => Ok(RootSearch::Puct),
            "gumbel" => {
                let r = RootSearch::Gumbel { m, scale, c_visit, c_scale };
                r.check()?;
                Ok(r)
            }
            _ => Err(format!("root_search {name:?}: 'puct' or 'gumbel'")),
        }
    }

    fn check(&self) -> Result<(), String> {
        if let RootSearch::Gumbel { m, scale, c_visit, c_scale } = *self {
            if m < 1 {
                return Err(format!("gumbel_m {m}: at least 1 (the options sequential halving considers)"));
            }
            for (k, x) in [("gumbel_scale", scale), ("gumbel_c_visit", c_visit), ("gumbel_c_scale", c_scale)] {
                if !(x.is_finite() && x >= 0.0) {
                    return Err(format!("{k} {x}: a finite number >= 0"));
                }
            }
        }
        Ok(())
    }
}

/// Whose information a search leaf's VALUE comes from (docs/search.md 3.6, seat key
/// `leaf_value_view`). The priors always come from the leaf's decider's own view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafView {
    /// `encode(decider)` gives the leaf's priors and its value: one row (the default, the search
    /// of before bit for bit).
    Mover,
    /// A leaf decided by another seat than the tree's root decider is evaluated twice, in the same
    /// batch: its priors from `encode(decider)` and its value and margin from a value-only row
    /// `encode(viewer)` (the root decider's information: its own hidden cards as they are, the
    /// other seats' unknown; no options). A leaf of the root decider is one row as before.
    Root,
}

impl LeafView {
    /// From the seat setting `leaf_value_view` ('mover' | 'root').
    pub fn parse(name: &str) -> Result<LeafView, String> {
        match name {
            "mover" => Ok(LeafView::Mover),
            "root" => Ok(LeafView::Root),
            _ => Err(format!("leaf_value_view {name:?}: 'mover' or 'root'")),
        }
    }
}

/// The value target `q` of a value-only training sample (`NnSeat::view_samples`, docs/search.md
/// 3.6): its `z` is the game's result for the sample's observer either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewTarget {
    /// The root win estimate of the observer's own next search (full or fast, not minor) after the
    /// sample; `z` when it never searches again in the game. The observer only learns more until
    /// then, so its next search's value is an unbiased estimate of the outcome given what it knows
    /// at the sample (as a decider's sample's q is its own search's).
    NextQ,
    /// `q` = `z`: the outcome only.
    Z,
}

impl ViewTarget {
    /// From the seat setting `view_sample_target` ('next_q' | 'z').
    pub fn parse(name: &str) -> Result<ViewTarget, String> {
        match name {
            "next_q" => Ok(ViewTarget::NextQ),
            "z" => Ok(ViewTarget::Z),
            _ => Err(format!("view_sample_target {name:?}: 'next_q' or 'z'")),
        }
    }
}

/// Whether recorded decision `decision` (the game's searched decisions before it) of game `gid` of
/// a driver seeded `seed` also records the value-only samples of a seat that records a share
/// `share` of its decisions' (`NnSeat::view_samples`): a hash of (seed, gid, decision), so no
/// random stream changes (the Python reference's `play.view_pick` is the same arithmetic).
pub fn view_pick(seed: u64, gid: u32, decision: u32, share: f64) -> bool {
    if share >= 1.0 {
        return true;
    }
    if share <= 0.0 {
        return false;
    }
    let h = split64(seed ^ split64((((gid as u64) << 32) | decision as u64) ^ 0x7669_6577));
    ((h >> 11) as f64 * (1.0 / (1u64 << 53) as f64)) < share
}

/// The root minimum visits a search takes (docs/search.md 3.5): the seat's `k`
/// (`NnSeat::root_min_visits`) for a full search (`full`: a self-play seat's recorded searches,
/// every search of an evaluation seat; never a fast or minor one, like root noise) of `m` options
/// with `2 <= m <= max_actions` and the budget for it (`2 m k <= target`, the search's simulations:
/// the floor takes at most half of them); else 0 (the search of before, bit for bit).
pub fn root_floor(k: u32, max_actions: u32, full: bool, m: usize, target: u32) -> u32 {
    let m = m as u64;
    if k == 0 || !full || m < 2 || m > max_actions as u64 || 2 * m * k as u64 > target as u64 {
        return 0;
    }
    k
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
    /// Whether the seat's full searches become training records (with `DriverCfg::record` and
    /// `selfplay`; default true). False: a self-play seat that searches, samples and draws its
    /// playout cap exactly as a recorded one but records nothing - a league opponent, a past
    /// checkpoint playing the current network (zero's run setting `selfplay_league`, docs/az.md 14).
    pub record: bool,
    /// Root minimum visits (docs/search.md 3.5; 0 = off, the default): a full search whose root
    /// has 2..=`root_min_visits_max_actions` options gives each at least this many visits, when
    /// its budget allows (`root_floor`). The forced walks are left out of the visit counts the
    /// targets, temperature sampling and the 'visits' rule read, and out of the root value `q`;
    /// Q, the 'cq' target and the 'cq' rule use them.
    pub root_min_visits: u32,
    /// The largest root (number of options) the floor applies to (default 4).
    pub root_min_visits_max_actions: u32,
    /// Whose view a leaf's value comes from (docs/search.md 3.6; default `LeafView::Mover`).
    pub leaf_value_view: LeafView,
    /// Value-only samples (docs/search.md 3.6; 0 = off, the default): a recorded decision of this
    /// seat also records, with this probability (`view_pick`), one sample per other seat of the
    /// game that records too (a self-play network seat with `record`): `encode(other)` of the real
    /// state, in that seat's perspective, no options (no policy target), `q` per `view_target`.
    pub view_samples: f64,
    pub view_target: ViewTarget,
    /// The root's belief (docs/search.md 3.8; 0 = off, the default): a search that takes it asks
    /// the network for the root's auxiliary output once (one `ROW_BELIEF` row) and deals every
    /// simulation's world with `SearchGame::determinize_with` at this tilt (alpha > 0).
    pub belief_tilt: f64,
    /// The belief's clip R >= 1 (default 20): every weight of the deal in [1/R, R].
    pub belief_clip: f64,
    /// The share of a belief search's simulations dealt uniformly all the same (in [0, 1); default 0).
    pub belief_mix: f64,
    /// Whether fast (`sims_fast`) and minor (`sims_minor`) searches take the belief too (default
    /// false: full searches only - every search of an evaluation seat, a self-play seat's recorded
    /// ones).
    pub belief_fast: bool,
    /// The root search (docs/search.md 3.7; default `RootSearch::Puct`).
    pub root_search: RootSearch,
    /// The exhaustive lookahead (docs/search.md 3.9; None, the default: the seat searches with MCTS):
    /// every decision of the seat is decided by `lookahead::Look` with these settings, its rows in the
    /// driver's batches; the seat's MCTS settings serve its final fallback (`Last::Mcts`). An
    /// evaluation seat (no records).
    pub look: Option<Box<LookCfg>>,
    /// Late sampling (docs/search.md 3.10; 0 = off, the default: the search of before bit for bit): a
    /// self-play seat's searched decisions at or after its first `temp_moves` - full and fast searches,
    /// never minor ones - are sampled with probability proportional to N^(1 / temp_late) over the root
    /// visit counts (without forced walks), instead of always playing the rule's move (`select`).
    pub temp_late: f64,
    /// Late sampling's margin gate (0 = no gate, the default): with it only the options whose
    /// completed Q (`tree::completed_q`, the decider's utility the 'cq' rule reads) is at most this much
    /// below the rule's move's are eligible; the rule's move always is, an unvisited option never.
    pub temp_late_margin: f64,
}

impl NnSeat {
    /// Evaluation seat defaults.
    pub fn eval(key: u16, sims: u32) -> NnSeat {
        NnSeat { key, sims, c_puct: 1.0, selfplay: false, alpha: 0.3, eps: 0.25, temp_moves: 30, sims_fast: None, p_full: 1.0, sims_minor: None, chance_k: 0, margin_w: 0.0, redet: true, policy_target: PolicyTarget::Visits, select: Select::Visits, record: true, root_min_visits: 0, root_min_visits_max_actions: 4, leaf_value_view: LeafView::Mover, view_samples: 0.0, view_target: ViewTarget::NextQ, belief_tilt: 0.0, belief_clip: BELIEF_CLIP, belief_mix: 0.0, belief_fast: false, root_search: RootSearch::Puct, look: None, temp_late: 0.0, temp_late_margin: 0.0 }
    }

    /// The late-sampling settings' check (`Driver::new`; docs/search.md 3.10): both finite >= 0; a
    /// margin only with `temp_late` > 0 (a gate of nothing would be a silent no-op); `temp_late` only on a
    /// self-play seat (an evaluation seat never samples) and not with the Gumbel root search (its noise
    /// is its temperature: exclusive, as its root minimum visits are).
    pub fn check_temp_late(&self) -> Result<(), String> {
        let (t, m) = (self.temp_late, self.temp_late_margin);
        if !(t.is_finite() && t >= 0.0) {
            return Err(format!("temp_late {t}: a finite number >= 0 (0 = off: the rule's move after temp_moves)"));
        }
        if !(m.is_finite() && m >= 0.0) {
            return Err(format!("temp_late_margin {m}: a finite number >= 0 (0 = no gate)"));
        }
        if m > 0.0 && t == 0.0 {
            return Err(format!("temp_late_margin {m}: a gate of temp_late > 0 (temp_late is 0: nothing is sampled late)"));
        }
        if t > 0.0 && !self.selfplay {
            return Err(format!("temp_late {t}: a self-play seat's (an evaluation seat never samples its moves)"));
        }
        if t > 0.0 && matches!(self.root_search, RootSearch::Gumbel { .. }) {
            return Err(format!("temp_late {t} with root_search 'gumbel': exclusive (the Gumbel noise is its temperature; docs/search.md 3.7)"));
        }
        Ok(())
    }

    /// The belief settings' check (`Driver::new`; the bindings' messages): tilt finite >= 0, clip
    /// finite >= 1, mix in [0, 1) (1 would deal every world uniformly: the tilt a silent no-op).
    pub fn check_belief(tilt: f64, clip: f64, mix: f64) -> Result<(), String> {
        if !(tilt.is_finite() && tilt >= 0.0) {
            return Err(format!("belief_tilt {tilt}: a finite number >= 0 (0 = off)"));
        }
        if !(clip.is_finite() && clip >= 1.0) {
            return Err(format!("belief_clip {clip}: a finite number >= 1 (every weight in [1/clip, clip])"));
        }
        if !(mix.is_finite() && (0.0..1.0).contains(&mix)) {
            return Err(format!("belief_mix {mix}: a share in [0, 1) of the simulations dealt uniformly (1 would make belief_tilt a no-op: set it to 0 instead)"));
        }
        Ok(())
    }

    /// Whether this seat's search of a decision takes the root's belief (`full`: a full search;
    /// docs/search.md 3.8).
    pub fn takes_belief(&self, full: bool) -> bool {
        self.belief_tilt > 0.0 && (full || self.belief_fast)
    }
}

/// Late sampling's weights (docs/search.md 3.10) of a decision whose rule's move is option `rule`:
/// per root option `nv^(1 / temp)` (`nv`: the visit counts the sampling reads, forced walks left out)
/// for the eligible options, else 0. Without a gate (`margin` 0) every option is eligible; with one, the
/// rule's move and the visited options whose completed Q (`q`, `tree::completed_q`) is at most `margin`
/// below the rule's move's. An option without visits weighs 0 either way.
pub fn late_weights(nv: &[u32], q: &[f64], rule: usize, temp: f64, margin: f64) -> Vec<f64> {
    let e = 1.0 / temp;
    let qr = q.get(rule).copied().unwrap_or(0.0);
    nv.iter()
        .enumerate()
        .map(|(k, &c)| {
            let ok = k == rule || margin <= 0.0 || qr - q[k] <= margin;
            if ok && c > 0 {
                (c as f64).powf(e)
            } else {
                0.0
            }
        })
        .collect()
}

/// The option late sampling plays from its weights (`late_weights`) and a uniform draw `u` in [0, 1):
/// the first option whose cumulative weight passes `u` x the total (the last weighted one if rounding
/// leaves none); the rule's move `rule` when nothing weighs anything.
pub fn late_pick(w: &[f64], rule: usize, u: f64) -> usize {
    let tot: f64 = w.iter().sum();
    if !(tot > 0.0) {
        return rule;
    }
    let mut r = u * tot;
    let mut last = rule;
    for (k, &x) in w.iter().enumerate() {
        if x > 0.0 {
            last = k;
            r -= x;
            if r < 0.0 {
                return k;
            }
        }
    }
    last
}

/// The default belief clip R (`NnSeat::belief_clip`).
pub const BELIEF_CLIP: f64 = 20.0;

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
    /// Record the game's auxiliary targets (`SearchGame::aux_target`, the decider's at each
    /// recorded decision) with every sample (`Record::aux`, `take_samples()['aux']`; docs/search.md
    /// 1.2); the game must have some (`SearchGame::AUX`). Off: nothing changes.
    pub aux: bool,
    /// Paired games (docs/search.md 2.10, docs/az.md 20.3): games `2k` and `2k + 1` draw every random
    /// stream from the pair's id `k` instead of their own gid - the same deal, shuffles and chance
    /// stream (the game's seed), the same driver streams - so a match whose table swaps the seats
    /// between the two games of a pair (zero.h2h, the Elo matches) plays each deal once from each
    /// seat. Off (the default): every game's streams derive from its gid (nothing changes).
    pub paired: bool,
    /// Sparse network batches (docs/search.md 2.2): every row's features as its non-zero (index,
    /// value) pairs (`Batch::xs_off`, `xs_idx`, `xs_val`; `x` empty) instead of `NUM_FEATURES` floats -
    /// an MTG row is ~150 non-zeros of 37,620: the host's copies of the dense rows bound the
    /// lookahead's big batches (docs/search.md 3.9). The same features, the same games. Off (the
    /// default): dense rows (nothing changes).
    pub sparse: bool,
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
    /// `DriverCfg::move_log` with late sampling (docs/search.md 3.10): the game's first late-sampled
    /// decisions (as many as `move_log`), a diagnostic; empty otherwise.
    pub late: Vec<LateMove>,
}

/// A late-sampled decision (`GameResult::late`, docs/search.md 3.10): its index among the game's searched
/// decisions, the game's round then, a full search (else a fast one), the root's options and those late
/// sampling weighed (> 0), the rule's move and the move played (actions) and their completed Q (the
/// decider's utility).
#[derive(Clone, Debug, PartialEq)]
pub struct LateMove {
    pub decision: u32,
    pub round: u32,
    pub full: bool,
    pub options: u16,
    pub weighed: u16,
    pub rule: u16,
    pub played: u16,
    pub q_rule: f64,
    pub q_played: f64,
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
    /// The decider's auxiliary targets at this decision (`DriverCfg::aux`), else empty.
    pub aux: Vec<f32>,
}

/// The kind of a network batch row (`Batch::extra`, docs/search.md 2.2): a leaf's own row (or a
/// deferred terminal's).
pub const ROW_LEAF: u8 = 0;
/// An extra row of the leaf row before it: its tree viewer's value-only row (no options;
/// `LeafView::Root`, docs/search.md 3.6). Extra rows are flagged by kind so that other features can
/// add their own (a kind > 0 always belongs to the closest leaf row before it, of the same game).
pub const ROW_VALUE: u8 = 1;
/// A search's root belief row (docs/search.md 3.8): `encode(viewer)` of the root (the live game's
/// decision, the tree viewer's own view), no options, `player` the viewer. The driver reads only
/// its auxiliary output (`Driver::submit_aux`: the network's predictions of the game's `AUX`
/// targets); its value and margin are ignored. It follows the root's own leaf row when the root
/// was fresh (the decision's first leaf), else it is the game's only row of the batch (a reused
/// root: the search waits for it) - the one kind that may stand without a leaf row before it.
pub const ROW_BELIEF: u8 = 2;

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
    /// Network batches: the kind of every row (`ROW_LEAF`, or an extra row's kind: `ROW_VALUE`,
    /// `ROW_BELIEF`).
    pub extra: Vec<u8>,
    /// Sparse network batches (`DriverCfg::sparse`; else empty): row b's non-zero features are
    /// `xs_idx[xs_off[b]..xs_off[b + 1]]` (feature indexes, ascending) with values `xs_val[..]`.
    pub xs_off: Vec<i64>,
    pub xs_idx: Vec<u32>,
    pub xs_val: Vec<f32>,
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
    extra: Vec<u8>,
    states: Vec<String>,
    /// Sparse rows (`DriverCfg::sparse`): per row its non-zeros' count, their indexes and values; a
    /// row is encoded into `scratch` (zeroed) and sealed into these (`seal`).
    sparse: bool,
    xs_nnz: Vec<u32>,
    xs_idx: Vec<u32>,
    xs_val: Vec<f32>,
    scratch: Vec<f32>,
}

impl LeafBuf {
    fn clear(&mut self) {
        self.x.clear();
        self.player.clear();
        self.n_players.clear();
        self.gid.clear();
        self.nopt.clear();
        self.opt_action.clear();
        self.extra.clear();
        self.states.clear();
        self.xs_nnz.clear();
        self.xs_idx.clear();
        self.xs_val.clear();
    }
    fn len(&self) -> usize {
        self.player.len()
    }
    /// The features of a new row, zeroed, for the caller to encode into: a slice of `x` (dense), or the
    /// scratch row (sparse: `seal` it afterwards).
    fn row_x(&mut self, f: usize) -> &mut [f32] {
        if self.sparse {
            if self.scratch.len() != f {
                self.scratch.clear();
                self.scratch.resize(f, 0.0);
            }
            &mut self.scratch
        } else {
            let off = self.x.len();
            self.x.resize(off + f, 0.0);
            &mut self.x[off..]
        }
    }
    /// A sparse buffer's scratch row into its sparse arrays (its non-zero floats, -0.0 included; the
    /// scratch zeroed again); nothing for a dense one.
    fn seal(&mut self) {
        if !self.sparse {
            return;
        }
        let mut k = 0u32;
        for (i, v) in self.scratch.iter_mut().enumerate() {
            if v.to_bits() != 0 {
                self.xs_idx.push(i as u32);
                self.xs_val.push(*v);
                *v = 0.0;
                k += 1;
            }
        }
        self.xs_nnz.push(k);
    }
    /// Drop the open row's features (dense: the last row of `x`; sparse: the scratch row, zeroed).
    fn unrow(&mut self, f: usize) {
        if self.sparse {
            self.scratch.iter_mut().for_each(|v| *v = 0.0);
        } else {
            self.x.truncate(self.x.len() - f);
        }
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
    aux: Vec<f32>,
}

impl ParkedRecord {
    fn park(r: Record) -> ParkedRecord {
        ParkedRecord { x: r.x.iter().map(|&v| f32_to_f16(v)).collect(), player: r.player, n_players: r.n_players, opts: r.opts, pi: r.pi, net: r.net, aux: r.aux }
    }
    fn unpark(self) -> Record {
        Record { x: self.x.iter().map(|&h| f16_to_f32(h)).collect(), player: self.player, n_players: self.n_players, opts: self.opts, pi: self.pi, z: [0.0; MAX_SEATS], q: [0.0; MAX_SEATS], sc: [0.0; MAX_SEATS], net: self.net, aux: self.aux }
    }
    fn bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.x.len() * 2 + self.opts.len() * 2 + self.pi.len() * 4 + self.aux.len() * 4
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
    /// `aux`: the `ROW_BELIEF` rows' auxiliary outputs in row order (`aux_width()` each), `aux_row`
    /// per batch row its index there (`NO_AUX`: not a belief row; empty: no belief rows).
    Nn { opt_off: Vec<i32>, priors: Vec<f32>, value: Vec<f32>, margin: Vec<f32>, bases: Vec<u32>, aux: Vec<f32>, aux_row: Vec<u32> },
    Ext { payloads: Vec<String>, bases: Vec<u32> },
}

/// `Eval::Nn::aux_row` of a row that is not a belief row.
const NO_AUX: u32 = u32::MAX;

enum Pending {
    /// `vrow`: the value comes from the next row, a value-only row of the tree's viewer
    /// (`LeafView::Root` at another seat's leaf). `brow`: the row after it (and after the value
    /// row) is the search's root belief row (`ROW_BELIEF`, with a fresh root's first leaf).
    Leaf { key: u16, local: u32, seat: usize, node: u32, vrow: bool, brow: bool },
    /// A reused root's belief row alone: the search waits for it (docs/search.md 3.8).
    Belief { key: u16, local: u32, seat: usize },
    /// A lookahead's round of rows from `local` (docs/search.md 3.9).
    Look { key: u16, local: u32 },
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
    /// The current search takes the root's belief (`NnSeat::takes_belief`, docs/search.md 3.8).
    belief: bool,
    decisions: u32,
    ext_moves: u32,
    steps: u64,
    records: Vec<Record>,
    rq: Vec<Option<[f64; MAX_SEATS]>>,
    /// Value-only records waiting for their observer's next search (`ViewTarget::NextQ`):
    /// (record index, observer seat).
    view_wait: Vec<(u32, u8)>,
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
    /// The logged late-sampled decisions (`DriverCfg::move_log`, `GameResult::late`).
    late: Vec<LateMove>,
    /// The current decision's Gumbel noise per root option (`RootSearch::Gumbel`; empty: none),
    /// kept for a restarted search of the same decision.
    gnoise: Vec<f64>,
    /// The lookahead of the current decision (a lookahead seat, docs/search.md 3.9), and whether the
    /// decision fell back to the seat's MCTS search (every stage over its budget).
    look: Option<Box<Look<G>>>,
    look_mcts: bool,
    /// The guarded player (`LookCfg::guard`): the lookahead's mean per root option and the margin, for
    /// the MCTS search's decision of the same decision.
    look_guard: Option<(Vec<(u16, f64)>, f64)>,
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
            belief: false,
            decisions: 0,
            ext_moves: 0,
            steps: 0,
            records: Vec::new(),
            rq: Vec::new(),
            view_wait: Vec::new(),
            pending: None,
            ext_payload: None,
            path: Vec::new(),
            legal: Vec::new(),
            legal_ok: false,
            epoch: 0,
            net: 0,
            moves: Vec::new(),
            late: Vec::new(),
            gnoise: Vec::new(),
            look: None,
            look_mcts: false,
            look_guard: None,
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
    extra: *mut u8,
    /// Sparse batches: their non-zeros' offsets, indexes, values.
    xs_off: *mut i64,
    xs_idx: *mut u32,
    xs_val: *mut f32,
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
    /// Sparse batches: every chunk's first non-zero in the batch.
    nz_base: Vec<usize>,
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
    /// Value-only rows sent with leaves (`LeafView::Root`) and value-only samples recorded.
    view_rows: AtomicU64,
    view_recorded: AtomicU64,
    /// Root belief rows sent (`ROW_BELIEF`, docs/search.md 3.8).
    belief_rows: AtomicU64,
    /// The longest a game's step took (one slot's advance between two of its leaves: its search
    /// and moves, ns), that game's gid, and the steps over SLOW_STEP_NS.
    step_max_ns: AtomicU64,
    step_max_gid: AtomicU64,
    slow_steps: AtomicU64,
    /// The lookahead seats' decisions and their counters (docs/search.md 3.9; `LookTotals`).
    look: Mutex<LookTotals>,
    /// Late sampling's counters (docs/search.md 3.10; `LateTotals`).
    late: Mutex<LateTotals>,
}

/// Late sampling's counters (`DriverStats::late`, docs/search.md 3.10), full and fast searches apart:
/// the late-sampled decisions, those where one option alone weighed anything (no choice: the rule's
/// move), the deviations (the move played is not the rule's) and their summed completed-Q cost (the
/// rule's move's Q - the played move's, the decider's utility).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LateTotals {
    pub decisions_full: u64,
    pub decisions_fast: u64,
    pub single_full: u64,
    pub single_fast: u64,
    pub deviations_full: u64,
    pub deviations_fast: u64,
    pub cost_full: f64,
    pub cost_fast: f64,
}

impl LateTotals {
    pub fn add(&mut self, o: &LateTotals) {
        self.decisions_full += o.decisions_full;
        self.decisions_fast += o.decisions_fast;
        self.single_full += o.single_full;
        self.single_fast += o.single_fast;
        self.deviations_full += o.deviations_full;
        self.deviations_fast += o.deviations_fast;
        self.cost_full += o.cost_full;
        self.cost_fast += o.cost_fast;
    }
    pub fn decisions(&self) -> u64 {
        self.decisions_full + self.decisions_fast
    }
}

/// The lookahead seats' counters summed over their decisions (`DriverStats::look`, docs/search.md
/// 3.9): decisions, the lookahead's nodes / leaves / rows / rounds of the stage that decided, the
/// decisions that needed a fallback stage, the final fallback, a minor root's top prior, aborted
/// stages' nodes, the most live member games of one decision, the rows of every round.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LookTotals {
    pub decisions: u64,
    pub nodes: u64,
    pub leaves: u64,
    pub prior_rows: u64,
    pub value_rows: u64,
    pub value_dups: u64,
    pub minor_steps: u64,
    pub groups: u64,
    pub cut: u64,
    pub rounds: u64,
    pub rows: u64,
    /// Decisions a fallback stage decided, decisions the final fallback decided (and of those, the
    /// MCTS searches), minor roots (the network's top prior, no lookahead).
    pub fallback_stage: u64,
    pub fallback_last: u64,
    pub fallback_mcts: u64,
    pub minor_root: u64,
    pub aborted_nodes: u64,
    pub peak_games: u64,
    /// Decisions per stage that decided (index = `LookStats::stage`; the last = the final fallback).
    pub by_stage: Vec<u64>,
    /// Wall time of the lookahead's own work on the search threads (s, summed over threads).
    pub busy_s: f64,
    /// The guarded player (`LookCfg::guard`): its decisions compared and those where the lookahead's
    /// move was played (its margin over the MCTS move above the guard).
    pub guard_decisions: u64,
    pub guard_overrides: u64,
}

impl LookTotals {
    pub fn add(&mut self, s: &crate::lookahead::LookStats, mcts: bool) {
        self.decisions += 1;
        self.nodes += s.nodes;
        self.leaves += s.leaves;
        self.prior_rows += s.prior_rows;
        self.value_rows += s.value_rows;
        self.value_dups += s.value_dups;
        self.minor_steps += s.minor_steps;
        self.groups += s.groups;
        self.cut += s.cut;
        self.rounds += s.rounds;
        self.rows += s.rows;
        self.fallback_stage += (s.stage > 0 && !s.last) as u64;
        self.fallback_last += s.last as u64;
        self.fallback_mcts += mcts as u64;
        self.minor_root += s.minor_root as u64;
        self.aborted_nodes += s.aborted_nodes;
        self.peak_games = self.peak_games.max(s.peak_games);
        let k = s.stage as usize;
        if self.by_stage.len() <= k {
            self.by_stage.resize(k + 1, 0);
        }
        self.by_stage[k] += 1;
    }
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
    /// Value-only rows of the network batches (`LeafView::Root`: one per leaf of a seat other
    /// than its tree's root decider; `leaves` counts the leaves) and value-only samples recorded
    /// (`NnSeat::view_samples`; `recorded` counts the decisions).
    pub view_rows: u64,
    pub view_recorded: u64,
    /// Root belief rows of the network batches (`ROW_BELIEF`: one per search that takes the belief,
    /// docs/search.md 3.8).
    pub belief_rows: u64,
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
    /// The lookahead seats' counters (docs/search.md 3.9; all 0 without one).
    pub look: LookTotals,
    /// Late sampling's counters (docs/search.md 3.10; all 0 without it).
    pub late: LateTotals,
}

/// The games in progress.
#[derive(Clone, Debug, Default)]
pub struct InFlight {
    pub games: u64,
    pub oldest_age_games: u64,
    pub max_steps: u64,
    pub max_rounds: u32,
}

/// An answered-to-be batch (`Driver::outstanding`): group, key, rows, opt_off, belief rows.
type Outstanding = (usize, u16, usize, Vec<i32>, Vec<u32>);

pub struct Driver<G: SearchGame> {
    sh: Arc<Shared<G>>,
    workers: Vec<JoinHandle<()>>,
    gstate: Vec<GState>,
    cursor: usize,
    ready: VecDeque<Batch>,
    /// batch id -> (group, key, rows, opt_off of a network batch, its root belief rows' indexes)
    outstanding: HashMap<u64, Outstanding>,
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
                    if !(n.view_samples.is_finite() && (0.0..=1.0).contains(&n.view_samples)) {
                        return Err(format!("view_samples {}: a share in [0, 1] (0 = off)", n.view_samples));
                    }
                    if n.root_min_visits > 0 && n.root_min_visits_max_actions < 2 {
                        return Err(format!("root_min_visits_max_actions {}: at least 2 (a root of one option is not searched)", n.root_min_visits_max_actions));
                    }
                    NnSeat::check_belief(n.belief_tilt, n.belief_clip, n.belief_mix)?;
                    if n.belief_tilt > 0.0 && G::aux_width() == 0 {
                        return Err(format!("belief_tilt {}: {} has no auxiliary targets (SearchGame::AUX) for a belief to predict", n.belief_tilt, G::NAME));
                    }
                    n.root_search.check()?;
                    n.check_temp_late()?;
                    if let Some(lc) = &n.look {
                        // (docs/search.md 3.9: an evaluation seat; its rows are batched like leaves)
                        lc.check()?;
                        if n.selfplay {
                            return Err("lookahead: an evaluation seat (selfplay false): it records no training samples yet".into());
                        }
                    }
                    if matches!(n.root_search, RootSearch::Gumbel { .. }) {
                        // (docs/search.md 3.7: sequential halving already visits every considered option)
                        if n.root_min_visits > 0 {
                            return Err(format!("root_min_visits {} with root_search 'gumbel': exclusive (sequential halving already gives every considered option its visits)", n.root_min_visits));
                        }
                        if cfg.record && n.selfplay && n.record && n.policy_target == PolicyTarget::Visits {
                            return Err("root_search 'gumbel' records the policy_target 'cq' (its visit counts are sequential halving's schedule, not a policy)".into());
                        }
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
        if cfg.aux && G::aux_width() == 0 {
            return Err(format!("aux: {} has no auxiliary targets (SearchGame::AUX)", G::NAME));
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
                let bufs = (0..nkeys).map(|_| LeafBuf { sparse: cfg.sparse, ..Default::default() }).collect();
                chunks.push(Mutex::new(Chunk { slots: (a..b).map(|_| Slot::empty()).collect(), bufs, results: Vec::new(), samples: Vec::new(), parked: Vec::new() }));
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
                let brows: Vec<u32> = b.extra.iter().enumerate().filter(|(_, &e)| e == ROW_BELIEF).map(|(r, _)| r as u32).collect();
                self.outstanding.insert(b.id, (b.group, b.key, b.b, b.opt_off.clone(), brows));
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
                    // (leaves: a leaf's extra rows are not leaves of their own)
                    self.dropped_leaves += if b.external { b.b } else { b.extra.iter().filter(|&&e| e == ROW_LEAF).count() } as u64;
                }
                if !out.batches.is_empty() {
                    for ch in &self.sh.groups[g].chunks {
                        for s in lock(ch).slots.iter_mut() {
                            // (a lookahead's unanswered round: its decision is looked at again)
                            if matches!(s.pending, Some(Pending::Look { .. })) {
                                s.look = None;
                            }
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
    /// (relative to the decider: column j = seat (player + j) % n_players). A batch with root
    /// belief rows needs `submit_aux`.
    pub fn submit(&mut self, id: u64, priors: &[f32], value: &[f32], margin: &[f32]) -> Result<(), String> {
        self.submit_aux(id, priors, value, margin, &[])
    }

    /// `submit` with the auxiliary outputs of the batch's root belief rows (`ROW_BELIEF`,
    /// docs/search.md 3.8): `aux` = [belief rows x aux_width()] in row order, the network's
    /// predictions of the game's `AUX` targets for those rows (empty when the batch has none). A
    /// batch with belief rows and no (or a wrongly sized) `aux` is an error: an evaluator that
    /// cannot compute the belief never leaves the search dealing uniformly in silence.
    pub fn submit_aux(&mut self, id: u64, priors: &[f32], value: &[f32], margin: &[f32], aux: &[f32]) -> Result<(), String> {
        let Some((g, key, b, opt_off, brows)) = self.outstanding.get(&id).map(|o| (o.0, o.1, o.2, &o.3, &o.4)) else {
            return Err(format!("unknown or already answered batch id {id}"));
        };
        let w = G::aux_width();
        if aux.len() != brows.len() * w {
            return Err(if brows.is_empty() {
                format!("batch {id} has no root belief rows: {} auxiliary values given", aux.len())
            } else {
                format!(
                    "batch {id} has {} root belief rows (ROW_BELIEF: a seat's belief_tilt > 0): its evaluator must answer their auxiliary outputs ({} x {w} values, got {}; a network without the auxiliary head cannot serve belief_tilt)",
                    brows.len(),
                    brows.len(),
                    aux.len()
                )
            });
        }
        if let Some(x) = aux.iter().find(|x| !x.is_finite() || **x < 0.0) {
            return Err(format!("batch {id}: a root belief row's auxiliary output {x} (predictions are finite and >= 0)"));
        }
        let mut aux_row = Vec::new();
        if !brows.is_empty() {
            aux_row = vec![NO_AUX; b];
            for (k, &r) in brows.iter().enumerate() {
                aux_row[r as usize] = k as u32;
            }
        }
        if self.sh.cfg.keys[key as usize].1 != KeyKind::Nn {
            return Err(format!("batch {id} is an external batch: use submit_external"));
        }
        let nopt = *opt_off.last().unwrap_or(&0) as usize;
        if priors.len() != nopt || value.len() != b * MAX_SEATS || margin.len() != b * MAX_SEATS {
            return Err(format!("evaluation shapes: priors {} (want {nopt}), value {} / margin {} (want {})", priors.len(), value.len(), margin.len(), b * MAX_SEATS));
        }
        let (_, _, _, opt_off, _) = self.outstanding.remove(&id).unwrap();
        let grp = &self.sh.groups[g];
        let bases = std::mem::take(&mut lock(&grp.bases)[key as usize]);
        write(&grp.evals)[key as usize] = Some(Eval::Nn { opt_off, priors: priors.to_vec(), value: value.to_vec(), margin: margin.to_vec(), bases, aux: aux.to_vec(), aux_row });
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
        let done = worker::Done { gid, tag: p.tag.take(), n: p.n, steps: p.steps, decisions: p.decisions, ext_moves: p.ext_moves, moves: Vec::new(), late: Vec::new() };
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
            view_rows: c.view_rows.load(Ordering::Relaxed),
            view_recorded: c.view_recorded.load(Ordering::Relaxed),
            belief_rows: c.belief_rows.load(Ordering::Relaxed),
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
            look: lock(&c.look).clone(),
            late: lock(&c.late).clone(),
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
    /// Auxiliary targets [n * aw] (`DriverCfg::aux`; aw = 0: none recorded).
    pub aw: usize,
    pub aux: Vec<f32>,
}

pub fn sample_arrays(recs: &[Record], f: usize) -> SampleArrays {
    let n = recs.len();
    let aw = recs.first().map_or(0, |r| r.aux.len());
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
        aw,
        aux: Vec::with_capacity(n * aw),
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
        assert_eq!(r.aux.len(), aw, "records with different auxiliary target widths");
        s.aux.extend_from_slice(&r.aux);
    }
    s
}

#[cfg(test)]
mod tests;
