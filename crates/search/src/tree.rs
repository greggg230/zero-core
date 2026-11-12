//! One search tree (docs/search.md 3; Python reference: `zero/mcts.py`). A port of Dominion
//! Zero's `engine/src/search/tree.rs`, generic over `SearchGame`.
//!
//! * A **node** is one decision as the search sees it: decider seat, legal options (fixed
//!   for the node), priors `P` (`P0` before root noise), visit counts `N`, value sums `W`
//!   (the decider's component of the utility vector), `vec` = the network's utility vector
//!   at expansion (first-play urgency), `tot` = backups through the node, `wsum` = sum of
//!   backed-up win vectors (absolute seats).
//! * An **edge** per (node, option) holds its children keyed by (chance key, signature of the
//!   next decision) and `outs`, the chance keys sampled through it (repeats kept).
//! * A **transition** = `step(a)` then `step(o)` while the pending decision has a single
//!   option `o` (any seat); its chance key = every chance outcome of the whole transition.
//! * signature = (decider, legal actions in order): two decisions share a node iff equal, so
//!   a node's option list never changes even if hidden information differs between
//!   simulations.
//!
//! Nodes do not store states: every simulation clones the root state, determinizes it (a
//! fresh world per simulation with `redet`) and replays the path. Everything lives in flat
//! arrays (no allocation per node); a reused subtree keeps the old arrays until `compact`.
//!
//! Policy targets of recorded decisions (`driver::PolicyTarget`): the root visit distribution,
//! or the completed-Q target of Gumbel AlphaZero (`completed_q_policy`, `Tree::cq_target`).
//!
//! **Deferred terminals** (`SearchGame::deferred`, docs/search.md 3.2): a walk that ends the
//! game at a deferred terminal reaches a child node of no options decided by the tree's viewer
//! (keyed like any child: chance key, viewer, no actions). Unexpanded, it is a leaf the network
//! evaluates (`encode(viewer)`, no priors); `expand_leaf` stores its utility in `vec` and its win
//! vector in `wsum` (`tot` = 1), and every later walk that reaches it backs that value up
//! again. `result()` is never called there. **Teams** (`SearchGame::team_size` = T, `Tree::team`):
//! utilities are `T x win[seat]` (+ the margin term), win vectors stay as `result()` (sum 1).
//!
//! **Root minimum visits** (`Tree::set_root_min`, docs/search.md 3.5; off by default): a search
//! may give every root option at least k visits. A walk whose root option PUCT chose is never
//! changed; when PUCT's choice already has k visits and another root option has fewer, the walk
//! takes the least-visited root option instead (the lowest index on ties) - a **forced walk**.
//! The tree counts the forced walks per root option and sums their backed-up win vectors, so the
//! search's visit-based outputs can leave them out (`root_visits_pruned`, `root_win_pruned`)
//! while its Q-based ones use them (Q, `cq_target`).
//!
//! **The root's belief** (`Tree::set_belief`, docs/search.md 3.8; off by default): a search may hold
//! the network's prediction of the hidden cards for its viewer (`Belief`, computed once per decision
//! from the root's features); every simulation then deals its world with
//! `SearchGame::determinize_with` instead of `determinize` - but a `belief_mix` share, dealt
//! uniformly as before. `None` (the default, and after `reset` / `advance`) is the tree of before,
//! bit for bit.
//! **Gumbel root search** (`Tree::set_gumbel`, docs/search.md 3.7; off by default): Gumbel
//! AlphaZero's root (Danihelka et al. 2022, "Policy improvement by planning with Gumbel"; the
//! arithmetic of DeepMind's mctx `gumbel_muzero_policy`). The search's root walks follow
//! sequential halving (`gumbel_sequence`): walk j may only take a root option whose visits of THIS
//! search equal the schedule's `seq[j]`, and among those the one maximising `g + ln P0 +
//! sigma(completedQ)` (`completed_q_logits`, the 'cq' target's arithmetic; g = the search's Gumbel
//! noise per option). The first `min(m, options)` walks visit the top options by `g + ln P0` (the
//! Gumbel-top-k sample), then the field is halved until two remain. Below the root PUCT is
//! unchanged. The final move is `gumbel_choose` (the same score among the options this search
//! visited most), the root value `gumbel_q` (the improved policy's value). A reused root's earlier
//! visits count in Q and sigma but not in the schedule (this search's walks are counted apart).
//!
//! **Information-set merging** (`Tree::set_merge`, docs/search.md 3.16; off by default, `merge.rs`): an edge's
//! children become links (the signature above) to nodes merged by the root decider's observation - the tree is
//! then a DAG, the other seats' nodes subset-armed over the union of their worlds' options. Off, nothing of it
//! runs: the tree of before, bit for bit.
use std::hash::{Hash, Hasher};

use crate::game::{Belief, SearchGame, MAX_SEATS};
use crate::noise;
use crate::rng::Rng;

const NIL: u32 = u32::MAX;
/// Node count above which a reused root triggers compaction.
const COMPACT_AT: usize = 1 << 16;

mod merge;
pub use merge::MergeStats;
use merge::Merge;

/// Per-option statistics of an expanded node.
#[derive(Clone, Copy, Debug)]
struct Stat {
    p: f32,
    p0: f32,
    n: u32,
    w: f64,
    /// First child node of this edge (linked through `Node::next`), or NIL. In a merged tree: the first link
    /// (`merge::Link`, linked through `Link::next`).
    kid: u32,
    /// Newest sampled-outcome record (`Tree::outs`, linked), or NIL.
    outs: u32,
    nouts: u32,
}

/// A sampled chance outcome stored on an edge.
#[derive(Clone, Copy, Debug)]
struct Out {
    key_off: u32,
    key_len: u32,
    next: u32,
}

#[derive(Clone, Debug)]
struct Node {
    player: u8,
    acts_off: u32,
    nacts: u32,
    key_off: u32,
    key_len: u32,
    hash: u64,
    next: u32,
    stat_off: u32,
    tot: u32,
    vec: [f64; MAX_SEATS],
    wsum: [f64; MAX_SEATS],
}

/// Outcome of one simulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sim {
    /// The walk reached an unexpanded node: the scratch game holds the leaf state; the caller
    /// evaluates `encode(player)` and calls `expand_leaf` + `backup` with `leaf_values_team`
    /// (`expand` + `leaf_values` for a game without teams or deferred terminals). A deferred
    /// terminal's node has no options (no priors). In a merged tree (`set_merge`) the node may be an
    /// expanded node of another seat whose union lacks one of this world's options (a growth leaf): its
    /// row's options are `leaf_acts(node)` - always read them there.
    Leaf { node: u32, player: usize },
    /// The walk ended the game (or reached an evaluated deferred terminal); its value was
    /// backed up.
    Terminal,
}

/// One edge of `Tree::shape` (a diagnostic): the depth of its node (root 0), whether the tree's viewer
/// decides that node, its visits, its child nodes (`opp_kids` of them decided by another seat,
/// `expanded_kids` expanded).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EdgeShape {
    pub depth: u32,
    pub own: bool,
    pub n: u32,
    pub kids: u32,
    pub opp_kids: u32,
    pub expanded_kids: u32,
    /// Among the other-seat children: their distinct option lists (decider and legal options), their
    /// distinct chance keys, the backups through them (`tot`) and the backups through the children of the
    /// most backed-up option list (the strategy-fusion sizing of docs/search.md 3.16).
    pub opp_lists: u32,
    pub opp_keys: u32,
    pub opp_tot: u32,
    pub opp_modal_tot: u32,
}

/// The shape of a tree under its root (`Tree::shape`, docs/search.md 3.15): nodes reachable, expanded,
/// expanded nodes per depth (root 0), the deepest node, every edge with a visit or a child, and the
/// main line (the most visited option from the root on, then that edge's most backed-up child).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Shape {
    pub nodes: u32,
    pub expanded: u32,
    pub depth_hist: Vec<u32>,
    /// Expanded nodes by the number of nodes above them on their path decided by another seat than
    /// the viewer (the other seat's replies a walk to them modelled; root 0).
    pub opp_hist: Vec<u32>,
    /// Expanded nodes by the number of decider changes on their path (root 0).
    pub switch_hist: Vec<u32>,
    pub max_depth: u32,
    pub edges: Vec<EdgeShape>,
    pub main_line: Vec<EdgeShape>,
}

/// Search settings of one tree walk.
#[derive(Clone, Copy, Debug)]
pub struct Walk {
    pub c_puct: f64,
    pub chance_k: u32,
    pub margin_w: f64,
}

/// Reusable buffers of a simulation.
#[derive(Default, Clone, Debug)]
pub struct Scratch {
    pub path: Vec<(u32, u32)>,
    pub key: Vec<u64>,
    pub legal: Vec<u16>,
}

fn sig_hash(key: &[u64], player: usize, acts: &[u16]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut h);
    player.hash(&mut h);
    acts.hash(&mut h);
    h.finish()
}

/// `step(a)`, then auto-step single-option decisions while `auto(seat)` holds; the chance keys
/// of the whole transition go to `key`, and `legal` ends up holding the legal actions of the
/// pending decision (empty if the game is over). `forced`: outcomes to replay (chance_k).
pub fn transition_with<G: SearchGame>(g: &mut G, a: u16, forced: Option<&[u64]>, key: &mut Vec<u64>, legal: &mut Vec<u16>, auto: impl Fn(usize) -> bool) -> Result<(), String> {
    transition_each(g, a, forced, key, legal, auto, |_| {})
}

/// `transition_with`, calling `each(action)` after every step it makes - `a`, then each single-option
/// decision it steps (the driver's trace of a live game's real moves, docs/search.md 2.15).
pub fn transition_each<G: SearchGame>(g: &mut G, a: u16, forced: Option<&[u64]>, key: &mut Vec<u64>, legal: &mut Vec<u16>, auto: impl Fn(usize) -> bool, mut each: impl FnMut(u16)) -> Result<(), String> {
    key.clear();
    legal.clear();
    if let Some(f) = forced {
        g.force_chance(f);
    }
    let r = (|| {
        g.step(a, key)?;
        each(a);
        while let Some(p) = g.to_move() {
            legal.clear();
            g.legal_actions(legal);
            if legal.is_empty() {
                return Err(format!("seat {p} to move without legal actions"));
            }
            if legal.len() != 1 || !auto(p) {
                break;
            }
            let o = legal[0];
            g.step(o, key)?;
            each(o);
            legal.clear();
        }
        Ok(())
    })();
    if forced.is_some() {
        g.force_chance(&[]);
    }
    r
}

/// The search transition: auto-steps every single-option decision.
pub fn transition<G: SearchGame>(g: &mut G, a: u16, forced: Option<&[u64]>, key: &mut Vec<u64>, legal: &mut Vec<u16>) -> Result<(), String> {
    transition_with(g, a, forced, key, legal, |_| true)
}

/// Network output (relative to the decider) -> (util, win) over absolute seats (a game of
/// single-seat teams: `leaf_values_team` with team 1).
pub fn leaf_values(value_rel: &[f32], margin_rel: &[f32], player: usize, n: usize, margin_w: f64) -> ([f64; MAX_SEATS], [f64; MAX_SEATS]) {
    leaf_values_team(value_rel, margin_rel, player, n, margin_w, 1)
}

/// Network output (relative to the decider) -> (util, win) over absolute seats, for teams of
/// `team` seats (`SearchGame::team_size`): util = team x win (+ margin_w tanh(margin)), win =
/// the network's value (sums to 1). Team 1 is `leaf_values` bit for bit.
pub fn leaf_values_team(value_rel: &[f32], margin_rel: &[f32], player: usize, n: usize, margin_w: f64, team: usize) -> ([f64; MAX_SEATS], [f64; MAX_SEATS]) {
    let t = team.max(1) as f64;
    let mut win = [0.0; MAX_SEATS];
    let mut util = [0.0; MAX_SEATS];
    for j in 0..n {
        let s = (player + j) % n;
        let w = value_rel[j] as f64;
        win[s] = w;
        let tw = t * w;
        util[s] = if margin_w != 0.0 { tw + margin_w * (margin_rel[j] as f64).tanh() } else { tw };
    }
    (util, win)
}

/// Floor of a prior probability inside `completed_q_policy` (`f32::MIN_POSITIVE`, as mctx's
/// `finfo(float32).tiny`): a prior that underflowed to 0 stays finite in the log and in the
/// mixed value's weights.
pub const PRIOR_FLOOR: f64 = f32::MIN_POSITIVE as f64;

/// The completed-Q policy-improvement target of Gumbel AlphaZero (Danihelka et al. 2022,
/// "Policy improvement by planning with Gumbel"; mctx `qtransform_completed_by_mix_value`
/// without its per-node min-max rescaling, docs/search.md 3.1):
///
/// `pi' = softmax(ln P0 + (c_visit + max_b N_b) * c_scale * completedQ)` with
/// `completedQ_a = W_a / N_a` for visited options and `v_mix` for unvisited ones,
/// `v_mix = (v + sum_b N_b * sum_{a: N_a > 0} P0_a Q_a / sum_{a: N_a > 0} P0_a) / (1 + sum_b N_b)`
/// (their eq. 33; `v` = the network's value of the node for its decider; = `v` with no visits).
///
/// `p0`: the noise-free priors, `n` / `w`: visit counts and value sums (the decider's utility,
/// so Q is P(win) in [0, 1] with `margin_w` 0: no further normalisation), `v`: the decider's
/// network utility. Computed in f64; the result sums to 1 (f32).
pub fn completed_q_policy(p0: &[f32], n: &[u32], w: &[f64], v: f64, c_visit: f64, c_scale: f64) -> Vec<f32> {
    completed_q_policy_w(p0, n, w, v, c_visit, c_scale, 1.0)
}

/// `completed_q_policy` with a prior exponent `prior_w` (beta; docs/search.md 3.13):
/// `softmax(beta ln max(P0, PRIOR_FLOOR) + (c_visit + max N) c_scale completedQ)`, i.e.
/// `pi' ~ P0^beta exp(sigma completedQ)`. Only the prior's term is scaled: `v_mix` keeps its P0
/// weights, and `completed_q_logits` (the 'cq' move rule's and the Gumbel root's arithmetic) has
/// no beta. 1.0 = `completed_q_policy` bit for bit (the logits are not touched). The logits are
/// `completed_q_logits` plus `(beta - 1) ln max(P0, PRIOR_FLOOR)` (the Python reference's
/// arithmetic too, `mcts.completed_q_policy`).
pub fn completed_q_policy_w(p0: &[f32], n: &[u32], w: &[f64], v: f64, c_visit: f64, c_scale: f64, prior_w: f64) -> Vec<f32> {
    let mut lg = completed_q_logits(p0, n, w, v, c_visit, c_scale);
    if prior_w != 1.0 {
        for (x, &p) in lg.iter_mut().zip(p0) {
            *x += (prior_w - 1.0) * (p as f64).max(PRIOR_FLOOR).ln();
        }
    }
    let m = lg.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut s = 0.0;
    for x in lg.iter_mut() {
        *x = (*x - m).exp();
        s += *x;
    }
    lg.iter().map(|&x| (x / s) as f32).collect()
}

/// The logits of `completed_q_policy`: `ln max(P0, PRIOR_FLOOR) + (c_visit + max N) c_scale
/// completedQ` per option (f64; the policy is their softmax). Their argmax among the options a
/// guard admits is the `driver::Select::Cq` move (docs/search.md 3.3).
pub fn completed_q_logits(p0: &[f32], n: &[u32], w: &[f64], v: f64, c_visit: f64, c_scale: f64) -> Vec<f64> {
    let k = p0.len();
    assert!(n.len() == k && w.len() == k, "completed_q_policy: {k} priors, {} visit counts, {} value sums", n.len(), w.len());
    let mut sum_n = 0.0;
    let mut max_n = 0u32;
    let mut sum_p = 0.0;
    let mut sum_pq = 0.0;
    for i in 0..k {
        if n[i] > 0 {
            let p = (p0[i] as f64).max(PRIOR_FLOOR);
            sum_n += n[i] as f64;
            max_n = max_n.max(n[i]);
            sum_p += p;
            sum_pq += p * (w[i] / n[i] as f64);
        }
    }
    let v_mix = if sum_n > 0.0 { (v + sum_n * (sum_pq / sum_p)) / (1.0 + sum_n) } else { v };
    let scale = (c_visit + max_n as f64) * c_scale;
    (0..k)
        .map(|i| {
            let q = if n[i] > 0 { w[i] / n[i] as f64 } else { v_mix };
            (p0[i] as f64).max(PRIOR_FLOOR).ln() + scale * q
        })
        .collect()
}

/// (util, win) of a finished game (util = team_size x win (+ margin_w tanh(margin))).
pub fn terminal_values<G: SearchGame>(g: &G, margin_w: f64) -> ([f64; MAX_SEATS], [f64; MAX_SEATS]) {
    let r = g.result();
    let n = g.num_players();
    let t = g.team_size().max(1) as f64;
    let mut win = [0.0; MAX_SEATS];
    for i in 0..n {
        win[i] = r[i] as f64;
    }
    if margin_w == 0.0 && t == 1.0 {
        return (win, win);
    }
    let mut util = [0.0; MAX_SEATS];
    for i in 0..n {
        util[i] = if margin_w != 0.0 { t * win[i] + margin_w * (g.margin(i) as f64).tanh() } else { t * win[i] };
    }
    (util, win)
}

/// Sequential halving's schedule (mctx `get_sequence_of_considered_visits`): for each of a search's
/// `n` root walks, the visit count (this search's) a root option must have to be taken by it, with
/// `m` options considered (`min(gumbel_m, options)`). Rounds of `max(1, n / (ceil(log2 m) x
/// considered))` walks per considered option, the considered halved after each round (at least 2);
/// `m <= 1`: 0, 1, 2, ... (one option takes every walk). Its length is `n`.
pub fn gumbel_sequence(m: u32, n: u32) -> Vec<u32> {
    let n = n as usize;
    if m <= 1 {
        return (0..n as u32).collect();
    }
    let log2max = m.next_power_of_two().trailing_zeros();
    let mut seq = Vec::with_capacity(n + m as usize);
    let mut visits = vec![0u32; m as usize];
    let mut nc = m as usize;
    while seq.len() < n {
        let extra = (n / (log2max as usize * nc)).max(1);
        for _ in 0..extra {
            seq.extend_from_slice(&visits[..nc]);
            for v in visits[..nc].iter_mut() {
                *v += 1;
            }
        }
        nc = (nc / 2).max(2);
    }
    seq.truncate(n);
    seq
}

/// The index of the highest `score` among the indices `ok` admits (the first maximum on exact ties),
/// None when it admits none.
fn best_of(score: &[f64], ok: impl Fn(usize) -> bool) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, &x) in score.iter().enumerate() {
        if ok(i) && best.is_none_or(|b| x > score[b]) {
            best = Some(i);
        }
    }
    best
}

/// The Gumbel root search of the current search (`Tree::set_gumbel`).
#[derive(Clone, Debug)]
struct GumbelRoot {
    /// Sequential halving's schedule over this search's root walks (`gumbel_sequence`).
    seq: Vec<u32>,
    /// The Gumbel noise per root option (0 without noise).
    g: Vec<f64>,
    /// sigma's completed-Q transform (`completed_q_logits`).
    c_visit: f64,
    c_scale: f64,
    /// Per root option: this search's walks and the sum of their backed-up win vectors.
    n: Vec<u32>,
    win: Vec<[f64; MAX_SEATS]>,
    /// This search's root walks so far (sum of `n`).
    walks: u32,
}

#[derive(Clone, Debug)]
pub struct Tree<G: SearchGame> {
    /// The root decider (whose information set the tree searches).
    pub viewer: usize,
    pub redet: bool,
    /// Players in the game.
    pub n: usize,
    /// Seats per team (`SearchGame::team_size`): utilities are team x win.
    pub team: usize,
    pub root: u32,
    nodes: Vec<Node>,
    stats: Vec<Stat>,
    acts: Vec<u16>,
    keys: Vec<u64>,
    outs: Vec<Out>,
    /// `redet == false`: the one determinized world every simulation clones.
    base: Option<G>,
    /// Root minimum visits of the current search (`set_root_min`; 0 = off).
    root_min: u32,
    /// Per root option: this search's forced walks backed up through it.
    forced: Vec<u32>,
    /// The sum of the forced walks' backed-up win vectors (absolute seats) and their number.
    forced_win: [f64; MAX_SEATS],
    forced_n: u32,
    /// The root option of the walk in flight if the floor forced it (NIL: not forced).
    walk_forced: u32,
    /// The root's belief of the current search (`set_belief`; None: every world dealt uniformly)
    /// and the share of its simulations still dealt uniformly.
    belief: Option<Belief>,
    belief_mix: f64,
    /// The Gumbel root search of the current search (`set_gumbel`; None = PUCT at the root).
    gumbel: Option<Box<GumbelRoot>>,
    /// Information-set merging (`set_merge`, docs/search.md 3.16; None = the tree of before).
    merge: Option<Box<Merge>>,
}

/// Deal a world for `viewer` (docs/search.md 3.8): `determinize_with` the belief, but a `mix` share
/// of the calls (one draw of `rng` when 0 < mix) and every call without a belief `determinize`.
/// (Also the lookahead's deal of its worlds, docs/search.md 3.9.)
pub(crate) fn deal<G: SearchGame>(g: &mut G, viewer: usize, belief: Option<&Belief>, mix: f64, rng: &mut Rng) {
    match belief {
        Some(b) if !(mix > 0.0 && rng.unit() < mix) => g.determinize_with(viewer, rng, b),
        _ => g.determinize(viewer, rng),
    }
}

impl<G: SearchGame> Default for Tree<G> {
    fn default() -> Self {
        Tree {
            viewer: 0,
            redet: true,
            n: 0,
            team: 1,
            root: 0,
            nodes: Vec::new(),
            stats: Vec::new(),
            acts: Vec::new(),
            keys: Vec::new(),
            outs: Vec::new(),
            base: None,
            root_min: 0,
            forced: Vec::new(),
            forced_win: [0.0; MAX_SEATS],
            forced_n: 0,
            walk_forced: NIL,
            belief: None,
            belief_mix: 0.0,
            gumbel: None,
            merge: None,
        }
    }
}

impl<G: SearchGame> Tree<G> {
    /// A search for the pending decision of `game` (decided by `viewer`, legal actions `legal`).
    pub fn new(game: &G, viewer: usize, legal: &[u16], redet: bool, rng: &mut Rng) -> Tree<G> {
        let mut t = Tree::default();
        t.reset(game, viewer, legal, redet, rng);
        t
    }

    /// Reuse this tree's buffers for a fresh search (see `new`).
    pub fn reset(&mut self, game: &G, viewer: usize, legal: &[u16], redet: bool, rng: &mut Rng) {
        self.viewer = viewer;
        self.redet = redet;
        self.n = game.num_players();
        self.team = game.team_size().max(1);
        self.nodes.clear();
        self.stats.clear();
        self.acts.clear();
        self.keys.clear();
        self.outs.clear();
        // (the merge state goes: `set_merge` after a reset turns it on again, docs/search.md 3.16)
        self.merge = None;
        self.root = self.push_node(&[], viewer, legal, 0);
        self.set_root_min(0);
        self.belief = None;
        self.belief_mix = 0.0;
        self.gumbel = None;
        self.set_base(game, rng);
    }

    /// The root's belief for the search starting now (docs/search.md 3.8): every simulation deals
    /// its world with `determinize_with(viewer, rng, belief)`, a `mix` share (in [0, 1)) with
    /// `determinize` as before; `None` = every world uniform (the tree of before, bit for bit).
    /// Without `redet` the one world of the decision is dealt again from `live` with it. `reset`
    /// and `advance` clear it (a belief is the prediction for one decision).
    pub fn set_belief(&mut self, belief: Option<Belief>, mix: f64, live: &G, rng: &mut Rng) {
        self.belief = belief;
        self.belief_mix = if self.belief.is_some() { mix } else { 0.0 };
        if !self.redet {
            self.set_base(live, rng);
        }
    }

    /// The root's belief of the current search (None: uniform worlds).
    pub fn belief(&self) -> Option<&Belief> {
        self.belief.as_ref()
    }

    /// The share of the current search's simulations dealt uniformly although it holds a belief.
    pub fn belief_mix(&self) -> f64 {
        self.belief_mix
    }

    /// Root minimum visits for the search starting now (docs/search.md 3.5): every root option
    /// gets at least `k` visits (0 = off: the search of before, bit for bit). Clears the forced
    /// walks' accounting; `reset` and `advance` turn it off. The caller decides whether a search
    /// takes it (`driver::root_floor`: full searches of a few options with the budget for it).
    pub fn set_root_min(&mut self, k: u32) {
        self.root_min = k;
        self.forced.clear();
        if k > 0 {
            self.forced.resize(self.nodes[self.root as usize].nacts as usize, 0);
        }
        self.forced_win = [0.0; MAX_SEATS];
        self.forced_n = 0;
        self.walk_forced = NIL;
    }

    /// The root minimum visits of the current search (0 = off).
    pub fn root_min(&self) -> u32 {
        self.root_min
    }

    /// Per root option: the forced walks of this search (empty when the floor is off).
    pub fn root_forced(&self) -> &[u32] {
        &self.forced
    }

    /// The root's visit counts without this search's forced walks (N_i - forced_i; the visit
    /// counts when the floor is off): the visit distribution PUCT chose, for the visit-based
    /// outputs (the 'visits' target, temperature sampling, the 'visits' move rule).
    pub fn root_visits_pruned(&self) -> Vec<u32> {
        let mut v = self.visits(self.root);
        for (x, &f) in v.iter_mut().zip(&self.forced) {
            *x -= f;
        }
        v
    }

    /// The search's average win vector at the root without this search's forced walks (`win`
    /// when nothing was forced; None when no other backup went through the root).
    pub fn root_win_pruned(&self) -> Option<[f64; MAX_SEATS]> {
        if self.forced_n == 0 {
            return self.win(self.root);
        }
        let nd = &self.nodes[self.root as usize];
        let t = nd.tot.saturating_sub(self.forced_n);
        if t == 0 {
            return None;
        }
        let mut w = [0.0; MAX_SEATS];
        for j in 0..self.n {
            w[j] = (nd.wsum[j] - self.forced_win[j]) / t as f64;
        }
        Some(w)
    }

    /// The root option a walk takes under the floor, given PUCT's choice `i0`: `i0` unless it has
    /// at least `root_min` visits and another root option has fewer - then the least-visited root
    /// option (the lowest index on ties): a forced walk.
    fn force_root(&mut self, i0: usize) -> usize {
        let k = self.root_min;
        let st = self.stats_of(self.root);
        if st[i0].n < k {
            return i0;
        }
        let mut j = 0;
        for (i, s) in st.iter().enumerate() {
            if s.n < st[j].n {
                j = i;
            }
        }
        if st[j].n >= k {
            return i0;
        }
        self.walk_forced = j as u32;
        j
    }

    /// The Gumbel root search for the search starting now (docs/search.md 3.7): sequential halving
    /// over `min(m, options)` considered root options, its schedule that of `budget` root walks (the
    /// driver: the search's simulations - a fresh root's own evaluation takes the first, so its walks
    /// follow the schedule's first `budget - 1` entries; walks past it take the most walked), Gumbel noise
    /// `noise` per root option (empty: none, g = 0; else one per option), sigma's transform
    /// `c_visit`, `c_scale` (the 'cq' target's arithmetic). Exclusive with the root minimum visits
    /// (the floor is turned off). `reset` and `advance` turn it off.
    pub fn set_gumbel(&mut self, m: u32, budget: u32, noise: &[f64], c_visit: f64, c_scale: f64) {
        let k = self.nodes[self.root as usize].nacts as usize;
        assert!(noise.is_empty() || noise.len() == k, "set_gumbel: {} noise values for {k} root options", noise.len());
        self.set_root_min(0);
        let considered = m.min(k as u32);
        self.gumbel = Some(Box::new(GumbelRoot {
            seq: gumbel_sequence(considered, budget),
            g: if noise.is_empty() { vec![0.0; k] } else { noise.to_vec() },
            c_visit,
            c_scale,
            n: vec![0; k],
            win: vec![[0.0; MAX_SEATS]; k],
            walks: 0,
        }));
    }

    /// Back to PUCT at the root (the Gumbel root search off).
    pub fn clear_gumbel(&mut self) {
        self.gumbel = None;
    }

    /// The current search's root is a Gumbel root search's (`set_gumbel`).
    pub fn gumbel_on(&self) -> bool {
        self.gumbel.is_some()
    }

    /// Per root option: the current Gumbel search's walks (empty when off).
    pub fn gumbel_visits(&self) -> &[u32] {
        self.gumbel.as_ref().map_or(&[], |g| &g.n)
    }

    /// Per root option: the current Gumbel search's noise (empty when off).
    pub fn gumbel_noise(&self) -> &[f64] {
        self.gumbel.as_ref().map_or(&[], |g| &g.g)
    }

    /// Per root option: the sum of the win vectors the current Gumbel search's walks backed up
    /// (absolute seats; empty when off).
    pub fn gumbel_win(&self) -> &[[f64; MAX_SEATS]] {
        self.gumbel.as_ref().map_or(&[], |g| &g.win)
    }

    /// The current Gumbel search's schedule (empty when off).
    pub fn gumbel_schedule(&self) -> &[u32] {
        self.gumbel.as_ref().map_or(&[], |g| &g.seq)
    }

    /// `g + completed_q_logits` per root option (sigma from every visit of the root, a reused root's
    /// too): the score of the root walks and of the final move.
    fn gumbel_scores(&self, gr: &GumbelRoot) -> Vec<f64> {
        let st = self.stats_of(self.root);
        let p0: Vec<f32> = st.iter().map(|s| s.p0).collect();
        let n: Vec<u32> = st.iter().map(|s| s.n).collect();
        let w: Vec<f64> = st.iter().map(|s| s.w).collect();
        let mut lg = completed_q_logits(&p0, &n, &w, self.value0(self.root), gr.c_visit, gr.c_scale);
        for (x, g) in lg.iter_mut().zip(&gr.g) {
            *x += g;
        }
        lg
    }

    /// The root option of the next walk: among the options whose walks of this search equal the
    /// schedule's entry for it (past the schedule: the most walked ones), the highest score, the
    /// first maximum on exact ties (mctx `gumbel_muzero_root_action_selection`).
    fn gumbel_select(&self) -> usize {
        let gr = self.gumbel.as_ref().unwrap();
        let want = match gr.seq.get(gr.walks as usize) {
            Some(&v) => v,
            None => gr.n.iter().copied().max().unwrap_or(0),
        };
        let sc = self.gumbel_scores(gr);
        best_of(&sc, |i| gr.n[i] == want).unwrap_or_else(|| best_of(&sc, |_| true).unwrap_or(0))
    }

    /// The Gumbel search's move (docs/search.md 3.7; mctx `gumbel_muzero_policy`'s action): among the
    /// root options this search walked most (the last survivors of sequential halving; every option
    /// when it walked none), the highest `g + ln P0 + sigma(completedQ)`, the first maximum on exact
    /// ties. 0 when the root has no statistics; None when the Gumbel search is off.
    pub fn gumbel_choose(&self) -> Option<usize> {
        let gr = self.gumbel.as_ref()?;
        if !self.expanded(self.root) {
            return Some(0);
        }
        let top = gr.n.iter().copied().max().unwrap_or(0);
        let sc = self.gumbel_scores(gr);
        Some(best_of(&sc, |i| gr.n[i] == top).unwrap_or(0))
    }

    /// The Gumbel search's root value (absolute seats; docs/search.md 3.7): the improved policy's
    /// value, `sum_i pi'_i Wv_i / sum_i pi'_i` over the root options this search walked, pi' =
    /// `completed_q_policy` with the search's transform (the 'cq' target when the transforms agree),
    /// Wv_i = this search's mean backed-up win vector through option i. Sequential halving spends its
    /// first round on every considered option, so the mean of the root's backups (`win`) would mix
    /// the options it ruled out into the value; pi' is the policy the search's move samples (the
    /// Gumbel-max trick). None when this search walked no root option (or the Gumbel search is off).
    pub fn gumbel_q(&self) -> Option<[f64; MAX_SEATS]> {
        let gr = self.gumbel.as_ref()?;
        if gr.walks == 0 || !self.expanded(self.root) {
            return None;
        }
        let st = self.stats_of(self.root);
        let p0: Vec<f32> = st.iter().map(|s| s.p0).collect();
        let n: Vec<u32> = st.iter().map(|s| s.n).collect();
        let w: Vec<f64> = st.iter().map(|s| s.w).collect();
        let lg = completed_q_logits(&p0, &n, &w, self.value0(self.root), gr.c_visit, gr.c_scale);
        let m = lg.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let (mut tot, mut q) = (0.0, [0.0; MAX_SEATS]);
        for i in 0..lg.len() {
            if gr.n[i] == 0 {
                continue;
            }
            let p = (lg[i] - m).exp();
            tot += p;
            for j in 0..MAX_SEATS {
                q[j] += p * gr.win[i][j] / gr.n[i] as f64;
            }
        }
        for x in q.iter_mut() {
            *x /= tot;
        }
        Some(q)
    }

    /// Start a search from `game` (the live game): with `redet` every simulation clones and
    /// determinizes it; without, one determinized world is sampled now.
    pub fn set_base(&mut self, game: &G, rng: &mut Rng) {
        if self.redet {
            self.base = None;
        } else {
            let mut g = match self.base.take() {
                Some(mut b) => {
                    b.clone_from(game);
                    b
                }
                None => game.clone(),
            };
            deal(&mut g, self.viewer, self.belief.as_ref(), self.belief_mix, rng);
            self.base = Some(g);
        }
    }

    fn push_node(&mut self, key: &[u64], player: usize, acts: &[u16], hash: u64) -> u32 {
        let id = self.nodes.len() as u32;
        let acts_off = self.acts.len() as u32;
        self.acts.extend_from_slice(acts);
        let key_off = self.keys.len() as u32;
        self.keys.extend_from_slice(key);
        self.nodes.push(Node {
            player: player as u8,
            acts_off,
            nacts: acts.len() as u32,
            key_off,
            key_len: key.len() as u32,
            hash,
            next: NIL,
            stat_off: NIL,
            tot: 0,
            vec: [0.0; MAX_SEATS],
            wsum: [0.0; MAX_SEATS],
        });
        if let Some(m) = self.merge.as_mut() {
            m.pushed();
        }
        id
    }

    fn node_key(&self, id: u32) -> &[u64] {
        let nd = &self.nodes[id as usize];
        &self.keys[nd.key_off as usize..(nd.key_off + nd.key_len) as usize]
    }

    fn node_matches(&self, id: u32, hash: u64, key: &[u64], player: usize, acts: &[u16]) -> bool {
        let nd = &self.nodes[id as usize];
        nd.hash == hash && nd.player as usize == player && self.node_key(id) == key && self.acts(id) == acts
    }

    fn find_kid(&self, node: u32, i: usize, key: &[u64], player: usize, acts: &[u16], hash: u64) -> Option<u32> {
        let so = self.nodes[node as usize].stat_off;
        if so == NIL {
            return None;
        }
        let mut k = self.stats[so as usize + i].kid;
        while k != NIL {
            if self.node_matches(k, hash, key, player, acts) {
                return Some(k);
            }
            k = self.nodes[k as usize].next;
        }
        None
    }

    // ---------------------------------------------------------------- queries

    pub fn acts(&self, node: u32) -> &[u16] {
        let nd = &self.nodes[node as usize];
        &self.acts[nd.acts_off as usize..(nd.acts_off + nd.nacts) as usize]
    }
    pub fn player(&self, node: u32) -> usize {
        self.nodes[node as usize].player as usize
    }
    pub fn expanded(&self, node: u32) -> bool {
        self.nodes[node as usize].stat_off != NIL
    }
    pub fn tot(&self, node: u32) -> u32 {
        self.nodes[node as usize].tot
    }
    fn stats_of(&self, node: u32) -> &[Stat] {
        let nd = &self.nodes[node as usize];
        if nd.stat_off == NIL {
            return &[];
        }
        &self.stats[nd.stat_off as usize..(nd.stat_off + nd.nacts) as usize]
    }
    /// Visit counts per option (empty if not expanded).
    pub fn visits(&self, node: u32) -> Vec<u32> {
        self.stats_of(node).iter().map(|s| s.n).collect()
    }
    pub fn priors(&self, node: u32) -> Vec<f32> {
        self.stats_of(node).iter().map(|s| s.p).collect()
    }
    pub fn priors0(&self, node: u32) -> Vec<f32> {
        self.stats_of(node).iter().map(|s| s.p0).collect()
    }
    pub fn values(&self, node: u32) -> Vec<f64> {
        self.stats_of(node).iter().map(|s| s.w).collect()
    }
    /// Visit count of option i (0 if not expanded).
    pub fn n_of(&self, node: u32, i: usize) -> u32 {
        self.stats_of(node).get(i).map(|s| s.n).unwrap_or(0)
    }
    /// Value sum W of option i (decider's utility; 0 if not expanded).
    pub fn w_of(&self, node: u32, i: usize) -> f64 {
        self.stats_of(node).get(i).map(|s| s.w).unwrap_or(0.0)
    }
    /// Prior P0 of option i (before noise; 0 if not expanded).
    pub fn p0_of(&self, node: u32, i: usize) -> f32 {
        self.stats_of(node).get(i).map(|s| s.p0).unwrap_or(0.0)
    }
    /// The network's utility vector stored at expansion (absolute seats).
    pub fn vec_of(&self, node: u32) -> [f64; MAX_SEATS] {
        self.nodes[node as usize].vec
    }
    /// The network's utility of `node` for its decider (stored at expansion; 0 before).
    pub fn value0(&self, node: u32) -> f64 {
        let nd = &self.nodes[node as usize];
        nd.vec[nd.player as usize]
    }
    /// The completed-Q policy target at `node` (`completed_q_policy` over its P0, N, W and its
    /// decider's network utility; empty if not expanded).
    pub fn cq_target(&self, node: u32, c_visit: f64, c_scale: f64) -> Vec<f32> {
        self.cq_target_w(node, c_visit, c_scale, 1.0)
    }
    /// `cq_target` with the prior exponent `prior_w` (`completed_q_policy_w`, docs/search.md
    /// 3.13; 1.0 = `cq_target`): the recorded target of a seat's `PolicyTarget::CompletedQ`.
    pub fn cq_target_w(&self, node: u32, c_visit: f64, c_scale: f64, prior_w: f64) -> Vec<f32> {
        let st = self.stats_of(node);
        let p0: Vec<f32> = st.iter().map(|s| s.p0).collect();
        let n: Vec<u32> = st.iter().map(|s| s.n).collect();
        let w: Vec<f64> = st.iter().map(|s| s.w).collect();
        completed_q_policy_w(&p0, &n, &w, self.value0(node), c_visit, c_scale, prior_w)
    }
    /// The search's average win vector at `node` (absolute seats), None before any backup.
    pub fn win(&self, node: u32) -> Option<[f64; MAX_SEATS]> {
        let nd = &self.nodes[node as usize];
        if nd.tot == 0 {
            return None;
        }
        let mut w = [0.0; MAX_SEATS];
        for j in 0..self.n {
            w[j] = nd.wsum[j] / nd.tot as f64;
        }
        Some(w)
    }
    /// Children of (node, option i): (chance key, child node).
    pub fn kids(&self, node: u32, i: usize) -> Vec<(Vec<u64>, u32)> {
        let mut out = Vec::new();
        let so = self.nodes[node as usize].stat_off;
        if so == NIL {
            return out;
        }
        if self.merge.is_some() {
            // (a merged tree: the edge's links, newest first - their signature's key and the node they lead to)
            return self.edge_links(node, i).into_iter().map(|l| (self.link_key(l).to_vec(), self.link_node(l))).collect();
        }
        let mut k = self.stats[so as usize + i].kid;
        while k != NIL {
            out.push((self.node_key(k).to_vec(), k));
            k = self.nodes[k as usize].next;
        }
        out
    }
    /// Chance keys sampled through (node, option i), oldest first (repeats kept).
    pub fn outs(&self, node: u32, i: usize) -> Vec<Vec<u64>> {
        let mut v = Vec::new();
        let so = self.nodes[node as usize].stat_off;
        if so == NIL {
            return v;
        }
        let mut o = self.stats[so as usize + i].outs;
        while o != NIL {
            let r = self.outs[o as usize];
            v.push(self.keys[r.key_off as usize..(r.key_off + r.key_len) as usize].to_vec());
            o = r.next;
        }
        v.reverse();
        v
    }
    /// Nodes stored (incl. garbage above a reused root).
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// The shape of the tree under the root (a read-only diagnostic, docs/search.md 3.15): every node
    /// reachable from the root once, at its shortest depth (root 0). In a fresh search each expanded node
    /// was one walk's leaf, so `depth_hist` is the walks' leaf depths but those that ended at a terminal.
    pub fn shape(&self) -> Shape {
        let mut s = Shape::default();
        let mut depth: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        // (per node reached: the opponent nodes above it, the decider changes above it)
        let mut above: std::collections::HashMap<u32, (u32, u32)> = std::collections::HashMap::new();
        let mut order = vec![self.root];
        depth.insert(self.root, 0);
        above.insert(self.root, (0, 0));
        let mut k = 0;
        while k < order.len() {
            let id = order[k];
            k += 1;
            let d = depth[&id];
            s.nodes += 1;
            s.max_depth = s.max_depth.max(d);
            if !self.expanded(id) {
                continue;
            }
            s.expanded += 1;
            if s.depth_hist.len() <= d as usize {
                s.depth_hist.resize(d as usize + 1, 0);
            }
            s.depth_hist[d as usize] += 1;
            let (oa, sa) = above[&id];
            for (h, x) in [(&mut s.opp_hist, oa), (&mut s.switch_hist, sa)] {
                if h.len() <= x as usize {
                    h.resize(x as usize + 1, 0);
                }
                h[x as usize] += 1;
            }
            let own = self.player(id) == self.viewer;
            for i in 0..self.nodes[id as usize].nacts as usize {
                let n = self.n_of(id, i);
                let kids = self.kid_ids(id, i);
                if n == 0 && kids.is_empty() {
                    continue;
                }
                let mut e = EdgeShape { depth: d, own, n, kids: kids.len() as u32, ..Default::default() };
                self.opp_split(&kids, &mut e);
                for &c in &kids {
                    e.opp_kids += (self.player(c) != self.viewer) as u32;
                    e.expanded_kids += self.expanded(c) as u32;
                    if let std::collections::hash_map::Entry::Vacant(v) = depth.entry(c) {
                        v.insert(d + 1);
                        above.insert(c, (oa + !own as u32, sa + (self.player(c) != self.player(id)) as u32));
                        order.push(c);
                    }
                }
                s.edges.push(e);
            }
        }
        // the main line: the most visited option from the root on (the first maximum on ties)
        let mut id = self.root;
        let mut seen = std::collections::HashSet::new();
        while self.expanded(id) && seen.insert(id) {
            let st = self.stats_of(id);
            let Some(i) = (0..st.len()).max_by(|&a, &b| st[a].n.cmp(&st[b].n).then(b.cmp(&a))) else { break };
            if st[i].n == 0 {
                break;
            }
            let kids = self.kid_ids(id, i);
            let mut e = EdgeShape { depth: s.main_line.len() as u32, own: self.player(id) == self.viewer, n: st[i].n, kids: kids.len() as u32, ..Default::default() };
            self.opp_split(&kids, &mut e);
            for &c in &kids {
                e.opp_kids += (self.player(c) != self.viewer) as u32;
                e.expanded_kids += self.expanded(c) as u32;
            }
            s.main_line.push(e);
            // the edge's most visited child (backups through it: tot)
            let Some(&c) = kids.iter().max_by_key(|&&c| self.nodes[c as usize].tot) else { break };
            id = c;
        }
        s
    }

    /// `EdgeShape`'s other-seat split of an edge's children `kids`: their distinct option lists, chance keys,
    /// backups, and the backups of the most backed-up option list.
    fn opp_split(&self, kids: &[u32], e: &mut EdgeShape) {
        let mut lists: std::collections::HashMap<(usize, &[u16]), u32> = std::collections::HashMap::new();
        let mut keys: std::collections::HashSet<&[u64]> = std::collections::HashSet::new();
        for &c in kids {
            if self.player(c) == self.viewer {
                continue;
            }
            let t = self.nodes[c as usize].tot;
            *lists.entry((self.player(c), self.acts(c))).or_insert(0) += t;
            keys.insert(self.node_key(c));
            e.opp_tot += t;
        }
        e.opp_lists = lists.len() as u32;
        e.opp_keys = keys.len() as u32;
        e.opp_modal_tot = lists.values().copied().max().unwrap_or(0);
    }

    /// The child nodes of (node, option i), newest first.
    fn kid_ids(&self, node: u32, i: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let so = self.nodes[node as usize].stat_off;
        if so == NIL {
            return out;
        }
        if self.merge.is_some() {
            // (a merged tree: the distinct nodes the edge's links lead to)
            for l in self.edge_links(node, i) {
                let c = self.link_node(l);
                if !out.contains(&c) {
                    out.push(c);
                }
            }
            return out;
        }
        let mut k = self.stats[so as usize + i].kid;
        while k != NIL {
            out.push(k);
            k = self.nodes[k as usize].next;
        }
        out
    }

    // ---------------------------------------------------------------- search

    /// PUCT: argmax_i Q_i + c * P_i * sqrt(max(tot, 1)) / (1 + N_i), Q_i = W_i / N_i or the
    /// node's own network utility for its decider when N_i = 0; ties -> lowest index.
    fn select(&self, node: u32, c_puct: f64) -> usize {
        let nd = &self.nodes[node as usize];
        let sq = c_puct * (if nd.tot > 0 { nd.tot as f64 } else { 1.0 }).sqrt();
        let fpu = nd.vec[nd.player as usize];
        let mut best_i = 0;
        let mut best = -1e30;
        for (i, s) in self.stats_of(node).iter().enumerate() {
            let n = s.n;
            let q = if n > 0 { s.w / n as f64 } else { fpu };
            let u = q + sq * s.p as f64 / (1.0 + n as f64);
            if u > best {
                best_i = i;
                best = u;
            }
        }
        best_i
    }

    /// One simulation from `live` (the real game; ignored when `redet` is off). `g` is scratch
    /// space and holds the leaf state when the result is `Sim::Leaf`; `sc.path` receives the
    /// (node, option) pairs walked. Errors are game inconsistencies (an option illegal in a
    /// determinized world, a determinization that changed the root decider's options).
    pub fn simulate(&mut self, live: &G, g: &mut G, rng: &mut Rng, w: &Walk, sc: &mut Scratch) -> Result<Sim, String> {
        self.simulate_with(live, g, rng, w, sc, |_, _, _| None)
    }

    /// `simulate` with a selection hook: at every expanded node of the walk, `hook(tree,
    /// node, g)` (g = the scratch game at that node) may return an option index to take
    /// instead of the PUCT choice (web analysis: forced minimum visits per candidate).
    /// `simulate` is this with a hook that never overrides (monomorphized away).
    pub fn simulate_with<H>(&mut self, live: &G, g: &mut G, rng: &mut Rng, w: &Walk, sc: &mut Scratch, mut hook: H) -> Result<Sim, String>
    where
        H: FnMut(&Self, u32, &G) -> Option<usize>,
    {
        sc.path.clear();
        self.walk_forced = NIL;
        if let Some(m) = self.merge.as_mut() {
            m.grow_node = NIL;
            m.enc_node = NIL;
        }
        g.clone_from(self.base.as_ref().unwrap_or(live));
        if self.redet {
            deal(g, self.viewer, self.belief.as_ref(), self.belief_mix, rng);
            sc.legal.clear();
            g.legal_actions(&mut sc.legal);
            if g.to_move() != Some(self.player(self.root)) || sc.legal.as_slice() != self.acts(self.root) {
                return Err(format!("determinize() changed the root decision: {:?} -> {:?}", self.acts(self.root), sc.legal));
            }
        }
        let mut node = self.root;
        loop {
            if !self.expanded(node) {
                return Ok(Sim::Leaf { node, player: self.player(node) });
            }
            // (a merged tree's node of another seat: subset-armed over this world's options, sc.legal)
            let subset = self.merge.is_some() && self.player(node) != self.viewer;
            let i = match hook(self, node, g) {
                Some(i) => i,
                // (the Gumbel root search, docs/search.md 3.7: sequential halving at the root, PUCT below)
                None if node == self.root && self.gumbel.is_some() => self.gumbel_select(),
                None if subset => self.select_avail(node, w.c_puct, &sc.legal),
                None => {
                    let i = self.select(node, w.c_puct);
                    if self.root_min > 0 && node == self.root {
                        self.force_root(i)
                    } else {
                        i
                    }
                }
            };
            if subset {
                self.mark_avail(node, &sc.legal);
            }
            sc.path.push((node, i as u32));
            let si = self.nodes[node as usize].stat_off as usize + i;
            let st = self.stats[si];
            let a = self.acts(node)[i];
            let mut forced: Option<(u32, u32)> = None;
            if w.chance_k > 0 && st.nouts >= w.chance_k {
                let pick = rng.below(st.nouts) as usize;
                // outs are linked newest first; index from the oldest like Python's list
                let mut o = st.outs;
                for _ in 0..(st.nouts as usize - 1 - pick) {
                    o = self.outs[o as usize].next;
                }
                let r = self.outs[o as usize];
                forced = Some((r.key_off, r.key_len));
            }
            let res = match forced {
                Some((off, len)) => {
                    let f: Vec<u64> = self.keys[off as usize..(off + len) as usize].to_vec();
                    transition(g, a, Some(&f), &mut sc.key, &mut sc.legal)
                }
                None => transition(g, a, None, &mut sc.key, &mut sc.legal),
            };
            if let Err(e) = res {
                return Err(format!("search transition failed: {e} (option {a} of a node of seat {})", self.player(node)));
            }
            if !sc.key.is_empty() && forced.is_none() {
                let key_off = self.keys.len() as u32;
                self.keys.extend_from_slice(&sc.key);
                let id = self.outs.len() as u32;
                self.outs.push(Out { key_off, key_len: sc.key.len() as u32, next: st.outs });
                let s = &mut self.stats[si];
                s.outs = id;
                s.nouts += 1;
            }
            let Some(p) = g.to_move() else {
                if g.deferred() || g.aborted() {
                    // a deferred terminal: a no-option node of the viewer, evaluated once by the
                    // network, its stored value backed up on every later visit (module docs); an
                    // aborted game (no result, `SearchGame::aborted`) is evaluated the same way
                    let v = self.viewer;
                    let h = sig_hash(&sc.key, v, &[]);
                    let k = if self.merge.is_some() {
                        self.deferred_child(node, i, &sc.key, h)
                    } else {
                        match self.find_kid(node, i, &sc.key, v, &[], h) {
                            Some(k) => k,
                            None => {
                                let k = self.push_node(&sc.key, v, &[], h);
                                self.nodes[k as usize].next = self.stats[si].kid;
                                self.stats[si].kid = k;
                                k
                            }
                        }
                    };
                    if !self.expanded(k) {
                        return Ok(Sim::Leaf { node: k, player: v });
                    }
                    let nd = &self.nodes[k as usize];
                    let (util, win) = (nd.vec, nd.wsum);
                    self.backup(&sc.path, &util, &win);
                    return Ok(Sim::Terminal);
                }
                let (util, win) = terminal_values(g, w.margin_w);
                self.backup(&sc.path, &util, &win);
                return Ok(Sim::Terminal);
            };
            let h = sig_hash(&sc.key, p, &sc.legal);
            if self.merge.is_some() {
                // (docs/search.md 3.16: a known link, else the reached observation's node; a growth leaf ends the walk)
                let (k, grow) = self.merged_child(node, i, h, g, sc);
                if grow {
                    return Ok(Sim::Leaf { node: k, player: p });
                }
                node = k;
                continue;
            }
            node = match self.find_kid(node, i, &sc.key, p, &sc.legal, h) {
                Some(k) => k,
                None => {
                    let k = self.push_node(&sc.key, p, &sc.legal, h);
                    self.nodes[k as usize].next = self.stats[si].kid;
                    self.stats[si].kid = k;
                    k
                }
            };
        }
    }

    /// Expand `node` with the network's priors over its options and utility vector.
    pub fn expand(&mut self, node: u32, priors: &[f32], util: &[f64; MAX_SEATS]) {
        if self.merge.as_ref().is_some_and(|m| m.grow_node == node) && self.expanded(node) {
            // (a merged tree's growth leaf: its union grows, this world's priors join the means)
            self.grow(node, priors);
            return;
        }
        let nd = &self.nodes[node as usize];
        debug_assert_eq!(nd.stat_off, NIL, "node expanded twice");
        debug_assert_eq!(priors.len(), nd.nacts as usize);
        let off = self.stats.len() as u32;
        for &p in priors {
            self.stats.push(Stat { p, p0: p, n: 0, w: 0.0, kid: NIL, outs: NIL, nouts: 0 });
        }
        let nd = &mut self.nodes[node as usize];
        nd.stat_off = off;
        nd.vec = *util;
        nd.wsum = [0.0; MAX_SEATS];
        if let Some(m) = self.merge.as_mut() {
            m.expanded(priors);
        }
    }

    /// `expand` that also keeps the leaf's win vector: a deferred terminal's node (no options)
    /// stores it (`wsum`, `tot` = 1) to back it up on later visits (module docs).
    pub fn expand_leaf(&mut self, node: u32, priors: &[f32], util: &[f64; MAX_SEATS], win: &[f64; MAX_SEATS]) {
        self.expand(node, priors, util);
        let nd = &mut self.nodes[node as usize];
        if nd.nacts == 0 {
            nd.wsum = *win;
            nd.tot = 1;
        }
    }

    /// The node is a deferred terminal's (no options).
    pub fn is_deferred_leaf(&self, node: u32) -> bool {
        self.nodes[node as usize].nacts == 0
    }

    /// N_i += 1, W_i += util[decider], tot += 1, wsum += win along the path. The backup of a
    /// forced walk (the last `simulate`'s, `set_root_min`) is also counted as forced.
    pub fn backup(&mut self, path: &[(u32, u32)], util: &[f64; MAX_SEATS], win: &[f64; MAX_SEATS]) {
        if self.walk_forced != NIL {
            let i = self.walk_forced as usize;
            self.walk_forced = NIL;
            debug_assert_eq!(path.first().copied(), Some((self.root, i as u32)), "a forced walk's backup");
            self.forced[i] += 1;
            for j in 0..MAX_SEATS {
                self.forced_win[j] += win[j];
            }
            self.forced_n += 1;
        }
        if let Some(gr) = self.gumbel.as_mut() {
            // (a Gumbel search counts its own root walks: the schedule and the root value)
            if let Some(&(node, i)) = path.first() {
                if node == self.root {
                    let i = i as usize;
                    gr.n[i] += 1;
                    gr.walks += 1;
                    for j in 0..MAX_SEATS {
                        gr.win[i][j] += win[j];
                    }
                }
            }
        }
        for &(node, i) in path {
            let nd = &mut self.nodes[node as usize];
            let si = nd.stat_off as usize + i as usize;
            let pl = nd.player as usize;
            nd.tot += 1;
            for j in 0..MAX_SEATS {
                nd.wsum[j] += win[j];
            }
            let s = &mut self.stats[si];
            s.n += 1;
            s.w += util[pl];
        }
    }

    /// Root Dirichlet noise: P = (1 - eps) P0 + eps Dir(alpha), if >= 2 options.
    pub fn add_noise(&mut self, node: u32, alpha: f64, eps: f64, rng: &mut Rng) {
        let n = self.nodes[node as usize].nacts as usize;
        let so = self.nodes[node as usize].stat_off;
        if n < 2 || so == NIL {
            return;
        }
        let x = noise::dirichlet(alpha, n, rng);
        for (k, s) in self.stats[so as usize..so as usize + n].iter_mut().enumerate() {
            s.p = ((1.0 - eps) * s.p0 as f64 + eps * x[k]) as f32;
        }
    }

    /// Move the root to the subtree reached by real action `a` with real chance key `key` and
    /// next decision (`player`, `legal`). Returns false (tree unchanged) if it does not exist.
    pub fn advance(&mut self, a: u16, key: &[u64], player: usize, legal: &[u16]) -> bool {
        let root = self.root;
        if !self.expanded(root) {
            return false;
        }
        let Some(i) = self.acts(root).iter().position(|&x| x == a) else { return false };
        match self.find_kid(root, i, key, player, legal, sig_hash(key, player, legal)) {
            Some(k) => {
                self.root = k;
                self.viewer = player;
                self.set_root_min(0);
                self.belief = None;
                self.belief_mix = 0.0;
                self.gumbel = None;
                if self.nodes.len() > COMPACT_AT {
                    self.compact();
                }
                true
            }
            None => false,
        }
    }

    /// Copy the subtree under `root` into fresh arrays (drops the garbage above it).
    pub fn compact(&mut self) {
        if self.merge.is_some() {
            // (a DAG: each reachable node once, docs/search.md 3.16)
            return self.compact_merged();
        }
        let mut t: Tree<G> = Tree {
            viewer: self.viewer,
            redet: self.redet,
            n: self.n,
            team: self.team,
            root: 0,
            base: self.base.take(),
            root_min: self.root_min,
            forced: std::mem::take(&mut self.forced),
            forced_win: self.forced_win,
            forced_n: self.forced_n,
            walk_forced: self.walk_forced,
            belief: self.belief.take(),
            belief_mix: self.belief_mix,
            gumbel: self.gumbel.take(),
            ..Default::default()
        };
        let mut map = std::collections::HashMap::new();
        let mut order = vec![self.root];
        let mut k = 0;
        while k < order.len() {
            let id = order[k];
            k += 1;
            let so = self.nodes[id as usize].stat_off;
            if so != NIL {
                for i in 0..self.nodes[id as usize].nacts {
                    let mut c = self.stats[(so + i) as usize].kid;
                    while c != NIL {
                        order.push(c);
                        c = self.nodes[c as usize].next;
                    }
                }
            }
        }
        for (new, &old) in order.iter().enumerate() {
            map.insert(old, new as u32);
        }
        let m = |x: u32| if x == NIL { NIL } else { map[&x] };
        for &old in &order {
            let nd = &self.nodes[old as usize];
            let mut n2 = nd.clone();
            n2.acts_off = t.acts.len() as u32;
            t.acts.extend_from_slice(&self.acts[nd.acts_off as usize..(nd.acts_off + nd.nacts) as usize]);
            n2.key_off = t.keys.len() as u32;
            t.keys.extend_from_slice(&self.keys[nd.key_off as usize..(nd.key_off + nd.key_len) as usize]);
            n2.next = if old == self.root { NIL } else { m(nd.next) };
            if nd.stat_off != NIL {
                n2.stat_off = t.stats.len() as u32;
                for i in 0..nd.nacts {
                    let mut s = self.stats[(nd.stat_off + i) as usize];
                    s.kid = m(s.kid);
                    let mut recs = Vec::new();
                    let mut o = s.outs;
                    while o != NIL {
                        recs.push(self.outs[o as usize]);
                        o = self.outs[o as usize].next;
                    }
                    let mut next = NIL;
                    for r in recs.iter().rev() {
                        let key_off = t.keys.len() as u32;
                        t.keys.extend_from_slice(&self.keys[r.key_off as usize..(r.key_off + r.key_len) as usize]);
                        let id = t.outs.len() as u32;
                        t.outs.push(Out { key_off, key_len: r.key_len, next });
                        next = id;
                    }
                    s.outs = next;
                    t.stats.push(s);
                }
            }
            t.nodes.push(n2);
        }
        *self = t;
    }
}

#[cfg(test)]
mod tests;
