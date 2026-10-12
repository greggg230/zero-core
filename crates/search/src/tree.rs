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
use std::hash::{Hash, Hasher};

use crate::game::{SearchGame, MAX_SEATS};
use crate::noise;
use crate::rng::Rng;

const NIL: u32 = u32::MAX;
/// Node count above which a reused root triggers compaction.
const COMPACT_AT: usize = 1 << 16;

/// Per-option statistics of an expanded node.
#[derive(Clone, Copy, Debug)]
struct Stat {
    p: f32,
    p0: f32,
    n: u32,
    w: f64,
    /// First child node of this edge (linked through `Node::next`), or NIL.
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
    /// evaluates `encode(player)` and calls `expand` + `backup`.
    Leaf { node: u32, player: usize },
    /// The walk ended the game; the result was backed up.
    Terminal,
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
    key.clear();
    legal.clear();
    if let Some(f) = forced {
        g.force_chance(f);
    }
    let r = (|| {
        g.step(a, key)?;
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

/// Network output (relative to the decider) -> (util, win) over absolute seats.
pub fn leaf_values(value_rel: &[f32], margin_rel: &[f32], player: usize, n: usize, margin_w: f64) -> ([f64; MAX_SEATS], [f64; MAX_SEATS]) {
    let mut win = [0.0; MAX_SEATS];
    let mut util = [0.0; MAX_SEATS];
    for j in 0..n {
        let s = (player + j) % n;
        let w = value_rel[j] as f64;
        win[s] = w;
        util[s] = if margin_w != 0.0 { w + margin_w * (margin_rel[j] as f64).tanh() } else { w };
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
    let mut lg: Vec<f64> = (0..k)
        .map(|i| {
            let q = if n[i] > 0 { w[i] / n[i] as f64 } else { v_mix };
            (p0[i] as f64).max(PRIOR_FLOOR).ln() + scale * q
        })
        .collect();
    let m = lg.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut s = 0.0;
    for x in lg.iter_mut() {
        *x = (*x - m).exp();
        s += *x;
    }
    lg.iter().map(|&x| (x / s) as f32).collect()
}

/// (util, win) of a finished game.
pub fn terminal_values<G: SearchGame>(g: &G, margin_w: f64) -> ([f64; MAX_SEATS], [f64; MAX_SEATS]) {
    let r = g.result();
    let n = g.num_players();
    let mut win = [0.0; MAX_SEATS];
    for i in 0..n {
        win[i] = r[i] as f64;
    }
    if margin_w == 0.0 {
        return (win, win);
    }
    let mut util = [0.0; MAX_SEATS];
    for i in 0..n {
        util[i] = win[i] + margin_w * (g.margin(i) as f64).tanh();
    }
    (util, win)
}

#[derive(Clone, Debug)]
pub struct Tree<G: SearchGame> {
    /// The root decider (whose information set the tree searches).
    pub viewer: usize,
    pub redet: bool,
    /// Players in the game.
    pub n: usize,
    pub root: u32,
    nodes: Vec<Node>,
    stats: Vec<Stat>,
    acts: Vec<u16>,
    keys: Vec<u64>,
    outs: Vec<Out>,
    /// `redet == false`: the one determinized world every simulation clones.
    base: Option<G>,
}

impl<G: SearchGame> Default for Tree<G> {
    fn default() -> Self {
        Tree { viewer: 0, redet: true, n: 0, root: 0, nodes: Vec::new(), stats: Vec::new(), acts: Vec::new(), keys: Vec::new(), outs: Vec::new(), base: None }
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
        self.nodes.clear();
        self.stats.clear();
        self.acts.clear();
        self.keys.clear();
        self.outs.clear();
        self.root = self.push_node(&[], viewer, legal, 0);
        self.set_base(game, rng);
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
            g.determinize(self.viewer, rng);
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
        let st = self.stats_of(node);
        let p0: Vec<f32> = st.iter().map(|s| s.p0).collect();
        let n: Vec<u32> = st.iter().map(|s| s.n).collect();
        let w: Vec<f64> = st.iter().map(|s| s.w).collect();
        completed_q_policy(&p0, &n, &w, self.value0(node), c_visit, c_scale)
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
        g.clone_from(self.base.as_ref().unwrap_or(live));
        if self.redet {
            g.determinize(self.viewer, rng);
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
            let i = match hook(self, node, g) {
                Some(i) => i,
                None => self.select(node, w.c_puct),
            };
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
                let (util, win) = terminal_values(g, w.margin_w);
                self.backup(&sc.path, &util, &win);
                return Ok(Sim::Terminal);
            };
            let h = sig_hash(&sc.key, p, &sc.legal);
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
    }

    /// N_i += 1, W_i += util[decider], tot += 1, wsum += win along the path.
    pub fn backup(&mut self, path: &[(u32, u32)], util: &[f64; MAX_SEATS], win: &[f64; MAX_SEATS]) {
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
        let mut t: Tree<G> = Tree { viewer: self.viewer, redet: self.redet, n: self.n, root: 0, base: self.base.take(), ..Default::default() };
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
