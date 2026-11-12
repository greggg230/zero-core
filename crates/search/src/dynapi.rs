//! Object-safe wrappers over a `SearchGame` type, its trees and drivers, so the bindings
//! (crate `zero_py`; a game's Python module `<package>._engine`) can hold any registered game
//! behind one class each: registering a game is one line in the game's `zero_py::engine_module!`.
use std::any::Any;

use crate::driver::{Batch, Driver, DriverCfg, DriverStats, GameResult, InFlight, KeyKind, Record, Select, ROW_BELIEF};
use crate::game::{bot_names, Belief, SearchGame, MAX_SEATS};
use crate::lookahead::{Answers, Line, Look, LookCfg, LookStats, OptionValue, Outcome, RowSink, Step};
use crate::rng::Rng;
use crate::tree::{leaf_values_team, Scratch, Sim, Tree, Walk};

/// Static facts about a game type.
#[derive(Clone, Debug)]
pub struct GameInfo {
    pub name: &'static str,
    pub num_actions: usize,
    pub num_features: usize,
    pub min_players: usize,
    pub max_players: usize,
    pub bots: Vec<&'static str>,
    /// Auxiliary target segments (name, width) (`SearchGame::AUX`; empty: none).
    pub aux: Vec<(&'static str, usize)>,
}

pub fn game_info<G: SearchGame>() -> GameInfo {
    GameInfo { name: G::NAME, num_actions: G::NUM_ACTIONS, num_features: G::NUM_FEATURES, min_players: G::MIN_PLAYERS, max_players: G::MAX_PLAYERS.min(MAX_SEATS), bots: bot_names::<G>(), aux: G::AUX.to_vec() }
}

/// One game, any type.
pub trait DynGame: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn clone_box(&self) -> Box<dyn DynGame>;
    fn info(&self) -> GameInfo;
    fn num_players(&self) -> usize;
    fn to_move(&self) -> Option<usize>;
    fn legal(&self) -> Vec<u16>;
    fn step(&mut self, a: u16) -> Result<Vec<u64>, String>;
    fn force_chance(&mut self, forced: &[u64]);
    fn result(&self) -> Vec<f32>;
    fn margin(&self, seat: usize) -> f32;
    fn score(&self, seat: usize) -> f32;
    fn determinize(&mut self, viewer: usize, seed: u64);
    /// `SearchGame::determinize_with` (docs/search.md 3.8) with the belief `pred` (`aux_width()`
    /// floats), `tilt`, `clip` and the per-target tilts `tilts` ((target, alpha), docs/search.md
    /// 3.8.1; empty: none).
    fn determinize_with(&mut self, viewer: usize, seed: u64, pred: &[f32], tilt: f64, clip: f64, tilts: &[(String, f64)]) -> Result<(), String>;
    fn encode(&self, viewer: usize) -> Vec<f32>;
    fn rounds(&self) -> u32;
    fn export_state(&self) -> String;
    fn apply_external(&mut self, payload: &str) -> Result<Vec<u64>, String>;
    /// Deferred results and teams (`SearchGame::deferred`, `deferred_payload`, `resolve`,
    /// `team_size`).
    fn deferred(&self) -> bool;
    /// Over without a result (`SearchGame::aborted`).
    fn aborted(&self) -> bool;
    fn deferred_payload(&self) -> Result<String, String>;
    fn resolve(&mut self, results: &str) -> Result<(), String>;
    fn team_size(&self) -> usize;
    /// `SearchGame::aux_target` for `viewer` (`aux_width()` floats; none: empty).
    fn aux_target(&self, viewer: usize) -> Vec<f32>;
    /// A search tree for this game's pending decision.
    fn new_tree(&self, redet: bool, seed: u64) -> Result<Box<dyn DynTree>, String>;
    /// The lookahead of this game's pending decision (docs/search.md 3.9; `lookahead::Look::new`):
    /// its worlds given (`worlds`, games of this type at the same decision), or dealt from `seeds`, or
    /// from `seed`'s stream; the root belief as settings to ask for (`belief` (tilt, clip, mix): one
    /// belief row first) or as predictions given (`pred`, with `belief`'s tilt / clip / mix).
    #[allow(clippy::too_many_arguments)]
    fn lookahead(&self, cfg: LookCfg, worlds: Vec<&dyn DynGame>, seeds: Vec<u64>, seed: u64, belief: Option<(f64, f64, f64)>, pred: Option<Vec<f32>>, lines: bool) -> Result<Box<dyn DynLook>, String>;
}

impl<G: SearchGame> DynGame for G {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn DynGame> {
        Box::new(self.clone())
    }
    fn info(&self) -> GameInfo {
        game_info::<G>()
    }
    fn num_players(&self) -> usize {
        SearchGame::num_players(self)
    }
    fn to_move(&self) -> Option<usize> {
        SearchGame::to_move(self)
    }
    fn legal(&self) -> Vec<u16> {
        let mut v = Vec::new();
        self.legal_actions(&mut v);
        v
    }
    fn step(&mut self, a: u16) -> Result<Vec<u64>, String> {
        let mut k = Vec::new();
        SearchGame::step(self, a, &mut k)?;
        Ok(k)
    }
    fn force_chance(&mut self, forced: &[u64]) {
        SearchGame::force_chance(self, forced)
    }
    fn result(&self) -> Vec<f32> {
        SearchGame::result(self)[..SearchGame::num_players(self)].to_vec()
    }
    fn margin(&self, seat: usize) -> f32 {
        SearchGame::margin(self, seat)
    }
    fn score(&self, seat: usize) -> f32 {
        SearchGame::score(self, seat)
    }
    fn determinize(&mut self, viewer: usize, seed: u64) {
        SearchGame::determinize(self, viewer, &mut Rng::new(seed))
    }
    fn determinize_with(&mut self, viewer: usize, seed: u64, pred: &[f32], tilt: f64, clip: f64, tilts: &[(String, f64)]) -> Result<(), String> {
        let b = belief_of::<G>(pred, tilt, clip, tilts)?;
        SearchGame::determinize_with(self, viewer, &mut Rng::new(seed), &b);
        Ok(())
    }
    fn encode(&self, viewer: usize) -> Vec<f32> {
        let mut v = vec![0.0; G::NUM_FEATURES];
        SearchGame::encode(self, viewer, &mut v);
        v
    }
    fn rounds(&self) -> u32 {
        SearchGame::rounds(self)
    }
    fn export_state(&self) -> String {
        SearchGame::export_state(self)
    }
    fn apply_external(&mut self, payload: &str) -> Result<Vec<u64>, String> {
        let mut k = Vec::new();
        SearchGame::apply_external(self, payload, &mut k)?;
        Ok(k)
    }
    fn deferred(&self) -> bool {
        SearchGame::to_move(self).is_none() && SearchGame::deferred(self)
    }
    fn aborted(&self) -> bool {
        SearchGame::to_move(self).is_none() && SearchGame::aborted(self)
    }
    fn deferred_payload(&self) -> Result<String, String> {
        SearchGame::deferred_payload(self)
    }
    fn resolve(&mut self, results: &str) -> Result<(), String> {
        SearchGame::resolve(self, results)
    }
    fn team_size(&self) -> usize {
        SearchGame::team_size(self)
    }
    fn aux_target(&self, viewer: usize) -> Vec<f32> {
        let mut v = vec![0.0; G::aux_width()];
        SearchGame::aux_target(self, viewer, &mut v);
        v
    }
    fn new_tree(&self, redet: bool, seed: u64) -> Result<Box<dyn DynTree>, String> {
        let viewer = SearchGame::to_move(self).ok_or("the game is over")?;
        let mut rng = Rng::new(seed);
        let legal = DynGame::legal(self);
        let t = Tree::new(self, viewer, &legal, redet, &mut rng);
        Ok(Box::new(TreeBox { t, rng, scratch: None, sc: Scratch::default(), leaf: None }))
    }
    fn lookahead(&self, cfg: LookCfg, worlds: Vec<&dyn DynGame>, seeds: Vec<u64>, seed: u64, belief: Option<(f64, f64, f64)>, pred: Option<Vec<f32>>, lines: bool) -> Result<Box<dyn DynLook>, String> {
        let viewer = SearchGame::to_move(self).ok_or("the game is over")?;
        let ws = worlds.into_iter().map(|w| downcast::<G>(w).cloned()).collect::<Result<Vec<G>, String>>()?;
        if !ws.is_empty() && ws.len() != cfg.worlds as usize {
            return Err(format!("lookahead: {} worlds given, the settings say {}", ws.len(), cfg.worlds));
        }
        let mut l = Look::new(cfg, self, viewer, belief, ws, seeds, lines)?;
        if let Some(p) = pred {
            let (t, c, _) = belief.ok_or("lookahead: belief predictions without the belief's tilt / clip / mix")?;
            l.set_belief(belief_of::<G>(&p, t, c, &[])?)?;
        }
        Ok(Box::new(LookBox { l, rng: Rng::new(seed), buf: RowsBuf { n: SearchGame::num_players(self), ..Default::default() }, waiting: false }))
    }
}

fn downcast<G: SearchGame>(g: &dyn DynGame) -> Result<&G, String> {
    g.as_any().downcast_ref::<G>().ok_or_else(|| format!("expected a {} game", G::NAME))
}

/// A `Belief` from the bindings' arguments, checked (docs/search.md 3.8): `aux_width()` finite
/// predictions >= 0, a tilt > 0 (or a per-target tilt > 0, 3.8.1) and the clip / mix rules of
/// `NnSeat::check_belief`.
fn belief_of<G: SearchGame>(pred: &[f32], tilt: f64, clip: f64, tilts: &[(String, f64)]) -> Result<Belief, String> {
    crate::driver::NnSeat::check_belief(tilt, clip, 0.0)?;
    crate::driver::NnSeat::check_belief_tilts(tilts, G::AUX)?;
    let mut seat = crate::driver::NnSeat::eval(0, 1);
    seat.belief_tilt = tilt;
    seat.belief_tilts = tilts.to_vec();
    if !seat.belief_on() {
        return Err(format!("a belief's tilt {tilt}: > 0, or a per-target tilt > 0 (a search without a belief deals uniformly)"));
    }
    if pred.len() != G::aux_width() || G::aux_width() == 0 {
        return Err(format!("a belief of {} values: {} has {} auxiliary target values (SearchGame::AUX)", pred.len(), G::NAME, G::aux_width()));
    }
    if let Some(x) = pred.iter().find(|x| !x.is_finite() || **x < 0.0) {
        return Err(format!("a belief's prediction {x}: finite and >= 0"));
    }
    Ok(Belief { pred: pred.to_vec(), tilt, clip, tilts: seat.target_tilts(G::AUX) })
}

/// A leaf of a Python-driven search: decider, features, legal actions; with root-view values
/// (docs/search.md 3.6) the features of the tree's viewer when another seat decides the leaf.
pub struct Leaf {
    pub player: usize,
    pub x: Vec<f32>,
    pub acts: Vec<u16>,
    pub vx: Option<Vec<f32>>,
}

/// One search tree driven from Python (parity tests, mixed Rust-vs-Python matches).
pub trait DynTree: Send + Sync {
    fn set_base(&mut self, live: &dyn DynGame) -> Result<(), String>;
    /// One simulation: None if it ended in a terminal state (backed up), else the leaf to
    /// evaluate (then `expand`). `root_view`: a leaf of another seat than the viewer also gets
    /// the viewer's features (`Leaf::vx`), whose value `expand` then takes.
    fn simulate(&mut self, live: &dyn DynGame, c_puct: f64, chance_k: u32, margin_w: f64, root_view: bool) -> Result<Option<Leaf>, String>;
    /// Evaluate the last leaf: priors over its options, value / margin relative to its decider
    /// (to the viewer when the leaf came with `vx`).
    fn expand(&mut self, priors: &[f32], value: &[f32], margin: &[f32], margin_w: f64) -> Result<(), String>;
    fn add_noise(&mut self, alpha: f64, eps: f64, seed: Option<u64>);
    /// Move the root to the child for real action `a`, chance key `key` and the decision of
    /// `after` (the game after the move). False = no such subtree.
    fn advance(&mut self, a: u16, key: &[u64], after: &dyn DynGame) -> Result<bool, String>;
    fn root_acts(&self) -> Vec<u16>;
    fn root_player(&self) -> usize;
    fn root_visits(&self) -> Vec<u32>;
    fn root_priors(&self) -> Vec<f32>;
    fn root_priors0(&self) -> Vec<f32>;
    fn root_values(&self) -> Vec<f64>;
    fn root_tot(&self) -> u32;
    fn root_expanded(&self) -> bool;
    fn root_win(&self) -> Option<Vec<f64>>;
    /// The network's utility of the root for its decider (0 before expansion).
    fn root_value0(&self) -> f64;
    /// The completed-Q policy target at the root (`tree::completed_q_policy_w` with the prior
    /// exponent `prior_w`, 1.0 = `completed_q_policy`; empty before expansion).
    fn root_cq_target(&self, c_visit: f64, c_scale: f64, prior_w: f64) -> Vec<f32>;
    /// The option the driver's move rule `sel` chooses at the root (`Select::choose`; 0 before
    /// expansion): the same code as a driver's non-sampled moves.
    fn root_select(&self, sel: &Select) -> usize;
    /// Root minimum visits from now on (`Tree::set_root_min`, docs/search.md 3.5; 0 = off) and
    /// the forced walks' accounting: per root option, the visit counts without them, the root
    /// win vector without them.
    fn set_root_min(&mut self, k: u32);
    /// The root's belief from now on (`Tree::set_belief`, docs/search.md 3.8; None = uniform worlds):
    /// predictions, tilt, clip, the uniform share `mix`, the per-target tilts `tilts` (docs/search.md 3.8.1;
    /// empty: none); `live` re-deals a tree without `redet`.
    fn set_belief(&mut self, pred: Option<&[f32]>, tilt: f64, clip: f64, mix: f64, tilts: &[(String, f64)], live: &dyn DynGame) -> Result<(), String>;
    /// The current search's belief predictions (None: uniform worlds).
    fn belief(&self) -> Option<Vec<f32>>;
    fn root_min(&self) -> u32;
    fn root_forced(&self) -> Vec<u32>;
    fn root_visits_pruned(&self) -> Vec<u32>;
    fn root_win_pruned(&self) -> Option<Vec<f64>>;
    /// The Gumbel root search from now on (`Tree::set_gumbel`, docs/search.md 3.7) or off
    /// (`clear_gumbel`); its move (`gumbel_choose`: None when off), root value (`gumbel_q`), this
    /// search's walks per root option, its noise and its schedule.
    fn set_gumbel(&mut self, m: u32, budget: u32, noise: &[f64], c_visit: f64, c_scale: f64) -> Result<(), String>;
    fn clear_gumbel(&mut self);
    fn gumbel_on(&self) -> bool;
    fn gumbel_choose(&self) -> Option<usize>;
    fn gumbel_q(&self) -> Option<Vec<f64>>;
    fn gumbel_visits(&self) -> Vec<u32>;
    fn gumbel_noise(&self) -> Vec<f64>;
    fn gumbel_schedule(&self) -> Vec<u32>;
    fn gumbel_win(&self) -> Vec<Vec<f64>>;
    fn redet(&self) -> bool;
    /// The tree's viewer (the root decider).
    fn viewer(&self) -> usize;
    fn edge_kids(&self, i: usize) -> Vec<(Vec<u64>, usize)>;
    fn edge_outs(&self, i: usize) -> Vec<Vec<u64>>;
    fn n_nodes(&self) -> usize;
    /// Information-set merging (`Tree::set_merge`, docs/search.md 3.16): a fresh tree only; whether it is on; its
    /// counters (None when off).
    fn set_merge(&mut self, on: bool) -> Result<(), String>;
    fn merge_on(&self) -> bool;
    fn merge_stats(&self) -> Option<crate::tree::MergeStats>;
    /// The tree's shape under its root (`Tree::shape`, a diagnostic: docs/search.md 3.15).
    fn shape(&self) -> crate::tree::Shape;
    /// The (node, option) pairs the last simulation walked (its depth: a leaf's, or a terminal's).
    fn path_len(&self) -> usize;
}

struct TreeBox<G: SearchGame> {
    t: Tree<G>,
    rng: Rng,
    scratch: Option<G>,
    sc: Scratch,
    leaf: Option<(u32, usize)>,
}

impl<G: SearchGame> DynTree for TreeBox<G> {
    fn set_base(&mut self, live: &dyn DynGame) -> Result<(), String> {
        let g = downcast::<G>(live)?;
        self.t.set_base(g, &mut self.rng);
        Ok(())
    }
    fn simulate(&mut self, live: &dyn DynGame, c_puct: f64, chance_k: u32, margin_w: f64, root_view: bool) -> Result<Option<Leaf>, String> {
        if self.leaf.is_some() {
            return Err("expand the previous leaf first".into());
        }
        let live = downcast::<G>(live)?;
        let g = self.scratch.get_or_insert_with(|| live.clone());
        let w = Walk { c_puct, chance_k, margin_w };
        match self.t.simulate(live, g, &mut self.rng, &w, &mut self.sc)? {
            Sim::Terminal => Ok(None),
            Sim::Leaf { node, player } => {
                let mut x = vec![0.0; G::NUM_FEATURES];
                g.encode(player, &mut x);
                let viewer = self.t.viewer;
                let vx = (root_view && player != viewer).then(|| {
                    let mut v = vec![0.0; G::NUM_FEATURES];
                    g.encode(viewer, &mut v);
                    v
                });
                // (the value's seat: the viewer with a value-only row, else the decider)
                self.leaf = Some((node, if vx.is_some() { viewer } else { player }));
                // (a merged tree's growth leaf: this world's options, docs/search.md 3.16)
                Ok(Some(Leaf { player, x, acts: self.t.leaf_acts(node).to_vec(), vx }))
            }
        }
    }
    fn expand(&mut self, priors: &[f32], value: &[f32], margin: &[f32], margin_w: f64) -> Result<(), String> {
        let (node, player) = self.leaf.take().ok_or("no leaf to expand")?;
        if priors.len() != self.t.leaf_acts(node).len() {
            self.leaf = Some((node, player));
            return Err("priors length != options".into());
        }
        let mut v = [0.0f32; MAX_SEATS];
        let mut m = [0.0f32; MAX_SEATS];
        for (i, x) in value.iter().take(MAX_SEATS).enumerate() {
            v[i] = *x;
        }
        for (i, x) in margin.iter().take(MAX_SEATS).enumerate() {
            m[i] = *x;
        }
        let (util, win) = leaf_values_team(&v, &m, player, self.t.n, margin_w, self.t.team);
        self.t.expand_leaf(node, priors, &util, &win);
        self.t.backup(&self.sc.path, &util, &win);
        Ok(())
    }
    fn add_noise(&mut self, alpha: f64, eps: f64, seed: Option<u64>) {
        let r = self.t.root;
        match seed {
            Some(s) => self.t.add_noise(r, alpha, eps, &mut Rng::new(s)),
            None => self.t.add_noise(r, alpha, eps, &mut self.rng),
        }
    }
    fn advance(&mut self, a: u16, key: &[u64], after: &dyn DynGame) -> Result<bool, String> {
        let g = downcast::<G>(after)?;
        let Some(p) = SearchGame::to_move(g) else { return Ok(false) };
        let legal = DynGame::legal(g);
        // (a merged tree finds the new root by the real observation too: docs/search.md 3.16)
        Ok(self.t.advance_in(a, key, p, &legal, g))
    }
    fn root_acts(&self) -> Vec<u16> {
        self.t.acts(self.t.root).to_vec()
    }
    fn root_player(&self) -> usize {
        self.t.player(self.t.root)
    }
    fn root_visits(&self) -> Vec<u32> {
        self.t.visits(self.t.root)
    }
    fn root_priors(&self) -> Vec<f32> {
        self.t.priors(self.t.root)
    }
    fn root_priors0(&self) -> Vec<f32> {
        self.t.priors0(self.t.root)
    }
    fn root_values(&self) -> Vec<f64> {
        self.t.values(self.t.root)
    }
    fn root_tot(&self) -> u32 {
        self.t.tot(self.t.root)
    }
    fn root_expanded(&self) -> bool {
        self.t.expanded(self.t.root)
    }
    fn root_win(&self) -> Option<Vec<f64>> {
        self.t.win(self.t.root).map(|w| w[..self.t.n].to_vec())
    }
    fn root_value0(&self) -> f64 {
        self.t.value0(self.t.root)
    }
    fn root_cq_target(&self, c_visit: f64, c_scale: f64, prior_w: f64) -> Vec<f32> {
        self.t.cq_target_w(self.t.root, c_visit, c_scale, prior_w)
    }
    fn root_select(&self, sel: &Select) -> usize {
        let r = self.t.root;
        sel.choose(&self.t.priors0(r), &self.t.visits(r), &self.t.values(r), self.t.value0(r))
    }
    fn set_root_min(&mut self, k: u32) {
        self.t.set_root_min(k)
    }
    fn set_belief(&mut self, pred: Option<&[f32]>, tilt: f64, clip: f64, mix: f64, tilts: &[(String, f64)], live: &dyn DynGame) -> Result<(), String> {
        let g = downcast::<G>(live)?;
        let b = match pred {
            Some(p) => {
                crate::driver::NnSeat::check_belief(tilt, clip, mix)?;
                Some(belief_of::<G>(p, tilt, clip, tilts)?)
            }
            None => None,
        };
        self.t.set_belief(b, mix, g, &mut self.rng);
        Ok(())
    }
    fn belief(&self) -> Option<Vec<f32>> {
        self.t.belief().map(|b| b.pred.clone())
    }
    fn root_min(&self) -> u32 {
        self.t.root_min()
    }
    fn root_forced(&self) -> Vec<u32> {
        self.t.root_forced().to_vec()
    }
    fn root_visits_pruned(&self) -> Vec<u32> {
        self.t.root_visits_pruned()
    }
    fn root_win_pruned(&self) -> Option<Vec<f64>> {
        self.t.root_win_pruned().map(|w| w[..self.t.n].to_vec())
    }
    fn set_gumbel(&mut self, m: u32, budget: u32, noise: &[f64], c_visit: f64, c_scale: f64) -> Result<(), String> {
        let k = self.t.acts(self.t.root).len();
        if !noise.is_empty() && noise.len() != k {
            return Err(format!("set_gumbel: {} noise values for {k} root options", noise.len()));
        }
        crate::driver::RootSearch::parse("gumbel", m, 0.0, c_visit, c_scale)?;
        self.t.set_gumbel(m, budget, noise, c_visit, c_scale);
        Ok(())
    }
    fn clear_gumbel(&mut self) {
        self.t.clear_gumbel()
    }
    fn gumbel_on(&self) -> bool {
        self.t.gumbel_on()
    }
    fn gumbel_choose(&self) -> Option<usize> {
        self.t.gumbel_choose()
    }
    fn gumbel_q(&self) -> Option<Vec<f64>> {
        self.t.gumbel_q().map(|w| w[..self.t.n].to_vec())
    }
    fn gumbel_visits(&self) -> Vec<u32> {
        self.t.gumbel_visits().to_vec()
    }
    fn gumbel_noise(&self) -> Vec<f64> {
        self.t.gumbel_noise().to_vec()
    }
    fn gumbel_schedule(&self) -> Vec<u32> {
        self.t.gumbel_schedule().to_vec()
    }
    fn gumbel_win(&self) -> Vec<Vec<f64>> {
        self.t.gumbel_win().iter().map(|w| w[..self.t.n].to_vec()).collect()
    }
    fn viewer(&self) -> usize {
        self.t.viewer
    }
    fn redet(&self) -> bool {
        self.t.redet
    }
    fn edge_kids(&self, i: usize) -> Vec<(Vec<u64>, usize)> {
        self.t.kids(self.t.root, i).into_iter().map(|(k, c)| (k, self.t.player(c))).collect()
    }
    fn edge_outs(&self, i: usize) -> Vec<Vec<u64>> {
        self.t.outs(self.t.root, i)
    }
    fn n_nodes(&self) -> usize {
        self.t.n_nodes()
    }
    fn set_merge(&mut self, on: bool) -> Result<(), String> {
        if self.leaf.is_some() {
            return Err("set_merge: a fresh tree only (a leaf is pending)".into());
        }
        self.t.set_merge(on)
    }
    fn merge_on(&self) -> bool {
        self.t.merge_on()
    }
    fn merge_stats(&self) -> Option<crate::tree::MergeStats> {
        self.t.merge_stats()
    }
    fn shape(&self) -> crate::tree::Shape {
        self.t.shape()
    }
    fn path_len(&self) -> usize {
        self.sc.path.len()
    }
}

/// One round of a lookahead's rows for Python (`DynLook::rows`): a network batch as the driver's
/// (`Batch`'s arrays: features [b * F], option segments, player, players, row kinds), borrowed from
/// the lookahead's own buffers (kept across rounds: a fresh 150 MB of features a round cost more in
/// page faults than the lookahead's own work).
pub struct LookRows<'a> {
    pub b: usize,
    pub x: &'a [f32],
    pub opt_off: &'a [i32],
    pub opt_action: &'a [i32],
    pub player: &'a [i32],
    pub n_players: &'a [i32],
    pub extra: &'a [u8],
}

/// One lookahead driven from Python (the position API, docs/search.md 3.9).
pub trait DynLook: Send + Sync {
    /// Run to the next round (its rows: `rows`) -> its rows' count, None when the decision is made.
    fn step(&mut self) -> Result<Option<usize>, String>;
    /// The last round's rows (`step`).
    fn rows(&self) -> LookRows<'_>;
    /// The answers of the last round: priors over its options, values [b x MAX_SEATS] relative to
    /// each row's player, its belief rows' auxiliary outputs in row order.
    fn submit(&mut self, priors: &[f32], value: &[f32], aux: &[f32]) -> Result<(), String>;
    fn outcome(&self) -> Option<Outcome>;
    fn options(&mut self) -> Result<Vec<OptionValue>, String>;
    fn stats(&self) -> LookStats;
    fn line(&mut self, world: usize, act: u16) -> Result<Option<Line>, String>;
    fn worlds(&self) -> Vec<Box<dyn DynGame>>;
    fn belief(&self) -> Option<Vec<f32>>;
    /// The settings of the stage that ran (`LookCfg::to_json`).
    fn cfg_json(&self) -> String;
}

/// The rows of a round in Python's arrays (`DynLook::step`).
#[derive(Default)]
struct RowsBuf {
    x: Vec<f32>,
    nopt: Vec<i32>,
    opt_action: Vec<i32>,
    player: Vec<i32>,
    n: usize,
    extra: Vec<u8>,
    opt_off: Vec<i32>,
    n_players: Vec<i32>,
}

impl RowsBuf {
    fn clear(&mut self) {
        self.x.clear();
        self.nopt.clear();
        self.opt_action.clear();
        self.player.clear();
        self.extra.clear();
        self.opt_off.clear();
        self.n_players.clear();
    }
}

impl<G: SearchGame> RowSink<G> for RowsBuf {
    fn push_row(&mut self, viewer: usize, opts: &[u16], kind: u8) -> &mut [f32] {
        self.nopt.push(opts.len() as i32);
        self.opt_action.extend(opts.iter().map(|&a| a as i32));
        self.player.push(viewer as i32);
        self.extra.push(kind);
        let off = self.x.len();
        self.x.resize(off + G::NUM_FEATURES, 0.0);
        &mut self.x[off..]
    }
    fn pop_row(&mut self) {
        let k = self.nopt.pop().unwrap_or(0) as usize;
        self.opt_action.truncate(self.opt_action.len() - k);
        self.player.pop();
        self.extra.pop();
        self.x.truncate(self.x.len() - G::NUM_FEATURES);
    }
}

struct LookBox<G: SearchGame> {
    l: Look<G>,
    rng: Rng,
    /// The rows of the last round (kept across rounds) and whether they wait for their answers.
    buf: RowsBuf,
    waiting: bool,
}

impl<G: SearchGame> DynLook for LookBox<G> {
    fn step(&mut self) -> Result<Option<usize>, String> {
        if self.waiting {
            return Err("lookahead: submit the last rows' answers first".into());
        }
        self.buf.clear();
        match self.l.step(&mut self.buf, &mut self.rng)? {
            Step::Done => Ok(None),
            Step::Rows(b) => {
                let buf = &mut self.buf;
                buf.opt_off.push(0);
                let mut o = 0;
                for &k in &buf.nopt {
                    o += k;
                    buf.opt_off.push(o);
                }
                buf.n_players.resize(b, buf.n as i32);
                self.waiting = true;
                Ok(Some(b))
            }
        }
    }
    fn rows(&self) -> LookRows<'_> {
        let b = &self.buf;
        LookRows { b: b.extra.len(), x: &b.x, opt_off: &b.opt_off, opt_action: &b.opt_action, player: &b.player, n_players: &b.n_players, extra: &b.extra }
    }
    fn submit(&mut self, priors: &[f32], value: &[f32], aux: &[f32]) -> Result<(), String> {
        if !self.waiting {
            return Err("lookahead: no rows to answer".into());
        }
        let (opt_off, extra) = (&self.buf.opt_off, &self.buf.extra);
        let b = extra.len();
        let nopt = *opt_off.last().unwrap_or(&0) as usize;
        if priors.len() != nopt || value.len() != b * MAX_SEATS {
            return Err(format!("lookahead answers: priors {} (want {nopt}), value {} (want {})", priors.len(), value.len(), b * MAX_SEATS));
        }
        let w = G::aux_width();
        let mut aux_row = vec![u32::MAX; b];
        let mut k = 0;
        for (r, &e) in extra.iter().enumerate() {
            if e == ROW_BELIEF {
                aux_row[r] = k;
                k += 1;
            }
        }
        if aux.len() != k as usize * w {
            return Err(format!("lookahead answers: {} auxiliary values for {k} belief rows of {w}", aux.len()));
        }
        if let Some(x) = aux.iter().find(|x| !x.is_finite() || **x < 0.0) {
            return Err(format!("lookahead answers: a root belief output {x} (finite and >= 0)"));
        }
        self.l.answer(&Answers { base: 0, opt_off, priors, value, aux, aux_row: &aux_row, aux_width: w })?;
        self.waiting = false;
        Ok(())
    }
    fn outcome(&self) -> Option<Outcome> {
        self.l.outcome()
    }
    fn options(&mut self) -> Result<Vec<OptionValue>, String> {
        self.l.options()
    }
    fn stats(&self) -> LookStats {
        self.l.stats().clone()
    }
    fn line(&mut self, world: usize, act: u16) -> Result<Option<Line>, String> {
        self.l.line(world, act)
    }
    fn worlds(&self) -> Vec<Box<dyn DynGame>> {
        self.l.worlds().iter().map(|w| Box::new(w.clone()) as Box<dyn DynGame>).collect()
    }
    fn belief(&self) -> Option<Vec<f32>> {
        self.l.belief().map(|b| b.pred.clone())
    }
    fn cfg_json(&self) -> String {
        self.l.cfg().to_json()
    }
}

/// A driver, any game type.
pub trait DynDriver: Send + Sync {
    fn info(&self) -> GameInfo;
    fn next_batch(&mut self, block: bool) -> Result<Option<Batch>, String>;
    /// A network batch's answer; `aux`: its root belief rows' auxiliary outputs (`Driver::submit_aux`).
    fn submit(&mut self, id: u64, priors: &[f32], value: &[f32], margin: &[f32], aux: &[f32]) -> Result<(), String>;
    fn submit_external(&mut self, id: u64, payloads: Vec<String>) -> Result<(), String>;
    fn finished(&self) -> bool;
    fn paused(&self) -> bool;
    fn resume(&mut self, quota: u64, net: u32) -> Result<(), String>;
    /// A rolling driver's first game id (`Driver::set_first_gid`).
    fn set_first_gid(&mut self, gid: u32) -> Result<(), String>;
    fn end_phase(&mut self);
    fn take_results(&mut self) -> Vec<GameResult>;
    fn take_samples(&mut self) -> Vec<Record>;
    /// Deferred results and queued setups (`Driver::take_deferred`, `resolve`, `pending`,
    /// `enqueue`, `queued`).
    fn take_deferred(&mut self) -> Vec<(u32, String)>;
    fn resolve(&mut self, gid: u32, results: &str) -> Result<(), String>;
    fn pending(&self) -> usize;
    fn enqueue(&mut self, tag: &str, spec: &str) -> Result<(), String>;
    fn queued(&self) -> usize;
    fn stats(&self) -> DriverStats;
    fn in_flight(&self) -> InFlight;
    /// Evaluator keys in index order, and whether each is external.
    fn keys(&self) -> Vec<(String, bool)>;
}

impl<G: SearchGame> DynDriver for Driver<G> {
    fn info(&self) -> GameInfo {
        game_info::<G>()
    }
    fn next_batch(&mut self, block: bool) -> Result<Option<Batch>, String> {
        Driver::next_batch(self, block)
    }
    fn submit(&mut self, id: u64, priors: &[f32], value: &[f32], margin: &[f32], aux: &[f32]) -> Result<(), String> {
        Driver::submit_aux(self, id, priors, value, margin, aux)
    }
    fn submit_external(&mut self, id: u64, payloads: Vec<String>) -> Result<(), String> {
        Driver::submit_external(self, id, payloads)
    }
    fn finished(&self) -> bool {
        Driver::finished(self)
    }
    fn paused(&self) -> bool {
        Driver::paused(self)
    }
    fn resume(&mut self, quota: u64, net: u32) -> Result<(), String> {
        Driver::resume(self, quota, net)
    }
    fn set_first_gid(&mut self, gid: u32) -> Result<(), String> {
        Driver::set_first_gid(self, gid)
    }
    fn end_phase(&mut self) {
        Driver::end_phase(self)
    }
    fn take_results(&mut self) -> Vec<GameResult> {
        Driver::take_results(self)
    }
    fn take_samples(&mut self) -> Vec<Record> {
        Driver::take_samples(self)
    }
    fn take_deferred(&mut self) -> Vec<(u32, String)> {
        Driver::take_deferred(self)
    }
    fn resolve(&mut self, gid: u32, results: &str) -> Result<(), String> {
        Driver::resolve(self, gid, results)
    }
    fn pending(&self) -> usize {
        Driver::pending(self)
    }
    fn enqueue(&mut self, tag: &str, spec: &str) -> Result<(), String> {
        Driver::enqueue(self, tag, spec)
    }
    fn queued(&self) -> usize {
        Driver::queued(self)
    }
    fn stats(&self) -> DriverStats {
        Driver::stats(self)
    }
    fn in_flight(&self) -> InFlight {
        Driver::in_flight(self)
    }
    fn keys(&self) -> Vec<(String, bool)> {
        self.cfg().keys.iter().map(|(k, t)| (k.clone(), *t == KeyKind::External)).collect()
    }
}

/// A new game of type G (bindings).
pub fn boxed_game<G: SearchGame>(spec: &str, players: usize, seed: u64) -> Result<Box<dyn DynGame>, String> {
    let s = G::parse_spec(spec)?;
    Ok(Box::new(G::new_game(&s, players, seed)?))
}

/// A new driver for game type G (bindings).
pub fn boxed_driver<G: SearchGame>(cfg: DriverCfg) -> Result<Box<dyn DynDriver>, String> {
    Ok(Box::new(Driver::<G>::new(cfg)?))
}
