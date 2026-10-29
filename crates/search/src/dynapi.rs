//! Object-safe wrappers over a `SearchGame` type, its trees and drivers, so the bindings
//! (crate `zero_py`; a game's Python module `<package>._engine`) can hold any registered game
//! behind one class each: registering a game is one line in the game's `zero_py::engine_module!`.
use std::any::Any;

use crate::driver::{Batch, Driver, DriverCfg, DriverStats, GameResult, InFlight, KeyKind, Record, Select};
use crate::game::{bot_names, SearchGame, MAX_SEATS};
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
}

pub fn game_info<G: SearchGame>() -> GameInfo {
    GameInfo { name: G::NAME, num_actions: G::NUM_ACTIONS, num_features: G::NUM_FEATURES, min_players: G::MIN_PLAYERS, max_players: G::MAX_PLAYERS.min(MAX_SEATS), bots: bot_names::<G>() }
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
    fn encode(&self, viewer: usize) -> Vec<f32>;
    fn rounds(&self) -> u32;
    fn export_state(&self) -> String;
    fn apply_external(&mut self, payload: &str) -> Result<Vec<u64>, String>;
    /// Deferred results and teams (`SearchGame::deferred`, `deferred_payload`, `resolve`,
    /// `team_size`).
    fn deferred(&self) -> bool;
    fn deferred_payload(&self) -> Result<String, String>;
    fn resolve(&mut self, results: &str) -> Result<(), String>;
    fn team_size(&self) -> usize;
    /// A search tree for this game's pending decision.
    fn new_tree(&self, redet: bool, seed: u64) -> Result<Box<dyn DynTree>, String>;
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
    fn deferred_payload(&self) -> Result<String, String> {
        SearchGame::deferred_payload(self)
    }
    fn resolve(&mut self, results: &str) -> Result<(), String> {
        SearchGame::resolve(self, results)
    }
    fn team_size(&self) -> usize {
        SearchGame::team_size(self)
    }
    fn new_tree(&self, redet: bool, seed: u64) -> Result<Box<dyn DynTree>, String> {
        let viewer = SearchGame::to_move(self).ok_or("the game is over")?;
        let mut rng = Rng::new(seed);
        let legal = DynGame::legal(self);
        let t = Tree::new(self, viewer, &legal, redet, &mut rng);
        Ok(Box::new(TreeBox { t, rng, scratch: None, sc: Scratch::default(), leaf: None }))
    }
}

fn downcast<G: SearchGame>(g: &dyn DynGame) -> Result<&G, String> {
    g.as_any().downcast_ref::<G>().ok_or_else(|| format!("expected a {} game", G::NAME))
}

/// A leaf of a Python-driven search: decider, features, legal actions.
pub struct Leaf {
    pub player: usize,
    pub x: Vec<f32>,
    pub acts: Vec<u16>,
}

/// One search tree driven from Python (parity tests, mixed Rust-vs-Python matches).
pub trait DynTree: Send + Sync {
    fn set_base(&mut self, live: &dyn DynGame) -> Result<(), String>;
    /// One simulation: None if it ended in a terminal state (backed up), else the leaf to
    /// evaluate (then `expand`).
    fn simulate(&mut self, live: &dyn DynGame, c_puct: f64, chance_k: u32, margin_w: f64) -> Result<Option<Leaf>, String>;
    /// Evaluate the last leaf: priors over its options, value / margin relative to its decider.
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
    /// The completed-Q policy target at the root (`tree::completed_q_policy`; empty before
    /// expansion).
    fn root_cq_target(&self, c_visit: f64, c_scale: f64) -> Vec<f32>;
    /// The option the driver's move rule `sel` chooses at the root (`Select::choose`; 0 before
    /// expansion): the same code as a driver's non-sampled moves.
    fn root_select(&self, sel: &Select) -> usize;
    /// Root minimum visits from now on (`Tree::set_root_min`, docs/search.md 3.5; 0 = off) and
    /// the forced walks' accounting: per root option, the visit counts without them, the root
    /// win vector without them.
    fn set_root_min(&mut self, k: u32);
    fn root_min(&self) -> u32;
    fn root_forced(&self) -> Vec<u32>;
    fn root_visits_pruned(&self) -> Vec<u32>;
    fn root_win_pruned(&self) -> Option<Vec<f64>>;
    fn redet(&self) -> bool;
    fn edge_kids(&self, i: usize) -> Vec<(Vec<u64>, usize)>;
    fn edge_outs(&self, i: usize) -> Vec<Vec<u64>>;
    fn n_nodes(&self) -> usize;
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
    fn simulate(&mut self, live: &dyn DynGame, c_puct: f64, chance_k: u32, margin_w: f64) -> Result<Option<Leaf>, String> {
        if self.leaf.is_some() {
            return Err("expand the previous leaf first".into());
        }
        let live = downcast::<G>(live)?;
        let g = self.scratch.get_or_insert_with(|| live.clone());
        let w = Walk { c_puct, chance_k, margin_w };
        match self.t.simulate(live, g, &mut self.rng, &w, &mut self.sc)? {
            Sim::Terminal => Ok(None),
            Sim::Leaf { node, player } => {
                self.leaf = Some((node, player));
                let mut x = vec![0.0; G::NUM_FEATURES];
                g.encode(player, &mut x);
                Ok(Some(Leaf { player, x, acts: self.t.acts(node).to_vec() }))
            }
        }
    }
    fn expand(&mut self, priors: &[f32], value: &[f32], margin: &[f32], margin_w: f64) -> Result<(), String> {
        let (node, player) = self.leaf.take().ok_or("no leaf to expand")?;
        if priors.len() != self.t.acts(node).len() {
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
        Ok(self.t.advance(a, key, p, &legal))
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
    fn root_cq_target(&self, c_visit: f64, c_scale: f64) -> Vec<f32> {
        self.t.cq_target(self.t.root, c_visit, c_scale)
    }
    fn root_select(&self, sel: &Select) -> usize {
        let r = self.t.root;
        sel.choose(&self.t.priors0(r), &self.t.visits(r), &self.t.values(r), self.t.value0(r))
    }
    fn set_root_min(&mut self, k: u32) {
        self.t.set_root_min(k)
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
}

/// A driver, any game type.
pub trait DynDriver: Send + Sync {
    fn info(&self) -> GameInfo;
    fn next_batch(&mut self, block: bool) -> Result<Option<Batch>, String>;
    fn submit(&mut self, id: u64, priors: &[f32], value: &[f32], margin: &[f32]) -> Result<(), String>;
    fn submit_external(&mut self, id: u64, payloads: Vec<String>) -> Result<(), String>;
    fn finished(&self) -> bool;
    fn paused(&self) -> bool;
    fn resume(&mut self, quota: u64, net: u32) -> Result<(), String>;
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
    fn submit(&mut self, id: u64, priors: &[f32], value: &[f32], margin: &[f32]) -> Result<(), String> {
        Driver::submit(self, id, priors, value, margin)
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
