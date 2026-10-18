//! Object-safe wrappers over a `SearchGame` type, its trees and drivers, so the bindings
//! (crate `zero_py`; a game's Python module `<package>._engine`) can hold any registered game
//! behind one class each: registering a game is one line in the game's `zero_py::engine_module!`.
use std::any::Any;

use crate::driver::{Batch, Driver, DriverCfg, DriverStats, GameResult, InFlight, KeyKind, Record};
use crate::game::{bot_names, SearchGame, MAX_SEATS};
use crate::rng::Rng;
use crate::tree::{leaf_values, Scratch, Sim, Tree, Walk};

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
    /// `step` + the external-format moves it completed (`SearchGame::step_external`).
    fn step_external(&mut self, a: u16) -> Result<(Vec<u64>, Vec<String>), String>;
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
    fn step_external(&mut self, a: u16) -> Result<(Vec<u64>, Vec<String>), String> {
        let mut k = Vec::new();
        let mut m = Vec::new();
        SearchGame::step_external(self, a, &mut k, &mut m)?;
        Ok((k, m))
    }
    fn new_tree(&self, redet: bool, seed: u64) -> Result<Box<dyn DynTree>, String> {
        let viewer = SearchGame::to_move(self).ok_or("the game is over")?;
        let mut rng = Rng::new(seed);
        let legal = DynGame::legal(self);
        let t = Tree::new(self, viewer, &legal, redet, &mut rng);
        Ok(Box::new(TreeBox { t, rng, scratch: None, sc: Scratch::default(), leaf: None, batch: Vec::new(), vl: 0.0 }))
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
    /// Up to `k` simulations whose leaves are evaluated together (one network call for one
    /// game: a live move): each walk to a leaf puts virtual loss `vloss` on its path
    /// (`Tree::apply_vloss`) so the next walks of the batch spread out; a walk that ends the
    /// game is backed up at once; a walk reaching a leaf already in the batch is dropped
    /// (after `k` such collisions the batch closes early). -> (the distinct leaves,
    /// simulations done = leaves + game-ending walks, <= k). Then `expand_batch` with their
    /// evaluations, in order. k = 1 is exactly `simulate` + `expand`.
    fn simulate_batch(&mut self, live: &dyn DynGame, k: usize, c_puct: f64, chance_k: u32, margin_w: f64, vloss: f64) -> Result<(Vec<Leaf>, u32), String>;
    /// Evaluate the leaves of the last `simulate_batch`: per leaf its priors, value and margin
    /// (relative to its decider). The virtual loss is removed and the real values backed up.
    fn expand_batch(&mut self, priors: &[Vec<f32>], value: &[Vec<f32>], margin: &[Vec<f32>], margin_w: f64) -> Result<(), String>;
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
    /// The leaves of a `simulate_batch` awaiting `expand_batch`: (node, decider, path).
    batch: Vec<(u32, usize, Vec<(u32, u32)>)>,
    vl: f64,
}

impl<G: SearchGame> DynTree for TreeBox<G> {
    fn set_base(&mut self, live: &dyn DynGame) -> Result<(), String> {
        let g = downcast::<G>(live)?;
        self.t.set_base(g, &mut self.rng);
        Ok(())
    }
    fn simulate(&mut self, live: &dyn DynGame, c_puct: f64, chance_k: u32, margin_w: f64) -> Result<Option<Leaf>, String> {
        if self.leaf.is_some() || !self.batch.is_empty() {
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
        let (util, win) = leaf_values(&v, &m, player, self.t.n, margin_w);
        self.t.expand(node, priors, &util);
        self.t.backup(&self.sc.path, &util, &win);
        Ok(())
    }
    fn simulate_batch(&mut self, live: &dyn DynGame, k: usize, c_puct: f64, chance_k: u32, margin_w: f64, vloss: f64) -> Result<(Vec<Leaf>, u32), String> {
        if self.leaf.is_some() || !self.batch.is_empty() {
            return Err("expand the previous leaves first".into());
        }
        let live = downcast::<G>(live)?;
        let g = self.scratch.get_or_insert_with(|| live.clone());
        let w = Walk { c_puct, chance_k, margin_w };
        self.vl = vloss;
        let mut leaves = Vec::new();
        let (mut done, mut collisions) = (0u32, 0usize);
        while (done as usize) < k && collisions < k {
            match self.t.simulate(live, g, &mut self.rng, &w, &mut self.sc)? {
                Sim::Terminal => done += 1,
                Sim::Leaf { node, player } => {
                    if self.batch.iter().any(|b| b.0 == node) {
                        collisions += 1;
                        continue;
                    }
                    self.t.apply_vloss(&self.sc.path, vloss);
                    let mut x = vec![0.0; G::NUM_FEATURES];
                    g.encode(player, &mut x);
                    leaves.push(Leaf { player, x, acts: self.t.acts(node).to_vec() });
                    self.batch.push((node, player, self.sc.path.clone()));
                    done += 1;
                }
            }
        }
        Ok((leaves, done))
    }
    fn expand_batch(&mut self, priors: &[Vec<f32>], value: &[Vec<f32>], margin: &[Vec<f32>], margin_w: f64) -> Result<(), String> {
        let n = self.batch.len();
        if priors.len() != n || value.len() != n || margin.len() != n {
            return Err(format!("expand_batch: {n} leaves, {} / {} / {} evaluations", priors.len(), value.len(), margin.len()));
        }
        for (j, (node, _, _)) in self.batch.iter().enumerate() {
            if priors[j].len() != self.t.acts(*node).len() {
                return Err(format!("expand_batch: leaf {j}: {} priors for {} options", priors[j].len(), self.t.acts(*node).len()));
            }
        }
        let batch = std::mem::take(&mut self.batch);
        for (j, (node, player, path)) in batch.into_iter().enumerate() {
            let mut v = [0.0f32; MAX_SEATS];
            let mut m = [0.0f32; MAX_SEATS];
            for (i, x) in value[j].iter().take(MAX_SEATS).enumerate() {
                v[i] = *x;
            }
            for (i, x) in margin[j].iter().take(MAX_SEATS).enumerate() {
                m[i] = *x;
            }
            let (util, win) = leaf_values(&v, &m, player, self.t.n, margin_w);
            self.t.revert_vloss(&path, self.vl);
            self.t.expand(node, &priors[j], &util);
            self.t.backup(&path, &util, &win);
        }
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

/// A game of type G from an external state (`SearchGame::import_state`; bindings).
pub fn boxed_import<G: SearchGame>(spec: &str, state: &str) -> Result<Box<dyn DynGame>, String> {
    let s = G::parse_spec(spec)?;
    Ok(Box::new(G::import_state(&s, state)?))
}

/// A new driver for game type G (bindings).
pub fn boxed_driver<G: SearchGame>(cfg: DriverCfg) -> Result<Box<dyn DynDriver>, String> {
    Ok(Box::new(Driver::<G>::new(cfg)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::toy::ToyGame;

    /// Deterministic stand-in network: priors from a hash of the actions, value from the
    /// features (a function of the leaf only).
    fn net(l: &Leaf) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let raw: Vec<f32> = l.acts.iter().map(|&a| 1.0 + ((a as u32).wrapping_mul(2654435761) >> 28) as f32).collect();
        let s: f32 = raw.iter().sum();
        let h: f32 = l.x.iter().enumerate().map(|(i, v)| v * (((i as u32).wrapping_mul(40503) % 7) as f32 - 3.0)).sum();
        let w = 1.0 / (1.0 + (-h / 8.0).exp());
        (raw.iter().map(|x| x / s).collect(), vec![w, 1.0 - w], vec![0.0, 0.0])
    }

    fn mid_game(seed: u64, steps: usize) -> Box<dyn DynGame> {
        let mut g = boxed_game::<ToyGame>("{}", 2, seed).unwrap();
        let mut rng = Rng::new(seed);
        for _ in 0..steps {
            if g.to_move().is_none() {
                break;
            }
            let l = g.legal();
            g.step(l[rng.below(l.len() as u32) as usize]).unwrap();
        }
        g
    }

    fn batched(g: &dyn DynGame, sims: u32, k: usize, vloss: f64, seed: u64) -> Box<dyn DynTree> {
        let mut t = g.new_tree(true, seed).unwrap();
        let mut done = 0;
        while done < sims {
            let (leaves, d) = t.simulate_batch(g, k.min((sims - done) as usize), 1.0, 2, 0.0, vloss).unwrap();
            let ev: Vec<_> = leaves.iter().map(net).collect();
            t.expand_batch(&ev.iter().map(|e| e.0.clone()).collect::<Vec<_>>(), &ev.iter().map(|e| e.1.clone()).collect::<Vec<_>>(), &ev.iter().map(|e| e.2.clone()).collect::<Vec<_>>(), 0.0).unwrap();
            done += d;
            assert!(d >= 1, "a batch made no progress");
        }
        t
    }

    #[test]
    fn batch_of_one_is_the_sequential_search() {
        for seed in 1..6u64 {
            let g = mid_game(seed, 6);
            if g.to_move().is_none() {
                continue;
            }
            let mut a = g.new_tree(true, 99 + seed).unwrap();
            for _ in 0..300 {
                if let Some(l) = a.simulate(&*g, 1.0, 2, 0.0).unwrap() {
                    let (p, v, m) = net(&l);
                    a.expand(&p, &v, &m, 0.0).unwrap();
                }
            }
            let b = batched(&*g, 300, 1, 0.0, 99 + seed);
            assert_eq!(a.root_visits(), b.root_visits(), "seed {seed}");
            assert_eq!(a.root_values(), b.root_values(), "seed {seed}");
        }
    }

    #[test]
    fn batches_leave_no_virtual_loss_and_spread_out() {
        for seed in 1..6u64 {
            let g = mid_game(seed, 4);
            if g.to_move().is_none() {
                continue;
            }
            for &vl in &[0.0, -5.0] {
                let t = batched(&*g, 512, 16, vl, 7 + seed);
                let n: u32 = t.root_visits().iter().sum();
                assert_eq!(n, t.root_tot(), "seed {seed} vl {vl}");
                assert_eq!(n, 512 - 1, "seed {seed} vl {vl}: every simulation but the root's expansion backs up through the root");
                for (w, &nv) in t.root_values().iter().zip(t.root_visits().iter()) {
                    if nv > 0 {
                        let q = w / nv as f64;
                        assert!((-1e-9..=1.0 + 1e-9).contains(&q), "seed {seed} vl {vl}: Q {q} outside [0, 1]: virtual loss left");
                    }
                }
            }
        }
    }
}
