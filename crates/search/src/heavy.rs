//! Heavy decisions' reveal probe (docs/search.md 3.11): does an option of the pending decision lead,
//! within a short line of play, to a transition that reveals hidden information to the decider?
//!
//! A probe works on one world determinized for the decider (`SearchGame::determinize(viewer)`), so
//! the keys `step` pushes are the decider's view (another player's private facts masked, in a game
//! that masks them); the same world for every line, so lines are comparable key for key. A **line**
//! from root option `a`: `step(a)`, then at most `horizon` steps in all (every `step` counts, the
//! single-option ones too), stopping when the game is over or a step ends in a later round
//! (`SearchGame::rounds`: what a new round reveals - MTG's next draw step, a toy's next round card -
//! comes with time, whatever the option; that step's keys are not read). On the way:
//!
//! * a decision of one option: stepped (the search's transition auto-steps it too);
//! * another seat's decision: its passive option (`SearchGame::passive_action`: MTG's pass priority,
//!   no attackers, no blocks, decline) or, without one, its first option - the line asks what the
//!   decider's own play reveals, not what an opponent might choose to show;
//! * the decider's minor decision (`SearchGame::minor_decision`: MTG's mana payments): its first
//!   option;
//! * the decider's decision without a passive option (a forced choice: a required target, a mode):
//!   every option branches into a line of its own (depth unchanged) - part of the play in progress;
//! * the decider's decision with a passive option: while the line has deviations left (`depth`, the
//!   root option being the first), every other option branches with one fewer; the line itself
//!   continues with the passive option (so a follow-up action may come once the stack has resolved,
//!   MTG: cast Mishra's Bauble, then activate it, at depth 2).
//!
//! A key **reveals** when `SearchGame::reveals(key, viewer)` says so (default: every key; MTG: the
//! decider's own draws and looks, another player's card made public, coins and dice - never its
//! single-option bookkeeping keys, never another player's masked private draw). The root's passive
//! option is never probed: its line (no deviations) is the **baseline** - what the game reveals
//! anyway if the decider does nothing (another player's discard, a trigger) - and a line reveals only
//! keys beyond the baseline's (multiset). An option reveals when one of its lines does. The probe
//! stops after `lines` lines in all (none found then: not revealing) and, with `first`, at the first
//! revealing option. The driver probes only a self-play seat's decisions with `sims_heavy`, neither
//! minor nor light; nothing here changes the game, the tree or any random stream of the driver.
use crate::game::SearchGame;
use crate::rng::Rng;

/// The probe's settings (`NnSeat::heavy_depth`, `heavy_horizon`, `heavy_lines`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProbeCfg {
    /// Deviations from the passive play per line, the root option included (>= 1).
    pub depth: u32,
    /// Steps per line (>= 1).
    pub horizon: u32,
    /// Lines per probe, every branch included (>= 1).
    pub lines: u32,
}

impl Default for ProbeCfg {
    fn default() -> ProbeCfg {
        ProbeCfg { depth: 2, horizon: 12, lines: 64 }
    }
}

impl ProbeCfg {
    pub fn check(&self) -> Result<(), String> {
        if !(1..=4).contains(&self.depth) {
            return Err(format!("heavy_depth {}: 1..=4 (deviations per line, the root option the first)", self.depth));
        }
        if !(1..=64).contains(&self.horizon) {
            return Err(format!("heavy_horizon {}: 1..=64 steps", self.horizon));
        }
        if !(1..=4096).contains(&self.lines) {
            return Err(format!("heavy_lines {}: 1..=4096", self.lines));
        }
        Ok(())
    }
}

/// A probe's answer: per root option (the decision's legal order) whether it reveals (false for the
/// options not probed), and its cost.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProbeOut {
    pub reveal: Vec<bool>,
    /// Root options probed (the candidates, less the passive one).
    pub probed: u32,
    /// Lines walked (the baseline's included) and steps taken.
    pub lines: u32,
    pub steps: u32,
    /// The line budget ran out.
    pub exhausted: bool,
}

impl ProbeOut {
    pub fn any(&self) -> bool {
        self.reveal.iter().any(|&r| r)
    }
}

/// The revealing keys of a line so far (a small multiset).
#[derive(Clone, Default)]
struct Seen(Vec<(u64, u32)>);

impl Seen {
    fn add(&mut self, k: u64) -> u32 {
        for e in self.0.iter_mut() {
            if e.0 == k {
                e.1 += 1;
                return e.1;
            }
        }
        self.0.push((k, 1));
        1
    }
    fn count(&self, k: u64) -> u32 {
        self.0.iter().find(|e| e.0 == k).map_or(0, |e| e.1)
    }
}

struct Probe<'a, G: SearchGame> {
    viewer: usize,
    round: u32,
    cfg: &'a ProbeCfg,
    base: Seen,
    lines: u32,
    steps: u32,
    exhausted: bool,
    key: Vec<u64>,
    legal: Vec<u16>,
    _g: std::marker::PhantomData<G>,
}

/// What a step did: the game ended or went to a later round (the line stops), revealed a key beyond
/// the baseline, or neither.
enum Stepped {
    Stop,
    Reveal,
    Go,
}

impl<G: SearchGame> Probe<'_, G> {
    fn step(&mut self, g: &mut G, a: u16, seen: &mut Seen) -> Result<Stepped, String> {
        self.steps += 1;
        self.key.clear();
        g.step(a, &mut self.key).map_err(|e| format!("heavy probe: option {a}: {e}"))?;
        if g.rounds() != self.round {
            return Ok(Stepped::Stop);
        }
        let mut hit = false;
        for i in 0..self.key.len() {
            let k = self.key[i];
            if g.reveals(k, self.viewer) && seen.add(k) > self.base.count(k) {
                hit = true;
            }
        }
        if hit {
            return Ok(Stepped::Reveal);
        }
        Ok(if g.to_move().is_none() { Stepped::Stop } else { Stepped::Go })
    }

    /// A line from `a` in `g` (consumed) with `dev` deviations left after it and `left` steps:
    /// whether it (or a line it branches into) reveals.
    fn line(&mut self, mut g: G, a: u16, dev: u32, mut left: u32, mut seen: Seen) -> Result<bool, String> {
        if self.lines >= self.cfg.lines {
            self.exhausted = true;
            return Ok(false);
        }
        self.lines += 1;
        let mut a = a;
        loop {
            match self.step(&mut g, a, &mut seen)? {
                Stepped::Reveal => return Ok(true),
                Stepped::Stop => return Ok(false),
                Stepped::Go => {}
            }
            left -= 1;
            if left == 0 {
                return Ok(false);
            }
            let p = g.to_move().unwrap();
            self.legal.clear();
            g.legal_actions(&mut self.legal);
            if self.legal.is_empty() {
                return Err(format!("heavy probe: seat {p} to move without legal actions"));
            }
            if self.legal.len() == 1 {
                a = self.legal[0];
                continue;
            }
            let passive = g.passive_action(&self.legal).filter(|x| self.legal.contains(x));
            if p != self.viewer {
                a = passive.unwrap_or(self.legal[0]);
                continue;
            }
            if g.minor_decision() {
                a = self.legal[0];
                continue;
            }
            let opts = self.legal.clone();
            match passive {
                None => {
                    // a forced choice: every option is part of the play in progress
                    for &b in &opts {
                        if self.line(g.clone(), b, dev, left, seen.clone())? {
                            return Ok(true);
                        }
                    }
                    return Ok(false);
                }
                Some(x) => {
                    if dev > 0 {
                        for &b in opts.iter().filter(|&&b| b != x) {
                            if self.line(g.clone(), b, dev - 1, left, seen.clone())? {
                                return Ok(true);
                            }
                        }
                    }
                    a = x;
                }
            }
        }
    }

    /// The revealing keys of the passive line from the root (no deviations, `horizon` steps).
    fn baseline(&mut self, mut g: G, x: u16) -> Result<Seen, String> {
        self.lines += 1;
        let mut seen = Seen::default();
        let mut a = x;
        let mut left = self.cfg.horizon;
        loop {
            self.steps += 1;
            self.key.clear();
            g.step(a, &mut self.key).map_err(|e| format!("heavy probe: passive option {a}: {e}"))?;
            if g.rounds() != self.round {
                return Ok(seen);
            }
            for i in 0..self.key.len() {
                if g.reveals(self.key[i], self.viewer) {
                    seen.add(self.key[i]);
                }
            }
            left -= 1;
            let Some(p) = g.to_move() else { return Ok(seen) };
            if left == 0 {
                return Ok(seen);
            }
            self.legal.clear();
            g.legal_actions(&mut self.legal);
            if self.legal.is_empty() {
                return Err(format!("heavy probe: seat {p} to move without legal actions"));
            }
            let passive = g.passive_action(&self.legal).filter(|x| self.legal.contains(x));
            a = if self.legal.len() == 1 || (p == self.viewer && g.minor_decision()) { self.legal[0] } else { passive.unwrap_or(self.legal[0]) };
        }
    }
}

/// Probe the pending decision of `live` (its decider `viewer`, legal actions `legal` in their order):
/// for each root option with `cand[i]` (all when `cand` is empty), whether a line from it reveals
/// hidden information to the decider (module docs). `rng` deals the probe's world (a stream of the
/// caller's: the driver derives one per decision, so the probe never touches the search's streams).
/// `first`: stop at the first revealing option.
pub fn probe<G: SearchGame>(live: &G, viewer: usize, legal: &[u16], cand: &[bool], cfg: &ProbeCfg, rng: &mut Rng, first: bool) -> Result<ProbeOut, String> {
    let mut out = ProbeOut { reveal: vec![false; legal.len()], ..ProbeOut::default() };
    if live.to_move() != Some(viewer) || legal.len() < 2 {
        return Ok(out);
    }
    let mut w = live.clone();
    w.determinize(viewer, rng);
    let mut pr: Probe<'_, G> = Probe { viewer, round: w.rounds(), cfg, base: Seen::default(), lines: 0, steps: 0, exhausted: false, key: Vec::new(), legal: Vec::new(), _g: std::marker::PhantomData };
    let passive = w.passive_action(legal).filter(|x| legal.contains(x));
    if let Some(x) = passive {
        pr.base = pr.baseline(w.clone(), x)?;
    }
    for (i, &a) in legal.iter().enumerate() {
        if Some(a) == passive || (!cand.is_empty() && !cand[i]) {
            continue;
        }
        out.probed += 1;
        if pr.line(w.clone(), a, cfg.depth - 1, cfg.horizon, Seen::default())? {
            out.reveal[i] = true;
            if first {
                break;
            }
        }
    }
    out.lines = pr.lines;
    out.steps = pr.steps;
    out.exhausted = pr.exhausted;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peek_game::*;

    fn probe_of(depth: u32, horizon: u32, lines: u32) -> ProbeOut {
        let g = Peek::new_game(&(), 2, 3).unwrap();
        let mut legal = Vec::new();
        g.legal_actions(&mut legal);
        probe(&g, 0, &legal, &[], &ProbeCfg { depth, horizon, lines }, &mut Rng::new(9), false).unwrap()
    }

    #[test]
    fn the_probe_finds_the_revealing_lines() {
        // legal [PASS, PEEK, GUESS0, GUESS1, SLOW, ROUND]: PASS is the baseline (never probed; its
        // public discard is no revelation for the others); PEEK reveals through the forced choice
        // (AT_COIN) and its passive pass (depth 1); SLOW needs a follow-up action (depth 2); ROUND's
        // revelation comes with the next round (never)
        let o = probe_of(1, 12, 64);
        assert_eq!(o.reveal, vec![false, true, false, false, false, false]);
        assert_eq!(o.probed, 5);
        assert!(!o.exhausted);
        let o = probe_of(2, 12, 64);
        assert_eq!(o.reveal, vec![false, true, false, false, true, false]);
        // a horizon too short for PEEK's line (PEEK, AT_COIN, PASS: three steps)
        assert_eq!(probe_of(1, 2, 64).reveal, vec![false; 6]);
        assert_eq!(probe_of(1, 3, 64).reveal[1], true);
        // the line budget: the baseline and one line, PEEK's (its branch AT_COIN not walked)
        let o = probe_of(1, 12, 2);
        assert!(o.exhausted && !o.any());
    }

    #[test]
    fn candidates_first_and_the_world_stay_the_callers() {
        let g = Peek::new_game(&(), 2, 3).unwrap();
        let mut legal = Vec::new();
        g.legal_actions(&mut legal);
        let cfg = ProbeCfg { depth: 2, horizon: 12, lines: 64 };
        // only SLOW a candidate
        let o = probe(&g, 0, &legal, &[true, false, false, false, true, false], &cfg, &mut Rng::new(1), false).unwrap();
        assert_eq!((o.reveal.clone(), o.probed), (vec![false, false, false, false, true, false], 1));
        // first: stops at PEEK
        let o = probe(&g, 0, &legal, &[], &cfg, &mut Rng::new(1), true).unwrap();
        assert_eq!(o.reveal, vec![false, true, false, false, false, false]);
        assert_eq!(o.probed, 1);
        // the live game is untouched (no observer, the same coin), the answer is the same in every world
        assert!(g.observer.is_none() && g.phase == 0);
        for s in 0..16 {
            assert_eq!(probe(&g, 0, &legal, &[], &cfg, &mut Rng::new(s), false).unwrap().reveal, vec![false, true, false, false, true, false]);
        }
        // a seat that is not to move, or a one-option decision: nothing probed
        assert_eq!(probe(&g, 1, &legal, &[], &cfg, &mut Rng::new(1), false).unwrap().probed, 0);
        assert_eq!(probe(&g, 0, &legal[..1], &[], &cfg, &mut Rng::new(1), false).unwrap().probed, 0);
    }

    /// Seat 0 at the look (phase 2: PASS looks at the coin), every simulation through PASS (its prior
    /// 1, values constant): the coins its walks reveal, per value.
    fn coins_through_the_look(stratified: bool, sims: usize, seed: u64) -> [usize; 2] {
        use crate::tree::{leaf_values, Scratch, Sim, Tree, Walk};
        let mut g = Peek::new_game(&(), 2, seed).unwrap();
        g.step(PEEK, &mut Vec::new()).unwrap();
        g.step(AT_COIN, &mut Vec::new()).unwrap();
        let mut legal = Vec::new();
        g.legal_actions(&mut legal);
        let mut rng = Rng::new(seed);
        let mut t = Tree::new(&g, 0, &legal, true, &mut rng);
        t.set_stratified(stratified);
        assert_eq!(t.stratified(), stratified);
        let (mut sg, mut sc) = (g.clone(), Scratch::default());
        let w = Walk { c_puct: 1.0, chance_k: 0, margin_w: 0.0 };
        for _ in 0..sims {
            if let Sim::Leaf { node, player } = t.simulate(&g, &mut sg, &mut rng, &w, &mut sc).unwrap() {
                let n = t.acts(node).len();
                let pri: Vec<f32> = if node == t.root { (0..n).map(|i| (i == 0) as u8 as f32).collect() } else { vec![1.0 / n as f32; n] };
                let (u, win) = leaf_values(&[0.5, 0.5], &[0.0, 0.0], player, 2, 0.0);
                t.expand_leaf(node, &pri, &u, &win);
                t.backup(&sc.path, &u, &win);
            }
        }
        assert_eq!(t.visits(t.root)[0] as usize, sims - 1, "every walk through the look");
        let mut c = [0usize; 2];
        for k in t.outs(t.root, 0) {
            c[(k[0] & 1) as usize] += 1;
        }
        c
    }

    #[test]
    fn stratified_worlds_reveal_the_units_without_replacement() {
        // 63 walks through the look (the root's own evaluation takes the first simulation): stratified,
        // the coin's two units alternate within each pass of the permutation - never more than one apart
        // (a walk takes the next unit of the simulation's pass); independent worlds drift
        let mut drift = 0;
        for seed in 0..8 {
            let s = coins_through_the_look(true, 64, seed);
            assert_eq!(s[0] + s[1], 63);
            assert!(s[0].abs_diff(s[1]) <= 2, "stratified {s:?} (seed {seed})");
            let p = coins_through_the_look(false, 64, seed);
            assert_eq!(p[0] + p[1], 63);
            drift = drift.max(p[0].abs_diff(p[1]));
        }
        assert!(drift > 2, "independent worlds never drifted ({drift})");
    }

    #[test]
    fn probe_settings_are_checked() {
        assert!(ProbeCfg::default().check().is_ok());
        for bad in [ProbeCfg { depth: 0, ..ProbeCfg::default() }, ProbeCfg { depth: 5, ..ProbeCfg::default() }, ProbeCfg { horizon: 0, ..ProbeCfg::default() }, ProbeCfg { lines: 0, ..ProbeCfg::default() }] {
            assert!(bad.check().is_err(), "{bad:?}");
        }
    }
}
