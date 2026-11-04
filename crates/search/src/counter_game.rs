//! A test game for information-set merging (docs/search.md 3.16): "cast or hold, counter or allow" - MTG Zero's
//! Daze / Force of Will in miniature, where the other seat's option list depends on its hidden hand.
//!
//! Seat 1 holds two hidden counters, Daze and Force, each with probability 1/2 (`determinize(0)` deals the unused
//! ones again; seat 0 has nothing hidden). `ROUNDS` rounds: seat 0 `CAST`s (0) or `HOLD`s (1). A cast is answered
//! by seat 1 with `ALLOW` (2), `DAZE` (3) if it holds an unused Daze, `FORCE` (4) if it holds an unused Force; a
//! seat 1 holding neither has `ALLOW` alone and is auto-stepped by the search - so after a cast the next decision
//! is seat 1's (with one of three option lists) in three worlds of four and seat 0's next round in the fourth. A
//! counter is spent and public once used. A resolved cast scores 1 for seat 0, a hold 0.4, a countered cast 0; at
//! the end P(seat 0 wins) = 0.2 + 0.3 score (clamped to [0, 1]).
//!
//! The spec (`CounterSpec`, JSON `{"rounds": R, "forget": bool}`; `{}` = 2 rounds, not forgetful) sets the rounds and
//! whether the features forget the round and the score: a forgetful game's observations repeat along a line (the
//! merged tree's path check, its aliases).
//!
//! The tree of before keys seat 1's node by its option list (up to three nodes under one cast), and seat 0's next
//! round under each of them and under the auto-stepped world: four nodes for one information set of seat 0. The
//! merged tree holds one seat-1 node with the union of the options and one seat-0 node reached from two parents at
//! two depths.
use crate::game::{SearchGame, MAX_SEATS};
use crate::rng::Rng;

pub const CAST: u16 = 0;
pub const HOLD: u16 = 1;
pub const ALLOW: u16 = 2;
pub const DAZE: u16 = 3;
pub const FORCE: u16 = 4;
/// The default number of rounds.
pub const ROUNDS: u8 = 2;

/// Rounds, and whether `encode` leaves out the round and the score.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CounterSpec {
    pub rounds: u8,
    pub forget: bool,
}

#[derive(Clone, Debug)]
pub struct Counter {
    pub spec: CounterSpec,
    /// Seat 1's counters: [Daze, Force] held, and whether each was used (public).
    pub held: [bool; 2],
    pub used: [bool; 2],
    pub round: u8,
    /// A cast waits for seat 1's answer.
    pub pending: bool,
    /// Seat 0's points (tenths: a hold 4, a resolved cast 10).
    pub score10: u32,
}

impl Counter {
    pub fn with_hand(daze: bool, force: bool) -> Counter {
        Counter::with_spec(CounterSpec { rounds: ROUNDS, forget: false }, daze, force)
    }

    pub fn with_spec(spec: CounterSpec, daze: bool, force: bool) -> Counter {
        Counter { spec, held: [daze, force], used: [false; 2], round: 0, pending: false, score10: 0 }
    }

    fn over(&self) -> bool {
        self.round >= self.spec.rounds
    }
}

impl SearchGame for Counter {
    type Spec = CounterSpec;
    const NAME: &'static str = "counter";
    const NUM_ACTIONS: usize = 5;
    const NUM_FEATURES: usize = 8;
    const MIN_PLAYERS: usize = 2;
    const MAX_PLAYERS: usize = 2;

    fn parse_spec(json: &str) -> Result<CounterSpec, String> {
        let v: serde_json::Value = serde_json::from_str(if json.trim().is_empty() { "{}" } else { json }).map_err(|e| format!("counter spec: {e}"))?;
        let rounds = v.get("rounds").and_then(|x| x.as_u64()).unwrap_or(ROUNDS as u64);
        if !(1..=8).contains(&rounds) {
            return Err(format!("counter spec: rounds {rounds} (1..=8)"));
        }
        Ok(CounterSpec { rounds: rounds as u8, forget: v.get("forget").and_then(|x| x.as_bool()).unwrap_or(false) })
    }
    fn new_game(spec: &CounterSpec, players: usize, seed: u64) -> Result<Counter, String> {
        if players != 2 {
            return Err("counter: 2 players".into());
        }
        let mut r = Rng::new(seed);
        Ok(Counter::with_spec(*spec, r.below(2) == 1, r.below(2) == 1))
    }
    fn num_players(&self) -> usize {
        2
    }
    fn to_move(&self) -> Option<usize> {
        if self.over() {
            None
        } else {
            Some(self.pending as usize)
        }
    }
    fn legal_actions(&self, out: &mut Vec<u16>) {
        if self.over() {
            return;
        }
        if !self.pending {
            out.extend_from_slice(&[CAST, HOLD]);
            return;
        }
        out.push(ALLOW);
        for (c, a) in [(0, DAZE), (1, FORCE)] {
            if self.held[c] && !self.used[c] {
                out.push(a);
            }
        }
    }
    fn step(&mut self, a: u16, _chance: &mut Vec<u64>) -> Result<(), String> {
        if self.over() {
            return Err("counter: the game is over".into());
        }
        match (self.pending, a) {
            (false, CAST) => self.pending = true,
            (false, HOLD) => {
                self.score10 += 4;
                self.round += 1;
            }
            (true, ALLOW) => {
                self.score10 += 10;
                self.pending = false;
                self.round += 1;
            }
            (true, DAZE) | (true, FORCE) => {
                let c = (a - DAZE) as usize;
                if !self.held[c] || self.used[c] {
                    return Err(format!("counter: seat 1 cannot play {a}"));
                }
                self.used[c] = true;
                self.pending = false;
                self.round += 1;
            }
            _ => return Err(format!("counter: action {a} is not legal here")),
        }
        Ok(())
    }
    fn result(&self) -> [f32; MAX_SEATS] {
        let p = (0.2 + 0.03 * self.score10 as f32).min(1.0);
        let mut r = [0.0; MAX_SEATS];
        r[0] = p;
        r[1] = 1.0 - p;
        r
    }
    fn margin(&self, _seat: usize) -> f32 {
        0.0
    }
    fn score(&self, seat: usize) -> f32 {
        if seat == 0 {
            self.score10 as f32 / 10.0
        } else {
            0.0
        }
    }
    fn determinize(&mut self, viewer: usize, rng: &mut Rng) {
        if viewer == 0 {
            for c in 0..2 {
                if !self.used[c] {
                    self.held[c] = rng.below(2) == 1;
                }
            }
        }
    }
    fn encode(&self, viewer: usize, out: &mut [f32]) {
        out[0] = (viewer == 0) as u8 as f32;
        if !self.spec.forget {
            out[1] = (self.spec.rounds - self.round) as f32 / 8.0;
            out[2] = self.score10 as f32 / 10.0;
        }
        out[3] = self.pending as u8 as f32;
        out[4] = self.used[0] as u8 as f32;
        out[5] = self.used[1] as u8 as f32;
        if viewer == 1 {
            out[6] = (self.held[0] && !self.used[0]) as u8 as f32;
            out[7] = (self.held[1] && !self.used[1]) as u8 as f32;
        }
    }
}

/// The stand-in network on `Counter` features and the row's options: priors by option (CAST 3, HOLD 2, ALLOW 2,
/// DAZE 1, FORCE 1, normalised over the row's options; uniform without options) and the value relative to the
/// row's viewer: seat 0's expected points (each cast still to come worth 0.6, less 0.4 per counter seat 1 holds -
/// known to seat 1, 0.5 each to seat 0) through the result's formula.
pub fn counter_eval(x: &[f32], opts: &[u16]) -> (Vec<f32>, [f32; MAX_SEATS]) {
    let w = |a: u16| match a {
        CAST => 3.0,
        HOLD | ALLOW => 2.0,
        _ => 1.0,
    };
    let s: f32 = opts.iter().map(|&a| w(a)).sum();
    let pri = opts.iter().map(|&a| w(a) / s).collect();
    let viewer0 = x[0] != 0.0;
    let left = x[1] * 8.0;
    let unused = 2.0 - x[4] - x[5];
    let counters = if viewer0 { 0.5 * unused } else { x[6] + x[7] };
    let pts = x[2] + 0.6 * left - 0.4 * counters.min(left);
    let v0 = (0.2 + 0.3 * pts).clamp(0.02, 0.98);
    let v = if viewer0 { v0 } else { 1.0 - v0 };
    let mut val = [0.0; MAX_SEATS];
    val[0] = v;
    val[1] = 1.0 - v;
    (pri, val)
}
