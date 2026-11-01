//! A test game for the lookahead (docs/search.md 3.9): "peek, then guess". A hidden coin (0 / 1)
//! that seat 0 cannot see; seat 1 knows it. Seat 0 first PEEKs (sees the coin; a revelation keyed as
//! chance) or GOes on, then pays its mana - a minor decision (TAP_A / TAP_B, `minor_decision`) -, then
//! seat 1 BLOCKs or PASSes, then seat 0 GUESSes the coin. Seat 0 wins with the payoff p = (0.9 after a
//! peek, else 1.0) x (0.8 if blocked, else 1.0) when the guess is right (result [p, 1 - p]), else
//! loses ([0, 1]).
//!
//! Per world (perfect information) the coin is known anyway: GO then the right guess is best, and
//! looking is worth nothing (it costs 0.1). Per information set seat 0's guess after GO is ONE choice
//! for both coins (its observation is the same): 1/2 of the payoff; after PEEK the coin splits the
//! groups and the guess is always right: the value of information. Turns (`rounds`, `active_seat`):
//! seat 0's first move and payment are turn 1, seat 1's reply turn 2, the guess turn 3.
//!
//! `peek_eval` is the stand-in network on `encode(viewer)`: priors BLOCK 0.25 / PASS 0.75 at seat 1's
//! decision, TAP_A 0.4 / TAP_B 0.6 at the payment, uniform elsewhere; the value of the viewer's
//! information (seat 0: the payoff if the coin is known, half of it if not - blocked or not as far as
//! decided; 0.5 before its first move; seat 1: one minus seat 0's, the coin known).
use crate::game::{belief_weight, Belief, SearchGame, MAX_SEATS};
use crate::rng::Rng;

pub const PEEK: u16 = 0;
pub const GO: u16 = 1;
pub const TAP_A: u16 = 2;
pub const TAP_B: u16 = 3;
pub const BLOCK: u16 = 4;
pub const PASS: u16 = 5;
pub const GUESS0: u16 = 6;
pub const GUESS1: u16 = 7;

#[derive(Clone, Debug, PartialEq)]
pub struct Peek {
    pub coin: u8,
    pub peeked: bool,
    pub blocked: bool,
    pub tapped: u8,
    pub guess: Option<u8>,
    /// 0 first move, 1 payment, 2 seat 1's reply, 3 the guess, 4 over.
    pub stage: u8,
}

impl Peek {
    pub fn with_coin(coin: u8) -> Peek {
        Peek { coin, peeked: false, blocked: false, tapped: 0, guess: None, stage: 0 }
    }

    pub fn payoff(&self) -> f32 {
        (if self.peeked { 0.9 } else { 1.0 }) * (if self.blocked { 0.8 } else { 1.0 })
    }
}

impl SearchGame for Peek {
    type Spec = ();
    const NAME: &'static str = "peek";
    const NUM_ACTIONS: usize = 8;
    const NUM_FEATURES: usize = 8;
    const MIN_PLAYERS: usize = 2;
    const MAX_PLAYERS: usize = 2;

    fn parse_spec(_json: &str) -> Result<(), String> {
        Ok(())
    }
    fn new_game(_spec: &(), players: usize, seed: u64) -> Result<Peek, String> {
        if players != 2 {
            return Err("peek: 2 players".into());
        }
        Ok(Peek::with_coin(Rng::new(seed).below(2) as u8))
    }
    fn num_players(&self) -> usize {
        2
    }
    fn to_move(&self) -> Option<usize> {
        match self.stage {
            0 | 1 | 3 => Some(0),
            2 => Some(1),
            _ => None,
        }
    }
    fn legal_actions(&self, out: &mut Vec<u16>) {
        match self.stage {
            0 => out.extend_from_slice(&[PEEK, GO]),
            1 => out.extend_from_slice(&[TAP_A, TAP_B]),
            2 => out.extend_from_slice(&[BLOCK, PASS]),
            3 => out.extend_from_slice(&[GUESS0, GUESS1]),
            _ => {}
        }
    }
    fn step(&mut self, a: u16, chance: &mut Vec<u64>) -> Result<(), String> {
        match (self.stage, a) {
            (0, PEEK) => {
                self.peeked = true;
                chance.push(1 + self.coin as u64);
            }
            (0, GO) => {}
            (1, TAP_A) | (1, TAP_B) => self.tapped = (a - TAP_A + 1) as u8,
            (2, BLOCK) => self.blocked = true,
            (2, PASS) => {}
            (3, GUESS0) | (3, GUESS1) => self.guess = Some((a - GUESS0) as u8),
            _ => return Err(format!("peek: action {a} at stage {}", self.stage)),
        }
        self.stage += 1;
        Ok(())
    }
    fn result(&self) -> [f32; MAX_SEATS] {
        let mut r = [0.0; MAX_SEATS];
        let p = if self.guess == Some(self.coin) { self.payoff() } else { 0.0 };
        r[0] = p;
        r[1] = 1.0 - p;
        r
    }
    fn margin(&self, _seat: usize) -> f32 {
        0.0
    }
    fn score(&self, seat: usize) -> f32 {
        self.result()[seat]
    }
    fn determinize(&mut self, viewer: usize, rng: &mut Rng) {
        if viewer == 0 && !self.peeked {
            self.coin = rng.below(2) as u8;
        }
    }
    const AUX: &'static [(&'static str, usize)] = &[("coin", 2)];
    fn aux_target(&self, _viewer: usize, out: &mut [f32]) {
        out[self.coin as usize] = 1.0;
    }
    /// The coin 1 with probability w1 / (w0 + w1) (`belief_weight` of the belief's [coin 0, coin 1]
    /// against the uniform deal's 1/2 each).
    fn determinize_with(&mut self, viewer: usize, rng: &mut Rng, belief: &Belief) {
        if viewer == 0 && !self.peeked {
            let w1 = belief_weight(belief.pred[1] as f64, 0.5, belief.tilt, belief.clip);
            let w0 = belief_weight(belief.pred[0] as f64, 0.5, belief.tilt, belief.clip);
            self.coin = (rng.unit() < w1 / (w0 + w1)) as u8;
        }
    }
    fn encode(&self, viewer: usize, out: &mut [f32]) {
        let known = viewer == 1 || self.peeked;
        out[0] = self.stage as f32 / 4.0;
        out[1] = self.peeked as u8 as f32;
        out[2] = known as u8 as f32;
        out[3] = if known { self.coin as f32 } else { 0.0 };
        out[4] = self.blocked as u8 as f32;
        out[5] = self.tapped as f32;
        out[6] = viewer as f32;
        out[7] = self.guess.map_or(0.0, |g| 1.0 + g as f32);
    }
    fn rounds(&self) -> u32 {
        match self.stage {
            0 | 1 => 1,
            2 => 2,
            _ => 3,
        }
    }
    fn minor_decision(&self) -> bool {
        self.stage == 1
    }
    fn active_seat(&self) -> Option<usize> {
        Some(if self.stage == 2 { 1 } else { 0 })
    }
}

/// The stand-in network (module docs) on `encode(viewer)` features with `nopts` options: (priors,
/// value relative to the viewer).
pub fn peek_eval(x: &[f32], nopts: usize) -> (Vec<f32>, [f32; MAX_SEATS]) {
    let stage = (x[0] * 4.0).round() as u8;
    let pri = match (stage, nopts) {
        (1, 2) => vec![0.4, 0.6],
        (2, 2) => vec![0.25, 0.75],
        (_, 0) => Vec::new(),
        (_, k) => vec![1.0 / k as f32; k],
    };
    let viewer = x[6] as usize;
    let peeked = x[1] != 0.0;
    let known = x[2] != 0.0;
    let payoff = (if peeked { 0.9 } else { 1.0 }) * (if x[4] != 0.0 { 0.8 } else { 1.0 });
    let v0 = if stage == 0 { 0.5 } else if known && viewer == 0 || peeked { payoff } else { 0.5 * payoff };
    let mut val = [0.0; MAX_SEATS];
    if viewer == 0 {
        val[0] = v0;
        val[1] = 1.0 - v0;
    } else {
        // seat 1 knows the coin: seat 0's value if it guesses right exactly when it peeked
        let s0 = if stage == 0 { 0.5 } else if peeked { payoff } else { 0.5 * payoff };
        val[0] = 1.0 - s0;
        val[1] = s0;
    }
    (pri, val)
}
