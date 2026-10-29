//! A test game for the root minimum visits (docs/search.md 3.5): "keep or mulligan". Each of two
//! seats is dealt a hand, good or bad (1 in 2), and in turn keeps it (`KEEP`, action 0) or takes a
//! mulligan (`MULL`, action 1: a new hand, good 3 in 4, at a cost). Then the game ends:
//! P(seat 0 wins) = 0.5 + 0.3 (quality 0 - quality 1), quality = 1 (good, kept), 0.7 (good after a
//! mulligan), 0 (bad). The pending mulligan draws are hidden (`determinize` reseeds them).
//!
//! `mull_eval` is the stand-in network of MTG Zero's Delver mirror at the mulligan (the Fable
//! review, 2026-10-29): a prior of 0.999 on KEEP whatever the hand, and the exact value of the
//! position if every seat still to decide keeps. With a bad hand a mulligan is worth ~0.16 more
//! than keeping, with a good one ~0.14 less - and at 64 simulations PUCT never visits it.
use crate::game::{SearchGame, MAX_SEATS};
use crate::rng::Rng;

pub const KEEP: u16 = 0;
pub const MULL: u16 = 1;
/// The prior of `MULL` in `mull_eval`.
pub const P_MULL: f32 = 0.001;

#[derive(Clone, Debug)]
pub struct Mull {
    /// 1 = a good hand, 0 = a bad one, per seat.
    pub hand: [u8; 2],
    pub mulled: [bool; 2],
    /// Seats that have decided (0, 1, 2 = over).
    pub decided: u8,
    rng: u64,
}

impl Mull {
    pub fn quality(&self, s: usize) -> f32 {
        match (self.hand[s], self.mulled[s]) {
            (1, false) => 1.0,
            (1, true) => 0.7,
            _ => 0.0,
        }
    }

    /// A game with these hands, seat 0 to decide.
    pub fn with_hands(h0: u8, h1: u8, seed: u64) -> Mull {
        Mull { hand: [h0, h1], mulled: [false; 2], decided: 0, rng: seed }
    }
}

impl SearchGame for Mull {
    type Spec = ();
    const NAME: &'static str = "mull";
    const NUM_ACTIONS: usize = 2;
    const NUM_FEATURES: usize = 4;
    const MIN_PLAYERS: usize = 2;
    const MAX_PLAYERS: usize = 2;

    fn parse_spec(_json: &str) -> Result<(), String> {
        Ok(())
    }
    fn new_game(_spec: &(), players: usize, seed: u64) -> Result<Mull, String> {
        if players != 2 {
            return Err("mull: 2 players".into());
        }
        let mut r = Rng::new(seed);
        let h = [r.below(2) as u8, r.below(2) as u8];
        Ok(Mull::with_hands(h[0], h[1], r.next_u64()))
    }
    fn num_players(&self) -> usize {
        2
    }
    fn to_move(&self) -> Option<usize> {
        (self.decided < 2).then_some(self.decided as usize)
    }
    fn legal_actions(&self, out: &mut Vec<u16>) {
        if self.decided < 2 {
            out.extend_from_slice(&[KEEP, MULL]);
        }
    }
    fn step(&mut self, a: u16, chance: &mut Vec<u64>) -> Result<(), String> {
        let s = self.to_move().ok_or("mull: the game is over")?;
        match a {
            KEEP => {}
            MULL => {
                let mut r = Rng::new(self.rng);
                let good = r.below(4) != 0;
                self.rng = r.next_u64();
                self.hand[s] = good as u8;
                self.mulled[s] = true;
                chance.push(good as u64);
            }
            _ => return Err(format!("mull: action {a}")),
        }
        self.decided += 1;
        Ok(())
    }
    fn result(&self) -> [f32; MAX_SEATS] {
        let p = 0.5 + 0.3 * (self.quality(0) - self.quality(1));
        let mut r = [0.0; MAX_SEATS];
        r[0] = p;
        r[1] = 1.0 - p;
        r
    }
    fn margin(&self, _seat: usize) -> f32 {
        0.0
    }
    fn score(&self, seat: usize) -> f32 {
        self.quality(seat)
    }
    fn determinize(&mut self, _viewer: usize, rng: &mut Rng) {
        self.rng = rng.next_u64();
    }
    fn encode(&self, viewer: usize, out: &mut [f32]) {
        out[0] = self.quality(viewer);
        out[1] = self.quality(1 - viewer);
        out[2] = self.decided as f32;
        out[3] = self.mulled[viewer] as u8 as f32;
    }
}

/// The stand-in network on `encode(decider)` features: priors over `nopts` options (KEEP 0.999,
/// MULL 0.001; none at a terminal) and the value relative to the decider ([v, 1 - v]).
pub fn mull_eval(x: &[f32], nopts: usize) -> (Vec<f32>, [f32; MAX_SEATS]) {
    let pri = if nopts == 2 { vec![1.0 - P_MULL, P_MULL] } else { vec![1.0 / nopts.max(1) as f32; nopts] };
    let v = 0.5 + 0.3 * (x[0] - x[1]);
    let mut val = [0.0; MAX_SEATS];
    val[0] = v;
    val[1] = 1.0 - v;
    (pri, val)
}
