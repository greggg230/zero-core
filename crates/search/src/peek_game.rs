//! A test game for heavy decisions' reveal probe (docs/search.md 3.11, `heavy::probe`) and the
//! driver's heavy searches: "peek". Seat 0 may look at a hidden coin before it guesses it.
use crate::game::{SearchGame, MAX_SEATS};
use crate::rng::Rng;

/// A test game for the probe: seat 0 holds a hidden coin c (0/1, unknown to it) and decides at
/// round 0: `PASS` (0: do nothing; seat 1 then discards its card publicly, the end), `PEEK` (1:
/// a choice of where to look - `AT_COIN` (4) or `AT_NOTHING` (5), a forced choice -, then its own
/// priority again, where `PASS` resolves the look and `GUESS0` / `GUESS1` end the game), `GUESS0`
/// / `GUESS1` (2, 3: the end), `SLOW` (6: nothing now; at its next own decision `LOOK` (7) looks
/// at the coin - a follow-up action, depth 2), `ROUND` (8: the round ends and the next round
/// reveals the coin to everyone). Keys: (kind << 8) | payload; kind 1 = a look by seat 0, kind 2
/// = seat 1's card made public, kind 3 = bookkeeping (never reveals), kind 4 = seat 1's private
/// draw (masked for seat 0).
#[derive(Clone, Debug)]
pub struct Peek {
    pub coin: u8,
    pub card: u8,
    pub phase: u8,
    pub round: u32,
    pub over: bool,
    pub observer: Option<usize>,
    /// Seat 0's guess (the end): right = seat 0 wins, wrong = seat 1 wins; no guess = a draw.
    pub guess: Option<u8>,
}

pub const PASS: u16 = 0;
pub const PEEK: u16 = 1;
pub const GUESS0: u16 = 2;
pub const GUESS1: u16 = 3;
pub const AT_COIN: u16 = 4;
pub const AT_NOTHING: u16 = 5;
pub const SLOW: u16 = 6;
pub const LOOK: u16 = 7;
pub const ROUND: u16 = 8;

impl SearchGame for Peek {
    type Spec = ();
    const NAME: &'static str = "peek";
    const NUM_ACTIONS: usize = 9;
    const NUM_FEATURES: usize = 4;
    const MIN_PLAYERS: usize = 2;
    const MAX_PLAYERS: usize = 2;
    fn parse_spec(_: &str) -> Result<(), String> {
        Ok(())
    }
    fn new_game(_: &(), _: usize, seed: u64) -> Result<Peek, String> {
        let mut r = Rng::new(seed);
        Ok(Peek { coin: r.below(2) as u8, card: r.below(5) as u8, phase: 0, round: 0, over: false, observer: None, guess: None })
    }
    fn num_players(&self) -> usize {
        2
    }
    fn to_move(&self) -> Option<usize> {
        if self.over {
            return None;
        }
        Some(if self.phase == 4 { 1 } else { 0 })
    }
    fn legal_actions(&self, out: &mut Vec<u16>) {
        match (self.over, self.phase) {
            (true, _) => {}
            (_, 0) => out.extend_from_slice(&[PASS, PEEK, GUESS0, GUESS1, SLOW, ROUND]),
            (_, 1) => out.extend_from_slice(&[AT_COIN, AT_NOTHING]),
            (_, 2) | (_, 3) => out.extend_from_slice(&[PASS, GUESS0, GUESS1]),
            (_, 4) => out.extend_from_slice(&[PASS, GUESS0]),
            (_, 5) => out.extend_from_slice(&[PASS, LOOK, GUESS0]),
            _ => out.extend_from_slice(&[GUESS0, GUESS1]),
        }
    }
    fn step(&mut self, a: u16, chance: &mut Vec<u64>) -> Result<(), String> {
        let masked = |s: &Peek, to: usize, v: u64| if s.observer.is_some_and(|o| o != to) { 0xFF } else { v };
        match (self.phase, a) {
            (0, PASS) => {
                // seat 1 draws (private) and then decides (its passive option: PASS)
                chance.push(4 << 8 | masked(self, 1, self.card as u64));
                self.phase = 4;
            }
            (0, PEEK) => self.phase = 1,
            (0, SLOW) => {
                chance.push(3 << 8);
                self.phase = 5;
            }
            (0, ROUND) => {
                self.round += 1;
                chance.push(2 << 8 | self.coin as u64);
                self.phase = 6;
            }
            (1, AT_COIN) => self.phase = 2,
            (1, AT_NOTHING) => self.phase = 3,
            (2, PASS) => {
                chance.push(1 << 8 | self.coin as u64);
                self.phase = 6;
            }
            (3, PASS) | (5, PASS) => self.phase = 6,
            (5, LOOK) => {
                chance.push(1 << 8 | self.coin as u64);
                self.phase = 6;
            }
            (4, PASS) | (4, GUESS0) => {
                // seat 1's card goes public whatever it does
                chance.push(2 << 8 | 0x10 | self.card as u64);
                self.over = true;
            }
            (4, _) => return Err(format!("peek: {a} in phase 4")),
            (_, GUESS0) | (_, GUESS1) => {
                self.guess = Some((a - GUESS0) as u8);
                self.over = true;
            }
            _ => return Err(format!("peek: {a} in phase {}", self.phase)),
        }
        Ok(())
    }
    fn result(&self) -> [f32; MAX_SEATS] {
        let mut r = [0.0; MAX_SEATS];
        let w = match self.guess {
            Some(g) if g == self.coin => 1.0,
            Some(_) => 0.0,
            None => 0.5,
        };
        r[0] = w;
        r[1] = 1.0 - w;
        r
    }
    fn margin(&self, _: usize) -> f32 {
        0.0
    }
    fn score(&self, _: usize) -> f32 {
        0.0
    }
    fn determinize(&mut self, viewer: usize, rng: &mut Rng) {
        self.observer = Some(viewer);
        if viewer == 0 {
            self.coin = rng.below(2) as u8;
            self.card = rng.below(5) as u8;
        }
    }
    fn encode(&self, _: usize, out: &mut [f32]) {
        out[0] = self.phase as f32;
    }
    fn rounds(&self) -> u32 {
        self.round
    }
    /// One light class (docs/search.md 3.10), `start`: seat 0's first decision.
    const LIGHT_CLASSES: &'static [&'static str] = &["start"];
    fn light_class(&self) -> Option<usize> {
        (self.phase == 0 && !self.over).then_some(0)
    }
    fn passive_action(&self, legal: &[u16]) -> Option<u16> {
        legal.contains(&PASS).then_some(PASS)
    }
    fn reveals(&self, key: u64, viewer: usize) -> bool {
        match key >> 8 {
            1 => viewer == 0,
            2 => viewer == 0,
            4 => viewer == 1 && key & 0xFF != 0xFF,
            _ => false,
        }
    }
}

