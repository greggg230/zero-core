//! "Pick", a test game with transpositions for near-tie branches' merges (docs/search.md 2.11.1; `merge_key`): the
//! driver's tests and the native worker's records tests play it. Spec `{"turns": 8, "keys": true}` (the defaults).
use crate::game::{SearchGame, MAX_SEATS};
use crate::rng::Rng;

/// "Pick": two players alternate `turns` turns (seat 0 first); the mover adds 0, 1 or 2 to its own sum; the higher sum
/// wins (a tie is a draw). Picks commute, so lines transpose: seat 0 picking 1 then 2 reaches the state of 2 then 1
/// when seat 1 picked the same between. `keys`: `merge_key` = (turn, sums) - the whole state; without it None.
#[derive(Clone, Debug)]
pub struct PickGame {
    pub turns: u8,
    pub keys: bool,
    pub turn: u8,
    pub sum: [u8; 2],
}

impl SearchGame for PickGame {
    type Spec = (u8, bool);
    const NAME: &'static str = "pick";
    const NUM_ACTIONS: usize = 3;
    const NUM_FEATURES: usize = 4;
    const MIN_PLAYERS: usize = 2;
    const MAX_PLAYERS: usize = 2;

    fn parse_spec(json: &str) -> Result<(u8, bool), String> {
        let v: serde_json::Value = serde_json::from_str(if json.trim().is_empty() { "{}" } else { json }).map_err(|e| e.to_string())?;
        Ok((v.get("turns").and_then(|x| x.as_u64()).unwrap_or(8) as u8, v.get("keys").and_then(|x| x.as_bool()).unwrap_or(true)))
    }
    fn new_game(spec: &(u8, bool), players: usize, _seed: u64) -> Result<PickGame, String> {
        if players != 2 {
            return Err("pick: 2 players".into());
        }
        Ok(PickGame { turns: spec.0, keys: spec.1, turn: 0, sum: [0, 0] })
    }
    fn num_players(&self) -> usize {
        2
    }
    fn to_move(&self) -> Option<usize> {
        (self.turn < self.turns).then_some(self.turn as usize % 2)
    }
    fn legal_actions(&self, out: &mut Vec<u16>) {
        if self.turn < self.turns {
            out.extend_from_slice(&[0, 1, 2]);
        }
    }
    fn step(&mut self, a: u16, _chance: &mut Vec<u64>) -> Result<(), String> {
        if a > 2 || self.turn >= self.turns {
            return Err(format!("pick: {a} is not legal"));
        }
        self.sum[self.turn as usize % 2] += a as u8;
        self.turn += 1;
        Ok(())
    }
    fn result(&self) -> [f32; MAX_SEATS] {
        let mut r = [0.0; MAX_SEATS];
        match self.sum[0].cmp(&self.sum[1]) {
            std::cmp::Ordering::Greater => r[0] = 1.0,
            std::cmp::Ordering::Less => r[1] = 1.0,
            std::cmp::Ordering::Equal => r[..2].fill(0.5),
        }
        r
    }
    fn margin(&self, seat: usize) -> f32 {
        (self.sum[seat] as f32 - self.sum[1 - seat] as f32) / 16.0
    }
    fn score(&self, seat: usize) -> f32 {
        self.sum[seat] as f32
    }
    fn determinize(&mut self, _viewer: usize, _rng: &mut Rng) {}
    fn encode(&self, viewer: usize, out: &mut [f32]) {
        out[0] = self.turn as f32 / 8.0;
        out[1] = self.sum[viewer] as f32 / 8.0;
        out[2] = self.sum[1 - viewer] as f32 / 8.0;
        out[3] = (self.turn as usize % 2 == viewer) as u8 as f32;
    }
    fn merge_key(&self) -> Option<u128> {
        self.keys.then(|| (self.turn as u128) << 16 | (self.sum[0] as u128) << 8 | self.sum[1] as u128)
    }
}
