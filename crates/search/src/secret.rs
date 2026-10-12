//! "Secret hands": a tiny 2-player game with a hidden, simultaneous hand choice, the core's test
//! bed for the opponent-model options (docs/search.md 10: decision classes, the oracle,
//! policy-sampled simultaneous choices, the belief prior) - a miniature of Glass Road's `select`
//! + card duel.
//!
//! Rules. Five cards 0..4. **Pick** (simultaneous in the rules, sequentialised: seat 0 first,
//! then seat 1): each seat secretly chooses a hand of 2 cards as 2 ascending atomic picks
//! (`CARD(c)`); the two hands may share cards (every seat owns a copy of every card). **Play**:
//! 2 tricks; seat t leads trick t with a card of its hand (public). The other seat must
//! **follow** with the same card if it holds it (automatic: a revelation, keyed; the trick is a
//! tie), otherwise it plays any card of its hand. The higher card scores its value + 1 for its
//! player. Most points wins; ties split.
//!
//! Hidden from a viewer: the other seat's hand (and, during Pick, its committed hand; a later
//! chooser's partial picks). Knowledge: a lead that the other seat did not follow proves it
//! does not hold that card (`excl`); the consistent hands are the cards it played plus a
//! subset of the cards neither played nor excluded. Features never show whether the other seat
//! has committed its hand (role-blind choice, as Glass Road's `opp_selected`).
//!
//! Hooks: classes `play` (0) / `pick` (1); the oracle keeps the true hands; `policy_sampled` =
//! a Pick decision; `uncommit_simultaneous` takes back the other seat's committed hand while
//! the viewer picks (the world's pick order becomes viewer first); the belief starts at a seat's
//! first pick (key 0), its particle is the 2-card hand as a 5-bit mask, the target is the other
//! seat's hand during Play.
use crate::game::{SearchGame, MAX_SEATS};
use crate::rng::Rng;

pub const NUM_CARDS: u8 = 5;
const ALL: u8 = (1 << NUM_CARDS) - 1;
pub const HAND: u32 = 2;
pub const PHASE_PICK: u8 = 0;
pub const PHASE_PLAY: u8 = 1;
pub const PHASE_OVER: u8 = 2;
/// Feature layout: phase (3), my hand (5), my partial picks (5, decider only), I decide (1),
/// my played (5), opp played (5), opp excluded (5), trick (2), lead card (5), I lead (1),
/// points / 10 (2).
pub const NUM_FEATURES: usize = 3 + 5 + 5 + 1 + 5 + 5 + 5 + 2 + 5 + 1 + 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SecretGame {
    pub phase: u8,
    /// Cards still in hand / the chosen hand, per seat (bit masks).
    pub hands: [u8; 2],
    pub chosen: [u8; 2],
    pub committed: [bool; 2],
    /// Pick order of this world (seats); normally [0, 1].
    pub order: [u8; 2],
    /// The current picker's partial picks.
    pub partial: u8,
    pub trick: u8,
    pub lead: Option<u8>,
    pub played: [u8; 2],
    pub excl: [u8; 2],
    pub points: [u8; 2],
}

#[derive(Clone, Debug, Default)]
pub struct SecretSpec {}

fn bits(m: u8) -> u32 {
    m.count_ones()
}

/// A uniform k-subset of `pool`.
fn sample_subset(pool: u8, k: u32, rng: &mut Rng) -> u8 {
    let mut cards: Vec<u8> = (0..NUM_CARDS).filter(|&c| pool >> c & 1 == 1).collect();
    assert!(k as usize <= cards.len(), "secret: {k} of {} cards", cards.len());
    let mut out = 0u8;
    for i in 0..k as usize {
        let j = i + rng.below((cards.len() - i) as u32) as usize;
        cards.swap(i, j);
        out |= 1 << cards[i];
    }
    out
}

impl SecretGame {
    pub fn new() -> SecretGame {
        SecretGame { phase: PHASE_PICK, hands: [0; 2], chosen: [0; 2], committed: [false; 2], order: [0, 1], partial: 0, trick: 0, lead: None, played: [0; 2], excl: [0; 2], points: [0; 2] }
    }

    /// The seat picking now (Pick phase).
    fn picker(&self) -> Option<usize> {
        self.order.iter().map(|&s| s as usize).find(|&s| !self.committed[s])
    }

    /// Cards the other seat may still hold, as far as `viewer` knows (Play phase).
    pub fn possible(&self, viewer: usize) -> u8 {
        let q = 1 - viewer;
        ALL & !self.played[q] & !self.excl[q]
    }

    fn finish_trick(&mut self, leader: usize, c_lead: u8, follower: usize, c_follow: u8) {
        if c_lead > c_follow {
            self.points[leader] += c_lead + 1;
        } else if c_follow > c_lead {
            self.points[follower] += c_follow + 1;
        }
        self.lead = None;
        self.trick += 1;
        if self.trick >= HAND as u8 {
            self.phase = PHASE_OVER;
        }
    }
}

impl Default for SecretGame {
    fn default() -> Self {
        SecretGame::new()
    }
}

impl SearchGame for SecretGame {
    type Spec = SecretSpec;
    const NAME: &'static str = "secret";
    const NUM_ACTIONS: usize = NUM_CARDS as usize;
    const NUM_FEATURES: usize = NUM_FEATURES;
    const MIN_PLAYERS: usize = 2;
    const MAX_PLAYERS: usize = 2;

    fn parse_spec(json: &str) -> Result<SecretSpec, String> {
        let v: serde_json::Value = if json.trim().is_empty() { serde_json::json!({}) } else { serde_json::from_str(json).map_err(|e| format!("secret spec: {e}"))? };
        let o = v.as_object().ok_or("secret spec: a JSON object")?;
        if let Some(k) = o.keys().next() {
            return Err(format!("secret spec: unknown key {k:?}"));
        }
        Ok(SecretSpec {})
    }

    fn new_game(_spec: &SecretSpec, players: usize, _seed: u64) -> Result<Self, String> {
        if players != 2 {
            return Err(format!("secret: {players} players (2 only)"));
        }
        Ok(SecretGame::new())
    }

    fn num_players(&self) -> usize {
        2
    }

    fn to_move(&self) -> Option<usize> {
        match self.phase {
            PHASE_PICK => self.picker(),
            PHASE_PLAY => {
                let leader = self.trick as usize % 2;
                Some(if self.lead.is_some() { 1 - leader } else { leader })
            }
            _ => None,
        }
    }

    fn legal_actions(&self, out: &mut Vec<u16>) {
        match self.phase {
            PHASE_PICK => {
                let start = (0..NUM_CARDS).rev().find(|&c| self.partial >> c & 1 == 1).map(|c| c + 1).unwrap_or(0);
                let need = HAND - bits(self.partial);
                for c in start..NUM_CARDS {
                    if (NUM_CARDS - c) as u32 >= need {
                        out.push(c as u16);
                    }
                }
            }
            PHASE_PLAY => {
                let p = self.to_move().unwrap();
                for c in 0..NUM_CARDS {
                    if self.hands[p] >> c & 1 == 1 {
                        out.push(c as u16);
                    }
                }
            }
            _ => {}
        }
    }

    fn step(&mut self, a: u16, chance: &mut Vec<u64>) -> Result<(), String> {
        let mut legal = Vec::new();
        self.legal_actions(&mut legal);
        if !legal.contains(&a) {
            return Err(format!("secret: illegal action {a} (legal {legal:?})"));
        }
        let c = a as u8;
        match self.phase {
            PHASE_PICK => {
                let p = self.picker().unwrap();
                self.partial |= 1 << c;
                if bits(self.partial) == HAND {
                    self.chosen[p] = self.partial;
                    self.hands[p] = self.partial;
                    self.committed[p] = true;
                    self.partial = 0;
                    if self.committed.iter().all(|&x| x) {
                        self.phase = PHASE_PLAY;
                        self.order = [0, 1];
                    }
                }
            }
            PHASE_PLAY => {
                let p = self.to_move().unwrap();
                let q = 1 - p;
                self.hands[p] &= !(1 << c);
                self.played[p] |= 1 << c;
                match self.lead {
                    None => {
                        // q must follow if it holds c: revealed (keyed), a tie
                        let follow = self.hands[q] >> c & 1 == 1;
                        chance.push(1 << 16 | (self.trick as u64) << 8 | (c as u64) << 1 | follow as u64);
                        if follow {
                            self.hands[q] &= !(1 << c);
                            self.played[q] |= 1 << c;
                            self.finish_trick(p, c, q, c);
                        } else {
                            self.excl[q] |= 1 << c;
                            self.lead = Some(c);
                        }
                    }
                    Some(l) => self.finish_trick(q, l, p, c),
                }
            }
            _ => return Err("secret: the game is over".into()),
        }
        Ok(())
    }

    fn result(&self) -> [f32; MAX_SEATS] {
        let mut r = [0.0; MAX_SEATS];
        match self.points[0].cmp(&self.points[1]) {
            std::cmp::Ordering::Greater => r[0] = 1.0,
            std::cmp::Ordering::Less => r[1] = 1.0,
            std::cmp::Ordering::Equal => {
                r[0] = 0.5;
                r[1] = 0.5;
            }
        }
        r
    }

    fn margin(&self, seat: usize) -> f32 {
        (self.points[seat] as f32 - self.points[1 - seat] as f32) / 5.0
    }

    fn score(&self, seat: usize) -> f32 {
        self.points[seat] as f32
    }

    fn determinize(&mut self, viewer: usize, rng: &mut Rng) {
        let q = 1 - viewer;
        match self.phase {
            PHASE_PICK => {
                if self.committed[q] {
                    let h = sample_subset(ALL, HAND, rng);
                    self.chosen[q] = h;
                    self.hands[q] = h;
                }
                // a later chooser's partial picks are hidden from the earlier chooser
                if self.picker() == Some(q) && self.partial != 0 {
                    let h = sample_subset(ALL, HAND, rng);
                    let k = bits(self.partial);
                    let mut part = 0u8;
                    for c in 0..NUM_CARDS {
                        if h >> c & 1 == 1 && bits(part) < k {
                            part |= 1 << c;
                        }
                    }
                    self.partial = part;
                }
            }
            PHASE_PLAY => {
                let k = bits(self.hands[q]);
                let h = sample_subset(self.possible(viewer), k, rng);
                self.hands[q] = h;
                self.chosen[q] = h | self.played[q];
            }
            _ => {}
        }
    }

    fn encode(&self, viewer: usize, out: &mut [f32]) {
        let q = 1 - viewer;
        let mut i = 0;
        let put = |out: &mut [f32], k: usize, v: f32| {
            out[k] = v;
        };
        out[i + self.phase as usize] = 1.0;
        i += 3;
        let mask = |out: &mut [f32], i: &mut usize, m: u8| {
            for c in 0..NUM_CARDS as usize {
                out[*i + c] = (m >> c & 1) as f32;
            }
            *i += NUM_CARDS as usize;
        };
        mask(out, &mut i, self.hands[viewer]);
        let mine = self.to_move() == Some(viewer);
        mask(out, &mut i, if mine && self.phase == PHASE_PICK { self.partial } else { 0 });
        put(out, i, mine as u8 as f32);
        i += 1;
        mask(out, &mut i, self.played[viewer]);
        mask(out, &mut i, self.played[q]);
        mask(out, &mut i, self.excl[q]);
        if self.phase == PHASE_PLAY {
            out[i + self.trick as usize] = 1.0;
        }
        i += 2;
        mask(out, &mut i, self.lead.map(|c| 1u8 << c).unwrap_or(0));
        put(out, i, (self.phase == PHASE_PLAY && self.trick as usize % 2 == viewer) as u8 as f32);
        i += 1;
        put(out, i, self.points[viewer] as f32 / 10.0);
        put(out, i + 1, self.points[q] as f32 / 10.0);
        i += 2;
        debug_assert_eq!(i, NUM_FEATURES);
    }

    fn rounds(&self) -> u32 {
        self.trick as u32
    }

    const DECISION_CLASSES: &'static [&'static str] = &["play", "pick"];

    fn decision_class(&self) -> usize {
        (self.phase == PHASE_PICK) as usize
    }

    const HAS_ORACLE: bool = true;

    /// Nothing but the hands is hidden: the oracle world is the true one.
    fn determinize_oracle(&mut self, _viewer: usize, _rng: &mut Rng) {}

    const HAS_PSAMPLE: bool = true;

    fn policy_sampled(&self) -> bool {
        self.phase == PHASE_PICK
    }

    fn uncommit_simultaneous(&mut self, viewer: usize) {
        let q = 1 - viewer;
        if self.phase == PHASE_PICK && self.picker() == Some(viewer) && self.committed[q] {
            self.committed[q] = false;
            self.chosen[q] = 0;
            self.hands[q] = 0;
            self.order = [viewer as u8, q as u8];
        }
    }

    const HAS_BELIEF: bool = true;

    fn belief_start(&self) -> Option<u64> {
        (self.phase == PHASE_PICK && self.partial == 0).then_some(0)
    }

    fn belief_particle(&self, chooser: usize) -> Option<u64> {
        self.committed[chooser].then_some(self.chosen[chooser] as u64)
    }

    fn belief_target(&self, viewer: usize) -> Option<(usize, u64)> {
        let q = 1 - viewer;
        (self.phase == PHASE_PLAY && self.hands[q] != 0).then_some((q, 0))
    }

    fn belief_consistent(&self, viewer: usize, particle: u64) -> bool {
        let q = 1 - viewer;
        if particle > ALL as u64 || bits(particle as u8) != HAND {
            return false;
        }
        let x = particle as u8;
        let rest = x & !self.played[q];
        x & self.played[q] == self.played[q] && rest & self.excl[q] == 0 && bits(rest) == bits(self.hands[q])
    }

    fn determinize_particle(&mut self, viewer: usize, particle: u64, _rng: &mut Rng) {
        let q = 1 - viewer;
        let x = particle as u8;
        self.chosen[q] = x;
        self.hands[q] = x & !self.played[q];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::transition;

    fn random_game(seed: u64, mut f: impl FnMut(&SecretGame)) -> SecretGame {
        let mut g = SecretGame::new();
        let mut rng = Rng::new(seed);
        let (mut key, mut legal) = (Vec::new(), Vec::new());
        while g.to_move().is_some() {
            f(&g);
            legal.clear();
            g.legal_actions(&mut legal);
            let a = legal[rng.below(legal.len() as u32) as usize];
            transition(&mut g, a, None, &mut key, &mut legal).unwrap();
        }
        g
    }

    fn feats(g: &SecretGame, v: usize) -> Vec<f32> {
        let mut x = vec![0.0; NUM_FEATURES];
        g.encode(v, &mut x);
        x
    }

    #[test]
    fn games_end_and_determinize_keeps_what_the_viewer_knows() {
        let mut rng = Rng::new(9);
        let mut checks = 0;
        for seed in 0..200 {
            let g = random_game(seed, |g| {
                for v in 0..2 {
                    let mut h = g.clone();
                    h.determinize(v, &mut rng);
                    assert_eq!(feats(&h, v), feats(g, v), "features of seat {v} after determinize({v})");
                    let (mut a, mut b) = (Vec::new(), Vec::new());
                    g.legal_actions(&mut a);
                    h.legal_actions(&mut b);
                    if g.to_move() == Some(v) {
                        assert_eq!(a, b);
                    }
                    assert_eq!((h.hands[v], h.played, h.excl), (g.hands[v], g.played, g.excl));
                    if g.phase == PHASE_PLAY {
                        // the true hand is consistent; sampled hands too
                        assert!(g.belief_consistent(v, g.chosen[1 - v] as u64));
                        assert!(h.belief_consistent(v, h.chosen[1 - v] as u64) && g.belief_consistent(v, h.chosen[1 - v] as u64));
                    }
                    checks += 1;
                }
            });
            assert_eq!(g.phase, PHASE_OVER);
            let r = g.result();
            assert!((r[0] + r[1] - 1.0).abs() < 1e-6);
        }
        assert!(checks > 2000);
    }

    #[test]
    fn uncommit_lets_the_viewer_pick_first() {
        let mut g = SecretGame::new();
        let (mut key, mut legal) = (Vec::new(), Vec::new());
        transition(&mut g, 1, None, &mut key, &mut legal).unwrap();
        transition(&mut g, 3, None, &mut key, &mut legal).unwrap();
        assert_eq!(g.to_move(), Some(1));
        assert!(g.policy_sampled() && g.belief_start() == Some(0) && g.belief_particle(0) == Some(0b1010));
        let mut w = g.clone();
        w.uncommit_simultaneous(1);
        assert_eq!(w.to_move(), Some(1));
        assert_eq!(feats(&w, 1), feats(&g, 1), "the commit was hidden from seat 1");
        // seat 1 picks, then seat 0 (again), then the play phase
        transition(&mut w, 0, None, &mut key, &mut legal).unwrap();
        transition(&mut w, 4, None, &mut key, &mut legal).unwrap();
        assert_eq!(w.to_move(), Some(0));
        assert!(w.policy_sampled() && w.belief_start() == Some(0));
        transition(&mut w, 2, None, &mut key, &mut legal).unwrap();
        transition(&mut w, 3, None, &mut key, &mut legal).unwrap();
        assert_eq!((w.phase, w.chosen), (PHASE_PLAY, [0b1100, 0b10001]));
        assert_eq!(w.belief_target(1), Some((0, 0)));
        // nothing to take back for the first chooser
        let mut s = SecretGame::new();
        s.uncommit_simultaneous(0);
        assert_eq!(s, SecretGame::new());
    }
}
