//! "Toy team" (`toy_team`): a small two-team game whose result is **deferred** - decided by
//! episodes of the toy game (`toy.rs`) played after it - the core's test bed for deferred
//! terminals, teams and co-training (docs/search.md 1.1, 5.1; docs/az.md 11). Shaped like MTG
//! Zero's `team_draft`: a draft whose picks matter only through the matches they lead to.
//!
//! Rules. 2 or 4 seats, seat `s` in team `s % 2` (A = even seats; team size `n / 2`). A shuffled
//! deck of 16 cards (ids 0..15, card `c` is worth `c % 4` food). Each of `rounds` rounds (spec,
//! 1-3, default 2) deals an offer of `n + 1` cards from the deck, then every seat picks one card
//! of it in turn (round `r` starts at seat `r % n`), and the last card is discarded. After the
//! last pick the game is over at its **deferred terminal**: `to_move()` None, `deferred()` true,
//! `result()` and `margin()` panic until `resolve`. The payload lists the matches as toy specs
//! (`{"episodes": [{"food": [a, b]}, ...], "pairs": [[A seat, B seat], ...]}`): A player `i`
//! (seat `2i`) against B player `(i + d) mod T` (seat `2((i + d) mod T) + 1`) for every rotation
//! `d` of `0..T` (1v1: one match, 2v2: four), each player starting with its picks' total value
//! as extra food (the A player is toy seat 0). `resolve` takes the matches' results in order
//! (`[{"result": [a, b], ...}, ...]` or `[[a, b], ...]`, each 1/0, 0/1 or 0.5/0.5): a team wins
//! with more match wins; `result()[s]` = the team's 1 / 0 / 0.5 divided by the team size,
//! `margin(s)` = (team match wins - opponent match wins) / matches, `score(s)` = team match wins.
//!
//! Hidden information and chance: the undealt deck (`determinize` reshuffles it); each later
//! offer is a chance outcome, keyed `round << 32 | mask of its cards` (`force_chance` deals a
//! queued offer when its cards are still undealt, else falls back; one key consumed per deal).
//! Bots: `random` and `greedy` (the most valuable card, the lowest id on ties).
use serde_json::{json, Value};

use crate::game::{Bot, SearchGame, MAX_SEATS};
use crate::rng::Rng;

pub const CARDS: usize = 16;
pub const MAX_ROUNDS: usize = 3;
pub const MAXP: usize = 4;
pub const NUM_ACTIONS: usize = CARDS;
pub const NUM_FEATURES: usize = 64;
const NONE: u8 = u8::MAX;

/// The food a card is worth.
pub fn card_value(c: u8) -> i16 {
    (c % 4) as i16
}

#[derive(Clone, Debug)]
pub struct ToyTeamSpec {
    pub rounds: u8,
}

/// The matches' outcome (after `resolve`).
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    pub results: Vec<[f32; 2]>,
    pub wins: [u32; 2],
    pub team: [f32; 2],
}

#[derive(Clone, Debug)]
pub struct ToyTeam {
    pub n: u8,
    pub rounds: u8,
    /// Current round (0-based; = rounds once over).
    pub round: u8,
    /// Picks made in the current round.
    pub turn: u8,
    pub deck: [u8; CARDS],
    /// Cards dealt so far (`deck[..drawn]`).
    pub drawn: u8,
    /// The current offer (sorted).
    pub offer: Vec<u8>,
    pub picks: [[u8; MAX_ROUNDS]; MAXP],
    pub done: bool,
    pub outcome: Option<Outcome>,
    forced: Vec<u64>,
    fpos: usize,
}

impl ToyTeam {
    pub fn new(n: usize, rounds: u8, seed: u64) -> ToyTeam {
        assert!(n == 2 || n == 4);
        let mut rng = Rng::new(seed);
        let mut deck = [0u8; CARDS];
        for (i, c) in deck.iter_mut().enumerate() {
            *c = i as u8;
        }
        rng.shuffle(&mut deck);
        let mut g = ToyTeam { n: n as u8, rounds, round: 0, turn: 0, deck, drawn: 0, offer: Vec::new(), picks: [[NONE; MAX_ROUNDS]; MAXP], done: false, outcome: None, forced: Vec::new(), fpos: 0 };
        g.deal(&mut Vec::new(), false);
        g
    }

    pub fn team_size_of(&self) -> usize {
        self.n as usize / 2
    }

    pub fn decider(&self) -> Option<usize> {
        if self.done {
            None
        } else {
            Some((self.round as usize + self.turn as usize) % self.n as usize)
        }
    }

    /// A seat's drafted food (the total value of its picks).
    pub fn bonus(&self, seat: usize) -> i16 {
        self.picks[seat].iter().filter(|&&c| c != NONE).map(|&c| card_value(c)).sum()
    }

    /// The matches: (A seat, B seat), rotation by rotation.
    pub fn pairs(&self) -> Vec<(usize, usize)> {
        let t = self.team_size_of();
        let mut v = Vec::new();
        for d in 0..t {
            for i in 0..t {
                v.push((2 * i, 2 * ((i + d) % t) + 1));
            }
        }
        v
    }

    fn deal(&mut self, chance: &mut Vec<u64>, emit: bool) {
        let k = self.n as usize + 1;
        let d = self.drawn as usize;
        if emit && self.fpos < self.forced.len() {
            let f = self.forced[self.fpos];
            self.fpos += 1;
            let mask = f & 0xffff;
            if (f >> 32) as u8 == self.round && mask.count_ones() as usize == k && (0..d).all(|i| mask & (1 << self.deck[i]) == 0) {
                // the queued offer is still undealt: bring its cards to the top of the deck
                let mut j = d;
                for i in d..CARDS {
                    if mask & (1 << self.deck[i]) != 0 {
                        self.deck.swap(i, j);
                        j += 1;
                    }
                }
            }
        }
        self.offer = self.deck[d..d + k].to_vec();
        self.offer.sort_unstable();
        self.drawn += k as u8;
        if emit {
            let mask: u64 = self.offer.iter().map(|&c| 1u64 << c).sum();
            chance.push(((self.round as u64) << 32) | mask);
        }
    }

    fn outcome_of(&self, results: &[[f32; 2]]) -> Result<Outcome, String> {
        let want = self.pairs().len();
        if results.len() != want {
            return Err(format!("toy_team: {} results for {want} matches", results.len()));
        }
        let mut wins = [0u32; 2];
        for r in results {
            match (r[0], r[1]) {
                (a, b) if a == 1.0 && b == 0.0 => wins[0] += 1,
                (a, b) if a == 0.0 && b == 1.0 => wins[1] += 1,
                (a, b) if a == 0.5 && b == 0.5 => {}
                _ => return Err(format!("toy_team: a match result {r:?} (1/0, 0/1 or 0.5/0.5)")),
            }
        }
        let team = match wins[0].cmp(&wins[1]) {
            std::cmp::Ordering::Greater => [1.0, 0.0],
            std::cmp::Ordering::Less => [0.0, 1.0],
            std::cmp::Ordering::Equal => [0.5, 0.5],
        };
        Ok(Outcome { results: results.to_vec(), wins, team })
    }

    fn unresolved(&self, what: &str) -> ! {
        panic!("toy_team: {what}() at the deferred terminal before resolve (the search and the driver must never read it)")
    }
}

impl SearchGame for ToyTeam {
    type Spec = ToyTeamSpec;
    const NAME: &'static str = "toy_team";
    const NUM_ACTIONS: usize = NUM_ACTIONS;
    const NUM_FEATURES: usize = NUM_FEATURES;
    const MIN_PLAYERS: usize = 2;
    const MAX_PLAYERS: usize = MAXP;

    fn parse_spec(json: &str) -> Result<ToyTeamSpec, String> {
        let v: Value = if json.trim().is_empty() { json!({}) } else { serde_json::from_str(json).map_err(|e| format!("toy_team spec: {e}"))? };
        let o = v.as_object().ok_or("toy_team spec: a JSON object")?;
        let mut spec = ToyTeamSpec { rounds: 2 };
        for (k, x) in o {
            match k.as_str() {
                "rounds" => spec.rounds = x.as_u64().filter(|r| (1..=MAX_ROUNDS as u64).contains(r)).ok_or("toy_team spec: rounds 1-3")? as u8,
                _ => return Err(format!("toy_team spec: unknown key {k:?}")),
            }
        }
        Ok(spec)
    }

    fn new_game(spec: &ToyTeamSpec, players: usize, seed: u64) -> Result<Self, String> {
        if players != 2 && players != 4 {
            return Err(format!("toy_team: {players} players (2 or 4)"));
        }
        Ok(ToyTeam::new(players, spec.rounds, seed))
    }

    fn num_players(&self) -> usize {
        self.n as usize
    }

    fn to_move(&self) -> Option<usize> {
        self.decider()
    }

    fn legal_actions(&self, out: &mut Vec<u16>) {
        if !self.done {
            out.extend(self.offer.iter().map(|&c| c as u16));
        }
    }

    fn step(&mut self, a: u16, chance: &mut Vec<u64>) -> Result<(), String> {
        let Some(p) = self.decider() else { return Err("toy_team: the game is over".into()) };
        let Some(i) = self.offer.iter().position(|&c| c as u16 == a) else {
            return Err(format!("toy_team: card {a} is not in the offer {:?}", self.offer));
        };
        let c = self.offer.remove(i);
        self.picks[p][self.round as usize] = c;
        self.turn += 1;
        if self.turn == self.n {
            self.turn = 0;
            self.round += 1;
            self.offer.clear();
            if self.round == self.rounds {
                self.done = true;
            } else {
                self.deal(chance, true);
            }
        }
        Ok(())
    }

    fn force_chance(&mut self, forced: &[u64]) {
        self.forced.clear();
        self.forced.extend_from_slice(forced);
        self.fpos = 0;
    }

    /// The team's 1 / 0 / 0.5 divided by the team size; **panics** at the deferred terminal
    /// before `resolve`.
    fn result(&self) -> [f32; MAX_SEATS] {
        let n = self.n as usize;
        let mut r = [0.0; MAX_SEATS];
        match &self.outcome {
            Some(o) => {
                let t = self.team_size_of() as f32;
                for (s, x) in r.iter_mut().enumerate().take(n) {
                    *x = o.team[s % 2] / t;
                }
            }
            None if self.done => self.unresolved("result"),
            None => r[..n].fill(1.0 / n as f32),
        }
        r
    }

    fn margin(&self, seat: usize) -> f32 {
        match &self.outcome {
            Some(o) => (o.wins[seat % 2] as f32 - o.wins[1 - seat % 2] as f32) / o.results.len().max(1) as f32,
            None if self.done => self.unresolved("margin"),
            None => 0.0,
        }
    }

    fn score(&self, seat: usize) -> f32 {
        self.outcome.as_ref().map_or(0.0, |o| o.wins[seat % 2] as f32)
    }

    fn determinize(&mut self, _viewer: usize, rng: &mut Rng) {
        let d = self.drawn as usize;
        rng.shuffle(&mut self.deck[d..]);
    }

    fn encode(&self, viewer: usize, out: &mut [f32]) {
        let n = self.n as usize;
        let rel = |s: usize| (s + n - viewer) % n;
        if self.done {
            out[3] = 1.0;
        } else {
            out[self.round as usize] = 1.0;
        }
        out[4] = self.turn as f32 / 4.0;
        out[5 + self.team_size_of() - 1] = 1.0;
        if let Some(p) = self.decider() {
            out[7 + rel(p)] = 1.0;
        }
        for j in 0..n {
            let s = (viewer + j) % n;
            let o = 11 + 6 * j;
            let picks: Vec<u8> = self.picks[s].iter().copied().filter(|&c| c != NONE).collect();
            out[o] = 1.0;
            out[o + 1] = (s % 2 == viewer % 2) as u8 as f32;
            out[o + 2] = self.bonus(s) as f32 / 9.0;
            out[o + 3] = picks.len() as f32 / 3.0;
            out[o + 4] = picks.iter().map(|&c| card_value(c)).max().unwrap_or(0) as f32 / 3.0;
            out[o + 5] = (Some(s) == self.decider()) as u8 as f32;
        }
        for &c in &self.offer {
            out[35 + c as usize] = 1.0;
        }
        out[51] = self.drawn as f32 / CARDS as f32;
        out[52] = (viewer % 2) as f32;
        out[53] = self.rounds as f32 / 3.0;
    }

    fn rounds(&self) -> u32 {
        self.round as u32
    }

    fn deferred(&self) -> bool {
        self.done && self.outcome.is_none()
    }

    fn deferred_payload(&self) -> Result<String, String> {
        if !self.deferred() {
            return Err("toy_team: not at the deferred terminal".into());
        }
        let pairs = self.pairs();
        let eps: Vec<Value> = pairs.iter().map(|&(a, b)| json!({"food": [self.bonus(a), self.bonus(b)]})).collect();
        let pv: Vec<Value> = pairs.iter().map(|&(a, b)| json!([a, b])).collect();
        Ok(json!({"episodes": eps, "pairs": pv}).to_string())
    }

    fn resolve(&mut self, results: &str) -> Result<(), String> {
        if !self.deferred() {
            return Err("toy_team: nothing to resolve (not at the deferred terminal, or resolved)".into());
        }
        let v: Value = serde_json::from_str(results).map_err(|e| format!("toy_team results: {e}"))?;
        let a = v.as_array().ok_or("toy_team results: a list")?;
        let mut out = Vec::with_capacity(a.len());
        for x in a {
            let r = x.get("result").unwrap_or(x);
            let r = r.as_array().filter(|r| r.len() == 2).ok_or("toy_team results: each a pair [a, b] (or {\"result\": [a, b]})")?;
            let f = |y: &Value| y.as_f64().map(|z| z as f32).ok_or("toy_team results: numbers");
            out.push([f(&r[0])?, f(&r[1])?]);
        }
        self.outcome = Some(self.outcome_of(&out)?);
        Ok(())
    }

    fn team_size(&self) -> usize {
        self.team_size_of()
    }

    fn bot_names() -> &'static [&'static str] {
        &["greedy"]
    }

    fn make_bot(_spec: &ToyTeamSpec, name: &str, _seed: u64) -> Option<Box<dyn Bot<Self>>> {
        match name {
            "greedy" => Some(Box::new(Greedy)),
            _ => None,
        }
    }
}

/// The most valuable card, the lowest id on ties.
pub struct Greedy;

impl Bot<ToyTeam> for Greedy {
    fn choose(&mut self, _g: &ToyTeam, legal: &[u16]) -> u16 {
        let mut best = legal[0];
        for &a in legal {
            if card_value(a as u8) > card_value(best as u8) {
                best = a;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests;
