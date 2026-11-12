//! "Toy match" (`toy_match`): a best-of-three match of toy games ("Mini Cavern", `toy.rs`) between two players, each
//! game its own episode - the core's test bed for **chained episodes** (docs/search.md 1.3, 2.14; docs/az.md 42),
//! shaped like MTG Zero's best-of-three with one episode per game (its ENGINE.md 121).
//!
//! Rules. Games until a player has won two or three games have been played (a drawn game - equal toy scores -
//! counts as a game); the match's result is by game wins (1 / 0, or 0.5 each on equal wins). Game 1 starts as a toy
//! game does (its start player drawn from its seed). Before each later game the loser of the previous game (after a
//! draw: the player who chose for it, or game 1's start player) chooses who starts it (`A_FIRST`: itself,
//! `A_SECOND`: the other player; the analogue of Magic's play-or-draw choice): that decision is the first of the next
//! episode, and no game is in progress during it (`can_concede` false). The next game's seed comes from the match's
//! RNG after the choice; its setup (the first round's card and marker) is a revelation key (`1 << 40 | game << 32 |
//! the toy's round key`).
//!
//! Episodes: an episode ends at a game's end (`to_move()` None: `between`, or the match over); `result()` and
//! `margin()` are that game's. `next_episode(None)`: the match over -> `Over(the match's result)`; else the state with
//! the next game's choice pending. `next_episode(Some(p))` (p resigned the game in progress): the game counted as lost
//! by p, then the same. Features: the toy game's (the current or last game), then the match block (`MATCH_F`).
//! Spec: `{}` (no keys). Bots: `random`, `heuristic` (the toy's; it chooses to start).
use crate::game::{Bot, Next, SearchGame, MAX_SEATS};
use crate::rng::Rng;
use crate::toy::{self, ToyGame};

pub const A_FIRST: u16 = toy::NUM_ACTIONS as u16;
pub const A_SECOND: u16 = A_FIRST + 1;
pub const NUM_ACTIONS: usize = toy::NUM_ACTIONS + 2;
/// The match block: game number / 3, my wins / 2, the other's wins / 2, draws / 3, a choice pending, I choose.
pub const MATCH_F: usize = 6;
pub const NUM_FEATURES: usize = toy::NUM_FEATURES + MATCH_F;
pub const GAMES: u8 = 3;
pub const WINS: u8 = 2;

#[derive(Clone, Debug)]
pub struct ToyMatch {
    /// The current (or last) game.
    pub g: ToyGame,
    /// The current (or last) game, 1-based.
    pub game_no: u8,
    /// Per game: 0 not played, 1 + p = p won, 3 = a draw.
    pub results: [u8; 3],
    /// Per game: who chose its start player (game 1: its start player).
    pub choosers: [u8; 3],
    pub wins: [u8; 2],
    pub draws: u8,
    /// Rounds of the games before the current one.
    pub rounds_before: u32,
    /// The match is over.
    pub over: bool,
    /// A game ended, the match did not: the episode is over (`next_episode` begins the next one).
    pub between: bool,
    /// The next game's start choice pending: the chooser.
    pub choosing: Option<u8>,
    /// Seeds the later games (`determinize` reseeds it).
    pub rng: Rng,
}

impl ToyMatch {
    pub fn new(seed: u64) -> ToyMatch {
        let g = ToyGame::new(2, seed);
        let start = g.start;
        ToyMatch {
            g,
            game_no: 1,
            results: [0; 3],
            choosers: [start, 0, 0],
            wins: [0; 2],
            draws: 0,
            rounds_before: 0,
            over: false,
            between: false,
            choosing: None,
            rng: Rng::new(seed ^ 0x7079_6d61_7463_6821),
        }
    }

    /// The current game ended (`winner` None: a draw): scored; the match over, or the episode over (`between`).
    fn score_game(&mut self, winner: Option<u8>) {
        let i = (self.game_no - 1) as usize;
        self.rounds_before += self.g.round as u32;
        match winner {
            Some(w) => {
                self.wins[w as usize] += 1;
                self.results[i] = 1 + w;
            }
            None => {
                self.draws += 1;
                self.results[i] = 3;
            }
        }
        if self.wins.iter().any(|&w| w >= WINS) || self.game_no >= GAMES {
            self.over = true;
        } else {
            self.between = true;
        }
    }

    /// The finished game's winner by the toy's result (None: equal scores).
    fn game_winner(&self) -> Option<u8> {
        let r = SearchGame::result(&self.g);
        if r[0] > r[1] {
            Some(0)
        } else if r[1] > r[0] {
            Some(1)
        } else {
            None
        }
    }

    /// The match's result by game wins.
    pub fn match_result(&self) -> [f32; MAX_SEATS] {
        let mut r = [0.0; MAX_SEATS];
        match self.wins[0].cmp(&self.wins[1]) {
            std::cmp::Ordering::Greater => r[0] = 1.0,
            std::cmp::Ordering::Less => r[1] = 1.0,
            std::cmp::Ordering::Equal => {
                r[0] = 0.5;
                r[1] = 0.5;
            }
        }
        r
    }

    /// From `between`: the next game's start choice pending (the loser of the last game chooses; after a draw, the
    /// player who chose for it).
    fn begin_choice(&mut self) {
        let i = (self.game_no - 1) as usize;
        let chooser = match self.results[i] {
            r @ (1 | 2) => 2 - r, // (the loser: p won -> 1 - p)
            _ => self.choosers[i],
        };
        self.between = false;
        self.choosing = Some(chooser);
    }

    /// The chooser chose `a`: the next game, its seed from the match's RNG, its start player as chosen.
    fn start_next(&mut self, a: u16, chance: &mut Vec<u64>) {
        let c = self.choosing.take().expect("a choice pending");
        let start = if a == A_FIRST { c } else { 1 - c };
        let seed = self.rng.next_u64();
        let mut g = ToyGame::new(2, seed);
        g.start = start;
        g.cur = start;
        for k in 0..2usize {
            g.players[(start as usize + k) % 2].food = 2 + k as i16;
        }
        self.g = g;
        self.game_no += 1;
        self.choosers[(self.game_no - 1) as usize] = c;
        let first = (1u64 << 16) | ((self.g.order[0] as u64) << 8) | self.g.markers[0] as u64;
        chance.push((1u64 << 40) | ((self.game_no as u64) << 32) | first);
    }
}

impl SearchGame for ToyMatch {
    type Spec = ();
    const NAME: &'static str = "toy_match";
    const NUM_ACTIONS: usize = NUM_ACTIONS;
    const NUM_FEATURES: usize = NUM_FEATURES;
    const MIN_PLAYERS: usize = 2;
    const MAX_PLAYERS: usize = 2;

    fn parse_spec(json: &str) -> Result<(), String> {
        let v: serde_json::Value = if json.trim().is_empty() { serde_json::json!({}) } else { serde_json::from_str(json).map_err(|e| format!("toy_match spec: {e}"))? };
        let o = v.as_object().ok_or("toy_match spec: a JSON object")?;
        if let Some(k) = o.keys().next() {
            return Err(format!("toy_match spec: unknown key {k:?} (it has none)"));
        }
        Ok(())
    }

    fn new_game(_spec: &(), players: usize, seed: u64) -> Result<Self, String> {
        if players != 2 {
            return Err(format!("toy_match: {players} players (two only)"));
        }
        Ok(ToyMatch::new(seed))
    }

    fn num_players(&self) -> usize {
        2
    }

    fn to_move(&self) -> Option<usize> {
        if self.over || self.between {
            return None;
        }
        if let Some(c) = self.choosing {
            return Some(c as usize);
        }
        SearchGame::to_move(&self.g)
    }

    fn legal_actions(&self, out: &mut Vec<u16>) {
        if self.over || self.between {
            out.clear();
            return;
        }
        if self.choosing.is_some() {
            out.clear();
            out.extend_from_slice(&[A_FIRST, A_SECOND]);
            return;
        }
        out.clear();
        self.g.legal(out);
    }

    fn step(&mut self, a: u16, chance: &mut Vec<u64>) -> Result<(), String> {
        if self.over || self.between {
            return Err("toy_match: the episode is over".into());
        }
        if self.choosing.is_some() {
            if a != A_FIRST && a != A_SECOND {
                return Err(format!("toy_match: illegal action {a} for the start choice"));
            }
            self.start_next(a, chance);
            return Ok(());
        }
        self.g.apply(a, chance)?;
        if self.g.over {
            let w = self.game_winner();
            self.score_game(w);
        }
        Ok(())
    }

    fn force_chance(&mut self, forced: &[u64]) {
        SearchGame::force_chance(&mut self.g, forced)
    }

    /// The current (at an episode's end: the finished) game's result.
    fn result(&self) -> [f32; MAX_SEATS] {
        SearchGame::result(&self.g)
    }

    fn margin(&self, seat: usize) -> f32 {
        SearchGame::margin(&self.g, seat)
    }

    /// Game wins.
    fn score(&self, seat: usize) -> f32 {
        self.wins[seat] as f32
    }

    fn determinize(&mut self, viewer: usize, rng: &mut Rng) {
        if self.choosing.is_none() && !self.g.over {
            SearchGame::determinize(&mut self.g, viewer, rng);
        }
        // (the later games' seeds are hidden)
        self.rng = Rng::new(rng.next_u64());
    }

    fn encode(&self, viewer: usize, out: &mut [f32]) {
        SearchGame::encode(&self.g, viewer, &mut out[..toy::NUM_FEATURES]);
        let x = &mut out[toy::NUM_FEATURES..];
        x[0] = self.game_no as f32 / 3.0;
        x[1] = self.wins[viewer] as f32 / 2.0;
        x[2] = self.wins[1 - viewer] as f32 / 2.0;
        x[3] = self.draws as f32 / 3.0;
        x[4] = self.choosing.is_some() as u8 as f32;
        x[5] = (self.choosing == Some(viewer as u8)) as u8 as f32;
    }

    fn rounds(&self) -> u32 {
        self.rounds_before + if self.over || self.between || self.choosing.is_some() { 0 } else { self.g.round as u32 }
    }

    fn export_state(&self) -> String {
        format!(
            "{{\"game\":{},\"wins\":{:?},\"draws\":{},\"over\":{},\"between\":{},\"choosing\":{},\"toy\":{}}}",
            self.game_no,
            self.wins,
            self.draws,
            self.over,
            self.between,
            self.choosing.map_or("null".to_string(), |c| c.to_string()),
            self.g.to_json()
        )
    }

    fn bot_names() -> &'static [&'static str] {
        &["heuristic"]
    }

    fn make_bot(_spec: &(), name: &str, _seed: u64) -> Option<Box<dyn Bot<Self>>> {
        match name {
            "heuristic" => Some(Box::new(MatchHeuristic)),
            _ => None,
        }
    }

    const AUX: &'static [(&'static str, usize)] = &[("next_card", toy::N_CARDS)];

    fn aux_target(&self, viewer: usize, out: &mut [f32]) {
        if self.choosing.is_none() {
            SearchGame::aux_target(&self.g, viewer, out);
        }
    }

    fn chained(&self) -> bool {
        true
    }

    fn next_episode(&self, conceded: Option<usize>) -> Result<Next<Self>, String> {
        let mut m = self.clone();
        match conceded {
            None => {
                if !(m.over || m.between) {
                    return Err("toy_match: next_episode before the game's end".into());
                }
            }
            Some(p) => {
                if m.over || m.between || m.choosing.is_some() || p > 1 {
                    return Err(format!("toy_match: seat {p} concedes, but no game is in progress"));
                }
                // (the game in progress counted as lost by the conceding seat)
                m.score_game(Some(1 - p as u8));
            }
        }
        if m.over {
            return Ok(Next::Over(m.match_result()));
        }
        m.begin_choice();
        Ok(Next::Continue(m))
    }

    /// No game is in progress during the start choice.
    fn can_concede(&self) -> bool {
        self.choosing.is_none()
    }

    fn minor_decision(&self) -> bool {
        false
    }
}

/// The toy's heuristic as a match player; it chooses to start.
pub struct MatchHeuristic;

impl Bot<ToyMatch> for MatchHeuristic {
    fn choose(&mut self, g: &ToyMatch, legal: &[u16]) -> u16 {
        if g.choosing.is_some() {
            return A_FIRST;
        }
        g.g.heuristic(legal)
    }
}

