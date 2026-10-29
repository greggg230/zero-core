//! "Mini Cavern": a small worker-placement game shaped like Caverna, used to prove the search
//! and the AlphaZero loop end to end (docs/search.md 5). What it exercises:
//!
//! * 2-4 players, several workers each, turns in seat order from the start player;
//! * **multi-step turns**: placing on Farm asks plow? (PLOW / SKIP) then a counts decision for
//!   sowing (INC0 ... DONE); Build, Trade and Feast are counts decisions too (the partial
//!   count is part of the state and of the features, like the AZ layers' partial
//!   compound responses of Caverna and Glass Road);
//! * **chance with hidden future outcomes**: one round card per round, revealed at the start
//!   of the round from a hidden order within its stage (rounds 1-3: Grain / Growth / Build,
//!   rounds 4-6: Urgent growth / Trade / Feast), and a hidden harvest marker per round 2-5
//!   (two full harvests, two feed-only rounds, in a hidden order; round 6 is a full harvest);
//!   `determinize` reshuffles both, `force_chance` replays a sampled reveal;
//! * integer scores with **ties** (split).
//!
//! Rules. Each player starts with 2 workers, 2 rooms, food 2 + its position after the start
//! player. Spaces (one worker each): Timber (+2 wood per round, accumulates), Quarry (+1
//! stone, accumulates), Fishing (+2 food, accumulates), Start (1 food, start player next
//! round), Farm (plow one field for 1 wood, then sow: each sown grain fills an empty field
//! with 3 grain), and with 3+ players Timber2 (+1 wood, accumulates) and Grain (1 grain).
//! Round cards stay available once revealed: Grain (2 grain), Growth (+1 worker if rooms >
//! workers, else 2 food), Build (rooms for 2 wood + 1 stone each, at most 4 rooms), Urgent
//! growth (+1 worker up to 4, else 3 food), Trade (1 stone -> 2 food, 1 wood -> 1 food, any
//! number), Feast (1 grain -> 2 food, any number). A worker born this round places from the
//! next round. Harvest (full): every sown field yields 1 grain, then every worker eats 2
//! food (a newborn 1); feed-only: every adult eats 1 food. Missing food is paid with grain,
//! then 1 begging marker per missing food. Score: 3 per worker, 1 per room beyond 2, 1 per
//! plowed field, 1 per 2 grain (supply + fields), 1 per stone, -3 per begging marker.
//!
//! Spec: `{}`, or `{"food": [a, b, ...]}`: extra starting food per seat (the matches of the
//! team toy game `toy_team`, whose drafted cards are food; `toy_team.rs`).
use crate::game::{Bot, SearchGame, MAX_SEATS};
use crate::rng::Rng;

pub const ROUNDS: u8 = 6;
pub const N_SPACES: usize = 13;
pub const N_CARDS: usize = 6;
pub const MAXP: usize = 4;

pub const S_TIMBER: usize = 0;
pub const S_QUARRY: usize = 1;
pub const S_FISH: usize = 2;
pub const S_START: usize = 3;
pub const S_FARM: usize = 4;
pub const S_TIMBER2: usize = 5;
pub const S_GRAIN: usize = 6;
/// Space of round card c = CARD0 + c.
pub const CARD0: usize = 7;
pub const C_GRAIN: u8 = 0;
pub const C_GROWTH: u8 = 1;
pub const C_BUILD: u8 = 2;
pub const C_URGENT: u8 = 3;
pub const C_TRADE: u8 = 4;
pub const C_FEAST: u8 = 5;

pub const M_NONE: u8 = 0;
pub const M_HARVEST: u8 = 1;
pub const M_FEED: u8 = 2;

/// Actions: PLACE(space) = space id (0..13), then the sub-decision actions.
pub const A_PLOW: u16 = 13;
pub const A_SKIP: u16 = 14;
pub const A_INC0: u16 = 15;
pub const A_INC1: u16 = 16;
pub const A_DONE: u16 = 17;
pub const NUM_ACTIONS: usize = 18;

pub const SPACE_NAMES: [&str; N_SPACES] = ["timber", "quarry", "fishing", "start", "farm", "timber2", "grain", "c_grain", "c_growth", "c_build", "c_urgent", "c_trade", "c_feast"];
pub const ACTION_NAMES: [&str; NUM_ACTIONS] = ["timber", "quarry", "fishing", "start", "farm", "timber2", "grain", "c_grain", "c_growth", "c_build", "c_urgent", "c_trade", "c_feast", "plow", "skip", "inc0", "inc1", "done"];

const FREE: u8 = 255;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sub {
    Place,
    Plow,
    Sow(u8),
    Build(u8),
    Trade(u8, u8),
    Feast(u8),
}

impl Sub {
    fn index(self) -> usize {
        match self {
            Sub::Place => 0,
            Sub::Plow => 1,
            Sub::Sow(_) => 2,
            Sub::Build(_) => 3,
            Sub::Trade(..) => 4,
            Sub::Feast(_) => 5,
        }
    }
    fn name(self) -> &'static str {
        ["place", "plow", "sow", "build", "trade", "feast"][self.index()]
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Player {
    /// Workers incl. newborns.
    pub workers: u8,
    /// Born this round (do not place until the next round).
    pub newborn: u8,
    pub placed: u8,
    pub food: i16,
    pub wood: i16,
    pub stone: i16,
    pub grain: i16,
    pub plowed: u8,
    pub field: [u8; 4],
    pub rooms: u8,
    pub beg: u8,
}

impl Player {
    fn avail(&self) -> u8 {
        self.workers - self.newborn - self.placed
    }
    fn empty_fields(&self) -> u8 {
        (0..self.plowed as usize).filter(|&i| self.field[i] == 0).count() as u8
    }
    fn field_grain(&self) -> i16 {
        self.field[..self.plowed as usize].iter().map(|&g| g as i16).sum()
    }
    pub fn score(&self) -> i32 {
        3 * self.workers as i32 + (self.rooms as i32 - 2) + self.plowed as i32 + (self.grain + self.field_grain()) as i32 / 2 + self.stone as i32 - 3 * self.beg as i32
    }
}

#[derive(Clone, Debug, Default)]
pub struct ToySpec {
    /// Extra starting food per seat (`{"food": [...]}`; empty: none).
    pub food: Vec<i16>,
}

#[derive(Clone, Debug)]
pub struct ToyGame {
    pub n: u8,
    /// Current round 1..=6.
    pub round: u8,
    pub start: u8,
    /// Who took Start this round (FREE = nobody).
    pub start_taker: u8,
    pub cur: u8,
    pub sub: Sub,
    pub over: bool,
    /// Round card per round (index round - 1); revealed for rounds <= `round`.
    pub order: [u8; ROUNDS as usize],
    /// Harvest marker per round (index round - 1); revealed for rounds <= `round`.
    pub markers: [u8; ROUNDS as usize],
    pub occ: [u8; N_SPACES],
    pub goods: [u8; N_SPACES],
    pub players: [Player; MAXP],
    forced: [u64; 4],
    nforced: u8,
}

fn chance_key(round: u8, card: u8, marker: u8) -> u64 {
    ((round as u64) << 16) | ((card as u64) << 8) | marker as u64
}

/// Index range of the round cards in the same stage as round r (1-based) -> 0-based range.
fn stage_range(r: u8) -> std::ops::Range<usize> {
    if r <= 3 {
        0..3
    } else {
        3..6
    }
}

impl ToyGame {
    pub fn new(n: usize, seed: u64) -> ToyGame {
        assert!((2..=MAXP).contains(&n));
        let mut rng = Rng::new(seed);
        let mut order = [C_GRAIN, C_GROWTH, C_BUILD, C_URGENT, C_TRADE, C_FEAST];
        rng.shuffle(&mut order[0..3]);
        rng.shuffle(&mut order[3..6]);
        let mut markers = [M_NONE, M_HARVEST, M_HARVEST, M_FEED, M_FEED, M_HARVEST];
        rng.shuffle(&mut markers[1..5]);
        let start = rng.below(n as u32) as u8;
        let mut players = [Player::default(); MAXP];
        for k in 0..n {
            let p = &mut players[(start as usize + k) % n];
            p.workers = 2;
            p.rooms = 2;
            p.food = 2 + k as i16;
        }
        let mut g = ToyGame {
            n: n as u8,
            round: 0,
            start,
            start_taker: FREE,
            cur: start,
            sub: Sub::Place,
            over: false,
            order,
            markers,
            occ: [FREE; N_SPACES],
            goods: [0; N_SPACES],
            players,
            forced: [0; 4],
            nforced: 0,
        };
        let mut sink = Vec::new();
        g.begin_round(&mut sink);
        g
    }

    pub fn present(&self, s: usize) -> bool {
        match s {
            0..=4 => true,
            5 | 6 => self.n >= 3,
            _ => self.card_revealed((s - CARD0) as u8),
        }
    }

    pub fn card_revealed(&self, c: u8) -> bool {
        (0..self.round as usize).any(|i| self.order[i] == c)
    }

    pub fn scores(&self) -> [i32; MAXP] {
        let mut s = [0; MAXP];
        for i in 0..self.n as usize {
            s[i] = self.players[i].score();
        }
        s
    }

    fn begin_round(&mut self, chance: &mut Vec<u64>) {
        self.round += 1;
        let r = self.round;
        let ri = r as usize - 1;
        if self.nforced > 0 {
            let k = self.forced[0];
            self.forced.copy_within(1..4, 0);
            self.nforced -= 1;
            if (k >> 16) as u8 == r {
                let c = ((k >> 8) & 0xff) as u8;
                let m = (k & 0xff) as u8;
                if self.order[ri] != c {
                    if let Some(j) = (ri..stage_range(r).end).find(|&j| self.order[j] == c) {
                        self.order.swap(ri, j);
                    }
                }
                if (2..=5).contains(&r) && self.markers[ri] != m {
                    if let Some(j) = (ri..5).find(|&j| self.markers[j] == m) {
                        self.markers.swap(ri, j);
                    }
                }
            }
        }
        chance.push(chance_key(r, self.order[ri], self.markers[ri]));
        for p in self.players[..self.n as usize].iter_mut() {
            p.newborn = 0;
            p.placed = 0;
        }
        self.occ = [FREE; N_SPACES];
        self.goods[S_TIMBER] += 2;
        self.goods[S_QUARRY] += 1;
        self.goods[S_FISH] += 2;
        if self.n >= 3 {
            self.goods[S_TIMBER2] += 1;
        }
        self.start_taker = FREE;
        self.cur = self.start;
        self.sub = Sub::Place;
    }

    fn feed(&mut self, marker: u8) {
        for i in 0..self.n as usize {
            let p = &mut self.players[i];
            let adults = (p.workers - p.newborn) as i16;
            let need = match marker {
                M_HARVEST => {
                    for f in 0..p.plowed as usize {
                        if p.field[f] > 0 {
                            p.field[f] -= 1;
                            p.grain += 1;
                        }
                    }
                    2 * adults + p.newborn as i16
                }
                M_FEED => adults,
                _ => 0,
            };
            if p.food >= need {
                p.food -= need;
            } else {
                let mut short = need - p.food;
                p.food = 0;
                let g = short.min(p.grain);
                p.grain -= g;
                short -= g;
                p.beg += short as u8;
            }
        }
    }

    fn end_turn(&mut self, chance: &mut Vec<u64>) {
        self.sub = Sub::Place;
        let n = self.n as usize;
        for k in 1..=n {
            let q = (self.cur as usize + k) % n;
            if self.players[q].avail() > 0 {
                self.cur = q as u8;
                return;
            }
        }
        // round over
        self.feed(self.markers[self.round as usize - 1]);
        if self.round == ROUNDS {
            self.over = true;
            return;
        }
        if self.start_taker != FREE {
            self.start = self.start_taker;
        }
        self.begin_round(chance);
    }

    fn can_plow(p: &Player) -> bool {
        p.wood >= 1 && p.plowed < 4
    }
    fn max_sow(p: &Player) -> u8 {
        (p.grain.max(0) as u8).min(p.empty_fields())
    }
    fn can_build(p: &Player, k: u8) -> bool {
        let k1 = k as i16 + 1;
        p.wood >= 2 * k1 && p.stone >= k1 && p.rooms + k < 4
    }

    fn enter_sow(&mut self, chance: &mut Vec<u64>) {
        if Self::max_sow(&self.players[self.cur as usize]) > 0 {
            self.sub = Sub::Sow(0);
        } else {
            self.end_turn(chance);
        }
    }

    fn place(&mut self, s: usize, chance: &mut Vec<u64>) {
        let c = self.cur as usize;
        self.occ[s] = c as u8;
        self.players[c].placed += 1;
        let goods = std::mem::take(&mut self.goods[s]) as i16;
        let p = &mut self.players[c];
        match s {
            S_TIMBER | S_TIMBER2 => p.wood += goods,
            S_QUARRY => p.stone += goods,
            S_FISH => p.food += goods,
            S_START => {
                p.food += 1;
                self.start_taker = c as u8;
            }
            S_FARM => {
                if Self::can_plow(p) {
                    self.sub = Sub::Plow;
                    return;
                }
                return self.enter_sow(chance);
            }
            S_GRAIN => p.grain += 1,
            _ => match (s - CARD0) as u8 {
                C_GRAIN => p.grain += 2,
                C_GROWTH => {
                    if p.workers < p.rooms && p.workers < 4 {
                        p.workers += 1;
                        p.newborn += 1;
                    } else {
                        p.food += 2;
                    }
                }
                C_URGENT => {
                    if p.workers < 4 {
                        p.workers += 1;
                        p.newborn += 1;
                    } else {
                        p.food += 3;
                    }
                }
                C_BUILD => {
                    if Self::can_build(p, 0) {
                        self.sub = Sub::Build(0);
                        return;
                    }
                }
                C_TRADE => {
                    if p.stone >= 1 || p.wood >= 1 {
                        self.sub = Sub::Trade(0, 0);
                        return;
                    }
                }
                C_FEAST => {
                    if p.grain >= 1 {
                        self.sub = Sub::Feast(0);
                        return;
                    }
                }
                _ => unreachable!(),
            },
        }
        self.end_turn(chance);
    }

    pub fn legal(&self, out: &mut Vec<u16>) {
        if self.over {
            return;
        }
        let p = &self.players[self.cur as usize];
        match self.sub {
            Sub::Place => {
                for s in 0..N_SPACES {
                    if self.occ[s] == FREE && self.present(s) {
                        out.push(s as u16);
                    }
                }
            }
            Sub::Plow => out.extend_from_slice(&[A_PLOW, A_SKIP]),
            Sub::Sow(k) => {
                if k < Self::max_sow(p) {
                    out.push(A_INC0);
                }
                out.push(A_DONE);
            }
            Sub::Build(k) => {
                if Self::can_build(p, k) {
                    out.push(A_INC0);
                }
                out.push(A_DONE);
            }
            Sub::Trade(a, b) => {
                if p.stone > a as i16 {
                    out.push(A_INC0);
                }
                if p.wood > b as i16 {
                    out.push(A_INC1);
                }
                out.push(A_DONE);
            }
            Sub::Feast(k) => {
                if (k as i16) < p.grain {
                    out.push(A_INC0);
                }
                out.push(A_DONE);
            }
        }
    }

    pub fn apply(&mut self, a: u16, chance: &mut Vec<u64>) -> Result<(), String> {
        let mut legal = Vec::with_capacity(16);
        self.legal(&mut legal);
        if !legal.contains(&a) {
            return Err(format!("illegal action {a} ({}) in sub-decision {:?}; legal {legal:?}", ACTION_NAMES.get(a as usize).unwrap_or(&"?"), self.sub));
        }
        let c = self.cur as usize;
        match self.sub {
            Sub::Place => self.place(a as usize, chance),
            Sub::Plow => {
                if a == A_PLOW {
                    let p = &mut self.players[c];
                    p.wood -= 1;
                    p.field[p.plowed as usize] = 0;
                    p.plowed += 1;
                }
                self.enter_sow(chance);
            }
            Sub::Sow(k) => {
                if a == A_INC0 {
                    self.sub = Sub::Sow(k + 1);
                } else {
                    let p = &mut self.players[c];
                    let mut left = k;
                    for f in 0..p.plowed as usize {
                        if left > 0 && p.field[f] == 0 {
                            p.field[f] = 3;
                            left -= 1;
                        }
                    }
                    p.grain -= k as i16;
                    self.end_turn(chance);
                }
            }
            Sub::Build(k) => {
                if a == A_INC0 {
                    self.sub = Sub::Build(k + 1);
                } else {
                    let p = &mut self.players[c];
                    p.rooms += k;
                    p.wood -= 2 * k as i16;
                    p.stone -= k as i16;
                    self.end_turn(chance);
                }
            }
            Sub::Trade(x, y) => match a {
                A_INC0 => self.sub = Sub::Trade(x + 1, y),
                A_INC1 => self.sub = Sub::Trade(x, y + 1),
                _ => {
                    let p = &mut self.players[c];
                    p.stone -= x as i16;
                    p.wood -= y as i16;
                    p.food += 2 * x as i16 + y as i16;
                    self.end_turn(chance);
                }
            },
            Sub::Feast(k) => {
                if a == A_INC0 {
                    self.sub = Sub::Feast(k + 1);
                } else {
                    let p = &mut self.players[c];
                    p.grain -= k as i16;
                    p.food += 2 * k as i16;
                    self.end_turn(chance);
                }
            }
        }
        Ok(())
    }

    /// The deterministic heuristic yardstick (also ported to Python: `zero/toybots.py`;
    /// both must choose identically). Plans food for the coming feeding with a buffer, grows
    /// only with food to spare, builds rooms to grow, farms, takes points in the last round.
    pub fn heuristic(&self, legal: &[u16]) -> u16 {
        let has = |a: u16| legal.contains(&a);
        let p = &self.players[self.cur as usize];
        let marker = self.markers[self.round as usize - 1];
        let adults = (p.workers - p.newborn) as i16;
        let sown = p.field[..p.plowed as usize].iter().filter(|&&g| g > 0).count() as i16;
        let need = if marker == M_FEED { adults } else { 2 * adults + p.newborn as i16 };
        let have = p.food + p.grain + if marker == M_HARVEST { sown } else { 0 };
        match self.sub {
            Sub::Plow => {
                if has(A_PLOW) {
                    A_PLOW
                } else {
                    A_SKIP
                }
            }
            Sub::Sow(_) | Sub::Build(_) => {
                if has(A_INC0) {
                    A_INC0
                } else {
                    A_DONE
                }
            }
            Sub::Trade(x, y) => {
                if have + 2 * x as i16 + (y as i16) < need + 1 {
                    if has(A_INC0) {
                        return A_INC0;
                    }
                    if has(A_INC1) {
                        return A_INC1;
                    }
                }
                A_DONE
            }
            Sub::Feast(k) => {
                if have + (k as i16) < need + 2 && has(A_INC0) {
                    A_INC0
                } else {
                    A_DONE
                }
            }
            Sub::Place => {
                let free = |s: usize| has(s as u16);
                let card = |c: u8| CARD0 + c as usize;
                let mut pick: Option<usize> = None;
                let mut want = |cond: bool, s: usize| {
                    if pick.is_none() && cond && free(s) {
                        pick = Some(s);
                    }
                };
                let short = have < need;
                let spare = have >= need + 2;
                let last = self.round == ROUNDS;
                want(short && self.goods[S_FISH] >= 2, S_FISH);
                want(short && p.grain >= 1, card(C_FEAST));
                want(short && (p.stone >= 1 || p.wood >= 1), card(C_TRADE));
                want(short && self.goods[S_FISH] >= 1, S_FISH);
                want(short, S_START);
                want(!last && spare && p.workers < p.rooms && p.workers < 4, card(C_GROWTH));
                want(!last && spare && p.workers < 4, card(C_URGENT));
                want(self.round <= 4 && Self::can_build(p, 0) && p.rooms <= p.workers, card(C_BUILD));
                want(p.grain >= 1 && (p.empty_fields() > 0 || Self::can_plow(p)), S_FARM);
                want(last && self.goods[S_QUARRY] >= 1, S_QUARRY);
                want(true, card(C_GRAIN));
                want(true, S_GRAIN);
                want(self.goods[S_FISH] >= 3, S_FISH);
                want(self.goods[S_TIMBER] >= 3 || (p.wood < 2 && self.goods[S_TIMBER] >= 2), S_TIMBER);
                want(self.goods[S_QUARRY] >= 2 || (p.stone < 1 && self.goods[S_QUARRY] >= 1), S_QUARRY);
                for s in [S_FISH, S_TIMBER, S_QUARRY, S_TIMBER2, S_START, S_FARM] {
                    want(true, s);
                }
                match pick {
                    Some(s) => s as u16,
                    None => legal[0],
                }
            }
        }
    }

    /// The state as JSON for an external seat (`zero/toybots.py` reads it).
    pub fn to_json(&self) -> String {
        use serde_json::json;
        let n = self.n as usize;
        let rv = self.round as usize;
        let players: Vec<_> = (0..n)
            .map(|i| {
                let p = &self.players[i];
                json!({"workers": p.workers, "newborn": p.newborn, "placed": p.placed, "food": p.food, "wood": p.wood,
                       "stone": p.stone, "grain": p.grain, "plowed": p.plowed, "field": &p.field[..p.plowed as usize],
                       "rooms": p.rooms, "beg": p.beg, "score": p.score()})
            })
            .collect();
        let (k, a, b) = match self.sub {
            Sub::Sow(k) | Sub::Build(k) | Sub::Feast(k) => (k, 0, 0),
            Sub::Trade(a, b) => (0, a, b),
            _ => (0, 0, 0),
        };
        let mut legal = Vec::new();
        self.legal(&mut legal);
        let v = json!({
            "game": "toy", "n": n, "round": self.round, "start": self.start, "cur": self.cur, "over": self.over,
            "sub": self.sub.name(), "sub_k": k, "sub_a": a, "sub_b": b,
            "cards": (0..ROUNDS as usize).map(|i| if i < rv { self.order[i] as i32 } else { -1 }).collect::<Vec<_>>(),
            "markers": (0..ROUNDS as usize).map(|i| if i < rv { self.markers[i] as i32 } else { -1 }).collect::<Vec<_>>(),
            "occ": self.occ.iter().map(|&o| if o == FREE { -1 } else { o as i32 }).collect::<Vec<_>>(),
            "goods": self.goods.to_vec(),
            "present": (0..N_SPACES).map(|s| self.present(s)).collect::<Vec<_>>(),
            "players": players,
            "legal": legal,
        });
        v.to_string()
    }
}

/// Feature layout (viewer-relative seats j = (viewer + j) % n).
pub const NUM_FEATURES: usize = 6 + 3 + 4 + 12 + 6 + N_SPACES * 7 + MAXP * 16 + 9;

impl SearchGame for ToyGame {
    type Spec = ToySpec;
    const NAME: &'static str = "toy";
    const NUM_ACTIONS: usize = NUM_ACTIONS;
    const NUM_FEATURES: usize = NUM_FEATURES;
    const MIN_PLAYERS: usize = 2;
    const MAX_PLAYERS: usize = MAXP;

    fn parse_spec(json: &str) -> Result<ToySpec, String> {
        let v: serde_json::Value = if json.trim().is_empty() { serde_json::json!({}) } else { serde_json::from_str(json).map_err(|e| format!("toy spec: {e}"))? };
        let o = v.as_object().ok_or("toy spec: a JSON object")?;
        let mut spec = ToySpec::default();
        for (k, x) in o {
            match k.as_str() {
                "food" => {
                    let a = x.as_array().filter(|a| (1..=MAXP).contains(&a.len())).ok_or("toy spec: food is a list of 1-4 numbers")?;
                    for f in a {
                        let f = f.as_i64().filter(|f| (0..=99).contains(f)).ok_or("toy spec: food values are integers 0..99")?;
                        spec.food.push(f as i16);
                    }
                }
                _ => return Err(format!("toy spec: unknown key {k:?}")),
            }
        }
        Ok(spec)
    }

    fn new_game(spec: &ToySpec, players: usize, seed: u64) -> Result<Self, String> {
        if !(2..=MAXP).contains(&players) {
            return Err(format!("toy: {players} players (2..4)"));
        }
        if spec.food.len() > players {
            return Err(format!("toy: food for {} seats, {players} players", spec.food.len()));
        }
        let mut g = ToyGame::new(players, seed);
        for (i, &f) in spec.food.iter().enumerate() {
            g.players[i].food += f;
        }
        Ok(g)
    }

    fn num_players(&self) -> usize {
        self.n as usize
    }

    fn to_move(&self) -> Option<usize> {
        if self.over {
            None
        } else {
            Some(self.cur as usize)
        }
    }

    fn legal_actions(&self, out: &mut Vec<u16>) {
        self.legal(out)
    }

    fn step(&mut self, a: u16, chance: &mut Vec<u64>) -> Result<(), String> {
        self.apply(a, chance)
    }

    fn force_chance(&mut self, forced: &[u64]) {
        let k = forced.len().min(4);
        self.forced[..k].copy_from_slice(&forced[..k]);
        self.nforced = k as u8;
    }

    fn result(&self) -> [f32; MAX_SEATS] {
        let s = self.scores();
        let n = self.n as usize;
        let best = s[..n].iter().copied().max().unwrap();
        let k = s[..n].iter().filter(|&&x| x == best).count() as f32;
        let mut r = [0.0; MAX_SEATS];
        for i in 0..n {
            if s[i] == best {
                r[i] = 1.0 / k;
            }
        }
        r
    }

    fn margin(&self, seat: usize) -> f32 {
        let s = self.scores();
        let other = (0..self.n as usize).filter(|&i| i != seat).map(|i| s[i]).max().unwrap();
        (s[seat] - other) as f32 / 10.0
    }

    fn score(&self, seat: usize) -> f32 {
        self.players[seat].score() as f32
    }

    fn determinize(&mut self, _viewer: usize, rng: &mut Rng) {
        // everybody sees the same: unrevealed round cards within their stage, unrevealed
        // harvest markers among rounds 2-5
        let r = self.round as usize;
        if r < 3 {
            rng.shuffle(&mut self.order[r..3]);
        }
        let lo = r.max(3);
        if lo < 6 {
            rng.shuffle(&mut self.order[lo..6]);
        }
        let lo = r.max(1);
        if lo < 5 {
            rng.shuffle(&mut self.markers[lo..5]);
        }
    }

    fn encode(&self, viewer: usize, out: &mut [f32]) {
        let n = self.n as usize;
        let rel = |s: usize| (s + n - viewer) % n;
        let mut i = 0;
        // round one-hot
        out[i + (self.round as usize - 1).min(5)] = 1.0;
        i += 6;
        out[i + n - 2] = 1.0;
        i += 3;
        out[i + rel(self.start as usize)] = 1.0;
        i += 4;
        for r in 0..ROUNDS as usize {
            if r < self.round as usize {
                match self.markers[r] {
                    M_HARVEST => out[i + 2 * r] = 1.0,
                    M_FEED => out[i + 2 * r + 1] = 1.0,
                    _ => {}
                }
            }
        }
        i += 12;
        for c in 0..N_CARDS {
            if self.card_revealed(c as u8) {
                out[i + c] = 1.0;
            }
        }
        i += 6;
        for s in 0..N_SPACES {
            if self.present(s) {
                out[i] = 1.0;
                if self.occ[s] != FREE {
                    out[i + 1] = 1.0;
                    out[i + 2 + rel(self.occ[s] as usize)] = 1.0;
                }
                out[i + 6] = self.goods[s] as f32 / 4.0;
            }
            i += 7;
        }
        let sc = self.scores();
        for j in 0..MAXP {
            if j < n {
                let s = (viewer + j) % n;
                let p = &self.players[s];
                out[i] = 1.0;
                out[i + 1] = p.workers as f32 / 4.0;
                out[i + 2] = p.newborn as f32;
                out[i + 3] = p.placed as f32 / 4.0;
                out[i + 4] = p.avail() as f32 / 4.0;
                out[i + 5] = p.food as f32 / 8.0;
                out[i + 6] = p.wood as f32 / 8.0;
                out[i + 7] = p.stone as f32 / 4.0;
                out[i + 8] = p.grain as f32 / 8.0;
                out[i + 9] = p.plowed as f32 / 4.0;
                out[i + 10] = p.empty_fields() as f32 / 4.0;
                out[i + 11] = p.field_grain() as f32 / 12.0;
                out[i + 12] = p.rooms as f32 / 4.0;
                out[i + 13] = p.beg as f32 / 4.0;
                out[i + 14] = sc[s] as f32 / 20.0;
                out[i + 15] = (s == self.cur as usize) as u8 as f32;
            }
            i += 16;
        }
        out[i + self.sub.index()] = 1.0;
        let (k, a, b) = match self.sub {
            Sub::Sow(k) | Sub::Build(k) | Sub::Feast(k) => (k, 0, 0),
            Sub::Trade(a, b) => (0, a, b),
            _ => (0, 0, 0),
        };
        out[i + 6] = k as f32 / 4.0;
        out[i + 7] = a as f32 / 4.0;
        out[i + 8] = b as f32 / 4.0;
        i += 9;
        debug_assert_eq!(i, NUM_FEATURES);
    }

    fn rounds(&self) -> u32 {
        self.round as u32
    }

    fn export_state(&self) -> String {
        self.to_json()
    }

    /// Payload: a JSON list of action ids, applied in order (each must be legal then).
    fn apply_external(&mut self, payload: &str, chance: &mut Vec<u64>) -> Result<(), String> {
        let v: Vec<u16> = serde_json::from_str(payload).map_err(|e| format!("toy external payload {payload:?}: {e}"))?;
        if v.is_empty() {
            return Err("toy external payload: no actions".into());
        }
        for a in v {
            self.apply(a, chance)?;
        }
        Ok(())
    }

    fn bot_names() -> &'static [&'static str] {
        &["heuristic"]
    }

    fn make_bot(_spec: &ToySpec, name: &str, _seed: u64) -> Option<Box<dyn Bot<Self>>> {
        match name {
            "heuristic" => Some(Box::new(HeuristicBot)),
            _ => None,
        }
    }

    /// One auxiliary target (docs/search.md 1.2): the next round's card, hidden until its round
    /// (`order[round]`, a one-hot over the 6 cards; all zero in the last round: no target). The
    /// toy's analogue of MTG Zero's belief target - hidden information the encoding never shows,
    /// whose best prediction from public information is uniform over the unrevealed cards of the
    /// next round's stage (the order is random within each stage).
    const AUX: &'static [(&'static str, usize)] = &[("next_card", N_CARDS)];

    fn aux_target(&self, _viewer: usize, out: &mut [f32]) {
        let r = self.round as usize;
        if !self.over && r < ROUNDS as usize {
            out[self.order[r] as usize] = 1.0;
        }
    }

    /// `determinize` dealt by a belief of `next_card` (docs/search.md 3.8): the next round's card
    /// (`order[round]`) drawn among the unrevealed cards of its stage with the weights
    /// `belief_weight(p_c, 1 / cards, tilt, clip)` (p = the belief's distribution over the 6 cards;
    /// the uniform deal gives each unrevealed card of the stage 1 / cards), then everything else as
    /// `determinize` deals it. In the last round (no next card): `determinize`.
    fn determinize_with(&mut self, viewer: usize, rng: &mut Rng, belief: &crate::game::Belief) {
        let r = self.round as usize;
        if self.over || r >= ROUNDS as usize || belief.pred.len() < N_CARDS {
            return self.determinize(viewer, rng);
        }
        let hi = if r < 3 { 3 } else { 6 };
        let n = hi - r;
        let u = 1.0 / n as f64;
        let w: Vec<f64> = (r..hi).map(|i| crate::game::belief_weight(belief.pred[self.order[i] as usize] as f64, u, belief.tilt, belief.clip)).collect();
        let mut x = rng.unit() * w.iter().sum::<f64>();
        let mut k = n - 1;
        for (j, &wj) in w.iter().enumerate() {
            x -= wj;
            if x < 0.0 {
                k = j;
                break;
            }
        }
        self.order.swap(r, r + k);
        if r + 1 < hi {
            rng.shuffle(&mut self.order[r + 1..hi]);
        }
        if hi == 3 {
            rng.shuffle(&mut self.order[3..6]);
        }
        let lo = r.max(1);
        if lo < 5 {
            rng.shuffle(&mut self.markers[lo..5]);
        }
    }
}

/// `ToyGame::heuristic` as a driver bot.
pub struct HeuristicBot;

impl Bot<ToyGame> for HeuristicBot {
    fn choose(&mut self, g: &ToyGame, legal: &[u16]) -> u16 {
        g.heuristic(legal)
    }
}

#[cfg(test)]
mod tests;
