//! "Toy tail" (`toy_tail`): the core's test game for variable-length rows (docs/search.md 1.4, 2.15, 5.3) - an
//! unbounded ordered list of items, a capped head segment with a tail, a tail-only segment and paged options with
//! pointer overrides, so the whole path (rows, keys, the limit, records, shards) is proven without MTG.
//!
//! Rules. Two players. An ordered list of items, each a value 1..9, marked by a player or not; the start has
//! `items0` items (spec, default 3), drawn from the game's seed. On a turn the player to move takes one move:
//! * `PASS` (0), `ADD k` (1..=3: k new items at the end, each value a chance outcome; legal while the list keeps at
//!   most `MAX_ITEMS`), `MARK` (12: the last item, unmarked, gets the player's mark) - on the first page only;
//! * `TAKE j` (4..=10) the item at position `w * 7 + j` of page `w`: removed, its value scored, +2 if marked;
//! * `NEXT` (11): the next page of 7 items, the same player (not a move; `ask_paged` semantics, n = 8: offered
//!   while a later window holds an item).
//!
//! The game is over after `max_moves` moves (spec, default 40) or when the list is empty after two moves without
//! an `ADD`; the higher score wins, a tie is split. `margin` = score difference / 10.
//!
//! Encoding (token layout `package.TOY_TAIL_TOKENS`, `NUM_FEATURES` 72): globals [0, 8) = the viewer to move,
//! score difference / 16, moves / 40, items / 8, page / 4, marks / 4, 0, 0. Segment `items` (index 0, offset 8,
//! 4 head slots x 16, `tail`): per item at position `pos` (head slot `pos < 4`, else a tail token of slot `pos`)
//! present, value / 9, marked, marked by the viewer, on the current page, pos / 8, the value one-hot over floats
//! 6-15. Segment `marks` (index 1, tail-only, width 4): per marked item a token of slot `pos`: present, pos / 8,
//! by the viewer, 0. Overrides: `TAKE` of a position >= 4 on the first page and every `TAKE` on a page >= 1 point
//! at `(items, col 0, pos)`; `NEXT` and `MARK` keep the generic logit (no static slot holds them).
//!
//! Chance: each added item's value, key `CHANCE | value` (forceable). `determinize` reseeds the future values
//! only (perfect information otherwise). `encode` panics on a state with a tail (R1): more than 4 items or a mark.
//!
//! Chained episodes (opt-in, spec `"episodes": k` > 1; docs/search.md 1.3, 2.14 - variable-length rows across a
//! contest's episodes): a contest of k games, each an episode; game e (from 0) starts from `items0 + e * items_step`
//! items (spec `items_step`, default 0: a later game can pass the row limit) drawn from the contest's RNG, the other
//! player to move first; the contest's result splits its points by game wins (a drawn game half each), a conceded
//! game lost by its conceding seat. The encoding is the game's alone (the contest is not observed). Without
//! `episodes` (or 1) the game of before, byte for byte.
//!
//! `toy_tail_wide` (`TailWideGame`, layout `package.TOY_TAIL_WIDE_TOKENS`): the same game with `WIDE_EXTRA` more
//! tail-only segments after `marks` - `x02` .. `x21` (indices 2 .. 21, width 4, no head) -, more tail segments than
//! MTG Zero's layout v7 and indices past its highest, so new tail segments are proven to need no worker or exporter
//! change (docs/az.md 46.7). Each item at position `pos` also gives a token of segment `2 + (value + pos) mod
//! WIDE_EXTRA`, slot `pos`: present, value / 9, marked, pos / 8. Rows of at most 96 tokens; all else toy_tail's.
use crate::game::{Next, SearchGame, MAX_SEATS};
use crate::rng::Rng;
use crate::tail::{pack_tok, TailBuf, TailSeg};

pub const NUM_ACTIONS: usize = 16;
pub const NUM_FEATURES: usize = 72;
pub const A_PASS: u16 = 0;
pub const A_ADD1: u16 = 1;
pub const A_TAKE0: u16 = 4;
pub const A_NEXT: u16 = 11;
pub const A_MARK: u16 = 12;
/// Items per page (`TAKE` 0..=6) and head slots of `items`.
pub const PAGE: usize = 7;
pub const HEAD: usize = 4;
/// `ADD` keeps the list at most this long (a rule, not an encoding cap: normal play stays under the row limit).
pub const MAX_ITEMS: usize = 24;
pub const SEG_ITEMS: u16 = 0;
pub const SEG_MARKS: u16 = 1;
pub const W_ITEM: usize = 16;
pub const W_MARK: usize = 4;
const ITEMS_OFF: usize = 8;
const CHANCE: u64 = 0x7461_696c_0000_0000;

pub static TAILS: [TailSeg; 2] = [TailSeg { seg: SEG_ITEMS, name: "items", width: W_ITEM, head: HEAD }, TailSeg { seg: SEG_MARKS, name: "marks", width: W_MARK, head: 0 }];
/// `toy_tail_wide`'s tail-only segments after `marks` (module docs).
pub const WIDE_EXTRA: usize = 20;
pub const W_X: usize = 4;
const fn xseg(seg: u16, name: &'static str) -> TailSeg {
    TailSeg { seg, name, width: W_X, head: 0 }
}
pub static WIDE_TAILS: [TailSeg; 2 + WIDE_EXTRA] = [
    TailSeg { seg: SEG_ITEMS, name: "items", width: W_ITEM, head: HEAD },
    TailSeg { seg: SEG_MARKS, name: "marks", width: W_MARK, head: 0 },
    xseg(2, "x02"),
    xseg(3, "x03"),
    xseg(4, "x04"),
    xseg(5, "x05"),
    xseg(6, "x06"),
    xseg(7, "x07"),
    xseg(8, "x08"),
    xseg(9, "x09"),
    xseg(10, "x10"),
    xseg(11, "x11"),
    xseg(12, "x12"),
    xseg(13, "x13"),
    xseg(14, "x14"),
    xseg(15, "x15"),
    xseg(16, "x16"),
    xseg(17, "x17"),
    xseg(18, "x18"),
    xseg(19, "x19"),
    xseg(20, "x20"),
    xseg(21, "x21"),
];

#[derive(Clone, Debug)]
pub struct TailSpec {
    pub items0: usize,
    pub max_moves: u32,
    /// Games of a chained contest (1: not chained).
    pub episodes: u32,
    /// Items more at each later game's start.
    pub items_step: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Item {
    value: u8,
    /// The marking player + 1 (0: unmarked).
    mark: u8,
}

/// The game with `X` extra tail-only segments: `TailGame` (0, `toy_tail`) or `TailWideGame` (`WIDE_EXTRA`,
/// `toy_tail_wide`); no other X.
pub type TailGame = TailGameOf<0>;
pub type TailWideGame = TailGameOf<WIDE_EXTRA>;

#[derive(Clone, Debug)]
pub struct TailGameOf<const X: usize> {
    items: Vec<Item>,
    scores: [i32; 2],
    moves: u32,
    max_moves: u32,
    cur: u8,
    page: u32,
    /// The last two moves were `ADD`s (newest first).
    adds: [bool; 2],
    over: bool,
    rng: Rng,
    forced: Vec<u64>,
    /// A chained contest (`episodes` > 1): its spec, the games played before this one and their points per seat.
    contest: Option<Box<Contest>>,
}

#[derive(Clone, Debug)]
struct Contest {
    spec: TailSpec,
    played: u32,
    points: [f32; 2],
}

impl<const X: usize> TailGameOf<X> {
    fn value(&mut self) -> u8 {
        while let Some(k) = (!self.forced.is_empty()).then(|| self.forced.remove(0)) {
            let v = k ^ CHANCE;
            if (1..=9).contains(&v) {
                return v as u8;
            }
        }
        1 + self.rng.below(9) as u8
    }
    fn end_turn(&mut self, add: bool) {
        self.moves += 1;
        self.adds = [add, self.adds[0]];
        self.page = 0;
        self.cur ^= 1;
        self.over = self.moves >= self.max_moves || (self.items.is_empty() && self.moves >= 2 && !self.adds[0] && !self.adds[1]);
    }
    fn on_page(&self, pos: usize) -> bool {
        let lo = self.page as usize * PAGE;
        (lo..lo + PAGE).contains(&pos)
    }
    /// The 16 floats of the item at `pos` from `viewer`'s view.
    fn item_floats(&self, viewer: usize, pos: usize, out: &mut [f32]) {
        let it = self.items[pos];
        out[0] = 1.0;
        out[1] = it.value as f32 / 9.0;
        out[2] = (it.mark > 0) as u8 as f32;
        out[3] = (it.mark == viewer as u8 + 1) as u8 as f32;
        out[4] = self.on_page(pos) as u8 as f32;
        out[5] = pos as f32 / 8.0;
        out[6 + it.value as usize] = 1.0;
    }
    pub fn items(&self) -> usize {
        self.items.len()
    }
}

impl<const X: usize> SearchGame for TailGameOf<X> {
    type Spec = TailSpec;
    const NAME: &'static str = if X == 0 { "toy_tail" } else { "toy_tail_wide" };
    const NUM_ACTIONS: usize = NUM_ACTIONS;
    const NUM_FEATURES: usize = NUM_FEATURES;
    const MIN_PLAYERS: usize = 2;
    const MAX_PLAYERS: usize = 2;
    const TAILS: &'static [TailSeg] = if X == 0 { &TAILS } else { &WIDE_TAILS };
    const OPTION_POINTERS: bool = true;
    const MAX_ROW_TOKENS: usize = if X == 0 { 64 } else { 96 };
    const HEAD_TOKENS: usize = 1 + HEAD;

    fn parse_spec(json: &str) -> Result<TailSpec, String> {
        let v: serde_json::Value = if json.trim().is_empty() { serde_json::json!({}) } else { serde_json::from_str(json).map_err(|e| format!("toy_tail spec: {e}"))? };
        let o = v.as_object().ok_or("toy_tail spec: a JSON object")?;
        let mut spec = TailSpec { items0: 3, max_moves: 40, episodes: 1, items_step: 0 };
        for (k, x) in o {
            match k.as_str() {
                "items0" => spec.items0 = x.as_u64().filter(|&n| n <= 100_000).ok_or("toy_tail spec: items0 is an integer 0..100000")? as usize,
                "max_moves" => spec.max_moves = x.as_u64().filter(|&n| (1..=10_000).contains(&n)).ok_or("toy_tail spec: max_moves is an integer 1..10000")? as u32,
                "episodes" => spec.episodes = x.as_u64().filter(|&n| (1..=9).contains(&n)).ok_or("toy_tail spec: episodes is an integer 1..9")? as u32,
                "items_step" => spec.items_step = x.as_u64().filter(|&n| n <= 100_000).ok_or("toy_tail spec: items_step is an integer 0..100000")? as usize,
                _ => return Err(format!("toy_tail spec: unknown key {k:?}")),
            }
        }
        Ok(spec)
    }

    fn new_game(spec: &TailSpec, players: usize, seed: u64) -> Result<Self, String> {
        if players != 2 {
            return Err(format!("toy_tail: {players} players (2 only)"));
        }
        let mut rng = Rng::new(seed);
        let items = (0..spec.items0).map(|_| Item { value: 1 + rng.below(9) as u8, mark: 0 }).collect();
        let contest = (spec.episodes > 1).then(|| Box::new(Contest { spec: spec.clone(), played: 0, points: [0.0; 2] }));
        const { assert!(X == 0 || X == WIDE_EXTRA, "toy_tail: X is 0 or WIDE_EXTRA") };
        Ok(Self { items, scores: [0; 2], moves: 0, max_moves: spec.max_moves, cur: 0, page: 0, adds: [false; 2], over: false, rng, forced: Vec::new(), contest })
    }

    fn num_players(&self) -> usize {
        2
    }

    fn to_move(&self) -> Option<usize> {
        (!self.over).then_some(self.cur as usize)
    }

    fn legal_actions(&self, out: &mut Vec<u16>) {
        out.clear();
        let n = self.items.len();
        let lo = self.page as usize * PAGE;
        if self.page == 0 {
            out.push(A_PASS);
            for k in 1..=3usize {
                if n + k <= MAX_ITEMS {
                    out.push(A_ADD1 + k as u16 - 1);
                }
            }
        }
        for j in 0..PAGE {
            if lo + j < n {
                out.push(A_TAKE0 + j as u16);
            }
        }
        if n > lo + PAGE {
            out.push(A_NEXT);
        }
        if self.page == 0 && self.items.last().is_some_and(|it| it.mark == 0) {
            out.push(A_MARK);
        }
    }

    fn step(&mut self, a: u16, chance: &mut Vec<u64>) -> Result<(), String> {
        let mut legal = Vec::new();
        self.legal_actions(&mut legal);
        if self.over || !legal.contains(&a) {
            return Err(format!("toy_tail: action {a} is not legal (legal {legal:?})"));
        }
        match a {
            A_PASS => self.end_turn(false),
            1..=3 => {
                for _ in 0..a {
                    let v = self.value();
                    chance.push(CHANCE | v as u64);
                    self.items.push(Item { value: v, mark: 0 });
                }
                self.end_turn(true);
            }
            4..=10 => {
                let pos = self.page as usize * PAGE + (a - A_TAKE0) as usize;
                let it = self.items.remove(pos);
                self.scores[self.cur as usize] += it.value as i32 + if it.mark > 0 { 2 } else { 0 };
                self.end_turn(false);
            }
            A_NEXT => self.page += 1,
            A_MARK => {
                let c = self.cur + 1;
                self.items.last_mut().unwrap().mark = c;
                self.end_turn(false);
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    fn force_chance(&mut self, forced: &[u64]) {
        self.forced = forced.to_vec();
    }

    fn result(&self) -> [f32; MAX_SEATS] {
        let mut r = [0.0; MAX_SEATS];
        match self.scores[0].cmp(&self.scores[1]) {
            std::cmp::Ordering::Greater => r[0] = 1.0,
            std::cmp::Ordering::Less => r[1] = 1.0,
            std::cmp::Ordering::Equal => r[..2].fill(0.5),
        }
        r
    }

    fn margin(&self, seat: usize) -> f32 {
        (self.scores[seat] - self.scores[1 - seat]) as f32 / 10.0
    }

    fn score(&self, seat: usize) -> f32 {
        self.scores[seat] as f32
    }

    fn rounds(&self) -> u32 {
        self.moves / 2
    }

    fn determinize(&mut self, _viewer: usize, rng: &mut Rng) {
        self.rng = Rng::new(rng.next_u64());
    }

    fn encode(&self, viewer: usize, out: &mut [f32]) {
        // (R1: a head-only caller never gets a row whose tail it would drop)
        let mut tb = TailBuf::new(Self::TAILS);
        self.encode_row(viewer, out, &mut tb);
        assert!(tb.is_empty(), "toy_tail: encode of a state with {} tail tokens (use encode_row)", tb.len());
    }

    fn encode_row(&self, viewer: usize, head: &mut [f32], tail: &mut TailBuf) {
        let marks = self.items.iter().filter(|it| it.mark > 0).count();
        head[0] = (self.cur as usize == viewer && !self.over) as u8 as f32;
        head[1] = (self.scores[viewer] - self.scores[1 - viewer]) as f32 / 16.0;
        head[2] = self.moves as f32 / 40.0;
        head[3] = self.items.len() as f32 / 8.0;
        head[4] = self.page as f32 / 4.0;
        head[5] = marks as f32 / 4.0;
        for pos in 0..self.items.len() {
            if pos < HEAD {
                let o = ITEMS_OFF + pos * W_ITEM;
                self.item_floats(viewer, pos, &mut head[o..o + W_ITEM]);
            } else {
                self.item_floats(viewer, pos, tail.token(SEG_ITEMS, pos as u32));
            }
        }
        for (pos, it) in self.items.iter().enumerate() {
            if it.mark > 0 {
                let t = tail.token(SEG_MARKS, pos as u32);
                t[0] = 1.0;
                t[1] = pos as f32 / 8.0;
                t[2] = (it.mark == viewer as u8 + 1) as u8 as f32;
            }
        }
        if X > 0 {
            // (toy_tail_wide: each item's token in one of the extra segments, module docs)
            for (pos, it) in self.items.iter().enumerate() {
                let t = tail.token(2 + ((it.value as usize + pos) % X) as u16, pos as u32);
                t[0] = 1.0;
                t[1] = it.value as f32 / 9.0;
                t[2] = (it.mark > 0) as u8 as f32;
                t[3] = pos as f32 / 8.0;
            }
        }
    }

    fn option_overrides(&self, _viewer: usize, legal: &[u16], out: &mut Vec<(u32, i64)>) {
        for (i, &a) in legal.iter().enumerate() {
            if (A_TAKE0..A_TAKE0 + PAGE as u16).contains(&a) {
                let pos = self.page as usize * PAGE + (a - A_TAKE0) as usize;
                if self.page >= 1 || pos >= HEAD {
                    out.push((i as u32, pack_tok(SEG_ITEMS, 0, pos as u32)));
                }
            }
        }
    }

    fn export_state(&self) -> String {
        let mut v = serde_json::json!({
            "items": self.items.iter().map(|it| [it.value, it.mark]).collect::<Vec<_>>(),
            "scores": self.scores, "moves": self.moves, "to_move": self.cur, "page": self.page, "over": self.over,
        });
        if let Some(c) = &self.contest {
            v["game"] = serde_json::json!(c.played);
            v["points"] = serde_json::json!(c.points);
        }
        v.to_string()
    }

    fn chained(&self) -> bool {
        self.contest.is_some()
    }

    fn next_episode(&self, conceded: Option<usize>) -> Result<Next<Self>, String> {
        let Some(c) = &self.contest else {
            return match conceded {
                None => Ok(Next::Over(self.result())),
                Some(s) => Err(format!("toy_tail: seat {s} conceded, but the game does not chain its episodes")),
            };
        };
        let r = match conceded {
            None if self.over => self.result(),
            None => return Err("toy_tail: next_episode before the game's end".into()),
            Some(p) if p < 2 && !self.over => {
                let mut r = [0.0; MAX_SEATS];
                r[1 - p] = 1.0;
                r
            }
            Some(p) => return Err(format!("toy_tail: seat {p} concedes, but no game is in progress")),
        };
        let mut c = c.clone();
        c.points[0] += r[0];
        c.points[1] += r[1];
        c.played += 1;
        if c.played >= c.spec.episodes {
            let mut out = [0.0; MAX_SEATS];
            match c.points[0].partial_cmp(&c.points[1]) {
                Some(std::cmp::Ordering::Greater) => out[0] = 1.0,
                Some(std::cmp::Ordering::Less) => out[1] = 1.0,
                _ => out[..2].fill(0.5),
            }
            return Ok(Next::Over(out));
        }
        let mut rng = Rng::new(self.rng.clone().next_u64() ^ 0x6570_6973_6f64_6500 ^ c.played as u64);
        let n = c.spec.items0 + c.played as usize * c.spec.items_step;
        let items = (0..n).map(|_| Item { value: 1 + rng.below(9) as u8, mark: 0 }).collect();
        let cur = (c.played % 2) as u8;
        Ok(Next::Continue(Self { items, scores: [0; 2], moves: 0, max_moves: c.spec.max_moves, cur, page: 0, adds: [false; 2], over: false, rng, forced: Vec::new(), contest: Some(c) }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn game(json: &str, seed: u64) -> TailGame {
        TailGame::new_game(&TailGame::parse_spec(json).unwrap(), 2, seed).unwrap()
    }

    fn row(g: &TailGame, viewer: usize) -> (Vec<f32>, TailBuf) {
        let mut head = vec![0.0; NUM_FEATURES];
        let mut tb = TailBuf::new(&TAILS);
        g.encode_row(viewer, &mut head, &mut tb);
        tb.seal();
        (head, tb)
    }

    #[test]
    fn a_short_list_is_head_only_and_encode_matches() {
        let g = game("{}", 1);
        let (head, tb) = row(&g, 0);
        assert!(tb.is_empty());
        let mut x = vec![0.0; NUM_FEATURES];
        g.encode(0, &mut x);
        assert_eq!(x, head);
        assert_eq!(head[ITEMS_OFF], 1.0);
        assert_eq!(head[ITEMS_OFF + 3 * W_ITEM], 0.0, "3 items: slot 3 absent");
    }

    #[test]
    fn items_past_the_head_and_marks_are_tail_tokens() {
        let mut g = game(r#"{"items0": 9}"#, 2);
        let mut ch = Vec::new();
        g.step(A_MARK, &mut ch).unwrap();
        let (head, tb) = row(&g, 0);
        // items 4..8 then the mark of item 8
        assert_eq!(tb.seg(), &[0, 0, 0, 0, 0, 1]);
        assert_eq!(tb.slot(), &[4, 5, 6, 7, 8, 8]);
        let v = tb.val();
        let it8 = &v[4 * W_ITEM..5 * W_ITEM];
        assert_eq!(it8[0], 1.0);
        assert_eq!(it8[2], 1.0, "marked");
        assert_eq!(it8[3], 1.0, "by the viewer");
        assert_eq!(&v[5 * W_ITEM..], &[1.0, 1.0, 1.0, 0.0]);
        let (_, tb1) = row(&g, 1);
        assert_eq!(tb1.val()[4 * W_ITEM + 3], 0.0, "not by viewer 1");
        assert_eq!(head[5], 0.25);
    }

    #[test]
    #[should_panic(expected = "use encode_row")]
    fn encode_refuses_a_tail() {
        let g = game(r#"{"items0": 5}"#, 3);
        let mut x = vec![0.0; NUM_FEATURES];
        g.encode(0, &mut x);
    }

    #[test]
    fn pages_and_overrides() {
        let mut g = game(r#"{"items0": 16}"#, 4);
        let mut legal = Vec::new();
        g.legal_actions(&mut legal);
        assert_eq!(legal, vec![A_PASS, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, A_NEXT, A_MARK]);
        let mut ov = Vec::new();
        g.option_overrides(1, &legal, &mut ov);
        assert_eq!(ov, vec![(8, pack_tok(0, 0, 4)), (9, pack_tok(0, 0, 5)), (10, pack_tok(0, 0, 6))]);
        let mut ch = Vec::new();
        g.step(A_NEXT, &mut ch).unwrap();
        g.legal_actions(&mut legal);
        assert_eq!(legal, vec![4, 5, 6, 7, 8, 9, 10, A_NEXT]);
        ov.clear();
        g.option_overrides(0, &legal, &mut ov);
        assert_eq!(ov.len(), 7);
        assert_eq!(ov[0], (0, pack_tok(0, 0, 7)));
        g.step(A_NEXT, &mut ch).unwrap();
        g.legal_actions(&mut legal);
        assert_eq!(legal, vec![4, 5], "items 14 and 15, no later window");
        g.step(5, &mut ch).unwrap();
        assert_eq!(g.items(), 15);
        assert_eq!(g.to_move(), Some(1));
        g.legal_actions(&mut legal);
        assert_eq!(legal[0], A_PASS, "back on the first page");
    }

    #[test]
    fn adds_are_forceable_chance() {
        let mut g = game("{}", 5);
        g.force_chance(&[CHANCE | 7, CHANCE | 2]);
        let mut ch = Vec::new();
        g.step(3, &mut ch).unwrap();
        assert_eq!(ch.len(), 3);
        assert_eq!(&ch[..2], &[CHANCE | 7, CHANCE | 2]);
        assert_eq!(g.items(), 6);
    }

    #[test]
    fn the_game_ends() {
        for seed in 0..20 {
            let mut g = game("{}", seed);
            let mut r = Rng::new(seed);
            let mut legal = Vec::new();
            let mut ch = Vec::new();
            let mut steps = 0;
            while g.to_move().is_some() {
                g.legal_actions(&mut legal);
                let a = legal[r.below(legal.len() as u32) as usize];
                g.step(a, &mut ch).unwrap();
                steps += 1;
                assert!(steps < 10_000);
            }
            let res = g.result();
            assert!((res[0] + res[1] - 1.0).abs() < 1e-6);
        }
    }
}
