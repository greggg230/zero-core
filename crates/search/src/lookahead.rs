//! Exhaustive lookahead per information set (docs/search.md 3.9; opt-in, a seat's `lookahead`).
//!
//! The Rust port of the Delver Lab's x-ply lookahead (delver-lab `delverlab/xply_is.py`, the per-world
//! tree `delverlab/xply.py`): every option of both players' decisions expanded to a horizon, in a set of
//! dealt worlds, the leaves valued by the network from the root decider's view, then backed up - our
//! decisions per information set, the opponent's per world by a model. It ignores the prior where it
//! expands every option (a 0.9%-prior move gets the same look as the favourite).
//!
//! * **Plies** are decisions with >= 2 options, of either seat (`tree::transition` steps the
//!   single-option ones); a **minor** decision (`SearchGame::minor_decision`, MTG's mana payments)
//!   takes the network's top prior and counts no ply.
//! * **Horizon**: `plies` moves; or until the turn passes (`Horizon::Turn`: the first decision of the
//!   next turn is the leaf) or the decider's next turn begins (`Horizon::NextTurn`,
//!   `SearchGame::active_seat`), at most `max_plies` deep (a line cut there counts in `cut`).
//! * **Branching**: every option, or the `topk` of highest prior (`topk_from_turn`: every option
//!   before that turn; `topk_root`: the root too - the lab's information-set solver caps the root, its
//!   per-world tree never does).
//! * **Our decisions** (`Mode::Infoset`, the default): every world is expanded in lockstep along the
//!   same action sequences; at each of the root decider's decisions the members whose observation is
//!   the same - an exact hash of `encode(R)` and the legal options, the lab's key - form one group that
//!   takes ONE action, the one with the best weighted average over its worlds ("max of average"), and
//!   every member gets its value under that choice. A reveal (a draw, a look) changes the encoding and
//!   splits the groups: the value of information. `Mode::World`: every member its own group (each
//!   world a perfect-information tree, the lab's per-world upper bound).
//! * **The opponent** decides per world: `Opp::Min` its reply worst for us (each reply's member
//!   weighted 1 / replies in our later averages: we learn the reply, not why), `Opp::Expect` the
//!   prior-weighted mean of its replies (the prior from its own view; member weights scaled by it),
//!   `Opp::Top` its top-prior reply only.
//! * **Leaves** (the horizon, the ply cap, a state without options): the network's value of a
//!   value-only row `encode(R)` (P(R wins)); a finished game its `result()[R]` (a deferred terminal is
//!   valued by the network).
//! * **Worlds** are dealt once per decision from the root decider's view: `determinize_with` the
//!   root's belief (the seat's tilt / clip; a `mix` share `determinize`), or `determinize` without one
//!   (`tree::deal`, the search's own deal), from explicit seeds or the driver's search stream.
//! * **Budget** (`budget` nodes per decision, 0 = none): over it the attempt is abandoned and the next
//!   stage runs on the same worlds - `fallback_topk` options below the root, then each of
//!   `fallback_plies`, then the final fallback (`Last::Mcts`: the seat's MCTS search of the decision;
//!   `Last::Prior`: the network's top prior). `LookStats` counts what each decision took.
//!
//! **Evaluation protocol** (`Look::step` / `Look::answer`): the lookahead runs level by level (one
//! level = the members between two of our decisions) and asks for network rows through a `RowSink`:
//! priors where they choose what to expand or weigh (minor decisions, the opponent's under `Top` /
//! `Expect` or a cap, our groups under a cap) block the level; leaf values never do and fill the
//! batches up to `rows` (a value round is sent once `rows` of them wait, so leaf games are dropped
//! early). The driver puts every game's rows of a round into its batches (one GPU batch for every game
//! in flight, as MCTS leaves); `PyLookahead` hands them to Python (the position API).
//!
//! **Leaf values by observation** (`dedup`, default on): a leaf's value is the network's answer to
//! `encode(R)` alone, so leaves whose encodings are equal - the same hash, `obs_hash` - share one row
//! (worlds the root decider cannot tell apart, after the same moves: most of them, early in a game). The
//! values are the same up to the network's float rounding at another batch shape; `LookStats::
//! value_dups` counts the shared ones (`value_rows` the rows sent; the lab's value rows = their sum).
//!
//! Not ported: the lab's `constraint` (restricting our lines to a class) and its reveal annotations
//! (`ev`: a line replayed in Python recovers them from the chance keys).
use std::collections::HashMap;

use serde_json::Value;

use crate::driver::{ROW_BELIEF, ROW_LEAF};
use crate::game::{Belief, SearchGame, MAX_SEATS};
use crate::rng::{split64, Rng};
use crate::tree::{deal, transition};

const NIL: u32 = u32::MAX;
/// The action of a root member (no move led to it).
pub const NO_ACT: u16 = u16::MAX;
/// The tolerance of the "max of average": a later action replaces the best only by more than this
/// (the lab's `avg > best_avg + 1e-12`).
pub const TIE: f64 = 1e-12;

/// The opponent model (`LookCfg::opp`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Opp {
    /// Its reply worst for us, per world (the lab's 'min'; "best reply").
    Min,
    /// The prior-weighted mean of its replies (the lab's 'expect'; "policy").
    Expect,
    /// Its top-prior reply only (the lab's 'top').
    Top,
}

/// Whose information our decisions use (`LookCfg::mode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// One choice per information set (the root decider's observation): the default.
    Infoset,
    /// One choice per member (a perfect-information tree per world: the lab's per-world upper bound).
    World,
}

/// Where a line ends (`LookCfg::horizon`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Horizon {
    /// After `plies` moves.
    Plies,
    /// At the first decision of the next turn (at most `max_plies` moves).
    Turn,
    /// At the first decision of the root decider's next turn (at most `max_plies` moves).
    NextTurn,
}

/// The last fallback of a decision over the budget (`LookCfg::last`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Last {
    /// The seat's MCTS search of the decision (its `sims`, rule, belief: the evaluation seat).
    Mcts,
    /// The network's top prior at the root (from the decider's view).
    Prior,
}

/// A lookahead seat's settings (seat key `lookahead`, a JSON object: `LookCfg::parse`).
#[derive(Clone, Debug, PartialEq)]
pub struct LookCfg {
    pub plies: u32,
    pub horizon: Horizon,
    pub max_plies: u32,
    pub worlds: u32,
    pub topk: u32,
    pub topk_root: bool,
    pub topk_from_turn: Option<u32>,
    pub opp: Opp,
    pub mode: Mode,
    pub budget: u64,
    pub fallback_topk: u32,
    /// None: derived (`LookCfg::stages`).
    pub fallback_plies: Option<Vec<u32>>,
    pub last: Last,
    pub rows: u32,
    pub dedup: bool,
}

impl Default for LookCfg {
    fn default() -> Self {
        LookCfg { plies: 4, horizon: Horizon::Plies, max_plies: 40, worlds: 8, topk: 0, topk_root: false, topk_from_turn: None, opp: Opp::Min, mode: Mode::Infoset, budget: 0, fallback_topk: 4, fallback_plies: None, last: Last::Mcts, rows: 512, dedup: true }
    }
}

/// The settings `LookCfg::parse` knows.
pub const LOOK_KEYS: &[&str] = &["plies", "horizon", "max_plies", "worlds", "topk", "topk_root", "topk_from_turn", "opp", "mode", "budget", "fallback_topk", "fallback_plies", "fallback", "rows", "dedup"];

fn uint(v: &Value, k: &str) -> Result<u64, String> {
    v.as_u64().or_else(|| v.as_f64().filter(|x| x.fract() == 0.0 && *x >= 0.0).map(|x| x as u64)).ok_or_else(|| format!("lookahead {k}: a whole number >= 0 (got {v})"))
}

fn u32_of(v: &Value, k: &str) -> Result<u32, String> {
    let x = uint(v, k)?;
    u32::try_from(x).map_err(|_| format!("lookahead {k}: {x} is too large"))
}

fn text<'a>(v: &'a Value, k: &str) -> Result<&'a str, String> {
    v.as_str().ok_or_else(|| format!("lookahead {k}: a string (got {v})"))
}

fn boolean(v: &Value, k: &str) -> Result<bool, String> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::Number(n) if n.as_u64() == Some(0) || n.as_u64() == Some(1) => Ok(n.as_u64() == Some(1)),
        _ => Err(format!("lookahead {k}: true / false (got {v})")),
    }
}

impl LookCfg {
    /// The settings from a JSON object (unknown keys, wrong types and bad values are errors - never
    /// ignored): plies, horizon ('plies' | 'turn' | 'next_turn'), max_plies, worlds, topk (0 = every
    /// option), topk_root, topk_from_turn, opp ('min' | 'best' | 'expect' | 'policy' | 'top'), mode
    /// ('infoset' | 'world'), budget (nodes, 0 = none), fallback_topk, fallback_plies (a list or one
    /// number), fallback ('mcts' | 'prior'), rows (rows per batch), dedup (leaf values shared by
    /// observation, default true).
    pub fn parse(json: &str) -> Result<LookCfg, String> {
        let v: Value = serde_json::from_str(json).map_err(|e| format!("lookahead: {e}"))?;
        let o = v.as_object().ok_or_else(|| format!("lookahead: a JSON object of settings, not {v}"))?;
        let mut c = LookCfg::default();
        for (k, x) in o {
            match k.as_str() {
                "plies" => c.plies = u32_of(x, k)?,
                "horizon" => {
                    c.horizon = match text(x, k)? {
                        "plies" => Horizon::Plies,
                        "turn" => Horizon::Turn,
                        "next_turn" => Horizon::NextTurn,
                        s => return Err(format!("lookahead horizon {s:?}: 'plies', 'turn' or 'next_turn'")),
                    }
                }
                "max_plies" => c.max_plies = u32_of(x, k)?,
                "worlds" => c.worlds = u32_of(x, k)?,
                "topk" => c.topk = u32_of(x, k)?,
                "topk_root" => c.topk_root = boolean(x, k)?,
                "topk_from_turn" => c.topk_from_turn = if x.is_null() { None } else { Some(u32_of(x, k)?) },
                "opp" => {
                    c.opp = match text(x, k)? {
                        "min" | "best" => Opp::Min,
                        "expect" | "policy" => Opp::Expect,
                        "top" => Opp::Top,
                        s => return Err(format!("lookahead opp {s:?}: 'min' (or 'best'), 'expect' (or 'policy') or 'top'")),
                    }
                }
                "mode" => {
                    c.mode = match text(x, k)? {
                        "infoset" => Mode::Infoset,
                        "world" => Mode::World,
                        s => return Err(format!("lookahead mode {s:?}: 'infoset' or 'world'")),
                    }
                }
                "budget" => c.budget = uint(x, k)?,
                "fallback_topk" => c.fallback_topk = u32_of(x, k)?,
                "fallback_plies" => {
                    c.fallback_plies = Some(match x {
                        Value::Array(a) => a.iter().map(|y| u32_of(y, k)).collect::<Result<Vec<_>, _>>()?,
                        Value::Null => return Err("lookahead fallback_plies: a list of plies ([] = none)".into()),
                        y => vec![u32_of(y, k)?],
                    })
                }
                "fallback" => {
                    c.last = match text(x, k)? {
                        "mcts" => Last::Mcts,
                        "prior" => Last::Prior,
                        s => return Err(format!("lookahead fallback {s:?}: 'mcts' or 'prior'")),
                    }
                }
                "rows" => c.rows = u32_of(x, k)?,
                "dedup" => c.dedup = boolean(x, k)?,
                _ => return Err(format!("lookahead: unknown setting {k:?} (one of {LOOK_KEYS:?})")),
            }
        }
        c.check()?;
        Ok(c)
    }

    /// The settings' consistency (`parse` calls it; so does `Driver::new`).
    pub fn check(&self) -> Result<(), String> {
        if self.horizon == Horizon::Plies && self.plies == 0 {
            return Err("lookahead plies 0: at least 1".into());
        }
        if self.max_plies == 0 {
            return Err("lookahead max_plies 0: at least 1".into());
        }
        if !(1..=4096).contains(&self.worlds) {
            return Err(format!("lookahead worlds {}: 1 to 4096", self.worlds));
        }
        if !(1..=1 << 20).contains(&self.rows) {
            return Err(format!("lookahead rows {}: 1 to {}", self.rows, 1 << 20));
        }
        if self.topk == 0 && (self.topk_root || self.topk_from_turn.is_some()) {
            return Err("lookahead topk_root / topk_from_turn: settings of a top-k lookahead (topk > 0)".into());
        }
        if let Some(fp) = &self.fallback_plies {
            if fp.contains(&0) {
                return Err("lookahead fallback_plies: each at least 1".into());
            }
            if self.budget == 0 && !fp.is_empty() {
                return Err("lookahead fallback_plies: a budget's fallback (budget > 0)".into());
            }
        }
        Ok(())
    }

    /// The moves of a line at most (the ply cap): `plies`, or `max_plies` with a horizon.
    pub fn limit(&self) -> u32 {
        if self.horizon == Horizon::Plies {
            self.plies
        } else {
            self.max_plies
        }
    }

    /// The stages a decision tries in turn while over the budget: this, then (budget > 0) the top-k
    /// stage (`fallback_topk` below the root, `topk_from_turn` off, when it caps more than this), then
    /// each of `fallback_plies` (fixed plies, smaller than the stage before for a fixed-plies one;
    /// None: one stage at max(2, plies - 2) below `plies`, or 4 for a horizon), each with the top-k of
    /// the stage before. The final fallback (`last`) comes after the last stage.
    pub fn stages(&self) -> Vec<LookCfg> {
        let mut v = vec![self.clone()];
        if self.budget == 0 {
            return v;
        }
        let mut cur = self.clone();
        let k = self.fallback_topk;
        if k > 0 && (cur.topk == 0 || cur.topk > k) {
            cur.topk = k;
            cur.topk_from_turn = None;
            v.push(cur.clone());
        }
        let fp = match &self.fallback_plies {
            Some(f) => f.clone(),
            None if self.horizon == Horizon::Plies => {
                let p = self.plies.saturating_sub(2).max(2);
                if p < self.plies {
                    vec![p]
                } else {
                    Vec::new()
                }
            }
            None => vec![4],
        };
        for p in fp {
            if cur.horizon == Horizon::Plies && p >= cur.plies {
                continue;
            }
            cur.horizon = Horizon::Plies;
            cur.plies = p;
            v.push(cur.clone());
        }
        v
    }

    /// The settings as a JSON object (reports; `parse` reads it back).
    pub fn to_json(&self) -> String {
        let h = match self.horizon {
            Horizon::Plies => "plies",
            Horizon::Turn => "turn",
            Horizon::NextTurn => "next_turn",
        };
        let o = match self.opp {
            Opp::Min => "min",
            Opp::Expect => "expect",
            Opp::Top => "top",
        };
        let m = match self.mode {
            Mode::Infoset => "infoset",
            Mode::World => "world",
        };
        let l = match self.last {
            Last::Mcts => "mcts",
            Last::Prior => "prior",
        };
        let mut j = serde_json::json!({"plies": self.plies, "horizon": h, "max_plies": self.max_plies, "worlds": self.worlds, "topk": self.topk, "topk_root": self.topk_root, "topk_from_turn": self.topk_from_turn, "opp": o, "mode": m, "budget": self.budget, "fallback_topk": self.fallback_topk, "fallback": l, "rows": self.rows, "dedup": self.dedup});
        if let Some(fp) = &self.fallback_plies {
            j["fallback_plies"] = serde_json::json!(fp);
        }
        j.to_string()
    }
}

/// Where a leaf's line ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum End {
    /// The game is over (its result; a deferred terminal: the network's value).
    Terminal,
    /// The horizon's moment (`Horizon::Turn` / `NextTurn`).
    Horizon,
    /// The ply cap under a horizon (`max_plies`, counted in `LookStats::cut`).
    Cut,
    /// `plies` moves (no horizon).
    Plies,
    /// A pending decision without options (never in a well-formed game).
    NoOptions,
}

impl End {
    pub fn name(self) -> &'static str {
        match self {
            End::Terminal => "game over",
            End::Horizon => "horizon",
            End::Cut => "cut",
            End::Plies => "plies",
            End::NoOptions => "no options",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Not classified yet (or between a minor step and its next classification).
    New,
    Leaf(End),
    /// At the root decider's decision (its group in `Mem::group`).
    Ours,
    /// At the opponent's decision (its replies in `kid_ids[kids_off..]`).
    Opp,
}

/// One member: a world's state at a node of the lookahead.
struct Mem<G> {
    g: Option<Box<G>>,
    wt: f64,
    /// NaN until known (a leaf's evaluation, an opponent decision's aggregate, our group's choice).
    value: f64,
    kids_off: u32,
    nkids: u32,
    group: u32,
    /// Our decision: the chosen child (after the backup).
    choice: u32,
    world: u16,
    plies: u16,
    /// The move into this member and who made it (NO_ACT: a root).
    act: u16,
    mover: u8,
    kind: Kind,
}

struct Group {
    members: Vec<u32>,
    acts: Vec<u16>,
    /// Act-major: kids[j * members.len() + i] = member i's child after acts[j].
    kids: Vec<u32>,
    best: u32,
}

#[derive(Clone, Copy, Debug)]
enum Req {
    /// The root's belief row (`ROW_BELIEF`: its auxiliary output).
    Belief,
    /// The root's prior from the decider's view (a minor root decision, or the final fallback `Prior`).
    RootPrior,
    /// A member's prior from its mover's view (a minor decision, an opponent decision).
    Prior(u32),
    /// Our group's prior (its first member, our view).
    Group(u32),
    /// A leaf's value (`encode(R)`, value-only).
    Value(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Belief,
    RootPrior,
    Deal,
    Classify,
    AdvWait,
    Solve,
    SolveWait,
    Flush,
    Done,
}

/// What `Look::step` asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// This many rows were pushed into the sink: answer them (`Look::answer`), then step again.
    Rows(usize),
    /// The decision is made: `Look::outcome`.
    Done,
}

/// The decision of a lookahead (`Look::outcome`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Play this action.
    Act(u16),
    /// Over the budget at every stage: search the decision with the seat's MCTS (`Last::Mcts`).
    Mcts,
}

/// Where the network rows of a lookahead go (`Look::step`): one row = `g.encode(viewer)`, the
/// options `opts` (empty: a value-only row), its kind (`ROW_LEAF`, `ROW_BELIEF`).
pub trait RowSink<G: SearchGame> {
    /// A new row (`viewer`'s, its options, its kind) -> its `NUM_FEATURES` features, zeroed, for the
    /// caller to encode.
    fn push_row(&mut self, viewer: usize, opts: &[u16], kind: u8) -> &mut [f32];
    /// Drop the last row pushed (a leaf whose observation another row has: `dedup`).
    fn pop_row(&mut self);
    /// A row of `g.encode(viewer)`.
    fn row(&mut self, g: &G, viewer: usize, opts: &[u16], kind: u8) {
        let x = self.push_row(viewer, opts, kind);
        g.encode(viewer, x);
    }
}

/// The network's answers to a round's rows (`Look::answer`): rows `base..base + n` of a batch with
/// option segments `opt_off`, priors over them, values [rows x MAX_SEATS] relative to each row's
/// player, the root belief rows' auxiliary outputs (`aux`, `aux_row[b]` = row b's index there or
/// u32::MAX; both empty without belief rows).
pub struct Answers<'a> {
    pub base: usize,
    pub opt_off: &'a [i32],
    pub priors: &'a [f32],
    pub value: &'a [f32],
    pub aux: &'a [f32],
    pub aux_row: &'a [u32],
    pub aux_width: usize,
}

/// A decision's counters (`Look::stats`; the driver sums them, `DriverStats::look_*`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LookStats {
    /// Nodes expanded (children created), as the lab counts them: of the stage that decided.
    pub nodes: u64,
    pub leaves: u64,
    pub prior_rows: u64,
    pub value_rows: u64,
    /// Leaves valued by another leaf's row (the same observation: `dedup`).
    pub value_dups: u64,
    pub minor_steps: u64,
    pub cut: u64,
    pub groups: u64,
    /// Rounds of rows (network batches this decision waited for) and their rows, every stage's.
    pub rounds: u64,
    pub rows: u64,
    /// The stage that decided (0 = as configured; `LookCfg::stages`), and whether the final fallback
    /// did (`Outcome::Mcts`, or the top prior of `Last::Prior`).
    pub stage: u32,
    pub last: bool,
    /// Nodes of abandoned stages (over the budget).
    pub aborted_nodes: u64,
    /// A minor root decision: the network's top prior (no lookahead).
    pub minor_root: bool,
    /// The most live member games at once.
    pub peak_games: u64,
}

/// One root option's outcome over the worlds (`Look::options`).
#[derive(Clone, Debug)]
pub struct OptionValue {
    pub act: u16,
    /// (world, value) for each world whose root group expanded it.
    pub values: Vec<(u16, f64)>,
    pub mean: f64,
    pub sd: f64,
}

/// A line under the solved policy (`Look::line`): (player, action, minor) moves, where it ended.
pub type Line = (Vec<(u8, u16, bool)>, End);

/// The lookahead of one decision (module docs).
pub struct Look<G: SearchGame> {
    stages: Vec<LookCfg>,
    stage: usize,
    cfg: LookCfg,
    viewer: usize,
    turn0: u32,
    live: Box<G>,
    root_legal: Vec<u16>,
    /// The root's belief: (tilt, clip, mix) to ask for (None: uniform worlds) and the answer.
    belief_ask: Option<(f64, f64, f64)>,
    belief: Option<Belief>,
    /// Explicit worlds (the position API) or deal seeds; else `deal_rng` deals them.
    seeds: Vec<u64>,
    worlds: Vec<G>,
    mems: Vec<Mem<G>>,
    kid_ids: Vec<u32>,
    kid_probs: Vec<f64>,
    groups: Vec<Group>,
    level: u32,
    roots: Vec<u32>,
    pending: Vec<u32>,
    frontier: Vec<u32>,
    adv_minor: Vec<u32>,
    adv_opp: Vec<u32>,
    level_groups: (usize, usize),
    prior_q: std::collections::VecDeque<Req>,
    value_q: std::collections::VecDeque<u32>,
    inflight: Vec<Req>,
    mem_pri: HashMap<u32, Vec<f32>>,
    group_pri: HashMap<u32, Vec<f32>>,
    root_pri: Option<Vec<f32>>,
    phase: Phase,
    outcome: Option<Outcome>,
    st: LookStats,
    live_games: u64,
    keep_lines: bool,
    minors: HashMap<u32, Vec<(u8, u16)>>,
    xbuf: Vec<f32>,
    key: Vec<u64>,
    legal: Vec<u16>,
    /// `dedup`: per observation hash the value, or the members waiting for the row sent this round;
    /// per sent value row its hash.
    vcache: HashMap<u128, VEntry>,
    vhash: HashMap<u32, u128>,
}

enum VEntry {
    Wait(Vec<u32>),
    Val(f64),
}

/// A 128-bit hash of an encoding's float bits (the observation key; the lab's blake2b of the bytes):
/// the (position, bits) of its non-zero pairs of floats (-0.0 counts: its bits are not 0) through two
/// independent multiply-xorshift lanes, finished by `split64`. The encodings are sparse (a Delver
/// position: ~150 non-zero floats of 37,620): blocks of zero bits are skipped (an OR over the block).
pub fn obs_hash(x: &[f32]) -> u128 {
    #[inline(always)]
    fn mix(a: &mut u64, b: &mut u64, i: u64, v: u64) {
        let k = v ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        *a = (*a ^ k).wrapping_mul(0xD6E8_FEB8_6659_FD93);
        *a ^= *a >> 32;
        *b = (b.wrapping_add(k ^ (i << 1))).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(23);
    }
    // SAFETY: f32 and u32 have the same size and alignment; the bits are read, never written
    let u: &[u32] = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u32, x.len()) };
    let mut a: u64 = 0x243F_6A88_85A3_08D3;
    let mut b: u64 = 0x1319_8A2E_0370_7344;
    const BLOCK: usize = 64;
    let mut blocks = u.chunks_exact(BLOCK);
    let mut base = 0u64;
    for blk in &mut blocks {
        let mut any = 0u32;
        for &w in blk {
            any |= w;
        }
        if any != 0 {
            for (j, w) in blk.chunks_exact(2).enumerate() {
                let v = w[0] as u64 | ((w[1] as u64) << 32);
                if v != 0 {
                    mix(&mut a, &mut b, base + j as u64, v);
                }
            }
        }
        base += (BLOCK / 2) as u64;
    }
    for (j, w) in blocks.remainder().chunks(2).enumerate() {
        let v = w[0] as u64 | (w.get(1).map_or(0, |&f| f as u64) << 32);
        if v != 0 {
            mix(&mut a, &mut b, base + j as u64, v);
        }
    }
    let n = x.len() as u64;
    ((split64(a ^ n) as u128) << 64) | split64(b ^ n.rotate_left(17)) as u128
}

/// The first index of the largest prior (Python's `max(range(n), key=p.__getitem__)`).
fn argmax_first(p: &[f32]) -> usize {
    let mut b = 0;
    for i in 1..p.len() {
        if p[i] > p[b] {
            b = i;
        }
    }
    b
}

/// Indexes 0..n sorted by descending prior, ties in index order (Python's stable `sort(key=-p)`).
fn by_prior(p: &[f32]) -> Vec<usize> {
    let mut o: Vec<usize> = (0..p.len()).collect();
    o.sort_by(|&i, &j| p[j].partial_cmp(&p[i]).unwrap_or(std::cmp::Ordering::Equal));
    o
}

enum Abort {
    Budget,
    Err(String),
}

impl From<String> for Abort {
    fn from(e: String) -> Abort {
        Abort::Err(e)
    }
}

impl From<&str> for Abort {
    fn from(e: &str) -> Abort {
        Abort::Err(e.to_string())
    }
}

impl<G: SearchGame> Look<G> {
    /// The lookahead of `live`'s pending decision (`viewer` decides it): `cfg`, the root's belief
    /// settings `belief` ((tilt, clip, mix), tilt > 0: one `ROW_BELIEF` row first; None: uniform
    /// worlds), and the worlds - `worlds` given (each at the viewer's decision with the same options),
    /// else dealt from `seeds` (one per world, `cfg.worlds` of them), else from `rng` by `deal_from`.
    /// `keep_lines`: keep what `line` needs (the minor moves).
    pub fn new(cfg: LookCfg, live: &G, viewer: usize, belief: Option<(f64, f64, f64)>, worlds: Vec<G>, seeds: Vec<u64>, keep_lines: bool) -> Result<Look<G>, String> {
        cfg.check()?;
        if live.to_move() != Some(viewer) {
            return Err(format!("lookahead: seat {viewer} is not to move ({:?})", live.to_move()));
        }
        let mut root_legal = Vec::new();
        live.legal_actions(&mut root_legal);
        if root_legal.is_empty() {
            return Err("lookahead: a decision without options".into());
        }
        if cfg.horizon == Horizon::NextTurn && live.active_seat().is_none() {
            return Err(format!("lookahead horizon 'next_turn': {} has no turns (SearchGame::active_seat)", G::NAME));
        }
        if !worlds.is_empty() && !seeds.is_empty() {
            return Err("lookahead: worlds or deal seeds, not both".into());
        }
        if !seeds.is_empty() && seeds.len() != cfg.worlds as usize {
            return Err(format!("lookahead: {} deal seeds for {} worlds", seeds.len(), cfg.worlds));
        }
        if let Some((t, c, m)) = belief {
            crate::driver::NnSeat::check_belief(t, c, m)?;
            if t <= 0.0 {
                return Err("lookahead: a belief's tilt > 0 (None: uniform worlds)".into());
            }
            if G::aux_width() == 0 {
                return Err(format!("lookahead: a belief for {}, which has no auxiliary targets (SearchGame::AUX)", G::NAME));
            }
            if !worlds.is_empty() {
                return Err("lookahead: given worlds are dealt already (no belief)".into());
            }
        }
        let stages = cfg.stages();
        let minor = live.minor_decision();
        let phase = if minor {
            Phase::RootPrior
        } else if belief.is_some() {
            Phase::Belief
        } else {
            Phase::Deal
        };
        Ok(Look {
            stages,
            stage: 0,
            cfg,
            viewer,
            turn0: live.rounds(),
            live: Box::new(live.clone()),
            root_legal,
            belief_ask: belief,
            belief: None,
            seeds,
            worlds,
            mems: Vec::new(),
            kid_ids: Vec::new(),
            kid_probs: Vec::new(),
            groups: Vec::new(),
            level: 0,
            roots: Vec::new(),
            pending: Vec::new(),
            frontier: Vec::new(),
            adv_minor: Vec::new(),
            adv_opp: Vec::new(),
            level_groups: (0, 0),
            prior_q: Default::default(),
            value_q: Default::default(),
            inflight: Vec::new(),
            mem_pri: HashMap::new(),
            group_pri: HashMap::new(),
            root_pri: None,
            phase,
            outcome: None,
            st: LookStats { minor_root: minor, ..Default::default() },
            live_games: 0,
            keep_lines,
            minors: HashMap::new(),
            xbuf: vec![0.0; G::NUM_FEATURES],
            key: Vec::new(),
            legal: Vec::new(),
            vcache: HashMap::new(),
            vhash: HashMap::new(),
        })
    }

    /// The settings of the stage running (or that decided).
    pub fn cfg(&self) -> &LookCfg {
        &self.cfg
    }

    pub fn stats(&self) -> &LookStats {
        &self.st
    }

    /// The decision (after `Step::Done`).
    pub fn outcome(&self) -> Option<Outcome> {
        self.outcome
    }

    /// The dealt worlds (after the deal).
    pub fn worlds(&self) -> &[G] {
        &self.worlds
    }

    /// The root belief's predictions the worlds were dealt by (None: uniform).
    pub fn belief(&self) -> Option<&Belief> {
        self.belief.as_ref()
    }

    /// The root's belief given (its predictions computed outside: no belief row is asked for); the
    /// lookahead must have been created with belief settings and not have stepped yet.
    pub fn set_belief(&mut self, b: Belief) -> Result<(), String> {
        if self.belief_ask.is_none() || self.phase != Phase::Belief {
            return Err("lookahead: a belief for a lookahead without belief settings, or after it started".into());
        }
        if b.pred.len() != G::aux_width() {
            return Err(format!("lookahead: a belief of {} values ({} has {})", b.pred.len(), G::NAME, G::aux_width()));
        }
        self.belief = Some(b);
        Ok(())
    }

    /// Deal the worlds (unless given): one `tree::deal` per world (the belief's `determinize_with`, a
    /// mix share `determinize`), from its seed or else from `rng`.
    fn deal_from(&mut self, rng: &mut Rng) -> Result<(), String> {
        if !self.worlds.is_empty() || self.phase == Phase::RootPrior {
            return Ok(());
        }
        if self.belief_ask.is_some() && self.belief.is_none() {
            return Err("internal: the worlds dealt before the root's belief".into());
        }
        let mix = self.belief_ask.map_or(0.0, |b| b.2);
        for i in 0..self.cfg.worlds as usize {
            let mut g = (*self.live).clone();
            if self.seeds.is_empty() {
                deal(&mut g, self.viewer, self.belief.as_ref(), mix, rng);
            } else {
                deal(&mut g, self.viewer, self.belief.as_ref(), mix, &mut Rng::new(self.seeds[i]));
            }
            self.worlds.push(g);
        }
        Ok(())
    }

    fn check_worlds(&mut self) -> Result<(), String> {
        for (i, w) in self.worlds.iter().enumerate() {
            self.legal.clear();
            w.legal_actions(&mut self.legal);
            if w.to_move() != Some(self.viewer) || self.legal != self.root_legal {
                return Err(format!("lookahead world {i}: not the root decision ({:?} {:?} -> {:?} {:?})", Some(self.viewer), self.root_legal, w.to_move(), self.legal));
            }
        }
        Ok(())
    }

    /// (Re)start the stage `self.stage` on the worlds.
    fn start_stage(&mut self) {
        self.cfg = self.stages[self.stage].clone();
        self.mems.clear();
        self.kid_ids.clear();
        self.kid_probs.clear();
        self.groups.clear();
        self.level = 0;
        self.roots.clear();
        self.pending.clear();
        self.frontier.clear();
        self.adv_minor.clear();
        self.adv_opp.clear();
        self.prior_q.clear();
        self.value_q.clear();
        self.mem_pri.clear();
        self.group_pri.clear();
        self.minors.clear();
        self.vcache.clear();
        self.vhash.clear();
        self.live_games = 0;
        let k = std::mem::take(&mut self.st);
        self.st = LookStats { stage: self.stage as u32, aborted_nodes: k.aborted_nodes, rounds: k.rounds, rows: k.rows, minor_root: k.minor_root, peak_games: k.peak_games, ..Default::default() };
        for (i, w) in self.worlds.iter().enumerate() {
            let id = self.mems.len() as u32;
            self.mems.push(Mem { g: Some(Box::new(w.clone())), wt: 1.0, value: f64::NAN, kids_off: 0, nkids: 0, group: NIL, choice: NIL, world: i as u16, plies: 0, act: NO_ACT, mover: self.viewer as u8, kind: Kind::New });
            self.roots.push(id);
        }
        self.live_games = self.worlds.len() as u64;
        self.pending = self.roots.clone();
        self.phase = Phase::Classify;
    }

    /// Over the budget: the next stage (its nodes counted as aborted), or the final fallback.
    fn next_stage(&mut self) {
        self.st.aborted_nodes += self.st.nodes;
        self.inflight.clear();
        if self.stage + 1 < self.stages.len() {
            self.stage += 1;
            self.start_stage();
            return;
        }
        // (every stage over the budget)
        self.mems.clear();
        self.groups.clear();
        self.prior_q.clear();
        self.value_q.clear();
        self.st.last = true;
        self.st.stage = self.stages.len() as u32;
        match self.cfg.last {
            Last::Mcts => {
                self.outcome = Some(Outcome::Mcts);
                self.phase = Phase::Done;
            }
            Last::Prior => self.phase = Phase::RootPrior,
        }
    }

    fn add(&mut self, n: u64) -> Result<(), Abort> {
        self.st.nodes += n;
        if self.cfg.budget > 0 && self.st.nodes > self.cfg.budget {
            return Err(Abort::Budget);
        }
        Ok(())
    }

    fn reached(&self, g: &G) -> bool {
        match self.cfg.horizon {
            Horizon::Plies => false,
            Horizon::Turn => g.rounds() > self.turn0,
            Horizon::NextTurn => g.rounds() > self.turn0 && g.active_seat() == Some(self.viewer),
        }
    }

    fn capped(&self, g: &G) -> bool {
        self.cfg.topk > 0 && self.cfg.topk_from_turn.is_none_or(|t| g.rounds() >= t)
    }

    fn game(&self, m: u32) -> Result<&G, String> {
        self.mems[m as usize].g.as_deref().ok_or_else(|| format!("internal: member {m} has no game"))
    }

    fn leaf(&mut self, m: u32, end: End) {
        self.mems[m as usize].kind = Kind::Leaf(end);
        self.st.leaves += 1;
    }

    /// Classify the pending members (the lab's `advance` loop body): finished games, the horizon,
    /// minor decisions, the ply cap, our decisions, the opponent's.
    fn classify(&mut self) -> Result<(), String> {
        let pending = std::mem::take(&mut self.pending);
        let r = self.viewer;
        for m in pending {
            let (over, deferred, reached, minor, plies) = {
                let g = self.game(m)?;
                (g.to_move().is_none(), g.deferred(), self.reached(g), g.minor_decision(), self.mems[m as usize].plies as u32)
            };
            if over {
                self.leaf(m, End::Terminal);
                if deferred {
                    self.value_q.push_back(m);
                } else {
                    let v = self.game(m)?.result()[r] as f64;
                    let mm = &mut self.mems[m as usize];
                    mm.value = v;
                    mm.g = None;
                    self.live_games -= 1;
                }
            } else if reached {
                self.leaf(m, End::Horizon);
                self.value_q.push_back(m);
            } else if minor {
                self.adv_minor.push(m);
                self.prior_q.push_back(Req::Prior(m));
            } else if plies >= self.cfg.limit() {
                if self.cfg.horizon != Horizon::Plies {
                    self.st.cut += 1;
                    self.leaf(m, End::Cut);
                } else {
                    self.leaf(m, End::Plies);
                }
                self.value_q.push_back(m);
            } else {
                self.legal.clear();
                let g = self.mems[m as usize].g.as_deref().ok_or("internal: a pending member without a game")?;
                g.legal_actions(&mut self.legal);
                let to = g.to_move();
                let capped = self.capped(g);
                if self.legal.is_empty() {
                    self.leaf(m, End::NoOptions);
                    self.value_q.push_back(m);
                } else if to == Some(r) {
                    self.mems[m as usize].kind = Kind::Ours;
                    self.frontier.push(m);
                } else {
                    self.adv_opp.push(m);
                    if matches!(self.cfg.opp, Opp::Top | Opp::Expect) || (capped && self.legal.len() > self.cfg.topk as usize) {
                        self.prior_q.push_back(Req::Prior(m));
                    }
                }
            }
        }
        Ok(())
    }

    /// The classified minor and opponent decisions, their priors in: minor decisions take the top
    /// prior (stepped in place), the opponent's are expanded by the model (the lab's order: the minor
    /// members, then the opponent's children) -> the next pending members.
    fn advance(&mut self) -> Result<(), Abort> {
        let mut nxt = Vec::new();
        for m in std::mem::take(&mut self.adv_minor) {
            let p = self.mem_pri.remove(&m).ok_or_else(|| format!("internal: no prior for minor member {m}"))?;
            let mut g = self.mems[m as usize].g.take().ok_or("internal: a minor member without a game")?;
            self.legal.clear();
            g.legal_actions(&mut self.legal);
            if p.len() != self.legal.len() {
                return Err(Abort::Err(format!("internal: {} priors for {} options", p.len(), self.legal.len())));
            }
            let a = self.legal[argmax_first(&p)];
            let mover = g.to_move().ok_or("internal: a minor member at a finished game")? as u8;
            transition(&mut *g, a, None, &mut self.key, &mut self.legal).map_err(|e| format!("lookahead: a minor move {a} failed: {e}"))?;
            self.mems[m as usize].g = Some(g);
            self.mems[m as usize].kind = Kind::New;
            self.st.minor_steps += 1;
            if self.keep_lines {
                self.minors.entry(m).or_default().push((mover, a));
            }
            nxt.push(m);
        }
        let opp = std::mem::take(&mut self.adv_opp);
        for m in opp {
            let p = self.mem_pri.remove(&m);
            let g = self.mems[m as usize].g.take().ok_or("internal: an opponent member without a game")?;
            let mut legal = Vec::new();
            g.legal_actions(&mut legal);
            let mut order: Vec<usize> = match &p {
                Some(p) => {
                    if p.len() != legal.len() {
                        return Err(Abort::Err(format!("internal: {} priors for {} options", p.len(), legal.len())));
                    }
                    by_prior(p)
                }
                None => (0..legal.len()).collect(),
            };
            if self.cfg.opp == Opp::Top {
                order.truncate(1);
            } else if self.capped(&g) {
                order.truncate(self.cfg.topk as usize);
            }
            self.add(order.len() as u64)?;
            let (wt, plies, world) = {
                let mm = &self.mems[m as usize];
                (mm.wt, mm.plies, mm.world)
            };
            let mover = g.to_move().ok_or("internal: an opponent member at a finished game")? as u8;
            let probs: Vec<f64> = match &p {
                Some(p) => order.iter().map(|&i| p[i] as f64).collect(),
                None => vec![1.0; order.len()],
            };
            let tot: f64 = probs.iter().sum();
            let tot = if tot != 0.0 { tot } else { 1.0 };
            let off = self.kid_ids.len() as u32;
            for (j, &i) in order.iter().enumerate() {
                let mut c = Box::new((*g).clone());
                transition(&mut *c, legal[i], None, &mut self.key, &mut self.legal).map_err(|e| format!("lookahead: the opponent's move {} failed: {e}", legal[i]))?;
                let cw = match self.cfg.opp {
                    Opp::Min => wt / order.len() as f64,
                    Opp::Expect => wt * probs[j] / tot,
                    Opp::Top => wt,
                };
                let id = self.mems.len() as u32;
                self.mems.push(Mem { g: Some(c), wt: cw, value: f64::NAN, kids_off: 0, nkids: 0, group: NIL, choice: NIL, world, plies: plies + 1, act: legal[i], mover, kind: Kind::New });
                self.kid_ids.push(id);
                self.kid_probs.push(probs[j]);
                nxt.push(id);
            }
            self.live_games += order.len() as u64;
            self.live_games -= 1;
            let mm = &mut self.mems[m as usize];
            mm.kind = Kind::Opp;
            mm.kids_off = off;
            mm.nkids = order.len() as u32;
        }
        self.st.peak_games = self.st.peak_games.max(self.live_games);
        self.pending = nxt;
        Ok(())
    }

    /// Our decisions of this level's frontier into groups (the lab's `solve` before its expansion):
    /// by observation (infoset) or one per member (world); the capped groups' priors requested.
    fn group(&mut self) -> Result<bool, String> {
        let ours: Vec<u32> = std::mem::take(&mut self.frontier);
        if ours.is_empty() {
            return Ok(false);
        }
        let first = self.groups.len();
        let mut by_key: HashMap<(u128, Vec<u16>), u32> = HashMap::new();
        for m in ours {
            let g = self.mems[m as usize].g.as_deref().ok_or("internal: our member without a game")?;
            let mut legal = Vec::new();
            g.legal_actions(&mut legal);
            let gi = match self.cfg.mode {
                Mode::World => None,
                Mode::Infoset => {
                    self.xbuf.iter_mut().for_each(|x| *x = 0.0);
                    g.encode(self.viewer, &mut self.xbuf);
                    let k = (obs_hash(&self.xbuf), legal.clone());
                    match by_key.get(&k) {
                        Some(&gi) => Some(gi),
                        None => {
                            by_key.insert(k, self.groups.len() as u32);
                            None
                        }
                    }
                }
            };
            let gi = match gi {
                Some(gi) => gi,
                None => {
                    self.groups.push(Group { members: Vec::new(), acts: legal, kids: Vec::new(), best: NIL });
                    (self.groups.len() - 1) as u32
                }
            };
            self.groups[gi as usize].members.push(m);
            self.mems[m as usize].group = gi;
        }
        self.level_groups = (first, self.groups.len());
        self.st.groups += (self.groups.len() - first) as u64;
        for gi in first..self.groups.len() {
            let rep = self.groups[gi].members[0];
            let n = self.groups[gi].acts.len();
            let capped = self.capped(self.game(rep)?);
            if capped && n > self.cfg.topk as usize && (self.level > 0 || self.cfg.topk_root) {
                self.prior_q.push_back(Req::Group(gi as u32));
            }
        }
        Ok(true)
    }

    /// Expand this level's groups (the lab's `solve` expansion): each group's actions (every option,
    /// or its top-k by our prior) for every member, in the lab's order (group, action, member).
    fn expand(&mut self) -> Result<(), Abort> {
        let (a, b) = self.level_groups;
        let mut children = Vec::new();
        for gi in a..b {
            let mut acts = std::mem::take(&mut self.groups[gi].acts);
            if let Some(p) = self.group_pri.remove(&(gi as u32)) {
                if p.len() != acts.len() {
                    return Err(Abort::Err(format!("internal: {} priors for {} options", p.len(), acts.len())));
                }
                let mut order = by_prior(&p);
                order.truncate(self.cfg.topk as usize);
                acts = order.iter().map(|&i| acts[i]).collect();
            }
            let members = std::mem::take(&mut self.groups[gi].members);
            self.add((acts.len() * members.len()) as u64)?;
            let mut kids = Vec::with_capacity(acts.len() * members.len());
            for &act in &acts {
                for &m in &members {
                    let (wt, plies, world) = {
                        let mm = &self.mems[m as usize];
                        (mm.wt, mm.plies, mm.world)
                    };
                    let mut c = Box::new(self.game(m)?.clone());
                    transition(&mut *c, act, None, &mut self.key, &mut self.legal).map_err(|e| format!("lookahead: our move {act} failed: {e}"))?;
                    let id = self.mems.len() as u32;
                    self.mems.push(Mem { g: Some(c), wt, value: f64::NAN, kids_off: 0, nkids: 0, group: NIL, choice: NIL, world, plies: plies + 1, act, mover: self.viewer as u8, kind: Kind::New });
                    kids.push(id);
                    children.push(id);
                }
            }
            self.live_games += kids.len() as u64;
            for &m in &members {
                self.mems[m as usize].g = None;
            }
            self.live_games -= members.len() as u64;
            let gr = &mut self.groups[gi];
            gr.acts = acts;
            gr.members = members;
            gr.kids = kids;
        }
        self.st.peak_games = self.st.peak_games.max(self.live_games);
        self.level += 1;
        self.pending = children;
        Ok(())
    }

    /// A member's value (the lab's `value`): a leaf's evaluation, our decision's choice, an opponent
    /// decision's aggregate over its replies by the model (memoized).
    fn value(&mut self, m: u32) -> Result<f64, String> {
        let mm = &self.mems[m as usize];
        if !mm.value.is_nan() {
            return Ok(mm.value);
        }
        match mm.kind {
            Kind::Opp => {
                let (off, n) = (mm.kids_off as usize, mm.nkids as usize);
                let mut vs = Vec::with_capacity(n);
                for j in 0..n {
                    let k = self.kid_ids[off + j];
                    vs.push(self.value(k)?);
                }
                let v = match self.cfg.opp {
                    Opp::Top => vs[0],
                    Opp::Min => vs.iter().copied().fold(f64::INFINITY, f64::min),
                    Opp::Expect => {
                        let ps = &self.kid_probs[off..off + n];
                        let tot: f64 = ps.iter().sum();
                        if tot > 0.0 {
                            ps.iter().zip(&vs).map(|(p, v)| p * v).sum::<f64>() / tot
                        } else {
                            vs.iter().sum::<f64>() / n as f64
                        }
                    }
                };
                self.mems[m as usize].value = v;
                Ok(v)
            }
            k => Err(format!("internal: member {m} ({k:?}) has no value")),
        }
    }

    /// Back up every level from the deepest (the lab's `solve` after its recursion): each group's
    /// best action by the weighted average over its members (ties within `TIE`: the first).
    fn backup(&mut self) -> Result<(), String> {
        for gi in (0..self.groups.len()).rev() {
            let n = self.groups[gi].members.len();
            let na = self.groups[gi].acts.len();
            let tot: f64 = self.groups[gi].members.iter().map(|&m| self.mems[m as usize].wt).sum();
            let tot = if tot != 0.0 { tot } else { 1.0 };
            let mut best = None;
            let mut best_avg = 0.0;
            for j in 0..na {
                let mut s = 0.0;
                for i in 0..n {
                    let (m, k) = (self.groups[gi].members[i], self.groups[gi].kids[j * n + i]);
                    let w = self.mems[m as usize].wt;
                    s += w * self.value(k)?;
                }
                let avg = s / tot;
                if best.is_none() || avg > best_avg + TIE {
                    best = Some(j);
                    best_avg = avg;
                }
            }
            let j = best.ok_or("internal: a group without actions")?;
            self.groups[gi].best = j as u32;
            for i in 0..n {
                let (m, k) = (self.groups[gi].members[i], self.groups[gi].kids[j * n + i]);
                let v = self.value(k)?;
                let mm = &mut self.mems[m as usize];
                mm.choice = k;
                mm.value = v;
            }
        }
        Ok(())
    }

    /// Push this round's rows: the blocking priors first, then leaf values up to `rows` (with `dedup`
    /// a leaf whose observation was sent already takes that row's value) -> Some(Step::Rows), or None
    /// when every leaf left was valued by an earlier row (no round).
    fn emit(&mut self, sink: &mut dyn RowSink<G>) -> Result<Option<Step>, String> {
        let cap = self.cfg.rows as usize;
        while self.inflight.len() < cap {
            let Some(q) = self.prior_q.pop_front() else { break };
            match q {
                Req::Belief | Req::RootPrior => {
                    self.legal.clear();
                    let g: &G = &self.live;
                    if matches!(q, Req::RootPrior) {
                        g.legal_actions(&mut self.legal);
                    }
                    sink.row(g, self.viewer, &self.legal, if matches!(q, Req::Belief) { ROW_BELIEF } else { ROW_LEAF });
                }
                Req::Prior(m) => {
                    let g = self.mems[m as usize].g.as_deref().ok_or("internal: a prior row without a game")?;
                    self.legal.clear();
                    g.legal_actions(&mut self.legal);
                    let p = g.to_move().ok_or("internal: a prior row at a finished game")?;
                    sink.row(g, p, &self.legal, ROW_LEAF);
                    self.st.prior_rows += 1;
                }
                Req::Group(gi) => {
                    let rep = self.groups[gi as usize].members[0];
                    let g = self.mems[rep as usize].g.as_deref().ok_or("internal: a group prior row without a game")?;
                    sink.row(g, self.viewer, &self.groups[gi as usize].acts, ROW_LEAF);
                    self.st.prior_rows += 1;
                }
                Req::Value(_) => return Err("internal: a value in the prior queue".into()),
            }
            self.inflight.push(q);
        }
        while self.inflight.len() < cap {
            let Some(m) = self.value_q.pop_front() else { break };
            let g = self.mems[m as usize].g.take().ok_or("internal: a value row without a game")?;
            self.live_games -= 1;
            if !self.cfg.dedup {
                sink.row(&g, self.viewer, &[], ROW_LEAF);
            } else {
                // (encoded in place, the row dropped again when its observation was sent already)
                let x = sink.push_row(self.viewer, &[], ROW_LEAF);
                g.encode(self.viewer, x);
                let h = obs_hash(x);
                match self.vcache.get_mut(&h) {
                    Some(VEntry::Val(v)) => {
                        sink.pop_row();
                        self.mems[m as usize].value = *v;
                        self.st.value_dups += 1;
                        continue;
                    }
                    Some(VEntry::Wait(w)) => {
                        sink.pop_row();
                        w.push(m);
                        self.st.value_dups += 1;
                        continue;
                    }
                    None => {
                        self.vcache.insert(h, VEntry::Wait(Vec::new()));
                        self.vhash.insert(m, h);
                    }
                }
            }
            self.st.value_rows += 1;
            self.inflight.push(Req::Value(m));
        }
        if self.inflight.is_empty() {
            return Ok(None);
        }
        self.st.rounds += 1;
        self.st.rows += self.inflight.len() as u64;
        Ok(Some(Step::Rows(self.inflight.len())))
    }

    /// The network's answers to the last `Step::Rows` (row i of the round = row `ans.base + i`).
    pub fn answer(&mut self, ans: &Answers<'_>) -> Result<(), String> {
        let inflight = std::mem::take(&mut self.inflight);
        for (i, q) in inflight.iter().enumerate() {
            let b = ans.base + i;
            if b + 1 >= ans.opt_off.len() || (b + 1) * MAX_SEATS > ans.value.len() {
                return Err(format!("lookahead: row {b} is not in the answers"));
            }
            let pri = &ans.priors[ans.opt_off[b] as usize..ans.opt_off[b + 1] as usize];
            match *q {
                Req::Belief => {
                    let k = ans.aux_row.get(b).copied().unwrap_or(u32::MAX);
                    if k == u32::MAX || ans.aux_width == 0 {
                        return Err("lookahead: the root belief row has no auxiliary output".into());
                    }
                    let w = ans.aux_width;
                    let (t, c, _) = self.belief_ask.ok_or("internal: a belief answer without a belief")?;
                    self.belief = Some(Belief { pred: ans.aux[k as usize * w..(k as usize + 1) * w].to_vec(), tilt: t, clip: c });
                }
                Req::RootPrior => self.root_pri = Some(pri.to_vec()),
                Req::Prior(m) => {
                    self.mem_pri.insert(m, pri.to_vec());
                }
                Req::Group(gi) => {
                    self.group_pri.insert(gi, pri.to_vec());
                }
                Req::Value(m) => {
                    let v = ans.value[b * MAX_SEATS] as f64;
                    if !v.is_finite() {
                        return Err(format!("lookahead: a leaf value {v}"));
                    }
                    self.mems[m as usize].value = v;
                    if let Some(h) = self.vhash.remove(&m) {
                        if let Some(VEntry::Wait(w)) = self.vcache.insert(h, VEntry::Val(v)) {
                            for k in w {
                                self.mems[k as usize].value = v;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Run until rows are needed (`Step::Rows`: answer them) or the decision is made (`Step::Done`).
    /// `rng` deals the worlds when neither worlds nor seeds were given (the driver's search stream).
    pub fn step(&mut self, sink: &mut dyn RowSink<G>, rng: &mut Rng) -> Result<Step, String> {
        if !self.inflight.is_empty() {
            return Err("lookahead: answer the last rows first".into());
        }
        loop {
            match self.phase {
                Phase::Belief => {
                    if self.belief.is_some() {
                        self.phase = Phase::Deal;
                        continue;
                    }
                    self.prior_q.push_back(Req::Belief);
                    if let Some(r) = self.emit(sink)? {
                        return Ok(r);
                    }
                    continue;
                }
                Phase::RootPrior => {
                    if let Some(p) = self.root_pri.take() {
                        if p.len() != self.root_legal.len() {
                            return Err(format!("internal: {} root priors for {} options", p.len(), self.root_legal.len()));
                        }
                        self.outcome = Some(Outcome::Act(self.root_legal[argmax_first(&p)]));
                        self.phase = Phase::Done;
                        continue;
                    }
                    self.prior_q.push_back(Req::RootPrior);
                    if let Some(r) = self.emit(sink)? {
                        return Ok(r);
                    }
                    continue;
                }
                Phase::Deal => {
                    if self.worlds.is_empty() {
                        self.deal_from(rng)?;
                    }
                    self.check_worlds()?;
                    self.start_stage();
                }
                Phase::Classify => {
                    if self.value_q.len() >= self.cfg.rows as usize {
                        if let Some(r) = self.emit(sink)? {
                        return Ok(r);
                    }
                    continue;
                    }
                    self.classify()?;
                    if self.adv_minor.is_empty() && self.adv_opp.is_empty() {
                        self.phase = Phase::Solve;
                        continue;
                    }
                    if self.prior_q.is_empty() {
                        match self.advance() {
                            Ok(()) => continue,
                            Err(Abort::Budget) => {
                                self.next_stage();
                                continue;
                            }
                            Err(Abort::Err(e)) => return Err(e),
                        }
                    }
                    self.phase = Phase::AdvWait;
                    if let Some(r) = self.emit(sink)? {
                        return Ok(r);
                    }
                    continue;
                }
                Phase::AdvWait => {
                    if !self.prior_q.is_empty() {
                        if let Some(r) = self.emit(sink)? {
                        return Ok(r);
                    }
                    continue;
                    }
                    match self.advance() {
                        Ok(()) => self.phase = Phase::Classify,
                        Err(Abort::Budget) => self.next_stage(),
                        Err(Abort::Err(e)) => return Err(e),
                    }
                }
                Phase::Solve => {
                    if self.value_q.len() >= self.cfg.rows as usize {
                        if let Some(r) = self.emit(sink)? {
                        return Ok(r);
                    }
                    continue;
                    }
                    if !self.group()? {
                        self.phase = Phase::Flush;
                        continue;
                    }
                    if self.prior_q.is_empty() {
                        match self.expand() {
                            Ok(()) => self.phase = Phase::Classify,
                            Err(Abort::Budget) => self.next_stage(),
                            Err(Abort::Err(e)) => return Err(e),
                        }
                        continue;
                    }
                    self.phase = Phase::SolveWait;
                    if let Some(r) = self.emit(sink)? {
                        return Ok(r);
                    }
                    continue;
                }
                Phase::SolveWait => {
                    if !self.prior_q.is_empty() {
                        if let Some(r) = self.emit(sink)? {
                        return Ok(r);
                    }
                    continue;
                    }
                    match self.expand() {
                        Ok(()) => self.phase = Phase::Classify,
                        Err(Abort::Budget) => self.next_stage(),
                        Err(Abort::Err(e)) => return Err(e),
                    }
                }
                Phase::Flush => {
                    if !self.value_q.is_empty() {
                        if let Some(r) = self.emit(sink)? {
                        return Ok(r);
                    }
                    continue;
                    }
                    self.backup()?;
                    let opts = self.options()?;
                    // the decision: the best mean over the worlds, the first in the root order on ties
                    // (the lab sorts its options by value, stably)
                    let mut best: Option<&OptionValue> = None;
                    for o in &opts {
                        if best.is_none_or(|b| o.mean > b.mean) {
                            best = Some(o);
                        }
                    }
                    self.outcome = Some(Outcome::Act(best.ok_or("internal: no root option")?.act));
                    self.phase = Phase::Done;
                }
                Phase::Done => return Ok(Step::Done),
            }
        }
    }

    /// Per root option (in the order of the first world's root group, then any other world's new
    /// ones): its value in each world whose group expanded it, their mean and sd (n - 1).
    pub fn options(&mut self) -> Result<Vec<OptionValue>, String> {
        let mut out: Vec<OptionValue> = Vec::new();
        for ri in 0..self.roots.len() {
            let r = self.roots[ri];
            let mm = &self.mems[r as usize];
            if mm.group == NIL {
                continue;
            }
            let world = mm.world;
            let gi = mm.group as usize;
            let n = self.groups[gi].members.len();
            let i = self.groups[gi].members.iter().position(|&x| x == r).ok_or("internal: a root outside its group")?;
            for j in 0..self.groups[gi].acts.len() {
                let a = self.groups[gi].acts[j];
                let k = self.groups[gi].kids[j * n + i];
                let v = self.value(k)?;
                match out.iter_mut().find(|o| o.act == a) {
                    Some(o) => o.values.push((world, v)),
                    None => out.push(OptionValue { act: a, values: vec![(world, v)], mean: 0.0, sd: 0.0 }),
                }
            }
        }
        for o in out.iter_mut() {
            let n = o.values.len() as f64;
            o.mean = o.values.iter().map(|x| x.1).sum::<f64>() / n;
            o.sd = if o.values.len() > 1 { (o.values.iter().map(|x| (x.1 - o.mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt() } else { 0.0 };
        }
        Ok(out)
    }

    /// The line of world `world` after root option `act` under the solved policy (ours: the group's
    /// choice; the opponent's: its top move / its reply worst for us / its most likely), with the
    /// minor moves (`keep_lines`), and where it ended. None: the world's root group did not expand it.
    pub fn line(&mut self, world: usize, act: u16) -> Result<Option<Line>, String> {
        let Some(&r) = self.roots.get(world) else { return Ok(None) };
        let gi = self.mems[r as usize].group;
        if gi == NIL {
            return Ok(None);
        }
        let gr = &self.groups[gi as usize];
        let n = gr.members.len();
        let Some(j) = gr.acts.iter().position(|&a| a == act) else { return Ok(None) };
        let i = gr.members.iter().position(|&x| x == r).ok_or("internal: a root outside its group")?;
        let mut out = Vec::new();
        let mut m = gr.kids[j * n + i];
        out.push((self.viewer as u8, act, false));
        loop {
            if let Some(ms) = self.minors.get(&m) {
                for &(mover, a) in ms {
                    out.push((mover, a, true));
                }
            }
            let mm = &self.mems[m as usize];
            match mm.kind {
                Kind::Opp => {
                    let (off, nk) = (mm.kids_off as usize, mm.nkids as usize);
                    let idx = match self.cfg.opp {
                        Opp::Top => 0,
                        Opp::Min => {
                            let mut b = 0;
                            let mut bv = f64::INFINITY;
                            for jj in 0..nk {
                                let v = self.value(self.kid_ids[off + jj])?;
                                if v < bv {
                                    b = jj;
                                    bv = v;
                                }
                            }
                            b
                        }
                        Opp::Expect => {
                            let ps = &self.kid_probs[off..off + nk];
                            let mut b = 0;
                            for jj in 1..nk {
                                if ps[jj] > ps[b] {
                                    b = jj;
                                }
                            }
                            b
                        }
                    };
                    let k = self.kid_ids[off + idx];
                    let km = &self.mems[k as usize];
                    out.push((km.mover, km.act, false));
                    m = k;
                }
                Kind::Ours if mm.choice != NIL => {
                    let k = mm.choice;
                    out.push((self.viewer as u8, self.mems[k as usize].act, false));
                    m = k;
                }
                Kind::Leaf(e) => return Ok(Some((out, e))),
                k => return Err(format!("internal: a line through a member of kind {k:?}")),
            }
        }
    }

    /// The members' games still held (memory; 0 after the decision).
    pub fn live_games(&self) -> u64 {
        self.live_games
    }
}

#[cfg(test)]
mod tests;
