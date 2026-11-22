//! Game records (zero-farm DESIGN.md 6.1, docs/search.md 2.17): one self-play game as what replays it - its
//! seed, every real move (forced single-option steps and external answers included) - plus only what the
//! search produced: per training row its step index, decider, kind (leaf / value-only), policy target as
//! float16 over that step's options, root value `q` (float16 per seat, relative to the decider) and network
//! id. Everything else of a training row (`x`, the options, `z`, `sc`, the auxiliary targets) is the game's:
//! `replay` rebuilds the rows from a record exactly as the driver emitted them.
//!
//! A driver records its games' traces when asked (`Driver::set_trace`, off by default: nothing changes):
//! every finished game's `GameResult::trace` holds its seed, its real moves and the game's own record of
//! its setup (`SearchGame::record_setup`: MTG Zero's per-seat decks), and each training record its
//! `Record::step` (the real moves played before its decision). `game_json` turns a finished game and its
//! samples into a record (JSON, one object); `replay` plays it back.
//!
//! The record of one game (`records/1`):
//!
//! ```text
//! {"gid": u32, "n": players, "seed": u64 (the game's: new_game(spec, n, seed)),
//!  "steps": [action (u16) | "external answer" (string) | {"next": conceding seat | null} (a chained contest's
//!            episode boundary: SearchGame::next_episode), ...],
//!  "over": bool, "aborted": bool, "resigned": winner seat | null,
//!  "abort_reason": "max_row_tokens" (only a game the driver aborted at its row limit, docs/search.md 2.15: not
//!                  over, no samples of its own; left out otherwise),
//!  "result": [f32 bits per seat], "margin": [f32 bits per seat] (over games; else []), "rounds": u32,
//!  "setup": "<SearchGame::record_setup>" (left out when empty),
//!  "episodes": [{"over", "aborted", "resigned", "result", "margin", "rounds", "setup"}, ...] (a chained contest's
//!              episodes in order - each its own outcome, which its samples' z and sc are; the top-level fields
//!              are then the contest's, its "margin" [] and "resigned" null),
//!  "samples": {"step": [...], "player": [...], "kind": [0 leaf | 1 value-only, ...], "net": [...],
//!              "pi": [f16 bits over each leaf row's options, in order], "q": [f16 bits x n per row]}}
//! ```
//!
//! Replay is the validation (DESIGN.md 6.1): an illegal or impossible move, a sample whose decider is not
//! the step's (or whose options don't match its policy's length), a terminal result, margin, round count or
//! setup that differs from the record's - per episode of a chained contest too, and the contest's own result
//! from `next_episode` - any of them rejects the whole record. The search's outputs (`pi`, `q`) can't be checked
//! without searching again; they are checked to be finite and in [0, 1].
//!
//! A value-only decision (docs/search.md 3.20, `SearchGame::value_only_decision`) is a value-only sample (kind 1) of
//! the seat to move: the replay accepts one only where the game says so (and the decision has two or more options),
//! and refuses a leaf sample there (unless the game names the decision's policy: a teacher decision, 3.19). Nothing new
//! in the format.
//!
//! A game with variable-length rows (docs/search.md 2.15: `SearchGame::TAILS` / `OPTION_POINTERS`) replays each
//! row as the driver built it - the head, its sealed tail tokens and its pointer overrides against the row's
//! options (`driver::record_row`) - and refuses a row past the game's `MAX_ROW_TOKENS` (the driver would have
//! aborted that game: no samples). A game the driver aborted at the limit (`abort_reason`) replays to a game still
//! in progress, as the driver left it.
use serde_json::{json, Map, Value};

use crate::driver::{f16_to_f32, f32_to_f16, record_row, GameResult, Record};
use crate::game::{Next, SearchGame, MAX_SEATS};

/// The record format's version (`records/1`).
pub const RECORDS_VERSION: u32 = 1;
/// `records/2`: `records/1` whose files may also hold **branch games** (`DriverCfg::branch`, docs/search.md 2.11 and
/// 2.17): a branch's record carries its game's gid and seed, every real move from the game's start (its game's moves
/// to the branch point, then its own) and a `branch` block; it replays with its game's record (`replay_branch`). A
/// writer of a driver without branches writes `records/1`, byte for byte as before.
pub const RECORDS_VERSION_BRANCH: u32 = 2;

/// One real move of a game.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TraceStep {
    /// A legal action (decisions with a single option included: the driver steps them as moves).
    Act(u16),
    /// An external seat's answer (`SearchGame::apply_external`'s payload).
    Ext(String),
    /// A chained contest's episode ended and the next began (`SearchGame::next_episode`, docs/search.md 2.14): the
    /// seat that conceded the episode (the resign rule), or None when it ended by the rules.
    Next(Option<u8>),
}

/// What replays a finished game (`Driver::set_trace`): its seed (`SearchGame::new_game`'s), its real moves
/// in order, the game's record of its setup (`SearchGame::record_setup`) and, for a game that ended with a
/// result, each seat's margin (`SearchGame::margin`, what its samples' `sc` hold); a chained contest's each
/// episode's setup record (its margins are in the row's episodes).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GameTrace {
    pub seed: u64,
    pub steps: Vec<TraceStep>,
    pub setup: String,
    /// `margin(seat)` per seat when the game is over (a result or a resignation), else empty; empty for a contest.
    pub margins: Vec<f32>,
    /// A chained contest: each episode's `record_setup` at its end, in order; empty for any other game.
    pub episode_setups: Vec<String>,
}

/// Kind of a sample in a record: a searched decision's row (with its options and policy target).
pub const KIND_LEAF: u8 = 0;
/// Kind of a sample in a record: a value-only row (no options; `NnSeat::view_samples`).
pub const KIND_VALUE: u8 = 1;

fn bits32(v: &[f32]) -> Vec<Value> {
    v.iter().map(|x| json!(x.to_bits())).collect()
}

/// A trace's steps as a record holds them: an action as its number, an external answer as its payload string, an
/// episode boundary as `{"next": <the conceding seat or null>}` (also a dropped game's, `GameFailure::to_json`).
pub fn steps_json(steps: &[TraceStep]) -> Vec<Value> {
    steps
        .iter()
        .map(|s| match s {
            TraceStep::Act(a) => json!(a),
            TraceStep::Ext(p) => json!(p),
            TraceStep::Next(c) => json!({"next": c}),
        })
        .collect()
}

/// The record of finished game `row` (its trace on: `Driver::set_trace`) and its emitted samples (`row.samples`
/// of them, in order, as `take_samples` gave them) - one JSON object (module docs).
pub fn game_json(row: &GameResult, samples: &[Record]) -> Result<Value, String> {
    let t = row.trace.as_ref().ok_or_else(|| format!("game {}: no trace (Driver::set_trace)", row.gid))?;
    if samples.len() != row.samples as usize {
        return Err(format!("game {}: {} samples given, its row says {}", row.gid, samples.len(), row.samples));
    }
    let n = row.n;
    let steps = steps_json(&t.steps);
    let (mut st, mut pl, mut kind, mut net, mut pi, mut q) = (vec![], vec![], vec![], vec![], vec![], vec![]);
    for r in samples {
        if r.n_players as usize != n {
            return Err(format!("game {}: a sample of {} players in a game of {n}", row.gid, r.n_players));
        }
        st.push(json!(r.step));
        pl.push(json!(r.player));
        let k = if r.opts.is_empty() { KIND_VALUE } else { KIND_LEAF };
        if k == KIND_VALUE && !r.pi.is_empty() {
            return Err(format!("game {}: a value-only sample with a policy", row.gid));
        }
        if r.pi.len() != r.opts.len() {
            return Err(format!("game {}: a sample's policy over {} values for {} options", row.gid, r.pi.len(), r.opts.len()));
        }
        kind.push(json!(k));
        net.push(json!(r.net));
        pi.extend(r.pi.iter().map(|&x| json!(f32_to_f16(x))));
        q.extend(r.q[..n].iter().map(|&x| json!(f32_to_f16(x))));
    }
    let resigned = row.resign.as_ref().filter(|r| r.resigned).and_then(|r| r.winner);
    let mut o = Map::new();
    o.insert("gid".into(), json!(row.gid));
    o.insert("n".into(), json!(n));
    o.insert("seed".into(), json!(t.seed));
    o.insert("steps".into(), Value::Array(steps));
    o.insert("over".into(), json!(row.over));
    o.insert("aborted".into(), json!(row.aborted));
    o.insert("resigned".into(), json!(resigned));
    if let Some(why) = row.abort_reason {
        // (a game the driver aborted at its row limit, docs/search.md 2.15; the key exists only then)
        o.insert("abort_reason".into(), json!(why));
    }
    o.insert("result".into(), Value::Array(bits32(&row.result)));
    o.insert("margin".into(), Value::Array(bits32(&t.margins)));
    o.insert("rounds".into(), json!(row.rounds));
    if !t.setup.is_empty() {
        o.insert("setup".into(), json!(t.setup));
    }
    if let Some(b) = &row.branch {
        // a branch game (records/2, docs/search.md 2.17): where it leaves its game - `from` real moves of the game, then
        // its own - and its origin; a merged near-tie branch's result and margins are its game's
        o.insert(
            "branch".into(),
            json!({"from": b.trace_from, "at": b.at, "of": b.of, "decision": b.decision, "step": b.step, "action": b.action, "moves": b.moves,
                   "played": b.played, "same_first": b.same_first, "alt": b.alt, "gap": b.gap, "tag": b.tag, "merged": b.merged}),
        );
    }
    if let Some(k) = row.neartie {
        // (the near-tie rule's eligible decisions of the game, docs/search.md 2.11.1: a tally, not replayed)
        o.insert("neartie".into(), json!(k));
    }
    if !row.episodes.is_empty() {
        if t.episode_setups.len() != row.episodes.len() {
            return Err(format!("game {}: {} episodes, {} setup records", row.gid, row.episodes.len(), t.episode_setups.len()));
        }
        let eps: Vec<Value> = row
            .episodes
            .iter()
            .zip(&t.episode_setups)
            .map(|(e, s)| {
                let w = e.resign.as_ref().filter(|r| r.resigned).and_then(|r| r.winner);
                json!({"over": e.over, "aborted": e.aborted, "resigned": w, "result": bits32(&e.result), "margin": bits32(&e.margins),
                       "rounds": e.rounds, "setup": s})
            })
            .collect();
        o.insert("episodes".into(), Value::Array(eps));
    }
    o.insert("samples".into(), json!({"step": st, "player": pl, "kind": kind, "net": net, "pi": pi, "q": q}));
    Ok(Value::Object(o))
}

/// A replayed record: the training rows exactly as the driver emitted them (`Record`s in the record's order:
/// `x` = `encode(decider)`, the options, `z` = the result, `sc` = the margins, `aux` = the decider's auxiliary
/// targets with `aux` on; `pi`, `q`, `net` the record's) and the game's facts.
#[derive(Clone, Debug)]
pub struct Replayed {
    pub gid: u32,
    pub n: usize,
    pub seed: u64,
    pub over: bool,
    pub aborted: bool,
    pub resigned: Option<u8>,
    /// The driver aborted the game at its row limit ("max_row_tokens", docs/search.md 2.15).
    pub abort_reason: Option<&'static str>,
    pub result: Vec<f32>,
    pub rounds: u32,
    /// Real moves replayed (a contest's episode boundaries included).
    pub steps: usize,
    pub setup: String,
    /// A chained contest's episodes (1 for any other game).
    pub episodes: usize,
    pub records: Vec<Record>,
    /// A branch game's record (`replay_branch`): its game's real moves it starts with, and - merged - its searched
    /// decisions after its deviation (its rows then carry its game's outcome). None for any other game.
    pub branch: Option<(u32, Option<u32>)>,
}

/// What a branch record is checked against (`replay_branch`): where it leaves its game, and - merged - its game's
/// outcome and the merge keys of its game's decision states after the branch point.
struct BranchCtx {
    from: usize,
    merged: Option<u32>,
    result: Vec<f32>,
    margin: Vec<f32>,
    keys: std::collections::HashSet<u128>,
}

fn field<'a>(o: &'a Map<String, Value>, k: &str) -> Result<&'a Value, String> {
    o.get(k).ok_or_else(|| format!("record: no {k:?}"))
}

fn uint(v: &Value, what: &str, max: u64) -> Result<u64, String> {
    v.as_u64().filter(|&x| x <= max).ok_or_else(|| format!("record: {what} {v} is not an integer in [0, {max}]"))
}

fn uints(o: &Map<String, Value>, k: &str, max: u64) -> Result<Vec<u64>, String> {
    let a = field(o, k)?.as_array().ok_or_else(|| format!("record: {k} is not a list"))?;
    a.iter().map(|v| uint(v, k, max)).collect()
}

fn f32s(o: &Map<String, Value>, k: &str) -> Result<Vec<f32>, String> {
    Ok(uints(o, k, u32::MAX as u64)?.into_iter().map(|b| f32::from_bits(b as u32)).collect())
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn setup_of(o: &Map<String, Value>) -> Result<String, String> {
    match o.get("setup") {
        None => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(v) => Err(format!("record: setup {v} is not a string")),
    }
}

/// An episode's outcome as the record states it (a whole game's: the top-level fields; a contest's: one of its
/// `episodes`).
struct Ep {
    over: bool,
    aborted: bool,
    resigned: Option<u8>,
    result: Vec<f32>,
    margin: Vec<f32>,
    rounds: u32,
    setup: String,
}

fn ep_of(o: &Map<String, Value>, n: usize) -> Result<Ep, String> {
    let resigned = match field(o, "resigned")? {
        Value::Null => None,
        v => Some(uint(v, "resigned", n as u64 - 1)? as u8),
    };
    Ok(Ep {
        over: field(o, "over")?.as_bool().ok_or("record: over is not a bool")?,
        aborted: field(o, "aborted")?.as_bool().ok_or("record: aborted is not a bool")?,
        resigned,
        result: f32s(o, "result")?,
        margin: f32s(o, "margin")?,
        rounds: uint(field(o, "rounds")?, "rounds", u32::MAX as u64)? as u32,
        setup: setup_of(o)?,
    })
}

/// The end of an episode (a whole game's, or a contest's episode) checked against the record's `e`, as the driver
/// ended it (driver::worker::finish / finish_episode, emit) -> (its result: the samples' z, its margins: their sc).
/// `chained`: a contest's episode (its margins are zeros when it is not over, the driver's `EpisodeRow`).
/// `row_abort`: the driver aborted it at its row limit (the record's `abort_reason`; docs/search.md 2.15) - not over,
/// aborted, the game still in progress where the driver stopped it.
fn end_episode<G: SearchGame>(g: &G, e: &Ep, n: usize, samples: usize, chained: bool, row_abort: bool, what: &str) -> Result<(Vec<f32>, Vec<f32>), String> {
    if e.resigned.is_some() && (!e.over || e.aborted) {
        return Err(format!("record: {what}: a resigned episode is over and not aborted"));
    }
    if row_abort && (e.over || !e.aborted || e.resigned.is_some() || g.to_move().is_none()) {
        return Err(format!("record: {what}: aborted at the row limit, so aborted, not over, not resigned and still in progress (over {}, aborted {}, resigned {:?}, the replay {})",
                           e.over, e.aborted, e.resigned, if g.to_move().is_none() { "ended" } else { "in progress" }));
    }
    if samples > 0 && !e.over {
        return Err(format!("record: {what}: samples from an episode without a result (one cut short emits none)"));
    }
    let ended = g.to_move().is_none();
    let res: Vec<f32> = match e.resigned {
        Some(w) => {
            if ended || !g.can_concede() {
                return Err(format!("record: {what}: resigned where no decision could be conceded"));
            }
            (0..n).map(|s| if s == w as usize { 1.0 } else { 0.0 }).collect()
        }
        None if e.over => {
            if !ended || g.aborted() || g.deferred() {
                return Err(format!("record: {what}: over, its replay {}", if ended { "aborted or deferred" } else { "is not" }));
            }
            g.result()[..n].to_vec()
        }
        None => {
            if ended && !g.aborted() {
                return Err(format!("record: {what}: without a result, but its replay finished with one"));
            }
            if e.aborted != (row_abort || (ended && g.aborted())) {
                return Err(format!("record: {what}: aborted {}, the replay {}", e.aborted, ended && g.aborted()));
            }
            vec![1.0 / n as f32; n]
        }
    };
    if !same_bits(&res, &e.result) {
        return Err(format!("record: {what}: result {:?}, the replay's {res:?}", e.result));
    }
    let margins: Vec<f32> = if e.over {
        (0..n).map(|s| g.margin(s)).collect()
    } else if chained {
        vec![0.0; n]
    } else {
        Vec::new()
    };
    if !same_bits(&margins, &e.margin) {
        return Err(format!("record: {what}: margins {:?}, the replay's {margins:?}", e.margin));
    }
    if g.rounds() != e.rounds {
        return Err(format!("record: {what}: {} rounds, the replay's {}", e.rounds, g.rounds()));
    }
    let set = g.record_setup();
    if set != e.setup {
        return Err(format!("record: {what}: the game's setup differs from the record's (record {} bytes, replay {} bytes)", e.setup.len(), set.len()));
    }
    Ok((res, margins))
}

/// Label records `from..` with their episode's outcome (driver::worker::emit: z and sc relative to the decider).
fn label(records: &mut [Record], res: &[f32], margins: &[f32], n: usize) {
    for r in records {
        for c in 0..n {
            let s = (r.player as usize + c) % n;
            r.z[c] = res[s];
            r.sc[c] = margins[s];
        }
    }
}

/// Replay one game record (`game_json`'s object) of game type `G` with spec `spec` -> its training rows and
/// facts, or why the record is refused (module docs). `aux`: the rows carry the decider's auxiliary targets
/// (the driver's `aux`, the run's auxiliary targets on). The game is replayed on a copy of the new game, never
/// on the game `new_game` made (a game may do what only its live game does - MTG Zero's EoE game log - and a
/// replay is not a live game).
pub fn replay<G: SearchGame>(spec: &G::Spec, rec: &Value, aux: bool) -> Result<Replayed, String> {
    let o = rec.as_object().ok_or("record: not a JSON object")?;
    if o.contains_key("branch") {
        return Err("record: a branch game's record replays with its game's record (record::replay_branch)".into());
    }
    replay_core::<G>(spec, o, aux, None)
}

/// Replay a branch game's record (records/2: its `branch` block) with its game's record `parent` (the record of the
/// game it branched from: the same gid, seed and players, no `branch` block) -> its rows and facts, as `replay`. On top
/// of `replay`'s checks: its first `branch.from` real moves are its game's; a near-tie branch's next move is its
/// runner-up `branch.alt` and not its game's move there; no row at or before the branch point. A branch that played to
/// its end is checked as any game (its own outcome). A **merged** branch (`branch.merged`, docs/search.md 2.11.1) ends
/// at a decision of a game in progress: its result and margins must be its game's (its game over), the state's
/// `SearchGame::merge_key` must be one of its game's decision states' after the branch point, and its rows take that
/// outcome.
pub fn replay_branch<G: SearchGame>(spec: &G::Spec, rec: &Value, parent: &Value, aux: bool) -> Result<Replayed, String> {
    let o = rec.as_object().ok_or("record: not a JSON object")?;
    let po = parent.as_object().ok_or("record: the game's record is not a JSON object")?;
    let b = field(o, "branch")?.as_object().ok_or("record: branch is not an object")?;
    if po.contains_key("branch") {
        return Err("record: a branch's game record is itself a branch (a branch never branches)".into());
    }
    for k in ["gid", "n", "seed"] {
        if field(o, k)? != field(po, k)? {
            return Err(format!("record: the branch's {k} {} is not its game's {}", field(o, k)?, field(po, k)?));
        }
    }
    let n = uint(field(o, "n")?, "n", MAX_SEATS as u64)? as usize;
    let steps = field(o, "steps")?.as_array().ok_or("record: steps is not a list")?;
    let psteps = field(po, "steps")?.as_array().ok_or("record: the game's steps is not a list")?;
    let from = uint(field(b, "from")?, "branch.from", u32::MAX as u64)? as usize;
    if from >= steps.len() || from >= psteps.len() {
        return Err(format!("record: the branch leaves its game at move {from} of the game's {} and its own {} (it plays at least one move there)", psteps.len(), steps.len()));
    }
    if steps[..from] != psteps[..from] {
        return Err(format!("record: the branch's first {from} moves are not its game's"));
    }
    if let Some(a) = b.get("alt").filter(|v| !v.is_null()) {
        let a = uint(a, "branch.alt", u16::MAX as u64)?;
        if steps[from].as_u64() != Some(a) || psteps[from].as_u64() == Some(a) {
            return Err(format!("record: a near-tie branch plays its runner-up {a} where its game played {} (the branch played {})", psteps[from], steps[from]));
        }
    }
    let merged = match b.get("merged") {
        None | Some(Value::Null) => None,
        Some(v) => Some(uint(v, "branch.merged", u32::MAX as u64)? as u32),
    };
    let mut ctx = BranchCtx { from, merged, result: Vec::new(), margin: Vec::new(), keys: Default::default() };
    if merged.is_some() {
        let pe = ep_of(po, n)?;
        if !pe.over || pe.aborted {
            return Err("record: a merged branch takes its game's outcome, and its game has none (not over, or aborted)".into());
        }
        ctx.result = pe.result;
        ctx.margin = pe.margin;
        ctx.keys = merge_keys::<G>(spec, n, field(po, "seed")?.as_u64().ok_or("record: seed is not a u64")?, psteps, from)?;
    }
    replay_core::<G>(spec, o, aux, Some(&ctx))
}

/// The merge keys of a game's decision states after move `from` (a seat to move with two options or more), its real
/// moves `steps` replayed from `seed` (actions only: a merged branch's game is a game of actions).
fn merge_keys<G: SearchGame>(spec: &G::Spec, n: usize, seed: u64, steps: &[Value], from: usize) -> Result<std::collections::HashSet<u128>, String> {
    let mut g = G::new_game(spec, n, seed).map_err(|e| format!("record: the game's new_game: {e}"))?.clone();
    let (mut keys, mut legal, mut chance) = (std::collections::HashSet::new(), Vec::new(), Vec::new());
    for (k, v) in steps.iter().enumerate() {
        legal.clear();
        g.legal_actions(&mut legal);
        if k > from && g.to_move().is_some() && legal.len() >= 2 {
            if let Some(key) = g.merge_key() {
                keys.insert(key);
            }
        }
        let a = v.as_u64().filter(|&a| a <= u16::MAX as u64).ok_or_else(|| format!("record: the game's step {k} ({v}) is not an action"))? as u16;
        if !legal.contains(&a) {
            return Err(format!("record: the game's step {k}: action {a} is not legal"));
        }
        chance.clear();
        g.step(a, &mut chance).map_err(|e| format!("record: the game's step {k}: {e}"))?;
    }
    Ok(keys)
}

fn replay_core<G: SearchGame>(spec: &G::Spec, o: &Map<String, Value>, aux: bool, br: Option<&BranchCtx>) -> Result<Replayed, String> {
    let gid = uint(field(o, "gid")?, "gid", u32::MAX as u64)? as u32;
    let n = uint(field(o, "n")?, "n", MAX_SEATS as u64)? as usize;
    if n < G::MIN_PLAYERS || n > G::MAX_PLAYERS.min(MAX_SEATS) {
        return Err(format!("record: {n} players ({} plays {}..={})", G::NAME, G::MIN_PLAYERS, G::MAX_PLAYERS));
    }
    let seed = field(o, "seed")?.as_u64().ok_or("record: seed is not a u64")?;
    let top = ep_of(o, n)?;
    let abort_reason: Option<&'static str> = match o.get("abort_reason") {
        None => None,
        Some(Value::String(s)) if s == "max_row_tokens" => Some("max_row_tokens"),
        Some(v) => return Err(format!("record: abort_reason {v} (the only reason a driver aborts a game is \"max_row_tokens\")")),
    };
    if abort_reason.is_some() && (G::TAILS.is_empty() || top.over || !top.aborted) {
        return Err(format!("record: aborted at the row limit: {} (over {}, aborted {})", if G::TAILS.is_empty() { format!("{} has no tail tokens, so no row limit", G::NAME) } else { "so aborted and not over".to_string() }, top.over, top.aborted));
    }
    let row_abort = abort_reason.is_some();
    if top.result.len() != n {
        return Err(format!("record: a result of {} seats in a game of {n}", top.result.len()));
    }
    let eps: Vec<Ep> = match o.get("episodes") {
        None => Vec::new(),
        Some(Value::Array(a)) if !a.is_empty() => a
            .iter()
            .map(|v| v.as_object().ok_or_else(|| "record: an episode is not an object".to_string()).and_then(|e| ep_of(e, n)))
            .collect::<Result<_, _>>()?,
        Some(v) => return Err(format!("record: episodes {v}: a non-empty list")),
    };
    let steps = field(o, "steps")?.as_array().ok_or("record: steps is not a list")?;
    let s = field(o, "samples")?.as_object().ok_or("record: samples is not an object")?;
    let st = uints(s, "step", steps.len() as u64)?;
    let pl = uints(s, "player", n as u64 - 1)?;
    let kind = uints(s, "kind", KIND_VALUE as u64)?;
    let net = uints(s, "net", u32::MAX as u64)?;
    let pi = uints(s, "pi", u16::MAX as u64)?;
    let q = uints(s, "q", u16::MAX as u64)?;
    let ns = st.len();
    if pl.len() != ns || kind.len() != ns || net.len() != ns || q.len() != ns * n {
        return Err(format!("record: samples of {ns} steps, {} players, {} kinds, {} nets, {} q values ({n} per row)", pl.len(), kind.len(), net.len(), q.len()));
    }
    if st.windows(2).any(|w| w[0] > w[1]) {
        return Err("record: the samples' steps are not in order".into());
    }
    if let Some(b) = br {
        // (a branch records nothing at or before its deviation: docs/search.md 2.11)
        if st.first().is_some_and(|&k| k as usize <= b.from) {
            return Err(format!("record: a branch's row at move {} (it leaves its game at move {})", st[0], b.from));
        }
    }
    let unit = |h: u64| {
        let x = f16_to_f32(h as u16);
        x.is_finite() && (0.0..=1.0).contains(&x)
    };
    if let Some(i) = pi.iter().position(|&h| !unit(h)) {
        return Err(format!("record: policy value {i} (f16 {:#06x}) is not in [0, 1]", pi[i]));
    }
    if let Some(i) = q.iter().position(|&h| !unit(h)) {
        return Err(format!("record: q value {i} (f16 {:#06x}) is not in [0, 1]", q[i]));
    }

    // the game, replayed on a copy of the game new_game made (module docs)
    let mut g = G::new_game(spec, n, seed).map_err(|e| format!("record: new_game: {e}"))?.clone();
    if g.num_players() != n {
        return Err(format!("record: new_game made a game of {} players, the record says {n}", g.num_players()));
    }
    let chained = g.chained();
    if chained == eps.is_empty() {
        return Err(format!("record: {} game {}, the record {} episodes", G::NAME, if chained { "chains its episodes" } else { "does not chain episodes" }, if eps.is_empty() { "lists no" } else { "lists" }));
    }
    let mut records: Vec<Record> = Vec::with_capacity(ns);
    let mut legal: Vec<u16> = Vec::new();
    let mut chance: Vec<u64> = Vec::new();
    let (mut j, mut pk) = (0usize, 0usize);
    let (mut ep, mut ep_first) = (0usize, 0usize);
    for k in 0..=steps.len() {
        while j < ns && st[j] as usize == k {
            let p = pl[j] as usize;
            let to_move = g.to_move();
            let mut opts = Vec::new();
            let mut pis = Vec::new();
            if kind[j] as u8 == KIND_LEAF {
                if to_move != Some(p) {
                    return Err(format!("record: sample {j} (step {k}) is seat {p}'s decision, the game has {to_move:?} to move"));
                }
                legal.clear();
                g.legal_actions(&mut legal);
                if legal.len() < 2 {
                    return Err(format!("record: sample {j} (step {k}): a decision of {} option(s) is never searched", legal.len()));
                }
                if pk + legal.len() > pi.len() {
                    return Err(format!("record: sample {j} (step {k}) has {} options, the policy {} values left", legal.len(), pi.len() - pk));
                }
                // (a value-only decision is recorded without its policy, docs/search.md 3.20 - unless a teacher names it)
                if g.value_only_decision() {
                    let mut t = Vec::new();
                    if !g.teacher_policy(&legal, &mut t) {
                        return Err(format!("record: sample {j} (step {k}): seat {p}'s decision is a value-only decision (recorded without a policy), the record has its policy"));
                    }
                }
                opts = legal.clone();
                pis = pi[pk..pk + legal.len()].iter().map(|&h| f16_to_f32(h as u16)).collect();
                pk += legal.len();
            } else if to_move == Some(p) {
                // the decider's own value-only row: only at a value-only decision it searched (docs/search.md 3.20)
                legal.clear();
                g.legal_actions(&mut legal);
                if !g.value_only_decision() || legal.len() < 2 {
                    return Err(format!("record: value-only sample {j} (step {k}) of seat {p}, the seat to move, at a decision of {} option(s) that is {}a value-only decision", legal.len(), if g.value_only_decision() { "" } else { "not " }));
                }
            } else if to_move.is_none() {
                return Err(format!("record: value-only sample {j} (step {k}) of seat {p}: the game has {to_move:?} to move"));
            }
            let mut x = vec![0.0f32; G::NUM_FEATURES];
            let (tail, ov) = record_row(&g, p, &opts, &mut x);
            if !G::TAILS.is_empty() && G::HEAD_TOKENS + tail.len() > G::MAX_ROW_TOKENS {
                return Err(format!("record: sample {j} (step {k}): a row of capacity {} > max_row_tokens {} (the driver aborts such a game: no samples)", G::HEAD_TOKENS + tail.len(), G::MAX_ROW_TOKENS));
            }
            let mut a = Vec::new();
            if aux {
                a.resize(G::aux_width(), 0.0);
                g.aux_target(p, &mut a);
            }
            let mut qv = [0.0f32; MAX_SEATS];
            for (c, &h) in q[j * n..(j + 1) * n].iter().enumerate() {
                qv[c] = f16_to_f32(h as u16);
            }
            records.push(Record { x, player: p as u8, n_players: n as u8, opts, pi: pis, z: [0.0; MAX_SEATS], q: qv, sc: [0.0; MAX_SEATS], net: net[j] as u32, aux: a, step: k as u32, tail, ov });
            j += 1;
        }
        if k == steps.len() {
            break;
        }
        chance.clear();
        match &steps[k] {
            Value::Object(m) => {
                // a chained contest's episode boundary (TraceStep::Next)
                let conceded = match m.get("next") {
                    Some(Value::Null) => None,
                    Some(v) if m.len() == 1 => Some(uint(v, "next", n as u64 - 1)? as usize),
                    _ => return Err(format!("record: step {k}: {} is not an episode boundary", Value::Object(m.clone()))),
                };
                if !chained || ep + 1 >= eps.len() {
                    return Err(format!("record: step {k}: an episode boundary beyond the record's {} episodes", eps.len()));
                }
                let e = &eps[ep];
                let want = match e.resigned {
                    Some(w) if n == 2 => Some(1 - w as usize),
                    Some(_) => return Err("record: a conceded episode of more than two seats".into()),
                    None if e.over => None,
                    None => return Err(format!("record: step {k}: the contest goes on after an episode without a result")),
                };
                if conceded != want {
                    return Err(format!("record: step {k}: episode {ep} conceded by {conceded:?}, its outcome says {want:?}"));
                }
                let (res, margins) = end_episode(&g, e, n, j - ep_first, true, false, &format!("episode {ep}"))?;
                label(&mut records[ep_first..j], &res, &margins, n);
                g = match g.next_episode(conceded).map_err(|e| format!("record: step {k}: next_episode: {e}"))? {
                    Next::Continue(ng) if ng.num_players() == n => ng,
                    Next::Continue(_) => return Err(format!("record: step {k}: next_episode changed the players")),
                    Next::Over(_) => return Err(format!("record: step {k}: the contest ended here, the record goes on")),
                };
                ep += 1;
                ep_first = j;
            }
            Value::String(payload) => {
                if g.to_move().is_none() {
                    return Err(format!("record: step {k} of {} after the game ended", steps.len()));
                }
                g.apply_external(payload, &mut chance).map_err(|e| format!("record: step {k}: external answer: {e}"))?
            }
            v => {
                let Some(p) = g.to_move() else {
                    return Err(format!("record: step {k} of {} after the game ended", steps.len()));
                };
                let a = uint(v, "step", u16::MAX as u64)? as u16;
                legal.clear();
                g.legal_actions(&mut legal);
                if !legal.contains(&a) {
                    return Err(format!("record: step {k}: action {a} is not one of seat {p}'s {} legal actions", legal.len()));
                }
                g.step(a, &mut chance).map_err(|e| format!("record: step {k}: action {a}: {e}"))?;
            }
        }
    }
    if j != ns {
        return Err(format!("record: {} samples after the last step", ns - j));
    }
    if pk != pi.len() {
        return Err(format!("record: {} policy values left over", pi.len() - pk));
    }
    let result = if chained {
        // the contest's last episode, then the contest's own end (driver::worker::finish_episode)
        if ep + 1 != eps.len() {
            return Err(format!("record: {} episodes replayed, the record lists {}", ep + 1, eps.len()));
        }
        let e = &eps[ep];
        let (res, margins) = end_episode(&g, e, n, j - ep_first, true, row_abort, &format!("episode {ep}"))?;
        label(&mut records[ep_first..j], &res, &margins, n);
        let loser = match e.resigned {
            Some(w) if n == 2 => Some(1 - w as usize),
            Some(_) => return Err("record: a conceded episode of more than two seats".into()),
            None => None,
        };
        let contest: Option<Vec<f32>> = if e.over {
            match g.next_episode(loser).map_err(|e| format!("record: the last episode: next_episode: {e}"))? {
                Next::Over(r) => Some(r[..n].to_vec()),
                Next::Continue(_) => return Err("record: the contest goes on after the record's last episode".into()),
            }
        } else {
            None
        };
        if top.over != contest.is_some() || top.resigned.is_some() || !top.margin.is_empty() {
            return Err(format!("record: the contest's over {} (resigned {:?}, margins {}), its replay {}", top.over, top.resigned, top.margin.len(), contest.is_some()));
        }
        let aborted = row_abort || (e.resigned.is_none() && g.to_move().is_none() && g.aborted());
        if top.aborted != aborted {
            return Err(format!("record: the contest's aborted {}, its replay {aborted}", top.aborted));
        }
        let r = contest.unwrap_or_else(|| vec![1.0 / n as f32; n]);
        if !same_bits(&r, &top.result) {
            return Err(format!("record: the contest's result {:?}, the replay's {r:?}", top.result));
        }
        if g.rounds() != top.rounds {
            return Err(format!("record: the contest's {} rounds, the replay's {}", top.rounds, g.rounds()));
        }
        if g.record_setup() != top.setup {
            return Err("record: the contest's setup differs from the record's".into());
        }
        r
    } else if let Some(b) = br.filter(|b| b.merged.is_some()) {
        // a merged near-tie branch (docs/search.md 2.11.1): ended at a decision of a game in progress, equivalent to one
        // of its game's after the branch point; its rows take its game's outcome
        legal.clear();
        g.legal_actions(&mut legal);
        if g.to_move().is_none() || legal.len() < 2 {
            return Err("record: a merged branch ends at a decision (a seat to move with two options or more)".into());
        }
        if !top.over || top.aborted || top.resigned.is_some() || row_abort {
            return Err(format!("record: a merged branch is over (its game's outcome), not aborted nor resigned (over {}, aborted {}, resigned {:?})", top.over, top.aborted, top.resigned));
        }
        if !same_bits(&top.result, &b.result) || !same_bits(&top.margin, &b.margin) {
            return Err(format!("record: a merged branch's result {:?} / margins {:?} are not its game's {:?} / {:?}", top.result, top.margin, b.result, b.margin));
        }
        if g.rounds() != top.rounds || g.record_setup() != top.setup {
            return Err("record: a merged branch's rounds or setup differ from its replay's".into());
        }
        if !g.merge_key().is_some_and(|k| b.keys.contains(&k)) {
            return Err("record: a merged branch's last state is not one of its game's after the branch point (merge_key)".into());
        }
        label(&mut records[..], &top.result, &top.margin, n);
        top.result.clone()
    } else {
        let (res, margins) = end_episode(&g, &top, n, j, false, row_abort, "the game")?;
        label(&mut records[..], &res, &margins, n);
        res
    };
    let branch = br.map(|b| (b.from as u32, b.merged));
    Ok(Replayed { gid, n, seed, over: top.over, aborted: top.aborted, resigned: top.resigned, abort_reason, result, rounds: top.rounds, steps: steps.len(), setup: top.setup, episodes: eps.len().max(1), records, branch })
}

/// The seed of game `gid` of a driver seeded `seed` (`SearchGame::new_game`'s, as the driver starts it:
/// `driver::worker::init_slot`; `paired`: the pair's stream, `DriverCfg::paired`). A record's `seed` must be its
/// game id's: a worker can't choose its games.
pub fn game_seed(seed: u64, gid: u32, paired: bool) -> u64 {
    let stream = if paired { gid / 2 } else { gid };
    crate::rng::Rng::new(crate::rng::split64(seed ^ crate::rng::split64(stream as u64 + 1))).next_u64()
}

/// `replay` from JSON text: the spec (`SearchGame::parse_spec`) and one game's record.
pub fn replay_json<G: SearchGame>(spec: &str, rec: &str, aux: bool) -> Result<Replayed, String> {
    let sp = G::parse_spec(spec)?;
    let v: Value = serde_json::from_str(rec).map_err(|e| format!("record: not JSON: {e}"))?;
    replay::<G>(&sp, &v, aux)
}

/// `replay_branch` from JSON text: the spec, a branch game's record and its game's record.
pub fn replay_branch_json<G: SearchGame>(spec: &str, rec: &str, parent: &str, aux: bool) -> Result<Replayed, String> {
    let sp = G::parse_spec(spec)?;
    let v: Value = serde_json::from_str(rec).map_err(|e| format!("record: not JSON: {e}"))?;
    let p: Value = serde_json::from_str(parent).map_err(|e| format!("record: the game's record is not JSON: {e}"))?;
    replay_branch::<G>(&sp, &v, &p, aux)
}

/// What `replay_failure` did when nothing panicked: the real moves replayed, whether the recorded `"move"` panic's
/// action was played, and the random rollouts' count and steps.
#[derive(Clone, Debug, PartialEq)]
pub struct FailureReplay {
    pub steps: usize,
    pub move_played: bool,
    pub rollouts: u32,
    pub rollout_steps: u64,
}

/// Replay a dropped game's failure (`driver::GameFailure::to_json`, docs/search.md 2.18) of game type `G` - the game
/// from its spec, players and seed, its real moves (`steps`; an episode boundary as `next_episode`), then where it
/// panicked: a `"move"` panic's action (`panic_move`) and the single-option decisions after it; then `rollouts`
/// random continuations of the reached state (rollout r: its own stream of `seed`; every second one determinized from
/// the decider's view first, as a search's world; each step encodes the decider's row and its option overrides as a
/// search leaf does; at most `max_steps` steps; through a chained contest's episodes). A panic is NOT caught: it
/// reaches the caller (the point - the engine module's binding turns it into Python's PanicException). Without one ->
/// what was replayed; an error when the failure cannot be replayed (an archive start or a branch game: their prefix is
/// not in the record; an illegal recorded move).
pub fn replay_failure<G: SearchGame>(spec: &G::Spec, f: &Value, rollouts: u32, seed: u64, max_steps: u32) -> Result<FailureReplay, String> {
    let o = f.as_object().ok_or("failure: not a JSON object")?;
    if !o.get("start").map_or(true, Value::is_null) || !o.get("branch").map_or(true, Value::is_null) {
        return Err("failure: an archive start's or a branch game's moves begin after a prefix the record does not hold".into());
    }
    let n = uint(field(o, "players")?, "players", MAX_SEATS as u64)? as usize;
    let game_seed = field(o, "game_seed")?.as_u64().ok_or("failure: game_seed is not a u64")?;
    let steps = field(o, "steps")?.as_array().ok_or("failure: steps is not a list")?;
    let mut g = G::new_game(spec, n, game_seed).map_err(|e| format!("failure: new_game: {e}"))?.clone();
    let mut legal: Vec<u16> = Vec::new();
    let mut chance: Vec<u64> = Vec::new();
    for (k, s) in steps.iter().enumerate() {
        chance.clear();
        match s {
            Value::Object(m) => {
                let conceded = match m.get("next") {
                    Some(Value::Null) => None,
                    Some(v) => Some(uint(v, "next", n as u64 - 1)? as usize),
                    None => return Err(format!("failure: step {k}: {s} is not an episode boundary")),
                };
                g = match g.next_episode(conceded).map_err(|e| format!("failure: step {k}: next_episode: {e}"))? {
                    Next::Continue(ng) => ng,
                    Next::Over(_) => return Err(format!("failure: step {k}: the contest ended here, the record goes on")),
                };
            }
            Value::String(p) => g.apply_external(p, &mut chance).map_err(|e| format!("failure: step {k}: external answer: {e}"))?,
            v => {
                let a = uint(v, "step", u16::MAX as u64)? as u16;
                legal.clear();
                g.legal_actions(&mut legal);
                if !legal.contains(&a) {
                    return Err(format!("failure: step {k}: action {a} is not legal ({} legal actions)", legal.len()));
                }
                g.step(a, &mut chance).map_err(|e| format!("failure: step {k}: action {a}: {e}"))?;
            }
        }
    }
    let mut move_played = false;
    if o.get("phase").and_then(Value::as_str) == Some("move") {
        if let Some(a) = o.get("panic_move").and_then(Value::as_u64) {
            let a = a as u16;
            legal.clear();
            g.legal_actions(&mut legal);
            if !legal.contains(&a) {
                return Err(format!("failure: the panic's move {a} is not legal after the recorded moves"));
            }
            chance.clear();
            g.step(a, &mut chance).map_err(|e| format!("failure: the panic's move {a}: {e}"))?;
            move_played = true;
        }
        // (the single-option decisions the driver steps itself after a real move: tree::transition_each)
        while g.to_move().is_some() {
            legal.clear();
            g.legal_actions(&mut legal);
            if legal.len() != 1 {
                break;
            }
            chance.clear();
            g.step(legal[0], &mut chance).map_err(|e| format!("failure: a single-option step after the move: {e}"))?;
        }
    }
    let mut x = vec![0.0f32; G::NUM_FEATURES];
    let mut total = 0u64;
    for r in 0..rollouts {
        let mut rng = crate::rng::Rng::new(crate::rng::split64(seed ^ crate::rng::split64(r as u64 + 1)));
        let mut w = g.clone();
        if r % 2 == 1 {
            if let Some(p) = w.to_move() {
                w.determinize(p, &mut rng);
            }
        }
        for _ in 0..max_steps {
            let p = match w.to_move() {
                Some(p) => p,
                None if w.chained() => match w.next_episode(None) {
                    Ok(Next::Continue(ng)) => {
                        w = ng;
                        continue;
                    }
                    _ => break,
                },
                None => break,
            };
            legal.clear();
            w.legal_actions(&mut legal);
            if legal.is_empty() {
                break;
            }
            let _ = record_row(&w, p, &legal, &mut x);
            let a = legal[rng.below(legal.len() as u32) as usize];
            chance.clear();
            if w.step(a, &mut chance).is_err() {
                break;
            }
            total += 1;
        }
    }
    Ok(FailureReplay { steps: steps.len(), move_played, rollouts, rollout_steps: total })
}

/// `replay_failure` from JSON text: the failure's own spec (`spec`, `SearchGame::parse_spec`) -> its result as JSON
/// (`{"steps", "move_played", "rollouts", "rollout_steps"}`).
pub fn replay_failure_json<G: SearchGame>(failure: &str, rollouts: u32, seed: u64, max_steps: u32) -> Result<String, String> {
    let v: Value = serde_json::from_str(failure).map_err(|e| format!("failure: not JSON: {e}"))?;
    let spec = v.get("spec").and_then(Value::as_str).ok_or("failure: no spec")?;
    let sp = G::parse_spec(spec)?;
    let r = replay_failure::<G>(&sp, &v, rollouts, seed, max_steps)?;
    Ok(json!({"steps": r.steps, "move_played": r.move_played, "rollouts": r.rollouts, "rollout_steps": r.rollout_steps}).to_string())
}
