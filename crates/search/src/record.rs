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
use serde_json::{json, Map, Value};

use crate::driver::{f16_to_f32, f32_to_f16, GameResult, Record};
use crate::game::{Next, SearchGame, MAX_SEATS};

/// The record format's version (`records/1`).
pub const RECORDS_VERSION: u32 = 1;

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

/// The record of finished game `row` (its trace on: `Driver::set_trace`) and its emitted samples (`row.samples`
/// of them, in order, as `take_samples` gave them) - one JSON object (module docs).
pub fn game_json(row: &GameResult, samples: &[Record]) -> Result<Value, String> {
    let t = row.trace.as_ref().ok_or_else(|| format!("game {}: no trace (Driver::set_trace)", row.gid))?;
    if samples.len() != row.samples as usize {
        return Err(format!("game {}: {} samples given, its row says {}", row.gid, samples.len(), row.samples));
    }
    let n = row.n;
    let steps: Vec<Value> = t
        .steps
        .iter()
        .map(|s| match s {
            TraceStep::Act(a) => json!(a),
            TraceStep::Ext(p) => json!(p),
            TraceStep::Next(c) => json!({"next": c}),
        })
        .collect();
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
    o.insert("result".into(), Value::Array(bits32(&row.result)));
    o.insert("margin".into(), Value::Array(bits32(&t.margins)));
    o.insert("rounds".into(), json!(row.rounds));
    if !t.setup.is_empty() {
        o.insert("setup".into(), json!(t.setup));
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
    pub result: Vec<f32>,
    pub rounds: u32,
    /// Real moves replayed (a contest's episode boundaries included).
    pub steps: usize,
    pub setup: String,
    /// A chained contest's episodes (1 for any other game).
    pub episodes: usize,
    pub records: Vec<Record>,
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
fn end_episode<G: SearchGame>(g: &G, e: &Ep, n: usize, samples: usize, chained: bool, what: &str) -> Result<(Vec<f32>, Vec<f32>), String> {
    if e.resigned.is_some() && (!e.over || e.aborted) {
        return Err(format!("record: {what}: a resigned episode is over and not aborted"));
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
            if e.aborted != (ended && g.aborted()) {
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
    let gid = uint(field(o, "gid")?, "gid", u32::MAX as u64)? as u32;
    let n = uint(field(o, "n")?, "n", MAX_SEATS as u64)? as usize;
    if n < G::MIN_PLAYERS || n > G::MAX_PLAYERS.min(MAX_SEATS) {
        return Err(format!("record: {n} players ({} plays {}..={})", G::NAME, G::MIN_PLAYERS, G::MAX_PLAYERS));
    }
    let seed = field(o, "seed")?.as_u64().ok_or("record: seed is not a u64")?;
    let top = ep_of(o, n)?;
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
                opts = legal.clone();
                pis = pi[pk..pk + legal.len()].iter().map(|&h| f16_to_f32(h as u16)).collect();
                pk += legal.len();
            } else if to_move.is_none() || to_move == Some(p) {
                return Err(format!("record: value-only sample {j} (step {k}) of seat {p}: the game has {to_move:?} to move"));
            }
            let mut x = vec![0.0f32; G::NUM_FEATURES];
            g.encode(p, &mut x);
            let mut a = Vec::new();
            if aux {
                a.resize(G::aux_width(), 0.0);
                g.aux_target(p, &mut a);
            }
            let mut qv = [0.0f32; MAX_SEATS];
            for (c, &h) in q[j * n..(j + 1) * n].iter().enumerate() {
                qv[c] = f16_to_f32(h as u16);
            }
            records.push(Record { x, player: p as u8, n_players: n as u8, opts, pi: pis, z: [0.0; MAX_SEATS], q: qv, sc: [0.0; MAX_SEATS], net: net[j] as u32, aux: a, step: k as u32 });
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
                let (res, margins) = end_episode(&g, e, n, j - ep_first, true, &format!("episode {ep}"))?;
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
        let (res, margins) = end_episode(&g, e, n, j - ep_first, true, &format!("episode {ep}"))?;
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
        let aborted = e.resigned.is_none() && g.to_move().is_none() && g.aborted();
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
    } else {
        let (res, margins) = end_episode(&g, &top, n, j, false, "the game")?;
        label(&mut records[..], &res, &margins, n);
        res
    };
    Ok(Replayed { gid, n, seed, over: top.over, aborted: top.aborted, resigned: top.resigned, result, rounds: top.rounds, steps: steps.len(), setup: top.setup, episodes: eps.len().max(1), records })
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
