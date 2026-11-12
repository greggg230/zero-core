//! The driver's settings from JSON (docs/search.md 4.1): seat settings -> `SeatCfg`, `branch` ->
//! `BranchCfg`, `starts` -> `StartsCfg`, and a whole `DriverCfg` with its de-duplicated seat table.
//! The one parser of these settings: the Python bindings (`zero_py`'s `SearchDriver`, which hands
//! its dicts over as JSON values) and native workers (zero-farm's `zero_worker`, reading a run's
//! settings) both call it, so a seat setting reaches every driver in one change.
//!
//! The rules are the bindings' since the beginning (moved here unchanged, zero-farm phase 0b,
//! docs/az.md 42), in JSON terms:
//! * a missing setting or `null` is its default;
//! * an integer setting takes a JSON integer in range or a boolean (Python's `int` accepts `True`), a
//!   float setting any number or a boolean, a text setting a string; anything else is an error;
//! * a flag (`selfplay`, `redet`, `record`, `belief_fast`, `infoset_merge`, branch's `differ`) is
//!   true when its value is truthy as in Python (a non-zero number, a non-empty string / list /
//!   object, `true`);
//! * an unknown setting is an error - a network seat's (`SEAT_KEYS`), a bot or external seat's (only
//!   `bot` / `external`), branch's, starts'.
use serde_json::{Map, Value};

use crate::driver::{BranchCfg, DriverCfg, KeyKind, LeafView, NnSeat, PolicyTarget, RootSearch, SeatCfg, Select, StartsCfg, ViewTarget, BELIEF_CLIP};
use crate::lookahead::LookCfg;

/// The settings of a network seat (each `NnSeat` field's key; docs/search.md 3).
pub const SEAT_KEYS: &[&str] = &["key", "sims", "c_puct", "selfplay", "alpha", "eps", "temp_moves", "sims_fast", "sims_minor", "p_full", "chance_k", "margin_w", "redet", "policy_target", "cq_c_visit", "cq_c_scale", "select", "select_min_visits", "select_c_visit", "select_c_scale", "record", "root_min_visits", "root_min_visits_max_actions", "leaf_value_view", "view_samples", "view_sample_target", "belief_tilt", "belief_clip", "belief_mix", "belief_fast", "belief_tilts", "root_search", "gumbel_m", "gumbel_scale", "gumbel_c_visit", "gumbel_c_scale", "lookahead", "eval_cache", "resign_t", "resign_k", "resign_noresign", "cq_prior_w", "infoset_merge", "value_choice_worlds", "value_choice_temp"];

/// The settings of a branch (docs/search.md 2.11) and of archive starts.
pub const BRANCH_KEYS: &[&str] = &["p", "moves", "temp", "differ"];
pub const STARTS_KEYS: &[&str] = &["p", "specs"];

/// The value of `k` in `d`: None when missing or null (the setting's default).
fn opt<'a>(d: &'a Map<String, Value>, k: &str) -> Option<&'a Value> {
    d.get(k).filter(|v| !v.is_null())
}

fn kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "a list",
        Value::Object(_) => "an object",
    }
}

/// An integer setting (u64 range first; a boolean is 0 / 1, as Python's int).
fn int_of(k: &str, v: &Value) -> Result<i128, String> {
    match v {
        Value::Bool(b) => Ok(*b as i128),
        Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                Ok(u as i128)
            } else if let Some(i) = n.as_i64() {
                Ok(i as i128)
            } else {
                Err(format!("{k}: an integer, not {n}"))
            }
        }
        _ => Err(format!("{k}: an integer, not {} ({v})", kind(v))),
    }
}

fn u32_of(k: &str, v: &Value) -> Result<u32, String> {
    let i = int_of(k, v)?;
    u32::try_from(i).map_err(|_| format!("{k} {i}: out of range (0..={})", u32::MAX))
}

fn f64_of(k: &str, v: &Value) -> Result<f64, String> {
    match v {
        Value::Bool(b) => Ok(*b as u8 as f64),
        Value::Number(n) => n.as_f64().ok_or_else(|| format!("{k}: a number, not {n}")),
        _ => Err(format!("{k}: a number, not {} ({v})", kind(v))),
    }
}

fn str_of<'a>(k: &str, v: &'a Value) -> Result<&'a str, String> {
    v.as_str().ok_or_else(|| format!("{k}: a string, not {} ({v})", kind(v)))
}

/// Python's truthiness of a JSON value (null is never asked: it is the default).
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|x| x != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn get_u32(d: &Map<String, Value>, k: &str, default: u32) -> Result<u32, String> {
    opt(d, k).map_or(Ok(default), |v| u32_of(k, v))
}

fn get_f64(d: &Map<String, Value>, k: &str, default: f64) -> Result<f64, String> {
    opt(d, k).map_or(Ok(default), |v| f64_of(k, v))
}

fn get_str<'a>(d: &'a Map<String, Value>, k: &str, default: &'a str) -> Result<&'a str, String> {
    opt(d, k).map_or(Ok(default), |v| str_of(k, v))
}

fn flag(d: &Map<String, Value>, k: &str, default: bool) -> bool {
    opt(d, k).map_or(default, truthy)
}

fn object<'a>(what: &str, v: &'a Value) -> Result<&'a Map<String, Value>, String> {
    v.as_object().ok_or_else(|| format!("{what}: an object of settings, not {} ({v})", kind(v)))
}

/// A spec-like value as the JSON text the game parses: a string as it is, null "{}" (the game's
/// default), anything else serialised (`serde_json`: keys in sorted order - a caller that must
/// reproduce another serialisation, as zero_py Python's `json.dumps`, hands over the string).
pub fn spec_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => "{}".into(),
        other => other.to_string(),
    }
}

/// The evaluator key `name` of kind `kind` -> its index in `keys` (added at the end when new); one
/// name for both a network and an external seat is an error.
pub fn key_index(keys: &mut Vec<(String, KeyKind)>, name: &str, kind: KeyKind) -> Result<u16, String> {
    match keys.iter().position(|k| k.0 == name) {
        Some(i) if keys[i].1 == kind => Ok(i as u16),
        Some(_) => Err(format!("evaluator key {name:?} used for both a network and an external seat")),
        None => {
            keys.push((name.to_string(), kind));
            Ok((keys.len() - 1) as u16)
        }
    }
}

/// `branch` {p, moves, temp, differ} (docs/search.md 2.11; another key is an error).
pub fn parse_branch(v: &Value) -> Result<BranchCfg, String> {
    let d = object("branch", v)?;
    for k in d.keys() {
        if !BRANCH_KEYS.contains(&k.as_str()) {
            return Err(format!("branch: unknown key {k:?} (p, moves, temp, differ)"));
        }
    }
    let p = f64_of("branch p", d.get("p").ok_or("branch: p (the share of games that branch) is required")?)?;
    let b = BranchCfg { p, moves: get_u32(d, "moves", 3)?, temp: get_f64(d, "temp", 2.0)?, differ: flag(d, "differ", false) };
    b.check()?;
    Ok(b)
}

/// `starts` {p, specs: [[id, spec JSON or object], ...]} (docs/search.md 2.11).
pub fn parse_starts(v: &Value) -> Result<StartsCfg, String> {
    let d = object("starts", v)?;
    for k in d.keys() {
        if !STARTS_KEYS.contains(&k.as_str()) {
            return Err(format!("starts: unknown key {k:?} (p, specs)"));
        }
    }
    let p = f64_of("starts p", d.get("p").ok_or("starts: p (the share of games started from the archive) is required")?)?;
    let list = d.get("specs").ok_or("starts: specs ([(id, spec), ...]) is required")?;
    let list = list.as_array().ok_or_else(|| format!("starts: specs is a list of (id, spec), not {}", kind(list)))?;
    let mut specs = Vec::with_capacity(list.len());
    for e in list {
        match e.as_array().map(|a| a.as_slice()) {
            Some([Value::String(id), sp]) => specs.push((id.clone(), spec_text(sp))),
            _ => return Err("starts: each entry is (id, spec)".into()),
        }
    }
    let c = StartsCfg { p, specs };
    c.check()?;
    Ok(c)
}

/// A seat: {'bot': name} | {'external': key} | a network seat's settings (`SEAT_KEYS`, defaults =
/// `NnSeat::eval`'s; docs/search.md 3). Evaluator keys are added to `keys`.
pub fn parse_seat(v: &Value, keys: &mut Vec<(String, KeyKind)>) -> Result<SeatCfg, String> {
    let d = object("seat", v)?;
    for (only, what) in [("bot", "a bot seat"), ("external", "an external seat")] {
        if let Some(x) = d.get(only) {
            if let Some(k) = d.keys().find(|k| k.as_str() != only) {
                return Err(format!("seat: unknown setting {k:?} of {what} ({only:?} only)"));
            }
            let name = str_of(only, x)?;
            return Ok(if only == "bot" { SeatCfg::Bot(name.to_string()) } else { SeatCfg::External(key_index(keys, name, KeyKind::External)?) });
        }
    }
    for k in d.keys() {
        if !SEAT_KEYS.contains(&k.as_str()) {
            return Err(format!("seat: unknown setting {k:?} (bot, external, {SEAT_KEYS:?})"));
        }
    }
    let key = key_index(keys, get_str(d, "key", "m")?, KeyKind::Nn)?;
    let sims_fast = opt(d, "sims_fast").map(|v| u32_of("sims_fast", v)).transpose()?;
    let sims_minor = opt(d, "sims_minor").map(|v| u32_of("sims_minor", v)).transpose()?;
    // training target of recorded decisions: 'visits' (default) | 'cq' (docs/search.md 3.1)
    let (cq_c_visit, cq_c_scale) = (get_f64(d, "cq_c_visit", 50.0)?, get_f64(d, "cq_c_scale", 1.0)?);
    let policy_target = PolicyTarget::parse(get_str(d, "policy_target", "visits")?, cq_c_visit, cq_c_scale)?;
    // the prior's exponent beta in the 'cq' target (docs/search.md 3.13; absent: 1.0, the target of before)
    let policy_target = policy_target.with_prior_w(get_f64(d, "cq_prior_w", 1.0)?)?;
    let select = get_str(d, "select", "visits")?;
    for k in ["select_c_visit", "select_c_scale"] {
        if select == "visits" && opt(d, k).is_some() {
            return Err(format!("{k}: a setting of select 'cq' ('visits' has no completed-Q transform)"));
        }
    }
    // the root search: 'puct' (default) | 'gumbel' with its settings (docs/search.md 3.7; sigma's
    // transform absent: the seat's cq_c_visit, cq_c_scale)
    let root_search = get_str(d, "root_search", "puct")?;
    for k in ["gumbel_m", "gumbel_scale", "gumbel_c_visit", "gumbel_c_scale"] {
        if root_search != "gumbel" && opt(d, k).is_some() {
            return Err(format!("{k}: a setting of root_search 'gumbel'"));
        }
    }
    let root_search = RootSearch::parse(root_search, get_u32(d, "gumbel_m", 16)?, get_f64(d, "gumbel_scale", 1.0)?, get_f64(d, "gumbel_c_visit", cq_c_visit)?, get_f64(d, "gumbel_c_scale", cq_c_scale)?)?;
    // the exhaustive lookahead (docs/search.md 3.9): an object of its settings (or their JSON), null = off
    let look = match opt(d, "lookahead") {
        Some(v) => Some(Box::new(LookCfg::parse(&spec_text(v))?)),
        None => None,
    };
    // per-target tilts of the root's belief ({target: alpha}, docs/search.md 3.8.1): sorted by name
    let belief_tilts = match opt(d, "belief_tilts") {
        Some(v) => {
            let o = v.as_object().ok_or("belief_tilts: a dict {target: alpha}")?;
            let mut t = Vec::with_capacity(o.len());
            for (name, a) in o {
                t.push((name.clone(), f64_of("belief_tilts: a dict {target: alpha}", a).map_err(|_| "belief_tilts: a dict {target: alpha}".to_string())?));
            }
            t.sort_by(|a, b| a.0.cmp(&b.0));
            t
        }
        None => Vec::new(),
    };
    Ok(SeatCfg::Nn(NnSeat {
        key,
        sims: get_u32(d, "sims", 100)?,
        c_puct: get_f64(d, "c_puct", 1.0)?,
        selfplay: flag(d, "selfplay", false),
        alpha: get_f64(d, "alpha", 0.3)?,
        eps: get_f64(d, "eps", 0.25)?,
        temp_moves: get_u32(d, "temp_moves", 30)?,
        sims_fast,
        p_full: get_f64(d, "p_full", 1.0)?,
        sims_minor,
        chance_k: get_u32(d, "chance_k", 0)?,
        margin_w: get_f64(d, "margin_w", 0.0)?,
        redet: flag(d, "redet", true),
        policy_target,
        // the rule of the moves it does not sample: 'visits' (default) | 'cq' with the guard
        // select_min_visits and the completed-Q transform select_c_visit, select_c_scale (absent:
        // the target's cq_c_visit, cq_c_scale; docs/search.md 3.3)
        select: Select::parse(select, get_u32(d, "select_min_visits", 0)?, get_f64(d, "select_c_visit", cq_c_visit)?, get_f64(d, "select_c_scale", cq_c_scale)?)?,
        // whether its full searches are recorded (absent: true; false: a league opponent, a past
        // checkpoint whose moves are not the current policy's, docs/az.md 14)
        record: flag(d, "record", true),
        // root minimum visits of its full searches (0 = off) on roots of up to
        // root_min_visits_max_actions options (docs/search.md 3.5)
        root_min_visits: get_u32(d, "root_min_visits", 0)?,
        root_min_visits_max_actions: get_u32(d, "root_min_visits_max_actions", 4)?,
        // whose view a leaf's value comes from: 'mover' (default) | 'root' (docs/search.md 3.6)
        leaf_value_view: LeafView::parse(get_str(d, "leaf_value_view", "mover")?)?,
        // value-only samples: a share of the recorded decisions (0 = off) and their q target
        view_samples: get_f64(d, "view_samples", 0.0)?,
        view_target: ViewTarget::parse(get_str(d, "view_sample_target", "next_q")?)?,
        // the root's belief (docs/search.md 3.8): its tilt (0 = off), clip, uniform share, and
        // whether fast and minor searches take it too
        belief_tilt: get_f64(d, "belief_tilt", 0.0)?,
        belief_clip: get_f64(d, "belief_clip", BELIEF_CLIP)?,
        belief_mix: get_f64(d, "belief_mix", 0.0)?,
        belief_fast: flag(d, "belief_fast", false),
        belief_tilts,
        root_search,
        look,
        // the game's evaluation cache: at most this many answers per game (0 = off; docs/search.md 3.12)
        eval_cache: get_u32(d, "eval_cache", 0)?,
        // self-play resignation (docs/search.md 3.14): the threshold t (0 = off), the consecutive decided
        // full searches k, the share of games that never resign (checked by the driver)
        resign_t: get_f64(d, "resign_t", 0.0)?,
        resign_k: get_u32(d, "resign_k", 5)?,
        resign_noresign: get_f64(d, "resign_noresign", 0.1)?,
        // information-set merging (docs/search.md 3.16; absent: false, the tree of before; checked by the driver:
        // root-view leaf values, an evaluation seat)
        infoset_merge: flag(d, "infoset_merge", false),
        // value choices (docs/search.md 3.17; absent: 0 worlds, off): the worlds that value each option of a
        // decision the game marks, and the temperature of the choice (0: the best mean value)
        value_choice_worlds: get_u32(d, "value_choice_worlds", 0)?,
        value_choice_temp: get_f64(d, "value_choice_temp", 0.0)?,
    }))
}

/// The driver arguments besides its seats (`zero_py`'s `SearchDriver(...)` keywords, their defaults).
#[derive(Clone, Debug)]
pub struct DriverArgs {
    /// The game's spec as JSON text ("{}": the game's default).
    pub spec: String,
    pub concurrency: usize,
    pub seed: u64,
    pub record: bool,
    pub threads: usize,
    pub groups: usize,
    pub max_steps: u64,
    pub rolling: bool,
    pub net: u32,
    pub queue_only: bool,
    pub move_log: u32,
    pub aux: bool,
    pub paired: bool,
    pub sparse: bool,
}

impl Default for DriverArgs {
    fn default() -> DriverArgs {
        DriverArgs { spec: "{}".into(), concurrency: 64, seed: 0, record: false, threads: 4, groups: 2, max_steps: 100_000, rolling: false, net: 0, queue_only: false, move_log: 0, aux: false, paired: false, sparse: false }
    }
}

/// A whole driver's settings: `seats` = per game its seats' settings (`parse_seat`), each distinct
/// seat once in the table (`DriverCfg::seats`, `games` indexing it), then `branch` and `starts`.
pub fn driver_cfg(seats: &[Vec<Value>], branch: Option<&Value>, starts: Option<&Value>, a: DriverArgs) -> Result<DriverCfg, String> {
    let mut keys = Vec::new();
    let mut table: Vec<SeatCfg> = Vec::new();
    let mut games = Vec::with_capacity(seats.len());
    for row in seats {
        let mut ix = Vec::with_capacity(row.len());
        for s in row {
            let sc = parse_seat(s, &mut keys)?;
            let i = match table.iter().position(|x| *x == sc) {
                Some(i) => i,
                None => {
                    table.push(sc);
                    table.len() - 1
                }
            };
            ix.push(u16::try_from(i).map_err(|_| "more than 65,536 distinct seats".to_string())?);
        }
        games.push(ix);
    }
    let branch = branch.map(parse_branch).transpose()?;
    let starts = starts.map(parse_starts).transpose()?;
    let DriverArgs { spec, concurrency, seed, record, threads, groups, max_steps, rolling, net, queue_only, move_log, aux, paired, sparse } = a;
    Ok(DriverCfg { spec, seats: table, games, keys, concurrency, seed, record, threads, groups, max_steps, rolling, net, queue_only, move_log, aux, paired, sparse, branch, starts })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn nn(v: Value) -> Result<NnSeat, String> {
        match parse_seat(&v, &mut Vec::new())? {
            SeatCfg::Nn(n) => Ok(n),
            s => Err(format!("not a network seat: {s:?}")),
        }
    }

    #[test]
    fn defaults_are_the_eval_seat() {
        assert_eq!(nn(json!({})).unwrap(), NnSeat::eval(0, 100));
        assert_eq!(nn(json!({"sims": null, "redet": null})).unwrap(), NnSeat::eval(0, 100));
    }

    #[test]
    fn python_value_rules() {
        // a boolean is an integer / a float; a float is not an integer; flags are truthy
        let s = nn(json!({"sims": true, "c_puct": false, "selfplay": "yes", "redet": 0, "record": [], "alpha": 1})).unwrap();
        assert_eq!((s.sims, s.c_puct, s.selfplay, s.redet, s.record, s.alpha), (1, 0.0, true, false, false, 1.0));
        assert!(nn(json!({"sims": 64.0})).is_err());
        assert!(nn(json!({"sims": -1})).is_err());
        assert!(nn(json!({"sims": 4294967296u64})).is_err());
        assert!(nn(json!({"select": 1})).is_err());
    }

    #[test]
    fn unknown_settings_are_errors() {
        assert!(nn(json!({"sim": 8})).unwrap_err().contains("unknown setting \"sim\""));
        assert!(parse_seat(&json!({"bot": "random", "sims": 8}), &mut Vec::new()).is_err());
        assert!(parse_seat(&json!({"external": "x", "bot": "random"}), &mut Vec::new()).is_err());
        assert!(parse_branch(&json!({"p": 0.5, "moves": 2, "x": 1})).is_err());
        assert!(parse_starts(&json!({"p": 0.5, "specs": [["a", "{}"]], "q": 1})).is_err());
        assert!(parse_seat(&json!("nn"), &mut Vec::new()).is_err());
    }

    #[test]
    fn keys_and_table() {
        let m = json!({"sims": 8, "selfplay": true});
        let o = json!({"key": "o", "sims": 8});
        let cfg = driver_cfg(&[vec![m.clone(), o.clone()], vec![o.clone(), m.clone()], vec![json!({"bot": "random"}), json!({"external": "js"})]], None, None, DriverArgs::default()).unwrap();
        assert_eq!(cfg.games, vec![vec![0, 1], vec![1, 0], vec![2, 3]]);
        assert_eq!(cfg.keys, vec![("m".to_string(), KeyKind::Nn), ("o".into(), KeyKind::Nn), ("js".into(), KeyKind::External)]);
        assert!(driver_cfg(&[vec![json!({"key": "js"}), json!({"external": "js"})]], None, None, DriverArgs::default()).is_err());
    }

    #[test]
    fn branch_starts_and_subsettings() {
        let b = parse_branch(&json!({"p": 0.25, "differ": 1})).unwrap();
        assert_eq!(b, BranchCfg { p: 0.25, moves: 3, temp: 2.0, differ: true });
        assert!(parse_branch(&json!({"moves": 2})).is_err());
        let s = parse_starts(&json!({"p": 1, "specs": [["a", "{\"x\": 1}"], ["b", {"y": 2}]]})).unwrap();
        assert_eq!(s.specs, vec![("a".to_string(), "{\"x\": 1}".to_string()), ("b".into(), "{\"y\":2}".into())]);
        let t = nn(json!({"belief_tilts": {"deck": 1, "belief": 0.5}, "select": "cq", "select_c_scale": 0.3, "cq_c_scale": 2})).unwrap();
        assert_eq!(t.belief_tilts, vec![("belief".to_string(), 0.5), ("deck".into(), 1.0)]);
        assert_eq!(t.select, Select::Cq { c_visit: 50.0, c_scale: 0.3, min_visits: 0 });
        assert!(nn(json!({"select_c_visit": 3})).is_err());
        assert!(nn(json!({"gumbel_m": 3})).is_err());
        let g = nn(json!({"root_search": "gumbel", "cq_c_visit": 10})).unwrap();
        assert_eq!(g.root_search, RootSearch::Gumbel { m: 16, scale: 1.0, c_visit: 10.0, c_scale: 1.0 });
    }
}
