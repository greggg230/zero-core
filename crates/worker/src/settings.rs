//! A run's settings -> the self-play driver a worker plays: the Rust port of what a Python remote
//! self-play worker builds from its mirror's `config.json` (zero-core `remoteplay.Worker._load`:
//! `Run.nn_seat('m', sims, selfplay=True)` = `games.nn_seat` with `run.belief_keys`,
//! `resign.seat_keys`, `run.value_choice_keys` and the `mcts` checks; `Run.driver(table, seed,
//! record=True, rolling=True, net, queue_only=True)`), then zero_search's shared parser
//! (`settings::driver_cfg`) - so the seat dict is the Python one, key for key and type for type, and
//! its parsing is the same code. tools/driver_settings_dump.py `--compare-worker` checks every
//! preset and run config against the Python path (docs/az.md 42).
//!
//! Python's value rules are kept where they decide anything: `c[k]` (a missing key is an error;
//! null passes through as None), `c.get(k, d)` (only a missing key takes d), `x or d` (a falsy x
//! takes d), `float()` / `int()` (an int truncates a float; a boolean is 0 / 1), truthiness. One
//! deliberate difference: a string where `float()` / `int()` would parse a number is refused
//! (never seen in a run's settings; the worker fails loudly rather than guess).
use serde_json::{Map, Value};
use zero_search::driver::DriverCfg;
use zero_search::dynapi::GameInfo;
use zero_search::settings::{driver_cfg, DriverArgs};

use crate::pyjson::PyJson;

pub const BELIEF_CLIP: f64 = 20.0;
pub const VALUE_CHOICE_MAX_WORLDS: f64 = 1024.0;

/// A run's settings (`config.json`): a JSON object, keys in file order.
#[derive(Clone, Debug)]
pub struct Settings {
    pub raw: PyJson,
}

fn kind(v: &PyJson) -> &'static str {
    match v {
        PyJson::Null => "None",
        PyJson::Bool(_) => "a bool",
        PyJson::Int(_) => "an int",
        PyJson::Float(_) => "a float",
        PyJson::Str(_) => "a str",
        PyJson::List(_) => "a list",
        PyJson::Dict(_) => "a dict",
    }
}

pub fn truthy(v: &PyJson) -> bool {
    match v {
        PyJson::Null => false,
        PyJson::Bool(b) => *b,
        PyJson::Int(i) => *i != 0,
        PyJson::Float(f) => *f != 0.0,
        PyJson::Str(s) => !s.is_empty(),
        PyJson::List(l) => !l.is_empty(),
        PyJson::Dict(d) => !d.is_empty(),
    }
}

/// Python's `float(v)` (a string refused: see the module docs).
pub fn py_float(what: &str, v: &PyJson) -> Result<f64, String> {
    match v {
        PyJson::Bool(b) => Ok(*b as u8 as f64),
        PyJson::Int(i) => Ok(*i as f64),
        PyJson::Float(f) => Ok(*f),
        _ => Err(format!("{what} {}: {} where a number is needed", v.dumps(), kind(v))),
    }
}

/// Python's `int(v)` (a float truncated toward 0; a string refused).
pub fn py_int(what: &str, v: &PyJson) -> Result<i128, String> {
    match v {
        PyJson::Bool(b) => Ok(*b as i128),
        PyJson::Int(i) => Ok(*i),
        PyJson::Float(f) if f.is_finite() => Ok(f.trunc() as i128),
        _ => Err(format!("{what} {}: {} where an integer is needed", v.dumps(), kind(v))),
    }
}

/// `isinstance(v, (int, float)) and not isinstance(v, bool)`.
fn is_num(v: &PyJson) -> bool {
    matches!(v, PyJson::Int(_) | PyJson::Float(_))
}

/// `isinstance(v, int) and not isinstance(v, bool)`.
fn is_int(v: &PyJson) -> bool {
    matches!(v, PyJson::Int(_))
}

fn num_val(v: &PyJson) -> f64 {
    match v {
        PyJson::Int(i) => *i as f64,
        PyJson::Float(f) => *f,
        PyJson::Bool(b) => *b as u8 as f64,
        _ => f64::NAN,
    }
}

/// Python's `v == s` for a string s.
fn is_str(v: &PyJson, s: &str) -> bool {
    matches!(v, PyJson::Str(t) if t == s)
}

/// Python's `v == x` for a number x (a bool is 0 / 1; anything else unequal).
fn eq_num(v: &PyJson, x: f64) -> bool {
    matches!(v, PyJson::Int(_) | PyJson::Float(_) | PyJson::Bool(_)) && num_val(v) == x
}

impl Settings {
    pub fn parse(text: &str) -> Result<Settings, String> {
        let raw = PyJson::parse(text)?;
        if !matches!(raw, PyJson::Dict(_)) {
            return Err("settings: a JSON object".into());
        }
        Ok(Settings { raw })
    }

    pub fn from_value(v: &Value) -> Result<Settings, String> {
        Settings::parse(&v.to_string())
    }

    pub fn keys(&self) -> Vec<&str> {
        match &self.raw {
            PyJson::Dict(v) => v.iter().map(|e| e.0.as_str()).collect(),
            _ => Vec::new(),
        }
    }

    /// `c.get(k)`: None when missing (a null value is Some(Null)).
    pub fn get(&self, k: &str) -> Option<&PyJson> {
        self.raw.get(k)
    }

    /// `c[k]`.
    pub fn req(&self, k: &str) -> Result<&PyJson, String> {
        self.get(k).ok_or_else(|| format!("settings: {k:?} is missing (KeyError)"))
    }

    /// `c.get(k, d)`.
    fn get_or<'a>(&'a self, k: &str, d: &'a PyJson) -> &'a PyJson {
        self.get(k).unwrap_or(d)
    }

    /// `c.get(k, d) or d`.
    fn get_or_falsy<'a>(&'a self, k: &str, d: &'a PyJson) -> &'a PyJson {
        match self.get(k) {
            Some(v) if truthy(v) => v,
            _ => d,
        }
    }

    /// Set (or replace) a setting.
    pub fn set(&mut self, k: &str, v: PyJson) {
        if let PyJson::Dict(d) = &mut self.raw {
            if let Some(e) = d.iter_mut().find(|e| e.0 == k) {
                e.1 = v;
            } else {
                d.push((k.to_string(), v));
            }
        }
    }
}

fn val(v: &PyJson) -> Result<Value, String> {
    v.to_value()
}

/// `run.belief_keys(c, 'selfplay')` -> (tilt, clip, mix, fast, tilts) as `games.nn_seat` gets them
/// (its defaults when the belief is off: 0, None, 0, False, None).
#[allow(clippy::type_complexity)]
fn belief_keys(c: &Settings) -> Result<(f64, Option<f64>, f64, bool, Vec<(String, f64)>), String> {
    // mcts.belief_settings(cfg, 'selfplay')
    let z = PyJson::Float(0.0);
    let clip_d = PyJson::Int(20);
    let zi = PyJson::Int(0);
    let tilt = py_float("selfplay_belief_tilt", c.get_or_falsy("selfplay_belief_tilt", &z))?;
    let clip = py_float("belief_clip", c.get_or_falsy("belief_clip", &clip_d))?;
    let mix = py_float("belief_mix", c.get_or_falsy("belief_mix", &z))?;
    let fast = py_int("belief_fast", c.get_or_falsy("belief_fast", &zi))? != 0;
    // mcts.belief_tilts(cfg, 'selfplay'): checked, the zeros dropped, sorted
    let empty = PyJson::Dict(Vec::new());
    let t = c.get_or_falsy("selfplay_belief_tilts", &empty);
    let tilts = check_belief_tilts(t, "selfplay_belief_tilts")?;
    let tilts: Vec<(String, f64)> = tilts.into_iter().filter(|x| x.1 > 0.0).collect();
    if tilt != 0.0 || !tilts.is_empty() {
        Ok((tilt, Some(clip), mix, fast, tilts))
    } else {
        Ok((0.0, None, 0.0, false, Vec::new()))
    }
}

/// `mcts.check_belief_tilts` -> the tilts as floats, sorted by name.
fn check_belief_tilts(t: &PyJson, what: &str) -> Result<Vec<(String, f64)>, String> {
    let PyJson::Dict(d) = t else {
        return Err(format!("{what} {}: a dict {{target: tilt}} (docs/search.md 3.8.1)", t.dumps()));
    };
    let mut out = Vec::new();
    for (k, v) in d {
        if !is_num(v) || !num_val(v).is_finite() || num_val(v) < 0.0 {
            return Err(format!("{what} {k:?}: {}: a target name and a finite number >= 0 (0 = off)", v.dumps()));
        }
        out.push((k.clone(), num_val(v)));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// The self-play seat of a run's settings, as the seat dict Python builds
/// (`Run.nn_seat(key, sims, selfplay=True)`; `sims` = `c['sims']` as given).
pub fn selfplay_seat(c: &Settings, key: &str) -> Result<Value, String> {
    let zero = PyJson::Int(0);
    let zf = PyJson::Float(0.0);
    let four = PyJson::Int(4);
    let sixteen = PyJson::Int(16);
    let one_f = PyJson::Float(1.0);
    let visits = PyJson::Str("visits".into());
    let mover = PyJson::Str("mover".into());
    let next_q = PyJson::Str("next_q".into());
    let puct = PyJson::Str("puct".into());
    let sims = c.req("sims")?;
    let c_puct = c.req("c_puct")?;
    let alpha = c.req("alpha")?;
    let eps = c.req("eps")?;
    let temp_moves = c.req("temp_moves")?;
    let sims_fast = c.req("sims_fast")?;
    let p_full = c.req("p_full")?;
    let chance_k = c.req("chance_k")?;
    let margin_w = c.req("margin_w")?;
    let redet = truthy(c.req("redet")?);
    let policy_target = c.get_or("policy_target", &visits);
    let cq_c_visit = c.get_or("cq_c_visit", &PyJson::Float(50.0)).clone();
    let cq_c_scale = c.get_or("cq_c_scale", &one_f).clone();
    let sims_minor = c.get_or_falsy("sims_minor", &PyJson::Null);
    let select = c.get_or("selfplay_select", &visits);
    let select_min_visits = c.get_or("selfplay_select_min_visits", &zero);
    let select_c_visit = c.get_or("selfplay_select_c_visit", &PyJson::Null);
    let select_c_scale = c.get_or("selfplay_select_c_scale", &PyJson::Null);
    let root_min_visits = c.get_or("selfplay_root_min_visits", &zero);
    let root_min_max = c.get_or("root_min_visits_max_actions", &four);
    let leaf_value_view = c.get_or("selfplay_leaf_value_view", &mover);
    let view_samples = c.get_or("selfplay_view_samples", &zf);
    let view_sample_target = c.get_or("view_sample_target", &next_q);
    let root_search = c.get_or("selfplay_root_search", &puct);
    let gumbel_m = c.get_or("gumbel_m", &sixteen);
    let gumbel_scale = c.get_or("selfplay_gumbel_scale", &one_f);
    let eval_cache = c.get_or("eval_cache", &zero);
    let cq_prior_w = c.get_or("cq_prior_w", &one_f);
    let infoset_merge = truthy(c.get_or("selfplay_infoset_merge", &zero));
    let (belief_tilt, belief_clip, belief_mix, belief_fast, belief_tilts) = belief_keys(c)?;
    // resign.seat_keys(c)
    let (resign_t, resign_k, resign_noresign) = if py_float("selfplay_resign_t", c.get_or_falsy("selfplay_resign_t", &zf))? > 0.0 {
        (PyJson::Float(py_float("selfplay_resign_t", c.req("selfplay_resign_t")?)?), PyJson::Int(py_int("selfplay_resign_k", c.get_or("selfplay_resign_k", &PyJson::Int(5)))?), PyJson::Float(py_float("selfplay_resign_noresign", c.get_or("selfplay_resign_noresign", &PyJson::Float(0.1)))?))
    } else {
        (PyJson::Float(0.0), PyJson::Int(5), PyJson::Float(0.1))
    };
    // run.value_choice_keys(c, 'selfplay')
    let w = py_int("selfplay_value_choice_worlds", c.get_or_falsy("selfplay_value_choice_worlds", &zero))?;
    let (vc_worlds, vc_temp) = if w != 0 { (PyJson::Int(w), PyJson::Float(py_float("selfplay_value_choice_temp", c.get_or_falsy("selfplay_value_choice_temp", &zf))?)) } else { (PyJson::Int(0), PyJson::Float(0.0)) };

    // ---- games.nn_seat's checks, in its order
    // mcts.check_select
    if !(is_str(select, "visits") || is_str(select, "cq")) {
        return Err(format!("select {}: one of ('visits', 'cq')", select.dumps()));
    }
    if !is_int(select_min_visits) || num_val(select_min_visits) < 0.0 {
        return Err(format!("select_min_visits {}: an integer >= 0", select_min_visits.dumps()));
    }
    if is_str(select, "visits") && truthy(select_min_visits) {
        return Err(format!("select_min_visits {}: a guard of select 'cq' ('visits' has none)", select_min_visits.dumps()));
    }
    // mcts.check_cq_prior_w
    if !is_num(cq_prior_w) || !(num_val(cq_prior_w) > 0.0 && num_val(cq_prior_w) <= 1.0) {
        return Err(format!("cq_prior_w {}: the prior's exponent in the 'cq' target, in (0, 1] (1 = off)", cq_prior_w.dumps()));
    }
    if !eq_num(cq_prior_w, 1.0) && !is_str(policy_target, "cq") {
        return Err(format!("cq_prior_w {}: a setting of policy_target 'cq' ({} has no prior term)", cq_prior_w.dumps(), policy_target.dumps()));
    }
    // mcts.check_root_search (gumbel_c_visit / _c_scale: None)
    if !(is_str(root_search, "puct") || is_str(root_search, "gumbel")) {
        return Err(format!("root_search {}: one of ('puct', 'gumbel')", root_search.dumps()));
    }
    if !is_int(gumbel_m) || num_val(gumbel_m) < 1.0 {
        return Err(format!("gumbel_m {}: an integer >= 1", gumbel_m.dumps()));
    }
    if !matches!(gumbel_scale, PyJson::Null) && (!is_num(gumbel_scale) || !num_val(gumbel_scale).is_finite() || num_val(gumbel_scale) < 0.0) {
        return Err(format!("gumbel_scale {}: a finite number >= 0", gumbel_scale.dumps()));
    }
    if is_str(root_search, "gumbel") && truthy(root_min_visits) {
        return Err("root_min_visits with root_search 'gumbel': exclusive (sequential halving already visits every considered option; docs/search.md 3.7)".into());
    }
    // mcts.check_select_transform
    for (name, x) in [("c_visit", select_c_visit), ("c_scale", select_c_scale)] {
        if matches!(x, PyJson::Null) {
            continue;
        }
        if !is_num(x) || !num_val(x).is_finite() || num_val(x) < 0.0 {
            return Err(format!("select_{name} {}: a finite number >= 0 (unset: cq_{name})", x.dumps()));
        }
        if is_str(select, "visits") {
            return Err(format!("select_{name} {}: a setting of select 'cq' ('visits' has no completed-Q transform)", x.dumps()));
        }
    }
    // mcts.check_root_min
    if !is_int(root_min_visits) || num_val(root_min_visits) < 0.0 {
        return Err(format!("root_min_visits {}: an integer >= 0 (0 = off)", root_min_visits.dumps()));
    }
    if !is_int(root_min_max) || num_val(root_min_max) < 2.0 {
        return Err(format!("root_min_visits_max_actions {}: an integer >= 2", root_min_max.dumps()));
    }
    // mcts.check_leaf_view
    if !(is_str(leaf_value_view, "mover") || is_str(leaf_value_view, "root")) {
        return Err(format!("leaf_value_view {}: one of ('mover', 'root')", leaf_value_view.dumps()));
    }
    if !is_num(view_samples) || !(0.0..=1.0).contains(&num_val(view_samples)) {
        return Err(format!("view_samples {}: a share in [0, 1] (0 = off)", view_samples.dumps()));
    }
    if !(is_str(view_sample_target, "next_q") || is_str(view_sample_target, "z")) {
        return Err(format!("view_sample_target {}: one of ('next_q', 'z')", view_sample_target.dumps()));
    }
    // mcts.check_belief (the values are floats already)
    if !(belief_tilt.is_finite() && belief_tilt >= 0.0) {
        return Err(format!("belief_tilt {belief_tilt}: a finite number >= 0 (0 = off)"));
    }
    if let Some(cl) = belief_clip {
        if !(cl.is_finite() && cl >= 1.0) {
            return Err(format!("belief_clip {cl}: a finite number >= 1 (every weight in [1/clip, clip])"));
        }
    }
    if !(belief_mix.is_finite() && (0.0..1.0).contains(&belief_mix)) {
        return Err(format!("belief_mix {belief_mix}: a share in [0, 1) of the simulations dealt uniformly (1 would make belief_tilt a no-op: set it to 0)"));
    }
    // (check_belief_tilts: done by belief_keys; `if v` drops none: they are > 0)
    // mcts.check_merge (a bool)
    if infoset_merge && !is_str(leaf_value_view, "root") {
        return Err("infoset_merge: needs leaf_value_view 'root' (a merged node of another seat is valued from the root decider's view; docs/search.md 3.16)".into());
    }
    // mcts.check_value_choice
    let wv = num_val(&vc_worlds);
    if wv.trunc() != wv || !(0.0..=VALUE_CHOICE_MAX_WORLDS).contains(&wv) {
        return Err(format!("value_choice_worlds {}: worlds per option, an integer in [0, 1024] (0 = off)", vc_worlds.dumps()));
    }
    let tv = num_val(&vc_temp);
    if !(tv.is_finite() && tv >= 0.0) {
        return Err(format!("value_choice_temp {}: a temperature >= 0 (0: the best mean value)", vc_temp.dumps()));
    }
    // eval_cache: `int(eval_cache) != eval_cache or eval_cache < 0`
    let ec = py_int("eval_cache", eval_cache)?;
    if (ec as f64) != num_val(eval_cache) || num_val(eval_cache) < 0.0 {
        return Err(format!("eval_cache {}: answers per game, an integer >= 0 (0 = off)", eval_cache.dumps()));
    }
    // resign.check_values (the values are numbers already)
    let (rt, rk, rn) = (num_val(&resign_t), &resign_k, num_val(&resign_noresign));
    if !(0.0..0.5).contains(&rt) {
        return Err(format!("resign_t {rt}: a threshold in [0, 0.5) (0 = off; a search is decided when a seat's root win estimate is >= 1 - t)"));
    }
    if num_val(rk) < 1.0 {
        return Err(format!("resign_k {}: an integer >= 1 (consecutive decided full searches)", rk.dumps()));
    }
    if !(0.0..=1.0).contains(&rn) {
        return Err(format!("resign_noresign {rn}: a share in [0, 1] of the games that never resign"));
    }

    // ---- the dict
    let mut d = Map::new();
    d.insert("key".into(), Value::String(key.into()));
    d.insert("sims".into(), val(sims)?);
    d.insert("c_puct".into(), val(c_puct)?);
    d.insert("selfplay".into(), Value::Bool(true));
    d.insert("alpha".into(), val(alpha)?);
    d.insert("eps".into(), val(eps)?);
    d.insert("temp_moves".into(), val(temp_moves)?);
    d.insert("sims_fast".into(), val(sims_fast)?);
    d.insert("p_full".into(), val(p_full)?);
    d.insert("chance_k".into(), val(chance_k)?);
    d.insert("margin_w".into(), val(margin_w)?);
    d.insert("redet".into(), Value::Bool(redet));
    let f = |x: f64| serde_json::Number::from_f64(x).map(Value::Number).ok_or_else(|| format!("{x}: not a finite number"));
    // mcts.select_transform (computed unconditionally: float(None) fails as in Python)
    let sel = (
        py_float("selfplay_select_c_visit / cq_c_visit", if matches!(select_c_visit, PyJson::Null) { &cq_c_visit } else { select_c_visit })?,
        py_float("selfplay_select_c_scale / cq_c_scale", if matches!(select_c_scale, PyJson::Null) { &cq_c_scale } else { select_c_scale })?,
    );
    if !is_str(policy_target, "visits") {
        d.insert("policy_target".into(), val(policy_target)?);
        d.insert("cq_c_visit".into(), f(py_float("cq_c_visit", &cq_c_visit)?)?);
        d.insert("cq_c_scale".into(), f(py_float("cq_c_scale", &cq_c_scale)?)?);
    } else if !is_str(select, "visits") {
        d.insert("cq_c_visit".into(), f(sel.0)?);
        d.insert("cq_c_scale".into(), f(sel.1)?);
    }
    if !is_str(select, "visits") {
        d.insert("select".into(), val(select)?);
        d.insert("select_min_visits".into(), Value::from(py_int("selfplay_select_min_visits", select_min_visits)? as i64));
        if !is_str(policy_target, "visits") && sel != (py_float("cq_c_visit", &cq_c_visit)?, py_float("cq_c_scale", &cq_c_scale)?) {
            d.insert("select_c_visit".into(), f(sel.0)?);
            d.insert("select_c_scale".into(), f(sel.1)?);
        }
    }
    if truthy(sims_minor) {
        d.insert("sims_minor".into(), Value::from(py_int("sims_minor", sims_minor)? as i64));
    }
    if truthy(root_min_visits) {
        d.insert("root_min_visits".into(), Value::from(py_int("selfplay_root_min_visits", root_min_visits)? as i64));
        d.insert("root_min_visits_max_actions".into(), Value::from(py_int("root_min_visits_max_actions", root_min_max)? as i64));
    }
    if !is_str(leaf_value_view, "mover") {
        d.insert("leaf_value_view".into(), val(leaf_value_view)?);
    }
    if truthy(view_samples) {
        d.insert("view_samples".into(), f(py_float("selfplay_view_samples", view_samples)?)?);
        d.insert("view_sample_target".into(), val(view_sample_target)?);
    }
    if belief_tilt != 0.0 || !belief_tilts.is_empty() {
        d.insert("belief_tilt".into(), f(belief_tilt)?);
        d.insert("belief_clip".into(), f(belief_clip.unwrap_or(BELIEF_CLIP))?);
        d.insert("belief_mix".into(), f(belief_mix)?);
        d.insert("belief_fast".into(), Value::Bool(belief_fast));
        if !belief_tilts.is_empty() {
            let mut t = Map::new();
            for (k, a) in &belief_tilts {
                t.insert(k.clone(), f(*a)?);
            }
            d.insert("belief_tilts".into(), Value::Object(t));
        }
    }
    if is_str(root_search, "gumbel") {
        let (cv, cs) = if is_str(policy_target, "cq") { (py_float("cq_c_visit", &cq_c_visit)?, py_float("cq_c_scale", &cq_c_scale)?) } else { sel };
        d.insert("root_search".into(), Value::String("gumbel".into()));
        d.insert("gumbel_m".into(), Value::from(py_int("gumbel_m", gumbel_m)? as i64));
        d.insert("gumbel_scale".into(), f(py_float("selfplay_gumbel_scale", gumbel_scale)?)?);
        d.insert("gumbel_c_visit".into(), f(cv)?);
        d.insert("gumbel_c_scale".into(), f(cs)?);
    }
    if truthy(eval_cache) {
        d.insert("eval_cache".into(), Value::from(ec as i64));
    }
    if truthy(&resign_t) {
        d.insert("resign_t".into(), f(num_val(&resign_t))?);
        d.insert("resign_k".into(), Value::from(py_int("resign_k", &resign_k)? as i64));
        d.insert("resign_noresign".into(), f(num_val(&resign_noresign))?);
    }
    if !eq_num(cq_prior_w, 1.0) {
        d.insert("cq_prior_w".into(), f(py_float("cq_prior_w", cq_prior_w)?)?);
    }
    if infoset_merge {
        d.insert("infoset_merge".into(), Value::Bool(true));
    }
    if truthy(&vc_worlds) {
        d.insert("value_choice_worlds".into(), Value::from(py_int("value_choice_worlds", &vc_worlds)? as i64));
        d.insert("value_choice_temp".into(), f(num_val(&vc_temp))?);
    }
    Ok(Value::Object(d))
}

/// `run.aux_settings(cfg, info)`: whether a run records its game's auxiliary targets (a weight
/// `aux_<target>_w` > 0 of one of the game's targets; the game package declares the engine's
/// targets, which Python checks).
pub fn aux_on(c: &Settings, info: &GameInfo) -> Result<bool, String> {
    let z = PyJson::Float(0.0);
    let mut on = false;
    for (name, _) in &info.aux {
        let k = format!("aux_{name}_w");
        if py_float(&k, c.get_or_falsy(&k, &z))? > 0.0 {
            on = true;
        }
    }
    // run.reader_settings: a reader on needs its target's weight
    for (name, _) in &info.aux {
        let r = format!("reader_{name}");
        if py_int(&r, c.get_or_falsy(&r, &PyJson::Int(0)))? != 0 {
            let k = format!("aux_{name}_w");
            if py_float(&k, c.get_or_falsy(&k, &z))? <= 0.0 {
                return Err(format!("{r} 1 needs {k} > 0: its target's auxiliary loss trains the reader (docs/az.md 36)"));
            }
        }
    }
    Ok(on)
}

/// `spec_json(c['spec'])`: '' or None -> "{}", a string as it is, else Python's `json.dumps`.
pub fn spec_of(c: &Settings) -> Result<String, String> {
    let s = c.req("spec")?;
    Ok(match s {
        PyJson::Null => "{}".into(),
        PyJson::Str(t) if t.is_empty() => "{}".into(),
        PyJson::Str(t) => t.clone(),
        other => other.dumps(),
    })
}

fn usize_of(c: &Settings, k: &str) -> Result<usize, String> {
    match c.req(k)? {
        PyJson::Bool(b) => Ok(*b as usize),
        PyJson::Int(i) => usize::try_from(*i).map_err(|_| format!("{k} {i}: out of range")),
        v => Err(format!("{k} {}: an integer", v.dumps())),
    }
}

/// The self-play driver of a Python remote worker for these settings (`remoteplay.Worker._load` with
/// `seed`, `net`): rolling, queue_only, recording, the run's player counts. The Python-only features
/// (the self-play league, branched games, archive starts) refuse: `check-config` lists them.
pub fn selfplay_driver_cfg(c: &Settings, info: &GameInfo, seed: u64, net: u32) -> Result<DriverCfg, String> {
    if py_float("selfplay_league", c.get_or_falsy("selfplay_league", &PyJson::Int(0)))? > 0.0 {
        return Err("selfplay_league: the self-play league is Python-only (zero/league.py): unsupported by the native worker".into());
    }
    if py_float("selfplay_branch", c.get_or_falsy("selfplay_branch", &PyJson::Float(0.0)))? > 0.0 {
        return Err("selfplay_branch: branched games are set up in Python (zero/forks.py): unsupported by the native worker".into());
    }
    if truthy(c.get_or_falsy("selfplay_archive", &PyJson::Str(String::new()))) {
        return Err("selfplay_archive: archive starts are set up in Python (zero/forks.py): unsupported by the native worker".into());
    }
    let seat = selfplay_seat(c, "m")?;
    let PyJson::List(players) = c.req("players")? else {
        return Err("players: a list of player counts".into());
    };
    let mut table = Vec::with_capacity(players.len());
    for p in players {
        let n = py_int("players", p)?;
        if !is_int(p) || n < 0 {
            return Err(format!("players: {} is not a player count", p.dumps()));
        }
        table.push(vec![seat.clone(); n as usize]);
    }
    let args = DriverArgs {
        spec: spec_of(c)?,
        concurrency: usize_of(c, "conc")?,
        seed,
        record: true,
        threads: usize_of(c, "threads")?,
        groups: usize_of(c, "groups")?,
        max_steps: usize_of(c, "max_steps")? as u64,
        rolling: true,
        net,
        queue_only: true,
        move_log: 0,
        aux: aux_on(c, info)?,
        paired: false,
        sparse: truthy(c.get("sparse").unwrap_or(&PyJson::Null)),
    };
    driver_cfg(&table, None, None, args)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info() -> GameInfo {
        GameInfo { name: "t", num_actions: 4, num_features: 3, min_players: 2, max_players: 2, bots: vec![], aux: vec![("belief", 4), ("deck", 4)] }
    }

    fn base() -> Settings {
        Settings::parse(r#"{"spec": "", "players": [2], "conc": 8, "threads": 2, "groups": 2, "max_steps": 100, "sims": 64, "sims_fast": 16, "p_full": 0.25, "c_puct": 1.0, "alpha": 0.3, "eps": 0.25, "temp_moves": 20, "chance_k": 0, "margin_w": 0.0, "redet": 1}"#).unwrap()
    }

    #[test]
    fn the_default_seat() {
        let s = selfplay_seat(&base(), "m").unwrap();
        assert_eq!(s, serde_json::json!({"key": "m", "sims": 64, "c_puct": 1.0, "selfplay": true, "alpha": 0.3, "eps": 0.25, "temp_moves": 20, "sims_fast": 16, "p_full": 0.25, "chance_k": 0, "margin_w": 0.0, "redet": true}));
    }

    #[test]
    fn options_as_python() {
        let mut c = base();
        c.set("policy_target", PyJson::Str("cq".into()));
        c.set("cq_c_scale", PyJson::Float(0.3));
        c.set("sims_minor", PyJson::Int(4));
        c.set("eval_cache", PyJson::Int(8192));
        c.set("selfplay_belief_tilts", PyJson::parse(r#"{"deck": 1, "belief": 0}"#).unwrap());
        c.set("selfplay_select", PyJson::Str("cq".into()));
        c.set("selfplay_select_c_scale", PyJson::Float(0.1));
        let s = selfplay_seat(&c, "m").unwrap();
        let o = s.as_object().unwrap();
        assert_eq!(o["policy_target"], "cq");
        assert_eq!(o["cq_c_scale"], 0.3);
        assert_eq!(o["sims_minor"], 4);
        assert_eq!(o["eval_cache"], 8192);
        assert_eq!(o["belief_tilts"], serde_json::json!({"deck": 1.0}));
        assert_eq!(o["belief_tilt"], 0.0);
        assert_eq!(o["belief_clip"], 20.0);
        assert_eq!(o["select"], "cq");
        assert_eq!(o["select_c_scale"], 0.1);
        assert_eq!(o["select_c_visit"], 50.0);
        let d = selfplay_driver_cfg(&c, &info(), 3, 1).unwrap();
        assert!(d.rolling && d.queue_only && d.record && !d.aux && d.games == vec![vec![0, 0]]);
        c.set("aux_deck_w", PyJson::Float(0.5));
        c.set("reader_deck", PyJson::Int(1));
        assert!(selfplay_driver_cfg(&c, &info(), 3, 1).unwrap().aux);
        c.set("aux_deck_w", PyJson::Float(0.0));
        assert!(selfplay_driver_cfg(&c, &info(), 3, 1).is_err());
    }

    #[test]
    fn python_only_features_refuse() {
        for (k, v) in [("selfplay_league", "0.25"), ("selfplay_branch", "0.1"), ("selfplay_archive", "\"a.json\"")] {
            let mut c = base();
            c.set(k, PyJson::parse(v).unwrap());
            assert!(selfplay_driver_cfg(&c, &info(), 0, 0).unwrap_err().contains(k));
        }
        let mut c = base();
        c.set("selfplay_league", PyJson::Float(0.0));
        c.set("selfplay_archive", PyJson::Str(String::new()));
        assert!(selfplay_driver_cfg(&c, &info(), 0, 0).is_ok());
    }

    #[test]
    fn spec_text_is_json_dumps() {
        let mut c = base();
        c.set("spec", PyJson::parse(r#"{"b": 1, "a": {"x": 0.5}}"#).unwrap());
        assert_eq!(spec_of(&c).unwrap(), r#"{"b": 1, "a": {"x": 0.5}}"#);
    }
}
