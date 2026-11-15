//! A run's settings -> the self-play driver a worker plays: the Rust port of what a Python remote
//! self-play worker builds from its mirror's `config.json` (zero-core `remoteplay.Worker._load`:
//! `Run.nn_seat('m', sims, selfplay=True)` = `games.nn_seat` with `run.belief_keys`,
//! `resign.seat_keys`, `run.value_choice_keys` and the `mcts` checks; `Run.driver(table, seed,
//! record=True, rolling=True, net, queue_only=True)`), then zero_search's shared parser
//! (`settings::driver_cfg`) - so the seat dict is the Python one, key for key and type for type, and
//! its parsing is the same code. tools/driver_settings_dump.py `--compare-worker` checks every
//! preset and run config against the Python path (docs/az.md 44).
//!
//! Evaluation (zero-farm eval jobs, docs/az.md 44.2): `eval_seat` is `Run.nn_seat(key, sims)` (the same
//! `games.nn_seat` port), `eval_driver_cfg` the driver of `Run.play_match` / zero.h2h's seat order, and
//! `eval_lease_cfg` how a lease of its games is played.
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

/// `run.belief_keys(c, pre)` (`pre` 'selfplay' | 'eval') -> (tilt, clip, mix, fast, tilts) as
/// `games.nn_seat` gets them (its defaults when the belief is off: 0, None, 0, False, None).
#[allow(clippy::type_complexity)]
fn belief_keys(c: &Settings, pre: &str) -> Result<(f64, Option<f64>, f64, bool, Vec<(String, f64)>), String> {
    // mcts.belief_settings(cfg, pre)
    let z = PyJson::Float(0.0);
    let clip_d = PyJson::Int(20);
    let zi = PyJson::Int(0);
    let tk = format!("{pre}_belief_tilt");
    let tilt = py_float(&tk, c.get_or_falsy(&tk, &z))?;
    let clip = py_float("belief_clip", c.get_or_falsy("belief_clip", &clip_d))?;
    let mix = py_float("belief_mix", c.get_or_falsy("belief_mix", &z))?;
    let fast = py_int("belief_fast", c.get_or_falsy("belief_fast", &zi))? != 0;
    // mcts.belief_tilts(cfg, pre): checked, the zeros dropped, sorted
    let empty = PyJson::Dict(Vec::new());
    let tsk = format!("{pre}_belief_tilts");
    let t = c.get_or_falsy(&tsk, &empty);
    let tilts = check_belief_tilts(t, &tsk)?;
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

/// `run.value_choice_keys(c, pre)` -> (worlds, temp) as `games.nn_seat` gets them (0, 0.0: off).
fn value_choice_keys(c: &Settings, pre: &str) -> Result<(PyJson, PyJson), String> {
    let (zero, zf) = (PyJson::Int(0), PyJson::Float(0.0));
    let wk = format!("{pre}_value_choice_worlds");
    let w = py_int(&wk, c.get_or_falsy(&wk, &zero))?;
    if w == 0 {
        return Ok((PyJson::Int(0), PyJson::Float(0.0)));
    }
    let tk = format!("{pre}_value_choice_temp");
    Ok((PyJson::Int(w), PyJson::Float(py_float(&tk, c.get_or_falsy(&tk, &zf))?)))
}

/// `games.nn_seat`'s arguments (Python values; the names its keywords).
struct SeatArgs {
    key: String,
    /// A setting's prefix in messages (`selfplay_` or `eval_`).
    pre: &'static str,
    sims: PyJson,
    c_puct: PyJson,
    selfplay: bool,
    alpha: PyJson,
    eps: PyJson,
    temp_moves: PyJson,
    sims_fast: PyJson,
    p_full: PyJson,
    chance_k: PyJson,
    margin_w: PyJson,
    redet: bool,
    policy_target: PyJson,
    cq_c_visit: PyJson,
    cq_c_scale: PyJson,
    sims_minor: PyJson,
    select: PyJson,
    select_min_visits: PyJson,
    select_c_visit: PyJson,
    select_c_scale: PyJson,
    root_min_visits: PyJson,
    root_min_max: PyJson,
    leaf_value_view: PyJson,
    view_samples: PyJson,
    view_sample_target: PyJson,
    #[allow(clippy::type_complexity)]
    belief: (f64, Option<f64>, f64, bool, Vec<(String, f64)>),
    root_search: PyJson,
    gumbel_m: PyJson,
    gumbel_scale: PyJson,
    eval_cache: PyJson,
    resign: (PyJson, PyJson, PyJson),
    cq_prior_w: PyJson,
    infoset_merge: bool,
    value_choice: (PyJson, PyJson),
}

/// The self-play seat of a run's settings, as the seat dict Python builds
/// (`Run.nn_seat(key, sims, selfplay=True)`; `sims` = `c['sims']` as given).
pub fn selfplay_seat(c: &Settings, key: &str) -> Result<Value, String> {
    let zero = PyJson::Int(0);
    let zf = PyJson::Float(0.0);
    // resign.seat_keys(c)
    let resign = if py_float("selfplay_resign_t", c.get_or_falsy("selfplay_resign_t", &zf))? > 0.0 {
        (
            PyJson::Float(py_float("selfplay_resign_t", c.req("selfplay_resign_t")?)?),
            PyJson::Int(py_int("selfplay_resign_k", c.get_or("selfplay_resign_k", &PyJson::Int(5)))?),
            PyJson::Float(py_float("selfplay_resign_noresign", c.get_or("selfplay_resign_noresign", &PyJson::Float(0.1)))?),
        )
    } else {
        (PyJson::Float(0.0), PyJson::Int(5), PyJson::Float(0.1))
    };
    let g = |k: &str, d: PyJson| c.get(k).cloned().unwrap_or(d);
    nn_seat(SeatArgs {
        key: key.into(),
        pre: "selfplay_",
        sims: c.req("sims")?.clone(),
        c_puct: c.req("c_puct")?.clone(),
        selfplay: true,
        alpha: c.req("alpha")?.clone(),
        eps: c.req("eps")?.clone(),
        temp_moves: c.req("temp_moves")?.clone(),
        sims_fast: c.req("sims_fast")?.clone(),
        p_full: c.req("p_full")?.clone(),
        chance_k: c.req("chance_k")?.clone(),
        margin_w: c.req("margin_w")?.clone(),
        redet: truthy(c.req("redet")?),
        policy_target: g("policy_target", PyJson::Str("visits".into())),
        cq_c_visit: g("cq_c_visit", PyJson::Float(50.0)),
        cq_c_scale: g("cq_c_scale", PyJson::Float(1.0)),
        sims_minor: c.get_or_falsy("sims_minor", &PyJson::Null).clone(),
        select: g("selfplay_select", PyJson::Str("visits".into())),
        select_min_visits: g("selfplay_select_min_visits", zero.clone()),
        select_c_visit: g("selfplay_select_c_visit", PyJson::Null),
        select_c_scale: g("selfplay_select_c_scale", PyJson::Null),
        root_min_visits: g("selfplay_root_min_visits", zero.clone()),
        root_min_max: g("root_min_visits_max_actions", PyJson::Int(4)),
        leaf_value_view: g("selfplay_leaf_value_view", PyJson::Str("mover".into())),
        view_samples: g("selfplay_view_samples", zf.clone()),
        view_sample_target: g("view_sample_target", PyJson::Str("next_q".into())),
        belief: belief_keys(c, "selfplay")?,
        root_search: g("selfplay_root_search", PyJson::Str("puct".into())),
        gumbel_m: g("gumbel_m", PyJson::Int(16)),
        gumbel_scale: g("selfplay_gumbel_scale", PyJson::Float(1.0)),
        eval_cache: g("eval_cache", zero.clone()),
        resign,
        cq_prior_w: g("cq_prior_w", PyJson::Float(1.0)),
        infoset_merge: truthy(c.get_or("selfplay_infoset_merge", &zero)),
        value_choice: value_choice_keys(c, "selfplay")?,
    })
}

/// The evaluation seat of a run's settings, as the seat dict Python builds (`Run.nn_seat(key, sims)`: the
/// seat of every evaluation match - the yardstick, random and Elo matches, zero-farm's eval jobs -: no
/// noise, no temperature, the move rule eval_select; `sims` as given, `c['eval_sims']` in an evaluation).
pub fn eval_seat(c: &Settings, key: &str, sims: &PyJson) -> Result<Value, String> {
    let zero = PyJson::Int(0);
    let g = |k: &str, d: PyJson| c.get(k).cloned().unwrap_or(d);
    nn_seat(SeatArgs {
        key: key.into(),
        pre: "eval_",
        sims: sims.clone(),
        c_puct: c.req("c_puct")?.clone(),
        selfplay: false,
        // (games.nn_seat's defaults: an evaluation seat's noise and temperature settings are never read)
        alpha: PyJson::Float(0.3),
        eps: PyJson::Float(0.25),
        temp_moves: PyJson::Int(20),
        sims_fast: PyJson::Null,
        p_full: PyJson::Float(1.0),
        chance_k: c.req("chance_k")?.clone(),
        margin_w: c.req("margin_w")?.clone(),
        redet: truthy(c.req("redet")?),
        policy_target: PyJson::Str("visits".into()),
        cq_c_visit: g("cq_c_visit", PyJson::Float(50.0)),
        cq_c_scale: g("cq_c_scale", PyJson::Float(1.0)),
        sims_minor: PyJson::Null,
        select: g("eval_select", PyJson::Str("visits".into())),
        select_min_visits: g("eval_select_min_visits", zero.clone()),
        select_c_visit: g("eval_select_c_visit", PyJson::Null),
        select_c_scale: g("eval_select_c_scale", PyJson::Null),
        root_min_visits: g("eval_root_min_visits", zero.clone()),
        root_min_max: g("root_min_visits_max_actions", PyJson::Int(4)),
        leaf_value_view: g("eval_leaf_value_view", PyJson::Str("mover".into())),
        view_samples: PyJson::Float(0.0),
        view_sample_target: PyJson::Str("next_q".into()),
        belief: belief_keys(c, "eval")?,
        root_search: g("eval_root_search", PyJson::Str("puct".into())),
        gumbel_m: g("gumbel_m", PyJson::Int(16)),
        gumbel_scale: g("eval_gumbel_scale", PyJson::Float(0.0)),
        eval_cache: g("eval_cache", zero.clone()),
        resign: (PyJson::Float(0.0), PyJson::Int(5), PyJson::Float(0.1)),
        cq_prior_w: PyJson::Float(1.0),
        infoset_merge: truthy(c.get_or("eval_infoset_merge", &zero)),
        value_choice: value_choice_keys(c, "eval")?,
    })
}

/// `games.nn_seat`: its checks in its order, then the dict (key for key, type for type).
fn nn_seat(a: SeatArgs) -> Result<Value, String> {
    let pre = a.pre;
    let (select, select_min_visits, policy_target, cq_prior_w) = (&a.select, &a.select_min_visits, &a.policy_target, &a.cq_prior_w);
    let (root_search, gumbel_m, gumbel_scale, root_min_visits, root_min_max) = (&a.root_search, &a.gumbel_m, &a.gumbel_scale, &a.root_min_visits, &a.root_min_max);
    let (select_c_visit, select_c_scale, leaf_value_view, view_samples, view_sample_target) = (&a.select_c_visit, &a.select_c_scale, &a.leaf_value_view, &a.view_samples, &a.view_sample_target);
    let (cq_c_visit, cq_c_scale, eval_cache, infoset_merge) = (&a.cq_c_visit, &a.cq_c_scale, &a.eval_cache, a.infoset_merge);
    let (belief_tilt, belief_clip, belief_mix, belief_fast, belief_tilts) = (a.belief.0, a.belief.1, a.belief.2, a.belief.3, &a.belief.4);
    let (resign_t, resign_k, resign_noresign) = (&a.resign.0, &a.resign.1, &a.resign.2);
    let (vc_worlds, vc_temp) = (&a.value_choice.0, &a.value_choice.1);

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
    let wv = num_val(vc_worlds);
    if wv.trunc() != wv || !(0.0..=VALUE_CHOICE_MAX_WORLDS).contains(&wv) {
        return Err(format!("value_choice_worlds {}: worlds per option, an integer in [0, 1024] (0 = off)", vc_worlds.dumps()));
    }
    let tv = num_val(vc_temp);
    if !(tv.is_finite() && tv >= 0.0) {
        return Err(format!("value_choice_temp {}: a temperature >= 0 (0: the best mean value)", vc_temp.dumps()));
    }
    // eval_cache: `int(eval_cache) != eval_cache or eval_cache < 0`
    let ec = py_int("eval_cache", eval_cache)?;
    if (ec as f64) != num_val(eval_cache) || num_val(eval_cache) < 0.0 {
        return Err(format!("eval_cache {}: answers per game, an integer >= 0 (0 = off)", eval_cache.dumps()));
    }
    // resign.check_values (the values are numbers already)
    let (rt, rk, rn) = (num_val(resign_t), resign_k, num_val(resign_noresign));
    if !(0.0..0.5).contains(&rt) {
        return Err(format!("resign_t {rt}: a threshold in [0, 0.5) (0 = off; a search is decided when a seat's root win estimate is >= 1 - t)"));
    }
    if num_val(rk) < 1.0 {
        return Err(format!("resign_k {}: an integer >= 1 (consecutive decided full searches)", rk.dumps()));
    }
    if !(0.0..=1.0).contains(&rn) {
        return Err(format!("resign_noresign {rn}: a share in [0, 1] of the games that never resign"));
    }
    if truthy(resign_t) && !a.selfplay {
        return Err(format!("resign_t {rt}: a self-play seat only (evaluations play their games out)"));
    }

    // ---- the dict
    let mut d = Map::new();
    d.insert("key".into(), Value::String(a.key.clone()));
    d.insert("sims".into(), val(&a.sims)?);
    d.insert("c_puct".into(), val(&a.c_puct)?);
    d.insert("selfplay".into(), Value::Bool(a.selfplay));
    d.insert("alpha".into(), val(&a.alpha)?);
    d.insert("eps".into(), val(&a.eps)?);
    d.insert("temp_moves".into(), val(&a.temp_moves)?);
    d.insert("sims_fast".into(), val(&a.sims_fast)?);
    d.insert("p_full".into(), val(&a.p_full)?);
    d.insert("chance_k".into(), val(&a.chance_k)?);
    d.insert("margin_w".into(), val(&a.margin_w)?);
    d.insert("redet".into(), Value::Bool(a.redet));
    let f = |x: f64| serde_json::Number::from_f64(x).map(Value::Number).ok_or_else(|| format!("{x}: not a finite number"));
    // mcts.select_transform (computed unconditionally: float(None) fails as in Python)
    let sel = (
        py_float(&format!("{pre}select_c_visit / cq_c_visit"), if matches!(select_c_visit, PyJson::Null) { cq_c_visit } else { select_c_visit })?,
        py_float(&format!("{pre}select_c_scale / cq_c_scale"), if matches!(select_c_scale, PyJson::Null) { cq_c_scale } else { select_c_scale })?,
    );
    if !is_str(policy_target, "visits") {
        d.insert("policy_target".into(), val(policy_target)?);
        d.insert("cq_c_visit".into(), f(py_float("cq_c_visit", cq_c_visit)?)?);
        d.insert("cq_c_scale".into(), f(py_float("cq_c_scale", cq_c_scale)?)?);
    } else if !is_str(select, "visits") {
        d.insert("cq_c_visit".into(), f(sel.0)?);
        d.insert("cq_c_scale".into(), f(sel.1)?);
    }
    if !is_str(select, "visits") {
        d.insert("select".into(), val(select)?);
        d.insert("select_min_visits".into(), Value::from(py_int(&format!("{pre}select_min_visits"), select_min_visits)? as i64));
        if !is_str(policy_target, "visits") && sel != (py_float("cq_c_visit", cq_c_visit)?, py_float("cq_c_scale", cq_c_scale)?) {
            d.insert("select_c_visit".into(), f(sel.0)?);
            d.insert("select_c_scale".into(), f(sel.1)?);
        }
    }
    if truthy(&a.sims_minor) {
        d.insert("sims_minor".into(), Value::from(py_int("sims_minor", &a.sims_minor)? as i64));
    }
    if truthy(root_min_visits) {
        d.insert("root_min_visits".into(), Value::from(py_int(&format!("{pre}root_min_visits"), root_min_visits)? as i64));
        d.insert("root_min_visits_max_actions".into(), Value::from(py_int("root_min_visits_max_actions", root_min_max)? as i64));
    }
    if !is_str(leaf_value_view, "mover") {
        d.insert("leaf_value_view".into(), val(leaf_value_view)?);
    }
    if truthy(view_samples) {
        d.insert("view_samples".into(), f(py_float(&format!("{pre}view_samples"), view_samples)?)?);
        d.insert("view_sample_target".into(), val(view_sample_target)?);
    }
    if belief_tilt != 0.0 || !belief_tilts.is_empty() {
        d.insert("belief_tilt".into(), f(belief_tilt)?);
        d.insert("belief_clip".into(), f(belief_clip.unwrap_or(BELIEF_CLIP))?);
        d.insert("belief_mix".into(), f(belief_mix)?);
        d.insert("belief_fast".into(), Value::Bool(belief_fast));
        if !belief_tilts.is_empty() {
            let mut t = Map::new();
            for (k, x) in belief_tilts {
                t.insert(k.clone(), f(*x)?);
            }
            d.insert("belief_tilts".into(), Value::Object(t));
        }
    }
    if is_str(root_search, "gumbel") {
        let (cv, cs) = if is_str(policy_target, "cq") { (py_float("cq_c_visit", cq_c_visit)?, py_float("cq_c_scale", cq_c_scale)?) } else { sel };
        d.insert("root_search".into(), Value::String("gumbel".into()));
        d.insert("gumbel_m".into(), Value::from(py_int("gumbel_m", gumbel_m)? as i64));
        d.insert("gumbel_scale".into(), f(py_float(&format!("{pre}gumbel_scale"), gumbel_scale)?)?);
        d.insert("gumbel_c_visit".into(), f(cv)?);
        d.insert("gumbel_c_scale".into(), f(cs)?);
    }
    if truthy(eval_cache) {
        d.insert("eval_cache".into(), Value::from(ec as i64));
    }
    if truthy(resign_t) {
        d.insert("resign_t".into(), f(num_val(resign_t))?);
        d.insert("resign_k".into(), Value::from(py_int("resign_k", resign_k)? as i64));
        d.insert("resign_noresign".into(), f(num_val(resign_noresign))?);
    }
    if !eq_num(cq_prior_w, 1.0) {
        d.insert("cq_prior_w".into(), f(py_float("cq_prior_w", cq_prior_w)?)?);
    }
    if infoset_merge {
        d.insert("infoset_merge".into(), Value::Bool(true));
    }
    if truthy(vc_worlds) {
        d.insert("value_choice_worlds".into(), Value::from(py_int("value_choice_worlds", vc_worlds)? as i64));
        d.insert("value_choice_temp".into(), f(num_val(vc_temp))?);
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

/// A seat of an eval job (zero-farm PROTOCOL.md 5.2): a network, by its generation in the job's `models`,
/// or a built-in bot of the game.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvalSeat {
    Net(u32),
    Bot(String),
}

impl EvalSeat {
    /// `{"net": <gen>}` or `{"bot": "<name>"}` (nothing else in it); an external bot (`ext:<name>`, a
    /// process on the desktop) is refused.
    pub fn parse(v: &Value) -> Result<EvalSeat, String> {
        let o = v.as_object().ok_or_else(|| format!("seat {v}: {{\"net\": <gen>}} or {{\"bot\": <name>}}"))?;
        if o.len() != 1 {
            return Err(format!("seat {v}: one key, net or bot"));
        }
        match (o.get("net"), o.get("bot")) {
            (Some(g), None) => match g.as_u64() {
                Some(g) if g <= u32::MAX as u64 => Ok(EvalSeat::Net(g as u32)),
                _ => Err(format!("seat {v}: net is a checkpoint generation (an integer >= 0)")),
            },
            (None, Some(Value::String(b))) if b.starts_with("ext:") => Err(format!("seat {v}: an external bot ({b}) runs on the desktop, not in a worker (zero.run's ext: players)")),
            (None, Some(Value::String(b))) if !b.is_empty() => Ok(EvalSeat::Bot(b.clone())),
            _ => Err(format!("seat {v}: {{\"net\": <gen>}} or {{\"bot\": <name>}}")),
        }
    }

    /// The seat as the result rows write it (`json.dumps(seat, separators=(',', ':'))`).
    pub fn pyjson(&self) -> PyJson {
        match self {
            EvalSeat::Net(g) => PyJson::Dict(vec![("net".into(), PyJson::Int(*g as i128))]),
            EvalSeat::Bot(b) => PyJson::Dict(vec![("bot".into(), PyJson::Str(b.clone()))]),
        }
    }
}

/// The settings an evaluation match reads (`Run.nn_seat(key, c['eval_sims'])`, `Run.driver(seats, seed,
/// conc=..., paired=...)` with the run's `eval_spec`); every other run setting is unused by an eval job.
/// zero-core's tests/test_worker_settings.py checks the list against what those functions read.
pub const EVAL_READS: &[&str] = &[
    "game", "eval_spec", "conc", "threads", "groups", "max_steps", "sparse", "eval_sims", "c_puct", "chance_k", "margin_w",
    "redet", "cq_c_visit", "cq_c_scale", "eval_select", "eval_select_min_visits", "eval_select_c_visit",
    "eval_select_c_scale", "eval_root_min_visits", "root_min_visits_max_actions", "eval_leaf_value_view",
    "eval_root_search", "gumbel_m", "eval_gumbel_scale", "eval_cache", "eval_infoset_merge", "eval_belief_tilt",
    "belief_clip", "belief_mix", "belief_fast", "eval_belief_tilts", "eval_value_choice_worlds", "eval_value_choice_temp",
];

/// The evaluation games' spec: the run's `eval_spec` (`zero.games.spec_json`: a dict as `json.dumps` writes
/// it). Python falls back to the game package's `eval_specs` hook (Python code) when it is unset, so an eval
/// job needs it set: the desktop resolves it (zero.farm_h2h does: `Run.driver`'s rule).
pub fn eval_spec_of(c: &Settings) -> Result<String, String> {
    match c.get("eval_spec") {
        Some(PyJson::Str(t)) if !t.is_empty() => Ok(t.clone()),
        Some(v @ PyJson::Dict(d)) if !d.is_empty() => Ok(v.dumps()),
        Some(v) if truthy(v) => Err(format!("eval_spec {}: a spec (JSON text or an object)", v.dumps())),
        _ => Err("eval_spec is unset: an evaluation plays the game package's eval spec (its Python eval_specs hook, \
                  `Run.driver`), which a worker cannot run - the desktop resolves it into the settings (zero.farm_h2h does)"
            .into()),
    }
}

/// The driver Python's `Run.play_match` builds for an evaluation of `games` games between `seats` (game g:
/// `seats` for even g, reversed for odd g; a network seat is `Run.nn_seat(key, c['eval_sims'])` with key 'm'
/// in `seats[0]` and 'o' in `seats[1]`, a bot `G.seat_dict(name)`): `Run.driver(table, seed, conc=min(conc,
/// games), paired=paired)` - a fixed table of game ids 0..games, the run's eval spec, threads, groups,
/// max_steps and sparse (`conc`, `threads`, `groups`: a job's own over the run's). The eval job plays it as
/// [`eval_lease_cfg`] does.
#[allow(clippy::too_many_arguments)]
pub fn eval_driver_cfg(c: &Settings, info: &GameInfo, seats: &[EvalSeat], games: u64, seed: u64, paired: bool, conc: Option<usize>, threads: Option<usize>, groups: Option<usize>) -> Result<DriverCfg, String> {
    if seats.len() != 2 {
        return Err(format!("an eval job has 2 seats (got {}): a match of two players, seats swapped between games", seats.len()));
    }
    if !(info.min_players..=info.max_players).contains(&2) {
        return Err(format!("{}: {}..{} players, an evaluation match has 2", info.name, info.min_players, info.max_players));
    }
    let sims = c.req("eval_sims")?;
    let mut dicts = Vec::with_capacity(2);
    for (k, s) in seats.iter().enumerate() {
        dicts.push(match s {
            EvalSeat::Net(_) => eval_seat(c, ["m", "o"][k], sims)?,
            EvalSeat::Bot(b) => {
                if !info.bots.contains(&b.as_str()) {
                    return Err(format!("bot {b:?}: {}'s built-in bots are {:?}", info.name, info.bots));
                }
                serde_json::json!({"bot": b})
            }
        });
    }
    let table: Vec<Vec<Value>> = (0..games).map(|g| if g % 2 == 0 { vec![dicts[0].clone(), dicts[1].clone()] } else { vec![dicts[1].clone(), dicts[0].clone()] }).collect();
    let conc = conc.unwrap_or(usize_of(c, "conc")?).min(games.max(1) as usize);
    let args = DriverArgs {
        spec: eval_spec_of(c)?,
        concurrency: conc,
        seed,
        record: false,
        threads: match threads {
            Some(t) => t,
            None => usize_of(c, "threads")?,
        },
        groups: match groups {
            Some(g) => g,
            None => usize_of(c, "groups")?,
        },
        max_steps: usize_of(c, "max_steps")? as u64,
        rolling: false,
        net: 0,
        queue_only: false,
        move_log: 0,
        aux: false,
        paired,
        sparse: truthy(c.get("sparse").unwrap_or(&PyJson::Null)),
        max_row_tokens: None,
    };
    driver_cfg(&table, None, None, args)
}

/// The driver an eval job plays a lease with: [`eval_driver_cfg`]'s match as a rolling queue_only driver whose
/// table is its first two rows (game gid takes row gid % 2: the match's seats for every game id) and sparse
/// rows (the graph reads them; the same rows). With every game's setup (the eval spec) queued before its
/// first resume and `set_first_gid(gid_base)` it plays games gid_base.. of the match - with one group the
/// fixed table's batches in its order (zero_search `a_queued_rolling_driver_plays_a_fixed_tables_match_
/// batch_for_batch`, docs/search.md 2.16).
pub fn eval_lease_cfg(mut m: DriverCfg) -> DriverCfg {
    m.games.truncate(2);
    if m.games.len() == 1 {
        let r: Vec<u16> = m.games[0].iter().rev().copied().collect();
        m.games.push(r);
    }
    m.rolling = true;
    m.queue_only = true;
    m.sparse = true;
    m
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
        // (the game's own row limit, as the Python remote worker's driver: docs/search.md 2.15)
        max_row_tokens: None,
    };
    driver_cfg(&table, None, None, args)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info() -> GameInfo {
        GameInfo { name: "t", num_actions: 4, num_features: 3, min_players: 2, max_players: 2, bots: vec![], aux: vec![("belief", 4), ("deck", 4)], tails: vec![], option_pointers: false, max_row_tokens: None, head_tokens: 0 }
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
