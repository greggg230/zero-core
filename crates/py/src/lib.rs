//! Zero Core's PyO3 bindings (docs/search.md 4): the classes behind every game's Python module
//! `<package>._engine` - the search driver, single search trees and game objects for each
//! registered game type. A game's binding crate is a `cdylib` with ONE macro call:
//!
//! ```ignore
//! zero_py::engine_module!(_engine, {
//!     "toy" => zero_search::toy::ToyGame,
//!     "glassroad" => grz_engine::az::GlassRoadAz,
//! });
//! ```
//!
//! (its `Cargo.toml` depends on `zero_py`, `zero_search`, its engine crate and
//! `pyo3 = { version = "0.23", features = ["extension-module"] }`). The module exposes `Game`,
//! `SearchTree`, `SearchDriver`, `SampleRecords` (`SearchDriver.take_records`), `Lookahead` (the
//! position API of docs/search.md 3.9),
//! `game_names()`, `game_info_of(name)`, `lookahead_settings(settings)`, `driver_settings(seats, ...)`
//! (the `DriverCfg` a `SearchDriver` would get, as text: a diagnostic) and `MAX_SEATS`.
use std::sync::OnceLock;

use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyList, PyString, PyTuple};
use serde_json::{Map, Value};

use zero_search::driver::{noresign_pick, root_floor, sample_arrays_rows, view_pick, Batch, DriverCfg, Record, Select, BELIEF_CLIP};
use zero_search::settings::{self, DriverArgs};
use zero_search::dynapi::{boxed_driver, boxed_game, game_info, DynDriver, DynGame, DynLook, DynTree, GameInfo};
use zero_search::lookahead::{LookCfg, Outcome};
use zero_search::record::Replayed;
use zero_search::tail::TailRow;
use zero_search::{SearchGame, MAX_SEATS};

mod pack;

/// For `engine_module!` (the game crate needs its own `pyo3` dependency too: the PyO3 macros
/// refer to `::pyo3`).
pub use pyo3;

/// One registered game type: its name in Python and its constructors.
pub struct GameEntry {
    pub name: &'static str,
    pub new_game: fn(&str, usize, u64) -> Result<Box<dyn DynGame>, String>,
    pub new_driver: fn(DriverCfg) -> Result<Box<dyn DynDriver>, String>,
    pub info: fn() -> GameInfo,
    /// `zero_search::record::replay_json` of the game type (docs/search.md 2.17).
    pub replay: fn(&str, &str, bool) -> Result<Replayed, String>,
}

impl GameEntry {
    pub const fn of<G: SearchGame>(name: &'static str) -> GameEntry {
        GameEntry { name, new_game: boxed_game::<G>, new_driver: boxed_driver::<G>, info: game_info::<G>, replay: zero_search::record::replay_json::<G> }
    }
}

/// The games of this extension module (each game's cdylib links its own copy of this crate).
static GAMES: OnceLock<&'static [GameEntry]> = OnceLock::new();

/// Define the `#[pymodule]` `$module` (the last component of the Python module name, e.g.
/// `_engine` for `grz._engine`) holding the listed games: `"name" => SearchGame type`.
#[macro_export]
macro_rules! engine_module {
    ($module:ident, { $($name:literal => $ty:ty),* $(,)? }) => {
        static ZERO_PY_GAMES: &[$crate::GameEntry] = &[$($crate::GameEntry::of::<$ty>($name)),*];

        #[$crate::pyo3::pymodule]
        fn $module(m: &$crate::pyo3::Bound<'_, $crate::pyo3::types::PyModule>) -> $crate::pyo3::PyResult<()> {
            $crate::init_module(m, ZERO_PY_GAMES)
        }
    };
}

/// Fill a game's extension module (called by `engine_module!`).
pub fn init_module(m: &Bound<'_, PyModule>, games: &'static [GameEntry]) -> PyResult<()> {
    for (i, g) in games.iter().enumerate() {
        if games[..i].iter().any(|h| h.name == g.name) {
            return Err(PyValueError::new_err(format!("game {:?} registered twice", g.name)));
        }
    }
    let have = *GAMES.get_or_init(|| games);
    if !std::ptr::eq(have, games) {
        return Err(PyRuntimeError::new_err("zero_py: one extension module per game registry"));
    }
    m.add_class::<PyGame>()?;
    m.add_class::<PySearchTree>()?;
    m.add_class::<PySearchDriver>()?;
    m.add_class::<PySampleRecords>()?;
    m.add_class::<PyLookahead>()?;
    m.add_function(wrap_pyfunction!(lookahead_settings, m)?)?;
    m.add_function(wrap_pyfunction!(driver_settings, m)?)?;
    m.add_function(wrap_pyfunction!(game_names, m)?)?;
    m.add_function(wrap_pyfunction!(game_info_of, m)?)?;
    m.add_function(wrap_pyfunction!(root_floor_of, m)?)?;
    m.add_function(wrap_pyfunction!(view_pick_of, m)?)?;
    m.add_function(wrap_pyfunction!(noresign_pick_of, m)?)?;
    m.add_function(wrap_pyfunction!(belief_weight_of, m)?)?;
    m.add_function(wrap_pyfunction!(gumbel_sequence_of, m)?)?;
    m.add_function(wrap_pyfunction!(completed_q_policy_of, m)?)?;
    m.add_function(wrap_pyfunction!(replay_rows, m)?)?;
    m.add_function(wrap_pyfunction!(game_seed_of, m)?)?;
    m.add_function(wrap_pyfunction!(pack_sparse_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(pack_tail_tokens, m)?)?;
    m.add("MAX_SEATS", MAX_SEATS)?;
    m.add("BELIEF_CLIP", BELIEF_CLIP)?;
    Ok(())
}

fn registered() -> &'static [GameEntry] {
    GAMES.get().copied().unwrap_or(&[])
}

fn entry(name: &str) -> Result<&'static GameEntry, String> {
    registered().iter().find(|g| g.name == name).ok_or_else(|| unknown(name))
}

fn new_game(name: &str, spec: &str, players: usize, seed: u64) -> Result<Box<dyn DynGame>, String> {
    (entry(name)?.new_game)(spec, players, seed)
}

fn new_driver(name: &str, cfg: DriverCfg) -> Result<Box<dyn DynDriver>, String> {
    (entry(name)?.new_driver)(cfg)
}

fn info_of(name: &str) -> Result<GameInfo, String> {
    Ok((entry(name)?.info)())
}

fn unknown(name: &str) -> String {
    let names: Vec<&str> = registered().iter().map(|g| g.name).collect();
    format!("unknown game {name:?} (registered: {names:?})")
}

fn verr(e: String) -> PyErr {
    PyValueError::new_err(e)
}

fn rerr(e: String) -> PyErr {
    PyRuntimeError::new_err(e)
}

/// A spec given as a JSON string, a dict (serialised with `json.dumps`) or None ("{}").
fn spec_json(spec: Option<&Bound<'_, PyAny>>) -> PyResult<String> {
    match spec {
        None => Ok("{}".into()),
        Some(s) if s.is_none() => Ok("{}".into()),
        Some(s) => {
            if let Ok(t) = s.extract::<String>() {
                return Ok(t);
            }
            let json = s.py().import("json")?;
            json.call_method1("dumps", (s,))?.extract()
        }
    }
}

/// A Python value as JSON for the shared settings parser (`zero_search::settings`): None, bool, str,
/// int, float (finite: JSON has no NaN / inf), list / tuple, dict with string keys; an int-like
/// (`__index__`, e.g. numpy's) as an integer, a float-like as a float. Another type is an error.
fn to_json(v: &Bound<'_, PyAny>) -> PyResult<Value> {
    if v.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = v.downcast::<PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }
    if let Ok(s) = v.downcast::<PyString>() {
        return Ok(Value::String(s.to_str()?.to_owned()));
    }
    if let Ok(d) = v.downcast::<PyDict>() {
        let mut m = Map::new();
        for (k, x) in d.iter() {
            let k: String = k.extract().map_err(|_| PyValueError::new_err(format!("a setting's name is a string, not {}", k.repr().map_or_else(|_| "?".into(), |r| r.to_string()))))?;
            m.insert(k, to_json(&x)?);
        }
        return Ok(Value::Object(m));
    }
    if v.downcast::<PyList>().is_ok() || v.downcast::<PyTuple>().is_ok() {
        return v.try_iter()?.map(|x| to_json(&x?)).collect::<PyResult<Vec<_>>>().map(Value::Array);
    }
    let float = |f: f64| serde_json::Number::from_f64(f).map(Value::Number).ok_or_else(|| PyValueError::new_err(format!("setting value {f}: not a finite number (JSON has no NaN / inf)")));
    if v.downcast::<PyFloat>().is_ok() {
        return float(v.extract::<f64>()?);
    }
    if let Ok(i) = v.extract::<i64>() {
        return Ok(Value::from(i));
    }
    if let Ok(u) = v.extract::<u64>() {
        return Ok(Value::from(u));
    }
    if let Ok(f) = v.extract::<f64>() {
        return float(f);
    }
    Err(PyValueError::new_err(format!("setting value {}: not a JSON value (None, bool, str, int, float, list, dict)", v.repr().map_or_else(|_| "?".into(), |r| r.to_string()))))
}

/// `starts` as JSON with each entry's spec as Python gave it to the game since the beginning
/// (`spec_json`: a string as it is, None "{}", a dict `json.dumps`'d - its key order and spacing
/// kept); anything else as it is (the parser says what is wrong).
fn starts_json(d: &Bound<'_, PyDict>) -> PyResult<Value> {
    let mut m = Map::new();
    for (k, x) in d.iter() {
        let k: String = k.extract().map_err(|_| PyValueError::new_err("starts: a setting's name is a string"))?;
        if k == "specs" && x.downcast::<PyString>().is_err() {
            let mut list = Vec::new();
            for e in x.try_iter()? {
                let e = e?;
                // (a tuple or a list of two: the id, the spec)
                let two: Option<Vec<Bound<'_, PyAny>>> = if e.downcast::<PyTuple>().is_ok() || e.downcast::<PyList>().is_ok() { e.try_iter()?.collect::<PyResult<Vec<_>>>().ok() } else { None };
                let pair = two.filter(|v| v.len() == 2).and_then(|v| Some((v[0].extract::<String>().ok()?, v[1].clone())));
                list.push(match pair {
                    Some((id, sp)) => Value::Array(vec![Value::String(id), Value::String(spec_json(Some(&sp))?)]),
                    None => to_json(&e)?,
                });
            }
            m.insert(k, Value::Array(list));
        } else {
            m.insert(k, to_json(&x)?);
        }
    }
    Ok(Value::Object(m))
}

/// A result row's resignation (`ResignInfo`, docs/search.md 3.14) as a dict.
fn resign_dict<'py>(py: Python<'py>, rs: &zero_search::driver::ResignInfo) -> PyResult<Bound<'py, PyDict>> {
    let rd = PyDict::new(py);
    rd.set_item("resigned", rs.resigned)?;
    rd.set_item("noresign", rs.noresign)?;
    rd.set_item("winner", rs.winner)?;
    rd.set_item("decision", rs.decision)?;
    rd.set_item("step", rs.step)?;
    rd.set_item("records", rs.records)?;
    Ok(rd)
}

fn info_dict<'py>(py: Python<'py>, i: &GameInfo) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("name", i.name)?;
    d.set_item("num_actions", i.num_actions)?;
    d.set_item("num_features", i.num_features)?;
    d.set_item("min_players", i.min_players)?;
    d.set_item("max_players", i.max_players)?;
    d.set_item("bots", i.bots.clone())?;
    d.set_item("max_seats", MAX_SEATS)?;
    d.set_item("aux_layout", i.aux.clone())?;
    d.set_item("aux_width", i.aux.iter().map(|s| s.1).sum::<usize>())?;
    // variable-length rows (docs/search.md 1.4): empty / False / None / 0 for a game without them
    let tails = PyList::empty(py);
    for t in &i.tails {
        let td = PyDict::new(py);
        td.set_item("seg", t.seg)?;
        td.set_item("name", t.name)?;
        td.set_item("width", t.width)?;
        td.set_item("head", t.head)?;
        tails.append(td)?;
    }
    d.set_item("tails", tails)?;
    d.set_item("option_pointers", i.option_pointers)?;
    d.set_item("max_row_tokens", i.max_row_tokens)?;
    d.set_item("head_tokens", i.head_tokens)?;
    Ok(d)
}

/// The root minimum visits a search takes (`driver::root_floor`, docs/search.md 3.5): `k` for a
/// full search of `m` options, 2 <= m <= max_actions, 2 m k <= target simulations; else 0.
#[pyfunction]
#[pyo3(name = "root_floor")]
fn root_floor_of(k: u32, max_actions: u32, full: bool, m: usize, target: u32) -> u32 {
    root_floor(k, max_actions, full, m, target)
}

/// Whether recorded decision `decision` of game `gid` (driver seed `seed`) records its value-only
/// samples at share `share` (`driver::view_pick`, docs/search.md 3.6).
#[pyfunction]
#[pyo3(name = "view_pick")]
fn view_pick_of(seed: u64, gid: u32, decision: u32, share: f64) -> bool {
    view_pick(seed, gid, decision, share)
}

/// Whether game `gid` of a driver seeded `seed` is one of the share `share` of games that never resign
/// (`driver::noresign_pick`, docs/search.md 3.14).
#[pyfunction]
#[pyo3(name = "noresign_pick")]
fn noresign_pick_of(seed: u64, gid: u32, share: f64) -> bool {
    noresign_pick(seed, gid, share)
}

/// The weight of a category of hidden cards in a belief-weighted deal (`game::belief_weight`,
/// docs/search.md 3.8): clip((max(pred, 0) / max(uniform, eps))^tilt, 1/clip, clip).
#[pyfunction]
#[pyo3(name = "belief_weight")]
fn belief_weight_of(pred: f64, uniform: f64, tilt: f64, clip: f64) -> f64 {
    zero_search::belief_weight(pred, uniform, tilt, clip)
}

/// Sequential halving's schedule (`tree::gumbel_sequence`, docs/search.md 3.7): per root walk of a
/// search of `n` root walks over `m` considered options, the visits (of that search) an option needs
/// to be taken.
#[pyfunction]
#[pyo3(name = "gumbel_sequence")]
fn gumbel_sequence_of(m: u32, n: u32) -> Vec<u32> {
    zero_search::tree::gumbel_sequence(m, n)
}

/// The completed-Q policy target of one node's statistics (`tree::completed_q_policy_w`,
/// docs/search.md 3.1 / 3.13): priors before noise, visits, value sums, the network's value for the
/// decider, sigma's transform and the prior exponent (1.0 = `completed_q_policy`). The parity tests'
/// handle on the Rust arithmetic.
#[pyfunction]
#[pyo3(name = "completed_q_policy", signature = (p0, n, w, v, c_visit=50.0, c_scale=1.0, prior_w=1.0))]
fn completed_q_policy_of(p0: Vec<f32>, n: Vec<u32>, w: Vec<f64>, v: f64, c_visit: f64, c_scale: f64, prior_w: f64) -> PyResult<Vec<f32>> {
    if n.len() != p0.len() || w.len() != p0.len() {
        return Err(PyValueError::new_err(format!("completed_q_policy: {} priors, {} visit counts, {} value sums", p0.len(), n.len(), w.len())));
    }
    Ok(zero_search::tree::completed_q_policy_w(&p0, &n, &w, v, c_visit, c_scale, prior_w))
}

fn contiguous<T>(r: Result<T, numpy::NotContiguousError>, what: &str) -> PyResult<T> {
    r.map_err(|_| PyValueError::new_err(format!("{what}: a contiguous array")))
}

/// A token network's sparse batch packed for its CUDA graphs, off the GIL (`pack::sparse`; `zero.infer.TokenGraphRunner`,
/// docs/az.md 43.12): the non-zeros' flat indexes and values into `fi` / `val`, each row's tokens into `ntok`
/// (padding rows 1), the token positions into `full` (row stride Tp), the head presence into `pres` when given ->
/// (Tp, the longest row's tokens); Tp = 0 when a row is longer than `long_t` (>= 0).
#[pyfunction]
#[pyo3(signature = (xs_off, xs_idx, xs_val, nfeat, flag_col, base, t_off, long_t, t_step, fi, val, ntok, full, pres=None))]
#[allow(clippy::too_many_arguments)]
fn pack_sparse_tokens(
    py: Python<'_>,
    xs_off: PyReadonlyArray1<'_, i64>,
    xs_idx: PyReadonlyArray1<'_, u32>,
    xs_val: PyReadonlyArray1<'_, f32>,
    nfeat: usize,
    flag_col: PyReadonlyArray1<'_, i64>,
    base: PyReadonlyArray1<'_, bool>,
    t_off: Option<PyReadonlyArray1<'_, i64>>,
    long_t: i64,
    t_step: usize,
    mut fi: PyReadwriteArray1<'_, i64>,
    mut val: PyReadwriteArray1<'_, f32>,
    mut ntok: PyReadwriteArray1<'_, i64>,
    mut full: PyReadwriteArray1<'_, i64>,
    mut pres: Option<PyReadwriteArray1<'_, bool>>,
) -> PyResult<(usize, usize)> {
    let (a, i, v) = (contiguous(xs_off.as_slice(), "xs_off")?, contiguous(xs_idx.as_slice(), "xs_idx")?, contiguous(xs_val.as_slice(), "xs_val")?);
    let (fc, bs) = (contiguous(flag_col.as_slice(), "flag_col")?, contiguous(base.as_slice(), "base")?);
    let t = match &t_off {
        Some(t) => Some(contiguous(t.as_slice(), "t_off")?),
        None => None,
    };
    let (fo, vo) = (contiguous(fi.as_slice_mut(), "fi")?, contiguous(val.as_slice_mut(), "val")?);
    let (no, uo) = (contiguous(ntok.as_slice_mut(), "ntok")?, contiguous(full.as_slice_mut(), "full")?);
    let po = match pres.as_mut() {
        Some(p) => Some(contiguous(p.as_slice_mut(), "pres")?),
        None => None,
    };
    py.allow_threads(move || pack::sparse(a, i, v, nfeat, fc, bs, t, long_t, t_step, fo, vo, no, uo, po)).map_err(verr)
}

/// A batch's tail tokens padded for a graph of `ntp` entries, off the GIL (`pack::tail`, docs/az.md 43.12): rows,
/// segments, slots into `tint` [3 * ntp], the floats into `tval` [ntp * wt], with TokenGraphRunner's checks.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn pack_tail_tokens(
    py: Python<'_>,
    t_off: PyReadonlyArray1<'_, i64>,
    t_seg: PyReadonlyArray1<'_, u16>,
    t_slot: PyReadonlyArray1<'_, u32>,
    t_val: PyReadonlyArray1<'_, f32>,
    width: PyReadonlyArray1<'_, i64>,
    net_tail: PyReadonlyArray1<'_, bool>,
    head: PyReadonlyArray1<'_, i64>,
    wt: usize,
    ntp: usize,
    pad_seg: i64,
    pad_slot: i64,
    mut tint: PyReadwriteArray1<'_, i64>,
    mut tval: PyReadwriteArray1<'_, f32>,
) -> PyResult<()> {
    let (o, s, l, v) = (contiguous(t_off.as_slice(), "t_off")?, contiguous(t_seg.as_slice(), "t_seg")?, contiguous(t_slot.as_slice(), "t_slot")?, contiguous(t_val.as_slice(), "t_val")?);
    let (w, nt, h) = (contiguous(width.as_slice(), "width")?, contiguous(net_tail.as_slice(), "net_tail")?, contiguous(head.as_slice(), "head")?);
    let (ti, tv) = (contiguous(tint.as_slice_mut(), "tint")?, contiguous(tval.as_slice_mut(), "tval")?);
    py.allow_threads(move || pack::tail(o, s, l, v, w, nt, h, wt, ntp, pad_seg, pad_slot, ti, tv)).map_err(verr)
}

/// A lookahead's settings checked (docs/search.md 3.9; a dict or its JSON) -> their canonical JSON
/// (every setting, defaults filled in); a ValueError names a bad or unknown one.
#[pyfunction]
fn lookahead_settings(settings: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(LookCfg::parse(&spec_json(Some(settings))?).map_err(verr)?.to_json())
}

/// Registered game names.
#[pyfunction]
fn game_names() -> Vec<&'static str> {
    registered().iter().map(|g| g.name).collect()
}

/// {name, num_actions, num_features, min_players, max_players, bots, max_seats, aux_layout
/// ([(segment, width)]: `SearchGame::AUX`), aux_width}.
#[pyfunction]
fn game_info_of<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyDict>> {
    info_dict(py, &info_of(name).map_err(verr)?)
}

// ------------------------------------------------------------------------ Game

/// One game of any registered type: `Game('toy', spec=None, players=2, seed=0)`.
#[pyclass(name = "Game")]
pub struct PyGame {
    g: Box<dyn DynGame>,
}

#[pymethods]
impl PyGame {
    #[new]
    #[pyo3(signature = (game, spec=None, players=2, seed=0))]
    fn new(game: &str, spec: Option<&Bound<'_, PyAny>>, players: usize, seed: u64) -> PyResult<Self> {
        let s = spec_json(spec)?;
        Ok(PyGame { g: new_game(game, &s, players, seed).map_err(verr)? })
    }
    fn clone(&self) -> PyGame {
        PyGame { g: self.g.clone_box() }
    }
    fn __copy__(&self) -> PyGame {
        self.clone()
    }
    #[getter]
    fn name(&self) -> &'static str {
        self.g.info().name
    }
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        info_dict(py, &self.g.info())
    }
    #[getter]
    fn n_players(&self) -> usize {
        self.g.num_players()
    }
    /// The seat to decide, None when the game is over.
    #[getter]
    fn to_move(&self) -> Option<usize> {
        self.g.to_move()
    }
    #[getter]
    fn over(&self) -> bool {
        self.g.to_move().is_none()
    }
    fn legal(&self) -> Vec<u16> {
        self.g.legal()
    }
    /// Apply a legal action -> the chance outcome keys that occurred.
    fn step(&mut self, a: u16) -> PyResult<Vec<u64>> {
        self.g.step(a).map_err(verr)
    }
    fn force_chance(&mut self, keys: Vec<u64>) {
        self.g.force_chance(&keys)
    }
    /// The result per seat (once over); refused at a deferred terminal before `resolve`.
    fn result(&self) -> PyResult<Vec<f32>> {
        self.not_deferred("result")?;
        Ok(self.g.result())
    }
    fn margin(&self, seat: usize) -> PyResult<f32> {
        self.not_deferred("margin")?;
        Ok(self.g.margin(seat))
    }
    /// Over at a deferred terminal, waiting for `resolve` (docs/search.md 1.1).
    #[getter]
    fn deferred(&self) -> bool {
        self.g.deferred()
    }
    /// Over without a result (`SearchGame::aborted`): no result, margin or samples.
    #[getter]
    fn aborted(&self) -> bool {
        self.g.aborted()
    }
    /// What must be played to resolve the deferred terminal (JSON string).
    fn deferred_payload(&self) -> PyResult<String> {
        self.g.deferred_payload().map_err(verr)
    }
    /// The deferred terminal's outcome from its episodes' results: a JSON string, or a list
    /// (serialised with `json.dumps`).
    fn resolve(&mut self, results: &Bound<'_, PyAny>) -> PyResult<()> {
        let r = spec_json(Some(results))?;
        self.g.resolve(&r).map_err(verr)
    }
    /// Seats per team (seat s is in team s % (n_players / team_size)).
    #[getter]
    fn team_size(&self) -> usize {
        self.g.team_size()
    }
    fn score(&self, seat: usize) -> f32 {
        self.g.score(seat)
    }
    fn scores(&self) -> Vec<f32> {
        (0..self.g.num_players()).map(|i| self.g.score(i)).collect()
    }
    #[getter]
    fn rounds(&self) -> u32 {
        self.g.rounds()
    }
    /// The auxiliary targets of the pending decision for `viewer` (f32 [aux_width], from the true
    /// state: a training label, `SearchGame::aux_target`; docs/search.md 1.2).
    fn aux_target<'py>(&self, py: Python<'py>, viewer: usize) -> PyResult<Bound<'py, numpy::PyArray1<f32>>> {
        if viewer >= self.g.num_players() {
            return Err(PyValueError::new_err(format!("viewer {viewer}: the game has {} players", self.g.num_players())));
        }
        Ok(self.g.aux_target(viewer).into_pyarray(py))
    }
    fn determinize(&mut self, viewer: usize, seed: u64) {
        self.g.determinize(viewer, seed)
    }
    /// `determinize` dealt by a belief (`SearchGame::determinize_with`, docs/search.md 3.8): `belief`
    /// = the predictions of the game's auxiliary targets for `viewer` (f32 [aux_width]), `tilt` > 0,
    /// `clip` >= 1.
    /// `tilts` = per-target tilts {target: alpha} (docs/search.md 3.8.1; None: none - then `tilt` > 0).
    #[pyo3(signature = (viewer, seed, belief, tilt=1.0, clip=BELIEF_CLIP, tilts=None))]
    fn determinize_with(&mut self, viewer: usize, seed: u64, belief: PyReadonlyArray1<'_, f32>, tilt: f64, clip: f64, tilts: Option<std::collections::BTreeMap<String, f64>>) -> PyResult<()> {
        let p = belief.as_slice().map_err(|e| PyValueError::new_err(format!("belief: {e}")))?;
        let t: Vec<(String, f64)> = tilts.unwrap_or_default().into_iter().collect();
        self.g.determinize_with(viewer, seed, p, tilt, clip, &t).map_err(verr)
    }
    /// The head (`NUM_FEATURES` floats); ValueError for a state whose row has a tail (R1: use `encode_row`).
    fn encode<'py>(&self, py: Python<'py>, viewer: usize) -> PyResult<Bound<'py, numpy::PyArray1<f32>>> {
        Ok(self.g.encode(viewer).map_err(verr)?.into_pyarray(py))
    }
    /// (head f32 [F], {'t_seg': u16 [NT], 't_slot': u32 [NT], 't_val': f32 [...]}): the row with its sealed tail
    /// (docs/search.md 1.4).
    fn encode_row<'py>(&self, py: Python<'py>, viewer: usize) -> PyResult<(Bound<'py, numpy::PyArray1<f32>>, Bound<'py, PyDict>)> {
        let (x, t) = self.g.encode_row(viewer);
        Ok((x.into_pyarray(py), tail_dict(py, t)?))
    }
    /// (ov_opt int32 [NO], ov_tok int64 [NO]): `viewer`'s pointer overrides against `legal()` (docs/search.md 2.15).
    fn option_overrides<'py>(&self, py: Python<'py>, viewer: usize) -> (Bound<'py, numpy::PyArray1<i32>>, Bound<'py, numpy::PyArray1<i64>>) {
        ov_arrays(py, &self.g.option_overrides(viewer))
    }
    fn export_state(&self) -> String {
        self.g.export_state()
    }
    fn apply_external(&mut self, payload: &str) -> PyResult<Vec<u64>> {
        self.g.apply_external(payload).map_err(verr)
    }
    /// The game chains its episodes (`SearchGame::chained`, docs/search.md 1.3): an episode ends where it is over, and
    /// `next_episode` says how the contest goes on.
    #[getter]
    fn chained(&self) -> bool {
        self.g.chained()
    }
    /// A chained game's next episode from this ended one (`conceded`: the seat that resigned it, at a decision in
    /// progress; None: it is over) -> (the next episode's game, None), or (None, the contest's result per seat).
    #[pyo3(signature = (conceded=None))]
    fn next_episode(&self, conceded: Option<usize>) -> PyResult<(Option<PyGame>, Option<Vec<f32>>)> {
        let (g, r) = self.g.next_episode(conceded).map_err(verr)?;
        Ok((g.map(|g| PyGame { g }), r))
    }
    /// The self-play resign rule may count the pending decision (`SearchGame::can_concede`).
    #[getter]
    fn can_concede(&self) -> bool {
        self.g.can_concede()
    }
    /// A Rust search tree for the pending decision.
    #[pyo3(signature = (redet=true, seed=0))]
    fn search_tree(&self, redet: bool, seed: u64) -> PyResult<PySearchTree> {
        Ok(PySearchTree { t: self.g.new_tree(redet, seed).map_err(verr)? })
    }
}

/// A sealed row tail as `Game.encode_row`'s dict {'t_seg' u16, 't_slot' u32, 't_val' f32} (docs/search.md 2.15).
fn tail_dict(py: Python<'_>, t: TailRow) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("t_seg", t.seg.into_pyarray(py))?;
    d.set_item("t_slot", t.slot.into_pyarray(py))?;
    d.set_item("t_val", t.val.into_pyarray(py))?;
    Ok(d)
}

/// A row's pointer overrides as `Game.option_overrides`' (ov_opt int32 [NO], ov_tok int64 [NO]).
fn ov_arrays<'py>(py: Python<'py>, ov: &[(u32, i64)]) -> (Bound<'py, numpy::PyArray1<i32>>, Bound<'py, numpy::PyArray1<i64>>) {
    let opt: Vec<i32> = ov.iter().map(|o| o.0 as i32).collect();
    let tok: Vec<i64> = ov.iter().map(|o| o.1).collect();
    (opt.into_pyarray(py), tok.into_pyarray(py))
}

impl PyGame {
    fn not_deferred(&self, what: &str) -> PyResult<()> {
        if self.g.deferred() {
            return Err(PyValueError::new_err(format!(
                "{}: {what}() at a deferred terminal: the result comes from its episodes (deferred_payload, then resolve)",
                self.g.info().name
            )));
        }
        Ok(())
    }
}

// ------------------------------------------------------------------------ SearchTree

/// One Rust search driven from Python (parity tests, mixed matches): `game.search_tree()`.
#[pyclass(name = "SearchTree")]
pub struct PySearchTree {
    t: Box<dyn DynTree>,
}

#[pymethods]
impl PySearchTree {
    fn set_base(&mut self, game: &PyGame) -> PyResult<()> {
        self.t.set_base(&*game.g).map_err(verr)
    }
    /// One simulation: None if it ended in a terminal state (backed up), else
    /// (decider, features, legal actions) of the leaf - call `expand` with its evaluation.
    /// `root_view` (docs/search.md 3.6): a 4-tuple, the 4th the leaf's features from the tree's
    /// viewer (the root decider) when another seat decides it, else None; `expand` then takes the
    /// value and margin of those features (relative to the viewer), the priors of the decider's.
    /// `rows` (docs/search.md 2.15): a 5-tuple (decider, features, legal actions, the viewer's features or None,
    /// rows), `rows` None for a game without tail segments or pointer overrides, else (tail, ov, vtail): the
    /// decider's row's tail {'t_seg', 't_slot', 't_val'} (`Game.encode_row`'s; None without tail segments), its
    /// overrides (ov_opt int32, ov_tok int64) against the leaf's options (`Game.option_overrides`'; None without
    /// pointer overrides) and the viewer's value-only row's tail (None without one). Required for a game with
    /// them (RuntimeError otherwise: a head-only caller never drops them).
    #[pyo3(signature = (game, c_puct, chance_k=0, margin_w=0.0, root_view=false, rows=false))]
    #[allow(clippy::too_many_arguments)]
    fn simulate<'py>(&mut self, py: Python<'py>, game: &PyGame, c_puct: f64, chance_k: u32, margin_w: f64, root_view: bool, rows: bool) -> PyResult<Option<Bound<'py, pyo3::types::PyTuple>>> {
        match self.t.simulate(&*game.g, c_puct, chance_k, margin_w, root_view, rows).map_err(rerr)? {
            None => Ok(None),
            Some(l) if rows => {
                let r: Option<PyObject> = match l.rows {
                    None => None,
                    Some(lr) => {
                        let tail = lr.tail.map(|t| tail_dict(py, t)).transpose()?;
                        let ov = lr.ov.map(|ov| ov_arrays(py, &ov));
                        let vtail = lr.vtail.map(|t| tail_dict(py, t)).transpose()?;
                        Some((tail, ov, vtail).into_pyobject(py)?.into_any().unbind())
                    }
                };
                Ok(Some((l.player, l.x.into_pyarray(py), l.acts, l.vx.map(|v| v.into_pyarray(py)), r).into_pyobject(py)?))
            }
            Some(l) if root_view => Ok(Some((l.player, l.x.into_pyarray(py), l.acts, l.vx.map(|v| v.into_pyarray(py))).into_pyobject(py)?)),
            Some(l) => Ok(Some((l.player, l.x.into_pyarray(py), l.acts).into_pyobject(py)?)),
        }
    }
    /// Expand the last leaf: priors over its options, value / margin relative to its decider (to
    /// the tree's viewer when `simulate(root_view=True)` gave its features).
    #[pyo3(signature = (priors, value, margin, margin_w=0.0))]
    fn expand(&mut self, priors: Vec<f32>, value: Vec<f32>, margin: Vec<f32>, margin_w: f64) -> PyResult<()> {
        self.t.expand(&priors, &value, &margin, margin_w).map_err(rerr)
    }
    /// The tree's viewer (the root decider whose information set it searches).
    #[getter]
    fn viewer(&self) -> usize {
        self.t.viewer()
    }
    #[pyo3(signature = (alpha, eps, seed=None))]
    fn add_noise(&mut self, alpha: f64, eps: f64, seed: Option<u64>) {
        self.t.add_noise(alpha, eps, seed)
    }
    fn advance(&mut self, a: u16, key: Vec<u64>, game: &PyGame) -> PyResult<bool> {
        self.t.advance(a, &key, &*game.g).map_err(verr)
    }
    #[getter]
    fn root_acts(&self) -> Vec<u16> {
        self.t.root_acts()
    }
    #[getter]
    fn root_player(&self) -> usize {
        self.t.root_player()
    }
    #[getter]
    fn root_visits(&self) -> Vec<u32> {
        self.t.root_visits()
    }
    #[getter]
    fn root_priors(&self) -> Vec<f32> {
        self.t.root_priors()
    }
    #[getter]
    fn root_priors0(&self) -> Vec<f32> {
        self.t.root_priors0()
    }
    #[getter]
    fn root_values(&self) -> Vec<f64> {
        self.t.root_values()
    }
    #[getter]
    fn root_tot(&self) -> u32 {
        self.t.root_tot()
    }
    #[getter]
    fn root_expanded(&self) -> bool {
        self.t.root_expanded()
    }
    #[getter]
    fn redet(&self) -> bool {
        self.t.redet()
    }
    fn root_win(&self) -> Option<Vec<f64>> {
        self.t.root_win()
    }
    /// The network's utility of the root for its decider (stored at expansion).
    #[getter]
    fn root_value0(&self) -> f64 {
        self.t.root_value0()
    }
    /// The completed-Q policy target at the root (docs/search.md 3.1; [] before expansion), with
    /// the prior exponent `prior_w` (docs/search.md 3.13; 1.0 = the target of before).
    #[pyo3(signature = (c_visit=50.0, c_scale=1.0, prior_w=1.0))]
    fn root_cq_target(&self, c_visit: f64, c_scale: f64, prior_w: f64) -> Vec<f32> {
        self.t.root_cq_target(c_visit, c_scale, prior_w)
    }
    /// The option index a driver's move rule chooses at the root (`select` 'visits' | 'cq' with
    /// the guard `min_visits` and the completed-Q transform; docs/search.md 3.3): the driver's
    /// own code (`Select::choose`), for the Python driver's 'rs' seats and the parity tests.
    #[pyo3(signature = (select="visits", min_visits=0, c_visit=50.0, c_scale=1.0))]
    fn root_select(&self, select: &str, min_visits: u32, c_visit: f64, c_scale: f64) -> PyResult<usize> {
        let sel = Select::parse(select, min_visits, c_visit, c_scale).map_err(verr)?;
        Ok(self.t.root_select(&sel))
    }
    /// Root minimum visits from now on (docs/search.md 3.5; 0 = off; clears the forced walks'
    /// accounting): the floor of the driver's full searches (`root_floor`).
    fn set_root_min(&mut self, k: u32) {
        self.t.set_root_min(k)
    }
    /// The root's belief from now on (docs/search.md 3.8): `belief` = the network's predictions of the
    /// game's auxiliary targets for the root decider (f32 [aux_width]; None: uniform worlds, the
    /// search of before), `tilt` > 0, `clip`, `mix` (the share of simulations dealt uniformly);
    /// `game`: the live game (a tree without redet deals its world again). `advance` clears it.
    /// `tilts`: per-target tilts {target: alpha} (docs/search.md 3.8.1; None: none).
    #[pyo3(signature = (belief, game, tilt=1.0, clip=BELIEF_CLIP, mix=0.0, tilts=None))]
    fn set_belief(&mut self, belief: Option<PyReadonlyArray1<'_, f32>>, game: &PyGame, tilt: f64, clip: f64, mix: f64, tilts: Option<std::collections::BTreeMap<String, f64>>) -> PyResult<()> {
        let p = match &belief {
            Some(b) => Some(b.as_slice().map_err(|e| PyValueError::new_err(format!("belief: {e}")))?),
            None => None,
        };
        let t: Vec<(String, f64)> = tilts.unwrap_or_default().into_iter().collect();
        self.t.set_belief(p, tilt, clip, mix, &t, &*game.g).map_err(verr)
    }
    /// The current search's belief predictions (None: uniform worlds).
    #[getter]
    fn belief(&self) -> Option<Vec<f32>> {
        self.t.belief()
    }
    #[getter]
    fn root_min(&self) -> u32 {
        self.t.root_min()
    }
    /// Per root option: this search's forced walks ([] when the floor is off).
    #[getter]
    fn root_forced(&self) -> Vec<u32> {
        self.t.root_forced()
    }
    /// The root's visit counts without the forced walks (= root_visits when the floor is off).
    #[getter]
    fn root_visits_pruned(&self) -> Vec<u32> {
        self.t.root_visits_pruned()
    }
    /// The root's search win vector without the forced walks (None: no other backup).
    fn root_win_pruned(&self) -> Option<Vec<f64>> {
        self.t.root_win_pruned()
    }
    /// The Gumbel root search from now on (docs/search.md 3.7): sequential halving over `min(m,
    /// options)` root options for `budget` root walks, `noise` the Gumbel noise per root option
    /// (None or []: none), sigma's completed-Q transform `c_visit` / `c_scale`. Off: `clear_gumbel`.
    #[pyo3(signature = (m=16, budget=64, noise=None, c_visit=50.0, c_scale=1.0))]
    fn set_gumbel(&mut self, m: u32, budget: u32, noise: Option<Vec<f64>>, c_visit: f64, c_scale: f64) -> PyResult<()> {
        self.t.set_gumbel(m, budget, &noise.unwrap_or_default(), c_visit, c_scale).map_err(verr)
    }
    fn clear_gumbel(&mut self) {
        self.t.clear_gumbel()
    }
    #[getter]
    fn gumbel_on(&self) -> bool {
        self.t.gumbel_on()
    }
    /// The Gumbel search's move (an option index; None when it is off): the driver's own code.
    fn gumbel_choose(&self) -> Option<usize> {
        self.t.gumbel_choose()
    }
    /// The Gumbel search's root value: the improved policy's win vector over the options this search
    /// walked (absolute seats; None before any walk or when off).
    fn gumbel_q(&self) -> Option<Vec<f64>> {
        self.t.gumbel_q()
    }
    /// Per root option: this Gumbel search's walks ([] when off).
    #[getter]
    fn gumbel_visits(&self) -> Vec<u32> {
        self.t.gumbel_visits()
    }
    #[getter]
    fn gumbel_noise(&self) -> Vec<f64> {
        self.t.gumbel_noise()
    }
    #[getter]
    fn gumbel_schedule(&self) -> Vec<u32> {
        self.t.gumbel_schedule()
    }
    /// Per root option: the win vectors this Gumbel search's walks backed up, summed ([] when off).
    #[getter]
    fn gumbel_win(&self) -> Vec<Vec<f64>> {
        self.t.gumbel_win()
    }
    /// Children of root option i: [(chance key, decider)].
    fn edge_kids(&self, i: usize) -> Vec<(Vec<u64>, usize)> {
        self.t.edge_kids(i)
    }
    fn edge_outs(&self, i: usize) -> Vec<Vec<u64>> {
        self.t.edge_outs(i)
    }
    #[getter]
    fn n_nodes(&self) -> usize {
        self.t.n_nodes()
    }
    /// The tree's shape under its root (a read-only diagnostic, docs/search.md 3.15): a dict of
    /// nodes, expanded, depth_hist (expanded nodes per depth, root 0), opp_hist / switch_hist (expanded nodes
    /// by the other seat's nodes / the decider changes above them), max_depth, edges and main_line
    /// (each a list of (depth, own, visits, kids, opp_kids, expanded_kids, opp_lists, opp_keys, opp_tot,
    /// opp_modal_tot): the other-seat children's distinct option lists and chance keys, their backups and
    /// those of the most backed-up option list).
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.t.shape();
        let d = PyDict::new(py);
        #[allow(clippy::type_complexity)]
        let e = |v: &[zero_search::tree::EdgeShape]| -> Vec<(u32, bool, u32, u32, u32, u32, u32, u32, u32, u32)> { v.iter().map(|x| (x.depth, x.own, x.n, x.kids, x.opp_kids, x.expanded_kids, x.opp_lists, x.opp_keys, x.opp_tot, x.opp_modal_tot)).collect() };
        d.set_item("nodes", s.nodes)?;
        d.set_item("expanded", s.expanded)?;
        d.set_item("depth_hist", s.depth_hist.clone())?;
        d.set_item("opp_hist", s.opp_hist.clone())?;
        d.set_item("switch_hist", s.switch_hist.clone())?;
        d.set_item("max_depth", s.max_depth)?;
        d.set_item("edges", e(&s.edges))?;
        d.set_item("main_line", e(&s.main_line))?;
        Ok(d)
    }
    /// The number of (node, option) pairs the last `simulate` walked (a diagnostic: its leaf's or
    /// terminal's depth, root 0).
    #[getter]
    fn path_len(&self) -> usize {
        self.t.path_len()
    }
    /// Information-set merging from now on (docs/search.md 3.16; a fresh tree only: right after
    /// `game.search_tree()`, before its first simulation). Use it with `simulate(root_view=True)`: an other
    /// seat's merged node takes the viewer's value. A leaf may then be a growth leaf (an expanded node of
    /// another seat whose world has a new option): `simulate` gives its world's options, `expand` takes them.
    fn set_merge(&mut self, on: bool) -> PyResult<()> {
        self.t.set_merge(on).map_err(verr)
    }
    #[getter]
    fn merge_on(&self) -> bool {
        self.t.merge_on()
    }
    /// The merged tree's counters since `set_merge` (None when off): links (new child signatures), encodes
    /// (observations encoded), merged (new signatures that found a node by its observation), aliases (fresh
    /// nodes for a walk that would revisit one), growths (growth leaves).
    fn merge_stats<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(m) = self.t.merge_stats() else { return Ok(None) };
        let d = PyDict::new(py);
        d.set_item("links", m.links)?;
        d.set_item("encodes", m.encodes)?;
        d.set_item("merged", m.merged)?;
        d.set_item("aliases", m.aliases)?;
        d.set_item("growths", m.growths)?;
        Ok(Some(d))
    }
}

// ------------------------------------------------------------------------ SearchDriver

/// The `DriverCfg` of `SearchDriver(game, seats, ...)`'s arguments (the seat dicts parsed, the
/// distinct seats de-duplicated into one table).
#[allow(clippy::too_many_arguments)]
fn driver_cfg(seats: &Bound<'_, PyList>, spec: Option<&Bound<'_, PyAny>>, concurrency: usize, seed: u64, record: bool, threads: usize, groups: usize, max_steps: u64, rolling: bool, net: u32, queue_only: bool, move_log: u32, aux: bool, paired: bool, sparse: bool, branch: Option<&Bound<'_, PyDict>>, starts: Option<&Bound<'_, PyDict>>, max_row_tokens: Option<usize>) -> PyResult<DriverCfg> {
    let spec = spec_json(spec)?;
    let mut games = Vec::with_capacity(seats.len());
    for g in seats.iter() {
        let row: Vec<Bound<'_, PyDict>> = g.extract()?;
        games.push(row.iter().map(|s| to_json(s.as_any())).collect::<PyResult<Vec<_>>>()?);
    }
    let branch = branch.map(|b| to_json(b.as_any())).transpose()?;
    let starts = starts.map(starts_json).transpose()?;
    let args = DriverArgs { spec, concurrency, seed, record, threads, groups, max_steps, rolling, net, queue_only, move_log, aux, paired, sparse, max_row_tokens };
    settings::driver_cfg(&games, branch.as_ref(), starts.as_ref(), args).map_err(verr)
}

/// The driver settings `SearchDriver(game, seats, ...)` would be built with, as the Rust `DriverCfg`'s
/// pretty Debug text (a diagnostic: no driver is made, the game is not checked): the seat dicts parsed,
/// the distinct seats in one table. zero-farm's settings tools compare it across parsers and with the
/// native worker's (docs/az.md 44).
#[pyfunction]
#[pyo3(signature = (seats, spec=None, concurrency=64, seed=0, record=false, threads=4, groups=2, max_steps=100_000, rolling=false, net=0, queue_only=false, move_log=0, aux=false, paired=false, sparse=false, branch=None, starts=None, max_row_tokens=None))]
#[allow(clippy::too_many_arguments)]
fn driver_settings(seats: &Bound<'_, PyList>, spec: Option<&Bound<'_, PyAny>>, concurrency: usize, seed: u64, record: bool, threads: usize, groups: usize, max_steps: u64, rolling: bool, net: u32, queue_only: bool, move_log: u32, aux: bool, paired: bool, sparse: bool, branch: Option<&Bound<'_, PyDict>>, starts: Option<&Bound<'_, PyDict>>, max_row_tokens: Option<usize>) -> PyResult<String> {
    let cfg = driver_cfg(seats, spec, concurrency, seed, record, threads, groups, max_steps, rolling, net, queue_only, move_log, aux, paired, sparse, branch, starts, max_row_tokens)?;
    Ok(format!("{cfg:#?}"))
}

/// The batched driver (docs/search.md 2). `seats` = per game a list of seat dicts; `aux`: record
/// the game's auxiliary targets with the samples (`take_samples()['aux']`, docs/search.md 1.2);
/// `paired`: games 2k and 2k + 1 share every random stream (the deal), `DriverCfg::paired`;
/// `sparse`: network batches carry each row's non-zero features (`xs_off`, `xs_idx`, `xs_val`,
/// `n_features`) instead of `x` (`DriverCfg::sparse`, docs/search.md 2.2); `branch` {p, moves, temp}:
/// branched games, `starts` {p, specs}: archive starts (docs/search.md 2.11; None: off).
#[pyclass(name = "SearchDriver")]
pub struct PySearchDriver {
    d: Box<dyn DynDriver>,
    nfeat: usize,
    /// The game declares tail segments / pointer overrides (docs/search.md 2.15): the batches, samples and stats
    /// carry their keys.
    rows: RowKinds,
}

/// Which variable-length arrays a game's batches and samples carry (docs/search.md 2.15).
#[derive(Clone, Copy)]
struct RowKinds {
    tails: bool,
    pointers: bool,
}

impl RowKinds {
    fn of(i: &GameInfo) -> RowKinds {
        RowKinds { tails: !i.tails.is_empty(), pointers: i.option_pointers }
    }
}

#[pymethods]
impl PySearchDriver {
    #[new]
    #[pyo3(signature = (game, seats, spec=None, concurrency=64, seed=0, record=false, threads=4, groups=2, max_steps=100_000, rolling=false, net=0, queue_only=false, move_log=0, aux=false, paired=false, sparse=false, branch=None, starts=None, max_row_tokens=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(game: &str, seats: &Bound<'_, PyList>, spec: Option<&Bound<'_, PyAny>>, concurrency: usize, seed: u64, record: bool, threads: usize, groups: usize, max_steps: u64, rolling: bool, net: u32, queue_only: bool, move_log: u32, aux: bool, paired: bool, sparse: bool, branch: Option<&Bound<'_, PyDict>>, starts: Option<&Bound<'_, PyDict>>, max_row_tokens: Option<usize>) -> PyResult<Self> {
        let cfg = driver_cfg(seats, spec, concurrency, seed, record, threads, groups, max_steps, rolling, net, queue_only, move_log, aux, paired, sparse, branch, starts, max_row_tokens)?;
        let d = new_driver(game, cfg).map_err(verr)?;
        let info = d.info();
        Ok(PySearchDriver { d, nfeat: info.num_features, rows: RowKinds::of(&info) })
    }

    /// (id, key, kind, arrays) with kind 'nn' | 'ext', or None when every game has finished
    /// (rolling: the phase ended). Releases the GIL while it waits for the search threads.
    fn next_batch(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let d = &mut self.d;
        let r = py.allow_threads(|| d.next_batch(true)).map_err(rerr)?;
        self.to_py(py, r)
    }

    /// `next_batch` without waiting (None also when the next batch is not ready).
    fn poll_batch(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let r = self.d.next_batch(false).map_err(rerr)?;
        self.to_py(py, r)
    }

    /// Network evaluation of batch `id`: priors f32 [options], value / margin f32 [B, MAX_SEATS];
    /// `aux` f32 [belief rows, aux_width]: the auxiliary outputs of its root belief rows (`extra` ==
    /// ROW_BELIEF, docs/search.md 3.8), in row order - required when it has some.
    /// Releases the GIL while it hands the answers over (docs/az.md 43.12).
    #[pyo3(signature = (id, priors, value, margin, aux=None))]
    fn submit(&mut self, py: Python<'_>, id: u64, priors: PyReadonlyArray1<'_, f32>, value: PyReadonlyArray2<'_, f32>, margin: PyReadonlyArray2<'_, f32>, aux: Option<PyReadonlyArray2<'_, f32>>) -> PyResult<()> {
        if value.shape()[1] != MAX_SEATS || margin.shape()[1] != MAX_SEATS {
            return Err(PyValueError::new_err(format!("value / margin must be [B, {MAX_SEATS}]")));
        }
        let p = priors.as_slice().map_err(|e| PyValueError::new_err(format!("priors: {e}")))?;
        let v = value.as_slice().map_err(|e| PyValueError::new_err(format!("value: {e}")))?;
        let m = margin.as_slice().map_err(|e| PyValueError::new_err(format!("margin: {e}")))?;
        let a = match &aux {
            Some(a) => a.as_slice().map_err(|e| PyValueError::new_err(format!("aux: {e}")))?,
            None => &[],
        };
        let d = &mut self.d;
        py.allow_threads(move || d.submit(id, p, v, m, a)).map_err(verr)
    }

    /// Answers of external batch `id`: one payload string per request.
    fn submit_external(&mut self, id: u64, payloads: Vec<String>) -> PyResult<()> {
        self.d.submit_external(id, payloads).map_err(verr)
    }

    #[getter]
    fn finished(&self) -> bool {
        self.d.finished()
    }
    #[getter]
    fn paused(&self) -> bool {
        self.d.paused()
    }
    fn resume(&mut self, quota: u64, net: u32) -> PyResult<()> {
        self.d.resume(quota, net).map_err(rerr)
    }
    /// Rolling mode, between phases: the games started from now on play `spec` (docs/search.md 2.13).
    fn set_spec(&mut self, spec: &str) -> PyResult<()> {
        self.d.set_spec(spec).map_err(rerr)
    }
    fn end_phase(&mut self) {
        self.d.end_phase()
    }
    /// Games parked at a deferred terminal since the last call: [(gid, payload JSON)].
    fn take_deferred(&mut self) -> Vec<(u32, String)> {
        self.d.take_deferred()
    }
    /// The outcome of parked game `gid` from its episodes' results (a JSON string, or a list
    /// serialised with `json.dumps`): its result row and samples are emitted.
    fn resolve(&mut self, gid: u32, results: &Bound<'_, PyAny>) -> PyResult<()> {
        let r = spec_json(Some(results))?;
        self.d.resolve(gid, &r).map_err(verr)
    }
    /// Games parked at a deferred terminal.
    #[getter]
    fn pending(&self) -> usize {
        self.d.pending()
    }
    /// Queue a setup: the next game a free slot starts uses `spec` (a JSON string or a dict) and
    /// its result row carries `tag`.
    fn enqueue(&mut self, tag: &str, spec: &Bound<'_, PyAny>) -> PyResult<()> {
        let s = spec_json(Some(spec))?;
        self.d.enqueue(tag, &s).map_err(verr)
    }
    /// Setups queued and not started yet.
    #[getter]
    fn queued(&self) -> usize {
        self.d.queued()
    }
    /// Evaluator keys in index order: [(name, external)].
    #[getter]
    fn keys(&self) -> Vec<(String, bool)> {
        self.d.keys()
    }
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        info_dict(py, &self.d.info())
    }

    /// Finished-game rows (docs/search.md 2.5).
    fn take_results<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for r in self.d.take_results() {
            out.append(result_row(py, &r)?)?;
        }
        Ok(out)
    }

    /// Trace every game (`Driver::set_trace`, docs/search.md 2.17): before the first resume.
    fn set_trace(&mut self, on: bool) -> PyResult<()> {
        self.d.set_trace(on).map_err(verr)
    }

    /// A traced driver's finished games (docs/search.md 2.17): take_results and take_samples at one point, with each
    /// game's record (`zero_search::record::game_json`, JSON text) -> (rows, records, samples dict or None).
    fn take_game_records<'py>(&mut self, py: Python<'py>) -> PyResult<(Bound<'py, PyList>, Vec<String>, Option<PyObject>)> {
        let rows = self.d.take_results();
        let recs = self.d.take_samples();
        let out = PyList::empty(py);
        let mut texts = Vec::with_capacity(rows.len());
        let mut a = 0usize;
        for r in &rows {
            let b = a + r.samples as usize;
            if b > recs.len() {
                return Err(rerr(format!("game {} has {} samples, {} left", r.gid, r.samples, recs.len() - a)));
            }
            texts.push(zero_search::record::game_json(r, &recs[a..b]).map_err(rerr)?.to_string());
            out.append(result_row(py, r)?)?;
            a = b;
        }
        if a != recs.len() {
            return Err(rerr(format!("{} samples left over after the finished games", recs.len() - a)));
        }
        let s = samples_dict(py, recs, self.nfeat, self.rows)?;
        Ok((out, texts, s))
    }

    /// Training sample arrays (docs/search.md 2.4) or None.
    fn take_samples(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        samples_dict(py, self.d.take_samples(), self.nfeat, self.rows)
    }

    /// The training records as the driver holds them, or None (docs/search.md 2.4): take_samples
    /// without its conversion - the records move out (no copy), and `SampleRecords.arrays()` turns
    /// them into take_samples' arrays later, on any thread (zero/data.py ThreadedShardWriter: the
    /// thread that drives the search never pays the copy; docs/az.md 31).
    fn take_records(&mut self) -> Option<PySampleRecords> {
        let recs = self.d.take_samples();
        if recs.is_empty() {
            return None;
        }
        Some(PySampleRecords::new(recs, self.nfeat, self.rows))
    }

    /// Throughput counters.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.d.stats();
        let d = PyDict::new(py);
        d.set_item("sims", s.sims)?;
        d.set_item("leaves", s.leaves)?;
        d.set_item("ext_requests", s.ext_requests)?;
        d.set_item("batches", s.batches)?;
        d.set_item("busy_s", s.busy_s)?;
        d.set_item("assemble_s", s.assemble_s)?;
        d.set_item("wait_s", s.wait_s)?;
        d.set_item("wall_s", s.wall_s)?;
        d.set_item("threads", s.threads)?;
        d.set_item("groups", s.groups)?;
        d.set_item("games_started", s.games_started)?;
        d.set_item("games_finished", s.games_finished)?;
        d.set_item("phase_games", s.phase_games)?;
        d.set_item("quota", s.quota)?;
        d.set_item("dropped_leaves", s.dropped_leaves)?;
        d.set_item("restart_sims", s.restart_sims)?;
        d.set_item("restarts", s.restarts)?;
        d.set_item("recorded", s.recorded)?;
        d.set_item("minor", s.minor)?;
        d.set_item("view_rows", s.view_rows)?;
        d.set_item("view_recorded", s.view_recorded)?;
        d.set_item("belief_rows", s.belief_rows)?;
        d.set_item("cache_hits", s.cache_hits)?;
        d.set_item("cache_leaves", s.cache_leaves)?;
        d.set_item("cache_partial", s.cache_partial)?;
        d.set_item("cache_clears", s.cache_clears)?;
        d.set_item("merge_links", s.merge_links)?;
        d.set_item("merge_merged", s.merge_merged)?;
        d.set_item("merge_aliases", s.merge_aliases)?;
        d.set_item("merge_growths", s.merge_growths)?;
        d.set_item("merge_shared", s.merge_shared)?;
        d.set_item("emitted", s.emitted)?;
        d.set_item("epoch", s.epoch)?;
        d.set_item("net", s.net)?;
        d.set_item("parked", s.parked)?;
        d.set_item("parked_bytes", s.parked_bytes)?;
        d.set_item("deferred", s.deferred)?;
        d.set_item("resolved", s.resolved)?;
        d.set_item("queued", s.queued)?;
        d.set_item("step_max_s", s.step_max_s)?;
        d.set_item("step_max_gid", s.step_max_gid)?;
        d.set_item("slow_steps", s.slow_steps)?;
        d.set_item("branches", s.branches)?;
        d.set_item("dev_moves", s.dev_moves)?;
        d.set_item("starts", s.starts)?;
        d.set_item("value_choices", s.value_choices)?;
        d.set_item("value_choice_rows", s.value_choice_rows)?;
        if self.rows.tails {
            // variable-length rows (docs/search.md 2.15; only a game with tail segments)
            d.set_item("tail_rows", s.tail_rows)?;
            d.set_item("tail_tokens", s.tail_tokens)?;
            d.set_item("tail_max", s.tail_max)?;
            d.set_item("aborted_row_tokens", s.aborted_row_tokens)?;
        }
        if s.look.decisions > 0 {
            // the lookahead seats' counters (docs/search.md 3.9; only a driver with lookahead decisions)
            let l = &s.look;
            let ld = PyDict::new(py);
            ld.set_item("decisions", l.decisions)?;
            ld.set_item("nodes", l.nodes)?;
            ld.set_item("leaves", l.leaves)?;
            ld.set_item("prior_rows", l.prior_rows)?;
            ld.set_item("value_rows", l.value_rows)?;
            ld.set_item("value_dups", l.value_dups)?;
            ld.set_item("minor_steps", l.minor_steps)?;
            ld.set_item("groups", l.groups)?;
            ld.set_item("cut", l.cut)?;
            ld.set_item("rounds", l.rounds)?;
            ld.set_item("rows", l.rows)?;
            ld.set_item("fallback_stage", l.fallback_stage)?;
            ld.set_item("fallback_last", l.fallback_last)?;
            ld.set_item("fallback_mcts", l.fallback_mcts)?;
            ld.set_item("minor_root", l.minor_root)?;
            ld.set_item("aborted_nodes", l.aborted_nodes)?;
            ld.set_item("peak_games", l.peak_games)?;
            ld.set_item("by_stage", l.by_stage.clone())?;
            ld.set_item("busy_s", l.busy_s)?;
            ld.set_item("guard_decisions", l.guard_decisions)?;
            ld.set_item("guard_overrides", l.guard_overrides)?;
            d.set_item("look", ld)?;
        }
        Ok(d)
    }

    fn in_flight<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.d.in_flight();
        let d = PyDict::new(py);
        d.set_item("games", s.games)?;
        d.set_item("oldest_age_games", s.oldest_age_games)?;
        d.set_item("max_steps", s.max_steps)?;
        d.set_item("max_rounds", s.max_rounds)?;
        Ok(d)
    }
}

/// take_samples' dict of `recs` (docs/search.md 2.4), None if there are none: the arrays built and
/// the records freed off the GIL. `SearchDriver.take_samples` and `SampleRecords.arrays` (one
/// conversion for both).
fn samples_dict(py: Python<'_>, recs: Vec<Record>, f: usize, rows: RowKinds) -> PyResult<Option<PyObject>> {
    if recs.is_empty() {
        return Ok(None);
    }
    let s = py.allow_threads(move || {
        let s = sample_arrays_rows(&recs, f, rows.tails, rows.pointers);
        drop(recs);
        s
    });
    let n = s.n;
    let a = PyDict::new(py);
    a.set_item("x", s.x.into_pyarray(py).reshape([n, f])?)?;
    a.set_item("player", s.player.into_pyarray(py))?;
    a.set_item("n_players", s.n_players.into_pyarray(py))?;
    a.set_item("opt_off", s.opt_off.into_pyarray(py))?;
    a.set_item("opt_action", s.opt_action.into_pyarray(py))?;
    a.set_item("pi", s.pi.into_pyarray(py))?;
    a.set_item("z", s.z.into_pyarray(py).reshape([n, MAX_SEATS])?)?;
    a.set_item("q", s.q.into_pyarray(py).reshape([n, MAX_SEATS])?)?;
    a.set_item("sc", s.sc.into_pyarray(py).reshape([n, MAX_SEATS])?)?;
    a.set_item("net", s.net.into_pyarray(py))?;
    if s.aw > 0 {
        // the auxiliary targets (a driver created with aux=True; docs/search.md 1.2)
        a.set_item("aux", s.aux.into_pyarray(py).reshape([n, s.aw])?)?;
    }
    if s.tails {
        // variable-length rows (docs/search.md 2.15): the records' tail CSR (n + 1 offsets)
        a.set_item("t_off", s.t_off.into_pyarray(py))?;
        a.set_item("t_seg", s.t_seg.into_pyarray(py))?;
        a.set_item("t_slot", s.t_slot.into_pyarray(py))?;
        a.set_item("t_val", s.t_val.into_pyarray(py))?;
    }
    if s.pointers {
        a.set_item("ov_off", s.ov_off.into_pyarray(py))?;
        a.set_item("ov_opt", s.ov_opt.into_pyarray(py))?;
        a.set_item("ov_tok", s.ov_tok.into_pyarray(py))?;
    }
    Ok(Some(a.into_any().unbind()))
}

/// One game record replayed (docs/search.md 2.17, zero-farm's self-play records: `zero_search::record`): game
/// `game` with spec `spec` (JSON text), `record` one game's record (JSON text), `aux` its rows carry the
/// auxiliary targets -> (the rows as `take_samples` gives them - None for a game without samples -, the game's
/// facts {gid, n, seed, over, aborted, resigned, result, rounds, steps, setup, rows; abort_reason for a game the
/// driver aborted at its row limit}). A record replay refuses
/// raises ValueError with the reason (the record is rejected whole).
#[pyfunction]
#[pyo3(signature = (game, spec, record, aux=false))]
fn replay_rows(py: Python<'_>, game: &str, spec: &str, record: &str, aux: bool) -> PyResult<(Option<PyObject>, PyObject)> {
    let e = entry(game).map_err(verr)?;
    let (spec, record) = (spec.to_string(), record.to_string());
    let rp = py.allow_threads(move || (e.replay)(&spec, &record, aux)).map_err(verr)?;
    let info = PyDict::new(py);
    info.set_item("gid", rp.gid)?;
    info.set_item("n", rp.n)?;
    info.set_item("seed", rp.seed)?;
    info.set_item("over", rp.over)?;
    info.set_item("aborted", rp.aborted)?;
    info.set_item("resigned", rp.resigned)?;
    if let Some(why) = rp.abort_reason {
        // the driver aborted the game at its row limit (docs/search.md 2.15): no samples of its own
        info.set_item("abort_reason", why)?;
    }
    info.set_item("result", rp.result.clone())?;
    info.set_item("rounds", rp.rounds)?;
    info.set_item("steps", rp.steps)?;
    info.set_item("setup", rp.setup.clone())?;
    info.set_item("rows", rp.records.len())?;
    let gi = (e.info)();
    let s = samples_dict(py, rp.records, gi.num_features, RowKinds::of(&gi))?;
    Ok((s, info.into_any().unbind()))
}

/// The seed of game `gid` of a driver seeded `seed` (`zero_search::record::game_seed`: what `new_game` gets).
#[pyfunction]
#[pyo3(signature = (seed, gid, paired=false))]
fn game_seed_of(seed: u64, gid: u32, paired: bool) -> u64 {
    zero_search::record::game_seed(seed, gid, paired)
}

/// A finished game's row as `take_results` gives it (docs/search.md 2.5).
fn result_row<'py>(py: Python<'py>, r: &zero_search::driver::GameResult) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("gid", r.gid)?;
    d.set_item("n", r.n)?;
    d.set_item("over", r.over)?;
    d.set_item("aborted", r.aborted)?;
    d.set_item("result", r.result.clone())?;
    d.set_item("scores", r.scores.clone())?;
    d.set_item("steps", r.steps)?;
    d.set_item("decisions", r.decisions)?;
    d.set_item("ext_moves", r.ext_moves)?;
    d.set_item("rounds", r.rounds)?;
    d.set_item("samples", r.samples)?;
    d.set_item("tag", r.tag.clone())?;
    if !r.moves.is_empty() {
        // DriverCfg::move_log (a diagnostic): [(seat, action, legal actions)]
        d.set_item("moves", r.moves.clone())?;
    }
    if let Some(b) = &r.branch {
        // a branch game (DriverCfg::branch, docs/search.md 2.11)
        let bd = PyDict::new(py);
        bd.set_item("at", b.at)?;
        bd.set_item("of", b.of)?;
        bd.set_item("decision", b.decision)?;
        bd.set_item("step", b.step)?;
        bd.set_item("action", b.action)?;
        bd.set_item("moves", b.moves)?;
        bd.set_item("played", b.played)?;
        bd.set_item("same_first", b.same_first)?;
        bd.set_item("start", b.start)?;
        d.set_item("branch", bd)?;
    }
    if let Some(j) = r.start {
        // a game started from an archive entry (DriverCfg::starts)
        d.set_item("start", j)?;
    }
    if !r.episodes.is_empty() {
        // a chained game's contest (SearchGame::chained, docs/search.md 2.14): its episodes
        let eps = PyList::empty(py);
        for e in &r.episodes {
            let ed = PyDict::new(py);
            ed.set_item("over", e.over)?;
            ed.set_item("aborted", e.aborted)?;
            ed.set_item("result", e.result.clone())?;
            ed.set_item("margins", e.margins.clone())?;
            ed.set_item("steps", e.steps)?;
            ed.set_item("decisions", e.decisions)?;
            ed.set_item("ext_moves", e.ext_moves)?;
            ed.set_item("rounds", e.rounds)?;
            ed.set_item("samples", e.samples)?;
            if let Some(rs) = &e.resign {
                ed.set_item("resign", resign_dict(py, rs)?)?;
            }
            eps.append(ed)?;
        }
        d.set_item("episodes", eps)?;
    }
    if let Some(rs) = &r.resign {
        // self-play resignation (NnSeat::resign_t, docs/search.md 3.14): only when the game's seats resign
        d.set_item("resign", resign_dict(py, rs)?)?;
    }
    if let Some(why) = r.abort_reason {
        // the driver aborted the game (docs/search.md 2.15: "max_row_tokens")
        d.set_item("abort_reason", why)?;
    }
    Ok(d)
}

/// The bytes a record holds (itself and its arrays' contents): `SampleRecords.nbytes`, a bounded
/// queue's measure.
fn record_bytes(r: &Record) -> usize {
    std::mem::size_of::<Record>() + 4 * r.x.len() + 2 * r.opts.len() + 4 * r.pi.len() + 4 * r.aux.len() + 6 * r.tail.seg.len() + 4 * r.tail.val.len() + 12 * r.ov.len()
}

/// Training records taken from a driver as they are (`SearchDriver.take_records`, docs/search.md
/// 2.4): `len()` records holding `nbytes`; `split(sizes)` cuts them into consecutive parts (a
/// finished game's records each: zero/workqueue.py split_records) and `arrays()` converts them
/// into take_samples' dict - value for value the arrays take_samples gives at the point the records
/// were taken, cut where the parts are - once: the records are consumed.
#[pyclass(name = "SampleRecords")]
pub struct PySampleRecords {
    recs: Vec<Record>,
    nfeat: usize,
    rows: RowKinds,
    bytes: usize,
    used: bool,
}

impl PySampleRecords {
    fn new(recs: Vec<Record>, nfeat: usize, rows: RowKinds) -> PySampleRecords {
        let bytes = recs.iter().map(record_bytes).sum();
        PySampleRecords { recs, nfeat, rows, bytes, used: false }
    }

    fn check(&self) -> PyResult<()> {
        if self.used {
            return Err(PyRuntimeError::new_err("SampleRecords: these records were converted or split already"));
        }
        Ok(())
    }
}

#[pymethods]
impl PySampleRecords {
    fn __len__(&self) -> usize {
        self.recs.len()
    }

    /// Bytes the records hold (the records and their arrays).
    #[getter]
    fn nbytes(&self) -> usize {
        self.bytes
    }

    /// Cut into consecutive parts of `sizes` records each (every size > 0, their sum `len()`): new
    /// SampleRecords in order; this one is used up.
    fn split(&mut self, sizes: Vec<usize>) -> PyResult<Vec<PySampleRecords>> {
        self.check()?;
        let total: usize = sizes.iter().sum();
        if total != self.recs.len() || sizes.iter().any(|&k| k == 0) {
            return Err(PyValueError::new_err(format!(
                "SampleRecords.split: sizes {:?} (sum {}) for {} records (every size > 0, their sum the records)",
                sizes,
                total,
                self.recs.len()
            )));
        }
        self.used = true;
        let (nfeat, rows) = (self.nfeat, self.rows);
        let mut rest = std::mem::take(&mut self.recs).into_iter();
        Ok(sizes.iter().map(|&k| PySampleRecords::new(rest.by_ref().take(k).collect(), nfeat, rows)).collect())
    }

    /// take_samples' dict of these records (docs/search.md 2.4), built off the GIL; once (a second
    /// call, or one after `split`, raises).
    fn arrays(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        self.check()?;
        self.used = true;
        samples_dict(py, std::mem::take(&mut self.recs), self.nfeat, self.rows)
    }
}

impl PySearchDriver {
    fn to_py(&self, py: Python<'_>, r: Option<Batch>) -> PyResult<Option<PyObject>> {
        let Some(b) = r else { return Ok(None) };
        let (key, external) = self.d.keys()[b.key as usize].clone();
        let a = PyDict::new(py);
        let n = b.b;
        a.set_item("player", b.player.into_pyarray(py))?;
        a.set_item("n_players", b.n_players.into_pyarray(py))?;
        a.set_item("gid", b.gid.into_pyarray(py))?;
        if external {
            a.set_item("states", b.states)?;
            return Ok(Some((b.id, key, "ext", a).into_pyobject(py)?.into_any().unbind()));
        }
        if b.xs_off.is_empty() {
            a.set_item("x", b.x.into_pyarray(py).reshape([n, self.nfeat])?)?;
        } else {
            // a sparse batch (`sparse=True`, docs/search.md 2.2): each row's non-zero features
            a.set_item("xs_off", b.xs_off.into_pyarray(py))?;
            a.set_item("xs_idx", b.xs_idx.into_pyarray(py))?;
            a.set_item("xs_val", b.xs_val.into_pyarray(py))?;
            a.set_item("n_features", self.nfeat)?;
        }
        a.set_item("opt_off", b.opt_off.into_pyarray(py))?;
        a.set_item("opt_action", b.opt_action.into_pyarray(py))?;
        a.set_item("extra", b.extra.into_pyarray(py))?;
        if self.rows.tails {
            // variable-length rows (docs/search.md 2.15): the rows' tail CSR (B + 1 offsets)
            a.set_item("t_off", b.t_off.into_pyarray(py))?;
            a.set_item("t_seg", b.t_seg.into_pyarray(py))?;
            a.set_item("t_slot", b.t_slot.into_pyarray(py))?;
            a.set_item("t_val", b.t_val.into_pyarray(py))?;
        }
        if self.rows.pointers {
            a.set_item("ov_off", b.ov_off.into_pyarray(py))?;
            a.set_item("ov_opt", b.ov_opt.into_pyarray(py))?;
            a.set_item("ov_tok", b.ov_tok.into_pyarray(py))?;
        }
        Ok(Some((b.id, key, "nn", a).into_pyobject(py)?.into_any().unbind()))
    }
}

// ------------------------------------------------------------------------ Lookahead

/// The exhaustive lookahead of one position (docs/search.md 3.9; the position API behind
/// `zero.lookahead.evaluate_position`): `Lookahead(game, settings, worlds=None, seeds=None, seed=0,
/// belief_tilt=0.0, belief_clip=BELIEF_CLIP, belief_mix=0.0, belief=None, lines=True)` - the pending
/// decision of `game`; its worlds given (`worlds`: games at the same decision), else dealt from `seeds`
/// (one per world) or `seed`'s stream; with `belief_tilt` > 0 dealt by the root's belief: `belief` (the
/// predictions, f32 [aux_width]) or, None, asked for in a first round's `ROW_BELIEF` row. Drive it with
/// `next_batch()` (None: decided) and `submit(priors, value, aux=None)`, then read `result()`, `line()`.
#[pyclass(name = "Lookahead")]
pub struct PyLookahead {
    l: Box<dyn DynLook>,
    nfeat: usize,
    /// The features of every round, one array kept across rounds (fresh memory a round is slower
    /// than the lookahead): a round's `x` is a view of its first rows, valid until the next round.
    xbuf: Option<Py<numpy::PyArray2<f32>>>,
}

#[pymethods]
impl PyLookahead {
    #[new]
    #[pyo3(signature = (game, settings, worlds=None, seeds=None, seed=0, belief_tilt=0.0, belief_clip=BELIEF_CLIP, belief_mix=0.0, belief=None, lines=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(game: &PyGame, settings: &Bound<'_, PyAny>, worlds: Option<Vec<PyRef<'_, PyGame>>>, seeds: Option<Vec<u64>>, seed: u64, belief_tilt: f64, belief_clip: f64, belief_mix: f64, belief: Option<PyReadonlyArray1<'_, f32>>, lines: bool) -> PyResult<Self> {
        let cfg = LookCfg::parse(&spec_json(Some(settings))?).map_err(verr)?;
        let ws: Vec<&dyn DynGame> = worlds.as_ref().map(|v| v.iter().map(|w| &*w.g).collect()).unwrap_or_default();
        let b = (belief_tilt > 0.0).then_some((belief_tilt, belief_clip, belief_mix));
        if b.is_none() && belief.is_some() {
            return Err(PyValueError::new_err("lookahead: belief predictions without a belief_tilt > 0"));
        }
        let pred = match &belief {
            Some(p) => Some(p.as_slice().map_err(|e| PyValueError::new_err(format!("belief: {e}")))?.to_vec()),
            None => None,
        };
        let l = game.g.lookahead(cfg, ws, seeds.unwrap_or_default(), seed, b, pred, lines).map_err(verr)?;
        Ok(PyLookahead { l, nfeat: game.g.info().num_features, xbuf: None })
    }

    /// The next round's rows - {x [b, F], opt_off, opt_action, player, n_players, extra} (a network
    /// batch as the driver's; `x` a view of an array the lookahead reuses: valid until the next
    /// round), with each row's tail (`t_off` [b + 1], `t_seg`, `t_slot`, `t_val`) for a game with tail segments
    /// and its overrides (`ov_off` [b + 1], `ov_opt`, `ov_tok`) for one with pointer overrides (docs/search.md 2.4,
    /// 2.15; a row past the game's row limit raises) - or None when the decision is made.
    fn next_batch<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let l = &mut self.l;
        let Some(n) = py.allow_threads(|| l.step()).map_err(rerr)? else { return Ok(None) };
        let f = self.nfeat;
        let cap = self.xbuf.as_ref().map_or(0, |x| x.bind(py).shape()[0]);
        if cap < n {
            let rows = n.max(cap * 2).max(64);
            self.xbuf = Some(numpy::PyArray2::<f32>::zeros(py, [rows, f], false).unbind());
        }
        let xb = self.xbuf.as_ref().unwrap().bind(py);
        let r = self.l.rows();
        {
            // SAFETY: the array is this object's own, never handed out but as the views below, which
            // the protocol reads before the next round overwrites them (under the GIL here)
            let dst = unsafe { xb.as_slice_mut() }.map_err(|e| PyRuntimeError::new_err(format!("lookahead buffer: {e}")))?;
            dst[..n * f].copy_from_slice(&r.x[..n * f]);
        }
        let a = PyDict::new(py);
        a.set_item("x", xb.call_method1("__getitem__", (pyo3::types::PySlice::new(py, 0, n as isize, 1),))?)?;
        a.set_item("opt_off", r.opt_off.to_vec().into_pyarray(py))?;
        a.set_item("opt_action", r.opt_action.to_vec().into_pyarray(py))?;
        a.set_item("player", r.player.to_vec().into_pyarray(py))?;
        a.set_item("n_players", r.n_players.to_vec().into_pyarray(py))?;
        a.set_item("extra", r.extra.to_vec().into_pyarray(py))?;
        if !r.t_off.is_empty() {
            // variable-length rows (docs/search.md 2.15; a game with tail segments only)
            a.set_item("t_off", r.t_off.to_vec().into_pyarray(py))?;
            a.set_item("t_seg", r.t_seg.to_vec().into_pyarray(py))?;
            a.set_item("t_slot", r.t_slot.to_vec().into_pyarray(py))?;
            a.set_item("t_val", r.t_val.to_vec().into_pyarray(py))?;
        }
        if !r.ov_off.is_empty() {
            // (a game with pointer overrides only)
            a.set_item("ov_off", r.ov_off.to_vec().into_pyarray(py))?;
            a.set_item("ov_opt", r.ov_opt.to_vec().into_pyarray(py))?;
            a.set_item("ov_tok", r.ov_tok.to_vec().into_pyarray(py))?;
        }
        Ok(Some(a))
    }

    /// The last round's answers: priors f32 [options], value f32 [b, MAX_SEATS] (relative to each row's
    /// player), `aux` f32 [belief rows, aux_width] when the round has a root belief row.
    #[pyo3(signature = (priors, value, aux=None))]
    fn submit(&mut self, priors: PyReadonlyArray1<'_, f32>, value: PyReadonlyArray2<'_, f32>, aux: Option<PyReadonlyArray2<'_, f32>>) -> PyResult<()> {
        if value.shape()[1] != MAX_SEATS {
            return Err(PyValueError::new_err(format!("value must be [B, {MAX_SEATS}]")));
        }
        let p = priors.as_slice().map_err(|e| PyValueError::new_err(format!("priors: {e}")))?;
        let v = value.as_slice().map_err(|e| PyValueError::new_err(format!("value: {e}")))?;
        let a = match &aux {
            Some(a) => a.as_slice().map_err(|e| PyValueError::new_err(format!("aux: {e}")))?,
            None => &[],
        };
        self.l.submit(p, v, a).map_err(verr)
    }

    /// After the decision: {'outcome': the chosen action (or 'mcts': every stage over the budget),
    /// 'options': [{'action', 'values': [[world, value], ...], 'mean', 'sd'}] in the root's order,
    /// 'stats': the counters, 'settings': the stage's settings (JSON)}.
    fn result<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        match self.l.outcome() {
            None => return Err(PyRuntimeError::new_err("lookahead: not decided yet (next_batch until None)")),
            Some(Outcome::Act(a)) => d.set_item("outcome", a)?,
            Some(Outcome::Mcts) => d.set_item("outcome", "mcts")?,
        }
        let opts = PyList::empty(py);
        let st = self.l.stats();
        if !st.last && !st.minor_root {
            for o in self.l.options().map_err(rerr)? {
                let od = PyDict::new(py);
                od.set_item("action", o.act)?;
                od.set_item("values", o.values.iter().map(|&(w, v)| (w, v)).collect::<Vec<_>>())?;
                od.set_item("mean", o.mean)?;
                od.set_item("sd", o.sd)?;
                opts.append(od)?;
            }
        }
        d.set_item("options", opts)?;
        let sd = PyDict::new(py);
        sd.set_item("nodes", st.nodes)?;
        sd.set_item("leaves", st.leaves)?;
        sd.set_item("prior_rows", st.prior_rows)?;
        sd.set_item("value_rows", st.value_rows)?;
        sd.set_item("value_dups", st.value_dups)?;
        sd.set_item("minor_steps", st.minor_steps)?;
        sd.set_item("cut", st.cut)?;
        sd.set_item("groups", st.groups)?;
        sd.set_item("rounds", st.rounds)?;
        sd.set_item("rows", st.rows)?;
        sd.set_item("stage", st.stage)?;
        sd.set_item("last", st.last)?;
        sd.set_item("aborted_nodes", st.aborted_nodes)?;
        sd.set_item("minor_root", st.minor_root)?;
        sd.set_item("peak_games", st.peak_games)?;
        d.set_item("stats", sd)?;
        d.set_item("settings", self.l.cfg_json())?;
        Ok(d)
    }

    /// World `world`'s line after root option `action` under the solved policy: ([(player, action,
    /// minor)], where it ended: 'game over' | 'horizon' | 'cut' | 'plies' | 'no options'), None when
    /// the world's root group did not expand it.
    fn line(&mut self, world: usize, action: u16) -> PyResult<Option<(Vec<(u8, u16, bool)>, &'static str)>> {
        Ok(self.l.line(world, action).map_err(rerr)?.map(|(l, e)| (l, e.name())))
    }

    /// The worlds (after the deal): games at the root decision.
    fn worlds(&self) -> Vec<PyGame> {
        self.l.worlds().into_iter().map(|g| PyGame { g }).collect()
    }

    /// The root belief's predictions the worlds were dealt by (None: uniform worlds).
    #[getter]
    fn belief(&self) -> Option<Vec<f32>> {
        self.l.belief()
    }

    /// The game's facts (`game_info_of`'s dict: its tail segments and pointer overrides name the rows' keys).
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        info_dict(py, &self.l.info())
    }
}
