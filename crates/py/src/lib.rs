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
//! `SearchTree`, `SearchDriver`, `Lookahead` (the position API of docs/search.md 3.9),
//! `game_names()`, `game_info_of(name)`, `lookahead_settings(settings)` and `MAX_SEATS`.
use std::sync::OnceLock;

use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use zero_search::driver::{root_floor, sample_arrays, view_pick, Batch, DriverCfg, KeyKind, LeafView, NnSeat, PolicyTarget, RootSearch, SeatCfg, Select, ViewTarget, BELIEF_CLIP};
use zero_search::dynapi::{boxed_driver, boxed_game, game_info, DynDriver, DynGame, DynLook, DynTree, GameInfo};
use zero_search::lookahead::{LookCfg, Outcome};
use zero_search::{SearchGame, MAX_SEATS};

/// For `engine_module!` (the game crate needs its own `pyo3` dependency too: the PyO3 macros
/// refer to `::pyo3`).
pub use pyo3;

/// One registered game type: its name in Python and its constructors.
pub struct GameEntry {
    pub name: &'static str,
    pub new_game: fn(&str, usize, u64) -> Result<Box<dyn DynGame>, String>,
    pub new_driver: fn(DriverCfg) -> Result<Box<dyn DynDriver>, String>,
    pub info: fn() -> GameInfo,
}

impl GameEntry {
    pub const fn of<G: SearchGame>(name: &'static str) -> GameEntry {
        GameEntry { name, new_game: boxed_game::<G>, new_driver: boxed_driver::<G>, info: game_info::<G> }
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
    m.add_class::<PyLookahead>()?;
    m.add_function(wrap_pyfunction!(lookahead_settings, m)?)?;
    m.add_function(wrap_pyfunction!(game_names, m)?)?;
    m.add_function(wrap_pyfunction!(game_info_of, m)?)?;
    m.add_function(wrap_pyfunction!(root_floor_of, m)?)?;
    m.add_function(wrap_pyfunction!(view_pick_of, m)?)?;
    m.add_function(wrap_pyfunction!(belief_weight_of, m)?)?;
    m.add_function(wrap_pyfunction!(gumbel_sequence_of, m)?)?;
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

fn get<'py, T: FromPyObject<'py>>(d: &Bound<'py, PyDict>, k: &str, default: T) -> PyResult<T> {
    match d.get_item(k)? {
        Some(v) if !v.is_none() => v.extract(),
        _ => Ok(default),
    }
}

fn flag(d: &Bound<'_, PyDict>, k: &str, default: bool) -> PyResult<bool> {
    match d.get_item(k)? {
        Some(v) if !v.is_none() => v.is_truthy(),
        _ => Ok(default),
    }
}

const SEAT_KEYS: &[&str] = &["key", "sims", "c_puct", "selfplay", "alpha", "eps", "temp_moves", "sims_fast", "sims_minor", "p_full", "chance_k", "margin_w", "redet", "policy_target", "cq_c_visit", "cq_c_scale", "select", "select_min_visits", "select_c_visit", "select_c_scale", "record", "root_min_visits", "root_min_visits_max_actions", "leaf_value_view", "view_samples", "view_sample_target", "belief_tilt", "belief_clip", "belief_mix", "belief_fast", "root_search", "gumbel_m", "gumbel_scale", "gumbel_c_visit", "gumbel_c_scale", "lookahead"];

fn key_index(keys: &mut Vec<(String, KeyKind)>, name: String, kind: KeyKind) -> PyResult<u16> {
    match keys.iter().position(|k| k.0 == name) {
        Some(i) if keys[i].1 == kind => Ok(i as u16),
        Some(_) => Err(PyValueError::new_err(format!("evaluator key {name:?} used for both a network and an external seat"))),
        None => {
            keys.push((name, kind));
            Ok((keys.len() - 1) as u16)
        }
    }
}

/// Seat dict: {'bot': name} | {'external': key} | network seat settings (SEAT_KEYS).
fn parse_seat(d: &Bound<'_, PyDict>, keys: &mut Vec<(String, KeyKind)>) -> PyResult<SeatCfg> {
    if let Some(b) = d.get_item("bot")? {
        return Ok(SeatCfg::Bot(b.extract()?));
    }
    if let Some(e) = d.get_item("external")? {
        return Ok(SeatCfg::External(key_index(keys, e.extract()?, KeyKind::External)?));
    }
    for k in d.keys() {
        let k: String = k.extract()?;
        if !SEAT_KEYS.contains(&k.as_str()) {
            return Err(PyValueError::new_err(format!("seat: unknown setting {k:?} (bot, external, {SEAT_KEYS:?})")));
        }
    }
    let key = key_index(keys, get(d, "key", "m".to_string())?, KeyKind::Nn)?;
    let sims_fast: Option<u32> = match d.get_item("sims_fast")? {
        Some(v) if !v.is_none() => Some(v.extract()?),
        _ => None,
    };
    let sims_minor: Option<u32> = match d.get_item("sims_minor")? {
        Some(v) if !v.is_none() => Some(v.extract()?),
        _ => None,
    };
    // training target of recorded decisions: 'visits' (default) | 'cq' (docs/search.md 3.1)
    let (cq_c_visit, cq_c_scale) = (get(d, "cq_c_visit", 50.0f64)?, get(d, "cq_c_scale", 1.0f64)?);
    let policy_target = PolicyTarget::parse(&get(d, "policy_target", "visits".to_string())?, cq_c_visit, cq_c_scale).map_err(verr)?;
    let select: String = get(d, "select", "visits".to_string())?;
    for k in ["select_c_visit", "select_c_scale"] {
        if select == "visits" && matches!(d.get_item(k)?, Some(v) if !v.is_none()) {
            return Err(PyValueError::new_err(format!("{k}: a setting of select 'cq' ('visits' has no completed-Q transform)")));
        }
    }
    // the root search: 'puct' (default) | 'gumbel' with its settings (docs/search.md 3.7; sigma's
    // transform absent: the seat's cq_c_visit, cq_c_scale)
    let root_search: String = get(d, "root_search", "puct".to_string())?;
    for k in ["gumbel_m", "gumbel_scale", "gumbel_c_visit", "gumbel_c_scale"] {
        if root_search != "gumbel" && matches!(d.get_item(k)?, Some(v) if !v.is_none()) {
            return Err(PyValueError::new_err(format!("{k}: a setting of root_search 'gumbel'")));
        }
    }
    let root_search = RootSearch::parse(&root_search, get(d, "gumbel_m", 16u32)?, get(d, "gumbel_scale", 1.0f64)?, get(d, "gumbel_c_visit", cq_c_visit)?, get(d, "gumbel_c_scale", cq_c_scale)?).map_err(verr)?;
    // the exhaustive lookahead (docs/search.md 3.9): a dict of its settings (or their JSON), None = off
    let look = match d.get_item("lookahead")? {
        Some(v) if !v.is_none() => Some(Box::new(LookCfg::parse(&spec_json(Some(&v))?).map_err(verr)?)),
        _ => None,
    };
    Ok(SeatCfg::Nn(NnSeat {
        key,
        sims: get(d, "sims", 100u32)?,
        c_puct: get(d, "c_puct", 1.0f64)?,
        selfplay: flag(d, "selfplay", false)?,
        alpha: get(d, "alpha", 0.3f64)?,
        eps: get(d, "eps", 0.25f64)?,
        temp_moves: get(d, "temp_moves", 30u32)?,
        sims_fast,
        p_full: get(d, "p_full", 1.0f64)?,
        sims_minor,
        chance_k: get(d, "chance_k", 0u32)?,
        margin_w: get(d, "margin_w", 0.0f64)?,
        redet: flag(d, "redet", true)?,
        policy_target,
        // the rule of the moves it does not sample: 'visits' (default) | 'cq' with the guard
        // select_min_visits and the completed-Q transform select_c_visit, select_c_scale (absent:
        // the target's cq_c_visit, cq_c_scale; docs/search.md 3.3)
        select: Select::parse(&select, get(d, "select_min_visits", 0u32)?, get(d, "select_c_visit", cq_c_visit)?, get(d, "select_c_scale", cq_c_scale)?).map_err(verr)?,
        // whether its full searches are recorded (absent: true; false: a league opponent, a
        // past checkpoint whose moves are not the current policy's, docs/az.md 14)
        record: flag(d, "record", true)?,
        // root minimum visits of its full searches (0 = off) on roots of up to
        // root_min_visits_max_actions options (docs/search.md 3.5)
        root_min_visits: get(d, "root_min_visits", 0u32)?,
        root_min_visits_max_actions: get(d, "root_min_visits_max_actions", 4u32)?,
        // whose view a leaf's value comes from: 'mover' (default) | 'root' (docs/search.md 3.6)
        leaf_value_view: LeafView::parse(&get(d, "leaf_value_view", "mover".to_string())?).map_err(verr)?,
        // value-only samples: a share of the recorded decisions (0 = off) and their q target
        view_samples: get(d, "view_samples", 0.0f64)?,
        view_target: ViewTarget::parse(&get(d, "view_sample_target", "next_q".to_string())?).map_err(verr)?,
        // the root's belief (docs/search.md 3.8): its tilt (0 = off), clip, uniform share, and
        // whether fast and minor searches take it too
        belief_tilt: get(d, "belief_tilt", 0.0f64)?,
        belief_clip: get(d, "belief_clip", BELIEF_CLIP)?,
        belief_mix: get(d, "belief_mix", 0.0f64)?,
        belief_fast: flag(d, "belief_fast", false)?,
        root_search,
        look,
    }))
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
    #[pyo3(signature = (viewer, seed, belief, tilt=1.0, clip=BELIEF_CLIP))]
    fn determinize_with(&mut self, viewer: usize, seed: u64, belief: PyReadonlyArray1<'_, f32>, tilt: f64, clip: f64) -> PyResult<()> {
        let p = belief.as_slice().map_err(|e| PyValueError::new_err(format!("belief: {e}")))?;
        self.g.determinize_with(viewer, seed, p, tilt, clip).map_err(verr)
    }
    fn encode<'py>(&self, py: Python<'py>, viewer: usize) -> Bound<'py, numpy::PyArray1<f32>> {
        self.g.encode(viewer).into_pyarray(py)
    }
    fn export_state(&self) -> String {
        self.g.export_state()
    }
    fn apply_external(&mut self, payload: &str) -> PyResult<Vec<u64>> {
        self.g.apply_external(payload).map_err(verr)
    }
    /// A Rust search tree for the pending decision.
    #[pyo3(signature = (redet=true, seed=0))]
    fn search_tree(&self, redet: bool, seed: u64) -> PyResult<PySearchTree> {
        Ok(PySearchTree { t: self.g.new_tree(redet, seed).map_err(verr)? })
    }
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
    #[pyo3(signature = (game, c_puct, chance_k=0, margin_w=0.0, root_view=false))]
    fn simulate<'py>(&mut self, py: Python<'py>, game: &PyGame, c_puct: f64, chance_k: u32, margin_w: f64, root_view: bool) -> PyResult<Option<Bound<'py, pyo3::types::PyTuple>>> {
        match self.t.simulate(&*game.g, c_puct, chance_k, margin_w, root_view).map_err(rerr)? {
            None => Ok(None),
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
    /// The completed-Q policy target at the root (docs/search.md 3.1; [] before expansion).
    #[pyo3(signature = (c_visit=50.0, c_scale=1.0))]
    fn root_cq_target(&self, c_visit: f64, c_scale: f64) -> Vec<f32> {
        self.t.root_cq_target(c_visit, c_scale)
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
    #[pyo3(signature = (belief, game, tilt=1.0, clip=BELIEF_CLIP, mix=0.0))]
    fn set_belief(&mut self, belief: Option<PyReadonlyArray1<'_, f32>>, game: &PyGame, tilt: f64, clip: f64, mix: f64) -> PyResult<()> {
        let p = match &belief {
            Some(b) => Some(b.as_slice().map_err(|e| PyValueError::new_err(format!("belief: {e}")))?),
            None => None,
        };
        self.t.set_belief(p, tilt, clip, mix, &*game.g).map_err(verr)
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
}

// ------------------------------------------------------------------------ SearchDriver

/// The batched driver (docs/search.md 2). `seats` = per game a list of seat dicts; `aux`: record
/// the game's auxiliary targets with the samples (`take_samples()['aux']`, docs/search.md 1.2);
/// `paired`: games 2k and 2k + 1 share every random stream (the deal), `DriverCfg::paired`.
#[pyclass(name = "SearchDriver")]
pub struct PySearchDriver {
    d: Box<dyn DynDriver>,
    nfeat: usize,
}

#[pymethods]
impl PySearchDriver {
    #[new]
    #[pyo3(signature = (game, seats, spec=None, concurrency=64, seed=0, record=false, threads=4, groups=2, max_steps=100_000, rolling=false, net=0, queue_only=false, move_log=0, aux=false, paired=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(game: &str, seats: &Bound<'_, PyList>, spec: Option<&Bound<'_, PyAny>>, concurrency: usize, seed: u64, record: bool, threads: usize, groups: usize, max_steps: u64, rolling: bool, net: u32, queue_only: bool, move_log: u32, aux: bool, paired: bool) -> PyResult<Self> {
        let spec = spec_json(spec)?;
        let mut keys = Vec::new();
        let mut table: Vec<SeatCfg> = Vec::new();
        let mut games = Vec::with_capacity(seats.len());
        for g in seats.iter() {
            let row: Vec<Bound<'_, PyDict>> = g.extract()?;
            let mut ix = Vec::with_capacity(row.len());
            for s in row {
                let sc = parse_seat(&s, &mut keys)?;
                let i = match table.iter().position(|x| *x == sc) {
                    Some(i) => i,
                    None => {
                        table.push(sc);
                        table.len() - 1
                    }
                };
                ix.push(i as u16);
            }
            games.push(ix);
        }
        let cfg = DriverCfg { spec, seats: table, games, keys, concurrency, seed, record, threads, groups, max_steps, rolling, net, queue_only, move_log, aux, paired };
        let d = new_driver(game, cfg).map_err(verr)?;
        let nfeat = d.info().num_features;
        Ok(PySearchDriver { d, nfeat })
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
    #[pyo3(signature = (id, priors, value, margin, aux=None))]
    fn submit(&mut self, id: u64, priors: PyReadonlyArray1<'_, f32>, value: PyReadonlyArray2<'_, f32>, margin: PyReadonlyArray2<'_, f32>, aux: Option<PyReadonlyArray2<'_, f32>>) -> PyResult<()> {
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
        self.d.submit(id, p, v, m, a).map_err(verr)
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
            let d = PyDict::new(py);
            d.set_item("gid", r.gid)?;
            d.set_item("n", r.n)?;
            d.set_item("over", r.over)?;
            d.set_item("result", r.result)?;
            d.set_item("scores", r.scores)?;
            d.set_item("steps", r.steps)?;
            d.set_item("decisions", r.decisions)?;
            d.set_item("ext_moves", r.ext_moves)?;
            d.set_item("rounds", r.rounds)?;
            d.set_item("samples", r.samples)?;
            d.set_item("tag", r.tag)?;
            if !r.moves.is_empty() {
                // DriverCfg::move_log (a diagnostic): [(seat, action, legal actions)]
                d.set_item("moves", r.moves)?;
            }
            out.append(d)?;
        }
        Ok(out)
    }

    /// Training sample arrays (docs/search.md 2.4) or None.
    fn take_samples(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let recs = self.d.take_samples();
        if recs.is_empty() {
            return Ok(None);
        }
        let f = self.nfeat;
        let s = py.allow_threads(|| sample_arrays(&recs, f));
        drop(recs);
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
        Ok(Some(a.into_any().unbind()))
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
        a.set_item("x", b.x.into_pyarray(py).reshape([n, self.nfeat])?)?;
        a.set_item("opt_off", b.opt_off.into_pyarray(py))?;
        a.set_item("opt_action", b.opt_action.into_pyarray(py))?;
        a.set_item("extra", b.extra.into_pyarray(py))?;
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
    /// round) - or None when the decision is made.
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
}
