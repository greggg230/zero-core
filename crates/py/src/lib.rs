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
//! `SearchTree`, `SearchDriver`, `game_names()`, `game_info_of(name)` and `MAX_SEATS`.
use std::sync::OnceLock;

use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use zero_search::driver::{sample_arrays, Batch, DriverCfg, KeyKind, NnSeat, PolicyTarget, SeatCfg};
use zero_search::dynapi::{boxed_driver, boxed_game, boxed_import, game_info, DynDriver, DynGame, DynTree, GameInfo};
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
    /// (spec, external state) -> a game (`SearchGame::import_state`)
    pub from_state: fn(&str, &str) -> Result<Box<dyn DynGame>, String>,
}

impl GameEntry {
    pub const fn of<G: SearchGame>(name: &'static str) -> GameEntry {
        GameEntry { name, new_game: boxed_game::<G>, new_driver: boxed_driver::<G>, info: game_info::<G>, from_state: boxed_import::<G> }
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
    m.add_function(wrap_pyfunction!(game_names, m)?)?;
    m.add_function(wrap_pyfunction!(game_info_of, m)?)?;
    m.add("MAX_SEATS", MAX_SEATS)?;
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

const SEAT_KEYS: &[&str] = &["key", "sims", "c_puct", "selfplay", "alpha", "eps", "temp_moves", "sims_fast", "p_full", "chance_k", "margin_w", "redet", "policy_target", "cq_c_visit", "cq_c_scale"];

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
    // training target of recorded decisions: 'visits' (default) | 'cq' (docs/search.md 3.1)
    let policy_target = PolicyTarget::parse(&get(d, "policy_target", "visits".to_string())?, get(d, "cq_c_visit", 50.0f64)?, get(d, "cq_c_scale", 1.0f64)?).map_err(verr)?;
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
        chance_k: get(d, "chance_k", 0u32)?,
        margin_w: get(d, "margin_w", 0.0f64)?,
        redet: flag(d, "redet", true)?,
        policy_target,
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
    Ok(d)
}

/// Registered game names.
#[pyfunction]
fn game_names() -> Vec<&'static str> {
    registered().iter().map(|g| g.name).collect()
}

/// {name, num_actions, num_features, min_players, max_players, bots, max_seats}.
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
    /// A game from an external state (the format of `export_state`, e.g. a web app's game):
    /// `Game.from_state('kingdombuilder', state_json, spec=None)`.
    #[staticmethod]
    #[pyo3(signature = (game, state, spec=None))]
    fn from_state(game: &str, state: &str, spec: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let s = spec_json(spec)?;
        Ok(PyGame { g: (entry(game).map_err(verr)?.from_state)(&s, state).map_err(verr)? })
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
    fn result(&self) -> Vec<f32> {
        self.g.result()
    }
    fn margin(&self, seat: usize) -> f32 {
        self.g.margin(seat)
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
    fn determinize(&mut self, viewer: usize, seed: u64) {
        self.g.determinize(viewer, seed)
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
    /// `step` -> (chance keys, the external-format moves (JSON strings) the action completed).
    fn step_external(&mut self, a: u16) -> PyResult<(Vec<u64>, Vec<String>)> {
        self.g.step_external(a).map_err(verr)
    }
    /// A Rust search tree for the pending decision.
    #[pyo3(signature = (redet=true, seed=0))]
    fn search_tree(&self, redet: bool, seed: u64) -> PyResult<PySearchTree> {
        Ok(PySearchTree { t: self.g.new_tree(redet, seed).map_err(verr)? })
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
    #[pyo3(signature = (game, c_puct, chance_k=0, margin_w=0.0))]
    fn simulate<'py>(&mut self, py: Python<'py>, game: &PyGame, c_puct: f64, chance_k: u32, margin_w: f64) -> PyResult<Option<(usize, Bound<'py, numpy::PyArray1<f32>>, Vec<u16>)>> {
        match self.t.simulate(&*game.g, c_puct, chance_k, margin_w).map_err(rerr)? {
            None => Ok(None),
            Some(l) => Ok(Some((l.player, l.x.into_pyarray(py), l.acts))),
        }
    }
    #[pyo3(signature = (priors, value, margin, margin_w=0.0))]
    fn expand(&mut self, priors: Vec<f32>, value: Vec<f32>, margin: Vec<f32>, margin_w: f64) -> PyResult<()> {
        self.t.expand(&priors, &value, &margin, margin_w).map_err(rerr)
    }
    /// Up to k simulations evaluated together (virtual loss `vloss` on each walk's path) ->
    /// ([(decider, features, legal actions)] of the distinct leaves, simulations done); then
    /// `expand_batch` with their evaluations in order.
    #[pyo3(signature = (game, k, c_puct, chance_k=0, margin_w=0.0, vloss=0.0))]
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn simulate_batch<'py>(&mut self, py: Python<'py>, game: &PyGame, k: usize, c_puct: f64, chance_k: u32, margin_w: f64, vloss: f64) -> PyResult<(Vec<(usize, Bound<'py, numpy::PyArray1<f32>>, Vec<u16>)>, u32)> {
        let (leaves, done) = self.t.simulate_batch(&*game.g, k, c_puct, chance_k, margin_w, vloss).map_err(rerr)?;
        Ok((leaves.into_iter().map(|l| (l.player, l.x.into_pyarray(py), l.acts)).collect(), done))
    }
    #[pyo3(signature = (priors, value, margin, margin_w=0.0))]
    fn expand_batch(&mut self, priors: Vec<Vec<f32>>, value: Vec<Vec<f32>>, margin: Vec<Vec<f32>>, margin_w: f64) -> PyResult<()> {
        self.t.expand_batch(&priors, &value, &margin, margin_w).map_err(rerr)
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

/// The batched driver (docs/search.md 2). `seats` = per game a list of seat dicts.
#[pyclass(name = "SearchDriver")]
pub struct PySearchDriver {
    d: Box<dyn DynDriver>,
    nfeat: usize,
}

#[pymethods]
impl PySearchDriver {
    #[new]
    #[pyo3(signature = (game, seats, spec=None, concurrency=64, seed=0, record=false, threads=4, groups=2, max_steps=100_000, rolling=false, net=0))]
    #[allow(clippy::too_many_arguments)]
    fn new(game: &str, seats: &Bound<'_, PyList>, spec: Option<&Bound<'_, PyAny>>, concurrency: usize, seed: u64, record: bool, threads: usize, groups: usize, max_steps: u64, rolling: bool, net: u32) -> PyResult<Self> {
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
        let cfg = DriverCfg { spec, seats: table, games, keys, concurrency, seed, record, threads, groups, max_steps, rolling, net };
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

    /// Network evaluation of batch `id`: priors f32 [options], value / margin f32 [B, MAX_SEATS].
    fn submit(&mut self, id: u64, priors: PyReadonlyArray1<'_, f32>, value: PyReadonlyArray2<'_, f32>, margin: PyReadonlyArray2<'_, f32>) -> PyResult<()> {
        if value.shape()[1] != MAX_SEATS || margin.shape()[1] != MAX_SEATS {
            return Err(PyValueError::new_err(format!("value / margin must be [B, {MAX_SEATS}]")));
        }
        let p = priors.as_slice().map_err(|e| PyValueError::new_err(format!("priors: {e}")))?;
        let v = value.as_slice().map_err(|e| PyValueError::new_err(format!("value: {e}")))?;
        let m = margin.as_slice().map_err(|e| PyValueError::new_err(format!("margin: {e}")))?;
        self.d.submit(id, p, v, m).map_err(verr)
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
        d.set_item("emitted", s.emitted)?;
        d.set_item("epoch", s.epoch)?;
        d.set_item("net", s.net)?;
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
        Ok(Some((b.id, key, "nn", a).into_pyobject(py)?.into_any().unbind()))
    }
}
