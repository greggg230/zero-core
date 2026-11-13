//! zero-core's native worker (zero-farm, docs/az.md 44): a game's worker binary is one call -
//!
//! ```ignore
//! static GAMES: &[zero_worker::GameEntry] = &[zero_worker::GameEntry::of::<engine::az::MtgAz>("mtg"), ...];
//! fn main() { std::process::exit(zero_worker::main(GAMES, "mtg")) }
//! ```
//!
//! It plays self-play jobs of any run's settings with a `zero-onnx/1` network through ONNX Runtime on
//! the CPU (no Python, no torch) and writes zero-core's sample shards. Game-agnostic: everything here
//! is zero_search plus ONNX Runtime; the game crate only registers its `SearchGame` types.
//!
//! * `settings`: a run's settings -> the Python remote worker's self-play driver (a port of
//!   `Run.nn_seat` / `Run.driver`), parsed by zero_search's shared settings parser;
//! * `classify`: `check-config`, every run setting classified (`run_settings.json`);
//! * `eval`: the evaluator trait, the generic post-processing and the uniform evaluator;
//! * `session`: the one ONNX Runtime session factory (zero-farm PROTOCOL.md 8): execution providers in the
//!   job's order, each accepted only through the self-test, the nodes left to the CPU, `release_device`;
//!   `gpu`: the PC's adapters by LUID (DXGI); `winml`: Windows ML's provider catalog; `probe`: the
//!   `probe-providers` command (which providers pass on this PC);
//! * `job`: a self-play job's loop (the remote worker's), its control (yield / resume / finish / quit /
//!   checkpoint);
//! * `shard` / `npz`: zero-core's shard format; `proto`: the agent<->worker protocol `aw/1` and the CLI;
//! * `evaljob`: eval jobs (zero-farm PROTOCOL.md 5.2) - games of an evaluation match between two
//!   networks (or a network and a bot), played as `Run.play_match` plays them, one result row per game;
//! * `fuzz`: fuzz jobs - a game's fuzz loops over a lease's seeds in supervised chunk processes
//!   (the game registers a [`fuzz::FuzzGame`]: `zero_worker::main_with`);
//! * `selftest`: the per-machine check of a network's answers against torch's (`reference.npz`).
pub mod classify;
pub mod eval;
pub mod evaljob;
pub mod fuzz;
pub mod gpu;
pub mod job;
pub mod npz;
pub mod probe;
pub mod proto;
pub mod pyjson;
pub mod records;
pub mod selftest;
pub mod session;
pub mod settings;
pub mod shard;
pub mod winml;

use zero_search::driver::DriverCfg;
use zero_search::dynapi::{boxed_driver, game_info, DynDriver, GameInfo};
use zero_search::SearchGame;

/// One registered game type of a worker binary.
pub struct GameEntry {
    pub name: &'static str,
    pub new_driver: fn(DriverCfg) -> Result<Box<dyn DynDriver>, String>,
    pub info: fn() -> GameInfo,
}

impl GameEntry {
    pub const fn of<G: SearchGame>(name: &'static str) -> GameEntry {
        GameEntry { name, new_driver: boxed_driver::<G>, info: game_info::<G> }
    }
}

/// The protocol version this crate speaks (zero-farm PROTOCOL.md 5) and the network files it reads.
pub const AW: u32 = 1;
pub const ZERO_ONNX: &[u32] = &[1];

/// What a worker binary runs: its games' self-play and eval jobs, and its game's fuzz loops.
pub struct Worker {
    pub games: &'static [GameEntry],
    /// The game repo (`hello`'s `game`, e.g. 'mtg').
    pub repo: &'static str,
    /// Self-play and eval jobs (the network jobs; a fuzz-profile build leaves them out: kinds `["fuzz"]`).
    pub selfplay: bool,
    pub fuzz: Option<&'static dyn fuzz::FuzzGame>,
    /// The game package's own run settings and their classes (`classify::PackageSettings`; check-config).
    pub package_settings: classify::PackageSettings,
}

impl Worker {
    /// The job kinds `hello` announces.
    pub fn kinds(&self) -> Vec<&'static str> {
        let mut k = Vec::new();
        if self.selfplay {
            k.push("selfplay");
            k.push("eval");
        }
        if self.fuzz.is_some() {
            k.push("fuzz");
        }
        k
    }
}

/// The worker binary's entry point (self-play only): `args` from the command line -> the exit status.
pub fn main(games: &'static [GameEntry], repo: &'static str) -> i32 {
    main_with(Box::leak(Box::new(Worker { games, repo, selfplay: true, fuzz: None, package_settings: &[] })))
}

/// The stack of the thread the worker's command runs on: TensorRT-RTX's graph partitioning recurses deeply
/// enough to overflow the 1 MB main-thread stack of a Windows process (zero-farm PROTOCOL.md 8).
pub const MAIN_STACK: usize = 256 << 20;

/// The worker binary's entry point (the command runs on a thread with a [`MAIN_STACK`] stack).
pub fn main_with(w: &'static Worker) -> i32 {
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::thread::Builder::new()
        .name("zero-worker-main".into())
        .stack_size(MAIN_STACK)
        .spawn(move || proto::cli_with(w, &args))
        .expect("spawn the worker's main thread")
        .join()
        .unwrap_or(101)
}
