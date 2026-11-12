//! zero-core's native worker (zero-farm, docs/az.md 42): a game's worker binary is one call -
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
//! * `eval`: the ONNX Runtime evaluator (metadata checks, 64-row pieces, the generic post-processing)
//!   and the uniform evaluator;
//! * `job`: a self-play job's loop (the remote worker's), its control (yield / resume / finish / quit /
//!   checkpoint);
//! * `shard` / `npz`: zero-core's shard format; `proto`: the agent<->worker protocol `aw/1` and the CLI.
pub mod classify;
pub mod eval;
pub mod job;
pub mod npz;
pub mod proto;
pub mod pyjson;
pub mod settings;
pub mod shard;

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

/// The worker binary's entry point: `args` from the command line -> the exit status.
pub fn main(games: &'static [GameEntry], repo: &'static str) -> i32 {
    let args: Vec<String> = std::env::args().skip(1).collect();
    proto::cli(games, repo, &args)
}
