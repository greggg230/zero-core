//! zero-core's native worker (zero-farm, docs/az.md 44): a game's worker binary is one call -
//!
//! ```ignore
//! static GAMES: &[zero_worker::GameEntry] = &[zero_worker::GameEntry::of::<engine::az::MtgAz>("mtg"), ...];
//! fn main() { std::process::exit(zero_worker::main(GAMES, "mtg")) }
//! ```
//!
//! It plays self-play jobs of any run's settings with a `zero-onnx/1` or `/2` network (docs/az.md 41, 46) through ONNX Runtime on
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
pub mod platform;
pub mod probe;
pub mod probe_variants;
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
    /// Why a game of this spec (JSON) would replay differently off the reference platform (`SearchGame::libm_replay`,
    /// `platform` module docs), else None; an error for a spec the game refuses.
    pub libm_replay: fn(&str) -> Result<Option<String>, String>,
}

impl GameEntry {
    pub const fn of<G: SearchGame>(name: &'static str) -> GameEntry {
        GameEntry { name, new_driver: boxed_driver::<G>, info: game_info::<G>, libm_replay: libm_replay_of::<G> }
    }
}

fn libm_replay_of<G: SearchGame>(spec: &str) -> Result<Option<String>, String> {
    Ok(G::libm_replay(&G::parse_spec(spec)?))
}

/// The protocol version this crate speaks (zero-farm PROTOCOL.md 5) and the network files it reads.
pub const AW: u32 = 1;
pub const ZERO_ONNX: &[u32] = &[1, 2];

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

    /// The protocol features `hello` and `info` announce (zero-farm PROTOCOL.md 10.5): `rolling` - rolling self-play
    /// jobs (`job::Rolling`) -, `setting_classes` - check-config honours a run's declared setting classes
    /// (`classify`, docs/az.md 61) - and `infer_workers` - a self-play job line's `infer_workers` runs that many
    /// evaluator threads (docs/az.md 62.1), on every platform - in a build with self-play.
    pub fn features(&self) -> Vec<&'static str> {
        if self.selfplay {
            vec![FEATURE_ROLLING, FEATURE_SETTING_CLASSES, FEATURE_INFER_WORKERS]
        } else {
            Vec::new()
        }
    }
}

/// The feature name of rolling self-play jobs (zero-farm PROTOCOL.md 10.5).
pub const FEATURE_ROLLING: &str = "rolling";

/// The feature name of declared setting classes (`classify::CLASSES_KEY`, docs/az.md 61): a trainer adds its classes
/// to a run's settings only for a build that announces it.
pub const FEATURE_SETTING_CLASSES: &str = "setting_classes";

/// The feature name of evaluator threads (docs/az.md 62.1): the job line's `infer_workers` W (1 to 64) builds the
/// network W times on W evaluator threads, on Windows as on macOS. An agent varies W only for a build that announces
/// it (an older build ignores the field and runs one session; a Mac's build without it splits `infer_threads` itself).
pub const FEATURE_INFER_WORKERS: &str = "infer_workers";

/// The worker binary's entry point (self-play only): `args` from the command line -> the exit status.
pub fn main(games: &'static [GameEntry], repo: &'static str) -> i32 {
    main_with(Box::leak(Box::new(Worker { games, repo, selfplay: true, fuzz: None, package_settings: &[] })))
}

/// The stack of the thread the worker's command runs on: TensorRT-RTX's graph partitioning recurses deeply
/// enough to overflow the 1 MB main-thread stack of a Windows process (zero-farm PROTOCOL.md 8).
pub const MAIN_STACK: usize = 256 << 20;

/// ONNX Runtime's environment variable for the shortest sequence its CUDA MultiHeadAttention runs float32 through the
/// memory-efficient (cutlass) kernel; shorter ones take the unfused path (separate products, transposes and softmax).
pub const ORT_EFFICIENT_ATTENTION_FP32: &str = "ORT_MIN_SEQ_LEN_EFFICIENT_ATTENTION_FP32";

/// [`ORT_EFFICIENT_ATTENTION_FP32`]'s value in a worker process unless already set (docs/az.md 66): ONNX Runtime's
/// default (256) sends a token network's attention (tens of tokens a row) to the unfused path - on a farm PC's NVIDIA
/// GPU a fast graph's 1,024-row batch took 50.5 ms through it and 40.4 ms through the fused kernel (8 blocks: ~12 ms of
/// kernels -> 3.2 ms). With an attention bias the fused kernel needs a sequence of a multiple of 16: the fast graph's
/// `t_align` (`session::t_align_for`). The CPU providers never read it.
pub const ORT_EFFICIENT_ATTENTION_FP32_MIN: &str = "1";

/// The worker binary's entry point (the command runs on a thread with a [`MAIN_STACK`] stack).
pub fn main_with(w: &'static Worker) -> i32 {
    // (before any thread or ONNX Runtime exists: the variable is read when a CUDA session builds its kernels)
    if std::env::var_os(ORT_EFFICIENT_ATTENTION_FP32).is_none() {
        std::env::set_var(ORT_EFFICIENT_ATTENTION_FP32, ORT_EFFICIENT_ATTENTION_FP32_MIN);
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::thread::Builder::new()
        .name("zero-worker-main".into())
        .stack_size(MAIN_STACK)
        .spawn(move || proto::cli_with(w, &args))
        .expect("spawn the worker's main thread")
        .join()
        .unwrap_or(101)
}
