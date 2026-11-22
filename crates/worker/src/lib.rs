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
    /// A game NAME whose meaning changed (zero-core `GamePackage.game_aliases`, docs/az.md 73): this entry is the name's
    /// layout revision `layout` - settings, graphs and headers naming it carry `game_layout` = that revision. None: a
    /// name that never changed (no `game_layout` read or written).
    pub layout: Option<u32>,
    /// This entry is the game an aliased name meant before its change (`before_of` = that name): settings or a graph
    /// naming that name without a `game_layout` play this entry. None: no name's old game.
    pub before_of: Option<&'static str>,
}

impl GameEntry {
    pub const fn of<G: SearchGame>(name: &'static str) -> GameEntry {
        GameEntry { name, new_driver: boxed_driver::<G>, info: game_info::<G>, libm_replay: libm_replay_of::<G>, layout: None, before_of: None }
    }

    /// This entry as layout revision `rev` of its (aliased) name (docs/az.md 73).
    pub const fn layout(mut self, rev: u32) -> GameEntry {
        self.layout = Some(rev);
        self
    }

    /// This entry as the game the name `name` meant before its layout revision (docs/az.md 73).
    pub const fn before_of(mut self, name: &'static str) -> GameEntry {
        self.before_of = Some(name);
        self
    }

    /// The layout record written with this game's name (`game_layout`), or nothing.
    pub fn stamp(&self, into: &mut serde_json::Map<String, serde_json::Value>) {
        if let Some(r) = self.layout {
            into.insert("game_layout".into(), serde_json::json!(r));
        }
    }
}

/// The entry a record naming game `name` with layout record `layout` (`game_layout`; None: none) is (zero-core
/// `GamePackage.resolve_game`, docs/az.md 73): an aliased name (an entry with a `layout`) without a layout record is the
/// entry `before_of` it, with its revision the entry itself; another revision, or a layout record of a name that never
/// changed, is refused - a layout is never guessed.
pub fn resolve_game<'a>(games: &'a [GameEntry], name: &str, layout: Option<u64>) -> Result<&'a GameEntry, String> {
    let names = || games.iter().map(|g| match g.layout { Some(r) => format!("{} (layout {r})", g.name), None => g.name.to_string() }).collect::<Vec<_>>();
    match (games.iter().find(|g| g.name == name), layout) {
        (Some(g), Some(l)) if g.layout.map(u64::from) == Some(l) => Ok(g),
        (Some(g), None) if g.layout.is_none() => Ok(g),
        (Some(g), None) => games.iter().find(|e| e.before_of == Some(name)).ok_or_else(|| format!("game {name:?} without a game_layout: the game {name:?} meant before its layout {} - this worker has none (docs/az.md 73)", g.layout.unwrap_or(0))),
        (Some(g), Some(l)) => Err(format!("game {name:?} with game_layout {l}: this worker plays {name:?} {} (docs/az.md 73)", g.layout.map_or("with no layout revision".into(), |r| format!("as layout {r}")))),
        (None, None) if games.iter().any(|e| e.before_of == Some(name)) => Ok(games.iter().find(|e| e.before_of == Some(name)).unwrap()),
        (None, _) => Err(format!("game {name:?}{}: this worker plays {:?}", layout.map_or(String::new(), |l| format!(" with game_layout {l}")), names())),
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
    /// (`classify`, docs/az.md 61) -, `infer_workers` - a self-play job line's `infer_workers` runs that many
    /// evaluator threads (docs/az.md 62.1), on every platform - and `seat_settings` - an eval job's network seat may
    /// carry its own settings (`{"net": gen, "set": {...}}`, docs/az.md 68) - in a build with self-play.
    pub fn features(&self) -> Vec<&'static str> {
        if self.selfplay {
            vec![FEATURE_ROLLING, FEATURE_SETTING_CLASSES, FEATURE_INFER_WORKERS, FEATURE_SEAT_SETTINGS]
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

/// The feature name of per-seat evaluation settings (docs/az.md 68; zero-farm PROTOCOL.md 5.2): an eval job's network
/// seat `{"net": gen, "set": {<setting>: <value>}}` plays the job's settings with its set over them (`settings::
/// SEAT_READS` keys only). The desktop sends a seat with a set only to a build that announces it (an older build refuses
/// the seat: "one key, net or bot").
pub const FEATURE_SEAT_SETTINGS: &str = "seat_settings";

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

#[cfg(test)]
mod tests {
    use super::*;
    use zero_search::toy::ToyGame;

    // (docs/az.md 73: "toy" is the stand-in for an aliased name - revision 2 - whose old game is "toy_old")
    static GAMES: &[GameEntry] = &[GameEntry::of::<ToyGame>("toy").layout(2), GameEntry::of::<ToyGame>("toy_old").before_of("toy"), GameEntry::of::<ToyGame>("other")];

    #[test]
    fn an_aliased_name_resolves_by_its_layout_record() {
        assert_eq!(resolve_game(GAMES, "toy", Some(2)).unwrap().name, "toy");
        assert_eq!(resolve_game(GAMES, "toy", None).unwrap().name, "toy_old");
        assert_eq!(resolve_game(GAMES, "toy_old", None).unwrap().name, "toy_old");
        assert_eq!(resolve_game(GAMES, "other", None).unwrap().name, "other");
        assert!(resolve_game(GAMES, "toy", Some(3)).err().unwrap().contains("as layout 2"));
        assert!(resolve_game(GAMES, "toy_old", Some(2)).is_err());
        assert!(resolve_game(GAMES, "other", Some(1)).is_err());
        assert!(resolve_game(GAMES, "nope", None).is_err());
        // a build that registers only the old game still plays the name without a layout record
        static OLD: &[GameEntry] = &[GameEntry::of::<ToyGame>("toy_old").before_of("toy")];
        assert_eq!(resolve_game(OLD, "toy", None).unwrap().name, "toy_old");
        assert!(resolve_game(OLD, "toy", Some(2)).is_err());
        let mut m = serde_json::Map::new();
        GAMES[0].stamp(&mut m);
        assert_eq!(m.get("game_layout"), Some(&serde_json::json!(2)));
        let mut m = serde_json::Map::new();
        GAMES[1].stamp(&mut m);
        assert!(m.is_empty());
    }

    #[test]
    fn a_graph_is_the_jobs_game_by_name_and_layout_record() {
        let new = eval::Expect::of_entry(&GAMES[0], false);
        let old = eval::Expect::of_entry(&GAMES[1], false);
        assert!(new.game_problem(Some("toy"), Some("2")).is_none());
        assert!(new.game_problem(Some("toy"), None).is_some());       // a graph of before the change: the old game's
        assert!(new.game_problem(Some("toy_old"), None).is_some());
        assert!(new.game_problem(Some("toy"), Some("x")).is_some());
        assert!(old.game_problem(Some("toy_old"), None).is_none());
        assert!(old.game_problem(Some("toy"), None).is_none());       // exported before the change, as "toy"
        assert!(old.game_problem(Some("toy"), Some("2")).is_some());
        let plain = eval::Expect::of_entry(&GAMES[2], false);
        assert!(plain.game_problem(Some("other"), None).is_none());
        assert!(plain.game_problem(Some("other"), Some("1")).is_some());
    }
}
