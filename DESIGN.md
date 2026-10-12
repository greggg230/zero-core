# Zero Core - shared AlphaZero framework (design contract)

The user's AlphaZero projects share one lineage but are copies today:

| Project | Repo | Package | Notes |
|---|---|---|---|
| St. Pete Zero | `~/stpete-zero` (branch master) | `stpz` | standalone: Python engine + own Rust driver (`stpz_rs`), completed-Q targets, ladder; no PAUSE support yet |
| Dominion Zero | `~/dominion-zero` | `domz` | ancestor of the search; rolling, Mac remote workers, Vast experiments, CUDA-graph evaluator, exploration games, token-transformer encoder |
| Caverna Zero | `~/caverna-zero` | `cvz` | game-agnostic `SearchGame` search crate, rolling, completed-Q, async external eval, wasm web player + ladder |
| Glass Road Zero | `~/glassroad-zero` | `grz` | a copy of Caverna's stack + async eval + completed-Q (newest, cleanest) |

Goal: improvements made for one game (policy targets, distributed workers, eval speed-ups,
run control, web players) land once and every game can use them, without ever disturbing a
live run.

## 1. Shape

* `~/zero-core` (this repo, local git, `main`): everything game-independent.
  * Rust crate `zero_search`: the `SearchGame` trait, tree (PUCT, chance + revelation keys,
    completed-Q targets), batched self-play driver (rolling, external seats), toy game.
  * Rust scaffolding for each game's PyO3 module (e.g. a `games!` macro), so a game's
    `_engine` module = the core driver + that game's `SearchGame` impl.
  * Python package `zero`: network, inference (incl. Dominion's CUDA-graph evaluator later),
    data shards, the run loop (rolling, playout cap, value_q, policy targets, async external
    eval, PAUSE / STOP), Elo, per-game dashboard, supervisor / hidden launcher / kill / pause /
    fork tools, h2h, target-quality tools; later Dominion's remote (Mac) workers and cloud
    experiments, Caverna's wasm web player + ladder.
  * `dashboard/` (package `zerodash`): the cross-game overview (section 3).
* Each game repo keeps only what is game-specific: rules engine + parity harness vs its web
  app, AZ decision layer (actions, features, knowledge / determinization), built-in bots,
  presets, web-app glue.
* **Pinning:** each game depends on a pinned commit of zero-core (e.g. a git submodule at
  `third_party/zero-core` + Cargo path dependency + `sys.path`), never on a global install
  that would change other games' behaviour. Upgrading a game = moving its pin, at a restart
  point of its runs. Pinned run worktrees pin the core too.
* **Guard rails for every migration:** golden digests of self-play records identical before /
  after, existing checkpoints load and play identically, the game's parity tests pass, a
  cross-game smoke script in zero-core (toy + each game's smoke run) stays green.

## 2. Migration order

1. Extract the core from Glass Road Zero; switch Glass Road and Caverna to it.
2. Bring Dominion's remote (Mac) workers, cloud experiment runner and CUDA-graph evaluator
   into the core as optional modules.
3. St. Pete: its Rust engine implements `SearchGame` on the core driver (Python engine stays
   for parity); it gains PAUSE, rolling, remote workers.
4. Dominion last (its search is coupled to its token encoder), if ever.

## 3. Cross-game dashboard (`zerodash`, port 5200, hub id `zero_dash`)

One page for every game's runs (phone-friendly, binds 0.0.0.0, reached over Tailscale):

* Per project and run: state (running / pausing / paused / stopping / stopped / crashed /
  stale), iteration and when it finished, phase, latest yardstick eval, Elo + a small trend,
  throughput, plus a link into the game's own dashboard (St. Pete :5230, Dominion :5240,
  Caverna :5260, Glass Road :5280).
* Machine strip: GPU (memory, power, utilisation via nvidia-smi), CPU load, free disk on C:
  and E:, the Mac's load when reachable.
* Controls: Pause / Resume per run, **Pause all** and **Resume all**. Pause-all writes the
  standard PAUSE file with `"by": "zerodash pause-all <id>"`; Resume-all removes only PAUSE
  files that a pause-all wrote, never pauses made for other reasons (a run paused by hand
  stays paused).
* Configuration: a `projects.json` listing each project (repo, run directories - including
  `E:/stpete-zero/runs` and, once St. Pete moves, Mac runs reached over SSH -, game dashboard
  URL, pause support). No code imports from the game repos: it reads run files only.

PAUSE contract (as in Glass Road / Caverna / Dominion `runctl.py`): `<run>/PAUSE` holds JSON
`{"time": <epoch>, "by": "<who>"}`; the run notices it within seconds, stops at its next pause
point, exits with status 75; the supervisor waits and relaunches the run when PAUSE is
removed (it resumes from its newest checkpoint).

## 4. Process

* Repos are private on GitHub (`greggg230/zero-core`, `greggg230/glassroad-zero`,
  `greggg230/caverna-zero`); the desktop checkouts track them as `origin`. Cloud agents
  (Claude Code on the web) work on branches and push them; the orchestrator (desktop) fetches,
  runs the checks that need the desktop (live checkpoints, the E: parity traces, the GPU, live
  run dirs, the Mac), merges into `main`, and switches games at restart points of their runs.
* Cloud agents can't reach the desktop, its runs, E: or the Mac. Anything needing local data
  gets a script the orchestrator runs locally (e.g. checkpoint-compatibility checks).
* `dashboard/fixtures/` holds a snapshot of real run files from each project (configs, metrics
  tails, logs, run state, PAUSE files) and `reference_code/` copies of each game's dashboard /
  runctl / supervisor, so the dashboard can be built and tested without the desktop.
* Commit messages end with a blank line then
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
