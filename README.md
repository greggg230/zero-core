# Zero Core

A shared **AlphaZero** framework: a fast, game-agnostic Monte Carlo tree search in Rust plus the
self-play / training / evaluation loop in Python (PyTorch). Several game projects are built on
it. Each one supplies only its rules engine and how the game looks to a network:

| Game repo | Package | Game |
|---|---|---|
| [mtg-zero](https://github.com/greggg230/mtg-zero) | `mtgz` | Magic: The Gathering: team cube drafting and constructed play |
| [kingdombuilder-zero](https://github.com/greggg230/kingdombuilder-zero) | `kbz` | Kingdom Builder |
| [caverna-zero](https://github.com/greggg230/caverna-zero) | `cvz` | Caverna |
| [glassroad-zero](https://github.com/greggg230/glassroad-zero) | `grz` | Glass Road |

> **Which repo do you want?** If you want to train or play one of the games, start in **that
> game's repo**. It pulls in the exact zero-core commit it needs as a git submodule, so you
> don't clone this repo separately. Come here to work on the framework itself or to add a new
> game.

For coding agents (Claude Code, Codex, ...): read [`AGENTS.md`](AGENTS.md) too. It has the rules
of the road for changing this repo.

## How it fits together

```
 a game repo (e.g. mtg-zero)                       zero-core (this repo, pinned in the game
 ─────────────────────────                         repo as third_party/zero-core)
 engine/   Rust rules engine                       crates/search   SearchGame trait, MCTS tree,
           + `impl SearchGame` (actions,                           batched multi-threaded driver
             features, hidden info, chance)        crates/py       PyO3 classes behind <pkg>._engine
 py/       one-line binding crate  ───────────▶    python/zero     network, inference, training
           (zero_py::engine_module!)                               loop, Elo, dashboard, tools
 <pkg>/az  thin shims: `python -m <pkg>.az.run`    dashboard/      zerodash: every game's runs
           IS `zero.run`, registered presets                       on one page
```

One iteration of the loop (`python/zero/run.py`):

1. **Self-play.** The Rust driver plays hundreds of games at once. Each MCTS simulation that
   reaches a new position sends that position's features to Python in a batch. The network
   scores the batch, the Rust trees continue, and every searched decision becomes a training
   sample (`runs/<run>/data/iter_NNNN/*.npz`).
2. **Training.** PyTorch trains the policy and value heads on a sliding window of recent
   iterations.
3. **Evaluation.** The new checkpoint plays a fixed **yardstick** opponent (usually the game's
   built-in heuristic bot, rated Elo 1000), a random player, and earlier checkpoints. The
   result is an Elo curve (`elo/ratings.json`, the dashboard).

## Prerequisites

| | Version we use | Notes |
|---|---|---|
| Git + **Bash** | | The build scripts are bash. On Windows, use **Git Bash**. |
| Rust | stable, 1.95 | [rustup.rs](https://rustup.rs). Windows needs the MSVC Build Tools, macOS needs the Xcode command line tools, Linux needs a C compiler. |
| Python | **3.13** | PyO3 0.23 builds for up to 3.13, so **3.14 won't work**. The interpreter called `python` is the one the engine is built for. To build for another one, set `PYTHON=/path/to/python`. |
| Python packages | torch 2.11, numpy 2.3 | `pip install torch numpy psutil flask pytest`. The CPU-only torch is fine for everything here. |
| GPU | optional | NVIDIA + CUDA torch for real training runs (`infer_device` / `train_device` = `cuda`) |

## Quick start (about 10 minutes)

```bash
git clone https://github.com/greggg230/zero-core.git
cd zero-core

python -m venv ~/.venvs/zero                 # outside the repo
source ~/.venvs/zero/bin/activate            # Git Bash on Windows: source ~/.venvs/zero/Scripts/activate
pip install torch numpy psutil flask pytest

# Windows only: the scripts default to the maintainer's E: drive. Pick a target dir.
export CARGO_TARGET_DIR="$PWD/target"
# macOS only: the Python extension must resolve Python's symbols at load time
export RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup"

bash tools/build_py.sh                       # ~30 s: builds the toy engine into python/zero/_engine.{pyd,so}
cargo test                                   # ~15 s: the Rust search (zero_search, zero_webnet)
python -m pytest tests -q                    # ~10 min: Rust-vs-Python parity, the loop, resume, PAUSE/STOP, ...
PYTHONPATH=python python -m zero.run --run runs/toy-smoke --preset toy-smoke   # a 2-iteration toy run, ~5 s
```

If the last command prints iterations with `selfplay`, `train` and `elo` lines and ends with
`done`, everything works. (A few tests in `tests/test_remoteplay.py` time real threads. On a
heavily loaded machine one can fail; rerun that file on its own.) Look in `runs/toy-smoke/`: `metrics.jsonl` has one JSON line per
iteration, `models/` holds the checkpoints, and `log.txt` is the log.

A longer toy run that actually learns (section 6 of [`docs/az.md`](docs/az.md): 24 iterations,
about 15 minutes on a desktop CPU):

```bash
PYTHONPATH=python python -m zero.run --run runs/toy1 --preset toy-1
PYTHONPATH=python python -m zero.dashboard --runs runs --port 8000    # then open http://<this machine>:8000
```

## Concepts in 2 minutes

* **`SearchGame`.** The Rust trait a game implements (`crates/search/src/game.rs`, explained
  in [`docs/search.md`](docs/search.md) section 1). It covers the legal actions as `u16` ids,
  `step`, the result, a fixed-width feature vector (`encode`), and the hidden-information
  hooks. `determinize(viewer)` resamples what that player can't see. Chance outcomes are
  reported as keys, so the tree can branch on them.
* **Driver.** Runs many games and trees in parallel and batches their network requests (the
  leaf protocol, `docs/search.md` section 2). With `rolling` self-play, one persistent driver
  keeps `conc` games in flight across iterations.
* **Game package.** How a game registers with the Python side (`zero.package.GamePackage`): its
  compiled module, presets, default spec, external players and dashboard port.
* **Preset.** A named bundle of run settings (`toy-smoke`, `toy-1`, and each game's own, like
  `mtg-smoke`). Settings resolve in this order: `run.DEFAULTS` < preset < the run's saved
  `config.json` < command-line flags. A run directory keeps its settings forever. The table
  of every setting is in `docs/az.md` section 1.2, or run `python -m zero.run --help`.
* **Run directory.** Everything about one run lives in `runs/<name>/`: config, metrics, logs,
  checkpoints, data shards, Elo. Restarting the same command resumes the run from its newest
  checkpoint.
* **Run control.** `touch runs/<run>/STOP` stops the run cleanly after the current iteration.
  `python tools/pause_run.py runs/<run>` pauses it within seconds (`--resume` to continue).
  `python tools/supervise.py runs/<run> --preset ...` restarts a run after a crash.
* **Pin.** A game repo records one exact zero-core commit (`third_party/zero-core`), and
  upgrading the core in a game is a deliberate "move the pin" commit. See
  [`docs/pinning.md`](docs/pinning.md).
* **Opt-in features.** Most of `docs/az.md` from section 9 on describes features that are
  **off by default** (completed-Q targets, Gumbel root search, sparse batches, the league,
  belief heads, ...). Each section explains why the feature exists, what it measured and how to
  turn it on.

## Repository map

| Path | What |
|---|---|
| `crates/search` (`zero_search`) | `SearchGame` trait, MCTS tree (chance + revelation keys, completed-Q targets), batched self-play driver (rolling, external seats), the toy game (docs/search.md) |
| `crates/py` (`zero_py`) | PyO3 classes behind every game's `<pkg>._engine`; a game's binding crate is one `zero_py::engine_module!` call |
| `crates/toy_py` | the core's own `zero._engine` (toy games only) |
| `crates/webnet` (`zero_webnet`) | the network's forward pass over float16 `.bin` weights for browser players (native = wasm32 bit for bit) |
| `python/zero` | network, inference, data, the run loop + preset registry, run control (PAUSE / STOP), Elo, per-game dashboard, reference MCTS + parity, h2h, replays, `zero.package` (the game package registry), `zero.tools` (supervise, kill, pause, fork, launch_hidden) |
| `tools/` | `build_engine.sh` (every repo's PyO3 build), `build_py.sh`, `smoke_all.sh` (core + every game), `selfplay_digest.py` / `check_checkpoint_compat.py` (old vs new checkout), the run-control scripts |
| `tests/` | the toy tests (Rust vs Python search parity, loop resume / STOP / PAUSE, supervisor, fork, policy targets, external seats, ...) |
| `docs/` | see below |
| `dashboard/` | `zerodash`, the cross-game dashboard (a separate package; its `projects.json` lists the maintainer's machines and paths) |

## Docs: what to read when

| You want to... | Read |
|---|---|
| understand the design and why the core exists | [`DESIGN.md`](DESIGN.md) (short) |
| run, control and read a training run | [`docs/az.md`](docs/az.md) sections 1-5 (running, run control, files, training, evaluation and Elo) |
| understand the search: trait, driver protocol, MCTS details | [`docs/search.md`](docs/search.md) sections 1-3 |
| add a new game | [`docs/new_game.md`](docs/new_game.md). Copy Glass Road Zero or Caverna Zero and rename. |
| change which core a game uses | [`docs/pinning.md`](docs/pinning.md) |
| record games for a viewer | [`docs/replays.md`](docs/replays.md) |
| learn what a specific opt-in feature does | `docs/az.md`, the section that names the setting |

`docs/az.md` and `docs/search.md` are long (thousands of lines) and written as a lab notebook.
Every section is dated and gives measured results. Search them for a setting name instead of
reading them top to bottom.

## Everyday commands

```bash
bash tools/build_py.sh                         # rebuild the toy engine after Rust changes
cargo test                                     # Rust tests
python -m pytest tests -q                      # Python tests (toy games)
PYTHONPATH=python python -m zero.run --help    # every run setting
PYTHONPATH=python python -m zero.h2h A.pt B.pt --games 200    # head-to-head of two toy checkpoints
bash tools/smoke_all.sh                        # core tests + each sibling game repo's build, tests and smoke run
```

`smoke_all.sh` looks for the game repos next to this one (`../glassroad-zero`, `../caverna-zero`,
`../kingdombuilder-zero`; or pass their paths). Some games' yardsticks are JavaScript bots, so
it needs Node.js too. In a game repo, the same commands go through the game's package, for
example `python -m mtgz.az.run`, `python -m mtgz.az.h2h` and `python -m mtgz.az.dashboard`.

## Things that look odd but are intentional

* **The E: drive, the Mac, Tailscale and "the desktop".** The docs often mention the
  maintainer's machines: `E:/cargo-target/...` as the default Windows build dir, a MacBook used
  as a remote worker (`eval_remote` / `selfplay_remote` = `mac`), and port numbers registered
  in a local hub. None of these is required. Set `CARGO_TARGET_DIR` and leave the remote
  settings empty.
* **Never a global install.** `zero` is never `pip install`ed. Each game checkout puts its own
  pinned copy on `sys.path` (`<pkg>/__init__.py`), so two games, or a live run and a branch
  being edited, can't change each other's behaviour.
* **Defaults never change behaviour.** A new feature arrives as a setting that is off by
  default, and the tests check that a run without it plays the same games bit for bit. Old runs
  can resume on new code.
* **Live runs run from a separate pinned worktree** (`git worktree add --detach ...`), never
  from the checkout someone is editing.

## Contributing

Work on a branch, push it, and open a PR against `main`. Never squash or rebase a branch that a
game already pins: the pinned commit would disappear from GitHub and that game's submodule
would break. Merge with fast-forwards or merge commits instead. See [`AGENTS.md`](AGENTS.md) for
the checks a change needs.
