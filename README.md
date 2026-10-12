# Zero Core

Shared AlphaZero framework for the St. Pete, Dominion, Caverna and Glass Road Zero projects, plus the cross-game dashboard (`dashboard/`, port 5200). See DESIGN.md.

| Path | What |
|---|---|
| `crates/search` (`zero_search`) | `SearchGame` trait, MCTS tree (chance + revelation keys, completed-Q targets), batched self-play driver (rolling, external seats), the toy game (docs/search.md) |
| `crates/py` (`zero_py`) | PyO3 classes behind every game's `<pkg>._engine`; a game's binding crate is one `zero_py::engine_module!` call |
| `crates/toy_py` | the core's own `zero._engine` (toy only) |
| `python/zero` | network, inference, data, the run loop + preset registry, run control (PAUSE / STOP), Elo, per-game dashboard, reference MCTS + parity, h2h, policy-target tools, `zero.package` (the game package registry), `zero.tools` (supervise, kill, pause, fork, launch_hidden) (docs/az.md) |
| `tools/` | `build_engine.sh` (every repo's PyO3 build), `build_py.sh`, `smoke_all.sh` (core + every game), `selfplay_digest.py` / `check_checkpoint_compat.py` (old vs new checkout), the tool scripts for toy runs |
| `tests/` | the toy tests (Rust vs Python search parity, loop resume / STOP / PAUSE, supervisor, fork, policy targets, external seats) |
| `docs/` | `search.md`, `az.md`, `new_game.md` (adding a game), `pinning.md` (how games pin the core) |
| `dashboard/` | `zerodash`, the cross-game dashboard (separate package) |

Games pin a commit of this repo as the git submodule `third_party/zero-core` (docs/pinning.md):
Glass Road Zero (`grz`) and Caverna Zero (`cvz`) so far.

```
bash tools/build_py.sh                    # python/zero/_engine.pyd (toy); CARGO_TARGET_DIR or E:/cargo-target/zero-core/<branch>
cargo test                                # zero_search
python -m pytest tests                    # the toy tests
bash tools/smoke_all.sh                   # + each sibling game repo's build, tests, smoke run
```
