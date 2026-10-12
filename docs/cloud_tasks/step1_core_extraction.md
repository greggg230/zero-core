# Cloud task: Zero Core step 1 - extract the shared framework, migrate Glass Road + Caverna

Start a Claude Code cloud session on `greggg230/zero-core` with this prompt:

---

You are working on **zero-core**, a shared AlphaZero framework for the user's game projects. Read `DESIGN.md` in this repo fully first: it is the contract (sections 1, 2 step 1, and 4 "Process" apply to you). Your job is **step 1: extract the core from Glass Road Zero and switch Glass Road Zero and Caverna Zero onto it**, on branches, pushing them to GitHub. An orchestrator on the user's Windows desktop will fetch your branches, run the checks that need the desktop, and merge.

## Repos (all private, you have push access; clone the two game repos next to this one, use the add_repo tool with access "push" if needed)
- `greggg230/zero-core` (this repo) -> work on branch `core`. Do NOT touch `dashboard/` (another agent owns it on branch `dashboard`).
- `greggg230/glassroad-zero` (package `grz`, main) -> branch `zero-core`.
- `greggg230/caverna-zero` (package `cvz`, main) -> branch `zero-core`.
Push all three branches early and often. Never push to `main` of any repo. No pull requests.

## Background
Glass Road Zero is a copy of Caverna Zero's game-independent stack plus later improvements; Caverna has since received the same improvements (async external-player evaluation `ext_groups` / `_drive_async`, completed-Q policy targets `policy_target`) plus extras (wasm web player `wasm/` + `web/`, ladder tools, `cvz/az/h2h.py`, `cvz/az/targets.py`). Diff them first: `search/`, `py/`, `grz/az/*` vs `cvz/az/*`, `tools/*`, `tests/*`, `docs/search.md`, `docs/az.md`. The union of the game-independent parts is the core. Each repo's `DESIGN.md`, `docs/search.md`, `docs/az.md` explain the design. Game-specific parts stay in each game repo: `engine/` (rules port + AZ layer: actions, features, knowledge), `reference/` (vendored JS engine + harness), game presets, the JS-bot external seat for each game's Node bot server, web glue, game-specific tests.

## Deliverables
1. zero-core (branch `core`):
   - Rust crate `zero_search` (from `search/`: SearchGame trait, tree incl. completed-Q targets, driver incl. rolling + external seats, dynapi, noise, rng, toy game + tests).
   - Rust support for each game's PyO3 module so a game's `_engine` module = core driver + its game(s) with minimal per-game code (e.g. a macro or generic crate the game's `py/` crate calls); keep the toy game available.
   - Python package `zero` (from `grz/az` / `cvz/az`: net, infer, data, run loop with a preset registry games extend, runctl, reference MCTS, parity, elo, dashboard, games/externals registry, play, protocol, bench, h2h, targets, toybots) and the tools (supervise, launch_hidden (Windows), kill_run, pause_run, fork_run, a build helper) parameterised by the game package instead of hard-coded `grz`/`cvz`. Caverna's wasm/web player + ladder: move the game-independent parts into the core if feasible, else list them for step 2.
   - Tests (toy: Rust vs Python search parity, loop resume/STOP/PAUSE, fork, policy targets) + `tools/smoke_all.sh` running the core tests and each migrated game's smoke test.
   - Docs: `docs/search.md`, `docs/az.md` (moved from the games), a short "adding a new game" guide.
2. **Pinning** (DESIGN.md 1): each game depends on a pinned zero-core commit, e.g. a git submodule `third_party/zero-core` pointing at `https://github.com/greggg230/zero-core` + Cargo path dependency + the Python package on sys.path. Must work with the desktop's workflow: runs are launched from pinned git worktrees (`git worktree add --detach ...` then `bash tools/build_py.sh`), on Windows with Git Bash, system Python 3.13, `CARGO_TARGET_DIR` set by the caller. Document how to init/update the submodule in a fresh worktree.
3. Migrate both games on their `zero-core` branches: delete the duplicated code, use the core; keep the commands that live runs and the user's hub rely on working, or thin shims with identical CLIs: `python -m grz.az.run`, `python -m grz.az.dashboard --host 0.0.0.0 --port 5280`, `python -m cvz.az.dashboard --host 0.0.0.0 --port 5260`, `python tools/supervise.py <run>`, `tools/pause_run.py`, `tools/kill_run.py`, `tools/fork_run.py`, `tools/launch_hidden.py`, `tools/build_py.sh`, `python -m cvz.az.h2h`, `python -m cvz.az.targets`, `python -m grz.az.targets`. A run directory created by the old code (its `config.json` keys, `metrics.jsonl`, `elo/`, model checkpoints, data shards) must resume with the new code.
4. **Guard rails (all must pass; report numbers):**
   - Golden digests: self-play record digests for `policy_target` visits and cq on the toy and on each game, identical before (old `main`) and after.
   - Checkpoint compatibility: save checkpoints with the old code (e.g. a short smoke run), load them with the new code, same evaluation results for fixed seeds. Also write `tools/check_checkpoint_compat.py` that the orchestrator will run on the desktop against the real live checkpoints (it compares old-main vs new-branch play on a given .pt with fixed seeds).
   - Each game's full existing test suite (`cargo test`, `pytest`), and engine parity: Glass Road `cargo run --release --bin parity -- --prefix p2_ <traces>` on traces you generate with `node reference/harness/gen_all.mjs` (2p; a reduced set is fine here, e.g. 1,000 games per intro setting - the orchestrator runs the full 20,000-game gate on the desktop); Caverna's engine parity tests likewise.
   - A short smoke training run per game (CPU) and the toy loop proof.
5. Final summary (as your last message): layout, pinning mechanism, what moved where, test/digest/parity numbers, the exact commands for the orchestrator to verify on the desktop and to switch live runs (at a restart point, from new pinned worktrees), and what's left for step 2 (Dominion's remote Mac workers, Vast experiments, CUDA-graph evaluator - their code is not in these repos; just leave clean extension points) and step 3 (St. Pete).

## Notes
- You are on Linux in the cloud; the desktop is Windows. Keep Windows-only bits (launch_hidden, wmic usage, priority classes) working on Windows and skipped/guarded on Linux. There is no GPU here: CUDA paths must still be exercised by the desktop later; keep `--device cuda` options intact.
- Install what you need (Rust stable, Node 22, Python 3.13 if available else 3.12, torch CPU, numpy, pytest, flask, maturin).
- Commit messages end with a blank line then `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
