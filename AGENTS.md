# Notes for coding agents working on zero-core

Read [`README.md`](README.md) first: what the repo is, how to build it, the concepts. This file
is the rules of the road for **changing** it. Several game repos pin commits of this repo and
train long runs on them, so the main risk of a core change is silently changing how an
existing game plays or trains.

## Setup and the checks you can always run

```bash
export CARGO_TARGET_DIR="$PWD/target"   # one target dir per checkout (see "Pitfalls")
export RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup"   # macOS only (no build.rs adds them)
bash tools/build_py.sh                  # builds python/zero/_engine (the toy games); rerun after any Rust change
cargo test                              # zero_search + zero_webnet, ~15 s
python -m pytest tests -q               # the toy tests, ~10 min; -k <name> while iterating
PYTHONPATH=python python -m zero.run --run runs/toy-smoke --preset toy-smoke    # end-to-end loop, ~5 s
```

Python 3.13 (PyO3 0.23 does not build for 3.14) with `torch numpy psutil flask pytest`. Use Git
Bash on Windows. Every script is bash.

## Ground rules

1. **Defaults are behaviour-preserving.** A new capability arrives as a run setting (a key in
   `zero.run.DEFAULTS`) that is **off by default**, and with it off, runs play and train
   exactly as before, bit for bit. Prove it with a test, and for anything near the search or
   self-play, with golden digests (`tools/selfplay_digest.py`, `tools/check_checkpoint_compat.py
   --old <checkout before> --new <checkout after> --package <pkg> --net random --game <game>`).
   A deliberate behaviour change has to be called out as one and documented. It must never
   ride along unannounced.
2. **Never rewrite published history.** Games pin commits by sha, so a squash or rebase of a
   pinned branch deletes the commit a game's submodule points to. Merge with fast-forwards or
   merge commits.
3. **Don't break the games.** Every core change has to keep compiling against the game repos
   (`bash tools/smoke_all.sh --with-this-core --build-only`, run from a checkout with your
   change committed; a worktree is fine, and its summary must say `games: N of N checked` - it
   fails when it finds no game, `docs/az.md` 29). A change to search or loop behaviour also
   needs the full `bash tools/smoke_all.sh --with-this-core` (each game's build, tests and
   smoke run).
4. **Encodings are guarded.** A checkpoint is a network for one feature layout. The encoding
   guard (`net.check_encoding`, `docs/az.md` 10.4) refuses to resume across a layout change
   (exit 78). Don't weaken it. Runs move across layouts with `tools/fork_run.py`, never by
   resuming.
5. **Keep game-package imports light.** `zero.package` and a game's `<pkg>/az/__init__.py`
   must not import torch or numpy, because the run-control tools import them to find `runctl`.
   Preset and default keys must be `zero.run.DEFAULTS` keys, which `register` checks.
6. **Run control is a contract** shared with the games and `zerodash`: `<run>/STOP`,
   `<run>/PAUSE` (JSON `{"time", "by"}`, exit 75), exit 78 = refused, `run_state.json` /
   `supervisor_state.json`. Don't change these files or exit codes casually (`docs/az.md`
   section 2).
7. **Never touch a live run.** Runs train from their own pinned worktrees. Don't edit, delete or
   resume anything under someone's `runs/` unless you were asked to. `runs/` is gitignored.

## Documenting a change

The docs are a lab notebook, and each feature gets a numbered, dated section:

* A run setting: add it to the settings table in `docs/az.md` 1.2 (name, default, meaning,
  plus a pointer to its section), and give it a section of its own with **why**, **what it
  does**, **off by default** (how to turn it on), **costs**, **tests and checks**, and
  **measured** (real numbers, the machine, the date).
* Search semantics and the driver protocol go in `docs/search.md`. Adding-a-game steps go in
  `docs/new_game.md`. The pinning workflow is in `docs/pinning.md`.
* Keep the README's quick start working. If you change a command it shows, run it.

Commit messages say what changed, the setting names, and how it was checked (tests,
digests). An agent's commit ends with its own `Co-Authored-By:` trailer.

## Pitfalls

* **One `CARGO_TARGET_DIR` per checkout.** Two checkouts sharing a target dir can silently reuse
  each other's build (cargo's dep-info paths are workspace-relative), so a worktree "tests" code
  it doesn't contain. `tools/build_py.sh` defaults to `E:/cargo-target/zero-core/<branch>` on
  Windows (the maintainer's machine) and to `<repo>/target` elsewhere. Set it explicitly.
* **Windows locks a loaded `.pyd`.** `build_engine.sh` moves the old one aside (`*.old`) if a
  running process holds it. Processes started earlier keep the old build.
* **`docs/` mention the maintainer's setup**: the desktop, the E: drive, a MacBook remote
  (`eval_remote` / `selfplay_remote` = `mac`), the hub and its port registry, Tailscale. You
  don't need any of it. Remote settings default to empty, meaning "everything on this
  machine".
* **Long tests.** The full pytest run takes ~10 minutes. Use `-k` while iterating, then run it
  all once before you push. `tests/test_remoteplay.py` times real threads and can fail on a
  loaded machine: rerun it alone before you call it a regression.
