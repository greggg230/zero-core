# Pinning: how a game depends on zero-core

Every game repo depends on **one pinned commit** of zero-core, never on a global install
(DESIGN.md 1): upgrading a game = moving its pin, at a restart point of its runs, and a pinned
run worktree pins the core too.

## The mechanism

* **git submodule** `third_party/zero-core` -> `https://github.com/greggg230/zero-core` (no
  `branch` in `.gitmodules`: the pin is the commit recorded in the game's tree, the "gitlink").
* **Cargo**: the game's crates use path dependencies into the submodule -
  `zero_search = { path = "../third_party/zero-core/crates/search" }` (the engine crate, for the
  `SearchGame` trait) and `zero_py` (the binding crate, `py/`). The game's workspace
  `Cargo.toml` has `exclude = ["third_party"]` (the core has its own workspace) and the game's
  `Cargo.lock` covers the core's crates too.
* **Python**: `<pkg>/__init__.py` puts `<repo>/third_party/zero-core/python` first on
  `sys.path` and refuses to run without it (ImportError with the fix) or next to another core
  already imported. `<pkg>/az/__init__.py` registers the game package (`zero.package`). Child
  processes started by the supervisor (`python -m <pkg>.az.run`, cwd = the repo) get the same
  core the same way.
* **Build**: `bash tools/build_py.sh` (the game's) initialises the submodule when it is empty
  (`git submodule update --init third_party/zero-core`), refuses to build when the checked-out
  core is not the pinned commit (`ZERO_CORE_UNPINNED=1` overrides, for core development), then
  runs the core's `tools/build_engine.sh` (cargo build of the game's cdylib, copied to
  `<pkg>/_engine.pyd`). `CARGO_TARGET_DIR` set by the caller wins; the default is
  `E:/cargo-target/<game>/<branch>` (detached worktrees: `<game>/HEAD`, so set it per worktree).

## Workflows

Fresh clone / existing checkout after pulling a branch that has the submodule:

```
git submodule update --init third_party/zero-core      # or just: bash tools/build_py.sh
```

Pinned run worktree (the desktop's workflow, unchanged apart from the submodule being
initialised by build_py.sh):

```
git -C ~/glassroad-zero worktree add --detach ~/glassroad-zero-wt/run-<sha> <sha>
cd ~/glassroad-zero-wt/run-<sha>
CARGO_TARGET_DIR=E:/cargo-target/glassroad-zero/run-<sha> bash tools/build_py.sh   # clones the pinned core from GitHub
python tools/launch_hidden.py runs/<run>/supervisor_console.txt python tools/supervise.py C:/Users/.../glassroad-zero/runs/<run>
```

(The submodule's git directory lives under the main repo's `.git/worktrees/<name>/modules/`.
Removing such a worktree needs `git worktree remove --force` - git refuses worktrees with
submodules otherwise; check it holds no junctions first.)

Moving a game's pin (at a restart point of its runs):

```
cd ~/glassroad-zero                                  # or a branch worktree
git -C third_party/zero-core fetch origin
git -C third_party/zero-core checkout <core commit>  # a commit that is on GitHub
bash tools/build_py.sh && python -m pytest tests     # + the guard rails below
git add third_party/zero-core && git commit -m "zero-core <short sha>: ..."
```

Developing core and game together: edit and commit in `third_party/zero-core` (it is a normal
clone; push the branch to GitHub), stage the new gitlink in the game (`git add
third_party/zero-core`; build_py.sh compares against the staged pin), or build with
`ZERO_CORE_UNPINNED=1` while iterating. Pins must only reference commits reachable on GitHub:
**merge zero-core branches fast-forward / with merge commits, never squash or rebase a branch
that a game pins** (the pinned commit would disappear).

## Guard rails when a pin moves

* Golden digests and checkpoint compatibility, old checkout vs new checkout (each must be built):

  ```
  python third_party/zero-core/tools/check_checkpoint_compat.py --old <old checkout> --new <new checkout> \
      --package grz --net random --game toy
  python third_party/zero-core/tools/check_checkpoint_compat.py --old <old> --new <new> --package grz --net random --game glassroad
  python third_party/zero-core/tools/check_checkpoint_compat.py --old <old> --new <new> --package grz \
      --model runs/grz2p_cq/models/iter_NNNN.pt
  ```

  Identical self-play record digests (policy targets visits and cq) and match results, or the
  change is a deliberate behaviour change to be documented (and the old runs not resumed on it
  silently).
* A pin (or a game commit) that changes the game's **encoding** (feature or action counts, a
  token layout, a board grid): the run's checkpoints are networks of the old encoding, and the
  core's encoding guard refuses to resume them (exit 78, `docs/az.md` 10.4) - move the run with
  `tools/fork_run.py` (a widen into a new run, 10.3). check_checkpoint_compat itself takes such a
  checkpoint on purpose (it compares the two builds).
* The game's `cargo test` and `pytest`, its engine parity gate, the core's
  `tools/smoke_all.sh`.
