# Adding a new game

What a game repo provides on top of zero-core (Caverna Zero `cvz` and Glass Road Zero `grz` are
the worked examples; copy one and rename). Everything else - search, driver, network, loop,
Elo, dashboard, tools - comes from the pinned core (docs/pinning.md).

## 1. Layout

```
<game>-zero/
  third_party/zero-core/        git submodule, pinned (git submodule add https://github.com/greggg230/zero-core third_party/zero-core)
  Cargo.toml                    workspace: members ["engine", "py"], default-members ["engine"], exclude ["third_party"]
  engine/                       rules engine + parity harness + the AZ decision layer (impl SearchGame)
  py/                           the binding crate (cdylib <pkg>_py -> Python module <pkg>._engine)
  <pkg>/__init__.py             puts the pinned core on sys.path (copy grz/__init__.py)
  <pkg>/az/__init__.py          registers the game package (GamePackage: presets, externals, ...)
  <pkg>/az/<module>.py          shims of zero.<module> (run, dashboard, targets, h2h, ...)
  <pkg>/az/games.py             zero.games + the game's external players (e.g. the JS bot adapter)
  tools/build_py.sh             pin check + the core's tools/build_engine.sh
  tools/{supervise,kill_run,pause_run,fork_run,launch_hidden}.py   thin scripts around zero.tools
  tests/                        the game's tests (engine, bindings of its game, external players, presets, smoke)
```

## 2. Rust

1. The AZ layer implements `zero_search::SearchGame` (docs/search.md 1: decisions, chance and
   revelation keys, `determinize`, `encode`, values, `export_state` / `apply_external` for an
   external player, built-in bots). `engine/Cargo.toml`:
   `zero_search = { path = "../third_party/zero-core/crates/search" }`.
2. `py/Cargo.toml`: `crate-type = ["cdylib"]`, dependencies `zero_py` and `zero_search` (paths
   into `third_party/zero-core/crates/`), the engine crate, and
   `pyo3 = { version = "0.23", features = ["extension-module"] }`. `py/src/lib.rs`:

   ```rust
   zero_py::engine_module!(_engine, {
       "toy" => zero_search::toy::ToyGame,
       "mygame" => mygame_engine::az::MyGameAz,
   });
   ```

3. Check the trait contract with the toy-style tests (features invariant under `determinize`,
   legal actions unchanged, chance keys canonical) and the core's parity tools:
   `python -m <pkg>.az.parity positions --game mygame` (Rust vs Python search statistics).

## 3. Python

`<pkg>/__init__.py`: copy Glass Road's (it locates `third_party/zero-core/python`, fails loudly
when the submodule is missing or another core is already imported).

`<pkg>/az/__init__.py`:

```python
import os
from zero.package import GamePackage, register

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRESETS = {'mygame-smoke': dict(game='mygame', spec='{}', iters=3, games=32, ...),
           'mygame-2p': dict(game='mygame', rolling=1, infer_device='cuda', ...)}
PACKAGE = register(GamePackage(
    'mgz', 'MyGame Zero', ROOT, game='mygame', specs={'mygame': '{}'},
    dashboard_port=5300, hub_id='mygame_zero', env_prefix='MGZ',
    defaults={},                                  # zero.run.DEFAULTS overrides (e.g. ext_groups)
    presets=PRESETS,
    externals={'jsbot': 'mgz.az.games:_jsbot'},   # 'module:attr' factories, imported lazily
    decision_kinds={'mygame': 'mgz.az.games:decision_kinds'},   # optional, for zero.targets
))
```

Keep it light (no torch / numpy imports): the tools import the package to find `runctl`.
Preset and default keys must be `zero.run.DEFAULTS` keys (checked at registration).

Shims, one per core module the game's commands use (`run`, `dashboard`, `targets`, `h2h`,
`bench`, `parity`, `net`, `data`, `elo`, `infer`, `mcts`, `play`, `protocol`, `runctl`,
`toybots`):

```python
# <pkg>/az/run.py
import sys
import mgz.az  # noqa: F401  (registers the game package)
from zero import run as _core

if __name__ == '__main__':
    _core.cli()            # main() for the others
else:
    sys.modules[__name__] = _core     # mgz.az.run IS zero.run
```

`<pkg>/az/games.py`: `from zero.games import *` plus the external player classes; an external
player is `factory(seed) -> callable(LeafBatch) -> [payload str per request]` (docs/search.md
2.7), answered on worker threads (`infer.drive`), so it must be thread-safe and derive its
seeds from (factory seed, game id, the game's request count) for reproducible evaluations.

`tools/<name>.py` (supervise, kill_run, pause_run, fork_run, launch_hidden): copy Glass Road's
(they import the package, then `zero.tools.<name>.main()`). `tools/build_py.sh`: copy Glass
Road's and change the package / crate / target-dir names on its last line.

## 4. Hub, dashboard, runs

Register the dashboard in the hub (`hub/app.py` PROJECTS) with the package's port:
`python -m <pkg>.az.dashboard --host 0.0.0.0 --port <port>`, cwd = the repo. Runs live in
`<repo>/runs/` and are launched from pinned worktrees (docs/az.md 1). The cross-game dashboard
(`zerodash`) reads run files only; add the repo's runs directory to its `projects.json`.

## 5. Tests

* `cargo test` (the engine), `python -m pytest tests` (the game's tests: its engine bindings,
  the external player adapter, presets / defaults, a smoke iteration of `<game>-smoke`, the
  commands through the shims: Glass Road Zero `tests/test_loop.py` is the template).
* The core's `tools/smoke_all.sh <game repo>...` runs the core tests and each game's smoke run.
