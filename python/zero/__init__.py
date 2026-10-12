"""Zero Core: the shared AlphaZero framework (DESIGN.md, docs/az.md, docs/search.md).

Modules: `games` (game types of the active package's compiled module, seats, external players,
the driver wrapper), `protocol` (leaf batches), `infer` (network evaluation, the drive loop),
`net`, `data`, `run` (the training loop + preset registry), `runctl` (PAUSE / STOP), `elo`,
`dashboard`, `mcts` / `parity` / `play` (Python reference search), `toybots`, `bench`, `h2h`,
`targets`, `tools` (supervise / kill / pause / fork / launch_hidden). A game package registers
itself with `zero.package.register(...)` (docs/new_game.md); without one the core runs its own
toy game (`zero._engine`).

Light on purpose: importing `zero` (or `zero.runctl`, `zero.package`) does not import torch.
"""
from .package import GamePackage, active, register  # noqa: F401

__version__ = '0.1.0'
