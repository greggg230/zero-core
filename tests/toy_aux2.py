"""A second auxiliary target for the toy, for docs/az.md 37 (widening an auxiliary head): `next_half`, the
half of the next round's hidden card ([card < 3, card >= 3] as counts: multinomial, 2 wide; none in the last
round, as `next_card`). The toy's engine declares only `next_card` (SearchGame::AUX), so this is added on the
Python side: `install()` declares it in the core's game package (with `aux_next_half_w`), makes the engine
info the run checks the package against list it, and has the shard writer compute it from each recorded
`next_card` as the samples are written (local self-play: every self-play path of a toy run writes its shards
through `data.ShardWriter`). The run then records, builds and trains it like a target of the engine.

  python tests/toy_aux2.py --run runs/x --preset toy-smoke --aux-next-half-w 1 [...]   # zero.run with it
"""
import os
import sys

import numpy as np

HALF = {'name': 'next_half', 'width': 2, 'loss': 'multinomial'}
LAYOUT2 = [['next_card', 6, 'multinomial'], ['next_half', 2, 'multinomial']]


def half_of(next_card):
    """`next_half` [N, 2] of recorded `next_card` targets [N, 6]."""
    t = np.asarray(next_card, np.float32)
    return np.stack([t[:, :3].sum(1), t[:, 3:6].sum(1)], 1)


def install():
    """Declare `next_half` for the toy in this process (idempotent)."""
    from zero import data, games, package
    if any((t['name'] if isinstance(t, dict) else t[0]) == HALF['name'] for t in package.CORE.aux['toy']):
        return
    package.CORE.aux['toy'] = list(package.CORE.aux['toy']) + [HALF]
    run = sys.modules.get('zero.run')
    if run is not None:
        run.apply_package(package.CORE)            # (aux_next_half_w among the settings)
    info0 = games.info

    def info(game):
        d = info0(game)
        if game == 'toy':
            d['aux_layout'] = list(d['aux_layout']) + [(HALF['name'], HALF['width'])]
        return d
    games.info = info
    add0 = data.ShardWriter.add

    def add(self, s):
        if s is not None and 'aux' in s and s['aux'].shape[1] == 6:
            s = dict(s, aux=np.concatenate([s['aux'], half_of(s['aux']).astype(s['aux'].dtype)], 1))
        return add0(self, s)
    data.ShardWriter.add = add


def main(argv=None):
    install()
    from zero import run                           # (its import applies the package: aux_next_half_w)
    if argv is None:
        run.cli()
    else:
        run.main(argv)


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))
    main()
