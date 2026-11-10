"""A second auxiliary target for the toy, for docs/az.md 37 (widening an auxiliary head): `next_parity`, the
parity of the next round's hidden card ([even, odd] as counts: multinomial, 2 wide; none in the last round, as
`next_card`). Not public: the next round's stage (rounds 1-3: cards 0-2, 4-6: cards 3-5) has one or two even
cards, and the revealed ones say which are left (`best_public`: the best loss from public information).

The toy's engine declares only `next_card` (SearchGame::AUX), so this is added on the Python side: `install()`
declares it in the core's game package (with `aux_next_parity_w`), makes the engine info the run checks the
package against list it, and has the shard writer compute it from each recorded
`next_card` as the samples are written (local self-play: every self-play path of a toy run writes its shards
through `data.ShardWriter`). The run then records, builds and trains it like a target of the engine.

  python tests/toy_aux2.py --run runs/x --preset toy-smoke --aux-next-parity-w 1 [...]   # zero.run with it
"""
import os
import sys

import numpy as np

PARITY = {'name': 'next_parity', 'width': 2, 'loss': 'multinomial'}
LAYOUT2 = [['next_card', 6, 'multinomial'], ['next_parity', 2, 'multinomial']]


def parity_of(next_card):
    """`next_parity` [N, 2] of recorded `next_card` targets [N, 6]."""
    t = np.asarray(next_card, np.float32)
    return np.stack([t[:, 0::2].sum(1), t[:, 1::2].sum(1)], 1)


def best_public(x, t):
    """The best `next_parity` loss from public information per row with a target (nats): the even share of the
    next round's stage's unrevealed cards (x [N, F] features: the round one-hot 0-5, the revealed round cards
    25-30; t [N, 2] the targets)."""
    x, t = np.asarray(x, np.float32), np.asarray(t, np.float32)
    nxt = x[:, 0:6].argmax(1) + 2                                  # the next round, 2..7
    left = x[:, 25:31] < 0.5
    stage = np.where((nxt <= 3)[:, None], np.arange(6)[None, :] < 3, np.arange(6)[None, :] >= 3)
    k = (left & stage).sum(1)
    even = (left & stage & (np.arange(6)[None, :] % 2 == 0)).sum(1)
    p = np.stack([even, k - even], 1) / np.maximum(k, 1)[:, None]
    have = t.sum(1) > 0
    return -(t * np.log(np.clip(p, 1e-12, 1))).sum(1)[have]


def install():
    """Declare `next_parity` for the toy in this process (idempotent)."""
    from zero import data, games, package
    if any((t['name'] if isinstance(t, dict) else t[0]) == PARITY['name'] for t in package.CORE.aux['toy']):
        return
    package.CORE.aux['toy'] = list(package.CORE.aux['toy']) + [PARITY]
    run = sys.modules.get('zero.run')
    if run is not None:
        run.apply_package(package.CORE)            # (aux_next_parity_w among the settings)
    info0 = games.info

    def info(game):
        d = info0(game)
        if game == 'toy':
            d['aux_layout'] = list(d['aux_layout']) + [(PARITY['name'], PARITY['width'])]
        return d
    games.info = info
    add0 = data.ShardWriter.add

    def add(self, s):
        if s is not None and 'aux' in s and s['aux'].shape[1] == 6:
            s = dict(s, aux=np.concatenate([s['aux'], parity_of(s['aux']).astype(s['aux'].dtype)], 1))
        return add0(self, s)
    data.ShardWriter.add = add


def main(argv=None):
    install()
    from zero import run                           # (its import applies the package: aux_next_parity_w)
    if argv is None:
        run.cli()
    else:
        run.main(argv)


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))
    main()
