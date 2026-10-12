"""Head-to-head match between two checkpoints, seats alternating: the readout of an A/B whose
yardstick is saturated (Caverna Zero's cvz2p wins 98-99.5% against the JS bot; docs/az.md 9).

  python -m <pkg>.az.h2h runs/cvz2p_cq/models/iter_0180.pt runs/cvz2p_vis/models/iter_0180.pt
      [--games 600 --sims 64 --chance-k 2 --device cuda --threads 4 --seed 0]
  (python -m zero.h2h ... in zero-core: toy checkpoints)

Both seats search with the evaluation settings of run.py (`eval_sims`, chance_k, no noise,
argmax moves); the game comes from checkpoint A (`Net.config['game']`), its spec from the game
package (`GamePackage.specs`, e.g. Caverna {"full": true}, Glass Road {"intro": false}) unless
--spec is given. Prints one JSON line: A's score (+- 1 se), its mean score margin, the Elo
difference. The models are only read.
"""
import argparse
import json
import math
import time

from . import games as G, package
from .infer import NetEvaluator, drive
from .net import load


def match(a, b, games=600, sims=64, chance_k=2, c_puct=1.0, seed=0, threads=4, groups=2, device='cpu', game=None,
          spec=None, conc=None):
    """Checkpoint `a` vs checkpoint `b` (paths), `games` 2-player games, seats alternating. -> dict."""
    ma, _ = load(a, device)
    mb, _ = load(b, device)
    game = game or ma.config.get('game', 'toy')
    if mb.config.get('game', game) != game:
        raise ValueError(f'{a} plays {game}, {b} plays {mb.config.get("game")}')
    spec = package.active().specs.get(game) if spec is None else spec
    sa = G.nn_seat('a', sims, c_puct, chance_k=chance_k)
    sb = G.nn_seat('b', sims, c_puct, chance_k=chance_k)
    seats = [[sa, sb] if g % 2 == 0 else [sb, sa] for g in range(games)]
    drv = G.Driver(game, seats, spec=spec, concurrency=conc or min(games, 1024), seed=seed, threads=threads,
                   groups=groups, max_steps=5000)
    t0 = time.time()
    drive(drv, {'a': NetEvaluator(ma, device), 'b': NetEvaluator(mb, device)})
    rs = drv.take_results()
    n = len(rs)
    w = sum(r['result'][r['gid'] % 2] for r in rs) / n
    margin = sum(r['scores'][r['gid'] % 2] - r['scores'][1 - r['gid'] % 2] for r in rs) / n
    s = min(max(w, 1e-3), 1 - 1e-3)
    return {'a': a, 'b': b, 'game': game, 'games': n, 'sims': sims, 'chance_k': chance_k, 'a_score': round(w, 4),
            'se': round(math.sqrt(w * (1 - w) / n), 4), 'a_margin': round(margin, 2),
            'elo_diff': round(400 * math.log10(s / (1 - s)), 1), 'unfinished': sum(1 for r in rs if not r['over']),
            'seconds': round(time.time() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--games', type=int, default=600)
    ap.add_argument('--sims', type=int, default=64)
    ap.add_argument('--chance-k', type=int, default=2)
    ap.add_argument('--c-puct', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--device', default='cpu', help="network device ('cuda' works; the GPU is shared)")
    ap.add_argument('--spec', default=None, help="game spec JSON (default: the game package's spec of the game)")
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    try:
        import os
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10)
    except Exception:                              # noqa: BLE001
        pass
    print(json.dumps(match(a.a, a.b, a.games, a.sims, a.chance_k, a.c_puct, a.seed, a.threads, device=a.device,
                           spec=a.spec)), flush=True)


if __name__ == '__main__':
    main()
