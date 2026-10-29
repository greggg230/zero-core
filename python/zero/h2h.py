"""Head-to-head match between two checkpoints, seats alternating: the readout of an A/B whose
yardstick is saturated (Caverna Zero's cvz2p wins 98-99.5% against the JS bot; docs/az.md 9).

  python -m <pkg>.az.h2h runs/cvz2p_cq/models/iter_0180.pt runs/cvz2p_vis/models/iter_0180.pt
      [--games 600 --sims 64 --chance-k 2 --device cuda --threads 4 --seed 0]
      [--select cq [--select-min-visits 2]] [--a-select cq] [--b-select visits] [--move-log 200]
  (python -m zero.h2h ... in zero-core: toy checkpoints)

Both seats search with the evaluation settings of run.py (`eval_sims`, chance_k, no noise, no
temperature); each side's move rule (docs/search.md 3.3) is its checkpoint's run's eval_select
(runs/<run>/config.json next to models/: its rule, guard and completed-Q transform; 'visits'
without one), or --select / --select-min-visits for both, --a-select / --b-select (and their
--*-min-visits) for one side - the same checkpoint with two rules is an A/B of the rule. The game
comes from checkpoint A (`Net.config['game']`), its spec from the game package
(`GamePackage.specs`, e.g. Caverna {"full": true}, Glass Road {"intro": false}) unless --spec is
given. Prints one JSON line: A's score (+- 1 se), its mean score margin, the Elo difference, each
side's rule (`select_a`, `select_b`). The models are only read.
"""
import argparse
import json
import math
import os
import time

from . import games as G, package
from .infer import NetEvaluator, drive
from .mcts import check_select, select_tag
from .net import load


def run_rule(ckpt):
    """The move-rule seat settings of a checkpoint's run (its config.json's eval_select, guard
    and completed-Q transform; {} = the default) for a checkpoint runs/<run>/models/<file>.pt."""
    d = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
    try:
        with open(os.path.join(d, 'config.json'), encoding='utf-8') as f:
            c = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(c, dict) or c.get('eval_select', 'visits') == 'visits':
        return {}
    return {'select': c['eval_select'], 'select_min_visits': int(c.get('eval_select_min_visits', 0)),
            'cq_c_visit': float(c.get('cq_c_visit', 50.0)), 'cq_c_scale': float(c.get('cq_c_scale', 1.0))}


def rule_keys(select=None, min_visits=0, c_visit=50.0, c_scale=1.0):
    """Seat settings of a move rule given on the command line (None: not given)."""
    if select is None:
        return None
    check_select(select, min_visits)
    if select == 'visits':
        return {}
    return {'select': select, 'select_min_visits': int(min_visits), 'cq_c_visit': float(c_visit),
            'cq_c_scale': float(c_scale)}


def _tag(seat):
    return select_tag(seat.get('select', 'visits'), seat.get('select_min_visits', 0), seat.get('cq_c_visit', 50.0),
                      seat.get('cq_c_scale', 1.0))


def match(a, b, games=600, sims=64, chance_k=2, c_puct=1.0, seed=0, threads=4, groups=2, device='cpu', game=None,
          spec=None, conc=None, seat_a=None, seat_b=None, max_steps=5000, move_log=0, rows=False):
    """Checkpoint `a` vs checkpoint `b` (paths), `games` 2-player games, seats alternating. -> dict.
    `seat_a` / `seat_b`: each side's move-rule seat settings ({'select': 'cq', 'select_min_visits':
    2, ...}; {} = the default rule; None = its checkpoint's run's eval_select, `run_rule`).
    `move_log`: the driver's move log (the first N moves of each game in its row), `rows`: the
    games' result rows under 'rows'."""
    ma, _ = load(a, device)
    mb, _ = load(b, device)
    game = game or ma.config.get('game', 'toy')
    if mb.config.get('game', game) != game:
        raise ValueError(f'{a} plays {game}, {b} plays {mb.config.get("game")}')
    spec = package.active().specs.get(game) if spec is None else spec
    sa = G.nn_seat('a', sims, c_puct, chance_k=chance_k)
    sb = G.nn_seat('b', sims, c_puct, chance_k=chance_k)
    sa.update(run_rule(a) if seat_a is None else seat_a)
    sb.update(run_rule(b) if seat_b is None else seat_b)
    seats = [[sa, sb] if g % 2 == 0 else [sb, sa] for g in range(games)]
    kw = {'move_log': move_log} if move_log else {}
    drv = G.Driver(game, seats, spec=spec, concurrency=conc or min(games, 1024), seed=seed, threads=threads,
                   groups=groups, max_steps=max_steps, **kw)
    t0 = time.time()
    drive(drv, {'a': NetEvaluator(ma, device), 'b': NetEvaluator(mb, device)})
    rs = drv.take_results()
    n = len(rs)
    w = sum(r['result'][r['gid'] % 2] for r in rs) / n
    margin = sum(r['scores'][r['gid'] % 2] - r['scores'][1 - r['gid'] % 2] for r in rs) / n
    s = min(max(w, 1e-3), 1 - 1e-3)
    out = {'a': a, 'b': b, 'game': game, 'games': n, 'sims': sims, 'chance_k': chance_k, 'a_score': round(w, 4),
           'se': round(math.sqrt(w * (1 - w) / n), 4), 'a_margin': round(margin, 2),
           'elo_diff': round(400 * math.log10(s / (1 - s)), 1), 'unfinished': sum(1 for r in rs if not r['over']),
           'seconds': round(time.time() - t0, 1), 'select_a': _tag(sa), 'select_b': _tag(sb)}
    if rows:
        out['rows'] = sorted(rs, key=lambda r: r['gid'])
    return out


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
    ap.add_argument('--select', default=None, choices=['visits', 'cq'],
                    help="both sides' move rule (default: each checkpoint's run's eval_select)")
    ap.add_argument('--select-min-visits', type=int, default=0)
    for side in ('a', 'b'):
        ap.add_argument(f'--{side}-select', default=None, choices=['visits', 'cq'], help=f"side {side.upper()}'s move rule")
        ap.add_argument(f'--{side}-select-min-visits', type=int, default=None)
    ap.add_argument('--cq-c-visit', type=float, default=50.0, help="the 'cq' rule's completed-Q transform")
    ap.add_argument('--cq-c-scale', type=float, default=1.0)
    ap.add_argument('--move-log', type=int, default=0, help='log the first N moves of every game (with --rows)')
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10)
    except Exception:                              # noqa: BLE001
        pass
    sides = []
    for side in ('a', 'b'):
        rule = getattr(a, f'{side}_select') or a.select
        mv = getattr(a, f'{side}_select_min_visits')
        sides.append(rule_keys(rule, a.select_min_visits if mv is None else mv, a.cq_c_visit, a.cq_c_scale))
    print(json.dumps(match(a.a, a.b, a.games, a.sims, a.chance_k, a.c_puct, a.seed, a.threads, device=a.device,
                           spec=a.spec, seat_a=sides[0], seat_b=sides[1], move_log=a.move_log)), flush=True)


if __name__ == '__main__':
    main()
