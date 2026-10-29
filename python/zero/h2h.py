"""Head-to-head match between two checkpoints, seats alternating: the readout of an A/B whose
yardstick is saturated (Caverna Zero's cvz2p wins 98-99.5% against the JS bot; docs/az.md 9).

  python -m <pkg>.az.h2h runs/cvz2p_cq/models/iter_0180.pt runs/cvz2p_vis/models/iter_0180.pt
      [--games 600 --sims 64 --chance-k 2 --device cuda --threads 4 --seed 0]
      [--select cq [--select-min-visits 2]] [--a-select cq] [--b-select visits] [--move-log 200]
      [--cq-c-scale 0.3] [--a-cq-c-scale 0.3 --b-cq-c-scale 1.0] [--a-cq-c-visit 50 ...]
  (python -m zero.h2h ... in zero-core: toy checkpoints)

Both seats search with the evaluation settings of run.py (`eval_sims`, chance_k, no noise, no
temperature); each side's move rule (docs/search.md 3.3) is its checkpoint's run's eval_select
(runs/<run>/config.json next to models/: its rule, guard and the rule's completed-Q transform -
eval_select_c_visit / _c_scale, unset: cq_c_visit / cq_c_scale; 'visits' without one), or --select /
--select-min-visits for both, --a-select / --b-select (and their --*-min-visits) for one side - the
same checkpoint with two rules is an A/B of the rule. The 'cq' rule's transform: --cq-c-visit /
--cq-c-scale for both sides, --a-cq-c-* / --b-cq-c-* for one (over the both-sides flag); a side
without one takes 50 / 1 when its rule is given here, else its run's - so `A A --select cq
--a-cq-c-scale 0.3 --b-cq-c-scale 1.0` is an A/B of the selection scale on one checkpoint. A
side's transform flag on a 'visits' side is an error, as are both-sides flags when no side plays
'cq'. The game comes from checkpoint A (`Net.config['game']`), its spec from the game package
(`GamePackage.specs`, e.g. Caverna {"full": true}, Glass Road {"intro": false}) unless --spec is
given. Prints one JSON line: A's score (+- 1 se), its mean score margin, the Elo difference, each
side's rule (`select_a`, `select_b`: e.g. 'cq-n4-v50-s0.3'). The models are only read.
"""
import argparse
import json
import math
import os
import time

from . import games as G, package
from .infer import NetEvaluator, drive
from .mcts import check_select, check_select_transform, forced_tag, select_settings, select_tag, select_transform
from .net import load


def run_rule(ckpt):
    """The move-rule seat settings of a checkpoint's run (its config.json's eval_select, guard
    and the rule's completed-Q transform, `mcts.select_settings`; {} = the default) for a
    checkpoint runs/<run>/models/<file>.pt. An evaluation seat has no 'cq' target, so the rule's
    transform goes in as cq_c_visit / cq_c_scale (as `games.nn_seat` writes it). The run's
    evaluation root minimum visits (eval_root_min_visits, docs/search.md 3.5) come with it."""
    d = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
    try:
        with open(os.path.join(d, 'config.json'), encoding='utf-8') as f:
            c = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(c, dict):
        return {}
    out = {}
    if c.get('eval_root_min_visits'):
        out = {'root_min_visits': int(c['eval_root_min_visits']),
               'root_min_visits_max_actions': int(c.get('root_min_visits_max_actions', 4))}
    if c.get('eval_select', 'visits') == 'visits':
        return out
    rule, mv, cv, cs = select_settings(c, 'eval_select')
    return dict({'select': rule, 'select_min_visits': mv, 'cq_c_visit': cv, 'cq_c_scale': cs}, **out)


def rule_keys(select=None, min_visits=0, c_visit=50.0, c_scale=1.0):
    """Seat settings of a move rule given on the command line (None: not given)."""
    if select is None:
        return None
    check_select(select, min_visits)
    if select == 'visits':
        return {}
    check_select_transform(select, float(c_visit), float(c_scale), what='--cq')
    return {'select': select, 'select_min_visits': int(min_visits), 'cq_c_visit': float(c_visit),
            'cq_c_scale': float(c_scale)}


def side_keys(ckpt, select=None, min_visits=0, c_visit=None, c_scale=None):
    """A side's move-rule seat settings: `select` given (rule_keys; a transform not given: 50 / 1)
    or its checkpoint's run's (run_rule), with the 'cq' rule's transform `c_visit` / `c_scale` where
    given (None: not given). A transform given for a 'visits' side is a ValueError."""
    if select is not None:
        keys = rule_keys(select, min_visits, 50.0 if c_visit is None else c_visit, 1.0 if c_scale is None else c_scale)
    else:
        keys = dict(run_rule(ckpt))
    if c_visit is None and c_scale is None:
        return keys
    if keys.get('select', 'visits') == 'visits':
        raise ValueError(f"{os.path.basename(ckpt)}: a completed-Q transform for a side whose rule is 'visits'")
    cv, cs = select_transform(keys['cq_c_visit'], keys['cq_c_scale'], c_visit, c_scale)
    check_select_transform('cq', cv, cs, what='--cq')
    return dict(keys, cq_c_visit=cv, cq_c_scale=cs)


def _tag(seat):
    cv, cs = select_transform(seat.get('cq_c_visit', 50.0), seat.get('cq_c_scale', 1.0), seat.get('select_c_visit'),
                              seat.get('select_c_scale'))
    return select_tag(seat.get('select', 'visits'), seat.get('select_min_visits', 0), cv, cs) + forced_tag(
        seat.get('root_min_visits', 0), seat.get('root_min_visits_max_actions', 4))


def match(a, b, games=600, sims=64, chance_k=2, c_puct=1.0, seed=0, threads=4, groups=2, device='cpu', game=None,
          spec=None, conc=None, seat_a=None, seat_b=None, max_steps=5000, move_log=0, rows=False):
    """Checkpoint `a` vs checkpoint `b` (paths), `games` 2-player games, seats alternating. -> dict.
    `seat_a` / `seat_b`: each side's move-rule seat settings ({'select': 'cq', 'select_min_visits':
    2, ...}; {} = the default rule; None = its checkpoint's run's eval_select, `run_rule`).
    `move_log`: the driver's move log (the first N moves of each game in its row), `rows`: the
    games' result rows under 'rows'."""
    # (the encoding guard, docs/az.md 10.4: A against the game played, B against its own game,
    # which must be the same)
    ma, _ = load(a, device, game=game)
    game = game or ma.config.get('game', 'toy')
    mb, _ = load(b, device)
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
    ap.add_argument('--cq-c-visit', type=float, default=None,
                    help="the 'cq' rule's completed-Q transform, both sides (default: 50 / 1 with --select, else "
                         "each checkpoint's run's)")
    ap.add_argument('--cq-c-scale', type=float, default=None)
    for side in ('a', 'b'):
        ap.add_argument(f'--{side}-cq-c-visit', type=float, default=None,
                        help=f"side {side.upper()}'s 'cq' transform (over --cq-c-visit / --cq-c-scale)")
        ap.add_argument(f'--{side}-cq-c-scale', type=float, default=None)
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
    for side, ckpt in (('a', a.a), ('b', a.b)):
        rule = getattr(a, f'{side}_select') or a.select
        mv = getattr(a, f'{side}_select_min_visits')
        mv = a.select_min_visits if mv is None else mv
        own = getattr(a, f'{side}_cq_c_visit'), getattr(a, f'{side}_cq_c_scale')
        both = a.cq_c_visit, a.cq_c_scale
        base = side_keys(ckpt, rule, mv)
        if base.get('select', 'visits') == 'visits':
            if own != (None, None):
                ap.error(f"--{side}-cq-c-*: side {side.upper()} plays 'visits' (no completed-Q transform)")
            sides.append(base)
            continue
        cv, cs = (o if o is not None else b for o, b in zip(own, both))
        sides.append(side_keys(ckpt, rule, mv, cv, cs))
    if (a.cq_c_visit, a.cq_c_scale) != (None, None) and not any(s.get('select') == 'cq' for s in sides):
        ap.error("--cq-c-visit / --cq-c-scale: no side plays 'cq'")
    print(json.dumps(match(a.a, a.b, a.games, a.sims, a.chance_k, a.c_puct, a.seed, a.threads, device=a.device,
                           spec=a.spec, seat_a=sides[0], seat_b=sides[1], move_log=a.move_log)), flush=True)


if __name__ == '__main__':
    main()
