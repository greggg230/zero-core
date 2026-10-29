"""Digests of self-play records and match results of ONE checkout, for old-vs-new comparisons
(tools/check_checkpoint_compat.py runs this in a subprocess per checkout; docs/pinning.md).

  python tools/selfplay_digest.py --root <checkout> --package grz [--game glassroad --spec '{"intro": false}']
      [--net uniform | --net random | --model runs/x/models/iter_0040.pt] [--games 24 --sims 32 --sims-fast 8
      --p-full 0.25 --chance-k 2 --temp-moves 30 --threads 2 --seed 2026 --targets visits,cq --match-games 16]

Works on a game checkout from before zero-core (its `<pkg>.az.games` / `.infer` / `.net` modules)
and after (the same names are shims of the core), and on zero-core itself (`--root
<core>/python --package zero`). Imports only from `--root`; prints one JSON object:

* per policy target ('visits', 'cq'): the number of recorded samples and the sha256 of the
  self-play records (every sample array, rows in a canonical order) and of the game results
  (gid, steps, scores, decisions, rounds) - self-play of `--games` games with the evaluator;
* `match`: the network (or the uniform evaluator) vs the built-in 'random' bot, seats
  alternating, evaluation settings (no noise): results digest and the network's score.

Evaluators: 'uniform' (exact arithmetic: uniform priors, value 1/n), 'random' (a randomly
initialised network, torch seed 0, 64 x 2), or a checkpoint (`--model`). The torch evaluators
run on the CPU with 1 thread.
"""
import argparse
import hashlib
import importlib
import json
import os
import sys


def _mod(pkg, name):
    for m in (f'{pkg}.az.{name}', f'{pkg}.{name}'):
        try:
            return importlib.import_module(m)
        except ModuleNotFoundError as e:
            if e.name not in (m, f'{pkg}.az'):
                raise
    raise ModuleNotFoundError(f'{pkg}: no module {name}')


def records_digest(parts, np):
    """sha256 over the sample rows (each row's arrays), rows sorted by their own hash (the order
    games finish in may depend on thread timing; the rows may not)."""
    rows = []
    for s in parts:
        if s is None or not len(s['player']):
            continue
        off = s['opt_off']
        for b in range(len(s['player'])):
            h = hashlib.sha256()
            for k in ('x', 'player', 'n_players', 'z', 'q', 'sc', 'net'):
                h.update(np.ascontiguousarray(s[k][b]).tobytes())
            for k in ('opt_action', 'pi'):
                h.update(np.ascontiguousarray(s[k][off[b]:off[b + 1]]).tobytes())
            rows.append(h.hexdigest())
    rows.sort()
    return len(rows), hashlib.sha256(''.join(rows).encode()).hexdigest()


def results_digest(results):
    key = sorted((int(r['gid']), int(r['steps']), [round(float(x), 6) for x in r['scores']], int(r['decisions']),
                  int(r['rounds']), bool(r['over'])) for r in results)
    return hashlib.sha256(json.dumps(key).encode()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', required=True, help='checkout root (zero-core: <core>/python)')
    ap.add_argument('--package', required=True)
    ap.add_argument('--game', default=None, help='default: the checkpoint\'s game, else toy')
    ap.add_argument('--spec', default=None)
    ap.add_argument('--net', default='uniform', choices=('uniform', 'random'))
    ap.add_argument('--model', default=None)
    ap.add_argument('--games', type=int, default=24)
    ap.add_argument('--sims', type=int, default=32)
    ap.add_argument('--sims-fast', type=int, default=8)
    ap.add_argument('--p-full', type=float, default=0.25)
    ap.add_argument('--chance-k', type=int, default=2)
    ap.add_argument('--temp-moves', type=int, default=30)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--groups', type=int, default=2)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--targets', default='visits,cq')
    ap.add_argument('--match-games', type=int, default=16)
    ap.add_argument('--max-steps', type=int, default=5000)
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.abspath(a.root))
    importlib.import_module(a.package)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    G = _mod(a.package, 'games')
    infer = _mod(a.package, 'infer')
    net = _mod(a.package, 'net')
    out = {'package': a.package, 'root': os.path.abspath(a.root), 'engine': os.path.abspath(G.engine().__file__)}
    if a.model:
        # the encoding guard's opt-out (docs/az.md 10.4): an old-vs-new comparison may play a
        # checkpoint on a build of another encoding on purpose (checkouts from before the guard
        # have no such argument)
        import inspect
        opt = {'allow_mismatch': True}
        model, extra = net.load(a.model, 'cpu', **(opt if 'allow_mismatch' in inspect.signature(net.load).parameters
                                                   else {}))
        game = a.game or model.config.get('game', 'toy')
        ev = infer.NetEvaluator(model, 'cpu', **(opt if 'allow_mismatch' in
                                                 inspect.signature(infer.NetEvaluator).parameters else {}))
        out['evaluator'] = {'model': os.path.abspath(a.model), 'iter': extra.get('iter'), 'config': model.config}
    else:
        game = a.game or 'toy'
        info = G.info(game)
        if a.net == 'random':
            torch.manual_seed(0)
            ev = infer.NetEvaluator(net.Net(info['num_features'], info['num_actions'], 64, 2, game=game).eval(), 'cpu')
        else:
            ev = infer.UniformEvaluator()
        out['evaluator'] = a.net
    spec = a.spec if a.spec is not None else ''
    out.update(game=game, spec=spec, games=a.games, sims=a.sims, sims_fast=a.sims_fast, p_full=a.p_full,
               chance_k=a.chance_k, seed=a.seed, targets={})
    for target in [t for t in a.targets.split(',') if t]:
        kw = dict(selfplay=True, temp_moves=a.temp_moves, sims_fast=a.sims_fast, p_full=a.p_full, chance_k=a.chance_k)
        if target != 'visits':
            kw.update(policy_target=target, cq_c_visit=50.0, cq_c_scale=1.0)
        seat = G.nn_seat('m', a.sims, **kw)
        drv = G.Driver(game, [[seat, seat]] * a.games, spec=spec, concurrency=a.games, seed=a.seed, record=True,
                       threads=a.threads, groups=a.groups, max_steps=a.max_steps)
        parts = []
        infer.drive(drv, {'m': ev}, on_batch=lambda lb: parts.append(drv.take_samples()))
        parts.append(drv.take_samples())
        n, d = records_digest(parts, np)
        res = drv.take_results()
        out['targets'][target] = {'samples': n, 'records': d, 'results': results_digest(res), 'games': len(res)}
    if a.match_games:
        nn = G.nn_seat('m', a.sims, chance_k=a.chance_k)
        bot = {'bot': 'random'}
        seats = [[nn, bot] if g % 2 == 0 else [bot, nn] for g in range(a.match_games)]
        drv = G.Driver(game, seats, spec=spec, concurrency=a.match_games, seed=a.seed + 1, threads=a.threads,
                       groups=a.groups, max_steps=a.max_steps)
        infer.drive(drv, {'m': ev})
        res = drv.take_results()
        out['match'] = {'opp': 'random', 'games': len(res), 'results': results_digest(res),
                        'score': sum(float(r['result'][r['gid'] % 2]) for r in res)}
    print(json.dumps(out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
