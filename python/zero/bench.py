"""Throughput measurements (docs/az.md 7).

  python -m <pkg>.az.bench search   [--threads 4 --games 400 --sims 64]   # Rust driver, constant evaluator
  python -m <pkg>.az.bench infer    [--device cpu|cuda --hidden 128 --blocks 3]
  python -m <pkg>.az.bench selfplay [--model PATH | --hidden H --blocks B] [--game caverna --spec {...}] [--threads 4 --games 400]
(`python -m zero.bench ...` in zero-core: the toy game.)
"""
import argparse
import json
import time

import numpy as np
import torch

from . import games as G
from .infer import NetEvaluator, UniformEvaluator, drive
from .net import Net, load


def bench_search(game='toy', threads=4, games=400, sims=64, conc=256, groups=2, spec=None):
    seat = G.nn_seat('m', sims, selfplay=True, sims_fast=sims // 4, p_full=0.25)
    drv = G.Driver(game, [[seat, seat]] * games, spec=spec, concurrency=conc, threads=threads, groups=groups, record=True)
    ev = UniformEvaluator()
    t0 = time.time()
    drive(drv, {'m': ev}, on_batch=lambda lb: drv.take_samples())
    secs = time.time() - t0
    st = drv.stats()
    return {'what': 'search (uniform evaluator)', 'threads': threads, 'games': games, 'seconds': round(secs, 2),
            'sims_per_s': round(st['sims'] / secs), 'leaves_per_s': round(st['leaves'] / secs),
            'search_busy_s': round(st['busy_s'], 2), 'us_per_sim_thread': round(st['busy_s'] / max(1, st['sims']) * 1e6, 2),
            'games_per_s': round(games / secs, 1)}


def bench_infer(game='toy', device='cpu', hidden=128, blocks=3, batches=(64, 256, 1024), seconds=1.0):
    info = G.info(game)
    m = Net(info['num_features'], info['num_actions'], hidden, blocks).eval()
    ev = NetEvaluator(m, device)
    out = {'what': 'inference', 'device': device, 'hidden': hidden, 'blocks': blocks, 'params': m.n_params(),
           'torch_threads': torch.get_num_threads(), 'leaves_per_s': {}}
    rng = np.random.default_rng(0)
    for B in batches:
        x = rng.normal(size=(B, info['num_features'])).astype(np.float32)
        n = np.full(B, 2, np.int32)
        k = 6
        off = np.arange(B + 1, dtype=np.int32) * k
        act = np.tile(np.arange(k, dtype=np.int32), B)
        for _ in range(3):
            ev.evaluate(x, n, off, act)
        t0 = time.time()
        c = 0
        while time.time() - t0 < seconds:
            ev.evaluate(x, n, off, act)
            c += 1
        if device == 'cuda':
            torch.cuda.synchronize()
        out['leaves_per_s'][B] = round(B * c / (time.time() - t0))
    return out


def bench_selfplay(model, game='toy', threads=4, games=400, sims=64, conc=256, spec=None, hidden=128, blocks=3, device='cpu'):
    if model:
        m, _ = load(model)
    else:                                          # a fresh (random) network of the given size
        info = G.info(game)
        m = Net(info['num_features'], info['num_actions'], hidden, blocks).eval()
    ev = NetEvaluator(m, device)
    seat = G.nn_seat('m', sims, selfplay=True, sims_fast=sims // 4, p_full=0.25)
    drv = G.Driver(game, [[seat, seat]] * games, spec=spec, concurrency=conc, threads=threads, record=True)
    t0 = time.time()
    drive(drv, {'m': ev}, on_batch=lambda lb: drv.take_samples())
    secs = time.time() - t0
    st = drv.stats()
    return {'what': 'self-play (network)', 'game': game, 'device': device, 'params': m.n_params(), 'threads': threads,
            'games': games, 'sims': sims, 'seconds': round(secs, 2), 'sims_per_s': round(st['sims'] / secs),
            'leaves_per_s': round(st['leaves'] / secs), 'games_per_h': round(games / secs * 3600),
            'infer_share': round(ev.seconds / secs, 3), 'avg_batch': round(ev.leaves / max(1, ev.calls), 1),
            'caller_wait_s': round(st['wait_s'], 2)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['search', 'infer', 'selfplay'])
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--games', type=int, default=400)
    ap.add_argument('--sims', type=int, default=64)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--blocks', type=int, default=3)
    ap.add_argument('--model', default=None)
    ap.add_argument('--torch-threads', type=int, default=1)
    ap.add_argument('--game', default='toy')
    ap.add_argument('--spec', default=None)
    ap.add_argument('--conc', type=int, default=256)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.torch_threads)
    if a.what == 'search':
        r = bench_search(game=a.game, threads=a.threads, games=a.games, sims=a.sims, conc=a.conc, spec=a.spec)
    elif a.what == 'infer':
        r = bench_infer(game=a.game, device=a.device, hidden=a.hidden, blocks=a.blocks)
    else:
        r = bench_selfplay(a.model, game=a.game, threads=a.threads, games=a.games, sims=a.sims, conc=a.conc, spec=a.spec,
                           hidden=a.hidden, blocks=a.blocks, device=a.device)
    print(json.dumps(r))


if __name__ == '__main__':
    main()
