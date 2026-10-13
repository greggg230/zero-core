"""Head-to-head match between two network seats, seats alternating: two checkpoints (the readout
of an A/B whose yardstick is saturated: Caverna Zero's cvz2p wins 98-99.5% against the JS bot;
docs/az.md 9), one checkpoint under two search settings (Glass Road's oracle and opponent-model
matches, docs/search.md 10), or a checkpoint against an external player ('ext:jsbot').

  python -m <pkg>.az.h2h runs/cvz2p_cq/models/iter_0180.pt runs/cvz2p_vis/models/iter_0180.pt
  python -m <pkg>.az.h2h runs/grz2p_rb/models/iter_0143.pt --a-opts '{"psample": true}'
  python -m <pkg>.az.h2h runs/grz2p_rb/models/iter_0143.pt ext:jsbot --a-opts '{"belief_mix": 0.5}'
      [--games 600 --sims 64 --chance-k 2 --device cuda --threads 4 --seed 0 --label X --out F]
  (python -m zero.h2h ... in zero-core: toy checkpoints)

Both seats search with the evaluation settings of run.py (`eval_sims`, chance_k, no noise,
argmax moves) plus their `--a-opts` / `--b-opts` (any network-seat setting: `psample`,
`belief_mix`, `oracle_hands`, `class_sims`, `mask_features`, `redet`, `c_puct`, ...). B defaults
to A's checkpoint (one evaluator is shared). The game comes from checkpoint A
(`Net.config['game']`), its spec from the game package (`GamePackage.specs`, e.g. Caverna
{"full": true}, Glass Road {"intro": false}) unless --spec is given. Prints one JSON line: A's
score (+- 1 se), its mean score margin, the Elo difference, A's score by seat (in Glass Road
seat 0 always chooses its hand first; the start player is random), the seat-adjusted Elo edge
(`seat_adjusted_elo`: with a strong seat advantage the alternating score is compressed towards
1/2) and the driver's opponent-model counters. The models are only read.
"""
import argparse
import json
import math
import time

from . import games as G, package
from .infer import NetEvaluator, drive
from .net import load


def seat_adjusted_elo(s0, n0, s1, n1):
    """Bradley-Terry with a seat term: P(A wins | A in seat 0) = sigmoid(d + h), in seat 1
    sigmoid(d - h) -> A's Elo edge d, its se (delta method) and the seat-0 edge h. A strong seat
    advantage compresses the plain alternating score towards 1/2; d is not compressed."""
    def lg(p, n):
        p = min(max(p, 0.5 / n), 1 - 0.5 / n)
        return math.log(p / (1 - p)), 1.0 / (n * p * (1 - p))
    (l0, v0), (l1, v1) = lg(s0, n0), lg(s1, n1)
    k = 400 / math.log(10)
    return k * (l0 + l1) / 2, k * math.sqrt(v0 + v1) / 2, k * (l0 - l1) / 2


def _score(rs, seat_of_a):
    n = len(rs)
    if not n:
        return None, None
    w = sum(r['result'][seat_of_a(r)] for r in rs) / n
    return w, math.sqrt(max(w * (1 - w), 1e-9) / n)


STAT_KEYS = ('leaves', 'psampled', 'belief_rows', 'belief_jobs', 'belief_targets', 'belief_searches', 'belief_missing',
             'belief_worlds')


def match(a, b=None, games=600, sims=64, chance_k=2, c_puct=1.0, seed=0, threads=4, groups=2, device='cpu', game=None,
          spec=None, conc=None, a_opts=None, b_opts=None):
    """`a` vs `b` (checkpoint paths; b None = a; 'ext:<name>' = an external player such as
    'ext:jsbot'), `games` 2-player games, seats alternating (A is seat 0 in even games).
    a_opts / b_opts: extra network-seat settings. -> dict."""
    ma, _ = load(a, device)
    ea = NetEvaluator(ma, device)
    externals = {}
    if b is not None and b.startswith('ext:'):
        mb, eb = ma, None
        externals[b[4:]] = G.external(b[4:], seed)
    elif b is None or b == a:
        b, eb, mb = a, ea, ma
    else:
        mb, _ = load(b, device)
        eb = NetEvaluator(mb, device)
    game = game or ma.config.get('game', 'toy')
    if mb.config.get('game', game) != game:
        raise ValueError(f'{a} plays {game}, {b} plays {mb.config.get("game")}')
    spec = package.active().specs.get(game) if spec is None else spec
    sa = G.nn_seat('a', sims, c_puct, chance_k=chance_k)
    sb = G.seat_dict(b) if externals else G.nn_seat('b', sims, c_puct, chance_k=chance_k)
    sa.update(a_opts or {})
    if externals:
        if b_opts:
            raise ValueError('b_opts: B is an external player')
    else:
        sb.update(b_opts or {})
    seats = [[sa, sb] if g % 2 == 0 else [sb, sa] for g in range(games)]
    if externals and groups == 2:
        groups = package.active().defaults.get('ext_groups', 2)
    drv = G.Driver(game, seats, spec=spec, concurrency=conc or min(games, 1024), seed=seed, threads=threads,
                   groups=groups, max_steps=5000)
    t0 = time.time()
    evs = {'a': ea} if externals else {'a': ea, 'b': eb}
    drive(drv, evs, externals)
    rs = drv.take_results()
    n = len(rs)
    seat = lambda r: r['gid'] % 2                  # noqa: E731  (A's seat)
    w, se = _score(rs, seat)
    margin = sum(r['scores'][seat(r)] - r['scores'][1 - seat(r)] for r in rs) / n
    s = min(max(w, 1e-3), 1 - 1e-3)
    by_seat = {}
    for k in (0, 1):
        sw, sse = _score([r for r in rs if seat(r) == k], seat)
        by_seat[k] = {'games': sum(1 for r in rs if seat(r) == k), 'a_score': round(sw, 4), 'se': round(sse, 4)}
    d, dse, h = seat_adjusted_elo(by_seat[0]['a_score'], by_seat[0]['games'], by_seat[1]['a_score'],
                                  by_seat[1]['games'])
    st = drv.stats()
    return {'a': a, 'b': b, 'a_opts': a_opts or {}, 'b_opts': b_opts or {}, 'game': game, 'spec': spec, 'games': n,
            'sims': sims, 'chance_k': chance_k, 'seed': seed, 'a_score': round(w, 4), 'se': round(se, 4),
            'a_margin': round(margin, 2), 'elo_diff': round(400 * math.log10(s / (1 - s)), 1),
            'a_by_seat': by_seat, 'elo_seat_adj': round(d, 1), 'elo_seat_adj_se': round(dse, 1),
            'seat0_edge_elo': round(h, 1), 'unfinished': sum(1 for r in rs if not r['over']),
            'stats': {k: st[k] for k in STAT_KEYS if k in st}, 'seconds': round(time.time() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    ap.add_argument('a')
    ap.add_argument('b', nargs='?', default=None, help="checkpoint B (default: A's) or an external player ('ext:jsbot')")
    ap.add_argument('--games', type=int, default=600)
    ap.add_argument('--sims', type=int, default=64)
    ap.add_argument('--chance-k', type=int, default=2)
    ap.add_argument('--c-puct', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--groups', type=int, default=2, help='driver groups (an external B: the package\'s ext_groups)')
    ap.add_argument('--conc', type=int, default=None)
    ap.add_argument('--device', default='cpu', help="network device ('cuda' works; the GPU is shared)")
    ap.add_argument('--spec', default=None, help="game spec JSON (default: the game package's spec of the game)")
    ap.add_argument('--a-opts', default=None, help='JSON: extra seat settings of A, e.g. {"psample": true}')
    ap.add_argument('--b-opts', default=None, help='JSON: extra seat settings of B')
    ap.add_argument('--label', default=None, help='a name stored in the output line')
    ap.add_argument('--out', default=None, help='also append the JSON line to this file')
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    try:
        import os
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10)
    except Exception:                              # noqa: BLE001
        pass
    r = match(a.a, a.b, a.games, a.sims, a.chance_k, a.c_puct, a.seed, a.threads, a.groups, device=a.device,
              spec=a.spec, conc=a.conc, a_opts=json.loads(a.a_opts) if a.a_opts else None,
              b_opts=json.loads(a.b_opts) if a.b_opts else None)
    if a.label:
        r = {'label': a.label, **r}
    line = json.dumps(r)
    print(line, flush=True)
    if a.out:
        with open(a.out, 'a', encoding='utf-8') as f:
            f.write(line + '\n')


if __name__ == '__main__':
    main()
