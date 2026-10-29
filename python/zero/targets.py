"""Policy targets: visit counts vs completed Q (docs/search.md 3.1, docs/az.md 9).

  python -m <pkg>.az.targets sharpness --model runs/cvz2p/models/iter_0167.pt [--games 100
      --sims 128 --sims-fast 32 --c-scales 1.0,0.3,0.1 --threads 3 --device cuda]
  python -m <pkg>.az.targets quality --model runs/cvz2p/models/iter_0167.pt [--positions 300
      --sims 128 --reps 2 --ref-sims 2048 --device cuda]

Defaults: the game package's game and spec (`GamePackage.game` / `specs`: Caverna 2p
`{"full": true}`, Glass Road 2p `{"intro": false}`; the core: the toy) at the 2p runs'
self-play settings (128 / 32 sims, p_full 0.25, chance_k 2, temp_moves 30); `--game toy` for
the toy.

`sharpness` plays the same self-play games once per arm ('visits', then 'cq' with every c_scale): the
search does not depend on the policy target, so with the same seed every arm plays the same
games and records the same positions (checked); only the targets `pi` differ. Reports per arm
the normalised entropy H(pi) / ln(#options), the max share, the share of positions whose
target argmax differs from the visit target's, KL(pi || P0) to the network prior (P0 = the
network on the recorded features: the same features the root was expanded with), by decision
kind: all, two options, and per game what the package registers (`GamePackage.decision_kinds`;
Caverna: worker placements = decisions offering a SPACE action, other decisions with 3+
options; Glass Road: card choices, other 3+; otherwise 3+ options).

`quality` scores the targets against a much larger search: positions from games played by
sampling the network prior; per position `reps` independent searches of `sims` simulations
(root noise as in self-play) give each target (prior P0, visits, completed Q per c_scale,
greedy = one-hot argmax of the completed Q), and one independent reference search of
`ref_sims` simulations (no noise) gives Q_ref (completed with its v_mix). Per target: regret =
max Q_ref - E_pi[Q_ref] (P(win) the target gives up against the reference's best option), its
mass on that best option, and how often its argmax is it. The model is only read.
"""
import argparse
import json
import time

import numpy as np
import torch

from . import games as G
from . import mcts, package
from .infer import NetEvaluator, drive
from .net import load
from .protocol import concat_samples


def selfplay_samples(model, game, spec, games, seat, seed=0, conc=None, threads=3, device='cpu'):
    """Self-play `games` games with network `model` (a path) and seat dict `seat`. ->
    (results sorted by gid, concatenated samples)."""
    m, _ = load(model, device, game=game)
    ev = NetEvaluator(m, device)
    drv = G.Driver(game, [[seat, seat]] * games, spec=spec, concurrency=conc or games, seed=seed, record=True,
                   threads=threads, groups=2, max_steps=5000)
    parts = []
    drive(drv, {'m': ev}, on_batch=lambda lb: parts.append(drv.take_samples()))
    parts.append(drv.take_samples())
    return sorted(drv.take_results(), key=lambda r: r['gid']), concat_samples(parts)


def segments(s):
    off = np.asarray(s['opt_off'], np.int64)
    n = np.diff(off)
    return off, n, np.repeat(np.arange(len(n)), n)


def seg_argmax(v, off, n):
    """First maximum of every segment."""
    out = np.zeros(len(n), np.int64)
    for k in range(len(n)):
        out[k] = int(np.argmax(v[off[k]:off[k + 1]]))
    return out


def per_position(s, pi=None):
    """-> dict of per-position arrays: options, normalised entropy, max share, argmax of `pi`
    (default s['pi'])."""
    off, n, seg = segments(s)
    p = np.asarray(s['pi'] if pi is None else pi, np.float64)
    plogp = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    h = -np.bincount(seg, plogp, minlength=len(n))
    mx = np.zeros(len(n))
    np.maximum.at(mx, seg, p)
    return {'n': n, 'h_norm': h / np.log(np.maximum(n, 2)), 'max': mx, 'argmax': seg_argmax(p, off, n)}


def network_prior(model, s, device='cpu'):
    """P0 over each position's options: the network on the recorded features."""
    m, _ = load(model, device)
    ev = NetEvaluator(m, device)
    pri = []
    B = 2048
    off = np.asarray(s['opt_off'], np.int64)
    for a in range(0, len(s['player']), B):
        b = min(len(s['player']), a + B)
        o = off[a:b + 1]
        p, _, _ = ev.evaluate(np.ascontiguousarray(s['x'][a:b], np.float32), np.asarray(s['n_players'][a:b], np.int32),
                              (o - o[0]).astype(np.int32), np.asarray(s['opt_action'][o[0]:o[-1]], np.int32))
        pri.append(p)
    return np.concatenate(pri).astype(np.float64)


def option_mask(s, lo, hi, every=False):
    """Per recorded position: does any (every=True: every) option have an action id in
    [lo, hi)? (for a game package's decision kinds)"""
    off, n, _ = segments(s)
    a = np.asarray(s['opt_action'])
    f = np.all if every else np.any
    return np.array([bool(f((a[off[k]:off[k + 1]] >= lo) & (a[off[k]:off[k + 1]] < hi))) for k in range(len(n))],
                    bool)


def decision_kinds(game, s):
    """Masks over the recorded positions of a sample dict: 'all', 'two options' and per game
    what the game package registers for `game` (`GamePackage.decision_kinds[game](s)` -> {kind:
    mask}); otherwise '3+ options'."""
    off, n, _ = segments(s)
    kinds = {'all': np.ones(len(n), bool), 'two options': n == 2}
    fn = package.active().decision_kinds.get(game)
    if fn is None:
        kinds['3+ options'] = n >= 3
    else:
        kinds.update(package.resolve(fn)(s))
    return kinds


def kl(p, q, seg, n):
    p = np.asarray(p, np.float64)
    t = np.where(p > 0, p * (np.log(np.where(p > 0, p, 1.0)) - np.log(np.maximum(q, 1e-30))), 0.0)
    return np.bincount(seg, t, minlength=n)


def _game_spec(game, spec):
    """None -> the game package's defaults (its game; that game's spec)."""
    pkg = package.active()
    game = game or pkg.game
    return game, (pkg.specs.get(game) if spec is None else spec)


def compare(model, game=None, spec=None, games=100, sims=128, sims_fast=32, p_full=0.25,
            c_puct=1.0, chance_k=2, temp_moves=30, c_visit=50.0, c_scales=(1.0, 0.3, 0.1), seed=0, threads=3,
            verbose=True, device='cpu'):
    """-> report dict (see the module docs). game / spec None: the game package's defaults."""
    game, spec = _game_spec(game, spec)
    base = dict(sims=sims, c_puct=c_puct, selfplay=True, alpha=0.3, eps=0.25, temp_moves=temp_moves,
                sims_fast=sims_fast, p_full=p_full, chance_k=chance_k, margin_w=0.0, redet=True)
    arms = [('visits', G.nn_seat('m', **base))]
    arms += [(f'cq c_scale {cs:g}', G.nn_seat('m', **base, policy_target='cq', cq_c_visit=c_visit, cq_c_scale=cs))
             for cs in c_scales]
    out = {'model': model, 'game': game, 'games': games, 'sims': sims, 'sims_fast': sims_fast, 'p_full': p_full,
           'chance_k': chance_k, 'c_visit': c_visit, 'device': device, 'arms': {}}
    ref = None
    for name, seat in arms:
        t0 = time.time()
        res, s = selfplay_samples(model, game, spec, games, seat, seed=seed, threads=threads, device=device)
        if ref is None:
            ref = (res, s)
            prior = network_prior(model, s, device)
            off, n, seg = segments(s)
            kinds = decision_kinds(game, s)
            pp = per_position(s, prior)
            out['positions'] = int(len(n))
            out['options'] = round(float(n.mean()), 2)
            out['prior'] = {k: {'positions': int(m.sum()), 'h_norm': round(float(pp['h_norm'][m].mean()), 4),
                                'max_share': round(float(pp['max'][m].mean()), 4)} for k, m in kinds.items()}
            vis = per_position(s)
        else:
            assert [(r['gid'], r['steps'], r['scores']) for r in res] == \
                [(r['gid'], r['steps'], r['scores']) for r in ref[0]], 'arms played different games'
            for k in ('x', 'opt_off', 'opt_action', 'z'):
                assert np.array_equal(s[k], ref[1][k]), f'arms recorded different positions ({k})'
        pa = per_position(s)
        d = kl(s['pi'], prior, seg, len(n))
        out['arms'][name] = {k: {'h_norm': round(float(pa['h_norm'][m].mean()), 4),
                                 'max_share': round(float(pa['max'][m].mean()), 4),
                                 'argmax_differs_from_visits': round(float((pa['argmax'] != vis['argmax'])[m].mean()), 4),
                                 'argmax_differs_from_prior': round(float((pa['argmax'] != pp['argmax'])[m].mean()), 4),
                                 'kl_to_prior': round(float(d[m].mean()), 4)}
                             for k, m in kinds.items()}
        out['arms'][name]['seconds'] = round(time.time() - t0, 1)
        if verbose:
            print(name, json.dumps(out['arms'][name]), flush=True)
    return out


def prior_positions(model, game, spec, n, seed=0, keep=0.15, device='cpu'):
    """`n` positions (decisions with >= 2 options) from games in which both seats sample the
    network prior."""
    m, _ = load(model, device, game=game)
    ev = NetEvaluator(m, device)
    rng = np.random.default_rng(seed)
    out, k = [], 0
    while len(out) < n:
        g = G.engine().Game(game, spec, 2, seed * 100_003 + k)
        k += 1
        while not g.over and len(out) < n:
            legal = g.legal()
            if len(legal) > 1:
                pri, _, _ = ev.evaluate(g.encode(g.to_move)[None], np.array([g.n_players], np.int32),
                                        np.array([0, len(legal)], np.int32), np.asarray(legal, np.int32))
                if rng.random() < keep:
                    out.append(g.clone())
                a = legal[int(rng.choice(len(legal), p=pri / pri.sum()))]
            else:
                a = legal[0]
            g.step(int(a))
    return out


def search_many(games, ev, sims, seeds, c_puct=1.0, chance_k=0, noise=None):
    """One Rust search tree per game, `sims` simulations each, the leaves of all trees
    evaluated in one batch per round. noise = (alpha, eps): root Dirichlet noise once the root
    is expanded (as self-play's full searches). -> the trees."""
    trees = [g.search_tree(True, int(s)) for g, s in zip(games, seeds)]
    for _ in range(sims):
        leaves = []
        for k, (g, t) in enumerate(zip(games, trees)):
            r = t.simulate(g, c_puct, chance_k)
            if r is not None:
                leaves.append((k, r))
        if not leaves:
            continue
        acts = [r[2] for _, r in leaves]
        off = np.zeros(len(acts) + 1, np.int32)
        off[1:] = np.cumsum([len(a) for a in acts])
        npl = np.asarray([games[k].n_players for k, _ in leaves], np.int32)
        pri, val, mar = ev.evaluate(np.stack([r[1] for _, r in leaves]), npl, off,
                                    np.concatenate([np.asarray(a, np.int32) for a in acts]))
        for b, (k, _) in enumerate(leaves):
            t = trees[k]
            was = t.root_expanded
            t.expand(pri[off[b]:off[b + 1]].tolist(), val[b].tolist(), mar[b].tolist())
            if not was and noise is not None:
                t.add_noise(noise[0], noise[1], int(seeds[k]) * 31 + 7)
    return trees


def completed_q(t):
    """Completed Q of a searched root (Q of visited options, v_mix of the others)."""
    nv = np.asarray(t.root_visits, np.float64)
    w = np.asarray(t.root_values, np.float64)
    p = np.maximum(np.asarray(t.root_priors0, np.float64), mcts.PRIOR_FLOOR)
    vis = nv > 0
    q = np.where(vis, w / np.maximum(nv, 1), 0.0)
    if vis.any():
        v_mix = (t.root_value0 + nv.sum() * (p[vis] * q[vis]).sum() / p[vis].sum()) / (1 + nv.sum())
    else:
        v_mix = t.root_value0
    return np.where(vis, q, v_mix)


def quality(model, game=None, spec=None, positions=300, sims=128, reps=2, ref_sims=2048,
            c_visit=50.0, c_scales=(1.0, 0.3, 0.1), chance_k=2, seed=0, verbose=True, device='cpu'):
    """-> report dict (see the module docs). game / spec None: the game package's defaults."""
    game, spec = _game_spec(game, spec)
    t0 = time.time()
    gs = prior_positions(model, game, spec, positions, seed, device=device)
    m, _ = load(model, device, game=game)
    ev = NetEvaluator(m, device)
    names = ['prior', 'visits'] + [f'cq c_scale {cs:g}' for cs in c_scales] + ['greedy Q']
    acc = {k: {'regret': [], 'best_mass': [], 'argmax_best': []} for k in names}
    ref = search_many(gs, ev, ref_sims, [seed * 7 + 1_000_000 + k for k in range(len(gs))], chance_k=chance_k)
    qref = [completed_q(t) for t in ref]
    for r in range(reps):
        trees = search_many(gs, ev, sims, [seed * 7 + 10_000 * (r + 1) + k for k in range(len(gs))],
                            chance_k=chance_k, noise=(0.3, 0.25))
        for t, qr in zip(trees, qref):
            nv = np.asarray(t.root_visits, np.float64)
            tg = {'prior': np.asarray(t.root_priors0, np.float64), 'visits': nv / max(nv.sum(), 1)}
            for cs in c_scales:
                tg[f'cq c_scale {cs:g}'] = np.asarray(t.root_cq_target(c_visit, cs), np.float64)
            g1 = np.zeros(len(nv))
            g1[int(np.argmax(completed_q(t)))] = 1
            tg['greedy Q'] = g1
            best = int(np.argmax(qr))
            for k, pi in tg.items():
                pi = pi / pi.sum()
                acc[k]['regret'].append(float(qr.max() - pi @ qr))
                acc[k]['best_mass'].append(float(pi[best]))
                acc[k]['argmax_best'].append(float(int(np.argmax(pi)) == best))
    out = {'model': model, 'game': game, 'positions': len(gs),
           'options': round(float(np.mean([len(g.legal()) for g in gs])), 2), 'sims': sims, 'reps': reps,
           'ref_sims': ref_sims, 'c_visit': c_visit, 'chance_k': chance_k, 'device': device, 'targets': {}}
    for k, d in acc.items():
        out['targets'][k] = {m_: round(float(np.mean(v)), 4) for m_, v in d.items()}
        out['targets'][k]['regret_se'] = round(float(np.std(d['regret']) / np.sqrt(len(d['regret']))), 4)
    out['seconds'] = round(time.time() - t0, 1)
    if verbose:
        for k, d in out['targets'].items():
            print(f'{k:>16}: regret {100 * d["regret"]:.2f} +-{100 * d["regret_se"]:.2f} pts, mass on best '
                  f'{d["best_mass"]:.3f}, argmax = best {d["argmax_best"]:.3f}', flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('sharpness', 'quality'):
        p = sub.add_parser(name)
        p.add_argument('--model', required=True)
        p.add_argument('--game', default=package.active().game)
        p.add_argument('--spec', default=None, help="game spec JSON (default: the game package's spec of the game)")
        p.add_argument('--device', default='cpu', help="network device ('cuda' works; the GPU is shared)")
        p.add_argument('--sims', type=int, default=128)
        p.add_argument('--chance-k', type=int, default=2)
        p.add_argument('--c-visit', type=float, default=50.0)
        p.add_argument('--c-scales', default='1.0,0.3,0.1')
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--out', default=None, help='write the report (JSON) here too')
        if name == 'sharpness':
            p.add_argument('--games', type=int, default=100)
            p.add_argument('--sims-fast', type=int, default=32)
            p.add_argument('--p-full', type=float, default=0.25)
            p.add_argument('--threads', type=int, default=3)
        else:
            p.add_argument('--positions', type=int, default=300)
            p.add_argument('--reps', type=int, default=2)
            p.add_argument('--ref-sims', type=int, default=2048)
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    try:
        import os
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10)
    except Exception:                              # noqa: BLE001
        pass
    cs = tuple(float(x) for x in a.c_scales.split(','))
    spec = a.spec
    if a.cmd == 'sharpness':
        r = compare(a.model, a.game, spec, a.games, a.sims, a.sims_fast, a.p_full, chance_k=a.chance_k,
                    c_visit=a.c_visit, c_scales=cs, seed=a.seed, threads=a.threads, device=a.device)
    else:
        r = quality(a.model, a.game, spec, a.positions, a.sims, a.reps, a.ref_sims, a.c_visit, cs, a.chance_k, a.seed,
                    device=a.device)
    print(json.dumps(r))
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(r, f, indent=1)


if __name__ == '__main__':
    main()
