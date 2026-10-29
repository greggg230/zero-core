"""Rust search vs the Python reference search (docs/search.md 6). Parity cannot be bit-exact
(different RNG streams), so it is statistical:

* `positions`: the same positions searched by both trees with the same deterministic stand-in
  evaluator (a fixed random linear function of the features: priors over the legal actions,
  value of the decider), many repetitions each; the total-variation distance between the mean
  root visit distributions of Rust and Python must be at the noise floor (Python vs Python
  with other seeds), and the root win estimates must agree. The same for the completed-Q
  policy targets (docs/search.md 3.1) of the searched roots; `cq_formula` checks that the
  Python formula on a Rust root's statistics gives the Rust target.
* `select_formula`: the move rule (docs/search.md 3.3) - the Python `mcts.select_index` on Rust
  roots' statistics against the Rust driver's own choice (`SearchTree.root_select`, the code of a
  driver's non-sampled moves), exactly (but on near-ties of the logits, ulps apart).
* `forced`: the root minimum visits (docs/search.md 3.5) - positions searched by both trees
  with the floor k and a lopsided stand-in prior (0.999 on one option): every option reaches k
  visits in both, and the mean visit distributions with and without the forced walks, the forced
  walks per option, the completed-Q targets and the root win without the forced walks agree at the
  noise floor (Python vs Python).
* `h2h`: games between a Rust-tree seat and a Python-tree seat (`play.PyDriver`, seats
  alternating), both searching with the same network and settings (and move rule): ~50% within
  error.

  python -m zero.parity positions [--sims 128 --reps 32 --chance-k 0 --cq-c-scale 1.0]
  python -m zero.parity positions --game toy_team --players 4     (deferred leaves, teams of 2)
  python -m zero.parity forced [--k 4 --sims 96 --reps 24]         (root minimum visits)
  python -m zero.parity h2h --model runs/toy1/models/iter_0010.pt [--games 2000 --sims 32 --select cq]
  (or python -m <pkg>.az.parity ... in a game repo)
"""
import argparse
import json
import math
import random
import time

import numpy as np

from . import mcts
from .games import engine
from .protocol import MAX_SEATS


class LinearEval:
    """Deterministic stand-in evaluator: priors = softmax(x Wp[:, acts]), value of the decider =
    sigmoid(x wv), the rest split evenly; margin 0."""

    def __init__(self, num_features, num_actions, seed=0, scale=1.0):
        r = np.random.default_rng(seed)
        self.Wp = r.normal(0, scale, (num_features, num_actions))
        self.wv = r.normal(0, scale, num_features)

    def __call__(self, x, acts, n):
        if len(acts):
            lg = x.astype(np.float64) @ self.Wp[:, acts]
            e = np.exp(lg - lg.max())
            pri = (e / e.sum()).tolist()
        else:                                      # a deferred terminal: a value, no priors
            pri = []
        w = 1.0 / (1.0 + math.exp(-float(x.astype(np.float64) @ self.wv)))
        v = [0.0] * MAX_SEATS
        v[0] = w
        for j in range(1, n):
            v[j] = (1 - w) / (n - 1)
        return pri, v, [0.0] * MAX_SEATS


def search_py(g, ev, sims, seed, c_puct=1.0, chance_k=0, redet=True, cq=None, tree=False, root_min=0):
    """-> (root visits, root win[, completed-Q target with cq = (c_visit, c_scale)][, the tree]);
    `root_min`: the root minimum visits of the search (docs/search.md 3.5; 0 = off)."""
    rng = random.Random(seed)
    t = mcts.Tree(g, redet, rng)
    t.set_root_min(root_min)
    for _ in range(sims):
        r = mcts.simulate(t, rng, c_puct, chance_k)
        if r is None:
            continue
        node, leaf, path = r
        pri, v, m = ev(leaf.encode(node.player), node.acts, g.n_players)
        mcts.expand_and_backup(node, path, pri, v, m, g.n_players, team=t.team, tree=t)
    out = (np.asarray(t.root.Nv, float), mcts.root_win(t.root))
    if cq is not None:
        out += (np.asarray(mcts.cq_target(t.root, *cq)),)
    return out + (t,) if tree else out


def search_rs(g, ev, sims, seed, c_puct=1.0, chance_k=0, redet=True, cq=None, tree=False, root_min=0):
    """`search_py` with the Rust tree (`<package>._engine.SearchTree`)."""
    t = g.search_tree(redet, seed)
    t.set_root_min(root_min)
    for _ in range(sims):
        r = t.simulate(g, c_puct, chance_k)
        if r is None:
            continue
        p, x, acts = r
        pri, v, m = ev(x, acts, g.n_players)
        t.expand(pri, v, m)
    out = (np.asarray(t.root_visits, float), t.root_win())
    if cq is not None:
        out += (np.asarray(t.root_cq_target(*cq), float),)
    return out + (t,) if tree else out


def sample_positions(game='toy', n=12, seed=0, players=2, depth=(5, 60)):
    """Positions from random play at varying depths (the decider has >= 3 options)."""
    E = engine()
    rng = random.Random(seed)
    out = []
    k = 0
    lo, hi = depth
    while len(out) < n:
        g = E.Game(game, None, players, 10_000 + seed * 1000 + k)
        k += 1
        depth = rng.randrange(lo, hi)
        for _ in range(depth):
            if g.over:
                break
            g.step(rng.choice(g.legal()))
        if not g.over and len(g.legal()) >= 3:
            out.append(g)
    return out


def tv(a, b):
    return 0.5 * float(np.abs(a / a.sum() - b / b.sum()).sum())


def positions(game="toy", n=12, sims=96, reps=24, chance_k=0, c_puct_rs=1.0, seed=0, verbose=False, scale=0.15,
              cq=(50.0, 1.0), players=2, depth=(5, 60)):
    """-> dict(tv_rs_py, tv_py_py, win_diff, tv_cq_rs_py, tv_cq_py_py) averaged over the
    positions (tv_cq_*: the same for the mean completed-Q targets with cq = (c_visit, c_scale)).
    `players` / `depth`: the positions' player count and random-play depth range (toy_team: 4
    players, depth 0-8, so that searches reach its deferred terminals)."""
    info = engine().game_info_of(game)
    ev = LinearEval(info["num_features"], info["num_actions"], seed=1, scale=scale)
    res = {'tv_rs_py': [], 'tv_py_py': [], 'win_diff': [], 'tv_cq_rs_py': [], 'tv_cq_py_py': []}
    for i, g in enumerate(sample_positions(game, n, seed, players, depth)):
        rs = [search_rs(g, ev, sims, 1000 + r, c_puct_rs, chance_k, cq=cq) for r in range(reps)]
        pa = [search_py(g, ev, sims, 2000 + r, 1.0, chance_k, cq=cq) for r in range(reps)]
        pb = [search_py(g, ev, sims, 3000 + r, 1.0, chance_k, cq=cq) for r in range(reps)]
        mr, ma, mb = (np.mean([x[0] for x in s], 0) for s in (rs, pa, pb))
        res['tv_rs_py'].append(tv(mr, ma))
        res['tv_py_py'].append(tv(ma, mb))
        cr, ca, cb = (np.mean([x[2] for x in s], 0) for s in (rs, pa, pb))
        res['tv_cq_rs_py'].append(tv(cr, ca))
        res['tv_cq_py_py'].append(tv(ca, cb))
        wr = np.mean([x[1][g.to_move] for x in rs])
        wa = np.mean([x[1][g.to_move] for x in pa])
        res['win_diff'].append(abs(wr - wa))
        if verbose:
            print(f'position {i}: {len(g.legal())} options, TV rs-py {res["tv_rs_py"][-1]:.3f} py-py '
                  f'{res["tv_py_py"][-1]:.3f}, cq TV rs-py {res["tv_cq_rs_py"][-1]:.3f} py-py '
                  f'{res["tv_cq_py_py"][-1]:.3f}, root win rs {wr:.3f} py {wa:.3f}', flush=True)
    return {k: float(np.mean(v)) for k, v in res.items()} | {'positions': n, 'sims': sims, 'reps': reps,
                                                               'cq': list(cq)}


class SkewEval:
    """`LinearEval` with a lopsided prior: 1 - eps on the option its priors favour, eps shared by the
    others in proportion (MTG Zero's Delver mulligan: >= 0.98 on KEEP in 433 of 444 decisions)."""

    def __init__(self, base, eps=0.001):
        self.base, self.eps = base, eps

    def __call__(self, x, acts, n):
        pri, v, m = self.base(x, acts, n)
        if len(pri) >= 2:
            b = int(np.argmax(pri))
            rest = sum(p for i, p in enumerate(pri) if i != b)
            pri = [1 - self.eps if i == b else self.eps * p / rest for i, p in enumerate(pri)]
        return pri, v, m


def _py_pruned(t):
    return np.asarray(mcts.root_visits_pruned(t), float), np.asarray(t.forced, float), mcts.root_win_pruned(t)


def _rs_pruned(t):
    return np.asarray(t.root_visits_pruned, float), np.asarray(t.root_forced, float), t.root_win_pruned()


def forced(game='toy', n=10, sims=96, reps=24, k=4, seed=0, scale=0.15, eps=0.001, cq=(50.0, 1.0), max_options=6,
           verbose=False, k_rs=None):
    """The root minimum visits in both trees (docs/search.md 3.5): positions of 2..max_options options
    (2 k options <= sims), the lopsided `SkewEval`, the floor `k`. -> dict: `floor_met` (every option
    has >= k visits in every search of both trees), the TVs Rust-Python and Python-Python of the mean
    visit distributions (`tv_*`), of the mean distributions without the forced walks (`tv_pruned_*`),
    of the mean forced walks per option (`tv_forced_*`), of the mean completed-Q targets (`tv_cq_*`),
    the mean forced walks per search (`forced_rs`, `forced_py`), |Rust - Python| of the decider's
    mean root win without the forced walks (`win_diff`), averaged over the positions; `starved_k0`:
    the share of options without a visit at k 0 (Rust). `k_rs`: another floor for the Rust tree (a
    power check)."""
    info = engine().game_info_of(game)
    ev = SkewEval(LinearEval(info['num_features'], info['num_actions'], seed=1, scale=scale), eps)
    E = engine()
    rng = random.Random(seed)
    pos = []
    i = 0
    while len(pos) < n:
        g = E.Game(game, None, 2, 20_000 + seed * 1000 + i)
        i += 1
        for _ in range(rng.randrange(3, 60)):
            if g.over:
                break
            g.step(rng.choice(g.legal()))
        if not g.over and 2 <= len(g.legal()) <= max_options and 2 * len(g.legal()) * k <= sims:
            pos.append(g)
    names = ('tv', 'tv_pruned', 'tv_forced', 'tv_cq')
    res = {f'{a}_{b}': [] for a in names for b in ('rs_py', 'py_py')}
    res.update(win_diff=[], forced_rs=[], forced_py=[], starved_k0=[])
    met = True
    for j, g in enumerate(pos):
        def runs(fn, pruned, base, kk=k):
            out = []
            for r in range(reps):
                nv, _, cqt, t = fn(g, ev, sims, base + r, cq=cq, tree=True, root_min=kk)
                pv, fv, pw = pruned(t)
                out.append((nv, pv, fv, cqt, pw))
            return out
        rs = runs(search_rs, _rs_pruned, 1000, k if k_rs is None else k_rs)
        pa = runs(search_py, _py_pruned, 2000)
        pb = runs(search_py, _py_pruned, 3000)
        met &= all(x[0].min() >= k for s in (rs, pa, pb) for x in s)
        for name, idx in zip(names, range(4)):
            mr, ma, mb = (np.mean([x[idx] for x in s], 0) for s in (rs, pa, pb))
            if name == 'tv_forced' and min(mr.sum(), ma.sum(), mb.sum()) == 0:
                res[f'{name}_rs_py'].append(0.0)
                res[f'{name}_py_py'].append(0.0)
                continue
            res[f'{name}_rs_py'].append(tv(mr, ma))
            res[f'{name}_py_py'].append(tv(ma, mb))
        res['forced_rs'].append(float(np.mean([x[2].sum() for x in rs])))
        res['forced_py'].append(float(np.mean([x[2].sum() for x in pa])))
        wr = np.mean([x[4][g.to_move] for x in rs])
        wa = np.mean([x[4][g.to_move] for x in pa])
        res['win_diff'].append(abs(wr - wa))
        k0 = [search_rs(g, ev, sims, 4000 + r)[0] for r in range(4)]
        res['starved_k0'].append(float(np.mean([(x == 0).mean() for x in k0])))
        if verbose:
            print(f'position {j}: {len(g.legal())} options, forced rs {res["forced_rs"][-1]:.2f} py '
                  f'{res["forced_py"][-1]:.2f}, TV rs-py {res["tv_rs_py"][-1]:.3f} (py-py {res["tv_py_py"][-1]:.3f}), '
                  f'pruned {res["tv_pruned_rs_py"][-1]:.3f} ({res["tv_pruned_py_py"][-1]:.3f}), cq '
                  f'{res["tv_cq_rs_py"][-1]:.3f} ({res["tv_cq_py_py"][-1]:.3f}), starved at k 0 '
                  f'{res["starved_k0"][-1]:.2f}', flush=True)
    return {kk: float(np.mean(v)) for kk, v in res.items()} | {'floor_met': bool(met), 'positions': n, 'sims': sims,
                                                                 'reps': reps, 'k': k}


def cq_formula(game='toy', n=8, sims=64, seed=0, cq_settings=((50.0, 1.0), (50.0, 0.1), (10.0, 3.0)), scale=0.15):
    """The Python `completed_q_policy` on Rust roots' statistics (P0, N, W, network value) vs
    the Rust tree's own target. -> max |difference|."""
    info = engine().game_info_of(game)
    ev = LinearEval(info["num_features"], info["num_actions"], seed=1, scale=scale)
    worst = 0.0
    for k, g in enumerate(sample_positions(game, n, seed)):
        *_, t = search_rs(g, ev, sims, 500 + k, tree=True)
        for cv, cs in cq_settings:
            py = mcts.completed_q_policy(t.root_priors0, t.root_visits, t.root_values, t.root_value0, cv, cs)
            worst = max(worst, float(np.abs(np.asarray(py) - np.asarray(t.root_cq_target(cv, cs))).max()))
    return worst


SELECT_RULES = (('visits', 0, 50.0, 1.0), ('cq', 0, 50.0, 1.0), ('cq', 1, 50.0, 1.0), ('cq', 2, 50.0, 1.0),
                ('cq', 4, 10.0, 3.0), ('cq', 0, 50.0, 0.1), ('cq', 4, 50.0, 0.3))


def select_formula(game='toy', n=8, sims=(4, 8, 24, 64), seed=0, rules=SELECT_RULES, scale=0.15, eps=1e-9):
    """The Python `select_index` on Rust roots' statistics (P0, N, W, network value) vs the Rust
    tree's own `root_select` for each rule (rule, min_visits, c_visit, c_scale). -> {checked,
    agree, near_ties, mismatches}: a disagreement counts as a near-tie when the two options'
    logits are within `eps` (f64 rounding of ln / exp between the platforms' libm), else it is a
    mismatch (listed)."""
    info = engine().game_info_of(game)
    ev = LinearEval(info["num_features"], info["num_actions"], seed=1, scale=scale)
    out = {'checked': 0, 'agree': 0, 'near_ties': 0, 'mismatches': []}
    for k, g in enumerate(sample_positions(game, n, seed)):
        for ns in sims:
            *_, t = search_rs(g, ev, ns, 700 + 31 * k + ns, tree=True)
            P0, Nv, W, v = t.root_priors0, t.root_visits, t.root_values, t.root_value0
            for rule, mv, cv, cs in rules:
                py = mcts.select_index(P0, Nv, W, v, rule, mv, cv, cs)
                rs = t.root_select(rule, mv, cv, cs)
                out['checked'] += 1
                if py == rs:
                    out['agree'] += 1
                    continue
                lg = mcts.completed_q_logits(P0, Nv, W, v, cv, cs) if rule == 'cq' else None
                if lg is not None and abs(lg[py] - lg[rs]) < eps:
                    out['near_ties'] += 1
                else:
                    out['mismatches'].append({'position': k, 'sims': ns, 'rule': [rule, mv, cv, cs], 'py': py, 'rs': rs})
    return out


def h2h(model, games=2000, sims=32, chance_k=0, conc=128, seed=0, game='toy', verbose=True, select='visits',
        select_min_visits=0, select_c_visit=None, select_c_scale=None):
    """Rust-tree seat vs Python-tree seat, same network and move rule (and the 'cq' rule's own
    transform select_c_visit / select_c_scale, None: 50 / 1), seats alternating. -> summary."""
    import torch
    from .infer import NetEvaluator, drive
    from .net import load
    from .play import NNSeat, PyDriver
    torch.set_num_threads(1)
    m, _ = load(model)
    ev = NetEvaluator(m)
    kw = dict(select=select, select_min_visits=select_min_visits, select_c_visit=select_c_visit,
              select_c_scale=select_c_scale)
    rs = NNSeat('m', sims, chance_k=chance_k, impl='rs', **kw)
    py = NNSeat('m', sims, chance_k=chance_k, impl='py', **kw)
    t0 = time.time()
    drv = PyDriver(game, games, lambda g: [rs, py] if g % 2 == 0 else [py, rs], concurrency=conc, seed=seed)
    drive(drv, {'m': ev})
    rows = drv.take_results()
    sc = [r['result'][0 if r['gid'] % 2 == 0 else 1] for r in rows]
    w = float(np.mean(sc))
    se = float(np.std(sc) / math.sqrt(len(sc)))
    out = {'rust_score': round(w, 4), 'se': round(se, 4), 'ci95': round(1.96 * se, 4), 'games': len(rows),
           'sims': sims, 'chance_k': chance_k, 'seconds': round(time.time() - t0, 1), 'model': model,
           'select': rs.select_tag}
    if verbose:
        print(json.dumps(out))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('positions')
    p.add_argument('--positions', type=int, default=12)
    p.add_argument('--sims', type=int, default=128)
    p.add_argument('--reps', type=int, default=32)
    p.add_argument('--chance-k', type=int, default=0)
    p.add_argument('--c-puct-rs', type=float, default=1.0, help='a deliberately different Rust c_puct (power check)')
    p.add_argument('--game', default='toy')
    p.add_argument('--players', type=int, default=2)
    p.add_argument('--depth', type=int, nargs=2, default=[5, 60], help='random-play depth range of the positions')
    p.add_argument('--cq-c-visit', type=float, default=50.0)
    p.add_argument('--cq-c-scale', type=float, default=1.0)
    h = sub.add_parser('h2h')
    h.add_argument('--model', required=True)
    h.add_argument('--games', type=int, default=2000)
    h.add_argument('--sims', type=int, default=32)
    h.add_argument('--chance-k', type=int, default=0)
    h.add_argument('--seed', type=int, default=0)
    h.add_argument('--select', default='visits', choices=['visits', 'cq'])
    h.add_argument('--select-min-visits', type=int, default=0)
    h.add_argument('--select-c-visit', type=float, default=None, help="the 'cq' rule's transform (default 50 / 1)")
    h.add_argument('--select-c-scale', type=float, default=None)
    sf = sub.add_parser('select')
    sf.add_argument('--game', default='toy')
    sf.add_argument('--positions', type=int, default=8)
    fo = sub.add_parser('forced')
    fo.add_argument('--game', default='toy')
    fo.add_argument('--positions', type=int, default=10)
    fo.add_argument('--sims', type=int, default=96)
    fo.add_argument('--reps', type=int, default=24)
    fo.add_argument('--k', type=int, default=4)
    fo.add_argument('--eps', type=float, default=0.001, help='the prior left to the options but the favourite')
    fo.add_argument('--cq-c-scale', type=float, default=1.0)
    a = ap.parse_args(argv)
    if a.cmd == 'positions':
        print(json.dumps(positions(a.game, n=a.positions, sims=a.sims, reps=a.reps, chance_k=a.chance_k,
                                   c_puct_rs=a.c_puct_rs, verbose=True, cq=(a.cq_c_visit, a.cq_c_scale),
                                   players=a.players, depth=tuple(a.depth))))
    elif a.cmd == 'select':
        print(json.dumps(select_formula(a.game, n=a.positions)))
    elif a.cmd == 'forced':
        print(json.dumps(forced(a.game, n=a.positions, sims=a.sims, reps=a.reps, k=a.k, eps=a.eps,
                                cq=(50.0, a.cq_c_scale), verbose=True)))
    else:
        h2h(a.model, a.games, a.sims, a.chance_k, seed=a.seed, select=a.select, select_min_visits=a.select_min_visits,
            select_c_visit=a.select_c_visit, select_c_scale=a.select_c_scale)


if __name__ == '__main__':
    main()
