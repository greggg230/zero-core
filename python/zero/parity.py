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
* `root_view`: root-view leaf values (docs/search.md 3.6) - positions searched by both trees with
  the leaves of the other seat valued from the root decider's view: the mean visit distributions,
  completed-Q targets and root win agree at the noise floor (Python vs Python), and differ from the
  mover-view search of the same positions (the setting matters: a power check).
* `belief`: the root's belief (docs/search.md 3.8) - positions searched by both trees with a fixed
  belief of the hidden cards (the toy: its next round's card, a sharp prediction on one unrevealed
  card) dealing their worlds: the mean visit distributions, completed-Q targets and root win agree at
  the noise floor (Python vs Python), the believed card's share of the leaves' worlds is the deal's
  in both, and the searches differ from the uniform deal's of the same positions (a power check).
* `gumbel`: the Gumbel root search (docs/search.md 3.7) - positions searched by both trees with the
  same Gumbel noise per repetition (only the trees' determinization / chance streams differ): the
  mean visit distributions, the mean walks of the searches themselves, the completed-Q targets, the
  moves (`gumbel_choose`) and the root values (`gumbel_q`) agree at the noise floor (Python vs
  Python with the same noise); `gumbel_formula`: the Python `gumbel_choose` / `gumbel_q` / next
  root walk on Rust roots' statistics against the Rust tree's own, exactly.
* `h2h`: games between a Rust-tree seat and a Python-tree seat (`play.PyDriver`, seats
  alternating), both searching with the same network and settings (and move rule): ~50% within
  error.

Variable-length rows (a game with tail segments / pointer overrides, docs/search.md 2.15; `--game toy_tail`): both
trees' leaves carry their rows' tails and overrides (`mcts.leaf_row`, `SearchTree.simulate(rows=True)`) and the
stand-in `LinearEval` reads them (its value every tail token's floats, its priors each override and its target
token's floats), so the statistics above compare the searches of the full rows; `rows_tv` the power check (the Rust
tree fed its rows without their tails and overrides). `spec`: the positions' game spec (toy_tail: `{"items0": 9}`,
every row with a tail).

  python -m zero.parity positions [--sims 128 --reps 32 --chance-k 0 --cq-c-scale 1.0]
  python -m zero.parity positions --game toy_team --players 4     (deferred leaves, teams of 2)
  python -m zero.parity forced [--k 4 --sims 96 --reps 24]         (root minimum visits)
  python -m zero.parity root_view [--sims 96 --reps 24]            (root-view leaf values)
  python -m zero.parity belief [--sims 96 --reps 24 --tilt 1]       (the root's belief)
  python -m zero.parity gumbel [--sims 64 --reps 32 --m 16 --scale 1]  (the Gumbel root search)
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
    sigmoid(x wv), the rest split evenly; margin 0. `tails` (a game's `info['tails']`; docs/search.md 2.15): it
    takes each row's `tail` and `ov` too (`mcts.leaf_row`'s) - every tail token adds its floats . wt[seg] (scaled by
    its slot) to the value's logit, every override adds wo[0] (the generic logit) or wo[1] + its target token's
    floats . wt[seg] to its option's prior logit -, so a search that dropped any of them would differ. Its extra
    weights are drawn after Wp and wv: without tails, the evaluator of before."""

    def __init__(self, num_features, num_actions, seed=0, scale=1.0, tails=None):
        r = np.random.default_rng(seed)
        self.Wp = r.normal(0, scale, (num_features, num_actions))
        self.wv = r.normal(0, scale, num_features)
        self.widths = {int(t['seg']): int(t['width']) for t in tails or []}
        if self.widths:
            self.wt = {k: r.normal(0, 4 * scale, w) for k, w in sorted(self.widths.items())}
            self.wo = r.normal(0, 4 * scale, 2)

    def _row(self, k, tail, ov):
        """The row's tail and overrides -> (the value logit's term, each option's prior logit term)."""
        v, lg, tok = 0.0, np.zeros(k), {}
        if tail is not None:
            o = 0
            val = np.asarray(tail['t_val'], np.float64)
            for s_, l_ in zip(np.asarray(tail['t_seg']).tolist(), np.asarray(tail['t_slot']).tolist()):
                w = self.widths[s_]
                d = float(val[o:o + w] @ self.wt[s_])
                o += w
                v += d * (1 + 0.25 * (l_ % 3))
                tok[(s_, l_)] = d
        if ov is not None:
            for i, t in zip(np.asarray(ov[0]).tolist(), np.asarray(ov[1]).tolist()):
                lg[i] += self.wo[0] if t < 0 else self.wo[1] + tok.get(((t >> 40) & 0xffff, t & 0xffffffff), 0.0)
        return v, lg

    def __call__(self, x, acts, n, tail=None, ov=None):
        dv, dl = self._row(len(acts), tail, ov) if (tail is not None or ov is not None) else (0.0, 0.0)
        if len(acts):
            lg = x.astype(np.float64) @ self.Wp[:, acts] + dl
            e = np.exp(lg - lg.max())
            pri = (e / e.sum()).tolist()
        else:                                      # a deferred terminal: a value, no priors
            pri = []
        w = 1.0 / (1.0 + math.exp(-(float(x.astype(np.float64) @ self.wv) + dv)))
        v = [0.0] * MAX_SEATS
        v[0] = w
        for j in range(1, n):
            v[j] = (1 - w) / (n - 1)
        return pri, v, [0.0] * MAX_SEATS


def stand_in(game, seed=1, scale=0.15):
    """The parity checks' `LinearEval` of a game (its tail segments: it reads the rows' tails and overrides)."""
    info = engine().game_info_of(game)
    return LinearEval(info['num_features'], info['num_actions'], seed=seed, scale=scale, tails=info.get('tails') or None)


def _rows(g):
    """The game's variable-length rows (`mcts.rows_of`; None: none)."""
    return mcts.rows_of(dict(g.info()))


def _row_kw(rk, tail, ov):
    """An evaluator call's row keywords: none without variable-length rows (the call of before)."""
    if rk is None:
        return {}
    if rk.over(tail) is not None:
        raise ValueError(f'max_row_tokens: a leaf row of capacity {rk.capacity(tail)} > {rk.limit} (docs/search.md 2.15)')
    return {'tail': tail, 'ov': ov}


_NO_OV = (np.zeros(0, np.int32), np.zeros(0, np.int64))


def search_py(g, ev, sims, seed, c_puct=1.0, chance_k=0, redet=True, cq=None, tree=False, root_min=0,
              root_view=False, belief=None, worlds=None, gumbel=None):
    """-> (root visits, root win[, completed-Q target with cq = (c_visit, c_scale)][, the tree]);
    `root_min`: the root minimum visits of the search (docs/search.md 3.5; 0 = off); `root_view`:
    another seat's leaves take their value from the viewer's features (docs/search.md 3.6); `belief` =
    (predictions, tilt, clip, mix): the root's belief dealing the worlds (docs/search.md 3.8); `worlds`:
    a list that gets each leaf's world's auxiliary targets for the viewer (`Game.aux_target`) when the
    leaf is in the root's round; `gumbel` = (m, noise, c_visit, c_scale): a Gumbel root search (docs/search.md
    3.7; the schedule of `sims` simulations, as the driver's: a fresh root's walks are its first sims - 1)."""
    rng = random.Random(seed)
    t = mcts.Tree(g, redet, rng)
    t.set_root_min(root_min)
    if belief is not None:
        t.set_belief((np.asarray(belief[0], np.float32), belief[1], belief[2]), belief[3], g, rng)
    if gumbel is not None:
        t.set_gumbel(gumbel[0], sims, gumbel[1], gumbel[2], gumbel[3])
    rk = _rows(g)
    for _ in range(sims):
        r = mcts.simulate(t, rng, c_puct, chance_k)
        if r is None:
            continue
        node, leaf, path = r
        if worlds is not None and leaf.rounds == g.rounds:
            worlds.append(np.asarray(leaf.aux_target(t.viewer)))
        x, tl, ov = mcts.leaf_row(leaf, node.player, rk, opts=bool(node.acts))
        pri, v, m = ev(x, node.acts, g.n_players, **_row_kw(rk, tl, ov))
        vp = None
        if root_view and node.player != t.viewer:
            vx, vt, _ = mcts.leaf_row(leaf, t.viewer, rk)
            _, v, m = ev(vx, [], g.n_players, **_row_kw(rk, vt, None if rk is None or not rk.pointers else _NO_OV))
            vp = t.viewer
        mcts.expand_and_backup(node, path, pri, v, m, g.n_players, team=t.team, tree=t, value_player=vp)
    out = (np.asarray(t.root.Nv, float), mcts.root_win(t.root))
    if cq is not None:
        out += (np.asarray(mcts.cq_target(t.root, *cq)),)
    return out + (t,) if tree else out


def search_rs(g, ev, sims, seed, c_puct=1.0, chance_k=0, redet=True, cq=None, tree=False, root_min=0,
              root_view=False, belief=None, gumbel=None, drop_rows=False):
    """`search_py` with the Rust tree (`<package>._engine.SearchTree`; `gumbel`: as there). `drop_rows`: its leaves'
    rows evaluated without their tails and overrides (a power check of variable-length rows' parity)."""
    rk = _rows(g)
    t = g.search_tree(redet, seed)
    t.set_root_min(root_min)
    if belief is not None:
        t.set_belief(np.asarray(belief[0], np.float32), g, belief[1], belief[2], belief[3])
    if gumbel is not None:
        t.set_gumbel(gumbel[0], sims, list(gumbel[1]), gumbel[2], gumbel[3])
    for _ in range(sims):
        if rk is not None:
            r = t.simulate(g, c_puct, chance_k, root_view=root_view, rows=True)
        else:
            r = t.simulate(g, c_puct, chance_k, root_view=True) if root_view else t.simulate(g, c_puct, chance_k)
        if r is None:
            continue
        p, x, acts = r[:3]
        rr = (None, None, None) if rk is None else r[4]
        if rk is not None and drop_rows:
            # (the power check: the rows without their tails and overrides)
            et = {'t_seg': np.zeros(0, np.uint16), 't_slot': np.zeros(0, np.uint32), 't_val': np.zeros(0, np.float32)}                 if rk.widths else None
            rr = (et, _NO_OV if rk.pointers else None, et)
        pri, v, m = ev(x, acts, g.n_players, **_row_kw(rk, rr[0], rr[1]))
        if root_view and r[3] is not None:
            _, v, m = ev(r[3], [], g.n_players, **_row_kw(rk, rr[2], None if rk is None or not rk.pointers else _NO_OV))
        t.expand(pri, v, m)
    out = (np.asarray(t.root_visits, float), t.root_win())
    if cq is not None:
        out += (np.asarray(t.root_cq_target(*cq), float),)
    return out + (t,) if tree else out


def sample_positions(game='toy', n=12, seed=0, players=2, depth=(5, 60), spec=None):
    """Positions from random play at varying depths (the decider has >= 3 options); `spec`: the game's spec
    (None: its default)."""
    E = engine()
    rng = random.Random(seed)
    out = []
    k = 0
    lo, hi = depth
    while len(out) < n:
        g = E.Game(game, spec, players, 10_000 + seed * 1000 + k)
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
              cq=(50.0, 1.0), players=2, depth=(5, 60), spec=None, drop_rows=False):
    """-> dict(tv_rs_py, tv_py_py, win_diff, tv_cq_rs_py, tv_cq_py_py) averaged over the
    positions (tv_cq_*: the same for the mean completed-Q targets with cq = (c_visit, c_scale)).
    `players` / `depth`: the positions' player count and random-play depth range (toy_team: 4
    players, depth 0-8, so that searches reach its deferred terminals). `spec`: the game's (toy_tail: rows with
    tails); `drop_rows`: the Rust tree's rows without their tails and overrides (a power check)."""
    ev = stand_in(game, scale=scale)
    res = {'tv_rs_py': [], 'tv_py_py': [], 'win_diff': [], 'tv_cq_rs_py': [], 'tv_cq_py_py': []}
    for i, g in enumerate(sample_positions(game, n, seed, players, depth, spec)):
        rs = [search_rs(g, ev, sims, 1000 + r, c_puct_rs, chance_k, cq=cq, drop_rows=drop_rows) for r in range(reps)]
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


def root_view(game='toy', n=10, sims=96, reps=24, seed=0, scale=0.15, cq=(50.0, 1.0), verbose=False, rs_view=True,
              spec=None):
    """Root-view leaf values in both trees (docs/search.md 3.6): positions of random play, the
    stand-in `LinearEval` (its value reads the row's features: a viewer's features give another value
    than the decider's), both trees with `root_view`. -> dict: the TVs Rust-Python and Python-Python of
    the mean visit distributions (`tv_*`) and completed-Q targets (`tv_cq_*`), |Rust - Python| of the
    decider's mean root win (`win_diff`), and the TV between the Python tree's root-view and
    mover-view searches (`tv_view`: what the setting changes), averaged over the positions.
    `rs_view` False: the Rust tree searches mover-view (a power check: its TV to Python is then about
    `tv_view`). `spec`: the game's (toy_tail: the value-only rows carry tails too)."""
    ev = stand_in(game, scale=scale)
    res = {k: [] for k in ('tv_rs_py', 'tv_py_py', 'tv_cq_rs_py', 'tv_cq_py_py', 'win_diff', 'tv_view')}
    for i, g in enumerate(sample_positions(game, n, seed, spec=spec)):
        rs = [search_rs(g, ev, sims, 1000 + r, cq=cq, root_view=rs_view) for r in range(reps)]
        pa = [search_py(g, ev, sims, 2000 + r, cq=cq, root_view=True) for r in range(reps)]
        pb = [search_py(g, ev, sims, 3000 + r, cq=cq, root_view=True) for r in range(reps)]
        pm = [search_py(g, ev, sims, 4000 + r, cq=cq) for r in range(reps)]
        mr, ma, mb, mm = (np.mean([x[0] for x in s], 0) for s in (rs, pa, pb, pm))
        res['tv_rs_py'].append(tv(mr, ma))
        res['tv_py_py'].append(tv(ma, mb))
        res['tv_view'].append(tv(mm, ma))
        cr, ca, cb = (np.mean([x[2] for x in s], 0) for s in (rs, pa, pb))
        res['tv_cq_rs_py'].append(tv(cr, ca))
        res['tv_cq_py_py'].append(tv(ca, cb))
        wr = np.mean([x[1][g.to_move] for x in rs])
        wa = np.mean([x[1][g.to_move] for x in pa])
        res['win_diff'].append(abs(wr - wa))
        if verbose:
            print(f'position {i}: {len(g.legal())} options, TV rs-py {res["tv_rs_py"][-1]:.4f} py-py '
                  f'{res["tv_py_py"][-1]:.4f}, root vs mover view {res["tv_view"][-1]:.4f}, cq TV rs-py '
                  f'{res["tv_cq_rs_py"][-1]:.4f} py-py {res["tv_cq_py_py"][-1]:.4f}, root win rs {wr:.4f} py {wa:.4f}',
                  flush=True)
    return {k: float(np.mean(v)) for k, v in res.items()} | {'positions': n, 'sims': sims, 'reps': reps}


def belief_positions(n=8, seed=0):
    """Toy positions (>= 3 options) whose next round's card is hidden among >= 2 unrevealed cards of its
    stage, each with a belief sharp on one of those cards (its auxiliary target's column) -> [(game,
    predictions, the believed card, the cards it is among)]."""
    out = []
    for g in sample_positions('toy', 4 * n, seed, depth=(3, 60)):
        tgt = np.asarray(g.aux_target(g.to_move))
        if not tgt.any():
            continue                               # the last round: no next card
        # the cards it can be: those the deal of a few worlds gives
        cards = set()
        for s in range(40):
            w = g.clone()
            w.determinize(g.to_move, 7000 + s)
            cards.add(int(np.argmax(w.aux_target(g.to_move))))
        if len(cards) < 2:
            continue
        c = sorted(cards)[0]
        pred = np.full(6, 0.02, np.float32)
        pred[c] = 1.0 - 0.02 * 5
        out.append((g, pred, c, sorted(cards)))
        if len(out) == n:
            break
    return out


def belief(n=8, sims=96, reps=24, seed=0, scale=0.15, cq=(50.0, 1.0), tilt=1.0, clip=20.0, mix=0.0, verbose=False):
    """The root's belief in both trees (docs/search.md 3.8; the toy's next round's card): `belief_positions`,
    the stand-in `LinearEval`, both trees with the belief (tilt, clip, mix). -> dict: the TVs Rust-Python
    and Python-Python of the mean visit distributions (`tv_*`) and completed-Q targets (`tv_cq_*`), |Rust -
    Python| of the decider's mean root win (`win_diff`), the TV between the Python tree's belief and
    uniform searches (`tv_belief`: what the belief changes), and the believed card's share of the Python
    leaves' worlds (`share`) against the deal's own (`share_want`: w_c / sum w over the cards it is among,
    `zero.games.engine().belief_weight`), averaged over the positions."""
    E = engine()
    info = E.game_info_of('toy')
    ev = LinearEval(info['num_features'], info['num_actions'], seed=1, scale=scale)
    res = {k: [] for k in ('tv_rs_py', 'tv_py_py', 'tv_cq_rs_py', 'tv_cq_py_py', 'win_diff', 'tv_belief', 'share',
                           'share_want')}
    for i, (g, pred, c, cards) in enumerate(belief_positions(n, seed)):
        b = (pred, tilt, clip, mix)
        worlds = []
        rs = [search_rs(g, ev, sims, 1000 + r, cq=cq, belief=b) for r in range(reps)]
        pa = [search_py(g, ev, sims, 2000 + r, cq=cq, belief=b, worlds=worlds) for r in range(reps)]
        pb = [search_py(g, ev, sims, 3000 + r, cq=cq, belief=b) for r in range(reps)]
        pu = [search_py(g, ev, sims, 4000 + r, cq=cq) for r in range(reps)]
        mr, ma, mb, mu = (np.mean([x[0] for x in s], 0) for s in (rs, pa, pb, pu))
        res['tv_rs_py'].append(tv(mr, ma))
        res['tv_py_py'].append(tv(ma, mb))
        res['tv_belief'].append(tv(mu, ma))
        cr, ca, cb = (np.mean([x[2] for x in s], 0) for s in (rs, pa, pb))
        res['tv_cq_rs_py'].append(tv(cr, ca))
        res['tv_cq_py_py'].append(tv(ca, cb))
        wr = np.mean([x[1][g.to_move] for x in rs])
        wa = np.mean([x[1][g.to_move] for x in pa])
        res['win_diff'].append(abs(wr - wa))
        k = len(cards)
        w = {d: E.belief_weight(float(pred[d]), 1.0 / k, tilt, clip) for d in cards}
        want = (1 - mix) * w[c] / sum(w.values()) + mix / k
        res['share'].append(float(np.mean([int(np.argmax(x)) == c for x in worlds])) if worlds else want)
        res['share_want'].append(want)
        if verbose:
            print(f'position {i}: {len(g.legal())} options, next card among {cards}, TV rs-py {res["tv_rs_py"][-1]:.4f} '
                  f'py-py {res["tv_py_py"][-1]:.4f}, belief vs uniform {res["tv_belief"][-1]:.4f}, cq TV rs-py '
                  f'{res["tv_cq_rs_py"][-1]:.4f} py-py {res["tv_cq_py_py"][-1]:.4f}, root win rs {wr:.4f} py {wa:.4f}, '
                  f'card {c} share {res["share"][-1]:.3f} (deal {want:.3f})', flush=True)
    return {k: float(np.mean(v)) for k, v in res.items()} | {'positions': n, 'sims': sims, 'reps': reps, 'tilt': tilt,
                                                              'clip': clip, 'mix': mix}


def gumbel(game='toy', n=10, sims=64, reps=32, m=16, scale=1.0, seed=0, ev_scale=0.15, cq=(50.0, 0.3), eps=None,
           verbose=False, m_rs=None):
    """The Gumbel root search in both trees (docs/search.md 3.7): positions of random play, the
    stand-in `LinearEval` (`eps`: its lopsided `SkewEval`), repetition r of every tree with the same
    Gumbel noise (`scale` x Gumbel(0, 1) per root option from seed r), sigma's transform `cq`. -> dict:
    TVs Rust-Python and Python-Python (another tree stream, the same noise) of the mean root visit
    distributions (`tv_*`), of the mean walks of the searches (`tv_walks_*`), of the mean completed-Q
    targets (`tv_cq_*`) and of the move distributions (`tv_move_*`), |Rust - Python| of the decider's
    mean root value (`gumbel_q`, `q_diff`), the share of repetitions whose moves agree (`agree_*`);
    `budget_ok`: every search walked exactly sims - 1 root walks over min(m, options) options. `m_rs`:
    another m for the Rust tree (a power check)."""
    info = engine().game_info_of(game)
    ev = LinearEval(info['num_features'], info['num_actions'], seed=1, scale=ev_scale)
    if eps is not None:
        ev = SkewEval(ev, eps)
    names = ('tv', 'tv_walks', 'tv_cq', 'tv_move')
    res = {f'{a}_{b}': [] for a in names for b in ('rs_py', 'py_py')}
    res.update(q_diff=[], agree_rs_py=[], agree_py_py=[])
    ok = True
    for i, g in enumerate(sample_positions(game, n, seed)):
        k = len(g.legal())
        noise = [list(scale * np.random.default_rng(10_000 + r).gumbel(size=k)) for r in range(reps)]

        def runs(fn, base, mm):
            out = []
            for r in range(reps):
                nv, w, c, t = fn(g, ev, sims, base + r, cq=cq, tree=True, gumbel=(mm, noise[r], *cq))
                if fn is search_rs:
                    walks, mv, q = np.asarray(t.gumbel_visits, float), t.gumbel_choose(), t.gumbel_q()
                else:
                    walks, mv, q = np.asarray(t.gumbel.n, float), mcts.gumbel_choose(t), mcts.gumbel_q(t)
                out.append((nv, walks, c, mv, q))
            return out
        rs = runs(search_rs, 1000, m if m_rs is None else m_rs)
        pa = runs(search_py, 2000, m)
        pb = runs(search_py, 3000, m)
        for s_ in (rs, pa, pb):
            ok &= all(x[1].sum() == sims - 1 and (x[1] > 0).sum() == min(m, k, sims - 1) for x in s_) or s_ is rs and m_rs
        for name, idx in (('tv', 0), ('tv_walks', 1), ('tv_cq', 2)):
            mr, ma, mb = (np.mean([x[idx] for x in s_], 0) for s_ in (rs, pa, pb))
            res[f'{name}_rs_py'].append(tv(mr, ma))
            res[f'{name}_py_py'].append(tv(ma, mb))
        mr, ma, mb = (np.bincount([x[3] for x in s_], minlength=k).astype(float) for s_ in (rs, pa, pb))
        res['tv_move_rs_py'].append(tv(mr, ma))
        res['tv_move_py_py'].append(tv(ma, mb))
        res['agree_rs_py'].append(float(np.mean([a[3] == b[3] for a, b in zip(rs, pa)])))
        res['agree_py_py'].append(float(np.mean([a[3] == b[3] for a, b in zip(pb, pa)])))
        p = g.to_move
        res['q_diff'].append(abs(np.mean([x[4][p] for x in rs]) - np.mean([x[4][p] for x in pa])))
        if verbose:
            print(f'position {i}: {k} options, TV rs-py {res["tv_rs_py"][-1]:.4f} (py-py {res["tv_py_py"][-1]:.4f}), '
                  f'walks {res["tv_walks_rs_py"][-1]:.4f} ({res["tv_walks_py_py"][-1]:.4f}), cq '
                  f'{res["tv_cq_rs_py"][-1]:.4f} ({res["tv_cq_py_py"][-1]:.4f}), moves {res["tv_move_rs_py"][-1]:.3f} '
                  f'({res["tv_move_py_py"][-1]:.3f}), agree {res["agree_rs_py"][-1]:.2f} ({res["agree_py_py"][-1]:.2f}), '
                  f'q diff {res["q_diff"][-1]:.4f}', flush=True)
    return {kk: float(np.mean(v)) for kk, v in res.items()} | {'budget_ok': bool(ok), 'positions': n, 'sims': sims,
                                                                 'reps': reps, 'm': m, 'scale': scale, 'cq': list(cq)}


def gumbel_formula(game='toy', n=8, sims=(2, 5, 17, 64), ms=(1, 2, 4, 16), seed=0, scale=0.15, eps=1e-9):
    """The Python Gumbel formulas (`mcts.gumbel_choose`, `gumbel_q`, `gumbel_select`) on Rust roots'
    statistics (P0, N, W, network value, the search's walks, win sums, noise, schedule) against the
    Rust tree's move and root value - and the next root walk: the Rust tree's next simulation's root
    option (its path, `edge_outs` / visits) - exactly but on near-ties. -> {checked, agree, near_ties,
    q_max_diff, mismatches}."""
    info = engine().game_info_of(game)
    ev = LinearEval(info['num_features'], info['num_actions'], seed=1, scale=scale)
    out = {'checked': 0, 'agree': 0, 'near_ties': 0, 'q_max_diff': 0.0, 'mismatches': []}
    for k, g in enumerate(sample_positions(game, n, seed)):
        for ns in sims:
            for m in ms:
                noise = list(np.random.default_rng(31 * k + ns + m).gumbel(size=len(g.legal())))
                *_, t = search_rs(g, ev, ns, 900 + 31 * k + ns, tree=True, gumbel=(m, noise, 50.0, 0.3))
                # a Python tree holding the Rust root's statistics
                pt = mcts.Tree(g, True, random.Random(0))
                r = pt.root
                r.expand(t.root_priors0, [0.0] * g.n_players, g.n_players)
                r.Nv, r.W = list(t.root_visits), list(t.root_values)
                r.vec = [0.0] * g.n_players
                r.vec[r.player] = t.root_value0
                pt.set_gumbel(m, ns, noise, 50.0, 0.3)
                pt.gumbel.n = list(t.gumbel_visits)
                pt.gumbel.win = [list(w) for w in t.gumbel_win]
                pt.gumbel.walks = sum(pt.gumbel.n)
                out['checked'] += 2
                py, rs = mcts.gumbel_choose(pt), t.gumbel_choose()
                qp, qr = mcts.gumbel_q(pt), t.gumbel_q()
                if (qp is None) != (qr is None):
                    out['mismatches'].append({'position': k, 'sims': ns, 'm': m, 'q': [qp, qr]})
                    continue
                if qp is not None:
                    out['q_max_diff'] = max(out['q_max_diff'], float(np.abs(np.asarray(qp) - np.asarray(qr)).max()))
                # the next root walk: the Python rule on these statistics vs the option the Rust tree's next
                # simulation takes (its walks per option before and after)
                nxt = mcts.gumbel_select(pt)
                before = list(t.gumbel_visits)
                leaf = t.simulate(g, 1.0)
                if leaf is not None:
                    t.expand(*ev(leaf[1], leaf[2], g.n_players))
                took = [i for i, (a, b) in enumerate(zip(before, t.gumbel_visits)) if b != a]
                sc = mcts.gumbel_scores(pt)
                for what, a, b in (('move', py, rs), ('walk', nxt, took[0] if took else -1)):
                    if a == b:
                        out['agree'] += 1
                    elif b >= 0 and abs(sc[a] - sc[b]) < eps:
                        out['near_ties'] += 1
                    else:
                        out['mismatches'].append({'position': k, 'sims': ns, 'm': m, what: [a, b]})
    return out


def cq_formula(game='toy', n=8, sims=64, seed=0, cq_settings=((50.0, 1.0), (50.0, 0.1), (10.0, 3.0)), scale=0.15,
               prior_ws=(1.0,)):
    """The Python `completed_q_policy` on Rust roots' statistics (P0, N, W, network value) vs
    the Rust tree's own target, at each prior exponent of `prior_ws` (docs/search.md 3.13). -> max
    |difference|."""
    info = engine().game_info_of(game)
    ev = LinearEval(info["num_features"], info["num_actions"], seed=1, scale=scale)
    worst = 0.0
    for k, g in enumerate(sample_positions(game, n, seed)):
        *_, t = search_rs(g, ev, sims, 500 + k, tree=True)
        for cv, cs in cq_settings:
            for pw in prior_ws:
                py = mcts.completed_q_policy(t.root_priors0, t.root_visits, t.root_values, t.root_value0, cv, cs, pw)
                worst = max(worst, float(np.abs(np.asarray(py) - np.asarray(t.root_cq_target(cv, cs, pw))).max()))
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
        select_min_visits=0, select_c_visit=None, select_c_scale=None, spec=None, device='cpu'):
    """Rust-tree seat vs Python-tree seat, same network and move rule (and the 'cq' rule's own
    transform select_c_visit / select_c_scale, None: 50 / 1), seats alternating; `spec` the game's (a game with
    variable-length rows: both seats' leaves carry them, docs/search.md 2.15). -> summary."""
    import torch
    from .infer import NetEvaluator, drive
    from .net import load
    from .play import NNSeat, PyDriver
    torch.set_num_threads(1)
    m, _ = load(model, device)
    ev = NetEvaluator(m, device)
    kw = dict(select=select, select_min_visits=select_min_visits, select_c_visit=select_c_visit,
              select_c_scale=select_c_scale)
    rs = NNSeat('m', sims, chance_k=chance_k, impl='rs', **kw)
    py = NNSeat('m', sims, chance_k=chance_k, impl='py', **kw)
    t0 = time.time()
    drv = PyDriver(game, games, lambda g: [rs, py] if g % 2 == 0 else [py, rs], spec=spec, concurrency=conc, seed=seed)
    drive(drv, {'m': ev})
    rows = drv.take_results()
    sc = [r['result'][0 if r['gid'] % 2 == 0 else 1] for r in rows]
    w = float(np.mean(sc))
    se = float(np.std(sc) / math.sqrt(len(sc)))
    if any(r['aborted'] for r in rows):
        raise RuntimeError(f'h2h: {sum(r["aborted"] for r in rows)} of {len(rows)} games aborted (a row past the game\'s '
                           f'max_row_tokens, docs/search.md 2.15): not a parity sample')
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
    p.add_argument('--spec', default=None, help='the game spec (JSON; toy_tail: \'{"items0": 9}\')')
    p.add_argument('--drop-rows', action='store_true', help='the Rust tree without the rows\' tails (a power check)')
    h = sub.add_parser('h2h')
    h.add_argument('--model', required=True)
    h.add_argument('--game', default='toy')
    h.add_argument('--spec', default=None)
    h.add_argument('--device', default='cpu')
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
    bl = sub.add_parser('belief')
    bl.add_argument('--positions', type=int, default=8)
    bl.add_argument('--sims', type=int, default=96)
    bl.add_argument('--reps', type=int, default=24)
    bl.add_argument('--tilt', type=float, default=1.0)
    bl.add_argument('--clip', type=float, default=20.0)
    bl.add_argument('--mix', type=float, default=0.0)
    rv = sub.add_parser('root_view')
    rv.add_argument('--game', default='toy')
    rv.add_argument('--positions', type=int, default=10)
    rv.add_argument('--sims', type=int, default=96)
    rv.add_argument('--reps', type=int, default=24)
    rv.add_argument('--spec', default=None)
    gp = sub.add_parser('gumbel')
    gp.add_argument('--game', default='toy')
    gp.add_argument('--positions', type=int, default=10)
    gp.add_argument('--sims', type=int, default=64)
    gp.add_argument('--reps', type=int, default=32)
    gp.add_argument('--m', type=int, default=16)
    gp.add_argument('--scale', type=float, default=1.0, help='the Gumbel noise scale (0: none)')
    gp.add_argument('--eps', type=float, default=None, help="a lopsided prior (SkewEval's eps)")
    gp.add_argument('--cq-c-scale', type=float, default=0.3)
    a = ap.parse_args(argv)
    if a.cmd == 'positions':
        print(json.dumps(positions(a.game, n=a.positions, sims=a.sims, reps=a.reps, chance_k=a.chance_k,
                                   c_puct_rs=a.c_puct_rs, verbose=True, cq=(a.cq_c_visit, a.cq_c_scale),
                                   players=a.players, depth=tuple(a.depth), spec=a.spec, drop_rows=a.drop_rows)))
    elif a.cmd == 'select':
        print(json.dumps(select_formula(a.game, n=a.positions)))
    elif a.cmd == 'belief':
        print(json.dumps(belief(n=a.positions, sims=a.sims, reps=a.reps, tilt=a.tilt, clip=a.clip, mix=a.mix,
                                verbose=True)))
    elif a.cmd == 'root_view':
        print(json.dumps(root_view(a.game, n=a.positions, sims=a.sims, reps=a.reps, verbose=True, spec=a.spec)))
    elif a.cmd == 'gumbel':
        print(json.dumps(gumbel(a.game, n=a.positions, sims=a.sims, reps=a.reps, m=a.m, scale=a.scale, eps=a.eps,
                                cq=(50.0, a.cq_c_scale), verbose=True)))
        print(json.dumps(gumbel_formula(a.game)))
    elif a.cmd == 'forced':
        print(json.dumps(forced(a.game, n=a.positions, sims=a.sims, reps=a.reps, k=a.k, eps=a.eps,
                                cq=(50.0, a.cq_c_scale), verbose=True)))
    else:
        h2h(a.model, a.games, a.sims, a.chance_k, seed=a.seed, game=a.game, select=a.select,
            select_min_visits=a.select_min_visits, select_c_visit=a.select_c_visit, select_c_scale=a.select_c_scale,
            spec=a.spec, device=a.device)


if __name__ == '__main__':
    main()
