"""Root minimum visits ("forced playouts", docs/search.md 3.5, docs/az.md 5.5): run settings
selfplay_root_min_visits / eval_root_min_visits / root_min_visits_max_actions, seat keys
root_min_visits / root_min_visits_max_actions, `mcts.root_floor` (= the Rust driver's), the Python
reference tree and driver against the Rust ones, the Elo pool's name, toy runs: off = the runs of
before byte for byte; self-play only = the evaluations' pool unchanged; evaluation only = the
training byte for byte, another pool."""
import json
import os
import sys
import types

import numpy as np
import pytest

from zero import elo, games as G, h2h, mcts, parity
from zero.play import NNSeat, PyDriver
from zero.run import DEFAULTS, Run, check_eval_settings, read_metrics

from test_select import _play, _Rand
from test_select_scale import _runs, _training_files

INFO = G.info('toy')
DASH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dashboard')


# ------------------------------------------------------------------ settings, seats, names

def test_settings_seats_and_names(tmp_path):
    assert (DEFAULTS['selfplay_root_min_visits'], DEFAULTS['eval_root_min_visits'],
            DEFAULTS['root_min_visits_max_actions']) == (0, 0, 4)
    # off: the seat dict of before, key for key; on: both keys
    assert G.nn_seat('m', 8) == G.nn_seat('m', 8, root_min_visits=0, root_min_visits_max_actions=6)
    assert 'root_min_visits' not in G.nn_seat('m', 8)
    s = G.nn_seat('m', 8, root_min_visits=4)
    assert (s['root_min_visits'], s['root_min_visits_max_actions']) == (4, 4)
    for bad in (dict(root_min_visits=-1), dict(root_min_visits=1.5), dict(root_min_visits=True),
                dict(root_min_visits=2, root_min_visits_max_actions=1)):
        with pytest.raises(ValueError):
            G.nn_seat('m', 8, **bad)
        with pytest.raises(ValueError):
            NNSeat('m', **bad)
    # the Rust driver takes the keys, and refuses a floor no root can take
    G.Driver('toy', [[s, s]], concurrency=1)
    with pytest.raises(ValueError, match='root_min_visits_max_actions'):
        G.Driver('toy', [[dict(s, root_min_visits_max_actions=1)] * 2], concurrency=1)
    # run settings: validated (resolver_cfg too: the same check)
    for bad in ({'selfplay_root_min_visits': -2}, {'eval_root_min_visits': 0.5},
                {'selfplay_root_min_visits': 4, 'root_min_visits_max_actions': 1}):
        with pytest.raises(SystemExit):
            check_eval_settings(dict(DEFAULTS, **bad))
    check_eval_settings(dict(DEFAULTS, selfplay_root_min_visits=4, eval_root_min_visits=8, root_min_visits_max_actions=6))
    # Run.nn_seat: self-play's floor on the self-play seat, the evaluation's on every evaluation seat
    cfg = dict(DEFAULTS, selfplay_root_min_visits=4)
    r = types.SimpleNamespace(cfg=cfg)
    assert Run.nn_seat(r, 'm', 64, selfplay=True)['root_min_visits'] == 4
    assert 'root_min_visits' not in Run.nn_seat(r, 'm', 32)
    r.cfg = dict(DEFAULTS, eval_root_min_visits=8, root_min_visits_max_actions=6)
    assert 'root_min_visits' not in Run.nn_seat(r, 'm', 64, selfplay=True)
    assert (Run.nn_seat(r, 'm', 32)['root_min_visits'], Run.nn_seat(r, 'm', 32)['root_min_visits_max_actions']) == (8, 6)
    # names: the evaluation's floor names its Elo pool; self-play's never does
    assert mcts.forced_tag(0) == '' and mcts.forced_tag(4) == '-f4' and mcts.forced_tag(4, 6) == '-f4a6'
    cq = {'eval_select': 'cq', 'eval_select_min_visits': 4, 'eval_select_c_scale': 0.3}
    cases = [({}, 'visits'), ({'selfplay_root_min_visits': 4}, 'visits'), ({'eval_root_min_visits': 4}, 'visits-f4'),
             ({'eval_root_min_visits': 4, 'root_min_visits_max_actions': 6}, 'visits-f4a6'),
             ({'selfplay_root_min_visits': 4, 'root_min_visits_max_actions': 6}, 'visits'),
             (cq, 'cq-n4-v50-s0.3'), (dict(cq, eval_root_min_visits=4), 'cq-n4-v50-s0.3-f4')]
    sys.path.insert(0, DASH)
    from zerodash import runfiles
    for k, (c, tag) in enumerate(cases):
        assert elo.eval_tag(c) == tag, c
        d = tmp_path / f'run{k}'
        os.makedirs(d / 'models')
        json.dump(c, open(d / 'config.json', 'w'))
        assert elo.active_pool(str(d)) == (tag, elo.pool_dir(str(d), tag))
        assert runfiles.elo_pool(str(d)) == elo.pool_dir(str(d), tag), c
        # h2h follows a run's evaluation floor (and names it)
        rule = h2h.run_rule(str(d / 'models' / 'iter_0000.pt'))
        assert rule.get('root_min_visits', 0) == c.get('eval_root_min_visits', 0)
        assert h2h._tag(dict(G.nn_seat('a', 8), **rule)) == tag


def test_root_floor_matches_the_rust_driver():
    E = G.engine()
    for k in (0, 1, 2, 4, 8):
        for mx in (2, 4, 6):
            for full in (False, True):
                for m in (1, 2, 3, 4, 5, 7):
                    for target in (1, 4, 8, 15, 16, 32, 64):
                        assert mcts.root_floor(k, mx, full, m, target) == E.root_floor(k, mx, full, m, target), \
                            (k, mx, full, m, target)
    assert mcts.root_floor(4, 4, True, 2, 16) == 4 and mcts.root_floor(4, 4, True, 2, 15) == 0


# ------------------------------------------------------------------ the trees

def test_rust_and_python_trees_force_alike():
    """Both trees with the floor 4 on toy roots of 2-6 options under a lopsided prior (0.999): every
    option reaches 4 visits, the same forced walks, and the visit distributions with and without
    them, the completed-Q targets and the root win without them at the Python-Python noise floor;
    a Rust floor of 5 against Python's 4 is seen (power)."""
    r = parity.forced(n=6, reps=32, k=4)
    assert r['floor_met'] and r['forced_rs'] == r['forced_py'] > 0
    assert r['starved_k0'] > 0.3, r                    # what the floor is for
    assert r['tv_rs_py'] <= 3 * r['tv_py_py'] + 0.002, r
    assert r['tv_pruned_rs_py'] <= 3 * r['tv_pruned_py_py'] + 0.002, r
    assert r['tv_cq_rs_py'] <= 2 * r['tv_cq_py_py'] + 0.01, r
    assert r['win_diff'] < 0.01, r
    p = parity.forced(n=6, reps=8, k=4, k_rs=5)
    assert p['forced_rs'] > p['forced_py'] and p['tv_rs_py'] > 10 * r['tv_py_py'] + 0.005, p


def test_python_trees_count_their_forced_walks():
    """The Python reference's accounting: forced walks only below the floor, at most k per option,
    N - forced = the pruned counts, and the root win without them; advance (PySearch) clears it."""
    ev = parity.SkewEval(parity.LinearEval(INFO['num_features'], INFO['num_actions'], seed=1, scale=0.15))
    for g in parity.sample_positions('toy', 6, seed=3):
        if 2 * len(g.legal()) * 3 > 64:
            continue
        nv, _, t = parity.search_py(g, ev, 64, 9, tree=True, root_min=3)
        assert nv.min() >= 3 and all(f <= 3 for f in t.forced) and sum(t.forced) == t.forced_n > 0
        assert mcts.root_visits_pruned(t) == [n - f for n, f in zip(t.root.Nv, t.forced)]
        w = mcts.root_win_pruned(t)
        tot = t.root.tot - t.forced_n
        assert np.allclose([x * tot + f for x, f in zip(w, t.forced_win)], t.root.wsum)
        t.set_root_min(0)
        assert (t.forced, t.forced_n, mcts.root_visits_pruned(t)) == ([], 0, list(t.root.Nv))


# ------------------------------------------------------------------ the drivers

class _Skew(_Rand):
    """`_Rand` with the lopsided prior of `parity.SkewEval` (0.999 on its favourite option)."""

    def __init__(self, seed=0):
        super().__init__(seed)
        self.ev = parity.SkewEval(self.ev)


def _favourite_mass(samples, ev):
    """Mean policy-target mass off each record's prior favourite."""
    out = []
    for s in samples:
        off = s['opt_off']
        for b in range(len(off) - 1):
            acts = s['opt_action'][off[b]:off[b + 1]]
            pri, _, _ = ev.ev(s['x'][b], acts, 2)
            out.append(1.0 - float(s['pi'][off[b] + int(np.argmax(pri))]))
    return float(np.mean(out)), len(out)


def _cat(parts):
    from zero.protocol import concat_samples
    return [concat_samples([p for p in parts if p is not None])]


@pytest.mark.parametrize('impl', ['py', 'rs', 'rust-driver'])
def test_drivers_prune_the_forced_walks_from_the_visits_targets(impl):
    """Self-play with 'visits' targets at a lopsided prior, 24 simulations, the floor 2 on roots of up
    to 12 options: the targets' mass off the prior's favourite stays the floor-off run's (the forced
    walks are pruned; kept, they would be 2 (m - 1) / 23 of it). The Python driver's seats ('py' tree,
    'rs' tree) and the Rust driver alike."""
    ev = _Skew(3)

    def play(k):
        if impl == 'rust-driver':
            seat = G.nn_seat('m', 24, selfplay=True, eps=0.0, sims_fast=None, p_full=1.0, root_min_visits=k,
                             root_min_visits_max_actions=12)
            drv = G.Driver('toy', [[seat, seat]] * 24, concurrency=24, seed=5, record=True, threads=2)
            parts = []
            while True:
                lb = drv.next_batch()
                if lb is None:
                    break
                drv.submit(lb.id, *ev(lb))
                parts.append(drv.take_samples())
            parts.append(drv.take_samples())
            return [r['scores'] for r in sorted(drv.take_results(), key=lambda r: r['gid'])], _cat(parts)
        seat = NNSeat('m', 24, selfplay=True, eps=0.0, impl=impl, root_min_visits=k, root_min_visits_max_actions=12)
        drv = PyDriver('toy', 24, lambda g: [seat, seat], concurrency=24, seed=5, record=True)
        rows = _play(drv, ev)
        return [r['scores'] for r in sorted(rows, key=lambda r: r['gid'])], [drv.take_samples()]
    g0, s0 = play(0)
    g2, s2 = play(2)
    # (the moves are sampled from the pruned counts, which at this prior are mostly the floor-off
    # run's: the games may well be the same; the root values q are not - the forced walks ran)
    q0 = np.concatenate([s['q'][:, 0] for s in s0])
    q2 = np.concatenate([s['q'][:, 0] for s in s2])
    assert g0 != g2 or not np.array_equal(q0, q2), 'the floor changed nothing'
    m0, n0 = _favourite_mass(s0, ev)
    m2, n2 = _favourite_mass(s2, ev)
    assert n0 > 300 and n2 > 300
    for s in s2:
        off = s['opt_off']
        assert all(abs(s['pi'][off[b]:off[b + 1]].sum() - 1) < 1e-5 for b in range(len(off) - 1))
    print(f'{impl}: off-favourite target mass k 0 {m0:.4f} ({n0} records), k 2 {m2:.4f} ({n2})')
    assert m2 < m0 + 0.02, f'{impl}: the targets hold forced visits ({m2:.4f} vs {m0:.4f} without the floor)'


# ------------------------------------------------------------------ toy runs

def test_runs_off_selfplay_and_eval(tmp_path):
    """Toy-smoke runs (2 iterations, 'cq' targets):
    * the settings given at their defaults: every shard, checkpoint and Elo file of the run without
      them, byte for byte;
    * self-play's floor: other shards (the searches changed), the evaluations' pool unchanged (elo/);
    * the evaluation's floor: every shard and checkpoint byte for byte (training untouched), the
      evaluations in the pool elo/visits-f1a6/ (zerodash's too)."""
    r = _runs(tmp_path, {
        'base': [],
        'zero': ['--selfplay-root-min-visits', '0', '--eval-root-min-visits', '0', '--root-min-visits-max-actions', '4'],
        'sp': ['--selfplay-root-min-visits', '1', '--root-min-visits-max-actions', '6'],
        'ev': ['--eval-root-min-visits', '1', '--root-min-visits-max-actions', '6'],
    })
    a, b = _training_files(r['base']), _training_files(r['zero'])
    assert len(a) >= 6 and a == b
    for f in ('results.json', 'ratings.json'):
        ja, jb = (json.load(open(r[k] / 'elo' / f)) for k in ('base', 'zero'))
        ja.pop('updated', None), jb.pop('updated', None)
        assert ja == jb, f
    # self-play's floor: the data changes, the pool does not
    sp = _training_files(r['sp'])
    assert sp['models/iter_0000.pt'] == a['models/iter_0000.pt'] and sp != a
    assert elo.active_pool(str(r['sp']))[0] == 'visits' and os.path.exists(r['sp'] / 'elo' / 'ratings.json')
    assert {m['eval_yard'].get('select', 'visits') for m in read_metrics(str(r['sp']))} == {'visits'}
    # the evaluation's floor: the training of before, its own pool
    assert _training_files(r['ev']) == a
    assert elo.active_pool(str(r['ev']))[0] == 'visits-f1a6'
    assert os.path.isdir(r['ev'] / 'elo' / 'visits-f1a6') and not os.path.exists(r['ev'] / 'elo' / 'ratings.json')
    assert {m['eval_yard']['select'] for m in read_metrics(str(r['ev']))} == {'visits-f1a6'}
    sys.path.insert(0, DASH)
    from zerodash import runfiles
    for k in r:
        assert runfiles.elo_pool(str(r[k])) == elo.active_pool(str(r[k]))[1], k
    cfg = json.load(open(r['sp'] / 'config.json'))
    assert (cfg['selfplay_root_min_visits'], cfg['eval_root_min_visits'], cfg['root_min_visits_max_actions']) == (1, 0, 6)
