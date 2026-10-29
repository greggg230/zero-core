"""The Gumbel root search (docs/search.md 3.7, docs/az.md 19): run settings selfplay_root_search /
eval_root_search / gumbel_m / selfplay_gumbel_scale / eval_gumbel_scale, seat keys root_search /
gumbel_m / gumbel_scale / gumbel_c_visit / gumbel_c_scale, sequential halving's schedule (= mctx's =
the Rust tree's), the Python reference tree and driver against the Rust ones (exactly on the
formulas, statistically in searches), the Elo pool's name, zero.h2h's per-side root search, toy runs:
off = the runs of before byte for byte; self-play only = the evaluations' pool unchanged; evaluation
only = the training byte for byte, another pool."""
import json
import os
import sys
import types

import numpy as np
import pytest

from zero import elo, games as G, h2h, mcts, parity, replay
from zero.play import NNSeat, PyDriver
from zero.run import DEFAULTS, Run, check_eval_settings, read_metrics

from test_select import _play, _Rand
from test_select_scale import _runs, _training_files
from test_root_min_visits import _cat, _Skew

INFO = G.info('toy')
DASH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dashboard')
GKEYS = ('root_search', 'gumbel_m', 'gumbel_scale', 'gumbel_c_visit', 'gumbel_c_scale')


# ------------------------------------------------------------------ settings, seats, names

def test_settings_seats_and_names(tmp_path):
    assert (DEFAULTS['selfplay_root_search'], DEFAULTS['eval_root_search'], DEFAULTS['gumbel_m'],
            DEFAULTS['selfplay_gumbel_scale'], DEFAULTS['eval_gumbel_scale']) == ('puct', 'puct', 16, 1.0, 0.0)
    # off: the seat dict of before, key for key (the Gumbel arguments unused); on: the resolved keys
    assert G.nn_seat('m', 8) == G.nn_seat('m', 8, root_search='puct', gumbel_m=4, gumbel_scale=0.5)
    assert not any(k in G.nn_seat('m', 8) for k in GKEYS)
    sp = G.nn_seat('m', 64, selfplay=True, policy_target='cq', cq_c_scale=0.3, root_search='gumbel')
    assert {k: sp[k] for k in GKEYS} == dict(root_search='gumbel', gumbel_m=16, gumbel_scale=1.0, gumbel_c_visit=50.0,
                                             gumbel_c_scale=0.3)
    # (self-play: sigma is the 'cq' target's even with another self-play rule transform)
    sp2 = G.nn_seat('m', 64, selfplay=True, policy_target='cq', cq_c_scale=0.3, select='cq', select_c_scale=1.0,
                    root_search='gumbel')
    assert sp2['gumbel_c_scale'] == 0.3
    ev = G.nn_seat('m', 32, select='cq', select_min_visits=4, select_c_scale=0.3, cq_c_scale=1.0, root_search='gumbel',
                   gumbel_scale=0.0)
    assert (ev['gumbel_c_visit'], ev['gumbel_c_scale'], ev['gumbel_scale']) == (50.0, 0.3, 0.0)
    assert G.nn_seat('m', 32, root_search='gumbel', gumbel_c_scale=2.0)['gumbel_c_scale'] == 2.0
    for bad in (dict(root_search='gumble'), dict(root_search='gumbel', gumbel_m=0), dict(root_search='gumbel', gumbel_m=2.5),
                dict(root_search='gumbel', gumbel_scale=-1.0), dict(root_search='gumbel', gumbel_c_scale=float('nan')),
                dict(root_search='gumbel', root_min_visits=4)):
        with pytest.raises(ValueError):
            G.nn_seat('m', 8, **bad)
        with pytest.raises(ValueError):
            NNSeat('m', **bad)
    # the Rust driver takes the keys, refuses Gumbel keys on a PUCT seat, a 'visits' target it would record
    G.Driver('toy', [[sp, sp]], concurrency=1, record=True)
    G.Driver('toy', [[ev, ev]], concurrency=1)
    with pytest.raises(ValueError, match="root_search 'gumbel'"):
        G.Driver('toy', [[dict(G.nn_seat('m', 8), gumbel_m=4)] * 2], concurrency=1)
    vis = G.nn_seat('m', 64, selfplay=True, root_search='gumbel')
    with pytest.raises(ValueError, match="policy_target 'cq'"):
        G.Driver('toy', [[vis, vis]], concurrency=1, record=True)
    G.Driver('toy', [[vis, vis]], concurrency=1, record=False)
    with pytest.raises(ValueError, match='exclusive'):
        G.Driver('toy', [[dict(sp, root_min_visits=4)] * 2], concurrency=1)
    # run settings: validated (resolver_cfg too: the same check)
    for bad in ({'selfplay_root_search': 'gumble'}, {'eval_root_search': 'gumbel', 'gumbel_m': 0},
                {'eval_root_search': 'gumbel', 'eval_gumbel_scale': -1.0},
                {'selfplay_root_search': 'gumbel'},                                  # policy_target 'visits'
                {'selfplay_root_search': 'gumbel', 'policy_target': 'cq', 'selfplay_root_min_visits': 4},
                {'eval_root_search': 'gumbel', 'eval_root_min_visits': 4}):
        with pytest.raises(SystemExit):
            check_eval_settings(dict(DEFAULTS, **bad))
    check_eval_settings(dict(DEFAULTS, selfplay_root_search='gumbel', policy_target='cq', eval_root_search='gumbel',
                             eval_root_min_visits=0, selfplay_root_min_visits=0))
    check_eval_settings(dict(DEFAULTS, eval_root_search='gumbel', selfplay_root_min_visits=4))   # other sides
    # Run.nn_seat: self-play's sigma the target's, the evaluation's the rule's transform
    cfg = dict(DEFAULTS, selfplay_root_search='gumbel', policy_target='cq', cq_c_scale=0.3, eval_select='cq',
               eval_select_min_visits=4, eval_select_c_scale=0.5)
    r = types.SimpleNamespace(cfg=cfg)
    s = Run.nn_seat(r, 'm', 64, selfplay=True)
    assert (s['root_search'], s['gumbel_scale'], s['gumbel_c_scale']) == ('gumbel', 1.0, 0.3)
    assert 'root_search' not in Run.nn_seat(r, 'm', 32)
    r.cfg = dict(cfg, selfplay_root_search='puct', eval_root_search='gumbel', gumbel_m=4)
    assert 'root_search' not in Run.nn_seat(r, 'm', 64, selfplay=True)
    e = Run.nn_seat(r, 'm', 32)
    assert (e['root_search'], e['gumbel_m'], e['gumbel_scale'], e['gumbel_c_scale']) == ('gumbel', 4, 0.0, 0.5)
    # names: the evaluation's root search names its Elo pool; self-play's never does
    assert mcts.gumbel_tag() == 'gumbel-m16' and mcts.gumbel_tag(4, 50, 0.3, 1.0) == 'gumbel-m4-v50-s0.3-g1'
    cq = {'eval_select': 'cq', 'eval_select_min_visits': 4, 'eval_select_c_scale': 0.3}
    cases = [({}, 'visits'), ({'selfplay_root_search': 'gumbel', 'policy_target': 'cq'}, 'visits'),
             ({'eval_root_search': 'gumbel'}, 'gumbel-m16'), (dict(cq, eval_root_search='gumbel'), 'gumbel-m16-v50-s0.3'),
             (dict(cq, eval_root_search='gumbel', gumbel_m=4, eval_gumbel_scale=1.0), 'gumbel-m4-v50-s0.3-g1'),
             (dict(cq, eval_root_search='gumbel', eval_leaf_value_view='root'), 'gumbel-m16-v50-s0.3-rv'), (cq, 'cq-n4-v50-s0.3')]
    sys.path.insert(0, DASH)
    from zerodash import runfiles
    for k, (c, tag) in enumerate(cases):
        assert elo.eval_tag(c) == tag, c
        d = tmp_path / f'run{k}'
        os.makedirs(d / 'models')
        json.dump(c, open(d / 'config.json', 'w'))
        assert elo.active_pool(str(d)) == (tag, elo.pool_dir(str(d), tag))
        assert runfiles.elo_pool(str(d)) == elo.pool_dir(str(d), tag), c
        # h2h follows a run's evaluation root search (and names it), also with the rule given on the
        # command line (root_keys); --a-root-search overrides it
        ck = str(d / 'models' / 'iter_0000.pt')
        rule = h2h.run_rule(ck)
        assert rule.get('root_search', 'puct') == c.get('eval_root_search', 'puct')
        assert h2h._tag(dict(G.nn_seat('a', 8), **rule)) == tag
        given = h2h.root_keys(ck, h2h.rule_keys('cq', 4, 50.0, 0.3))
        assert given.get('root_search', 'puct') == c.get('eval_root_search', 'puct')
        if given.get('root_search') == 'gumbel':
            assert given['gumbel_c_scale'] == 0.3, 'a gumbel side takes its rule transform'
        assert 'root_search' not in h2h.root_keys(ck, rule, 'puct')
        g = h2h.root_keys(ck, {}, 'gumbel', 8, 0.5)
        assert (g['gumbel_m'], g['gumbel_scale']) == (8, 0.5)
        # zero.replay's seats: the run's root search, sigma as Run.nn_seat
        seat = replay._nn('m', 16, dict(DEFAULTS, **c), False, 0)
        assert seat.root_search == c.get('eval_root_search', 'puct')
    rs = replay._nn('m', 16, dict(DEFAULTS, selfplay_root_search='gumbel', policy_target='cq', cq_c_scale=0.3), True, 0)
    assert (rs.root_search, rs.gumbel_scale, rs.gumbel_c_scale) == ('gumbel', 1.0, 0.3)


def test_sequence_and_noise_scale_match_the_rust_tree():
    E = G.engine()
    for m in range(0, 20):
        for n in list(range(0, 70)) + [100, 127, 128, 200]:
            assert mcts.gumbel_sequence(m, n) == list(E.gumbel_sequence(m, n)), (m, n)
    assert mcts.gumbel_sequence(4, 8) == [0, 0, 0, 0, 1, 1, 2, 2]
    assert mcts.gumbel_noise_scale(True, 3, 4, 1.0) == 1.0 and mcts.gumbel_noise_scale(True, 4, 4, 1.0) == 0.0
    assert mcts.gumbel_noise_scale(False, 99, 0, 0.5) == 0.5


# ------------------------------------------------------------------ the trees

def test_python_formulas_on_rust_roots_are_exact():
    """`gumbel_choose`, `gumbel_q` and the next root walk of the Python reference on the statistics of
    Rust Gumbel roots (m 1, 2, 4, 16; 2-64 simulations) are the Rust tree's own, exactly."""
    r = parity.gumbel_formula()
    assert r['checked'] == 256 and r['agree'] + r['near_ties'] == r['checked'] and not r['mismatches'], r
    assert r['q_max_diff'] < 1e-12, r


def test_rust_and_python_trees_search_alike():
    """Both trees with the same Gumbel noise per repetition: the visit distributions, the searches'
    own walks, the completed-Q targets, the moves and the root values at the Python-Python floor; a
    Rust m of 2 against Python's 16 is seen (power)."""
    r = parity.gumbel(n=6, reps=24, m=16)
    assert r['budget_ok'], r
    assert r['tv_rs_py'] <= 2 * r['tv_py_py'] + 0.005, r
    assert r['tv_cq_rs_py'] <= 2 * r['tv_cq_py_py'] + 0.005, r
    assert r['tv_move_rs_py'] <= 2 * r['tv_move_py_py'] + 0.05, r
    assert r['q_diff'] < 0.005, r
    p = parity.gumbel(n=6, reps=8, m=16, m_rs=2)
    assert p['tv_rs_py'] > 5 * r['tv_py_py'] + 0.02, p
    # the lopsided prior of the Delver mulligan: still at the floor (every considered option walked)
    s = parity.gumbel(n=6, reps=16, m=16, eps=0.001)
    assert s['budget_ok'] and s['tv_rs_py'] <= 2 * s['tv_py_py'] + 0.005, s


def test_python_tree_halves_and_budgets_a_reused_root():
    ev = parity.LinearEval(INFO['num_features'], INFO['num_actions'], seed=1, scale=0.15)
    for g in parity.sample_positions('toy', 6, seed=4):
        k = len(g.legal())
        nv, _, t = parity.search_py(g, ev, 64, 3, tree=True, gumbel=(4, [], 50.0, 1.0))
        c = min(4, k)
        cnt = [0] * c
        for v in mcts.gumbel_sequence(c, 64)[:63]:          # the schedule of 64 simulations, 63 walks
            cnt[cnt.index(v)] += 1
        assert sorted(nv, reverse=True)[:c] == sorted(cnt, reverse=True) and nv.sum() == 63
        assert list(t.gumbel.n) == list(nv) and t.gumbel.walks == 63
        # set_gumbel again on the same (expanded) root: 20 more walks, the old visits kept
        before = list(t.root.Nv)
        t.set_gumbel(16, 20, None, 50.0, 1.0)
        import random
        rng = random.Random(1)
        for _ in range(20):
            res = mcts.simulate(t, rng, 1.0)
            if res is not None:
                node, leaf, path = res
                pri, v, m = ev(leaf.encode(node.player), node.acts, g.n_players)
                mcts.expand_and_backup(node, path, pri, v, m, g.n_players, tree=t)
        assert sum(t.gumbel.n) == 20 and [a - b for a, b in zip(t.root.Nv, before)] == t.gumbel.n
        assert t.root_min == 0


# ------------------------------------------------------------------ the drivers

def _stats(samples, ev):
    """Mean target mass off each record's prior favourite, its standard error, and the records."""
    out = []
    for s in samples:
        off = s['opt_off']
        for b in range(len(off) - 1):
            acts = s['opt_action'][off[b]:off[b + 1]]
            if not len(acts):
                continue
            pri, _, _ = ev.ev(s['x'][b], acts, 2)
            out.append(1.0 - float(s['pi'][off[b] + int(np.argmax(pri))]))
    return float(np.mean(out)), float(np.std(out) / np.sqrt(len(out))), len(out)


@pytest.mark.parametrize('impl', ['py', 'rs', 'rust-driver'])
def test_drivers_search_and_record_alike(impl):
    """Self-play with the Gumbel root (sigma (50, 1), m 16, 24 simulations, a lopsided prior where PUCT
    starves every option but one): the Python driver's seats ('py' tree, 'rs' tree) and the Rust
    driver record valid 'cq' targets that move off the prior's favourite (PUCT's barely do), alike; the
    games differ from PUCT's."""
    ev = _Skew(3)

    def play(rs):
        if impl == 'rust-driver':
            seat = G.nn_seat('m', 24, selfplay=True, eps=0.0, sims_fast=None, p_full=1.0, policy_target='cq',
                             root_search=rs)
            drv = G.Driver('toy', [[seat, seat]] * 32, concurrency=32, seed=5, record=True, threads=2)
            parts = []
            while True:
                lb = drv.next_batch()
                if lb is None:
                    break
                drv.submit(lb.id, *ev(lb))
                parts.append(drv.take_samples())
            parts.append(drv.take_samples())
            return [r['scores'] for r in sorted(drv.take_results(), key=lambda r: r['gid'])], _cat(parts)
        seat = NNSeat('m', 24, selfplay=True, eps=0.0, impl=impl, policy_target='cq', root_search=rs)
        drv = PyDriver('toy', 32, lambda g: [seat, seat], concurrency=32, seed=5, record=True)
        rows = _play(drv, ev)
        return [r['scores'] for r in sorted(rows, key=lambda r: r['gid'])], [drv.take_samples()]
    gp, sp = play('puct')
    gg, sg = play('gumbel')
    assert gp != gg
    mp, _, n_p = _stats(sp, ev)
    mg, se, n_g = _stats(sg, ev)
    for s in sg:
        off = s['opt_off']
        assert all(abs(s['pi'][off[b]:off[b + 1]].sum() - 1) < 1e-5 for b in range(len(off) - 1))
        assert np.allclose(s['q'][:, 0] + s['q'][:, 1], 1.0, atol=1e-4)
    print(f'{impl}: off-favourite cq target mass puct {mp:.4f} ({n_p}), gumbel {mg:.4f} ({n_g})')
    assert n_g > 300 and mg > mp + 0.02, (mp, mg)
    # (the three drivers' Gumbel figures agree within their noise - the Rust driver seeds its games
    # otherwise than the Python driver: checked across the parametrization)
    test_drivers_search_and_record_alike.seen[impl] = (mg, se)


test_drivers_search_and_record_alike.seen = {}


def test_drivers_agree():
    seen = test_drivers_search_and_record_alike.seen
    if len(seen) < 3:
        pytest.skip('needs the three driver runs')
    v = list(seen.values())
    for a, sa in v:
        for b, sb in v:
            assert abs(a - b) < 3.5 * (sa * sa + sb * sb) ** 0.5 + 0.005, seen


def test_python_driver_refuses_a_visits_target():
    seat = NNSeat('m', 8, selfplay=True, root_search='gumbel')
    drv = PyDriver('toy', 2, lambda g: [seat, seat], concurrency=2, seed=0, record=True)
    with pytest.raises(ValueError, match="policy_target 'cq'"):
        _play(drv, _Rand())


def test_python_driver_noise_and_temperature():
    """A self-play Gumbel seat with temp_moves 0 plays an evaluation seat's games without noise (the
    'py' and 'rs' seats alike); within temp_moves it samples."""
    def games(seat):
        drv = PyDriver('toy', 16, lambda g: [seat, seat], concurrency=16, seed=8)
        return sorted(((r['gid'], r['steps'], tuple(r['scores'])) for r in _play(drv, _Rand(2))))
    for impl in ('py', 'rs'):
        ev0 = games(NNSeat('m', 12, impl=impl, root_search='gumbel', gumbel_scale=0.0))
        assert games(NNSeat('m', 12, impl=impl, selfplay=True, temp_moves=0, root_search='gumbel')) == ev0
        assert games(NNSeat('m', 12, impl=impl, selfplay=True, temp_moves=100, root_search='gumbel')) != ev0


# ------------------------------------------------------------------ toy runs

def test_runs_off_selfplay_and_eval(tmp_path):
    """Toy-smoke runs (2 iterations, 'cq' targets):
    * the settings given at their defaults: every shard, checkpoint and Elo file of the run without
      them, byte for byte;
    * self-play's Gumbel search: other shards, the evaluations' pool unchanged (elo/);
    * the evaluation's: every shard and checkpoint byte for byte (training untouched), the evaluations
      in the pool elo/gumbel-m16/ (zerodash's too); zero.h2h follows each run's root search."""
    r = _runs(tmp_path, {
        'base': [],
        'zero': ['--selfplay-root-search', 'puct', '--eval-root-search', 'puct', '--gumbel-m', '8',
                 '--selfplay-gumbel-scale', '0.5', '--eval-gumbel-scale', '1'],
        'sp': ['--selfplay-root-search', 'gumbel'],
        'ev': ['--eval-root-search', 'gumbel'],
    })
    a, b = _training_files(r['base']), _training_files(r['zero'])
    assert len(a) >= 6 and a == b
    for f in ('results.json', 'ratings.json'):
        ja, jb = (json.load(open(r[k] / 'elo' / f)) for k in ('base', 'zero'))
        ja.pop('updated', None), jb.pop('updated', None)
        assert ja == jb, f
    sp = _training_files(r['sp'])
    assert sp['models/iter_0000.pt'] == a['models/iter_0000.pt'] and sp != a
    assert elo.active_pool(str(r['sp']))[0] == 'visits' and os.path.exists(r['sp'] / 'elo' / 'ratings.json')
    assert _training_files(r['ev']) == a
    assert elo.active_pool(str(r['ev']))[0] == 'gumbel-m16'
    assert os.path.isdir(r['ev'] / 'elo' / 'gumbel-m16') and not os.path.exists(r['ev'] / 'elo' / 'ratings.json')
    assert {m['eval_yard']['select'] for m in read_metrics(str(r['ev']))} == {'gumbel-m16'}
    sys.path.insert(0, DASH)
    from zerodash import runfiles
    for k in r:
        assert runfiles.elo_pool(str(r[k])) == elo.active_pool(str(r[k]))[1], k
    # zero.h2h: the Gumbel-evaluated run's side searches with it, the other with PUCT
    m = h2h.match(str(r['ev'] / 'models' / 'iter_0001.pt'), str(r['base'] / 'models' / 'iter_0001.pt'), games=20,
                  sims=8, chance_k=0)
    assert (m['select_a'], m['select_b'], m['games']) == ('gumbel-m16', 'visits', 20)
    # zero.replay: the runs' seats search as their runs (a self-play replay of the Gumbel self-play run
    # samples its moves - its noise - within temp_moves; an evaluation replay of the Gumbel-evaluated run
    # does not), and the frames' root value is the search's
    log = lambda *x: None                                                   # noqa: E731
    sp = replay.record(str(r['sp'] / 'models' / 'iter_0001.pt'), opp='self', games=1, sims=8, chance_k=0, seed=1,
                       selfplay=True, log=log)
    ev = replay.record(str(r['ev'] / 'models' / 'iter_0001.pt'), opp='random', games=1, sims=8, chance_k=0, seed=1,
                       log=log)
    f_sp = [f for x in sp for f in x['frames'] if f['k'] == 'nn']
    f_ev = [f for x in ev for f in x['frames'] if f['k'] == 'nn']
    assert f_sp and f_ev and any(f['sampled'] for f in f_sp) and not any(f['sampled'] for f in f_ev)
    assert all(f['w'] is not None and abs(sum(f['w']) - 1) < 1e-4 for f in f_ev)
    assert ev[0]['seats'][ev[0]['seats'][0]['kind'] != 'nn']['select'] == 'gumbel-m16'


@pytest.mark.parametrize('driver', ['rust', 'py', 'rs'])
def test_gumbel_with_a_root_belief(driver):
    """The Gumbel root and the root's belief together (docs/search.md 3.7, 3.8) with a network with the
    auxiliary head: self-play with fast searches and the belief on every search (belief_fast, so reused
    roots wait for their belief row inside a Gumbel search) plays every game to its end with valid 'cq'
    targets, one belief row per search, and other targets than the Gumbel search without the belief;
    the Rust driver, the Python driver's 'py' and 'rs' seats. Its evaluation's name carries both."""
    from zero.infer import NetEvaluator, drive
    from test_aux import TOY_AUX, nets
    ev = NetEvaluator(nets(TOY_AUX)['mlp'])

    def play(tilt):
        kw = dict(selfplay=True, sims_fast=4, p_full=0.5, policy_target='cq', root_search='gumbel')
        if tilt:
            kw.update(belief_tilt=tilt, belief_fast=True)
        if driver == 'rust':
            seat = G.nn_seat('m', 12, **kw)
            drv = G.Driver('toy', [[seat, seat]] * 16, concurrency=16, seed=3, record=True, threads=2)
        else:
            seat = NNSeat('m', sims=12, impl=driver, **kw)
            drv = PyDriver('toy', 16, lambda gid: [seat, seat], concurrency=16, seed=3, record=True)
        drive(drv, {'m': ev})
        rows = drv.stats()['belief_rows'] if driver == 'rust' else drv.belief_rows
        return drv.take_results(), drv.take_samples(), rows
    res, smp, rows = play(1.0)
    assert len(res) == 16 and all(r['over'] for r in res)
    off = smp['opt_off']
    assert all(abs(smp['pi'][off[b]:off[b + 1]].sum() - 1) < 1e-4 for b in range(len(off) - 1))
    assert np.allclose(smp['q'][:, 0] + smp['q'][:, 1], 1.0, atol=1e-4)
    assert rows >= len(smp['player']) > 0, (rows, len(smp['player']))
    _, smp0, rows0 = play(0.0)
    assert rows0 == 0
    assert len(smp0['pi']) != len(smp['pi']) or not np.array_equal(smp0['pi'], smp['pi'])
    c = dict(DEFAULTS, eval_root_search='gumbel', eval_belief_tilt=1.0, eval_select='cq', eval_select_c_scale=0.3)
    assert elo.eval_tag(c) == 'gumbel-m16-v50-s0.3-b1'
