"""Deferred results and teams (docs/search.md 1.1, 2.8-2.9, 3.2): the toy team game `toy_team`
through the bindings, the search's deferred leaves and team utilities (Rust vs the Python
reference), the driver's parking, `resolve`, queued setups, `queue_only` and tags."""
import json
import random

import numpy as np
import pytest

from zero import _engine as E
from zero import games as G
from zero import mcts, package, parity, protocol
from zero.infer import NetEvaluator, UniformEvaluator, drive, segment_softmax_np
from zero.play import NNSeat, PyDriver

INFO = E.game_info_of('toy_team')


def play_to_end(seed, players=4, spec=None):
    g = E.Game('toy_team', spec, players, seed)
    r = random.Random(seed)
    while not g.over:
        g.step(r.choice(g.legal()))
    return g


def test_toy_team_through_the_bindings():
    assert INFO['num_actions'] == 16 and INFO['num_features'] == 64 and set(INFO['bots']) == {'random', 'greedy'}
    g = E.Game('toy_team', None, 4, 1)
    assert g.team_size == 2 and not g.deferred and E.Game('toy', None, 3, 0).team_size == 1
    with pytest.raises(ValueError):
        g.resolve([[1, 0]])                         # nothing to resolve yet
    g = play_to_end(1)
    assert g.over and g.deferred
    with pytest.raises(ValueError, match='deferred terminal'):
        g.result()
    with pytest.raises(ValueError, match='deferred terminal'):
        g.margin(0)
    p = json.loads(g.deferred_payload())
    assert len(p['episodes']) == 4 and p['pairs'] == [[0, 1], [2, 3], [0, 3], [2, 1]]
    with pytest.raises(ValueError):
        g.resolve([[1, 0]] * 3)                     # 4 matches
    with pytest.raises(ValueError):
        g.resolve([[0.6, 0.4]] * 4)
    assert g.deferred
    g.resolve(json.dumps([{'result': [1.0, 0.0], 'scores': [9, 3]}, [0, 1], [1, 0], [0.5, 0.5]]))
    assert not g.deferred and g.result() == [0.5, 0.0, 0.5, 0.0] and g.margin(0) == 0.25 and g.scores() == [2, 1, 2, 1]
    # the toy's food spec (the matches' setups)
    a, b = E.Game('toy', {'food': [5, 0]}, 2, 3), E.Game('toy', None, 2, 3)
    assert json.loads(a.export_state())['players'][0]['food'] == json.loads(b.export_state())['players'][0]['food'] + 5
    for bad in ({'food': [1, 2, 3]}, {'food': 'x'}, {'food': [-1, 0]}, {'nope': 1}):
        with pytest.raises(ValueError):
            E.Game('toy', bad, 2, 0)


def search_both(g, sims, seed, ev):
    nv_py, w_py, cq_py, tp = parity.search_py(g, ev, sims, seed, cq=(50.0, 1.0), tree=True)
    nv_rs, w_rs, cq_rs, tr = parity.search_rs(g, ev, sims, seed, cq=(50.0, 1.0), tree=True)
    return (nv_py, w_py, cq_py, tp), (nv_rs, w_rs, cq_rs, tr)


def test_deferred_leaves_search_identically_without_chance():
    """In the last round nothing is left to chance: the Rust and the Python search are then
    deterministic and must agree exactly (visits, root win, completed-Q target), the deferred
    terminals being evaluated once each by the network and their values backed up after."""
    ev = parity.LinearEval(INFO['num_features'], INFO['num_actions'], seed=3, scale=0.3)
    checked = 0
    for seed in range(12):
        g = E.Game('toy_team', None, 4, 50 + seed)
        r = random.Random(seed)
        for _ in range(4 + seed % 3):              # into round 2 (4 seats x 2 rounds)
            g.step(r.choice(g.legal()))
        assert not g.over and len(g.legal()) >= 2
        calls = []

        def counting(x, acts, n):
            calls.append(len(acts))
            return ev(x, acts, n)
        (nv_py, w_py, cq_py, tp), (nv_rs, w_rs, cq_rs, tr) = search_both(g, 200, seed, counting)
        assert np.array_equal(nv_py, nv_rs), (nv_py, nv_rs)
        assert np.allclose(w_py, w_rs, atol=1e-9) and abs(sum(w_py) - 1) < 1e-9
        assert np.allclose(cq_py, cq_rs, atol=1e-6)
        # deferred leaves (no options) were evaluated, each once per tree; utilities are 2 x win
        deferred = [c for c in calls if c == 0]
        assert deferred and len(deferred) % 2 == 0
        assert tp.team == 2 and tr.root_value0 <= 1.0
        q = [w / n for w, n in zip(tr.root_values, tr.root_visits) if n]
        assert all(0.0 <= v <= 1.0 for v in q)
        checked += 1
    assert checked == 12


def test_rust_matches_python_on_team_games():
    r = parity.positions('toy_team', n=12, sims=96, reps=24, players=4, depth=(0, 8))
    assert r['tv_rs_py'] < 1.6 * r['tv_py_py'] + 0.003, r
    assert r['win_diff'] < 0.02, r
    assert r['tv_cq_rs_py'] < 1.6 * r['tv_cq_py_py'] + 0.006, r
    r1 = parity.positions('toy_team', n=8, sims=96, reps=24, players=2, depth=(0, 4))
    assert r1['tv_rs_py'] < 1.6 * r1['tv_py_py'] + 0.003, r1
    # the check has power here too
    p = parity.positions('toy_team', n=8, sims=96, reps=24, players=4, depth=(0, 8), c_puct_rs=1.3)
    assert p['tv_rs_py'] > 3 * p['tv_py_py'], p


def test_reference_values_with_teams():
    util, win = mcts.leaf_values([0.3, 0.2, 0.3, 0.2], [0.0] * 4, 1, 4, 0.0, team=2)
    assert win == [0.2, 0.3, 0.2, 0.3] and util == [0.4, 0.6, 0.4, 0.6]
    assert mcts.leaf_values([0.6, 0.4], [0.0] * 2, 0, 2, 0.0) == ([0.6, 0.4], [0.6, 0.4])


def test_empty_option_segments_in_the_evaluators():
    off = np.array([0, 3, 3, 5, 5], np.int32)
    lg = np.array([1.0, 2.0, 3.0, 0.5, 0.5], np.float32)
    p = segment_softmax_np(lg, off)
    assert np.allclose(p, protocol.segment_softmax(lg, off)) and abs(p[:3].sum() - 1) < 1e-6
    assert len(segment_softmax_np(np.zeros(0, np.float32), np.array([0, 0], np.int32))) == 0
    a = {'x': np.zeros((4, INFO['num_features']), np.float32), 'opt_off': off, 'opt_action': np.array([1, 2, 3, 4, 5], np.int32),
         'player': np.zeros(4, np.int32), 'n_players': np.full(4, 4, np.int32), 'gid': np.arange(4, dtype=np.int64)}
    lb = protocol.LeafBatch(0, 'm', 'nn', a)
    protocol.validate_batch(lb, INFO['num_features'], INFO['num_actions'])
    from zero.net import Net
    for ev in (UniformEvaluator(), NetEvaluator(Net(INFO['num_features'], INFO['num_actions'], 16, 1))):
        pri, v, m = ev(lb)
        protocol.validate_evaluation(lb, pri, v, m)
        assert pri.shape == (5,) and v.shape == (4, 8) and np.allclose(v[:, :4].sum(1), 1)


def test_python_driver_refuses_to_finish_a_deferred_game():
    seat = NNSeat('m', 4)
    drv = PyDriver('toy_team', 1, lambda g: [seat, 'random', seat, 'random'], concurrency=1, seed=1)
    with pytest.raises(RuntimeError, match='deferred terminal'):
        drive(drv, {'m': UniformEvaluator()})


def resolve_with_toy_matches(drv, mdrv, bot='heuristic'):
    """Every parked event's episodes through the queue_only toy driver `mdrv` -> resolved count."""
    eps_of, res_of = package.active().deferred_of('toy_team')
    want = {}
    for gid, payload in drv.take_deferred():
        eps = eps_of(payload)
        want[gid] = (payload, len(eps))
        for k, spec in enumerate(eps):
            mdrv.enqueue(f'{gid}/{k}', spec)
    drive(mdrv, {})
    rows = {}
    for r in mdrv.take_results():
        g, k = map(int, r['tag'].split('/'))
        rows.setdefault(g, {})[k] = r
    for gid, (payload, n) in want.items():
        assert sorted(rows[gid]) == list(range(n))
        drv.resolve(gid, res_of(payload, [rows[gid][k] for k in range(n)]))
    return len(want)


def test_driver_parks_and_resolves_through_the_bindings():
    seat = G.nn_seat('m', 8, selfplay=True, sims_fast=4, p_full=0.5)
    drv = G.Driver('toy_team', [[seat] * 4] * 6, concurrency=6, threads=2, seed=4, record=True)
    lbs = []
    drive(drv, {'m': UniformEvaluator()}, on_batch=lambda lb: lbs.append(lb))
    deferred_rows = sum(int(np.sum(np.diff(lb.opt_off) == 0)) for lb in lbs)
    assert deferred_rows > 0, 'the search evaluated deferred terminals'
    for lb in lbs:
        protocol.validate_batch(lb, INFO['num_features'], INFO['num_actions'])
    assert drv.pending == 6 and drv.take_results() == [] and drv.take_samples() is None and not drv.finished
    st = drv.stats()
    assert st['parked'] == 6 and st['parked_bytes'] > 0 and st['resolved'] == 0
    m = G.Driver('toy', [[{'bot': 'heuristic'}] * 2], concurrency=8, threads=1, seed=1, rolling=True, queue_only=True)
    m.resume(1 << 40, 0)
    assert m.next_batch() is None and m.queued == 0
    with pytest.raises(ValueError):
        m.enqueue('bad', {'food': 'x'})
    with pytest.raises(ValueError):
        drv.resolve(99, [])
    assert resolve_with_toy_matches(drv, m) == 6
    assert drv.pending == 0 and drv.finished
    rows, s = drv.take_results(), drv.take_samples()
    assert len(rows) == 6 and all(r['over'] and r['tag'] is None for r in rows)
    protocol.validate_samples(s, INFO['num_features'])
    assert len(s['player']) == sum(r['samples'] for r in rows) > 0
    for b in range(len(s['player'])):
        assert s['z'][b, 0] == s['z'][b, 2] and s['sc'][b, 0] == s['sc'][b, 2]
        assert s['z'][b, 0] in (0.0, 0.25, 0.5)
    assert drv.stats()['resolved'] == 6 and m.stats()['games_finished'] == 24
    with pytest.raises(ValueError):
        G.Driver('toy', [[{'bot': 'heuristic'}] * 2], queue_only=True)    # queue_only needs rolling


def test_queued_setups_carry_tags_and_their_spec():
    m = G.Driver('toy', [[{'bot': 'heuristic'}] * 2], concurrency=4, threads=1, seed=9, rolling=True, queue_only=True)
    m.resume(1 << 40, 0)
    for k in range(5):
        m.enqueue(f'x{k}', {'food': [30 * (k % 2), 0]})
    drive(m, {})
    rows = sorted(m.take_results(), key=lambda r: r['gid'])
    assert [r['tag'] for r in rows] == [f'x{k}' for k in range(5)]
    # 30 extra food for seat 0 in the odd games: it wins them
    assert all(r['result'][0] == 1.0 for r in rows if r['gid'] % 2 == 1)
    plain = G.Driver('toy', [[{'bot': 'heuristic'}] * 2], concurrency=4, threads=1, seed=9)
    drive(plain, {})
    assert all(r['tag'] is None for r in plain.take_results())
