"""The bindings: Game, SearchTree, SearchDriver (Rust) and the Python reference driver speak
the leaf-batch protocol (docs/search.md 2)."""
import numpy as np
import pytest

from zero import _engine as E
from zero import games as G
from zero import protocol
from zero.infer import UniformEvaluator, drive
from zero.play import NNSeat, PyDriver

INFO = E.game_info_of('toy')


def test_game_info_and_registry():
    assert 'toy' in E.game_names()
    assert INFO['num_actions'] == 18 and INFO['num_features'] > 100
    assert INFO['max_seats'] == protocol.MAX_SEATS == E.MAX_SEATS
    assert {'random', 'heuristic'} <= set(INFO['bots'])
    with pytest.raises(ValueError):
        E.Game('nope', None, 2, 0)
    with pytest.raises(ValueError):
        E.Game('toy', None, 9, 0)
    with pytest.raises(ValueError):
        E.Game('toy', {'bad': 1}, 2, 0)


def test_game_api():
    g = E.Game('toy', None, 2, 3)
    h = g.clone()
    keys = []
    rng = np.random.default_rng(0)
    while not g.over:
        legal = g.legal()
        assert legal and len(set(legal)) == len(legal)
        x = g.encode(g.to_move)
        assert x.dtype == np.float32 and x.shape == (INFO['num_features'],)
        keys += g.step(int(rng.choice(legal)))
    assert len(keys) == 5 and [k >> 16 for k in keys] == [2, 3, 4, 5, 6]
    assert abs(sum(g.result()) - 1) < 1e-6 and g.rounds == 6
    assert not h.over and h.rounds == 1                # clones are independent
    with pytest.raises(ValueError):
        h.step(99)


def last_decision_of_round_1(seed):
    g = E.Game('toy', None, 2, seed)
    while True:
        legal = g.legal()
        k = g.clone()
        if k.step(legal[0]):
            return g
        g.step(legal[0])


def test_determinize_and_force():
    g = last_decision_of_round_1(5)
    before = g.legal()
    x = g.encode(g.to_move)
    outcomes = set()
    for s in range(30):
        d = g.clone()
        d.determinize(g.to_move, s)
        assert d.legal() == before and np.array_equal(d.encode(g.to_move), x)
        outcomes.add(tuple(d.step(before[0])))
    assert len(outcomes) > 1, 'determinization reshuffles the hidden round order'
    key = next(iter(outcomes))[0]
    for s in range(10):
        d = g.clone()
        d.determinize(g.to_move, 100 + s)
        d.force_chance([key])
        assert d.step(before[0]) == [key]


def test_search_tree_api():
    g = E.Game('toy', None, 2, 1)
    t = g.search_tree(True, 0)
    for _ in range(64):
        r = t.simulate(g, 1.0)
        if r is None:
            continue
        p, x, acts = r
        assert x.shape == (INFO['num_features'],) and len(acts) >= 2
        t.expand([1 / len(acts)] * len(acts), [0.5, 0.5], [0, 0])
    assert t.root_expanded and sum(t.root_visits) == 63 and t.root_acts == g.legal()
    assert abs(sum(t.root_win()) - 1) < 1e-6
    # the shape diagnostic (docs/search.md 3.15): every node once, the walk length of the last simulation
    sh = t.shape()
    assert sh['nodes'] == t.n_nodes and sum(sh['depth_hist']) == sh['expanded'] <= 64 and sh['depth_hist'][0] == 1
    assert sum(e[2] for e in sh['edges'] if e[0] == 0) == 63 and sh['main_line'][0][2] == max(t.root_visits)
    assert sum(sh['opp_hist']) == sum(sh['switch_hist']) == sh['expanded'] and 1 <= t.path_len <= sh['max_depth']
    with pytest.raises(RuntimeError):
        t.expand([1.0], [1.0], [0.0])                  # no pending leaf


def _run(drv, ev=None, ext=None):
    res, smp = [], []

    def on_batch(lb):
        protocol.validate_batch(lb, INFO['num_features'], INFO['num_actions'])
        res.extend(drv.take_results())
        smp.append(drv.take_samples())
    ev = ev or UniformEvaluator()
    drive(drv, {'m': ev, 'o': ev}, ext, on_batch=on_batch)
    res.extend(drv.take_results())
    smp.append(drv.take_samples())
    return sorted(res, key=lambda r: r['gid']), protocol.concat_samples(smp)


def selfplay_seat(sims=16):
    return G.nn_seat('m', sims, selfplay=True, sims_fast=4, p_full=0.5, temp_moves=10)


def test_rust_driver_protocol_and_samples():
    drv = G.Driver('toy', [[selfplay_seat(), selfplay_seat()] for _ in range(20)], concurrency=10, threads=2,
                   seed=3, record=True)
    res, s = _run(drv)
    assert len(res) == 20 and all(r['over'] for r in res) and drv.finished
    protocol.validate_samples(s, INFO['num_features'])
    assert len(s['player']) == sum(r['samples'] for r in res) > 0
    st = drv.stats()
    assert st['games_finished'] == 20 and st['leaves'] > 0


def test_rust_driver_reproducible_across_threads():
    def go(threads, groups):
        drv = G.Driver('toy', [[selfplay_seat(), {'bot': 'heuristic'}] for _ in range(12)], concurrency=12,
                       threads=threads, groups=groups, seed=11, record=True)
        res, s = _run(drv)
        return [(r['gid'], r['steps'], r['scores']) for r in res], s['pi']
    a, pa = go(1, 1)
    b, pb = go(4, 3)
    assert a == b and np.array_equal(pa, pb)


def test_python_driver_protocol():
    seat = NNSeat('m', 16, selfplay=True, sims_fast=4, p_full=0.5, temp_moves=10)
    drv = PyDriver('toy', 8, lambda g: [seat, 'heuristic' if g % 2 else 'random'], concurrency=4, seed=1, record=True)
    res, s = _run(drv)
    assert len(res) == 8 and all(r['over'] for r in res)
    protocol.validate_samples(s, INFO['num_features'])


def test_mixed_driver_runs_both_trees():
    rs, py = NNSeat('m', 16, impl='rs'), NNSeat('m', 16, impl='py')
    drv = PyDriver('toy', 6, lambda g: [rs, py] if g % 2 == 0 else [py, rs], concurrency=6, seed=2)
    res, _ = _run(drv)
    assert len(res) == 6


def test_arena_keys_and_seat_validation():
    drv = G.Driver('toy', [[G.nn_seat('m', 8), G.nn_seat('o', 8)]] * 4, concurrency=4, threads=2)
    assert [k for k, _ in drv.keys] == ['m', 'o']
    res, _ = _run(drv)
    assert len(res) == 4
    with pytest.raises(ValueError):
        G.Driver('toy', [[{'bot': 'nope'}, {'bot': 'random'}]])
    with pytest.raises(ValueError):
        G.Driver('toy', [[{'key': 'm', 'simz': 3}, {'bot': 'random'}]])
    with pytest.raises(ValueError):
        G.Driver('toy', [[{'key': 'x'}, {'external': 'x'}]])


def test_evaluation_shape_errors():
    drv = G.Driver('toy', [[G.nn_seat('m', 8), {'bot': 'random'}]], concurrency=1, threads=1)
    lb = drv.next_batch()
    with pytest.raises(ValueError):
        drv.submit(lb.id, np.zeros(1, np.float32), np.zeros((1, 8), np.float32), np.zeros((1, 8), np.float32))
    pri, v, m = UniformEvaluator()(lb)
    protocol.validate_evaluation(lb, pri, v, m)
    drv.submit(lb.id, pri, v, m)
    with pytest.raises(ValueError):
        drv.submit(lb.id, pri, v, m)


def test_rolling_driver_phases():
    seat = selfplay_seat(8)
    drv = G.Driver('toy', [[seat, seat]], concurrency=8, threads=2, rolling=True, record=True)
    assert drv.paused and drv.next_batch() is None
    fin = 0
    for net in (0, 0, 1):
        drv.resume(5, net)
        drive(drv, {'m': UniformEvaluator()})
        fin += len(drv.take_results())
        assert drv.paused and drv.in_flight()['games'] > 0
    st = drv.stats()
    assert st['games_finished'] == fin >= 15 and st['epoch'] == 1
