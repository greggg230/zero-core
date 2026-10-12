"""Opponent models and analysis options of network seats (docs/search.md 10) on the core's
"secret" game (a tiny hidden simultaneous hand choice + card duel): the game hooks through the
bindings, policy-sampled simultaneous choices in the Python reference tree vs the Rust tree,
the belief prior and traces through the driver, the h2h tool's seat options and by-seat
scores. The Rust-side unit tests: crates/search (secret, belief, tree, driver tests)."""
import math
import random

import numpy as np
import pytest

from zero import _engine as E
from zero import games as G
from zero import mcts, parity
from zero.infer import UniformEvaluator, drive
from zero.play import NNSeat, PyDriver

SEC = E.game_info_of('secret')


def after(acts):
    g = E.Game('secret', None, 2, 0)
    for a in acts:
        g.step(a)
    return g


def test_info_and_hooks():
    assert {'toy', 'secret'} <= set(E.game_names())
    assert (SEC['classes'], SEC['oracle'], SEC['psample'], SEC['belief']) == (['play', 'pick'], True, True, True)
    toy = E.game_info_of('toy')
    assert (toy['classes'], toy['oracle'], toy['psample'], toy['belief']) == ([], False, False, False)
    g = E.Game('secret', None, 2, 0)
    assert g.policy_sampled and g.decision_class == 1 and g.belief_start() == 0 and g.belief_target(0) is None
    g = after([1, 3])
    assert g.to_move == 1 and g.belief_particle(0) == 0b1010 and g.belief_particle(1) is None
    w = g.clone()
    w.uncommit_simultaneous(1)
    assert w.to_move == 1 and np.array_equal(w.encode(1), g.encode(1)) and w.belief_particle(0) is None
    g = after([1, 3, 0, 4])
    assert not g.policy_sampled and g.decision_class == 0 and g.belief_target(0) == (1, 0)
    assert g.belief_consistent(0, 0b10001) and not g.belief_consistent(0, 0b111)
    h = g.clone()
    h.determinize_particle(0, 0b00110, 5)
    assert h.belief_particle(1) == 0b00110 and np.array_equal(h.encode(0), g.encode(0))
    with pytest.raises(ValueError):
        g.clone().determinize_particle(0, 0b111, 5)
    g.clone().determinize(0, 1, oracle=True)
    with pytest.raises(ValueError):
        E.Game('toy', None, 2, 1).determinize(0, 1, oracle=True)
    with pytest.raises(ValueError):
        E.Game('toy', None, 2, 1).search_tree(True, 0, psample=True)
    with pytest.raises(ValueError):
        E.Game('toy', None, 2, 1).search_tree(True, 0, belief=[1], belief_mix=0.5)


def test_rust_tree_belief_and_counters():
    g = after([1, 3, 0, 4])
    t = g.search_tree(True, 3, belief=[0b00110, 0b111, 0b11000], belief_mix=1.0)
    assert sorted(t.belief) == [0b00110, 0b11000]
    ev = parity.LinearEval(SEC['num_features'], SEC['num_actions'], seed=1, scale=0.15)
    for _ in range(50):
        r = t.simulate(g, 1.0)
        if r is not None:
            t.expand(*ev(r[1], r[2], 2))
    assert t.belief_worlds > 0 and t.sampled == 0
    t.set_belief(g, [])
    assert t.belief == []


def pick_positions():
    """Hand-choice positions (>= 3 options): the first chooser's picks and the second chooser's
    (whose root worlds take the first chooser's hand back under psample)."""
    return [after([]), after([0]), after([1, 3]), after([2, 4]), after([0, 4, 1]), after([1, 2])]


def test_python_reference_matches_rust_with_psample():
    poss = pick_positions()
    assert all(g.policy_sampled and len(g.legal()) >= 3 for g in poss)
    r = parity.positions('secret', sims=96, reps=24, psample=True, poss=poss)
    assert r['tv_rs_py'] < 1.6 * r['tv_py_py'] + 0.004, r
    assert r['win_diff'] < 0.02, r
    assert r['tv_cq_rs_py'] < 1.6 * r['tv_cq_py_py'] + 0.006, r


def test_psample_parity_check_has_power():
    """Rust with psample vs Python without (and the reverse) differ far above the noise floor."""
    ev = parity.LinearEval(SEC['num_features'], SEC['num_actions'], seed=1, scale=0.15)
    tvs, floors = [], []
    for g in [after([1, 3]), after([2, 4]), after([])]:
        rs = np.mean([parity.search_rs(g, ev, 96, 100 + k, psample=True)[0] for k in range(24)], 0)
        py = np.mean([parity.search_py(g, ev, 96, 200 + k, psample=False)[0] for k in range(24)], 0)
        pa = np.mean([parity.search_py(g, ev, 96, 300 + k, psample=True)[0] for k in range(24)], 0)
        pb = np.mean([parity.search_py(g, ev, 96, 400 + k, psample=True)[0] for k in range(24)], 0)
        tvs.append(parity.tv(rs, py))
        floors.append(parity.tv(pa, pb))
    assert np.mean(tvs) > 3 * np.mean(floors), (tvs, floors)


def test_python_tree_samples_the_other_seats_choice():
    """In the reference tree the other seat's hand-choice nodes follow P0 (the prior), the
    root decider's are searched, and the second chooser's worlds pick the first chooser's hand
    after its own."""
    g = after([1, 3])
    ev = parity.LinearEval(SEC['num_features'], SEC['num_actions'], seed=1, scale=0.15)
    rng = random.Random(0)
    t = mcts.Tree(g, True, rng, psample=True)
    again = 0
    for _ in range(400):
        r = mcts.simulate(t, rng, 1.0)
        if r is None:
            continue
        node, leaf, path = r
        again += leaf.to_move == 0 and leaf.policy_sampled
        mcts.expand_and_backup(node, path, *ev(leaf.encode(node.player), node.acts, 2), 2)
    assert again > 0 and t.sampled > 0


def test_driver_options_through_the_bindings():
    S = G.seat_dict
    both = G.nn_seat('m', sims=12, selfplay=True, sims_fast=4, p_full=0.5, psample=True, belief_mix=0.5,
                     belief_temp=2.0, belief_particles=16)
    assert both['psample'] is True and both['belief_mix'] == 0.5 and both['belief_particles'] == 16
    plain = G.nn_seat('m', sims=12)
    assert not {'psample', 'belief_mix', 'oracle_hands', 'class_sims', 'mask_features'} & set(plain)
    d = G.Driver('secret', [[both, both], [both, S('random')], [S('random'), both]] * 4, concurrency=12, seed=3,
                 threads=2, record=True, trace=True)
    drive(d, {'m': UniformEvaluator()})
    st = d.stats()
    assert st['psampled'] > 0 and st['belief_jobs'] > 0 and st['belief_rows'] > 0 and st['belief_missing'] == 0
    res = d.take_results()
    assert len(res) == 12 and all(r['over'] for r in res)
    for r in res:                                  # traces replay
        g = E.Game('secret', None, 2, r['seed'])
        for a in r['actions']:
            g.step(a)
            while not g.over and len(g.legal()) == 1:
                g.step(g.legal()[0])
        assert g.over and g.scores() == r['scores']
    # oracle / class_sims / mask_features parse against the game's classes
    ora = G.nn_seat('m', sims=6, oracle_hands='pick', oracle_p=0.5, class_sims={'pick': 12}, mask_features=[0, 3])
    assert ora['oracle_hands'] == 'pick' and ora['class_sims'] == {'pick': 12} and ora['mask_features'] == [0, 3]
    d = G.Driver('secret', [[ora, S('random')]] * 2, concurrency=2, seed=1, threads=1, groups=1, record=True)
    drive(d, {'m': UniformEvaluator()})
    assert all(r['over'] for r in d.take_results())
    bad = [G.nn_seat('m', sims=4, oracle_hands='deal'), G.nn_seat('m', sims=4, class_sims={'deal': 8}),
           G.nn_seat('m', sims=4, class_sims={'pick': 0}), G.nn_seat('m', sims=4, oracle_hands=True, oracle_p=0.0),
           G.nn_seat('m', sims=4, mask_features=[SEC['num_features']]), G.nn_seat('m', sims=4, belief_mix=1.5),
           G.nn_seat('m', sims=4, belief_mix=0.5, belief_temp=0.0)]
    for b in bad:
        with pytest.raises(ValueError):
            G.Driver('secret', [[b, S('random')]], concurrency=1)
    for b in (G.nn_seat('m', sims=4, psample=True), G.nn_seat('m', sims=4, belief_mix=0.5),
              G.nn_seat('m', sims=4, oracle_hands=True)):
        with pytest.raises(ValueError):
            G.Driver('toy', [[b, S('random')]], concurrency=1)
    with pytest.raises(ValueError):
        G.Driver('secret', [[G.nn_seat('m', sims=4), S('ext:toyrandom')]], concurrency=1, trace=True)


def test_python_driver_psample_both_trees():
    for impl in ('py', 'rs'):
        seat = NNSeat('m', 12, psample=True, impl=impl)
        d = PyDriver('secret', 6, lambda g: [seat, 'random'] if g % 2 == 0 else ['random', seat], seed=1)
        drive(d, {'m': UniformEvaluator()})
        assert len(d.take_results()) == 6
    with pytest.raises(ValueError):
        PyDriver('toy', 2, lambda g: [NNSeat('m', 8, psample=True), 'random'])


def test_h2h_seat_options_and_seats(tmp_path):
    import torch
    from zero import h2h
    from zero.net import Net, save
    torch.manual_seed(0)
    path = str(tmp_path / 'secret.pt')
    save(Net(SEC['num_features'], SEC['num_actions'], hidden=32, blocks=1, game='secret').eval(), path, {'iter': 0})
    r = h2h.match(path, games=16, sims=8, chance_k=0, threads=1, a_opts={'psample': True, 'belief_mix': 1.0})
    assert r['game'] == 'secret' and r['games'] == 16 and r['b'] == path and r['a_opts']['psample'] is True
    assert set(r['a_by_seat']) == {0, 1} and r['a_by_seat'][0]['games'] == 8
    assert r['stats']['psampled'] > 0 and r['stats']['belief_jobs'] > 0
    d, se, h = h2h.seat_adjusted_elo(0.9, 100, 0.5, 100)
    assert d > 0 and se > 0 and h > 0
    assert abs(h2h.seat_adjusted_elo(0.6, 100, 0.6, 100)[2]) < 1e-9
    assert math.isclose(h2h.seat_adjusted_elo(0.75, 100, 0.25, 100)[0], 0.0, abs_tol=1e-9)
