"""The Python reference MCTS and statistical parity with the Rust tree (docs/search.md 6)."""
import random

import numpy as np

from zero import _engine as E
from zero import mcts, parity

INFO = E.game_info_of('toy')


def stand_in():
    return parity.LinearEval(INFO['num_features'], INFO['num_actions'], seed=0, scale=0.15)


def test_reference_search_basics():
    g = E.Game('toy', None, 2, 4)
    nv, w = parity.search_py(g, stand_in(), 100, seed=0)
    assert nv.sum() == 99 and abs(sum(w) - 1) < 1e-9
    nv2, _ = parity.search_py(g, stand_in(), 100, seed=0)
    assert np.array_equal(nv, nv2)


def test_reference_chance_k_caps_outcomes():
    g = E.Game('toy', None, 2, 5)
    while True:                                   # the last decision of round 1
        legal = g.legal()
        if g.clone().step(legal[0]):
            break
        g.step(legal[0])
    ev = stand_in()
    rng = random.Random(0)
    t = mcts.Tree(g, True, rng)
    for _ in range(300):
        r = mcts.simulate(t, rng, 1.0, chance_k=2)
        if r is not None:
            node, leaf, path = r
            mcts.expand_and_backup(node, path, *ev(leaf.encode(node.player), node.acts, 2), 2)
    seen = 0
    for e in t.root.kids:
        if e is not None and e.outs:
            seen += 1
            assert len({k for k, _, _ in e.kids}) <= 2 and len(e.outs) <= 2
    assert seen > 0


def test_rust_matches_python_root_statistics():
    r = parity.positions(n=8, sims=96, reps=16)
    assert r['tv_rs_py'] < 1.6 * r['tv_py_py'] + 0.003, r
    assert r['win_diff'] < 0.02, r
    # the completed-Q targets of the same searches (docs/search.md 3.1)
    assert r['tv_cq_rs_py'] < 1.6 * r['tv_cq_py_py'] + 0.004, r


def test_rust_matches_python_cq_targets_softer_scale():
    r = parity.positions(n=8, sims=96, reps=16, cq=(50.0, 0.1))
    assert r['tv_cq_rs_py'] < 1.6 * r['tv_cq_py_py'] + 0.003, r


def test_parity_check_has_power():
    r = parity.positions(n=6, sims=96, reps=16, c_puct_rs=1.3)
    assert r['tv_rs_py'] > 3 * r['tv_py_py'], r
    assert r['tv_cq_rs_py'] > 2 * r['tv_cq_py_py'], r


def test_rust_matches_python_with_chance_k():
    r = parity.positions(n=6, sims=96, reps=16, chance_k=2)
    assert r['tv_rs_py'] < 1.6 * r['tv_py_py'] + 0.004, r
    assert r['tv_cq_rs_py'] < 1.6 * r['tv_cq_py_py'] + 0.006, r
