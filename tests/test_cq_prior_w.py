"""The prior exponent of the 'cq' target (`cq_prior_w` = beta, docs/search.md 3.13, docs/az.md 26):
pi ~ P0^beta exp(sigma completedQ) - the formula (hand-checked; the Python reference = the Rust tree), the
drivers (only the recorded pi changes; 1.0 = the records of before), the seat key, the run setting."""
import math

import numpy as np
import pytest

from zero import _engine as E
from zero import games as G
from zero import mcts, parity, protocol, run
from zero.infer import drive
from zero.play import NNSeat, PyDriver

INFO = E.game_info_of('toy')

# (P0, N, W, v, c_visit, c_scale, beta) -> the target, 50-digit mpmath (E:/mtg-zero-scratch/ideas-1/
# ref_numbers.py; the Rust test search/src/tree/tests.rs::completed_q_prior_w_by_hand has the same numbers)
HAND = [
    (([0.5, 0.3, 0.2], [6, 3, 0], [3.6, 2.4, 0.0], 0.5, 50, 0.1, 0.5), [0.2354746509, 0.5590229101, 0.205502439]),
    (([0.5, 0.3, 0.2], [6, 3, 0], [3.6, 2.4, 0.0], 0.5, 50, 1.0, 0.7), [1.954679451e-5, 0.9997228533, 0.0002575999492]),
    (([0.9, 0.08, 0.02], [27, 4, 1], [27 * 0.40, 4 * 0.47, 0.60], 0.45, 50, 0.3, 0.6),
     [0.07987888371, 0.09418779954, 0.8259333168]),
    (([0.97, 0.02, 0.01], [60, 3, 0], [60 * 0.55, 3 * 0.52, 0.0], 0.55, 50, 0.3, 0.8),
     [0.9597958246, 0.01598210311, 0.02422207231]),
    (([0.8, 0.2], [0, 0], [0.0, 0.0], 0.5, 50, 1.0, 0.5), [2 / 3, 1 / 3]),
]


def close(a, b, rel=1e-6):
    return len(a) == len(b) and all(abs(x - y) <= rel * abs(y) + 1e-12 for x, y in zip(a, b))


def test_formula_hand_checked_python_and_rust():
    for args, want in HAND:
        py = mcts.completed_q_policy(*args)
        rs = E.completed_q_policy(*args)
        assert close(py, want), (args, py)
        assert close(rs, want), (args, rs)
    # more (P0, N, W, beta): Python = Rust (the Rust output is f32), a zero prior and a single option too
    rng = np.random.default_rng(3)
    for _ in range(40):
        k = int(rng.integers(1, 7))
        p0 = rng.dirichlet(np.full(k, 0.3)).astype(np.float32)
        p0[rng.random(k) < 0.15] = 0.0
        n = rng.integers(0, 30, size=k) * (rng.random(k) < 0.7)
        w = n * rng.random(k)
        v, cs, beta = float(rng.random()), float(rng.choice([0.1, 0.3, 1.0])), float(rng.choice([0.3, 0.6, 0.8, 1.0]))
        args = ([float(x) for x in p0], [int(x) for x in n], [float(x) for x in w], v, 50.0, cs, beta)
        py, rs = mcts.completed_q_policy(*args), E.completed_q_policy(*args)
        assert np.abs(np.asarray(py) - np.asarray(rs)).max() < 1e-6, (args, py, rs)
        assert abs(sum(rs) - 1) < 1e-5 and all(math.isfinite(x) and x >= 0 for x in rs)
    # 1.0 is the target of before, bit for bit (both languages)
    for args, _ in HAND:
        base = args[:6]
        assert mcts.completed_q_policy(*base, 1.0) == mcts.completed_q_policy(*base)
        assert E.completed_q_policy(*base, 1.0) == E.completed_q_policy(*base)
    with pytest.raises(ValueError):
        E.completed_q_policy([0.5, 0.5], [1], [0.5, 0.5], 0.5)


def test_python_formula_on_rust_roots_matches_rust_at_every_beta():
    assert parity.cq_formula('toy', n=6, sims=64, prior_ws=(1.0, 0.8, 0.5, 0.2)) < 1e-6


def parity_eval(info):
    """A deterministic, position-dependent batch evaluator with skewed priors (`parity.LinearEval`)."""
    le = parity.LinearEval(info['num_features'], info['num_actions'], seed=1, scale=0.15)

    def ev(lb):
        pri, val = [], np.zeros((lb.B, protocol.MAX_SEATS), np.float32)
        off = lb.opt_off
        for b in range(lb.B):
            p, v, _ = le(lb.x[b], lb.opt_action[off[b]:off[b + 1]], int(lb.n_players[b]))
            pri.extend(p)
            val[b] = v
        return np.asarray(pri, np.float32), val, np.zeros((lb.B, protocol.MAX_SEATS), np.float32)
    return ev


def rust_selfplay(seat, n=16, seed=5):
    drv = G.Driver('toy', [[seat, seat]] * n, concurrency=8, threads=2, seed=seed, record=True, max_steps=5000)
    parts = []
    drive(drv, {'m': parity_eval(INFO)}, on_batch=lambda lb: parts.append(drv.take_samples()))
    parts.append(drv.take_samples())
    res = sorted(drv.take_results(), key=lambda r: r['gid'])
    return res, protocol.concat_samples(parts)


def sharpness(s):
    from zero.data import target_sharpness
    _, h, mx = target_sharpness(s)
    return h.mean(), mx.mean()


def toy_cq(**kw):
    return G.nn_seat('m', 16, selfplay=True, sims_fast=4, p_full=0.5, temp_moves=10, policy_target='cq',
                     cq_c_scale=0.3, **kw)


def test_rust_driver_beta_changes_only_pi():
    r1, s1 = rust_selfplay(toy_cq())
    rx, sx = rust_selfplay(toy_cq(cq_prior_w=1.0))
    rb, sb = rust_selfplay(toy_cq(cq_prior_w=0.6))
    key = [(r['gid'], r['steps'], r['scores'], r['samples']) for r in r1]
    assert key == [(r['gid'], r['steps'], r['scores'], r['samples']) for r in rb]
    for k in protocol.SAMPLE_KEYS:
        assert np.array_equal(s1[k], sx[k]), k                  # 1.0 given = the default, bit for bit
        if k != 'pi':
            assert np.array_equal(s1[k], sb[k]), k              # moves, q, z, features, options: unchanged
    protocol.validate_samples(sb, INFO['num_features'])
    assert not np.array_equal(s1['pi'], sb['pi'])
    (h1, m1), (hb, mb) = sharpness(s1), sharpness(sb)
    assert hb > h1 and mb < m1, (h1, m1, hb, mb)


@pytest.mark.parametrize('impl', ['py', 'rs'])
def test_python_driver_beta_changes_only_pi(impl):
    ev = parity_eval(INFO)

    def go(**kw):
        seat = NNSeat('m', 16, selfplay=True, sims_fast=4, p_full=0.5, temp_moves=10, impl=impl, policy_target='cq',
                      cq_c_scale=0.3, **kw)
        drv = PyDriver('toy', 6, lambda g: [seat, seat], concurrency=6, seed=3, record=True)
        drive(drv, {'m': ev})
        return sorted(drv.take_results(), key=lambda r: r['gid']), drv.take_samples()
    r1, s1 = go()
    rb, sb = go(cq_prior_w=0.6)
    assert [(r['steps'], r['scores']) for r in r1] == [(r['steps'], r['scores']) for r in rb]
    for k in protocol.SAMPLE_KEYS:
        if k != 'pi':
            assert np.array_equal(s1[k], sb[k]), k
    assert not np.array_equal(s1['pi'], sb['pi'])
    assert sharpness(sb)[0] > sharpness(s1)[0]


def test_seat_key_and_refusals():
    assert 'cq_prior_w' not in toy_cq() and 'cq_prior_w' not in toy_cq(cq_prior_w=1.0)
    assert toy_cq(cq_prior_w=0.8)['cq_prior_w'] == 0.8
    for bad in (0.0, -0.2, 1.5, float('nan'), True):
        with pytest.raises(ValueError):
            toy_cq(cq_prior_w=bad)
        with pytest.raises(ValueError):
            NNSeat('m', 8, policy_target='cq', cq_prior_w=bad)
    with pytest.raises(ValueError):                                     # 'visits' has no prior term
        G.nn_seat('m', 8, cq_prior_w=0.8)
    with pytest.raises(ValueError):
        NNSeat('m', 8, cq_prior_w=0.8)
    # the bindings refuse it too (a seat dict built by hand)
    for bad in (0.0, 1.2, -1.0):
        with pytest.raises(ValueError):
            G.Driver('toy', [[dict(toy_cq(), cq_prior_w=bad), {'bot': 'random'}]])
    with pytest.raises(ValueError):
        G.Driver('toy', [[dict(G.nn_seat('m', 8), cq_prior_w=0.8), {'bot': 'random'}]])
    G.Driver('toy', [[toy_cq(cq_prior_w=0.5), {'bot': 'random'}]])


def test_run_setting_reaches_the_selfplay_seat_only():
    cfg = run.resolve_config(run.parse(['--run', 'x', '--policy-target', 'cq', '--cq-c-scale', '0.3',
                                        '--cq-prior-w', '0.8']), {})
    assert cfg['cq_prior_w'] == 0.8
    R = run.Run('x', cfg)
    assert R.nn_seat('m', 16, selfplay=True)['cq_prior_w'] == 0.8
    assert 'cq_prior_w' not in R.nn_seat('m', 16)                    # evaluation seats: no records, no key
    base = run.resolve_config(run.parse(['--run', 'x', '--policy-target', 'cq']), {})
    assert base['cq_prior_w'] == 1.0 and 'cq_prior_w' not in run.Run('x', base).nn_seat('m', 16, selfplay=True)
    for argv in (['--policy-target', 'cq', '--cq-prior-w', '0'], ['--policy-target', 'cq', '--cq-prior-w', '1.5'],
                 ['--cq-prior-w', '0.8']):                            # the last: 'visits' has no prior term
        with pytest.raises(SystemExit):
            run.resolve_config(run.parse(['--run', 'x'] + argv), {})
