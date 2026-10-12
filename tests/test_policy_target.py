"""Policy targets of recorded decisions (docs/search.md 3.1): 'visits' (unchanged) and Gumbel
AlphaZero's completed-Q target 'cq' - the formula (hand-checked, Python = Rust), the drivers
(only pi changes), seat settings, target statistics and the run settings."""
import math

import numpy as np
import pytest

from zero import _engine as E
from zero import games as G
from zero import mcts, parity, protocol, run
from zero.data import TargetStats, target_sharpness
from zero.infer import drive
from zero.play import NNSeat, PyDriver

INFO = E.game_info_of('toy')


def close(a, b, rel=1e-6):
    return len(a) == len(b) and all(abs(x - y) <= rel * abs(y) + 1e-12 for x, y in zip(a, b))


def test_completed_q_hand_checked():
    # the Rust unit test's numbers (search/src/tree/tests.rs), reference: 40-digit decimals
    p0, n, w = [0.5, 0.3, 0.2], [6, 3, 0], [3.6, 2.4, 0.0]
    assert close(mcts.completed_q_policy(p0, n, w, 0.5, 50, 1.0), [2.278460898e-05, 9.997491130e-01, 2.281023727e-04])
    assert close(mcts.completed_q_policy(p0, n, w, 0.5, 50, 0.1), [2.949098763e-01, 5.423134643e-01, 1.627766594e-01])
    pi = mcts.completed_q_policy([0.25] * 4, [5, 5, 0, 0], [2.0, 2.0, 0.0, 0.0], 0.3, 50, 1.0)
    assert close(pi, [3.112296656e-01, 3.112296656e-01, 1.887703344e-01, 1.887703344e-01])
    assert pi[0] == pi[1] and pi[2] == pi[3]


def test_completed_q_edge_cases():
    assert mcts.completed_q_policy([1.0], [7], [3.0], 0.4) == [1.0]
    assert close(mcts.completed_q_policy([0.1, 0.6, 0.3], [0, 0, 0], [0.0] * 3, 0.7), [0.1, 0.6, 0.3])
    assert close(mcts.completed_q_policy([0.1, 0.6, 0.3], [4, 0, 9], [4.0, 0.0, 0.0], 0.7, 50, 0.0), [0.1, 0.6, 0.3])
    pi = mcts.completed_q_policy([1.0, 0.0, 0.0], [3, 1, 0], [1.5, 0.9, 0.0], 0.5)
    assert all(math.isfinite(x) and x >= 0 for x in pi) and abs(sum(pi) - 1) < 1e-12 and pi[0] > 0.999


def test_python_formula_on_rust_roots_matches_rust():
    assert parity.cq_formula('toy', n=6, sims=64) < 1e-6


def toy_seat(**kw):
    return G.nn_seat('m', 16, selfplay=True, sims_fast=4, p_full=0.5, temp_moves=10, **kw)


def rust_selfplay(seat, n=16, seed=5, game='toy', spec=None, ev=None):
    info = G.info(game)
    ev = ev or parity_eval(info)
    drv = G.Driver(game, [[seat, seat]] * n, spec=spec, concurrency=8, threads=2, seed=seed, record=True,
                   max_steps=5000)
    parts = []
    drive(drv, {'m': ev}, on_batch=lambda lb: parts.append(drv.take_samples()))
    parts.append(drv.take_samples())
    res = sorted(drv.take_results(), key=lambda r: r['gid'])
    return res, protocol.concat_samples(parts)


def parity_eval(info):
    """A deterministic, position-dependent batch evaluator (the parity stand-in, batched)."""
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


def norm_entropy(s):
    _, h, mx = target_sharpness(s)
    return h.mean(), mx.mean()


@pytest.mark.parametrize('game,spec,n', [('toy', None, 16)])   # each game repo adds its game
def test_rust_driver_cq_changes_only_pi(game, spec, n):
    rv, sv = rust_selfplay(G.nn_seat('m', 16, selfplay=True, sims_fast=4, p_full=0.5, temp_moves=10), n, game=game,
                           spec=spec)
    rx, sx = rust_selfplay(G.nn_seat('m', 16, selfplay=True, sims_fast=4, p_full=0.5, temp_moves=10,
                                     policy_target='visits'), n, game=game, spec=spec)
    rc, sc = rust_selfplay(G.nn_seat('m', 16, selfplay=True, sims_fast=4, p_full=0.5, temp_moves=10,
                                     policy_target='cq', cq_c_visit=50, cq_c_scale=1.0), n, game=game, spec=spec)
    key = [(r['gid'], r['steps'], r['scores'], r['samples']) for r in rv]
    assert key == [(r['gid'], r['steps'], r['scores'], r['samples']) for r in rc]
    for k in protocol.SAMPLE_KEYS:                 # explicit 'visits' = the default, bit for bit
        assert np.array_equal(sv[k], sx[k]), k
    for k in protocol.SAMPLE_KEYS:
        if k != 'pi':
            assert np.array_equal(sv[k], sc[k]), k
    protocol.validate_samples(sc, G.info(game)['num_features'])
    off = sc['opt_off']
    sums = np.add.reduceat(sc['pi'].astype(np.float64), off[:-1])
    assert np.allclose(sums, 1, atol=1e-5) and np.all(sc['pi'] >= 0)
    assert not np.array_equal(sv['pi'], sc['pi'])
    (hv, mv), (hc, mc) = norm_entropy(sv), norm_entropy(sc)
    assert hc < hv and mc > mv, (hv, mv, hc, mc)


@pytest.mark.parametrize('impl', ['py', 'rs'])
def test_python_driver_cq_targets(impl):
    ev = parity_eval(INFO)

    def go(**kw):
        seat = NNSeat('m', 16, selfplay=True, sims_fast=4, p_full=0.5, temp_moves=10, impl=impl, **kw)
        drv = PyDriver('toy', 6, lambda g: [seat, seat], concurrency=6, seed=3, record=True)
        drive(drv, {'m': ev})
        return sorted(drv.take_results(), key=lambda r: r['gid']), drv.take_samples()
    rv, sv = go()
    rc, sc = go(policy_target='cq', cq_c_visit=50.0, cq_c_scale=1.0)
    assert [(r['steps'], r['scores']) for r in rv] == [(r['steps'], r['scores']) for r in rc]
    for k in protocol.SAMPLE_KEYS:
        if k != 'pi':
            assert np.array_equal(sv[k], sc[k]), k
    protocol.validate_samples(sc, INFO['num_features'])
    assert not np.array_equal(sv['pi'], sc['pi'])
    assert norm_entropy(sc)[0] < norm_entropy(sv)[0]


def test_seat_settings_are_validated():
    with pytest.raises(ValueError):
        G.Driver('toy', [[toy_seat(policy_target='argmax'), {'bot': 'random'}]])
    with pytest.raises(ValueError):
        G.Driver('toy', [[toy_seat(policy_target='cq', cq_c_scale=-1.0), {'bot': 'random'}]])
    with pytest.raises(ValueError):
        NNSeat('m', 8, policy_target='nope')
    assert 'policy_target' not in G.nn_seat('m', 8)
    assert G.nn_seat('m', 8, policy_target='cq', cq_c_scale=0.3)['cq_c_scale'] == 0.3


def test_target_statistics():
    s = {'player': np.zeros(3, np.int32), 'opt_off': np.array([0, 2, 6, 7]),
         'pi': np.array([0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0], np.float32)}
    n, h, mx = target_sharpness(s)
    assert n.tolist() == [2, 4] and np.allclose(h, [1.0, 0.0]) and np.allclose(mx, [0.5, 1.0])
    t = TargetStats()
    t.add(s)
    t.add(None)
    assert t.summary('cq') == {'kind': 'cq', 'samples': 2, 'options': 3.0, 'h_norm': 0.5, 'max_share': 0.75}


def test_run_settings():
    args = run.parse(['--run', 'x', '--policy-target', 'cq', '--cq-c-scale', '0.3'])
    cfg = run.resolve_config(args, {})
    assert cfg['policy_target'] == 'cq' and cfg['cq_c_scale'] == 0.3 and cfg['cq_c_visit'] == 50.0
    seat = run.Run('x', cfg).nn_seat('m', 16, selfplay=True)
    assert seat['policy_target'] == 'cq' and seat['cq_c_scale'] == 0.3
    assert 'policy_target' not in run.Run('x', cfg).nn_seat('m', 16)          # evaluation seats: no records
    assert run.resolve_config(run.parse(['--run', 'x']), {})['policy_target'] == 'visits'
    assert 'policy_target' not in run.Run('x', run.resolve_config(run.parse(['--run', 'x']), {})).nn_seat('m', 8, True)
    with pytest.raises(SystemExit):
        run.resolve_config(run.parse(['--run', 'x', '--policy-target', 'argmax']), {})
    assert {'toy-smoke', 'toy-1'} <= set(run.PRESETS)          # game presets: each game repo's tests


def test_targets_tool_on_toy(tmp_path):
    import torch
    from zero import targets
    from zero.net import Net, save
    torch.manual_seed(0)
    path = str(tmp_path / 'm.pt')
    save(Net(INFO['num_features'], INFO['num_actions'], hidden=32, blocks=1).eval(), path, {'iter': 0})
    r = targets.compare(path, 'toy', None, games=4, sims=16, sims_fast=4, p_full=0.5, chance_k=0, c_scales=(1.0, 0.1),
                        threads=1, verbose=False)
    assert set(r['arms']) == {'visits', 'cq c_scale 1', 'cq c_scale 0.1'} and r['positions'] > 0
    assert r['arms']['visits']['all']['argmax_differs_from_visits'] == 0
    assert r['arms']['cq c_scale 1']['all']['h_norm'] < r['arms']['cq c_scale 0.1']['all']['h_norm']
    q = targets.quality(path, 'toy', None, positions=6, sims=16, reps=1, ref_sims=48, chance_k=0, verbose=False)
    assert q['positions'] == 6 and set(q['targets']) >= {'prior', 'visits', 'cq c_scale 1', 'greedy Q'}
    assert all(0 <= d['best_mass'] <= 1 and d['regret'] >= 0 for d in q['targets'].values())


def test_h2h_tool_on_toy(tmp_path):
    """zero.h2h: the A/B readout (two checkpoints, seats alternating)."""
    import torch
    from zero import h2h
    from zero.net import Net, save
    paths = []
    for k in range(2):
        torch.manual_seed(k)
        paths.append(str(tmp_path / f'm{k}.pt'))
        save(Net(INFO['num_features'], INFO['num_actions'], hidden=32, blocks=1, game='toy').eval(), paths[-1], {'iter': k})
    r = h2h.match(paths[0], paths[1], games=12, sims=8, chance_k=0, threads=1)
    assert r['game'] == 'toy' and r['games'] == 12 and 0 <= r['a_score'] <= 1 and r['unfinished'] == 0
    assert r['elo_diff'] == round(400 * math.log10(min(max(r['a_score'], 1e-3), 1 - 1e-3) /
                                                   (1 - min(max(r['a_score'], 1e-3), 1 - 1e-3))), 1)
