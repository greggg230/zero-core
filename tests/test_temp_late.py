"""Late sampling (docs/search.md 3.10, docs/az.md 22: seat keys temp_late / temp_late_margin, run settings
selfplay_temp_late / selfplay_temp_late_margin): the Python formulas are the Rust driver's on Rust roots,
by hand, the seat settings and their checks, the Python and Rust drivers late-sampling alike in games,
off = the games of before, a run's metrics and config, replays marking late-sampled moves, and remote
self-play refusing a checkout without it."""
import json
import math

import pytest

from zero import games as G, mcts, parity
from zero.play import NNSeat, PyDriver
from zero.run import read_metrics

from test_loop import run_loop
from test_remoteplay import env_for
from test_select import _ckpts, _play, _Rand


def test_python_late_sampling_is_the_rust_drivers_on_rust_roots():
    """mcts.completed_q / late_weights / late_pick on Rust roots' statistics == the bindings' (the Rust
    driver's code) for every temperature and gate on a grid of draws; the gates bind on these roots."""
    r = parity.late_formula('toy', n=8)
    assert r['checked'] == 8 * 4 * len(parity.LATE_SETTINGS) * 64
    assert r['agree'] + r['near_ties'] == r['checked'] and not r['mismatches'], r['mismatches'][:5]
    assert r['gated'] > 0


def test_late_formulas_by_hand():
    nv, q = [10, 5, 0, 1], [0.60, 0.59, 0.90, 0.50]
    assert mcts.late_weights(nv, q, 0, 1.0) == [10.0, 5.0, 0.0, 1.0]
    assert mcts.late_weights(nv, q, 0, 0.5) == [100.0, 25.0, 0.0, 1.0]
    assert mcts.late_weights(nv, q, 0, 1.0, 0.02) == [10.0, 5.0, 0.0, 0.0]
    assert mcts.late_weights(nv, q, 1, 1.0, 0.005) == [10.0, 5.0, 0.0, 0.0]
    assert mcts.late_weights(nv, q, 3, 1.0, 0.001) == [10.0, 5.0, 0.0, 1.0]       # the rule's move always
    w = [10.0, 5.0, 0.0, 0.0]
    assert [mcts.late_pick(w, 0, u) for u in (0.0, 0.66, 0.67, 0.999999)] == [0, 0, 1, 1]
    assert mcts.late_pick([0.0, 0.0], 1, 0.3) == 1 and mcts.late_pick([], 0, 0.3) == 0
    E = G.engine()
    for rule in range(4):
        for temp, margin in ((1.0, 0.0), (0.7, 0.02), (2.0, 0.005)):
            assert mcts.late_weights(nv, q, rule, temp, margin) == E.late_weights(nv, q, rule, temp, margin)
    # the completed Q: W / N, v_mix for the unvisited option; the logits of before are made of it (priors
    # exact in f32, as a Rust tree stores them)
    P0, N, W, v = [0.5, 0.25, 0.25], [3, 1, 0], [1.8, 0.4, 0.0], 0.45
    cq = mcts.completed_q(P0, N, W, v)
    assert cq[:2] == [0.6, 0.4] and abs(cq[2] - (0.45 + 4 * (0.5 * 0.6 + 0.25 * 0.4) / 0.75) / 5) < 1e-12
    assert cq == E.completed_q(P0, N, W, v)
    lg = mcts.completed_q_logits(P0, N, W, v, 50.0, 0.3)
    assert all(abs(lg[i] - (math.log(P0[i]) + 53 * 0.3 * cq[i])) < 1e-12 for i in range(3))


def test_seat_settings_are_validated():
    # off: the seat dicts of before (no keys)
    base = G.nn_seat('m', 8, selfplay=True, temp_moves=10)
    assert 'temp_late' not in base and G.nn_seat('m', 8, selfplay=True, temp_moves=10, temp_late=0.0) == base
    s = G.nn_seat('m', 8, selfplay=True, temp_late=0.7, temp_late_margin=0.02)
    assert (s['temp_late'], s['temp_late_margin']) == (0.7, 0.02)
    for kw, msg in (({'temp_late': -1.0}, 'temp_late -1'), ({'temp_late': float('nan')}, 'temp_late nan'),
                    ({'temp_late_margin': 0.05}, 'a gate of temp_late'), ({'temp_late': 1.0, 'selfplay': False}, 'self-play'),
                    ({'temp_late': 1.0, 'root_search': 'gumbel', 'policy_target': 'cq'}, 'exclusive')):
        with pytest.raises(ValueError, match=msg):
            G.nn_seat('m', 8, **{'selfplay': True, **kw})
        with pytest.raises(ValueError, match=msg):
            NNSeat('m', 8, **{'selfplay': True, **kw})
    # the Rust driver checks them too
    for extra, msg in (({'temp_late': -0.5}, 'temp_late -0.5'), ({'temp_late_margin': 0.1}, 'a gate of temp_late'),
                       ({'temp_late': 1.0, 'selfplay': False}, 'self-play seat')):
        with pytest.raises(ValueError, match=msg):
            G.Driver('toy', [[dict(G.nn_seat('m', 4, selfplay=True), **extra)] * 2], concurrency=1)
    from zero.run import DEFAULTS, check_eval_settings
    assert (DEFAULTS['selfplay_temp_late'], DEFAULTS['selfplay_temp_late_margin']) == (0.0, 0.0)
    for bad in ({'selfplay_temp_late': -1.0}, {'selfplay_temp_late_margin': 0.05},
                {'selfplay_temp_late': 1.0, 'selfplay_root_search': 'gumbel', 'policy_target': 'cq'}):
        with pytest.raises(SystemExit):
            check_eval_settings(dict(DEFAULTS, **bad))
    check_eval_settings(dict(DEFAULTS, selfplay_temp_late=0.5, selfplay_temp_late_margin=0.02))


SEAT = dict(sims=16, selfplay=True, temp_moves=10, sims_fast=4, p_full=0.5)


def _rust_selfplay(n, seed, **late):
    s = G.nn_seat('m', **SEAT, **late)
    drv = G.Driver('toy', [[s, s]] * n, concurrency=min(n, 64), seed=seed, threads=2, groups=1)
    rows = _play(drv, _Rand(2))
    return rows, drv.stats()


def _py_selfplay(n, seed, impl, **late):
    s = NNSeat('m', impl=impl, **SEAT, **late)
    drv = PyDriver('toy', n, lambda g: [s, s], concurrency=min(n, 64), seed=seed)
    rows = _play(drv, _Rand(2))
    return rows, drv.late


def test_off_is_the_games_of_before():
    """temp_late 0 given explicitly: the same games as without the key, in both drivers; no counters."""
    a, st = _rust_selfplay(16, 3)
    b, st2 = _rust_selfplay(16, 3, temp_late=0.0)
    key = lambda rows: sorted((r['gid'], r['steps'], tuple(r['scores'])) for r in rows)   # noqa: E731
    assert key(a) == key(b) and 'late' not in st and 'late' not in st2
    pa, la = _py_selfplay(8, 3, 'py')
    pb, lb = _py_selfplay(8, 3, 'py', temp_late=0.0)
    assert key(pa) == key(pb) and la['decisions_full'] == la['decisions_fast'] == 0


def _rates(late):
    n = {k: late['decisions_' + k] for k in ('full', 'fast')}
    return {k: (late['deviations_' + k] / n[k], late['single_' + k] / n[k], n[k]) for k in n}


def test_python_and_rust_drivers_late_sample_alike():
    """Toy self-play at T 1 with a 0.02 gate (fast searches at 4 simulations, full at 16): the Python
    driver's reference trees ('py'), its Rust trees ('rs') and the Rust driver deviate as often and are
    gated as often, full and fast apart (statistically: their streams differ); every game ends. And the
    late-sampled games differ from the rule's."""
    late = dict(temp_late=1.0, temp_late_margin=0.02)
    rows, st = _rust_selfplay(160, 5, **late)
    assert len(rows) == 160 and all(r['over'] for r in rows)
    rs = _rates(st['late'])
    for impl in ('py', 'rs'):
        prow, pl = _py_selfplay(160, 7, impl, **late)
        assert len(prow) == 160 and all(r['over'] for r in prow)
        py = _rates(pl)
        for k in ('full', 'fast'):
            (a, sa, na), (b, sb, nb) = rs[k], py[k]
            se = math.sqrt(a * (1 - a) / na + b * (1 - b) / nb)
            assert abs(a - b) < 4 * se + 0.01, (impl, k, rs, py)
            se = math.sqrt(sa * (1 - sa) / na + sb * (1 - sb) / nb)
            assert abs(sa - sb) < 4 * se + 0.01, (impl, k, rs, py)
        print(impl, py, 'rust', rs)
    assert 0.05 < rs['full'][0] < 0.95 and 0.05 < rs['fast'][0] < 0.95
    # the counters' total = every searched decision after temp_moves
    assert sum(rs[k][2] for k in rs) == sum(max(0, r['decisions'] - 10) for r in rows)


def test_a_runs_settings_metrics_and_log(tmp_path):
    run = tmp_path / 'late'
    p = run_loop(run, '--iters', '2', '--temp-moves', '6', '--selfplay-temp-late', '0.7',
                 '--selfplay-temp-late-margin', '0.05')
    assert p.returncode == 0, p.stdout + p.stderr
    cfg = json.load(open(run / 'config.json'))
    assert (cfg['selfplay_temp_late'], cfg['selfplay_temp_late_margin']) == (0.7, 0.05)
    ms = [m for m in read_metrics(str(run)) if 'selfplay' in m]
    assert len(ms) == 2
    for m in ms:
        t = m['selfplay']['temp_late']
        assert t['decisions']['all'] == t['decisions']['full'] + t['decisions']['fast'] > 0
        assert 0 < t['rate']['all'] < 1 and t['per_game']['deviations'] > 0
        assert t['deviations']['all'] == round(t['rate']['all'] * t['decisions']['all'])
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert log.count('"temp_late": {"decisions"') == 2
    # off: no metrics key; a gate alone is refused
    plain = tmp_path / 'plain'
    assert run_loop(plain, '--iters', '1').returncode == 0
    assert all('temp_late' not in m.get('selfplay', {}) for m in read_metrics(str(plain)))
    p = run_loop(tmp_path / 'bad', '--iters', '1', '--selfplay-temp-late-margin', '0.05')
    assert p.returncode != 0 and 'a gate of selfplay_temp_late' in p.stdout + p.stderr


def test_replays_mark_late_samples(tmp_path):
    """--selfplay replays of a run with late sampling: every network decision after temp_moves is
    sampled (marked 'sampled' and 'late'); without it, those decisions are not sampled."""
    from zero import replay
    a, _ = _ckpts(tmp_path, {'spec': '', 'temp_moves': 4, 'selfplay_temp_late': 1.0})
    reps = replay.record(a, opp='self', games=2, sims=8, chance_k=0, seed=1, selfplay=True, log=lambda *x: None)
    a2, _ = _ckpts(tmp_path / 'plain', {'spec': '', 'temp_moves': 4})
    plain = replay.record(a2, opp='self', games=2, sims=8, chance_k=0, seed=1, selfplay=True, log=lambda *x: None)
    for r, late in ((reps, True), (plain, False)):
        for rep in r:
            nn = [f for f in rep['frames'] if f['k'] == 'nn']
            assert len(nn) > 4
            assert all(f['sampled'] and 'late' not in f for f in nn[:4])
            assert all(f['sampled'] is late and f.get('late', False) is late for f in nn[4:] if f['sims'] > 0)


def test_remote_selfplay_refuses_a_checkout_without_late_sampling(tmp_path):
    """A worker whose checkout has no late sampling (its hello says so: it would play the rule's moves in
    silence) is killed at its hello; this machine plays every game. A checkout with it plays its share."""
    run = tmp_path / 'sp'
    p = run_loop(run, '--iters', '1', '--conc', '64', '--games', '200', '--selfplay-remote', 'local',
                 '--selfplay-temp-late', '1', env=env_for(tmp_path, {'no_late': 1}))
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))[1]
    assert m['selfplay']['games'] == 200 and m['selfplay']['remote']['games'] == 0
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'REFUSED: its checkout plays late sampling as None' in log
    ok = tmp_path / 'ok'
    p = run_loop(ok, '--iters', '3', '--conc', '64', '--games', '200', '--selfplay-remote', 'local',
                 '--selfplay-temp-late', '1', env=env_for(tmp_path / 'm2'))
    assert p.returncode == 0, p.stdout + p.stderr
    assert sum(m['selfplay']['remote']['games'] for m in read_metrics(str(ok))[1:]) > 0
    assert 'REFUSED' not in (ok / 'log.txt').read_text(encoding='utf-8')
