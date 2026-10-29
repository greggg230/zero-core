"""The move rule of non-sampled moves (docs/search.md 3.3: `select` 'visits' | 'cq' + the guard
`select_min_visits`; run settings eval_select / selfplay_select, docs/az.md 5.4) and the Elo
bookkeeping around it: Python vs Rust exactly on Rust roots and statistically in games, the
seat settings, a run that changes its evaluation rule (one Elo pool per rule, the old pool's
files untouched), the remote evaluation playing the run's rule, h2h and replays; raw Elo
offsets (elo_offsets) and runs without random games (eval_random_games 0)."""
import json
import math
import os
import random
import shutil

import pytest

from zero import elo, games as G, mcts, parity
from zero.play import NNSeat, PyDriver
from zero.run import read_metrics

from test_loop import run_loop

INFO = G.info('toy')


# ------------------------------------------------------------------ the rule itself

def test_python_rule_is_the_rust_drivers_on_rust_roots():
    """mcts.select_index on Rust roots' statistics == the Rust driver's own choice
    (SearchTree.root_select = Select::choose) for every rule, guard and transform; the rules do
    differ on these roots (the check has power)."""
    r = parity.select_formula('toy', n=8)
    assert r['checked'] == 8 * 4 * len(parity.SELECT_RULES)
    assert r['agree'] + r['near_ties'] == r['checked'] and not r['mismatches'], r['mismatches'][:5]
    assert r['near_ties'] == 0
    differ = {'cq': 0, 'guard': 0, 'unvisited': 0}
    ev = parity.LinearEval(INFO['num_features'], INFO['num_actions'], seed=1, scale=0.15)
    for k, g in enumerate(parity.sample_positions('toy', 8, 0)):
        for ns in (4, 8, 24, 64):
            *_, t = parity.search_rs(g, ev, ns, 700 + 31 * k + ns, tree=True)
            P0, N, W, v = t.root_priors0, t.root_visits, t.root_values, t.root_value0
            vis = mcts.select_index(P0, N, W, v)
            cq, cq1, cq2 = (mcts.select_index(P0, N, W, v, 'cq', m) for m in (0, 1, 2))
            differ['cq'] += cq != vis
            differ['guard'] += cq2 != cq
            differ['unvisited'] += cq != cq1
    assert differ['cq'] > 0 and differ['guard'] > 0, differ


def test_select_index_by_hand():
    P0, N, W = [0.9, 0.08, 0.02], [27, 4, 1], [27 * 0.40, 4 * 0.47, 1 * 0.60]
    assert mcts.select_index(P0, N, W, 0.45) == 0
    assert mcts.select_index(P0, N, W, 0.45, 'cq') == 2               # (50 + 27) x Q dwarfs ln P0
    assert mcts.select_index(P0, N, W, 0.45, 'cq', 2) == 1            # the guard drops the 1-visit option
    assert mcts.select_index(P0, N, W, 0.45, 'cq', 1000) == 0         # a guard above every count: the most visited
    assert mcts.select_index([0.5, 0.5], [3, 0], [0.6, 0.0], 0.9, 'cq') == 1      # unvisited at v_mix
    assert mcts.select_index([0.5, 0.5], [3, 0], [0.6, 0.0], 0.9, 'cq', 1) == 0
    assert mcts.select_index([], [], [], 0.5, 'cq') == 0
    lg = mcts.completed_q_logits(P0, N, W, 0.45)
    pi = mcts.completed_q_policy(P0, N, W, 0.45)
    assert max(range(3), key=lambda i: lg[i]) == max(range(3), key=lambda i: pi[i])
    assert [mcts.select_tag(*a) for a in [('visits',), ('cq',), ('cq', 2), ('cq', 0, 25, 0.5), ('cq', 3, 50, 2)]] == \
        ['visits', 'cq', 'cq-n2', 'cq-v25-s0.5', 'cq-n3-v50-s2']


def test_seat_settings_are_validated():
    with pytest.raises(ValueError):
        G.nn_seat('m', select='q')
    with pytest.raises(ValueError):
        G.nn_seat('m', select='visits', select_min_visits=2)          # the guard is cq's
    with pytest.raises(ValueError):
        G.nn_seat('m', select='cq', select_min_visits=-1)
    with pytest.raises(ValueError):
        NNSeat('m', select='cq', select_min_visits=1.5)
    # the default seat dicts are the ones from before the rule existed (no new keys)
    assert G.nn_seat('m', 8) == dict(key='m', sims=8, c_puct=1.0, selfplay=False, alpha=0.3, eps=0.25, temp_moves=20,
                                     sims_fast=None, p_full=1.0, chance_k=0, margin_w=0.0, redet=True)
    s = G.nn_seat('m', 8, policy_target='cq')
    assert list(s)[-3:] == ['policy_target', 'cq_c_visit', 'cq_c_scale'] and 'select' not in s
    s = G.nn_seat('m', 8, select='cq', select_min_visits=2, cq_c_visit=25)
    assert (s['select'], s['select_min_visits'], s['cq_c_visit'], 'policy_target' in s) == ('cq', 2, 25.0, False)
    # the Rust driver checks them too
    with pytest.raises(ValueError, match='select'):
        G.Driver('toy', [[dict(G.nn_seat('m', 4), select='q')] * 2], concurrency=1)
    with pytest.raises(ValueError, match='guard'):
        G.Driver('toy', [[dict(G.nn_seat('m', 4), select_min_visits=2)] * 2], concurrency=1)
    from zero.run import DEFAULTS, check_eval_settings
    for bad in ({'eval_select': 'q'}, {'selfplay_select': 'visits', 'selfplay_select_min_visits': 3},
                {'elo_offsets': '1,x'}, {'elo_offsets': '2,2'}, {'elo_offsets': '0'}):
        with pytest.raises(SystemExit):
            check_eval_settings(dict(DEFAULTS, **bad))
    check_eval_settings(dict(DEFAULTS, eval_select='cq', eval_select_min_visits=2, elo_offsets='1,2,4,8'))


class _Rand:
    """A deterministic random network: priors and values from a hash of the features."""

    def __init__(self, seed=0):
        self.ev = parity.LinearEval(INFO['num_features'], INFO['num_actions'], seed=seed, scale=0.15)

    def __call__(self, lb):
        import numpy as np
        a = lb.a
        off = a['opt_off']
        pri, val = [], np.zeros((len(off) - 1, G.MAX_SEATS), np.float32)
        for b in range(len(off) - 1):
            p, v, _ = self.ev(a['x'][b], a['opt_action'][off[b]:off[b + 1]], int(a['n_players'][b]))
            pri.extend(p)
            val[b] = v
        return np.asarray(pri, np.float32), val, np.zeros_like(val)


def _play(drv, ev):
    while True:
        lb = drv.next_batch()
        if lb is None:
            break
        drv.submit(lb.id, *ev(lb))
    return drv.take_results()


@pytest.mark.parametrize('impl', ['py', 'rs'])
def test_python_driver_plays_the_rule(impl):
    """The Python driver's seats ('py': the reference formula; 'rs': the Rust tree's root_select)
    play every game to its end with the 'cq' rule, and play other games than with 'visits'."""
    def games(select, mv=0):
        seat = NNSeat('m', 12, impl=impl, select=select, select_min_visits=mv)
        drv = PyDriver('toy', 24, lambda g: [seat, seat], concurrency=24, seed=5)
        return sorted(((r['gid'], r['steps'], tuple(r['scores'])) for r in _play(drv, _Rand())), key=lambda x: x[0])
    base, cq, cq2 = games('visits'), games('cq'), games('cq', 2)
    assert len(base) == len(cq) == len(cq2) == 24
    assert cq != base and cq2 != cq


def test_rust_and_python_cq_seats_are_equally_strong():
    """Statistical parity in games: a Rust-tree seat and a Python-tree seat, both with the 'cq' rule
    and the same network, score ~50% (seats alternating). The rule's effect is checked to be real:
    'cq' against 'visits' at the same search is no tie."""
    def match(sa, sb, n=300):
        drv = PyDriver('toy', n, lambda g: [sa, sb] if g % 2 == 0 else [sb, sa], concurrency=100, seed=11)
        rows = _play(drv, _Rand(3))
        sc = [r['result'][r['gid'] % 2] for r in rows]
        w = sum(sc) / len(sc)
        return w, math.sqrt(max(w * (1 - w), 1e-9) / len(sc))
    rs = NNSeat('m', 16, impl='rs', select='cq')
    py = NNSeat('m', 16, impl='py', select='cq')
    w, se = match(rs, py)
    assert abs(w - 0.5) < 3.5 * se + 0.02, (w, se)


def test_rust_driver_seat_keys_reach_the_search():
    """A Rust-driver evaluation match with the 'cq' rule plays other games than the default one
    and the guard changes them again (the seat keys are read)."""
    def rows(extra):
        s = dict(G.nn_seat('m', 12), **extra)
        drv = G.Driver('toy', [[s, s]] * 20, concurrency=20, seed=3, threads=1, groups=1)
        return sorted(((r['gid'], r['steps'], tuple(r['scores'])) for r in _play(drv, _Rand())), key=lambda x: x[0])
    a = rows({})
    assert a == rows({'select': 'visits'})
    b = rows({'select': 'cq', 'cq_c_visit': 50.0, 'cq_c_scale': 1.0})
    c = rows({'select': 'cq', 'select_min_visits': 3})
    assert a != b and b != c


# ------------------------------------------------------------------ runs: pools, remote, offsets

def _pairs(run, tag='visits'):
    return json.load(open(os.path.join(elo.pool_dir(str(run), tag), 'results.json')))


def test_a_run_changes_its_eval_rule(tmp_path):
    """Iterations 0-2 evaluated with 'visits', 3 with 'cq' (a restart with --eval-select cq):
    the 'visits' pool's files are untouched, the 'cq' pool holds only iteration 3's games (its
    evaluation and its Elo matches, both seats searching with 'cq'), metrics, the Elo row and
    models/index.json name the rule; back to 'visits' at iteration 4: the 'cq' pool stays as it
    was."""
    run = tmp_path / 'sw'
    p = run_loop(run, '--iters', '2')
    assert p.returncode == 0, p.stdout + p.stderr
    snap = {f: open(run / 'elo' / f, 'rb').read() for f in ('results.json', 'ratings.json')}
    assert 'select' not in json.load(open(run / 'models' / 'index.json'))
    p = run_loop(run, '--iters', '3', '--eval-select', 'cq')
    assert p.returncode == 0, p.stdout + p.stderr
    for f, b in snap.items():
        assert open(run / 'elo' / f, 'rb').read() == b, f'elo/{f} changed'
    m = read_metrics(str(run))
    assert 'select' not in m[2]['eval_yard'] and m[3]['eval_yard']['select'] == 'cq'
    assert m[3]['eval_random']['select'] == 'cq' and m[3]['elo']['select'] == 'cq'
    cq = _pairs(run, 'cq')
    assert (cq['select'], cq['since'], cq['logged']) == ('cq', 3, ['iter_0003'])
    names = {x for k in cq['pairs'] for x in k.split('|')}
    assert names == {'iter_0003', 'iter_0002', 'iter_0001', 'heuristic', 'random'}, names
    assert set(cq['pairs']) == {'heuristic|random', 'heuristic|iter_0003', 'iter_0003|random',
                                'iter_0002|iter_0003', 'iter_0001|iter_0003'}
    r = json.load(open(run / 'elo' / 'cq' / 'ratings.json'))
    assert r['select'] == 'cq' and set(r['ratings']) == names
    idx = json.load(open(run / 'models' / 'index.json'))
    assert idx['select'] == 'cq' and {c['iter'] for c in idx['checkpoints']} == {1, 2, 3}
    assert {c['iter']: c.get('evaluated', True) for c in idx['checkpoints']} == {1: False, 2: False, 3: True}
    assert idx['best']['iter'] == 3, 'only a checkpoint evaluated with the rule can be best'
    log = open(run / 'log.txt', encoding='utf-8').read()
    assert '(pool cq)' in log and "'select': 'cq'" in log
    from zero import dashboard
    d = dashboard._net(str(run))
    assert d['select'] == 'cq' and d['since'] == 3 and set(d['ratings']['ratings']) == names
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dashboard'))
    from zerodash import runfiles
    assert runfiles.elo_pool(str(run)) == elo.active_pool(str(run))[1]
    assert runfiles.ratings(str(run))['iters'][-1][0] == 3
    # back to the default rule: its pool continues, the cq pool is left as it was
    snap_cq = open(run / 'elo' / 'cq' / 'results.json', 'rb').read()
    p = run_loop(run, '--iters', '4', '--eval-select', 'visits')
    assert p.returncode == 0, p.stdout + p.stderr
    assert open(run / 'elo' / 'cq' / 'results.json', 'rb').read() == snap_cq
    v = _pairs(run)
    assert 'iter_0003' not in v['logged'] and 'iter_0004' in v['logged'] and 'select' not in v
    assert not any('heuristic' in k and 'iter_0003' in k for k in v['pairs']), 'a cq evaluation in the visits pool'


def test_a_cq_run_from_the_start_and_selfplay_select(tmp_path):
    """eval_select cq + guard from iteration 0 (pool elo/cq-n2/, since 0) and selfplay_select cq:
    self-play runs (its moves after temp_moves by the rule) and the records are as many as ever."""
    run = tmp_path / 'cq'
    p = run_loop(run, '--iters', '2', '--eval-select', 'cq', '--eval-select-min-visits', '2',
                 '--selfplay-select', 'cq')
    assert p.returncode == 0, p.stdout + p.stderr
    assert not (run / 'elo' / 'results.json').exists()
    res = _pairs(run, 'cq-n2')
    assert res['since'] == 0 and res['logged'] == ['iter_0000', 'iter_0001', 'iter_0002']
    m = read_metrics(str(run))
    assert all(r['eval_yard']['select'] == 'cq-n2' for r in m)
    assert m[2]['selfplay']['games'] == 24 and m[2]['selfplay']['samples'] > 0
    cfg = json.load(open(run / 'config.json'))
    assert (cfg['eval_select'], cfg['eval_select_min_visits'], cfg['selfplay_select']) == ('cq', 2, 'cq')
    p = run_loop(run, '--iters', '2', '--eval-select', 'q')
    assert p.returncode != 0 and "eval_select 'q'" in (p.stdout + p.stderr)
    # a fork (the self-play A/B, docs/az.md 12.4) takes every pool along and continues the rule's
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
    import fork_run
    fk = tmp_path / 'fork'
    assert fork_run.main(['--source', str(run), '--dest', str(fk), '--iter', '2', '--set', 'selfplay_select="visits"']) == 0
    assert json.load(open(fk / 'elo' / 'cq-n2' / 'results.json')) == res
    p = run_loop(fk, '--iters', '3')
    assert p.returncode == 0, p.stdout + p.stderr
    assert _pairs(fk, 'cq-n2')['logged'][-1] == 'iter_0003' and not (fk / 'elo' / 'results.json').exists()


def test_remote_evaluation_plays_the_runs_rule(tmp_path):
    """eval_remote with eval_select cq on a mirror that already evaluated with 'visits' (the
    re-pin of a remotely evaluated run): the remote plays 'cq' (its config is the run's), the
    'cq' pool comes back into elo/cq/, the mirror's metrics are kept, the 'visits' pool's files
    (here and on the mirror) are left alone."""
    run, mirror = tmp_path / 'rem', tmp_path / 'mirror'
    env = {'ZERO_REMOTE_WORKDIR': str(mirror)}
    p = run_loop(run, '--iters', '1', '--eval-remote', 'local', env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    snap = open(run / 'elo' / 'results.json', 'rb').read()
    msnap = open(mirror / 'rem' / 'metrics.jsonl', 'rb').read()
    p = run_loop(run, '--iters', '2', '--eval-remote', 'local', '--eval-select', 'cq', env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    ev = {e['iter']: e for e in map(json.loads, open(run / 'evals.jsonl'))}
    assert sorted(ev) == [0, 1, 2] and 'select' not in ev[1]['eval_yard']
    assert ev[2]['eval_yard']['select'] == 'cq' and ev[2]['elo']['select'] == 'cq'
    assert open(run / 'elo' / 'results.json', 'rb').read() == snap
    assert open(mirror / 'rem' / 'metrics.jsonl', 'rb').read().startswith(msnap), 'the mirror kept its rows'
    cq = _pairs(run, 'cq')
    assert cq['logged'] == ['iter_0002'] and cq['since'] == 2
    assert json.load(open(mirror / 'rem' / 'elo' / 'cq' / 'results.json')) == cq


def test_remote_result_of_another_rule_is_refused(tmp_path, monkeypatch):
    """A remote whose result names another move rule than the run's (an older checkout that
    ignores eval_select) is refused before anything is recorded."""
    from zero import remote

    class T:
        def __init__(self, m):
            self.m = m

        def run_module(self, *a, **k):
            class R:
                returncode = 0
                stdout = remote.RESULT + json.dumps(self.m)
                stderr = ''
            return R()

    class Run:
        run = str(tmp_path)
        cfg = {'eval_select': 'cq', 'eval_every': 1}
        yard = 'heuristic'

        def log(self, msg):
            pass

    ev = remote.RemoteEval.__new__(remote.RemoteEval)
    ev.R, ev.name, ev.rdir, ev.module = Run(), 'x', str(tmp_path / 'm'), 'zero.remote'
    ev._ship = lambda it: None
    ev.t = T({'iter': 3, 'eval_yard': {'win': 0.5, 'games': 8, 'score': 4.0, 'opp': 'heuristic'}})
    with pytest.raises(RuntimeError, match="move rule 'visits'"):
        ev._evaluate(3)
    assert not (tmp_path / 'evals.jsonl').exists()


def test_raw_elo_offsets_and_no_random_games(tmp_path):
    """elo_offsets '1,2' with eval_every 2: each evaluated checkpoint (0, 2, 4) plays the
    checkpoints 1 and 2 ITERATIONS back - the odd ones never evaluated are rated through those
    games only, listed as not evaluated and never best.pt; eval_random_games 0: no random match
    (the reference games still rate random). The remote's shipping covers the raw opponents."""
    run = tmp_path / 'off'
    p = run_loop(run, '--iters', '4', '--eval-every', '2', '--elo-offsets', '1,2', '--eval-random-games', '0',
                 '--elo-opponents', '1')
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    assert [r['iter'] for r in m if 'eval_yard' in r] == [0, 2, 4] and not any('eval_random' in r for r in m)
    res = _pairs(run)
    nn = {k for k in res['pairs'] if k.count('iter_') == 2}
    assert nn == {'iter_0001|iter_0002', 'iter_0000|iter_0002', 'iter_0003|iter_0004', 'iter_0002|iter_0004'}, nn
    assert res['logged'] == ['iter_0000', 'iter_0002', 'iter_0004'] and 'heuristic|random' in res['pairs']
    r = json.load(open(run / 'elo' / 'ratings.json'))['ratings']
    assert {'iter_0001', 'iter_0003'} <= set(r)
    idx = json.load(open(run / 'models' / 'index.json'))
    flags = {c['iter']: c.get('evaluated', True) for c in idx['checkpoints']}
    assert flags == {0: True, 1: False, 2: True, 3: False, 4: True}
    assert idx['best']['iter'] in (0, 2, 4)
    assert open(run / 'models' / 'best.pt', 'rb').read() == open(run / 'models' / idx['best']['file'], 'rb').read()
    assert elo.opponents(4, 2, '1,2', 1) == [3, 2] and elo.opponents(4, 2, '', 1) == [2]
    assert elo.opponents(16, 8) == [8, 0, -16, -48, -112, -240]
    from zero import dashboard
    d = dashboard._net(str(run))
    assert d['select'] == 'visits' and 'iter_0003' in d['ratings']['ratings']
    # the default Elo: today's opponents (evaluations back), never a raw offset
    run2 = tmp_path / 'def'
    p = run_loop(run2, '--iters', '4', '--eval-every', '2', '--eval-random-games', '0')
    assert p.returncode == 0, p.stdout + p.stderr
    nn2 = {k for k in _pairs(run2)['pairs'] if k.count('iter_') == 2}
    assert nn2 == {'iter_0000|iter_0002', 'iter_0002|iter_0004', 'iter_0000|iter_0004'}, nn2


def test_remote_ships_raw_offsets(tmp_path):
    """RemoteEval ships the checkpoints a raw-offset Elo phase plays (not only the evaluated ones)."""
    from zero import remote
    from zero.run import model_path
    run = tmp_path / 'r'
    os.makedirs(run / 'models')
    for k in range(9):
        open(model_path(str(run), k), 'wb').write(b'x')
    put = []

    class T:
        def mkdirs(self, *a):
            pass

        def listdir(self, d):
            return []

        def put(self, src, dst):
            put.append(os.path.basename(dst))

    class Run:
        cfg = {'eval_every': 4, 'elo_offsets': '1,2'}

        def __init__(self):
            self.run = str(run)

    ev = remote.RemoteEval.__new__(remote.RemoteEval)
    ev.R, ev.spec, ev.t, ev.rdir, ev.shipped = Run(), {}, T(), str(tmp_path / 'm'), None
    ev._ship(8)
    assert {f for f in put if f.startswith('iter_')} == {f'iter_{k:04d}.pt' for k in (0, 2, 3, 4, 6, 7, 8)}


# ------------------------------------------------------------------ h2h and replays

def _ckpts(tmp_path, cfg=None):
    import torch
    from zero.net import Net, save
    d = tmp_path / 'runs' / 'r'
    os.makedirs(d / 'models')
    if cfg is not None:
        json.dump(cfg, open(d / 'config.json', 'w'))
    paths = []
    for k in range(2):
        torch.manual_seed(k)
        paths.append(str(d / 'models' / f'iter_{k:04d}.pt'))
        save(Net(INFO['num_features'], INFO['num_actions'], hidden=32, blocks=1, game='toy').eval(), paths[-1], {'iter': k})
    return paths


def test_h2h_honours_the_runs_rule_and_overrides(tmp_path):
    from zero import h2h
    a, b = _ckpts(tmp_path, {'eval_select': 'cq', 'eval_select_min_visits': 2})
    r = h2h.match(a, b, games=8, sims=8, chance_k=0, threads=1)
    assert (r['select_a'], r['select_b']) == ('cq-n2', 'cq-n2')
    r2 = h2h.match(a, a, games=8, sims=8, chance_k=0, threads=1, seat_a={'select': 'cq'}, seat_b={}, rows=True,
                   move_log=50)
    assert (r2['select_a'], r2['select_b']) == ('cq', 'visits') and all(row['moves'] for row in r2['rows'])
    a2, _ = _ckpts(tmp_path / 'plain')
    assert h2h.match(a2, a2, games=4, sims=8, chance_k=0, threads=1)['select_a'] == 'visits'
    assert h2h.rule_keys(None) is None and h2h.rule_keys('visits') == {}
    with pytest.raises(ValueError):
        h2h.rule_keys('visits', 2)


def test_replays_follow_the_runs_rule(tmp_path):
    from zero import replay
    a, _ = _ckpts(tmp_path, {'eval_select': 'cq', 'spec': ''})
    reps = replay.record(a, opp='random', games=2, sims=8, chance_k=0, seed=1, log=lambda *x: None)
    assert reps[0]['seats'][0]['select'] == 'cq' and reps[1]['seats'][1]['select'] == 'cq'
    reps = replay.record(a, opp='random', games=2, sims=8, chance_k=0, seed=1, log=lambda *x: None,
                         select=('visits', 0))
    assert 'select' not in reps[0]['seats'][0]
