"""The self-play league (zero/league.py, docs/az.md 14): a share of the self-play games against a
past checkpoint of the run. The games' kinds and tables (exact shares, slots and seats in turn); the
settings' checks; through the bindings, a league seat records nothing and plays the games of a
recorded one (only the current network's samples, z / q from its seat); toy runs with the league on
(classic, rolling, the shared queue with the remote 'local': the quota exact, the league's games on
both machines, the metrics, the past checkpoints shipped to the mirror; a worker without the league
refused); the picks (deterministic, uniform and pfsp); fork_run linking the league's checkpoints;
the default seat dicts and Run unchanged with the setting off."""
import glob
import json
import os
import sys

import numpy as np
import pytest

from zero import league as L
from zero.run import DEFAULTS, read_metrics

from test_loop import ROOT, run_loop
from test_remoteplay import check_quota, env_for, gone, remote_shards, worker_pids

POOL1 = ['--selfplay-league-pool', '1,2']


def cfg(**kw):
    return dict(DEFAULTS, **kw)


# ---------------------------------------------------------------- tables
def test_kinds_are_exact_spread_and_in_turn():
    c = cfg(selfplay_league=0.25, selfplay_league_slots=2)
    ks = L.kinds(c, 48, [2] * 48)
    lg = [k for k in ks if k is not None]
    assert len(lg) == 12                                   # floor(48 x 0.25)
    # spread: one league game in every 4 consecutive games
    assert all(sum(k is not None for k in ks[i:i + 4]) == 1 for i in range(0, 48, 4))
    # the slots in turn, each slot's past seat in turn
    assert [k[0] for k in lg] == [0, 1] * 6
    assert [k[1] for k in lg] == [0, 0, 1, 1] * 3
    for f, n in ((0.3, 1000), (0.2, 7), (1.0, 5), (0.01, 99)):
        assert sum(k is not None for k in L.kinds(cfg(selfplay_league=f), n, [2] * n)) == int(n * f + 1e-9), f
    # the rolling cycle holds every (slot, seat) its share exactly
    for f, k, players in ((0.25, 1, [2]), (0.3, 2, [2]), (0.5, 3, [2, 3])):
        c = cfg(selfplay_league=f, selfplay_league_slots=k, players=players)
        n = L.cycle(c)
        games, ks = L.rolling_table(c, {'key': 'm', 'selfplay': True})
        assert len(games) == len(ks) == n
        assert sum(x is not None for x in ks) == round(n * f)
        assert [len(g) for g in games] == [players[t % len(players)] for t in range(n)]
        per = {}
        for x in ks:
            if x is not None:
                per[x[0]] = per.get(x[0], 0) + 1
        assert sorted(per) == list(range(k)) and len(set(per.values())) == 1, per


def test_tables_seats():
    seat = {'key': 'm', 'sims': 8, 'selfplay': True}
    c = cfg(selfplay_league=0.5)
    games, ks = L.table(c, seat, [2, 2, 2, 2])
    assert ks == [None, (0, 0), None, (0, 1)]
    assert games[0] == [seat, seat]
    assert games[1] == [dict(seat, key='L0', record=False), seat]
    assert games[3] == [seat, dict(seat, key='L0', record=False)]
    assert 'record' not in seat                            # the run's seat is untouched


def test_settings_checked():
    L.check_settings(cfg())                                # off: nothing checked
    L.check_settings(cfg(selfplay_league=0.25))
    L.check_settings(cfg(selfplay_league=1.0, selfplay_league_pool='all', selfplay_league_pick='pfsp',
                         selfplay_league_slots=3))
    for bad in (dict(selfplay_league=1.5), dict(selfplay_league=-0.1), dict(selfplay_league=0.2, selfplay_league_pool='0,2'),
                dict(selfplay_league=0.2, selfplay_league_pool='x'), dict(selfplay_league=0.2, selfplay_league_pool='2,2'),
                dict(selfplay_league=0.2, selfplay_league_slots=0), dict(selfplay_league=0.2, selfplay_league_pick='best'),
                dict(selfplay_league=0.2, resolver='toy')):
        with pytest.raises(SystemExit):
            L.check_settings(cfg(**bad))
    assert L.parse_pool('') == L.parse_pool(L.DEFAULT_POOL) == (2, 4, 8, 16, 32)
    assert L.parse_pool('all') == 'all' and L.parse_pool('3, 1') == (3, 1)
    # the run's command line: a co-training run, or one whose resolver plays a league, is refused
    from zero import run as R
    with pytest.raises(SystemExit):
        R.resolve_config(R.parse(['--run', 'x', '--selfplay-league', '0.2', '--resolver', 'toy']), {})
    with pytest.raises(SystemExit):
        R.resolve_config(R.parse(['--run', 'x', '--resolver-cfg', '{"selfplay_league": 0.2}']), {})
    with pytest.raises(SystemExit):
        R.resolve_config(R.parse(['--run', 'x', '--selfplay-league', '0.2', '--selfplay-league-pool', 'nope']), {})
    c = R.resolve_config(R.parse(['--run', 'x', '--selfplay-league', '0.2', '--selfplay-league-pool', 'all']), {})
    assert c['selfplay_league'] == 0.2 and c['selfplay_league_pool'] == 'all'


def test_defaults_unchanged(tmp_path):
    """Off by default: no League, the self-play seat dict as before (no `record` key)."""
    from zero import run as R
    assert DEFAULTS['selfplay_league'] == 0.0 and DEFAULTS['selfplay_league_pool'] == L.DEFAULT_POOL
    r = R.Run(str(tmp_path), dict(DEFAULTS))
    assert r.league is None
    assert 'record' not in r.nn_seat('m', 8, selfplay=True)
    assert R.Run(str(tmp_path), cfg(selfplay_league=0.25)).league is not None


# ---------------------------------------------------------------- through the bindings
def _toy_run(tmp_path, n_ckpts=2, **kw):
    import torch  # noqa: F401
    from zero import run as R
    from zero.net import save
    c = cfg(game='toy', hidden=32, blocks=1, sims=8, sims_fast=4, p_full=0.5, conc=24, threads=2, **kw)
    os.makedirs(tmp_path / 'models', exist_ok=True)
    r = R.Run(str(tmp_path), c)
    for i in range(n_ckpts):
        c['seed'] = i                                      # different networks
        save(r.new_model(), R.model_path(str(tmp_path), i), {'iter': i})
    c['seed'] = 0
    return R.Run(str(tmp_path), c)


def _play(r, games, evs):
    from zero.infer import drive
    from zero.workqueue import split_games
    drv = r.driver(games, seed=5, record=True)
    rows, smp = [], []

    def take(lb=None):
        rr, s = drv.take_results(), drv.take_samples()
        if rr:
            rows.extend(split_games(rr, s))
    drive(drv, evs, on_batch=take)
    take()
    return sorted(rows, key=lambda x: x[0]['gid'])


def test_a_league_seat_records_nothing_and_plays_the_same_games(tmp_path):
    """The current network (checkpoint 1) against a past one (checkpoint 0) in half of the games:
    the league games' samples are the current network's decisions only - every sample's decider is
    its seat, z its result there - and, with the league seat recorded instead, the same games move for
    move whose current-seat samples are exactly these."""
    from zero.run import model_path
    r = _toy_run(tmp_path, selfplay_league=0.5, selfplay_league_pool='1')
    seat = r.nn_seat('m', 8, selfplay=True)
    plan = r.league.plan(2)
    assert plan['slots'] == [0] and plan['candidates'] == [0]
    evs = {'m': r.evaluator(model_path(str(tmp_path), 1))}
    r.league.evaluators(2, evs)
    assert set(evs) == {'m', 'L0'}
    games, ks = L.table(r.cfg, seat, [2] * 24)
    got = _play(r, games, evs)
    recorded = [[dict(s, record=True) if s.get('key') == 'L0' else s for s in g] for g in games]
    ref = _play(r, recorded, evs)
    assert len(got) == len(ref) == 24
    n_league = n_mirror = 0
    for (row, sd), (rrow, rsd), kd in zip(got, ref, ks):
        assert (row['gid'], row['steps'], row['scores'], row['result']) == (rrow['gid'], rrow['steps'], rrow['scores'], rrow['result'])
        if kd is None:                                     # a mirror game: both seats, as before
            n_mirror += 1
            assert set(sd['player'].tolist()) == {0, 1}
            for k in sd:
                assert np.array_equal(sd[k], rsd[k]), k
            continue
        n_league += 1
        me = 1 - kd[1]                                     # the current network's seat
        assert sd is not None and set(sd['player'].tolist()) == {me}
        assert set(rsd['player'].tolist()) == {0, 1}
        keep = np.flatnonzero(rsd['player'] == me)
        assert len(keep) == len(sd['player']) == row['samples'] < rrow['samples']
        for k in ('x', 'player', 'n_players', 'z', 'q', 'sc', 'net'):
            assert np.array_equal(sd[k], rsd[k][keep]), k
        assert np.allclose(sd['z'][:, 0], row['result'][me])
    assert n_league == 12 and n_mirror == 12
    # the attribution of the run's metrics: the current network's score per league game
    rows = L.League.attribute([row for row, _ in got], ks, plan['slots'])
    t = L.tally(rows)
    assert list(t) == ['iter_0000'] and t['iter_0000'][1] == 12
    assert t['iter_0000'][0] == sum(row['result'][1 - kd[1]] for (row, _), kd in zip(got, ks) if kd is not None)


def test_the_past_network_is_guarded(tmp_path):
    """A past checkpoint of another encoding is refused (zero.net's encoding guard)."""
    import torch
    from zero.net import EncodingMismatch
    from zero.run import model_path
    r = _toy_run(tmp_path, selfplay_league=0.5, selfplay_league_pool='1')
    ck = torch.load(model_path(str(tmp_path), 0), weights_only=False)
    ck['config']['features'] = ck['config']['features'] - 1
    torch.save(ck, model_path(str(tmp_path), 0))
    with pytest.raises(EncodingMismatch):
        r.league.evaluators(2, {})


# ---------------------------------------------------------------- picks
def test_plans_are_deterministic_and_follow_the_pool(tmp_path):
    from zero.run import model_path
    from zero.net import save
    r = _toy_run(tmp_path, n_ckpts=1, selfplay_league=0.25, selfplay_league_pool='1,2,4', selfplay_league_slots=2)
    m = r.new_model()
    for i in range(1, 12):
        save(m, model_path(str(tmp_path), i), {'iter': i})
    assert r.league.plan(1) == {'candidates': [], 'slots': [0, 0], 'self': True}
    assert r.league.plan(2) == {'candidates': [0], 'slots': [0, 0]}
    assert r.league.plan(4)['candidates'] == [2, 1] and sorted(r.league.plan(4)['slots']) == [1, 2]
    p = r.league.plan(12)
    assert p['candidates'] == [10, 9, 7] and len(set(p['slots'])) == 2 and set(p['slots']) <= {10, 9, 7}
    from zero import run as R
    again = R.Run(str(tmp_path), r.cfg).league
    assert [again.plan(i) for i in range(1, 12)] == [r.league.plan(i) for i in range(1, 12)]
    assert r.league.files(12) == {'L0': f'models/iter_{p["slots"][0]:04d}.pt', 'L1': f'models/iter_{p["slots"][1]:04d}.pt'}
    # 'all': every earlier checkpoint
    a = R.Run(str(tmp_path), dict(r.cfg, selfplay_league_pool='all')).league
    assert a.plan(12)['candidates'] == list(range(10, -1, -1))
    # uniform: every candidate picked about as often
    u = R.Run(str(tmp_path), dict(r.cfg, selfplay_league_pool='1,2,3,4,5,6,7,8', selfplay_league_slots=1)).league
    seen = {}
    for i in range(10, 400):
        u.candidates = lambda it: [9, 8, 7, 6]
        j = u.plan(i)['slots'][0]
        seen[j] = seen.get(j, 0) + 1
    assert set(seen) == {9, 8, 7, 6} and min(seen.values()) > 70, seen


def test_pfsp_prefers_the_checkpoints_that_still_win(tmp_path):
    """pick 'pfsp': weight (1 - x)^2, x the current networks' decayed score against each checkpoint
    from the metrics rows of iterations <= it - 2 (unmeasured: 0.5)."""
    r = _toy_run(tmp_path, n_ckpts=1, selfplay_league=0.25, selfplay_league_pick='pfsp')
    rows = [{'iter': 8, 'selfplay': {'league': {'pool': {'iter_0001': {'games': 100, 'score': 90.0},
                                                          'iter_0002': {'games': 100, 'score': 40.0}}}}},
            {'iter': 9, 'selfplay': {'league': {'pool': {'iter_0003': {'games': 100, 'score': 99.0}}}}}]
    with open(tmp_path / 'metrics.jsonl', 'w') as f:
        for m in rows:
            f.write(json.dumps(m) + '\n')
    w = r.league.weights([1, 2, 3, 4], 10)                 # (rows <= 8: iteration 9's not yet)
    x1, x2 = (90 + 2) / (100 + 4), (40 + 2) / (100 + 4)
    assert np.allclose(w, [(1 - x1) ** 2, (1 - x2) ** 2, 0.25, 0.25])
    h = r.league.history(9)
    assert np.isclose(h['iter_0001'][1], 100 * L.DECAY) and np.isclose(h['iter_0003'][1], 100)
    lg = r.league
    lg.candidates = lambda it: [1, 2, 4]
    picks = {}
    for seed in range(300):                                # P = 0.02, 0.57, 0.40
        lg.seed, lg.plans = seed, {}
        j = lg.plan(10)['slots'][0]
        picks[j] = picks.get(j, 0) + 1
    assert picks[2] > picks[4] > picks.get(1, 0), picks
    assert set(lg.plan(10)['weights']) == {'iter_0001', 'iter_0002', 'iter_0004'}


# ---------------------------------------------------------------- runs
@pytest.fixture(scope='module')
def league_runs(tmp_path_factory):
    base = tmp_path_factory.mktemp('league')
    out = {}
    for name, extra in (('classic', []), ('rolling', ['--rolling', '1', '--selfplay-league-slots', '2'])):
        run = base / name
        p = run_loop(run, '--iters', '3', '--games', '48', '--conc', '48', '--selfplay-league', '0.25', *POOL1, *extra)
        assert p.returncode == 0, p.stdout + p.stderr
        out[name] = run
    return out


def test_league_runs_metrics(league_runs):
    for name, run in league_runs.items():
        m = read_metrics(str(run))
        assert [r['iter'] for r in m] == [0, 1, 2, 3]
        for r in m[1:]:
            sp, lg = r['selfplay'], r['selfplay']['league']
            assert lg['games'] == sum(v['games'] for v in lg['pool'].values()) == lg['local'] and lg['remote'] == 0
            assert lg['share'] == round(lg['games'] / sp['games'], 4)
            if name == 'classic':
                assert lg['games'] == 12                   # floor(48 x 0.25), exactly
            else:
                assert 8 <= lg['games'] <= 18              # the finished games (the started ones: exactly)
            assert 0 <= lg['win'] <= 1 and set(lg['slots']) == ({'L0'} if name == 'classic' else {'L0', 'L1'})
            assert set(lg['pool']) <= set(lg['slots'].values())
        assert m[1]['selfplay']['league'].get('self') and m[1]['selfplay']['league']['candidates'] == []
        assert m[2]['selfplay']['league']['candidates'] == ['iter_0000'] and 'self' not in m[2]['selfplay']['league']
        assert m[3]['selfplay']['league']['candidates'] == ['iter_0001', 'iter_0000']
        log = (run / 'log.txt').read_text(encoding='utf-8')
        assert 'iter 3 league: ' in log and 'the current network scores' in log
        # every sample in the shards is one of the games' (the league games' current side only)
        for r in m[1:]:
            n = 0
            for f in glob.glob(str(run / 'data' / f'iter_{r["iter"]:04d}' / '*.npz')):
                with np.load(f) as z:
                    n += len(z['player'])
            assert n == r['selfplay']['samples']


def test_a_mirror_run_has_no_league(tmp_path):
    p = run_loop(tmp_path / 'r', '--iters', '1')
    assert p.returncode == 0, p.stderr
    m = read_metrics(str(tmp_path / 'r'))
    assert 'league' not in m[1]['selfplay'] and ' league: ' not in (tmp_path / 'r' / 'log.txt').read_text(encoding='utf-8')


def test_league_with_remote_selfplay(tmp_path):
    """The shared queue with the remote 'local' and the league on: the quota exact, league games on
    both machines (the worker's table and slots the run's: the phase names the slots' checkpoints,
    shipped into the mirror and kept there by the worker's clean-up), the remote's tallies in the
    metrics, lingering through training; no worker outlives the run."""
    run = tmp_path / 'sp'
    p = run_loop(run, '--iters', '4', '--conc', '64', '--games', '600', '--selfplay-remote', 'local',
                 '--selfplay-remote-linger', '1', '--selfplay-league', '0.25', '--selfplay-league-pool', '1,3',
                 env=env_for(tmp_path))
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    check_quota(m, 600)
    assert any(r['selfplay']['league']['remote'] > 0 for r in m[1:]), [r['selfplay']['league'] for r in m[1:]]
    for r in m[1:]:
        lg, sp = r['selfplay']['league'], r['selfplay']
        assert lg['local'] + lg['remote'] == lg['games'] == sum(v['games'] for v in lg['pool'].values())
        assert 0.18 < lg['share'] < 0.32, lg
        assert 'league' not in sp['remote']                # (moved into selfplay.league)
        assert len(remote_shards(run, r['iter'])) == sp['remote']['shards']
    assert m[4]['selfplay']['league']['candidates'] == ['iter_0002', 'iter_0000']
    rd = tmp_path / 'mirror' / 'sp'
    assert json.load(open(rd / 'config.json'))['selfplay_league'] == 0.25
    # the mirror after phase 4: checkpoints >= 1 and the phase's slot (iter_0000 only if named)
    left = sorted(os.path.basename(p) for p in glob.glob(str(rd / 'models' / 'iter_*.pt')))
    slot = m[4]['selfplay']['league']['slots']['L0']
    assert left == sorted({'iter_0001.pt', 'iter_0002.pt', 'iter_0003.pt', f'{slot}.pt'}), (left, slot)
    assert gone(worker_pids(run))


def test_a_worker_without_the_league_is_refused(tmp_path):
    """A worker whose checkout has no league (it would play mirror games and record both seats) is
    killed at its hello; this machine plays every game, the league's share as without a remote."""
    run = tmp_path / 'sp'
    p = run_loop(run, '--iters', '2', '--conc', '64', '--games', '200', '--selfplay-remote', 'local',
                 '--selfplay-league', '0.25', *POOL1, env=env_for(tmp_path, {'no_league': 1}))
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    for r in m[1:]:
        assert r['selfplay']['games'] == 200 and r['selfplay']['remote']['games'] == 0
        assert r['selfplay']['league']['remote'] == 0 and r['selfplay']['league']['games'] == r['selfplay']['league']['local']
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'REFUSED: its checkout has no self-play league' in log


def test_fork_links_the_leagues_checkpoints(league_runs, tmp_path):
    """fork_run --set selfplay_league=...: the fork gets every checkpoint its league may play (back
    to the pool's furthest offset), not only the Elo matches'."""
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import fork_run
    src = league_runs['classic']
    plain, lg = tmp_path / 'plain', tmp_path / 'lg'
    assert fork_run.main(['--source', str(src), '--dest', str(plain), '--iter', '3', '--set', 'selfplay_league=0',
                          '--set', 'elo_offsets="1"']) == 0
    assert fork_run.main(['--source', str(src), '--dest', str(lg), '--iter', '3', '--set', 'elo_offsets="1"',
                          '--set', 'selfplay_league_pool="3"']) == 0
    have = lambda d: sorted(os.path.basename(p) for p in glob.glob(str(d / 'models' / 'iter_*.pt')))   # noqa: E731
    assert have(plain) == ['iter_0001.pt', 'iter_0002.pt', 'iter_0003.pt']     # (the Elo matches')
    assert have(lg) == ['iter_0000.pt', 'iter_0001.pt', 'iter_0002.pt', 'iter_0003.pt']
    assert L.fork_needs(dict(DEFAULTS, selfplay_league=0.2, selfplay_league_pool='2,8'), 20) == set(range(12, 21))
    assert L.fork_needs(dict(DEFAULTS, selfplay_league=0.2, selfplay_league_pool='all'), 5) == set(range(6))
    assert L.fork_needs(dict(DEFAULTS), 20) == set()
    p = run_loop(lg, '--iters', '4')
    assert p.returncode == 0, p.stderr
    assert read_metrics(str(lg))[-1]['selfplay']['league']['candidates'] == ['iter_0000']
