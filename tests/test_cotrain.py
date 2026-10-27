"""Co-training (zero/cotrain.py, docs/az.md 11) end to end on the toys: toy_team events (2v2) whose
matches are toy games played by the toy network, both networks trained in one run. Iterations,
the two directories, exact event quotas, progress, evaluation and Elo of both, resume mid
iteration and after the checkpoints, the non-finite guard for each net, resolver_init from a
single-game checkpoint, fork_run copying both nets, settings validation."""
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time

import pytest
import torch

from zero import runctl
from zero.run import latest_iter, read_metrics

from test_loop import PYPATH, ROOT, run_loop

CO = ['--preset', 'toy-cotrain-smoke']


def co_loop(run, *extra, env=None, timeout=600):
    e = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH, **(env or {}))
    return subprocess.run([sys.executable, '-m', 'zero.run', '--run', str(run), *CO, *extra], cwd=ROOT,
                          capture_output=True, text=True, timeout=timeout, env=e)


@pytest.fixture(scope='module')
def co_run(tmp_path_factory):
    run = tmp_path_factory.mktemp('runs') / 'co'
    p = co_loop(run, '--iters', '2', '--draft-eval-net-every', '2')
    assert p.returncode == 0, p.stdout + p.stderr
    return run


def test_cotrain_iterations_both_nets(co_run):
    run, res = co_run, co_run / 'resolver'
    mp, mr = read_metrics(str(run)), read_metrics(str(res))
    assert [r['iter'] for r in mp] == [0, 1, 2] and [r['iter'] for r in mr] == [0, 1, 2]
    for it in (1, 2):
        sp, rp = mp[it]['selfplay'], mr[it]['selfplay']
        assert sp['games'] == 8 and sp['unfinished'] == 0 and sp['samples'] > 0, sp
        co = sp['cotrain']
        assert co['events'] == 8 and co['episodes'] == rp['games'] > 0 and rp['samples'] > 0
        assert co['pending_max'] == 32 and co['parked_mb_max'] > 0 and co['event_latency_s'][0] > 0
        assert mp[it]['train']['steps'] >= 10 and mr[it]['train']['steps'] >= 10
        assert os.listdir(run / 'data' / f'iter_{it:04d}') and os.listdir(res / 'data' / f'iter_{it:04d}')
        for d in (run, res):
            assert (d / 'models' / f'iter_{it:04d}.pt').exists() and not (d / 'models' / f'iter_{it:04d}.json').exists()
    # evaluation: the primary's games vs its yardstick and random (episodes by the resolver's
    # yardstick), its resolver-net games every draft_eval_net_every; the resolver as a single run
    assert mp[2]['eval_yard']['opp'] == 'greedy' and mp[2]['eval_yard']['games'] == 8 and mp[2]['eval_random']['games'] == 4
    assert 'eval_yard_net' in mp[0] and 'eval_yard_net' not in mp[1] and mp[2]['eval_yard_net']['games'] == 8
    assert mr[2]['eval_yard']['opp'] == 'heuristic' and mr[2]['eval_random']['games'] == 4
    rp_, rr = (json.load(open(d / 'elo' / 'ratings.json')) for d in (run, res))
    assert rp_['anchor'] == 'greedy' and {'iter_0000', 'iter_0001', 'iter_0002', 'random'} <= set(rp_['ratings'])
    assert rr['anchor'] == 'heuristic' and {'iter_0000', 'iter_0001', 'iter_0002', 'random'} <= set(rr['ratings'])
    pairs = json.load(open(run / 'elo' / 'results.json'))['pairs']
    assert pairs['iter_0001|iter_0002'][1] == 4 and pairs['iter_0000|iter_0002'][1] == 4
    cfg, rcfg = json.load(open(run / 'config.json')), json.load(open(res / 'config.json'))
    assert cfg['resolver'] == 'toy' and cfg['resolver_cfg']['preset'] == 'toy-smoke'
    assert rcfg['game'] == 'toy' and rcfg['yardstick'] == 'heuristic' and rcfg['resolver'] == ''
    assert json.load(open(run / 'progress.json'))['phase'] == 'idle'
    assert runctl.read_state(str(run), 'run')['state'] == 'done'
    assert not (run / 'debug').exists() and not (res / 'debug').exists()
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'co-training run' in log and '[resolver] iter 2 train' in log


def test_cotrain_pause_mid_iteration_replays_it(tmp_path):
    """PAUSE during iteration 1's self-play (many events): exit 75 before either checkpoint;
    resumed, the iteration is replayed from its start (both data directories cleared)."""
    run = tmp_path / 'p'
    run.mkdir()
    proc = subprocess.Popen([sys.executable, '-m', 'zero.run', '--run', str(run), *CO, '--iters', '2', '--games', '5000'],
                            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH))
    deadline = time.time() + 180
    prog = {}
    while time.time() < deadline:
        try:
            prog = json.load(open(run / 'progress.json'))
        except (OSError, ValueError):
            pass
        if prog.get('iter') == 1 and prog.get('phase') == 'selfplay':
            break
        time.sleep(0.1)
    assert {'events_resolved', 'target', 'drafting', 'pending_events', 'matches_queued', 'matches_in_flight',
            'matches_done'} <= set(prog), prog
    assert prog['target'] == 5000
    runctl.request_pause(str(run), by='test')
    out, _ = proc.communicate(timeout=120)
    assert proc.returncode == runctl.EXIT_PAUSED, out
    for d in (run, run / 'resolver'):
        assert latest_iter(str(d)) == 0 and (d / 'data' / 'iter_0001').exists()
    runctl.request_resume(str(run))
    p = co_loop(run, '--iters', '2', '--games', '8')
    assert p.returncode == 0, p.stdout + p.stderr
    mp, mr = read_metrics(str(run)), read_metrics(str(run / 'resolver'))
    assert [r['iter'] for r in mp] == [0, 1, 2] and [r['iter'] for r in mr] == [0, 1, 2]
    assert mp[1]['selfplay']['games'] == 8
    # the replayed iteration's shards are the new ones only
    from zero.data import read_shard, shard_paths
    for d, m in ((run, mp), (run / 'resolver', mr)):
        rows = sum(len(read_shard(s)['player']) for s in shard_paths(str(d / 'data' / 'iter_0001')))
        assert rows == m[1]['selfplay']['samples']


def test_cotrain_resume_after_checkpoints_finishes_evaluations(co_run, tmp_path):
    """Interrupted after both checkpoints of iteration 2: the restart finishes its evaluations
    (from models/iter_0002.json of each net) without replaying self-play or training."""
    run = tmp_path / 'r'
    shutil.copytree(co_run, run)
    for d in (run, run / 'resolver'):
        m = read_metrics(str(d))
        last = {k: v for k, v in m[-1].items() if k in ('iter', 'selfplay', 'train', 'phases')}
        last['resumed_seconds'] = 1.0
        with open(d / 'metrics.jsonl', 'w', encoding='utf-8') as f:
            for r in m[:-1]:
                f.write(json.dumps(r) + '\n')
        (d / 'models' / 'iter_0002.json').write_text(json.dumps(last))
    before = {p: os.path.getmtime(p) for p in (run / 'models' / 'iter_0002.pt', run / 'resolver' / 'models' / 'iter_0002.pt')}
    p = co_loop(run, '--iters', '2')
    assert p.returncode == 0, p.stdout + p.stderr
    assert 'iter 2: finishing its evaluations (interrupted)' in p.stdout
    for d in (run, run / 'resolver'):
        m = read_metrics(str(d))
        assert [r['iter'] for r in m] == [0, 1, 2] and 'eval_yard' in m[2] and m[2]['selfplay']['games'] > 0
        assert not (d / 'models' / 'iter_0002.json').exists()
    assert all(os.path.getmtime(p) == t for p, t in before.items()), 'no retraining'


RUNNER = textwrap.dedent('''
    import sys
    import numpy as np
    import zero.data as D, zero.net as N, zero.run as R
    mode = sys.argv[1]
    if mode.startswith('loss-nan-'):                  # the 3rd training step of that net's game
        game = {'primary': 'toy_team', 'resolver': 'toy'}[mode[9:]]
        f0 = N.Net.forward
        calls = [0]
        def forward(self, x, n_players):
            p, v, m = f0(self, x, n_players)
            if self.training and self.config['game'] == game:
                calls[0] += 1
                if calls[0] == 3:
                    v = v.clone()
                    v[5] = float('nan')
            return p, v, m
        N.Net.forward = forward
    elif mode == 'x-overflow-resolver':
        w0 = D.write_shard
        def write_shard(path, s, meta, bad_path=None):
            if meta.get('net') == 'resolver':
                s = dict(s)
                s['x'] = s['x'].copy()
                s['x'][3, 5] = 1e6
            return w0(path, s, meta, bad_path)
        D.write_shard = write_shard
    R.main(sys.argv[2:])
''')


def run_injected(tmp_path, mode):
    run = tmp_path / mode
    script = tmp_path / 'inject.py'
    script.write_text(RUNNER)
    e = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH)
    p = subprocess.run([sys.executable, str(script), mode, '--run', str(run), *CO, '--iters', '2'], cwd=ROOT,
                       capture_output=True, text=True, timeout=600, env=e)
    return run, p


@pytest.mark.parametrize('mode', ['loss-nan-primary', 'loss-nan-resolver', 'x-overflow-resolver'])
def test_cotrain_nonfinite_names_the_net(tmp_path, mode):
    run, p = run_injected(tmp_path, mode)
    assert p.returncode != 0, p.stdout + p.stderr
    msg = p.stderr.strip().splitlines()[-1]
    print(msg)
    assert 'zero.finite.NonFinite' in msg
    net = 'primary net (toy_team)' if mode.endswith('primary') else 'resolver net (toy)'
    assert f'NonFinite: {net}: ' in msg, msg
    # neither network got checkpoint 1 nor an optimizer state
    for d in (run, run / 'resolver'):
        assert (d / 'models' / 'iter_0000.pt').exists() and not (d / 'models' / 'iter_0001.pt').exists()
        assert not (d / 'models' / 'opt_latest.pt').exists()
    assert json.load(open(run / 'run_state.json'))['state'] == 'aborted'
    debug = run / ('debug' if mode.endswith('primary') else 'resolver/debug')
    other = run / ('resolver/debug' if mode.endswith('primary') else 'debug')
    if mode.startswith('loss-nan'):
        assert 'iter 1 training step 3 of ' in msg
        assert os.listdir(debug) == ['nonfinite_iter_0001_step_0003.pt']
    else:
        assert 'non-finite sample values in shard r00_000.npz' in msg and '"net": "resolver"' in msg
        assert os.listdir(debug) == ['iter_0001_r00_000.bad.npz']
    assert not other.exists()


def test_cotrain_resolver_init_from_a_single_game_checkpoint(tmp_path):
    src = tmp_path / 'single'
    p = run_loop(src, '--iters', '1')
    assert p.returncode == 0, p.stdout + p.stderr
    ck = src / 'models' / 'iter_0001.pt'
    run = tmp_path / 'init'
    p = co_loop(run, '--iters', '1', '--resolver-init', str(ck))
    assert p.returncode == 0, p.stdout + p.stderr
    a, b = torch.load(ck, weights_only=False), torch.load(run / 'resolver' / 'models' / 'iter_0000.pt', weights_only=False)
    assert a['config'] == b['config'] and b['extra']['init_from'] == os.path.abspath(ck)
    assert all(torch.equal(a['state'][k], b['state'][k]) for k in a['state'])
    assert 'resolver iteration 0 = ' in (run / 'log.txt').read_text(encoding='utf-8')
    assert [r['iter'] for r in read_metrics(str(run / 'resolver'))] == [0, 1]
    # a checkpoint of another game is refused
    bad = co_loop(tmp_path / 'bad', '--iters', '1', '--resolver-init', str(run / 'models' / 'iter_0000.pt'))
    assert bad.returncode != 0 and "a 'toy_team' network, the resolver is 'toy'" in bad.stderr


def test_fork_run_copies_both_nets(co_run, tmp_path):
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import fork_run
    dest = tmp_path / 'f'
    assert fork_run.main(['--source', str(co_run), '--dest', str(dest), '--iter', '1', '--set', 'reuse=8']) == 0
    for d in (dest, dest / 'resolver'):
        assert latest_iter(str(d)) == 1 and [r['iter'] for r in read_metrics(str(d))] == [0, 1]
        assert (d / 'models' / 'opt_latest.pt').exists() is False or torch.load(d / 'models' / 'opt_latest.pt', weights_only=False)['iter'] == 1
        assert os.listdir(d / 'data' / 'iter_0001')
    assert json.load(open(dest / 'config.json'))['reuse'] == 8
    assert json.load(open(dest / 'resolver' / 'config.json'))['game'] == 'toy'
    p = co_loop(dest, '--iters', '2')
    assert p.returncode == 0, p.stdout + p.stderr
    assert [r['iter'] for r in read_metrics(str(dest / 'resolver'))] == [0, 1, 2]
    (dest / 'STOP').write_text('')
    p = co_loop(dest, '--iters', '3')
    assert p.returncode == 0 and 'STOP file found' in p.stdout and latest_iter(str(dest)) == 2


@pytest.mark.parametrize('bad, msg', [('{"bogus": 1}', "unknown settings ['bogus']"),
                                      ('{"pending_max": 3}', "['pending_max'] are the primary run's settings"),
                                      ('{"game": "toy_team"}', "game 'toy_team' is not the resolver 'toy'")])
def test_cotrain_settings_are_checked(tmp_path, bad, msg):
    p = co_loop(tmp_path / 'x', '--iters', '1', '--resolver-cfg', bad)
    assert p.returncode != 0 and msg in p.stdout + p.stderr, p.stdout + p.stderr
