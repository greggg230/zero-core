"""Self-play in several processes (run setting selfplay_procs, zero/selfplay_procs.py) at
toy-smoke scale: rolling and classic iterations, the merged metrics, both workers' shards in
the training window, determinism, PAUSE (exit 75, no worker left behind), and a worker killed
mid-phase or failing with an exception fails the run loudly (no partial iteration)."""
import json
import os
import subprocess
import sys
import time

import numpy as np
import psutil
import pytest

from zero import run as zrun, runctl
from zero.data import read_shard, shard_paths
from zero.run import latest_iter, read_metrics
from zero.selfplay_procs import merge, share, worker_cfg, worker_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYPATH = os.pathsep.join([os.path.join(ROOT, 'python')] + [p for p in [os.environ.get('PYTHONPATH')] if p])
SMOKE = ['--preset', 'toy-smoke', '--conc', '24', '--games', '24', '--eval-games', '8', '--eval-random-games', '4',
         '--elo-games', '4', '--ref-games', '8']
ENV = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH)


def run_loop(run, *extra, timeout=300):
    return subprocess.run([sys.executable, '-m', 'zero.run', '--run', str(run), *SMOKE, *extra], cwd=ROOT,
                          capture_output=True, text=True, timeout=timeout, env=ENV)


def start_loop(run, out, *extra):
    run.mkdir(exist_ok=True)
    return subprocess.Popen([sys.executable, '-m', 'zero.run', '--run', str(run), *SMOKE, *extra], cwd=ROOT,
                            stdout=open(out, 'w', encoding='utf-8'), stderr=subprocess.STDOUT, env=ENV)


def wait_for(pred, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.1)
    return False


def progress(run):
    try:
        with open(run / 'progress.json', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def workers_of(run):
    return (runctl.read_state(str(run), 'run') or {}).get('selfplay_workers') or []


def playing(run, it=1):
    """Iteration `it`'s self-play is under way in the workers."""
    p = progress(run)
    return p.get('phase') == 'selfplay' and p.get('iter') == it and p.get('procs') == 2 and len(workers_of(run)) == 2


def all_gone(ws, timeout=10):
    return wait_for(lambda: not any(runctl.alive(w) for w in ws), timeout)


# ------------------------------------------------------------------ units

def test_split_seeds_config():
    assert [share(25, 2, k) for k in range(2)] == [13, 12] and [share(10, 3, k) for k in range(3)] == [4, 3, 3]
    cfg = dict(zrun.DEFAULTS, conc=1025, threads=6, groups=2)
    ws = [worker_cfg(cfg, 2, k) for k in range(2)]
    assert [w['conc'] for w in ws] == [513, 512] and {w['threads'] for w in ws} == {3}
    assert all(w['groups'] == 2 and w['selfplay_procs'] == 1 for w in ws)
    assert worker_cfg(dict(cfg, threads=1), 4, 3)['threads'] == 1
    assert len({worker_seed(7919 * s + it, k) for s in range(3) for it in range(1, 5) for k in range(4)}) == 48
    args = zrun.parse(['--run', 'x'])
    for bad in ({'selfplay_procs': 0}, {'selfplay_procs': 3, 'games': 2}, {'selfplay_procs': 3, 'conc': 2}):
        with pytest.raises(SystemExit):
            zrun.resolve_config(args, bad)
    assert zrun.resolve_config(args, {})['selfplay_procs'] == 1


def test_merge():
    def part(games, fin, samples, leaves, secs, calls, kind_n):
        o = {'games': games, 'unfinished': games - fin, 'samples': samples, 'shards': 1, 'bytes': 100 * samples,
             'leaves': leaves, 'leaves_per_s': leaves / secs, 'seconds': secs, 'infer_s': secs / 2,
             'avg_batch': leaves / calls, 'rolling': {'in_flight': 5, 'restarts': 1, 'restart_sims': 2, 'dropped_leaves': 3}}
        raw = {'fin': fin, 'decisions': 30 * games, 'steps': 40 * games, 'rounds': 6 * fin, 'score': 2.0 * fin,
               'ev_calls': calls, 'ev_leaves': leaves, 'target': (kind_n, 5.0 * kind_n, 0.8 * kind_n, 0.4 * kind_n)}
        return o, raw
    m = merge([part(10, 10, 100, 1000, 2.0, 10, 100), part(14, 12, 60, 3000, 3.0, 20, 60)], 4.0, 'visits', True,
              pids=[11, 12])
    assert m['games'] == 24 and m['unfinished'] == 2 and m['samples'] == 160 and m['shards'] == 2
    assert m['leaves'] == 4000 and m['leaves_per_s'] == 1000 and m['games_per_h'] == 24 / 4 * 3600
    assert m['decisions_per_game'] == 30 and m['steps_per_game'] == 40 and m['rounds'] == 6 and m['avg_score'] == 2
    assert abs(m['avg_batch'] - 4000 / 30) < 1e-9 and m['infer_s'] == 2.5
    assert m['target'] == {'kind': 'visits', 'samples': 160, 'options': 5.0, 'h_norm': 0.8, 'max_share': 0.4}
    assert m['rolling'] == {'in_flight': 10, 'restarts': 2, 'restart_sims': 4, 'dropped_leaves': 6}
    assert m['procs'] == 2 and [w['pid'] for w in m['workers']] == [11, 12] and m['workers'][1]['games'] == 14


# ------------------------------------------------------------------ runs

@pytest.fixture(scope='module')
def rolling_run(tmp_path_factory):
    run = tmp_path_factory.mktemp('procs') / 'roll'
    p = run_loop(run, '--iters', '2', '--rolling', '1', '--selfplay-procs', '2')
    assert p.returncode == 0, p.stdout + p.stderr
    return run


def test_rolling_two_procs(rolling_run):
    run = rolling_run
    m = read_metrics(str(run))
    assert [r['iter'] for r in m] == [0, 1, 2]
    ws = workers_of(run)
    assert len(ws) == 2 and len({w['pid'] for w in ws}) == 2
    for r in m[1:]:
        sp = r['selfplay']
        assert sp['procs'] == 2 and sp['games'] >= 24 and sp['samples'] > 0 and sp['shards'] == 2
        assert [w['pid'] for w in sp['workers']] == [w['pid'] for w in ws]
        assert sum(w['games'] for w in sp['workers']) == sp['games'] and all(w['games'] >= 12 for w in sp['workers'])
        assert sum(w['samples'] for w in sp['workers']) == sp['samples']
        assert sp['leaves'] == sum(w['leaves'] for w in sp['workers']) > 0
        assert sp['target']['samples'] == sp['samples'] and set(sp['rolling']) == {'in_flight', 'restarts', 'restart_sims',
                                                                                    'dropped_leaves'}
        d = run / 'data' / f'iter_{r["iter"]:04d}'
        names = [os.path.basename(p) for p in shard_paths(str(d))]
        assert names == ['r00_000.npz', 'r01_000.npz']
        metas = [read_shard(str(d / n))['meta'] for n in names]
        assert [mt['worker'] for mt in metas] == [0, 1] and {mt['procs'] for mt in metas} == {2}
        assert sum(mt['rows'] for mt in metas) == sp['samples'] == sum(w['samples'] for w in sp['workers'])
        assert [mt['rows'] for mt in metas] == [w['samples'] for w in sp['workers']]
        for n in names:                    # rolling: games carried over keep their older networks' samples
            nets = {int(v) for v in np.unique(read_shard(str(d / n))['net'])}
            assert r['iter'] - 1 in nets and nets <= set(range(r['iter']))
    # training read both workers' shards: iteration 1 trains on iteration 1 only, 2 on 1 and 2
    assert m[1]['train']['window_shards'] == 2 and m[1]['train']['window_samples'] == m[1]['selfplay']['samples']
    assert m[2]['train']['window_shards'] == 4
    assert m[2]['train']['window_samples'] == m[1]['selfplay']['samples'] + m[2]['selfplay']['samples']
    assert 'startup_s' in m[1]['selfplay'] and 'startup_s' not in m[2]['selfplay']   # the same workers
    assert runctl.read_state(str(run), 'run')['state'] == 'done'
    assert all_gone(ws), 'the workers exit with the run'
    for k in range(2):
        assert 'starting' in (run / f'selfplay_w{k}.txt').read_text(encoding='utf-8')
    assert 'self-play workers: 2 processes' in (run / 'log.txt').read_text(encoding='utf-8')


def test_classic_two_procs_deterministic(tmp_path):
    rows = []
    for name in ('a', 'b'):
        run = tmp_path / name
        p = run_loop(run, '--iters', '1', '--rolling', '0', '--selfplay-procs', '2', '--games', '25')
        assert p.returncode == 0, p.stdout + p.stderr
        sp = read_metrics(str(run))[1]['selfplay']
        assert sp['games'] == 25 and [w['games'] for w in sp['workers']] == [13, 12] and 'rolling' not in sp
        d = run / 'data' / 'iter_0001'
        assert [os.path.basename(p) for p in shard_paths(str(d))] == ['c00_000.npz', 'c01_000.npz']
        assert read_metrics(str(run))[1]['train']['window_shards'] == 2
        rows.append((sp['samples'], sp['avg_score'], [read_shard(p) for p in shard_paths(str(d))]))
    (na, sa, a), (nb, sb, b) = rows
    assert na == nb and sa == sb
    for x, y in zip(a, b):
        assert all(np.array_equal(x[k], y[k]) for k in x if k != 'meta')
    assert not (a[0]['x'].shape == a[1]['x'].shape and np.array_equal(a[0]['x'], a[1]['x'])), \
        'the workers play different games'


def test_pause_two_procs_exits_75_and_resumes(tmp_path):
    run = tmp_path / 'p'
    proc = start_loop(run, tmp_path / 'out.txt', '--iters', '3', '--games', '4000', '--conc', '64', '--rolling', '1',
                      '--selfplay-procs', '2')
    try:
        assert wait_for(lambda: playing(run) or proc.poll() is not None, 180), 'self-play never started'
        assert proc.poll() is None, (tmp_path / 'out.txt').read_text(encoding='utf-8')
        ws = workers_of(run)
        runctl.request_pause(str(run), by='test')
        t0 = time.time()
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    out = (tmp_path / 'out.txt').read_text(encoding='utf-8')
    assert proc.returncode == runctl.EXIT_PAUSED, out
    assert time.time() - t0 < 30
    assert runctl.read_state(str(run), 'run')['state'] == 'paused'
    assert all_gone(ws), 'no worker outlives the paused run'
    assert [r['iter'] for r in read_metrics(str(run))] == [0] and latest_iter(str(run)) == 0
    assert any('stopped by the run process' in (run / f'selfplay_w{k}.txt').read_text(encoding='utf-8') for k in range(2))
    runctl.request_resume(str(run))
    p = run_loop(run, '--iters', '1', '--games', '24', '--conc', '24')
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    assert [r['iter'] for r in m] == [0, 1] and m[1]['selfplay']['procs'] == 2 and m[1]['selfplay']['games'] >= 24


def test_killed_worker_fails_the_run(tmp_path):
    run = tmp_path / 'k'
    proc = start_loop(run, tmp_path / 'out.txt', '--iters', '3', '--games', '4000', '--conc', '64', '--rolling', '1',
                      '--selfplay-procs', '2')
    try:
        assert wait_for(lambda: playing(run) or proc.poll() is not None, 180), 'self-play never started'
        assert proc.poll() is None, (tmp_path / 'out.txt').read_text(encoding='utf-8')
        ws = workers_of(run)
        victim = psutil.Process(ws[1]['pid'])
        assert runctl.alive(ws[1]) and victim.ppid() == proc.pid
        victim.kill()
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    out = (tmp_path / 'out.txt').read_text(encoding='utf-8')
    assert proc.returncode not in (0, runctl.EXIT_PAUSED), out
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'run aborted: WorkerError: self-play worker 1' in log, log
    assert runctl.read_state(str(run), 'run')['state'] == 'aborted'
    assert [r['iter'] for r in read_metrics(str(run))] == [0] and latest_iter(str(run)) == 0
    assert all_gone(ws), 'the other worker is stopped too'


def test_killed_run_leaves_no_workers(tmp_path):
    """Windows does not kill children with their parent: the workers notice by themselves."""
    run = tmp_path / 'o'
    proc = start_loop(run, tmp_path / 'out.txt', '--iters', '3', '--games', '4000', '--conc', '64', '--rolling', '1',
                      '--selfplay-procs', '2')
    try:
        assert wait_for(lambda: playing(run) or proc.poll() is not None, 180), 'self-play never started'
        assert proc.poll() is None, (tmp_path / 'out.txt').read_text(encoding='utf-8')
        ws = workers_of(run)
        assert all(runctl.alive(w) for w in ws)
        proc.kill()
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert all_gone(ws, timeout=15), 'orphaned self-play workers'
    assert any('the run process is gone' in (run / f'selfplay_w{k}.txt').read_text(encoding='utf-8') for k in range(2))


def test_worker_error_fails_the_run(tmp_path):
    """A worker's exception (here: the driver refuses 9-player games) ends the run with the
    worker's traceback; nothing of the iteration is kept as done."""
    run = tmp_path / 'e'
    p = run_loop(run, '--iters', '1', '--rolling', '1', '--selfplay-procs', '2', '--players', '9', '--eval-games', '0',
                 '--eval-random-games', '0', '--ref-games', '0')
    assert p.returncode not in (0, runctl.EXIT_PAUSED), p.stdout + p.stderr
    assert 'WorkerError: self-play worker' in p.stderr and '9 seats' in p.stderr, p.stderr
    assert [r['iter'] for r in read_metrics(str(run))] == [0] and latest_iter(str(run)) == 0
    assert all_gone(workers_of(run))
