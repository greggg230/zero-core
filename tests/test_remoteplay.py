"""Remote self-play (zero/remoteplay.py, zero/workqueue.py, docs/az.md 5.3 and 11.6) end to end on
the toys with the remote 'local' (a mirror directory and worker subprocesses, as remote.py's tests):
a single-game run's shared quota (exact, across both machines, games beyond it held for the next
iteration), its remote shards joining data/iter_i/ and its training, the caps following the
measured throughput of a slow / fast pair and keeping the rows within about one checkpoint, a worker
slower than a phase capped at its floor, the quota moving to this machine while the remote backs
off, a worker dying with accepted games not in a shard yet, lingering through training, late shards
dropped, an unreachable remote never blocking, a non-finite shard quarantined (the run fails as for
a local one), the back-off while an evaluation holds its lock, PAUSE mid-phase and resume (no worker
outlives the run); a co-training run's resolver episodes played by the remote (exact event quotas,
the local driver never starved), episodes requeued on a timeout, given back by a worker backing off
and requeued when a worker dies, the settings refused; the coordinator's second consumer, the
compressed shards and the streamed window as units."""
import glob
import json
import os
import re
import subprocess
import sys
import textwrap
import threading
import time

import numpy as np
import pytest

from zero import runctl
from zero.run import PRESETS, read_metrics

from test_cotrain import co_loop
from test_loop import PYPATH, ROOT, run_loop

TOY = ['--conc', '64', '--games', '600']           # toy-smoke phases of ~2.5 s: the worker joins in iteration 2


def env_for(tmp_path, test=None, **extra):
    e = {'ZERO_REMOTE_WORKDIR': str(tmp_path / 'mirror'), 'ZERO_RSP_RETRY_S': '1'}
    if test:
        e['ZERO_RSP_TEST'] = json.dumps(test)
    e.update(extra)
    return e


def remote_shards(run, it=None):
    pat = f'iter_{it:04d}' if it is not None else 'iter_*'
    return sorted(glob.glob(os.path.join(str(run), 'data', pat, 'm[0-9][0-9]-*.npz')))


def worker_pids(run):
    log = open(os.path.join(str(run), 'log.txt'), encoding='utf-8').read()
    return [int(p) for p in re.findall(r'worker \d+ up \(pid (\d+)', log)]


def gone(pids, within=20):
    import psutil
    t = time.time() + within
    while time.time() < t:
        if not any(psutil.pid_exists(p) for p in pids):
            return True
        time.sleep(0.2)
    return False


@pytest.fixture(scope='module')
def rsp_run(tmp_path_factory):
    base = tmp_path_factory.mktemp('rsp')
    run = base / 'sp'
    p = run_loop(run, '--iters', '3', *TOY, '--selfplay-remote', 'local', env=env_for(base))
    assert p.returncode == 0, p.stdout + p.stderr
    return run, base / 'mirror'


def check_quota(m, games):
    """Every iteration exactly `games` games, this machine's and the remote's, none short."""
    for r in m[1:]:
        sp, q = r['selfplay'], r['selfplay']['queue']
        assert q['quota'] == q['games'] == games and q['short'] == 0, q
        assert sp['games'] + sp['remote']['games'] == games, (sp['games'], sp['remote'])
        assert sum(c.get('games', 0) for c in q['consumers'].values()) == games, q['consumers']


def test_remote_shards_join_the_iteration_and_its_training(rsp_run):
    """The shared quota: exactly `games` games per iteration between the two machines (the games
    beyond it held for the next); the remote's compressed shards land in data/iter_i/ next to the
    run's own, every one checked and counted; training's steps follow the local + remote samples;
    the mirror keeps nothing it shipped; no worker, slot or busy marker outlives the run."""
    run, mirror = rsp_run
    m = read_metrics(str(run))
    assert [r['iter'] for r in m] == [0, 1, 2, 3]
    contributed = [r for r in m[1:] if r['selfplay']['remote']['samples']]
    assert len(contributed) >= 1, [r['selfplay']['remote'] for r in m[1:]]
    for r in m[1:]:
        it, sp, rem = r['iter'], r['selfplay'], r['selfplay']['remote']
        files = remote_shards(run, it)
        assert len(files) == rem['shards']
        rows = 0
        for f in files:
            with np.load(f) as z:
                meta = json.loads(str(z['meta']))
                assert meta['source'] == 'remote' and meta['iter'] == it and meta['game'] == 'toy'
                assert len(z['player']) == meta['rows']
                rows += meta['rows']
                assert int(z['net'].max()) <= it - 1
        assert rows == rem['samples'] and (rem['samples_played'] > 0 or not rem['samples'])
        assert rem['late'] == 0 and rem['quarantined'] == 0 and rem['rejected'] == 0 and rem['lost'] == 0
        assert rem['wait_s'] < 10
        steps = max(10, int(np.ceil((sp['samples'] + rem['samples']) * 4.0 / 64)))
        assert r['train']['steps'] == steps
        assert r['train']['window_shards'] >= 1 + len(files)
        assert rem['bytes'] == sum(os.path.getsize(f) for f in files)
    r = contributed[-1]['selfplay']['remote']
    assert r['games'] > 0 and r['leaves'] > 0 and r['leaves_per_s'] > 0 and r['workers'] == 1
    check_quota(m, 600)
    q = m[3]['selfplay']['queue']
    assert set(q['consumers']) == {'local', 'w0'} and q['consumers']['w0']['games'] == m[3]['selfplay']['remote']['games']
    assert q['staleness']['summary']['rows'] == m[3]['selfplay']['samples'] + m[3]['selfplay']['remote']['samples']
    # the remote's phase i starts as soon as checkpoint i - 1 is written, before its evaluation here
    assert m[1]['selfplay']['remote']['lead_s'] == 0 and all(x['selfplay']['remote']['lead_s'] > 0 for x in m[2:])
    rd = mirror / 'sp'
    assert json.load(open(rd / 'config.json'))['selfplay_remote'] == ''
    assert json.load(open(rd / 'selfplay.json'))['mode'] == 'selfplay'
    assert not glob.glob(str(rd / 'data' / '*' / '*.npz')), 'fetched shards are deleted on the remote'
    assert not glob.glob(str(rd / '.w*.lock')) and not glob.glob(str(mirror / '.busy-*'))
    assert gone(worker_pids(run))
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert "self-play also on the remote 'local'" in log and 'iter 3 remote self-play: 1 worker(s)' in log
    assert 'iter 3 shared queue: quota 600: 600 games (local ' in log


# phases of ~1-2 s on the toy (64 / 32 simulations, half of the decisions full): long enough for the
# caps to be measured; 256 slots for this machine and the worker each, a quota of 192 games
PAIR = ['--conc', '256', '--games', '192', '--sims', '64', '--sims-fast', '32', '--p-full', '0.5']


def pair_env(tmp_path, test=None, **extra):
    """Status lines every 0.2 s and a worker's first rate after 0.5 s of its phase (the toys' phases
    are seconds, MTG's minutes)."""
    return env_for(tmp_path, dict({'status_s': 0.2}, **(test or {})), ZERO_RSP_PROVISIONAL_S='0.5', **extra)


def model_caps(q):
    """The caps the shared queue's model gives for a metrics row's rates, head starts and units."""
    from zero.workqueue import Shares
    sh = Shares(q['inflight'])
    sh.restore(q['rates'])
    for n, st in q['rates'].items():
        if not n.startswith('_'):
            sh.consumer(n, st['lo'], st['hi'], cautious=n != 'local')
    sh.units = q['units']
    return sh.caps()


def test_caps_follow_throughput_and_rows_stay_within_one_checkpoint(tmp_path):
    """A slow / fast pair (the worker process plays at about half this machine's driver's rate):
    every iteration exactly its 192 games; the next caps are the model's (inflight x rate x (head
    start + the joint phase)), the slower consumer the smaller share; once the caps are measured the
    rows are within one checkpoint of the iteration's (at 768 slots each, every slot full - the old
    behaviour - drifts to a mean lag of 2 and more, docs/az.md 5.3)."""
    run = tmp_path / 'pair'
    p = run_loop(run, '--iters', '5', *PAIR, '--selfplay-remote', 'local', env=pair_env(tmp_path))
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    check_quota(m, 192)
    for r in m[1:]:
        q = r['selfplay']['queue']
        assert {n: c['cap_next'] for n, c in q['consumers'].items()} == model_caps(q), q
    for r in m[3:]:
        q = r['selfplay']['queue']
        c = q['consumers']
        assert c['w0']['rate'] < c['local']['rate'] and 0 < c['w0']['games'] < c['local']['games'], c
        assert sum(min(v['cap'], 256) for v in c.values()) <= 1.15 * 192 + 4, c
        st = q['staleness']['summary']
        assert st['le1'] >= 0.95 and st['mean'] <= 0.8, q['staleness']
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'iter 5 shared queue: quota 192: 192 games (local ' in log and 'staleness: mean' in log


def test_a_worker_slower_than_a_phase_is_capped_at_its_floor(tmp_path):
    """A worker whose games take several phases (here 10 ms more per batch) is measured by its
    leaves per second - it finishes nothing to count - so its cap falls to a few games, and this
    machine plays (nearly) the whole quota with its own cap at the quota."""
    run = tmp_path / 'slow'
    p = run_loop(run, '--iters', '4', *PAIR, '--selfplay-remote', 'local', env=pair_env(tmp_path, {'slow_s': 0.01}))
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    check_quota(m, 192)
    for r in m[2:]:
        c = r['selfplay']['queue']['consumers']
        assert c['w0']['cap_next'] <= 10 and c['local']['cap_next'] >= 170, c
        assert c['local']['games'] >= 170, c


def test_backoff_moves_the_quota_to_this_machine(tmp_path):
    """While the worker backs off (an evaluation's lock) it takes no new games: its cap is 0 and this
    machine's is the whole quota at once, so the iteration never waits for it; once the lock is gone
    the shares come back."""
    wd = tmp_path / 'mirror'
    lock = wd / '.eval.lock'
    lock.mkdir(parents=True)
    holder = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])
    (lock / 'owner').write_text(f'{holder.pid} {time.time()}')
    run = tmp_path / 'bq'

    def release():
        t = time.time() + 300
        while time.time() < t:
            try:
                if any(r['iter'] == 2 for r in read_metrics(str(run))):
                    break
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(0.05)
        holder.kill()
        holder.wait()
    th = threading.Thread(target=release, daemon=True)
    th.start()
    try:
        p = run_loop(run, '--iters', '5', *PAIR, '--selfplay-remote', 'local', env=pair_env(tmp_path))
    finally:
        holder.kill()
        holder.wait()
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    check_quota(m, 192)
    held, back = m[2]['selfplay'], m[5]['selfplay']
    assert held['remote']['backoff_s'] > 0.3 and held['remote']['games'] == 0, held['remote']
    hc, bc = held['queue']['consumers'], back['queue']['consumers']
    assert hc['local']['games'] == 192 and hc['local']['cap_max'] == 192 and hc['w0']['games'] == 0, hc
    assert back['remote']['games'] > 0 and bc['local']['cap_max'] < 185 and bc['w0']['cap'] > 0, bc


def test_a_dying_worker_gives_its_accepted_games_back(tmp_path):
    """A worker that dies with accepted games whose samples never reached a shard (here after 30, every
    session): the quota needs them again and this machine plays them; every iteration is exact."""
    run = tmp_path / 'dg'
    p = run_loop(run, '--iters', '3', *PAIR, '--selfplay-remote', 'local',
                 env=pair_env(tmp_path, {'die_after_accepts': 30}))
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    check_quota(m, 192)
    rem = [r['selfplay']['remote'] for r in m[1:]]
    assert sum(r['lost'] for r in rem) >= 30 and sum(r['session_errors'] for r in rem) > 0, rem
    assert all(r['games'] < 30 for r in rem), 'a session dies at its 30th accepted game: none of those arrive'
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert re.search(r'worker 0 ended \(exit 9, .*30 accepted games lost with it \(the quota needs them again\)', log), \
        log[-3000:]


def test_linger_plays_through_training(tmp_path):
    """selfplay_remote_linger 1: between phases (the run trains, evaluates) the worker plays on with
    its checkpoint; the games it finishes then count for the next iteration (its quota takes them
    first) and their rows are one checkpoint behind it; quotas stay exact."""
    run = tmp_path / 'lg'
    p = run_loop(run, '--iters', '4', *PAIR, '--selfplay-remote', 'local', '--selfplay-remote-linger', '1',
                 env=pair_env(tmp_path))
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    check_quota(m, 192)
    assert json.load(open(tmp_path / 'mirror' / 'lg' / 'selfplay.json'))['linger'] is True
    lingered = [r['selfplay']['remote'].get('linger') for r in m[2:]]
    assert any(x and x['games'] > 0 and x['seconds'] > 0 for x in lingered), lingered
    for r in m[3:]:
        st = r['selfplay']['queue']['staleness']
        assert st['remote'].get('1', 0) > 0 and st['summary']['mean'] <= 1.0, st
    wl = (tmp_path / 'mirror' / 'lg' / 'w00.log').read_text(encoding='utf-8')
    assert 'lingering between phases' in wl and '; lingering' in wl


def test_late_shards_are_dropped(tmp_path):
    """A remote slower than selfplay_remote_wait: its shards of phase i come after the phase closed
    and are dropped (reported as late), never in iteration i or any later one; the run waits at
    most the deadline."""
    run = tmp_path / 'late'
    # phases of ~6 s; every shard announced 3 s after it is written: after its phase's 0.5 s deadline
    p = run_loop(run, '--iters', '4', '--conc', '64', '--games', '1500', '--selfplay-remote', 'local',
                 '--selfplay-remote-wait', '0.5', env=env_for(tmp_path, {'delay_shard_s': 3}))
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    rem = [r['selfplay']['remote'] for r in m[1:]]
    assert not remote_shards(run)
    assert sum(r['samples'] for r in rem) == 0
    assert sum(r['late'] + r['late_prev'] for r in rem) >= 1, rem
    assert all(r['wait_s'] < 3 for r in rem), rem
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert re.search(r'worker 0: shard m00-[0-9a-f]{4}_000\.npz of iteration \d came after its phase closed: dropped', log)


DEAD = textwrap.dedent('''
    from zero import package
    import zero.run as R
    package.active().remotes['dead'] = {'host': 'nobody@unreachable.invalid', 'repo': '/nowhere', 'python': 'python3',
                                        'workdir': '/nowhere/zero-remote'}
    R.cli()
''')


def test_unreachable_remote_never_blocks(tmp_path):
    """selfplay_remote = a host that cannot be reached: every worker start fails and is retried
    (logged), the run self-plays its quota here and trains on its own samples only."""
    script = tmp_path / 'dead.py'
    script.write_text(DEAD)
    run = tmp_path / 'dead'
    t0 = time.time()
    p = subprocess.run([sys.executable, str(script), '--run', str(run), '--preset', 'toy-smoke', '--iters', '2', *TOY,
                        '--eval-games', '8', '--eval-random-games', '4', '--elo-games', '4', '--ref-games', '8',
                        '--selfplay-remote', 'dead'], cwd=ROOT, capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH, ZERO_RSP_RETRY_S='1'))
    assert p.returncode == 0, p.stdout + p.stderr
    assert time.time() - t0 < 120
    m = read_metrics(str(run))
    assert [r['iter'] for r in m] == [0, 1, 2]
    for r in m[1:]:
        rem = r['selfplay']['remote']
        assert rem['samples'] == 0 and rem['workers'] == 0 and rem['wait_s'] < 1
        assert r['train']['steps'] == max(10, int(np.ceil(r['selfplay']['samples'] * 4.0 / 64)))
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'remote self-play dead: worker 0 not started: RuntimeError: mkdir on nobody@unreachable.invalid' in log, \
        log[-2000:]
    assert not remote_shards(run)


def test_nonfinite_remote_shard_is_quarantined(tmp_path):
    """A remote shard with a NaN (here a file damaged after the worker's own check) is caught on
    arrival: quarantined in <run>/debug/, never in data/, and the run fails with NonFinite at the
    end of the phase, before any training on it."""
    run = tmp_path / 'nan'
    p = run_loop(run, '--iters', '3', *TOY, '--selfplay-remote', 'local', env=env_for(tmp_path, {'corrupt': 1}))
    assert p.returncode != 0
    out = p.stdout + p.stderr
    assert 'NonFinite' in out and 'non-finite sample values in its shard' in out, out[-3000:]
    bad = glob.glob(str(run / 'debug' / 'iter_*_m00-*.bad.npz'))
    assert len(bad) == 1
    it = int(os.path.basename(bad[0])[5:9])
    assert not remote_shards(run)
    assert not os.path.exists(run / 'models' / f'iter_{it:04d}.pt'), 'no training on it'
    assert runctl.read_state(str(run), 'run')['state'] == 'aborted'
    with np.load(bad[0]) as z:
        assert np.isnan(z['x'].astype(np.float32)).any()
    assert gone(worker_pids(run))


def test_backs_off_while_an_evaluation_holds_its_lock(tmp_path):
    """Remote evaluation first: while <workdir>/.eval.lock is held by a live process the worker
    asks for no leaf batches (its running time stops, no busy marker); once the lock is gone it
    plays again."""
    wd = tmp_path / 'mirror'
    lock = wd / '.eval.lock'
    lock.mkdir(parents=True)
    holder = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])
    (lock / 'owner').write_text(f'{holder.pid} {time.time()}')
    run = tmp_path / 'bo'
    released = {}

    def release():
        # the lock goes away once iteration 2 is done with it held throughout (its metrics row; not
        # progress.json, written every 2 s: the toy's phases are shorter than that on a fast machine)
        t = time.time() + 300
        while time.time() < t:
            try:
                done = any(r['iter'] == 2 for r in read_metrics(str(run)))
            except (OSError, ValueError, KeyError):
                done = False
            if done:
                released['busy_while_held'] = glob.glob(str(wd / '.busy-*'))
                holder.kill()
                holder.wait()
                released['t'] = time.time()
                return
            time.sleep(0.05)
    th = threading.Thread(target=release, daemon=True)
    th.start()
    try:
        p = run_loop(run, '--iters', '4', '--conc', '64', '--games', '1500', '--selfplay-remote', 'local',
                     env=env_for(tmp_path))
    finally:
        holder.kill()
        holder.wait()
    assert p.returncode == 0, p.stdout + p.stderr
    assert 't' in released, 'iteration 2 never finished'
    assert released['busy_while_held'] == []
    m = read_metrics(str(run))
    rem = {r['iter']: r['selfplay']['remote'] for r in m[1:]}
    assert rem[2]['backoff_s'] > 0.5, rem
    assert rem[4]['games'] > 0 and rem[4]['backoff_s'] == 0, rem
    wl = (wd / 'bo' / 'w00.log').read_text(encoding='utf-8')
    assert 'backing off:' in wl and '.eval.lock is held' in wl and 'continuing after' in wl


BACKGROUND = textwrap.dedent('''
    from zero import package
    import zero.run as R
    package.active().remotes['local'] = {'selfplay': {'backoff': 'background'}}
    R.cli()
''')


def test_background_backoff(tmp_path):
    """The remote's `selfplay.backoff` 'background': on macOS the worker plays on under the Darwin
    background policy while the lock is held (its games go on; its running time stops); elsewhere
    (no such policy) it stops as 'stop' does."""
    wd = tmp_path / 'mirror'
    lock = wd / '.eval.lock'
    lock.mkdir(parents=True)
    holder = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])
    (lock / 'owner').write_text(f'{holder.pid} {time.time()}')
    script = tmp_path / 'bg.py'
    script.write_text(BACKGROUND)
    run = tmp_path / 'bg'
    try:
        p = subprocess.run([sys.executable, str(script), '--run', str(run), '--preset', 'toy-smoke', '--iters', '3',
                            '--conc', '64', '--games', '1500', '--eval-games', '8', '--eval-random-games', '4',
                            '--elo-games', '4', '--ref-games', '8', '--selfplay-remote', 'local'], cwd=ROOT,
                           capture_output=True, text=True, timeout=600,
                           env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH, **env_for(tmp_path)))
    finally:
        holder.kill()
        holder.wait()
    assert p.returncode == 0, p.stdout + p.stderr
    assert json.load(open(wd / 'bg' / 'selfplay.json'))['backoff'] == 'background'
    rem = [r['selfplay']['remote'] for r in read_metrics(str(run))[1:]]
    wl = (wd / 'bg' / 'w00.log').read_text(encoding='utf-8')
    if sys.platform == 'darwin':
        assert 'playing on in the background' in wl and 'background policy not' not in wl, wl
        assert rem[-1]['games'] > 0 and rem[-1]['backoff_s'] > 0, rem
    else:
        assert 'backing off:' in wl and 'playing on in the background' not in wl
        assert sum(r['games'] for r in rem) == 0, rem


def test_pause_mid_phase_then_resume(tmp_path):
    """PAUSE while the remote plays: the run exits 75 and its worker with it; the restart replays
    the iteration (its data directory cleared: no shard of the old session in it) with a new
    worker, and finishes."""
    run = tmp_path / 'pz'
    e = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH, **env_for(tmp_path))
    with open(tmp_path / 'pz.out', 'w', encoding='utf-8') as out:
        proc = subprocess.Popen([sys.executable, '-m', 'zero.run', '--run', str(run), '--preset', 'toy-smoke',
                                 '--iters', '3', '--conc', '64', '--games', '4000', '--eval-games', '8',
                                 '--eval-random-games', '4', '--elo-games', '4', '--ref-games', '8',
                                 '--selfplay-remote', 'local'], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, env=e)
        try:
            t = time.time() + 180
            while time.time() < t:
                try:
                    prog = json.load(open(run / 'progress.json'))
                except (OSError, ValueError):
                    prog = {}
                if (prog.get('remote') or {}).get('games', 0) > 50:
                    break
                time.sleep(0.1)
            assert (prog.get('remote') or {}).get('games', 0) > 50, prog
            it = prog['iter']
            runctl.request_pause(str(run), by='test')
            proc.wait(timeout=120)
        finally:
            if proc.poll() is None:
                proc.kill()
    assert proc.returncode == runctl.EXIT_PAUSED, (tmp_path / 'pz.out').read_text(encoding='utf-8')[-3000:]
    pids = worker_pids(run)
    assert pids and gone(pids), 'the worker quits with the run'
    assert not glob.glob(str(tmp_path / 'mirror' / 'pz' / '.w*.lock'))
    runctl.request_resume(str(run))
    p = run_loop(run, '--iters', '3', '--games', '600', '--selfplay-remote', 'local', env=env_for(tmp_path))
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    assert [r['iter'] for r in m] == [0, 1, 2, 3]
    sids = {os.path.basename(f).split('_')[0] for f in remote_shards(run, it)}
    old = {f'm00-{s}' for s in re.findall(r'\[w0 ([0-9a-f]{4}) pid ' + str(pids[0]),
                                          (tmp_path / 'mirror' / 'pz' / 'w00.log').read_text(encoding='utf-8'))}
    assert not (sids & old), 'the replayed iteration holds no shard of the paused session'
    assert gone(worker_pids(run))


# ---------------------------------------------------------------- co-training: the resolver's episodes

def rcfg(**kw):
    return json.dumps(dict(PRESETS['toy-cotrain-smoke']['resolver_cfg'], selfplay_remote='local', **kw))


def check_cotrain(run, games=40):
    """Exact event quotas, the remote's rows counted once, its shards in resolver/data/."""
    mp, mr = read_metrics(str(run)), read_metrics(str(run / 'resolver'))
    assert [r['iter'] for r in mp] == [r['iter'] for r in mr] == [0, 1, 2, 3]
    for a, b in zip(mp[1:], mr[1:]):
        co, rem = a['selfplay']['cotrain'], b['selfplay']['remote']
        assert a['selfplay']['games'] == co['events'] == games
        assert co['episodes'] == b['selfplay']['games'] + rem['matches']
        assert co['episodes_remote'] == rem['matches']
        files = remote_shards(run / 'resolver', a['iter'])
        assert len(files) == rem['shards']
        steps = max(10, int(np.ceil((b['selfplay']['samples'] + rem['samples']) * 4.0 / 64)))
        assert b['train']['steps'] == steps
    return mp, mr


def test_cotrain_resolver_episodes_on_the_remote(tmp_path):
    """resolver_cfg selfplay_remote 'local': the remote takes queued episodes next to the local
    resolver driver, its rows resolve events as the local ones do (every iteration exactly its
    quota of events), its samples come back as shards into resolver/data/iter_i/."""
    run = tmp_path / 'co'
    p = co_loop(run, '--iters', '3', '--games', '40', '--resolver-cfg', rcfg(), env=env_for(tmp_path))
    assert p.returncode == 0, p.stdout + p.stderr
    mp, mr = check_cotrain(run)
    rem = [r['selfplay']['remote'] for r in mr[1:]]
    assert sum(r['matches'] for r in rem) > 0 and sum(r['samples'] for r in rem) > 0, rem
    assert rem[0]['lead_s'] == 0 and all(r['lead_s'] > 0 for r in rem[1:]), rem   # it plays on during evaluations
    assert all(r['duplicates_dropped'] == 0 and r['requeued'] == 0 for r in rem), rem
    assert mp[1]['selfplay']['cotrain']['pending_max'] == 2 * 16 + (16 + 4)
    assert json.load(open(tmp_path / 'mirror' / 'co.resolver' / 'selfplay.json'))['mode'] == 'matches'
    assert not remote_shards(run), 'the primary self-plays here only'
    for a, b in zip(mp[1:], mr[1:]):           # the shared queue: both consumers, the local one never idle
        q = b['selfplay']['queue']
        assert set(q['consumers']) == {'local', 'w0'} and b['selfplay']['games'] > 0, q
        assert q['consumers']['local']['games'] == b['selfplay']['games'] and q['units'] in (None, 160), q
        assert q['staleness']['summary']['rows'] == b['selfplay']['samples'] + b['selfplay']['remote']['samples']
        assert 'staleness' in a['selfplay']['cotrain']
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert "[resolver] episodes also on the remote 'local'" in log and '[resolver] iter 3 remote self-play:' in log
    assert '[resolver] iter 3 shared queue: caps local ' in log
    assert gone(worker_pids(run))


BIGREMOTE = textwrap.dedent('''
    from zero import package
    import zero.run as R
    package.active().remotes['local'] = {'selfplay': {'overrides': {'conc': 64}}}
    R.cli()
''')


def test_cotrain_the_local_driver_is_never_starved(tmp_path):
    """A remote with four times the local resolver driver's room (64 slots + 16 queued against 16 +
    16): the coordinator keeps the local driver's share in reserve, so it plays matches every
    iteration however fast the remote takes them (td1's iteration 7: the desktop played none of
    1,812)."""
    script = tmp_path / 'big.py'
    script.write_text(BIGREMOTE)
    run = tmp_path / 'big'
    p = subprocess.run([sys.executable, str(script), '--run', str(run), '--preset', 'toy-cotrain-smoke', '--iters', '3',
                        '--games', '40', '--resolver-cfg', rcfg()], cwd=ROOT, capture_output=True, text=True,
                       timeout=600, env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH,
                                            **env_for(tmp_path)))
    assert p.returncode == 0, p.stdout + p.stderr
    mp, mr = check_cotrain(run)
    for b in mr[1:]:
        q = b['selfplay']['queue']
        tot = b['selfplay']['games'] + b['selfplay']['remote']['matches']
        assert b['selfplay']['games'] >= 0.15 * tot, (b['selfplay']['games'], tot, q['consumers'])
        assert q['consumers']['w0']['cap'] <= 80 and q['local_cap'] <= 32, q


def test_cotrain_backoff_gives_matches_back(tmp_path):
    """A worker that backs off (an evaluation's lock) for longer than selfplay_remote_giveback (here
    1 s) takes no new matches and gives its outstanding ones back: requeued here at the front, the
    local driver plays them, a late copy of the worker's is dropped; the quotas stay exact."""
    wd = tmp_path / 'mirror'
    run = tmp_path / 'gb'
    state = {}

    def hold():
        t = time.time() + 300
        while time.time() < t:
            try:
                prog = json.load(open(run / 'progress.json'))
            except (OSError, ValueError):
                prog = {}
            mr = prog.get('matches_remote') or {}
            if prog.get('iter', 0) >= 2 and mr.get('outstanding', 0) >= 4:
                lock = wd / '.eval.lock'
                lock.mkdir(parents=True, exist_ok=True)
                h = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])
                (lock / 'owner').write_text(f'{h.pid} {time.time()}')
                state['held'] = mr['outstanding']
                time.sleep(6)
                h.kill()
                h.wait()
                return
            time.sleep(0.05)
    th = threading.Thread(target=hold, daemon=True)
    th.start()
    p = co_loop(run, '--iters', '3', '--games', '40', '--resolver-cfg', rcfg(selfplay_remote_giveback=1.0),
                env=env_for(tmp_path, {'slow_s': 0.02, 'status_s': 0.2}))
    th.join(5)
    assert p.returncode == 0, p.stdout + p.stderr
    assert state.get('held'), 'the worker never had matches outstanding'
    mp, mr = check_cotrain(run)
    rem = [r['selfplay']['remote'] for r in mr[1:]]
    assert sum(r['requeued_backoff'] for r in rem) > 0 and sum(r['backoff_s'] for r in rem) > 1, rem
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert re.search(r'worker 0: backing off for \d+s \(.*\.eval\.lock\): its \d+ outstanding episodes requeued here',
                     log), log[-3000:]


def test_cotrain_episodes_requeued_on_timeout(tmp_path):
    """A slow remote: episodes it has not finished within selfplay_remote_timeout s of its running
    time are requeued here; whichever copy finishes second is dropped; the quotas stay exact."""
    run = tmp_path / 'slow'
    p = co_loop(run, '--iters', '3', '--games', '40', '--resolver-cfg', rcfg(selfplay_remote_timeout=0.5),
                env=env_for(tmp_path, {'slow_s': 0.03, 'status_s': 0.2}))
    assert p.returncode == 0, p.stdout + p.stderr
    mp, mr = check_cotrain(run)
    rem = [r['selfplay']['remote'] for r in mr[1:]]
    assert sum(r['requeued_timeout'] for r in rem) > 0, rem
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'not finished within 0.5s of its running time: requeued here' in log


def test_cotrain_worker_death_requeues_its_episodes(tmp_path):
    """A worker that dies mid-phase (here after 15 rows, every time): its outstanding episodes are
    requeued here at once, a new worker starts after the retry delay, the quotas stay exact."""
    run = tmp_path / 'die'
    p = co_loop(run, '--iters', '3', '--games', '40', '--resolver-cfg', rcfg(), env=env_for(tmp_path, {'die_after_rows': 15}))
    assert p.returncode == 0, p.stdout + p.stderr
    mp, mr = check_cotrain(run)
    rem = [r['selfplay']['remote'] for r in mr[1:]]
    assert sum(r['requeued'] for r in rem) > 0 and sum(r['session_errors'] for r in rem) > 0, rem
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert re.search(r'worker 0 ended \(exit 9, .*episodes requeued here', log), log[-3000:]


@pytest.mark.parametrize('extra, msg', [
    (['--selfplay-remote', 'local'], "the resolver's episodes go to a remote with resolver_cfg's selfplay_remote"),
    (['--resolver-cfg', json.dumps({'preset': 'toy-smoke', 'selfplay_remote': 'nowhere'})],
     "resolver_cfg: selfplay_remote 'nowhere': not in the game package remotes")])
def test_cotrain_selfplay_remote_settings_checked(tmp_path, extra, msg):
    p = co_loop(tmp_path / 'x', '--iters', '1', *extra)
    assert p.returncode != 0 and msg in p.stdout + p.stderr, p.stdout + p.stderr


def test_unknown_selfplay_remote_is_refused(tmp_path):
    p = run_loop(tmp_path / 'u', '--iters', '1', '--selfplay-remote', 'nowhere')
    assert p.returncode != 0 and "selfplay_remote 'nowhere': not in the game package remotes" in p.stdout + p.stderr


def test_a_workdir_that_is_not_a_remote_path_is_refused(tmp_path):
    """Git Bash turns ZERO_SELFPLAY_WORKDIR=/Users/... into C:/Program Files/Git/Users/... for a
    Windows process: refused at start (ssh would create it under the remote's home)."""
    script = tmp_path / 'dead.py'
    script.write_text(DEAD)
    p = subprocess.run([sys.executable, str(script), '--run', str(tmp_path / 'w'), '--preset', 'toy-smoke', '--iters', '1',
                        '--selfplay-remote', 'dead'], cwd=ROOT, capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH,
                                ZERO_SELFPLAY_WORKDIR='C:/Program Files/Git/Users/u/mtgz-selfplay'))
    assert p.returncode != 0
    assert "the remote self-play workdir 'C:/Program Files/Git/Users/u/mtgz-selfplay' is not an absolute path" in \
        p.stdout + p.stderr


# ---------------------------------------------------------------- units

class FakeDrv:
    def __init__(self):
        self.q = []
        self.res = []

    @property
    def queued(self):
        return len(self.q)

    def enqueue(self, tag, spec):
        self.q.append((tag, spec))

    def take_results(self):
        r, self.res = self.res, []
        return r


def test_coordinator_second_consumer_and_duplicates():
    """The local feed keeps at most local_cap queued; the remote takes the rest oldest first;
    requeued episodes go to the front; the second copy of a requeued episode is dropped, whichever
    consumer finishes first; an unknown tag is still an error."""
    from zero.cotrain import Coordinator
    co = Coordinator(lambda p: [f'{p}-{k}' for k in range(3)], lambda p, rows: json.dumps([r['result'] for r in rows]))
    now = time.time()
    for g in (1, 2):
        co.events[g] = {'payload': str(g), 'n': 3, 'rows': {}, 't': now}
        for k in range(3):
            co.to_queue.append((f'{g}/{k}', f'{g}-{k}'))
        co.outstanding += 3
    co.local_cap = 2
    drv = FakeDrv()
    assert co.feed(drv) == 2 and [t for t, _ in drv.q] == ['1/0', '1/1']
    assert co.feed(drv) == 0
    got = co.take_for_remote(3)
    assert [t for t, _ in got] == ['1/2', '2/0', '2/1']
    co.requeue([got[0]])                           # the remote is late with 1/2: back to the front
    assert co.to_queue[0][0] == '1/2'
    row = lambda tag: {'tag': tag, 'result': [1.0, 0.0], 'scores': [1, 0]}  # noqa: E731
    drv.res = [row('1/0'), row('1/1')]
    assert co.collect(drv) == 2
    assert co.collect_rows([row('1/2')]) == 1      # the remote's copy first: taken, the event ready
    assert co.ready and co.ready[0][0] == 1 and 1 not in co.events
    drv.res = [row('1/2')]                         # the local copy afterwards: dropped
    assert co.collect(drv) == 0 and co.dup_dropped == 1
    assert co.collect_rows([row('2/0'), row('2/1')]) == 2
    with pytest.raises(RuntimeError, match='unknown tag'):
        co.collect_rows([row('7/0')])
    assert co.outstanding == 1 and len(co.take_remote_rows()) == 3 and len(co.take_rows()) == 2


def test_lock_held(tmp_path):
    from zero.remoteplay import lock_held
    d = tmp_path / '.eval.lock'
    assert lock_held([str(d)]) is None
    d.mkdir()
    assert lock_held([str(d)]) == str(d), 'being created (no owner yet)'
    holder = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        (d / 'owner').write_text(f'{holder.pid} {time.time()}')
        assert lock_held([str(tmp_path / 'none'), str(d)]) == str(d)
    finally:
        holder.kill()
        holder.wait()
        holder.wait()
    assert lock_held([str(d)]) is None, 'a dead owner does not count'


def test_compressed_shards(tmp_path):
    """write_shard(compress=1): a compressed npz np.load reads like a plain one; on_write sees each
    shard; the non-finite guard quarantines a compressed shard too."""
    from zero.data import ShardWriter, read_shard
    from zero.finite import NonFinite
    rng = np.random.default_rng(0)
    N, F = 200, 64
    off = np.arange(N + 1) * 3
    x = np.zeros((N, F), np.float32)
    x[:, :5] = rng.standard_normal((N, 5))
    s = {'x': x, 'player': np.zeros(N, np.int64), 'n_players': np.full(N, 2), 'opt_off': off,
         'opt_action': rng.integers(0, 9, off[-1]), 'pi': np.full(off[-1], 1 / 3, np.float32),
         'z': np.zeros((N, 8), np.float32), 'q': np.zeros((N, 8), np.float32), 'sc': np.zeros((N, 8), np.float32),
         'net': np.zeros(N, np.int32)}
    seen = []
    for comp in (0, 1):
        w = ShardWriter(str(tmp_path / f'd{comp}'), 'm00-abcd', rows=150, meta={'game': 'g'}, compress=comp,
                        on_write=lambda p, r, b: seen.append((os.path.basename(p), r, b)))
        w.add(s)
        w.close()
    plain, comp = seen[:1], seen[1:]
    assert [r for _, r, _ in plain] == [r for _, r, _ in comp] == [200]
    assert comp[0][2] < plain[0][2] / 3
    a = read_shard(str(tmp_path / 'd0' / 'm00-abcd_000.npz'))
    b = read_shard(str(tmp_path / 'd1' / 'm00-abcd_000.npz'))
    assert all(np.array_equal(a[k], b[k]) for k in a if k != 'meta') and a['meta'] == b['meta']
    s['x'][3, 7] = np.nan
    w = ShardWriter(str(tmp_path / 'd2'), 'm00-abcd', meta={'game': 'g'}, compress=1)
    w.add(s)
    with pytest.raises(NonFinite):
        w.close()
    assert os.path.exists(tmp_path / 'd2' / 'm00-abcd_000.bad.npz')
