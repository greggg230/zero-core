"""Remote self-play (zero/remoteplay.py, docs/az.md 5.3 and 11.6) end to end on the toys with the
remote 'local' (a mirror directory and worker subprocesses, as remote.py's tests): a single-game
run's remote shards joining data/iter_i/ and its training, late shards dropped, an unreachable
remote never blocking, a non-finite shard quarantined (the run fails as for a local one), the
back-off while an evaluation holds its lock, PAUSE mid-phase and resume (no worker outlives the
run); a co-training run's resolver episodes played by the remote (exact event quotas), episodes
requeued on a timeout and when a worker dies, the settings refused; the coordinator's second
consumer, the compressed shards and the streamed window as units."""
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


def test_remote_shards_join_the_iteration_and_its_training(rsp_run):
    """The remote's compressed shards land in data/iter_i/ next to the run's own, every one checked
    and counted; training's steps follow the local + remote samples; the mirror keeps nothing it
    shipped; no worker, slot or busy marker outlives the run."""
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
        assert rows == rem['samples'] == rem['samples_played'] if rem['samples'] else rows == 0
        assert rem['late'] == 0 and rem['quarantined'] == 0 and rem['rejected'] == 0
        assert rem['wait_s'] < 10
        steps = max(10, int(np.ceil((sp['samples'] + rem['samples']) * 4.0 / 64)))
        assert r['train']['steps'] == steps
        assert r['train']['window_shards'] >= 1 + len(files)
        assert rem['bytes'] == sum(os.path.getsize(f) for f in files)
    r = contributed[-1]['selfplay']['remote']
    assert r['games'] > 0 and r['leaves'] > 0 and r['leaves_per_s'] > 0 and r['workers'] == 1
    rd = mirror / 'sp'
    assert json.load(open(rd / 'config.json'))['selfplay_remote'] == ''
    assert json.load(open(rd / 'selfplay.json'))['mode'] == 'selfplay'
    assert not glob.glob(str(rd / 'data' / '*' / '*.npz')), 'fetched shards are deleted on the remote'
    assert not glob.glob(str(rd / '.w*.lock')) and not glob.glob(str(mirror / '.busy-*'))
    assert gone(worker_pids(run))
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert "self-play also on the remote 'local'" in log and 'iter 3 remote self-play: 1 worker(s)' in log


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
        # the lock goes away once iteration 2's self-play has run a while with it held
        t = time.time() + 240
        while time.time() < t:
            try:
                prog = json.load(open(run / 'progress.json'))
            except (OSError, ValueError):
                prog = {}
            if prog.get('iter') == 2 and prog.get('phase') == 'selfplay' and \
                    (prog.get('remote') or {}).get('state') == 'backing off':
                busy = glob.glob(str(wd / '.busy-*'))
                released['busy_while_held'] = busy
                holder.kill()
                released['t'] = time.time()
                return
            time.sleep(0.1)
    th = threading.Thread(target=release, daemon=True)
    th.start()
    try:
        p = run_loop(run, '--iters', '4', '--conc', '64', '--games', '1500', '--selfplay-remote', 'local',
                     env=env_for(tmp_path))
    finally:
        holder.kill()
    assert p.returncode == 0, p.stdout + p.stderr
    assert 't' in released, 'iteration 2 never showed the remote backing off'
    assert released['busy_while_held'] == []
    m = read_metrics(str(run))
    rem = {r['iter']: r['selfplay']['remote'] for r in m[1:]}
    assert rem[2]['backoff_s'] > 0.5, rem
    assert rem[4]['games'] > 0 and rem[4]['backoff_s'] == 0, rem
    wl = (wd / 'bo' / 'w00.log').read_text(encoding='utf-8')
    assert 'backing off:' in wl and '.eval.lock is held' in wl and 'continuing after' in wl


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
    assert all(r['duplicates_dropped'] == 0 and r['requeued'] == 0 for r in rem), rem
    assert mp[1]['selfplay']['cotrain']['pending_max'] == 2 * 16 + (16 + 4)
    assert json.load(open(tmp_path / 'mirror' / 'co.resolver' / 'selfplay.json'))['mode'] == 'matches'
    assert not remote_shards(run), 'the primary self-plays here only'
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert "[resolver] episodes also on the remote 'local'" in log and '[resolver] iter 3 remote self-play:' in log
    assert gone(worker_pids(run))


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
