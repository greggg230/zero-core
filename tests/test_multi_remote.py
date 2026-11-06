"""Remote self-play on several hosts (zero/remoteplay.py, docs/az.md 34) end to end on the toy, every host
a 'local' one (a worker subprocess of this machine with a selfplay workdir of its own, as test_remoteplay's
'local' remote): a list of remotes sharing one iteration's quota exactly (their shards in data/iter_i/
under names of their own, their consumers '<name>/w<k>', metrics and progress per host, each host's env
in its workers), the run directory's `selfplay_remotes` file (hosts joining and leaving a live run between
iterations, nothing lost or counted twice, an empty file = this machine alone, the file alone turning
remote self-play on), parametrised names (a callable 'prefix:' remote), hosts whose spec fails skipped
and retried, one remote exactly as before (its metrics' keys and log lines), selfplay_local 0 (the
hosts play the quota, this machine only when none takes work), a co-training resolver on two hosts; and
the units: remote_spec's parametrised names, the file's format, Shares.rename / drop."""
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

from zero.run import read_metrics

from test_loop import PYPATH, ROOT, SMOKE
from test_remoteplay import TOY, check_quota, gone

# the package's remotes of these tests: 'a', 'b', 'c' (host 'local', each its own selfplay workdir and
# an env var naming it), 'vast:<box>' (a parametrised remote: a callable called with the box), 'bad:<x>'
# (a callable that raises) and 'broken' (a Mapping whose resolution raises: a remote not ready)
MAIN = textwrap.dedent('''
    import os, sys
    from collections.abc import Mapping
    from zero import package
    import zero.run as R
    base = os.environ['ZERO_TEST_HOSTS']
    pkg = package.active()

    def host(name):
        return {'host': 'local', 'repo': pkg.root, 'python': sys.executable,
                'workdir': os.path.join(base, 'eval-' + name), 'env': {'ZERO_TEST_HOST': name},
                'selfplay': {'workdir': os.path.join(base, 'sp-' + name)}}

    for n in ('a', 'b', 'c'):
        pkg.remotes[n] = host(n)
    pkg.remotes['twin'] = dict(host('a'), env={'ZERO_TEST_HOST': 'twin'})   # a's machine and workdir

    def bad(x):
        raise RuntimeError(f'no box {x}')

    class Broken(Mapping):
        def __getitem__(self, k):
            raise RuntimeError('its checkout is not the run\\'s commit')
        def __iter__(self):
            raise RuntimeError('its checkout is not the run\\'s commit')
        def __len__(self):
            return 0

    pkg.remotes['vast:'] = lambda box: host('vast-' + box)
    pkg.remotes['bad:'] = bad
    pkg.remotes['broken'] = Broken()
    R.cli()
''')


def env_for(base, test=None, **extra):
    e = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH, ZERO_TEST_HOSTS=str(base), ZERO_RSP_RETRY_S='1',
             ZERO_REMOTE_WORKDIR=str(base / 'mirror-local'),
             ZERO_RSP_TEST=json.dumps(dict({'echo_env': ['ZERO_TEST_HOST']}, **(test or {}))))
    e.update(extra)
    return e


def main_script(base):
    p = base / 'main.py'
    if not p.exists():
        p.write_text(MAIN, encoding='utf-8')
    return str(p)


def cmd(base, run, *extra):
    return [sys.executable, main_script(base), '--run', str(run), *SMOKE, *extra]


def run_hosts(base, run, *extra, env=None, timeout=600):
    return subprocess.run(cmd(base, run, *extra), cwd=ROOT, capture_output=True, text=True, timeout=timeout,
                          env=env or env_for(base))


def host_shards(run, it, tag):
    """Host `tag`'s shards of iteration `it` in the run's data (their local names start with its tag)."""
    return sorted(glob.glob(os.path.join(str(run), 'data', f'iter_{it:04d}', f'{tag}-m[0-9][0-9]-*.npz')))


def games_of(r, name):
    """A metrics row's accepted games of host `name` (its consumers'), or this machine's ('local')."""
    cons = r['selfplay']['queue']['consumers']
    if name == 'local':
        return cons.get('local', {}).get('games', 0)
    return sum(v.get('games', 0) for n, v in cons.items() if n.startswith(name + '/'))


def worker_log(base, host, run):
    return (base / f'sp-{host}' / run.name / 'w00.log').read_text(encoding='utf-8')


def test_two_hosts_share_one_quota(tmp_path):
    """selfplay_remote 'a,b': every iteration exactly its 600 games across this machine and both hosts;
    each host's workers are its consumers ('a/w0', 'b/w0') with its own mirror, env and seed salt; their
    shards land in data/iter_i/ under names of their own with the host in their meta; the metrics carry
    the aggregate (`selfplay.remote`, host = the list) and each host's record (`selfplay.remotes`), whose
    games and samples add up to it; the log has a line per host; no worker outlives the run."""
    run = tmp_path / 'ab'
    p = run_hosts(tmp_path, run, '--iters', '3', *TOY, '--selfplay-remote', 'a,b')
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    assert [r['iter'] for r in m] == [0, 1, 2, 3]
    check_quota(m, 600)
    for r in m[1:]:
        it, sp = r['iter'], r['selfplay']
        rem, rems, q = sp['remote'], sp['remotes'], sp['queue']
        assert rem['host'] == ['a', 'b'] and rem['procs'] == 2 and set(rems) == {'a', 'b'}, rem
        assert set(q['consumers']) == {'local', 'a/w0', 'b/w0'}, q['consumers']
        assert set(q['rates']) - {'_leaves_per_unit'} == {'local', 'a/w0', 'b/w0'}
        for k in ('games', 'samples', 'shards', 'bytes', 'games_played', 'accepted', 'lost', 'late', 'session_errors'):
            assert rem[k] == sum(x[k] for x in rems.values()), (k, rem, rems)
        assert set(rem['caps']) == {'local', 'a/w0', 'b/w0'}
        for name, x in rems.items():
            assert x['host'] == name and x['procs'] == 1 and set(x['caps']) <= {f'{name}/w0'}, x
            files = host_shards(run, it, name)
            assert len(files) == x['shards'], (name, files, x)
            rows = 0
            for f in files:
                with np.load(f) as z:
                    meta = json.loads(str(z['meta']))
                    assert meta['source'] == 'remote' and meta['remote'] == name and meta['iter'] == it
                    rows += meta['rows']
            assert rows == x['samples'] and games_of(r, name) == x['games'], (name, x)
        assert not glob.glob(os.path.join(str(run), 'data', f'iter_{it:04d}', 'm[0-9][0-9]-*.npz'))
    assert all(sum(r['selfplay']['remotes'][h]['games'] for r in m[1:]) > 0 for h in ('a', 'b')), \
        [r['selfplay']['remotes'] for r in m[1:]]
    sps = {h: json.load(open(tmp_path / f'sp-{h}' / 'ab' / 'selfplay.json')) for h in ('a', 'b')}
    assert {h: s['remote'] for h, s in sps.items()} == {'a': 'a', 'b': 'b'}
    assert sps['a']['seed_salt'] != sps['b']['seed_salt'] and sps['a']['workdir'] != sps['b']['workdir']
    for h in ('a', 'b'):
        assert f"env ZERO_TEST_HOST='{h}'" in worker_log(tmp_path, h, run)
        assert not glob.glob(str(tmp_path / f'sp-{h}' / 'ab' / '.w*.lock'))
        assert not glob.glob(str(tmp_path / f'sp-{h}' / '.busy-*'))
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert "self-play also on the remotes of selfplay_remote 'a,b': 'a' (local, 1 worker(s), mirror " in log
    assert re.search(r'iter 3 remote self-play: 2 worker\(s\), \d+ games', log)
    assert re.search(r'iter 3 remote self-play a: 1 worker\(s\), \d+ games', log)
    assert re.search(r'iter 3 remote self-play b: 1 worker\(s\), \d+ games', log)
    assert re.search(r'iter 3 shared queue: quota 600: 600 games \(local \d+, a/w0 \d+, b/w0 \d+\)', log), log[-3000:]
    assert gone([int(x) for x in re.findall(r'worker 0 up \(pid (\d+)', log)])


def test_twin_names_of_one_mirror_are_refused(tmp_path):
    """Two names for one machine and workdir ('twin' is 'a' with another env) would take each other's
    worker slots: the second is not added (logged, retried) and the run plays on with the first."""
    run = tmp_path / 'tw'
    p = run_hosts(tmp_path, run, '--iters', '2', *TOY, '--selfplay-remote', 'a,twin')
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    check_quota(m, 600)
    assert all(set(r['selfplay']['queue']['consumers']) == {'local', 'a/w0'} for r in m[1:])
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert "remote self-play twin: not added: its mirror " in log and "is 'a''s" in log


class FileChanger(threading.Thread):
    """Writes the run's selfplay_remotes file when its metrics reach given iterations: a change written
    once row k is there applies from iteration k + 2 (the remote's phase k + 1 began before row k).
    Keeps the progress.json snapshots of the self-play phases it sees."""

    def __init__(self, rundir, plan):
        super().__init__(daemon=True)
        self.rundir, self.plan, self.done = str(rundir), dict(plan), []
        self.progress = []
        self.halt = False

    def run(self):
        path = os.path.join(self.rundir, 'metrics.jsonl')
        prog = os.path.join(self.rundir, 'progress.json')
        while not self.halt:
            n = len(read_metrics(self.rundir)) if os.path.exists(path) else 0
            for k in sorted(self.plan):
                if n > k:                          # row k is there (rows 0..n-1)
                    write_file(self.rundir, self.plan.pop(k))
                    self.done.append(k)
            try:
                with open(prog, encoding='utf-8') as f:
                    pg = json.load(f)
                if pg.get('phase') == 'selfplay' and (not self.progress or self.progress[-1] != pg):
                    self.progress.append(pg)
            except (OSError, ValueError):
                pass
            time.sleep(0.1)


def write_file(run, names, comment=True):
    path = os.path.join(str(run), 'selfplay_remotes')
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        if comment:
            f.write('# the hosts of this run\n\n')
        for n in names:
            f.write(f'{n}   # a host\n')
    os.replace(tmp, path)


def test_the_file_adds_and_removes_hosts_of_a_live_run(tmp_path):
    """The run directory's selfplay_remotes file - alone, the setting empty - turns remote self-play on
    and is re-read at every phase start: [a] at the start, b added (it starts and plays), a removed (it
    takes no work at once, its workers quit; every iteration still exactly its games, none counted twice),
    then empty (this machine alone, nothing waits). Every change is logged; progress.json has each host."""
    run = tmp_path / 'dyn'
    run.mkdir()
    write_file(run, ['a'])
    ch = FileChanger(run, {2: ['a', 'b'], 4: ['b'], 6: []})
    ch.start()
    try:
        p = run_hosts(tmp_path, run, '--iters', '8', *TOY, timeout=900)
    finally:
        ch.halt = True
    assert p.returncode == 0, p.stdout + p.stderr
    assert ch.done == [2, 4, 6]
    m = read_metrics(str(run))
    assert [r['iter'] for r in m] == list(range(9))
    check_quota(m, 600)
    want = {1: {'a'}, 2: {'a'}, 3: {'a'}, 4: {'a', 'b'}, 5: {'a', 'b'}, 6: {'b'}, 7: {'b'}, 8: set()}
    for r in m[1:]:
        it, sp = r['iter'], r['selfplay']
        hosts = want[it]
        assert set(sp['remotes']) == hosts, (it, sp['remotes'])
        assert set(sp['queue']['consumers']) == {'local'} | {f'{h}/w0' for h in hosts}, (it, sp['queue']['consumers'])
        assert sp['remote']['host'] == sorted(hosts), sp['remote']
        assert sp['remote']['games'] == sum(x['games'] for x in sp['remotes'].values())
        for h in {'a', 'b'} - hosts:               # an unlisted host takes no game, no shard
            assert games_of(r, h) == 0 and not host_shards(run, it, h), (it, h)
    assert games_of(m[4], 'b') + games_of(m[5], 'b') > 0, [r['selfplay']['remotes'] for r in m[4:6]]
    assert m[8]['selfplay']['remote']['games'] == 0 and games_of(m[8], 'local') == 600
    assert m[8]['selfplay']['remote']['wait_s'] < 2
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'self-play also on the remotes of the file ' in log
    for line in ('remote self-play: hosts now [a, b] (+b)', 'remote self-play: hosts now [b] (-a)',
                 'remote self-play: hosts now [] (-b): this machine plays alone', 'remote self-play a: removed: its 1 '
                 'worker(s) quit', 'remote self-play b: added (local, 1 worker(s), mirror '):
        assert line in log, (line, log[-4000:])
    pids = {h: [int(x) for x in re.findall(rf'remote self-play {h}: worker 0 up \(pid (\d+)', log)] for h in 'ab'}
    assert pids['a'] and pids['b'] and gone(pids['a'] + pids['b'])
    assert f"env ZERO_TEST_HOST='b'" in worker_log(tmp_path, 'b', run)
    pr = [x for x in ch.progress if x.get('remotes')]
    assert pr and all(set(x['remotes']) <= {'a', 'b'} for x in pr), ch.progress[-3:]
    assert any(x['remote']['host'] == ['a', 'b'] and set(x['remotes']) == {'a', 'b'} for x in pr), pr[-3:]
    x = pr[-1]['remotes']
    assert all({'host', 'workers', 'state', 'games', 'in_flight', 'samples', 'shards', 'caps', 'held'} <= set(v)
               for v in x.values()), x


def test_parametrised_names_and_hosts_that_fail(tmp_path):
    """A parametrised name ('vast:x': the package's callable 'vast:' called with 'x') is a host like any;
    a name whose spec fails - its callable raises ('bad:1'), a Mapping whose checks fail ('broken'), no
    such remote ('nowhere') - is skipped with a log line and retried at every phase, the run going on
    with the others (every iteration exactly its games)."""
    run = tmp_path / 'pv'
    p = run_hosts(tmp_path, run, '--iters', '2', *TOY, '--selfplay-remote', 'vast:x,bad:1,broken,nowhere')
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    check_quota(m, 600)
    for r in m[1:]:
        sp = r['selfplay']
        assert set(sp['queue']['consumers']) == {'local', 'vast:x/w0'} and set(sp['remotes']) == {'vast:x'}, sp
        assert len(host_shards(run, r['iter'], 'vast_x')) == sp['remotes']['vast:x']['shards']
    assert sum(r['selfplay']['remotes']['vast:x']['games'] for r in m[1:]) > 0
    assert (tmp_path / 'sp-vast-x' / 'pv' / 'config.json').exists()
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert "env ZERO_TEST_HOST='vast-x'" in worker_log(tmp_path, 'vast-x', run)
    for name, why in (('bad:1', 'RuntimeError: no box 1'), ('broken', "RuntimeError: its checkout is not the run's commit"),
                      ('nowhere', "SystemExit: selfplay_remote 'nowhere': not in the game package remotes")):
        n = log.count(f'remote self-play {name}: not added ({why}')
        assert n >= 3, (name, n, log[-3000:])        # at the start and at every phase start (iterations 1, 2)
    assert log.count('retried at the next phase') >= 9


def test_one_remote_is_exactly_as_before(tmp_path):
    """One name keeps everything of before (docs/az.md 5.3): its consumers 'w0', no `selfplay.remotes`,
    the remote record's and the queue's keys of before (captured from the code before the hosts, d499716),
    the log lines of before, shards named m<KK>-<sid>_SSS.npz, no host name or seed salt in the mirror."""
    run = tmp_path / 'one'
    p = run_hosts(tmp_path, run, '--iters', '2', *TOY, '--selfplay-remote', 'a')
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    check_quota(m, 600)
    for r in m[1:]:
        sp = r['selfplay']
        assert 'remotes' not in sp and set(sp['queue']['consumers']) == {'local', 'w0'}
        assert set(sp['remote']) == BEFORE_REMOTE, set(sp['remote']) ^ BEFORE_REMOTE
        assert set(sp['queue']) == BEFORE_QUEUE and sp['remote']['host'] == 'a'
        assert set(sp['queue']['rates']) == {'local', 'w0', '_leaves_per_unit'}
        assert len(glob.glob(os.path.join(str(run), 'data', f'iter_{r["iter"]:04d}', 'm[0-9][0-9]-*.npz'))) == \
            sp['remote']['shards']
    spj = json.load(open(tmp_path / 'sp-a' / 'one' / 'selfplay.json'))
    assert 'remote' not in spj and 'seed_salt' not in spj
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert re.search(r"self-play also on the remote 'a' \(local, 1 worker\(s\), mirror .*\): 600 games per iteration in "
                     r"total, in flight per machine its throughput share x 1\n", log)
    assert re.search(r'iter 2 remote self-play: 1 worker\(s\), \d+ games, \d+ samples in \d+ shards \([\d.]+ MB\), \d+ '
                     r'leaves/s \(\d+ running\), \d+s \(\d+s backing off, started \d+s before this machine\'s phase\), '
                     r'waited [\d.]+s; \d+ played, \d+ held for the next iteration\n', log)
    assert re.search(r'iter 2 shared queue: quota 600: 600 games \(local \d+, w0 \d+\); caps local [\d.]+ -> \d+, w0 '
                     r'[\d.]+ -> \d+; rates/s local [\d.e+-]+, w0 [\d.e+-]+; staleness: mean', log)
    assert 'remote self-play a: worker 0 up (pid ' in log
    assert 'hosts now' not in log and 'remote self-play a: ' in log and 'remote self-play a/' not in log


# the selfplay.remote record's and selfplay.queue's keys of a single-game run with one remote, from the code
# before the hosts (zero-core d499716, the same toy run: test_remoteplay's settings)
BEFORE_REMOTE = {'host', 'procs', 'workers', 'in_phase', 'ended', 'games', 'games_played', 'samples', 'samples_played',
                 'shards', 'bytes', 'leaves', 'leaves_per_s', 'leaves_per_running_s', 'seconds', 'running_s',
                 'backoff_s', 'in_flight', 'late', 'late_prev', 'rejected', 'quarantined', 'fetch_failed', 'fetch_s',
                 'wait_s', 'session_errors', 'lead_s', 'caps', 'accepted', 'held', 'lost'}
BEFORE_QUEUE = {'quota', 'games', 'short', 'held', 'inflight', 'batch_inflight', 'units', 'consumers', 'rates',
                'staleness'}


def test_selfplay_local_0_leaves_the_quota_to_the_hosts(tmp_path):
    """selfplay_local 0 with the file [a, b]: once a worker takes work this machine starts no games (the
    run only trains) - its last iterations' games all the hosts'; the file emptied, no worker takes work
    and this machine plays the quota itself (the switch logged both ways); every iteration exact."""
    run = tmp_path / 'nl'
    run.mkdir()
    write_file(run, ['a', 'b'], comment=False)
    ch = FileChanger(run, {4: []})
    ch.start()
    try:
        p = run_hosts(tmp_path, run, '--iters', '6', *TOY, '--selfplay-local', '0', timeout=900)
    finally:
        ch.halt = True
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    assert [r['iter'] for r in m] == list(range(7))
    check_quota(m, 600)
    assert games_of(m[5], 'local') == 0 and games_of(m[5], 'a') + games_of(m[5], 'b') == 600, \
        [r['selfplay']['queue']['consumers'] for r in m[1:]]
    assert m[5]['selfplay']['queue']['consumers']['local']['cap'] == 0
    assert games_of(m[6], 'local') == 600 and m[6]['selfplay']['remote']['games'] == 0
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'remote self-play: this machine takes no games now (' in log
    assert 'remote self-play: this machine plays again (no remote worker takes work; selfplay_local 0)' in log
    assert "this machine plays only while no remote worker takes work (selfplay_local 0)" in log


def test_selfplay_local_is_checked(tmp_path):
    p = run_hosts(tmp_path, tmp_path / 'x', '--iters', '1', '--selfplay-local', '2')
    assert p.returncode != 0 and 'selfplay_local 2: 0 or 1' in p.stdout + p.stderr


def test_cotrain_resolver_on_two_hosts(tmp_path):
    """A co-training run's resolver with resolver_cfg selfplay_remote 'a,b': both hosts take queued
    episodes beside the local resolver driver (consumers 'a/w0', 'b/w0'), every iteration exactly its
    events; a selfplay_remotes file in a co-training run is refused (its resolver's remotes are the
    setting's)."""
    from zero.run import PRESETS
    run = tmp_path / 'co'
    rc = json.dumps(dict(PRESETS['toy-cotrain-smoke']['resolver_cfg'], selfplay_remote='a,b'))
    p = subprocess.run([sys.executable, main_script(tmp_path), '--run', str(run), '--preset', 'toy-cotrain-smoke',
                        '--iters', '3', '--games', '40', '--resolver-cfg', rc], cwd=ROOT, capture_output=True,
                       text=True, timeout=900, env=env_for(tmp_path))
    assert p.returncode == 0, p.stdout + p.stderr
    mp, mr = read_metrics(str(run)), read_metrics(str(run / 'resolver'))
    assert [r['iter'] for r in mp] == [r['iter'] for r in mr] == [0, 1, 2, 3]
    for a, b in zip(mp[1:], mr[1:]):
        co, rem = a['selfplay']['cotrain'], b['selfplay']['remote']
        assert a['selfplay']['games'] == co['events'] == 40
        assert co['episodes'] == b['selfplay']['games'] + rem['matches']
        assert set(b['selfplay']['queue']['consumers']) == {'local', 'a/w0', 'b/w0'}
        assert set(b['selfplay']['remotes']) == {'a', 'b'} and rem['host'] == ['a', 'b']
        assert rem['matches'] == sum(x['matches'] for x in b['selfplay']['remotes'].values())
    assert sum(r['selfplay']['remote']['matches'] for r in mr[1:]) > 0
    assert json.load(open(tmp_path / 'sp-a' / 'co.resolver' / 'selfplay.json'))['mode'] == 'matches'
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert "[resolver] episodes also on the remotes 'a' (local, 1 worker(s), mirror " in log
    write_file(run, ['a'])
    p = subprocess.run([sys.executable, main_script(tmp_path), '--run', str(run), '--preset', 'toy-cotrain-smoke',
                        '--iters', '4', '--games', '40', '--resolver-cfg', rc], cwd=ROOT, capture_output=True,
                       text=True, timeout=300, env=env_for(tmp_path))
    assert p.returncode != 0 and 'never from the file' in p.stdout + p.stderr, p.stdout + p.stderr


# ---------------------------------------------------------------- units

def test_remote_spec_parametrised_names(monkeypatch):
    """remote_spec: a name not in the remotes with a ':' calls its prefix's callable ('vast:') with the rest
    - at every resolution, its result a dict or a Mapping (resolved then); 'vast:' itself calls it with '';
    a plain remote stays a plain one; no such prefix (or not callable) = the SystemExit of before; eval_remote
    resolves the same way."""
    from collections.abc import Mapping
    from zero import package, remote
    calls = []

    class Lazy(Mapping):
        def __init__(self, box):
            self.box = box

        def _d(self):
            calls.append(('resolved', self.box))
            return {'host': f'u@{self.box}', 'repo': '/r', 'python': 'python3', 'workdir': f'/w/{self.box}'}

        def __getitem__(self, k):
            return self._d()[k]

        def __iter__(self):
            return iter(self._d())

        def __len__(self):
            return len(self._d())

    def vast(box):
        calls.append(('called', box))
        return Lazy(box)

    rem = package.active().remotes
    added = {'vast:': vast, 'plain': {'host': 'u@p', 'repo': '/r', 'python': 'python3', 'workdir': '/p'},
             'nofn:': {'host': 'u@n', 'repo': '/r', 'python': 'python3', 'workdir': '/n'}}
    rem.update(added)
    monkeypatch.delenv('ZERO_EVAL_WORKDIR', raising=False)
    try:
        s = remote.remote_spec('vast:w1', 'selfplay_remote')
        assert s['host'] == 'u@w1' and s['workdir'] == '/w/w1' and s['env'] == {} and s['overrides'] == {}
        assert calls[0] == ('called', 'w1') and ('resolved', 'w1') in calls
        assert remote.remote_spec('vast:w1')['host'] == 'u@w1' and calls.count(('called', 'w1')) == 2
        assert remote.remote_spec('vast:')['host'] == 'u@'
        assert remote.remote_spec('vast:a:b')['host'] == 'u@a:b'
        assert remote.remote_spec('plain')['host'] == 'u@p'
        for bad in ('nope', 'nope:x', 'nofn:x', 'plain:x'):
            with pytest.raises(SystemExit, match=f"selfplay_remote {bad!r}: not in the game package remotes"):
                remote.remote_spec(bad, 'selfplay_remote')
        assert remote.known_remote('vast:z') and remote.known_remote('plain') and remote.known_remote('local')
        assert not remote.known_remote('nope') and not remote.known_remote('nofn:x')
        assert remote.known_remote('vast:') and calls.count(('called', 'z')) == 0, 'known_remote resolves nothing'
    finally:
        for k in added:
            del rem[k]


def test_the_file_and_the_setting(tmp_path):
    from zero.remoteplay import read_remotes_file, remote_names, remotes_file_in_use
    assert remote_names('') == [] and remote_names(None) == []
    assert remote_names('mac') == ['mac'] and remote_names(' mac , vast:w1,mac,, ') == ['mac', 'vast:w1']
    f = tmp_path / 'selfplay_remotes'
    assert read_remotes_file(str(f)) is None and not remotes_file_in_use(str(tmp_path))
    f.write_text('# hosts\n\nmac\n  vast:w1   # the rented box\nmac\n#vast:w2\n', encoding='utf-8')
    assert read_remotes_file(str(f)) == ['mac', 'vast:w1'] and remotes_file_in_use(str(tmp_path))
    f.write_text('', encoding='utf-8')
    assert read_remotes_file(str(f)) == []


def test_shares_rename_and_drop():
    from zero.workqueue import Shares
    sh = Shares(1.0)
    sh.consumer('local', 2, 64)
    sh.consumer('w0', 2, 64, cautious=True)
    sh.observe('w0', 30, 10.0, 1.0)
    sh.units = 100
    before = sh.caps()
    sh.rename('w0', 'mac/w0')
    assert set(sh.c) == {'local', 'mac/w0'} and sh.c['mac/w0']['rate'] == 3.0
    after = sh.caps()
    assert after['mac/w0'] == before['w0'] and after['local'] == before['local']
    sh.rename('nope', 'x')
    sh.drop('mac/w0')
    sh.drop('nope')
    assert set(sh.caps()) == {'local'} and sh.caps()['local'] == 64



def test_restore_and_one_remote_becoming_a_list(tmp_path, monkeypatch):
    """In process (no worker starts: no phase is open): a restart's rates restored by name - one remote takes
    its own '<name>/w<k>' as 'w<k>' and drops the others'; a list maps a row of its one setting name's 'w<k>'
    to '<name>/w<k>' only when the setting names one remote, and keeps a host's rates until it joins - and the
    file written in a run of one remote renames its consumers in place ('w0' -> 'a/w0', the rate kept) while
    the new host resolves on a thread and joins as '<name>/w0'."""
    import types
    from zero import package
    from zero.remoteplay import Host, RemotePlay
    from zero.run import DEFAULTS
    rem = package.active().remotes
    for n in 'abc':
        monkeypatch.setitem(rem, n, {'host': 'local', 'repo': package.active().root, 'python': sys.executable,
                                     'workdir': str(tmp_path / f'eval-{n}'),
                                     'selfplay': {'workdir': str(tmp_path / f'sp-{n}')}})
    monkeypatch.delenv('ZERO_SELFPLAY_WORKDIR', raising=False)
    logs = []

    def make(setting, d):
        os.makedirs(tmp_path / d, exist_ok=True)
        R = types.SimpleNamespace(cfg=dict(DEFAULTS, game='toy', conc=64, groups=2, selfplay_remote=setting),
                                  run=str(tmp_path / d), log=logs.append, league=None, forks=None)
        rp = RemotePlay(R, setting)
        rp.local(2, 64)
        return rp

    rp = make('a', 'one')
    try:
        rp.restore({'local': {'rate': 2.0}, 'a/w0': {'rate': 3.0, 'lead': 1.5}, 'b/w0': {'rate': 5.0},
                    '_leaves_per_unit': 10.0})
        assert set(rp.shares.c) == {'w0', 'local'} and rp.shares.c['w0']['rate'] == 3.0
        assert rp.shares.c['w0']['lead'] == 1.5 and rp.shares.lpu == 10.0 and not rp.multi
        write_file(rp.R.run, ['a', 'b'])
        with rp.mu:
            rp._membership(1)
        assert rp.multi and set(rp.shares.c) == {'a/w0', 'local'} and rp.shares.c['a/w0']['rate'] == 3.0
        t = time.time() + 20
        while 'b' not in rp.hosts and time.time() < t:
            time.sleep(0.05)
        assert set(rp.shares.c) == {'a/w0', 'local', 'b/w0'} and rp.hosts['b'].salt and rp.hosts['a'].salt
        assert rp.hosts['a'].tag == 'a' and rp.hosts['b'].cname(0) == 'b/w0'
        assert 'remote self-play: hosts now [a, b] (+b)' in logs
        os.remove(os.path.join(rp.R.run, 'selfplay_remotes'))      # the setting decides again: one remote
        with rp.mu:
            rp._membership(2)
        assert not rp.multi and set(rp.shares.c) == {'w0', 'local'} and rp.shares.c['w0']['rate'] == 3.0
        assert list(rp.hosts) == ['a'] and rp.hosts['a'].salt == 0 and 'remote self-play: hosts now [a] (-b)' in logs
    finally:
        rp.close()
    rp = make('a,b', 'two')
    try:
        rp.restore({'local': {'rate': 2.0}, 'w0': {'rate': 4.0}, 'b/w0': {'rate': 5.0}, 'c/w0': {'rate': 7.0}})
        assert set(rp.shares.c) == {'a/w0', 'b/w0', 'local'} and rp.shares.c['b/w0']['rate'] == 5.0
        assert rp.shares.c['a/w0']['rate'] is None, "a list of two names: a row's 'w0' is no host's"
        with rp.mu:
            assert rp._add_host(Host(rp, 'c'))
        assert rp.shares.c['c/w0']['rate'] == 7.0 and rp.procs == 3 and len(rp.fths) == 3
    finally:
        rp.close()
    rp = make('b', 'map')
    try:
        write_file(rp.R.run, ['b', 'c'])
        rp.close()
        rp = make('b', 'map')                      # (the file in use at the start: a list)
        rp.restore({'w0': {'rate': 6.0}})
        assert rp.multi and rp.shares.c['b/w0']['rate'] == 6.0, "the setting's one name: its row's 'w0'"
    finally:
        rp.close()
