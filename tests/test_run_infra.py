"""The run loop's infrastructure at its edges (the Fable review of 2026-11-12, docs/az.md 46): a co-training episode requeued more than once (its later copies dropped, never a crash or a second count; never
handed back to the worker it was taken from; a worker ignores a tag it still holds), fork_run --iter K cutting the Elo
pools at K and copying the window the fork's settings read, the end-of-run drain of remote evaluations (no time limit,
STOP logged and the run ends aborted), the remote transport telling a failure from "absent", a device the machine does
not have refused at the start, the DONE marker of a finished self-play phase (a resume after a crash in training
trains on it instead of playing it again; a marker that does not match its directory is ignored), the supervisor
giving up on the same crash three times in a row, run state files written atomically, and the self-play metrics
whose populations differ in a rolling phase."""
import json
import os
import subprocess
import sys
import textwrap
import threading

import numpy as np
import pytest

from zero import runctl
from zero.run import read_metrics

from test_loop import PYPATH, ROOT, SMOKE, run_loop


# ---------------------------------------------------------------- co-training: requeued episodes (docs/az.md 11.6)

def _coordinator():
    from zero.cotrain import Coordinator
    return Coordinator(lambda payload: ['{}', '{}'], lambda payload, rows: json.dumps([r['result'] for r in rows]))


class _Parked:
    def __init__(self, deferred):
        self.deferred = deferred

    def take_deferred(self):
        d, self.deferred = self.deferred, []
        return d


def _requeued_twice():
    """An event of two episodes; episode 7/0 goes out, is requeued (overdue), taken again, requeued again: three
    copies in flight."""
    co = _coordinator()
    co.take_events(_Parked([(7, 'payload')]))
    items = [co.to_queue.popleft(), co.to_queue.popleft()]
    co.requeue([items[0]])
    co.to_queue.popleft()
    co.requeue([items[0]])
    co.to_queue.popleft()
    return co


def row(tag):
    return {'tag': tag, 'result': [1.0, 0.0], 'over': True}


def test_an_episode_requeued_twice_drops_its_third_copy_after_its_event():
    """Three copies of 7/0: the first is taken, the event completes with 7/1; the second and the third are both
    dropped (one spare per requeue) - before the fix the third raised 'unknown tag' and the run aborted."""
    co = _requeued_twice()
    assert co.collect_rows([row('7/0')]) == 1
    assert co.collect_rows([row('7/1')]) == 1 and 7 not in co.events and co.ready
    assert co.collect_rows([row('7/0')]) == 0
    assert co.collect_rows([row('7/0')]) == 0
    assert co.dup_dropped == 2 and not co.dupe_ok and co.outstanding == 0
    with pytest.raises(RuntimeError, match='unknown tag'):
        co.collect_rows([row('7/0')])                 # a fourth copy never existed: still an error


def test_an_episode_requeued_twice_is_counted_once_while_its_event_is_open():
    """The later copies arrive while the event is still open: dropped against the spares - `outstanding` counts the
    episode once (before the fix: twice, below zero) and the metrics see one row."""
    co = _requeued_twice()
    assert co.outstanding == 2
    assert [co.collect_rows([row('7/0')]) for _ in range(3)] == [1, 0, 0]
    assert co.outstanding == 1 and 7 in co.events and co.dup_dropped == 2
    assert len(co.take_remote_rows()) == 1
    assert co.collect_rows([row('7/1')]) == 1 and co.outstanding == 0 and 7 not in co.events


def test_a_second_row_of_an_episode_never_requeued_is_an_error():
    co = _coordinator()
    co.take_events(_Parked([(3, 'payload')]))
    assert co.collect_rows([row('3/0')]) == 1
    with pytest.raises(RuntimeError, match='never requeued'):
        co.collect_rows([row('3/0')])


def test_take_for_remote_skips_a_workers_own_requeued_tags():
    """A worker's requeued tags stay queued for the other consumers (at the front, in order); the rest are taken."""
    co = _coordinator()
    co.take_events(_Parked([(1, 'a'), (2, 'b')]))       # 1/0 1/1 2/0 2/1
    got = co.take_for_remote(4, skip={'1/0', '2/0'})
    assert [t for t, _ in got] == ['1/1', '2/1']
    assert [t for t, _ in co.to_queue] == ['1/0', '2/0']
    assert [t for t, _ in co.take_for_remote(4)] == ['1/0', '2/0']


class _Sess:
    """The parts of a remoteplay Session that _timeouts / _giveback / _assign use."""
    def __init__(self, name='w0'):
        self.name, self.hello, self.k = name, {'ev': 'hello'}, 0
        self.assigned, self.gave, self.sent = {}, set(), []
        self.status = {'active_s': 0.0}
        self.bo_since, self.gave_back = None, False
        self.h = self

    def send(self, obj):
        self.sent.append(obj)

    def log(self, msg):
        pass


def test_an_overdue_episode_never_goes_back_to_the_same_worker():
    """remoteplay._timeouts requeues an overdue episode and frees the worker's room; _assign in the same tick does
    not hand that tag straight back to it (it goes to the other consumers) - before the fix the slow worker played it
    a second time and a third copy could follow."""
    from zero.remoteplay import RemotePlay
    rp = RemotePlay.__new__(RemotePlay)
    rp.mu = threading.Condition()
    rp.co = co = _coordinator()
    rp.cur = None
    rp.capv = {'w0': 2}
    rp.timeout_s, rp.overdue, rp.overdue_min = 0.5, 0.0, 0.5
    rp.shares = type('S', (), {'c': {}})()
    co.take_events(_Parked([(1, 'a')]))                  # 1/0 1/1
    s = _Sess()
    rp._assign(s)
    assert sorted(s.assigned) == ['1/0', '1/1']
    s.status = {'active_s': 10.0}                        # both overdue
    rp._timeouts(s)
    assert not s.assigned and s.gave == {'1/0', '1/1'} and [t for t, _ in co.to_queue] == ['1/0', '1/1']
    rp._assign(s)
    assert not s.assigned and len(co.to_queue) == 2, 'never handed back to the worker it was taken from'
    other = _Sess('w1')
    rp.capv['w1'] = 2
    rp._assign(other)
    assert sorted(other.assigned) == ['1/0', '1/1']


def test_a_worker_ignores_an_episode_it_still_holds():
    """The worker side: an `enqueue` of a tag it has not answered yet is ignored (its one row answers both)."""
    from zero.remoteplay import Worker
    w = Worker.__new__(Worker)
    w.holding, w.enq, w.backlog, w.drv, w.k, w.sid, w.rdir = set(), 0, [], None, 0, 'abcd', os.getcwd()
    w.note = lambda msg: None
    w._control({'cmd': 'enqueue', 'tag': '5/0', 'spec': 's'}, None)
    w._control({'cmd': 'enqueue', 'tag': '5/0', 'spec': 's'}, None)
    w._control({'cmd': 'enqueue', 'tag': '5/1', 'spec': 's'}, None)
    assert w.backlog == [('5/0', 's'), ('5/1', 's')] and w.enq == 2


# ---------------------------------------------------------------- fork_run --iter K (docs/az.md 9)

def _source(tmp_path, n=6, rows=10):
    src = tmp_path / 'src'
    (src / 'models').mkdir(parents=True)
    (src / 'elo' / 'visits-b1').mkdir(parents=True)
    for k in range(n):
        (src / 'models' / f'iter_{k:04d}.pt').write_bytes(b'x')
        d = src / 'data' / f'iter_{k:04d}'
        d.mkdir(parents=True)
        np.savez(d / 'c00_000.npz', meta=json.dumps({'rows': rows}), player=np.zeros(rows, np.int8))
    res = {'anchor': 'heuristic', 'pairs': {'iter_0004|iter_0005': [30.0, 50], 'heuristic|iter_0005': [10.0, 100],
                                           'iter_0002|iter_0003': [20.0, 50], 'heuristic|random': [80.0, 100]},
           'logged': ['iter_0003', 'iter_0004', 'iter_0005']}
    (src / 'elo' / 'results.json').write_text(json.dumps(res))
    (src / 'elo' / 'ratings.json').write_text('{"ratings": {}}')
    (src / 'elo' / 'visits-b1' / 'results.json').write_text(json.dumps(
        {'pairs': {'iter_0005|heuristic': [1.0, 2]}, 'logged': ['iter_0005'], 'since': 5, 'select': 'visits-b1'}))
    with open(src / 'metrics.jsonl', 'w') as f:
        for k in range(n):
            f.write(json.dumps({'iter': k}) + '\n')
    return src


def test_fork_at_an_older_iteration_cuts_the_elo_pools_at_it(tmp_path):
    """fork_run --iter 3 of a source rated up to 5: every pool loses the pairs and the `logged` names of checkpoints
    after 3 (the fork's own iter_0004, iter_0005 play their matches and enter the pool); a rule's pool begun after 3
    loses its `since`; ratings.json is not copied (refit) - before the fix all of it came along."""
    from zero import elo
    from zero.tools import fork_run
    src, dst = _source(tmp_path), tmp_path / 'dst'
    cfg = {'eval_every': 1, 'window': 2, 'window_grow': 1, 'game': 'toy', 'min_window_samples': 0}
    fork_run._copy_run(str(src), str(dst), 3, cfg, copy=True)
    got = json.loads((dst / 'elo' / 'results.json').read_text())
    assert [json.loads(x)['iter'] for x in (dst / 'metrics.jsonl').read_text().splitlines()] == [0, 1, 2, 3]
    assert set(got['pairs']) == {'iter_0002|iter_0003', 'heuristic|random'} and got['logged'] == ['iter_0003']
    assert not elo.has(got, 'iter_0005', 'iter_0004') and got['anchor'] == 'heuristic'
    rule = json.loads((dst / 'elo' / 'visits-b1' / 'results.json').read_text())
    assert rule['pairs'] == {} and rule['logged'] == [] and 'since' not in rule and rule['select'] == 'visits-b1'
    assert not (dst / 'elo' / 'ratings.json').exists()


def test_fork_copies_the_window_its_own_settings_read(tmp_path):
    """The data directories follow the fork's settings (window after --set, min_window_samples), not only the
    source's window; never a quarantined shard."""
    from zero.tools import fork_run
    src = _source(tmp_path, n=8, rows=10)
    (src / 'data' / 'iter_0007' / 'c00_001.bad.npz').write_bytes(b'bad')
    cfg = {'eval_every': 1, 'window': 2, 'window_grow': 0, 'game': 'toy', 'min_window_samples': 0}
    dst = tmp_path / 'a'
    fork_run._copy_run(str(src), str(dst), 7, cfg, copy=True, settings=dict(cfg, window=5))
    # (window 5 at iteration 8: its own directory and four behind it, plus one more as before)
    assert sorted(os.listdir(dst / 'data')) == [f'iter_{k:04d}' for k in range(3, 8)]
    assert os.listdir(dst / 'data' / 'iter_0007') == ['c00_000.npz']
    dst = tmp_path / 'b'                                 # 45 rows needed: 5 directories of 10 behind the new one
    fork_run._copy_run(str(src), str(dst), 7, cfg, copy=True, settings=dict(cfg, min_window_samples=45))
    assert sorted(os.listdir(dst / 'data')) == [f'iter_{k:04d}' for k in range(3, 8)]


# ---------------------------------------------------------------- remote evaluation: the drain, the transport (5.2.1)

def _queued(its):
    from zero import remote
    ev = remote.RemoteEval.__new__(remote.RemoteEval)
    ev.lock, ev.queued, ev.name = threading.Lock(), set(its), 'mac'
    return ev


def test_the_end_of_run_drain_waits_and_says_what_stop_left_missing(tmp_path):
    """RemoteEval.drain: no time limit (an evaluation that comes back after a while is waited for, progress.json
    rewritten every poll); STOP during the wait: drain_remote_evals logs the iterations not recorded and raises (the
    run ends aborted, never 'done' with evaluations missing) - before the fix a 6 h timeout's False was dropped."""
    from zero.run import RemoteEvalsMissing, drain_remote_evals
    run = str(tmp_path)
    ev = _queued({12})
    polls = []
    threading.Timer(0.3, lambda: ev.queued.discard(12)).start()
    assert ev.drain(on_poll=polls.append, poll_s=0.05) is True and polls and polls[0] == [12]
    ev = _queued({12, 16})
    open(os.path.join(run, 'STOP'), 'w').close()
    logged = []
    with pytest.raises(RemoteEvalsMissing, match=r'remote mac: iterations \[12, 16\]'):
        drain_remote_evals(run, [ev], lambda where='': None, logged.append)
    assert 'remote evaluations NOT recorded (STOP during the end-of-run wait): remote mac: iterations [12, 16]' in logged[0]
    os.remove(os.path.join(run, 'STOP'))
    ev = _queued(set())
    drain_remote_evals(run, [ev], lambda where='': None, logged.append)
    assert len(logged) == 1


def test_the_transport_tells_a_failure_from_absent(monkeypatch, tmp_path):
    """Transport.listdir raises on ssh's own failure (before: an empty listing, so a new-looking mirror got this
    machine's pool files over its own); get raises when the copy fails and the file is there (or the check fails),
    False only when it is absent."""
    from zero import remote
    t = remote.Transport({'host': 'nobody@example.invalid'})

    class R:
        def __init__(self, rc, out='', err=''):
            self.returncode, self.stdout, self.stderr = rc, out, err
    monkeypatch.setattr(t, '_ssh', lambda sh, timeout: R(255, err='ssh: Could not resolve hostname'))
    with pytest.raises(RuntimeError, match='failed .exit 255.'):
        t.listdir('/x/models')
    monkeypatch.setattr(t, '_ssh', lambda sh, timeout: R(0, out=''))
    assert t.listdir('/x/missing') == []
    monkeypatch.setattr(t, '_ssh', lambda sh, timeout: R(0, out='a.pt\nb.pt\n'))
    assert t.listdir('/x/models') == ['a.pt', 'b.pt']
    monkeypatch.setattr(remote.subprocess, 'run', lambda *a, **k: R(1, err='lost connection'))
    for answer, want in (('no', False), ('yes', RuntimeError), (None, RuntimeError)):
        monkeypatch.setattr(t, '_ssh', lambda sh, timeout, a=answer: R(0, out=a + '\n') if a else R(255))
        if want is False:
            assert t.get('/x/results.json', str(tmp_path / 'r.json')) is False
        else:
            with pytest.raises(RuntimeError, match='scp .* failed'):
                t.get('/x/results.json', str(tmp_path / 'r.json'))


# ---------------------------------------------------------------- devices (docs/az.md 46.7)

def test_a_device_the_machine_does_not_have_is_refused(monkeypatch):
    import torch
    from zero.net import EncodingMismatch
    from zero.run import DeviceRefused, check_devices
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    check_devices({'train_device': 'cpu', 'infer_device': 'cpu', 'heldout_device': ''})
    for k in ('train_device', 'heldout_device', 'infer_device'):
        with pytest.raises(DeviceRefused, match=f"resolver_cfg: {k} 'cuda'"):
            check_devices({k: 'cuda'}, 'resolver_cfg: ')
    assert issubclass(DeviceRefused, EncodingMismatch)


NOCUDA = textwrap.dedent('''
    import torch
    torch.cuda.is_available = lambda: False
    import zero.run as R
    R.cli()
''')


def test_train_device_cuda_without_cuda_refuses_the_run(tmp_path):
    """train_device 'cuda' on a machine without CUDA: exit 78 at the start, nothing written (before: every
    iteration trained on the CPU, the only sign "device": "cpu" in the metrics)."""
    script = tmp_path / 'nocuda.py'
    script.write_text(NOCUDA)
    run = tmp_path / 'r'
    p = subprocess.run([sys.executable, str(script), '--run', str(run), *SMOKE, '--iters', '1', '--train-device', 'cuda'],
                       cwd=ROOT, capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH))
    assert p.returncode == runctl.EXIT_REFUSED, p.stdout + p.stderr
    assert "DeviceRefused: train_device 'cuda'" in p.stderr
    assert not (run / 'models' / 'iter_0000.pt').exists()
    assert runctl.read_state(str(run), 'run')['state'] == 'refused'


# ---------------------------------------------------------------- the DONE marker of a finished phase (docs/az.md 46.1)

def _phase_dir(tmp_path, rows=(5, 7)):
    d = tmp_path / 'data' / 'iter_0003'
    d.mkdir(parents=True)
    for i, n in enumerate(rows):
        np.savez(d / f'r00_{i:03d}.npz', meta=json.dumps({'rows': n}), player=np.zeros(n, np.int8))
    return d


def test_the_done_marker_is_verified_against_its_directory(tmp_path):
    from zero.run import read_phase_done, write_phase_done
    d = _phase_dir(tmp_path)
    sp = {'samples': 9, 'remote': {'samples': 3}, 'games': 4}
    write_phase_done(str(d), 3, sp, 12.5)
    rec, why = read_phase_done(str(d), 3)
    assert why is None and rec['selfplay'] == sp and rec['selfplay_s'] == 12.5
    assert rec['shards'] == {'r00_000.npz': {'bytes': os.path.getsize(d / 'r00_000.npz'), 'rows': 5},
                             'r00_001.npz': {'bytes': os.path.getsize(d / 'r00_001.npz'), 'rows': 7}}
    assert read_phase_done(str(d), 4)[1] == 'it is of iteration 3'
    assert read_phase_done(str(tmp_path / 'none'), 3) == (None, '')
    np.savez(d / 'm00-abcd_000.npz', meta=json.dumps({'rows': 1}), player=np.zeros(1, np.int8))
    assert 'the shards differ' in read_phase_done(str(d), 3)[1]
    os.remove(d / 'm00-abcd_000.npz')
    np.savez(d / 'r00_001.npz', meta=json.dumps({'rows': 6}), player=np.zeros(6, np.int8))
    assert 'r00_001.npz is' in read_phase_done(str(d), 3)[1]
    with pytest.raises(RuntimeError, match='hold 11 samples, the phase reported 12'):
        write_phase_done(str(d), 3, sp, 1.0)            # never a marker for a directory short of its samples
    (d / 'DONE.json').write_text('{"iter": 3, "shar')
    assert read_phase_done(str(d), 3)[1].startswith('unreadable')


CRASH_IN_TRAINING = textwrap.dedent('''
    import os, sys
    import zero.run as R
    t0 = R.Run.train_model
    def train_model(self, it, n):
        if it == int(os.environ['CRASH_AT']) and not os.path.exists(os.path.join(self.run, 'crashed')):
            open(os.path.join(self.run, 'crashed'), 'w').close()
            raise RuntimeError('test: crash in training')
        return t0(self, it, n)
    R.Run.train_model = train_model
    R.cli()
''')


@pytest.mark.parametrize('rolling', [0, 1])
def test_a_crash_in_training_resumes_without_playing_the_phase_again(tmp_path, rolling):
    """Iteration 2 crashes in its training: its self-play phase had finished (DONE.json in data/iter_0002). The
    restart trains on that directory as it is - its shards untouched, the metrics row its phase's (marked
    selfplay_resumed, its seconds counted) - instead of dropping it and playing the phase again."""
    script = tmp_path / 'crash.py'
    script.write_text(CRASH_IN_TRAINING)
    run = tmp_path / 'r'
    e = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH, CRASH_AT='2')
    p = subprocess.run([sys.executable, str(script), '--run', str(run), *SMOKE, '--iters', '3', '--rolling', str(rolling)],
                       cwd=ROOT, capture_output=True, text=True, timeout=300, env=e)
    assert p.returncode != 0 and 'test: crash in training' in p.stderr, p.stdout + p.stderr
    d = run / 'data' / 'iter_0002'
    done = json.loads((d / 'DONE.json').read_text())
    before = {f: os.stat(d / f).st_mtime_ns for f in os.listdir(d) if f.endswith('.npz')}
    assert before and set(before) == set(done['shards'])
    p = subprocess.run([sys.executable, str(script), '--run', str(run), *SMOKE, '--iters', '3', '--rolling', str(rolling)],
                       cwd=ROOT, capture_output=True, text=True, timeout=300, env=e)
    assert p.returncode == 0, p.stdout + p.stderr
    assert {f: os.stat(d / f).st_mtime_ns for f in os.listdir(d) if f.endswith('.npz')} == before
    m = read_metrics(str(run))
    assert [r['iter'] for r in m] == [0, 1, 2, 3]
    assert m[2]['selfplay'] == done['selfplay'] and m[2]['selfplay_resumed'] is True
    assert m[2]['phases']['selfplay_s'] == done['selfplay_s'] and m[2]['seconds'] > done['selfplay_s']
    assert 'selfplay_resumed' not in m[1] and 'selfplay_resumed' not in m[3]
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'iter 2: its self-play phase finished before the interruption (DONE.json' in log
    assert log.count('iter 2 selfplay:') == 1


def test_a_done_marker_that_does_not_match_replays_the_phase(tmp_path):
    """A marker whose directory changed (a shard gone): logged, the phase played again as before the marker."""
    script = tmp_path / 'crash.py'
    script.write_text(CRASH_IN_TRAINING)
    run = tmp_path / 'r'
    e = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH, CRASH_AT='1')
    p = subprocess.run([sys.executable, str(script), '--run', str(run), *SMOKE, '--iters', '1'], cwd=ROOT,
                       capture_output=True, text=True, timeout=300, env=e)
    assert p.returncode != 0
    d = run / 'data' / 'iter_0001'
    os.remove(sorted(d.glob('*.npz'))[0])
    p = subprocess.run([sys.executable, str(script), '--run', str(run), *SMOKE, '--iters', '1'], cwd=ROOT,
                       capture_output=True, text=True, timeout=300, env=e)
    assert p.returncode == 0, p.stdout + p.stderr
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'data' in log and 'DONE.json does not match its directory (the shards differ' in log
    assert log.count('iter 1 selfplay:') == 2 and 'selfplay_resumed' not in read_metrics(str(run))[1]


# ---------------------------------------------------------------- the supervisor (docs/az.md 46.2)

SAME_CRASH = '''import os, sys
sys.path.insert(0, os.environ["CORE_PY"])
from zero import runctl
run = sys.argv[sys.argv.index("--run") + 1]
p = os.path.join(run, "count")
k = open(p).read() if os.path.exists(p) else ""
open(p, "w").write(k + "x")
runctl.write_state(run, "run", state="running")
if len(k) == 0:
    runctl.write_state(run, "run", state="aborted", error="RuntimeError: a first, different crash")
else:
    runctl.write_state(run, "run", state="aborted",
                       error=f"NonFinite: iter 7 training step 3 of 40: non-finite value (policy {0.1 + len(k)}, value nan)")
sys.exit(1)
'''


def test_the_supervisor_gives_up_on_the_same_crash_three_times(tmp_path):
    """A run that crashes the same way every time (its error text, decimals masked): after the third in a row the
    supervisor stops - state 'refused' with the error - instead of restarting it forever; a different crash first
    does not count."""
    run = tmp_path / 'sup'
    run.mkdir()
    (run / 'config.json').write_text('{}')
    fake = tmp_path / 'fake.py'
    fake.write_text(SAME_CRASH)
    env = dict(os.environ, ZERO_SUPERVISE_WAIT='0.1', ZERO_SUPERVISE_BACKOFF='0.1', CORE_PY=os.path.join(ROOT, 'python'),
               ZERO_SUPERVISE_CMD=json.dumps([sys.executable, str(fake)]))
    sup = subprocess.run([sys.executable, 'tools/supervise.py', str(run)], cwd=ROOT, env=env, timeout=120)
    assert sup.returncode == 0
    assert (run / 'count').read_text() == 'xxxx'
    st = runctl.read_state(str(run), 'supervisor')
    assert st['state'] == 'refused' and st['why'] == 'the same crash 3 times in a row' and 'NonFinite' in st['error']
    log = (run / 'supervisor.log').read_text()
    assert 'the same crash 3 times in a row' in log and 'not restarting' in log


# ---------------------------------------------------------------- state files (B20)

def test_run_state_files_are_written_atomically(monkeypatch, tmp_path, capsys):
    """_write_json: a file nobody could replace raises when a resume needs it (strict) and leaves no temporary file
    either way; progress.json's miss is said on stderr."""
    from zero import run as R
    monkeypatch.setattr(R.time, 'sleep', lambda s: None)

    def deny(a, b):
        raise PermissionError('held open')
    monkeypatch.setattr(R.os, 'replace', deny)
    p = str(tmp_path / 'iter_0003.json')
    with pytest.raises(PermissionError):
        R._write_json(p, {'iter': 3}, strict=True)
    R._write_json(str(tmp_path / 'progress.json'), {'phase': 'train'})
    assert os.listdir(tmp_path) == []
    assert 'progress.json: not rewritten' in capsys.readouterr().err


# ---------------------------------------------------------------- metrics of a rolling phase (docs/az.md 46.9)

def test_rolling_phases_report_counters_not_mixed_ratios(tmp_path):
    """A rolling driver's counters cover every game in flight and its finished games include ones begun earlier: its
    phase reports the minor decisions as a count (`minor`), not per finished game; a fresh driver's phase (rolling 0)
    keeps `minor_per_game` (exact: it started and finished every game)."""
    run = tmp_path / 'roll'
    p = run_loop(run, '--iters', '2', '--rolling', '1')
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    assert all('minor' in r['selfplay'] and 'minor_per_game' not in r['selfplay'] for r in m[1:])
    run = tmp_path / 'fresh'
    p = run_loop(run, '--iters', '1')
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    assert 'minor_per_game' in m[1]['selfplay'] and 'minor' not in m[1]['selfplay']
