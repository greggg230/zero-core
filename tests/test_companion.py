"""The supervisor's companion process (docs/az.md 49): one per run across the run's restarts and pauses, started
hidden when the run has written its config.json, ended with the run - its last word 'finished' when the run reached
its last iteration, 'stopped' on STOP, when the supervisor is killed (the companion notices) or kill_run hits it -,
and never the run's end when it cannot start (a loud line, the run goes on). Fake runs and fake companions
(ZERO_SUPERVISE_CMD / ZERO_SUPERVISE_COMPANION)."""
import json
import os
import subprocess
import sys
import time

import psutil
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'python'))
from zero import runctl  # noqa: E402

# a fake run: `mode` file in the run dir - 'finish' (runs ~1.5 s, exit 0), 'crash-then-finish' (exit 1 the first
# time), 'until-stop' (exit 0 once STOP exists), 'pause' (exit 75 while PAUSE exists, else finish), 'forever'
FAKE_RUN = '''import os, sys, time
sys.path.insert(0, os.environ["CORE_PY"])
from zero import runctl
run = sys.argv[sys.argv.index("--run") + 1]
mode = open(os.path.join(run, "mode")).read().strip()
n = os.path.join(run, "starts")
k = len(open(n).read()) if os.path.exists(n) else 0
open(n, "a").write("x")
runctl.write_state(run, "run", state="running")
if mode == "crash-then-finish" and k == 0:
    time.sleep(1.0)
    sys.exit(1)
if mode == "pause":
    t0 = time.time()
    while time.time() - t0 < 1.5:
        if runctl.pause_requested(run):
            runctl.write_state(run, "run", state="paused")
            sys.exit(75)
        time.sleep(0.05)
    if k == 0:                                   # the first start waits for the test's PAUSE
        while not runctl.pause_requested(run):
            time.sleep(0.05)
        runctl.write_state(run, "run", state="paused")
        sys.exit(75)
if mode == "until-stop":
    while not runctl.stop_requested(run):
        time.sleep(0.05)
if mode == "forever":
    while True:
        time.sleep(0.5)
time.sleep(1.5)
runctl.write_state(run, "run", state="done")
sys.exit(0)
'''

# a fake companion: logs its start, what it sees while the run is paused, and how the run ended (runctl.supervisor_end)
FAKE_COMPANION = '''import os, sys, time
sys.path.insert(0, os.environ["CORE_PY"])
from zero import runctl
run = os.environ["ZERO_COMPANION_RUN"]
ev = os.path.join(run, "comp_events.txt")
def say(s):
    with open(ev, "a") as f:
        f.write(f"{os.getpid()} {s}\\n")
say("start")
if os.environ.get("COMP_EXIT"):
    sys.exit(int(os.environ["COMP_EXIT"]))
seen = False
while True:
    e = runctl.supervisor_end()
    if e:
        say("end " + e)
        sys.exit(0)
    if not seen and runctl.run_status(run)["state"] == "paused":
        seen = True
        say("paused")
    time.sleep(0.1)
'''


def _env(tmp_path, companion=None, **more):
    fake = tmp_path / 'fake_run.py'
    fake.write_text(FAKE_RUN)
    comp = tmp_path / 'fake_companion.py'
    comp.write_text(FAKE_COMPANION)
    env = dict(os.environ, ZERO_SUPERVISE_WAIT='0.1', ZERO_SUPERVISE_BACKOFF='0.1', ZERO_SUPERVISE_POLL='0.1',
               ZERO_SUPERVISE_COMPANION_POLL='0.2', ZERO_SUPERVISE_COMPANION_STOP_WAIT='20',
               ZERO_SUPERVISE_COMPANION_WAIT='20', CORE_PY=os.path.join(ROOT, 'python'),
               ZERO_SUPERVISE_CMD=json.dumps([sys.executable, str(fake)]),
               ZERO_SUPERVISE_COMPANION=json.dumps(companion if companion is not None else
                                                   {'name': 'fakepub', 'cmd': [sys.executable, str(comp)]}))
    env.update(more)
    return env


def _run(tmp_path, mode):
    run = tmp_path / 'r'
    run.mkdir()
    (run / 'config.json').write_text('{}')
    (run / 'mode').write_text(mode)
    return run


def _events(run):
    p = run / 'comp_events.txt'
    return [line.split(' ', 1) for line in p.read_text().splitlines()] if p.exists() else []


def _text(p):
    return p.read_text() if p.exists() else ''


def _wait(cond, timeout=60, what=''):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return
        time.sleep(0.1)
    raise AssertionError(f'timed out waiting for {what}')


LAUNCHER = '''import subprocess, sys
sys.exit(subprocess.run([sys.executable] + sys.argv[1:]).returncode)
'''


@pytest.mark.live_run_hazard
@pytest.mark.parametrize('launcher', [False, True])
def test_one_companion_across_a_crash_and_it_ends_finished(tmp_path, launcher):
    """The run crashes once and is restarted: the same companion goes on (one per run); the run finishes: the
    supervisor's terminal state says `ended` 'finished', the companion reads it, says so and exits, and the
    supervisor waited for it. Also when the launched process starts the run as its child (a Windows venv's
    python.exe: the run state's pid is not the launched one; supervise.launched_state, docs/az.md 47.2)."""
    run = _run(tmp_path, 'crash-then-finish')
    env = _env(tmp_path)
    if launcher:
        (tmp_path / 'launcher.py').write_text(LAUNCHER)
        env['ZERO_SUPERVISE_CMD'] = json.dumps([sys.executable, str(tmp_path / 'launcher.py'), str(tmp_path / 'fake_run.py')])
    sup = subprocess.run([sys.executable, 'tools/supervise.py', str(run)], cwd=ROOT, env=env, timeout=120)
    assert sup.returncode == 0
    assert (run / 'starts').read_text() == 'xx'
    ev = _events(run)
    assert [e[1] for e in ev] == ['start', 'end finished'] and ev[0][0] == ev[1][0]
    st = runctl.read_state(str(run), 'supervisor')
    assert st['state'] == 'done' and st['ended'] == 'finished' and st['companion'] is None
    log = (run / 'supervisor.log').read_text()
    assert log.count('companion fakepub started') == 1 and 'companion fakepub ended with code 0' in log
    assert (run / 'fakepub_console.txt').exists()


@pytest.mark.live_run_hazard
def test_stop_ends_the_companion_stopped(tmp_path):
    run = _run(tmp_path, 'until-stop')
    sup = subprocess.Popen([sys.executable, 'tools/supervise.py', str(run)], cwd=ROOT, env=_env(tmp_path))
    try:
        _wait(lambda: [e[1] for e in _events(run)] == ['start'], what='the companion')
        (run / 'STOP').write_text('')
        assert sup.wait(timeout=60) == 0
    finally:
        if sup.poll() is None:
            sup.kill()
    assert [e[1] for e in _events(run)] == ['start', 'end stopped']
    assert runctl.read_state(str(run), 'supervisor')['ended'] == 'stopped'


@pytest.mark.live_run_hazard
def test_the_companion_stays_up_while_the_run_is_paused(tmp_path):
    """PAUSE: the run exits 75, the supervisor waits, the companion stays (it sees the run paused); resumed, the same
    companion goes on to the run's end."""
    run = _run(tmp_path, 'pause')
    sup = subprocess.Popen([sys.executable, 'tools/supervise.py', str(run)], cwd=ROOT, env=_env(tmp_path))
    try:
        _wait(lambda: [e[1] for e in _events(run)] == ['start'], what='the companion')
        runctl.request_pause(str(run))
        _wait(lambda: 'paused' in [e[1] for e in _events(run)], what='the companion seeing the pause')
        pid = int(_events(run)[0][0])
        time.sleep(1.0)
        assert psutil.pid_exists(pid) and runctl.read_state(str(run), 'supervisor')['state'] == 'paused'
        runctl.request_resume(str(run))
        assert sup.wait(timeout=60) == 0
    finally:
        if sup.poll() is None:
            sup.kill()
    ev = _events(run)
    assert [e[1] for e in ev] == ['start', 'paused', 'end finished'] and len({e[0] for e in ev}) == 1


def _kill_runs(run):
    rs = runctl.read_state(str(run), 'run')
    if runctl.alive(rs):
        psutil.Process(rs['pid']).kill()


@pytest.mark.live_run_hazard
def test_a_killed_supervisor_leaves_a_companion_that_ends_itself_stopped(tmp_path):
    run = _run(tmp_path, 'forever')
    sup = subprocess.Popen([sys.executable, 'tools/supervise.py', str(run)], cwd=ROOT, env=_env(tmp_path))
    try:
        _wait(lambda: [e[1] for e in _events(run)] == ['start'], what='the companion')
        sup.kill()                                   # abruptly: no terminal state written
        sup.wait()
        _wait(lambda: [e[1] for e in _events(run)] == ['start', 'end stopped'], timeout=30, what='the companion ending')
        assert runctl.read_state(str(run), 'supervisor')['state'] == 'running'
    finally:
        if sup.poll() is None:
            sup.kill()
        _kill_runs(run)


@pytest.mark.live_run_hazard
def test_kill_run_spares_the_companion_which_ends_itself_stopped(tmp_path):
    run = _run(tmp_path, 'forever')
    env = _env(tmp_path)
    sup = subprocess.Popen([sys.executable, 'tools/supervise.py', str(run)], cwd=ROOT, env=env)
    try:
        _wait(lambda: [e[1] for e in _events(run)] == ['start'], what='the companion')
        _wait(lambda: (runctl.read_state(str(run), 'supervisor') or {}).get('companion'), what='its state')
        out = subprocess.run([sys.executable, 'tools/kill_run.py', str(run)], cwd=ROOT, env=env, capture_output=True,
                             text=True, timeout=60)
        assert 'left the companion' in out.stdout, out.stdout + out.stderr
        _wait(lambda: [e[1] for e in _events(run)] == ['start', 'end stopped'], timeout=30, what='the companion ending')
    finally:
        if sup.poll() is None:
            sup.kill()
        _kill_runs(run)


@pytest.mark.live_run_hazard
def test_a_companion_that_cannot_start_is_loud_and_the_run_goes_on(tmp_path):
    run = _run(tmp_path, 'finish')
    env = _env(tmp_path, companion={'name': 'nopub', 'cmd': [str(tmp_path / 'no-such-program.exe')]})
    sup = subprocess.run([sys.executable, 'tools/supervise.py', str(run)], cwd=ROOT, env=env, timeout=120,
                         capture_output=True, text=True)
    assert sup.returncode == 0 and 'COMPANION NOT STARTED (nopub)' in sup.stderr
    st = runctl.read_state(str(run), 'supervisor')
    assert st['ended'] == 'finished' and 'its command failed' in st['companion_error']
    assert 'COMPANION NOT STARTED (nopub): its command failed' in (run / 'supervisor.log').read_text()


@pytest.mark.live_run_hazard
def test_a_companion_that_keeps_exiting_is_restarted_three_times_then_given_up(tmp_path):
    run = _run(tmp_path, 'until-stop')
    sup = subprocess.Popen([sys.executable, 'tools/supervise.py', str(run)], cwd=ROOT,
                           env=_env(tmp_path, COMP_EXIT='3'), stderr=subprocess.PIPE, text=True)
    try:
        _wait(lambda: 'COMPANION NOT STARTED' in _text(run / 'supervisor.log'), what='giving up')
        (run / 'STOP').write_text('')
        _, err = sup.communicate(timeout=60)
    finally:
        if sup.poll() is None:
            sup.kill()
    assert [e[1] for e in _events(run)] == ['start'] * 4
    log = (run / 'supervisor.log').read_text()
    assert log.count('exited with code 3 while the run goes on: starting it again') == 3
    assert 'exited with code 3 4 times within an hour' in log and 'COMPANION NOT STARTED (fakepub)' in err


@pytest.mark.live_run_hazard
def test_none_means_no_companion_and_the_run_as_before(tmp_path):
    run = _run(tmp_path, 'finish')
    sup = subprocess.run([sys.executable, 'tools/supervise.py', str(run)], cwd=ROOT, env=_env(tmp_path, companion={}),
                         timeout=120)
    assert sup.returncode == 0 and not _events(run)
    log = (run / 'supervisor.log').read_text()
    assert 'companion fakepub' not in log and 'COMPANION' not in log


@pytest.mark.live_run_hazard
def test_a_new_supervisor_stops_the_companion_a_killed_one_left(tmp_path):
    run = _run(tmp_path, 'finish')
    left = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
    try:
        dead = subprocess.Popen([sys.executable, '-c', 'pass'])
        dead.wait()
        (run / 'supervisor_state.json').write_text(json.dumps({
            'pid': dead.pid, 'started': 1.0, 'state': 'running',
            'companion': {'name': 'old', 'pid': left.pid, 'started': psutil.Process(left.pid).create_time()}}))
        sup = subprocess.run([sys.executable, 'tools/supervise.py', str(run)], cwd=ROOT, env=_env(tmp_path, companion={}),
                             timeout=120)
        assert sup.returncode == 0
        assert left.wait(timeout=10) != 0
        assert f'stopped companion old pid {left.pid} left by supervisor {dead.pid}' in (run / 'supervisor.log').read_text()
    finally:
        if left.poll() is None:
            left.kill()


def test_supervisor_end_reads_the_terminal_state_then_the_process(tmp_path):
    run = str(tmp_path)
    me = os.getpid()
    started = psutil.Process(me).create_time()
    runctl.write_state(run, 'supervisor', state='running')
    assert runctl.supervisor_end(run, me, started) is None
    runctl.write_state(run, 'supervisor', state='paused')
    assert runctl.supervisor_end(run, me, started) is None
    runctl.write_state(run, 'supervisor', state='done', ended='finished')
    assert runctl.supervisor_end(run, me, started) == 'finished'
    runctl.write_state(run, 'supervisor', state='refused', ended='gave_up')
    assert runctl.supervisor_end(run, me, started) == 'gave_up'
    runctl.write_state(run, 'supervisor', state='done', ended=None)
    assert runctl.supervisor_end(run, me, started) == 'stopped'
    # the supervisor process gone without a terminal state, or another supervisor's run now: stopped
    runctl.write_state(run, 'supervisor', state='running')
    assert runctl.supervisor_end(run, me, started + 100) == 'stopped'
    assert runctl.supervisor_end(run, me + 1, started) == 'stopped'
    with pytest.raises(ValueError):
        runctl.supervisor_end(None, None, None)


def test_the_package_hook_is_validated():
    from zero.package import GamePackage
    g = GamePackage('zz', 'Z', ROOT, companion=lambda run, cfg: cfg.get('c'))
    assert g.companion_of('/r', {}) is None
    assert g.companion_of('/r', {'c': {'cmd': ['x', 'y']}}) == {'name': 'companion', 'cmd': ['x', 'y'],
                                                               'log': 'companion_console.txt', 'env': {}}
    assert g.companion_of('/r', {'c': {'cmd': ['x'], 'name': 'pub', 'env': {'A': 1}}})['log'] == 'pub_console.txt'
    with pytest.raises(ValueError):
        g.companion_of('/r', {'c': {'cmd': 'x y'}})
