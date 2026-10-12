"""Keep an AlphaZero run going unattended.

  python tools/supervise.py runs/toy1 [extra domz.az.run args...]

* runs `python -m domz.az.run --run <run> ...` and restarts it (resuming from the newest
  checkpoint) whenever it exits with an error, after killing any orphaned workers;
* backs off if it keeps crashing (5 crashes within an hour -> wait 15 minutes);
* pause / resume (domz/az/runctl.py): when the run exits paused (status 75: a PAUSE file in
  the run dir; tools/pause_run.py, the dashboard's Pause button) - or a PAUSE file is
  present when it would (re)start the run - it waits, polling every second, until PAUSE is
  removed and then relaunches the run at once (not a crash: no restart delay, no back-off);
  a STOP file appearing while it waits ends the supervisor (the run stays paused);
* asks Windows not to sleep while it is running (also while paused: Resume from a phone
  needs the desktop awake);
* stops when the run exits cleanly (finished, or a STOP file in the run dir).
A STOP file left over from an earlier stop is moved to STOP.old when the supervisor starts
(starting it again means: run); a PAUSE file is kept (the run stays paused until it goes).
The run's console output goes to <run>/stdout.txt in UTF-8. Logs to <run>/supervisor.log;
its state (for the dashboard) is in <run>/supervisor_state.json. DOMZ_SUPERVISE_WAIT /
DOMZ_SUPERVISE_BACKOFF override the restart delays (60 s / 900 s; tests use 1 s),
DOMZ_SUPERVISE_POLL the PAUSE polling interval (1 s), DOMZ_SUPERVISE_CMD (a JSON list) the
run command (tests: a fake run).
"""
import ctypes
import json
import os
import subprocess
import sys
import time

import psutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from domz.az import runctl  # noqa: E402  (light: no torch)


def log(run, msg):
    with open(os.path.join(run, 'supervisor.log'), 'a', encoding='utf-8') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + msg + '\n')


def keep_awake():
    if os.name == 'nt':
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def kill_orphans(pid):
    """Pool workers of a dead run keep its pid as their parent pid."""
    try:                                   # one snapshot (~0.1 s); process_iter's first pass takes ~10 s on Windows
        from psutil import _psplatform
        kids = [c for c, pp in _psplatform.ppid_map().items() if pp == pid]
    except Exception:                      # noqa: BLE001 - platforms without ppid_map
        kids = [p.pid for p in psutil.process_iter(['pid', 'ppid']) if p.info['ppid'] == pid]
    n = 0
    for c in kids:
        try:
            p = psutil.Process(c)
            for k in p.children(recursive=True):
                k.kill()
            p.kill()
            n += 1
        except psutil.NoSuchProcess:
            pass
    return n


def ago(s):
    return f'{s:.0f}s' if s < 120 else f'{s / 60:.1f} min' if s < 7200 else f'{s / 3600:.1f} h'


def wait_resume(run, poll_s):
    """The run is paused: wait until its PAUSE file is removed -> True (relaunch now), or
    False if a STOP file appears meanwhile (the supervisor ends; the run stays paused)."""
    pause = runctl.pause_path(run)
    t0 = time.time()
    since = runctl.pause_since(run) or t0
    runctl.write_state(run, 'supervisor', state='paused', run_pid=None, paused_since=since)
    while os.path.exists(pause):
        if runctl.stop_requested(run):
            log(run, 'STOP file present while paused: supervisor done (the run stays paused; relaunching the '
                     'supervisor waits for PAUSE to be removed)')
            runctl.write_state(run, 'supervisor', state='done')
            return False
        time.sleep(poll_s)
    log(run, f'PAUSE removed after {ago(time.time() - since)} paused: resuming the run')
    runctl.write_state(run, 'supervisor', state='resuming')
    return True


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    if len(sys.argv) < 2 or sys.argv[1].startswith('-'):
        # `--help` (or a missing run dir) must not become a run directory that crash-loops
        print(__doc__)
        sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help') else 2)
    run = sys.argv[1]
    extra = sys.argv[2:]
    os.makedirs(run, exist_ok=True)
    keep_awake()
    crashes = []
    log(run, f'supervisor {os.getpid()} starting: {run} {" ".join(extra)}; open-file limit '
             f'{runctl.raise_nofile() or "n/a"} (soft, hard)')
    runctl.write_state(run, 'supervisor', state='starting', run_pid=None, args=extra)
    stop = os.path.join(run, 'STOP')
    if os.path.exists(stop):
        os.replace(stop, stop + '.old')
        log(run, 'moved a STOP file left from an earlier stop to STOP.old')
    wait_s = float(os.environ.get('DOMZ_SUPERVISE_WAIT', 60))
    backoff_s = float(os.environ.get('DOMZ_SUPERVISE_BACKOFF', 900))
    poll_s = float(os.environ.get('DOMZ_SUPERVISE_POLL', 1))
    cmd = json.loads(os.environ['DOMZ_SUPERVISE_CMD']) if os.environ.get('DOMZ_SUPERVISE_CMD') else \
        [sys.executable, '-m', 'domz.az.run']
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    while True:
        if runctl.pause_requested(run):
            log(run, 'PAUSE file present: the run stays paused until it is removed')
            if not wait_resume(run, poll_s):
                return
        out = open(os.path.join(run, 'stdout.txt'), 'a', encoding='utf-8')
        p = subprocess.Popen(cmd + ['--run', run] + extra, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, env=env)
        log(run, f'started domz.az.run pid {p.pid}')
        runctl.write_state(run, 'supervisor', state='running', run_pid=p.pid)
        code = p.wait()
        out.close()
        killed = kill_orphans(p.pid)
        log(run, f'domz.az.run exited with code {code}; killed {killed} orphaned workers')
        if code == 0:
            log(run, 'clean exit (finished or STOP file); supervisor done')
            runctl.write_state(run, 'supervisor', state='done', run_pid=None)
            return
        if os.path.exists(stop):
            log(run, 'STOP file present; not restarting')
            runctl.write_state(run, 'supervisor', state='done', run_pid=None)
            return
        if code == runctl.EXIT_PAUSED or runctl.pause_requested(run):
            log(run, f'run paused (exit code {code}): waiting for the PAUSE file to be removed '
                     f'(polling every {poll_s:g}s)')
            if not wait_resume(run, poll_s):
                return
            continue
        now = time.time()
        crashes = [t for t in crashes if now - t < 3600] + [now]
        wait = backoff_s if len(crashes) >= 5 else wait_s
        log(run, f'crash #{len(crashes)} in the last hour; restarting in {wait}s')
        runctl.write_state(run, 'supervisor', state='restarting', run_pid=None, restart_at=now + wait)
        time.sleep(wait)


if __name__ == '__main__':
    main()
