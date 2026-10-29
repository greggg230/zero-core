"""Keep an AlphaZero run going unattended.

  python tools/supervise.py runs/toy1 [extra <pkg>.az.run args...]     (in a game repo)

* runs `python -m <pkg>.az.run --run <run> ...` (the game package's `run_module`, from the
  game repo's root; the core alone: `python -m zero.run`) and restarts it (resuming from the
  newest checkpoint) whenever it exits with an error, after killing any orphaned workers;
* backs off if it keeps crashing (5 crashes within an hour -> wait 15 minutes);
* pause / resume (zero/runctl.py): when the run exits paused (status 75: a PAUSE file in
  the run dir; tools/pause_run.py, the dashboard's Pause button) - or a PAUSE file is
  present when it would (re)start the run - it waits, polling every second, until PAUSE is
  removed and then relaunches the run at once (not a crash: no restart delay, no back-off);
  a STOP file appearing while it waits ends the supervisor (the run stays paused);
* asks Windows not to sleep while it is running (also while paused: Resume from a phone
  needs the desktop awake);
* stops when the run exits cleanly (finished, or a STOP file in the run dir), and when it
  refuses its checkpoint (exit status 78, runctl.EXIT_REFUSED: a network of another encoding
  than this build's, docs/az.md 10.4 - restarting would refuse again);
* optional stall watchdog (<P>_SUPERVISE_STALL seconds, default 0 = off): a run whose progress
  files (progress.json, log.txt, stdout.txt) have not changed for that long while no PAUSE is
  requested is killed (with its workers) and restarted like a crash - a hung run (e.g. a
  deadlocked search driver) otherwise stays alive forever. Self-play rewrites progress.json
  every few seconds but training and evaluation phases may write nothing for a long time:
  pick a limit well above the longest of those (e.g. 3600).
A STOP file left over from an earlier stop is moved to STOP.old when the supervisor starts
(starting it again means: run); a PAUSE file is kept (the run stays paused until it goes).
The run's console output goes to <run>/stdout.txt in UTF-8. Logs to <run>/supervisor.log;
its state (for the dashboard) is in <run>/supervisor_state.json. The environment variables use
the game package's prefix (GRZ_, CVZ_; the core: ZERO_): <P>_SUPERVISE_WAIT /
<P>_SUPERVISE_BACKOFF override the restart delays (60 s / 900 s; tests use 1 s),
<P>_SUPERVISE_POLL the PAUSE polling interval (1 s), <P>_SUPERVISE_CMD (a JSON list) the
run command (tests: a fake run).
"""
import ctypes
import json
import os
import subprocess
import sys
import time

import psutil

from .. import package, runctl  # light: no torch


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


def last_activity(run, since):
    """Newest modification time of the run's progress files, at least `since`."""
    t = since
    for f in ('progress.json', 'log.txt', 'stdout.txt'):
        try:
            t = max(t, os.path.getmtime(os.path.join(run, f)))
        except OSError:
            pass
    return t


def wait_run(run, p, stall_s):
    """Wait for the run process `p` to exit -> its exit code. With stall_s > 0 a run without
    progress for stall_s seconds (no PAUSE requested) is killed with its children first."""
    if stall_s <= 0:
        return p.wait()
    t0 = time.time()
    while True:
        try:
            return p.wait(timeout=min(max(stall_s / 10, 0.05), 30))
        except subprocess.TimeoutExpired:
            pass
        idle = time.time() - last_activity(run, t0)
        if idle > stall_s and not runctl.pause_requested(run):
            log(run, f'stall watchdog: no progress for {ago(idle)} (limit {ago(stall_s)}): killing pid {p.pid}')
            try:
                for k in psutil.Process(p.pid).children(recursive=True):
                    k.kill()
            except psutil.NoSuchProcess:
                pass
            p.kill()
            code = p.wait()
            return code if code else 1


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


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    pkg = package.active()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    if not argv or argv[0].startswith('-'):
        # `--help` (or a missing run dir) must not become a run directory that crash-loops
        print(__doc__)
        sys.exit(0 if argv and argv[0] in ('-h', '--help') else 2)
    run = argv[0]
    extra = argv[1:]
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
    wait_s = float(pkg.env('SUPERVISE_WAIT', 60))
    backoff_s = float(pkg.env('SUPERVISE_BACKOFF', 900))
    poll_s = float(pkg.env('SUPERVISE_POLL', 1))
    stall_s = float(pkg.env('SUPERVISE_STALL', 0))
    cmd = json.loads(pkg.env('SUPERVISE_CMD')) if pkg.env('SUPERVISE_CMD') else \
        [sys.executable, '-m', pkg.run_module]
    name = pkg.run_module
    # the run imports the same core as this supervisor (a game's run would find it through its
    # package anyway; `python -m zero.run` needs it)
    core = os.path.join(package.CORE_ROOT, 'python')
    env = dict(os.environ, PYTHONIOENCODING='utf-8',
               PYTHONPATH=os.pathsep.join([core] + [p for p in [os.environ.get('PYTHONPATH')] if p]))
    while True:
        if runctl.pause_requested(run):
            log(run, 'PAUSE file present: the run stays paused until it is removed')
            if not wait_resume(run, poll_s):
                return
        out = open(os.path.join(run, 'stdout.txt'), 'a', encoding='utf-8')
        p = subprocess.Popen(cmd + ['--run', run] + extra, cwd=pkg.root, stdout=out, stderr=subprocess.STDOUT, env=env)
        log(run, f'started {name} pid {p.pid}')
        runctl.write_state(run, 'supervisor', state='running', run_pid=p.pid)
        code = wait_run(run, p, stall_s)
        out.close()
        killed = kill_orphans(p.pid)
        log(run, f'{name} exited with code {code}; killed {killed} orphaned workers')
        if code == 0:
            log(run, 'clean exit (finished or STOP file); supervisor done')
            runctl.write_state(run, 'supervisor', state='done', run_pid=None)
            return
        if os.path.exists(stop):
            log(run, 'STOP file present; not restarting')
            runctl.write_state(run, 'supervisor', state='done', run_pid=None)
            return
        if code == runctl.EXIT_REFUSED:
            log(run, f'the run REFUSED its checkpoint (exit code {code}: a network of another encoding than this '
                     f"build's, see log.txt; docs/az.md 10.4): not restarting, supervisor done")
            runctl.write_state(run, 'supervisor', state='refused', run_pid=None)
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
