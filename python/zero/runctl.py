"""Run control: the PAUSE / STOP files, the paused exit status and a run's state
(docs/az.md 2; ported from Dominion Zero's domz/az/runctl.py).

Two control files in a run directory:

* STOP  - end the run cleanly after the current iteration; the supervisor exits with it.
* PAUSE - give the machine back within seconds and continue later without relaunching
  anything: the run stops at its next pause point (between self-play batches, training steps,
  evaluation batches) and exits with EXIT_PAUSED. tools/supervise.py waits until PAUSE is
  removed, then relaunches the run, which resumes from its newest checkpoint (an iteration
  interrupted before its checkpoint is replayed; one interrupted after it finishes its
  evaluation).

State files, written atomically and read by the dashboard and tools/pause_run.py:

* <run>/run_state.json        - the run process: pid, started (process creation time),
  state (running | pausing | paused | done | aborted | refused), since, and the pause timings
* <run>/supervisor_state.json - tools/supervise.py: pid, started, state (running | paused |
  resuming | restarting | done | refused), run_pid, since; at a terminal state `ended` (finished | stopped |
  refused | gave_up) and, with a companion process (docs/az.md 49), `companion` / `companion_error`

`run_status(run)` combines them with the control files and the processes' liveness into one
state: running / pausing / paused / resuming / starting / restarting / stopping / stopped
(None when the run's process predates these files).
"""
import json
import os
import threading
import time

PAUSE = 'PAUSE'
STOP = 'STOP'
EXIT_PAUSED = 75                  # sysexits EX_TEMPFAIL: "try again later"
EXIT_REFUSED = 78                 # sysexits EX_CONFIG: the run refused its checkpoint (a network of another
                                  # encoding than this build's, docs/az.md 10.4): no restart fixes that
STATE_FILES = {'run': 'run_state.json', 'supervisor': 'supervisor_state.json'}


def raise_nofile(want=65536):
    """Raise this process's open-file soft limit toward `want`, within its hard limit
    (children inherit it) -> (soft, hard) after; None without `resource` (Windows). Containers
    often start with a soft limit of 1024 ; macOS with 256 (capped at 10240)."""
    try:
        import resource
    except ImportError:
        return None
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    inf = resource.RLIM_INFINITY
    for goal in (want, 10240, 4096):
        goal = goal if hard == inf else min(goal, hard)
        if soft == inf or soft >= goal:
            break
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (goal, hard))
            break
        except (ValueError, OSError):
            continue
    return tuple(resource.getrlimit(resource.RLIMIT_NOFILE))


class Paused(BaseException):
    """Raised in the run process once a PAUSE file is noticed; a BaseException so that no
    `except Exception` on the way swallows it."""


# ------------------------------------------------------------------ control files

def pause_path(run):
    return os.path.join(run, PAUSE)


def pause_requested(run):
    return os.path.exists(pause_path(run))


def stop_requested(run):
    return os.path.exists(os.path.join(run, STOP))


def is_run_dir(run):
    """A directory a run has used (config.json or metrics.jsonl): only those get a PAUSE."""
    return os.path.isdir(run) and any(os.path.isfile(os.path.join(run, f)) for f in ('config.json', 'metrics.jsonl'))


def request_pause(run, by=''):
    """Create <run>/PAUSE (an existing one is left as it is: its time is when the pause was
    asked for) -> True if it was created now."""
    p = pause_path(run)
    if os.path.exists(p):
        return False
    tmp = p + f'.{os.getpid()}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'time': time.time(), 'by': by}, f)
    try:
        os.replace(tmp, p)
    except OSError:
        os.remove(tmp)
        raise
    return True


def request_resume(run):
    """Remove <run>/PAUSE -> True if there was one."""
    try:
        os.remove(pause_path(run))
        return True
    except FileNotFoundError:
        return False


def pause_since(run):
    """When the pause was asked for (the PAUSE file's time), or None."""
    try:
        return os.path.getmtime(pause_path(run))
    except OSError:
        return None


# ------------------------------------------------------------------ state files

_lock = threading.Lock()


def _proc_started(pid=None):
    try:
        import psutil
        return psutil.Process(pid).create_time()
    except Exception:                              # noqa: BLE001
        return None


def write_state(run, who, **kw):
    """Update <run>/<who>_state.json for this process (`who`: 'run' | 'supervisor')."""
    path = os.path.join(run, STATE_FILES[who])
    with _lock:
        st = read_state(run, who) or {}
        if st.get('pid') != os.getpid():           # a new process: start afresh
            st = {'pid': os.getpid(), 'started': _proc_started()}
        if kw.get('state') and kw['state'] != st.get('state'):
            st['since'] = time.time()
        st.update(kw)
        st['updated'] = time.time()
        tmp = path + f'.{os.getpid()}.tmp'
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(st, f)
            for k in range(20):                    # a reader may hold the file open (Windows)
                try:
                    os.replace(tmp, path)
                    break
                except PermissionError:
                    time.sleep(0.05)
        except OSError:
            pass
    return st


def read_state(run, who):
    try:
        with open(os.path.join(run, STATE_FILES[who]), encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def alive(st):
    """The process a state file describes is still running (same pid AND creation time:
    pids are reused)."""
    if not st or not st.get('pid'):
        return False
    try:
        import psutil
        p = psutil.Process(int(st['pid']))
        if st.get('started') is not None and abs(p.create_time() - float(st['started'])) > 1.0:
            return False
        return p.status() != psutil.STATUS_ZOMBIE
    except Exception:                              # noqa: BLE001 - gone, or not ours to see
        return False


def run_status(run):
    """One state for the run (dashboard, tools/pause_run.py):

    running / stopping (STOP file) / pausing (PAUSE file, the run process still exiting) /
    paused (PAUSE file, no run process) / resuming (PAUSE removed, the supervisor about to
    relaunch) / starting / restarting (the supervisor's wait after a crash) / stopped
    (no run process, no PAUSE: finished, STOP, aborted or killed) - or None if the run has no
    state files (its process predates pause support: no control from here).
    -> dict(state, detail, pause, stop, paused_since, resumable, run, supervisor)."""
    rs, ss = read_state(run, 'run'), read_state(run, 'supervisor')
    # a run process that wrote 'paused' has stopped its workers and is exiting: not running
    r_alive = alive(rs) and (rs or {}).get('state') != 'paused'
    s_alive = alive(ss)
    pause, stop = pause_requested(run), stop_requested(run)
    since = pause_since(run) if pause else None
    out = {'pause': pause, 'stop': stop, 'paused_since': since, 'run': rs, 'supervisor': ss,
           'run_alive': r_alive, 'supervisor_alive': s_alive, 'resumable': s_alive}
    if rs is None and ss is None:
        out.update(state=None, detail="This run's process predates pause / resume (no state files).")
        return out
    if r_alive:
        if pause:
            st, detail = 'pausing', 'Stopping at the next pause point.'
        elif stop:
            st, detail = 'stopping', 'STOP file: the run ends after the current iteration.'
        else:
            st, detail = 'running', ''
    elif pause:
        st = 'paused'
        detail = ('The supervisor relaunches the run when PAUSE is removed (Resume).' if s_alive else
                  'No supervisor is running: resuming needs the launch command (README).')
    elif s_alive:
        sst = (ss or {}).get('state')
        st = {'paused': 'resuming', 'resuming': 'resuming', 'restarting': 'restarting'}.get(sst, 'starting')
        detail = {'resuming': 'PAUSE removed: the supervisor is relaunching the run.',
                  'restarting': 'The run crashed; the supervisor restarts it after a short wait.'}.get(st, '')
    else:
        st = 'stopped'
        rst = (rs or {}).get('state')
        detail = {'done': 'Finished, or ended by STOP.', 'aborted': 'Aborted (see the log).',
                  'refused': "Refused its checkpoint: a network of another encoding than this build's (see the "
                             'log; tools/fork_run.py widens a run onto a new encoding).',
                  'paused': 'Paused, and the PAUSE file is gone, but no supervisor is running: relaunch it.'
                  }.get(rst, 'No run process (killed or crashed).')
    out.update(state=st, detail=detail)
    return out


# ------------------------------------------------------------------ a companion's end (docs/az.md 49)

# the supervisor's terminal states and how it ended (supervisor_state.json `ended`): 'finished' (the run reached its
# last iteration), 'stopped' (STOP, the supervisor ended or interrupted), 'refused' (the run refused its checkpoint),
# 'gave_up' (the same crash again and again)
SUPERVISOR_DONE = ('done', 'refused')
COMPANION_ENV = ('ZERO_SUPERVISOR_PID', 'ZERO_SUPERVISOR_STARTED', 'ZERO_COMPANION_RUN')


def supervisor_end(run=None, pid=None, started=None):
    """For a companion process the supervisor started (zero.tools.supervise, docs/az.md 49): None while its
    supervisor goes on (the run running, paused, restarting), else how the run ended - the supervisor's `ended`
    ('finished', 'stopped', 'refused', 'gave_up') once it wrote a terminal state, 'stopped' when the supervisor is
    gone without one (killed: its process no longer exists) or another supervisor took the run over. `run`, `pid`,
    `started`: default the environment the supervisor gives its companion (ZERO_COMPANION_RUN, ZERO_SUPERVISOR_PID,
    ZERO_SUPERVISOR_STARTED: the supervisor's pid and process creation time)."""
    run = run or os.environ.get(COMPANION_ENV[2])
    pid = int(pid if pid is not None else os.environ.get(COMPANION_ENV[0]) or 0)
    if started is None and os.environ.get(COMPANION_ENV[1]):
        started = float(os.environ[COMPANION_ENV[1]])
    if not run or not pid:
        raise ValueError('supervisor_end: no run / supervisor pid (not started by a supervisor?)')

    def ended():
        st = read_state(run, 'supervisor') or {}
        if st.get('pid') == pid and st.get('state') in SUPERVISOR_DONE:
            return st.get('ended') or 'stopped'
        if st.get('pid') not in (None, pid):
            return 'stopped'                       # another supervisor's run now
        return None
    e = ended()
    if e:
        return e
    if not alive({'pid': pid, 'started': started}):
        return ended() or 'stopped'                # (a terminal state written just before it exited wins)
    return None


# ------------------------------------------------------------------ live runs on this machine (docs/az.md 51)

LIVE_ENV = 'ZERO_LIVE_RUNS'                 # '0': report none, '1': report one (tests, overrides); unset: look
HAZARD_ENV = 'ZERO_ALLOW_LIVE_RUN_HAZARD'   # '1': run the live-run-hazard tests even beside a live run
FENCE_ENV = 'ZERO_LIVE_RUN_FENCE'          # '1': fence on any platform; '0': never (default: Windows only)


def _supervisor_target(cmd):
    """The run directory a supervisor command line supervises ('' when it names none), or None when the command is
    not a supervisor: `python [opts] tools/supervise.py <run> ...` (any game's copy) or `python [opts] -m
    zero.tools.supervise <run> ...`. Only the interpreter's first program counts (a tool that merely passes a path
    ending in supervise.py is not a supervisor)."""
    k = 1
    while k < len(cmd):
        a = cmd[k]
        if a == '-m':
            if k + 1 < len(cmd) and cmd[k + 1] == 'zero.tools.supervise':
                rest = [x for x in cmd[k + 2:] if not x.startswith('-')]
                return rest[0] if rest else ''
            return None
        if a in ('-c',):
            return None
        if a.startswith('-'):
            k += 1
            continue
        if os.path.basename(a.replace('\\', '/')) == 'supervise.py':
            rest = [x for x in cmd[k + 1:] if not x.startswith('-')]
            return rest[0] if rest else ''
        return None
    return None


def live_supervisors():
    """The supervisors (zero.tools.supervise, any game's tools/supervise.py) running on this machine ->
    [{'pid', 'run', 'cmd'}] (a Windows venv's launcher and its child are both listed). A live supervisor is what a
    process-churning test must not run beside: on 2026-11-14 a test storm recycled a live run's parent pid and a
    test supervisor's orphan sweep killed the live run (docs/az.md 47.2; the sweep's creation-time check fixed that
    case, the fence keeps the whole class away: docs/az.md 51). ZERO_LIVE_RUNS=0 / 1 overrides the look (tests).
    One process scan: a few seconds on Windows."""
    v = os.environ.get(LIVE_ENV, '').strip()
    if v in ('0', '1'):
        return [{'pid': 0, 'run': f'({LIVE_ENV}=1)', 'cmd': []}] if v == '1' else []
    import psutil
    out = []
    me = os.getpid()
    for p in psutil.process_iter(['name']):
        name = (p.info.get('name') or '').lower()
        if not name.startswith('python') or p.pid == me:
            continue
        try:
            cmd = p.cmdline()
        except Exception:                          # noqa: BLE001 - gone, or not ours to read
            continue
        run = _supervisor_target(cmd)
        if run is not None:
            out.append({'pid': p.pid, 'run': run, 'cmd': cmd})
    return out


FENCE_PREFIX = 'live-run fence:'


def live_run_fence_reason(sups=None):
    """Why the live-run-hazard tests are skipped here (None: they run): a live supervisor on this machine
    (`sups`: live_supervisors(), looked up when None) on Windows (ZERO_LIVE_RUN_FENCE=1: any platform, 0: never),
    unless ZERO_ALLOW_LIVE_RUN_HAZARD=1. The reason starts
    FENCE_PREFIX (tools count the fenced tests in pytest's -rs lines: mtg-zero's tools/gate.py)."""
    if os.environ.get(HAZARD_ENV, '').strip() == '1':
        return None
    # Windows only by default: the hazard is Windows's (a dead parent's pid kept and recycled: 47.2); macOS and Linux
    # re-parent orphans, so the Mac - which hosts other projects' long runs - runs these tests (docs/az.md 51)
    fence = os.environ.get(FENCE_ENV, '').strip()
    if fence == '0' or (fence != '1' and os.name != 'nt'):
        return None
    sups = live_supervisors() if sups is None else sups
    if not sups:
        return None
    runs = sorted({s['run'] for s in sups})
    return (f'{FENCE_PREFIX} {len(runs)} live run(s) on this machine ({", ".join(runs)[:160]}); the test starts '
            f'supervisors or churns processes (docs/az.md 51): run it where no run is live, or set {HAZARD_ENV}=1')


def fence_items(config, items):
    """The pytest hook body (a conftest's pytest_collection_modifyitems): the items marked `live_run_hazard` are
    skipped with live_run_fence_reason() while a supervisor is live on this machine. Looks only when such an item
    was collected (the scan costs seconds)."""
    marked = [it for it in items if it.get_closest_marker('live_run_hazard')]
    if not marked:
        return 0
    why = live_run_fence_reason()
    if not why:
        return 0
    import pytest
    for it in marked:
        it.add_marker(pytest.mark.skip(reason=why))
    return len(marked)


# ------------------------------------------------------------------ the run process's watch

class PauseWatch:
    """Watches <run>/PAUSE from a daemon thread every `poll` seconds. Once it appears: `seen`
    (time noticed), `requested` (the file's time), `on_pause()` is called on the watch thread
    (run.py: the state file, a log line), then `is_set()` / `check()` (raises Paused) tell
    the run's loops, and `watchdog(self)` - if set - runs on the watch thread (run.py: kills
    the workers / exits if the main thread does not pause in time; it waits on `done`).
    `close()` ends the watch and any watchdog."""

    def __init__(self, run, poll=0.5, on_pause=None, watchdog=None):
        self.run, self.poll = run, poll
        self.path = pause_path(run)
        self.on_pause = on_pause
        self.watchdog = watchdog
        self.seen = self.requested = None
        self.evt = threading.Event()
        self.done = threading.Event()
        self.th = threading.Thread(target=self._loop, daemon=True, name='zero-pause-watch')
        self.th.start()

    def _loop(self):
        while not self.done.wait(self.poll):
            if os.path.exists(self.path):
                self.trigger()
                if self.watchdog is not None:
                    self.watchdog(self)
                return

    def trigger(self):
        if self.evt.is_set():
            return
        self.seen = time.time()
        self.requested = pause_since(self.run) or self.seen
        if self.on_pause is not None:
            try:
                self.on_pause()
            except Exception:                      # noqa: BLE001 - never keep the pause from the loops
                pass
        self.evt.set()

    def is_set(self):
        return self.evt.is_set()

    def check(self, where=''):
        if self.evt.is_set():
            raise Paused(where)

    def close(self):
        self.done.set()
