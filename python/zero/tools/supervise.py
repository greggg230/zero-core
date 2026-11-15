"""Keep an AlphaZero run going unattended.

  python tools/supervise.py runs/toy1 [extra <pkg>.az.run args...]     (in a game repo)

* runs `python -m <pkg>.az.run --run <run> ...` (the game package's `run_module`, from the
  game repo's root; the core alone: `python -m zero.run`) and restarts it (resuming from the
  newest checkpoint) whenever it exits with an error, after killing any orphaned workers;
* backs off if it keeps crashing (5 crashes within an hour -> wait 15 minutes);
* gives up on a deterministic failure: the same crash (the run's own error text in run_state.json, its
  decimals masked - a NonFinite's losses differ in the last digits on a GPU) <P>_SUPERVISE_SAME_CRASHES
  times in a row (default 3; 0 = never) ends the supervisor - state 'refused', with the error - instead
  of replaying the iteration forever (docs/az.md 47.2);
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
* a companion process (docs/az.md 49; the game package's `companion` hook, e.g. MTG Zero's delverai publisher,
  or <P>_SUPERVISE_COMPANION: a JSON {name, cmd, log, env}, tests): started from the game repo's root when the run
  has written its config.json (asked again at every start of the run: the same command keeps the running one, a
  changed command restarts it, None stops it), hidden and below normal priority, its console in <run>/<log>; one per
  run across the run's restarts, crashes and pauses (it stays up while the run is paused); one that exits while the
  run goes on is started again, at most 3 times an hour. The supervisor's terminal state says how the run ended
  (`ended` in supervisor_state.json: finished / stopped / refused / gave_up) - the companion reads it
  (runctl.supervisor_end), does its last work and exits; the supervisor waits for it (<P>_SUPERVISE_COMPANION_WAIT
  seconds after a finished run, 3600; <P>_SUPERVISE_COMPANION_STOP_WAIT otherwise, 120) and then kills it. A
  companion that cannot start (its hook raises, its command fails) is logged loudly (COMPANION NOT STARTED) and the
  run goes on: deliberately no reason to stop training. A companion left by an earlier supervisor of the run (one
  killed abruptly: its companion notices that and exits by itself) is stopped when a new supervisor starts.
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
import re
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


def kill_orphans(pid, since):
    """Pool workers of a dead run keep its pid as their parent pid - and so does any process whose own parent, long
    dead, had that number before (Windows reuses pids quickly and keeps a dead parent's pid as the parent pid): only
    processes created after `since` (the time the supervisor launched the run, just before it) can be the run's, so
    only those are killed. Without that check a run's exit killed an unrelated process tree whose dead parent's pid the
    run had been given - on 2026-11-14 a live run's supervisor and run, when a test churned ~1,000 short supervised
    processes on the same machine (docs/az.md 47.2)."""
    try:                                   # one snapshot (~0.1 s); process_iter's first pass takes ~10 s on Windows
        from psutil import _psplatform
        kids = [c for c, pp in _psplatform.ppid_map().items() if pp == pid]
    except Exception:                      # noqa: BLE001 - platforms without ppid_map
        kids = [p.pid for p in psutil.process_iter(['pid', 'ppid']) if p.info['ppid'] == pid]
    n = 0
    for c in kids:
        try:
            p = psutil.Process(c)
            if p.create_time() < since:     # (an older process: its parent was another process with this pid)
                continue
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


def wait_run(run, p, stall_s, tick=None, tick_s=5.0):
    """Wait for the run process `p` to exit -> its exit code. With stall_s > 0 a run without
    progress for stall_s seconds (no PAUSE requested) is killed with its children first. `tick()`: called every
    `tick_s` seconds meanwhile (the companion's care)."""
    if stall_s <= 0 and tick is None:
        return p.wait()
    t0 = time.time()
    step = min(max(stall_s / 10, 0.05), 30) if stall_s > 0 else 30
    if tick is not None:
        step = min(step, tick_s)
    while True:
        try:
            return p.wait(timeout=step)
        except subprocess.TimeoutExpired:
            pass
        if tick is not None:
            tick()
        if stall_s <= 0:
            continue
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


def launched_state(run, pid, since):
    """run_state.json when the run process the supervisor launched at `since` (pid `pid`) wrote it, else None. The
    process that writes it can be a child of `pid`: a Windows venv's python.exe is a launcher that starts the base
    interpreter as its child (zero-farm's venv runs the pool's run so), so the state's pid is accepted when it is
    `pid` or its process started after the launch (only the supervisor launches the run; docs/az.md 47.2)."""
    st = runctl.read_state(run, 'run') or {}
    started = st.get('started')
    if st.get('pid') == pid or (isinstance(started, (int, float)) and started >= since):
        return st
    return None


def crash_key(run, pid, since=float('inf')):
    """The crash of run process `pid` (launched at `since`) as a comparable key: its own error text (run_state.json,
    written by the run when it aborts: `launched_state`) with the decimals masked; None when that process wrote no
    error (killed, a segfault, the stall watchdog: never counted as the same crash)."""
    st = launched_state(run, pid, since) or {}
    if st.get('state') != 'aborted' or not st.get('error'):
        return None
    return re.sub(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?', '#', str(st['error']))


def wait_resume(run, poll_s, tick=None):
    """The run is paused: wait until its PAUSE file is removed -> True (relaunch now), or
    False if a STOP file appears meanwhile (the supervisor ends; the run stays paused). `tick()`: called at every
    poll (the companion's care: it stays up while the run is paused)."""
    pause = runctl.pause_path(run)
    t0 = time.time()
    since = runctl.pause_since(run) or t0
    runctl.write_state(run, 'supervisor', state='paused', run_pid=None, paused_since=since)
    while os.path.exists(pause):
        if runctl.stop_requested(run):
            log(run, 'STOP file present while paused: supervisor done (the run stays paused; relaunching the '
                     'supervisor waits for PAUSE to be removed)')
            runctl.write_state(run, 'supervisor', state='done', ended='stopped')
            return False
        if tick is not None:
            tick()
        time.sleep(poll_s)
    log(run, f'PAUSE removed after {ago(time.time() - since)} paused: resuming the run')
    runctl.write_state(run, 'supervisor', state='resuming')
    return True


def kill_tree(pid):
    """Kill process `pid` and its descendants -> True if it was there."""
    try:
        p = psutil.Process(pid)
        for k in p.children(recursive=True):
            try:
                k.kill()
            except psutil.NoSuchProcess:
                pass
        p.kill()
        return True
    except psutil.NoSuchProcess:
        return False


class Companion:
    """The run's companion process (docs/az.md 49): at most one, kept across the run's restarts.

    `sync(cfg)` at every start of the run (its config.json written): starts it, keeps it (the same command), restarts
    it (another command) or stops it (none); `check()` while the run goes on or is paused: one that exited is started
    again (at most `restarts` times an hour); `finish(ended, wait_s)` when the supervisor ends: the companion reads the
    supervisor's terminal state and exits by itself - waited for, then killed. Its pid and creation time are in the
    supervisor's state (`companion`), so a new supervisor stops one that an abruptly killed supervisor left."""

    def __init__(self, run, pkg, restarts=3):
        self.run = run
        self.pkg = pkg
        self.p = None
        self.spec = None
        self.out = None
        self.restarts = restarts
        self.deaths = []
        self.gave_up = False
        self.override = pkg.env('SUPERVISE_COMPANION')

    def resolve(self, cfg):
        """-> the companion spec for the run's settings `cfg`, or None; raises when the hook refuses."""
        if self.override is not None:
            spec = json.loads(self.override)
            if not spec:
                return None
            name = str(spec.get('name') or 'companion')
            return {'name': name, 'cmd': [str(x) for x in spec['cmd']], 'log': str(spec.get('log') or f'{name}_console.txt'),
                    'env': {str(k): str(v) for k, v in (spec.get('env') or {}).items()}}
        return self.pkg.companion_of(os.path.abspath(self.run), cfg)

    def wanted(self):
        """Whether this game package can have a companion at all (a hook, or the environment's override): without
        one the supervisor waits for the run as before."""
        return self.pkg.companion is not None or self.override is not None

    def alive(self):
        return self.p is not None and self.p.poll() is None

    def _state(self):
        runctl.write_state(self.run, 'supervisor', companion=(
            {'name': self.spec['name'], 'pid': self.p.pid, 'started': runctl._proc_started(self.p.pid),
             'cmd': self.spec['cmd']} if self.alive() else None))

    def _start(self):
        spec = self.spec
        env = dict(os.environ, PYTHONIOENCODING='utf-8', **spec['env'])
        env.update({runctl.COMPANION_ENV[0]: str(os.getpid()),
                    runctl.COMPANION_ENV[1]: repr(runctl._proc_started() or 0.0),
                    runctl.COMPANION_ENV[2]: os.path.abspath(self.run)})
        kw = {}
        if os.name == 'nt':
            # hidden (no console window) and below normal priority: the run comes first
            kw['creationflags'] = 0x08000000 | 0x00004000      # CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS
        else:
            kw['preexec_fn'] = lambda: os.nice(10)
        self.out = open(os.path.join(self.run, spec['log']), 'a', encoding='utf-8')
        try:
            self.p = subprocess.Popen(spec['cmd'], cwd=self.pkg.root, stdout=self.out, stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL, env=env, **kw)
        except OSError as e:
            self.out.close()
            self.p = None
            self.gave_up = True
            self.loud(f'its command failed: {e!r}')
            return False
        log(self.run, f"companion {spec['name']} started: pid {self.p.pid}, console {spec['log']}: "
                      f"{' '.join(spec['cmd'])}")
        self._state()
        return True

    def loud(self, why):
        """The companion could not start: said loudly (the log, the state, stderr); the run goes on."""
        name = (self.spec or {}).get('name', 'companion')
        log(self.run, f'COMPANION NOT STARTED ({name}): {why} - the run goes on without it (a companion is never '
                      f'worth stopping the run for; docs/az.md 49)')
        runctl.write_state(self.run, 'supervisor', companion=None, companion_error=f'{name}: {why}'[:1000])
        print(f'supervise: COMPANION NOT STARTED ({name}): {why}', file=sys.stderr, flush=True)

    def sync(self, cfg):
        """The run (re)started with the settings `cfg`: the companion they ask for, running."""
        try:
            spec = self.resolve(cfg)
        except Exception as e:                 # noqa: BLE001 - a refusing hook must not take the supervisor down
            self.stop('its hook refused')
            self.spec = None
            self.loud(f'{type(e).__name__}: {e}')
            return
        except SystemExit as e:                # (a hook reusing a command line's refusal)
            self.stop('its hook refused')
            self.spec = None
            self.loud(f'SystemExit: {e}')
            return
        if spec is None:
            if self.alive():
                self.stop('the run asks for none now')
            self.spec = None
            return
        if self.alive() and spec['cmd'] == self.spec['cmd']:
            return                             # one per run: the running one goes on
        if self.alive():
            self.stop('its command changed')
        self.spec, self.deaths, self.gave_up = spec, [], False
        runctl.write_state(self.run, 'supervisor', companion_error=None)
        self._start()

    def check(self):
        """While the run goes on (or is paused): a companion that exited is started again, at most `restarts` times
        an hour."""
        if self.spec is None or self.p is None or self.gave_up or self.p.poll() is None:
            return
        code = self.p.returncode
        self.out.close()
        now = time.time()
        self.deaths = [t for t in self.deaths if now - t < 3600] + [now]
        if len(self.deaths) > self.restarts:
            self.gave_up = True
            self.loud(f'it exited with code {code} {len(self.deaths)} times within an hour (see {self.spec["log"]}): '
                      f'not started again until the run restarts with another command for it')
            return
        log(self.run, f"companion {self.spec['name']} exited with code {code} while the run goes on: starting it again")
        self._start()

    def stop(self, why):
        if not self.alive():
            return
        log(self.run, f"companion {self.spec['name']} pid {self.p.pid} stopped: {why}")
        kill_tree(self.p.pid)
        self.p.wait()
        self.out.close()
        self._state()

    def finish(self, ended, wait_s):
        """The supervisor ends (its terminal state, with `ended`, already written): wait up to `wait_s` seconds for
        the companion to do its last work and exit, then kill it."""
        if not self.alive():
            return
        name, pid = self.spec['name'], self.p.pid
        log(self.run, f'waiting up to {ago(wait_s)} for companion {name} (pid {pid}) to end (the run: {ended})')
        try:
            code = self.p.wait(timeout=wait_s)
            log(self.run, f'companion {name} ended with code {code}')
        except subprocess.TimeoutExpired:
            log(self.run, f'companion {name} pid {pid} still running after {ago(wait_s)}: killed')
            kill_tree(pid)
            self.p.wait()
        self.out.close()
        runctl.write_state(self.run, 'supervisor', companion=None)


def stop_left_companion(run, st):
    """A companion that an earlier supervisor of the run left (`st`: that supervisor's state, read before this one
    writes its own) and that is still alive - that supervisor was killed and the companion has not noticed yet - is
    stopped: a new one starts with the run."""
    c = (st or {}).get('companion') or {}
    if c.get('pid') and runctl.alive(c) and not runctl.alive(st):
        kill_tree(int(c['pid']))
        log(run, f"stopped companion {c.get('name')} pid {c['pid']} left by supervisor {st.get('pid')}")


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
    log(run, f'supervisor {os.getpid()} starting: {run} {" ".join(extra)}; open-file limit '
             f'{runctl.raise_nofile() or "n/a"} (soft, hard)')
    stop_left_companion(run, runctl.read_state(run, 'supervisor'))
    runctl.write_state(run, 'supervisor', state='starting', run_pid=None, args=extra)
    comp = Companion(run, pkg)
    ended = 'stopped'
    try:
        ended = supervise(run, extra, pkg, comp)
    except BaseException as e:
        # the supervisor itself failed or was interrupted (Ctrl+C): a terminal state for the companion to read
        log(run, f'supervisor ended by {type(e).__name__}: {e}')
        runctl.write_state(run, 'supervisor', state='done', run_pid=None, ended='stopped')
        raise
    finally:
        wait = float(pkg.env('SUPERVISE_COMPANION_WAIT', 3600) if ended == 'finished'
                     else pkg.env('SUPERVISE_COMPANION_STOP_WAIT', 120))
        comp.finish(ended, wait)


def supervise(run, extra, pkg, comp):
    """The supervisor's loop -> how the run ended ('finished', 'stopped', 'refused', 'gave_up'), also written as the
    terminal state's `ended`."""
    crashes = []
    stop = os.path.join(run, 'STOP')
    if os.path.exists(stop):
        os.replace(stop, stop + '.old')
        log(run, 'moved a STOP file left from an earlier stop to STOP.old')
    wait_s = float(pkg.env('SUPERVISE_WAIT', 60))
    backoff_s = float(pkg.env('SUPERVISE_BACKOFF', 900))
    poll_s = float(pkg.env('SUPERVISE_POLL', 1))
    stall_s = float(pkg.env('SUPERVISE_STALL', 0))
    tick_s = float(pkg.env('SUPERVISE_COMPANION_POLL', 5))
    same_n = int(float(pkg.env('SUPERVISE_SAME_CRASHES', 3)))
    last_key, same = None, 0
    cmd = json.loads(pkg.env('SUPERVISE_CMD')) if pkg.env('SUPERVISE_CMD') else \
        [sys.executable, '-m', pkg.run_module]
    name = pkg.run_module
    # the run imports the same core as this supervisor (a game's run would find it through its
    # package anyway; `python -m zero.run` needs it)
    core = os.path.join(package.CORE_ROOT, 'python')
    env = dict(os.environ, PYTHONIOENCODING='utf-8',
               PYTHONPATH=os.pathsep.join([core] + [p for p in [os.environ.get('PYTHONPATH')] if p]))

    def done(state='done', **kw):
        runctl.write_state(run, 'supervisor', state=state, run_pid=None, **kw)
        return kw['ended']

    while True:
        if runctl.pause_requested(run):
            log(run, 'PAUSE file present: the run stays paused until it is removed')
            if not wait_resume(run, poll_s, comp.check):
                return 'stopped'
        out = open(os.path.join(run, 'stdout.txt'), 'a', encoding='utf-8')
        since = time.time()
        p = subprocess.Popen(cmd + ['--run', run] + extra, cwd=pkg.root, stdout=out, stderr=subprocess.STDOUT, env=env)
        log(run, f'started {name} pid {p.pid}')
        runctl.write_state(run, 'supervisor', state='running', run_pid=p.pid)
        synced = [False]

        def tick():
            # the companion follows the run's settings of this start: asked once the run has written its config.json
            # (its state file names this process); then kept alive
            if not synced[0]:
                rs = runctl.read_state(run, 'run') or {}
                if rs.get('pid') == p.pid:
                    synced[0] = True
                    try:
                        with open(os.path.join(run, 'config.json'), encoding='utf-8') as f:
                            cfg = json.load(f)
                    except (OSError, ValueError) as e:
                        comp.loud(f'the run\'s config.json unreadable: {e!r}')
                        return
                    comp.sync(cfg)
                    return
            comp.check()
        code = wait_run(run, p, stall_s, tick if comp.wanted() else None, tick_s)
        out.close()
        killed = kill_orphans(p.pid, since)
        # (a launcher's child - a Windows venv's interpreter - is the run whose pool workers outlive it: launched_state)
        st = launched_state(run, p.pid, since)
        if st and isinstance(st.get('pid'), int) and st['pid'] != p.pid:
            killed += kill_orphans(st['pid'], since)
        log(run, f'{name} exited with code {code}; killed {killed} orphaned workers')
        if comp.wanted() and not synced[0]:
            tick()                             # (a run that ended within the first poll: its companion all the same)
        if code == 0:
            if os.path.exists(stop):
                log(run, 'clean exit (STOP file); supervisor done')
                return done(ended='stopped')
            log(run, 'clean exit (finished); supervisor done')
            return done(ended='finished')
        if os.path.exists(stop):
            log(run, 'STOP file present; not restarting')
            return done(ended='stopped')
        if code == runctl.EXIT_REFUSED:
            log(run, f'the run REFUSED its checkpoint (exit code {code}: a network of another encoding than this '
                     f"build's, see log.txt; docs/az.md 10.4): not restarting, supervisor done")
            return done('refused', ended='refused')
        if code == runctl.EXIT_PAUSED or runctl.pause_requested(run):
            log(run, f'run paused (exit code {code}): waiting for the PAUSE file to be removed '
                     f'(polling every {poll_s:g}s)')
            if not wait_resume(run, poll_s, comp.check):
                return 'stopped'
            continue
        key = crash_key(run, p.pid, since)
        same = same + 1 if key is not None and key == last_key else (1 if key is not None else 0)
        last_key = key
        if same_n > 0 and same >= same_n:
            err = (runctl.read_state(run, 'run') or {}).get('error', '')
            log(run, f'the same crash {same} times in a row ({err[:400]}): a deterministic failure, not restarting - '
                     f'supervisor done (fix the cause, then start the supervisor again; docs/az.md 47.2)')
            return done('refused', why=f'the same crash {same} times in a row', error=err[:1000], ended='gave_up')
        now = time.time()
        crashes = [t for t in crashes if now - t < 3600] + [now]
        wait = backoff_s if len(crashes) >= 5 else wait_s
        log(run, f'crash #{len(crashes)} in the last hour; restarting in {wait}s')
        runctl.write_state(run, 'supervisor', state='restarting', run_pid=None, restart_at=now + wait)
        t_end = time.time() + wait
        while time.time() < t_end:             # (the companion stays cared for during the wait)
            time.sleep(max(0.0, min(tick_s, t_end - time.time())))
            comp.check()


if __name__ == '__main__':
    main()
