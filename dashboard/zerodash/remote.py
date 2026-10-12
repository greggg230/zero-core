"""Runs on another host (St. Pete's az6 runs on the Mac), reached over SSH.

A remote run is configured with its directory on the host and a local mirror directory that a
separate sync keeps fresh (metrics.jsonl, log.txt, progress, elo/ ...; preserving mtimes, e.g.
rsync -a). The dashboard reads the mirror for everything except what must be current:

* a background probe (every ~60 s, never blocking a page) runs one SSH command that prints the
  run's PAUSE / STOP / state files / progress.json, the tail of supervisor.log, `ps` for the
  pids those name and the host's clock. The run's control state and liveness come from it;
  while the host does not answer they come from the mirror and liveness is unknown.
* Pause writes <path>/PAUSE on the host (never replacing one), Resume removes it, and Resume
  all removes it only if it still holds exactly the pause-all content it was read with.

Every SSH call is `ssh -o BatchMode=yes -o ConnectTimeout=5 <host> <sh command>`; paths are
single-quoted (a leading ~/ becomes "$HOME"/...).
"""
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time

from . import procs, runfiles

SSH_OPTS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5']
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)     # no console flash from the hub's hidden launch


class SshError(RuntimeError):
    pass


def _last_line(err, out, code):
    lines = (err or out or '').strip().splitlines()
    return f'ssh: {lines[-1][:200]}' if lines else f'ssh exited with status {code}'


class SshRunner:
    """Runs one command on a host: run(host, command) -> (returncode, stdout, stderr)."""

    def __init__(self, exe=None, timeout=25):
        self.exe = exe
        self.timeout = timeout

    def argv(self, host, command):
        exe = self.exe or shutil.which('ssh')
        if not exe:
            raise SshError('ssh is not installed')
        return [exe] + SSH_OPTS + [host, command]

    def run(self, host, command, timeout=None):
        try:
            p = subprocess.run(self.argv(host, command), capture_output=True, text=True, encoding='utf-8',
                               errors='replace', timeout=timeout or self.timeout, stdin=subprocess.DEVNULL,
                               creationflags=NO_WINDOW)
        except subprocess.TimeoutExpired:
            raise SshError(f'ssh {host}: timed out') from None
        except OSError as e:
            raise SshError(f'ssh {host}: {e}') from None
        return p.returncode, p.stdout, p.stderr


def qpath(path):
    """A remote path as a sh word: '~/x y' -> "$HOME"/'x y'."""
    if path == '~':
        return '"$HOME"'
    if path.startswith('~/'):
        return '"$HOME"/' + shlex.quote(path[2:])
    return shlex.quote(path)


def _cd(path):
    return f"cd -- {qpath(path)} 2>/dev/null || {{ echo '@@nodir'; exit 0; }}\n"


PROBE_FILES = ('PAUSE', 'STOP', 'run_state.json', 'supervisor_state.json', 'progress.json')


def probe_script(path):
    files = ' '.join(PROBE_FILES)
    return (_cd(path) +
            'echo "@@now $(date +%s)"\n'
            f'for f in {files}; do if [ -f "$f" ]; then echo "@@file $f"; cat "$f"; echo; echo "@@end"; fi; done\n'
            'echo "@@suplog"; tail -n 40 supervisor.log 2>/dev/null; echo "@@end"\n'
            'pids=$( { cat run_state.json supervisor_state.json 2>/dev/null; tail -n 40 supervisor.log 2>/dev/null; } '
            '| grep -oE \'"(run_)?pid": *[0-9]+|supervisor [0-9]+ starting|pid [0-9]+\' | grep -oE \'[0-9]+\' '
            '| sort -u | paste -s -d , - )\n'
            'echo "@@ps"; if [ -n "$pids" ]; then ps -o pid=,etime=,command= -p "$pids" 2>/dev/null; fi; echo "@@end"\n')


def parse_etime(s):
    """ps etime '[[dd-]hh:]mm:ss' -> seconds (None if malformed)."""
    m = re.match(r'^(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)$', s.strip())
    if not m:
        return None
    d, h, mi, se = (int(x) if x else 0 for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + se


def parse_probe(out):
    """The probe's output -> {'nodir', 'now', 'files': {name: text}, 'suplog': [lines],
    'procs': {pid: start epoch (host clock)}}."""
    res = {'nodir': False, 'now': None, 'files': {}, 'suplog': [], 'procs': {}}
    cur, buf = None, []
    for line in out.splitlines():
        if cur is None:
            if line == '@@nodir':
                res['nodir'] = True
            elif line.startswith('@@now '):
                try:
                    res['now'] = float(line[6:])
                except ValueError:
                    pass
            elif line.startswith('@@file '):
                cur, buf = ('file', line[7:].strip()), []
            elif line == '@@suplog':
                cur, buf = ('suplog', None), []
            elif line == '@@ps':
                cur, buf = ('ps', None), []
            continue
        if line == '@@end':
            kind, name = cur
            if kind == 'file':
                text = '\n'.join(buf)
                res['files'][name] = text[:-1] if text.endswith('\n') else text
            elif kind == 'suplog':
                res['suplog'] = [ln for ln in buf if ln.strip()]
            else:
                for ln in buf:
                    parts = ln.split(None, 2)
                    if len(parts) >= 2 and parts[0].isdigit():
                        el = parse_etime(parts[1])
                        if el is not None and res['now'] is not None:
                            res['procs'][int(parts[0])] = res['now'] - el
                        else:
                            res['procs'][int(parts[0])] = None
            cur, buf = None, []
            continue
        buf.append(line)
    return res


class RemoteLiveness:
    """Liveness from a probe: pid in its `ps` output, started within a few seconds of the
    recorded creation time (ps etime has 1 s resolution)."""

    def __init__(self, probe):
        self.probe = probe

    def alive(self, pid, started=None, near=None):
        if not self.probe:
            return None
        if not pid or int(pid) not in self.probe['procs']:
            return False
        return procs.check(self.probe['procs'][int(pid)], started, near, tol=3.0)


class RemoteRun:
    """One remote run: its configuration, the newest probe result and the SSH operations."""

    def __init__(self, spec, ssh, clock=time.time):
        self.spec = spec
        self.ssh = ssh
        self.clock = clock
        self.probe = None             # parsed output of the last successful probe
        self.probe_time = None
        self.error = None
        self.error_time = None
        self.lock = threading.Lock()

    @property
    def host(self):
        return self.spec['ssh']

    def refresh(self):
        """Run the probe now (the background thread calls this)."""
        try:
            code, out, err = self.ssh.run(self.host, probe_script(self.spec['path']))
            if code != 0:
                raise SshError(_last_line(err, out, code))
            p = parse_probe(out)
            if p['nodir']:
                raise SshError(f"{self.spec['path']}: no such directory on {self.host}")
            with self.lock:
                self.probe, self.probe_time, self.error = p, self.clock(), None
        except SshError as e:
            with self.lock:
                self.error, self.error_time = str(e), self.clock()
        return self.probe

    def fresh_probe(self, max_age=180):
        with self.lock:
            if self.probe and self.probe_time and self.clock() - self.probe_time <= max_age and not self.error:
                return self.probe
        return None

    def view(self):
        with self.lock:
            return {'host': self.host, 'path': self.spec['path'], 'label': self.spec.get('label'),
                    'probe_age_s': round(self.clock() - self.probe_time) if self.probe_time else None,
                    'error': self.error}

    # -------------------------------------------------------------- control over SSH

    def write_pause(self, content):
        """Create PAUSE on the host unless there is one -> True if created now."""
        script = (_cd(self.spec['path']) +
                  "if [ -e PAUSE ]; then echo '@@exists'; exit 0; fi\n"
                  f"printf '%s' {shlex.quote(content)} > PAUSE.zerodash.tmp && mv -f PAUSE.zerodash.tmp PAUSE "
                  "&& echo '@@created'\n")
        out = self._call(script)
        return '@@created' in out

    def remove_pause(self, expect=None):
        """Remove PAUSE on the host -> True if removed. With `expect`, only if it still holds
        exactly that text (Resume all: a pause made since by someone else stays)."""
        if expect is None:
            script = _cd(self.spec['path']) + "if [ -e PAUSE ]; then rm -f PAUSE && echo '@@removed'; else echo '@@absent'; fi\n"
        else:
            script = (_cd(self.spec['path']) +
                      "if [ ! -e PAUSE ]; then echo '@@absent'; exit 0; fi\n"
                      f"if [ \"$(cat PAUSE)\" = {shlex.quote(expect)} ]; then rm -f PAUSE && echo '@@removed'; "
                      "else echo '@@kept'; fi\n")
        out = self._call(script)
        return '@@removed' in out

    def read_pause(self):
        """The host's PAUSE now -> its text, or None without one."""
        out = self._call(_cd(self.spec['path']) +
                         "if [ -f PAUSE ]; then echo '@@file PAUSE'; cat PAUSE; echo; echo '@@end'; fi\n")
        return parse_probe(out)['files'].get('PAUSE')

    def _call(self, script):
        code, out, err = self.ssh.run(self.host, script)
        if code != 0:
            raise SshError(_last_line(err, out, code))
        if '@@nodir' in out:
            raise SshError(f"{self.spec['path']}: no such directory on {self.host}")
        return out


class RemoteSource:
    """summary.py's view of a remote run: files from the mirror, control / state / processes
    from the newest probe (the mirror's copies while the host does not answer)."""
    lag = 900                     # the separate sync may refresh the mirror only every ~10 min

    def __init__(self, rr, mirror):
        self.rr = rr
        self.dir = mirror
        self.p = rr.fresh_probe()

    @property
    def remote(self):
        return self.rr.view()

    def _file(self, name):
        if self.p is not None:
            return self.p['files'].get(name)
        return runfiles.read_text(os.path.join(self.dir, name), 65536) if self.dir else None

    def _json(self, name):
        t = self._file(name)
        try:
            j = json.loads(t) if t else None
        except ValueError:
            return None
        return j if isinstance(j, dict) else None

    def pause(self):
        t = self._file('PAUSE')
        if t is None:
            return None
        mt = runfiles.mtime(os.path.join(self.dir, 'PAUSE')) if self.p is None and self.dir else None
        return runfiles.pause_from_text(t, mt)

    def stop(self):
        return self._file('STOP') is not None

    def state_file(self, who):
        return self._json(f'{who}_state.json')

    def progress(self):
        return self._json('progress.json')

    def supervisor_lines(self):
        if self.p is not None:
            return self.p['suplog']
        return runfiles.tail_lines(os.path.join(self.dir, 'supervisor.log'), 60) if self.dir else []

    def liveness(self, default):
        return RemoteLiveness(self.p)


class Prober:
    """Refreshes every remote run's probe in the background, one after another."""

    def __init__(self, runs, interval=60):
        self.runs = runs
        self.interval = interval
        self.wake = threading.Event()
        self.th = None

    def start(self):
        if self.runs and self.th is None:
            self.th = threading.Thread(target=self._loop, daemon=True, name='zerodash-remote-probe')
            self.th.start()

    def poke(self):
        self.wake.set()

    def _loop(self):
        while True:
            for rr in list(self.runs):
                try:
                    rr.refresh()
                except Exception as e:             # noqa: BLE001 - keep probing the others
                    rr.error, rr.error_time = f'{type(e).__name__}: {e}', rr.clock()
            self.wake.wait(self.interval)
            self.wake.clear()
