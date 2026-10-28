"""Remote self-play (docs/az.md 5.3; co-training 11.6): another machine self-plays for a run with
the run's checkpoint i-1 while the run self-plays iteration i here, and its sample shards come
back into `data/iter_i/`; in a co-training run it plays a share of the resolver's queued episodes
instead, their result rows going back into the coordinator by tag.

Run settings: `selfplay_remote` (a name in the game package's `remotes`; 'local' = this machine,
the tests; '' = off), `selfplay_remote_procs` (worker processes; 0 = the remote's `procs`),
`selfplay_remote_wait` (seconds the end of a phase waits for the remote's last shards),
`selfplay_remote_timeout` (co-training: seconds of a worker's running time after which an
episode it has not finished is requeued here). A co-training run: `resolver_cfg['selfplay_remote']`
(the resolver's episodes); the primary's own self-play (the drafting) stays here.

The remote spec's optional `selfplay` dict: {workdir (default the remote's `workdir`; the
environment's ZERO_SELFPLAY_WORKDIR overrides it), procs (default 1), nice (the workers'
niceness, default 0), locks ([lock directories]: a worker backs off while one is held by a live
process; the remote's own `<workdir>/.eval.lock` always: remote evaluation comes first), backoff
('stop', the default: no leaf batches while a lock is held; 'background': macOS only, the worker
plays on under the Darwin background policy - the lowest CPU band, the efficiency cores - and
leaves it once the lock is gone; elsewhere 'stop'),
overrides (run settings for the workers: concurrency, threads, groups, inference; the shards
default to at most 8192 rows), env}.

Local side (`RemotePlay`: threads of the run process; the main thread never waits for the remote
except in `end`, bounded by `selfplay_remote_wait`):

* a manager thread keeps `procs` persistent **sessions**: `ssh host 'cd repo && python -m
  <pkg>.az.remote selfplay MIRROR K'` (or a local subprocess), one worker each; its stdin is the
  control channel (JSON lines: phase, end, enqueue, fetched, ping, quit), its stdout the worker's
  events (lines `ZERO_SP {json}`: hello, status, shard, row, end, error). A session that fails or
  stays silent for `STALL_S` is killed and started again after `RETRY_S`. The mirror
  `<workdir>/<mirror>` gets config.json (the run's settings + the remote's selfplay overrides) and
  selfplay.json (the worker's mode, locks, niceness, the run's id) at each session start, and
  checkpoint i-1 at each phase start;
* `begin(it)`: the sessions get `phase it` (their drivers' new search epoch with checkpoint
  i-1); `end(it)`: they get `end it`, finish their phase (games in flight carry over, as the run's
  own rolling driver's do) and flush their shards; `end` waits for the shards announced in phase
  i at most `selfplay_remote_wait` s: what is not back by then is dropped (a shard of phase i
  never enters another iteration);
* fetch threads (one per worker) copy each announced shard (a temporary name), check it (the non-finite guard,
  zero/finite.py, and the run's game / features / actions) and renames it into `data/iter_i/` -
  only while phase i accepts. A non-finite shard is quarantined to `<run>/debug/` and the run
  fails with NonFinite at the end of the phase, as a local one would (the window never sees it);
* co-training (`co`, the Coordinator): the manager takes queued episodes from the coordinator (at
  most `cap` outstanding per session) and sends them to a worker; its rows go into the coordinator
  by tag; a session that dies or stalls has its outstanding episodes requeued here (at the front);
  an episode not finished within `selfplay_remote_timeout` s of its worker's running time (its
  back-offs not counted) is requeued too, and whichever copy finishes second is dropped (at worst
  played twice).

Remote side (`serve` -> `Worker`): one rolling driver (self-play: the run's seats; co-training:
queue_only, the queued episodes) persistent across the run's iterations. It backs off - stops
asking for leaf batches, so its threads idle - while one of its locks is held, marks itself busy
for lower-priority jobs (`<workdir>/.busy-<mirror>-w<K>`: an owner file "<pid> <time>" as the eval
lock's) while a phase runs, writes compressed shards (`data/iter_i/m<KK>-<sid>_SSS.npz`, zlib level
1, on a background thread) and exits when its stdin closes, on `quit`, or when no control line
came for `watchdog_s` (an orphan never outlives its run by long). A worker started for a slot
(mirror, K) that a live worker of the same run still holds stops that one first; another run's
is refused.
"""
import glob
import json
import os
import queue
import random
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections import deque

from . import package
from .remote import Transport, _alive, remote_spec

TAG = 'ZERO_SP '
BIG = 1 << 62                     # a remote driver's quota: its phases end by end_phase()
RETRY_S = 60.0                    # a failed session is started again after this (env ZERO_RSP_RETRY_S)
STALL_S = 300.0                   # a session silent this long is killed (its worker stalled)
PING_S = 20.0                     # the run pings its workers this often ...
WATCHDOG_S = 180.0                # ... and a worker exits after this long without a control line
STATUS_S = 5.0                    # a worker's status line during a phase
IDLE_STATUS_S = 30.0              # ... and between phases
LOCK_POLL_S = 2.0                 # how often a worker looks at its locks
SHARD_ROWS = 8192                 # a worker's shard size unless the remote's overrides set shard_rows
COMPRESS = 1                      # zlib level of the remote shards (data.savez_level)


def _now():
    return time.time()


def lock_held(locks):
    """The first lock directory of `locks` held by a live process (its `owner` file "<pid> ..."),
    or being created (no owner file yet, the directory younger than 60 s); else None."""
    for d in locks:
        if not d or not os.path.isdir(d):
            continue
        try:
            with open(os.path.join(d, 'owner'), encoding='utf-8') as f:
                pid = int(f.read().split()[0])
        except (OSError, ValueError, IndexError):
            try:
                if _now() - os.path.getmtime(d) < 60:
                    return d
            except OSError:
                pass
            continue
        if pid != os.getpid() and _alive(pid):
            return d
    return None


# ================================================================== the local side

class Session:
    """One remote worker process and its pipes (reader, writer and stderr threads). Events from
    the worker go to the owner's `_event` on the reader thread."""

    def __init__(self, rp, k):
        self.rp, self.k = rp, k
        self.proc = None
        self.wq = queue.Queue()
        self.err = deque(maxlen=60)
        self.eof = threading.Event()
        self.hello = None
        self.phase = None                          # the iteration whose phase was sent
        self.ended = {}                            # iteration -> the worker's end event
        self.status = {}
        self.error = None
        self.last = _now()                         # last line from the worker
        self.started = _now()
        self.pinged = _now()
        self.assigned = {}                         # co-training: tag -> (worker active_s then, spec)
        self.handled = False

    def start(self):
        rp = self.rp
        args = ['selfplay', rp.rdir, str(self.k)]
        env = {k: str(v) for k, v in rp.env.items()}
        kw = dict(stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                  encoding='utf-8', errors='replace', bufsize=1)
        if os.name == 'nt':
            kw['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        if rp.t.local:
            env.setdefault('PYTHONPATH', os.pathsep.join(p for p in sys.path if p))
            self.proc = subprocess.Popen([rp.spec['python'], '-m', rp.module, *args], cwd=rp.spec['repo'],
                                         env=dict(os.environ, **env), **kw)
        else:
            exports = ' '.join(f'{k}={shlex.quote(v)}' for k, v in env.items())
            sh = (f'cd {shlex.quote(rp.spec["repo"])} && {exports} exec {shlex.quote(rp.spec["python"])} -m '
                  f'{rp.module} ' + ' '.join(shlex.quote(a) for a in args))
            self.proc = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o',
                                          'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4', rp.spec['host'], sh],
                                         **kw)
        for fn, nm in ((self._read, 'out'), (self._write, 'in'), (self._drain, 'err')):
            threading.Thread(target=fn, name=f'zero-rsp-w{self.k}-{nm}', daemon=True).start()

    def send(self, obj):
        self.wq.put(json.dumps(obj, separators=(',', ':')))

    def alive(self):
        return self.proc is not None and not self.eof.is_set() and self.proc.poll() is None

    def _read(self):
        try:
            for line in self.proc.stdout:
                self.last = _now()
                if line.startswith(TAG):
                    try:
                        ev = json.loads(line[len(TAG):])
                    except ValueError:
                        self.err.append(line.rstrip())
                        continue
                    self.rp._event(self, ev)
                elif line.strip():
                    self.err.append(line.rstrip())
        except (OSError, ValueError):
            pass
        finally:
            self.eof.set()
            self.rp._wake()

    def _write(self):
        while True:
            line = self.wq.get()
            if line is None:
                break
            try:
                self.proc.stdin.write(line + '\n')
                self.proc.stdin.flush()
            except (OSError, ValueError):
                break
        try:
            self.proc.stdin.close()
        except (OSError, ValueError):
            pass

    def _drain(self):
        try:
            for line in self.proc.stderr:
                if line.strip():
                    self.err.append(line.rstrip())
        except (OSError, ValueError):
            pass

    def tail(self, n=8):
        return ' | '.join(list(self.err)[-n:])

    def kill(self):
        self.wq.put(None)
        try:
            self.proc.kill()
        except (OSError, AttributeError):
            pass

    def close(self, grace=10.0):
        """quit, close its stdin, wait `grace` s for it to exit, then kill."""
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.send({'cmd': 'quit'})
            self.wq.put(None)
            try:
                self.proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
        self.kill()


class RemotePlay:
    """A run's self-play on the remote `name` (module docstring). `mode` 'selfplay' (a single-game
    run's self-play: shards) or 'matches' (a co-training run's resolver episodes from the
    coordinator given to `begin`: rows and shards). `mirror`: the mirror's name in the remote's
    selfplay workdir (default the run directory's name; a co-training run's resolver:
    '<run name>.resolver')."""

    def __init__(self, R, name, mode='selfplay', mirror=None):
        from .run import DEFAULTS
        self.R, self.name, self.mode = R, name, mode
        c = R.cfg
        self.spec = remote_spec(name, 'selfplay_remote')
        sp = dict(self.spec.get('selfplay') or {})
        self.t = Transport(self.spec)
        self.workdir = (os.environ.get('ZERO_SELFPLAY_WORKDIR') or sp.get('workdir') or self.spec['workdir']).rstrip('/\\')
        if not self.t.local and not self.workdir.startswith('/'):
            # e.g. Git Bash turned $ZERO_SELFPLAY_WORKDIR=/Users/... into C:/Program Files/Git/Users/...:
            # ssh and scp would create that relative to the remote's home
            raise SystemExit(f'selfplay_remote {name!r}: the remote self-play workdir {self.workdir!r} is not an '
                             f'absolute path on the remote (a Windows shell converted $ZERO_SELFPLAY_WORKDIR? Git '
                             f'Bash: MSYS2_ENV_CONV_EXCL=ZERO_SELFPLAY_WORKDIR)')
        self.mirror = mirror or os.path.basename(os.path.normpath(R.run))
        self.rdir = self.workdir + '/' + self.mirror
        self.procs = max(1, int(c.get('selfplay_remote_procs') or sp.get('procs') or 1))
        self.wait_s = float(c.get('selfplay_remote_wait', DEFAULTS.get('selfplay_remote_wait', 30.0)))
        self.timeout_s = float(c.get('selfplay_remote_timeout', DEFAULTS.get('selfplay_remote_timeout', 3600.0)))
        self.retry_s = float(os.environ.get('ZERO_RSP_RETRY_S', RETRY_S))
        self.env = dict(self.spec.get('env') or {}, **(sp.get('env') or {}))
        self.overrides = dict(sp.get('overrides') or {})
        self.locks = [self.spec['workdir'].rstrip('/\\') + '/.eval.lock'] + list(sp.get('locks') or [])
        self.nice = int(sp.get('nice', 0))
        self.backoff = sp.get('backoff', 'stop')
        if self.backoff not in ('stop', 'background'):
            raise SystemExit(f"selfplay_remote {name!r}: backoff {self.backoff!r}: 'stop' or 'background'")
        pkg = package.active()
        self.module = f'{pkg.name}.az.remote' if pkg.name != 'zero' else 'zero.remote'
        self.run_id = f'{socket.gethostname()}:{os.path.abspath(R.run)}'
        conc = int(self.overrides.get('conc') or c['conc'])
        self.cap = conc + max(1, conc // 4)        # co-training: episodes outstanding per worker
        self.mu = threading.Condition()
        self.sess = [None] * self.procs
        self.next_try = [0.0] * self.procs
        self.fails = 0
        self.streak = 0                            # failures since a worker last said hello
        self.t_faillog = 0.0
        self.want = None                           # the iteration whose phase should run (None: between)
        self.accepting = None                      # the iteration whose shards are taken in
        self.co = None
        self.cur = None                            # the phase's counters
        self.model_shipped = None
        self.ship_retry = 0.0
        self.mirror_ready = False
        self.fatal = None                          # a NonFinite to raise on the main thread
        self.closed = False
        self.late_total = 0
        self.late_reported = 0
        self.fq = queue.Queue()
        self.mth = threading.Thread(target=self._loop, name='zero-rsp', daemon=True)
        # one fetch thread per worker: the workers' last shards of a phase come back side by side (each
        # copy is an scp with its own ssh handshake, ~1-2 s, then the check)
        self.fths = [threading.Thread(target=self._fetcher, name=f'zero-rsp-fetch{k}', daemon=True)
                     for k in range(max(2, self.procs))]
        self.mth.start()
        for th in self.fths:
            th.start()

    def log(self, msg):
        self.R.log(f'remote self-play {self.name}: {msg}')

    def fail_log(self, msg):
        """A failure to start or keep a worker: logged the first times, then every 10 minutes (a
        remote asleep for a night is not 500 lines)."""
        self.streak += 1
        if self.streak <= 3 or _now() - self.t_faillog > 600:
            self.t_faillog = _now()
            self.log(msg + (f' ({self.streak} failures in a row; logged every 10 minutes now)' if self.streak > 3 else ''))

    def _wake(self):
        with self.mu:
            self.mu.notify_all()

    def capacity(self):
        """Episodes outstanding on the remote at most (co-training: the backpressure's share)."""
        return self.procs * self.cap

    # ---------------------------------------------------------------- the main thread's API
    def begin(self, it, co=None):
        """Iteration `it`'s self-play starts here: the remote's starts as soon as it can (never
        waited for)."""
        with self.mu:
            if self.fatal is not None:
                raise self.fatal
            self.want = it
            self.accepting = it
            if co is not None:
                self.co = co
            self.cur = {'it': it, 't0': _now(), 'announced': 0, 'accepted': 0, 'rows': 0, 'bytes': 0, 'pending': 0,
                        'rejected': 0, 'quarantined': 0, 'failed': 0, 'fetch_s': 0.0, 'assigned': 0, 'matches': 0,
                        'dup': 0, 'requeued': 0, 'requeued_timeout': 0, 'errors': 0}
            self.mu.notify_all()

    def end(self, it):
        """Iteration `it`'s self-play ended here: the remote's phase ends too; waits at most
        `selfplay_remote_wait` s for its last shards. -> the remote's share (metrics)."""
        t0 = _now()
        with self.mu:
            self.want = None
            targets = [s for s in self.sess if s is not None and s.alive() and s.phase == it]
            for s in targets:
                s.send({'cmd': 'end', 'iter': it})
            deadline = t0 + self.wait_s
            while True:
                done = all(it in s.ended or not s.alive() for s in targets) and self.cur['pending'] == 0
                if done or _now() >= deadline:
                    break
                self.mu.wait(min(0.25, max(0.01, deadline - _now())))
            self.accepting = None
            cur = dict(self.cur)
            ends = [s.ended.get(it) for s in targets]
            late = cur['announced'] - cur['accepted'] - cur['rejected'] - cur['quarantined'] - cur['failed']
            late_prev, self.late_reported = self.late_total - self.late_reported, self.late_total
            live = sum(1 for s in self.sess if s is not None and s.alive())
            out = {'host': self.name, 'procs': self.procs, 'workers': live, 'in_phase': len(targets),
                   'ended': sum(1 for e in ends if e)}
        ends = [e for e in ends if e]
        secs = max([e['seconds'] for e in ends], default=0.0)
        leaves = sum(e['leaves'] for e in ends)
        out.update(games=sum(e['games'] for e in ends), samples=cur['rows'], samples_played=sum(e['samples'] for e in ends),
                   shards=cur['accepted'], bytes=cur['bytes'], leaves=leaves,
                   leaves_per_s=round(sum(e['leaves'] / max(e['seconds'], 1e-9) for e in ends), 1),
                   leaves_per_running_s=round(sum(e['leaves'] / max(e['active_s'], 1e-9) for e in ends), 1),
                   seconds=round(secs, 1), running_s=round(max([e['active_s'] for e in ends], default=0.0), 1),
                   backoff_s=round(max([e['backoff_s'] for e in ends], default=0.0), 1),
                   in_flight=sum(e.get('in_flight', 0) for e in ends), late=late, late_prev=late_prev,
                   rejected=cur['rejected'], quarantined=cur['quarantined'], fetch_failed=cur['failed'],
                   fetch_s=round(cur['fetch_s'], 1), wait_s=round(_now() - t0, 2), session_errors=cur['errors'])
        if self.mode == 'matches':
            with self.mu:
                outstanding = sum(len(s.assigned) for s in self.sess if s is not None)
            out.update(matches=cur['matches'], assigned=cur['assigned'], requeued=cur['requeued'],
                       requeued_timeout=cur['requeued_timeout'], duplicates_dropped=cur['dup'], outstanding=outstanding)
        if self.fatal is not None:
            raise self.fatal
        return out

    def progress(self):
        """The remote's state for progress.json."""
        with self.mu:
            live = [s for s in self.sess if s is not None and s.alive()]
            st = [s.status for s in live if s.status]
            back = any(x.get('backoff') for x in st)
            bg = any(x.get('background') for x in st)
            state = ('background' if bg else 'backing off' if back else 'playing') if any(x.get('phase') for x in st) \
                else ('idle' if live else ('connecting' if any(s is not None for s in self.sess) else 'offline'))
            out = {'host': self.name, 'workers': len(live), 'state': state,
                   'games': sum(x.get('games', 0) for x in st), 'in_flight': sum(x.get('in_flight', 0) for x in st),
                   'samples': self.cur['rows'] if self.cur else 0, 'shards': self.cur['accepted'] if self.cur else 0}
            if self.mode == 'matches':
                out['outstanding'] = sum(len(s.assigned) for s in self.sess if s is not None)
                out['matches'] = self.cur['matches'] if self.cur else 0
        return out

    def close(self):
        """The run is ending (STOP, PAUSE, an error, the last iteration): the workers quit (twice
        is harmless)."""
        with self.mu:
            if self.closed:
                return
            self.closed = True
            self.want = None
            self.accepting = None
            sess = [s for s in self.sess if s is not None]
            self.mu.notify_all()
        ths = [threading.Thread(target=s.close, daemon=True) for s in sess]
        for th in ths:
            th.start()
        for th in ths:
            th.join(15)
        for _ in self.fths:
            self.fq.put(None)

    # ---------------------------------------------------------------- events (reader threads)
    def _event(self, s, ev):
        kind = ev.get('ev')
        rows = None
        with self.mu:
            if kind == 'hello':
                s.hello = ev
                self.streak = 0
                if self.fails or s.k == 0:
                    self.log(f'worker {s.k} up (pid {ev.get("pid")} on {ev.get("host")}, mirror {self.rdir})')
            elif kind == 'status':
                s.status = ev
            elif kind == 'shard':
                if self.cur is not None and ev.get('iter') == self.accepting == self.cur['it']:
                    self.cur['pending'] += 1
                    self.cur['announced'] += 1
                    self.fq.put((s, ev))
                else:
                    self.late_total += 1               # its phase is over: dropped
                    s.send({'cmd': 'fetched', 'path': ev.get('path')})
                    self.log(f'worker {s.k}: shard {os.path.basename(str(ev.get("path")))} of iteration '
                             f'{ev.get("iter")} came after its phase closed: dropped')
            elif kind == 'row':
                rows = [ev['row']]
                s.assigned.pop(ev['row'].get('tag'), None)
            elif kind == 'end':
                s.ended[ev['iter']] = ev
            elif kind == 'error':
                s.error = ev
            self.mu.notify_all()
        if rows is not None and self.co is not None:
            co = self.co
            try:
                n = co.collect_rows(rows)
            except BaseException as e:                 # noqa: BLE001 - raised on the main thread
                co.error = e
                with co.cv:
                    co.cv.notify_all()
                return
            with self.mu:
                if self.cur is not None:
                    self.cur['matches'] += n
                    self.cur['dup'] += len(rows) - n

    # ---------------------------------------------------------------- the manager thread
    def _loop(self):
        last_err = 0.0
        while True:
            with self.mu:
                if self.closed:
                    return
                want = self.want
            try:
                self._tick(want)
            except Exception as e:                     # noqa: BLE001 - logged, retried
                if _now() - last_err > 60:
                    last_err = _now()
                    self.log(f'{type(e).__name__}: {str(e)[-400:]}')
            with self.mu:
                if self.closed:
                    return
                self.mu.wait(0.25)

    def _tick(self, want):
        now = _now()
        for k in range(self.procs):
            s = self.sess[k]
            if s is not None and not s.alive() and not s.handled:
                self._lost(s)
                s = None
            if s is None:
                if want is not None and now >= self.next_try[k]:
                    try:
                        self._start(k)
                    except Exception as e:             # noqa: BLE001 - logged, retried
                        self.fails += 1
                        self.next_try[k] = now + self.retry_s
                        if self.cur is not None:
                            self.cur['errors'] += 1
                        self.fail_log(f'worker {k} not started: {type(e).__name__}: {str(e)[-300:]}; retry in '
                                      f'{self.retry_s:.0f}s')
                continue
            if now - s.last > STALL_S:
                self.log(f'worker {k} silent for {now - s.last:.0f}s: killed')
                s.kill()
                continue
            if now - s.pinged > PING_S:
                s.pinged = now
                s.send({'cmd': 'ping'})
            if want is not None and s.hello is not None and s.phase != want:
                if self._ship_model(want):
                    with self.mu:
                        if self.want == want and s.alive():
                            s.send({'cmd': 'phase', 'iter': want, 'model': f'models/iter_{want - 1:04d}.pt'})
                            s.phase = want
            if self.mode == 'matches':
                self._timeouts(s)
                if want is not None and s.phase == want:
                    self._assign(s)

    def _prepare_mirror(self):
        if self.mirror_ready:
            return
        self.t.mkdirs(self.rdir + '/models', self.rdir + '/data')
        cfg = dict(self.R.cfg, **self.overrides)
        cfg.update(eval_remote='', selfplay_remote='', resolver='', resolver_cfg={}, resolver_init='')
        if 'shard_rows' not in self.overrides:
            cfg['shard_rows'] = min(int(cfg['shard_rows']), SHARD_ROWS)
        try:
            test = json.loads(os.environ.get('ZERO_RSP_TEST') or '{}')
        except ValueError:
            test = {}
        spj = {'mode': self.mode, 'locks': self.locks, 'nice': self.nice, 'backoff': self.backoff, 'run_id': self.run_id,
               'workdir': self.workdir, 'mirror': self.mirror, 'watchdog_s': WATCHDOG_S, 'test': test}
        for name, obj in (('config.json', cfg), ('selfplay.json', spj)):
            tmp = os.path.join(self.R.run, f'.remote_selfplay_{name}')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(obj, f, indent=1)
            self.t.put(tmp, self.rdir + '/' + name)
            try:
                os.remove(tmp)
            except OSError:
                pass
        self.mirror_ready = True

    def _start(self, k):
        self._prepare_mirror()
        s = Session(self, k)
        s.start()
        with self.mu:
            self.sess[k] = s

    def _ship_model(self, want):
        from .run import model_path
        if self.model_shipped == want - 1:
            return True
        if _now() < self.ship_retry:
            return False
        try:
            self.t.put(model_path(self.R.run, want - 1), f'{self.rdir}/models/iter_{want - 1:04d}.pt')
        except Exception as e:                         # noqa: BLE001 - retried
            self.ship_retry = _now() + 30
            self.log(f'checkpoint {want - 1} not shipped: {type(e).__name__}: {str(e)[-300:]}; retry in 30s')
            return False
        self.model_shipped = want - 1
        return True

    def _lost(self, s):
        """Session `s` ended (its pipes are closed: every row it sent was taken)."""
        s.handled = True
        rc = None
        if s.proc is not None:
            try:
                rc = s.proc.wait(timeout=5)            # its stdout closed: the exit code follows
            except subprocess.TimeoutExpired:
                s.kill()
                rc = s.proc.poll()
        why = (s.error or {}).get('msg') or s.tail()
        lived = _now() - s.started
        requeue = []
        with self.mu:
            if self.sess[s.k] is s:
                self.sess[s.k] = None
            if not self.closed:
                self.fails += 1
                self.next_try[s.k] = _now() + (0.0 if lived > 120 and not s.error else self.retry_s)
                if self.cur is not None:
                    self.cur['errors'] += 1
            requeue = [(tag, spec) for tag, (_t, spec) in s.assigned.items()]
            s.assigned.clear()
            # the next session starts from a mirror prepared again (it may be what failed: removed,
            # half-written) and gets its checkpoint again (6 MB)
            self.mirror_ready = False
            self.model_shipped = None
        if requeue and self.co is not None:
            self.co.requeue(requeue)
            with self.mu:
                if self.cur is not None:
                    self.cur['requeued'] += len(requeue)
        if self.closed:
            return
        self.fail_log(f'worker {s.k} ended (exit {rc}, after {lived:.0f}s): {str(why)[-500:]}'
                      + (f'; {len(requeue)} episodes requeued here' if requeue else '')
                      + f'; restart in {max(0.0, self.next_try[s.k] - _now()):.0f}s')
        err = s.error or {}
        if err.get('kind') == 'nonfinite':
            self._quarantine_remote(err)

    def _quarantine_remote(self, err):
        """The worker's own write check found a non-finite shard (its bad copy in the mirror's
        debug/): fetch it into <run>/debug/ and fail the run (as a local bad shard does)."""
        from .finite import NonFinite
        dst = None
        bad = err.get('bad')
        if bad:
            os.makedirs(self.R.debug_dir(), exist_ok=True)
            dst = os.path.join(self.R.debug_dir(), 'remote_' + os.path.basename(bad))
            try:
                if not self.t.get(bad, dst):
                    dst = None
            except Exception:                          # noqa: BLE001
                dst = None
        with self.mu:
            self.fatal = NonFinite(f'remote self-play ({self.name}): {err.get("msg", "")[-1500:]}'
                                   + (f' (quarantined here as {dst})' if dst else ''))
            self.mu.notify_all()

    # ---------------------------------------------------------------- co-training
    def _assign(self, s):
        co = self.co
        if co is None or co.closing or s.hello is None:
            return
        room = self.cap - len(s.assigned)
        if room <= 0:
            return
        items = co.take_for_remote(room)
        if not items:
            return
        act = float(s.status.get('active_s', 0.0))
        with self.mu:
            for tag, spec in items:
                s.assigned[tag] = (act, spec)
                s.send({'cmd': 'enqueue', 'tag': tag, 'spec': spec})
            if self.cur is not None:
                self.cur['assigned'] += len(items)

    def _timeouts(self, s):
        if self.co is None or 'active_s' not in s.status:
            return
        act = float(s.status['active_s'])
        with self.mu:
            late = [(tag, spec) for tag, (t, spec) in s.assigned.items() if act - t > self.timeout_s]
            for tag, _ in late:
                del s.assigned[tag]
        if late:
            self.co.requeue(late)
            with self.mu:
                if self.cur is not None:
                    self.cur['requeued_timeout'] += len(late)
            self.log(f'worker {s.k}: {len(late)} episodes not finished within {self.timeout_s:g}s of its running '
                     f'time: requeued here (a second result is dropped)')

    # ---------------------------------------------------------------- shards (the fetch thread)
    def _fetcher(self):
        while True:
            item = self.fq.get()
            if item is None:
                return
            s, ev = item
            it = ev['iter']
            name = os.path.basename(ev['path'])
            d = os.path.join(self.R.run, 'data', f'iter_{it:04d}')
            final = os.path.join(d, name)
            tmp = final[:-len('.npz')] + '.tmp.npz'   # never a loadable shard name (data.shard_paths)
            t0 = _now()
            res, rows = 'failed', 0
            try:
                with self.mu:
                    go = self.accepting == it
                if go:
                    os.makedirs(d, exist_ok=True)
                    if not self.t.get(ev['path'], tmp):
                        raise RuntimeError(f'copy of {ev["path"]} failed')
                    res, rows = self._check(tmp, it, name)
                    if res == 'ok':
                        with self.mu:
                            if self.accepting == it:
                                os.replace(tmp, final)
                                res = 'accepted'
                            else:
                                res = 'late'
                else:
                    res = 'late'
            except Exception as e:                     # noqa: BLE001 - the shard is lost, the run goes on
                self.log(f'shard {name} of iteration {it} not fetched: {type(e).__name__}: {str(e)[-300:]}')
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            s.send({'cmd': 'fetched', 'path': ev['path']})
            with self.mu:
                c = self.cur
                if c is not None and c['it'] == it:
                    c['pending'] -= 1
                    c['fetch_s'] += _now() - t0
                    if res == 'accepted':
                        c['accepted'] += 1
                        c['rows'] += rows
                        c['bytes'] += int(ev.get('bytes', 0))
                    elif res in ('rejected', 'quarantined'):
                        c[res] += 1
                    elif res == 'failed':
                        c['failed'] += 1
                self.mu.notify_all()

    def _check(self, path, it, name):
        """A fetched shard -> ('ok', rows) | ('rejected', 0) (another game's or encoder's: logged)
        | ('quarantined', 0) (non-finite: moved to <run>/debug/, the run fails at end())."""
        from .data import read_shard
        from .finite import NonFinite, shard_problems
        sh = read_shard(path)
        meta = sh['meta']
        info = self.R.info
        want = (self.R.cfg['game'], info['num_features'], info['num_actions'])
        got = (meta.get('game'), meta.get('features'), meta.get('actions'))
        if got != want:
            self.log(f'shard {name} of iteration {it} REJECTED: game / features / actions {got}, the run\'s {want} '
                     f'(the remote\'s code is not the run\'s?)')
            return 'rejected', 0
        probs = shard_problems(sh)
        if probs:
            os.makedirs(self.R.debug_dir(), exist_ok=True)
            bad = os.path.join(self.R.debug_dir(), f'iter_{it:04d}_{name[:-len(".npz")]}.bad.npz')
            os.replace(path, bad)
            msg = (f'remote self-play ({self.name}): non-finite sample values in its shard {name} of iteration {it} '
                   f'(meta {json.dumps(meta)}): ' + '; '.join(probs) + f'. Quarantined in {bad} (never loaded by the '
                   f'window); the run stops at the end of this self-play phase')
            self.log(msg)
            with self.mu:
                self.fatal = NonFinite(msg)
            return 'quarantined', 0
        return 'ok', int(meta.get('rows', len(sh['player'])))


# ================================================================== the remote side

class _BgWriter:
    """A ShardWriter on a thread of its own: the driver's thread hands it sample dicts and never
    waits for a shard's compression."""

    def __init__(self, writer):
        self.w = writer
        self.q = queue.Queue()
        self.error = None
        self.th = threading.Thread(target=self._run, name='zero-rsp-writer', daemon=True)
        self.th.start()

    def _run(self):
        while True:
            s = self.q.get()
            if s is None:
                return
            if self.error is not None:
                continue
            try:
                self.w.add(s)
            except BaseException as e:                 # noqa: BLE001 - raised on the driver's thread
                self.error = e

    def add(self, s):
        if s is not None and len(s['player']):
            self.q.put(s)

    def check(self):
        if self.error is not None:
            raise self.error

    def close(self):
        self.q.put(None)
        self.th.join()
        self.check()
        return self.w.close()


class Worker:
    """`python -m <pkg>.az.remote selfplay MIRROR K` (module docstring)."""

    def __init__(self, rdir, k):
        self.rdir, self.k = rdir, k
        with open(os.path.join(rdir, 'config.json'), encoding='utf-8') as f:
            self.cfg = json.load(f)
        with open(os.path.join(rdir, 'selfplay.json'), encoding='utf-8') as f:
            self.sp = json.load(f)
        self.mode = self.sp.get('mode', 'selfplay')
        self.test = self.sp.get('test') or {}
        self.sid = '%04x' % random.SystemRandom().getrandbits(16)
        self.prefix = f'm{k:02d}-{self.sid}'
        self.locks = list(self.sp.get('locks') or [])
        # 'background': play on under the Darwin background policy while a lock is held (macOS);
        # elsewhere, or if the policy cannot be set, 'stop'
        self.bg_mode = self.sp.get('backoff') == 'background' and hasattr(os, 'PRIO_DARWIN_BG')
        self.watchdog_s = float(self.sp.get('watchdog_s', WATCHDOG_S))
        self.status_s = float(self.test.get('status_s', STATUS_S))
        self.slot = os.path.join(rdir, f'.w{k:02d}.lock')
        wd = self.sp.get('workdir')
        self.busy = os.path.join(wd, f'.busy-{self.sp.get("mirror", "run")}-w{k:02d}') if wd else None
        self.out_mu = threading.Lock()
        self.ctl = queue.Queue()
        self.last_ctl = _now()
        self.closing = False
        self.stash = None                          # a phase command that came while ending a phase
        self.backlog = []                          # episodes queued before the driver exists
        self.drv = self.evr = None
        self.active_s = 0.0                        # running time in phases (back-offs excluded)
        self.t_status = 0.0
        self.rows_out = 0
        self.shards_out = 0
        self.mine = False

    # ---- output
    def emit(self, ev, **kw):
        line = TAG + json.dumps(dict(ev=ev, w=self.k, **kw), separators=(',', ':'))
        with self.out_mu:
            try:
                sys.stdout.write(line + '\n')
                sys.stdout.flush()
            except (OSError, ValueError):
                self.closing = True

    def note(self, msg):
        """The worker's own log (<mirror>/w<K>.log) and stderr."""
        line = time.strftime('%Y-%m-%d %H:%M:%S ') + f'[w{self.k} {self.sid} pid {os.getpid()}] {msg}'
        try:
            with open(os.path.join(self.rdir, f'w{self.k:02d}.log'), 'a', encoding='utf-8') as f:
                f.write(line + '\n')
        except OSError:
            pass
        print(line, file=sys.stderr, flush=True)

    # ---- control
    def _stdin(self):
        try:
            for line in sys.stdin:
                self.last_ctl = _now()
                try:
                    self.ctl.put(json.loads(line))
                except ValueError:
                    pass
        except (OSError, ValueError):
            pass
        self.ctl.put({'cmd': 'eof'})

    def _get(self, timeout):
        try:
            return self.ctl.get(timeout=timeout)
        except queue.Empty:
            return None

    def _watchdog(self):
        if not self.closing and _now() - self.last_ctl > self.watchdog_s:
            self.note(f'no control line for {_now() - self.last_ctl:.0f}s: the run is gone; exiting')
            self.closing = True

    def _control(self, msg, ph):
        cmd = msg.get('cmd')
        if cmd == 'enqueue':
            if self.drv is None:
                self.backlog.append((msg['tag'], msg['spec']))
            else:
                self.drv.enqueue(msg['tag'], msg['spec'])
        elif cmd == 'end':
            if ph is not None:
                self._end_phase(ph)
            else:                                  # nothing running: answer at once
                self.emit('end', iter=msg.get('iter'), games=0, samples=0, shards=0, bytes=0, leaves=0, seconds=0.0,
                          active_s=0.0, backoff_s=0.0, in_flight=0)
        elif cmd in ('quit', 'eof'):
            self.closing = True
            if ph is not None:
                self._end_phase(ph)
        elif cmd in ('fetched', 'drop'):
            self._rm(msg.get('path'))
        elif cmd == 'phase':
            if ph is not None:                     # the run moved on while this phase ends
                self.stash = msg
                self._end_phase(ph)
            else:
                self._phase(msg)

    def _controls(self, ph):
        while True:
            try:
                msg = self.ctl.get_nowait()
            except queue.Empty:
                return
            self._control(msg, ph)

    def _rm(self, path):
        if not path:
            return
        root = os.path.abspath(self.rdir)
        p = os.path.abspath(path)
        if p.startswith(root + os.sep):
            try:
                os.remove(p)
            except OSError:
                pass

    # ---- slot, busy marker
    def _take_slot(self):
        me = f'{os.getpid()} {_now()} {self.sp.get("run_id", "")}'
        for _ in range(100):
            try:
                os.mkdir(self.slot)
            except FileExistsError:
                try:
                    with open(os.path.join(self.slot, 'owner'), encoding='utf-8') as f:
                        parts = f.read().split(' ', 2)
                    pid, run = int(parts[0]), (parts[2] if len(parts) > 2 else '')
                except (OSError, ValueError, IndexError):
                    try:
                        if _now() - os.path.getmtime(self.slot) > 30:
                            shutil.rmtree(self.slot, ignore_errors=True)
                    except OSError:
                        pass
                    time.sleep(0.5)
                    continue
                if not _alive(pid):
                    shutil.rmtree(self.slot, ignore_errors=True)
                    continue
                if run.strip() != self.sp.get('run_id', ''):
                    raise SystemExit(f'remote self-play: the worker slot {self.slot} is held by another run '
                                     f'({run.strip()}, pid {pid}): two runs with the same mirror name?')
                self.note(f'the same run\'s previous worker pid {pid} still holds the slot: stopping it')
                _terminate(pid)
                shutil.rmtree(self.slot, ignore_errors=True)
                continue
            with open(os.path.join(self.slot, 'owner'), 'w', encoding='utf-8') as f:
                f.write(me)
            self.mine = True
            return
        raise RuntimeError(f'remote self-play: cannot take the worker slot {self.slot}')

    def _release_slot(self):
        if not self.mine:
            return
        try:
            with open(os.path.join(self.slot, 'owner'), encoding='utf-8') as f:
                pid = int(f.read().split()[0])
        except (OSError, ValueError, IndexError):
            pid = os.getpid()
        if pid == os.getpid():
            shutil.rmtree(self.slot, ignore_errors=True)

    def _busy(self, on):
        if not self.busy:
            return
        if on:
            try:
                os.makedirs(self.busy, exist_ok=True)
                with open(os.path.join(self.busy, 'owner'), 'w', encoding='utf-8') as f:
                    f.write(f'{os.getpid()} {_now()}')
            except OSError:
                pass
        else:
            shutil.rmtree(self.busy, ignore_errors=True)

    # ---- the driver
    def _setup(self):
        n = int(self.sp.get('nice', 0))
        if n > 0 and hasattr(os, 'nice'):
            try:
                os.nice(max(0, n - os.nice(0)))
            except OSError:
                pass
        import torch
        torch.set_num_threads(max(1, int(self.cfg.get('torch_threads', 1))))

    def _load(self, it, model):
        from .infer import NetEvaluator
        from .net import load
        from .run import Run
        c = self.cfg
        m, _ = load(os.path.join(self.rdir, model), c['infer_device'])
        if self.drv is None:
            R = Run(self.rdir, c)
            self.evr = NetEvaluator(m, c['infer_device'], workers=c.get('infer_workers', 1))
            seat = R.nn_seat('m', c['sims'], selfplay=True)
            seed = int(c['seed']) * 7919 + it + 7_777_777 * (self.k + 1)
            if self.mode == 'matches':
                self.drv = R.driver([[seat] * c['players'][0]], seed=seed, record=True, rolling=True, net=it - 1,
                                    queue_only=True)
            else:
                self.drv = R.driver([[seat] * p for p in c['players']], seed=seed, record=True, rolling=True, net=it - 1)
            for tag, spec in self.backlog:
                self.drv.enqueue(tag, spec)
            self.backlog = []
            self.info = R.info
        else:
            self.evr.set_weights(m.state_dict())
        for p in glob.glob(os.path.join(self.rdir, 'models', 'iter_*.pt')):
            try:
                if int(os.path.basename(p)[5:9]) < it - 3:
                    os.remove(p)
            except (OSError, ValueError):
                pass

    def _cleanup(self):
        """This worker slot's shards left in the mirror (fetched ones are deleted at once; these are
        the late, the dropped and an earlier session's)."""
        for p in glob.glob(os.path.join(self.rdir, 'data', 'iter_*', f'm{self.k:02d}-*')):
            try:
                os.remove(p)
            except OSError:
                pass
        for d in glob.glob(os.path.join(self.rdir, 'data', 'iter_*')):
            try:
                os.rmdir(d)
            except OSError:
                pass

    def _on_shard(self, it):
        def cb(path, rows, nbytes):
            self.shards_out += 1
            if self.test.get('corrupt') and self.shards_out == 1:
                _corrupt(path)                     # the tests: a bad file the arrival check must catch
            if self.test.get('delay_shard_s'):
                time.sleep(float(self.test['delay_shard_s']))
            self.emit('shard', iter=it, path=path, rows=rows, bytes=nbytes)
            if self.test.get('die_after_shards') and self.shards_out >= int(self.test['die_after_shards']):
                self.note('test: dying after a shard')
                os._exit(9)
        return cb

    def _end_phase(self, ph):
        if not ph['ending']:
            ph['ending'] = True
            self.drv.end_phase()

    def _status(self, ph, force=False, backoff=None):
        now = _now()
        if not force and now - self.t_status < (self.status_s if ph is not None else IDLE_STATUS_S):
            return
        self.t_status = now
        if ph is None:
            self.emit('status', phase=None, active_s=round(self.active_s, 1), in_flight=0, games=0)
            return
        bo = ph['backoff'] + (now - ph['bo_t0'] if ph['bo_t0'] else 0.0)
        st = self.drv.stats()
        self.emit('status', phase=ph['it'], games=ph['games'], samples=ph['samples'], leaves=st['leaves'] - ph['leaves0'],
                  in_flight=self.drv.in_flight()['games'], queued=self.drv.queued,
                  active_s=round(self.active_s + (now - ph['t0']) - bo, 1), phase_s=round(now - ph['t0'], 1),
                  backoff=backoff or ph.get('bg'), background=bool(ph.get('bg')))

    def _background(self, on):
        """The Darwin background policy for the whole process (its Rust and torch threads too):
        the lowest CPU band, the efficiency cores. -> True if it was set / cleared."""
        try:
            os.setpriority(os.PRIO_DARWIN_PROCESS, 0, os.PRIO_DARWIN_BG if on else 0)
            return True
        except (AttributeError, OSError) as e:
            self.note(f'background policy not {"set" if on else "cleared"}: {e}')
            return False

    def _backoff(self, ph):
        now = _now()
        if now - ph['t_lock'] < LOCK_POLL_S:
            return
        ph['t_lock'] = now
        why = lock_held(self.locks)
        if ph.get('bg'):                           # in the background: leave it once the lock is gone
            if not why:
                self._background(False)
                took = now - ph['bo_t0']
                ph['backoff'] += took
                ph['bo_t0'] = 0.0
                ph['bg'] = None
                self.note(f'out of the background after {took:.0f}s')
                self._busy(True)
                self._status(ph, force=True)
            return
        if not why:
            return
        if self.bg_mode and self._background(True):
            self.note(f'{why} is held: playing on in the background (efficiency cores, lowest priority)')
            self._busy(False)
            ph['bo_t0'] = now
            ph['bg'] = why
            self._status(ph, force=True, backoff=why)
            return
        self.note(f'backing off: {why} is held')
        self._busy(False)
        ph['bo_t0'] = now
        self._status(ph, force=True, backoff=why)
        while not ph['ending'] and not self.closing:
            msg = self._get(1.0)
            if msg is not None:
                self._control(msg, ph)
            self._watchdog()
            if _now() - ph['t_lock'] >= LOCK_POLL_S:
                ph['t_lock'] = _now()
                if not lock_held(self.locks):
                    break
            self._status(ph, backoff=why)
        took = _now() - ph['bo_t0']
        ph['backoff'] += took
        ph['bo_t0'] = 0.0
        self.note(f'continuing after {took:.0f}s')
        if not ph['ending'] and not self.closing:
            self._busy(True)
        if self.closing:
            self._end_phase(ph)

    def _phase(self, msg):
        from .data import ShardWriter
        from .infer import drive
        it = int(msg['iter'])
        self._load(it, msg['model'])
        self._cleanup()
        c = self.cfg
        drv = self.drv
        meta = {'game': c['game'], 'features': self.info['num_features'], 'actions': self.info['num_actions'],
                'iter': it, 'source': 'remote', 'worker': self.k, 'session': self.sid}
        writer = ShardWriter(os.path.join(self.rdir, 'data', f'iter_{it:04d}'), self.prefix, rows=int(c['shard_rows']),
                             meta=meta, bad_dir=os.path.join(self.rdir, 'debug'), compress=COMPRESS,
                             on_write=self._on_shard(it))
        bw = _BgWriter(writer)
        st0 = drv.stats()
        drv.resume(BIG, it - 1)
        ph = {'it': it, 'games': 0, 'samples': 0, 't0': _now(), 'backoff': 0.0, 'bo_t0': 0.0, 'ending': False,
              't_lock': 0.0, 'leaves0': st0['leaves']}
        self._busy(True)
        slow = float(self.test.get('slow_s') or 0)
        die_rows = int(self.test.get('die_after_rows') or 0)

        def pump(lb=None):
            self._controls(ph)
            self._watchdog()
            if self.closing:
                self._end_phase(ph)
            s = drv.take_samples()
            if s is not None:
                ph['samples'] += len(s['player'])
                bw.add(s)
            bw.check()
            for r in drv.take_results():
                ph['games'] += 1
                if self.mode == 'matches':
                    self.emit('row', row=r)
                    self.rows_out += 1
                    if die_rows and self.rows_out >= die_rows:
                        self.note('test: dying after a row')
                        os._exit(9)
            if slow:
                time.sleep(slow)
            if not ph['ending']:
                self._backoff(ph)
            self._status(ph)

        while True:
            pump()
            drive(drv, {'m': self.evr}, on_batch=pump)
            if ph['ending']:
                break
            msg = self._get(0.5)               # queue_only with nothing to play: wait for episodes
            if msg is not None:
                self._control(msg, ph)
            self._watchdog()
            if self.closing:
                self._end_phase(ph)
            self._status(ph)
        pump()
        if ph.get('bg'):
            self._background(False)
            ph['backoff'] += _now() - ph['bo_t0']
            ph['bo_t0'] = 0.0
            ph['bg'] = None
        w = bw.close()
        st = drv.stats()
        now = _now()
        secs = now - ph['t0']
        self.active_s += secs - ph['backoff']
        self._busy(False)
        self.emit('end', iter=it, games=ph['games'], samples=ph['samples'], shards=w['shards'], bytes=w['bytes'],
                  leaves=st['leaves'] - ph['leaves0'], seconds=round(secs, 2), active_s=round(secs - ph['backoff'], 2),
                  backoff_s=round(ph['backoff'], 2), in_flight=drv.in_flight()['games'], queued=drv.queued)
        self.note(f'phase {it}: {ph["games"]} games, {ph["samples"]} samples in {w["shards"]} shards '
                  f'({w["bytes"] / 1e6:.1f} MB), {(st["leaves"] - ph["leaves0"]) / max(secs - ph["backoff"], 1.0):.0f} '
                  f'leaves/s running, {secs:.0f}s ({ph["backoff"]:.0f}s backing off)')

    def main(self):
        from .finite import NonFinite
        try:
            self._take_slot()
            self._setup()
            threading.Thread(target=self._stdin, name='zero-rsp-stdin', daemon=True).start()
            self.note(f'started: mode {self.mode}, mirror {self.rdir}, conc {self.cfg["conc"]}, threads '
                      f'{self.cfg["threads"]}, groups {self.cfg["groups"]}, infer_workers {self.cfg.get("infer_workers", 1)}, '
                      f'torch threads {self.cfg.get("torch_threads", 1)}, nice {self.sp.get("nice", 0)}')
            self.emit('hello', pid=os.getpid(), host=socket.gethostname(), sid=self.sid, conc=self.cfg['conc'],
                      mode=self.mode)
            while not self.closing:
                if self.stash is not None:
                    msg, self.stash = self.stash, None
                else:
                    msg = self._get(1.0)
                if msg is not None:
                    self._control(msg, None)
                self._watchdog()
                self._status(None)
            return 0
        except NonFinite as e:
            bad = None
            for ln in str(e).split('. '):
                if 'kept in ' in ln:
                    bad = ln.split('kept in ', 1)[1].split(' (never')[0].strip()
            self.emit('error', kind='nonfinite', msg=str(e)[-3000:], bad=bad)
            self.note(f'NonFinite: {e}')
            return 1
        except SystemExit as e:
            self.emit('error', kind='refused', msg=str(e))
            self.note(f'refused: {e}')
            return 2
        except BaseException as e:                     # noqa: BLE001 - reported to the run
            tb = traceback.format_exc()
            self.emit('error', kind='error', msg=f'{type(e).__name__}: {e}'[-2000:])
            self.note(tb)
            return 1
        finally:
            self._busy(False)
            self._release_slot()


def _terminate(pid, grace=10.0):
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    t = _now() + grace
    while _now() < t:
        if not _alive(pid):
            return
        time.sleep(0.2)
    try:
        os.kill(pid, getattr(signal, 'SIGKILL', signal.SIGTERM))
    except OSError:
        pass


def _corrupt(path):
    """Tests: rewrite a shard with a NaN in x (as a damaged file would arrive)."""
    import numpy as np
    from .data import read_shard, savez_level
    sh = read_shard(path)
    meta = sh.pop('meta')
    x = sh['x'].copy()
    x[0, 0] = np.nan
    sh['x'] = x
    savez_level(path + '.c.tmp', COMPRESS, dict(meta=np.array(json.dumps(meta)), **sh))
    os.replace(path + '.c.tmp', path)


def serve(argv):
    """`python -m <pkg>.az.remote selfplay MIRROR K`."""
    if len(argv) != 2:
        raise SystemExit('usage: python -m <pkg>.az.remote selfplay MIRROR_DIR WORKER')
    w = Worker(argv[0], int(argv[1]))
    code = w.main()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except (OSError, ValueError):
        pass
    os._exit(code)
