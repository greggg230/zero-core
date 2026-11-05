"""Remote self-play (docs/az.md 5.3; co-training 11.6): another machine self-plays for a run, and
both machines pull an iteration's work from one shared queue (zero/workqueue.py).

* **A single-game run**: iteration i needs `games` finished games in total, across this machine
  and the remote's workers, all searched with checkpoint i-1 (the remote's sample shards come back
  into `data/iter_i/`). Every machine starts games as it has room under its in-flight cap (its
  measured throughput's share of the iteration's games x `selfplay_remote_inflight`; a batch-bound
  one - CUDA / MPS inference - its slots while its games finish within
  `selfplay_remote_batch_inflight` iterations), and a finished game counts only once the run
  **accepts** it: this machine's at once, a worker's when the run answers its `fin` line. The
  iteration ends when `games` are accepted - exactly; games finishing beyond the quota wait, held
  with their samples, for the next iteration (its quota takes them first). A worker never holds a
  share of the quota: a remote that backs off, sleeps or dies leaves the rest of the iteration to
  this machine (whose share grows at once).
* **A co-training run** (`resolver_cfg['selfplay_remote']`): the resolver's matches are the
  coordinator's queue (zero/cotrain.py); this machine's resolver driver takes its share first (the
  coordinator keeps it in reserve), the workers take theirs as they have room under their caps;
  rows come back by tag. A worker backing off takes no new matches, and after
  `selfplay_remote_giveback` s gives its outstanding ones back (requeued here, the second copy of a
  row dropped); a match outstanding longer than `selfplay_remote_overdue` x its worker's expected
  latency (at most `selfplay_remote_timeout` s) of that worker's running time is requeued too.

Run settings: `selfplay_remote` (a name in the game package's `remotes`; 'local' = this machine,
the tests; '' = off), `selfplay_remote_procs` (worker processes; 0 = the remote's `procs`),
`selfplay_remote_wait` (seconds the end of a phase waits for the remote's last shards),
`selfplay_remote_inflight` (each machine's in-flight cap, in iterations of its throughput),
`selfplay_remote_batch` ('auto': the consumers whose inference is CUDA / MPS are batch-bound;
'local', 'remote', 'all', 'none') and `selfplay_remote_batch_inflight` (a batch-bound consumer's
cap is its slots while its games take at most this many iterations, zero/workqueue.py),
`selfplay_remote_giveback`, `selfplay_remote_overdue`, `selfplay_remote_timeout` (co-training),
`selfplay_remote_linger` (0/1: the workers keep playing with their checkpoint while the run trains;
those games and rows count for the next iteration, at most one checkpoint stale).

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
  control channel (JSON lines: phase, end, cap, accept, enqueue, fetched, ping, quit), its stdout
  the worker's events (lines `ZERO_SP {json}`: hello, status, fin, shard, row, end, error). A
  session that fails or stays silent for `STALL_S` is killed and started again after `RETRY_S`.
  The mirror `<workdir>/<mirror>` gets config.json (the run's settings + the remote's selfplay
  overrides) and selfplay.json (the worker's mode, locks, niceness, linger, the run's id) at each
  session start, and checkpoint i-1 at each phase start; it computes the caps (`Shares`) every tick
  and sends a worker its own (`cap`, single-game) or assigns it matches under it (co-training);
* `begin(it)`: the sessions get `phase it` (their drivers' new search epoch with checkpoint i-1);
  `end(it)` (or the quota met, single-game): they get `end it`, finish their phase (games in flight
  carry over, as the run's own rolling driver's do) and flush their shards; `end` waits for the
  shards announced in phase i at most `selfplay_remote_wait` s: what is not back by then is dropped
  (a shard of phase i never enters another iteration);
* fetch threads (one per worker) copy each announced shard (a temporary name), check it (the
  non-finite guard, zero/finite.py, and the run's game / features / actions) and rename it into
  `data/iter_i/` - only while phase i accepts. A non-finite shard is quarantined to `<run>/debug/`
  and the run fails with NonFinite at the end of the phase, as a local one would (the window never
  sees it). Every arrived shard's rows go into the iteration's staleness histogram;
* co-training (`co`, the Coordinator): the manager takes queued episodes from the coordinator (at
  most its cap outstanding per session) and sends them to a worker; its rows go into the
  coordinator by tag; a session that dies or stalls has its outstanding episodes requeued here (at
  the front), as has one backing off for longer than the give-back delay or an episode overdue;
  whichever copy finishes second is dropped (at worst played twice).

Remote side (`serve` -> `Worker`): one rolling `queue_only` driver (self-play: the run's seats and
spec, a setup queued for every game the worker may start under its cap; co-training: the queued
episodes) persistent across the run's iterations. It backs off - stops asking for leaf batches, so
its threads idle - while one of its locks is held, marks itself busy for lower-priority jobs
(`<workdir>/.busy-<mirror>-w<K>`: an owner file "<pid> <time>" as the eval lock's) while a phase
runs, writes compressed shards (`data/iter_i/m<KK>-<sid>_SSS.npz`, zlib level 1, on a background
thread) of the games the run accepted and exits when its stdin closes, on `quit`, or when no
control line came for `watchdog_s` (an orphan never outlives its run by long). A worker started for
a slot (mirror, K) that a live worker of the same run still holds stops that one first; another
run's is refused.
"""
import contextlib
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
from .shardthread import SampleClock, sample_rows, split_records
from .workqueue import Shares, lag_hist, merge_hist, split_games

TAG = 'ZERO_SP '


def _belief_of(cfg):
    """The root-belief keys of a run's self-play seat (`run.belief_keys`, {} when off): what a worker's
    hello reports and the run checks (docs/az.md 18.5)."""
    from .run import belief_keys
    return belief_keys(cfg, 'selfplay')


def _resign_of(cfg):
    """A run's self-play resignation (`resign.fingerprint`, '' when off): what a worker's hello reports and the
    run checks (docs/az.md 27)."""
    from .resign import fingerprint
    return fingerprint(cfg)


def _merge_of(cfg):
    """A run's self-play information-set merging (`selfplay_infoset_merge`, '' when off): what a worker's hello
    reports and the run checks (docs/az.md 30) - an older checkout would search its games unmerged in silence."""
    return 'ism' if int(cfg.get('selfplay_infoset_merge', 0) or 0) else ''


def _forks_of(fk):
    """A run's branched games and archive starts as its driver plays them (`forks.fingerprint` of a Run's
    `forks`, '' when off): what a worker's hello reports and the run checks (docs/az.md 24)."""
    from .forks import fingerprint
    return fingerprint(fk)
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
OVERDUE_MIN_S = 120.0             # co-training: an episode is never overdue sooner (env ZERO_RSP_OVERDUE_MIN_S)
PROVISIONAL_S = 10.0              # an unmeasured worker's first rate after this much of its running time
                                  # in a phase (its status lines; env ZERO_RSP_PROVISIONAL_S)
BATCH_MODES = ('auto', 'local', 'remote', 'all', 'none')   # selfplay_remote_batch: which consumers are batch-bound
BATCH_DEVICES = ('cuda', 'mps')   # 'auto': inference on these batches (its rate grows with its in-flight)


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
        if pid != os.getpid() and _owner_alive(pid):
            return d
    return None


def _owner_alive(pid):
    """A lock owner is alive: its process exists and is not a zombie (a dead child its parent has
    not reaped still answers kill(pid, 0) on POSIX)."""
    if not _alive(pid):
        return False
    try:
        import psutil
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except ImportError:
        return True
    except Exception:                              # noqa: BLE001 - gone meanwhile
        return False


# ================================================================== the local side

class Session:
    """One remote worker process and its pipes (reader, writer and stderr threads). Events from
    the worker go to the owner's `_event` on the reader thread."""

    def __init__(self, rp, k):
        self.rp, self.k = rp, k
        self.name = f'w{k}'
        self.proc = None
        self.wq = queue.Queue()
        self.err = deque(maxlen=60)
        self.eof = threading.Event()
        self.hello = None
        self.phase = None                          # the iteration whose phase was sent
        self.end_sent = None                       # ... and whose end was sent
        self.ended = {}                            # iteration -> the worker's end event
        self.status = {}
        self.error = None
        self.last = _now()                         # last line from the worker
        self.started = _now()
        self.pinged = _now()
        self.assigned = {}                         # co-training: tag -> (worker active_s then, spec)
        self.handled = False
        # single-game: games the worker reported finished (it holds them until accepted), accepted
        self.fin = 0
        self.acc = 0
        self.acc_ph = 0                            # accepted in the open phase, and of those in announced shards
        self.ship_ph = 0
        self.bo_since = None                       # its back-off began (a status line said so)
        self.gave_back = False
        self.sent_cap = None
        self.util = [0.0, 0]                       # co-training: outstanding / cap, sampled while active

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

    def backing_off(self):
        return bool(self.status.get('backoff'))

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
    run's self-play: the shared quota of games, shards) or 'matches' (a co-training run's resolver
    episodes from the coordinator given to `begin`: rows and shards). `mirror`: the mirror's name in
    the remote's selfplay workdir (default the run directory's name; a co-training run's resolver:
    '<run name>.resolver'). The run's own driver is the consumer 'local' (`local(lo, hi)`), whose
    cap `local_cap()` gives."""

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

        def setting(k):
            v = c.get(k)
            return DEFAULTS.get(k) if v is None else v
        self.wait_s = float(setting('selfplay_remote_wait'))
        self.timeout_s = float(setting('selfplay_remote_timeout'))
        self.inflight = float(setting('selfplay_remote_inflight'))
        self.batch = str(setting('selfplay_remote_batch') or 'auto')
        if self.batch not in BATCH_MODES:
            raise SystemExit(f'selfplay_remote_batch {self.batch!r}: one of {", ".join(BATCH_MODES)}')
        self.batch_inflight = float(setting('selfplay_remote_batch_inflight') or 0.0)
        self.giveback_s = float(setting('selfplay_remote_giveback'))
        self.overdue = float(setting('selfplay_remote_overdue'))
        self.linger = bool(int(setting('selfplay_remote_linger') or 0))
        self.overdue_min = float(os.environ.get('ZERO_RSP_OVERDUE_MIN_S', OVERDUE_MIN_S))
        self.provisional_s = float(os.environ.get('ZERO_RSP_PROVISIONAL_S', PROVISIONAL_S))
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
        groups = int(self.overrides.get('groups') or c['groups'])
        # a worker's most outstanding: its driver's slots (self-play: it starts its own games), plus a
        # queue in its driver (co-training: matches travel to it, so its slots never wait for them)
        self.cap = conc + max(1, conc // 4) if mode == 'matches' else conc
        self.shares = Shares(self.inflight, self.batch_inflight)
        wbatch = self.batch_bound('remote', self.overrides.get('infer_device') or c['infer_device'])
        for k in range(self.procs):
            self.shares.consumer(f'w{k}', max(2, groups), self.cap, cautious=True, batch=wbatch, slots=conc)
        self.capv = {}                             # the caps now (the manager's last computation)
        self.events = None                         # co-training: events per iteration (the primary's games)
        self.q = None                              # single-game: the open phase's quota
        self.mu = threading.Condition()
        self.sess = [None] * self.procs
        self.next_try = [0.0] * self.procs
        self.fails = 0
        self.streak = 0                            # failures since a worker last said hello
        self.t_faillog = 0.0
        self.refusals = 0                          # workers that refused the run's checkpoint (docs/az.md 10.4)
        self.t_refused = 0.0
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

    # ---------------------------------------------------------------- the shared queue's model
    def batch_bound(self, who, device):
        """Whether consumer `who` ('local', or 'remote': a worker) inferring on `device` is batch-bound
        (selfplay_remote_batch; zero/workqueue.py)."""
        if self.batch == 'auto':
            return str(device) in BATCH_DEVICES
        return self.batch in ('all', who)

    def local(self, lo, hi, device='cpu', slots=None):
        """The run's own driver as a consumer of the queue: its cap's bounds (a few games per group;
        its driver's slots, `slots` when `hi` holds a queue beyond them), its inference `device`
        (CUDA / MPS: batch-bound, with selfplay_remote_batch 'auto')."""
        with self.mu:
            self.shares.consumer('local', lo, hi, batch=self.batch_bound('local', device), slots=slots)
            self._refresh()

    def batch_note(self):
        """The log's note on the batch-bound consumers ('' if none)."""
        with self.mu:
            names = [n for n, c in self.shares.c.items() if c.get('batch')]
        if not names or self.shares.fb <= self.shares.f:
            return ''
        return (f'; batch-bound ({", ".join(names)}): its slots while its games take at most '
                f'{self.shares.fb:g} iterations')

    def restore(self, rates):
        """The consumers' rates from the run's last metrics row (a restart: the first caps follow
        them, not the slots)."""
        with self.mu:
            self.shares.restore(rates)
            self._refresh()

    def local_cap(self):
        with self.mu:
            return self.capv.get('local', 1)

    def _units(self):
        """An iteration's units: a single-game run's games; a co-training run's matches (its events x
        their mean matches, once events have parked; None before: caps = the consumers' slots)."""
        if self.mode == 'selfplay':
            return int(self.R.cfg['games'])
        co = self.co
        if co is None or not self.events or not getattr(co, 'ev_total', 0):
            return None
        return max(1, round(self.events * co.ep_total / co.ev_total))

    def _refresh(self):
        """(mu held) Who takes work now, the caps, and the local driver's cap in the coordinator. A
        worker not measured yet gets its first rate from its status lines once it has run for
        `provisional_s` in a phase."""
        sh = self.shares
        for k in range(self.procs):
            s = self.sess[k]
            sh.set_active(f'w{k}', s is not None and s.alive() and s.hello is not None and not s.backing_off())
            st = s.status if s is not None else {}
            if st.get('phase') is not None and st.get('run_s', 0) >= self.provisional_s                     and sh.provisional(f'w{k}', st.get('leaves'), st['run_s']):
                self.log(f'worker {k}: first measured at {st["leaves"] / st["run_s"]:.0f} leaves/s in '
                         f'{st["run_s"]:.0f}s of its phase')
        sh.units = self._units()
        self.capv = sh.caps()
        if self.co is not None and 'local' in self.capv:
            self.co.local_cap = self.capv['local']

    # ---------------------------------------------------------------- the main thread's API
    def began(self, it):
        """The remote's phase `it` has begun already (started early: right after checkpoint it - 1
        was written, while the run evaluated it)."""
        with self.mu:
            return self.want == it and self.accepting == it

    def begin(self, it, co=None):
        """Iteration `it`'s self-play starts: the remote's as soon as it can (never waited for).
        The run calls it as soon as checkpoint it - 1 is written (before its evaluation) and again
        when its own phase starts (then a no-op). Single-game: the iteration's quota opens."""
        with self.mu:
            if self.fatal is not None:
                raise self.fatal
            if co is not None:
                self.co = co
            if self.want == it and self.accepting == it:
                self.cur['t_local'] = _now()           # this machine's phase starts now
                self._refresh()
                return
            self.want = it
            self.accepting = it
            self.cur = {'it': it, 't0': _now(), 't_local': _now(), 'announced': 0, 'accepted': 0, 'rows': 0, 'bytes': 0,
                        'pending': 0, 'rejected': 0, 'quarantined': 0, 'failed': 0, 'fetch_s': 0.0, 'assigned': 0,
                        'matches': 0, 'dup': 0, 'requeued': 0, 'requeued_timeout': 0, 'requeued_backoff': 0, 'errors': 0,
                        'games': 0, 'lost': 0, 'stale': {}, 'capsum': {}, 'capmax': {}, 'capn': 0}
            for s in self.sess:
                if s is not None:
                    s.acc_ph = s.ship_ph = 0
                    s.util = [0.0, 0]
            if self.mode == 'selfplay':
                self.q = {'it': it, 'need': int(self.R.cfg['games']), 'done': 0, 'by': {}, 'open': True}
            self._refresh()
            self.mu.notify_all()

    # ---- single-game: the quota
    def need(self, it):
        """Games iteration `it` still needs (single-game)."""
        with self.mu:
            q = self.q
            return max(0, q['need'] - q['done']) if q is not None and q['it'] == it and q['open'] else 0

    def accept_local(self, it, n):
        """This machine finished `n` more games (oldest first): how many count for iteration `it`
        (the rest wait for the next one)."""
        with self.mu:
            a = self._take(it, 'local', n)
            self._close_if_met()
            return a

    def _take(self, it, who, n):
        """(mu held) Up to `n` games of `who` into iteration `it`'s quota."""
        q = self.q
        if n <= 0 or q is None or q['it'] != it or not q['open']:
            return 0
        a = min(n, q['need'] - q['done'])
        if a <= 0:
            return 0
        q['done'] += a
        q['by'][who] = q['by'].get(who, 0) + a
        return a

    def _close_if_met(self):
        """(mu held) The quota met: closed, the workers told to end their phase (after any `accept`
        sent for it: a worker takes its control lines in order)."""
        q = self.q
        if q is not None and q['open'] and q['done'] >= q['need']:
            self._close_quota()

    def _give_back(self, s, it, n):
        """(mu held) `n` games counted for worker `s` in iteration `it` will not arrive: the quota
        needs them again, while this machine still plays the phase (then reopened: the workers have
        ended theirs, this machine plays the rest). -> False if too late (the iteration is short)."""
        q = self.q
        if n <= 0 or q is None or q['it'] != it or self.want != it:
            return False
        q['done'] -= n
        q['by'][s.name] = q['by'].get(s.name, 0) - n
        if not q['open'] and q['done'] < q['need']:
            q['open'] = True
        self.mu.notify_all()
        return True

    def _close_quota(self):
        """(mu held) The quota is met: every worker in the phase ends it now (its later games wait)."""
        q = self.q
        q['open'] = False
        for s in self.sess:
            if s is not None and s.alive() and s.phase == q['it'] and s.end_sent != q['it']:
                s.send({'cmd': 'end', 'iter': q['it']})
                s.end_sent = q['it']
        self.mu.notify_all()

    def _accept_remote(self, s):
        """(mu held) Worker `s`'s finished games, as many as the quota still needs."""
        q = self.q
        if q is None or s.phase != q['it'] or s.end_sent == q['it'] or not s.alive():
            return
        a = self._take(q['it'], s.name, s.fin - s.acc)
        if a:
            s.acc += a
            s.acc_ph += a
            s.send({'cmd': 'accept', 'iter': q['it'], 'n': a})
            self._close_if_met()

    def end(self, it, local=None):
        """Iteration `it`'s self-play ended here: the remote's phase ends too; waits at most
        `selfplay_remote_wait` s for its last shards. `local`: this machine's phase ({units, run_s,
        util}: the shared queue's measurement). -> the remote's share (metrics), with `queue`: the
        consumers' caps, rates and the remote rows' staleness."""
        t0 = _now()
        with self.mu:
            self.want = None
            if self.q is not None and self.q['it'] == it:
                self.q['open'] = False
            targets = [s for s in self.sess if s is not None and s.alive() and s.phase == it]
            for s in targets:
                if s.end_sent != it:
                    s.send({'cmd': 'end', 'iter': it})
                    s.end_sent = it
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
            # the shared queue: each consumer's phase measured, the next caps
            ncap = max(1, cur['capn'])
            phase_caps = {n: round(v / ncap, 1) for n, v in cur['capsum'].items()}
            meas = {}
            for s in self.sess:
                e = s.ended.get(it) if s is not None else None
                if not e:
                    continue
                util = (s.util[0] / s.util[1]) if (self.mode == 'matches' and s.util[1]) else 1.0
                # its head start (single-game: its running time before this machine's phase began;
                # co-training: none - the matches come with the drafting)
                lead = max(0.0, cur['t_local'] - cur['t0'] - e.get('backoff_s', 0.0)) if self.mode == 'selfplay'                     else 0.0
                meas[s.name] = {'units': e.get('games', 0), 'run_s': e.get('active_s', 0.0), 'util': util, 'lead': lead,
                                'leaves': e.get('leaves')}
            if local is not None:
                meas['local'] = {'units': local.get('units', 0), 'run_s': local.get('run_s', 0.0),
                                 'util': local.get('util'), 'leaves': local.get('leaves')}
            got = self.shares.observe_phase(meas)
            for n, m in meas.items():
                m['run_s'] = round(m['run_s'], 1)
                m['util'] = None if m.get('util') is None else round(m['util'], 3)
                m['rate_measured'] = None if got.get(n) is None else round(got[n], 4)
            self._refresh()
            rates = self.shares.rates()
            consumers = {}
            for n in self.shares.c:
                m = {k: v for k, v in meas.get(n, {}).items() if k not in ('leaves', 'lead')}
                consumers[n] = dict(m, cap=phase_caps.get(n), cap_max=cur['capmax'].get(n), cap_next=self.capv.get(n),
                                    rate=round(rates.get(n, 0.0), 4))
            quota = None
            if self.q is not None and self.q['it'] == it:
                quota = {'need': self.q['need'], 'accepted': self.q['done'], 'by': dict(self.q['by'])}
            held = sum(max(0, s.fin - s.acc) for s in self.sess if s is not None and s.alive())
            qblock = {'units': self.shares.units, 'inflight': self.inflight, 'batch_inflight': self.shares.fb,
                      'consumers': consumers, 'rates': self.shares.state(), 'stale': dict(cur['stale']), 'quota': quota,
                      'held_remote': held}
        ends = [e for e in ends if e]
        secs = max([e['seconds'] for e in ends], default=0.0)
        leaves = sum(e['leaves'] for e in ends)
        # single-game: the games whose samples arrived (accepted shards' games, and accepted games with
        # no samples - cut at max_steps - that are in no shard); co-training: the matches it played
        games = cur['games'] + sum(e.get('unshipped', 0) for e in ends) if self.mode == 'selfplay' \
            else sum(e['games'] for e in ends)
        out.update(games=games, games_played=sum(e['games'] for e in ends), samples=cur['rows'],
                   samples_played=sum(e['samples'] for e in ends),
                   shards=cur['accepted'], bytes=cur['bytes'], leaves=leaves,
                   leaves_per_s=round(sum(e['leaves'] / max(e['seconds'], 1e-9) for e in ends), 1),
                   leaves_per_running_s=round(sum(e['leaves'] / max(e['active_s'], 1e-9) for e in ends), 1),
                   seconds=round(secs, 1), running_s=round(max([e['active_s'] for e in ends], default=0.0), 1),
                   backoff_s=round(max([e['backoff_s'] for e in ends], default=0.0), 1),
                   in_flight=sum(e.get('in_flight', 0) for e in ends), late=late, late_prev=late_prev,
                   rejected=cur['rejected'], quarantined=cur['quarantined'], fetch_failed=cur['failed'],
                   fetch_s=round(cur['fetch_s'], 1), wait_s=round(_now() - t0, 2), session_errors=cur['errors'],
                   lead_s=round(cur['t_local'] - cur['t0'], 1), caps=phase_caps)
        if self.mode == 'selfplay':
            out.update(accepted=sum(e.get('accepted', 0) for e in ends), held=qblock['held_remote'], lost=cur['lost'])
            if any('league' in e for e in ends):       # the self-play league: its accepted games' (league.tally)
                from .league import merge
                lg = {}
                for e in ends:
                    lg = merge(lg, e.get('league'))
                out['league'] = lg
            if any('forks' in e for e in ends):        # branched games, archive starts: its accepted games' (forks.tally)
                from .forks import merge as fmerge
                fk = None
                for e in ends:
                    fk = fmerge(fk, e.get('forks'))
                out['forks'] = fk
            if any('resign' in e for e in ends):       # self-play resignation: its accepted games' (resign.tally)
                from .resign import merge as rmerge
                rs = None
                for e in ends:
                    rs = rmerge(rs, e.get('resign'))
                out['resign'] = rs
        if self.mode == 'matches':
            with self.mu:
                outstanding = sum(len(s.assigned) for s in self.sess if s is not None)
            out.update(matches=cur['matches'], assigned=cur['assigned'], requeued=cur['requeued'],
                       requeued_timeout=cur['requeued_timeout'], requeued_backoff=cur['requeued_backoff'],
                       duplicates_dropped=cur['dup'], outstanding=outstanding)
        if any('shard_writer' in e for e in ends):  # shard_writer_thread (docs/az.md 31): the workers' summed
            ws = [e['shard_writer'] for e in ends if 'shard_writer' in e]
            out['shard_writer'] = dict({k: round(sum(w.get(k, 0) for w in ws), 3) for k in
                                        ('items', 'rows', 'waits', 'wait_s', 'close_s', 'writer_s', 'samples_s')},
                                       queue_peak_mb=max(w.get('queue_peak_mb', 0) for w in ws), workers=len(ws))
        if any(e.get('linger') for e in ends):
            out['linger'] = {'games': sum((e.get('linger') or {}).get('games', 0) for e in ends),
                             'seconds': round(max((e.get('linger') or {}).get('seconds', 0.0) for e in ends), 1)}
        out['queue'] = qblock
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
                   'samples': self.cur['rows'] if self.cur else 0, 'shards': self.cur['accepted'] if self.cur else 0,
                   'caps': dict(self.capv)}
            if self.mode == 'selfplay' and self.q is not None:
                out['games'] = sum(v for k, v in self.q['by'].items() if k != 'local')
                out['quota'] = {'need': self.q['need'], 'accepted': self.q['done'], 'open': self.q['open']}
                out['held'] = sum(max(0, s.fin - s.acc) for s in live)
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
            if self.q is not None:
                self.q['open'] = False
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
                if self.R.league is not None and self.mode == 'selfplay' and not ev.get('league'):
                    # a checkout without the self-play league would play mirror games and record both
                    # seats of the league's games: never used (zero/league.py)
                    s.no_league = True
                want_b = _belief_of(self.R.cfg)
                if want_b and ev.get('belief') != want_b:
                    # a checkout without the root's belief (or reading the settings otherwise) would deal
                    # its worlds uniformly in silence: never used (docs/az.md 18.5)
                    s.no_belief = ev.get('belief')
                want_f = _forks_of(getattr(self.R, 'forks', None)) if self.mode == 'selfplay' else ''
                if want_f and ev.get('forks') != want_f:
                    # a checkout without branched games / archive starts (or with another archive) would
                    # play its games without them in silence: never used (docs/az.md 24)
                    s.no_forks = ev.get('forks')
                want_r = _resign_of(self.R.cfg) if self.mode == 'selfplay' else ''
                if want_r and ev.get('resign') != want_r:
                    # a checkout without self-play resignation (or with other settings) would play its games out
                    # in silence: never used (docs/az.md 27)
                    s.no_resign = ev.get('resign')
                want_m = _merge_of(self.R.cfg) if self.mode == 'selfplay' else ''
                if want_m and ev.get('merge') != want_m:
                    # a checkout without self-play information-set merging would search unmerged in silence:
                    # never used (docs/az.md 30)
                    s.no_merge = ev.get('merge')
                if self.fails or s.k == 0:
                    self.log(f'worker {s.k} up (pid {ev.get("pid")} on {ev.get("host")}, mirror {self.rdir})')
            elif kind == 'status':
                s.status = ev
                if ev.get('backoff'):
                    if s.bo_since is None:
                        s.bo_since = _now()
                else:
                    s.bo_since = None
                    s.gave_back = False
            elif kind == 'fin':                        # single-game: games finished there, held until accepted
                s.fin += int(ev.get('n', 0))
                self._accept_remote(s)
            elif kind == 'unaccepted':                 # an accept it could not take (should not happen)
                n = int(ev.get('n', 0))
                s.acc -= n
                s.acc_ph -= n
                self._give_back(s, ev.get('iter'), n)
            elif kind == 'shard':
                if self.cur is not None and ev.get('iter') == self.accepting == self.cur['it']:
                    self.cur['pending'] += 1
                    self.cur['announced'] += 1
                    s.ship_ph += int(ev.get('games', 0))
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
        with self.mu:
            self._refresh()
            if self.cur is not None and want is not None and self.cur['it'] == want:
                self.cur['capn'] += 1
                for n, v in self.capv.items():
                    self.cur['capsum'][n] = self.cur['capsum'].get(n, 0) + v
                    self.cur['capmax'][n] = max(self.cur['capmax'].get(n, 0), v)
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
            if getattr(s, 'no_league', False):
                self.next_try[k] = now + max(self.retry_s, 600.0)
                self.fail_log(f'worker {k} REFUSED: its checkout has no self-play league (the run plays one, '
                              f'selfplay_league {self.R.cfg["selfplay_league"]}); killed, retried in 10 minutes')
                s.kill()
                continue
            if hasattr(s, 'no_forks'):
                self.next_try[k] = now + max(self.retry_s, 600.0)
                self.fail_log(f'worker {k} REFUSED: its checkout plays branched games / archive starts as '
                              f'{s.no_forks!r}, the run as {_forks_of(self.R.forks)!r} (an older checkout has none: '
                              f'docs/az.md 24); killed, retried in 10 minutes')
                s.kill()
                continue
            if hasattr(s, 'no_resign'):
                self.next_try[k] = now + max(self.retry_s, 600.0)
                self.fail_log(f'worker {k} REFUSED: its checkout plays self-play resignation as {s.no_resign!r}, the run '
                              f'as {_resign_of(self.R.cfg)!r} (an older checkout has none: docs/az.md 27); killed, '
                              f'retried in 10 minutes')
                s.kill()
                continue
            if hasattr(s, 'no_merge'):
                self.next_try[k] = now + max(self.retry_s, 600.0)
                self.fail_log(f'worker {k} REFUSED: its checkout plays self-play information-set merging as '
                              f'{s.no_merge!r}, the run as {_merge_of(self.R.cfg)!r} (an older checkout has none: '
                              f'docs/az.md 30); killed, retried in 10 minutes')
                s.kill()
                continue
            if hasattr(s, 'no_belief'):
                self.next_try[k] = now + max(self.retry_s, 600.0)
                self.fail_log(f'worker {k} REFUSED: its checkout plays the root belief as {s.no_belief!r}, the run as '
                              f'{_belief_of(self.R.cfg)!r} (an older checkout has none: docs/az.md 18.5); killed, retried '
                              f'in 10 minutes')
                s.kill()
                continue
            if now - s.pinged > PING_S:
                s.pinged = now
                s.send({'cmd': 'ping'})
            if s.hello is not None and self.mode == 'selfplay':
                with self.mu:
                    cap = self.capv.get(s.name, 0)
                    if cap != s.sent_cap:
                        s.send({'cmd': 'cap', 'n': cap})
                        s.sent_cap = cap
            go = want is not None and s.hello is not None and s.phase != want
            if go and self.mode == 'selfplay':
                with self.mu:                          # the quota met already: nothing to play for
                    go = self.q is not None and self.q['it'] == want and self.q['open']
            if go and self._ship_model(want):
                with self.mu:
                    if self.want == want and s.alive():
                        msg = {'cmd': 'phase', 'iter': want, 'model': f'models/iter_{want - 1:04d}.pt'}
                        if self.R.league is not None and self.mode == 'selfplay':
                            msg['league'] = self.R.league.files(want)   # {key: models/iter_j.pt} (shipped)
                        s.send(msg)
                        s.phase = want
                        s.acc_ph = s.ship_ph = 0
                        if self.mode == 'selfplay':
                            self._accept_remote(s)     # the games it held from the last phase count first
            if self.mode == 'matches':
                self._giveback(s)
                self._timeouts(s)
                if want is not None and s.phase == want:
                    self._assign(s)
                    with self.mu:
                        cap = self.capv.get(s.name, 0)
                        if cap > 0 and not s.backing_off():
                            s.util[0] += min(1.0, len(s.assigned) / cap)
                            s.util[1] += 1

    def _prepare_mirror(self):
        if self.mirror_ready:
            return
        self.t.mkdirs(self.rdir + '/models', self.rdir + '/data')
        cfg = dict(self.R.cfg, **self.overrides)
        cfg.update(eval_remote='', selfplay_remote='', resolver='', resolver_cfg={}, resolver_init='')
        fk = getattr(self.R, 'forks', None)
        if self.mode == 'selfplay' and fk is not None and fk.archive:
            # the archive starts' file (zero/forks.py) into the mirror: the workers read it there
            from .forks import MIRROR_NAME
            self.t.put(fk.archive['path'], self.rdir + '/' + MIRROR_NAME)
            cfg['selfplay_archive'] = MIRROR_NAME
        if 'shard_rows' not in self.overrides:
            cfg['shard_rows'] = min(int(cfg['shard_rows']), SHARD_ROWS)
        try:
            test = json.loads(os.environ.get('ZERO_RSP_TEST') or '{}')
        except ValueError:
            test = {}
        spj = {'mode': self.mode, 'locks': self.locks, 'nice': self.nice, 'backoff': self.backoff, 'run_id': self.run_id,
               'workdir': self.workdir, 'mirror': self.mirror, 'watchdog_s': WATCHDOG_S, 'linger': self.linger,
               'test': test}
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
        """Checkpoint want - 1 into the mirror (once per phase); with the self-play league
        (zero/league.py) also the past checkpoints of the phase's slots (every phase: a worker deletes
        the checkpoints older than its phase - 3 that its phase does not name)."""
        from .run import model_path
        if self.model_shipped == want - 1:
            return True
        if _now() < self.ship_retry:
            return False
        names = [want - 1]
        if self.R.league is not None and self.mode == 'selfplay':
            names += sorted(set(self.R.league.plan(want)['slots']) - {want - 1})
        for j in names:
            try:
                self.t.put(model_path(self.R.run, j), f'{self.rdir}/models/iter_{j:04d}.pt')
            except Exception as e:                     # noqa: BLE001 - retried
                self.ship_retry = _now() + 30
                self.log(f'checkpoint {j} not shipped: {type(e).__name__}: {str(e)[-300:]}; retry in 30s')
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
        lost = 0
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
            # single-game: games it had accepted whose samples never reached a shard are not this
            # iteration's after all - the quota needs them again (while this machine still plays)
            q = self.q
            if q is not None and s.phase == q['it']:
                lost = max(0, s.acc_ph - s.ship_ph)
                if lost and self._give_back(s, q['it'], lost):
                    self.cur['lost'] += lost
            # the next session starts from a mirror prepared again (it may be what failed: removed,
            # half-written) and gets its checkpoint again (6 MB)
            self.mirror_ready = False
            self.model_shipped = None
            self._refresh()
        if requeue and self.co is not None:
            self.co.requeue(requeue)
            with self.mu:
                if self.cur is not None:
                    self.cur['requeued'] += len(requeue)
        if self.closed:
            return
        err = s.error or {}
        tail = ((f'; {len(requeue)} episodes requeued here' if requeue else '')
                + (f'; {lost} accepted games lost with it (the quota needs them again)' if lost else '')
                + f'; restart in {max(0.0, self.next_try[s.k] - _now()):.0f}s')
        if err.get('kind') == 'encoding':
            # the worker's build refused the run's checkpoint: a network of another encoding than its
            # checkout's (docs/az.md 10.4) - this machine plays on alone; in full the first times,
            # then every 10 minutes (the worker said hello first: fail_log's streak restarts)
            self.refusals += 1
            if self.refusals <= 3 or _now() - self.t_refused > 600:
                self.t_refused = _now()
                self.log(f"worker {s.k} REFUSED the run's checkpoint (its checkout {self.spec.get('repo')} on "
                         f"{self.spec.get('host')} is not the run's encoding?): {why}{tail}"
                         + (f' ({self.refusals} refusals; logged every 10 minutes now)' if self.refusals > 3 else ''))
            return
        self.fail_log(f'worker {s.k} ended (exit {rc}, after {lived:.0f}s): {str(why)[-500:]}' + tail)
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
        if co is None or s.hello is None:
            return
        with self.mu:
            cap = self.capv.get(s.name, 0)
        room = cap - len(s.assigned)
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

    def _giveback(self, s):
        """A worker backing off for `selfplay_remote_giveback` s gives its outstanding episodes back
        (requeued here; whichever copy finishes second is dropped)."""
        if self.co is None or s.bo_since is None or s.gave_back or _now() - s.bo_since < self.giveback_s:
            return
        with self.mu:
            back = [(tag, spec) for tag, (_t, spec) in s.assigned.items()]
            s.assigned.clear()
            s.gave_back = True
        if back:
            self.co.requeue(back)
            with self.mu:
                if self.cur is not None:
                    self.cur['requeued_backoff'] += len(back)
            self.log(f'worker {s.k}: backing off for {_now() - s.bo_since:.0f}s ({s.status.get("backoff")}): its '
                     f'{len(back)} outstanding episodes requeued here (a second result is dropped)')

    def overdue_s(self, s):
        """How long (its running time) an episode may stay outstanding on worker `s`: at most
        `selfplay_remote_timeout`; with `selfplay_remote_overdue` k, k x its expected latency (its cap
        over its measured throughput), never less than OVERDUE_MIN_S."""
        lim = self.timeout_s
        with self.mu:
            c = self.shares.c.get(s.name)
            cap = self.capv.get(s.name, 0)
        if self.overdue > 0 and c is not None and c['rate'] and cap > 0:
            lim = min(lim, max(self.overdue_min, self.overdue * cap / c['rate']))
        return lim

    def _timeouts(self, s):
        if self.co is None or 'active_s' not in s.status:
            return
        act = float(s.status['active_s'])
        lim = self.overdue_s(s)
        with self.mu:
            late = [(tag, spec) for tag, (t, spec) in s.assigned.items() if act - t > lim]
            for tag, _ in late:
                del s.assigned[tag]
        if late:
            self.co.requeue(late)
            with self.mu:
                if self.cur is not None:
                    self.cur['requeued_timeout'] += len(late)
            self.log(f'worker {s.k}: {len(late)} episodes not finished within {lim:g}s of its running '
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
            res, rows, net = 'failed', 0, None
            try:
                with self.mu:
                    go = self.accepting == it
                if go:
                    os.makedirs(d, exist_ok=True)
                    if not self.t.get(ev['path'], tmp):
                        raise RuntimeError(f'copy of {ev["path"]} failed')
                    res, rows, net = self._check(tmp, it, name)
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
                        c['games'] += int(ev.get('games', 0))
                        lag_hist(net, it, c['stale'])
                    elif res in ('rejected', 'quarantined'):
                        c[res] += 1
                    elif res == 'failed':
                        c['failed'] += 1
                self.mu.notify_all()

    def _check(self, path, it, name):
        """A fetched shard -> ('ok', rows, net) | ('rejected', 0, None) (another game's or encoder's:
        logged) | ('quarantined', 0, None) (non-finite: moved to <run>/debug/, the run fails at end())."""
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
            return 'rejected', 0, None
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
            return 'quarantined', 0, None
        if getattr(self.R, 'aux', None) and 'aux' not in sh:
            # (accepted - its policy / value rows train as any - but its auxiliary rows are masked:
            # the remote's checkout predates the run's auxiliary targets, docs/az.md 15)
            self.log(f'shard {name} of iteration {it} has NO auxiliary targets (the run trains {self.R.aux}): '
                     f"is the remote's checkout the run's?")
        return 'ok', int(meta.get('rows', len(sh['player']))), sh.get('net')


def staleness(local, remote):
    """The iteration's staleness block: the histograms of this machine's rows, the remote's and all
    of them (zero/workqueue.py: rows by checkpoint lag)."""
    from .workqueue import hist_summary
    allh = merge_hist(local, remote)
    return {'local': merge_hist(local), 'remote': merge_hist(remote), 'all': allh, 'summary': hist_summary(allh)}


# ================================================================== the remote side

class _BgWriter:
    """A ShardWriter on a thread of its own: the driver's thread hands it sample dicts and never
    waits for a shard's compression. `games`: the games added since its last shard (a shard holds
    whole games: the writer writes its whole buffer)."""

    def __init__(self, writer):
        self.w = writer
        self.q = queue.Queue()
        self.error = None
        self.games = 0
        self.th = threading.Thread(target=self._run, name='zero-rsp-writer', daemon=True)
        self.th.start()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            if self.error is not None:
                continue
            s, g = item
            try:
                self.games += g
                if s is not None:
                    self.w.add(s)
            except BaseException as e:                 # noqa: BLE001 - raised on the driver's thread
                self.error = e

    def add(self, s, games=0):
        if s is not None and not len(s['player']):
            s = None
        if s is not None or games:
            self.q.put((s, games))

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
        self.linger = bool(self.sp.get('linger'))
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
        # self-play: the games this worker may have in flight (the run's `cap`; its slots until it
        # says), the games it finished that wait for the run's `accept` (with their samples), the
        # setups it queued in its driver and the games it took back out
        self.cap = None
        self.held = deque()
        self.enq = 0
        self.taken = 0
        self.held_samples = []                     # co-training, lingering: rows sent, samples for the next phase
        # the self-play league (zero/league.py): its table's kinds, its slots' evaluators and checkpoints
        from . import league as L
        self.lg_on = self.mode == 'selfplay' and L.enabled(self.cfg)
        self.lg_kinds = None
        self.lg_evs = {}
        self.lg_slots = None
        from .games import spec_json
        self.spec = spec_json(self.cfg.get('spec'))
        # shard_writer_thread (docs/az.md 31, the run's config): the records taken as they are (take_records) and
        # converted, written and announced on a bounded writer thread (zero/shardthread.py); off: take_samples and
        # the unbounded _BgWriter, as before
        self.threaded = bool(self.cfg.get('shard_writer_thread'))

    def _forks(self):
        """This worker's branched games and archive starts (zero/forks.py: the mirror's config, its
        archive.json), None when off."""
        if self.mode != 'selfplay':
            return None
        if not hasattr(self, 'fk'):
            from .forks import setup
            self.fk = setup(self.rdir, self.cfg)
        return self.fk

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
            self.enq += 1
            if self.drv is None:
                self.backlog.append((msg['tag'], msg['spec']))
            else:
                self.drv.enqueue(msg['tag'], msg['spec'])
        elif cmd == 'cap':
            self.cap = max(0, int(msg.get('n', 0)))
        elif cmd == 'accept':
            self._accept(msg, ph)
        elif cmd == 'end':
            if ph is not None:
                if ph['it'] == msg.get('iter') and not ph['closed']:
                    if self.linger and not self.closing:
                        self._close(ph)            # the phase's shards are done; play on into the next
                    else:
                        self._end_phase(ph)
            else:                                  # nothing running: answer at once
                self.emit('end', iter=msg.get('iter'), games=0, samples=0, shards=0, bytes=0, leaves=0, seconds=0.0,
                          active_s=0.0, backoff_s=0.0, in_flight=0, accepted=0, unshipped=0)
        elif cmd in ('quit', 'eof'):
            self.closing = True
            if ph is not None:
                self._end_phase(ph)
        elif cmd in ('fetched', 'drop'):
            self._rm(msg.get('path'))
        elif cmd == 'phase':
            if ph is not None:                     # the run moved on while this phase ends (or lingers)
                self.stash = msg
                self._end_phase(ph)
            else:
                self._phase(msg)

    def _controls(self, ph):
        while self.stash is None:                  # after a stashed `phase`, the rest is the new phase's
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

    def _load(self, it, model, league=None):
        from .infer import NetEvaluator
        from .net import load
        from .run import Run
        from . import league as L
        c = self.cfg
        # (the encoding guard: a checkpoint of another encoding than this checkout's raises
        # EncodingMismatch, reported to the run as an 'encoding' error; docs/az.md 10.4. The
        # tests' 'load_game': a checkout whose game feeds another encoding)
        m, _ = load(os.path.join(self.rdir, model), c['infer_device'], game=self.test.get('load_game') or c['game'])
        if self.drv is None:
            R = Run(self.rdir, c)
            self.evr = NetEvaluator(m, c['infer_device'], workers=c.get('infer_workers', 1))
            seat = R.nn_seat('m', c['sims'], selfplay=True)
            seed = int(c['seed']) * 7919 + it + 7_777_777 * (self.k + 1)
            # both modes play queued setups only: the run's episodes (co-training), or a setup of the
            # run's spec for every game the worker may start under its cap (self-play)
            table = [[seat] * c['players'][0]] if self.mode == 'matches' else [[seat] * p for p in c['players']]
            if self.lg_on:
                table, self.lg_kinds = L.rolling_table(c, seat)    # the run's league table
            self.drv = R.driver(table, seed=seed, record=True, rolling=True, net=it - 1, queue_only=True)
            for tag, spec in self.backlog:
                self.drv.enqueue(tag, spec)
            self.backlog = []
            self.info = R.info
        else:
            self.evr.set_weights(m.state_dict())
        keep = set()
        if self.lg_on:
            # the phase's league slots: {key: models/iter_j.pt}, shipped by the run (zero/league.py)
            if not league or set(league) != {L.key(t) for t in range(int(c.get('selfplay_league_slots', 1)))}:
                raise SystemExit(f'remote self-play: the run plays a league but its phase names the slots {league!r}')
            for k, rel in league.items():
                self.lg_evs[k] = L.load_into(self.lg_evs.get(k), os.path.join(self.rdir, *rel.split('/')), c)
                keep.add(os.path.basename(rel))
            self.lg_slots = [int(os.path.basename(league[L.key(t)])[5:9]) for t in range(len(league))]
        for p in glob.glob(os.path.join(self.rdir, 'models', 'iter_*.pt')):
            try:
                if int(os.path.basename(p)[5:9]) < it - 3 and os.path.basename(p) not in keep:
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

    def _on_shard(self, it, box):
        def cb(path, rows, nbytes):
            bw = box.get('bw')
            games = 0
            if bw is not None:
                games, bw.games = bw.games, 0
            self.shards_out += 1
            if self.test.get('corrupt') and self.shards_out == 1:
                _corrupt(path)                     # the tests: a bad file the arrival check must catch
            if self.test.get('delay_shard_s'):
                time.sleep(float(self.test['delay_shard_s']))
            self.emit('shard', iter=it, path=path, rows=rows, bytes=nbytes, games=games)
            if self.test.get('die_after_shards') and self.shards_out >= int(self.test['die_after_shards']):
                self.note('test: dying after a shard')
                os._exit(9)
        return cb

    def _end_phase(self, ph):
        if not ph['ending']:
            ph['ending'] = True
            self.drv.end_phase()

    def outstanding(self):
        """Setups queued in the driver or playing (queue_only: every setup comes back as one row, and a
        branch of its game as a second one, zero/forks.py: `taken` counts the first)."""
        return self.enq - self.taken

    def _topup(self, ph):
        """Self-play: queue a setup of the run's spec for every game this worker may still start -
        under its cap; between phases (lingering) also counting the games it holds."""
        if ph['ending']:
            return
        cap = self.cap if self.cap is not None else int(self.cfg['conc'])
        room = cap - self.outstanding() - (len(self.held) if ph['closed'] else 0)
        for _ in range(max(0, room)):
            self.drv.enqueue(f'{self.sid}:{self.enq}', self.spec)
            self.enq += 1

    def _accept(self, msg, ph):
        """The run counts `n` more of the held games for its iteration: their samples go into this
        phase's shards (oldest first)."""
        n = int(msg.get('n', 0))
        if ph is None or ph['it'] != msg.get('iter') or ph['closed']:
            self.note(f'accept {msg} with no open phase {msg.get("iter")} here: given back')
            self.emit('unaccepted', iter=msg.get('iter'), n=n)    # the run counts them again
            return
        n = min(n, len(self.held))
        for _ in range(n):
            row, sd = self.held.popleft()
            ph['bw'].add(sd, 1)
            ph['accepted'] += 1
            if self.lg_on:
                from .league import tally
                tally([row], ph['league'])
            if ph.get('forks') is not None:
                from .forks import tally as ftally
                ftally([row], ph['forks'], self._forks().kinds())
            if ph.get('resign') is not None:
                from .resign import tally as rtally
                rtally([row], ph['resign'])
        die = int(self.test.get('die_after_accepts') or 0)
        if die and ph['accepted'] >= die:
            self.note('test: dying with accepted games in no shard yet')
            os._exit(9)

    def _finished(self, ph, rows, s):
        """Games the driver finished: self-play - held for the run's accept (`fin`); co-training - the
        rows to the run at once, the samples into this phase's shards (lingering: the next one's)."""
        if self.lg_on:
            from .league import League
            League.attribute(rows, self.lg_kinds, self.lg_slots)     # (the slots' checkpoints now)
        games = split_records(rows, s) if self.threaded else split_games(rows, s)
        from .forks import setups
        self.taken += setups(rows)                 # (a branch is its game's second row: zero/forks.py)
        if ph['closed']:
            ph['linger_games'] += len(games)
        else:
            ph['games'] += len(games)
        if self.mode == 'matches':
            die_rows = int(self.test.get('die_after_rows') or 0)
            for row, sd in games:
                self.emit('row', row=row)
                self.rows_out += 1
                if sd is not None:
                    ph['samples'] += sample_rows(sd)
                    if ph['closed']:
                        self.held_samples.append(sd)
                    else:
                        ph['bw'].add(sd, 1)
                elif not ph['closed']:
                    ph['bw'].add(None, 1)
                if die_rows and self.rows_out >= die_rows:
                    self.note('test: dying after a row')
                    os._exit(9)
        else:
            self.held.extend(games)
            ph['samples'] += sum(sample_rows(sd) for _, sd in games if sd is not None)
            if games:
                self.emit('fin', iter=ph['it'], n=len(games))

    def _status(self, ph, force=False, backoff=None):
        now = _now()
        if not force and now - self.t_status < (self.status_s if ph is not None else IDLE_STATUS_S):
            return
        self.t_status = now
        if ph is None:
            self.emit('status', phase=None, active_s=round(self.active_s, 1), in_flight=0, games=0, held=len(self.held),
                      outstanding=self.outstanding(), cap=self.cap)
            return
        bo = ph['backoff'] + (now - ph['bo_t0'] if ph['bo_t0'] else 0.0)
        st = self.drv.stats()
        self.emit('status', phase=ph['it'], games=ph['games'], samples=ph['samples'], leaves=st['leaves'] - ph['leaves0'],
                  in_flight=self.drv.in_flight()['games'], queued=self.drv.queued, held=len(self.held),
                  outstanding=self.outstanding(), cap=self.cap, linger=ph['closed'],
                  active_s=round(self.active_s + (now - ph['t0']) - bo, 1), phase_s=round(now - ph['t0'], 1),
                  run_s=round(now - ph['t0'] - bo, 1),
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
        self._status(ph, force=True)

    def _close(self, ph):
        """The phase's accounting ends: its last shard, the `end` line with its counters. Without
        linger the driver's phase has ended too; lingering, it plays on (into the next phase)."""
        if ph['closed']:
            return
        if ph.get('bg'):
            self._background(False)
            ph['backoff'] += _now() - ph['bo_t0']
            ph['bo_t0'] = 0.0
            ph['bg'] = None
        bw = ph['bw']
        w = bw.close()
        unshipped = bw.games                       # games added after the last shard: no samples (no shard)
        st = self.drv.stats()
        now = _now()
        secs = now - ph['t0']
        bo = ph['backoff'] + (now - ph['bo_t0'] if ph['bo_t0'] else 0.0)
        ph['closed'] = True
        ph['t_close'] = now
        ph['leaves_close'] = st['leaves']
        extra = {'league': ph['league']} if self.lg_on else {}
        if ph.get('forks') is not None:
            extra['forks'] = ph['forks']
        if ph.get('resign') is not None:
            extra['resign'] = ph['resign']
        if self.threaded:
            extra['shard_writer'] = dict(bw.metrics(), samples_s=round(ph['clock'].s, 3))
        self.emit('end', iter=ph['it'], games=ph['games'], samples=ph['samples'], shards=w['shards'], bytes=w['bytes'],
                  leaves=st['leaves'] - ph['leaves0'], seconds=round(secs, 2), active_s=round(secs - bo, 2),
                  backoff_s=round(bo, 2), in_flight=self.drv.in_flight()['games'], queued=self.drv.queued,
                  accepted=ph['accepted'], unshipped=unshipped, held=len(self.held),
                  linger=ph.get('linger_prev'), **extra)
        self.note(f'phase {ph["it"]}: {ph["games"]} games ({ph["accepted"]} accepted, {len(self.held)} held), '
                  f'{ph["samples"]} samples in {w["shards"]} shards ({w["bytes"] / 1e6:.1f} MB), '
                  f'{(st["leaves"] - ph["leaves0"]) / max(secs - bo, 1.0):.0f} leaves/s running, {secs:.0f}s '
                  f'({bo:.0f}s backing off)' + ('; lingering' if self.linger and not ph['ending'] else ''))

    def _phase(self, msg):
        from .data import ShardWriter
        from .infer import drive
        it = int(msg['iter'])
        self._load(it, msg['model'], msg.get('league'))
        self._cleanup()
        c = self.cfg
        drv = self.drv
        meta = {'game': c['game'], 'features': self.info['num_features'], 'actions': self.info['num_actions'],
                'iter': it, 'source': 'remote', 'worker': self.k, 'session': self.sid}
        box = {}
        writer = ShardWriter(os.path.join(self.rdir, 'data', f'iter_{it:04d}'), self.prefix, rows=int(c['shard_rows']),
                             meta=meta, bad_dir=os.path.join(self.rdir, 'debug'), compress=COMPRESS,
                             on_write=self._on_shard(it, box))
        if self.threaded:
            from .shardthread import MB, ThreadedShardWriter
            if not hasattr(drv, 'take_records'):
                raise SystemExit('shard_writer_thread: this engine module has no SearchDriver.take_records (built from '
                                 'an older zero-core): rebuild it')
            bw = ThreadedShardWriter(writer, max_bytes=float(c.get('shard_writer_queue_mb', 4096)) * MB,
                                     name='zero-rsp-writer')
        else:
            bw = _BgWriter(writer)
        box['bw'] = bw
        st0 = drv.stats()
        drv.resume(BIG, it - 1)
        ph = {'it': it, 'games': 0, 'samples': 0, 't0': _now(), 'backoff': 0.0, 'bo_t0': 0.0, 'ending': False,
              't_lock': 0.0, 'leaves0': st0['leaves'], 'bw': bw, 'closed': False, 'accepted': 0,
              'linger_games': 0, 'linger_prev': getattr(self, 'linger_prev', None), 'league': {},
              'clock': SampleClock() if self.threaded else contextlib.nullcontext()}
        if self.mode == 'selfplay' and self._forks() is not None:
            from .forks import new_tally
            ph['forks'] = new_tally()
        from . import resign as RS
        if self.mode == 'selfplay' and RS.enabled(c):
            ph['resign'] = RS.new_tally()
        evs = {'m': self.evr, **self.lg_evs} if self.lg_on else {'m': self.evr}
        self.linger_prev = None
        for sd in self.held_samples:               # co-training: rows finished while lingering, sent already
            bw.add(sd, 1)
        self.held_samples = []
        self._busy(True)
        slow = float(self.test.get('slow_s') or 0)

        def pump(lb=None):
            self._controls(ph)
            self._watchdog()
            if self.closing:
                self._end_phase(ph)
            with ph['clock']:
                rows = drv.take_results()
                s = drv.take_records() if self.threaded else drv.take_samples()
                if rows:
                    self._finished(ph, rows, s)
                elif s is not None:
                    raise RuntimeError('remote self-play: samples without finished games')
            if not ph['closed']:
                bw.check()
            if slow:
                time.sleep(slow)
            if not ph['ending']:
                self._backoff(ph)
            if self.mode == 'selfplay':
                self._topup(ph)
            self._status(ph)

        while True:
            pump()
            drive(drv, evs, on_batch=pump)
            if ph['ending']:
                break
            msg = self._get(0.5) if self.stash is None else None   # queue_only, nothing to play: wait
            if msg is not None:
                self._control(msg, ph)
            self._watchdog()
            if self.closing:
                self._end_phase(ph)
            self._status(ph)
        pump()
        if ph['closed']:                           # lingered: those games are the next phase's
            now = _now()
            self.linger_prev = {'games': ph['linger_games'], 'seconds': round(now - ph['t_close'], 1)}
        self._close(ph)
        now = _now()
        self.active_s += (now - ph['t0']) - ph['backoff']
        self._busy(False)

    def main(self):
        from .finite import NonFinite
        from .net import EncodingMismatch
        from .runctl import EXIT_REFUSED
        try:
            self._take_slot()
            self._setup()
            threading.Thread(target=self._stdin, name='zero-rsp-stdin', daemon=True).start()
            self.note(f'started: mode {self.mode}, mirror {self.rdir}, conc {self.cfg["conc"]}, threads '
                      f'{self.cfg["threads"]}, groups {self.cfg["groups"]}, infer_workers {self.cfg.get("infer_workers", 1)}, '
                      f'torch threads {self.cfg.get("torch_threads", 1)}, nice {self.sp.get("nice", 0)}'
                      + (', lingering between phases' if self.linger else ''))
            # (the tests: an older checkout, without the league / the root's belief)
            self.emit('hello', pid=os.getpid(), host=socket.gethostname(), sid=self.sid, conc=self.cfg['conc'],
                      mode=self.mode, **({} if self.test.get('no_league') else {'league': 1}),
                      **({} if self.test.get('no_belief') else {'belief': _belief_of(self.cfg)}),
                      **({} if self.test.get('no_forks') else {'forks': _forks_of(self._forks())}),
                      **({} if self.test.get('no_resign') else {'resign': _resign_of(self.cfg)}),
                      **({} if self.test.get('no_merge') else {'merge': _merge_of(self.cfg)}))
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
        except EncodingMismatch as e:
            self.emit('error', kind='encoding', msg=str(e)[-3000:])
            self.note(f'REFUSED the checkpoint: {e}')
            return EXIT_REFUSED
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
