"""Remote evaluation (docs/az.md 5.2): a run's checkpoint evaluations - the yardstick and random
games, the Elo matches against older checkpoints - played on another machine while the run keeps
training (run setting `eval_remote` = a name in the game package's `remotes`).

Local side (`RemoteEval`, a thread of the run process): each evaluated iteration is shipped to
the remote's mirror of the run (`<workdir>/<run name>`: config.json with the remote's overrides,
the checkpoints the evaluation needs), evaluated there with `python -m <pkg>.az.remote eval
<mirror> <iter>`, and its result appended to the run's `evals.jsonl` ({iter, eval_yard,
eval_random, elo, time, host, seconds}); the mirror's elo/ files and models/index.json are copied
back (the remote owns the run's Elo while `eval_remote` is on). The dashboards merge evals.jsonl
into the iterations' rows. Evaluations missing after a restart are queued again; a failure (the
remote asleep or offline) is retried every `RETRY_S`.

Remote side (`main`): the run's own `Run.evaluate` + `Run.elo_update` on the mirror, one
evaluation at a time per remote workdir (a lock directory), the result printed as one line
`ZERO_REMOTE_RESULT {json}`.

Co-training runs (zero/cotrain.py, docs/az.md 11.5): the resolver's own evaluations go to the
remote named by `resolver_cfg['eval_remote']`, exactly as a single-game run's: `RemoteEval` of the
resolver's `Run` (`<run>/resolver/`: its evals.jsonl, elo/, models/index.json), the mirror
`<workdir>/<run name>.resolver` (a sibling of the single-game mirrors: the same lock and
overrides.json), the resolver's config.json there (a single-game run's settings: the remote side
needs nothing else). The primary's evaluation stays local, except its events played by the
resolver's network (`eval_yard_net`) with the run setting `draft_eval_net_remote` (docs/az.md 11.7):
`RemoteNetEval`, the mirror `<workdir>/<run name>` laid out as the run (both networks' checkpoints),
`python -m <pkg>.az.remote evalnet <mirror> <iter>` there (`CoRun.eval_net` under the same lock), the
result in the run's `evals.jsonl` ({iter, eval_yard_net, ...}).

A remote: {host: 'user@host' (or 'local': this machine, the tests), repo: the game repo there
(same code as the run's checkout), python, workdir, env: {...}, overrides: run settings for the
remote ({'infer_device': 'mps', 'threads': 12}), selfplay: {...} (optional: remote self-play's
workdir, worker processes, niceness, locks to back off for, its own overrides; zero/remoteplay.py,
docs/az.md 5.3)}.

`python -m <pkg>.az.remote selfplay MIRROR WORKER`: a remote self-play worker (zero/remoteplay.py).
`python -m <pkg>.az.remote evalnet MIRROR ITER`: a co-training run's eval_yard_net (above).
"""
import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from . import elo, package

RETRY_S = 300
RESULT = 'ZERO_REMOTE_RESULT '
LOCK_STALE_S = 4 * 3600


def remote_spec(name, setting='eval_remote'):
    """The game package's remote `name` as a dict (a Mapping such as MTG Zero's RunCommitRemote
    resolves here: its checks run now). 'local': this machine (the tests). `setting`: the run
    setting that names it (the error message)."""
    pkg = package.active()
    if name == 'local':
        return dict({'host': 'local', 'repo': pkg.root, 'python': sys.executable,
                     'workdir': os.environ.get('ZERO_REMOTE_WORKDIR', os.path.join(tempfile.gettempdir(), 'zero-remote')),
                     'env': {}, 'overrides': {}}, **pkg.remotes.get('local', {}))
    if name not in pkg.remotes:
        raise SystemExit(f'{setting} {name!r}: not in the game package remotes ({sorted(pkg.remotes)})')
    spec = dict({'env': {}, 'overrides': {}}, **pkg.remotes[name])
    # another evaluation workdir on the remote (an end-to-end test beside a live run's workdir: its
    # own mirrors and its own .eval.lock); an absolute path there (Git Bash rewrites /Users/... in a
    # Windows process's environment: MSYS2_ENV_CONV_EXCL=ZERO_EVAL_WORKDIR)
    wd = os.environ.get('ZERO_EVAL_WORKDIR')
    if wd:
        if not wd.startswith('/'):
            raise SystemExit(f'ZERO_EVAL_WORKDIR {wd!r}: not an absolute path on the remote {name!r} (a Windows '
                             f'shell converted it? Git Bash: MSYS2_ENV_CONV_EXCL=ZERO_EVAL_WORKDIR)')
        spec['workdir'] = wd.rstrip('/')
    return spec


class Transport:
    """Files and commands on a remote ('local': this machine)."""

    def __init__(self, spec):
        self.s = spec
        self.local = spec['host'] == 'local'

    def _ssh(self, sh, timeout):
        return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', self.s['host'], sh],
                              capture_output=True, text=True, timeout=timeout)

    def mkdirs(self, *paths):
        if self.local:
            for p in paths:
                os.makedirs(p, exist_ok=True)
            return
        r = self._ssh('mkdir -p ' + ' '.join(shlex.quote(p) for p in paths), 60)
        if r.returncode:
            raise RuntimeError(f'mkdir on {self.s["host"]}: {r.stderr.strip()[-300:]}')

    def put(self, src, dst):
        """Copy a file to the remote under a temporary name, then rename it: an interrupted copy
        (the run stopped mid-transfer) never leaves a partial file under the real name, where it
        would count as shipped and every evaluation that loads it would fail."""
        if self.local:
            shutil.copyfile(src, dst + '.tmp')
            os.replace(dst + '.tmp', dst)
            return
        r = subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', src, f'{self.s["host"]}:{dst}.tmp'],
                           capture_output=True, text=True, timeout=600)
        if r.returncode:
            raise RuntimeError(f'scp {src} -> {self.s["host"]}: {r.stderr.strip()[-300:]}')
        r = self._ssh(f'mv -f {shlex.quote(dst + ".tmp")} {shlex.quote(dst)}', 60)
        if r.returncode:
            raise RuntimeError(f'mv {dst} on {self.s["host"]}: {r.stderr.strip()[-300:]}')

    def get(self, src, dst):
        """Copy a remote file here (False if it does not exist)."""
        if self.local:
            if not os.path.exists(src):
                return False
            shutil.copyfile(src, dst + '.tmp')
            os.replace(dst + '.tmp', dst)
            return True
        r = subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', f'{self.s["host"]}:{src}', dst + '.tmp'],
                           capture_output=True, text=True, timeout=600)
        if r.returncode:
            return False
        os.replace(dst + '.tmp', dst)
        return True

    def listdir(self, d):
        if self.local:
            return os.listdir(d) if os.path.isdir(d) else []
        r = self._ssh(f'ls {shlex.quote(d)} 2>/dev/null', 60)
        return r.stdout.split()

    def run_module(self, module, args, timeout):
        env = {k: str(v) for k, v in self.s.get('env', {}).items()}
        if self.local:
            env.setdefault('PYTHONPATH', os.pathsep.join(p for p in sys.path if p))
            return subprocess.run([self.s['python'], '-m', module, *args], cwd=self.s['repo'], capture_output=True,
                                  text=True, timeout=timeout, env=dict(os.environ, **env))
        exports = ' '.join(f'{k}={shlex.quote(v)}' for k, v in env.items())
        sh = (f'cd {shlex.quote(self.s["repo"])} && {exports} {shlex.quote(self.s["python"])} -m {module} '
              + ' '.join(shlex.quote(a) for a in args))
        return self._ssh(sh, timeout)


class RemoteEval:
    """The run's evaluations on the remote `name` (see the module docstring). `mirror`: the
    mirror's directory name in the remote's workdir (default the run directory's name; a
    co-training run's resolver: '<run name>.resolver')."""

    def __init__(self, R, name, mirror=None):
        self.R = R
        self.name = name
        self.spec = remote_spec(name)
        self.t = Transport(self.spec)
        run_name = mirror or os.path.basename(os.path.normpath(R.run))
        self.rdir = self.spec['workdir'].rstrip('/') + '/' + run_name
        self.module = f'{package.active().name}.az.remote' if package.active().name != 'zero' else 'zero.remote'
        self.q = queue.Queue()
        self.queued = set()
        self.shipped = None
        self.lock = threading.Lock()
        self.th = threading.Thread(target=self._loop, name='zero-remote-eval', daemon=True)
        self.th.start()

    # ---- queue
    def done_iters(self):
        """Iterations evaluated already: here before eval_remote (metrics rows with an evaluation)
        or remotely (evals.jsonl)."""
        from .run import read_metrics
        out = {m['iter'] for m in read_metrics(self.R.run) if m.get('eval_yard')}
        p = os.path.join(self.R.run, 'evals.jsonl')
        if not os.path.exists(p):
            return out
        with open(p, encoding='utf-8') as f:
            for line in f:
                try:
                    out.add(json.loads(line)['iter'])
                except (ValueError, KeyError):
                    pass
        return out

    def enqueue(self, it):
        with self.lock:
            if it in self.queued:
                return
            self.queued.add(it)
        self.q.put(it)

    def backfill(self, latest):
        """Queue the evaluation iterations up to `latest` without a result (after a restart)."""
        from .run import model_path
        e = max(1, int(self.R.cfg['eval_every']))
        done = self.done_iters()
        for k in sorted({0, *range(e, latest + 1, e)}):
            if k not in done and os.path.exists(model_path(self.R.run, k)):
                self.enqueue(k)

    def drain(self, timeout=6 * 3600, check=None):
        """Wait until every queued evaluation has a result (the run finished its iterations).
        `check`: called every poll (a co-training run's pause check: PAUSE ends the wait, the
        restart queues the missing evaluations again)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                if not self.queued:
                    return True
            if check is not None:
                check()
            time.sleep(2)
        return False

    # ---- the worker
    def _loop(self):
        while True:
            it = self.q.get()
            while True:
                try:
                    self._evaluate(it)
                    break
                except Exception as ex:                # noqa: BLE001 (logged, retried)
                    self.R.log(f'remote eval {self.name} iter {it}: {type(ex).__name__}: {str(ex)[-400:]}; retry in {RETRY_S}s')
                    self.shipped = None
                    time.sleep(RETRY_S)
            with self.lock:
                self.queued.discard(it)

    def _ship(self, it):
        from .run import model_path
        c = self.R.cfg
        if self.shipped is None:
            self.t.mkdirs(self.rdir + '/models', self.rdir + '/elo')
            self.shipped = set(self.t.listdir(self.rdir + '/models'))
            if 'results.json' not in self.t.listdir(self.rdir + '/elo'):
                # a new mirror: the run's Elo so far (results incl. the evaluations logged here
                # before eval_remote) and its metrics, so the remote continues them
                for src, dst in ((os.path.join(self.R.run, 'elo', 'results.json'), '/elo/results.json'),
                                 (os.path.join(self.R.run, 'metrics.jsonl'), '/metrics.jsonl')):
                    if os.path.exists(src):
                        self.t.put(src, self.rdir + dst)
        cfg = dict(c, eval_remote='', **self.spec.get('overrides', {}))
        tmp = os.path.join(self.R.run, '.remote_config.json')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=1)
        self.t.put(tmp, self.rdir + '/config.json')
        e = max(1, int(c['eval_every']))
        # every checkpoint Run.elo_update may play up to `it` (each evaluated one and its offsets)
        need = sorted({j for k in [0, *range(e, it + 1, e)] for j in [k] + [k - off * e for off in elo.OFFSETS]
                       if 0 <= j <= it})
        for k in need:
            fn = elo.ckpt_name(k) + '.pt'
            if fn not in self.shipped and os.path.exists(model_path(self.R.run, k)):
                self.t.put(model_path(self.R.run, k), self.rdir + '/models/' + fn)
                self.shipped.add(fn)

    def _evaluate(self, it):
        t0 = time.time()
        self._ship(it)
        r = self.t.run_module(self.module, ['eval', self.rdir, str(it)], timeout=4 * 3600)
        lines = [ln for ln in (r.stdout or '').splitlines() if ln.startswith(RESULT)]
        if r.returncode or not lines:
            raise RuntimeError(f'exit {r.returncode}: {(r.stderr or r.stdout or "").strip()[-600:]}')
        m = json.loads(lines[-1][len(RESULT):])
        row = {'iter': it, 'time': time.time(), 'host': self.name, 'seconds': round(time.time() - t0, 1)}
        for k in ('eval_yard', 'eval_random', 'elo'):
            if k in m:
                row[k] = m[k]
        os.makedirs(os.path.join(self.R.run, 'elo'), exist_ok=True)
        for fn in ('results.json', 'ratings.json'):
            self.t.get(self.rdir + '/elo/' + fn, os.path.join(self.R.run, 'elo', fn))
        self.t.get(self.rdir + '/models/index.json', os.path.join(self.R.run, 'models', 'index.json'))
        with open(os.path.join(self.R.run, 'evals.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        ey = row.get('eval_yard')
        if ey:
            self.R.log(f'iter {it} vs {ey.get("opp", self.R.yard)} (remote {self.name}, {row["seconds"]:.0f}s): {ey}')
        if row.get('elo'):
            self.R.log(f'iter {it} elo {row["elo"]["elo"]:.0f} +-{1.96 * row["elo"]["se"]:.0f} (remote {self.name})')


class RemoteNetEval(RemoteEval):
    """A co-training run's primary evaluated with its episodes played by the resolver's network
    (`eval_yard_net`, docs/az.md 11.7) on the remote `name` (run setting `draft_eval_net_remote`):
    the mirror `<workdir>/<run name>` is laid out as the run (config.json: the run's settings with
    the remote's overrides on both networks' settings and every remote setting off; models/ and
    resolver/models/: both networks' checkpoints of each such iteration); `python -m <pkg>.az.remote
    evalnet <mirror> <iter>` plays the events there (`CoRun.eval_net`, the code that plays them here)
    under the workdir's evaluation lock; the result is appended to the run's `evals.jsonl` ({iter,
    eval_yard_net, time, host, seconds, remote_s}), which the dashboard merges into the primary's
    rows. The queue, the thread, the retries (an unreachable remote never blocks training) and the
    end-of-run drain are RemoteEval's. `co`: the run's `CoRun`."""

    def __init__(self, co, name):
        self.co = co
        super().__init__(co.P, name)

    def done_iters(self):
        """Iterations whose eval_yard_net exists: played here (metrics rows) or remotely (evals.jsonl)."""
        from .run import read_metrics
        out = {m['iter'] for m in read_metrics(self.R.run) if m.get('eval_yard_net')}
        p = os.path.join(self.R.run, 'evals.jsonl')
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    if e.get('eval_yard_net') and 'iter' in e:
                        out.add(e['iter'])
        return out

    def backfill(self, latest):
        """Queue the iterations up to `latest` whose eval_yard_net is due and missing (a restart)."""
        from .run import model_path
        done = self.done_iters()
        for k in range(0, latest + 1):
            if self.co.eval_net_due(k) and k not in done and os.path.exists(model_path(self.R.run, k)) \
                    and os.path.exists(model_path(self.co.R.run, k)):
                self.enqueue(k)

    def _ship(self, it):
        from .run import DEFAULTS, model_path
        if self.shipped is None:
            self.t.mkdirs(self.rdir + '/models', self.rdir + '/resolver/models')
            self.shipped = ({'models/' + f for f in self.t.listdir(self.rdir + '/models')}
                            | {'resolver/models/' + f for f in self.t.listdir(self.rdir + '/resolver/models')})
        c = self.R.cfg
        ov = dict(self.spec.get('overrides', {}))
        off = {'eval_remote': '', 'selfplay_remote': ''}
        rc = dict(c.get('resolver_cfg') or {}, **off, **{k: v for k, v in ov.items() if k in DEFAULTS})
        cfg = dict(c, **off, draft_eval_net_remote='', **ov)
        cfg['resolver_cfg'] = rc
        tmp = os.path.join(self.R.run, '.remote_evalnet_config.json')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=1)
        self.t.put(tmp, self.rdir + '/config.json')
        fn = elo.ckpt_name(it) + '.pt'
        for sub, run in (('models/', self.R.run), ('resolver/models/', self.co.R.run)):
            if sub + fn not in self.shipped:
                self.t.put(model_path(run, it), self.rdir + '/' + sub + fn)
                self.shipped.add(sub + fn)

    def _evaluate(self, it):
        t0 = time.time()
        self._ship(it)
        r = self.t.run_module(self.module, ['evalnet', self.rdir, str(it)], timeout=4 * 3600)
        lines = [ln for ln in (r.stdout or '').splitlines() if ln.startswith(RESULT)]
        if r.returncode or not lines:
            raise RuntimeError(f'exit {r.returncode}: {(r.stderr or r.stdout or "").strip()[-600:]}')
        m = json.loads(lines[-1][len(RESULT):])
        if not m.get('eval_yard_net'):
            raise RuntimeError(f'no eval_yard_net in the result: {lines[-1][:300]}')
        row = {'iter': it, 'eval_yard_net': m['eval_yard_net'], 'time': time.time(), 'host': self.name,
               'seconds': round(time.time() - t0, 1), 'remote_s': m.get('seconds')}
        with open(os.path.join(self.R.run, 'evals.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
        e = row['eval_yard_net']
        self.R.log(f'iter {it} vs {e.get("opp", self.R.yard)} (episodes by the resolver net; remote {self.name}, '
                   f'{row["seconds"]:.0f}s): {e}')


# ------------------------------------------------------------------ the remote side

class _Lock:
    """One remote evaluation at a time per workdir (a lock directory; stale after LOCK_STALE_S)."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        while True:
            try:
                os.mkdir(self.path)
                with open(os.path.join(self.path, 'owner'), 'w') as f:
                    f.write(f'{os.getpid()} {time.time()}')
                return self
            except FileExistsError:
                try:
                    with open(os.path.join(self.path, 'owner')) as f:
                        pid = int(f.read().split()[0])
                    alive = _alive(pid)
                except (OSError, ValueError, IndexError):
                    alive = True                       # being created: wait
                try:
                    if not alive or time.time() - os.path.getmtime(self.path) > LOCK_STALE_S:
                        shutil.rmtree(self.path, ignore_errors=True)
                        continue
                except OSError:
                    continue
                time.sleep(5)

    def __exit__(self, *a):
        shutil.rmtree(self.path, ignore_errors=True)


def _alive(pid):
    """Is process `pid` alive? (On Windows `os.kill(pid, 0)` would TERMINATE it: psutil there.)"""
    if os.name == 'nt':
        try:
            import psutil
            return psutil.pid_exists(pid)
        except ImportError:
            return True                            # unknown: treat as alive (a lock then waits)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


def evaluate_mirror(rdir, it):
    """Run.evaluate + Run.elo_update of iteration `it` on the mirror `rdir`. -> the metrics row."""
    from .run import Run, append_metrics
    with open(os.path.join(rdir, 'config.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    # the remote machine's own settings for every run it evaluates (<workdir>/overrides.json, e.g.
    # {"groups": 1, "elo_games": 50}: fewer, larger inference batches suit a slower GPU)
    cfg.update(_overrides(rdir))
    R = Run(rdir, cfg)
    m = {'iter': it}
    R.evaluate(it, m)
    ratings, best = R.elo_update(it, m)
    me = ratings.get(elo.ckpt_name(it))
    if me:
        m['elo'] = {'elo': round(me[0], 1), 'se': round(me[1], 1)}
    m['time'] = time.time()
    append_metrics(rdir, m)
    return m


def _overrides(rdir):
    """The remote machine's own settings for every run it evaluates (<workdir>/overrides.json)."""
    ov = os.path.join(os.path.dirname(os.path.normpath(rdir)), 'overrides.json')
    if not os.path.exists(ov):
        return {}
    with open(ov, encoding='utf-8') as f:
        return json.load(f)


def evaluate_net_mirror(rdir, it):
    """A co-training run's `eval_yard_net` of iteration `it` on the mirror `rdir` (RemoteNetEval's:
    the run's settings, both networks' checkpoints): `CoRun.eval_net`, the run's own code. The
    remote's overrides.json applies to both networks' settings. -> {iter, eval_yard_net, seconds,
    time} (also appended to the mirror's metrics.jsonl)."""
    from .cotrain import CO_KEYS, CoRun
    from .run import DEFAULTS, append_metrics
    with open(os.path.join(rdir, 'config.json'), encoding='utf-8') as f:
        cfg = json.load(f)
    ov = _overrides(rdir)
    cfg.update(ov)
    cfg['resolver_cfg'] = dict(cfg.get('resolver_cfg') or {},
                               **{k: v for k, v in ov.items() if k in DEFAULTS and k not in CO_KEYS})
    co = CoRun(rdir, cfg)
    t0 = time.time()
    m = {'iter': it, 'eval_yard_net': co.eval_net(it)}
    m['seconds'] = round(time.time() - t0, 1)
    m['time'] = time.time()
    append_metrics(rdir, m)
    return m


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['selfplay']:
        # remote self-play (docs/az.md 5.3): a worker serving the run over its stdin / stdout
        from .remoteplay import serve
        return serve(argv[1:])
    if len(argv) != 3 or argv[0] not in ('eval', 'evalnet'):
        raise SystemExit('usage: python -m <pkg>.az.remote eval MIRROR_DIR ITER | evalnet MIRROR_DIR ITER | '
                         'selfplay MIRROR_DIR WORKER')
    rdir, it = argv[1], int(argv[2])
    with _Lock(os.path.join(os.path.dirname(os.path.normpath(rdir)), '.eval.lock')):
        m = (evaluate_net_mirror if argv[0] == 'evalnet' else evaluate_mirror)(rdir, it)
    print(RESULT + json.dumps(m), flush=True)


if __name__ == '__main__':
    main()
