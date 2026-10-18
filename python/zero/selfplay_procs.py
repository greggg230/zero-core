"""Self-play in several local processes: run setting `selfplay_procs` (docs/az.md 7.1).

One self-play process tops out at what its single Python drive loop can feed the GPU: Kingdom
Builder Zero, ~51k network leaves/s whatever conc / groups / threads (the loop waits ~55% on
CUDA graph replay and ~30% on the next search batch). Independent processes, each with its own
CUDA context, `NetEvaluator` and Rust `SearchDriver`, overlap those waits: two processes with
half the games each reached 64-65k leaves/s together (+25-28%). Two drive loops as threads of
one process deadlocked (both in `GraphRunner.result`'s `Event.synchronize`): processes only.

The run process coordinates (`Pool`): it starts N worker processes (multiprocessing, spawn
context) at its first self-play phase and keeps them until the run process ends. Worker k
plays games quota_k = games / N (the remainder to the first workers) with `conc` / N games in
flight (the remainder likewise), max(1, threads / N) search threads and the run's groups, from
the iteration's checkpoint; it streams its samples into `data/iter_NNNN/r{k:02d}_SSS.npz`
(rolling; classic self-play: `c{k:02d}_SSS.npz`), so the workers' shard names never collide
and the training window reads them all (`data.shard_paths`: every shard of the directory).
Rolling self-play: each worker keeps its own persistent rolling driver across iterations (the
driver's `resume(quota_k, net)`; new weights via `set_weights`), exactly as the single process
does. Classic self-play: the run samples the games' player counts as it does with one process
and gives each worker its slice.

Seeds: worker k's driver seed is the single process's seed (rolling: seed * 7919 + the
iteration that creates the driver; classic: seed * 1000003 + iteration) + k * 2**40.

Protocol (one duplex Pipe per worker; on each side only the main thread uses it):

    run -> worker   ('play', job)       play iteration job['iter'] (checkpoint iter - 1)
                    ('quit',)           exit now (also mid-phase: the iteration is abandoned)
    worker -> run   ('ready', info)     started (pid, process creation time)
                    ('progress', games) every ~2 s while playing
                    ('done', out, raw)  the phase's metrics (Run.play_selfplay) + raw sums
                    ('error', text)     an exception (traceback); the worker exits

Failure is loud: a worker's error, its process ending without its result, a closed pipe or
`ZERO_SELFPLAY_STALL_S` (600) seconds without a message from a playing worker raise
`WorkerError` in the run process (the run aborts; the supervisor restarts it); a worker whose
games or shards do not add up is an error too, never a partial iteration. PAUSE: the run
process notices it (`Run.check`) while waiting for its workers, tells them to quit and exits
75; a worker checks its pipe after every batch. No orphans: a worker exits when its pipe
closes or the run process dies (a watch thread on the parent's process handle), so a killed
run process does not leave workers behind (Windows does not kill children with their parent).
Each worker's stdout / stderr (Python tracebacks, faulthandler dumps) go to
`<run>/selfplay_w{k}.txt`; their pids are in run_state.json (`selfplay_workers`).

Spawn re-imports the parent's main module in each worker: `python -m <pkg>.az.run` is fine; a
script that drives `Run` itself needs the usual `if __name__ == '__main__':` guard. The worker
activates the run's game package by importing its `run_module` (which registers it).
"""
import faulthandler
import importlib
import multiprocessing as mp
import multiprocessing.connection as mpc
import os
import sys
import threading
import time
import traceback

from . import package, runctl
from .runctl import Paused

READY_S = 600.0                  # a worker's start: imports, CUDA context, the engine
SEED_STRIDE = 2 ** 40            # worker k's driver seed = the single process's + k * this
EXIT_ORPHAN = 3                  # a worker whose run process died


class WorkerError(RuntimeError):
    """A self-play worker failed (error, death, silence or a short result)."""


def share(total, n, k):
    """Worker k's part of `total` split over n workers (the remainder to the first ones)."""
    return total // n + (1 if k < total % n else 0)


def worker_cfg(cfg, n, k):
    """Worker k's run settings: its share of the games in flight and of the search threads."""
    return dict(cfg, selfplay_procs=1, conc=max(1, share(int(cfg['conc']), n, k)),
                threads=max(1, int(cfg['threads']) // n))


def worker_seed(base, k):
    return int(base) + k * SEED_STRIDE


def stall_s():
    return float(os.environ.get('ZERO_SELFPLAY_STALL_S', '600'))


# ------------------------------------------------------------------ the worker process

class _Link:
    """The worker's end of its pipe as its Run's pause object: `check()` (after every batch)
    raises Paused once the run process says 'quit' or the pipe breaks."""

    def __init__(self, conn):
        self.conn = conn
        self.t = 0.0
        self.why = None

    def check(self, where=''):
        now = time.monotonic()
        if now - self.t < 0.02:
            return
        self.t = now
        try:
            if self.conn.poll():
                msg = self.conn.recv()
                if msg[0] != 'quit':
                    raise RuntimeError(f'self-play worker: unexpected message {msg[0]!r} while playing')
                self.why = 'stopped by the run process'
        except (EOFError, OSError):
            self.why = 'the pipe to the run process closed (the run process is gone)'
        if self.why:
            raise Paused(f'{where}: {self.why}')


def _watch_parent(poll=1.0):
    """Exit the moment the run process is gone (its process handle / fd is signalled)."""
    parent = mp.parent_process()
    if parent is None:
        return

    def loop():
        while True:
            time.sleep(poll)
            if not parent.is_alive():
                print(time.strftime('%H:%M:%S ') + 'the run process is gone: exiting', flush=True)
                os._exit(EXIT_ORPHAN)
    threading.Thread(target=loop, daemon=True, name='zero-parent-watch').start()


def _activate(pkg_name, run_module):
    """Make the run's game package active in this process (spawn: a fresh interpreter)."""
    if package.active().name != pkg_name:
        importlib.import_module(run_module)        # a game's run module imports (registers) its package
    if package.active().name != pkg_name:
        raise RuntimeError(f'self-play worker: game package {pkg_name!r} not registered by {run_module}')


def _exit(code):
    for s in (sys.stdout, sys.stderr):
        try:
            s.flush()
        except Exception:                          # noqa: BLE001
            pass
    os._exit(code)


def worker_main(conn, init):
    """Entry point of self-play worker init['k'] (a spawned process)."""
    k = init['k']
    try:
        out = open(init['log'], 'a', encoding='utf-8', buffering=1)
        sys.stdout = sys.stderr = out
        faulthandler.enable(out)
    except OSError:
        pass
    print(time.strftime('%Y-%m-%d %H:%M:%S ') + f'self-play worker {k}/{init["n"]} pid {os.getpid()} starting', flush=True)
    _watch_parent()
    try:
        _activate(init['package'], init['run_module'])
        import torch
        from . import games as G, run as zrun
        cfg = init['cfg']
        print(f'package {package.active().name} (engine {G.engine().__name__}), game {cfg["game"]}, conc {cfg["conc"]}, '
              f'threads {cfg["threads"]}, groups {cfg["groups"]}, infer_device {cfg["infer_device"]}', flush=True)
        torch.set_num_threads(max(1, int(cfg['torch_threads'])))
        zrun.below_normal()
        link = _Link(conn)
        R = zrun.Run(init['run'], cfg, pause=link)
        conn.send(('ready', {'pid': os.getpid(), 'started': runctl._proc_started()}))
        while True:
            msg = conn.recv()
            if msg[0] == 'quit':
                print(time.strftime('%H:%M:%S ') + 'quit (the run process asked)', flush=True)
                _exit(0)
            if msg[0] != 'play':
                raise RuntimeError(f'unexpected message {msg[0]!r}')
            job = msg[1]
            try:
                res = R.play_selfplay(job['iter'], job['dir'], job['games'], job['seed'], job['prefix'],
                                      counts=job.get('counts'), progress=lambda g: conn.send(('progress', g)),
                                      meta_extra={'worker': k, 'procs': init['n']})
            except Paused as e:
                print(time.strftime('%H:%M:%S ') + f'iter {job["iter"]} {e}: exiting', flush=True)
                _exit(0)
            conn.send(('done',) + tuple(res))
    except (EOFError, BrokenPipeError, ConnectionResetError):
        print(time.strftime('%H:%M:%S ') + 'the pipe to the run process closed (the run process is gone): exiting',
              flush=True)
        _exit(0)
    except BaseException:                          # noqa: BLE001 - every failure goes to the run process
        tb = traceback.format_exc()
        print(tb, flush=True)
        try:
            conn.send(('error', tb))
        except Exception:                          # noqa: BLE001
            pass
        _exit(1)


# ------------------------------------------------------------------ the run process's side

class _Worker:
    def __init__(self, k, proc, conn):
        self.k, self.proc, self.conn = k, proc, conn
        self.info = {}
        self.last = time.time()                    # last message


class Pool:
    """N self-play worker processes of a run (`Run.selfplay` with selfplay_procs > 1)."""

    def __init__(self, R):
        self.R = R
        self.n = int(R.cfg['selfplay_procs'])
        self.workers = []

    # -------------------------------------------------------------- lifecycle
    def start(self):
        R, n = self.R, self.n
        ctx = mp.get_context('spawn')
        pkg = package.active()
        run = os.path.abspath(R.run)
        t0 = time.time()
        for k in range(n):
            here, there = ctx.Pipe(duplex=True)
            init = {'k': k, 'n': n, 'run': run, 'cfg': worker_cfg(R.cfg, n, k), 'package': pkg.name,
                    'run_module': pkg.run_module, 'log': os.path.join(run, f'selfplay_w{k}.txt')}
            p = ctx.Process(target=worker_main, args=(there, init), name=f'zero-selfplay-w{k}', daemon=True)
            p.start()
            there.close()
            self.workers.append(_Worker(k, p, here))
        info = self._gather('ready', 'self-play workers starting', READY_S)
        for w in self.workers:
            w.info = info[w.k][0]
        secs = time.time() - t0
        runctl.write_state(R.run, 'run', selfplay_workers=[w.info for w in self.workers])
        R.log(f'self-play workers: {n} processes (pids {", ".join(str(w.info["pid"]) for w in self.workers)}) '
              f'ready in {secs:.1f}s; per worker conc {[worker_cfg(R.cfg, n, k)["conc"] for k in range(n)]}, '
              f'threads {worker_cfg(R.cfg, n, 0)["threads"]}, groups {R.cfg["groups"]}')
        return secs

    def close(self, timeout=15.0):
        """Tell every worker to quit; wait up to `timeout` s, then terminate stragglers."""
        ws, self.workers = self.workers, []
        for w in ws:
            try:
                w.conn.send(('quit',))
            except (OSError, ValueError):
                pass
        deadline = time.time() + timeout
        for w in ws:
            w.proc.join(max(0.0, deadline - time.time()))
            if w.proc.is_alive():
                w.proc.terminate()
                w.proc.join(5)
            try:
                w.conn.close()
            except OSError:
                pass

    # -------------------------------------------------------------- messages
    def _gather(self, final, what, timeout, on_progress=None):
        """Messages until every worker has sent `final` -> {k: payload tuple}. PAUSE raises Paused
        (Run.check); a worker's error, death, closed pipe or silence raises WorkerError."""
        got = {}
        t0 = time.time()
        for w in self.workers:
            w.last = t0
        while len(got) < len(self.workers):
            self.R.check(what)
            pend = [w for w in self.workers if w.k not in got]
            ready = mpc.wait([w.conn for w in pend] + [w.proc.sentinel for w in pend], timeout=0.25)
            now = time.time()
            for w in pend:
                if w.conn not in ready and w.proc.sentinel not in ready:
                    if now - w.last > timeout:
                        raise WorkerError(f'self-play worker {w.k} sent nothing for {now - w.last:.0f}s ({what})')
                    continue
                for msg in self._recv(w):
                    w.last = now
                    if msg is None:
                        w.proc.join(2)
                        raise WorkerError(f'self-play worker {w.k} (pid {w.proc.pid}) is gone ({what}): its pipe '
                                          f'closed, exit code {w.proc.exitcode}; see selfplay_w{w.k}.txt')
                    if msg[0] == 'error':
                        raise WorkerError(f'self-play worker {w.k} (pid {w.proc.pid}) failed ({what}):\n{msg[1]}')
                    if msg[0] == 'progress':
                        if on_progress is not None:
                            on_progress(w.k, msg[1])
                    elif msg[0] == final:
                        got[w.k] = msg[1:]
                    else:
                        raise WorkerError(f'self-play worker {w.k}: unexpected message {msg[0]!r} ({what})')
                if w.k not in got and w.proc.sentinel in ready and not w.proc.is_alive():
                    raise WorkerError(f'self-play worker {w.k} (pid {w.proc.pid}) exited with code {w.proc.exitcode} '
                                      f'without its result ({what}); see selfplay_w{w.k}.txt')
        return got

    @staticmethod
    def _recv(w):
        """Every message waiting from worker w; a closed pipe ends the list with None."""
        out = []
        try:
            while w.conn.poll():
                out.append(w.conn.recv())
        except (EOFError, OSError):
            out.append(None)
        return out

    # -------------------------------------------------------------- one self-play phase
    def selfplay(self, it, d):
        """Self-play of iteration `it` into the (empty) directory `d` -> the merged metrics."""
        from .run import _write_json
        import numpy as np
        R, c, n = self.R, self.R.cfg, self.n
        startup = self.start() if not self.workers else None
        rolling = bool(c['rolling'])
        quotas = [share(int(c['games']), n, k) for k in range(n)]
        counts = None
        if not rolling:                            # the player counts one process would sample
            rng = np.random.default_rng(c['seed'] * 1_000_003 + it)
            counts = [int(p) for p in rng.choice(c['players'], size=c['games'])]
        base = c['seed'] * 7919 + it if rolling else c['seed'] * 1_000_003 + it
        os.makedirs(d, exist_ok=True)
        t0 = time.time()
        pos = 0
        for w, q in zip(self.workers, quotas):
            job = {'iter': it, 'dir': os.path.abspath(d), 'games': q, 'seed': worker_seed(base, w.k),
                   'prefix': f'{"r" if rolling else "c"}{w.k:02d}',
                   'counts': counts[pos:pos + q] if counts is not None else None}
            pos += q
            try:
                w.conn.send(('play', job))
            except (OSError, ValueError) as e:
                w.proc.join(2)
                raise WorkerError(f'self-play worker {w.k} (pid {w.proc.pid}) is gone before iteration {it} '
                                  f'({type(e).__name__}: {e}; exit code {w.proc.exitcode}); see selfplay_w{w.k}.txt') from e
        prog = os.path.join(R.run, 'progress.json')
        games = [0] * n
        last = [0.0]

        def on_progress(k, g):
            games[k] = g
            if time.time() - last[0] > 2:
                last[0] = time.time()
                _write_json(prog, {'iter': it, 'phase': 'selfplay', 'games': sum(games), 'target': c['games'],
                                   'procs': n, 'time': time.time()})
        got = self._gather('done', f'self-play iteration {it}', stall_s(), on_progress)
        secs = time.time() - t0
        _write_json(prog, {'iter': it, 'phase': 'train', 'time': time.time()})
        parts = [got[k] for k in range(n)]
        check_parts(parts, quotas, rolling, d)
        out = merge(parts, secs, c.get('policy_target', 'visits'), rolling,
                    pids=[w.info.get('pid') for w in self.workers])
        if startup is not None:
            out['startup_s'] = startup
        return out


def check_parts(parts, quotas, rolling, d):
    """Every worker played its quota (rolling: at least it) and its shards are on disk."""
    from .data import shard_paths
    for k, ((o, raw), q) in enumerate(zip(parts, quotas)):
        if o['games'] < q or (not rolling and o['games'] != q):
            raise WorkerError(f'self-play worker {k}: {o["games"]} games, quota {q}')
        prefix = f'{"r" if rolling else "c"}{k:02d}_'
        have = [p for p in shard_paths(d) if os.path.basename(p).startswith(prefix)]
        if len(have) != o['shards']:
            raise WorkerError(f'self-play worker {k}: {o["shards"]} shards reported, {len(have)} in {d}')
    if sum(o['samples'] for o, _ in parts) == 0:
        raise WorkerError(f'self-play: {sum(o["games"] for o, _ in parts)} games and no samples')


def merge(parts, secs, kind='visits', rolling=False, pids=None):
    """The workers' (metrics, raw sums) -> one metrics dict with Run.play_selfplay's keys:
    counts summed, per-game / per-sample means weighted, throughput over the phase's wall time
    `secs` (infer_s: summed over the workers); + procs and a per-worker summary."""
    from .data import TargetStats
    outs = [p[0] for p in parts]
    raws = [p[1] for p in parts]
    G = sum(o['games'] for o in outs)
    fin = sum(r['fin'] for r in raws)
    leaves = sum(o['leaves'] for o in outs)
    ts = TargetStats()
    for r in raws:
        n, opts, h, mx = r['target']
        ts.n += n
        ts.opts += opts
        ts.h += h
        ts.mx += mx
    calls = sum(r['ev_calls'] for r in raws)
    out = {'games': G, 'unfinished': sum(o['unfinished'] for o in outs),
           'samples': sum(o['samples'] for o in outs), 'shards': sum(o['shards'] for o in outs),
           'bytes': sum(o['bytes'] for o in outs),
           'decisions_per_game': sum(r['decisions'] for r in raws) / max(1, G),
           'steps_per_game': sum(r['steps'] for r in raws) / max(1, G),
           'rounds': sum(r['rounds'] for r in raws) / max(1, fin),
           'avg_score': sum(r['score'] for r in raws) / max(1, fin),
           'leaves': leaves, 'leaves_per_s': leaves / max(secs, 1e-9),
           'games_per_h': G / max(secs, 1e-9) * 3600, 'seconds': secs,
           'infer_s': sum(o['infer_s'] for o in outs), 'avg_batch': sum(r['ev_leaves'] for r in raws) / max(1, calls),
           'target': ts.summary(kind)}
    if rolling:
        out['rolling'] = {key: sum(o['rolling'][key] for o in outs)
                          for key in ('in_flight', 'restarts', 'restart_sims', 'dropped_leaves')}
    out['procs'] = len(parts)
    out['workers'] = [{'k': k, 'pid': (pids or [None] * len(outs))[k], 'games': o['games'], 'samples': o['samples'],
                       'leaves': o['leaves'], 'leaves_per_s': round(o['leaves_per_s'], 1),
                       'seconds': round(o['seconds'], 2), 'infer_s': round(o['infer_s'], 2),
                       'avg_batch': round(o['avg_batch'], 1)} for k, o in enumerate(outs)]
    return out
