"""The farm's torch worker (docs/az.md 78; zero-farm's per-machine torch runtime, PROTOCOL rev 27 draft): an `aw/1`
self-play worker (zero-farm PROTOCOL.md 5, 10.5) that plays through the trainer's own inference path instead of
ONNX Runtime - for the owner's machines whose GPU the native worker's graph leaves idle (an RTX 5090 reached ~0.4x of
the trainer's torch self-play with it).

    <torch worker build>/worker_main.py serve        (the build's entry: zero-farm's zf-pyshim.exe runs
                                                      <runtime>/python/python.exe -I -X utf8 -u worker_main.py serve)

**What it plays.** A job is the native worker's (zero-core crates/worker `job.rs`, the Python remote worker's
driver): the games `gid_base .. gid_base + games` of a rolling, queue_only, recording driver of the run's self-play
seat (`zero.run.Run.nn_seat('m', sims, selfplay=True)`, a table per `players` entry) with the job's `seed`, numbered
from `gid_base` (`SearchDriver.set_first_gid`), sparse batches, the job's search threads and its `conc` / `groups`
when given - built by `Run.driver` from the job's settings over `zero.run.DEFAULTS`, so the same games, seeds, ids and
per-seat settings as the native worker. Its network batches go to `zero.infer.NetEvaluator` exactly as the trainer's
local self-play builds it (`Run.evaluator`: `infer_precision`, `infer_token_chunks`, `infer_workers`; a token network's
CUDA graphs, `TokenGraphRunner`, as ZERO_TOKEN_GRAPHS=1 gives the trainer), with the run's inference settings carried
by the gen's `torch-weights/1` file (zero.torch_weights: they are trainer-class settings, never in a template), driven
by `infer._drive_pipelined`'s loop (up to ZERO_INFLIGHT batches launched, submitted in launch order; the evaluator's
`launch` / `result`) with a yield gate before each new batch. Every finished game is written as its record
(`SearchDriver.set_trace`, `take_game_records`: zero_search::record::game_json, the text the native worker writes)
into gzip'd JSON-lines files of the native worker's format (`RecordWriter`: header, one line per game, files cut at
`shard_rows` rows, records/2 with branched games), so the pool driver replays them unchanged; each row keeps the
network that searched it (`net`).

**aw/1.** `hello` first, before torch is imported (aw 1, build_id = sha256(manifest.json next to worker_main.py)[:24],
game, kinds ['selfplay'], zero_onnx [], models ['torch-weights/1'], features ['rolling'], pid); a `job` (kind
selfplay; `model` = the gen's torch variant - anything else is refused, fatal; `reference` = the gen's reference set;
optional `gen`, `sha256` of the model file, `conc`, `groups`, `threads`, `prefix`, `output` 'records', `rolling`,
`flush_s`, `gpu` {luid}); the self-test (`selftest`, provider 'torch-cuda': the reference rows through the very
evaluator it plays with, against the stored torch answers, within the precision's tolerance, and twice bit for bit);
`fin`, `records`, `status` (Stats; `provider`, plus `vram_mb`: what torch holds on the GPU), `low`, `end` (reason
done | finish | quit, stats, unfinished {gids}); `checkpoint` (a rolling job builds and self-tests the new network on a
thread while the games go on, then the driver's phase change; a classic job switches at once), `extend`, `yield`
(no batch started, the ones in flight answered, the evaluator and its CUDA graphs dropped, torch's cache emptied:
`yielded`), `resume` (rebuilt; a failure is fatal), `finish`, `quit`, `ping`. stdout carries the protocol's JSON
lines only (fd 1 is pointed at stderr: anything a library prints goes to the log).

Tests: tests/test_farm_torch_worker.py (a CPU network and the toy: the protocol, records replayed by `replay_rows`,
checkpoint / yield / finish / quit, the rows equal to the trainer's local driver's at fp32 batch 1).
"""
import collections
import gzip
import hashlib
import json
import os
import queue
import sys
import threading
import time

AW = 1
KINDS = ['selfplay']
FEATURES = ['rolling']
BIG = 1 << 62                        # a rolling driver's quota (remoteplay.BIG)
STATUS_S = 10.0
FLUSH_S = 120.0                      # a rolling job's default flush_s (PROTOCOL 10.5)
YIELD_WAIT_S = 1.8                   # `yielded` within 2 s of `yield` (PROTOCOL 5)
PROVIDER = {'cuda': 'torch-cuda', 'cpu': 'torch-cpu'}
# The self-test's tolerances per precision (largest |difference| on any number, largest mean per item): float32 on the
# GPU moves torch's CPU answers by float32 rounding; bfloat16 autocast (the trainer's infer_precision 'bf16') by
# bfloat16's - measured on MTG Zero's meta3 network (gen 1334, 256 reference rows, an RTX 5090, CUDA graphs): largest
# 0.078 (a value), means 1.4e-3 (priors), 9e-4 (value), 2.9e-3 (margin), 6e-6 (aux): the bounds leave ~2.5x / ~3.5x
# room (docs/az.md 78); TF32 between the two. A broken evaluator misses by far more (a shifted reference: every number).
TOL = {'fp32': (1e-3, None), 'tf32': (5e-2, 3e-3), 'bf16': (0.2, 1e-2)}


class Fatal(Exception):
    """An error that ends the job: the `error` event (fatal) and exit 1."""


# ------------------------------------------------------------------------------------------------ output
class Out:
    """aw/1 lines on the protocol's stdout (one writer at a time), the log on stderr."""

    def __init__(self, f):
        self.f = f
        self.mu = threading.Lock()

    def event(self, ev, **kw):
        line = json.dumps(dict(ev=ev, **kw), separators=(',', ':'), allow_nan=False, default=_jsonable)
        with self.mu:
            self.f.write(line + '\n')
            self.f.flush()

    def log(self, msg):
        print(time.strftime('%H:%M:%S ') + str(msg), file=sys.stderr, flush=True)


def _jsonable(x):
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return x.item()
    except ImportError:
        pass
    raise TypeError(f'{type(x).__name__} is not JSON serializable')


def protocol_stdout():
    """The real stdout as the protocol's channel, and fd 1 / sys.stdout pointed at stderr from now on."""
    fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(fd, 'w', encoding='utf-8', newline='\n', buffering=1)


def build_id(build_dir):
    """sha256(manifest.json next to worker_main.py)[:24] (PROTOCOL 2), else env ZERO_FARM_BUILD_ID, else 'unsigned-dev'
    (the native worker's rule)."""
    try:
        with open(os.path.join(build_dir, 'manifest.json'), 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:24]
    except OSError:
        return os.environ.get('ZERO_FARM_BUILD_ID') or 'unsigned-dev'


def ranges(gids):
    """Half-open [a, b) ranges covering `gids`."""
    out = []
    for g in sorted(set(int(x) for x in gids)):
        if out and out[-1][1] == g:
            out[-1][1] = g + 1
        else:
            out.append([g, g + 1])
    return out


def missing_ranges(lo, hi, done):
    d = sorted({int(g) for g in done if lo <= g < hi})
    out, at = [], lo
    for g in d:
        if g > at:
            out.append([at, g])
        at = g + 1
    if at < hi:
        out.append([at, hi])
    return out


def rss_mb():
    """The process's working set (Windows) / resident set, MB (0 if unknown)."""
    try:
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t),
                            ('WorkingSetSize', ctypes.c_size_t), ('rest', ctypes.c_size_t * 6)]
            p = PMC()
            p.cb = ctypes.sizeof(PMC)
            k32 = ctypes.WinDLL('kernel32')
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            ps = ctypes.WinDLL('psapi')
            ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            if ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(p), p.cb):
                return int(p.WorkingSetSize >> 20), int(p.PeakWorkingSetSize >> 20)
            return 0, 0
        with open('/proc/self/status', encoding='ascii') as f:
            for ln in f:
                if ln.startswith('VmRSS:'):
                    return int(ln.split()[1]) // 1024, 0
    except Exception:  # noqa: BLE001 - a statistic
        pass
    return 0, 0


# ------------------------------------------------------------------------------------------------ records
class RecordWriter:
    """The native worker's records files (crates/worker records.rs): gzip'd JSON lines, a header (sorted keys, as
    serde_json writes a map) then one game record per line; a file once `rows` rows are held (whole games); a branched
    game held until its branch's line (records/2)."""

    def __init__(self, out_dir, prefix, rows, header, branches=False):
        os.makedirs(out_dir, exist_ok=True)
        self.dir, self.prefix, self.rows, self.header = out_dir, prefix, max(1, int(rows)), dict(header)
        self.version = 2 if branches else 1
        self.lines, self.gids, self.nets, self.nbuf = [], [], [], 0
        self.failures = []
        self.waiting = set()
        self.seq = 0
        self.written = []

    def add(self, row, text, nets):
        """A finished game: its result row, its record's text, its samples' network ids -> a written file or None."""
        n = int(row.get('samples', 0))
        self.lines.append(text)
        self.gids.append(int(row['gid']))
        self.nets.append((n, int(min(nets)) if len(nets) else None))
        self.nbuf += n
        if row.get('branch'):
            self.waiting.discard(int(row['gid']))
        elif row.get('branched'):
            if self.version != 2:
                raise Fatal(f'game {row["gid"]}: a branched game in a records/1 writer')
            self.waiting.add(int(row['gid']))
        if self.nbuf >= self.rows and not self.waiting:
            return self.flush()
        return None

    def add_failure(self, f):
        self.failures.append(f)

    def pending(self):
        return len(self.lines)

    def flush(self):
        """Write what is held but a game still waiting for its branch."""
        if not self.waiting:
            return self._write()
        keep = [(l, g, x) for l, g, x in zip(self.lines, self.gids, self.nets) if g in self.waiting]
        out = [(l, g, x) for l, g, x in zip(self.lines, self.gids, self.nets) if g not in self.waiting]
        self.lines, self.gids, self.nets = [x[0] for x in out], [x[1] for x in out], [x[2] for x in out]
        self.nbuf = sum(x[0] for x in self.nets)
        w = self._write() if self.lines else None
        self.lines, self.gids, self.nets = [x[0] for x in keep], [x[1] for x in keep], [x[2] for x in keep]
        self.nbuf = sum(x[0] for x in self.nets)
        return w

    def finish(self):
        self.waiting.clear()
        return self._write()

    def _write(self):
        if not self.lines:
            return None
        lines, gids, nets, rows = self.lines, self.gids, self.nets, self.nbuf
        self.lines, self.gids, self.nets, self.nbuf = [], [], [], 0
        gs = [g for _, g in nets if g is not None]
        gen = min(gs) if gs else None
        h = dict(self.header, records=self.version, games=len(gids), rows=rows, gids=ranges(gids), gen=gen)
        if self.failures:
            h['failures'], self.failures = self.failures, []
        name = f'{self.prefix}_{self.seq:03d}.rec.jsonl.gz'
        path = os.path.join(self.dir, name)
        tmp = f'{path}.{os.getpid()}.tmp'
        with open(tmp, 'wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', compresslevel=6, fileobj=raw, mtime=0) as z:
                z.write(json.dumps(h, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8') + b'\n')
                for ln in lines:
                    z.write(ln.encode('utf-8') + b'\n')
        os.replace(tmp, path)
        w = {'seq': self.seq, 'path': path, 'rows': rows, 'bytes': os.path.getsize(path), 'games': len(gids), 'gids': gids,
             'gen': gen}
        self.seq += 1
        self.written.append(w)
        return w


# ------------------------------------------------------------------------------------------------ control
class Control:
    """The job's control state, shared by the agent's line reader (this process's main thread) and the job thread."""

    def __init__(self):
        self.mu = threading.Lock()
        self.cv = threading.Condition(self.mu)
        self.yielded = False
        self.quit = False
        self.finish = False
        self.checkpoint = None
        self.extend = None
        self.released = threading.Event()          # the job thread dropped its evaluators (a yield answered)
        self.released.set()
        self.restored = queue.Queue()               # a resume's outcome (None or the error) from the job thread
        self.gate_waiting = False                   # the job thread waits in its yield gate (evaluators released)
        self.resume_pending = False                 # ... and a resume waits for its rebuild's outcome


# ------------------------------------------------------------------------------------------------ the job
class Progress:
    """Throughput counters (PROTOCOL 3 Stats, the native worker's `Progress`)."""

    def __init__(self):
        self.mu = threading.Lock()
        self.games = self.started = self.rows = self.shards = self.leaves = self.sims = 0
        self.decisions = self.infer_batches = self.infer_rows = 0
        self.infer_ms = self.wait_ms = self.search_ms = 0.0
        self.batch_ms = []
        self.yields = 0
        self.yield_ms_max = 0.0
        self.concurrency = self.threads = 0
        self.wall_s = 0.0
        self.provider = ''
        self.finished_gids = []
        self.branches = self.merges = self.neartie = 0
        self.selftest_max_abs = None
        self.vram_mb = 0

    def stats(self):
        with self.mu:
            ms = sorted(self.batch_ms)

            def q(p):
                return None if not ms else round(float(ms[int(round((len(ms) - 1) * p))]), 3)
            rss, _ = rss_mb()
            st = self.selftest_max_abs
            v = {'games': self.games, 'decisions': self.decisions, 'leaves': self.leaves, 'infer_batches': self.infer_batches,
                 'search_ms': round(self.search_ms), 'infer_ms_p50': q(0.5), 'infer_ms_p95': q(0.95),
                 'infer_ms': round(self.infer_ms), 'wait_ms': round(self.wait_ms), 'rss_mb': rss, 'threads': self.threads,
                 'concurrency': self.concurrency, 'provider': self.provider, 'yields': self.yields,
                 'yield_ms_max': round(self.yield_ms_max, 1), 'rows': self.rows, 'shards': self.shards, 'sims': self.sims,
                 'games_started': self.started, 'infer_rows': self.infer_rows, 'wall_s': round(self.wall_s, 1),
                 'selftest_max_abs': None if st is None else (st if st == st and st != float('inf') else 'inf'),
                 'leaves_per_s': round(self.leaves / self.wall_s, 1) if self.wall_s > 0 else 0.0,
                 'vram_mb': self.vram_mb}
            if self.branches or self.neartie:
                v.update(branches=self.branches, merges=self.merges, neartie=self.neartie)
            return v


def job_settings(settings, threads=None, conc=None, groups=None):
    """A job's settings -> the run config its driver is built from: zero.run.DEFAULTS, the template's settings over
    them (its `setting_classes` declaration dropped: not a setting), the game resolved (`game` + `game_layout`, docs/az.md
    73), the job's search threads / conc / groups over the template's, sparse batches (the native worker's rule). The
    features the native worker refuses (the self-play league, archive starts: their games don't replay from a seed) are
    refused here too (Fatal)."""
    from . import package, run as R
    s = dict(settings or {})
    s.pop('setting_classes', None)
    if not s.get('game'):
        raise Fatal('job settings: no game')
    cfg = dict(R.DEFAULTS)
    cfg.update(s)
    try:
        package.set_game(cfg, package.resolve_game(s['game'], s.get('game_layout'), 'job settings'))
    except Exception as e:  # noqa: BLE001 - GameAliasError and kin
        raise Fatal(f'job settings: {e}') from None
    if float(cfg.get('selfplay_league') or 0) > 0:
        raise Fatal('selfplay_league: the self-play league is not played by farm workers (its games need the league table)')
    if cfg.get('selfplay_archive'):
        raise Fatal('selfplay_archive: archive starts are not played by farm workers (their games do not replay from a seed)')
    if threads is not None:
        cfg['threads'] = int(threads)
    if conc is not None:
        cfg['conc'] = int(conc)
    if groups is not None:
        cfg['groups'] = int(groups)
    cfg['sparse'] = 1
    return cfg


def pick_device(job):
    """The job's torch device: 'cpu' when the job (or env ZERO_TORCH_WORKER_DEVICE) says 'cpu' (tests); else the CUDA
    device of the job's adapter (`gpu` {luid, vendor, name}, PROTOCOL 8): torch reports no LUID, so the one CUDA device of
    that name when exactly one has it (an owner machine has one NVIDIA GPU), else cuda:0."""
    import torch
    want = job.get('device') or os.environ.get('ZERO_TORCH_WORKER_DEVICE') or 'cuda'
    if want == 'cpu':
        return 'cpu'
    if not torch.cuda.is_available():
        raise Fatal('no CUDA device: torch.cuda.is_available() is False (this torch worker plays on an NVIDIA GPU)')
    if want != 'cuda':
        return want
    name = str((job.get('gpu') or {}).get('name') or '').strip().lower()
    if name:
        hit = [i for i in range(torch.cuda.device_count()) if torch.cuda.get_device_name(i).strip().lower() == name]
        if len(hit) == 1:
            return f'cuda:{hit[0]}'
    return 'cuda:0'


def load_reference(path, tails=None):
    """A reference set (zero.onnx_export's reference.npz) -> ([LeafBatch per recorded batch], (priors, value, margin, aux)
    per batch, meta): the rows exactly as a driver hands them to the evaluator (sparse, the driver's dtypes)."""
    import numpy as np
    from .protocol import LeafBatch
    with np.load(path) as z:
        a = {k: z[k] for k in z.files}
    meta = json.loads(str(a['meta']))
    bo = a['batch_off'].astype(np.int64)
    xs_off, opt_off = a['xs_off'].astype(np.int64), a['opt_off'].astype(np.int64)
    belief = (a['extra'] == 2)
    brow = np.concatenate([[0], np.cumsum(belief)]).astype(np.int64)
    batches, want = [], []
    has_t, has_ov = 't_off' in a, 'ov_off' in a
    for k in range(len(bo) - 1):
        r0, r1 = int(bo[k]), int(bo[k + 1])
        x0, x1 = int(xs_off[r0]), int(xs_off[r1])
        o0, o1 = int(opt_off[r0]), int(opt_off[r1])
        d = {'xs_off': (xs_off[r0:r1 + 1] - x0).astype(np.int64), 'xs_idx': a['xs_idx'][x0:x1].astype(np.uint32),
             'xs_val': a['xs_val'][x0:x1].astype(np.float32), 'n_features': int(meta['features']),
             'n_players': a['n_players'][r0:r1].astype(np.int32), 'player': np.zeros(r1 - r0, np.int32),
             'gid': np.zeros(r1 - r0, np.int64), 'extra': a['extra'][r0:r1].astype(np.uint8),
             'opt_off': (opt_off[r0:r1 + 1] - o0).astype(np.int32), 'opt_action': a['opt_action'][o0:o1].astype(np.int32)}
        if has_t:
            t_off = a['t_off'].astype(np.int64)
            t0, t1 = int(t_off[r0]), int(t_off[r1])
            widths = {int(s): int(w) for s, _n, w, _h in meta['t_segs']}
            # (t_val holds each token's floats consecutively: its offset by the widths of the tokens before it)
            seg = a['t_seg'].astype(np.int64)
            fl = np.concatenate([[0], np.cumsum([widths[int(s)] for s in seg])]).astype(np.int64)
            d.update(t_off=(t_off[r0:r1 + 1] - t0).astype(np.int64), t_seg=a['t_seg'][t0:t1].astype(np.uint16),
                     t_slot=a['t_slot'][t0:t1].astype(np.uint32), t_val=a['t_val'][int(fl[t0]):int(fl[t1])].astype(np.float32))
        if has_ov:
            ov_off = a['ov_off'].astype(np.int64)
            v0, v1 = int(ov_off[r0]), int(ov_off[r1])
            d.update(ov_off=(ov_off[r0:r1 + 1] - v0).astype(np.int64), ov_opt=a['ov_opt'][v0:v1].astype(np.int32),
                     ov_tok=a['ov_tok'][v0:v1].astype(np.int64))
        batches.append(LeafBatch(k, 'm', 'nn', d, tails=tails))
        aux = a['aux'][int(brow[r0]):int(brow[r1])] if a['aux'].size else a['aux']
        want.append((a['priors'][o0:o1], a['value'][r0:r1], a['margin'][r0:r1], aux))
    return batches, want, meta


def answer_all(ev, batches):
    """The evaluator's answers to the batches, through `launch` / `result` as play uses it."""
    out = []
    for lb in batches:
        r = ev.result(ev.launch(lb)) if hasattr(ev, 'launch') else ev(lb)
        out.append(tuple(r))
    return out


def self_test(ev, ref_path, precision, tails=None):
    """The per-machine self-test (module docs) -> the `selftest` event's fields; raises Fatal when it fails."""
    import numpy as np
    t0 = time.time()
    batches, want, _ = load_reference(ref_path, tails)
    a = answer_all(ev, batches)
    b = answer_all(ev, batches)
    det = all(all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(p, q)) for p, q in zip(a, b))
    items = [0.0, 0.0, 0.0, 0.0]
    sums, counts = [0.0] * 4, [0] * 4
    for got, w in zip(a, want):
        for i in range(4):
            g = np.asarray(got[i] if i < len(got) else np.zeros((0,), np.float32), np.float64)
            x = np.asarray(w[i], np.float64)
            if i == 3 and x.size == 0:
                continue
            if g.shape != x.shape:
                items[i] = float('inf')
                continue
            if g.size:
                d = np.abs(g - x)
                m = float(d.max())
                items[i] = max(items[i], m if np.isfinite(m) else float('inf'))
                sums[i] += float(d.sum())
                counts[i] += d.size
    tol, mean_tol = TOL.get(precision, TOL['fp32'])
    means = [s / c if c else 0.0 for s, c in zip(sums, counts)]
    max_abs = max(items)
    ok = det and max_abs <= tol and (mean_tol is None or max(means) <= mean_tol)
    f = lambda v: v if v != float('inf') and v == v else 'inf'  # noqa: E731
    out = {'max_abs': f(max_abs), 'ok': bool(ok), 'deterministic': bool(det), 'rows': int(sum(lb.B for lb in batches)),
           'items': dict(zip(('priors', 'value', 'margin', 'aux'), map(f, items))),
           'means': dict(zip(('priors', 'value', 'margin', 'aux'), (round(m, 7) for m in means))),
           'tol': tol, 'mean_tol': mean_tol, 'precision': precision, 'seconds': round(time.time() - t0, 3)}
    return out


class Job:
    """One self-play job (module docs; crates/worker job.rs's `run_job`)."""

    def __init__(self, line, base, out, ctl, progress, bid):
        self.line, self.base, self.out, self.ctl, self.p, self.bid = line, base, out, ctl, progress, bid
        u = lambda k: None if line.get(k) is None else int(line[k])  # noqa: E731
        if line.get('kind') != 'selfplay':
            raise Fatal(f'job kind {line.get("kind")!r}: this torch worker build runs {KINDS}')
        if not isinstance(line.get('settings'), dict):
            raise Fatal('job: settings must be an object')
        self.lease = str(line.get('lease_id') or 'job')
        self.games = u('games')
        if self.games is None:
            raise Fatal('job: games')
        self.gid_base = u('gid_base') or 0
        self.seed = u('seed') or 0
        self.threads = max(1, u('threads') or 1)
        self.out_dir = os.path.join(base, line.get('out_dir') or 'out')
        self.prefix = str(line.get('prefix') or 'f')
        output = line.get('output') or 'records'
        if output != 'records':
            raise Fatal(f'job: output {output!r}: the torch worker writes game records only')
        self.uniform = line.get('model') is None and line.get('uniform') is True
        if line.get('model') is None and not self.uniform:
            raise Fatal('job: no model (a self-play job needs its network)')
        self.model = None if self.uniform else os.path.join(base, line['model'])
        self.sha256 = line.get('sha256')
        self.reference = os.path.join(base, line['reference']) if line.get('reference') else None
        r = line.get('rolling')
        if r not in (None, False, True):
            raise Fatal(f'job: rolling {r!r}: true or false')
        self.rolling = r is True
        fs = line.get('flush_s')
        self.flush_s = FLUSH_S if fs is None else float(fs)
        if self.rolling and not self.flush_s > 0:
            raise Fatal(f'job: flush_s {fs!r}: a number of seconds > 0')
        self.status_s = float(line.get('status_s') or STATUS_S)
        self.cfg = job_settings(line['settings'], self.threads, u('conc'), u('groups'))
        self.gen = u('gen')
        if self.gen is None and self.model is not None:
            self.gen = None                     # (from the file, `_load_net`)
        self.inflight = int(os.environ.get('ZERO_INFLIGHT', '2'))
        self.device = None
        self.ev = None                          # the evaluator playing now
        self.net = None                         # (its file's dict: rebuilt after a yield)
        self.infer = None
        self.building = None                    # a rolling checkpoint's build: (ck, thread, box)
        self.next_ck = None
        self.switching = None
        self.staged = None                      # the built evaluator of the checkpoint being switched to

    # ---- the network
    def _check_sha(self, path, want, gen=None):
        if not want:
            return
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for b in iter(lambda: f.read(1 << 20), b''):
                h.update(b)
        if h.hexdigest().lower() != str(want).lower():
            raise Fatal(f'{"gen %s: " % gen if gen is not None else ""}{path}: the file\'s sha256 is not the job\'s ({want})')

    def _read_net(self, path, sha, gen=None):
        from . import torch_weights as TW
        self._check_sha(path, sha, gen)
        try:
            d = TW.read(path)
        except TW.NotTorchWeights as e:
            raise Fatal(str(e)) from None
        inf = TW.infer_of({})
        inf.update({k: d['infer'][k] for k in TW.INFER_KEYS if k in (d.get('infer') or {})})
        return d, inf

    def _evaluator(self, d, inf):
        """The trainer's evaluator (`Run.evaluator`'s NetEvaluator) for a read torch-weights file."""
        from . import torch_weights as TW
        from .infer import NetEvaluator
        m = TW.build(d, self.device, game=self.cfg['game'])
        graphs = True if (str(self.device).startswith('cuda') and m.config.get('arch') == 'tokens'
                          and os.environ.get('ZERO_CUDA_GRAPHS', '1') != '0') else None
        return NetEvaluator(m, self.device, graphs=graphs, workers=inf['workers'], token_chunks=inf['token_chunks'],
                            precision=inf['precision'])

    def _build(self, d, inf, ref, gen, sink):
        """An evaluator that passed the self-test on `ref` (the `selftest` event to `sink`); Fatal when it fails."""
        ev = self._evaluator(d, inf)
        if ref:
            st = self_test(ev, ref, ev.precision, self.tails)
            ev_line = dict(provider=self.provider, **st)
            if gen is not None and sink is not None:
                ev_line['gen'] = gen
            if sink is not None:
                sink(ev_line)
            with self.p.mu:
                self.p.selftest_max_abs = st['max_abs'] if st['max_abs'] != 'inf' else float('inf')
            if not st['ok']:
                raise Fatal(f'self-test failed ({self.provider}, {ev.precision}): max |difference| {st["max_abs"]} '
                            f'(tolerance {st["tol"]}; per item {st["items"]}, means {st["means"]}), deterministic '
                            f'{st["deterministic"]}')
        return ev

    def _release(self):
        """Drop the evaluators (and their CUDA graphs) and empty torch's cache: no video memory held."""
        import gc
        self.ev = None
        self.staged = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _vram(self):
        try:
            import torch
            if str(self.device).startswith('cuda'):
                return int(torch.cuda.memory_reserved(self.device) >> 20)
        except Exception:  # noqa: BLE001
            pass
        return 0

    # ---- control
    def _gate(self, pending):
        """Before a new batch: a yield finishes the batches in flight, drops the evaluators and waits for resume/quit."""
        ctl = self.ctl
        if not ctl.yielded:
            return
        t0 = time.time()
        while pending:
            self._finish_oldest(pending)
        self._release()
        with self.p.mu:
            self.p.vram_mb = self._vram()
        ctl.released.set()
        with ctl.cv:
            ctl.gate_waiting = True
            while ctl.yielded and not ctl.quit:
                ctl.cv.wait(0.5)
            ctl.gate_waiting = False
            report, ctl.resume_pending = ctl.resume_pending, False
            resumed = not ctl.quit
        err = None
        if resumed and not self.uniform:
            try:
                self.ev = self._evaluator(self.net, self.infer)
            except Exception as e:  # noqa: BLE001 - the resume fails: fatal (PROTOCOL 5)
                err = f'the evaluator could not be rebuilt: {type(e).__name__}: {e}'
        if report:
            ctl.restored.put(err)
        if err:
            raise Fatal(f'resume: {err}')
        self.out.log(f'yield gate: {time.time() - t0:.1f} s')

    # ---- the loop
    def _snapshot(self):
        st = self.drv.stats()
        with self.p.mu:
            self.p.games = self.finished
            self.p.started = self.enq - int(self.drv.queued)
            self.p.leaves = int(st['leaves'])
            self.p.sims = int(st['sims'])
            self.p.wait_ms = float(st['wait_s']) * 1e3
            self.p.search_ms = float(st['busy_s']) * 1e3
            self.p.wall_s = time.time() - self.t0

    def _emit_file(self, w):
        with self.p.mu:
            self.p.rows += w['rows']
            self.p.shards += 1
        rel = os.path.relpath(w['path'], self.out_dir).replace('\\', '/')
        self.out.event('records', seq=w['seq'], path=rel, games=w['games'], rows=w['rows'], bytes=w['bytes'],
                       gids=ranges(w['gids']), gen=w['gen'])

    def _flush_held(self):
        w = self.rec.flush()
        if w:
            self._emit_file(w)
        self.held_since = None

    def _start_build(self, ck):
        box = {}

        def run():
            try:
                d, inf = self._read_net(ck['model'], ck.get('sha256'), ck['gen'])
                evs = []
                box['ev'] = self._build(d, inf, ck.get('reference'), ck['gen'], lambda e: evs.append(e))
                box['net'], box['infer'], box['events'] = d, inf, evs
            except BaseException as e:  # noqa: BLE001 - collected by the job
                box['error'] = e
        th = threading.Thread(target=run, name='torch-worker-checkpoint', daemon=True)
        th.start()
        self.out.log(f'checkpoint {ck["gen"]} ({ck["model"]}): building its evaluator while the games go on')
        self.building = (ck, th, box)

    def _rolling_checkpoint(self, ck):
        if ck is not None:
            if self.building is not None or self.switching is not None:
                self.next_ck = ck
            elif not self.ended:
                self._start_build(ck)
        if self.building is not None and not self.building[1].is_alive():
            c, _, box = self.building
            self.building = None
            for e in box.get('events', []):
                self.out.event('selftest', **{k: v for k, v in e.items()})
            if 'error' in box:
                e = box['error']
                raise e if isinstance(e, Fatal) else Fatal(f'checkpoint {c["gen"]}: {type(e).__name__}: {e}')
            if self.next_ck is not None:
                n, self.next_ck = self.next_ck, None
                self.out.log(f'checkpoint {c["gen"]}: superseded by {n["gen"]} before the switch')
                if not self.ended:
                    self._start_build(n)
            elif not self.ended:
                self.staged = box
                self.drv.end_phase()
                self.switching = c

    def pump(self):
        ctl = self.ctl
        with ctl.mu:
            quit_, finish, ck, extend = ctl.quit, ctl.finish, ctl.checkpoint, ctl.extend
            ctl.checkpoint = None
        if quit_ and not self.ended:
            self.drv.end_phase()
            self.ended = True
            self.stop_topup = True
        if finish:
            self.stop_topup = True
        if self.rolling:
            if extend is not None and extend > self.n_games:
                self.n_games = extend
            self._rolling_checkpoint(ck)
        elif ck is not None and not self.ended:
            self.drv.end_phase()
            self.switching = ck
        rows, texts, samples = self.drv.take_game_records()
        for f in self.drv.take_failures():
            self.out.log(f'game failure (dropped after a panic): {f}')
            try:
                self.rec.add_failure(json.loads(f))
            except ValueError:
                self.rec.add_failure(f)
        nets = samples['net'] if samples is not None else []
        at = 0
        n = 0
        for r, text in zip(rows, texts):
            k = int(r.get('samples', 0))
            net = nets[at:at + k]
            at += k
            with self.p.mu:
                if r.get('branch'):
                    self.p.branches += 1
                    self.p.merges += 1 if (r['branch'] or {}).get('merged') is not None else 0
                else:
                    self.p.finished_gids.append(int(r['gid']))
                self.p.neartie += int(r.get('neartie') or 0)
                self.p.decisions += int(r.get('decisions') or 0)
            if not r.get('branch'):
                n += 1
            if text == '':
                continue                        # (a game dropped after a panic: no record)
            w = self.rec.add(r, text, net)
            if w:
                self._emit_file(w)
            if self.rolling:
                if self.rec.pending() == 0:
                    self.held_since = None
                elif self.held_since is None:
                    self.held_since = time.time()
        if samples is not None and at != len(nets):
            raise Fatal(f'{len(nets) - at} samples left over after the finished games')
        self.finished += n
        if n:
            self.out.event('fin', n=n)
        if not self.stop_topup and self.switching is None:
            room = min(max(0, self.cap - (self.enq - self.finished)), self.n_games - self.enq)
            for _ in range(room):
                self.drv.enqueue(f'{self.lease}:{self.enq}', self.spec)
                self.enq += 1
        if self.rolling:
            if not self.stop_topup and self.n_games - self.enq < self.cap and self.low_sent != self.n_games:
                self.low_sent = self.n_games
                self.out.event('low', started=self.enq, games=self.n_games)
            if self.held_since is not None and time.time() - self.held_since >= self.flush_s:
                self._flush_held()
        self._snapshot()

    def _finish_oldest(self, pending):
        lb, ev, h, t0 = pending.popleft()
        ans = ev.result(h)
        dt = time.perf_counter() - t0
        with self.p.mu:
            self.p.infer_batches += 1
            self.p.infer_ms += dt * 1e3
            if len(self.p.batch_ms) < 1 << 20:
                self.p.batch_ms.append(dt * 1e3)
        self.drv.submit(lb.id, *ans)
        self.pump()

    def drive(self):
        """infer._drive_pipelined's loop with the yield gate (module docs), until the driver has no batch."""
        pending = collections.deque()
        while True:
            self._gate(pending)
            lb = self.drv.poll_batch() if pending else self.drv.next_batch()
            if lb is None:
                if not pending:
                    return
                self._finish_oldest(pending)
                continue
            if lb.external:
                raise Fatal('an external batch: the torch worker has no external players')
            with self.p.mu:
                self.p.infer_rows += lb.B
            t0 = time.perf_counter()
            ev = self.ev
            h = ev.launch(lb) if hasattr(ev, 'launch') else ('now', ev(lb))
            pending.append((lb, _Now(ev) if not hasattr(ev, 'launch') else ev, h, t0))
            # (no reference to the evaluator outlives its batches here: a yield must free it - its CUDA graphs, its
            # weights - and a local of this frame kept the whole evaluator on the GPU through the gate)
            ev = h = lb = None
            if len(pending) >= self.inflight:
                self._finish_oldest(pending)

    def run(self):
        import torch
        from . import games as G
        from .run import Run
        from .infer import UniformEvaluator
        self.t0 = time.time()
        torch.set_num_threads(1)
        self.device = 'cpu' if self.uniform else pick_device(self.line)
        self.provider = 'uniform' if self.uniform else PROVIDER['cuda' if str(self.device).startswith('cuda') else 'cpu']
        info = G.info(self.cfg['game'])
        self.tails = G.tail_widths(info)
        if not self.uniform:
            self.net, self.infer = self._read_net(self.model, self.sha256, self.gen)
            torch.set_num_threads(max(1, int(self.infer['torch_threads'])))
            if self.gen is None:
                g = self.net.get('gen')
                if g is None:
                    raise Fatal(f'{self.model}: no gen in the file: give the job\'s gen')
                self.gen = int(g)
        else:
            self.gen = self.gen or 0
        if str(self.device).startswith('cuda'):
            torch.cuda.set_device(self.device)
        R = Run(self.base, self.cfg)
        seat = R.nn_seat('m', self.cfg['sims'], selfplay=True)
        table = [[seat] * int(p) for p in self.cfg['players']]
        self.drv = R.driver(table, seed=self.seed, record=True, rolling=True, net=self.gen, queue_only=True)
        branches = R.forks is not None and bool(R.forks.branch)
        self.cap = int(self.cfg['conc'])
        if self.cap == 0:
            raise Fatal('conc 0: no game would ever start (the setup cap is the games in flight)')
        self.spec = G.spec_json(self.cfg.get('spec'))
        if self.gid_base:
            self.drv.set_first_gid(self.gid_base)
        self.drv.set_trace(True)
        # (the evaluator before the driver's first batch; a yield that came while it loaded is answered by the gate)
        if self.uniform:
            self.ev = UniformEvaluator()
        else:
            self.ev = self._build(self.net, self.infer, self.reference, None, lambda e: self.out.event('selftest', **e))
        self.out.log(f'job {self.lease}: {self.games} games from gid {self.gid_base}, seed {self.seed}, net {self.gen}; '
                     f'driver: conc {self.cap}, threads {self.cfg["threads"]}, groups {self.cfg["groups"]}, sparse; '
                     f'evaluator: {self.provider} on {self.device}'
                     + (f', {self.infer["precision"]}, graphs {getattr(self.ev, "fast", None) is not None}, token_chunks '
                        f'{self.infer["token_chunks"]}, workers {self.infer["workers"]}' if self.infer else ''))
        with self.p.mu:
            self.p.concurrency = self.cap
            self.p.threads = int(self.cfg['threads'])
            self.p.provider = self.provider
        self.drv.resume(BIG, self.gen)
        hdr = {'game': info['name'], **_layout_stamp(info['name']), 'spec': self.spec, 'aux': bool(getattr(R, 'aux', None)),
               'features': int(info['num_features']), 'actions': int(info['num_actions']),
               'aux_width': int(sum(int(a[1]) for a in (info.get('aux') or []))), 'seed': self.seed, 'lease': self.lease,
               'build_id': self.bid}
        self.rec = RecordWriter(self.out_dir, self.prefix, int(self.cfg['shard_rows']), hdr, branches=branches)
        self.enq = self.finished = 0
        self.n_games = self.games
        self.stop_topup = self.ended = False
        self.low_sent = None
        self.held_since = None
        self.pump()
        idle = 0
        while True:
            self.drive()
            if self.rolling:
                if self.ended:
                    self.switching = None
                if self.switching is not None:
                    c, self.switching = self.switching, None
                    box, self.staged = self.staged, None
                    self.ev = None
                    self._release()
                    self.ev, self.net, self.infer = box['ev'], box['net'], box['infer']
                    self.drv.resume(BIG, c['gen'])
                    self.out.log(f'checkpoint {c["gen"]} ({c["model"]}): the driver resumed with it')
                    if self.next_ck is not None:
                        n, self.next_ck = self.next_ck, None
                        self._start_build(n)
                    idle = 0
                    self.pump()
                    continue
                if self.ended or (self.stop_topup and self.enq == self.finished):
                    break
                idle += 1
                if idle > 1:
                    time.sleep(0.02)
                self._gate([])
                self.pump()
                continue
            if self.switching is not None:
                c, self.switching = self.switching, None
                if self.uniform:
                    raise Fatal('checkpoint: a uniform-evaluator job has no network to switch')
                d, inf = self._read_net(c['model'], c.get('sha256'), c['gen'])
                self._release()
                self.ev = self._build(d, inf, c.get('reference'), None, lambda e: self.out.event('selftest', **e))
                self.net, self.infer = d, inf
                self.drv.resume(BIG, c['gen'])
                self.out.log(f'checkpoint {c["gen"]} ({c["model"]}): the driver resumed with it')
                self.pump()
                continue
            if self.ended or self.finished >= self.games or (self.stop_topup and self.enq == self.finished):
                break
            idle += 1
            if idle > 1:
                time.sleep(0.02)
            self._gate([])
            self.pump()
        self.pump()
        w = self.rec.finish()
        if w:
            self._emit_file(w)
        if self.rec.failures:
            self.out.log(f'{len(self.rec.failures)} game failure(s) after the last finished game: in this log only')
        self._snapshot()
        with self.ctl.mu:
            quit_ = self.ctl.quit
        reason = 'done' if (not self.rolling and self.finished >= self.games) else ('quit' if quit_ else 'finish')
        st = self.p.stats()
        st['peak_rss_mb'] = rss_mb()[1]
        st['unshipped'] = self.rec.pending()
        st['seconds'] = round(time.time() - self.t0, 2)
        self.ev = None
        self._release()
        return {'reason': reason, 'stats': st,
                'unfinished': {'gids': missing_ranges(self.gid_base, self.gid_base + self.n_games, self.p.finished_gids)}}


class _Now:
    """An evaluator without launch/result (the uniform one): answered at launch."""

    def __init__(self, ev):
        self.ev = ev

    def result(self, h):
        return h[1]


def _layout_stamp(game):
    from . import package
    return package.layout_stamp(game)


# ------------------------------------------------------------------------------------------------ serve
def _pipe_lines():
    """Windows, stdin a pipe: its lines without ever leaving a read pending (PeekNamedPipe, then read what is there).
    A thread blocked in a synchronous ReadFile on the pipe serializes every other call on that file object - and a DLL
    whose C runtime starts up queries the standard handles: importing numpy or torch on the job's thread then hangs
    for as long as no line comes (seen with numpy's _multiarray_umath). None when stdin is not a pipe here."""
    if os.name != 'nt':
        return None
    import ctypes
    import msvcrt
    from ctypes import wintypes
    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    k32.GetFileType.argtypes = [wintypes.HANDLE]
    k32.PeekNamedPipe.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p,
                                  ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    try:
        h = msvcrt.get_osfhandle(sys.stdin.fileno())
    except (OSError, ValueError, AttributeError):
        return None
    if k32.GetFileType(h) != 3:                     # FILE_TYPE_PIPE
        return None

    def gen():
        buf = b''
        avail = wintypes.DWORD()
        while True:
            if not k32.PeekNamedPipe(h, None, 0, None, ctypes.byref(avail), None):
                break                               # (broken: the agent is gone)
            if avail.value == 0:
                time.sleep(0.02)
                continue
            data = os.read(sys.stdin.fileno(), avail.value)
            if not data:
                break
            buf += data
            while b'\n' in buf:
                ln, buf = buf.split(b'\n', 1)
                yield ln.decode('utf-8', 'replace')
        if buf.strip():
            yield buf.decode('utf-8', 'replace')
    return gen()


def _stdin_lines(out):
    q = queue.Queue()
    src = _pipe_lines()

    def read():
        try:
            for ln in (src if src is not None else sys.stdin):
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    q.put(json.loads(ln))
                except ValueError as e:
                    out.event('error', message=f'a line that is not JSON: {e}', fatal=False)
        except (OSError, ValueError):
            pass
        q.put(None)
    threading.Thread(target=read, name='torch-worker-stdin', daemon=True).start()
    return q


def _control(msg, job, ctl, progress, out, base):
    """One control line during a job -> a fatal message or None (crates/worker proto.rs `control`)."""
    op = msg.get('op')
    if op == 'extend' and job.rolling:
        g = msg.get('games')
        if not isinstance(g, int):
            out.event('error', message='extend: games (the job\'s new total) is required', fatal=False)
        else:
            with ctl.mu:
                if ctl.extend is not None and ctl.extend >= g:
                    out.log(f'extend to {g}: not above the games asked before (ignored)')
                else:
                    ctl.extend = g
    elif op == 'yield':
        t0 = time.time()
        with ctl.cv:
            already = ctl.yielded
            if not already:
                ctl.yielded = True
                ctl.released.clear()
                ctl.cv.notify_all()
        if not already:
            ctl.released.wait(YIELD_WAIT_S)
            dt = time.time() - t0
            with progress.mu:
                progress.yields += 1
                progress.yield_ms_max = max(progress.yield_ms_max, dt * 1e3)
            out.log(f'yielded in {dt * 1e3:.0f} ms' + ('' if ctl.released.is_set() else ' (the evaluator not released yet)'))
        else:
            out.log('yield: already yielded')
        out.event('yielded')
    elif op == 'resume':
        with ctl.cv:
            ctl.yielded = False
            report = ctl.gate_waiting
            if report:
                ctl.resume_pending = True
            ctl.cv.notify_all()
        if report:
            # (the gate released the evaluators and waits: it rebuilds them now; its outcome decides)
            try:
                err = ctl.restored.get(timeout=120)
            except queue.Empty:
                err = 'no answer from the job within 120 s'
            if err:
                return f'resume: {err}'
        out.event('resumed')
    elif op == 'finish':
        with ctl.mu:
            ctl.finish = True
    elif op == 'quit':
        with ctl.cv:
            ctl.quit = True
            ctl.yielded = False
            ctl.cv.notify_all()
    elif op == 'checkpoint':
        g, m = msg.get('gen'), msg.get('model')
        if not isinstance(g, int) or not m:
            out.event('error', message='checkpoint: gen and model are required', fatal=False)
        else:
            with ctl.mu:
                ctl.checkpoint = {'gen': g, 'model': os.path.join(base, m), 'sha256': msg.get('sha256'),
                                  'reference': os.path.join(base, msg['reference']) if msg.get('reference') else None}
    elif op == 'ping':
        out.event('status', stats=progress.stats())
    elif op == 'job':
        out.event('error', message='a second job line: one job per worker process', fatal=False)
    else:
        out.event('error', message=f'unknown op {op!r}', fatal=False)
    return None


def hello(bid, repo):
    return dict(aw=AW, build_id=bid, game=repo, kinds=list(KINDS), zero_onnx=[], models=['torch-weights/1'],
                features=list(FEATURES), pid=os.getpid())


def serve(build_dir, repo, out=None, stdin_q=None, hello_sent=False):
    """`worker_main.py serve` (module docs) -> the exit code. `out` / `stdin_q`: the tests' channels; `hello_sent`: the
    entry sent `hello` itself (before it imported its game package)."""
    out = out or Out(protocol_stdout())
    bid = build_id(build_dir)
    if not hello_sent:
        out.event('hello', **hello(bid, repo))
    q = stdin_q or _stdin_lines(out)
    base = os.getcwd()
    while True:
        msg = q.get()
        if msg is None:
            return 0
        op = msg.get('op')
        if op == 'job':
            return _serve_job(msg, base, out, q, bid)
        if op == 'ping':
            out.event('status', stats={'games': 0})
        elif op == 'quit':
            out.event('end', reason='quit', stats={}, unfinished=None)
            return 0
        elif op == 'yield':
            out.event('yielded')
        elif op == 'resume':
            out.event('resumed')
        else:
            out.event('error', message=f'op {op!r} before a job', fatal=False)


def _serve_job(line, base, out, q, bid):
    ctl, progress = Control(), Progress()
    try:
        job = Job(line, base, out, ctl, progress, bid)
    except Fatal as e:
        out.event('error', message=str(e), fatal=True)
        return 1
    box = {}

    def run():
        try:
            box['outcome'] = job.run()
        except Fatal as e:
            box['error'] = str(e)
        except BaseException as e:  # noqa: BLE001 - every failure reaches the agent
            import traceback
            out.log(traceback.format_exc())
            box['error'] = f'{type(e).__name__}: {e}'
        finally:
            ctl.released.set()
            with ctl.mu:
                if ctl.resume_pending:
                    ctl.restored.put(box.get('error') or 'the job ended')
    th = threading.Thread(target=run, name='torch-worker-job', daemon=True)
    th.start()
    t_status = time.time()
    eof = False
    fatal = None
    while th.is_alive():
        if not eof:
            try:
                msg = q.get(timeout=0.1)
            except queue.Empty:
                msg = False
            if msg is None:
                eof = True
                out.log('stdin closed: quitting the job')
                with ctl.cv:
                    ctl.quit = True
                    ctl.yielded = False
                    ctl.cv.notify_all()
            elif msg:
                e = _control(msg, job, ctl, progress, out, base)
                if e:
                    out.log(f'{e}: fatal - quitting the job')
                    eof = True
                    fatal = e
                    with ctl.cv:
                        ctl.quit = True
                        ctl.yielded = False
                        ctl.cv.notify_all()
        else:
            th.join(0.1)
        if time.time() - t_status >= job.status_s:
            t_status = time.time()
            with progress.mu:
                if job.device is not None:
                    progress.vram_mb = job._vram()
            out.event('status', stats=progress.stats())
    th.join()
    if fatal or 'error' in box:
        out.event('error', message=fatal or box['error'], fatal=True)
        return 1
    o = box['outcome']
    out.event('end', reason=o['reason'], stats=o['stats'], unfinished=o['unfinished'])
    return 0


def info(build_dir, repo):
    """`worker_main.py info`: what the native worker's `info` says of a build (zero.farm_pool's rolling check reads its
    `build_id` and `features`)."""
    plat = 'windows-x64' if os.name == 'nt' else ('macos-arm64' if sys.platform == 'darwin' else sys.platform)
    return {'repo': repo, 'aw': AW, 'zero_onnx': [], 'models': ['torch-weights/1'], 'build_id': build_id(build_dir),
            'kinds': list(KINDS), 'features': list(FEATURES), 'platform': plat, 'runtime': 'torch'}


def main(argv, build_dir, repo):
    """worker_main.py's body: `serve` (the build's entry) and `info` (other flags ignored, e.g. the native worker's
    `--ort-dll none`)."""
    if argv[:1] == ['serve']:
        return serve(build_dir, repo)
    if argv[:1] == ['info']:
        print(json.dumps(info(build_dir, repo)))
        return 0
    if argv[:1] == ['selftest']:
        return selftest_cli(argv[1:])
    print('usage: worker_main.py serve | info | selftest --model W --reference R [--device cuda]   (zero-farm PROTOCOL.md 5)',
          file=sys.stderr)
    return 2


def selftest_cli(argv):
    """`worker_main.py selftest --model <torch-weights> --reference <reference.npz> [--device cuda|cpu]`: the job's
    self-test on this machine (the release's torch-selftest step) -> the `selftest` event's fields as JSON; exit 0 iff ok."""
    import argparse
    ap = argparse.ArgumentParser(prog='worker_main.py selftest')
    ap.add_argument('--model', required=True)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)
    from . import games as G, package, torch_weights as TW
    d = TW.read(a.model)
    game = package.record_game(d['net_config'])
    job = Job.__new__(Job)
    job.device = 'cpu' if a.device == 'cpu' else pick_device({'device': a.device})
    job.cfg = {'game': game}
    job.tails = G.tail_widths(G.info(game))
    job.p = Progress()
    job.provider = PROVIDER['cuda' if str(job.device).startswith('cuda') else 'cpu']
    inf = TW.infer_of({})
    inf.update({k: d['infer'][k] for k in TW.INFER_KEYS if k in (d.get('infer') or {})})
    got = []
    try:
        job._build(d, inf, a.reference, d.get('gen'), got.append)
    except Fatal as e:
        print(json.dumps(dict(got[0] if got else {}, error=str(e))))
        return 1
    print(json.dumps(got[0]))
    return 0
