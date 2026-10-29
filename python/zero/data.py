"""Training data (docs/az.md 3): sample shards and the RAM-bounded training window.

Shards: `data/iter_NNNN/<task>_SSS.npz` (uncompressed npz, written to a temporary name and
renamed: a crash never leaves a half-written shard; remote self-play's shards, `m<KK>-<sid>_SSS.npz`,
are compressed npz - zlib level 1, ~58x for MTG's sparse features - which `np.load` reads alike:
docs/az.md 5.3). Arrays: `x` float16 [N, F], `player`,
`n_players` int8, `opt_off` int32 [N+1], `opt_action` int16/int32, `pi` float16, `z` / `q` /
`sc` float16 [N, P] (P = most players in the shard), `net` int32, optionally `aux` float16 [N, W]
(the deciders' auxiliary targets: self-play with an auxiliary loss on, docs/az.md 15), and `meta` (a
JSON string: game, features, actions, rows, task info). Float16 targets are renormalised on load.
Non-finite guard (zero/finite.py, docs/az.md 4.1): `write_shard` checks x, pi, z, q, sc after the
float16 cast; a shard with a NaN / inf is written as `*.bad.npz` instead (the window never loads
it: `shard_paths` skips it) and the run fails (`NonFinite`). The `Window` checks every shard it
loads the same way, and remembers where each row came from (`provenance`).

`TargetStats`: sharpness of the recorded policy targets (self-play metrics).

Window: the newest iterations' shards loaded into RAM (`Window(paths, ...)`), every shard
subsampled at the same rate when the total exceeds the RAM budget - the rate from the shards'
row counts (their meta), each shard subsampled as it is loaded (remote self-play triples a run's
samples per iteration; the rows kept are those of loading every shard first); `batch(idx)` builds
torch tensors: features cut / zero-padded to the network's width, dense legal masks and policy
targets over the actions, value / margin targets over MAX_SEATS; a window with `aux_width` W > 0 also
`aux` [B, W] and `aux_mask` [B] (1 = the row's shard has auxiliary targets; a shard written without
them - before the targets were on, e.g. a fork's older window - gives zeros with mask 0, so the
auxiliary losses skip its rows and the shards stay usable). The shards load in parallel
(`threads`, default `WINDOW_THREADS`): inflating a compressed shard, its CRC and the copies
release the GIL, so the window of MTG's Delver run (40 shards, 265k rows, 17.9 GB) loads in a
fraction of the 41 s it took one shard at a time. Each shard's rows are drawn in path order
before any loading (the random stream of the sequential load) and its features written into
their place in the one preallocated `x`: the window is the same, array for array, and the peak
is the window plus the shards being loaded (it was twice the window: the per-shard copies, then
their concatenation).

WindowCache (run setting `window_cache`): the shards a run's windows keep between iterations, each
shard's full arrays as loaded, so a window loads only the shards it has not seen (an iteration's
window shares all but its newest iteration's shards with the last one). The window then points at
the cached shards' features (no copy: `batch` gathers its rows shard by shard) and is the window
of loading them all, array for array. Only while the window fits its budget (no subsample: the
window is exactly its shards); a window over budget empties the cache and loads as without it.
The cached shards stay in RAM between trainings (up to the budget), where a window was freed.

`batches` (run setting `train_prefetch`): a training's batches in order; with prefetch, built that
many steps ahead on a background thread (the window's gather, the float32 features and the dense
targets: ~60 ms a batch for MTG's 33,652 features) while the training step runs on the device. The
rows are drawn from the same generator in the same order, so the batches are the same.
"""
import glob
import json
import os
import queue
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .finite import NonFinite, shard_problems
from .protocol import MAX_SEATS, concat_samples

EXT = '.npz'
BAD = '.bad.npz'                  # a quarantined shard (non-finite values): never loaded
# Shards loaded at once by a Window (env ZERO_WINDOW_THREADS overrides; 1 = one at a time).
WINDOW_THREADS = int(os.environ.get('ZERO_WINDOW_THREADS', '0')) or min(8, os.cpu_count() or 1)


def _narrow_int(a):
    lo, hi = (int(a.min()), int(a.max())) if a.size else (0, 0)
    for dt in (np.int8, np.int16, np.int32):
        if np.iinfo(dt).min <= lo and hi <= np.iinfo(dt).max:
            return a.astype(dt)
    return a.astype(np.int64)


def compact(s):
    """Sample arrays (driver format) -> the stored dtypes."""
    n = s['n_players']
    P = int(n.max()) if len(n) else 1
    return {'x': s['x'].astype(np.float16), 'player': s['player'].astype(np.int8),
            'n_players': n.astype(np.int8), 'opt_off': s['opt_off'].astype(np.int32),
            'opt_action': _narrow_int(s['opt_action']), 'pi': s['pi'].astype(np.float16),
            'z': s['z'][:, :P].astype(np.float16), 'q': s['q'][:, :P].astype(np.float16),
            'sc': s['sc'][:, :P].astype(np.float16), 'net': s['net'].astype(np.int32),
            **({'aux': s['aux'].astype(np.float16)} if 'aux' in s else {})}


def write_shard(path, s, meta, bad_path=None, compress=0):
    """Write the samples `s` to `path` (atomic) -> (rows, bytes). Non-finite values after the
    float16 cast: the shard goes to `bad_path` (default: `path` as *.bad.npz) and NonFinite is
    raised naming the arrays, rows, columns and the meta. `compress`: 0 = a plain npz (np.savez),
    1-9 = a compressed npz at that zlib level (remote self-play: 1)."""
    c = compact(s)
    meta = dict(meta, rows=int(len(c['player'])))
    probs = shard_problems(c)
    if probs:
        bad = bad_path or path[:-len(EXT)] + BAD
        os.makedirs(os.path.dirname(os.path.abspath(bad)), exist_ok=True)
        _save_npz(bad, c, dict(meta, nonfinite=probs, shard=path), compress)
        raise NonFinite(f'non-finite sample values in shard {os.path.basename(path)} (meta {json.dumps(meta)}): '
                        + '; '.join(probs) + f'. The samples were kept in {bad} (never loaded by the window); '
                        f'the run stops here')
    _save_npz(path, c, meta, compress)
    return meta['rows'], os.path.getsize(path)


def savez_level(path, level, arrays):
    """np.savez_compressed at zlib level `level` (numpy's own uses zlib's default, 6: several times
    slower on MTG's shards for a few percent of size). Readable by np.load."""
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=int(level)) as zf:
        for k, a in arrays.items():
            with zf.open(k + '.npy', 'w', force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(a), allow_pickle=False)


def _save_npz(path, c, meta, compress=0):
    tmp = path + f'.{os.getpid()}.tmp.npz'
    if compress:
        savez_level(tmp, compress, dict(meta=np.array(json.dumps(meta)), **c))
    else:
        np.savez(tmp, meta=np.array(json.dumps(meta)), **c)
    for k in range(40):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if k == 39:
                raise
            import time
            time.sleep(0.25)


def read_shard(path):
    with np.load(path) as z:
        out = {k: z[k] for k in z.files if k != 'meta'}
        out['meta'] = json.loads(str(z['meta']))
    return out


def shard_paths(d):
    """The loadable shards of a data directory (not *.bad.npz, not *.tmp.npz)."""
    return sorted(p for p in glob.glob(os.path.join(d, '*' + EXT)) if not p.endswith((BAD, '.tmp' + EXT)))


def shard_rows(path):
    """A shard's sample count from its meta (only that member is read; a shard without `rows` in
    its meta: its arrays are counted)."""
    with np.load(path) as z:
        rows = json.loads(str(z['meta'])).get('rows')
        return int(rows) if rows is not None else int(len(z['player']))


class ShardWriter:
    """Collects driver sample dicts and writes a shard every `rows` samples (and at close).
    `bad_dir`: where a shard with non-finite values is quarantined, as
    `<data dir name>_<prefix>_SSS.bad.npz` (default: next to the good ones, `*.bad.npz`).
    `compress`: zlib level of the shards (0 = plain npz). `on_write(path, rows, bytes)`: called
    after each shard is written (remote self-play announces its shards so)."""

    def __init__(self, d, prefix, rows=65536, meta=None, bad_dir=None, compress=0, on_write=None):
        os.makedirs(d, exist_ok=True)
        self.d, self.prefix, self.rows, self.meta, self.bad_dir = d, prefix, rows, dict(meta or {}), bad_dir
        self.compress, self.on_write = compress, on_write
        self.buf = []
        self.nbuf = 0
        self.k = 0
        self.samples = 0
        self.bytes = 0
        self.shards = 0

    def add(self, s):
        if s is None or not len(s['player']):
            return
        self.buf.append(s)
        self.nbuf += len(s['player'])
        if self.nbuf >= self.rows:
            self.flush()

    def flush(self):
        s = concat_samples(self.buf)
        self.buf, self.nbuf = [], 0
        if s is None:
            return
        name = f'{self.prefix}_{self.k:03d}'
        bad = None
        if self.bad_dir:
            bad = os.path.join(self.bad_dir, f'{os.path.basename(os.path.normpath(self.d))}_{name}{BAD}')
        path = os.path.join(self.d, name + EXT)
        kw = {'compress': self.compress} if self.compress else {}     # local shards: the call as it was
        rows, nbytes = write_shard(path, s, self.meta, bad, **kw)
        self.k += 1
        self.samples += rows
        self.bytes += nbytes
        self.shards += 1
        if self.on_write is not None:
            self.on_write(path, rows, nbytes)

    def close(self):
        self.flush()
        return {'samples': self.samples, 'bytes': self.bytes, 'shards': self.shards}


def target_sharpness(s):
    """Per sample of a driver sample dict: (options, normalised entropy H(pi) / ln(options),
    max share) of its policy target `pi`. Samples with one option are skipped (not recorded
    by the drivers anyway)."""
    off = np.asarray(s['opt_off'], np.int64)
    pi = np.asarray(s['pi'], np.float64)
    n = np.diff(off)
    keep = n >= 2
    if not keep.any() or not len(pi):
        return np.zeros(0), np.zeros(0), np.zeros(0)
    seg = np.repeat(np.arange(len(n)), n)
    plogp = np.where(pi > 0, pi * np.log(np.where(pi > 0, pi, 1.0)), 0.0)
    h = -np.bincount(seg, plogp, minlength=len(n))
    mx = np.zeros(len(n))
    np.maximum.at(mx, seg, pi)
    nn = n[keep].astype(np.float64)
    return nn, h[keep] / np.log(nn), mx[keep]


class TargetStats:
    """Running means of the policy targets' sharpness over driver sample dicts."""

    def __init__(self):
        self.n = 0
        self.opts = 0.0
        self.h = 0.0
        self.mx = 0.0

    def add(self, s):
        if s is None or not len(s['player']):
            return
        nn, h, mx = target_sharpness(s)
        self.n += len(nn)
        self.opts += float(nn.sum())
        self.h += float(h.sum())
        self.mx += float(mx.sum())

    def summary(self, kind='visits'):
        k = max(1, self.n)
        return {'kind': kind, 'samples': self.n, 'options': round(self.opts / k, 3), 'h_norm': round(self.h / k, 4),
                'max_share': round(self.mx / k, 4)}


def _ranges(starts, lens):
    """Concatenated aranges [starts[i], starts[i] + lens[i])."""
    tot = int(lens.sum())
    if tot == 0:
        return np.zeros(0, np.int64)
    return np.arange(tot) - np.repeat(np.cumsum(lens) - lens, lens) + np.repeat(starts, lens)


def _file_key(path):
    st = os.stat(path)
    return st.st_size, st.st_mtime_ns


def _load_part(path, N, idx, F, xdst=None, AW=0):
    """One shard's part of a window: rows `idx` (None = all N), features cut / zero-padded to F
    written into `xdst` (a fresh array when None: then also returned as 'x'); `AW` > 0: the auxiliary
    targets too ('aux' float16 [rows, AW] and 'aux_mask': zeros and False for a shard without them)."""
    p = read_shard(path)
    probs = shard_problems(p)
    if probs:
        raise NonFinite(f'non-finite sample values in shard {path} (meta {json.dumps(p["meta"])}): '
                        + '; '.join(probs) + '. Move it out of the data directory (or rename it to '
                        '*.bad.npz) to train without it')
    if len(p['player']) != N:
        raise ValueError(f'shard {path}: {len(p["player"])} rows, its meta says {N}')
    if idx is None:
        idx = np.arange(N)
        sel = slice(None)
    else:
        sel = idx
    out = {}
    if xdst is None:
        xdst = out['x'] = np.empty((len(idx), F), np.float16)
    w = min(F, p['x'].shape[1])
    xdst[:, :w] = p['x'][sel, :w]
    if w < F:
        xdst[:, w:] = 0
    off = p['opt_off'].astype(np.int64)
    lens = off[idx + 1] - off[idx]
    pos = _ranges(off[idx], lens)
    out.update(src=idx.astype(np.int32), player=p['player'][sel].astype(np.int64),
               n_players=p['n_players'][sel].astype(np.int64), lens=lens,
               opt_action=p['opt_action'][pos].astype(np.int64), pi=p['pi'][pos].astype(np.float32))
    for key in ('z', 'q', 'sc'):
        a = np.zeros((len(idx), MAX_SEATS), np.float32)
        v = p[key][sel].astype(np.float32)
        a[:, :v.shape[1]] = v
        out[key] = a
    if AW:
        if 'aux' in p:
            if p['aux'].shape[1] != AW:
                raise ValueError(f'shard {path}: auxiliary targets of width {p["aux"].shape[1]}, the window wants {AW}')
            out['aux'] = p['aux'][sel].astype(np.float16)
            out['aux_mask'] = np.ones(len(idx), bool)
        else:
            out['aux'] = np.zeros((len(idx), AW), np.float16)
            out['aux_mask'] = np.zeros(len(idx), bool)
    return out


def _run_all(fn, n, threads):
    """[fn(k) for k in range(n)] on up to `threads` threads; the first failure in order is raised."""
    t = max(1, min(threads or WINDOW_THREADS, n))
    if t == 1:
        return [fn(k) for k in range(n)]
    with ThreadPoolExecutor(t) as ex:
        futs = [ex.submit(fn, k) for k in range(n)]
        return [f.result() for f in futs]


def batches(W, rng, n, size, device='cpu', prefetch=0):
    """A training's `n` batches: `size` window rows drawn from `rng` (uniform, with replacement),
    one draw per batch in order -> (rows, W.batch(rows, device)). `prefetch` > 0: drawn and built
    up to that many batches ahead on a background thread (as CPU tensors, moved to `device` as
    they are taken) while the caller trains on the current one - the same draws, the same
    batches. A failure there is raised here; closing the generator (or dropping it) stops the
    thread."""
    if not prefetch:
        for _ in range(n):
            idx = rng.integers(W.N, size=size)
            yield idx, W.batch(idx, device)
        return
    q = queue.Queue(maxsize=prefetch)
    stop = threading.Event()

    def put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                pass
        return False

    def produce():
        try:
            for _ in range(n):
                idx = rng.integers(W.N, size=size)
                if not put((idx, W.batch(idx))):
                    return
        except BaseException as e:             # noqa: BLE001  (re-raised by the consumer)
            put(e)

    th = threading.Thread(target=produce, name='zero-batch-prefetch', daemon=True)
    th.start()
    try:
        for _ in range(n):
            while True:
                try:
                    item = q.get(timeout=1.0)
                    break
                except queue.Empty:
                    if not th.is_alive() and q.empty():
                        raise RuntimeError('the batch prefetch thread ended early')
            if isinstance(item, BaseException):
                raise item
            idx, b = item
            yield idx, (b if device == 'cpu' else {k: v.to(device) for k, v in b.items()})
    finally:
        stop.set()
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
        th.join(timeout=10)


class WindowCache:
    """Shards kept between a run's windows (module docstring): path -> (its size and mtime, the
    number of features, its part with the features of all its rows)."""

    def __init__(self):
        self.parts = {}

    def clear(self):
        self.parts.clear()

    @property
    def nbytes(self):
        return sum(a.nbytes for _, _, part in self.parts.values() for a in part.values())


class Window:
    """The training window in RAM. paths: shard files; `budget_rows`: keep at most this many
    samples (a uniform random share of every shard). A shard with a NaN / inf raises NonFinite.
    `threads`: shards loaded at once (default WINDOW_THREADS); the result does not depend on it.
    `cache`: a WindowCache (module docstring): used while the window fits its budget; `reused` =
    the rows it supplied. `aux_width` W > 0: the auxiliary targets too (module docstring; `aux_rows`
    = the rows that have them)."""

    def __init__(self, paths, num_features, num_actions, budget_rows=None, seed=0, threads=None, cache=None,
                 aux_width=0):
        self.F, self.A = num_features, num_actions
        self.AW = AW = int(aux_width)
        self.paths = list(paths)
        self.reused = 0
        self._x = self._xs = None
        rng = np.random.default_rng(seed)
        rows = [shard_rows(p) for p in self.paths]
        total = sum(rows)
        keep = 1.0 if not budget_rows or total <= budget_rows else budget_rows / total
        # Each shard's rows, drawn in path order (the random stream of loading one at a time).
        picks = [None if keep >= 1.0 else np.sort(rng.choice(N, size=max(1, int(round(N * keep))), replace=False))
                 for N in rows]
        kept = [N if idx is None else len(idx) for N, idx in zip(rows, picks)]
        self.starts = np.concatenate([[0], np.cumsum(kept)]).astype(np.int64)
        self.N = int(self.starts[-1])
        if self.N == 0:
            raise ValueError('empty training window')
        if cache is not None and keep >= 1.0:
            # (the auxiliary width only when on: the keys of before otherwise)
            keys = {path: (_file_key(path), (self.F, AW) if AW else self.F) for path in self.paths}
            for path in list(cache.parts):         # gone from the window, rewritten, another width
                if path not in keys or cache.parts[path][:2] != keys[path]:
                    del cache.parts[path]
            todo = [k for k, path in enumerate(self.paths) if path not in cache.parts]
            loaded = _run_all(lambda j: _load_part(self.paths[todo[j]], rows[todo[j]], None, self.F, AW=AW), len(todo),
                              threads)
            for k, part in zip(todo, loaded):
                cache.parts[self.paths[k]] = keys[self.paths[k]] + (part,)
            self.reused = self.N - sum(rows[k] for k in todo)
            parts = [cache.parts[path][2] for path in self.paths]
            self._xs = [q['x'] for q in parts]
        else:
            if cache is not None:
                cache.clear()                      # over budget: subsampled shards are not cached
            self._x = np.empty((self.N, self.F), np.float16)
            parts = _run_all(lambda k: _load_part(self.paths[k], rows[k], picks[k], self.F,
                                                  self._x[self.starts[k]:self.starts[k + 1]], AW=AW), len(self.paths),
                             threads)
        self.src_row = np.concatenate([q['src'] for q in parts])     # window row -> its row in its shard
        self.player = np.concatenate([q['player'] for q in parts])
        self.n_players = np.concatenate([q['n_players'] for q in parts])
        self.opt_off = np.concatenate([np.zeros(1, np.int64), np.cumsum(np.concatenate([q['lens'] for q in parts]))])
        self.opt_action = np.concatenate([q['opt_action'] for q in parts])
        self.pi = np.concatenate([q['pi'] for q in parts])
        if self.opt_action.max() >= self.A:
            raise ValueError(f'action id {self.opt_action.max()} >= the network\'s {self.A} actions')
        self.z = np.concatenate([q['z'] for q in parts])
        self.q = np.concatenate([q['q'] for q in parts])
        self.sc = np.concatenate([q['sc'] for q in parts])
        self.loaded = total
        self.aux = np.concatenate([q['aux'] for q in parts]) if AW else None
        self.aux_mask = np.concatenate([q['aux_mask'] for q in parts]) if AW else None
        self.aux_rows = int(self.aux_mask.sum()) if AW else 0
        xbytes = self._x.nbytes if self._x is not None else sum(a.nbytes for a in self._xs)
        self.nbytes = xbytes + sum(a.nbytes for a in (self.player, self.n_players, self.opt_off, self.opt_action,
                                                      self.pi, self.z, self.q, self.sc)
                                   + ((self.aux, self.aux_mask) if AW else ()))

    @property
    def x(self):
        """The features [N, F] float16 (from the cache: concatenated, a copy - tests, debugging)."""
        return self._x if self._x is not None else np.concatenate(self._xs)

    def features(self, idx):
        """The features of window rows `idx`, float16 [len(idx), F]."""
        if self._x is not None:
            return self._x[idx]
        idx = np.asarray(idx, np.int64)
        sh = np.searchsorted(self.starts, idx, side='right') - 1
        order = np.argsort(sh, kind='stable')       # the rows grouped by shard
        cut = np.flatnonzero(np.diff(sh[order])) + 1
        out = np.empty((len(idx), self.F), np.float16)
        for g in np.split(order, cut):
            k = sh[g[0]]
            out[g] = self._xs[k][idx[g] - self.starts[k]]
        return out

    def provenance(self, idx):
        """Window rows -> [(shard path, its rows in that shard, sorted)]."""
        idx = np.asarray(idx, np.int64)
        sh = np.searchsorted(self.starts, idx, side='right') - 1
        return [(self.paths[k], np.unique(self.src_row[idx[sh == k]])) for k in np.unique(sh)]

    def batch(self, idx, device='cpu'):
        B = len(idx)
        off = self.opt_off
        starts = off[idx]
        lens = off[idx + 1] - starts
        pos = _ranges(starts, lens)
        rows = np.repeat(np.arange(B), lens)
        acts = self.opt_action[pos]
        mask = np.zeros((B, self.A), bool)
        pi = np.zeros((B, self.A), np.float32)
        mask[rows, acts] = True
        pi[rows, acts] = self.pi[pos]
        pi /= np.maximum(pi.sum(1, keepdims=True), 1e-8)

        def norm(a):
            return a / np.maximum(a.sum(1, keepdims=True), 1e-8)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        out = {'x': t(self.features(idx).astype(np.float32)), 'mask': t(mask), 'pi': t(pi),
               'z': t(norm(self.z[idx])), 'q': t(norm(self.q[idx])), 'sc': t(self.sc[idx]),
               'n_players': t(self.n_players[idx])}
        if self.AW:
            out['aux'] = t(self.aux[idx].astype(np.float32))
            out['aux_mask'] = t(self.aux_mask[idx].astype(np.float32))
        return out
