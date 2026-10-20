"""Training data (docs/az.md 3): sample shards and the RAM-bounded training window.

Shards: `data/iter_NNNN/<task>_SSS.npz` (uncompressed npz, written to a temporary name and
renamed: a crash never leaves a half-written shard). Arrays: `x` float16 [N, F], `player`,
`n_players` int8, `opt_off` int32 [N+1], `opt_action` int16/int32, `pi` float16, `z` / `q` /
`sc` float16 [N, P] (P = most players in the shard), `net` int32, optionally `aux` float16
[N, W] (the game's auxiliary targets, `SearchGame::AUX`; self-play with aux targets on), and
`meta` (a JSON string: game, features, actions, rows, task info). Float16 targets are
renormalised on load.

`TargetStats`: sharpness of the recorded policy targets (self-play metrics).

Window: the newest iterations' shards loaded into RAM (`Window(paths, ...)`), every shard
subsampled at the same rate when the total exceeds the RAM budget; `batch(idx)` builds
torch tensors: features cut / zero-padded to the network's width, dense legal masks and
policy targets over the actions, value / margin targets over MAX_SEATS; with `aux_width` W > 0
also the auxiliary targets `aux` [B, W] and `aux_mask` [B] (1 = the row has them: rows of
shards written without aux targets - e.g. a fork's older window - are zeros with mask 0, so
the auxiliary losses skip them).

Direct reads (`ShardFile`, run setting window_load 'direct' / 'prefetch', docs/az.md 4.2): np.load
reads an npz member through zipfile (256 KB chunks + CRC32, ~300 MB/s of shard) and `Window(paths)`
holds every shard whole before subsampling (Kingdom Builder: ~33 GB for an 8 GB window). A shard
is an uncompressed zip (np.savez), so each array's bytes sit at a fixed file offset (`npz_layout`):
`ShardFile.read_rows` reads only the kept rows of `x` / `aux` straight into the destination
(contiguous runs, one read each). `sample_rows`: a shard's kept rows = a prefix of a random
permutation of its rows (seeded by the training iteration and the shard's name), so the rows for
a smaller keep are a subset of those for a larger one - a window planned before the newest
iteration's shards exist (`WindowPrefetch`, during self-play) is completed later by reading only
what is missing (`WindowCache.fetch`). `WindowCache.window(plan)` -> `ShardWindow` (the same
`batch` and statistics as `Window`; `x` stays in per-shard chunks: no 8 GB concatenation).
"""
import glob
import json
import os
import struct
import threading
import time
import traceback
import zipfile
import zlib

import numpy as np
import torch

from .protocol import MAX_SEATS, concat_samples

EXT = '.npz'
READ_MB = 64                     # ShardFile.read_rows: at most this much per read call (stop checks between)
SMALL = ('player', 'n_players', 'opt_off', 'opt_action', 'pi', 'z', 'q', 'sc')


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


def write_shard(path, s, meta):
    c = compact(s)
    meta = dict(meta, rows=int(len(c['player'])))
    tmp = path + f'.{os.getpid()}.tmp.npz'
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
    return meta['rows'], os.path.getsize(path)


def read_shard(path):
    with np.load(path) as z:
        out = {k: z[k] for k in z.files if k != 'meta'}
        out['meta'] = json.loads(str(z['meta']))
    return out


def shard_paths(d):
    return sorted(glob.glob(os.path.join(d, '*' + EXT)))


class ShardWriter:
    """Collects driver sample dicts and writes a shard every `rows` samples (and at close)."""

    def __init__(self, d, prefix, rows=65536, meta=None):
        os.makedirs(d, exist_ok=True)
        self.d, self.prefix, self.rows, self.meta = d, prefix, rows, dict(meta or {})
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
        path = os.path.join(self.d, f'{self.prefix}_{self.k:03d}{EXT}')
        rows, nbytes = write_shard(path, s, self.meta)
        self.k += 1
        self.samples += rows
        self.bytes += nbytes
        self.shards += 1

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


class Window:
    """The training window in RAM. paths: shard files; `budget_rows`: keep at most this many
    samples (a uniform random share of every shard); `aux_width` > 0: load the auxiliary
    targets (a shard's `aux` must have that width; shards without them give masked rows)."""

    def __init__(self, paths, num_features, num_actions, budget_rows=None, seed=0, aux_width=0):
        self.F, self.A = num_features, num_actions
        self.AW = int(aux_width)
        rng = np.random.default_rng(seed)
        parts = [read_shard(p) for p in paths]
        total = sum(len(p['player']) for p in parts)
        keep = 1.0 if not budget_rows or total <= budget_rows else budget_rows / total
        xs, pl, npl, offs, acts, pis, zs, qs, scs = [], [], [], [np.zeros(1, np.int64)], [], [], [], [], []
        auxs, amask = [], []
        base = 0
        for p in parts:
            N = len(p['player'])
            idx = np.arange(N) if keep >= 1.0 else np.sort(rng.choice(N, size=max(1, int(round(N * keep))), replace=False))
            x = p['x'][idx]
            if x.shape[1] > self.F:
                x = x[:, :self.F]
            elif x.shape[1] < self.F:
                x = np.pad(x, ((0, 0), (0, self.F - x.shape[1])))
            xs.append(x.astype(np.float16))
            pl.append(p['player'][idx].astype(np.int64))
            npl.append(p['n_players'][idx].astype(np.int64))
            off = p['opt_off'].astype(np.int64)
            lens = off[idx + 1] - off[idx]
            pos = _ranges(off[idx], lens)
            acts.append(p['opt_action'][pos].astype(np.int64))
            pis.append(p['pi'][pos].astype(np.float32))
            offs.append(np.cumsum(lens) + base)
            base += int(lens.sum())
            for dst, k in ((zs, 'z'), (qs, 'q'), (scs, 'sc')):
                a = np.zeros((len(idx), MAX_SEATS), np.float32)
                v = p[k][idx].astype(np.float32)
                a[:, :v.shape[1]] = v
                dst.append(a)
            if self.AW:
                if 'aux' in p:
                    if p['aux'].shape[1] != self.AW:
                        raise ValueError(f'shard aux targets of width {p["aux"].shape[1]}, the window wants {self.AW}')
                    auxs.append(p['aux'][idx].astype(np.float16))
                    amask.append(np.ones(len(idx), bool))
                else:
                    auxs.append(np.zeros((len(idx), self.AW), np.float16))
                    amask.append(np.zeros(len(idx), bool))
        self.N = sum(len(x) for x in xs)
        if self.N == 0:
            raise ValueError('empty training window')
        self.x = np.concatenate(xs)
        self.player = np.concatenate(pl)
        self.n_players = np.concatenate(npl)
        self.opt_off = np.concatenate(offs)
        self.opt_action = np.concatenate(acts)
        self.pi = np.concatenate(pis)
        if self.opt_action.max() >= self.A:
            raise ValueError(f'action id {self.opt_action.max()} >= the network\'s {self.A} actions')
        self.z = np.concatenate(zs)
        self.q = np.concatenate(qs)
        self.sc = np.concatenate(scs)
        self.loaded = total
        self.aux = np.concatenate(auxs) if self.AW else None
        self.aux_mask = np.concatenate(amask) if self.AW else None
        # rows with auxiliary targets
        self.aux_rows = int(self.aux_mask.sum()) if self.AW else 0
        self.nbytes = sum(a.nbytes for a in (self.x, self.player, self.n_players, self.opt_off, self.opt_action,
                                             self.pi, self.z, self.q, self.sc) + ((self.aux, self.aux_mask) if self.AW else ()))

    def _x(self, idx):
        """Rows idx of the features as float32 [B, F]."""
        return self.x[idx].astype(np.float32)

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
        out = {'x': t(self._x(idx)), 'mask': t(mask), 'pi': t(pi),
               'z': t(norm(self.z[idx])), 'q': t(norm(self.q[idx])), 'sc': t(self.sc[idx]),
               'n_players': t(self.n_players[idx])}
        if self.AW:
            out['aux'] = t(self.aux[idx].astype(np.float32))
            out['aux_mask'] = t(self.aux_mask[idx].astype(np.float32))
        return out


# ------------------------------------------------------------------ direct shard reads

class Stopped(Exception):
    """A read stopped by its `stop()` callable (a pause, or the prefetch being cancelled)."""


def npz_layout(path):
    """Where each array of an uncompressed npz (np.savez: every member stored) starts in the file
    -> {name: (dtype, shape, offset)}; None when a member is compressed or not a plain C-order
    .npy (`ShardFile` then reads the shard with np.load)."""
    out = {}
    with open(path, 'rb') as f:
        with zipfile.ZipFile(f) as zf:
            infos = zf.infolist()
        for zi in infos:
            if zi.compress_type != zipfile.ZIP_STORED or not zi.filename.endswith('.npy'):
                return None
            f.seek(zi.header_offset)
            h = f.read(30)
            if len(h) != 30 or h[:4] != b'PK\x03\x04':
                raise ValueError(f'{path}: no local file header for {zi.filename}')
            n, m = struct.unpack('<HH', h[26:30])
            start = zi.header_offset + 30 + n + m          # the local header's own extra field (zip64)
            f.seek(start)
            read_header = {(1, 0): np.lib.format.read_array_header_1_0,
                           (2, 0): np.lib.format.read_array_header_2_0}.get(np.lib.format.read_magic(f))
            if read_header is None:
                return None
            shape, fortran, dtype = read_header(f)
            if (fortran and len(shape) > 1) or dtype.hasobject:
                return None
            off = f.tell()
            nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if off - start + nbytes != zi.file_size:
                raise ValueError(f'{path}: member {zi.filename} holds {zi.file_size} bytes, its header says '
                                 f'{off - start + nbytes}')
            out[zi.filename[:-4]] = (dtype, tuple(shape), off)
    return out


def _readinto(f, a):
    """Fill the contiguous array `a` from the file's current position."""
    mv = memoryview(a.reshape(-1).view(np.uint8))
    got = 0
    while got < len(mv):
        k = f.readinto(mv[got:])
        if not k:
            raise EOFError(f'{f.name}: {len(mv) - got} bytes missing')
        got += k


class ShardFile:
    """A shard opened for direct reads (`npz_layout`): `meta`, `rows`, `has(name)`, `width(name)`,
    `read(name)` (a whole array) and `read_rows(name, rows)` (only those rows of an [N, ...] array:
    Kingdom Builder keeps ~12.5k of a shard's ~52k rows, 0.5 of its 2 GB). A shard that
    npz_layout cannot map (compressed) is read whole by np.load. A context manager (one handle)."""

    def __init__(self, path):
        self.path = path
        self.lay = npz_layout(path)
        self.f = None
        self._whole = None
        if self.lay is None:
            self._whole = read_shard(path)
            self.meta = self._whole['meta']
            self.rows = len(self._whole['player'])
        else:
            self.f = open(path, 'rb', buffering=0)
            self.meta = json.loads(str(self.read('meta')))
            self.rows = int(self.lay['player'][1][0])

    def close(self):
        f, self.f = self.f, None
        if f is not None:
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def has(self, name):
        return name in (self._whole if self._whole is not None else self.lay)

    def width(self, name):
        return (self._whole[name].shape if self._whole is not None else self.lay[name][1])[1]

    def read(self, name):
        if self._whole is not None:
            return self._whole[name]
        dt, shape, off = self.lay[name]
        buf = np.empty(int(np.prod(shape, dtype=np.int64)) * dt.itemsize, np.uint8)
        self.f.seek(off)
        _readinto(self.f, buf)
        return buf.view(dt).reshape(shape)

    def read_rows(self, name, rows, stop=None):
        """Rows `rows` (sorted, unique) of the [N, ...] array `name`: each run of consecutive rows
        is one read (at most READ_MB) straight into the result; `stop()` is asked before every
        read (True: raises Stopped)."""
        rows = np.asarray(rows, np.int64)
        if self._whole is not None:
            return self._whole[name][rows]
        dt, shape, off = self.lay[name]
        out = np.empty((len(rows),) + shape[1:], dt)
        if not len(rows):
            return out
        if rows[0] < 0 or rows[-1] >= shape[0] or (len(rows) > 1 and np.any(np.diff(rows) <= 0)):
            raise IndexError(f'{self.path}: rows of {name} must be sorted, unique and < {shape[0]}')
        rb = out[:1].nbytes
        flat = out.reshape(len(rows), -1).view(np.uint8)
        cap = max(1, (READ_MB << 20) // rb)
        cut = np.flatnonzero(np.diff(rows) != 1) + 1
        for a, b in zip([0] + cut.tolist(), cut.tolist() + [len(rows)]):
            for a2 in range(a, b, cap):
                if stop is not None and stop():
                    raise Stopped(f'{self.path}: stopped')
                self.f.seek(off + int(rows[a2]) * rb)
                _readinto(self.f, flat[a2:min(b, a2 + cap)])
        return out


def shard_rows(path):
    """Samples in a shard (from its layout: no data read)."""
    lay = npz_layout(path)
    if lay is not None:
        return int(lay['player'][1][0])
    with np.load(path) as z:
        return int(json.loads(str(z['meta']))['rows'])


def shard_key(path):
    """'iter_NNNN/<file>': a shard's name within its run (also the name of a fork's hard link)."""
    return os.path.basename(os.path.dirname(os.path.abspath(path))) + '/' + os.path.basename(path)


def keep_share(total, budget_rows):
    """The share of every shard a window keeps (Window's rule): all of it within the budget."""
    return 1.0 if not budget_rows or total <= budget_rows else budget_rows / total


def kept_count(n, keep):
    return n if keep >= 1.0 else min(n, max(1, int(round(n * keep))))


def sample_rows(n, keep, seed, key):
    """The rows (sorted) a window with the share `keep` keeps of a shard of n samples: the first
    kept_count(n, keep) of a random permutation seeded by (seed, key) - a uniform random subset
    without replacement, as Window draws them, and the rows for a smaller share are a subset of
    those for a larger one (a prefetched window is completed by reading only what is missing)."""
    m = kept_count(n, keep)
    if m >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng([int(seed) % (1 << 63), zlib.crc32(key.encode('utf-8'))])
    return np.sort(rng.permutation(n)[:m]).astype(np.int64)


class Part:
    """Rows `rows` (sorted) of one shard: `x` / `aux` as stored (the shard's width, float16), the
    other arrays in their stored dtypes, the options of those rows (`opt_off` from 0)."""

    def __init__(self, path, n, rows):
        self.path, self.n, self.rows = path, n, rows
        self.x = self.aux = None

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.x, self.aux, self.player, self.n_players, self.opt_off, self.opt_action,
                                      self.pi, self.z, self.q, self.sc) if a is not None)


def read_part(path, rows, aux_width=0, stop=None):
    """Rows `rows` (sorted, unique) of the shard `path` -> Part; with aux_width > 0 also its auxiliary
    targets if the shard has them (of that width, else ValueError, as Window)."""
    rows = np.asarray(rows, np.int64)
    with ShardFile(path) as sf:
        p = Part(path, sf.rows, rows)
        has_aux = bool(aux_width) and sf.has('aux')
        if has_aux and sf.width('aux') != aux_width:
            raise ValueError(f'shard aux targets of width {sf.width("aux")}, the window wants {aux_width}')
        s = {k: sf.read(k) for k in SMALL}
        if len(s['player']) != sf.rows or len(s['opt_off']) != sf.rows + 1:
            raise ValueError(f'{path}: inconsistent shard ({len(s["player"])} rows, {len(s["opt_off"])} offsets)')
        off = s['opt_off'].astype(np.int64)
        lens = off[rows + 1] - off[rows]
        pos = _ranges(off[rows], lens)
        p.opt_action, p.pi = s['opt_action'][pos], s['pi'][pos]
        p.opt_off = np.concatenate([np.zeros(1, np.int64), np.cumsum(lens)])
        for k in ('player', 'n_players', 'z', 'q', 'sc'):
            setattr(p, k, s[k][rows])
        p.x = sf.read_rows('x', rows, stop)
        if has_aux:
            p.aux = sf.read_rows('aux', rows, stop)
    return p


class WindowCache:
    """Shard parts read for a window: path -> disjoint Parts. `fetch(path, rows)` reads the rows not
    held yet (thread-safe: one path per thread), `window(plan, F, A)` -> ShardWindow."""

    def __init__(self, aux_width=0):
        self.aw = int(aux_width)
        self.parts = {}
        self.lock = threading.Lock()
        self.read_rows = 0
        self.read_bytes = 0

    def rows(self, path):
        ps = self.parts.get(path) or []
        if len(ps) == 1:
            return ps[0].rows
        return np.sort(np.concatenate([p.rows for p in ps])) if ps else np.zeros(0, np.int64)

    def missing(self, path, rows):
        return np.setdiff1d(rows, self.rows(path), assume_unique=True)

    def fetch(self, path, rows, stop=None):
        need = self.missing(path, rows)
        if not len(need):
            return 0
        p = read_part(path, need, self.aw, stop)
        with self.lock:
            self.parts.setdefault(path, []).append(p)
            self.read_rows += len(need)
            self.read_bytes += p.nbytes
        return len(need)

    def fetch_all(self, items, threads=1, stop=None):
        """fetch(path, rows) for every (path, rows), on up to `threads` threads -> rows read."""
        items = [(p, r) for p, r in items if len(self.missing(p, r))]
        if threads <= 1 or len(items) <= 1:
            return sum(self.fetch(p, r, stop) for p, r in items)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(min(threads, len(items)), thread_name_prefix='zero-window-read') as ex:
            return sum(ex.map(lambda it: self.fetch(it[0], it[1], stop), items))

    def keep_only(self, paths):
        """Drop the parts of shards not in `paths` -> rows dropped."""
        gone = [p for p in self.parts if p not in paths]
        n = sum(len(q.rows) for p in gone for q in self.parts[p])
        for p in gone:
            del self.parts[p]
        return n

    def held_rows(self):
        return sum(len(q.rows) for ps in self.parts.values() for q in ps)

    def window(self, plan, num_features, num_actions):
        return ShardWindow(plan, self, num_features, num_actions)


class ShardWindow(Window):
    """The training window built from shard parts (`WindowCache.window`): `plan` = [(path, n, rows)]
    in window order, every row held by the cache. The arrays, statistics and `batch` of `Window`,
    rows in the same order (shard by shard, ascending rows), except that the features stay in
    their parts' chunks (`x` builds the [N, F] array on request: tests, tools)."""

    def __init__(self, plan, cache, num_features, num_actions):
        self.F, self.A = num_features, num_actions
        self.AW = cache.aw
        self._chunks, cids = [], {}
        cid, loc = [], []
        pl, npl, offs, acts, pis, zs, qs, scs = [], [], [np.zeros(1, np.int64)], [], [], [], [], []
        auxs, amask = [], []
        base = 0
        for path, n, want in plan:
            want = np.asarray(want, np.int64)
            ps = cache.parts.get(path) or []
            if not ps:
                raise ValueError(f'window: no rows of {path} were read')
            # each wanted row's part and its position there
            allrows = np.concatenate([p.rows for p in ps])
            owner = np.concatenate([np.full(len(p.rows), j, np.int64) for j, p in enumerate(ps)])
            local = np.concatenate([np.arange(len(p.rows), dtype=np.int64) for p in ps])
            order = np.argsort(allrows, kind='stable')
            k = np.searchsorted(allrows[order], want)
            if np.any(k >= len(allrows)) or not np.array_equal(allrows[order[np.minimum(k, len(allrows) - 1)]], want):
                raise ValueError(f'window: rows of {path} missing from its parts')
            own, lo = owner[order[k]], local[order[k]]
            R = len(want)
            lens = np.zeros(R, np.int64)
            for j, p in enumerate(ps):
                m = own == j
                lens[m] = p.opt_off[lo[m] + 1] - p.opt_off[lo[m]]
            o = np.concatenate([np.zeros(1, np.int64), np.cumsum(lens)])
            a = np.empty(int(o[-1]), np.int64)
            pi = np.empty(int(o[-1]), np.float32)
            player, nplay = np.empty(R, np.int64), np.empty(R, np.int64)
            zz = [np.zeros((R, MAX_SEATS), np.float32) for _ in range(3)]
            au = np.zeros((R, self.AW), np.float16) if self.AW else None
            am = np.zeros(R, bool) if self.AW else None
            cw = np.empty(R, np.int32)
            for j, p in enumerate(ps):
                m = np.flatnonzero(own == j)
                if not len(m):
                    continue
                if id(p) not in cids:
                    cids[id(p)] = len(self._chunks)
                    self._chunks.append(p.x)
                cw[m] = cids[id(p)]
                lj = lo[m]
                dst = _ranges(o[m], lens[m])
                src = _ranges(p.opt_off[lj], lens[m])
                a[dst] = p.opt_action[src]
                pi[dst] = p.pi[src]
                player[m] = p.player[lj]
                nplay[m] = p.n_players[lj]
                for dz, k2 in zip(zz, ('z', 'q', 'sc')):
                    v = getattr(p, k2)
                    dz[m, :v.shape[1]] = v[lj]
                if self.AW and p.aux is not None:
                    au[m] = p.aux[lj]
                    am[m] = True
            cid.append(cw)
            loc.append(lo)
            pl.append(player)
            npl.append(nplay)
            offs.append(o[1:] + base)
            base += int(o[-1])
            acts.append(a)
            pis.append(pi)
            zs.append(zz[0])
            qs.append(zz[1])
            scs.append(zz[2])
            if self.AW:
                auxs.append(au)
                amask.append(am)
        self.N = sum(len(c) for c in cid)
        if self.N == 0:
            raise ValueError('empty training window')
        self._cid = np.concatenate(cid)
        self._loc = np.concatenate(loc)
        self._wide = all(c.shape[1] >= self.F for c in self._chunks)
        self.player = np.concatenate(pl)
        self.n_players = np.concatenate(npl)
        self.opt_off = np.concatenate(offs)
        self.opt_action = np.concatenate(acts)
        self.pi = np.concatenate(pis)
        if self.opt_action.max() >= self.A:
            raise ValueError(f"action id {self.opt_action.max()} >= the network's {self.A} actions")
        self.z = np.concatenate(zs)
        self.q = np.concatenate(qs)
        self.sc = np.concatenate(scs)
        self.loaded = int(sum(n for _, n, _ in plan))
        self.aux = np.concatenate(auxs) if self.AW else None
        self.aux_mask = np.concatenate(amask) if self.AW else None
        self.aux_rows = int(self.aux_mask.sum()) if self.AW else 0
        self.chunk_rows = int(sum(len(c) for c in self._chunks))     # rows held (>= N: a prefetch's spare rows)
        self.nbytes = sum(c.nbytes for c in self._chunks) + sum(
            a.nbytes for a in (self._cid, self._loc, self.player, self.n_players, self.opt_off, self.opt_action,
                               self.pi, self.z, self.q, self.sc) + ((self.aux, self.aux_mask) if self.AW else ()))

    def _gather(self, idx, dtype):
        idx = np.asarray(idx, np.int64)
        out = (np.empty if self._wide else np.zeros)((len(idx), self.F), dtype)
        cid, loc = self._cid[idx], self._loc[idx]
        order = np.argsort(cid, kind='stable')
        cs = cid[order]
        cut = np.flatnonzero(np.diff(cs)) + 1
        for a, b in zip([0] + cut.tolist(), cut.tolist() + [len(idx)]):
            if a == b:
                continue
            ch = self._chunks[int(cs[a])]
            w = min(ch.shape[1], self.F)
            r = order[a:b]
            out[r, :w] = ch[loc[r], :w]
        return out

    def _x(self, idx):
        return self._gather(idx, np.float32)

    @property
    def x(self):
        """The window's features [N, F] float16 (a copy: tests and tools)."""
        return self._gather(np.arange(self.N), np.float16)


class WindowPrefetch:
    """Reads a window's shard parts on a background thread (run.py: during self-play) into a
    WindowCache. `plan`: [(path, rows)], or a callable returning it (called on the thread: listing
    and opening the shards happens there too). `stop()`: the thread stops before its next read (at
    most READ_MB) and ends; `stop_also()` (e.g. the run's PAUSE watch) is asked as well. `wait()`
    joins the thread -> seconds waited. Failures never escape the thread: `error` holds the
    traceback (the reader of the cache reads whatever is missing)."""

    def __init__(self, plan, aux_width=0, stop_also=None, name='zero-window-prefetch'):
        self.cache = WindowCache(aux_width)
        self.evt = threading.Event()
        self.stop_also = stop_also
        self.error = None
        self.stopped = False
        self.items = None
        self.t0 = time.time()
        self.t1 = None
        self.th = threading.Thread(target=self._run, args=(plan,), daemon=True, name=name)
        self.th.start()

    def _halt(self):
        return self.evt.is_set() or (self.stop_also is not None and bool(self.stop_also()))

    def _run(self, plan):
        try:
            self.items = plan() if callable(plan) else list(plan)
            for path, rows in self.items:
                if self._halt():
                    raise Stopped('prefetch stopped')
                self.cache.fetch(path, rows, stop=self._halt)
        except Stopped:
            self.stopped = True
        except BaseException:                      # noqa: BLE001 - reported by the reader
            self.error = traceback.format_exc()
        finally:
            self.t1 = time.time()

    @property
    def done(self):
        return self.t1 is not None

    @property
    def seconds(self):
        return (self.t1 or time.time()) - self.t0

    def wait(self, timeout=None):
        t = time.time()
        self.th.join(timeout)
        return time.time() - t

    def stop(self, timeout=30.0):
        """Stop and join -> True if the thread has ended."""
        self.evt.set()
        self.th.join(timeout)
        return not self.th.is_alive()
