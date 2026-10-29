"""Training data (docs/az.md 3): sample shards and the RAM-bounded training window.

Shards: `data/iter_NNNN/<task>_SSS.npz` (uncompressed npz, written to a temporary name and
renamed: a crash never leaves a half-written shard; remote self-play's shards, `m<KK>-<sid>_SSS.npz`,
are compressed npz - zlib level 1, ~58x for MTG's sparse features - which `np.load` reads alike:
docs/az.md 5.3). Arrays: `x` float16 [N, F], `player`,
`n_players` int8, `opt_off` int32 [N+1], `opt_action` int16/int32, `pi` float16, `z` / `q` /
`sc` float16 [N, P] (P = most players in the shard), `net` int32, and `meta` (a JSON string:
game, features, actions, rows, task info). Float16 targets are renormalised on load.
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
targets over the actions, value / margin targets over MAX_SEATS. The shards load in parallel
(`threads`, default `WINDOW_THREADS`): inflating a compressed shard, its CRC and the copies
release the GIL, so the window of MTG's Delver run (40 shards, 265k rows, 17.9 GB) loads in a
fraction of the 41 s it took one shard at a time. Each shard's rows are drawn in path order
before any loading (the random stream of the sequential load) and its features written into
their place in the one preallocated `x`: the window is the same, array for array, and the peak
is the window plus the shards being loaded (it was twice the window: the per-shard copies, then
their concatenation).
"""
import glob
import json
import os
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
            'sc': s['sc'][:, :P].astype(np.float16), 'net': s['net'].astype(np.int32)}


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


class Window:
    """The training window in RAM. paths: shard files; `budget_rows`: keep at most this many
    samples (a uniform random share of every shard). A shard with a NaN / inf raises NonFinite.
    `threads`: shards loaded at once (default WINDOW_THREADS); the result does not depend on it."""

    def __init__(self, paths, num_features, num_actions, budget_rows=None, seed=0, threads=None):
        self.F, self.A = num_features, num_actions
        self.paths = list(paths)
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
        self.x = np.empty((self.N, self.F), np.float16)

        def load(k):
            path, idx, N = self.paths[k], picks[k], rows[k]
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
            w = min(self.F, p['x'].shape[1])
            dst = self.x[self.starts[k]:self.starts[k + 1]]
            dst[:, :w] = p['x'][sel, :w]
            if w < self.F:
                dst[:, w:] = 0
            off = p['opt_off'].astype(np.int64)
            lens = off[idx + 1] - off[idx]
            pos = _ranges(off[idx], lens)
            out = {'src': idx.astype(np.int32), 'player': p['player'][sel].astype(np.int64),
                   'n_players': p['n_players'][sel].astype(np.int64), 'lens': lens,
                   'opt_action': p['opt_action'][pos].astype(np.int64), 'pi': p['pi'][pos].astype(np.float32)}
            for key in ('z', 'q', 'sc'):
                a = np.zeros((len(idx), MAX_SEATS), np.float32)
                v = p[key][sel].astype(np.float32)
                a[:, :v.shape[1]] = v
                out[key] = a
            return out

        n_threads = max(1, min(threads or WINDOW_THREADS, len(self.paths)))
        if n_threads == 1:
            parts = [load(k) for k in range(len(self.paths))]
        else:
            with ThreadPoolExecutor(n_threads) as ex:
                futs = [ex.submit(load, k) for k in range(len(self.paths))]
                parts = [f.result() for f in futs]          # the first failure in path order is raised
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
        self.nbytes = sum(a.nbytes for a in (self.x, self.player, self.n_players, self.opt_off, self.opt_action,
                                             self.pi, self.z, self.q, self.sc))

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
        return {'x': t(self.x[idx].astype(np.float32)), 'mask': t(mask), 'pi': t(pi),
                'z': t(norm(self.z[idx])), 'q': t(norm(self.q[idx])), 'sc': t(self.sc[idx]),
                'n_players': t(self.n_players[idx])}
