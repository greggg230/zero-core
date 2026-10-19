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
"""
import glob
import json
import os

import numpy as np
import torch

from .protocol import MAX_SEATS, concat_samples

EXT = '.npz'


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
        out = {'x': t(self.x[idx].astype(np.float32)), 'mask': t(mask), 'pi': t(pi),
               'z': t(norm(self.z[idx])), 'q': t(norm(self.q[idx])), 'sc': t(self.sc[idx]),
               'n_players': t(self.n_players[idx])}
        if self.AW:
            out['aux'] = t(self.aux[idx].astype(np.float32))
            out['aux_mask'] = t(self.aux_mask[idx].astype(np.float32))
        return out
