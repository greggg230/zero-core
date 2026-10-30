"""Held-out losses (docs/az.md 20.2): a checkpoint's value and policy losses on a FROZEN set of sample
rows no checkpoint trains on - a cheap learning signal beside the games (MTG Zero's Delver line,
2026-10-30: its Elo pool and its saturated yardstick could not see its progress).

  python -m zero.heldout make --from runs/x/data/iter_0191 --rows 8192 --out heldout/x_191.npz
  python -m zero.heldout eval --shards heldout/x_191.npz --ckpt runs/x/models/iter_0100.pt [...]
  (a game repo: python -m <pkg>.az.heldout ...)

The set: shards in the run's own format (zero/data.py), usually one built by `make` - a uniform random
subset (fixed seed) of an iteration's self-play rows, written as one compressed shard - so the rows are
fixed for good (the run's data/ directory keeps only `keep_data` iterations). Iteration i's rows train
checkpoints i .. i + window - 1: they are held out for every other checkpoint (the run's next ones
included), and `eval` / the run say which checkpoints saw them only if told (`seen`).

Per kind of row (`losses`): **decision** rows (the recorded searches: options and a policy target) and
**view** rows (value-only samples of the other seat's view, docs/az.md 16: no options), each with
  value_ce     the value head's cross-entropy against the game's outcome z (nats; the training loss
               with value_q 0), mean over the rows
  value_acc    the share of rows whose most likely seat won (rows with one winner only: a draw has none)
  value_brier  the squared error of P(win) against z, mean over the seats present
and the decision rows also
  policy_ce    the cross-entropy of the policy over the legal actions against the recorded target
               (the search's visits or completed-Q target), mean over the decisions with >= 2 options
  policy_top1  the share of those whose policy argmax is the target's argmax
The numbers of the same rows are compared across checkpoints (a paired comparison: the rows' own
difficulty cancels), never across sets: `set` names the set (a hash of the shards' names, sizes and
rows) in every report.

Run settings (zero/run.py): `heldout_shards` (paths, a list or comma-separated; '' = off, the default:
nothing changes) - every evaluated iteration of the run gets `heldout` in its metrics row and a log
line, computed in the run process (also with eval_remote: a few seconds on `heldout_device`, '' = the
run's train_device). The set is loaded once (its rows stay in RAM: 8,192 MTG rows ~ 0.6 GB) and must be
of the run's game and encoding (its meta's features / actions = the network's: else refused).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

from .data import Window, read_shard, shard_paths, shard_rows, write_shard
from .protocol import MAX_SEATS


def expand(paths):
    """Shard paths: files as given, a directory -> its loadable shards (data.shard_paths)."""
    if isinstance(paths, str):
        paths = [p for p in paths.split(',') if p.strip()]
    out = []
    for p in paths:
        p = str(p).strip()
        out += shard_paths(p) if os.path.isdir(p) else [p]
    return out


def shard_meta(path):
    with np.load(path) as z:
        return json.loads(str(z['meta']))


def set_name(paths):
    """A short name of a held-out set: the hash of its shards' file names, sizes and rows."""
    h = hashlib.sha256()
    for p in paths:
        h.update(f'{os.path.basename(p)}|{os.path.getsize(p)}|{shard_rows(p)};'.encode())
    return h.hexdigest()[:12]


class HeldOut:
    """A frozen held-out set in RAM (module docstring): `paths` (shards or directories); the network's
    `features` / `actions` must be the shards' (their meta), `game` too when the meta names one."""

    def __init__(self, paths, game=None, features=None, actions=None):
        self.paths = expand(paths)
        if not self.paths:
            raise ValueError('held-out set: no shards')
        missing = [p for p in self.paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f'held-out shards missing: {missing}')
        metas = [shard_meta(p) for p in self.paths]
        F = {m.get('features') for m in metas}
        A = {m.get('actions') for m in metas}
        games = {m.get('game') for m in metas} - {None}
        if len(F) != 1 or len(A) != 1 or len(games) > 1:
            raise ValueError(f'held-out shards of different games or encodings: features {sorted(map(str, F))}, '
                             f'actions {sorted(map(str, A))}, games {sorted(games)}')
        self.features, self.actions = F.pop(), A.pop()
        self.game = games.pop() if games else None
        if game is not None and self.game is not None and game != self.game:
            raise ValueError(f'held-out shards of the game {self.game!r}, not {game!r}')
        self.check(features, actions)
        self.W = Window(self.paths, self.features, self.actions)
        self.name = set_name(self.paths)
        self.sources = sorted({m.get('heldout', {}).get('from', m.get('iter')) for m in metas}, key=str)

    def check(self, features=None, actions=None, what='the network'):
        """The network's width must be the set's (a set of another encoding would be cut / padded in silence)."""
        if features is not None and int(features) != int(self.features):
            raise ValueError(f'held-out shards of {self.features} features, {what} has {features} '
                             f'(another encoding: build a set with this build)')
        if actions is not None and int(actions) != int(self.actions):
            raise ValueError(f'held-out shards of {self.actions} actions, {what} has {actions}')

    def evaluate(self, model, device='cpu', batch=1024):
        """`losses` of `model` (a net.load-ed network) on the set -> dict (+ rows, set, seconds)."""
        from .distill import outputs
        self.check(model.features, model.config.get('actions'), 'the checkpoint')
        t0 = time.time()
        out = losses(outputs(model, self.W, device, batch=batch), self.W)
        out.update(set=self.name, seconds=round(time.time() - t0, 2))
        return out


def losses(O, W):
    """Per kind of row ('decision', 'view' when the set has value-only rows): value_ce, value_acc,
    value_brier, and for the decisions policy_ce / policy_top1 (module docstring); O = distill.outputs
    of a network on window W. -> {'rows', 'decision': {...}, 'view': {...}}."""
    eps = 1e-12
    lens = np.diff(W.opt_off)
    n = W.n_players
    present = np.arange(MAX_SEATS)[None, :] < n[:, None]
    z = W.z.astype(np.float64)
    z = z / np.maximum(z.sum(1, keepdims=True), 1e-8)
    v = O['v'].astype(np.float64)
    vce = -(z * np.log(np.maximum(v, eps)) * present).sum(1)
    brier = (((v - z) ** 2) * present).sum(1) / np.maximum(n, 1)
    zmax = np.where(present, z, -1).max(1)
    one = (np.where(present, z, -1) == zmax[:, None]).sum(1) == 1        # one winner (a draw: none)
    acc = np.where(present, v, -1).argmax(1) == np.where(present, z, -1).argmax(1)
    starts = W.opt_off[:-1]
    dec = lens >= 2
    out = {'rows': int(W.N)}
    pce = top = None
    if dec.any():
        pin = W.pi.astype(np.float64)
        rows = np.repeat(np.arange(W.N), lens)
        tot = np.bincount(rows, weights=pin, minlength=W.N)
        pin = pin / np.maximum(tot[rows], 1e-8)
        ce = -np.bincount(rows, weights=pin * np.log(np.maximum(O['pi'].astype(np.float64), eps)), minlength=W.N)
        pce = ce
        # argmax of each row's options (the first maximum, as distill's): the policy's vs the target's
        top = np.zeros(W.N, bool)
        idx = np.flatnonzero(dec)
        for r in idx:
            s, e = starts[r], starts[r] + lens[r]
            top[r] = int(np.argmax(O['pi'][s:e])) == int(np.argmax(W.pi[s:e]))

    def mean(a, m):
        return round(float(a[m].mean()), 5) if m.any() else None

    kinds = {'decision': lens > 0, 'view': lens == 0}
    for kind, km in kinds.items():
        if not km.any():
            continue
        d = {'rows': int(km.sum()), 'value_ce': mean(vce, km), 'value_acc': mean(acc.astype(float), km & one),
             'value_brier': mean(brier, km)}
        if kind == 'decision':
            dm = km & dec
            d['decisions'] = int(dm.sum())
            if pce is not None:
                d['policy_ce'] = mean(pce, dm)
                d['policy_top1'] = mean(top.astype(float), dm)
        out[kind] = d
    return out


def fmt(h):
    """A held-out result in one log line."""
    parts = []
    for kind in ('decision', 'view'):
        d = h.get(kind)
        if not d:
            continue
        s = f'{kind} value_ce {d["value_ce"]:.4f} acc {d["value_acc"]:.4f}'
        if d.get('policy_ce') is not None:
            s += f' policy_ce {d["policy_ce"]:.4f} top1 {d["policy_top1"]:.4f}'
        parts.append(s + f' ({d["rows"]} rows)')
    return f'(set {h.get("set")}): ' + '; '.join(parts)


def make(src, out, rows=8192, seed=0):
    """A frozen held-out set: a uniform random `rows` of the shards `src` (files or directories; seed
    `seed`: the Window's draw), written to `out` as one compressed shard (zlib 1) whose meta names its
    sources ('heldout': {'from', 'paths', 'rows', 'seed', 'of'}). -> (rows, bytes)."""
    paths = expand(src)
    if not paths:
        raise ValueError(f'no shards in {src}')
    metas = [shard_meta(p) for p in paths]
    F, A = metas[0]['features'], metas[0]['actions']
    if any((m.get('features'), m.get('actions'), m.get('game')) != (F, A, metas[0].get('game')) for m in metas):
        raise ValueError('shards of different games or encodings')
    aw = 0
    with np.load(paths[0]) as z0:
        if 'aux' in z0.files:
            aw = int(z0['aux'].shape[1])
    W = Window(paths, F, A, budget_rows=int(rows), seed=int(seed), aux_width=aw)
    s = {'x': W.x, 'player': W.player, 'n_players': W.n_players, 'opt_off': W.opt_off, 'opt_action': W.opt_action,
         'pi': W.pi, 'z': W.z, 'q': W.q, 'sc': W.sc, 'net': np.zeros(W.N, np.int32)}
    if aw and bool(W.aux_mask.all()):
        s['aux'] = W.aux
    iters = sorted({m.get('iter') for m in metas if m.get('iter') is not None})
    meta = {k: metas[0][k] for k in ('game', 'features', 'actions') if k in metas[0]}
    meta['heldout'] = {'from': iters[0] if len(iters) == 1 else iters, 'paths': [os.path.basename(p) for p in paths],
                       'rows': int(W.N), 'of': int(W.loaded), 'seed': int(seed)}
    os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
    return write_shard(out, s, meta, compress=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    mk = sub.add_parser('make', help='build a frozen held-out set from shards')
    mk.add_argument('--from', dest='src', nargs='+', required=True, help='shards or data/iter_NNNN directories')
    mk.add_argument('--rows', type=int, default=8192)
    mk.add_argument('--seed', type=int, default=0)
    mk.add_argument('--out', required=True)
    ev = sub.add_parser('eval', help='held-out losses of checkpoints (one JSON line each)')
    ev.add_argument('--shards', nargs='+', required=True)
    ev.add_argument('--ckpt', nargs='+', required=True)
    ev.add_argument('--device', default='cpu')
    ev.add_argument('--torch-threads', type=int, default=4)
    a = ap.parse_args(argv)
    if a.cmd == 'make':
        n, b = make(a.src, a.out, a.rows, a.seed)
        print(json.dumps({'out': a.out, 'rows': n, 'bytes': b, 'set': set_name([a.out])}), flush=True)
        return
    import torch
    from .net import load
    torch.set_num_threads(max(1, a.torch_threads))
    H = None
    for ck in a.ckpt:
        m, extra = load(ck, a.device)
        if H is None:
            H = HeldOut(a.shards, game=m.config.get('game'))
        r = H.evaluate(m, a.device)
        r['ckpt'] = ck
        if isinstance(extra, dict) and 'iter' in extra:
            r['iter'] = extra['iter']
        print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
