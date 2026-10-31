"""Throughput measurements (docs/az.md 7).

  python -m <pkg>.az.bench search   [--threads 4 --games 400 --sims 64]   # Rust driver, constant evaluator
  python -m <pkg>.az.bench infer    [--device cpu|cuda --hidden 128 --blocks 3 --arch mlp|tokens | --model PATH]
                                    [--shard S ... --batches 96,192 --workers 4 --seconds 3 --ops]
  python -m <pkg>.az.bench train    [--hidden 256 --blocks 6 --arch tokens | --model PATH] --shard S [--batch 512 --steps 5]
  python -m <pkg>.az.bench selfplay [--model PATH | --hidden H --blocks B --arch A] [--game caverna --spec {...}] [--threads 4 --games 400]
(`python -m zero.bench ...` in zero-core: the toy game.)

`infer` options (all opt-in: without them the measurement is the one it always was): `--batches`
the batch sizes, `--workers k` k batches evaluated concurrently on Python threads, each with
`--torch-threads` intra-op threads (run setting `infer_workers`: a CPU self-play's shape),
`--seconds` per batch size, `--shard` (repeatable) real positions, `--model` a checkpoint instead
of a fresh network of the given size, `--ops` the operators one forward pass dispatches and its
multiply-adds (eager CUDA inference of a token network is bound by its operators, not by its
arithmetic: docs/az.md 10.5). `train`: seconds per training step (forward, backward, AdamW) at
`--batch` on real positions, and the operators of one step. `infer` / `selfplay --graphs 0 | 1`: the
evaluator's CUDA graphs (default: NetEvaluator's - a conv network's on, a token network's env
ZERO_TOKEN_GRAPHS=1; docs/az.md 10.6); the output then names `graphs` (the share of batches through them).
"""
import argparse
import json
import time

import numpy as np
import torch

from . import games as G
from .infer import NetEvaluator, UniformEvaluator, drive
from .net import load, make_net


def bench_search(game='toy', threads=4, games=400, sims=64, conc=256, groups=2, spec=None):
    seat = G.nn_seat('m', sims, selfplay=True, sims_fast=sims // 4, p_full=0.25)
    drv = G.Driver(game, [[seat, seat]] * games, spec=spec, concurrency=conc, threads=threads, groups=groups, record=True)
    ev = UniformEvaluator()
    t0 = time.time()
    drive(drv, {'m': ev}, on_batch=lambda lb: drv.take_samples())
    secs = time.time() - t0
    st = drv.stats()
    return {'what': 'search (uniform evaluator)', 'threads': threads, 'games': games, 'seconds': round(secs, 2),
            'sims_per_s': round(st['sims'] / secs), 'leaves_per_s': round(st['leaves'] / secs),
            'search_busy_s': round(st['busy_s'], 2), 'us_per_sim_thread': round(st['busy_s'] / max(1, st['sims']) * 1e6, 2),
            'games_per_s': round(games / secs, 1)}


# operators that only make views or read metadata (no kernel on a device); `item` / `nonzero`
# synchronise a CUDA stream with the host
VIEWS = {'view', 'reshape', '_unsafe_view', 'slice', 'select', 'unsqueeze', 'squeeze', 'permute', 'transpose', 'unbind',
         'flatten', 'expand', 'alias', 't', 'detach', 'split', 'chunk', 'narrow', 'as_strided', 'view_as', 'unflatten'}
SYNCS = {'item', 'nonzero', 'nonzero_numpy', '_local_scalar_dense'}


class OpCount:
    """A dispatch mode counting the aten operators run under it (`ops`; `kernels`: those that are
    not views; `syncs`: those that wait for the device) and the multiply-adds of the matrix
    products and the attention (`macs`)."""

    def __new__(cls):
        from torch.utils._python_dispatch import TorchDispatchMode

        class _Count(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.ops = self.kernels = self.syncs = self.macs = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                self.ops += 1
                name = str(func.overloadpacket).split('.')[-1]
                self.kernels += name not in VIEWS
                self.syncs += name in SYNCS
                if name in ('mm', 'bmm', 'matmul'):
                    self.macs += int(np.prod(args[0].shape)) * int(args[1].shape[-1])
                elif name == 'addmm':
                    self.macs += int(np.prod(args[1].shape)) * int(args[2].shape[-1])
                elif name == 'linear':
                    self.macs += int(np.prod(args[0].shape)) * int(args[1].shape[0])
                elif 'scaled_dot_product' in name:
                    self.macs += 2 * int(np.prod(args[0].shape)) * int(args[1].shape[-2])
                return out
        return _Count()


def count_ops(model, x, n):
    """{ops, kernels, syncs, macs} of one forward pass of `model` on rows `x` (inference mode;
    `OpCount`)."""
    c = OpCount()
    with torch.inference_mode(), c:
        model(x, n)
    return {'ops': c.ops, 'kernels': c.kernels, 'syncs': c.syncs, 'macs': c.macs}


def _net(game, model, arch, hidden, blocks, heads, ff):
    """A checkpoint's network (`model`) or a fresh one of the given size -> (net, description)."""
    if model:
        m, _ = load(model, game=game)
    else:
        m = make_net(G.info(game), game, arch, hidden, blocks, heads=heads, ff=ff)
    c = m.config
    arch = c.get('arch', 'mlp')
    d = {'arch': arch, 'hidden': c.get('d', c.get('hidden', c.get('ch'))), 'blocks': c.get('layers', c.get('blocks')),
         'params': m.n_params()}
    if arch == 'tokens':
        d.update(heads=c['heads'], ff=c['ff'])
    return m, d


def _graphs_out(out, evs, graphs):
    if graphs is not None:
        calls = sum(e.calls for e in evs)
        out['graphs'] = round(sum(e.graph_calls for e in evs) / calls, 4) if calls else None


def bench_infer(game='toy', device='cpu', hidden=128, blocks=3, batches=(64, 256, 1024), seconds=1.0, arch='mlp',
                heads=4, ff=0, xs=None, model=None, workers=1, ops=False, graphs=None):
    """xs: feature rows to sample the batches from (real positions; a token network's cost
    depends on how many objects they hold), else random normal features. `model`: a checkpoint
    (else a fresh network of the given size); `workers` > 1: that many batches evaluated at once
    on Python threads (each its own rows, `torch_threads` intra-op threads each: run setting
    infer_workers); `ops`: the operators and multiply-adds per leaf of one forward pass per batch
    size (`count_ops`); `graphs`: the evaluators' CUDA graphs (None: their default)."""
    info = G.info(game)
    m, desc = _net(game, model, arch, hidden, blocks, heads, ff)
    out = dict({'what': 'inference', 'device': device}, **desc, torch_threads=torch.get_num_threads(), leaves_per_s={})
    if workers > 1:
        out['workers'] = workers
    if ops:
        out['ops'], out['macs_per_leaf'] = {}, {}
    rng = np.random.default_rng(0)
    evs = [NetEvaluator(m, device, graphs=graphs) for _ in range(max(1, workers))]
    for B in batches:
        def rows():
            if xs is None:
                return rng.normal(size=(B, info['num_features'])).astype(np.float32)
            return np.ascontiguousarray(xs[rng.choice(len(xs), B)], dtype=np.float32)
        n = np.full(B, 2, np.int32)
        k = 6
        off = np.arange(B + 1, dtype=np.int32) * k
        act = np.tile(np.arange(k, dtype=np.int32), B)
        if workers <= 1:
            ev, x = evs[0], rows()
            if ops:
                dev = next(m.parameters()).device
                o = count_ops(m, torch.from_numpy(x).to(dev), torch.from_numpy(n.astype(np.int64)).to(dev))
                out['macs_per_leaf'][B] = round(o.pop('macs') / B)
                out['ops'][B] = o
            for _ in range(3):
                ev.evaluate(x, n, off, act)
            t0 = time.time()
            c = 0
            while time.time() - t0 < seconds:
                ev.evaluate(x, n, off, act)
                c += 1
            if device == 'cuda':
                torch.cuda.synchronize()
            out['leaves_per_s'][B] = round(B * c / (time.time() - t0))
            _graphs_out(out, evs, graphs)
            continue
        import threading
        xb = [rows() for _ in evs]
        if ops:
            dev = next(m.parameters()).device
            o = count_ops(m, torch.from_numpy(xb[0]).to(dev), torch.from_numpy(n.astype(np.int64)).to(dev))
            out['macs_per_leaf'][B] = round(o.pop('macs') / B)
            out['ops'][B] = o
        counts = [0] * len(evs)
        go = threading.Barrier(len(evs) + 1)
        stop = threading.Event()

        def work(j):
            for _ in range(2):
                evs[j].evaluate(xb[j], n, off, act)
            go.wait()
            while not stop.is_set():
                evs[j].evaluate(xb[j], n, off, act)
                counts[j] += 1
        ths = [threading.Thread(target=work, args=(j,), daemon=True) for j in range(len(evs))]
        for t in ths:
            t.start()
        go.wait()
        t0 = time.time()
        c0 = sum(counts)
        time.sleep(seconds)
        done = sum(counts) - c0
        secs = time.time() - t0
        stop.set()
        for t in ths:
            t.join()
        out['leaves_per_s'][B] = round(B * done / secs)
    _graphs_out(out, evs, graphs)
    return out


def bench_train(xs, game='toy', device='cpu', hidden=128, blocks=3, arch='mlp', heads=4, ff=0, model=None, batch=512,
                steps=5):
    """Seconds per training step (forward, a policy / value loss, backward, AdamW) at `batch` rows of
    `xs`, and the operators of one step (`OpCount`)."""
    m, desc = _net(game, model, arch, hidden, blocks, heads, ff)
    m.to(device).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    rng = np.random.default_rng(0)
    A = m.config['actions']

    def step(x):
        plog, vlog, mp = m(x, torch.full((len(x),), 2, dtype=torch.long, device=device))
        loss = torch.logsumexp(plog, 1).mean() + torch.logsumexp(vlog, 1).mean() + (mp ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
    x = torch.from_numpy(np.ascontiguousarray(xs[rng.choice(len(xs), batch)], dtype=np.float32)).to(device)
    step(x)
    c = OpCount()
    with c:
        step(x)
    t0 = time.time()
    for _ in range(steps):
        step(x)
    if device == 'cuda':
        torch.cuda.synchronize()
    secs = (time.time() - t0) / steps
    return dict({'what': 'training step', 'device': device}, **desc, actions=A, batch=batch,
                torch_threads=torch.get_num_threads(), seconds_per_step=round(secs, 3),
                ops_per_step={'ops': c.ops, 'kernels': c.kernels, 'syncs': c.syncs}, macs_per_row=round(c.macs / batch))


def bench_selfplay(model, game='toy', threads=4, games=400, sims=64, conc=256, spec=None, hidden=128, blocks=3, device='cpu',
                   arch='mlp', heads=4, ff=0, graphs=None):
    if model:
        m, _ = load(model, game=game)
    else:                                          # a fresh (random) network of the given size
        m = make_net(G.info(game), game, arch, hidden, blocks, heads=heads, ff=ff)
    ev = NetEvaluator(m, device, graphs=graphs)
    seat = G.nn_seat('m', sims, selfplay=True, sims_fast=sims // 4, p_full=0.25)
    drv = G.Driver(game, [[seat, seat]] * games, spec=spec, concurrency=conc, threads=threads, record=True)
    t0 = time.time()
    drive(drv, {'m': ev}, on_batch=lambda lb: drv.take_samples())
    secs = time.time() - t0
    st = drv.stats()
    out = {'what': 'self-play (network)', 'game': game, 'device': device, 'params': m.n_params(), 'threads': threads,
           'games': games, 'sims': sims, 'seconds': round(secs, 2), 'sims_per_s': round(st['sims'] / secs),
           'leaves_per_s': round(st['leaves'] / secs), 'games_per_h': round(games / secs * 3600),
           'infer_share': round(ev.seconds / secs, 3), 'avg_batch': round(ev.leaves / max(1, ev.calls), 1),
           'caller_wait_s': round(st['wait_s'], 2)}
    _graphs_out(out, [ev], graphs)
    return out


def _shard_rows(paths):
    parts = []
    for p in paths:
        with np.load(p) as z:
            parts.append(z['x'][:8192])
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['search', 'infer', 'train', 'selfplay'])
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--games', type=int, default=400)
    ap.add_argument('--sims', type=int, default=64)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--blocks', type=int, default=3)
    ap.add_argument('--arch', default='mlp')
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--ff', type=int, default=0)
    ap.add_argument('--shard', action='append', default=None,
                    help="infer / train: sample the batches from this shard's features (repeatable)")
    ap.add_argument('--model', default=None, help='selfplay / infer / train: a checkpoint instead of a fresh network')
    ap.add_argument('--torch-threads', type=int, default=1)
    ap.add_argument('--game', default='toy')
    ap.add_argument('--spec', default=None)
    ap.add_argument('--conc', type=int, default=256)
    ap.add_argument('--batches', default=None, help='infer: batch sizes, e.g. 96,192 (default 64,256,1024)')
    ap.add_argument('--workers', type=int, default=1, help='infer: batches evaluated concurrently on threads')
    ap.add_argument('--seconds', type=float, default=1.0, help='infer: seconds per batch size')
    ap.add_argument('--ops', action='store_true', help='infer: operators and multiply-adds of one forward pass')
    ap.add_argument('--batch', type=int, default=512, help='train: rows per step')
    ap.add_argument('--steps', type=int, default=5, help='train: timed steps')
    ap.add_argument('--graphs', type=int, default=None, choices=[0, 1],
                    help="infer / selfplay: the evaluator's CUDA graphs (default: its own; docs/az.md 10.6)")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.torch_threads)
    if a.what == 'search':
        r = bench_search(game=a.game, threads=a.threads, games=a.games, sims=a.sims, conc=a.conc, spec=a.spec)
    elif a.what == 'infer':
        xs = _shard_rows(a.shard) if a.shard else None
        kw = {'batches': tuple(int(b) for b in a.batches.split(','))} if a.batches else {}
        r = bench_infer(game=a.game, device=a.device, hidden=a.hidden, blocks=a.blocks, arch=a.arch, heads=a.heads, ff=a.ff,
                        xs=xs, model=a.model, workers=a.workers, seconds=a.seconds, ops=a.ops,
                        graphs=None if a.graphs is None else bool(a.graphs), **kw)
    elif a.what == 'train':
        if not a.shard:
            ap.error('train: --shard (real positions) is required')
        r = bench_train(_shard_rows(a.shard), game=a.game, device=a.device, hidden=a.hidden, blocks=a.blocks, arch=a.arch,
                        heads=a.heads, ff=a.ff, model=a.model, batch=a.batch, steps=a.steps)
    else:
        r = bench_selfplay(a.model, game=a.game, threads=a.threads, games=a.games, sims=a.sims, conc=a.conc, spec=a.spec,
                           hidden=a.hidden, blocks=a.blocks, device=a.device, arch=a.arch, heads=a.heads, ff=a.ff,
                           graphs=None if a.graphs is None else bool(a.graphs))
    print(json.dumps(r))


if __name__ == '__main__':
    main()
