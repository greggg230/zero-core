"""Batched inference for the leaf-batch protocol (docs/search.md 2) and the drive loop.

`NetEvaluator(model, device)` answers a network LeafBatch: the features go through the
network in one call, the policy logits of each leaf's legal actions are gathered and
softmaxed per leaf (a segment softmax on the device), the value head is softmaxed over the
seats present. CPU and CUDA; the same object can take new weights in place (`set_weights`,
rolling self-play). `UniformEvaluator` is the no-network stand-in (tests, iteration 0 checks).

Conv networks (`net.GridNet`) on CUDA run through `GraphRunner` (ZERO_CUDA_GRAPHS=0 turns it
off): BatchNorm folded into the convolutions, channels-last activations, bfloat16, one CUDA
graph per padded batch size (multiples of 64, one shared memory pool), one pinned upload and
one download per batch, and the legal options' softmax on the CPU (numpy). Kingdom Builder,
batch 512 on a shared RTX 5090: 13.6 ms per call eager (7.3 forward, 2.1 upload, ~4 gathers,
softmax and three downloads).

`drive` keeps up to two network batches in flight when the evaluators can (`launch` /
`result`) and the driver can be polled (`poll_batch`): the next batch is uploaded and replayed
while the previous one's results come back and are softmaxed on the CPU (one CUDA stream keeps
the graphs' static inputs in order; the pinned host buffers alternate between two slots).
ZERO_INFLIGHT=1: one batch at a time, as before.

`NetEvaluator(..., workers=k)` (k > 1; run setting `infer_workers`): eager batches are evaluated
on k Python threads (torch releases the GIL), and `drive` keeps up to k batches in flight (the
driver needs as many `groups`). For small networks on a CPU, where one batch cannot keep the
cores busy (a token network's many small ops: docs/az.md 10), several concurrent batches with
few intra-op threads each go faster than one batch with many. The answers are the same as
evaluating inline (each batch is computed alone; results are submitted in launch order).

`drive(driver, evaluators, externals)` runs any driver to the end (or to the end of a rolling
phase): network batches go to `evaluators[key]`, external batches to `externals[key]` (one
payload string per request). `check()` is called after every batch (PAUSE: raises).
External batches are answered on worker threads while the other groups keep searching and
evaluating (drivers with `poll_batch`): answering them inline stalled every group for the
external player's latency at each search step. Caverna: 200 games vs the JS bot (~18 ms per
bot decision) 129 s inline -> 110 s async, same games (run.py `ext_groups`, docs/az.md 5); the
Glass Road JS bot (~0.1 s per decision): 12.4 -> 3.75 min.
"""
import copy
import os
import time

import numpy as np
import torch

from .protocol import MAX_SEATS

ROW_STEP = 64                    # CUDA graphs are captured for batch sizes padded to this step


def segment_softmax_np(logits, off):
    """Softmax of a flat float array over the segments of `off` (every segment non-empty);
    vectorized (protocol.segment_softmax is the per-segment reference)."""
    cnt = np.diff(off)
    starts = off[:-1]
    lg = logits.astype(np.float32, copy=False)
    mx = np.maximum.reduceat(lg, starts)
    e = np.exp(lg - np.repeat(mx, cnt))
    return (e / np.repeat(np.add.reduceat(e, starts), cnt)).astype(np.float32)


def _fold(model):
    """An inference copy of a GridNet: each conv + BatchNorm folded into one conv, weights and
    activations channels-last."""
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    m = copy.deepcopy(model).eval()
    m.stem = fuse_conv_bn_eval(m.stem, m.stem_n)
    m.stem_n = torch.nn.Identity()
    for b in m.blocks:
        b.c1 = fuse_conv_bn_eval(b.c1, b.n1)
        b.n1 = torch.nn.Identity()
        b.c2 = fuse_conv_bn_eval(b.c2, b.n2)
        b.n2 = torch.nn.Identity()
    m = m.to(memory_format=torch.channels_last)
    m.channels_last = True
    return m


class GraphRunner:
    """CUDA graphs of a GridNet's evaluation (see the module docstring). `evaluate` has
    NetEvaluator.evaluate's contract; `refresh()` after the source model's weights changed (the
    folded weights are rewritten in place: the captured graphs keep reading them)."""

    def __init__(self, model):
        self.base = model
        self.dev = next(model.parameters()).device
        self.net = _fold(model)
        self.F = model.features
        self.A = model.config['actions']
        self.S = model.seats
        self.graphs = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.slots = [None, None]                    # per slot: pinned in / n / out buffers
        self.busy = [None, None]                     # per slot: its launch not yet collected
        self.turn = 0

    def refresh(self):
        new = _fold(self.base).state_dict()
        with torch.no_grad():
            for k, t in self.net.state_dict().items():
                t.copy_(new[k])

    def _buffers(self, slot, Bp):
        b = self.slots[slot]
        if b is None or b[0].shape[0] < Bp:
            n = max(Bp, 1024)
            b = self.slots[slot] = (torch.empty((n, self.F), dtype=torch.float32).pin_memory(),
                                    torch.empty(n, dtype=torch.int64).pin_memory(),
                                    torch.empty((n, self.A + 2 * self.S), dtype=torch.float32).pin_memory())
        return b

    def _run(self, x, n):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            plog, vlog, mar = self.net(x, n)
        return torch.cat([plog.float(), torch.softmax(vlog.float(), 1), mar.float()], 1)

    def _graph(self, Bp):
        g = self.graphs.get(Bp)
        if g is None:
            x = torch.zeros((Bp, self.F), device=self.dev)
            n = torch.full((Bp,), 2, dtype=torch.int64, device=self.dev)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), torch.no_grad():
                for _ in range(2):                   # warm-up (cuDNN picks its kernels)
                    self._run(x, n)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(graph, pool=self.pool):
                out = self._run(x, n)
            g = self.graphs[Bp] = (graph, x, n, out)
        return g

    def launch(self, x, n_players, opt_off, opt_action):
        """Upload a batch and queue its graph and download; -> a handle for `result` (at most two
        launches outstanding: a third waits for the oldest)."""
        B = x.shape[0]
        Bp = -(-B // ROW_STEP) * ROW_STEP
        slot = self.turn
        self.turn ^= 1
        if self.busy[slot] is not None:              # the slot's buffers are still in use
            self.busy[slot].synchronize()
        pin_in, pin_n, pin_out = self._buffers(slot, Bp)
        graph, xs, ns, out = self._graph(Bp)
        pin_in[:B].numpy()[:] = x[:, :self.F] if x.shape[1] >= self.F else np.pad(x, ((0, 0), (0, self.F - x.shape[1])))
        pin_n[:B].numpy()[:] = n_players
        xs[:B].copy_(pin_in[:B], non_blocking=True)
        ns[:B].copy_(pin_n[:B], non_blocking=True)
        graph.replay()
        pin_out[:B].copy_(out[:B], non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self.busy[slot] = ev
        return (ev, slot, B, n_players, opt_off, opt_action)

    def result(self, h):
        ev, slot, B, n_players, opt_off, opt_action = h
        ev.synchronize()
        if self.busy[slot] is ev:
            self.busy[slot] = None
        o = self.slots[slot][2][:B].numpy()
        A, S = self.A, self.S
        rows = np.repeat(np.arange(B), np.diff(opt_off))
        pri = segment_softmax_np(o[rows, opt_action], opt_off)
        value = np.zeros((B, MAX_SEATS), np.float32)
        margin = np.zeros((B, MAX_SEATS), np.float32)
        value[:, :S] = o[:, A:A + S]
        margin[:, :S] = o[:, A + S:A + 2 * S]
        value[np.arange(MAX_SEATS)[None, :] >= np.asarray(n_players)[:, None]] = 0.0
        return pri, value, margin

    def evaluate(self, x, n_players, opt_off, opt_action):
        return self.result(self.launch(x, n_players, opt_off, opt_action))


def use_amp(model, device):
    """bfloat16 autocast for a conv network (`net.GridNet`) on CUDA (~1.6-2.6x the leaves/s of
    float32 at batches 512-1024); the MLP keeps float32 (faster as is)."""
    return torch.device(device).type == 'cuda' and getattr(model, 'config', {}).get('arch') == 'grid'


class NetEvaluator:
    def __init__(self, model, device='cpu', graphs=None, workers=1):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.amp = use_amp(model, self.device)
        if graphs is None:
            graphs = self.amp and os.environ.get('ZERO_CUDA_GRAPHS', '1') != '0'
        self.fast = GraphRunner(self.model) if graphs else None
        self.calls = 0
        self.leaves = 0
        self.seconds = 0.0
        # eager batches on worker threads (module docstring); None: computed in launch()
        self.workers = max(1, int(workers)) if self.fast is None else 1
        self.pool = None
        if self.workers > 1:
            import threading
            from concurrent.futures import ThreadPoolExecutor
            self.pool = ThreadPoolExecutor(self.workers, thread_name_prefix='zero-infer')
            self.lock = threading.Lock()

    def set_weights(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.eval()
        if self.fast is not None:
            self.fast.refresh()

    @torch.inference_mode()
    def evaluate(self, x, n_players, opt_off, opt_action):
        """numpy in -> (priors [options], value [B, MAX_SEATS], margin [B, MAX_SEATS]) numpy."""
        t0 = time.perf_counter()
        if self._graphable(opt_off):
            r = self.fast.evaluate(x, n_players, opt_off, opt_action)
            self.calls += 1
            self.leaves += x.shape[0]
            self.seconds += time.perf_counter() - t0
            return r
        dev = self.device
        B = x.shape[0]
        xt = torch.from_numpy(x).to(dev, non_blocking=True)
        nt = torch.from_numpy(n_players.astype(np.int64)).to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.amp):
            plog, vlog, mar = self.model(xt, nt)
        cnt = torch.from_numpy(np.diff(opt_off).astype(np.int64)).to(dev)
        rows = torch.repeat_interleave(torch.arange(B, device=dev), cnt)
        acts = torch.from_numpy(opt_action.astype(np.int64)).to(dev)
        lg = plog[rows, acts].float()
        mx = torch.full((B,), -torch.inf, device=dev).scatter_reduce(0, rows, lg, 'amax', include_self=True)
        e = torch.exp(lg - mx[rows])
        s = torch.zeros(B, device=dev).index_add_(0, rows, e)
        pri = (e / s[rows]).cpu().numpy().astype(np.float32)
        val = torch.softmax(vlog.float(), dim=1)
        seats = self.model.seats
        value = np.zeros((B, MAX_SEATS), np.float32)
        margin = np.zeros((B, MAX_SEATS), np.float32)
        value[:, :seats] = val.cpu().numpy()
        margin[:, :seats] = mar.float().cpu().numpy()
        value[np.arange(MAX_SEATS)[None, :] >= n_players[:, None]] = 0.0
        if self.pool is not None:
            with self.lock:
                self.calls += 1
                self.leaves += B
                self.seconds += time.perf_counter() - t0
        else:
            self.calls += 1
            self.leaves += B
            self.seconds += time.perf_counter() - t0
        return pri, value, margin

    def __call__(self, lb):
        return self.evaluate(lb.x, lb.n_players, lb.opt_off, lb.opt_action)

    def _graphable(self, opt_off):
        return self.fast is not None and len(opt_off) > 1 and bool(np.all(np.diff(opt_off) > 0))

    def launch(self, lb):
        """Start evaluating a LeafBatch; `result(handle)` -> what `ev(lb)` returns. Eager
        evaluators compute at once (or on a worker thread, `workers` > 1)."""
        if self.pool is not None:
            return ('f', self.pool.submit(self.evaluate, lb.x, lb.n_players, lb.opt_off, lb.opt_action), 0)
        t0 = time.perf_counter()
        if self._graphable(lb.opt_off):
            h = ('g', self.fast.launch(lb.x, lb.n_players, lb.opt_off, lb.opt_action), lb.B)
        else:
            h = ('d', self.evaluate(lb.x, lb.n_players, lb.opt_off, lb.opt_action), 0)
        self.seconds += time.perf_counter() - t0
        return h

    def result(self, h):
        kind, v, B = h
        if kind == 'f':
            return v.result()
        if kind == 'd':
            return v
        t0 = time.perf_counter()
        r = self.fast.result(v)
        self.calls += 1
        self.leaves += B
        self.seconds += time.perf_counter() - t0
        return r


class UniformEvaluator:
    """Uniform priors, value 1/n for every seat present, margin 0."""

    def __call__(self, lb):
        off = lb.opt_off
        cnt = np.diff(off)
        pri = np.repeat((1.0 / cnt).astype(np.float32), cnt)
        B = lb.B
        n = lb.n_players.astype(np.float32)[:, None]
        value = ((np.arange(MAX_SEATS)[None, :] < n) / n).astype(np.float32)
        return pri, value, np.zeros((B, MAX_SEATS), np.float32)


def drive(driver, evaluators, externals=None, check=None, on_batch=None, ext_threads=16):
    """Run `driver` until next_batch() returns None. -> number of batches."""
    externals = externals or {}
    if externals and hasattr(driver, 'poll_batch'):
        return _drive_async(driver, evaluators, externals, check, on_batch, ext_threads)
    inflight = int(os.environ.get('ZERO_INFLIGHT', '2'))
    if inflight > 1:
        inflight = max([inflight] + [getattr(e, 'workers', 1) for e in evaluators.values()])
    if inflight > 1 and hasattr(driver, 'poll_batch') and evaluators and all(hasattr(e, 'launch') for e in evaluators.values()):
        return _drive_pipelined(driver, evaluators, check, on_batch, inflight)
    nb = 0
    while True:
        lb = driver.next_batch()
        if lb is None:
            return nb
        if lb.external:
            driver.submit_external(lb.id, _external(externals, lb)(lb))
        else:
            driver.submit(lb.id, *evaluators[lb.key](lb))
        nb += 1
        if on_batch is not None:
            on_batch(lb)
        if check is not None:
            check()


def _drive_pipelined(driver, evaluators, check, on_batch, inflight):
    """`drive` with up to `inflight` network batches launched before the oldest is collected
    (no external players). Results are submitted in launch order."""
    from collections import deque
    pending = deque()                              # (batch, evaluator, handle)
    nb = 0

    def finish():
        nonlocal nb
        lb, ev, h = pending.popleft()
        driver.submit(lb.id, *ev.result(h))
        nb += 1
        if on_batch is not None:
            on_batch(lb)
        if check is not None:
            check()

    while True:
        lb = driver.poll_batch() if pending else driver.next_batch()
        if lb is None:
            if not pending:
                return nb
            finish()                               # nothing new ready: the oldest's results
            continue
        if lb.external:
            raise RuntimeError('external batch in a drive without external players')
        ev = evaluators[lb.key]
        pending.append((lb, ev, ev.launch(lb)))
        if len(pending) >= inflight:
            finish()


def _external(externals, lb):
    ans = externals.get(lb.key)
    if ans is None:
        raise KeyError(f'no external player for key {lb.key!r} (have {sorted(externals)})')
    return ans


def _drive_async(driver, evaluators, externals, check, on_batch, ext_threads):
    """`drive` with external batches answered on worker threads: while a group waits for its
    external answers the driver searches and evaluates the other groups (`poll_batch`), and
    answers are submitted as they arrive. Same batches, same answers (each external player's
    seeds depend on the game and its request count, not on timing)."""
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    nb = 0
    pending = {}                                   # batch id -> future of its payloads
    ex = ThreadPoolExecutor(max_workers=ext_threads, thread_name_prefix='zero-ext')
    try:
        while True:
            for bid in [b for b, f in pending.items() if f.done()]:
                driver.submit_external(bid, pending.pop(bid).result())
            lb = driver.poll_batch() if pending else driver.next_batch()
            if lb is None:
                if not pending:
                    return nb
                wait(list(pending.values()), timeout=0.002, return_when=FIRST_COMPLETED)
                continue
            if lb.external:
                pending[lb.id] = ex.submit(_external(externals, lb), lb)
            else:
                driver.submit(lb.id, *evaluators[lb.key](lb))
            nb += 1
            if on_batch is not None:
                on_batch(lb)
            if check is not None:
                check()
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
