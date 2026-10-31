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

Token networks (`net.TokenNet`) on CUDA run through `TokenGraphRunner` when asked (opt-in, default
off: `NetEvaluator(graphs=True)`, or env ZERO_TOKEN_GRAPHS=1 for every token evaluator of the
process; ZERO_CUDA_GRAPHS=0 turns that env off too; zero.h2h / zero.bench `--graphs`): float32, the
host packs the batch (`token_presence`, `token_index`), one CUDA graph per padded (rows, tokens)
shape runs `TokenNet.forward_static`, root belief rows' auxiliary predictions included (docs/az.md
10.6: MTG Zero's Delver network ~3-6x eager's leaves/s on a shared RTX 5090, eager's answers to
float32 rounding). Asked for and not possible is said on stderr (a CPU evaluator, ZERO_CUDA_GRAPHS=0)
or an error (`graphs=True` on a CPU evaluator); `NetEvaluator.graph_calls` counts the batches the
graphs answered.

`drive` keeps up to two network batches in flight when the evaluators can (`launch` /
`result`) and the driver can be polled (`poll_batch`): the next batch is uploaded and replayed
while the previous one's results come back and are softmaxed on the CPU (one CUDA stream keeps
the graphs' static inputs in order; the pinned host buffers alternate between two slots).
ZERO_INFLIGHT=1: one batch at a time, as before.

Auxiliary heads (docs/az.md 15): the search's evaluations compute them only for a batch's root
belief rows (`extra` == ROW_BELIEF, docs/search.md 3.8: the head on those rows alone, in the same
forward pass; the answer's fourth item `aux` [belief rows, aux width], each target's predictions
concatenated in the head's order - a network without the head raises net.HeadMissing; such a batch
skips a conv network's CUDA graphs, a token network's answer it); `NetEvaluator.aux(x, n_players)` is
the inference API for them - one forward pass of the given rows (e.g. a decision's root, MTG Zero's
belief of the opponent's hand) -> each target's predictions (`net.aux_predict`). `set_weights` takes a state with or without the head either way (a rolling
evaluator built before a run turned its auxiliary target on, or one with the head given an older
state).

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

from .protocol import MAX_SEATS, belief_rows

ROW_STEP = 64                    # CUDA graphs are captured for batch sizes padded to this step


def segment_softmax_np(logits, off):
    """Softmax of a flat float array over the segments of `off` (empty segments - deferred
    terminals - hold nothing); vectorized (protocol.segment_softmax is the per-segment reference)."""
    cnt = np.diff(off)
    starts = off[:-1]
    if len(cnt) and cnt.min() == 0:
        keep = cnt > 0
        cnt, starts = cnt[keep], starts[keep]
        if not len(cnt):
            return np.zeros(0, np.float32)
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


T_STEP = 16                      # token networks' graphs: a batch's tokens per row padded to this step
TOKEN_GRAPHS_ENV = 'ZERO_TOKEN_GRAPHS'


def token_presence(x, pidx):
    """A token network's tokens of feature rows x [B, F] (numpy): [B, 1 + slots] bool, the global
    token (always) then each slot of the full token axis (`pidx` = TokenNet.present_index(): its
    present flag's feature, -1 = always present). Its row sums are the rows' token counts."""
    pidx = np.asarray(pidx, np.int64)
    P = np.ones((x.shape[0], 1 + len(pidx)), bool)
    has = pidx >= 0
    P[:, 1:][:, has] = x[:, pidx[has]] > 0.5
    return P


def token_index(P, full):
    """Fill `full` [>= B, T] (int64 numpy, T >= the longest row's tokens) with each row's token
    positions in the full token axis (`token_presence` P [B, 1 + slots]), in layout order, the rest
    0 (padding) - TokenNet.forward_static's `full`."""
    full[:] = 0
    r, c = np.nonzero(P)                             # row-major: each row's tokens in layout order
    full[r, (np.cumsum(P, 1) - 1)[r, c]] = c
    return full


class TokenGraphRunner:
    """CUDA graphs of a token network's evaluation (`net.TokenNet`, docs/az.md 10.6); opt-in
    (NetEvaluator's `graphs`, env ZERO_TOKEN_GRAPHS=1). `evaluate` / `launch` / `result` have
    GraphRunner's contract plus the root belief rows' auxiliary predictions.

    The eager forward packs each batch by its data (`nonzero` per segment, the rows sorted by
    token count, chunks of 64 rows each as long as its longest row: MTG Zero's Delver network, 13
    host syncs and 445 kernels for 64 rows, docs/az.md 10.5-10.6). Here the host reads the slots' present flags from the features (numpy),
    pads the batch to B (a multiple of ROW_STEP) rows x T (the longest row's tokens, a multiple of
    T_STEP), and uploads each row's token positions and count; one graph per (B, T), captured on
    first use, runs `TokenNet.forward_static` - every slot embedded, the tokens gathered, the
    whole batch one chunk -, the value softmax and, for a network with the auxiliary head, the
    head's predictions of every row (`net.aux_predict`; only a batch's root belief rows are
    downloaded). One upload of the features (pinned host buffers, a set per batch in flight: the
    host copies the features outside the segments and the present slots, `_copy_rows`), one
    download; the options' softmax on the CPU (numpy), as GraphRunner. Every graph shares one
    memory pool and one set of device inputs (the graphs run in order on one stream). The graphs
    read the model's own parameters: `set_weights` (load_state_dict, in place) needs no refresh.

    float32 (`amp`: bfloat16 autocast, for measurements only - never a default). The answers
    are eager's up to float32 rounding (other matrix shapes: docs/az.md 10.6 has the measured
    differences); deterministic for a given batch."""

    def __init__(self, model, amp=False):
        from .net import TokenNet
        if not isinstance(model, TokenNet):
            raise TypeError(f'TokenGraphRunner: a token network (net.TokenNet), not {type(model).__name__}')
        self.net = model
        self.dev = next(model.parameters()).device
        if self.dev.type != 'cuda':
            raise ValueError(f'TokenGraphRunner: CUDA graphs need a CUDA network, not {self.dev}')
        self.F = model.features
        self.A = model.config['actions']
        self.S = model.seats
        self.amp = bool(amp)
        self.layout = list(getattr(model, 'aux_layout', None) or [])
        self.W = sum(int(w) for _, w, _ in self.layout)
        self.pidx = np.asarray(model.present_index(), np.int64)
        self.n_tok = 1 + len(self.pidx)
        self.t_cap = -(-self.n_tok // T_STEP) * T_STEP
        # what the host copies of a row (`_copy_rows`): every feature outside the segments (runs of
        # them), and of each segment with a present flag only the present slots (the graphs never read
        # an absent slot's floats: its token is not gathered)
        seg = np.zeros(self.F, bool)
        self.slot_segs, col = [], 1
        for s in model.segs:
            o, k, w = s['offset'], s['count'], s['width']
            if s['present'] is not None:
                seg[o:o + k * w] = True
                self.slot_segs.append((o, k, w, col))
            col += k
        edges = np.flatnonzero(np.diff(np.concatenate([[1], seg.astype(np.int8), [1]])))
        self.runs = [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]
        self.graphs = {}                             # (B, T) -> (graph, x, n, full, ntok, out, aux)
        self.pool = torch.cuda.graph_pool_handle()
        self.inp = None                              # shared device inputs (rows, x, n, full, ntok)
        self.slots = []                              # pinned host buffers, one set per launch in flight
        self.free = []                               # their indexes not in flight
        self.captures = 0
        self.replays = 0

    def refresh(self):
        """Nothing to do: the graphs read the model's parameters, which set_weights rewrites in place."""

    def _inputs(self, Bp):
        if self.inp is None or self.inp[0] < Bp:
            cap = max(Bp, 512)
            with torch.inference_mode(False):
                self.inp = (cap, torch.zeros((cap, self.F), device=self.dev),
                            torch.full((cap,), 2, dtype=torch.int64, device=self.dev),
                            torch.zeros(cap * self.t_cap, dtype=torch.int64, device=self.dev),
                            torch.ones(cap, dtype=torch.int64, device=self.dev))
        return self.inp

    def _slot(self, Bp):
        """A free set of pinned host buffers for `Bp` rows (a new one when every set is in flight:
        a launch's buffers are reused only after its `result`)."""
        k = self.free.pop() if self.free else None
        if k is None:
            k = len(self.slots)
            self.slots.append(None)
        b = self.slots[k]
        if b is None or b[0].shape[0] < Bp:
            n = max(Bp, 256)
            with torch.inference_mode(False):
                b = self.slots[k] = (torch.zeros((n, self.F), dtype=torch.float32).pin_memory(),
                                     torch.empty(n, dtype=torch.int64).pin_memory(),
                                     torch.empty(n * self.t_cap, dtype=torch.int64).pin_memory(),
                                     torch.empty(n, dtype=torch.int64).pin_memory(),
                                     torch.empty((n, self.A + 2 * self.S), dtype=torch.float32).pin_memory(),
                                     torch.empty((n, max(1, self.W)), dtype=torch.float32).pin_memory(),
                                     torch.empty(n, dtype=torch.int64).pin_memory())
        return k, b

    def _run(self, x, n, full, ntok):
        from .net import aux_predict
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.amp):
            r = self.net.forward_static(x, n, full, ntok, aux=bool(self.layout))
        plog, vlog, mar = r[:3]
        out = torch.cat([plog.float(), torch.softmax(vlog.float(), 1), mar.float()], 1)
        aux = None
        if self.layout:
            pred = aux_predict(self.layout, r[3])
            aux = torch.cat([pred[t[0]].float() for t in self.layout], 1)
        return out, aux

    def _graph(self, Bp, Tp):
        g = self.graphs.get((Bp, Tp))
        if g is None:
            _, X, N, FULL, NT = self._inputs(Bp)
            x, n, full, ntok = X[:Bp], N[:Bp], FULL[:Bp * Tp].view(Bp, Tp), NT[:Bp]
            with torch.inference_mode(False), torch.no_grad():
                full.zero_()
                ntok.fill_(1)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(2):               # warm-up (cuBLAS, the attention kernels)
                        self._run(x, n, full, ntok)
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self.pool):
                    out, aux = self._run(x, n, full, ntok)
            g = self.graphs[(Bp, Tp)] = (graph, x, n, full, ntok, out, aux)
            self.captures += 1
        return g

    def _copy_rows(self, x, P, dst):
        """x [B, F] into the pinned rows `dst`: the features outside the segments and the present slots
        (an absent slot keeps an earlier row's floats, or the buffer's zeros: never read)."""
        for a, b in self.runs:
            dst[:, a:b] = x[:, a:b]
        for o, k, w, col in self.slot_segs:
            r, s = np.nonzero(P[:, col:col + k])
            if len(r):
                d = dst[:, o:o + k * w]
                d.shape = (len(d), k, w)                 # (a view, or an error: never a silent copy)
                d[r, s] = x[:, o:o + k * w].reshape(-1, k, w)[r, s]

    def launch(self, x, n_players, opt_off, opt_action, brows=None):
        """Upload a batch and queue its graph and download; -> a handle for `result` (any number of
        launches outstanding: each has its own pinned buffers until its result). `brows`: the root
        belief rows whose auxiliary predictions `result` returns (`protocol.belief_rows`)."""
        B = x.shape[0]
        Bp = -(-B // ROW_STEP) * ROW_STEP
        if x.shape[1] != self.F:
            x = x[:, :self.F] if x.shape[1] > self.F else np.pad(x, ((0, 0), (0, self.F - x.shape[1])))
        P = token_presence(x, self.pidx)
        cnt = P.sum(1)
        Tp = -(-int(cnt.max()) // T_STEP) * T_STEP
        nb = 0 if brows is None else len(brows)
        if nb and not self.layout:
            raise RuntimeError('TokenGraphRunner: root belief rows for a network without the auxiliary head')
        slot, (pin_x, pin_n, pin_full, pin_ntok, pin_out, pin_aux, pin_b) = self._slot(Bp)
        graph, xs, ns, fs, ts, out, aux = self._graph(Bp, Tp)
        self._copy_rows(x, P, pin_x[:B].numpy())
        pn = pin_n[:Bp].numpy()
        pn[:B] = n_players
        pn[B:] = 2
        pt = pin_ntok[:Bp].numpy()
        pt[:B] = cnt
        pt[B:] = 1
        token_index(P, pin_full[:Bp * Tp].numpy().reshape(Bp, Tp))
        if nb:
            pin_b[:nb].numpy()[:] = brows
        with torch.inference_mode(False):
            xs[:B].copy_(pin_x[:B], non_blocking=True)
            ns.copy_(pin_n[:Bp], non_blocking=True)
            fs.view(-1).copy_(pin_full[:Bp * Tp], non_blocking=True)
            ts.copy_(pin_ntok[:Bp], non_blocking=True)
            graph.replay()
            pin_out[:B].copy_(out[:B], non_blocking=True)
            if nb:                                   # (stream-ordered: no host sync)
                bi = pin_b[:nb].to(self.dev, non_blocking=True)
                pin_aux[:nb].copy_(aux.index_select(0, bi), non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
        self.replays += 1
        return (ev, slot, B, n_players, opt_off, opt_action, nb)

    def result(self, h):
        ev, slot, B, n_players, opt_off, opt_action, nb = h
        ev.synchronize()
        o = self.slots[slot][4][:B].numpy()
        A, S = self.A, self.S
        rows = np.repeat(np.arange(B), np.diff(opt_off))
        pri = segment_softmax_np(o[rows, opt_action], opt_off)
        value = np.zeros((B, MAX_SEATS), np.float32)
        margin = np.zeros((B, MAX_SEATS), np.float32)
        value[:, :S] = o[:, A:A + S]
        margin[:, :S] = o[:, A + S:A + 2 * S]
        value[np.arange(MAX_SEATS)[None, :] >= np.asarray(n_players)[:, None]] = 0.0
        aux = self.slots[slot][5][:nb].numpy().copy() if nb else None
        self.free.append(slot)
        if nb:
            return pri, value, margin, aux
        return pri, value, margin

    def evaluate(self, x, n_players, opt_off, opt_action, brows=None):
        return self.result(self.launch(x, n_players, opt_off, opt_action, brows))


def token_graphs_default():
    """Whether a token network's NetEvaluator runs CUDA graphs when its `graphs` is not given: env
    ZERO_TOKEN_GRAPHS=1 (default off), unless ZERO_CUDA_GRAPHS=0 (every network's graphs off) ->
    (on, why off when asked)."""
    asked = os.environ.get(TOKEN_GRAPHS_ENV, '0') not in ('', '0')
    if asked and os.environ.get('ZERO_CUDA_GRAPHS', '1') == '0':
        return False, f'{TOKEN_GRAPHS_ENV}=1 but ZERO_CUDA_GRAPHS=0'
    return asked, None


def _say(msg):
    import sys
    print(f'zero.infer: {msg}', file=sys.stderr, flush=True)


def use_amp(model, device):
    """bfloat16 autocast for a conv network (`net.GridNet`) on CUDA (~1.6-2.6x the leaves/s of
    float32 at batches 512-1024); the MLP keeps float32 (faster as is)."""
    return torch.device(device).type == 'cuda' and getattr(model, 'config', {}).get('arch') == 'grid'


class NetEvaluator:
    """`model` answering LeafBatches (module docstring). The encoding guard's last line (docs/az.md
    10.4): a batch whose rows are not as wide as the network's input (a network of another
    encoding: it would read the first floats of each row, or fail later) raises
    net.EncodingMismatch; `allow_mismatch=True` skips it (tools/selfplay_digest.py)."""

    def __init__(self, model, device='cpu', graphs=None, workers=1, allow_mismatch=False):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.width = None if allow_mismatch else getattr(model, 'features', None)
        self.amp = use_amp(model, self.device)
        if getattr(model, 'config', {}).get('arch') == 'tokens':
            self.fast = self._token_graphs(graphs)
        else:
            if graphs is None:
                graphs = self.amp and os.environ.get('ZERO_CUDA_GRAPHS', '1') != '0'
            elif graphs and getattr(model, 'config', {}).get('arch') != 'grid':
                raise ValueError(f"CUDA graphs asked for an arch {getattr(model, 'config', {}).get('arch', 'mlp')!r} "
                                 f"network: conv ('grid') and token ('tokens') networks have them")
            self.fast = GraphRunner(self.model) if graphs else None
        self.fast_aux = isinstance(self.fast, TokenGraphRunner)   # its graphs answer root belief rows too
        self.calls = 0
        self.leaves = 0
        self.seconds = 0.0
        self.graph_calls = 0                         # batches answered by the CUDA graphs
        # eager batches on worker threads (module docstring); None: computed in launch()
        self.workers = max(1, int(workers)) if self.fast is None else 1
        self.pool = None
        if self.workers > 1:
            import threading
            from concurrent.futures import ThreadPoolExecutor
            self.pool = ThreadPoolExecutor(self.workers, thread_name_prefix='zero-infer')
            self.lock = threading.Lock()

    def _token_graphs(self, graphs):
        """A token network's TokenGraphRunner when its CUDA graphs are asked for (`graphs` True, or
        None and env ZERO_TOKEN_GRAPHS=1: `token_graphs_default`), else None. Asked for and not given
        is said on stderr (an env on a CPU evaluator, ZERO_CUDA_GRAPHS=0), or an error (`graphs`
        True on a CPU evaluator); so is what was given."""
        if graphs is None:
            graphs, why = token_graphs_default()
            if why:
                _say(f'token network CUDA graphs off ({why}): eager inference')
            if graphs and self.device.type != 'cuda':
                _say(f'{TOKEN_GRAPHS_ENV}=1 on a {self.device.type} evaluator: CUDA graphs need CUDA, eager inference')
                return None
        if not graphs:
            return None
        if self.device.type != 'cuda':
            raise ValueError(f'token network CUDA graphs asked for on a {self.device.type} evaluator (CUDA only)')
        _say(f'token network CUDA graphs on (float32; batches padded to {ROW_STEP} rows x {T_STEP} tokens)')
        return TokenGraphRunner(self.model)

    def set_weights(self, state_dict):
        """New weights in place. The auxiliary head's (training only, `net.add_aux`) are optional both
        ways: a network built without the head takes a state with it (they are dropped) and one with
        the head keeps its own when the state has none; everything else must match."""
        own = self.model.state_dict()
        sd = {k: v for k, v in state_dict.items() if k in own or not k.startswith('aux_')}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if unexpected or any(not k.startswith('aux_') for k in missing):
            raise RuntimeError(f'set_weights: state mismatch (missing {missing}, unexpected {unexpected})')
        self.model.eval()
        if self.fast is not None:
            self.fast.refresh()

    def _check_width(self, x):
        if self.width is not None and x.shape[1] != self.width:
            from .net import EncodingMismatch
            c = getattr(self.model, 'config', {})
            raise EncodingMismatch(f'a leaf batch of {x.shape[1]:,} features per row for a network of {self.width:,} '
                                   f'(game {c.get("game")!r}, arch {c.get("arch", "mlp")!r}): the network was built for '
                                   f'another encoding than its driver feeds it')

    @torch.inference_mode()
    def evaluate(self, x, n_players, opt_off, opt_action, extra=None):
        """numpy in -> (priors [options], value [B, MAX_SEATS], margin [B, MAX_SEATS]) numpy; with root
        belief rows in `extra` (docs/search.md 3.8) a fourth item, their auxiliary predictions
        [belief rows, aux width]."""
        t0 = time.perf_counter()
        self._check_width(x)
        brows = belief_rows(extra)
        if len(brows):
            self._check_head()
        if self._graphable(opt_off, brows):
            r = self.fast.evaluate(x, n_players, opt_off, opt_action, *((brows,) if len(brows) else ()))
            self.calls += 1
            self.graph_calls += 1
            self.leaves += x.shape[0]
            self.seconds += time.perf_counter() - t0
            return r
        dev = self.device
        B = x.shape[0]
        xt = torch.from_numpy(x).to(dev, non_blocking=True)
        nt = torch.from_numpy(n_players.astype(np.int64)).to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.amp):
            if len(brows):
                plog, vlog, mar, raw = self.model(xt, nt, aux=True, aux_rows=torch.from_numpy(brows).to(dev))
            else:
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
        if len(brows):
            from .net import aux_predict
            pred = aux_predict(self.aux_layout, raw)
            aux = np.concatenate([pred[t[0]].float().cpu().numpy() for t in self.aux_layout], 1).astype(np.float32)
        if self.pool is not None:
            with self.lock:
                self.calls += 1
                self.leaves += B
                self.seconds += time.perf_counter() - t0
        else:
            self.calls += 1
            self.leaves += B
            self.seconds += time.perf_counter() - t0
        if len(brows):
            return pri, value, margin, aux
        return pri, value, margin

    def _check_head(self):
        """Root belief rows need the auxiliary head (net.HeadMissing otherwise: never a silent uniform
        deal)."""
        if not self.aux_layout:
            from .net import HeadMissing
            c = getattr(self.model, 'config', {})
            raise HeadMissing(f'a leaf batch with root belief rows (a seat\'s belief_tilt > 0) for a network without the '
                              f'auxiliary head (game {c.get("game")!r}, arch {c.get("arch", "mlp")!r}): the root\'s belief '
                              f'needs a checkpoint with a trained head (docs/search.md 3.8)')

    def __call__(self, lb):
        return self.evaluate(lb.x, lb.n_players, lb.opt_off, lb.opt_action, lb.a.get('extra'))

    @property
    def aux_layout(self):
        """The network's auxiliary targets [[name, width, loss], ...] ([]: no auxiliary head)."""
        return list(getattr(self.model, 'aux_layout', None) or [])

    @torch.inference_mode()
    def aux(self, x, n_players):
        """The auxiliary head's predictions for rows `x` [B, F] (numpy; n_players [B]) -> {name: float32
        numpy [B, width]} (`net.aux_predict`: a multinomial target's distribution, a poisson target's
        rates, an mse target's values); {} for a network without the head. One forward pass, outside
        the search's batches (its evaluations never compute the head)."""
        if not self.aux_layout:
            return {}
        from .net import aux_predict
        self._check_width(x)
        xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.device)
        nt = torch.from_numpy(np.asarray(n_players, np.int64)).to(self.device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.amp):
            raw = self.model(xt, nt, aux=True)[3]
        return {k: v.float().cpu().numpy() for k, v in aux_predict(self.aux_layout, raw).items()}

    def _graphable(self, opt_off, brows):
        """The batch goes through the CUDA graphs (a conv network's skip batches with root belief
        rows; a token network's answer them)."""
        return self.fast is not None and len(opt_off) > 1 and (self.fast_aux or not len(brows))

    def launch(self, lb):
        """Start evaluating a LeafBatch; `result(handle)` -> what `ev(lb)` returns. Eager
        evaluators compute at once (or on a worker thread, `workers` > 1)."""
        extra = lb.a.get('extra')
        if self.pool is not None:
            return ('f', self.pool.submit(self.evaluate, lb.x, lb.n_players, lb.opt_off, lb.opt_action, extra), 0)
        t0 = time.perf_counter()
        brows = belief_rows(extra)
        if self._graphable(lb.opt_off, brows):
            self._check_width(lb.x)
            if len(brows):
                self._check_head()
            h = ('g', self.fast.launch(lb.x, lb.n_players, lb.opt_off, lb.opt_action, *((brows,) if len(brows) else ())),
                 lb.B)
        else:
            h = ('d', self.evaluate(lb.x, lb.n_players, lb.opt_off, lb.opt_action, extra), 0)
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
        self.graph_calls += 1
        self.leaves += B
        self.seconds += time.perf_counter() - t0
        return r


class UniformEvaluator:
    """Uniform priors, value 1/n for every seat present, margin 0."""

    def __call__(self, lb):
        off = lb.opt_off
        cnt = np.diff(off)
        pri = np.repeat((1.0 / np.maximum(cnt, 1)).astype(np.float32), cnt)
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
