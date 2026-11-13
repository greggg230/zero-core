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

Readers (docs/az.md 36) are part of the network's forward (they feed the trunk), so every path here
computes them; `NetEvaluator.reader(x, n_players)` is their API, a reader declared `belief` answers
its target in the root belief rows and `aux` (`net.aux_answer`), and `set_network(model)` replaces
`set_weights` where a new checkpoint may have gained a reader (a state cannot carry one).

`NetEvaluator(..., max_rows=n)` (opt-in, default None: every batch whole, as before): a batch of more
than n rows is evaluated in consecutive pieces of at most n rows, their answers concatenated (the same
answers up to the networks' float rounding at another batch shape). The lookahead's batches (docs/search.md
3.9: up to `rows` per game in flight) can hold thousands of rows; the pieces keep a token network's CUDA
graphs to a few shapes (one graph per padded shape) and its memory bounded. Such a batch is answered at
once (`launch` computes it: no pipelining for it).

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
# A capture checks only its own thread's CUDA calls ('thread_local'; torch's default 'global' fails any
# other thread's sync or allocation while it captures - and the capture with it): a run's local linger
# (run setting selfplay_local_linger, docs/az.md 5.3) captures a new batch shape on its thread while the
# run trains on another. One thread: the same as 'global'
CAPTURE_MODE = 'thread_local'


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
            with torch.no_grad(), torch.cuda.graph(graph, pool=self.pool, capture_error_mode=CAPTURE_MODE):
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
# A token batch of at least this many rows has its host copy split over COPY_THREADS threads (numpy
# releases the GIL in the copies; the same bytes copied): the lookahead's batches of thousands of rows
# (docs/search.md 3.9) are bound by the host's copy of the 150 KB rows. MCTS batches (tens to a few
# hundred rows) copy on the caller's thread as before.
COPY_PAR_ROWS = 512
COPY_THREADS = 4
_COPY_POOL = []


def _copy_pool():
    if not _COPY_POOL:
        from concurrent.futures import ThreadPoolExecutor
        _COPY_POOL.append(ThreadPoolExecutor(COPY_THREADS, thread_name_prefix='zero-copy'))
    return _COPY_POOL[0]


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


# Variable-length token sets (docs/az.md 43.7): a token network with tail segments or pointer overrides runs
# its CUDA graphs for the rows of at most LONG_T tokens; a longer row is evaluated eagerly, outside the graphs,
# in groups of rows whose attention matrices (rows x heads x T^2 floats) stay under LONG_BYTES.
LONG_T = 128
LONG_BYTES = 1 << 29

# Length-sorted chunks (docs/az.md 43.11; opt-in, the run setting `infer_token_chunks`): a batch's rows sorted by
# their token counts and cut into at most `chunks` pieces of whole ROW_STEP rows (the last any size), each through
# its own graph of its own longest row - the cut that minimises the graphs' estimated cost (token slots, attention
# weighted by T / CHUNK_ATT, plus CHUNK_FIXED slots per graph). The answers move by float32 rounding (each row's
# outputs depend on the shape of the batch it runs in: docs/az.md 10.6).
CHUNK_ATT = 256
CHUNK_FIXED = 4096


def chunk_rows(cnt, chunks):
    """Rows of token counts `cnt` cut into at most `chunks` length-sorted pieces (`CHUNK_*` above) -> [row index
    arrays, each ascending], or None when one piece is the cheapest (or chunks < 2)."""
    import itertools
    cnt = np.asarray(cnt, np.int64)
    B = len(cnt)
    if chunks < 2 or B <= ROW_STEP:
        return None
    order = np.argsort(cnt, kind='stable')
    c = cnt[order]

    def cost(lo, hi):
        Bp = -(-(hi - lo) // ROW_STEP) * ROW_STEP
        Tp = -(-int(c[hi - 1]) // T_STEP) * T_STEP
        return Bp * Tp * (1.0 + Tp / CHUNK_ATT) + CHUNK_FIXED

    best, edges = cost(0, B), None
    cuts = range(ROW_STEP, B, ROW_STEP)
    for k in range(1, int(chunks)):
        for cs in itertools.combinations(cuts, k):
            e = (0,) + cs + (B,)
            v = sum(cost(a, b) for a, b in zip(e[:-1], e[1:]))
            if v < best:
                best, edges = v, e
    if edges is None:
        return None
    return [np.sort(order[a:b]) for a, b in zip(edges[:-1], edges[1:])]


def token_index_tail(P, tcnt, n_tok, full):
    """`token_index` plus each row's tail tokens (`tcnt` [B] per row, the batch's tail in row order) after its
    head tokens, at their extended positions n_tok + j (j: the token's index in the batch's tail; docs/az.md
    43.5) - forward_static's `full` of a batch with tails (`rows.token_full`'s layout)."""
    token_index(P, full)
    tcnt = np.asarray(tcnt, np.int64)
    NT = int(tcnt.sum())
    if NT:
        rows = np.repeat(np.arange(len(tcnt)), tcnt)
        loc = np.arange(NT) - (np.cumsum(tcnt) - tcnt)[rows]
        full[rows, P.sum(1)[rows] + loc] = n_tok + np.arange(NT)
    return full


def tail_bucket(n):
    """A graph's tail tokens / overrides padded to 0 or 16 * 2^k (docs/az.md 43.7)."""
    n = int(n)
    if n <= 0:
        return 0
    b = 16
    while b < n:
        b *= 2
    return b


def take_options(opt_off, opt_action, rows):
    """The options of rows `rows` (an index array) of a batch -> (their opt_off, their opt_action)."""
    opt_off = np.asarray(opt_off, np.int64)
    rows = np.asarray(rows, np.int64)
    cnt = opt_off[rows + 1] - opt_off[rows]
    off = np.zeros(len(rows) + 1, np.int64)
    np.cumsum(cnt, out=off[1:])
    idx = np.repeat(opt_off[rows] - off[:-1], cnt) + np.arange(off[-1])
    return off, np.asarray(opt_action)[idx]


def ov_torch(res, device):
    """`rows.resolve`'s overrides (numpy) -> TokenNet's `ov` (int64 tensors on `device`)."""
    return {k: torch.from_numpy(np.ascontiguousarray(res[k], dtype=np.int64)).to(device) for k in ('row', 'act', 'ref', 'col')}


class _Split:
    """A batch whose rows of more than LONG_T tokens were evaluated eagerly (`TokenGraphRunner`): the graphs'
    handle of the other rows (or None) and the long rows' outputs, merged by `result`."""
    __slots__ = ('h', 'o', 'aux', 'short', 'long', 'B', 'n_players', 'opt_off', 'opt_action', 'brows')

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# A launch's upload and its graph's integer inputs (docs/az.md 43.12): regions of int64 words, each starting on a
# multiple of IN_ALIGN words (256 bytes: an allocation's own alignment, so no kernel sees another alignment than in
# a buffer of its own)
IN_ALIGN = 32


def _al(n):
    return -(-int(n) // IN_ALIGN) * IN_ALIGN


class _Down:
    """A launch's download (`TokenGraphRunner._stage_down`): its outputs copied out of the graph's on the device stream
    (`done` recorded after them), the copies to the pinned host buffers queued later on the download stream
    (`_flush_downs`: after the next launch's uploads, or at its result), `ev` recorded after them (None until)."""
    __slots__ = ('done', 'items', 'ev')

    def __init__(self, done, items):
        self.done, self.items, self.ev = done, items, None


class _Parts:
    """A batch run as length-sorted chunks (`TokenGraphRunner`, `chunk_rows`): each chunk's rows and handle, merged by
    `result` into the batch's row order."""
    __slots__ = ('parts', 'hs', 'B', 'n_players', 'opt_off', 'opt_action', 'brows')

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


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

    Per launch (docs/az.md 43.12): one upload of one pinned int64 buffer (a sparse batch's non-zeros, the
    belief rows, the graph's integer inputs and tail floats at its key's `_layout`) and one device copy into
    the graph's region of the shared inputs; a sparse batch packed by the engine module's
    `pack_sparse_tokens` / `pack_tail_tokens` (Rust, off the GIL; the dense path numpy); a launch's
    download queued by the next launch just before its replay (or at its result), never ahead of an
    upload on the copy engine.

    float32 (`amp`: bfloat16 autocast, for measurements only - never a default). The answers
    are eager's up to float32 rounding (other matrix shapes: docs/az.md 10.6 has the measured
    differences); deterministic for a given batch."""

    def __init__(self, model, amp=False, chunks=1):
        from .net import TokenNet
        if not isinstance(model, TokenNet):
            raise TypeError(f'TokenGraphRunner: a token network (net.TokenNet), not {type(model).__name__}')
        self.net = model
        self.chunks = max(1, int(chunks or 1))       # length-sorted chunks per batch (1: the batch whole)
        self.chunked = 0                             # batches run as chunks
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
            if s['present'] is not None and k:       # (a tail-only segment has no head slots)
                seg[o:o + k * w] = True
                self.slot_segs.append((o, k, w, col))
            col += k
        edges = np.flatnonzero(np.diff(np.concatenate([[1], seg.astype(np.int8), [1]])))
        self.runs = [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]
        self.graphs = {}                             # key -> (graph, x, its input region, its layout, out, aux)
        self.pool = torch.cuda.graph_pool_handle()
        self.inp = None                              # shared device input rows (capacity, X [cap, F])
        # the graphs' integer inputs (and the tails' floats): one shared int64 buffer, each graph's views at its key's
        # layout (`_layout`) - a launch fills them with one copy (docs/az.md 43.12)
        self.ibuf = None
        self.slots = []                              # pinned host buffers, one set per launch in flight
        self.free = []                               # their indexes not in flight
        self.sslots = []                             # the same for sparse batches (`launch_sparse`)
        self.sfree = []
        # a sparse row's present flags (`launch_sparse`): per feature its slot's column of the presence
        # matrix (1 + slot) when it is a present flag, else -1; slots without a flag are always present
        self.flag_col = np.full(self.F, -1, np.int64)
        has = self.pidx >= 0
        self.flag_col[self.pidx[has]] = 1 + np.flatnonzero(has)
        self.always = 1 + np.flatnonzero(~has)
        # the token columns every row has (the global token, the slots without a flag): the packer's (docs/az.md 43.12)
        self.base = np.zeros(self.n_tok, bool)
        self.base[0] = True
        self.base[self.always] = True
        self.engine = None                           # the engine module whose packers a sparse batch goes through
        self.widths_lut = {}                         # a batch's tail widths -> per segment its width (`_pack_tail`)
        self.captures = 0
        self.replays = 0
        # the transfers beside the graphs (docs/az.md 43.11): each buffer set's uploads on a copy stream into device
        # staging buffers of its own (while the graph before runs), its outputs copied out of the graph's on the
        # device and downloaded on a third stream; the shared input rows zeroed again only where the last sparse
        # batch wrote them
        self.dstage = {}                             # buffer set key -> {name: device staging tensor}
        self.xstate = {}                             # input rows' storage -> {'X', 'idx', 'n', 'all'}
        self.streams = None                          # (copy stream, download stream)
        # launches whose downloads are not queued yet (`_Down`): queued by the next launch after its uploads and its
        # inputs' copies, just before its replay, or at their result (a download waits for its graph; queued before
        # an upload or a device copy, it holds them back on this machine's copy engines until that graph ends - and
        # the next graph waits for its inputs: docs/az.md 43.12)
        self.downs = []
        # variable-length token sets (docs/az.md 43.7): tails and overrides uploaded beside the rows, graphs
        # keyed by (rows, tokens, tail tokens, overrides), rows of more than LONG_T tokens eager
        self.tailnet = bool(model.tails) or model.option_pointers
        self.long_rows = 0                           # rows evaluated eagerly (more than LONG_T tokens)
        if self.tailnet:
            self.t_cap = max(self.t_cap, -(-LONG_T // T_STEP) * T_STEP)
            self.lay = model.config['layout']
            self.Wt = model.t_width
            self.seg_tail = np.zeros(len(model.segs), bool)
            self.seg_tail[model.tails] = True
            self.seg_count = np.array([s['count'] for s in model.segs], np.int64)
            # a padded tail entry: row 0, the first tail segment, slot its head count, floats 0
            self.pad_seg = model.tails[0] if model.tails else 0
            self.pad_slot = int(self.seg_count[self.pad_seg])
            # the layout's tables `_check_ov_host` reads, on the host once (a read of the device's per batch would wait
            # for the graphs in flight)
            self.seg_nk = model.seg_nk.cpu().numpy()
            self.pos_seg = model.pos_seg.cpu().numpy()

    def refresh(self):
        """Nothing to do: the graphs read the model's parameters, which set_weights rewrites in place."""

    def _inputs(self, Bp):
        if self.inp is None or self.inp[0] < Bp:
            cap = max(Bp, 512)
            with torch.inference_mode(False):
                self.inp = (cap, torch.zeros((cap, self.F), device=self.dev))
            X = self.inp[1]
            self.xstate[X.data_ptr()] = {'X': X, 'idx': None, 'n': 0, 'all': False}     # (zeros)
        return self.inp

    def _ibuf(self, L):
        """The graphs' shared integer inputs, at least L words (a graph keeps the views it was captured with: a bigger
        buffer is a new one, the graphs before keep theirs)."""
        if self.ibuf is None or self.ibuf.shape[0] < L:
            with torch.inference_mode(False):
                self.ibuf = torch.zeros(max(L, 1 << 16), dtype=torch.int64, device=self.dev)
        return self.ibuf

    def _layout(self, Bp, Tp, NTp=None, NOp=None):
        """A graph's integer inputs as regions of the shared buffer (`_ibuf`) -> ({name: offset}, words): n [Bp], ntok
        [Bp], full [Bp * Tp]; with tails / overrides (NTp not None) tint [3 * NTp] (rows, segments, slots), oint [4 *
        NOp] (rows, actions, refs, columns), tval [NTp, Wt] float32. A launch uploads the same layout after its own
        non-zeros and belief rows (`_up_head`) and copies it into the graph's region in one copy."""
        o = {'n': 0, 'ntok': _al(Bp)}
        o['full'] = o['ntok'] + _al(Bp)
        e = o['full'] + _al(Bp * Tp)
        if NTp is not None:
            o['tint'] = e
            e += _al(3 * NTp)
            o['oint'] = e
            e += _al(4 * NOp)
            o['tval'] = e
            e += _al((NTp * self.Wt + 1) // 2)
        return o, e

    @staticmethod
    def _up_head(nnz, nb):
        """A launch's upload (int64 words) before its graph's inputs: the non-zeros' flat indexes [nnz], their values
        [nnz float32], the root belief rows [nb] -> (their offsets, the graph inputs' offset)."""
        vo = _al(nnz)
        bo = vo + _al((nnz + 1) // 2)
        return 0, vo, bo, bo + _al(nb)

    def _streams(self):
        if self.streams is None:
            self.streams = (torch.cuda.Stream(self.dev), torch.cuda.Stream(self.dev))
        return self.streams

    def _mirror(self, key, name, pin):
        """Buffer set `key`'s device staging tensor `name`: the shape and dtype of its pinned buffer `pin` (made
        again when that grows; a set is reused only after its launch's result, so never while in use)."""
        d = self.dstage.setdefault(key, {})
        t = d.get(name)
        if t is None or t.shape != pin.shape or t.dtype != pin.dtype:
            with torch.inference_mode(False):
                t = d[name] = torch.empty(pin.shape, dtype=pin.dtype, device=self.dev)
        return t

    def _upload(self, key, pin_i, U):
        """A launch's one upload on the copy stream: its pinned words pin_i[:U] into its buffer set's device staging ->
        (the staging, the event after the copy). The copy overlaps the graph running before (the staging is the set's
        own); the device stream waits for the event before it reads the staging."""
        d = self.dstage.setdefault(key, {})
        st = d.get('i')
        cs = self._streams()[0]
        if st is None or st.shape != pin_i.shape:
            with torch.inference_mode(False):
                st = d['i'] = torch.empty(pin_i.shape, dtype=torch.int64, device=self.dev)
            # (a staging tensor just made: its memory may have been the device stream's - wait for that stream once)
            cs.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(cs):
            st[:U].copy_(pin_i[:U], non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(cs)
        return st, ev

    def _rezero(self, xs):
        """The graphs' input rows (the storage under `xs`) back to zeros before a sparse batch's scatter: where the
        last sparse batch wrote (its non-zeros' places, kept on the device), or wholly after a dense batch."""
        X = xs if xs._base is None else xs._base
        st = self.xstate[X.data_ptr()]
        if st['all']:
            X.zero_()
        elif st['n']:
            X.view(-1).index_fill_(0, st['idx'][:st['n']], 0.0)
        st['all'], st['n'] = False, 0
        return st

    def _scatter(self, xs, fd, vd):
        """A sparse batch's rows: the input rows zeros again (`_rezero`), the non-zeros scattered in, their places
        kept for the next batch's `_rezero` - the rows `xs.zero_(); index_copy_` built, bit for bit."""
        st = self._rezero(xs)
        nnz = fd.shape[0]
        if nnz:
            xs.view(-1).index_copy_(0, fd, vd)
            if st['idx'] is None or st['idx'].shape[0] < nnz:
                with torch.inference_mode(False):
                    st['idx'] = torch.empty(max(nnz, 1 << 16, 0 if st['idx'] is None else 2 * st['idx'].shape[0]),
                                            dtype=torch.int64, device=self.dev)
            st['idx'][:nnz].copy_(fd)
            st['n'] = nnz

    def _dense_rows(self, xs, pin_x, B):
        """A dense batch's rows into the graphs' input rows (on the device stream: the graph before reads them);
        the storage then counts as written everywhere (`_rezero`)."""
        xs[:B].copy_(pin_x[:B], non_blocking=True)
        X = xs if xs._base is None else xs._base
        self.xstate[X.data_ptr()]['all'] = True

    def _stage_down(self, key, out, aux, B, nb, bi, pin_out, pin_aux):
        """After a replay: its outputs copied out of the graph's (on the device stream, so the next replay of the
        same graph may overwrite them) -> the launch's `_Down`, its download queued by `_flush_downs` (by the next
        launch just before its replay, or at its result: `_wait_down`)."""
        dout = self._mirror(key, 'out', pin_out)
        dout[:B].copy_(out[:B])
        items = [(pin_out[:B], dout[:B])]
        if nb:
            daux = self._mirror(key, 'aux', pin_aux)
            torch.index_select(aux, 0, bi, out=daux[:nb])
            items.append((pin_aux[:nb], daux[:nb]))
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(self.dev))
        d = _Down(done, items)
        self.downs.append(d)
        return d

    def _flush_downs(self, here=False):
        """The downloads of the launches not queued yet (`downs`), in launch order, on the download stream: each waits
        for its graph (`done`) - with `here`, for the device stream as it is now, after the next launch's input copies
        (a download that ran beside them would hold them back on the copy engine) -, its event recorded after its
        copies."""
        if not self.downs:
            return
        ds = self._streams()[1]
        if here:
            now = torch.cuda.Event()
            now.record(torch.cuda.current_stream(self.dev))
        with torch.inference_mode(False), torch.cuda.stream(ds):
            for d in self.downs:
                ds.wait_event(now if here else d.done)
                for pin, dev in d.items:
                    pin.copy_(dev, non_blocking=True)
                d.ev = torch.cuda.Event()
                d.ev.record(ds)
        self.downs.clear()

    def _wait_down(self, d):
        """Wait for a launch's download (queued now if it is not yet)."""
        if d.ev is None:
            self._flush_downs()
        d.ev.synchronize()

    def _slot(self, Bp, U):
        """A free set of pinned host buffers for a dense batch of `Bp` rows and an upload of U words (a new one when
        every set is in flight: a launch's buffers are reused only after its `result`) -> (its index, (rows, upload
        words, outputs, auxiliary outputs))."""
        k = self.free.pop() if self.free else None
        if k is None:
            k = len(self.slots)
            self.slots.append(None)
        b = self.slots[k]
        if b is None or b[0].shape[0] < Bp:
            n = max(Bp, 256)
            with torch.inference_mode(False):
                b = self.slots[k] = (torch.zeros((n, self.F), dtype=torch.float32).pin_memory(),
                                     torch.empty(max(U, 1 << 16), dtype=torch.int64).pin_memory(),
                                     torch.empty((n, self.A + 2 * self.S), dtype=torch.float32).pin_memory(),
                                     torch.empty((n, max(1, self.W)), dtype=torch.float32).pin_memory())
        if b[1].shape[0] < U:
            with torch.inference_mode(False):
                b = self.slots[k] = (b[0], torch.empty(max(U, 2 * b[1].shape[0]), dtype=torch.int64).pin_memory(),
                                     b[2], b[3])
        return k, b

    def _run(self, x, n, full, ntok, tail=None, ov=None):
        from .net import aux_answer
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.amp):
            if tail is None and ov is None:
                r = self.net.forward_static(x, n, full, ntok, aux=bool(self.layout))
            else:
                r = self.net.forward_static(x, n, full, ntok, aux=bool(self.layout), tail=tail, ov=ov)
        plog, vlog, mar = r[:3]
        out = torch.cat([plog.float(), torch.softmax(vlog.float(), 1), mar.float()], 1)
        aux = None
        if self.layout:
            pred = aux_answer(self.net, r[3])
            aux = torch.cat([pred[t[0]].float() for t in self.layout], 1)
        return out, aux



    def _pad_inputs(self, tail, ov):
        """Device tails / overrides set to padding (docs/az.md 43.5: tail row 0, the first tail segment, slot
        its count, floats 0; override action `actions`, ref -1)."""
        if tail is not None:
            tail['row'].zero_()
            tail['seg'].fill_(self.pad_seg)
            tail['slot'].fill_(self.pad_slot)
            tail['val'].zero_()
        if ov is not None:
            ov['row'].zero_()
            ov['act'].fill_(self.A)
            ov['ref'].fill_(-1)
            ov['col'].zero_()

    def _graph(self, Bp, Tp, NTp=None, NOp=None):
        key = (Bp, Tp) if NTp is None else (Bp, Tp, NTp, NOp)
        g = self.graphs.get(key)
        if g is None:
            _, X = self._inputs(Bp)
            o, L = self._layout(Bp, Tp, NTp, NOp)
            IN = self._ibuf(L)
            with torch.inference_mode(False), torch.no_grad():
                # (the views of the shared inputs made outside inference mode: normal tensors the graph's copy writes)
                x, n, ntok, reg = X[:Bp], IN[o['n']:o['n'] + Bp], IN[o['ntok']:o['ntok'] + Bp], IN[:L]
                full = IN[o['full']:o['full'] + Bp * Tp].view(Bp, Tp)
                tail = ov = None
                if NTp is not None:                  # (a network with tails or overrides, docs/az.md 43.7)
                    tint, oint = IN[o['tint']:o['tint'] + 3 * NTp], IN[o['oint']:o['oint'] + 4 * NOp]
                    if self.net.tails:
                        tv = IN[o['tval']:o['tval'] + (NTp * self.Wt + 1) // 2].view(torch.float32)
                        tail = {'row': tint[:NTp], 'seg': tint[NTp:2 * NTp], 'slot': tint[2 * NTp:],
                                'val': tv[:NTp * self.Wt].view(NTp, self.Wt)}
                    if self.net.option_pointers:
                        ov = {k: oint[i * NOp:(i + 1) * NOp] for i, k in enumerate(('row', 'act', 'ref', 'col'))}
                n.fill_(2)
                full.zero_()
                ntok.fill_(1)
                self._pad_inputs(tail, ov)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(2):               # warm-up (cuBLAS, the attention kernels)
                        self._run(x, n, full, ntok, tail, ov)
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self.pool, capture_error_mode=CAPTURE_MODE):
                    out, aux = self._run(x, n, full, ntok, tail, ov)
            g = self.graphs[key] = (graph, x, reg, o, out, aux)
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

    def launch(self, x, n_players, opt_off, opt_action, brows=None, tail=None, ov=None):
        """Upload a batch and queue its graph and download; -> a handle for `result` (any number of
        launches outstanding: each has its own pinned buffers until its result). `brows`: the root
        belief rows whose auxiliary predictions `result` returns (`protocol.belief_rows`). `tail` /
        `ov` (`rows.Tail` / `rows.Overrides`: a network with tail segments / option_pointers needs them,
        any other refuses them; docs/az.md 43.7): `_launch_tails`. With `chunks` > 1: length-sorted chunks
        (`chunk_rows`)."""
        if self.chunks > 1:
            if x.shape[1] != self.F:
                x = x[:, :self.F] if x.shape[1] > self.F else np.pad(x, ((0, 0), (0, self.F - x.shape[1])))
            cnt = token_presence(x, self.pidx).sum(1) + (tail.counts() if tail is not None else 0)
            parts = chunk_rows(cnt, self.chunks)
            if parts is not None:
                return self._launch_parts(x, None, n_players, opt_off, opt_action, brows, tail, ov, parts)
        return self._launch1(x, n_players, opt_off, opt_action, brows, tail, ov)

    def _launch1(self, x, n_players, opt_off, opt_action, brows=None, tail=None, ov=None):
        if self.tailnet or tail is not None or ov is not None:
            return self._launch_tails(x, None, n_players, opt_off, opt_action, brows, tail, ov)
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
        o, L = self._layout(Bp, Tp)
        _, _, bo, R0 = self._up_head(0, nb)
        slot, (pin_x, pin_i, pin_out, pin_aux) = self._slot(Bp, R0 + L)
        graph, xs, reg, _, out, aux = self._graph(Bp, Tp)
        if B >= COPY_PAR_ROWS:
            step = -(-B // COPY_THREADS)
            dst = pin_x[:B].numpy()
            futs = [_copy_pool().submit(self._copy_rows, x[a:a + step], P[a:a + step], dst[a:a + step])
                    for a in range(0, B, step)]
            for f in futs:
                f.result()
        else:
            self._copy_rows(x, P, pin_x[:B].numpy())
        I = pin_i.numpy()
        pn = I[R0 + o['n']:R0 + o['n'] + Bp]
        pn[:B] = n_players
        pn[B:] = 2
        pt = I[R0 + o['ntok']:R0 + o['ntok'] + Bp]
        pt[:B] = cnt
        pt[B:] = 1
        token_index(P, I[R0 + o['full']:R0 + o['full'] + Bp * Tp].reshape(Bp, Tp))
        if nb:
            I[bo:bo + nb] = brows
        key = ('d', slot)
        with torch.inference_mode(False):
            st, up = self._upload(key, pin_i, R0 + L)
            torch.cuda.current_stream(self.dev).wait_event(up)
            self._dense_rows(xs, pin_x, B)
            reg.copy_(st[R0:R0 + L])
            self._flush_downs(True)
            graph.replay()
            ev = self._stage_down(key, out, aux, B, nb, st[bo:bo + nb], pin_out, pin_aux)
        self.replays += 1
        return (ev, slot, B, n_players, opt_off, opt_action, nb)

    def _sslot(self, Bp, U):
        """A free set of pinned host buffers of a sparse batch (`launch_sparse`): the dense set's without the rows ->
        (its index, (None, upload words, outputs, auxiliary outputs))."""
        k = self.sfree.pop() if self.sfree else None
        if k is None:
            k = len(self.sslots)
            self.sslots.append(None)
        b = self.sslots[k]
        if b is None or b[2].shape[0] < Bp or b[1].shape[0] < U:
            n = max(Bp, 256, 0 if b is None else b[2].shape[0])
            u = max(U, 1 << 18, 0 if b is None else 2 * b[1].shape[0])
            with torch.inference_mode(False):
                b = self.sslots[k] = (None, torch.empty(u, dtype=torch.int64).pin_memory(),
                                      torch.empty((n, self.A + 2 * self.S), dtype=torch.float32).pin_memory(),
                                      torch.empty((n, max(1, self.W)), dtype=torch.float32).pin_memory())
        return k, b

    def _packer(self):
        """The active game package's engine module (`games.engine()`), whose `pack_sparse_tokens` /
        `pack_tail_tokens` pack a sparse batch off the GIL (docs/az.md 43.12)."""
        if self.engine is None:
            from . import games as G
            e = G.engine()
            if not hasattr(e, 'pack_sparse_tokens'):
                raise RuntimeError(f'{e.__name__} has no pack_sparse_tokens: a build of a zero-core before docs/az.md 43.12 '
                                   f'under this one - rebuild it (tools/build_py.sh)')
            self.engine = e
        return self.engine

    def _pack_sparse(self, sp, t_off, long_t, Bp, I, R0, o, pres=None):
        """A sparse batch `sp` (xs_off int64, xs_idx, xs_val) into a set's upload words `I` (numpy int64; `_up_head`'s
        regions, the graph's inputs `o` from R0): the non-zeros' flat indexes and values, each row's tokens (head +
        tail, `t_off`'s; the padding rows 1) and the token positions (`token_index` / `token_index_tail`'s, row stride
        Tp), the head presence into `pres` [B * n_tok] when given -> (Tp, the longest row's tokens); Tp 0 (nothing but
        the counts) when a row is longer than `long_t` (>= 0)."""
        xs_off, xs_idx, xs_val = sp
        nnz = int(xs_off[-1])
        if xs_idx.dtype != np.uint32:
            xs_idx = np.asarray(xs_idx)
            if len(xs_idx) and (int(xs_idx.min()) < 0 or int(xs_idx.max()) >= self.F):
                raise ValueError(f'TokenGraphRunner: a sparse feature index out of [0, {self.F})')
        xs_idx = np.ascontiguousarray(xs_idx, np.uint32)
        xs_val = np.ascontiguousarray(xs_val, np.float32)
        _, vo, _, _ = self._up_head(nnz, 0)
        return self._packer().pack_sparse_tokens(
            xs_off, xs_idx, xs_val, self.F, self.flag_col, self.base, t_off, int(long_t), T_STEP, I[:nnz],
            I[vo:vo + (nnz + 1) // 2].view(np.float32)[:nnz], I[R0 + o['ntok']:R0 + o['ntok'] + Bp],
            I[R0 + o['full']:], pres)

    def _pack_tail(self, tail, NTp, I, at):
        """A batch's tail into a set's upload words `I` at the graph inputs' region `at` (`_layout`'s tint, tval): rows,
        segments, slots and the floats padded to NTp entries, checked (`pack::tail`: `_check_tail_host`'s checks;
        docs/az.md 43.12)."""
        k = tuple(sorted(tail.widths.items()))
        w = self.widths_lut.get(k)
        if w is None:
            w = np.zeros(max(len(self.seg_count), max(tail.widths, default=-1) + 1), np.int64)
            for g, v in tail.widths.items():
                w[int(g)] = int(v)
            self.widths_lut[k] = w
        ti, tv = at
        self._packer().pack_tail_tokens(
            np.ascontiguousarray(tail.off, np.int64), np.ascontiguousarray(tail.seg, np.uint16),
            np.ascontiguousarray(tail.slot, np.uint32), np.ascontiguousarray(tail.val, np.float32), w, self.seg_tail,
            self.seg_count, self.Wt, NTp, int(self.pad_seg), int(self.pad_slot), I[ti:ti + 3 * NTp],
            I[tv:tv + (NTp * self.Wt + 1) // 2].view(np.float32)[:NTp * self.Wt])

    def presence_sparse(self, xs_off, xs_idx, xs_val):
        """`token_presence` of sparse rows: [B, 1 + slots] bool, the global token and the slots without a
        flag always, each flagged slot when its flag's value > 0.5."""
        B = len(xs_off) - 1
        P = np.zeros((B, self.n_tok), bool)
        P[:, 0] = True
        P[:, self.always] = True
        rows = np.repeat(np.arange(B), np.diff(xs_off))
        col = self.flag_col[np.asarray(xs_idx, np.int64)]
        hit = (col >= 0) & (np.asarray(xs_val) > 0.5)
        P[rows[hit], col[hit]] = True
        return P

    def launch_sparse(self, xs_off, xs_idx, xs_val, n_players, opt_off, opt_action, brows=None, tail=None, ov=None):
        """`launch` of a sparse batch (protocol.py: each row's non-zero features): the presence from the
        non-zeros, the rows built on the device (the graphs' input rows zeroed, the non-zeros scattered
        in: the same floats wherever the graphs read) - no dense row on the host. The answers are
        `launch`'s on the densified rows, bit for bit. `tail` / `ov`: as `launch`'s."""
        if self.chunks > 1:
            cnt = self.presence_sparse(xs_off, xs_idx, xs_val).sum(1) + (tail.counts() if tail is not None else 0)
            parts = chunk_rows(cnt, self.chunks)
            if parts is not None:
                sp = (np.asarray(xs_off, np.int64), np.asarray(xs_idx), np.asarray(xs_val))
                return self._launch_parts(None, sp, n_players, opt_off, opt_action, brows, tail, ov, parts)
        return self._launch_sparse1(xs_off, xs_idx, xs_val, n_players, opt_off, opt_action, brows, tail, ov)

    def _launch_sparse1(self, xs_off, xs_idx, xs_val, n_players, opt_off, opt_action, brows=None, tail=None, ov=None):
        if self.tailnet or tail is not None or ov is not None:
            return self._launch_tails(None, (xs_off, xs_idx, xs_val), n_players, opt_off, opt_action, brows, tail, ov)
        xs_off = np.ascontiguousarray(xs_off, np.int64)
        B = len(xs_off) - 1
        Bp = -(-B // ROW_STEP) * ROW_STEP
        nnz = int(xs_off[-1])
        nb = 0 if brows is None else len(brows)
        if nb and not self.layout:
            raise RuntimeError('TokenGraphRunner: root belief rows for a network without the auxiliary head')
        _, vo, bo, R0 = self._up_head(nnz, nb)
        o, Lmax = self._layout(Bp, self.t_cap)
        slot, (_, pin_i, pin_out, pin_aux) = self._sslot(Bp, R0 + Lmax)
        I = pin_i.numpy()
        try:
            # (the rows' non-zeros, tokens and token positions: `presence_sparse` and `token_index`'s, packed off the GIL)
            Tp, _ = self._pack_sparse((xs_off, xs_idx, xs_val), None, -1, Bp, I, R0, o)
        except Exception:
            self.sfree.append(slot)
            raise
        o, L = self._layout(Bp, Tp)
        graph, xs, reg, _, out, aux = self._graph(Bp, Tp)
        pn = I[R0 + o['n']:R0 + o['n'] + Bp]
        pn[:B] = n_players
        pn[B:] = 2
        if nb:
            I[bo:bo + nb] = brows
        key = ('s', slot)
        with torch.inference_mode(False):
            st, up = self._upload(key, pin_i, R0 + L)
            torch.cuda.current_stream(self.dev).wait_event(up)
            self._scatter(xs, st[:nnz], st[vo:vo + (nnz + 1) // 2].view(torch.float32)[:nnz])
            reg.copy_(st[R0:R0 + L])
            self._flush_downs(True)
            graph.replay()
            ev = self._stage_down(key, out, aux, B, nb, st[bo:bo + nb], pin_out, pin_aux)
        self.replays += 1
        return (ev, ('s', slot), B, n_players, opt_off, opt_action, nb)

    def _tail_args(self, tail, ov):
        net = self.net
        if net.tails and tail is None:
            raise ValueError("TokenGraphRunner: a network with tail segments needs the batch's tail (rows.Tail; R5)")
        if not net.tails and tail is not None:
            raise ValueError('TokenGraphRunner: a batch with a tail for a network without tail segments (R5)')
        if net.option_pointers and ov is None:
            raise ValueError("TokenGraphRunner: a network with option_pointers needs the batch's overrides (R5)")
        if not net.option_pointers and ov is not None:
            raise ValueError('TokenGraphRunner: a batch with pointer overrides for a network without option_pointers (R5)')

    def _check_tail_host(self, tr, ts, tsl, tv, B):
        """A batch's padded tail (numpy) is this network's (R5): tail segments, slots at or past the head and
        below 2^32, rows in [0, B) ascending, floats [NT, Wt]."""
        NT = len(ts)
        if tv.ndim != 2 or tv.shape != (NT, self.Wt) or len(tr) != NT or len(tsl) != NT:
            raise ValueError(f"TokenGraphRunner: a tail of {NT} tokens with floats {tv.shape} (the network's tail "
                             f"width {self.Wt})")
        if not NT:
            return
        sc = np.clip(ts, 0, len(self.seg_tail) - 1)
        bad = ((ts < 0) | (ts >= len(self.seg_tail)) | ~self.seg_tail[sc] | (tsl < self.seg_count[sc]) | (tsl >= 1 << 32)
               | (tr < 0) | (tr >= B))
        bad[1:] |= tr[1:] < tr[:-1]
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            raise ValueError(f'TokenGraphRunner: tail token {i} (row {tr[i]}, segment {ts[i]}, slot {tsl[i]}) is not a '
                             f'tail token of this network ({self.net.tail_segments()}; rows ascending)')

    def _check_ov_host(self, res, P, tr, ts, B, NT):
        """A batch's resolved overrides (numpy) for the graphs, which cannot check them (the network checks them
        in eager): each target a present head slot (`P`, the head presence) or a tail token of the same row,
        the column one its segment has, ref -1 or a token, the action in [0, actions]."""
        r, a, ref, col = (np.asarray(res[k], np.int64) for k in ('row', 'act', 'ref', 'col'))
        n_tok = self.n_tok
        nk, pos_seg = self.seg_nk, self.pos_seg
        bad = (ref < -1) | (ref == 0) | (ref >= n_tok + NT) | (r < 0) | (r >= B) | (a < 0) | (a > self.A) | (col < 0)
        rc = np.clip(r, 0, B - 1)
        hd = (ref > 0) & (ref < n_tok)
        hr = np.clip(ref, 0, n_tok - 1)
        seg = pos_seg[hr]
        bad |= hd & ~P[rc, hr]
        if NT:
            tl = ref >= n_tok
            tj = np.clip(ref - n_tok, 0, NT - 1)
            bad |= tl & (tr[tj] != r)
            seg = np.where(tl, ts[tj], seg)
        bad |= (ref > 0) & (col >= nk[np.clip(seg, 0, len(nk) - 1)])
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            raise ValueError(f'TokenGraphRunner: pointer override {i} (row {r[i]}, action {a[i]}, ref {ref[i]}, column '
                             f'{col[i]}): its target is not a present token of its row with that column (docs/az.md 43.7)')



    def _launch_tails(self, x, sp, n_players, opt_off, opt_action, brows, tail, ov):
        """`launch` / `launch_sparse` of a network with tail segments or pointer overrides (docs/az.md 43.7):
        the rows of at most LONG_T tokens (head tokens + tail tokens) through the graph of (rows, tokens,
        tail tokens, overrides), each padded; the longer rows evaluated eagerly now (`_eager_rows`), their
        answers merged by `result`."""
        self._tail_args(tail, ov)
        if x is not None:
            B = x.shape[0]
            if x.shape[1] != self.F:
                x = x[:, :self.F] if x.shape[1] > self.F else np.pad(x, ((0, 0), (0, self.F - x.shape[1])))
            P = token_presence(x, self.pidx)
        else:
            xs_off = np.ascontiguousarray(sp[0], np.int64)
            sp = (xs_off, np.asarray(sp[1]), np.asarray(sp[2]))
            B = len(xs_off) - 1
            # (packed off the GIL straight into a buffer set: docs/az.md 43.12; a batch with a row of more than LONG_T
            # tokens goes on below, its rows split)
            h = self._graph_tails_sparse(B, sp, n_players, opt_off, opt_action, brows, tail, ov)
            if h is not None:
                return h
            P = self.presence_sparse(*sp)
        tcnt = tail.counts() if tail is not None else np.zeros(B, np.int64)
        cnt = P.sum(1) + tcnt
        long = cnt > LONG_T
        if not long.any():
            return self._graph_tails(B, x, sp, P, tcnt, cnt, n_players, opt_off, opt_action, brows, tail, ov)
        short, lng = np.flatnonzero(~long), np.flatnonzero(long)
        brows = np.asarray([] if brows is None else brows, np.int64)
        where = np.full(B, -1, np.int64)

        def part(rows):
            where[:] = -1
            where[rows] = np.arange(len(rows))
            off, act = take_options(opt_off, opt_action, rows)
            br = where[brows[where[brows] >= 0]] if len(brows) else np.zeros(0, np.int64)
            if x is not None:
                xx = x[rows]
            else:
                o0, o1 = sp[0][rows], sp[0][rows + 1]
                xo = np.zeros(len(rows) + 1, np.int64)
                np.cumsum(o1 - o0, out=xo[1:])
                ii = np.repeat(o0 - xo[:-1], o1 - o0) + np.arange(xo[-1])
                xx = (xo, sp[1][ii], sp[2][ii])
            return (xx, np.asarray(n_players)[rows], off, act, br,
                    None if tail is None else tail.take(rows), None if ov is None else ov.take(rows))
        h = None
        if len(short):
            xx, nn, off, act, br, tt, oo = part(short)
            h = (self._launch1(xx, nn, off, act, br if len(br) else None, tt, oo) if x is not None else
                 self._launch_sparse1(*xx, nn, off, act, br if len(br) else None, tt, oo))
        xx, nn, off, act, br, tt, oo = part(lng)
        if x is None:
            from .protocol import densify
            xx = densify(*xx, self.F)
        o, aux = self._eager_rows(xx, nn, off, act, br, tt, oo, cnt[lng])
        self.long_rows += len(lng)
        return _Split(h=h, o=o, aux=aux, short=short, long=lng, B=B, n_players=n_players, opt_off=opt_off,
                      opt_action=opt_action, brows=brows)

    def _launch_parts(self, x, sp, n_players, opt_off, opt_action, brows, tail, ov, parts):
        """A batch as length-sorted chunks (`chunk_rows`' `parts`): each chunk's rows (dense `x` or sparse `sp`), its
        options, belief rows, tail and overrides taken, launched as a batch of its own -> a `_Parts` handle."""
        B = len(n_players)
        brows = np.asarray([] if brows is None else brows, np.int64)
        where = np.full(B, -1, np.int64)
        hs = []
        for rows in parts:
            where[:] = -1
            where[rows] = np.arange(len(rows))
            off, act = take_options(opt_off, opt_action, rows)
            br = where[brows[where[brows] >= 0]] if len(brows) else np.zeros(0, np.int64)
            tt = None if tail is None else tail.take(rows)
            oo = None if ov is None else ov.take(rows)
            nn = np.asarray(n_players)[rows]
            if x is not None:
                hs.append(self._launch1(x[rows], nn, off, act, br if len(br) else None, tt, oo))
            else:
                o0, o1 = sp[0][rows], sp[0][rows + 1]
                xo = np.zeros(len(rows) + 1, np.int64)
                np.cumsum(o1 - o0, out=xo[1:])
                ii = np.repeat(o0 - xo[:-1], o1 - o0) + np.arange(xo[-1])
                hs.append(self._launch_sparse1(xo, sp[1][ii], sp[2][ii], nn, off, act, br if len(br) else None, tt, oo))
        self.chunked += 1
        return _Parts(parts=parts, hs=hs, B=B, n_players=n_players, opt_off=np.asarray(opt_off, np.int64),
                      opt_action=opt_action, brows=brows)

    def _result_parts(self, h):
        """`result` of a `_Parts`: each chunk's answers put back in the batch's row (and option) order."""
        nb = len(h.brows)
        pri = np.zeros(len(h.opt_action), np.float32)
        value = np.zeros((h.B, MAX_SEATS), np.float32)
        margin = np.zeros((h.B, MAX_SEATS), np.float32)
        aux = np.zeros((nb, self.W), np.float32) if nb else None
        for rows, hh in zip(h.parts, h.hs):
            r = self.result(hh)
            lo, hi = h.opt_off[rows], h.opt_off[rows + 1]
            idx = np.repeat(lo, hi - lo) + (np.arange(int((hi - lo).sum())) - np.repeat(np.cumsum(hi - lo) - (hi - lo), hi - lo))
            pri[idx] = r[0]
            value[rows] = r[1]
            margin[rows] = r[2]
            if nb and len(r) > 3:
                aux[np.isin(h.brows, rows)] = r[3]
        if nb:
            return pri, value, margin, aux
        return pri, value, margin

    def _eager_rows(self, x, n_players, opt_off, opt_action, brows, tail, ov, cnt):
        """Rows of more than LONG_T tokens through the eager forward, in groups whose attention matrices stay
        under LONG_BYTES (at least one row a group) -> (their outputs [L, actions + 2 seats] as the graphs'
        `out`, their root belief rows' auxiliary predictions [len(brows), W] or None)."""
        from . import rows as R
        from .net import aux_answer
        net, dev = self.net, self.dev
        heads = net.blocks[0].h if len(net.blocks) else 1
        L = x.shape[0]
        o = np.zeros((L, self.A + 2 * self.S), np.float32)
        aux = np.zeros((len(brows), max(1, self.W)), np.float32) if len(brows) else None
        isb = np.zeros(L, bool)
        isb[brows] = True
        bpos = np.cumsum(isb) - 1                     # a belief row's index among `brows`
        a = 0
        with torch.no_grad():
            while a < L:
                b, cost = a, 0
                while b < L and (b == a or cost + 4 * heads * int(cnt[b]) ** 2 <= LONG_BYTES):
                    cost += 4 * heads * int(cnt[b]) ** 2
                    b += 1
                rows = np.arange(a, b)
                off, act = take_options(opt_off, opt_action, rows)
                kw = {}
                tt = None if tail is None else tail.take(rows)
                if net.tails:
                    kw['tail'] = tt.to_torch(dev)
                if net.option_pointers:
                    kw['ov'] = ov_torch(R.resolve(tt, ov.take(rows), off, act, self.lay), dev)
                xt = torch.from_numpy(np.ascontiguousarray(x[a:b], dtype=np.float32)).to(dev)
                nt = torch.from_numpy(np.asarray(n_players[a:b], np.int64)).to(dev)
                gb = np.flatnonzero(isb[a:b])
                if len(gb):
                    plog, vlog, mar, raw = net(xt, nt, aux=True, aux_rows=torch.from_numpy(gb).to(dev), **kw)
                    pred = aux_answer(net, raw)
                    aux[bpos[a + gb]] = torch.cat([pred[t[0]].float() for t in self.layout], 1).cpu().numpy()
                else:
                    plog, vlog, mar = net(xt, nt, **kw)
                o[a:b] = torch.cat([plog.float(), torch.softmax(vlog.float(), 1), mar.float()], 1).cpu().numpy()
                a = b
        return o, aux

    def _graph_tails(self, B, x, sp, P, tcnt, cnt, n_players, opt_off, opt_action, brows, tail, ov):
        """One graph launch of a dense batch of a network with tails / overrides whose rows are at most LONG_T tokens
        (a sparse one: `_graph_tails_sparse`): the copied features, the extended `full` (`token_index_tail`), the tail
        padded to NTp entries and the overrides resolved on the host (`rows.resolve`) padded to NOp (docs/az.md 43.5)
        -> `result`'s handle."""
        from . import rows as R
        if x is None:
            raise RuntimeError('internal: _graph_tails of a sparse batch (_graph_tails_sparse packs those)')
        net = self.net
        Bp = -(-B // ROW_STEP) * ROW_STEP
        Tp = -(-int(cnt.max()) // T_STEP) * T_STEP
        nb = 0 if brows is None else len(brows)
        if nb and not self.layout:
            raise RuntimeError('TokenGraphRunner: root belief rows for a network without the auxiliary head')
        NT = 0
        if tail is not None:
            tr, ts, tsl, tv = tail.padded()
            tr, ts, tsl = (np.asarray(a, np.int64) for a in (tr, ts, tsl))
            tv = np.asarray(tv, np.float32)
            self._check_tail_host(tr, ts, tsl, tv, B)
            NT = len(ts)
        res = R.resolve(tail, ov, opt_off, opt_action, self.lay) if net.option_pointers else None
        NO = 0 if res is None else len(res['row'])
        if NO:
            self._check_ov_host(res, P, tr if NT else None, ts if NT else None, B, NT)
        NTp, NOp = tail_bucket(NT), tail_bucket(NO)
        o, L = self._layout(Bp, Tp, NTp, NOp)
        _, _, bo, R0 = self._up_head(0, nb)
        slot, (pin_x, pin_i, pin_out, pin_aux) = self._slot(Bp, R0 + L)
        key = ('d', slot)
        graph, xs, reg, _, out, aux = self._graph(Bp, Tp, NTp, NOp)
        self._copy_rows(x, P, pin_x[:B].numpy())
        I = pin_i.numpy()
        pn = I[R0 + o['n']:R0 + o['n'] + Bp]
        pn[:B] = n_players
        pn[B:] = 2
        pt = I[R0 + o['ntok']:R0 + o['ntok'] + Bp]
        pt[:B] = cnt
        pt[B:] = 1
        token_index_tail(P, tcnt, self.n_tok, I[R0 + o['full']:R0 + o['full'] + Bp * Tp].reshape(Bp, Tp))
        if nb:
            I[bo:bo + nb] = brows
        if net.tails and NTp:
            a = I[R0 + o['tint']:R0 + o['tint'] + 3 * NTp].reshape(3, NTp)     # (the graph's rows, segments, slots)
            for i, (v, pad) in enumerate(((tr, 0), (ts, self.pad_seg), (tsl, self.pad_slot))):
                a[i, :NT] = v
                a[i, NT:] = pad
            t0 = R0 + o['tval']
            a = I[t0:t0 + (NTp * self.Wt + 1) // 2].view(np.float32)[:NTp * self.Wt].reshape(NTp, self.Wt)
            a[:NT] = tv
            a[NT:] = 0.0
        if net.option_pointers and NOp:
            a = I[R0 + o['oint']:R0 + o['oint'] + 4 * NOp].reshape(4, NOp)    # (rows, actions, refs, columns)
            for i, (k, pad) in enumerate((('row', 0), ('act', self.A), ('ref', -1), ('col', 0))):
                a[i, :NO] = res[k]
                a[i, NO:] = pad
        with torch.inference_mode(False):
            st, up = self._upload(key, pin_i, R0 + L)
            torch.cuda.current_stream(self.dev).wait_event(up)
            self._dense_rows(xs, pin_x, B)
            reg.copy_(st[R0:R0 + L])
            self._flush_downs(True)
            graph.replay()
            ev = self._stage_down(key, out, aux, B, nb, st[bo:bo + nb], pin_out, pin_aux)
        self.replays += 1
        return (ev, slot, B, n_players, opt_off, opt_action, nb)

    def _graph_tails_sparse(self, B, sp, n_players, opt_off, opt_action, brows, tail, ov):
        """`_graph_tails` of a sparse batch, packed off the GIL (`_pack_sparse`, `_pack_tail`; docs/az.md 43.12): the
        same inputs, bit for bit -> `result`'s handle, or None (nothing launched) when a row has more than LONG_T
        tokens."""
        from . import rows as R
        net = self.net
        Bp = -(-B // ROW_STEP) * ROW_STEP
        nb = 0 if brows is None else len(brows)
        if nb and not self.layout:
            raise RuntimeError('TokenGraphRunner: root belief rows for a network without the auxiliary head')
        nnz = int(sp[0][-1])
        NT = tail.NT if tail is not None else 0
        NO = ov.NO if ov is not None else 0          # (`rows.resolve`: one entry per override)
        NTp, NOp = tail_bucket(NT), tail_bucket(NO)
        # (the head presence: only for the overrides' checks)
        pres = np.empty(B * self.n_tok, bool) if NO else None
        _, vo, bo, R0 = self._up_head(nnz, nb)
        o, Lmax = self._layout(Bp, self.t_cap, NTp, NOp)
        slot, (_, pin_i, pin_out, pin_aux) = self._sslot(Bp, R0 + Lmax)
        I = pin_i.numpy()
        try:
            Tp, _ = self._pack_sparse(sp, None if tail is None else np.ascontiguousarray(tail.off, np.int64), LONG_T, Bp,
                                      I, R0, o, pres)
            if not Tp:
                self.sfree.append(slot)
                return None
            res = R.resolve(tail, ov, opt_off, opt_action, self.lay) if net.option_pointers else None
            if (0 if res is None else len(res['row'])) != NO:
                raise RuntimeError(f'internal: {NO} overrides resolved into {len(res["row"])}')
            if NO:
                tr = np.repeat(np.arange(B, dtype=np.int64), tail.counts()) if NT else None
                ts = tail.seg.astype(np.int64) if NT else None
                self._check_ov_host(res, pres.reshape(B, self.n_tok), tr, ts, B, NT)
            o, L = self._layout(Bp, Tp, NTp, NOp)
            graph, xs, reg, _, out, aux = self._graph(Bp, Tp, NTp, NOp)
            if net.tails and NTp:
                self._pack_tail(tail, NTp, I, (R0 + o['tint'], R0 + o['tval']))
        except Exception:
            self.sfree.append(slot)
            raise
        pn = I[R0 + o['n']:R0 + o['n'] + Bp]
        pn[:B] = n_players
        pn[B:] = 2
        if nb:
            I[bo:bo + nb] = brows
        if net.option_pointers and NOp:
            a = I[R0 + o['oint']:R0 + o['oint'] + 4 * NOp].reshape(4, NOp)    # (rows, actions, refs, columns)
            for i, (k, pad) in enumerate((('row', 0), ('act', self.A), ('ref', -1), ('col', 0))):
                a[i, :NO] = res[k]
                a[i, NO:] = pad
        key = ('s', slot)
        with torch.inference_mode(False):
            st, up = self._upload(key, pin_i, R0 + L)
            torch.cuda.current_stream(self.dev).wait_event(up)
            self._scatter(xs, st[:nnz], st[vo:vo + (nnz + 1) // 2].view(torch.float32)[:nnz])
            reg.copy_(st[R0:R0 + L])
            self._flush_downs(True)
            graph.replay()
            ev = self._stage_down(key, out, aux, B, nb, st[bo:bo + nb], pin_out, pin_aux)
        self.replays += 1
        return (ev, ('s', slot), B, n_players, opt_off, opt_action, nb)

    def _result_split(self, h):
        """`result` of a `_Split`: the graphs' rows and the eager long rows merged in row order."""
        o = np.zeros((h.B, self.A + 2 * self.S), np.float32)
        nb = len(h.brows)
        aux = np.zeros((nb, max(1, self.W)), np.float32) if nb else None
        o[h.long] = h.o
        if h.h is not None:
            ev, slot, Bs, _, _, _, nbs = h.h
            self._wait_down(ev)
            bufs = self.slots
            if isinstance(slot, tuple):
                bufs, slot = self.sslots, slot[1]
            o[h.short] = bufs[slot][2][:Bs].numpy()
            if nbs:
                aux[np.isin(h.brows, h.short)] = bufs[slot][3][:nbs].numpy()
            (self.sfree if bufs is self.sslots else self.free).append(slot)
        if nb and h.aux is not None:
            aux[np.isin(h.brows, h.long)] = h.aux
        A, S, B = self.A, self.S, h.B
        rows = np.repeat(np.arange(B), np.diff(h.opt_off))
        pri = segment_softmax_np(o[rows, h.opt_action], h.opt_off)
        value = np.zeros((B, MAX_SEATS), np.float32)
        margin = np.zeros((B, MAX_SEATS), np.float32)
        value[:, :S] = o[:, A:A + S]
        margin[:, :S] = o[:, A + S:A + 2 * S]
        value[np.arange(MAX_SEATS)[None, :] >= np.asarray(h.n_players)[:, None]] = 0.0
        if nb:
            return pri, value, margin, aux[:, :self.W]
        return pri, value, margin

    def result(self, h):
        if isinstance(h, _Parts):
            return self._result_parts(h)
        if isinstance(h, _Split):
            return self._result_split(h)
        ev, slot, B, n_players, opt_off, opt_action, nb = h
        self._wait_down(ev)
        bufs = self.slots
        if isinstance(slot, tuple):                  # (a sparse batch's buffers, `launch_sparse`)
            bufs, slot = self.sslots, slot[1]
        o = bufs[slot][2][:B].numpy()
        A, S = self.A, self.S
        rows = np.repeat(np.arange(B), np.diff(opt_off))
        pri = segment_softmax_np(o[rows, opt_action], opt_off)
        value = np.zeros((B, MAX_SEATS), np.float32)
        margin = np.zeros((B, MAX_SEATS), np.float32)
        value[:, :S] = o[:, A:A + S]
        margin[:, :S] = o[:, A + S:A + 2 * S]
        value[np.arange(MAX_SEATS)[None, :] >= np.asarray(n_players)[:, None]] = 0.0
        aux = bufs[slot][3][:nb].numpy().copy() if nb else None
        (self.sfree if bufs is self.sslots else self.free).append(slot)
        if nb:
            return pri, value, margin, aux
        return pri, value, margin

    def evaluate(self, x, n_players, opt_off, opt_action, brows=None, tail=None, ov=None):
        return self.result(self.launch(x, n_players, opt_off, opt_action, brows, tail, ov))


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
    net.EncodingMismatch; `allow_mismatch=True` skips it (tools/selfplay_digest.py).

    Variable-length token sets (docs/az.md 43.7): a token network with tail segments / option_pointers
    takes each batch's tail and overrides (`LeafBatch.tail` / `.ov`, `rows.Tail` / `rows.Overrides`; the
    overrides resolved on the host, `rows.resolve`); every other network refuses a batch that carries them,
    and the evaluator of a game that declares tails (`game_info()['tails']`) refuses at its construction any
    network but a token network with the same tail segments (ValueError, R4)."""

    def __init__(self, model, device='cpu', graphs=None, workers=1, allow_mismatch=False, max_rows=None,
                 token_chunks=1):
        if max_rows is not None and int(max_rows) < 1:
            raise ValueError(f'max_rows {max_rows}: at least 1 (None: every batch whole)')
        # a token network's CUDA graphs: each batch as at most this many length-sorted chunks (docs/az.md 43.11;
        # opt-in, the run setting infer_token_chunks: the answers move by float32 rounding); 1 = the batch whole
        self.token_chunks = max(1, int(token_chunks or 1))
        self.max_rows = None if max_rows is None else int(max_rows)
        self.pieces = 0                              # batches split by max_rows
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.width = None if allow_mismatch else getattr(model, 'features', None)
        self.amp = use_amp(model, self.device)
        self._check_game_tails()
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
        # of `seconds`, those of the CUDA graphs' `launch` (packing, the upload, the replay queued) and `result` (the
        # wait for the device and the download) in drive's pipelined loop: self-play's `timing` metrics (docs/az.md 44.2)
        self.launch_s = 0.0
        self.result_s = 0.0
        # eager batches on worker threads (module docstring); None: computed in launch()
        self.workers = max(1, int(workers)) if self.fast is None else 1
        self.pool = None
        if self.workers > 1:
            import threading
            from concurrent.futures import ThreadPoolExecutor
            self.pool = ThreadPoolExecutor(self.workers, thread_name_prefix='zero-infer')
            self.lock = threading.Lock()

    @property
    def tailnet(self):
        """The network takes tails or overrides (a token network with tail segments or option_pointers)."""
        return bool(getattr(self.model, 'tails', None)) or bool(getattr(self.model, 'option_pointers', False))

    def _check_game_tails(self):
        """R4 (docs/az.md 43.8): the network's game declares tail segments (`games.info(game)['tails']`) -> the
        network must be a token network with exactly those tail segments (index, name, width, head slots), else
        ValueError (an MLP / GridNet / other layout would drop every row's tail). A game this build cannot
        describe is left to the encoding guard (a batch with a tail is refused anyway: `_tail_kw`)."""
        game = getattr(self.model, 'config', {}).get('game')
        if game is None:
            return
        try:
            from . import games as G
            want = G.info(game).get('tails') or []
        except Exception:                          # noqa: BLE001 - no engine / not this build's game
            return
        if not want:
            return
        want = [(int(t['seg']), str(t['name']), int(t['width']), int(t['head'])) for t in want]
        have = self.model.tail_segments() if hasattr(self.model, 'tail_segments') else None
        if have != want:
            c = getattr(self.model, 'config', {})
            raise ValueError(f'NetEvaluator: the game {game!r} has tail segments {want} (index, name, width, head slots); '
                             f'the network (arch {c.get("arch", "mlp")!r}) has {have}: it would drop or misread the rows\' '
                             f'tails (a token network of the game\'s layout is needed; docs/az.md 43.8)')

    def _tail_kw(self, tail, ov):
        """A batch's tail / overrides as the network takes them: {} for a network without tails or pointers (which
        refuses a batch with either: R5), else {'tail': rows.Tail, 'ov': rows.Overrides} (each one its network
        needs, required)."""
        if not self.tailnet:
            if tail is not None or ov is not None:
                c = getattr(self.model, 'config', {})
                raise ValueError(f'NetEvaluator: a batch with tail tokens / pointer overrides for a network without tail '
                                 f'segments or option_pointers (arch {c.get("arch", "mlp")!r}): it would drop them (R5)')
            return {}
        kw = {}
        for k, v, need in (('tail', tail, bool(self.model.tails)), ('ov', ov, self.model.option_pointers)):
            if need and v is None:
                raise ValueError(f'NetEvaluator: the network needs each batch\'s {k} (rows.{"Tail" if k == "tail" else "Overrides"}'
                                 f', possibly empty; R5)')
            if not need and v is not None:
                raise ValueError(f'NetEvaluator: a batch with {k} for a network that takes none (R5)')
            if need:
                kw[k] = v
        return kw

    def _torch_tail(self, kw, opt_off, opt_action):
        """`_tail_kw`'s host tail / overrides -> the forward's (`tail` on the device; the overrides resolved on
        the host, `rows.resolve`)."""
        if not kw:
            return {}
        out = {}
        if 'tail' in kw:
            out['tail'] = kw['tail'].to_torch(self.device)
        if 'ov' in kw:
            from . import rows as R
            out['ov'] = ov_torch(R.resolve(kw.get('tail'), kw['ov'], np.asarray(opt_off), np.asarray(opt_action),
                                           self.model.config['layout']), self.device)
        return out

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
        _say(f'token network CUDA graphs on (float32; batches padded to {ROW_STEP} rows x {T_STEP} tokens'
             + (f'; at most {self.token_chunks} length-sorted chunks a batch' if self.token_chunks > 1 else '') + ')')
        return TokenGraphRunner(self.model, chunks=self.token_chunks)

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

    def set_network(self, model):
        """New weights from a loaded network `model`: `set_weights` of its state, unless its readers differ from
        the evaluator's network's (docs/az.md 36: a reader is part of the trunk's input - a state cannot carry it
        into a network without one, and `set_weights` refuses such a state): then the evaluator takes `model`
        itself, on its device, its CUDA graphs rebuilt (a rolling evaluator of a run whose readers were added at
        its last training). So too when both networks have an auxiliary head and the heads differ (targets or
        parts: a head widened, docs/az.md 37 - its state has other shapes)."""
        from .net import reader_layout
        a, b = model.config, self.model.config
        head = bool(a.get('aux')) and bool(b.get('aux')) and any(a.get(k) != b.get(k) for k in ('aux', 'aux_parts'))
        if not head and reader_layout(a.get('readers')) == reader_layout(b.get('readers')):
            return self.set_weights(model.state_dict())
        self.model = model.to(self.device).eval()
        self._check_game_tails()
        if isinstance(self.fast, TokenGraphRunner):
            self.fast = TokenGraphRunner(self.model, chunks=self.token_chunks)
        elif self.fast is not None:
            self.fast = GraphRunner(self.model)

    def _check_width(self, x):
        if self.width is not None and x.shape[1] != self.width:
            from .net import EncodingMismatch
            c = getattr(self.model, 'config', {})
            raise EncodingMismatch(f'a leaf batch of {x.shape[1]:,} features per row for a network of {self.width:,} '
                                   f'(game {c.get("game")!r}, arch {c.get("arch", "mlp")!r}): the network was built for '
                                   f'another encoding than its driver feeds it')

    def _pieces(self, x, n_players, opt_off, opt_action, extra, tail=None, ov=None):
        """`evaluate` of a batch over max_rows rows: consecutive pieces of at most max_rows rows, their
        answers concatenated (priors in option order, rows in row order, belief rows in row order).
        Through the CUDA graphs two pieces are in flight (the next one packed and uploaded while the
        device runs the last). A batch's tail / overrides are cut with it (docs/az.md 43.7)."""
        B, M = x.shape[0], self.max_rows
        self.pieces += 1
        opt_off = np.asarray(opt_off)
        kw = self._tail_kw(tail, ov)
        parts = []
        for a in range(0, B, M):
            b = min(B, a + M)
            off = (opt_off[a:b + 1] - opt_off[a]).astype(opt_off.dtype)
            ex = None if extra is None else extra[a:b]
            parts.append((x[a:b], n_players[a:b], off, opt_action[opt_off[a]:opt_off[b]], ex,
                          {k: v.take(np.arange(a, b)) for k, v in kw.items()}))
        res = []
        if self.fast is not None and self.fast_aux:
            # (a token network's graphs answer every piece, belief rows included: pipelined)
            self._check_width(x)
            t0 = time.perf_counter()
            pending = []
            for xx, nn, oo, aa, ex, tk in parts:
                br = belief_rows(ex)
                if len(br):
                    self._check_head()
                pending.append((self.fast.launch(xx, nn, oo, aa, *((br,) if len(br) else ()), **tk), xx.shape[0]))
                if len(pending) >= 2:
                    h, rows = pending.pop(0)
                    res.append(self.fast.result(h))
            for h, rows in pending:
                res.append(self.fast.result(h))
            self.calls += len(parts)
            self.graph_calls += len(parts)
            self.leaves += B
            self.seconds += time.perf_counter() - t0
        else:
            res = [self.evaluate(*p[:5], _whole=True, **p[5]) for p in parts]
        out = (np.concatenate([r[0] for r in res]).astype(np.float32, copy=False), np.concatenate([r[1] for r in res]),
               np.concatenate([r[2] for r in res]))
        aux = [r[3] for r in res if len(r) > 3]
        if aux:
            return out + (np.concatenate(aux),)
        return out

    @torch.inference_mode()
    def evaluate(self, x, n_players, opt_off, opt_action, extra=None, tail=None, ov=None, _whole=False):
        """numpy in -> (priors [options], value [B, MAX_SEATS], margin [B, MAX_SEATS]) numpy; with root
        belief rows in `extra` (docs/search.md 3.8) a fourth item, their auxiliary predictions
        [belief rows, aux width]. With `max_rows`, a bigger batch goes in pieces (`_pieces`). `tail` /
        `ov`: the batch's (`rows.Tail` / `rows.Overrides`, docs/az.md 43.7; a network with tail segments /
        option_pointers needs them, any other refuses them)."""
        if self.max_rows is not None and not _whole and x.shape[0] > self.max_rows:
            return self._pieces(x, n_players, opt_off, opt_action, extra, tail, ov)
        t0 = time.perf_counter()
        self._check_width(x)
        kw = self._tail_kw(tail, ov)
        brows = belief_rows(extra)
        if len(brows):
            self._check_head()
        if self._graphable(opt_off, brows):
            r = self.fast.evaluate(x, n_players, opt_off, opt_action, *((brows,) if len(brows) else ()), **kw)
            self.calls += 1
            self.graph_calls += 1
            self.leaves += x.shape[0]
            self.seconds += time.perf_counter() - t0
            return r
        dev = self.device
        B = x.shape[0]
        xt = torch.from_numpy(x).to(dev, non_blocking=True)
        nt = torch.from_numpy(n_players.astype(np.int64)).to(dev)
        tk = self._torch_tail(kw, opt_off, opt_action)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.amp):
            if len(brows):
                plog, vlog, mar, raw = self.model(xt, nt, aux=True, aux_rows=torch.from_numpy(brows).to(dev), **tk)
            else:
                plog, vlog, mar = self.model(xt, nt, **tk)
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
            from .net import aux_answer
            pred = aux_answer(self.model, raw)
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
        if lb.a.get('xs_val') is not None and self._graphable(lb.opt_off, belief_rows(lb.a.get('extra'))):
            return self.result(self.launch(lb))
        return self.evaluate(lb.x, lb.n_players, lb.opt_off, lb.opt_action, lb.a.get('extra'), *_lb_tail(lb))

    def _launch_sparse(self, lb):
        """A sparse batch through a token network's graphs (`TokenGraphRunner.launch_sparse`), in pieces of
        at most max_rows rows (all launched, collected by `result`) -> a handle ('s', [piece handles], B)."""
        if self.width is not None and int(lb.a['n_features']) != self.width:
            from .net import EncodingMismatch
            raise EncodingMismatch(f'a sparse leaf batch of {int(lb.a["n_features"]):,} features per row for a network of '
                                   f'{self.width:,}: the network was built for another encoding than its driver feeds it')
        extra = lb.a.get('extra')
        xs_off, xs_idx, xs_val = np.asarray(lb.a['xs_off'], np.int64), lb.a['xs_idx'], lb.a['xs_val']
        opt_off = np.asarray(lb.opt_off)
        B = lb.B
        M = self.max_rows or B
        kw = self._tail_kw(*_lb_tail(lb))
        hs = []
        for a in range(0, B, M):
            b = min(B, a + M)
            br = belief_rows(None if extra is None else extra[a:b])
            if len(br):
                self._check_head()
            off = xs_off[a:b + 1] - xs_off[a]
            oo = (opt_off[a:b + 1] - opt_off[a]).astype(opt_off.dtype)
            tk = kw if (a == 0 and b == B) else {k: v.take(np.arange(a, b)) for k, v in kw.items()}
            hs.append(self.fast.launch_sparse(off, xs_idx[xs_off[a]:xs_off[b]], xs_val[xs_off[a]:xs_off[b]],
                                              lb.n_players[a:b], oo, lb.opt_action[opt_off[a]:opt_off[b]],
                                              *((br,) if len(br) else ()), **tk))
        if len(hs) > 1:
            self.pieces += 1
        return ('s', hs, B)

    @property
    def aux_layout(self):
        """The network's auxiliary targets [[name, width, loss], ...] ([]: no auxiliary head)."""
        return list(getattr(self.model, 'aux_layout', None) or [])

    @torch.inference_mode()
    def aux(self, x, n_players, tail=None):
        """The auxiliary head's predictions for rows `x` [B, F] (numpy; n_players [B]) -> {name: float32
        numpy [B, width]} (`net.aux_predict`: a multinomial target's distribution, a poisson target's
        rates, an mse target's values); {} for a network without the head. One forward pass, outside
        the search's batches (its evaluations never compute the head). `tail`: the rows' (`rows.Tail`; a
        network with tail segments needs it: docs/az.md 43.7)."""
        if not self.aux_layout:
            return {}
        from .net import aux_answer
        return {k: v.float().cpu().numpy() for k, v in aux_answer(self.model, self._aux_raw(x, n_players, tail)).items()}

    def _aux_raw(self, x, n_players, tail=None):
        self._check_width(x)
        xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.device)
        nt = torch.from_numpy(np.asarray(n_players, np.int64)).to(self.device)
        tk = {}
        if not self.tailnet:
            self._tail_kw(tail, None)                # (a tail for a network without tails: refused, R5)
        else:
            if self.model.tails:
                if tail is None:
                    raise ValueError("NetEvaluator: the network needs the rows' tail (rows.Tail, possibly empty; R5)")
                tk['tail'] = tail.to_torch(self.device)
            elif tail is not None:
                raise ValueError('NetEvaluator: a tail for a network without tail segments (R5)')
            if self.model.option_pointers:
                # (the auxiliary outputs read no policy logit: no options, so no overrides - an empty set)
                tk['ov'] = {k: torch.zeros(0, dtype=torch.long, device=self.device) for k in ('row', 'act', 'ref', 'col')}
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.amp):
            return self.model(xt, nt, aux=True, **tk)[3]

    @torch.inference_mode()
    def reader(self, x, n_players, tail=None):
        """The readers' predictions for rows `x` [B, F] (docs/az.md 36) -> {target: float32 numpy [B, width]}
        (`net.reader_predict`); {} for a network without readers. One forward pass, outside the search's
        batches (as `aux`; `tail` as its)."""
        if not getattr(self.model, 'reader_specs', None):
            return {}
        from .net import reader_predict
        return {k: v.float().cpu().numpy() for k, v in reader_predict(self.model, self._aux_raw(x, n_players, tail)).items()}

    def _graphable(self, opt_off, brows):
        """The batch goes through the CUDA graphs (a conv network's skip batches with root belief
        rows; a token network's answer them)."""
        return self.fast is not None and len(opt_off) > 1 and (self.fast_aux or not len(brows))

    def launch(self, lb):
        """Start evaluating a LeafBatch; `result(handle)` -> what `ev(lb)` returns. Eager
        evaluators compute at once (or on a worker thread, `workers` > 1)."""
        extra = lb.a.get('extra')
        if lb.a.get('xs_val') is not None and self._graphable(lb.opt_off, belief_rows(extra)):
            t0 = time.perf_counter()
            h = self._launch_sparse(lb)
            dt = time.perf_counter() - t0
            self.seconds += dt
            self.launch_s += dt
            return h
        tl = _lb_tail(lb)
        if self.max_rows is not None and lb.B > self.max_rows:
            # (in pieces, answered now: `_pieces`)
            return ('d', self.evaluate(lb.x, lb.n_players, lb.opt_off, lb.opt_action, extra, *tl), 0)
        if self.pool is not None:
            return ('f', self.pool.submit(self.evaluate, lb.x, lb.n_players, lb.opt_off, lb.opt_action, extra, *tl), 0)
        t0 = time.perf_counter()
        brows = belief_rows(extra)
        if self._graphable(lb.opt_off, brows):
            self._check_width(lb.x)
            if len(brows):
                self._check_head()
            h = ('g', self.fast.launch(lb.x, lb.n_players, lb.opt_off, lb.opt_action, *((brows,) if len(brows) else ()),
                                       **self._tail_kw(*tl)), lb.B)
            self.launch_s += time.perf_counter() - t0
        else:
            h = ('d', self.evaluate(lb.x, lb.n_players, lb.opt_off, lb.opt_action, extra, *tl), 0)
        self.seconds += time.perf_counter() - t0
        return h

    def result(self, h):
        kind, v, B = h
        if kind == 'f':
            return v.result()
        if kind == 'd':
            return v
        if kind == 's':
            # (a sparse batch's pieces, in order)
            t0 = time.perf_counter()
            res = [self.fast.result(x) for x in v]
            self.calls += len(res)
            self.graph_calls += len(res)
            self.leaves += B
            dt = time.perf_counter() - t0
            self.seconds += dt
            self.result_s += dt
            if len(res) == 1:
                return res[0]
            out = (np.concatenate([r[0] for r in res]).astype(np.float32, copy=False),
                   np.concatenate([r[1] for r in res]), np.concatenate([r[2] for r in res]))
            aux = [r[3] for r in res if len(r) > 3]
            return out + (np.concatenate(aux),) if aux else out
        t0 = time.perf_counter()
        r = self.fast.result(v)
        self.calls += 1
        self.graph_calls += 1
        self.leaves += B
        dt = time.perf_counter() - t0
        self.seconds += dt
        self.result_s += dt
        return r


def _lb_tail(lb):
    """A LeafBatch's tail and overrides (`LeafBatch.tail` / `.ov`: rows.Tail / rows.Overrides, docs/az.md 43.7)
    -> (tail, ov), None each when the batch has none (a game without tails or pointer overrides)."""
    return getattr(lb, 'tail', None), getattr(lb, 'ov', None)


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
