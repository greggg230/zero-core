"""zero-onnx/1: a zero-core network as one self-contained ONNX graph (docs/az.md 41).

A native worker (zero-farm) answers a run's network batches through ONNX Runtime without any code of
its own for the network: the graph takes the driver's sparse network batch exactly as the driver
produces it (a driver created with `sparse=True`, protocol.py) and returns what every zero-core
evaluator computes before its generic last step. Everything specific to a network - the token
network's packing of its objects, its readers, its heads, the auxiliary targets' transforms - is
inside the graph, exported from the same `net.py` the run trains, so a new architecture needs no
worker change, only an exporter that can export it (and the parity gate, `zero.onnx_parity`, that
proves it did).

zero-onnx/1 (every architecture):

| | name | type | shape | |
|---|---|---|---|---|
| in | `x_ptr` | i64 | [B+1] | the rows' offsets into the non-zeros (`xs_off`) |
| in | `x_idx` | i32 | [nnz] | feature indexes, ascending per row (`xs_idx`; u32 in the driver, F < 2^31) |
| in | `x_val` | f32 | [nnz] | their values (`xs_val`) |
| in | `n_players` | i64 | [B] | |
| in | `belief` | bool | [B] | the rows whose auxiliary answer is wanted (`extra` == ROW_BELIEF) |
| out | `policy` | f32 | [B, A] | raw logits over the game's actions |
| out | `value` | f32 | [B, 8] | P(win) per relative seat (softmax over the seats present), 0 beyond n_players |
| out | `margin` | f32 | [B, 8] | the margin head as it is (0 beyond the network's seats) |
| out | `aux` | f32 | [B, W] | `net.aux_answer` per target of the head's layout (softmax / exp / raw), 0 on rows without `belief`; W = 0 without the head |

The worker's generic step (`answer`, the same for every network): each row's priors = the softmax of
`policy[row, its legal actions]` (max-subtracted, float32), `value` and `margin` as they are, the
auxiliary answer = `aux[belief rows]` - `NetEvaluator.evaluate`'s answer.

The model's metadata (`metadata`): zero_onnx=1, protocol, game, arch, features, actions, seats,
aux_layout (JSON [[name, width, loss], ...]), aux_width, packing, checkpoint, checkpoint_sha256,
opset, unique_scatters, exporter, torch.

zero-onnx/2 (docs/az.md 46; `ZeroOnnx2`, `onnx_version`): a token network with tail segments, pointer overrides or
gates - MTG Zero's layout v7 - takes zero-onnx/1's inputs plus the driver's options and variable-length rows
(`opt_off` i64 [B+1], `opt_action` i32, `t_off` i64 [B+1], `t_seg` i32, each tail slot as `t_slot_hi` / `t_slot_lo`
i32 - its high and low 16 bits, `slot_pieces` -, `t_val` f32 - each tail token's floats consecutive -, `ov_off` i64
[B+1], `ov_opt` i32, and each packed override token as `ov_seg` / `ov_col` / `ov_slot_hi` / `ov_slot_lo` i32,
`override_pieces`) and answers zero-onnx/1's outputs plus `refused` i64 [B] (per row, the tail tokens and overrides
the network refuses: 0 everywhere, or the batch is refused); metadata zero_onnx=2 and the network's `tails`,
`option_pointers`, `max_row_tokens`, `tail_width`, `gates`. Packing 'packed' only. No integer in the graph reaches
2^31 (DirectML computes int64 in 32 bits): the host splits slots and tokens, the exporter refuses a constant outside
int32 (`check_int32`), the feeder a piece too large (`int32_bound`). Every other network exports zero-onnx/1, the
same graph as before.

Every exported graph is written without the exporter's per-node annotations (`strip_annotations`: stack traces with
the exporting checkout's file paths and line numbers, doc strings): the same checkpoint exports to the same bytes from
any checkout, and no graph carries a local path.

Every graph a GPU provider runs as the CPU does (docs/az.md 41.2; metadata `unique_scatters` 1 - every packing but
'compact' and 'host', whose policy scatter is `TokenNet._trunk`'s): every scatter writes unique indices into data with
a place for each (`_csr_rows`' spare places for the row ends; the policy's last writer of each action made explicit,
`_scatter_last`; a refused pointer override to a spare place of its own), no integer leaves int32 (`check_int32` on
every export, zero-onnx/1 too), opset 17 (`opset17`: DirectML runs ScatterElements up to version 16). The CPU's answers
are the earlier graph's, bit for bit; `scatter_writes` checks a graph's scatters on real inputs.

How a token network (`net.TokenNet`) packs its objects inside the graph (`packing`):

* 'mask': every slot of the full token axis is a token (the global token and all 170 of MTG
  Zero's slots, as `forward_static` embeds them), the absent ones masked out of the attention (as
  keys) and of the pooling, their pointer logits sent to the dummy column. Static shapes but B: the
  worker needs nothing but the interface.
* 'compact': the present tokens gathered in layout order inside the graph (TopK over the presence,
  T = the batch's longest row: a data-dependent shape), then `forward_static`'s trunk on every row
  padded to T. Also needs nothing but the interface.
* 'packed': in-graph compaction without padding: the batch's N present tokens as one list [N, d]
  (NonZero over the presence) for every per-token layer (LayerNorms, the projections, the MLPs, the
  pointer head: 2-D Gemms), only the attention and the pooling on the rows padded to T (gathered in,
  scattered back). The same interface; the default: the fastest on the CPU (docs/az.md 41).
* 'host': `forward_static`'s signature kept as two more inputs, `tok_full` i64 [B, T] and `tok_n` i64
  [B] (`infer.token_presence` / `token_index`, packed by the caller; metadata `present_index`). Not
  architecture-agnostic (the worker would carry TokenNet's packing): measured, not recommended
  (docs/az.md 41).

An absent slot's floats are never part of the arithmetic in any packing: its identity digits are
zeroed before they are decoded and its token is zero (NaN or junk there changes nothing, as with the
CUDA graphs' host copy, docs/az.md 10.6). Conv (`GridNet`) and MLP (`Net`) networks have no packing:
their forward on the densified rows.

The reference set (`--reference N`, `write_reference`; zero-farm's per-machine self-test, DESIGN.md 2.1
point 5): N real network batch rows - whole batches a driver recorded (`tools/onnx_record.py`, or a
worker's `--record-batches`), taken spread over the files - with torch's answers to them (the eager
`infer.NetEvaluator` on the CPU, batch by batch: what a run's own workers answer), in
`reference.npz`: `meta` (JSON: zero_onnx, game, features, actions, aux_width, checkpoint,
checkpoint_sha256, rows, belief / value rows, batches, tol, sources, torch), the rows as the driver
sends them (`xs_off` i64 [R+1], `xs_idx` i32, `xs_val` f32, `n_players` i64 [R], `extra` u8 [R],
`opt_off` i64 [R+1], `opt_action` i32), `batch_off` i64 [K+1], and the answers: `priors` f32 [options],
`value` f32 [R, 8], `margin` f32 [R, 8], `aux` f32 [belief rows, W]. A worker answers it through its own
evaluator before it plays and must match within 2e-5 (zero_worker `selftest`); the exporter checks the
graph against it through onnxruntime first (`check_reference`), so a checkpoint whose graph fails is
never published.

Exporter: torch 2.x's dynamo-based `torch.onnx.export(..., dynamo=True)` (needs `onnx` and
`onnxscript`) at opset 18, rewritten to opset 17 (`opset17`; LayerNormalization, exact GELU through Erf, the attention
decomposed into MatMul / Softmax with the boolean mask, the policy's per-row sort a TopK). Only this module and
`zero.onnx_parity` import onnx / onnxruntime; nothing on a run's path does.
"""
import copy
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .net import GridNet, Net, TokenNet, _read, aux_answer, varlen_features
from .protocol import MAX_SEATS

PROTOCOL = 'zero-onnx/1'
OPSET = 17
# what torch's dynamo exporter writes (`opset17` rewrites it to OPSET)
EXPORTER_OPSET = 18
INPUTS = ('x_ptr', 'x_idx', 'x_val', 'n_players', 'belief')
HOST_INPUTS = ('tok_full', 'tok_n')
OUTPUTS = ('policy', 'value', 'margin', 'aux')
# zero-onnx/2 (docs/az.md 46): a token network with variable-length token sets - the driver's options, tail tokens and
# pointer overrides as more inputs, and per row the inputs the network refuses as one more output
PROTOCOL2 = 'zero-onnx/2'
VARLEN_INPUTS = ('opt_off', 'opt_action', 't_off', 't_seg', 't_slot_hi', 't_slot_lo', 't_val', 'ov_off', 'ov_opt',
                 'ov_seg', 'ov_col', 'ov_slot_hi', 'ov_slot_lo')
INT32_MAX = (1 << 31) - 1
INPUTS2 = INPUTS + VARLEN_INPUTS
OUTPUTS2 = OUTPUTS + ('refused',)
PACKINGS = ('mask', 'compact', 'packed', 'host')


def _csr_rows(off, n):
    """Each of the `n` elements' row of a CSR over rows (`off` [B+1]), with no data-dependent shape, no scatter with a
    reduction (TensorRT's ONNX parser refuses one and with it the whole graph; DirectML runs none) and every scatter
    writing unique indices into data with a place for each (docs/az.md 41.2: a GPU provider refuses more indices than
    places and leaves the winner of a duplicate undefined). In the merged sequence of the n elements and the B row ends,
    row b's end sits at off[b+1] + b (distinct even for empty rows): a scatter of ones there and a running sum count the
    ends up to each merged position; an element at merged position j is element j - count[j], its row count[j] - a
    scatter of the rows to the elements' places [0, n) and the ends' to their own spare places n + (their index), then
    the first n read by a Gather (not a Slice: DirectML's graph fusion fails on Scatter -> Slice -> Mul). Integers below
    n + B: the same rows as the earlier one-spare-slot scatter and the scatter-add before it (docs/az.md 41.1)."""
    B = off.shape[0] - 1
    dev = off.device
    ends = off[1:] + torch.arange(B, dtype=torch.int64, device=dev)
    marks = torch.zeros(n + B, dtype=torch.int64, device=dev).scatter(0, ends, torch.ones_like(ends))
    count = marks.cumsum(0)
    slot = torch.where(marks > 0, count + (n - 1), torch.arange(n + B, dtype=torch.int64, device=dev) - count)
    rows = torch.zeros(n + B, dtype=torch.int64, device=dev).scatter(0, slot, count)
    return rows.index_select(0, torch.arange(n, dtype=torch.int64, device=dev))


def densify(x_ptr, x_idx, x_val, features):
    """Sparse rows (CSR: `x_ptr` [B+1], `x_idx` [nnz], `x_val` [nnz]) -> f32 [B, features], with no
    data-dependent shape and no scatter with a reduction: each non-zero's row from `_csr_rows` (a scatter of ones at
    the row ends, a running sum, a scatter to unique places), then one scatter of the values (protocol.densify; unique
    indices: a row's feature indexes ascend). Integers throughout: the same rows as the earlier scatter-add (zero-farm
    PROTOCOL.md 8; docs/az.md 41.1, 41.2)."""
    B = x_ptr.shape[0] - 1
    rows = _csr_rows(x_ptr, x_idx.shape[0])
    flat = rows * features + x_idx.long()
    return torch.zeros(B * features, dtype=torch.float32, device=x_val.device).scatter(0, flat, x_val).view(B, features)


def _scatter_last(base, idx, val):
    """base [B, S].scatter(1, idx [B, L], val [B, L]) as the CPU computes it when a row's indices repeat - the last write
    of a row (the highest position) wins, as in torch's CPU scatter and ONNX Runtime's sequential CPU kernel - with every
    write to its own place (docs/az.md 41.2): a GPU provider writes in parallel, so ONNX leaves the winner of a duplicate
    undefined. The writes of each row sorted by (index, position) (TopK on index L + position: unique keys, so no tie
    to break; no integer division); the last of each index's run writes its place, every other write its own spare
    place S + its sorted position past the row's end; the first S places read back by a Gather. Every integer below
    S L + L (`L` static: the caller checks it against int32). The same answer as the plain scatter, bit for bit."""
    B, S = base.shape
    L = idx.shape[1]
    dev = idx.device
    pos = torch.arange(L, dtype=torch.int64, device=dev)
    p = torch.sort(idx * L + pos[None, :], dim=1).indices
    g = idx.gather(1, p)
    nxt = g.index_select(1, torch.clamp(pos + 1, max=L - 1))          # (the next write of the sorted row)
    last = (g != nxt) | (pos == L - 1)[None, :]
    tgt = torch.where(last, g, (S + pos)[None, :])
    out = torch.cat([base, base.new_zeros(B, L)], 1).scatter(1, tgt, val.gather(1, p))
    return out.index_select(1, torch.arange(S, dtype=torch.int64, device=dev))


def _policy_full(gen, acts, lg, flat, cpos, n_tok):
    """The static scatter of a packed token list's pointer logits into the policy, per row (`_scatter_last`): gen [B, A1]
    (the generic logits, the dummy column last), acts / lg [N, K] each packed head token's actions (the dummy column
    `A1 - 1` for none) and logits, flat [B n_tok] the full token axis's presence, cpos [B n_tok] a present position's
    packed index. Each row's writes laid out on its full token axis ([B, n_tok K]: absent slots write the dummy column) -
    a row's packed tokens are in layout order, so a write's place on the axis orders the row's writes as the packed
    list's flat scatter did: the same last writer of every action, bit for bit."""
    B, A1 = gen.shape
    N, K = acts.shape
    fidx = torch.where(flat, cpos, N)
    a = torch.cat([acts, acts.new_full((1, K), A1 - 1)]).index_select(0, fidx).view(B, n_tok * K)
    v = torch.cat([lg, lg.new_zeros(1, K)]).index_select(0, fidx).view(B, n_tok * K)
    return _scatter_last(gen, a, v)


def _check_policy_keys(net):
    """`_scatter_last`'s integers below 2^31 (DirectML computes int64 in 32 bits; docs/az.md 46.7, 41.2): the policy's
    sort keys reach (actions + 1) n_tok K."""
    K = int(net.tok_cols.shape[1])
    if (int(net.actions) + 2) * int(net.n_tok) * K > INT32_MAX:
        raise ValueError(f'zero-onnx: {net.actions} actions x {net.n_tok} token positions x {K} pointer columns: the '
                         f'policy scatter\'s keys pass int32 (docs/az.md 41.2)')


def _identity(sl, idspec, vocab):
    """Identities [B, k] of slots sl [B, k, width] from their one-hot digits (`TokenNet._identity` /
    `_static_identities`' arithmetic, per segment), `vocab` = none."""
    at, idx, ok = idspec['at'], None, None
    for r in idspec['radix']:
        d = sl[..., at:at + r]
        dig = (d @ torch.arange(r, dtype=sl.dtype, device=sl.device)).round().long()
        has = d.sum(-1) > 0.5
        idx = dig if idx is None else idx * r + dig
        ok = has if ok is None else ok & has
        at += r
    return torch.where(ok & (idx < vocab), idx, torch.full_like(idx, vocab))


def slot_tokens(net, x, entry=None):
    """Every token of a TokenNet's full token axis (the global token at 0, then the segments' slots in
    layout order) for rows x [B, F] -> (h [B, n_tok, d], present [B, n_tok] bool, identities [B, n_tok]
    long (`vocab`: none) or None without id_actions). A present slot's token is `forward_static`'s; an
    absent slot's is zero and its identity `vocab` (its digits are zeroed before they are decoded and
    its token replaced: NaN or junk in its floats never reaches the graph's arithmetic)."""
    B = x.shape[0]
    g = net._global(x, entry)
    toks = [(g + net.pos[0])[:, None]]
    pres = [torch.ones(B, 1, dtype=torch.bool, device=x.device)]
    ids = [torch.full((B, 1), net.vocab, dtype=torch.long, device=x.device)] if net.n_id else None
    start = 1
    for s, lin in zip(net.segs, net.seg_in):
        if not s['count']:
            continue                       # (a tail-only segment: no head slot, no feature; docs/az.md 43.5)
        o, k, w = s['offset'], s['count'], s['width']
        sl = x[:, o:o + k * w].reshape(B, k, w)
        p = sl[..., s['present']] > 0.5 if s['present'] is not None else None
        t = lin(sl) + net.pos[start:start + k]
        if s['id'] is not None:
            at, n = s['id']['at'], sum(s['id']['radix'])
            dig = sl[..., at:at + n]
            if p is not None:
                dig = torch.where(p[..., None], dig, torch.zeros_like(dig))
            i = _identity(dig, {'at': 0, 'radix': s['id']['radix']}, net.vocab)
            t = t + net.emb(i)
        else:
            i = torch.full((B, k), net.vocab, dtype=torch.long, device=x.device)
        if p is None:
            p = torch.ones(B, k, dtype=torch.bool, device=x.device)
        else:
            t = torch.where(p[..., None], t, torch.zeros_like(t))
        toks.append(t)
        pres.append(p)
        if ids is not None:
            ids.append(i)
        start += k
    return torch.cat(toks, 1), torch.cat(pres, 1), (torch.cat(ids, 1) if ids is not None else None)


def _trunk_mask(net, h, keep, ids):
    """`TokenNet._trunk` over the full token axis: h [B, n_tok, d] every slot's token, keep [B, n_tok]
    the present ones (attended to, pooled, scoring their actions) -> (z [B, 2 d], logits [B, actions + 1],
    the last column the dummy one)."""
    B = h.shape[0]
    k4 = keep[:, None, None, :]
    for blk in net.blocks:
        h = blk(h, k4)
    h = net.ln(h) * keep[..., None]
    n = keep.sum(1)
    z = torch.cat([h[:, 0], h.sum(1) / n[:, None].to(h.dtype)], 1)
    pc = F.pad(net.gen(z), (0, 1))
    cols = net.tok_cols[None].expand(B, -1, -1)                   # [B, n_tok, K]: the full axis's own columns
    acts = net.tok_acts[None].expand(B, -1, -1)
    if net.n_id:                                                  # actions by identity
        ia = net.tok_idbase[None] + ids
        ia = torch.where((ids < net.vocab) & (ia < net.actions), ia, net.actions)
        acts = torch.where(net.tok_isid[None], ia[..., None], acts)
    acts = torch.where(keep[..., None] & (cols >= 0), acts, net.actions)
    lg = net.ptr(F.gelu(net.pk(h) + net.pq(z)[:, None])).gather(2, cols.clamp(min=0))
    return z, _scatter_last(pc, acts.flatten(1), lg.flatten(1))


def _row_tokens(pres, n_tok):
    """The batch's longest row T (>= 2: the tracer cannot tell T = 1 from a broadcast; a batch of global
    tokens alone gets a padding token) and each row's token positions in the full axis [B, T] (its present
    tokens in layout order, then absent slots as padding: TopK over the presence) - in-graph token_index."""
    n = pres.sum(1)
    lo = min(2, n_tok)
    T = n.max().clamp(min=lo).item()                              # (a data-dependent shape: dynamic in the graph)
    torch._check(T >= lo)
    torch._check(T <= n_tok)
    ar = torch.arange(n_tok, device=pres.device)
    key = torch.where(pres, ar, ar + n_tok)                       # present first, each part in layout order
    return n, T, torch.topk(key, T, dim=1, largest=False, sorted=True).indices


def _trunk_packed(net, h, pres, ids):
    """`TokenNet._trunk` with the tokens packed (in-graph compaction without padding): the per-token layers
    (LayerNorms, the attention's projections, the MLPs, the pointer head) run on the batch's N present tokens
    [N, d] (row-major, each row's in layout order), only the attention and the pooling on the rows padded to
    the longest (gathered in, gathered back), the policy's scatter per row on the full token axis (`_policy_full`).
    N >= 2 is assumed while tracing (MTG Zero's rows hold the global and the context token: N >= 2 B); the graph keeps
    no such guard. -> (z [B, 2 d], logits [B, actions + 1])."""
    B, NT, D = h.shape
    flat = pres.reshape(-1)
    sel = torch.nonzero(flat).squeeze(1)                          # [N] the present tokens, row-major
    N = sel.shape[0]
    torch._check(N >= 2)
    row = sel // NT
    slot = sel - row * NT
    n, T, full = _row_tokens(pres, NT)
    start = n.cumsum(0) - n                                       # each row's first packed token
    kpos = torch.arange(N, device=h.device) - start[row]          # a token's place in its row
    cpos = flat.long().cumsum(0) - 1                              # packed index of a present flat position
    valid = torch.arange(T, device=h.device)[None, :] < n[:, None]
    gidx = torch.where(valid, cpos[(torch.arange(B, device=h.device)[:, None] * NT + full).reshape(-1)].view(B, T),
                       torch.full_like(full, N))                  # padded (b, t) <- packed token, N = a zero row
    back = row * T + kpos                                         # packed token <- padded (b, t)
    bias = torch.where(valid, 0.0, float('-inf'))[:, None, None, :].to(h.dtype)
    hp = h.reshape(B * NT, D).index_select(0, sel)
    H = net.blocks[0].h
    hd = D // H
    for blk in net.blocks:
        qkv = blk.qkv(blk.ln1(hp))
        qkv = torch.cat([qkv, qkv.new_zeros(1, 3 * D)]).index_select(0, gidx.reshape(-1))
        q, k, v = qkv.view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        s = torch.matmul(q, k.transpose(-2, -1)) * (hd ** -0.5) + bias
        a = torch.matmul(torch.softmax(s, -1), v).transpose(1, 2).reshape(B * T, D)
        hp = hp + blk.out(a.index_select(0, back))
        hp = hp + blk.ff2(F.gelu(blk.ff1(blk.ln2(hp))))
    hp = net.ln(hp)
    # the pooling through the padded rows (a scatter-add of the tokens into their rows would be ScatterND with
    # duplicate indices, which ONNX Runtime's CPU kernel adds up on several threads at once: lost updates)
    pad = torch.cat([hp, hp.new_zeros(1, D)]).index_select(0, gidx.reshape(-1)).view(B, T, D)
    z = torch.cat([pad[:, 0], pad.sum(1) / n[:, None].to(hp.dtype)], 1)
    pc = F.pad(net.gen(z), (0, 1))
    cols = net.tok_cols.index_select(0, slot)                     # [N, K]
    acts = net.tok_acts.index_select(0, slot)
    if net.n_id:                                                  # actions by identity
        it = ids.reshape(-1).index_select(0, sel)
        ia = net.tok_idbase.index_select(0, slot) + it
        ia = torch.where((it < net.vocab) & (ia < net.actions), ia, net.actions)
        acts = torch.where(net.tok_isid.index_select(0, slot), ia[:, None], acts)
    acts = torch.where(cols >= 0, acts, net.actions)
    lg = net.ptr(F.gelu(net.pk(hp) + net.pq(z).index_select(0, row))).gather(1, cols.clamp(min=0))
    return z, _policy_full(pc, acts, lg, flat, cpos, NT)


class ZeroOnnx(nn.Module):
    """A network behind the zero-onnx/1 interface (module docstring): `forward(x_ptr, x_idx, x_val,
    n_players, belief[, tok_full, tok_n])` -> (policy, value, margin, aux)."""

    VERSION = 1

    def __init__(self, net, packing='packed'):
        super().__init__()
        self.net = net
        self.tokens = isinstance(net, TokenNet)
        if not isinstance(net, (TokenNet, GridNet, Net)):
            raise TypeError(f'zero-onnx: no export for a {type(net).__name__} (TokenNet, GridNet, Net)')
        vf = varlen_features(net)
        if vf and self.VERSION == 1:  # (R14, docs/az.md 43.8: the fixed-width interface would drop them: zero-onnx/2)
            raise ValueError(f'zero-onnx/1: a network with variable-length token sets ({"; ".join(vf)}) has no zero-onnx/1 graph')
        if self.tokens and packing not in PACKINGS:
            raise ValueError(f'zero-onnx: packing {packing!r} (one of {PACKINGS})')
        self.packing = packing if self.tokens else None
        if self.packing in ('mask', 'packed'):
            _check_policy_keys(net)
        self.layout = list(getattr(net, 'aux_layout', None) or [])
        self.W = sum(int(w) for _, w, _ in self.layout)

    def _core(self, x, n_players, tok_full, tok_n):
        """(policy logits [B, actions], value logits [B, seats], margin [B, seats], raw auxiliary outputs or None)."""
        net, aux = self.net, bool(self.layout)
        if not self.tokens:
            r = net(x, n_players, aux=aux)
            return r[0], r[1], r[2], (r[3] if aux else None)
        rraw = ent = None
        if net.readers is not None:
            rraw, ent = _read(net, x)
        h, pres, ids = slot_tokens(net, x, ent)
        if self.packing == 'mask':
            z, pol = _trunk_mask(net, h, pres, ids)
        elif self.packing == 'packed':
            z, pol = _trunk_packed(net, h, pres, ids)
        else:
            if self.packing == 'compact':
                n, _, full = _row_tokens(pres, net.n_tok)
            else:
                full, n = tok_full, tok_n
            hp = h.gather(1, full[..., None].expand(-1, -1, net.d))
            z, pol = net._trunk(hp, n, full, ids.gather(1, full) if ids is not None else None)
        r = net._heads(z, pol, n_players, aux, None, rraw)
        return r[0], r[1], r[2], (r[3] if aux else None)

    def forward(self, x_ptr, x_idx, x_val, n_players, belief, tok_full=None, tok_n=None):
        net = self.net
        x = densify(x_ptr, x_idx, x_val, net.features)
        plog, vlog, mar, raw = self._core(x, n_players, tok_full, tok_n)
        B, S = x.shape[0], net.seats
        seat = torch.arange(MAX_SEATS, device=x.device)[None, :]
        val = torch.softmax(vlog.float(), 1)
        mar = mar.float()
        if S < MAX_SEATS:
            val = F.pad(val, (0, MAX_SEATS - S))
            mar = F.pad(mar, (0, MAX_SEATS - S))
        value = torch.where(seat < n_players[:, None], val, torch.zeros_like(val))
        if self.layout:
            pred = aux_answer(net, raw)
            aux = torch.cat([pred[t[0]].float() for t in self.layout], 1)
            aux = torch.where(belief[:, None], aux, torch.zeros_like(aux))
        else:
            aux = torch.zeros(B, 0, dtype=torch.float32, device=x.device)
        return plog.float(), value, mar, aux



# ---------------------------------------------------------------- zero-onnx/2: variable-length rows (docs/az.md 46)
def onnx_version(net):
    """The zero-onnx version a network exports as: 2 for a token network with variable-length token sets (tail
    segments, pointer overrides, gates: `net.varlen_features`), 1 for every other network (the graph of before)."""
    return 2 if varlen_features(net) else 1


def _row_sum(flag, off):
    """Per row of a CSR (`off` [B+1]) the number of its elements with `flag` (exact integers: a running sum read at
    the offsets, no scatter)."""
    cs = torch.cat([torch.zeros(1, dtype=torch.int64, device=flag.device), flag.long().cumsum(0)])
    return cs.index_select(0, off[1:]) - cs.index_select(0, off[:-1])


def slot_pieces(slot):
    """Tail slots (u32: any integers, one outside [0, 2^32) given a high half of -1, which the graph refuses as it
    refused such a slot) as zero-onnx/2's two int32 inputs: (slot >> 16, slot & 0xffff)."""
    s = np.asarray(slot).astype(np.int64)
    hi = np.where((s < 0) | (s >= 1 << 32), -1, s >> 16)
    return hi.astype(np.int32), (s & 0xffff).astype(np.int32)


def override_pieces(tok):
    """The driver's packed override tokens (`rows.pack_tok`: seg << 40 | col << 32 | slot; -1 the generic logit) as
    zero-onnx/2's four int32 inputs (seg, col, slot >> 16, slot & 0xffff): -1 -> (-1, 0, 0, 0), any other negative ->
    (-2, 0, 0, 0) (refused, as the packed token was); seg is tok >> 40 whole (one past the network's segments is
    refused, not masked) - zero_worker's `override_pieces`."""
    t = np.asarray(tok, np.int64)
    pos = t >= 0
    tc = np.where(pos, t, 0)
    seg = np.where(pos, tc >> 40, np.where(t == -1, -1, -2))
    return (seg.astype(np.int32), ((tc >> 32) & 0xff).astype(np.int32), ((tc >> 16) & 0xffff).astype(np.int32),
            (tc & 0xffff).astype(np.int32))


def int32_bound(B, nnz, features, actions, tail_width, t_off, NT, L, K, NO):
    """An upper bound of every integer a zero-onnx/2 graph computes on a piece (docs/az.md 46.2; zero_worker's
    `int32_bound`): the densified rows' flat index B (F + 1), the packed token list's and the padded rows' indices
    B (F + 1 + the most tail tokens of a row), the policy's B (A + 1) and the overrides' NO + 1 after it (a refused
    override's spare place, docs/az.md 46.8), the tail floats' (NT + 1) Wt and L + 1, the options' K + 1, the merged CSR
    sequence's nnz + B. A feeder refuses a piece at 2^31 or more."""
    t_max = int(np.diff(np.asarray(t_off, np.int64)).max()) if len(t_off) > 1 else 0
    return max(int(nnz) + B, B * (int(features) + 1 + t_max), B * (int(actions) + 1) + int(NO) + 1,
               (int(NT) + 1) * int(tail_width), int(L) + 1, int(K) + 1)


def _iclamp(x, lo=None, hi=None):
    """An int64 tensor clamped to [lo, hi] by Where (ONNX Runtime 1.31.1's CPU Clip and Min on int64 answer wrongly
    around 2^32 - measured: Clip(4294967295, min 0) gave 0, Min(x, 4294967295) gave 4294967295 for every x -; Where
    is exact everywhere). `lo` / `hi` ints or the graph's symbolic sizes."""
    if lo is not None:
        x = torch.where(x < lo, torch.full_like(x, lo), x)
    if hi is not None:
        x = torch.where(x > hi, torch.full_like(x, hi), x)
    return x


def _prev(a):
    """`a` shifted by one (element i: a[i - 1]; element 0: a[0]) - no slice of a possibly empty axis."""
    i = _iclamp(torch.arange(a.shape[0], device=a.device) - 1, 0)
    return a.index_select(0, i)


def _psi2(hi, lo, pow2, sc):
    """`TokenNet._psi` of offsets n = hi 2^16 + lo in [0, 2^32), given as their 16-bit halves (int64 in [0, 2^16): no
    integer reaches 2^31): the 32 binary digits by integer division of each half (`//` exports as ONNX's integer Div;
    `torch.div` with a rounding mode would go through floats and `>>` by a tensor does not export; `pow2` 2^0 .. 2^15),
    log2(n + 1) / 32 computed in float64 (n exact there) and rounded, the sin / cos part read from psi's own table of
    n mod 256 = lo mod 256 (`sc`, `TokenNet._psi_sc`: the same floats)."""
    bits = torch.cat([torch.remainder(lo[:, None] // pow2[None, :], 2), torch.remainder(hi[:, None] // pow2[None, :], 2)],
                     1).to(torch.float32)
    n = hi.to(torch.float64) * 65536.0 + lo.to(torch.float64)
    lg = (torch.log2(n + 1.0) / 32.0).to(torch.float32)[:, None]
    return torch.cat([bits, lg, sc.index_select(0, torch.remainder(lo, 256))], 1)


def _lt2(ahi, alo, bhi, blo):
    """(ahi, alo) < (bhi, blo) for 16-bit halves: a < b on the whole numbers, no integer formed."""
    return (ahi < bhi) | ((ahi == bhi) & (alo < blo))


def _pointer2(net, hid):
    """`TokenNet._pointer` as inference runs it (no gradients): each part of `ptr_parts` a product of its own."""
    parts = getattr(net, 'ptr_parts', None)
    if not parts or len(parts) == 1:
        return net.ptr(hid)
    W, b = net.ptr.weight, net.ptr.bias
    return torch.cat([F.linear(hid, W[a:e], b[a:e]) for a, e in parts], -1)


def _lerp(a, b, w):
    """torch.lerp(a, b, w) (float32, w in [0, 1]) as ATen computes it: a + w (b - a) below 0.5, b - (b - a) (1 - w)
    from 0.5 - exactly a at w = 0 and exactly b at w = 1 (docs/az.md 43.6's blend)."""
    return torch.where(w < 0.5, a + w * (b - a), b - (b - a) * (1.0 - w))


def _outputs(net, layout, B, plog, vlog, mar, raw, n_players, belief):
    """The interface's policy, value, margin and aux from the network's heads (`ZeroOnnx.forward`'s last step)."""
    S = net.seats
    dev = plog.device
    seat = torch.arange(MAX_SEATS, device=dev)[None, :]
    val = torch.softmax(vlog.float(), 1)
    mar = mar.float()
    if S < MAX_SEATS:
        val = F.pad(val, (0, MAX_SEATS - S))
        mar = F.pad(mar, (0, MAX_SEATS - S))
    value = torch.where(seat < n_players[:, None], val, torch.zeros_like(val))
    if layout:
        pred = aux_answer(net, raw)
        aux = torch.cat([pred[t[0]].float() for t in layout], 1)
        aux = torch.where(belief[:, None], aux, torch.zeros_like(aux))
    else:
        aux = torch.zeros(B, 0, dtype=torch.float32, device=dev)
    return plog.float(), value, mar, aux


class ZeroOnnx2(ZeroOnnx):
    """A token network with variable-length token sets behind the zero-onnx/2 interface (module docstring, docs/az.md
    46): `forward(x_ptr, x_idx, x_val, n_players, belief, opt_off, opt_action, t_off, t_seg, t_slot_hi, t_slot_lo, t_val,
    ov_off, ov_opt, ov_seg, ov_col, ov_slot_hi, ov_slot_lo)` -> (policy, value, margin, aux, refused); no integer of
    the graph reaches 2^31 (docs/az.md 46.2). Packing 'packed' only (each row's tail tokens follow
    its present head tokens in the packed list)."""

    VERSION = 2

    def __init__(self, net, packing='packed'):
        super().__init__(net, packing)
        if not self.tokens:
            raise TypeError(f'zero-onnx/2: a token network\'s interface (got a {type(net).__name__})')
        if not varlen_features(net):
            raise ValueError('zero-onnx/2: a network without variable-length token sets exports as zero-onnx/1')
        if self.packing != 'packed':
            raise ValueError(f'zero-onnx/2: packing {self.packing!r}: only \'packed\' (the tails follow each row\'s '
                             f'present head tokens in the packed list)')
        segs = net.segs
        S, D = len(segs), net.d
        lt = lambda v: torch.tensor(v, dtype=torch.long)
        start, c0, nk, p, c = [], [], [], 1, 0
        for s in segs:
            k = len(s['actions']) + ('id_actions' in s) + int(s.get('columns', 0) or 0)
            start.append(p)
            c0.append(c)
            nk.append(k)
            p += s['count']
            c += k
        tk = [-1] * S
        gi = [-1] * S
        for k, j in enumerate(net.tails):
            tk[j] = k
        for k, j in enumerate(net.gated):
            gi[j] = k
        istail = [bool(s.get('tail')) for s in segs]
        # (no integer of the graph reaches 2^31, docs/az.md 46.2: the sizes its integers are built from, checked here;
        # the graph's constants by check_int32, a piece's sizes by int32_bound)
        if S >= 1 << 15 or p + c >= 1 << 30 or int(net.actions) >= 1 << 30:
            raise ValueError(f'zero-onnx/2: {S} segments, {p} head tokens, {c} pointer columns, {net.actions} actions: '
                             f'past the graph\'s int32 integers')
        for s in segs:
            if s['id'] is not None and int(np.prod(s['id']['radix'], dtype=np.float64)) >= 1 << 30:
                raise ValueError(f'zero-onnx/2: segment {s["name"]!r}: identity radix {s["id"]["radix"]} past int32')
        counts = [int(s['count']) for s in segs]
        self.register_buffer('v_width', lt([s['width'] if t else 0 for s, t in zip(segs, istail)]), persistent=False)
        self.register_buffer('v_istail', torch.tensor(istail), persistent=False)
        self.register_buffer('v_count_hi', lt([k >> 16 for k in counts]), persistent=False)
        self.register_buffer('v_count_lo', lt([k & 0xffff for k in counts]), persistent=False)
        self.register_buffer('v_start', lt(start), persistent=False)
        self.register_buffer('v_c0', lt(c0), persistent=False)
        self.register_buffer('v_nk', lt(nk), persistent=False)
        self.register_buffer('v_tk', lt(tk), persistent=False)
        self.register_buffer('v_gi', lt(gi), persistent=False)
        self.register_buffer('v_pow2', torch.tensor([1 << j for j in range(16)], dtype=torch.long), persistent=False)
        self.Wt = int(getattr(net, 't_width', 0) or 0)
        if net.tails:
            # the tail segments' input maps and position maps as one product each (`_TailEmbed`'s matrices: segment
            # k's rows k d .. (k + 1) d, its weights in its first `width` columns, zeros after)
            K = len(net.tails)
            W = torch.zeros(K * D, self.Wt)
            b = torch.zeros(K * D)
            P = torch.zeros(K * D, net.pos_tail[0].weight.shape[1])
            for k, j in enumerate(net.tails):
                w = segs[j]['width']
                W[k * D:(k + 1) * D, :w] = net.seg_in[j].weight.detach()
                b[k * D:(k + 1) * D] = net.seg_in[j].bias.detach()
                P[k * D:(k + 1) * D] = net.pos_tail[k].weight.detach()
            self.register_buffer('v_tW', W, persistent=False)
            self.register_buffer('v_tb', b, persistent=False)
            self.register_buffer('v_tP', P, persistent=False)
            self.register_buffer('v_sc', net._psi_sc(torch.device('cpu')).clone(), persistent=False)
        if net.gated:
            # inference's gate weights g = max(theta, 0) (docs/az.md 43.6)
            self.register_buffer('v_g', net.gate.detach().clamp(min=0).float().clone(), persistent=False)

    def _tail_embed(self, val, segc, shi, slo):
        """Tail tokens' embeddings [NT, d] (`TokenNet._tail_embed`: the segment's input map of its floats + its
        position tail_base[k] + pos_tail[k](psi(slot - count)) + its identity's embedding); the slots as 16-bit halves
        (`shi`, `slo` in [0, 2^16)), slot - count by halves with a borrow (clamped at 0: a slot below its head is
        refused)."""
        net = self.net
        NT, D = val.shape[0], net.d
        k = _iclamp(self.v_tk.index_select(0, segc), 0)
        dlo = slo - self.v_count_lo.index_select(0, segc)
        borrow = (dlo < 0).long()
        dlo = dlo + borrow * 65536
        dhi = shi - self.v_count_hi.index_select(0, segc) - borrow
        neg = dhi < 0
        dhi = torch.where(neg, 0, dhi)
        dlo = torch.where(neg, 0, dlo)
        pick = k[:, None, None].expand(NT, 1, D)
        x = F.linear(val, self.v_tW, self.v_tb).view(NT, -1, D).gather(1, pick).squeeze(1)
        pos = F.linear(_psi2(dhi, dlo, self.v_pow2, self.v_sc), self.v_tP).view(NT, -1, D).gather(1, pick).squeeze(1)
        t = x + (net.tail_base.index_select(0, k) + pos)
        for j in net.tails:
            s = net.segs[j]
            if s['id'] is None:
                continue
            i = _identity(val[:, :s['width']], s['id'], net.vocab)
            t = t + torch.where((segc == j)[:, None], net.emb(i), torch.zeros((), dtype=t.dtype))
        return t

    def _trunk2(self, h, pres, ids, opt_off, opt_action, t_off, t_seg, t_slot_hi, t_slot_lo, t_val, ov_off, ov_opt,
                ov_seg, ov_col, ov_slot_hi, ov_slot_lo):
        """`_trunk_packed` with each row's tail tokens after its present head tokens, the gated tokens weighted
        (docs/az.md 43.6) and the pointer overrides applied (43.7) -> (z [B, 2 d], logits [B, actions + 1], refused
        [B]: per row its tail tokens and overrides the network refuses - R5 / R6, which torch raises on). Slots and
        override tokens in int32 pieces (`slot_pieces`, `override_pieces`): no integer here reaches 2^31."""
        net = self.net
        B, NTOK, D = h.shape
        dev = h.device
        S = len(net.segs)
        A1 = net.actions + 1
        NT = t_seg.shape[0]
        NO = ov_opt.shape[0]
        L = t_val.shape[0]
        H16 = 65535
        one = lambda v: torch.full((1,), v, dtype=torch.long, device=dev)  # noqa: E731
        # ---- the tail tokens, checked (R5: a tail segment's, a slot past its head and below 2^32 - halves in
        # [0, 2^16) -, each row's sorted by (segment, slot) without a duplicate; the floats as many as the tokens' widths)
        seg = t_seg.long()
        segc = _iclamp(seg, 0, S - 1)
        shi, slo = t_slot_hi.long(), t_slot_lo.long()
        sok = (shi >= 0) & (shi <= H16) & (slo >= 0) & (slo <= H16)
        shi, slo = _iclamp(shi, 0, H16), _iclamp(slo, 0, H16)
        trow = _csr_rows(t_off, NT)
        tw = self.v_width.index_select(0, segc)
        k1 = segc * 65536 + shi                                   # (segment, slot) as (k1, slo): k1 < S 2^16 < 2^31
        first = torch.arange(NT, device=dev) == 0
        below = _lt2(shi, slo, self.v_count_hi.index_select(0, segc), self.v_count_lo.index_select(0, segc))
        tbad = ((segc != seg) | ~self.v_istail.index_select(0, segc) | below | ~sok
                | (~first & (_prev(trow) == trow) & ~_lt2(_prev(k1), _prev(slo), k1, slo)))
        # (the sum with a 0 appended: ONNX Runtime 1.31.1 reduces an empty axis to a wrong shape - measured)
        lbad = (torch.cat([tw, tw.new_zeros(1)]).sum() != L).long()
        # ---- their floats [NT, Wt] (each token's width(seg) floats, consecutive in t_val) and embeddings
        if net.tails:
            voff = tw.cumsum(0) - tw
            ar = torch.arange(self.Wt, device=dev)
            inw = ar[None, :] < tw[:, None]
            tv = torch.cat([t_val, t_val.new_zeros(1)])
            cidx = _iclamp(voff[:, None] + ar[None, :], 0, L)
            val = torch.where(inw, tv.index_select(0, cidx.reshape(-1)).view(NT, self.Wt), torch.zeros((), dtype=tv.dtype))
            # (embedded with one sentinel token after the batch's - the first tail segment's first slot, floats 0 -,
            # dropped after: no empty axis, whose reductions ONNX Runtime 1.31.1 answers with a wrong shape)
            j0 = net.tails[0]
            c0 = int(net.segs[j0]['count'])
            te = self._tail_embed(torch.cat([val, val.new_zeros(1, self.Wt)]), torch.cat([segc, one(j0)]),
                                  torch.cat([shi, one(c0 >> 16)]), torch.cat([slo, one(c0 & 0xffff)]))[:NT]
        else:
            te = h.new_zeros(NT, D)
        # ---- the packed list: the N present head tokens (row-major, layout order), then the NT tail tokens
        flat = pres.reshape(-1)
        sel = torch.nonzero(flat).squeeze(1)
        N = sel.shape[0]
        torch._check(N >= 2)
        row_h = sel // NTOK
        slot_h = sel - row_h * NTOK
        n_h = pres.sum(1)
        n_t = t_off[1:] - t_off[:-1]
        n = n_h + n_t
        T = _iclamp(n.max(), 2).item()                           # (a data-dependent shape: dynamic in the graph)
        torch._check(T >= 2)
        torch._check(T <= NTOK + NT)
        start_h = n_h.cumsum(0) - n_h
        ar = torch.arange(T, device=dev)[None, :]
        is_h = ar < n_h[:, None]
        valid = ar < n[:, None]
        gidx = torch.where(is_h, start_h[:, None] + ar,
                           torch.where(valid, N + t_off[:-1][:, None] + (ar - n_h[:, None]), torch.full_like(ar, N + NT)))
        back = torch.cat([row_h * T + (torch.arange(N, device=dev) - start_h.index_select(0, row_h)),
                          trow * T + n_h.index_select(0, trow) + (torch.arange(NT, device=dev) - t_off.index_select(0, trow))])
        rows = torch.cat([row_h, trow])
        hp = torch.cat([h.reshape(B * NTOK, D).index_select(0, sel), te])
        if net.gated:
            # (a token's weight: 1, a gated segment's g, 0 padding - the attention's float mask log w, the pooling)
            gi = self.v_gi.index_select(0, segc)
            wt = torch.where(gi >= 0, self.v_g.index_select(0, _iclamp(gi, 0)), torch.ones((), dtype=torch.float32))
            wpad = torch.cat([torch.ones(N, dtype=torch.float32, device=dev), wt,
                              torch.zeros(1, dtype=torch.float32, device=dev)]).index_select(0, gidx.reshape(-1)).view(B, T)
            bias = torch.log(wpad)[:, None, None, :].to(h.dtype)
        else:
            bias = torch.where(valid, 0.0, float('-inf'))[:, None, None, :].to(h.dtype)
        H = net.blocks[0].h
        hd = D // H
        for blk in net.blocks:
            qkv = blk.qkv(blk.ln1(hp))
            qkv = torch.cat([qkv, qkv.new_zeros(1, 3 * D)]).index_select(0, gidx.reshape(-1))
            q, k, v = qkv.view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
            s = torch.matmul(q, k.transpose(-2, -1)) * (hd ** -0.5) + bias
            a = torch.matmul(torch.softmax(s, -1), v).transpose(1, 2).reshape(B * T, D)
            hp = hp + blk.out(a.index_select(0, back))
            hp = hp + blk.ff2(F.gelu(blk.ff1(blk.ln2(hp))))
        hp = net.ln(hp)
        # the pooling through the padded rows (no scatter-add: ONNX Runtime's CPU ScatterND add races, docs/az.md 41)
        pad = torch.cat([hp, hp.new_zeros(1, D)]).index_select(0, gidx.reshape(-1)).view(B, T, D)
        if net.gated:
            zm = (pad * wpad[..., None]).sum(1) / wpad.sum(1, keepdim=True)
        else:
            zm = pad.sum(1) / n[:, None].to(hp.dtype)
        z = torch.cat([pad[:, 0], zm], 1)
        gen = F.pad(net.gen(z), (0, 1))
        # the static scatter of the head tokens' actions (a tail token has none: docs/az.md 43.7)
        cols = net.tok_cols.index_select(0, slot_h)
        acts = net.tok_acts.index_select(0, slot_h)
        if net.n_id:
            it = ids.reshape(-1).index_select(0, sel)
            ia = net.tok_idbase.index_select(0, slot_h) + it
            ia = torch.where((it < net.vocab) & (ia < net.actions), ia, net.actions)
            acts = torch.where(net.tok_isid.index_select(0, slot_h), ia[:, None], acts)
        acts = torch.where(cols >= 0, acts, net.actions)
        P = _pointer2(net, F.gelu(net.pk(hp) + net.pq(z).index_select(0, rows)))
        C = P.shape[1]
        lg = P[:N].gather(1, _iclamp(cols, 0))
        cposh = flat.long().cumsum(0) - 1                         # a present head position's packed index
        pol = _policy_full(gen, acts, lg, flat, cposh, NTOK).reshape(-1)
        if net.option_pointers:
            # ---- the overrides (docs/az.md 43.7): each one's row, its option's action, its target token (a present
            # head slot or a tail token of its row) and column; checked as rows.resolve and TokenNet._ov_index check
            # (one sentinel override after the batch's - row 0, option -1: refused, writing row 0's dummy column, and
            # left out of `refused` - so no axis is ever empty: ONNX Runtime's reductions of one answer a wrong shape)
            orow = torch.cat([_csr_rows(ov_off, NO), one(0)])
            opt = torch.cat([ov_opt.long(), one(-1)])
            nopt = opt_off[1:] - opt_off[:-1]
            obad = (opt < 0) | (opt >= nopt.index_select(0, orow))
            # (a refused override's action: the dummy column `actions`, at the front of the options' actions; a refused
            # option replaced by 0 before the sum, so no sum leaves the options' range)
            oa = torch.cat([one(net.actions), opt_action.long()])
            act = oa.index_select(0, torch.where(obad, 0, opt_off.index_select(0, orow) + torch.where(obad, 0, opt) + 1))
            # the target token's pieces (`override_pieces`): seg -1 the generic logit, below -1 refused; col and the
            # slot's halves clamped to their fields before any arithmetic
            s_ = torch.cat([ov_seg.long(), one(-1)])
            col = torch.cat([ov_col.long(), one(0)])
            ohi = torch.cat([ov_slot_hi.long(), one(0)])
            olo = torch.cat([ov_slot_lo.long(), one(0)])
            isgen = s_ == -1
            pok = (col >= 0) & (col <= 255) & (ohi >= 0) & (ohi <= H16) & (olo >= 0) & (olo <= H16)
            c_ = _iclamp(col, 0, 255)
            ohi, olo = _iclamp(ohi, 0, H16), _iclamp(olo, 0, H16)
            sc_ = _iclamp(s_, 0, S - 1)
            head = _lt2(ohi, olo, self.v_count_hi.index_select(0, sc_), self.v_count_lo.index_select(0, sc_))
            # (a head target's slot as one integer only where it is below its segment's head: < 2^30)
            l_ = torch.where(head, ohi, 0) * 65536 + torch.where(head, olo, 0)
            pos = _iclamp(self.v_start.index_select(0, sc_) + l_, 0, NTOK - 1)
            fpos = orow * NTOK + pos
            hpres = flat.index_select(0, fpos)
            hidx = cposh.index_select(0, fpos)
            # a tail target: the token of its row with that (segment, slot) - a match against every tail token (and
            # one that matches nothing, so an empty tail is no empty axis)
            tr_ = torch.cat([trow, one(-1)])
            ts_ = torch.cat([seg, one(-1)])
            th_ = torch.cat([shi, one(-1)])
            tl_ = torch.cat([slo, one(-1)])
            eq = ((tr_[None, :] == orow[:, None]) & (ts_[None, :] == s_[:, None]) & (th_[None, :] == ohi[:, None])
                  & (tl_[None, :] == olo[:, None]))
            found = eq.any(1)
            tidx = N + eq.long().argmax(1)
            obad = obad | (s_ < -1) | (~isgen & ((s_ >= S) | ~pok | (c_ >= self.v_nk.index_select(0, sc_))
                                                  | (head & ~hpres) | (~head & ~found)))
            p = _iclamp(torch.where(head, hidx, tidx), 0, N + NT - 1)
            gcol = _iclamp(self.v_c0.index_select(0, sc_) + c_, 0, C - 1)
            # (the target's row of the pointer logits, then its column: no flat index p C + col)
            v = P.index_select(0, p).gather(1, gcol[:, None]).squeeze(1)
            g0 = gen.reshape(-1).index_select(0, orow * A1 + act)
            if net.gated:
                gi_ = self.v_gi.index_select(0, sc_)
                isg = ~isgen & (gi_ >= 0)
                bw = self.v_g.index_select(0, _iclamp(gi_, 0)).clamp(max=1.0)
                v = torch.where(isg, _lerp(g0, v, bw), v)
            ovv = torch.where(isgen, g0, v)
            # one override per (row, action): of two with the same place (a refused one's: its row's dummy column) all
            # but the last are refused (which of two would win is not defined) - every later override compared with
            # every earlier one (no scatter: the last writer of a duplicate is what a GPU leaves undefined, docs/az.md
            # 46.8); then every write goes to its own place: an accepted one to its action, a refused one to its own
            # spare place past the policy (B A1 + its index; `int32_bound`'s B (A + 1) + NO + 1), read off by a Gather
            ar_o = torch.arange(NO + 1, device=dev)
            okey = torch.where(obad, orow * A1 + net.actions, orow * A1 + act)
            later = (okey[None, :] == okey[:, None]) & (ar_o[None, :] > ar_o[:, None])
            obad = obad | later.any(1)
            okey = torch.where(obad, B * A1 + ar_o, orow * A1 + act)
            pol = torch.cat([pol, pol.new_zeros(NO + 1)]).scatter(0, okey, ovv).index_select(
                0, torch.arange(B * A1, dtype=torch.int64, device=dev))
            oref = _row_sum(obad[:NO], ov_off)
        else:
            oref = ov_off[1:] - ov_off[:-1]                       # (a network without pointers refuses every one)
        refused = _row_sum(tbad, t_off) + oref + lbad
        return z, pol.view(B, A1), refused

    def forward(self, x_ptr, x_idx, x_val, n_players, belief, opt_off, opt_action, t_off, t_seg, t_slot_hi, t_slot_lo,
                t_val, ov_off, ov_opt, ov_seg, ov_col, ov_slot_hi, ov_slot_lo):
        net = self.net
        x = densify(x_ptr, x_idx, x_val, net.features)
        aux = bool(self.layout)
        rraw = ent = None
        if net.readers is not None:
            rraw, ent = _read(net, x)
        h, pres, ids = slot_tokens(net, x, ent)
        z, pol, refused = self._trunk2(h, pres, ids, opt_off, opt_action, t_off, t_seg, t_slot_hi, t_slot_lo, t_val,
                                       ov_off, ov_opt, ov_seg, ov_col, ov_slot_hi, ov_slot_lo)
        r = net._heads(z, pol, n_players, aux, None, rraw)
        policy, value, margin, aux_out = _outputs(net, self.layout, x.shape[0], r[0], r[1], r[2],
                                                  r[3] if aux else None, n_players, belief)
        return policy, value, margin, aux_out, refused

def example_inputs(net, packing):
    """Inputs to trace with: 3 rows, 7 non-zeros (distinct sizes, none 0 / 1: nothing specialised), every
    row a valid one (the token network's identities and flags aside: the values are not looked at)."""
    B, F_ = 3, net.features
    idx = torch.tensor([0, 1, 2, 0, 1, 0, 2], dtype=torch.int32) % F_
    args = [torch.tensor([0, 3, 5, 7], dtype=torch.int64), idx, torch.ones(7), torch.full((B,), 2, dtype=torch.int64),
            torch.tensor([True, False, True])]
    if packing == 'host':
        args += [torch.zeros(B, 5, dtype=torch.int64), torch.ones(B, dtype=torch.int64)]
    return tuple(args)


def dynamic_shapes(packing):
    """The inputs' dynamic dimensions: B (rows), nnz (non-zeros) and, packed by the host, T (tokens)."""
    B = torch.export.Dim('B', min=1, max=1 << 20)
    N = torch.export.Dim('nnz', min=1, max=1 << 30)
    d = {'x_ptr': {0: B + 1}, 'x_idx': {0: N}, 'x_val': {0: N}, 'n_players': {0: B}, 'belief': {0: B}}
    if packing == 'host':
        T = torch.export.Dim('T', min=1, max=4096)
        d.update(tok_full={0: B, 1: T}, tok_n={0: B})
    else:
        d.update(tok_full=None, tok_n=None)
    return d


def example_inputs2(net):
    """zero-onnx/2's inputs to trace with: `example_inputs`' rows, 5 options, 4 tail tokens (the first tail segment's,
    rows 0 and 2; or segment 0's for a network without tail segments: refused, still traced), 6 tail floats per
    token, 2 overrides (generic): every size distinct and none 0 / 1 (nothing specialised)."""
    args = list(example_inputs(net, 'packed'))
    A = int(net.actions)
    j = net.tails[0] if net.tails else 0
    w = net.segs[j]['width'] if net.tails else 6
    c = net.segs[j]['count']
    hi, lo = slot_pieces([c, c + 1, c, c + 2])
    i32 = lambda v: torch.as_tensor(np.asarray(v, np.int32))  # noqa: E731
    args += [torch.tensor([0, 2, 3, 5], dtype=torch.int64), torch.tensor([0, 1, 0, 1, 2], dtype=torch.int32) % A,
             torch.tensor([0, 2, 2, 4], dtype=torch.int64), torch.full((4,), j, dtype=torch.int32), i32(hi), i32(lo),
             torch.zeros(4 * w), torch.tensor([0, 1, 1, 2], dtype=torch.int64), torch.tensor([1, 0], dtype=torch.int32)]
    args += [i32(x) for x in override_pieces([-1, -1])]
    return tuple(args)


def dynamic_shapes2():
    """zero-onnx/2's dynamic dimensions: B, nnz, K (options), NT (tail tokens), L (tail floats), NO (overrides)."""
    B = torch.export.Dim('B', min=1, max=1 << 20)
    N = torch.export.Dim('nnz', min=1, max=1 << 30)
    K = torch.export.Dim('K', min=0, max=1 << 30)
    NT = torch.export.Dim('NT', min=0, max=1 << 24)
    L = torch.export.Dim('L', min=0, max=1 << 30)
    NO = torch.export.Dim('NO', min=0, max=1 << 24)
    return {'x_ptr': {0: B + 1}, 'x_idx': {0: N}, 'x_val': {0: N}, 'n_players': {0: B}, 'belief': {0: B},
            'opt_off': {0: B + 1}, 'opt_action': {0: K}, 't_off': {0: B + 1}, 't_seg': {0: NT},
            't_slot_hi': {0: NT}, 't_slot_lo': {0: NT}, 't_val': {0: L}, 'ov_off': {0: B + 1}, 'ov_opt': {0: NO},
            'ov_seg': {0: NO}, 'ov_col': {0: NO}, 'ov_slot_hi': {0: NO}, 'ov_slot_lo': {0: NO}}


def metadata2(net, extra=None):
    """zero-onnx/2's metadata: `metadata`'s, version 2, plus what a worker checks against its game (`tails`
    [[segment, name, width, head slots], ...] - the game's `game_info()['tails']` -, `option_pointers`) and what the
    network reads (`max_row_tokens`, `tail_width`, `gates` {segment: g}); `extra` last, as given."""
    meta = metadata(net, 'packed')
    meta.update({'zero_onnx': '2', 'protocol': PROTOCOL2, 'inputs': json.dumps(list(INPUTS2)),
                 'outputs': json.dumps(list(OUTPUTS2)),
                 'tails': json.dumps([[int(j), str(n), int(w), int(h)] for j, n, w, h in net.tail_segments()]),
                 'option_pointers': json.dumps(bool(net.option_pointers)),
                 'max_row_tokens': json.dumps(net.config['layout'].get('max_row_tokens')),
                 'tail_width': str(int(getattr(net, 't_width', 0) or 0)), 'gates': json.dumps(net.gates())})
    for k, v in (extra or {}).items():
        meta[str(k)] = v if isinstance(v, str) else json.dumps(v)
    return meta


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def metadata(net, packing, extra=None):
    """The model's metadata (all strings): the interface, the network's encoding and auxiliary layout,
    the export settings; `extra` (checkpoint, checkpoint_sha256, ...) added as given."""
    c = net.config
    layout = [list(t) for t in (getattr(net, 'aux_layout', None) or [])]
    meta = {'zero_onnx': '1', 'protocol': PROTOCOL, 'game': str(c.get('game')), 'arch': str(c.get('arch', 'mlp')),
            'features': str(int(c['features'])), 'actions': str(int(c['actions'])), 'seats': str(MAX_SEATS),
            'aux_layout': json.dumps(layout), 'aux_width': str(sum(int(t[1]) for t in layout)),
            'inputs': json.dumps(list(INPUTS) + (list(HOST_INPUTS) if packing == 'host' else [])),
            'outputs': json.dumps(list(OUTPUTS)), 'packing': str(packing or 'none'), 'opset': str(OPSET),
            'unique_scatters': '0' if packing in ('compact', 'host') else '1',
            'exporter': 'torch.onnx.export(dynamo=True)', 'torch': torch.__version__}
    if packing == 'host':
        meta['present_index'] = json.dumps(net.present_index())
    for k, v in (extra or {}).items():
        meta[str(k)] = v if isinstance(v, str) else json.dumps(v)
    return meta


def portable(model):
    """Rewrite an exported graph so that every GPU execution provider runs it, keeping every answer bit
    for bit (zero-farm PROTOCOL.md 8; docs/az.md 41.1) -> {rewrite: count}. In place.

    * A MatMul whose second operand is a 1-D constant (the identity digits' `digits @ arange`): DirectML
      answers it with zeros. -> MatMul(A, w[:, None]) then Squeeze(-1): the same product, one more axis
      (ONNX Runtime's CPU kernel promotes a 1-D operand to [K, 1] itself: the same GEMM, the same bits).
    * A Reshape with allowzero=1 (the dynamo exporter's `view`): DirectML fails on it at run time. ->
      allowzero=0 on the shape with its zeros turned into -1 (`Where(Equal(S, 0), -1, S)`; a constant shape
      without zeros is kept as it is). The same output for every shape with at most one zero - the only
      zero an exported shape holds is the empty token list of a piece without present tokens; a shape with
      two zeros would raise an error, never answer differently.
    """
    import numpy as np
    from onnx import helper, numpy_helper
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    consts = {n.output[0]: n for n in g.node if n.op_type == 'Constant' and n.attribute and n.attribute[0].name == 'value'}
    used = {x for n in g.node for x in list(n.input) + list(n.output)} | set(inits) | {i.name for i in g.input}

    def const(name):
        if name in inits:
            return numpy_helper.to_array(inits[name])
        if name in consts:
            return numpy_helper.to_array(consts[name].attribute[0].t)
        return None

    def fresh(base):
        k = 0
        while f'portable/{base}/{k}' in used:
            k += 1
        n = f'portable/{base}/{k}'
        used.add(n)
        return n

    def init(base, arr):
        n = fresh(base)
        g.initializer.append(numpy_helper.from_array(arr, n))
        return n

    stats = {'matmul_1d': 0, 'reshape_allowzero': 0, 'reshape_const': 0}
    out = []
    zero = minus1 = axes = None
    for n in g.node:
        if n.op_type == 'MatMul' and len(n.input) == 2:
            w = const(n.input[1])
            if w is not None and w.ndim == 1:
                axes = axes or init('axes_last', np.array([-1], np.int64))
                col = init('col', np.ascontiguousarray(w.reshape(-1, 1)))
                mid = fresh('mm2d')
                out.append(helper.make_node('MatMul', [n.input[0], col], [mid], name=n.name))
                out.append(helper.make_node('Squeeze', [mid, axes], [n.output[0]], name=fresh('squeeze')))
                stats['matmul_1d'] += 1
                continue
        if n.op_type == 'Reshape':
            az = next((a for a in n.attribute if a.name == 'allowzero'), None)
            if az is not None and az.i == 1:
                s = const(n.input[1])
                if s is not None and not (s == 0).any():
                    az.i = 0
                    stats['reshape_const'] += 1
                elif s is None:
                    zero = zero or init('zero', np.array(0, np.int64))
                    minus1 = minus1 or init('minus1', np.array(-1, np.int64))
                    eq, s2 = fresh('is0'), fresh('shape')
                    out.append(helper.make_node('Equal', [n.input[1], zero], [eq], name=fresh('equal')))
                    out.append(helper.make_node('Where', [eq, minus1, n.input[1]], [s2], name=fresh('where')))
                    az.i = 0
                    n.input[1] = s2
                    stats['reshape_allowzero'] += 1
        out.append(n)
    del g.node[:]
    g.node.extend(out)
    # initializers no node reads any more (the 1-D operands replaced)
    read = {x for n in g.node for x in n.input} | {o.name for o in g.output}
    keep = [i for i in g.initializer if i.name in read]
    if len(keep) != len(g.initializer):
        del g.initializer[:]
        g.initializer.extend(keep)
    return stats


def portable_file(src, dst):
    """`portable` on an exported file (written atomically; metadata `portable` 1; the exporter's annotations stripped:
    `strip_annotations`) -> the rewrites."""
    import onnx
    model = onnx.load(src)
    stats = portable(model)
    strip_annotations(model)
    if any(e.key == 'zero_onnx' and e.value == '2' for e in model.metadata_props):
        int32_slices(model)
        check_int32(model)
    for e in model.metadata_props:
        if e.key == 'portable':
            break
    else:
        e = model.metadata_props.add()
        e.key = 'portable'
    e.value = '1'
    onnx.checker.check_model(model)
    tmp = f'{dst}.{os.getpid()}.tmp'
    onnx.save_model(model, tmp, save_as_external_data=False)
    os.replace(tmp, dst)
    return stats


def export(net, path, packing='packed', extra=None, opset=OPSET, make_portable=True):
    """Export `net` (a TokenNet, GridNet or Net, as loaded: `net.load` / `load_checkpoint`) to the ONNX
    file `path` (one file, no external data; written atomically) -> its metadata: zero-onnx/1, or zero-onnx/2 for
    a token network with variable-length token sets (`onnx_version`; docs/az.md 46). `packing`: a token network's
    ('mask', 'compact', 'host'; module docstring; zero-onnx/2: 'packed' only), ignored for the others. `extra`:
    more metadata (the checkpoint's id and sha256: `export_checkpoint`). `make_portable`: `portable`'s
    rewrites (metadata `portable` 1), which change no answer and let the GPU providers run the graph. `opset`: 17 (the
    default: the exporter's opset-18 graph rewritten by `opset17`, docs/az.md 41.2) or 18 (as the exporter writes it).
    Every scatter of the mask and packed graphs writes unique indices (metadata `unique_scatters` 1; docs/az.md 41.2)."""
    net = copy.deepcopy(net).float().cpu().eval()
    for p in net.parameters():
        p.requires_grad_(False)
    if isinstance(net, TokenNet):
        net.chunk = 1 << 30                       # (forward's chunking is never traced; set for clarity)
    if onnx_version(net) == 2:
        # (variable-length token sets: zero-onnx/2, docs/az.md 46)
        mod = ZeroOnnx2(net, packing).eval()
        args, names, outs, dyn = example_inputs2(net), list(INPUTS2), list(OUTPUTS2), dynamic_shapes2()
        with torch.no_grad():
            prog = torch.onnx.export(mod, args, None, dynamo=True, opset_version=EXPORTER_OPSET, input_names=names,
                                     output_names=outs, dynamic_shapes=dyn, optimize=True, verbose=False)
        return _save(prog.model_proto, metadata2(net, extra), path, opset, make_portable)
    mod = ZeroOnnx(net, packing).eval()
    packing = mod.packing
    args = example_inputs(net, packing)
    names = list(INPUTS) + (list(HOST_INPUTS) if packing == 'host' else [])
    dyn = dynamic_shapes(packing)
    if packing != 'host':
        dyn = {k: v for k, v in dyn.items() if k in names}
    with torch.no_grad():
        prog = torch.onnx.export(mod, args, None, dynamo=True, opset_version=EXPORTER_OPSET, input_names=names,
                                 output_names=list(OUTPUTS), dynamic_shapes=dyn, optimize=True, verbose=False)
    return _save(prog.model_proto, metadata(net, packing, extra), path, opset, make_portable)


def strip_annotations(model):
    """Remove what the exporter writes for debugging only - each node's `metadata_props` (`pkg.torch.onnx.stack_trace`
    with the exporting checkout's file paths and line numbers, name scopes, class hierarchy, fx node, the rewriter's
    rule names) and every doc string (model, graph, nodes, inputs / outputs / value infos, initializers, functions,
    subgraphs) - so a checkpoint exports to the same bytes from any checkout and no graph carries a local path. No node,
    attribute, initializer, name or the model's own metadata changes. In place -> the number of entries removed."""
    k = 0

    def clean(x):
        nonlocal k
        if getattr(x, 'doc_string', ''):
            x.doc_string = ''
            k += 1
        if hasattr(x, 'metadata_props') and len(x.metadata_props):
            k += len(x.metadata_props)
            del x.metadata_props[:]

    def graph(g):
        clean(g)
        for n in g.node:
            node(n)
        for v in list(g.input) + list(g.output) + list(g.value_info):
            clean(v)
        for i in g.initializer:
            clean(i)

    def node(n):
        clean(n)
        for a in n.attribute:
            if a.HasField('g'):
                graph(a.g)
            for sg in a.graphs:
                graph(sg)

    model.doc_string = ''
    graph(model.graph)
    for f in model.functions:
        f.doc_string = ''
        for n in f.node:
            node(n)
        if hasattr(f, 'metadata_props'):
            k += len(f.metadata_props)
            del f.metadata_props[:]
    return k


def int32_slices(model):
    """zero-onnx/2: a Slice's start / end constant outside int32 (the exporter's 2^63 - 1 for "to the end") replaced by
    the int32 bound of its sign - the same slice for every axis below 2^31 elements (ONNX clamps them to the axis), and
    no zero-onnx/2 axis reaches 2^31 (`int32_bound`). In place -> the number of inputs replaced."""
    from onnx import numpy_helper
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    consts = {n.output[0]: n for n in g.node if n.op_type == 'Constant' and n.attribute and n.attribute[0].name == 'value'}
    used = {x for n in g.node for x in list(n.input) + list(n.output)} | set(inits) | {i.name for i in g.input}
    k = 0
    for n in g.node:
        if n.op_type != 'Slice':
            continue
        for j in (1, 2):
            if j >= len(n.input) or not n.input[j]:
                continue
            name = n.input[j]
            t = inits[name] if name in inits else (consts[name].attribute[0].t if name in consts else None)
            if t is None:
                continue
            a = numpy_helper.to_array(t)
            if a.dtype != np.int64 or not ((a > INT32_MAX) | (a < -INT32_MAX - 1)).any():
                continue
            b = np.clip(a, -INT32_MAX - 1, INT32_MAX).astype(np.int64)
            i = 0
            while f'int32/{name}/{i}' in used:
                i += 1
            new = f'int32/{name}/{i}'
            used.add(new)
            g.initializer.append(numpy_helper.from_array(b, new))
            n.input[j] = new
            k += 1
    read = {x for n in g.node for x in n.input} | {o.name for o in g.output}
    keep = [i for i in g.initializer if i.name in read]
    if len(keep) != len(g.initializer):
        del g.initializer[:]
        g.initializer.extend(keep)
    return k


def check_int32(model):
    """zero-onnx/2's refusal (docs/az.md 46.2): a graph with an integer constant outside int32 (an initializer, a
    Constant's value / value_int(s), a ConstantOfShape's value, any tensor attribute; in subgraphs and functions too) or
    any BitShift is not exported - DirectML computes int64 in 32 bits, so such a value would be read wrong there. ->
    the number of integer constants checked."""
    from onnx import AttributeProto, numpy_helper
    bad, k = [], 0

    def chk(where, a):
        nonlocal k
        a = np.asarray(a)
        if a.dtype.kind not in 'iu' or not a.size:
            return
        k += 1
        if int(a.min()) < -INT32_MAX - 1 or int(a.max()) > INT32_MAX:
            bad.append(f'{where} ({a.dtype}, {int(a.min())} .. {int(a.max())})')

    def nodes(ns, scope):
        for n in ns:
            if n.op_type == 'BitShift':
                bad.append(f'{scope}{n.name}: BitShift')
            for a in n.attribute:
                if a.type == AttributeProto.TENSOR:
                    chk(f'{scope}{n.name}.{a.name}', numpy_helper.to_array(a.t))
                elif a.type == AttributeProto.TENSORS:
                    for t in a.tensors:
                        chk(f'{scope}{n.name}.{a.name}', numpy_helper.to_array(t))
                elif n.op_type == 'Constant' and a.type == AttributeProto.INT:
                    chk(f'{scope}{n.name}.{a.name}', np.array([a.i]))
                elif n.op_type == 'Constant' and a.type == AttributeProto.INTS:
                    chk(f'{scope}{n.name}.{a.name}', np.array(a.ints, np.int64))
                if a.HasField('g'):
                    graph(a.g, f'{scope}{n.name}/')
                for sg in a.graphs:
                    graph(sg, f'{scope}{n.name}/')

    def graph(g, scope=''):
        for i in g.initializer:
            chk(f'{scope}initializer {i.name}', numpy_helper.to_array(i))
        nodes(g.node, scope)

    graph(model.graph)
    for f in model.functions:
        nodes(f.node, f'{f.name}/')
    if bad:
        raise ValueError(f'zero-onnx/2: {len(bad)} integer constant(s) outside int32 or shift(s) in the graph (DirectML '
                         f'computes int64 in 32 bits; docs/az.md 46.2): ' + '; '.join(bad[:8]))
    return k


def int64_max(model, feeds, threads=4):
    """The largest |value| of every int64 tensor a graph computes on the given inputs - the runtime half of
    zero-onnx/2's int32 rule beside `check_int32` (docs/az.md 46.7). `model`: a ModelProto (not changed: a copy has
    each int64 intermediate, typed by ONNX shape inference, made a graph output) or a path; `feeds`: input dicts
    (`onnx_parity.feeds`), run by ONNX Runtime's CPU EP -> {'tensors': int64 tensors read, 'untyped': outputs shape
    inference could not type (not read), 'max_abs': the largest |value|, 'top': [(name, max_abs), ...] the 8 largest}.
    Every value must stay below 2^31."""
    import onnx
    import onnxruntime as ort
    m = onnx.ModelProto()
    if isinstance(model, (str, os.PathLike)):
        m.CopyFrom(onnx.load(model))
    else:
        m.CopyFrom(model)
    mi = onnx.shape_inference.infer_shapes(m)
    types = {v.name: v.type.tensor_type.elem_type for v in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output)}
    have = {o.name for o in m.graph.output}
    inits = {i.name for i in m.graph.initializer}
    outs = [o for n in m.graph.node for o in n.output if o and o not in have and o not in inits]
    names = [o for o in outs if types.get(o) == onnx.TensorProto.INT64]
    for nm in names:
        m.graph.output.append(onnx.helper.make_tensor_value_info(nm, onnx.TensorProto.INT64, None))
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(m.SerializeToString(), so, providers=['CPUExecutionProvider'])
    onames = [o.name for o in sess.get_outputs()]
    best = {}
    for f in feeds:
        for nm, v in zip(onames, sess.run(None, f)):
            if v.dtype == np.int64 and v.size:
                best[nm] = max(best.get(nm, 0), int(np.abs(v).max()))
    top = sorted(best.items(), key=lambda kv: -kv[1])
    return {'tensors': len(best), 'untyped': sum(1 for o in outs if o not in types), 'max_abs': top[0][1] if top else 0,
            'top': top[:8]}


def scatter_writes(model, feeds, threads=4):
    """The runtime half of `unique_scatters` (docs/az.md 41.2): what every ScatterElements / ScatterND of a graph writes
    on the given inputs - a copy of the model (`model`: a ModelProto, not changed, or a path) with each scatter's data
    and indices also graph outputs, run by ONNX Runtime's CPU EP -> {'scatters': scatter nodes, 'runs': node runs read,
    'writes': indices written, 'duplicates': writes to a place another write of the same run also writes (a GPU leaves
    the winner undefined), 'over': runs with more indices along the axis than the data has places (DirectML refuses
    them), 'nodes': {node: [writes, duplicates, over]}}. Both must be 0 for a graph a GPU provider runs as the CPU does."""
    import onnx
    import onnxruntime as ort
    m = onnx.ModelProto()
    m.CopyFrom(onnx.load(model) if isinstance(model, (str, os.PathLike)) else model)
    have = {o.name for o in m.graph.output}
    sc = []
    for n in m.graph.node:
        if n.op_type in ('ScatterElements', 'ScatterND'):
            axis = next((a.i for a in n.attribute if a.name == 'axis'), 0)
            sc.append((n.name or n.output[0], n.op_type, axis, n.input[0], n.input[1]))
            for x in (n.input[0], n.input[1]):
                if x not in have:
                    m.graph.output.append(onnx.helper.make_empty_tensor_value_info(x))
                    have.add(x)
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(m.SerializeToString(), so, providers=['CPUExecutionProvider'])
    onames = [o.name for o in sess.get_outputs()]
    out = {'scatters': len(sc), 'runs': 0, 'writes': 0, 'duplicates': 0, 'over': 0, 'nodes': {}}
    for f in feeds:
        got = dict(zip(onames, sess.run(None, f)))
        for name, op, axis, d, i in sc:
            data, idx = np.asarray(got[d]), np.asarray(got[i]).astype(np.int64)
            if op == 'ScatterElements':
                ax = axis % data.ndim
                coords = list(np.indices(idx.shape, sparse=False))
                coords[ax] = np.where(idx < 0, idx + data.shape[ax], idx)
                tgt = np.ravel_multi_index([c.reshape(-1) for c in coords], data.shape) if idx.size else np.zeros(0, np.int64)
                over = idx.shape[ax] > data.shape[ax]
            else:
                k = idx.shape[-1]
                ix = idx.reshape(-1, k)
                ix = np.where(ix < 0, ix + np.asarray(data.shape[:k]), ix)
                tgt = np.ravel_multi_index(ix.T, data.shape[:k]) if ix.size else np.zeros(0, np.int64)
                over = False
            dup = int(tgt.size - np.unique(tgt).size)
            rec = out['nodes'].setdefault(name, [0, 0, 0])
            rec[0] += int(tgt.size)
            rec[1] += dup
            rec[2] += int(over)
            out['runs'] += 1
            out['writes'] += int(tgt.size)
            out['duplicates'] += dup
            out['over'] += int(over)
    return out


# the operators of opset 18 whose axes moved from an attribute to an input (opset 13's form below 18)
_REDUCE_AXES18 = ('ReduceL1', 'ReduceL2', 'ReduceLogSum', 'ReduceLogSumExp', 'ReduceMax', 'ReduceMean', 'ReduceMin',
                  'ReduceProd', 'ReduceSumSquare')


def opset17(model):
    """An exported opset-18 graph as opset 17, in place (docs/az.md 41.2: DirectML registers ScatterElements up to version
    16, so under opset 18 it leaves every scatter to the CPU; torch 2.11's exporter writes opset 18 whatever it is asked
    and onnx's version converter has no Pad 18 -> 17 adapter) -> {operator: nodes rewritten}. Every node whose operator
    has another version under opset 17 than under 18 (`onnx.defs`) is rewritten to its opset-17 form or refused:
    ReduceMax / Mean / Min / ... -18's constant `axes` input -> the `axes` attribute (`noop_with_empty_axes` dropped, a
    1 refused), ScatterElements / ScatterND -18 with reduction none / add / mul as they are (version 16), Pad -18 without
    its `axes` input as it is (version 13); anything else (Split's num_outputs, Resize-18, ...) raises. The same
    computation: each rewritten node computes what it did. Checked by onnx.checker's full check (strict shape
    inference) under opset 17."""
    import onnx
    from onnx import defs, helper, numpy_helper
    g = model.graph
    ver = {o.domain or 'ai.onnx': o.version for o in model.opset_import}
    if ver.get('ai.onnx') == 17:
        return {}
    if ver.get('ai.onnx') != 18:
        raise ValueError(f'opset17: an opset-{ver.get("ai.onnx")} graph (the exporter writes 18)')
    consts = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for n in g.node:
        if n.op_type == 'Constant' and n.attribute and n.attribute[0].name == 'value':
            consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    stats = {}

    def nodes(ns, where):
        for n in ns:
            for a in n.attribute:
                if a.HasField('g') or len(a.graphs):
                    raise ValueError(f'opset17: {where}{n.name} ({n.op_type}) has a subgraph')
            if n.domain not in ('', 'ai.onnx'):
                continue
            if defs.get_schema(n.op_type, 18, '').since_version == defs.get_schema(n.op_type, 17, '').since_version:
                continue
            if n.op_type in _REDUCE_AXES18:
                noop = next((a.i for a in n.attribute if a.name == 'noop_with_empty_axes'), 0)
                axes = None
                if len(n.input) > 1 and n.input[1]:
                    if n.input[1] not in consts:
                        raise ValueError(f'opset17: {where}{n.name} ({n.op_type}): axes not a constant')
                    axes = [int(x) for x in np.atleast_1d(consts[n.input[1]])]
                    del n.input[1:]
                if noop and not axes:
                    raise ValueError(f'opset17: {where}{n.name} ({n.op_type}): noop_with_empty_axes=1 without axes')
                keep = [a for a in n.attribute if a.name != 'noop_with_empty_axes']
                del n.attribute[:]
                n.attribute.extend(keep)
                if axes:
                    n.attribute.append(helper.make_attribute('axes', axes))
            elif n.op_type in ('ScatterElements', 'ScatterND'):
                red = next((a.s.decode() for a in n.attribute if a.name == 'reduction'), 'none')
                if red not in ('none', 'add', 'mul'):
                    raise ValueError(f'opset17: {where}{n.name} ({n.op_type}): reduction {red!r} (opset 18 only)')
            elif n.op_type == 'Pad':
                if len(n.input) > 3 and n.input[3]:
                    raise ValueError(f'opset17: {where}{n.name} (Pad): an axes input (opset 18 only)')
            else:
                raise ValueError(f'opset17: {where}{n.name} ({n.op_type}): changed in opset 18, no opset-17 form here')
            stats[n.op_type] = stats.get(n.op_type, 0) + 1

    nodes(g.node, '')
    for f in model.functions:
        nodes(f.node, f'{f.name}/')
        for o in f.opset_import:
            if o.domain in ('', 'ai.onnx'):
                o.version = 17
    for o in model.opset_import:
        if o.domain in ('', 'ai.onnx'):
            o.version = 17
    onnx.checker.check_model(model, full_check=True)
    return stats


def _save(model, meta, path, opset, make_portable, int32=True):
    """The exported model with its metadata (`portable`'s rewrites first; the exporter's annotations stripped;
    `int32_slices` then `check_int32`; `opset17` for opset 17), checked, written atomically -> the metadata."""
    import onnx
    meta['opset'] = str(opset)
    if make_portable:
        portable(model)
        meta['portable'] = '1'
    strip_annotations(model)
    if int32:
        int32_slices(model)
        check_int32(model)
    if opset == 17:
        opset17(model)
    elif opset != 18:
        raise ValueError(f'zero-onnx: opset {opset} (17, the default, or 18)')
    del model.metadata_props[:]
    for k, v in meta.items():
        e = model.metadata_props.add()
        e.key, e.value = k, v
    model.producer_name = 'zero-core zero.onnx_export'
    onnx.checker.check_model(model)
    tmp = f'{path}.{os.getpid()}.tmp'
    onnx.save_model(model, tmp, save_as_external_data=False)
    os.replace(tmp, path)
    return meta


def load_checkpoint(path, game=None, allow_mismatch=False):
    """A checkpoint as a run loads it for inference -> (network, its metadata extra): `net.load` (the
    network built from its config - buffers such as TokenNet's token tables rebuilt from it, not
    saved - and its state loaded; the encoding guard against the active game package unless
    `allow_mismatch`; an auxiliary head widened to the package's declared targets, docs/az.md 37),
    plus {checkpoint, checkpoint_sha256, checkpoint_iter}."""
    from .net import load
    m, ex = load(path, 'cpu', game=game, allow_mismatch=allow_mismatch)
    info = {'checkpoint': os.path.basename(path), 'checkpoint_sha256': file_sha256(path)}
    if ex.get('iter') is not None:
        info['checkpoint_iter'] = str(ex['iter'])
    return m, info


def export_checkpoint(path, out, packing='packed', game=None, allow_mismatch=False, checkpoint_id=None):
    """`load_checkpoint` + `export` -> the metadata written. `checkpoint_id`: the id the run publishes
    the checkpoint under (default its file name)."""
    m, info = load_checkpoint(path, game, allow_mismatch)
    if checkpoint_id:
        info['checkpoint'] = checkpoint_id
    return export(m, out, packing, info)


REFERENCE_TOL = 2e-5


def reference_rows(rows_list, n):
    """`n` real rows of recorded batch files (`onnx_parity.Rows`): whole recorded batches spread over the
    files (every k-th batch first, then the others; rows with tail tokens or pointer overrides: the batches holding
    overrides first, then those holding tail tokens - a stable order), the last one cut to make `n` -> (Rows, batch_off)."""
    from .onnx_parity import cat
    batches = []
    for ri, rows in enumerate(rows_list):
        bo = rows.batch_off if rows.batch_off is not None else np.arange(0, rows.B + 64, 64).clip(max=rows.B)
        bo = np.unique(np.asarray(bo, np.int64))
        batches += [(ri, int(a), int(b)) for a, b in zip(bo[:-1], bo[1:]) if b > a]
    if not batches:
        raise ValueError('no recorded batches')
    total = sum(b - a for _, a, b in batches)
    if total < n:
        raise ValueError(f'the batch files hold {total} rows, fewer than the {n} asked')
    per = total / len(batches)
    stride = max(1, int(len(batches) // max(1, int(np.ceil(n / per)))))
    order = batches[::stride] + [b for i, b in enumerate(batches) if i % stride]

    def lacks(ri, a, b):
        """(no override, no tail token) of a batch: the ones holding overrides first (rarer), then tail tokens."""
        r = rows_list[ri]
        return tuple(x is None or int(x.off[b] - x.off[a]) == 0 for x in (r.ov, r.tail))
    if any(r.tail is not None or r.ov is not None for r in rows_list):
        # (zero-onnx/2's self-test must answer tail tokens and overrides: the batches holding them first)
        order = sorted(order, key=lambda b: lacks(*b))
    parts, offs, got = [], [0], 0
    for ri, a, b in order:
        if got >= n:
            break
        b = min(b, a + (n - got))
        parts.append(rows_list[ri].take(np.arange(a, b)))
        got += b - a
        offs.append(got)
    return cat(parts, source='+'.join(sorted({r.source for r in rows_list}))), np.asarray(offs, np.int64)


def write_reference(net, rows_list, out, n=256, info=None, tol=REFERENCE_TOL):
    """The reference set of `net` (module docs) from recorded batch files -> its metadata. Every belief
    row gets the auxiliary answer; a network with an auxiliary head needs belief rows among the `n`."""
    import numpy as np
    from .infer import NetEvaluator
    from .onnx_parity import eager_answer
    from .protocol import ROW_BELIEF, ROW_VALUE
    rows, bo = reference_rows(rows_list, n)
    net = copy.deepcopy(net).float().cpu().eval()
    layout = [list(t) for t in (getattr(net, 'aux_layout', None) or [])]
    n_belief = int((rows.extra == ROW_BELIEF).sum())
    if layout and n_belief == 0:
        raise ValueError(f'the {n} reference rows hold no belief row: the auxiliary head would go untested (more rows, or batches with belief rows)')
    v2 = onnx_version(net) == 2
    if v2:
        # (zero-onnx/2: the self-test answers tail tokens and overrides too - never a reference set without them; the
        # files keep the driver's arrays - packed override tokens, u32 slots -, the feeders split them)
        st = rows.tail_stats()
        if net.tails and not st['tail_tokens']:
            raise ValueError(f'the {n} reference rows hold no tail token: the network\'s tail segments would go untested '
                             f'(record batches with tail tokens)')
        if net.option_pointers and not st['overrides']:
            raise ValueError(f'the {n} reference rows hold no pointer override: the network\'s overrides would go untested '
                             f'(record batches with overrides)')
    ev = NetEvaluator(net, 'cpu')
    pri, val, mar, aux = [], [], [], []
    for a, b in zip(bo[:-1], bo[1:]):
        ans = eager_answer(ev, rows.take(np.arange(a, b)))
        pri.append(np.asarray(ans[0], np.float32))
        val.append(np.asarray(ans[1], np.float32))
        mar.append(np.asarray(ans[2], np.float32))
        if len(ans) > 3 and ans[3] is not None:
            aux.append(np.asarray(ans[3], np.float32))
    W = sum(int(t[1]) for t in layout)
    aux = np.concatenate(aux) if aux else np.zeros((0, W), np.float32)
    meta = {'zero_onnx': '2' if v2 else '1', 'game': str(net.config.get('game')), 'features': int(net.config['features']),
            'actions': int(net.config['actions']), 'aux_width': W, 'rows': int(rows.B), 'belief_rows': n_belief,
            'value_rows': int((rows.extra == ROW_VALUE).sum()), 'batches': int(len(bo) - 1), 'tol': tol,
            'sources': rows.source, 'torch': torch.__version__, 'answers': 'infer.NetEvaluator, eager, CPU'}
    if v2:
        meta.update(rows.tail_stats())
        meta['t_segs'] = [[int(j), str(nm), int(w), int(h)] for j, nm, w, h in net.tail_segments()]
        meta['option_pointers'] = bool(net.option_pointers)
    for k, v in (info or {}).items():
        meta[str(k)] = v
    arrays = {'meta': np.array(json.dumps(meta)), 'xs_off': rows.xs_off.astype(np.int64), 'xs_idx': rows.xs_idx.astype(np.int32),
              'xs_val': rows.xs_val.astype(np.float32), 'n_players': rows.n_players.astype(np.int64),
              'extra': rows.extra.astype(np.uint8), 'opt_off': rows.opt_off.astype(np.int64),
              'opt_action': rows.opt_action.astype(np.int32), 'batch_off': bo, 'priors': np.concatenate(pri),
              'value': np.concatenate(val), 'margin': np.concatenate(mar), 'aux': aux.astype(np.float32)}
    if v2:
        # the rows' tail tokens (the driver's arrays: each token's floats consecutive) and overrides (docs/az.md 46)
        from .onnx_parity import model_tail_kw
        kw = model_tail_kw(net, rows)
        if 'tail' in kw:
            tl = kw['tail']
            arrays.update(t_off=tl.off.astype(np.int64), t_seg=tl.seg.astype(np.uint16), t_slot=tl.slot.astype(np.uint32),
                          t_val=tl.val.astype(np.float32))
        if 'ov' in kw:
            o = kw['ov']
            arrays.update(ov_off=o.off.astype(np.int64), ov_opt=o.opt.astype(np.int32), ov_tok=o.tok.astype(np.int64))
    tmp = f'{out}.{os.getpid()}.tmp'
    with open(tmp, 'wb') as f:
        np.savez(f, **arrays)
    os.replace(tmp, out)
    return meta


def load_reference(path):
    """reference.npz -> (onnx_parity.Rows, batch_off, (priors, value, margin, aux), meta)."""
    import numpy as np
    from .onnx_parity import Rows, recorded_rows
    with np.load(path) as z:
        meta = json.loads(str(z['meta']))
        tail, ov = recorded_rows({k: z[k] for k in z.files if k != 'meta'}, meta)
        rows = Rows(z['xs_off'], z['xs_idx'], z['xs_val'], z['n_players'], z['opt_off'], z['opt_action'], z['extra'],
                    meta['features'], batch_off=z['batch_off'], source=os.path.basename(path), tail=tail, ov=ov)
        return rows, z['batch_off'], (z['priors'], z['value'], z['margin'], z['aux']), meta


def check_reference(onnx_path, ref_path, threads=4):
    """The graph through onnxruntime (CPU EP, the worker's generic step, batch by batch) against the
    reference set's torch answers -> {max_abs, priors, value, margin, aux} (the worker's self-test,
    in Python)."""
    import numpy as np
    from .onnx_parity import answer, run_chunked, session
    rows, bo, want, meta = load_reference(ref_path)
    sess = session(onnx_path, threads)
    got = [[], [], [], []]
    for a, b in zip(bo[:-1], bo[1:]):
        sub = rows.take(np.arange(a, b))
        ans = answer(run_chunked(sess, sub), sub)
        for i, v in enumerate(ans):
            got[i].append(np.asarray(v, np.float32))
    out = {}
    for i, k in enumerate(('priors', 'value', 'margin', 'aux')):
        g = np.concatenate(got[i]) if got[i] else np.zeros_like(want[i])
        w = np.asarray(want[i], np.float32)
        if g.shape != w.shape or not np.isfinite(g).all() or not np.isfinite(w).all():
            out[k] = float('inf')
        else:
            out[k] = float(np.abs(g.astype(np.float64) - w.astype(np.float64)).max()) if w.size else 0.0
    out['max_abs'] = max(out.values())
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Export a checkpoint to a zero-onnx/1 ONNX graph (docs/az.md 41).')
    ap.add_argument('checkpoint')
    ap.add_argument('out')
    ap.add_argument('--packing', default='packed', choices=PACKINGS)
    ap.add_argument('--game', default=None)
    ap.add_argument('--id', default=None, help='the checkpoint id in the metadata (default: its file name)')
    ap.add_argument('--allow-mismatch', action='store_true',
                    help='skip the encoding guard (no game package for this checkpoint in this process)')
    ap.add_argument('--reference', type=int, default=0, metavar='N',
                    help='also write the reference set: N real batch rows with torch\'s answers (zero-farm\'s per-machine self-test)')
    ap.add_argument('--reference-batches', nargs='+', default=None, metavar='NPZ',
                    help='recorded driver batches (tools/onnx_record.py or a worker\'s --record-batches) the rows come from')
    ap.add_argument('--reference-out', default=None, help='where (default: reference.npz next to OUT)')
    ap.add_argument('--threads', type=int, default=4, help='onnxruntime threads of the reference check')
    ap.add_argument('--no-portable', action='store_true', help="skip portable()'s rewrites (they change no answer)")
    ap.add_argument('--rewrite', action='store_true',
                    help="CHECKPOINT is an exported .onnx: write it to OUT with portable()'s rewrites (no torch)")
    a = ap.parse_args(argv)
    if a.rewrite:
        print(json.dumps({'out': a.out, 'rewrites': portable_file(a.checkpoint, a.out)}))
        return 0
    if a.reference and not a.reference_batches:
        ap.error('--reference needs --reference-batches')
    m, info = load_checkpoint(a.checkpoint, a.game, a.allow_mismatch)
    if a.id:
        info['checkpoint'] = a.id
    meta = export(m, a.out, a.packing, info, make_portable=not a.no_portable)
    res = {'out': a.out, 'bytes': os.path.getsize(a.out), **meta}
    ok = True
    if a.reference:
        from .onnx_parity import Rows
        ref = a.reference_out or os.path.join(os.path.dirname(os.path.abspath(a.out)), 'reference.npz')
        rmeta = write_reference(m, [Rows.load(b) for b in a.reference_batches], ref, a.reference, info)
        chk = check_reference(a.out, ref, a.threads)
        ok = chk['max_abs'] <= REFERENCE_TOL
        res['reference'] = {'path': ref, 'bytes': os.path.getsize(ref), 'sha256': file_sha256(ref), 'rows': rmeta['rows'],
                            'belief_rows': rmeta['belief_rows'], 'value_rows': rmeta['value_rows'], 'batches': rmeta['batches'],
                            'onnxruntime_vs_torch': chk, 'passed': ok}
    print(json.dumps(res))
    if not ok:
        print(f'REFERENCE CHECK FAILED: the graph differs from torch by {res["reference"]["onnxruntime_vs_torch"]["max_abs"]:g} '
              f'(> {REFERENCE_TOL:g}) on the reference rows: do not publish this checkpoint', file=__import__('sys').stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
