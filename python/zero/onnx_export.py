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
opset, exporter, torch.

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
`onnxscript`), opset 18 (LayerNormalization, exact GELU through Erf, the attention decomposed into
MatMul / Softmax with the boolean mask). Only this module and `zero.onnx_parity` import onnx /
onnxruntime; nothing on a run's path does.
"""
import copy
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .net import GridNet, Net, TokenNet, _read, aux_answer
from .protocol import MAX_SEATS

PROTOCOL = 'zero-onnx/1'
OPSET = 18
INPUTS = ('x_ptr', 'x_idx', 'x_val', 'n_players', 'belief')
HOST_INPUTS = ('tok_full', 'tok_n')
OUTPUTS = ('policy', 'value', 'margin', 'aux')
PACKINGS = ('mask', 'compact', 'packed', 'host')


def densify(x_ptr, x_idx, x_val, features):
    """Sparse rows (CSR: `x_ptr` [B+1], `x_idx` [nnz], `x_val` [nnz]) -> f32 [B, features], with no
    data-dependent shape and no scatter with a reduction (TensorRT's ONNX parser refuses one and with it the
    whole graph; DirectML runs none): each non-zero's row is the number of row ends before it. In the merged
    sequence of the nnz non-zeros and the B row ends, row b's end sits at x_ptr[b + 1] + b (distinct even
    for empty rows): a scatter of ones there and a running sum count the ends up to each merged position; a
    non-zero at merged position j is non-zero j - count[j], its row count[j] (a scatter to unique slots, the
    ends' writes to one spare slot). Then one scatter of the values (protocol.densify). Integers throughout:
    the same rows as the earlier scatter-add (zero-farm PROTOCOL.md 8; docs/az.md 41.1)."""
    B = x_ptr.shape[0] - 1
    nnz = x_idx.shape[0]
    dev = x_ptr.device
    ends = x_ptr[1:] + torch.arange(B, dtype=torch.int64, device=dev)
    marks = torch.zeros(nnz + B, dtype=torch.int64, device=dev).scatter(0, ends, torch.ones_like(ends))
    count = marks.cumsum(0)
    slot = torch.where(marks > 0, torch.full_like(count, nnz), torch.arange(nnz + B, dtype=torch.int64, device=dev) - count)
    rows = torch.zeros(nnz + 1, dtype=torch.int64, device=dev).scatter(0, slot, count)[:nnz]
    flat = rows * features + x_idx.long()
    return torch.zeros(B * features, dtype=torch.float32, device=x_val.device).scatter(0, flat, x_val).view(B, features)


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
    return z, pc.scatter(1, acts.flatten(1), lg.flatten(1))


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
    the longest (gathered in, gathered back). N >= 2 is assumed while tracing (MTG Zero's rows hold the global
    and the context token: N >= 2 B); the graph keeps no such guard. -> (z [B, 2 d], logits [B, actions + 1])."""
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
    A1 = net.actions + 1
    pol = pc.reshape(-1).scatter(0, (row[:, None] * A1 + acts).reshape(-1), lg.reshape(-1)).view(B, A1)
    return z, pol


class ZeroOnnx(nn.Module):
    """A network behind the zero-onnx/1 interface (module docstring): `forward(x_ptr, x_idx, x_val,
    n_players, belief[, tok_full, tok_n])` -> (policy, value, margin, aux)."""

    def __init__(self, net, packing='packed'):
        super().__init__()
        self.net = net
        self.tokens = isinstance(net, TokenNet)
        if not isinstance(net, (TokenNet, GridNet, Net)):
            raise TypeError(f'zero-onnx: no export for a {type(net).__name__} (TokenNet, GridNet, Net)')
        if self.tokens and packing not in PACKINGS:
            raise ValueError(f'zero-onnx: packing {packing!r} (one of {PACKINGS})')
        self.packing = packing if self.tokens else None
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
    """`portable` on an exported file (written atomically; metadata `portable` 1) -> the rewrites."""
    import onnx
    model = onnx.load(src)
    stats = portable(model)
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
    file `path` (zero-onnx/1; one file, no external data; written atomically) -> its metadata. `packing`:
    a token network's ('mask', 'compact', 'host'; module docstring), ignored for the others. `extra`:
    more metadata (the checkpoint's id and sha256: `export_checkpoint`). `make_portable`: `portable`'s
    rewrites (metadata `portable` 1), which change no answer and let the GPU providers run the graph."""
    import onnx
    net = copy.deepcopy(net).float().cpu().eval()
    for p in net.parameters():
        p.requires_grad_(False)
    if isinstance(net, TokenNet):
        net.chunk = 1 << 30                       # (forward's chunking is never traced; set for clarity)
    mod = ZeroOnnx(net, packing).eval()
    packing = mod.packing
    args = example_inputs(net, packing)
    names = list(INPUTS) + (list(HOST_INPUTS) if packing == 'host' else [])
    dyn = dynamic_shapes(packing)
    if packing != 'host':
        dyn = {k: v for k, v in dyn.items() if k in names}
    with torch.no_grad():
        prog = torch.onnx.export(mod, args, None, dynamo=True, opset_version=opset, input_names=names,
                                 output_names=list(OUTPUTS), dynamic_shapes=dyn, optimize=True, verbose=False)
    model = prog.model_proto
    meta = metadata(net, packing, extra)
    meta['opset'] = str(opset)
    if make_portable:
        portable(model)
        meta['portable'] = '1'
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
    files (every k-th batch first, then the others), the last one cut to make `n` -> (Rows, batch_off)."""
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
    meta = {'zero_onnx': '1', 'game': str(net.config.get('game')), 'features': int(net.config['features']),
            'actions': int(net.config['actions']), 'aux_width': W, 'rows': int(rows.B), 'belief_rows': n_belief,
            'value_rows': int((rows.extra == ROW_VALUE).sum()), 'batches': int(len(bo) - 1), 'tol': tol,
            'sources': rows.source, 'torch': torch.__version__, 'answers': 'infer.NetEvaluator, eager, CPU'}
    for k, v in (info or {}).items():
        meta[str(k)] = v
    arrays = {'meta': np.array(json.dumps(meta)), 'xs_off': rows.xs_off.astype(np.int64), 'xs_idx': rows.xs_idx.astype(np.int32),
              'xs_val': rows.xs_val.astype(np.float32), 'n_players': rows.n_players.astype(np.int64),
              'extra': rows.extra.astype(np.uint8), 'opt_off': rows.opt_off.astype(np.int64),
              'opt_action': rows.opt_action.astype(np.int32), 'batch_off': bo, 'priors': np.concatenate(pri),
              'value': np.concatenate(val), 'margin': np.concatenate(mar), 'aux': aux.astype(np.float32)}
    tmp = f'{out}.{os.getpid()}.tmp'
    with open(tmp, 'wb') as f:
        np.savez(f, **arrays)
    os.replace(tmp, out)
    return meta


def load_reference(path):
    """reference.npz -> (onnx_parity.Rows, batch_off, (priors, value, margin, aux), meta)."""
    import numpy as np
    from .onnx_parity import Rows
    with np.load(path) as z:
        meta = json.loads(str(z['meta']))
        rows = Rows(z['xs_off'], z['xs_idx'], z['xs_val'], z['n_players'], z['opt_off'], z['opt_action'], z['extra'],
                    meta['features'], batch_off=z['batch_off'], source=os.path.basename(path))
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
