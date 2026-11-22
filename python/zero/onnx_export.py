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

The fast graph (docs/az.md 63; `export(..., fast=True)`, `ZeroOnnx2Fast`, the run setting `pool_export` 'fast'): a
zero-onnx/2 network's graph with TokenNet's own arithmetic - the same interface, metadata (plus `fast`, `attention`) and
answers (within the parity gate's 2e-5) at a fraction of the CPU time: the sparse rows read without densifying them, the
head slots embedded where present only, the bags and the global features' map from the non-zeros, one fused attention
node per block (ONNX Runtime's MultiHeadAttention by default; 'attention23' / 'matmul' the other forms), one Gelu node per MLP, the
layer norms as InstanceNormalization, the static policy one scatter of unique places, float32 only.

Every exported graph is written without the exporter's per-node annotations (`strip_annotations`: stack traces with
the exporting checkout's file paths and line numbers, doc strings): the same checkpoint exports to the same bytes from
any checkout, and no graph carries a local path.

Every graph a GPU provider runs as the CPU does (docs/az.md 41.2; metadata `unique_scatters` 1 - every packing but
'compact' and 'host', whose policy scatter is `TokenNet._trunk`'s): every scatter writes unique indices into data with
a place for each (`_csr_rows`' spare places for the row ends; the policy's last writer of each action made explicit,
`_scatter_last`; a refused pointer override to a spare place of its own), no integer leaves int32 (`check_int32` on
every export, zero-onnx/1 too), opset 17 (`opset17`: DirectML runs ScatterElements up to version 16). The CPU's answers
are the earlier graph's, bit for bit; `scatter_writes` checks a graph's scatters on real inputs. A token network with
`dup_reduce` 'max' (docs/az.md 58) gets the largest claim of each action instead of the last (`_scatter_max`: the writes
sorted by value, then the same unique scatter; metadata `dup_reduce` 'max'); 'mask' and 'packed' only.

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

from .net import GridNet, Net, TokenNet, _predict, _read, aux_answer, varlen_features
from .protocol import MAX_SEATS

# ---------------------------------------------------------------- fused attention (opt-in; docs/az.md 63)
# `attention` of `export`: how a graph computes the attention. None (the default graph: Q K^T and A V as MatMuls, an
# additive mask, Softmax), 'mha' (ONNX Runtime's com.microsoft MultiHeadAttention, one node per block; the graph's opset
# unchanged) or 'attention23' (ONNX's opset-23 Attention, one node per block; the whole graph at opset 23). The module
# computes in torch what the default path computes (the custom op's eager arithmetic is that path's); only the exported
# graph differs: the exporter writes a placeholder node (`FUSED_DOMAIN`.FusedAttention) that `fused_attention` rewrites
# into the chosen operator. The fast graph (`fast`, docs/az.md 63) takes 'matmul' too (its unfused form).
ATTENTIONS = (None, 'mha', 'attention23')
FAST_ATTENTIONS = ('mha', 'attention23', 'matmul')
STATIC_ATTENTIONS = ('mha', 'matmul')     # (the static graph's: the GPU's forms, docs/az.md 67)
FUSED_DOMAIN = 'zero.onnx'
# a graph variant by name (`graph` of export_template / swap_export / farm_h2h.export_cached; the run setting
# `pool_export`, docs/az.md 63) -> (attention, fast): '' the default graph; 'fast' the fast graph with com.microsoft
# MultiHeadAttention at opset 17 (every runtime pack the farm ships runs it on its provider: the CPU packs and the CUDA
# pack); 'fast-att23' with ONNX's opset-23 Attention (the fastest on the CPU; the CUDA pack's ONNX Runtime 1.24.4 has no
# CUDA kernel of opset 19-23's versions of a dozen operators, which then run on the CPU); 'fast-matmul' the MatMul form
GRAPHS = {'': (None, False), 'fast': ('mha', True), 'fast-att23': ('attention23', True), 'fast-matmul': ('matmul', True),
          'static': ('mha', 'static')}       # (docs/az.md 67: no data-dependent shape, for the GPU)


def graph_options(graph):
    """`GRAPHS[graph]`: (attention, fast); ValueError for another name."""
    if (graph or '') not in GRAPHS:
        raise ValueError(f'graph {graph!r}: one of {sorted(GRAPHS)}')
    return GRAPHS[graph or '']


def graph_opset(opset, attention):
    """The opset a graph is written at: `opset`, or 23 with ONNX's Attention ('attention23')."""
    return 23 if attention == 'attention23' else int(opset)


@torch.library.custom_op('zero_onnx::fused_attention', mutates_args=())
def _fused_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, heads: int) -> torch.Tensor:
    """softmax(Q K^T / sqrt(hd) + bias) V per head: q, k, v [B, T, D] (D = heads x hd), bias [B, 1, T, T] (additive: 0, a
    gated key's log weight, -inf for padding; the same for every query: ONNX Runtime's kernels of both fused operators
    want the query length in the mask, its CPU Attention-23 refuses the spec's broadcast) -> [B, T, D]; the default
    path's arithmetic."""
    B, T, D = q.shape
    hd = D // heads
    qh, kh, vh = (x.reshape(B, T, heads, hd).transpose(1, 2) for x in (q, k, v))
    s = torch.matmul(qh, kh.transpose(-2, -1)) * (hd ** -0.5) + bias
    return torch.matmul(torch.softmax(s, -1), vh).transpose(1, 2).reshape(B, T, D)


@_fused_attention.register_fake
def _fused_attention_fake(q, k, v, bias, heads):
    return torch.empty_like(q)


def _placeholder_schema():
    """The placeholder's ONNX schema, registered once (onnxscript builds nodes of registered operators only)."""
    from onnx import defs
    if defs.has('FusedAttention', FUSED_DOMAIN):
        return
    T = defs.OpSchema.FormalParameter
    defs.register_schema(defs.OpSchema(
        'FusedAttention', FUSED_DOMAIN, 1, doc='zero-onnx placeholder: rewritten by onnx_export.fused_attention',
        inputs=[T('q', 'T'), T('k', 'T'), T('v', 'T'), T('bias', 'T')], outputs=[T('y', 'T')],
        type_constraints=[('T', ['tensor(float)'], 'float32')],
        attributes=[defs.OpSchema.Attribute('num_heads', defs.OpSchema.AttrType.INT, 'heads')]))


def _fused_attention_onnx(q, k, v, bias, heads: int):
    """The exporter's translation: the placeholder node (`fused_attention` rewrites it)."""
    import onnxscript
    _placeholder_schema()
    zo = onnxscript.values.Opset(FUSED_DOMAIN, 1)
    return zo.FusedAttention(q, k, v, bias, num_heads=heads)


def fused_attention(model, kind):
    """The placeholder nodes (`FUSED_DOMAIN`.FusedAttention: q, k, v [B, T, D], bias [B, 1, T, T] -> [B, T, D]) rewritten
    into `kind`'s operator, in place -> the nodes rewritten. 'mha': com.microsoft MultiHeadAttention (query, key, value,
    no bias, no key_padding_mask, attention_bias = bias; scale 1/sqrt(hd)) - the opset of the rest unchanged.
    'attention23': Attention (q_num_heads = kv_num_heads = heads, attn_mask = bias; scale 1/sqrt(hd)) with the default
    domain at opset 23 (the operators the exporter writes compute the same there: `opset_bump`)."""
    from onnx import helper
    if kind not in ATTENTIONS[1:]:
        raise ValueError(f'fused attention {kind!r}: one of {ATTENTIONS[1:]}')
    g = model.graph
    out, n = [], 0
    for node in g.node:
        if node.domain != FUSED_DOMAIN:
            out.append(node)
            continue
        if node.op_type != 'FusedAttention':
            raise ValueError(f'{FUSED_DOMAIN}.{node.op_type}: no rewrite')
        heads = next(a.i for a in node.attribute if a.name == 'num_heads')
        q, k, v, bias = node.input
        if kind == 'mha':
            out.append(helper.make_node('MultiHeadAttention', [q, k, v, '', '', bias], list(node.output), name=node.name,
                                        domain='com.microsoft', num_heads=heads))
        else:
            out.append(helper.make_node('Attention', [q, k, v, bias], list(node.output), name=node.name,
                                        q_num_heads=heads, kv_num_heads=heads))
        n += 1
    del g.node[:]
    g.node.extend(out)
    imp = [o for o in model.opset_import if o.domain != FUSED_DOMAIN]
    del model.opset_import[:]
    model.opset_import.extend(imp)
    if kind == 'mha' and not any(o.domain == 'com.microsoft' for o in imp):
        model.opset_import.append(helper.make_opsetid('com.microsoft', 1))
    if kind == 'attention23':
        opset_bump(model, 23)
    return n


def fuse_gelu(model):
    """The exporter's exact GELU - x * (0.5 * (1 + Erf(x / sqrt 2))): Div, Erf, Add, Mul, Mul, each but the last read
    once - rewritten as one ONNX Runtime com.microsoft Gelu node (the same function, one kernel; its CPU and CUDA
    providers have had it for years, where ONNX's own Gelu needs opset 20), in place -> the nodes rewritten. ONNX
    Runtime's CPU optimizer fuses its own pattern (0.5 x (1 + ...)), not this order: the fast graph's five passes over the
    MLP's hidden units become one."""
    from onnx import helper, numpy_helper
    g = model.graph
    consts = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for n in g.node:
        if n.op_type == 'Constant' and n.attribute and n.attribute[0].name == 'value':
            consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    uses = {}
    for n in g.node:
        for x in n.input:
            uses[x] = uses.get(x, 0) + 1
    for o in g.output:
        uses[o.name] = uses.get(o.name, 0) + 1
    prod = {o: n for n in g.node for o in n.output}

    def const(x, v):
        a = consts.get(x)
        return a is not None and a.size == 1 and abs(float(a.reshape(-1)[0]) - v) < 1e-6 * max(1.0, abs(v))

    def other(n, x):
        return n.input[1] if n.input[0] == x else n.input[0]
    drop, repl = set(), {}
    for m2 in g.node:
        if m2.op_type != 'Mul' or len(m2.input) != 2:
            continue
        for a in m2.input:
            m1 = prod.get(other(m2, a))
            if m1 is None or m1.op_type != 'Mul' or uses.get(m1.output[0]) != 1:
                continue
            ad = next((prod.get(x) for x in m1.input if prod.get(x) is not None and prod[x].op_type == 'Add'), None)
            if ad is None or not const(other(m1, ad.output[0]), 0.5) or uses.get(ad.output[0]) != 1:
                continue
            er = next((prod.get(x) for x in ad.input if prod.get(x) is not None and prod[x].op_type == 'Erf'), None)
            if er is None or not const(other(ad, er.output[0]), 1.0) or uses.get(er.output[0]) != 1:
                continue
            dv = prod.get(er.input[0])
            if (dv is None or dv.op_type != 'Div' or dv.input[0] != a or not const(dv.input[1], 2.0 ** 0.5)
                    or uses.get(dv.output[0]) != 1):
                continue
            drop.update(id(x) for x in (dv, er, ad, m1))
            repl[id(m2)] = helper.make_node('Gelu', [a], list(m2.output), name=m2.name, domain='com.microsoft')
            break
    if not repl:
        return 0
    out = [repl.get(id(n), n) for n in g.node if id(n) not in drop]
    del g.node[:]
    g.node.extend(out)
    if not any(o.domain == 'com.microsoft' for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid('com.microsoft', 1))
    return len(repl)


def layernorm_instance(model):
    """Every LayerNormalization over the last axis (scale and bias given, its mean / inverse deviation outputs unused)
    rewritten as InstanceNormalization of x viewed as [rows, 1, features] - one channel, scale 1, bias 0: the same
    normalization (the mean and the biased variance over the features, epsilon inside the square root) - then x scale
    + bias, in place -> the nodes rewritten. ONNX Runtime 1.31.1's CPU LayerNormalization on Apple silicon takes ~5x
    InstanceNormalization's time (a [350, 128] input: 134 against 26 us on the Mac's M4 Max; on the desktop's i9 35
    against 45 us - measured 2026-11-18, docs/az.md 63)."""
    from onnx import helper, numpy_helper
    g = model.graph
    used = {x for n in g.node for x in n.input} | {o.name for o in g.output}
    names = {x for n in g.node for x in list(n.input) + list(n.output)} | {i.name for i in g.initializer}
    k = 0

    def fresh(base):
        nonlocal k
        while f'ln_inst/{base}/{k}' in names:
            k += 1
        n = f'ln_inst/{base}/{k}'
        names.add(n)
        return n
    out, n_rw, consts = [], 0, None
    for n in g.node:
        ax = next((a.i for a in n.attribute if a.name == 'axis'), -1)
        if (n.op_type != 'LayerNormalization' or n.domain not in ('', 'ai.onnx') or ax != -1 or len(n.input) < 3
                or not n.input[2] or any(o in used for o in n.output[1:])):
            out.append(n)
            continue
        if consts is None:
            consts = (fresh('one'), fresh('zero'), fresh('axis1'))
            g.initializer.extend([numpy_helper.from_array(np.ones(1, np.float32), consts[0]),
                                  numpy_helper.from_array(np.zeros(1, np.float32), consts[1]),
                                  numpy_helper.from_array(np.ones(1, np.int64), consts[2])])
        eps = next((a.f for a in n.attribute if a.name == 'epsilon'), 1e-5)
        x3, y3, y2, ys = fresh('x3'), fresh('y3'), fresh('y2'), fresh('ys')
        out += [helper.make_node('Unsqueeze', [n.input[0], consts[2]], [x3], name=fresh('unsqueeze')),
                helper.make_node('InstanceNormalization', [x3, consts[0], consts[1]], [y3], name=n.name, epsilon=eps),
                helper.make_node('Squeeze', [y3, consts[2]], [y2], name=fresh('squeeze')),
                helper.make_node('Mul', [y2, n.input[1]], [ys], name=fresh('scale')),
                helper.make_node('Add', [ys, n.input[2]], [n.output[0]], name=fresh('bias'))]
        n_rw += 1
    del g.node[:]
    g.node.extend(out)
    return n_rw


def opset_bump(model, to):
    """The default domain's opset raised to `to`, in place, for a graph whose operators compute the same there: every
    node's operator must have no other version between its opset and `to` but versions that only add types (refused
    otherwise, naming them)."""
    from onnx import defs
    cur = next(o.version for o in model.opset_import if o.domain in ('', 'ai.onnx'))
    # (versions 19-23 that change only the types an operator takes - float8, int4, bfloat16, ... - or add what our
    # graphs never set; checked against onnx.defs below: an operator not listed with a newer version is refused)
    same = {'Cast', 'CastLike', 'Constant', 'ConstantOfShape', 'Equal', 'Flatten', 'Identity', 'If', 'Loop', 'Pad',
            'QuantizeLinear', 'DequantizeLinear', 'Reshape', 'Scan', 'Shape', 'Size', 'Squeeze', 'Transpose', 'Unsqueeze',
            'Where', 'Gather', 'Expand', 'Concat', 'Slice', 'ReduceMax', 'ReduceMin', 'ReduceSum', 'ReduceMean',
            'Attention', 'MatMul', 'Add', 'Sub', 'Mul', 'Div', 'Range', 'Less', 'Greater', 'LessOrEqual', 'GreaterOrEqual',
            'And', 'Or', 'Not', 'Neg', 'Softmax', 'Gemm', 'LayerNormalization', 'Erf', 'Sqrt', 'Pow', 'Log', 'Exp', 'Max',
            'Min', 'Clip', 'NonZero', 'ScatterND', 'ScatterElements', 'GatherElements', 'GatherND', 'CumSum', 'Split',
            'Tile', 'TopK', 'ArgMax', 'Abs', 'Sigmoid', 'Tanh', 'Relu', 'Floor', 'Ceil', 'Mod', 'Sign', 'IsNaN', 'IsInf'}
    bad = set()
    for node in model.graph.node:
        if node.domain not in ('', 'ai.onnx'):
            continue
        b = defs.get_schema(node.op_type, to, '').since_version
        if b > cur:
            continue                      # (an operator new since `cur` - the Attention just written - is `to`'s own)
        a = defs.get_schema(node.op_type, cur, '').since_version
        if a != b and node.op_type not in same:
            bad.add(f'{node.op_type} ({a} -> {b})')
    if bad:
        raise ValueError(f'opset {cur} -> {to}: operators with another version there: {sorted(bad)}')
    for o in model.opset_import:
        if o.domain in ('', 'ai.onnx'):
            o.version = to

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


CUMSUM_BLOCK = 512                 # `_cumsum0`'s blocks (the fast graph's long running sums, docs/az.md 66)
# the fast graph's optional input `t_align` (docs/az.md 66): its padded rows' token axis a multiple of it (at most
# T_ALIGN_PAD more than the longest row: `int32_bound`); default 1 (an initializer: the graph as before when not fed).
# T_ALIGN is what a GPU worker feeds (metadata `t_align`): ONNX Runtime's CUDA MultiHeadAttention runs float32 with an
# attention bias through its fused kernel only when the sequence is a multiple of 16 (16-byte aligned bias rows).
T_ALIGN = 16
T_ALIGN_PAD = 15


def _optional_input(model, name, value):
    """Graph input `name` given an initializer of the same name (`value`, int64 0-d): an input a feeder may leave out
    (ONNX: an overridable initializer; ONNX Runtime feeds it when given, never folds it). In place."""
    from onnx import numpy_helper
    if not any(i.name == name for i in model.graph.input):
        raise ValueError(f'no graph input {name!r} to give a default')
    model.graph.initializer.append(numpy_helper.from_array(np.asarray(value, np.int64), name))


def _cumsum0(v, blocked=False):
    """`v.cumsum(0)` of a 1-d integer tensor. `blocked` (the fast graph's, docs/az.md 66): in two levels - running sums
    within blocks of CUMSUM_BLOCK elements (`v` zero-padded to whole blocks), then over the blocks' totals, added back -,
    the same integers: ONNX Runtime 1.24.4's CUDA CumSum costs more than linearly in the axis's length (measured on a
    farm PC's NVIDIA GPU: 0.37 ms over 16k elements, 1.65 ms over 64k, 16.4 ms over 262k - a 1,024-row batch's
    non-zeros) and 0.18 ms over [464, 512] along its rows; the CPU's runs both forms in linear time. Only for an axis
    sized by the graph's inputs (non-zeros, tail tokens, overrides, rows): over a data-dependent length (a NonZero's
    count) the exporter dropped the padding (a [-1, 512] Reshape of 33,830 elements failed on CUDA - measured)."""
    if not blocked:
        return v.cumsum(0)
    n = v.shape[0]
    k = CUMSUM_BLOCK
    nb = (n + (k - 1)) // k
    inner = torch.cat([v, v.new_zeros(nb * k - n)]).view(nb, k).cumsum(1)
    tot = inner[:, k - 1]
    return (inner + (tot.cumsum(0) - tot)[:, None]).reshape(-1)[:n]


def _csr_rows(off, n, blocked=False):
    """Each of the `n` elements' row of a CSR over rows (`off` [B+1]), with no data-dependent shape, no scatter with a
    reduction (TensorRT's ONNX parser refuses one and with it the whole graph; DirectML runs none) and every scatter
    writing unique indices into data with a place for each (docs/az.md 41.2: a GPU provider refuses more indices than
    places and leaves the winner of a duplicate undefined). In the merged sequence of the n elements and the B row ends,
    row b's end sits at off[b+1] + b (distinct even for empty rows): a scatter of ones there and a running sum count the
    ends up to each merged position; an element at merged position j is element j - count[j], its row count[j] - a
    scatter of the rows to the elements' places [0, n) and the ends' to their own spare places n + (their index), then
    the first n read by a Gather (not a Slice: DirectML's graph fusion fails on Scatter -> Slice -> Mul). Integers below
    n + B: the same rows as the earlier one-spare-slot scatter and the scatter-add before it (docs/az.md 41.1).
    `blocked`: the running sum in two levels (`_cumsum0`; the fast graph's), the same rows."""
    B = off.shape[0] - 1
    dev = off.device
    ends = off[1:] + torch.arange(B, dtype=torch.int64, device=dev)
    marks = torch.zeros(n + B, dtype=torch.int64, device=dev).scatter(0, ends, torch.ones_like(ends))
    count = _cumsum0(marks, blocked)
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


def _scatter_max(base, idx, val):
    """base [B, S].scatter_reduce(1, idx [B, L], val [B, L], 'amax', include_self=False) - TokenNet's policy scatter
    under `dup_reduce` 'max' (docs/az.md 58) - with every write to its own place: the writes of each row put in
    ascending order of their values (a sort: TopK), then `_scatter_last` - the last write of each index in that order is
    its largest. The answer is one of the writes (no arithmetic), so the same as torch's on every device, bit for bit;
    equal values tie to equal bits. Every scatter unique, no accumulate scatter (docs/az.md 41.2)."""
    p = torch.sort(val, dim=1).indices
    return _scatter_last(base, idx.gather(1, p), val.gather(1, p))


def _scatter_policy(net, base, idx, val):
    """The policy's static scatter as `net` (a TokenNet) computes it (`TokenNet._scatter_policy`, docs/az.md 58): its
    `dup_reduce` None the CPU's last writer (`_scatter_last`, the graphs of before), 'max' the largest claim
    (`_scatter_max`)."""
    if getattr(net, 'dup_reduce', None) == 'max':
        return _scatter_max(base, idx, val)
    return _scatter_last(base, idx, val)


def _policy_full(gen, acts, lg, flat, cpos, n_tok, net=None):
    """The static scatter of a packed token list's pointer logits into the policy, per row (`_scatter_last`): gen [B, A1]
    (the generic logits, the dummy column last), acts / lg [N, K] each packed head token's actions (the dummy column
    `A1 - 1` for none) and logits, flat [B n_tok] the full token axis's presence, cpos [B n_tok] a present position's
    packed index. Each row's writes laid out on its full token axis ([B, n_tok K]: absent slots write the dummy column) -
    a row's packed tokens are in layout order, so a write's place on the axis orders the row's writes as the packed
    list's flat scatter did: the same last writer of every action, bit for bit. `net`'s `dup_reduce` 'max': the largest
    claim of every action instead (`_scatter_policy`; the order then does not matter)."""
    B, A1 = gen.shape
    N, K = acts.shape
    fidx = torch.where(flat, cpos, N)
    a = torch.cat([acts, acts.new_full((1, K), A1 - 1)]).index_select(0, fidx).view(B, n_tok * K)
    v = torch.cat([lg, lg.new_zeros(1, K)]).index_select(0, fidx).view(B, n_tok * K)
    return _scatter_policy(net, gen, a, v)


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


def _digit_tables(radix):
    """An identity layout's (digit values [R, n_digits], digit indicators [R, n_digits], mixed-radix multipliers
    [n_digits]) - `TokenNet._digit_tables`."""
    R, nd = sum(radix), len(radix)
    vals, ind = torch.zeros(R, nd), torch.zeros(R, nd)
    mult, o = [], 0
    for k, r in enumerate(radix):
        vals[o:o + r, k] = torch.arange(r, dtype=torch.float32)
        ind[o:o + r, k] = 1
        o += r
        mult.append(int(np.prod(radix[k + 1:], dtype=np.int64)))
    return vals, ind, torch.tensor(mult, dtype=torch.long)


def _identity_digits(blk, vals, ind, mult, vocab):
    """Identities [n] of digit blocks blk [n, R] by two products (`TokenNet._identity_digits`: the same identities as
    `_identity` on one-hot or empty digit blocks, each product summing one exact term per digit), `vocab` = none."""
    dig = torch.matmul(blk, vals).round().long()
    ok = (torch.matmul(blk, ind) > 0.5).all(1)
    idx = (dig * mult).sum(1)
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
    return z, _scatter_policy(net, pc, acts.flatten(1), lg.flatten(1))


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
    return z, _policy_full(pc, acts, lg, flat, cpos, NT, net)


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
        elif self.packing and getattr(net, 'dup_reduce', None) is not None:
            # (their policy scatter is TokenNet._trunk's own: scatter_reduce, an opset-18 reduction opset17 refuses)
            raise ValueError(f'zero-onnx: packing {self.packing!r} has no export of dup_reduce '
                             f'{net.dup_reduce!r} (docs/az.md 58): mask or packed')
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


def _psi2(hi, lo, pow2, sc, f32=False):
    """`TokenNet._psi` of offsets n = hi 2^16 + lo in [0, 2^32), given as their 16-bit halves (int64 in [0, 2^16): no
    integer reaches 2^31): the 32 binary digits by integer division of each half (`//` exports as ONNX's integer Div;
    `torch.div` with a rounding mode would go through floats and `>>` by a tensor does not export; `pow2` 2^0 .. 2^15),
    log2(n + 1) / 32 computed in float64 (n exact there) and rounded, the sin / cos part read from psi's own table of
    n mod 256 = lo mod 256 (`sc`, `TokenNet._psi_sc`: the same floats). `f32` (the fast graph, docs/az.md 63): log2(n + 1)
    / 32 in float32 (n + 1 exact below 2^24; within a few units in the last place of TokenNet's - no float64 in the
    graph)."""
    bits = torch.cat([torch.remainder(lo[:, None] // pow2[None, :], 2), torch.remainder(hi[:, None] // pow2[None, :], 2)],
                     1).to(torch.float32)
    if f32:
        lg = (torch.log2(hi.to(torch.float32) * 65536.0 + lo.to(torch.float32) + 1.0) / 32.0)[:, None]
    else:
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

    # the override targets' match (0 / 1) is ArgMax'ed in this dtype: int64 here, as every graph before; the static
    # graph's in float32 (the same first index of the largest), which ONNX Runtime's CUDA provider runs - its ArgMax has no
    # int64 kernel, so an int64 ArgMax is the static graph's only CPU node (docs/az.md 72)
    ARGMAX_DTYPE = torch.int64
    # a block of zeros appended to the policy before a scatter: as a Concat with zeros (every graph before), or as a Pad
    # of the policy (the fast graph: the same tensor; ONNX Runtime's CUDA host time of those two Concats was ~20 % of a
    # call's, docs/az.md 72)
    PAD_ZEROS = False

    VERSION = 2

    def __init__(self, net, packing='packed', attention=None):
        super().__init__(net, packing)
        if attention not in ATTENTIONS:
            raise ValueError(f'attention {attention!r}: one of {ATTENTIONS}')
        self.attention = attention
        self.psi_f32 = False                     # (the fast graph's: float32 only, `_psi2`)
        self.blocked = False                     # (the fast graph's: long running sums in two levels, `_cumsum0`)
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
        x = self._tail_maps(val, k, pick)
        pos = F.linear(_psi2(dhi, dlo, self.v_pow2, self.v_sc, self.psi_f32), self.v_tP).view(NT, -1, D).gather(
            1, pick).squeeze(1)
        t = x + (net.tail_base.index_select(0, k) + pos)
        return self._tail_ids(t, val, segc)

    def _tail_maps(self, val, k, pick):
        """Each tail token's segment's input map of its floats val [NT, Wt] (k: its tail index, pick: k as a gather index
        [NT, 1, d]): every tail segment's map on every token as one product, each token's own picked."""
        return F.linear(val, self.v_tW, self.v_tb).view(val.shape[0], -1, self.net.d).gather(1, pick).squeeze(1)

    def _tail_ids(self, t, val, segc):
        """The tail tokens' identity embeddings added to their embeddings t (a token of a segment with an identity)."""
        net = self.net
        for j in net.tails:
            s = net.segs[j]
            if s['id'] is None:
                continue
            i = _identity(val[:, :s['width']], s['id'], net.vocab)
            t = t + torch.where((segc == j)[:, None], net.emb(i), torch.zeros((), dtype=t.dtype))
        return t

    def _tails(self, ref, t_off, t_seg, t_slot_hi, t_slot_lo, t_val, embed=None):
        """The tail tokens, checked, and their embeddings -> (seg, segc (clamped), slot halves shi / slo (clamped), each
        token's row, tbad (R5 per token), lbad (the floats do not fit the widths), embeddings [NT, d]). `ref`: a tensor of
        the trunk's dtype and device. `embed`: the embedding of (floats [n, Wt], clamped segments, shi, slo), default
        `_tail_embed`, run on the tokens and one sentinel token after them (no empty axis)."""
        net = self.net
        D = net.d
        dev = ref.device
        S = len(net.segs)
        NT = t_seg.shape[0]
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
        trow = _csr_rows(t_off, NT, self.blocked)
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
            te = (embed or self._tail_embed)(torch.cat([val, val.new_zeros(1, self.Wt)]), torch.cat([segc, one(j0)]),
                                             torch.cat([shi, one(c0 >> 16)]), torch.cat([slo, one(c0 & 0xffff)]))[:NT]
        else:
            te = ref.new_zeros(NT, D)
        return seg, segc, shi, slo, trow, tbad, lbad, te

    def _overrides(self, pol, gen, P, flat, cposh, N, NTOK, trow, seg, shi, slo, opt_off, opt_action, ov_off, ov_opt, ov_seg,
                   ov_col, ov_slot_hi, ov_slot_lo, slice_back=False):
        """The pointer overrides applied to the static policy (`pol` [B A1], flat) -> (pol, per row the overrides
        refused). `gen` [B, A1] the generic logits, `P` [N + NT, C] every packed token's pointer logits (head tokens
        first, then the tail tokens), `flat` [B NTOK] the head positions' presence, `cposh` [B NTOK] a present head
        position's packed index, `trow` / `seg` / `shi` / `slo` the tail tokens' rows, segments and slot halves.
        `slice_back` (the fast graph): the policy read back from the scatter's buffer by a Slice, not a Gather."""
        net = self.net
        B = gen.shape[0]
        dev = gen.device
        S = len(net.segs)
        A1 = net.actions + 1
        NT = trow.shape[0]
        NO = ov_opt.shape[0]
        C = P.shape[1]
        H16 = 65535
        one = lambda v: torch.full((1,), v, dtype=torch.long, device=dev)  # noqa: E731
        if net.option_pointers:
            # ---- the overrides (docs/az.md 43.7): each one's row, its option's action, its target token (a present
            # head slot or a tail token of its row) and column; checked as rows.resolve and TokenNet._ov_index check
            # (one sentinel override after the batch's - row 0, option -1: refused, writing row 0's dummy column, and
            # left out of `refused` - so no axis is ever empty: ONNX Runtime's reductions of one answer a wrong shape)
            orow = torch.cat([_csr_rows(ov_off, NO, self.blocked), one(0)])
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
            tidx = N + eq.to(self.ARGMAX_DTYPE).argmax(1)
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
            pol = (F.pad(pol, (0, NO + 1)) if self.PAD_ZEROS else torch.cat([pol, pol.new_zeros(NO + 1)])).scatter(0, okey, ovv)
            if slice_back:
                pol = pol[:B * A1]
            else:
                pol = pol.index_select(0, torch.arange(B * A1, dtype=torch.int64, device=dev))
            oref = _row_sum(obad[:NO], ov_off)
        else:
            oref = ov_off[1:] - ov_off[:-1]                       # (a network without pointers refuses every one)
        return pol, oref

    def _trunk2(self, h, pres, ids, opt_off, opt_action, t_off, t_seg, t_slot_hi, t_slot_lo, t_val, ov_off, ov_opt,
                ov_seg, ov_col, ov_slot_hi, ov_slot_lo):
        """`_trunk_packed` with each row's tail tokens after its present head tokens, the gated tokens weighted
        (docs/az.md 43.6) and the pointer overrides applied (43.7) -> (z [B, 2 d], logits [B, actions + 1], refused
        [B]: per row its tail tokens and overrides the network refuses - R5 / R6, which torch raises on). Slots and
        override tokens in int32 pieces (`slot_pieces`, `override_pieces`): no integer here reaches 2^31."""
        net = self.net
        B, NTOK, D = h.shape
        dev = h.device
        A1 = net.actions + 1
        NT = t_seg.shape[0]
        seg, segc, shi, slo, trow, tbad, lbad, te = self._tails(h, t_off, t_seg, t_slot_hi, t_slot_lo, t_val)
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
        if self.attention:
            bias4 = bias.expand(B, 1, T, T)                       # (once for every block: the fused operators' mask)
        for blk in net.blocks:
            qkv = blk.qkv(blk.ln1(hp))
            qkv = torch.cat([qkv, qkv.new_zeros(1, 3 * D)]).index_select(0, gidx.reshape(-1))
            if self.attention:
                # (opt-in, docs/az.md 63: one fused node in the graph; the same arithmetic in torch)
                q3, k3, v3 = qkv.view(B, T, 3, D).unbind(2)
                a = torch.ops.zero_onnx.fused_attention(q3, k3, v3, bias4, H).reshape(B * T, D)
            else:
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
        lg = P[:N].gather(1, _iclamp(cols, 0))
        cposh = flat.long().cumsum(0) - 1                         # a present head position's packed index
        pol = _policy_full(gen, acts, lg, flat, cposh, NTOK, net).reshape(-1)
        pol, oref = self._overrides(pol, gen, P, flat, cposh, N, NTOK, trow, seg, shi, slo, opt_off, opt_action, ov_off,
                                    ov_opt, ov_seg, ov_col, ov_slot_hi, ov_slot_lo)
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


class ZeroOnnx2Fast(ZeroOnnx2):
    """zero-onnx/2 with TokenNet's own arithmetic (`export(..., fast=True)`, docs/az.md 63): the interface, metadata (plus
    `fast`, `attention`) and answers of `ZeroOnnx2` (within float32 rounding: the parity gate's 2e-5), a graph that does
    the work the eager `TokenNet.forward` does instead of more:

    * the sparse rows read without densifying them (`_inputs`: the default graph zero-fills and scatters [B, features],
      MTG Zero's 56,384 floats a row): each non-zero's kind and place from tables of the features - the presence flags,
      the readers' inputs and the present head slots' floats each one scatter to unique places;
    * the head slots embedded where present only (`_slots`, `TokenNet._pack`): one NonZero over every head slot, each
      present slot's floats [n, widest width]; each segment's input map on its own present slots (a NonZero of its own,
      the maps put back in place by one scatter of unique rows); the identities decoded once per identity layout
      (`TokenNet._identity_digits`' two products) - where the default graph maps, decodes and zeroes all of a row's
      slots (MTG Zero's 170);
    * the global features' map and the bags from the non-zeros (`_gmap`, `_bag`: each row's non-zeros in a range of
      features are a run of its ascending ones): the dense products' non-zero terms only (MTG Zero's bags: 23 x 1024
      identities a row, ~77 non-zero);
    * the packed list (the global tokens, the present head slots row-major, the tail tokens): each token's place in its
      padded row from the presence's running sum, the padded rows' tokens by one scatter of unique places; padding keys
      read a real token (masked: no zero row per block);
    * the attention one fused node per block (`attention`: 'mha' - com.microsoft MultiHeadAttention, opset 17 -,
      'attention23' - ONNX's Attention, opset 23 -, or 'matmul', the default graph's MatMul / Softmax), its mask built
      once; one Gelu node per MLP (`fuse_gelu`), the layer norms as InstanceNormalization (`layernorm_instance`);
    * the tail segments' input maps in up to three groups of similar width (`_groups`: each group one product over its
      widest member's floats - a token's floats are zeros past its own width: the same terms -, the token's own block
      picked; the default graph's one product over all of them, 21 x 128 outputs a token for MTG Zero);
    * the static policy one scatter of unique places when every static write names its own action (the network's
      tables, checked at construction; else `_policy_full`, the default graph's);
    * float32 only (`_psi2`'s f32).

    The tail tokens' checks, the gates, the overrides, `refused` and the heads are `ZeroOnnx2`'s. Every scatter unique,
    no accumulate scatter, int32 integers (docs/az.md 41.2, 46.7). The CUDA pack's ONNX Runtime (1.24.4) shaped two
    choices (docs/az.md 63): no data-dependent count is used as a size or an offset beyond the batch's own (its
    allocation planner reused live buffers in graphs that did), and no running sum runs over the non-zeros but
    `_csr_rows`' (its CumSum takes ~45 ns an element)."""

    PAD_ZEROS = True


    def __init__(self, net, packing='packed', attention='mha'):
        if attention not in FAST_ATTENTIONS:
            raise ValueError(f'fast graph: attention {attention!r} (one of {FAST_ATTENTIONS})')
        super().__init__(net, packing, None if attention == 'matmul' else attention)
        self.psi_f32 = True
        self.blocked = True
        segs, n_tok = net.segs, int(net.n_tok)
        F_ = int(net.features)
        # the sparse rows read without densifying them (`_inputs`): each feature's kind and place - a head slot's
        # position in the full token axis (>= 1) and its column there, a global feature's place in `gidx` (position -1),
        # a bag's (-2: `_bag`); per position its present column (-1: always present)
        G = int(net.gidx.numel())
        if not G:
            raise ValueError('fast graph: a token network without global features')
        # gidx as ranges of consecutive features [(first, end, its place in gidx)] (`_gmap`)
        gl = net.gidx.tolist()
        self.g_ranges = []
        for k, f in enumerate(gl):
            if k and f == gl[k - 1] + 1:
                self.g_ranges[-1][1] = f + 1
            else:
                self.g_ranges.append([f, f + 1, k])
        f_tok = torch.full((F_,), -2, dtype=torch.int32)
        f_col = torch.zeros(F_, dtype=torch.int32)
        f_tok[net.gidx] = -1
        f_col[net.gidx] = torch.arange(G, dtype=torch.int32)
        t_pcol, t_w = [-1] * n_tok, [0] * n_tok
        always = [False] * n_tok
        always[0] = True                                          # (the global token)
        self.heads = []
        st = 1
        for j, s in enumerate(segs):
            k, w, o = int(s['count']), int(s['width']), int(s['offset'] or 0)
            if not k:
                continue                                          # (a tail-only segment: no head slot)
            for i in range(k):
                t_w[st + i] = w
                f_tok[o + i * w:o + (i + 1) * w] = st + i
                f_col[o + i * w:o + (i + 1) * w] = torch.arange(w, dtype=torch.int32)
                if s['present'] is None:
                    always[st + i] = True
                else:
                    t_pcol[st + i] = int(s['present'])
            self.heads.append({'seg': j, 'start': st, 'count': k, 'width': w, 'offset': o, 'id': s['id']})
            st += k
        lt = lambda v: torch.tensor(v, dtype=torch.long)  # noqa: E731
        self.register_buffer('f_tok', f_tok, persistent=False)
        self.register_buffer('f_col', f_col, persistent=False)
        self.register_buffer('f_always', torch.tensor(always), persistent=False)
        self.register_buffer('t_pcol', lt(t_pcol), persistent=False)
        self.register_buffer('t_w', lt(t_w), persistent=False)
        # a reader's features (docs/az.md 36): each feature's place among its inputs (-1: not one)
        for n, r in enumerate(net.readers or []):
            fr = torch.full((F_,), -1, dtype=torch.int32)
            fr[r.idx] = torch.arange(int(r.idx.numel()), dtype=torch.int32)
            self.register_buffer(f'f_rd{n}', fr, persistent=False)
        # identity layouts (radix): one decode per layout - the head segments' present slots grouped by theirs, the
        # tail tokens per (at, radix) with the segments that use it (TokenNet._tail_embed_static's)
        self.radixes = []

        def layout(radix):
            radix = tuple(int(r) for r in radix)
            if radix not in self.radixes:
                vals, ind, mult = _digit_tables(radix)
                k = len(self.radixes)
                self.register_buffer(f'f_dval{k}', vals, persistent=False)
                self.register_buffer(f'f_dind{k}', ind, persistent=False)
                self.register_buffer(f'f_dmul{k}', mult, persistent=False)
                self.radixes.append(radix)
            return self.radixes.index(radix)
        t_lay, t_at = [-1] * n_tok, [0] * n_tok
        for hs in self.heads:
            if hs['id'] is not None:
                for i in range(hs['count']):
                    t_lay[hs['start'] + i] = layout(hs['id']['radix'])
                    t_at[hs['start'] + i] = int(hs['id']['at'])
        self.h_lays = sorted(set(x for x in t_lay if x >= 0))
        self.register_buffer('t_lay', lt(t_lay), persistent=False)
        self.register_buffer('t_at', lt(t_at), persistent=False)
        tid = {}
        for j in net.tails:
            if segs[j]['id'] is not None:
                tid.setdefault((int(segs[j]['id']['at']), layout(segs[j]['id']['radix'])), []).append(j)
        self.t_ids = []
        for n, ((at, k), js) in enumerate(tid.items()):
            uses = torch.zeros(len(segs), dtype=torch.bool)
            uses[js] = True
            self.register_buffer(f't_uses{n}', uses, persistent=False)
            self.t_ids.append((at, k))
        # the tail segments' input maps in width groups (`_tail_maps`)
        self.t_groups = []
        if net.tails:
            tb = self._groups([int(segs[j]['width']) for j in net.tails], [net.seg_in[j] for j in net.tails], 't')
            self.register_buffer('t_block', lt(tb), persistent=False)
        # the bounds of every run (`_runs`: the bag groups', the global features' ranges), counted below at once
        bounds = sorted({int(x) for lo, hi, _ in self.g_ranges for x in (lo, hi)}
                        | {int(x) for b in net.bags for x in (b['offset'], b['offset'] + b['count'] * b['width'])})
        self.r_bound = {x: k for k, x in enumerate(bounds)}
        self.register_buffer('r_bounds', torch.tensor(bounds, dtype=torch.int32), persistent=False)
        # each bag group's identity and bag of a feature (its place in the group: tables, no integer division)
        for n, b in enumerate(net.bags):
            f = torch.arange(int(b['count']) * int(b['width']))
            self.register_buffer(f'b_bag{n}', f // int(b['width']), persistent=False)
            self.register_buffer(f'b_col{n}', f % int(b['width']), persistent=False)
        # the static policy by one scatter: every static write (a head position's pointer column with an action) names an
        # action no other write names and no action comes from an identity - each row's writes then go to places of
        # their own (`forward`; its places below B (actions + 1 + n_tok K) <= B (features + 1): `int32_bound`); else
        # `_policy_full` (the default graph's sorted last writer, or the largest claim)
        cols, acts = net.tok_cols[:n_tok], net.tok_acts[:n_tok]
        src = acts[(cols >= 0) & (acts < int(net.actions))]
        self.punique = (not net.n_id and int(torch.unique(src).numel()) == int(src.numel())
                        and int(net.actions) + n_tok * int(cols.shape[1]) <= F_)

    def _groups(self, widths, maps, tag):
        """Input maps of segments of these `widths` (their nn.Linear `maps`) in up to three groups of similar width - the
        split with the least work (sum of group members x group width) -, each group's maps one product over its widest
        member's floats (a token's floats are zeros past its own width: the same terms): buffers `<tag>_W<g>` [members d,
        width] / `<tag>_b<g>`, the group widths appended to `<tag>_groups` -> each segment's block (its place among the
        groups' blocks, group after group: `_grouped`)."""
        K, D = len(widths), self.net.d
        order = sorted(range(K), key=lambda k: widths[k])
        best = None
        for a in range(1, K + 1):
            for b in range(a, K + 1):
                parts = [p for p in (order[:a], order[a:b], order[b:]) if p]
                cost = sum(len(p) * max(widths[k] for k in p) for p in parts)
                if best is None or cost < best[0]:
                    best = (cost, parts)
        block, m = [0] * K, 0
        for gi, part in enumerate(best[1]):
            Wg = max(widths[k] for k in part)
            Wm, bv = torch.zeros(len(part) * D, Wg), torch.zeros(len(part) * D)
            for q, k in enumerate(part):
                Wm[q * D:(q + 1) * D, :widths[k]] = maps[k].weight.detach()
                bv[q * D:(q + 1) * D] = maps[k].bias.detach()
                block[k] = m + q
            m += len(part)
            self.register_buffer(f'{tag}_W{gi}', Wm, persistent=False)
            self.register_buffer(f'{tag}_b{gi}', bv, persistent=False)
            getattr(self, f'{tag}_groups').append(Wg)
        return block

    def _grouped(self, x, tag, block):
        """Each token's own segment's input map of its floats x [n, width]: every group's product, the token's block
        (`block` [n]: its place in the groups' blocks, in order) picked from its group's (a Where per group: no
        concatenation of the groups' outputs)."""
        n, D = x.shape[0], self.net.d
        y, base = None, 0
        for gi, Wg in enumerate(getattr(self, f'{tag}_groups')):
            W = getattr(self, f'{tag}_W{gi}')
            K = W.shape[0] // D
            yg = F.linear(x[:, :Wg], W, getattr(self, f'{tag}_b{gi}')).view(n, K, D)
            loc = block - base
            pg = yg.gather(1, _iclamp(loc, 0, K - 1)[:, None, None].expand(n, 1, D)).squeeze(1)
            y = pg if y is None else torch.where(((loc >= 0) & (loc < K))[:, None], pg, y)
            base += K
        return y

    def _ident(self, blk, k):
        """Identities of digit blocks blk [n, R] of identity layout k (`_identity_digits`)."""
        return _identity_digits(blk, getattr(self, f'f_dval{k}'), getattr(self, f'f_dind{k}'), getattr(self, f'f_dmul{k}'),
                                self.net.vocab)

    def _tail_maps(self, val, k, pick):
        """Each tail token's segment's input map, by width group (`_grouped`)."""
        return self._grouped(val, 't', self.t_block.index_select(0, k))

    def _tail_ids(self, t, val, segc):
        """The tail tokens' identity embeddings, each identity layout's digits decoded once for every token
        (`TokenNet._tail_embed_static`'s: the same identities as the default graph's decode per segment; a token adds
        exactly its own segment's term, as there)."""
        net = self.net
        for n, (at, k) in enumerate(self.t_ids):
            i = self._ident(val[:, at:at + sum(self.radixes[k])], k)
            t = t + torch.where(getattr(self, f't_uses{n}').index_select(0, segc)[:, None], net.emb(i),
                                torch.zeros((), dtype=t.dtype))
        return t

    def _bag(self, gi, b, below, x_ptr, x_idx, x_val):
        """`TokenNet._global`'s pooling of one bag group b ([B, count, d + 1]: each bag's count-weighted sum of the
        identity embeddings, then its total) from the sparse rows: a row's non-zeros inside the group are a run of its
        ascending features (its first and their number from running sums of `x_idx < bound`, exact integers); the run
        padded to the batch's longest (M), each non-zero's bag and identity (its feature's place in the group), the
        weights [B, count, M] (the value where the bag matches, else 0) times the identities' embeddings [B, M, d]: the
        dense product's non-zero terms, in its order (a zero term adds nothing), and its totals exactly."""
        net = self.net
        o, k, w = int(b['offset']), int(b['count']), int(b['width'])
        B = x_ptr.shape[0] - 1
        dev = x_val.device
        f, v, inb = self._runs(o, o + k * w, below, x_ptr, x_idx, x_val)
        M = f.shape[1]
        f = torch.where(inb, f - o, 0).reshape(-1)
        q = getattr(self, f'b_bag{gi}').index_select(0, f).view(B, M)
        c = getattr(self, f'b_col{gi}').index_select(0, f)
        hit = (q[:, None, :] == torch.arange(k, device=dev)[None, :, None]) & inb[:, None, :]
        A = torch.where(hit, v[:, None, :], torch.zeros((), dtype=v.dtype))
        E = net.emb.weight.index_select(0, c).view(B, M, net.d)
        return torch.cat([torch.matmul(A, E), A.sum(-1, keepdim=True)], -1)

    def _scatter_rows(self, n, dest, ok, val):
        """A dense buffer of `n` places from the non-zeros: each non-zero with `ok` to its place `dest`, every other one to a
        spare place of its own past the end (n + its index: every index unique, docs/az.md 41.2); the first n sliced off
        (a Slice after a Scatter, which DirectML's graph fusion fails on - it refuses zero-onnx/2 graphs anyway)."""
        nnz = val.shape[0]
        dev = val.device
        at = torch.where(ok, dest, n + torch.arange(nnz, device=dev))
        buf = torch.zeros(n + nnz, dtype=val.dtype, device=dev).scatter(0, at, val)
        return buf.index_select(0, torch.arange(n, device=dev))

    def _below(self, x_ptr, x_idx, row):
        """Each row's non-zeros below each run bound (`r_bounds`) -> [B, bounds]: a row's first non-zero at or past a bound
        (its features ascend: at most one, the first of its row or one after a non-zero below the bound) marks the place,
        a scatter of the marks' positions to unique places per bound (a row without one keeps its end); no running sum
        over the non-zeros (ONNX Runtime 1.24.4's CUDA CumSum takes ~45 ns an element: 8 ms over a 1,024-row batch's 180k
        non-zeros). `row`: each non-zero's row (`_csr_rows`)."""
        B = x_ptr.shape[0] - 1
        nnz = x_idx.shape[0]
        dev = x_ptr.device
        f = x_idx.long()
        ar = torch.arange(nnz, device=dev)
        prev = f.index_select(0, _iclamp(ar - 1, 0))
        first = ar == x_ptr.index_select(0, row)
        bd = self.r_bounds.long()
        mark = (f[:, None] >= bd[None, :]) & (first[:, None] | (prev[:, None] < bd[None, :]))     # [nnz, bounds]
        ends = x_ptr[1:]
        pos = []
        for k in range(bd.shape[0]):
            at = torch.where(mark[:, k], row, B + ar)
            pos.append(torch.cat([ends, ar]).scatter(0, at, ar)[:B])
        return torch.stack(pos, 1) - x_ptr[:-1][:, None]

    def _runs(self, lo, hi, below, x_ptr, x_idx, x_val):
        """Each row's non-zeros with a feature in [lo, hi) - a run of its ascending features: its first and their number
        from the counts below the bounds (`below`, `_below`) -, padded to the batch's longest (M >= 1) -> (features [B, M]
        (0 in the padding), values [B, M] (0 in the padding), in-run mask [B, M])."""
        B = x_ptr.shape[0] - 1
        dev = x_val.device
        a = below[:, self.r_bound[lo]]
        n = below[:, self.r_bound[hi]] - a
        M = _iclamp(n.max(), 1).item()                            # (a data-dependent shape: dynamic in the graph)
        torch._check(M >= 1)
        ar = torch.arange(M, device=dev)
        inb = ar[None, :] < n[:, None]
        at = torch.where(inb, (x_ptr[:-1] + a)[:, None] + ar[None, :], 0).reshape(-1)
        f = torch.where(inb, x_idx.index_select(0, at).view(B, M).long(), 0)
        v = torch.where(inb, x_val.index_select(0, at).view(B, M), torch.zeros((), dtype=x_val.dtype))
        return f, v, inb

    def _gmap(self, below, x_ptr, x_idx, x_val):
        """`net.g` of the global features (`gidx`) from the sparse rows: per range of consecutive global features each
        row's run (`_runs`), the map's columns of its features times their values, summed - the dense product's non-zero
        terms (a zero term adds nothing) - plus the bias -> [B, d]."""
        net = self.net
        B = x_ptr.shape[0] - 1
        fs, vs = [], []
        for lo, hi, base in self.g_ranges:
            f, v, inb = self._runs(lo, hi, below, x_ptr, x_idx, x_val)
            fs.append(torch.where(inb, f - lo + base, 0))
            vs.append(v)
        c, v = torch.cat(fs, 1), torch.cat(vs, 1)
        W = net.g.weight.t().index_select(0, c.reshape(-1)).view(B, -1, net.d)
        return torch.matmul(v[:, None, :], W).squeeze(1) + net.g.bias

    def _inputs(self, x_ptr, x_idx, x_val):
        """What the network reads of the sparse rows, without densifying them (the default graph's [B, features]): each
        non-zero's row (`_csr_rows`) and its feature's kind and place (the tables) -> (the presence [B, n_tok], each
        reader's inputs [B, n], the present head slots' floats (row-major, each slot's own width), the
        present head slots (b, position) (NonZero: the global token aside), each non-zero's row). A non-zero lands in one unique
        place of one buffer (a row's features ascend, each is one row's once), anything else in a spare place: absent
        slots' floats never reach the arithmetic. The global features: `_gmap`; the bags: `_bag`."""
        net = self.net
        B = x_ptr.shape[0] - 1
        NTOK = net.n_tok
        nnz = x_idx.shape[0]
        row = _csr_rows(x_ptr, nnz, True)
        fi = x_idx.long()
        tok = self.f_tok.index_select(0, fi).long()
        col = self.f_col.index_select(0, fi).long()
        rd = []
        for n, r in enumerate(net.readers or []):
            p = getattr(self, f'f_rd{n}').index_select(0, fi).long()
            L = r.idx.shape[0]
            rd.append(self._scatter_rows(B * L, row * L + p, p >= 0, x_val).view(B, L))
        # the presence: each slot's present flag (its segment's present column) > 0.5, or a position always present
        slot = tok >= 1
        tokc = _iclamp(tok, 0)
        fp = row * NTOK + tokc
        flag = slot & (col == self.t_pcol.index_select(0, tokc))
        pres = (self._scatter_rows(B * NTOK, fp, flag, x_val).view(B, NTOK) > 0.5) | self.f_always[None, :]
        # the present head slots (the global token aside), row-major: (b, position); each one's own floats one after
        # the other in that order (its first float `off`: the widths before it)
        ps = torch.cat([torch.zeros_like(pres[:, :1]), pres[:, 1:]], 1)
        nz = torch.nonzero(ps)
        cr = ps.long().cumsum(1)                                 # (two short running sums, not one over B n_tok)
        rc = cr[:, -1]
        cs = ((rc.cumsum(0) - rc)[:, None] + cr - 1).reshape(-1) # a present slot's place among them
        wt = self.t_w.index_select(0, nz[:, 1])
        off = torch.cat([wt.cumsum(0) - wt, wt.sum(0, keepdim=True)])     # (n + 1: never an empty axis)
        tot = off[-1].item()                                     # (a data-dependent size: dynamic in the graph)
        torch._check(tot >= 0)
        place = off.index_select(0, _iclamp(cs.index_select(0, fp), 0))
        floats = self._scatter_rows(tot, place + col, slot & ps.reshape(-1).index_select(0, fp), x_val)
        return pres, rd, floats, nz, row, (ps, cs, off)

    def _read(self, rd):
        """`net._read` on the readers' inputs [B, n] each -> (their raw outputs, what they add to the global token)."""
        net = self.net
        raws, entry = [], None
        for r, xr in zip(net.readers, rd):
            raw = r.mlp(xr)
            raws.append(raw)
            if r.proj is not None:
                e = r.proj(_predict(r.spec['loss'], raw.float()))
                entry = e if entry is None else entry + e
        return (torch.cat(raws, 1) if len(raws) > 1 else raws[0]), entry

    def _global(self, x_ptr, x_idx, x_val, row, entry):
        """`TokenNet._global` (the global token's input [B, d] before its position embedding) from the sparse rows: the
        global features' map (`_gmap`), the bags (`_bag`), the readers' `entry`."""
        net = self.net
        below = self._below(x_ptr, x_idx, row)
        g = self._gmap(below, x_ptr, x_idx, x_val)
        if net.bag is not None:
            g = g + net.bag(torch.cat([self._bag(n, b, below, x_ptr, x_idx, x_val) for n, b in enumerate(net.bags)],
                                      1).flatten(1))
        if entry is not None:
            g = g + entry
        return g

    def _slots(self, floats, nz, psc):
        """The present head slots' tokens (`TokenNet._pack`: the segment's input map of the slot's floats + its position's
        embedding + its identity's embedding, a segment with an identity), in the row-major order of `nz` (one NonZero
        over every head slot): each segment's map on its own present slots - a NonZero of its own, their rows of
        `floats` by their places (`psc`: the presence without the global token and each position's place in that
        order) - and put back in place by one scatter of unique rows; the positions' embeddings; each identity layout's
        digits decoded once for every slot (`_identity_digits`) -> (tokens [n, d], rows [n], positions [n], identities
        [n] (vocab: none)). No segment's count is used as a size or an offset (only the tensors' own shapes): ONNX
        Runtime 1.24.4's CUDA allocation planner reused live buffers in graphs that computed with such counts."""
        net = self.net
        b, j = nz[:, 0], nz[:, 1]
        ps, cs, off = psc
        NTOK = net.n_tok
        outs, idxs = [], []
        for hs in self.heads:
            st, k, w = hs['start'], hs['count'], hs['width']
            z = torch.nonzero(ps[:, st:st + k])
            idx = cs.index_select(0, z[:, 0] * NTOK + (z[:, 1] + st))
            fl = floats.index_select(0, (off.index_select(0, idx)[:, None] + torch.arange(w, device=idx.device)[None, :])
                                     .reshape(-1)).view(-1, w)
            outs.append(net.seg_in[hs['seg']](fl))
            idxs.append(idx)
        maps = floats.new_zeros(j.shape[0], net.d).index_put((torch.cat(idxs),), torch.cat(outs))
        t = maps + net.pos.index_select(0, j)
        lay = self.t_lay.index_select(0, j)
        ident = torch.full_like(j, net.vocab)
        for k in self.h_lays:
            R = sum(self.radixes[k])
            cols = (off[:-1] + self.t_at.index_select(0, j))[:, None] + torch.arange(R, device=j.device)[None, :]
            # (a slot without this layout reads the buffer's first floats - inside it: a slot has floats -, replaced)
            cols = torch.where((lay == k)[:, None], cols, 0)
            ik = self._ident(floats.index_select(0, cols.reshape(-1)).view(-1, R), k)
            ident = torch.where(lay == k, ik, ident)
        if self.h_lays:
            t = t + torch.where((lay >= 0)[:, None], net.emb(ident), torch.zeros((), dtype=t.dtype))
        return t, b, j, ident

    def forward(self, x_ptr, x_idx, x_val, n_players, belief, opt_off, opt_action, t_off, t_seg, t_slot_hi, t_slot_lo,
                t_val, ov_off, ov_opt, ov_seg, ov_col, ov_slot_hi, ov_slot_lo, t_align=None):
        net = self.net
        B = n_players.shape[0]
        dev = x_val.device
        D, NTOK, A1 = net.d, net.n_tok, net.actions + 1
        aux = bool(self.layout)
        pres, rd, floats, nz, row, psc = self._inputs(x_ptr, x_idx, x_val)
        rraw = ent = None
        if net.readers is not None:
            rraw, ent = self._read(rd)
        g = self._global(x_ptr, x_idx, x_val, row, ent)
        ts, bs, js, ids_s = self._slots(floats, nz, psc)
        seg, segc, shi, slo, trow, tbad, lbad, te = self._tails(g, t_off, t_seg, t_slot_hi, t_slot_lo, t_val)
        NT = t_seg.shape[0]
        # ---- the packed list: the global tokens (row order), the present head slots (row-major), the tail tokens
        ar_b = torch.arange(B, device=dev)
        row_h = torch.cat([ar_b, bs])
        j_h = torch.cat([torch.zeros_like(ar_b), js])
        N = row_h.shape[0]
        torch._check(N >= 2)
        hp = torch.cat([g + net.pos[0], ts, te])
        # each token's place in its padded row: a head token's its rank among the row's present positions (layout
        # order), a tail token's after the row's head tokens; the padded rows' tokens by one scatter of unique places
        rk = pres.long().cumsum(1) - 1                             # [B, n_tok] (>= 0: the global token is present)
        n_h = pres.sum(1)
        n = n_h + (t_off[1:] - t_off[:-1])
        nmax = _iclamp(n.max(), 2)
        # `t_align` (optional, default 1: as before; docs/az.md 66): the padded rows' token axis rounded up to a multiple
        # of it, at most `pad` past the longest row - more padding keys, masked: the same answers up to float32 rounding
        pad = max(0, min(T_ALIGN_PAD, int(net.features) + 1 - NTOK))   # (`int32_bound`'s B (F + 1 + tail tokens))
        if t_align is not None:
            a = _iclamp(t_align.reshape(()), 1, T_ALIGN_PAD + 1)
            up = (nmax + (a - 1)) // a * a
            nmax = torch.where(up < nmax + pad, up, nmax + pad)
        T = nmax.item()                                          # (a data-dependent shape: dynamic in the graph)
        torch._check(T >= 2)
        torch._check(T <= NTOK + NT + pad)
        back = torch.cat([row_h * T + rk.reshape(-1).index_select(0, row_h * NTOK + j_h),
                          trow * T + n_h.index_select(0, trow) + (torch.arange(NT, device=dev) - t_off.index_select(0, trow))])
        M = N + NT
        rows = torch.cat([row_h, trow])
        gidx = torch.full((B * T,), M, dtype=torch.long, device=dev).scatter(0, back, torch.arange(M, device=dev))
        valid = (gidx < M).view(B, T)
        gq = torch.where(valid.reshape(-1), gidx, 0)              # (a padding key reads token 0: masked)
        if net.gated:
            # (a token's weight: 1, a gated segment's g, 0 padding - the attention's float mask log w, the pooling)
            gi = self.v_gi.index_select(0, segc)
            wt = torch.where(gi >= 0, self.v_g.index_select(0, _iclamp(gi, 0)), torch.ones((), dtype=torch.float32))
            wpad = torch.cat([torch.ones(N, dtype=torch.float32, device=dev), wt,
                              torch.zeros(1, dtype=torch.float32, device=dev)]).index_select(0, gidx).view(B, T)
            bias = torch.log(wpad)[:, None, None, :].to(hp.dtype)
        else:
            bias = torch.where(valid, 0.0, float('-inf'))[:, None, None, :].to(hp.dtype)
        H = net.blocks[0].h
        hd = D // H
        if self.attention:
            bias4 = bias.expand(B, 1, T, T)
        for blk in net.blocks:
            qkv = blk.qkv(blk.ln1(hp)).index_select(0, gq)
            if self.attention:
                # (one product split in three: q, k and v each by a product and a gather of its own - no Split - broke
                # ONNX Runtime 1.24.4's CUDA allocation planner, a ScatterElements' sizes 15,404 against 27,064 at 1,024
                # rows: measured, docs/az.md 66)
                q3, k3, v3 = torch.split(qkv.view(B, T, 3 * D), [D, D, D], dim=2)
                a = torch.ops.zero_onnx.fused_attention(q3, k3, v3, bias4, H).reshape(B * T, D)
            else:
                q, k, v = qkv.view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
                s = torch.matmul(q, k.transpose(-2, -1)) * (hd ** -0.5) + bias
                a = torch.matmul(torch.softmax(s, -1), v).transpose(1, 2).reshape(B * T, D)
            hp = hp + blk.out(a.index_select(0, back))
            hp = hp + blk.ff2(F.gelu(blk.ff1(blk.ln2(hp))))
        hp = net.ln(hp)
        pad = torch.cat([hp, hp.new_zeros(1, D)]).index_select(0, gidx).view(B, T, D)
        if net.gated:
            zm = (pad * wpad[..., None]).sum(1) / wpad.sum(1, keepdim=True)
        else:
            zm = pad.sum(1) / n[:, None].to(hp.dtype)
        z = torch.cat([hp[:B], zm], 1)                             # (the global tokens: the list's first B, row order)
        gen = F.pad(net.gen(z), (0, 1))
        P = _pointer2(net, F.gelu(net.pk(hp) + net.pq(z).index_select(0, rows)))
        cposh = gidx.view(B, T).gather(1, rk)                     # a present head position's packed index
        flat = pres.reshape(-1)
        cols = net.tok_cols.index_select(0, j_h)
        acts = net.tok_acts.index_select(0, j_h)
        # (the head tokens' rows of P by a Gather)
        ar_h = torch.ones_like(row_h).cumsum(0) - 1
        lg = P.index_select(0, ar_h).gather(1, _iclamp(cols, 0))
        if self.punique:
            # (every write its own action: the present head tokens' pointer logits scattered straight into the generic
            # logits, every other write to a spare place of its own - the plain scatter's answer, bit for bit)
            ok = (cols >= 0) & (acts < net.actions)
            # (each (token, column) its own spare place B A1 + token C + column: the head tokens' running count ar_h
            # times the columns, not a running sum over all n_h C writes - ONNX Runtime's CUDA CumSum, docs/az.md 66)
            spare = B * A1 + ar_h[:, None] * cols.shape[1] + torch.arange(cols.shape[1], device=dev)[None, :]
            dest = torch.where(ok, row_h[:, None] * A1 + acts, spare)
            base = F.pad(gen.reshape(-1), (0, lg.numel())) if self.PAD_ZEROS else torch.cat([gen.reshape(-1), torch.zeros_like(lg).reshape(-1)])
            pol = base.scatter(0, dest.reshape(-1), lg.reshape(-1))[:B * A1]
        else:
            if net.n_id:
                it = torch.cat([torch.full_like(ar_b, net.vocab), ids_s])
                ia = net.tok_idbase.index_select(0, j_h) + it
                ia = torch.where((it < net.vocab) & (ia < net.actions), ia, net.actions)
                acts = torch.where(net.tok_isid.index_select(0, j_h), ia[:, None], acts)
            acts = torch.where(cols >= 0, acts, net.actions)
            pol = _policy_full(gen, acts, lg, flat, cposh.reshape(-1), NTOK, net).reshape(-1)
        pol, oref = self._overrides(pol, gen, P, flat, cposh.reshape(-1), N, NTOK, trow, seg, shi, slo, opt_off,
                                    opt_action, ov_off, ov_opt, ov_seg, ov_col, ov_slot_hi, ov_slot_lo, slice_back=True)
        refused = _row_sum(tbad, t_off) + oref + lbad
        r = net._heads(z, pol.view(B, A1), n_players, aux, None, rraw)
        policy, value, margin, aux_out = _outputs(net, self.layout, B, r[0], r[1], r[2], r[3] if aux else None, n_players,
                                                  belief)
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


class ZeroOnnx2Static(ZeroOnnx2):
    """zero-onnx/2 with no data-dependent shape (`export(..., fast='static')`, graph 'static', docs/az.md 67): the default
    graph's interface plus one input `t_cap` [T] (any dtype; only its length is read) - every row padded to T tokens, a
    feeder's choice of at least each row's head and tail tokens (metadata `presence`: `TokenNet.present_index`, from
    which a feeder counts a row's head tokens; `t_align`: the multiple a GPU feeder rounds T up to). For the GPU: no
    NonZero, no value read back to the host, every size the inputs' own - so nothing waits on the device mid-run and
    ONNX Runtime can fold every shape computation.

    The arithmetic is `ZeroOnnx2`'s (the default graph: the rows densified, every head slot embedded, the tails, gates,
    overrides and refusals), laid out as `TokenNet.forward_static` lays it: the padded rows [B T] built once by a scatter
    of unique places (a row's present head positions at their rank among them, then its tail tokens, the rest a zero
    row), the blocks run on [B T] (no gather or scatter per block), the pooling masked. The static policy and the
    overrides are `ZeroOnnx2`'s, fed the head positions in full ([B n_tok]: a position's own pointer logits row, absent
    ones a zero row) - the same writes in the same order. A row whose tokens pass T is refused (`refused` + 1: a
    feeder's mistake, never answered)."""

    ARGMAX_DTYPE = torch.float32

    def __init__(self, net, packing='packed', attention='mha'):
        if attention not in STATIC_ATTENTIONS:
            raise ValueError(f'static graph: attention {attention!r} (one of {STATIC_ATTENTIONS})')
        super().__init__(net, packing, None if attention == 'matmul' else attention)

    def _trunk_static(self, h, pres, ids, T, opt_off, opt_action, t_off, t_seg, t_slot_hi, t_slot_lo, t_val, ov_off,
                      ov_opt, ov_seg, ov_col, ov_slot_hi, ov_slot_lo):
        """`ZeroOnnx2._trunk2` on rows padded to T tokens -> (z [B, 2 d], logits [B, actions + 1], refused [B])."""
        net = self.net
        B, NTOK, D = h.shape
        dev = h.device
        A1 = net.actions + 1
        NT = t_seg.shape[0]
        BN = B * NTOK
        seg, segc, shi, slo, trow, tbad, lbad, te = self._tails(h, t_off, t_seg, t_slot_hi, t_slot_lo, t_val)
        flat = pres.reshape(-1)
        n_h = pres.sum(1)
        n = n_h + (t_off[1:] - t_off[:-1])
        # ---- the padded rows: each place's source - a head position (b, j) at b T + its rank among the row's present
        # ones, a tail token after its row's head tokens, every other place the zero row; one scatter of unique places
        # (a token past T, a row's refusal, to a spare place of its own past B T)
        ar_bn = torch.arange(BN, device=dev)
        brow = ar_bn // NTOK
        rk = (pres.long().cumsum(1) - 1).reshape(-1)              # (a short running sum: along each row's n_tok)
        hok = flat & (rk < T)
        dest_h = torch.where(hok, brow * T + rk, B * T + ar_bn)
        ar_t = torch.arange(NT, device=dev)
        pos_t = n_h.index_select(0, trow) + (ar_t - t_off.index_select(0, trow))
        tok = pos_t < T
        dest_t = torch.where(tok, trow * T + pos_t, B * T + BN + ar_t)
        zrow = BN + NT
        gidx = torch.full((B * T + BN + NT,), zrow, dtype=torch.long, device=dev).scatter(
            0, torch.cat([dest_h, dest_t]), torch.cat([ar_bn, BN + ar_t]))
        gidx = gidx.index_select(0, torch.arange(B * T, dtype=torch.long, device=dev))
        hp = torch.cat([h.reshape(BN, D), te, h.new_zeros(1, D)]).index_select(0, gidx)
        valid = (gidx < zrow).view(B, T)
        if net.gated:
            # (a token's weight: 1, a gated segment's g, 0 padding - the attention's float mask log w, the pooling)
            gi = self.v_gi.index_select(0, segc)
            wt = torch.where(gi >= 0, self.v_g.index_select(0, _iclamp(gi, 0)), torch.ones((), dtype=torch.float32))
            wpad = torch.cat([torch.ones(BN, dtype=torch.float32, device=dev), wt,
                              torch.zeros(1, dtype=torch.float32, device=dev)]).index_select(0, gidx).view(B, T)
            bias = torch.log(wpad)[:, None, None, :].to(h.dtype)
        else:
            bias = torch.where(valid, 0.0, float('-inf'))[:, None, None, :].to(h.dtype)
        H = net.blocks[0].h
        hd = D // H
        if self.attention:
            bias4 = bias.expand(B, 1, T, T)
        for blk in net.blocks:
            qkv = blk.qkv(blk.ln1(hp))
            if self.attention:
                q3, k3, v3 = torch.split(qkv.view(B, T, 3 * D), [D, D, D], dim=2)
                a = torch.ops.zero_onnx.fused_attention(q3, k3, v3, bias4, H).reshape(B * T, D)
            else:
                q, k, v = qkv.view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
                s = torch.matmul(q, k.transpose(-2, -1)) * (hd ** -0.5) + bias
                a = torch.matmul(torch.softmax(s, -1), v).transpose(1, 2).reshape(B * T, D)
            hp = hp + blk.out(a)
            hp = hp + blk.ff2(F.gelu(blk.ff1(blk.ln2(hp))))
        hp = net.ln(hp)
        pad = hp.view(B, T, D)
        # (the pooling: a padding place weighs 0 - where the default graph's padding rows are zeros)
        if net.gated:
            zm = (pad * wpad[..., None]).sum(1) / wpad.sum(1, keepdim=True)
        else:
            zm = (pad * valid[..., None].to(hp.dtype)).sum(1) / n[:, None].to(hp.dtype)
        z = torch.cat([pad[:, 0], zm], 1)
        gen = F.pad(net.gen(z), (0, 1))
        P = _pointer2(net, F.gelu(net.pk(hp) + net.pq(z).index_select(0, torch.arange(B * T, device=dev) // T)))
        # ---- the head positions in full ([B n_tok]): each one's pointer logits row (absent or past T: a zero row),
        # its actions; then the tail tokens' rows - `_policy_full` and `_overrides` as the packed list feeds them
        P0 = torch.cat([P, P.new_zeros(1, P.shape[1])])
        Ph = P0.index_select(0, torch.where(hok, dest_h, B * T))
        Pt = P0.index_select(0, torch.where(tok, dest_t, B * T))
        slot_h = ar_bn - brow * NTOK
        cols = net.tok_cols.index_select(0, slot_h)
        acts = net.tok_acts.index_select(0, slot_h)
        if net.n_id:
            it = ids.reshape(-1)
            ia = net.tok_idbase.index_select(0, slot_h) + it
            ia = torch.where((it < net.vocab) & (ia < net.actions), ia, net.actions)
            acts = torch.where(net.tok_isid.index_select(0, slot_h), ia[:, None], acts)
        acts = torch.where(cols >= 0, acts, net.actions)
        lg = Ph.gather(1, _iclamp(cols, 0))
        pol = _policy_full(gen, acts, lg, flat, ar_bn, NTOK, net).reshape(-1)
        pol, oref = self._overrides(pol, gen, torch.cat([Ph, Pt]), flat, ar_bn, BN, NTOK, trow, seg, shi, slo, opt_off,
                                    opt_action, ov_off, ov_opt, ov_seg, ov_col, ov_slot_hi, ov_slot_lo)
        refused = _row_sum(tbad, t_off) + oref + lbad + (n > T).long()
        return z, pol.view(B, A1), refused

    def forward(self, x_ptr, x_idx, x_val, n_players, belief, opt_off, opt_action, t_off, t_seg, t_slot_hi, t_slot_lo,
                t_val, ov_off, ov_opt, ov_seg, ov_col, ov_slot_hi, ov_slot_lo, t_cap):
        net = self.net
        x = densify(x_ptr, x_idx, x_val, net.features)
        aux = bool(self.layout)
        rraw = ent = None
        if net.readers is not None:
            rraw, ent = _read(net, x)
        h, pres, ids = slot_tokens(net, x, ent)
        z, pol, refused = self._trunk_static(h, pres, ids, t_cap.shape[0], opt_off, opt_action, t_off, t_seg, t_slot_hi,
                                             t_slot_lo, t_val, ov_off, ov_opt, ov_seg, ov_col, ov_slot_hi, ov_slot_lo)
        r = net._heads(z, pol, n_players, aux, None, rraw)
        policy, value, margin, aux_out = _outputs(net, self.layout, x.shape[0], r[0], r[1], r[2],
                                                  r[3] if aux else None, n_players, belief)
        return policy, value, margin, aux_out, refused


def static_t(n, align=1):
    """The static graph's T for rows of `n` tokens each (head + tail): their most, at least 2, rounded up to a multiple
    of `align` (docs/az.md 67)."""
    t = max(2, int(np.max(n)) if len(n) else 2)
    return -(-t // max(1, int(align))) * max(1, int(align))


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


def metadata2(net, extra=None, attention=None, fast=False):
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
    if fast == 'static':
        # (docs/az.md 67: ZeroOnnx2Static's graph - its input t_cap, the head positions' present flags a feeder counts
        # a row's head tokens with, the multiple a GPU feeder rounds T up to)
        meta['static'] = '1'
        meta['inputs'] = json.dumps(list(INPUTS2) + ['t_cap'])
        meta['presence'] = json.dumps([int(i) for i in net.present_index()])
        meta['t_align'] = str(T_ALIGN)
    elif fast:
        meta['fast'] = '1'                       # (docs/az.md 63: ZeroOnnx2Fast's graph)
        meta['t_align'] = str(T_ALIGN)           # (docs/az.md 66: its optional input, what a GPU feeder gives it)
    if attention:
        meta['attention'] = str(attention)
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
    if getattr(net, 'dup_reduce', None) is not None:
        meta['dup_reduce'] = str(net.dup_reduce)  # (docs/az.md 58; absent: the plain scatter's last writer, as before)
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


def export(net, path, packing='packed', extra=None, opset=OPSET, make_portable=True, attention=None, fast=False):
    """Export `net` (a TokenNet, GridNet or Net, as loaded: `net.load` / `load_checkpoint`) to the ONNX
    file `path` (one file, no external data; written atomically) -> its metadata: zero-onnx/1, or zero-onnx/2 for
    a token network with variable-length token sets (`onnx_version`; docs/az.md 46). `packing`: a token network's
    ('mask', 'compact', 'host'; module docstring; zero-onnx/2: 'packed' only), ignored for the others. `extra`:
    more metadata (the checkpoint's id and sha256: `export_checkpoint`). `make_portable`: `portable`'s
    rewrites (metadata `portable` 1), which change no answer and let the GPU providers run the graph. `opset`: 17 (the
    default: the exporter's opset-18 graph rewritten by `opset17`, docs/az.md 41.2) or 18 (as the exporter writes it).
    Every scatter of the mask and packed graphs writes unique indices (metadata `unique_scatters` 1; docs/az.md 41.2).
    `attention` (zero-onnx/2 only; default None: as always): 'mha' or 'attention23', a fused attention node per block
    (`ATTENTIONS`, docs/az.md 63; metadata `attention`). `fast` (zero-onnx/2 only; default False: the graph as always):
    `ZeroOnnx2Fast`'s graph - TokenNet's own arithmetic, the same interface and answers (docs/az.md 63; metadata `fast`
    '1' and `attention`, default 'mha': `FAST_ATTENTIONS`)."""
    attention = fast_attention(attention, fast)
    mod, args, names, outs, dyn, meta_of = _export_module(net, packing, attention, fast)
    table = {torch.ops.zero_onnx.fused_attention.default: _fused_attention_onnx} if mod_fused(attention) else None
    with torch.no_grad():
        prog = torch.onnx.export(mod, args, None, dynamo=True, opset_version=EXPORTER_OPSET, input_names=names,
                                 output_names=outs, dynamic_shapes=dyn, optimize=True, verbose=False,
                                 **({'custom_translation_table': table} if table else {}))
    meta = meta_of(extra)
    return _save(prog.model_proto, meta, path, opset, make_portable, attention=attention if mod_fused(attention) else None,
                 gelu=bool(fast), instance=fast is True)


def fast_attention(attention, fast):
    """The attention a graph is exported with: `attention` as given, the fast graph's default ('mha') when `fast`
    and None; refused when it is not one the graph takes (`ATTENTIONS`; the fast graph's `FAST_ATTENTIONS`; the static
    graph's - `fast` 'static' - `STATIC_ATTENTIONS`)."""
    if fast == 'static':
        attention = STATIC_ATTENTIONS[0] if attention is None else attention
        if attention not in STATIC_ATTENTIONS:
            raise ValueError(f'static graph: attention {attention!r} (one of {STATIC_ATTENTIONS})')
    elif fast:
        attention = FAST_ATTENTIONS[0] if attention is None else attention
        if attention not in FAST_ATTENTIONS:
            raise ValueError(f'fast graph: attention {attention!r} (one of {FAST_ATTENTIONS})')
    elif attention not in ATTENTIONS:
        raise ValueError(f'attention {attention!r}: one of {ATTENTIONS}')
    return attention


def mod_fused(attention):
    """Whether `attention` puts a fused node in the graph ('mha', 'attention23'; not None or 'matmul')."""
    return attention in ATTENTIONS[1:]


def _export_module(net, packing='packed', attention=None, fast=False):
    """What `export` traces for `net`: (the interface module - ZeroOnnx2 for zero-onnx/2 (ZeroOnnx2Fast with `fast`),
    else ZeroOnnx - around a float32 CPU eval copy of `net`, the example inputs, the input names, the output names, the
    dynamic shapes, `extra` -> the graph's metadata)."""
    net = copy.deepcopy(net).float().cpu().eval()
    for p in net.parameters():
        p.requires_grad_(False)
    if isinstance(net, TokenNet):
        net.chunk = 1 << 30                       # (forward's chunking is never traced; set for clarity)
    attention = fast_attention(attention, fast)
    if onnx_version(net) == 2:
        # (variable-length token sets: zero-onnx/2, docs/az.md 46)
        if fast == 'static':
            # (the static graph's input `t_cap` last: only its length is read, the padded rows' T; docs/az.md 67)
            mod = ZeroOnnx2Static(net, packing, attention).eval()
            T = int(net.n_tok) + 7                   # (traced with room for every example row's tokens)
            return (mod, tuple(example_inputs2(net)) + (torch.zeros(T, dtype=torch.int32),), list(INPUTS2) + ['t_cap'],
                    list(OUTPUTS2), dict(dynamic_shapes2(), t_cap={0: torch.export.Dim('T', min=2, max=1 << 20)}),
                    lambda extra: metadata2(net, extra, attention=attention, fast=fast))
        if fast:
            # (the fast graph's optional input `t_align` last, traced with 1: docs/az.md 66)
            mod = ZeroOnnx2Fast(net, packing, attention).eval()
            return (mod, tuple(example_inputs2(net)) + (torch.tensor(1, dtype=torch.int64),), list(INPUTS2) + ['t_align'],
                    list(OUTPUTS2), dict(dynamic_shapes2(), t_align=None),
                    lambda extra: metadata2(net, extra, attention=attention, fast=fast))
        mod = ZeroOnnx2(net, packing, attention).eval()
        return (mod, example_inputs2(net), list(INPUTS2), list(OUTPUTS2), dynamic_shapes2(),
                lambda extra: metadata2(net, extra, attention=attention, fast=fast))
    if attention or fast:
        raise ValueError(f'attention {attention!r}, fast {fast!r}: a zero-onnx/2 graph\'s options (docs/az.md 63)')
    mod = ZeroOnnx(net, packing).eval()
    packing = mod.packing
    args = example_inputs(net, packing)
    names = list(INPUTS) + (list(HOST_INPUTS) if packing == 'host' else [])
    dyn = dynamic_shapes(packing)
    if packing != 'host':
        dyn = {k: v for k, v in dyn.items() if k in names}
    return mod, args, names, list(OUTPUTS), dyn, lambda extra: metadata(net, packing, extra)


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
    its `axes` input as it is (version 13), Split -18 with its `split` input (the sizes) and no num_outputs as it is
    (version 13); anything else (Split's num_outputs, Resize-18, ...) raises. The same
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
            elif n.op_type == 'Split':
                if any(a.name == 'num_outputs' for a in n.attribute) or len(n.input) < 2 or not n.input[1]:
                    raise ValueError(f'opset17: {where}{n.name} (Split): num_outputs or no split input (opset 18 only)')
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


def _save(model, meta, path, opset, make_portable, int32=True, attention=None, gelu=False, instance=True):
    """The exported model with its metadata (`fused_attention`'s, then with `gelu` (the fast graph's) `fuse_gelu`'s and
    `layernorm_instance`'s rewrites, then `portable`'s; the
    exporter's annotations stripped; `int32_slices` then `check_int32`; `opset17` for opset 17), checked, written
    atomically -> the metadata."""
    import onnx
    if attention:
        fused_attention(model, attention)
        if attention == 'attention23':
            opset = 23
    if gelu:
        fuse_gelu(model)
        if instance:
            # (the fast graph's: the layer norms as InstanceNormalization, its optional input t_align; not the static
            # graph's - the GPU's LayerNormalization is the faster there, docs/az.md 67)
            layernorm_instance(model)
            _optional_input(model, 't_align', 1)     # (the fast graph's optional input: default 1, docs/az.md 66)
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
    elif opset not in (18, 23) or (opset == 23 and attention != 'attention23'):
        raise ValueError(f'zero-onnx: opset {opset} (17, the default, or 18; 23 with attention23)')
    _write_metadata(model, meta)
    onnx.checker.check_model(model)
    _write_model(model, path)
    return meta


def _write_metadata(model, meta):
    """The graph's metadata replaced by `meta` (in its order) and its producer named. In place."""
    del model.metadata_props[:]
    for k, v in meta.items():
        e = model.metadata_props.add()
        e.key, e.value = k, v
    model.producer_name = 'zero-core zero.onnx_export'


FP16_KEEP = ('ConstantOfShape',)  # (`to_float16`: kept float32 - the converter leaves its value attribute float32)


def _toposort(graph):
    """The graph's nodes reordered so that each comes after the producers of its inputs, otherwise in their order: each
    node in turn, the producers it still waits for placed just before it (depth first) - a node out of place (the
    float16 converter's input casts, appended after their readers) moves to before its first reader and nothing else
    moves. The order matters on the CUDA pack: ONNX Runtime 1.24.4 plans its buffer reuse from it, and a sort by passes
    (every ready node, then again) reordered the whole graph into a plan that broke on the self-test's batches ("Shape
    mismatch attempting to re-use buffer" - measured, docs/az.md 67). In place; ValueError for an input nothing produces
    or a cycle."""
    have = {i.name for i in graph.input} | {i.name for i in graph.initializer} | {''}
    prod = {o: n for n in graph.node for o in n.output}
    state, out = {}, []                      # id(node) -> 1 visiting, 2 placed

    def visit(n):
        stack = [(n, iter(n.input))]
        state[id(n)] = 1
        while stack:
            node, it = stack[-1]
            for x in it:
                if x in have:
                    continue
                p = prod.get(x)
                if p is None:
                    raise ValueError(f'graph: node {node.name} reads {x!r}, which nothing produces')
                if state.get(id(p)) == 1:
                    raise ValueError(f'graph: a cycle through {p.name}')
                if not state.get(id(p)):
                    state[id(p)] = 1
                    stack.append((p, iter(p.input)))
                    break
            else:
                stack.pop()
                state[id(node)] = 2
                out.append(node)
                have.update(node.output)
    for n in list(graph.node):
        if not state.get(id(n)):
            visit(n)
    del graph.node[:]
    graph.node.extend(out)


def to_float16(src, dst):
    """A float16 copy of the zero-onnx graph at `src`, written to `dst` (docs/az.md 67; for GPU workers: the user's
    decision of 2026-11-19 to accept reduced precision there): its float32 arithmetic in float16, its inputs and outputs
    float32 (onnxruntime.transformers' convert_float_to_float16 with keep_io_types; float32 kept for the converter's
    default list and `FP16_KEEP`), its metadata `src`'s plus `precision` 'fp16' - a worker self-tests such a graph at its
    float16 tolerances (zero_worker `selftest::TOL_FP16` 5e-2, `MEAN_TOL_FP16` 3e-3 per item; measured on a live run's
    network: priors 4.2e-3, value 8.5e-3, margin 1.0e-2 at most) and still bit for bit the same twice -> the metadata.
    Deterministic: the same `src` gives the same bytes."""
    import onnx
    from onnxruntime.transformers import float16
    m = onnx.load(src)
    meta = {p.key: p.value for p in m.metadata_props}
    if meta.get('precision', 'fp32') != 'fp32':
        raise ValueError(f'{src}: already precision {meta["precision"]!r}')
    m16 = float16.convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=True,
                                           op_block_list=list(float16.DEFAULT_OP_BLOCK_LIST) + list(FP16_KEEP))
    _toposort(m16.graph)                     # (the converter appends its input casts after their readers)
    meta['precision'] = 'fp16'
    _write_metadata(m16, meta)
    onnx.checker.check_model(m16)
    _write_model(m16, dst)
    return meta


def _write_model(model, path):
    """One file, no external data, written atomically."""
    import onnx
    tmp = f'{path}.{os.getpid()}.tmp'
    onnx.save_model(model, tmp, save_as_external_data=False)
    os.replace(tmp, path)


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


def export_checkpoint(path, out, packing='packed', game=None, allow_mismatch=False, checkpoint_id=None, attention=None,
                      fast=False):
    """`load_checkpoint` + `export` -> the metadata written. `checkpoint_id`: the id the run publishes
    the checkpoint under (default its file name)."""
    m, info = load_checkpoint(path, game, allow_mismatch)
    if checkpoint_id:
        info['checkpoint'] = checkpoint_id
    return export(m, out, packing, info, attention=attention, fast=fast)


# ---------------------------------------------------------------- the weight-swap template (docs/az.md 53)
TEMPLATE = 1                       # the template's format and its classification (`export_template`)
_PROBE_SEED = 0x5eed


def module_tensors(mod):
    """Every tensor of an interface module (`_export_module`) by the name the exporter gives its initializer: its
    parameters and buffers (non-persistent ones too: ZeroOnnx2's `v_*`, computed from the weights at construction), the
    first name of a shared tensor -> {name: tensor}."""
    out = {}
    for k, v in list(mod.named_parameters()) + list(mod.named_buffers()):
        out.setdefault(k, v.detach())
    return out


def _versions():
    from importlib import metadata as md
    out = {'torch': torch.__version__}
    for k in ('onnx', 'onnxscript', 'onnx-ir'):
        try:
            out[k] = md.version(k)
        except md.PackageNotFoundError:
            out[k] = None
    return out


def _code_sha256():
    """This zero-core's Python (every zero/*.py: the network, the interface, the exporter's rewrites), hashed."""
    d = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for name in sorted(f for f in os.listdir(d) if f.endswith('.py')):
        with open(os.path.join(d, name), 'rb') as f:
            h.update(name.encode() + b'\0' + hashlib.sha256(f.read()).digest())
    return h.hexdigest()


def export_fingerprint(net, packing='packed', opset=OPSET, make_portable=True, graph=''):
    """The architecture fingerprint a weight-swap template is kept under (docs/az.md 53) -> (sha256 hex, its fields):
    the network's class and config (its layout, sizes, heads, auxiliary targets), the interface module, the packing,
    every module tensor's name, shape and dtype, the export settings (opset, portable), the exporter's versions (torch,
    onnx, onnxscript, onnx-ir), this zero-core's Python and TEMPLATE. Two checkpoints with the same fingerprint export
    to the same graph but for their weights; any other change gives another fingerprint (a fresh full export). `graph`: the
    graph variant (`GRAPHS`; '' the default graph, its fields as before)."""
    attention, fast = graph_options(graph)
    mod = _export_module(net, packing, attention, fast)[0]
    fields = {'template': TEMPLATE, 'class': type(net).__name__, 'module': type(mod).__name__,
              'config': json.loads(json.dumps(net.config, sort_keys=True, default=str)),
              'aux_layout': [list(t) for t in (getattr(net, 'aux_layout', None) or [])], 'packing': mod.packing,
              'opset': int(opset), 'portable': bool(make_portable),
              'tensors': [[k, list(v.shape), str(v.dtype)] for k, v in module_tensors(mod).items()],
              'versions': _versions(), 'code': _code_sha256()}
    if graph:
        fields['graph'] = graph
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest(), fields


def _tensor_bytes(t):
    return np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()


def _init_is(init, t):
    """Is initializer `init` exactly tensor `t` (dtype, shape, raw bytes)?"""
    from onnx import helper
    a = t.detach().cpu().numpy()
    return (helper.tensor_dtype_to_np_dtype(init.data_type) == a.dtype and list(init.dims) == list(a.shape)
            and init.HasField('raw_data') and init.raw_data == _tensor_bytes(t))


def _without_data(model):
    """A copy of `model` with every initializer's data and the model's metadata cleared (the graph's structure, names,
    dtypes and shapes; a swap writes the metadata of its own network: `gates` holds the gates' values)."""
    import onnx
    m = onnx.ModelProto()
    m.CopyFrom(model)
    del m.metadata_props[:]
    for i in m.graph.initializer:
        for f in ('raw_data', 'float_data', 'int32_data', 'int64_data', 'double_data', 'uint64_data', 'string_data'):
            i.ClearField(f)
    return m


_CODE_BASE = 1 << 22               # the coded weights' first value (`_weights_like`; exact float32 integers up to 2^24)


def _weight_sources(mod):
    """The weights a graph can be computed from: the interface module's network's floating-point state (its parameters
    and saved buffers), by the exporter's names (`net.<key>`) in the network's order -> [(name, float32 tensor)]."""
    return [(f'net.{k}', v.detach()) for k, v in mod.net.state_dict().items() if v.is_floating_point()]


def _weights_like(net, how, seed=_PROBE_SEED):
    """A float32 CPU copy of `net` with every floating-point state tensor replaced: 'random' (N(0, 0.5^2), independent
    of the network's values: no exact 0 or 1, either sign) or 'coded' (element i of the network's floating-point state,
    in `_weight_sources`' order, = 2^22 + i: a value names the element it came from)."""
    p = copy.deepcopy(net).float().cpu()
    g = torch.Generator().manual_seed(seed)
    k = 0
    with torch.no_grad():
        for t in p.state_dict(keep_vars=True).values():
            if not t.is_floating_point():
                continue
            if how == 'random':
                t.copy_(torch.randn(t.shape, generator=g) * 0.5)
            else:
                t.copy_((_CODE_BASE + torch.arange(k, k + t.numel(), dtype=torch.float64)).reshape(t.shape))
                k += t.numel()
    if how == 'coded' and _CODE_BASE + k > 1 << 24:
        raise ValueError(f'weight-swap template: {k} weights, past the coded export\'s {(1 << 24) - _CODE_BASE}')
    return p


def _bits(a):
    a = np.ascontiguousarray(a)
    return a.view(np.uint8).reshape(a.shape + (a.dtype.itemsize,)) if a.dtype.itemsize else a


def export_template(net, path, packing='packed', opset=OPSET, make_portable=True, graph=''):
    """The weight-swap template of `net`'s architecture (docs/az.md 53): the graph `export` writes for it with random
    weights, written to `path`, its derived-weight recipes to `<path>.recipes.npz` -> the template's description
    {template, fingerprint, fields, weights, derived, constants, sha256, recipes_sha256}.

    The architecture is exported twice, with random weights and with coded ones (`_weights_like`), and the two graphs
    compared. They must be the same graph (nodes, names, inputs and outputs, attributes, opset, functions, metadata:
    everything but the initializers' data); an initializer with the same bytes in both is a constant (an arange, a
    shape, a table built from the config); one that differs must be either exactly a module tensor of each by its name
    (`module_tensors`: a weight, or a buffer the interface computes from the weights - `weights`), or elements picked
    from the weights (a slice, a transpose, a concatenation the exporter folded; `derived`: each element either the same
    in both - a constant - or a code naming one weight element, the same pick reproducing the random export's values
    exactly); anything else raises ValueError, naming them: the exporter computed something from the weights that a swap
    cannot rebuild, and no template is made. Random weights hold no exact 0 or 1, so the template is the graph of a
    network whose values trigger none of the exporter's value-dependent rewrites (a Gemm's all-zero bias dropped, for
    one): `swap_export` writes that graph for every checkpoint. `constants`: the module tensors among the constants (a
    swap checks they are unchanged). `graph`: the graph variant (`GRAPHS`)."""
    import onnx
    from onnx import numpy_helper
    attention, fast = graph_options(graph)
    fp, fields = export_fingerprint(net, packing, opset, make_portable, graph)
    rnd, cod = _weights_like(net, 'random'), _weights_like(net, 'coded')
    cpath = f'{path}.coded.{os.getpid()}.onnx'
    try:
        export(rnd, path, packing, None, opset, make_portable, attention=attention, fast=fast)
        export(cod, cpath, packing, None, opset, make_portable, attention=attention, fast=fast)
        a, b = onnx.load(path), onnx.load(cpath)
    finally:
        if os.path.exists(cpath):
            os.remove(cpath)
    if _without_data(a).SerializeToString() != _without_data(b).SerializeToString():
        raise ValueError(f'weight-swap template: the graph depends on the weights\' values ({len(a.graph.node)} vs '
                         f'{len(b.graph.node)} nodes, {len(a.graph.initializer)} vs {len(b.graph.initializer)} '
                         f'initializers, random vs coded weights): no template')
    ma, mb = _export_module(rnd, packing, attention, fast)[0], _export_module(cod, packing, attention, fast)[0]
    ta, tb = module_tensors(ma), module_tensors(mb)
    sa = np.concatenate([t.numpy().ravel() for _, t in _weight_sources(ma)])
    weights, consts, bad, recipes = [], [], [], {}
    for ia, ib in zip(a.graph.initializer, b.graph.initializer):
        if ia.SerializeToString() == ib.SerializeToString():
            if ia.name in ta:
                if not _init_is(ia, ta[ia.name]):
                    bad.append(f'{ia.name} (a constant named as a module tensor it is not)')
                consts.append(ia.name)
            continue
        if ia.name in ta and ia.name in tb and _init_is(ia, ta[ia.name]) and _init_is(ib, tb[ia.name]):
            weights.append(ia.name)
            continue
        A, B = numpy_helper.to_array(ia), numpy_helper.to_array(ib)
        if A.dtype != np.float32 or B.shape != A.shape:
            bad.append(f'{ia.name} ({A.dtype} {list(A.shape)})')
            continue
        same = (_bits(A) == _bits(B)).all(-1)
        c = B[~same].astype(np.float64)
        idx = np.full(A.shape, -1, np.int64)
        if not (np.all(c == np.floor(c)) and np.all(c >= _CODE_BASE) and np.all(c < _CODE_BASE + sa.size)):
            bad.append(f'{ia.name} ({A.dtype} {list(A.shape)}: not elements of the weights)')
            continue
        idx[~same] = (c - _CODE_BASE).astype(np.int64)
        if not np.array_equal(_bits(sa[idx[~same]]), _bits(A[~same])):
            bad.append(f'{ia.name} ({A.dtype} {list(A.shape)}: the pick does not give the random export\'s values)')
            continue
        recipes[ia.name] = idx
    if bad:
        raise ValueError(f'weight-swap template: {len(bad)} initializer(s) the weights change that a swap cannot '
                         f'rebuild, no template: ' + '; '.join(bad[:8]))
    if not weights and not recipes:
        raise ValueError('weight-swap template: no initializer depends on the weights')
    rpath = f'{path}.recipes.npz'
    tmp = f'{rpath}.{os.getpid()}.tmp'
    with open(tmp, 'wb') as f:
        np.savez(f, **{f'r{k}': v for k, v in enumerate(recipes.values())})
    os.replace(tmp, rpath)
    return {'template': TEMPLATE, 'fingerprint': fp, 'fields': fields, 'weights': weights,
            'derived': list(recipes), 'constants': consts,
            'sources': [[k, list(t.shape)] for k, t in _weight_sources(ma)], 'sha256': file_sha256(path),
            'recipes_sha256': file_sha256(rpath)}


def swap_export(net, template_path, template, path, packing='packed', extra=None, opset=OPSET, make_portable=True,
                graph=''):
    """`net` written to `path` from a weight-swap template (`export_template`: the template's graph `template_path`, its
    recipes beside it, its description `template`; docs/az.md 53): each weight initializer replaced by `net`'s module
    tensor of its name, each derived one rebuilt by its recipe from `net`'s weights, the metadata `net`'s and `extra`'s
    (as `export` writes it), then checked (`check_swapped`) -> the metadata. Refused (ValueError, nothing written) when
    `net`'s fingerprint is not the template's, a template file is not the template's (sha256), a weight has no tensor of
    its name or another dtype or shape, or a module tensor the template holds constant changed. `graph`: the graph
    variant (`GRAPHS`), the template's."""
    import onnx
    if template.get('template') != TEMPLATE:
        raise ValueError(f'weight-swap template format {template.get("template")}, this zero-core reads {TEMPLATE}')
    attention, fast = graph_options(graph)
    fp, _ = export_fingerprint(net, packing, opset, make_portable, graph)
    if fp != template['fingerprint']:
        raise ValueError(f'weight-swap: the network\'s fingerprint {fp[:16]} is not the template\'s '
                         f'{template["fingerprint"][:16]} (another architecture or exporter: a new template)')
    rpath = f'{template_path}.recipes.npz'
    if file_sha256(template_path) != template['sha256'] or file_sha256(rpath) != template['recipes_sha256']:
        raise ValueError(f'weight-swap: {template_path} or its recipes are not the template\'s (sha256)')
    mod, _a, _n, _o, _d, meta_of = _export_module(net, packing, attention, fast)
    ts = module_tensors(mod)
    src = _weight_sources(mod)
    if [[k, list(t.shape)] for k, t in src] != template['sources']:
        raise ValueError('weight-swap: the network\'s weights are not the template\'s (names or shapes)')
    model = onnx.load(template_path)
    inits = {i.name: i for i in model.graph.initializer}
    want = list(template['weights'])
    bad = [f'{k} (no initializer)' for k in want + template['derived'] if k not in inits]
    bad += [f'{k} (no tensor)' for k in want if k not in ts]
    for k in want:
        if k in inits and k in ts:
            i, a = inits[k], np.ascontiguousarray(ts[k].cpu().numpy())
            dt = onnx.helper.tensor_dtype_to_np_dtype(i.data_type)
            if dt != a.dtype or list(i.dims) != list(a.shape) or not i.HasField('raw_data'):
                bad.append(f'{k} ({a.dtype} {list(a.shape)}, the graph\'s {dt} {list(i.dims)})')
            else:
                i.raw_data = a.tobytes()
    if template['derived']:
        flat = np.concatenate([t.cpu().numpy().astype(np.float32, copy=False).ravel() for _, t in src])
        with np.load(rpath) as z:
            for n, k in enumerate(template['derived']):
                if k not in inits:
                    continue
                i, idx = inits[k], z[f'r{n}']
                if onnx.helper.tensor_dtype_to_np_dtype(i.data_type) != np.float32 or list(i.dims) != list(idx.shape) \
                        or not i.HasField('raw_data') or idx.max(initial=-1) >= flat.size:
                    bad.append(f'{k} (its recipe does not fit the graph)')
                    continue
                a = np.frombuffer(i.raw_data, np.float32).reshape(idx.shape).copy()
                a[idx >= 0] = flat[idx[idx >= 0]]
                i.raw_data = a.tobytes()
    for k in template['constants']:
        if k not in inits or k not in ts or not _init_is(inits[k], ts[k]):
            bad.append(f'{k} (held constant by the template, changed)')
    if bad:
        raise ValueError(f'weight-swap: {len(bad)} tensor(s) do not match the template: ' + '; '.join(bad[:8]))
    meta = meta_of(extra)
    opset = graph_opset(opset, attention)
    meta['opset'] = str(opset)
    if make_portable:
        meta['portable'] = '1'
    _write_metadata(model, meta)
    check_swapped(model, opset, make_portable)
    _write_model(model, path)
    return meta


def check_swapped(model, opset=OPSET, make_portable=True):
    """The graph checks of a swapped graph (docs/az.md 53): `portable` (on a copy) finds nothing left to rewrite (every
    GPU-safe form in place), `int32_slices` nothing to replace, `check_int32` passes (no integer constant outside int32,
    no BitShift), the opset is `opset`, and onnx.checker's full check (strict shape inference) passes -> `portable`'s
    stats (all 0). ValueError otherwise."""
    import onnx
    m = onnx.ModelProto()
    m.CopyFrom(model)
    stats = portable(m) if make_portable else {}
    if any(stats.values()):
        raise ValueError(f'swapped graph: portable would still rewrite {stats}')
    if int32_slices(m):
        raise ValueError('swapped graph: a Slice bound outside int32')
    check_int32(m)
    ver = {o.domain or 'ai.onnx': o.version for o in model.opset_import}
    if ver.get('ai.onnx') != int(opset):
        raise ValueError(f'swapped graph: opset {ver.get("ai.onnx")}, not {opset}')
    onnx.checker.check_model(model, full_check=True)
    return stats


REFERENCE_TOL = 2e-5


def reference_rows(rows_list, n):
    """`n` real rows of recorded batch files (`onnx_parity.Rows`): whole recorded batches spread over the
    files (every k-th batch first, then the others; rows with tail tokens or pointer overrides: the batches holding
    overrides first, then those holding tail tokens - a stable order), the last one cut to make `n` -> (Rows, batch_off).
    Every kind of row the self-test must answer that the files hold is in: the first batch (in that order) holding an
    override, one holding a belief row and one holding a tail token come first - before, the override and tail
    batches could fill the `n` rows without a belief row, and write_reference refused a network with an auxiliary
    head (the farm pool's export of delver-bo3-pb2's gen 40, 2026-11-14; docs/az.md 46.9)."""
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
    from .protocol import ROW_BELIEF

    def holds(ri, a, b):
        """(override, belief row, tail token) held by a batch."""
        r = rows_list[ri]
        no_ov, no_tail = lacks(ri, a, b)
        return (not no_ov, r.extra is not None and bool((np.asarray(r.extra[a:b]) == ROW_BELIEF).any()), not no_tail)
    first = []
    for k in range(3):
        if not any(holds(*c)[k] for c in first):
            hit = next((c for c in order if holds(*c)[k]), None)
            if hit is not None and hit not in first:
                first.append(hit)
    order = first + [c for c in order if c not in first]
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
    rows, bo = reference_rows(rows_list, n)
    return write_reference_rows(net, rows, bo, out, info, tol)


def write_reference_rows(net, rows, bo, out, info=None, tol=REFERENCE_TOL):
    """`write_reference` on rows already taken (`reference_rows`: the rows and their batches `bo`) -> its metadata: the
    reference set of `net` on those positions, torch's answers batch by batch. The farm pool keeps a run's positions
    and answers each new checkpoint on them (docs/az.md 53): any network of the architecture answers any rows of its
    encoding, so positions recorded with an older checkpoint are as valid a self-test as new ones."""
    import numpy as np
    from .infer import NetEvaluator
    from .onnx_parity import eager_answer
    from .protocol import ROW_BELIEF, ROW_VALUE
    n = int(rows.B)
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
    ap.add_argument('--attention', default=None, choices=[x for x in ATTENTIONS if x],
                    help='zero-onnx/2: a fused attention node per block (docs/az.md 63; default: as always)')
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
    meta = export(m, a.out, a.packing, info, make_portable=not a.no_portable, attention=a.attention)
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
