"""The probe's graph variants for a diagnostic worker build (zero-farm PROTOCOL.md 8 `probe-providers`;
crates/worker/src/probe_variants.rs): one archive (`variants.npz`) that the worker embeds when built with
`ZERO_PROBE_VARIANTS=<it>` (crates/worker/build.rs). It may hold real networks, so it is never committed.

    PYTHONPATH=python python tools/onnx_probe_variants.py OUT.npz [--net NAME=MODEL.onnx,rows=REF.npz[,static=1]
        [,group=G][,providers=cpu+dml][,budget=S][,note=TEXT]] ...

Groups, in the order the worker runs them (most diagnostic value first):
* `copy`: micro graphs, each testing one way an execution provider can mishandle a graph output (an output
  that aliases an input, a constant or constant-foldable output, a broadcast / transposed / reshaped view, an
  empty output, an empty batch, a scalar, int64 / int32 / bool outputs, an output that feeds another). Written
  for the AMD GPU provider's failure (Windows ML's AMDGPUExecutionProvider 1.8.65 / MIGraphX:
  `hipMemcpyWithStream(output_data, gpu_resource.data(), resource_shape.bytes(), DeviceToDevice)` = "invalid
  argument" - the copy ONNX Runtime's MIGraphX provider makes for the outputs MIGraphX did not bind as
  `#output_N` parameters).
* `ops`: one operator each, as our graphs use them (dynamic shapes, data-dependent ones too).
* `stage`: the built-in probe graph cut after 1/8, 2/8, ... of its nodes (outputs = every tensor crossing the
  cut: the first cut a provider fails on brackets the node), on the probe's own reference batches (1, 7, 64
  and 120 rows: the 1-row batch has no belief row, so the reader runs on zero rows); `stage_full` is the whole
  graph (and the AMD providers' partition dump).
* `--net` graphs: zero-onnx/1 networks timed on real rows (`net` kind: the CPU's answers are the expected ones;
  static = one session per batch size with free-dimension overrides B, `B + 1`, nnz).
Expected outputs of `raw` variants come from ONNX Runtime's CPU provider here.
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper as H
from onnx import numpy_helper as NH

HERE = os.path.dirname(os.path.abspath(__file__))
OPSET = 18
F32, I64, I32, BOOL = TP.FLOAT, TP.INT64, TP.INT32, TP.BOOL


def model(nodes, inputs, outputs, inits=(), name='v'):
    g = H.make_graph(nodes, name, inputs, outputs, initializer=list(inits))
    m = H.make_model(g, opset_imports=[H.make_opsetid('', OPSET)], producer_name='zero-probe-variants')
    m.ir_version = 9
    onnx.checker.check_model(m, full_check=True)
    return m


def vi(name, t, shape):
    return H.make_tensor_value_info(name, t, shape)


def const(name, arr):
    return NH.from_array(np.asarray(arr), name)


def ort_run(m, feeds):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    s = ort.InferenceSession(m.SerializeToString(), so, providers=['CPUExecutionProvider'])
    outs = s.run(None, feeds)
    return {o.name: v for o, v in zip(s.get_outputs(), outs)}


def rng_x(n, w=4, seed=1):
    return np.random.default_rng(seed + n).standard_normal((n, w)).astype(np.float32)


# ------------------------------------------------------------------------------------------------- copy group

def copy_group():
    X = vi('x', F32, ['N', 4])
    two = const('two', np.float32(2.0))
    one = const('one', np.float32(1.0))
    y1 = H.make_node('Mul', ['x', 'two'], ['y1'])
    out = []

    def add(name, nodes, outs, inits=(), inputs=None, cases=None, note=''):
        out.append((name, model(nodes, inputs or [X], outs, inits), cases, note))

    add('copy_control', [H.make_node('Mul', ['x', 'two'], ['t']), H.make_node('Add', ['t', 'one'], ['y'])], [vi('y', F32, ['N', 4])], [two, one],
        note='y = 2x + 1: a computed output (bound as a program output, no copy): the control')
    add('copy_identity', [y1, H.make_node('Identity', ['x'], ['y2'])], [vi('y1', F32, ['N', 4]), vi('y2', F32, ['N', 4])], [two],
        note='y2 = Identity(x): an output aliasing an input')
    add('copy_shape', [y1, H.make_node('Shape', ['x'], ['y2'])], [vi('y1', F32, ['N', 4]), vi('y2', I64, [2])], [two],
        note='y2 = Shape(x): an int64 output MIGraphX folds to a literal at compile time')
    add('copy_shape_arith', [y1, H.make_node('Shape', ['x'], ['s']), H.make_node('Gather', ['s', 'i0'], ['n'], axis=0),
                             H.make_node('Mul', ['n', 'k3'], ['m']), H.make_node('Unsqueeze', ['m', 'ax0'], ['y2'])],
        [vi('y1', F32, ['N', 4]), vi('y2', I64, [1])], [two, const('i0', np.int64(0)), const('k3', np.int64(3)), const('ax0', np.array([0], np.int64))],
        note='y2 = [3 N]: shape arithmetic as an output (a CPU node would consume it)')
    add('copy_const', [y1, H.make_node('Constant', [], ['y2'], value=const('c', np.array([1.5, -2.0, 3.25], np.float32)))],
        [vi('y1', F32, ['N', 4]), vi('y2', F32, [3])], [two], note='y2 = a constant')
    add('copy_expand_const', [y1, H.make_node('Shape', ['x'], ['s']), H.make_node('Expand', ['c', 's'], ['y2'])],
        [vi('y1', F32, ['N', 4]), vi('y2', F32, ['N', 4])], [two, const('c', np.array([[1.0, 2.0, 3.0, 4.0]], np.float32))],
        note='y2 = Expand(const [1,4], Shape(x)): a broadcast literal')
    add('copy_expand_x', [y1, H.make_node('Slice', ['x', 's0', 's1', 'ax0'], ['r']), H.make_node('Shape', ['x'], ['s']), H.make_node('Expand', ['r', 's'], ['y2'])],
        [vi('y1', F32, ['N', 4]), vi('y2', F32, ['N', 4])], [two, const('s0', np.array([0], np.int64)), const('s1', np.array([1], np.int64)), const('ax0', np.array([0], np.int64))],
        note='y2 = Expand(x[0:1], Shape(x)): a broadcast view of an input')
    add('copy_transpose', [y1, H.make_node('Transpose', ['x'], ['y2'], perm=[1, 0])], [vi('y1', F32, ['N', 4]), vi('y2', F32, [4, 'N'])], [two],
        note='y2 = Transpose(x): a transposed view of an input')
    add('copy_reshape', [y1, H.make_node('Reshape', ['x', 'm1'], ['y2'])], [vi('y1', F32, ['N', 4]), vi('y2', F32, ['M'])], [two, const('m1', np.array([-1], np.int64))],
        note='y2 = Reshape(x, [-1]): a reshaped view of an input')
    add('copy_slice', [y1, H.make_node('Slice', ['x', 's0', 's2', 'ax1'], ['y2'])], [vi('y1', F32, ['N', 4]), vi('y2', F32, ['N', 2])],
        [two, const('s0', np.array([0], np.int64)), const('s2', np.array([2], np.int64)), const('ax1', np.array([1], np.int64))],
        note='y2 = x[:, 0:2]: a strided view of an input')
    add('copy_zero', [y1, H.make_node('Slice', ['x', 's0', 's0b', 'ax1'], ['y2'])], [vi('y1', F32, ['N', 4]), vi('y2', F32, ['N', 0])],
        [two, const('s0', np.array([0], np.int64)), const('s0b', np.array([0], np.int64)), const('ax1', np.array([1], np.int64))],
        note='y2 = x[:, 0:0]: an empty output')
    add('copy_zero_mul', [y1, H.make_node('Slice', ['y1', 's0', 's0b', 'ax1'], ['z']), H.make_node('Mul', ['z', 'two'], ['y2'])],
        [vi('y1', F32, ['N', 4]), vi('y2', F32, ['N', 0])], [two, const('s0', np.array([0], np.int64)), const('s0b', np.array([0], np.int64)), const('ax1', np.array([1], np.int64))],
        note='y2 = 2 * y1[:, 0:0]: an empty computed output')
    add('copy_scalar', [y1, H.make_node('ReduceSum', ['x'], ['y2'], keepdims=0)], [vi('y1', F32, ['N', 4]), vi('y2', F32, [])], [two],
        note='y2 = sum(x): a 0-d output')
    add('copy_int64', [H.make_node('Add', ['xi', 'k1'], ['y'])], [vi('y', I64, ['N'])], [const('k1', np.int64(1))], inputs=[vi('xi', I64, ['N'])],
        note='y = xi + 1 (int64)')
    add('copy_int32_cast', [H.make_node('Cast', ['xi'], ['c'], to=I64), H.make_node('Mul', ['c', 'k2'], ['y'])], [vi('y', I64, ['N'])],
        [const('k2', np.int64(2))], inputs=[vi('xi', I32, ['N'])], note='y = 2 int64(xi int32) (x_idx is int32)')
    add('copy_bool', [y1, H.make_node('Greater', ['x', 'zero'], ['y2'])], [vi('y1', F32, ['N', 4]), vi('y2', BOOL, ['N', 4])], [two, const('zero', np.float32(0.0))],
        note='y2 = x > 0 (bool output)')
    add('copy_where_zeros', [H.make_node('Greater', ['x', 'zero'], ['g']), H.make_node('Shape', ['x'], ['s']),
                             H.make_node('ConstantOfShape', ['s'], ['z'], value=const('v0', np.array([0.0], np.float32))), H.make_node('Where', ['g', 'x', 'z'], ['y'])],
        [vi('y', F32, ['N', 4])], [const('zero', np.float32(0.0))], note='y = where(x > 0, x, zeros_like(x)) (the value output\'s shape)')
    add('copy_chain', [y1, H.make_node('Add', ['y1', 'one'], ['y2'])], [vi('y1', F32, ['N', 4]), vi('y2', F32, ['N', 4])], [two, one],
        note='y1 = 2x and y2 = y1 + 1 both outputs (an output that feeds another node)')
    add('copy_gemm', [H.make_node('Gemm', ['x', 'w', 'b'], ['y'])], [vi('y', F32, ['N', 3])],
        [const('w', np.arange(12, dtype=np.float32).reshape(4, 3) / 10), const('b', np.array([0.5, -0.5, 1.0], np.float32))], note='y = Gemm(x, W, b)')
    res = []
    for name, m, cases, note in out:
        ins = [i for i in m.graph.input]
        cs = []
        for n in ([5, 0] if name in ('copy_control', 'copy_identity', 'copy_zero', 'copy_gemm') else [5]):
            feeds = {}
            for i in ins:
                et = i.type.tensor_type.elem_type
                if et == F32:
                    feeds[i.name] = rng_x(n)
                elif et == I64:
                    feeds[i.name] = np.arange(n, dtype=np.int64) * 3 - 2
                elif et == I32:
                    feeds[i.name] = (np.arange(n, dtype=np.int32) * 7) % 11
            cs.append((f'n{n}', feeds, ort_run(m, feeds), {}))
        res.append(dict(name=name, group='copy', model=m, cases=cs, note=note, tol=1e-6))
    return res


# -------------------------------------------------------------------------------------------------- ops group

def ops_group():
    res = []
    rng = np.random.default_rng(7)

    def add(name, m, feeds_list, note, tol=1e-5):
        cs = [(f'c{k}', f, ort_run(m, f), {}) for k, f in enumerate(feeds_list)]
        res.append(dict(name=name, group='ops', model=m, cases=cs, note=note, tol=tol))

    # NonZero over a flattened mask -> [K]
    m = model([H.make_node('Reshape', ['m', 'm1'], ['f']), H.make_node('NonZero', ['f'], ['nz']), H.make_node('Squeeze', ['nz', 'ax0'], ['y'])],
              [vi('m', BOOL, ['B', 8])], [vi('y', I64, ['K'])], [const('m1', np.array([-1], np.int64)), const('ax0', np.array([0], np.int64))])
    add('op_nonzero', m, [{'m': rng.random((5, 8)) < 0.4}, {'m': np.zeros((3, 8), bool)}], 'NonZero (packed: the present tokens), incl. none present')
    # TopK with k from the data (as _row_tokens: k = the longest row)
    m = model([H.make_node('ReduceMax', ['n', ''], ['mx'], keepdims=1), H.make_node('TopK', ['key', 'mx'], ['v', 'y'], axis=1, largest=0, sorted=1)],
              [vi('key', I64, ['B', 8]), vi('n', I64, ['B'])], [vi('y', I64, ['B', 'T'])])
    key = np.stack([rng.permutation(8) for _ in range(4)]).astype(np.int64)
    add('op_topk', m, [{'key': key, 'n': np.array([3, 5, 2, 4], np.int64)}], 'TopK with a data-dependent k (in-graph token index)')
    # ScatterElements 1-D to unique slots (densify)
    m = model([H.make_node('Shape', ['idx'], ['s']), H.make_node('Add', ['s', 'k4'], ['s4']),
               H.make_node('ConstantOfShape', ['s4'], ['z'], value=const('v0', np.array([0.0], np.float32))),
               H.make_node('ScatterElements', ['z', 'idx', 'upd'], ['y'], axis=0)],
              [vi('idx', I64, ['K']), vi('upd', F32, ['K'])], [vi('y', F32, ['M'])], [const('k4', np.array([4], np.int64))])
    add('op_scatter1d', m, [{'idx': rng.permutation(9)[:5].astype(np.int64), 'upd': rng.standard_normal(5).astype(np.float32)}], 'ScatterElements 1-D, unique slots (densify)')
    # ScatterElements 2-D (the policy scatter): unique columns per row
    m = model([H.make_node('ScatterElements', ['pc', 'idx', 'upd'], ['y'], axis=1)],
              [vi('pc', F32, ['B', 10]), vi('idx', I64, ['B', 'K']), vi('upd', F32, ['B', 'K'])], [vi('y', F32, ['B', 10])])
    idx = np.stack([rng.permutation(10)[:3] for _ in range(4)]).astype(np.int64)
    add('op_scatter2d', m, [{'pc': rng.standard_normal((4, 10)).astype(np.float32), 'idx': idx, 'upd': rng.standard_normal((4, 3)).astype(np.float32)}], 'ScatterElements 2-D (the policy scatter)')
    m = model([H.make_node('GatherElements', ['d', 'idx'], ['y'], axis=1)], [vi('d', F32, ['B', 6]), vi('idx', I64, ['B', 'K'])], [vi('y', F32, ['B', 'K'])])
    add('op_gather_elements', m, [{'d': rng.standard_normal((4, 6)).astype(np.float32), 'idx': rng.integers(0, 6, (4, 3)).astype(np.int64)}], 'GatherElements')
    m = model([H.make_node('GatherND', ['d', 'idx'], ['y'])], [vi('d', F32, ['B', 5, 3]), vi('idx', I64, ['K', 2])], [vi('y', F32, ['K', 3])])
    add('op_gathernd', m, [{'d': rng.standard_normal((4, 5, 3)).astype(np.float32), 'idx': np.stack([rng.integers(0, 4, 6), rng.integers(0, 5, 6)], 1).astype(np.int64)}], 'GatherND')
    m = model([H.make_node('CumSum', ['x', 'ax0s'], ['y'])], [vi('x', I64, ['N'])], [vi('y', I64, ['N'])], [const('ax0s', np.int64(0))])
    add('op_cumsum', m, [{'x': rng.integers(0, 3, 9).astype(np.int64)}], 'CumSum int64')
    m = model([H.make_node('ReduceMax', ['n', ''], ['mx'], keepdims=0), H.make_node('Range', ['z0', 'mx', 'o1'], ['y'])],
              [vi('n', I64, ['B'])], [vi('y', I64, ['T'])], [const('z0', np.int64(0)), const('o1', np.int64(1))])
    add('op_range', m, [{'n': np.array([3, 6, 2], np.int64)}], 'Range with a data-dependent limit')
    m = model([H.make_node('ConstantOfShape', ['s'], ['y'], value=const('v1', np.array([1.5], np.float32)))], [vi('s', I64, [2])], [vi('y', F32, ['A', 'B'])])
    add('op_constant_of_shape', m, [{'s': np.array([3, 4], np.int64)}], 'ConstantOfShape from an input')
    m = model([H.make_node('Expand', ['x', 's'], ['y'])], [vi('x', F32, [1, 4]), vi('s', I64, [2])], [vi('y', F32, ['A', 4])])
    add('op_expand', m, [{'x': rng.standard_normal((1, 4)).astype(np.float32), 's': np.array([5, 4], np.int64)}], 'Expand to an input shape')
    m = model([H.make_node('Pad', ['x', 'pads', 'cv'], ['y'], mode='constant')], [vi('x', F32, ['N', 4])], [vi('y', F32, ['N', 5])],
              [const('pads', np.array([0, 0, 0, 1], np.int64)), const('cv', np.float32(0.0))])
    add('op_pad', m, [{'x': rng_x(5)}], 'Pad (one column: the dummy action)')
    m = model([H.make_node('Greater', ['x', 'zero'], ['g']), H.make_node('Where', ['g', 'zero', 'ninf'], ['b']), H.make_node('Add', ['x', 'b'], ['s']),
               H.make_node('Softmax', ['s'], ['y'], axis=-1)],
              [vi('x', F32, ['N', 6])], [vi('y', F32, ['N', 6])], [const('zero', np.float32(0.0)), const('ninf', np.float32(-np.inf))])
    xx = rng.standard_normal((5, 6)).astype(np.float32)
    xx[:, 0] = -1.0
    add('op_masked_softmax', m, [{'x': xx}], 'Softmax with -inf where masked (attention bias)')
    m = model([H.make_node('LayerNormalization', ['x', 'sc', 'bi'], ['y'], axis=-1, epsilon=1e-5)], [vi('x', F32, ['N', 8])], [vi('y', F32, ['N', 8])],
              [const('sc', rng.standard_normal(8).astype(np.float32)), const('bi', rng.standard_normal(8).astype(np.float32))])
    add('op_layernorm', m, [{'x': rng_x(5, 8)}], 'LayerNormalization')
    m = model([H.make_node('Mul', ['x', 'rs2'], ['a']), H.make_node('Erf', ['a'], ['e']), H.make_node('Add', ['e', 'one'], ['b']),
               H.make_node('Mul', ['x', 'b'], ['c']), H.make_node('Mul', ['c', 'half'], ['y'])],
              [vi('x', F32, ['N', 4])], [vi('y', F32, ['N', 4])], [const('rs2', np.float32(1 / np.sqrt(2))), const('one', np.float32(1.0)), const('half', np.float32(0.5))])
    add('op_gelu', m, [{'x': rng_x(5)}], 'exact GELU (Erf)')
    m = model([H.make_node('Round', ['x'], ['r']), H.make_node('Cast', ['r'], ['y'], to=I64)], [vi('x', F32, ['N', 4])], [vi('y', I64, ['N', 4])])
    add('op_round_cast', m, [{'x': (rng_x(5) * 3).astype(np.float32)}], 'Round + Cast to int64 (identity digits)')
    m = model([H.make_node('Mod', ['x', 'k7'], ['y'], fmod=0)], [vi('x', I64, ['N'])], [vi('y', I64, ['N'])], [const('k7', np.int64(7))])
    add('op_mod', m, [{'x': rng.integers(0, 100, 9).astype(np.int64)}], 'Mod int64')
    m = model([H.make_node('MatMul', ['x', 'w'], ['p']), H.make_node('Squeeze', ['p', 'ax1'], ['y'])], [vi('x', F32, ['N', 4])], [vi('y', F32, ['N'])],
              [const('w', np.arange(4, dtype=np.float32)[:, None]), const('ax1', np.array([1], np.int64))])
    add('op_matmul_col', m, [{'x': rng_x(5)}], 'MatMul with a [k, 1] constant + Squeeze (portable identity decoding)')
    m = model([H.make_node('Gather', ['d', 'idx'], ['y'], axis=0)], [vi('d', F32, ['N', 4]), vi('idx', I64, ['K'])], [vi('y', F32, ['K', 4])])
    add('op_index_select', m, [{'d': rng_x(6), 'idx': np.array([5, 0, 2, 2, 1], np.int64)}], 'Gather rows (index_select)')
    m = model([H.make_node('Greater', ['x', 'zero'], ['g']), H.make_node('Less', ['x', 'one'], ['l']), H.make_node('And', ['g', 'l'], ['a']),
               H.make_node('Not', ['a'], ['y'])], [vi('x', F32, ['N', 4])], [vi('y', BOOL, ['N', 4])], [const('zero', np.float32(0.0)), const('one', np.float32(1.0))])
    add('op_bool_logic', m, [{'x': rng_x(5)}], 'Greater / Less / And / Not')
    return res


# ------------------------------------------------------------------------------------------------ stage group

def ref_batches(ref_path):
    r = np.load(ref_path)
    bo, off = r['batch_off'], r['xs_off']
    out = []
    for k in range(len(bo) - 1):
        a, b = int(bo[k]), int(bo[k + 1])
        lo, hi = int(off[a]), int(off[b])
        out.append((f'b{b - a}', {'x_ptr': (off[a:b + 1] - off[a]).astype(np.int64), 'x_idx': r['xs_idx'][lo:hi].astype(np.int32),
                                   'x_val': r['xs_val'][lo:hi].astype(np.float32), 'n_players': r['n_players'][a:b].astype(np.int64),
                                   'belief': r['extra'][a:b] == 2}))
    return out


def stage_group(probe_path, ref_path, cuts=8):
    full = onnx.load(probe_path)
    full = onnx.shape_inference.infer_shapes(full)
    g = full.graph
    nodes = list(g.node)
    graph_in = {i.name for i in g.input} | {i.name for i in g.initializer}
    batches = ref_batches(ref_path)
    res = []
    for k in range(1, cuts + 1):
        upto = round(len(nodes) * k / cuts)
        made = set()
        for n in nodes[:upto]:
            made.update(o for o in n.output if o)
        if k == cuts:
            outs = [o.name for o in g.output]
        else:
            later = set()
            for n in nodes[upto:]:
                later.update(i for i in n.input if i)
            later.update(o.name for o in g.output)
            outs = sorted(t for t in made & later if t not in graph_in)
        sub = onnx.utils.Extractor(full).extract_model([i.name for i in g.input], outs)
        used = {i for n in sub.graph.node for i in n.input}
        # inputs a cut does not use are dropped by the extractor; feed only the ones it kept
        names = [i.name for i in sub.graph.input]
        sub.opset_import[0].version = full.opset_import[0].version
        cs = []
        for bname, feeds in batches:
            f = {n: feeds[n] for n in names}
            cs.append((bname, f, ort_run(sub, f), {}))
        name = 'stage_full' if k == cuts else f'stage_{k}of{cuts}'
        res.append(dict(name=name, group='stage', model=sub, cases=cs, tol=2e-5, dump=(k == cuts),
                        note=f'the probe graph up to node {upto} of {len(nodes)} ({nodes[upto - 1].op_type} {nodes[upto - 1].name}); {len(outs)} output(s) cross the cut',
                        cut_outputs=outs, unused_inputs=sorted(set(names) - used)))
    return res


# --------------------------------------------------------------------------------------------------- archive

def u8(b):
    return np.frombuffer(b, dtype=np.uint8)


def build(out, extra_nets):
    members = {}
    variants = []
    raw = copy_group() + ops_group() + stage_group(os.path.join(HERE, '..', 'crates', 'worker', 'probe', 'probe.onnx'),
                                                    os.path.join(HERE, '..', 'crates', 'worker', 'probe', 'probe_reference.npz'))
    for v in raw:
        mname = f"m/{v['name']}.onnx"
        members[mname] = u8(v['model'].SerializeToString())
        cases = []
        for cname, feeds, expect, dims in v['cases']:
            ins, exs = {}, {}
            for k, a in feeds.items():
                key = f"c/{v['name']}/{cname}/in.{k}"
                members[key] = np.asarray(a, order="C")
                ins[k] = key
            for k, a in expect.items():
                key = f"c/{v['name']}/{cname}/out.{k}"
                members[key] = np.asarray(a, order="C")
                exs[k] = key
            cases.append({'name': cname, 'inputs': ins, 'expect': exs, 'dims': dims})
        e = {'name': v['name'], 'group': v['group'], 'kind': 'raw', 'model': mname, 'cases': cases, 'tol': v['tol'], 'note': v['note']}
        if v.get('dump'):
            e['dump'] = True
        # which providers: the AMD GPU provider everywhere (the question); the CPU as the archive's own check; DirectML
        # (the fallback, passes elsewhere) and the deprecated MIGraphX provider only where they add something
        if v['group'] == 'copy':
            e['providers'] = ['cpu', 'amdgpu', 'migraphx', 'dml', 'openvino', 'tensorrt_rtx']
        elif v['group'] == 'ops':
            e['providers'] = ['cpu', 'amdgpu', 'dml', 'openvino', 'tensorrt_rtx']
        elif v['name'] == 'stage_full':
            e['providers'] = ['cpu', 'amdgpu', 'migraphx', 'dml', 'openvino', 'tensorrt_rtx']
        else:
            e['providers'] = ['amdgpu', 'openvino', 'tensorrt_rtx']
        if v['group'] == 'stage':
            e['cut_outputs'] = v['cut_outputs']
            e['budget_s'] = 60
        variants.append(e)
        if v['name'] == 'stage_full':
            # the whole graph with less of ONNX Runtime's own rewriting before the providers partition it
            for opt in ('basic', 'disable'):
                variants.append(dict(e, name=f'stage_full_opt{opt}', opt=opt, dump=False, providers=['amdgpu', 'migraphx', 'dml', 'openvino', 'tensorrt_rtx'],
                                     note=f'the whole probe graph, graph optimisation level {opt} (the factory uses all)'))
    rows_seen = {}
    for spec in extra_nets:
        name, rest = spec.split('=', 1)
        parts = rest.split(',')
        opts = dict(p.split('=', 1) for p in parts[1:])
        mpath = parts[0]
        mname = f'm/{name}.onnx'
        members[mname] = u8(open(mpath, 'rb').read())
        rpath = os.path.abspath(opts['rows'])
        if rpath not in rows_seen:
            rows_seen[rpath] = f'r/rows{len(rows_seen)}.npz'
            members[rows_seen[rpath]] = u8(open(rpath, 'rb').read())
        e = {'name': name, 'group': opts.get('group', 'net'), 'kind': 'net', 'model': mname, 'rows': rows_seen[rpath],
             'batches': [int(x) for x in opts.get('batches', '16+64+256+1024').split('+')], 'static': opts.get('static', '0') == '1',
             'tol': float(opts.get('tol', 2e-5)), 'note': opts.get('note', ''), 'budget_s': float(opts.get('budget', 150))}
        if 'providers' in opts:
            e['providers'] = opts['providers'].split('+')
        variants.append(e)
    # the order the worker runs the groups in ("cache": the job's own networks from the agent's cache)
    order = ['copy', 'ops', 'stage', 'cache', 'real', 'probe_net']
    members['manifest'] = u8(json.dumps({'version': 1, 'order': order, 'variants': variants}).encode())
    tmp = out + '.tmp.npz'
    np.savez_compressed(tmp, **members)
    os.replace(tmp, out)
    return variants, os.path.getsize(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('out')
    ap.add_argument('--net', action='append', default=[], help='NAME=MODEL.onnx,rows=REF.npz[,static=1][,group=G][,providers=a+b][,batches=16+64][,budget=S][,note=T]')
    a = ap.parse_args()
    variants, size = build(a.out, a.net)
    groups = {}
    for v in variants:
        groups.setdefault(v['group'], []).append(v['name'])
    print(json.dumps({'out': a.out, 'bytes': size, 'variants': len(variants), 'groups': groups}, indent=1))


if __name__ == '__main__':
    main()
