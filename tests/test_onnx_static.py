"""The static zero-onnx/2 graph (docs/az.md 67: `zero.onnx_export.ZeroOnnx2Static`, `export(..., fast='static')`, graph
'static'): the default graph's arithmetic on rows padded to the length of its input `t_cap` - no data-dependent shape, for
the GPU. Its answers are torch's (`zero.onnx_parity`, eager and forward_static) within 2e-5 on the toy's layouts and a
richer one (fed `t_cap` from the metadata's `presence` and `t_align`, as a GPU worker feeds it), the same for any
`t_cap` long enough; a row whose tokens pass T is refused; no NonZero and no value read back anywhere in the graph.
Skipped without onnx / onnxscript / onnxruntime."""
import json

import numpy as np
import pytest
import torch

pytest.importorskip('onnx')
pytest.importorskip('onnxscript')
pytest.importorskip('onnxruntime')

from zero import onnx_export as OX, onnx_parity as OP  # noqa: E402

from test_onnx2 import QUIET, _items_only, gate_rows, kinds_of  # noqa: E402
from test_onnx_fast import READER, RICH, RICH_ID, rich_batch, rich_net  # noqa: E402
from test_tail_net import OFFERS, TOY_GATED, TOY_TAIL_TOKENS_BASE, rand_net, toy_batch  # noqa: E402


def export(net, tmp_path, name='s', attention=None):
    path = str(tmp_path / f'{name}.onnx')
    meta = OX.export(net, path, 'packed', {'checkpoint': 'toy'}, fast='static', attention=attention)
    return path, meta


def gate(net, path, rows_list, sizes):
    ok, res = OP.run_gate(net, {'packed': path}, rows_list, sizes=sizes, threads=1, **QUIET)
    assert ok, [r for r in res if not r['pass']]
    assert max(r['max'] for r in res) < 2e-5
    return res


@pytest.mark.parametrize('case', ['tails', 'gated0', 'gated-open', 'blend'])
def test_static_parity_on_the_toy_layouts(tmp_path, case):
    """test_onnx2's parity cases through the static graph: within 2e-5 of eager and forward_static, the same bits run
    twice, batches of 1, 7, 64 and 150 rows, NaN in every absent head slot, the rows with tails alone, 64-row chunks."""
    lay, gates, p_offer = {'tails': (TOY_TAIL_TOKENS_BASE, None, 0.0), 'gated0': (TOY_GATED, [0.0], 0.0),
                           'gated-open': (TOY_GATED, [0.37], 0.0), 'blend': (OFFERS, [0.6, 0.45], 0.6)}[case]
    net = rand_net(lay, seed=3)
    if gates is not None:
        with torch.no_grad():
            net.gate.copy_(torch.tensor(gates))
    path, meta = export(net, tmp_path)
    assert meta['static'] == '1' and meta['attention'] == 'mha' and meta['t_align'] == str(OX.T_ALIGN)
    assert json.loads(meta['presence']) == [int(i) for i in net.present_index()]
    B = 150
    d = toy_batch(B, seed=11, p_offer=p_offer)
    if case == 'tails':
        d = {**d, **{k: v for k, v in _items_only(d).items()}}
    res = gate(net, path, [gate_rows(net, d, kinds_of(B, 4))], (1, 7, 64, 150))
    assert {r['case'] for r in res} >= {'B1', 'B150', 'B150-nan', 'varlen', 'varlen-nan'}


@pytest.mark.parametrize('attention', ['mha', 'matmul'])
def test_static_parity_on_a_rich_layout(tmp_path, attention):
    """Bags, three head segments (one without an identity, one always present), two global ranges, a reader, gated
    tails and overrides; slots near 2^32; batches without tail tokens or overrides - within 2e-5 with each attention."""
    net = rich_net(RICH, seed=4, readers=READER)
    path, meta = export(net, tmp_path, attention=attention)
    assert meta['attention'] == attention
    rows = gate_rows(net, rich_batch(120, 7), kinds_of(120, 8))
    big = gate_rows(net, rich_batch(40, 9, big=True), kinds_of(40, 1))
    none = gate_rows(net, rich_batch(30, 3, p_tail=0.0, p_mark=0.0, p_gen=0.0, p_offer=0.0), kinds_of(30, 2))
    assert none.tail.NT == 0 and none.ov.NO == 0 and int(big.tail.slot.max()) >= (1 << 32) - 2
    gate(net, path, [rows, big, none], (1, 7, 64, 120))


def test_static_identity_actions(tmp_path):
    """Identity actions where two equal items claim one action: the default graph's policy scatter (the last writer)
    and dup_reduce 'max' (the largest claim), within 2e-5 of torch on rows whose claims collide."""
    for reduce in (None, 'max'):
        net = rich_net(RICH_ID, seed=5)
        net.dup_reduce = reduce
        path, _ = export(net, tmp_path, name=f'id-{reduce}')
        gate(net, path, [gate_rows(net, rich_batch(80, 13, dup=True), kinds_of(80, 3))], (1, 80))


def test_static_graph_structure_and_t_cap(tmp_path):
    """No NonZero, no read-back of a value (every size the inputs'); LayerNormalization kept (not the fast graph's
    InstanceNormalization), one fused attention and one Gelu per block; `t_cap` a required input of its own; the same
    answers for any T at least the rows' tokens; a row whose tokens pass T refused (every other row answered as with a
    long enough T)."""
    import onnx
    import onnxruntime as ort
    net = rich_net(RICH, seed=1, readers=READER)
    path, meta = export(net, tmp_path)
    g = onnx.load(path)
    ops = [n.op_type for n in g.graph.node]
    assert 'NonZero' not in ops and ops.count('MultiHeadAttention') == 2 and ops.count('Gelu') == 3
    assert ops.count('LayerNormalization') == 5 and 'InstanceNormalization' not in ops
    assert [i.name for i in g.graph.input][-1] == 't_cap' and not [i for i in g.graph.initializer if i.name == 't_cap']
    assert json.loads(meta['inputs'])[-1] == 't_cap'
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    rows = gate_rows(net, rich_batch(40, 9), kinds_of(40, 5))
    sizes = OP.graph_sizes(sess)
    f = OP.feeds(rows, v2=sizes)
    T = len(f['t_cap'])
    assert T % OX.T_ALIGN == 0
    n = OP.presence_sparse(rows, sizes['presence']).sum(1) + np.diff(f['t_off'])
    assert int(n.max()) <= T < int(n.max()) + OX.T_ALIGN
    names = [o.name for o in sess.get_outputs()]
    base = dict(zip(names, sess.run(None, f)))
    assert not base['refused'].any()
    for t in (int(n.max()), T + 7, 3 * T):
        got = dict(zip(names, sess.run(None, dict(f, t_cap=np.zeros(t, np.int32)))))
        for k in names:
            if k == 'refused':
                assert np.array_equal(got[k], base[k])
            else:
                a, b = base[k], got[k]
                assert np.array_equal(np.isnan(a), np.isnan(b)) and np.nanmax(np.abs(a - b)) <= 2e-6, (t, k)
    short = int(n.max()) - 1
    got = dict(zip(names, sess.run(None, dict(f, t_cap=np.zeros(short, np.int32)))))
    over = n > short
    assert over.any() and np.array_equal(got['refused'] > 0, over)
    for k in ('policy', 'value', 'margin'):
        a, b = base[k][~over], got[k][~over]
        assert np.nanmax(np.abs(a - b)) <= 2e-6, k
    # the static graph's own weight-swap template (graph 'static')
    tdir = tmp_path / 'tpl'
    tdir.mkdir()
    tpath = str(tdir / 'template.onnx')
    tmpl = OX.export_template(net, tpath, graph='static')
    out = str(tmp_path / 'swapped.onnx')
    OX.swap_export(net, tpath, tmpl, out, extra={'checkpoint': 'toy'}, graph='static')
    fresh = str(tmp_path / 'fresh.onnx')
    OX.export(net, fresh, 'packed', {'checkpoint': 'toy'}, fast='static')
    a = ort.InferenceSession(out, providers=['CPUExecutionProvider']).run(None, f)
    b = ort.InferenceSession(fresh, providers=['CPUExecutionProvider']).run(None, f)
    for x, y in zip(a, b):
        assert np.array_equal(x, y, equal_nan=True) if x.dtype.kind == 'f' else np.array_equal(x, y)
