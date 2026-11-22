"""The fast graph at fixed capacities (docs/az.md 74: `zero.onnx_export.ZeroOnnx2FastFixed`, `export(..., fast='fixed')`,
graph 'fast-fixed'): the fast graph's arithmetic with every size an input's - `s_cap` (the present head slots), `m_cap`
(a row's non-zeros in a run of features), `t_cap` (a row's tokens) - for ONNX Runtime's CUDA graphs. Its answers are
torch's (`zero.onnx_parity`, eager and forward_static) within 2e-5 on the toy's layouts and a richer one (fed the
capacities from the metadata, as a GPU worker computes them), the same for any capacities large enough; a batch past a
capacity refused (the rows past T or MR; every row past S); no NonZero and no value read back anywhere in the graph.
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


def export(net, tmp_path, name='x', attention=None):
    path = str(tmp_path / f'{name}.onnx')
    meta = OX.export(net, path, 'packed', {'checkpoint': 'toy'}, fast='fixed', attention=attention)
    return path, meta


def gate(net, path, rows_list, sizes):
    ok, res = OP.run_gate(net, {'packed': path}, rows_list, sizes=sizes, threads=1, **QUIET)
    assert ok, [r for r in res if not r['pass']]
    assert max(r['max'] for r in res) < 2e-5
    return res


@pytest.mark.parametrize('case', ['tails', 'gated0', 'gated-open', 'blend'])
def test_fixed_parity_on_the_toy_layouts(tmp_path, case):
    """test_onnx2's parity cases through the fixed graph: within 2e-5 of eager and forward_static, batches of 1, 7, 64
    and 150 rows, NaN in every absent head slot, the rows with tails alone, 64-row chunks."""
    lay, gates, p_offer = {'tails': (TOY_TAIL_TOKENS_BASE, None, 0.0), 'gated0': (TOY_GATED, [0.0], 0.0),
                           'gated-open': (TOY_GATED, [0.37], 0.0), 'blend': (OFFERS, [0.6, 0.45], 0.6)}[case]
    net = rand_net(lay, seed=3)
    if gates is not None:
        with torch.no_grad():
            net.gate.copy_(torch.tensor(gates))
    path, meta = export(net, tmp_path)
    assert meta['fixed'] == '1' and meta['attention'] == 'mha' and meta['t_align'] == str(OX.T_ALIGN)
    assert json.loads(meta['presence']) == [int(i) for i in net.present_index()]
    assert json.loads(meta['inputs'])[-3:] == list(OX.FIXED_INPUTS)
    B = 150
    d = toy_batch(B, seed=11, p_offer=p_offer)
    if case == 'tails':
        d = {**d, **{k: v for k, v in _items_only(d).items()}}
    res = gate(net, path, [gate_rows(net, d, kinds_of(B, 4))], (1, 7, 64, 150))
    assert {r['case'] for r in res} >= {'B1', 'B150', 'B150-nan', 'varlen', 'varlen-nan'}


@pytest.mark.parametrize('attention', ['mha', 'matmul'])
def test_fixed_parity_on_a_rich_layout(tmp_path, attention):
    """Bags, three head segments (one without an identity, one always present), two global ranges, a reader, gated
    tails and overrides; slots near 2^32; batches without tail tokens or overrides - within 2e-5 with each attention."""
    net = rich_net(RICH, seed=4, readers=READER)
    path, meta = export(net, tmp_path, attention=attention)
    assert meta['attention'] == attention
    rows = gate_rows(net, rich_batch(120, 7), kinds_of(120, 8))
    big = gate_rows(net, rich_batch(40, 9, big=True), kinds_of(40, 1))
    none = gate_rows(net, rich_batch(30, 3, p_tail=0.0, p_mark=0.0, p_gen=0.0, p_offer=0.0), kinds_of(30, 2))
    gate(net, path, [rows, big, none], (1, 7, 64, 120))


def test_fixed_identity_actions(tmp_path):
    """Identity actions where two equal items claim one action (the last writer; dup_reduce 'max'), within 2e-5."""
    for reduce in (None, 'max'):
        net = rich_net(RICH_ID, seed=5)
        net.dup_reduce = reduce
        path, _ = export(net, tmp_path, name=f'id-{reduce}')
        gate(net, path, [gate_rows(net, rich_batch(80, 13, dup=True), kinds_of(80, 3))], (1, 80))


def test_fixed_graph_structure_and_capacities(tmp_path):
    """No NonZero and no read-back of a value; the fast graph's InstanceNormalization, Gelu and fused attention; the
    capacities required inputs of their own, no t_align; the same answers (to 2e-6) for any capacities at least the
    batch's; a batch past a capacity refused where it is (rows past T or MR; every row past S), the other rows answered
    as before; its own weight-swap template."""
    import onnx
    import onnxruntime as ort
    net = rich_net(RICH, seed=1, readers=READER)
    path, meta = export(net, tmp_path)
    g = onnx.load(path)
    ops = [n.op_type for n in g.graph.node]
    assert 'NonZero' not in ops and ops.count('MultiHeadAttention') == 2 and ops.count('Gelu') == 3
    assert 'InstanceNormalization' in ops
    names_in = [i.name for i in g.graph.input]
    assert names_in[-3:] == list(OX.FIXED_INPUTS) and 't_align' not in names_in
    assert json.loads(meta['runs'])
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    rows = gate_rows(net, rich_batch(40, 9), kinds_of(40, 5))
    sizes = OP.graph_sizes(sess)
    assert sizes['fixed'] and sizes['runs'] == json.loads(meta['runs'])
    f = OP.feeds(rows, v2=sizes)
    S, MR, T = (len(f[k]) for k in OX.FIXED_INPUTS)
    names = [o.name for o in sess.get_outputs()]
    base = dict(zip(names, sess.run(None, f)))
    assert not base['refused'].any()

    def run(**caps):
        return dict(zip(names, sess.run(None, dict(f, **{k: np.zeros(c, np.int32) for k, c in caps.items()}))))

    def same(got, keep=None):
        for k in names:
            a, b = base[k], got[k]
            if keep is not None:
                a, b = a[keep], b[keep]
            if k == 'refused':
                assert np.array_equal(a, b)
            else:
                assert np.array_equal(np.isnan(a), np.isnan(b)) and np.nanmax(np.abs(a - b)) <= 2e-6, k

    for caps in ((S + 5, MR, T), (S, MR + 9, T), (S, MR, T + 7), (2 * S, 3 * MR, 3 * T)):
        same(run(**dict(zip(OX.FIXED_INPUTS, caps))))
    # past T: those rows refused, the rest as before
    n = OP.presence_sparse(rows, sizes['presence']).sum(1) + np.diff(f['t_off'])
    got = run(t_cap=int(n.max()) - 1)
    over = n > int(n.max()) - 1
    assert over.any() and np.array_equal(got['refused'] > 0, over)
    same(got, ~over)
    # past MR: those rows refused, the rest as before
    if MR > 1:
        got = run(m_cap=MR - 1)
        bad = got['refused'] > 0
        assert bad.any() and not bad.all()
        same(got, ~bad)
    # past S: every row refused
    assert (run(s_cap=S - 1)['refused'] > 0).all()
    # its own weight-swap template (graph 'fast-fixed')
    tdir = tmp_path / 'tpl'
    tdir.mkdir()
    tpath = str(tdir / 'template.onnx')
    tmpl = OX.export_template(net, tpath, graph='fast-fixed')
    out = str(tmp_path / 'swapped.onnx')
    OX.swap_export(net, tpath, tmpl, out, extra={'checkpoint': 'toy'}, graph='fast-fixed')
    fresh = str(tmp_path / 'fresh.onnx')
    OX.export(net, fresh, 'packed', {'checkpoint': 'toy'}, fast='fixed')
    a = ort.InferenceSession(out, providers=['CPUExecutionProvider']).run(None, f)
    b = ort.InferenceSession(fresh, providers=['CPUExecutionProvider']).run(None, f)
    for x, y in zip(a, b):
        assert np.array_equal(x, y, equal_nan=True) if x.dtype.kind == 'f' else np.array_equal(x, y)
