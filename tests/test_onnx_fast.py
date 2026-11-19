"""The fast zero-onnx/2 graph (docs/az.md 63: `zero.onnx_export.ZeroOnnx2Fast`, `export(..., fast=True)`, the run setting
`pool_export`): TokenNet's own arithmetic behind the same interface. Its answers are torch's (`zero.onnx_parity`, eager
and forward_static) within 2e-5, the same bits run twice, on the toy's layouts (tails with identities, gated segments
shut and open, an override's blend, slots near 2^32, empty tails and overrides) and on a richer layout (bags, several
head segments, an always-present one, one without an identity, globals in two ranges, a reader) with each attention
form; identity actions take the default graph's policy scatter (its refusals: test_onnx2's, parametrized); its interface is the default graph's, float32 only, every scatter unique, int32 integers; the weight-swap
template takes it (`graph='fast'`). Skipped without onnx / onnxscript / onnxruntime."""
import json

import numpy as np
import pytest
import torch

pytest.importorskip('onnx')
pytest.importorskip('onnxscript')
pytest.importorskip('onnxruntime')

from zero import onnx_export as OX, onnx_parity as OP  # noqa: E402
from zero.net import TokenNet  # noqa: E402

from test_onnx2 import QUIET, _items_only, gate_rows, kinds_of  # noqa: E402
from test_tail_net import F, OFFERS, TOY_GATED, TOY_TAIL_TOKENS_BASE, rand_net, toy_batch  # noqa: E402

# a richer layout on top of the toy's 72 features: a bag group (2 bags of the 16 identities), a head segment without an
# identity (3 slots, a present flag, static actions of their own) and an always-present one (no flag) after the toy's
# segments, and four more globals at the end (gidx in two ranges); a reader of the bags and the first globals
F2, A2 = F + 32 + 15 + 4 + 4, 32
RICH = dict(OFFERS, segments=OFFERS['segments'] + [
    {"name": "slots2", "offset": F + 32, "count": 3, "width": 5, "present": 0, "id": None, "actions": [[16, 1]]},
    {"name": "ctx", "offset": F + 47, "count": 1, "width": 4, "present": None, "id": None, "actions": []}],
    bags=[{"name": "deck", "offset": F, "count": 2, "width": 16}],
    globals=[{"name": "globals", "offset": 0, "count": 8}, {"name": "more", "offset": F + 51, "count": 4}])
# identity actions on the items: two equal items claim one action (the policy's last writer, `_policy_full`)
RICH_ID = dict(RICH, segments=[dict(RICH['segments'][0], id_actions=19)] + RICH['segments'][1:])
READER = [{'target': 'a', 'width': 5, 'loss': 'multinomial', 'inputs': [[0, 8], [F, 32]], 'hidden': 8, 'enter': 'trunk',
           'belief': True}]


def rich_net(layout, seed=0, readers=None, gates=(0.6, 0.45)):
    torch.manual_seed(seed)
    m = TokenNet(F2, A2, layout, d=16, layers=2, heads=2, ff=32, head=16, aux=[['a', 5, 'multinomial']], game='toy',
                 readers=readers).eval()
    with torch.no_grad():
        for lin in [m.gen, m.ptr, m.val[-1], m.mar[-1], m.aux_head[2]]:
            lin.weight.normal_(0, 0.3)
            lin.bias.normal_(0, 0.3)
        for r in (m.readers or []):
            for lin in (r.mlp[2], r.proj):
                lin.weight.normal_(0, 0.3)
                lin.bias.normal_(0, 0.3)
        m.gate.copy_(torch.tensor(gates))
    return m


def rich_batch(B, seed, dup=False, **kw):
    """toy_batch's rows (`kw` its options; offers with p 0.5 by default) widened to F2: bag counts (1-3 of a few
    identities per bag), the slots2 slots (each present with p 0.5: the flag and 4 floats), ctx's 4 floats, the 4
    globals at the end. `dup`: equal items (identity actions claimed twice)."""
    d = toy_batch(B, seed=seed, **dict({'p_offer': 0.5}, **kw))
    rng = np.random.default_rng(seed + 100)
    x = np.zeros((B, F2), np.float32)
    x[:, :F] = d['x']
    if dup:
        for s in range(1, 4):                       # each item slot's identity copied from slot 0 now and then
            o, o0 = 8 + 16 * s, 8
            r = np.flatnonzero((x[:, o] > 0.5) & (x[:, o0] > 0.5) & (rng.random(B) < 0.5))
            x[np.ix_(r, np.arange(o + 6, o + 16))] = x[np.ix_(r, np.arange(o0 + 6, o0 + 16))]
    for b in range(B):
        for g in range(2):
            ids = rng.choice(16, size=int(rng.integers(0, 6)), replace=False)
            x[b, F + 16 * g + ids] = rng.integers(1, 4, len(ids))
    pres = rng.random((B, 3)) < 0.5
    for s in range(3):
        o = F + 32 + 5 * s
        r = np.flatnonzero(pres[:, s])
        x[r, o] = 1
        x[r, o + 1:o + 5] = rng.normal(size=(len(r), 4))
    x[:, F + 47:F + 51] = rng.normal(size=(B, 4))
    x[:, F + 51:F + 55] = rng.normal(size=(B, 4)) * (rng.random((B, 4)) < 0.5)
    return dict(d, x=x)


def export(net, tmp_path, name='m', attention=None):
    path = str(tmp_path / f'{name}.onnx')
    meta = OX.export(net, path, 'packed', {'checkpoint': 'toy'}, fast=True, attention=attention)
    return path, meta


def gate(net, path, rows_list, sizes):
    ok, res = OP.run_gate(net, {'packed': path}, rows_list, sizes=sizes, threads=1, **QUIET)
    assert ok, [r for r in res if not r['pass']]
    assert max(r['max'] for r in res) < 2e-5
    return res


@pytest.mark.parametrize('case', ['tails', 'gated0', 'gated-open', 'blend'])
def test_fast_parity_on_the_toy_layouts(tmp_path, case):
    """test_onnx2's parity cases through the fast graph: ORT's answers are eager's and forward_static's within 2e-5,
    bit for bit run twice, on batches of 1, 7, 64 and 150 rows (leaf, value-only and belief rows), NaN in every absent
    head slot, the rows with tails alone, the worker's 64-row chunks."""
    lay, gates, p_offer = {'tails': (TOY_TAIL_TOKENS_BASE, None, 0.0), 'gated0': (TOY_GATED, [0.0], 0.0),
                           'gated-open': (TOY_GATED, [0.37], 0.0), 'blend': (OFFERS, [0.6, 0.45], 0.6)}[case]
    net = rand_net(lay, seed=3)
    if gates is not None:
        with torch.no_grad():
            net.gate.copy_(torch.tensor(gates))
    path, meta = export(net, tmp_path)
    assert meta['fast'] == '1' and meta['attention'] == 'mha' and meta['opset'] == '17'
    B = 150
    d = toy_batch(B, seed=11, p_offer=p_offer)
    if case == 'tails':
        d = {**d, **{k: v for k, v in _items_only(d).items()}}
    res = gate(net, path, [gate_rows(net, d, kinds_of(B, 4))], (1, 7, 64, 150))
    assert {r['case'] for r in res} >= {'B1', 'B150', 'B150-nan', 'varlen', 'varlen-nan'}
    v = [r for r in res if r['case'] == 'varlen'][0]
    assert v['tail_tokens'] > 50 and v['overrides'] > 20, v


@pytest.mark.parametrize('attention', ['attention23', 'mha', 'matmul'])
def test_fast_parity_on_a_rich_layout(tmp_path, attention):
    """Bags, three head segments (one without an identity, one always present), globals in two ranges, a reader into
    the global token, gated tail segments and overrides: within 2e-5 of torch with each attention form; slots near 2^32;
    batches without any tail token or override."""
    net = rich_net(RICH, seed=4, readers=READER)
    mod = OX._export_module(net, attention=attention, fast=True)[0]
    assert mod.punique and len(mod.heads) == 3 and len(mod.g_ranges) == 2 and mod.h_lays == [0]
    assert int((mod.t_lay[1:] < 0).sum()) == 4 and len(mod.t_groups) >= 1      # (slots2's and ctx's positions: no identity)
    path, meta = export(net, tmp_path, attention=attention)
    assert meta['attention'] == attention and meta['opset'] == ('23' if attention == 'attention23' else '17')
    rows = gate_rows(net, rich_batch(120, 7), kinds_of(120, 8))
    big = gate_rows(net, rich_batch(40, 9, big=True), kinds_of(40, 1))
    none = gate_rows(net, rich_batch(30, 3, p_tail=0.0, p_mark=0.0, p_gen=0.0, p_offer=0.0), kinds_of(30, 2))
    assert none.tail.NT == 0 and none.ov.NO == 0 and int(big.tail.slot.max()) >= (1 << 32) - 2
    gate(net, path, [rows, big, none], (1, 7, 64, 120))


def test_fast_identity_actions_take_the_policy_scatter(tmp_path):
    """Identity actions (two equal items claim one action): the static writes are not each their own action, so the
    fast graph scatters the policy as the default graph does (`_policy_full`: the last writer); with dup_reduce 'max'
    the largest claim - within 2e-5 of torch either way, on rows where claims collide."""
    for dup_reduce in (None, 'max'):
        net = rich_net(RICH_ID, seed=5)
        net.dup_reduce = dup_reduce
        assert not OX._export_module(net, fast=True)[0].punique
        path, _ = export(net, tmp_path, f'id{dup_reduce}')
        gate(net, path, [gate_rows(net, rich_batch(100, 12, dup=True), kinds_of(100, 6))], (1, 64, 100))


def test_fast_graph_interface_and_structure(tmp_path):
    """The fast graph's inputs, outputs, dtypes and metadata are the default graph's (but `fast`, `attention`, `opset`);
    float32 only (no float64 tensor, cast or constant anywhere); one fused attention node per block, the GELUs one
    node each, no LayerNormalization (InstanceNormalization over one channel); no integer outside int32; every scatter
    unique with a place for each on real rows."""
    import onnx
    T = onnx.TensorProto
    net = rich_net(RICH, seed=1, readers=READER)
    p0 = str(tmp_path / 'default.onnx')
    m0meta = OX.export(net, p0, 'packed', {'checkpoint': 'toy'})
    p1, m1meta = export(net, tmp_path)
    g0, g1 = onnx.load(p0), onnx.load(p1)
    sig = lambda g: [(i.name, i.type.tensor_type.elem_type, len(i.type.tensor_type.shape.dim))  # noqa: E731
                     for i in list(g.graph.input) + list(g.graph.output)]
    assert sig(g0) == sig(g1)
    md0, md1 = ({p.key: p.value for p in g.metadata_props} for g in (g0, g1))
    assert {k: v for k, v in md1.items() if k not in ('fast', 'attention', 'opset')} == {k: v for k, v in md0.items()
                                                                                        if k != 'opset'}
    assert md1['fast'] == '1' and md1['attention'] == 'mha' and md1['opset'] == '17' and m1meta == md1
    assert 'fast' not in m0meta and 'attention' not in m0meta
    doubles = [i.name for i in g1.graph.initializer if i.data_type == T.DOUBLE]
    doubles += [n.name for n in g1.graph.node if n.op_type == 'Cast' and any(a.name == 'to' and a.i == T.DOUBLE
                                                                           for a in n.attribute)]
    vi = onnx.shape_inference.infer_shapes(g1)
    doubles += [v.name for v in vi.graph.value_info if v.type.tensor_type.elem_type == T.DOUBLE]
    assert not doubles, doubles
    ops = [n.op_type for n in g1.graph.node]
    assert ops.count('MultiHeadAttention') == 2 and ops.count('Gelu') == 3 and 'Erf' not in ops
    assert 'LayerNormalization' not in ops and ops.count('InstanceNormalization') == 5
    assert OX.check_int32(g1) > 0
    rows = gate_rows(net, rich_batch(40, 9), kinds_of(40, 5))
    feeds = [OP.feeds(rows, v2=True), OP.feeds(rows.take(np.arange(7)), v2=True)]
    r = OX.int64_max(g1, feeds, threads=1)
    assert r['untyped'] == 0 and 0 < r['max_abs'] < 1 << 31, r
    sw = OX.scatter_writes(g1, feeds, threads=1)
    assert sw['scatters'] > 0 and sw['duplicates'] == 0 and sw['over'] == 0, sw
    assert not any(n.op_type in ('ScatterND', 'ScatterElements') and any(a.name == 'reduction' and a.s not in (b'', b'none')
                                                                         for a in n.attribute) for n in g1.graph.node)
    with pytest.raises(ValueError, match='attention'):
        OX.export(net, str(tmp_path / 'x.onnx'), fast=True, attention='flash')
    with pytest.raises(ValueError, match='zero-onnx/2'):
        OX._export_module(rand_net(dict(TOY_TAIL_TOKENS_BASE, segments=[
            {k: v for k, v in TOY_TAIL_TOKENS_BASE['segments'][0].items() if k != 'tail'}], option_pointers=False,
            max_row_tokens=None)), fast=True)


def test_fast_weight_swap_template(tmp_path):
    """`graph='fast'`: the template (`export_template`) of the fast graph, another checkpoint of the architecture
    swapped in (`swap_export`) - its outputs the fresh fast export's bit for bit; the fingerprint names the graph, so the
    default graph's template is another (`farm_h2h.export_cached`'s cache directory)."""
    import onnxruntime as ort
    a, b = rich_net(RICH, seed=2, readers=READER), rich_net(RICH, seed=3, readers=READER, gates=(0.2, 0.9))
    fp_fast, fields = OX.export_fingerprint(a, graph='fast')
    assert fields['graph'] == 'fast' and fp_fast != OX.export_fingerprint(a)[0] and 'graph' not in OX.export_fingerprint(a)[1]
    tpl = str(tmp_path / 'template.onnx')
    t = OX.export_template(a, tpl, graph='fast')
    assert t['weights']
    sw, fresh = str(tmp_path / 'swapped.onnx'), str(tmp_path / 'fresh.onnx')
    meta = OX.swap_export(b, tpl, t, sw, 'packed', {'checkpoint': 'b'}, graph='fast')
    OX.export(b, fresh, 'packed', {'checkpoint': 'b'}, fast=True)
    assert meta['opset'] == '17' and meta['fast'] == '1'
    rows = gate_rows(b, rich_batch(60, 21), kinds_of(60, 3))
    f = OP.feeds(rows, v2=True)
    o1 = ort.InferenceSession(sw, providers=['CPUExecutionProvider']).run(None, f)
    o2 = ort.InferenceSession(fresh, providers=['CPUExecutionProvider']).run(None, f)
    for u, v in zip(o1, o2):
        assert np.array_equal(u, v)
    with pytest.raises(ValueError, match='fingerprint'):
        OX.swap_export(b, tpl, t, sw, 'packed', {'checkpoint': 'b'})
    assert json.loads(json.dumps(OX.GRAPHS)) and OX.graph_options('fast') == ('mha', True)
    with pytest.raises(ValueError, match='graph'):
        OX.graph_options('fastest')
