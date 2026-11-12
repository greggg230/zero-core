"""zero-onnx/1 (docs/az.md 41): every architecture exported to one ONNX graph that takes the driver's sparse
batch (`zero.onnx_export`), run by ONNX Runtime's CPU execution provider, answers what torch answers
(`zero.onnx_parity`): token networks in each packing (plain and id_actions layouts, NaN in absent slots,
junk in the host's padding, batches of 1 row), readers and a widened auxiliary head, MLP and conv
networks, a network without the head; real toy games through the graph; the in-graph densify; the
metadata. Skipped without onnx / onnxscript / onnxruntime (they are not run dependencies)."""
import json

import numpy as np
import pytest
import torch

pytest.importorskip('onnx')
pytest.importorskip('onnxscript')
pytest.importorskip('onnxruntime')

from zero import games as G, onnx_export as OX, onnx_parity as OP  # noqa: E402
from zero.infer import NetEvaluator, drive  # noqa: E402
from zero.net import widen_aux  # noqa: E402
from zero.protocol import ROW_BELIEF, ROW_LEAF, ROW_VALUE, belief_rows, densify, validate_evaluation  # noqa: E402

from test_net_data import TOK_A, TOK_F, TOK_LAYOUT, TOK_LAYOUT_ID, tok_x  # noqa: E402
from test_reader import TOY_AUX, TOY_READER, nets, shake, toy_x  # noqa: E402
from test_token_graphs import net as tok_net  # noqa: E402

QUIET = dict(log=lambda s: None)


def rows_of(x, npl, kinds, A, seed=0):
    """Rows (onnx_parity.Rows) of dense rows x: leaf rows with 2-5 legal actions, value-only and belief rows
    with none."""
    rng = np.random.default_rng(seed)
    cnt = np.where(kinds == ROW_LEAF, rng.integers(2, 6, len(x)), 0)
    off = np.concatenate([[0], np.cumsum(cnt)])
    act = np.concatenate([np.sort(rng.choice(A, c, replace=False)) for c in cnt] + [np.zeros(0, np.int64)]).astype(np.int32)
    return OP.Rows.from_dense(np.ascontiguousarray(x, np.float32), npl, off, act, kinds)


def kinds_of(B, seed, belief=True):
    rng = np.random.default_rng(seed)
    p = [0.6, 0.3, 0.1] if belief else [0.7, 0.3, 0.0]
    k = rng.choice([ROW_LEAF, ROW_VALUE, ROW_BELIEF], B, p=p).astype(np.uint8)
    k[0] = ROW_LEAF
    return k


def test_densify_matches_protocol():
    """The graph's densify (a scatter-add of the row ends and a running sum: no data-dependent shape) is
    protocol.densify: empty rows anywhere (first, last, consecutive), one row."""
    rng = np.random.default_rng(0)
    for B, F in ((1, 7), (6, 11), (40, 50)):
        cnt = rng.integers(0, 4, B)
        cnt[0] = 0
        cnt[-1] = 0 if B > 2 else cnt[-1]
        off = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
        idx = np.concatenate([np.sort(rng.choice(F, c, replace=False)) for c in cnt] + [np.zeros(0, np.int64)])
        if not len(idx):
            continue
        val = rng.normal(size=len(idx)).astype(np.float32)
        got = OX.densify(torch.from_numpy(off), torch.from_numpy(idx.astype(np.int32)), torch.from_numpy(val), F)
        assert np.array_equal(got.numpy(), densify(off, idx, val, F))


def test_densify_in_onnx_runtime(tmp_path):
    """The densify ONNX Runtime runs (its scatter-add of the row ends meets duplicate indices at empty rows):
    protocol.densify's rows, every time, on 4 threads."""
    import onnxruntime as ort

    class D(torch.nn.Module):
        def forward(self, x_ptr, x_idx, x_val):
            return OX.densify(x_ptr, x_idx, x_val, 50)
    B = torch.export.Dim('B', min=1)
    N = torch.export.Dim('nnz', min=1)
    ex = (torch.tensor([0, 2, 2, 3]), torch.tensor([1, 4, 7], dtype=torch.int32), torch.ones(3))
    path = str(tmp_path / 'd.onnx')
    torch.onnx.export(D(), ex, path, dynamo=True, input_names=['x_ptr', 'x_idx', 'x_val'],
                      dynamic_shapes={'x_ptr': {0: B + 1}, 'x_idx': {0: N}, 'x_val': {0: N}}, verbose=False)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(1)
    for B_ in (1, 9, 300):
        cnt = rng.integers(0, 4, B_) * (rng.random(B_) < 0.6)
        cnt[-1] = max(cnt[-1], 1)
        off = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
        idx = np.concatenate([np.sort(rng.choice(50, c, replace=False)) for c in cnt]).astype(np.int32)
        val = rng.normal(size=len(idx)).astype(np.float32)
        for _ in range(3):
            assert np.array_equal(sess.run(None, {'x_ptr': off, 'x_idx': idx, 'x_val': val})[0], densify(off, idx, val, 50))


@pytest.mark.parametrize('layout', ['plain', 'id_actions'])
@pytest.mark.parametrize('packing', OX.PACKINGS)
def test_token_network_parity(tmp_path, layout, packing):
    """A token network (auxiliary head, random output layers) in each packing: ORT's answers are forward_static's
    and eager's within 2e-5 on batches of 1, 7, 64 and 150 rows with leaf, value-only and belief rows, as they
    are and with NaN in every absent slot (and junk in the host's padding)."""
    m = tok_net(TOK_LAYOUT if layout == 'plain' else TOK_LAYOUT_ID)
    path = str(tmp_path / f'{packing}.onnx')
    OX.export(m, path, packing, {'checkpoint': 'toy'})
    B = 150
    rows = rows_of(tok_x(B, 3).numpy(), np.random.default_rng(1).integers(2, 5, B), kinds_of(B, 2), TOK_A)
    ok, res = OP.run_gate(m, {packing: path}, [rows], sizes=(1, 7, 64, 150), threads=1, **QUIET)
    assert ok, [r for r in res if not r['pass']]
    assert {r['case'] for r in res} >= {'B1', 'B1-nan', 'B150', 'B150-nan'}
    assert max(r['max'] for r in res) < 2e-5 and all(r['belief_rows'] >= 0 for r in res)


@pytest.mark.parametrize('arch', ['tokens', 'mlp', 'grid'])
def test_readers_widened_head_mlp_and_grid(tmp_path, arch):
    """Every architecture with a reader (its prediction in the trunk; declared `belief`: it answers its target)
    and an auxiliary head widened by a poisson target (aux_parts, docs/az.md 37): ORT's answers within 2e-5 of
    torch's on real toy positions."""
    reader = [dict(TOY_READER[0], belief=True)]
    m = nets(aux=TOY_AUX, readers=reader)[arch]
    # (outputs of order 1, as a trained network's: the tolerance is absolute, and a conv tower shaken at 0.3
    # gives margins of ~170, whose float32 rounding alone is ~1e-4 - 7e-7 relative, torch and ORT alike)
    shake(m, 0.05 if arch == 'grid' else 0.3)
    m = widen_aux(m, TOY_AUX + [['rate', 3, 'poisson']]).eval()
    shake(m, 0.01, 'aux_more')                     # (rates of a few: exp turns float rounding into large absolute errors)
    assert m.config['aux_parts'] == [['next_card'], ['rate']]
    path = str(tmp_path / f'{arch}.onnx')
    meta = OX.export(m, path, 'mask', {'checkpoint': 'toy'})
    assert json.loads(meta['aux_layout']) == [['next_card', 6, 'multinomial'], ['rate', 3, 'poisson']]
    assert meta['aux_width'] == '9' and meta['packing'] == ('mask' if arch == 'tokens' else 'none')
    x = toy_x(96, 4)
    info = G.info('toy')
    rows = rows_of(x, np.full(len(x), 2), kinds_of(len(x), 5), info['num_actions'])
    ok, res = OP.run_gate(m, {'mask': path}, [rows], sizes=(1, 7, 96), threads=1, **QUIET)
    assert ok, [r for r in res if not r['pass']]


def test_without_auxiliary_head(tmp_path):
    """A network without the head: `aux` is [B, 0] (W = 0), metadata aux_width 0."""
    import onnxruntime as ort
    m = tok_net(TOK_LAYOUT, aux=None)
    path = str(tmp_path / 'm.onnx')
    meta = OX.export(m, path, 'compact')
    assert meta['aux_width'] == '0' and meta['aux_layout'] == '[]'
    rows = rows_of(tok_x(9, 1).numpy(), np.full(9, 3), kinds_of(9, 1, belief=False), TOK_A)
    outs = ort.InferenceSession(path, providers=['CPUExecutionProvider']).run(None, OP.feeds(rows))
    assert outs[3].shape == (9, 0) and outs[0].shape == (9, TOK_A) and outs[1].shape == outs[2].shape == (9, 8)
    ok, res = OP.run_gate(m, {'compact': path}, [rows], sizes=(9,), threads=1, **QUIET)
    assert ok


def test_interface_and_metadata(tmp_path):
    """zero-onnx/1's names, dtypes and dynamic dimensions; the metadata a worker reads."""
    import onnx
    m = tok_net(TOK_LAYOUT)
    path = str(tmp_path / 'm.onnx')
    OX.export(m, path, 'mask', {'checkpoint': 'iter_0007.pt', 'checkpoint_sha256': 'ab' * 32})
    p = onnx.load(path)
    T = onnx.TensorProto
    ins = {i.name: (i.type.tensor_type.elem_type, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim])
           for i in p.graph.input}
    outs = {o.name: (o.type.tensor_type.elem_type, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim])
            for o in p.graph.output}
    assert list(ins) == list(OX.INPUTS) and list(outs) == list(OX.OUTPUTS)
    assert [ins[k][0] for k in OX.INPUTS] == [T.INT64, T.INT32, T.FLOAT, T.INT64, T.BOOL]
    assert all(outs[k][0] == T.FLOAT for k in OX.OUTPUTS)
    assert ins['x_idx'][1] == ins['x_val'][1] and ins['n_players'][1] == ins['belief'][1] == [outs['policy'][1][0]]
    assert outs['policy'][1][1] == TOK_A and outs['value'][1][1] == outs['margin'][1][1] == 8 and outs['aux'][1][1] == 8
    meta = {e.key: e.value for e in p.metadata_props}
    assert meta['zero_onnx'] == '1' and meta['protocol'] == 'zero-onnx/1' and meta['game'] == 'toy'
    assert int(meta['features']) == TOK_F and int(meta['actions']) == TOK_A and meta['checkpoint'] == 'iter_0007.pt'
    assert meta['checkpoint_sha256'] == 'ab' * 32 and int(meta['opset']) >= 18
    assert len(p.graph.initializer) and not any(t.data_location == T.EXTERNAL for t in p.graph.initializer)


def test_toy_games_through_the_graph(tmp_path):
    """Real driver batches (sparse; leaf, value-only and root belief rows) answered by the graph through the
    worker's generic step: every answer valid and within 2e-5 of the eager evaluator's, the games played to
    the end."""
    import onnxruntime as ort
    info = G.info('toy')
    torch.manual_seed(7)
    m = nets(aux=TOY_AUX)['tokens']
    shake(m, 0.3)
    path = str(tmp_path / 'm.onnx')
    OX.export(m, path, 'compact')
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    ev = NetEvaluator(m, 'cpu')
    seat = G.nn_seat('m', 8, selfplay=True, sims_fast=4, p_full=0.5, leaf_value_view='root', view_samples=0.5,
                     belief_tilt=1.0)
    drv = G.Driver('toy', [[seat, seat]] * 16, concurrency=16, threads=2, groups=1, seed=5, record=True, sparse=True)
    seen = {'batches': 0, 'belief': 0, 'value': 0, 'worst': 0.0}

    class Graph:
        def __call__(self, lb):
            assert lb.sparse
            rows = OP.Rows(lb.a['xs_off'], lb.a['xs_idx'], lb.a['xs_val'], lb.a['n_players'], lb.opt_off,
                           lb.opt_action, lb.a['extra'], info['num_features'])
            got = OP.answer(sess.run(None, OP.feeds(rows)), rows)
            want = ev(lb)
            seen['worst'] = max([seen['worst']] + OP.diffs(got, want))
            validate_evaluation(lb, *got, aux_width=6)
            seen['batches'] += 1
            seen['belief'] += len(belief_rows(lb.a['extra']))
            seen['value'] += int((lb.a['extra'] == ROW_VALUE).sum())
            return got
    drive(drv, {'m': Graph()})
    res = drv.take_results()
    assert len(res) == 16 and all(r['over'] for r in res)
    assert seen['batches'] > 0 and seen['belief'] > 0 and seen['value'] > 0 and seen['worst'] < 2e-5
