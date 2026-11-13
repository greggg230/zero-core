"""zero-onnx/2 (docs/az.md 46): a token network with variable-length token sets - tail tokens, gated segments,
pointer overrides - exported to one ONNX graph that takes the driver's batch with its options, tail tokens and
overrides (`zero.onnx_export.ZeroOnnx2`), run by ONNX Runtime's CPU execution provider: its answers are torch's
(`zero.onnx_parity`, eager and forward_static) within 2e-5, the same bits run twice, on the toy's layouts (tails with
identities, a gated tail-only segment at g = 0 and opened, an overridden gated segment's blend, slots near 2^32),
batches without any tail token or override, one row; what the network refuses is refused row by row (`refused`);
real toy_tail driver batches through `OnnxEvaluator`; the reference set with tails and overrides; a network without
them still zero-onnx/1. Skipped without onnx / onnxscript / onnxruntime."""
import json

import numpy as np
import pytest
import torch

pytest.importorskip('onnx')
pytest.importorskip('onnxscript')
pytest.importorskip('onnxruntime')

from zero import games as G, onnx_export as OX, onnx_parity as OP, rows as RW  # noqa: E402
from zero.infer import NetEvaluator, drive  # noqa: E402
from zero.protocol import ROW_BELIEF, ROW_LEAF, ROW_VALUE, belief_rows  # noqa: E402

from test_tail_net import A, F, OFFERS, TOY_GATED, TOY_TAIL_TOKENS, TOY_TAIL_TOKENS_BASE, rand_net, toy_batch  # noqa: E402

QUIET = dict(log=lambda s: None)


def gate_rows(net, d, kinds):
    """onnx_parity.Rows of a toy batch (test_tail_net.toy_batch) with row kinds: a value-only or belief row keeps its
    tail and loses its options (and so its overrides)."""
    B = len(kinds)
    widths = {j: (net.segs[j]['width'], net.segs[j]['count'], k) for k, j in enumerate(net.tails)}
    tail = RW.Tail(d['t_off'], d['t_seg'], d['t_slot'], d['t_val'], widths) if net.tails else None
    ov = RW.Overrides(d['ov_off'], d['ov_opt'], d['ov_tok'])
    leaf = kinds == ROW_LEAF
    oo, oa, vo, vp, vt = [0], [], [0], [], []
    for b in range(B):
        if leaf[b]:
            oa.extend(d['opt_action'][d['opt_off'][b]:d['opt_off'][b + 1]])
            vp.extend(ov.opt[ov.off[b]:ov.off[b + 1]])
            vt.extend(ov.tok[ov.off[b]:ov.off[b + 1]])
        oo.append(len(oa))
        vo.append(len(vp))
    ov = RW.Overrides(np.asarray(vo), np.asarray(vp, np.int32), np.asarray(vt, np.int64)) if net.option_pointers else None
    return OP.Rows.from_dense(d['x'], d['n_players'], np.asarray(oo, np.int64), np.asarray(oa, np.int32), kinds,
                              tail=tail, ov=ov)


def kinds_of(B, seed):
    rng = np.random.default_rng(seed)
    k = rng.choice([ROW_LEAF, ROW_VALUE, ROW_BELIEF], B, p=[0.7, 0.2, 0.1]).astype(np.uint8)
    k[0] = ROW_LEAF
    return k


def export(net, tmp_path, name='m'):
    path = str(tmp_path / f'{name}.onnx')
    meta = OX.export(net, path, 'packed', {'checkpoint': 'toy'})
    return path, meta


@pytest.mark.parametrize('case', ['tails', 'gated0', 'gated-open', 'blend'])
def test_parity_on_rows_with_tails_and_overrides(tmp_path, case):
    """ORT's answers are eager's and forward_static's within 2e-5, bit for bit run twice: tails on items (identities,
    overrides into tail items, generic overrides), a gated tail-only segment (marks) shut (g = 0: its tokens change
    nothing) and opened, a gated segment with a pointer column (offers: an override into it is the blend of the
    generic and the pointer logit, g in (0, 1) and past 1); batches of 1, 7, 64 and 150 rows (leaf, value-only and
    belief rows), NaN in every absent head slot, the rows with tails alone, the worker's 64-row chunks."""
    lay, gates, p_offer = {'tails': (TOY_TAIL_TOKENS_BASE, None, 0.0), 'gated0': (TOY_GATED, [0.0], 0.0),
                           'gated-open': (TOY_GATED, [0.37], 0.0), 'blend': (OFFERS, [0.6, 0.45], 0.6)}[case]
    net = rand_net(lay, seed=3)
    if gates is not None:
        with torch.no_grad():
            net.gate.copy_(torch.tensor(gates))
    path, meta = export(net, tmp_path)
    assert meta['zero_onnx'] == '2' and meta['protocol'] == 'zero-onnx/2'
    B = 150
    d = toy_batch(B, seed=11, p_offer=p_offer)
    if case == 'tails':
        d = {**d, **{k: v for k, v in _items_only(d).items()}}
    rows = gate_rows(net, d, kinds_of(B, 4))
    ok, res = OP.run_gate(net, {'packed': path}, [rows], sizes=(1, 7, 64, 150), threads=1, **QUIET)
    assert ok, [r for r in res if not r['pass']]
    assert {r['case'] for r in res} >= {'B1', 'B150', 'B150-nan', 'varlen', 'varlen-nan'}
    assert any(r['packing'] == 'packed-c64' for r in res)
    v = [r for r in res if r['case'] == 'varlen'][0]
    assert v['tail_tokens'] > 50 and v['overrides'] > 20, v
    if case != 'tails':
        assert v['gated_tokens'] > 10, v
    assert max(r['max'] for r in res) < 2e-5


def _items_only(d):
    """The batch's tail tokens of segment 0 (items) only (a layout without marks)."""
    B = len(d['t_off']) - 1
    vo = np.concatenate([[0], np.cumsum([16 if s == 0 else 4 for s in d['t_seg']])])
    t_off, seg, slot, val = [0], [], [], []
    for b in range(B):
        for j in range(d['t_off'][b], d['t_off'][b + 1]):
            if d['t_seg'][j] == 0:
                seg.append(0)
                slot.append(d['t_slot'][j])
                val.extend(d['t_val'][vo[j]:vo[j + 1]])
        t_off.append(len(seg))
    return {'t_off': np.asarray(t_off, np.int64), 't_seg': np.asarray(seg, np.uint16), 't_slot': np.asarray(slot, np.uint32),
            't_val': np.asarray(val, np.float32)}


def test_slots_near_two_to_the_32_and_batches_without_tails(tmp_path):
    """Tail slots up to 2^32 - 1 (psi's digits exact from the integer), one row, and batches with no tail token and no
    override at all (the graph's empty tail and overrides) and with tails but no override."""
    net = rand_net(TOY_GATED, seed=5)
    with torch.no_grad():
        net.gate.fill_(0.8)
    path, _ = export(net, tmp_path)
    d = toy_batch(40, seed=2, big=True)
    rows = gate_rows(net, d, kinds_of(40, 1))
    plain = toy_batch(30, seed=3, p_tail=0.0, p_mark=0.0, p_gen=0.0)
    none = gate_rows(net, plain, kinds_of(30, 2))
    assert none.tail.NT == 0 and none.ov.NO == 0
    nogen = gate_rows(net, toy_batch(30, seed=4, p_tail=0.0, p_gen=0.0), kinds_of(30, 3))
    assert nogen.tail.NT > 0 and nogen.ov.NO == 0
    ok, res = OP.run_gate(net, {'packed': path}, [rows, none, nogen], sizes=(1, 7, 30), threads=1, **QUIET)
    assert ok, [r for r in res if not r['pass']]
    assert int(rows.tail.slot.max()) >= (1 << 32) - 2


def test_the_graph_refuses_what_the_network_refuses(tmp_path):
    """Inputs torch raises on (R5 / R6) are counted per row in `refused` (the other rows 0, still answered): an
    override of an option out of its row, to an absent head slot, to a tail token the row lacks, a column past its
    segment's, a second override of the same action, a token -2; a tail token of a segment without a tail, at a head
    slot, out of order; tail floats that do not fit the tokens. `answer` raises on any (onnx_parity.Refused)."""
    import onnxruntime as ort
    net = rand_net(OFFERS, seed=7)
    path, _ = export(net, tmp_path)
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    base = gate_rows(net, toy_batch(12, seed=9, p_head=0.5, p_offer=0.5), np.full(12, ROW_LEAF, np.uint8))

    def run(rows):
        out = sess.run(None, OP.feeds(rows, v2=True))
        return out, np.asarray(out[4])
    out, ref = run(base)
    assert not ref.any()
    OP.answer(out, base)
    absent = next(b for b in range(12) if not base.dense()[b, 8 + 16 * 3])      # a row whose head slot 3 is absent

    def with_ov(b, opt, tok):
        o = base.ov
        cnt = o.counts().copy()
        parts = []
        for r in range(12):
            a, e = o.off[r], o.off[r + 1]
            p, t = list(o.opt[a:e]), list(o.tok[a:e])
            if r == b:
                p.append(opt)
                t.append(tok)
                cnt[r] += 1
            parts.append((p, t))
        new = RW.Overrides(np.concatenate([[0], np.cumsum(cnt)]), np.concatenate([np.asarray(p, np.int32) for p, _ in parts]),
                           np.concatenate([np.asarray(t, np.int64) for _, t in parts]))
        return OP.Rows(base.xs_off, base.xs_idx, base.xs_val, base.n_players, base.opt_off, base.opt_action, base.extra,
                       base.F, tail=base.tail, ov=new)
    nopt = int(base.opt_off[absent + 1] - base.opt_off[absent])
    bad = [with_ov(absent, nopt, -1),                                        # an option out of its row
           with_ov(absent, 0, int(RW.pack_tok(0, 0, 3))),                     # an absent head slot
           with_ov(absent, 0, int(RW.pack_tok(0, 0, 33))),                    # a tail token the row lacks
           with_ov(absent, 0, int(RW.pack_tok(2, 3, 0))),                     # a column past the segment's
           with_ov(absent, 0, -2)]                                            # neither -1 nor a token
    for rows in bad:
        out, ref = run(rows)
        assert ref[absent] == 1 and ref.sum() == 1, ref
        with pytest.raises(OP.Refused):
            OP.answer(out, rows)
    # a second override of one action: options 0 and 0 (the same action twice)
    o = base.ov
    r0 = next(b for b in range(12) if o.counts()[b] > 0)
    out, ref = run(with_ov(r0, int(o.opt[o.off[r0]]), int(o.tok[o.off[r0]])))
    assert ref[r0] == 1 and ref.sum() == 1, ref
    # tail tokens: of a segment without a tail (none here: segment 3 does not exist), at a head slot, out of order
    t = base.tail
    rt = next(b for b in range(12) if t.counts()[b] >= 2)
    j = int(t.off[rt])
    for seg, slot in ((7, 9), (0, 1), (0, 1 << 32), (0, -1)):              # (and slots outside u32: a high half -1)
        s2, l2 = t.seg.astype(np.int64).copy(), t.slot.astype(np.int64).copy()
        s2[j], l2[j] = seg, slot
        rows = OP.Rows(base.xs_off, base.xs_idx, base.xs_val, base.n_players, base.opt_off, base.opt_action, base.extra,
                       base.F, tail=None, ov=base.ov)
        f = OP.feeds(rows, v2=True)
        hi, lo = OX.slot_pieces(l2)
        f.update(t_off=t.off, t_seg=s2.astype(np.int32), t_slot_hi=hi, t_slot_lo=lo, t_val=t.val.astype(np.float32))
        ref = np.asarray(sess.run(None, f)[4])
        assert ref[rt] >= 1, (seg, slot, ref)
    f = OP.feeds(base, v2=True)
    f['t_val'] = f['t_val'][:-1]                                             # one float short
    assert (np.asarray(sess.run(None, f)[4]) >= 1).all()
    # pieces outside their fields (no host splits a token so; the graph refuses them, never reads them wrapped)
    for key, val in (('t_slot_lo', 1 << 16), ('t_slot_hi', -3), ('t_slot_hi', 1 << 16)):
        f = OP.feeds(base, v2=True)
        f[key] = f[key].copy()
        f[key][j] = val
        assert np.asarray(sess.run(None, f)[4])[rt] >= 1, (key, val)
    rv = next(b for b in range(12) if base.ov.counts()[b] > 0 and base.ov.tok[base.ov.off[b]] >= 0)
    jv = int(base.ov.off[rv])
    for key, val in (('ov_col', 256), ('ov_col', -1), ('ov_slot_lo', 1 << 16), ('ov_slot_hi', -1), ('ov_seg', -5)):
        f = OP.feeds(base, v2=True)
        f[key] = f[key].copy()
        f[key][jv] = val
        ref = np.asarray(sess.run(None, f)[4])
        assert ref[rv] == 1 and ref.sum() == 1, (key, val, ref)


def test_int32_pieces_and_the_piece_bound():
    """The host's split (zero_worker's `override_pieces` / slot halves): every piece in int32, the packed token and the
    slot back from them exactly; -1 the generic logit, other negatives refused (-2); seg whole. The piece bound
    (`int32_bound`, the numbers of zero_worker's unit test) and a piece over 2^31 refused by the feeder."""
    toks = [int(RW.pack_tok(s, c, l)) for s, c, l in ((0, 0, 0), (2, 7, (1 << 32) - 1), (65535, 255, 65536), (17, 3, 123456789))]
    seg, col, hi, lo = OX.override_pieces(toks)
    assert all(x.dtype == np.int32 and (x >= 0).all() for x in (seg, col, hi, lo))
    back = (seg.astype(np.int64) << 40) | (col.astype(np.int64) << 32) | (hi.astype(np.int64) << 16) | lo
    assert back.tolist() == toks
    assert [x.tolist() for x in OX.override_pieces([-1, -7, 1 << 60])] == [[-1, -2, 1 << 20], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    h, l_ = OX.slot_pieces(np.array([0, 5, 65536, (1 << 32) - 1, 1 << 32, -1], np.int64))
    assert h.tolist() == [0, 0, 1, 65535, -1, -1] and l_[:4].tolist() == [0, 5, 0, 65535]
    t_off = np.array([0, 1, 3])
    assert OX.int32_bound(2, 9, 100, 4, 3, t_off, 3, 7, 5, 3) == 2 * (100 + 1 + 2)
    assert OX.int32_bound(2, 9, 1, 4, 3, t_off, 3, 7, 5, 3) == (3 + 1) * 3
    assert OX.int32_bound(2, 9, 1, 100, 3, t_off, 3, 7, 5, 3) == 2 * (100 + 1)
    net = rand_net(OFFERS, seed=7)
    rows = gate_rows(net, toy_batch(12, seed=9), np.full(12, ROW_LEAF, np.uint8))
    assert set(OP.varlen_feeds(rows, {'actions': A, 'tail_width': 16})) == set(OX.VARLEN_INPUTS)
    with pytest.raises(ValueError, match='2\\^31'):
        OP.varlen_feeds(rows, {'actions': (1 << 31) // 12, 'tail_width': 16})


def test_interface_metadata_and_v1_unchanged(tmp_path):
    """zero-onnx/2's inputs, outputs, dtypes and metadata (tails, option_pointers, max_row_tokens, gates); a network
    without tail segments, pointer overrides or gates exports zero-onnx/1 as before, and each interface refuses the
    other's network."""
    import onnx
    net = rand_net(OFFERS, seed=1)
    path, meta = export(net, tmp_path)
    m = onnx.load(path)
    ins = {i.name: i.type.tensor_type.elem_type for i in m.graph.input}
    T = onnx.TensorProto
    assert list(ins) == list(OX.INPUTS2)
    assert ins == {'x_ptr': T.INT64, 'x_idx': T.INT32, 'x_val': T.FLOAT, 'n_players': T.INT64, 'belief': T.BOOL,
                   'opt_off': T.INT64, 'opt_action': T.INT32, 't_off': T.INT64, 't_seg': T.INT32, 't_slot_hi': T.INT32,
                   't_slot_lo': T.INT32, 't_val': T.FLOAT, 'ov_off': T.INT64, 'ov_opt': T.INT32, 'ov_seg': T.INT32,
                   'ov_col': T.INT32, 'ov_slot_hi': T.INT32, 'ov_slot_lo': T.INT32}
    assert [o.name for o in m.graph.output] == list(OX.OUTPUTS2)
    assert m.graph.output[4].type.tensor_type.elem_type == T.INT64
    # no integer constant outside int32, no shift (DirectML computes int64 in 32 bits); the check refuses one
    assert OX.check_int32(m) > 0 and not any(n.op_type == 'BitShift' for n in m.graph.node)
    m2 = onnx.load(path)
    m2.graph.initializer.append(onnx.numpy_helper.from_array(np.array([1 << 40], np.int64), 'big'))
    with pytest.raises(ValueError, match='outside int32'):
        OX.check_int32(m2)
    m3 = onnx.load(path)
    m3.graph.node.append(onnx.helper.make_node('Constant', [], ['c3'], value_ints=[0, -(1 << 31) - 1]))
    with pytest.raises(ValueError, match='outside int32'):
        OX.check_int32(m3)
    # the runtime half: every int64 value the graph computes on real-looking rows (tails, overrides) below 2^31
    rows = gate_rows(net, toy_batch(40, seed=9, p_head=0.5, p_offer=0.5), kinds_of(40, 5))
    r = OX.int64_max(m, [OP.feeds(rows, v2=True), OP.feeds(rows.take(np.arange(7)), v2=True)], threads=1)
    assert r['tensors'] > 50 and r['untyped'] == 0 and 0 < r['max_abs'] < 1 << 31, r
    assert len(m.graph.output) == len(OX.OUTPUTS2)                           # (the model given is not changed)
    md = {p.key: p.value for p in m.metadata_props}
    assert md['zero_onnx'] == '2' and json.loads(md['tails']) == [[0, 'items', 16, 4], [1, 'marks', 4, 0], [2, 'offers', 4, 0]]
    assert md['option_pointers'] == 'true' and md['max_row_tokens'] == '64' and md['tail_width'] == '16'
    assert json.loads(md['gates']) == {'marks': 1.0, 'offers': 1.0}
    assert not any(n.op_type == 'ScatterND' and any(a.name == 'reduction' and a.s not in (b'', b'none') for a in n.attribute)
                   for n in m.graph.node), 'no ScatterND with a reduction (ONNX Runtime CPU races on duplicates)'
    plain = rand_net(dict(TOY_TAIL_TOKENS_BASE, segments=[{k: v for k, v in TOY_TAIL_TOKENS_BASE['segments'][0].items()
                                                           if k != 'tail'}], option_pointers=False, max_row_tokens=None),
                     seed=1)
    assert OX.onnx_version(plain) == 1
    p1, m1 = export(plain, tmp_path, 'v1')
    assert m1['zero_onnx'] == '1' and [i.name for i in onnx.load(p1).graph.input] == list(OX.INPUTS)
    # both graphs without the exporter's annotations (stack traces with the checkout's paths, doc strings)
    for p in (path, p1):
        g = onnx.load(p)
        assert not g.doc_string and not g.graph.doc_string
        assert not any(len(n.metadata_props) or n.doc_string for n in g.graph.node), p
        assert not any(len(v.metadata_props) or v.doc_string for v in list(g.graph.value_info) + list(g.graph.input))
        assert b'stack_trace' not in open(p, 'rb').read() and b'onnx_export.py' not in open(p, 'rb').read()
    with pytest.raises(ValueError, match='zero-onnx/1'):
        OX.ZeroOnnx(net)
    with pytest.raises(ValueError, match='zero-onnx/1'):
        OX.ZeroOnnx2(plain)
    with pytest.raises(ValueError, match='packed'):
        OX.ZeroOnnx2(net, 'mask')


def test_toy_tail_games_through_the_graph(tmp_path):
    """Real toy_tail driver batches (sparse; tail tokens, overrides, leaf and value-only rows) answered by
    the graph through `OnnxEvaluator`: every answer within 2e-5 of the eager evaluator's, the games played to the end
    (rows with tail tokens and overrides among them)."""
    from zero.onnx_eval import OnnxEvaluator
    info = G.info('toy_tail')
    net = rand_net(TOY_TAIL_TOKENS, seed=2)
    assert info['num_features'] == F and info['num_actions'] == A
    path, _ = export(net, tmp_path)
    oe = OnnxEvaluator(path, threads=1, expect={'game': 'toy', 'features': info['num_features']})
    ev = NetEvaluator(net, 'cpu')
    seat = G.nn_seat('m', 8, selfplay=True, sims_fast=4, p_full=0.5, leaf_value_view='root', view_samples=0.5)
    drv = G.Driver('toy_tail', [[seat, seat]] * 12, spec='{"items0": 9}', concurrency=12, threads=2, groups=1, seed=5,
                   record=True, sparse=True)
    seen = {'batches': 0, 'tails': 0, 'ovs': 0, 'value': 0, 'worst': 0.0}

    def graph(lb):
        got = oe(lb)
        want = ev(lb)
        seen['worst'] = max([seen['worst']] + OP.diffs(got, want))
        seen['batches'] += 1
        seen['tails'] += int(lb.a['t_off'][-1])
        seen['ovs'] += int(lb.a['ov_off'][-1])
        seen['value'] += int((lb.a['extra'] == ROW_VALUE).sum())
        return got
    drive(drv, {'m': graph})
    res = drv.take_results()
    assert len(res) == 12 and all(r['over'] for r in res)
    assert seen['tails'] > 0 and seen['ovs'] > 0 and seen['value'] > 0 and seen['worst'] < 2e-5, seen


def test_reference_set_with_tails_and_overrides(tmp_path):
    """A zero-onnx/2 network's reference set holds rows with tail tokens and overrides (the batches holding them first)
    and their arrays (the driver's dtypes, t_segs in its meta); the graph through onnxruntime is within 2e-5 of it;
    rows without a tail token, or without an override, are refused."""
    net = rand_net(OFFERS, seed=4)
    with torch.no_grad():
        net.gate.copy_(torch.tensor([0.5, 0.0]))
    path, _ = export(net, tmp_path)
    B = 200
    rows = gate_rows(net, toy_batch(B, seed=6, p_tail=0.1, p_mark=0.1, p_gen=0.05, p_offer=0.1), kinds_of(B, 8))
    rows.batch_off = np.array([0, 40, 90, 130, 170, 200], np.int64)
    ref = str(tmp_path / 'reference.npz')
    meta = OX.write_reference(net, [rows], ref, 80, {'checkpoint': 'toy'})
    assert meta['zero_onnx'] == '2' and meta['tail_tokens'] > 0 and meta['overrides'] > 0
    assert meta['t_segs'] == [[0, 'items', 16, 4], [1, 'marks', 4, 0], [2, 'offers', 4, 0]]
    with np.load(ref) as z:
        dt = {k: z[k].dtype for k in ('t_off', 't_seg', 't_slot', 't_val', 'ov_off', 'ov_opt', 'ov_tok')}
        assert dt == {'t_off': np.int64, 't_seg': np.uint16, 't_slot': np.uint32, 't_val': np.float32, 'ov_off': np.int64,
                      'ov_opt': np.int32, 'ov_tok': np.int64}
    chk = OX.check_reference(path, ref, threads=1)
    assert chk['max_abs'] < 2e-5, chk
    r2, _bo, _want, _m = OX.load_reference(ref)
    assert r2.tail.NT == meta['tail_tokens'] and r2.ov.NO == meta['overrides']
    none = gate_rows(net, toy_batch(B, seed=6, p_tail=0.0, p_mark=0.0, p_gen=0.0, p_offer=0.0), kinds_of(B, 8))
    none.batch_off = rows.batch_off
    with pytest.raises(ValueError, match='tail token'):
        OX.write_reference(net, [none], str(tmp_path / 'x.npz'), 80)


def test_new_tail_segments_need_no_code_change(tmp_path):
    """Tail segments past layout v7's (docs/az.md 46.7): `toy_tail_wide` - toy_tail plus 20 gated tail-only segments
    x02 .. x21, more tail segments than MTG Zero's v7 and indices past its highest - exports, plays through
    `OnnxEvaluator` and replays its records with no code of the exporter, the evaluator or the records knowing them
    (everything from the network's layout, the graph's metadata and the driver's arrays): the graph's `tails` are the
    game's, every answer within 2e-5 of eager with the gates open, the extra segments' tokens in the rows, and every
    game's record replays to the driver's rows byte for byte."""
    from zero.onnx_eval import OnnxEvaluator
    from zero.package import TOY_TAIL_WIDE_TOKENS
    from zero.workqueue import split_games
    info = G.info('toy_tail_wide')
    segs = [t['seg'] for t in info['tails']]
    assert segs == list(range(22)) and info['max_row_tokens'] == 96
    net = rand_net(TOY_TAIL_WIDE_TOKENS, seed=8)
    assert net.tails == list(range(22)) and net.gated == list(range(2, 22))
    with torch.no_grad():
        net.gate.copy_(torch.linspace(0.15, 1.3, 20))
    path, meta = export(net, tmp_path)
    assert meta['zero_onnx'] == '2'
    assert json.loads(meta['tails']) == [[t['seg'], t['name'], t['width'], t['head']] for t in info['tails']]
    assert len(json.loads(meta['gates'])) == 20 and meta['max_row_tokens'] == '96'
    oe = OnnxEvaluator(path, threads=1, expect={'game': 'toy', 'features': info['num_features']})
    ev = NetEvaluator(net, 'cpu')
    seat = G.nn_seat('m', 8, selfplay=True, sims_fast=4, p_full=0.5, leaf_value_view='root', view_samples=0.5)
    spec = '{"items0": 12}'
    # (a traced rolling driver, as the farm's: each finished game's record and samples)
    drv = G.Driver('toy_tail_wide', [[seat, seat]], spec=spec, concurrency=6, threads=2, groups=1, seed=7,
                   record=True, sparse=True, rolling=True)
    drv.set_trace(True)
    seen = {'batches': 0, 'extra': 0, 'segs': set(), 'ovs': 0, 'worst': 0.0}

    def graph(lb):
        got = oe(lb)
        seen['worst'] = max([seen['worst']] + OP.diffs(got, ev(lb)))
        seen['batches'] += 1
        t = lb.a['t_seg']
        seen['extra'] += int((t >= 2).sum())
        seen['segs'] |= set(int(s) for s in np.unique(t))
        seen['ovs'] += int(lb.a['ov_off'][-1])
        return got
    games = []
    while len(games) < 10:
        drv.resume(10 - len(games), 0)
        drive(drv, {'m': graph})
        rows, texts, s = drv.take_game_records()
        games += [(sd, text) for (_row, sd), text in zip(split_games(rows, s, G.tail_widths(info)), texts)]
    assert seen['extra'] > 200 and seen['segs'] == set(range(22)) and seen['ovs'] > 0 and seen['worst'] < 2e-5, seen
    E = G.engine()
    n = extra = 0
    for sd, text in games[:10]:
        s2, _info = E.replay_rows('toy_tail_wide', spec, text, True)
        if sd is None:
            assert s2 is None
            continue
        assert set(s2) == set(sd)
        for k in sd:
            a, b = np.asarray(sd[k]), np.asarray(s2[k])
            if k in ('pi', 'q'):
                a, b = a.astype(np.float16), b.astype(np.float16)    # (a record carries them as float16, as shards do)
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), k
        n += 1
        extra += int((np.asarray(sd['t_seg']) >= 2).sum())
    assert n == 10 and extra > 100, (n, extra)
