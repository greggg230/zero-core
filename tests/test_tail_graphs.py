"""CUDA graphs of a token network with tails and pointer overrides (docs/az.md 43.7): `TokenGraphRunner` through
`NetEvaluator` against eager inference (priors, value, margin, root belief rows, value-only rows; dense and
sparse batches; gated tokens at g = 0 and g > 0; batches in pieces), graphs keyed by (rows, tokens, tail
tokens, overrides), rows of more than LONG_T tokens evaluated eagerly beside the graphs, a batch without a tail
token through the code of before, the widen's exactness through the graphs, new weights in place, the host's
refusals. Built from tensors and the toy's literal layout (test_tail_net)."""
import numpy as np
import pytest
import torch

from zero import infer
from zero.infer import NetEvaluator, tail_bucket, take_options, token_index_tail, token_presence
from zero.net import widen
from zero.protocol import ROW_BELIEF, ROW_LEAF, ROW_VALUE

from test_tail_net import (F, HEAD_ONLY, OFFERS, R, TOY_GATED, TOY_TAIL_TOKENS, keep_segs, leaf_batch, rand_net,
                           toy_batch)

CUDA = torch.cuda.is_available()
cuda = pytest.mark.skipif(not CUDA, reason='CUDA')
WIDTHS = {0: 16, 1: 4, 2: 4}


def tails(net, d):
    widths = {j: net.segs[j]['width'] for j in net.tails}
    return R.Tail.from_arrays(d, widths), R.Overrides.from_arrays(d)


def gated_net(layout=OFFERS, seed=0, theta=(0.0, 0.0), aux=True):
    m = rand_net(layout, seed, aux=[['a', 5, 'multinomial']] if aux else None)
    with torch.no_grad():
        if m.gated:
            m.gate.data[:] = torch.tensor(theta[:len(m.gated)])
        m.ptr.weight.normal_(0, 0.5)
    return m


def with_values(d, seed=0):
    """`d` with a third of its rows value-only (no options) and a few root belief rows (`extra`)."""
    rng = np.random.default_rng(seed)
    B = len(d['x'])
    vonly = rng.random(B) < 0.3
    keep = np.repeat(~vonly, np.diff(d['opt_off']))
    cnt = np.where(vonly, 0, np.diff(d['opt_off']))
    d = dict(d, opt_action=d['opt_action'][keep], opt_off=np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64))
    # (a value-only row's overrides go too)
    o = {'ov_off': [0], 'ov_opt': [], 'ov_tok': []}
    for b in range(B):
        if not vonly[b]:
            o['ov_opt'].extend(d['ov_opt'][d['ov_off'][b]:d['ov_off'][b + 1]])
            o['ov_tok'].extend(d['ov_tok'][d['ov_off'][b]:d['ov_off'][b + 1]])
        o['ov_off'].append(len(o['ov_opt']))
    d.update({k: np.array(v, np.int32 if k == 'ov_opt' else np.int64) for k, v in o.items()})
    extra = np.where(vonly, ROW_VALUE, ROW_LEAF).astype(np.int8)
    extra[rng.random(B) < 0.1] = ROW_BELIEF
    d['extra'] = extra
    return d


def evaluate(ev, d, net):
    tl, ov = tails(net, d)
    return ev.evaluate(d['x'], d['n_players'], d['opt_off'], d['opt_action'], d.get('extra'),
                       tail=tl if net.tails else None, ov=ov if net.option_pointers else None)


def close(a, b, tol=2e-5):
    return len(a) == len(b) and all(np.allclose(u, v, atol=tol) for u, v in zip(a, b))


def sparse_lb(d, net):
    x = d['x']
    nz = x != 0
    arrays = {'xs_off': np.concatenate([[0], np.cumsum(nz.sum(1))]).astype(np.int64),
              'xs_idx': np.nonzero(nz)[1].astype(np.uint32), 'xs_val': x[nz].astype(np.float32), 'n_features': F,
              'player': np.zeros(len(x), np.int64), 'n_players': d['n_players'], 'opt_off': d['opt_off'],
              'opt_action': d['opt_action']}
    if d.get('extra') is not None:
        arrays['extra'] = d['extra']
    return leaf_batch(arrays, d, net)


def test_host_packing_with_tails():
    """`token_index_tail`: each row's head tokens in layout order, then its tail tokens at n_tok + their index
    in the batch's tail; `tail_bucket`; `take_options`."""
    m = rand_net(TOY_TAIL_TOKENS, 0)
    d = toy_batch(9, 1, p_tail=0.7, p_mark=0.7)
    P = token_presence(d['x'], m.present_index())
    tcnt = np.diff(d['t_off'])
    full = token_index_tail(P, tcnt, m.n_tok, np.full((10, 20), 99, np.int64))
    j = 0
    for b in range(9):
        heads = list(np.flatnonzero(P[b]))
        row = heads + [m.n_tok + j + k for k in range(tcnt[b])]
        assert full[b, :len(row)].tolist() == row and not full[b, len(row):].any()
        j += tcnt[b]
    assert not full[9].any()
    assert [tail_bucket(n) for n in (0, 1, 16, 17, 32, 33, 300)] == [0, 16, 16, 32, 32, 64, 512]
    off, act = take_options(d['opt_off'], d['opt_action'], np.array([2, 0, 5]))
    assert act.tolist() == (d['opt_action'][d['opt_off'][2]:d['opt_off'][3]].tolist()
                            + d['opt_action'][:d['opt_off'][1]].tolist()
                            + d['opt_action'][d['opt_off'][5]:d['opt_off'][6]].tolist())


@cuda
@pytest.mark.parametrize('theta', [(0.0, 0.0), (0.8, 0.4)])
def test_graphs_match_eager_with_tails(theta):
    """Through the graphs (dense and sparse batches, in pieces too) the answers are eager's to float32 rounding:
    tail tokens, gated ones (g = 0 and g > 0), overrides into head slots, tail tokens and a gated segment,
    generic overrides, value-only rows, root belief rows; every batch through the graphs, one graph per (rows,
    tokens, tail tokens, overrides) bucket; replays deterministic."""
    net = gated_net(theta=theta).cuda()
    eager = NetEvaluator(net, 'cuda', graphs=False)
    ev = NetEvaluator(net, 'cuda', graphs=True)
    for B, seed in ((1, 1), (7, 2), (64, 3), (130, 4)):
        d = with_values(toy_batch(B, seed, p_tail=0.5, p_mark=0.5, p_offer=0.4), seed)
        want = evaluate(eager, d, net)
        got = evaluate(ev, d, net)
        assert close(want, got), (B, [float(np.abs(a - b).max()) for a, b in zip(want, got)])
        assert all(np.array_equal(a, b) for a, b in zip(got, evaluate(ev, d, net)))          # deterministic
        lb = sparse_lb(d, net)
        assert all(np.array_equal(a, b) for a, b in zip(ev(lb), got))                     # sparse: bit for bit
        assert all(np.array_equal(a, b) for a, b in zip(ev.result(ev.launch(lb)), got))
    assert ev.graph_calls == ev.calls and ev.fast.long_rows == 0
    keys = list(ev.fast.graphs)
    assert all(len(k) == 4 and k[2] == tail_bucket(k[2]) and k[3] == tail_bucket(k[3]) for k in keys)
    assert any(k[2] for k in keys) and any(k[3] for k in keys)
    pieces = NetEvaluator(net, 'cuda', graphs=True, max_rows=16)
    d = with_values(toy_batch(70, 9, p_tail=0.5, p_mark=0.5, p_offer=0.4), 9)
    assert close(evaluate(eager, d, net), evaluate(pieces, d, net)) and pieces.pieces == 1
    assert all(np.array_equal(a, b) for a, b in zip(pieces(sparse_lb(d, pieces.model)), evaluate(pieces, d, net)))


@cuda
def test_a_batch_without_tail_tokens_runs_the_graph_of_before():
    """A tail network's batch with no tail token and no override runs the graph (rows, tokens, 0, 0): the
    static forward of before - bit for bit the head-only network's graphs on the same rows (a widen: the same
    weights)."""
    m = rand_net(HEAD_ONLY, 5)
    w = widen(m, layout=TOY_TAIL_TOKENS)
    m, w = m.cuda(), w.cuda()
    d = toy_batch(90, 6, p_tail=0, p_mark=0, p_gen=0)
    old = NetEvaluator(m, 'cuda', graphs=True).evaluate(d['x'], d['n_players'], d['opt_off'], d['opt_action'])
    ev = NetEvaluator(w, 'cuda', graphs=True)
    new = evaluate(ev, d, w)
    assert all(np.array_equal(a, b) for a, b in zip(old, new))
    assert list(ev.fast.graphs) == [(128, 16, 0, 0)]


@cuda
def test_the_widen_is_exact_through_the_graphs():
    """The widened network's graphs on rows whose only new tokens are gated (g = 0) answer as the old network's
    graphs, bit for bit where the batch's token bucket is the same (here every batch), to rounding else."""
    m = rand_net(HEAD_ONLY, 7)
    w = widen(m, layout=TOY_TAIL_TOKENS)
    m, w = m.cuda(), w.cuda()
    eo, ew = NetEvaluator(m, 'cuda', graphs=True), NetEvaluator(w, 'cuda', graphs=True)
    for B, seed in ((5, 8), (64, 9), (200, 10)):
        d = with_values(keep_segs(toy_batch(B, seed, p_tail=0, p_mark=0.6, p_gen=0), {1}), seed)
        assert len(d['t_seg'])
        old = eo.evaluate(d['x'], d['n_players'], d['opt_off'], d['opt_action'], d['extra'])
        new = evaluate(ew, d, w)
        assert all(np.array_equal(a, b) for a, b in zip(old, new)), [float(np.abs(a - b).max()) for a, b in zip(old, new)]


@cuda
def test_long_rows_run_eagerly_beside_the_graphs(monkeypatch):
    """Rows of more than LONG_T tokens are evaluated eagerly (in groups bounded by LONG_BYTES), the others
    through the graphs, the answers merged in row order (belief rows too): eager's answers; dense and sparse.
    Real rows past 128 tokens, and LONG_T lowered so that a batch mixes both."""
    lay = dict(TOY_GATED, max_row_tokens=4096)
    net = gated_net(lay, 11, theta=(0.5,)).cuda()
    eager = NetEvaluator(net, 'cuda', graphs=False)
    ev = NetEvaluator(net, 'cuda', graphs=True)
    d = with_values(toy_batch(20, 12, p_tail=0.5, p_mark=0.5), 12)
    # three rows get 140 more tail items (slots 40..179)
    d = _long_rows(d, [3, 7, 15], 140)
    assert (np.diff(d['t_off'])[[3, 7, 15]] > 128).all()
    want = evaluate(eager, d, net)
    got = evaluate(ev, d, net)
    assert close(want, got) and ev.fast.long_rows == 3
    assert all(np.array_equal(a, b) for a, b in zip(ev(sparse_lb(d, net)), got))
    monkeypatch.setattr(infer, 'LONG_T', 6)
    monkeypatch.setattr(infer, 'LONG_BYTES', 4 * 2 * 40 ** 2 * 3)     # (groups of a few rows)
    ev2 = NetEvaluator(net, 'cuda', graphs=True)
    d2 = with_values(toy_batch(50, 13, p_tail=0.6, p_mark=0.6), 13)
    assert close(evaluate(eager, d2, net), evaluate(ev2, d2, net))
    assert 0 < ev2.fast.long_rows < 50


def _long_rows(d, rows, k):
    """`d` with k more tail items (slots 40.., after the row's own items, before its marks) in each of `rows`."""
    d = dict(d)
    B = len(d['t_off']) - 1
    vo = np.concatenate([[0], np.cumsum([WIDTHS[int(s)] for s in d['t_seg']])]).astype(np.int64)
    seg, slot, val, off = [], [], [], [0]
    for b in range(B):
        idx = list(range(d['t_off'][b], d['t_off'][b + 1]))
        items = [j for j in idx if d['t_seg'][j] == 0]
        rest = [j for j in idx if d['t_seg'][j] != 0]
        for j in items:
            seg.append(0)
            slot.append(int(d['t_slot'][j]))
            val.extend(d['t_val'][vo[j]:vo[j + 1]])
        if b in rows:
            for s in range(40, 40 + k):
                v = np.zeros(16, np.float32)
                v[0], v[1], v[6 + s % 10] = 1, s / 100, 1
                seg.append(0)
                slot.append(s)
                val.extend(v)
        for j in rest:
            seg.append(int(d['t_seg'][j]))
            slot.append(int(d['t_slot'][j]))
            val.extend(d['t_val'][vo[j]:vo[j + 1]])
        off.append(len(seg))
    d.update({'t_off': np.array(off, np.int64), 't_seg': np.array(seg, np.uint16), 't_slot': np.array(slot, np.uint32),
              't_val': np.array(val, np.float32)})
    return d


@cuda
def test_new_weights_in_place_and_gates_seen_by_the_next_replay():
    net = gated_net(theta=(0.0, 0.0)).cuda()
    ev = NetEvaluator(net, 'cuda', graphs=True)
    d = with_values(toy_batch(40, 14, p_tail=0.5, p_mark=0.7, p_offer=0.5), 14)
    a = evaluate(ev, d, net)
    st = {k: v.clone() for k, v in net.state_dict().items()}
    st['gate'] = torch.tensor([0.9, 0.7])
    ev.set_weights(st)
    b = evaluate(ev, d, net)
    c = evaluate(NetEvaluator(net, 'cuda', graphs=False), d, net)
    assert close(b, c) and not close(a, b, 1e-4)


@cuda
def test_the_graphs_refuse_bad_tails_and_overrides():
    """The host checks what the graphs cannot (R5): a tail token at a head slot; an override to an absent head slot, to another row's tail token, a column the segment
    lacks; a batch without its tail / overrides; a non-tail network given them."""
    net = gated_net(seed=15).cuda()
    ev = NetEvaluator(net, 'cuda', graphs=True)
    d = toy_batch(12, 16, p_head=0.5, p_tail=0.8, p_mark=0.8, p_gen=0)
    tl, ov = tails(net, d)
    j0 = int(np.flatnonzero(d['t_seg'] == 0)[0])
    for chg in ({'t_slot': (j0, 2)},):                     # (an unknown segment: the rows container's KeyError)
        bad = {k: v.copy() for k, v in d.items()}
        for k, (i, v) in chg.items():
            bad[k][i] = v
        with pytest.raises(ValueError):
            evaluate(ev, bad, net)
    P = token_presence(d['x'], net.present_index())
    br, bs = np.argwhere(~P[:, 1:5])[0]
    o = R.Overrides.from_arrays({'ov_off': np.array([0] * (br + 1) + [1] * (12 - br), np.int64),
                                 'ov_opt': np.array([0], np.int32), 'ov_tok': np.array([int(bs)], np.int64)})
    with pytest.raises(ValueError, match='override'):
        ev.evaluate(d['x'], d['n_players'], d['opt_off'], d['opt_action'], tail=tl, ov=o)
    with pytest.raises(ValueError):
        ev.evaluate(d['x'], d['n_players'], d['opt_off'], d['opt_action'], ov=ov)
    with pytest.raises(ValueError):
        ev.evaluate(d['x'], d['n_players'], d['opt_off'], d['opt_action'], tail=tl)
    old = NetEvaluator(rand_net(HEAD_ONLY, 15).cuda(), 'cuda', graphs=True)
    with pytest.raises(ValueError):
        old.evaluate(d['x'], d['n_players'], d['opt_off'], d['opt_action'], tail=tl)
    with pytest.raises(ValueError):
        old.fast.launch(d['x'], d['n_players'], d['opt_off'], d['opt_action'], tail=tl)
