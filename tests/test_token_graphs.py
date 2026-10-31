"""CUDA graphs of the token network's inference (docs/az.md 10.6): `TokenNet.forward_static` (the
static-shape forward the graphs capture) against `forward`, the host's packing (`infer.token_presence`,
`token_index`), `infer.TokenGraphRunner` through `NetEvaluator` against eager inference (priors, value,
margin, the root belief rows' auxiliary predictions, value-only rows; deterministic replays, new weights
in place, launches in flight), games through the graphs, and the opt-in: off by default (env
ZERO_TOKEN_GRAPHS, ZERO_CUDA_GRAPHS=0, a CPU evaluator), zero.h2h's flags."""
import json

import numpy as np
import pytest
import torch

from zero import games as G, h2h
from zero.infer import NetEvaluator, TokenGraphRunner, drive, token_index, token_presence
from zero.net import TokenNet, make_net, save
from zero.protocol import ROW_BELIEF, ROW_LEAF, ROW_VALUE, belief_rows, validate_evaluation

from test_net_data import TOK_A, TOK_F, TOK_LAYOUT, TOK_LAYOUT_ID, tok_x

CUDA = torch.cuda.is_available()
AUX = [['a', 5, 'multinomial'], ['b', 3, 'mse']]


def net(layout=TOK_LAYOUT_ID, aux=AUX, seed=0):
    """A small token network with random policy, value, margin and auxiliary output layers."""
    torch.manual_seed(seed)
    m = TokenNet(TOK_F, TOK_A, layout, d=16, layers=2, heads=2, ff=32, head=16, aux=aux).eval()
    with torch.no_grad():
        for lin in [m.gen, m.ptr, m.val[-1], m.mar[-1]] + ([m.aux_head[2]] if aux else []):
            lin.weight.normal_(0, 0.3)
            lin.bias.normal_(0, 0.3)
    return m


def static_inputs(m, x, pad_t=0, pad_rows=0, junk=False):
    """forward_static's (full, ntok) for rows x (torch, CPU), T = the longest row + pad_t, plus
    pad_rows rows of padding (ntok 1); junk: the padding positions hold other slots."""
    P = token_presence(x.numpy(), m.present_index())
    B = len(P)
    T = int(P.sum(1).max()) + pad_t
    full = token_index(P, np.zeros((B + pad_rows, T), np.int64))
    ntok = np.concatenate([P.sum(1), np.ones(pad_rows, np.int64)])
    if junk:
        rng = np.random.default_rng(0)
        pad = np.arange(T)[None, :] >= ntok[:, None]
        full[pad] = rng.integers(0, m.n_tok, size=int(pad.sum()))
    return torch.from_numpy(full), torch.from_numpy(ntok)


@pytest.mark.parametrize('layout', ['plain', 'id_actions'])
def test_forward_static_is_forward(layout):
    """The static-shape forward computes `forward`'s outputs (policy, value, margin, the auxiliary head
    on every row) whatever the padding: longer rows, padding positions holding other slots, padding
    rows."""
    m = net(TOK_LAYOUT if layout == 'plain' else TOK_LAYOUT_ID)
    for B, seed in ((1, 1), (9, 2), (37, 3)):
        x = tok_x(B, seed)
        n = torch.tensor(([2, 3, 4] * B)[:B])
        with torch.no_grad():
            want = m(x, n, aux=True)
            for pad_t, pad_rows, junk in ((0, 0, False), (5, 0, True), (3, 6, True)):
                full, ntok = static_inputs(m, x, pad_t, pad_rows, junk)
                xp = torch.cat([x, tok_x(pad_rows, 9)]) if pad_rows else x
                npad = torch.cat([n, torch.full((pad_rows,), 2)]) if pad_rows else n
                got = m.forward_static(xp, npad, full, ntok, aux=True)
                for a, b in zip(want, got):
                    assert torch.allclose(a, b[:B], atol=1e-5), (B, pad_t, pad_rows, (a - b[:B]).abs().max())


def test_token_presence_and_index():
    m = net()
    x = tok_x(6, 4)
    P = token_presence(x.numpy(), m.present_index())
    assert P.shape == (6, m.n_tok) and P[:, 0].all() and P[:, 1].all()      # the global token, the ctx slot
    for b in range(6):
        for s in range(4):
            assert P[b, 2 + s] == (x[b, 43 + 9 * s] > 0.5)
        for s in range(3):
            assert P[b, 6 + s] == (x[b, 79 + 4 * s + 3] > 0.5)
    full = token_index(P, np.full((8, int(P.sum(1).max()) + 2), 99, np.int64))
    for b in range(6):
        k = int(P[b].sum())
        assert full[b, :k].tolist() == np.flatnonzero(P[b]).tolist() and not full[b, k:].any()
    assert not full[6:].any()


# ---------------------------------------------------------------- the opt-in

def test_off_by_default_and_the_switches(monkeypatch, capsys):
    """A token network's evaluator is eager unless asked: env ZERO_TOKEN_GRAPHS=1 (CUDA only: said and
    eager on a CPU evaluator; ZERO_CUDA_GRAPHS=0 overrides it, said) or graphs=True (an error on a CPU
    evaluator). The other networks ignore the env; graphs=False is eager whatever the env."""
    from zero.net import Net
    monkeypatch.delenv('ZERO_TOKEN_GRAPHS', raising=False)
    monkeypatch.delenv('ZERO_CUDA_GRAPHS', raising=False)
    m = net()
    assert NetEvaluator(m).fast is None
    if CUDA:
        assert NetEvaluator(net(), 'cuda').fast is None
    monkeypatch.setenv('ZERO_TOKEN_GRAPHS', '1')
    assert NetEvaluator(m).fast is None
    assert 'CUDA graphs need CUDA, eager inference' in capsys.readouterr().err
    with pytest.raises(ValueError, match='CUDA only'):
        NetEvaluator(m, graphs=True)
    assert NetEvaluator(Net(TOK_F, TOK_A, hidden=8, blocks=1)).fast is None
    assert NetEvaluator(m, graphs=False).fast is None
    if CUDA:
        ev = NetEvaluator(net(), 'cuda')
        assert isinstance(ev.fast, TokenGraphRunner) and ev.workers == 1
        assert 'token network CUDA graphs on' in capsys.readouterr().err
        assert NetEvaluator(net(), 'cuda', graphs=False).fast is None
        monkeypatch.setenv('ZERO_CUDA_GRAPHS', '0')
        assert NetEvaluator(net(), 'cuda').fast is None
        assert 'ZERO_CUDA_GRAPHS=0' in capsys.readouterr().err
        assert isinstance(NetEvaluator(net(), 'cuda', graphs=True).fast, TokenGraphRunner)
    monkeypatch.setenv('ZERO_TOKEN_GRAPHS', '0')
    assert NetEvaluator(m).fast is None


def test_h2h_graph_flags(tmp_path):
    """zero.h2h --graphs / --a-graphs / --b-graphs: given, the output's graphs_a / graphs_b (each side's
    share of batches through the graphs); not given, the match of before (no keys); a side asked for graphs
    on a CPU evaluator is an error."""
    info = G.info('toy')
    torch.manual_seed(5)
    path = str(tmp_path / 'tok.pt')
    save(make_net(info, 'toy', 'tokens', 16, 1, heads=2), path)
    base = [path, path, '--games', '4', '--sims', '4', '--threads', '1']
    out = {}
    for flags in ([], ['--graphs', '0'], ['--a-graphs', '0', '--b-graphs', '0']):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            h2h.main(base + flags)
        out[' '.join(flags)] = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert 'graphs_a' not in out['']
    for k in ('--graphs 0', '--a-graphs 0 --b-graphs 0'):
        assert out[k]['graphs_a'] == out[k]['graphs_b'] == 0.0
        assert {x: out[k][x] for x in ('a_score', 'a_margin', 'games')} == \
               {x: out[''][x] for x in ('a_score', 'a_margin', 'games')}
    with pytest.raises(ValueError, match='CUDA only'):
        h2h.main(base + ['--a-graphs', '1'])
    assert h2h._evaluators(make_net(info, 'toy', 'tokens', 16, 1, heads=2), make_net(info, 'toy', 'tokens', 16, 1, heads=2),
                           'cpu', (False, None))['b'].fast is None


# ---------------------------------------------------------------- CUDA: the graphs against eager

def a_batch(m, B, seed):
    """A leaf batch of B rows: decision rows with 2-5 options, value-only rows and root belief rows (no
    options), as the search sends them."""
    rng = np.random.default_rng(seed)
    x = tok_x(B, seed).numpy()
    kind = rng.choice([ROW_LEAF, ROW_VALUE, ROW_BELIEF], B, p=[0.6, 0.3, 0.1]).astype(np.uint8)
    kind[0] = ROW_LEAF
    cnt = np.where(kind == ROW_LEAF, rng.integers(2, 6, B), 0)
    off = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int32)
    act = np.concatenate([np.sort(rng.choice(TOK_A, c, replace=False)) for c in cnt]).astype(np.int32)
    npl = rng.integers(2, 5, B).astype(np.int32)
    return x, npl, off, act, kind


def close(a, b, tol=2e-5):
    assert len(a) == len(b)
    for u, v in zip(a, b):
        assert u.shape == v.shape and u.dtype == v.dtype and np.abs(u - v).max() <= tol, np.abs(u - v).max()


@pytest.mark.skipif(not CUDA, reason='CUDA graphs need CUDA')
def test_graphs_match_eager_and_follow_new_weights():
    import copy
    m = net().cuda()
    eager = NetEvaluator(m, 'cuda', graphs=False)
    fast = NetEvaluator(copy.deepcopy(m), 'cuda', graphs=True)
    assert isinstance(fast.fast, TokenGraphRunner) and eager.fast is None
    for step in range(2):
        for B in (1, 63, 64, 65, 130):
            x, npl, off, act, kind = a_batch(m, B, B + step)
            e = eager.evaluate(x, npl, off, act, kind)
            f = fast.evaluate(x, npl, off, act, kind)
            assert len(f) == (4 if len(belief_rows(kind)) else 3)
            close(e, f)
            assert all(np.array_equal(u, v) for u, v in zip(f, fast.evaluate(x, npl, off, act, kind))), 'replays differ'
            assert np.all(f[1][np.arange(8)[None, :] >= npl[:, None]] == 0)
            # without belief rows: three items, as eager
            close(eager.evaluate(x, npl, off, act), fast.evaluate(x, npl, off, act))
        # new weights (rolling self-play): the graphs read them in place
        new = {k: v + 0.05 * torch.randn_like(v) for k, v in m.state_dict().items()}
        eager.set_weights(new)
        fast.set_weights(new)
    assert fast.graph_calls == fast.calls > 0 and eager.graph_calls == 0
    assert fast.fast.captures == len(fast.fast.graphs) < fast.fast.replays


@pytest.mark.skipif(not CUDA, reason='CUDA graphs need CUDA')
def test_absent_slots_are_never_read():
    """The host copies the features outside the segments and the present slots only: an absent slot's
    floats (here NaN, its present flag aside) change nothing, eager or through the graphs, before and
    after batches that filled those slots."""
    m = net().cuda()
    eager = NetEvaluator(m, 'cuda', graphs=False)
    fast = NetEvaluator(m, 'cuda', graphs=True)
    x, npl, off, act, kind = a_batch(m, 50, 61)
    want = eager.evaluate(x, npl, off, act, kind)
    bad = x.copy()
    for s in range(4):
        o = 43 + 9 * s
        bad[bad[:, o] < 0.5, o + 1:o + 9] = np.nan
    for s in range(3):
        o = 79 + 4 * s
        bad[bad[:, o + 3] < 0.5, o:o + 3] = np.nan
    assert np.isnan(bad).any()
    close(want, eager.evaluate(bad, npl, off, act, kind), 0)
    for k in range(3):
        if k:
            fast.evaluate(*a_batch(m, 64, 70 + k))       # (other rows in the host buffers)
        close(want, fast.evaluate(bad, npl, off, act, kind))


@pytest.mark.skipif(not CUDA, reason='CUDA graphs need CUDA')
def test_launches_in_flight():
    """Several launches before their results (drive's pipeline, ZERO_INFLIGHT > 2): each keeps its own host
    buffers; the answers are those of one at a time."""
    from zero.protocol import LeafBatch
    m = net().cuda()
    ev = NetEvaluator(m, 'cuda', graphs=True)
    lbs = []
    for k, B in enumerate((70, 5, 130, 64)):
        x, npl, off, act, kind = a_batch(m, B, 40 + k)
        lbs.append(LeafBatch(k, 'm', 'nn', {'x': x, 'n_players': npl, 'opt_off': off, 'opt_action': act,
                                            'player': np.zeros(B, np.int32), 'gid': np.arange(B, dtype=np.int64),
                                            'extra': kind}))
    one = [ev(lb) for lb in lbs]
    hs = [ev.launch(lb) for lb in lbs]
    for want, h in zip(one, hs):
        got = ev.result(h)
        assert all(np.array_equal(u, v) for u, v in zip(want, got))
    assert len(ev.fast.slots) >= 4


@pytest.mark.skipif(not CUDA, reason='CUDA graphs need CUDA')
@pytest.mark.parametrize('graphs', [False, True])
def test_games_through_the_graphs(graphs):
    """Toy games with the root's belief and root-view leaves (root belief rows, value-only rows) through a
    token network: every batch through the graphs, every answer valid; eager the same games' shapes."""
    from test_aux import TOY_AUX
    info = G.info('toy')
    torch.manual_seed(7)
    m = make_net(info, 'toy', 'tokens', 16, 1, heads=2, aux=TOY_AUX).cuda()
    ev = NetEvaluator(m, 'cuda', graphs=graphs)
    seat = G.nn_seat('m', 8, selfplay=True, sims_fast=4, p_full=0.5, leaf_value_view='root', view_samples=0.5,
                     belief_tilt=1.0)
    drv = G.Driver('toy', [[seat, seat]] * 24, concurrency=24, threads=2, groups=2, seed=5, record=True)
    seen = {'rows': 0, 'value_rows': 0}

    class Checked:
        def __call__(self, lb):
            r = ev(lb)
            validate_evaluation(lb, *r, aux_width=6)
            seen['rows'] += len(belief_rows(lb.a.get('extra')))
            seen['value_rows'] += int((lb.a['extra'] == ROW_VALUE).sum())
            return r
    drive(drv, {'m': Checked()})                  # (one batch at a time: every answer checked)
    res = drv.take_results()
    assert len(res) == 24 and all(r['over'] for r in res)
    assert seen['rows'] > 0 and seen['value_rows'] > 0
    assert ev.graph_calls == (ev.calls if graphs else 0) and ev.calls > 0
    # drive's pipeline (launch / result, two batches in flight)
    drv = G.Driver('toy', [[seat, seat]] * 24, concurrency=24, threads=2, groups=2, seed=6, record=True)
    calls = ev.calls
    drive(drv, {'m': ev})
    assert len(drv.take_results()) == 24 and ev.calls > calls
    assert ev.graph_calls == (ev.calls if graphs else 0)
    assert drv.stats()['belief_rows'] > 0
