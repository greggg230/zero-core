"""The exhaustive lookahead (docs/search.md 3.9): its settings (parse, validation: a bad or unknown
setting is an error, never ignored), the seat key `lookahead` in a driver, the position API
(`zero.lookahead.evaluate_position`, the bindings' `Lookahead`), zero.h2h's --a-lookahead / --b-lookahead /
--lookahead / --max-rows / --conc, NetEvaluator(max_rows=) - on the toy game; the lookahead off = the
driver and h2h of before."""
import json

import numpy as np
import pytest

from zero import games as G, h2h, lookahead as LA
from zero.infer import NetEvaluator, UniformEvaluator, drive

INFO = G.info('toy')


def nets(tmp_path, n=2):
    import torch
    from zero.net import Net, save
    out = []
    for k in range(n):
        torch.manual_seed(k)
        out.append(str(tmp_path / f'm{k}.pt'))
        save(Net(INFO['num_features'], INFO['num_actions'], hidden=32, blocks=1, game='toy').eval(), out[-1], {'iter': k})
    return out


def toy_net(seed=0):
    import torch
    from zero.net import Net
    torch.manual_seed(seed)
    return Net(INFO['num_features'], INFO['num_actions'], hidden=32, blocks=1, game='toy').eval()


# ------------------------------------------------------------------ settings

def test_settings_parse_and_check():
    d = LA.parse('plies=6,worlds=8,topk=0,opp=best,mode=infoset,budget=200000')
    assert d == {'plies': 6, 'worlds': 8, 'topk': 0, 'opp': 'best', 'mode': 'infoset', 'budget': 200000}
    s = LA.settings(d)
    assert s['opp'] == 'min' and s['plies'] == 6 and s['fallback'] == 'mcts' and s['dedup'] is True
    assert set(s) == set(LA.KEYS) - {'fallback_plies'}, 'every setting (fallback_plies only when given)'
    assert LA.parse('dedup=0') == {'dedup': False}
    assert LA.parse('plies=6,budget=100,fallback_plies=4/2')['fallback_plies'] == [4, 2]
    assert LA.parse('topk=3,topk_root=1')['topk_root'] is True
    assert LA.parse('{"plies": 2, "opp": "top"}') == {'plies': 2, 'opp': 'top'}
    assert LA.tag(d) == 'la-p6-w8-all-min-is-b200000'
    assert LA.tag({'horizon': 'next_turn', 'topk': 4, 'mode': 'world'}) == 'la-nt40-w8-k4-min-w'
    for bad in ('plys=3', 'plies=x', 'plies=0', 'opp=max', 'mode=perfect', 'topk_root=1', 'fallback_plies=4',
                'worlds=0', 'rows=0', 'horizon=game', 'plies', 'plies=4,plies=5', 'topk_root=yes', 'dedup=2'):
        with pytest.raises(ValueError):
            LA.parse(bad)
    with pytest.raises(ValueError, match='unknown setting'):
        LA.settings({'plies': 4, 'ply': 2})


def test_the_seat_key_is_checked_by_the_driver():
    seat = G.nn_seat('m', 8)
    with pytest.raises(ValueError, match='unknown setting'):
        G.Driver('toy', [[dict(seat, lookahead={'plys': 2}), seat]], concurrency=1)
    with pytest.raises(ValueError, match='evaluation seat'):
        G.Driver('toy', [[dict(G.nn_seat('m', 8, selfplay=True), lookahead={'plies': 2}), seat]], concurrency=1)
    # None is no lookahead: the seat of before
    G.Driver('toy', [[dict(seat, lookahead=None), seat]], concurrency=1)


# ------------------------------------------------------------------ the driver

def _match(look, games=6, threads=2, seed=5):
    seat = G.nn_seat('m', 8)
    la = dict(seat, lookahead=look) if look else seat
    drv = G.Driver('toy', [[la, seat] if g % 2 == 0 else [seat, la] for g in range(games)], concurrency=games, seed=seed,
                   threads=threads, move_log=400)
    drive(drv, {'m': UniformEvaluator()})
    return sorted(drv.take_results(), key=lambda r: r['gid']), drv.stats()


def test_a_lookahead_seat_plays_reproducibly_and_counts():
    rs, st = _match({'plies': 2, 'worlds': 2})
    assert len(rs) == 6 and all(r['over'] for r in rs)
    lk = st['look']
    assert lk['decisions'] > 0 and lk['nodes'] > 0 and lk['rows'] > 0 and lk['fallback_last'] == 0
    assert sum(lk['by_stage']) == lk['decisions']
    rs2, st2 = _match({'plies': 2, 'worlds': 2}, threads=4)
    assert [r['moves'] for r in rs] == [r['moves'] for r in rs2], 'the same games whatever the threads'
    assert st2['look']['nodes'] == lk['nodes']
    # without a lookahead: no 'look' counters (the stats of before)
    _, st0 = _match(None)
    assert 'look' not in st0


def test_the_budget_falls_back_to_mcts_in_the_driver():
    rs, st = _match({'plies': 3, 'worlds': 4, 'budget': 2})
    lk = st['look']
    assert len(rs) == 6 and lk['fallback_mcts'] == lk['fallback_last'] > 0
    assert st['sims'] > 0
    rs, st = _match({'plies': 3, 'worlds': 4, 'budget': 2, 'fallback': 'prior'})
    assert st['look']['fallback_last'] > 0 and st['look']['fallback_mcts'] == 0


# ------------------------------------------------------------------ the position API

def _toy_spot(seed=3):
    g = G.engine().Game('toy', None, 2, seed)
    while len(g.legal()) < 2:
        g.step(g.legal()[0])
    return g


def test_evaluate_position_on_the_toy_game():
    g = _toy_spot()
    ev = NetEvaluator(toy_net(), 'cpu')
    st = {'plies': 2, 'worlds': 3}
    r = LA.evaluate_position(g, ev, st, seeds=[1, 2, 3])
    acts = sorted(o['action'] for o in r['options'])
    assert acts == sorted(g.legal())
    assert all(o['n'] == 3 and 0 <= o['value'] <= 1 for o in r['options'])
    assert r['outcome'] == r['options'][0]['action']
    assert [o['value'] for o in r['options']] == sorted((o['value'] for o in r['options']), reverse=True)
    assert all(o['line'][0]['a'] == o['action'] for o in r['options'] if o.get('line'))
    again = LA.evaluate_position(g, ev, st, seeds=[1, 2, 3])
    assert [(o['action'], o['values']) for o in again['options']] == [(o['action'], o['values']) for o in r['options']]
    # the same worlds given explicitly: the same lookahead
    la = G.engine().Lookahead(g, st, seeds=[1, 2, 3])
    LA.drive(la, ev)
    w = LA.evaluate_position(g, ev, st, worlds=la.worlds())
    assert [(o['action'], o['values']) for o in w['options']] == [(o['action'], o['values']) for o in r['options']]
    # per world, top-k at the root
    pw = LA.evaluate_position(g, ev, dict(st, mode='world', topk=1, topk_root=True), seeds=[1, 2, 3])
    assert len(pw['options']) == 1
    # refusals
    with pytest.raises(ValueError):
        G.engine().Lookahead(g, {'plies': 2, 'worlds': 3}, seeds=[1, 2])
    with pytest.raises(ValueError):
        G.engine().Lookahead(g, {'plies': 2, 'worlds': 2}, worlds=la.worlds())


def test_the_protocol_refuses_answers_out_of_turn():
    g = _toy_spot()
    la = G.engine().Lookahead(g, {'plies': 1, 'worlds': 2}, seeds=[1, 2])
    b = la.next_batch()
    with pytest.raises(RuntimeError):
        la.next_batch()
    with pytest.raises(ValueError):
        la.submit(np.zeros(1, np.float32), np.zeros((1, 8), np.float32))
    pri, val, _ = UniformEvaluator()(type('LB', (), {'opt_off': b['opt_off'], 'B': len(b['player']),
                                                      'n_players': b['n_players']})())
    la.submit(pri, val)
    with pytest.raises(RuntimeError, match='not decided'):
        la.result()


# ------------------------------------------------------------------ zero.h2h and the evaluator

def test_h2h_lookahead_flags(tmp_path, capsys):
    pa, pb = nets(tmp_path)
    r = h2h.match(pa, pb, games=4, sims=4, chance_k=0, threads=2, paired=True,
                  seat_a={'lookahead': {'plies': 1, 'worlds': 2}})
    assert r['lookahead_a']['plies'] == 1 and r['lookahead_tag_a'] == 'la-p1-w2-all-min-is'
    assert r['look']['decisions'] > 0 and r['max_rows'] == 1024 and 'lookahead_b' not in r
    plain = h2h.match(pa, pb, games=4, sims=4, chance_k=0, threads=2, paired=True)
    assert not any(k.startswith('look') or k == 'max_rows' for k in plain), 'no lookahead: the output of before'
    h2h.main([pa, pb, '--games', '4', '--sims', '4', '--chance-k', '0', '--threads', '2', '--paired',
              '--b-lookahead', 'plies=1,worlds=2', '--conc', '2', '--max-rows', '64'])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out['lookahead_b']['worlds'] == 2 and out['max_rows'] == 64 and 'lookahead_a' not in out
    for bad in (['--a-lookahead', 'plys=1'], ['--lookahead', 'opp=max'], ['--conc', '0'], ['--max-rows', '0']):
        with pytest.raises(SystemExit):
            h2h.main([pa, pb, '--games', '2', '--sims', '4'] + bad)


def test_max_rows_pieces_answer_as_the_whole_batch():
    m = toy_net(1)
    whole, pieces = NetEvaluator(m, 'cpu'), NetEvaluator(m, 'cpu', max_rows=3)
    rng = np.random.default_rng(0)
    B = 10
    x = rng.random((B, INFO['num_features']), dtype=np.float32)
    n = np.full(B, 2, np.int32)
    cnt = rng.integers(0, 4, B)
    off = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int32)
    acts = rng.integers(0, INFO['num_actions'], int(off[-1])).astype(np.int32)
    a, b = whole.evaluate(x, n, off, acts), pieces.evaluate(x, n, off, acts)
    for u, v in zip(a, b):
        assert u.shape == v.shape and np.allclose(u, v, atol=1e-6)
    assert pieces.pieces == 1 and whole.pieces == 0
    with pytest.raises(ValueError):
        NetEvaluator(m, 'cpu', max_rows=0)


# ------------------------------------------------------------------ sparse batches (docs/search.md 2.2)

def test_a_sparse_driver_plays_the_same_games_as_a_dense_one():
    from zero.protocol import densify
    seat = G.nn_seat('m', 8, leaf_value_view='root')
    la = dict(G.nn_seat('m', 8), lookahead={'plies': 2, 'worlds': 2})
    ev = NetEvaluator(toy_net(2), 'cpu')
    out = {}
    for sparse in (False, True):
        drv = G.Driver('toy', [[la, seat], [seat, la]] * 2, concurrency=4, seed=9, threads=2, move_log=400, sparse=sparse)
        xs = []

        def on(lb, xs=xs):
            assert lb.sparse == sparse
            if sparse:
                assert lb.a['xs_off'][-1] == len(lb.a['xs_val']) and 0 < len(lb.a['xs_val']) < lb.B * INFO['num_features']
            xs.append(np.asarray(lb.x).copy())             # (a sparse batch densified: by the eager evaluator)
        drive(drv, {'m': ev}, on_batch=on)
        out[sparse] = (sorted(drv.take_results(), key=lambda r: r['gid']), xs)
    assert [r['moves'] for r in out[True][0]] == [r['moves'] for r in out[False][0]]
    assert len(out[True][1]) == len(out[False][1]) and all(np.array_equal(a, b) for a, b in zip(*(out[k][1] for k in (False, True))))
    assert np.array_equal(densify([0, 2, 2], [1, 3], [5.0, -0.0], 4), np.array([[0, 5, 0, -0.0], [0, 0, 0, 0]], np.float32))


@pytest.mark.skipif(not __import__('torch').cuda.is_available(), reason='CUDA graphs need CUDA')
def test_sparse_batches_through_the_token_graphs_answer_as_dense_ones():
    from test_net_data import tok_x
    from test_token_graphs import net
    from zero.protocol import LeafBatch, ROW_BELIEF
    m = net().cuda()
    dense, sparse = (NetEvaluator(m, "cuda", graphs=True, max_rows=64) for _ in range(2))
    rng = np.random.default_rng(1)
    B = 150
    x = tok_x(B, seed=3).numpy()
    x[rng.random(x.shape) < 0.5] = 0.0
    cnt = rng.integers(0, 4, B)
    cnt[cnt == 1] = 2
    off = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int32)
    acts = rng.integers(0, m.config['actions'], int(off[-1])).astype(np.int32)
    extra = np.zeros(B, np.uint8)
    extra[[3, 77]] = ROW_BELIEF
    n = np.full(B, 2, np.int32)
    nzr, nzc = np.nonzero(x)
    xs_off = np.concatenate([[0], np.cumsum(np.bincount(nzr, minlength=B))]).astype(np.int64)
    a = {'opt_off': off, 'opt_action': acts, 'player': np.zeros(B, np.int32), 'n_players': n, 'gid': np.zeros(B, np.int64),
         'extra': extra, 'xs_off': xs_off, 'xs_idx': nzc.astype(np.uint32), 'xs_val': x[nzr, nzc], 'n_features': x.shape[1]}
    lb = LeafBatch(0, 'm', 'nn', a)
    got = sparse.result(sparse.launch(lb))
    want = dense.evaluate(x, n, off, acts, extra)
    assert len(got) == len(want) == 4
    for u, v in zip(got, want):
        assert u.shape == v.shape and np.array_equal(u, v), np.abs(u - v).max()
    assert sparse.graph_calls == 3 and sparse.pieces == 1
