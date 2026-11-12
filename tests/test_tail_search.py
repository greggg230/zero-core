"""Variable-length rows in the Python search and its consumers (docs/search.md 2.15, 6; docs/az.md 43.4): the Rust
tree driven from Python (`SearchTree.simulate(rows=True)`), the reference tree's leaves (`mcts.leaf_row`), the
Python driver (`play.PyDriver`, its 'py' and 'rs' seats), parity, `zero.replay`, `zero.targets quality` and the
lookahead's position API carry every row's tail and overrides exactly - on `toy_tail`, with a token network where a
network answers - or refuse loudly; a game without them takes the paths of before."""
import numpy as np
import pytest
import torch

from zero import _engine as E
from zero import games as G
from zero import lookahead as LA
from zero import mcts, net, package, parity, play, protocol, replay, targets
from zero import rows as R
from zero.infer import NetEvaluator

INFO = E.game_info_of('toy_tail')
LAY = package.TOY_TAIL_TOKENS
WIDTHS = G.tail_widths(dict(INFO))
RK = mcts.rows_of(dict(INFO))
SPEC = {'items0': 9}


def _consistent(x, tail):
    """A toy_tail row's tail matches its head: an item token per item past the 4 head slots, a mark token per mark
    (globals: items / 8 at float 3, marks / 4 at float 5), each token's position float its slot / 8."""
    seg = np.asarray(tail['t_seg'])
    slot = np.asarray(tail['t_slot'])
    val = np.asarray(tail['t_val'], np.float32)
    items, marks = int(round(float(x[3]) * 8)), int(round(float(x[5]) * 4))
    if int((seg == 0).sum()) != max(0, items - 4) or int((seg == 1).sum()) != marks:
        return False
    o = 0
    for s, l in zip(seg.tolist(), slot.tolist()):
        w = WIDTHS[s][0]
        if abs(val[o + (5 if s == 0 else 1)] - l / 8) > 1e-6:
            return False
        o += w
    return o == len(val)


def _same_tail(a, b):
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in ('t_seg', 't_slot', 't_val'))


def _check_ov(ov, k):
    opt = np.asarray(ov[0])
    assert opt.dtype == np.int32 and np.asarray(ov[1]).dtype == np.int64
    assert np.all(np.diff(opt) > 0) and (not len(opt) or (opt.min() >= 0 and opt.max() < k))


def test_search_tree_leaves_carry_their_rows():
    g = E.Game('toy_tail', SPEC, 2, 3)
    t = g.search_tree(True, 0)
    with pytest.raises(RuntimeError, match=r'rows=True'):
        t.simulate(g, 1.0, 0, 0.0, False)              # a head-only caller: refused, never a dropped tail
    first = t.simulate(g, 1.0, 0, 0.0, True, rows=True)
    p, x, acts, vx, (tail, ov, vt) = first
    # the root's leaf: the game's own row (`Game.encode_row`, `option_overrides`), as the reference tree's
    hx, ht = g.encode_row(p)
    assert np.array_equal(x, hx) and _same_tail(tail, ht) and vx is None and vt is None
    gx, gt, gov = mcts.leaf_row(g, p, RK, opts=True)
    assert np.array_equal(gx, x) and _same_tail(gt, tail)
    assert np.array_equal(gov[0], ov[0]) and np.array_equal(gov[1], ov[1]) and len(ov[0]) == 3    # TAKE of 4..6
    t.expand([1.0 / len(acts)] * len(acts), [0.5, 0.5], [0.0, 0.0])
    seen = {'tails': 0, 'vrows': 0}
    for _ in range(60):
        r = t.simulate(g, 1.0, 0, 0.0, True, rows=True)
        if r is None:
            continue
        p, x, acts, vx, (tail, ov, vt) = r
        assert _consistent(x, tail)
        _check_ov(ov, len(acts))
        seen['tails'] += len(tail['t_seg']) > 0
        if vx is not None:
            assert vt is not None and _consistent(vx, vt) and p != t.viewer
            seen['vrows'] += 1
        else:
            assert vt is None
        t.expand([1.0 / max(1, len(acts))] * len(acts), [0.5, 0.5], [0.0, 0.0])
    assert seen['tails'] > 20 and seen['vrows'] > 5, seen
    # a game without variable-length rows: the calls of before, and rows=True gives no rows
    g = E.Game('toy', None, 2, 3)
    t = g.search_tree(True, 0)
    assert len(t.simulate(g, 1.0, 0)) == 3
    t.expand([0.5] * len(g.legal()), [0.5, 0.5], [0.0, 0.0])
    r = t.simulate(g, 1.0, 0, 0.0, True, rows=True)
    assert r is None or (len(r) == 5 and r[4] is None)
    assert mcts.rows_of(dict(E.game_info_of('toy'))) is None


def test_rust_and_python_searches_agree_on_tail_rows():
    """docs/search.md 6: toy_tail positions (every row with a tail), the stand-in reading every tail token and
    override: the Rust tree's statistics at the Python-Python noise floor; fed without its rows' tails and
    overrides it is told apart at once (the rows matter to the check). (The completed-Q targets at c_scale 0.1: the
    stand-in's values are far apart on these positions, so its c_scale 1 targets are near one-hot and their TV is
    noise at 16 repetitions - docs/search.md 6.)"""
    r = parity.positions('toy_tail', n=8, sims=96, reps=16, spec=SPEC, cq=(50.0, 0.1))
    assert r['tv_rs_py'] < 1.6 * r['tv_py_py'] + 0.003, r
    assert r['win_diff'] < 0.01, r
    assert r['tv_cq_rs_py'] < 1.6 * r['tv_cq_py_py'] + 0.004, r
    p = parity.positions('toy_tail', n=6, sims=96, reps=16, spec=SPEC, drop_rows=True, cq=(50.0, 0.1))
    assert p['tv_rs_py'] > 10 * p['tv_py_py'] + 0.05, p
    assert p['tv_cq_rs_py'] > 10 * p['tv_cq_py_py'] + 0.05, p
    v = parity.root_view('toy_tail', n=6, reps=12, sims=64, spec=SPEC)
    assert v['tv_rs_py'] <= 3 * v['tv_py_py'] + 0.003, v
    assert v['win_diff'] < 0.01, v


def _echo(b):
    from test_tail_rows import echo
    return echo(b, b.x)


@pytest.mark.parametrize('impl', ['py', 'rs'])
def test_the_python_driver_carries_the_rows(impl):
    """Self-play with root-view values and value-only samples: every batch's rows, the value-only rows' and the
    records' tails and overrides are their states' (a row's tail matches its own head), resolve against their rows,
    and the samples carry them as the Rust driver's."""
    seat = play.NNSeat('m', sims=16, selfplay=True, impl=impl, leaf_value_view='root', view_samples=1.0, temp_moves=4)
    d = play.PyDriver('toy_tail', 6, lambda k: [seat, seat], spec={'items0': 8, 'max_moves': 16}, concurrency=6, seed=2,
                      record=True)
    n = {'rows': 0, 'tails': 0, 'ov': 0, 'value_rows': 0}
    while True:
        b = d.next_batch()
        if b is None:
            break
        protocol.validate_batch(b, 72, 16)
        assert b.tails == WIDTHS
        x, tail, ov = b.x, b.tail, b.ov
        R.resolve(tail, ov, b.opt_off, b.opt_action, LAY, present=R.presence(x, LAY))       # every pointer in its row
        for r in range(b.B):
            t = tail.take([r])
            assert _consistent(x[r], {'t_seg': t.seg, 't_slot': t.slot, 't_val': t.val})
        n['rows'] += b.B
        n['tails'] += int((tail.counts() > 0).sum())
        n['ov'] += ov.NO
        n['value_rows'] += int((b.extra == protocol.ROW_VALUE).sum())
        d.submit(b.id, *_echo(b))
    s = d.take_samples()
    protocol.validate_samples(s, 72, tails=WIDTHS)
    assert all(k in s for k in protocol.ROW_SAMPLE_KEYS)
    st = R.Tail.from_arrays(s, WIDTHS)
    for i in range(len(s['player'])):
        t = st.take([i])
        assert _consistent(s['x'][i], {'t_seg': t.seg, 't_slot': t.slot, 't_val': t.val})
    vo = np.diff(s['opt_off']) == 0
    assert vo.any() and (~vo).any()
    ovs = R.Overrides.from_arrays(s)
    assert not ovs.counts()[vo].any() and ovs.NO > 0
    res = d.take_results()
    assert len(res) == 6 and all(r['over'] and not r['aborted'] and 'abort_reason' not in r for r in res)
    assert n['tails'] > 0.3 * n['rows'] and n['ov'] > 0 and n['value_rows'] > 0, n
    assert d.aborted_row_tokens == 0


@pytest.mark.parametrize('impl', ['py', 'rs'])
def test_the_python_driver_aborts_a_game_over_the_row_limit(impl, capfd):
    seat = play.NNSeat('m', sims=8, selfplay=True, impl=impl)
    d = play.PyDriver('toy_tail', 3, lambda k: [seat, seat], spec={'items0': 70}, concurrency=3, seed=1, record=True)
    assert d.next_batch() is None                     # 66 tail tokens: capacity 71 > 64 at the first row
    rows = d.take_results()
    assert len(rows) == 3 and all(r['aborted'] and not r['over'] and r['abort_reason'] == 'max_row_tokens'
                                  and r['samples'] == 0 for r in rows)
    assert d.aborted_row_tokens == 3 and d.take_samples() is None
    assert 'max_row_tokens: game 0' in capfd.readouterr().err
    # 63 items: the root's row fits (capacity 64), a leaf after MARK does not (a prior favouring MARK walks there at
    # once) - aborted mid-search, after the rows it sent before
    d = play.PyDriver('toy_tail', 2, lambda k: [seat, seat], spec={'items0': 63}, concurrency=2, seed=1, record=True)
    sent = 0
    while True:
        b = d.next_batch()
        if b is None:
            break
        assert np.all(5 + b.tail.counts() <= 64)
        sent += b.B
        pri, v, m = _echo(b)
        for r in range(b.B):
            seg = slice(b.opt_off[r], b.opt_off[r + 1])
            p = np.where(b.opt_action[seg] == 12, 50.0, 1.0) * pri[seg]
            pri[seg] = p / p.sum()
        d.submit(b.id, pri, v, m)
    rows = d.take_results()
    assert sent > 0 and all(r['abort_reason'] == 'max_row_tokens' for r in rows), rows
    assert 'its leaf row has capacity 65 > 64' in capfd.readouterr().err


def _game_rows(impl, spec, view, n=3, sims=24):
    """Evaluation games of the Python driver (no noise, no temperature) against the row echo -> (results, per game
    its rows in order: the head, tail, overrides, options and kind, as bytes)."""
    seat = play.NNSeat('m', sims=sims, impl=impl, leaf_value_view=view)
    d = play.PyDriver('toy_tail', n, lambda k: [seat, seat], spec=spec, concurrency=n, seed=1)
    rows = {}
    while True:
        b = d.next_batch()
        if b is None:
            break
        x, t, o = b.x, b.tail, b.ov
        for r in range(b.B):
            tt, oo = t.take([r]), o.take([r])
            rows.setdefault(int(b.gid[r]), []).append(
                (x[r].tobytes(), tt.seg.tobytes(), tt.slot.tobytes(), tt.val.tobytes(), oo.opt.tobytes(),
                 oo.tok.tobytes(), b.opt_action[b.opt_off[r]:b.opt_off[r + 1]].tobytes(), int(b.extra[r])))
        d.submit(b.id, *_echo(b))
    res = sorted(d.take_results(), key=lambda r: r['gid'])
    return [(r['gid'], r['steps'], [float(s) for s in r['scores']], r['over']) for r in res], rows


@pytest.mark.parametrize('view', ['mover', 'root'])
def test_games_without_chance_are_identical_in_both_trees(view):
    """45 items, 20 moves: ADD is never legal (the list keeps at most 24), so nothing is left to chance and the
    searches are deterministic - the Python reference tree and the Rust tree send the same rows, byte for byte
    (head, tail, overrides; with root-view values the value-only rows too), in the same order, and play the same
    games (docs/search.md 6: positions without chance search identically)."""
    spec = {'items0': 45, 'max_moves': 20}
    py = _game_rows('py', spec, view)
    rs = _game_rows('rs', spec, view)
    assert py == rs
    assert all(r[3] for r in py[0]) and sum(len(v) for v in py[1].values()) > 900


def test_games_without_rows_take_the_paths_of_before():
    for impl in ('py', 'rs'):
        seat = play.NNSeat('m', sims=8, selfplay=True, impl=impl, leaf_value_view='root', view_samples=1.0)
        d = play.PyDriver('toy', 2, lambda k: [seat, seat], concurrency=2, seed=1, record=True)
        assert d.rk is None
        b = d.next_batch()
        assert b.tails is None and b.tail is None and b.ov is None
        assert not any(k.startswith(('t_', 'ov_')) for k in b.a)
        from zero.infer import UniformEvaluator, drive
        d.submit(b.id, *UniformEvaluator()(b))
        drive(d, {'m': UniformEvaluator()})
        s = d.take_samples()
        assert not any(k.startswith(('t_', 'ov_')) for k in s)
        assert all('abort_reason' not in r for r in d.take_results())
    la = E.Lookahead(E.Game('toy', None, 2, 1), {'plies': 1, 'worlds': 2}, seed=0)
    b = la.next_batch()
    assert not any(k.startswith(('t_', 'ov_')) for k in b)


class CountingEvaluator(NetEvaluator):
    """A NetEvaluator that counts the rows' tail tokens and overrides it was given."""
    seen = {'calls': 0, 'tail_tokens': 0, 'overrides': 0}

    def evaluate(self, x, n_players, opt_off, opt_action, extra=None, tail=None, ov=None, **kw):
        CountingEvaluator.seen['calls'] += 1
        CountingEvaluator.seen['tail_tokens'] += 0 if tail is None else tail.NT
        CountingEvaluator.seen['overrides'] += 0 if ov is None else ov.NO
        return super().evaluate(x, n_players, opt_off, opt_action, extra, tail=tail, ov=ov, **kw)


@pytest.fixture
def tail_ckpt(tmp_path, monkeypatch):
    """A small token network of toy_tail (its tail segments and pointer overrides), saved; evaluators counted."""
    torch.manual_seed(0)
    m = net.make_net(G.info('toy_tail'), 'toy_tail', 'tokens', hidden=16, blocks=1, heads=2)
    path = str(tmp_path / 'iter_0001.pt')
    net.save(m, path)
    CountingEvaluator.seen = {'calls': 0, 'tail_tokens': 0, 'overrides': 0}
    for mod in (replay, targets, parity):
        monkeypatch.setattr(mod, 'NetEvaluator', CountingEvaluator, raising=False)
    monkeypatch.setattr('zero.infer.NetEvaluator', CountingEvaluator)
    return path


def test_replay_records_tail_games_with_a_token_network(tail_ckpt):
    spec = {'items0': 7, 'max_moves': 14}
    reps = replay.record(tail_ckpt, opp='self', games=2, sims=8, spec=spec, chance_k=0, seed=1, log=lambda *a: None)
    assert len(reps) == 2
    for r in reps:
        assert r['format'] == replay.FORMAT and r['over'] and 'abort_reason' not in r and r['result'] is not None
        nn = [f for f in r['frames'] if f['k'] == 'nn']
        assert nn and all(sum(f['n']) == f['sims'] and len(f['acts']) == len(f['p0']) for f in nn)
        assert max(len(s['items']) for s in r['states']) > 4                  # rows with tails were searched
    assert CountingEvaluator.seen['tail_tokens'] > 0 and CountingEvaluator.seen['overrides'] > 0
    # a lookahead seat (the position API's rows, with their tails and overrides)
    CountingEvaluator.seen = {'calls': 0, 'tail_tokens': 0, 'overrides': 0}
    reps = replay.record(tail_ckpt, opp='self', games=1, sims=8, spec=spec, chance_k=0, seed=2, log=lambda *a: None,
                         lookahead={'plies': 2, 'worlds': 2, 'topk': 3})
    look = [f['look'] for f in reps[0]['frames'] if 'look' in f]
    assert look and all(f['options'] for f in look) and reps[0]['over']
    assert CountingEvaluator.seen['tail_tokens'] > 0 and CountingEvaluator.seen['overrides'] > 0


def test_the_lookahead_position_api_carries_tail_rows(tail_ckpt):
    m, _ = net.load(tail_ckpt)
    ev = CountingEvaluator(m)
    g = E.Game('toy_tail', SPEC, 2, 4)
    la = E.Lookahead(g, {'plies': 2, 'worlds': 3, 'topk': 3}, seed=1)
    assert la.info()['name'] == 'toy_tail'
    rounds = 0
    while True:
        b = la.next_batch()
        if b is None:
            break
        B = len(b['player'])
        assert b['t_off'].shape == (B + 1,) and b['ov_off'].shape == (B + 1,)
        protocol.validate_rows(b, B, np.diff(b['opt_off']), WIDTHS)
        tail, ov = R.Tail.from_arrays(b, WIDTHS), R.Overrides.from_arrays(b)
        R.resolve(tail, ov, b['opt_off'], b['opt_action'], LAY, present=R.presence(b['x'], LAY))
        for r in range(B):
            t = tail.take([r])
            assert _consistent(b['x'][r], {'t_seg': t.seg, 't_slot': t.slot, 't_val': t.val})
        out = ev.evaluate(b['x'], b['n_players'], b['opt_off'], b['opt_action'], b['extra'], tail=tail, ov=ov)
        la.submit(out[0], out[1])
        rounds += 1
    assert rounds > 0 and la.result()['options']
    r = LA.evaluate_position(g, ev, {'plies': 2, 'worlds': 3}, seed=1)
    assert r['options'] and r['rows'] > 0
    assert CountingEvaluator.seen['tail_tokens'] > 0 and CountingEvaluator.seen['overrides'] > 0
    # past the row limit (63 items: the root fits, a leaf after MARK has capacity 65): the position API raises
    la = E.Lookahead(E.Game('toy_tail', {'items0': 63}, 2, 4), {'plies': 1, 'worlds': 1}, seed=1)
    with pytest.raises(RuntimeError, match='max_row_tokens'):
        LA.drive(la, ev)


def test_targets_quality_and_parity_h2h_on_tail_games(tail_ckpt):
    r = targets.quality(tail_ckpt, game='toy_tail', spec=SPEC, positions=4, sims=8, reps=1, ref_sims=16,
                        c_scales=(1.0,), chance_k=0, verbose=False)
    assert r['positions'] == 4 and set(r['targets']) == {'prior', 'visits', 'cq c_scale 1', 'greedy Q'}
    assert CountingEvaluator.seen['tail_tokens'] > 0 and CountingEvaluator.seen['overrides'] > 0
    CountingEvaluator.seen = {'calls': 0, 'tail_tokens': 0, 'overrides': 0}
    h = parity.h2h(tail_ckpt, games=8, sims=8, conc=8, game='toy_tail', spec={'items0': 6, 'max_moves': 12},
                   verbose=False)
    assert h['games'] == 8 and 0.0 <= h['rust_score'] <= 1.0
    assert CountingEvaluator.seen['tail_tokens'] > 0
