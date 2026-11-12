"""Variable-length rows (docs/search.md 1.4, 2.15; docs/az.md 43): the bindings, `zero.rows`, the protocol and the
consumers of batches / samples - each carries a row's tail tokens and pointer overrides exactly, or refuses loudly
(never a silent drop). The game is `toy_tail` (crates/search/src/toy_tail.rs)."""
import json

import numpy as np
import pytest

from zero import _engine as E
from zero import games as G
from zero import package, protocol
from zero import rows as R

INFO = E.game_info_of('toy_tail')
LAY = package.TOY_TAIL_TOKENS
WIDTHS = G.tail_widths(dict(INFO))


def test_game_info_declares_the_rows():
    assert INFO['tails'] == [{'seg': 0, 'name': 'items', 'width': 16, 'head': 4},
                             {'seg': 1, 'name': 'marks', 'width': 4, 'head': 0}]
    assert INFO['option_pointers'] is True and INFO['max_row_tokens'] == 64 and INFO['head_tokens'] == 5
    assert INFO['num_features'] == 72 and INFO['num_actions'] == 16
    t = E.game_info_of('toy')
    assert (t['tails'], t['option_pointers'], t['max_row_tokens'], t['head_tokens']) == ([], False, None, 0)
    assert R.layout_tails(LAY) == {0: (16, 4, 0), 1: (4, 0, 1)}
    assert R.head_tokens(LAY) == 5 and R.seg_columns(LAY) == [1, 0]
    assert package.rows_problems(LAY, dict(INFO)) == []
    assert package.rows_problems(package.TOY_TOKENS, dict(t)) == []


def test_encode_refuses_a_tail_and_encode_row_carries_it():
    g = E.Game('toy_tail', {'items0': 9}, 2, 1)
    with pytest.raises(ValueError, match='use encode_row'):
        g.encode(0)
    x, t = g.encode_row(0)
    assert x.shape == (72,) and x.dtype == np.float32
    assert t['t_seg'].dtype == np.uint16 and t['t_slot'].dtype == np.uint32 and t['t_val'].dtype == np.float32
    assert t['t_slot'].tolist() == [4, 5, 6, 7, 8] and len(t['t_val']) == 5 * 16
    opt, tok = g.option_overrides(0)
    legal = g.legal()
    assert opt.dtype == np.int32 and tok.dtype == np.int64
    assert [legal[o] for o in opt] == [8, 9, 10] and tok.tolist() == R.pack_tok(0, 0, [4, 5, 6]).tolist()
    small = E.Game('toy_tail', {'items0': 2}, 2, 1)
    assert np.array_equal(small.encode(0), small.encode_row(0)[0])     # no tail: encode is the head


def test_a_python_driven_tree_refuses_tail_games():
    g = E.Game('toy_tail', None, 2, 0)
    t = g.search_tree(True, 0)
    with pytest.raises(RuntimeError, match='carry no tail'):
        t.simulate(g, 1.0, 0, 0.0, False)


def test_the_lookahead_refuses_tail_rows():
    g = E.Game('toy_tail', {'items0': 9}, 2, 0)
    with pytest.raises(Exception, match='tail rows not supported yet'):
        lk = E.Lookahead(g, {'plies': 1, 'worlds': 1}, seed=0)
        lk.next_batch()


def test_pack_and_rows_containers():
    tok = R.pack_tok([0, 3, 65535], [0, 1, 255], [0, 7, 2 ** 32 - 1])
    assert [list(a) for a in R.unpack_tok(tok)] == [[0, 3, 65535], [0, 1, 255], [0, 7, 2 ** 32 - 1]]
    w = {0: (2, 1, 0), 2: (1, 0, 1)}
    t = R.Tail([0, 2, 2, 3], np.array([0, 2, 0], np.uint16), np.array([1, 5, 3], np.uint32),
               np.array([1, 2, 3, 4, 5], np.float32), w)
    assert t.counts().tolist() == [2, 0, 1] and t.voff().tolist() == [0, 2, 3, 5]
    s = t.take([2, 0, 0])
    assert s.off.tolist() == [0, 1, 3, 5] and s.slot.tolist() == [3, 1, 5, 1, 5]
    assert s.val.tolist() == [4, 5, 1, 2, 3, 1, 2, 3]
    c = R.Tail.concat([t, s])
    assert c.N == 6 and c.NT == 8 and c.val.tolist() == [1, 2, 3, 4, 5, 4, 5, 1, 2, 3, 1, 2, 3]
    row, seg, slot, val = t.padded()
    assert row.tolist() == [0, 0, 2] and val.tolist() == [[1, 2], [3, 0], [4, 5]]
    tt = t.to_torch('cpu', pad_to=5)
    assert tt['seg'].tolist()[-2:] == [0, 0] and tt['slot'].tolist()[-2:] == [1, 1] and tt['row'].tolist()[-2:] == [0, 0]
    with pytest.raises(ValueError, match='not tail segments'):
        R.Tail([0, 1], np.array([1], np.uint16), np.array([3], np.uint32), np.zeros(2, np.float32), w)
    with pytest.raises(ValueError, match='floats'):
        R.Tail([0, 1], np.array([0], np.uint16), np.array([3], np.uint32), np.zeros(3, np.float32), w)
    with pytest.raises(ValueError, match='sorted'):
        R.Tail([0, 2], np.array([0, 0], np.uint16), np.array([3, 3], np.uint32), np.zeros(4, np.float32), w).check()
    with pytest.raises(ValueError, match='head slots'):
        R.Tail([0, 1], np.array([0], np.uint16), np.array([0], np.uint32), np.zeros(2, np.float32), w).check()
    o = R.Overrides([0, 1, 1, 3], np.array([2, 0, 1], np.int32), np.array([-1, 5, 6], np.int64))
    assert o.take([2, 0]).opt.tolist() == [0, 1, 2] and R.Overrides.concat([o, o]).N == 6


def _drive(spec, n=6, sparse=False, cache=0, seed=3):
    """toy_tail self-play with a row-echo evaluator: every batch validated and resolved; -> (batches, samples)."""
    seat = G.nn_seat(sims=16, selfplay=True, sims_fast=4, p_full=0.5, eval_cache=cache)
    d = G.Driver('toy_tail', [[seat, seat]] * n, spec=spec, concurrency=n, seed=seed, record=True, sparse=sparse)
    seen, parts = [], []
    while True:
        b = d.next_batch()
        if b is None:
            break
        if not sparse:
            protocol.validate_batch(b, 72, 16)
        x = b.x
        r = R.resolve(b.tail, b.ov, b.opt_off, b.opt_action, LAY, present=R.presence(x, LAY))
        seen.append((x.copy(), b.tail, b.ov, r))
        pri, v, m = echo(b, x)
        d.submit(b.id, pri, v, m)
        parts.append(d.take_samples())
    parts.append(d.take_samples())
    return seen, protocol.concat_samples(parts), d


def echo(b, x):
    """A row echo: priors favouring overridden options, a value from every head float, tail float and override."""
    B, off = b.B, b.opt_off
    tail, ov = b.tail, b.ov
    h = (x * ((np.arange(x.shape[1]) * 7 % 13) - 6.0)).sum(1)
    if tail is not None and tail.NT:
        row = np.repeat(np.arange(B), tail.counts())
        tw = tail.tok_width()
        fi = np.arange(len(tail.val)) - np.repeat(tail.voff()[:-1], tw)     # (a float's index in its token: the row alone)
        per = np.add.reduceat(tail.val * ((fi * 5 % 11) - 5.0), tail.voff()[:-1])
        per = np.where(tw > 0, per, 0.0)
        np.add.at(h, row, per * (1 + tail.slot % 3))
    pri = np.zeros(int(off[-1]), np.float32)
    orow = np.repeat(np.arange(B), ov.counts())
    for r in range(B):
        k = off[r + 1] - off[r]
        p = np.ones(k)
        p[ov.opt[orow == r]] = 2.0
        pri[off[r]:off[r + 1]] = p / p.sum()
    w = 1 / (1 + np.exp(-0.1 * h))
    v = np.zeros((B, 8), np.float32)
    v[:, 0], v[:, 1] = w, 1 - w
    return pri, v, np.zeros((B, 8), np.float32)


def test_batches_and_samples_carry_the_rows():
    seen, s, d = _drive({'items0': 7})
    assert any(t.NT for _, t, _, _ in seen) and any(o.NO for _, _, o, _ in seen)
    # resolved: a head target's place in the full axis, a tail target's n_tok + its index
    for x, t, o, r in seen:
        assert np.all((r['ref'] == -1) | (r['ref'] >= 1))
        assert np.all(r['ref'][o.tok >= 0] >= 0)
    protocol.validate_samples(s, 72, tails=WIDTHS)
    assert all(k in s for k in protocol.ROW_SAMPLE_KEYS)
    st = d.stats()
    assert st['tail_rows'] > 0 and st['tail_tokens'] >= st['tail_rows'] and st['aborted_row_tokens'] == 0
    # take_rows / concat_samples round trip, the tail cut with its rows
    n = len(s['player'])
    parts = [protocol.take_rows(s, a, min(n, a + 7), WIDTHS) for a in range(0, n, 7)]
    back = protocol.concat_samples(parts)
    for k in s:
        assert np.array_equal(np.asarray(back[k]), np.asarray(s[k])), k
    with pytest.raises(ValueError, match='tail widths'):
        protocol.take_rows(s, 0, 3)
    with pytest.raises(ValueError, match='some parts only'):
        protocol.concat_samples([parts[0], {k: v for k, v in parts[1].items() if not k.startswith('t_')}])


def test_records_split_and_arrays_equal_take_samples():
    seat = G.nn_seat(sims=12, selfplay=True)
    mk = lambda: G.Driver('toy_tail', [[seat, seat]] * 4, spec={'items0': 8}, concurrency=4, seed=5, record=True)

    def run(take):
        d = mk()
        out, rows = [], []
        while True:
            b = d.next_batch()
            if b is None:
                break
            d.submit(b.id, *echo(b, b.x))
            out.append(take(d))
        out.append(take(d))
        return out
    a = protocol.concat_samples(run(lambda d: d.take_samples()))
    recs = [r for r in run(lambda d: d.take_records()) if r is not None]
    b = protocol.concat_samples([r.arrays() for r in recs])
    for k in a:
        assert np.array_equal(a[k], b[k]), k
    assert 't_off' in b and 'ov_off' in b


def test_eval_cache_and_sparse_batches_keep_the_rows():
    _, s0, _ = _drive({'items0': 9}, cache=0)
    _, s1, d1 = _drive({'items0': 9}, cache=4096, sparse=True)
    assert d1.stats()['cache_hits'] > 0
    key = lambda s: sorted(json.dumps([s['x'][i].tolist(), s['pi'][s['opt_off'][i]:s['opt_off'][i + 1]].tolist()])
                           for i in range(len(s['player'])))
    assert key(s0) == key(s1)
    assert np.array_equal(np.sort(s0['t_val']), np.sort(s1['t_val']))


def test_the_row_limit_aborts_the_game_counted_and_logged(capfd):
    seat = G.nn_seat(sims=8, selfplay=True)
    d = G.Driver('toy_tail', [[seat, seat]] * 3, spec={'items0': 70}, concurrency=3, seed=1, record=True)
    assert d.next_batch() is None
    rows = d.take_results()
    assert len(rows) == 3 and all(r['aborted'] and not r['over'] and r['abort_reason'] == 'max_row_tokens'
                                  and r['samples'] == 0 for r in rows)
    assert d.stats()['aborted_row_tokens'] == 3
    assert 'max_row_tokens: game 0' in capfd.readouterr().err
    # lowered by the driver setting; never raised
    d = G.Driver('toy_tail', [[seat, seat]] * 2, spec={'items0': 9}, concurrency=2, seed=1, record=True,
                 max_row_tokens=9)
    assert d.next_batch() is None and all(r['abort_reason'] == 'max_row_tokens' for r in d.take_results())
    with pytest.raises(ValueError, match='only lower'):
        G.Driver('toy_tail', [[seat, seat]], spec={}, concurrency=1, max_row_tokens=65)


def test_games_without_rows_are_unchanged():
    seat = G.nn_seat(sims=8, selfplay=True)
    d = G.Driver('toy', [[seat, seat]] * 2, concurrency=2, seed=1, record=True)
    b = d.next_batch()
    assert b.tail is None and b.ov is None and b.tails is None
    assert not any(k.startswith(('t_', 'ov_')) for k in b.a)
    assert 'tail_rows' not in d.stats()
    r = d.take_results()
    assert all('abort_reason' not in x for x in r)


def test_package_refuses_a_stale_build():
    info = dict(INFO)
    with pytest.raises(package.RowsMismatch, match='head_tokens'):
        package.check_rows('toy_tail', dict(info, head_tokens=0))
    with pytest.raises(package.RowsMismatch, match='tail segments'):
        package.check_rows('toy_tail', dict(info, tails=info['tails'][:1]))
    with pytest.raises(package.RowsMismatch, match='max_row_tokens'):
        package.check_rows('toy_tail', dict(info, max_row_tokens=128))
    with pytest.raises(package.RowsMismatch, match='option_pointers'):
        package.check_rows('toy_tail', dict(info, option_pointers=False))
    package.check_rows('toy_tail', info)
    package.check_rows('toy', dict(E.game_info_of('toy')))


def test_resolve_refuses_bad_pointers():
    tail = R.Tail([0, 1], np.array([0], np.uint16), np.array([6], np.uint32), np.zeros(16, np.float32), WIDTHS)
    off, act = np.array([0, 3]), np.array([4, 10, 12])
    P = np.zeros((1, 5), bool)
    P[0, :3] = True                                                     # the global token, items slots 0 and 1
    ok = R.resolve(tail, R.Overrides([0, 2], np.array([0, 1], np.int32), np.array([R.pack_tok(0, 0, 1), R.pack_tok(0, 0, 6)])),
                   off, act, LAY, present=P)
    assert ok['ref'].tolist() == [2, 5] and ok['act'].tolist() == [4, 10] and ok['col'].tolist() == [0, 0]
    bad = [((3,), (R.pack_tok(0, 0, 1),), 'option 3'),
           ((0,), (R.pack_tok(0, 0, 3),), 'not present'),
           ((0,), (R.pack_tok(0, 0, 7),), 'does not have'),
           ((0,), (R.pack_tok(0, 1, 6),), 'column 1'),
           ((0,), (R.pack_tok(1, 0, 6),), 'column 0'),
           ((0,), (R.pack_tok(5, 0, 6),), 'segment 5')]
    for opt, tok, msg in bad:
        with pytest.raises(ValueError, match=msg):
            R.resolve(tail, R.Overrides([0, 1], np.array(opt, np.int32), np.array(tok, np.int64)), off, act, LAY, present=P)


def test_remote_workers_need_the_capability():
    from zero import remoteplay as RP
    assert RP._rows_of(dict(INFO)) and not RP._rows_of(dict(E.game_info_of('toy')))
    seen, s, _ = _drive({'items0': 6}, n=3)
    rows = [{'gid': 0, 'samples': 2}, {'gid': 1, 'samples': 0}, {'gid': 2, 'samples': len(s['player']) - 2}]
    parts = RP.split_games_rows(rows, s, WIDTHS)
    back = protocol.concat_samples([p for _, p in parts if p is not None])
    for k in s:
        assert np.array_equal(back[k], s[k]), k
