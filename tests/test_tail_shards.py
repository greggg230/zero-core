"""Shard format v2 and the training window with variable-length rows (docs/az.md 43): written sparse only, read back
exactly (float16 tail floats), refused where a reader would drop them (R7-R11), the window's tails and overrides
through its dense / sparse / cached / subsampled loads and its batches, the held-out set, distill's outputs,
bench's shard rows, the finite guard."""
import json
import os

import numpy as np
import pytest
import torch

from zero import _engine as E
from zero import data, games as G, package, protocol
from zero import rows as R
from zero.finite import shard_problems

INFO = dict(E.game_info_of('toy_tail'))
LAY = package.TOY_TAIL_TOKENS
WIDTHS = G.tail_widths(INFO)


def samples(n_games=6, seed=3, items0=7):
    from test_tail_rows import echo
    seat = G.nn_seat(sims=12, selfplay=True)
    d = G.Driver('toy_tail', [[seat, seat]] * n_games, spec={'items0': items0}, concurrency=n_games, seed=seed, record=True)
    parts = []
    while True:
        b = d.next_batch()
        if b is None:
            break
        d.submit(b.id, *echo(b, b.x))
        parts.append(d.take_samples())
    parts.append(d.take_samples())
    return protocol.concat_samples(parts)


@pytest.fixture(scope='module')
def S():
    return samples()


def write(tmp, s, name='a', rows=64, **meta):
    m = dict({'game': 'toy_tail', 'features': 72, 'actions': 16, 'iter': 1}, **data.shard_format({'shard_sparse': 1}, INFO), **meta)
    w = data.ShardWriter(str(tmp), name, rows=rows, meta=m)
    w.add(s)
    w.close()
    return sorted(data.shard_paths(str(tmp)))


def test_v2_shards_round_trip(tmp_path, S):
    paths = write(tmp_path, S)
    p = data.read_shard(paths[0])
    assert p['meta']['x_version'] == 2 and p['meta']['t_segs'] == [[0, 'items', 16, 4], [1, 'marks', 4, 0]]
    assert p['meta']['option_pointers'] is True and p['meta']['max_row_tokens'] == 64
    assert p['t_ptr'].dtype == np.int64 and p['t_seg'].dtype == np.uint16 and p['t_slot'].dtype == np.uint32
    assert p['t_val'].dtype == np.float16 and p['ov_ptr'].dtype == np.int64 and p['ov_tok'].dtype == np.int64
    allp = [data.read_shard(q) for q in paths]
    tails = [data.shard_tail(q, WIDTHS) for q in allp]
    t = R.Tail.concat([a for a, _ in tails])
    o = R.Overrides.concat([b for _, b in tails])
    ref = R.Tail.from_arrays(S, WIDTHS)
    assert np.array_equal(t.off, ref.off) and np.array_equal(t.seg, ref.seg) and np.array_equal(t.slot, ref.slot)
    assert np.array_equal(t.val, ref.val.astype(np.float16))
    assert np.array_equal(o.tok, S['ov_tok']) and np.array_equal(o.off, S['ov_off'])
    assert 't_ptr' not in data.read_shard(paths[0], head_only=True)


def test_refusals(tmp_path, S):
    # R8: dense shards with tails
    with pytest.raises(ValueError, match='sparse only'):
        data.shard_format({'shard_sparse': 0}, INFO)
    with pytest.raises(ValueError, match='format v2 writer'):
        data.ShardWriter(str(tmp_path / 'd'), 'x', meta={'game': 'toy_tail'}).add(S)
    with pytest.raises(ValueError, match='would be dropped'):
        data.write_shard(str(tmp_path / 'x.npz'), S, {'x_format': 'sparse'})
    with pytest.raises(ValueError, match='R8'):
        data.write_shard(str(tmp_path / 'x.npz'), S, {'t_segs': R.info_segs(INFO), 'option_pointers': True})
    paths = write(tmp_path / 'ok', S)
    # R10: the features alone
    with pytest.raises(ValueError, match='R10'):
        data.shard_features(paths[0])
    assert data.shard_features(paths[0], head_only=True).shape[1] == 72
    # R11: a window of a network without tails
    with pytest.raises(ValueError, match='R11'):
        data.Window(paths, 72, 16)
    # R9: other tail segments; a v1 shard for a network with tails
    other = dict(LAY, segments=[dict(LAY['segments'][0], width=17), LAY['segments'][1]])
    with pytest.raises(ValueError, match='R9'):
        data.Window(paths, 72, 16, layout=other)
    v1 = tmp_path / 'v1'
    toy = {k: v for k, v in S.items() if k not in protocol.ROW_SAMPLE_KEYS}
    w = data.ShardWriter(str(v1), 'b', meta={'game': 'toy_tail', 'x_format': 'sparse'})
    w.add(toy)
    w.close()
    with pytest.raises(ValueError, match='R9'):
        data.Window(data.shard_paths(str(v1)), 72, 16, layout=LAY)
    # R7: a newer version
    z = dict(np.load(paths[0]))
    meta = json.loads(str(z.pop('meta')))
    meta['x_version'] = 3
    bad = str(tmp_path / 'v3.npz')
    np.savez(bad, meta=np.array(json.dumps(meta)), **z)
    with pytest.raises(ValueError, match='up to version 2'):
        data.read_shard(bad)
    # convert: v2 stays sparse v2, never dense
    with pytest.raises(ValueError, match='never dense'):
        data.convert_shard(paths[0], str(tmp_path / 'c.npz'), sparse=False)
    data.convert_shard(paths[0], str(tmp_path / 'c.npz'), sparse=True, compress=1)
    a, b = data.read_shard(paths[0]), data.read_shard(str(tmp_path / 'c.npz'))
    assert b['meta']['x_version'] == 2
    for k in data.ROW_KEYS:
        assert np.array_equal(a[k], b[k]), k


def test_finite_guard_reads_tail_floats(tmp_path, S):
    s = dict(S, t_val=S['t_val'].copy())
    s['t_val'][3] = 1e6                                          # float16 overflow: inf
    with pytest.raises(data.NonFinite, match='t_val'):
        write(tmp_path, s)
    assert any(p.endswith('.bad.npz') for p in os.listdir(tmp_path))
    c = data.compact_rows(S, data.rows_meta(INFO))
    assert shard_problems(c) == []


@pytest.mark.parametrize('kind', ['dense', 'sparse', 'cache', 'sparse-cache', 'budget'])
def test_the_window_carries_the_rows(tmp_path, S, kind):
    paths = write(tmp_path, S, rows=50)
    kw = {'layout': LAY}
    if 'sparse' in kind:
        kw['sparse'] = True
    if 'cache' in kind:
        kw['cache'] = data.WindowCache()
    N = len(S['player'])
    if kind == 'budget':
        kw['budget_rows'] = N // 2
    W = data.Window(paths, 72, 16, **kw)
    if 'cache' in kind:
        W = data.Window(paths, 72, 16, **kw)                    # (the cached parts)
        assert W.reused == W.N and kw['cache'].nbytes > 0
    ref, ov = R.Tail.from_arrays(S, WIDTHS), R.Overrides.from_arrays(S)
    src = np.concatenate([data.shard_rows(p) * [0] + np.arange(data.shard_rows(p)) for p in paths])
    starts = np.concatenate([[0], np.cumsum([data.shard_rows(p) for p in paths])])
    sh = np.searchsorted(W.starts, np.arange(W.N), side='right') - 1
    rows = starts[sh] + W.src_row                               # each window row's sample row
    t = ref.take(rows)
    assert np.array_equal(W.tail.off, t.off) and np.array_equal(W.tail.slot, t.slot)
    assert np.array_equal(W.tail.val, t.val.astype(np.float16))
    assert np.array_equal(W.ov.tok, ov.take(rows).tok)
    assert W.nbytes > W.tail.nbytes()
    idx = np.arange(W.N)[::-1][:40]
    b = W.batch(idx, tokens=(R.present_index(LAY), 8))
    tb = t.take(idx)
    assert np.array_equal(b['tail']['val'].numpy(), tb.padded()[3].astype(np.float16).astype(np.float32))
    assert b['tail']['row'].tolist() == np.repeat(np.arange(len(idx)), tb.counts()).tolist()
    # the static axis: each row's head tokens, then its tail tokens at n_tok + j
    full, ntok = b['full'].numpy(), b['ntok'].numpy()
    P = R.presence(W.features(idx, np.float32), LAY)
    assert ntok.tolist() == (P.sum(1) + tb.counts()).tolist()
    j0 = np.concatenate([[0], np.cumsum(tb.counts())])
    for r in range(len(idx)):
        nh = int(P[r].sum())
        assert full[r, nh:ntok[r]].tolist() == list(range(5 + j0[r], 5 + j0[r + 1]))
    # the overrides resolved: a pointer to a tail token is n_tok + its index in the batch's tail
    o = b['ov']
    ovb = ov.take(rows).take(idx)
    assert len(o['ref']) == ovb.NO
    seg, col, slot = R.unpack_tok(ovb.tok[ovb.tok >= 0])
    tail_ref = o['ref'].numpy()[ovb.tok >= 0][slot >= 4]
    assert np.all(tail_ref >= 5)
    pr = np.repeat(np.arange(len(idx)), tb.counts())
    hit = tail_ref - 5
    assert np.array_equal(tb.slot[hit], slot[slot >= 4])
    # the eager batch carries the same and the prefetching feed too
    e = W.batch(idx)
    assert torch.equal(e['tail']['val'], b['tail']['val']) and torch.equal(e['ov']['ref'], b['ov']['ref'])
    feed = data.batches(W, np.random.default_rng(0), 2, 16, 'cpu', prefetch=2, tokens=(R.present_index(LAY), 8))
    for _, fb in feed:
        assert 'tail' in fb and 'ov' in fb


class FakeNet(torch.nn.Module):
    """A token network's interface with tails (the row echo of a network): records what it is fed."""
    def __init__(self, lay=LAY):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.features, self.seats, self.config = 72, 2, {'actions': 16, 'layout': lay, 'arch': 'tokens'}
        self.fed = []

    def forward(self, x, n, aux=False, tail=None, ov=None):
        self.fed.append((tail, ov))
        B = x.shape[0]
        return torch.zeros(B, 16) + self.w, torch.zeros(B, 2), torch.zeros(B, 2)


def test_distill_outputs_and_heldout_feed_the_rows(tmp_path, S):
    from zero import distill, heldout
    paths = write(tmp_path / 's', S)
    W = data.Window(paths, 72, 16, layout=LAY)
    m = FakeNet()
    distill.outputs(m, W, batch=32)
    assert m.fed and all(t is not None and o is not None for t, o in m.fed)
    assert sum(int(t['row'].numel()) for t, _ in m.fed) == W.tail.NT
    # a held-out set keeps its tails (made from v2 shards), the network's layout given at evaluation
    out = str(tmp_path / 'h' / 'set.npz')
    heldout.make(paths, out, rows=40, seed=1)
    h = data.read_shard(out)
    assert h['meta']['x_version'] == 2 and int(h['t_ptr'][-1]) > 0
    H = heldout.HeldOut([out])
    assert H.W.tail is not None
    with pytest.raises(ValueError, match='use_layout'):
        H.W.batch(np.arange(4))
    m2 = FakeNet()
    H.evaluate(m2)
    assert all(t is not None for t, _ in m2.fed)
    with pytest.raises(ValueError, match='R9'):
        H.W.use_layout(dict(LAY, segments=LAY['segments'][:1]))


def test_bench_shard_rows_carry_the_tails(tmp_path, S):
    from zero import bench
    paths = write(tmp_path, S)
    xs, tail = bench._shard_rows(paths)
    assert xs.shape == (len(S['player']), 72) and tail.N == len(S['player'])
    assert np.array_equal(tail.slot, S['t_slot'])


def test_shards_of_games_without_rows_are_unchanged(tmp_path):
    s = {'x': np.zeros((3, 4), np.float32), 'player': np.zeros(3, np.int32), 'n_players': np.full(3, 2, np.int32),
         'opt_off': np.array([0, 2, 4, 6], np.int64), 'opt_action': np.array([0, 1, 0, 1, 0, 1], np.int32),
         'pi': np.full(6, 0.5, np.float32), 'z': np.zeros((3, 8), np.float32), 'q': np.zeros((3, 8), np.float32),
         'sc': np.zeros((3, 8), np.float32), 'net': np.zeros(3, np.uint32)}
    assert data.shard_format({'shard_sparse': 1}, dict(E.game_info_of('toy'))) == {'x_format': 'sparse'}
    assert data.shard_format({'shard_sparse': 0}, dict(E.game_info_of('toy'))) == {}
    data.write_shard(str(tmp_path / 'a.npz'), s, {'x_format': 'sparse'})
    p = data.read_shard(str(tmp_path / 'a.npz'))
    assert p['meta']['x_version'] == 1 and not any(k in p for k in data.ROW_KEYS)
    data.write_shard(str(tmp_path / 'b.npz'), s, {})
    assert 'x_version' not in data.read_shard(str(tmp_path / 'b.npz'))['meta']
    W = data.Window([str(tmp_path / 'a.npz'), str(tmp_path / 'b.npz')], 4, 2)
    assert W.tail is None and W.ov is None and 'tail' not in W.batch(np.arange(3))
