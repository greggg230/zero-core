"""Network, batched inference, sample shards and the training window, Elo."""
import os

import numpy as np
import torch

from zero import _engine as E
from zero import elo, games as G, protocol
from zero.data import ShardWriter, Window, read_shard, shard_paths
from zero.infer import NetEvaluator, UniformEvaluator, drive
from zero.net import Net, load, save, widen

INFO = E.game_info_of('toy')
F, A = INFO['num_features'], INFO['num_actions']


def a_batch(n=12, seed=0):
    seat = G.nn_seat('m', 8)
    drv = G.Driver('toy', [[seat, {'bot': 'random'}]] * n, concurrency=n, threads=1, seed=seed)
    return drv, drv.next_batch()


def test_net_masks_absent_seats():
    torch.manual_seed(0)
    m = Net(F, A, hidden=32, blocks=1).eval()
    x = torch.randn(4, F)
    n = torch.tensor([2, 3, 4, 2])
    p, v, mg = m(x, n)
    assert p.shape == (4, A) and v.shape == (4, protocol.MAX_SEATS) and mg.shape == (4, protocol.MAX_SEATS)
    sm = torch.softmax(v, 1)
    for b in range(4):
        assert torch.all(sm[b, n[b]:] < 1e-12) and abs(sm[b].sum().item() - 1) < 1e-5


def test_evaluator_matches_numpy_reference():
    torch.manual_seed(1)
    m = Net(F, A, hidden=32, blocks=1).eval()
    ev = NetEvaluator(m)
    drv, lb = a_batch()
    pri, val, mar = ev(lb)
    protocol.validate_evaluation(lb, pri, val, mar)
    with torch.no_grad():
        p, v, mg = m(torch.from_numpy(lb.x), torch.from_numpy(lb.n_players.astype(np.int64)))
    p = p.numpy()
    rows = np.repeat(np.arange(lb.B), np.diff(lb.opt_off))
    ref = protocol.segment_softmax(p[rows, lb.opt_action], lb.opt_off)
    assert np.allclose(pri, ref, atol=1e-6)
    assert np.allclose(val[:, :2], torch.softmax(v, 1).numpy()[:, :2], atol=1e-6)


def test_save_load_widen(tmp_path):
    torch.manual_seed(2)
    m = Net(F - 10, A - 2, hidden=32, blocks=2).eval()
    p = str(tmp_path / 'm.pt')
    save(m, p, {'iter': 3})
    # (a test network of no game of this build: the encoding guard's opt-out, as widen's callers)
    m2, extra = load(p, allow_mismatch=True)
    assert extra['iter'] == 3
    x = torch.randn(5, F)
    n = torch.full((5,), 2)
    w = widen(m2, F, A)
    with torch.no_grad():
        a, b = m2(x, n), w(x, n)
    assert torch.allclose(a[0], b[0][:, :A - 2], atol=1e-6) and torch.allclose(a[1], b[1], atol=1e-6)
    assert torch.all(b[0][:, A - 2:] == 0)


def test_shards_and_window_roundtrip(tmp_path):
    seat = G.nn_seat('m', 8, selfplay=True, sims_fast=4, p_full=0.5)
    drv = G.Driver('toy', [[seat, seat]] * 16, concurrency=16, threads=2, seed=4, record=True)
    w = ShardWriter(str(tmp_path / 'iter_0001'), 'c00', rows=150, meta={'game': 'toy'})
    parts = []

    def grab(lb):
        s = drv.take_samples()
        parts.append(s)
        w.add(s)
    drive(drv, {'m': UniformEvaluator()}, on_batch=grab)
    s = drv.take_samples()
    parts.append(s)
    w.add(s)
    st = w.close()
    full = protocol.concat_samples(parts)
    N = len(full['player'])
    assert st['samples'] == N and st['shards'] >= 2
    sh = read_shard(shard_paths(str(tmp_path / 'iter_0001'))[0])
    assert sh['meta']['game'] == 'toy' and sh['x'].dtype == np.float16
    win = Window(shard_paths(str(tmp_path / 'iter_0001')), F, A)
    assert win.N == N
    idx = np.arange(min(N, 40))
    b = win.batch(idx)
    assert b['x'].shape == (len(idx), F) and b['mask'].shape == (len(idx), A)
    assert torch.allclose(b['pi'].sum(1), torch.ones(len(idx)), atol=1e-3)
    # the dense targets reproduce the driver's sparse ones
    for k in range(5):
        lo, hi = full['opt_off'][k], full['opt_off'][k + 1]
        acts = full['opt_action'][lo:hi]
        assert set(np.nonzero(b['mask'][k].numpy())[0]) == set(acts.tolist())
        assert np.allclose(b['pi'][k].numpy()[acts], full['pi'][lo:hi], atol=2e-3)
        assert np.allclose(b['x'][k].numpy(), full['x'][k], atol=2e-3 * np.abs(full['x'][k]).max() + 1e-3)
    sub = Window(shard_paths(str(tmp_path / 'iter_0001')), F, A, budget_rows=N // 2)
    assert abs(sub.N - N // 2) <= 3
    # a narrower network reads a prefix
    assert Window(shard_paths(str(tmp_path / 'iter_0001')), F - 5, A).x.shape[1] == F - 5


def _window_loading_everything_first(paths, nf, budget, seed):
    """The window as it was built before it streamed (every shard in RAM, then subsampled): the
    rows, in order, of each shard that it keeps."""
    rng = np.random.default_rng(seed)
    parts = [read_shard(p) for p in paths]
    total = sum(len(p['player']) for p in parts)
    keep = 1.0 if not budget or total <= budget else budget / total
    out = []
    for p in parts:
        N = len(p['player'])
        idx = np.arange(N) if keep >= 1.0 else np.sort(rng.choice(N, size=max(1, int(round(N * keep))), replace=False))
        x = p['x'][idx][:, :nf]
        out.append((idx, np.pad(x, ((0, 0), (0, nf - x.shape[1]))), p['pi'], p['opt_off'], p['z'][idx]))
    return out


def test_window_streams_shards_with_the_same_rows(tmp_path):
    """The window subsamples each shard as it loads it (the rate from the shards' metas: the peak
    is the window plus one shard) and keeps exactly the rows of loading everything first; plain
    and compressed shards (remote self-play's) alike."""
    from zero.data import shard_rows
    rng = np.random.default_rng(5)
    paths = []
    for k, N in enumerate([300, 1, 777, 50, 1200]):
        nopt = rng.integers(2, 6, size=N)
        off = np.concatenate([[0], np.cumsum(nopt)])
        s = {'x': rng.standard_normal((N, F)).astype(np.float32), 'player': rng.integers(0, 2, N),
             'n_players': np.full(N, 2), 'opt_off': off, 'opt_action': rng.integers(0, A, off[-1]),
             'pi': rng.random(off[-1]).astype(np.float32), 'z': rng.random((N, 8)).astype(np.float32),
             'q': rng.random((N, 8)).astype(np.float32), 'sc': rng.random((N, 8)).astype(np.float32),
             'net': np.zeros(N, np.int32)}
        w = ShardWriter(str(tmp_path), f's{k}', rows=10 ** 6, meta={'game': 'toy'}, compress=k % 2)
        w.add(s)
        w.close()
        paths.append(str(tmp_path / f's{k}_000.npz'))
        assert shard_rows(paths[-1]) == N
    keys = ('x', 'player', 'n_players', 'opt_off', 'opt_action', 'pi', 'z', 'q', 'sc', 'src_row', 'starts')
    for budget in (None, 10 ** 6, 1000, 333, 2):
        for nf in (F, F - 7, F + 3):
            win = Window(paths, nf, A, budget_rows=budget, seed=3)
            ref = _window_loading_everything_first(paths, nf, budget, seed=3)
            assert np.array_equal(win.src_row, np.concatenate([r[0] for r in ref]))
            assert np.array_equal(win.x, np.concatenate([r[1] for r in ref]).astype(np.float16))
            assert np.array_equal(win.z[:, :2], np.concatenate([r[4] for r in ref]).astype(np.float32)[:, :2])
            assert win.loaded == 2328 and win.N == len(win.src_row)
            # the shards load in parallel; the window is the same whatever the number of threads
            for threads in (1, 3):
                other = Window(paths, nf, A, budget_rows=budget, seed=3, threads=threads)
                for k in keys:
                    a, b = getattr(win, k), getattr(other, k)
                    assert a.dtype == b.dtype and np.array_equal(a, b), (k, threads)
                assert (other.N, other.loaded, other.nbytes) == (win.N, win.loaded, win.nbytes)


def _random_shard(path_dir, name, N, rng, compress=0):
    nopt = rng.integers(2, 6, size=N)
    off = np.concatenate([[0], np.cumsum(nopt)])
    s = {'x': rng.standard_normal((N, F)).astype(np.float32), 'player': rng.integers(0, 2, N),
         'n_players': np.full(N, 2), 'opt_off': off, 'opt_action': rng.integers(0, A, off[-1]),
         'pi': rng.random(off[-1]).astype(np.float32), 'z': rng.random((N, 8)).astype(np.float32),
         'q': rng.random((N, 8)).astype(np.float32), 'sc': rng.random((N, 8)).astype(np.float32),
         'net': np.zeros(N, np.int32)}
    w = ShardWriter(str(path_dir), name, rows=10 ** 6, meta={'game': 'toy'}, compress=compress)
    w.add(s)
    w.close()
    return str(path_dir / f'{name}_000.npz')


def test_window_cache_is_the_same_window(tmp_path):
    """A WindowCache (run setting window_cache): windows sliding over shards load only their new
    shards and are the windows of loading everything, array for array and batch for batch; a
    rewritten shard or another network width loads again; a window over budget empties the cache
    and subsamples as without it."""
    import time
    from zero.data import WindowCache
    rng = np.random.default_rng(9)
    paths = [_random_shard(tmp_path, f's{k}', N, rng, compress=k % 2) for k, N in enumerate([400, 3, 250, 900, 60])]
    keys = ('x', 'player', 'n_players', 'opt_off', 'opt_action', 'pi', 'z', 'q', 'sc', 'src_row', 'starts')

    def same(a, b):
        for k in keys:
            u, v = getattr(a, k), getattr(b, k)
            assert u.dtype == v.dtype and np.array_equal(u, v), k
        assert (a.N, a.loaded, a.nbytes) == (b.N, b.loaded, b.nbytes)
        idx = np.random.default_rng(a.N).integers(a.N, size=97)
        ba, bb = a.batch(idx), b.batch(idx)
        for k in ba:
            assert torch.equal(ba[k], bb[k]), k

    cache = WindowCache()
    steps = [(paths[0:3], F, None, 0), (paths[1:4], F, None, 253), (paths[2:5], F, None, 1150),
             (paths[2:5], F, 10 ** 6, 1210), (paths[2:5], F - 7, None, 0), (paths[2:5], F - 7, 500, 0),
             (paths[1:4], F - 7, None, 0), (paths[1:4], F - 7, None, 1153)]
    for k, (ps, nf, budget, reused) in enumerate(steps):
        win = Window(ps, nf, A, budget_rows=budget, seed=k, cache=cache)
        same(win, Window(ps, nf, A, budget_rows=budget, seed=k))
        assert win.reused == reused, (k, win.reused)
        if budget == 500:                  # over budget: subsampled, the cache emptied
            assert win.N < 1210 and not cache.parts
        else:
            assert sorted(cache.parts) == sorted(ps) and cache.nbytes > 0
    # a shard written again under the same name is loaded again
    time.sleep(0.05)
    _random_shard(tmp_path, 's2', 250, np.random.default_rng(10))
    win = Window(paths[1:4], F - 7, A, seed=1, cache=cache)
    same(win, Window(paths[1:4], F - 7, A, seed=1))
    assert win.reused == 903


def test_batches_prefetched_are_the_same(tmp_path):
    """data.batches with prefetch (run setting train_prefetch): the same draws and batches in the
    same order as without; a failure while building is raised to the consumer; closing early
    stops the thread."""
    import threading
    from zero.data import batches
    rng0 = np.random.default_rng(4)
    paths = [_random_shard(tmp_path, f'p{k}', N, rng0, compress=k % 2) for k, N in enumerate([300, 41, 520])]
    win = Window(paths, F, A)
    ref = list(batches(win, np.random.default_rng(11), 25, 64))
    for pre in (1, 3):
        got = list(batches(win, np.random.default_rng(11), 25, 64, prefetch=pre))
        assert len(got) == len(ref)
        for (i0, b0), (i1, b1) in zip(ref, got):
            assert np.array_equal(i0, i1)
            assert b0.keys() == b1.keys() and all(torch.equal(b0[k], b1[k]) for k in b0)

    class Broken:
        N = win.N
        calls = 0

        def batch(self, idx, device='cpu'):
            Broken.calls += 1
            if Broken.calls == 4:
                raise ValueError('bad batch')
            return win.batch(idx, device)
    g = batches(Broken(), np.random.default_rng(1), 10, 8, prefetch=2)
    assert len([next(g) for _ in range(3)]) == 3
    try:
        next(g)
        raise AssertionError('no error')
    except ValueError as e:
        assert 'bad batch' in str(e)
    before = threading.active_count()
    g = batches(win, np.random.default_rng(2), 1000, 8, prefetch=2)
    next(g)
    g.close()
    assert not any(t.name == 'zero-batch-prefetch' and t.is_alive() for t in threading.enumerate())
    assert threading.active_count() <= before


def test_elo_fit_recovers_ratings():
    rng = np.random.default_rng(0)
    true = {'yard': 1000, 'a': 1200, 'b': 800, 'c': 1500}
    res = {'pairs': {}, 'anchor': 'yard'}
    names = list(true)
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = names[i], names[j]
            p = 1 / (1 + 10 ** (-(true[a] - true[b]) / 400))
            n = 4000
            elo.add(res, a, b, float(rng.binomial(n, p)), n)
    r = elo.fit(res)
    for k, v in true.items():
        assert abs(r[k][0] - v) < 40, (k, r[k])
    assert r['yard'][0] == 1000
    assert elo.has(res, 'b', 'a') and not elo.has(res, 'a', 'zz')


def test_webexport_bin_format(tmp_path):
    """zero.webexport: the .bin weight file the browser players' zero_webnet reads."""
    import json
    import struct

    from zero.webexport import export_model
    torch.manual_seed(0)
    path = str(tmp_path / 'm.pt')
    save(Net(INFO['num_features'], INFO['num_actions'], hidden=32, blocks=1, game='toy').eval(), path, {'iter': 3})
    n, worst = export_model(path, str(tmp_path / 'm.bin'))
    b = open(tmp_path / 'm.bin', 'rb').read()
    assert b[:4] == b'CVZW' and struct.unpack('<II', b[4:12])[0] == 1
    hl = struct.unpack('<II', b[4:12])[1]
    h = json.loads(b[12:12 + hl])
    assert (12 + hl) % 2 == 0 and h['iter'] == 3 and h['hidden'] == 32 and h['game'] == 'toy'
    assert not any(t['name'].startswith('mar.') for t in h['tensors'])
    assert len(b) == 12 + hl + 2 * n and worst < 1e-3


GRID = {'planes': 2, 'h': 4, 'w': 5, 'offset': 3, 'cell_action0': 2, 'hex': 'odd-r'}


def test_gridnet_heads_save_load_widen(tmp_path):
    from zero.net import GridNet, build
    torch.manual_seed(3)
    feats, acts = 3 + 2 * 4 * 5 + 7, 2 + 20 + 4
    m = GridNet(feats, acts, GRID, ch=8, blocks=2, head=16, ghid=16).eval()
    x = torch.randn(6, feats)
    n = torch.tensor([2, 3, 2, 4, 2, 2])
    with torch.no_grad():
        p, v, mg = m(x, n)
    assert p.shape == (6, acts) and v.shape == (6, protocol.MAX_SEATS) and mg.shape == (6, protocol.MAX_SEATS)
    sm = torch.softmax(v, 1)
    for b in range(6):
        assert torch.all(sm[b, n[b]:] < 1e-12)
    # a cell logit moves with its cell's planes, not with the other cells'
    x2 = x.clone()
    x2[:, 3 + 7] += 5.0                              # plane 0, cell 7 (row 1, col 2)
    with torch.no_grad():
        p2 = m(x2, n)[0]
    d = (p2 - p).abs()[:, 2:22]
    assert d[:, 7].min() > 0
    path = str(tmp_path / 'g.pt')
    save(m, path, {'iter': 1})
    m2, _ = load(path, allow_mismatch=True)               # (a test network: the guard's opt-out)
    assert type(m2).__name__ == 'GridNet' and m2.config == m.config
    with torch.no_grad():
        assert torch.allclose(m2(x, n)[0], p, atol=1e-6)
    w = widen(m2, feats + 9, acts + 3)
    xw = torch.cat([x, torch.randn(6, 9)], 1)
    with torch.no_grad():
        a, b = m2(x, n), w(xw, n)
    assert torch.allclose(a[0], b[0][:, :acts], atol=1e-5) and torch.allclose(a[1], b[1], atol=1e-5)
    assert torch.all(b[0][:, acts:] == 0)
    assert type(build(Net(F, A, 8, 1).config)).__name__ == 'Net'


def test_gridnet_rejects_a_grid_that_does_not_fit():
    import pytest
    from zero.net import GridNet
    with pytest.raises(ValueError):
        GridNet(10, 30, GRID)


def test_gridnet_plane_segments():
    """Planes in two blocks (a layout appending planes after its globals) = the same network as
    one block over the planes gathered in that order."""
    from zero.net import GridNet
    torch.manual_seed(4)
    cells = 4 * 5
    feats = 3 + 2 * cells + 7 + 1 * cells            # globals 3, planes 2, globals 7, plane 1
    grid = {'h': 4, 'w': 5, 'segments': [[3, 2], [3 + 2 * cells + 7, 1]], 'cell_action0': 2, 'planes': 3}
    m = GridNet(feats, 2 + cells + 4, grid, ch=8, blocks=1, head=16, ghid=16).eval()
    assert m.config['grid']['planes'] == 3 and m.config['grid']['offset'] == 3
    x = torch.randn(5, feats)
    n = torch.full((5,), 2)
    # the same weights on the reordered input: [3 globals, planes 0-1, plane 2, 7 globals]
    one = GridNet(feats, 2 + cells + 4, {'h': 4, 'w': 5, 'planes': 3, 'offset': 3, 'cell_action0': 2}, ch=8, blocks=1,
                  head=16, ghid=16).eval()
    one.load_state_dict(m.state_dict())
    o2 = 3 + 2 * cells
    xr = torch.cat([x[:, :o2], x[:, o2 + 7:], x[:, o2:o2 + 7]], 1)
    with torch.no_grad():
        a, b = m(x, n), one(xr, n)
    assert all(torch.allclose(u, v, atol=1e-5) for u, v in zip(a, b))
    w = widen(m, feats + 6)
    with torch.no_grad():
        c = w(torch.cat([x, torch.randn(5, 6)], 1), n)
    assert all(torch.allclose(u, v, atol=1e-5) for u, v in zip(a, c))


def test_segment_softmax_np_matches_the_reference():
    from zero.infer import segment_softmax_np
    rng = np.random.default_rng(5)
    cnt = rng.integers(1, 9, size=50)
    off = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
    lg = rng.normal(size=off[-1]).astype(np.float32) * 5
    assert np.allclose(segment_softmax_np(lg, off), protocol.segment_softmax(lg, off), atol=1e-6)


def test_graph_runner_matches_eager_and_follows_new_weights():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip('CUDA graphs need CUDA')
    from zero.net import GridNet
    torch.manual_seed(6)
    feats, acts = 3 + 2 * 4 * 5 + 7, 2 + 20 + 4
    m = GridNet(feats, acts, GRID, ch=16, blocks=2, head=16, ghid=16)
    m.train()
    with torch.no_grad():                             # non-trivial BatchNorm statistics
        for _ in range(3):
            m(torch.randn(64, feats), torch.full((64,), 2))
    m.eval()
    rng = np.random.default_rng(7)
    B = 70                                            # padded to 128 inside
    x = rng.normal(size=(B, feats)).astype(np.float32)
    npl = np.array([2, 3] * 35, np.int32)
    cnt = rng.integers(1, 6, size=B)
    off = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int32)
    opt = np.concatenate([np.sort(rng.choice(acts, size=c, replace=False)) for c in cnt]).astype(np.int32)
    eager = NetEvaluator(m, 'cuda', graphs=False)
    fast = NetEvaluator(copy_of(m), 'cuda', graphs=True)
    assert fast.fast is not None and eager.fast is None
    for _ in range(2):
        pe, ve, me = eager.evaluate(x, npl, off, opt)
        pf, vf, mf = fast.evaluate(x, npl, off, opt)
        assert np.abs(pe - pf).max() < 0.03 and np.abs(ve - vf).max() < 0.03 and np.abs(me - mf).max() < 0.1
        assert np.all(vf[npl == 2][:, 2:] == 0)
        # new weights (rolling self-play): both evaluators take them; the graphs read them in place
        new = {k: (v + 0.05 * torch.randn_like(v) if v.is_floating_point() and 'running' not in k else v)
               for k, v in m.state_dict().items()}
        eager.set_weights(new)
        fast.set_weights(new)


def copy_of(m):
    import copy
    return copy.deepcopy(m)


def test_gridnet_widen_adds_planes():
    """A grid with a new plane block after the old ones: the widened net computes what the old
    one did, whatever the new plane holds (its first-conv weights are zero)."""
    from zero.net import GridNet
    torch.manual_seed(8)
    cells = 4 * 5
    feats = 3 + 2 * cells + 7
    old = GridNet(feats, 2 + cells + 4, GRID, ch=8, blocks=1, head=16, ghid=16).eval()
    feats2 = feats + cells + 2                       # + one plane, + 2 globals
    grid2 = dict(GRID, planes=3, segments=[[3, 2], [feats, 1]])
    w = widen(old, feats2, grid=grid2)
    assert w.config['grid']['planes'] == 3
    x = torch.randn(4, feats)
    n = torch.full((4,), 2)
    x2 = torch.cat([x, torch.randn(4, cells + 2)], 1)
    with torch.no_grad():
        a, b = old(x, n), w(x2, n)
    assert all(torch.allclose(u, v, atol=1e-5) for u, v in zip(a, b))
    import pytest
    with pytest.raises(ValueError):
        widen(old, feats2, grid=dict(GRID, planes=3, segments=[[feats, 1], [3, 2]]))


# ---------------------------------------------------------------- arch 'tokens' (net.TokenNet)
# A synthetic object layout: 5 globals, a context slot (always present, identity one-hot of 12
# at 1), 2 bags of 12, 4 "units" (width 9: present at 0, identity as digits radix [3, 4] at 1,
# actions 3 + slot and 20 + 2 * slot), 3 "cards" (width 4: present at 3, no identity, action
# 30 + slot), then 2 more globals.
TOK_LAYOUT = {'vocab': 12,
              'segments': [{'name': 'ctx', 'offset': 5, 'count': 1, 'width': 14, 'present': None,
                            'id': {'at': 1, 'radix': [12]}},
                           {'name': 'units', 'offset': 19 + 24, 'count': 4, 'width': 9, 'present': 0,
                            'id': {'at': 1, 'radix': [3, 4]}, 'actions': [[3, 1], [20, 2]]},
                           {'name': 'cards', 'offset': 19 + 24 + 36, 'count': 3, 'width': 4, 'present': 3,
                            'actions': [[30, 1]]}],
              'bags': [{'name': 'zones', 'offset': 19, 'count': 2, 'width': 12}]}
TOK_F, TOK_A = 19 + 24 + 36 + 12 + 2, 40


def tok_x(B, seed=0):
    """Random features with valid presence flags and identity digits."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, TOK_F, generator=g)
    x[:, 6:18] = 0
    x[torch.arange(B), 6 + torch.randint(12, (B,), generator=g)] = 1.0           # context identity
    x[:, 19:43] = torch.randint(0, 3, (B, 24), generator=g).float() * 0.25        # bag counts
    for s in range(4):
        o = 43 + 9 * s
        x[:, o] = (torch.rand(B, generator=g) < 0.6).float()
        x[:, o + 1:o + 8] = 0
        x[torch.arange(B), o + 1 + torch.randint(3, (B,), generator=g)] = 1.0
        x[torch.arange(B), o + 4 + torch.randint(4, (B,), generator=g)] = 1.0
    for s in range(3):
        x[:, 79 + 4 * s + 3] = (torch.rand(B, generator=g) < 0.5).float()
    return x


def tok_net(seed=0, policy=True, **kw):
    """A small TokenNet; policy: random policy output layers (they start at zero: a uniform prior)."""
    from zero.net import TokenNet
    torch.manual_seed(seed)
    m = TokenNet(TOK_F, TOK_A, TOK_LAYOUT, **dict(dict(d=16, layers=2, heads=2, ff=32, head=16), **kw)).eval()
    if policy:
        with torch.no_grad():
            for lin in (m.gen, m.ptr, m.val[-1], m.mar[-1]):
                lin.weight.normal_(0, 0.3)
                lin.bias.normal_(0, 0.3)
    return m


def test_tokennet_policy_starts_uniform():
    """A new TokenNet's policy is uniform (its output layers start at zero); its value is not."""
    m = tok_net(16, policy=False)
    with torch.no_grad():
        p, v, mg = m(tok_x(5, 6), torch.tensor([2, 3, 2, 4, 2]))
    assert torch.all(p == 0)
    assert v[:, :2].std() > 0 and mg.abs().sum() > 0


def test_tokennet_heads_masks_and_chunks():
    m = tok_net(10)
    x = tok_x(37, 1)
    n = torch.tensor([2, 3, 4] * 12 + [2])
    with torch.no_grad():
        p, v, mg = m(x, n)
        assert p.shape == (37, TOK_A) and v.shape == (37, protocol.MAX_SEATS) and mg.shape == (37, protocol.MAX_SEATS)
        sm = torch.softmax(v, 1)
        assert all(torch.all(sm[b, n[b]:] < 1e-12) for b in range(37))
        # rows are independent of the batch and of the chunking (packing, sorting, padding)
        m.chunk = 5
        p5 = m(x, n)
        m.chunk = 1000
        p1 = m(x, n)
        one = [m(x[b:b + 1], n[b:b + 1]) for b in range(37)]
    for a, b in zip(p5, p1):
        assert torch.allclose(a, b, atol=1e-5)
    for k in range(3):
        assert torch.allclose(torch.cat([o[k] for o in one]), p1[k], atol=1e-5)
    # an absent slot is masked: its floats (all but the present flag) change nothing
    x2 = x.clone()
    for s in range(4):
        absent = x[:, 43 + 9 * s] < 0.5
        x2[absent, 43 + 9 * s + 1:43 + 9 * s + 9] = torch.randn(int(absent.sum()), 8)
    for s in range(3):
        absent = x[:, 79 + 4 * s + 3] < 0.5
        x2[absent, 79 + 4 * s:79 + 4 * s + 3] += 3.0
    with torch.no_grad():
        q = m(x2, n)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(p1, q))
    # a present slot is read
    x3 = x.clone()
    x3[:, 43 + 7] += 2.0
    with torch.no_grad():
        assert not torch.allclose(m(x3, n)[1], p1[1])


def test_tokennet_pointer_logits_land_on_the_slot_actions():
    m = tok_net(11)
    with torch.no_grad():
        m.gen.weight.zero_()
        m.gen.bias.zero_()
        m.ptr.weight.zero_()
        m.ptr.bias.copy_(torch.tensor([1.0, 2.0, 3.0]))   # columns: units (3 + s), units (20 + 2 s), cards (30 + s)
        x = tok_x(9, 2)
        p = m(x, torch.full((9,), 2))[0]
    for b in range(9):
        want = torch.zeros(TOK_A)
        for s in range(4):
            if x[b, 43 + 9 * s] > 0.5:
                want[3 + s], want[20 + 2 * s] = 1.0, 2.0
        for s in range(3):
            if x[b, 79 + 4 * s + 3] > 0.5:
                want[30 + s] = 3.0
        assert torch.equal(p[b], want), (b, p[b], want)


def test_tokennet_identity_digits():
    m = tok_net(12)
    rows = torch.zeros(5, 9)
    for k, (a, b) in enumerate([(0, 0), (2, 3), (1, 2), (2, 0)]):
        rows[k, 1 + a] = 1.0
        rows[k, 4 + b] = 1.0
    rows[4, 1] = 1.0                                      # a missing digit: no identity
    ids = m._identity(rows, TOK_LAYOUT['segments'][1]['id'])
    assert ids.tolist() == [0, 11, 6, 8, 12]
    # the identity is read through the shared embedding: two slots with the same identity and
    # the same floats get the same token
    x = tok_x(1, 3)
    x[0, 43:91] = 0
    for s in (0, 1):
        o = 43 + 9 * s
        x[0, o], x[0, o + 2], x[0, o + 5] = 1.0, 1.0, 1.0
    with torch.no_grad():
        h, n, full = m.pack(x)
    assert int(n[0]) == 4 and full[0, :4].tolist() == [0, 1, 2, 3]
    assert torch.allclose(h[0, 2] - m.pos[2], h[0, 3] - m.pos[3], atol=1e-6)


def test_tokennet_save_load_widen_train_step(tmp_path):
    from zero.net import build
    m = tok_net(13)
    x = tok_x(8, 4)
    n = torch.full((8,), 2)
    with torch.no_grad():
        p = m(x, n)
    path = str(tmp_path / 't.pt')
    save(m, path, {'iter': 2})
    m2, extra = load(path, allow_mismatch=True)           # (a test network: the guard's opt-out)
    assert type(m2).__name__ == 'TokenNet' and m2.config == m.config and extra['iter'] == 2
    assert type(build(m.config)).__name__ == 'TokenNet'
    with torch.no_grad():
        assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(m2(x, n), p))
        w = widen(m2, TOK_F + 3, TOK_A + 5)
        q = w(torch.cat([x, torch.randn(8, 3)], 1), n)
    assert torch.allclose(q[0][:, :TOK_A], p[0], atol=1e-5) and torch.all(q[0][:, TOK_A:] == 0)
    assert torch.allclose(q[1], p[1], atol=1e-5) and torch.allclose(q[2], p[2], atol=1e-5)
    # a training step moves every parameter group that the batch reaches
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    plog, vlog, mp = m(x, n)
    loss = plog.logsumexp(1).mean() + vlog.logsumexp(1).mean() + mp.pow(2).mean()
    loss.backward()
    for k in ('emb.weight', 'g.weight', 'bag.weight', 'seg_in.1.weight', 'pos', 'blocks.0.qkv.weight', 'ptr.weight'):
        assert dict(m.named_parameters())[k].grad.abs().sum() > 0, k
    opt.step()


def test_tokennet_layout_validation():
    import pytest
    from zero.net import TokenNet, tokens_layout
    lay = tokens_layout(TOK_LAYOUT, TOK_F, TOK_A)
    assert [s['name'] for s in lay['segments']] == ['ctx', 'units', 'cards'] and lay['bags'][0]['count'] == 2
    bad = [dict(TOK_LAYOUT, segments=TOK_LAYOUT['segments'] + [{'name': 'x', 'offset': 50, 'count': 1, 'width': 3}]),
           dict(TOK_LAYOUT, segments=[dict(TOK_LAYOUT['segments'][2], actions=[[3, 1]])] + TOK_LAYOUT['segments'][1:2]),
           dict(TOK_LAYOUT, segments=[dict(TOK_LAYOUT['segments'][2], actions=[[38, 1]])]),
           dict(TOK_LAYOUT, segments=[dict(TOK_LAYOUT['segments'][1], id={'at': 4, 'radix': [3, 4]})]),
           dict(TOK_LAYOUT, bags=[{'name': 'z', 'offset': 19, 'count': 2, 'width': 13}])]
    for b in bad:
        with pytest.raises(ValueError):
            tokens_layout(b, TOK_F, TOK_A)
    with pytest.raises(ValueError):
        TokenNet(TOK_F, TOK_A, TOK_LAYOUT, d=10, heads=4)


# The same layout with actions by identity: a unit slot also scores action 7 + its identity
# ([7, 19) holds no slot action of the layout).
TOK_LAYOUT_ID = dict(TOK_LAYOUT, segments=[TOK_LAYOUT['segments'][0], dict(TOK_LAYOUT['segments'][1], id_actions=7),
                                           TOK_LAYOUT['segments'][2]])


def test_tokennet_actions_by_identity():
    """`id_actions` (MTG Zero's team draft: a card's pick is its identity): a present slot scores
    base + its identity from its token; the other actions are unchanged; a layout without the key
    builds the network it did before (no identity column, the normalised layout unchanged)."""
    import pytest
    from zero.net import TokenNet, tokens_layout
    plain = tokens_layout(TOK_LAYOUT, TOK_F, TOK_A)
    assert all('id_actions' not in sg for sg in plain['segments'])
    m0 = tok_net(21)
    assert m0.n_id == 0 and m0.ptr.out_features == 3 and not hasattr(m0, 'tok_isid')
    lay = tokens_layout(TOK_LAYOUT_ID, TOK_F, TOK_A)
    assert [sg.get('id_actions') for sg in lay['segments']] == [None, 7, None]
    torch.manual_seed(21)
    m = TokenNet(TOK_F, TOK_A, TOK_LAYOUT_ID, d=16, layers=2, heads=2, ff=32, head=16).eval()
    assert m.n_id == 1 and m.ptr.out_features == 4 and m.config['layout']['segments'][1]['id_actions'] == 7
    x = tok_x(11, 5)
    n = torch.full((11,), 2)
    with torch.no_grad():
        m.gen.weight.zero_()
        m.gen.bias.zero_()
        m.ptr.weight.zero_()
        m.ptr.bias.copy_(torch.tensor([1.0, 2.0, 4.0, 3.0]))   # units (3 + s), units (20 + 2 s), units by identity, cards
        p = m(x, n)[0]
        m.chunk = 3
        p3 = m(x, n)[0]
    assert torch.equal(p, p3)
    seg = TOK_LAYOUT['segments'][1]
    for b in range(11):
        want = torch.zeros(TOK_A)
        for sl in range(4):
            o = 43 + 9 * sl
            if x[b, o] > 0.5:
                want[3 + sl], want[20 + 2 * sl] = 1.0, 2.0
                want[7 + int(m._identity(x[b, o:o + 9][None], seg['id'])[0])] = 4.0
        for sl in range(3):
            if x[b, 79 + 4 * sl + 3] > 0.5:
                want[30 + sl] = 3.0
        assert torch.equal(p[b], want), (b, p[b], want)
    # the identity column learns (its pointer weights get a gradient through the actions)
    m.train()
    plog = m(x, n)[0]
    plog[:, 7:19].logsumexp(1).sum().backward()
    assert m.ptr.weight.grad[2].abs().sum() > 0 and m.ptr.weight.grad[0].abs().sum() == 0
    # validation: an identity is needed, the base inside the actions, no slot action in its range
    for bad in (dict(TOK_LAYOUT, segments=[dict(TOK_LAYOUT['segments'][2], id_actions=0)]),
                dict(TOK_LAYOUT, segments=[dict(TOK_LAYOUT['segments'][1], id_actions=TOK_A)]),
                dict(TOK_LAYOUT, segments=[dict(TOK_LAYOUT['segments'][1], id_actions=0)]),
                dict(TOK_LAYOUT, segments=TOK_LAYOUT_ID['segments'][1:2] + [dict(TOK_LAYOUT['segments'][2], actions=[[10, 1]])])):
        with pytest.raises(ValueError):
            tokens_layout(bad, TOK_F, TOK_A)


def test_tokennet_actions_by_identity_save_load_widen(tmp_path):
    from zero.net import TokenNet
    torch.manual_seed(22)
    m = TokenNet(TOK_F, TOK_A, TOK_LAYOUT_ID, d=16, layers=1, heads=2, ff=32, head=16).eval()
    with torch.no_grad():
        m.ptr.weight.normal_(0, 0.3)
        m.ptr.bias.normal_(0, 0.3)
    x, n = tok_x(6, 7), torch.full((6,), 2)
    path = str(tmp_path / 'id.pt')
    save(m, path, {'iter': 1})
    m2, _ = load(path, allow_mismatch=True)               # (a test network: the guard's opt-out)
    with torch.no_grad():
        assert all(torch.equal(a, b) for a, b in zip(m(x, n), m2(x, n)))
        w = widen(m2, TOK_F + 2, TOK_A + 4)
        q = w(torch.cat([x, torch.zeros(6, 2)], 1), n)
        p = m(x, n)
    assert torch.allclose(q[0][:, :TOK_A], p[0], atol=1e-5)


# A wider version of TOK_LAYOUT (a layout change as MTG Zero's v4 -> v5, docs/az.md 10.3): the
# same tokens; the units' slots 3 floats wider and the cards' 2 (at the slots' ends), a third
# row of the zones bag, a new bag, 3 new globals after the old ones; every block after a wider
# one moves.
TOK_LAYOUT2 = {'vocab': 12,
               'segments': [{'name': 'ctx', 'offset': 5, 'count': 1, 'width': 14, 'present': None,
                             'id': {'at': 1, 'radix': [12]}},
                            {'name': 'units', 'offset': 19 + 36, 'count': 4, 'width': 12, 'present': 0,
                             'id': {'at': 1, 'radix': [3, 4]}, 'actions': [[3, 1], [20, 2]]},
                            {'name': 'cards', 'offset': 19 + 36 + 48, 'count': 3, 'width': 6, 'present': 3,
                             'actions': [[30, 1]]}],
               'bags': [{'name': 'zones', 'offset': 19, 'count': 3, 'width': 12},
                        {'name': 'hidden', 'offset': 19 + 36 + 48 + 18 + 5, 'count': 1, 'width': 12}]}
TOK_F2 = 19 + 36 + 48 + 18 + 5 + 12


def widened_x(x, fmap, F2, seed=0):
    """An input of the wider layout whose old features are `x` (the rest random: any value)."""
    g = torch.Generator().manual_seed(seed)
    x2 = torch.randn(x.shape[0], F2, generator=g) * 3
    x2[:, fmap] = x
    return x2


def test_tokennet_widen_across_a_layout_change(tmp_path):
    """`widen(..., layout=)` (docs/az.md 10.3): every new input column (wider slots, bag rows, a
    new bag, new globals) gets zero weights, so on any input of the new layout the widened
    network's policy, value and margin are the old network's on the old part of it; the same
    tokens (positions, pointer columns); new actions get logit 0. Saved and loaded it stays so."""
    from zero.net import token_feature_map, tokens_layout
    m = tok_net(31)
    with torch.no_grad():
        m.bag.weight.normal_(0, 0.3)                 # (every map non-trivial)
    lay2 = tokens_layout(TOK_LAYOUT2, TOK_F2, TOK_A + 2)
    fmap = token_feature_map(m.config['layout'], TOK_F, lay2, TOK_F2)
    assert fmap[:5] == list(range(5)) and fmap[5:19] == list(range(5, 19))     # globals, ctx
    assert fmap[19:43] == list(range(19, 43))                                   # the zones' old rows
    assert fmap[43:52] == list(range(55, 64)) and fmap[52] == 67                # units: slot 0, slot 1 at 55 + 12
    assert fmap[79:83] == list(range(103, 107)) and fmap[83] == 109             # cards: 6 a slot
    assert fmap[91:93] == [121, 122]                                            # the old trailing globals
    w = widen(m, TOK_F2, TOK_A + 2, layout=TOK_LAYOUT2)
    assert w.config['layout'] == lay2 and w.n_tok == m.n_tok and w.ptr.out_features == m.ptr.out_features
    path = str(tmp_path / 'w.pt')
    save(w, path)
    w2, _ = load(path, allow_mismatch=True)               # (a test network: the guard's opt-out)
    x = tok_x(40, 32)
    n = torch.randint(2, 5, (40,))
    with torch.no_grad():
        p = m(x, n)
        for seed in range(3):
            q = w2(widened_x(x, fmap, TOK_F2, seed), n)
            assert torch.allclose(q[0][:, :TOK_A], p[0], atol=1e-5) and torch.all(q[0][:, TOK_A:] == 0)
            assert torch.allclose(q[1], p[1], atol=1e-5) and torch.allclose(q[2], p[2], atol=1e-5)
        # (a widen to the layout the network has is the old widen: new features are globals)
        a = widen(m, TOK_F + 3, TOK_A + 5, layout=TOK_LAYOUT).state_dict()
        b = widen(m, TOK_F + 3, TOK_A + 5).state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    # the widened network trains: its new columns get gradients
    w.train()
    plog, vlog, mp = w(widened_x(x, fmap, TOK_F2), n)
    (plog.logsumexp(1).mean() + vlog.logsumexp(1).mean() + mp.pow(2).mean()).backward()
    assert w.seg_in[1].weight.grad[:, 9:].abs().sum() > 0 and w.g.weight.grad[:, 7:].abs().sum() > 0
    assert w.bag.weight.grad[:, 2 * 17:].abs().sum() > 0


def test_tokennet_widen_by_declared_global_blocks():
    """Declared global blocks (layout key `globals`): the network is the one it is without them
    (same globals, same parameters); a widen maps the old globals by block name, so a block may
    grow at its end and new blocks may come between the old ones."""
    import pytest
    from zero.net import TokenNet, token_feature_map, tokens_layout
    decl = dict(TOK_LAYOUT, globals=[{'name': 'head', 'offset': 0, 'count': 5}, {'name': 'tail', 'offset': 91, 'count': 2}])
    lay = tokens_layout(decl, TOK_F, TOK_A)
    assert lay['globals'][1] == {'name': 'tail', 'offset': 91, 'count': 2}
    assert 'globals' not in tokens_layout(TOK_LAYOUT, TOK_F, TOK_A)
    torch.manual_seed(33)
    m = TokenNet(TOK_F, TOK_A, decl, d=16, layers=2, heads=2, ff=32, head=16).eval()
    torch.manual_seed(33)
    m0 = TokenNet(TOK_F, TOK_A, TOK_LAYOUT, d=16, layers=2, heads=2, ff=32, head=16).eval()
    assert torch.equal(m.gidx, m0.gidx)
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m0.state_dict().values()))
    with torch.no_grad():
        for lin in (m.gen, m.ptr, m.val[-1], m.mar[-1]):
            lin.weight.normal_(0, 0.3)
    # head grows 5 -> 7 (everything after it moves by 2), a block 'mid' joins before 'tail'
    def sh(s, k):
        return dict(s, offset=s['offset'] + k)
    new = {'vocab': 12, 'segments': [sh(s, 2) for s in TOK_LAYOUT['segments']], 'bags': [sh(TOK_LAYOUT['bags'][0], 2)],
           'globals': [{'name': 'head', 'offset': 0, 'count': 7}, {'name': 'mid', 'offset': 93, 'count': 2},
                       {'name': 'tail', 'offset': 95, 'count': 2}]}
    F2 = TOK_F + 4
    fmap = token_feature_map(m.config['layout'], TOK_F, tokens_layout(new, F2, TOK_A), F2)
    assert fmap[:5] == list(range(5)) and fmap[5] == 7 and fmap[91:] == [95, 96]
    w = widen(m, F2, layout=new)
    x = tok_x(30, 34)
    n = torch.full((30,), 2)
    with torch.no_grad():
        p, q = m(x, n), w(widened_x(x, fmap, F2), n)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(p, q))
    # validation of the blocks: all globals declared, no overlap, feature order, distinct names
    g = decl['globals']
    for bad in ([g[0]], [g[0], dict(g[1], offset=90, count=3)], [g[1], g[0]], [g[0], dict(g[1], name='head')]):
        with pytest.raises(ValueError):
            tokens_layout(dict(TOK_LAYOUT, globals=bad), TOK_F, TOK_A)


def test_tokennet_widen_refuses_a_layout_it_cannot_follow_exactly():
    """A new segment or more slots (new tokens), a narrower slot, another identity or present
    flag, segments in another order, a bag gone, another vocabulary: no exact widen exists,
    widen refuses."""
    import pytest
    m = tok_net(35)
    s0, u, c = TOK_LAYOUT2['segments']
    bad = [dict(TOK_LAYOUT2, segments=[s0, u, c, {'name': 'x', 'offset': 140, 'count': 1, 'width': 3}]),
           dict(TOK_LAYOUT2, segments=[s0, dict(u, count=5), dict(c, offset=c['offset'] + 12)]),
           dict(TOK_LAYOUT2, segments=[s0, dict(u, width=8), c]),
           dict(TOK_LAYOUT2, segments=[s0, dict(u, present=11), c]),
           dict(TOK_LAYOUT2, segments=[s0, dict(u, id={'at': 2, 'radix': [3, 4]}), c]),
           dict(TOK_LAYOUT2, segments=[s0, c, u]),
           dict(TOK_LAYOUT2, bags=[dict(TOK_LAYOUT2['bags'][0], name='zonez'), TOK_LAYOUT2['bags'][1]]),
           dict(TOK_LAYOUT2, vocab=13)]
    for b in bad:
        with pytest.raises(ValueError):
            widen(m, TOK_F2 + 20, TOK_A, layout=b)


def test_fork_run_widens_across_a_layout_change(tmp_path, monkeypatch):
    """tools/fork_run.py <src.pt> <run>: a token network follows the game package's layout as it
    is now (here the toy's layout replaced by TOK_LAYOUT2): iteration 0 of the new run computes
    what the source computed."""
    import sys
    from zero import package
    from zero.net import token_feature_map
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
    import fork_run
    m = tok_net(37)
    src = str(tmp_path / 'src.pt')
    save(m, src, {'iter': 7})
    monkeypatch.setitem(package.active().tokens, 'toy', TOK_LAYOUT2)
    monkeypatch.setattr(G, 'info', lambda g: dict(INFO, num_features=TOK_F2, num_actions=TOK_A))
    assert fork_run.main([src, str(tmp_path / 'run')]) == 0
    w, extra = load(str(tmp_path / 'run' / 'models' / 'iter_0000.pt'))
    assert extra['iter'] == 0 and extra['source_extra'] == {'iter': 7} and w.config['features'] == TOK_F2
    fmap = token_feature_map(m.config['layout'], TOK_F, w.config['layout'], TOK_F2)
    x = tok_x(16, 38)
    n = torch.full((16,), 2)
    with torch.no_grad():
        assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(m(x, n), w(widened_x(x, fmap, TOK_F2), n)))


def test_tokennet_toy_layout_evaluator():
    """The core's toy layout (package.TOY_TOKENS) through the evaluator: a valid answer."""
    from zero.net import make_net
    torch.manual_seed(14)
    m = make_net(INFO, 'toy', 'tokens', hidden=16, blocks=1, heads=2)
    assert m.config['arch'] == 'tokens' and m.config['ff'] == 32
    ev = NetEvaluator(m)
    drv, lb = a_batch(n=20, seed=3)
    pri, val, mar = ev(lb)
    protocol.validate_evaluation(lb, pri, val, mar)


def test_threaded_evaluator_plays_the_same_games():
    """NetEvaluator(workers=3): batches evaluated on threads, three in flight (drive), the same
    games and records as evaluating inline."""
    torch.manual_seed(15)
    m = Net(F, A, hidden=32, blocks=1).eval()
    seat = G.nn_seat('m', 8, selfplay=True, sims_fast=4, p_full=0.5)
    out = []
    for workers in (1, 3):
        drv = G.Driver('toy', [[seat, seat]] * 24, concurrency=24, threads=2, groups=3, seed=9, record=True)
        ev = NetEvaluator(m, workers=workers)
        drive(drv, {'m': ev})
        res = sorted(drv.take_results(), key=lambda r: r['gid'])
        smp = drv.take_samples()
        out.append(([(r['gid'], r['steps'], r['scores']) for r in res], smp['pi'].tobytes(), smp['x'].tobytes()))
        assert ev.calls > 0 and ev.leaves > 0
    assert out[0] == out[1]
