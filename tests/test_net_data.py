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
    m2, extra = load(p)
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
    m2, _ = load(path)
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
