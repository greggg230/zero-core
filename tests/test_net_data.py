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
