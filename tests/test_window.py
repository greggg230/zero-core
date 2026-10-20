"""The training window's direct reads and prefetch (run setting window_load, docs/az.md 4.2):
ShardFile reads exactly what np.load reads; sample_rows keeps Window's counts with uniform rows
and nested shares; a ShardWindow equals the legacy Window given the same rows (aux targets and
masks, width cut / pad, rows split over parts); the prefetched window equals the direct one
(rows read when training starts: only the new iteration's, top-ups when the estimate was high,
spare rows when it was low; the window still filling); PAUSE stops a prefetch cleanly (thread
ended, files closed); a failed prefetch is reported and the window read anyway."""
import os
import threading
import time

import numpy as np
import pytest
import torch

from zero import _engine as E
from zero import data as D, runctl
from zero.data import (ShardFile, Window, WindowCache, keep_share, kept_count, npz_layout, read_shard, sample_rows,
                       shard_key, shard_paths, shard_rows, write_shard)
from zero.net import save

INFO = E.game_info_of('toy')
F, A = INFO['num_features'], INFO['num_actions']
AW = INFO['aux_width']


def fake_samples(n, seed, aux_w=0, width=F, players=2):
    rng = np.random.default_rng(seed)
    lens = rng.integers(2, 7, n)
    off = np.concatenate([[0], np.cumsum(lens)])
    s = {'x': rng.standard_normal((n, width)).astype(np.float32), 'player': rng.integers(0, players, n),
         'n_players': np.full(n, players), 'opt_off': off, 'opt_action': rng.integers(0, A, off[-1]),
         'pi': rng.random(off[-1]).astype(np.float32), 'z': rng.random((n, players)), 'q': rng.random((n, players)),
         'sc': rng.standard_normal((n, players)), 'net': np.zeros(n, np.int64)}
    if aux_w:
        s['aux'] = rng.standard_normal((n, aux_w)).astype(np.float32)
    return s


def fake_shard(path, n, seed, aux_w=0, width=F):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_shard(path, fake_samples(n, seed, aux_w, width), {'game': 'toy', 'features': width})
    return path


def write_iter(run, it, sizes, aux_w=0, seed=0):
    """Iteration it's shards r00_000, r01_000 ... with `sizes` rows."""
    d = os.path.join(run, 'data', f'iter_{it:04d}')
    return [fake_shard(os.path.join(d, f'r{k:02d}_000.npz'), n, seed * 1000 + it * 10 + k, aux_w)
            for k, n in enumerate(sizes)]


REAL_RNG = np.random.default_rng
ARRAYS = ('player', 'n_players', 'opt_off', 'opt_action', 'pi', 'z', 'q', 'sc', 'aux', 'aux_mask')


def same_window(a, b):
    assert a.N == b.N and a.loaded == b.loaded and a.aux_rows == b.aux_rows and a.F == b.F and a.AW == b.AW
    for k in ARRAYS:
        u, v = getattr(a, k), getattr(b, k)
        if u is None or v is None:
            assert u is None and v is None, k
            continue
        assert u.dtype == v.dtype and u.shape == v.shape and np.array_equal(u, v), k
    assert np.array_equal(a.x, b.x)
    idx = np.random.default_rng(1).integers(a.N, size=min(64, 4 * a.N))
    ba, bb = a.batch(idx), b.batch(idx)
    assert ba.keys() == bb.keys() and all(torch.equal(ba[k], bb[k]) for k in ba)


# ---------------------------------------------------------------- ShardFile

def test_shardfile_reads_what_np_load_reads(tmp_path):
    p = fake_shard(str(tmp_path / 'iter_0003' / 'r00_000.npz'), 300, 1, aux_w=AW)
    z = read_shard(p)
    lay = npz_layout(p)
    assert set(lay) == set(z) and lay['x'][1] == z['x'].shape and lay['x'][0] == np.float16
    with ShardFile(p) as sf:
        assert sf.meta == z['meta'] and sf.rows == 300 == shard_rows(p) and sf.has('aux') and sf.width('aux') == AW
        for k in z:
            if k != 'meta':
                a = sf.read(k)
                assert a.dtype == z[k].dtype and np.array_equal(a, z[k]), k
        rng = np.random.default_rng(0)
        for m in (0, 1, 7, 150, 300):
            rows = np.sort(rng.choice(300, m, replace=False))
            for k in ('x', 'aux', 'z'):
                assert np.array_equal(sf.read_rows(k, rows), z[k][rows])
        old = D.READ_MB
        try:                                       # reads of at most one row each
            D.READ_MB = 0
            assert np.array_equal(sf.read_rows('x', np.arange(300)), z['x'])
        finally:
            D.READ_MB = old
        with pytest.raises(IndexError):
            sf.read_rows('x', np.array([5, 3]))
        with pytest.raises(IndexError):
            sf.read_rows('x', np.array([300]))
        with pytest.raises(D.Stopped):
            sf.read_rows('x', np.arange(10), stop=lambda: True)
    assert shard_key(p) == 'iter_0003/r00_000.npz'
    # a compressed shard: np.load reads it (the same arrays)
    c = str(tmp_path / 'c.npz')
    np.savez_compressed(c, meta=np.array('{"rows": 300}'), **{k: v for k, v in z.items() if k != 'meta'})
    assert npz_layout(c) is None and shard_rows(c) == 300
    with ShardFile(c) as sf:
        assert sf.rows == 300 and np.array_equal(sf.read_rows('x', np.array([2, 9])), z['x'][[2, 9]])


# ---------------------------------------------------------------- sample_rows

def test_sample_rows_counts_nested_and_uniform():
    for n, keep in ((1000, 0.24), (37, 0.5), (5, 0.01), (10, 1.0), (10, 2.0), (0, 0.3)):
        r = sample_rows(n, keep, 7, 'iter_0001/r00_000.npz')
        # Window's count: max(1, round(n keep)) of every shard (all of it at keep >= 1)
        assert len(r) == (n if keep >= 1 else min(n, max(1, int(round(n * keep))))) == kept_count(n, keep)
        assert np.all(np.diff(r) > 0) and (not len(r) or (r[0] >= 0 and r[-1] < n))
    # nested: a smaller share's rows are among a larger share's (same seed and shard)
    for lo, hi in ((0.2, 0.25), (0.1, 0.9), (0.24, 1.0)):
        a, b = sample_rows(500, lo, 3, 'k'), sample_rows(500, hi, 3, 'k')
        assert np.all(np.isin(a, b))
    # other seeds / shards: other rows
    assert not np.array_equal(sample_rows(500, 0.2, 3, 'k'), sample_rows(500, 0.2, 4, 'k'))
    assert not np.array_equal(sample_rows(500, 0.2, 3, 'k'), sample_rows(500, 0.2, 3, 'j'))
    # uniform: every row kept with probability m / n (as rng.choice without replacement)
    n, keep, T = 40, 0.25, 4000
    cnt = np.zeros(n)
    for s in range(T):
        cnt[sample_rows(n, keep, s, 'u')] += 1
    p = kept_count(n, keep) / n
    sd = np.sqrt(T * p * (1 - p))
    assert np.all(np.abs(cnt - T * p) < 5 * sd)
    assert keep_share(100, 0) == keep_share(100, 100) == 1.0 and keep_share(100, 25) == 0.25


# ---------------------------------------------------------------- ShardWindow == Window

class FakeRng:
    """Window's draws replaced by given rows (so legacy and new windows hold the same rows)."""

    def __init__(self, rows):
        self.rows = list(rows)

    def choice(self, n, size, replace):
        r = self.rows.pop(0)
        assert len(r) == size and not replace
        return REAL_RNG(len(r)).permutation(r)       # any order: Window sorts them


def legacy_window(monkeypatch, plan, width, budget, aw, seed=5):
    with monkeypatch.context() as m:
        m.setattr(D.np.random, 'default_rng', lambda s: FakeRng([r for _, _, r in plan]))
        return Window([p for p, _, _ in plan], width, A, budget_rows=budget, seed=seed, aux_width=aw)


@pytest.mark.parametrize('width', [F, F - 5, F + 3])
def test_shard_window_equals_legacy_window(tmp_path, monkeypatch, width):
    run = str(tmp_path)
    paths = write_iter(run, 2, [120, 80], aux_w=AW) + write_iter(run, 1, [100], aux_w=0)
    ns = [shard_rows(p) for p in paths]
    budget = int(0.6 * sum(ns))
    keep = keep_share(sum(ns), budget)
    plan = [(p, n, sample_rows(n, keep, 5, shard_key(p))) for p, n in zip(paths, ns)]
    for aw in (AW, 0):
        cache = WindowCache(aw)
        # rows split over parts: a smaller share first, the rest later (a prefetch's top-up)
        cache.fetch_all([(p, sample_rows(n, keep * 0.7, 5, shard_key(p))) for p, n, _ in plan])
        cache.fetch_all([(p, r) for p, _, r in plan], threads=3)
        assert all(len(cache.parts[p]) == 2 for p, _, _ in plan)
        W = cache.window(plan, width, A)
        L = legacy_window(monkeypatch, plan, width, budget, aw)
        assert W.N == L.N == sum(len(r) for _, _, r in plan)
        same_window(W, L)
        if aw:
            n2 = sum(len(r) for _, _, r in plan[:2])
            assert W.aux_rows == n2 and W.aux_mask[:n2].all() and not W.aux_mask[n2:].any()
        else:
            assert W.aux is None and 'aux' not in W.batch(np.arange(3))
    # spare rows (a larger share read than the window keeps) are skipped
    cache = WindowCache(AW)
    cache.fetch_all([(p, sample_rows(n, 1.0, 5, shard_key(p))) for p, n, _ in plan])
    same_window(cache.window(plan, width, A), legacy_window(monkeypatch, plan, width, budget, AW))
    with pytest.raises(ValueError):                # rows that were never read
        WindowCache(AW).window(plan, width, A)
    with pytest.raises(ValueError):                # another aux width, as Window
        WindowCache(AW + 1).fetch(paths[0], plan[0][2])


# ---------------------------------------------------------------- the run: prefetch vs direct

def make_run(run, it, **over):
    from zero.run import DEFAULTS, PRESETS, Run, model_path
    cfg = dict(DEFAULTS)
    cfg.update(PRESETS['toy-smoke'])
    cfg.update(hidden=32, blocks=1, **over)
    os.makedirs(os.path.join(run, 'models'), exist_ok=True)
    R = Run(run, cfg)
    if not os.path.exists(model_path(run, it - 1)):
        save(R.new_model(), model_path(run, it - 1), {'iter': it - 1})
    return R


def budget_gb(rows, aw=0):
    return rows * (F * 2 + 8 * 16 + 3 * 8 * 4 + (2 * aw + 1 if aw else 0)) / 1e9


def prefetch_then_load(run, it, new_sizes, aux_w=0, **over):
    """A prefetch started before iteration it's shards exist (its self-play), those shards written,
    then the window loaded -> (window, stats); and the direct window of the same data."""
    R = make_run(run, it, window_load='prefetch', **over)
    R.start_prefetch(it)
    assert R.prefetch is not None
    write_iter(run, it, new_sizes, aux_w)          # self-play writes the new iteration meanwhile
    R.prefetch.wait()
    W, n, st = R.load_window(it, F, A)
    assert R.prefetch is None and st['mode'] == 'prefetch' and 'prefetch_error' not in st
    Rd = make_run(run, it, window_load='direct', **over)
    Wd, nd, sd = Rd.load_window(it, F, A)
    assert n == nd and sd['read_rows'] == Wd.N and 'prefetch_rows' not in sd
    same_window(W, Wd)
    return W, st


def test_prefetch_equals_direct_and_reads_only_the_new_iteration(tmp_path):
    run = str(tmp_path / 'r')
    for it in range(1, 6):
        write_iter(run, it, [150, 130])
    olds = 5 * 280
    # window 4 of the 6 iterations; a budget of ~half of it
    W, st = prefetch_then_load(run, 6, [150, 130], window=4, window_grow=0, min_window_samples=0,
                               ram_budget_gb=budget_gb(560))
    assert W.loaded == 4 * 280 and st['prefetch_shards'] == 6 and olds > W.loaded
    assert abs(W.N - 560) <= 8 and 0.45 < st['keep'] < 0.55
    # the estimate (0.97 x the previous iteration) is low: the prefetched rows contain the final
    # ones; only the new iteration's rows are read when training starts
    assert st['prefetch_keep'] >= st['keep']
    new = [p for p in shard_paths(os.path.join(run, 'data', 'iter_0006'))]
    assert st['read_rows'] == sum(kept_count(shard_rows(p), st['keep']) for p in new)
    assert st['prefetch_used_rows'] == W.N - st['read_rows'] and st['prefetch_used_rows'] <= st['prefetch_rows']
    assert st['prefetch_spare_rows'] == st['prefetch_rows'] - st['prefetch_used_rows'] < 0.05 * W.N
    assert W.chunk_rows == W.N + st['prefetch_spare_rows']
    assert {'prefetch_s', 'prefetch_wait_s', 'read_s', 'assemble_s', 'prefetch_mb'} <= set(st)


@pytest.mark.parametrize('new', [60, 400])
def test_prefetch_top_up_and_spare_rows(tmp_path, new):
    """A new iteration much smaller than estimated: the share is higher, the older shards' extra
    rows are read when training starts; much larger: more spare prefetched rows. Same window."""
    run = str(tmp_path / 'r')
    for it in range(1, 4):
        write_iter(run, it, [200])
    W, st = prefetch_then_load(run, 4, [new], window=4, window_grow=0, min_window_samples=0,
                               ram_budget_gb=budget_gb(400))
    kn = kept_count(new, st['keep'])
    if new < 200:
        assert st['keep'] > st['prefetch_keep'] and st['read_rows'] > kn and st['prefetch_spare_rows'] == 0
    else:
        assert st['keep'] < st['prefetch_keep'] and st['read_rows'] == kn and st['prefetch_spare_rows'] > 0


def test_prefetch_window_still_filling(tmp_path):
    """window_grow (1 + it // 2 iterations) and min_window_samples: the prefetch plans the same
    iterations (keep 1: every row), or one more when the estimate is short of the minimum."""
    run = str(tmp_path / 'r')
    write_iter(run, 1, [90])
    W, st = prefetch_then_load(run, 2, [90], min_window_samples=0)
    assert W.N == W.loaded == 180 and st['keep'] == 1.0 and st['read_rows'] == 90 and st['prefetch_rows'] == 90
    write_iter(run, 3, [90])
    # it 4: window_iters 3 (iterations 4, 3, 2); min samples 400 extends it to iteration 1
    W, st = prefetch_then_load(run, 4, [90], min_window_samples=400)
    assert W.loaded == 360 and st['prefetch_shards'] == 3
    # an estimate short of the minimum: the prefetch reads iteration 1, the final window does not need it
    W, st = prefetch_then_load(run, 5, [400], min_window_samples=600)
    assert W.loaded == 400 + 270 and st['prefetch_shards'] == 4 and st['prefetch_spare_rows'] == 90
    # iteration 1 (nothing older): nothing to prefetch
    run1 = str(tmp_path / 'one')
    W, st = prefetch_then_load(run1, 1, [50], min_window_samples=0)
    assert W.N == 50 and st['prefetch_rows'] == 0 and st['read_rows'] == 50


def test_prefetch_with_aux_targets(tmp_path, monkeypatch):
    """Auxiliary targets on, older shards without them (a fork): masks and targets as Window's."""
    run = str(tmp_path / 'r')
    write_iter(run, 1, [100], aux_w=0)
    write_iter(run, 2, [100], aux_w=AW)
    W, st = prefetch_then_load(run, 3, [100], aux_w=AW, min_window_samples=0, window_grow=0,
                               ram_budget_gb=budget_gb(150, AW), aux_own_w=1.0)
    assert W.AW == AW and 0 < W.aux_rows < W.N and abs(W.N - 150) <= 3
    R = make_run(run, 3, aux_own_w=1.0, min_window_samples=0, window_grow=0, ram_budget_gb=budget_gb(150, AW))
    rows = R.window_rows(3)
    keep = keep_share(sum(n for _, n in rows), R.window_budget(F))
    plan = [(p, n, sample_rows(n, keep, 3, shard_key(p))) for p, n in rows]
    same_window(W, legacy_window(monkeypatch, plan, F, R.window_budget(F), AW, seed=3))
    assert W.aux_mask.sum() == sum(len(r) for p, _, r in plan if 'iter_0001' not in p)


def test_pause_stops_a_prefetch_cleanly(tmp_path, monkeypatch):
    """PAUSE mid-prefetch (the run's PauseWatch): the thread stops before its next read and ends;
    every shard file is closed (deletable on Windows); Run.close() after it is a no-op."""
    run = str(tmp_path / 'r')
    for it in range(1, 5):
        write_iter(run, it, [200, 200])
    real = D._readinto

    def slow(f, a):                                # 10 ms per read: the prefetch takes seconds
        time.sleep(0.01)
        real(f, a)
    monkeypatch.setattr(D, '_readinto', slow)
    monkeypatch.setattr(D, 'READ_MB', 0)           # one row per read
    R = make_run(run, 5, window=8, min_window_samples=0)
    R.pw = runctl.PauseWatch(run, poll=0.05)
    try:
        R.start_prefetch(5)
        pf = R.prefetch
        assert pf is not None
        deadline = time.time() + 30
        while pf.cache.read_rows == 0 and time.time() < deadline and not pf.done:
            time.sleep(0.02)
        assert pf.cache.read_rows > 0 and not pf.done, 'the prefetch is under way'
        runctl.request_pause(run, by='test')
        t0 = time.time()
        pf.th.join(10)
        assert not pf.th.is_alive() and time.time() - t0 < 5
        assert pf.stopped and pf.error is None and pf.cache.read_rows < 1600
        assert not [t for t in threading.enumerate() if t.name.startswith('zero-window')]
        # the main thread pauses (Paused) and closes the run: nothing left to stop
        R.close()
        assert R.prefetch is None
        for p in shard_paths(os.path.join(run, 'data', 'iter_0004')):
            os.remove(p)                           # the shards it read: no handle left open
    finally:
        R.pw.close()
    # Run.close() stops a running prefetch too (a failure or PAUSE in self-play)
    runctl.request_resume(run)
    R2 = make_run(run, 5, window=8, min_window_samples=0)
    R2.start_prefetch(5)
    pf = R2.prefetch
    time.sleep(0.2)
    R2.close()
    assert R2.prefetch is None and not pf.th.is_alive() and pf.stopped and pf.error is None


def test_failed_prefetch_is_reported_and_the_window_read(tmp_path, monkeypatch):
    run = str(tmp_path / 'r')
    for it in range(1, 3):
        write_iter(run, it, [100])
    real = D.read_part
    boom = {'on': True}

    def flaky(path, rows, aux_width=0, stop=None):
        if boom['on'] and threading.current_thread().name == 'zero-window-prefetch':
            raise OSError('disk on fire')
        return real(path, rows, aux_width, stop)
    monkeypatch.setattr(D, 'read_part', flaky)
    R = make_run(run, 3, min_window_samples=0, window_grow=0)
    R.start_prefetch(3)
    R.prefetch.wait()
    assert 'disk on fire' in R.prefetch.error
    write_iter(run, 3, [100])
    W, n, st = R.load_window(3, F, A)
    assert st['prefetch_error'] == 'OSError: disk on fire' and st['read_rows'] == W.N == 300
    assert 'WINDOW PREFETCH FAILED' in open(os.path.join(run, 'log.txt'), encoding='utf-8').read()


def test_train_window_load_modes(tmp_path):
    from zero.run import parse, resolve_config
    run = str(tmp_path / 'r')
    write_iter(run, 1, [120, 90])
    for mode in ('legacy', 'direct', 'prefetch'):
        R = make_run(run, 1, window_load=mode, min_steps=3, min_window_samples=0)
        if mode == 'prefetch':
            R.start_prefetch(1)                    # (iteration 1: nothing older to read)
        tr = R.train(1, 210)
        w = tr['window']
        assert w['mode'] == mode and tr['window_samples'] == 210 and tr['window_shards'] == 2
        assert (mode == 'legacy') == ('read_rows' not in w)
        assert (mode == 'prefetch') == ('prefetch_rows' in w)
    assert resolve_config(parse(['--run', 'x']), {})['window_load'] == 'prefetch'
    with pytest.raises(SystemExit):
        resolve_config(parse(['--run', 'x', '--window-load', 'bogus']), {})
