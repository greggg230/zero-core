"""Run setting shard_writer_thread (zero/shardthread.py, docs/az.md 31): self-play's records taken as they are
(SearchDriver.take_records) and converted, tallied and written on a writer thread of their own.

The binding (take_records + SampleRecords.split / arrays give take_samples' arrays, cut per game, value for
value and dtype for dtype); the writer (the synchronous ShardWriter's shards byte for byte, file for file -
plain and compressed, its on_write calls, the queue's bound and waits, errors re-raised in the caller,
NonFinite quarantined as before, abort completing the shard being written and leaving no temporary file);
the loop (off = no writer thread, no metrics key; on = an iteration's shards byte for byte the synchronous
ones, its target metrics equal; a rolling run's quotas exact; the shared queue with the remote 'local' and the
local linger - quotas exact, the workers' writer threads, the linger's held records; PAUSE mid-phase with the
writer thread busy: exit 75, no half-written shard, the iteration replayed; co-training refused)."""
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from zero import data as D, games as G, runctl
from zero.data import ShardWriter
from zero.finite import NonFinite
from zero.infer import UniformEvaluator, drive
from zero.run import read_metrics
from zero.shardthread import MB, ThreadedShardWriter, item_bytes, sample_rows, split_records, to_arrays
from zero.workqueue import split_games

from test_loop import PYPATH, ROOT, SMOKE, run_loop


def toy_takes(take, games=16, seed=11):
    """[(results, take(driver))] after every batch of a toy self-play driver (uniform evaluator)."""
    seat = G.nn_seat('m', 8, 1.0, True, 0.3, 0.25, 4, 4, 0.5)
    drv = G.Driver('toy', [[seat] * p for p in [2, 3] * (games // 2)], concurrency=games, seed=seed, record=True,
                   threads=2, groups=2)
    out = []
    drive(drv, {'m': UniformEvaluator()}, on_batch=lambda lb: out.append((drv.take_results(), take(drv))))
    out.append((drv.take_results(), take(drv)))
    return out


def per_game(takes, records):
    """The games' sample items in finishing order: dicts (split_games) or SampleRecords (split_records)."""
    items = []
    for rows, s in takes:
        for _, sd in (split_records(rows, s) if records else split_games(rows, s)):
            items.append(sd)
    return items


def file_bytes(d):
    return {os.path.basename(p): open(p, 'rb').read() for p in sorted(glob.glob(os.path.join(d, '*')))}


def test_take_records_gives_take_samples_arrays():
    """take_records + split per game + arrays() = take_samples + split_games, array for array (dtypes too);
    a SampleRecords converts or splits once; its length and bytes; wrong sizes refused."""
    a = toy_takes(lambda d: d.take_samples())
    b = toy_takes(lambda d: d.take_records())
    assert [[r['gid'] for r in ra] for ra, _ in a] == [[r['gid'] for r in rb] for rb, _ in b]
    for (_, sa), (_, sb) in zip(a, b):
        assert (sa is None) == (sb is None)
        if sb is not None:
            assert len(sb) == sample_rows(sa) == sample_rows(sb) > 0
            # (the records hold their rows' non-zeros, 8 bytes each, not the dense rows: docs/az.md 44.3)
            nnz = int((sa['x'].view(np.uint32) != 0).sum())
            assert 8 * nnz <= sb.nbytes and item_bytes(sb) == sb.nbytes
    ga, gb = per_game(a, False), per_game(b, True)
    assert len(ga) == len(gb) and sum(1 for s in ga if s is not None) > 10
    rows = 0
    for da, rb in zip(ga, gb):
        assert (da is None) == (rb is None)
        if da is None:
            continue
        db = rb.arrays()
        assert db.keys() == da.keys()
        for k in da:
            assert db[k].dtype == da[k].dtype and db[k].shape == da[k].shape and np.array_equal(db[k], da[k]), k
        with pytest.raises(RuntimeError, match='converted or split already'):
            rb.arrays()
        rows += len(da['player'])
    assert rows > 100
    s = next(x for _, x in toy_takes(lambda d: d.take_records()) if x is not None and len(x) > 1)
    n = len(s)
    with pytest.raises(ValueError, match='sizes'):
        s.split([n + 1])
    with pytest.raises(ValueError, match='sizes'):
        s.split([0, n])
    parts = s.split([1, n - 1])
    assert len(s) == 0 and [len(p) for p in parts] == [1, n - 1]
    with pytest.raises(RuntimeError, match='converted or split already'):
        s.split([0])


def test_split_records_errors_like_split_games():
    takes = toy_takes(lambda d: d.take_records())
    rows, s = next((r, x) for r, x in takes if x is not None and len(r) >= 1)
    bad = [dict(rows[0], samples=len(s) + 5)] + rows[1:]
    with pytest.raises(RuntimeError, match='left in the batch'):
        split_records(bad, s)
    short = [dict(r, samples=0) for r in rows]
    with pytest.raises(RuntimeError, match='left over'):
        split_records(short, s)
    assert split_records([], None) == []


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('compress', [0, 1])
@pytest.mark.parametrize('kind', ['records', 'dicts'])
def test_threaded_writer_writes_the_synchronous_shards_byte_for_byte(tmp_path, kind, compress, sparse):
    """The same items (per game, as the shared queue hands them; or whole takes) through a ShardWriter and
    through a ThreadedShardWriter (a queue bound smaller than one item: every add waits): the same files,
    byte for byte, the same on_write calls (path names, rows, bytes) and counts; the queue's metrics. A sparse
    writer (`x_format` sparse): the threaded one takes the records' non-zeros (`arrays(sparse=True)`, docs/az.md
    44.4), the synchronous one scans take_samples' dense rows (`SparseRows.from_dense`) - the same shards."""
    takes_s = toy_takes(lambda d: d.take_samples(), games=24)
    takes_r = toy_takes(lambda d: d.take_records(), games=24)
    sync_items = per_game(takes_s, False)
    thr_items = per_game(takes_r, kind == 'records') if kind == 'records' else per_game(takes_s, False)
    meta = {'game': 'toy', 'iter': 3, **({'x_format': 'sparse'} if sparse else {})}
    calls = {'sync': [], 'thr': []}
    sw = ShardWriter(str(tmp_path / 'sync'), 'r00', rows=97, meta=meta, compress=compress,
                     on_write=lambda p, r, b: calls['sync'].append((os.path.basename(p), r, b)))
    for s in sync_items:
        sw.add(s)
    ws = sw.close()
    tw = ThreadedShardWriter(ShardWriter(str(tmp_path / 'thr'), 'r00', rows=97, meta=meta, compress=compress,
                                         on_write=lambda p, r, b: calls['thr'].append((os.path.basename(p), r, b))),
                             max_bytes=64)
    for s in thr_items:
        tw.add(s)
    wt = tw.close()
    assert wt == ws and ws['shards'] >= 3
    assert calls['thr'] == calls['sync']
    a, b = file_bytes(tmp_path / 'sync'), file_bytes(tmp_path / 'thr')
    assert list(a) == list(b) and all(a[k] == b[k] for k in a), [k for k in a if a.get(k) != b.get(k)]
    m = tw.metrics()
    n_items = sum(1 for s in thr_items if s is not None)
    assert m['items'] == n_items and m['rows'] == ws['samples'] and m['waits'] > 0 and m['writer_s'] > 0
    assert not tw.th.is_alive()


def test_threaded_writer_whole_takes_and_tallies(tmp_path):
    """Whole takes (this machine's plain self-play: every 32 batches), `on_item` run on the writer thread
    on each converted dict in order (TargetStats: the same summary), float16 features handed to the writer;
    `games` counted before each item reaches the writer (the remote worker's announcements)."""
    takes_s = toy_takes(lambda d: d.take_samples(), games=24)
    takes_r = toy_takes(lambda d: d.take_records(), games=24)
    ts_sync, ts_thr = D.TargetStats(), D.TargetStats()
    sw = ShardWriter(str(tmp_path / 'sync'), 'c00', rows=150, meta={'iter': 1})
    for _, s in takes_s:
        ts_sync.add(s)
        sw.add(s)
    sw.close()
    seen = []

    def on_item(s):
        assert threading.current_thread().name == 'zero-shard-writer'
        assert s['x'].dtype == np.float16
        seen.append(len(s['player']))
        ts_thr.add(s)
    games_at_write = []
    w = ShardWriter(str(tmp_path / 'thr'), 'c00', rows=150, meta={'iter': 1},
                    on_write=lambda p, r, b: games_at_write.append(tw.games))
    tw = ThreadedShardWriter(w, on_item=on_item)
    for _, s in takes_r:
        tw.add(s, games=1)
    tw.close()
    assert ts_thr.summary() == ts_sync.summary() and ts_sync.n > 0
    assert seen == [sample_rows(s) for _, s in takes_s if s is not None]
    assert games_at_write and games_at_write == sorted(games_at_write)
    a, b = file_bytes(tmp_path / 'sync'), file_bytes(tmp_path / 'thr')
    assert list(a) == list(b) and all(a[k] == b[k] for k in a)
    x = next(s for _, s in takes_s if s is not None)
    h = to_arrays(dict(x))
    assert h['x'].dtype == np.float16 and np.array_equal(h['x'], x['x'].astype(np.float16))


def nan_items(n=5, rows=20, F=8):
    out = []
    for k in range(n):
        x = np.ones((rows, F), np.float32)
        if k == 2:
            x[3, 1] = np.nan
        out.append({'x': x, 'player': np.zeros(rows, np.int32), 'n_players': np.full(rows, 2, np.int32),
                    'opt_off': np.arange(rows + 1, dtype=np.int64) * 2, 'opt_action': np.zeros(2 * rows, np.int32),
                    'pi': np.full(2 * rows, 0.5, np.float32), 'z': np.zeros((rows, 6), np.float32),
                    'q': np.zeros((rows, 6), np.float32), 'sc': np.zeros((rows, 6), np.float32),
                    'net': np.zeros(rows, np.uint32)})
    return out


def test_writer_errors_reach_the_caller(tmp_path):
    """A non-finite shard on the writer thread: quarantined as the synchronous writer does (the same bad file,
    byte for byte), and NonFinite raised in the caller - by a later add or by close - never lost; the writer
    discards what comes after it. Another exception (a failing disk) likewise."""
    with pytest.raises(NonFinite):
        sw = ShardWriter(str(tmp_path / 's' / 'data'), 'r00', rows=40, meta={'iter': 1}, bad_dir=str(tmp_path / 's'))
        for s in nan_items():
            sw.add(s)
        sw.close()
    tw = ThreadedShardWriter(ShardWriter(str(tmp_path / 't' / 'data'), 'r00', rows=40, meta={'iter': 1},
                                         bad_dir=str(tmp_path / 't')))
    with pytest.raises(NonFinite, match='non-finite sample values'):
        for s in nan_items():
            tw.add(s)
            time.sleep(0.05)
        tw.close()
    with pytest.raises(NonFinite):
        tw.check()
    tw.abort()
    assert not tw.th.is_alive()
    a, b = file_bytes(tmp_path / 's' / 'data'), file_bytes(tmp_path / 't' / 'data')
    assert list(a) == list(b) == ['r00_000.npz'] and a == b
    bad_s = glob.glob(str(tmp_path / 's' / '*.bad.npz'))
    bad_t = glob.glob(str(tmp_path / 't' / '*.bad.npz'))
    assert [os.path.basename(f) for f in bad_s] == [os.path.basename(f) for f in bad_t] == ['data_r00_001.bad.npz']
    with np.load(bad_s[0]) as zs, np.load(bad_t[0]) as zt:      # (the same but for the meta's own path)
        assert zs.files == zt.files
        for k in zs.files:
            if k != 'meta':
                assert zs[k].dtype == zt[k].dtype and np.array_equal(zs[k], zt[k], equal_nan=True), k
        ms, mt = json.loads(str(zs['meta'])), json.loads(str(zt['meta']))
        assert ms.pop('shard') != mt.pop('shard') and ms == mt

    class Broken(ShardWriter):
        def flush(self):
            raise OSError('disk full (test)')
    tw = ThreadedShardWriter(Broken(str(tmp_path / 'b'), 'r00', rows=10))
    tw.add(nan_items(1)[0])
    with pytest.raises(OSError, match='disk full'):
        tw.close()


def test_a_full_queue_blocks_the_caller_and_an_error_frees_it(tmp_path, monkeypatch):
    """max_bytes: add waits while the queue holds the bound (counted); an item larger than the bound passes
    alone; a writer that fails while the caller waits releases it with the error."""
    gate = threading.Event()
    orig = D.write_shard

    def slow(*a, **kw):
        gate.wait(10)
        return orig(*a, **kw)
    monkeypatch.setattr(D, 'write_shard', slow)
    items = nan_items(6)[:2] + nan_items(6)[3:]          # finite ones
    one = item_bytes(items[0])
    tw = ThreadedShardWriter(ShardWriter(str(tmp_path / 'q'), 'r00', rows=20), max_bytes=one + 1)
    tw.add(items[0])                                  # taken by the writer, which blocks in its first shard
    time.sleep(0.2)
    tw.add(items[1])                                  # queued (alone: fits)
    t = threading.Thread(target=tw.add, args=(items[2],))
    t.start()
    time.sleep(0.3)
    assert t.is_alive() and tw.waits == 1             # waits for room
    gate.set()
    t.join(10)
    assert not t.is_alive()
    for s in items[3:]:
        tw.add(s)
    assert tw.close()['shards'] == 5 and tw.wait_s > 0.2 and tw.peak <= one + 1

    def fail(*a, **kw):
        time.sleep(0.3)
        raise OSError('write failed (test)')
    monkeypatch.setattr(D, 'write_shard', fail)
    tw = ThreadedShardWriter(ShardWriter(str(tmp_path / 'f'), 'r00', rows=20), max_bytes=one + 1)
    tw.add(items[0])
    time.sleep(0.05)
    tw.add(items[1])
    with pytest.raises(OSError, match='write failed'):
        tw.add(items[2])                              # waited, then the writer's error
    tw.abort()


def test_abort_finishes_the_shard_being_written(tmp_path, monkeypatch):
    """abort() while the writer is inside a shard: that shard is completed (loadable, its final name, no
    temporary file left), the queued rest dropped, the thread joined - what a PAUSE mid-phase does."""
    started = threading.Event()
    orig = D._save_npz

    def slow(path, c, meta, compress=0):
        started.set()
        time.sleep(0.5)
        return orig(path, c, meta, compress)
    monkeypatch.setattr(D, '_save_npz', slow)
    items = nan_items(6)[:2] + nan_items(6)[3:]
    tw = ThreadedShardWriter(ShardWriter(str(tmp_path / 'a'), 'r00', rows=20))
    for s in items:
        tw.add(s)
    assert started.wait(5)
    tw.abort()
    assert not tw.th.is_alive()
    files = sorted(os.listdir(tmp_path / 'a'))
    assert files == ['r00_000.npz'], files
    with np.load(tmp_path / 'a' / 'r00_000.npz') as z:
        assert len(z['player']) == 20
    with pytest.raises(RuntimeError, match='add after abort'):
        tw.add(items[0])


# ---------------------------------------------------------------- the loop

@pytest.fixture(scope='module')
def smoke_run(tmp_path_factory):
    run = tmp_path_factory.mktemp('swt') / 'smoke'
    p = run_loop(run, '--iters', '2')
    assert p.returncode == 0, p.stdout + p.stderr
    return run


def test_selfplay_shards_are_the_synchronous_ones(smoke_run, tmp_path):
    """An iteration's self-play with the setting on writes the shards of the setting off byte for byte (small
    shards: several), the same samples, games and target metrics; off: no writer thread and no metrics key
    (the loop of before), on: the `shard_writer` block."""
    from zero import run as R
    with open(os.path.join(smoke_run, 'config.json')) as f:
        cfg = json.load(f)
    assert R.DEFAULTS['shard_writer_thread'] == 0 and R.DEFAULTS['shard_writer_queue_mb'] == 4096
    out, files = {}, {}
    for on in (0, 1):
        run = tmp_path / f'w{on}'
        shutil.copytree(smoke_run, run)
        threads0 = {t.name for t in threading.enumerate()}
        out[on] = R.Run(str(run), dict(cfg, shard_writer_thread=on, shard_rows=60, games=48)).selfplay(2)
        assert 'zero-shard-writer' not in {t.name for t in threading.enumerate()} - threads0
        files[on] = file_bytes(run / 'data' / 'iter_0002')
    assert len(files[0]) >= 3 and list(files[0]) == list(files[1])
    assert all(files[0][k] == files[1][k] for k in files[0])
    for k in ('games', 'samples', 'shards', 'bytes', 'target', 'decisions_per_game'):
        assert out[0][k] == out[1][k], k
    assert 'shard_writer' not in out[0]
    w = out[1]['shard_writer']
    assert w['rows'] == out[1]['samples'] and w['items'] >= 1 and w['samples_s'] >= 0 and w['queue_mb'] == 4096


def test_rolling_run_quotas_exact(tmp_path):
    """A rolling run with the writer thread: every iteration exactly `games` finished games, its shards hold
    the samples it reports, the writer's rows match; the log line."""
    run = tmp_path / 'roll'
    p = run_loop(run, '--iters', '3', '--rolling', '1', '--conc', '64', '--games', '96', '--shard-rows', '150',
                 '--shard-writer-thread', '1', '--shard-writer-queue-mb', '1')
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    for r in m[1:]:
        sp = r['selfplay']
        assert sp['games'] == 96 and sp['shard_writer']['rows'] == sp['samples'], sp
        rows = 0
        for f in sorted(glob.glob(str(run / 'data' / f'iter_{r["iter"]:04d}' / '*.npz'))):
            with np.load(f) as z:
                rows += len(z['player'])
        assert rows == sp['samples'] and sp['shards'] >= 2
    assert 'shard writer thread: caller' in (run / 'log.txt').read_text(encoding='utf-8')
    assert json.load(open(run / 'config.json'))['shard_writer_thread'] == 1


def test_shared_queue_local_linger_and_remote_workers(tmp_path):
    """The shared queue with the remote 'local' (its workers read the setting from the run's config) and the
    local linger, the writer thread on everywhere: quotas exact every iteration, the linger's held games'
    records counted in the next iteration (one checkpoint stale), the workers' writer threads' metrics,
    every shard of an iteration (this machine's and the remote's) holding the rows its metrics report."""
    from test_remoteplay import PAIR, check_quota, pair_env, remote_shards
    run = tmp_path / 'sq'
    p = run_loop(run, '--iters', '4', *PAIR, '--selfplay-remote', 'local', '--selfplay-local-linger', '1',
                 '--min-steps', '150', '--shard-writer-thread', '1', '--shard-rows', '500', env=pair_env(tmp_path))
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    check_quota(m, 192)
    lg = [r.get('linger') for r in m[1:]]
    assert all(x is not None and 'samples_s' in x for x in lg[:-1]) and lg[-1] is None, lg
    assert any(x['games'] > 0 for x in lg[:-1]), lg
    stale = [r['selfplay']['queue']['staleness']['local'] for r in m[2:]]
    assert any(s.get('1', 0) > 0 for s in stale), stale
    for r in m[1:]:
        sp, it = r['selfplay'], r['iter']
        assert sp['shard_writer']['rows'] == sp['samples']
        local = [f for f in glob.glob(str(run / 'data' / f'iter_{it:04d}' / 'r00_*.npz'))]
        assert sum(D.shard_rows(f) for f in local) == sp['samples']
        assert sum(D.shard_rows(f) for f in remote_shards(run, it)) == sp['remote']['samples']
        assert not glob.glob(str(run / 'data' / f'iter_{it:04d}' / '*.tmp.npz'))
    rw = [r['selfplay']['remote'].get('shard_writer') for r in m[1:]]
    assert any(w and w['workers'] >= 1 and w['rows'] > 0 for w in rw), rw
    from test_loop import check_timing
    for r in m[1:]:
        check_timing(r['selfplay'])                # (the shared queue's local driver, docs/az.md 44.2)
    assert json.load(open(tmp_path / 'mirror' / 'sq' / 'config.json'))['shard_writer_thread'] == 1


def test_pause_mid_phase_with_the_writer_busy(tmp_path):
    """PAUSE while self-play writes shards on the writer thread (each shard write slowed): exit 75 at the next
    pause point, the writer stopped with its shard completed - no temporary file, every shard in the data
    directory loadable whole -, then the resume replays the iteration and finishes."""
    run = tmp_path / 'pz'
    run.mkdir()
    slow = os.path.join(str(tmp_path), 'slow_run.py')
    with open(slow, 'w') as f:
        # every shard's temporary file written, then 0.3 s before its rename: a process killed mid-write leaves
        # it behind (the writer falls behind the toy's samples: it is always inside a shard)
        f.write('import time\nimport numpy as np\nimport zero.run as R\n_o = np.savez\n'
                'def _slow(*a, **k):\n    _o(*a, **k)\n    time.sleep(0.3)\nnp.savez = _slow\nR.cli()\n')
    console = open(tmp_path / 'pz.out', 'w+', encoding='utf-8')
    proc = subprocess.Popen([sys.executable, slow, '--run', str(run), *SMOKE, '--iters', '3', '--games', '2000',
                             '--conc', '64', '--rolling', '1', '--shard-rows', '100', '--shard-writer-thread', '1'],
                            cwd=ROOT, stdout=console, stderr=subprocess.STDOUT, text=True,
                            env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH))
    deadline = time.time() + 120
    d1 = run / 'data' / 'iter_0001'
    while len(glob.glob(str(d1 / 'r00_*.npz'))) < 2 and time.time() < deadline and proc.poll() is None:
        time.sleep(0.05)
    runctl.request_pause(str(run), by='test')
    t0 = time.time()
    proc.wait(timeout=120)
    assert time.time() - t0 < 60
    console.seek(0)
    out = console.read()
    console.close()
    assert proc.returncode == runctl.EXIT_PAUSED, out[-3000:]
    assert not glob.glob(str(run / 'data' / '*' / '*.tmp.npz')), os.listdir(d1)
    shards = glob.glob(str(run / 'data' / '*' / '*.npz'))
    assert len(shards) >= 2
    for f in shards:
        with np.load(f) as z:
            assert len(z['player']) == json.loads(str(z['meta']))['rows'] >= 100
    runctl.request_resume(str(run))
    p = run_loop(run, '--iters', '1', '--games', '48', '--rolling', '1', '--shard-writer-thread', '1')
    assert p.returncode == 0, p.stdout + p.stderr
    assert [r['iter'] for r in read_metrics(str(run))] == [0, 1]


def test_cotraining_is_refused(tmp_path):
    from test_cotrain import co_loop
    p = co_loop(tmp_path / 'co', '--iters', '1', '--shard-writer-thread', '1')
    assert p.returncode != 0 and 'shard_writer_thread: single-game runs only' in p.stderr, p.stderr[-2000:]
    p = run_loop(tmp_path / 'q', '--iters', '1', '--shard-writer-thread', '1', '--shard-writer-queue-mb', '0')
    assert p.returncode != 0 and 'shard_writer_queue_mb' in p.stderr, p.stderr[-2000:]


def test_loop_fails_on_nonfinite_samples_from_the_writer_thread(tmp_path):
    """A non-finite value found by the writer thread fails the run as the synchronous writer does
    (test_nonfinite.py's injection): NonFinite in the run's own traceback and log, the shard quarantined, no
    checkpoint trained on it, the writer thread stopped."""
    from test_nonfinite import RUNNER, assert_failed_without_checkpoint
    run = tmp_path / 'x-overflow'
    script = tmp_path / 'inject.py'
    script.write_text(RUNNER)
    p = subprocess.run([sys.executable, str(script), 'x-overflow', '--run', str(run), *SMOKE, '--iters', '2',
                        '--shard-writer-thread', '1'], cwd=ROOT, capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH))
    assert_failed_without_checkpoint(run, p)
    msg = p.stderr.strip().splitlines()[-1]
    assert 'non-finite sample values in shard c00_000.npz' in msg and 'x: 1 row(s) [3], column(s) [5] (1 +inf)' in msg
    assert os.listdir(run / 'debug') == ['iter_0001_c00_000.bad.npz']
    assert D.shard_paths(str(run / 'data' / 'iter_0001')) == []


def test_a_writer_keeping_non_zeros_gets_the_float32_rows(tmp_path):
    """The float16 cast is for a writer that buffers dense rows; one keeping its rows' non-zeros (an attribute
    `sparse` true, as zero-core feat/sparse-shards' ShardWriter) gets the records' float32 non-zeros to cast
    itself (`SampleRecords.arrays(sparse=True)`: no dense rows at all, docs/az.md 44.4)."""
    got = []

    class Sparse(ShardWriter):
        def add(self, s):
            got.append(('x' in s, s['xs_val'].dtype, s['xs_idx'].dtype, s['xs_off'].dtype, int(s['n_features'])))
            super().add(s)
    takes = toy_takes(lambda d: d.take_records(), games=8)
    # (a real sparse writer: its meta asks for the format - data.shard_format, run setting shard_sparse - and
    # ShardWriter sets `sparse` from it in its constructor; a class attribute would be shadowed by it)
    w = Sparse(str(tmp_path / 's'), 'r00', rows=1000, meta={'x_format': 'sparse'})
    assert w.sparse
    tw = ThreadedShardWriter(w)
    for _, s in takes:
        tw.add(s)
    tw.close()
    assert got and set(got) == {(False, np.dtype(np.float32), np.dtype(np.uint32), np.dtype(np.int64), G.info('toy')['num_features'])}


def test_a_dense_writer_refuses_the_non_zeros(tmp_path):
    takes = toy_takes(lambda d: d.take_records(), games=4)
    s = next(r for _, r in takes if r is not None).arrays(sparse=True)
    with pytest.raises(ValueError, match='dense shard writer'):
        ShardWriter(str(tmp_path / 'd'), 'r00', rows=1000, meta={'game': 'toy'}).add(s)
