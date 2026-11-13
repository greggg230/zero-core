"""Self-play's training records written on a thread of their own (run setting `shard_writer_thread`,
docs/az.md 31).

The thread that drives a self-play driver (`zero.infer.drive`: next_batch -> evaluator -> submit) also
handled the samples: every few batches it took them from the driver - `take_samples`, a copy of every
record into one dense float32 array (MTG: 37,620 floats a row) - converted them to the shards' dtypes
and wrote .npz shards, while the network and the search threads waited for it. With the setting on:

* the caller only takes: `SearchDriver.take_records()` moves the driver's records out as they are (no
  copy, docs/search.md 2.4), `split_records` cuts them per finished game (the records move, as
  `zero.workqueue.split_games` slices a sample dict), and `ThreadedShardWriter.add` queues them;
* one writer thread ('zero-shard-writer') converts each queued item (`SampleRecords.arrays()`: the
  arrays take_samples gives, built off the GIL; the features cast to float16, the type dense shards
  store, which halves what the writer buffers - not for a writer keeping its rows' non-zeros, an
  attribute `sparse` true) and calls the wrapped `zero.data.ShardWriter`'s `add`,
  then its `close`, in the order the items came: the same records in the same order through the same
  writer, so the shards are the synchronous writer's byte for byte (rows, order, file names, meta,
  quarantined *.bad.npz). `on_item(s)` runs there too, on each converted dict (TargetStats, the
  staleness histogram).

Bounded: at most `max_bytes` of queued items (SampleRecords.nbytes, a dict's arrays' bytes); `add`
blocks while the queue is full (counted: `waits`, `wait_s`) and an item larger than the bound passes
when the queue is empty. An exception on the writer thread (NonFinite, OSError, ...) is re-raised in
the caller by its next `add`, `check` or `close` (the exception itself) and the writer discards what it
gets after it: an error is never lost. `abort()` (any exception leaving a phase: PAUSE, an error): the
writer finishes the item it is on - a shard being written is completed and renamed, never left
half-written (nor its temporary file) - drops the rest and is joined. `close()` waits for the queue,
the writer's last shard (`ShardWriter.close` on the writer thread) and the thread, then returns
ShardWriter.close's counts: every shard is on disk before the window can read it.

Variable-length rows (docs/az.md 43): the records carry their tails and overrides (`Record::tail` / `ov`), which
`split_records` moves with them and `SampleRecords.arrays()` returns as `t_*` / `ov_*` (the writer: format v2).

`games`: the remote worker's count of games added since its last shard (as remoteplay._BgWriter's;
`add(s, games)` counts them on the writer thread before the item reaches the ShardWriter, so its
`on_write` callback reads the games of exactly the shard it announces).
"""
import threading
import time
from collections import deque

import numpy as np

MB = 1 << 20


def sample_rows(s):
    """Samples in a driver sample dict, a SampleRecords or None (0)."""
    if s is None:
        return 0
    if isinstance(s, dict):
        return len(s['player'])
    return len(s)


def item_bytes(s):
    """Bytes a queued item holds: SampleRecords.nbytes, a sample dict's arrays' bytes, 0 for None."""
    if s is None:
        return 0
    if isinstance(s, dict):
        return sum(int(getattr(a, 'nbytes', 0)) for a in s.values())
    return int(s.nbytes)


def to_arrays(s, half=True):
    """A queued item -> the sample dict a ShardWriter takes (None: none): a SampleRecords converted
    (take_samples' arrays); `half`: float32 features cast to float16 (the type `compact` stores: the
    same values, so the same shard bytes, in half the memory while the writer buffers them)."""
    if s is not None and not isinstance(s, dict):
        s = s.arrays()
    if half and s is not None and getattr(s.get('x'), 'dtype', None) == np.float32:
        s = dict(s, x=s['x'].astype(np.float16))
    return s


def split_records(rows, recs):
    """`zero.workqueue.split_games` for a SampleRecords (`SearchDriver.take_records` at the point of
    `take_results`): [(row, SampleRecords or None)], game k's records the rows[k]['samples'] after game
    k - 1's (moved, not copied). The same errors as split_games."""
    n = sample_rows(recs)
    sizes = []
    a = 0
    for r in rows:
        k = int(r.get('samples', 0) or 0)
        if k and a + k > n:
            raise RuntimeError(f'split_games: game {r.get("gid")} has {k} samples, {n - a} left in the batch')
        sizes.append(k)
        a += k
    if a != n:
        raise RuntimeError(f'split_games: {n - a} samples left over after {len(rows)} games')
    parts = iter(recs.split([k for k in sizes if k])) if n else iter(())
    return [(r, next(parts) if k else None) for r, k in zip(rows, sizes)]


class SampleClock:
    """Seconds the caller spends in its sample handling (`with clock:`)."""

    def __init__(self):
        self.s = 0.0
        self.t = 0.0

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.s += time.perf_counter() - self.t
        return False


class ThreadedShardWriter:
    """A `zero.data.ShardWriter` (`writer`) fed from a bounded queue by a thread of its own (module
    docstring). Caller side: `add(s, games=0)`, `check()`, `close()`, `abort()`, `metrics()`."""

    def __init__(self, writer, max_bytes=4096 * MB, on_item=None, name='zero-shard-writer'):
        self.w = writer
        self.max_bytes = max(1, int(max_bytes))
        self.on_item = on_item
        self.cv = threading.Condition()
        self.items = deque()                       # (item, games, bytes), oldest first
        self.queued = 0                            # bytes queued (not yet taken by the writer)
        self.error = None                          # the writer thread's exception
        self.state = 'open'                        # 'open' | 'closing' (drain, then close) | 'abort' | 'done'
        self.result = None                         # ShardWriter.close()'s counts
        self.games = 0
        # metrics: the caller's waits for room, the writer's busy time, the queue's peak
        self.n_items = self.n_rows = self.waits = 0
        self.wait_s = self.close_s = self.writer_s = 0.0
        self.peak = 0
        self.dropped = 0                           # items an abort or an error discarded
        self.th = threading.Thread(target=self._run, name=name, daemon=True)
        self.th.start()

    # ------------------------------------------------------------ the caller's side
    def check(self):
        """Raise the writer thread's exception, if it had one."""
        if self.error is not None:
            raise self.error

    def add(self, s, games=0):
        """Queue a sample dict / SampleRecords (None or empty: only `games` counted, if any); blocks
        while the queue holds `max_bytes`."""
        self.check()
        if self.state != 'open':
            raise RuntimeError(f'ThreadedShardWriter: add after {self.state}')
        rows = sample_rows(s)
        if not rows:
            s = None
        if s is None and not games:
            return
        n = item_bytes(s)
        with self.cv:
            if self.queued and self.queued + n > self.max_bytes:
                t = time.perf_counter()
                self.waits += 1
                while self.queued and self.queued + n > self.max_bytes and self.error is None:
                    self.cv.wait(0.5)
                self.wait_s += time.perf_counter() - t
            if self.error is None:
                self.items.append((s, games, n))
                self.queued += n
                self.peak = max(self.peak, self.queued)
                self.n_items += 1
                self.n_rows += rows
                self.cv.notify_all()
        self.check()

    def close(self):
        """Write everything queued and the last shard, end the thread -> ShardWriter.close()'s counts.
        The writer's exception, if any, is raised here."""
        t = time.perf_counter()
        with self.cv:
            if self.state == 'open':
                self.state = 'closing'
            self.cv.notify_all()
        self.th.join()
        self.close_s += time.perf_counter() - t
        self.check()
        if self.state == 'abort':
            raise RuntimeError('ThreadedShardWriter: closed after an abort')
        return self.result

    def abort(self):
        """Stop: the writer finishes the item it is on (a shard being written is completed), drops the
        rest, and the thread is joined. Never raises (the caller's own exception is on its way)."""
        with self.cv:
            if self.state in ('open', 'closing'):
                self.state = 'abort'
            self.cv.notify_all()
        self.th.join()

    def metrics(self):
        """The writer's counters (the self-play metrics' `shard_writer` block, docs/az.md 31)."""
        return {'items': self.n_items, 'rows': self.n_rows, 'waits': self.waits, 'wait_s': round(self.wait_s, 3),
                'close_s': round(self.close_s, 3), 'writer_s': round(self.writer_s, 3),
                'queue_peak_mb': round(self.peak / MB, 1), 'queue_mb': round(self.max_bytes / MB, 1)}

    # ------------------------------------------------------------ the writer thread
    def _run(self):
        while True:
            with self.cv:
                while not self.items and self.state == 'open':
                    self.cv.wait()
                if self.state == 'abort' or (self.error is not None and self.state != 'open'):
                    self.dropped += len(self.items)
                    self.items.clear()
                    self.queued = 0
                    self.cv.notify_all()
                    return
                if not self.items:                 # closing, the queue drained: the last shard
                    break
                s, g, n = self.items.popleft()
                self.queued -= n
                self.cv.notify_all()
            if self.error is not None:
                self.dropped += 1
                continue
            t = time.perf_counter()
            try:
                self.games += g
                if s is not None:
                    # (a writer that keeps its rows' non-zeros, `sparse`, casts them itself: no dense copy)
                    s = to_arrays(s, half=not getattr(self.w, 'sparse', False))
                    if self.on_item is not None:
                        self.on_item(s)
                    self.w.add(s)
            except BaseException as e:             # noqa: BLE001 - re-raised on the caller's thread
                self.error = e
            del s
            self.writer_s += time.perf_counter() - t
        t = time.perf_counter()
        try:
            if self.error is None:
                self.result = self.w.close()
        except BaseException as e:                 # noqa: BLE001 - re-raised by close()
            self.error = e
        self.writer_s += time.perf_counter() - t
        with self.cv:
            self.state = 'done' if self.error is None else self.state
            self.cv.notify_all()
