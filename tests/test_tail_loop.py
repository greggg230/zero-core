"""toy_tail end to end through a tiny run's data path (docs/az.md 43): the run's own self-play phase (`Run._selfplay`:
the driver, `infer.drive`, the shard writer - synchronous and on its thread) with a "row echo" evaluator instead of a
network, its v2 shards read back by the training window and its batches, the run's tail metrics, and the run's
refusals at start (R4, R8, R15, R16)."""
import os

import numpy as np
import pytest

from zero import data, package
from zero import rows as R
from zero import run as RUN

from test_tail_rows import echo

LAY = package.TOY_TAIL_TOKENS


class EchoEval:
    """An evaluator answering each row from the row alone (`test_tail_rows.echo`): head floats, tail tokens,
    overrides - what a network with tails would read, without one."""
    seconds = calls = leaves = 0

    def __call__(self, lb):
        self.calls += 1
        self.leaves += lb.B
        return echo(lb, lb.x)


def cfg(**kw):
    c = dict(RUN.DEFAULTS)
    c.update(game='toy_tail', spec={'items0': 6, 'max_moves': 24}, games=12, conc=12, sims=8, sims_fast=4, p_full=0.5,
             threads=2, groups=2, arch='tokens', shard_sparse=1, shard_rows=40, rolling=0, players=[2], seed=5,
             yardstick='random')
    c.update(kw)
    return c


def selfplay(tmp, **kw):
    R_ = RUN.Run(str(tmp), cfg(**kw))
    R_.evaluator = lambda path: EchoEval()
    R_.check_rows()
    out = R_._selfplay(1)
    return out, data.shard_paths(os.path.join(str(tmp), 'data', 'iter_0001'))


def test_a_tiny_run_carries_the_rows_into_training_batches(tmp_path):
    out, paths = selfplay(tmp_path / 'a')
    assert out['games'] == 12 and out['samples'] > 0 and paths
    t = out['tail']
    assert t['rows'] > 0 and t['tokens'] >= t['rows'] and t['max'] >= 3 and t['aborted_row_tokens'] == 0
    metas = [data.read_shard(p)['meta'] for p in paths]
    assert all(m['x_version'] == 2 and m['t_segs'] == [[0, 'items', 16, 4], [1, 'marks', 4, 0]] for m in metas)
    # the shard writer thread writes the same shards, array for array
    out2, paths2 = selfplay(tmp_path / 'b', shard_writer_thread=1)
    assert [os.path.basename(p) for p in paths2] == [os.path.basename(p) for p in paths]
    for p, q in zip(paths, paths2):
        a, b = data.read_shard(p, dense=False), data.read_shard(q, dense=False)
        assert sorted(a) == sorted(b)
        for k in a:
            if k != 'meta':
                assert np.array_equal(a[k], b[k]), k
    # the window and its batches: every override resolved against its row's tokens, the static axis built
    W = data.Window(paths, 72, 16, layout=LAY)
    assert W.N == out['samples'] and W.tail.NT == sum(int(data.read_shard(p)['t_ptr'][-1]) for p in paths)
    n = 0
    for idx, b in data.batches(W, np.random.default_rng(0), 4, 32, tokens=(R.present_index(LAY), 8)):
        assert b['ntok'].max() <= 64 and b['full'].shape[1] % 8 == 0
        assert len(b['ov']['row']) == W.ov.take(idx).NO
        n += 1
    assert n == 4


def test_a_game_over_the_limit_is_aborted_counted_and_never_recorded(tmp_path):
    out, paths = selfplay(tmp_path, spec={'items0': 70}, games=4, conc=4)
    assert out['games'] == 4 and out['unfinished'] == 4 and out['samples'] == 0
    assert out['tail']['aborted_row_tokens'] == 4 and out['tail']['rows'] == 0
    assert not paths or all(data.shard_rows(p) == 0 for p in paths)


def test_the_run_refuses_what_cannot_carry_the_rows(tmp_path, monkeypatch):
    with pytest.raises(RUN.RowsRefused, match='R4'):
        RUN.Run(str(tmp_path), cfg(arch='mlp')).check_rows()
    with pytest.raises(RUN.RowsRefused, match='R8'):
        RUN.Run(str(tmp_path), cfg(shard_sparse=0)).check_rows()
    r = RUN.Run(str(tmp_path), cfg())
    r.info = dict(r.info, head_tokens=0)
    with pytest.raises(RUN.RowsRefused, match='head_tokens'):
        r.check_rows()
    monkeypatch.setitem(package.CORE.tokens, 'toy_tail', dict(LAY, draft=True))
    with pytest.raises(RUN.RowsRefused, match='R16'):
        RUN.Run(str(tmp_path), cfg()).check_rows()
    RUN.Run(str(tmp_path), cfg(allow_draft_layout=1)).check_rows()
    # a refusal is an EncodingMismatch: exit 78, never restarted
    assert issubclass(RUN.RowsRefused, RUN.EncodingMismatch)
    # the toy and its layout: nothing to refuse
    RUN.Run(str(tmp_path), cfg(game='toy', spec={}, arch='mlp', shard_sparse=0)).check_rows()


def test_chained_contests_carry_the_rows_into_shards_and_batches(tmp_path):
    """toy_tail as chained contests (spec `episodes`, docs/az.md 42 with 43) through the run's self-play phase: one row
    a contest, every game's samples behind it with their tails and overrides in the v2 shards (synchronous and on the
    writer thread, array for array), the window and its batches resolving every override; a later game over the row
    limit ends its contest aborted, the earlier games' samples kept."""
    out, paths = selfplay(tmp_path / 'a', spec={'items0': 6, 'max_moves': 12, 'episodes': 2}, games=6, conc=6)
    assert out['games'] == 6 and out['samples'] > 0 and paths
    assert out['episodes']['games'] == 12 and out['episodes']['contests'] == 6
    assert out['tail']['rows'] > 0 and out['tail']['aborted_row_tokens'] == 0
    out2, paths2 = selfplay(tmp_path / 'b', spec={'items0': 6, 'max_moves': 12, 'episodes': 2}, games=6, conc=6,
                            shard_writer_thread=1)
    for p, q in zip(paths, paths2):
        a, b = data.read_shard(p, dense=False), data.read_shard(q, dense=False)
        assert sorted(a) == sorted(b) and all(np.array_equal(a[k], b[k]) for k in a if k != 'meta')
    W = data.Window(paths, 72, 16, layout=LAY)
    assert W.N == out['samples'] and W.tail.NT == sum(int(data.read_shard(p)['t_ptr'][-1]) for p in paths) > 0
    for idx, b in data.batches(W, np.random.default_rng(0), 3, 16, tokens=(R.present_index(LAY), 8)):
        assert len(b['ov']['row']) == W.ov.take(idx).NO
    out, paths = selfplay(tmp_path / 'c', spec={'items0': 6, 'max_moves': 8, 'episodes': 3, 'items_step': 70}, games=4,
                          conc=4)
    assert out['games'] == 4 and out['unfinished'] == 4 and out['tail']['aborted_row_tokens'] == 4
    assert out['samples'] > 0 and sum(data.shard_rows(p) for p in paths) == out['samples']
