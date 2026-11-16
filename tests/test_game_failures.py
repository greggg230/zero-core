"""Games dropped after a panic (docs/search.md 2.18): the run writes each failure's replay record to
<run>/game_failures.jsonl, counts the dropped games in the iteration's metrics, and fails only when too many of an
iteration's games panicked (a broken build, not a rare bug). The driver's side (the dropped game's aborted row, its
failure record, every other game unchanged) is the Rust test `a_panicking_game_is_dropped_and_the_run_goes_on`."""
import json
import os

import pytest

from zero import run


class FakeDriver:
    def __init__(self, fails):
        self.fails = [json.dumps(f) for f in fails]

    def take_failures(self):
        out, self.fails = self.fails, []
        return out


def bare_run(tmp_path):
    r = object.__new__(run.Run)
    r.run = str(tmp_path)
    r.lines = []
    r.log = r.lines.append
    return r


def failure(gid):
    return {'gid': gid, 'driver_seed': 7, 'game_seed': 99 + gid, 'players': 2, 'spec': '{}', 'tag': None, 'start': None,
            'branch': None, 'steps': [3, 1, 4], 'message': 'engine bug'}


def rows(n, dropped):
    return [{'gid': g, 'over': g not in dropped, 'abort_reason': 'panic' if g in dropped else None} for g in range(n)]


def test_a_dropped_game_is_recorded_counted_and_logged(tmp_path):
    r = bare_run(tmp_path)
    out = {}
    r.failure_metrics(5, out, rows(100, {17}), FakeDriver([failure(17)]))
    assert out['game_failures'] == 1
    with open(os.path.join(tmp_path, run.GAME_FAILURES), encoding='utf-8') as f:
        recs = [json.loads(line) for line in f]
    assert len(recs) == 1 and recs[0]['gid'] == 17 and recs[0]['iter'] == 5 and recs[0]['steps'] == [3, 1, 4]
    assert any('game 17' in line and 'dropped after a panic' in line for line in r.lines)
    # the next iteration appends
    r.failure_metrics(6, {}, rows(100, {3}), FakeDriver([failure(3)]))
    with open(os.path.join(tmp_path, run.GAME_FAILURES), encoding='utf-8') as f:
        assert [json.loads(line)['gid'] for line in f] == [17, 3]


def test_no_failure_changes_nothing(tmp_path):
    r = bare_run(tmp_path)
    out = {}
    r.failure_metrics(1, out, rows(10, set()), FakeDriver([]))
    assert out == {} and r.lines == [] and not os.path.exists(os.path.join(tmp_path, run.GAME_FAILURES))


def test_a_remote_s_dropped_games_count_without_local_records(tmp_path):
    out = {}
    bare_run(tmp_path).failure_metrics(2, out, rows(50, {4}), FakeDriver([]))
    assert out['game_failures'] == 1


def test_a_panic_storm_fails_the_run(tmp_path):
    r = bare_run(tmp_path)
    limit = max(3, int(100 * run.GAME_FAILURE_LIMIT))
    r.failure_metrics(1, {}, rows(100, set(range(limit))), FakeDriver([failure(g) for g in range(limit)]))
    with pytest.raises(RuntimeError, match='broken build'):
        r.failure_metrics(2, {}, rows(100, set(range(limit + 1))), FakeDriver([failure(g) for g in range(limit + 1)]))
    # the records were written before the run failed (what replays the storm)
    with open(os.path.join(tmp_path, run.GAME_FAILURES), encoding='utf-8') as f:
        assert sum(1 for _ in f) == 2 * limit + 1
