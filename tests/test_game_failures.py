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
    assert len(recs) == 1 and recs[0]['game_id'] == 17 and recs[0]['iter'] == 5 and recs[0]['moves'] == [3, 1, 4]
    assert recs[0]['source'] == 'local' and recs[0]['site'] == 'unknown' and out['panics'] == 1
    for k in ('t', 'iter', 'source', 'game_id', 'driver_seed', 'game_seed', 'players', 'spec', 'moves', 'message', 'site',
              'engine_commit', 'build', 'repro'):
        assert k in recs[0], k
    assert recs[0]['repro'].endswith(f'replay "{os.path.abspath(os.path.join(tmp_path, run.GAME_FAILURES))}" --line 1')
    assert any('game 17' in line and 'dropped after a panic' in line for line in r.lines)
    # the next iteration appends
    r.failure_metrics(6, {}, rows(100, {3}), FakeDriver([failure(3)]))
    with open(os.path.join(tmp_path, run.GAME_FAILURES), encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]
    assert [x['game_id'] for x in lines] == [17, 3] and lines[1]['repro'].endswith('--line 2')


def test_no_failure_changes_nothing(tmp_path):
    r = bare_run(tmp_path)
    out = {}
    r.failure_metrics(1, out, rows(10, set()), FakeDriver([]))
    assert out == {'panics': 0} and r.lines == [] and not os.path.exists(os.path.join(tmp_path, run.GAME_FAILURES))


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


class FakeRemote:
    def __init__(self, fails):
        self.fails = fails

    def take_failures(self):
        out, self.fails = self.fails, []
        return out


def test_remote_and_pool_failures_land_in_the_same_file(tmp_path):
    """docs/search.md 2.19: a remote worker's dropped game (its rows came back aborted) and the pool's (records only)
    are recorded with their source and the executor's build; `panics` counts each dropped game once."""
    r = bare_run(tmp_path)
    mac = dict(failure(4), engine_commit='c0ffee', build='b' * 24, site='src/x.rs:1:2')
    pool = dict(failure(9), build='f' * 24, message='thread panicked at engine/src/y.rs:10:5:' + chr(10) + 'boom 12')
    r.rsp = FakeRemote([(mac, 'mac'), (pool, 'pool')])
    out = {}
    r.failure_metrics(3, out, rows(60, {4}), FakeDriver([]))
    assert out['panics'] == 2 and out['game_failures'] == 1
    with open(os.path.join(tmp_path, run.GAME_FAILURES), encoding='utf-8') as f:
        recs = [json.loads(line) for line in f]
    assert [(x['source'], x['game_id'], x['build']) for x in recs] == [('mac', 4, 'b' * 24), ('pool', 9, 'f' * 24)]
    assert recs[0]['engine_commit'] == 'c0ffee' and recs[0]['site'] == 'src/x.rs:1:2'
    assert recs[1]['site'] == 'engine/src/y.rs:10:5'


def test_failure_key_masks_digits_and_keys_by_site():
    from zero import game_failure as GF
    a = GF.failure_key('a.rs:1:2', 'index 12 out of range' + chr(10) + 'more 3')
    assert a == GF.failure_key('a.rs:1:2', 'index 99 out of range') != GF.failure_key('b.rs:1:2', 'index 12 out of range')


def test_the_pool_takes_failures_from_a_records_header():
    from zero import remoteplay
    rp = object.__new__(remoteplay.RemotePlay)
    import threading
    rp.mu, rp.failures = threading.Condition(), []
    rp.add_failures('pool', [{'gid': 1}])
    assert rp.take_failures() == [({'gid': 1}, 'pool')] and rp.take_failures() == []


def test_a_remote_worker_s_failure_event_reaches_the_run(monkeypatch):
    """The remote worker sends each dropped game's record as a `failure` event (with its checkout's commit and engine
    build); the run side keeps it with the remote's name as its source."""
    import threading
    from types import SimpleNamespace
    from zero import game_failure as GF, remoteplay
    w = object.__new__(remoteplay.Worker)
    sent = []
    w.emit = lambda ev, **kw: sent.append(dict(ev=ev, **kw))
    monkeypatch.setattr(GF, 'local_ids', lambda: ('abc123', 'e' * 24))
    w._failure(json.dumps(failure(2)))
    assert sent == [{'ev': 'failure', 'failure': failure(2), 'engine_commit': 'abc123', 'build': 'e' * 24}]
    rp = object.__new__(remoteplay.RemotePlay)
    rp.mu, rp.failures, rp.co = threading.Condition(), [], None
    s = SimpleNamespace(h=SimpleNamespace(name='mac'))
    rp._event(s, sent[0])
    (f, src), = rp.take_failures()
    assert src == 'mac' and f['gid'] == 2 and (f['engine_commit'], f['build']) == ('abc123', 'e' * 24)
