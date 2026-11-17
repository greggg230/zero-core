"""Game records (zero/gamerec.py, `game_records`, docs/az.md 59; the driver's game log, docs/search.md 2.20): off, a run
writes none; on, the same shards and checkpoints byte for byte, and every accepted game of every source - this
machine's driver, a remote worker's, the farm pool's - is one record that replays to its outcome; the reader takes only
the files DONE.json lists and a file cut short yields the lines before the cut."""
import base64
import glob
import gzip
import json
import os
import struct

import numpy as np
import pytest

from zero import farm_pool as FP, gamerec as GR, games as G
from zero.run import read_metrics

from test_loop import run_loop

import test_farm_pool as TFP
from test_farm_pool import pool  # noqa: F401  (the fixture)
from test_remoteplay import TOY, env_for


def replays(rec, spec):
    """A record played back from its seed: the engine's game at its end -> (over, result)."""
    E = G.engine()
    g = E.Game('toy', spec, rec['n'], rec['seed'])
    for a in GR.actions_of(rec):
        g.step(int(a))
    return g.to_move is None, list(g.result()) if g.to_move is None else None


def check_records(run, it, specs, recs):
    for r in recs:
        assert r['it'] == it and r['spec'] in specs and r['n_actions'] == len(GR.actions_of(r))
        over, res = replays(r, specs[r['spec']])
        if r['resigned'] is None and r['over']:
            assert over and [round(x, 6) for x in res[:r['n']]] == [round(x, 6) for x in r['result']], r['gid']
        elif r['resigned'] is not None:
            assert r['over'] and r['result'][r['resigned']] == 1.0


def shard_bytes(run):
    out = {}
    for p in sorted(glob.glob(os.path.join(str(run), 'data', '*', '*.npz'))):
        with np.load(p) as z:
            out[os.path.relpath(p, str(run))] = {k: z[k].tobytes() for k in z.files if k != 'meta'}
    return out


@pytest.fixture(scope='module')
def two_runs(tmp_path_factory):
    base = tmp_path_factory.mktemp('gamerec')
    out = {}
    for v in (0, 1):
        p = run_loop(base / f'r{v}', '--iters', '2', '--game-records', str(v))
        assert p.returncode == 0, p.stdout + p.stderr
        out[v] = base / f'r{v}'
    return out


def test_records_change_no_shard_and_no_weight(two_runs):
    import torch
    r0, r1 = two_runs[0], two_runs[1]
    assert not os.path.exists(r0 / 'records'), 'off: no records'
    a, b = shard_bytes(r0), shard_bytes(r1)
    assert a and a.keys() == b.keys() and all(a[k] == b[k] for k in a), 'the same shards byte for byte'
    for it in (1, 2):
        x = torch.load(r0 / 'models' / f'iter_{it:04d}.pt', weights_only=False)['state']
        y = torch.load(r1 / 'models' / f'iter_{it:04d}.pt', weights_only=False)['state']
        assert x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x)


def test_every_accepted_game_is_a_record_that_replays(two_runs):
    run = two_runs[1]
    assert GR.done_iterations(str(run)) == [1, 2]
    m = read_metrics(str(run))
    for it in (1, 2):
        d = GR.read_done(str(run), it)
        specs, recs = GR.read_iteration(str(run), it)
        assert len(recs) == d['games'] == m[it]['selfplay']['games'] == d['accepted']['local'] and 'missing' not in d
        assert m[it]['selfplay']['records']['games'] == d['games'] and d['bytes'] > 0
        assert sorted(r['gid'] for r in recs) == sorted(set(r['gid'] for r in recs)), 'each game once'
        check_records(run, it, specs, recs)
        assert all(r['src'] == 'local' for r in recs)
        assert all(0 <= r['nets'][0] <= r['nets'][1] <= it - 1 for r in recs if r['nets']) and any(r['nets'] for r in recs)
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'iter 2 game records: 24 games' in log


def test_a_remote_workers_games_are_recorded(tmp_path):
    run = tmp_path / 'sp'
    p = run_loop(run, '--iters', '3', *TOY, '--selfplay-remote', 'local', '--game-records', '1', env=env_for(tmp_path))
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    by_w0 = 0
    for it in (1, 2, 3):
        d = GR.read_done(str(run), it)
        specs, recs = GR.read_iteration(str(run), it)
        q = m[it]['selfplay']['queue']
        assert d['games'] == q['games'] == 600 and 'missing' not in d, d
        assert d['by_source'] == {k: v for k, v in d['accepted'].items()}
        check_records(run, it, specs, recs)
        by_w0 += d['by_source'].get('w0', 0)
    assert by_w0 > 0, 'the worker contributed recorded games'
    # an older worker checkout (no records): its accepted games are the iteration's `missing`, the run plays on
    run2 = tmp_path / 'old'
    p = run_loop(run2, '--iters', '3', *TOY, '--selfplay-remote', 'local', '--game-records', '1',
                 env=env_for(tmp_path / 'm2', test={'no_records': 1}))
    assert p.returncode == 0, p.stdout + p.stderr
    miss = [(GR.read_done(str(run2), it) or {}).get('missing') for it in (1, 2, 3)]
    assert any(x and x.get('w0') for x in miss) and all(set(x or {}) <= {'w0'} for x in miss), miss
    assert 'this checkout sends no game records' in (run2 / 'log.txt').read_text(encoding='utf-8')


def test_farm_records_of_accepted_games(pool):  # noqa: F811
    from zero.remoteplay import RemotePlay
    fake, R, _ = pool
    R.cfg['game_records'] = 1
    recs_w = GR.Records(R.run)
    R.gamerecs = lambda: recs_w
    seed = FP.coordinator_seed(TFP.FARM)
    games = TFP.farm_games(30, seed)
    by = {gid: t for gid, t, _ in games}
    rp = RemotePlay(R, R.cfg['selfplay_remote'])
    try:
        rp.begin(1)
        TFP.wait(lambda: 0 in fake.ckpts, 'gen 0 registered')
        rp.accept_local(1, 4)
        fake.add('m_a', [[0, max(by) + 1]], TFP.records_file(TFP.header(seed), [t for _, t, _ in games]))
        TFP.wait(lambda: rp.need(1) == 0, 'the quota met')
        out = rp.end(1)
    finally:
        rp.close()
    acc = out['queue']['quota']['by']
    s = R.gamerecs().close(1, acc)
    assert s['games'] == 20 and s['by_source'] == {f'pool:{TFP.FARM}': 20} and s['missing'] == {'local': 4}
    specs, recs = GR.read_iteration(R.run, 1)
    assert len(recs) == 20 and all(r['machine'] == GR.machine_tag('m_a') and 'm_a' not in json.dumps(r) for r in recs)
    for r in recs:
        src = json.loads(by[r['gid']])
        assert GR.actions_of(r) == src['steps'] and r['seed'] == src['seed'] and r['setup'] == src.get('setup', '')
        assert r['result'] == [struct.unpack('<f', struct.pack('<I', b))[0] for b in src['result']]
        nets = src['samples']['net']
        assert r['nets'] == ([min(nets), max(nets)] if nets else None)
    check_records(R.run, 1, specs, recs)


def test_the_reader_takes_listed_files_and_survives_a_cut(tmp_path):
    run = str(tmp_path)
    w = GR.Records(run)
    spec = '{"a": 1}'
    rec = {'v': 1, 'gid': 3, 'seed': 9, 'spec': GR.spec_key(spec), 'n': 2, 'actions': base64.b64encode(
        struct.pack('<3H', 1, 2, 3)).decode(), 'n_actions': 3}
    w.add(5, 'local', [(rec, spec), (dict(rec, gid=4), spec)])
    assert GR.read_iteration(run, 5) is None, 'no DONE.json yet'
    s = w.close(5, {'local': 2, 'w0': 1})
    assert s['games'] == 2 and s['missing'] == {'w0': 1}
    assert GR.spec_key('{"a":1}') == GR.spec_key(spec), 'one key for one spec value'
    # a file of another (interrupted) attempt is not listed: never read
    other = os.path.join(GR.iter_dir(run, 5), 'local-1.jsonl.gz')
    with gzip.open(other, 'wt') as f:
        f.write(json.dumps(dict(rec, gid=99)) + '\n')
    specs, recs = GR.read_iteration(run, 5)
    assert [r['gid'] for r in recs] == [3, 4] and GR.actions_of(recs[0]) == [1, 2, 3] and specs == {rec['spec']: spec}
    # cut short: the lines before the cut
    path = os.path.join(GR.iter_dir(run, 5), next(iter(GR.read_done(run, 5)['files'])))
    body = open(path, 'rb').read()
    big = gzip.compress(body[:0] + gzip.decompress(body) * 50)
    with open(path, 'wb') as f:
        f.write(big[:len(big) * 2 // 3])
    _, cut = GR.read_iteration(run, 5)
    assert 0 < len(cut) < 100
