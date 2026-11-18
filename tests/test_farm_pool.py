"""zero.farm_pool (docs/az.md 45): the farm pool of a run's self-play against a fake coordinator - each iteration's
checkpoint published (exported, uploaded, a new definition version, then registered current), the farm's records
replayed into ordinary shards of data/iter_i/ whose rows equal, bit for bit, the shards the farm's games would have
written directly (auxiliary targets and value-only rows included), the iteration's quota counted with the run's own
games, the games beyond it held for the next iteration, stale ones dropped, a tampered record refused (counted against
its machine and kept in debug/pool/), the generation's running total posted before each ack; and the record format
round trip of a traced driver (`take_game_records` -> `replay_rows`)."""
import gzip
import hashlib
import json
import os
import threading
import time
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

from zero import farm_def as FD, farm_pool as FP, games as G
from zero.data import ShardWriter, read_shard, shard_format, shard_paths
from zero.infer import drive
from zero.run import DEFAULTS
from zero.workqueue import split_games

TOKEN = 'pool-test-token'
FARM = 'toy-farm'
BUILD = 'b' * 24
INFO = G.info('toy')


class Hashed:
    """Uniform priors; a value from a hash of each row's features (q varies across rows)."""

    def __call__(self, lb):
        off = lb.opt_off
        cnt = np.diff(off)
        pri = np.repeat((1.0 / np.maximum(cnt, 1)).astype(np.float32), cnt)
        x = lb.x
        h = (np.asarray(x, np.float32) * ((np.arange(x.shape[1]) * 7 % 13) - 6.0)).sum(1)
        w = (1.0 / (1.0 + np.exp(-h * 0.1))).astype(np.float32)
        v = np.zeros((lb.B, 8), np.float32)
        v[:, 0] = w
        v[:, 1] = 1.0 - w
        return pri, v, np.zeros((lb.B, 8), np.float32)


def farm_games(n, seed, net=0, sims=8, spec=''):
    """A farm worker's games 0..n-1 (a traced rolling toy driver seeded as the farm's: auxiliary targets, value-only
    rows) -> [(gid, record text, the game's samples dict or None)] in the order they finished."""
    seat = G.nn_seat('m', sims, selfplay=True, sims_fast=2, p_full=0.5, view_samples=0.5)
    drv = G.Driver('toy', [[seat, seat]], spec=spec, concurrency=6, seed=seed, record=True, threads=2, groups=1,
                   rolling=True, aux=True)
    drv.set_trace(True)
    out = []
    while len(out) < n:
        drv.resume(n - len(out), net)
        drive(drv, {'m': Hashed()})
        rows, texts, s = drv.take_game_records()
        for (row, sd), t in zip(split_games(rows, s), texts):
            out.append((row['gid'], t, sd))
    return out[:n]


def records_file(header, texts):
    return gzip.compress(('\n'.join([json.dumps(header)] + texts) + '\n').encode())


def header(seed, spec='{}', build=BUILD, aux=True):
    return {'records': 1, 'game': 'toy', 'spec': spec, 'aux': aux, 'features': INFO['num_features'],
            'actions': INFO['num_actions'], 'aux_width': 8, 'seed': seed, 'lease': 'l', 'build_id': build}


class Fake:
    """The coordinator's desktop API in memory."""

    def __init__(self):
        self.runs, self.objects, self.ckpts, self.results, self.calls, self.accepts = {}, {}, {}, [], [], []
        self.raw = []                                      # (method, path, request body bytes)
        self.acked = 0
        self.mu = threading.Lock()

    def add(self, machine, ids, body, gen=0, kind='selfplay'):
        with self.mu:
            key = f'results/{FARM}/{len(self.results)}'
            self.objects[key] = body
            self.results.append({'id': len(self.results) + 1, 'key': key, 'lease_id': f'l{len(self.results)}', 'seq': 0,
                                 'machine_id': machine, 'kind': kind, 'gen': gen, 'games': sum(b - a for a, b in ids),
                                 'ids': ids, 'dup_ids': [], 'sha256': hashlib.sha256(body).hexdigest(), 'size': len(body),
                                 't': 0})

    def __call__(self, method, url, headers, data):
        assert headers['authorization'] == f'Bearer {TOKEN}'
        u = urlparse(url)
        p, q = u.path, parse_qs(u.query)
        body = json.loads(data) if data and headers.get('content-type') == 'application/json' else None
        with self.mu:
            self.calls.append((method, p, body))
            self.raw.append((method, p, data))
            if method == 'PUT' and p.startswith('/v1/admin/artifact/'):
                self.objects[p[len('/v1/admin/artifact/'):]] = data
                return 200, b'{"ok": true}'
            if method == 'GET' and p.startswith('/v1/admin/artifact/'):
                k = p[len('/v1/admin/artifact/'):]
                return (200, self.objects[k]) if k in self.objects else (404, b'{"error": "not_found"}')
            if p == f'/v1/admin/run/{FARM}' and method == 'PUT':
                self.runs[FARM] = body
                return 200, b'{"ok": true}'
            if p == f'/v1/admin/run/{FARM}' and method == 'GET':
                return (200, json.dumps(self.runs[FARM]).encode()) if FARM in self.runs else (404, b'{"error": "not_found"}')
            if p.startswith(f'/v1/admin/run/{FARM}/checkpoint/'):
                gen = int(p.rsplit('/', 1)[1])
                assert any(c[1] == f'/v1/admin/artifact/ckpt/{FARM}/{gen}/model.onnx' for c in self.calls), 'uploaded first'
                self.ckpts[gen] = body
                return 200, b'{"ok": true, "current": true}'
            if p == f'/v1/admin/run/{FARM}/results':
                after = int(q.get('after', ['0'])[0])
                lim = int(q.get('limit', ['500'])[0])
                rs = [r for r in self.results if r['id'] > after][:lim]
                return 200, json.dumps({'results': rs, 'cursor': rs[-1]['id'] if rs else after}).encode()
            if p == f'/v1/admin/run/{FARM}/accept':
                self.accepts.append((body['gen'], body['games'], self.acked))
                return 200, b'{"ok": true}'
            if p == f'/v1/admin/run/{FARM}/ack':
                self.acked = body['cursor']
                return 200, b'{"ok": true}'
        raise AssertionError(f'unexpected {method} {p}')


class Run:
    """The parts of zero.run.Run the pool reads."""

    def __init__(self, run, cfg):
        self.run, self.cfg, self.info, self.aux = str(run), cfg, INFO, ['next_card']
        self.lines = []

    def log(self, msg):
        self.lines.append(msg)

    def debug_dir(self):
        return os.path.join(self.run, 'debug')


def wait(cond, what, t=60):
    t0 = time.time()
    while not cond():
        if time.time() - t0 > t:
            raise AssertionError(f'timed out: {what}')
        time.sleep(0.02)


@pytest.fixture
def pool(tmp_path, monkeypatch):
    fake = Fake()
    real = FD.Client
    monkeypatch.setattr(FD, 'Client', lambda url=FD.URL, **k: real(url, token=TOKEN, transport=fake, tries=1))
    published = []

    def export(self, ckpt, gen, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        out = {}
        for k, name in (('model', 'model.onnx'), ('reference', 'reference.npz')):
            p = os.path.join(out_dir, name)
            with open(p, 'wb') as f:
                f.write(f'{k} of gen {gen}'.encode())
            out[k], out[f'{k}_sha256' if k == 'reference' else 'sha256'] = p, FD.file_sha256(p)
            out[f'{k}_size'] = os.path.getsize(p)
        return out

    def define(self, client, info):
        doc = json.load(open(info['path'], encoding='utf-8'))
        published.append(doc)
        fake.runs.setdefault(FARM, {})['definition'] = {'version': info['version'], 'sha256': info['sha256'], 'key': info['key']}
        return dict(info, published=True)

    monkeypatch.setattr(FP.Pool, '_export', export)
    monkeypatch.setattr(FP.Pool, '_define', define)
    monkeypatch.setattr(FP, 'play_settings', lambda worker, build, cfg, d, log=None: {'game': 'toy', 'spec': '', 'players': [2]})
    cfg = dict(DEFAULTS, game='toy', games=24, players=[2], spec='', shard_rows=40, shard_sparse=1, sparse=1,
               selfplay_remote=f'pool:{FARM}', pool_build=BUILD, pool_poll_s=0.05, pool_url='http://fake',
               pool_rolling='')      # classic leases (the default is 'auto' since 2026-11-17: test_default_...)
    run = tmp_path / 'run'
    os.makedirs(run / 'models')
    return fake, Run(run, cfg), published


def test_records_round_trip_a_traced_driver():
    seed = FP.coordinator_seed(FARM)
    games = farm_games(12, seed)
    E = G.engine()
    rows = views = 0
    for gid, text, sd in games:
        s, info = E.replay_rows('toy', '{}', text, True)
        assert info['gid'] == gid and info['seed'] == E.game_seed_of(seed, gid)
        if sd is None:
            assert s is None
            continue
        assert set(s) == set(sd)
        for k in sd:
            a, b = np.asarray(sd[k]), np.asarray(s[k])
            if k in ('pi', 'q'):
                a, b = a.astype(np.float16), b.astype(np.float16)    # (a record carries them as float16, as shards do)
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), k
        rows += len(sd['player'])
        views += int((np.diff(sd['opt_off']) == 0).sum())
    assert rows > 0 and views > 0
    with pytest.raises(ValueError, match='record: '):
        bad = json.loads(games[0][1])
        bad['steps'][3] = 999
        E.replay_rows('toy', '{}', json.dumps(bad), True)


def test_the_pool_publishes_ingests_counts_and_holds(pool):
    from zero.remoteplay import RemotePlay
    fake, R, published = pool
    seed = FP.coordinator_seed(FARM)
    games = farm_games(30, seed)
    rp = RemotePlay(R, R.cfg['selfplay_remote'])
    try:
        assert not rp.hosts and list(rp.pools) == [f'pool:{FARM}']
        rp.begin(1)
        wait(lambda: 0 in fake.ckpts, 'gen 0 registered')
        # the order: run (no self-play) -> artifacts -> definition -> checkpoint registered -> run with self-play
        calls = [(m, p) for m, p, _ in fake.calls]
        i_def = len(published)
        assert i_def == 1 and published[0]['checkpoints'].keys() == {'0'}
        assert published[0]['jobs'] == {'selfplay': {'settings': {'game': 'toy', 'spec': '', 'players': [2]}}, 'eval': None,
                                        'fuzz': None}
        assert calls.index(('PUT', f'/v1/admin/run/{FARM}')) < calls.index(('PUT', f'/v1/admin/artifact/ckpt/{FARM}/0/model.onnx'))
        assert fake.runs[FARM]['selfplay']['max_stale'] == 1 and fake.runs[FARM]['selfplay']['quota'] == 24
        first_put = next(b for m, p, b in fake.calls if m == 'PUT' and p == f'/v1/admin/run/{FARM}')
        assert first_put['selfplay'] is None, 'no self-play before a definition signs a gen'
        # (zero-farm PROTOCOL rev 9: the farm run's leases go only to agents that take its template)
        assert all(b['min_agent'] == '0.2.8' for m, p, b in fake.calls if m == 'PUT' and p == f'/v1/admin/run/{FARM}')
        # this machine plays 4 of the iteration's 24 games itself
        assert rp.accept_local(1, 4) == 4
        by = {gid: (t, sd) for gid, t, sd in games}
        order = [gid for gid, _, _ in games]
        ga, gb = [g for g in order if g < 16], [g for g in order if g >= 16]
        hi = max(order) + 1
        fake.add('m_a', [[0, 16]], records_file(header(seed), [by[g][0] for g in ga]))
        # a tampered record (its first game's move changed): the whole chunk refused, counted against its machine
        tam = json.loads(by[order[0]][0])
        tam['steps'][2] = 999
        fake.add('m_bad', [[100, 101]], records_file(header(seed), [json.dumps(dict(tam, gid=100))]))
        fake.add('m_b', [[16, hi]], records_file(header(seed), [by[g][0] for g in gb]))
        held = len(ga) + len(gb) - 20
        wait(lambda: rp.need(1) == 0, 'the quota met')
        out = rp.end(1)
        pr = out['pools'][f'pool:{FARM}']
        assert pr['games'] == 20 and pr['refused'] == 1 and pr['held'] == held, pr
        assert out['queue']['quota']['by'] == {'local': 4, f'pool:{FARM}': 20}
        st = json.load(open(os.path.join(R.run, 'pool', FARM, 'state.json')))
        assert st['refused'] == {'m_bad': 1}
        assert os.listdir(os.path.join(R.run, 'debug', 'pool', 'm_bad'))
        # the running total for gen 0 (20 accepted + 10 held) went out before the ack of those results
        wait(lambda: any(g == 0 and n == 20 + held for g, n, _ in fake.accepts), 'the total posted')
        first = next(acked for g, n, acked in fake.accepts if (g, n) == (0, 20 + held))
        assert first < 3, 'posted before the ack of the results it counts'
        wait(lambda: fake.acked == 3, 'acked')
        # the rows: the pool's shards == the shards of the same games written directly (arrival order, per machine)
        d = os.path.join(R.run, 'data', 'iter_0001')
        mine = [os.path.basename(p) for p in shard_paths(d)]
        assert mine and all(m.startswith('pool-m_') for m in mine)
        accepted = ga + gb[:20 - len(ga)]
        check_shards(R, d, {'m_a': [g for g in accepted if g < 16], 'm_b': [g for g in accepted if g >= 16]}, by, 1)
        # iteration 2: the held games come first (their network 0 >= 2 - 1 - max_stale)
        rp.begin(2)
        wait(lambda: 1 in fake.ckpts, 'gen 1 registered')
        assert set(published[-1]['checkpoints']) == {'0', '1'}, 'the gen and max_stale gens before it'
        wait(lambda: rp.pool_iteration() == 2 and rp.need(2) == 24 - held, 'the held games accepted')
        # iteration 3: a game of network 0 is too stale (0 < 3 - 1 - 1): dropped
        rp.accept_local(2, 24 - held)
        out2 = rp.end(2)
        assert out2['pools'][f'pool:{FARM}']['games'] == held
        check_shards(R, os.path.join(R.run, 'data', 'iter_0002'), {'m_b': gb[20 - len(ga):]}, by, 2)
        rp.begin(3)
        wait(lambda: 2 in fake.ckpts, 'gen 2 registered')
        extra = farm_games(2, seed)
        fake.add('m_a', [[0, max(g for g, _, _ in extra) + 1]], records_file(header(seed), [t for _, t, _ in extra]))
        wait(lambda: rp.pools[f'pool:{FARM}'].cur.get('stale_dropped', 0) == 2, 'stale games dropped')
        out3 = rp.end(3)
        assert out3['pools'][f'pool:{FARM}']['stale_dropped'] == 2 and out3['pools'][f'pool:{FARM}']['games'] == 0
    finally:
        rp.close()


def check_shards(R, d, by_machine, by, it):
    meta0 = {'game': 'toy', 'features': INFO['num_features'], 'actions': INFO['num_actions'], 'iter': it, 'source': 'pool',
             'farm_run': FARM, **shard_format(R.cfg)}
    for m, gids in by_machine.items():
        ref = os.path.join(R.run, f'ref-{m}-{it}')
        w = ShardWriter(ref, f'pool-{m}', rows=R.cfg['shard_rows'], meta=dict(meta0, machine=m))
        for g in gids:
            if by[g][1] is not None:
                w.add({k: np.array(v) for k, v in by[g][1].items()})
        w.close()
        want = shard_paths(ref)
        got = [p for p in shard_paths(d) if os.path.basename(p).startswith(f'pool-{m}_')]
        assert len(want) == len(got) and want
        for a, b in zip(want, got):
            x, y = np.load(a), np.load(b)
            assert sorted(x.files) == sorted(y.files)
            for k in x.files:
                if k == 'meta':
                    assert json.loads(str(x[k])) == json.loads(str(y[k]))
                    continue
                assert x[k].dtype == y[k].dtype and x[k].shape == y[k].shape and x[k].tobytes() == y[k].tobytes(), (m, k)
        assert read_shard(got[0])['meta']['source'] == 'pool'


class QuotaRP:
    """The parts of zero.remoteplay.RemotePlay a Pool calls: iteration 1 open, a quota of `quota` games."""

    def __init__(self, R, quota=100):
        self.R, self.quota, self.taken = R, quota, 0

    def pool_iteration(self):
        return 1

    def accept_pool(self, it, name, n):
        if self.taken + n > self.quota:
            return False
        self.taken += n
        return True


def _manifests(*mans):
    raws = [json.dumps(m).encode() for m in mans]
    return [(hashlib.sha256(r).hexdigest()[:24], r) for r in raws]


def test_pool_alternates_in_the_puts_definitions_and_records(pool):
    """pool_alternates and pool_gpu_games_per_lease (zero-farm PROTOCOL rev 11, 9.4). Without them (the defaults) every
    PUT of the farm run is, byte for byte, the body of before and each definition's builds are [pool_build]. Resumed
    with an alternate (the worker signed for the CUDA pack) and a GPU lease size: the next gen's definition lists the
    alternate's build after the default one (runtime unchanged), the run is PUT again with `alternates` and
    `selfplay.gpu_games_per_lease` once that definition is registered, and the alternate's records are accepted
    (another build's are refused); the next start with the same settings PUTs nothing more."""
    fake, R, published = pool
    w = {'manifest': 1, 'kind': 'worker', 'version': 'toy-test', 'game': 'zero-toy', 'entry': 'w.exe', 'files': []}
    (rt, rt_raw), (cuda, cuda_raw) = _manifests(
        {'manifest': 1, 'kind': 'runtime', 'version': 'cpu-test', 'files': []},
        {'manifest': 1, 'kind': 'runtime', 'version': 'cuda-test', 'min_agent': '0.2.10',
         'gpu': {'vendor': 'nvidia', 'min_driver': '527.41', 'providers': ['cuda']}, 'files': []})
    (build, build_raw), (alt, alt_raw) = _manifests(dict(w, requires_runtime=rt), dict(w, requires_runtime=cuda))
    for bid, raw in ((rt, rt_raw), (cuda, cuda_raw), (build, build_raw), (alt, alt_raw)):
        fake.objects[f'builds/{bid}/manifest.json'] = raw
    R.cfg.update(pool_build=build, pool_runtime=rt)
    name = f'pool:{FARM}'
    run_puts = lambda: [d for m, p, d in fake.raw if m == 'PUT' and p == f'/v1/admin/run/{FARM}']

    def stop(p):
        p.close()
        p.th.join(10)
        assert not p.th.is_alive()
    # no alternates: the bodies of before (b0930da's _selfplay_body), byte for byte
    p = FP.Pool(QuotaRP(R), name)
    try:
        p.begin(1)
        wait(lambda: 0 in fake.ckpts and len(run_puts()) == 2, 'gen 0 registered, the run PUT twice')
    finally:
        stop(p)
    sp = {'quota': 24, 'slack': 0.1, 'max_stale': 1, 'games_per_lease': 'auto', 'gpu': 'any', 'job': {}}
    old = {'game': 'toy', 'build_id': build, 'runtime': rt, 'priority': 1.0, 'min_agent': '0.2.8', 'selfplay': sp,
           'eval': None, 'state': 'active'}
    assert run_puts() == [json.dumps(dict(old, selfplay=None)).encode(), json.dumps(old).encode()]
    doc0 = published[0]
    assert doc0['builds'] == [build] and doc0['runtime'] == rt and doc0['game'] == 'zero-toy'
    assert doc0 == FD.definition('run', FARM, doc0['version'], 'zero-toy', [build], rt, doc0['checkpoints'], doc0['jobs'],
                                 doc0['created'], doc0['expires'])
    st = json.load(open(os.path.join(R.run, 'pool', FARM, 'state.json')))
    assert 'put_extras' not in st
    # a bad setting is refused when the pool starts
    for k, v in (('pool_alternates', 'xyz'), ('pool_alternates', f'{alt},{alt}'), ('pool_alternates', [alt] * 5),
                 ('pool_alt_vendor', 'NVIDIA'), ('pool_gpu_games_per_lease', 'lots'), ('pool_gpu_games_per_lease', -2)):
        R.cfg[k] = v
        with pytest.raises(SystemExit, match=k):
            FP.Pool(QuotaRP(R), name)
        R.cfg[k] = DEFAULTS[k]
    assert FP.alternate_ids(f' {alt}, {build} ') == [alt, build] and FP.alternate_ids([alt]) == [alt] and FP.alternate_ids('') == []
    # resumed with the CUDA alternate (a comma-separated setting) and GPU leases of 'auto' size (PROTOCOL rev 11)
    R.cfg.update(pool_alternates=alt, pool_gpu_games_per_lease='auto')
    rp = QuotaRP(R)
    p = FP.Pool(rp, name)
    try:
        assert alt in p.describe()
        p.begin(2)
        wait(lambda: 1 in fake.ckpts and len(run_puts()) == 3, 'gen 1 registered, the run PUT again')
        want = [{'build_id': alt, 'runtime': cuda, 'when': {'vendor': 'nvidia'}}]
        put = json.loads(run_puts()[-1])
        assert put == dict(old, alternates=want, selfplay=dict(sp, gpu_games_per_lease='auto'))
        assert list(put)[:6] == ['game', 'build_id', 'runtime', 'priority', 'min_agent', 'alternates']
        assert list(put['selfplay']) == ['quota', 'slack', 'max_stale', 'games_per_lease', 'gpu_games_per_lease', 'gpu', 'job']
        doc1 = published[-1]
        assert doc1['builds'] == [build, alt] and doc1['runtime'] == rt and set(doc1['checkpoints']) == {'0', '1'}
        i_reg = next(i for i, (m, pth, _) in enumerate(fake.calls) if pth == f'/v1/admin/run/{FARM}/checkpoint/1')
        i_put = max(i for i, (m, pth, _) in enumerate(fake.calls) if m == 'PUT' and pth == f'/v1/admin/run/{FARM}')
        assert i_reg < i_put, 'PUT with the alternate once a definition lists its build'
        wait(lambda: p.st.get('put_extras') == {'alternates': want, 'gpu_games_per_lease': 'auto'}, 'what was PUT kept')
        # records: the alternate's worker build is accepted, another build refused
        seed = FP.coordinator_seed(FARM)
        games = farm_games(4, seed)
        fake.add('m_gpu', [[0, max(g for g, _, _ in games) + 1]], records_file(header(seed, build=alt), [t for _, t, _ in games]))
        fake.add('m_x', [[100, 101]], records_file(header(seed, build='e' * 24), [json.dumps(dict(json.loads(games[0][1]), gid=100))]))
        wait(lambda: rp.taken == 4 and p.st['refused'].get('m_x') == 1, 'the alternate\'s games accepted, the other refused')
        assert p.st['refused'] == {'m_x': 1}
    finally:
        stop(p)
    # the same alternates on the next start: no extra PUT
    n = len(run_puts())
    p = FP.Pool(QuotaRP(R), name)
    try:
        p.begin(3)
        wait(lambda: 2 in fake.ckpts, 'gen 2 registered')
    finally:
        stop(p)
    assert len(run_puts()) == n and published[-1]['builds'] == [build, alt]


def test_play_settings_of_a_worker_older_than_trainer_settings(tmp_path, monkeypatch):
    """A worker build older than the pool's own settings (pool_alternates, pool_alt_vendor, pool_gpu_games_per_lease)
    calls them "unknown" in check-config: this zero-core's run_settings.json classifies them trainer-only, so each is
    logged and the template is exactly a current worker's; an "unknown" setting the desktop classifies play, or one
    it doesn't know at all, still refuses."""
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worker_stub.py')
    monkeypatch.setenv('WORKER_STUB_BUILD', BUILD)
    monkeypatch.delenv('WORKER_STUB_UNKNOWN', raising=False)
    cfg = dict(DEFAULTS, game='toy', players=[2], spec='')
    lines = []
    new = FP.play_settings(worker, BUILD, cfg, str(tmp_path / 'new'), lines.append)
    assert lines == [] and new and not any(k.startswith('pool_') for k in new)
    trainer = ['pool_alternates', 'pool_alt_vendor', 'pool_gpu_games_per_lease']
    monkeypatch.setenv('WORKER_STUB_UNKNOWN', ','.join(trainer))
    assert FP.play_settings(worker, BUILD, cfg, str(tmp_path / 'old'), lines.append) == new
    assert lines == [f'worker build {BUILD} doesn\'t know trainer-only setting {k}: ignored' for k in trainer]
    play_key = next(k for k in new if k != 'game')
    for c, unknown in ((cfg, f'pool_alternates,{play_key}'), (dict(cfg, no_such_setting=1), 'no_such_setting')):
        monkeypatch.setenv('WORKER_STUB_UNKNOWN', unknown)
        with pytest.raises(SystemExit, match='refuses the run\'s settings'):
            FP.play_settings(worker, BUILD, c, str(tmp_path / 'bad'), lines.append)


def test_the_pool_follows_the_phase_spec_of_the_game_package(pool, monkeypatch):
    """docs/az.md 52: an iteration whose phase spec (the game package's selfplay_specs hook, through Run.selfplay_spec)
    differs from the run's gets a definition whose self-play template plays it; the records played with it are
    accepted and replayed with it, a spec no gen was defined with is refused; the run's own spec stays accepted."""
    from zero.remoteplay import RemotePlay
    fake, R, published = pool
    monkeypatch.setattr(FP, 'play_settings', lambda worker, build, cfg, d, log=None: {'game': 'toy', 'spec': cfg['spec'],
                                                                                      'players': [2]})
    phase = '{"food": [20, 0]}'
    asked = []

    def selfplay_spec(it):
        asked.append(it)
        return phase if it >= 2 else R.cfg['spec']
    R.selfplay_spec = selfplay_spec
    seed = FP.coordinator_seed(FARM)
    rp = RemotePlay(R, R.cfg['selfplay_remote'])
    try:
        rp.begin(1)
        wait(lambda: 0 in fake.ckpts, 'gen 0 registered')
        assert published[-1]['jobs']['selfplay']['settings']['spec'] == '', 'iteration 1 plays the run\'s spec'
        assert rp.spec_of(1) == R.cfg['spec'] and rp.specs[1] is None
        rp.accept_local(1, 24)
        rp.end(1)
        rp.begin(2)
        rp.begin(2)                                # (a second begin of one iteration asks nothing again)
        assert asked == [1, 2]
        wait(lambda: 1 in fake.ckpts, 'gen 1 registered')
        assert published[-1]['jobs']['selfplay']['settings']['spec'] == phase, 'gen 1\'s template plays the phase spec'
        assert rp.spec_of(2) == phase
        st = json.load(open(os.path.join(R.run, 'pool', FARM, 'state.json')))
        assert st['gens']['1']['spec'] == phase and 'spec' not in st['gens']['0']
        p = rp.pools[f'pool:{FARM}']
        assert p.known_spec(phase) == phase and p.known_spec('') == p.spec and p.known_spec('{"food": [3, 0]}') is None
        games = farm_games(6, seed, net=1, spec=phase)
        hi = max(g for g, _, _ in games) + 1
        fake.add('m_a', [[0, hi]], records_file(header(seed, spec=phase), [t for _, t, _ in games]), gen=1)
        wait(lambda: rp.need(2) == 24 - 6, 'the phase spec\'s games accepted')
        other = farm_games(2, seed, net=1, spec='{"food": [3, 0]}')
        fake.add('m_c', [[0, max(g for g, _, _ in other) + 1]], records_file(header(seed, spec='{"food": [3, 0]}'),
                                                                              [t for _, t, _ in other]), gen=1)
        wait(lambda: json.load(open(os.path.join(R.run, 'pool', FARM, 'state.json'))).get('refused', {}).get('m_c') == 1,
             'a record of an unknown spec refused')
    finally:
        rp.close()


def test_a_remote_worker_takes_the_phase_spec(monkeypatch):
    """docs/az.md 52: a remote self-play worker's `phase` message may carry the phase's spec - the setups it queues
    from then on play it; without the key, the run's."""
    from zero.remoteplay import Worker

    class Stop(Exception):
        pass
    w = object.__new__(Worker)
    w.cfg = {'spec': '{"food": [1, 0]}'}

    def load(*a, **k):
        raise Stop()
    w._load = load
    for msg, want in (({'cmd': 'phase', 'iter': 2, 'model': 'm', 'spec': '{"food": [20, 0]}'}, '{"food": [20, 0]}'),
                      ({'cmd': 'phase', 'iter': 3, 'model': 'm'}, '{"food": [1, 0]}')):
        with pytest.raises(Stop):
            w._phase(msg)
        assert json.loads(w.spec) == json.loads(want)


def test_selfplay_local_0_leaves_the_games_to_a_delivering_farm(pool):
    from zero.remoteplay import RemotePlay
    fake, R, _ = pool
    R.cfg.update(selfplay_local=0, pool_live_s=1.0)
    seed = FP.coordinator_seed(FARM)
    rp = RemotePlay(R, R.cfg['selfplay_remote'])
    try:
        rp.local(2, 8)
        rp.begin(1)
        wait(lambda: 0 in fake.ckpts, 'gen 0 registered')
        assert rp.local_cap() > 0, 'no farm game yet: this machine plays'
        g = farm_games(2, seed)
        fake.add('m_a', [[0, max(x for x, _, _ in g) + 1]], records_file(header(seed), [t for _, t, _ in g]))
        wait(lambda: rp.local_cap() == 0, 'the farm delivers: this machine plays nothing')
        assert any('takes no games now' in ln for ln in R.lines)
        wait(lambda: rp.local_cap() > 0, 'the farm quiet for pool_live_s: this machine plays again')
    finally:
        rp.close()


def test_progress_polling_never_deadlocks_a_draining_backlog(pool):
    """The pb3 hang (docs/az.md 55): the run's thread polls RemotePlay.progress (RemotePlay.mu, then each pool's mu)
    while the pool's thread drains a large held backlog into the quota (once Pool.mu, then RemotePlay.mu per game) -
    the ABBA deadlock. With the lock order (Pool.mu innermost, never held across a RemotePlay call) both finish."""
    import sys
    from zero.remoteplay import RemotePlay
    fake, R, published = pool
    n = 4000
    R.cfg['games'] = n
    rp = RemotePlay(R, R.cfg['selfplay_remote'])
    p = rp.pools[f'pool:{FARM}']
    stuck = False
    old = sys.getswitchinterval()
    try:
        rp.begin(1)
        wait(lambda: 0 in fake.ckpts, 'gen 0 registered')
        # the backlog (as after a long pause): games without rows, so the drain is all lock traffic
        with p.mu:
            p.held.extend({'gid': i, 'machine': 'm', 'net': None, 's': None, 'rows': 0, 'replay_s': 0.0}
                          for i in range(n))
        done = threading.Event()
        polls = [0]

        def poll():
            while not done.is_set():
                rp.progress()
                polls[0] += 1

        def drain():
            p._drain()
            done.set()

        sys.setswitchinterval(1e-5)            # (many thread switches: the race at almost every game)
        ths = [threading.Thread(target=poll, daemon=True), threading.Thread(target=drain, daemon=True)]
        for th in ths:
            th.start()
        end = time.time() + 30                 # (one deadline for both: a deadlock fails the test in 30 s)
        for th in ths:
            th.join(max(0.0, end - time.time()))
        stuck = any(th.is_alive() for th in ths)
        assert not stuck, f'deadlock: progress() and the drain hung ({polls[0]} polls, {len(p.held)} games held)'
        assert polls[0] > 0
        sys.setswitchinterval(old)
        wait(lambda: rp.need(1) == 0, 'the quota met')
        assert rp.q['by'] == {f'pool:{FARM}': n} and not p.held
        out = rp.end(1)
        assert out['pools'][f'pool:{FARM}']['games'] == n
    finally:
        sys.setswitchinterval(old)
        if not stuck:                           # (a deadlocked RemotePlay.mu would hang close: the threads are daemons)
            rp.close()


# ---------------------------------------------------------------- rolling leases (zero-farm PROTOCOL rev 15, docs/az.md 54)
STUB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worker_stub.py')


def test_rolling_setting():
    assert FP.rolling_setting('') is None and FP.rolling_setting(None) is None and FP.rolling_setting('  ') is None
    assert FP.rolling_setting('auto') == {'inflight': 'auto', 'gpu_inflight': 'auto'}
    full = {'flush_s': 60, 'inflight': 16, 'gpu_inflight': 'auto', 'block': 4096, 'max_s': 3600}
    want = {'inflight': 16, 'gpu_inflight': 'auto', 'block': 4096, 'max_s': 3600, 'flush_s': 60}
    assert FP.rolling_setting(full) == want and list(FP.rolling_setting(full)) == list(want)
    assert FP.rolling_setting(json.dumps(full)) == want
    assert FP.rolling_setting({'inflight': 8}) == {'inflight': 8}
    for bad in ('lots', '[1]', {}, {'inflight': 0}, {'inflight': 4097}, {'inflight': True}, {'inflight': 2.0},
                {'max_s': 599}, {'flush_s': 5}, {'block': 0}, {'block': 'auto'}, {'speed': 1}):
        with pytest.raises(ValueError, match='rolling'):
            FP.rolling_setting(bad)


def test_pool_rolling_checks_the_worker_and_puts_rolling(pool, monkeypatch):
    """pool_rolling: a worker build that does not announce `rolling` is refused loudly (nothing published); one that
    does: every self-play PUT carries `selfplay.rolling` (and only then: off, test_pool_alternates_... checks the bodies
    of before byte for byte); an alternate whose executable is the default's is covered, another is logged unchecked."""
    fake, R, published = pool
    w = {'manifest': 1, 'kind': 'worker', 'version': 'toy-test', 'game': 'zero-toy', 'entry': 'w.exe'}
    (rt, rt_raw), (cuda, cuda_raw) = _manifests(
        {'manifest': 1, 'kind': 'runtime', 'version': 'cpu-test', 'files': []},
        {'manifest': 1, 'kind': 'runtime', 'version': 'cuda-test', 'gpu': {'vendor': 'nvidia', 'providers': ['cuda']},
         'files': []})
    exe = {'path': 'w.exe', 'sha256': 'a' * 64, 'size': 1}
    (build, build_raw), (alt, alt_raw), (other, other_raw) = _manifests(
        dict(w, requires_runtime=rt, files=[exe]), dict(w, requires_runtime=cuda, files=[exe]),
        dict(w, requires_runtime=cuda, version='toy-other', files=[dict(exe, sha256='c' * 64)]))
    for bid, raw in ((rt, rt_raw), (cuda, cuda_raw), (build, build_raw), (alt, alt_raw), (other, other_raw)):
        fake.objects[f'builds/{bid}/manifest.json'] = raw
    roll = {'inflight': 64, 'gpu_inflight': 'auto', 'flush_s': 60}
    R.cfg.update(pool_build=build, pool_runtime=rt, pool_worker=STUB, pool_rolling=json.dumps(roll),
                 pool_alternates=f'{alt},{other}')
    monkeypatch.setenv('WORKER_STUB_BUILD', build)
    monkeypatch.delenv('WORKER_STUB_FEATURES', raising=False)
    p = FP.Pool(QuotaRP(R), f'pool:{FARM}')
    try:
        assert 'rolling {"inflight": 64' in p.describe()
        p.begin(1)
        wait(lambda: p.error and 'pool_rolling' in p.error, 'a worker without rolling refused')
        assert "not 'rolling'" in p.error and not fake.ckpts and not published
        assert not any(m == 'PUT' and pth == f'/v1/admin/run/{FARM}' for m, pth, _ in fake.calls), 'nothing PUT'
        monkeypatch.setenv('WORKER_STUB_FEATURES', 'rolling')
        wait(lambda: 0 in fake.ckpts and p.st.get('selfplay_put'), 'gen 0 registered once the worker plays rolling jobs')
    finally:
        p.close()
        p.th.join(10)
    puts = [b for m, pth, b in fake.calls if m == 'PUT' and pth == f'/v1/admin/run/{FARM}']
    assert puts[0]['selfplay'] is None and puts[-1]['selfplay']['rolling'] == roll
    assert list(puts[-1]['selfplay'])[-1] == 'rolling'
    assert p.st['put_extras']['rolling'] == roll
    lines = '\n'.join(R.lines)
    assert f"alternate {other}'s executable is not the default build's" in lines
    assert f"alternate {alt}'s executable" not in lines, "the default build's executable: covered"
    assert 'the worker build plays rolling jobs' in lines


@pytest.mark.parametrize('setting', ['default', ''])
def test_default_pool_rolling_is_auto(pool, monkeypatch, setting):
    """The deliberate default of 2026-11-17 (docs/az.md 54.7): a farm pool with the default `pool_rolling` ('auto')
    PUTs `selfplay.rolling` {"inflight": "auto", "gpu_inflight": "auto"}; pool_rolling '' PUTs no `rolling` key."""
    from zero.run import _CORE_DEFAULTS
    assert _CORE_DEFAULTS['pool_rolling'] == 'auto'
    fake, R, published = pool
    if setting == 'default':
        del R.cfg['pool_rolling']             # (unset: the pool takes DEFAULTS', as for a run that never named it)
    else:
        R.cfg['pool_rolling'] = setting
    R.cfg['pool_worker'] = STUB
    monkeypatch.setenv('WORKER_STUB_BUILD', BUILD)
    monkeypatch.setenv('WORKER_STUB_FEATURES', 'rolling')
    p = FP.Pool(QuotaRP(R), f'pool:{FARM}')
    try:
        p.begin(1)
        wait(lambda: 0 in fake.ckpts and p.st.get('selfplay_put'), 'gen 0 registered')
    finally:
        p.close()
        p.th.join(10)
    puts = [b for m, pth, b in fake.calls if m == 'PUT' and pth == f'/v1/admin/run/{FARM}']
    sp = [b['selfplay'] for b in puts if b.get('selfplay')]
    assert sp
    if setting == 'default':
        assert all(s['rolling'] == {'inflight': 'auto', 'gpu_inflight': 'auto'} for s in sp)
        assert 'the worker build plays rolling jobs' in '\n'.join(R.lines)
    else:
        assert all('rolling' not in s for s in sp)


class StaleRP(QuotaRP):
    def __init__(self, R, it):
        super().__init__(R)
        self.it = it

    def pool_iteration(self):
        return self.it


def _held(sd, nets, gid, machine='m_a'):
    """A held game (as Pool._chunk keeps it: sparse rows) of samples `sd` whose rows' networks are `nets`."""
    s = {k: np.array(v) for k, v in sd.items()}
    s['net'] = np.asarray(nets, s['net'].dtype)
    s['xs'] = FP._sparse_rows(s)
    del s['x']
    return {'gid': gid, 'machine': machine, 'net': int(min(nets)), 's': s, 'rows': len(nets), 'replay_s': 0.0}


@pytest.mark.parametrize('rolling', [False, True])
def test_staleness_per_row_with_rolling(pool, rolling):
    """Iteration 3, max_stale 1: rows of a network < 1 are stale. Classic (pool_rolling off): a game is dropped when its
    oldest row is stale (as before). Rolling: only the stale rows are dropped, the rest of the game kept (its shard rows
    == the game's newer rows written directly), the game dropped only when none is left; the staleness histogram
    counts the kept rows."""
    from zero.protocol import take_rows
    fake, R, _ = pool
    if rolling:
        R.cfg['pool_rolling'] = 'auto'
    games = [sd for _, _, sd in farm_games(8, FP.coordinator_seed(FARM)) if sd is not None and len(sd['player']) >= 4][:3]
    assert len(games) == 3
    na, nc = len(games[0]['player']), len(games[2]['player'])
    nets_a = [0] * 2 + [1] * (na - 3) + [2]                  # straddles: 2 stale rows, the rest kept
    nets_b = [0] * len(games[1]['player'])                   # all stale
    nets_c = [2] * nc                                        # fresh
    p = FP.Pool(StaleRP(R, 3), f'pool:{FARM}')
    try:
        with p.mu:
            p.held.extend([_held(games[0], nets_a, 1), _held(games[1], nets_b, 2), _held(games[2], nets_c, 3)])
        p._drain()
        for wr in p.writers.values():
            wr.close()
        cur = p.cur
    finally:
        p.close()
        p.th.join(10)
    d = os.path.join(R.run, 'data', 'iter_0003')
    ref = os.path.join(R.run, 'ref-stale')
    meta = {'game': 'toy', 'features': INFO['num_features'], 'actions': INFO['num_actions'], 'iter': 3, 'source': 'pool',
            'machine': 'm_a', 'farm_run': FARM, **shard_format(R.cfg)}
    w = ShardWriter(ref, 'pool-m_a', rows=R.cfg['shard_rows'], meta=meta)
    for sd, nets in ((games[0], nets_a), (games[2], nets_c)):
        s = {k: np.array(v) for k, v in sd.items()}
        s['net'] = np.asarray(nets, s['net'].dtype)
        if rolling and sd is games[0]:
            s = take_rows(s, 2, na)
        if rolling or sd is games[2]:
            w.add(s)
    w.close()
    want, got = shard_paths(ref), shard_paths(d)
    assert len(want) == len(got) and want
    for a, b in zip(want, got):
        x, y = np.load(a), np.load(b)
        assert sorted(x.files) == sorted(y.files)
        for k in x.files:
            if k != 'meta':
                assert x[k].tobytes() == y[k].tobytes(), k
    if rolling:
        assert (cur['games'], cur['stale_dropped'], cur['stale_rows_dropped']) == (2, 1, 2 + len(nets_b))
        assert cur['rows'] == na - 2 + nc
        assert cur['stale'] == {'1': na - 3, '0': 1 + nc}, cur['stale']
    else:
        assert (cur['games'], cur['stale_dropped'], cur['stale_rows_dropped']) == (1, 2, 0)
        assert cur['stale'] == {'0': nc}


def test_keep_rows_is_take_rows_for_any_rows():
    """keep_rows (rolling staleness): any set of a sample dict's rows == the concatenation of protocol.take_rows of
    their runs, the variable-length rows' CSRs (tails, pointer overrides: toy_tail) and sparse features included."""
    from test_tail_shards import WIDTHS as widths, samples
    from zero.data import SparseRows
    from zero.protocol import concat_samples, take_rows
    s = samples(n_games=3)
    assert 't_off' in s and 'ov_off' in s and len(s['player']) > 12
    n = len(s['player'])
    for rows in ([0, 2, 3, n - 1], list(range(3, n)), [n // 2], list(range(n))):
        runs, a = [], rows[0]
        for i in range(1, len(rows) + 1):
            if i == len(rows) or rows[i] != rows[i - 1] + 1:
                runs.append((a, rows[i - 1] + 1))
                if i < len(rows):
                    a = rows[i]
        want = concat_samples([take_rows(s, x, y, widths) for x, y in runs])
        got = FP.keep_rows(s, rows, widths)
        assert sorted(got) == sorted(want)
        for k in want:
            g, wv = np.asarray(got[k]), np.asarray(want[k])
            assert g.dtype == wv.dtype and g.shape == wv.shape and g.tobytes() == wv.tobytes(), (rows, k)
        sp = dict(s, xs=SparseRows.from_dense(s['x'], s['x'].shape[1]))
        del sp['x']
        got = FP.keep_rows(sp, rows, widths)
        assert got['xs'].dense(np.arange(len(rows))).tobytes() == want['x'].astype(np.float16).tobytes()
