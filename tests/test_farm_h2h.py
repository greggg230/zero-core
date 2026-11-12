"""zero.farm_h2h (docs/az.md 42.2) against a fake coordinator: the settings resolved (eval_spec, eval_sims), both
checkpoints exported with their reference sets, the run / artifacts / gens / eval batch put in order, the
results polled, downloaded, credited as the coordinator credits them (a pair only from one lease: a lone game
held by its chunk and completed by a later chunk's `also`; dup_ids; other kinds ignored) and acked,
the readout (zero.h2h.summary's fields) of the farm's rows; --resume reads the same, --cancel pauses the run;
the admin token never printed. The fake plays the batch's games with zero.farm_eval.match_driver and the
uploaded graphs (OnnxEvaluator): what a native worker plays (tools/worker_eval_gate.py).

Signed definitions (zero-farm PROTOCOL.md 7, docs/az.md 42.3): farm-release is a stub (tests/farm_release_stub.py)
that records its arguments and registers the definition with the fake coordinator, which then accepts the eval batch
only when its job is the definition's template and its gens are the definition's files."""
import hashlib
import json
import os
import re
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip('onnx')
pytest.importorskip('onnxscript')
pytest.importorskip('onnxruntime')

from zero import farm_def as FD, farm_eval as FE, farm_h2h as FH, games as G  # noqa: E402
import farm_release_stub as STUB  # noqa: E402

TOKEN = 'secret-admin-token-xyz'
INFO = G.info('toy')
RT = 'c' * 24
# the worker build's signed manifest (its id: sha256[:24] of these bytes); its game is the repo's, not the settings'
MANIFEST = json.dumps({'manifest': 1, 'kind': 'worker', 'version': 'toy-test', 'game': 'zero-toy', 'requires_runtime': RT,
                       'entry': 'w.exe', 'files': []}).encode()
BID = hashlib.sha256(MANIFEST).hexdigest()[:24]


def _ckpts(tmp_path):
    import torch
    from zero.net import Net, save
    d = tmp_path / 'runs' / 'tr'
    os.makedirs(d / 'models')
    cfg = {'game': 'toy', 'eval_sims': 6, 'conc': 4, 'threads': 1, 'groups': 1, 'eval_select': 'cq'}
    json.dump(cfg, open(d / 'config.json', 'w'))
    paths = []
    for k in (3, 5):
        torch.manual_seed(k)
        paths.append(str(d / 'models' / f'iter_{k:04d}.pt'))
        save(Net(INFO['num_features'], INFO['num_actions'], hidden=32, blocks=1, game='toy').eval(), paths[-1], {'iter': k})
    return paths


class Fake:
    """The coordinator's desktop API in memory, and a farm that plays the eval batch when results are listed."""

    # leases: (lo, hi, machine, the chunk files' first games): lease 0 splits pair (2, 3) across its two chunks
    def __init__(self, leases=((0, 6, 'm_a', (0, 3)), (6, 10, 'm_b', (6,)), (4, 8, 'm_b', (4,)))):
        self.runs, self.objects, self.ckpts, self.results, self.calls = {}, {}, {}, [], []
        self.leases = list(leases)
        self.acked = 0
        self.played = None
        self.defs = os.environ.get('FARM_STUB_STATE')      # the stub farm-release registers definitions here

    def definition(self, run):
        return STUB.current(self.defs, 'run', run)

    def __call__(self, method, url, headers, data):
        assert headers['authorization'] == f'Bearer {TOKEN}'
        u = urlparse(url)
        p, q = u.path, parse_qs(u.query)
        self.calls.append((method, p))
        body = json.loads(data) if data and headers.get('content-type') == 'application/json' else None
        ok = lambda x: (200, json.dumps(x).encode())
        m = re.match(r'^/v1/admin/run/([a-z0-9-]+)$', p)
        if m and method == 'PUT':
            run = m.group(1)
            assert body['selfplay'] is None and body['build_id'] == BID and body['runtime'] == RT
            if body['eval']:
                assert body['eval']['checkpoints'] == [1, 2]
                assert all((run, g) in self.ckpts for g in (1, 2)), 'gens before the batch'
                # rev 6: leases only within the current definition - the batch's job is its template exactly
                df = self.definition(run)
                assert df, 'a definition before the batch'
                doc = df['doc']
                assert doc['jobs']['eval'] == body['eval']['job'] and body['build_id'] in doc['builds']
                assert doc['runtime'] == body['runtime'] and set(doc['checkpoints']) == {'1', '2'}
            self.runs[run] = body
            return ok({'ok': True, 'created': True})
        if m and method == 'GET':
            if m.group(1) not in self.runs:
                return 404, json.dumps({'error': 'not_found', 'message': f'no run {m.group(1)}'}).encode()
            ev = self.runs[m.group(1)]['eval']
            done = len({g for r in self.results if r.get('kind') == 'eval' for lo, hi in r['ids'] for g in range(lo, hi)})
            df = self.definition(m.group(1))
            return ok({'run': m.group(1), 'eval_state': ev and {'batch': ev['batch'], 'games': ev['games'], 'done': done,
                                                                 'leased': 0, 'unissued': ev['games'] - done},
                       'definition': df and {k: df[k] for k in ('version', 'sha256', 'key')}})
        m = re.match(r'^/v1/admin/artifact/(.+)$', p)
        if m and method == 'PUT':
            assert hashlib.sha256(data).hexdigest() == headers['x-sha256']
            self.objects[m.group(1)] = data
            return ok({'ok': True, 'key': m.group(1), 'size': len(data)})
        if m and method == 'GET':
            if m.group(1) == f'builds/{BID}/manifest.json':
                return 200, MANIFEST
            return 200, self.objects[m.group(1)]
        m = re.match(r'^/v1/admin/run/([a-z0-9-]+)/checkpoint/(\d+)$', p)
        if m and method == 'PUT':
            run, gen = m.group(1), int(m.group(2))
            assert run in self.runs
            for k, h in (('model', 'sha256'), ('reference', 'reference_sha256')):
                assert body[k] == f'ckpt/{run}/{gen}/' + ('model.onnx' if k == 'model' else 'reference.npz')
                assert hashlib.sha256(self.objects[body[k]]).hexdigest() == body[h]
            df = self.definition(run)
            assert df, 'the definition names the checkpoints before they are registered'
            c = df['doc']['checkpoints'][str(gen)]
            assert (c['model_sha256'], c['model_size']) == (body['sha256'], len(self.objects[body['model']]))
            assert (c['reference_sha256'], c['reference_size']) == (body['reference_sha256'], len(self.objects[body['reference']]))
            self.ckpts[(run, gen)] = body
            return ok({'ok': True, 'unchanged': False})
        m = re.match(r'^/v1/admin/run/([a-z0-9-]+)/results$', p)
        if m and method == 'GET':
            self.play(m.group(1))
            after = int(q['after'][0])
            res = [r for r in self.results if r['id'] > after][:int(q['limit'][0])]
            return ok({'results': res, 'cursor': res[-1]['id'] if res else after})
        m = re.match(r'^/v1/admin/run/([a-z0-9-]+)/ack$', p)
        if m and method == 'POST':
            self.acked = body['cursor']
            return ok({'ok': True, 'acked': body['cursor']})
        if p == '/v1/admin/machines':
            return ok({'machines': [{'machine_id': 'm_a', 'name': "Sam's PC"}, {'machine_id': 'm_b', 'name': 'Ann'}]})
        return 404, json.dumps({'error': 'not_found', 'message': p}).encode()

    def play(self, run):
        """Play the batch once (the match's games with the uploaded graphs), then hand them out as leases."""
        ev = (self.runs.get(run) or {}).get('eval')
        if not ev or self.played is not None:
            return
        import tempfile
        from zero.onnx_eval import OnnxEvaluator
        from zero.infer import drive
        job = ev['job']
        evs = {}
        with tempfile.TemporaryDirectory() as d:
            for gen, key in ((1, 'm'), (2, 'o')):
                path = os.path.join(d, f'{gen}.onnx')
                with open(path, 'wb') as f:
                    f.write(self.objects[self.ckpts[(run, gen)]['model']])
                evs[key] = OnnxEvaluator(path, threads=1)
            drv = FE.match_driver(job['settings'], job['seats'], ev['games'], job['seed'], paired=job['paired'])
            drive(drv, evs)
            self.played = {r['gid']: FE.row(r, job['seats'], True) for r in drv.take_results()}
        # a self-play result of another kind first (ignored)
        self.results.append({'id': 1, 'key': 'results/x/1', 'kind': 'selfplay', 'machine_id': 'm_a', 'ids': [], 'dup_ids': []})
        credited = set()
        rid = 2
        for i, (lo, hi, mach, cuts) in enumerate(self.leases):
            held = []                              # (seq, key, gid): a lone game of a pair, held by its chunk
            ends = list(cuts[1:]) + [hi]
            for seq, (a, b) in enumerate(zip(cuts, ends)):
                body = ''.join(FE.row_json(self.played[g]) + '\n' for g in range(a, b)).encode()
                key = f'results/{run}/l{i}/{seq}-{hashlib.sha256(body).hexdigest()[:16]}'
                self.objects[key] = body
                gs = set(range(a, b))
                own = sorted(g for g in gs if g ^ 1 in gs or any(h[2] == g ^ 1 for h in held))
                lone = sorted(g for g in gs if g not in own)
                also = [{'seq': h[0], 'key': h[1], 'ids': [[h[2], h[2] + 1]]} for h in held if h[2] ^ 1 in gs]
                held = [h for h in held if h[2] ^ 1 not in gs] + [(seq, key, g) for g in lone]
                ids = sorted(own + [x['ids'][0][0] for x in also])
                dup = sorted(g for g in ids if g in credited)
                credited |= set(ids)
                self.results.append({'id': rid, 'key': key, 'lease_id': f'l{i}', 'seq': seq, 'machine_id': mach, 'kind': 'eval',
                                     'batch': ev['batch'], 'games': len(ids), 'ids': [[g, g + 1] for g in own],
                                     'also': also, 'dup_ids': [[g, g + 1] for g in dup],
                                     'sha256': hashlib.sha256(body).hexdigest(), 'size': len(body), 't': 0})
                rid += 1


def test_a_head_to_head_end_to_end(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv('ZEROFARM_ADMIN_TOKEN', TOKEN)
    stub = STUB.setup(monkeypatch, tmp_path, TOKEN)
    a, b = _ckpts(tmp_path)
    fake = Fake()
    cl = FH.Client('http://fake', transport=fake)
    st_dir = str(tmp_path / 'state')
    rc = FH.main(['--a', a, '--b', b, '--games', '10', '--build', BID, '--runtime', RT, '--state-dir', st_dir,
                  '--poll', '0', '--threads', '1', '--seed', '3'], client=cl)
    assert rc == 0
    out, err = capsys.readouterr()
    assert TOKEN not in out + err and TOKEN not in json.dumps(fake.runs)
    run = 'h2h-tr-3v5'
    assert run in fake.runs
    # the order: run, gen 1 (model, reference, checkpoint), gen 2, the eval batch
    puts = [c for c in fake.calls if c[0] == 'PUT']
    assert puts[0] == ('PUT', f'/v1/admin/run/{run}') and puts[-1] == ('PUT', f'/v1/admin/run/{run}')
    # the definition (version 1: the coordinator had none), signed and published once, between the run and its gens
    calls = STUB.calls(stub['log'])
    assert len(calls) == 1 and calls[0]['token_env'] and calls[0]['token_ok']
    dpath = os.path.join(str(tmp_path / 'definitions'), 'run', run, '1', 'definition.json')
    assert calls[0]['argv'] == ['define', os.path.abspath(dpath), '--key', stub['key'], '--publish', '--url', 'http://fake']
    assert not any(TOKEN in x for x in calls[0]['argv'])
    assert os.path.exists(dpath + '.minisig')
    i_def = fake.calls.index(('GET', f'/v1/admin/run/{run}'))
    assert fake.calls.index(puts[0]) < i_def < fake.calls.index(puts[1])
    with open(dpath, 'rb') as f:
        raw = f.read()
    doc = json.loads(raw)
    assert raw == FD.encode(doc)
    assert (doc['kind'], doc['id'], doc['version'], doc['game']) == ('run', run, 1, 'zero-toy')   # the build's game
    assert doc['builds'] == [BID] and doc['runtime'] == RT and doc['expires'] == doc['created'] + 30 * 86400
    assert doc['jobs'] == {'selfplay': None, 'eval': fake.runs[run]['eval']['job'], 'fuzz': None}
    for g in (1, 2):
        ck = fake.ckpts[(run, g)]
        assert doc['checkpoints'][str(g)] == {'model_sha256': ck['sha256'], 'model_size': len(fake.objects[ck['model']]),
                                              'reference_sha256': ck['reference_sha256'],
                                              'reference_size': len(fake.objects[ck['reference']])}
    assert STUB.current(stub['state'], 'run', run)['sha256'] == FD.file_sha256(dpath)
    assert [c[1] for c in puts[1:4]] == [f'/v1/admin/artifact/ckpt/{run}/1/model.onnx', f'/v1/admin/artifact/ckpt/{run}/1/reference.npz',
                                         f'/v1/admin/run/{run}/checkpoint/1']
    ev = fake.runs[run]['eval']
    assert ev['games'] == 10 and ev['batch'] == 'h2h'
    job = ev['job']
    assert job['seats'] == [{'net': 1}, {'net': 2}] and job['paired'] is True and job['seed'] == 3
    s = job['settings']
    assert s['eval_sims'] == 6 and s['game'] == 'toy' and s['eval_spec'] and s['eval_select'] == 'cq'
    # the readout: zero.h2h.summary of the farm's rows (each game once: the duplicates of lease 3 skipped)
    full = json.loads(out.strip().splitlines()[-1])
    want = FE.summary(list(fake.played.values()), True, {'net': 1})
    for k in ('games', 'a_score', 'se', 'a_margin', 'elo_diff', 'elo_se', 'pairs', 'se_games', 'unfinished', 'elo_95'):
        assert full[k] == want[k], k
    assert full['games'] == 10 and full['machines'] == {"Sam's PC": 6, 'Ann': 4}
    assert full['select_a'] == 'cq' and full['run'] == run and full['paired'] is True
    assert 'Elo A - B' in err and 'machines:' in err and 'unissued)' in err
    assert fake.acked == 5
    assert any(r.get('also') for r in fake.results), 'the test splits a pair across a lease\'s chunks'
    rows = FH.rows_of(os.path.join(st_dir, run))
    assert sorted(r['gid'] for r in rows) == list(range(10))
    # --resume reads the same; --cancel pauses the run (its batch kept)
    FH.main(['--resume', run, '--state-dir', st_dir, '--poll', '0'], client=cl)
    again = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert again['a_score'] == full['a_score'] and again['games'] == 10
    assert len(STUB.calls(stub['log'])) == 1, 'a published definition is not published again'
    with open(os.path.join(st_dir, run, 'state.json'), encoding='utf-8') as f:
        st = json.load(f)
    assert st['definition']['published'] and st['definition']['version'] == 1
    FH.main(['--cancel', run, '--state-dir', st_dir], client=cl)
    assert fake.runs[run]['state'] == 'paused' and fake.runs[run]['eval']['batch'] == 'h2h'
    # a runtime the build does not require: refused before the export, nothing put
    n = len(fake.calls)
    with pytest.raises(SystemExit, match='requires runtime'):
        FH.main(['--a', a, '--b', b, '--games', '10', '--build', BID, '--runtime', 'd' * 24, '--state-dir', st_dir,
                 '--name', 'other'], client=cl)
    assert [c for c in fake.calls[n:] if c[0] != 'GET'] == [] and not os.path.exists(os.path.join(st_dir, 'h2h-other'))
    # a second start of the same run is refused (resume it)
    with pytest.raises(SystemExit):
        FH.main(['--a', a, '--b', b, '--games', '10', '--build', BID, '--runtime', RT, '--state-dir', st_dir],
                client=cl)


def test_zero_h2h_plays_the_runs_evaluation(tmp_path, monkeypatch, capsys):
    """zero.h2h of a run's checkpoints plays Run.play_match's games (docs/az.md 42.3): each side's seat takes its run's
    margin_w, redet and eval_cache, the match A's run's evaluation spec - so its rows equal the farm's Python path
    (farm_eval.match_driver at the settings farm_h2h puts) game for game."""
    import torch
    from zero import h2h, package, run as R
    from zero.infer import NetEvaluator, drive
    from zero.net import load
    torch.set_num_threads(1)
    a, b = _ckpts(tmp_path)
    cfg_path = tmp_path / 'runs' / 'tr' / 'config.json'
    saved = json.load(open(cfg_path))
    saved.update(margin_w=0.5, redet=0, eval_cache=64, chance_k=1, c_puct=1.5, max_steps=400)
    json.dump(saved, open(cfg_path, 'w'))
    # the seats: Run.nn_seat's (farm_eval.seat_dicts) but for the evaluator key
    cfg = FE.resolve_eval_spec(h2h.run_settings(a))
    fs = FE.seat_dicts(cfg, [{'net': 1}, {'net': 2}])
    ma, mb, _ = h2h._load_pair(a, b, 'cpu', None)
    hs = h2h._seats(a, b, ma, mb, 'toy', cfg['eval_sims'], cfg['c_puct'], cfg['chance_k'], None, None, None, None)
    for f, s in zip(fs, hs):
        assert {k: v for k, v in f.items() if k != 'key'} == {k: v for k, v in s.items() if k != 'key'}
    assert hs[0]['margin_w'] == 0.5 and hs[0]['redet'] is False and hs[0]['eval_cache'] == 64
    # the games: h2h.match against the farm's Python path, 8 paired games
    n, seed = 8, 11
    out = h2h.match(a, b, games=n, sims=cfg['eval_sims'], chance_k=cfg['chance_k'], c_puct=cfg['c_puct'], seed=seed,
                    threads=cfg['threads'], groups=cfg['groups'], conc=min(cfg['conc'], n), max_steps=cfg['max_steps'],
                    paired=True, rows=True, sparse=bool(cfg.get('sparse')))
    drv = FE.match_driver(cfg, [{'net': 1}, {'net': 2}], n, seed, paired=True)
    drive(drv, {'m': NetEvaluator(load(a, 'cpu')[0], 'cpu'), 'o': NetEvaluator(load(b, 'cpu')[0], 'cpu')})
    key = lambda r: (r['gid'], list(r['result']), list(r['scores']), r['rounds'], r['over'])
    farm = sorted((key(r) for r in drv.take_results()), key=lambda x: x[0])
    assert [key(r) for r in out['rows']] == farm and len(farm) == n
    # the spec: the run's eval_spec, else the package's eval spec of its spec, else (no run) the package's default
    saved['eval_spec'] = '{"x": 1}'
    json.dump(saved, open(cfg_path, 'w'))
    assert h2h.run_eval_spec(a, 'toy') == '{"x": 1}' and h2h.match_spec(a, b, 'toy') == '{"x": 1}'
    assert h2h.match_spec(a, b, 'toy', '{"y": 2}') == '{"y": 2}'
    saved['eval_spec'] = ''
    json.dump(saved, open(cfg_path, 'w'))
    monkeypatch.setitem(package.active().eval_specs, 'toy', lambda spec: '{"evaluated": "' + spec + '"}')
    assert h2h.run_eval_spec(a, 'toy') == '{"evaluated": "' + saved.get('spec', R.DEFAULTS['spec']) + '"}'
    os.remove(cfg_path)
    assert h2h.run_eval_spec(a, 'toy') == package.active().specs.get('toy') and h2h.run_seat_keys(a) == {}


def test_settings_slug_and_refusals(tmp_path, monkeypatch):
    monkeypatch.setenv('ZEROFARM_ADMIN_TOKEN', TOKEN)
    assert FH.run_slug('runs/eoe1/models/iter_0080.pt', 'runs/eoe1/models/iter_0070.pt') == 'h2h-eoe1-80v70'
    with pytest.raises(SystemExit):
        FH.main(['--a', 'x', '--b', 'y', '--games', '9', '--build', 'b', '--runtime', 'c'])
    with pytest.raises(ValueError):
        FE.check_seats([{'net': 1}, {'bot': 'ext:jsbot'}])
    assert FE.elo_interval(0.5, 0.0) == (0.0, 0.0)
    lo, hi = FE.elo_interval(0.6, 0.02)
    assert lo < FE.elo_of(0.6) < hi
    # rows: seat 0's margin, A's sign by its seat (h2h's summary reads it as a local match's)
    rows = [{'gid': 0, 'pair': 0, 'seats': [{'net': 1}, {'net': 2}], 'result': [1.0, 0.0], 'margin': 3.0, 'rounds': 5, 'over': True},
            {'gid': 1, 'pair': 0, 'seats': [{'net': 2}, {'net': 1}], 'result': [1.0, 0.0], 'margin': 2.0, 'rounds': 5, 'over': True}]
    s = FE.summary(rows, True, {'net': 1})
    assert s['a_score'] == 0.5 and s['a_margin'] == 0.5 and s['pairs']['split'] == 1
    with pytest.raises(ValueError):
        FE.summary(rows, True, {'net': 2})
    assert FE.row_json(rows[0]) == '{"gid":0,"pair":0,"seats":[{"net":1},{"net":2}],"result":[1.0,0.0],"margin":3.0,"rounds":5,"over":true}'
