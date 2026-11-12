"""zero.farm_h2h (docs/az.md 42.2) against a fake coordinator: the settings resolved (eval_spec, eval_sims), both
checkpoints exported with their reference sets, the run / artifacts / gens / eval batch put in order, the
results polled, downloaded, credited as the coordinator credits them (a pair only from one lease: a lone game
held by its chunk and completed by a later chunk's `also`; dup_ids; other kinds ignored) and acked,
the readout (zero.h2h.summary's fields) of the farm's rows; --resume reads the same, --cancel pauses the run;
the admin token never printed. The fake plays the batch's games with zero.farm_eval.match_driver and the
uploaded graphs (OnnxEvaluator): what a native worker plays (tools/worker_eval_gate.py)."""
import hashlib
import json
import os
import re
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip('onnx')
pytest.importorskip('onnxscript')
pytest.importorskip('onnxruntime')

from zero import farm_eval as FE, farm_h2h as FH, games as G  # noqa: E402

TOKEN = 'secret-admin-token-xyz'
INFO = G.info('toy')


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
            assert body['selfplay'] is None and body['build_id'] == 'b' * 24 and body['runtime'] == 'c' * 24
            if body['eval']:
                assert body['eval']['checkpoints'] == [1, 2]
                assert all((run, g) in self.ckpts for g in (1, 2)), 'gens before the batch'
            self.runs[run] = body
            return ok({'ok': True, 'created': True})
        if m and method == 'GET':
            ev = self.runs[m.group(1)]['eval']
            done = len({g for r in self.results if r.get('kind') == 'eval' for lo, hi in r['ids'] for g in range(lo, hi)})
            return ok({'run': m.group(1), 'eval_state': ev and {'batch': ev['batch'], 'games': ev['games'], 'done': done,
                                                                 'leased': 0, 'unissued': ev['games'] - done}})
        m = re.match(r'^/v1/admin/artifact/(.+)$', p)
        if m and method == 'PUT':
            assert hashlib.sha256(data).hexdigest() == headers['x-sha256']
            self.objects[m.group(1)] = data
            return ok({'ok': True, 'key': m.group(1), 'size': len(data)})
        if m and method == 'GET':
            return 200, self.objects[m.group(1)]
        m = re.match(r'^/v1/admin/run/([a-z0-9-]+)/checkpoint/(\d+)$', p)
        if m and method == 'PUT':
            run, gen = m.group(1), int(m.group(2))
            assert run in self.runs
            for k, h in (('model', 'sha256'), ('reference', 'reference_sha256')):
                assert body[k] == f'ckpt/{run}/{gen}/' + ('model.onnx' if k == 'model' else 'reference.npz')
                assert hashlib.sha256(self.objects[body[k]]).hexdigest() == body[h]
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
    a, b = _ckpts(tmp_path)
    fake = Fake()
    cl = FH.Client('http://fake', transport=fake)
    st_dir = str(tmp_path / 'state')
    rc = FH.main(['--a', a, '--b', b, '--games', '10', '--build', 'b' * 24, '--runtime', 'c' * 24, '--state-dir', st_dir,
                  '--poll', '0', '--threads', '1', '--seed', '3'], client=cl)
    assert rc == 0
    out, err = capsys.readouterr()
    assert TOKEN not in out + err and TOKEN not in json.dumps(fake.runs)
    run = 'h2h-tr-3v5'
    assert run in fake.runs
    # the order: run, gen 1 (model, reference, checkpoint), gen 2, the eval batch
    puts = [c for c in fake.calls if c[0] == 'PUT']
    assert puts[0] == ('PUT', f'/v1/admin/run/{run}') and puts[-1] == ('PUT', f'/v1/admin/run/{run}')
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
    FH.main(['--cancel', run, '--state-dir', st_dir], client=cl)
    assert fake.runs[run]['state'] == 'paused' and fake.runs[run]['eval']['batch'] == 'h2h'
    # a second start of the same run is refused (resume it)
    with pytest.raises(SystemExit):
        FH.main(['--a', a, '--b', b, '--games', '10', '--build', 'b' * 24, '--runtime', 'c' * 24, '--state-dir', st_dir],
                client=cl)


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
