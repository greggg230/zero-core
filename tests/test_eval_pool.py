"""A run's evaluation on the farm (eval_remote 'pool:<farm run>', zero/eval_pool.py, docs/az.md 65) against a fake
coordinator that plays each eval batch as the native worker would (tests/farm_play.py: a traced rolling driver on the
uploaded ONNX graphs) and hands it out as two machines' leases (a duplicate chunk too):

* the evaluation records what RemoteEval records for a remote (evals.jsonl row with eval_yard / elo / elo_anchored,
  the Elo pool's files, models/index.json, best.pt, the game records with their players), and plays the same matches
  as a remote evaluation of the same run (the pool's pairs and game counts equal; the anchors fixed and calibrated as
  there) - every match one batch: its gens registered, a signed definition whose eval template is the batch's job, the
  batch;
* a restart resumes the batch it was collecting (no second batch); a batch the farm half-delivers is dropped at its
  deadline and its missing games played here by the worker (the stub's `eval`) - the same games;
* a row its record does not support (seats, digest) or a spot replay that refuses a game refuses the batch: logged,
  and its games played here; with no deadline the evaluation fails (retried with a new batch);
* refusals at the start: the self-play pool's farm run, no pool_worker, a deadline without the runtime pack's ONNX
  Runtime next to the worker; the farm report counts only the fallback's games as this machine's."""
import hashlib
import json
import os
import re
import tempfile
import threading
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip('onnx')
pytest.importorskip('onnxscript')
pytest.importorskip('onnxruntime')

import torch  # noqa: E402

from zero import elo, eval_pool as EP, farm_def as FD, farm_eval as FE, remote  # noqa: E402
from zero.net import save  # noqa: E402
from zero.run import DEFAULTS, PRESETS, Run, model_path  # noqa: E402
import farm_play  # noqa: E402
import farm_release_stub as STUB  # noqa: E402

TOKEN = 'secret-admin-token-xyz'
RT = 'c' * 24
MANIFEST = json.dumps({'manifest': 1, 'kind': 'worker', 'version': 'toy-test', 'game': 'zero-toy', 'requires_runtime': RT,
                       'entry': 'w.exe', 'files': []}).encode()
BID = hashlib.sha256(MANIFEST).hexdigest()[:24]
MANIFESTS = {BID: MANIFEST}
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worker_stub.py')
FARM = 'tr-eval'


class FakeFarm:
    """The coordinator's desktop API in memory for one eval-only farm run: one batch at a time (a new batch reserves
    the run's next game ids, a paired one from an even id), played when results are listed. `mode`: 'deliver' (two
    machines' leases + a duplicate chunk), 'partial' (the first lease only), 'none', 'seat' (a row's seats swapped),
    'move' (a record's move changed, its digest recomputed)."""

    def __init__(self, mode='deliver'):
        self.mode = mode
        self.runs, self.objects, self.ckpts, self.results, self.calls, self.puts = {}, {}, {}, [], [], []
        self.batches, self.cur, self.next_gid, self.played = {}, {}, {}, set()
        self.defs = os.environ.get('FARM_STUB_STATE')
        self.acked = 0

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
            assert body['game'] == 'zero-toy' and body['state'] == 'active'
            ev = body['eval']
            if ev:
                assert all((run, g) in self.ckpts for g in ev['checkpoints']), 'gens before the batch'
                doc = self.definition(run)['doc']
                assert doc['jobs']['eval'] == {k: v for k, v in ev['job'].items() if k != 'seed'}, 'the template'
                assert sorted(int(g) for g in doc['checkpoints']) == sorted(ev['checkpoints'])
                assert sorted({s['net'] for s in ev['job']['seats'] if 'net' in s}) == sorted(ev['checkpoints'])
                assert ev['job']['settings']['eval_trace'] == 1 and BID in doc['builds']
                FD.check_no_paths(ev['job']['settings'], 'settings')
                if (run, ev['batch']) not in self.batches:
                    lo = self.next_gid.get(run, 0)
                    lo += lo % 2 if ev['job']['paired'] else 0
                    self.batches[(run, ev['batch'])] = {'lo': lo, 'hi': lo + ev['games'], 'ev': ev}
                    self.next_gid[run] = lo + ev['games']
                self.cur[run] = ev['batch']
            else:
                self.cur.pop(run, None)
            self.runs[run] = body
            self.puts.append(body)
            return ok({'ok': True, 'created': True})
        if m and method == 'GET':
            run = m.group(1)
            if run not in self.runs:
                return 404, json.dumps({'error': 'not_found', 'message': f'no run {run}'}).encode()
            b = self.batches.get((run, self.cur.get(run)))
            es = b and {'batch': self.cur[run], 'games': b['ev']['games'], 'lo': b['lo'], 'hi': b['hi']}
            df = self.definition(run)
            return ok({'run': run, 'eval_state': es, 'definition': df and {k: df[k] for k in ('version', 'sha256', 'key')}})
        m = re.match(r'^/v1/admin/artifact/(.+)$', p)
        if m and method == 'PUT':
            assert hashlib.sha256(data).hexdigest() == headers['x-sha256']
            self.objects[m.group(1)] = data
            return ok({'ok': True, 'key': m.group(1), 'size': len(data)})
        if m and method == 'GET':
            mm = re.match(r'^builds/([0-9a-f]{24})/manifest\.json$', m.group(1))
            if mm and mm.group(1) in MANIFESTS:
                return 200, MANIFESTS[mm.group(1)]
            return 200, self.objects[m.group(1)]
        m = re.match(r'^/v1/admin/run/([a-z0-9-]+)/checkpoint/(\d+)$', p)
        if m and method == 'PUT':
            run, gen = m.group(1), int(m.group(2))
            assert run in self.runs and (run, gen) not in self.ckpts, 'a gen registered once'
            assert body['model'] == f'ckpt/{run}/{gen}/model.onnx' and body['reference'] == f'ckpt/{run}/{gen}/reference.npz'
            assert hashlib.sha256(self.objects[body['model']]).hexdigest() == body['sha256']
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
        return 404, json.dumps({'error': 'not_found', 'message': p}).encode()

    def chunk(self, run, name, lease, seq, machine, rows, ids, dup=()):
        body = ''.join(FE.row_json(r) + '\n' for r in rows).encode()
        key = f'results/{run}/{name}/{lease}/{seq}'
        self.objects[key] = body
        self.results.append({'id': len(self.results) + 1, 'key': key, 'lease_id': lease, 'seq': seq, 'machine_id': machine,
                             'kind': 'eval', 'batch': name, 'games': len(ids), 'ids': [[g, g + 1] for g in ids],
                             'also': [], 'dup_ids': [[g, g + 1] for g in dup], 'sha256': hashlib.sha256(body).hexdigest(),
                             'size': len(body), 't': 0})

    def play(self, run):
        name = self.cur.get(run)
        if name is None or name in self.played or self.mode == 'none':
            return
        self.played.add(name)
        b = self.batches[(run, name)]
        ev, lo, hi = b['ev'], b['lo'], b['hi']
        with tempfile.TemporaryDirectory() as d:
            models = {}
            for g in ev['checkpoints']:
                models[g] = os.path.join(d, f'{g}.onnx')
                with open(models[g], 'wb') as f:
                    f.write(self.objects[self.ckpts[(run, g)]['model']])
            rows = farm_play.play_traced(ev['job']['settings'], ev['job']['seats'], lo, hi, ev['job']['seed'],
                                         ev['job']['paired'], models)
        if self.mode == 'seat':
            rows[1] = dict(rows[1], seats=list(reversed(rows[1]['seats'])))
        if self.mode == 'move':
            r = json.loads(json.dumps(rows[0]))
            r['rec']['steps'][0] = int(r['rec']['steps'][0]) + 1000
            r['digest'] = FE.trace_digest(r['rec'])
            rows[0] = r
        mid = lo + (hi - lo) // 2
        mid -= (mid - lo) % 2
        self.chunk(run, name, 'l0', 0, 'm_a', rows[:mid - lo], list(range(lo, mid)))
        if self.mode != 'partial':
            self.chunk(run, name, 'l1', 0, 'm_b', rows[mid - lo:], list(range(mid, hi)))
            # (a late copy of the first pair: credited before, skipped)
            self.chunk(run, name, 'l2', 0, 'm_b', rows[:2], [lo, lo + 1], dup=[lo, lo + 1])


def _models(R, run, iters, anchors):
    def perturbed(k):
        m = R.new_model()
        torch.manual_seed(100 + k)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.3 * torch.randn_like(p))
        return m
    for k in iters:
        save(perturbed(k), model_path(run, k), {'iter': k})
    paths = []
    for i in anchors:
        p = os.path.join(os.path.dirname(run), 'anchors', f'a{i}.pt')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        save(perturbed(50 + i), p, {'iter': 0})
        paths.append(p)
    return paths


def _cfg(run, anchors, **over):
    cfg = dict(DEFAULTS, **PRESETS['toy-smoke'])
    cfg.update(game='toy', yardstick='heuristic', eval_games=4, eval_random_games=0, elo_games=2, elo_paired=1,
               elo_together=1, ref_games=2, eval_sims=4, conc=8, threads=1, game_records=1, eval_every=1,
               elo_anchors=','.join(f'a{i}={p}' for i, p in enumerate(anchors)), elo_anchor_games=4,
               elo_anchor_calib_games=4, eval_select='cq', pool_build=BID, pool_runtime=RT, pool_worker=WORKER,
               eval_pool_deadline_s=0, eval_pool_spot=1.0, eval_pool_poll_s=20)
    cfg.update(over)
    return cfg


def _setup(tmp_path, monkeypatch, name='r', mode='deliver', **over):
    monkeypatch.setenv('ZEROFARM_ADMIN_TOKEN', TOKEN)
    if not os.environ.get('FARM_STUB_STATE'):
        STUB.setup(monkeypatch, tmp_path, TOKEN)
    monkeypatch.setenv('WORKER_STUB_BUILD', BID)
    run = str(tmp_path / 'runs' / name)
    os.makedirs(os.path.join(run, 'models'))
    anchors_dir = tmp_path / 'runs' / 'anchors'
    cfg0 = _cfg(run, [])
    R0 = Run(run, cfg0)
    anchors = _models(R0, run, (0, 1, 2), (0, 1)) if not anchors_dir.exists() else \
        [str(anchors_dir / 'a0.pt'), str(anchors_dir / 'a1.pt')]
    if anchors_dir.exists() and not os.path.exists(model_path(run, 0)):
        _models(R0, run, (0, 1, 2), ())
    cfg = _cfg(run, anchors, eval_remote=f'pool:{FARM}', **over)
    with open(os.path.join(run, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f)
    return run, cfg, FakeFarm(mode)


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _farm_eval(R, fake, clock=None, cl=None):
    clock = clock or Clock()
    cl = cl or FD.Client('http://fake', transport=fake)
    fm = EP.FarmMatches(R.cfg, R.run, FARM, R.log, client=cl, clock=clock, sleep=clock.sleep)
    return EP.FarmEval(R, f'pool:{FARM}', matches=fm)


def _records(run):
    from zero import gamerec as GR
    out = []
    for kind, sub, path in GR.match_dirs(os.path.join(run, 'records')):
        for f in sorted(os.listdir(path)):
            specs, recs = GR.read_file(os.path.join(path, f))
            out += [(kind, sub, r) for r in recs]
    return out


def test_a_farm_evaluation_records_what_a_remote_records(tmp_path, monkeypatch):
    run, cfg, fake = _setup(tmp_path, monkeypatch)
    R = Run(run, cfg)
    ev = _farm_eval(R, fake)
    for it in (1, 2):
        ev._evaluate(it)
    rows = [json.loads(x) for x in open(os.path.join(run, 'evals.jsonl'), encoding='utf-8')]
    assert [r['iter'] for r in rows] == [1, 2] and all(r['host'] == f'pool:{FARM}' for r in rows)
    tag = elo.eval_tag(cfg)
    for r in rows:
        assert r['eval_yard']['games'] == 4 and r['eval_yard']['opp'] == 'heuristic' and r['eval_yard']['select'] == tag
        assert r['elo']['select'] == tag and r['elo_anchored']['select'] == tag and r['records'] >= 1
        p = r['eval_pool']
        assert p['farm_run'] == FARM and p['local'] == 0 and p['farm'] == p['games'] and p['machines'] == 2
        assert p['spot'] == p['farm']                       # (eval_pool_spot 1: every farm game replayed)
    # iteration 1: the yardstick, its Elo match, a1's calibration, both anchors; iteration 2: the yardstick, its two
    # Elo opponents, both anchors - one batch each, gens registered once
    assert rows[0]['eval_pool']['games'] == 4 + 2 + 4 + 2 * 4 and len(rows[0]['eval_pool']['batches']) == 5
    assert rows[1]['eval_pool']['games'] == 4 + 2 * 2 + 2 * 4 and len(rows[1]['eval_pool']['batches']) == 5
    assert sorted(g for _, g in fake.ckpts) == [0, 1, 2, EP.ANCHOR_GEN0, EP.ANCHOR_GEN0 + 1]
    # every batch put once with its own definition version; the dup chunk skipped; everything acked
    batch_puts = [b['eval']['batch'] for b in fake.puts if b['eval']]
    assert len(batch_puts) == len(set(batch_puts)) == 10
    assert len(STUB.calls(os.environ['FARM_STUB_LOG'])) == 10 and fake.acked == len(fake.results)
    assert TOKEN not in json.dumps(fake.puts) and TOKEN not in open(os.path.join(run, 'log.txt'), encoding='utf-8').read()
    # the Elo pool: the files RemoteEval copies back; the anchors fixed (a1 calibrated against a0)
    pool = elo.pool_dir(run, tag)
    res = json.load(open(os.path.join(pool, 'results.json'), encoding='utf-8'))
    assert set(res['anchors']) == {'anchor:a0', 'anchor:a1'} and res['anchors']['anchor:a0']['elo'] == 1000.0
    assert 'calibrated' in res['anchors']['anchor:a1']['how']
    for fn in ('ratings.json', 'anchored.json'):
        assert os.path.exists(os.path.join(pool, fn))
    assert os.path.exists(os.path.join(run, 'models', 'index.json'))
    # the game records: the farm's games with the players a local match names; the reference games here
    recs = _records(run)
    kinds = {}
    for kind, sub, r in recs:
        kinds[kind] = kinds.get(kind, 0) + 1
        if kind != 'ref':                                   # (the reference games: bots, played here)
            assert r['src'] == 'farm' and r['machine'] and r['meta']['farm_run'] == FARM, r
    assert kinds == {'eval': 8, 'elo': 2 + 4, 'ref': 2, 'anchor': 4 + 2 * 4 + 2 * 4}, kinds
    labels = {(k, tuple(p['label'] for p in r['players'])) for k, _, r in recs}
    assert ('anchor', ('iter_0001', 'anchor:a0')) in labels and ('anchor', ('anchor:a0', 'iter_0001')) in labels
    assert ('anchor', ('anchor:a1', 'anchor:a0')) in labels and ('eval', ('heuristic', 'iter_0002')) in labels
    # the same matches as a remote evaluation of the same run (another directory, the 'local' remote)
    run_b = str(tmp_path / 'runs' / 'rb')
    os.makedirs(os.path.join(run_b, 'models'))
    for k in (0, 1, 2):
        with open(model_path(run, k), 'rb') as f, open(model_path(run_b, k), 'wb') as g:
            g.write(f.read())
    cfg_b = dict(cfg, eval_remote='local')
    Rb = Run(run_b, cfg_b)
    monkeypatch.setenv('ZERO_REMOTE_WORKDIR', str(tmp_path / 'remote'))
    rv = remote.RemoteEval.__new__(remote.RemoteEval)
    rv.R, rv.name, rv.spec = Rb, 'local', remote.remote_spec('local')
    rv.t = remote.Transport(rv.spec)
    rv.rdir = rv.spec['workdir'].rstrip('/') + '/rb'
    rv.module, rv.shipped, rv.anchors_shipped, rv.lock, rv.queued = 'zero.remote', None, set(), threading.Lock(), set()
    rv.inputs_shipped = {}

    def run_module(module, args, timeout):
        m = remote.evaluate_mirror(args[1], int(args[2]))
        return SimpleNamespace(returncode=0, stdout=remote.RESULT + json.dumps(dict(m, caps=['tails'])) + '\n', stderr='')
    rv.t.run_module = run_module
    for it in (1, 2):
        rv._evaluate(it)
    res_b = json.load(open(os.path.join(elo.pool_dir(run_b, tag), 'results.json'), encoding='utf-8'))
    assert set(res['anchors']) == set(res_b['anchors']) and sorted(res) == sorted(res_b)
    assert {k: n for k, (_, n) in res['pairs'].items()} == {k: n for k, (_, n) in res_b['pairs'].items()}
    assert res['logged'] == res_b['logged']
    rows_b = [json.loads(x) for x in open(os.path.join(run_b, 'evals.jsonl'), encoding='utf-8')]
    assert [sorted(k for k in r if k != 'eval_pool') for r in rows] == [sorted(r) for r in rows_b]
    kinds_b = {}
    for kind, _, r in _records(run_b):
        kinds_b[kind] = kinds_b.get(kind, 0) + 1
    assert kinds_b == kinds
    # a second evaluation of an iteration plays nothing new (the pool has its matches); a new FarmEval (a restart)
    # with a batch in flight resumes it
    n = len(fake.puts)
    ev._evaluate(2)
    assert len(fake.puts) == n


def test_a_restart_resumes_the_batch_and_the_deadline_plays_the_rest_here(tmp_path, monkeypatch):
    run, cfg, fake = _setup(tmp_path, monkeypatch, mode='partial', eval_pool_deadline_s=60, elo_anchors='',
                            elo_games=0, eval_games=8)
    import zero.eval_pool as ep
    monkeypatch.setattr(ep, 'ort_library', lambda worker, runtime: ('ort.dll', []))
    R = Run(run, cfg)
    clock = Clock()
    ev = _farm_eval(R, fake, clock)
    # the batch submitted, then the process "dies" before collecting: the next one resumes it
    fm = ev.matches
    bt, _ = None, None
    real_collect = fm._collect

    def dies(bt, settings):
        raise KeyboardInterrupt('the run stopped')
    fm._collect = dies
    with pytest.raises(KeyboardInterrupt):
        ev._evaluate(1)
    fm._collect = real_collect
    ev2 = _farm_eval(R, fake, clock)
    ev2._evaluate(1)
    batch_puts = [b['eval']['batch'] for b in fake.puts if b['eval']]
    assert len(batch_puts) == 1, batch_puts                 # (resumed: the same batch, not a second one)
    assert fake.puts[-1]['eval'] is None                    # (dropped at the deadline)
    row = [json.loads(x) for x in open(os.path.join(run, 'evals.jsonl'), encoding='utf-8')][-1]
    p = row['eval_pool']
    assert p['games'] == 8 and p['local'] == 4 and p['farm'] == 4 and p['machines'] == 1 and p['spot'] == 4
    assert row['eval_yard']['games'] == 8
    # the fallback's games are the farm's games: the same seeds, seats and results as the fake farm plays them
    st = json.load(open(os.path.join(run, 'eval_pool', FARM, 'state.json'), encoding='utf-8'))
    (b,) = st['batches'].values()
    assert b['local'] == [[b['lo'] + 4, b['hi']]] and b['status'] == 'done'
    got = {r['gid']: r for r in ev2.matches.rows(b)}
    models = {g: ev2.matches.gen_files(g)['model'] for g in ev2.matches.gens(b)}
    settings = json.load(open(b['definition']['path'], encoding='utf-8'))['jobs']['eval']['settings']
    want = farm_play.play_traced(settings, b['seats'], b['lo'], b['hi'], b['seed'], b['paired'], models)
    assert [got[w['gid']]['digest'] for w in want] == [w['digest'] for w in want]
    recs = _records(run)
    assert sorted(r['src'] for k, _, r in recs if k == 'eval') == ['farm'] * 4 + ['local'] * 4


@pytest.mark.parametrize('mode', ['seat', 'move'])
def test_a_refused_batch_is_played_here(tmp_path, monkeypatch, mode):
    run, cfg, fake = _setup(tmp_path, monkeypatch, mode=mode, eval_pool_deadline_s=60, elo_anchors='', elo_games=0)
    import zero.eval_pool as ep
    monkeypatch.setattr(ep, 'ort_library', lambda worker, runtime: ('ort.dll', []))
    R = Run(run, cfg)
    ev = _farm_eval(R, fake)
    ev._evaluate(1)
    row = [json.loads(x) for x in open(os.path.join(run, 'evals.jsonl'), encoding='utf-8')][-1]
    assert row['eval_pool']['local'] == 4 and row['eval_pool']['farm'] == 0
    ref = [json.loads(x) for x in open(os.path.join(run, 'eval_pool', FARM, 'refused.jsonl'), encoding='utf-8')]
    assert len(ref) == 1 and ('seats' in ref[0]['why'] if mode == 'seat' else 'spot check' in ref[0]['why'])
    assert fake.puts[-1]['eval'] is None


def test_without_a_deadline_a_refused_batch_fails_the_evaluation(tmp_path, monkeypatch):
    run, cfg, fake = _setup(tmp_path, monkeypatch, mode='seat', elo_anchors='', elo_games=0)
    R = Run(run, cfg)
    ev = _farm_eval(R, fake)
    with pytest.raises(EP.EvalPoolError, match='refused'):
        ev._evaluate(1)
    assert not os.path.exists(os.path.join(run, 'evals.jsonl'))
    fake.mode = 'deliver'
    ev._evaluate(1)                                         # (the retry: a new batch)
    puts = [b['eval']['batch'] for b in fake.puts if b['eval']]
    assert len(puts) == 2 and puts[0] != puts[1]
    row = [json.loads(x) for x in open(os.path.join(run, 'evals.jsonl'), encoding='utf-8')][-1]
    assert row['eval_pool']['farm'] == 4


def test_refusals_at_the_start(tmp_path, monkeypatch):
    run, cfg, fake = _setup(tmp_path, monkeypatch, elo_anchors='')
    R = Run(run, dict(cfg, selfplay_remote=f'mac,pool:{FARM}'))
    with pytest.raises(SystemExit, match='self-play pool'):
        EP.FarmEval(R, f'pool:{FARM}')
    with pytest.raises(SystemExit, match='pool_worker'):
        EP.FarmMatches(dict(cfg, pool_worker=''), run, FARM, R.log, client=object())
    with pytest.raises(SystemExit, match='ONNX Runtime'):
        EP.FarmMatches(dict(cfg, eval_pool_deadline_s=60), run, FARM, R.log, client=object())
    with pytest.raises(SystemExit, match='slug'):
        EP.FarmMatches(cfg, run, 'Bad_Name', R.log, client=object())
    # the runtime pack next to the worker build (the farm builds' layout): found
    builds = tmp_path / 'builds'
    (builds / BID).mkdir(parents=True)
    (builds / RT).mkdir()
    (builds / RT / 'onnxruntime.dll').write_bytes(b'x')
    exe = builds / BID / 'w.exe'
    exe.write_bytes(b'x')
    assert EP.ort_library(str(exe), RT)[0] == str(builds / RT / 'onnxruntime.dll')
    # the self-play pool's weight-swap templates and positions: copied once into the farm run's own export cache
    tpl = os.path.join(run, 'pool', 'tr-sp', 'export', 'f' * 16)
    os.makedirs(tpl)
    open(os.path.join(tpl, 'template.json'), 'w').write('{}')
    fm = EP.FarmMatches(dict(cfg, selfplay_remote='pool:tr-sp'), run, FARM, R.log, client=object())
    cache = os.path.join(fm.dir, 'export')
    fm._seed_export_cache(cache)
    assert os.path.exists(os.path.join(cache, 'f' * 16, 'template.json'))
    os.remove(os.path.join(tpl, 'template.json'))
    fm._seed_export_cache(cache)                            # (once: never again, never written back)
    assert os.path.exists(os.path.join(cache, 'f' * 16, 'template.json'))


def test_off_by_default_and_the_farm_report(tmp_path):
    assert DEFAULTS['eval_remote'] == '' and DEFAULTS['eval_pool_deadline_s'] == 1800.0
    assert not EP.is_pool('mac') and not EP.is_pool('') and EP.is_pool('pool:x')
    # the own-machines report: a farm evaluation's row counts only the games its deadline played here (this machine's);
    # the farm's games and the iteration's Elo pool games are the coordinator's to count; a remote's row as before
    from zero import farm_report as FR
    now = 1_800_000_000.0
    run = tmp_path / 'runs' / 'fa'
    os.makedirs(run / 'elo')
    with open(run / 'evals.jsonl', 'w', encoding='utf-8') as f:
        f.write(json.dumps({'iter': 16, 'time': now - 600, 'host': 'mac', 'seconds': 400.0,
                            'eval_yard': {'games': 0}}) + '\n')
        f.write(json.dumps({'iter': 32, 'time': now - 300, 'host': f'pool:{FARM}', 'seconds': 900.0,
                            'eval_yard': {'games': 0}, 'eval_pool': {'games': 200, 'farm': 194, 'local': 6,
                                                                     'local_s': 120.0}}) + '\n')
    with open(run / 'elo' / 'results.json', 'w', encoding='utf-8') as f:
        json.dump({'pairs': {'anchor:a|iter_0016': [50.0, 100], 'anchor:a|iter_0032': [50.0, 100]}}, f)
    os.utime(run / 'evals.jsonl', (now - 60, now - 60))
    cfg = dict(FR.DEFAULTS, fuzz=None)
    body, rep = FR.build_report(cfg, [str(tmp_path / 'runs')], now, fuzz=False)
    s = {x['id']: x for x in body['sources']}
    assert s['mac']['eval_games']['1h'] == 100                      # (iteration 16's anchor games, on the Mac)
    assert s[rep.local]['eval_games']['1h'] == 6                    # (the fallback's games only)
    assert not any(k.startswith('pool:') for k in s)
