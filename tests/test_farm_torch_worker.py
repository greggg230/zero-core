"""The farm's torch worker (zero.farm_torch_worker, docs/az.md 80) on the toy and a small CPU network: aw/1's hello before
torch, a job's records replayed by the trainer's path equal to the trainer's local self-play rows (fp32, deterministic
batches), the model refusal, the self-test, a rolling job's checkpoint / yield / resume / extend / finish, quit, a failed
resume, and the torch-weights/1 file."""
import importlib.util
import json
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import torch

from zero import farm_torch_worker as W, games as G, torch_weights as TW
from zero.net import make_net, save
from zero.run import DEFAULTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 4242


def gate_tool():
    spec = importlib.util.spec_from_file_location('torch_worker_gate', os.path.join(ROOT, 'tools', 'torch_worker_gate.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def settings(**kw):
    s = dict(DEFAULTS)
    s.update(game='toy', spec='', players=[2], conc=6, threads=1, groups=1, sims=12, sims_fast=4, shard_rows=40)
    s.update(kw)
    return json.loads(json.dumps(s, default=str))


def net(path, seed=0, it=3, cfg=None):
    torch.manual_seed(seed)
    m = make_net(G.info('toy'), 'toy', 'mlp', 16, 1)
    save(m, path, {'iter': it})
    return TW.export(path, path.replace('.pt', '-torch.pt'), cfg or {'infer_precision': 'fp32'}, gen=it)


class Cap:
    """The worker's events, kept; `wait(pred)` until one matches."""

    def __init__(self):
        self.ev = []
        self.cv = threading.Condition()

    def event(self, ev, **kw):
        with self.cv:
            self.ev.append(json.loads(json.dumps(dict(ev=ev, **kw))))
            self.cv.notify_all()

    def log(self, msg):
        pass

    def wait(self, pred, timeout=120):
        t0 = time.time()
        with self.cv:
            while True:
                hit = [e for e in self.ev if pred(e)]
                if hit:
                    return hit[0]
                left = timeout - (time.time() - t0)
                if left <= 0:
                    raise AssertionError(f'no such event within {timeout} s; events: {[e["ev"] for e in self.ev]}')
                self.cv.wait(left)

    def of(self, ev):
        with self.cv:
            return [e for e in self.ev if e['ev'] == ev]


def serve(tmp_path, monkeypatch, job):
    """The worker on a thread in `tmp_path` (its cwd), the job sent -> (events, the control queue, the thread, rc box)."""
    monkeypatch.chdir(tmp_path)
    out, q, box = Cap(), queue.Queue(), {}
    q.put(job)
    th = threading.Thread(target=lambda: box.update(rc=W.serve(str(tmp_path), 'zero', out=out, stdin_q=q)), daemon=True)
    th.start()
    return out, q, th, box


def job_line(model, **kw):
    j = {'op': 'job', 'lease_id': 'L', 'kind': 'selfplay', 'settings': settings(), 'games': 8, 'gid_base': 20,
         'seed': SEED, 'model': model, 'reference': None, 'threads': 1, 'out_dir': 'out', 'device': 'cpu'}
    j.update(kw)
    return j


def test_hello_comes_before_torch_is_imported(tmp_path):
    code = ('import sys; sys.path.insert(0, sys.argv[1]); from zero import farm_torch_worker as W; '
            'h = W.hello(W.build_id(sys.argv[2]), "zero"); print("torch" in sys.modules, "numpy" in sys.modules); '
            'print(__import__("json").dumps(h))')
    p = subprocess.run([sys.executable, '-c', code, os.path.join(ROOT, 'python'), str(tmp_path)], capture_output=True,
                       text=True, check=True)
    flags, h = p.stdout.splitlines()
    assert flags == 'False False'
    h = json.loads(h)
    assert h['aw'] == 1 and h['kinds'] == ['selfplay'] and h['zero_onnx'] == [] and h['models'] == ['torch-weights/1']
    assert 'rolling' in h['features'] and h['build_id'] == 'unsigned-dev'
    (tmp_path / 'manifest.json').write_bytes(b'{"x": 1}')
    import hashlib
    assert W.build_id(str(tmp_path)) == hashlib.sha256(b'{"x": 1}').hexdigest()[:24]


def test_quit_before_a_job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out, q = Cap(), queue.Queue()
    for m in ({'op': 'ping'}, {'op': 'yield'}, {'op': 'quit'}):
        q.put(m)
    assert W.serve(str(tmp_path), 'zero', out=out, stdin_q=q) == 0
    assert [e['ev'] for e in out.ev] == ['hello', 'status', 'yielded', 'end'] and out.ev[-1]['reason'] == 'quit'


def test_a_jobs_records_replay_to_the_trainers_rows(tmp_path, monkeypatch):
    model = net(str(tmp_path / 'iter_0003.pt'))
    out, q, th, box = serve(tmp_path, monkeypatch, job_line(model, gen=3, groups=2, threads=2))
    th.join(300)
    assert box.get('rc') == 0, out.ev[-3:]
    end = out.ev[-1]
    assert end['ev'] == 'end' and end['reason'] == 'done' and end['unfinished'] == {'gids': []}
    assert end['stats']['provider'] == 'torch-cpu' and end['stats']['games'] == 8
    recs = out.of('records')
    assert recs and sum(r['games'] for r in recs) == 8 and all(r['gen'] == 3 for r in recs)
    assert sorted(g for r in recs for a, b in r['gids'] for g in range(a, b)) == list(range(20, 28))
    assert sum(r['n'] for r in out.of('fin')) == 8
    T = gate_tool()
    cfg = W.job_settings(settings(), 2, None, 2)
    got, probs, heads = T.replay_files(str(tmp_path / 'out'), 'toy', False, SEED)
    assert not probs and {g for g, _ in got} == set(range(20, 28))
    assert all(h['build_id'] == 'unsigned-dev' and h['lease'] == 'L' and h['records'] == 1 for h in heads)
    want = T.trainer_rows(cfg, str(tmp_path / 'iter_0003.pt'), 3, 8, SEED, 20, 'cpu', str(tmp_path / 'trainer'))
    for (g, _), s in got.items():
        ok, why = T.same(s, want[g][1])
        assert ok, f'game {g}: {why}'


def test_an_onnx_model_is_refused(tmp_path, monkeypatch):
    (tmp_path / 'model.onnx').write_bytes(b'\x08\x07\x12\x07pytorch')
    out, q, th, box = serve(tmp_path, monkeypatch, job_line(str(tmp_path / 'model.onnx'), gen=3))
    th.join(60)
    assert box['rc'] == 1
    e = out.of('error')[-1]
    assert e['fatal'] and 'torch-weights/1' in e['message']


def test_a_wrong_sha256_is_refused(tmp_path, monkeypatch):
    model = net(str(tmp_path / 'iter_0003.pt'))
    out, q, th, box = serve(tmp_path, monkeypatch, job_line(model, sha256='0' * 64))
    th.join(60)
    assert box['rc'] == 1 and 'sha256' in out.of('error')[-1]['message']


def reference(path, ckpt, n_batches=4):
    """A reference set as zero.onnx_export writes one: recorded sparse batches of a toy driver and torch's fp32 answers."""
    from zero.infer import NetEvaluator, drive
    from zero.net import load
    m, _ = load(ckpt, 'cpu', game='toy')
    ev = NetEvaluator(m, 'cpu')
    keep = []

    class Rec:
        def __call__(self, lb):
            r = ev(lb)
            if len(keep) < n_batches:
                keep.append(({k: np.array(lb.a[k]) for k in ('xs_off', 'xs_idx', 'xs_val', 'n_players', 'extra', 'opt_off',
                                                              'opt_action')}, [np.array(x) for x in r]))
            return r
    seat = G.nn_seat('m', 8)
    drv = G.Driver('toy', [[seat, seat]], spec='', concurrency=4, seed=1, threads=1, groups=1, sparse=True)
    drive(drv, {'m': Rec()})
    cat = lambda xs: np.concatenate(xs)  # noqa: E731
    xo, oo, bo = [0], [0], [0]
    for b, _ in keep:
        xo += list(xo[-1] + b['xs_off'][1:])
        oo += list(oo[-1] + b['opt_off'][1:])
        bo.append(bo[-1] + len(b['n_players']))
    meta = {'zero_onnx': 1, 'game': 'toy', 'features': int(G.info('toy')['num_features']), 'rows': bo[-1]}
    np.savez(path, meta=json.dumps(meta), xs_off=np.array(xo, np.int64), xs_idx=cat([b['xs_idx'] for b, _ in keep]).astype(np.int32),
             xs_val=cat([b['xs_val'] for b, _ in keep]).astype(np.float32),
             n_players=cat([b['n_players'] for b, _ in keep]).astype(np.int64),
             extra=cat([b['extra'] for b, _ in keep]).astype(np.uint8), opt_off=np.array(oo, np.int64),
             opt_action=cat([b['opt_action'] for b, _ in keep]).astype(np.int32), batch_off=np.array(bo, np.int64),
             priors=cat([r[0] for _, r in keep]), value=cat([r[1] for _, r in keep]), margin=cat([r[2] for _, r in keep]),
             aux=np.zeros((0, 0), np.float32))
    return path


def test_the_self_test_passes_and_a_wrong_reference_fails(tmp_path, monkeypatch):
    model = net(str(tmp_path / 'iter_0003.pt'))
    ref = reference(str(tmp_path / 'reference.npz'), str(tmp_path / 'iter_0003.pt'))
    out, q, th, box = serve(tmp_path, monkeypatch, job_line(model, reference=ref, games=2))
    th.join(120)
    assert box['rc'] == 0
    st = out.of('selftest')
    assert len(st) == 1 and st[0]['ok'] and st[0]['deterministic'] and st[0]['provider'] == 'torch-cpu'
    assert st[0]['max_abs'] < 1e-5 and st[0]['rows'] > 0
    with np.load(ref) as z:
        d = {k: z[k] for k in z.files}
    d['priors'] = d['priors'] + np.float32(0.01)
    np.savez(str(tmp_path / 'bad.npz'), **d)
    out, q, th, box = serve(tmp_path, monkeypatch, job_line(model, reference=str(tmp_path / 'bad.npz'), games=2, out_dir='o2'))
    th.join(120)
    assert box['rc'] == 1 and not out.of('selftest')[0]['ok'] and 'self-test failed' in out.of('error')[-1]['message']


def test_rolling_checkpoint_yield_resume_extend_finish(tmp_path, monkeypatch):
    m3 = net(str(tmp_path / 'iter_0003.pt'), seed=0, it=3)
    m4 = net(str(tmp_path / 'iter_0004.pt'), seed=1, it=4)
    ref4 = reference(str(tmp_path / 'ref4.npz'), str(tmp_path / 'iter_0004.pt'))
    out, q, th, box = serve(tmp_path, monkeypatch, job_line(m3, games=6, rolling=True, flush_s=0.5, status_s=0.5,
                                                             settings=settings(conc=3, shard_rows=100000)))
    out.wait(lambda e: e['ev'] == 'fin')
    q.put({'op': 'checkpoint', 'gen': 4, 'model': m4, 'reference': ref4})
    out.wait(lambda e: e['ev'] == 'selftest' and e.get('gen') == 4)
    q.put({'op': 'yield'})
    out.wait(lambda e: e['ev'] == 'yielded')
    n = len(out.of('fin'))
    time.sleep(0.5)
    assert len(out.of('fin')) == n, 'games finished while yielded'
    q.put({'op': 'resume'})
    out.wait(lambda e: e['ev'] == 'resumed')
    out.wait(lambda e: e['ev'] == 'low')
    q.put({'op': 'extend', 'games': 10})
    out.wait(lambda e: e['ev'] == 'low' and e['games'] == 10)
    out.wait(lambda e: e['ev'] == 'records' and any(b > 26 for _, b in e['gids']), timeout=180)
    q.put({'op': 'finish'})
    th.join(300)
    assert box['rc'] == 0
    end = out.ev[-1]
    assert end['reason'] == 'finish' and end['unfinished'] == {'gids': []}, end
    assert end['stats']['yields'] == 1
    T = gate_tool()
    got, probs, heads = T.replay_files(str(tmp_path / 'out'), 'toy', False, SEED)
    assert not probs and {g for g, _ in got} == set(range(20, 30))
    nets = set()
    for s in got.values():
        if s is not None:
            nets |= set(int(x) for x in s['net'])
    assert nets == {3, 4}, nets
    assert min(h['gen'] for h in heads) == 3 and max(h['gen'] for h in heads) >= 3


def test_quit_drops_the_games_in_flight(tmp_path, monkeypatch):
    m3 = net(str(tmp_path / 'iter_0003.pt'))
    out, q, th, box = serve(tmp_path, monkeypatch, job_line(m3, games=30, rolling=True, settings=settings(conc=4)))
    out.wait(lambda e: e['ev'] == 'fin')
    q.put({'op': 'quit'})
    th.join(120)
    assert box['rc'] == 0
    end = out.ev[-1]
    done = sum(r['games'] for r in out.of('records'))
    assert end['reason'] == 'quit' and done == end['stats']['games']
    left = sum(b - a for a, b in end['unfinished']['gids'])
    assert left == 30 - done and left > 0


def test_a_failed_resume_is_fatal(tmp_path, monkeypatch):
    m3 = net(str(tmp_path / 'iter_0003.pt'))
    calls = {'n': 0}
    real = W.Job._evaluator

    def flaky(self, d, inf):
        calls['n'] += 1
        if calls['n'] > 1:
            raise RuntimeError('CUBLAS_STATUS_ALLOC_FAILED (test)')
        return real(self, d, inf)
    monkeypatch.setattr(W.Job, '_evaluator', flaky)
    out, q, th, box = serve(tmp_path, monkeypatch, job_line(m3, games=40, rolling=True, settings=settings(conc=4)))
    out.wait(lambda e: e['ev'] == 'fin')
    q.put({'op': 'yield'})
    out.wait(lambda e: e['ev'] == 'yielded')
    q.put({'op': 'resume'})
    th.join(120)
    assert box['rc'] == 1 and not out.of('resumed')
    e = out.of('error')[-1]
    assert e['fatal'] and 'resume' in e['message'] and 'CUBLAS' in e['message']


def test_torch_weights_round_trip(tmp_path):
    p = str(tmp_path / 'iter_0005.pt')
    w = net(p, it=5, cfg={'infer_precision': 'bf16', 'infer_token_chunks': 2, 'infer_workers': 1, 'torch_threads': 2})
    d = torch.load(w, weights_only=True)
    assert d['format'] == 'torch-weights/1' and d['gen'] == 5 and set(d) >= {'net_config', 'state_dict', 'infer'}
    assert d['infer'] == {'precision': 'bf16', 'token_chunks': 2, 'workers': 1, 'torch_threads': 2}
    ck = torch.load(p, weights_only=False)
    assert 'optimizer' not in d and all(torch.equal(ck['state'][k], d['state_dict'][k]) for k in ck['state'])
    m, inf = TW.load(w, 'cpu', game='toy')
    from zero.net import load
    m0, _ = load(p, 'cpu', game='toy')
    x = torch.rand(3, m.features)
    n = torch.full((3,), 2, dtype=torch.long)
    for a, b in zip(m(x, n), m0(x, n)):
        assert torch.equal(a, b)
    (tmp_path / 'g.onnx').write_bytes(b'\x08\x07\x12\x07pytorch')
    with pytest.raises(TW.NotTorchWeights):
        TW.read(str(tmp_path / 'g.onnx'))
    with pytest.raises(TW.NotTorchWeights):
        TW.read(p)                        # (a checkpoint is not the variant: weights_only refuses it or its format)
