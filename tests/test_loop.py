"""The loop end to end at toy-smoke scale: iterations, resume, STOP, PAUSE (exit 75 + resume),
the supervisor with a fake run, kill_run, fork_run, the dashboard on a real run dir."""
import json
import os
import subprocess
import sys
import time

import pytest

from zero import runctl
from zero.run import latest_iter, read_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# `python -m zero.run` from the repo root needs the package dir on the path (a game's runs get
# it from their package's __init__)
PYPATH = os.pathsep.join([os.path.join(ROOT, 'python')] + [p for p in [os.environ.get('PYTHONPATH')] if p])
SMOKE = ['--preset', 'toy-smoke', '--conc', '24', '--games', '24', '--eval-games', '8', '--eval-random-games', '4',
         '--elo-games', '4', '--ref-games', '8']


def run_loop(run, *extra, env=None, timeout=300):
    e = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH, **(env or {}))
    return subprocess.run([sys.executable, '-m', 'zero.run', '--run', str(run), *SMOKE, *extra], cwd=ROOT,
                          capture_output=True, text=True, timeout=timeout, env=e)


@pytest.fixture(scope='module')
def smoke_run(tmp_path_factory):
    run = tmp_path_factory.mktemp('runs') / 'smoke'
    p = run_loop(run, '--iters', '2', '--yardstick', 'ext:toyheur')
    assert p.returncode == 0, p.stdout + p.stderr
    return run


def test_iterations_metrics_and_elo(smoke_run):
    m = read_metrics(str(smoke_run))
    assert [r['iter'] for r in m] == [0, 1, 2]
    assert m[1]['selfplay']['games'] == 24 and m[1]['selfplay']['samples'] > 0
    assert m[2]['train']['steps'] >= 10 and m[2]['eval_yard']['opp'] == 'ext:toyheur'
    r = json.load(open(smoke_run / 'elo' / 'ratings.json'))
    assert r['anchor'] == 'ext:toyheur' and r['ratings']['ext:toyheur']['elo'] == 1000
    assert {'iter_0000', 'iter_0001', 'iter_0002', 'random'} <= set(r['ratings'])
    idx = json.load(open(smoke_run / 'models' / 'index.json'))
    assert idx['best'] and os.path.exists(smoke_run / 'models' / 'best.pt')
    assert runctl.read_state(str(smoke_run), 'run')['state'] == 'done'


def test_window_cache_trains_the_same_network(smoke_run):
    """Run setting window_cache: training from the cached window (the second training of the same
    iteration reuses every shard) gives the network of training without it, weight for weight."""
    import torch
    from zero import run as R
    with open(os.path.join(smoke_run, 'config.json')) as f:
        cfg = json.load(f)
    cfg = dict(cfg, train_device='cpu')

    def trained(run_obj):
        model, _, stats = run_obj.train_model(2, 200)
        return {k: v.detach().clone() for k, v in model.state_dict().items()}, stats()

    ref, st0 = trained(R.Run(str(smoke_run), dict(cfg, window_cache=0)))
    cached = R.Run(str(smoke_run), dict(cfg, window_cache=1))
    first, st1 = trained(cached)
    again, st2 = trained(cached)
    assert st0['window_reused'] == 0 and st1['window_reused'] == 0 and st2['window_reused'] == st2['window_samples'] > 0
    for got in (first, again):
        assert ref.keys() == got.keys()
        for k in ref:
            assert torch.equal(ref[k], got[k]), k
    # train_prefetch (batches built ahead on a thread), alone and with the cache: the same network
    for extra in ({'train_prefetch': 2}, {'train_prefetch': 3, 'window_cache': 1}, {'window_sparse': 1},
                  {'window_sparse': 1, 'window_cache': 1, 'train_prefetch': 2}):
        got, _ = trained(R.Run(str(smoke_run), dict(cfg, **{'window_cache': 0, **extra})))
        for k in ref:
            assert torch.equal(ref[k], got[k]), (extra, k)
    # window_sparse with the cache over budget (a subsampled window): the network of the dense window,
    # its second training loading no shard
    small = dict(cfg, window_cache=0, ram_budget_gb=1e-4)
    ref_s, st_s = trained(R.Run(str(smoke_run), small))
    assert st_s['window_samples'] < st_s['window_loaded']
    sp = R.Run(str(smoke_run), dict(small, window_sparse=1, window_cache=1))
    for n in range(2):
        got, st = trained(sp)
        assert st['window_samples'] == st_s['window_samples'] and st['window_reused'] == (st['window_samples'] if n else 0)
        for k in ref_s:
            assert torch.equal(ref_s[k], got[k]), (n, k)


def test_sparse_setting_plays_the_same_selfplay(smoke_run, tmp_path, monkeypatch):
    """Run setting sparse: the run's drivers send sparse network batches, and an iteration's self-play
    is the same - its shards array for array - as with dense batches."""
    import shutil
    import numpy as np
    from zero import games as G, run as R
    with open(os.path.join(smoke_run, 'config.json')) as f:
        cfg = json.load(f)
    seen = []
    orig = G.Driver

    def spy(*a, **kw):
        seen.append(kw.get('sparse', False))
        return orig(*a, **kw)
    monkeypatch.setattr(G, 'Driver', spy)
    shards = {}
    for sp in (0, 1):
        run = tmp_path / f'sparse{sp}'
        shutil.copytree(smoke_run, run)
        R.Run(str(run), dict(cfg, sparse=sp)).selfplay(2)
        d = run / 'data' / 'iter_0002'
        shards[sp] = {p: dict(np.load(d / p)) for p in sorted(os.listdir(d)) if p.endswith('.npz')}
    assert seen == [False, True]
    assert shards[0].keys() == shards[1].keys() and shards[0]
    for p, arrs in shards[0].items():
        assert arrs.keys() == shards[1][p].keys(), p
        for k in arrs:
            assert np.array_equal(arrs[k], shards[1][p][k]), (p, k)


def test_resume_continues_and_stop_file(smoke_run, tmp_path):
    import shutil
    run = tmp_path / 'r'
    shutil.copytree(smoke_run, run)
    p = run_loop(run, '--iters', '3')
    assert p.returncode == 0, p.stderr
    assert [r['iter'] for r in read_metrics(str(run))] == [0, 1, 2, 3]
    (run / 'STOP').write_text('')
    p = run_loop(run, '--iters', '5')
    assert p.returncode == 0 and 'STOP file found' in p.stdout
    assert latest_iter(str(run)) == 3


def test_pause_exits_75_and_resume_finishes(tmp_path):
    run = tmp_path / 'p'
    run.mkdir()
    proc = subprocess.Popen([sys.executable, '-m', 'zero.run', '--run', str(run), *SMOKE, '--iters', '3',
                             '--games', '600', '--conc', '64'], cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH))
    deadline = time.time() + 120
    while not os.path.exists(run / 'progress.json') and time.time() < deadline:   # self-play of iteration 1
        time.sleep(0.2)
    runctl.request_pause(str(run), by='test')
    t0 = time.time()
    out, _ = proc.communicate(timeout=120)
    assert proc.returncode == runctl.EXIT_PAUSED, out
    assert time.time() - t0 < 30
    st = runctl.read_state(str(run), 'run')
    assert st['state'] == 'paused'
    assert runctl.run_status(str(run))['state'] == 'paused'
    # a run started while PAUSE exists does not start
    p = run_loop(run, '--iters', '1')
    assert p.returncode == runctl.EXIT_PAUSED
    runctl.request_resume(str(run))
    p = run_loop(run, '--iters', '1', '--games', '24')
    assert p.returncode == 0, p.stdout + p.stderr
    assert [r['iter'] for r in read_metrics(str(run))] == [0, 1]


FAKE_RUN = '''import os, sys
run = sys.argv[sys.argv.index("--run") + 1]
p = os.path.join(run, "count")
k = open(p).read() if os.path.exists(p) else ""
open(p, "w").write(k + "x")
if len(k) == 1:
    open(os.path.join(run, "PAUSE"), "w").write("{}")
    sys.exit(75)
sys.exit(3 if len(k) == 0 else 0)
'''


def test_supervisor_with_fake_run(tmp_path):
    run = tmp_path / 'sup'
    run.mkdir()
    (run / 'config.json').write_text('{}')
    fake = tmp_path / 'fake.py'
    # run 1 crashes, run 2 pauses itself (PAUSE + exit 75), run 3 finishes
    fake.write_text(FAKE_RUN)
    env = dict(os.environ, ZERO_SUPERVISE_WAIT='0.2', ZERO_SUPERVISE_POLL='0.1',
               ZERO_SUPERVISE_CMD=json.dumps([sys.executable, str(fake)]))
    sup = subprocess.Popen([sys.executable, 'tools/supervise.py', str(run)], cwd=ROOT, env=env)
    deadline = time.time() + 30
    while time.time() < deadline and (runctl.read_state(str(run), 'supervisor') or {}).get('state') != 'paused':
        time.sleep(0.05)
    assert (runctl.read_state(str(run), 'supervisor') or {}).get('state') == 'paused'
    assert (run / 'count').read_text() == 'xx'
    runctl.request_resume(str(run))
    assert sup.wait(timeout=30) == 0
    assert (run / 'count').read_text() == 'xxx'
    log = (run / 'supervisor.log').read_text()
    assert 'crash #1' in log and 'run paused' in log and 'clean exit' in log


STALLED_RUN = '''import os, sys, time
run = sys.argv[sys.argv.index("--run") + 1]
p = os.path.join(run, "count")
k = open(p).read() if os.path.exists(p) else ""
open(p, "w").write(k + "x")
if len(k) == 0:
    time.sleep(120)
'''


def test_supervisor_stall_watchdog(tmp_path):
    # run 1 hangs (no progress files change), run 2 finishes
    run = tmp_path / 'stall'
    run.mkdir()
    (run / 'config.json').write_text('{}')
    fake = tmp_path / 'fake.py'
    fake.write_text(STALLED_RUN)
    env = dict(os.environ, ZERO_SUPERVISE_WAIT='0.2', ZERO_SUPERVISE_POLL='0.1', ZERO_SUPERVISE_STALL='1',
               ZERO_SUPERVISE_CMD=json.dumps([sys.executable, str(fake)]))
    sup = subprocess.Popen([sys.executable, 'tools/supervise.py', str(run)], cwd=ROOT, env=env)
    assert sup.wait(timeout=60) == 0
    assert (run / 'count').read_text() == 'xx'
    log = (run / 'supervisor.log').read_text()
    assert 'stall watchdog: no progress' in log and 'crash #1' in log and 'clean exit' in log


def test_kill_run_finds_nothing_and_pause_tool(tmp_path):
    run = tmp_path / 'k'
    run.mkdir()
    (run / 'config.json').write_text('{}')
    p = subprocess.run([sys.executable, 'tools/kill_run.py', str(run)], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0 and 'no processes' in p.stdout
    p = subprocess.run([sys.executable, 'tools/pause_run.py', str(run)], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0 and (run / 'PAUSE').exists()
    p = subprocess.run([sys.executable, 'tools/pause_run.py', str(run), '--resume'], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0 and not (run / 'PAUSE').exists()


def test_fork_run(smoke_run, tmp_path):
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import fork_run
    assert fork_run.main([str(smoke_run / 'models' / 'iter_0002.pt'), str(tmp_path / 'n')]) == 0
    assert (tmp_path / 'n' / 'models' / 'iter_0000.pt').exists()
    c = tmp_path / 'c'
    assert fork_run.main(['--source', str(smoke_run), '--dest', str(c), '--iter', '1', '--set', 'reuse=8']) == 0
    assert json.load(open(c / 'config.json'))['reuse'] == 8
    assert latest_iter(str(c)) == 1 and [r['iter'] for r in read_metrics(str(c))] == [0, 1]
    p = run_loop(c, '--iters', '2')
    assert p.returncode == 0, p.stderr
    assert [r['iter'] for r in read_metrics(str(c))] == [0, 1, 2]


def test_fork_run_policy_target_ab(smoke_run, tmp_path):
    """The A/B procedure of docs/az.md 9.4: two forks of one checkpoint, a visits control and a
    completed-Q arm, play the same first self-play games; only the recorded targets differ."""
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import fork_run
    vis, cq = tmp_path / 'vis', tmp_path / 'cq'
    assert fork_run.main(['--source', str(smoke_run), '--dest', str(vis), '--iter', '1']) == 0
    assert fork_run.main(['--source', str(smoke_run), '--dest', str(cq), '--iter', '1', '--set', 'policy_target=cq',
                          '--set', 'cq_c_visit=50', '--set', 'cq_c_scale=1.0']) == 0
    assert json.load(open(cq / 'config.json'))['policy_target'] == 'cq'
    with pytest.raises(SystemExit):
        fork_run.main(['--source', str(smoke_run), '--dest', str(tmp_path / 'bad'), '--set', 'policy_targett=cq'])
    for run in (vis, cq):
        p = run_loop(run, '--iters', '2')
        assert p.returncode == 0, p.stderr
    sv, sc = (read_metrics(str(r))[-1]['selfplay'] for r in (vis, cq))
    assert sv['samples'] == sc['samples'] and sv['avg_score'] == sc['avg_score']
    assert sv['target']['kind'] == 'visits' and sc['target']['kind'] == 'cq'
    assert sc['target']['h_norm'] < sv['target']['h_norm'] and sc['target']['max_share'] > sv['target']['max_share']


def test_dashboard(smoke_run, tmp_path):
    from zero import dashboard
    app = dashboard.app
    app.config['RUNS'] = str(tmp_path / 'none')
    os.makedirs(app.config['RUNS'])
    c = app.test_client()
    assert c.get('/').status_code == 200 and b'No runs yet' in c.get('/').data
    assert c.get('/api/runs').get_json() == []
    app.config['RUNS'] = str(smoke_run.parent)
    runs = c.get('/api/runs').get_json()
    assert runs[0]['name'] == 'smoke'
    d = c.get('/api/run/smoke').get_json()
    assert len(d['metrics']) == 3 and d['anchor'] == 'ext:toyheur' and d['control']['state'] == 'stopped'
    assert c.get('/api/run/..%2Fsmoke').status_code in (404, 405)
    assert c.get('/api/run/nope').status_code == 404
    assert c.post('/api/run/smoke/pause').status_code == 400          # JSON body required
    r = c.post('/api/run/smoke/pause', json={})
    assert r.status_code == 200 and os.path.exists(smoke_run / 'PAUSE')
    r = c.post('/api/run/smoke/resume', json={})
    assert r.status_code == 200 and not os.path.exists(smoke_run / 'PAUSE')
    assert c.post('/api/run/smoke/explode', json={}).status_code == 404


def test_dashboard_deep_evals(tmp_path):
    """deep_evals.jsonl (long-search strength points from a job outside the loop): served by
    /api/run as 'deep', the last row of each iter and sims, sorted; a half-written line skipped."""
    from zero import dashboard
    run = tmp_path / 'runs' / 'deep'
    run.mkdir(parents=True)
    (run / 'config.json').write_text('{}')
    dashboard.app.config['RUNS'] = str(tmp_path / 'runs')
    c = dashboard.app.test_client()
    assert c.get('/api/run/deep').get_json()['deep'] == [] and b'deep-card' in c.get('/').data
    rows = [{'iter': 2, 'sims': 1024, 'opponent': 'jsbot', 'games': 40, 'win': 0.5, 'se': 0.08, 'margin': 1.0},
            {'iter': 1, 'sims': 2048, 'opponent': 'jsbot', 'games': 40, 'win': 0.6, 'se': 0.08, 'margin': 2.0},
            {'iter': 2, 'sims': 1024, 'opponent': 'jsbot', 'games': 40, 'win': 0.7, 'se': 0.07, 'margin': 3.0},
            {'iter': 3, 'sims': 1024}]
    (run / 'deep_evals.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n{"iter": 4, "sims"\n')
    deep = c.get('/api/run/deep').get_json()['deep']
    assert [(e['iter'], e['sims'], e['win']) for e in deep] == [(1, 2048, 0.6), (2, 1024, 0.7)]


def test_remote_evaluation_local(tmp_path):
    """eval_remote: the loop trains without evaluating; every evaluated checkpoint is evaluated on
    the remote (here 'local': a mirror dir and a subprocess) into evals.jsonl + elo/, which the
    dashboard merges into the iterations' rows."""
    run = tmp_path / 'rem'
    p = run_loop(run, '--iters', '2', '--yardstick', 'ext:toyheur', '--eval-remote', 'local',
                 env={'ZERO_REMOTE_WORKDIR': str(tmp_path / 'mirror')})
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    assert [r['iter'] for r in m] == [0, 1, 2] and not any('eval_yard' in r for r in m)
    ev = [json.loads(line) for line in open(run / 'evals.jsonl')]
    assert sorted(e['iter'] for e in ev) == [0, 1, 2]
    assert all(e['eval_yard']['games'] == 8 and e['eval_yard']['opp'] == 'ext:toyheur' and e['host'] == 'local' for e in ev)
    r = json.load(open(run / 'elo' / 'ratings.json'))
    assert {'iter_0000', 'iter_0002', 'random'} <= set(r['ratings'])
    assert (tmp_path / 'mirror' / 'rem' / 'models' / 'iter_0002.pt').exists()
    from zero import dashboard
    rows = dashboard._metrics(str(run))
    assert rows[2]['eval_yard'] == next(e for e in ev if e['iter'] == 2)['eval_yard'] and 'elo' in rows[2]


def test_remote_evaluation_continues_a_local_run(tmp_path):
    """Switching eval_remote on for a run evaluated here so far: nothing evaluated again, the
    remote continues the run's Elo (the local matches are kept)."""
    run = tmp_path / 'sw'
    p = run_loop(run, '--iters', '2', '--yardstick', 'ext:toyheur')
    assert p.returncode == 0, p.stdout + p.stderr
    before = json.load(open(run / 'elo' / 'results.json'))
    p = run_loop(run, '--iters', '4', '--yardstick', 'ext:toyheur', '--eval-remote', 'local',
                 env={'ZERO_REMOTE_WORKDIR': str(tmp_path / 'mirror')})
    assert p.returncode == 0, p.stdout + p.stderr
    ev = [json.loads(line) for line in open(run / 'evals.jsonl')]
    assert sorted(e['iter'] for e in ev) == [3, 4]
    after = json.load(open(run / 'elo' / 'results.json'))
    pairs = lambda r: {json.dumps(x, sort_keys=True) for x in (r['pairs'].items() if isinstance(r['pairs'], dict) else r['pairs'])}  # noqa: E731
    assert before['pairs'] and pairs(before) <= pairs(after), 'the local Elo matches are kept'
    assert set(before['logged']) <= set(after['logged'])
    r = json.load(open(run / 'elo' / 'ratings.json'))['ratings']
    assert {'iter_0001', 'iter_0004'} <= set(r)


def test_tokens_arch_cheap_elo_and_minor_settings(tmp_path):
    """arch 'tokens' (the toy's object layout) through the whole loop; Elo against the previous
    checkpoint only, at its own sims, all of an iteration's matches in one driver; sims_minor
    (the toy has no minor decisions: nothing changes)."""
    run = tmp_path / 'tok'
    p = run_loop(run, '--iters', '2', '--arch', 'tokens', '--hidden', '16', '--blocks', '1', '--heads', '2',
                 '--elo-opponents', '1', '--elo-sims', '4', '--elo-together', '1', '--sims-minor', '2')
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(run))
    assert [r['iter'] for r in m] == [0, 1, 2]
    assert m[2]['selfplay']['minor_per_game'] == 0 and m[2]['train']['steps'] >= 10
    from zero.net import load
    net, _ = load(str(run / 'models' / 'iter_0002.pt'))
    assert net.config['arch'] == 'tokens' and net.config['d'] == 16 and net.config['ff'] == 32
    pairs = json.load(open(run / 'elo' / 'results.json'))['pairs']
    games = {k: v[1] for k, v in pairs.items() if k.startswith('iter_') and '|iter_' in k}
    assert games == {'iter_0000|iter_0001': 4, 'iter_0001|iter_0002': 4}, pairs
    cfg = json.load(open(run / 'config.json'))
    assert (cfg['elo_opponents'], cfg['elo_sims'], cfg['elo_together'], cfg['sims_minor']) == (1, 4, 1, 2)


def test_eval_spec_is_the_spec_of_every_match(monkeypatch, tmp_path):
    """Drivers that record (self-play) get `spec`; every match gets `eval_spec` when set."""
    from zero import run as R
    seen = []

    class Fake:
        def __init__(self, game, seats, spec=None, **kw):
            seen.append((spec, kw['record']))
    monkeypatch.setattr(R.G, 'Driver', Fake)
    cfg = dict(R.DEFAULTS, spec='{"a": 1}', eval_spec='{"b": 2}')
    r = R.Run(str(tmp_path), cfg)
    r.driver([[{'bot': 'random'}] * 2], seed=1, record=True)
    r.driver([[{'bot': 'random'}] * 2], seed=1)
    r.cfg['eval_spec'] = ''
    r.driver([[{'bot': 'random'}] * 2], seed=1)
    from zero import package
    pkg = package.active()
    pkg.eval_specs['toy'] = lambda s: s + ' (eval)'          # a game package's evaluation spec
    try:
        r.driver([[{'bot': 'random'}] * 2], seed=1)
        r.driver([[{'bot': 'random'}] * 2], seed=1, record=True)
    finally:
        pkg.eval_specs.pop('toy')
    assert seen == [('{"a": 1}', True), ('{"b": 2}', False), ('{"a": 1}', False), ('{"a": 1} (eval)', False),
                    ('{"a": 1}', True)]
    assert R.DEFAULTS['eval_spec'] == '' and R.DEFAULTS['elo_opponents'] == 0 and R.DEFAULTS['sims_minor'] == 0
