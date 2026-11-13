"""The encoding guard (zero/net.py `check_encoding`, docs/az.md 10.4): a checkpoint whose network
was built for another encoding than the one this build's game feeds it is refused wherever it is
loaded - run resume (exit 78, not restarted by the supervisor), the evaluators and every match,
the optimizer state, co-training (both networks; resolver_init widens and checks), remote
evaluation and remote self-play (refused there, logged in full here), h2h, replay, the tools - and
only fork_run's widen, resolver_init, selfplay_digest (check_checkpoint_compat) and the widen tests
take one on purpose (`allow_mismatch=True`).

The "v4" stand-ins (MTG Zero's layout v4 -> v5, docs/ENGINE.md 60): the toy's encoding with the
players' slots one float narrower (`V4_TOKENS`, 191 features: a token network of it reads the toy's
first 191 floats without an error), and an MLP of 185 features. The toy build is "v5"."""
import json
import os
import subprocess
import sys

import pytest
import torch

from zero import games as G, package, runctl
from zero.infer import NetEvaluator
from zero.net import (EncodingMismatch, GridNet, Net, TokenNet, check_checkpoint, check_encoding, encoding_mismatch,
                      grid_spec, load, make_net, save, token_feature_map, tokens_layout)

from test_cotrain import co_loop
from test_loop import PYPATH, ROOT, run_loop

INFO = G.info('toy')
F, A = INFO['num_features'], INFO['num_actions']
TOY = package.TOY_TOKENS
V4_TOKENS = {'vocab': TOY['vocab'], 'segments': [dict(TOY['segments'][0]), dict(TOY['segments'][1], width=15)],
             'bags': [dict(b) for b in TOY['bags']]}
F4 = F - 4


def v4_token_net(seed=0):
    torch.manual_seed(seed)
    m = TokenNet(F4, A, V4_TOKENS, d=16, layers=1, heads=2, ff=32, head=16, game='toy').eval()
    with torch.no_grad():
        for lin in (m.gen, m.ptr):
            lin.weight.normal_(0, 0.3)
            lin.bias.normal_(0, 0.3)
    return m


@pytest.fixture
def v4(tmp_path):
    """Paths of a "v4" token checkpoint and a narrower MLP checkpoint (both labelled 'toy')."""
    tok, mlp = str(tmp_path / 'v4_tokens.pt'), str(tmp_path / 'v4_mlp.pt')
    save(v4_token_net(1), tok, {'iter': 5})
    torch.manual_seed(2)
    save(Net(F - 10, A, 16, 1, game='toy').eval(), mlp, {'iter': 5})
    return tok, mlp


def a_batch(n=8, seed=0):
    seat = G.nn_seat('m', 4)
    drv = G.Driver('toy', [[seat, {'bot': 'random'}]] * n, concurrency=n, threads=1, seed=seed)
    return drv.next_batch()


# ---------------------------------------------------------------- the check
def test_matching_networks_pass_and_the_layout_identity_is_recorded(tmp_path):
    for arch in ('mlp', 'tokens'):
        m = make_net(INFO, 'toy', arch, 16, 1, heads=2)
        assert encoding_mismatch(m.config) is None and encoding_mismatch(m.config, 'toy') is None
        p = str(tmp_path / f'{arch}.pt')
        save(m, p)
        check_checkpoint(p, 'toy')
        m2, _ = load(p, game='toy')
        assert m2.config == m.config
    # the toy layout has no identity: its configs are as before; a layout with one keeps it
    assert set(tokens_layout(TOY, F, A)) == {'vocab', 'segments', 'bags'}
    lay = tokens_layout(dict(TOY, version=5, name='toy-objects'), F, A)
    assert lay['version'] == 5 and lay['name'] == 'toy-objects'
    assert {k: v for k, v in lay.items() if k not in ('version', 'name')} == tokens_layout(TOY, F, A)


def test_a_v4_network_is_refused_naming_both_sides(v4):
    tok, mlp = v4
    with pytest.raises(EncodingMismatch) as e:
        check_checkpoint(tok)
    msg = str(e.value)
    assert tok in msg and f'features {F4} -> {F}' in msg and 'segment players: width 15 -> 16' in msg
    assert f"The network: 'toy': {F4:,} features" in msg and f"This build: 'toy': {F:,} features" in msg
    assert 'tools/fork_run.py' in msg and 'docs/az.md 10.4' in msg
    with pytest.raises(EncodingMismatch, match=f'features {F - 10} -> {F}'):
        check_checkpoint(mlp)
    assert issubclass(EncodingMismatch, ValueError)


def test_what_the_check_compares(monkeypatch):
    m = make_net(INFO, 'toy', 'tokens', 16, 1, heads=2)
    c = m.config
    # another game's network, an unknown game, the game a site plays (not the label's)
    with pytest.raises(EncodingMismatch, match="game 'toy_team' -> 'toy'"):
        check_encoding(dict(c, game='toy_team'), 'toy')
    with pytest.raises(EncodingMismatch, match="'nope' is not a game of this build"):
        check_encoding(dict(c, game='nope'))
    with pytest.raises(EncodingMismatch, match='features 195 -> 64'):
        check_encoding(c, 'toy_team')
    # the same width, another layout (a segment moved): refused by the layout alone
    lay = json.loads(json.dumps(c['layout']))
    lay['segments'][1]['offset'] -= 1
    with pytest.raises(EncodingMismatch, match='segment players: offset 122 -> 121|segment players: offset 121 -> 122'):
        check_encoding(dict(c, layout=lay))
    # declared global blocks change nothing in the network: not compared
    glob = dict(TOY, globals=[{'name': 'head', 'offset': 0, 'count': 25}, {'name': 'tail', 'offset': 186, 'count': 9}])
    assert encoding_mismatch(dict(c, layout=tokens_layout(glob, F, A))) is None
    # the layout's version: compared when both sides carry one
    monkeypatch.setitem(package.active().tokens, 'toy', dict(TOY, version=5))
    assert encoding_mismatch(c) is None                              # the network has none: by its structure
    assert encoding_mismatch(dict(c, layout=dict(c['layout'], version=5))) is None
    with pytest.raises(EncodingMismatch, match="token layout version 4 -> 5"):
        check_encoding(dict(c, layout=dict(c['layout'], version=4)))
    monkeypatch.undo()
    # an MLP (no layout metadata): the counts alone, whatever the package's layout
    assert encoding_mismatch(Net(F, A, 8, 1, game='toy').config) is None
    with pytest.raises(EncodingMismatch, match=f'actions {A - 2} -> {A}'):
        check_encoding(Net(F, A - 2, 8, 1, game='toy').config)


def test_a_conv_network_is_judged_by_its_board_grid(monkeypatch):
    grid = {'planes': 2, 'h': 3, 'w': 4, 'offset': 40, 'cell_action0': 2, 'hex': 'odd-r'}
    monkeypatch.setitem(package.active().grids, 'toy', grid)
    m = GridNet(F, A, grid, ch=4, blocks=1, head=8, ghid=8, game='toy')
    assert encoding_mismatch(m.config) is None
    old = dict(m.config, grid={k: v for k, v in m.config['grid'].items() if k != 'segments'})   # an older normal form
    assert grid_spec(old['grid']) == m.config['grid'] and encoding_mismatch(old) is None
    other = GridNet(F, A, dict(grid, planes=1), ch=4, blocks=1, head=8, ghid=8, game='toy')
    with pytest.raises(EncodingMismatch, match='board grid'):
        check_encoding(other.config)
    monkeypatch.delitem(package.active().grids, 'toy')
    assert encoding_mismatch(other.config) is None                  # no grid in the package: the counts alone


# ---------------------------------------------------------------- the load sites
def test_load_sites_refuse_a_v4_checkpoint(v4, tmp_path):
    from zero import bench, h2h, replay, run, targets, webexport
    tok, mlp = v4
    ok = str(tmp_path / 'ok.pt')
    save(make_net(INFO, 'toy', 'mlp', 16, 1), ok)
    for p in (tok, mlp):
        with pytest.raises(EncodingMismatch):
            load(p)
        with pytest.raises(EncodingMismatch):
            load(p, game='toy')
    R = run.Run(str(tmp_path / 'r'), run.resolve_config(run.parse(['--run', 'x']), {}))
    with pytest.raises(EncodingMismatch):
        R.evaluator(tok)                                             # every match, self-play and Elo
    R.evaluator(ok)
    with pytest.raises(EncodingMismatch):
        h2h.match(tok, ok, games=2, sims=2, threads=1)
    with pytest.raises(EncodingMismatch):
        h2h.match(ok, mlp, games=2, sims=2, threads=1)
    with pytest.raises(EncodingMismatch):
        replay.record(tok, games=1, sims=2, log=lambda *a: None)
    with pytest.raises(EncodingMismatch):
        replay.record(ok, opp=mlp, games=1, sims=2, log=lambda *a: None)
    with pytest.raises(EncodingMismatch):
        targets.selfplay_samples(tok, 'toy', '', 1, G.nn_seat('m', 2))
    with pytest.raises(EncodingMismatch):
        bench.bench_selfplay(mlp, game='toy', games=1)
    with pytest.raises(EncodingMismatch):
        webexport.export_model(mlp, str(tmp_path / 'm.bin'))
    # the opt-out: the network as it is
    m, extra = load(tok, allow_mismatch=True)
    assert m.config['features'] == F4 and extra['iter'] == 5


def test_the_evaluator_refuses_rows_of_another_width():
    """NetEvaluator's last line: a network narrower than the driver's rows would read their first
    floats; refused, unless told otherwise (selfplay_digest)."""
    m = v4_token_net(3)
    lb = a_batch()
    with pytest.raises(EncodingMismatch, match=f'{F:,} features per row for a network of {F4:,}'):
        NetEvaluator(m)(lb)
    ev = NetEvaluator(m, workers=2)                                  # (on a worker thread)
    with pytest.raises(EncodingMismatch):
        ev.result(ev.launch(lb))
    pri, val, mar = NetEvaluator(m, allow_mismatch=True)(lb)
    assert len(pri) == len(lb.opt_action)
    NetEvaluator(make_net(INFO, 'toy', 'tokens', 16, 1, heads=2))(lb)


def test_optimizer_state_of_another_network_is_not_loaded():
    from zero.run import opt_state_mismatch
    a, b = Net(F, A, 16, 1), Net(F - 10, A, 16, 1)
    st = {}
    for k, m in (('a', a), ('b', b)):
        opt = torch.optim.AdamW(m.parameters())
        m(torch.zeros(2, m.features), torch.full((2,), 2))[0].sum().backward()
        opt.step()
        st[k] = opt.state_dict()
    assert opt_state_mismatch(a, st['a']) is None
    why = opt_state_mismatch(a, st['b'])
    assert 'exp_avg' in why and '(16, 185)' in why and '(16, 195)' in why
    assert 'parameters' in opt_state_mismatch(Net(F, A, 16, 2), st['a'])


# ---------------------------------------------------------------- runs
def test_run_resume_refuses_exits_78_and_the_supervisor_does_not_restart(v4, tmp_path):
    tok, _ = v4
    run = tmp_path / 'v4run'
    (run / 'models').mkdir(parents=True)
    with open(tok, 'rb') as f:
        (run / 'models' / 'iter_0000.pt').write_bytes(f.read())
    p = run_loop(run, '--iters', '1')
    assert p.returncode == runctl.EXIT_REFUSED == 78, p.stdout + p.stderr
    assert 'EncodingMismatch' in p.stderr and 'segment players: width 15 -> 16' in p.stderr
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'run REFUSED' in log and 'iter 0' not in log
    assert sorted(os.listdir(run / 'models')) == ['iter_0000.pt'] and not (run / 'metrics.jsonl').exists()
    st = runctl.run_status(str(run))
    assert st['run']['state'] == 'refused' and st['state'] == 'stopped' and 'encoding' in st['detail']
    # the supervisor: no restart (a fake run that refuses)
    sup = tmp_path / 'sup'
    sup.mkdir()
    fake = tmp_path / 'fake.py'
    fake.write_text('import os, sys\nrun = sys.argv[sys.argv.index("--run") + 1]\n'
                    'p = os.path.join(run, "count")\nk = open(p).read() if os.path.exists(p) else ""\n'
                    'open(p, "w").write(k + "x")\nsys.exit(78)\n')
    env = dict(os.environ, ZERO_SUPERVISE_WAIT='0.1', ZERO_SUPERVISE_CMD=json.dumps([sys.executable, str(fake)]))
    s = subprocess.run([sys.executable, 'tools/supervise.py', str(sup)], cwd=ROOT, env=env, timeout=60)
    assert s.returncode == 0 and (sup / 'count').read_text() == 'x'
    assert 'REFUSED' in (sup / 'supervisor.log').read_text() and runctl.read_state(str(sup), 'supervisor')['state'] == 'refused'


def test_fork_run_widens_a_v4_checkpoint_exactly_and_the_fork_runs(v4, tmp_path):
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import fork_run
    tok, mlp = v4
    src = v4_token_net(1)
    fork = tmp_path / 'fork'
    assert fork_run.main([tok, str(fork)]) == 0
    w, extra = load(str(fork / 'models' / 'iter_0000.pt'))          # past the guard
    assert w.config['features'] == F and extra['source_extra'] == {'iter': 5}
    fmap = token_feature_map(src.config['layout'], F4, w.config['layout'], F)
    x = torch.from_numpy(a_batch(16, 5).x)
    n = torch.full((x.shape[0],), 2)
    with torch.no_grad():
        assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(src(x[:, fmap], n), w(x, n)))
    p = run_loop(fork, '--iters', '1')                              # the fork resumes past the guard
    assert p.returncode == 0, p.stdout + p.stderr
    assert (fork / 'models' / 'iter_0001.pt').exists()
    assert fork_run.main([mlp, str(tmp_path / 'fork_mlp')]) == 0
    check_checkpoint(str(tmp_path / 'fork_mlp' / 'models' / 'iter_0000.pt'))


def test_fork_run_refuses_to_write_a_fork_this_build_cannot_feed(v4, tmp_path, monkeypatch):
    """A fork the widen cannot bring onto this build's encoding is not written."""
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import fork_run
    tok, _ = v4
    monkeypatch.setattr('zero.net.widen', lambda old, *a, **k: old)   # (a widen that does not get there)
    with pytest.raises(EncodingMismatch, match='fork of'):
        fork_run.main([tok, str(tmp_path / 'bad')])
    assert not (tmp_path / 'bad' / 'models' / 'iter_0000.pt').exists()


def test_cotrain_resolver_init_widens_a_v4_checkpoint_and_resume_refuses_a_v4_resolver(v4, tmp_path):
    tok, _ = v4
    run = tmp_path / 'co'
    p = co_loop(run, '--iters', '1', '--resolver-init', tok)
    assert p.returncode == 0, p.stdout + p.stderr
    r0, _ = load(str(run / 'resolver' / 'models' / 'iter_0000.pt'), game='toy')
    src = v4_token_net(1)
    fmap = token_feature_map(src.config['layout'], F4, r0.config['layout'], F)
    x = torch.from_numpy(a_batch(12, 6).x)
    n = torch.full((x.shape[0],), 2)
    with torch.no_grad():
        assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(src(x[:, fmap], n), r0(x, n)))
    # a resolver checkpoint of the old encoding at the resume point: refused
    bad = tmp_path / 'co_bad'
    (bad / 'resolver' / 'models').mkdir(parents=True)
    (bad / 'models').mkdir(parents=True)
    with open(run / 'models' / 'iter_0000.pt', 'rb') as f:
        (bad / 'models' / 'iter_0000.pt').write_bytes(f.read())
    with open(tok, 'rb') as f:
        (bad / 'resolver' / 'models' / 'iter_0000.pt').write_bytes(f.read())
    p = co_loop(bad, '--iters', '1')
    assert p.returncode == 78, p.stdout + p.stderr
    assert 'resolver' in p.stderr and 'EncodingMismatch' in p.stderr


def test_cotrain_start_keeps_an_existing_checkpoint_0(v4, tmp_path):
    """Co-training's first start (no iteration with both networks) never overwrites a checkpoint: a
    primary forked into iteration 0 by fork_run survives, bit for bit, while resolver_init brings in
    the resolver; a resolver checkpoint 0 is kept too; an ambiguous run directory is refused."""
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import fork_run
    tok, _ = v4
    src = str(tmp_path / 'draft.pt')
    torch.manual_seed(123)                                           # (not the preset's fresh network)
    save(make_net(G.info('toy_team'), 'toy_team', 'mlp', 16, 1), src, {'iter': 9})
    run = tmp_path / 'forked'
    assert fork_run.main([src, str(run)]) == 0
    forked = torch.load(run / 'models' / 'iter_0000.pt', weights_only=False)
    p = co_loop(run, '--iters', '1', '--resolver-init', tok)
    assert p.returncode == 0, p.stdout + p.stderr
    after = torch.load(run / 'models' / 'iter_0000.pt', weights_only=False)
    assert after['extra'] == forked['extra'] and all(torch.equal(after['state'][k], forked['state'][k]) for k in forked['state'])
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert 'primary iteration 0 = the checkpoint already there, kept' in log and 'resolver iteration 0 = ' in log
    assert (run / 'models' / 'iter_0001.pt').exists()
    # a resolver checkpoint 0 already there: kept without resolver_init, the primary created
    run2 = tmp_path / 'rkept'
    (run2 / 'resolver' / 'models').mkdir(parents=True)
    r0 = run / 'resolver' / 'models' / 'iter_0000.pt'
    (run2 / 'resolver' / 'models' / 'iter_0000.pt').write_bytes(r0.read_bytes())
    p = co_loop(run2, '--iters', '0')
    assert p.returncode == 0, p.stdout + p.stderr
    assert (run2 / 'resolver' / 'models' / 'iter_0000.pt').read_bytes() == r0.read_bytes()
    assert (run2 / 'models' / 'iter_0000.pt').exists()
    # ... and refused with resolver_init (which one?), as is a run with later checkpoints but no
    # iteration with both
    run3 = tmp_path / 'both'
    (run3 / 'resolver' / 'models').mkdir(parents=True)
    (run3 / 'resolver' / 'models' / 'iter_0000.pt').write_bytes(r0.read_bytes())
    p = co_loop(run3, '--iters', '1', '--resolver-init', tok)
    assert p.returncode != 0 and 'resolver_init' in p.stderr and 'already there' in p.stderr
    assert (run3 / 'resolver' / 'models' / 'iter_0000.pt').read_bytes() == r0.read_bytes()
    assert not (run3 / 'models' / 'iter_0000.pt').exists()
    run4 = tmp_path / 'later'
    (run4 / 'models').mkdir(parents=True)
    (run4 / 'models' / 'iter_0003.pt').write_bytes((run / 'models' / 'iter_0001.pt').read_bytes())
    p = co_loop(run4, '--iters', '1')
    assert p.returncode != 0 and 'after iteration 0' in p.stderr
    assert sorted(os.listdir(run4 / 'models')) == ['iter_0003.pt']


# ---------------------------------------------------------------- the remotes
def test_remote_evaluation_refused_by_the_remote_build(v4, tmp_path, monkeypatch):
    """The remote side (`python -m <pkg>.az.remote eval`) refuses a checkpoint of another encoding
    (exit 78, one ZERO_REMOTE_REFUSED line); the run logs it in full and retries."""
    from zero import remote
    tok, _ = v4
    mirror = tmp_path / 'wd' / 'm'
    (mirror / 'models').mkdir(parents=True)
    with open(tok, 'rb') as f:
        (mirror / 'models' / 'iter_0000.pt').write_bytes(f.read())
    from zero.run import parse, resolve_config
    cfg = resolve_config(parse(['--run', 'x', '--preset', 'toy-smoke']), {})
    (mirror / 'config.json').write_text(json.dumps(cfg))
    e = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH)
    r = subprocess.run([sys.executable, '-m', 'zero.remote', 'eval', str(mirror), '0'], cwd=ROOT, capture_output=True,
                       text=True, timeout=120, env=e)
    assert r.returncode == 78, r.stdout + r.stderr
    ex = remote._refused(r)
    assert isinstance(ex, remote.RemoteRefused) and 'segment players: width 15 -> 16' in str(ex)
    assert not os.path.exists(tmp_path / 'wd' / '.eval.lock')                         # the lock is released
    # the run's side: logged in full (not cut at 400 characters), retried
    from types import SimpleNamespace
    logs = []
    fake = SimpleNamespace(run=str(tmp_path / 'run'), cfg=cfg, log=logs.append)
    monkeypatch.setattr(remote, 'RETRY_S', 0.01)
    calls = []

    def evaluate(it):
        calls.append(it)
        if len(calls) == 1:
            raise ex
    rv = remote.RemoteEval(fake, 'local')
    monkeypatch.setattr(rv, '_evaluate', evaluate)
    rv.enqueue(0)
    import time
    t_end = time.time() + 20                       # (drain has no time limit of its own, docs/az.md 47.3: the test's bound)
    assert rv.drain(stop=lambda: time.time() > t_end, poll_s=0.05)
    assert calls == [0, 0] and "REFUSED by the remote's build" in logs[0] and str(ex) in logs[0]


def test_remote_selfplay_worker_refuses_and_the_run_plays_on(tmp_path):
    """A worker whose checkout's game feeds another encoding (the tests' 'load_game') refuses the
    run's checkpoint: its error 'encoding', the run's log names it in full; this machine plays
    every game."""
    from test_remoteplay import TOY, env_for
    run = tmp_path / 'rsp'
    p = run_loop(run, '--iters', '2', *TOY, '--selfplay-remote', 'local', env=env_for(tmp_path, {'load_game': 'toy_team'}))
    assert p.returncode == 0, p.stdout + p.stderr
    log = (run / 'log.txt').read_text(encoding='utf-8')
    assert "REFUSED the run's checkpoint" in log and "game 'toy' -> 'toy_team'" in log
    from zero.run import read_metrics
    m = read_metrics(str(run))
    assert [r['iter'] for r in m] == [0, 1, 2]
    assert all(r['selfplay']['remote']['games'] == 0 and r['selfplay']['games'] == 600 for r in m[1:])


# ---------------------------------------------------------------- check_checkpoint_compat's opt-out
def test_selfplay_digest_plays_a_checkpoint_of_another_encoding(v4):
    """tools/selfplay_digest.py (check_checkpoint_compat) compares checkouts on purpose: it takes a
    narrower network as it is (the guard's opt-out)."""
    _, mlp = v4
    e = dict(os.environ, PYTHONIOENCODING='utf-8', CUDA_VISIBLE_DEVICES='-1')
    e.pop('PYTHONPATH', None)
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'selfplay_digest.py'), '--root',
                        os.path.join(ROOT, 'python'), '--package', 'zero', '--model', mlp, '--games', '2', '--sims', '4',
                        '--targets', 'visits', '--match-games', '2', '--game', 'toy'], cwd=ROOT, capture_output=True,
                       text=True, timeout=300, env=e)
    assert r.returncode == 0, r.stdout + r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out['evaluator']['config']['features'] == F - 10 and out['targets']['visits']['games'] == 2
