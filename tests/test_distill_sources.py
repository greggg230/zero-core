"""Multi-teacher distillation (docs/az.md 77): `tools/distill.py --sources`, the game package's row selector
(`GamePackage.row_selectors`, `zero.distill.select_rows`), `data.Window(select=...)`, `outputs(span=, into=)`,
the stratified batches (`shares`, `stratified_draw`, `distill(strata=)`) - on the toy game, whose test selector
takes the rows of one seat; and the single-teacher tool byte for byte as on the commit before the feature
(the golden digests below, recorded there)."""
import hashlib
import json
import os
import platform
import sys

import numpy as np
import pytest
import torch

from zero import data, distill as D, games as G, net, package

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def toy_shards(d, seed, games=30, rows=400):
    """Deterministic toy shards in `d` (no threads: games played one after the other with random legal actions,
    random policy targets): every decision of both seats, z / q the game's result from the row's seat first."""
    info = G.info('toy')
    meta = {'game': 'toy', 'features': info['num_features'], 'actions': info['num_actions']}
    rng = np.random.default_rng(seed)
    E = G.engine()
    w = data.ShardWriter(str(d), 'c00', rows=rows, meta=meta)
    for k in range(games):
        g = E.Game('toy', None, 2, seed * 1000 + k)
        xs, ps, offs, acts, pis = [], [], [0], [], []
        while not g.over:
            p = g.to_move
            legal = g.legal()
            xs.append(g.encode(p))
            ps.append(p)
            acts += list(legal)
            pis += list(rng.dirichlet(np.ones(len(legal))))
            offs.append(offs[-1] + len(legal))
            g.step(legal[int(rng.integers(len(legal)))])
        res = g.result()
        n = len(ps)
        z = np.array([[res[p], res[1 - p]] for p in ps], np.float32)
        sc = np.array([[g.margin(p), g.margin(1 - p)] for p in ps], np.float32)
        w.add({'x': np.stack(xs).astype(np.float32), 'player': np.array(ps), 'n_players': np.full(n, 2),
               'opt_off': np.array(offs), 'opt_action': np.array(acts), 'pi': np.array(pis, np.float32),
               'z': z, 'q': z, 'sc': sc, 'net': np.zeros(n, np.int32)})
    w.close()
    return d


@pytest.fixture(scope='module')
def shards(tmp_path_factory):
    root = tmp_path_factory.mktemp('distill_sources')
    toy_shards(root / 'data' / 'iter_0001', 1)
    toy_shards(root / 'data' / 'iter_0002', 2, games=12)
    return root


def mlp(seed, hidden=64, blocks=2):
    torch.manual_seed(seed)
    return net.make_net(G.info('toy'), 'toy', 'mlp', hidden, blocks)


def tool():
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import distill as tool_mod
    return tool_mod


def seat_selector(select, shard, ctx):
    """The toy's test selector: {"seat": k} = the rows of seat k, {"early": b} = the rows of the first 3 rounds
    (b true) or the others (the round's one-hot, features 0-5); AND-combined; exact and loud."""
    bad = sorted(set(select) - {'seat', 'early'})
    if bad:
        raise ValueError(f'unknown selector keys {bad}')
    with np.load(shard) as z:
        m = np.ones(len(z['player']), bool)
        if 'seat' in select:
            if select['seat'] not in (0, 1):
                raise ValueError(f'seat {select["seat"]!r}: the toy has seats 0 and 1')
            m &= z['player'] == int(select['seat'])
    if 'early' in select:
        m &= (data.shard_features(shard)[:, :6].argmax(1) < 3) == bool(select['early'])
    return m


@pytest.fixture
def selector(monkeypatch):
    monkeypatch.setitem(package.active().row_selectors, 'toy', seat_selector)
    return seat_selector


# ---------------------------------------------------------------- the single-teacher tool, byte for byte

# The digests of `single_teacher_digest` on the commit before the feature (c850a5d in a worktree of its own; the
# desktop's CPU, torch 2.11.0+cu128, Windows AMD64, Python 3.13; the generator above, run there by a script
# importing this module): the student's tensors, the progress log and the checkpoint's record, for an MLP student
# and a token student at teacher_w 0.5 with --keep-best. Floats are the machine's: the golden binds where it was
# recorded (another torch or machine skips, saying so).
GOLDEN_ENV = {'torch': '2.11.0+cu128', 'machine': 'AMD64'}
GOLDEN = {
    'mlp': ('3a7fcdf4ebe0d30766bbc64c99f42b716d67cf3495587ca35f9b04b8301f9a26',
            'dc2614a3238bae3912bf007133acc28e800bc2edaeb36b89cad28c93d11238ef',
            '1b9880d12690b52c2afd172ce461b315f96d0c71bb3a8132098efb56f1cba695'),
    'tokens': ('64ff4d642f16611f8badf3d330a6b70a1b45b157259ea37d748e7290ccfe00fa',
               '753199040a5d95af3c183bdaf7bd5ab2c58ffe25cd7cd95c5a70903c59bd06ca',
               'c6e2be52084222f454b460064db6b2f0c692e4008e132419b86c30c2bb89aa83'),
}
VOLATILE = ('time', 'seconds', 's_per_step', 'load_s', 'teacher_s', 'teacher', 'out')
CASES = {'mlp': ['--hidden', '32', '--blocks', '2', '--steps', '60', '--batch', '64', '--lr', '2e-3', '--warmup', '10',
                 '--eval-every', '30', '--report-every', '20'],
         'tokens': ['--arch', 'tokens', '--hidden', '16', '--blocks', '1', '--heads', '2', '--steps', '40', '--batch',
                    '32', '--teacher-w', '0.5', '--eval-every', '20', '--report-every', '10', '--keep-best']}


def single_teacher_digest(root, case, main):
    """The single-teacher tool (`main`: its `main`) on the shards under `root` -> (sha256 of the student's tensors,
    sha256 of its log without the volatile keys, sha256 of its checkpoint's record with the paths relative to
    `root` and the seconds and the teacher file's sha256 left out - torch writes a new id into every file -, the
    log's records)."""
    tpath = str(root / f'teacher-{case}.pt')
    net.save(mlp(5), tpath, {'iter': 7})
    out = str(root / f'student-{case}.pt')
    args = [tpath, out, '--data', str(root / 'data' / 'iter_0001'), '--heldout', str(root / 'data' / 'iter_0002'),
            '--save-every', '0', '--torch-threads', '1', '--normal-priority', '--seed', '3'] + CASES[case]
    assert main(args) == 0
    ck = torch.load(out, map_location='cpu', weights_only=False)
    s = ck['state']
    h = hashlib.sha256()
    for k in sorted(s):
        t = s[k].detach().cpu().contiguous()
        h.update(f'{k}|{t.dtype}|{tuple(t.shape)}|'.encode())
        h.update(t.numpy().tobytes())
    recs = [{k: v for k, v in json.loads(line).items() if k not in VOLATILE} for line in open(out + '.log.jsonl')]
    lh = hashlib.sha256(json.dumps(recs, sort_keys=True).encode()).hexdigest()
    top = os.path.abspath(str(root))

    def rel(o):
        if isinstance(o, dict):
            return {k: rel(v) for k, v in o.items() if k not in ('seconds', 'teacher_sha256')}
        if isinstance(o, (list, tuple)):
            return [rel(v) for v in o]
        if isinstance(o, str) and o.startswith(top):
            return os.path.relpath(o, top).replace(os.sep, '/')
        return o
    eh = hashlib.sha256(json.dumps(rel(ck['extra']), sort_keys=True, default=str).encode()).hexdigest()
    return h.hexdigest(), lh, eh, recs


@pytest.mark.parametrize('case', sorted(CASES))
def test_the_single_teacher_tool_is_byte_identical_to_the_commit_before(shards, case):
    """Same shards, teacher and seed: the student's weights, the progress log and the checkpoint's record equal
    those of the tool on the commit before the feature (GOLDEN), and the record has no key of the feature."""
    env = {'torch': torch.__version__, 'machine': platform.machine()}
    st, lh, eh, recs = single_teacher_digest(shards, case, tool().main)
    assert recs[0]['event'] == 'start' and recs[-1]['event'] == 'done'
    _, extra = net.load(str(shards / f'student-{case}.pt'), game='toy')
    assert 'sources' not in extra['distill']['settings'] and 'sources' not in extra['distill']
    if env != GOLDEN_ENV:
        pytest.skip(f'the golden binds on {GOLDEN_ENV} (here {env}; digests {st} {lh} {eh})')
    assert (st, lh, eh) == GOLDEN[case]


# ---------------------------------------------------------------- the parts


def test_shares_split_every_batch_by_weight():
    assert D.shares([1, 1, 1], 512) == [171, 171, 170]
    assert D.shares([3, 1], 64) == [48, 16] and sum(D.shares([0.3, 0.3, 0.4], 7)) == 7
    with pytest.raises(ValueError, match='no row'):
        D.shares([1000, 1], 64)
    with pytest.raises(ValueError, match='> 0'):
        D.shares([1, 0], 64)
    draw = D.stratified_draw([(0, 10, 3), (10, 1000, 5)])
    rows = np.concatenate([draw(np.random.default_rng(k)) for k in range(200)]).reshape(200, 8)
    assert (rows[:, :3] < 10).all() and (rows[:, 3:] >= 10).all() and (rows < 1000).all()
    assert len(np.unique(rows[:, :3])) == 10


def test_the_window_loads_the_selected_rows(shards):
    info = G.info('toy')
    F, A = info['num_features'], info['num_actions']
    paths = data.shard_paths(str(shards / 'data' / 'iter_0001'))
    full = data.Window(paths, F, A)
    sel = [np.flatnonzero(data.read_shard(p)['player'] == 1) for p in paths]
    W = data.Window(paths, F, A, select=sel)
    want = np.concatenate([full.starts[k] + s for k, s in enumerate(sel)])
    assert W.N == len(want) == W.loaded and (W.player == 1).all()
    assert np.array_equal(W.x, full.x[want]) and np.array_equal(W.z, full.z[want])
    assert np.array_equal(np.diff(W.opt_off), np.diff(full.opt_off)[want])
    small = data.Window(paths, F, A, budget_rows=W.N // 3, select=sel, seed=4)
    assert (small.player == 1).all() and abs(small.N - W.N // 3) <= len(paths) and small.loaded == W.N
    with pytest.raises(ValueError, match='sorted'):
        data.Window(paths[:1], F, A, select=[sel[0][::-1]])
    with pytest.raises(ValueError, match='no selected rows'):
        data.Window(paths[:1], F, A, select=[[]])


def test_each_teachers_answers_on_its_own_rows(shards):
    info = G.info('toy')
    W = data.Window(data.shard_paths(str(shards / 'data' / 'iter_0001')), info['num_features'], info['num_actions'])
    a, b = mlp(5), mlp(6)
    Ta, Tb = D.outputs(a, W), D.outputs(b, W)
    cut = W.N // 3
    T = D.outputs(a, W, span=(0, cut), batch=97)
    T = D.outputs(b, W, span=(cut, W.N), into=T, batch=50)
    pa = slice(0, int(W.opt_off[cut]))
    pb = slice(int(W.opt_off[cut]), None)
    assert np.allclose(T['pi'][pa], Ta['pi'][pa], atol=1e-6) and np.allclose(T['pi'][pb], Tb['pi'][pb], atol=1e-6)
    assert np.allclose(T['v'][:cut], Ta['v'][:cut], atol=1e-6) and np.allclose(T['v'][cut:], Tb['v'][cut:], atol=1e-6)
    assert np.allclose(T['m'][cut:], Tb['m'][cut:], atol=1e-5)


def test_the_row_selector_hook(shards, selector, monkeypatch):
    paths = data.shard_paths(str(shards / 'data' / 'iter_0001'))
    got = D.select_rows(paths, {'seat': 1}, 'toy', 'a')
    for p, s in zip(paths, got):
        assert np.array_equal(s, np.flatnonzero(data.read_shard(p)['player'] == 1))
    every = D.select_rows(paths, {}, 'toy', 'a')
    assert [len(s) for s in every] == [data.shard_rows(p) for p in paths]
    with pytest.raises(ValueError, match='unknown selector keys'):
        D.select_rows(paths, {'deck': 'x'}, 'toy', 'a')
    monkeypatch.setitem(package.active().row_selectors, 'toy', lambda s, p, c: np.ones(3, bool))
    with pytest.raises(ValueError, match='a bool array of its rows'):
        D.select_rows(paths, {'seat': 1}, 'toy', 'a')
    monkeypatch.delitem(package.active().row_selectors, 'toy')
    with pytest.raises(ValueError, match='has no row selector'):
        D.select_rows(paths, {'seat': 1}, 'toy', 'a')
    assert len(D.select_rows(paths, {}, 'toy', 'a')) == len(paths)       # (an empty selector needs no hook)


def test_the_selector_gets_the_shards_run_config(shards, monkeypatch, tmp_path):
    seen = []

    def sel(select, shard, ctx):
        seen.append((ctx['config'], ctx['config_path'], ctx['meta']['game'], ctx['source']))
        with np.load(shard) as z:
            return np.ones(len(z['player']), bool)
    monkeypatch.setitem(package.active().row_selectors, 'toy', sel)
    paths = data.shard_paths(str(shards / 'data' / 'iter_0002'))
    D.select_rows(paths, {'any': 1}, 'toy', 'src')
    assert seen[0] == (None, None, 'toy', 'src')                       # (no config.json beside the data dir)
    cfg = tmp_path / 'cfg.json'
    cfg.write_text(json.dumps({'spec': 'x'}))
    seen.clear()
    D.select_rows(paths, {'any': 1}, 'toy', 'src', config=str(cfg))
    assert seen[0][:2] == ({'spec': 'x'}, str(cfg))


# ---------------------------------------------------------------- the tool


def sources_file(root, tmp_path, **over):
    a, b, s = str(tmp_path / 'ta.pt'), str(tmp_path / 'tb.pt'), str(tmp_path / 'init.pt')
    net.save(mlp(5), a, {'iter': 11})
    net.save(mlp(6), b, {'iter': 12})
    net.save(mlp(9, 32, 2), s)
    d = str(root / 'data')
    spec = {'student': {'init': s},
            'sources': [{'name': 'early', 'teacher': a, 'data': [d], 'select': {'early': True}, 'weight': 3.0},
                        {'name': 'late', 'teacher': b, 'data': [d], 'select': {'early': False}, 'weight': 1.0}],
            'heldout': 'auto',
            'train': {'steps': 300, 'batch': 64, 'lr': 3e-3, 'warmup': 20, 'eval_every': 150, 'save_every': 100,
                      'report_every': 50, 'seed': 2}}
    spec.update(over)
    p = tmp_path / 'sources.json'
    p.write_text(json.dumps(spec))
    return str(p), spec


def test_two_teachers_each_on_its_own_rows(shards, selector, tmp_path):
    """Two teachers, each on its own rows (the first rounds' and the others'), weights 3:1, warm-started from a third
    network: the record and the log name each source's teacher, rows and held-out agreement; each source's KL to its
    own teacher falls, and the student ends closer to each source's teacher than to the other on its rows."""
    tm = tool()
    path, spec = sources_file(shards, tmp_path)
    out = str(tmp_path / 'stew.pt')
    assert tm.main(['--sources', path, out, '--torch-threads', '1', '--normal-priority']) == 0
    s, extra = net.load(out, game='toy')
    assert s.config['hidden'] == 32                                    # (the init's size)
    rec = extra['distill']
    assert extra['distilled_from'] == os.path.abspath(path)
    assert rec['sources_sha256'] == hashlib.sha256(open(path, 'rb').read()).hexdigest() and rec['step'] == 300
    assert [x['name'] for x in rec['sources']] == ['early', 'late']
    assert [x['batch_rows'] for x in rec['sources']] == [48, 16]
    held = data.shard_paths(str(shards / 'data' / 'iter_0002'))[-1]

    def early(p):
        return data.shard_features(p)[:, :6].argmax(1) < 3
    for x, flag, t in zip(rec['sources'], (True, False), ('ta.pt', 'tb.pt')):
        assert x['teacher_sha256'] == tm.sha256(str(tmp_path / t)) and x['teacher_iter'] == (11 if flag else 12)
        assert x['heldout_shards'] == [os.path.abspath(held)] and os.path.abspath(held) not in x['train_shards']
        rows = sum(int((early(p) == flag).sum()) for p in x['train_shards'])
        assert x['selected'] == rows == x['used'] and 0 < rows < x['rows']
        assert x['heldout_rows'] == int((early(held) == flag).sum())
    assert [h['step'] for h in rec['heldout']] == [0, 150, 300]
    for name in ('early', 'late'):
        kl = [h['sources'][name]['policy_kl'] for h in rec['heldout']]
        assert kl[-1] < kl[0] / 3, (name, kl)
    first, last = rec['heldout'][0], rec['heldout'][-1]
    assert last['policy_kl'] == pytest.approx(0.75 * last['sources']['early']['policy_kl']
                                              + 0.25 * last['sources']['late']['policy_kl'], abs=1e-6)
    assert last['policy_kl'] < first['policy_kl']
    log = [json.loads(line) for line in open(out + '.log.jsonl')]
    assert log[0]['event'] == 'start' and [x['weight'] for x in log[0]['sources']] == [3.0, 1.0]
    assert [x['event'] for x in log if x['event'] == 'heldout'] == ['heldout'] * 3
    assert all(set(x['sources']) == {'early', 'late'} for x in log if x['event'] == 'heldout')
    assert not os.path.exists(out + '.partial')
    info = G.info('toy')
    for flag, own, other in ((True, 5, 6), (False, 6, 5)):
        sub = data.Window([held], info['num_features'], info['num_actions'], select=[np.flatnonzero(early(held) == flag)])
        Ss = D.outputs(s, sub)
        assert D.agreement(Ss, D.outputs(mlp(own), sub), sub)['policy_kl'] < \
            D.agreement(Ss, D.outputs(mlp(other), sub), sub)['policy_kl']


def test_the_weights_set_each_sources_share(shards, selector, tmp_path):
    """Teachers that disagree on positions alike (the rows of each seat): the student fits each as its weight says -
    the heavier source ends closer to its teacher, either way round."""
    tm = tool()
    ends = {}
    for w0, w1 in ((3.0, 1.0), (1.0, 3.0)):
        path, spec = sources_file(shards, tmp_path, train={'steps': 150, 'batch': 64, 'lr': 3e-3, 'warmup': 20,
                                                           'eval_every': 0, 'save_every': 0, 'seed': 2})
        spec['sources'][0].update(name='seat0', select={'seat': 0}, weight=w0)
        spec['sources'][1].update(name='seat1', select={'seat': 1}, weight=w1)
        open(path, 'w').write(json.dumps(spec))
        out = str(tmp_path / f'w{w0:g}.pt')
        assert tm.main(['--sources', path, out, '--torch-threads', '1', '--normal-priority']) == 0
        h = net.load(out, game='toy')[1]['distill']['heldout'][-1]['sources']
        ends[w0] = (h['seat0']['policy_kl'], h['seat1']['policy_kl'])
    assert ends[3.0][0] < ends[3.0][1] and ends[1.0][0] > ends[1.0][1], ends


def test_a_fresh_student_explicit_heldout_and_resume(shards, selector, tmp_path):
    tm = tool()
    d1, d2 = str(shards / 'data' / 'iter_0001'), str(shards / 'data' / 'iter_0002')
    path, spec = sources_file(shards, tmp_path, student={'hidden': 16, 'blocks': 1})
    spec['sources'][0].update(data=[d1], heldout=[d2])
    spec['sources'][1].update(data=[d1, d2], heldout='none', select={})
    spec['train'].update(steps=200)
    open(path, 'w').write(json.dumps(spec))
    out = str(tmp_path / 'fresh.pt')
    base = ['--sources', path, out, '--torch-threads', '1', '--normal-priority']
    assert tm.main(base) == 0
    s, extra = net.load(out, game='toy')
    rec = extra['distill']
    assert s.config['hidden'] == 16 and rec['student_init'] is None
    assert rec['sources'][1]['heldout_shards'] == [] and rec['sources'][1]['selected'] == rec['sources'][1]['rows']
    assert set(rec['heldout'][-1]['sources']) == {'early'}
    # a partial checkpoint resumes at its step; one of another sources file is refused
    part = str(tmp_path / 'res.pt')
    torch.save({'config': s.config, 'state': s.state_dict(), 'opt': torch.optim.AdamW(s.parameters()).state_dict(),
                'step': 150, 'sources_sha256': rec['sources_sha256']}, part + '.partial')
    assert tm.main(['--sources', path, part, '--torch-threads', '1', '--normal-priority', '--resume']) == 0
    lines = [json.loads(line) for line in open(part + '.log.jsonl')]
    assert lines[0]['start'] == 150 and [x['step'] for x in lines if x['event'] == 'train'] == [200]
    torch.save({'config': s.config, 'state': s.state_dict(), 'opt': torch.optim.AdamW(s.parameters()).state_dict(),
                'step': 150, 'sources_sha256': 'f' * 64}, part + '.partial')
    with pytest.raises(SystemExit, match='another sources file'):
        tm.main(['--sources', path, part, '--torch-threads', '1', '--normal-priority', '--resume'])


def test_the_tool_is_loud(shards, selector, tmp_path, monkeypatch):
    tm = tool()
    path, spec = sources_file(shards, tmp_path)
    out = str(tmp_path / 'x.pt')
    run = ['--sources', path, out, '--torch-threads', '1', '--normal-priority']

    def write(s):
        open(path, 'w').write(json.dumps(s))
    with pytest.raises(SystemExit, match='refused flags: --steps'):
        tm.main(run + ['--steps', '5'])
    with pytest.raises(SystemExit, match='refused flags: --data'):
        tm.main(run + ['--data=x'])
    for bad, msg in ((dict(spec, extra=1), 'unknown keys'),
                     (dict(spec, train=dict(spec['train'], stepz=1)), 'train: unknown keys'),
                     (dict(spec, sources=[dict(spec['sources'][0], wieght=1)]), 'unknown keys'),
                     (dict(spec, sources=[spec['sources'][0], dict(spec['sources'][1], name='early')]), 'of its own'),
                     (dict(spec, sources=[dict(spec['sources'][0], weight=0)]), 'must be > 0'),
                     (dict(spec, student={'init': spec['student']['init'], 'hidden': 3}), '"init" alone'),
                     (dict(spec, sources=[dict(spec['sources'][0], select={'seat': 3})]), 'the toy has seats 0 and 1'),
                     (dict(spec, sources=[dict(spec['sources'][0], select={'deck': 'x'})]), 'unknown selector keys'),
                     (dict(spec, sources=[dict(spec['sources'][0], teacher='nope.pt')]), 'no such checkpoint')):
        write(bad)
        with pytest.raises(SystemExit, match=msg):
            tm.main(run)
    # a source whose selection is empty (its one shard with rows held out, or none at all)
    one = data.shard_paths(str(shards / 'data' / 'iter_0002'))[-1:]
    write(dict(spec, sources=[dict(spec['sources'][0], data=one)]))
    with pytest.raises(SystemExit, match='no training rows'):
        tm.main(run)
    monkeypatch.setitem(package.active().row_selectors, 'toy', lambda s, p, c: np.zeros(data.shard_rows(p), bool))
    write(spec)
    with pytest.raises(SystemExit, match='no training rows'):
        tm.main(run)
    monkeypatch.delitem(package.active().row_selectors, 'toy')
    with pytest.raises(SystemExit, match='has no row selector'):
        tm.main(run)
    assert not os.path.exists(out)
