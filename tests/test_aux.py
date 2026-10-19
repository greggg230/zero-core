"""Auxiliary training targets (docs/az.md 4.1): the toy's targets from each seat, recorded per
sample by the driver (aux=True), shards with and without them, the window's mask, the heads
(shapes, added to an older checkpoint with its outputs unchanged, stripped again), evaluators
taking states with / without heads, and training: weights 0 = exactly as without targets,
weights on = the losses, the heads added to a checkpoint without them, its optimizer state kept."""
import json
import os

import numpy as np
import pytest
import torch

from zero import _engine as E
from zero import games as G, protocol
from zero.data import ShardWriter, Window, read_shard, shard_paths
from zero.infer import NetEvaluator, UniformEvaluator, drive
from zero.net import GridNet, Net, add_aux, aux_segments, load, save, strip_aux, widen

INFO = E.game_info_of('toy')
F, A = INFO['num_features'], INFO['num_actions']
LAYOUT = aux_segments(INFO['aux_layout'])
AW = INFO['aux_width']
NS = 13                                          # toy: action spaces = the 'own' segment


def test_toy_layout_and_targets_from_each_seat():
    assert LAYOUT == [['own', NS], ['score', 24]] and AW == 37
    rng = np.random.default_rng(0)
    for n in (2, 3):
        g = E.Game('toy', players=n, seed=11 + n)
        while not g.over:
            leg = g.legal()
            g.step(int(leg[rng.integers(len(leg))]))
        t = [g.aux_targets(s) for s in range(n)]
        assert all(a.shape == (AW,) and a.dtype == np.float32 for a in t)
        sc = g.scores()
        for v in range(n):
            # every space is +1 for exactly the seat on it (or 0 for every seat)
            for j in range(n):
                tot = t[v][NS + 6 * j:NS + 6 * j + 6].sum() * 10
                assert abs(tot - sc[(v + j) % n]) < 1e-4
            assert np.all(t[v][NS + 6 * n:] == 0)
        own = np.stack([a[:NS] for a in t])
        assert np.all((own == 1).sum(0) <= 1) and np.all(own.sum(0)[(own == 1).any(0)] == 1 - (n - 1))
    with pytest.raises(ValueError):
        g.aux_targets(5)


def selfplay(d, prefix, aux, seed=4, games=16, rows=150):
    seat = G.nn_seat('m', 8, selfplay=True, sims_fast=4, p_full=0.5)
    kw = {'aux': True} if aux else {}
    drv = G.Driver('toy', [[seat, seat]] * games, concurrency=games, threads=2, seed=seed, record=True, **kw)
    w = ShardWriter(d, prefix, rows=rows, meta={'game': 'toy'})
    parts = []

    def grab(lb=None):
        s = drv.take_samples()
        parts.append(s)
        w.add(s)
    drive(drv, {'m': UniformEvaluator()}, on_batch=grab)
    grab()
    return w.close(), protocol.concat_samples(parts)


def test_driver_samples_shards_and_window_mask(tmp_path):
    d1, d0 = str(tmp_path / 'iter_0002'), str(tmp_path / 'iter_0001')
    st1, s1 = selfplay(d1, 'c00', aux=True)
    st0, s0 = selfplay(d0, 'c00', aux=False)
    protocol.validate_samples(s1, F)
    assert 'aux' not in s0 and s1['aux'].shape == (len(s1['player']), AW)
    # the same games either way: aux only adds the targets
    assert all(np.array_equal(s0[k], s1[k]) for k in protocol.SAMPLE_KEYS)
    # the 2p score blocks: the decider's first, the other seat's second (ownership = -own)
    a = s1['aux']
    assert np.allclose(a[:, NS:NS + 6].sum(1) - a[:, NS + 6:NS + 12].sum(1), s1['sc'][:, 0], atol=1e-5)
    sh = read_shard(shard_paths(d1)[0])
    assert sh['aux'].dtype == np.float16 and 'aux' not in read_shard(shard_paths(d0)[0])
    n1 = sh['aux'].shape[0]
    assert np.allclose(sh['aux'].astype(np.float32), a[:n1], atol=2e-3)
    paths = shard_paths(d1) + shard_paths(d0)
    W = Window(paths, F, A, aux_width=AW)
    N1, N0 = st1['samples'], st0['samples']
    assert W.N == N1 + N0 and W.aux_rows == N1 and W.aux.shape == (W.N, AW)
    assert W.aux_mask[:N1].all() and not W.aux_mask[N1:].any() and np.all(W.aux[N1:] == 0)
    b = W.batch(np.arange(W.N - 8, W.N))
    assert b['aux'].shape == (8, AW) and torch.all(b['aux_mask'] == 0)
    b = W.batch(np.arange(8))
    assert torch.all(b['aux_mask'] == 1) and torch.allclose(b['aux'], torch.from_numpy(a[:8]), atol=2e-3)
    # aux_width 0: the targets are ignored (the window as before); another width is an error
    W0 = Window(paths, F, A)
    assert W0.aux is None and 'aux' not in W0.batch(np.arange(4)) and np.array_equal(W0.x, W.x)
    with pytest.raises(ValueError):
        Window(paths, F, A, aux_width=AW + 1)


GRID = {'planes': 2, 'h': 4, 'w': 5, 'offset': 3, 'cell_action0': 2, 'hex': 'odd-r'}
GF, GA = 3 + 2 * 4 * 5 + 7, 2 + 20 + 4
GLAYOUT = [['own', 20], ['score', 6]]


def test_heads_shapes_and_layouts():
    torch.manual_seed(0)
    x, n = torch.randn(5, GF), torch.full((5,), 2)
    g = GridNet(GF, GA, GRID, ch=8, blocks=2, head=16, ghid=16, aux=GLAYOUT).eval()
    assert g.config['aux'] == GLAYOUT
    with torch.no_grad():
        out3, out4 = g(x, n), g(x, n, aux=True)
    assert len(out3) == 3 and len(out4) == 4 and out4[3].shape == (5, 26)
    assert all(torch.equal(u, v) for u, v in zip(out3, out4[:3]))
    assert torch.all(out4[3] == 0)                   # the heads' last layers start at zero
    # trained a little: the 'own' columns stay in [-1, 1] (tanh), each cell its own value
    with torch.no_grad():
        for p in (g.aux_own.weight, g.aux_own.bias, g.aux_score[2].weight):
            p.normal_()
        a = g(x, n, aux=True)[3]
    assert a[:, :20].abs().max() <= 1 and a[:, :20].std() > 0 and a[:, 20:].abs().max() > 0
    # a score-only layout; the MLP; bad layouts; no heads
    assert GridNet(GF, GA, GRID, ch=8, blocks=1, head=16, ghid=16, aux=[['score', 3]])(x, n, aux=True)[3].shape == (5, 3)
    m = Net(F, A, hidden=16, blocks=1, aux=LAYOUT).eval()
    with torch.no_grad():
        m.aux_head[2].weight.normal_(std=5.0)
        am = m(torch.randn(4, F), torch.full((4,), 2), aux=True)[3]
    assert am.shape == (4, AW) and am[:, :NS].abs().max() <= 1 and am[:, NS:].abs().max() > 1
    with pytest.raises(ValueError):
        GridNet(GF, GA, GRID, ch=8, blocks=1, aux=[['own', 19]])
    with pytest.raises(ValueError):
        Net(F, A, hidden=16, blocks=1)(torch.randn(2, F), torch.full((2,), 2), aux=True)


def test_add_aux_to_an_old_checkpoint_keeps_its_outputs(tmp_path):
    torch.manual_seed(1)
    old = GridNet(GF, GA, GRID, ch=8, blocks=2, head=16, ghid=16)
    old.train()
    with torch.no_grad():                             # non-trivial BatchNorm statistics
        for _ in range(3):
            old(torch.randn(32, GF), torch.full((32,), 2))
    old.eval()
    p = str(tmp_path / 'old.pt')
    save(old, p, {'iter': 7})
    m, _ = load(p)
    assert 'aux' not in m.config
    rng0 = torch.get_rng_state()
    a = add_aux(m, GLAYOUT, seed=3)
    assert torch.equal(torch.get_rng_state(), rng0)   # the global RNG is untouched
    assert a.config == dict(m.config, aux=GLAYOUT) and add_aux(a, GLAYOUT) is a and not a.training
    assert all(torch.equal(add_aux(m, GLAYOUT, seed=3).state_dict()[k], v) for k, v in a.state_dict().items())
    x, n = torch.randn(6, GF), torch.tensor([2, 3, 2, 2, 4, 2])
    with torch.no_grad():
        o, w = m(x, n), a(x, n, aux=True)
    assert all(torch.equal(u, v) for u, v in zip(o, w[:3])) and torch.all(w[3] == 0)
    with pytest.raises(ValueError):
        add_aux(a, [['score', 6]])
    # saved with heads, loaded back; stripped = the old network again
    pa = str(tmp_path / 'aux.pt')
    save(a, pa, {'iter': 8})
    b, _ = load(pa)
    assert b.config['aux'] == GLAYOUT and any(k.startswith('aux_') for k in b.state_dict())
    s = strip_aux(b)
    assert s.config == m.config and set(s.state_dict()) == set(m.state_dict())
    with torch.no_grad():
        assert all(torch.equal(u, v) for u, v in zip(s(x, n), o))
    # widen keeps the heads
    wd = widen(b, GF + 4)
    with torch.no_grad():
        ww = wd(torch.cat([x, torch.randn(6, 4)], 1), n, aux=True)
    assert wd.config['aux'] == GLAYOUT and all(torch.allclose(u, v, atol=1e-5) for u, v in zip(ww, b(x, n, aux=True)))


def test_evaluators_take_states_with_or_without_heads():
    torch.manual_seed(2)
    plain = Net(F, A, hidden=16, blocks=1).eval()
    withh = add_aux(plain, LAYOUT, seed=1)
    with torch.no_grad():
        withh.aux_head[2].weight.normal_()
    drv = G.Driver('toy', [[G.nn_seat('m', 8), {'bot': 'random'}]] * 6, concurrency=6, threads=1, seed=0)
    lb = drv.next_batch()
    ev = NetEvaluator(Net(F, A, hidden=16, blocks=1).eval())
    ev.set_weights(withh.state_dict())                # a newer checkpoint with heads
    ref = NetEvaluator(plain)(lb)
    assert all(np.array_equal(u, v) for u, v in zip(ev(lb), ref))
    ev2 = NetEvaluator(add_aux(Net(F, A, hidden=16, blocks=1).eval(), LAYOUT))
    ev2.set_weights(plain.state_dict())               # heads kept, the rest replaced
    assert all(np.array_equal(u, v) for u, v in zip(ev2(lb), ref))
    bad = {k: v for k, v in plain.state_dict().items() if not k.startswith('val.')}
    with pytest.raises(RuntimeError):
        ev.set_weights(bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graphs need CUDA')
def test_graph_runner_with_heads_matches_without():
    torch.manual_seed(3)
    m = GridNet(GF, GA, GRID, ch=16, blocks=2, head=16, ghid=16).eval()
    a = add_aux(m, GLAYOUT)
    rng = np.random.default_rng(4)
    B = 70
    x = rng.normal(size=(B, GF)).astype(np.float32)
    npl = np.full(B, 2, np.int32)
    cnt = rng.integers(2, 6, size=B)
    off = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int32)
    opt = np.concatenate([np.sort(rng.choice(GA, size=c, replace=False)) for c in cnt]).astype(np.int32)
    r0 = NetEvaluator(m, 'cuda', graphs=True).evaluate(x, npl, off, opt)
    r1 = NetEvaluator(a, 'cuda', graphs=True).evaluate(x, npl, off, opt)
    assert all(np.array_equal(u, v) for u, v in zip(r0, r1))


# ---------------------------------------------------------------- training
def make_run(run, **over):
    from zero.run import DEFAULTS, PRESETS, Run
    cfg = dict(DEFAULTS)
    cfg.update(PRESETS['toy-smoke'])
    cfg.update(min_steps=30, batch=64, hidden=32, blocks=1, **over)
    os.makedirs(os.path.join(run, 'models'), exist_ok=True)
    return Run(run, cfg)


def write_data(run, it, aux, seed):
    return selfplay(os.path.join(run, 'data', f'iter_{it:04d}'), 'c00', aux=aux, seed=seed, games=24, rows=10_000)[0]


def test_train_with_aux_weights_0_is_unchanged(tmp_path):
    """Weights 0: the shards' aux targets are ignored, nothing else changes: the same checkpoint,
    optimizer state and stats as training on shards without them."""
    from zero.run import model_path
    outs = []
    for k, aux in enumerate((False, True)):
        run = str(tmp_path / f'r{k}')
        R = make_run(run)
        assert R.aux == []
        save(R.new_model(), model_path(run, 0), {'iter': 0})
        st = write_data(run, 1, aux, seed=5)
        tr = R.train(1, st['samples'])
        assert 'aux' not in tr and not any(k.startswith('aux_') for k in tr)
        st2 = write_data(run, 2, aux, seed=6)
        tr2 = R.train(2, st2['samples'])
        outs.append((torch.load(model_path(run, 2), weights_only=False), tr2,
                     torch.load(os.path.join(run, 'models', 'opt_latest.pt'), weights_only=False)))
    (a, ta, oa), (b, tb, ob) = outs
    assert a['config'] == b['config'] and 'aux' not in a['config']
    assert all(torch.equal(a['state'][k], b['state'][k]) for k in a['state'])
    drop = ('seconds', 'samples_per_s', 'load_s')
    assert {k: v for k, v in ta.items() if k not in drop} == {k: v for k, v in tb.items() if k not in drop}
    sa, sb = oa['state']['state'], ob['state']['state']
    assert sa.keys() == sb.keys() and all(torch.equal(sa[i]['exp_avg'], sb[i]['exp_avg']) for i in sa)


def test_train_with_aux_adds_heads_and_learns(tmp_path):
    """Weights on, a checkpoint without heads and a window whose older shards have no targets
    (a fork): the heads are added, the optimizer state of the trunk is kept, the masked losses
    are logged; the next iteration continues with the heads (two parameter groups)."""
    from zero.run import model_path
    run = str(tmp_path / 'a')
    R0 = make_run(run)
    save(R0.new_model(), model_path(run, 0), {'iter': 0})
    R0.train(1, write_data(run, 1, False, seed=5)['samples'])        # iteration 1: no aux (the parent run)
    R = make_run(run, aux_own_w=1.0, aux_score_w=1.0)
    assert R.aux == LAYOUT
    st = write_data(run, 2, True, seed=6)
    tr = R.train(2, st['samples'])
    log = open(os.path.join(run, 'log.txt'), encoding='utf-8').read()
    assert 'auxiliary heads' in log and 'optimizer state not loaded' not in log
    assert tr['aux']['heads_added'] and 0 < tr['aux']['share'] < 1 and tr['aux']['rows'] == st['samples']
    assert 0 < tr['aux_own'] and 0 < tr['aux_score'] < tr['aux']['score_zero']
    ck = torch.load(model_path(run, 2), weights_only=False)
    assert ck['config']['aux'] == LAYOUT
    o = torch.load(os.path.join(run, 'models', 'opt_latest.pt'), weights_only=False)
    assert len(o['state']['param_groups']) == 2
    # the trunk's optimizer state continued (step counts beyond this iteration's steps)
    steps = [v['step'] for v in o['state']['state'].values()]
    assert max(float(s) for s in steps) > tr['steps'] and min(float(s) for s in steps) == tr['steps']
    tr3 = R.train(3, write_data(run, 3, True, seed=7)['samples'])
    assert 'heads_added' not in tr3['aux'] and tr3['aux']['share'] > tr['aux']['share']
    m, _ = load(model_path(run, 3))
    assert m.config['aux'] == LAYOUT
    # only the weights that are on: own 0 still logs its loss, but its head gets no gradient
    R2 = make_run(str(tmp_path / 'b'), aux_score_w=0.5)
    assert R2.aux == LAYOUT and R2.aux_w == {'own': 0.0, 'score': 0.5}


def test_aux_settings_are_checked(tmp_path):
    from zero.run import resolve_config, parse
    with pytest.raises(SystemExit):
        resolve_config(parse(['--run', 'x', '--aux-own-w', '-1']), {})
    cfg = resolve_config(parse(['--run', 'x']), {})
    assert cfg['aux_own_w'] == 0.0 and cfg['aux_score_w'] == 0.0
    assert json.loads(json.dumps(cfg))['aux_own_w'] == 0.0
