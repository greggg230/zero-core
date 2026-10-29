"""Auxiliary training targets (docs/az.md 15): the game's per-decision labels from the true state
(the toy's: the next round's hidden card), recorded by a driver created with aux=True, shards and
windows that carry them (old shards masked), the networks' auxiliary head (built last, add_aux /
strip_aux keep everything else exactly), the losses, the inference API, the run loop on and off,
forking a run without the head, and replays."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from zero import auxtarget as AUX, games as G, package
from zero.data import Window, WindowCache, read_shard, shard_paths, write_shard
from zero.infer import NetEvaluator
from zero.net import (EncodingMismatch, GridNet, Net, TokenNet, add_aux, aux_layout, aux_predict, build,
                      check_checkpoint, load, make_net, save, strip_aux)
from zero.run import DEFAULTS, Run, latest_iter, read_metrics, resolve_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYPATH = os.pathsep.join([os.path.join(ROOT, 'python')] + [p for p in [os.environ.get('PYTHONPATH')] if p])
SMOKE = ['--preset', 'toy-smoke', '--conc', '24', '--games', '24', '--eval-games', '8', '--eval-random-games', '4',
         '--elo-games', '4', '--ref-games', '8']
TOY_AUX = [['next_card', 6, 'multinomial']]


def run_loop(run, *extra, timeout=300):
    e = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH)
    return subprocess.run([sys.executable, '-m', 'zero.run', '--run', str(run), *SMOKE, *extra], cwd=ROOT,
                          capture_output=True, text=True, timeout=timeout, env=e)


def toy_x(n=16, seed=0):
    """Features of real toy positions (random play)."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        g = G.engine().Game('toy', '{}', 2, int(rng.integers(1 << 30)))
        for _ in range(int(rng.integers(0, 40))):
            if g.over:
                break
            legal = g.legal()
            g.step(legal[int(rng.integers(len(legal)))])
        if not g.over:
            out.append(np.asarray(g.encode(g.to_move), np.float32))
    return np.stack(out)


def nets(aux=None):
    info = G.info('toy')
    torch.manual_seed(3)
    m = make_net(info, 'toy', 'mlp', 32, 2, aux=aux)
    torch.manual_seed(3)
    t = make_net(info, 'toy', 'tokens', 32, 1, heads=2, aux=aux)
    grid = {'planes': 7, 'h': 1, 'w': 13, 'offset': 31, 'segments': [[31, 7]], 'cell_action0': 0, 'hex': None}
    torch.manual_seed(3)
    g = GridNet(info['num_features'], info['num_actions'], grid, ch=8, blocks=1, game='toy', aux=aux).eval()
    return {'mlp': m, 'tokens': t, 'grid': g}


# ----------------------------------------------------------------------------- the package, settings

def test_package_declares_the_toy_target_and_its_setting():
    pkg = package.active()
    assert pkg.aux_of('toy') == TOY_AUX and pkg.aux_of('toy_team') == []
    assert pkg.aux_settings() == {'aux_next_card_w': 0.0}
    assert DEFAULTS['aux_next_card_w'] == 0.0
    info = G.info('toy')
    assert [tuple(x) for x in info['aux_layout']] == [('next_card', 6)] and info['aux_width'] == 6
    assert G.info('toy_team')['aux_layout'] == [] and G.info('toy_team')['aux_width'] == 0
    with pytest.raises(ValueError):
        aux_layout([['a', 3, 'softmax']])
    with pytest.raises(ValueError):
        aux_layout([['a', 3, 'mse'], ['a', 2, 'mse']])


def test_settings_off_by_default_and_checked():
    import argparse
    args = argparse.Namespace(**{k: None for k in DEFAULTS})
    cfg = resolve_config(args, {'game': 'toy'})
    assert cfg['aux_next_card_w'] == 0.0
    r = Run('.', cfg)
    assert r.aux == [] and r.aux_w == {'next_card': 0.0}
    r = Run('.', dict(cfg, aux_next_card_w=0.5))
    assert r.aux == TOY_AUX and r.aux_w == {'next_card': 0.5}
    with pytest.raises(SystemExit):
        resolve_config(args, {'game': 'toy', 'aux_next_card_w': -1})


# ----------------------------------------------------------------------------- the engine and driver

def test_game_aux_target_is_the_true_hidden_card_never_an_input():
    rng = np.random.default_rng(1)
    differ = 0
    for s in range(40):
        g = G.engine().Game('toy', '{}', 2, s)
        for _ in range(int(rng.integers(0, 30))):
            if g.over:
                break
            legal = g.legal()
            g.step(legal[int(rng.integers(len(legal)))])
        if g.over:
            continue
        me = g.to_move
        t, x = np.asarray(g.aux_target(me)), np.asarray(g.encode(me))
        assert t.shape == (6,) and t.sum() in (0.0, 1.0)
        for k in range(3):
            d = g.clone()
            d.determinize(me, 100 * s + k)
            assert np.array_equal(np.asarray(d.encode(me)), x)
            differ += not np.array_equal(np.asarray(d.aux_target(me)), t)
    assert differ > 5
    with pytest.raises(ValueError):
        G.engine().Game('toy', '{}', 2, 0).aux_target(5)


def selfplay(aux, games=12, seed=4):
    seat = G.nn_seat('m', 8, selfplay=True, sims_fast=2, p_full=0.5, temp_moves=4)
    kw = {'aux': True} if aux else {}
    drv = G.Driver('toy', [[seat, seat]] * games, concurrency=games, seed=seed, record=True, threads=2, **kw)
    from zero.infer import UniformEvaluator, drive
    drive(drv, {'m': UniformEvaluator()})
    return drv.take_samples()


def test_driver_records_the_targets_and_nothing_else_changes():
    s0, s1 = selfplay(False), selfplay(True)
    assert 'aux' not in s0 and s1['aux'].shape == (len(s1['player']), 6) and s1['aux'].dtype == np.float32
    for k in s0:
        assert np.array_equal(s0[k], s1[k]), k
    tot = s1['aux'].sum(1)
    assert set(np.unique(tot)) <= {0.0, 1.0} and (tot == 1).mean() > 0.6
    with pytest.raises(ValueError):
        G.Driver('toy_team', [[G.seat_dict('random')] * 4], record=True, aux=True)


# ----------------------------------------------------------------------------- shards and windows

def test_shards_and_window_masks_old_shards(tmp_path):
    s0, s1 = selfplay(False, seed=5), selfplay(True, seed=6)
    write_shard(str(tmp_path / 'a_000.npz'), s0, {'game': 'toy'})
    write_shard(str(tmp_path / 'b_000.npz'), s1, {'game': 'toy'})
    a, b = read_shard(str(tmp_path / 'a_000.npz')), read_shard(str(tmp_path / 'b_000.npz'))
    assert 'aux' not in a and b['aux'].dtype == np.float16 and np.array_equal(b['aux'].astype(np.float32), s1['aux'])
    paths = shard_paths(str(tmp_path))
    info = G.info('toy')
    W0 = Window(paths, info['num_features'], info['num_actions'])
    W = Window(paths, info['num_features'], info['num_actions'], aux_width=6)
    assert W0.aux is None and W.N == W0.N == len(s0['player']) + len(s1['player'])
    assert np.array_equal(W.aux_mask, np.r_[np.zeros(len(s0['player']), bool), np.ones(len(s1['player']), bool)])
    assert W.aux_rows == len(s1['player']) and not W.aux[:len(s0['player'])].any()
    idx = np.arange(W.N)
    bb, b0 = W.batch(idx), W0.batch(idx)
    for k in b0:
        assert torch.equal(bb[k], b0[k]), k
    assert torch.equal(bb['aux_mask'], torch.from_numpy(W.aux_mask.astype(np.float32)))
    # the cache: a part loaded without targets is not reused for a window that wants them
    cache = WindowCache()
    Window(paths, info['num_features'], info['num_actions'], cache=cache)
    Wc = Window(paths, info['num_features'], info['num_actions'], cache=cache, aux_width=6)
    assert Wc.reused == 0 and np.array_equal(Wc.aux, W.aux)
    assert Window(paths, info['num_features'], info['num_actions'], cache=cache, aux_width=6).reused == W.N
    with pytest.raises(ValueError):
        Window(paths, info['num_features'], info['num_actions'], aux_width=5)


# ----------------------------------------------------------------------------- the networks

@pytest.mark.parametrize('arch', ['mlp', 'tokens', 'grid'])
def test_head_is_built_last_and_changes_no_other_output(arch):
    x = torch.from_numpy(toy_x())
    n = torch.full((len(x),), 2)
    plain, withh = nets()[arch], nets(TOY_AUX)[arch]
    assert 'aux' not in plain.config and withh.config['aux'] == TOY_AUX
    sd0, sd1 = plain.state_dict(), withh.state_dict()
    assert set(sd1) - set(sd0) and all(k.startswith('aux_') for k in set(sd1) - set(sd0))
    for k in sd0:
        assert torch.equal(sd0[k], sd1[k]), k
    with torch.no_grad():
        a, b = plain(x, n), withh(x, n)
        c = withh(x, n, aux=True)
    assert len(b) == 3 and len(c) == 4 and c[3].shape == (len(x), 6)
    for u, v, w in zip(a, b, c):
        assert torch.equal(u, v) and torch.equal(u, w)
    assert torch.equal(c[3], torch.zeros_like(c[3]))          # the last layer starts at zero
    with pytest.raises(ValueError):
        plain(x, n, aux=True)


@pytest.mark.parametrize('arch', ['mlp', 'tokens', 'grid'])
def test_add_aux_keeps_everything_else_exactly(arch):
    x = torch.from_numpy(toy_x(seed=2))
    n = torch.full((len(x),), 2)
    m = nets()[arch]
    with torch.no_grad():                                      # (non-zero heads: a trained network)
        for p in m.parameters():
            p.add_(0.01 * torch.randn_like(p))
    torch.manual_seed(123)
    r0 = torch.rand(3)
    torch.manual_seed(123)
    h = add_aux(m, TOY_AUX, seed=9)
    assert torch.equal(torch.rand(3), r0), 'add_aux leaves the global torch RNG alone'
    assert h.config['aux'] == TOY_AUX and add_aux(h, TOY_AUX) is h
    for k, v in m.state_dict().items():
        assert torch.equal(v, h.state_dict()[k]), k
    with torch.no_grad():
        for u, v in zip(m(x, n), h(x, n)):
            assert torch.equal(u, v)
    assert torch.equal(add_aux(m, TOY_AUX, seed=9).state_dict()['aux_head.0.weight'], h.state_dict()['aux_head.0.weight'])
    with pytest.raises(ValueError):
        add_aux(h, [['other', 3, 'mse']])
    s = strip_aux(h)
    assert 'aux' not in s.config and set(s.state_dict()) == set(m.state_dict())
    with torch.no_grad():
        for u, v in zip(m(x, n), s(x, n)):
            assert torch.equal(u, v)


def test_checkpoint_with_head_loads_and_passes_the_encoding_guard(tmp_path):
    t = nets(TOY_AUX)['tokens']
    p = str(tmp_path / 'h.pt')
    save(t, p, {'iter': 0})
    check_checkpoint(p)                                         # the head changes no input encoding
    m, _ = load(p)
    assert m.config['aux'] == TOY_AUX
    for k, v in t.state_dict().items():
        assert torch.equal(v, m.state_dict()[k])
    bad = dict(t.config, features=t.config['features'] + 1)
    with pytest.raises(EncodingMismatch):
        from zero.net import check_encoding
        check_encoding(bad, 'toy')


def test_predictions_and_losses():
    raw = torch.tensor([[0.0, 1.0, 2.0, 0.5, 0.0, 0.0]])
    pr = aux_predict(TOY_AUX, raw)
    assert torch.allclose(pr['next_card'].sum(1), torch.ones(1)) and pr['next_card'].argmax().item() == 2
    lay = [['c', 3, 'multinomial'], ['r', 2, 'poisson'], ['v', 1, 'mse']]
    raw = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.5], [3.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]])
    t = torch.tensor([[2.0, 0.0, 1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 2.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([1.0, 1.0, 0.0])
    L = AUX.losses(lay, raw, t, mask)
    # multinomial: only row 0 has a target (row 1 sums to 0, row 2 is masked); uniform logits
    assert abs(L['c']['loss'].item() - np.log(3)) < 1e-6 and L['c']['rows'].item() == 1
    assert L['c']['hit'].item() == 1.0                          # (argmax of a tie: column 0, in the hand)
    want = np.mean([np.exp(0) - 0, np.exp(1) - 0]) + np.mean([np.exp(0) - 2 * 0, np.exp(0) - 0])
    assert abs(L['r']['loss'].item() - want / 2) < 1e-6
    assert abs(L['v']['loss'].item() - ((0.5 - 1) ** 2 + 0.0) / 2) < 1e-6


def test_evaluator_aux_api_and_weights_either_way():
    x = toy_x(seed=3)
    plain, withh = nets()['tokens'], nets(TOY_AUX)['tokens']
    with torch.no_grad():
        for p in withh.aux_head.parameters():
            p.add_(0.1 * torch.randn_like(p))
    ev0, ev1 = NetEvaluator(plain), NetEvaluator(withh)
    assert ev0.aux(x, np.full(len(x), 2)) == {} and ev0.aux_layout == []
    a = ev1.aux(x, np.full(len(x), 2))
    assert set(a) == {'next_card'} and a['next_card'].shape == (len(x), 6)
    assert np.allclose(a['next_card'].sum(1), 1.0, atol=1e-5)
    # the search's evaluations are those of the network without the head
    from zero.games import engine  # noqa: F401
    off = np.arange(len(x) + 1, dtype=np.int32) * 2
    acts = np.tile(np.array([0, 3], np.int32), len(x))
    r0, r1 = ev0.evaluate(x, np.full(len(x), 2), off, acts), ev1.evaluate(x, np.full(len(x), 2), off, acts)
    for u, v in zip(r0, r1):
        assert np.array_equal(u, v)
    # set_weights: a state with the head into a network without, and the reverse
    ev0.set_weights(withh.state_dict())
    ev1.set_weights(plain.state_dict())
    assert ev1.aux_layout == TOY_AUX
    bad = dict(plain.state_dict())
    bad.pop('g.weight')
    with pytest.raises(RuntimeError):
        ev0.set_weights(bad)


# ----------------------------------------------------------------------------- the run loop

@pytest.fixture(scope='module')
def aux_run(tmp_path_factory):
    run = tmp_path_factory.mktemp('aux') / 'on'
    p = run_loop(run, '--iters', '2', '--arch', 'tokens', '--aux-next-card-w', '1.0')
    assert p.returncode == 0, p.stdout + p.stderr
    return run


def test_run_with_the_target_on(aux_run):
    m = read_metrics(str(aux_run))
    tr = m[2]['train']
    assert tr['aux']['rows'] == tr['window_samples'] and tr['aux']['weights'] == {'next_card': 1.0}
    assert 'head_added' not in tr['aux'] and 0 < tr['aux_next_card'] < np.log(6)
    assert 0 <= tr['aux_next_card_hit'] <= 1
    for it in (1, 2):
        for p in shard_paths(str(aux_run / 'data' / f'iter_{it:04d}')):
            assert read_shard(p)['aux'].shape[1] == 6
    for it in (0, 1, 2):
        ck = torch.load(str(aux_run / 'models' / f'iter_{it:04d}.pt'), weights_only=False)
        assert ck['config']['aux'] == TOY_AUX                    # a new run's first network has the head
    o = torch.load(str(aux_run / 'models' / 'opt_latest.pt'), weights_only=False)
    assert len(o['state']['param_groups']) == 2


@pytest.fixture(scope='module')
def plain_run(tmp_path_factory):
    run = tmp_path_factory.mktemp('aux') / 'off'
    p = run_loop(run, '--iters', '2', '--arch', 'tokens')
    assert p.returncode == 0, p.stdout + p.stderr
    return run


def test_run_off_records_and_builds_nothing(plain_run):
    m = read_metrics(str(plain_run))
    assert 'aux' not in m[2]['train'] and not any(k.startswith('aux_') for k in m[2]['train'])
    for p in shard_paths(str(plain_run / 'data' / 'iter_0002')):
        assert 'aux' not in read_shard(p)
    assert 'aux' not in torch.load(str(plain_run / 'models' / 'iter_0002.pt'), weights_only=False)['config']
    assert json.load(open(plain_run / 'config.json'))['aux_next_card_w'] == 0.0


def test_fork_without_the_head_adds_it_and_keeps_the_trunk_state(plain_run, tmp_path):
    """The carry-over (docs/az.md 15.4): a continuation of a run without the head, the target on
    (`fork_run --set aux_next_card_w=1`): its first training adds a fresh head to checkpoint K (the
    rest exactly K's), keeps the optimizer's moments for the trunk, trains on the window whose older
    shards have no targets (masked), and the network of K + 1 has the head."""
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import fork_run
    c = tmp_path / 'fork'
    assert fork_run.main(['--source', str(plain_run), '--dest', str(c), '--iter', '2', '--set', 'aux_next_card_w=1.0']) == 0
    assert json.load(open(c / 'config.json'))['aux_next_card_w'] == 1.0
    p = run_loop(c, '--iters', '3')
    assert p.returncode == 0, p.stdout + p.stderr
    m = read_metrics(str(c))
    tr = m[3]['train']
    assert tr['aux']['head_added'] is True and 0 < tr['aux']['share'] < 1      # iters 1-2: no targets
    assert 'optimizer state not loaded' not in open(c / 'log.txt').read()
    k2, _ = load(str(c / 'models' / 'iter_0002.pt'))
    k3, _ = load(str(c / 'models' / 'iter_0003.pt'))
    assert 'aux' not in k2.config and k3.config['aux'] == TOY_AUX
    o = torch.load(str(c / 'models' / 'opt_latest.pt'), weights_only=False)
    assert [len(g['params']) for g in o['state']['param_groups']] == [len(list(k2.parameters())), 4]
    # the rolling / shared evaluators take the new state into a network without the head
    ev = NetEvaluator(k2)
    ev.set_weights(k3.state_dict())


def test_replays_show_the_head(aux_run, plain_run):
    from zero import replay
    a = str(aux_run / 'models' / 'iter_0002.pt')
    reps = replay.record(a, opp='random', games=2, sims=8, chance_k=0, seed=1, log=lambda *x: None)
    nn = [f for r in reps for f in r['frames'] if f['k'] == 'nn']
    assert nn and all('aux' in f and len(f['aux']['next_card']) == 6 for f in nn)
    assert abs(sum(v for _, v in nn[0]['aux']['next_card']) - 1.0) < 1e-3
    seat = [s for s in reps[0]['seats'] if s['kind'] == 'nn'][0]
    assert seat['aux'] == ['next_card']
    b = str(plain_run / 'models' / 'iter_0002.pt')
    reps = replay.record(b, opp='random', games=1, sims=8, chance_k=0, seed=1, log=lambda *x: None)
    assert not any('aux' in f for r in reps for f in r['frames'])
    assert 'aux' not in reps[0]['seats'][0]
