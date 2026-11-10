"""Widening an auxiliary head (docs/az.md 37): a network whose head predicts some of the game's targets gains
the others - its head's last layer kept, the new targets' outputs from a layer of their own on the same hidden
layer, zero - so every output of before stays bit for bit. The network part (`widen_aux`, the refusals, the
readers, `load` / `save`), the training window (shards of the older layout placed, a mask per target), the
optimizer's groups, the evaluators; the run part: a run of the toy with a second target (tests/toy_aux2.py's
`next_half`) forked from a run with one, widened at its first training and learning the new target."""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from zero import auxtarget as AUX, games as G, package
from zero.data import ShardWriter, Window, WindowCache
from zero.net import (GridNet, add_aux, add_reader, aux_predict, aux_widen_problem, build, game_aux, load, make_net,
                      reader_layout, reader_predict, save, strip_aux, widen, widen_aux)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tests'))
import toy_aux2  # noqa: E402

PYPATH = os.pathsep.join([os.path.join(ROOT, 'python')] + [p for p in [os.environ.get('PYTHONPATH')] if p])
CARD = ['next_card', 6, 'multinomial']
HALF = ['next_half', 2, 'multinomial']
RATE = ['rate', 3, 'poisson']
LEVEL = ['level', 1, 'mse']
TOY_AUX = [CARD]
GRID = {'planes': 7, 'h': 1, 'w': 13, 'offset': 31, 'segments': [[31, 7]], 'cell_action0': 0, 'hex': None}
TOY_READER = [{'target': 'next_card', 'width': 6, 'loss': 'multinomial', 'inputs': [[0, 6], [25, 6]], 'hidden': 32,
               'enter': 'trunk', 'belief': False}]


def toy_x(n=64, seed=0):
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
    return torch.from_numpy(np.stack(out))


def nets(aux=None, readers=None):
    info = G.info('toy')
    torch.manual_seed(3)
    m = make_net(info, 'toy', 'mlp', 32, 2, aux=aux, readers=readers)
    torch.manual_seed(3)
    t = make_net(info, 'toy', 'tokens', 32, 1, heads=2, aux=aux, readers=readers)
    torch.manual_seed(3)
    g = GridNet(info['num_features'], info['num_actions'], GRID, ch=8, blocks=1, game='toy', aux=aux,
                readers=readers).eval()
    return {'mlp': m, 'tokens': t, 'grid': g}


def shake(m, scale=0.05, prefix=''):
    """Non-zero everything (a trained network), or the parameters under `prefix`."""
    with torch.no_grad():
        for k, p in m.named_parameters():
            if k.startswith(prefix):
                p.add_(scale * torch.randn_like(p))


def cols(layout, name):
    """The columns of target `name` in a layout's raw outputs."""
    return {n: sl for n, _, sl in AUX.columns(layout)}[name]


# ----------------------------------------------------------------------------- the network

def test_refusals_stay_loud():
    m = nets(TOY_AUX)['mlp']
    for want, why in (([HALF], 'not among'),                                   # removed
                      ([['card', 6, 'multinomial'], HALF], 'not among'),        # renamed
                      ([['next_card', 5, 'multinomial'], HALF], 'wide'),        # width changed
                      ([['next_card', 6, 'poisson'], HALF], 'wide'),            # loss changed
                      ([CARD], None)):                                          # the same: nothing to do
        if why is None:
            assert aux_widen_problem(TOY_AUX, want) == 'the same targets in another order ([\'next_card\'] -> ' \
                                                       '[\'next_card\'])' and widen_aux(m, want) is m
            continue
        assert why in aux_widen_problem(TOY_AUX, want)
        with pytest.raises(ValueError, match=why):
            widen_aux(m, want)
    two = widen_aux(m, [CARD, HALF])
    with pytest.raises(ValueError, match='another order'):                     # a pure reorder
        widen_aux(two, [HALF, CARD])
    with pytest.raises(ValueError, match='no auxiliary head'):
        widen_aux(nets()['mlp'], [CARD, HALF])
    with pytest.raises(ValueError, match='widen_aux'):                         # add_aux points to it
        add_aux(m, [CARD, HALF])


@pytest.mark.parametrize('arch', ['mlp', 'tokens', 'grid'])
def test_widen_keeps_every_output_bit_for_bit(arch):
    """The toy's head (next_card) widened by targets after it, before it, and twice: every parameter of before
    exactly, the policy / value / margin and each old target's raw outputs bit for bit on real positions (all
    rows and the belief rows' rows), the new ones zero (uniform / unit / zero predictions), the RNG untouched,
    the config rebuilding the same network, save / load keeping it and the layouts of before."""
    x, n = toy_x(seed=1), torch.full((64,), 2)
    m = nets(TOY_AUX)[arch]
    shake(m)
    with torch.no_grad():
        ref = m(x, n, aux=True)
        ref_rows = m(x, n, aux=True, aux_rows=torch.tensor([5, 0, 9]))[3]
    for lay in ([CARD, HALF], [HALF, CARD], [RATE, CARD, LEVEL]):
        torch.manual_seed(77)
        r0 = torch.rand(3)
        torch.manual_seed(77)
        w = widen_aux(m, lay)
        assert torch.equal(torch.rand(3), r0), 'widen_aux leaves the global torch RNG alone'
        assert w.config['aux'] == lay and w.config['aux_parts'] == [['next_card'], [t[0] for t in lay if t != CARD]]
        assert w.aux_older == [TOY_AUX] and w.aux_widened == TOY_AUX
        sd, wd = m.state_dict(), w.state_dict()
        assert set(wd) - set(sd) == {'aux_more.0.weight', 'aux_more.0.bias'} and not set(sd) - set(wd)
        assert all(torch.equal(v, wd[k]) for k, v in sd.items())
        assert not wd['aux_more.0.weight'].any() and not wd['aux_more.0.bias'].any()
        with torch.no_grad():
            out = w(x, n, aux=True)
            rows = w(x, n, aux=True, aux_rows=torch.tensor([5, 0, 9]))[3]
        for u, v in zip(ref[:3], out[:3]):
            assert torch.equal(u, v)
        c = cols(lay, 'next_card')
        assert torch.equal(out[3][:, c], ref[3]) and torch.equal(rows[:, c], ref_rows)
        assert out[3].shape[1] == sum(t[1] for t in lay)
        others = [i for i in range(out[3].shape[1]) if i not in range(c.start, c.stop)]
        assert not out[3][:, others].any()
        pred = aux_predict(lay, out[3])
        assert torch.equal(pred['next_card'], aux_predict(TOY_AUX, ref[3])['next_card'])
        for name, width, loss in lay:
            if name != 'next_card':
                want = {'multinomial': 1.0 / width, 'poisson': 1.0, 'mse': 0.0}[loss]
                assert torch.allclose(pred[name], torch.full_like(pred[name], want))
        b = build(w.config)                                    # the config builds the same network
        b.load_state_dict(wd)
        with torch.no_grad():
            assert all(torch.equal(u, v) for u, v in zip(b.eval()(x, n, aux=True), out))
    # twice: the first widen's part kept, a second one added
    w1 = widen_aux(m, [CARD, HALF])
    shake(w1, 0.3, 'aux_more.')
    w2 = widen_aux(w1, [LEVEL, CARD, HALF, RATE])
    assert w2.config['aux_parts'] == [['next_card'], ['next_half'], ['level', 'rate']]
    assert w2.aux_older == [TOY_AUX, [CARD, HALF]]
    with torch.no_grad():
        o1, o2 = w1(x, n, aux=True)[3], w2(x, n, aux=True)[3]
    lay2 = [LEVEL, CARD, HALF, RATE]
    for name in ('next_card', 'next_half'):
        assert torch.equal(o2[:, cols(lay2, name)], o1[:, cols([CARD, HALF], name)])


def test_save_load_and_the_game_targets(tmp_path, monkeypatch):
    """A checkpoint keeps a widened head and the layouts of before (extra 'aux_older'); `load` widens a head for
    some of the game's declared targets to all of them (the package declares next_half here) and leaves any
    other head as it is (the run refuses that where the head is needed)."""
    x, n = toy_x(seed=2), torch.full((64,), 2)
    m = nets(TOY_AUX)['tokens']
    shake(m)
    p1 = str(tmp_path / 'one.pt')
    save(m, p1, {'iter': 4})
    assert torch.load(p1, weights_only=False)['extra'] == {'iter': 4}
    a, extra = load(p1)                                        # the package of the toy: next_card only
    assert a.config == m.config and not hasattr(a, 'aux_older') and extra == {'iter': 4}
    monkeypatch.setitem(package.CORE.aux, 'toy', list(package.TOY_AUX) + [toy_aux2.HALF])
    b, extra = load(p1, game='toy')
    assert b.config['aux'] == [CARD, HALF] and b.aux_widened == TOY_AUX and b.aux_older == [TOY_AUX]
    with torch.no_grad():
        u, v = m(x, n, aux=True), b(x, n, aux=True)
    assert all(torch.equal(s, t) for s, t in zip(u[:3], v[:3])) and torch.equal(v[3][:, :6], u[3])
    p2 = str(tmp_path / 'two.pt')
    save(b, p2, {'iter': 5})
    ck = torch.load(p2, weights_only=False)
    assert ck['extra'] == {'iter': 5, 'aux_older': [TOY_AUX]} and ck['config']['aux_parts'] == [['next_card'],
                                                                                                  ['next_half']]
    c, _ = load(p2)
    assert c.aux_older == [TOY_AUX] and not hasattr(c, 'aux_widened')
    assert all(torch.equal(s, c.state_dict()[k]) for k, s in b.state_dict().items())
    assert game_aux(c) is c and game_aux(nets()['mlp']) is not None
    # a head the game's targets cannot widen (a target removed): loaded as it is
    p3 = str(tmp_path / 'other.pt')
    save(nets([['gone', 3, 'mse']])['mlp'], p3)
    assert load(p3)[0].config['aux'] == [['gone', 3, 'mse']]
    # the attributes travel with the network through strip_aux, add_reader and widen
    assert strip_aux(c).aux_older == [TOY_AUX] and 'aux_parts' not in strip_aux(c).config
    assert add_reader(c, TOY_READER, seed=1).aux_older == [TOY_AUX]
    assert widen(c, 200).aux_older == [TOY_AUX] and widen(c, 200).config['aux_parts'] == c.config['aux_parts']
    # a root belief (docs/az.md 18) for the game's targets: the widened head serves it, the head of before not
    from zero.net import HeadMissing, check_belief_head
    info0 = G.info
    monkeypatch.setattr(G, 'info', lambda g: dict(info0(g), aux_layout=list(info0(g)['aux_layout']) + [('next_half', 2)])
                        if g == 'toy' else info0(g))
    check_belief_head(b, 'toy')
    check_belief_head(load(p1, game='toy')[0], 'toy')
    with pytest.raises(HeadMissing, match='predicts'):
        check_belief_head(m, 'toy')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graphs need CUDA')
def test_cuda_graphs_answer_what_eager_answers_for_a_widened_head():
    """The token network's CUDA graphs (forward_static) and the conv network's (the folded copy) answer a widened
    head's root belief rows as the eager evaluator does, the old target's columns as before the widen."""
    from zero.infer import NetEvaluator
    from zero.protocol import ROW_BELIEF
    x = toy_x(40, seed=9).numpy()
    n2 = np.full(len(x), 2)
    off = np.arange(len(x) + 1, dtype=np.int32) * 2
    acts = np.tile(np.array([0, 3], np.int32), len(x))
    extra = np.zeros(len(x), np.uint8)
    extra[[1, 30]] = ROW_BELIEF
    for arch in ('tokens', 'grid'):
        m = nets(TOY_AUX)[arch]
        shake(m, 0.05)
        w = widen_aux(m, [HALF, CARD])
        shake(w, 0.3, 'aux_more.')
        eager = NetEvaluator(w, 'cpu')
        want, want_b = eager.evaluate(x, n2, off, acts), eager.evaluate(x, n2, off, acts, extra)
        ev = NetEvaluator(w.to('cuda'), 'cuda', graphs=True)
        got = ev.evaluate(x, n2, off, acts)
        assert ev.graph_calls == 1
        got_b = ev.evaluate(x, n2, off, acts, extra)             # (the conv graphs leave belief rows to eager)
        assert ev.graph_calls == (2 if arch == 'tokens' else 1) and got_b[3].shape == (2, 8)
        tol = 1e-4 if arch == 'tokens' else 3e-2                    # (the conv graphs run bfloat16)
        for u, v in zip(want + want_b, got + got_b):
            assert np.allclose(u, v, atol=tol, rtol=tol)


@pytest.mark.parametrize('arch', ['mlp', 'tokens'])
def test_widen_and_readers_compose(arch):
    """A reader (docs/az.md 36) and a widen in either order: the same network (the reader's parameters aside);
    a reader of the NEW target added after the widen (add_reader's job) changes no output either."""
    x, n = toy_x(seed=3), torch.full((64,), 2)
    m = nets(TOY_AUX)[arch]
    shake(m)
    a = add_reader(widen_aux(m, [CARD, HALF]), TOY_READER, seed=4)
    b = widen_aux(add_reader(m, TOY_READER, seed=4), [CARD, HALF])
    assert a.config == b.config and set(a.state_dict()) == set(b.state_dict())
    # (a reader's fresh weights depend on what was built before it: the same seed, other draws)
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items() if not k.startswith('readers.'))
    for net in (a, b):
        with torch.no_grad():
            o, ref = net(x, n, aux=True), m(x, n, aux=True)
        assert all(torch.equal(u, v) for u, v in zip(ref[:3], o[:3])) and torch.equal(o[3][:, :6], ref[3])
    half_reader = reader_layout([{'target': 'next_half', 'width': 2, 'loss': 'multinomial', 'inputs': [[0, 6], [25, 6]],
                                  'hidden': 8}])
    w = widen_aux(m, [CARD, HALF])
    r = add_reader(w, half_reader, seed=5)
    with torch.no_grad():
        p, q = w(x, n, aux=True), r(x, n, aux=True)
    assert all(torch.equal(u, v) for u, v in zip(p[:3], q[:3]))
    assert torch.equal(q[3][:, :8], p[3]) and q[3].shape[1] == 10      # the head's 8 columns, then the reader's
    shake(r, 0.3, 'readers.')
    with torch.no_grad():
        q = r(x, n, aux=True)
    assert torch.allclose(reader_predict(r, q[3])['next_half'], torch.softmax(q[3][:, 8:], 1))


def test_set_network_takes_a_widened_network():
    from zero.infer import NetEvaluator
    x = toy_x(seed=4).numpy()
    n2 = np.full(len(x), 2)
    off = np.arange(len(x) + 1, dtype=np.int32) * 2
    acts = np.tile(np.array([0, 3], np.int32), len(x))
    m = nets(TOY_AUX)['tokens']
    shake(m)
    ev = NetEvaluator(m)
    want = ev.evaluate(x, n2, off, acts)
    w = widen_aux(m, [CARD, HALF])
    ev.set_network(w)
    assert ev.model is w and set(ev.aux(x, n2)) == {'next_card', 'next_half'}
    assert all(np.array_equal(u, v) for u, v in zip(want, ev.evaluate(x, n2, off, acts)))
    ev.set_network(w)                                          # the same head: its weights, as before
    assert ev.model is w


# ----------------------------------------------------------------------------- the window and the losses

def selfplay(games=12, seed=4):
    """A toy self-play's samples with the next_card target (as a run records them)."""
    seat = G.nn_seat('m', 8, selfplay=True, sims_fast=2, p_full=0.5, temp_moves=4)
    drv = G.Driver('toy', [[seat, seat]] * games, concurrency=games, seed=seed, record=True, threads=2, aux=True)
    from zero.infer import UniformEvaluator, drive
    drive(drv, {'m': UniformEvaluator()})
    return drv.take_samples()


def test_older_columns_are_checked():
    two = [CARD, HALF]
    o = AUX.older_columns([HALF, CARD], [TOY_AUX])
    assert o['T'] == 2 and list(o['maps']) == [6]
    c, have = o['maps'][6]
    assert c.tolist() == [2, 3, 4, 5, 6, 7] and have.tolist() == [False, True]
    assert AUX.older_columns(two, None) is None and AUX.older_columns(two, []) is None
    for bad in ([[['next_card', 5, 'multinomial']]], [[HALF, ['x', 1, 'mse']]], [two]):
        with pytest.raises(ValueError):
            AUX.older_columns(two, bad)
    with pytest.raises(ValueError, match='either'):                     # two layouts of one width
        AUX.older_columns([CARD, ['a', 2, 'mse'], ['b', 2, 'mse']], [[CARD, ['a', 2, 'mse']], [CARD, ['b', 2, 'mse']]])


def test_window_places_older_shards_and_masks_each_target(tmp_path):
    """A window over shards without targets, of the older layout (next_card, 6 wide) and of the widened one
    (next_card + an 'mse' target, 7 wide): the older shard's targets in their columns, the new one's zero, the
    per-target mask (`aux_tmask`) of each row; through the window cache and the sparse window the same; a
    window without the older layouts refuses the narrower shard (as before). The losses count each target's
    rows only: the 'mse' target's loss is its mean over the rows that recorded it."""
    lay = [CARD, LEVEL]
    s0, s1, s2 = selfplay(seed=5), selfplay(seed=6), selfplay(seed=7)
    s0 = {k: v for k, v in s0.items() if k != 'aux'}                     # before any target
    s2 = dict(s2, aux=np.concatenate([s2['aux'], np.full((len(s2['player']), 1), 0.5, np.float32)], 1))
    for name, s in (('a', s0), ('b', s1), ('c', s2)):
        w = ShardWriter(str(tmp_path), name)
        w.add(s)
        w.close()
    paths = sorted(str(tmp_path / f) for f in os.listdir(tmp_path))
    info = G.info('toy')
    F_, A_ = info['num_features'], info['num_actions']
    older = AUX.older_columns(lay, [TOY_AUX])
    n0, n1, n2 = len(s0['player']), len(s1['player']), len(s2['player'])
    W = Window(paths, F_, A_, aux_width=7, aux_older=older)
    assert W.N == n0 + n1 + n2 and W.aux.shape == (W.N, 7)
    assert np.array_equal(W.aux_mask, np.r_[np.zeros(n0, bool), np.ones(n1 + n2, bool)])
    assert np.array_equal(W.aux_tmask, np.r_[np.zeros((n0, 2), bool), np.tile([True, False], (n1, 1)),
                                             np.ones((n2, 2), bool)])
    assert np.array_equal(W.aux[n0:n0 + n1, :6], s1['aux'].astype(np.float16)) and not W.aux[n0:n0 + n1, 6].any()
    assert np.array_equal(W.aux[n0 + n1:], s2['aux'].astype(np.float16))
    plain = Window(paths, F_, A_)                                        # every other array as without targets
    idx = np.arange(W.N)
    b, b0 = W.batch(idx), plain.batch(idx)
    for k in b0:
        assert torch.equal(b[k], b0[k]), k
    assert torch.equal(b['aux_tmask'], torch.from_numpy(W.aux_tmask.astype(np.float32)))
    for kw in ({'cache': WindowCache()}, {'sparse': True}, {'sparse': True, 'cache': WindowCache()}):
        V = Window(paths, F_, A_, aux_width=7, aux_older=older, **kw)
        assert np.array_equal(V.aux, W.aux) and np.array_equal(V.aux_tmask, W.aux_tmask)
        if 'cache' in kw:
            assert Window(paths, F_, A_, aux_width=7, aux_older=older, **kw).reused == W.N
    with pytest.raises(ValueError, match='width 6'):
        Window(paths, F_, A_, aux_width=7)
    # a subsampled window (over its budget) keeps the masks with their rows
    S = Window(paths, F_, A_, aux_width=7, aux_older=older, budget_rows=W.N // 2, seed=3)
    assert S.aux_tmask.shape == (S.N, 2) and np.array_equal(S.aux_tmask[:, 0], S.aux_mask)
    # the losses: each target over its own rows
    raw = torch.randn(W.N, 7)
    la = AUX.losses(lay, raw, b['aux'], b['aux_mask'], tmask=b['aux_tmask'])
    lvl = slice(n0 + n1, W.N)
    assert torch.allclose(la['level']['loss'], ((raw[lvl, 6] - 0.5) ** 2).mean()) and la['level']['rows'] == n2
    card = AUX.losses([CARD], raw[:, :6], b['aux'][:, :6], b['aux_mask'])['next_card']
    assert torch.equal(la['next_card']['loss'], card['loss'])            # (the multinomial: its rows as before)
    st = AUX.window_stats(W, lay)
    assert st['rows'] == n1 + n2 and st['level'] == {'rows': n2} and st['next_card']['rows'] == n1 + n2


def test_optimizer_groups_are_matched_by_their_parts():
    """`match_opt_groups`: the saved groups of the network of before (trunk, head, readers - whichever it had)
    are found among a widened network's groups, in the saved order, by their parameters' shapes."""
    from zero.run import match_opt_groups
    m = add_reader(widen_aux(nets(TOY_AUX)['mlp'], [CARD, HALF]), TOY_READER, seed=1)
    names = [n for n, _ in m.named_parameters()]
    params = dict(m.named_parameters())

    def group(pre, neg=()):
        return [params[n] for n in names if n.startswith(pre) and not any(n.startswith(x) for x in neg)]
    trunk, head, rdr, more = group('', ('aux_', 'readers.')), group('aux_head.'), group('readers.'), group('aux_more.0.')
    canon = [trunk, head, rdr, more]

    def state(groups, stepped=True):
        out, i, gs = {}, 0, []
        for g in groups:
            ids = []
            for p in g:
                if stepped:
                    out[i] = {'step': torch.tensor(3.0), 'exp_avg': torch.zeros_like(p), 'exp_avg_sq': torch.zeros_like(p)}
                ids.append(i)
                i += 1
            gs.append({'params': ids, 'lr': 1e-3})
        return {'state': out, 'param_groups': gs}
    assert match_opt_groups(state([trunk, head]), canon) == [0, 1]                 # before the widen (and the reader)
    assert match_opt_groups(state([trunk, head, rdr]), canon) == [0, 1, 2]
    assert match_opt_groups(state([trunk, head, more]), canon) == [0, 1, 3]       # a reader added after a widen
    assert match_opt_groups(state([trunk, head, more, rdr]), canon) == [0, 1, 3, 2]
    assert match_opt_groups(state([trunk, head, more], stepped=False), canon) == [0, 1, 3]
    other = [torch.zeros(3, 3)] * len(trunk)
    assert match_opt_groups(state([other]), canon) is None and match_opt_groups(state([]), canon) is None


# ----------------------------------------------------------------------------- the run loop

SMOKE = ['--preset', 'toy-smoke', '--conc', '24', '--games', '24', '--eval-games', '8', '--eval-random-games', '4',
         '--elo-games', '4', '--ref-games', '8']


def run_loop(run, *extra, second=False, timeout=600):
    """zero.run on the toy; `second`: with tests/toy_aux2.py's second target (next_half) declared."""
    e = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONPATH=PYPATH)
    cmd = [os.path.join(ROOT, 'tests', 'toy_aux2.py')] if second else ['-m', 'zero.run']
    return subprocess.run([sys.executable, *cmd, '--run', str(run), *SMOKE, *extra], cwd=ROOT, capture_output=True,
                          text=True, timeout=timeout, env=e)


def test_a_fork_widens_its_head_exactly_and_learns_the_new_target(tmp_path, monkeypatch):
    """A run with the toy's one target (next_card), forked (`fork_run --source`) and run on with a second
    target declared (next_half): its first training widens checkpoint K's head (logged; `train.aux.head_widened`),
    keeps the optimizer moments of every parameter of before (the new part a group of its own, fresh), trains
    on the window whose older shards have the first target only (their rows count for next_card, not for
    next_half) and learns the new target. Checkpoint K widened computes K's outputs bit for bit; the run with
    one target writes neither `aux_parts` nor `aux_older`."""
    from zero.run import read_metrics
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import fork_run
    src = tmp_path / 'src'
    p = run_loop(src, '--iters', '2', '--aux-next-card-w', '1.0')
    assert p.returncode == 0, p.stdout + p.stderr
    for it in (1, 2):
        ck = torch.load(str(src / 'models' / f'iter_{it:04d}.pt'), weights_only=False)
        assert 'aux_parts' not in ck['config'] and ck['extra'] == {'iter': it}
    c = tmp_path / 'fork'
    assert fork_run.main(['--source', str(src), '--dest', str(c), '--iter', '2']) == 0
    p = run_loop(c, '--iters', '4', '--aux-next-half-w', '1.0', second=True)
    assert p.returncode == 0, p.stdout + p.stderr
    log = open(c / 'log.txt', encoding='utf-8').read()
    assert "iter 3: auxiliary head of checkpoint 2 widened ['next_card'] -> ['next_card', 'next_half']" in log
    assert 'optimizer state not loaded' not in log and log.count('auxiliary head of checkpoint') == 1
    m = read_metrics(str(c))
    a3, a4 = m[3]['train']['aux'], m[4]['train']['aux']
    assert a3['head_widened'] == {'from': ['next_card'], 'added': ['next_half']} and 'head_widened' not in a4
    assert a3['next_half']['rows'] < a3['next_card']['rows'] == a3['rows']      # the older shard: next_card only
    assert m[4]['train']['aux_next_half'] < np.log(2) and m[4]['train']['aux_next_half'] < m[3]['train']['aux_next_half']
    for it in (3, 4):
        ck = torch.load(str(c / 'models' / f'iter_{it:04d}.pt'), weights_only=False)
        assert ck['config']['aux'] == toy_aux2.LAYOUT2 and ck['config']['aux_parts'] == [['next_card'], ['next_half']]
        assert ck['extra']['aux_older'] == [TOY_AUX]
    o = torch.load(str(c / 'models' / 'opt_latest.pt'), weights_only=False)['state']
    sizes = [len(g['params']) for g in o['param_groups']]
    assert sizes[1:] == [4, 2]                                          # trunk, the head of before, the new part
    steps = [int(o['state'][i]['step']) for i in range(sum(sizes))]
    assert len(set(steps[:sizes[0] + 4])) == 1 and steps[-1] < steps[0]  # the moments of before kept, the part's newer
    # checkpoint 2 widened: its outputs bit for bit (the package of this process declares next_card only)
    k2, _ = load(str(c / 'models' / 'iter_0002.pt'))
    assert k2.config['aux'] == TOY_AUX
    monkeypatch.setitem(package.CORE.aux, 'toy', list(package.TOY_AUX) + [toy_aux2.HALF])
    w2, _ = load(str(c / 'models' / 'iter_0002.pt'), game='toy')
    x, n = toy_x(256, seed=8), torch.full((256,), 2)
    with torch.no_grad():
        u, v = k2(x, n, aux=True), w2(x, n, aux=True)
    assert all(torch.equal(s, t) for s, t in zip(u[:3], v[:3])) and torch.equal(v[3][:, :6], u[3])
