"""Readers (docs/az.md 36): a small branch of the network that reads declared parts of the input (the
toy's: the round and the revealed round cards), predicts one auxiliary target (the toy's next_card) and
feeds that prediction into the trunk through a projection that starts at zero. The network part: built
last, exact carry-over (`add_reader`), the prediction reaches the trunk, the token network's static
forward and widen, the declarations; the run part: settings, a run with the reader, the carry-over of a
run without it, off byte for byte; the evaluators."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from zero import games as G, package
from zero.net import (GridNet, add_aux, add_reader, aux_answer, aux_predict, build, load, make_net, reader_layout,
                      reader_predict, save, strip_aux, widen)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYPATH = os.pathsep.join([os.path.join(ROOT, 'python')] + [p for p in [os.environ.get('PYTHONPATH')] if p])
TOY_AUX = [['next_card', 6, 'multinomial']]
# the toy's reader as the package declares it, completed with its target's width and loss
TOY_READER = [{'target': 'next_card', 'width': 6, 'loss': 'multinomial', 'inputs': [[0, 6], [25, 6]], 'hidden': 32,
               'enter': 'trunk', 'belief': False}]
GRID = {'planes': 7, 'h': 1, 'w': 13, 'offset': 31, 'segments': [[31, 7]], 'cell_action0': 0, 'hex': None}


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


# ----------------------------------------------------------------------------- the network

def test_declarations_are_checked():
    assert reader_layout(None) == [] and reader_layout(TOY_READER) == TOY_READER
    lay = reader_layout([{'target': 't', 'width': 3, 'loss': 'mse', 'inputs': [[4, 2]]}])
    assert lay == [{'target': 't', 'width': 3, 'loss': 'mse', 'inputs': [[4, 2]], 'hidden': 64, 'enter': 'trunk',
                    'belief': False}]
    for bad in ([dict(TOY_READER[0], inputs=[])], [dict(TOY_READER[0], inputs=[[0, 6], [5, 2]])],
                [dict(TOY_READER[0], loss='xent')], [dict(TOY_READER[0], enter='policy')],
                [dict(TOY_READER[0], width=0)], TOY_READER + TOY_READER):
        with pytest.raises(ValueError):
            reader_layout(bad)
    info = G.info('toy')
    with pytest.raises(ValueError):                               # past the features
        make_net(info, 'toy', 'mlp', 32, 1, readers=[dict(TOY_READER[0], inputs=[[190, 10]])])
    with pytest.raises(ValueError, match='bags and globals'):     # a token network: not a segment's floats
        make_net(info, 'toy', 'tokens', 32, 1, heads=2, readers=[dict(TOY_READER[0], inputs=[[31, 7]])])


@pytest.mark.parametrize('arch', ['mlp', 'tokens', 'grid'])
def test_reader_is_built_last_and_changes_nothing_at_first(arch):
    x = torch.from_numpy(toy_x())
    n = torch.full((len(x),), 2)
    for aux in (None, TOY_AUX):
        plain, withr = nets(aux)[arch], nets(aux, TOY_READER)[arch]
        assert 'readers' not in plain.config and withr.config['readers'] == TOY_READER
        sd0, sd1 = plain.state_dict(), withr.state_dict()
        new = set(sd1) - set(sd0)
        assert new and all(k.startswith('readers.0.') for k in new) and not set(sd0) - set(sd1)
        for k in sd0:
            assert torch.equal(sd0[k], sd1[k]), k
        with torch.no_grad():
            a, b, c = plain(x, n), withr(x, n), withr(x, n, aux=True)
        assert len(b) == 3 and len(c) == 4
        assert c[3].shape == (len(x), 6 * (2 if aux else 1))      # the head's columns, then the reader's
        for u, v, w in zip(a, b, c):
            assert torch.equal(u, v) and torch.equal(u, w)
        assert torch.equal(c[3], torch.zeros_like(c[3]))          # every last layer starts at zero
        assert torch.allclose(reader_predict(withr, c[3])['next_card'], torch.full((len(x), 6), 1 / 6))
        if aux:
            assert torch.equal(plain(x, n, aux=True)[3], c[3][:, :6])


@pytest.mark.parametrize('arch', ['mlp', 'tokens', 'grid'])
def test_add_reader_keeps_everything_exactly(arch):
    x = torch.from_numpy(toy_x(seed=2))
    n = torch.full((len(x),), 2)
    for aux in (None, TOY_AUX):
        m = nets(aux)[arch]
        shake(m)
        torch.manual_seed(123)
        r0 = torch.rand(3)
        torch.manual_seed(123)
        r = add_reader(m, TOY_READER, seed=9)
        assert torch.equal(torch.rand(3), r0), 'add_reader leaves the global torch RNG alone'
        assert r.config['readers'] == TOY_READER and add_reader(r, TOY_READER) is r
        for k, v in m.state_dict().items():
            assert torch.equal(v, r.state_dict()[k]), k
        with torch.no_grad():
            p, q = m(x, n, aux=bool(aux)), r(x, n, aux=bool(aux))
        for u, v in zip(p, q):
            assert torch.equal(u, v[:, :u.shape[1]])
        assert torch.equal(add_reader(m, TOY_READER, seed=9).state_dict()['readers.0.mlp.0.weight'],
                           r.state_dict()['readers.0.mlp.0.weight'])
        with pytest.raises(ValueError):
            add_reader(r, [dict(TOY_READER[0], hidden=8)])
        with pytest.raises(ValueError):
            add_reader(r, [])
    # the auxiliary head and the reader in either order: the same network, the same head
    m = nets()[arch]
    a = add_aux(add_reader(m, TOY_READER, seed=1), TOY_AUX, seed=2)
    b = add_reader(add_aux(m, TOY_AUX, seed=2), TOY_READER, seed=1)
    assert a.config == b.config and set(a.state_dict()) == set(b.state_dict())
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items() if not k.startswith('readers.'))
    s = strip_aux(a)                                              # strip_aux keeps the reader
    assert s.config['readers'] == TOY_READER and 'aux' not in s.config


@pytest.mark.parametrize('arch', ['mlp', 'tokens', 'grid'])
def test_the_prediction_reaches_the_trunk_and_reads_only_its_inputs(arch):
    x = torch.from_numpy(toy_x(seed=4))
    n = torch.full((len(x),), 2)
    m = nets(TOY_AUX, TOY_READER)[arch]
    shake(m, 0.3, 'readers.')
    with torch.no_grad():
        p = m(x, n, aux=True)
        base = nets(TOY_AUX)[arch](x, n, aux=True)
        assert not torch.allclose(p[1], base[1])                  # a trained reader moves the value
        # the reader reads features 0-5 and 25-30 only: change every other feature, its output stays
        y = x.clone()
        y[:, 6:25] += 1.0
        y[:, 31:] = y[:, 31:] * 0.5
        assert torch.equal(m(y, n, aux=True)[3][:, 6:], p[3][:, 6:])
        z = x.clone()
        z[:, 25:31] = 1.0 - z[:, 25:31]
        assert not torch.equal(m(z, n, aux=True)[3][:, 6:], p[3][:, 6:])
        # the belief rows' rows: the same outputs of those rows
        rows = torch.tensor([3, 0, 7])
        q = m(x, n, aux=True, aux_rows=rows)[3]
        assert torch.allclose(q, p[3][rows], atol=1e-6)
    pr = aux_answer(m, p[3])
    assert torch.allclose(pr['next_card'], aux_predict(TOY_AUX, p[3])['next_card'])    # the head's by default
    m.reader_specs = [dict(m.reader_specs[0], belief=True)]
    assert torch.allclose(aux_answer(m, p[3])['next_card'], reader_predict(m, p[3])['next_card'])


def test_detach_stops_only_the_trunks_gradient():
    x = torch.from_numpy(toy_x(seed=5))
    n = torch.full((len(x),), 2)
    for detach in (True, False):
        m = nets(TOY_AUX, TOY_READER)['tokens'].train()
        shake(m, 0.3, 'readers.')
        m.reader_detach = detach
        out = m(x, n)
        out[1].sum().backward()                                   # a trunk loss: the value
        g = m.readers[0].mlp[0].weight.grad
        assert m.readers[0].proj.weight.grad.abs().sum() > 0      # the projection learns either way
        assert (g is None or not g.abs().sum()) if detach else g.abs().sum() > 0


def test_token_static_forward_and_widen():
    from test_net_data import TOK_A, TOK_F, TOK_F2, TOK_LAYOUT2, tok_net, tok_x, widened_x
    from zero.infer import token_index, token_presence
    from zero.net import token_feature_map, tokens_layout
    # a reader over the bag 'zones' and the two trailing globals of TOK_LAYOUT
    spec = [{'target': 'z', 'width': 4, 'loss': 'multinomial', 'inputs': [[19, 24], [91, 2]], 'hidden': 8}]
    m = tok_net(31)
    r = add_reader(m, reader_layout(spec), seed=4)
    shake(r, 0.3, 'readers.')
    x = tok_x(40, 32)
    n = torch.randint(2, 5, (40,))
    with torch.no_grad():
        o = r(x, n, aux=True)
        P = token_presence(x.numpy(), r.present_index())
        T = int(P.sum(1).max())
        full = torch.from_numpy(token_index(P, np.zeros((len(x), T), np.int64)))
        s = r.forward_static(x, n, full, torch.from_numpy(P.sum(1)), aux=True)
    for u, v in zip(o, s):
        assert torch.allclose(u, v, atol=1e-5)
    # widen across the layout change of test_net_data: the reader reads the same floats where they went
    lay2 = tokens_layout(TOK_LAYOUT2, TOK_F2, TOK_A + 2)
    fmap = token_feature_map(r.config['layout'], TOK_F, lay2, TOK_F2)
    w = widen(r, TOK_F2, TOK_A + 2, layout=TOK_LAYOUT2)
    assert w.config['readers'][0]['inputs'] == [[19, 24], [121, 2]]
    with torch.no_grad():
        q = w(widened_x(x, fmap, TOK_F2), n, aux=True)
    assert torch.allclose(q[3], o[3], atol=1e-5) and torch.allclose(q[1], o[1], atol=1e-5)
    ww = widen(nets(None, TOY_READER)['mlp'], 200)                # new features after the old: the same inputs
    assert ww.config['readers'] == TOY_READER


def test_checkpoint_with_a_reader_loads_and_passes_the_encoding_guard(tmp_path):
    from zero.net import check_checkpoint
    t = nets(TOY_AUX, TOY_READER)['tokens']
    shake(t, 0.1, 'readers.')
    p = str(tmp_path / 'r.pt')
    save(t, p, {'iter': 0})
    check_checkpoint(p)
    m, _ = load(p)
    assert m.config['readers'] == TOY_READER and m.reader_detach is True
    for k, v in t.state_dict().items():
        assert torch.equal(v, m.state_dict()[k])
    assert build(m.config).config == m.config
