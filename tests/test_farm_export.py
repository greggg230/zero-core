"""The farm pool's fast per-generation export (docs/az.md 53): a weight-swap template per architecture
(`zero.onnx_export.export_template` / `swap_export`) and the run's reference positions kept
(`zero.farm_h2h.export_cached`). A swapped graph is a full export's graph, bit for bit (a network whose values trigger
no value-dependent rewrite) and answers as it does; the architecture fingerprint keys the template (another config,
export setting or exporter version: another template); a swap whose tensors do not match the template is refused; the
reference positions are recorded once and every new network's answers written on them, the reference.npz of before.
Skipped without onnx / onnxscript / onnxruntime."""
import json
import os

import numpy as np
import pytest
import torch

pytest.importorskip('onnx')
pytest.importorskip('onnxscript')
pytest.importorskip('onnxruntime')

from zero import farm_h2h as FH, games as G, net as N, onnx_export as OX, onnx_parity as OP  # noqa: E402

from test_tail_net import OFFERS, rand_net, toy_batch  # noqa: E402
from test_onnx2 import gate_rows, kinds_of  # noqa: E402

GAME, SPEC = 'toy_tail', '{"items0": 9}'


def randomized(net, seed):
    """`net` with every floating-point state tensor drawn anew (N(0, 0.3^2)): no exact 0 or 1 anywhere, so no
    value-dependent rewrite of the exporter (a Gemm's all-zero bias dropped) fires."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for t in net.state_dict(keep_vars=True).values():
            if t.is_floating_point():
                t.copy_(torch.randn(t.shape, generator=g) * 0.3)
    return net


def offers_net(seed):
    return randomized(rand_net(OFFERS, seed=seed), seed)


def toy_tail_net(seed, hidden=16):
    torch.manual_seed(seed)
    return randomized(N.make_net(G.info(GAME), GAME, 'tokens', hidden=hidden, blocks=1, heads=2), seed)


def rows_of(net, B=120, seed=6):
    rows = gate_rows(net, toy_batch(B, seed=seed, p_tail=0.1, p_mark=0.1, p_gen=0.05, p_offer=0.1), kinds_of(B, 8))
    rows.batch_off = np.array([0, 30, 60, 90, B], np.int64)
    return rows


def outputs(path, rows):
    s = OP.session(path, 1)
    return [OP.run_chunked(s, rows.take(np.arange(a, b))) for a, b in zip(rows.batch_off[:-1], rows.batch_off[1:])]


def same_bits(a, b):
    return all(all(np.array_equal(np.asarray(x).view(np.uint8), np.asarray(y).view(np.uint8)) for x, y in zip(p, q))
               for p, q in zip(a, b))


@pytest.fixture(scope='module')
def tpl(tmp_path_factory):
    d = tmp_path_factory.mktemp('tpl')
    path = str(d / 'template.onnx')
    return path, OX.export_template(offers_net(1), path)


def test_swap_equals_a_full_export(tpl, tmp_path):
    """Another network of the architecture swapped into the template is the file `export` writes for it, byte for byte
    (every weight, every buffer the interface computes from them - the tail maps, the gates -, the metadata), and both
    answer the same on rows with tails and overrides; the template describes its weights, constants and sources."""
    path, t = tpl
    assert t['template'] == OX.TEMPLATE and t['weights'] and {'v_tW', 'v_g'} <= set(t['weights'])
    assert 'v_width' in t['constants'] and t['sha256'] == OX.file_sha256(path)
    net = offers_net(7)
    info = {'checkpoint': 'toy/7', 'checkpoint_sha256': 'ab' * 32, 'checkpoint_iter': '7'}
    full, swap = str(tmp_path / 'full.onnx'), str(tmp_path / 'swap.onnx')
    m1 = OX.export(net, full, 'packed', info)
    m2 = OX.swap_export(net, path, t, swap, 'packed', info)
    assert m1 == m2
    with open(full, 'rb') as f, open(swap, 'rb') as g:
        assert f.read() == g.read()
    rows = rows_of(net)
    assert same_bits(outputs(swap, rows), outputs(full, rows))


def test_derived_weights_and_a_value_dependent_rewrite(tmp_path):
    """Weights the exporter folds into constants of its own (a widened pointer head's parts: slices of `ptr`) are rebuilt
    by their recipes; a network whose values trigger a value-dependent rewrite (a part with an all-zero bias: the full
    export drops the Gemm's bias input) gets the template's general graph, which answers bit for bit as the full
    export's (x + 0 = x)."""
    a = offers_net(2)
    a.ptr_parts = [(0, 2), (2, a.ptr.out_features)]
    path = str(tmp_path / 'template.onnx')
    t = OX.export_template(a, path)
    assert any(k.startswith('slice') for k in t['derived']), t['derived']
    b = offers_net(3)
    b.ptr_parts = a.ptr_parts
    with torch.no_grad():
        b.ptr.bias[2:].zero_()
    full, swap = str(tmp_path / 'full.onnx'), str(tmp_path / 'swap.onnx')
    OX.export(b, full, 'packed', {'checkpoint': 'b'})
    OX.swap_export(b, path, t, swap, 'packed', {'checkpoint': 'b'})
    rows = rows_of(b)
    assert same_bits(outputs(swap, rows), outputs(full, rows))
    ref = str(tmp_path / 'reference.npz')
    OX.write_reference(b, [rows], ref, 80, {'checkpoint': 'b'})
    assert OX.check_reference(swap, ref, 1)['max_abs'] < OX.REFERENCE_TOL


def test_fingerprint_invalidation(tpl, tmp_path, monkeypatch):
    """Same architecture, other weights: the same fingerprint. Another config, packing-relevant layout, opset, exporter
    version or zero-core Python: another one - and a swap onto the old template is refused (a new template, never a
    fallback)."""
    path, t = tpl
    fp = OX.export_fingerprint(offers_net(5))[0]
    assert fp == t['fingerprint']
    other = randomized(rand_net(OFFERS, seed=5, d=24), 5)
    assert OX.export_fingerprint(other)[0] != fp
    assert OX.export_fingerprint(offers_net(5), opset=18)[0] != fp
    with pytest.raises(ValueError, match='fingerprint'):
        OX.swap_export(other, path, t, str(tmp_path / 'x.onnx'))
    assert not os.path.exists(tmp_path / 'x.onnx')
    v = OX._versions()
    monkeypatch.setattr(OX, '_versions', lambda: dict(v, onnxscript='0.0.0'))
    assert OX.export_fingerprint(offers_net(5))[0] != fp
    monkeypatch.undo()
    monkeypatch.setattr(OX, '_code_sha256', lambda: '0' * 64)
    assert OX.export_fingerprint(offers_net(5))[0] != fp


def test_refusal_on_mismatched_tensors(tpl, tmp_path, monkeypatch):
    """A swap refuses (ValueError, nothing written) a weight without its tensor, a module tensor the template holds
    constant that changed, other weight sources, or a template file that is not the template's."""
    path, t = tpl
    out = str(tmp_path / 'x.onnx')
    net = offers_net(4)
    for bad, match in ((dict(t, weights=t['weights'] + ['net.nope']), 'net.nope'),
                       (dict(t, sources=t['sources'][1:]), 'weights are not'),
                       (dict(t, sha256='0' * 64), 'sha256'),
                       (dict(t, template=OX.TEMPLATE + 1), 'format')):
        with pytest.raises(ValueError, match=match):
            OX.swap_export(net, path, bad, out)
        assert not os.path.exists(out)
    orig = OX.ZeroOnnx2.__init__

    def init(self, n, packing='packed'):
        orig(self, n, packing)
        self.v_width = self.v_width + 1           # (a constant of the layout's, computed otherwise)
    monkeypatch.setattr(OX.ZeroOnnx2, '__init__', init)
    with pytest.raises(ValueError, match='constant'):
        OX.swap_export(net, path, t, out)
    assert not os.path.exists(out)


def test_reference_reuse(tmp_path, monkeypatch):
    """export_cached: the first gen builds the template and records the reference positions (a toy_tail self-play of its
    network); the next gen swaps its weights in and answers the same positions - no self-play - in the reference.npz of
    before (the same arrays, its own checkpoint in the meta), within the tolerance of its graph; answering the kept
    positions with the first network gives the first reference set's answers exactly (the positions round-trip); a
    network of another architecture gets a template of its own."""
    # (record_rows sets ZERO_INFLIGHT=1 for its self-play and leaves it set: restored after this test, or the later
    # tests' run processes inherit it - test_shard_thread's rolling quotas and test_remoteplay's did)
    monkeypatch.setenv('ZERO_INFLIGHT', os.environ.get('ZERO_INFLIGHT', '2'))
    cache = str(tmp_path / 'export')
    ck = []
    for s in (1, 2):
        p = str(tmp_path / f'iter_{s:04d}.pt')
        N.save(toy_tail_net(s), p, {'iter': s})
        ck.append(p)
    a = FH.export_cached(ck[0], 1, str(tmp_path / 'ckpt' / '1'), GAME, SPEC, cache, 'farm/1', threads=1,
                         checkpoint_iter=1, n=64)
    assert a['graph'] == 'template' and a['positions'] == 1
    assert set(a['steps']) == {'load', 'template', 'swap', 'record_rows', 'reference', 'check'}
    calls = []
    with monkeypatch.context() as mp:
        mp.setattr(FH, 'record_rows', lambda *x, **k: calls.append(1))
        b = FH.export_cached(ck[1], 2, str(tmp_path / 'ckpt' / '2'), GAME, SPEC, cache, 'farm/2', threads=1,
                             checkpoint_iter=2, n=64)
    assert not calls and b['graph'] == 'swap' and b['positions'] == 1
    assert set(b['steps']) == {'load', 'swap', 'reference', 'check'}
    assert b['check']['max_abs'] < OX.REFERENCE_TOL
    za, zb = np.load(a['reference']), np.load(b['reference'])
    assert set(za.files) == set(zb.files) and {'t_off', 'ov_off', 'priors'} <= set(za.files)
    rows_keys = [k for k in za.files if k not in ('meta', 'priors', 'value', 'margin', 'aux')]
    for k in rows_keys:
        assert za[k].dtype == zb[k].dtype and np.array_equal(za[k], zb[k]), k
    ma, mb = json.loads(str(za['meta'])), json.loads(str(zb['meta']))
    assert set(ma) == set(mb) and ma['sources'] == mb['sources'] == 'farm_h2h'
    assert (ma['checkpoint'], mb['checkpoint']) == ('farm/1', 'farm/2') and ma['checkpoint_sha256'] != mb['checkpoint_sha256']
    assert not np.array_equal(za['priors'], zb['priors'])
    # the kept positions answered by gen 1's network: gen 1's reference set exactly
    m1, _ = OX.load_checkpoint(ck[0], GAME)
    d = [x for x in os.listdir(cache)]
    assert len(d) == 1
    pos = [f for f in os.listdir(os.path.join(cache, d[0])) if f.startswith('positions-') and f.endswith('.npz')]
    rows, bo, _w, pm = OX.load_reference(os.path.join(cache, d[0], pos[0]))
    rows.source = pm['sources']
    again = str(tmp_path / 'again.npz')
    OX.write_reference_rows(m1, rows, bo, again, {'checkpoint': 'farm/1'})
    with np.load(again) as z:
        for k in za.files:
            if k != 'meta':
                assert z[k].dtype == za[k].dtype and np.array_equal(z[k], za[k]), k
    # another architecture: a template of its own
    p = str(tmp_path / 'iter_0003.pt')
    N.save(toy_tail_net(3, hidden=24), p, {'iter': 3})
    c = FH.export_cached(p, 3, str(tmp_path / 'ckpt' / '3'), GAME, SPEC, cache, 'farm/3', threads=1, n=64)
    assert c['graph'] == 'template' and c['positions'] == 3 and len(os.listdir(cache)) == 2
