"""Duplicate action claims (docs/az.md 58): the run setting dup_action_reduce ('legacy' default | 'max') and the token
network's `dup_reduce` - two tokens claiming one action (two equal cards' identity action) score it by the largest
claim, an exact, order-free reduction: the same answer on the CPU, on CUDA (repeated runs, eager and the CUDA graphs)
and through the ONNX export (every scatter unique). Legacy networks unchanged: no config entry, the plain scatter, the
export of before."""
import json

import numpy as np
import pytest
import torch

from zero import run as ZR
from zero.infer import NetEvaluator
from zero.net import TokenNet, build, check_dup_reduce, load, save, set_dup_reduce

from test_net_data import TOK_A, TOK_F, TOK_LAYOUT_ID, tok_x
from test_token_graphs import a_batch, close, net

CUDA = torch.cuda.is_available()


def claims(m, x, n):
    """Run m's eager forward on rows x and record the policy scatter's writes: (base, idx, val) per chunk."""
    got = []
    orig = m._scatter_policy

    def spy(base, idx, val):
        got.append((base.detach().clone(), idx.clone(), val.detach().clone()))
        return orig(base, idx, val)
    m._scatter_policy = spy
    try:
        with torch.no_grad():
            out = m(x, n)
    finally:
        del m._scatter_policy
    return out, got


def dup_rows(idx, A):
    """Per row of a scatter's writes: the real actions (< A) written more than once."""
    out = []
    for r in idx.tolist():
        a = [v for v in r if v < A]
        out.append({v for v in a if a.count(v) > 1})
    return out


# ------------------------------------------------------------------ the setting and the network property

def test_the_setting_is_opt_in_and_checked():
    assert ZR.DEFAULTS['dup_action_reduce'] == 'legacy'
    cfg = ZR.resolve_config(ZR.parse(['--run', 'x']), {})
    assert cfg['dup_action_reduce'] == 'legacy'
    cfg = ZR.resolve_config(ZR.parse(['--run', 'x', '--arch', 'tokens', '--dup-action-reduce', 'max']), {})
    assert cfg['dup_action_reduce'] == 'max'
    with pytest.raises(SystemExit, match='dup_action_reduce'):
        ZR.resolve_config(ZR.parse(['--run', 'x', '--arch', 'tokens', '--dup-action-reduce', 'mean']), {})
    with pytest.raises(SystemExit, match='token network'):
        ZR.resolve_config(ZR.parse(['--run', 'x', '--dup-action-reduce', 'max']), {})
    assert [check_dup_reduce(v) for v in (None, 'legacy', 'max')] == [None, None, 'max']


def test_the_network_property_is_versioned_in_its_config(tmp_path):
    """A legacy network's config has no entry (its checkpoints as before); 'max' is recorded, survives save / load /
    build, and `set_dup_reduce` turns it on and off without touching a weight."""
    legacy = net()
    assert legacy.dup_reduce is None and 'dup_reduce' not in legacy.config
    m = net()
    assert set_dup_reduce(m, 'max') and m.config['dup_reduce'] == 'max' and not set_dup_reduce(m, 'max')
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), legacy.state_dict().values()))
    p = str(tmp_path / 'm.pt')
    save(m, p, {'iter': 0})
    back, _ = load(p, game=None, allow_mismatch=True)
    assert back.dup_reduce == 'max' and build(back.config).dup_reduce == 'max'
    assert set_dup_reduce(back, 'legacy') and back.dup_reduce is None and 'dup_reduce' not in back.config
    assert TokenNet(TOK_F, TOK_A, TOK_LAYOUT_ID, d=16, layers=1, heads=2, dup_reduce='max').config['dup_reduce'] == 'max'
    with pytest.raises(ValueError, match='dup_action_reduce'):
        TokenNet(TOK_F, TOK_A, TOK_LAYOUT_ID, d=16, layers=1, heads=2, dup_reduce='first')


# ------------------------------------------------------------------ the reduction on the CPU

def test_max_is_the_largest_claim_and_legacy_is_unchanged():
    """On rows where two present units claim the same identity action: 'max' answers the largest of their logits (one
    of the claims, bit for bit), legacy the CPU's last write; every other action - and every row without a duplicate -
    is the same in both modes, bit for bit."""
    m = net()
    B = 120
    x, n = tok_x(B, 5), torch.full((B,), 2)
    m.chunk = 1000                                                # (one chunk: the writes in the rows' order)
    (pl, *_), got = claims(m, x, n)
    assert len(got) == 1
    base, idx, val = got[0]
    dups = dup_rows(idx, TOK_A)
    assert sum(bool(d) for d in dups) >= 15, sum(bool(d) for d in dups)
    set_dup_reduce(m, 'max')
    with torch.no_grad():
        pm = m(x, n)[0]
    for b in range(B):
        for a in range(TOK_A):
            w = val[b][idx[b] == a]
            if len(w) == 0:
                assert pm[b, a] == pl[b, a] == base[b, a]
            elif a in dups[b]:
                assert pm[b, a] == w.max() and pl[b, a] == w[-1], (b, a)
            else:
                assert pm[b, a] == pl[b, a] == w[0]


def test_max_keeps_forward_static_and_chunking_and_trains():
    """'max' in every forward of the token network (eager in any chunking, forward_static) and its gradient: the largest
    claim gets it (equal claims share it), a smaller claim none."""
    from zero.infer import token_index, token_presence
    m = net()
    set_dup_reduce(m, 'max')
    B = 70
    x, n = tok_x(B, 9), torch.full((B,), 3)
    with torch.no_grad():
        m.chunk = 1000
        whole = m(x, n)
        m.chunk = 7
        small = m(x, n)
        P = token_presence(x.numpy(), m.present_index())
        full = torch.from_numpy(token_index(P, np.zeros((B, int(P.sum(1).max())), np.int64)))
        st = m.forward_static(x, n, full, torch.from_numpy(P.sum(1)))
    for a, b, c in zip(whole, small, st):
        assert torch.allclose(a, b, atol=1e-5) and torch.allclose(a, c, atol=1e-5)
    # the gradient through the reduction: d(sum of the policy) / d(claims) = 1 on each action's largest claim
    base = torch.zeros(2, 5)
    idx = torch.tensor([[1, 1, 2, 4, 1], [3, 3, 0, 4, 4]])
    val = torch.tensor([[0.5, 2.0, 1.0, 7.0, -1.0], [1.5, 1.5, 0.0, 9.0, 9.0]], requires_grad=True)
    out = m._scatter_policy(base, idx, val)
    assert out.tolist() == [[0.0, 2.0, 1.0, 0.0, 7.0], [0.0, 0.0, 0.0, 1.5, 9.0]]
    out[:, :4].sum().backward()
    assert val.grad.tolist() == [[0.0, 1.0, 1.0, 0.0, 0.0], [0.5, 0.5, 1.0, 0.0, 0.0]]


@pytest.mark.parametrize('dup', [None, 'max'])
def test_the_eager_evaluator_on_the_cpu(dup):
    """NetEvaluator on the CPU answers the same bits twice in each mode (the setting changes only the duplicate
    claims)."""
    m = net()
    set_dup_reduce(m, dup)
    ev = NetEvaluator(m, 'cpu')
    x, npl, off, act, kind = a_batch(m, 65, 3)
    a, b = ev.evaluate(x, npl, off, act, kind), ev.evaluate(x, npl, off, act, kind)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))


# ------------------------------------------------------------------ CUDA: the same bits every run, and the CPU's

@pytest.mark.skipif(not CUDA, reason='CUDA')
def test_max_is_deterministic_on_cuda_and_matches_the_cpu():
    """'max' on CUDA: eager and the CUDA graphs answer the same bits on every repeat (legacy's duplicate claims are
    whichever write the GPU kept), within float32 rounding of the CPU's - on rows full of duplicate claims."""
    import copy
    m = net()
    set_dup_reduce(m, 'max')
    cpu = NetEvaluator(copy.deepcopy(m), 'cpu')
    eager = NetEvaluator(copy.deepcopy(m).cuda(), 'cuda', graphs=False)
    fast = NetEvaluator(copy.deepcopy(m).cuda(), 'cuda', graphs=True)
    for B in (1, 64, 130):
        x, npl, off, act, kind = a_batch(m, B, 100 + B)
        e = eager.evaluate(x, npl, off, act, kind)
        f = fast.evaluate(x, npl, off, act, kind)
        close(e, f)
        close(cpu.evaluate(x, npl, off, act, kind), e)
        for _ in range(20):
            assert all(np.array_equal(u, v) for u, v in zip(e, eager.evaluate(x, npl, off, act, kind)))
            assert all(np.array_equal(u, v) for u, v in zip(f, fast.evaluate(x, npl, off, act, kind)))


# ------------------------------------------------------------------ the ONNX export (farm workers)

@pytest.mark.parametrize('packing', ['mask', 'packed'])
def test_the_export_answers_the_largest_claim(tmp_path, packing):
    """'max' exported (zero-onnx/1, mask and packed): ONNX Runtime's CPU answers torch's within 2e-5 on rows full of
    duplicate claims (onnx_parity's gate), every scatter unique (`scatter_writes`), metadata `dup_reduce` 'max'; the
    legacy export of the same weights has no such entry. 'compact' / 'host' refuse it."""
    pytest.importorskip('onnx')
    pytest.importorskip('onnxscript')
    pytest.importorskip('onnxruntime')
    from zero import onnx_export as OX, onnx_parity as OP
    from test_onnx import QUIET, kinds_of, rows_of
    m = net()
    legacy = OX.export(m, str(tmp_path / 'legacy.onnx'), packing, {'checkpoint': 'toy'})
    assert 'dup_reduce' not in legacy
    set_dup_reduce(m, 'max')
    path = str(tmp_path / f'{packing}.onnx')
    meta = OX.export(m, path, packing, {'checkpoint': 'toy'})
    assert meta['dup_reduce'] == 'max' and meta['unique_scatters'] == '1' and meta['opset'] == '17'
    B = 150
    x = tok_x(B, 3)
    rows = rows_of(x.numpy(), np.random.default_rng(1).integers(2, 5, B), kinds_of(B, 2), TOK_A)
    (_, *_), got = claims(m, x, torch.full((B,), 2))
    assert sum(bool(d) for _, i, _ in got for d in dup_rows(i, TOK_A)) >= 20
    sw = OX.scatter_writes(path, [OP.feeds(rows.take(np.arange(s))) for s in (1, 7, B)], threads=1)
    assert sw['scatters'] == 4 and sw['duplicates'] == 0 and sw['over'] == 0, sw
    ok, res = OP.run_gate(m, {packing: path}, [rows], sizes=(1, 7, 64, 150), threads=1, **QUIET)
    assert ok, [r for r in res if not r['pass']]
    assert max(r['max'] for r in res) < 2e-5
    with pytest.raises(ValueError, match='dup_reduce'):
        OX.ZeroOnnx(m, 'compact')


def test_scatter_max_is_torchs_amax(tmp_path):
    """`_scatter_max` (the export's form) against torch's scatter_reduce amax, bit for bit, ties included - in torch and
    through ONNX Runtime at opset 17, with every write to its own place."""
    pytest.importorskip('onnx')
    pytest.importorskip('onnxscript')
    ort = pytest.importorskip('onnxruntime')
    import onnx
    from zero import onnx_export as OX

    class M(torch.nn.Module):
        def forward(self, base, idx, val):
            return OX._scatter_max(base, idx, val)
    Bd = torch.export.Dim('B', min=1)
    ex = (torch.zeros(3, 9), torch.zeros(3, 12, dtype=torch.int64), torch.zeros(3, 12))
    prog = torch.onnx.export(M(), ex, None, dynamo=True, opset_version=OX.EXPORTER_OPSET,
                             input_names=['base', 'idx', 'val'],
                             dynamic_shapes={'base': {0: Bd}, 'idx': {0: Bd}, 'val': {0: Bd}}, verbose=False)
    mp = prog.model_proto
    OX.opset17(mp)
    path = str(tmp_path / 'max.onnx')
    onnx.save(mp, path)
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(7)
    for B in (1, 5, 64):
        base = rng.normal(size=(B, 9)).astype(np.float32)
        idx = rng.integers(0, 9, (B, 12))
        val = rng.integers(-3, 4, (B, 12)).astype(np.float32) * 0.5        # (ties within an index's claims)
        want = torch.from_numpy(base).scatter_reduce(1, torch.from_numpy(idx), torch.from_numpy(val), 'amax',
                                                     include_self=False).numpy()
        got = OX._scatter_max(torch.from_numpy(base), torch.from_numpy(idx), torch.from_numpy(val)).numpy()
        f = {'base': base, 'idx': idx, 'val': val}
        assert np.array_equal(got, want) and np.array_equal(sess.run(None, f)[0], want)
        u = OX.scatter_writes(path, [f], threads=1)
        assert u['duplicates'] == 0 and u['over'] == 0, u


@pytest.mark.parametrize('case', ['tails-id', 'blend'])
def test_the_v2_export_answers_the_largest_claim(tmp_path, case):
    """zero-onnx/2 (tails, gates, pointer overrides) with 'max': ONNX Runtime's CPU answers torch's within 2e-5 - with
    identity actions on the head items (`tails-id`: repeated identities and slot actions claim the same actions)."""
    pytest.importorskip('onnx')
    pytest.importorskip('onnxscript')
    pytest.importorskip('onnxruntime')
    from zero import onnx_parity as OP
    from test_onnx2 import QUIET, _items_only, export, gate_rows, kinds_of
    from test_tail_net import A, OFFERS, TOY_TAIL_TOKENS_BASE, rand_net, toy_batch
    ided = dict(TOY_TAIL_TOKENS_BASE, segments=[dict(TOY_TAIL_TOKENS_BASE['segments'][0], id_actions=8)])
    lay, gates, p_offer = {'tails-id': (ided, None, 0.0), 'blend': (OFFERS, [0.6, 0.45], 0.6)}[case]
    m = rand_net(lay, seed=3)
    if gates is not None:
        with torch.no_grad():
            m.gate.copy_(torch.tensor(gates))
    set_dup_reduce(m, 'max')
    path, meta = export(m, tmp_path)
    assert meta['zero_onnx'] == '2' and meta['dup_reduce'] == 'max'
    B = 150
    d = toy_batch(B, seed=11, p_offer=p_offer)
    if case == 'tails-id':
        d = {**d, **_items_only(d)}
    rows = gate_rows(m, d, kinds_of(B, 4))
    seen = []
    orig = m._scatter_policy
    m._scatter_policy = lambda b, i, v: seen.append(i.clone()) or orig(b, i, v)
    try:
        ok, res = OP.run_gate(m, {'packed': path}, [rows], sizes=(1, 7, 64, 150), threads=1, **QUIET)
    finally:
        del m._scatter_policy
    assert ok, [r for r in res if not r['pass']]
    assert max(r['max'] for r in res) < 2e-5
    if case == 'tails-id':
        assert sum(bool(r) for i in seen for r in dup_rows(i, A)) >= 20


# ------------------------------------------------------------------ the run: new networks and the flip at training

def test_a_run_builds_and_flips_its_network(tmp_path):
    """A token run started with dup_action_reduce 'max' builds its network with it; a legacy run's checkpoints have no
    entry, and resumed with 'max' its next training records it (the log says so)."""
    from test_loop import run_loop
    cuda_off = {'CUDA_VISIBLE_DEVICES': '-1'}
    tok = ['--arch', 'tokens', '--hidden', '16', '--blocks', '1', '--heads', '2']
    new = tmp_path / 'new'
    p = run_loop(new, '--iters', '1', *tok, '--dup-action-reduce', 'max', env=cuda_off)
    assert p.returncode == 0, p.stdout + p.stderr
    for it in (0, 1):
        assert load(str(new / 'models' / f'iter_000{it}.pt'))[0].dup_reduce == 'max'
    old = tmp_path / 'old'
    p = run_loop(old, '--iters', '1', *tok, env=cuda_off)
    assert p.returncode == 0, p.stdout + p.stderr
    assert 'dup_reduce' not in torch.load(str(old / 'models' / 'iter_0001.pt'), weights_only=False)['config']
    p = run_loop(old, '--iters', '2', *tok, '--dup-action-reduce', 'max', env=cuda_off)
    assert p.returncode == 0, p.stdout + p.stderr
    assert load(str(old / 'models' / 'iter_0001.pt'))[0].dup_reduce is None
    assert load(str(old / 'models' / 'iter_0002.pt'))[0].dup_reduce == 'max'
    with open(old / 'config.json') as f:
        assert json.load(f)['dup_action_reduce'] == 'max'
    assert "reduces duplicate action claims by 'max'" in p.stdout
    with open(old / 'log.txt', encoding='utf-8') as f:
        assert "iter 2: checkpoint 1 reduces duplicate action claims by 'max'" in f.read()
