"""Network arithmetic precision (docs/az.md 57): the run settings infer_precision / train_tf32 (opt-in, float32 by
default), the TF32 flag scoped and locked (never leaking into another thread's block), the evaluators' answers at
each precision, and the training feed's flat tails - the same batches bit for bit (57.3)."""
import threading

import numpy as np
import pytest
import torch

from zero import data, precision as PR, run as ZR
from zero import rows as R
from zero.infer import NetEvaluator

from test_tail_graphs import close, evaluate, gated_net, with_values
from test_tail_net import toy_batch
from test_tail_shards import LAY, WIDTHS, samples, write

CUDA = torch.cuda.is_available()
cuda = pytest.mark.skipif(not CUDA, reason='CUDA')


# ------------------------------------------------------------------ settings

def test_the_settings_are_opt_in_and_checked():
    assert ZR.DEFAULTS['infer_precision'] == 'fp32' and ZR.DEFAULTS['train_tf32'] == 0
    cfg = ZR.resolve_config(ZR.parse(['--run', 'x']), {})
    assert cfg['infer_precision'] == 'fp32' and cfg['train_tf32'] == 0
    cfg = ZR.resolve_config(ZR.parse(['--run', 'x', '--infer-precision', 'BF16', '--train-tf32', '1']), {})
    assert cfg['infer_precision'] == 'bf16' and cfg['train_tf32'] == 1
    with pytest.raises(SystemExit, match='infer_precision'):
        ZR.resolve_config(ZR.parse(['--run', 'x', '--infer-precision', 'fp16']), {})
    with pytest.raises(SystemExit, match='train_tf32'):
        ZR.resolve_config(ZR.parse(['--run', 'x', '--train-tf32', '2']), {})
    assert [PR.check(p) for p in (None, '', 'fp32', 'tf32', 'bf16')] == ['fp32', 'fp32', 'fp32', 'tf32', 'bf16']


def test_a_cpu_evaluator_stays_float32(capsys):
    net = gated_net()
    ev = NetEvaluator(net, 'cpu', precision='bf16')
    assert ev.precision == 'fp32' and not ev.amp and not ev.tf32
    assert 'float32 (CUDA only' in capsys.readouterr().err
    d = with_values(toy_batch(9, 1, p_tail=0.5, p_mark=0.5, p_offer=0.4), 1)
    want = evaluate(NetEvaluator(net, 'cpu'), d, net)
    assert all(np.array_equal(a, b) for a, b in zip(evaluate(ev, d, net), want))
    with pytest.raises(ValueError, match='infer_precision'):
        NetEvaluator(net, 'cpu', precision='half')


def test_the_tf32_flag_is_scoped_and_locked():
    m = torch.backends.cuda.matmul
    assert m.allow_tf32 is False
    with PR.matmul_tf32(True):
        assert m.allow_tf32 is True
        with PR.matmul_tf32(False):                 # (re-entrant: the same thread)
            assert m.allow_tf32 is False
        assert m.allow_tf32 is True
    assert m.allow_tf32 is False
    with pytest.raises(RuntimeError):
        with PR.matmul_tf32(True):
            raise RuntimeError('x')
    assert m.allow_tf32 is False
    # another thread's block waits for this one: it never sees this block's flag, nor this block its
    seen, inside = [], threading.Event()

    def other():
        inside.wait()
        with PR.matmul_tf32(False):
            seen.append(m.allow_tf32)
    t = threading.Thread(target=other)
    t.start()
    with PR.matmul_tf32(True):
        inside.set()
        t.join(0.3)
        assert t.is_alive() and m.allow_tf32 is True     # (blocked on the lock)
    t.join(5)
    assert seen == [False] and m.allow_tf32 is False


def test_the_training_scope():
    """No context without either setting (the step of before); the run's own value with either."""
    import contextlib
    assert isinstance(PR.train_scope({}, 'cuda'), contextlib.nullcontext)
    assert isinstance(PR.train_scope({'infer_precision': 'bf16'}, 'cuda'), contextlib.nullcontext)
    assert isinstance(PR.train_scope({'train_tf32': 1}, 'cpu'), contextlib.nullcontext)
    m = torch.backends.cuda.matmul
    with PR.train_scope({'train_tf32': 1}, 'cuda'):
        assert m.allow_tf32 is True
    with PR.train_scope({'infer_precision': 'tf32'}, 'cuda'):
        assert m.allow_tf32 is False
    assert m.allow_tf32 is False


# ------------------------------------------------------------------ the evaluators on CUDA

@cuda
@pytest.mark.parametrize('graphs', [True, False])
def test_each_precision_answers_near_float32_and_fp32_is_the_evaluator_of_before(graphs):
    """'fp32' is the evaluator of before bit for bit; 'tf32' / 'bf16' move the answers by their rounding (bf16 visibly,
    within bf16's tolerance), the graphs deterministically; the TF32 flag is off again after every capture and forward."""
    torch.manual_seed(0)
    net = gated_net(theta=(0.8, 0.4)).cuda()
    d = with_values(toy_batch(64, 3, p_tail=0.5, p_mark=0.5, p_offer=0.4), 3)
    base = evaluate(NetEvaluator(net, 'cuda', graphs=graphs), d, net)
    fp32 = NetEvaluator(net, 'cuda', graphs=graphs, precision='fp32')
    assert not fp32.amp and not fp32.tf32
    if graphs:
        assert all(np.array_equal(a, b) for a, b in zip(evaluate(fp32, d, net), base))
    else:                                             # (eager CUDA sums with atomics: run to run within float32)
        assert close(evaluate(fp32, d, net), base)
    for p, tol in (('tf32', 2e-2), ('bf16', 1e-1)):
        ev = NetEvaluator(net, 'cuda', graphs=graphs, precision=p)
        assert ev.precision == p and ev.amp == (p == 'bf16') and ev.tf32 == (p == 'tf32')
        if graphs:
            assert ev.fast.amp == (p == 'bf16') and ev.fast.tf32 == (p == 'tf32')
        got = evaluate(ev, d, net)
        assert close(base, got, tol), (p, [float(np.abs(a - b).max()) for a, b in zip(base, got)])
        if graphs:
            assert all(np.array_equal(a, b) for a, b in zip(got, evaluate(ev, d, net)))
        if p == 'bf16':
            assert not all(np.array_equal(a, b) for a, b in zip(got, base))
        assert torch.backends.cuda.matmul.allow_tf32 is False


@cuda
def test_tf32_graphs_keep_their_captured_kernels():
    """A replay of a graph captured under TF32 answers as its capture did, with the flag off at the replay."""
    torch.manual_seed(0)
    net = gated_net().cuda()
    a = torch.randn(512, 512, device='cuda')
    with PR.matmul_tf32(True):
        t = (a @ a).cpu()
    f = (a @ a).cpu()
    if torch.equal(t, f):
        pytest.skip('this GPU computes TF32 matmuls as float32')
    d = with_values(toy_batch(64, 5, p_tail=0.5, p_mark=0.5, p_offer=0.4), 5)
    ev = NetEvaluator(net, 'cuda', graphs=True, precision='tf32')
    first = evaluate(ev, d, net)                      # (captures)
    assert all(np.array_equal(a, b) for a, b in zip(first, evaluate(ev, d, net)))   # (replays, flag off)
    base = evaluate(NetEvaluator(net, 'cuda', graphs=True), d, net)
    assert not all(np.array_equal(a, b) for a, b in zip(first, base))


@cuda
def test_h2h_names_a_precision_other_than_float32():
    from zero import h2h
    net = gated_net().cuda()
    evs = h2h._evaluators(net, net, 'cuda', True, precision='bf16')
    assert all(ev.precision == 'bf16' for ev in evs.values())
    out = {}
    h2h._precision_out(out, evs)
    assert out == {'precision': 'bf16'}
    evs = h2h._evaluators(net, net, 'cuda', True)
    out = {}
    h2h._precision_out(out, evs)
    assert out == {} and all(ev.precision == 'fp32' for ev in evs.values())


# ------------------------------------------------------------------ the training feed's flat tails (57.3)

def same(a, b, path=''):
    """Batches `a` / `b` equal bit for bit: the same keys in the same order, each tensor's dtype, shape, device and
    bytes."""
    assert list(a) == list(b), (path, list(a), list(b))
    for k in a:
        u, v = a[k], b[k]
        if isinstance(u, dict):
            same(u, v, f'{path}{k}.')
            continue
        assert u.dtype == v.dtype and u.shape == v.shape and u.device == v.device, (path + k, u.dtype, v.dtype, u.shape,
                                                                                     v.shape)
        assert torch.equal(u.view(torch.uint8) if u.numel() and u.is_floating_point() else u,
                           v.view(torch.uint8) if v.numel() and v.is_floating_point() else v), path + k


@pytest.fixture(scope='module')
def S():
    return samples()


@pytest.mark.parametrize('dtype', [np.float16, np.float32])
def test_expand_tail_is_to_torch_bit_for_bit(S, dtype):
    t = R.Tail.from_arrays(S, WIDTHS)
    t = R.Tail(t.off, t.seg, t.slot, t.val.astype(dtype), t._src())
    wt = max(t.widths.values())
    rng = np.random.default_rng(0)
    none = np.flatnonzero(t.counts() == 0)
    assert len(none) and (t.counts() > 0).any()
    picks = [rng.integers(t.N, size=40), np.arange(t.N), none[:5], none[:1], np.zeros(0, np.int64)]
    devs = ['cpu'] + (['cuda'] if CUDA else [])
    for idx in picks:
        tb = t.take(idx)
        for dev in devs:
            want = {'tail': tb.to_torch(dev)}
            flat = {'tail': tb.to_torch_flat(dev)}
            assert flat['tail']['vflat'].dtype == torch.from_numpy(tb.val).dtype     # (as stored: no padding)
            same(R.expand_tail(flat, wt), want)
    b = {'x': torch.zeros(1)}
    assert R.expand_tail(b, wt) is b


def _window(tmp_path, S, sparse):
    paths = write(tmp_path, S, rows=50)
    return data.Window(paths, 72, 16, layout=LAY, **({'sparse': True} if sparse else {}))


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('prefetch', [0, 2])
@pytest.mark.parametrize('static', [False, True])
def test_the_feed_with_flat_tails_is_the_feed_of_before(tmp_path, S, sparse, prefetch, static):
    """`batches` with flat tails (auto on CUDA; forced on the CPU here) yields the batches of before bit for bit: dense
    and sparse (coo) windows, with and without the prefetch thread, the static (token) batches and the plain ones."""
    if sparse and not static:
        pytest.skip('a sparse window feeds the static training only (coo)')
    W = _window(tmp_path, S, sparse)
    tokens = (R.present_index(LAY), 8) if static else None
    devs = ['cpu'] + (['cuda'] if CUDA else [])
    for dev in devs:
        def feed(flat):
            g = data.batches(W, np.random.default_rng(5), 6, 24, dev, prefetch, tokens=tokens, coo=sparse,
                             flat_tail=flat)
            return list(g)
        old = feed(False)
        for flat in ([True, None] if dev == 'cuda' else [True]):
            new = feed(flat)
            assert len(new) == len(old) == 6
            for (i0, b0), (i1, b1) in zip(old, new):
                assert np.array_equal(i0, i1)
                same(b1, b0)
                assert b1['tail']['val'].device.type == torch.device(dev).type


def test_flat_tails_are_auto_on_cuda_only(tmp_path, S, monkeypatch):
    W = _window(tmp_path, S, False)
    calls = []
    real = W.batch

    def spy(*a, **k):
        calls.append(bool(k.get('flat_tail')))
        return real(*a, **k)
    monkeypatch.setattr(W, 'batch', spy)
    list(data.batches(W, np.random.default_rng(0), 2, 8, 'cpu', 0))
    assert calls == [False, False]
    if CUDA:
        calls.clear()
        list(data.batches(W, np.random.default_rng(0), 2, 8, 'cuda', 2))
        assert calls == [True, True]
