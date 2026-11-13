"""Variable-length rows under CUDA autocast (bfloat16; docs/az.md 43.6, 43.9): the pointer blend into a gated segment is
exactly the generic logit at g = 0 and exactly the pointer logit at g = 1; training steps (eager and the static
forward, gates at 0 - the explicit attention - and > 0, tails, overrides into gated and ungated tokens) run with finite
gradients, lower the loss and match float32 to bfloat16's precision; the graphs' bfloat16 mode takes tails; run.py's
training of a token network on toy_tail with autocast on. (`run.py` and `zero.distill` train a token network in float32
- `infer.use_amp` is the conv network's -, so these cover any caller that turns autocast on, and the measurements' amp.)
"""
import numpy as np
import pytest
import torch

from zero import infer
from zero.net import TBlock, widen

from test_tail_net import (A, OFFERS_SEG, TOY_GATED, TOY_TAIL_TOKENS, TOY_TAIL_TOKENS_BASE, ov_of, rand_net, static_in,
                           tail_of, toy_batch, tx, unpack_tok)

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason='autocast on CUDA')
BF16 = torch.bfloat16
LAY = dict(TOY_TAIL_TOKENS, segments=TOY_TAIL_TOKENS['segments'] + [OFFERS_SEG])   # marks, offers gated by the widen


def gated_net(theta, seed=5):
    m = rand_net(TOY_TAIL_TOKENS_BASE, seed)
    w = widen(m, layout=LAY)
    with torch.no_grad():
        w.gate.data[:] = torch.tensor(theta)
        w.ptr.weight.normal_(0, 0.5)
        w.ptr.bias[1] = 2.0
    return w.cuda()


def batch(seed=6, B=48):
    d = toy_batch(B, seed, p_tail=0.6, p_mark=0.5, p_offer=0.5, p_gen=0.3)
    toks = [unpack_tok(t)[0] for t in d['ov_tok'] if t != -1]
    assert 0 in toks and 2 in toks                   # overrides into ungated items and the gated offers
    return d


def run(w, d, static, aux=True):
    x, n = tx(d, 'cuda')
    tl, ov = tail_of(w, d, 'cuda'), ov_of(w, d, 'cuda')
    if static:
        full, ntok = static_in(w, d, None, 'cuda')
        return w.forward_static(x, n, full, ntok, aux=aux, tail=tl, ov=ov)
    return w(x, n, aux=aux, tail=tl, ov=ov)


def loss_of(o):
    """A loss of every head (the value's as log-probabilities: its absent seats' logits are masked)."""
    return (o[0][:, :A].float().logsumexp(1).mean() - torch.log_softmax(o[1].float(), 1)[:, 0].mean()
            + o[2].float().pow(2).mean() + o[3].float().pow(2).mean())


def heads_of(o):
    """(policy logits, value probabilities, margin, auxiliary outputs) as float32, for comparisons."""
    return o[0][:, :A].float(), torch.softmax(o[1].float(), 1), o[2].float(), o[3].float()


def _eager_full(w, d):
    """The eager forward's extended positions of each row's packed tokens (`TokenNet._pack`)."""
    x, _ = tx(d, 'cuda')
    with torch.no_grad():
        return w._pack(x, tail=tail_of(w, d, 'cuda'))[2]


@pytest.mark.parametrize('static', [False, True])
@pytest.mark.parametrize('train', [False, True])
def test_the_gated_blend_is_exact_under_bf16(static, train):
    """Under bfloat16 autocast an override into a gated segment's token scores exactly the generic logit at g = 0 and
    exactly its pointer logit at g = 1 (the generic head's and the pointer head's outputs caught, compared bit for
    bit), in training (gradients on) and inference."""
    d = batch()
    ov = ov_of(gated_net([0.4, 0.0]), d, 'cuda')
    for theta, which in ((0.0, 'gen'), (1.0, 'ptr')):
        w = gated_net([0.4, theta])
        w.train(train)
        got = {}
        hook = w.gen.register_forward_hook(lambda m, i, o: got.__setitem__('gen', o))
        pointer = w._pointer
        w._pointer = lambda hid: got.__setitem__('ptr', pointer(hid)) or got['ptr']
        with torch.set_grad_enabled(train), torch.autocast('cuda', dtype=BF16):
            out = run(w, d, static, aux=False)
        hook.remove()
        del w._pointer
        tl = tail_of(w, d, 'cuda')
        off = (ov['ref'] >= w.n_tok) & (tl['seg'][(ov['ref'] - w.n_tok).clamp(min=0)] == 2)
        rows, acts, refs = ov['row'][off], ov['act'][off], ov['ref'][off]
        # (eager's policy accumulates its chunks in float32: bfloat16 logits upcast exactly)
        assert len(rows) > 5 and out[0].dtype == (BF16 if static else torch.float32)
        pol = out[0].float()
        if which == 'gen':
            assert torch.equal(pol[rows, acts], got['gen'][rows, acts].float())  # (B <= 64: eager's one chunk)
        else:
            full = static_in(w, d, None, 'cuda')[0] if static else _eager_full(w, d)
            pos = torch.stack([(full[r] == ref).nonzero()[0, 0] for r, ref in zip(rows.tolist(), refs.tolist())])
            P = got['ptr']                            # [B, T, columns] (the offers' column: 1)
            assert P.dtype == BF16 and torch.equal(pol[rows, acts], P[rows, pos, 1].float())


@pytest.mark.parametrize('static', [False, True])
@pytest.mark.parametrize('theta', [(0.0, 0.0), (0.5, 0.3)])
def test_training_steps_under_bf16(static, theta, monkeypatch):
    """A few AdamW steps under bfloat16 autocast (eager and the static forward; gates exactly 0 - the explicit attention -
    and > 0; tails; overrides into ungated items, gated offers and generic ones): every gradient finite, the gates'
    non-zero, the loss lower after them, the first step's outputs and loss float32's to bfloat16's precision."""
    calls = []
    orig = TBlock.forward_weighted
    monkeypatch.setattr(TBlock, 'forward_weighted', lambda self, *a: calls.append(1) or orig(self, *a))
    d = batch(7, 64)
    w = gated_net(list(theta)).train()
    with torch.no_grad():
        ref = run(w, d, static)
    opt = torch.optim.AdamW(w.parameters(), lr=3e-3)
    losses = []
    for step in range(6):
        with torch.autocast('cuda', dtype=BF16):
            o = run(w, d, static)
            loss = loss_of(o)
        if step == 0:
            for a, b in zip(heads_of(ref), heads_of(o)):
                assert (a - b).abs().max() <= 0.05 * max(1.0, a.abs().max().item()), (a - b).abs().max()
            assert abs(loss.item() - loss_of(ref).item()) < 0.02 * abs(loss_of(ref).item()) + 1e-3
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for name, p in w.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), name
        assert w.gate.grad is not None and (w.gate.grad != 0).all()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], losses
    assert bool(calls) == (theta[0] == 0.0)          # the explicit attention exactly while a present gate is 0


@pytest.mark.parametrize('with_ov', [False, True])
def test_static_training_at_a_shut_gate_under_bf16(with_ov):
    """Lane Z's case: a train_static step under bfloat16 with every gate exactly 0 and gated tokens present (the explicit
    attention's node, float32 inside), with and without overrides: finite gradients, the gates' non-zero."""
    d = batch(8, 64)
    if not with_ov:
        d = dict(d, ov_off=np.zeros(len(d['ov_off']), np.int64), ov_opt=np.zeros(0, np.int32),
                 ov_tok=np.zeros(0, np.int64))
    w = gated_net([0.0, 0.0]).train()
    with torch.autocast('cuda', dtype=BF16):
        loss = loss_of(run(w, d, True))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in w.parameters() if p.grad is not None)
    assert (w.gate.grad != 0).all()


def test_the_graphs_bf16_mode_takes_tails():
    """`TokenGraphRunner(amp=True)` (the measurements' bfloat16 graphs) with tails, gated tokens and overrides:
    eager float32's answers to bfloat16's precision."""
    from test_tail_graphs import evaluate, with_values
    w = gated_net([0.5, 0.0]).eval()
    d = with_values(batch(9, 70), 9)
    eager = infer.NetEvaluator(w, 'cuda', graphs=False)
    ev = infer.NetEvaluator(w, 'cuda', graphs=True)
    ev.fast = infer.TokenGraphRunner(w, amp=True)
    want, got = evaluate(eager, d, w), evaluate(ev, d, w)
    assert all(np.isfinite(g).all() for g in got)
    assert all(np.abs(a - b).max() < 0.05 for a, b in zip(want, got))


@pytest.mark.parametrize('static', [0, 1])
def test_run_trains_a_gated_toy_tail_network_under_autocast(tmp_path, monkeypatch, static):
    """run.py's training (`Run.train_model`) of a token network with gates on toy_tail on CUDA, with autocast forced on
    (`infer.use_amp` is the conv network's: a token network trains in float32 otherwise): the steps run, the losses,
    the gates and the trained weights finite."""
    from zero import package
    from zero import run as RUN
    from test_tail_train import gated_run
    monkeypatch.setitem(package.CORE.tokens, 'toy_tail', TOY_GATED)
    monkeypatch.setattr(RUN, 'use_amp', lambda model, device: True)
    R, out = gated_run(tmp_path, 0.0, train_device='cuda', train_static=static)
    model, ost, stats = R.train_model(1, out['samples'])
    s = stats()
    assert s['device'] == 'cuda' and s['steps'] >= 10
    assert all(np.isfinite(s[k]) for k in ('policy', 'value', 'margin'))
    assert set(s['tail']['gates']) == {'marks'} and np.isfinite(s['tail']['gates']['marks'])
    assert all(torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point())
