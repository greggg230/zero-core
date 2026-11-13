"""A widen's gated segment in training (docs/az.md 43.14): the new segment's gate starts exactly at 0 (the widen is
exact) and its forward weight stays g = max(theta, 0); a shut gate's shadow back-propagates the segment as if its
weight were `GATE_SHADOW`, so a segment carrying information learns it and its gate opens, while one carrying noise
may stay shut. A supervised toy on toy_tail's layouts: the value head's target depends on a global float (what the
old network reads) and, in the informative case, on a bit only the new gated segment's tokens carry - as the parity
of a one-hot code's index, which no feature of the widen's random input map carries on average (its gate's gradient
at g = 0 is then noise: without the shadow the segment never learns, and the gate wanders below 0 and stays)."""
import numpy as np
import pytest
import torch

from zero import net as N

F_, A_ = 72, 16
ITEMS = {"name": "items", "offset": 8, "count": 4, "width": 16, "present": 0, "id": {"at": 6, "radix": [10]},
         "actions": [[4, 1]], "tail": True}
CODE = {"name": "code", "offset": None, "count": 0, "width": 12, "present": 0, "id": None, "actions": [], "tail": True}
BASE = {"vocab": 16, "name": "toy_tail", "version": 1, "option_pointers": True, "max_row_tokens": 64,
        "segments": [ITEMS], "bags": [], "globals": [{"name": "globals", "offset": 0, "count": 8}]}
GATED = dict(BASE, segments=[ITEMS, dict(CODE, gate=True)])


def rows(rng, B, info, code=True):
    """B rows: globals (float 0 the old information), present items; with `code` 1-2 tokens of the gated segment a
    row, floats [present, slot / 8, 0, 0, one-hot over 4..11] whose hot index's parity is the row's bit (`info`) or an
    independent one (noise). -> x, n_players, the target win probability of seat 0, the tail."""
    x = np.zeros((B, F_), np.float32)
    x[:, :8] = rng.normal(size=(B, 8))
    pres = rng.random((B, 4)) < 0.6
    for s in range(4):
        o = 8 + 16 * s
        r = np.flatnonzero(pres[:, s])
        x[r, o] = 1
        x[r, o + 1:o + 6] = rng.normal(size=(len(r), 5))
        x[r, o + 6 + rng.integers(0, 10, len(r))] = 1
    bit = rng.integers(0, 2, B)
    y = 1 / (1 + np.exp(-(1.5 * x[:, 0] + (3.0 * (2 * bit - 1) if info else 0.0))))
    rr, sl, vals = [], [], []
    if code:
        for b in range(B):
            for s in sorted(rng.choice(8, size=int(rng.integers(1, 3)), replace=False)):
                v = np.zeros(16, np.float32)
                bb = bit[b] if info else rng.integers(0, 2)
                v[0], v[1], v[4 + 2 * int(rng.integers(0, 4)) + int(bb)] = 1, s / 8, 1.0
                rr.append(b)
                sl.append(int(s))
                vals.append(v)
    tail = {'row': torch.tensor(rr, dtype=torch.long), 'seg': torch.ones(len(rr), dtype=torch.long),
            'slot': torch.tensor(sl, dtype=torch.long),
            'val': torch.from_numpy(np.stack(vals)) if vals else torch.zeros(0, 16)}
    return torch.from_numpy(x), torch.full((B,), 2, dtype=torch.long), torch.from_numpy(y.astype(np.float32)), tail


NO_OV = {k: torch.zeros(0, dtype=torch.long) for k in ('row', 'act', 'ref', 'col')}


def loss_of(m, x, n, y, tail, static=False):
    if static:
        from zero.infer import token_index_tail, token_presence
        P = token_presence(x.numpy(), m.present_index())
        tc = np.bincount(tail['row'].numpy(), minlength=len(P)) if len(tail['row']) else np.zeros(len(P), np.int64)
        cnt = P.sum(1) + tc
        full = token_index_tail(P, tc, m.n_tok, np.zeros((len(P), int(cnt.max())), np.int64))
        _, vlog, _ = m.forward_static(x, n, torch.from_numpy(full), torch.from_numpy(cnt), tail=tail, ov=NO_OV)
    else:
        _, vlog, _ = m(x, n, tail=tail, ov=NO_OV)
    lp = torch.log_softmax(vlog[:, :2], 1)
    return -(y * lp[:, 0] + (1 - y) * lp[:, 1]).mean()


def train(m, steps, rng, info, code=True, static=False, lr=2e-3):
    gates = [m.gate] if m.gated else []
    rest = [p for p in m.parameters() if all(p is not g for g in gates)]
    o = torch.optim.AdamW([{'params': rest, 'weight_decay': 1e-4}] + ([{'params': gates, 'weight_decay': 0.0}]
                                                                       if gates else []), lr=lr)
    m.train()
    trace = []
    for _ in range(steps):
        x, n, y, tl = rows(rng, 128, info, code)
        o.zero_grad()
        loss_of(m, x, n, y, tl, static).backward()
        o.step()
        if m.gated:
            trace.append(float(m.gate.detach()[0]))
    m.eval()
    return trace


def held_out(m, info, code=True):
    with torch.no_grad():
        return float(loss_of(m, *rows(np.random.default_rng(777), 2048, info, code)))


def old_network(seed):
    torch.manual_seed(seed)
    m = N.TokenNet(F_, A_, BASE, d=16, layers=2, heads=2, ff=32, head=16, game='toy')
    train(m, 300, np.random.default_rng(seed), True, code=False)
    return m


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_an_informative_gated_segment_opens_and_beats_the_old_network(seed):
    """The widen exact at step 0 (inference on rows with the new tokens, static at the same token count); with the
    shadow, training opens the gate and reads the segment's bit: the loss falls far below the old network's best
    without it (the bit is worth ~0.4 nats a row)."""
    old = old_network(seed)
    base_loss = held_out(old, True, code=False)
    w = N.widen(old, layout=GATED)
    assert float(w.gate.detach()[0]) == 0.0 and w._shadow() == N.GATE_SHADOW > 0
    with torch.no_grad():
        x, n, y, tl = rows(np.random.default_rng(5), 64, True)
        none = {'row': torch.zeros(0, dtype=torch.long), 'seg': torch.zeros(0, dtype=torch.long),
                'slot': torch.zeros(0, dtype=torch.long), 'val': torch.zeros(0, 16)}
        assert all(torch.equal(a, b) for a, b in zip(old.eval()(x, n, tail=none, ov=NO_OV), w.eval()(x, n, tail=none, ov=NO_OV)))
    trace = train(w, 300, np.random.default_rng(100 + seed), True)
    loss = held_out(w, True)
    assert trace[-1] > 0.02, trace[::30]
    assert loss < base_loss - 0.2, (loss, base_loss)


@pytest.mark.parametrize('seed', [0, 1])
def test_a_noise_segment_does_no_harm(seed):
    """A gated segment of pure noise: the shadow does not force it open (its gate may stay shut or near it) and the
    widened network is no worse than the old one."""
    old = old_network(seed)
    base_loss = held_out(old, False, code=False)
    w = N.widen(old, layout=GATED)
    trace = train(w, 300, np.random.default_rng(200 + seed), False)
    assert max(trace) < 0.25, trace[::30]
    assert held_out(w, False) < base_loss + 0.02


def test_the_static_step_learns_a_shut_segment_too():
    """forward_static's training (train_static): the same shadow - the gate opens, the bit is read."""
    old = old_network(3)
    base_loss = held_out(old, True, code=False)
    w = N.widen(old, layout=GATED)
    trace = train(w, 300, np.random.default_rng(103), True, static=True)
    assert trace[-1] > 0.02 and held_out(w, True) < base_loss - 0.2, (trace[::30], held_out(w, True), base_loss)
