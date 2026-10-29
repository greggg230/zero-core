"""Distillation (docs/az.md 13): a NEW network - another size, even another architecture - trained to
compute what a teacher network computes, on a run's sample shards. The warm start of a bigger
network for a run that has plateaued (MTG Zero's Delver line, its docs/ENGINE.md 62): the student
starts where the teacher is instead of from scratch, then a run continues from it (`tools/fork_run.py
--source <run> --dest <fork> --net <student.pt>`: the run's window, iteration numbers and Elo
history, the student as its checkpoint K).

  teacher, _ = net.load(path, game=game)                           # the encoding guard (10.4)
  student = new_student(teacher, hidden=256, blocks=6, heads=8, ff=512, seed=1)
  W = data.Window(paths, student.features, A, budget_rows=...)     # the shards (this build's encoding)
  T = outputs(teacher, W, device)                                  # the teacher once per window row
  distill(student, W, T, steps=20000, ...)                         # the training (below)
  agreement(outputs(student, H, device), outputs(teacher, H, device), H)   # held-out rows H

`outputs` = a network's answers on every row of a window, as the search sees them: the policy as
probabilities over each row's legal actions (one float per option, aligned with the window's
`opt_action`), P(win) per seat present, the margin per seat.

The loss of a training step (rows b; w = `teacher_w`, vq = `value_q`; p_s, v_s, m_s the student's):
  policy   (1 - w) CE(pi_b, p_s) + w CE(p_t, p_s)                over the legal actions
  value    (1 - w) CE((1 - vq) z_b + vq q_b, v_s) + w CE(v_t, v_s)   over the seats present
  margin   margin_w ((1 - w) |sc_b - m_s|^2 + w |m_t - m_s|^2) / 2
w = 1 (the default) is pure distillation: the student learns the teacher's function - its policy over
every legal action, its value, its margin - rather than the self-play targets the teacher was trained
on (a recorded outcome z is one game's 0 / 1: the teacher's value is its estimate of the mean); w = 0
is the run's own supervised loss on the shards. AdamW, gradient-norm clip 5 (as the run), a linear
warm-up and a cosine decay of the learning rate to `lr_min`.

`agreement` (held-out rows: shards the student never trained on): KL(teacher || student) of the
policies over the legal actions (nats, mean over the decisions with at least 2 legal actions), the
share of those whose argmax agrees, the squared and absolute differences of P(win) (mean over the
seats present), the margin's squared difference; and each network's losses against the shards'
own targets (policy cross-entropy against the recorded pi, value cross-entropy against z, value
accuracy), so a student can be judged as a player of the data too.
"""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import batches
from .finite import NonFinite
from .infer import use_amp
from .net import make_net
from .protocol import MAX_SEATS


def new_student(teacher, arch=None, hidden=128, blocks=4, heads=4, ff=0, ch=64, seed=0, game=None):
    """A fresh network of the given size for the teacher's game (or `game`): `net.make_net` with this
    build's encoding (feature / action counts, the game package's token layout or board grid), its
    parameters drawn after torch.manual_seed(seed). `arch` default: the teacher's."""
    from . import games as G
    game = game or teacher.config.get('game', 'toy')
    arch = arch or teacher.config.get('arch', 'mlp')
    torch.manual_seed(seed)
    return make_net(G.info(game), game, arch, hidden, blocks, ch, heads, ff)


def _option_index(W, rows):
    """Option positions of window rows `rows` (in order) and their row numbers 0..len(rows)-1."""
    off = W.opt_off
    starts = off[rows]
    lens = off[rows + 1] - starts
    tot = int(lens.sum())
    pos = np.arange(tot) - np.repeat(np.cumsum(lens) - lens, lens) + np.repeat(starts, lens)
    return pos, np.repeat(np.arange(len(rows)), lens), lens


@torch.inference_mode()
def outputs(model, W, device=None, batch=1024):
    """`model`'s answers on every row of window `W` -> {'pi': float32 [options] (the policy over each
    row's legal actions, softmax of its logits there: aligned with W.opt_action), 'v': float32 [N,
    MAX_SEATS] (P(win) per seat, 0 for absent seats), 'm': float32 [N, MAX_SEATS] (margin)}. The
    model is moved to `device` when one is given; the rows go to the model's device."""
    if device is not None:
        model.to(device)
    device = next(model.parameters()).device
    model.eval()
    N = W.N
    pi = np.zeros(len(W.opt_action), np.float32)
    v = np.zeros((N, MAX_SEATS), np.float32)
    m = np.zeros((N, MAX_SEATS), np.float32)
    S = model.seats
    amp = use_amp(model, device)
    for lo in range(0, N, batch):
        rows = np.arange(lo, min(N, lo + batch))
        x = torch.from_numpy(W.features(rows).astype(np.float32)).to(device)
        n = torch.from_numpy(W.n_players[rows]).to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            plog, vlog, mar = model(x, n)
        pos, rid, _ = _option_index(W, rows)
        acts = torch.from_numpy(W.opt_action[pos]).to(device)
        r = torch.from_numpy(rid).to(device)
        lg = plog.float()[r, acts]
        mx = torch.full((len(rows),), -torch.inf, device=lg.device).scatter_reduce(0, r, lg, 'amax', include_self=True)
        e = torch.exp(lg - mx[r])
        s = torch.zeros(len(rows), device=lg.device).index_add_(0, r, e)
        pi[pos] = (e / s[r]).cpu().numpy()
        v[rows, :S] = torch.softmax(vlog.float(), 1).cpu().numpy()
        m[rows, :S] = mar.float().cpu().numpy()
    v[np.arange(MAX_SEATS)[None, :] >= W.n_players[:, None]] = 0.0
    return {'pi': pi, 'v': v, 'm': m}


class Taught:
    """A window whose batches also carry a teacher's answers (`outputs` of the same window): the dense
    `tpi` [B, actions] over the legal actions, `tv`, `tm` [B, MAX_SEATS]. Has what `data.batches`
    needs (N, batch), so training batches can be prefetched as the run's are."""

    def __init__(self, W, T):
        self.W, self.T, self.N = W, T, W.N

    def batch(self, idx, device='cpu'):
        b = self.W.batch(idx, device)
        pos, rid, _ = _option_index(self.W, idx)
        tpi = np.zeros((len(idx), self.W.A), np.float32)
        tpi[rid, self.W.opt_action[pos]] = self.T['pi'][pos]
        tpi /= np.maximum(tpi.sum(1, keepdims=True), 1e-8)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        b.update(tpi=t(tpi), tv=t(self.T['v'][idx]), tm=t(self.T['m'][idx]))
        return b


def losses(model, b, teacher_w=1.0, value_q=0.5, margin_w=0.5, amp=False):
    """The distillation loss of batch `b` (a `Taught` batch; module docstring) -> (total, terms: the
    policy / value / margin losses against the teacher and against the shards' targets, the policy KL
    to the teacher)."""
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
        plog, vlog, mp = model(b['x'], b['n_players'])
    plog, vlog, mp = plog.float(), vlog.float(), mp.float()
    S = model.seats
    logp = F.log_softmax(plog.masked_fill(~b['mask'], -1e9), dim=1)
    pt = b['tpi']
    p_t = -(pt * logp).sum(1).mean()
    p_d = -(b['pi'] * logp).sum(1).mean()
    ent_t = -(pt * torch.log(pt.clamp(min=1e-12))).sum(1).mean()
    lv = F.log_softmax(vlog, dim=1).clamp(min=-1e4)
    v_t = -(b['tv'][:, :S] * lv).sum(1).mean()
    zt = ((1 - value_q) * b['z'] + value_q * b['q'])[:, :S]
    v_d = -(zt * lv).sum(1).mean()
    present = (torch.arange(S, device=plog.device)[None, :] < b['n_players'][:, None]).float()
    m_t = (((mp - b['tm'][:, :S]) ** 2) * present).sum(1).mean() / 2
    m_d = (((mp - b['sc'][:, :S]) ** 2) * present).sum(1).mean() / 2
    w = float(teacher_w)
    total = (1 - w) * (p_d + v_d + margin_w * m_d) + w * (p_t + v_t + margin_w * m_t)
    return total, {'policy_t': p_t, 'value_t': v_t, 'margin_t': m_t, 'policy_d': p_d, 'value_d': v_d, 'margin_d': m_d,
                   'kl_t': p_t - ent_t}


def lr_at(step, steps, lr, lr_min, warmup):
    """Linear warm-up over `warmup` steps, then a cosine from `lr` down to `lr_min` at the last step."""
    if warmup and step < warmup:
        return lr * (step + 1) / warmup
    span = max(1, steps - warmup)
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * min(1.0, (step - warmup) / span)))


def distill(student, W, T, steps, batch=512, lr=5e-4, lr_min=1e-5, warmup=500, wd=1e-4, teacher_w=1.0, value_q=0.5,
            margin_w=0.5, device='cpu', seed=0, prefetch=2, opt=None, start=0, every=100, on_report=None,
            on_step=None, check=None):
    """Train `student` (in place) for steps start..steps-1 on window `W` with the teacher's answers `T`
    (`outputs`) by the loss of the module docstring. `opt`: an AdamW state to continue (a resumed
    distillation: `start` its step), else fresh. Every `every` steps `on_report(step, means of the
    loss terms over those steps, lr)`; `on_step(step, model, opt)` after each step (saving, held-out
    evaluations); `check()` before each (PAUSE). A NaN / inf loss or gradient norm raises
    finite.NonFinite before the step. -> (the AdamW optimizer, the last report)."""
    student.to(device).train()
    o = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=wd)
    if opt is not None:
        o.load_state_dict(opt)
    amp = use_amp(student, device)
    rng = np.random.default_rng(seed * 7919 + start)
    acc, k, last = {}, 0, {}
    feed = batches(Taught(W, T), rng, steps - start, batch, device, prefetch)
    try:
        for step in range(start, steps):
            if check is not None:
                check()
            g_lr = lr_at(step, steps, lr, lr_min, warmup)
            for g in o.param_groups:
                g['lr'] = g_lr
            _, b = next(feed)
            total, terms = losses(student, b, teacher_w, value_q, margin_w, amp)
            o.zero_grad(set_to_none=True)
            total.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            if not bool(torch.isfinite(torch.stack([total.detach(), gnorm.float()])).all()):
                raise NonFinite(f'distillation step {step + 1}: non-finite loss {float(total):.5g} or gradient norm '
                                f'{float(gnorm):.5g}; the student is not saved from here')
            o.step()
            for key, val in terms.items():           # (summed on the device: one sync per report)
                acc[key] = acc.get(key, 0.0) + val.detach()
            k += 1
            if on_step is not None:
                on_step(step + 1, student, o)
            if (step + 1) % every == 0 or step + 1 == steps:
                last = {key: round(float(val) / k, 5) for key, val in acc.items()}
                if on_report is not None:
                    on_report(step + 1, last, g_lr)
                acc, k = {}, 0
    finally:
        feed.close()
    student.eval()
    return o, last


def _segments(lens):
    """(starts of the non-empty segments, their row numbers) of option counts `lens`."""
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
    keep = lens > 0
    return starts[keep], np.flatnonzero(keep)


def _first_max(p, starts, lens):
    """Per non-empty segment, the position (within the flat array) of its first maximum."""
    rid = np.repeat(np.arange(len(starts)), lens)
    mx = np.maximum.reduceat(p, starts)
    pos = np.arange(len(p))
    return np.minimum.reduceat(np.where(p >= mx[rid], pos, len(p)), starts)


def agreement(S, T, W):
    """How network outputs `S` (`outputs` of window `W`) agree with the teacher's `T` on W's rows, and
    how both do against W's own targets -> a dict (module docstring)."""
    lens = np.diff(W.opt_off)
    starts, rows = _segments(lens)
    ln = lens[rows]
    eps = 1e-12
    ps, pt = np.maximum(S['pi'], eps), np.maximum(T['pi'], eps)
    kl = np.add.reduceat(T['pi'] * (np.log(pt) - np.log(ps)), starts)
    dec = ln >= 2                                   # decisions (a single legal action is no choice)
    top = _first_max(S['pi'], starts, ln) == _first_max(T['pi'], starts, ln)
    n = W.n_players
    present = np.arange(MAX_SEATS)[None, :] < n[:, None]
    dv = (S['v'] - T['v'])[present]
    dm = (S['m'] - T['m'])[present]

    pin = W.pi / np.maximum(np.repeat(np.add.reduceat(W.pi, starts), ln), 1e-8)   # the float16 targets renormalised

    def data_losses(O):
        ce = -np.add.reduceat(pin * np.log(np.maximum(O['pi'], eps)), starts)
        z = W.z / np.maximum(W.z.sum(1, keepdims=True), 1e-8)
        vce = -(z * np.log(np.maximum(O['v'], eps)) * present).sum(1)
        acc = (np.where(present, O['v'], -1).argmax(1) == z.argmax(1))
        return {'policy_ce': round(float(ce[dec].mean()) if dec.any() else 0.0, 5),
                'value_ce': round(float(vce.mean()), 5), 'value_acc': round(float(acc.mean()), 4)}
    return {'rows': int(W.N), 'decisions': int(dec.sum()),
            'policy_kl': round(float(kl[dec].mean()) if dec.any() else 0.0, 6),
            'policy_kl_p90': round(float(np.quantile(kl[dec], 0.9)) if dec.any() else 0.0, 6),
            'top1': round(float(top[dec].mean()) if dec.any() else 1.0, 4),
            'value_mse': round(float((dv ** 2).mean()), 7), 'value_mae': round(float(np.abs(dv).mean()), 5),
            'margin_mse': round(float((dm ** 2).mean()), 6),
            'net': data_losses(S), 'teacher': data_losses(T)}


def describe(model):
    """A network's size in a few keys (for logs and the checkpoint's record)."""
    c = model.config
    out = {'arch': c.get('arch', 'mlp'), 'params': model.n_params()}
    if out['arch'] == 'tokens':
        out.update(d=c['d'], layers=c['layers'], heads=c['heads'], ff=c['ff'])
    elif out['arch'] == 'grid':
        out.update(ch=c['ch'], blocks=c['blocks'])
    else:
        out.update(hidden=c['hidden'], blocks=c['blocks'])
    return out


def timed(fn, *a, **k):
    t0 = time.time()
    r = fn(*a, **k)
    return r, time.time() - t0
