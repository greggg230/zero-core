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

Auxiliary heads (docs/az.md 15, 13.6): a teacher with one (config `aux`: MTG Zero's belief) gets a
student with the same head (`new_student`: its layout, so a run that trains or reads the targets - the
root's belief, `*_belief_tilt` - loads the student as it loaded the teacher), distilled too. `outputs`
then also has 'aux' [N, W] (the head's predictions, `net.aux_predict`: a multinomial target's
distribution, a poisson target's rates, an mse target's values) and the window carries the shards' own
targets (`Window(aux_width=W)`). The loss adds, per target, aux_w x ((1 - wa) L(target_b) + wa
L(teacher's prediction)) with L the target's own loss (`auxtarget.losses`: for a multinomial the
cross-entropy to the distribution, i.e. the KL to it plus its entropy) and wa = `aux_teacher_w`
(default: `teacher_w`, so w = 1 is pure distillation of every head and w = 0 the run's own loss with
aux_<name>_w = aux_w, tested). The shards' term counts the rows with a target (the run's rule); the
teacher's term every row except a multinomial target's rows whose shard says the segment is empty (an
empty hand: n x p = 0 whatever p, so the teacher's p there was never trained and means nothing) -
including rows of shards recorded without targets, where the teacher's prediction is the only label.
`agreement` adds per target ('aux'): a multinomial's KL(teacher || student) on the teacher's rows (mean,
p90, per kind of row) and the top-1 agreement; another loss's squared difference; and each network's
loss against the shards' targets (a multinomial's `hit` too: the run's `train.aux_<name>` metrics).

Value-only rows (docs/az.md 16: `selfplay_view_samples`, no options): as the run's training, a window
that has them takes the policy terms (and the teacher's entropy) as means over each batch's decision
rows only, the value and margin terms over every row (`distill` decides from the window, as
`run.train_model` does); a window without them trains as before, bit for bit. `agreement` then reports
the value and margin agreement of each kind of row separately ('decision', 'view').
"""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import auxtarget as AUX
from .data import batches
from .finite import NonFinite
from .infer import use_amp
from .net import aux_layout, aux_predict, make_net
from .protocol import MAX_SEATS


def new_student(teacher, arch=None, hidden=128, blocks=4, heads=4, ff=0, ch=64, seed=0, game=None, aux='teacher'):
    """A fresh network of the given size for the teacher's game (or `game`): `net.make_net` with this
    build's encoding (feature / action counts, the game package's token layout or board grid), its
    parameters drawn after torch.manual_seed(seed). `arch` default: the teacher's. `aux`: its auxiliary
    targets (`net.aux_layout`), default the teacher's head (None / []: none); the head is built after
    every other parameter, so the rest is drawn as for a student without one."""
    from . import games as G
    game = game or teacher.config.get('game', 'toy')
    arch = arch or teacher.config.get('arch', 'mlp')
    if isinstance(aux, str) and aux == 'teacher':
        aux = teacher.config.get('aux')
    if teacher.config.get('readers'):
        # (docs/az.md 36: not built yet - a student without them would silently drop the teacher's input path)
        raise ValueError('distillation of a network with readers is not supported yet (docs/az.md 36)')
    torch.manual_seed(seed)
    return make_net(G.info(game), game, arch, hidden, blocks, ch, heads, ff, aux=aux_layout(aux) or None)


def aux_width(model):
    """The total width of `model`'s auxiliary head (0: none)."""
    return sum(w for _, w, _ in (getattr(model, 'aux_layout', None) or []))


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
    MAX_SEATS] (P(win) per seat, 0 for absent seats), 'm': float32 [N, MAX_SEATS] (margin)}; a model
    with an auxiliary head also 'aux': float32 [N, W] (its predictions, `net.aux_predict`, in its
    layout's order: what `NetEvaluator.aux` answers) and 'aux_layout'. The model is moved to `device`
    when one is given; the rows go to the model's device."""
    if device is not None:
        model.to(device)
    device = next(model.parameters()).device
    model.eval()
    N = W.N
    pi = np.zeros(len(W.opt_action), np.float32)
    v = np.zeros((N, MAX_SEATS), np.float32)
    m = np.zeros((N, MAX_SEATS), np.float32)
    S = model.seats
    layout = getattr(model, 'aux_layout', None) or []
    aux = np.zeros((N, aux_width(model)), np.float32) if layout else None
    amp = use_amp(model, device)
    for lo in range(0, N, batch):
        rows = np.arange(lo, min(N, lo + batch))
        x = torch.from_numpy(W.features(rows).astype(np.float32)).to(device)
        n = torch.from_numpy(W.n_players[rows]).to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            if layout:
                plog, vlog, mar, raw = model(x, n, aux=True)
            else:
                plog, vlog, mar = model(x, n)
        if layout:
            pred = aux_predict(layout, raw)
            aux[rows] = torch.cat([pred[name] for name, _, _ in layout], 1).cpu().numpy()
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
    if layout:
        return {'pi': pi, 'v': v, 'm': m, 'aux': aux, 'aux_layout': [list(t) for t in layout]}
    return {'pi': pi, 'v': v, 'm': m}


class Taught:
    """A window whose batches also carry a teacher's answers (`outputs` of the same window): the dense
    `tpi` [B, actions] over the legal actions, `tv`, `tm` [B, MAX_SEATS] (and with an auxiliary head
    `taux` [B, W]: its predictions). Has what `data.batches` needs (N, batch), so training batches can
    be prefetched as the run's are."""

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
        if 'aux' in self.T:
            b['taux'] = t(self.T['aux'][idx])
        return b


def teacher_aux_rows(layout, target, mask):
    """Per auxiliary target, the rows [B] (float) of the teacher's term (module docstring): every row
    but those of a multinomial target whose shard has the target (`mask` 1) with an empty segment."""
    out = {}
    for name, loss, sl in AUX.columns(layout):
        if loss == 'multinomial':
            out[name] = 1.0 - mask * (target[:, sl].float().sum(1) <= 0).float()
        else:
            out[name] = torch.ones_like(mask)
    return out


def aux_losses(layout, raw, b, aux_teacher_w):
    """The auxiliary terms of batch `b` (the teacher's predictions `taux`, the shards' `aux` /
    `aux_mask`; `raw` the student's raw outputs) -> {name: (blended loss, its loss against the teacher,
    against the shards, a multinomial's KL to the teacher or None)}: each target's own loss
    (`auxtarget.losses`) with the teacher's prediction as its target on `teacher_aux_rows` and with the
    shards' on their rows with a target, blended (1 - wa) shards + wa teacher (wa = `aux_teacher_w`).
    The KL: the cross-entropy minus the teacher's entropy, over the teacher's rows."""
    wa = float(aux_teacher_w)
    rows = teacher_aux_rows(layout, b['aux'], b['aux_mask'])
    data = AUX.losses(layout, raw, b['aux'], b['aux_mask'])
    out = {}
    for name, loss, sl in AUX.columns(layout):
        tt = b['taux'][:, sl].float()
        lt = AUX.losses([[name, sl.stop - sl.start, loss]], raw[:, sl], tt, rows[name])[name]['loss']
        kl = None
        if loss == 'multinomial':
            p = tt / tt.sum(1, keepdim=True).clamp(min=1e-8)
            ent = -(p * torch.log(p.clamp(min=1e-12))).sum(1)
            kl = lt - (ent * rows[name]).sum() / rows[name].sum().clamp(min=1.0)
        ld = data[name]['loss']
        out[name] = ((1 - wa) * ld + wa * lt, lt, ld, kl)
    return out


def losses(model, b, teacher_w=1.0, value_q=0.5, margin_w=0.5, amp=False, view=False, aux_w=1.0, aux_teacher_w=None):
    """The distillation loss of batch `b` (a `Taught` batch; module docstring) -> (total, terms: the
    policy / value / margin losses against the teacher and against the shards' targets, the policy KL
    to the teacher; with an auxiliary head, per target its losses against the teacher (`aux_<name>_t`)
    and the shards (`aux_<name>_d`) and a multinomial's KL to the teacher (`aux_<name>_kl`)). `view`:
    the window has value-only rows - the policy terms are means over the batch's decision rows (rows
    with options), as the run's training (module docstring). `aux_w`, `aux_teacher_w` (None:
    `teacher_w`): the auxiliary terms' weight and blend, used when the model has a head and `b` the
    teacher's predictions."""
    layout = getattr(model, 'aux_layout', None) or []
    aux = bool(layout) and 'taux' in b
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
        if aux:
            plog, vlog, mp, ap = model(b['x'], b['n_players'], aux=True)
        else:
            plog, vlog, mp = model(b['x'], b['n_players'])
    plog, vlog, mp = plog.float(), vlog.float(), mp.float()
    S = model.seats
    logp = F.log_softmax(plog.masked_fill(~b['mask'], -1e9), dim=1)
    pt = b['tpi']
    if view:                                       # (run.train_model's mask: the rows with options)
        dec = b['mask'].any(1).float()
        nd = dec.sum().clamp(min=1.0)

        def mean(per):
            return (per * dec).sum() / nd
    else:
        def mean(per):
            return per.mean()
    p_t = mean(-(pt * logp).sum(1))
    p_d = mean(-(b['pi'] * logp).sum(1))
    ent_t = mean(-(pt * torch.log(pt.clamp(min=1e-12))).sum(1))
    lv = F.log_softmax(vlog, dim=1).clamp(min=-1e4)
    v_t = -(b['tv'][:, :S] * lv).sum(1).mean()
    zt = ((1 - value_q) * b['z'] + value_q * b['q'])[:, :S]
    v_d = -(zt * lv).sum(1).mean()
    present = (torch.arange(S, device=plog.device)[None, :] < b['n_players'][:, None]).float()
    m_t = (((mp - b['tm'][:, :S]) ** 2) * present).sum(1).mean() / 2
    m_d = (((mp - b['sc'][:, :S]) ** 2) * present).sum(1).mean() / 2
    w = float(teacher_w)
    total = (1 - w) * (p_d + v_d + margin_w * m_d) + w * (p_t + v_t + margin_w * m_t)
    terms = {'policy_t': p_t, 'value_t': v_t, 'margin_t': m_t, 'policy_d': p_d, 'value_d': v_d, 'margin_d': m_d,
             'kl_t': p_t - ent_t}
    if aux:
        wa = w if aux_teacher_w is None else float(aux_teacher_w)
        for name, (la, lt, ld, kl) in aux_losses(layout, ap.float(), b, wa).items():
            total = total + float(aux_w) * la
            terms[f'aux_{name}_t'] = lt
            terms[f'aux_{name}_d'] = ld
            if kl is not None:
                terms[f'aux_{name}_kl'] = kl
    return total, terms


def lr_at(step, steps, lr, lr_min, warmup):
    """Linear warm-up over `warmup` steps, then a cosine from `lr` down to `lr_min` at the last step."""
    if warmup and step < warmup:
        return lr * (step + 1) / warmup
    span = max(1, steps - warmup)
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * min(1.0, (step - warmup) / span)))


def view_rows(W):
    """The value-only rows of window `W` (rows without options, docs/az.md 16)."""
    return int(np.count_nonzero(np.diff(W.opt_off) == 0))


def distill(student, W, T, steps, batch=512, lr=5e-4, lr_min=1e-5, warmup=500, wd=1e-4, teacher_w=1.0, value_q=0.5,
            margin_w=0.5, device='cpu', seed=0, prefetch=2, opt=None, start=0, every=100, on_report=None,
            on_step=None, check=None, aux_w=1.0, aux_teacher_w=None):
    """Train `student` (in place) for steps start..steps-1 on window `W` with the teacher's answers `T`
    (`outputs`) by the loss of the module docstring. `opt`: an AdamW state to continue (a resumed
    distillation: `start` its step), else fresh. Every `every` steps `on_report(step, means of the
    loss terms over those steps, lr)`; `on_step(step, model, opt)` after each step (saving, held-out
    evaluations); `check()` before each (PAUSE). A NaN / inf loss or gradient norm raises
    finite.NonFinite before the step. A window with value-only rows: the policy terms over each
    batch's decision rows (`losses(view=True)`). A student with an auxiliary head distils it from the
    teacher's predictions in `T` ('aux': the same head; the window carries the shards' targets,
    `Window(aux_width=)`), `aux_w` / `aux_teacher_w` its weight and blend; a student with a head and a
    `T` without the teacher's predictions is refused (the head would train on nothing); a student
    without one ignores them. -> (the AdamW optimizer, the last report)."""
    if getattr(student, 'reader_specs', None):
        raise ValueError('distillation into a network with readers is not supported yet (docs/az.md 36)')
    if aux_width(student):
        if 'aux' not in T or aux_layout(T.get('aux_layout')) != student.aux_layout:
            raise ValueError(f"the student's auxiliary head {student.aux_layout} needs the teacher's predictions for "
                             f"it (the teacher's head: {T.get('aux_layout')})")
        if getattr(W, 'AW', 0) != aux_width(student):
            raise ValueError(f'the window carries auxiliary targets of width {getattr(W, "AW", 0)}, the student\'s '
                             f'head {aux_width(student)} (data.Window(aux_width=...))')
    elif 'aux' in T:                               # (a student built without the head: nothing to distil it into)
        T = {k: v for k, v in T.items() if k not in ('aux', 'aux_layout')}
    view = view_rows(W) > 0
    kw = {'view': view, 'aux_w': aux_w, 'aux_teacher_w': aux_teacher_w} if view or 'aux' in T else {}
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
            total, terms = losses(student, b, teacher_w, value_q, margin_w, amp, **kw)
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


def _q(a, p):
    return float(np.quantile(a, p)) if len(a) else 0.0


def _mean(a):
    return float(a.mean()) if len(a) else 0.0


def aux_agreement(S, T, W, rows=None):
    """How auxiliary predictions `S['aux']` agree with the teacher's `T['aux']` on window `W`'s rows
    (module docstring) -> {name: {...}}; `rows`: a boolean mask of W's rows whose KL is also reported
    by kind ({kind: mask}, e.g. decision / view rows)."""
    layout = aux_layout(T.get('aux_layout'))
    have = getattr(W, 'AW', 0) == sum(w for _, w, _ in layout)
    tgt = W.aux.astype(np.float32) if have else None
    mask = W.aux_mask if have else np.zeros(W.N, bool)
    out = {}
    eps = 1e-12
    for name, loss, sl in AUX.columns(layout):
        s, t = S['aux'][:, sl].astype(np.float32), T['aux'][:, sl].astype(np.float32)
        r = {}
        if loss == 'multinomial':
            n = tgt[:, sl].sum(1) if have else np.ones(W.N)
            sel = ~(mask & (n <= 0))                 # the teacher's rows (teacher_aux_rows)
            ps = s / np.maximum(s.sum(1, keepdims=True), eps)
            pt = t / np.maximum(t.sum(1, keepdims=True), eps)
            kl = (pt * (np.log(np.maximum(pt, eps)) - np.log(np.maximum(ps, eps)))).sum(1)
            top = ps.argmax(1) == pt.argmax(1)
            r.update(rows=int(sel.sum()), kl=round(_mean(kl[sel]), 6), kl_p90=round(_q(kl[sel], 0.9), 6),
                     top1=round(_mean(top[sel]), 4))
            for kind, km in (rows or {}).items():
                r[kind] = {'rows': int((sel & km).sum()), 'kl': round(_mean(kl[sel & km]), 6)}
            if have:
                hv = mask & (n > 0)                    # the run's rows with a target

                def data(p):
                    ce = -((tgt[:, sl] / np.maximum(n, eps)[:, None]) * np.log(np.maximum(p, eps))).sum(1)
                    hit = np.take_along_axis(tgt[:, sl], p.argmax(1)[:, None], 1)[:, 0] > 0
                    return {'rows': int(hv.sum()), 'ce': round(_mean(ce[hv]), 5), 'hit': round(_mean(hit[hv]), 4)}
                r['net'], r['teacher'] = data(ps), data(pt)
        else:
            d = (s - t) ** 2
            r.update(rows=int(W.N), mse=round(float(d.mean()), 7))
            if have:
                def data(p):
                    tg = tgt[:, sl]
                    if loss == 'poisson':
                        o = np.log(np.maximum(p, eps))
                        per = (p - tg * o).mean(1)
                    else:
                        per = ((p - tg) ** 2).mean(1)
                    return {'rows': int(mask.sum()), 'loss': round(_mean(per[mask]), 6)}
                r['net'], r['teacher'] = data(s), data(t)
        out[name] = r
    return out


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
    z = W.z / np.maximum(W.z.sum(1, keepdims=True), 1e-8)

    def value_rows(O):
        """Per row: the value cross-entropy against z, whether the most likely seat won."""
        vce = -(z * np.log(np.maximum(O['v'], eps)) * present).sum(1)
        acc = (np.where(present, O['v'], -1).argmax(1) == z.argmax(1))
        return vce, acc

    def data_losses(O):
        ce = -np.add.reduceat(pin * np.log(np.maximum(O['pi'], eps)), starts)
        vce, acc = value_rows(O)
        return {'policy_ce': round(float(ce[dec].mean()) if dec.any() else 0.0, 5),
                'value_ce': round(float(vce.mean()), 5), 'value_acc': round(float(acc.mean()), 4)}
    out = {'rows': int(W.N), 'decisions': int(dec.sum()),
           'policy_kl': round(float(kl[dec].mean()) if dec.any() else 0.0, 6),
           'policy_kl_p90': round(float(np.quantile(kl[dec], 0.9)) if dec.any() else 0.0, 6),
           'top1': round(float(top[dec].mean()) if dec.any() else 1.0, 4),
           'value_mse': round(float((dv ** 2).mean()), 7), 'value_mae': round(float(np.abs(dv).mean()), 5),
           'margin_mse': round(float((dm ** 2).mean()), 6),
           'net': data_losses(S), 'teacher': data_losses(T)}
    view = lens == 0
    kinds = None
    if view.any():                                  # value-only rows (docs/az.md 16): each kind apart
        kinds = {'decision': ~view, 'view': view}
        vs, vt = value_rows(S), value_rows(T)
        for kind, km in kinds.items():
            pk = present & km[:, None]
            dvk, dmk = (S['v'] - T['v'])[pk], (S['m'] - T['m'])[pk]
            out[kind] = {'rows': int(km.sum()), 'value_mse': round(_mean(dvk ** 2), 7),
                         'value_mae': round(_mean(np.abs(dvk)), 5), 'margin_mse': round(_mean(dmk ** 2), 6),
                         'net': {'value_ce': round(_mean(vs[0][km]), 5), 'value_acc': round(_mean(vs[1][km]), 4)},
                         'teacher': {'value_ce': round(_mean(vt[0][km]), 5), 'value_acc': round(_mean(vt[1][km]), 4)}}
    if 'aux' in S and 'aux' in T:
        out['aux'] = aux_agreement(S, T, W, kinds)
    return out


def describe(model):
    """A network's size in a few keys (for logs and the checkpoint's record); its auxiliary head's
    targets when it has one."""
    c = model.config
    out = {'arch': c.get('arch', 'mlp'), 'params': model.n_params()}
    if out['arch'] == 'tokens':
        out.update(d=c['d'], layers=c['layers'], heads=c['heads'], ff=c['ff'])
    elif out['arch'] == 'grid':
        out.update(ch=c['ch'], blocks=c['blocks'])
    else:
        out.update(hidden=c['hidden'], blocks=c['blocks'])
    if c.get('aux'):
        out['aux'] = [list(t) for t in aux_layout(c['aux'])]
    return out


def timed(fn, *a, **k):
    t0 = time.time()
    r = fn(*a, **k)
    return r, time.time() - t0
