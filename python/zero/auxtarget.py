"""Auxiliary training targets (docs/az.md 15): the losses of an auxiliary head and their statistics.

A game declares targets (zero/package.py `aux`: name, width, loss; the engine's `SearchGame::AUX` and
`aux_target`, docs/search.md 1.2): a per-decision label vector computed from the TRUE state - it may
hold what the decider cannot see (MTG Zero's belief target: the opponent's actual hand) - which a
driver created with `aux=True` records with every sample (shard key `aux`). The network's auxiliary
head (zero/net.py) predicts it from the trunk; training adds `aux_<name>_w` x the target's loss.
Off (every weight 0, the default) nothing is recorded, built or trained: the run of before.

Losses (net.AUX_LOSSES), per sample row, averaged over the rows that have a target:
* 'multinomial': the segment is a count vector (a hand's cards by identity); the head's logits are a
  softmax over the segment and the loss is the cross-entropy against the counts normalised to sum 1
  (-sum_k t_k / n log p_k, n = sum_k t_k): the head learns the composition, and n x p is the expected
  count of each column given the (public) total. A row whose segment is all zero has no target.
* 'poisson': the head's outputs are log-rates, the loss the Poisson negative log-likelihood
  mean_k (exp(o_k) - t_k o_k) (counts without a known total).
* 'mse': mean_k (o_k - t_k)^2.
A row of a shard without targets (aux_mask 0: a fork's older window) never counts.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .net import aux_layout


def columns(layout):
    """[(name, loss, slice of the target vector)] of an auxiliary layout."""
    out, k = [], 0
    for name, w, loss in aux_layout(layout):
        out.append((name, loss, slice(k, k + w)))
        k += w
    return out


def losses(layout, raw, target, mask):
    """Each target's loss on a batch -> {name: {'loss': scalar tensor, 'rows': rows counted (tensor),
    'hit': multinomial only: the share of those rows whose most likely column is in the target}}.
    raw [B, W] (the head's outputs, float), target [B, W], mask [B] (1 = the row has targets)."""
    out = {}
    for name, loss, sl in columns(layout):
        o, t = raw[:, sl].float(), target[:, sl].float()
        if loss == 'multinomial':
            n = t.sum(1)
            have = mask * (n > 0).float()
            logp = F.log_softmax(o, 1)
            per = -(t / n.clamp(min=1e-8)[:, None] * logp).sum(1)
            rows = have.sum()
            v = (per * have).sum() / rows.clamp(min=1.0)
            with torch.no_grad():
                top = o.argmax(1, keepdim=True)
                hit = ((t.gather(1, top)[:, 0] > 0).float() * have).sum() / rows.clamp(min=1.0)
            out[name] = {'loss': v, 'rows': rows, 'hit': hit}
            continue
        if loss == 'poisson':
            oc = o.clamp(max=30.0)
            per = (torch.exp(oc) - t * oc).mean(1)
        else:
            per = ((o - t) ** 2).mean(1)
        rows = mask.sum()
        out[name] = {'loss': (per * mask).sum() / rows.clamp(min=1.0), 'rows': rows}
    return out


def window_stats(W, layout, rows=20_000, seed=0):
    """The training window's auxiliary targets: rows that have them (and their share), and per
    multinomial target the mean total count and the share of rows with a non-empty segment (on up to
    `rows` of those rows)."""
    have = np.flatnonzero(W.aux_mask)
    out = {'rows': int(len(have)), 'share': round(len(have) / max(1, W.N), 4)}
    if len(have):
        pick = have if len(have) <= rows else np.random.default_rng(seed).choice(have, rows, replace=False)
        a = W.aux[np.sort(pick)].astype(np.float32)
        for name, loss, sl in columns(layout):
            if loss == 'multinomial':
                n = a[:, sl].sum(1)
                out[name] = {'mean_total': round(float(n.mean()), 4), 'nonempty': round(float((n > 0).mean()), 4)}
    return out
