"""The toy's reader measured (docs/az.md 36): what it learns, whether the trunk's losses should train it
(reader_detach), what it costs.

  python tools/reader_measure.py learn --run <a toy run with aux_next_card_w > 0> [--steps 3000]
  python tools/reader_measure.py cost

`learn`: the run's shards, its last iteration held out; four networks from the same seed (the trunk's
initial weights identical: a reader is built last) trained on the same batches with the run's losses
(policy, value, margin_loss_w x margin, aux_w x each next_card loss): 'head' (the auxiliary head only, no
reader), 'reader' (the reader fed into the trunk, detached: reader_detach 1), 'reader-nodetach' (fed, the
trunk's losses train it too: reader_detach 0), 'reader-none' (enter 'none': the reader is an auxiliary
output only, nothing fed). Held-out losses (nats): next_card by the reader and by the head, against the
uniform prior over 6 (ln 6) and the best prediction from public information (uniform over the next
round's stage's unrevealed cards); policy, value. JSON lines on stdout.

`cost`: multiply-adds per position and wall time per forward (CPU, 4 threads; CUDA when there is one) of
the toy's token network with and without its reader, and of a reader of MTG Zero's planned size (~4 x 330
bag inputs -> hidden -> 330, projected into a 128-wide global token) beside a 128 x 4 token network's
21.9M multiply-adds.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))

from zero import auxtarget as AUX, games as G, package  # noqa: E402
from zero.data import Window, batches, shard_paths  # noqa: E402
from zero.net import make_net  # noqa: E402
from zero.protocol import MAX_SEATS  # noqa: E402

TOY_AUX = [['next_card', 6, 'multinomial']]


def bayes(x, t):
    """The best next_card loss from public information per row (nats): ln of the number of unrevealed cards
    of the next round's stage (rounds 1-3: cards 0-2, rounds 4-6: cards 3-5); x [N, F] features, t [N, 6]."""
    r = x[:, 0:6].argmax(1) + 1                                   # this round, 1..6
    nxt = r + 1
    rev = x[:, 25:31] > 0.5
    k = np.where(nxt <= 3, (~rev[:, 0:3]).sum(1), (~rev[:, 3:6]).sum(1))
    have = t.sum(1) > 0
    return np.log(np.maximum(k, 1))[have]


def losses(m, b, aux_w, mw):
    o = m(b['x'], b['n_players'], aux=True)
    plog, vlog, mp, raw = o[0].float(), o[1].float(), o[2].float(), o[3].float()
    S = m.seats
    logp = F.log_softmax(plog.masked_fill(~b['mask'], -1e9), dim=1)
    lp = -(b['pi'] * logp).sum(1).mean()
    lv = -(b['z'][:, :S] * F.log_softmax(vlog, dim=1).clamp(min=-1e4)).sum(1).mean()
    present = (torch.arange(S)[None, :] < b['n_players'][:, None]).float()
    lm = (((mp - b['sc'][:, :S]) ** 2) * present).sum(1).mean() / 2
    head = AUX.losses(TOY_AUX, raw[:, :6], b['aux'], b['aux_mask'])['next_card']
    out = {'policy': lp, 'value': lv, 'margin': lm, 'head': head['loss']}
    loss = lp + lv + mw * lm + aux_w * head['loss']
    if raw.shape[1] > 6:
        rd = AUX.losses(TOY_AUX, raw[:, 6:12], b['aux'], b['aux_mask'])['next_card']
        out['reader'] = rd['loss']
        loss = loss + aux_w * rd['loss']
    return loss, out


def learn(a):
    torch.set_num_threads(a.threads)
    info = G.info('toy')
    iters = sorted(glob.glob(os.path.join(a.run, 'data', 'iter_*')))
    train_p = [p for d in iters[:-1] for p in shard_paths(d)]
    held_p = shard_paths(iters[-1])
    F_, A_ = info['num_features'], info['num_actions']
    W = Window(train_p, F_, A_, budget_rows=10 ** 7, seed=0, aux_width=6)
    H = Window(held_p, F_, A_, budget_rows=10 ** 7, seed=0, aux_width=6)
    hb = H.batch(np.arange(H.N), 'cpu')
    hx, ht = hb['x'].numpy(), hb['aux'].numpy()
    opt_bayes = float(bayes(hx, ht).mean())
    print(json.dumps({'train_rows': W.N, 'held_rows': H.N, 'held_target_rows': int((ht.sum(1) > 0).sum()),
                      'uniform': round(math.log(6), 4), 'best_public': round(opt_bayes, 4)}), flush=True)
    reader = package.CORE.readers_of('toy')
    arms = {'head': None, 'reader': reader, 'reader-nodetach': reader,
            'reader-none': [dict(reader[0], enter='none')]}
    for name, rd in arms.items():
        if a.arms and name not in a.arms.split(','):
            continue
        torch.manual_seed(a.seed)
        m = make_net(info, 'toy', a.arch, a.hidden, a.blocks, heads=4, aux=TOY_AUX, readers=rd).train()
        m.reader_detach = name != 'reader-nodetach'
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
        rng = np.random.default_rng(a.seed)
        t0 = time.time()
        for step, (_, b) in enumerate(batches(W, rng, a.steps, a.batch)):
            loss, _ = losses(m, b, a.aux_w, a.margin_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            if (step + 1) % a.every == 0 or step + 1 == a.steps:
                m.eval()
                with torch.no_grad():
                    _, h = losses(m, hb, a.aux_w, a.margin_w)
                m.train()
                print(json.dumps(dict({k: round(float(v), 4) for k, v in h.items()}, arm=name, step=step + 1,
                                      s=round(time.time() - t0, 1))), flush=True)


def macs(m, x):
    """Multiply-adds of `m`'s Linear layers (and the attention products) per row of x, by forward hooks."""
    tot = [0]

    def lin(mod, inp, out):
        tot[0] += inp[0].numel() // inp[0].shape[-1] * mod.in_features * mod.out_features
    hs = [mod.register_forward_hook(lin) for mod in m.modules() if isinstance(mod, torch.nn.Linear)]
    with torch.no_grad():
        m(x, torch.full((len(x),), 2))
    for h in hs:
        h.remove()
    return tot[0] / len(x)


def wall(f, reps):
    f()
    t0 = time.perf_counter()
    for _ in range(reps):
        f()
    return (time.perf_counter() - t0) / reps


def cost(a):
    torch.set_num_threads(a.threads)
    from zero.net import Reader, TokenNet
    info = G.info('toy')
    rng = np.random.default_rng(0)
    xs = []
    while len(xs) < a.batch:
        g = G.engine().Game('toy', '{}', 2, int(rng.integers(1 << 30)))
        for _ in range(int(rng.integers(0, 40))):
            if g.over:
                break
            legal = g.legal()
            g.step(legal[int(rng.integers(len(legal)))])
        if not g.over:
            xs.append(np.asarray(g.encode(g.to_move), np.float32))
    x = torch.from_numpy(np.stack(xs))
    n = torch.full((len(x),), 2)
    reader = package.CORE.readers_of('toy')
    devs = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    for d, layers in ((128, 4), (64, 2)):
        row = {'net': f'toy tokens {d}x{layers}'}
        for tag, rd in (('plain', None), ('reader', reader)):
            torch.manual_seed(0)
            m = make_net(info, 'toy', 'tokens', d, layers, heads=4, readers=rd).eval()
            row[f'macs_{tag}'] = int(macs(m, x))
            for dev in devs:
                mm, xx, nn_ = m.to(dev), x.to(dev), n.to(dev)

                def f():
                    with torch.no_grad():
                        mm(xx, nn_)
                    if dev == 'cuda':
                        torch.cuda.synchronize()
                row[f'us_{tag}_{dev}'] = round(wall(f, a.reps) / len(x) * 1e6, 2)
        row['macs_share'] = round(row['macs_reader'] / row['macs_plain'] - 1, 4)
        print(json.dumps(row), flush=True)
    # MTG Zero's planned reader: 4 bags of ~330 identities (1,320 inputs) -> hidden -> 330, into d = 128
    for hid in (128, 256, 330):
        spec = {'target': 'deck', 'width': 330, 'loss': 'multinomial', 'inputs': [[0, 1320]], 'hidden': hid,
                'enter': 'trunk', 'belief': False}
        r = Reader(spec, 1320, 128).eval()
        mac = 1320 * hid + hid * 330 + 330 * 128
        row = {'reader': f'1320 -> {hid} -> 330 -> 128', 'macs': mac, 'share_of_21.9M': round(mac / 21.9e6, 4)}
        for dev in devs:
            rr = r.to(dev)
            xx = torch.rand(256, 1320, device=dev)

            def f():
                with torch.no_grad():
                    raw = rr.mlp(xx.index_select(1, rr.idx))
                    rr.proj(torch.softmax(raw, 1))
                if dev == 'cuda':
                    torch.cuda.synchronize()
            row[f'us_per_row_{dev}'] = round(wall(f, a.reps) / 256 * 1e6, 3)
        print(json.dumps(row), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('what', choices=['learn', 'cost'])
    ap.add_argument('--run')
    ap.add_argument('--arms', default='')
    ap.add_argument('--arch', default='tokens')
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--blocks', type=int, default=2)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--every', type=int, default=500)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--aux-w', type=float, default=0.5)
    ap.add_argument('--margin-w', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--reps', type=int, default=20)
    a = ap.parse_args(argv)
    if a.what == 'learn':
        if not a.run:
            ap.error('learn needs --run')
        learn(a)
    else:
        cost(a)


if __name__ == '__main__':
    main()
