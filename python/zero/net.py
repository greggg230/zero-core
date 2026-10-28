"""Policy / value / margin networks over the game's fixed-width feature vector:

* `Net` (config arch 'mlp', the default): a residual MLP with LayerNorm (stpz style; small
  enough for the browser: Caverna Zero's wasm player runs the same forward pass in Rust);
* `GridNet` (arch 'grid'): a residual 3x3 conv tower over the board planes of the features,
  for board games whose features hold per-cell planes (the game package's `grids`, e.g.
  Kingdom Builder's 16 planes x 20 x 20). The other features (globals) are embedded once and
  bias the channels of every block, with the board's mean / max pooled state; the cell
  actions get one logit per cell from the tower (1x1 conv), every other action from the pooled
  trunk. Offset-hex boards (odd rows shifted right) get a row-parity input plane, so the 3x3
  kernels can tell the two non-neighbour corners apart. Not exported to the browser players
  (zero.webexport) yet.

Heads:
  policy : logits over the game's NUM_ACTIONS global actions (masked to the legal ones by the
           caller: `infer.NetEvaluator`, the training loss)
  value  : logits over MAX_SEATS relative seats (seat j = (decider + j) % n), masked to the
           seats present -> softmax = P(seat wins)
  margin : each relative seat's margin target (`SearchGame::margin`), regression

Checkpoints (`save`, atomic): {config, state, extra}; `load` rebuilds the network from the
config; `widen` maps a checkpoint onto a larger feature / action space (new inputs get zero
weights, new actions zero logits), so a widened network computes what the source did.
"""
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .finite import check_state
from .protocol import MAX_SEATS


class Block(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.ln = nn.LayerNorm(h)
        self.fc1 = nn.Linear(h, h)
        self.fc2 = nn.Linear(h, h)

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(self.ln(x))))


class Net(nn.Module):
    def __init__(self, features, actions, hidden=256, blocks=4, seats=MAX_SEATS, game='toy', head=128):
        super().__init__()
        self.config = {'features': features, 'actions': actions, 'hidden': hidden, 'blocks': blocks,
                       'seats': seats, 'game': game, 'head': head}
        self.features = features
        self.seats = seats
        self.inp = nn.Linear(features, hidden)
        self.blocks = nn.ModuleList(Block(hidden) for _ in range(blocks))
        self.ln = nn.LayerNorm(hidden)
        self.pol = nn.Linear(hidden, actions)
        self.val = nn.Sequential(nn.Linear(hidden, head), nn.ReLU(), nn.Linear(head, seats))
        self.mar = nn.Sequential(nn.Linear(hidden, head), nn.ReLU(), nn.Linear(head, seats))

    def forward(self, x, n_players):
        """x [B, >= features] (an older, narrower network reads a prefix), n_players [B]
        -> policy logits [B, actions], value logits [B, seats] (absent seats -1e9),
        margin [B, seats]."""
        h = F.relu(self.inp(x[:, :self.features]))
        for b in self.blocks:
            h = b(h)
        h = F.relu(self.ln(h))
        present = torch.arange(self.seats, device=x.device)[None, :] < n_players[:, None]
        vlog = self.val(h).masked_fill(~present, -1e9)
        return self.pol(h), vlog, self.mar(h)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class GridBlock(nn.Module):
    """Residual block of two 3x3 convs (BatchNorm); the globals and the input's pooled mean / max
    bias the first conv's channels (KataGo-style global pooling)."""

    def __init__(self, ch, gdim):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.n1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.n2 = nn.BatchNorm2d(ch)
        self.gb = nn.Linear(gdim + 2 * ch, ch)

    def forward(self, h, g):
        pooled = torch.cat([g, h.mean((2, 3)), h.amax((2, 3))], 1)
        y = F.relu(self.n1(self.c1(h)) + self.gb(pooled)[:, :, None, None])
        return F.relu(h + self.n2(self.c2(y)))


class GridNet(nn.Module):
    """grid: {planes, h, w, offset (first plane feature), cell_action0 (the action of cell 0;
    cells are row-major, as the planes), hex ('odd-r' adds a row-parity plane, or none),
    segments (optional: [[offset, planes], ...] when the planes sit in several blocks of the
    features, e.g. a layout that appends planes after its globals; default [[offset, planes]])}.
    Every feature outside the plane blocks is a global."""

    def __init__(self, features, actions, grid, ch=64, blocks=6, seats=MAX_SEATS, game='toy', head=128, ghid=128,
                 arch='grid'):
        super().__init__()
        segs = [[int(o), int(k)] for o, k in grid.get('segments') or [[grid.get('offset', 0), grid['planes']]]]
        grid = {'planes': sum(k for _, k in segs), 'h': int(grid['h']), 'w': int(grid['w']), 'offset': segs[0][0],
                'cell_action0': int(grid.get('cell_action0', 0)), 'hex': grid.get('hex', 'odd-r'), 'segments': segs}
        self.config = {'arch': 'grid', 'features': features, 'actions': actions, 'grid': grid, 'ch': ch, 'blocks': blocks,
                       'seats': seats, 'game': game, 'head': head, 'ghid': ghid}
        self.features = features
        self.seats = seats
        P, H, W = grid['planes'], grid['h'], grid['w']
        self.P, self.H, self.W, self.off, self.a0 = P, H, W, grid['offset'], grid['cell_action0']
        self.n = P * H * W
        cells = H * W
        pidx = [o + i for o, k in segs for i in range(k * cells)]
        if max(pidx) >= features or len(set(pidx)) != len(pidx) or self.a0 + cells > actions:
            raise ValueError(f'grid {grid} does not fit {features} features / {actions} actions')
        # one block: slices; several: gathers (the globals in feature order, so widen appends)
        self.contig = len(segs) == 1
        inplane = set(pidx)
        self.register_buffer('pidx', torch.tensor(pidx, dtype=torch.long), persistent=False)
        self.register_buffer('gidx', torch.tensor([i for i in range(features) if i not in inplane], dtype=torch.long),
                             persistent=False)
        extra = 1 if grid['hex'] == 'odd-r' else 0
        par = torch.zeros(1, extra, H, W)
        if extra:
            par[:, :, 1::2, :] = 1.0
        self.register_buffer('par', par, persistent=False)
        self.g = nn.Sequential(nn.Linear(features - self.n, ghid), nn.ReLU())
        self.stem = nn.Conv2d(P + extra, ch, 3, padding=1, bias=False)
        self.stem_n = nn.BatchNorm2d(ch)
        self.stem_g = nn.Linear(ghid, ch)
        self.blocks = nn.ModuleList(GridBlock(ch, ghid) for _ in range(blocks))
        self.cell = nn.Conv2d(ch, 1, 1)
        self.trunk = nn.Sequential(nn.Linear(2 * ch + ghid, 256), nn.ReLU())
        self.other = nn.Linear(256, actions - H * W)
        self.val = nn.Sequential(nn.Linear(256, head), nn.ReLU(), nn.Linear(head, seats))
        self.mar = nn.Sequential(nn.Linear(256, head), nn.ReLU(), nn.Linear(head, seats))

    def forward(self, x, n_players):
        """As `Net.forward`."""
        B = x.shape[0]
        x = x[:, :self.features]
        o, n = self.off, self.n
        if self.contig:
            p = x[:, o:o + n].reshape(B, self.P, self.H, self.W)
            g = self.g(torch.cat([x[:, :o], x[:, o + n:]], 1))
        else:
            p = x.index_select(1, self.pidx).reshape(B, self.P, self.H, self.W)
            g = self.g(x.index_select(1, self.gidx))
        if self.par.shape[1]:
            p = torch.cat([p, self.par.expand(B, -1, -1, -1)], 1)
        if getattr(self, 'channels_last', False):        # infer.GraphRunner's folded copy
            p = p.contiguous(memory_format=torch.channels_last)
        h = F.relu(self.stem_n(self.stem(p)) + self.stem_g(g)[:, :, None, None])
        for b in self.blocks:
            h = b(h, g)
        cell = self.cell(h).reshape(B, self.H * self.W)
        e = self.trunk(torch.cat([h.mean((2, 3)), h.amax((2, 3)), g], 1))
        rest = self.other(e)
        pol = torch.cat([rest[:, :self.a0], cell, rest[:, self.a0:]], 1)
        present = torch.arange(self.seats, device=x.device)[None, :] < n_players[:, None]
        vlog = self.val(e).masked_fill(~present, -1e9)
        return pol, vlog, self.mar(e)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def tokens_layout(layout, features, actions):
    """The normalised object layout of a `TokenNet` (validated against the feature / action
    widths; module docstring of TokenNet). -> a plain dict (stored in the checkpoint config)."""
    vocab = int(layout['vocab'])
    segs, bags, used, acts, ids = [], [], set(), {}, []

    def claim(name, lo, n):
        if lo < 0 or lo + n > features:
            raise ValueError(f'tokens layout: {name} [{lo}, {lo + n}) does not fit {features} features')
        r = set(range(lo, lo + n))
        if r & used:
            raise ValueError(f'tokens layout: {name} overlaps another segment')
        used.update(r)

    for s in layout.get('segments', []):
        seg = {'name': str(s['name']), 'offset': int(s['offset']), 'count': int(s['count']), 'width': int(s['width']),
               'present': None if s.get('present') is None else int(s['present']), 'id': None,
               'actions': [[int(b), int(k)] for b, k in s.get('actions', [])]}
        if seg['count'] < 1 or seg['width'] < 1:
            raise ValueError(f'tokens layout: segment {seg["name"]} has no slots')
        if seg['present'] is not None and not 0 <= seg['present'] < seg['width']:
            raise ValueError(f'tokens layout: segment {seg["name"]}: present flag outside the slot')
        if s.get('id') is not None:
            radix = [int(r) for r in s['id']['radix']]
            at = int(s['id']['at'])
            if at < 0 or at + sum(radix) > seg['width'] or math.prod(radix) > vocab:
                raise ValueError(f'tokens layout: segment {seg["name"]}: identity {s["id"]} does not fit')
            seg['id'] = {'at': at, 'radix': radix}
        claim(seg['name'], seg['offset'], seg['count'] * seg['width'])
        for j, (base, stride) in enumerate(seg['actions']):
            for slot in range(seg['count']):
                a = base + slot * stride
                if not 0 <= a < actions:
                    raise ValueError(f'tokens layout: segment {seg["name"]}: action {a} outside {actions}')
                if a in acts:
                    raise ValueError(f'tokens layout: action {a} belongs to {acts[a]} and {seg["name"]}')
                acts[a] = seg['name']
        if s.get('id_actions') is not None:
            # (the key only when set: the normalised layouts of networks without it, stored in
            # their checkpoints, stay as they were)
            base = int(s['id_actions'])
            if seg['id'] is None:
                raise ValueError(f'tokens layout: segment {seg["name"]}: id_actions without an identity')
            if not 0 <= base < actions:
                raise ValueError(f'tokens layout: segment {seg["name"]}: id_actions {base} outside {actions}')
            seg['id_actions'] = base
            ids.append(seg)
        segs.append(seg)
    for seg in ids:
        lo, hi = seg['id_actions'], seg['id_actions'] + vocab
        clash = sorted(a for a in acts if lo <= a < hi)
        if clash:
            raise ValueError(f'tokens layout: segment {seg["name"]}: id_actions [{lo}, {hi}) holds action {clash[0]} '
                             f'of {acts[clash[0]]}')
    for b in layout.get('bags', []):
        bag = {'name': str(b['name']), 'offset': int(b['offset']), 'count': int(b['count']), 'width': int(b['width'])}
        if bag['width'] > vocab or bag['count'] < 1:
            raise ValueError(f'tokens layout: bag {bag["name"]} wider than the vocabulary {vocab}')
        claim(bag['name'], bag['offset'], bag['count'] * bag['width'])
        bags.append(bag)
    return {'vocab': vocab, 'segments': segs, 'bags': bags}


class TBlock(nn.Module):
    """Pre-norm transformer block (multi-head self-attention over the present tokens, GELU MLP)."""

    def __init__(self, d, heads, ff):
        super().__init__()
        self.h = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff)
        self.ff2 = nn.Linear(ff, d)

    def forward(self, h, keep):
        """h [B, T, d], keep [B, 1, 1, T] bool (False = padding: never attended to)."""
        B, T, D = h.shape
        q, k, v = self.qkv(self.ln1(h)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
        h = h + self.out(a.transpose(1, 2).reshape(B, T, D))
        return h + self.ff2(F.gelu(self.ff1(self.ln2(h))))


class TokenNet(nn.Module):
    """arch 'tokens': a transformer over the game's objects, read from its fixed-width features
    through a layout descriptor (the game package's `tokens[game]`, stored in the config), so the
    `SearchGame` trait, the driver and the shards stay as they are.

    layout = {vocab, segments: [...], bags: [...]}:
      * segment {name, offset, count, width, present, id, actions, id_actions}: `count` object
        slots of `width` floats from feature `offset` (slot s at offset + s * width). A slot is
        a token when its `present` float (index in the slot) is > 0.5 (present None: always).
        `id` {at, radix: [r0, r1, ...]}: the slot's identity as mixed-radix one-hot digits at
        `at` (index = sum digit_k * prod(r_{k+1}...)), embedded from a table of `vocab` + 1 rows
        shared by every segment and bag (the last row: no / unknown identity, e.g. a face-down
        card). `actions` [[base, stride], ...]: slot s scores the actions base + s * stride
        (one logit per pair, from the slot's token). `id_actions` base (optional; needs `id`):
        the slot also scores the action base + its identity (actions by identity: MTG Zero's
        team draft, where a card's pick is its identity wherever it sits), one more logit from
        the token; an identity whose action is outside the action space scores nothing, and of
        two tokens of a row with the same identity one scores it (a layout keeps identities
        distinct within a row). Every other action gets its logit from the pooled state.
      * bag {name, offset, count, width}: `count` multisets over the identities (`width` <=
        vocab counts each, e.g. the cards of a zone by identity); each is pooled as the
        count-weighted sum of the identity embeddings, and the pools feed the global token.
      * every feature outside the segments and bags is a global (the global token's input).

    Token = identity embedding + a linear map of the slot's floats + a learned embedding of
    (segment, slot) (the slot order means something: timestamps, top of the stack, whose
    library). Only present slots are embedded and attended to: the batch is packed (each row's
    tokens in layout order, the global token first), its rows sorted by token count and run in
    chunks of `chunk` rows, each padded to its own longest row, padding masked out of the
    attention. `layers` pre-norm blocks of width `d`. Heads read the global token's output and
    the mean of the row's tokens: value / margin as `Net`, and the generic policy logits; a
    slot's own actions are scored from its token, conditioned on the pooled state (pointer
    logits), and so is its identity's action in a segment with `id_actions`. An empty slot's
    actions fall back to the generic logits.
    """

    def __init__(self, features, actions, layout, d=128, layers=3, heads=4, ff=256, seats=MAX_SEATS, game='toy',
                 head=128, arch='tokens'):
        super().__init__()
        lay = tokens_layout(layout, features, actions)
        self.config = {'arch': 'tokens', 'features': features, 'actions': actions, 'layout': lay, 'd': d,
                       'layers': layers, 'heads': heads, 'ff': ff, 'seats': seats, 'game': game, 'head': head}
        if d % heads:
            raise ValueError(f'tokens: d {d} is not a multiple of heads {heads}')
        self.features, self.seats, self.actions, self.d = features, seats, actions, d
        self.segs, self.bags, self.vocab = lay['segments'], lay['bags'], lay['vocab']
        self.chunk = 64                      # rows per packed chunk (an execution detail, not saved)
        inside = set()
        for s in self.segs:
            inside.update(range(s['offset'], s['offset'] + s['count'] * s['width']))
        for b in self.bags:
            inside.update(range(b['offset'], b['offset'] + b['count'] * b['width']))
        gidx = [i for i in range(features) if i not in inside]
        self.register_buffer('gidx', torch.tensor(gidx, dtype=torch.long), persistent=False)
        self.emb = nn.Embedding(self.vocab + 1, d)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.g = nn.Linear(max(1, len(gidx)), d)
        nbag = sum(b['count'] for b in self.bags)
        self.bag = nn.Linear(nbag * (d + 1), d) if nbag else None
        self.seg_in = nn.ModuleList(nn.Linear(s['width'], d) for s in self.segs)
        self.n_tok = 1 + sum(s['count'] for s in self.segs)
        self.pos = nn.Parameter(torch.randn(self.n_tok, d) * 0.02)  # (segment, slot) embeddings; 0 = global
        self.blocks = nn.ModuleList(TBlock(d, heads, ff) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.gen = nn.Linear(2 * d, actions)
        self.pk = nn.Linear(d, d)
        self.pq = nn.Linear(2 * d, d, bias=False)
        # pointer head: one output column per (segment, action pair) and per segment with
        # id_actions; per position in the full token axis (0 = global, then the segments' slots)
        # its columns and action ids (-1 / `actions` = none: scattered into a dummy column); an
        # identity column's action is its base + the token's identity (`tok_isid`, `tok_idbase`)
        nk = [len(s['actions']) + ('id_actions' in s) for s in self.segs]
        self.n_id = sum('id_actions' in s for s in self.segs)
        self.ptr = nn.Linear(d, max(1, sum(nk)))
        K = max(nk + [1])
        cols = torch.full((self.n_tok, K), -1, dtype=torch.long)
        acts = torch.full((self.n_tok, K), actions, dtype=torch.long)
        isid = torch.zeros((self.n_tok, K), dtype=torch.bool)
        idbase = torch.zeros(self.n_tok, dtype=torch.long)
        start, c0 = 1, 0
        for s, k in zip(self.segs, nk):
            for slot in range(s['count']):
                for j, (base, stride) in enumerate(s['actions']):
                    cols[start + slot, j] = c0 + j
                    acts[start + slot, j] = base + slot * stride
                if 'id_actions' in s:
                    j = len(s['actions'])
                    cols[start + slot, j] = c0 + j
                    isid[start + slot, j] = True
                    idbase[start + slot] = s['id_actions']
            start += s['count']
            c0 += k
        self.register_buffer('tok_cols', cols, persistent=False)
        self.register_buffer('tok_acts', acts, persistent=False)
        if self.n_id:
            self.register_buffer('tok_isid', isid, persistent=False)
            self.register_buffer('tok_idbase', idbase, persistent=False)
        self.val = nn.Sequential(nn.Linear(2 * d, head), nn.ReLU(), nn.Linear(head, seats))
        self.mar = nn.Sequential(nn.Linear(2 * d, head), nn.ReLU(), nn.Linear(head, seats))
        # the policy starts uniform: with random output layers the generic logit of a frequent
        # action (MTG's PASS) and a segment's pointer logits are a few random constants, so one
        # of them can dominate every position (MTG run mtg2a: 55% of the prior on PASS where 25%
        # is uniform; its self-play games ran into the turn cap). The value head keeps its
        # random init: at zero every search is a tie, and ties go to the first option (MTG's
        # PASS) (MTG run mtg2n, docs/az.md 10)
        for lin in (self.gen, self.ptr):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _identity(self, sl, idspec):
        """Identity index [N] of slot rows [N, width] from their one-hot digits (vocab = none)."""
        at, idx, ok = idspec['at'], None, None
        for r in idspec['radix']:
            blk = sl[:, at:at + r]
            dig = (blk @ torch.arange(r, dtype=sl.dtype, device=sl.device)).round().long()
            has = blk.sum(1) > 0.5
            idx = dig if idx is None else idx * r + dig
            ok = has if ok is None else ok & has
            at += r
        return torch.where(ok & (idx < self.vocab), idx, torch.full_like(idx, self.vocab))

    def pack(self, x, ident=False):
        """The packed tokens: (h [B, T, d] (row b's n[b] tokens first, in layout order; the
        global token at 0), n [B], full [B, T] = each packed token's position in the full token
        axis (0 = global, then the segments' slots)); with `ident`, also each packed token's
        identity [B, T] (vocab: none)."""
        B, dev = x.shape[0], x.device
        segs, pres = [], [torch.ones(B, 1, dtype=torch.bool, device=dev)]
        for s in self.segs:
            o, k, w = s['offset'], s['count'], s['width']
            sl = x[:, o:o + k * w].reshape(B, k, w)
            segs.append(sl)
            pres.append(sl[..., s['present']] > 0.5 if s['present'] is not None
                        else torch.ones(B, k, dtype=torch.bool, device=dev))
        pres = torch.cat(pres, 1)                                     # [B, n_tok]
        n = pres.sum(1)
        T = int(n.max())
        cpos = pres.long().cumsum(1) - 1                             # packed index of a present token
        full = torch.zeros(B, T, dtype=torch.long, device=dev)
        g = self.g(x.index_select(1, self.gidx))
        if self.bag is not None:
            pools = []
            for b in self.bags:
                o, k, w = b['offset'], b['count'], b['width']
                c = x[:, o:o + k * w].reshape(B, k, w)
                pools.append(torch.cat([c @ self.emb.weight[:w], c.sum(-1, keepdim=True)], -1))
            g = g + self.bag(torch.cat(pools, 1).flatten(1))
        h = torch.cat([(g + self.pos[0])[:, None], g.new_zeros(B, T - 1, self.d)], 1)
        idt = torch.full((B, T), self.vocab, dtype=torch.long, device=dev) if ident else None
        start = 1
        for s, sl, lin in zip(self.segs, segs, self.seg_in):
            bi, si = pres[:, start:start + s['count']].nonzero(as_tuple=True)
            if len(bi):
                rows = sl[bi, si]                                     # [N, width]
                t = lin(rows) + self.pos[start + si]
                pi = cpos[bi, start + si]
                if s['id'] is not None:
                    i = self._identity(rows, s['id'])
                    t = t + self.emb(i)
                    if idt is not None:
                        idt[bi, pi] = i
                h = h.index_put((bi, pi), t)
                full[bi, pi] = start + si
            start += s['count']
        if ident:
            return h, n, full, idt
        return h, n, full

    def forward(self, x, n_players):
        """As `Net.forward`."""
        x = x[:, :self.features]
        if self.n_id:
            h, n, full, idt = self.pack(x, ident=True)
        else:
            h, n, full = self.pack(x)
        B, T, D = h.shape
        dev = x.device
        rows = torch.argsort(n, stable=True) if B > self.chunk else torch.arange(B, device=dev)
        z = h.new_zeros(B, 2 * D)
        pol = h.new_zeros(B, self.actions + 1)
        for c0 in range(0, B, self.chunk):
            r = rows[c0:c0 + self.chunk]
            nr = n[r]
            Tc = int(nr.max())
            hc = h[r, :Tc]
            keep = torch.arange(Tc, device=dev)[None, :] < nr[:, None]
            k4 = keep[:, None, None, :]
            for blk in self.blocks:
                hc = blk(hc, k4)
            hc = self.ln(hc) * keep[..., None]
            zc = torch.cat([hc[:, 0], hc.sum(1) / nr[:, None].to(hc.dtype)], 1)
            pc = F.pad(self.gen(zc), (0, 1))
            fc = full[r, :Tc]
            cols = self.tok_cols[fc]                                  # [Bc, Tc, K]
            acts = self.tok_acts[fc]
            if self.n_id:                                             # actions by identity
                ic = idt[r, :Tc]
                ia = self.tok_idbase[fc] + ic
                ia = torch.where((ic < self.vocab) & (ia < self.actions), ia, self.actions)
                acts = torch.where(self.tok_isid[fc], ia[..., None], acts)
            acts = torch.where(keep[..., None] & (cols >= 0), acts, self.actions)
            lg = self.ptr(F.gelu(self.pk(hc) + self.pq(zc)[:, None])).gather(2, cols.clamp(min=0))
            pc = pc.scatter(1, acts.flatten(1), lg.flatten(1))
            z = z.index_copy(0, r, zc)
            pol = pol.index_copy(0, r, pc)
        present = torch.arange(self.seats, device=dev)[None, :] < n_players[:, None]
        vlog = self.val(z).masked_fill(~present, -1e9)
        return pol[:, :self.actions], vlog, self.mar(z)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


ARCHS = {'grid': GridNet, 'tokens': TokenNet}


def build(config):
    """A network from its config (a checkpoint's): `GridNet` for arch 'grid', `TokenNet` for
    'tokens', else `Net`."""
    cfg = dict(config)
    arch = cfg.get('arch', 'mlp')
    if arch in ARCHS:
        return ARCHS[arch](**cfg)
    cfg.pop('arch', None)
    return Net(**cfg)


def _replace(tmp, dst, tries=40):
    """os.replace, retried: on Windows it fails while another process holds the target open."""
    for k in range(tries):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(0.25)


def save(model, path, extra=None):
    """Atomic (a crash never leaves a half-written checkpoint); a NaN / inf in any float tensor
    raises finite.NonFinite instead of writing it (docs/az.md 4.1)."""
    state = model.state_dict()
    check_state(state, path)
    tmp = path + f'.{os.getpid()}.tmp'
    torch.save({'config': model.config, 'state': state, 'extra': extra or {}}, tmp)
    _replace(tmp, path)


def load(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build(ck['config'])
    m.load_state_dict(ck['state'])
    m.to(device).eval()
    return m, ck.get('extra', {})


def _grid_index(cfg):
    """(plane feature indices in channel order, global feature indices) of a GridNet config."""
    g = cfg['grid']
    cells = g['h'] * g['w']
    segs = g.get('segments') or [[g.get('offset', 0), g['planes']]]
    planes = [o + i for o, k in segs for i in range(k * cells)]
    inplane = set(planes)
    return planes, [i for i in range(cfg['features']) if i not in inplane]


def make_net(info, game, arch='mlp', hidden=256, blocks=4, ch=64, heads=4, ff=0):
    """A new network for `game` (info = games.info(game)): 'mlp' (`Net`, hidden x blocks), 'grid'
    (`GridNet` over the game package's board planes, ch x blocks) or 'tokens' (`TokenNet` over
    the game package's object layout: d = hidden, layers = blocks, heads, ff (0: 2 x hidden))."""
    from . import package
    F_, A_ = info['num_features'], info['num_actions']
    if arch == 'grid':
        grid = package.active().grids.get(game)
        if not grid:
            raise SystemExit(f"arch 'grid': the game package has no grid for {game!r} (GamePackage.grids)")
        return GridNet(F_, A_, grid, ch, blocks, game=game).eval()
    if arch == 'tokens':
        layout = package.active().tokens_of(game)
        if not layout:
            raise SystemExit(f"arch 'tokens': the game package has no object layout for {game!r} (GamePackage.tokens)")
        return TokenNet(F_, A_, layout, d=hidden, layers=blocks, heads=heads, ff=ff or 2 * hidden, game=game).eval()
    if arch != 'mlp':
        raise SystemExit(f"unknown arch {arch!r} ('mlp', 'grid', 'tokens')")
    return Net(F_, A_, hidden, blocks, game=game).eval()


def widen(old, features=None, actions=None, grid=None):
    """A copy of `old` with `features` inputs / `actions` policy outputs (>= the old ones):
    new input columns get zero weights (the old outputs are unchanged), new actions zero
    weights and bias (logit 0). GridNet: `grid` = the new board grid (the game package's) when
    it adds planes (the old planes must come first, in the same order): the new planes get zero
    input weights in the first conv. TokenNet: its layout stays; new features (after the old
    ones) are new globals."""
    cfg = dict(old.config)
    cfg['features'] = features or cfg['features']
    cfg['actions'] = actions or cfg['actions']
    if cfg['features'] < old.config['features'] or cfg['actions'] < old.config['actions']:
        raise ValueError('widen only grows a network')
    if cfg.get('arch') == 'tokens':
        new = build(cfg)
        st = new.state_dict()
        for k, v in old.state_dict().items():
            if k == 'g.weight':                      # globals in feature order: new ones come last
                st[k].zero_()
                st[k][:, :v.shape[1]] = v
            elif k in ('gen.weight', 'gen.bias'):
                st[k].zero_()
                st[k][:v.shape[0]] = v
            else:
                st[k] = v
        new.load_state_dict(st)
        return new.eval()
    if grid is not None and cfg.get('arch') == 'grid':
        cfg['grid'] = dict(grid)
    new = build(cfg)
    st = new.state_dict()
    if cfg.get('arch') == 'grid' and new.config['grid'] != old.config['grid']:
        op, og = _grid_index(old.config)
        npl, ng = _grid_index(new.config)
        if npl[:len(op)] != op:
            raise ValueError('widen: the new grid must keep the old planes first, in order')
        if set(og) - set(ng):
            raise ValueError('widen: an old global became a plane')
        P0, P1 = old.config['grid']['planes'], new.config['grid']['planes']
        pos = {f: k for k, f in enumerate(og)}
        for k, v in old.state_dict().items():
            if k == 'stem.weight':                   # [ch, planes (+ parity), 3, 3]
                st[k].zero_()
                st[k][:, :P0] = v[:, :P0]
                if v.shape[1] > P0:                  # the row-parity plane comes last
                    st[k][:, P1:] = v[:, P0:]
            elif k == 'g.0.weight':
                st[k].zero_()
                for j, f in enumerate(ng):
                    if f in pos:
                        st[k][:, j] = v[:, pos[f]]
            elif k in ('other.weight', 'other.bias'):
                st[k].zero_()
                st[k][:v.shape[0]] = v
            else:
                st[k] = v
        new.load_state_dict(st)
        return new.eval()
    if cfg.get('arch') == 'grid':
        # new features come after the board planes: new globals columns; new actions come
        # after the cell block: new rows of the `other` head
        for k, v in old.state_dict().items():
            if k == 'g.0.weight':
                st[k].zero_()
                st[k][:, :v.shape[1]] = v
            elif k in ('other.weight', 'other.bias'):
                st[k].zero_()
                st[k][:v.shape[0]] = v
            else:
                st[k] = v
        new.load_state_dict(st)
        return new.eval()
    for k, v in old.state_dict().items():
        if k == 'inp.weight':
            st[k].zero_()
            st[k][:, :v.shape[1]] = v
        elif k in ('pol.weight', 'pol.bias'):
            st[k].zero_()
            st[k][:v.shape[0]] = v
        else:
            st[k] = v
    new.load_state_dict(st)
    return new.eval()
