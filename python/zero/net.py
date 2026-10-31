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
  aux    : optional (config key `aux` = the game's auxiliary targets [[name, width, loss], ...], the
           game package's `aux`; docs/az.md 15): one MLP from the trunk's pooled state (where value
           and margin read it) to the targets' raw outputs, returned only by `forward(x, n,
           aux=True)` - training, `infer.NetEvaluator.aux` -, so the search's inference and its CUDA
           graphs are unchanged. `aux_predict` turns the raw outputs into predictions by each
           target's loss ('multinomial': a softmax over the segment; 'poisson': exp, the rates;
           'mse': as they are). Built after every other parameter (a network with the head
           initialises the rest as one without), its last layer zero; `add_aux` gives a network
           without the head (an older checkpoint) a fresh one, everything else copied exactly
           (its policy / value / margin outputs unchanged), `strip_aux` removes it. The encoding
           guard ignores it: the head reads no input.

Checkpoints (`save`, atomic): {config, state, extra}; `load` rebuilds the network from the
config; `widen` maps a checkpoint onto a larger feature / action space (new inputs get zero
weights, new actions zero logits), so a widened network computes what the source did; a token
network also across a change of its object layout that only adds (`widen(..., layout=)`,
`token_feature_map`; docs/az.md 10.3).

The encoding guard (docs/az.md 10.4): `load` refuses a checkpoint whose network was built for
another encoding than the one this build's game feeds it (`check_encoding`: the game, the
feature and action counts, a token network's object layout, a conv network's board grid;
`EncodingMismatch`), since a network fed another encoding misreads it without an error (it
takes the floats where its own layout had them). `allow_mismatch=True` skips it, for the
callers that take such a network on purpose: tools/fork_run.py's widen, co-training's
resolver_init (both check what they write), tools/selfplay_digest.py (check_checkpoint_compat)
and the widen tests.
"""
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .finite import check_state
from .protocol import MAX_SEATS

AUX_LOSSES = ('multinomial', 'poisson', 'mse')


def aux_layout(aux):
    """Auxiliary targets ([[name, width, loss], ...], (name, width, loss) tuples or {name, width,
    loss} dicts: a config's `aux`, the game package's `aux_of`; None = none) -> [[name, width, loss],
    ...] (checked: names distinct, widths >= 1, a loss of AUX_LOSSES)."""
    out = []
    for t in aux or []:
        name, width, loss = (t['name'], t['width'], t.get('loss', 'mse')) if isinstance(t, dict) else t
        name, width, loss = str(name), int(width), str(loss)
        if width < 1 or loss not in AUX_LOSSES or any(o[0] == name for o in out):
            raise ValueError(f'auxiliary target {name!r} (width {width}, loss {loss!r}): a new name, width >= 1, '
                             f'a loss of {AUX_LOSSES}')
        out.append([name, width, loss])
    return out


def _aux_head(dim, head, layout):
    """The auxiliary head: an MLP dim -> head -> the targets' total width, its last layer zero."""
    m = nn.Sequential(nn.Linear(dim, head), nn.ReLU(), nn.Linear(head, sum(w for _, w, _ in layout)))
    nn.init.zeros_(m[2].weight)
    nn.init.zeros_(m[2].bias)
    return m


def _with_aux(model, dim, head, aux):
    """Give `model` its auxiliary head (config 'aux') when `aux` names targets (built last)."""
    model.aux_layout = aux_layout(aux)
    if model.aux_layout:
        model.config['aux'] = model.aux_layout
        model.aux_head = _aux_head(dim, head, model.aux_layout)


def _aux_out(model, z, rows=None):
    """The auxiliary head on the pooled states `z` (`rows`: of those rows only, an index tensor -
    the search's root belief rows, docs/search.md 3.8)."""
    if not model.aux_layout:
        raise ValueError('forward(aux=True): the network has no auxiliary head (net.add_aux)')
    return model.aux_head(z if rows is None else z[rows])


def aux_predict(layout, raw):
    """Raw auxiliary outputs [B, W] (`forward(..., aux=True)`'s fourth) -> {name: predictions [B, w]}
    by each target's loss: 'multinomial' the softmax over its segment (a distribution: MTG Zero's
    belief, x the hand size = expected counts), 'poisson' exp (rates = expected counts), 'mse' raw."""
    out, k = {}, 0
    for name, w, loss in aux_layout(layout):
        r = raw[:, k:k + w].float()
        if loss == 'multinomial':
            out[name] = torch.softmax(r, 1)
        elif loss == 'poisson':
            out[name] = torch.exp(r.clamp(max=30.0))
        else:
            out[name] = r
        k += w
    return out


class Block(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.ln = nn.LayerNorm(h)
        self.fc1 = nn.Linear(h, h)
        self.fc2 = nn.Linear(h, h)

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(self.ln(x))))


class Net(nn.Module):
    def __init__(self, features, actions, hidden=256, blocks=4, seats=MAX_SEATS, game='toy', head=128, aux=None):
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
        _with_aux(self, hidden, head, aux)

    def forward(self, x, n_players, aux=False, aux_rows=None):
        """x [B, >= features] (a narrower network reads a prefix: the evaluator refuses that
        unless told otherwise, docs/az.md 10.4), n_players [B] -> policy logits [B, actions],
        value logits [B, seats] (absent seats -1e9), margin [B, seats] (+ with `aux` the raw
        auxiliary outputs [B, aux width]; `aux_rows`, an index tensor: of those rows only)."""
        h = F.relu(self.inp(x[:, :self.features]))
        for b in self.blocks:
            h = b(h)
        h = F.relu(self.ln(h))
        present = torch.arange(self.seats, device=x.device)[None, :] < n_players[:, None]
        vlog = self.val(h).masked_fill(~present, -1e9)
        if aux:
            return self.pol(h), vlog, self.mar(h), _aux_out(self, h, aux_rows)
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


def grid_spec(grid):
    """A board grid of `GridNet` in its normal form (stored in the checkpoint config): {planes,
    h, w, offset, cell_action0, hex, segments}."""
    segs = [[int(o), int(k)] for o, k in grid.get('segments') or [[grid.get('offset', 0), grid['planes']]]]
    return {'planes': sum(k for _, k in segs), 'h': int(grid['h']), 'w': int(grid['w']), 'offset': segs[0][0],
            'cell_action0': int(grid.get('cell_action0', 0)), 'hex': grid.get('hex', 'odd-r'), 'segments': segs}


class GridNet(nn.Module):
    """grid: {planes, h, w, offset (first plane feature), cell_action0 (the action of cell 0;
    cells are row-major, as the planes), hex ('odd-r' adds a row-parity plane, or none),
    segments (optional: [[offset, planes], ...] when the planes sit in several blocks of the
    features, e.g. a layout that appends planes after its globals; default [[offset, planes]])}.
    Every feature outside the plane blocks is a global."""

    def __init__(self, features, actions, grid, ch=64, blocks=6, seats=MAX_SEATS, game='toy', head=128, ghid=128,
                 arch='grid', aux=None):
        super().__init__()
        grid = grid_spec(grid)
        segs = grid['segments']
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
        _with_aux(self, 256, head, aux)

    def forward(self, x, n_players, aux=False, aux_rows=None):
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
        if aux:
            return pol, vlog, self.mar(e), _aux_out(self, e, aux_rows)
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
    out = {'vocab': vocab, 'segments': segs, 'bags': bags}
    if layout.get('globals') is not None:
        # declared global blocks (optional; the key only when set, as id_actions): named runs of
        # globals in feature order that cover every feature outside the segments and bags. They
        # change nothing in the network (its globals are those features, in feature order); a
        # widen across a layout change maps the old globals onto the new ones by block name
        # (`widen`, docs/az.md 10.3)
        blocks, last = [], -1
        for g in layout['globals']:
            blk = {'name': str(g['name']), 'offset': int(g['offset']), 'count': int(g['count'])}
            if blk['count'] < 1 or blk['offset'] <= last:
                raise ValueError(f'tokens layout: global block {blk["name"]} is empty or out of feature order')
            if any(b['name'] == blk['name'] for b in blocks):
                raise ValueError(f'tokens layout: two global blocks named {blk["name"]}')
            claim(blk['name'], blk['offset'], blk['count'])
            last = blk['offset']
            blocks.append(blk)
        if len(used) != features:
            raise ValueError(f'tokens layout: the global blocks leave {features - len(used)} features undeclared')
        out['globals'] = blocks
    for k in ('name', 'version'):
        # the layout's identity (optional; the key only when set, as id_actions): nothing in the
        # network reads it; the encoding guard compares it when the checkpoint and the build both
        # carry one (`check_encoding`, docs/az.md 10.4)
        if layout.get(k) is not None:
            out[k] = layout[k]
    return out


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
      * globals [{name, offset, count}, ...] (optional): those globals as named blocks in feature
        order, covering all of them. The network is the same with or without them; `widen` maps
        a checkpoint across a layout change by them (docs/az.md 10.3).
      * name, version (optional): the layout's identity (MTG Zero's layout files: version 5),
        kept in the checkpoint config; only the encoding guard reads them (docs/az.md 10.4).

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
                 head=128, arch='tokens', aux=None):
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
        _with_aux(self, 2 * d, head, aux)

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

    def _global(self, x):
        """The global token's input [B, d] (before its position embedding): the globals' map plus
        the bags' (each bag pooled as the count-weighted sum of the identity embeddings and its
        total)."""
        B = x.shape[0]
        g = self.g(x.index_select(1, self.gidx))
        if self.bag is not None:
            pools = []
            for b in self.bags:
                o, k, w = b['offset'], b['count'], b['width']
                c = x[:, o:o + k * w].reshape(B, k, w)
                pools.append(torch.cat([c @ self.emb.weight[:w], c.sum(-1, keepdim=True)], -1))
            g = g + self.bag(torch.cat(pools, 1).flatten(1))
        return g

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
        g = self._global(x)
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

    def _trunk(self, hc, nr, fc, ic=None):
        """The blocks and the policy logits of packed rows: hc [Bc, Tc, d] (row b's nr[b] tokens
        first, then padding), fc [Bc, Tc] the tokens' positions in the full token axis, ic [Bc, Tc]
        their identities (a layout with id_actions) -> (zc [Bc, 2 d] the pooled state, pc [Bc,
        actions + 1] the policy logits; the last column a dummy for the padding's and the empty
        slots' scores). One chunk of `forward`, or a padded batch of `forward_static`."""
        Tc = hc.shape[1]
        keep = torch.arange(Tc, device=hc.device)[None, :] < nr[:, None]
        k4 = keep[:, None, None, :]
        for blk in self.blocks:
            hc = blk(hc, k4)
        hc = self.ln(hc) * keep[..., None]
        zc = torch.cat([hc[:, 0], hc.sum(1) / nr[:, None].to(hc.dtype)], 1)
        pc = F.pad(self.gen(zc), (0, 1))
        cols = self.tok_cols[fc]                                      # [Bc, Tc, K]
        acts = self.tok_acts[fc]
        if self.n_id:                                                 # actions by identity
            ia = self.tok_idbase[fc] + ic
            ia = torch.where((ic < self.vocab) & (ia < self.actions), ia, self.actions)
            acts = torch.where(self.tok_isid[fc], ia[..., None], acts)
        acts = torch.where(keep[..., None] & (cols >= 0), acts, self.actions)
        lg = self.ptr(F.gelu(self.pk(hc) + self.pq(zc)[:, None])).gather(2, cols.clamp(min=0))
        return zc, pc.scatter(1, acts.flatten(1), lg.flatten(1))

    def _heads(self, z, pol, n_players, aux, aux_rows):
        present = torch.arange(self.seats, device=z.device)[None, :] < n_players[:, None]
        vlog = self.val(z).masked_fill(~present, -1e9)
        if aux:
            return pol[:, :self.actions], vlog, self.mar(z), _aux_out(self, z, aux_rows)
        return pol[:, :self.actions], vlog, self.mar(z)

    def forward(self, x, n_players, aux=False, aux_rows=None):
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
            zc, pc = self._trunk(h[r, :Tc], nr, full[r, :Tc], idt[r, :Tc] if self.n_id else None)
            z = z.index_copy(0, r, zc)
            pol = pol.index_copy(0, r, pc)
        return self._heads(z, pol, n_players, aux, aux_rows)

    # ---- static shapes (CUDA graphs: infer.TokenGraphRunner, docs/az.md 10.6)
    def present_index(self):
        """Per slot of the full token axis after the global token (the segments' slots in layout
        order): the feature index of its present flag, -1 for a segment without one (always
        present) -> a list. The host computes a batch's tokens from it (`forward_static`'s
        `full` and `ntok`)."""
        out = []
        for s in self.segs:
            for slot in range(s['count']):
                out.append(-1 if s['present'] is None else s['offset'] + slot * s['width'] + s['present'])
        return out

    def _static_identities(self, x):
        """Every slot's identity and its embedding, for the segments with one -> {segment index: (ids
        [B, count], embeddings [B, count, d])}: `_identity`'s digits, computed for all the slots of the
        segments that share a radix at once (one gather of their digit floats, a few kernels per digit:
        `forward_static` stays at a few hundred kernels, not ~30 per segment). The column indexes are
        built once per device (an attribute, not a buffer: the state is unchanged)."""
        cache = getattr(self, '_static_ident', None)
        if cache is None or cache[0] != x.device:
            groups = {}
            for j, s in enumerate(self.segs):
                if s['id'] is not None:
                    radix = tuple(s['id']['radix'])
                    cols = [s['offset'] + slot * s['width'] + s['id']['at'] + c
                            for slot in range(s['count']) for c in range(sum(radix))]
                    groups.setdefault(radix, []).append((j, s['count'], cols))
            plan = []
            for radix, members in groups.items():
                cols = torch.tensor([c for _, _, cs in members for c in cs], dtype=torch.long, device=x.device)
                parts, k = [], 0
                for j, cnt, _ in members:
                    parts.append((j, k, k + cnt))
                    k += cnt
                plan.append((radix, cols, k, parts))
            cache = self._static_ident = (x.device, plan)
        B, out = x.shape[0], {}
        for radix, cols, ns, parts in cache[1]:
            blk = x.index_select(1, cols).view(B, ns, sum(radix))
            at, idx, ok = 0, None, None
            for r in radix:
                d = blk[..., at:at + r]
                dig = (d @ torch.arange(r, dtype=x.dtype, device=x.device)).round().long()
                has = d.sum(-1) > 0.5
                idx = dig if idx is None else idx * r + dig
                ok = has if ok is None else ok & has
                at += r
            i = torch.where(ok & (idx < self.vocab), idx, torch.full_like(idx, self.vocab))
            e = self.emb(i)
            for j, a, b in parts:
                out[j] = (i[:, a:b], e[:, a:b])
        return out

    def forward_static(self, x, n_players, full, ntok, aux=False):
        """`forward` with shapes fixed by the inputs' (no host sync, no data-dependent shape: a CUDA
        graph can capture it): every slot is embedded, present or not, and each row's tokens are
        gathered by `full` [B, T] (its present tokens' positions in the full token axis, in layout
        order, the global token's 0 first, then padding: any position, masked out), `ntok` [B] its
        token count (>= 1). All rows are run as one chunk of T tokens. The outputs are `forward`'s
        (`aux`: the head on every row); a row's outputs depend only on it and on the batch's shape
        (the kernels' blocking: float32 rounding)."""
        x = x[:, :self.features]
        B = x.shape[0]
        g = self._global(x)
        toks = [(g + self.pos[0])[:, None]]
        ids = [torch.full((B, 1), self.vocab, dtype=torch.long, device=x.device)] if self.n_id else None
        ident = self._static_identities(x)
        start = 1
        for j, (s, lin) in enumerate(zip(self.segs, self.seg_in)):
            o, k, w = s['offset'], s['count'], s['width']
            sl = x[:, o:o + k * w].reshape(B, k, w)
            t = lin(sl) + self.pos[start:start + k]
            if s['id'] is not None:
                i, e = ident[j]
                t = t + e
            else:
                i = torch.full((B, k), self.vocab, dtype=torch.long, device=x.device)
            if ids is not None:
                ids.append(i)
            toks.append(t)
            start += k
        h = torch.cat(toks, 1).gather(1, full[..., None].expand(-1, -1, self.d))
        idt = torch.cat(ids, 1).gather(1, full) if ids is not None else None
        z, pol = self._trunk(h, ntok, full, idt)
        return self._heads(z, pol, n_players, aux, None)

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


def add_aux(model, aux, seed=0):
    """`model` with the auxiliary head for the targets `aux` (`aux_layout`): a network without one
    (an older checkpoint) -> a copy with every other parameter and buffer exactly as they were, its
    policy / value / margin outputs unchanged, and a fresh head (initialised from `seed`, the global
    torch RNG untouched; its last layer zero: it predicts the uniform / unit / zero output and sends
    the trunk no gradient until it has learned something). A network with that head already: `model`
    itself; with another one: ValueError (strip_aux first)."""
    aux = aux_layout(aux)
    have = aux_layout(model.config.get('aux'))
    if have == aux:
        return model
    if have or not aux:
        raise ValueError(f'add_aux: the network has the auxiliary head {have}, wanted {aux} (strip_aux first)')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        new = build(dict(model.config, aux=aux))
    missing, unexpected = new.load_state_dict(model.state_dict(), strict=False)
    if unexpected or not missing or any(not k.startswith('aux_') for k in missing):
        raise RuntimeError(f'add_aux: state mismatch (missing {missing}, unexpected {unexpected})')
    dev = next(model.parameters()).device
    return new.to(dev).train(model.training)


def strip_aux(model):
    """A copy of `model` without its auxiliary head (config and state; the same policy / value /
    margin), e.g. for code that predates the head."""
    if not model.config.get('aux'):
        return model
    new = build({k: v for k, v in model.config.items() if k != 'aux'})
    new.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith('aux_')})
    dev = next(model.parameters()).device
    return new.to(dev).train(model.training)


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


def load(path, device='cpu', game=None, allow_mismatch=False):
    """A checkpoint -> (network, extra). The encoding guard first (docs/az.md 10.4): a network
    built for another encoding than the one this build's `game` feeds it (default: the game in its
    config) raises EncodingMismatch (`check_encoding`). `allow_mismatch=True` skips the guard, for
    the callers that take such a network on purpose: tools/fork_run.py's widen, co-training's
    resolver_init (both check what they write), tools/selfplay_digest.py and the widen tests."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if not allow_mismatch:
        check_encoding(ck['config'], game, path)
    m = build(ck['config'])
    m.load_state_dict(ck['state'])
    m.to(device).eval()
    return m, ck.get('extra', {})


# ---------------------------------------------------------------- the encoding guard (docs/az.md 10.4)
class EncodingMismatch(ValueError):
    """A network built for another encoding than the one this build's game feeds it
    (`check_encoding`): fed this build's inputs it would misread them without an error."""


class HeadMissing(EncodingMismatch):
    """A network without the auxiliary head a setting needs (the root's belief, `*_belief_tilt` > 0,
    docs/search.md 3.8), or with a head for other targets than the game's: it cannot serve the run. An
    EncodingMismatch, so every path that refuses a checkpoint refuses it (a run exits with
    EXIT_REFUSED and is not restarted; a remote reports it); never a search dealing uniformly in
    silence (docs/az.md 18.3)."""


def check_belief_head(model, game, what='belief_tilt'):
    """`model` can answer root belief rows for `game` (docs/search.md 3.8): it has the auxiliary head,
    for exactly the game's targets (names and widths in order) -> None, else HeadMissing (`what`: the
    setting that asks, for the message)."""
    from . import games as G
    have = [(t[0], int(t[1])) for t in aux_layout(getattr(model, 'aux_layout', None) or model.config.get('aux'))]
    want = [(n, int(w)) for n, w in G.info(game).get('aux_layout') or []]
    c = getattr(model, 'config', {})
    if not want:
        raise HeadMissing(f'{what}: the game {game!r} has no auxiliary targets for a belief to predict')
    if not have:
        raise HeadMissing(f'{what}: the network (game {c.get("game")!r}, arch {c.get("arch", "mlp")!r}) has no auxiliary '
                          f'head: the root\'s belief needs one trained on the game\'s targets {[n for n, _ in want]} (a run '
                          f'with aux_<name>_w > 0, docs/az.md 15; a checkpoint from before it cannot serve {what})')
    if have != want:
        raise HeadMissing(f'{what}: the network\'s auxiliary head predicts {have}, the game\'s targets are {want}')


ADVICE = ('Move a run across an encoding change with `python tools/fork_run.py <checkpoint> <new run>` (an exact '
          "widen when the change only adds inputs; docs/az.md 10.4), or use a build of the network's encoding.")


def build_encoding(game):
    """What this build's game `game` feeds a network: {game, features, actions, layout (the game
    package's object layout, normalised: arch 'tokens'), grid (its board grid: arch 'grid')}; the
    keys layout / grid only when the package has one. An unknown game: EncodingMismatch."""
    from . import games as G, package
    try:
        info = G.info(game)
    except ImportError:
        raise
    except Exception as e:                         # noqa: BLE001 - the engine's unknown game
        raise EncodingMismatch(f'game {game!r} is not a game of this build ({e})') from None
    enc = {'game': game, 'features': int(info['num_features']), 'actions': int(info['num_actions'])}
    pkg = package.active()
    lay = pkg.tokens_of(game)
    if lay:
        enc['layout'] = tokens_layout(lay, enc['features'], enc['actions'])
    grid = pkg.grids.get(game)
    if grid:
        enc['grid'] = grid_spec(grid)
    return enc


def _short(v, n=48):
    s = repr(v)
    return s if len(s) <= n else s[:n - 3] + '...'


def _layout_diff(a, b):
    """How token layout `b` (the build's) differs from `a` (the network's), in the parts the
    network reads (both normalised) -> a list of short strings."""
    out = []
    if a['vocab'] != b['vocab']:
        out.append(f'vocabulary {a["vocab"]} -> {b["vocab"]}')
    an, bn = [s['name'] for s in a['segments']], [s['name'] for s in b['segments']]
    if an != bn:
        out.append(f'segments {_short(an)} -> {_short(bn)}')
    else:
        for x, y in zip(a['segments'], b['segments']):
            ks = [k for k in sorted(set(x) | set(y)) if x.get(k) != y.get(k)]
            if ks:
                out.append(f'segment {x["name"]}: ' + ', '.join(f'{k} {_short(x.get(k), 24)} -> {_short(y.get(k), 24)}'
                                                               for k in ks[:2]) + (' ...' if len(ks) > 2 else ''))
    ab, bb = {g['name']: g for g in a['bags']}, {g['name']: g for g in b['bags']}
    for name in sorted(set(ab) | set(bb)):
        if ab.get(name) != bb.get(name):
            out.append(f'bag {name}: {_short(ab.get(name), 40)} -> {_short(bb.get(name), 40)}')
    return out[:3] + ([f'{len(out) - 3} more'] if len(out) > 3 else [])


def _count(v):
    return f'{v:,}' if isinstance(v, int) else repr(v)


def _describe(e, arch=None):
    s = f'{e.get("game")!r}: {_count(e.get("features"))} features, {_count(e.get("actions"))} actions'
    lay = e.get('layout')
    if lay is not None:
        ident = (f' {lay["name"]}' if lay.get('name') is not None else '') + \
                (f' v{lay["version"]}' if lay.get('version') is not None else '')
        nb = len(lay['bags'])
        s += (f', token layout{ident or " without a version"} ({len(lay["segments"])} segments, {nb} '
              f'bag{"" if nb == 1 else "s"})')
    if e.get('grid') is not None:
        s += f', board grid of {e["grid"]["planes"]} planes'
    if arch:
        s += f', arch {arch!r}'
    return s


def encoding_mismatch(config, game=None):
    """The differences between the encoding a network `config` was built for (a checkpoint's) and
    the one this build's `game` (default: the config's game) feeds it -> None when it matches,
    else (differences, the network's description, the build's). Judged by: the game; the feature
    and action counts; a token network's object layout in the parts the network reads (vocabulary,
    segments, bags: not the declared global blocks, which change nothing in it) and the layout's
    version / name when both sides carry one, if the game package has a layout for the game; a conv
    network's board grid, if the package has one. A network without layout or grid metadata (arch
    'mlp', older checkpoints) is judged by the game and the counts alone."""
    from . import package
    mine = config.get('game')
    game = game or mine or package.active().game
    enc = build_encoding(game)
    arch = config.get('arch', 'mlp')
    net = {'game': mine, 'features': config.get('features'), 'actions': config.get('actions')}
    diffs = []
    if mine is not None and mine != game:
        diffs.append(f'game {mine!r} -> {game!r}')
    for k in ('features', 'actions'):
        if net[k] != enc[k]:
            diffs.append(f'{k} {net[k]} -> {enc[k]}')
    build_desc = dict(enc)
    if arch == 'tokens':
        lay = config.get('layout') or {}
        try:
            lay = tokens_layout(lay, int(net['features']), int(net['actions']))
        except (KeyError, TypeError, ValueError):
            pass
        net['layout'] = lay
        want = enc.get('layout')
        if want is not None and isinstance(lay, dict) and 'segments' in lay:
            reads = ('vocab', 'segments', 'bags')
            if any(lay.get(k) != want.get(k) for k in reads):
                diffs.append('token layout: ' + '; '.join(_layout_diff(lay, want)))
            for k in ('name', 'version'):
                if lay.get(k) is not None and want.get(k) is not None and lay[k] != want[k]:
                    diffs.append(f'token layout {k} {lay[k]!r} -> {want[k]!r}')
    else:
        build_desc.pop('layout', None)
    if arch == 'grid':
        try:
            net['grid'] = grid_spec(config['grid'])
        except (KeyError, TypeError, ValueError):
            net['grid'] = config.get('grid')
        if enc.get('grid') is not None and net['grid'] != enc['grid']:
            diffs.append(f'board grid {_short(net["grid"], 60)} -> {_short(enc["grid"], 60)}')
    else:
        build_desc.pop('grid', None)
    if not diffs:
        return None
    return diffs, _describe(net, arch), _describe(build_desc)


def check_encoding(config, game=None, where=None):
    """Raise EncodingMismatch if a network `config` was built for another encoding than the one
    this build's `game` (default: the config's game) feeds it (`encoding_mismatch`); `where`: the
    checkpoint (the message)."""
    m = encoding_mismatch(config, game)
    if m is None:
        return
    diffs, net, build_ = m
    head = f'{where}: ' if where else ''
    raise EncodingMismatch(f"{head}a network of another encoding than this build's ({'; '.join(diffs)}). The network: "
                           f"{net}. This build: {build_}. Fed this build's inputs it would misread them without an "
                           f"error. {ADVICE}")


def check_checkpoint(path, game=None):
    """`check_encoding` of the checkpoint at `path` (its config only; the network is not built)."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    check_encoding(ck['config'], game, path)


def _grid_index(cfg):
    """(plane feature indices in channel order, global feature indices) of a GridNet config."""
    g = cfg['grid']
    cells = g['h'] * g['w']
    segs = g.get('segments') or [[g.get('offset', 0), g['planes']]]
    planes = [o + i for o, k in segs for i in range(k * cells)]
    inplane = set(planes)
    return planes, [i for i in range(cfg['features']) if i not in inplane]


def make_net(info, game, arch='mlp', hidden=256, blocks=4, ch=64, heads=4, ff=0, aux=None):
    """A new network for `game` (info = games.info(game)): 'mlp' (`Net`, hidden x blocks), 'grid'
    (`GridNet` over the game package's board planes, ch x blocks) or 'tokens' (`TokenNet` over
    the game package's object layout: d = hidden, layers = blocks, heads, ff (0: 2 x hidden)); `aux`:
    its auxiliary targets (`aux_layout`; None: no auxiliary head, the network of before)."""
    kw = {'aux': aux} if aux else {}
    from . import package
    F_, A_ = info['num_features'], info['num_actions']
    if arch == 'grid':
        grid = package.active().grids.get(game)
        if not grid:
            raise SystemExit(f"arch 'grid': the game package has no grid for {game!r} (GamePackage.grids)")
        return GridNet(F_, A_, grid, ch, blocks, game=game, **kw).eval()
    if arch == 'tokens':
        layout = package.active().tokens_of(game)
        if not layout:
            raise SystemExit(f"arch 'tokens': the game package has no object layout for {game!r} (GamePackage.tokens)")
        return TokenNet(F_, A_, layout, d=hidden, layers=blocks, heads=heads, ff=ff or 2 * hidden, game=game,
                        **kw).eval()
    if arch != 'mlp':
        raise SystemExit(f"unknown arch {arch!r} ('mlp', 'grid', 'tokens')")
    return Net(F_, A_, hidden, blocks, game=game, **kw).eval()


def _global_blocks(lay, features):
    """The globals of a normalised token layout as [(block name or None, [feature, ...]), ...] in
    feature order: its declared blocks, else one unnamed block of every feature outside the
    segments and bags."""
    if lay.get('globals') is not None:
        return [(b['name'], list(range(b['offset'], b['offset'] + b['count']))) for b in lay['globals']]
    inside = set()
    for s in lay['segments']:
        inside.update(range(s['offset'], s['offset'] + s['count'] * s['width']))
    for b in lay['bags']:
        inside.update(range(b['offset'], b['offset'] + b['count'] * b['width']))
    return [(None, [i for i in range(features) if i not in inside])]


def token_feature_map(old_lay, old_features, new_lay, new_features):
    """Where each feature of a token layout went in a wider one (docs/az.md 10.3): a list, per old
    feature, of its new index. Raises ValueError unless `new_lay` keeps the old layout's tokens and
    meanings and only adds inputs:
      * the same segments (names, order, slot counts, present flags, identities, actions): a new
        segment, or more slots, would add tokens to the attention; a segment's slots may get
        wider, the old floats first in each slot (the new ones after them);
      * every old bag (by name; the new layout may add bags) with the same width and its old rows
        first (new rows after them);
      * the old globals: by block name when the old layout declares blocks (the old floats first
        in each), else all of them as the first globals of the new layout in feature order (new
        globals after the old ones in feature order).
    Both layouts normalised (`tokens_layout`)."""
    if int(new_lay['vocab']) != int(old_lay['vocab']):
        raise ValueError(f'widen: the vocabulary changed ({old_lay["vocab"]} -> {new_lay["vocab"]})')
    os_, ns = old_lay['segments'], new_lay['segments']
    if [s['name'] for s in os_] != [s['name'] for s in ns]:
        raise ValueError(f'widen: a token layout must keep its segments (names, order): '
                         f'{[s["name"] for s in os_]} -> {[s["name"] for s in ns]} (new tokens change the attention)')
    fmap = [None] * old_features
    for a, b in zip(os_, ns):
        same = ('count', 'present', 'id', 'actions')
        if any(a.get(k) != b.get(k) for k in same) or a.get('id_actions') != b.get('id_actions'):
            raise ValueError(f'widen: segment {a["name"]} changed its slots, present flag, identity or actions')
        if b['width'] < a['width']:
            raise ValueError(f'widen: segment {a["name"]} got narrower ({a["width"]} -> {b["width"]})')
        for s in range(a['count']):
            for j in range(a['width']):
                fmap[a['offset'] + s * a['width'] + j] = b['offset'] + s * b['width'] + j
    nb = {b['name']: b for b in new_lay['bags']}
    for a in old_lay['bags']:
        b = nb.get(a['name'])
        if b is None or b['width'] != a['width'] or b['count'] < a['count']:
            raise ValueError(f'widen: bag {a["name"]} is gone, changed width or lost rows')
        for i in range(a['count'] * a['width']):
            fmap[a['offset'] + i] = b['offset'] + i
    og, ng = _global_blocks(old_lay, old_features), _global_blocks(new_lay, new_features)
    if og[0][0] is None:                         # no declared blocks: a prefix in feature order
        old_g, new_g = og[0][1], [f for _, fs in ng for f in fs]
        if len(new_g) < len(old_g):
            raise ValueError(f'widen: {len(old_g)} globals -> {len(new_g)}')
        pairs = zip(old_g, new_g)
    else:
        nd = dict(ng)
        pairs = []
        for name, fs in og:
            if name not in nd or len(nd[name]) < len(fs):
                raise ValueError(f'widen: global block {name} is gone or lost floats')
            pairs.extend(zip(fs, nd[name]))
    for f, t in pairs:
        fmap[f] = t
    if any(t is None for t in fmap) or len(set(fmap)) != old_features or max(fmap) >= new_features:
        raise ValueError('widen: the old features do not map one to one into the new layout')
    return fmap


def layout_sans_identity(lay):
    """A normalised token layout without its identity keys (name, version): the layout as the
    network reads it, plus its declared global blocks."""
    return {k: v for k, v in lay.items() if k not in ('name', 'version')}


def _widen_tokens(old, cfg, lay):
    """`widen` of a TokenNet onto the layout `lay` (normalised; module docs of `widen`)."""
    ol = old.config['layout']
    F0, F1 = old.config['features'], cfg['features']
    fmap = token_feature_map(ol, F0, lay, F1)
    cfg = dict(cfg, layout=lay)
    new = build(cfg)
    st = new.state_dict()
    old_st = old.state_dict()
    # the globals: columns of `g` in feature order (TokenNet.gidx)
    ogi = old.gidx.tolist()
    npos = {f: k for k, f in enumerate(new.gidx.tolist())}
    gcols = [npos[fmap[f]] for f in ogi]
    # the bags: rows of (d + 1) columns of `bag`, in bag order
    d = old.d

    def rows(l):
        out, r = {}, 0
        for b in l['bags']:
            for k in range(b['count']):
                out[(b['name'], k)] = r
                r += 1
        return out
    orow, nrow = rows(ol), rows(lay)
    seg = {f'seg_in.{i}.weight' for i in range(len(ol['segments']))}
    for k, v in old_st.items():
        t = st[k]
        if k == 'g.weight':
            t.zero_()
            t[:, gcols] = v
        elif k == 'bag.weight':
            t.zero_()
            for key, r in orow.items():
                n = nrow[key]
                t[:, n * (d + 1):(n + 1) * (d + 1)] = v[:, r * (d + 1):(r + 1) * (d + 1)]
        elif k in seg:                               # the slot's old floats first
            t.zero_()
            t[:, :v.shape[1]] = v
        elif k in ('gen.weight', 'gen.bias'):
            t.zero_()
            t[:v.shape[0]] = v
        else:
            if t.shape != v.shape:
                raise ValueError(f'widen: {k} {tuple(v.shape)} -> {tuple(t.shape)}')
            st[k] = v
    if 'bag.weight' in st and 'bag.weight' not in old_st:   # the old network had no bags
        st['bag.weight'].zero_()
        st['bag.bias'].zero_()
    new.load_state_dict(st)
    return new.eval()


def widen(old, features=None, actions=None, grid=None, layout=None):
    """A copy of `old` with `features` inputs / `actions` policy outputs (>= the old ones):
    new input columns get zero weights (the old outputs are unchanged), new actions zero
    weights and bias (logit 0). GridNet: `grid` = the new board grid (the game package's) when
    it adds planes (the old planes must come first, in the same order): the new planes get zero
    input weights in the first conv. TokenNet: without `layout`, or with the layout it has (its
    name / version may differ: the given one's are recorded), its layout stays and new features
    (after the old ones) are new globals; with a new `layout` (the
    game package's, `GamePackage.tokens`) the widen follows it segment by segment
    (`token_feature_map`: slots wider at their ends, bags with more rows or new bags, more
    globals; the same tokens): every new input column gets zero weights in its projection (the
    segment's input map, the bag map, the global map), so on any input of the new layout the
    network computes what `old` computed on the old features of it (docs/az.md 10.3)."""
    cfg = dict(old.config)
    cfg['features'] = features or cfg['features']
    cfg['actions'] = actions or cfg['actions']
    if cfg['features'] < old.config['features'] or cfg['actions'] < old.config['actions']:
        raise ValueError('widen only grows a network')
    if cfg.get('arch') == 'tokens' and layout is not None:
        lay = tokens_layout(layout, cfg['features'], cfg['actions'])
        if layout_sans_identity(lay) != layout_sans_identity(old.config['layout']):
            return _widen_tokens(old, cfg, lay)
        # the same layout: the old widen below, the network recording the new layout's identity
        # (name / version: e.g. a checkpoint from before they were recorded)
        cfg['layout'] = lay
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
