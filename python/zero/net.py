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
           guard ignores it: the head reads no input. `widen_aux` gives a head new targets (docs/az.md
           37: a game that declares more targets than a checkpoint's head predicts): the head's last
           layer stays as it was and the new targets' outputs come from a layer of their own on the
           same hidden layer (config `aux_parts`, `aux_more`), zero, so every output of before is
           computed by the same layers, bit for bit; `load` does it for a checkpoint whose head
           predicts a subset of the game's declared targets (`game_aux`).
  readers: optional (config key `readers`, the game package's `readers`; docs/az.md 36): per
           reader, a small MLP from declared ranges of the features (a token network: its bags and
           globals) to one auxiliary target's raw outputs, trained by that target's loss; its
           prediction is also an input of the trunk, through a projection that starts at zero
           (`Net`: the first layer's input; `GridNet`: the globals' embedding; `TokenNet`: the global
           token's input). Every forward computes them (they feed the trunk); `forward(..., aux=True)`
           appends their raw outputs after the head's. `add_reader` gives a network without readers
           fresh ones, everything else copied exactly (all its outputs unchanged).

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


def aux_parts_of(layout, parts):
    """The split of an auxiliary head's last layer (config `aux_parts`, docs/az.md 37) checked against its
    layout: [[target name, ...], ...] - part 0 the head's own last layer (`aux_head[2]`), part k >= 1 the layer
    `aux_more[k - 1]` of the targets the k-th `widen_aux` added; every target of `layout` in exactly one part,
    no part empty -> (parts, perm): perm the layout's columns in the parts' concatenated outputs (None when
    they are in the layout's order). No parts, or one part in the layout's order: (None, None) - the head of
    before (one layer)."""
    lay = aux_layout(layout)
    if not parts:
        return None, None
    parts = [[str(n) for n in p] for p in parts]
    names = [n for n, _, _ in lay]
    flat = [n for p in parts for n in p]
    if any(not p for p in parts) or len(set(flat)) != len(flat) or sorted(flat) != sorted(names):
        raise ValueError(f'aux_parts {parts}: every auxiliary target of {names} in exactly one part, no part empty')
    if len(parts) == 1 and flat == names:
        return None, None
    width = {n: w for n, w, _ in lay}
    start, k = {}, 0
    for n in flat:
        start[n] = k
        k += width[n]
    perm = [start[n] + i for n in names for i in range(width[n])]
    return parts, (None if perm == list(range(k)) else perm)


def _with_aux(model, dim, head, aux, parts=None):
    """Give `model` its auxiliary head (config 'aux') when `aux` names targets (built last); `parts`
    (config 'aux_parts', a widened head: `widen_aux`): its last layer split by targets, the parts after
    the first as `aux_more` (zero), registered right after the head (before the readers)."""
    model.aux_layout = aux_layout(aux)
    if model.aux_layout:
        model.config['aux'] = model.aux_layout
        parts, perm = aux_parts_of(model.aux_layout, parts)
        if parts is None:
            model.aux_head = _aux_head(dim, head, model.aux_layout)
            return
        model.config['aux_parts'] = parts
        spec = {n: [n, w, loss] for n, w, loss in model.aux_layout}
        model.aux_head = _aux_head(dim, head, [spec[n] for n in parts[0]])
        if len(parts) > 1:
            model.aux_more = nn.ModuleList(nn.Linear(head, sum(spec[n][1] for n in p)) for p in parts[1:])
            for lin in model.aux_more:
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
        if perm is not None:
            model.register_buffer('aux_perm', torch.tensor(perm, dtype=torch.long), persistent=False)


def _aux_raw(model, z):
    """The auxiliary head's raw outputs on the pooled states `z`, in its layout's order: one MLP; a widened
    head (docs/az.md 37) runs its hidden layer once and each part's last layer on it - the first part the
    layer of before, so its targets' outputs are the same computation bit for bit - concatenated and put in
    the layout's order."""
    more, perm = getattr(model, 'aux_more', None), getattr(model, 'aux_perm', None)
    if more is None and perm is None:
        return model.aux_head(z)
    h = model.aux_head[1](model.aux_head[0](z))
    out = model.aux_head[2](h)
    if more is not None:
        out = torch.cat([out] + [lin(h) for lin in more], 1)
    return out if perm is None else out.index_select(1, perm)


def _aux_out(model, z, rows=None, rraw=None):
    """The auxiliary head on the pooled states `z` (`rows`: of those rows only, an index tensor -
    the search's root belief rows, docs/search.md 3.8), followed by the readers' raw outputs `rraw`
    (docs/az.md 36; None: no readers)."""
    if not model.aux_layout and rraw is None:
        raise ValueError('forward(aux=True): the network has no auxiliary head (net.add_aux)')
    out = _aux_raw(model, z if rows is None else z[rows]) if model.aux_layout else None
    if rraw is None:
        return out
    rr = rraw if rows is None else rraw[rows]
    return rr if out is None else torch.cat([out, rr.to(out.dtype)], 1)


def _predict(loss, r):
    """A target's predictions from its raw outputs r [B, w] (float) by its loss (`aux_predict`)."""
    if loss == 'multinomial':
        return torch.softmax(r, 1)
    if loss == 'poisson':
        return torch.exp(r.clamp(max=30.0))
    return r


def aux_predict(layout, raw):
    """Raw auxiliary outputs [B, >= W] (`forward(..., aux=True)`'s fourth: the head's W columns first)
    -> {name: predictions [B, w]} by each target's loss: 'multinomial' the softmax over its segment (a
    distribution: MTG Zero's belief, x the hand size = expected counts), 'poisson' exp (rates =
    expected counts), 'mse' raw."""
    out, k = {}, 0
    for name, w, loss in aux_layout(layout):
        out[name] = _predict(loss, raw[:, k:k + w].float())
        k += w
    return out


# ---------------------------------------------------------------- readers (docs/az.md 36)
READER_ENTER = ('trunk', 'none')


def reader_layout(readers):
    """Readers ([{target, width, loss, inputs, hidden, enter, belief}, ...]: a config's `readers`, or the
    game package's declarations completed with their targets' width and loss, `GamePackage.readers_of`;
    None = none) -> their normal form (stored in the checkpoint config): [{'target', 'width', 'loss',
    'inputs': [[offset, count], ...], 'hidden', 'enter', 'belief'}, ...], checked (targets distinct, a
    loss of AUX_LOSSES, inputs non-empty and without a feature twice, enter of READER_ENTER)."""
    out = []
    for r in readers or []:
        s = {'target': str(r['target']), 'width': int(r['width']), 'loss': str(r['loss']),
             'inputs': [[int(o), int(k)] for o, k in r['inputs']], 'hidden': int(r.get('hidden', 64)),
             'enter': str(r.get('enter', 'trunk')), 'belief': bool(r.get('belief', False))}
        idx = [o + i for o, k in s['inputs'] for i in range(k)]
        bad = (s['width'] < 1 or s['loss'] not in AUX_LOSSES or s['hidden'] < 1 or s['enter'] not in READER_ENTER
               or not idx or min(idx) < 0 or any(k < 1 for _, k in s['inputs']) or len(set(idx)) != len(idx)
               or any(o['target'] == s['target'] for o in out))
        if bad:
            raise ValueError(f'reader {r!r}: a target of its own, width >= 1, a loss of {AUX_LOSSES}, inputs [[offset, '
                             f'count >= 1], ...] reading each feature once, hidden >= 1, enter of {READER_ENTER}')
        out.append(s)
    return out


def reader_index(spec):
    """The feature indexes a reader reads, in its input order."""
    return [o + i for o, k in spec['inputs'] for i in range(k)]


class Reader(nn.Module):
    """One reader (docs/az.md 36): an MLP from the features it declares (`spec['inputs']`) to its
    target's raw outputs, its last layer zero (it predicts what the target's loss predicts from zeros:
    uniform for 'multinomial'); with enter 'trunk' its prediction (`_predict`) goes into the trunk's
    input point through `proj` (width -> the trunk's width there), zero: a network that gains a reader
    computes what it computed before."""

    def __init__(self, spec, features, d_enter):
        super().__init__()
        idx = reader_index(spec)
        if max(idx) >= features:
            raise ValueError(f'reader {spec["target"]}: inputs up to feature {max(idx)}, the network has {features}')
        self.spec = spec
        self.register_buffer('idx', torch.tensor(idx, dtype=torch.long), persistent=False)
        h = spec['hidden']
        self.mlp = nn.Sequential(nn.Linear(len(idx), h), nn.ReLU(), nn.Linear(h, spec['width']))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
        self.proj = None
        if spec['enter'] == 'trunk':
            self.proj = nn.Linear(spec['width'], d_enter)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)


def _with_readers(model, features, d_enter, readers):
    """Give `model` its readers (config 'readers') when `readers` names any (built after every other
    parameter, the auxiliary head included). `model.reader_detach` (not saved: training only, run
    setting reader_detach): the prediction enters the trunk detached, so only its target's auxiliary
    loss trains the reader (docs/az.md 36)."""
    model.reader_specs = reader_layout(readers)
    model.reader_detach = True
    model.readers = None
    if model.reader_specs:
        model.config['readers'] = model.reader_specs
        model.readers = nn.ModuleList(Reader(s, features, d_enter) for s in model.reader_specs)


def _read(model, x):
    """The readers on rows x [B, features] -> (their raw outputs [B, sum of widths] in reader order,
    what they add to the trunk's input point [B, d_enter] or None)."""
    raws, entry = [], None
    for r in model.readers:
        raw = r.mlp(x.index_select(1, r.idx))
        raws.append(raw)
        if r.proj is not None:
            p = _predict(r.spec['loss'], raw.float())
            e = r.proj(p.detach() if model.reader_detach else p)
            entry = e if entry is None else entry + e
    return torch.cat(raws, 1) if len(raws) > 1 else raws[0], entry


def reader_predict(model, raw):
    """{target: predictions [B, w]} of `model`'s readers from `forward(..., aux=True)`'s fourth output (the
    readers' columns follow the head's W)."""
    k, out = sum(w for _, w, _ in (getattr(model, 'aux_layout', None) or [])), {}
    for s in getattr(model, 'reader_specs', None) or []:
        out[s['target']] = _predict(s['loss'], raw[:, k:k + s['width']].float())
        k += s['width']
    return out


def aux_answer(model, raw):
    """{name: predictions [B, w]} of `model`'s auxiliary outputs `raw` (`forward(..., aux=True)`'s fourth) in
    its head's layout - what the search's root belief rows and `NetEvaluator.aux` answer: the head's
    (`aux_predict`), except that a target whose reader is declared `belief` takes the reader's (docs/az.md
    36). A network without readers: `aux_predict` of its head."""
    out = aux_predict(model.aux_layout, raw)
    for s in getattr(model, 'reader_specs', None) or []:
        if s['belief'] and s['target'] in out:
            out[s['target']] = reader_predict(model, raw)[s['target']]
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
    def __init__(self, features, actions, hidden=256, blocks=4, seats=MAX_SEATS, game='toy', head=128, aux=None,
                 readers=None, aux_parts=None):
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
        _with_aux(self, hidden, head, aux, aux_parts)
        # readers (docs/az.md 36): their predictions are added to the first layer's input
        _with_readers(self, features, hidden, readers)

    def forward(self, x, n_players, aux=False, aux_rows=None):
        """x [B, >= features] (a narrower network reads a prefix: the evaluator refuses that
        unless told otherwise, docs/az.md 10.4), n_players [B] -> policy logits [B, actions],
        value logits [B, seats] (absent seats -1e9), margin [B, seats] (+ with `aux` the raw
        auxiliary outputs [B, aux width (+ the readers' widths)]; `aux_rows`, an index tensor: of
        those rows only)."""
        x = x[:, :self.features]
        h, rraw = self.inp(x), None
        if self.readers is not None:
            rraw, e = _read(self, x)
            if e is not None:
                h = h + e
        h = F.relu(h)
        for b in self.blocks:
            h = b(h)
        h = F.relu(self.ln(h))
        present = torch.arange(self.seats, device=x.device)[None, :] < n_players[:, None]
        vlog = self.val(h).masked_fill(~present, -1e9)
        if aux:
            return self.pol(h), vlog, self.mar(h), _aux_out(self, h, aux_rows, rraw)
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
                 arch='grid', aux=None, readers=None, aux_parts=None):
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
        _with_aux(self, 256, head, aux, aux_parts)
        # readers (docs/az.md 36): their predictions are added to the globals' embedding (which biases
        # the stem, every block and the pooled trunk)
        _with_readers(self, features, ghid, readers)

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
        rraw = None
        if self.readers is not None:
            rraw, ent = _read(self, x)
            if ent is not None:
                g = g + ent
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
            return pol, vlog, self.mar(e), _aux_out(self, e, aux_rows, rraw)
        return pol, vlog, self.mar(e)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def tokens_layout(layout, features, actions):
    """The normalised object layout of a `TokenNet` (validated against the feature / action
    widths; module docstring of TokenNet). -> a plain dict (stored in the checkpoint config).
    The keys of variable-length token sets (docs/az.md 43.5: segment `tail`, `gate`, `columns`, a
    tail-only segment's `count` 0 / `offset` null; layout `option_pointers`, `max_row_tokens`,
    `draft`) are kept only when set, so every layout without them normalises as before."""
    vocab = int(layout['vocab'])
    segs, bags, used, acts, ids = [], [], set(), {}, []
    pointers = bool(layout.get('option_pointers'))

    def claim(name, lo, n):
        if lo < 0 or lo + n > features:
            raise ValueError(f'tokens layout: {name} [{lo}, {lo + n}) does not fit {features} features')
        r = set(range(lo, lo + n))
        if r & used:
            raise ValueError(f'tokens layout: {name} overlaps another segment')
        used.update(r)

    for s in layout.get('segments', []):
        tail = bool(s.get('tail'))
        seg = {'name': str(s['name']), 'offset': None if s.get('offset') is None else int(s['offset']),
               'count': int(s['count']), 'width': int(s['width']),
               'present': None if s.get('present') is None else int(s['present']), 'id': None,
               'actions': [[int(b), int(k)] for b, k in s.get('actions', [])]}
        if (seg['count'] < 1 and not (tail and seg['count'] == 0)) or seg['width'] < 1:
            raise ValueError(f'tokens layout: segment {seg["name"]} has no slots')
        if (seg['count'] == 0) != (seg['offset'] is None):
            raise ValueError(f'tokens layout: segment {seg["name"]}: offset null exactly for a tail-only segment '
                             f'(count 0, tail), which claims no features')
        if seg['present'] is not None and not 0 <= seg['present'] < seg['width']:
            raise ValueError(f'tokens layout: segment {seg["name"]}: present flag outside the slot')
        if s.get('id') is not None:
            radix = [int(r) for r in s['id']['radix']]
            at = int(s['id']['at'])
            if at < 0 or at + sum(radix) > seg['width'] or math.prod(radix) > vocab:
                raise ValueError(f'tokens layout: segment {seg["name"]}: identity {s["id"]} does not fit')
            seg['id'] = {'at': at, 'radix': radix}
        if seg['count']:
            claim(seg['name'], seg['offset'], seg['count'] * seg['width'])
        # (a segment's static actions are its head slots': a tail token never gets one, so the
        # collision check covers the head slots only)
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
        # variable-length token sets (docs/az.md 43.5; each key only when set)
        if tail:
            seg['tail'] = True
        if s.get('gate'):
            if seg['count'] != 0:
                raise ValueError(f'tokens layout: segment {seg["name"]}: gate only on a tail-only segment (count 0)')
            seg['gate'] = True
        if s.get('columns') is not None:
            c = int(s['columns'])
            if c < 1 or not pointers:
                raise ValueError(f'tokens layout: segment {seg["name"]}: columns {c} needs columns >= 1 and the layout\'s '
                                 f'option_pointers')
            seg['columns'] = c
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
    # variable-length token sets (docs/az.md 43.5; each key only when set): rows carry pointer
    # overrides; a row's capacity 1 + sum(count) + its tail tokens may not pass max_row_tokens (required
    # with a tail segment); a draft layout (run launches and fork_run refuse it unless allowed)
    if pointers:
        out['option_pointers'] = True
    head = 1 + sum(s['count'] for s in segs)
    if any(s.get('tail') for s in segs) and layout.get('max_row_tokens') is None:
        raise ValueError('tokens layout: a layout with tail segments needs max_row_tokens')
    if layout.get('max_row_tokens') is not None:
        mrt = int(layout['max_row_tokens'])
        if mrt < head:
            raise ValueError(f'tokens layout: max_row_tokens {mrt} below the head\'s {head} token positions')
        out['max_row_tokens'] = mrt
    if layout.get('draft'):
        out['draft'] = True
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
        """h [B, T, d], keep [B, 1, 1, T] bool (False = padding: never attended to), or a float
        additive mask (gated tokens, docs/az.md 43.6: log of the keys' weights, -inf at 0)."""
        B, T, D = h.shape
        q, k, v = self.qkv(self.ln1(h)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
        h = h + self.out(a.transpose(1, 2).reshape(B, T, D))
        return h + self.ff2(F.gelu(self.ff1(self.ln2(h))))

    def forward_weighted(self, h, w4, pad3):
        """`forward` with weighted keys, as an explicit math attention (docs/az.md 43.6: training while a present gate
        is exactly 0, where SDPA's float mask log w would give the gate no finite gradient): p_ij = exp(s_ij - m_i) *
        w_j, out_i = sum_j p_ij v_j / sum_j p_ij, m_i the max over the non-padding keys; w4 [B, 1, 1, T] the keys'
        weights (1 ungated, g gated, 0 padding), pad3 [B heads, 1, T] the padding keys (`_trunk_ext` builds it once
        for every block). One autograd node (`_WeightedAttention`)."""
        B, T, D = h.shape
        # (q, k, v contiguous by one copy of the three: the batched products' [B H, T, dh] views without a copy each)
        q, k, v = self.qkv(self.ln1(h)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4).contiguous()
        a = _WeightedAttention.apply(q, k, v, w4, pad3)
        h = h + self.out(a.transpose(1, 2).reshape(B, T, D))
        return h + self.ff2(F.gelu(self.ff1(self.ln2(h))))


class _WeightedAttention(torch.autograd.Function):
    """The explicit weighted attention (docs/az.md 43.6: training while a present gate is exactly 0) as one autograd
    node: the forward is the explicit math of before, op for op (s = q k^T / sqrt(dh), padding -inf, e = exp(s - m)
    with m the max over the non-padding keys, p = e w, out = p v / sum_j p); the backward its gradients by hand: with
    a = p / Z, A = dO v^T, D_i = dO_i . out_i, R = (A - D) / Z - dv = a^T dO, dw_j = sum_{h,i} e_ij R_ij, ds = p R,
    dq = ds k / sqrt(dh), dk = ds^T q / sqrt(dh) (~16 operations a layer, where autograd's chain of the explicit
    math ran ~25 nodes: docs/az.md 43.9)."""

    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, q, k, v, w4, pad3):
        B, H, T, dh = q.shape
        scale = 1.0 / math.sqrt(dh)
        # (on [B H, T, .]: the batched products torch.matmul runs for these 4-d operands, one call each)
        q3, k3, v3 = (t.reshape(B * H, T, dh) for t in (q, k, v))
        w3 = w4.expand(B, H, 1, T).reshape(B * H, 1, T)
        s = torch.bmm(q3, k3.transpose(1, 2)) * scale
        s = s.masked_fill(pad3, float('-inf'))
        e = torch.exp(s - s.amax(-1, keepdim=True))
        p = e * w3
        Z = p.sum(-1, keepdim=True)
        out = torch.bmm(p, v3) / Z
        ctx.save_for_backward(q3, k3, v3, e, p, Z, out)
        ctx.dims = (B, H, T, dh, scale)
        return out.view(B, H, T, dh)

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, dO):
        q3, k3, v3, e, p, Z, out = ctx.saved_tensors
        B, H, T, dh, scale = ctx.dims
        dO = dO.reshape(B * H, T, dh)
        R = (torch.bmm(dO, v3.transpose(1, 2)) - (dO * out).sum(-1, keepdim=True)) / Z
        dv = torch.bmm((p / Z).transpose(1, 2), dO)
        dw = (e * R).sum(1).view(B, H, T).sum(1)[:, None, None, :] if ctx.needs_input_grad[3] else None
        ds = (p * R) * scale
        dq, dk = torch.bmm(ds, k3), torch.bmm(ds.transpose(1, 2), q3)
        return dq.view(B, H, T, dh), dk.view(B, H, T, dh), dv.view(B, H, T, dh), dw, None


class _TailEmbed(torch.autograd.Function):
    """`TokenNet._tail_embed_static` as one autograd node: the forward runs the ops of before, unrecorded; the
    backward computes the gradients autograd computed for them (gather -> scatter_add, addmm / mm -> the products
    in `mm_mat2_backward`'s operand order, index_copy -> index_select, cat -> narrow, index -> index_put with
    accumulate, embedding -> embedding_dense_backward), in one node instead of ~25 (a training step's ~200
    operations fewer: docs/az.md 43.9).

    Inputs: val [NT, Wt], psi [NT, PSI], tk [NT] (each token's tail index), ids / sel (per identity layout: the
    tokens' identities, whether their segment uses it), w_index (the input maps' places), Wt, tail_base, the
    identity embedding's weight, then the K tail segments' input-map weights, their biases, their position maps."""

    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, val, psi, tk, ids, sel, w_index, Wt, tail_base, emb_w, *params):
        K = len(params) // 3
        d = tail_base.shape[1]
        Ws, bs, Ps = params[:K], params[K:2 * K], params[2 * K:]
        NT = val.shape[0]
        W = torch.cat([w.reshape(-1) for w in Ws])
        W = W.new_zeros(K * d * Wt).index_copy(0, w_index, W).view(K * d, Wt)
        b = torch.cat(bs)
        pick = tk[:, None, None].expand(NT, 1, d)
        x = F.linear(val, W, b).view(NT, K, d).gather(1, pick).squeeze(1)
        P = torch.cat(Ps)
        pos = F.linear(psi, P).view(NT, K, d).gather(1, pick).squeeze(1)
        t = x + (tail_base[tk] + pos)
        for i, u in zip(ids, sel):
            e = F.embedding(i, emb_w)
            t = t + torch.where(u[:, None], e, torch.zeros((), dtype=e.dtype, device=e.device))
        ctx.save_for_backward(val, psi, tk, w_index, *ids, *sel)
        ctx.meta = (K, d, Wt, len(ids), [tuple(w.shape) for w in Ws], emb_w.shape[0])
        return t

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, G):
        K, d, Wt, nid, wshapes, nemb = ctx.meta
        val, psi, tk, w_index, *rest = ctx.saved_tensors
        ids, sel = rest[:nid], rest[nid:]
        need = ctx.needs_input_grad
        NT = val.shape[0]
        pick = tk[:, None, None].expand(NT, 1, d)
        # t = x + (tail_base[tk] + pos) + sum where(sel, emb(ids), 0): each term's gradient is G
        # x = Y.view(NT, K, d).gather(1, pick).squeeze(1), Y = addmm(b, val, W.t()): the gather's backward scatters G
        # to [NT, K, d] (pos's gather gets the same)
        GY = torch.zeros(NT, K, d, dtype=G.dtype, device=G.device).scatter_add_(1, pick, G[:, None]).view(NT, K * d)
        gW = GY.t().mm(val)                                  # [K d, Wt] (mm_mat2_backward of W.t(), transposed back)
        gb = GY.sum(0).view(K, d)
        gP = GY.t().mm(psi).view(K, d, -1)                   # [K, d, PSI]
        # tail_base[tk]'s gradient is the biases' (each sums G over its segment's tokens), the embedding's one index_add
        g_tb = gb.clone() if need[7] else None
        g_emb = None
        if need[8] and nid:
            g_emb = torch.zeros(nemb, d, dtype=G.dtype, device=G.device)
            for i, u in zip(ids, sel):
                g_emb.index_add_(0, i, torch.where(u[:, None], G, 0))
        grads_W = [g.view(shp) for g, shp in zip(gW.reshape(-1).index_select(0, w_index).split(
            [a * b for a, b in wshapes]), wshapes)]
        return (None, None, None, None, None, None, None, g_tb, g_emb, *grads_W, *gb.unbind(0), *gP.unbind(0))


# tail token positions (docs/az.md 43.5): psi(n), n = slot - count in [0, 2^32): its 32 binary digits,
# log2(n + 1) / 32, and sin / cos(2 pi (n mod 2^j) / 2^j) for j = 1..8 (interleaved: sin_1, cos_1, sin_2, ...)
PSI = 49


def _capturing(t):
    """A CUDA graph is being captured on the current stream (`t`: a tensor of the forward's device): the
    checks that need a host sync are skipped then (the host checked the graph's inputs)."""
    return t.is_cuda and torch.cuda.is_current_stream_capturing()


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

    Variable-length token sets (docs/az.md 43.5-43.8; all opt-in, a layout without them builds the
    network of before): a segment with `tail` takes its slots >= `count` as tail tokens (`forward*(...,
    tail=)`: (row, segment, slot, floats)), embedded as head slots (the segment's input map, the
    identity) with the position `tail_base[k] + pos_tail[k](psi(slot - count))`; a tail-only segment
    (`count` 0) has tail tokens only; a `gate`d segment's tokens weigh g = max(theta, 0) in the
    attention's keys and the pooling (exactly 0 at a widen: they change nothing else); `columns` gives a
    segment pointer columns without a static table; with `option_pointers` the rows' pointer overrides
    (`ov`, resolved on the host) score their options from a token's column (or the generic logit)
    after the static scatter.
    """

    def __init__(self, features, actions, layout, d=128, layers=3, heads=4, ff=256, seats=MAX_SEATS, game='toy',
                 head=128, arch='tokens', aux=None, readers=None, aux_parts=None):
        super().__init__()
        lay = tokens_layout(layout, features, actions)
        self.config = {'arch': 'tokens', 'features': features, 'actions': actions, 'layout': lay, 'd': d,
                       'layers': layers, 'heads': heads, 'ff': ff, 'seats': seats, 'game': game, 'head': head}
        if d % heads:
            raise ValueError(f'tokens: d {d} is not a multiple of heads {heads}')
        self.features, self.seats, self.actions, self.d = features, seats, actions, d
        self.segs, self.bags, self.vocab = lay['segments'], lay['bags'], lay['vocab']
        self.chunk = 64                      # rows per packed chunk (an execution detail, not saved)
        # variable-length token sets (docs/az.md 43.5): the tail segments and the gated ones (segment
        # indexes, layout order), pointer overrides; none of them: the network of before
        self.tails = [j for j, s in enumerate(self.segs) if s.get('tail')]
        self.gated = [j for j, s in enumerate(self.segs) if s.get('gate')]
        self.option_pointers = bool(lay.get('option_pointers'))
        inside = set()
        for s in self.segs:
            if not s['count']:               # (a tail-only segment claims no features)
                continue
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
        # (a segment's `columns`: pointer columns after its action pairs and identity column, with no
        # static table: scored only through pointer overrides, docs/az.md 43.7)
        nk = [len(s['actions']) + ('id_actions' in s) + s.get('columns', 0) for s in self.segs]
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
        if self.tails:
            # one more position, n_tok: every tail token's (no static action: a tail token is scored
            # only through pointer overrides, docs/az.md 43.7)
            cols = torch.cat([cols, torch.full((1, K), -1, dtype=torch.long)])
            acts = torch.cat([acts, torch.full((1, K), actions, dtype=torch.long)])
            isid = torch.cat([isid, torch.zeros((1, K), dtype=torch.bool)])
            idbase = torch.cat([idbase, torch.zeros(1, dtype=torch.long)])
        self.register_buffer('tok_cols', cols, persistent=False)
        self.register_buffer('tok_acts', acts, persistent=False)
        if self.n_id:
            self.register_buffer('tok_isid', isid, persistent=False)
            self.register_buffer('tok_idbase', idbase, persistent=False)
        if self.tails or self.option_pointers:
            self._tail_buffers(nk)
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
        if self.tails:
            # tail positions (docs/az.md 43.5): per tail segment k a base (a segment with head slots: its
            # last head slot's embedding, so its first tail slot starts where the head ends; tail-only:
            # N(0, 0.02)) and a map of psi(slot - count), N(0, 0.02)
            tb = torch.empty(len(self.tails), d)
            for k, j in enumerate(self.tails):
                if self.segs[j]['count']:
                    tb[k] = self.pos.data[self.seg_start[j] + self.segs[j]['count'] - 1]
                else:
                    nn.init.normal_(tb[k], std=0.02)
            self.tail_base = nn.Parameter(tb)
            self.pos_tail = nn.ModuleList(nn.Linear(PSI, d, bias=False) for _ in self.tails)
            for lin in self.pos_tail:
                nn.init.normal_(lin.weight, std=0.02)
        if self.gated:
            # the gated segments' gates theta (g = max(theta, 0)): 1 when built, 0 when a widen adds the
            # segment (docs/az.md 43.6)
            self.gate = nn.Parameter(torch.ones(len(self.gated)))
        _with_aux(self, 2 * d, head, aux, aux_parts)
        # readers (docs/az.md 36): they read the flat part of the features - the bags and the globals,
        # every feature outside the segments (what the CUDA graphs' host copy always copies) - and their
        # predictions are added to the global token's input
        in_seg = set()
        for s in self.segs:
            if s['count']:
                in_seg.update(range(s['offset'], s['offset'] + s['count'] * s['width']))
        for s in reader_layout(readers):
            seg = sorted(set(reader_index(s)) & in_seg)
            if seg:
                raise ValueError(f'reader {s["target"]}: a token network\'s reader reads its bags and globals (the '
                                 f'features outside the segments), not feature {seg[0]} of a segment')
        _with_readers(self, features, d, readers)

    # ---- variable-length token sets (docs/az.md 43.5-43.7)
    def _tail_buffers(self, nk):
        """The per-segment tables of tails and pointer overrides (non-persistent buffers: the state is
        unchanged): a segment's pointer columns (first global index, number), head slot count, whether it
        takes tail tokens, its index among the tail segments and among the gated ones (-1: not one); each
        head position's segment (the global token: -1); psi's shifts and moduli."""
        start, c0, p, c = [], [], 1, 0
        for s, k in zip(self.segs, nk):
            start.append(p)
            c0.append(c)
            p += s['count']
            c += k
        self.seg_start = start                      # a segment's first position in the full token axis
        tk, gi = [-1] * len(self.segs), [-1] * len(self.segs)
        for k, j in enumerate(self.tails):
            tk[j] = k
        for k, j in enumerate(self.gated):
            gi[j] = k

        def lt(v):
            return torch.tensor(v, dtype=torch.long)
        self.register_buffer('seg_c0', lt(c0), persistent=False)
        self.register_buffer('seg_nk', lt(nk), persistent=False)
        self.register_buffer('seg_count', lt([s['count'] for s in self.segs]), persistent=False)
        self.register_buffer('seg_istail', torch.tensor([bool(s.get('tail')) for s in self.segs]), persistent=False)
        # a tail token's lowest slot per segment (its head count; a segment without a tail: none, 2^62)
        self.register_buffer('seg_lo', lt([s['count'] if s.get('tail') else 1 << 62 for s in self.segs]), persistent=False)
        self.register_buffer('seg_tk', lt(tk), persistent=False)
        self.register_buffer('seg_gi', lt(gi), persistent=False)
        self.register_buffer('pos_seg', lt([-1] + [j for j, s in enumerate(self.segs) for _ in range(s['count'])]),
                             persistent=False)
        # the pointer head's parts (`_pointer`): each tail-only segment's columns apart, runs of the others
        parts, run = [], None
        for s, a, k in zip(self.segs, c0, nk):
            if not k:
                continue
            if s['count'] == 0:
                parts.append((a, a + k))
                run = None
            elif run is not None and parts[run][1] == a:
                parts[run] = (parts[run][0], a + k)
            else:
                parts.append((a, a + k))
                run = len(parts) - 1
        self.ptr_parts = parts or [(0, self.ptr.out_features)]
        self.register_buffer('psi_shift', torch.arange(32, dtype=torch.long), persistent=False)
        self.register_buffer('psi_mod', lt([2 ** j for j in range(1, 9)]), persistent=False)
        # a tail token's floats: the first width(segment) of t_width columns (the widest tail segment's)
        self.t_width = max([self.segs[j]['width'] for j in self.tails] or [0])

    def tail_segments(self):
        """The network's tail segments [(segment index, name, width, head slots), ...] in layout order (what a
        game's `game_info()['tails']` must be: `infer.NetEvaluator` refuses another, docs/az.md 43.8)."""
        return [(j, self.segs[j]['name'], self.segs[j]['width'], self.segs[j]['count']) for j in self.tails]

    def gates(self):
        """{gated segment name: g = max(theta, 0)} (docs/az.md 43.6: the run logs them; a gate that stays at 0
        is said loudly there); {} without gated segments."""
        if not self.gated:
            return {}
        g = self.gate.detach().clamp(min=0).cpu().tolist()
        return {self.segs[j]['name']: float(v) for j, v in zip(self.gated, g)}

    def _gate_g(self):
        """g = max(theta, 0) of the gated segments, with a straight-through gradient (backward the identity:
        a gate pushed below 0 can come back). Exact: theta + (relu(theta) - theta) is relu(theta)."""
        th = self.gate
        return th + (th.clamp(min=0) - th).detach()

    def _psi(self, n):
        """psi(n) [N, PSI] of int64 offsets n in [0, 2^32) (PSI's comment): the digits exact from the
        integers, the rest computed in float64 and rounded to float32 (the same on every device). The sin /
        cos part depends on n mod 256 only: read from a table of its 256 values (`_psi_sc`)."""
        bits = ((n[:, None] >> self.psi_shift) & 1).to(torch.float32)
        lg = (torch.log2(n.to(torch.float64) + 1.0) / 32.0).to(torch.float32)[:, None]
        return torch.cat([bits, lg, self._psi_sc(n.device)[n & 255]], 1)

    def _psi_sc(self, dev):
        """sin / cos(2 pi (m mod 2^j) / 2^j), j = 1..8, of m = 0..255 [256, 16] float32 (`_psi`), computed once per
        device by psi's own ops (float64 on that device, then rounded): psi's values, bit for bit."""
        cache = getattr(self, '_psitab', None)
        if cache is None or cache[0] != dev:
            m = torch.arange(256, dtype=torch.long, device=dev)
            mod = self.psi_mod.to(dev).to(torch.float64)
            ang = (m[:, None] & (self.psi_mod.to(dev) - 1)).to(torch.float64) * (2.0 * math.pi) / mod
            sc = torch.stack([torch.sin(ang), torch.cos(ang)], -1).flatten(1).to(torch.float32)
            cache = self._psitab = (dev, sc)
        return cache[1]

    def _tail_embed(self, tail, static):
        """Tail tokens' embeddings [NT, d], as a head slot's: the segment's input map of its floats + the
        position tail_base[k] + pos_tail[k](psi(slot - count)) + its identity's embedding (a segment with an
        identity). `static`: `_tail_embed_static`; else each present segment's tokens selected, its maps on
        them (forward: a segment's tokens embedded by the products a network without the other segments
        would run)."""
        seg, val = tail['seg'], tail['val']
        if static:
            return self._tail_embed_static(seg, tail['slot'], val)
        psi = self._psi(tail['slot'] - self.seg_count[seg])
        out = val.new_zeros(seg.shape[0], self.d)
        for k, j in enumerate(self.tails):
            s = self.segs[j]
            sel = (seg == j).nonzero(as_tuple=True)[0]
            if not len(sel):
                continue
            rows, ps = val[sel], psi[sel]
            t = self.seg_in[j](rows[:, :s['width']]) + (self.tail_base[k] + self.pos_tail[k](ps))
            if s['id'] is not None:
                t = t + self.emb(self._identity(rows, s['id']))
            out = out.index_put((sel,), t)
        return out

    def _tail_embed_static(self, seg, slot, val):
        """`_tail_embed` with no data-dependent shape and few kernels (forward_static, the CUDA graphs): every
        tail segment's input map on every token as one product (their weights in one matrix, each zero past
        its width: a token's floats are zeros there), every position map as one, each identity layout's digits
        decoded once; each token's own picked by its segment. A few dozen kernels whatever the number of tail
        segments (MTG Zero's v7: 13), where a loop over the segments ran ~45 each (forward) and as many again
        (backward)."""
        tab = self._tail_tables(val.device)
        ids = [self._identity_digits(val, spec) for spec, _ in tab['ids']]
        sel = [uses[seg] for _, uses in tab['ids']]
        return _TailEmbed.apply(val, self._psi(slot - self.seg_count[seg]), self.seg_tk[seg].clamp(min=0), ids, sel,
                                tab['w_index'],
                                self.t_width, self.tail_base, self.emb.weight,
                                *[self.seg_in[j].weight for j in self.tails], *[self.seg_in[j].bias for j in self.tails],
                                *[lin.weight for lin in self.pos_tail])

    def _identity_digits(self, sl, idspec):
        """`_identity` with every digit of the identity decoded at once (two products with the digit blocks'
        values and indicators, not a few operations per digit): the same identities on rows whose digit blocks
        are one-hot or empty (what an encoder writes; each product then sums one exact term per digit)."""
        tab = self._digit_tables(idspec, sl.device, sl.dtype)
        blk = sl[:, idspec['at']:idspec['at'] + tab[0].shape[0]]
        with torch.autocast(sl.device.type, enabled=False):      # (the products in the rows' float32 under amp)
            dig = (blk @ tab[0]).round().long()
            ok = ((blk @ tab[1]) > 0.5).all(1)
        idx = (dig * tab[2]).sum(1)
        return torch.where(ok & (idx < self.vocab), idx, torch.full_like(idx, self.vocab))

    def _digit_tables(self, idspec, dev, dtype):
        """(digit values [R, n_digits], digit indicators [R, n_digits], mixed-radix multipliers [n_digits]) of an
        identity layout, cached per device."""
        key = (idspec['at'], tuple(idspec['radix']), dev, dtype)
        cache = getattr(self, '_digtab', None)
        if cache is None:
            cache = self._digtab = {}
        if key not in cache:
            radix = list(idspec['radix'])
            R, nd = sum(radix), len(radix)
            vals, ind = torch.zeros(R, nd, dtype=dtype), torch.zeros(R, nd, dtype=dtype)
            mult, o = [], 0
            for k, r in enumerate(radix):
                vals[o:o + r, k] = torch.arange(r, dtype=dtype)
                ind[o:o + r, k] = 1
                o += r
                mult.append(math.prod(radix[k + 1:]))
            cache[key] = (vals.to(dev), ind.to(dev), torch.tensor(mult, dtype=torch.long, device=dev))
        return cache[key]

    def _tail_tables(self, dev):
        """`_tail_embed_static`'s tables, built once per device (an attribute: the state is unchanged): each
        tail segment's input-map weights' places in the [K * d, t_width] matrix, and per identity layout of the
        tail segments (`id`) the segments that use it."""
        cache = getattr(self, '_ttab', None)
        if cache is None or cache[0] != dev:
            d, Wt, idx, specs = self.d, self.t_width, [], {}
            for k, j in enumerate(self.tails):
                w = self.segs[j]['width']
                r = torch.arange(d)[:, None]
                idx.append(((k * d + r) * Wt + torch.arange(w)[None, :]).reshape(-1))
                if self.segs[j]['id'] is not None:
                    key = (self.segs[j]['id']['at'], tuple(self.segs[j]['id']['radix']))
                    specs.setdefault(key, []).append(j)
            ids = []
            for (at, radix), js in specs.items():
                uses = torch.zeros(len(self.segs), dtype=torch.bool)
                uses[js] = True
                ids.append(({'at': at, 'radix': list(radix)}, uses.to(dev)))
            cache = self._ttab = (dev, {'w_index': torch.cat(idx).to(dev), 'ids': ids})
        return cache[1]

    def _tail_weight(self, seg):
        """Tail tokens' weights [NT] in the attention's keys and the pooling: a gated segment's g, else 1."""
        gi = self.seg_gi[seg]
        g = self._gate_g()
        return torch.where(gi >= 0, g[gi.clamp(min=0)], torch.ones((), dtype=g.dtype, device=g.device))

    def _explicit(self, gi):
        """Whether the attention runs as the explicit math (docs/az.md 43.6): gradients on (training) and a
        gated segment with present tokens (`gi`: the present tail tokens' gated indexes, -1 = not gated) at
        g == 0. One host sync, training only."""
        f = self._explicit_flag(gi)
        return False if f is None else bool(f)

    def _explicit_flag(self, gi, clean=False):
        """`_explicit`'s decision as a 0-d bool tensor (no host sync; `gi` outside [0, G): not gated - `clean`: already
        in [0, G], G for not gated), None when gradients are off."""
        if not (torch.is_grad_enabled() and self.gate.requires_grad):
            return None
        G = len(self.gated)
        idx = gi if clean else torch.where((gi >= 0) & (gi < G), gi, G)
        pres = torch.zeros(G + 1, dtype=torch.bool, device=gi.device).index_fill(0, idx, True)
        return ((self.gate <= 0) & pres[:G]).any()

    def _tail_args(self, tail, ov):
        """The refusals of a forward's `tail` / `ov` (R5, docs/az.md 43.5) -> (NT, NO)."""
        if self.tails and tail is None:
            raise ValueError(f'TokenNet: a network with tail segments ({[self.segs[j]["name"] for j in self.tails]}) '
                             f'needs `tail` (possibly empty; docs/az.md 43.5)')
        if not self.tails and tail is not None:
            raise ValueError('TokenNet: `tail` given to a network without tail segments (docs/az.md 43.5)')
        if self.option_pointers and ov is None:
            raise ValueError('TokenNet: a network with option_pointers needs `ov` (possibly empty; docs/az.md 43.7)')
        if not self.option_pointers and ov is not None:
            raise ValueError('TokenNet: `ov` given to a network without option_pointers (docs/az.md 43.7)')
        return (0 if tail is None else int(tail['seg'].shape[0])), (0 if ov is None else int(ov['row'].shape[0]))

    def _check_tail(self, tail, B, eager, full=None, keep=None, defer=False):
        """A tail's shapes, and (unless a CUDA graph is being captured: the host checked it) that each token
        is a tail token of this network: a tail segment's, slot >= its head count and < 2^32 (R5); eager
        also its row in [0, B), rows ascending; with `full` / `keep` (forward_static) also every token
        position of `full` in [0, n_tok + NT) - one host sync for all. `defer`: no sync - a 0-d flag (or None)
        for the caller's one sync, `_raise_tail` after it."""
        seg, slot, row, val = tail['seg'], tail['slot'], tail['row'], tail['val']
        NT = seg.shape[0]
        if (tuple(row.shape) != (NT,) or tuple(slot.shape) != (NT,) or val.dim() != 2 or val.shape[0] != NT
                or val.shape[1] != self.t_width):
            raise ValueError(f'TokenNet: a tail of {NT} tokens needs row / seg / slot [{NT}] and val [{NT}, '
                             f'{self.t_width}] (got {tuple(row.shape)}, {tuple(slot.shape)}, {tuple(val.shape)})')
        if not NT or _capturing(seg):
            return
        sc = seg.clamp(0, len(self.segs) - 1)
        bad = (sc != seg) | (slot < self.seg_lo[sc]) | (slot >= 1 << 32)
        if eager:
            bad = bad | (row < 0) | (row >= B)
            if NT > 1:
                bad = bad | torch.cat([bad.new_zeros(1), row[1:] < row[:-1]])
        fbad = None if full is None else keep & (full.clamp(0, self.n_tok + NT - 1) != full)
        flag = bad.any() if fbad is None else (bad | fbad.any()).any()
        if defer:
            return flag
        if bool(flag):
            if fbad is not None and bool(fbad.any()):
                raise ValueError(f'forward_static: a token position of `full` outside [0, {self.n_tok + NT}) '
                                 f'(docs/az.md 43.5)')
            i = int(bad.nonzero()[0, 0])
            raise ValueError(f'TokenNet: tail token {i} (row {int(row[i])}, segment {int(seg[i])}, slot {int(slot[i])}) '
                             f'is not a tail token of this network: its tail segments are {self.tail_segments()} (index, '
                             f'name, width, head slots), a slot at or past the head and below 2^32, rows ascending')

    def _ov_index(self, ov, full, keep, tail, inv_head, inv_tail, defer=False):
        """The overrides of packed rows (docs/az.md 43.7): each one's target found by its extended position
        (`inv_head` [B, >= n_tok]: a head position's packed index; `inv_tail` [>= NT]: a tail token's) ->
        (row, action, packed index, global pointer column, gated target (None: no gates), its blend weight
        min(g, 1), generic (ref -1)). Checked (unless a graph is being captured): a target is a present token
        of its row with that column, ref -1 or a token, the action in [0, actions]."""
        r, a, ref, col = ov['row'], ov['act'], ov['ref'], ov['col']
        B, T = full.shape
        n_tok = self.n_tok
        NT = 0 if tail is None else tail['seg'].shape[0]
        gen = ref < 0
        hd = ref < n_tok
        rc = r.clamp(0, B - 1)
        hpos = ref.clamp(0, n_tok - 1)
        p = inv_head[rc, hpos]
        seg = self.pos_seg[hpos]
        if NT:
            tj = (ref - n_tok).clamp(0, NT - 1)
            p = torch.where(hd, p, inv_tail[tj])
            seg = torch.where(hd, seg, tail['seg'][tj])
        p = torch.where(gen, torch.zeros_like(p), p).clamp(0, T - 1)
        segc = seg.clamp(min=0)
        gc = (self.seg_c0[segc] + col).clamp(0, self.ptr.out_features - 1)
        flag = None
        if not _capturing(ref) and r.shape[0]:
            bad = (ref < -1) | (ref == 0) | (ref >= n_tok + NT) | (r < 0) | (r >= B) | (a < 0) | (a > self.actions)
            bad = bad | (~gen & ((full[rc, p] != ref) | ~keep[rc, p] | (seg < 0) | (col < 0)
                                 | (col >= self.seg_nk[segc])))
            flag = bad.any()
            if not defer and bool(flag):
                i = int(bad.nonzero()[0, 0])
                raise ValueError(f'TokenNet: pointer override {i} (row {int(r[i])}, action {int(a[i])}, ref {int(ref[i])}, '
                                 f'column {int(col[i])}): its target is not a present token of its row with that column '
                                 f'(ref -1 = the generic logit; docs/az.md 43.7)')
        isg = bw = None
        if self.gated:
            gi = self.seg_gi[segc]
            isg = ~gen & (gi >= 0)
            bw = self._gate_g()[gi.clamp(min=0)].clamp(max=1.0)
        if defer:
            return (r, a, p, gc, isg, bw, gen), flag
        return r, a, p, gc, isg, bw, gen

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

    def _global(self, x, entry=None):
        """The global token's input [B, d] (before its position embedding): the globals' map plus
        the bags' (each bag pooled as the count-weighted sum of the identity embeddings and its
        total), plus the readers' `entry` (`_read`, docs/az.md 36; None: none)."""
        B = x.shape[0]
        g = self.g(x.index_select(1, self.gidx))
        if self.bag is not None:
            pools = []
            for b in self.bags:
                o, k, w = b['offset'], b['count'], b['width']
                c = x[:, o:o + k * w].reshape(B, k, w)
                pools.append(torch.cat([c @ self.emb.weight[:w], c.sum(-1, keepdim=True)], -1))
            g = g + self.bag(torch.cat(pools, 1).flatten(1))
        if entry is not None:
            g = g + entry
        return g

    def pack(self, x, ident=False, entry=None):
        """The packed tokens: (h [B, T, d] (row b's n[b] tokens first, in layout order; the
        global token at 0), n [B], full [B, T] = each packed token's position in the full token
        axis (0 = global, then the segments' slots)); with `ident`, also each packed token's
        identity [B, T] (vocab: none). `entry`: the readers' (`_global`)."""
        r = self._pack(x, ident, entry)
        return r[:4] if ident else r[:3]

    def _pack(self, x, ident=False, entry=None, tail=None):
        """`pack` -> (h, n, full, idt (None without `ident`), cpos [B, n_tok] (a present head position's packed
        index), tpos [NT] (a tail token's) or None). With `tail` (docs/az.md 43.5) each row's tail tokens
        follow its head tokens (`n` counts them), their positions in `full` the extended ones (n_tok + their
        index in the tail), their identities none (they score no static action)."""
        B, dev = x.shape[0], x.device
        segs, pres = [], [torch.ones(B, 1, dtype=torch.bool, device=dev)]
        heads = [(s, lin) for s, lin in zip(self.segs, self.seg_in) if s['count']]   # (tail-only: no head)
        for s, _ in heads:
            o, k, w = s['offset'], s['count'], s['width']
            sl = x[:, o:o + k * w].reshape(B, k, w)
            segs.append(sl)
            pres.append(sl[..., s['present']] > 0.5 if s['present'] is not None
                        else torch.ones(B, k, dtype=torch.bool, device=dev))
        pres = torch.cat(pres, 1)                                     # [B, n_tok]
        n = pres.sum(1)
        NT = 0 if tail is None else tail['seg'].shape[0]
        tpos = None
        if NT:
            trow = tail['row']
            tcnt = torch.bincount(trow, minlength=B)
            tpos = n[trow] + torch.arange(NT, device=dev) - (tcnt.cumsum(0) - tcnt)[trow]
            n = n + tcnt
        T = int(n.max())
        cpos = pres.long().cumsum(1) - 1                             # packed index of a present token
        full = torch.zeros(B, T, dtype=torch.long, device=dev)
        g = self._global(x, entry)
        h = torch.cat([(g + self.pos[0])[:, None], g.new_zeros(B, T - 1, self.d)], 1)
        idt = torch.full((B, T), self.vocab, dtype=torch.long, device=dev) if ident else None
        start = 1
        for (s, lin), sl in zip(heads, segs):
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
        if NT:
            h = h.index_put((trow, tpos), self._tail_embed(tail, static=False))
            full[trow, tpos] = self.n_tok + torch.arange(NT, device=dev)
        return h, n, full, idt, cpos, tpos

    def _trunk(self, hc, nr, fc, ic=None, ext=None):
        """The blocks and the policy logits of packed rows: hc [Bc, Tc, d] (row b's nr[b] tokens
        first, then padding), fc [Bc, Tc] the tokens' positions in the full token axis, ic [Bc, Tc]
        their identities (a layout with id_actions) -> (zc [Bc, 2 d] the pooled state, pc [Bc,
        actions + 1] the policy logits; the last column a dummy for the padding's and the empty
        slots' scores). One chunk of `forward`, or a padded batch of `forward_static`. `ext`
        (rows with tail tokens or pointer overrides, docs/az.md 43.5-43.7): `_trunk_ext`'s keywords;
        None: the trunk of before, line for line."""
        if ext is not None:
            return self._trunk_ext(hc, nr, fc, ic, **ext)
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

    def _pointer(self, hid):
        """The pointer head's logits of every column ([..., columns]): `ptr`, its rows in parts (`ptr_parts`):
        each tail-only segment's columns a matrix product of their own, the columns between them one each - so
        a widen that appends segments (always tail-only) computes the old columns with the same products as
        before, bit for bit (docs/az.md 43.8). One part: `ptr` itself."""
        if len(self.ptr_parts) == 1 or torch.is_grad_enabled():
            # (training: one product - the parts exist for the widen's bit-exact evaluation, docs/az.md 43.8; their
            # row slices cost a training step a slice-backward each)
            return self.ptr(hid)
        W, b = self.ptr.weight, self.ptr.bias
        return torch.cat([F.linear(hid, W[a:e], b[a:e]) for a, e in self.ptr_parts], -1)

    def _trunk_ext(self, hc, nr, fc, ic=None, w=None, explicit=False, ov=None):
        """`_trunk` of rows with tail tokens or pointer overrides (docs/az.md 43.5-43.7). `w` [Bc, Tc]: the
        tokens' weights when gated tokens are present (1 the global and ungated tokens, g a gated one's, 0
        padding): the attention's keys and the pooling weighted - SDPA with the float mask log w (-inf at
        0), or the explicit math (`explicit`: training while a present gate is 0); None: today's bool mask
        and mean. A tail position (fc >= n_tok) has no static action. `ov` (`_ov_index`'s, rows local to
        these): after the static scatter each override's action gets its target token's logit in its
        column (a gated target's blended: gen + min(g, 1) (ptr - gen)) or, ref -1, the generic logit."""
        Tc = hc.shape[1]
        keep = torch.arange(Tc, device=hc.device)[None, :] < nr[:, None]
        if w is None:
            k4 = keep[:, None, None, :]
            for blk in self.blocks:
                hc = blk(hc, k4)
        elif explicit:
            # (`_WeightedAttention`'s padding mask, once for every block: the padding keys as [B heads, 1, T])
            H = self.blocks[0].h
            w4 = w[:, None, None, :]
            pad3 = ~keep[:, None, None, :].expand(-1, H, 1, -1).reshape(-1, 1, Tc)
            for blk in self.blocks:
                hc = blk.forward_weighted(hc, w4, pad3)
        else:
            m4 = torch.log(w)[:, None, None, :]
            for blk in self.blocks:
                hc = blk(hc, m4)
        hc = self.ln(hc) * keep[..., None]
        if w is None:
            zm = hc.sum(1) / nr[:, None].to(hc.dtype)
        else:
            zm = (hc * w[..., None]).sum(1) / w.sum(1, keepdim=True)
        zc = torch.cat([hc[:, 0], zm], 1)
        gen = F.pad(self.gen(zc), (0, 1))
        fcl = fc.clamp(max=self.n_tok) if self.tails else fc        # (n_tok: a tail token's, no action)
        cols = self.tok_cols[fcl]                                     # [Bc, Tc, K]
        acts = self.tok_acts[fcl]
        if self.n_id:                                                 # actions by identity
            ia = self.tok_idbase[fcl] + ic
            ia = torch.where((ic < self.vocab) & (ia < self.actions), ia, self.actions)
            acts = torch.where(self.tok_isid[fcl], ia[..., None], acts)
        acts = torch.where(keep[..., None] & (cols >= 0), acts, self.actions)
        P = self._pointer(F.gelu(self.pk(hc) + self.pq(zc)[:, None]))   # [Bc, Tc, columns]
        pc = gen.scatter(1, acts.flatten(1), P.gather(2, cols.clamp(min=0)).flatten(1))
        if ov is not None:
            r, a, p, gc, isg, bw, isgen = ov
            v, g0 = P[r, p, gc], gen[r, a]
            if isg is not None:
                # gen + min(g, 1) (ptr - gen) as torch.lerp, in the gates' float32 and cast back: exactly gen at 0 and
                # exactly ptr at 1 in every dtype (autocast's bfloat16 / float16 logits are exact in float32)
                v = torch.where(isg, torch.lerp(g0.float(), v.float(), bw.float()).to(v.dtype), v)
            pc = pc.index_put((r, a), torch.where(isgen, g0, v))
        return zc, pc

    def _heads(self, z, pol, n_players, aux, aux_rows, rraw=None):
        present = torch.arange(self.seats, device=z.device)[None, :] < n_players[:, None]
        vlog = self.val(z).masked_fill(~present, -1e9)
        if aux:
            return pol[:, :self.actions], vlog, self.mar(z), _aux_out(self, z, aux_rows, rraw)
        return pol[:, :self.actions], vlog, self.mar(z)

    def forward(self, x, n_players, aux=False, aux_rows=None, tail=None, ov=None):
        """As `Net.forward`. `tail` (required, possibly empty, iff the network has tail segments) = {'row':
        int64 [NT] ascending, 'seg': int64 [NT] (segment index), 'slot': int64 [NT], 'val': float32 [NT, Wt]};
        `ov` (required, possibly empty, iff option_pointers) = {'row', 'act' (the option's action), 'ref' (the
        target's extended position: n_tok + j a tail token; -1 the generic logit), 'col' (the segment's
        pointer column)}: int64 [NO] each, resolved on the host (docs/az.md 43.5-43.7). With no tail token
        and no override: the forward of before, line for line."""
        x = x[:, :self.features]
        NT, NO = self._tail_args(tail, ov)
        if NT or NO:
            return self._forward_ext(x, n_players, aux, aux_rows, tail, ov, NT, NO)
        rraw = ent = None
        if self.readers is not None:
            rraw, ent = _read(self, x)
        if self.n_id:
            h, n, full, idt = self.pack(x, ident=True, entry=ent)
        else:
            h, n, full = self.pack(x, entry=ent)
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
            z = z.index_copy(0, r, zc.to(z.dtype))
            pol = pol.index_copy(0, r, pc.to(pol.dtype))
        return self._heads(z, pol, n_players, aux, aux_rows, rraw)

    def _forward_ext(self, x, n_players, aux, aux_rows, tail, ov, NT, NO):
        """`forward` of a batch with tail tokens or pointer overrides (docs/az.md 43.5-43.7): each row's tail
        tokens packed after its head tokens, rows sorted by token count and chunked as `forward`; a chunk
        holding gated tokens runs the weighted trunk; the overrides are applied per chunk."""
        B, dev = x.shape[0], x.device
        if NT:
            self._check_tail(tail, B, eager=True)
        rraw = ent = None
        if self.readers is not None:
            rraw, ent = _read(self, x)
        h, n, full, idt, cpos, tpos = self._pack(x, ident=bool(self.n_id), entry=ent, tail=tail)
        T, D = h.shape[1], h.shape[2]
        tw = tg = None
        explicit = False
        if self.gated and NT:
            gi = self.seg_gi[tail['seg']]
            tw = (torch.arange(T, device=dev)[None, :] < n[:, None]).to(self.gate.dtype)
            tw = tw.index_put((tail['row'], tpos), self._tail_weight(tail['seg']))
            tg = torch.zeros(B, T, dtype=torch.bool, device=dev).index_put((tail['row'], tpos), gi >= 0)
            explicit = self._explicit(gi)
        ovx = opos = None
        rows = torch.argsort(n, stable=True) if B > self.chunk else torch.arange(B, device=dev)
        if NO:
            keep = torch.arange(T, device=dev)[None, :] < n[:, None]
            ovx = self._ov_index(ov, full, keep, tail, cpos, tpos)
            inv = torch.empty(B, dtype=torch.long, device=dev)
            inv[rows] = torch.arange(B, device=dev)
            opos = inv[ovx[0]]                                        # an override's row in the sorted order
        z = h.new_zeros(B, 2 * D)
        pol = h.new_zeros(B, self.actions + 1)
        for c0 in range(0, B, self.chunk):
            r = rows[c0:c0 + self.chunk]
            nr = n[r]
            Tc = int(nr.max())
            ext = {'w': None, 'explicit': explicit, 'ov': None}
            if tg is not None and bool(tg[r, :Tc].any()):
                ext['w'] = tw[r, :Tc]
            if ovx is not None:
                sel = ((opos >= c0) & (opos < c0 + len(r))).nonzero(as_tuple=True)[0]
                if len(sel):
                    ext['ov'] = (opos[sel] - c0,) + tuple(None if t is None else t[sel] for t in ovx[1:])
            zc, pc = self._trunk(h[r, :Tc], nr, full[r, :Tc], idt[r, :Tc] if self.n_id else None, ext)
            z = z.index_copy(0, r, zc.to(z.dtype))
            pol = pol.index_copy(0, r, pc.to(pol.dtype))
        return self._heads(z, pol, n_players, aux, aux_rows, rraw)

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
                if s['id'] is not None and s['count']:
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

    def forward_static(self, x, n_players, full, ntok, aux=False, tail=None, ov=None):
        """`forward` with shapes fixed by the inputs' (no host sync, no data-dependent shape: a CUDA
        graph can capture it): every slot is embedded, present or not, and each row's tokens are
        gathered by `full` [B, T] (its present tokens' positions in the full token axis, in layout
        order, the global token's 0 first, then padding: any position, masked out), `ntok` [B] its
        token count (>= 1). All rows are run as one chunk of T tokens. The outputs are `forward`'s
        (`aux`: the head on every row); a row's outputs depend only on it and on the batch's shape
        (the kernels' blocking: float32 rounding). `tail` / `ov` as `forward`'s (docs/az.md 43.5):
        `full` holds extended positions (a tail token at n_tok + its index in the tail, after the row's
        head tokens); padded tail entries (row 0, the first tail segment, slot its count, floats 0) and
        padded overrides (action `actions`, ref -1) are allowed; outside a graph capture the inputs are
        checked (one host sync) and, training with a present gate at 0, the explicit attention decided
        (another). With no tail token and no override: the forward of before, line for line."""
        x = x[:, :self.features]
        B = x.shape[0]
        NT, NO = self._tail_args(tail, ov)
        # (tails / overrides: their checks and the explicit attention's decision first, one host sync before this
        # forward queues its work - the device then waits only for what came before it: docs/az.md 43.9)
        prep = self._static_prep(B, full, ntok, tail, ov, NT, NO) if NT or NO else None
        rraw = ent = None
        if self.readers is not None:
            rraw, ent = _read(self, x)
        g = self._global(x, ent)
        toks = [(g + self.pos[0])[:, None]]
        ids = [torch.full((B, 1), self.vocab, dtype=torch.long, device=x.device)] if self.n_id else None
        ident = self._static_identities(x)
        start = 1
        for j, (s, lin) in enumerate(zip(self.segs, self.seg_in)):
            o, k, w = s['offset'], s['count'], s['width']
            if not k:                                # (a tail-only segment: no head slots)
                continue
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
        if NT or NO:
            return self._static_ext(x, n_players, full, ntok, aux, tail, NT, toks, ids, rraw, prep)
        h = torch.cat(toks, 1).gather(1, full[..., None].expand(-1, -1, self.d))
        idt = torch.cat(ids, 1).gather(1, full) if ids is not None else None
        z, pol = self._trunk(h, ntok, full, idt)
        return self._heads(z, pol, n_players, aux, None, rraw)

    def _static_prep(self, B, full, ntok, tail, ov, NT, NO):
        """`forward_static`'s inputs of a batch with tail tokens or overrides, before anything else (docs/az.md 43.5):
        the checks (R5) and the explicit attention's decision, as 0-d flags read by one host sync (none in a graph
        capture), the tokens' weights (gated segments), the overrides' targets by inverse maps of `full` -> a
        dict for `_static_ext`."""
        dev = full.device
        n_tok, T = self.n_tok, full.shape[1]
        keep = torch.arange(T, device=dev)[None, :] < ntok[:, None]
        tl = full >= n_tok
        tj = segc = tailc = None
        # the checks first, as 0-d flags read by one host sync (with the explicit attention's decision) before
        # anything indexes by the tail's segments; until then those indexes clamped (a bad tail is refused below)
        flags = {}
        if NT:
            f = self._check_tail(tail, B, eager=False, full=full, keep=keep, defer=True)
            if f is not None:
                flags['tail'] = f
            tj = (full - n_tok).clamp(0, NT - 1)
            segc = tail['seg'].clamp(0, len(self.segs) - 1)
            tailc = dict(tail, seg=segc)
        elif not _capturing(full) and bool((keep & ((full < 0) | tl)).any()):
            raise ValueError(f'forward_static: a token position of `full` outside [0, {n_tok}) and no tail tokens')
        w, explicit = None, False
        if self.gated and NT:
            # each token's weight: g of its gated segment, 1 for every other token (the gate vector extended by a
            # 1 at index G), 0 for padding
            G = len(self.gated)
            gi = self.seg_gi[segc][tj]
            gi = torch.where(keep & tl & (gi >= 0), gi, G)
            gext = torch.cat([self._gate_g(), torch.ones(1, dtype=self.gate.dtype, device=dev)])
            w = gext.gather(0, gi.reshape(-1)).view(B, T) * keep
            f = None if _capturing(full) else self._explicit_flag(gi.reshape(-1), clean=True)
            if f is not None:
                flags['explicit'] = f
        ovx = None
        if NO:
            pos = torch.arange(T, device=dev).expand(B, T)
            inv_h = torch.zeros(B, n_tok + 1, dtype=torch.long, device=dev).scatter_(
                1, torch.where(keep & ~tl, full, n_tok), pos)
            inv_t = None
            if NT:
                inv_t = torch.zeros(NT + 1, dtype=torch.long, device=dev).scatter_(
                    0, torch.where(keep & tl, tj, NT).reshape(-1), pos.reshape(-1))
            ovx, f = self._ov_index(ov, full, keep, tailc, inv_h, inv_t, defer=True)
            if f is not None:
                flags['ov'] = f
        pending = None
        if flags:
            # (read without draining the device: copied to pinned memory behind an event, waited for just before the
            # trunk (`_static_flags`) - by then the device is long past it, busy with this forward's embeddings)
            vec = torch.stack(list(flags.values()))
            if vec.is_cuda:
                host = torch.empty(vec.shape, dtype=vec.dtype, pin_memory=True)
                host.copy_(vec, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record()
                pending = (list(flags), host, ev)
            else:
                pending = (list(flags), vec, None)
        return {'keep': keep, 'tl': tl, 'tj': tj, 'w': w, 'explicit': explicit, 'ovx': ovx, 'pending': pending,
                'tailc': tailc, 'check': (tail, ov, B, full, keep, inv_h if NO else None, inv_t if NO else None)}

    def _static_flags(self, prep):
        """`_static_prep`'s flags read (its event waited for) -> the explicit attention's decision; a failed check
        raises here (R5: before the trunk, so before any output), with the details recomputed."""
        if prep['pending'] is None:
            return prep['explicit']
        names, host, ev = prep['pending']
        if ev is not None:
            ev.synchronize()
        got = dict(zip(names, host.tolist()))
        tail, ov, B, full, keep, inv_h, inv_t = prep['check']
        if got.get('tail'):
            self._check_tail(tail, B, eager=False, full=full, keep=keep)          # (raises, with the details)
        if got.get('ov'):
            self._ov_index(ov, full, keep, tail, inv_h, inv_t)
        return bool(got.get('explicit'))

    def _static_ext(self, x, n_players, full, ntok, aux, tail, NT, toks, ids, rraw, prep):
        """`forward_static` of a batch with tail tokens or overrides (docs/az.md 43.5; `_static_prep`'s inputs): the
        tokens gathered from cat(the head slots [B * n_tok, d], the tail tokens [NT, d]) by b * n_tok + p (a head
        position) or B * n_tok + j (tail token j); the weighted trunk whenever the network has gated segments
        (static: the weights say which tokens are gated)."""
        B, dev = x.shape[0], x.device
        n_tok, d, T = self.n_tok, self.d, full.shape[1]
        tl, tj = prep['tl'], prep['tj']
        # the tokens: gathered from cat(every head slot [B * n_tok, d], the tail tokens [NT, d])
        idx = torch.arange(B, device=dev)[:, None] * n_tok + full.clamp(0, n_tok - 1)
        src = torch.cat(toks, 1).reshape(B * n_tok, d)
        if NT:
            idx = torch.where(tl, B * n_tok + tj, idx)
            # (the tail's segments clamped: a bad tail is refused at `_static_flags`, before the trunk)
            src = torch.cat([src, self._tail_embed(prep['tailc'], static=True)])
        h = src.index_select(0, idx.reshape(-1)).view(B, T, d)
        idt = None
        if ids is not None:
            idt = torch.cat(ids, 1).gather(1, full.clamp(0, n_tok - 1))
            if NT:
                idt = torch.where(tl, torch.full_like(idt, self.vocab), idt)
        explicit = self._static_flags(prep)
        z, pol = self._trunk(h, ntok, full, idt, {'w': prep['w'], 'explicit': explicit, 'ov': prep['ovx']})
        return self._heads(z, pol, n_players, aux, None, rraw)

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
    itself; with another one: ValueError (a head for some of the targets: `widen_aux`; else strip_aux
    first)."""
    aux = aux_layout(aux)
    have = aux_layout(model.config.get('aux'))
    if have == aux:
        return model
    if have or not aux:
        raise ValueError(f'add_aux: the network has the auxiliary head {have}, wanted {aux} (a head for some of the '
                         f'targets: widen_aux; else strip_aux first)')
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
    new = build({k: v for k, v in model.config.items() if k not in ('aux', 'aux_parts')})
    new.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith('aux_')})
    dev = next(model.parameters()).device
    return _carry(model, new).to(dev).train(model.training)


def aux_widen_problem(have, want):
    """Why a head for the targets `have` cannot be widened to `want` (docs/az.md 37), or None: every target
    of `have` must be in `want` (by name) with the same width and loss, and `want` must add one. A target
    removed or renamed, a width or a loss changed, no head at all (`add_aux`) or the same targets in another
    order (shards of the two orders could not be told apart: the training window places a narrower shard's
    columns by its width) are refused."""
    have, want = aux_layout(have), aux_layout(want)
    if not have:
        return 'the network has no auxiliary head (add_aux gives it a fresh one)'
    spec = {n: (w, loss) for n, w, loss in want}
    for n, w, loss in have:
        if n not in spec:
            return (f'its target {n!r} is not among {[t[0] for t in want]} (a target removed or renamed: refused - '
                    f'the head would lose what it learned; net.strip_aux, then fork the run)')
        if spec[n] != (w, loss):
            return f'its target {n!r} is {w} wide ({loss}), wanted {spec[n][0]} wide ({spec[n][1]})'
    if len(want) == len(have):
        return f'the same targets in another order ({[t[0] for t in have]} -> {[t[0] for t in want]})'
    return None


def widen_aux(model, aux):
    """`model` with its auxiliary head widened to the targets `aux` (docs/az.md 37): every target of the
    head's in `aux` with the same width and loss, `aux` adding at least one (`aux_widen_problem`, else
    ValueError) -> a copy whose head's last layer is the one it had and whose new targets' outputs come from
    a layer of their own on the same hidden layer (config `aux_parts`, parameters `aux_more.<k>`), zero: every
    parameter and buffer of `model` exactly as it was, every output of before - policy, value, margin, each
    old target's raw outputs, the readers' - the same computation bit for bit (on any device: one wider layer
    would change the matrix product's kernel), each new target predicting what zeros predict (uniform for
    'multinomial'). The targets may be declared in any order: the outputs are put in `aux`'s order. The
    global torch RNG is untouched. The copy remembers the layout it was widened from (`aux_older`, saved in
    its checkpoints' extra: the training window places the narrower shards of before by it) and, until it is
    saved, `aux_widened` (that layout: the run logs it). A network with that head already: `model`."""
    aux = aux_layout(aux)
    have = aux_layout(model.config.get('aux'))
    if have == aux:
        return model
    why = aux_widen_problem(have, aux)
    if why:
        raise ValueError(f'widen_aux: {why}')
    old = {n for n, _, _ in have}
    parts = [list(p) for p in model.config.get('aux_parts') or [[n for n, _, _ in have]]]
    parts.append([n for n, _, _ in aux if n not in old])
    with torch.random.fork_rng(devices=[]):
        new = build(dict(model.config, aux=aux, aux_parts=parts))
    missing, unexpected = new.load_state_dict(model.state_dict(), strict=False)
    added = {f'aux_more.{len(parts) - 2}.weight', f'aux_more.{len(parts) - 2}.bias'}
    if unexpected or set(missing) != added:
        raise RuntimeError(f'widen_aux: state mismatch (missing {missing}, unexpected {unexpected})')
    _carry(model, new)
    new.aux_older = [list(map(list, o)) for o in getattr(model, 'aux_older', None) or []]
    if have not in new.aux_older:
        new.aux_older.append(have)
    new.aux_widened = have
    dev = next(model.parameters()).device
    return new.to(dev).train(model.training)


def game_aux(model, game=None):
    """`model` with its auxiliary head widened to `game`'s declared targets (the active game package's `aux`;
    default the game in its config) when they include the head's and add some (`widen_aux`, docs/az.md 37),
    else `model` as it is - a network without a head, with the game's head, or with a head the game's targets
    cannot widen (the run refuses that where the head is needed: its training, a root belief)."""
    have = aux_layout(model.config.get('aux'))
    if not have:
        return model
    from . import package
    want = package.active().aux_of(game or model.config.get('game'))
    if not want or want == have or aux_widen_problem(have, want):
        return model
    return widen_aux(model, want)


CARRIED = ('reader_detach', 'aux_older', 'aux_widened')

# parameters without weight decay (docs/az.md 43.6): a gated segment's gate theta (g = max(theta, 0)) - a scalar the
# decay would bias shut. Every other parameter is decayed as before.
NO_DECAY = ('gate',)


def varlen_features(model):
    """What of variable-length token sets (docs/az.md 43.5-43.7) a network has, as short descriptions ([]: none):
    tail segments, pointer overrides, gates - the parts an exporter of fixed-width rows would drop in silence (R14:
    `zero.webexport`, zero-onnx refuse a network with any of them)."""
    out = []
    segs = getattr(model, 'segs', None) or []
    if getattr(model, 'tails', None):
        out.append('tail segments ' + ', '.join(segs[j]['name'] for j in model.tails))
    if getattr(model, 'option_pointers', False):
        out.append('pointer overrides (option_pointers)')
    if getattr(model, 'gated', None):
        out.append('gates on ' + ', '.join(segs[j]['name'] for j in model.gated))
    return out


def no_decay_params(model):
    """`model`'s parameters that AdamW must not decay (`NO_DECAY`), in parameter order ([]: none - every optimizer
    of before, unchanged)."""
    return [p for n, p in model.named_parameters() if n in NO_DECAY]


def _carry(src, dst):
    """`dst` with `src`'s Python attributes that are not in its config or state (CARRIED: the readers'
    training gradient, the layouts its head was widened from) -> `dst`."""
    for k in CARRIED:
        if k in src.__dict__:
            setattr(dst, k, src.__dict__[k])
    return dst


def add_reader(model, readers, seed=0):
    """`model` with the readers `readers` (`reader_layout`, docs/az.md 36): a network without readers (an
    older checkpoint) -> a copy with every other parameter and buffer exactly as they were and fresh readers
    (initialised from `seed`, the global torch RNG untouched; each one's last layer and its projection into
    the trunk zero), so its policy / value / margin / auxiliary-head outputs are unchanged. A network with
    those readers already: `model` itself; with other ones: ValueError (a reader is part of the trunk's
    input: removing or changing one changes the network)."""
    readers = reader_layout(readers)
    have = reader_layout(model.config.get('readers'))
    if have == readers:
        return model
    if have or not readers:
        raise ValueError(f'add_reader: the network has the readers {have}, wanted {readers}')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        new = build(dict(model.config, readers=readers))
    missing, unexpected = new.load_state_dict(model.state_dict(), strict=False)
    if unexpected or not missing or any(not k.startswith('readers.') for k in missing):
        raise RuntimeError(f'add_reader: state mismatch (missing {missing}, unexpected {unexpected})')
    new.reader_detach = getattr(model, 'reader_detach', True)
    _carry(model, new)                             # (a widened head's layouts of before, docs/az.md 37)
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
    raises finite.NonFinite instead of writing it (docs/az.md 4.1). A network whose auxiliary head
    was widened (`widen_aux`) also saves the layouts it had before in `extra['aux_older']`."""
    state = model.state_dict()
    check_state(state, path)
    older = getattr(model, 'aux_older', None)
    if older:
        extra = dict(extra or {}, aux_older=[list(map(list, o)) for o in older])
    tmp = path + f'.{os.getpid()}.tmp'
    torch.save({'config': model.config, 'state': state, 'extra': extra or {}}, tmp)
    _replace(tmp, path)


def load(path, device='cpu', game=None, allow_mismatch=False):
    """A checkpoint -> (network, extra). The encoding guard first (docs/az.md 10.4): a network
    built for another encoding than the one this build's `game` feeds it (default: the game in its
    config) raises EncodingMismatch (`check_encoding`). `allow_mismatch=True` skips the guard, for
    the callers that take such a network on purpose: tools/fork_run.py's widen, co-training's
    resolver_init (both check what they write), tools/selfplay_digest.py and the widen tests.
    An auxiliary head that predicts some of the game's declared targets is widened to all of them
    (`game_aux`, docs/az.md 37: every output of before unchanged, bit for bit); the layouts a head was
    widened from come back as `aux_older` (from `extra`)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if not allow_mismatch:
        check_encoding(ck['config'], game, path)
    m = build(ck['config'])
    m.load_state_dict(ck['state'])
    extra = ck.get('extra', {})
    if extra.get('aux_older'):
        m.aux_older = [aux_layout(o) for o in extra['aux_older']]
    m = game_aux(m, game)
    m.to(device).eval()
    return m, extra


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
    for k in ('option_pointers', 'max_row_tokens'):          # (docs/az.md 43.5)
        if a.get(k) != b.get(k):
            out.append(f'{k} {a.get(k)!r} -> {b.get(k)!r}')
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
            # (segments: every key, the tails' `tail` / `gate` / `columns` included; the layout's pointer
            # overrides and row limit, docs/az.md 43.5 - absent on both sides for every layout before them)
            reads = ('vocab', 'segments', 'bags', 'option_pointers', 'max_row_tokens')
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


def make_net(info, game, arch='mlp', hidden=256, blocks=4, ch=64, heads=4, ff=0, aux=None, readers=None):
    """A new network for `game` (info = games.info(game)): 'mlp' (`Net`, hidden x blocks), 'grid'
    (`GridNet` over the game package's board planes, ch x blocks) or 'tokens' (`TokenNet` over
    the game package's object layout: d = hidden, layers = blocks, heads, ff (0: 2 x hidden)); `aux`:
    its auxiliary targets (`aux_layout`; None: no auxiliary head, the network of before); `readers`:
    its readers (`reader_layout`, docs/az.md 36; None: none, the network of before)."""
    kw = {'aux': aux} if aux else {}
    if readers:
        kw['readers'] = readers
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
        if s['count']:
            inside.update(range(s['offset'], s['offset'] + s['count'] * s['width']))
    for b in lay['bags']:
        inside.update(range(b['offset'], b['offset'] + b['count'] * b['width']))
    return [(None, [i for i in range(features) if i not in inside])]


def token_feature_map(old_lay, old_features, new_lay, new_features):
    """Where each feature of a token layout went in a wider one (docs/az.md 10.3): a list, per old
    feature, of its new index. Raises ValueError unless `new_lay` keeps the old layout's tokens and
    meanings and only adds inputs:
      * the same segments (names, order, slot counts, present flags, identities, actions, pointer
        columns, gates): a new segment with head slots, or more slots, would add tokens to the
        attention; a segment's slots may get wider, the old floats first in each slot (the new ones
        after them); a segment may gain a `tail` (never lose it); tail-only segments (count 0) may be
        appended after the old ones (docs/az.md 43.8: `_widen_tokens` gates them); the layout may gain
        `option_pointers` (never lose it) and `max_row_tokens`;
      * every old bag (by name; the new layout may add bags) with the same width and its old rows
        first (new rows after them);
      * the old globals: by block name when the old layout declares blocks (the old floats first
        in each), else all of them as the first globals of the new layout in feature order (new
        globals after the old ones in feature order).
    Both layouts normalised (`tokens_layout`)."""
    if int(new_lay['vocab']) != int(old_lay['vocab']):
        raise ValueError(f'widen: the vocabulary changed ({old_lay["vocab"]} -> {new_lay["vocab"]})')
    os_, ns = old_lay['segments'], new_lay['segments']
    if [s['name'] for s in ns[:len(os_)]] != [s['name'] for s in os_]:
        raise ValueError(f'widen: a token layout must keep its segments (names, order; new segments only after them): '
                         f'{[s["name"] for s in os_]} -> {[s["name"] for s in ns]} (new tokens change the attention)')
    for b in ns[len(os_):]:
        # (docs/az.md 43.8: a new segment may only hold tokens of a kind the old layout had no slot for -
        # tail-only, count 0, claiming no feature; `_widen_tokens` gates it)
        if b['count'] != 0 or not b.get('tail'):
            raise ValueError(f'widen: new segment {b["name"]} has head slots (count {b["count"]}): only tail-only segments '
                             f'(count 0, tail) may be appended (new head tokens change the attention)')
    if old_lay.get('option_pointers') and not new_lay.get('option_pointers'):
        raise ValueError('widen: the layout lost option_pointers (the network would ignore the overrides of its rows)')
    fmap = [None] * old_features
    for a, b in zip(os_, ns):
        same = ('count', 'present', 'id', 'actions', 'id_actions', 'columns', 'gate')
        if any(a.get(k) != b.get(k) for k in same):
            raise ValueError(f'widen: segment {a["name"]} changed its slots, present flag, identity, actions, pointer '
                             f'columns or gate')
        if a.get('tail') and not b.get('tail'):
            raise ValueError(f'widen: segment {a["name"]} lost its tail (its tail tokens would have no place)')
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
    """A normalised token layout without its identity keys (name, version) and its `draft` mark (docs/az.md
    43.5): the layout as the network reads it, plus its declared global blocks."""
    return {k: v for k, v in lay.items() if k not in ('name', 'version', 'draft')}


def _map_reader(spec, fmap):
    """A reader's spec with its inputs moved by a feature map (`token_feature_map`), as runs [[offset,
    count], ...] in its input order."""
    runs = []
    for f in (fmap[i] for i in reader_index(spec)):
        if runs and runs[-1][0] + runs[-1][1] == f:
            runs[-1][1] += 1
        else:
            runs.append([f, 1])
    return dict(spec, inputs=runs)


def _gate_new_segments(ol, lay, cfg, ungated=False):
    """The layout `lay` (normalised) with every segment it appends after the old layout `ol`'s gated
    (docs/az.md 43.8: `gate: true` written in, so the widened network's new tokens weigh g = 0 and change
    nothing) - unless `ungated` (the new tokens then perturb the outputs of the rows where they appear; a
    layout that gates one of them anyway is refused as contradictory)."""
    n_old, segs = len(ol['segments']), []
    for j, s in enumerate(lay['segments']):
        if j >= n_old:
            if ungated and s.get('gate'):
                raise ValueError(f'widen: ungated, but the layout gates the new segment {s["name"]}')
            if not ungated and not s.get('gate'):
                s = dict(s, gate=True)
        segs.append(s)
    return tokens_layout(dict(lay, segments=segs), cfg['features'], cfg['actions'])


def _ptr_columns(lay):
    """The pointer columns of a normalised token layout (`TokenNet.ptr`'s used rows)."""
    return sum(len(s['actions']) + ('id_actions' in s) + s.get('columns', 0) for s in lay['segments'])


def _widen_tokens(old, cfg, lay, ungated=False):
    """`widen` of a TokenNet onto the layout `lay` (normalised; module docs of `widen`)."""
    ol = old.config['layout']
    F0, F1 = old.config['features'], cfg['features']
    lay = _gate_new_segments(ol, lay, cfg, ungated)
    fmap = token_feature_map(ol, F0, lay, F1)
    cfg = dict(cfg, layout=lay)
    if cfg.get('readers'):
        # a reader reads the same floats where the new layout put them (its inputs' order, so its first
        # layer's columns, unchanged; docs/az.md 36)
        cfg['readers'] = [_map_reader(s, fmap) for s in reader_layout(cfg['readers'])]
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
    # (by segment name below: the tail positions, the gates, the pointer head's rows - docs/az.md 43.8)
    later = {'tail_base', 'gate', 'ptr.weight', 'ptr.bias'} | {f'pos_tail.{k}.weight' for k in range(len(old.tails))}
    for k, v in old_st.items():
        if k in later:
            continue
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
    # the pointer head: the old columns' rows (new segments' columns come after them), the new ones zero
    nc = _ptr_columns(ol)
    for k in ('ptr.weight', 'ptr.bias'):
        if st[k].shape == old_st[k].shape:
            st[k] = old_st[k]
        else:
            st[k] = torch.zeros_like(st[k])
            st[k][:nc] = old_st[k][:nc]
    # tail positions by segment name: a segment that had a tail keeps its own; one that gains a tail starts
    # at its last head slot's embedding (the old network's); a new tail-only segment keeps its fresh init
    names = [s['name'] for s in lay['segments']]
    otl = [names[j] for j in old.tails]
    if new.tails:
        tb = st['tail_base'].clone()
        for k, j in enumerate(new.tails):
            if names[j] in otl:
                ko = otl.index(names[j])
                tb[k] = old_st['tail_base'][ko]
                st[f'pos_tail.{k}.weight'] = old_st[f'pos_tail.{ko}.weight']
            elif lay['segments'][j]['count']:
                tb[k] = st['pos'][new.seg_start[j] + lay['segments'][j]['count'] - 1]
        st['tail_base'] = tb
    # gates: an old gated segment's as it was, a new one's exactly 0 (its tokens change nothing)
    if new.gated:
        ogl = [names[j] for j in old.gated]
        g = torch.zeros(len(new.gated))
        for k, j in enumerate(new.gated):
            if names[j] in ogl:
                g[k] = old_st['gate'][ogl.index(names[j])]
        st['gate'] = g
    new.load_state_dict(st)
    return new.eval()


def widen(old, features=None, actions=None, grid=None, layout=None, ungated=False):
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
    network computes what `old` computed on the old features of it (docs/az.md 10.3). Variable-length
    token sets (docs/az.md 43.8): a segment may gain a `tail` (new tail positions, unused by every old row)
    and tail-only segments may be appended, gated (`gate: true` written into the layout, g = 0: their
    tokens change nothing; `ungated=True`: not gated) with fresh input maps and positions; new pointer
    columns get zero rows. The copy keeps `old`'s attributes outside its config and state (`CARRIED`)."""
    return _carry(old, _widen(old, features, actions, grid, layout, ungated))


def _widen(old, features=None, actions=None, grid=None, layout=None, ungated=False):
    """`widen` without the carried attributes."""
    cfg = dict(old.config)
    cfg['features'] = features or cfg['features']
    cfg['actions'] = actions or cfg['actions']
    if cfg['features'] < old.config['features'] or cfg['actions'] < old.config['actions']:
        raise ValueError('widen only grows a network')
    if cfg.get('arch') == 'tokens' and layout is not None:
        lay = tokens_layout(layout, cfg['features'], cfg['actions'])
        if layout_sans_identity(lay) != layout_sans_identity(old.config['layout']):
            return _widen_tokens(old, cfg, lay, ungated)
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
