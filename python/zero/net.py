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
  aux    : optional (config key `aux` = the game's auxiliary target segments, `SearchGame::AUX`,
           e.g. [['own', 400], ['score', 12]]; KataGo-style auxiliary targets, docs/az.md 4.1):
           only returned by `forward(x, n, aux=True)` (training), so inference - `infer`,
           its CUDA graphs - is unchanged. Segment 'own' = final ownership per cell, tanh
           (GridNet: a 1x1 conv on the tower, one value per board cell; the grid's h x w cells);
           the other segments are regression values (GridNet: an MLP on the pooled trunk). The
           heads' last layers start at zero (they predict 0); `add_aux` adds them to a network
           without (an older checkpoint: everything else copied, its outputs unchanged),
           `strip_aux` removes them (older code cannot build a network with `aux`).

Checkpoints (`save`, atomic): {config, state, extra}; `load` rebuilds the network from the
config; `widen` maps a checkpoint onto a larger feature / action space (new inputs get zero
weights, new actions zero logits), so a widened network computes what the source did.
"""
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .protocol import MAX_SEATS


def aux_segments(aux):
    """An auxiliary target layout ([(name, width), ...], a game's `aux_layout` or a config's
    `aux`; None = none) -> [[name, width], ...]."""
    return [[str(n), int(w)] for n, w in (aux or [])]


def _zero(layer):
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


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
        # auxiliary heads last: the other parameters are initialised as without them
        self.aux_layout = aux_segments(aux)
        if self.aux_layout:
            self.config['aux'] = self.aux_layout
            width = sum(w for _, w in self.aux_layout)
            self.aux_head = nn.Sequential(nn.Linear(hidden, head), nn.ReLU(), nn.Linear(head, width))
            _zero(self.aux_head[2])
            tanh = torch.cat([torch.full((w,), n == 'own') for n, w in self.aux_layout])
            self.register_buffer('aux_tanh', tanh, persistent=False)

    def forward(self, x, n_players, aux=False):
        """x [B, >= features] (an older, narrower network reads a prefix), n_players [B]
        -> policy logits [B, actions], value logits [B, seats] (absent seats -1e9),
        margin [B, seats] (+ with aux=True the auxiliary predictions [B, aux width])."""
        h = F.relu(self.inp(x[:, :self.features]))
        for b in self.blocks:
            h = b(h)
        h = F.relu(self.ln(h))
        present = torch.arange(self.seats, device=x.device)[None, :] < n_players[:, None]
        vlog = self.val(h).masked_fill(~present, -1e9)
        if aux:
            if not self.aux_layout:
                raise ValueError('forward(aux=True): the network has no auxiliary heads (net.add_aux)')
            a = self.aux_head(h)
            return self.pol(h), vlog, self.mar(h), torch.where(self.aux_tanh, torch.tanh(a), a)
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
                 arch='grid', aux=None):
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
        # auxiliary heads last: the other parameters are initialised as without them
        self.aux_layout = aux_segments(aux)
        self.aux_k = sum(w for n, w in self.aux_layout if n != 'own')
        if self.aux_layout:
            names = [n for n, _ in self.aux_layout]
            if names.count('own') > 1 or any(w != cells for n, w in self.aux_layout if n == 'own'):
                raise ValueError(f"aux {self.aux_layout}: at most one 'own' segment, of the grid's {cells} cells")
            self.config['aux'] = self.aux_layout
            if 'own' in names:
                self.aux_own = nn.Conv2d(ch, 1, 1)
                _zero(self.aux_own)
            if self.aux_k:
                self.aux_score = nn.Sequential(nn.Linear(256, head), nn.ReLU(), nn.Linear(head, self.aux_k))
                _zero(self.aux_score[2])

    def forward(self, x, n_players, aux=False):
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
            if not self.aux_layout:
                raise ValueError('forward(aux=True): the network has no auxiliary heads (net.add_aux)')
            sc = self.aux_score(e) if self.aux_k else None
            parts, k = [], 0
            for name, w in self.aux_layout:
                if name == 'own':
                    parts.append(torch.tanh(self.aux_own(h).reshape(B, self.H * self.W)))
                else:
                    parts.append(sc[:, k:k + w])
                    k += w
            return pol, vlog, self.mar(e), torch.cat(parts, 1)
        return pol, vlog, self.mar(e)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def build(config):
    """A network from its config (a checkpoint's): `GridNet` for arch 'grid', else `Net`."""
    cfg = dict(config)
    if cfg.get('arch', 'mlp') == 'grid':
        return GridNet(**cfg)
    cfg.pop('arch', None)
    return Net(**cfg)


def add_aux(model, aux, seed=0):
    """`model` with auxiliary heads for the layout `aux` (`aux_segments`): a network without
    heads (an older checkpoint) -> a copy with every weight kept and new heads (initialised from
    `seed`; their last layers zero: they predict 0 and send the trunk no gradient until they
    have learned something), whose policy / value / margin outputs are unchanged. A network
    with that layout already: `model` itself; with another layout: ValueError."""
    aux = aux_segments(aux)
    have = aux_segments(model.config.get('aux'))
    if have == aux:
        return model
    if have or not aux:
        raise ValueError(f'add_aux: the network has auxiliary heads {have}, wanted {aux}')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        new = build(dict(model.config, aux=aux))
    missing, unexpected = new.load_state_dict(model.state_dict(), strict=False)
    if unexpected or not missing or any(not k.startswith('aux_') for k in missing):
        raise RuntimeError(f'add_aux: state mismatch (missing {missing}, unexpected {unexpected})')
    dev = next(model.parameters()).device
    return new.to(dev).train(model.training)


def strip_aux(model):
    """A copy of `model` without auxiliary heads (config and state; the same policy / value /
    margin): for code that predates them (e.g. a play server pinned to an older core)."""
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
    tmp = path + f'.{os.getpid()}.tmp'          # atomic: a crash never leaves a half-written checkpoint
    torch.save({'config': model.config, 'state': model.state_dict(), 'extra': extra or {}}, tmp)
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


def widen(old, features=None, actions=None, grid=None):
    """A copy of `old` with `features` inputs / `actions` policy outputs (>= the old ones):
    new input columns get zero weights (the old outputs are unchanged), new actions zero
    weights and bias (logit 0). GridNet: `grid` = the new board grid (the game package's) when
    it adds planes (the old planes must come first, in the same order): the new planes get zero
    input weights in the first conv."""
    cfg = dict(old.config)
    cfg['features'] = features or cfg['features']
    cfg['actions'] = actions or cfg['actions']
    if cfg['features'] < old.config['features'] or cfg['actions'] < old.config['actions']:
        raise ValueError('widen only grows a network')
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
