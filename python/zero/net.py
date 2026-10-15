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
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    cells are row-major, as the planes), hex ('odd-r' adds a row-parity plane, or none)}."""

    def __init__(self, features, actions, grid, ch=64, blocks=6, seats=MAX_SEATS, game='toy', head=128, ghid=128,
                 arch='grid'):
        super().__init__()
        grid = {'planes': int(grid['planes']), 'h': int(grid['h']), 'w': int(grid['w']), 'offset': int(grid.get('offset', 0)),
                'cell_action0': int(grid.get('cell_action0', 0)), 'hex': grid.get('hex', 'odd-r')}
        self.config = {'arch': 'grid', 'features': features, 'actions': actions, 'grid': grid, 'ch': ch, 'blocks': blocks,
                       'seats': seats, 'game': game, 'head': head, 'ghid': ghid}
        self.features = features
        self.seats = seats
        P, H, W = grid['planes'], grid['h'], grid['w']
        self.P, self.H, self.W, self.off, self.a0 = P, H, W, grid['offset'], grid['cell_action0']
        self.n = P * H * W
        if self.off + self.n > features or self.a0 + H * W > actions:
            raise ValueError(f'grid {grid} does not fit {features} features / {actions} actions')
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
        p = x[:, o:o + n].reshape(B, self.P, self.H, self.W)
        g = self.g(torch.cat([x[:, :o], x[:, o + n:]], 1))
        if self.par.shape[1]:
            p = torch.cat([p, self.par.expand(B, -1, -1, -1)], 1)
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


def build(config):
    """A network from its config (a checkpoint's): `GridNet` for arch 'grid', else `Net`."""
    cfg = dict(config)
    if cfg.get('arch', 'mlp') == 'grid':
        return GridNet(**cfg)
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
    tmp = path + f'.{os.getpid()}.tmp'          # atomic: a crash never leaves a half-written checkpoint
    torch.save({'config': model.config, 'state': model.state_dict(), 'extra': extra or {}}, tmp)
    _replace(tmp, path)


def load(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build(ck['config'])
    m.load_state_dict(ck['state'])
    m.to(device).eval()
    return m, ck.get('extra', {})


def widen(old, features=None, actions=None):
    """A copy of `old` with `features` inputs / `actions` policy outputs (>= the old ones):
    new input columns get zero weights (the old outputs are unchanged), new actions zero
    weights and bias (logit 0)."""
    cfg = dict(old.config)
    cfg['features'] = features or cfg['features']
    cfg['actions'] = actions or cfg['actions']
    if cfg['features'] < old.config['features'] or cfg['actions'] < old.config['actions']:
        raise ValueError('widen only grows a network')
    new = build(cfg)
    st = new.state_dict()
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
