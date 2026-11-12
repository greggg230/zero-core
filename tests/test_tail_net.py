"""Variable-length token sets in the token network (docs/az.md 43.5-43.8; the varlen API's section 4): tail
tokens after a row's fixed-width head, tail-only segments, gates exactly 0 at a widen, pointer overrides
resolved on the host, the static forward, the widen and its refusals, the evaluator, fork_run.

Built from tensors and the toy's layout written out literally (the API's 1.4, `package.TOY_TAIL_TOKENS`), not
from the Rust game. `zero.rows` (lane X) is used when it exists; until then a stub of its frozen API (below)
stands in for it, and the network-level tests use the stub's resolution either way (they test the network,
not the rows container)."""
import os
import sys
import types

import numpy as np
import pytest
import torch

from zero import games as G, package
from zero.infer import NetEvaluator, token_index_tail, token_presence
from zero.net import (EncodingMismatch, Net, TBlock, TokenNet, encoding_mismatch, layout_sans_identity, load, save,
                      tokens_layout, widen)

CUDA = torch.cuda.is_available()
DEVS = ['cpu'] + (['cuda'] if CUDA else [])

# ---------------------------------------------------------------- the toy's layouts (the API's 1.4, literally)
TOY_TAIL_TOKENS = {
    "vocab": 16, "name": "toy_tail", "version": 1, "option_pointers": True, "max_row_tokens": 64,
    "segments": [
        {"name": "items", "offset": 8, "count": 4, "width": 16, "present": 0, "id": {"at": 6, "radix": [10]},
         "actions": [[4, 1]], "tail": True},
        {"name": "marks", "offset": None, "count": 0, "width": 4, "present": 0, "id": None, "actions": [], "tail": True}],
    "bags": [],
    "globals": [{"name": "globals", "offset": 0, "count": 8}]}
TOY_TAIL_TOKENS_BASE = dict(TOY_TAIL_TOKENS, segments=TOY_TAIL_TOKENS['segments'][:1])
F, A = 72, 16
# the same objects without any key of the tails (a checkpoint of before: the widen adds the tail on items,
# option_pointers, max_row_tokens and the gated marks)
HEAD_ONLY = {"vocab": 16, "name": "toy_tail", "version": 1,
             "segments": [{k: v for k, v in TOY_TAIL_TOKENS['segments'][0].items() if k != 'tail'}],
             "bags": [], "globals": [{"name": "globals", "offset": 0, "count": 8}]}
# the game's layout as a widen writes it (the new segment gated)
TOY_GATED = dict(TOY_TAIL_TOKENS, segments=[TOY_TAIL_TOKENS['segments'][0],
                                            dict(TOY_TAIL_TOKENS['segments'][1], gate=True)])
# one more gated tail-only segment with a pointer column of its own (overrides into a gated segment: the blend)
OFFERS_SEG = {"name": "offers", "offset": None, "count": 0, "width": 4, "present": 0, "id": None, "actions": [],
              "tail": True, "columns": 1}
OFFERS = dict(TOY_GATED, segments=TOY_GATED['segments'] + [dict(OFFERS_SEG, gate=True)])
WIDTHS = {0: 16, 1: 4, 2: 4}
AUX = [['a', 5, 'multinomial']]


# ---------------------------------------------------------------- the rows API (zero.rows, lane X), a stub
def pack_tok(seg, col, slot):
    return (np.int64(seg) << np.int64(40)) | (np.int64(col) << np.int64(32)) | np.int64(slot)


def unpack_tok(tok):
    tok = np.asarray(tok, np.int64)
    return (tok >> 40) & 0xFFFF, (tok >> 32) & 0xFF, tok & 0xFFFFFFFF


class StubTail:
    """`rows.Tail` (the API's 5.1): a host CSR of rows' tail tokens."""

    def __init__(self, off, seg, slot, val, widths):
        self.off, self.seg, self.slot = np.asarray(off, np.int64), np.asarray(seg, np.uint16), np.asarray(slot, np.uint32)
        self.val, self.widths = np.asarray(val, np.float32), dict(widths)

    @classmethod
    def from_arrays(cls, d, widths, prefix='t_'):
        off = d.get(prefix + 'off', d.get(prefix + 'ptr'))
        return cls(off, d[prefix + 'seg'], d[prefix + 'slot'], d[prefix + 'val'], widths)

    @classmethod
    def empty(cls, n, widths):
        return cls(np.zeros(n + 1), [], [], [], widths)

    def _voff(self):
        w = np.array([self.widths[int(s)] for s in self.seg], np.int64)
        return np.concatenate([[0], np.cumsum(w)]).astype(np.int64)

    def take(self, rows):
        rows = np.asarray(rows, np.int64)
        vo = self._voff()
        toks = np.concatenate([np.arange(self.off[r], self.off[r + 1]) for r in rows]) if len(rows) else np.zeros(0, np.int64)
        toks = toks.astype(np.int64)
        off = np.concatenate([[0], np.cumsum(self.off[rows + 1] - self.off[rows])])
        val = np.concatenate([self.val[vo[t]:vo[t + 1]] for t in toks]) if len(toks) else np.zeros(0, np.float32)
        return StubTail(off, self.seg[toks], self.slot[toks], val, self.widths)

    @staticmethod
    def concat(parts):
        off = [0]
        for p in parts:
            off.extend((p.off[1:] + off[-1]).tolist())
        return StubTail(off, np.concatenate([p.seg for p in parts]), np.concatenate([p.slot for p in parts]),
                        np.concatenate([p.val for p in parts]), parts[0].widths)

    def counts(self):
        return np.diff(self.off)

    def nbytes(self):
        return self.off.nbytes + self.seg.nbytes + self.slot.nbytes + self.val.nbytes

    def padded(self):
        NT = len(self.seg)
        Wt = max(self.widths.values())
        val = np.zeros((NT, Wt), np.float32)
        wt = np.zeros(max(self.widths) + 1, np.int64)
        wt[list(self.widths)] = list(self.widths.values())
        w = wt[self.seg.astype(np.int64)]
        vo = np.concatenate([[0], np.cumsum(w)]).astype(np.int64)
        val[np.repeat(np.arange(NT), w), np.arange(vo[-1]) - np.repeat(vo[:-1], w)] = self.val
        row = np.repeat(np.arange(len(self.off) - 1), np.diff(self.off)).astype(np.int64)
        return row, self.seg.astype(np.int64), self.slot.astype(np.int64), val

    def to_torch(self, device, pad_to=None):
        row, seg, slot, val = self.padded()
        if pad_to is not None and pad_to > len(seg):
            k = pad_to - len(seg)
            s0 = min(self.widths)
            row = np.concatenate([row, np.zeros(k, np.int64)])
            seg = np.concatenate([seg, np.full(k, s0, np.int64)])
            slot = np.concatenate([slot, np.full(k, STUB_COUNTS[s0], np.int64)])
            val = np.concatenate([val, np.zeros((k, val.shape[1]), np.float32)])
        return {'row': torch.from_numpy(row).to(device), 'seg': torch.from_numpy(seg).to(device),
                'slot': torch.from_numpy(slot).to(device), 'val': torch.from_numpy(val).to(device)}


STUB_COUNTS = {0: 4, 1: 0, 2: 0}


class StubOverrides:
    """`rows.Overrides` (the API's 5.1)."""

    def __init__(self, off, opt, tok):
        self.off, self.opt, self.tok = np.asarray(off, np.int64), np.asarray(opt, np.int32), np.asarray(tok, np.int64)

    @classmethod
    def from_arrays(cls, d, prefix='ov_'):
        return cls(d.get(prefix + 'off', d.get(prefix + 'ptr')), d[prefix + 'opt'], d[prefix + 'tok'])

    @classmethod
    def empty(cls, n):
        return cls(np.zeros(n + 1), [], [])

    def take(self, rows):
        rows = np.asarray(rows, np.int64)
        idx = np.concatenate([np.arange(self.off[r], self.off[r + 1]) for r in rows]).astype(np.int64) if len(rows) \
            else np.zeros(0, np.int64)
        return StubOverrides(np.concatenate([[0], np.cumsum(self.off[rows + 1] - self.off[rows])]), self.opt[idx],
                             self.tok[idx])


def layout_tails(lay):
    return {j: (s['width'], s['count'], k) for k, (j, s) in enumerate((j, s) for j, s in enumerate(lay['segments'])
                                                                       if s.get('tail'))}


def stub_resolve(tail, ov, opt_off, opt_action, lay):
    """`rows.resolve` (the API's 5.1): each override -> (row, the option's action, the target's extended
    position, its column); ValueError for an option out of its row, a missing tail token, a column out of
    range (a head target's presence is the network's / the graphs' host check: no features here)."""
    segs = lay['segments']
    counts = [s['count'] for s in segs]
    start = np.concatenate([[1], 1 + np.cumsum(counts)]).astype(np.int64)
    n_tok = int(start[-1])
    ncol = [len(s['actions']) + ('id_actions' in s) + s.get('columns', 0) for s in segs]
    opt_off = np.asarray(opt_off, np.int64)
    where = {}
    if tail is not None and (np.asarray(ov.tok) != -1).any():
        row = np.repeat(np.arange(len(tail.off) - 1), np.diff(tail.off))
        for j, (r, s, sl) in enumerate(zip(row, tail.seg, tail.slot)):
            where[(int(r), int(s), int(sl))] = j
    out = {k: [] for k in ('row', 'act', 'ref', 'col')}
    for r in np.flatnonzero(np.diff(ov.off)):
        for i in range(ov.off[r], ov.off[r + 1]):
            o, t = int(ov.opt[i]), int(ov.tok[i])
            if not 0 <= o < opt_off[r + 1] - opt_off[r]:
                raise ValueError(f'override of option {o} out of row {r}')
            out['row'].append(int(r))
            out['act'].append(int(opt_action[opt_off[r] + o]))
            if t == -1:
                out['ref'].append(-1)
                out['col'].append(0)
                continue
            s, c, sl = (int(v) for v in unpack_tok(t))
            if c >= ncol[s]:
                raise ValueError(f'override column {c} of segment {s}')
            if sl < counts[s]:
                out['ref'].append(int(start[s] + sl))
            else:
                if (r, s, sl) not in where:
                    raise ValueError(f'override to a missing tail token {(r, s, sl)}')
                out['ref'].append(n_tok + where[(r, s, sl)])
            out['col'].append(c)
    return {k: np.asarray(v, np.int64) for k, v in out.items()}


def stub_token_full(P_head, tail_counts, n_tok, t_step):
    cnt = P_head.sum(1) + np.asarray(tail_counts)
    T = -(-int(cnt.max()) // t_step) * t_step
    return token_index_tail(P_head, tail_counts, n_tok, np.zeros((len(cnt), T), np.int64)), cnt.astype(np.int64)


def _rows_api():
    """zero.rows (lane X) when it exists, else the stub installed as zero.rows."""
    try:
        from zero import rows
        return rows
    except ImportError:
        pass
    import zero
    mod = types.ModuleType('zero.rows')
    mod.Tail, mod.Overrides, mod.pack_tok, mod.unpack_tok = StubTail, StubOverrides, pack_tok, unpack_tok
    mod.layout_tails, mod.resolve, mod.token_full = layout_tails, stub_resolve, stub_token_full
    sys.modules['zero.rows'] = mod
    zero.rows = mod
    return mod


R = _rows_api()


# ---------------------------------------------------------------- rows of the toy layout
def toy_batch(B, seed=0, p_head=0.6, p_tail=0.5, p_mark=0.5, p_gen=0.3, p_offer=0.0, big=False):
    """B rows of the toy layout as the driver sends them: features x [B, 72] (globals, 4 item slots, each present
    with p_head), a tail per row (items past the head with p_tail, 1-3 of them; marks with p_mark; offers with
    p_offer) as CSR arrays, options (PASS, TAKE of the present head items: static; TAKE of tail items and offers:
    overrides; a generic override with p_gen; MARK), the overrides' CSR. `big`: some slots near 2^32."""
    rng = np.random.default_rng(seed)
    x = np.zeros((B, F), np.float32)
    x[:, :8] = rng.normal(size=(B, 8))
    present = rng.random((B, 4)) < p_head
    for s in range(4):
        o = 8 + 16 * s
        r = np.flatnonzero(present[:, s])
        x[r, o] = 1
        x[r, o + 1:o + 6] = rng.normal(size=(len(r), 5))
        x[r, o + 6 + rng.integers(0, 10, len(r))] = 1
    t = {'t_off': [0], 't_seg': [], 't_slot': [], 't_val': []}
    o = {'opt_off': [0], 'opt_action': [], 'ov_off': [0], 'ov_opt': [], 'ov_tok': []}
    for b in range(B):
        acts, ov = [0] + [4 + s for s in range(4) if present[b, s]], []
        k = int(rng.integers(1, 4)) if rng.random() < p_tail else 0
        pool = np.arange(4, 40) if not big else np.array([4, 5, 1 << 20, (1 << 32) - 2, (1 << 32) - 1])
        for j, sl in enumerate(sorted(rng.choice(pool, size=k, replace=False))):
            v = np.zeros(16, np.float32)
            v[0], v[1:6], v[6 + rng.integers(10)] = 1, rng.normal(size=5), 1
            t['t_seg'].append(0)
            t['t_slot'].append(int(sl))
            t['t_val'].extend(v)
            acts.append(8 + j)                                    # TAKE of a tail item: an override
            ov.append((len(acts) - 1, pack_tok(0, 0, int(sl))))
        for sl in sorted(rng.choice(8, size=int(rng.integers(1, 3)) if rng.random() < p_mark else 0, replace=False)):
            t['t_seg'].append(1)
            t['t_slot'].append(int(sl))
            t['t_val'].extend([1.0, rng.random(), float(rng.random() < 0.5), 0.0])
        for j, sl in enumerate(sorted(rng.choice(6, size=int(rng.integers(1, 3)) if rng.random() < p_offer else 0,
                                                 replace=False))):
            t['t_seg'].append(2)
            t['t_slot'].append(int(sl))
            t['t_val'].extend([1.0, rng.normal(), rng.normal(), 0.0])
            acts.append(13 + j)                                   # an offer's option: an override into a gated segment
            ov.append((len(acts) - 1, pack_tok(2, 0, int(sl))))
        if rng.random() < p_gen:
            acts.append(11)                                       # a generic override
            ov.append((len(acts) - 1, -1))
        acts.append(12)                                           # MARK: the generic logit, no override
        o['opt_action'].extend(acts)
        o['opt_off'].append(len(o['opt_action']))
        for i, tok in ov:
            o['ov_opt'].append(i)
            o['ov_tok'].append(tok)
        o['ov_off'].append(len(o['ov_opt']))
        t['t_off'].append(len(t['t_seg']))
    d = {'x': x, 'n_players': rng.integers(2, 5, B).astype(np.int64),
         't_off': np.array(t['t_off'], np.int64), 't_seg': np.array(t['t_seg'], np.uint16),
         't_slot': np.array(t['t_slot'], np.uint32), 't_val': np.array(t['t_val'], np.float32),
         'opt_off': np.array(o['opt_off'], np.int64), 'opt_action': np.array(o['opt_action'], np.int64),
         'ov_off': np.array(o['ov_off'], np.int64), 'ov_opt': np.array(o['ov_opt'], np.int32),
         'ov_tok': np.array(o['ov_tok'], np.int64)}
    return d


def host_tail(d, widths=None):
    return StubTail.from_arrays(d, {k: v for k, v in WIDTHS.items() if k in (widths or WIDTHS)})


def host_ov(d):
    return StubOverrides.from_arrays(d)


def tail_of(net, d, dev='cpu'):
    """The network's `tail` of a batch (the widest tail segment of `net`'s layout)."""
    widths = {j: net.segs[j]['width'] for j in net.tails}
    return StubTail.from_arrays(d, widths).to_torch(dev)


def ov_of(net, d, dev='cpu'):
    widths = {j: net.segs[j]['width'] for j in net.tails}
    tl = StubTail.from_arrays(d, widths) if net.tails else None
    res = stub_resolve(tl, host_ov(d), d['opt_off'], d['opt_action'], net.config['layout'])
    return {k: torch.from_numpy(v).to(dev) for k, v in res.items()}


def no_tail(dev='cpu', wt=16):
    z = torch.zeros(0, dtype=torch.long, device=dev)
    return {'row': z, 'seg': z, 'slot': z, 'val': torch.zeros(0, wt, device=dev)}


def no_ov(dev='cpu'):
    z = torch.zeros(0, dtype=torch.long, device=dev)
    return {'row': z, 'act': z, 'ref': z, 'col': z}


def keep_segs(d, segs, gen=True):
    """The batch with only the tail tokens of segments `segs` (and the overrides into them; `gen`: the generic
    ones too)."""
    d = dict(d)
    B = len(d['t_off']) - 1
    t = {'t_off': [0], 't_seg': [], 't_slot': [], 't_val': []}
    vo = np.concatenate([[0], np.cumsum([WIDTHS[int(s)] for s in d['t_seg']])]).astype(np.int64)
    for b in range(B):
        for j in range(d['t_off'][b], d['t_off'][b + 1]):
            if int(d['t_seg'][j]) in segs:
                t['t_seg'].append(d['t_seg'][j])
                t['t_slot'].append(d['t_slot'][j])
                t['t_val'].extend(d['t_val'][vo[j]:vo[j + 1]])
        t['t_off'].append(len(t['t_seg']))
    o = {'ov_off': [0], 'ov_opt': [], 'ov_tok': []}
    for b in range(B):
        for i in range(d['ov_off'][b], d['ov_off'][b + 1]):
            tok = int(d['ov_tok'][i])
            if (tok == -1 and gen) or (tok != -1 and int(unpack_tok(tok)[0]) in segs):
                o['ov_opt'].append(d['ov_opt'][i])
                o['ov_tok'].append(tok)
        o['ov_off'].append(len(o['ov_opt']))
    d.update({'t_off': np.array(t['t_off'], np.int64), 't_seg': np.array(t['t_seg'], np.uint16),
              't_slot': np.array(t['t_slot'], np.uint32), 't_val': np.array(t['t_val'], np.float32),
              'ov_off': np.array(o['ov_off'], np.int64), 'ov_opt': np.array(o['ov_opt'], np.int32),
              'ov_tok': np.array(o['ov_tok'], np.int64)})
    return d


def leaf_batch(arrays, d, net):
    """A network LeafBatch of `arrays` with the batch's tail and overrides as the driver sends them: the CSR arrays
    (`t_off`, ..., `ov_tok`) and the tail segments (lane X's `LeafBatch(..., tails=)`, whose `.tail` / `.ov` read
    them); in a protocol without that argument, the rows.Tail / rows.Overrides objects in the arrays."""
    from zero.protocol import LeafBatch
    a = dict(arrays)
    if net.tails:
        a.update({k: d[k] for k in ('t_off', 't_seg', 't_slot', 't_val')})
    if net.option_pointers:
        a.update({k: d[k] for k in ('ov_off', 'ov_opt', 'ov_tok')})
    tails = {j: (net.segs[j]['width'], net.segs[j]['count'], k) for k, j in enumerate(net.tails)} or None
    try:
        return LeafBatch(0, 'm', 'net', a, tails=tails)
    except TypeError:
        widths = {j: net.segs[j]['width'] for j in net.tails}
        if net.tails:
            a['tail'] = R.Tail.from_arrays(d, widths)
        if net.option_pointers:
            a['ov'] = R.Overrides.from_arrays(d)
        return LeafBatch(0, 'm', 'net', a)


def rand_net(layout, seed=0, dev='cpu', aux=AUX, d=16, layers=2, heads=2):
    """A small token network with random policy, pointer, value, margin and auxiliary output layers."""
    torch.manual_seed(seed)
    m = TokenNet(F, A, layout, d=d, layers=layers, heads=heads, ff=2 * d, head=16, aux=aux, game='toy').eval()
    with torch.no_grad():
        for lin in [m.gen, m.ptr, m.val[-1], m.mar[-1]] + ([m.aux_head[2]] if aux else []):
            lin.weight.normal_(0, 0.3)
            lin.bias.normal_(0, 0.3)
    return m.to(dev)


def tx(d, dev='cpu'):
    return torch.from_numpy(d['x']).to(dev), torch.from_numpy(d['n_players']).to(dev)


def static_in(net, d, T=None, dev='cpu', with_tail=True):
    """forward_static's (full, ntok) of a batch: head tokens (`infer.token_presence`) then its tail tokens at
    their extended positions; T: the batch's tokens (default: its longest row's)."""
    P = token_presence(d['x'], net.present_index())
    tcnt = np.diff(d['t_off']) if with_tail else np.zeros(len(P), np.int64)
    cnt = P.sum(1) + tcnt
    full = token_index_tail(P, tcnt, net.n_tok, np.zeros((len(P), T or int(cnt.max())), np.int64))
    return torch.from_numpy(full).to(dev), torch.from_numpy(cnt).to(dev)


def equal(a, b):
    return len(a) == len(b) and all(torch.equal(u, v) for u, v in zip(a, b))


def maxdiff(a, b):
    return max(float((u - v).detach().abs().max()) for u, v in zip(a, b))


# ---------------------------------------------------------------- the layout
def test_layout_keys_normalise_only_when_set():
    """A layout without the new keys normalises as before (no new key; the token networks of before are the
    same: no new parameter, the same buffers); the new keys are kept when set; `draft` is not part of the
    layout the network reads."""
    old = tokens_layout(package.TOY_TOKENS, 195, 13)
    assert not {'option_pointers', 'max_row_tokens', 'draft'} & set(old)
    assert all(not {'tail', 'gate', 'columns'} & set(s) for s in old['segments'])
    m = TokenNet(195, 13, package.TOY_TOKENS, d=16, layers=1, heads=2, ff=32)
    assert not [k for k in m.state_dict() if k.startswith(('tail_base', 'pos_tail', 'gate'))]
    assert m.tails == [] and m.gated == [] and not m.option_pointers and m.tok_cols.shape[0] == m.n_tok
    lay = tokens_layout(TOY_TAIL_TOKENS, F, A)
    assert lay['option_pointers'] is True and lay['max_row_tokens'] == 64
    assert lay['segments'][0]['tail'] is True and lay['segments'][1] == {
        'name': 'marks', 'offset': None, 'count': 0, 'width': 4, 'present': 0, 'id': None, 'actions': [], 'tail': True}
    dr = tokens_layout(dict(TOY_TAIL_TOKENS, draft=True), F, A)
    assert dr['draft'] is True and layout_sans_identity(dr) == layout_sans_identity(lay)
    g = tokens_layout(OFFERS, F, A)
    assert g['segments'][2] == dict(OFFERS_SEG, gate=True)
    t = TokenNet(F, A, OFFERS, d=16, layers=1, heads=2, ff=32)
    assert t.tails == [0, 1, 2] and t.gated == [1, 2] and t.option_pointers
    assert t.tail_segments() == [(0, 'items', 16, 4), (1, 'marks', 4, 0), (2, 'offers', 4, 0)]
    assert t.gates() == {'marks': 1.0, 'offers': 1.0}                 # (built: 1; a widen: 0)
    assert t.ptr.out_features == 2 and t.seg_c0.tolist() == [0, 1, 1] and t.seg_nk.tolist() == [1, 0, 1]
    assert torch.equal(t.tail_base[0], t.pos[4]) and t.tail_base.shape == (3, 16)


def test_layout_validation():
    """The new keys' checks (the API's 1.3)."""
    s0, s1 = TOY_TAIL_TOKENS['segments']
    bad = [dict(TOY_TAIL_TOKENS, segments=[s0, dict(s1, tail=False)]),           # count 0 without tail
           dict(TOY_TAIL_TOKENS, segments=[s0, dict(s1, offset=60)]),            # tail-only with an offset
           dict(TOY_TAIL_TOKENS, segments=[dict(s0, offset=None), s1]),          # head slots without one
           dict(TOY_TAIL_TOKENS, segments=[dict(s0, gate=True), s1]),            # gate on a head-ful segment
           dict(TOY_TAIL_TOKENS, option_pointers=False, segments=[s0, dict(s1, columns=1)]),   # columns, no pointers
           dict(TOY_TAIL_TOKENS, segments=[s0, dict(s1, columns=0)]),
           {k: v for k, v in TOY_TAIL_TOKENS.items() if k != 'max_row_tokens'},  # tails without the limit
           dict(TOY_TAIL_TOKENS, max_row_tokens=4),                              # below 1 + 4 head slots
           dict(TOY_TAIL_TOKENS, segments=[s0, dict(s1, present=4)])]            # present outside the slot
    for b in bad:
        with pytest.raises(ValueError):
            tokens_layout(b, F, A)
    tokens_layout(dict(TOY_TAIL_TOKENS, max_row_tokens=5), F, A)
    # the action collision check covers head slots only (a tail-only segment's pairs are pointer columns)
    tokens_layout(dict(TOY_TAIL_TOKENS, segments=[s0, dict(s1, actions=[[4, 1]])]), F, A)


def test_psi_is_exact_for_every_u32():
    """A tail position's features (docs/az.md 43.5): the 32 digits of slot - count from the integers (2^32 - 1
    and 2^32 - 2 differ), log2(n + 1) / 32, sin / cos of n mod 2^j."""
    m = TokenNet(F, A, TOY_TAIL_TOKENS, d=16, layers=1, heads=2, ff=32)
    n = torch.tensor([0, 1, 5, (1 << 32) - 2, (1 << 32) - 1, 1 << 31])
    p = m._psi(n)
    assert p.shape == (6, 49) and p.dtype == torch.float32
    for i, v in enumerate(n.tolist()):
        assert p[i, :32].tolist() == [float((v >> k) & 1) for k in range(32)]
        assert p[i, 32] == np.float32(np.log2(v + 1.0) / 32)
        for j in range(1, 9):
            ang = 2 * np.pi * (v % (1 << j)) / (1 << j)
            assert p[i, 33 + 2 * (j - 1)] == np.float32(np.sin(ang)) and p[i, 34 + 2 * (j - 1)] == np.float32(np.cos(ang))
    assert len({tuple(r) for r in p.tolist()}) == 6


# ---------------------------------------------------------------- the exact widen
@pytest.mark.parametrize('dev', DEVS)
def test_exact_widen_adds_tails_and_a_gated_segment(dev, tmp_path):
    """The exact-widen proof (the API's 4.6): a network of the layout without tails, widened onto the toy's
    layout (a tail on items, option_pointers, the tail-only marks gated at exactly 0) computes the old network's
    outputs bit for bit - on rows without a tail (eager and static: the code of before), and on rows whose only
    new tokens are gated marks (static at the same T; eager where a chunk's T is the same) - also saved and
    loaded. Every output: policy, value, margin, the auxiliary head."""
    m = rand_net(HEAD_ONLY, 1)
    w = widen(m, layout=TOY_TAIL_TOKENS).to(dev)               # (a widen runs on the CPU, as fork_run's)
    m = m.to(dev)
    assert w.config['layout'] == tokens_layout(TOY_GATED, F, A) and w.gates() == {'marks': 0.0}
    assert torch.equal(w.tail_base[0], w.pos[4]) and w.n_tok == m.n_tok
    path = str(tmp_path / 'w.pt')
    save(w, path)
    w2, _ = load(path, device=dev, allow_mismatch=True)
    d0 = toy_batch(150, 2, p_tail=0, p_mark=0, p_gen=0)
    x, n = tx(d0, dev)
    with torch.no_grad():
        want = m(x, n, aux=True)
        for net in (w, w2):
            assert equal(want, net(x, n, aux=True, tail=no_tail(dev), ov=no_ov(dev)))
            full, ntok = static_in(m, d0, dev=dev)
            assert equal(m.forward_static(x, n, full, ntok, aux=True),
                         net.forward_static(x, n, full, ntok, aux=True, tail=no_tail(dev), ov=no_ov(dev)))
        # gated marks only (about half the rows): the static forward at the same T, bit for bit
        dm = keep_segs(toy_batch(150, 3, p_tail=0, p_mark=0.5, p_gen=0), {1})
        assert len(dm['t_seg']) > 40
        x, n = tx(dm, dev)
        T = int((token_presence(dm['x'], m.present_index()).sum(1) + np.diff(dm['t_off'])).max())
        fo, no = static_in(m, dm, T, dev, with_tail=False)
        fn, nn_ = static_in(w, dm, T, dev)
        assert equal(m.forward_static(x, n, fo, no, aux=True),
                     w.forward_static(x, n, fn, nn_, aux=True, tail=tail_of(w, dm, dev), ov=no_ov(dev)))
        # eager: one chunk (B <= 64) as long as the old network's (a row with 5 head tokens and no mark)
        small = keep_segs(toy_batch(40, 4, p_head=0.5, p_tail=0, p_mark=0.5, p_gen=0), {1})
        small['x'][0, 8::16][:4] = 1.0
        P = token_presence(small['x'], m.present_index())
        for b in np.flatnonzero(P.sum(1) + np.diff(small['t_off']) > 5):
            small = _drop_row_tail(small, b)
        assert (P.sum(1) + np.diff(small['t_off'])).max() == P.sum(1).max() == 5 and len(small['t_seg']) > 10
        x, n = tx(small, dev)
        assert equal(m(x, n, aux=True), w(x, n, aux=True, tail=tail_of(w, small, dev), ov=no_ov(dev)))
        # eager in general (chunks of other lengths): float rounding at most
        x, n = tx(dm, dev)
        assert maxdiff(m(x, n, aux=True), w(x, n, aux=True, tail=tail_of(w, dm, dev), ov=no_ov(dev))) < 1e-5


def _drop_row_tail(d, b):
    """The batch with row b's tail tokens (and overrides) removed."""
    keep = np.ones(len(d['t_seg']), bool)
    keep[d['t_off'][b]:d['t_off'][b + 1]] = False
    vo = np.concatenate([[0], np.cumsum([WIDTHS[int(s)] for s in d['t_seg']])]).astype(np.int64)
    d = dict(d)
    cnt = np.diff(d['t_off'])
    cnt[b] = 0
    vk = np.concatenate([np.arange(vo[j], vo[j + 1]) for j in np.flatnonzero(keep)]).astype(np.int64) if keep.any() \
        else np.zeros(0, np.int64)
    d.update({'t_off': np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64), 't_seg': d['t_seg'][keep],
              't_slot': d['t_slot'][keep], 't_val': d['t_val'][vk]})
    return d


@pytest.mark.parametrize('dev', DEVS)
def test_exact_widen_of_a_tail_network_with_overrides(dev):
    """A network that already has tails and overrides (TOY_TAIL_TOKENS_BASE) widened by the gated marks and the
    gated offers (a pointer column: overrides into a gated segment): on the base's rows (tail items, overrides,
    generic overrides) plus marks and offers with their overrides, the widened network computes the base's
    outputs - bit for bit in eager at the same chunk length (a segment's tokens embedded by one product of their
    own: the same as the base's), to float rounding in static (its tail embedding is one product over every tail
    token: the base's batch has fewer, so other kernel blocks) -: a gated token at g = 0 changes no other token,
    the pooling or a logit, and an override into a gated segment gives exactly the generic logit; at g = 1
    exactly the pointer logit (as the same network with the segment ungated)."""
    m = rand_net(TOY_TAIL_TOKENS_BASE, 5)
    lay = dict(TOY_TAIL_TOKENS, segments=TOY_TAIL_TOKENS['segments'] + [OFFERS_SEG])
    w = widen(m, layout=lay).to(dev)
    assert w.gates() == {'marks': 0.0, 'offers': 0.0} and w.ptr.out_features == 2 and w.ptr_parts == [(0, 1), (1, 2)]
    with torch.no_grad():
        w.ptr.weight[1].normal_(0, 1.0)                  # (the offers' column: a logit far from the generic one)
        w.ptr.bias[1] = 3.0
    m = m.to(dev)
    # eager, one chunk of the base's length: bit for bit
    d = toy_batch(60, 6, p_tail=0.5, p_mark=0.5, p_offer=0.5, p_gen=0.3)
    P = token_presence(d['x'], m.present_index())
    cap = int((P.sum(1) + np.diff(keep_segs(d, {0})['t_off'])).max())
    for r in np.flatnonzero(P.sum(1) + np.diff(d['t_off']) > cap):
        d = _drop_row_tail(d, r)
    d = _ov_into_tail(d)
    old = keep_segs(d, {0})
    assert len(d['t_seg']) > len(old['t_seg']) + 15 and int((np.diff(d['ov_off']) > np.diff(old['ov_off'])).sum()) > 5
    x, n = tx(d, dev)
    with torch.no_grad():
        want = m(x, n, aux=True, tail=tail_of(m, old, dev), ov=ov_of(m, old, dev))
        got = w(x, n, aux=True, tail=tail_of(w, d, dev), ov=ov_of(w, d, dev))
        assert equal(want, got)
    # static at the same T: to float rounding, the offers' options exactly their generic logits
    d = toy_batch(130, 6, p_tail=0.5, p_mark=0.5, p_offer=0.5, p_gen=0.3)
    old = keep_segs(d, {0})
    x, n = tx(d, dev)
    T = int((token_presence(d['x'], m.present_index()).sum(1) + np.diff(d['t_off'])).max())
    fo, no = static_in(m, old, T, dev)
    fn, nn_ = static_in(w, d, T, dev)
    tl, ov = tail_of(w, d, dev), ov_of(w, d, dev)
    isoff = (ov['ref'] >= w.n_tok) & (tl['seg'][(ov['ref'] - w.n_tok).clamp(min=0)] == 2)
    assert int(isoff.sum()) > 10
    with torch.no_grad():
        want = m.forward_static(x, n, fo, no, aux=True, tail=tail_of(m, old, dev), ov=ov_of(m, old, dev))
        got = w.forward_static(x, n, fn, nn_, aux=True, tail=tl, ov=ov)
        assert maxdiff(want, got) < 1e-5
        gen = dict(ov, ref=torch.where(isoff, -1, ov['ref']))
        assert equal(got, w.forward_static(x, n, fn, nn_, aux=True, tail=tl, ov=gen))
        # at g = 1 the pointer logit exactly: as the same network with the offers ungated
        w1 = widen(m.cpu(), layout=lay).to(dev)
        w1.load_state_dict(w.state_dict())
        w1.gate.data[1] = 1.0
        u = widen(widen(m.cpu(), layout=TOY_GATED), layout=dict(TOY_GATED, segments=TOY_GATED['segments'] + [OFFERS_SEG]),
                  ungated=True).to(dev)
        assert u.gated == [1] and u.gates() == {'marks': 0.0}
        u.load_state_dict({k: v for k, v in w.state_dict().items() if k != 'gate'} | {'gate': w.gate.data[:1].clone()})
        a = w1.forward_static(x, n, fn, nn_, tail=tl, ov=ov)
        assert equal(a, u.forward_static(x, n, fn, nn_, tail=tl, ov=ov))
        rows, acts = ov['row'][isoff], ov['act'][isoff]
        assert (got[0][rows, acts] != a[0][rows, acts]).all()


def _ov_into_tail(d):
    """(`toy_batch`'s rows keep their overrides only for tail tokens that are still there.)"""
    d = dict(d)
    o = {'ov_off': [0], 'ov_opt': [], 'ov_tok': []}
    for b in range(len(d['t_off']) - 1):
        have = {(int(s), int(sl)) for s, sl in zip(d['t_seg'][d['t_off'][b]:d['t_off'][b + 1]],
                                                    d['t_slot'][d['t_off'][b]:d['t_off'][b + 1]])}
        for i in range(d['ov_off'][b], d['ov_off'][b + 1]):
            tok = int(d['ov_tok'][i])
            if tok == -1 or tuple(int(v) for v in unpack_tok(tok)[::2]) in have:
                o['ov_opt'].append(d['ov_opt'][i])
                o['ov_tok'].append(tok)
        o['ov_off'].append(len(o['ov_opt']))
    d.update({'ov_off': np.array(o['ov_off'], np.int64), 'ov_opt': np.array(o['ov_opt'], np.int32),
              'ov_tok': np.array(o['ov_tok'], np.int64)})
    return d


def test_tails_change_outputs_only_where_present():
    """Tail tokens past the head (items at slots >= 4, ungated) change the outputs of their rows only (static
    at one T: every other row bit for bit), and each row's outputs depend on its tail (another slot, another
    float, another token count)."""
    w = rand_net(TOY_TAIL_TOKENS, 7)
    d0 = toy_batch(60, 8, p_tail=0, p_mark=0, p_gen=0)
    d1 = dict(toy_batch(60, 8, p_tail=0.4, p_mark=0, p_gen=0), x=d0['x'], n_players=d0['n_players'])
    has = np.diff(d1['t_off']) > 0
    assert 5 < has.sum() < 55
    x, n = tx(d0)
    T = 16
    f0, n0 = static_in(w, d0, T)
    f1, n1 = static_in(w, d1, T)
    with torch.no_grad():
        a = w.forward_static(x, n, f0, n0, tail=no_tail(), ov=no_ov())
        b = w.forward_static(x, n, f1, n1, tail=tail_of(w, d1), ov=ov_of(w, keep_segs(d1, {0}, gen=False)))
        for u, v in zip(a[1:], b[1:]):                              # value, margin
            assert torch.equal(u[~torch.from_numpy(has)], v[~torch.from_numpy(has)])
            assert ((u[torch.from_numpy(has)] - v[torch.from_numpy(has)]).abs().max(1).values > 1e-6).all()
        assert torch.equal(a[0][~torch.from_numpy(has)], b[0][~torch.from_numpy(has)])
        # a tail token's slot, floats and count each reach the outputs
        tl = tail_of(w, d1)
        base = w(x, n, tail=tl, ov=no_ov())[1]
        for change in ('slot', 'val', 'drop'):
            t2 = {k: v.clone() for k, v in tl.items()}
            if change == 'slot':
                t2['slot'][0] += 1
            elif change == 'val':
                t2['val'][0, 2] += 0.5
            else:
                t2 = {k: v[1:] for k, v in t2.items()}
            r = int(tl['row'][0])
            got = w(x, n, tail=t2, ov=no_ov())[1]
            assert (got[r] - base[r]).abs().max() > 1e-6


@pytest.mark.parametrize('dev', DEVS)
def test_static_forward_is_the_dynamic_one(dev):
    """forward_static with tails, gated tokens (g > 0 and g = 0) and overrides computes forward's outputs up to
    float32 rounding, whatever the padding: longer rows, padding positions holding other positions, padding
    rows, padded tail entries and padded overrides (the API's 2.6); big slots too."""
    w = rand_net(OFFERS, 9, dev)
    with torch.no_grad():
        w.gate.data[:] = torch.tensor([0.7, 0.0])
        w.ptr.weight.normal_(0, 0.5)
    for B, seed, big in ((1, 1, False), (37, 2, False), (90, 3, True)):
        d = toy_batch(B, seed, p_tail=0.6, p_mark=0.5, p_offer=0.4, big=big)
        x, n = tx(d, dev)
        tl, ov = tail_of(w, d, dev), ov_of(w, d, dev)
        with torch.no_grad():
            want = w(x, n, aux=True, tail=tl, ov=ov)
            for pad_t, pad_rows, junk, pad_nt, pad_no in ((0, 0, False, 0, 0), (5, 3, True, 7, 5)):
                full, ntok = static_in(w, d, int(static_in(w, d, dev=dev)[0].shape[1]) + pad_t, dev)
                if junk:
                    pad = torch.arange(full.shape[1], device=dev)[None] >= ntok[:, None]
                    full[pad] = torch.randint(0, w.n_tok, (int(pad.sum()),), device=dev)
                xp = torch.cat([x, x[torch.arange(pad_rows) % B]])
                npad = torch.cat([n, torch.full((pad_rows,), 2, device=dev)])
                fp = torch.cat([full, torch.zeros(pad_rows, full.shape[1], dtype=torch.long, device=dev)])
                tp = torch.cat([ntok, torch.ones(pad_rows, dtype=torch.long, device=dev)])
                t2 = {'row': torch.cat([tl['row'], torch.zeros(pad_nt, dtype=torch.long, device=dev)]),
                      'seg': torch.cat([tl['seg'], torch.zeros(pad_nt, dtype=torch.long, device=dev)]),
                      'slot': torch.cat([tl['slot'], torch.full((pad_nt,), 4, device=dev)]),
                      'val': torch.cat([tl['val'], torch.zeros(pad_nt, 16, device=dev)])}
                o2 = {'row': torch.cat([ov['row'], torch.zeros(pad_no, dtype=torch.long, device=dev)]),
                      'act': torch.cat([ov['act'], torch.full((pad_no,), A, device=dev)]),
                      'ref': torch.cat([ov['ref'], torch.full((pad_no,), -1, device=dev)]),
                      'col': torch.cat([ov['col'], torch.zeros(pad_no, dtype=torch.long, device=dev)])}
                got = w.forward_static(xp, npad, fp, tp, aux=True, tail=t2, ov=o2)
                for a, b in zip(want, got):
                    assert torch.allclose(a, b[:B], atol=2e-5), (B, pad_t, (a - b[:B]).abs().max())


# ---------------------------------------------------------------- gates: gradients and the two attention paths
def _gate_net(theta):
    m = rand_net(TOY_TAIL_TOKENS_BASE, 11)
    w = widen(m, layout=dict(TOY_TAIL_TOKENS, segments=TOY_TAIL_TOKENS['segments'] + [OFFERS_SEG])).train()
    with torch.no_grad():
        w.gate.data[:] = torch.tensor(theta)
        w.ptr.weight[1].normal_(0, 1.0)
    return w


@pytest.mark.parametrize('static', [False, True])
def test_gradients_reach_the_gates(static, monkeypatch):
    """At g = 0 (a widen) training runs the explicit attention and every gate gets a non-zero gradient - through
    the attention's keys, the pooling and the pointer blend -; so does a gate pushed below 0 (straight-through);
    with every present gate > 0 training runs SDPA with the float mask (finite gradients), the explicit path's
    outputs up to rounding; inference never runs the explicit path. Eager and static."""
    calls = []
    orig = TBlock.forward_weighted
    monkeypatch.setattr(TBlock, 'forward_weighted', lambda self, *a: calls.append(1) or orig(self, *a))
    d = toy_batch(48, 12, p_tail=0.5, p_mark=0.6, p_offer=0.5)
    x, n = tx(d)

    def run(w):
        tl, ov = tail_of(w, d), ov_of(w, d)
        if static:
            full, ntok = static_in(w, d, 32)
            return w.forward_static(x, n, full, ntok, aux=True, tail=tl, ov=ov)
        return w(x, n, aux=True, tail=tl, ov=ov)

    def grads(theta):
        w = _gate_net(theta)
        out = run(w)
        (out[0][:, :A].sum() + (out[1] ** 2).sum() + out[2].sum() + out[3].sum()).backward()
        return w, out, w.gate.grad.clone()

    w, out0, g0 = grads([0.0, 0.0])
    assert calls and (g0 != 0).all() and torch.isfinite(g0).all()
    # (at g = 0 a gated segment's own parameters get exactly no gradient - its tokens change nothing -, the
    # ungated tail segment's do)
    for k in ('pos_tail.0.weight', 'seg_in.0.weight'):
        assert w.get_parameter(k).grad.abs().sum() > 0, k
    assert w.tail_base.grad[0].abs().sum() > 0 and not w.tail_base.grad[1:].any()
    for k in ('pos_tail.1.weight', 'pos_tail.2.weight', 'seg_in.1.weight', 'seg_in.2.weight'):
        assert not w.get_parameter(k).grad.any(), k
    calls.clear()
    _, _, gneg = grads([-0.5, -0.25])                    # (g = 0 forward, the gradient straight through)
    assert calls and (gneg != 0).all()
    calls.clear()
    w, out_sdpa, gpos = grads([0.6, 0.3])
    assert not calls and (gpos != 0).all() and torch.isfinite(gpos).all()
    for k in ('tail_base', 'pos_tail.1.weight', 'pos_tail.2.weight', 'seg_in.1.weight', 'seg_in.2.weight'):
        assert w.get_parameter(k).grad.abs().sum() > 0 and torch.isfinite(w.get_parameter(k).grad).all(), k
    monkeypatch.setattr(TokenNet, '_explicit', lambda self, gi: True)
    w2 = _gate_net([0.6, 0.3])
    out_math = run(w2)
    assert maxdiff(out_sdpa, out_math) < 1e-5
    monkeypatch.undo()
    calls.clear()
    with torch.no_grad():                                # inference: SDPA's float mask, g = 0 or not
        run(_gate_net([0.0, 0.0]))
    assert not calls


def test_gates_are_logged_values():
    w = _gate_net([0.25, -1.0])
    assert w.gates() == {'marks': 0.25, 'offers': 0.0}


# ---------------------------------------------------------------- refusals (R5)
def test_forward_refuses_what_it_cannot_take():
    """R5: no tail for a network with tail segments, a tail for one without; no overrides with option_pointers,
    overrides without; a tail token of a segment without a tail, at a head slot, out of order, too narrow; an
    override to an absent head slot, a column the segment does not have, a ref past the batch's tokens."""
    w = rand_net(TOY_TAIL_TOKENS, 13)
    m = rand_net(HEAD_ONLY, 13)
    d = toy_batch(12, 14, p_head=0.5, p_tail=0.8, p_mark=0.8)
    x, n = tx(d)
    tl, ov = tail_of(w, d), ov_of(w, d)
    full, ntok = static_in(w, d)
    with pytest.raises(ValueError, match='needs `tail`'):
        w(x, n, ov=ov)
    with pytest.raises(ValueError, match='needs `ov`'):
        w(x, n, tail=tl)
    with pytest.raises(ValueError, match='without tail segments'):
        m(x, n, tail=tl)
    with pytest.raises(ValueError, match='without option_pointers'):
        m(x, n, ov=ov)
    with pytest.raises(ValueError, match='needs `tail`'):
        w.forward_static(x, n, full, ntok, ov=ov)

    def bad_tail(**chg):
        t = {k: v.clone() for k, v in tl.items()}
        for k, (i, v) in chg.items():
            t[k][i] = v
        return t
    i0 = int((tl['seg'] == 0).nonzero()[0, 0])
    for t in (bad_tail(slot=(i0, 3)),                     # an items token at a head slot
              bad_tail(seg=(0, 5)),                       # no such segment
              bad_tail(row=(0, 99)),
              {k: v.flip(0) for k, v in tl.items()},     # rows not ascending
              dict(tl, val=tl['val'][:, :4])):            # floats narrower than the widest tail segment
        with pytest.raises(ValueError):
            w(x, n, tail=t, ov=ov)
    for t in (bad_tail(slot=(i0, 3)), bad_tail(seg=(0, 5))):
        with pytest.raises(ValueError):
            w.forward_static(x, n, full, ntok, tail=t, ov=ov)
    # overrides
    P = token_presence(d['x'], w.present_index())
    br, bs = np.argwhere(~P[:, 1:5])[0]
    for o in ({'row': torch.tensor([int(br)]), 'act': torch.tensor([4]), 'ref': torch.tensor([1 + int(bs)]),
               'col': torch.tensor([0])},                                   # an absent head slot
              {'row': torch.tensor([0]), 'act': torch.tensor([4]), 'ref': torch.tensor([1]), 'col': torch.tensor([1])},
              {'row': torch.tensor([0]), 'act': torch.tensor([4]), 'ref': torch.tensor([w.n_tok + 999]),
               'col': torch.tensor([0])},
              {'row': torch.tensor([0]), 'act': torch.tensor([4]), 'ref': torch.tensor([0]), 'col': torch.tensor([0])}):
        with pytest.raises(ValueError, match='override'):
            w(x, n, tail=tl, ov=o)
        with pytest.raises(ValueError, match='override'):
            w.forward_static(x, n, full, ntok, tail=tl, ov=o)
    # a mark (segment without columns) as an override's target
    jm = int((tl['seg'] == 1).nonzero()[0, 0])
    o = {'row': tl['row'][jm:jm + 1], 'act': torch.tensor([9]), 'ref': torch.tensor([w.n_tok + jm]), 'col': torch.tensor([0])}
    with pytest.raises(ValueError, match='override'):
        w(x, n, tail=tl, ov=o)


def test_overrides_score_their_options():
    """An override's action takes its target's pointer logit (a head slot's column, a tail token's), or the
    generic logit (ref -1) even where a present head slot's static action would score it; applied after the
    static scatter; every other action as without overrides."""
    w = rand_net(TOY_TAIL_TOKENS, 15)
    d = toy_batch(20, 16, p_head=0.9, p_tail=0.6, p_mark=0.3, p_gen=0)
    x, n = tx(d)
    tl = tail_of(w, d)
    P = token_presence(d['x'], w.present_index())
    b = int(np.flatnonzero(P[:, 2])[0])                  # a row whose head slot 1 (action 5) is present
    with torch.no_grad():
        base = w(x, n, tail=tl, ov=no_ov())[0]
        gen = w.gen(_pooled(w, x, n, tl))
        # ref -1 on action 5 restores the generic logit; a head slot's column scores action 9 (no static slot)
        o = {'row': torch.tensor([b, b]), 'act': torch.tensor([5, 9]), 'ref': torch.tensor([-1, 2]), 'col': torch.tensor([0, 0])}
        got = w(x, n, tail=tl, ov=o)[0]
        assert torch.allclose(got[b, 5], gen[b, 5], atol=1e-6) and not torch.allclose(base[b, 5], gen[b, 5], atol=1e-4)
        assert torch.allclose(got[b, 9], base[b, 5], atol=1e-6)            # slot 1's pointer logit
        keep = torch.ones_like(got, dtype=torch.bool)
        keep[b, 5] = keep[b, 9] = False
        assert torch.equal(got[keep], base[keep])


def _pooled(w, x, n, tl):
    """The pooled state z of each row (the generic head's input), from a forward hook."""
    zs = []
    h = w.gen.register_forward_hook(lambda mod, inp, out: zs.append(inp[0]))
    w(x, n, tail=tl, ov=no_ov())
    h.remove()
    assert len(zs) == 1 and zs[0].shape[0] == x.shape[0]   # (one chunk, rows in order: B <= 64)
    return zs[0]


# ---------------------------------------------------------------- widen refusals and the encoding guard
def test_widen_refuses_what_is_not_exact():
    """Removing a tail or a gate, a new segment not at the end or with head slots, an ungated widen whose layout
    gates the new segment, losing option_pointers, changing a segment's columns: refused."""
    m = rand_net(TOY_GATED, 17)
    s0, s1 = TOY_GATED['segments']
    bad = [dict(TOY_GATED, segments=[{k: v for k, v in s0.items() if k != 'tail'}, s1]),
           dict(TOY_GATED, segments=[s0, {k: v for k, v in s1.items() if k != 'gate'}]),
           dict(TOY_GATED, segments=[s0, dict(OFFERS_SEG, gate=True), s1]),
           dict(TOY_GATED, segments=[s0, s1, dict(OFFERS_SEG, count=1, offset=72)]),
           dict(TOY_GATED, option_pointers=False),
           dict(TOY_GATED, segments=[dict(s0, columns=1), s1])]
    feats = [F, F, F, F + 4, F, F]
    for b, f in zip(bad, feats):
        with pytest.raises(ValueError):
            widen(m, f, A, layout=b)
    with pytest.raises(ValueError, match='ungated'):
        widen(m, layout=OFFERS, ungated=True)
    # (allowed: a draft mark, a new limit, the same layout)
    widen(m, layout=dict(TOY_GATED, draft=True, max_row_tokens=4096))


def test_ungated_widen_perturbs_the_rows_with_new_tokens():
    m = rand_net(HEAD_ONLY, 19)
    u = widen(m, layout=TOY_TAIL_TOKENS, ungated=True)
    assert u.gated == [] and 'gate' not in u.config['layout']['segments'][1]
    d = keep_segs(toy_batch(30, 20, p_tail=0, p_mark=0.5, p_gen=0), {1})
    x, n = tx(d)
    has = torch.from_numpy(np.diff(d['t_off']) > 0)
    with torch.no_grad():
        a, b = m(x, n)[1], u(x, n, tail=tail_of(u, d), ov=no_ov())[1]
    assert ((a[has] - b[has]).abs().max(1).values > 1e-6).all() and torch.allclose(a[~has], b[~has], atol=1e-6)


def test_encoding_guard_reads_the_tail_keys(monkeypatch):
    """check_encoding compares the segments' tail / gate / columns and the layout's option_pointers /
    max_row_tokens with the build's layout, not its draft mark."""
    monkeypatch.setitem(package.active().tokens, 'toy', TOY_GATED)
    info = G.info('toy')
    monkeypatch.setattr(G, 'info', lambda g: dict(info, num_features=F, num_actions=A))
    cfg = rand_net(TOY_GATED, 21).config
    assert encoding_mismatch(cfg, 'toy') is None
    monkeypatch.setitem(package.active().tokens, 'toy', dict(TOY_GATED, draft=True))
    assert encoding_mismatch(cfg, 'toy') is None
    for other, word in ((TOY_TAIL_TOKENS, 'gate'), (dict(TOY_GATED, max_row_tokens=4096), 'max_row_tokens'),
                        (HEAD_ONLY, 'segments'), (dict(TOY_GATED, segments=[dict(TOY_GATED['segments'][0], columns=1),
                                                                             TOY_GATED['segments'][1]]), 'columns')):
        monkeypatch.setitem(package.active().tokens, 'toy', other)
        diffs = encoding_mismatch(cfg, 'toy')
        assert diffs is not None and word in ' '.join(diffs[0]), (word, diffs)


# ---------------------------------------------------------------- the evaluator (eager; the graphs: test_tail_graphs)
def _answer(net, d, dev='cpu'):
    """The evaluator's answer computed by hand: the forward with the batch's tail and resolved overrides, the
    options' softmax per row, the value softmax."""
    x, n = tx(d, dev)
    with torch.no_grad():
        plog, vlog, mar = net(x, n, tail=tail_of(net, d, dev), ov=ov_of(net, d, dev))
    rows = np.repeat(np.arange(len(d['x'])), np.diff(d['opt_off']))
    lg = plog.cpu().numpy()[rows, d['opt_action']]
    pri = np.concatenate([np.exp(v - v.max()) / np.exp(v - v.max()).sum() for v in np.split(lg, d['opt_off'][1:-1])])
    return pri, torch.softmax(vlog, 1).cpu().numpy(), mar.cpu().numpy()


def test_evaluator_takes_tails_and_overrides():
    """NetEvaluator.evaluate(..., tail=rows.Tail, ov=rows.Overrides) answers as the forward on the resolved
    overrides; in pieces (max_rows) and through a LeafBatch the same; a batch with a tail for a network without
    tails, or none for one with them, is refused (R5)."""
    w = rand_net(TOY_GATED, 23)
    with torch.no_grad():
        w.gate.data[:] = 0.5
    d = toy_batch(70, 24)
    want = _answer(w, d)
    tl, ov = R.Tail.from_arrays(d, {0: 16, 1: 4}), R.Overrides.from_arrays(d)
    for ev in (NetEvaluator(w, 'cpu'), NetEvaluator(w, 'cpu', max_rows=16)):
        got = ev.evaluate(d['x'], d['n_players'], d['opt_off'], d['opt_action'], tail=tl, ov=ov)
        assert np.allclose(got[0], want[0], atol=1e-5) and np.allclose(got[1][:, :w.seats], want[1], atol=1e-5)
        assert np.allclose(got[2][:, :w.seats], want[2], atol=1e-5)
        arrays = {'x': d['x'], 'player': np.zeros(70, np.int64), 'n_players': d['n_players'], 'opt_off': d['opt_off'],
                  'opt_action': d['opt_action']}
        lb = leaf_batch(arrays, d, w)
        assert all(np.array_equal(a, b) for a, b in zip(ev(lb), got))
        assert all(np.array_equal(a, b) for a, b in zip(ev.result(ev.launch(lb)), got))
        with pytest.raises(ValueError):
            ev.evaluate(d['x'], d['n_players'], d['opt_off'], d['opt_action'], ov=ov)
    m = NetEvaluator(rand_net(HEAD_ONLY, 23), 'cpu')
    with pytest.raises(ValueError, match='R5'):
        m.evaluate(d['x'], d['n_players'], d['opt_off'], d['opt_action'], tail=tl)
    torch.manual_seed(0)
    mlp = NetEvaluator(Net(F, A, 16, 1), 'cpu')
    with pytest.raises(ValueError, match='R5'):
        mlp.evaluate(d['x'], d['n_players'], d['opt_off'], d['opt_action'], tail=tl, ov=ov)
    # the auxiliary head (no options: no overrides) takes the rows' tail
    aux = NetEvaluator(w, 'cpu').aux(d['x'], d['n_players'], tail=tl)
    assert aux['a'].shape == (70, 5)
    with pytest.raises(ValueError):
        NetEvaluator(w, 'cpu').aux(d['x'], d['n_players'])


def test_evaluator_refuses_another_network_for_a_tail_game(monkeypatch):
    """R4: a game that declares tails (game_info()['tails']) is served only by a token network with exactly
    those tail segments; an MLP, a token network without tails or with others is refused at construction."""
    info = G.info('toy')
    tails = [{'seg': 0, 'name': 'items', 'width': 16, 'head': 4}, {'seg': 1, 'name': 'marks', 'width': 4, 'head': 0}]
    monkeypatch.setattr(G, 'info', lambda g: dict(info, num_features=F, num_actions=A, tails=tails))
    NetEvaluator(rand_net(TOY_GATED, 25), 'cpu')
    torch.manual_seed(0)
    for net in (Net(F, A, 16, 1, game='toy'), rand_net(HEAD_ONLY, 25), rand_net(TOY_TAIL_TOKENS_BASE, 25),
                rand_net(OFFERS, 25)):
        with pytest.raises(ValueError, match='tail segments'):
            NetEvaluator(net, 'cpu')


# ---------------------------------------------------------------- fork_run (docs/az.md 43.8)
def _fork_run():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
    import fork_run
    return fork_run


def test_fork_run_widens_onto_a_tail_layout(tmp_path, monkeypatch):
    """fork_run <src.pt> <run> onto the game's layout with tails: the fork computes the source's outputs (rows
    without tails: bit for bit; with gated tokens: to rounding); the new segment gated when the game's layout
    gates it, else refused by the encoding guard unless --ungated; a draft layout refused unless --allow-draft
    (R16)."""
    fork_run = _fork_run()
    info = G.info('toy')
    monkeypatch.setattr(G, 'info', lambda g: dict(info, num_features=F, num_actions=A))
    m = rand_net(HEAD_ONLY, 27)
    src = str(tmp_path / 'src.pt')
    save(m, src, {'iter': 3})
    monkeypatch.setitem(package.active().tokens, 'toy', TOY_GATED)
    assert fork_run.main([src, str(tmp_path / 'a')]) == 0
    w, extra = load(str(tmp_path / 'a' / 'models' / 'iter_0000.pt'))
    assert w.gates() == {'marks': 0.0} and extra['source_extra'] == {'iter': 3}
    d = toy_batch(40, 28, p_tail=0, p_mark=0, p_gen=0)
    x, n = tx(d)
    with torch.no_grad():
        assert equal(m(x, n, aux=True), w(x, n, aux=True, tail=no_tail(), ov=no_ov()))
    # the game's layout does not gate the new segment: the gated fork is not the build's encoding
    monkeypatch.setitem(package.active().tokens, 'toy', TOY_TAIL_TOKENS)
    with pytest.raises(EncodingMismatch):
        fork_run.main([src, str(tmp_path / 'b')])
    assert not os.path.exists(str(tmp_path / 'b' / 'models' / 'iter_0000.pt'))
    assert fork_run.main([src, str(tmp_path / 'c'), '--ungated']) == 0
    assert load(str(tmp_path / 'c' / 'models' / 'iter_0000.pt'))[0].gated == []
    # a draft layout
    monkeypatch.setitem(package.active().tokens, 'toy', dict(TOY_GATED, draft=True))
    with pytest.raises(SystemExit, match='draft'):
        fork_run.main([src, str(tmp_path / 'd')])
    assert fork_run.main([src, str(tmp_path / 'e'), '--allow-draft']) == 0
