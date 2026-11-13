"""Variable-length rows on the host (docs/search.md 2.15, docs/az.md 43): a row's **tail** tokens past its
fixed-width head and its **pointer overrides**, as CSRs over rows (numpy). One container per kind, passed through
every consumer (batches, samples, shards, the window, the evaluator) instead of each growing its own arrays.

* `Tail`: `off` int64 [N+1] (row r's tokens are `[off[r], off[r+1])`), `seg` uint16 [NT] (the segment's index in
  the layout's `segments`), `slot` uint32 [NT] (>= the segment's head slots), `val` float32 or float16
  [Σ width(seg)] (each token's floats, consecutive; the offsets are the cumulative widths, never stored), `widths`
  {seg: width} of the layout's tail segments (and `heads` {seg: head slots} when known: padding needs them). A
  row's tokens are sorted by (seg, slot), without duplicates.
* `Overrides`: `off` int64 [N+1], `opt` int32 [NO] (the option's index in its row, ascending), `tok` int64 [NO]
  (`pack_tok(seg, col, slot)`, or -1: the generic logit).
* `resolve` turns overrides into the network's input (section 2.6 of the API): each override's row, action,
  extended position (head slot: its place in the full token axis; tail token: `n_tok` + its index in the batch's
  tail) and column - refusing (ValueError, R6) a pointer to an absent head slot, a missing tail token, a column
  out of range or an option out of its row.
* `token_full`: a static forward's `full` / `ntok` with each row's tail tokens after its head tokens.

Batch dicts and sample dicts name the CSR offsets `t_off` / `ov_off`; shards (format v2, zero/data.py) `t_ptr` /
`ov_ptr`. A game without tail segments has none of these keys anywhere (every path of before).
"""
import numpy as np

TAIL_KEYS = ('t_seg', 't_slot', 't_val')
OV_KEYS = ('ov_opt', 'ov_tok')


def pack_tok(seg, col, slot):
    """An override's target token: (seg << 40) | (col << 32) | slot, int64 (numpy arrays or ints)."""
    return (np.asarray(seg, np.int64) << 40) | (np.asarray(col, np.int64) << 32) | np.asarray(slot, np.int64)


def unpack_tok(tok):
    """`pack_tok`'s inverse -> (seg, col, slot) int64 arrays (`tok` >= 0)."""
    t = np.asarray(tok, np.int64)
    return (t >> 40) & 0xffff, (t >> 32) & 0xff, t & 0xffffffff


def _offsets(d, prefix):
    for k in (prefix + 'off', prefix + 'ptr'):
        if k in d:
            return np.asarray(d[k], np.int64)
    raise KeyError(f'{prefix}off / {prefix}ptr')


def _ranges(starts, lens):
    """The concatenated ranges [starts[i], starts[i] + lens[i])."""
    lens = np.asarray(lens, np.int64)
    if not len(lens) or not lens.sum():
        return np.zeros(0, np.int64)
    ends = np.cumsum(lens)
    base = np.repeat(np.asarray(starts, np.int64) - (ends - lens), lens)
    return base + np.arange(int(ends[-1]), dtype=np.int64)


def _width_heads(widths):
    """{seg: width} and {seg: head} (None if unknown) from {seg: width} or `layout_tails`' {seg: (width, count,
    k)}."""
    w, h = {}, {}
    for s, v in dict(widths).items():
        if isinstance(v, (tuple, list)):
            w[int(s)], h[int(s)] = int(v[0]), int(v[1])
        else:
            w[int(s)] = int(v)
    return w, (h if len(h) == len(w) else None)


class Tail:
    """The tail tokens of N rows (module docstring)."""
    __slots__ = ('off', 'seg', 'slot', 'val', 'widths', 'heads', '_vo')

    def __init__(self, off, seg, slot, val, widths):
        self.off = np.asarray(off, np.int64)
        self.seg = np.asarray(seg, np.uint16)
        self.slot = np.asarray(slot, np.uint32)
        self.val = np.asarray(val)
        if self.val.dtype not in (np.float32, np.float16):
            self.val = self.val.astype(np.float32)
        self.widths, self.heads = _width_heads(widths)
        self._vo = None
        if len(self.off) < 1 or self.off[0] != 0 or int(self.off[-1]) != len(self.seg) or len(self.slot) != len(self.seg):
            raise ValueError(f'Tail: offsets {self.off[:1]}..{self.off[-1:]} for {len(self.seg)} tokens / '
                             f'{len(self.slot)} slots')
        if np.any(np.diff(self.off) < 0):
            raise ValueError('Tail: offsets not ascending')
        unknown = set(np.unique(self.seg).tolist()) - set(self.widths)
        if unknown:
            raise ValueError(f'Tail: tokens of segments {sorted(unknown)}, not tail segments {sorted(self.widths)}')
        if int(self.voff()[-1]) != len(self.val):
            raise ValueError(f'Tail: {len(self.val)} floats for tokens of {int(self.voff()[-1])}')

    @property
    def N(self):
        return len(self.off) - 1

    @property
    def NT(self):
        return len(self.seg)

    def tok_width(self):
        """Each token's width (int64 [NT])."""
        lut = np.zeros(max(self.widths, default=0) + 1, np.int64)
        for s, w in self.widths.items():
            lut[s] = w
        return lut[self.seg.astype(np.int64)]

    def voff(self):
        """Each token's first float in `val` (int64 [NT + 1])."""
        if self._vo is None:
            self._vo = np.concatenate([np.zeros(1, np.int64), np.cumsum(self.tok_width())])
        return self._vo

    @classmethod
    def from_arrays(cls, d, widths, prefix='t_'):
        """From a batch / sample dict (`t_off`) or a shard (`t_ptr`), and the layout's tail widths ({seg: width}
        or `layout_tails`)."""
        return cls(_offsets(d, prefix), d[prefix + 'seg'], d[prefix + 'slot'], d[prefix + 'val'], widths)

    @classmethod
    def empty(cls, n, widths, dtype=np.float32):
        """N rows without a tail token."""
        return cls(np.zeros(int(n) + 1, np.int64), np.zeros(0, np.uint16), np.zeros(0, np.uint32), np.zeros(0, dtype),
                   widths)

    def _src(self):
        return {s: (w, self.heads[s]) for s, w in self.widths.items()} if self.heads is not None else self.widths

    def counts(self):
        """Tail tokens per row (int64 [N])."""
        return np.diff(self.off)

    def take(self, rows):
        """Rows `rows` (int array, any order, repeats allowed) as a new Tail."""
        rows = np.asarray(rows, np.int64)
        lo, hi = self.off[rows], self.off[rows + 1]
        lens = hi - lo
        toks = _ranges(lo, lens)
        vo = self.voff()
        fl = _ranges(vo[toks], vo[toks + 1] - vo[toks]) if len(toks) else np.zeros(0, np.int64)
        off = np.concatenate([np.zeros(1, np.int64), np.cumsum(lens)])
        return Tail(off, self.seg[toks], self.slot[toks], self.val[fl], self._src())

    @staticmethod
    def concat(parts):
        """The rows of `parts` (Tails of one layout) in order."""
        parts = list(parts)
        if not parts:
            raise ValueError('Tail.concat: no parts')
        w0 = parts[0].widths
        if any(p.widths != w0 for p in parts):
            raise ValueError('Tail.concat: parts of different tail segments')
        offs, base = [np.zeros(1, np.int64)], 0
        for p in parts:
            offs.append(p.off[1:] + base)
            base += int(p.off[-1])
        dt = np.result_type(*[p.val.dtype for p in parts])
        return Tail(np.concatenate(offs), np.concatenate([p.seg for p in parts]),
                    np.concatenate([p.slot for p in parts]), np.concatenate([p.val.astype(dt, copy=False) for p in parts]),
                    parts[0]._src())

    def nbytes(self):
        return int(self.off.nbytes + self.seg.nbytes + self.slot.nbytes + self.val.nbytes)

    def check(self):
        """Each row's tokens sorted by (seg, slot) without duplicates, slots past their segment's head (when the
        heads are known); ValueError otherwise."""
        if not self.NT:
            return
        row = np.repeat(np.arange(self.N), self.counts())
        key = (self.seg.astype(np.int64) << 32) | self.slot.astype(np.int64)
        same = row[1:] == row[:-1]
        if np.any(same & (key[1:] <= key[:-1])):
            raise ValueError('Tail: a row\'s tokens not sorted by (seg, slot) or duplicated')
        if self.heads is not None:
            lut = np.zeros(max(self.heads) + 1, np.int64)
            for s, h in self.heads.items():
                lut[s] = h
            if np.any(self.slot.astype(np.int64) < lut[self.seg.astype(np.int64)]):
                raise ValueError('Tail: a token inside its segment\'s head slots')

    def padded(self):
        """(row int64 [NT], seg int64 [NT], slot int64 [NT], val float32 [NT, Wt]): Wt = the widest tail segment, a
        token's floats in its first width(seg) columns, zeros after."""
        wt = max(self.widths.values(), default=0)
        row = np.repeat(np.arange(self.N, dtype=np.int64), self.counts())
        val = np.zeros((self.NT, wt), np.float32)
        if self.NT:
            mask = np.arange(wt)[None, :] < self.tok_width()[:, None]
            val[mask] = self.val.astype(np.float32, copy=False)
        return row, self.seg.astype(np.int64), self.slot.astype(np.int64), val

    def to_torch(self, device, pad_to=None):
        """The network's `tail` (API section 2.6) as tensors on `device`; padded to `pad_to` entries (row 0, the
        first tail segment, slot = its head slots, floats 0: never referenced) when given."""
        import torch
        row, seg, slot, val = self.padded()
        if pad_to is not None and pad_to > self.NT:
            if self.heads is None:
                raise ValueError('Tail.to_torch: padding needs the segments\' head slots (widths from layout_tails)')
            k = int(pad_to) - self.NT
            s0 = min(self.widths)
            row = np.concatenate([row, np.zeros(k, np.int64)])
            seg = np.concatenate([seg, np.full(k, s0, np.int64)])
            slot = np.concatenate([slot, np.full(k, self.heads[s0], np.int64)])
            val = np.concatenate([val, np.zeros((k, val.shape[1]), np.float32)])
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        return {'row': t(row), 'seg': t(seg), 'slot': t(slot), 'val': t(val)}

    def __repr__(self):
        return f'Tail(N={self.N}, NT={self.NT}, segs={sorted(self.widths)})'


class Overrides:
    """The pointer overrides of N rows (module docstring)."""
    __slots__ = ('off', 'opt', 'tok')

    def __init__(self, off, opt, tok):
        self.off = np.asarray(off, np.int64)
        self.opt = np.asarray(opt, np.int32)
        self.tok = np.asarray(tok, np.int64)
        if len(self.off) < 1 or self.off[0] != 0 or int(self.off[-1]) != len(self.opt) or len(self.tok) != len(self.opt):
            raise ValueError(f'Overrides: offsets ..{self.off[-1:]} for {len(self.opt)} options / {len(self.tok)} tokens')
        if np.any(np.diff(self.off) < 0):
            raise ValueError('Overrides: offsets not ascending')

    @property
    def N(self):
        return len(self.off) - 1

    @property
    def NO(self):
        return len(self.opt)

    @classmethod
    def from_arrays(cls, d, prefix='ov_'):
        return cls(_offsets(d, prefix), d[prefix + 'opt'], d[prefix + 'tok'])

    @classmethod
    def empty(cls, n):
        return cls(np.zeros(int(n) + 1, np.int64), np.zeros(0, np.int32), np.zeros(0, np.int64))

    def counts(self):
        return np.diff(self.off)

    def take(self, rows):
        rows = np.asarray(rows, np.int64)
        lo, hi = self.off[rows], self.off[rows + 1]
        pos = _ranges(lo, hi - lo)
        return Overrides(np.concatenate([np.zeros(1, np.int64), np.cumsum(hi - lo)]), self.opt[pos], self.tok[pos])

    @staticmethod
    def concat(parts):
        parts = list(parts)
        if not parts:
            raise ValueError('Overrides.concat: no parts')
        offs, base = [np.zeros(1, np.int64)], 0
        for p in parts:
            offs.append(p.off[1:] + base)
            base += int(p.off[-1])
        return Overrides(np.concatenate(offs), np.concatenate([p.opt for p in parts]), np.concatenate([p.tok for p in parts]))

    def nbytes(self):
        return int(self.off.nbytes + self.opt.nbytes + self.tok.nbytes)

    def __repr__(self):
        return f'Overrides(N={self.N}, NO={self.NO})'


def layout_tails(lay):
    """{seg index: (width, count, k)} of a layout's tail segments (`tail: true`; k = the order among them)."""
    out = {}
    for i, s in enumerate(lay.get('segments', [])):
        if s.get('tail'):
            out[i] = (int(s['width']), int(s.get('count') or 0), len(out))
    return out


def tail_segs(lay):
    """A layout's tail segments as shard meta `t_segs`: [[seg, name, width, count], ...] in layout order."""
    return [[i, str(s['name']), int(s['width']), int(s.get('count') or 0)]
            for i, s in enumerate(lay.get('segments', [])) if s.get('tail')]


def info_segs(info):
    """A game's tail segments (`games.info(game)['tails']`) as shard meta `t_segs`."""
    return [[int(t['seg']), str(t['name']), int(t['width']), int(t['head'])] for t in info.get('tails') or []]


def head_tokens(lay):
    """A layout's head tokens: 1 + Σ count over its segments (the global token and every head slot)."""
    return 1 + sum(int(s.get('count') or 0) for s in lay.get('segments', []))


def seg_columns(lay):
    """Each segment's pointer columns: its `actions` pairs, + 1 with `id_actions`, + its `columns`."""
    return [len(s.get('actions') or []) + (s.get('id_actions') is not None) + int(s.get('columns') or 0)
            for s in lay.get('segments', [])]


def present_index(lay):
    """Each head slot's present flag's feature in the full token axis order (-1: always present) - TokenNet's
    `present_index()` of the layout (head slots only)."""
    out = []
    for s in lay.get('segments', []):
        c = int(s.get('count') or 0)
        if not c:
            continue
        for slot in range(c):
            out.append(-1 if s.get('present') is None else int(s['offset']) + slot * int(s['width']) + int(s['present']))
    return np.asarray(out, np.int64)


def presence(x, lay):
    """Rows' head tokens present: bool [N, 1 + head slots] from dense features x [N, F] (`infer.token_presence`)."""
    pidx = present_index(lay)
    P = np.ones((x.shape[0], 1 + len(pidx)), bool)
    has = pidx >= 0
    P[:, 1:][:, has] = np.asarray(x)[:, pidx[has]] > 0.5
    return P


def resolve(tail, ov, opt_off, opt_action, lay, present=None):
    """Overrides as the network's `ov` (API section 2.6, numpy): {'row', 'act', 'ref', 'col'} int64 [NO] - the
    override's row, its option's action, the extended position of its target (a head slot's place in the full
    token axis, `1 + Σ_{s' < s} count(s') + slot`; a tail token's `n_tok` + its index in `tail` (`padded()` order);
    -1: the generic logit) and the pointer column. `present` (bool [N, n_tok], `presence`; None: not checked): the
    rows' head tokens. ValueError (R6) for an option out of its row, a column out of its segment's columns, a
    target head slot not present, a missing tail token, a token of no segment."""
    N = len(opt_off) - 1
    if ov.N != N or (tail is not None and tail.N != N):
        raise ValueError(f'resolve: {ov.N} override rows / {None if tail is None else tail.N} tail rows for {N} rows')
    if not ov.NO:
        # (no override: nothing to resolve or check - the arrays below, empty; most batches, docs/az.md 43.12)
        return {k: np.zeros(0, np.int64) for k in ('row', 'act', 'ref', 'col')}
    opt_off = np.asarray(opt_off, np.int64)
    opt_action = np.asarray(opt_action, np.int64)
    segs = lay.get('segments', [])
    counts = np.asarray([int(s.get('count') or 0) for s in segs], np.int64)
    first = np.concatenate([np.ones(1, np.int64), 1 + np.cumsum(counts)])[:len(segs)]
    n_tok = 1 + int(counts.sum())
    ncols = np.asarray(seg_columns(lay), np.int64)
    row = np.repeat(np.arange(N, dtype=np.int64), ov.counts())
    opt = ov.opt.astype(np.int64)
    nopt = opt_off[row + 1] - opt_off[row]
    if np.any((opt < 0) | (opt >= nopt)):
        k = int(np.flatnonzero((opt < 0) | (opt >= nopt))[0])
        raise ValueError(f'resolve: row {row[k]}: override of option {opt[k]}, the row has {nopt[k]} options')
    act = opt_action[opt_off[row] + opt]
    tok = ov.tok
    ref = np.full(len(tok), -1, np.int64)
    col = np.zeros(len(tok), np.int64)
    pt = np.flatnonzero(tok >= 0)
    if np.any(tok < -1):
        raise ValueError(f'resolve: token {int(tok[tok < -1][0])} is neither -1 nor a packed token')
    if len(pt):
        s, c, l = unpack_tok(tok[pt])
        if np.any(s >= len(segs)):
            raise ValueError(f'resolve: a token of segment {int(s.max())}; the layout has {len(segs)}')
        if np.any(c >= ncols[s]):
            k = int(np.flatnonzero(c >= ncols[s])[0])
            raise ValueError(f'resolve: column {int(c[k])} of segment {segs[s[k]]["name"]}, which has {int(ncols[s[k]])}')
        col[pt] = c
        head = l < counts[s]
        h = pt[head]
        if len(h):
            pos = first[s[head]] + l[head]
            if present is not None and not np.all(np.asarray(present)[row[h], pos]):
                k = int(np.flatnonzero(~np.asarray(present)[row[h], pos])[0])
                raise ValueError(f'resolve: row {int(row[h][k])}: a pointer to head slot {int(l[head][k])} of '
                                 f'{segs[s[head][k]]["name"]}, which is not present')
            ref[h] = pos
        t = pt[~head]
        if len(t):
            if tail is None:
                raise ValueError('resolve: a pointer to a tail token without a tail')
            kt = np.zeros(tail.NT, dtype=[('r', '<i8'), ('k', '<i8')])
            kt['r'] = np.repeat(np.arange(N, dtype=np.int64), tail.counts())
            kt['k'] = (tail.seg.astype(np.int64) << 32) | tail.slot.astype(np.int64)
            q = np.zeros(len(t), dtype=kt.dtype)
            q['r'] = row[t]
            q['k'] = (s[~head] << 32) | l[~head]
            j = np.searchsorted(kt, q)
            ok = (j < tail.NT)
            ok[ok] = kt[j[ok]] == q[ok]
            if not ok.all():
                k = int(np.flatnonzero(~ok)[0])
                raise ValueError(f'resolve: row {int(q["r"][k])}: a pointer to tail token ({segs[s[~head][k]]["name"]}, '
                                 f'slot {int(l[~head][k])}), which the row does not have')
            ref[t] = n_tok + j
    return {'row': row, 'act': act, 'ref': ref, 'col': col}


def ov_torch(r, device):
    """`resolve`'s dict as tensors on `device`."""
    import torch
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in r.items()}


def token_full(P_head, tail_counts, n_tok, t_step):
    """A static forward's (full int64 [B, T], ntok int64 [B]): `infer.token_index`'s layout of the head tokens
    (`P_head` bool [B, n_tok]: their places in the full token axis), then each row's tail tokens at `n_tok` + j (j:
    the token's index in the batch's tail), T the longest row's tokens rounded up to `t_step`."""
    P = np.asarray(P_head, bool)
    B = P.shape[0]
    tc = np.asarray(tail_counts, np.int64)
    nh = P.sum(1).astype(np.int64)
    ntok = nh + tc
    T = -(-int(ntok.max(initial=1)) // t_step) * t_step
    full = np.zeros((B, T), np.int64)
    r, c = np.nonzero(P)
    full[r, (np.cumsum(P, 1) - 1)[r, c]] = c
    if tc.sum():
        tr = np.repeat(np.arange(B), tc)
        j = np.arange(int(tc.sum()), dtype=np.int64)
        first = np.concatenate([np.zeros(1, np.int64), np.cumsum(tc)])[:-1]
        full[tr, nh[tr] + (j - first[tr])] = n_tok + j
    return full, ntok


def forward_kw(b):
    """A training batch's variable-length row inputs for the network (`Window.batch`): {'tail', 'ov'} as made, {} for
    a batch without (the forward's call of before)."""
    return {k: b[k] for k in ('tail', 'ov') if k in b}


def eval_kw(tail, ov):
    """`NetEvaluator.evaluate`'s variable-length row arguments: {'tail', 'ov'} for the ones given, {} without (the
    call of before)."""
    return {k: v for k, v in (('tail', tail), ('ov', ov)) if v is not None}


def sample_rows(s, widths, a=0, b=None):
    """A sample dict's tails and overrides of rows [a, b) -> (Tail or None, Overrides or None)."""
    N = len(s['player'])
    b = N if b is None else b
    rr = np.arange(a, b)
    tail = Tail.from_arrays(s, widths).take(rr) if 't_off' in s else None
    ov = Overrides.from_arrays(s).take(rr) if 'ov_off' in s else None
    return tail, ov


def shard_layout(meta):
    """A layout that names a v2 shard's tail segments and overrides only (from its meta `t_segs` /
    `option_pointers`): enough for `data.Window` to load them, not to resolve overrides (the other segments' head
    slots are unknown) - `Window.use_layout(network layout)` before `batch`. None for a shard without them."""
    t = (meta or {}).get('t_segs')
    if not t and not (meta or {}).get('option_pointers'):
        return None
    n = 1 + max([int(s) for s, *_ in t], default=-1)
    segs = [{'name': f'?{i}', 'count': 0, 'width': 1, 'placeholder': True} for i in range(n)]
    for s, name, w, h in t:
        segs[int(s)] = {'name': name, 'count': int(h), 'width': int(w), 'tail': True}
    out = {'segments': segs, 'placeholder': True}
    if (meta or {}).get('option_pointers'):
        out['option_pointers'] = True
    return out


def game_row(g, viewer, widths=None, pointers=False):
    """A game's (`<pkg>._engine.Game`) row of `viewer` for an evaluator: (x f32 [1, F], Tail of 1 row or None,
    Overrides of 1 row or None) - `encode_row` and, with `pointers`, `option_overrides` against `legal()` (a game
    without tail segments: `encode`, the call of before)."""
    if not widths and not pointers:
        return np.asarray(g.encode(viewer), np.float32)[None], None, None
    x, t = g.encode_row(viewer)
    tail = None
    if widths:
        n = len(t['t_seg'])
        tail = Tail(np.array([0, n], np.int64), t['t_seg'], t['t_slot'], t['t_val'], widths)
    ov = None
    if pointers:
        o, k = g.option_overrides(viewer)
        ov = Overrides(np.array([0, len(o)], np.int64), o, k)
    return np.asarray(x, np.float32)[None], tail, ov
