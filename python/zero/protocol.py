"""The leaf-batch protocol between a driver (Rust `<package>._engine.SearchDriver` or the Python
reference `play.PyDriver`) and the inference side (docs/search.md 2).

A network batch ('nn') holds B leaves:

| field | dtype / shape | content |
|---|---|---|
| `x` | f32 [B, F] | features of the leaf state from the decider's view (`SearchGame::encode`) |
| `opt_off` | i32 [B+1] | leaf b's legal actions are `opt_action[opt_off[b]:opt_off[b+1]]` (>= 2; none for a deferred terminal's leaf) |
| `opt_action` | i32 [sum] | global action ids, in the game's legal order |
| `player` | i32 [B] | deciding seat (absolute) |
| `n_players` | i32 [B] | players in that game |
| `gid` | i64 [B] | game index |
| `extra` | u8 [B] | the row's kind: `ROW_LEAF` (0) a leaf's own row, else an extra row of the closest leaf row before it (same game): `ROW_VALUE` (1) = its tree viewer's value-only row, no options (root-view leaf values, docs/search.md 3.6); `ROW_BELIEF` (2) = a search's root belief row, no options (the root's features for its decider: the driver reads only its auxiliary output; after the fresh root's own leaf row, or its game's only row; docs/search.md 3.8) |

A deferred terminal (`SearchGame::deferred`, docs/search.md 3.2) is a leaf with an empty option
segment: the evaluator returns its value and margin and no priors. So is a `ROW_VALUE` row; an
evaluator answers every row alike (the driver reads what it needs from each), so it can ignore
`extra`.

The answer: `priors` f32 [sum] (softmax over each leaf's legal actions), `value` f32 [B, 8]
(P(win) per seat RELATIVE to the decider: column j = seat (player + j) % n_players; columns
>= n_players are 0), `margin` f32 [B, 8] (same seat order) - and, only when the batch has
`ROW_BELIEF` rows, `aux` f32 [belief rows, aux width]: the network's auxiliary predictions
(`net.aux_predict`, the segments in the game's order) of those rows, in row order. An evaluator
that cannot (a network without the auxiliary head) must fail; the driver refuses an answer
without them (docs/search.md 3.8).

A sparse network batch (a driver created with `sparse=True`, docs/search.md 2.2) carries each row's
non-zero features instead of `x`: `xs_off` i64 [B+1], `xs_idx` u32 [nnz] (feature indexes, ascending
per row), `xs_val` f32 [nnz], `n_features` (F). `LeafBatch.x` densifies it on first use (an evaluator
that reads `x` sees the same rows); `NetEvaluator`'s token CUDA graphs scatter it on the device.

Variable-length rows (a game with tail segments / pointer overrides, docs/search.md 2.15): a network batch
also carries `t_off` i64 [B+1], `t_seg` u16 [NT], `t_slot` u32 [NT], `t_val` f32 [Σ width(t_seg)] (each row's
tail tokens, sorted by (seg, slot), slot >= the segment's head slots) and, with pointer overrides, `ov_off` i64
[B+1], `ov_opt` i32 [NO] (the option's index in its row, ascending), `ov_tok` i64 [NO] (`rows.pack_tok`, -1 = the
generic logit). `LeafBatch.tail` / `.ov` are them as `rows.Tail` / `rows.Overrides` (None without); `LeafBatch.x`
stays the head (densification never touches the tail). Sample dicts carry the same keys (N rows). A game without
them has none of these keys.

An external batch ('ext') holds B requests: `states` (list of str: the game's
`export_state()`), `player`, `n_players`, `gid`; the answer is one payload string per request
(applied by `SearchGame::apply_external`).
"""
import numpy as np

MAX_SEATS = 8
ROW_LEAF, ROW_VALUE = 0, 1                 # `extra`: a leaf's own row / its tree viewer's value-only row
ROW_BELIEF = 2                             # ... / a search's root belief row (docs/search.md 3.8)


def belief_rows(extra):
    """The indexes of a batch's root belief rows (`extra` == ROW_BELIEF; none for None)."""
    if extra is None:
        return np.zeros(0, np.int64)
    return np.flatnonzero(np.asarray(extra) == ROW_BELIEF)
SAMPLE_KEYS = ('x', 'player', 'n_players', 'opt_off', 'opt_action', 'pi', 'z', 'q', 'sc', 'net')
# in some sample dicts only: 'aux' f32 [N, W] = the deciders' auxiliary targets (a driver created with
# aux=True: `SearchGame::aux_target`, docs/search.md 1.2, docs/az.md 15); per sample row, like `x`
OPTIONAL_SAMPLE_KEYS = ('aux',)
ALL_SAMPLE_KEYS = SAMPLE_KEYS + OPTIONAL_SAMPLE_KEYS
# variable-length rows (docs/search.md 2.15): CSRs over the rows - never sliced by row index like the keys above
# (`take_rows`, `concat_samples` re-offset them); present together, in a game with tails / pointer overrides only
TAIL_SAMPLE_KEYS = ('t_off', 't_seg', 't_slot', 't_val')
OV_SAMPLE_KEYS = ('ov_off', 'ov_opt', 'ov_tok')
ROW_SAMPLE_KEYS = TAIL_SAMPLE_KEYS + OV_SAMPLE_KEYS


def has_rows(s):
    """Whether a batch / sample dict carries variable-length rows (any tail or override key)."""
    return s is not None and any(k in s for k in ROW_SAMPLE_KEYS)


class LeafBatch:
    __slots__ = ('id', 'key', 'kind', 'a', 'tails')

    def __init__(self, bid, key, kind, arrays, tails=None):
        self.id = bid
        self.key = key
        self.kind = kind
        self.a = arrays
        # the game's tail segments {seg: (width, head, k)} (`rows.layout_tails` of `games.Driver`; None: none)
        self.tails = tails

    @property
    def tail(self):
        """The rows' tail tokens (`rows.Tail`), None for a batch without them."""
        if 't_off' not in self.a:
            return None
        if not self.tails:
            raise ValueError('LeafBatch.tail: a batch with tail tokens but no tail segments (LeafBatch(tails=...))')
        from .rows import Tail
        return Tail.from_arrays(self.a, self.tails)

    @property
    def ov(self):
        """The rows' pointer overrides (`rows.Overrides`), None for a batch without them."""
        if 'ov_off' not in self.a:
            return None
        from .rows import Overrides
        return Overrides.from_arrays(self.a)

    @property
    def external(self):
        return self.kind == 'ext'

    @property
    def B(self):
        return len(self.a['player'])

    @property
    def sparse(self):
        """A sparse network batch (its rows' non-zeros: `xs_off`, `xs_idx`, `xs_val`)."""
        return 'xs_val' in self.a

    def __getattr__(self, k):
        if k == 'tails':                       # (an unpickled batch of before: no tail segments)
            return None
        try:
            return self.a[k]
        except KeyError:
            if k == 'x' and 'xs_val' in self.a:
                x = self.a['x'] = densify(self.a['xs_off'], self.a['xs_idx'], self.a['xs_val'], self.a['n_features'])
                return x
            raise AttributeError(k) from None

    def __repr__(self):
        return f'LeafBatch(id={self.id}, key={self.key!r}, kind={self.kind}, B={self.B})'


def densify(xs_off, xs_idx, xs_val, n_features):
    """A sparse batch's rows as f32 [B, F] (zeros but at the non-zeros)."""
    xs_off = np.asarray(xs_off)
    B = len(xs_off) - 1
    x = np.zeros((B, int(n_features)), np.float32)
    rows = np.repeat(np.arange(B), np.diff(xs_off))
    x[rows, np.asarray(xs_idx, np.int64)] = xs_val
    return x


def validate_batch(lb, num_features=None, num_actions=None):
    """Check a network batch against the protocol (tests)."""
    a = lb.a
    B = lb.B
    assert a['player'].dtype == np.int32 and a['n_players'].dtype == np.int32 and a['gid'].dtype == np.int64
    if lb.external:
        assert len(a['states']) == B and all(isinstance(s, str) for s in a['states'])
        return
    x, off, act = a['x'], a['opt_off'], a['opt_action']
    assert x.dtype == np.float32 and x.ndim == 2 and x.shape[0] == B and x.flags.c_contiguous
    if num_features is not None:
        assert x.shape[1] == num_features
    assert off.dtype == np.int32 and off.shape == (B + 1,) and off[0] == 0 and off[-1] == len(act)
    cnt = np.diff(off)
    assert np.all((cnt >= 2) | (cnt == 0)), 'single-option decisions never reach a batch (0 options: a deferred terminal)'
    assert act.dtype == np.int32 and (len(act) == 0 or act.min() >= 0)
    if num_actions is not None and len(act):
        assert act.max() < num_actions
    for b in range(B):
        seg = act[off[b]:off[b + 1]]
        assert len(set(seg.tolist())) == len(seg), 'duplicate legal action'
    assert np.all((a['player'] >= 0) & (a['player'] < a['n_players']))
    ex = a.get('extra')
    if ex is None:                          # (a batch made by hand: every row a leaf's)
        return
    assert ex.dtype == np.uint8 and ex.shape == (B,)
    for b in np.flatnonzero(ex):
        if ex[b] == ROW_BELIEF:
            # (after its game's root leaf row of the same decider, or its game's only row)
            assert cnt[b] == 0, 'a belief row: no options'
            after = b > 0 and ex[b - 1] == ROW_LEAF and a['gid'][b - 1] == a['gid'][b] and a['player'][b - 1] == a['player'][b]
            alone = int((a['gid'] == a['gid'][b]).sum()) == 1
            assert after or alone, 'a belief row neither after its root leaf row nor alone'
            continue
        lead = b - 1
        while lead >= 0 and ex[lead]:
            lead -= 1
        assert lead >= 0 and a['gid'][lead] == a['gid'][b], 'an extra row without its leaf row before it'
        if ex[b] == ROW_VALUE:
            assert cnt[b] == 0 and a['player'][b] != a['player'][lead], 'a value-only row: no options, the viewer'
    validate_rows(a, B, cnt, lb.tails)


def validate_rows(a, B, nopt, tails=None):
    """A batch / sample dict's variable-length rows (docs/search.md 2.15): the tail CSR's offsets, dtypes, order
    and widths, slots past their segment's head (`tails` {seg: (width, head, k)}: required with a tail), the
    overrides' offsets, order and range (`nopt` [B]: the rows' option counts). Nothing without them."""
    if 't_off' in a or any(k in a for k in TAIL_SAMPLE_KEYS):
        assert all(k in a for k in TAIL_SAMPLE_KEYS), 'a tail without all its arrays'
        off = a['t_off']
        assert off.dtype == np.int64 and off.shape == (B + 1,) and off[0] == 0 and np.all(np.diff(off) >= 0)
        assert a['t_seg'].dtype == np.uint16 and a['t_slot'].dtype == np.uint32 and a['t_val'].dtype == np.float32
        assert int(off[-1]) == len(a['t_seg']) == len(a['t_slot'])
        assert tails, 'a tail without its tail segments'
        from .rows import Tail
        t = Tail.from_arrays(a, tails)          # (widths: every token's segment a tail segment, the floats' count)
        t.check()                               # (sorted (seg, slot), no duplicate, slot >= head)
    if 'ov_off' in a or any(k in a for k in OV_SAMPLE_KEYS):
        assert all(k in a for k in OV_SAMPLE_KEYS), 'overrides without all their arrays'
        off = a['ov_off']
        assert off.dtype == np.int64 and off.shape == (B + 1,) and off[0] == 0 and np.all(np.diff(off) >= 0)
        assert a['ov_opt'].dtype == np.int32 and a['ov_tok'].dtype == np.int64 and int(off[-1]) == len(a['ov_opt'])
        opt, tok = a['ov_opt'].astype(np.int64), a['ov_tok']
        row = np.repeat(np.arange(B), np.diff(off))
        assert np.all((opt >= 0) & (opt < np.asarray(nopt)[row])), 'an override of an option out of its row'
        assert np.all(row[1:] != row[:-1]) or np.all(opt[1:][row[1:] == row[:-1]] > opt[:-1][row[1:] == row[:-1]]), \
            'a row\'s overrides not ascending'
        assert np.all(tok >= -1), 'an override token neither -1 nor a packed token'



def validate_evaluation(lb, priors, value, margin, aux=None, aux_width=None):
    off = lb.opt_off
    nb = len(belief_rows(lb.a.get('extra')))
    if nb:
        assert aux is not None and aux.dtype == np.float32 and aux.ndim == 2 and aux.shape[0] == nb
        if aux_width is not None:
            assert aux.shape[1] == aux_width
        assert np.all(np.isfinite(aux)) and np.all(aux >= 0)
    else:
        assert aux is None or len(aux) == 0
    assert priors.dtype == np.float32 and priors.shape == (off[-1],)
    assert value.dtype == np.float32 and value.shape == (lb.B, MAX_SEATS)
    assert margin.dtype == np.float32 and margin.shape == (lb.B, MAX_SEATS)
    for b in range(lb.B):
        assert off[b + 1] == off[b] or abs(priors[off[b]:off[b + 1]].sum() - 1) < 1e-4
        n = lb.n_players[b]
        assert abs(value[b, :n].sum() - 1) < 1e-4 and np.all(value[b, n:] == 0)


def validate_samples(s, num_features=None, tails=None):
    N = len(s['player'])
    assert s['x'].shape[0] == N and s['opt_off'].shape == (N + 1,)
    if num_features is not None:
        assert s['x'].shape[1] == num_features
    assert len(s['pi']) == len(s['opt_action']) == s['opt_off'][-1]
    for k in ('z', 'q', 'sc'):
        assert s[k].shape == (N, MAX_SEATS)
    if 'aux' in s:
        assert s['aux'].ndim == 2 and s['aux'].shape[0] == N
    validate_rows(s, N, np.diff(np.asarray(s['opt_off'], np.int64)), tails)
    off = s['opt_off']
    for b in range(N):
        # (a value-only sample has no options and no policy target: docs/search.md 3.6)
        assert off[b + 1] == off[b] or abs(s['pi'][off[b]:off[b + 1]].sum() - 1) < 1e-3
        n = s['n_players'][b]
        assert abs(s['z'][b, :n].sum() - 1) < 1e-4


def concat_samples(parts):
    """Concatenate sample dicts (re-offsetting opt_off; the tail and override CSRs re-offset too: in all parts or
    none, else ValueError). None for no parts."""
    parts = [p for p in parts if p is not None and len(p['player'])]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    out = {}
    for keys in (TAIL_SAMPLE_KEYS, OV_SAMPLE_KEYS):
        have = [all(k in p for k in keys) for p in parts]
        if any(any(k in p for k in keys) for p in parts) and not all(have):
            raise ValueError(f'concat_samples: {keys} in some parts only')
        if all(have):
            out.update(_concat_csr(parts, keys))
    for k in ALL_SAMPLE_KEYS:
        if k == 'opt_off':
            offs, base = [np.zeros(1, np.int64)], 0
            for p in parts:
                offs.append(p['opt_off'][1:].astype(np.int64) + base)
                base += int(p['opt_off'][-1])
            out[k] = np.concatenate(offs)
        elif k in parts[0]:
            if k in OPTIONAL_SAMPLE_KEYS and any(k not in p for p in parts):
                raise ValueError(f'concat_samples: {k!r} in some parts only')
            out[k] = np.concatenate([p[k] for p in parts])
    return out


def _concat_csr(parts, keys):
    """The CSR `keys` (offsets first) of sample dicts, concatenated (offsets re-based)."""
    offs, base = [np.zeros(1, np.int64)], 0
    for p in parts:
        offs.append(np.asarray(p[keys[0]][1:], np.int64) + base)
        base += int(p[keys[0]][-1])
    return {keys[0]: np.concatenate(offs), **{k: np.concatenate([p[k] for p in parts]) for k in keys[1:]}}


def take_rows(s, a, b, widths=None):
    """Rows [a, b) of a sample dict (every key: per-row arrays, the options' arrays by `opt_off`, the tail and
    override CSRs by their offsets; the tail's floats by the tokens' widths `widths` ({seg: width} or
    `rows.layout_tails`; required with a tail, ValueError otherwise)) - a part a consumer may handle as a whole
    sample dict."""
    off = np.asarray(s['opt_off'], np.int64)
    lo, hi = int(off[a]), int(off[b])
    out = {}
    for k, v in s.items():
        if k == 'opt_off':
            out[k] = (off[a:b + 1] - lo).astype(np.asarray(v).dtype)
        elif k in ('opt_action', 'pi'):
            out[k] = v[lo:hi]
        elif k not in ROW_SAMPLE_KEYS:
            out[k] = v[a:b]
    if 'ov_off' in s:
        o = np.asarray(s['ov_off'], np.int64)
        out['ov_off'] = o[a:b + 1] - o[a]
        out['ov_opt'], out['ov_tok'] = s['ov_opt'][o[a]:o[b]], s['ov_tok'][o[a]:o[b]]
    if 't_off' in s:
        if widths is None:
            raise ValueError('take_rows: a sample dict with a tail needs its tail widths')
        from .rows import Tail
        t = Tail.from_arrays(s, widths).take(np.arange(a, b))
        out.update(t_off=t.off, t_seg=t.seg, t_slot=t.slot, t_val=t.val)
    return out


def segment_softmax(logits, off):
    """Softmax of a flat array over the segments of `off` (numpy reference)."""
    out = np.empty_like(logits, dtype=np.float32)
    for b in range(len(off) - 1):
        seg = logits[off[b]:off[b + 1]].astype(np.float64)
        if not len(seg):                           # a deferred terminal's leaf: no options
            continue
        e = np.exp(seg - seg.max())
        out[off[b]:off[b + 1]] = e / e.sum()
    return out
