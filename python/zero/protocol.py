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

A deferred terminal (`SearchGame::deferred`, docs/search.md 3.2) is a leaf with an empty option
segment: the evaluator returns its value and margin and no priors.

The answer: `priors` f32 [sum] (softmax over each leaf's legal actions), `value` f32 [B, 8]
(P(win) per seat RELATIVE to the decider: column j = seat (player + j) % n_players; columns
>= n_players are 0), `margin` f32 [B, 8] (same seat order).

An external batch ('ext') holds B requests: `states` (list of str: the game's
`export_state()`), `player`, `n_players`, `gid`; the answer is one payload string per request
(applied by `SearchGame::apply_external`).
"""
import numpy as np

MAX_SEATS = 8
SAMPLE_KEYS = ('x', 'player', 'n_players', 'opt_off', 'opt_action', 'pi', 'z', 'q', 'sc', 'net')
# in some sample dicts only: 'aux' f32 [N, W] = the deciders' auxiliary targets (a driver created with
# aux=True: `SearchGame::aux_target`, docs/search.md 1.2, docs/az.md 15); per sample row, like `x`
OPTIONAL_SAMPLE_KEYS = ('aux',)
ALL_SAMPLE_KEYS = SAMPLE_KEYS + OPTIONAL_SAMPLE_KEYS


class LeafBatch:
    __slots__ = ('id', 'key', 'kind', 'a')

    def __init__(self, bid, key, kind, arrays):
        self.id = bid
        self.key = key
        self.kind = kind
        self.a = arrays

    @property
    def external(self):
        return self.kind == 'ext'

    @property
    def B(self):
        return len(self.a['player'])

    def __getattr__(self, k):
        try:
            return self.a[k]
        except KeyError:
            raise AttributeError(k) from None

    def __repr__(self):
        return f'LeafBatch(id={self.id}, key={self.key!r}, kind={self.kind}, B={self.B})'


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


def validate_evaluation(lb, priors, value, margin):
    off = lb.opt_off
    assert priors.dtype == np.float32 and priors.shape == (off[-1],)
    assert value.dtype == np.float32 and value.shape == (lb.B, MAX_SEATS)
    assert margin.dtype == np.float32 and margin.shape == (lb.B, MAX_SEATS)
    for b in range(lb.B):
        assert off[b + 1] == off[b] or abs(priors[off[b]:off[b + 1]].sum() - 1) < 1e-4
        n = lb.n_players[b]
        assert abs(value[b, :n].sum() - 1) < 1e-4 and np.all(value[b, n:] == 0)


def validate_samples(s, num_features=None):
    N = len(s['player'])
    assert s['x'].shape[0] == N and s['opt_off'].shape == (N + 1,)
    if num_features is not None:
        assert s['x'].shape[1] == num_features
    assert len(s['pi']) == len(s['opt_action']) == s['opt_off'][-1]
    for k in ('z', 'q', 'sc'):
        assert s[k].shape == (N, MAX_SEATS)
    if 'aux' in s:
        assert s['aux'].ndim == 2 and s['aux'].shape[0] == N
    off = s['opt_off']
    for b in range(N):
        # (a value-only sample has no options and no policy target: docs/search.md 3.6)
        assert off[b + 1] == off[b] or abs(s['pi'][off[b]:off[b + 1]].sum() - 1) < 1e-3
        n = s['n_players'][b]
        assert abs(s['z'][b, :n].sum() - 1) < 1e-4


def concat_samples(parts):
    """Concatenate sample dicts (re-offsetting opt_off). None for no parts."""
    parts = [p for p in parts if p is not None and len(p['player'])]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    out = {}
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
