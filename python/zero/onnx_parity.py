"""Parity gate 1 of zero-onnx/1 and /2 (docs/az.md 41, 46): a network's exported ONNX graph, run by ONNX Runtime's
CPU execution provider, against torch on real network batches - new code is right only if it matches. Rows with
tail tokens and pointer overrides (zero-onnx/2: `Rows.tail` / `.ov`, recorded batch files and v2 shards that carry
them) go to the graph and to torch with them, a case of their own (`varlen`), and a graph that refuses any of them
fails the case (`Refused`).

  python -m zero.onnx_parity <checkpoint> --batches <file.npz> [<file.npz> ...] [--packing mask,compact,packed,host]
      [--sizes 1,7,64,256,1024] [--tol 2e-5] [--onnx-dir DIR] [--reuse] [--game G] [--allow-mismatch]
      [--threads 4] [--json out.json]

Batch files: recorded driver batches (tools/onnx_record.py: the driver's sparse rows with their kinds:
leaf, value-only, root belief) or sample shards (`data.read_shard`; a shard row with no options is a
value-only row, and every 8th leaf row is followed by a root belief copy of it, as the driver places
them). Cases per file: each batch size of `--sizes` (consecutive rows, wrapping around), as they are
and with NaN in every absent slot's floats (the present flag aside: a token network never reads them);
a packing that takes the host's token positions ('host') also gets junk in its padding positions;
and the file's first recorded batches as the driver sent them. A case of more than 64 rows is also
answered in pieces of 64 rows grouped by their non-zero count (`run_chunked`, the worker's policy;
reported as '<packing>-c64').

Each case compares what the worker would answer from the graph (`answer`: the generic step - each
row's legal logits softmaxed per segment, value / margin as they are, `aux` of the belief rows) with
torch's answers on the same rows: `forward_static` on the host-packed rows (a token network; the
CUDA graphs' entry point, `infer.TokenGraphRunner`'s arithmetic on the CPU) and the eager evaluator
(`infer.NetEvaluator`): priors, value, margin and the belief rows' auxiliary predictions, max abs
difference within `--tol` (2e-5: tests/test_token_graphs.py's graphs-vs-eager tolerance). Also: value
exactly 0 beyond n_players, one auxiliary row per belief row, and the graph deterministic (each case run
twice: the outputs bit for bit the same - a kernel that races on duplicate indices is caught here). Prints one JSON line per case and a
summary; exit 1 when a case fails.

Needs onnx, onnxscript and onnxruntime (CPU build); nothing on a run's path imports this module.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

from .infer import NetEvaluator, segment_softmax_np, token_index, token_presence
from .protocol import MAX_SEATS, ROW_BELIEF, ROW_LEAF, ROW_VALUE, belief_rows

TOL = 2e-5
SIZES = (1, 7, 64, 256, 1024)
CHUNK = 64


class Rows:
    """Network batch rows in the driver's sparse form (protocol.py): xs_off i64 [B+1], xs_idx i32,
    xs_val f32, n_players i64 [B], opt_off i64 [B+1], opt_action i32, extra u8 [B]; `features` F;
    `batch_off` the recorded batches' first rows (None: not recorded batches); `tail` / `ov` the rows' tail
    tokens and pointer overrides (`rows.Tail` / `rows.Overrides`, docs/search.md 2.15; None: a game without)."""

    def __init__(self, xs_off, xs_idx, xs_val, n_players, opt_off, opt_action, extra, features, batch_off=None,
                 source='', tail=None, ov=None):
        self.xs_off = np.asarray(xs_off, np.int64)
        self.xs_idx = np.asarray(xs_idx, np.int32)
        self.xs_val = np.asarray(xs_val, np.float32)
        self.n_players = np.asarray(n_players, np.int64)
        self.opt_off = np.asarray(opt_off, np.int64)
        self.opt_action = np.asarray(opt_action, np.int32)
        self.extra = np.asarray(extra, np.uint8)
        self.F = int(features)
        self.batch_off = batch_off
        self.source = source
        self.tail = tail
        self.ov = ov
        for k, v in (('tail', tail), ('ov', ov)):
            if v is not None and v.N != self.B:
                raise ValueError(f'Rows: a {k} of {v.N} rows for {self.B} rows')

    def tail_stats(self):
        """{tail_rows, tail_tokens, override_rows, overrides} of the rows (zeros without)."""
        tc = self.tail.counts() if self.tail is not None else np.zeros(self.B, np.int64)
        oc = self.ov.counts() if self.ov is not None else np.zeros(self.B, np.int64)
        return {'tail_rows': int((tc > 0).sum()), 'tail_tokens': int(tc.sum()), 'override_rows': int((oc > 0).sum()),
                'overrides': int(oc.sum())}

    @property
    def B(self):
        return len(self.n_players)

    def take(self, idx):
        """The rows `idx` (an int array: any order, repeats allowed), offsets rebased."""
        idx = np.asarray(idx, np.int64)
        xa, xb = self.xs_off[idx], self.xs_off[idx + 1]
        oa, ob = self.opt_off[idx], self.opt_off[idx + 1]
        sel = lambda a, b: np.concatenate([np.arange(u, v) for u, v in zip(a, b)]) if len(a) else np.zeros(0, np.int64)
        xi, oi = sel(xa, xb).astype(np.int64), sel(oa, ob).astype(np.int64)
        return Rows(np.concatenate([[0], np.cumsum(xb - xa)]), self.xs_idx[xi], self.xs_val[xi], self.n_players[idx],
                    np.concatenate([[0], np.cumsum(ob - oa)]), self.opt_action[oi], self.extra[idx], self.F,
                    source=self.source, tail=None if self.tail is None else self.tail.take(idx),
                    ov=None if self.ov is None else self.ov.take(idx))

    def dense(self):
        x = np.zeros((self.B, self.F), np.float32)
        x[np.repeat(np.arange(self.B), np.diff(self.xs_off)), self.xs_idx] = self.xs_val
        return x

    @classmethod
    def from_dense(cls, x, n_players, opt_off, opt_action, extra, source='', tail=None, ov=None):
        nz = x != 0                                    # (NaN != 0: kept)
        r, c = np.nonzero(nz)
        off = np.concatenate([[0], np.cumsum(nz.sum(1))])
        return cls(off, c.astype(np.int32), x[r, c], n_players, opt_off, opt_action, extra, x.shape[1], source=source,
                   tail=tail, ov=ov)

    @classmethod
    def load(cls, path, max_rows=4096, belief_every=8):
        """A recorded batch file (tools/onnx_record.py) or a sample shard (at most `max_rows` rows of it:
        no-option rows are value-only rows; each `belief_every`-th leaf row gets a root belief copy after it)."""
        with np.load(path) as z:
            files = set(z.files)
            if 'xs_off' in files:
                meta = json.loads(str(z['meta']))
                tail, ov = recorded_rows({k: z[k] for k in files if k != 'meta'}, meta)
                return cls(z['xs_off'], z['xs_idx'], z['xs_val'], z['n_players'], z['opt_off'], z['opt_action'],
                           z['extra'], meta['features'], batch_off=z['batch_off'], source=os.path.basename(path),
                           tail=tail, ov=ov)
        from .data import read_shard, shard_tail
        s = read_shard(path, dense=False)
        stail, sov = shard_tail(s)
        if 'x' in s:
            x = s['x']
            N = min(len(x), max_rows)
            get = lambda a, b: np.asarray(x[a:b], np.float32)
        else:
            from .data import stored_rows
            xs = stored_rows(s)
            N = min(xs.N, max_rows)
            get = lambda a, b: np.asarray(xs.dense(np.arange(a, b)), np.float32)
        off, act, npl = s['opt_off'], s['opt_action'], s['n_players']
        parts, k = [], 0
        for a in range(0, N, 512):
            b = min(N, a + 512)
            xx = get(a, b)
            rows, kinds, oo, aa = [], [], [0], []
            for i in range(a, b):
                c = int(off[i + 1] - off[i])
                rows.append(i - a)
                kinds.append(ROW_LEAF if c else ROW_VALUE)
                aa.append(act[off[i]:off[i + 1]])
                oo.append(oo[-1] + c)
                if c:
                    k += 1
                    if k % belief_every == 0:
                        rows.append(i - a)
                        kinds.append(ROW_BELIEF)
                        oo.append(oo[-1])
            src = a + np.asarray(rows)
            kinds = np.asarray(kinds, np.uint8)
            # (a v2 shard's tails and overrides: a belief copy has its leaf row's tail and no option, so no override)
            pt = None if stail is None else stail.take(src)
            po = None if sov is None else _ov_rows(sov, src, kinds)
            parts.append(cls.from_dense(xx[rows], np.asarray(npl, np.int64)[src], np.asarray(oo),
                                        np.concatenate(aa).astype(np.int32), kinds, tail=pt, ov=po))
        return cat(parts, source=os.path.basename(path))


def _ov_rows(ov, src, kinds):
    """The overrides of rows `src` of `ov`, none for a belief copy (`kinds` ROW_BELIEF: no options)."""
    from .rows import Overrides
    parts = [Overrides.empty(1) if k == ROW_BELIEF else ov.take(np.array([r])) for r, k in zip(src, kinds)]
    return Overrides.concat(parts) if parts else Overrides.empty(0)


def recorded_rows(z, meta):
    """A recorded batch file's (tools/onnx_record.py, a reference set) tails and overrides -> (`rows.Tail` or None,
    `rows.Overrides` or None): `t_off` / `t_seg` / `t_slot` / `t_val` with meta `t_segs` [[segment, name, width, head
    slots], ...], `ov_off` / `ov_opt` / `ov_tok` (docs/search.md 2.15); None for a file without."""
    from .rows import Overrides, Tail
    tail = ov = None
    if 't_off' in z:
        segs = meta.get('t_segs')
        if not segs:
            raise ValueError('a recorded file with tail tokens but no t_segs in its meta')
        widths = {int(s): (int(w), int(h), k) for k, (s, _n, w, h) in enumerate(segs)}
        tail = Tail(z['t_off'], z['t_seg'], z['t_slot'], z['t_val'], widths)
    if 'ov_off' in z:
        ov = Overrides(z['ov_off'], z['ov_opt'], z['ov_tok'])
    return tail, ov


def cat(parts, source=''):
    """Rows of several Rows objects, in order."""
    from .rows import Overrides, Tail
    xo = np.concatenate([[0], np.cumsum(np.concatenate([np.diff(p.xs_off) for p in parts]))])
    oo = np.concatenate([[0], np.cumsum(np.concatenate([np.diff(p.opt_off) for p in parts]))])
    if len({(p.tail is None, p.ov is None) for p in parts}) != 1:
        raise ValueError('cat: rows with and without tails / overrides')
    return Rows(xo, np.concatenate([p.xs_idx for p in parts]), np.concatenate([p.xs_val for p in parts]),
                np.concatenate([p.n_players for p in parts]), oo, np.concatenate([p.opt_action for p in parts]),
                np.concatenate([p.extra for p in parts]), parts[0].F, source=source,
                tail=None if parts[0].tail is None else Tail.concat([p.tail for p in parts]),
                ov=None if parts[0].ov is None else Overrides.concat([p.ov for p in parts]))


def absent_nan(model, rows):
    """`rows` with NaN in every absent slot's floats but its present flag (a token network's segments with
    a flag) - a token network must never read them."""
    x = rows.dense()
    for s in model.segs:
        if s['present'] is None or not s['count']:
            continue                                   # (a tail-only segment has no head slot: docs/az.md 43.5)
        o, k, w = s['offset'], s['count'], s['width']
        v = x[:, o:o + k * w].reshape(-1, k, w)
        absent = v[..., s['present']] <= 0.5
        keep = v[..., s['present']].copy()
        v[absent] = np.nan
        v[..., s['present']] = keep
    return Rows.from_dense(x, rows.n_players, rows.opt_off, rows.opt_action, rows.extra, rows.source, rows.tail, rows.ov)


# ---------------------------------------------------------------- answers
# The exact exponential of the generic step's softmax (the native worker's `eval::exp_f32`, zero-farm
# docs/az.md 44.2): exp of a float32 through a fixed sequence of float64 IEEE operations - Cody-Waite
# reduction by ln 2, the Taylor polynomial of degree 13 by Horner, an exact scaling by 2^n, one rounding to
# float32 - with no libm and no fused multiply-add, so numpy here and the Rust worker on any machine compute
# the same bits (numpy's float32 `exp` and the C library's `expf` differ in the last bit now and then, and a
# prior's last bit can change a searched move). Its error before the rounding is ~5e-16 relative.
_INV_LN2 = 1.4426950408889634
_LN2_HI = 0.6931471803691238                       # fdlibm's ln2_hi: n * _LN2_HI exact for |n| < 2^21
_LN2_LO = 1.9082149292705877e-10
_EXP_C = tuple(1.0 / math.factorial(k) for k in range(14))


def exp_f32(d):
    """exp(d) for max-subtracted float32 logits d <= 0 -> float32 (the worker's `exp_f32`, bit for bit): NaN
    stays NaN, below -150 it is 0."""
    d = np.asarray(d, np.float32)
    x = d.astype(np.float64)
    nan = np.isnan(x)
    low = x < -150.0
    x = np.where(nan | low, 0.0, x)
    n = np.floor(x * _INV_LN2 + 0.5)
    r = (x - n * _LN2_HI) - n * _LN2_LO
    p = np.full_like(r, _EXP_C[13])
    for c in reversed(_EXP_C[:13]):
        p = p * r + c
    y = np.ldexp(p, n.astype(np.int64)).astype(np.float32)
    y[low] = 0.0
    y[nan] = np.nan
    return y


def exact_softmax(logits, off):
    """The generic step's softmax of a flat float32 array over the segments of `off` (empty segments hold
    nothing), the native worker's `softmax_exact` bit for bit: per segment m = the largest logit (NaN
    propagates), e = `exp_f32(l - m)` (the difference in float32), their float32 sum in order (not numpy's
    pairwise sum), e / sum in float32."""
    off = np.asarray(off, np.int64)
    lg = np.asarray(logits, np.float32)
    cnt = np.diff(off)
    starts = off[:-1]
    keep = cnt > 0
    cnt, starts = cnt[keep], starts[keep]
    if not len(cnt):
        return np.zeros(0, np.float32)
    mx = np.maximum.reduceat(lg, starts)
    e = exp_f32(lg - np.repeat(mx, cnt))
    s = np.zeros(len(cnt), np.float32)
    for j in range(int(cnt.max())):
        k = np.flatnonzero(cnt > j)
        s[k] = s[k] + e[starts[k] + j]
    return (e / np.repeat(s, cnt)).astype(np.float32)


class Refused(ValueError):
    """A zero-onnx/2 graph refused some of a batch's tail tokens / overrides (its `refused` output; docs/az.md 46):
    what torch raises on (R5 / R6), never answered."""


def answer(outs, rows):
    """The worker's generic step on the graph's outputs (policy, value, margin, aux[, refused]) -> what an evaluator
    returns: (priors [options] - each row's legal logits softmaxed per segment, float32, `exact_softmax`: the
    native worker's bits -, value [B, 8], margin [B, 8]) plus the belief rows' auxiliary answers when the batch
    has belief rows. A zero-onnx/2 graph's `refused` must be 0 on every row (`Refused` otherwise)."""
    policy, value, margin, aux = outs[:4]
    if len(outs) > 4:
        ref = np.asarray(outs[4])
        if ref.any():
            bad = np.flatnonzero(ref)
            raise Refused(f'the graph refused {int(ref.sum())} tail token(s) / override(s) in {len(bad)} row(s) '
                          f'(first: row {int(bad[0])}): not tokens / targets the network takes (R5 / R6, docs/az.md 46)')
    r = np.repeat(np.arange(rows.B), np.diff(rows.opt_off))
    pri = exact_softmax(policy[r, rows.opt_action], rows.opt_off)
    br = belief_rows(rows.extra)
    out = (pri, value, margin)
    return out + (aux[br],) if len(br) else out


def varlen_feeds(rows, sizes=None):
    """zero-onnx/2's more inputs for `rows` (docs/az.md 46): the options, the tail tokens (each one's floats
    consecutive, as the driver sends them; the slots as int32 halves, `onnx_export.slot_pieces`) and the overrides (the
    packed tokens as int32 pieces, `override_pieces`) - empty CSRs for rows without. `sizes` ({actions, tail_width}:
    `graph_sizes`): the piece refused at `int32_bound` >= 2^31, as zero_worker refuses it."""
    from .onnx_export import INT32_MAX, int32_bound, override_pieces, slot_pieces
    B = rows.B
    t, o = rows.tail, rows.ov
    z = np.zeros(B + 1, np.int64)
    e = np.zeros(0, np.int32)
    hi, lo = slot_pieces(t.slot) if t is not None else (e, e)
    ov = override_pieces(o.tok) if o is not None else (e, e, e, e)
    f = {'opt_off': rows.opt_off.astype(np.int64), 'opt_action': rows.opt_action.astype(np.int32),
         't_off': z if t is None else t.off.astype(np.int64),
         't_seg': e if t is None else t.seg.astype(np.int32), 't_slot_hi': hi, 't_slot_lo': lo,
         't_val': np.zeros(0, np.float32) if t is None else t.val.astype(np.float32),
         'ov_off': z if o is None else o.off.astype(np.int64),
         'ov_opt': e if o is None else o.opt.astype(np.int32),
         'ov_seg': ov[0], 'ov_col': ov[1], 'ov_slot_hi': ov[2], 'ov_slot_lo': ov[3]}
    if sizes is not None:
        m = int32_bound(B, len(rows.xs_idx), rows.F, sizes['actions'], sizes['tail_width'], f['t_off'],
                        len(f['t_seg']), len(f['t_val']), len(f['opt_action']), len(f['ov_opt']))
        if m > INT32_MAX:
            raise ValueError(f'a piece of {B} rows whose integers could reach {m} in the graph: zero-onnx/2 keeps every '
                             f'integer below 2^31 (docs/az.md 46.2; smaller pieces)')
    return f


def graph_sizes(sess):
    """A zero-onnx/2 session's {actions, tail_width} (`varlen_feeds`' bound); None for a zero-onnx/1 graph."""
    if not any(i.name == 't_off' for i in sess.get_inputs()):
        return None
    m = sess.get_modelmeta().custom_metadata_map
    return {'actions': int(m['actions']), 'tail_width': int(m.get('tail_width') or 0)}


def is_v2(sess):
    """The session's graph takes zero-onnx/2's inputs."""
    return any(i.name == 't_off' for i in sess.get_inputs())


def feeds(rows, packing=None, pidx=None, junk=False, seed=0, v2=False):
    """ONNX Runtime's inputs for `rows` (zero-onnx/1; 'host' also tok_full / tok_n from the host's packing,
    `junk`: padding positions hold random slots; `v2`: zero-onnx/2's, `varlen_feeds` - a `graph_sizes` dict also
    checks the piece's `int32_bound`)."""
    f = {'x_ptr': rows.xs_off, 'x_idx': rows.xs_idx, 'x_val': rows.xs_val, 'n_players': rows.n_players,
         'belief': rows.extra == ROW_BELIEF}
    if v2:
        f.update(varlen_feeds(rows, v2 if isinstance(v2, dict) else None))
    if packing == 'host':
        P = presence_sparse(rows, pidx)
        n = P.sum(1)
        T = max(2, int(n.max()))
        full = token_index(P, np.zeros((rows.B, T), np.int64))
        if junk:
            pad = np.arange(T)[None, :] >= n[:, None]
            full[pad] = np.random.default_rng(seed).integers(0, P.shape[1], size=int(pad.sum()))
        f['tok_full'] = full
        f['tok_n'] = n.astype(np.int64)
    return f


def run_chunked(sess, rows, chunk=64, packing=None, pidx=None):
    """The graph's outputs for `rows` evaluated in pieces of at most `chunk` rows, the rows grouped by their
    non-zero count (a worker-side policy that needs nothing of the network: a row's outputs depend only on
    it; rows of similar size pad less in a token network's attention, as the eager evaluator's chunks of
    64 rows sorted by token count) -> (policy, value, margin, aux) in the rows' order."""
    v2 = graph_sizes(sess)
    if rows.B <= chunk:
        return sess.run(None, feeds(rows, packing, pidx, v2=v2))
    order = np.argsort(np.diff(rows.xs_off), kind='stable')
    outs = None
    for a in range(0, rows.B, chunk):
        idx = order[a:a + chunk]
        o = sess.run(None, feeds(rows.take(idx), packing, pidx, v2=v2))
        if outs is None:
            outs = [np.empty((rows.B,) + v.shape[1:], v.dtype) for v in o]
        for dst, v in zip(outs, o):
            dst[idx] = v
    return outs


def presence_sparse(rows, pidx):
    """`infer.TokenGraphRunner.presence_sparse` from the sparse rows: [B, 1 + slots] bool."""
    pidx = np.asarray(pidx, np.int64)
    F_ = rows.F
    flag_col = np.full(F_, -1, np.int64)
    has = pidx >= 0
    flag_col[pidx[has]] = 1 + np.flatnonzero(has)
    P = np.zeros((rows.B, 1 + len(pidx)), bool)
    P[:, 0] = True
    P[:, 1 + np.flatnonzero(~has)] = True
    r = np.repeat(np.arange(rows.B), np.diff(rows.xs_off))
    col = flag_col[rows.xs_idx.astype(np.int64)]
    hit = (col >= 0) & (rows.xs_val > 0.5)
    P[r[hit], col[hit]] = True
    return P


def model_tail_kw(model, rows):
    """The rows' tail / overrides as `NetEvaluator.evaluate` takes them for `model` ({} for a network without; an
    empty one for rows without what the network needs)."""
    from .rows import Overrides, Tail
    kw = {}
    if getattr(model, 'tails', None):
        w = {j: (wd, c, k) for k, (j, _n, wd, c) in enumerate(model.tail_segments())}
        kw['tail'] = rows.tail if rows.tail is not None else Tail.empty(rows.B, w)
    elif rows.tail is not None:
        kw['tail'] = rows.tail                         # (refused by the network: R5)
    if getattr(model, 'option_pointers', False):
        kw['ov'] = rows.ov if rows.ov is not None else Overrides.empty(rows.B)
    elif rows.ov is not None:
        kw['ov'] = rows.ov
    return kw


@torch.inference_mode()
def static_answer(model, rows):
    """torch `TokenNet.forward_static` on the host-packed rows (`infer.token_presence` / `token_index`, padding
    0; with tails `rows.token_full` - each row's tail tokens after its head tokens - and the overrides resolved by
    `rows.resolve`) with `infer.TokenGraphRunner`'s post-processing, on the CPU -> an evaluator's answer."""
    from .net import aux_answer
    x = rows.dense()
    P = token_presence(x, model.present_index())
    layout = list(getattr(model, 'aux_layout', None) or [])
    kw = model_tail_kw(model, rows)
    if kw:
        from . import rows as R
        from .infer import ov_torch
        tc = kw['tail'].counts() if 'tail' in kw else np.zeros(rows.B, np.int64)
        full, n = R.token_full(P, tc, model.n_tok, 1)
        tk = {}
        if 'tail' in kw:
            tk['tail'] = kw['tail'].to_torch('cpu')
        if 'ov' in kw:
            tk['ov'] = ov_torch(R.resolve(kw.get('tail'), kw['ov'], rows.opt_off, rows.opt_action, model.config['layout'],
                                          present=P), 'cpu')
        r = model.forward_static(torch.from_numpy(x), torch.from_numpy(rows.n_players), torch.from_numpy(full),
                                 torch.from_numpy(n.astype(np.int64)), aux=bool(layout), **tk)
    else:
        n = P.sum(1)
        full = token_index(P, np.zeros((rows.B, int(n.max())), np.int64))
        r = model.forward_static(torch.from_numpy(x), torch.from_numpy(rows.n_players), torch.from_numpy(full),
                                 torch.from_numpy(n.astype(np.int64)), aux=bool(layout))
    S = model.seats
    value = np.zeros((rows.B, MAX_SEATS), np.float32)
    margin = np.zeros((rows.B, MAX_SEATS), np.float32)
    value[:, :S] = torch.softmax(r[1].float(), 1).numpy()
    margin[:, :S] = r[2].float().numpy()
    value[np.arange(MAX_SEATS)[None, :] >= rows.n_players[:, None]] = 0.0
    pol = r[0].float().numpy()
    aux = None
    if layout:
        pred = aux_answer(model, r[3])
        aux = torch.cat([pred[t[0]].float() for t in layout], 1).numpy()
    return answer((pol, value, margin, aux), rows)


def eager_answer(ev, rows):
    """The eager evaluator's answer (`NetEvaluator.evaluate` on the densified rows, with their tails and overrides)."""
    return ev.evaluate(rows.dense(), rows.n_players, rows.opt_off.astype(np.int32), rows.opt_action, rows.extra,
                       **model_tail_kw(ev.model, rows))


def diffs(a, b):
    """Max abs difference per item (priors, value, margin[, aux]); NaN anywhere -> inf."""
    if len(a) != len(b):
        raise AssertionError(f'answers of {len(a)} and {len(b)} items')
    out = []
    for u, v in zip(a, b):
        if u.shape != v.shape:
            raise AssertionError(f'shapes {u.shape} vs {v.shape}')
        d = float(np.abs(u.astype(np.float64) - v.astype(np.float64)).max()) if u.size else 0.0
        out.append(d if np.isfinite(d) else float('inf'))
    return out


ITEMS = ('priors', 'value', 'margin', 'aux')


def session(path, threads=4):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])


def check_invariants(got, rows):
    """value exactly 0 beyond n_players; one auxiliary row per belief row; priors sum to 1 per leaf."""
    value = got[1]
    if np.any(value[np.arange(MAX_SEATS)[None, :] >= rows.n_players[:, None]] != 0):
        return 'value not 0 beyond n_players'
    nb = len(belief_rows(rows.extra))
    if nb and (len(got) != 4 or got[3].shape[0] != nb):
        return 'auxiliary rows != belief rows'
    cnt = np.diff(rows.opt_off)
    if len(got[0]) and np.abs(np.add.reduceat(got[0], rows.opt_off[:-1][cnt > 0]) - 1).max() > 1e-5:
        return 'priors do not sum to 1'
    return None


def cases(model, rows, sizes, tokens, recorded=3):
    """(name, rows, junk) of the gate on `rows` (module docstring); rows with tail tokens or overrides also as a
    case of their own, every such row of the file (at most 1,024; 'varlen', and 'varlen-nan'), and the rows with
    overrides alone (at most 1,024; 'overrides': much rarer than tail tokens in a real game)."""
    R = rows.B
    out = []
    for s in sizes:
        start = (s * 7919) % R
        idx = (start + np.arange(s)) % R
        sub = rows.take(idx)
        out.append((f'B{s}', sub, False))
        if tokens:
            out.append((f'B{s}-nan', absent_nan(model, sub), True))
    if rows.tail is not None or rows.ov is not None:
        has = np.zeros(R, bool)
        if rows.tail is not None:
            has |= rows.tail.counts() > 0
        if rows.ov is not None:
            has |= rows.ov.counts() > 0
        idx = np.flatnonzero(has)[:1024]
        if len(idx):
            sub = rows.take(idx)
            out.append(('varlen', sub, False))
            out.append(('varlen-nan', absent_nan(model, sub), True))
        if rows.ov is not None and rows.ov.NO:
            out.append(('overrides', rows.take(np.flatnonzero(rows.ov.counts() > 0)[:1024]), False))
    if rows.batch_off is not None:
        bo = rows.batch_off
        for k in range(min(recorded, len(bo) - 1)):
            out.append((f'batch{k}', rows.take(np.arange(bo[k], bo[k + 1])), False))
    return out


def run_gate(model, onnx_paths, rows_list, sizes=SIZES, tol=TOL, threads=4, log=print):
    """The gate (module docstring): `onnx_paths` {packing: path} of `model`'s exports, `rows_list` Rows
    objects -> (passed, [result dicts])."""
    from .net import TokenNet
    tokens = isinstance(model, TokenNet)
    torch.set_num_threads(threads)
    ev = NetEvaluator(model, 'cpu')
    sess = {p: session(path, threads) for p, path in onnx_paths.items()}
    v2 = {p: graph_sizes(s) for p, s in sess.items()}
    pidx = model.present_index() if tokens else None
    gated = set(getattr(model, 'gated', None) or [])
    results, ok = [], True
    for rows in rows_list:
        if rows.F != model.features:
            raise SystemExit(f'{rows.source}: rows of {rows.F} features for a network of {model.features}')
        for name, sub, junk in cases(model, rows, sizes, tokens):
            eager = eager_answer(ev, sub)
            static = static_answer(model, sub) if tokens else None
            base = diffs(static, eager) if tokens else None
            runs = [(p, s, None) for p, s in sess.items()]
            if sub.B > CHUNK:
                runs += [(f'{p}-c{CHUNK}', s, p) for p, s in sess.items() if p != 'host']
            for p, s, chunked in runs:
                t0 = time.perf_counter()
                if chunked:                                       # (the worker's chunks: `run_chunked`)
                    outs = run_chunked(s, sub, CHUNK)
                    secs = time.perf_counter() - t0
                    again = run_chunked(s, sub, CHUNK)
                else:
                    f = feeds(sub, p, pidx, junk=junk and p == 'host', v2=v2[chunked or p])
                    outs = s.run(None, f)
                    secs = time.perf_counter() - t0
                    again = s.run(None, f)
                try:
                    got = answer(outs, sub)
                except Refused as e:
                    got, bad = None, str(e)
                else:
                    bad = check_invariants(got, sub)
                if bad is None and not all(np.array_equal(u, v, equal_nan=True) for u, v in zip(outs, again)):
                    bad = 'not deterministic: the same inputs twice gave other outputs'
                d_e = diffs(got, eager) if got is not None else [float('inf')] * len(eager)
                d_s = (diffs(got, static) if got is not None else [float('inf')] * len(static)) if tokens else None
                worst = max(d_e + (d_s or []))
                fail = bad is not None or worst > tol
                ok &= not fail
                r = {'source': rows.source, 'case': name, 'packing': p, 'rows': sub.B,
                     'leaf': int((sub.extra == ROW_LEAF).sum()), 'value_rows': int((sub.extra == ROW_VALUE).sum()),
                     'belief_rows': int((sub.extra == ROW_BELIEF).sum()),
                     'ort_vs_eager': dict(zip(ITEMS, d_e)), 'ort_vs_static': dict(zip(ITEMS, d_s)) if tokens else None,
                     'static_vs_eager': dict(zip(ITEMS, base)) if tokens else None, 'max': worst,
                     'ort_ms': round(secs * 1e3, 2), 'pass': not fail}
                if sub.tail is not None or sub.ov is not None:
                    r.update(sub.tail_stats())
                    r['gated_tokens'] = int(np.isin(sub.tail.seg.astype(np.int64), list(gated)).sum()) \
                        if sub.tail is not None and gated else 0
                if bad:
                    r['invariant'] = bad
                results.append(r)
                log(json.dumps(r))
    return ok, results


def summarize(results):
    """Per packing: cases, failures, the worst ORT-vs-torch difference per item."""
    out = {}
    for r in results:
        s = out.setdefault(r['packing'], {'cases': 0, 'failed': 0, 'worst': {k: 0.0 for k in ITEMS}})
        s['cases'] += 1
        s['failed'] += 0 if r['pass'] else 1
        for d in (r['ort_vs_eager'], r['ort_vs_static'] or {}):
            for k, v in d.items():
                s['worst'][k] = max(s['worst'][k], v)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='zero-onnx parity gate: ONNX Runtime (CPU EP) vs torch (docs/az.md 41, 46)')
    ap.add_argument('checkpoint')
    ap.add_argument('--batches', nargs='+', required=True)
    ap.add_argument('--packing', default='mask,compact,packed,host')
    ap.add_argument('--sizes', default=','.join(map(str, SIZES)))
    ap.add_argument('--tol', type=float, default=TOL)
    ap.add_argument('--onnx-dir', default=None, help='where the exports go (default: next to the checkpoint)')
    ap.add_argument('--reuse', action='store_true', help='reuse an export whose metadata names this checkpoint\'s sha256')
    ap.add_argument('--game', default=None)
    ap.add_argument('--allow-mismatch', action='store_true')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--max-rows', type=int, default=4096, help='rows read from a shard')
    ap.add_argument('--json', default=None, help='write the results here')
    a = ap.parse_args(argv)
    from . import onnx_export as OX
    model, info = OX.load_checkpoint(a.checkpoint, a.game, a.allow_mismatch)
    d = a.onnx_dir or os.path.dirname(os.path.abspath(a.checkpoint))
    os.makedirs(d, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.checkpoint))[0]
    from .net import TokenNet
    packs = [p for p in a.packing.split(',') if p] if isinstance(model, TokenNet) else ['none']
    if OX.onnx_version(model) == 2 and packs != ['packed']:
        print(f'zero-onnx/2 (variable-length token sets): packing \'packed\' only (not {packs})', file=sys.stderr)
        packs = ['packed']
    paths, export_s = {}, {}
    for p in packs:
        path = os.path.join(d, f'{stem}.{p}.onnx')
        if not (a.reuse and os.path.exists(path) and read_metadata(path).get('checkpoint_sha256') == info['checkpoint_sha256']):
            t0 = time.perf_counter()
            OX.export(model, path, p if p != 'none' else 'mask', info)
            export_s[p] = round(time.perf_counter() - t0, 1)
        paths[p] = path
    rows_list = [Rows.load(b, a.max_rows) for b in a.batches]
    sizes = [int(s) for s in a.sizes.split(',') if s]
    ok, results = run_gate(model, paths, rows_list, sizes, a.tol, a.threads)
    summ = {'checkpoint': a.checkpoint, 'sha256': info['checkpoint_sha256'], 'tol': a.tol, 'passed': ok,
            'packings': summarize(results), 'export_seconds': export_s,
            'onnx_bytes': {p: os.path.getsize(v) for p, v in paths.items()}}
    print(json.dumps(summ))
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump({'summary': summ, 'results': results}, f, indent=1)
    print(f'ONNX PARITY {"PASSED" if ok else "FAILED"} (tol {a.tol:g})', file=sys.stderr)
    return 0 if ok else 1


def read_metadata(path):
    import onnx
    m = onnx.load(path, load_external_data=False)
    return {p.key: p.value for p in m.metadata_props}


if __name__ == '__main__':
    sys.exit(main())
