"""The engine module's packers of a token network's sparse batch (docs/az.md 43.12: `pack_sparse_tokens`,
`pack_tail_tokens`, crates/py/src/pack.rs) against infer.py's numpy path - `TokenGraphRunner.presence_sparse`,
`token_index` / `token_index_tail`, the non-zeros' flat indexes, `rows.Tail.padded` padded as the graphs take it:
the same integers and floats, bit for bit. CPU only (the packers are host code)."""
import types

import numpy as np
import pytest

from zero import games as G
from zero.infer import T_STEP, TokenGraphRunner, tail_bucket, token_index, token_index_tail
from zero.rows import Tail

E = G.engine()


def runner_like(F, n_tok, rng, n_always=3):
    """The parts of a TokenGraphRunner `presence_sparse` and the packers read: per feature its flag's column (-1:
    none), the columns without a flag (always present), the base columns."""
    cols = np.arange(1, n_tok)
    always = np.sort(rng.choice(cols, n_always, replace=False))
    flagged = np.setdiff1d(cols, always)
    flag_col = np.full(F, -1, np.int64)
    flag_col[rng.choice(F, len(flagged), replace=False)] = flagged
    base = np.zeros(n_tok, bool)
    base[0] = True
    base[always] = True
    return types.SimpleNamespace(flag_col=flag_col, always=always, n_tok=n_tok, base=base, F=F)


def sparse_batch(B, F, flag_col, rng):
    """Rows of non-zeros (ascending features) with present flags at 0, 0.5 (not present), 0.75, 1, random values."""
    flags = np.flatnonzero(flag_col >= 0)
    idx, off = [], [0]
    for _ in range(B):
        k = int(rng.integers(0, 40))
        f = np.union1d(rng.choice(F, k, replace=False), rng.choice(flags, int(rng.integers(0, 8)), replace=False))
        idx.append(f)
        off.append(off[-1] + len(f))
    idx = np.concatenate(idx).astype(np.uint32)
    val = rng.choice(np.array([0.5, 0.75, 1.0, 0.25, 0.5000001, -1.0], np.float32), len(idx))
    m = rng.random(len(idx)) < 0.3
    val[m] = rng.standard_normal(int(m.sum())).astype(np.float32)
    return np.asarray(off, np.int64), idx, val


@pytest.mark.parametrize('B,tails', [(1, False), (63, True), (64, False), (130, True), (200, True)])
def test_pack_sparse_tokens_is_the_numpy_path(B, tails):
    rng = np.random.default_rng(B)
    F, n_tok = 700, 45
    r = runner_like(F, n_tok, rng)
    xs_off, xs_idx, xs_val = sparse_batch(B, F, r.flag_col, rng)
    tcnt = rng.integers(0, 6, B) if tails else np.zeros(B, np.int64)
    t_off = np.concatenate([[0], np.cumsum(tcnt)]).astype(np.int64)
    Bp = -(-B // 64) * 64
    nnz = int(xs_off[-1])
    # numpy: TokenGraphRunner's own presence, the counts, Tp, the token positions, the flat indexes
    P = TokenGraphRunner.presence_sparse(r, xs_off, xs_idx, xs_val)
    cnt = P.sum(1) + tcnt
    Tp = -(-int(cnt.max()) // T_STEP) * T_STEP
    want_full = np.full((Bp, Tp), 7, np.int64)
    if tails:
        token_index_tail(P, tcnt, n_tok, want_full)
    else:
        token_index(P, want_full)
    want_fi = np.repeat(np.arange(B, dtype=np.int64), np.diff(xs_off)) * F + xs_idx.astype(np.int64)
    # the packer, into buffers holding junk
    fi, val = np.full(nnz, -3, np.int64), np.full(nnz, np.nan, np.float32)
    ntok, full, pres = np.full(Bp, -3, np.int64), np.full(Bp * 160, -3, np.int64), np.zeros(B * n_tok, bool)
    got = E.pack_sparse_tokens(xs_off, xs_idx, xs_val, F, r.flag_col, r.base, t_off if tails else None, 128, T_STEP,
                               fi, val, ntok, full, pres)
    assert got == (Tp, int(cnt.max()))
    assert np.array_equal(fi, want_fi) and np.array_equal(val.view(np.int32), xs_val.view(np.int32))
    assert np.array_equal(ntok[:B], cnt) and np.all(ntok[B:] == 1)
    assert np.array_equal(full[:Bp * Tp].reshape(Bp, Tp), want_full)
    assert np.array_equal(pres.reshape(B, n_tok), P)
    # a row longer than the graphs take: nothing launched (Tp 0), the counts
    assert E.pack_sparse_tokens(xs_off, xs_idx, xs_val, F, r.flag_col, r.base, t_off if tails else None,
                                int(cnt.max()) - 1, T_STEP, fi, val, ntok, full) == (0, int(cnt.max()))


def test_pack_sparse_tokens_refuses_a_bad_batch():
    rng = np.random.default_rng(5)
    F, n_tok = 300, 20
    r = runner_like(F, n_tok, rng)
    xs_off, xs_idx, xs_val = sparse_batch(10, F, r.flag_col, rng)
    nnz = int(xs_off[-1])
    bufs = (np.zeros(nnz, np.int64), np.zeros(nnz, np.float32), np.zeros(64, np.int64), np.zeros(64 * 64, np.int64))
    bad = xs_idx.copy()
    bad[-1] = F
    with pytest.raises(ValueError, match='feature'):
        E.pack_sparse_tokens(xs_off, bad, xs_val, F, r.flag_col, r.base, None, -1, T_STEP, *bufs)
    with pytest.raises(ValueError, match='offsets'):
        E.pack_sparse_tokens(xs_off[:-1], xs_idx, xs_val, F, r.flag_col, r.base, None, -1, T_STEP, *bufs)
    with pytest.raises(ValueError, match='room'):
        E.pack_sparse_tokens(xs_off, xs_idx, xs_val, F, r.flag_col, r.base, None, -1, T_STEP, *bufs[:3],
                             np.zeros(10, np.int64))


def a_tail(N, rng, widths, heads, dtype=np.float32):
    segs = sorted(widths)
    off, seg, slot, val = [0], [], [], []
    for _ in range(N):
        toks = sorted({(int(s), int(heads[s] + rng.integers(0, 9))) for s in rng.choice(segs, int(rng.integers(0, 5)))})
        for s, l in toks:
            seg.append(s)
            slot.append(l)
            val.append(rng.standard_normal(widths[s]).astype(dtype))
        off.append(len(seg))
    v = np.concatenate(val) if val else np.zeros(0, dtype)
    return Tail(np.asarray(off), np.asarray(seg, np.uint16), np.asarray(slot, np.uint32), v,
                {s: (widths[s], heads[s]) for s in widths})


@pytest.mark.parametrize('dtype', [np.float32, np.float16])
def test_pack_tail_tokens_is_the_padded_tail(dtype):
    rng = np.random.default_rng(11)
    widths, heads = {2: 7, 3: 7, 5: 4, 9: 2}, {2: 3, 3: 3, 5: 0, 9: 1}
    nseg = 11
    seg_tail = np.zeros(nseg, bool)
    seg_tail[list(widths)] = True
    seg_count = np.zeros(nseg, np.int64)
    for s, h in heads.items():
        seg_count[s] = h
    Wt, pad_seg = max(widths.values()), min(widths)
    for N in (1, 9, 40):
        t = a_tail(N, rng, widths, heads, dtype)
        NT, NTp = t.NT, max(tail_bucket(t.NT), 16)
        tr, ts, tsl, tv = t.padded()
        want_i = np.stack([np.concatenate([a, np.full(NTp - NT, p, np.int64)])
                           for a, p in ((tr, 0), (ts, pad_seg), (tsl, heads[pad_seg]))]).ravel()
        want_v = np.zeros((NTp, Wt), np.float32)
        want_v[:NT] = tv
        w = np.zeros(nseg, np.int64)
        for s, x in widths.items():
            w[s] = x
        ti, vv = np.full(3 * NTp, -9, np.int64), np.full(NTp * Wt, np.nan, np.float32)
        E.pack_tail_tokens(t.off, t.seg, t.slot, np.asarray(t.val, np.float32), w, seg_tail, seg_count, Wt, NTp, pad_seg,
                           heads[pad_seg], ti, vv)
        assert np.array_equal(ti, want_i)
        assert np.array_equal(vv.view(np.int32), want_v.ravel().view(np.int32))
    # the runner's checks: a slot inside its segment's head, a segment the network has no tail for, other floats
    t = Tail(np.array([0, 1, 2]), np.array([2, 5], np.uint16), np.array([3, 0], np.uint32),
             np.arange(11, dtype=np.float32), {s: (widths[s], heads[s]) for s in widths})
    ti, vv = np.zeros(3 * 16, np.int64), np.zeros(16 * Wt, np.float32)
    E.pack_tail_tokens(t.off, t.seg, t.slot, t.val, w, seg_tail, seg_count, Wt, 16, pad_seg, 0, ti, vv)
    with pytest.raises(ValueError, match='not a tail token'):
        E.pack_tail_tokens(t.off, t.seg, np.array([2, 0], np.uint32), t.val, w, seg_tail, seg_count, Wt, 16, pad_seg, 0,
                           ti, vv)
    no = seg_tail.copy()
    no[5] = False
    with pytest.raises(ValueError, match='not a tail token'):
        E.pack_tail_tokens(t.off, t.seg, t.slot, t.val, w, no, seg_count, Wt, 16, pad_seg, 0, ti, vv)
    with pytest.raises(ValueError, match='floats'):
        E.pack_tail_tokens(t.off, t.seg, t.slot, t.val[:-1], w, seg_tail, seg_count, Wt, 16, pad_seg, 0, ti, vv)
