//! A token network's sparse batch packed for its CUDA graphs (`zero.infer.TokenGraphRunner`, docs/az.md 43.12): the
//! host work of a launch done here, off the GIL, instead of numpy's many passes - the same integers and floats as
//! infer.py's numpy path (`presence_sparse`, `token_index` / `token_index_tail`, the non-zeros' flat indexes,
//! `rows.Tail.padded` and its packing), bit for bit.

/// A sparse batch's rows, their tokens and the graph's index arrays.
///
/// In: the batch's non-zeros (`xs_off` [B + 1] offsets from 0, `xs_idx` feature indexes, `xs_val` values), the
/// network's width `nfeat`, per feature its present flag's column of the token axis (`flag_col`, -1: not a flag),
/// the columns present in every row (`base` [n_tok]: the global token and the slots without a flag), the rows'
/// tail tokens (`t_off` [B + 1], None: no tail), the longest row the graphs take (`long_t`, < 0: any) and the
/// token step `t_step`.
///
/// Out: `fi` / `val` [nnz] each non-zero's flat index `row * nfeat + feature` and value; `ntok` [Bp] each row's
/// tokens (head + tail; 1 for the padding rows B..Bp); `full` [>= Bp * Tp] each row's token positions (its present
/// head columns ascending, then its tail tokens at `n_tok` + their index in the batch's tail, then 0; the padding
/// rows 0); `pres` [B * n_tok] (when given) the rows' present head columns. -> (Tp, the longest row's tokens); Tp
/// = 0 and only `ntok` (and `pres`) filled when a row is longer than `long_t`.
#[allow(clippy::too_many_arguments)]
pub fn sparse(
    xs_off: &[i64],
    xs_idx: &[u32],
    xs_val: &[f32],
    nfeat: usize,
    flag_col: &[i64],
    base: &[bool],
    t_off: Option<&[i64]>,
    long_t: i64,
    t_step: usize,
    fi: &mut [i64],
    val: &mut [f32],
    ntok: &mut [i64],
    full: &mut [i64],
    mut pres: Option<&mut [bool]>,
) -> Result<(usize, usize), String> {
    if xs_off.len() < 2 {
        return Err(format!("pack_sparse_tokens: {} row offsets (a batch of at least one row)", xs_off.len()));
    }
    let b = xs_off.len() - 1;
    let nnz = xs_idx.len();
    let n_tok = base.len();
    if xs_off[0] != 0 || xs_off[b] as usize != nnz || xs_val.len() != nnz || xs_off.windows(2).any(|w| w[1] < w[0]) {
        return Err(format!(
            "pack_sparse_tokens: row offsets {}..{} (ascending from 0) for {nnz} feature indexes, {} values",
            xs_off[0],
            xs_off[b],
            xs_val.len()
        ));
    }
    if flag_col.len() != nfeat || fi.len() < nnz || val.len() < nnz || ntok.len() < b || t_step == 0 || n_tok == 0 {
        return Err(format!(
            "pack_sparse_tokens: {} flag columns for {nfeat} features, room for {} / {} of {nnz} non-zeros, {} rows of {b}, \
             token step {t_step}, {n_tok} head tokens",
            flag_col.len(),
            fi.len(),
            val.len(),
            ntok.len()
        ));
    }
    if let Some(t) = t_off {
        if t.len() != b + 1 || t[0] != 0 || t.windows(2).any(|w| w[1] < w[0]) {
            return Err(format!("pack_sparse_tokens: {} tail offsets (ascending from 0) for {b} rows", t.len()));
        }
    }
    if let Some(p) = pres.as_deref() {
        if p.len() < b * n_tok {
            return Err(format!("pack_sparse_tokens: room for {} presence flags of {b} x {n_tok}", p.len()));
        }
    }
    // each row's present head columns, ascending (`cols[coff[r]..coff[r + 1]]`), from a bitset of the columns
    let words = n_tok.div_ceil(64);
    let mut base_bits = vec![0u64; words];
    for (c, &m) in base.iter().enumerate() {
        if m {
            base_bits[c / 64] |= 1 << (c % 64);
        }
    }
    let mut bits = vec![0u64; words];
    let mut cols: Vec<u32> = Vec::with_capacity(b * 32);
    let mut coff: Vec<usize> = Vec::with_capacity(b + 1);
    coff.push(0);
    let mut most = 0usize;
    for r in 0..b {
        bits.copy_from_slice(&base_bits);
        let (lo, hi) = (xs_off[r] as usize, xs_off[r + 1] as usize);
        let row = (r * nfeat) as i64;
        for k in lo..hi {
            let f = xs_idx[k] as usize;
            if f >= nfeat {
                return Err(format!("pack_sparse_tokens: row {r}: feature {f} of a network of {nfeat}"));
            }
            fi[k] = row + f as i64;
            let v = xs_val[k];
            val[k] = v;
            let c = flag_col[f];
            if c >= 0 && v > 0.5 {
                let c = c as usize;
                if c >= n_tok {
                    return Err(format!("pack_sparse_tokens: flag column {c} of {n_tok} head tokens"));
                }
                bits[c / 64] |= 1 << (c % 64);
            }
        }
        for (w, &x) in bits.iter().enumerate() {
            let mut x = x;
            while x != 0 {
                cols.push((w * 64 + x.trailing_zeros() as usize) as u32);
                x &= x - 1;
            }
        }
        if let Some(p) = pres.as_deref_mut() {
            let p = &mut p[r * n_tok..(r + 1) * n_tok];
            p.fill(false);
            for &c in &cols[coff[r]..] {
                p[c as usize] = true;
            }
        }
        coff.push(cols.len());
        let nt = coff[r + 1] - coff[r] + t_off.map_or(0, |t| (t[r + 1] - t[r]) as usize);
        ntok[r] = nt as i64;
        most = most.max(nt);
    }
    for x in ntok[b..].iter_mut() {
        *x = 1;
    }
    if long_t >= 0 && most as i64 > long_t {
        return Ok((0, most));
    }
    let tp = most.div_ceil(t_step) * t_step;
    let bp = ntok.len();
    if full.len() < bp * tp {
        return Err(format!("pack_sparse_tokens: room for {} token positions of {bp} x {tp}", full.len()));
    }
    for r in 0..b {
        let dst = &mut full[r * tp..(r + 1) * tp];
        let hc = &cols[coff[r]..coff[r + 1]];
        for (d, &c) in dst.iter_mut().zip(hc) {
            *d = c as i64;
        }
        let mut k = hc.len();
        if let Some(t) = t_off {
            for j in t[r]..t[r + 1] {
                dst[k] = n_tok as i64 + j;
                k += 1;
            }
        }
        for d in dst[k..].iter_mut() {
            *d = 0;
        }
    }
    for d in full[b * tp..bp * tp].iter_mut() {
        *d = 0;
    }
    Ok((tp, most))
}

/// A batch's tail tokens padded for a graph of `ntp` entries: `tint` [>= 3 * ntp] their rows, segments, slots in
/// turn (the padding: row 0, `pad_seg`, `pad_slot`), `tval` [>= ntp * wt] each token's floats in its first
/// width(segment) columns, zeros after (the padding: zeros) - `rows.Tail.padded()` packed as TokenGraphRunner's
/// graphs take it, with its checks (`_check_tail_host`): every token of a tail segment of the batch (`width`
/// [segments] > 0) and of the network (`net_tail`), at a slot at or past the segment's head slots (`head`), the
/// batch's widest tail segment `wt` wide; the floats exactly the tokens' widths.
#[allow(clippy::too_many_arguments)]
pub fn tail(
    t_off: &[i64],
    t_seg: &[u16],
    t_slot: &[u32],
    t_val: &[f32],
    width: &[i64],
    net_tail: &[bool],
    head: &[i64],
    wt: usize,
    ntp: usize,
    pad_seg: i64,
    pad_slot: i64,
    tint: &mut [i64],
    tval: &mut [f32],
) -> Result<(), String> {
    let nt = t_seg.len();
    if t_off.is_empty() || t_off[0] != 0 || t_off[t_off.len() - 1] as usize != nt || t_slot.len() != nt || t_off.windows(2).any(|w| w[1] < w[0]) {
        return Err(format!("pack_tail_tokens: tail offsets (ascending from 0) for {nt} tokens / {} slots", t_slot.len()));
    }
    if ntp < nt || tint.len() < 3 * ntp || tval.len() < ntp * wt {
        return Err(format!("pack_tail_tokens: {nt} tokens for a graph of {ntp} (room for {} / {})", tint.len(), tval.len()));
    }
    let widest = width.iter().copied().max().unwrap_or(0).max(0) as usize;
    if widest != wt {
        return Err(format!("pack_tail_tokens: a tail of floats {widest} wide (the network's tail width {wt})"));
    }
    let (rows, rest) = tint[..3 * ntp].split_at_mut(ntp);
    let (segs, slots) = rest.split_at_mut(ntp);
    let mut vo = 0usize;
    let mut i = 0usize;
    for r in 0..t_off.len() - 1 {
        for _ in t_off[r]..t_off[r + 1] {
            let s = t_seg[i] as usize;
            let w = if s < width.len() { width[s] } else { 0 };
            if w <= 0 || s >= net_tail.len() || !net_tail[s] || (t_slot[i] as i64) < head[s] {
                return Err(format!(
                    "TokenGraphRunner: tail token {i} (row {r}, segment {s}, slot {}) is not a tail token of this network",
                    t_slot[i]
                ));
            }
            let w = w as usize;
            if vo + w > t_val.len() {
                return Err(format!("pack_tail_tokens: {} floats for tokens of more", t_val.len()));
            }
            rows[i] = r as i64;
            segs[i] = s as i64;
            slots[i] = t_slot[i] as i64;
            let dst = &mut tval[i * wt..(i + 1) * wt];
            dst[..w].copy_from_slice(&t_val[vo..vo + w]);
            for d in dst[w..].iter_mut() {
                *d = 0.0;
            }
            vo += w;
            i += 1;
        }
    }
    if vo != t_val.len() {
        return Err(format!("pack_tail_tokens: {} floats for tokens of {vo}", t_val.len()));
    }
    for k in nt..ntp {
        rows[k] = 0;
        segs[k] = pad_seg;
        slots[k] = pad_slot;
    }
    for d in tval[nt * wt..ntp * wt].iter_mut() {
        *d = 0.0;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_rows_tokens_and_tails() {
        // 3 features of slots (flags at 1 and 4 -> columns 1, 2), column 3 always; 2 rows, padded to 4
        let flag_col = vec![-1, 1, -1, -1, 2, -1];
        let base = vec![true, false, false, true];
        let xs_off = vec![0i64, 3, 4];
        let xs_idx = vec![0u32, 1, 4, 4];
        let xs_val = vec![0.25f32, 1.0, 0.5, 0.75];
        let t_off = vec![0i64, 0, 2];
        let (mut fi, mut val, mut ntok, mut full) = (vec![0; 4], vec![0.0; 4], vec![0; 4], vec![9; 4 * 8]);
        let mut pres = vec![false; 8];
        let r = sparse(&xs_off, &xs_idx, &xs_val, 6, &flag_col, &base, Some(&t_off), 128, 4, &mut fi, &mut val, &mut ntok, &mut full, Some(&mut pres));
        assert_eq!(r, Ok((8, 5)));
        assert_eq!(fi, vec![0, 1, 4, 10]);
        assert_eq!(val, xs_val);
        assert_eq!(ntok, vec![3, 5, 1, 1]);
        // row 0: columns 0, 1 (flag 1.0), 3 (flag 4 is 0.5: not > 0.5); row 1: 0, 2, 3, then its 2 tail tokens (4 + 0,
        // 4 + 1); the padding rows 0
        assert_eq!(full[..16], [0, 1, 3, 0, 0, 0, 0, 0, 0, 2, 3, 4, 5, 0, 0, 0]);
        assert!(full[16..].iter().all(|&x| x == 0));
        assert_eq!(pres, vec![true, true, false, true, true, false, true, true]);
        // a row longer than long_t: nothing but the counts
        let r = sparse(&xs_off, &xs_idx, &xs_val, 6, &flag_col, &base, Some(&t_off), 4, 4, &mut fi, &mut val, &mut ntok, &mut full, None);
        assert_eq!(r, Ok((0, 5)));
        // a feature out of the network
        let bad = vec![0u32, 1, 4, 6];
        assert!(sparse(&xs_off, &bad, &xs_val, 6, &flag_col, &base, None, -1, 4, &mut fi, &mut val, &mut ntok, &mut full, None).is_err());
    }

    #[test]
    fn tail_padded_and_checked() {
        // segments 1 (width 2, head 1) and 2 (width 3, head 0); 2 rows: row 0 one token, row 1 two
        let width = vec![0i64, 2, 3];
        let net_tail = vec![false, true, true];
        let head = vec![0i64, 1, 0];
        let t_off = vec![0i64, 1, 3];
        let (t_seg, t_slot) = (vec![1u16, 1, 2], vec![1u32, 5, 0]);
        let t_val = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let (mut tint, mut tval) = (vec![-7i64; 12], vec![-7.0f32; 12]);
        tail(&t_off, &t_seg, &t_slot, &t_val, &width, &net_tail, &head, 3, 4, 1, 1, &mut tint, &mut tval).unwrap();
        assert_eq!(tint, vec![0, 1, 1, 0, 1, 1, 2, 1, 1, 5, 0, 1]);
        assert_eq!(tval, vec![1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0, 6.0, 7.0, 0.0, 0.0, 0.0]);
        // a token inside its segment's head slots; floats of another count
        let bad = vec![0u32, 5, 0];
        assert!(tail(&t_off, &t_seg, &bad, &t_val, &width, &net_tail, &head, 3, 4, 1, 1, &mut tint, &mut tval).is_err());
        assert!(tail(&t_off, &t_seg, &t_slot, &t_val[..6], &width, &net_tail, &head, 3, 4, 1, 1, &mut tint, &mut tval).is_err());
    }
}
