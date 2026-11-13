//! Variable-length rows (docs/search.md 1.4, 2.15): a network row is the fixed-width **head** (`encode`, the
//! game's `NUM_FEATURES` floats) plus a **tail** of extra tokens, one per object past a segment's head slots
//! (or of a tail-only segment), and, with `SearchGame::OPTION_POINTERS`, the row's **pointer overrides**.
//!
//! * A tail token is `(seg, slot, floats)`: `seg` the segment's index in the game's token layout (a
//!   `SearchGame::TAILS` entry), `slot >= ` the segment's head slots, `width(seg)` floats.
//! * A sealed tail is sorted by (seg, slot), without duplicates (`TailBuf::seal`).
//! * An override is `(opt, tok)`: the option's index in the row's option list and the packed target token
//!   (`pack_tok`), or -1 for the generic logit.
//!
//! A game without tails never builds a `TailBuf` row with a token and never emits an override: every path of
//! the driver is then the path of before (the golden digests are the proof).

/// A tail segment of the game's token layout (docs/search.md 1.4): its index in the layout's `segments`, name,
/// slot width and head slots. In layout order; must equal the layout's tail segments (zero.package checks).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TailSeg {
    pub seg: u16,
    pub name: &'static str,
    pub width: usize,
    pub head: usize,
}

/// The tail tokens of one row while a game encodes it (`SearchGame::encode_row`): tokens in any order,
/// sorted and checked by `seal`.
#[derive(Clone, Debug)]
pub struct TailBuf {
    seg: Vec<u16>,
    slot: Vec<u32>,
    val: Vec<f32>,
    segs: &'static [TailSeg],
    /// scratch of `seal`: the tokens' float offsets, the sort order, the sorted floats
    off: Vec<usize>,
    order: Vec<u32>,
    tmp: Vec<f32>,
}

impl Default for TailBuf {
    /// A buffer of no tail segments (a game without `TAILS`): `token` panics.
    fn default() -> Self {
        TailBuf::new(&[])
    }
}

impl TailBuf {
    pub fn new(segs: &'static [TailSeg]) -> Self {
        TailBuf { seg: Vec::new(), slot: Vec::new(), val: Vec::new(), segs, off: Vec::new(), order: Vec::new(), tmp: Vec::new() }
    }

    /// The tail segments this buffer accepts (the game's `TAILS`).
    pub fn segs(&self) -> &'static [TailSeg] {
        self.segs
    }

    /// The `TailSeg` of segment index `seg`, None if it is not a tail segment.
    pub fn seg_info(&self, seg: u16) -> Option<&'static TailSeg> {
        self.segs.iter().find(|s| s.seg == seg)
    }

    /// A zeroed slice of `seg`'s width for (seg, slot). Panics (an engine bug, R2) if `seg` is not a tail
    /// segment or `slot < head`.
    pub fn token(&mut self, seg: u16, slot: u32) -> &mut [f32] {
        let Some(s) = self.seg_info(seg) else {
            panic!("tail token of segment {seg}, which is not a tail segment (tail segments: {:?})", self.segs.iter().map(|s| (s.seg, s.name)).collect::<Vec<_>>());
        };
        if (slot as usize) < s.head {
            panic!("tail token ({} {seg}, slot {slot}) inside the segment's {} head slots", s.name, s.head);
        }
        let w = s.width;
        self.seg.push(seg);
        self.slot.push(slot);
        let off = self.val.len();
        self.val.resize(off + w, 0.0);
        &mut self.val[off..]
    }

    pub fn len(&self) -> usize {
        self.seg.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seg.is_empty()
    }

    pub fn clear(&mut self) {
        self.seg.clear();
        self.slot.clear();
        self.val.clear();
    }

    /// Sort by (seg, slot) (stable, values moved with them); panic (an engine bug, R2) on a duplicate (seg, slot).
    pub fn seal(&mut self) {
        let n = self.seg.len();
        if n == 0 {
            return;
        }
        let key = |i: usize| ((self.seg[i] as u64) << 32) | self.slot[i] as u64;
        let mut sorted = true;
        for i in 1..n {
            let (a, b) = (key(i - 1), key(i));
            if a == b {
                self.dup_panic(i);
            }
            if a > b {
                sorted = false;
            }
        }
        if sorted {
            return;
        }
        self.off.clear();
        let mut o = 0usize;
        for i in 0..n {
            self.off.push(o);
            o += self.width(self.seg[i]);
        }
        self.order.clear();
        self.order.extend(0..n as u32);
        let (seg, slot) = (&self.seg, &self.slot);
        self.order.sort_by_key(|&i| ((seg[i as usize] as u64) << 32) | slot[i as usize] as u64);
        for k in 1..n {
            let (a, b) = (self.order[k - 1] as usize, self.order[k] as usize);
            if self.seg[a] == self.seg[b] && self.slot[a] == self.slot[b] {
                self.dup_panic(b);
            }
        }
        self.tmp.clear();
        let mut seg2 = Vec::with_capacity(n);
        let mut slot2 = Vec::with_capacity(n);
        for &i in &self.order {
            let i = i as usize;
            let w = self.width(self.seg[i]);
            self.tmp.extend_from_slice(&self.val[self.off[i]..self.off[i] + w]);
            seg2.push(self.seg[i]);
            slot2.push(self.slot[i]);
        }
        self.seg = seg2;
        self.slot = slot2;
        std::mem::swap(&mut self.val, &mut self.tmp);
    }

    fn dup_panic(&self, i: usize) -> ! {
        let name = self.seg_info(self.seg[i]).map_or("?", |s| s.name);
        panic!("duplicate tail token ({name} {}, slot {})", self.seg[i], self.slot[i]);
    }

    fn width(&self, seg: u16) -> usize {
        self.seg_info(seg).map_or(0, |s| s.width)
    }

    pub fn seg(&self) -> &[u16] {
        &self.seg
    }
    pub fn slot(&self) -> &[u32] {
        &self.slot
    }
    pub fn val(&self) -> &[f32] {
        &self.val
    }

    /// Replace the tokens by a sealed row's (`seg` / `slot` sorted, `val` their floats): a row encoded elsewhere
    /// (a merged tree's encoding of a leaf, docs/search.md 3.16).
    pub fn set_sealed(&mut self, seg: &[u16], slot: &[u32], val: &[f32]) {
        self.seg.clear();
        self.seg.extend_from_slice(seg);
        self.slot.clear();
        self.slot.extend_from_slice(slot);
        self.val.clear();
        self.val.extend_from_slice(val);
    }

    /// A copy of the (sealed) tokens as a record's row.
    pub fn to_row(&self) -> TailRow {
        TailRow { seg: self.seg.clone(), slot: self.slot.clone(), val: self.val.clone() }
    }
}

/// A sealed row's tail tokens (records, samples): sorted by (seg, slot), each token's floats consecutive in
/// `val` (`width(seg)` each).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TailRow {
    pub seg: Vec<u16>,
    pub slot: Vec<u32>,
    pub val: Vec<f32>,
}

impl TailRow {
    pub fn len(&self) -> usize {
        self.seg.len()
    }
    pub fn is_empty(&self) -> bool {
        self.seg.is_empty()
    }
}

/// An override's target token (docs/search.md 2.15): `(seg << 40) | (col << 32) | slot`.
pub fn pack_tok(seg: u16, col: u8, slot: u32) -> i64 {
    ((seg as i64) << 40) | ((col as i64) << 32) | slot as i64
}

/// `pack_tok`'s inverse: (seg, col, slot). `tok` must be >= 0 (-1 is the generic logit, not a token).
pub fn unpack_tok(tok: i64) -> (u16, u8, u32) {
    debug_assert!(tok >= 0, "unpack_tok of {tok}");
    (((tok >> 40) & 0xffff) as u16, ((tok >> 32) & 0xff) as u8, (tok & 0xffff_ffff) as u32)
}

// The key mixing of a tail and overrides (docs/search.md 2.15, `tail_mix`). `lookahead::obs_hash` mixes a head
// pair of floats as (pair index i, bits), with i < 2^31 (a feature index is a u32). Every input of a tail mix has
// one of the top three bits of its index set, so no tail input can equal a head pair's input, and the four
// domains below are disjoint:
//   100 | t                         token t's header, value (seg << 32) | slot
//   010 | p                         the float at position p of the row's `val` (non-zero bits only)
//   110 | (k << 32) | opt           override k (k < 2^29), value tok as u64
//   111                             the counts, value (tokens << 32) | overrides
// A token's width is its segment's, so (headers, val positions) identify every float of every token: two rows
// whose tails or overrides differ in any bit, seg, slot or count feed different inputs.
const TAIL_TOKEN: u64 = 0x8000_0000_0000_0000;
const TAIL_FLOAT: u64 = 0x4000_0000_0000_0000;
const TAIL_OVERRIDE: u64 = 0xC000_0000_0000_0000;
const TAIL_COUNTS: u64 = 0xE000_0000_0000_0000;

/// Mix a sealed tail and overrides into an observation hash state (docs/search.md 2.15, K2): continues the
/// head's `(a, b)` lanes before `obs_finish`. An empty tail with no overrides mixes nothing (K1: the key of
/// before).
pub fn tail_mix(a: &mut u64, b: &mut u64, seg: &[u16], slot: &[u32], val: &[f32], ov: &[(u32, i64)]) {
    debug_assert_eq!(seg.len(), slot.len());
    if seg.is_empty() && ov.is_empty() {
        return;
    }
    use crate::lookahead::obs_mix as mix;
    for (t, (&s, &l)) in seg.iter().zip(slot).enumerate() {
        mix(a, b, TAIL_TOKEN | t as u64, ((s as u64) << 32) | l as u64);
    }
    for (p, v) in val.iter().enumerate() {
        let bits = v.to_bits();
        if bits != 0 {
            mix(a, b, TAIL_FLOAT | p as u64, bits as u64);
        }
    }
    for (k, &(opt, tok)) in ov.iter().enumerate() {
        assert!(k < 1 << 29, "tail_mix: {} overrides", ov.len());
        mix(a, b, TAIL_OVERRIDE | ((k as u64) << 32) | opt as u64, tok as u64);
    }
    mix(a, b, TAIL_COUNTS, ((seg.len() as u64) << 32) | ov.len() as u64);
}

#[cfg(test)]
mod tests {
    use super::*;

    static SEGS: [TailSeg; 2] = [TailSeg { seg: 0, name: "items", width: 3, head: 2 }, TailSeg { seg: 2, name: "marks", width: 1, head: 0 }];

    #[test]
    fn seal_sorts_tokens_with_their_floats() {
        let mut t = TailBuf::new(&SEGS);
        t.token(2, 5)[0] = 5.0;
        t.token(0, 3).copy_from_slice(&[1.0, 2.0, 3.0]);
        t.token(0, 2).copy_from_slice(&[4.0, 5.0, 6.0]);
        t.token(2, 0)[0] = 7.0;
        t.seal();
        assert_eq!(t.seg(), &[0, 0, 2, 2]);
        assert_eq!(t.slot(), &[2, 3, 0, 5]);
        assert_eq!(t.val(), &[4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 7.0, 5.0]);
        t.clear();
        assert!(t.is_empty());
        t.seal();
    }

    #[test]
    #[should_panic(expected = "duplicate tail token")]
    fn seal_refuses_duplicates() {
        let mut t = TailBuf::new(&SEGS);
        t.token(0, 3);
        t.token(2, 1);
        t.token(0, 3);
        t.seal();
    }

    #[test]
    #[should_panic(expected = "duplicate tail token")]
    fn seal_refuses_adjacent_duplicates() {
        let mut t = TailBuf::new(&SEGS);
        t.token(0, 3);
        t.token(0, 3);
        t.seal();
    }

    #[test]
    #[should_panic(expected = "inside the segment's 2 head slots")]
    fn token_refuses_head_slots() {
        TailBuf::new(&SEGS).token(0, 1);
    }

    #[test]
    #[should_panic(expected = "not a tail segment")]
    fn token_refuses_other_segments() {
        TailBuf::new(&SEGS).token(1, 9);
    }

    #[test]
    fn pack_round_trips() {
        for &(s, c, l) in &[(0u16, 0u8, 0u32), (7, 3, 99), (u16::MAX, u8::MAX, u32::MAX), (1, 0, 1 << 31)] {
            let t = pack_tok(s, c, l);
            assert!(t >= 0);
            assert_eq!(unpack_tok(t), (s, c, l));
        }
        assert_eq!(pack_tok(1, 2, 3), (1 << 40) | (2 << 32) | 3);
    }

    #[test]
    fn tail_mix_empty_is_a_no_op() {
        let (mut a, mut b) = (1u64, 2u64);
        tail_mix(&mut a, &mut b, &[], &[], &[], &[]);
        assert_eq!((a, b), (1, 2));
    }
}
