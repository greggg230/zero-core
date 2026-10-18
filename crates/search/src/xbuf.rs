//! Feature buffers of network batches (docs/search.md 2.2): the [B, F] matrix the driver hands
//! to the caller, as float32 or float16 (`DriverCfg::x_dtype`), in buffers recycled through a
//! per-driver pool.
//!
//! Why (Kingdom Builder xst, 19,870 features, batches of 256 = 20 MB of float32 per batch): a
//! fresh zeroed buffer per batch cost the fill jobs ~33 ms of page faults per batch and the
//! caller's thread ~3.7 ms to free it (Windows), on top of copying 20 MB into the pinned upload
//! buffer (~7 ms): the caller's single feed thread spent more than half its time moving bytes.
//! Pooled buffers are faulted in once; float16 halves every copy after the encoding (the
//! conversion runs in the parallel fill jobs). Training shards store features as float16, so
//! the network already learns from float16-rounded inputs; float32 stays the default and is
//! bit-exact with the game's encoding.
//!
//! The caller may donate buffers of its own (`XPool::donate`, e.g. pinned host memory from
//! torch: the fill jobs then write the features where the GPU upload reads them, and the feed
//! thread copies nothing). A donated buffer's owner (an opaque object, e.g. the Python array)
//! is kept alive as long as the pool or a batch holds the buffer.
use std::any::Any;
use std::sync::{Arc, Mutex};

/// Element type of a network batch's features.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum XDtype {
    /// The game's encoding as is.
    #[default]
    F32,
    /// IEEE half precision (round to nearest even, as numpy's `astype(np.float16)`), as bits.
    F16,
}

impl XDtype {
    /// 'f32' | 'f16' (also 'float32' / 'float16').
    pub fn parse(s: &str) -> Result<XDtype, String> {
        match s {
            "f32" | "float32" => Ok(XDtype::F32),
            "f16" | "float16" => Ok(XDtype::F16),
            _ => Err(format!("x_dtype {s:?}: 'f32' or 'f16'")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            XDtype::F32 => "f32",
            XDtype::F16 => "f16",
        }
    }
}

enum XData {
    F32(Vec<f32>),
    F16(Vec<u16>),
    Ext(ExtBuf),
}

/// A buffer owned by someone else (`XPool::donate`): `elems` elements of `dtype` at `ptr`,
/// valid while `owner` lives.
pub struct ExtBuf {
    ptr: *mut u8,
    elems: usize,
    dtype: XDtype,
    _owner: Box<dyn Any + Send + Sync>,
}

// SAFETY: the memory belongs to `_owner` (Send + Sync) and is only accessed through the pool's
// discipline: one XBuf at a time, the fill jobs writing disjoint rows.
unsafe impl Send for ExtBuf {}
unsafe impl Sync for ExtBuf {}

impl ExtBuf {
    /// # Safety
    /// `ptr` points to `elems` writable, suitably aligned elements of `dtype` that stay valid
    /// and are not accessed by anyone else while `owner` lives and the pool has the buffer.
    pub unsafe fn new(ptr: *mut u8, elems: usize, dtype: XDtype, owner: Box<dyn Any + Send + Sync>) -> ExtBuf {
        ExtBuf { ptr, elems, dtype, _owner: owner }
    }
}

impl XData {
    fn alloc(dtype: XDtype, n: usize) -> XData {
        match dtype {
            XDtype::F32 => XData::F32(vec![0.0; n]),
            XDtype::F16 => XData::F16(vec![0; n]),
        }
    }
    fn dtype(&self) -> XDtype {
        match self {
            XData::F32(_) => XDtype::F32,
            XData::F16(_) => XDtype::F16,
            XData::Ext(e) => e.dtype,
        }
    }
    fn cap(&self) -> usize {
        match self {
            XData::F32(v) => v.len(),
            XData::F16(v) => v.len(),
            XData::Ext(e) => e.elems,
        }
    }
}

/// Spare buffers of one driver: `elems` elements each (the largest batch: a group's slots x F),
/// at most `keep` own ones plus every donated one (more own ones are freed when returned).
pub struct XPool {
    dtype: XDtype,
    elems: usize,
    keep: usize,
    st: Mutex<PoolState>,
}

#[derive(Default)]
struct PoolState {
    free: Vec<XData>,
    donated: usize,
}

impl XPool {
    pub fn new(dtype: XDtype, elems: usize, keep: usize) -> Arc<XPool> {
        Arc::new(XPool { dtype, elems, keep, st: Mutex::new(PoolState::default()) })
    }

    pub fn dtype(&self) -> XDtype {
        self.dtype
    }
    /// Elements per buffer.
    pub fn elems(&self) -> usize {
        self.elems
    }
    /// Buffers worth donating: `keep` minus those donated so far.
    pub fn wanted(&self) -> usize {
        self.keep.saturating_sub(self.st.lock().unwrap().donated)
    }

    /// Add a buffer of the caller's (at least `elems` elements of the pool's dtype); batches
    /// prefer donated buffers.
    pub fn donate(&self, b: ExtBuf) -> Result<(), String> {
        if b.dtype != self.dtype || b.elems < self.elems {
            return Err(format!("donated feature buffer: {} x {} elements, the pool needs {} x {}", b.dtype.name(), b.elems, self.dtype.name(), self.elems));
        }
        let mut st = self.st.lock().unwrap();
        st.donated += 1;
        st.free.push(XData::Ext(b));
        Ok(())
    }

    /// A buffer for `len` elements (contents unspecified: the caller writes all of them).
    pub fn take(self: &Arc<XPool>, len: usize) -> XBuf {
        if len > self.elems {
            return XBuf { data: XData::alloc(self.dtype, len), len, pool: None };
        }
        let data = {
            let mut st = self.st.lock().unwrap();
            match st.free.iter().rposition(|d| matches!(d, XData::Ext(_))) {
                Some(i) => Some(st.free.swap_remove(i)),
                None => st.free.pop(),
            }
        };
        let data = data.unwrap_or_else(|| XData::alloc(self.dtype, self.elems));
        XBuf { data, len, pool: Some(self.clone()) }
    }

    fn put(&self, data: XData) {
        if data.dtype() == self.dtype && data.cap() >= self.elems {
            let mut st = self.st.lock().unwrap();
            let own = st.free.iter().filter(|d| !matches!(d, XData::Ext(_))).count();
            if matches!(data, XData::Ext(_)) || own < self.keep {
                st.free.push(data);
                return;
            }
        }
        drop(data); // outside the lock (a donated owner may take a while to drop)
    }

    /// Spare buffers now (tests).
    pub fn spare(&self) -> usize {
        self.st.lock().unwrap().free.len()
    }
}

/// The features of one network batch: `len` elements of a (pooled) buffer; dropping it returns
/// the buffer to its pool.
pub struct XBuf {
    data: XData,
    len: usize,
    pool: Option<Arc<XPool>>,
}

impl XBuf {
    /// A float32 buffer outside any pool.
    pub fn from_f32(v: Vec<f32>) -> XBuf {
        XBuf { len: v.len(), data: XData::F32(v), pool: None }
    }
    pub fn dtype(&self) -> XDtype {
        self.data.dtype()
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn as_f32(&self) -> Option<&[f32]> {
        match &self.data {
            XData::F32(v) => Some(&v[..self.len]),
            // SAFETY: see ExtBuf::new (the buffer is this XBuf's while it lives)
            XData::Ext(e) if e.dtype == XDtype::F32 => Some(unsafe { std::slice::from_raw_parts(e.ptr as *const f32, self.len) }),
            _ => None,
        }
    }
    /// The float16 bit patterns.
    pub fn as_f16_bits(&self) -> Option<&[u16]> {
        match &self.data {
            XData::F16(v) => Some(&v[..self.len]),
            // SAFETY: as in as_f32
            XData::Ext(e) if e.dtype == XDtype::F16 => Some(unsafe { std::slice::from_raw_parts(e.ptr as *const u16, self.len) }),
            _ => None,
        }
    }
    /// The values as float32 (float16 widened exactly).
    pub fn to_f32(&self) -> Vec<f32> {
        match (self.as_f32(), self.as_f16_bits()) {
            (Some(v), _) => v.to_vec(),
            (_, Some(v)) => v.iter().map(|&h| f16_to_f32(h)).collect(),
            _ => unreachable!(),
        }
    }
    /// In a donated buffer.
    pub fn donated(&self) -> bool {
        matches!(self.data, XData::Ext(_))
    }
    /// Raw destination for the fill jobs.
    pub(crate) fn ptr(&mut self) -> XPtr {
        match &mut self.data {
            XData::F32(v) => XPtr::F32(v.as_mut_ptr()),
            XData::F16(v) => XPtr::F16(v.as_mut_ptr()),
            XData::Ext(e) => match e.dtype {
                XDtype::F32 => XPtr::F32(e.ptr as *mut f32),
                XDtype::F16 => XPtr::F16(e.ptr as *mut u16),
            },
        }
    }
}

impl Default for XBuf {
    fn default() -> XBuf {
        XBuf::from_f32(Vec::new())
    }
}

impl Clone for XBuf {
    /// A copy outside the pool.
    fn clone(&self) -> XBuf {
        let data = match (self.as_f32(), self.as_f16_bits()) {
            (Some(v), _) => XData::F32(v.to_vec()),
            (_, Some(v)) => XData::F16(v.to_vec()),
            _ => unreachable!(),
        };
        XBuf { data, len: self.len, pool: None }
    }
}

impl std::fmt::Debug for XBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "XBuf({}, {} elements, pooled {}, donated {})", self.dtype().name(), self.len, self.pool.is_some(), self.donated())
    }
}

impl Drop for XBuf {
    fn drop(&mut self) {
        if let Some(p) = self.pool.take() {
            p.put(std::mem::replace(&mut self.data, XData::F32(Vec::new())));
        }
    }
}

/// Where a fill job writes its rows (see `driver::Dest`).
#[derive(Clone, Copy)]
pub(crate) enum XPtr {
    F32(*mut f32),
    F16(*mut u16),
}

impl XPtr {
    /// Write `src` at element offset `off`.
    ///
    /// # Safety
    /// `off .. off + src.len()` lies inside the buffer and no one else touches it meanwhile.
    pub(crate) unsafe fn write(self, off: usize, src: &[f32]) {
        match self {
            XPtr::F32(p) => std::slice::from_raw_parts_mut(p.add(off), src.len()).copy_from_slice(src),
            XPtr::F16(p) => f32_to_f16_slice(src, std::slice::from_raw_parts_mut(p.add(off), src.len())),
        }
    }
}

/// float32 -> float16 bits, round to nearest even (NaN stays NaN, overflow -> infinity).
pub fn f32_to_f16(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 | (man >> 13) as u16 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1u32 << shift) - 1);
        let mut r = m >> shift;
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

/// float16 bits -> float32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // subnormal: normalize
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// `dst[i] = f32_to_f16(src[i])` (F16C on x86-64 when the CPU has it: same results).
pub fn f32_to_f16_slice(src: &[f32], dst: &mut [u16]) {
    assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("f16c") && std::arch::is_x86_feature_detected!("avx") {
            // SAFETY: the CPU has the features (checked above).
            unsafe { f32_to_f16_f16c(src, dst) };
            return;
        }
    }
    f32_to_f16_soft(src, dst);
}

/// The portable conversion (tests compare it with F16C).
pub fn f32_to_f16_soft(src: &[f32], dst: &mut [u16]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = f32_to_f16(s);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn f32_to_f16_f16c(src: &[f32], dst: &mut [u16]) {
    use std::arch::x86_64::{__m128i, _mm256_cvtps_ph, _mm256_loadu_ps, _mm_storeu_si128, _MM_FROUND_TO_NEAREST_INT};
    let n = src.len();
    let mut i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(src.as_ptr().add(i));
        let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v);
        _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, h);
        i += 8;
    }
    for j in i..n {
        dst[j] = f32_to_f16(src[j]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn known_values() {
        let cases: &[(f32, u16)] = &[
            (0.0, 0x0000),
            (-0.0, 0x8000),
            (1.0, 0x3c00),
            (-2.0, 0xc000),
            (0.5, 0x3800),
            (65504.0, 0x7bff),
            (65520.0, 0x7c00),            // rounds up to infinity
            (1e9, 0x7c00),
            (f32::INFINITY, 0x7c00),
            (f32::NEG_INFINITY, 0xfc00),
            (6.103515625e-05, 0x0400),    // smallest normal
            (5.960464477539063e-08, 0x0001), // smallest subnormal
            (2.9802322387695312e-08, 0x0000), // half of it: ties to even (0)
            (4.470348358154297e-08, 0x0001),  // 0.75 of it: rounds up
            (1.0009765625, 0x3c01),
            (1.00048828125, 0x3c00),      // tie between 0x3c00 and 0x3c01: even
            (1.00146484375, 0x3c02),      // tie between 0x3c01 and 0x3c02: even
            (0.1, 0x2e66),
            (1.0 / 3.0, 0x3555),
        ];
        for &(v, h) in cases {
            assert_eq!(f32_to_f16(v), h, "{v}");
        }
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
        for h in 0..=u16::MAX {
            let v = f16_to_f32(h);
            if v.is_nan() {
                assert!(f16_to_f32(f32_to_f16(v)).is_nan());
            } else {
                assert_eq!(f32_to_f16(v), h, "round trip of {h:#06x}");
            }
        }
    }

    #[test]
    fn slice_matches_scalar() {
        let mut rng = Rng::new(5);
        let mut src: Vec<f32> = (0..100_003).map(|_| f32::from_bits(rng.next_u64() as u32)).collect();
        src.extend((0..10_000).map(|i| i as f32 / 7.0 - 300.0));
        src.extend([0.0, -0.0, 1.0, 0.5, 65519.0, 65520.0, 1e-8, f32::INFINITY]);
        let mut a = vec![0u16; src.len()];
        let mut b = vec![0u16; src.len()];
        f32_to_f16_slice(&src, &mut a);
        f32_to_f16_soft(&src, &mut b);
        for i in 0..src.len() {
            let (fa, fb) = (f16_to_f32(a[i]), f16_to_f32(b[i]));
            assert!(a[i] == b[i] || (fa.is_nan() && fb.is_nan()), "{} -> {:#06x} vs {:#06x}", src[i], a[i], b[i]);
        }
    }

    #[test]
    fn pool_recycles() {
        let p = XPool::new(XDtype::F16, 100, 2);
        let mut a = p.take(60);
        let ptr = match a.ptr() {
            XPtr::F16(q) => q as usize,
            XPtr::F32(_) => unreachable!(),
        };
        assert_eq!((a.len(), a.dtype()), (60, XDtype::F16));
        drop(a);
        assert_eq!(p.spare(), 1);
        let mut b = p.take(100);
        assert_eq!(p.spare(), 0);
        assert!(matches!(b.ptr(), XPtr::F16(q) if q as usize == ptr));
        let c = p.take(10);
        let d = p.take(10);
        let big = p.take(101);              // larger than the pool's buffers: not pooled
        drop((b, c, d, big));
        assert_eq!(p.spare(), 2);           // at most `keep`
        let e = p.take(5).clone();          // a clone is outside the pool
        drop(e);
        assert_eq!(p.spare(), 2);
    }

    #[test]
    fn donated_buffers_are_preferred_and_kept() {
        let p = XPool::new(XDtype::F32, 8, 1);
        assert_eq!(p.wanted(), 1);
        let mut mem: Vec<Box<[f32; 12]>> = (0..3).map(|_| Box::new([0.0f32; 12])).collect();
        let ptrs: Vec<usize> = mem.iter_mut().map(|b| b.as_mut_ptr() as usize).collect();
        let owners = Arc::new(());
        for (i, b) in mem.into_iter().enumerate() {
            let ptr = ptrs[i] as *mut u8;
            // SAFETY: the box (the owner) holds 12 f32s, nobody else touches them
            let e = unsafe { ExtBuf::new(ptr, if i == 2 { 4 } else { 12 }, XDtype::F32, Box::new((b, owners.clone()))) };
            let r = p.donate(e);
            assert_eq!(r.is_ok(), i < 2, "buffer {i}");
        }
        assert_eq!((p.wanted(), p.spare()), (0, 2));
        assert_eq!(Arc::strong_count(&owners), 3); // the too-small one was dropped
        let mut a = p.take(8);
        let mut b = p.take(8);
        let mut c = p.take(8); // no donated one left: an own buffer
        assert!(a.donated() && b.donated() && !c.donated());
        for (k, x) in [&mut a, &mut b, &mut c].into_iter().enumerate() {
            let src: Vec<f32> = (0..8).map(|i| (k * 10 + i) as f32).collect();
            // SAFETY: 8 elements fit every buffer
            unsafe { x.ptr().write(0, &src) };
            assert_eq!(x.to_f32(), src);
        }
        assert!(matches!(a.ptr(), XPtr::F32(q) if ptrs.contains(&(q as usize))));
        drop((a, b, c));
        assert_eq!(p.spare(), 3); // donated ones always come back
        assert!(p.take(8).donated());
        drop(p);
        assert_eq!(Arc::strong_count(&owners), 1); // the pool released its owners
    }
}
