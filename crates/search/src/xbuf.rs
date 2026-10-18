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
        }
    }
    fn cap(&self) -> usize {
        match self {
            XData::F32(v) => v.len(),
            XData::F16(v) => v.len(),
        }
    }
}

/// Spare buffers of one driver: `elems` elements each (the largest batch: a group's slots x F),
/// at most `keep` of them (more are freed when returned).
pub struct XPool {
    dtype: XDtype,
    elems: usize,
    keep: usize,
    free: Mutex<Vec<XData>>,
}

impl XPool {
    pub fn new(dtype: XDtype, elems: usize, keep: usize) -> Arc<XPool> {
        Arc::new(XPool { dtype, elems, keep, free: Mutex::new(Vec::new()) })
    }

    pub fn dtype(&self) -> XDtype {
        self.dtype
    }

    /// A buffer for `len` elements (contents unspecified: the caller writes all of them).
    pub fn take(self: &Arc<XPool>, len: usize) -> XBuf {
        if len > self.elems {
            return XBuf { data: XData::alloc(self.dtype, len), len, pool: None };
        }
        let data = self.free.lock().unwrap().pop().unwrap_or_else(|| XData::alloc(self.dtype, self.elems));
        XBuf { data, len, pool: Some(self.clone()) }
    }

    fn put(&self, data: XData) {
        if data.dtype() == self.dtype && data.cap() >= self.elems {
            let mut f = self.free.lock().unwrap();
            if f.len() < self.keep {
                f.push(data);
            }
        }
    }

    /// Spare buffers now (tests).
    pub fn spare(&self) -> usize {
        self.free.lock().unwrap().len()
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
            XData::F16(_) => None,
        }
    }
    /// The float16 bit patterns.
    pub fn as_f16_bits(&self) -> Option<&[u16]> {
        match &self.data {
            XData::F16(v) => Some(&v[..self.len]),
            XData::F32(_) => None,
        }
    }
    /// The values as float32 (float16 widened exactly).
    pub fn to_f32(&self) -> Vec<f32> {
        match &self.data {
            XData::F32(v) => v[..self.len].to_vec(),
            XData::F16(v) => v[..self.len].iter().map(|&h| f16_to_f32(h)).collect(),
        }
    }
    /// Raw destination for the fill jobs.
    pub(crate) fn ptr(&mut self) -> XPtr {
        match &mut self.data {
            XData::F32(v) => XPtr::F32(v.as_mut_ptr()),
            XData::F16(v) => XPtr::F16(v.as_mut_ptr()),
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
        let data = match &self.data {
            XData::F32(v) => XData::F32(v[..self.len].to_vec()),
            XData::F16(v) => XData::F16(v[..self.len].to_vec()),
        };
        XBuf { data, len: self.len, pool: None }
    }
}

impl std::fmt::Debug for XBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "XBuf({}, {} elements, pooled {})", self.dtype().name(), self.len, self.pool.is_some())
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
}
