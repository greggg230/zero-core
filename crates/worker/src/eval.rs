//! Evaluators of the driver's network batches - the evaluator trait, the generic post-processing every
//! zero-core evaluator does, and the uniform evaluator (`infer.UniformEvaluator`, exact arithmetic: the
//! plumbing gate's). The ONNX Runtime evaluator of a `zero-onnx/1` graph is `crate::session`'s
//! `InferenceSession` (zero-farm PROTOCOL.md 8: one session factory, any execution provider).
//!
//! An ONNX Runtime evaluator answers a sparse batch (`DriverCfg::sparse`): a batch of more than `CHUNK`
//! rows (on the CPU; a GPU provider takes whole batches, `session::GPU_CHUNK`) is run in pieces, the rows
//! ordered by their non-zero count (stable), each piece's outputs put back in row order (zero-core
//! `onnx_parity.run_chunked`, needing nothing of the network); then per row with options a float32
//! max-subtracted softmax of its legal options' logits ([`softmax_exact`], the same bits as zero-core's
//! `onnx_parity.exact_softmax`: zero-farm's eval gate compares games byte for byte), value and margin as the
//! graph gives them, and the auxiliary outputs of the root belief rows (`ROW_BELIEF`) only, in row order.
use zero_search::driver::{Batch, ROW_BELIEF};
use zero_search::MAX_SEATS;

/// Rows of one piece of a big batch (zero-farm phase 0a: pieces of <= 64 rows grouped by non-zero count).
pub const CHUNK: usize = 64;

/// An evaluator's answer to a network batch (`DynDriver::submit`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Answer {
    pub priors: Vec<f32>,
    pub value: Vec<f32>,
    pub margin: Vec<f32>,
    pub aux: Vec<f32>,
}

pub trait Evaluator: Send {
    fn evaluate(&mut self, b: &Batch) -> Result<Answer, String>;
    /// Drop whatever the evaluator holds that can be rebuilt (an ORT session: its memory; `yield`).
    fn release(&mut self) {}
    /// Rebuild what `release` dropped.
    fn restore(&mut self) -> Result<(), String> {
        Ok(())
    }
    fn describe(&self) -> String;
    /// The execution provider answering (Stats `provider`: "cpu", "dml", ...; "uniform" for no network).
    fn provider(&self) -> String {
        "cpu".into()
    }
}

/// Uniform priors, value 1/n for every seat present, margin 0 (`infer.UniformEvaluator`, the same
/// float32 values; it has no auxiliary head: a batch with root belief rows is refused).
pub struct Uniform;

impl Evaluator for Uniform {
    fn evaluate(&mut self, b: &Batch) -> Result<Answer, String> {
        if b.extra.contains(&ROW_BELIEF) {
            return Err("the uniform evaluator has no auxiliary head: a batch with root belief rows (a seat's belief) needs a network".into());
        }
        let mut a = Answer { value: vec![0.0; b.b * MAX_SEATS], margin: vec![0.0; b.b * MAX_SEATS], ..Default::default() };
        for r in 0..b.b {
            let k = (b.opt_off[r + 1] - b.opt_off[r]) as usize;
            let p = (1.0f64 / k.max(1) as f64) as f32;
            a.priors.extend(std::iter::repeat_n(p, k));
            let n = b.n_players[r] as f32;
            for j in 0..(b.n_players[r] as usize).min(MAX_SEATS) {
                a.value[r * MAX_SEATS + j] = 1.0f32 / n;
            }
        }
        Ok(a)
    }
    fn describe(&self) -> String {
        "uniform".into()
    }
    fn provider(&self) -> String {
        "uniform".into()
    }
}

/// What a `zero-onnx/1` graph must declare for a run (phase 0a's "For phase 0b" point 1).
#[derive(Clone, Debug)]
pub struct Expect {
    pub game: String,
    pub features: usize,
    pub actions: usize,
    /// The game's auxiliary width (`SearchGame::AUX`) and whether a seat takes the root belief (the
    /// graph must then answer it: its aux width = the game's).
    pub aux_width: usize,
    pub belief: bool,
}

/// ONNX Runtime's settings for a job's sessions (`crate::session::create_session` builds them).
#[derive(Clone, Debug)]
pub struct OrtCfg {
    /// The ONNX Runtime library (load-dynamic; the runtime pack's).
    pub dylib: std::path::PathBuf,
    pub intra_threads: usize,
    /// `session.intra_op.allow_spinning` (ORT's default: on).
    pub spinning: bool,
    /// The job line's `providers` (preference order, the CPU last; zero-farm PROTOCOL.md 8).
    pub providers: Vec<crate::session::ProviderKind>,
    /// The job line's `gpu`: the adapter GPU providers run on.
    pub device: Option<crate::session::GpuDevice>,
}

impl OrtCfg {
    /// CPU only (the job lines before PROTOCOL.md rev 7, the tools).
    pub fn cpu(dylib: std::path::PathBuf, intra_threads: usize, spinning: bool) -> OrtCfg {
        OrtCfg { dylib, intra_threads, spinning, providers: vec![crate::session::ProviderKind::Cpu], device: None }
    }
}

/// The full path of a loaded module, by its file name (Windows; else None).
pub fn module_path(name: &str) -> Option<String> {
    #[cfg(windows)]
    {
        extern "system" {
            fn GetModuleHandleW(name: *const u16) -> *mut core::ffi::c_void;
            fn GetModuleFileNameW(h: *mut core::ffi::c_void, buf: *mut u16, n: u32) -> u32;
        }
        let w: Vec<u16> = name.encode_utf16().chain([0]).collect();
        let h = unsafe { GetModuleHandleW(w.as_ptr()) };
        if h.is_null() {
            return None;
        }
        let mut buf = vec![0u16; 1024];
        let n = unsafe { GetModuleFileNameW(h, buf.as_mut_ptr(), buf.len() as u32) } as usize;
        (n > 0).then(|| String::from_utf16_lossy(&buf[..n]))
    }
    #[cfg(not(windows))]
    {
        let _ = name;
        None
    }
}

// The exact exponential of the softmax (`exp_f32`): exp of a float32 through float64 arithmetic of a fixed
// sequence of IEEE operations (Cody-Waite reduction by ln 2, the Taylor polynomial of degree 13 by Horner,
// an exact scaling by 2^n, one rounding to float32) - no libm, no FMA - so that every machine and numpy
// (zero-core `onnx_parity.exp_f32`) compute the same bits. Its error before the final rounding is ~5e-16
// relative: the float32 result is the correctly rounded exp except within ~1e-15 of a rounding boundary.
const INV_LN2: f64 = f64::from_bits(0x3ff7_1547_652b_82fe);
/// ln 2 split (fdlibm's ln2_hi / ln2_lo): n * LN2_HI is exact for |n| < 2^21.
const LN2_HI: f64 = f64::from_bits(0x3fe6_2e42_fee0_0000);
const LN2_LO: f64 = f64::from_bits(0x3dea_39ef_3579_3c76);
/// 1 / k!, k = 0..=13 (each the correctly rounded quotient).
const EXP_C: [f64; 14] = [
    f64::from_bits(0x3ff0_0000_0000_0000),
    f64::from_bits(0x3ff0_0000_0000_0000),
    f64::from_bits(0x3fe0_0000_0000_0000),
    f64::from_bits(0x3fc5_5555_5555_5555),
    f64::from_bits(0x3fa5_5555_5555_5555),
    f64::from_bits(0x3f81_1111_1111_1111),
    f64::from_bits(0x3f56_c16c_16c1_6c17),
    f64::from_bits(0x3f2a_01a0_1a01_a01a),
    f64::from_bits(0x3efa_01a0_1a01_a01a),
    f64::from_bits(0x3ec7_1de3_a556_c734),
    f64::from_bits(0x3e92_7e4f_b778_9f5c),
    f64::from_bits(0x3e5a_e645_67f5_44e4),
    f64::from_bits(0x3e21_eed8_eff8_d898),
    f64::from_bits(0x3de6_1246_13a8_6d09),
];

/// exp(d) as float32 for a max-subtracted logit d <= 0 (module docs: the same bits everywhere). NaN stays
/// NaN; below -150 (far under float32's smallest subnormal) it is 0.
pub fn exp_f32(d: f32) -> f32 {
    if d.is_nan() {
        return f32::NAN;
    }
    let x = d as f64;
    if x < -150.0 {
        return 0.0;
    }
    let n = (x * INV_LN2 + 0.5).floor();
    let r = (x - n * LN2_HI) - n * LN2_LO;
    let mut p = EXP_C[13];
    for k in (0..13).rev() {
        p = p * r + EXP_C[k];
    }
    // 2^n exactly (n in [-217, 0] here; the result is a normal float64)
    (p * f64::from_bits(((n as i64 + 1023) as u64) << 52)) as f32
}

/// The softmax of one row's legal logits, in place (zero-core `onnx_parity.exact_softmax`): m the largest
/// (NaN propagates), each e = `exp_f32(l - m)` (the difference in float32), their sum in float32 in order,
/// each e / sum in float32.
pub fn softmax_exact(v: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &l in v.iter() {
        if l.is_nan() || l > m {
            m = l;
        }
    }
    let mut s = 0f32;
    for p in v.iter_mut() {
        *p = exp_f32(*p - m);
        s += *p;
    }
    for p in v.iter_mut() {
        *p /= s;
    }
}

/// The generic step (zero-core `onnx_parity.answer`): per row with options the float32 softmax of its
/// legal options' logits (`softmax_exact`), value and margin as given, the belief rows' aux in row order.
pub fn answer(b: &Batch, actions: usize, aux_w: usize, policy: &[f32], value: Vec<f32>, margin: Vec<f32>, aux: &[f32]) -> Result<Answer, String> {
    let mut priors = Vec::with_capacity(*b.opt_off.last().unwrap_or(&0) as usize);
    for r in 0..b.b {
        let (lo, hi) = (b.opt_off[r] as usize, b.opt_off[r + 1] as usize);
        if lo == hi {
            continue;
        }
        let start = priors.len();
        for k in lo..hi {
            let a = b.opt_action[k] as usize;
            if a >= actions {
                return Err(format!("row {r}: action {a} beyond the network's {actions}"));
            }
            priors.push(policy[r * actions + a]);
        }
        softmax_exact(&mut priors[start..]);
    }
    let mut ax = Vec::new();
    for r in 0..b.b {
        if b.extra[r] == ROW_BELIEF {
            if aux_w == 0 {
                return Err("a root belief row for a network without the auxiliary head (HeadMissing)".into());
            }
            ax.extend_from_slice(&aux[r * aux_w..(r + 1) * aux_w]);
        }
    }
    Ok(Answer { priors, value, margin, aux: ax })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_f32_is_the_rounded_exponential() {
        assert_eq!(exp_f32(0.0), 1.0);
        assert_eq!(exp_f32(-200.0), 0.0);
        assert_eq!(exp_f32(f32::NEG_INFINITY), 0.0);
        assert!(exp_f32(f32::NAN).is_nan());
        // against float64's exp rounded to float32: equal, or (within ~1e-15 of a rounding boundary) one ulp
        let mut worst = 0u32;
        let mut x = 0f32;
        while x > -110.0 {
            let want = (x as f64).exp() as f32;
            let got = exp_f32(x);
            let ulps = (want.to_bits() as i64 - got.to_bits() as i64).unsigned_abs() as u32;
            worst = worst.max(ulps);
            x -= 0.000_731;
        }
        assert!(worst <= 1, "{worst} ulps");
    }

    #[test]
    fn softmax_exact_sums_to_one_and_keeps_order() {
        let mut v = vec![1.5f32, -0.25, 3.0, 3.0, -40.0];
        softmax_exact(&mut v);
        let s: f32 = v.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "{s}");
        assert!(v[2] == v[3] && v[2] > v[0] && v[0] > v[1] && v[1] > v[4] && v[4] > 0.0);
        let mut n = vec![1.0f32, f32::NAN];
        softmax_exact(&mut n);
        assert!(n.iter().all(|x| x.is_nan()));
    }
}
