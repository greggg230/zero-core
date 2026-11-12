//! Evaluators of the driver's network batches: an ONNX Runtime session on a `zero-onnx/1` graph
//! (zero-core `python/zero/onnx_export.py`; zero-farm docs/phase0a-onnx.md: the interface) with the
//! generic post-processing every zero-core evaluator does, and the uniform evaluator
//! (`infer.UniformEvaluator`, exact arithmetic: the plumbing gate's).
//!
//! The ORT evaluator (`OrtEvaluator`) answers a sparse batch (`DriverCfg::sparse`): a batch of more
//! than `CHUNK` rows is run in pieces of at most `CHUNK` rows, the rows ordered by their non-zero
//! count (stable), each piece's outputs put back in row order (zero-core `onnx_parity.run_chunked`,
//! needing nothing of the network); then per row with options a float32 max-subtracted softmax of its
//! legal options' logits ([`softmax_exact`], the same bits as zero-core's `onnx_parity.exact_softmax`:
//! zero-farm's eval gate compares games byte for byte), value and margin as the graph gives them, and the
//! auxiliary outputs of the root belief rows (`ROW_BELIEF`) only, in row order.
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
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

/// ONNX Runtime's settings for a session.
#[derive(Clone, Debug)]
pub struct OrtCfg {
    /// The ONNX Runtime library (load-dynamic).
    pub dylib: std::path::PathBuf,
    pub intra_threads: usize,
    /// `session.intra_op.allow_spinning` (ORT's default: on).
    pub spinning: bool,
}

static ENV: OnceLock<Result<ort::environment::Environment, String>> = OnceLock::new();

/// The MSVC runtime ONNX Runtime's Windows build imports (msvcp140, msvcp140_1, vcruntime140,
/// vcruntime140_1), in load order (each after what it imports).
pub const MSVC_RUNTIME: &[&str] = &["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll"];

/// Windows: the MSVC runtime DLLs found next to the ONNX Runtime library (a runtime pack's app-local
/// copies) loaded first, by full path, so ONNX Runtime binds to them and not to whatever older copy the
/// PC's system directory holds (an ONNX Runtime built with toolset 14.4x crashes on an older
/// msvcp140's `_Mtx_lock`). The worker itself links the CRT statically (`+crt-static`), so nothing has
/// loaded another copy before. -> the files loaded.
#[cfg(windows)]
fn preload_msvc_runtime(dir: &Path) -> Result<Vec<String>, String> {
    extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut core::ffi::c_void;
        fn GetLastError() -> u32;
    }
    let mut out = Vec::new();
    for n in MSVC_RUNTIME {
        let p = dir.join(n);
        if !p.is_file() {
            continue;
        }
        let w: Vec<u16> = p.as_os_str().to_string_lossy().encode_utf16().chain([0]).collect();
        if unsafe { LoadLibraryW(w.as_ptr()) }.is_null() {
            return Err(format!("{}: LoadLibrary failed (error {})", p.display(), unsafe { GetLastError() }));
        }
        out.push(p.display().to_string());
    }
    Ok(out)
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

/// The process's ONNX Runtime environment, the library loaded from `dylib` (once; a second call with
/// another library is an error). On Windows the MSVC runtime next to it is loaded first
/// (`preload_msvc_runtime`).
pub fn ort_env(dylib: &Path) -> Result<&'static ort::environment::Environment, String> {
    let r = ENV.get_or_init(|| {
        #[cfg(windows)]
        if let Some(d) = dylib.parent().filter(|d| !d.as_os_str().is_empty()) {
            preload_msvc_runtime(d)?;
        }
        let b = ort::init_from(dylib).map_err(|e| format!("ONNX Runtime {}: {e}", dylib.display()))?;
        b.with_name("zero-worker").build().map_err(|e| format!("ONNX Runtime environment: {e}"))
    });
    r.as_ref().map_err(|e| e.clone())
}

/// The loaded ONNX Runtime's version string.
pub fn ort_version() -> String {
    ort::info().to_string()
}

pub struct OrtEvaluator {
    path: std::path::PathBuf,
    cfg: OrtCfg,
    session: Option<Session>,
    pub meta: Vec<(String, String)>,
    actions: usize,
    aux_w: usize,
    /// Calls, rows, seconds in session.run (and the post-processing), per batch durations (ms).
    pub calls: u64,
    pub rows: u64,
    pub run_s: f64,
    pub batch_ms: Vec<f32>,
    pub load_s: f64,
}

fn oerr(e: impl std::fmt::Display) -> String {
    format!("ONNX Runtime: {e}")
}

impl OrtEvaluator {
    /// Load `path` and check its metadata against `want`.
    pub fn load(path: &Path, cfg: OrtCfg, want: &Expect) -> Result<OrtEvaluator, String> {
        let t0 = Instant::now();
        let session = Self::session(path, &cfg)?;
        let md = session.metadata().map_err(oerr)?;
        let mut meta = Vec::new();
        for k in md.custom_keys().map_err(oerr)? {
            let v = md.custom(&k).unwrap_or_default();
            meta.push((k, v));
        }
        drop(md);
        let get = |k: &str| meta.iter().find(|e| e.0 == k).map(|e| e.1.as_str());
        let num = |k: &str| -> Result<usize, String> { get(k).ok_or_else(|| format!("{}: no metadata {k:?} (not a zero-onnx/1 graph)", path.display()))?.parse().map_err(|_| format!("{}: metadata {k} {:?}", path.display(), get(k))) };
        if get("zero_onnx") != Some("1") {
            return Err(format!("{}: zero_onnx {:?}: this worker reads zero-onnx/1", path.display(), get("zero_onnx")));
        }
        if get("game") != Some(want.game.as_str()) {
            return Err(format!("{}: a network of game {:?}, the job plays {:?}", path.display(), get("game"), want.game));
        }
        let (f, a, w) = (num("features")?, num("actions")?, num("aux_width")?);
        if f != want.features {
            return Err(format!("{}: {f} features, the game's driver feeds {} (another encoding)", path.display(), want.features));
        }
        if a != want.actions {
            return Err(format!("{}: {a} actions, the game has {}", path.display(), want.actions));
        }
        if w != 0 && w != want.aux_width {
            return Err(format!("{}: auxiliary width {w}, the game's targets are {} wide", path.display(), want.aux_width));
        }
        if want.belief && w != want.aux_width {
            return Err(format!("{}: a seat takes the root's belief but the network has no auxiliary head for the game's targets (aux_width {w}, need {}): HeadMissing", path.display(), want.aux_width));
        }
        for name in ["x_ptr", "x_idx", "x_val", "n_players", "belief"] {
            if !session.inputs().iter().any(|i| i.name() == name) {
                return Err(format!("{}: no input {name:?}", path.display()));
            }
        }
        for name in ["policy", "value", "margin", "aux"] {
            if !session.outputs().iter().any(|o| o.name() == name) {
                return Err(format!("{}: no output {name:?}", path.display()));
            }
        }
        Ok(OrtEvaluator { path: path.to_path_buf(), cfg, session: Some(session), meta, actions: a, aux_w: w, calls: 0, rows: 0, run_s: 0.0, batch_ms: Vec::new(), load_s: t0.elapsed().as_secs_f64() })
    }

    fn session(path: &Path, cfg: &OrtCfg) -> Result<Session, String> {
        let env = ort_env(&cfg.dylib)?;
        let b = Session::builder(env).map_err(oerr)?;
        let b = b.with_intra_threads(cfg.intra_threads.max(1)).map_err(oerr)?;
        let b = b.with_inter_threads(1).map_err(oerr)?;
        let b = b.with_parallel_execution(false).map_err(oerr)?;
        let b = b.with_optimization_level(GraphOptimizationLevel::All).map_err(oerr)?;
        let mut b = b.with_intra_op_spinning(cfg.spinning).map_err(oerr)?;
        b.commit_from_file(path).map_err(|e| format!("{}: {}", path.display(), oerr(e)))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The graph's outputs for rows `idx` of `b` (a piece; offsets rebased) -> (policy [n, A], value
    /// [n, 8], margin [n, 8], aux [n, W]).
    #[allow(clippy::type_complexity)]
    fn run_piece(&mut self, b: &Batch, idx: &[usize]) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let n = idx.len();
        let mut x_ptr = Vec::with_capacity(n + 1);
        x_ptr.push(0i64);
        let nnz: usize = idx.iter().map(|&r| (b.xs_off[r + 1] - b.xs_off[r]) as usize).sum();
        let mut x_idx = Vec::with_capacity(nnz);
        let mut x_val = Vec::with_capacity(nnz);
        for &r in idx {
            let (lo, hi) = (b.xs_off[r] as usize, b.xs_off[r + 1] as usize);
            for k in lo..hi {
                let i = b.xs_idx[k];
                x_idx.push(i32::try_from(i).map_err(|_| format!("feature index {i} beyond i32"))?);
            }
            x_val.extend_from_slice(&b.xs_val[lo..hi]);
            x_ptr.push(x_idx.len() as i64);
        }
        let np: Vec<i64> = idx.iter().map(|&r| b.n_players[r] as i64).collect();
        let bel: Vec<bool> = idx.iter().map(|&r| b.extra[r] == ROW_BELIEF).collect();
        let s = self.session.as_mut().ok_or("the ONNX Runtime session was released (yield) and not rebuilt")?;
        let inputs = ort::inputs![
            "x_ptr" => Tensor::from_array((vec![n as i64 + 1], x_ptr)).map_err(oerr)?,
            "x_idx" => Tensor::from_array((vec![nnz as i64], x_idx)).map_err(oerr)?,
            "x_val" => Tensor::from_array((vec![nnz as i64], x_val)).map_err(oerr)?,
            "n_players" => Tensor::from_array((vec![n as i64], np)).map_err(oerr)?,
            "belief" => Tensor::from_array((vec![n as i64], bel)).map_err(oerr)?,
        ];
        let out = s.run(inputs).map_err(oerr)?;
        let take = |name: &str, w: usize| -> Result<Vec<f32>, String> {
            let (shape, data) = out[name].try_extract_tensor::<f32>().map_err(oerr)?;
            if shape.len() != 2 || shape[0] as usize != n || shape[1] as usize != w {
                return Err(format!("output {name}: shape {shape:?}, want [{n}, {w}]"));
            }
            Ok(data.to_vec())
        };
        Ok((take("policy", self.actions)?, take("value", MAX_SEATS)?, take("margin", MAX_SEATS)?, take("aux", self.aux_w)?))
    }

    /// The graph's raw outputs for a whole batch, in row order (chunked as the module docs say).
    #[allow(clippy::type_complexity)]
    pub fn raw(&mut self, b: &Batch) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), String> {
        if b.xs_off.len() != b.b + 1 {
            return Err("a dense network batch: the native worker's driver must be sparse (DriverCfg::sparse)".into());
        }
        let bb = b.b;
        if bb <= CHUNK {
            let idx: Vec<usize> = (0..bb).collect();
            return self.run_piece(b, &idx);
        }
        let mut order: Vec<usize> = (0..bb).collect();
        order.sort_by_key(|&r| b.xs_off[r + 1] - b.xs_off[r]);
        let (a, w) = (self.actions, self.aux_w);
        let mut pol = vec![0f32; bb * a];
        let mut val = vec![0f32; bb * MAX_SEATS];
        let mut mar = vec![0f32; bb * MAX_SEATS];
        let mut aux = vec![0f32; bb * w];
        for piece in order.chunks(CHUNK) {
            let (p, v, m, x) = self.run_piece(b, piece)?;
            for (k, &r) in piece.iter().enumerate() {
                pol[r * a..(r + 1) * a].copy_from_slice(&p[k * a..(k + 1) * a]);
                val[r * MAX_SEATS..(r + 1) * MAX_SEATS].copy_from_slice(&v[k * MAX_SEATS..(k + 1) * MAX_SEATS]);
                mar[r * MAX_SEATS..(r + 1) * MAX_SEATS].copy_from_slice(&m[k * MAX_SEATS..(k + 1) * MAX_SEATS]);
                aux[r * w..(r + 1) * w].copy_from_slice(&x[k * w..(k + 1) * w]);
            }
        }
        Ok((pol, val, mar, aux))
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

impl Evaluator for OrtEvaluator {
    fn evaluate(&mut self, b: &Batch) -> Result<Answer, String> {
        let t0 = Instant::now();
        let (p, v, m, x) = self.raw(b)?;
        let a = answer(b, self.actions, self.aux_w, &p, v, m, &x)?;
        let dt = t0.elapsed().as_secs_f64();
        self.calls += 1;
        self.rows += b.b as u64;
        self.run_s += dt;
        if self.batch_ms.len() < 1 << 20 {
            self.batch_ms.push((dt * 1e3) as f32);
        }
        Ok(a)
    }
    fn release(&mut self) {
        self.session = None;
    }
    fn restore(&mut self) -> Result<(), String> {
        if self.session.is_none() {
            self.session = Some(Self::session(&self.path, &self.cfg)?);
        }
        Ok(())
    }
    fn describe(&self) -> String {
        format!("onnx {} ({} intra-op threads, spinning {})", self.path.display(), self.cfg.intra_threads, self.cfg.spinning)
    }
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
