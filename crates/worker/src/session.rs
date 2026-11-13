//! The one ONNX Runtime session factory of a worker (zero-farm PROTOCOL.md 8, the GPU inference contract):
//! every inference session - self-play, eval, the self-test, the provider probe, the gate tools - is created
//! here, by [`create_session`], and nothing else in the worker creates one.
//!
//! [`create_session`] tries the job's providers in order (`providers` of the job line, `cpu` always last). A
//! provider is accepted only if its session builds and it passes the self-test on the network's reference set
//! (`reference.npz`, `selftest`): every answer within [`crate::selftest::TOL`] (2e-5) of the reference's, and
//! the whole set answered twice with bit-identical results. A GPU provider is never accepted without a
//! reference set. The nodes a provider leaves to the CPU (ONNX Runtime's node placement, read from its
//! verbose log while the session is built) are reported with the self-test. A provider that fails moves the
//! factory on to the next; a CPU session that fails the self-test is the job's end, as before.
//!
//! Providers and where they come from (the runtime pack, PROTOCOL.md 8 and DESIGN.md 8.5):
//! * `cpu`: every ONNX Runtime build.
//! * `dml` (DirectML): built into Windows ML's ONNX Runtime (and the DirectML builds); the adapter is chosen by
//!   its DXGI index. Any DirectX 12 GPU, Windows 10 1903 and later.
//! * `tensorrt_rtx`, `amdgpu`, `migraphx`, `openvino`: Windows ML's execution-provider catalog (Windows 11 24H2+;
//!   `crate::winml`), registered as plugin libraries and bound to the adapter by its LUID through ONNX
//!   Runtime's device list; or, for `tensorrt_rtx`, a runtime pack holding ONNX Runtime's own TensorRT-RTX
//!   build.
//! * `cuda`: a runtime pack holding ONNX Runtime's CUDA build and the CUDA libraries.
//!
//! GPU sessions are built with deterministic compute on, CUDA's TF32 off (float32 throughout); a provider
//! that drops precision fails the self-test and is not used. `release_device` (the agent's `yield`) drops
//! the session and with it every byte it holds on the GPU; the next `eval` rebuilds it with the provider
//! and adapter accepted before.
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use ort::logging::LogLevel;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;
use ort::value::Tensor;
use serde_json::{json, Value};
use zero_search::driver::{Batch, ROW_BELIEF};
use zero_search::MAX_SEATS;

use crate::eval::{answer, Answer, Evaluator, Expect, CHUNK};
use crate::gpu::{luid_hex, Adapter};

// ------------------------------------------------------------------------------------------------------------
// providers and devices

/// An ONNX Runtime execution provider (the protocol's names: [`ProviderKind::wire`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ProviderKind {
    Cpu,
    TensorRtRtx,
    Cuda,
    MiGraphX,
    AmdGpu,
    OpenVino,
    DirectMl,
}

impl ProviderKind {
    /// Every provider, in the default order of preference (the CPU last).
    pub const ALL: [ProviderKind; 7] = [
        ProviderKind::TensorRtRtx,
        ProviderKind::Cuda,
        ProviderKind::AmdGpu,
        ProviderKind::MiGraphX,
        ProviderKind::OpenVino,
        ProviderKind::DirectMl,
        ProviderKind::Cpu,
    ];

    pub fn wire(self) -> &'static str {
        match self {
            ProviderKind::Cpu => "cpu",
            ProviderKind::TensorRtRtx => "tensorrt_rtx",
            ProviderKind::Cuda => "cuda",
            ProviderKind::MiGraphX => "migraphx",
            ProviderKind::AmdGpu => "amdgpu",
            ProviderKind::OpenVino => "openvino",
            ProviderKind::DirectMl => "dml",
        }
    }

    pub fn parse(s: &str) -> Option<ProviderKind> {
        ProviderKind::ALL.into_iter().find(|p| p.wire() == s)
    }

    /// ONNX Runtime's names of the provider (plugin builds and Windows ML spell some differently).
    pub fn ort_names(self) -> &'static [&'static str] {
        match self {
            ProviderKind::Cpu => &["CPUExecutionProvider"],
            ProviderKind::TensorRtRtx => &["NvTensorRTRTXExecutionProvider", "NvTensorRtRtxExecutionProvider"],
            ProviderKind::Cuda => &["CUDAExecutionProvider"],
            ProviderKind::MiGraphX => &["MIGraphXExecutionProvider"],
            ProviderKind::AmdGpu => &["AMDGPUExecutionProvider"],
            ProviderKind::OpenVino => &["OpenVINOExecutionProvider"],
            ProviderKind::DirectMl => &["DmlExecutionProvider"],
        }
    }

    pub fn of_ort_name(n: &str) -> Option<ProviderKind> {
        ProviderKind::ALL.into_iter().find(|p| p.ort_names().iter().any(|x| x.eq_ignore_ascii_case(n)))
    }

    pub fn is_gpu(self) -> bool {
        self != ProviderKind::Cpu
    }

    /// The GPU vendor the provider serves (None: any vendor, or the CPU).
    pub fn vendor(self) -> Option<&'static str> {
        match self {
            ProviderKind::TensorRtRtx | ProviderKind::Cuda => Some("nvidia"),
            ProviderKind::MiGraphX | ProviderKind::AmdGpu => Some("amd"),
            ProviderKind::OpenVino => Some("intel"),
            ProviderKind::DirectMl | ProviderKind::Cpu => None,
        }
    }

    /// Whether the provider can run on `a`.
    pub fn serves(self, a: &Adapter) -> bool {
        self.is_gpu() && self.vendor().is_none_or(|v| v == a.vendor())
    }
}

/// A job line's provider list (`providers`), in order, the CPU always last; unknown names are skipped (an
/// agent may know providers this worker doesn't). Absent -> `["cpu"]` (PROTOCOL.md 8).
pub fn parse_providers(v: Option<&Value>) -> Result<Vec<ProviderKind>, String> {
    let mut out = Vec::new();
    if let Some(v) = v.filter(|v| !v.is_null()) {
        for x in v.as_array().ok_or("providers: a list of provider names")? {
            let s = x.as_str().ok_or("providers: a list of provider names")?;
            if let Some(p) = ProviderKind::parse(s) {
                if !out.contains(&p) {
                    out.push(p);
                }
            }
        }
    }
    Ok(normalize(&out))
}

/// `ps` without duplicates, the CPU moved to the end (and added).
pub fn normalize(ps: &[ProviderKind]) -> Vec<ProviderKind> {
    let mut out: Vec<ProviderKind> = Vec::new();
    for &p in ps {
        if p != ProviderKind::Cpu && !out.contains(&p) {
            out.push(p);
        }
    }
    out.push(ProviderKind::Cpu);
    out
}

/// The adapter a job's GPU work runs on (the job line's `gpu`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuDevice {
    pub luid: u64,
    pub vendor: String,
    pub name: String,
}

impl GpuDevice {
    /// `{"luid": "<hex>", "vendor", "name"}` or null.
    pub fn from_json(v: Option<&Value>) -> Result<Option<GpuDevice>, String> {
        let Some(v) = v.filter(|v| !v.is_null()) else { return Ok(None) };
        let luid = v.get("luid").and_then(Value::as_str).and_then(crate::gpu::parse_luid).ok_or("gpu: luid must be the adapter LUID in hex")?;
        let s = |k: &str| v.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        Ok(Some(GpuDevice { luid, vendor: s("vendor"), name: s("name") }))
    }
    pub fn of(a: &Adapter) -> GpuDevice {
        GpuDevice { luid: a.luid, vendor: a.vendor().into(), name: a.name.clone() }
    }
    pub fn to_json(&self) -> Value {
        json!({"luid": luid_hex(self.luid), "vendor": self.vendor, "name": self.name})
    }
}

// ------------------------------------------------------------------------------------------------------------
// the ONNX Runtime library, its environment, its providers

/// The MSVC runtime ONNX Runtime's Windows builds import (msvcp140, msvcp140_1, vcruntime140, vcruntime140_1),
/// in load order, and DirectML (Windows ML's ONNX Runtime imports it; System32 has an older copy that the
/// loader would otherwise prefer).
pub const PRELOAD: &[&str] = &["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll", "DirectML.dll"];

/// The process's ONNX Runtime: the library, its environment, what it offers.
pub struct Runtime {
    pub dylib: PathBuf,
    pub dir: PathBuf,
    pub env: ort::environment::Environment,
    /// ONNX Runtime's compiled-in providers (`GetAvailableProviders`).
    pub available: Vec<String>,
    /// Libraries loaded first from the runtime pack (full paths).
    pub preloaded: Vec<String>,
    catalog: Mutex<Option<CatalogState>>,
}

#[derive(Clone, Debug, Default)]
pub struct CatalogState {
    /// The catalog's providers as listed (after any `ensure_ready`).
    pub eps: Vec<crate::winml::CatalogEp>,
    /// Registered with ONNX Runtime: (catalog EP name, our kind).
    pub registered: Vec<(String, ProviderKind)>,
    pub errors: Vec<String>,
    /// Whether `ensure_ready` was allowed.
    pub ensured: bool,
}

static RUNTIME: OnceLock<Result<Runtime, String>> = OnceLock::new();

/// Messages of the session being built (its log id), while one is.
static CAPTURE: Mutex<Option<(String, Vec<String>)>> = Mutex::new(None);
/// One captured build at a time.
static BUILD: Mutex<()> = Mutex::new(());
/// Every ONNX Runtime message to stderr (`ZERO_WORKER_ORT_VERBOSE=1`: a measurement aid).
static VERBOSE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn logger(level: LogLevel, _category: &str, id: &str, _loc: &str, msg: &str) {
    if let Ok(mut c) = CAPTURE.lock() {
        if let Some((want, lines)) = c.as_mut() {
            if id == want.as_str() {
                lines.push(msg.to_string());
                return;
            }
        }
    }
    if matches!(level, LogLevel::Warning | LogLevel::Error | LogLevel::Fatal) || VERBOSE.load(std::sync::atomic::Ordering::Relaxed) {
        eprintln!("[onnxruntime {level:?}] {msg}");
    }
}

#[cfg(windows)]
fn preload(dir: &Path) -> Result<Vec<String>, String> {
    extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut core::ffi::c_void;
        fn GetLastError() -> u32;
    }
    let mut out = Vec::new();
    for n in PRELOAD {
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

#[cfg(not(windows))]
fn preload(_dir: &Path) -> Result<Vec<String>, String> {
    Ok(Vec::new())
}

fn available_providers() -> Vec<String> {
    let api = ort::api();
    let mut ptr: *mut *mut core::ffi::c_char = std::ptr::null_mut();
    let mut n: core::ffi::c_int = 0;
    let st = unsafe { (api.GetAvailableProviders)(&mut ptr, &mut n) };
    if !st.0.is_null() || ptr.is_null() {
        return Vec::new();
    }
    let out = (0..n as usize).map(|i| unsafe { std::ffi::CStr::from_ptr(*ptr.add(i)) }.to_string_lossy().to_string()).collect();
    let _ = unsafe { (api.ReleaseAvailableProviders)(ptr, n) };
    out
}

/// The process's ONNX Runtime, loaded from `dylib` (once; a second call with another library is an error).
/// On Windows the MSVC runtime and DirectML next to it are loaded first, by full path.
pub fn runtime(dylib: &Path) -> Result<&'static Runtime, String> {
    let r = RUNTIME.get_or_init(|| {
        let dir = dylib.parent().map(Path::to_path_buf).unwrap_or_default();
        let preloaded = if dir.as_os_str().is_empty() { Vec::new() } else { preload(&dir)? };
        let b = ort::init_from(dylib).map_err(|e| format!("ONNX Runtime {}: {e}", dylib.display()))?;
        let env = b.with_name("zero-worker").with_logger(Arc::new(logger)).build().map_err(|e| format!("ONNX Runtime environment: {e}"))?;
        let verbose = std::env::var("ZERO_WORKER_ORT_VERBOSE").is_ok_and(|v| v == "1");
        VERBOSE.store(verbose, std::sync::atomic::Ordering::Relaxed);
        env.set_log_level(if verbose { LogLevel::Verbose } else { LogLevel::Warning });
        Ok(Runtime { dylib: dylib.to_path_buf(), dir, env, available: available_providers(), preloaded, catalog: Mutex::new(None) })
    });
    let rt = r.as_ref().map_err(|e| e.clone())?;
    if rt.dylib != dylib {
        return Err(format!("ONNX Runtime is already loaded from {} (asked for {})", rt.dylib.display(), dylib.display()));
    }
    Ok(rt)
}

/// The loaded ONNX Runtime's version.
pub fn ort_version() -> String {
    ort::info().to_string()
}

/// Key-value pairs of ONNX Runtime (device metadata).
fn kv_pairs(kv: *const ort::sys::OrtKeyValuePairs) -> Vec<(String, String)> {
    if kv.is_null() {
        return Vec::new();
    }
    let api = ort::api();
    let (mut keys, mut vals, mut n) = (std::ptr::null(), std::ptr::null(), 0usize);
    unsafe { (api.GetKeyValuePairs)(kv, &mut keys, &mut vals, &mut n) };
    let s = |p: *const core::ffi::c_char| if p.is_null() { String::new() } else { unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().to_string() };
    (0..n).map(|i| unsafe { (s(*keys.add(i)), s(*vals.add(i))) }).collect()
}

/// One of ONNX Runtime's EP devices (`GetEpDevices`): a provider bound to a hardware device.
#[derive(Clone, Debug)]
pub struct EpDevice {
    pub ep: String,
    pub ep_vendor: String,
    pub kind: Option<ProviderKind>,
    pub hw_type: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub hw_vendor: String,
    pub metadata: Vec<(String, String)>,
    /// The hardware device's LUID (metadata `LUID`), when it has one.
    pub luid: Option<u64>,
}

impl EpDevice {
    pub fn to_json(&self) -> Value {
        json!({"ep": self.ep, "ep_vendor": self.ep_vendor, "kind": self.kind.map(|k| k.wire()), "type": self.hw_type,
               "vendor_id": self.vendor_id, "device_id": self.device_id, "vendor": self.hw_vendor,
               "luid": self.luid.map(luid_hex), "metadata": self.metadata.iter().map(|(k, v)| (k.clone(), json!(v))).collect::<serde_json::Map<_, _>>()})
    }
}

fn parse_metadata_luid(v: &str) -> Option<u64> {
    let t = v.trim();
    if let Some(h) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        return u64::from_str_radix(h, 16).ok();
    }
    t.parse::<u64>().ok()
}

impl Runtime {
    /// ONNX Runtime's EP devices (the plugin providers' and the built-in ones' that publish devices).
    pub fn ep_devices(&self) -> Vec<EpDevice> {
        let api = ort::api();
        self.env
            .devices()
            .map(|d| {
                let hw = d.hardware_device();
                let hwp: *const ort::sys::OrtHardwareDevice = ort::AsPointer::ptr(&hw);
                let metadata = kv_pairs(unsafe { (api.HardwareDevice_Metadata)(hwp) });
                let luid = metadata.iter().find(|(k, _)| k.eq_ignore_ascii_case("LUID")).and_then(|(_, v)| parse_metadata_luid(v));
                let ep = d.ep().unwrap_or("").to_string();
                EpDevice {
                    kind: ProviderKind::of_ort_name(&ep),
                    ep,
                    ep_vendor: d.ep_vendor().unwrap_or("").to_string(),
                    hw_type: format!("{:?}", hw.ty()),
                    vendor_id: unsafe { (api.HardwareDevice_VendorId)(hwp) },
                    device_id: hw.id(),
                    hw_vendor: hw.vendor().unwrap_or("").to_string(),
                    metadata,
                    luid,
                }
            })
            .collect()
    }

    /// Windows ML's catalog providers registered with ONNX Runtime (once per process; `ensure_ready` lets
    /// Windows ML download and install the wanted ones first - the probe's job, never a lease's).
    pub fn catalog(&self, wanted: &[ProviderKind], ensure_ready: bool) -> CatalogState {
        let mut g = self.catalog.lock().unwrap();
        if let Some(c) = g.as_ref() {
            if c.ensured || !ensure_ready {
                return c.clone();
            }
        }
        let mut st = CatalogState { ensured: ensure_ready, ..Default::default() };
        if !crate::winml::present(&self.dir) {
            st.errors.push(format!("the runtime pack has no {}", crate::winml::DLL));
            *g = Some(st.clone());
            return st;
        }
        let want = |name: &str| ensure_ready && ProviderKind::of_ort_name(name).is_some_and(|k| wanted.contains(&k));
        match crate::winml::list(&self.dir, &want) {
            Ok(eps) => {
                for e in &eps {
                    let Some(kind) = ProviderKind::of_ort_name(&e.name) else { continue };
                    if !e.certified || e.ready != crate::winml::Ready::Ready || e.library_path.is_empty() {
                        continue;
                    }
                    if g.as_ref().is_some_and(|c| c.registered.iter().any(|r| r.0 == e.name)) {
                        st.registered.push((e.name.clone(), kind));
                        continue;
                    }
                    match self.env.register_ep_library(e.name.clone(), crate::winml::library(e)) {
                        Ok(()) => st.registered.push((e.name.clone(), kind)),
                        Err(x) => st.errors.push(format!("{}: {x}", e.name)),
                    }
                }
                st.eps = eps;
            }
            Err(e) => st.errors.push(e),
        }
        *g = Some(st.clone());
        st
    }

    /// Whether this runtime can offer `p` at all (compiled in, or a registered plugin).
    pub fn offers(&self, p: ProviderKind) -> bool {
        p == ProviderKind::Cpu || p.ort_names().iter().any(|n| self.available.iter().any(|a| a.eq_ignore_ascii_case(n))) || self.ep_devices().iter().any(|d| d.kind == Some(p))
    }
}

// ------------------------------------------------------------------------------------------------------------
// the factory

/// What [`create_session`] builds.
pub struct SessionSpec<'a> {
    /// A `zero-onnx/1` or `/2` graph.
    pub model: &'a Path,
    /// Its reference set (`reference.npz`); required to accept any GPU provider.
    pub reference: Option<&'a Path>,
    /// Preference order (the job line's `providers`); the CPU is always tried last.
    pub providers: &'a [ProviderKind],
    /// The adapter to use (the job line's `gpu`); None: the first adapter a provider serves.
    pub device: Option<GpuDevice>,
    pub intra_threads: usize,
    /// The ONNX Runtime library (the runtime pack's).
    pub runtime: &'a Path,
    /// `session.intra_op.allow_spinning`.
    pub spinning: bool,
    /// What the job needs of the graph (game, features, actions, aux width); None: the graph's own metadata
    /// (the gate tools and the probe).
    pub expect: Option<&'a Expect>,
}

/// One provider's self-test (PROTOCOL.md 8; the worker's `selftest` event).
#[derive(Clone, Debug)]
pub struct SelfTest {
    pub provider: ProviderKind,
    /// The largest |answer - reference| over every number (inf when not finite or a size differs; 0 without
    /// a reference set).
    pub max_abs: f32,
    pub ok: bool,
    /// Nodes ONNX Runtime placed on the CPU under a GPU provider ("OpType (name)").
    pub cpu_fallback_nodes: Vec<String>,
    /// priors, value, margin, aux.
    pub items: [f64; 4],
    pub rows: usize,
    /// The two passes gave the same bits.
    pub deterministic: bool,
    pub seconds: f64,
    /// The adapter (GPU providers).
    pub device: Option<GpuDevice>,
    /// Seconds to build the session.
    pub build_s: f64,
    /// Seconds of each pass over the reference set (the first includes a GPU provider's warm-up).
    pub pass_s: [f64; 2],
}

impl SelfTest {
    pub fn to_json(&self) -> Value {
        let f = |x: f64| if x.is_finite() { json!(x) } else { json!("inf") };
        json!({"provider": self.provider.wire(), "max_abs": f(self.max_abs as f64), "ok": self.ok,
               "cpu_fallback_nodes": self.cpu_fallback_nodes, "rows": self.rows, "deterministic": self.deterministic,
               "items": {"priors": f(self.items[0]), "value": f(self.items[1]), "margin": f(self.items[2]), "aux": f(self.items[3])},
               "seconds": (self.seconds * 1000.0).round() / 1000.0, "build_s": (self.build_s * 1000.0).round() / 1000.0,
               "pass_s": [(self.pass_s[0] * 1e4).round() / 1e4, (self.pass_s[1] * 1e4).round() / 1e4],
               "gpu": self.device.as_ref().map(GpuDevice::to_json)})
    }
}

/// A provider that was tried and not used.
#[derive(Clone, Debug)]
pub struct Attempt {
    pub provider: ProviderKind,
    pub device: Option<GpuDevice>,
    pub why: String,
    pub selftest: Option<SelfTest>,
}

impl Attempt {
    pub fn to_json(&self) -> Value {
        json!({"provider": self.provider.wire(), "gpu": self.device.as_ref().map(GpuDevice::to_json), "why": self.why,
               "selftest": self.selftest.as_ref().map(SelfTest::to_json)})
    }
}

#[derive(Debug, Clone)]
pub struct SessionError(pub String);

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for SessionError {
    fn from(s: String) -> Self {
        SessionError(s)
    }
}

fn oerr(e: impl std::fmt::Display) -> String {
    format!("ONNX Runtime: {e}")
}

/// How a provider is bound to its adapter.
#[derive(Clone, Debug)]
enum Binding {
    Cpu,
    /// DirectML by DXGI index.
    Dml(u32),
    /// A provider appended by name, with its device ordinal (the adapter's index among its vendor's).
    Named(ProviderKind, u32),
    /// A plugin provider's EP device (index into `Runtime::ep_devices`, its EP name).
    Plugin(String, u64),
}

/// The adapter for `p`: the job's, else the first that `p` serves.
fn pick_adapter(p: ProviderKind, want: Option<&GpuDevice>, adapters: &[Adapter]) -> Result<Adapter, String> {
    match want {
        Some(d) => {
            let a = adapters.iter().find(|a| a.luid == d.luid).ok_or_else(|| format!("no adapter with LUID {} on this PC", luid_hex(d.luid)))?;
            if !p.serves(a) {
                return Err(format!("{} does not run on {} ({})", p.wire(), a.name, a.vendor()));
            }
            Ok(a.clone())
        }
        None => adapters.iter().find(|a| p.serves(a)).cloned().ok_or_else(|| format!("no {} adapter for {}", p.vendor().unwrap_or("GPU"), p.wire())),
    }
}

fn bind(rt: &Runtime, p: ProviderKind, a: Option<&Adapter>, adapters: &[Adapter]) -> Result<Binding, String> {
    if p == ProviderKind::Cpu {
        return Ok(Binding::Cpu);
    }
    let a = a.expect("a GPU provider has an adapter");
    // plugin providers (Windows ML's catalog): the EP device on this adapter
    if let Some(d) = rt.ep_devices().into_iter().find(|d| d.kind == Some(p) && d.luid == Some(a.luid)) {
        return Ok(Binding::Plugin(d.ep, a.luid));
    }
    let compiled = p.ort_names().iter().any(|n| rt.available.iter().any(|x| x.eq_ignore_ascii_case(n)));
    if !compiled {
        return Err(format!("this runtime pack has no {} provider", p.wire()));
    }
    Ok(match p {
        ProviderKind::DirectMl => Binding::Dml(a.dxgi_index),
        _ => Binding::Named(p, adapters.iter().filter(|x| x.vendor_id == a.vendor_id).position(|x| x.luid == a.luid).unwrap_or(0) as u32),
    })
}

/// TensorRT-RTX has no kernel for NonZero (the `packed` graph's token compaction: a data-dependent shape):
/// the provider leaves it to the CPU and builds its engines around it (`nv_op_types_to_exclude`).
pub const NVRTX_EXCLUDE: &str = "NonZero";

/// Whether this ONNX Runtime's TensorRT-RTX provider takes `nv_op_types_to_exclude` (1.23.2 refuses the key:
/// cleared after the first refusal).
static NVRTX_EXCLUDE_OK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// The most non-zeros of a piece a TensorRT-RTX engine takes (its optimisation profile's maximum; a
/// batch is cut into pieces under it, `InferenceSession::max_nnz`).
pub const NVRTX_MAX_NNZ: usize = GPU_CHUNK * 1024;

/// TensorRT-RTX's options: an explicit optimisation profile for zero-onnx/1's inputs (its implicit one
/// gives every dynamic axis the same maximum, which `x_ptr`'s B + 1 against `n_players`' B contradicts:
/// the engine then fails at run time), NonZero left to the CPU when the provider takes the option.
pub fn nvrtx_options() -> Vec<(String, String)> {
    let shapes = |b: usize, nnz: usize| format!("x_ptr:{},x_idx:{nnz},x_val:{nnz},n_players:{b},belief:{b}", b + 1);
    let mut o = vec![
        ("nv_profile_min_shapes".to_string(), shapes(1, 1)),
        ("nv_profile_opt_shapes".to_string(), shapes(256, 256 * 256)),
        ("nv_profile_max_shapes".to_string(), shapes(GPU_CHUNK, NVRTX_MAX_NNZ)),
    ];
    if NVRTX_EXCLUDE_OK.load(std::sync::atomic::Ordering::Relaxed) {
        o.push(("nv_op_types_to_exclude".to_string(), NVRTX_EXCLUDE.to_string()));
    }
    o
}

fn nvrtx(ord: u32) -> ort::ep::NVRTX {
    use ort::ep::ArbitrarilyConfigurableExecutionProvider;
    let mut ep = ort::ep::NVRTX::default().with_device_id(ord);
    for (k, v) in nvrtx_options() {
        ep = ep.with_arbitrary_config(k, v);
    }
    ep
}

fn builder(rt: &Runtime, b: &Binding, threads: usize, spinning: bool, log_id: Option<&str>) -> Result<SessionBuilder, String> {
    let mut sb = Session::builder(&rt.env).map_err(oerr)?;
    sb = sb.with_intra_threads(threads.max(1)).map_err(oerr)?;
    sb = sb.with_inter_threads(1).map_err(oerr)?;
    sb = sb.with_parallel_execution(false).map_err(oerr)?;
    sb = sb.with_optimization_level(GraphOptimizationLevel::All).map_err(oerr)?;
    sb = sb.with_intra_op_spinning(spinning).map_err(oerr)?;
    if !matches!(b, Binding::Cpu) {
        sb = sb.with_deterministic_compute(true).map_err(oerr)?;
    }
    if let Some(id) = log_id {
        sb = sb.with_log_id(id).map_err(oerr)?;
        sb = sb.with_log_level(LogLevel::Verbose).map_err(oerr)?;
    }
    match b {
        Binding::Cpu => {}
        Binding::Dml(idx) => {
            sb = sb.with_memory_pattern(false).map_err(oerr)?;
            sb = sb.with_execution_providers([ort::ep::DirectML::default().with_device_id(*idx as i32).build().error_on_failure()]).map_err(oerr)?;
        }
        Binding::Named(ProviderKind::Cuda, ord) => {
            let ep = ort::ep::CUDA::default().with_device_id(*ord as i32).with_tf32(false).with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Default);
            sb = sb.with_execution_providers([ep.build().error_on_failure()]).map_err(oerr)?;
        }
        Binding::Named(ProviderKind::TensorRtRtx, ord) => {
            sb = sb.with_execution_providers([nvrtx(*ord).build().error_on_failure()]).map_err(oerr)?;
        }
        Binding::Named(ProviderKind::OpenVino, _) => {
            sb = sb.with_execution_providers([ort::ep::OpenVINO::default().with_device_type("GPU").with_precision("FP32").build().error_on_failure()]).map_err(oerr)?;
        }
        Binding::Named(ProviderKind::MiGraphX, ord) => {
            sb = sb.with_execution_providers([ort::ep::MIGraphX::default().with_device_id(*ord as i32).with_fp16(false).build().error_on_failure()]).map_err(oerr)?;
        }
        Binding::Named(p, _) => return Err(format!("{}: no built-in binding", p.wire())),
        Binding::Plugin(ep, luid) => {
            let devs: Vec<_> = rt
                .env
                .devices()
                .filter(|d| d.ep().is_ok_and(|n| n == ep))
                .filter(|d| {
                    let hw = d.hardware_device();
                    let hwp: *const ort::sys::OrtHardwareDevice = ort::AsPointer::ptr(&hw);
                    kv_pairs(unsafe { (ort::api().HardwareDevice_Metadata)(hwp) }).iter().any(|(k, v)| k.eq_ignore_ascii_case("LUID") && parse_metadata_luid(v) == Some(*luid))
                })
                .collect();
            if devs.is_empty() {
                return Err(format!("{ep}: no device with LUID {}", luid_hex(*luid)));
            }
            // (a plugin's options are named "<its EP name>.<option>")
            let opts: Vec<(String, String)> = match ProviderKind::of_ort_name(ep) {
                Some(ProviderKind::TensorRtRtx) => nvrtx_options().into_iter().map(|(k, v)| (format!("{ep}.{k}"), v)).collect(),
                _ => Vec::new(),
            };
            sb = sb.with_devices(devs, Some(&opts)).map_err(oerr)?;
        }
    }
    Ok(sb)
}

/// ONNX Runtime's node placement from a session's verbose log ("Node placements", then per provider
/// "Node(s) placed on [P]. Number of nodes: N" and the nodes, or " All nodes placed on [P]") -> (provider,
/// "OpType (name)") pairs.
pub fn parse_placements(lines: &[String]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let Some(start) = lines.iter().position(|l| l.trim() == "Node placements") else { return out };
    let mut cur: Option<String> = None;
    for l in &lines[start + 1..] {
        let t = l.trim();
        if let Some(rest) = t.strip_prefix("All nodes placed on [") {
            // (no node list follows: every node on one provider)
            let p = rest.split(']').next().unwrap_or("").to_string();
            let n: usize = rest.rsplit("Number of nodes: ").next().and_then(|x| x.trim().parse().ok()).unwrap_or(0);
            for k in 0..n {
                out.push((p.clone(), format!("* (node {k})")));
            }
            break;
        }
        if let Some(rest) = t.strip_prefix("Node(s) placed on [") {
            cur = rest.split(']').next().map(String::from);
            continue;
        }
        let Some(p) = &cur else { break };
        // one node per line, or several on one ("Op (name), Op (name)")
        let mut any = false;
        for part in t.split("), ").map(|s| s.trim()) {
            let s = if part.ends_with(')') { part.to_string() } else { format!("{part})") };
            if let Some(i) = s.find(" (") {
                if i > 0 && !s[..i].contains(' ') {
                    out.push((p.clone(), s));
                    any = true;
                }
            }
        }
        if !any {
            break;
        }
    }
    out
}

static LOG_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Build a session for `b`, with the node placement read from ONNX Runtime's log when `capture`.
fn build(rt: &Runtime, model: &Path, b: &Binding, threads: usize, spinning: bool, capture: bool) -> Result<(Session, Vec<(String, String)>), String> {
    if matches!(b, Binding::Cpu) {
        return build_retry(rt, model, b, threads, spinning, capture);
    }
    // a GPU provider's graph partitioning (TensorRT-RTX's) recurses deeper than a thread's default stack: build on
    // a thread with the worker's big stack, whatever thread asked (the job's, the evaluator's after a yield)
    std::thread::scope(|sc| {
        std::thread::Builder::new()
            .name("zero-worker-session".into())
            .stack_size(crate::MAIN_STACK)
            .spawn_scoped(sc, || build_retry(rt, model, b, threads, spinning, capture))
            .map_err(|e| format!("spawn the session builder: {e}"))?
            .join()
            .map_err(|_| "the session builder panicked".to_string())?
    })
}

fn build_retry(rt: &Runtime, model: &Path, b: &Binding, threads: usize, spinning: bool, capture: bool) -> Result<(Session, Vec<(String, String)>), String> {
    let r = build_once(rt, model, b, threads, spinning, capture);
    if let (Err(e), Binding::Named(ProviderKind::TensorRtRtx, _)) = (&r, b) {
        if e.contains("Failed to load provider") && NVRTX_EXCLUDE_OK.swap(false, std::sync::atomic::Ordering::Relaxed) {
            // (an older TensorRT-RTX provider without nv_op_types_to_exclude: NonZero stays in its engines)
            return build_once(rt, model, b, threads, spinning, capture);
        }
    }
    r
}

fn build_once(rt: &Runtime, model: &Path, b: &Binding, threads: usize, spinning: bool, capture: bool) -> Result<(Session, Vec<(String, String)>), String> {
    if !capture {
        let mut sb = builder(rt, b, threads, spinning, None)?;
        return Ok((sb.commit_from_file(model).map_err(|e| format!("{}: {}", model.display(), oerr(e)))?, Vec::new()));
    }
    let _one = BUILD.lock().unwrap_or_else(|e| e.into_inner());
    let id = format!("zw-{}-{}", std::process::id(), LOG_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
    *CAPTURE.lock().unwrap() = Some((id.clone(), Vec::new()));
    let r = builder(rt, b, threads, spinning, Some(&id)).and_then(|mut sb| sb.commit_from_file(model).map_err(|e| format!("{}: {}", model.display(), oerr(e))));
    let lines = CAPTURE.lock().unwrap().take().map(|c| c.1).unwrap_or_default();
    // (a measurement aid: ONNX Runtime's whole verbose log of the build, appended to the file this names)
    if let Ok(f) = std::env::var("ZERO_WORKER_ORT_LOG") {
        if let Ok(mut o) = std::fs::OpenOptions::new().create(true).append(true).open(&f) {
            use std::io::Write;
            let _ = writeln!(o, "===== {id} {:?}", b);
            for l in &lines {
                let _ = writeln!(o, "{l}");
            }
        }
    }
    let s = r?;
    Ok((s, parse_placements(&lines)))
}

/// The graph's metadata (`metadata_props`).
fn metadata(s: &Session) -> Result<Vec<(String, String)>, String> {
    let md = s.metadata().map_err(oerr)?;
    let mut meta = Vec::new();
    for k in md.custom_keys().map_err(oerr)? {
        let v = md.custom(&k).unwrap_or_default();
        meta.push((k, v));
    }
    Ok(meta)
}

/// What a graph's metadata says about it (the expectation the gate tools and the probe use).
pub fn expect_of(meta: &[(String, String)], path: &Path) -> Result<Expect, String> {
    let get = |k: &str| meta.iter().find(|e| e.0 == k).map(|e| e.1.clone());
    let num = |k: &str| get(k).and_then(|v| v.parse::<usize>().ok()).ok_or_else(|| format!("{}: metadata {k}", path.display()));
    // (zero-onnx/2: its tail segments and overrides as declared - `varlen_of` then checks them against themselves)
    let mut tails = Vec::new();
    if let Some(t) = get("tails") {
        let v: Value = serde_json::from_str(&t).map_err(|e| format!("{}: metadata tails: {e}", path.display()))?;
        for e in v.as_array().into_iter().flatten() {
            let a = e.as_array().filter(|a| a.len() == 4).ok_or_else(|| format!("{}: metadata tails entry {e}", path.display()))?;
            tails.push((a[0].as_u64().unwrap_or(0) as u16, a[1].as_str().unwrap_or_default().to_string(), a[2].as_u64().unwrap_or(0) as usize, a[3].as_u64().unwrap_or(0) as usize));
        }
    }
    Ok(Expect { game: get("game").unwrap_or_default(), features: num("features")?, actions: num("actions")?, aux_width: num("aux_width")?, belief: false, tails, option_pointers: get("option_pointers").as_deref() == Some("true") })
}

/// zero-onnx/2's more inputs (zero-core docs/az.md 46): the driver's options, tail tokens and pointer overrides.
pub const VARLEN_INPUTS: [&str; 9] = ["opt_off", "opt_action", "t_off", "t_seg", "t_slot", "t_val", "ov_off", "ov_opt", "ov_tok"];

/// A zero-onnx/2 graph's tail segments' widths by segment index (0: not a tail segment) from its metadata (`tails`
/// [[segment, name, width, head slots], ...]), checked against the game's (`Expect::tails`, `option_pointers`); None
/// for a zero-onnx/1 graph, which a game with tail segments or pointer overrides refuses (its rows would be dropped).
pub fn varlen_of(meta: &[(String, String)], path: &Path, want: &Expect) -> Result<Option<Vec<usize>>, String> {
    let get = |k: &str| meta.iter().find(|e| e.0 == k).map(|e| e.1.as_str());
    if get("zero_onnx") != Some("2") {
        if !want.tails.is_empty() || want.option_pointers {
            return Err(format!("{}: a zero-onnx/{} graph for game {:?}, whose rows have tail tokens / pointer overrides (zero-onnx/2's, zero-core docs/az.md 46)", path.display(), get("zero_onnx").unwrap_or("?"), want.game));
        }
        return Ok(None);
    }
    let tails: Value = serde_json::from_str(get("tails").ok_or_else(|| format!("{}: a zero-onnx/2 graph without metadata tails", path.display()))?).map_err(|e| format!("{}: metadata tails: {e}", path.display()))?;
    let mut have: Vec<(u16, String, usize, usize)> = Vec::new();
    for t in tails.as_array().ok_or_else(|| format!("{}: metadata tails is not a list", path.display()))? {
        let a = t.as_array().filter(|a| a.len() == 4).ok_or_else(|| format!("{}: metadata tails entry {t}", path.display()))?;
        let n = |v: &Value| v.as_u64().ok_or_else(|| format!("{}: metadata tails entry {t}", path.display()));
        have.push((n(&a[0])? as u16, a[1].as_str().unwrap_or_default().to_string(), n(&a[2])? as usize, n(&a[3])? as usize));
    }
    let pointers = get("option_pointers") == Some("true");
    if have != want.tails || pointers != want.option_pointers {
        return Err(format!("{}: the network's tail segments {have:?} / pointer overrides {pointers}, the game's {:?} / {} (another layout)", path.display(), want.tails, want.option_pointers));
    }
    let mut widths = vec![0usize; have.iter().map(|t| t.0 as usize + 1).max().unwrap_or(0)];
    for (s, _, w, _) in &have {
        widths[*s as usize] = *w;
    }
    Ok(Some(widths))
}

/// The graph's metadata and inputs / outputs checked against `want` (phase 0a's "For phase 0b" point 1; zero-onnx/2:
/// its more inputs and the output `refused`, zero-core docs/az.md 46).
fn check_graph(s: &Session, meta: &[(String, String)], path: &Path, want: &Expect) -> Result<(usize, usize), String> {
    let get = |k: &str| meta.iter().find(|e| e.0 == k).map(|e| e.1.as_str());
    let num = |k: &str| -> Result<usize, String> { get(k).ok_or_else(|| format!("{}: no metadata {k:?} (not a zero-onnx graph)", path.display()))?.parse().map_err(|_| format!("{}: metadata {k} {:?}", path.display(), get(k))) };
    let v2 = match get("zero_onnx") {
        Some("1") => false,
        Some("2") => true,
        v => return Err(format!("{}: zero_onnx {v:?}: this worker reads zero-onnx/1 and zero-onnx/2", path.display())),
    };
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
    let more: &[&str] = if v2 { &VARLEN_INPUTS } else { &[] };
    for name in ["x_ptr", "x_idx", "x_val", "n_players", "belief"].iter().chain(more) {
        if !s.inputs().iter().any(|i| i.name() == *name) {
            return Err(format!("{}: no input {name:?}", path.display()));
        }
    }
    let outs: &[&str] = if v2 { &["policy", "value", "margin", "aux", "refused"] } else { &["policy", "value", "margin", "aux"] };
    for name in outs {
        if !s.outputs().iter().any(|o| o.name() == *name) {
            return Err(format!("{}: no output {name:?}", path.display()));
        }
    }
    Ok((a, w))
}

/// An accepted session: a provider, an adapter, the graph, its self-test (the module docs).
pub struct InferenceSession {
    path: PathBuf,
    rt: &'static Runtime,
    binding: Binding,
    threads: usize,
    spinning: bool,
    session: Option<Session>,
    provider: ProviderKind,
    device: Option<GpuDevice>,
    selftest: SelfTest,
    attempts: Vec<Attempt>,
    /// ONNX Runtime's node placement when the session was first built (GPU providers): (provider, node).
    pub placements: Vec<(String, String)>,
    pub meta: Vec<(String, String)>,
    actions: usize,
    aux_w: usize,
    /// A zero-onnx/2 graph: its tail segments' widths by segment index (`varlen_of`); None: zero-onnx/1.
    widths: Option<Vec<usize>>,
    /// Calls, rows, seconds in session.run (and the post-processing), per batch durations (ms).
    pub calls: u64,
    pub rows: u64,
    pub run_s: f64,
    pub batch_ms: Vec<f32>,
    pub load_s: f64,
    /// Rows per piece (`chunk_for`).
    pub chunk: usize,
    /// Non-zeros per piece at most (TensorRT-RTX's profile), None: no limit.
    pub max_nnz: Option<usize>,
}

/// Rows per piece of a GPU provider's batch: a GPU wants whole batches (the CPU's 64-row pieces cost it a
/// launch of every node per piece); a row's answer depends on its piece only by float rounding (each provider
/// passes its self-test with its own pieces).
pub const GPU_CHUNK: usize = 1024;

/// The piece size for `p` (`ZERO_WORKER_GPU_CHUNK` overrides the GPU's, for measurements).
pub fn chunk_for(p: ProviderKind) -> usize {
    if !p.is_gpu() {
        return CHUNK;
    }
    std::env::var("ZERO_WORKER_GPU_CHUNK").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&c| c > 0).unwrap_or(GPU_CHUNK)
}

fn same_bits(a: &Answer, b: &Answer) -> bool {
    let eq = |x: &[f32], y: &[f32]| x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits());
    eq(&a.priors, &b.priors) && eq(&a.value, &b.value) && eq(&a.margin, &b.margin) && eq(&a.aux, &b.aux)
}

fn answer_all(s: &mut InferenceSession, r: &crate::selftest::Reference) -> Result<Answer, String> {
    let mut got = Answer::default();
    for b in &r.batches {
        let a = s.evaluate_batch(b)?;
        got.priors.extend(a.priors);
        got.value.extend(a.value);
        got.margin.extend(a.margin);
        got.aux.extend(a.aux);
    }
    Ok(got)
}

/// The self-test of a built session: the reference set answered twice, compared with the reference's answers
/// and with itself.
fn self_test(s: &mut InferenceSession, r: &crate::selftest::Reference, fallback: Vec<String>, build_s: f64) -> Result<SelfTest, String> {
    let t0 = Instant::now();
    let a = answer_all(s, r)?;
    let p0 = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let b = answer_all(s, r)?;
    let p1 = t1.elapsed().as_secs_f64();
    let d = crate::selftest::compare(&a, &r.want);
    let deterministic = same_bits(&a, &b);
    let max = d.iter().cloned().fold(0.0f64, f64::max);
    Ok(SelfTest {
        provider: s.provider,
        max_abs: max as f32,
        ok: max <= crate::selftest::TOL && deterministic,
        cpu_fallback_nodes: fallback,
        items: d,
        rows: r.rows,
        deterministic,
        seconds: t0.elapsed().as_secs_f64(),
        device: s.device.clone(),
        build_s,
        pass_s: [p0, p1],
    })
}

/// Create the session a job evaluates with (the module docs; PROTOCOL.md 8).
pub fn create_session(spec: &SessionSpec) -> Result<InferenceSession, SessionError> {
    let rt = runtime(spec.runtime)?;
    let reference = match spec.reference {
        Some(p) => Some(crate::selftest::load(p)?),
        None => None,
    };
    let providers = normalize(spec.providers);
    let adapters = if providers.iter().any(|p| p.is_gpu()) { crate::gpu::adapters() } else { Vec::new() };
    let catalog_kinds: Vec<ProviderKind> = providers.iter().copied().filter(|p| CATALOG.contains(p)).collect();
    if !catalog_kinds.is_empty() && crate::winml::present(&rt.dir) {
        // (a job only uses providers Windows ML has installed already; the probe installs them)
        rt.catalog(&catalog_kinds, false);
    }
    let mut attempts = Vec::new();
    for &p in &providers {
        match try_one(rt, spec, reference.as_ref(), p, spec.device.as_ref(), &adapters) {
            Tried::Ok(mut s) => {
                if !s.selftest.ok && p.is_gpu() {
                    let why = if s.selftest.deterministic {
                        format!("self-test: max abs {:e} > {:e}", s.selftest.max_abs, crate::selftest::TOL)
                    } else {
                        "self-test: two passes gave different answers (not deterministic)".to_string()
                    };
                    attempts.push(Attempt { provider: p, device: s.device.clone(), why, selftest: Some(s.selftest.clone()) });
                    continue;
                }
                // (a CPU session that failed its self-test is returned: the caller reports it and ends the job)
                s.attempts = attempts;
                return Ok(s);
            }
            Tried::Skip(a) => attempts.push(a),
            Tried::Fatal(e) => return Err(SessionError(e)),
        }
    }
    Err(SessionError(format!("no provider could build a session: {}", attempts.iter().map(|a| format!("{}: {}", a.provider.wire(), a.why)).collect::<Vec<_>>().join("; "))))
}

/// The providers Windows ML's catalog brings.
pub const CATALOG: [ProviderKind; 4] = [ProviderKind::TensorRtRtx, ProviderKind::AmdGpu, ProviderKind::MiGraphX, ProviderKind::OpenVino];

/// One provider's try (`create_session`'s loop; the probe's per-adapter test).
pub enum Tried {
    /// Built (and self-tested when there is a reference set: `selftest().ok` says whether it passed).
    Ok(InferenceSession),
    /// Not usable here (the provider, the adapter, the build, or a self-test that could not run).
    Skip(Attempt),
    /// The graph or the reference set is not the job's, or the CPU can't build it: the job's error.
    Fatal(String),
}

/// Try `p` on `want` (else the first adapter it serves).
pub fn try_one(rt: &'static Runtime, spec: &SessionSpec, reference: Option<&crate::selftest::Reference>, p: ProviderKind, want: Option<&GpuDevice>, adapters: &[Adapter]) -> Tried {
    let skip = |device: Option<GpuDevice>, why: String| Tried::Skip(Attempt { provider: p, device, why, selftest: None });
    let adapter = if p.is_gpu() {
        if reference.is_none() {
            return skip(want.cloned(), "no reference set: a GPU provider is accepted only through the self-test".into());
        }
        match pick_adapter(p, want, adapters) {
            Ok(a) => Some(a),
            Err(why) => return skip(want.cloned(), why),
        }
    } else {
        None
    };
    let device = adapter.as_ref().map(GpuDevice::of);
    let binding = match bind(rt, p, adapter.as_ref(), adapters) {
        Ok(b) => b,
        Err(why) => return skip(device, why),
    };
    let t0 = Instant::now();
    let (session, placements) = match build(rt, spec.model, &binding, spec.intra_threads, spec.spinning, p.is_gpu()) {
        Ok(x) => x,
        Err(why) if p == ProviderKind::Cpu => return Tried::Fatal(why),
        Err(why) => return skip(device, why),
    };
    let build_s = t0.elapsed().as_secs_f64();
    if p.is_gpu() && !placements.is_empty() && placements.iter().all(|(prov, _)| prov == "CPUExecutionProvider") {
        return skip(device, format!("{} runs no node of this graph (ONNX Runtime placed all {} on the CPU)", p.wire(), placements.len()));
    }
    let meta = match metadata(&session) {
        Ok(m) => m,
        Err(e) => return Tried::Fatal(e),
    };
    let want_graph = match spec.expect {
        Some(w) => w.clone(),
        None => match expect_of(&meta, spec.model) {
            Ok(w) => w,
            Err(e) => return Tried::Fatal(e),
        },
    };
    // (a graph that isn't the job's is the job's error, whatever the provider)
    let (actions, aux_w) = match check_graph(&session, &meta, spec.model, &want_graph) {
        Ok(x) => x,
        Err(e) => return Tried::Fatal(e),
    };
    let widths = match varlen_of(&meta, spec.model, &want_graph) {
        Ok(x) => x,
        Err(e) => return Tried::Fatal(e),
    };
    if let Some(r) = reference {
        if let Err(e) = crate::selftest::check_meta(r, &want_graph, &meta) {
            return Tried::Fatal(e);
        }
    }
    let fallback: Vec<String> = if p.is_gpu() { placements.iter().filter(|(prov, _)| prov == "CPUExecutionProvider").map(|(_, n)| n.clone()).collect() } else { Vec::new() };
    let mut s = InferenceSession {
        path: spec.model.to_path_buf(),
        rt,
        binding,
        threads: spec.intra_threads,
        spinning: spec.spinning,
        session: Some(session),
        provider: p,
        device: device.clone(),
        selftest: SelfTest { provider: p, max_abs: 0.0, ok: true, cpu_fallback_nodes: fallback.clone(), items: [0.0; 4], rows: 0, deterministic: true, seconds: 0.0, device, build_s, pass_s: [0.0; 2] },
        attempts: Vec::new(),
        placements,
        meta,
        actions,
        aux_w,
        widths,
        calls: 0,
        rows: 0,
        run_s: 0.0,
        batch_ms: Vec::new(),
        load_s: build_s,
        chunk: chunk_for(p),
        max_nnz: (p == ProviderKind::TensorRtRtx).then_some(NVRTX_MAX_NNZ),
    };
    if let Some(r) = reference {
        match self_test(&mut s, r, fallback, build_s) {
            Ok(st) => s.selftest = st,
            Err(why) if p == ProviderKind::Cpu => return Tried::Fatal(why),
            Err(why) => return skip(s.device.clone(), why),
        }
        // (the self-test's batches are not the job's)
        s.calls = 0;
        s.rows = 0;
        s.run_s = 0.0;
        s.batch_ms.clear();
    }
    Tried::Ok(s)
}

impl InferenceSession {
    pub fn provider(&self) -> ProviderKind {
        self.provider
    }
    pub fn device(&self) -> Option<&GpuDevice> {
        self.device.as_ref()
    }
    pub fn selftest(&self) -> &SelfTest {
        &self.selftest
    }
    /// The providers tried before this one, and why each was not used.
    pub fn attempts(&self) -> &[Attempt] {
        &self.attempts
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn actions(&self) -> usize {
        self.actions
    }
    pub fn aux_width(&self) -> usize {
        self.aux_w
    }

    /// The `selftest` event (PROTOCOL.md 5 and 8): provider, max_abs, ok, cpu_fallback_nodes (+ the details).
    pub fn selftest_event(&self, gen: Option<u32>) -> Value {
        let mut v = self.selftest.to_json();
        v["ev"] = json!("selftest");
        if let Some(g) = gen {
            v["gen"] = json!(g);
        }
        if !self.attempts.is_empty() {
            v["tried"] = Value::Array(self.attempts.iter().map(Attempt::to_json).collect());
        }
        v
    }

    /// Free everything the session holds (on the GPU too); the next `eval` rebuilds it.
    pub fn release_device(&mut self) {
        self.session = None;
    }

    fn ensure(&mut self) -> Result<&mut Session, String> {
        if self.session.is_none() {
            let t0 = Instant::now();
            let (s, _) = build(self.rt, &self.path, &self.binding, self.threads, self.spinning, false)?;
            self.session = Some(s);
            self.load_s += t0.elapsed().as_secs_f64();
        }
        Ok(self.session.as_mut().expect("built"))
    }

    pub fn eval(&mut self, b: &Batch) -> Result<Answer, SessionError> {
        Evaluator::evaluate(self, b).map_err(SessionError)
    }

    /// The graph's outputs for rows `idx` of `b` (a piece; offsets rebased) -> (policy [n, A], value [n, 8],
    /// margin [n, 8], aux [n, W]).
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
        let (actions, aux_w) = (self.actions, self.aux_w);
        let vr = match &self.widths {
            Some(w) => Some(varlen_piece(b, idx, w)?),
            None => None,
        };
        let v2 = vr.is_some();
        let s = self.ensure()?;
        let mut inputs = ort::inputs![
            "x_ptr" => Tensor::from_array((vec![n as i64 + 1], x_ptr)).map_err(oerr)?,
            "x_idx" => Tensor::from_array((vec![nnz as i64], x_idx)).map_err(oerr)?,
            "x_val" => Tensor::from_array((vec![nnz as i64], x_val)).map_err(oerr)?,
            "n_players" => Tensor::from_array((vec![n as i64], np)).map_err(oerr)?,
            "belief" => Tensor::from_array((vec![n as i64], bel)).map_err(oerr)?,
        ];
        if let Some(v) = vr {
            // zero-onnx/2 (zero-core docs/az.md 46): the piece's options, tail tokens and overrides, as the driver sent them
            let len = |k: usize| vec![k as i64];
            inputs.extend(ort::inputs![
                "opt_off" => Tensor::from_array((len(v.opt_off.len()), v.opt_off)).map_err(oerr)?,
                "opt_action" => Tensor::from_array((len(v.opt_action.len()), v.opt_action)).map_err(oerr)?,
                "t_off" => Tensor::from_array((len(v.t_off.len()), v.t_off)).map_err(oerr)?,
                "t_seg" => Tensor::from_array((len(v.t_seg.len()), v.t_seg)).map_err(oerr)?,
                "t_slot" => Tensor::from_array((len(v.t_slot.len()), v.t_slot)).map_err(oerr)?,
                "t_val" => Tensor::from_array((len(v.t_val.len()), v.t_val)).map_err(oerr)?,
                "ov_off" => Tensor::from_array((len(v.ov_off.len()), v.ov_off)).map_err(oerr)?,
                "ov_opt" => Tensor::from_array((len(v.ov_opt.len()), v.ov_opt)).map_err(oerr)?,
                "ov_tok" => Tensor::from_array((len(v.ov_tok.len()), v.ov_tok)).map_err(oerr)?,
            ]);
        }
        let out = s.run(inputs).map_err(oerr)?;
        if v2 {
            // (a zero-onnx/2 graph's `refused`: per row the tail tokens / overrides it does not take - torch raises on
            // them, R5 / R6: never answered)
            let (shape, data) = out["refused"].try_extract_tensor::<i64>().map_err(oerr)?;
            if shape.len() != 1 || shape[0] as usize != n {
                return Err(format!("output refused: shape {shape:?}, want [{n}]"));
            }
            if let Some(k) = data.iter().position(|&x| x != 0) {
                return Err(format!("the network refused {} tail token(s) / pointer override(s) of this batch (first: row {}): not tokens / targets it takes (zero-core docs/az.md 46)", data.iter().sum::<i64>(), idx[k]));
            }
        }
        let take = |name: &str, w: usize| -> Result<Vec<f32>, String> {
            let (shape, data) = out[name].try_extract_tensor::<f32>().map_err(oerr)?;
            if shape.len() != 2 || shape[0] as usize != n || shape[1] as usize != w {
                return Err(format!("output {name}: shape {shape:?}, want [{n}, {w}]"));
            }
            Ok(data.to_vec())
        };
        Ok((take("policy", actions)?, take("value", MAX_SEATS)?, take("margin", MAX_SEATS)?, take("aux", aux_w)?))
    }

    /// A zero-onnx/2 graph (zero-core docs/az.md 46).
    pub fn is_v2(&self) -> bool {
        self.widths.is_some()
    }

    /// The graph's raw outputs for a whole batch, in row order: a batch of more than `chunk` rows (`CHUNK` on the
    /// CPU, `GPU_CHUNK` on a GPU) in pieces of at most `chunk` rows ordered by their non-zero count (stable),
    /// each piece's outputs put back in row order
    /// (zero-core `onnx_parity.run_chunked`).
    #[allow(clippy::type_complexity)]
    pub fn raw(&mut self, b: &Batch) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), String> {
        if b.xs_off.len() != b.b + 1 {
            return Err("a dense network batch: the native worker's driver must be sparse (DriverCfg::sparse)".into());
        }
        let bb = b.b;
        let chunk = self.chunk;
        let total = (b.xs_off[bb] - b.xs_off[0]) as usize;
        if bb <= chunk && self.max_nnz.is_none_or(|m| total <= m) {
            let idx: Vec<usize> = (0..bb).collect();
            return self.run_piece(b, &idx);
        }
        let mut order: Vec<usize> = (0..bb).collect();
        order.sort_by_key(|&r| b.xs_off[r + 1] - b.xs_off[r]);
        // pieces of at most `chunk` rows and `max_nnz` non-zeros, in that order
        let mut pieces: Vec<&[usize]> = Vec::new();
        let mut start = 0;
        let mut nnz = 0usize;
        for (k, &r) in order.iter().enumerate() {
            let n = (b.xs_off[r + 1] - b.xs_off[r]) as usize;
            if k > start && (k - start == chunk || self.max_nnz.is_some_and(|m| nnz + n > m)) {
                pieces.push(&order[start..k]);
                start = k;
                nnz = 0;
            }
            nnz += n;
        }
        pieces.push(&order[start..]);
        let (a, w) = (self.actions, self.aux_w);
        let mut pol = vec![0f32; bb * a];
        let mut val = vec![0f32; bb * MAX_SEATS];
        let mut mar = vec![0f32; bb * MAX_SEATS];
        let mut aux = vec![0f32; bb * w];
        for piece in pieces {
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

    fn evaluate_batch(&mut self, b: &Batch) -> Result<Answer, String> {
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
}

/// zero-onnx/2's more inputs for rows `idx` of a batch (zero-core docs/az.md 46), each CSR rebased: the options
/// (`opt_off` i64, `opt_action` i32), the tail tokens (`t_off` i64, `t_seg` i32, `t_slot` i64, `t_val` f32 - each
/// token's `width(seg)` floats, consecutive, `widths` the graph's) and the overrides (`ov_off` i64, `ov_opt` i32,
/// `ov_tok` i64). A batch of a game without tail segments (or without overrides) has empty CSRs of B + 1 zeros.
#[derive(Debug)]
pub struct VarlenPiece {
    pub opt_off: Vec<i64>,
    pub opt_action: Vec<i32>,
    pub t_off: Vec<i64>,
    pub t_seg: Vec<i32>,
    pub t_slot: Vec<i64>,
    pub t_val: Vec<f32>,
    pub ov_off: Vec<i64>,
    pub ov_opt: Vec<i32>,
    pub ov_tok: Vec<i64>,
}

pub fn varlen_piece(b: &Batch, idx: &[usize], widths: &[usize]) -> Result<VarlenPiece, String> {
    let mut v = VarlenPiece { opt_off: vec![0], opt_action: Vec::new(), t_off: vec![0], t_seg: Vec::new(), t_slot: Vec::new(), t_val: Vec::new(), ov_off: vec![0], ov_opt: Vec::new(), ov_tok: Vec::new() };
    let tails = !b.t_off.is_empty();
    let ovs = !b.ov_off.is_empty();
    if (tails && b.t_off.len() != b.b + 1) || (ovs && b.ov_off.len() != b.b + 1) {
        return Err(format!("a batch of {} rows with {} tail / {} override offsets", b.b, b.t_off.len(), b.ov_off.len()));
    }
    // each tail token's first float (its segment's width, the graph's)
    let mut voff = Vec::with_capacity(b.t_seg.len() + 1);
    voff.push(0usize);
    for &s in &b.t_seg {
        let w = widths.get(s as usize).copied().unwrap_or(0);
        if w == 0 {
            return Err(format!("a tail token of segment {s}: not one of the network's tail segments"));
        }
        voff.push(voff.last().unwrap() + w);
    }
    if *voff.last().unwrap() != b.t_val.len() {
        return Err(format!("{} tail floats for tail tokens of {}", b.t_val.len(), voff.last().unwrap()));
    }
    for &r in idx {
        let (o0, o1) = (b.opt_off[r] as usize, b.opt_off[r + 1] as usize);
        v.opt_action.extend_from_slice(&b.opt_action[o0..o1]);
        v.opt_off.push(v.opt_action.len() as i64);
        if tails {
            for j in b.t_off[r] as usize..b.t_off[r + 1] as usize {
                v.t_seg.push(b.t_seg[j] as i32);
                v.t_slot.push(b.t_slot[j] as i64);
                v.t_val.extend_from_slice(&b.t_val[voff[j]..voff[j + 1]]);
            }
        }
        v.t_off.push(v.t_seg.len() as i64);
        if ovs {
            for j in b.ov_off[r] as usize..b.ov_off[r + 1] as usize {
                v.ov_opt.push(b.ov_opt[j]);
                v.ov_tok.push(b.ov_tok[j]);
            }
        }
        v.ov_off.push(v.ov_opt.len() as i64);
    }
    Ok(v)
}

impl Evaluator for InferenceSession {
    fn evaluate(&mut self, b: &Batch) -> Result<Answer, String> {
        self.evaluate_batch(b)
    }
    fn release(&mut self) {
        self.release_device();
    }
    fn restore(&mut self) -> Result<(), String> {
        self.ensure().map(|_| ())
    }
    fn describe(&self) -> String {
        format!(
            "onnx {} ({}{}, {} intra-op threads, spinning {})",
            self.path.display(),
            self.provider.wire(),
            self.device.as_ref().map(|d| format!(" on {} [{}]", d.name, luid_hex(d.luid))).unwrap_or_default(),
            self.threads,
            self.spinning
        )
    }
    fn provider(&self) -> String {
        self.provider.wire().to_string()
    }
}

/// A job's network evaluator (self-play and eval jobs alike): the model FILE's sha256 checked before ONNX
/// Runtime reads a byte of it, then [`create_session`] with the job's providers, adapter and reference set; the
/// `selftest` event (with `gen` in an eval job) and Stats' `selftest_max_abs` / `provider`; an error - the job's
/// end - when not even the CPU answers as the reference does.
#[allow(clippy::too_many_arguments)]
pub fn build_evaluator(
    model: &Path,
    ort: &crate::eval::OrtCfg,
    sha256: Option<&str>,
    reference: Option<&Path>,
    want: &Expect,
    sink: &dyn crate::job::Sink,
    progress: &Mutex<crate::job::Progress>,
    gen: Option<u32>,
) -> Result<Box<dyn Evaluator>, String> {
    if let Some(h) = sha256 {
        crate::job::check_file_sha256(model, h, gen)?;
    }
    let spec = SessionSpec { model, reference, providers: &ort.providers, device: ort.device.clone(), intra_threads: ort.intra_threads, runtime: &ort.dylib, spinning: ort.spinning, expect: Some(want) };
    let s = create_session(&spec).map_err(|e| e.0)?;
    let g = gen.map_or(String::new(), |g| format!(" (gen {g})"));
    for a in s.attempts() {
        sink.log(&format!("provider {}{} not used{g}: {}", a.provider.wire(), a.device.as_ref().map(|d| format!(" on {}", d.name)).unwrap_or_default(), a.why));
    }
    {
        let mut p = progress.lock().unwrap();
        p.provider = s.provider().wire().to_string();
    }
    if reference.is_none() {
        sink.log("no reference set: the self-test is skipped (the CPU provider)");
        return Ok(Box::new(s));
    }
    sink.event(s.selftest_event(gen));
    let st = s.selftest().clone();
    progress.lock().unwrap().selftest_max_abs = Some(st.max_abs as f64);
    if !st.ok {
        return Err(format!(
            "selftest failed{g}: the network's answers on this machine differ from the reference's by {:e} (> {:e}; priors {:e}, value {:e}, margin {:e}, aux {:e}{}) on {} reference rows ({}): not playing with this evaluator",
            st.max_abs,
            crate::selftest::TOL,
            st.items[0],
            st.items[1],
            st.items[2],
            st.items[3],
            if st.deterministic { "" } else { "; two passes differed" },
            st.rows,
            reference.map(|r| r.display().to_string()).unwrap_or_default()
        ));
    }
    sink.log(&format!(
        "selftest passed{g}: {}{} max abs {:e} on {} rows, {} nodes on the CPU",
        s.provider().wire(),
        s.device().map(|d| format!(" on {}", d.name)).unwrap_or_default(),
        st.max_abs,
        st.rows,
        st.cpu_fallback_nodes.len()
    ));
    Ok(Box::new(s))
}

/// A graph's metadata, read through a CPU session (the rows' network id when a job names none; the gate tools).
pub fn read_metadata(runtime_dylib: &Path, model: &Path) -> Result<Vec<(String, String)>, String> {
    let rt = runtime(runtime_dylib)?;
    let (s, _) = build(rt, model, &Binding::Cpu, 1, false, false)?;
    metadata(&s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_lists_end_with_the_cpu() {
        let p = parse_providers(Some(&json!(["dml", "cpu", "tensorrt_rtx", "dml", "quantum"]))).unwrap();
        assert_eq!(p, vec![ProviderKind::DirectMl, ProviderKind::TensorRtRtx, ProviderKind::Cpu]);
        assert_eq!(parse_providers(None).unwrap(), vec![ProviderKind::Cpu]);
        assert_eq!(parse_providers(Some(&Value::Null)).unwrap(), vec![ProviderKind::Cpu]);
        assert!(parse_providers(Some(&json!("dml"))).is_err());
        for p in ProviderKind::ALL {
            assert_eq!(ProviderKind::parse(p.wire()), Some(p));
            assert_eq!(ProviderKind::of_ort_name(p.ort_names()[0]), Some(p));
        }
        assert_eq!(ProviderKind::of_ort_name("NvTensorRtRtxExecutionProvider"), Some(ProviderKind::TensorRtRtx));
    }

    #[test]
    fn gpu_lines_parse() {
        let d = GpuDevice::from_json(Some(&json!({"luid": "000000000000d1f3", "vendor": "amd", "name": "RX"}))).unwrap().unwrap();
        assert_eq!((d.luid, d.vendor.as_str()), (0xd1f3, "amd"));
        assert_eq!(GpuDevice::from_json(Some(&Value::Null)).unwrap(), None);
        assert!(GpuDevice::from_json(Some(&json!({"luid": "zz"}))).is_err());
        let a = Adapter { luid: 1, vendor_id: 0x1002, device_id: 0, name: "RX".into(), vram_mb: 1, dxgi_index: 0 };
        assert!(ProviderKind::DirectMl.serves(&a) && ProviderKind::AmdGpu.serves(&a) && !ProviderKind::Cuda.serves(&a) && !ProviderKind::Cpu.serves(&a));
    }

    #[test]
    fn placements_parse_both_forms() {
        let l = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let one = parse_placements(&l(&["x", "Node placements", " All nodes placed on [CPUExecutionProvider]. Number of nodes: 3", "other"]));
        assert_eq!(one.len(), 3);
        assert!(one.iter().all(|(p, _)| p == "CPUExecutionProvider"));
        let many = parse_placements(&l(&[
            "Node placements",
            " Node(s) placed on [DmlExecutionProvider]. Number of nodes: 2",
            "  Gemm (/l0/fc)",
            "  MatMul (/l0/mm)",
            " Node(s) placed on [CPUExecutionProvider]. Number of nodes: 2",
            "  NonZero (/pack/NonZero), Shape (/pack/Shape_1)",
            "Session state done",
        ]));
        assert_eq!(
            many,
            vec![
                ("DmlExecutionProvider".to_string(), "Gemm (/l0/fc)".to_string()),
                ("DmlExecutionProvider".to_string(), "MatMul (/l0/mm)".to_string()),
                ("CPUExecutionProvider".to_string(), "NonZero (/pack/NonZero)".to_string()),
                ("CPUExecutionProvider".to_string(), "Shape (/pack/Shape_1)".to_string()),
            ]
        );
    }

    #[test]
    fn a_pieces_tails_and_overrides_are_its_rows_rebased() {
        // 3 rows: row 0 two tail tokens (segment 0, width 3; segment 2, width 1), row 1 none, row 2 one (segment 0);
        // overrides: row 0 one, row 2 two
        let mut b = Batch::default();
        b.b = 3;
        b.opt_off = vec![0, 2, 3, 6];
        b.opt_action = vec![10, 11, 12, 13, 14, 15];
        b.t_off = vec![0, 2, 2, 3];
        b.t_seg = vec![0, 2, 0];
        b.t_slot = vec![5, 0, 4_294_967_295];
        b.t_val = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        b.ov_off = vec![0, 1, 1, 3];
        b.ov_opt = vec![1, 0, 2];
        b.ov_tok = vec![-1, (2 << 40) | 7, -1];
        let widths = vec![3, 0, 1];
        let v = varlen_piece(&b, &[2, 0], &widths).unwrap();
        assert_eq!((v.opt_off, v.opt_action), (vec![0, 3, 5], vec![13, 14, 15, 10, 11]));
        assert_eq!((v.t_off, v.t_seg, v.t_slot), (vec![0, 1, 3], vec![0, 0, 2], vec![4_294_967_295, 5, 0]));
        assert_eq!(v.t_val, vec![5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!((v.ov_off, v.ov_opt, v.ov_tok), (vec![0, 2, 3], vec![0, 2, 1], vec![(2 << 40) | 7, -1, -1]));
        // a game without tails (empty CSR in the batch): B + 1 zeros
        let mut c = b.clone();
        (c.t_off, c.t_seg, c.t_slot, c.t_val) = (vec![], vec![], vec![], vec![]);
        assert_eq!(varlen_piece(&c, &[1, 2], &widths).unwrap().t_off, vec![0, 0, 0]);
        // a token of a segment the network has no tail for, and floats that do not fit the tokens
        assert!(varlen_piece(&b, &[0], &[3, 0, 0]).unwrap_err().contains("segment 2"));
        let mut d = b.clone();
        d.t_val.pop();
        assert!(varlen_piece(&d, &[0], &widths).is_err());
    }
}
