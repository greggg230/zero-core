//! The probe's graph variants (zero-farm PROTOCOL 8 `probe-providers`): a diagnostic worker build embeds a set of
//! graphs (`ZERO_PROBE_VARIANTS` at build time, `build.rs`; made by `tools/onnx_probe_variants.py`) and
//! `probe-providers` runs each of them on every provider this PC offers, after the usual probe, so one round trip
//! to a PC we can't reach (a friend's AMD GPU) answers many questions at once: which graph shapes a provider
//! fails on and why, and how fast each provider is on the real networks at real batch sizes.
//!
//! Two kinds of variant:
//! * `raw`: any graph with its own inputs and expected outputs (`<case>/in.<name>.npy`, `out.<name>.npy`):
//!   micro graphs that each test one way an execution provider can fail (an output that aliases an input, a
//!   constant output, an empty output, one operator...), and cuts of the probe graph;
//! * `net`: a zero-onnx/1 graph fed real rows (a reference set's, cycled to each batch size) - the job's own graph
//!   from the agent's cache, or an embedded variant of it (another opset, another packing, static shapes: one
//!   session per batch size with free-dimension overrides, the batch padded by one dummy row carrying the padding
//!   non-zeros). A GPU provider's answers are compared with the CPU's on the same graph and rows (max abs per
//!   output, twice bit-identical), and every provider is timed (rows/s per batch size; the CPU at several thread
//!   counts).
//!
//! Each (group, provider) runs in a child process (`probe-variant`): a provider that crashes the process (a
//! driver's access violation) costs only that variant, and the child is started again on the next one. Results
//! stream to the worker log one line per variant as they come; the whole set stays under the probe's deadline
//! (the agent stops a probe after 15 minutes). The variants never change the probe's verdict (`providers`):
//! they are reported under `variants` only.
use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ort::session::Session;
use ort::value::Tensor;
use serde_json::{json, Map, Value};

use crate::gpu::Adapter;
use crate::npz::{read_npz, Arr};
use crate::session::{runtime, variant_session, GpuDevice, ProviderKind};

#[cfg(probe_variants)]
pub const ARCHIVE: &[u8] = include_bytes!(env!("ZERO_PROBE_VARIANTS_FILE"));
#[cfg(not(probe_variants))]
pub const ARCHIVE: &[u8] = &[];

/// Whether this build carries variants (then `probe-providers` runs them unless `--no-variants`).
pub fn embedded() -> bool {
    !ARCHIVE.is_empty()
}

/// What the parent runs.
pub struct Plan<'a> {
    /// The runtime pack's onnxruntime.dll (the children load the same one).
    pub dylib: &'a Path,
    pub adapters: &'a [Adapter],
    /// Only this adapter (LUID), as the probe's `--gpu`.
    pub gpu: Option<u64>,
    /// The providers to try (the CPU included), in the probe's order.
    pub providers: Vec<ProviderKind>,
    pub threads: usize,
    /// The CPU thread counts timed on the real networks.
    pub cpu_threads: Vec<usize>,
    /// Stop starting children after this; running ones are stopped at it.
    pub deadline: Instant,
    /// A writable directory (removed afterwards by the caller).
    pub work: &'a Path,
    /// Real networks from the agent's cache: (label, model.onnx, reference.npz).
    pub cache_nets: Vec<(String, PathBuf, PathBuf)>,
    /// Only variants whose name starts with one of these (empty: all).
    pub only: Vec<String>,
}

fn arr_bytes(a: &Arr) -> &[u8] {
    &a.data
}

/// The embedded archive -> (manifest, every member written out as a file under `dir`: models and npy files).
fn unpack(dir: &Path) -> Result<Value, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let z = dir.join("variants.npz");
    std::fs::write(&z, ARCHIVE).map_err(|e| format!("{}: {e}", z.display()))?;
    let m = read_npz(&z)?;
    let man = m.get("manifest").ok_or("variants: no manifest")?;
    let manifest: Value = serde_json::from_slice(arr_bytes(man)).map_err(|e| format!("variants manifest: {e}"))?;
    for (name, a) in &m {
        if name == "manifest" {
            continue;
        }
        let p = dir.join(name);
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        }
        // members holding whole files (graphs, reference sets) are byte arrays; the others are arrays (npy)
        if name.ends_with(".onnx") || name.ends_with(".npz") {
            std::fs::write(&p, arr_bytes(a)).map_err(|e| format!("{}: {e}", p.display()))?;
        } else {
            write_npy(&dir.join(format!("{name}.npy")), a)?;
        }
    }
    let _ = std::fs::remove_file(&z);
    Ok(manifest)
}

fn write_npy(p: &Path, a: &Arr) -> Result<(), String> {
    let shape = match a.shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", a.shape[0]),
        _ => format!("({})", a.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut h = format!("{{'descr': '{}', 'fortran_order': False, 'shape': {shape}, }}", a.descr);
    while (10 + h.len() + 1) % 64 != 0 {
        h.push(' ');
    }
    h.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(h.len() as u16).to_le_bytes());
    out.extend_from_slice(h.as_bytes());
    out.extend_from_slice(&a.data);
    std::fs::write(p, out).map_err(|e| format!("{}: {e}", p.display()))
}

fn read_npy(p: &Path) -> Result<Arr, String> {
    // (read_npz reads zips; a lone .npy is the same member format)
    let raw = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    crate::npz::parse_npy_bytes(&raw).map_err(|e| format!("{}: {e}", p.display()))
}

/// The manifest's variants with their files resolved under `dir`, plus the cache's real networks.
fn resolve(manifest: &Value, dir: &Path, plan: &Plan) -> Vec<Value> {
    let mut out = Vec::new();
    let fix = |s: &str| dir.join(s).display().to_string();
    for v in manifest["variants"].as_array().into_iter().flatten() {
        let mut v = v.clone();
        if let Some(m) = v["model"].as_str() {
            v["model"] = json!(fix(m));
        }
        if let Some(r) = v["rows"].as_str() {
            v["rows"] = json!(fix(r));
        }
        if let Some(cases) = v["cases"].as_array_mut() {
            for c in cases {
                for k in ["inputs", "expect"] {
                    if let Some(o) = c[k].as_object_mut() {
                        for (_, f) in o.iter_mut() {
                            if let Some(s) = f.as_str() {
                                *f = json!(format!("{}.npy", fix(s)));
                            }
                        }
                    }
                }
            }
        }
        out.push(v);
    }
    let n_cache = manifest["cache_nets"].as_u64().map(|n| n as usize).unwrap_or(usize::MAX);
    for (label, model, rows) in plan.cache_nets.iter().take(n_cache) {
        out.push(json!({"name": format!("cache_{label}"), "group": "cache", "kind": "net", "model": model.display().to_string(),
                        "rows": rows.display().to_string(), "batches": [4, 16, 64, 256, 1024], "static": false, "tol": crate::selftest::TOL,
                        "cpu_threads": plan.cpu_threads, "budget_s": 240, "note": "the job's own graph from the agent's cache",
                        // (not the deprecated MIGraphX provider: AMD's current one stands for it on the real networks)
                        "providers": ["cpu", "amdgpu", "dml", "tensorrt_rtx", "cuda", "openvino"]}));
    }
    out.retain(|v| plan.only.is_empty() || plan.only.iter().any(|o| v["name"].as_str().unwrap_or("").starts_with(o.as_str())));
    out
}

/// One child's work: these variants on one provider.
struct ChildJob {
    group: String,
    provider: ProviderKind,
    gpu: Option<GpuDevice>,
    variants: Vec<Value>,
    env: Vec<(String, String)>,
    /// Seconds the whole job may take.
    budget: f64,
}

fn variant_budget(v: &Value, p: ProviderKind) -> f64 {
    let base = v["budget_s"].as_f64().unwrap_or(match v["kind"].as_str() {
        Some("net") => 150.0,
        _ => 30.0,
    });
    // (the CPU builds in a fraction of a GPU provider's compile time)
    if p == ProviderKind::Cpu {
        base * 0.6
    } else {
        base
    }
}

/// The variants run on this PC, children per (group, provider), within the deadline.
pub fn run_all(plan: &Plan) -> Value {
    let t0 = Instant::now();
    let dir = plan.work.join("variants");
    let manifest = match unpack(&dir) {
        Ok(m) => m,
        Err(e) => return json!({"error": e}),
    };
    let variants = resolve(&manifest, &dir, plan);
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => return json!({"error": format!("current_exe: {e}")}),
    };
    // every (provider, adapter) the probe would try, the CPU first
    let mut targets: Vec<(ProviderKind, Option<GpuDevice>)> = Vec::new();
    for &p in &plan.providers {
        if p.is_gpu() {
            for a in plan.adapters.iter().filter(|a| p.serves(a) && plan.gpu.is_none_or(|l| l == a.luid)) {
                targets.push((p, Some(GpuDevice::of(a))));
            }
        } else {
            targets.push((p, None));
        }
    }
    // jobs in order of value: the groups in the manifest's order; within a group the variant's providers in the
    // probe's preference order, the CPU last for diagnostic groups (its answers are the expected ones) and first
    // for timed ones (the baseline)
    let mut groups: Vec<String> = Vec::new();
    for v in &variants {
        let g = v["group"].as_str().unwrap_or("misc").to_string();
        if !groups.contains(&g) {
            groups.push(g);
        }
    }
    // (the manifest's `order` first, the cache's real networks where it puts "cache", the rest after)
    let order: Vec<String> = manifest["order"].as_array().into_iter().flatten().filter_map(Value::as_str).map(String::from).collect();
    groups.sort_by_key(|g| order.iter().position(|o| o == g).unwrap_or(order.len()));
    let mut jobs: Vec<ChildJob> = Vec::new();
    for g in &groups {
        let in_group: Vec<&Value> = variants.iter().filter(|v| v["group"].as_str().unwrap_or("misc") == g).collect();
        let timed = in_group.iter().any(|v| v["kind"] == json!("net"));
        let mut order = targets.clone();
        // (the deprecated MIGraphX provider after AMD's current one)
        order.sort_by_key(|(p, _)| match p {
            ProviderKind::Cpu => if timed { 0 } else { 9 },
            ProviderKind::AmdGpu => 1,
            ProviderKind::DirectMl => 2,
            ProviderKind::MiGraphX => 5,
            _ => 3,
        });
        for (p, d) in order {
            let mine: Vec<Value> = in_group
                .iter()
                .filter(|v| v["providers"].as_array().is_none_or(|a| a.iter().any(|x| x.as_str() == Some(p.wire()))))
                .map(|v| (*v).clone())
                .collect();
            if mine.is_empty() {
                continue;
            }
            let budget = mine.iter().map(|v| variant_budget(v, p)).sum();
            jobs.push(ChildJob { group: g.clone(), provider: p, gpu: d.clone(), variants: mine.clone(), env: Vec::new(), budget });
            // the AMD providers' partition dump (ORT_MIGRAPHX_DUMP_MODEL_OPS: the subgraphs they compile, if the
            // provider honours it), once, on the full probe graph
            if matches!(p, ProviderKind::AmdGpu | ProviderKind::MiGraphX) {
                if let Some(full) = mine.iter().find(|v| v["dump"] == json!(true)) {
                    let mut dv = full.clone();
                    dv["name"] = json!(format!("{}_dump", full["name"].as_str().unwrap_or("?")));
                    jobs.push(ChildJob {
                        group: format!("{g}_dump"),
                        provider: p,
                        gpu: d.clone(),
                        variants: vec![dv],
                        env: vec![("ORT_MIGRAPHX_DUMP_MODEL_OPS".into(), "1".into()), ("ORT_MIGRAPHX_DUMP_MODEL_OPS_PATH".into(), ".".into())],
                        budget: 60.0,
                    });
                }
            }
        }
    }
    let mut results: Vec<Value> = Vec::new();
    let mut skipped: Vec<Value> = Vec::new();
    eprintln!("  variants: {} graph variant(s), {} child job(s) (deadline in {:.0} s)", variants.len(), jobs.len(), plan.deadline.saturating_duration_since(Instant::now()).as_secs_f64());
    for (k, job) in jobs.into_iter().enumerate() {
        let mut left: Vec<Value> = job.variants.clone();
        let mut restarts = 0;
        while !left.is_empty() {
            if Instant::now() + Duration::from_secs(5) >= plan.deadline {
                for v in &left {
                    skipped.push(json!({"variant": v["name"], "provider": job.provider.wire(), "why": "the probe's time ran out"}));
                }
                break;
            }
            let cdir = plan.work.join(format!("child-{k}-{restarts}"));
            let _ = std::fs::create_dir_all(&cdir);
            let (done, crashed) = run_child(&exe, plan, &job, &left, &cdir);
            for r in &done {
                eprintln!("{}", line(r));
            }
            let n_done = done.len();
            results.extend(done);
            left.drain(..n_done.min(left.len()));
            match crashed {
                None => break,
                Some(why) => {
                    if let Some(v) = left.first() {
                        let r = json!({"variant": v["name"], "group": job.group, "provider": job.provider.wire(), "gpu": job.gpu.as_ref().map(|d| d.to_json()), "ok": false, "why": why, "crashed": true});
                        eprintln!("{}", line(&r));
                        results.push(r);
                    }
                    left.drain(..1.min(left.len()));
                    restarts += 1;
                    if restarts > 12 {
                        for v in &left {
                            skipped.push(json!({"variant": v["name"], "provider": job.provider.wire(), "why": "too many crashes"}));
                        }
                        break;
                    }
                }
            }
        }
        // a dump job: the subgraphs the provider wrote, read back through the CPU
        if job.env.iter().any(|(k, _)| k == "ORT_MIGRAPHX_DUMP_MODEL_OPS") {
            let dumps = read_dumps(plan, &plan.work.join(format!("child-{k}-0")));
            eprintln!("  variant dump {} on {}: {} subgraph file(s) {}", job.group, job.provider.wire(), dumps.len(), Value::Array(dumps.clone()));
            results.push(json!({"variant": format!("{}:subgraphs", job.group), "provider": job.provider.wire(), "subgraphs": dumps}));
        }
    }
    json!({"variants": variants.len(), "results": results, "skipped": skipped, "seconds": (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0})
}

/// The `.onnx` files a dump job's child left in its working directory: each one's inputs and outputs.
fn read_dumps(plan: &Plan, dir: &Path) -> Vec<Value> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else { return out };
    let Ok(rt) = runtime(plan.dylib) else { return out };
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().and_then(|x| x.to_str()) != Some("onnx") {
            continue;
        }
        let size = e.metadata().map(|m| m.len()).unwrap_or(0);
        let io = match variant_session(rt, &p, ProviderKind::Cpu, None, plan.adapters, 1, &[], None) {
            Ok((s, _, _)) => json!({
                "inputs": s.inputs().iter().map(|o| format!("{}: {:?}", o.name(), o.dtype())).collect::<Vec<_>>(),
                "outputs": s.outputs().iter().map(|o| format!("{}: {:?}", o.name(), o.dtype())).collect::<Vec<_>>(),
            }),
            Err(e) => json!({"error": e}),
        };
        out.push(json!({"file": p.file_name().map(|f| f.to_string_lossy().to_string()), "bytes": size, "io": io}));
    }
    out
}

/// One variant result as a worker-log line.
pub fn line(r: &Value) -> String {
    let gpu = r["gpu"]["name"].as_str().map(|n| format!(" on {n}")).unwrap_or_default();
    let head = format!("  variant {} {}{}:", r["variant"].as_str().unwrap_or("?"), r["provider"].as_str().unwrap_or("?"), gpu);
    if r["ok"] != json!(true) && r.get("batches").is_none() && r.get("cases").is_none() {
        return format!("{head} FAIL {}", r["why"].as_str().unwrap_or("?"));
    }
    let mut s = format!("{head} {}", if r["ok"] == json!(true) { "PASS" } else { "FAIL" });
    if let Some(w) = r["why"].as_str() {
        s.push_str(&format!(" ({w})"));
    }
    if let Some(n) = r["cpu_nodes"].as_array() {
        if r["provider"] != json!("cpu") {
            s.push_str(&format!(", {} nodes on the CPU", n.len()));
        }
    }
    if let Some(b) = r["build_s"].as_f64() {
        s.push_str(&format!(", build {b:.2} s"));
    }
    if let Some(cs) = r["cases"].as_array() {
        let bad: Vec<String> = cs.iter().filter(|c| c["ok"] != json!(true)).map(|c| format!("{} ({})", c["name"].as_str().unwrap_or("?"), c["why"].as_str().map(String::from).unwrap_or_else(|| format!("max abs {}", c["max_abs"])))).collect();
        s.push_str(&format!(", cases {}/{} ok", cs.len() - bad.len(), cs.len()));
        if !bad.is_empty() {
            s.push_str(&format!(": {}", bad.join("; ")));
        }
    }
    // a GPU provider that failed: the nodes it left to the CPU, by name (the log is all a remote PC sends back; the
    // partition's edges follow from them and the graph)
    if r["ok"] != json!(true) && r["provider"] != json!("cpu") {
        if let Some(n) = r["cpu_nodes"].as_array().filter(|n| !n.is_empty()) {
            let names: Vec<&str> = n.iter().filter_map(Value::as_str).collect();
            let mut t = names.join(", ");
            if t.len() > 3000 {
                t = format!("{}...", t.chars().take(3000).collect::<String>());
            }
            s.push_str(&format!(" | nodes on the CPU: {t}"));
        }
    }
    if let Some(b) = r["batches"].as_object() {
        let mut b: Vec<(&String, &Value)> = b.iter().collect();
        b.sort_by_key(|(n, _)| n.parse::<usize>().unwrap_or(usize::MAX));
        let parts: Vec<String> = b
            .into_iter()
            .map(|(n, x)| {
                let mut t = format!("{n}: {}", x["rows_per_s"].as_f64().map(|v| format!("{v:.0}/s")).unwrap_or_else(|| x["why"].as_str().unwrap_or("-").chars().take(120).collect()));
                if let Some(m) = x["max_abs"].as_f64() {
                    t.push_str(&format!(" (max abs {m:.1e}{})", if x["deterministic"] == json!(false) { ", NOT deterministic" } else { "" }));
                }
                t
            })
            .collect();
        s.push_str(&format!("; rows/s by batch {}", parts.join(", ")));
    }
    if let Some(t) = r["cpu_threads"].as_object() {
        let mut t: Vec<(&String, &Value)> = t.iter().collect();
        t.sort_by_key(|(n, _)| n.parse::<usize>().unwrap_or(usize::MAX));
        for (th, b) in t {
            let mut b: Vec<(&String, &Value)> = b.as_object().into_iter().flatten().collect();
            b.sort_by_key(|(n, _)| n.parse::<usize>().unwrap_or(usize::MAX));
            let parts: Vec<String> = b.into_iter().map(|(n, x)| format!("{n}: {}", x.as_f64().map(|v| format!("{v:.0}")).unwrap_or("-".into()))).collect();
            s.push_str(&format!("; cpu {th} threads {}", parts.join(", ")));
        }
    }
    s
}

/// Runs one child on `left` -> (the results it finished, why it stopped early if it did).
fn run_child(exe: &Path, plan: &Plan, job: &ChildJob, left: &[Value], cdir: &Path) -> (Vec<Value>, Option<String>) {
    let spec = json!({"provider": job.provider.wire(), "gpu": job.gpu.as_ref().map(|d| crate::gpu::luid_hex(d.luid)), "threads": plan.threads,
                      "group": job.group, "variants": left});
    let sp = cdir.join("spec.json");
    let out = cdir.join("out.jsonl");
    if let Err(e) = std::fs::write(&sp, spec.to_string()) {
        return (Vec::new(), Some(format!("spec: {e}")));
    }
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("probe-variant").arg("--ort-dll").arg(plan.dylib).arg("--spec").arg(&sp).arg("--out").arg(&out).current_dir(cdir);
    cmd.stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null());
    for (k, v) in &job.env {
        cmd.env(k, v);
    }
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return (Vec::new(), Some(format!("could not start: {e}"))),
    };
    let stop_at = std::cmp::min(plan.deadline, Instant::now() + Duration::from_secs_f64(job.budget.max(10.0)));
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Some(s),
            Ok(None) => {}
            Err(_) => break None,
        }
        if Instant::now() >= stop_at {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let mut done = Vec::new();
    let mut started: Option<String> = None;
    if let Ok(f) = std::fs::File::open(&out) {
        for l in std::io::BufReader::new(f).lines().map_while(Result::ok) {
            let Ok(v) = serde_json::from_str::<Value>(&l) else { continue };
            if let Some(s) = v["start"].as_str() {
                started = Some(s.to_string());
            } else {
                started = None;
                done.push(v);
            }
        }
    }
    let why = match status {
        Some(s) if s.success() && done.len() >= left.len() => None,
        Some(s) => Some(format!("the child process exited ({s}) during {}", started.as_deref().unwrap_or("its start"))),
        None if Instant::now() >= plan.deadline => Some(format!("stopped at the probe's deadline during {}", started.as_deref().unwrap_or("its start"))),
        None => Some(format!("no answer within {:.0} s (stopped) during {}", job.budget, started.as_deref().unwrap_or("its start"))),
    };
    (done, why)
}

// ---------------------------------------------------------------------------------------------------------------
// the child: `probe-variant --ort-dll D --spec spec.json --out out.jsonl`

/// The child's entry: each variant of the spec on its provider, one JSON line per variant (and a `start` line
/// before each, so a crash is attributed).
pub fn child_main(dylib: &Path, spec: &Path, out: &Path) -> Result<i32, String> {
    let spec: Value = serde_json::from_slice(&std::fs::read(spec).map_err(|e| format!("{}: {e}", spec.display()))?).map_err(|e| format!("spec: {e}"))?;
    let p = ProviderKind::parse(spec["provider"].as_str().unwrap_or("")).ok_or("spec: provider")?;
    let threads = spec["threads"].as_u64().unwrap_or(2) as usize;
    let rt = runtime(dylib)?;
    let adapters = crate::gpu::adapters();
    if crate::session::CATALOG.contains(&p) {
        let _ = rt.catalog(&[p], false);
    }
    let want = match spec["gpu"].as_str() {
        Some(l) => {
            let luid = crate::gpu::parse_luid(l).ok_or("spec: gpu")?;
            adapters.iter().find(|a| a.luid == luid).map(GpuDevice::of)
        }
        None => None,
    };
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(out).map_err(|e| format!("{}: {e}", out.display()))?;
    for v in spec["variants"].as_array().into_iter().flatten() {
        let name = v["name"].as_str().unwrap_or("?").to_string();
        writeln!(f, "{}", json!({"start": name})).map_err(|e| e.to_string())?;
        f.flush().map_err(|e| e.to_string())?;
        let t0 = Instant::now();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_variant(rt, v, p, want.as_ref(), &adapters, threads)));
        let mut r = match r {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => json!({"ok": false, "why": e}),
            Err(_) => json!({"ok": false, "why": "panicked"}),
        };
        r["variant"] = json!(name);
        r["group"] = v["group"].clone();
        r["provider"] = json!(p.wire());
        if r.get("gpu").is_none() {
            r["gpu"] = json!(want.as_ref().map(|d| d.to_json()));
        }
        r["seconds"] = json!((t0.elapsed().as_secs_f64() * 100.0).round() / 100.0);
        writeln!(f, "{r}").map_err(|e| e.to_string())?;
        f.flush().map_err(|e| e.to_string())?;
    }
    Ok(0)
}

fn placement_json(r: &mut Value, placed: &[(String, String)], p: ProviderKind) {
    if !p.is_gpu() {
        return;
    }
    r["cpu_nodes"] = json!(placed.iter().filter(|(prov, _)| prov == "CPUExecutionProvider").map(|(_, n)| n.clone()).collect::<Vec<_>>());
    let mut m = BTreeMap::<String, u64>::new();
    for (prov, _) in placed {
        *m.entry(prov.clone()).or_default() += 1;
    }
    r["placed"] = json!(m);
}

fn run_variant(rt: &'static crate::session::Runtime, v: &Value, p: ProviderKind, want: Option<&GpuDevice>, adapters: &[Adapter], threads: usize) -> Result<Value, String> {
    match v["kind"].as_str() {
        Some("raw") => run_raw(rt, v, p, want, adapters, threads),
        Some("net") => run_net(rt, v, p, want, adapters, threads),
        k => Err(format!("unknown variant kind {k:?}")),
    }
}

/// A variant's graph optimisation level (`opt`: "disable", "basic", "extended"; else the factory's).
fn opt_of(v: &Value) -> Result<Option<ort::session::builder::GraphOptimizationLevel>, String> {
    use ort::session::builder::GraphOptimizationLevel as L;
    Ok(match v["opt"].as_str() {
        None => None,
        Some("disable") => Some(L::Disable),
        Some("basic") => Some(L::Level1),
        Some("extended") => Some(L::Level2),
        Some(o) => return Err(format!("opt {o:?}")),
    })
}

fn dims_of(v: &Value) -> Vec<(String, i64)> {
    v.as_object().into_iter().flatten().filter_map(|(k, x)| x.as_i64().map(|n| (k.clone(), n))).collect()
}

/// An npy array as an ONNX Runtime input.
fn input_value(a: &Arr) -> Result<ort::session::SessionInputValue<'static>, String> {
    let shape: Vec<i64> = a.shape.iter().map(|&d| d as i64).collect();
    let oe = |e: ort::Error| e.to_string();
    Ok(match a.descr.as_str() {
        "<f4" => Tensor::from_array((shape, a.data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<f32>>())).map_err(oe)?.into(),
        "<i8" => Tensor::from_array((shape, a.data.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect::<Vec<i64>>())).map_err(oe)?.into(),
        "<i4" => Tensor::from_array((shape, a.data.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<i32>>())).map_err(oe)?.into(),
        "|b1" => Tensor::from_array((shape, a.data.iter().map(|&b| b != 0).collect::<Vec<bool>>())).map_err(oe)?.into(),
        "|u1" => Tensor::from_array((shape, a.data.clone())).map_err(oe)?.into(),
        d => return Err(format!("input dtype {d}")),
    })
}

/// An output's numbers as f64 (with its shape), by the expected array's dtype.
fn output_numbers(out: &ort::session::SessionOutputs, name: &str, descr: &str) -> Result<(Vec<usize>, Vec<f64>), String> {
    let o = out.get(name).ok_or_else(|| format!("no output {name}"))?;
    let oe = |e: ort::Error| format!("output {name}: {e}");
    let (shape, v): (Vec<usize>, Vec<f64>) = match descr {
        "<f4" => {
            let (s, d) = o.try_extract_tensor::<f32>().map_err(oe)?;
            (s.iter().map(|&x| x as usize).collect(), d.iter().map(|&x| x as f64).collect())
        }
        "<i8" => {
            let (s, d) = o.try_extract_tensor::<i64>().map_err(oe)?;
            (s.iter().map(|&x| x as usize).collect(), d.iter().map(|&x| x as f64).collect())
        }
        "<i4" => {
            let (s, d) = o.try_extract_tensor::<i32>().map_err(oe)?;
            (s.iter().map(|&x| x as usize).collect(), d.iter().map(|&x| x as f64).collect())
        }
        "|b1" => {
            let (s, d) = o.try_extract_tensor::<bool>().map_err(oe)?;
            (s.iter().map(|&x| x as usize).collect(), d.iter().map(|&x| x as u8 as f64).collect())
        }
        d => return Err(format!("output dtype {d}")),
    };
    Ok((shape, v))
}

fn arr_numbers(a: &Arr) -> Vec<f64> {
    match a.descr.as_str() {
        "<f4" => a.data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
        "<i8" => a.data.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
        "<i4" => a.data.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
        "|b1" | "|u1" => a.data.iter().map(|&b| b as f64).collect(),
        _ => Vec::new(),
    }
}

/// Max abs difference (infinities equal when equal; a NaN on either side is infinite).
fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| if x == y { 0.0 } else if x.is_nan() || y.is_nan() { f64::INFINITY } else { (x - y).abs() })
        .fold(0.0, f64::max)
}

fn r4(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}

fn run_raw(rt: &'static crate::session::Runtime, v: &Value, p: ProviderKind, want: Option<&GpuDevice>, adapters: &[Adapter], threads: usize) -> Result<Value, String> {
    let model = PathBuf::from(v["model"].as_str().ok_or("model")?);
    let tol = v["tol"].as_f64().unwrap_or(1e-5);
    let mut r = json!({});
    let mut sessions: Vec<(Vec<(String, i64)>, Session)> = Vec::new();
    let mut cases = Vec::new();
    let mut all_ok = true;
    for c in v["cases"].as_array().into_iter().flatten() {
        let dims = dims_of(&c["dims"]);
        if !sessions.iter().any(|(d, _)| *d == dims) {
            let t = Instant::now();
            match variant_session(rt, &model, p, want, adapters, threads, &dims, opt_of(v)?) {
                Ok((s, placed, dev)) => {
                    if r.get("build_s").is_none() {
                        r["build_s"] = json!(r4(t.elapsed().as_secs_f64()));
                        placement_json(&mut r, &placed, p);
                        r["gpu"] = json!(dev.map(|d| d.to_json()));
                    }
                    sessions.push((dims.clone(), s));
                }
                Err(e) => return Ok(json!({"ok": false, "why": format!("build: {e}"), "build_s": r4(t.elapsed().as_secs_f64())})),
            }
        }
        let s = &mut sessions.iter_mut().find(|(d, _)| *d == dims).unwrap().1;
        let cname = c["name"].as_str().unwrap_or("?").to_string();
        let mut one = || -> Result<Value, String> {
            let mut feeds: Vec<(String, ort::session::SessionInputValue<'static>)> = Vec::new();
            for (k, f) in c["inputs"].as_object().into_iter().flatten() {
                feeds.push((k.clone(), input_value(&read_npy(Path::new(f.as_str().unwrap_or("")))?)?));
            }
            let expect: Vec<(String, Arr)> = c["expect"].as_object().into_iter().flatten().map(|(k, f)| Ok((k.clone(), read_npy(Path::new(f.as_str().unwrap_or("")))?))).collect::<Result<_, String>>()?;
            let t = Instant::now();
            let o1 = s.run(feeds).map_err(|e| format!("run: {e}"))?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let mut first: Vec<(Vec<usize>, Vec<f64>)> = Vec::new();
            for (k, a) in &expect {
                first.push(output_numbers(&o1, k, &a.descr)?);
            }
            drop(o1);
            let mut feeds2: Vec<(String, ort::session::SessionInputValue<'static>)> = Vec::new();
            for (k, f) in c["inputs"].as_object().into_iter().flatten() {
                feeds2.push((k.clone(), input_value(&read_npy(Path::new(f.as_str().unwrap_or("")))?)?));
            }
            let o2 = s.run(feeds2).map_err(|e| format!("second run: {e}"))?;
            let mut mab = Map::new();
            let mut ok = true;
            let mut det = true;
            let mut shapes_ok = true;
            for ((k, a), (shape, got)) in expect.iter().zip(&first) {
                let (_, again) = output_numbers(&o2, k, &a.descr)?;
                det &= again.len() == got.len() && again.iter().zip(got).all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()));
                if *shape != a.shape {
                    shapes_ok = false;
                    ok = false;
                    mab.insert(k.clone(), json!(format!("shape {shape:?}, want {:?}", a.shape)));
                    continue;
                }
                let m = max_abs(got, &arr_numbers(a));
                ok &= m <= tol;
                mab.insert(k.clone(), json!(m));
            }
            Ok(json!({"name": cname, "ok": ok && det, "max_abs": mab, "deterministic": det, "shape_ok": shapes_ok, "ms": r4(ms)}))
        };
        let res = one().unwrap_or_else(|e| json!({"name": cname, "ok": false, "why": e}));
        all_ok &= res["ok"] == json!(true);
        cases.push(res);
    }
    r["ok"] = json!(all_ok);
    r["cases"] = json!(cases);
    Ok(r)
}

// --- net variants

/// zero-onnx/1 inputs of rows `idx` of `b`, padded to `pad` = (rows, non-zeros) by one dummy row when given.
struct NetFeed {
    x_ptr: Vec<i64>,
    x_idx: Vec<i32>,
    x_val: Vec<f32>,
    np: Vec<i64>,
    bel: Vec<bool>,
}

fn net_feed(b: &zero_search::driver::Batch, pad_nnz: Option<usize>) -> Result<NetFeed, String> {
    let mut f = NetFeed { x_ptr: vec![0], x_idx: Vec::new(), x_val: Vec::new(), np: Vec::new(), bel: Vec::new() };
    for r in 0..b.b {
        let (lo, hi) = (b.xs_off[r] as usize, b.xs_off[r + 1] as usize);
        for k in lo..hi {
            f.x_idx.push(i32::try_from(b.xs_idx[k]).map_err(|_| "feature index beyond i32".to_string())?);
        }
        f.x_val.extend_from_slice(&b.xs_val[lo..hi]);
        f.x_ptr.push(f.x_idx.len() as i64);
        f.np.push(b.n_players[r] as i64);
        f.bel.push(b.extra[r] == zero_search::driver::ROW_BELIEF);
    }
    if let Some(cap) = pad_nnz {
        // one dummy row: the padding non-zeros (feature 0, value 0: duplicates write identical zeros)
        let extra = cap.checked_sub(f.x_idx.len()).filter(|&e| e >= 1).ok_or("padding: the cap is below the batch's non-zeros")?;
        f.x_idx.extend(std::iter::repeat_n(0i32, extra));
        f.x_val.extend(std::iter::repeat_n(0f32, extra));
        f.x_ptr.push(f.x_idx.len() as i64);
        f.np.push(2);
        f.bel.push(false);
    }
    Ok(f)
}

fn run_net_once(s: &mut Session, f: &NetFeed) -> Result<[(Vec<usize>, Vec<f32>); 4], String> {
    let n = f.np.len() as i64;
    let nnz = f.x_idx.len() as i64;
    let oe = |e: ort::Error| e.to_string();
    let inputs = ort::inputs![
        "x_ptr" => Tensor::from_array((vec![n + 1], f.x_ptr.clone())).map_err(oe)?,
        "x_idx" => Tensor::from_array((vec![nnz], f.x_idx.clone())).map_err(oe)?,
        "x_val" => Tensor::from_array((vec![nnz], f.x_val.clone())).map_err(oe)?,
        "n_players" => Tensor::from_array((vec![n], f.np.clone())).map_err(oe)?,
        "belief" => Tensor::from_array((vec![n], f.bel.clone())).map_err(oe)?,
    ];
    let out = s.run(inputs).map_err(|e| format!("run: {e}"))?;
    let take = |name: &str| -> Result<(Vec<usize>, Vec<f32>), String> {
        let (shape, data) = out[name].try_extract_tensor::<f32>().map_err(|e| format!("output {name}: {e}"))?;
        Ok((shape.iter().map(|&x| x as usize).collect(), data.to_vec()))
    };
    Ok([take("policy")?, take("value")?, take("margin")?, take("aux")?])
}

const OUTS: [&str; 4] = ["policy", "value", "margin", "aux"];

/// Times `f` (>= 3 runs and >= 0.3 s, at most 30 runs or 8 s) -> (runs per second, median ms).
fn timed(mut f: impl FnMut() -> Result<(), String>) -> Result<(f64, f64), String> {
    let t0 = Instant::now();
    let mut ms = Vec::new();
    while ms.len() < 30 && (ms.len() < 3 || t0.elapsed().as_secs_f64() < 0.3) && !(ms.len() >= 1 && t0.elapsed().as_secs_f64() > 8.0) {
        let t = Instant::now();
        f()?;
        ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let total = t0.elapsed().as_secs_f64();
    ms.sort_by(|a, b| a.total_cmp(b));
    Ok((ms.len() as f64 / total, ms[ms.len() / 2]))
}

fn run_net(rt: &'static crate::session::Runtime, v: &Value, p: ProviderKind, want: Option<&GpuDevice>, adapters: &[Adapter], threads: usize) -> Result<Value, String> {
    let model = PathBuf::from(v["model"].as_str().ok_or("model")?);
    let rows = PathBuf::from(v["rows"].as_str().ok_or("rows")?);
    let set = crate::selftest::load(&rows)?;
    let tol = v["tol"].as_f64().unwrap_or(crate::selftest::TOL);
    let stat = v["static"] == json!(true);
    let sizes: Vec<usize> = v["batches"].as_array().map(|a| a.iter().filter_map(Value::as_u64).map(|n| n as usize).collect()).unwrap_or_else(|| vec![16, 64, 256, 1024]);
    let batches: Vec<(usize, zero_search::driver::Batch)> = sizes.iter().map(|&n| (n, crate::probe::cycle_batch(&set, n))).collect();
    // static shapes: B = rows + 1 (the dummy row), nnz = the real non-zeros + at least 1, rounded up to 1,024
    let cap_of = |b: &zero_search::driver::Batch| (b.xs_off[b.b] as usize + 1).div_ceil(1024) * 1024;
    let dims_for = |b: &zero_search::driver::Batch| -> Vec<(String, i64)> {
        vec![("B".into(), b.b as i64 + 1), ("B + 1".into(), b.b as i64 + 2), ("nnz".into(), cap_of(b) as i64)]
    };
    let mut r = json!({"static": stat, "rows": set.rows});
    if p == ProviderKind::Cpu {
        // the baseline: rows/s at each batch size, at each thread count
        let mut by_t = Map::new();
        let ths: Vec<usize> = v["cpu_threads"].as_array().map(|a| a.iter().filter_map(Value::as_u64).map(|n| n as usize).collect()).filter(|x: &Vec<usize>| !x.is_empty()).unwrap_or_else(|| vec![threads]);
        let mut bmap = Map::new();
        for &t in &ths {
            let mut m = Map::new();
            let mut dynamic: Option<Session> = None;
            for (n, b) in &batches {
                let res = (|| -> Result<f64, String> {
                    let (mut s_static, pad) = if stat { (Some(variant_session(rt, &model, p, None, adapters, t, &dims_for(b), opt_of(v)?)?.0), Some(cap_of(b))) } else { (None, None) };
                    if !stat && dynamic.is_none() {
                        let tb = Instant::now();
                        dynamic = Some(variant_session(rt, &model, p, None, adapters, t, &[], opt_of(v)?)?.0);
                        if r.get("build_s").is_none() {
                            r["build_s"] = json!(r4(tb.elapsed().as_secs_f64()));
                        }
                    }
                    let s = s_static.as_mut().or(dynamic.as_mut()).unwrap();
                    let f = net_feed(b, pad)?;
                    run_net_once(s, &f)?;
                    let (per_s, _) = timed(|| run_net_once(s, &f).map(|_| ()))?;
                    Ok(per_s * *n as f64)
                })();
                match res {
                    Ok(x) => {
                        m.insert(n.to_string(), json!(x.round()));
                        if t == threads || (t == ths[0] && !ths.contains(&threads)) {
                            bmap.insert(n.to_string(), json!({"rows_per_s": x.round()}));
                        }
                    }
                    Err(e) => {
                        m.insert(n.to_string(), json!(e));
                    }
                }
            }
            by_t.insert(t.to_string(), Value::Object(m));
        }
        r["ok"] = json!(true);
        r["batches"] = Value::Object(bmap);
        r["cpu_threads"] = Value::Object(by_t);
        return Ok(r);
    }
    // a GPU provider: the CPU's answers on the same graph and rows are the expected ones
    let mut cpu = variant_session(rt, &model, ProviderKind::Cpu, None, adapters, threads, &[], None)?.0;
    let mut gpu_dyn: Option<Session> = None;
    let mut bmap = Map::new();
    let mut all_ok = true;
    for (n, b) in &batches {
        let res = (|| -> Result<Value, String> {
            let want_out = run_net_once(&mut cpu, &net_feed(b, None)?)?;
            let tb = Instant::now();
            let mut s_static = None;
            if stat {
                let (s, placed, dev) = variant_session(rt, &model, p, want, adapters, threads, &dims_for(b), opt_of(v)?)?;
                if r.get("cpu_nodes").is_none() {
                    placement_json(&mut r, &placed, p);
                    r["gpu"] = json!(dev.map(|d| d.to_json()));
                }
                s_static = Some(s);
            } else if gpu_dyn.is_none() {
                let (s, placed, dev) = variant_session(rt, &model, p, want, adapters, threads, &[], opt_of(v)?)?;
                placement_json(&mut r, &placed, p);
                r["gpu"] = json!(dev.map(|d| d.to_json()));
                r["build_s"] = json!(r4(tb.elapsed().as_secs_f64()));
                gpu_dyn = Some(s);
            }
            let build_s = tb.elapsed().as_secs_f64();
            let s = s_static.as_mut().or(gpu_dyn.as_mut()).unwrap();
            let f = net_feed(b, stat.then(|| cap_of(b)))?;
            let t1 = Instant::now();
            let a = run_net_once(s, &f)?;
            let first_ms = t1.elapsed().as_secs_f64() * 1e3;
            let a2 = run_net_once(s, &f)?;
            let det = a.iter().zip(&a2).all(|(x, y)| x.1.len() == y.1.len() && x.1.iter().zip(&y.1).all(|(p, q)| p.to_bits() == q.to_bits()));
            // the real rows only (a static batch's dummy row is the last)
            let mut mab = Map::new();
            let mut worst = 0f64;
            for (k, ((gs, g), (ws, w))) in a.iter().zip(&want_out).enumerate() {
                let width = ws.get(1).copied().unwrap_or(1);
                if gs.get(1).copied().unwrap_or(1) != width || gs[0] < ws[0] {
                    return Err(format!("output {}: shape {gs:?}, want {ws:?}", OUTS[k]));
                }
                let n_real = ws[0] * width;
                let m = max_abs(&g[..n_real].iter().map(|&x| x as f64).collect::<Vec<_>>(), &w[..n_real].iter().map(|&x| x as f64).collect::<Vec<_>>());
                worst = worst.max(m);
                mab.insert(OUTS[k].into(), json!(m));
            }
            let (per_s, med) = timed(|| run_net_once(s, &f).map(|_| ()))?;
            let mut x = json!({"max_abs": worst, "max_abs_by_output": mab, "deterministic": det, "rows_per_s": (per_s * *n as f64).round(), "ms": r4(med), "first_ms": r4(first_ms)});
            if stat {
                x["build_s"] = json!(r4(build_s));
            }
            Ok(x)
        })();
        let x = res.unwrap_or_else(|e| json!({"why": e}));
        let failed = x.get("why").is_some();
        all_ok &= x["max_abs"].as_f64().is_some_and(|m| m <= tol) && x["deterministic"] == json!(true);
        bmap.insert(n.to_string(), x);
        // (`stop_on_fail`: no further sessions of a provider that already failed - one failing session at most on a PC
        // where failures are risky)
        if failed && v["stop_on_fail"] == json!(true) {
            break;
        }
    }
    r["ok"] = json!(all_ok);
    if !all_ok {
        let firstbad = bmap.iter().find(|(_, x)| x.get("why").is_some()).map(|(n, x)| format!("batch {n}: {}", x["why"].as_str().unwrap_or("")));
        if let Some(w) = firstbad {
            r["why"] = json!(w.chars().take(400).collect::<String>());
        }
    }
    r["batches"] = Value::Object(bmap);
    Ok(r)
}

/// Real networks in the agent's cache (`<root>\cache\ckpt\<run>\<gen>\model.onnx` + `reference.npz`; the worker
/// runs from `<root>\cache\builds\<id>\`): the newest gen of each run, at most `max` runs, the newest first.
pub fn cache_nets(max: usize) -> Vec<(String, PathBuf, PathBuf)> {
    let Ok(exe) = std::env::current_exe() else { return Vec::new() };
    let Some(cache) = exe.parent().and_then(Path::parent).and_then(Path::parent) else { return Vec::new() };
    let ck = cache.join("ckpt");
    let mut nets: Vec<(std::time::SystemTime, String, PathBuf, PathBuf)> = Vec::new();
    let Ok(runs) = std::fs::read_dir(&ck) else { return Vec::new() };
    for run in runs.flatten() {
        let mut best: Option<(u64, PathBuf)> = None;
        for g in std::fs::read_dir(run.path()).into_iter().flatten().flatten() {
            let Ok(gen) = g.file_name().to_string_lossy().parse::<u64>() else { continue };
            if g.path().join("model.onnx").is_file() && g.path().join("reference.npz").is_file() && best.as_ref().is_none_or(|b| gen > b.0) {
                best = Some((gen, g.path()));
            }
        }
        if let Some((gen, d)) = best {
            let t = std::fs::metadata(d.join("model.onnx")).and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
            let label: String = format!("{}_{gen}", run.file_name().to_string_lossy()).chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
            nets.push((t, label, d.join("model.onnx"), d.join("reference.npz")));
        }
    }
    nets.sort_by(|a, b| b.0.cmp(&a.0));
    nets.into_iter().take(max).map(|(_, l, m, r)| (l, m, r)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(descr: &str, shape: Vec<usize>, data: Vec<u8>) -> Arr {
        Arr { descr: descr.into(), shape, data }
    }

    #[test]
    fn npy_files_round_trip_scalars_empties_and_bools() {
        let d = std::env::temp_dir().join(format!("zv-npy-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        for a in [
            arr("<f4", vec![], 1.5f32.to_le_bytes().to_vec()),
            arr("<f4", vec![3, 0], Vec::new()),
            arr("|b1", vec![2, 2], vec![1, 0, 0, 1]),
            arr("<i8", vec![2], [7i64, -3].iter().flat_map(|x| x.to_le_bytes()).collect()),
        ] {
            let p = d.join("x.npy");
            write_npy(&p, &a).unwrap();
            let b = read_npy(&p).unwrap();
            assert_eq!((b.descr.as_str(), &b.shape, &b.data), (a.descr.as_str(), &a.shape, &a.data));
            assert_eq!(arr_numbers(&b).len(), a.shape.iter().product::<usize>());
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn static_batches_get_one_dummy_row_with_the_padding() {
        let mut b = zero_search::driver::Batch::default();
        b.b = 2;
        b.xs_off = vec![0, 2, 3];
        b.xs_idx = vec![1, 5, 2];
        b.xs_val = vec![0.5, 1.0, 2.0];
        b.n_players = vec![2, 2];
        b.extra = vec![0, zero_search::driver::ROW_BELIEF];
        let f = net_feed(&b, None).unwrap();
        assert_eq!((f.x_ptr.clone(), f.bel.clone()), (vec![0, 2, 3], vec![false, true]));
        let f = net_feed(&b, Some(1024)).unwrap();
        assert_eq!((f.x_ptr.len(), *f.x_ptr.last().unwrap(), f.x_idx.len(), f.np.len()), (4, 1024, 1024, 3));
        assert!(f.x_idx[3..].iter().all(|&i| i == 0) && f.x_val[3..].iter().all(|&v| v == 0.0));
        assert!(net_feed(&b, Some(3)).is_err(), "a cap must leave the dummy row at least one non-zero");
    }

    #[test]
    fn max_abs_treats_nan_as_a_failure_and_equal_infinities_as_equal() {
        assert_eq!(max_abs(&[1.0, f64::NEG_INFINITY], &[1.5, f64::NEG_INFINITY]), 0.5);
        assert_eq!(max_abs(&[f64::NAN], &[0.0]), f64::INFINITY);
    }

    #[test]
    fn result_lines_read_like_the_probe_report() {
        let r = json!({"variant": "pb1", "provider": "dml", "gpu": {"name": "RX"}, "ok": true, "cpu_nodes": ["A (a)"], "build_s": 1.0,
                       "batches": {"1024": {"rows_per_s": 9000.0, "max_abs": 1e-6, "deterministic": true}, "16": {"rows_per_s": 100.0, "max_abs": 1e-6, "deterministic": true}}});
        let l = line(&r);
        assert!(l.starts_with("  variant pb1 dml on RX: PASS, 1 nodes on the CPU"), "{l}");
        assert!(l.find("16: 100/s").unwrap() < l.find("1024: 9000/s").unwrap(), "{l}");
        let f = line(&json!({"variant": "copy_x", "provider": "amdgpu", "ok": false, "why": "HIP failure"}));
        assert!(f.ends_with("FAIL HIP failure"), "{f}");
    }
}
