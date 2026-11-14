//! `probe-providers` (zero-farm PROTOCOL.md 8): which execution providers work on this PC, with this runtime
//! pack and these GPU drivers. For every candidate provider and every adapter it serves, the session factory's
//! own steps (`session::try_one`: build, self-test twice against the reference set, the nodes left to the CPU),
//! then a short timing; the CPU first as the baseline. The agent runs it in the background and caches the result
//! per (runtime build, driver version, adapter); the providers that passed become the profile's `providers`.
//!
//! The graph is built in (`probe/probe.onnx` + `probe/probe_reference.npz`, `tools/onnx_probe_graph.py`: a
//! small token network that uses every operator a real network's graph uses), or a real network (`--model`,
//! `--reference`: the gate's measurements). Windows ML's catalog providers are installed first when
//! `--ensure-ready` (the agent's probe), never by a job.
//!
//! Output: one JSON object (stdout, or `--json FILE`) and a readable report on stderr - the worker log, which
//! is what a friend's "Send diagnostics" upload carries.
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::{json, Map, Value};

use crate::eval::Evaluator;
use crate::gpu::Adapter;
use crate::session::{runtime, try_one, ProviderKind, SessionSpec, Tried};

pub const GRAPH: &[u8] = include_bytes!("../probe/probe.onnx");
pub const REFERENCE: &[u8] = include_bytes!("../probe/probe_reference.npz");

/// What to probe.
pub struct Opts {
    pub dylib: PathBuf,
    /// A real network and its reference set (else the built-in graph).
    pub model: Option<PathBuf>,
    pub reference: Option<PathBuf>,
    /// Candidate providers (default: all).
    pub providers: Vec<ProviderKind>,
    /// Only this adapter (LUID).
    pub gpu: Option<u64>,
    pub threads: usize,
    /// Let Windows ML install the catalog providers this PC's adapters could use.
    pub ensure_ready: bool,
    /// Timed passes over the reference set after the self-test.
    pub reps: usize,
    /// Batch sizes to time (rows cycled from the reference set; empty: the reference's own batches).
    pub bench_rows: Vec<usize>,
}

/// A batch of `n` rows cycled from the reference set's rows (timing only).
pub fn cycle_batch(r: &crate::selftest::Reference, n: usize) -> zero_search::driver::Batch {
    let mut b = zero_search::driver::Batch::default();
    b.b = n;
    b.xs_off.push(0);
    b.opt_off.push(0);
    let rows: Vec<(&zero_search::driver::Batch, usize)> = r.batches.iter().flat_map(|x| (0..x.b).map(move |i| (x, i))).collect();
    for k in 0..n {
        let (x, i) = rows[k % rows.len()];
        let (lo, hi) = (x.xs_off[i] as usize, x.xs_off[i + 1] as usize);
        b.xs_idx.extend_from_slice(&x.xs_idx[lo..hi]);
        b.xs_val.extend_from_slice(&x.xs_val[lo..hi]);
        b.xs_off.push(b.xs_idx.len() as i64);
        let (lo, hi) = (x.opt_off[i] as usize, x.opt_off[i + 1] as usize);
        b.opt_action.extend_from_slice(&x.opt_action[lo..hi]);
        b.opt_off.push(b.opt_action.len() as i32);
        b.n_players.push(x.n_players[i]);
        b.extra.push(x.extra[i]);
        b.player.push(0);
        b.gid.push(0);
    }
    b
}

/// `reps` evaluations of a batch of each size -> {rows: rows per second}.
fn bench_sizes(s: &mut crate::session::InferenceSession, r: &crate::selftest::Reference, sizes: &[usize], reps: usize) -> Result<Map<String, Value>, String> {
    let mut out = Map::new();
    for &n in sizes {
        let b = cycle_batch(r, n);
        s.evaluate(&b)?; // (warm-up: a GPU provider's first run of a shape)
        let t0 = Instant::now();
        for _ in 0..reps.max(1) {
            s.evaluate(&b)?;
        }
        let secs = t0.elapsed().as_secs_f64();
        out.insert(n.to_string(), json!(((n * reps.max(1)) as f64 / secs).round()));
    }
    Ok(out)
}

/// A provider the factory skipped (not usable here, or refused): an ordinary result entry, `ok` false with the `why`.
pub fn skipped_entry(r: &mut Value, at: &crate::session::Attempt) {
    r["ok"] = json!(false);
    r["why"] = json!(at.why);
}

fn r4(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}

/// Every reference batch `reps` times -> (rows per second, ms per batch median).
fn bench(s: &mut crate::session::InferenceSession, r: &crate::selftest::Reference, reps: usize) -> Result<(f64, f64), String> {
    let mut ms = Vec::new();
    let t0 = Instant::now();
    let mut rows = 0usize;
    for _ in 0..reps {
        for b in &r.batches {
            let t = Instant::now();
            s.evaluate(b)?;
            ms.push(t.elapsed().as_secs_f64() * 1e3);
            rows += b.b;
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    ms.sort_by(|a, b| a.total_cmp(b));
    let med = ms.get(ms.len() / 2).copied().unwrap_or(0.0);
    Ok((if secs > 0.0 { rows as f64 / secs } else { 0.0 }, med))
}

/// The op types of a node list ("OpType (name)") with their counts.
fn op_counts(nodes: &[String]) -> Map<String, Value> {
    let mut m = std::collections::BTreeMap::<String, u64>::new();
    for n in nodes {
        *m.entry(n.split(" (").next().unwrap_or(n).to_string()).or_default() += 1;
    }
    m.into_iter().map(|(k, v)| (k, json!(v))).collect()
}

pub fn run(o: &Opts) -> Result<Value, String> {
    let t0 = Instant::now();
    let rt = runtime(&o.dylib)?;
    let adapters: Vec<Adapter> = crate::gpu::adapters();
    // the catalog providers this PC's adapters could use
    let wanted: Vec<ProviderKind> = crate::session::CATALOG.iter().copied().filter(|p| o.providers.contains(p) && adapters.iter().any(|a| p.serves(a))).collect();
    let catalog = if crate::winml::present(&rt.dir) {
        let c = rt.catalog(&wanted, o.ensure_ready);
        json!({"present": true, "ensure_ready": o.ensure_ready, "eps": c.eps.iter().map(|e| e.to_json()).collect::<Vec<_>>(),
               "registered": c.registered.iter().map(|r| json!([r.0, r.1.wire()])).collect::<Vec<_>>(), "errors": c.errors})
    } else {
        json!({"present": false})
    };
    // (the built-in graph's files go under the working directory - the agent's work/probe - else the temp dir)
    let base = std::env::current_dir().ok().filter(|d| std::fs::metadata(d).is_ok_and(|m| !m.permissions().readonly())).unwrap_or_else(std::env::temp_dir);
    let tmp = base.join(format!("zero-probe-{}", std::process::id()));
    let (model, reference, graph) = match (&o.model, &o.reference) {
        (Some(m), Some(r)) => (m.clone(), r.clone(), m.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default()),
        (Some(_), None) => return Err("--model needs its --reference (the self-test)".into()),
        _ => {
            std::fs::create_dir_all(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
            let (m, r) = (tmp.join("probe.onnx"), tmp.join("probe_reference.npz"));
            std::fs::write(&m, GRAPH).map_err(|e| format!("{}: {e}", m.display()))?;
            std::fs::write(&r, REFERENCE).map_err(|e| format!("{}: {e}", r.display()))?;
            (m, r, "builtin".to_string())
        }
    };
    let set = crate::selftest::load(&reference)?;
    let spec = SessionSpec { model: &model, reference: Some(&reference), providers: &[], device: None, intra_threads: o.threads, runtime: &o.dylib, spinning: false, expect: None };
    let mut results = Vec::new();
    let mut passed: Vec<ProviderKind> = Vec::new();
    let mut cpu_rate = None;
    let mut fatal = None;
    // the CPU first (the baseline), then the GPU providers in preference order
    let mut order: Vec<ProviderKind> = vec![ProviderKind::Cpu];
    order.extend(ProviderKind::ALL.iter().copied().filter(|p| p.is_gpu() && o.providers.contains(p)));
    for p in order {
        let targets: Vec<Option<&Adapter>> = if p.is_gpu() {
            let t: Vec<Option<&Adapter>> = adapters.iter().filter(|a| p.serves(a) && o.gpu.is_none_or(|l| l == a.luid)).map(Some).collect();
            if t.is_empty() {
                results.push(json!({"provider": p.wire(), "ok": false, "why": format!("no adapter here that {} serves", p.wire())}));
                continue;
            }
            t
        } else {
            vec![None]
        };
        for a in targets {
            let dev = a.map(crate::session::GpuDevice::of);
            let mut r = json!({"provider": p.wire(), "gpu": dev.as_ref().map(|d| d.to_json())});
            match try_one(rt, &spec, Some(&set), p, dev.as_ref(), &adapters) {
                Tried::Ok(mut s) => {
                    let st = s.selftest().clone();
                    r["ok"] = json!(st.ok);
                    r["selftest"] = st.to_json();
                    r["cpu_fallback_ops"] = Value::Object(op_counts(&st.cpu_fallback_nodes));
                    r["placed"] = Value::Object(s.placements.iter().fold(Map::new(), |mut m, (prov, _)| {
                        let n = m.get(prov).and_then(Value::as_u64).unwrap_or(0);
                        m.insert(prov.clone(), json!(n + 1));
                        m
                    }));
                    if st.ok && o.reps > 0 && !o.bench_rows.is_empty() {
                        match bench_sizes(&mut s, &set, &o.bench_rows, o.reps) {
                            Ok(m) => r["rows_per_s_by_batch"] = Value::Object(m),
                            Err(e) => r["bench_error"] = json!(e),
                        }
                    }
                    r["chunk"] = json!(s.chunk);
                    if st.ok && o.reps > 0 {
                        match bench(&mut s, &set, o.reps) {
                            Ok((rate, med)) => {
                                r["rows_per_s"] = json!(rate.round());
                                r["batch_ms_median"] = json!(r4(med));
                                if p == ProviderKind::Cpu {
                                    cpu_rate = Some(rate);
                                } else if let Some(c) = cpu_rate.filter(|c| *c > 0.0) {
                                    r["vs_cpu"] = json!((rate / c * 100.0).round() / 100.0);
                                }
                            }
                            Err(e) => r["bench_error"] = json!(e),
                        }
                    }
                    if st.ok && !passed.contains(&p) {
                        passed.push(p);
                    }
                    s.release_device();
                }
                Tried::Skip(at) => skipped_entry(&mut r, &at),
                Tried::Fatal(e) => {
                    r["ok"] = json!(false);
                    r["why"] = json!(e.clone());
                    if p == ProviderKind::Cpu {
                        fatal = Some(e);
                    }
                }
            }
            results.push(r);
        }
        if fatal.is_some() {
            break;
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
    // the passed providers in preference order, the CPU last
    let providers: Vec<&str> = ProviderKind::ALL.iter().filter(|p| passed.contains(p)).map(|p| p.wire()).collect();
    // where each library was loaded from: the runtime pack, or elsewhere (System32's DirectML, a CUDA install)
    let mods: Map<String, Value> = ["onnxruntime.dll"]
        .iter()
        .chain(crate::session::PRELOAD)
        .chain([crate::winml::DLL].iter())
        .map(|n| (n.to_string(), json!(crate::eval::module_path(n).map(|p| if Path::new(&p).parent() == Some(rt.dir.as_path()) { "pack" } else { "elsewhere" }))))
        .collect();
    Ok(json!({
        "probe": 1,
        "build_id": crate::proto::build_id(),
        "ort": crate::session::ort_version(),
        "available": rt.available,
        "modules": mods,
        "adapters": adapters.iter().map(Adapter::to_json).collect::<Vec<_>>(),
        "catalog": catalog,
        "ep_devices": rt.ep_devices().iter().map(|d| d.to_json()).collect::<Vec<_>>(),
        "graph": graph,
        "rows": set.rows,
        "threads": o.threads,
        "results": results,
        "providers": providers,
        "error": fatal,
        "seconds": r4(t0.elapsed().as_secs_f64()),
    }))
}

/// The report for a person (the worker log): one line per provider and adapter.
pub fn report(v: &Value) -> String {
    let mut out = String::new();
    out.push_str(&format!("probe-providers: ONNX Runtime {} ({} compiled in: {}), graph {}, {} rows\n", v["ort"].as_str().unwrap_or("?"),
        v["available"].as_array().map(|a| a.len()).unwrap_or(0), v["available"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(", ")).unwrap_or_default(),
        v["graph"].as_str().unwrap_or("?"), v["rows"]));
    for a in v["adapters"].as_array().into_iter().flatten() {
        out.push_str(&format!("  adapter {} {} ({}, {} MB VRAM)\n", a["luid"].as_str().unwrap_or("?"), a["name"].as_str().unwrap_or("?"), a["vendor"].as_str().unwrap_or("?"), a["vram_mb"]));
    }
    let c = &v["catalog"];
    if c["present"] == json!(true) {
        out.push_str(&format!("  Windows ML catalog: {} provider(s){}\n", c["eps"].as_array().map(|a| a.len()).unwrap_or(0),
            c["errors"].as_array().filter(|e| !e.is_empty()).map(|e| format!("; errors: {}", e.iter().filter_map(Value::as_str).collect::<Vec<_>>().join("; "))).unwrap_or_default()));
        for e in c["eps"].as_array().into_iter().flatten() {
            out.push_str(&format!("    {} {} ready={} certified={} ensure={}\n", e["name"].as_str().unwrap_or("?"), e["version"].as_str().unwrap_or("?"), e["ready"].as_str().unwrap_or("?"), e["certified"], e["ensure"]));
        }
    } else {
        out.push_str("  Windows ML catalog: not in this runtime pack\n");
    }
    for r in v["results"].as_array().into_iter().flatten() {
        let gpu = r["gpu"]["name"].as_str().map(|n| format!(" on {n}")).unwrap_or_default();
        if r["ok"] == json!(true) {
            let st = &r["selftest"];
            out.push_str(&format!("  PASS {}{}: max abs {} deterministic {}, {} nodes on the CPU {}, build {} s, {} rows/s{}\n", r["provider"].as_str().unwrap_or("?"), gpu,
                st["max_abs"], st["deterministic"], st["cpu_fallback_nodes"].as_array().map(|a| a.len()).unwrap_or(0), r["cpu_fallback_ops"], st["build_s"], r["rows_per_s"],
                r.get("vs_cpu").map(|x| format!(" ({x}x the CPU)")).unwrap_or_default()));
            if let Some(m) = r["rows_per_s_by_batch"].as_object() {
                out.push_str(&format!("       rows/s by batch size: {}\n", m.iter().map(|(k, v)| format!("{k}: {v}")).collect::<Vec<_>>().join(", ")));
            }
        } else {
            let why = r["why"].as_str().map(String::from).unwrap_or_else(|| {
                let st = &r["selftest"];
                format!("self-test max abs {} deterministic {}", st["max_abs"], st["deterministic"])
            });
            out.push_str(&format!("  FAIL {}{}: {}\n", r["provider"].as_str().unwrap_or("?"), gpu, why));
        }
    }
    out.push_str(&format!("  providers: {} ({} s)\n", v["providers"], v["seconds"]));
    out
}
