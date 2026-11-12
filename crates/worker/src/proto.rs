//! The worker's command line and the agent<->worker protocol `aw/1` (zero-farm PROTOCOL.md 5).
//!
//! ```text
//! <worker> serve                         aw/1: JSON lines in (op) / out (ev); stdout JSON only, stderr the log
//! <worker> selfplay --settings F (--model M | --uniform) --games N [--gid-base G] [--seed S] [--net N]
//!          [--threads T] [--infer-threads I] [--conc C] [--groups G] [--out DIR] [--prefix P]
//!          [--ort-dll PATH] [--spinning 0|1] [--record-batches N --record-file F] [--sha256 H (the model file's)]
//!          [--status-s X] [--stdin-control] [--reference R] [--output shards|records|both (default shards)]
//!                                                  one job from flags (the same runner; events on stdout)
//! <worker> eval --settings F (--model GEN=M ... | --uniform) --seats JSON --games N [--gid-base G] [--seed S]
//!          [--paired] [--threads T] [--infer-threads I] [--conc C] [--groups G] [--out DIR] [--prefix P]
//!          [--reference GEN=R ...] [--sha256 GEN=H ...] [--ort-dll PATH] [--spinning 0|1] [--flush-games N]
//!          [--record-batches N --record-file F] [--status-s X] [--stdin-control]
//!                                                  one eval job from flags (events on stdout)
//! <worker> check-config <config.json> [--json] [--eval [--seats JSON]]
//!                                                  classify every setting (for self-play, or for an eval
//!                                                  job); exit 1 on unknown / unsupported
//! <worker> driver-cfg <config.json> --seed S --net N   the Python remote worker's DriverCfg (Debug text)
//! <worker> driver-cfg <config.json> --eval --seats JSON --games N --seed S [--paired] [--lease]
//!                                                  Run.play_match's DriverCfg (--lease: the eval job's)
//! <worker> eval-batches --model M --batches F --out F [--threads T] [--ort-dll P]
//!                                                  answers to recorded batches (the evaluator's gate)
//! <worker> info [--runtime-dir D | --ort-dll P]     games, protocol, ONNX Runtime (and where its DLLs came from)
//! <worker> fuzz --mode M [--env K=V ...] --first F --count N [--threads T] [--chunk C]
//!          [--seed-timeout S] [--out DIR] [--work DIR] [--keep-chunks] [--stdin-control]
//!                                                  a fuzz job from flags (events on stdout)
//! <worker> fuzz-check --mode M [--env K=V ...] [--seed S] [--commit C]
//!                                                  what a campaign's mode and env mean, and a repro
//! <worker> fuzz-chunk <spec.json>                  one chunk of a fuzz job (the supervisor's child)
//! ```
//!
//! `serve`: `hello` first (aw, build_id, game, kinds, zero_onnx, pid), then one `job` line (`op` job:
//! lease_id, kind 'selfplay', settings, games, gid_base, seed, model, reference (the self-test's set:
//! answered before the first batch, `selftest` {provider, max_abs, ok}; a failure is fatal), threads, out_dir;
//! optional gen - the rows' network id, default the graph's `checkpoint_iter` -, sha256 - the model FILE's,
//! checked before the file is loaded -, conc, groups, infer_threads, spinning, prefix, output - 'records' (the
//! default from zero-farm phase 3: game records, `records` events; zero-core docs/search.md 2.15), 'shards'
//! (sample shards, `shard` events) or 'both'); during it `yield` ->
//! `yielded` (no batch started, the session dropped), `resume` -> `resumed`, `finish`, `quit`, `checkpoint`
//! {gen, model, sha256? (the file's), reference?},
//! `ping` -> `status`; events `fin`, `records` / `shard` (seq, path under out_dir, games, rows, bytes, gids:
//! half-open [a, b) ranges, gen: the oldest network of its rows), `status` every <= 10 s (PROTOCOL.md 3 Stats fields), `end` (reason, stats,
//! unfinished {gids: [[a, b)]}), `error` (message, fatal). The process exits after `end` (or a fatal
//! error, status 1).
//!
//! An eval job (PROTOCOL.md 5.2, `evaljob`): kind 'eval', settings, seats ([{"net": gen} | {"bot": name}] x 2),
//! paired, games, gid_base, seed, models ({"<gen>": {"model", "sha256"?, "reference"?}}: every one loaded and
//! self-tested - `selftest` with its `gen` -, the file's sha256 checked), threads, out_dir, runtime_dir and
//! the tuning fields; its settings must pass `check-config --eval` (else a fatal error); `evals` events (as
//! `shard`'s, JSON-lines result files) instead of `shard`; `checkpoint` is refused (non-fatal).
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};

use crate::eval::{ort_env, ort_version, OrtCfg, OrtEvaluator};
use crate::job::{do_resume, do_yield, run_job, Checkpoint, Control, EvalKind, EvalLink, Job, Outcome, Progress, Sink};
use crate::fuzz::{FuzzControl, FuzzShared};
use crate::settings::Settings;
use crate::{classify, GameEntry, Worker, AW, ZERO_ONNX};

/// JSON lines on stdout (one writer at a time), the log on stderr.
pub struct Stdout {
    mu: Mutex<()>,
}

impl Sink for Stdout {
    fn event(&self, v: Value) {
        let _g = self.mu.lock().unwrap();
        let mut o = std::io::stdout().lock();
        let _ = writeln!(o, "{v}");
        let _ = o.flush();
    }
    fn log(&self, msg: &str) {
        eprintln!("[zero-worker {}] {msg}", std::process::id());
    }
}

struct Args<'a> {
    v: &'a [String],
}

impl Args<'_> {
    fn get(&self, k: &str) -> Option<&str> {
        self.v.iter().position(|a| a == k).and_then(|i| self.v.get(i + 1)).map(|s| s.as_str())
    }
    fn has(&self, k: &str) -> bool {
        self.v.iter().any(|a| a == k)
    }
    fn num<T: std::str::FromStr>(&self, k: &str) -> Result<Option<T>, String> {
        self.get(k).map(|s| s.parse::<T>().map_err(|_| format!("{k} {s:?}: a number"))).transpose()
    }
    fn all(&self, k: &str) -> Vec<&str> {
        self.v.windows(2).filter(|w| w[0] == k).map(|w| w[1].as_str()).collect()
    }
    fn positional(&self) -> Option<&str> {
        self.v.first().filter(|a| !a.starts_with("--")).map(|s| s.as_str())
    }
}

/// The ONNX Runtime library: `--ort-dll` / the job's `ort_dylib`, else the job's `runtime_dir` (a verified
/// runtime pack, PROTOCOL.md 5) or env ZEROFARM_RUNTIME_DIR holding it, else env ZERO_ORT_DYLIB /
/// ORT_DYLIB_PATH, else `onnxruntime.dll` (`libonnxruntime.so` / `.dylib`) next to the executable.
pub fn ort_dylib(given: Option<&str>) -> PathBuf {
    ort_dylib_in(given, None)
}

pub fn ort_dylib_in(given: Option<&str>, runtime_dir: Option<&str>) -> PathBuf {
    if let Some(g) = given {
        return PathBuf::from(g);
    }
    let name = if cfg!(windows) {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    };
    let env_rt = std::env::var("ZEROFARM_RUNTIME_DIR").ok().filter(|v| !v.is_empty());
    if let Some(d) = runtime_dir.map(String::from).or(env_rt) {
        return PathBuf::from(d).join(name);
    }
    for k in ["ZERO_ORT_DYLIB", "ORT_DYLIB_PATH"] {
        if let Ok(v) = std::env::var(k) {
            if !v.is_empty() {
                return PathBuf::from(v);
            }
        }
    }
    std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join(name))).unwrap_or_else(|| PathBuf::from(name))
}

/// The build id: the first 24 hex digits of sha256(manifest.json) next to the executable (PROTOCOL.md
/// 2), else env ZERO_FARM_BUILD_ID, else 'unsigned-dev'.
pub fn build_id() -> String {
    if let Some(p) = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join("manifest.json"))) {
        if let Ok(b) = std::fs::read(&p) {
            return sha256_hex(&b)[..24].to_string();
        }
    }
    std::env::var("ZERO_FARM_BUILD_ID").unwrap_or_else(|_| "unsigned-dev".into())
}

/// SHA-256 (FIPS 180-4), hex.
pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut m = data.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    m.push(0x80);
    while m.len() % 64 != 56 {
        m.push(0);
    }
    m.extend_from_slice(&bits.to_be_bytes());
    for c in m.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([c[4 * i], c[4 * i + 1], c[4 * i + 2], c[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(v[i]);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

fn game_of<'a>(games: &'a [GameEntry], s: &Settings) -> Result<&'a GameEntry, String> {
    let name = match s.get("game") {
        Some(crate::pyjson::PyJson::Str(g)) => g.clone(),
        other => return Err(format!("settings: game {:?}", other.map(|g| g.dumps()))),
    };
    games.iter().find(|g| g.name == name).ok_or_else(|| format!("game {name:?}: this worker plays {:?}", games.iter().map(|g| g.name).collect::<Vec<_>>()))
}

fn read_settings(path: &str) -> Result<Settings, String> {
    Settings::parse(&std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?).map_err(|e| format!("{path}: {e}"))
}

/// The rows' network id from a graph's metadata (`checkpoint_iter`), read without a session.
fn checkpoint_iter(model: &Path, ort: &OrtCfg) -> Result<u32, String> {
    let env = ort_env(&ort.dylib)?;
    let s = ort::session::Session::builder(env).map_err(|e| e.to_string())?.commit_from_file(model).map_err(|e| format!("{}: {e}", model.display()))?;
    let it = s.metadata().map_err(|e| e.to_string())?.custom("checkpoint_iter");
    it.and_then(|v| v.parse::<u32>().ok()).ok_or_else(|| format!("{}: no checkpoint_iter in its metadata: give the job's gen", model.display()))
}

/// `checkpoint_iter` of a model whose file's sha256 (when the job gives one) is checked first: ONNX Runtime
/// reads no byte of a file the job's hash doesn't name.
fn checkpoint_iter_checked(model: &Path, ort: &OrtCfg, sha256: Option<&str>) -> Result<u32, String> {
    if let Some(h) = sha256 {
        crate::job::check_file_sha256(model, h, None)?;
    }
    checkpoint_iter(model, ort)
}

fn job_from_value(v: &Value, base: &Path) -> Result<Job, String> {
    let o = v.as_object().ok_or("job: a JSON object")?;
    let kind = o.get("kind").and_then(|k| k.as_str()).unwrap_or("");
    if kind != "selfplay" {
        return Err(format!("job kind {kind:?}: this worker build runs 'selfplay' jobs (fuzz, eval, diag: not built yet)"));
    }
    // (the settings as the line wrote them - key order and int / float kept: a dict spec reaches the game
    // as Python's json.dumps of it)
    let settings = match o.get("_raw").and_then(|r| r.as_str()) {
        Some(raw) => match crate::pyjson::PyJson::parse(raw)?.get("settings") {
            Some(s @ crate::pyjson::PyJson::Dict(_)) => Settings { raw: s.clone() },
            _ => return Err("job: settings must be an object".into()),
        },
        None => Settings::from_value(o.get("settings").ok_or("job: no settings")?)?,
    };
    let u = |k: &str| o.get(k).and_then(|x| x.as_u64());
    let games = u("games").ok_or("job: games")?;
    let threads = u("threads").unwrap_or(1).max(1) as usize;
    let out_dir = base.join(o.get("out_dir").and_then(|x| x.as_str()).unwrap_or("out"));
    // ONNX Runtime comes from the job's runtime pack (`runtime_dir`, or ZEROFARM_RUNTIME_DIR the agent
    // sets), never from anything else on the PC (a Python environment, PATH, the build's own directory)
    let rt = o.get("runtime_dir").and_then(|x| x.as_str()).map(String::from).or_else(|| std::env::var("ZEROFARM_RUNTIME_DIR").ok().filter(|v| !v.is_empty()));
    let dylib = match (&rt, o.get("model").and_then(|m| m.as_str())) {
        (Some(d), _) => ort_dylib_in(None, Some(d)),
        (None, Some(_)) => return Err("job: a network job needs its runtime pack (runtime_dir or ZEROFARM_RUNTIME_DIR)".into()),
        (None, None) => PathBuf::new(),
    };
    let ort = OrtCfg { dylib, intra_threads: u("infer_threads").map_or(threads, |t| t as usize), spinning: o.get("spinning").and_then(|x| x.as_bool()).unwrap_or(false) };
    let sha256 = o.get("sha256").and_then(|x| x.as_str()).map(String::from);
    let reference = o.get("reference").and_then(|x| x.as_str()).map(|r| base.join(r));
    let eval = match o.get("model").and_then(|m| m.as_str()) {
        Some(m) => EvalKind::Onnx { model: base.join(m), ort, sha256, reference },
        None if o.get("uniform").and_then(|x| x.as_bool()) == Some(true) => EvalKind::Uniform,
        None => return Err("job: no model (a self-play job needs its network)".into()),
    };
    let net = match (u("gen"), &eval) {
        (Some(g), _) => g as u32,
        (None, EvalKind::Onnx { model, ort, sha256, .. }) => checkpoint_iter_checked(model, ort, sha256.as_deref())?,
        (None, EvalKind::Uniform) => 0,
    };
    let lease = o.get("lease_id").and_then(|x| x.as_str()).unwrap_or("job").to_string();
    let mut meta = Map::new();
    meta.insert("iter".into(), json!(net + 1));
    meta.insert("source".into(), json!("farm"));
    meta.insert("lease".into(), json!(lease));
    meta.insert("build_id".into(), json!(build_id()));
    Ok(Job {
        prefix: o.get("prefix").and_then(|x| x.as_str()).unwrap_or("f").to_string(),
        lease_id: lease,
        settings,
        eval,
        games,
        gid_base: u("gid_base").unwrap_or(0) as u32,
        seed: u("seed").unwrap_or(0),
        net,
        search_threads: Some(threads),
        conc: u("conc").map(|x| x as usize),
        groups: u("groups").map(|x| x as usize),
        out_dir,
        meta,
        record_batches: None,
        status_s: 10.0,
        // (zero-farm phase 3: a served self-play job writes game records unless its line says otherwise)
        output: crate::records::Output::parse(o.get("output").and_then(|x| x.as_str()).unwrap_or("records"))?,
    })
}

/// `GEN=VALUE` pairs of a flag (`--model 1=a.onnx`).
fn gen_pairs(a: &Args, k: &str) -> Result<Vec<(u32, String)>, String> {
    a.all(k).iter().map(|kv| kv.split_once('=').and_then(|(g, v)| g.parse::<u32>().ok().map(|g| (g, v.to_string()))).ok_or_else(|| format!("{k} {kv:?}: GEN=VALUE"))).collect()
}

fn seats_of_value(v: &Value) -> Result<Vec<crate::settings::EvalSeat>, String> {
    v.as_array().ok_or("seats: a list of two seats")?.iter().map(crate::settings::EvalSeat::parse).collect()
}

/// The job line of an eval job (PROTOCOL.md 5.2).
fn eval_job_from_value(v: &Value, base: &Path) -> Result<crate::evaljob::EvalJob, String> {
    use crate::evaljob::{EvalJob, ModelSpec};
    let o = v.as_object().ok_or("job: a JSON object")?;
    let settings = match o.get("_raw").and_then(|r| r.as_str()) {
        Some(raw) => match crate::pyjson::PyJson::parse(raw)?.get("settings") {
            Some(s @ crate::pyjson::PyJson::Dict(_)) => Settings { raw: s.clone() },
            _ => return Err("job: settings must be an object".into()),
        },
        None => Settings::from_value(o.get("settings").ok_or("job: no settings")?)?,
    };
    let u = |k: &str| o.get(k).and_then(|x| x.as_u64());
    let games = u("games").ok_or("job: games")?;
    let threads = u("threads").unwrap_or(1).max(1) as usize;
    let seats = seats_of_value(o.get("seats").ok_or("job: no seats")?)?;
    let mut models = std::collections::BTreeMap::new();
    if let Some(m) = o.get("models") {
        for (g, x) in m.as_object().ok_or("job: models is an object {\"<gen>\": {model, sha256, reference}}")? {
            let gen: u32 = g.parse().map_err(|_| format!("job: models key {g:?}: a generation"))?;
            let model = x.get("model").and_then(|p| p.as_str()).ok_or_else(|| format!("job: models[{g}] has no model"))?;
            models.insert(gen, ModelSpec { model: base.join(model), sha256: x.get("sha256").and_then(|h| h.as_str()).map(String::from), reference: x.get("reference").and_then(|r| r.as_str()).map(|r| base.join(r)) });
        }
    }
    let uniform = o.get("uniform").and_then(|x| x.as_bool()) == Some(true);
    let rt = o.get("runtime_dir").and_then(|x| x.as_str()).map(String::from).or_else(|| std::env::var("ZEROFARM_RUNTIME_DIR").ok().filter(|v| !v.is_empty()));
    let ort = if uniform || models.is_empty() {
        None
    } else {
        let d = rt.ok_or("job: a network job needs its runtime pack (runtime_dir or ZEROFARM_RUNTIME_DIR)")?;
        Some(OrtCfg { dylib: ort_dylib_in(None, Some(&d)), intra_threads: u("infer_threads").map_or(threads, |t| t as usize), spinning: o.get("spinning").and_then(|x| x.as_bool()).unwrap_or(false) })
    };
    Ok(EvalJob {
        lease_id: o.get("lease_id").and_then(|x| x.as_str()).unwrap_or("job").to_string(),
        settings,
        seats,
        models,
        ort,
        games,
        gid_base: u("gid_base").unwrap_or(0) as u32,
        seed: u("seed").unwrap_or(0),
        paired: o.get("paired").and_then(|x| x.as_bool()).unwrap_or(false),
        search_threads: Some(threads),
        conc: u("conc").map(|x| x as usize),
        groups: u("groups").map(|x| x as usize),
        out_dir: base.join(o.get("out_dir").and_then(|x| x.as_str()).unwrap_or("out")),
        prefix: o.get("prefix").and_then(|x| x.as_str()).unwrap_or("e").to_string(),
        flush_games: u("flush_games").unwrap_or(16).max(1) as usize,
        flush_s: 120.0,
        record_batches: None,
        status_s: 10.0,
    })
}

/// An eval job's settings as check-config --eval checks them: every setting classified for an eval job, the
/// evaluation spec set, the match's driver built (its seats: `Run.nn_seat`'s checks) -> the failures.
fn eval_settings_failures(w: &Worker, game: &GameEntry, s: &Settings, seats: &[crate::settings::EvalSeat]) -> Result<Vec<String>, String> {
    let info = (game.info)();
    let others: Vec<_> = w.games.iter().filter(|g| g.name != game.name).map(|g| (g.info)()).collect();
    let v = classify::classify_eval(s, &info, &others, w.package_settings)?;
    let mut bad: Vec<String> = classify::failures(&v).iter().map(|x| format!("{} = {}: {} ({})", x.key, x.value, x.status, x.why)).collect();
    if let Err(e) = crate::settings::eval_driver_cfg(s, &info, seats, 2, 0, true, None, None, None) {
        bad.push(format!("the evaluation match: {e}"));
    }
    Ok(bad)
}

fn end_event(o: &Outcome) -> Value {
    json!({"ev": "end", "reason": o.reason, "stats": o.stats, "unfinished": {"gids": o.unfinished.iter().map(|r| [r.0, r.1]).collect::<Vec<_>>()}})
}

/// Control lines during a job (yield / resume / finish / quit / checkpoint / ping).
fn control(line: &Value, ctl: &Control, link: &Mutex<Option<EvalLink>>, progress: &Mutex<Progress>, sink: &dyn Sink, base: &Path) {
    let op = line.get("op").and_then(|x| x.as_str()).unwrap_or("");
    match op {
        "yield" => {
            let l = link.lock().unwrap().clone();
            match do_yield(ctl, l.as_ref(), progress) {
                Some(dt) => sink.log(&format!("yielded in {:.0} ms", dt * 1e3)),
                None => sink.log("yield: already yielded"),
            }
            sink.event(json!({"ev": "yielded"}));
        }
        "resume" => {
            let l = link.lock().unwrap().clone();
            match do_resume(ctl, l.as_ref()) {
                Ok(_) => sink.event(json!({"ev": "resumed"})),
                Err(e) => sink.event(json!({"ev": "error", "message": format!("resume: {e}"), "fatal": false})),
            }
        }
        "finish" => ctl.finish(),
        "quit" => ctl.quit(),
        "checkpoint" => {
            let gen = line.get("gen").and_then(|x| x.as_u64());
            let model = line.get("model").and_then(|x| x.as_str());
            match (gen, model) {
                (Some(g), Some(m)) => ctl.checkpoint(Checkpoint { gen: g as u32, model: base.join(m), sha256: line.get("sha256").and_then(|x| x.as_str()).map(String::from), reference: line.get("reference").and_then(|x| x.as_str()).map(|r| base.join(r)) }),
                _ => sink.event(json!({"ev": "error", "message": "checkpoint: gen and model are required", "fatal": false})),
            }
        }
        "ping" => {
            let s = progress.lock().unwrap().stats();
            sink.event(json!({"ev": "status", "stats": s}));
        }
        "job" => sink.event(json!({"ev": "error", "message": "a second job line: one job per worker process", "fatal": false})),
        other => sink.event(json!({"ev": "error", "message": format!("unknown op {other:?}"), "fatal": false})),
    }
}

/// Run `job` on a thread while this thread reads control lines from `lines` (None: no control) and
/// emits a status line every `status_s` -> the outcome.
fn run_with_control(game: &'static GameEntry, mut job: Job, sink: Arc<Stdout>, lines: Option<std::sync::mpsc::Receiver<Option<Value>>>, base: PathBuf) -> Result<Outcome, String> {
    // (the status lines come from the control thread, which also runs while the job waits on its search threads)
    let status_s = job.status_s;
    job.status_s = f64::INFINITY;
    with_control(status_s, sink, lines, base, move |ctl, sink, progress, link_out| run_job(game, &job, ctl, sink, progress, link_out))
}

/// An eval job on a thread while this thread reads control lines (as `run_with_control`).
fn run_eval_with_control(game: &'static GameEntry, mut job: crate::evaljob::EvalJob, sink: Arc<Stdout>, lines: Option<std::sync::mpsc::Receiver<Option<Value>>>, base: PathBuf) -> Result<Outcome, String> {
    let status_s = job.status_s;
    job.status_s = f64::INFINITY;
    with_control(status_s, sink, lines, base, move |ctl, sink, progress, link_out| crate::evaljob::run_eval_job(game, &job, ctl, sink, progress, link_out))
}

type JobFn = dyn FnOnce(&Control, &dyn Sink, Arc<Mutex<Progress>>, &dyn Fn(EvalLink)) -> Result<Outcome, String> + Send;

/// Run a network job (`run`) on a thread while this thread reads control lines from `lines` (None: no
/// control) and emits a status line every `status_s` -> the outcome.
fn with_control(status_s: f64, sink: Arc<Stdout>, lines: Option<std::sync::mpsc::Receiver<Option<Value>>>, base: PathBuf, run: impl FnOnce(&Control, &dyn Sink, Arc<Mutex<Progress>>, &dyn Fn(EvalLink)) -> Result<Outcome, String> + Send + 'static) -> Result<Outcome, String> {
    let run: Box<JobFn> = Box::new(run);
    let ctl = Arc::new(Control::default());
    let link: Arc<Mutex<Option<EvalLink>>> = Arc::new(Mutex::new(None));
    let progress = Arc::new(Mutex::new(Progress::default()));
    let (c2, l2, p2, s2) = (ctl.clone(), link.clone(), progress.clone(), sink.clone());
    let worker = std::thread::Builder::new()
        .name("zero-worker-job".into())
        .spawn(move || {
            let lk = l2.clone();
            let c3 = c2.clone();
            run(&c2, &*s2, p2, &move |l| {
                // (a yield that came while the evaluator was loading: release it now)
                if c3.is_yielded() {
                    l.release();
                }
                *lk.lock().unwrap() = Some(l);
            })
        })
        .map_err(|e| format!("spawn: {e}"))?;
    let mut t_status = Instant::now();
    let mut eof = false;
    while !worker.is_finished() {
        match &lines {
            Some(rx) if !eof => match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(Some(v)) => control(&v, &ctl, &link, &progress, &*sink, &base),
                Ok(None) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    // stdin closed: the agent is gone - stop now
                    eof = true;
                    sink.log("stdin closed: quitting the job");
                    ctl.quit();
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            },
            _ => std::thread::sleep(Duration::from_millis(100)),
        }
        if t_status.elapsed().as_secs_f64() >= status_s {
            t_status = Instant::now();
            let s = progress.lock().unwrap().stats();
            sink.event(json!({"ev": "status", "stats": s}));
        }
    }
    worker.join().map_err(|_| "the job thread panicked".to_string())?
}

fn stdin_lines(sink: Arc<Stdout>) -> std::sync::mpsc::Receiver<Option<Value>> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("zero-worker-stdin".into())
        .spawn(move || {
            for line in std::io::stdin().lock().lines() {
                let Ok(line) = line else { break };
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<Value>(&line) {
                    Ok(mut v) => {
                        if v.get("op").and_then(|o| o.as_str()) == Some("job") {
                            v["_raw"] = Value::String(line.clone());
                        }
                        if tx.send(Some(v)).is_err() {
                            return;
                        }
                    }
                    Err(e) => sink.event(json!({"ev": "error", "message": format!("a line that is not JSON: {e}"), "fatal": false})),
                }
            }
            let _ = tx.send(None);
        })
        .expect("spawn the stdin reader");
    rx
}

fn serve(w: &'static Worker) -> i32 {
    let games = w.games;
    let sink = Arc::new(Stdout { mu: Mutex::new(()) });
    sink.event(json!({"ev": "hello", "aw": AW, "build_id": build_id(), "game": w.repo, "kinds": w.kinds(), "zero_onnx": if w.selfplay { json!(ZERO_ONNX) } else { json!([]) }, "pid": std::process::id(), "games": games.iter().map(|g| g.name).collect::<Vec<_>>()}));
    let rx = stdin_lines(sink.clone());
    let base = std::env::current_dir().unwrap_or_default();
    loop {
        let line = match rx.recv() {
            Ok(Some(v)) => v,
            _ => return 0,
        };
        match line.get("op").and_then(|x| x.as_str()) {
            Some("job") => {
                let kind = line.get("kind").and_then(|k| k.as_str()).unwrap_or("");
                match (kind, w.fuzz) {
                    ("fuzz", Some(fg)) => return serve_fuzz(fg, &line, sink, rx, base),
                    ("eval", _) if w.selfplay => return serve_eval(w, &line, sink, rx, base),
                    ("selfplay", _) if w.selfplay => {}
                    _ => {
                        sink.event(json!({"ev": "error", "message": format!("job kind {kind:?}: this worker build runs {:?}", w.kinds()), "fatal": true}));
                        return 1;
                    }
                }
                let job = match job_from_value(&line, &base).and_then(|j| Ok((game_of(games, &j.settings)?, j))) {
                    Ok(x) => x,
                    Err(e) => {
                        sink.event(json!({"ev": "error", "message": e, "fatal": true}));
                        return 1;
                    }
                };
                let (game, job) = job;
                return match run_with_control(game, job, sink.clone(), Some(rx), base) {
                    Ok(o) => {
                        sink.event(end_event(&o));
                        0
                    }
                    Err(e) => {
                        sink.event(json!({"ev": "error", "message": e, "fatal": true}));
                        1
                    }
                };
            }
            Some("ping") => sink.event(json!({"ev": "status", "stats": {"games": 0}})),
            Some("quit") => {
                sink.event(json!({"ev": "end", "reason": "quit", "stats": {}, "unfinished": Value::Null}));
                return 0;
            }
            Some("yield") => sink.event(json!({"ev": "yielded"})),
            Some("resume") => sink.event(json!({"ev": "resumed"})),
            other => sink.event(json!({"ev": "error", "message": format!("op {other:?} before a job"), "fatal": false})),
        }
    }
}

fn serve_eval(w: &'static Worker, line: &Value, sink: Arc<Stdout>, rx: std::sync::mpsc::Receiver<Option<Value>>, base: PathBuf) -> i32 {
    let job = eval_job_from_value(line, &base).and_then(|j| {
        let game = game_of(w.games, &j.settings)?;
        let bad = eval_settings_failures(w, game, &j.settings, &j.seats)?;
        if !bad.is_empty() {
            return Err(format!("the eval job's settings are refused (check-config --eval): {}", bad.join("; ")));
        }
        Ok((game, j))
    });
    let (game, job) = match job {
        Ok(x) => x,
        Err(e) => {
            sink.event(json!({"ev": "error", "message": e, "fatal": true}));
            return 1;
        }
    };
    match run_eval_with_control(game, job, sink.clone(), Some(rx), base) {
        Ok(o) => {
            sink.event(end_event(&o));
            0
        }
        Err(e) => {
            sink.event(json!({"ev": "error", "message": e, "fatal": true}));
            1
        }
    }
}

/// Control lines during a fuzz job.
fn fuzz_control(line: &Value, ctl: &FuzzControl, shared: &FuzzShared, sink: &dyn Sink) {
    match line.get("op").and_then(|x| x.as_str()).unwrap_or("") {
        "yield" => {
            match shared.do_yield() {
                Some(ms) => sink.log(&format!("yielded in {ms:.0} ms (chunk processes suspended)")),
                None => sink.log("yield: already yielded"),
            }
            sink.event(json!({"ev": "yielded"}));
        }
        "resume" => {
            shared.do_resume();
            sink.event(json!({"ev": "resumed"}));
        }
        "finish" => ctl.finish(),
        "quit" => ctl.quit(),
        "ping" => {
            let s = shared.progress.lock().unwrap().stats();
            sink.event(json!({"ev": "status", "stats": s}));
        }
        "checkpoint" => sink.event(json!({"ev": "error", "message": "checkpoint: a fuzz job has no network", "fatal": false})),
        "job" => sink.event(json!({"ev": "error", "message": "a second job line: one job per worker process", "fatal": false})),
        other => sink.event(json!({"ev": "error", "message": format!("unknown op {other:?}"), "fatal": false})),
    }
}

fn fuzz_end_event(o: &crate::fuzz::FuzzOutcome) -> Value {
    json!({"ev": "end", "reason": o.reason, "stats": o.stats, "unfinished": {"seeds": o.unfinished.json()}})
}

/// Run a fuzz job on a thread while this thread reads control lines (None: none) and emits a status
/// line every 10 s.
fn run_fuzz_with_control(fg: &'static dyn crate::fuzz::FuzzGame, job: crate::fuzz::FuzzJob, sink: Arc<Stdout>, lines: Option<std::sync::mpsc::Receiver<Option<Value>>>) -> Result<crate::fuzz::FuzzOutcome, String> {
    let ctl = Arc::new(FuzzControl::default());
    let shared = Arc::new(FuzzShared::default());
    let (c2, s2, k2) = (ctl.clone(), shared.clone(), sink.clone());
    let worker = std::thread::Builder::new()
        .name("zero-worker-fuzz".into())
        .spawn(move || crate::fuzz::run_fuzz(fg, &job, &c2, &*k2, &s2))
        .map_err(|e| format!("spawn: {e}"))?;
    let mut t_status = Instant::now();
    let mut eof = false;
    while !worker.is_finished() {
        match &lines {
            Some(rx) if !eof => match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(Some(v)) => fuzz_control(&v, &ctl, &shared, &*sink),
                Ok(None) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    eof = true;
                    sink.log("stdin closed: quitting the job");
                    shared.do_resume();
                    ctl.quit();
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            },
            _ => std::thread::sleep(Duration::from_millis(100)),
        }
        if t_status.elapsed().as_secs_f64() >= 10.0 {
            t_status = Instant::now();
            let s = shared.progress.lock().unwrap().stats();
            sink.event(json!({"ev": "status", "stats": s}));
        }
    }
    worker.join().map_err(|_| "the fuzz thread panicked".to_string())?
}

fn serve_fuzz(fg: &'static dyn crate::fuzz::FuzzGame, line: &Value, sink: Arc<Stdout>, rx: std::sync::mpsc::Receiver<Option<Value>>, base: PathBuf) -> i32 {
    let job = match crate::fuzz::job_from_value(line, &base).and_then(|j| fg.plan(&j.mode, &j.env).map(|_| j)) {
        Ok(j) => j,
        Err(e) => {
            sink.event(json!({"ev": "error", "message": e, "fatal": true}));
            return 1;
        }
    };
    match run_fuzz_with_control(fg, job, sink.clone(), Some(rx)) {
        Ok(o) => {
            sink.event(fuzz_end_event(&o));
            0
        }
        Err(e) => {
            sink.event(json!({"ev": "error", "message": e, "fatal": true}));
            1
        }
    }
}

fn env_args(a: &Args) -> Result<Vec<(String, String)>, String> {
    a.all("--env").iter().map(|kv| kv.split_once('=').map(|(k, v)| (k.to_string(), v.to_string())).ok_or_else(|| format!("--env {kv:?}: K=V"))).collect()
}

fn fuzz_cli(fg: &'static dyn crate::fuzz::FuzzGame, a: &Args) -> Result<i32, String> {
    let (env, fault) = crate::fuzz::split_env(env_args(a)?)?;
    let base = std::env::current_dir().unwrap_or_default();
    let job = crate::fuzz::FuzzJob {
        lease_id: a.get("--lease").unwrap_or("cli").to_string(),
        campaign: a.get("--campaign").map(String::from),
        mode: a.get("--mode").ok_or("--mode M")?.to_string(),
        env,
        first: a.num("--first")?.ok_or("--first F")?,
        count: a.num("--count")?.ok_or("--count N")?,
        threads: a.num("--threads")?.unwrap_or(1),
        chunk: a.num("--chunk")?,
        seed_timeout_s: a.num("--seed-timeout")?.unwrap_or(300.0),
        out_dir: base.join(a.get("--out").unwrap_or("out")),
        work_dir: base.join(a.get("--work").unwrap_or("fuzz")),
        announce_result: a.has("--announce"),
        keep_chunks: a.has("--keep-chunks"),
        fault,
    };
    fg.plan(&job.mode, &job.env)?;
    let sink = Arc::new(Stdout { mu: Mutex::new(()) });
    let lines = a.has("--stdin-control").then(|| stdin_lines(sink.clone()));
    let o = run_fuzz_with_control(fg, job, sink.clone(), lines)?;
    sink.event(fuzz_end_event(&o));
    Ok(0)
}

fn fuzz_check(fg: &'static dyn crate::fuzz::FuzzGame, a: &Args) -> Result<i32, String> {
    let (env, fault) = crate::fuzz::split_env(env_args(a)?)?;
    let mode = a.get("--mode").ok_or("--mode M")?;
    let plan = fg.plan(mode, &env)?;
    let (commit, profile) = match a.get("--commit") {
        Some(c) => (c.to_string(), a.get("--profile").unwrap_or(fg.profile()).to_string()),
        None => crate::fuzz::build_identity(fg)?,
    };
    let seed: u64 = a.num("--seed")?.unwrap_or(10_000_000);
    let mut renv = env.clone();
    if !fault.is_empty() {
        renv.push((crate::fuzz::FAULT_KEY.into(), crate::fuzz::faults_text(&fault)));
    }
    println!("{}", json!({"mode": plan.mode, "loop": plan.kind, "min_chunk": plan.min_chunk, "count_var": plan.count_var,
                          "env": env.iter().map(|(k, v)| json!([k, v])).collect::<Vec<_>>(), "profile": profile, "commit": commit,
                          "repro": fg.repro(&plan, &renv, seed, &commit, &profile)}));
    Ok(0)
}

fn selfplay_cli(games: &'static [GameEntry], a: &Args) -> Result<i32, String> {
    let settings = read_settings(a.get("--settings").ok_or("--settings <config.json>")?)?;
    let game = game_of(games, &settings)?;
    let threads: usize = a.num("--threads")?.unwrap_or(1);
    let ort = OrtCfg { dylib: ort_dylib(a.get("--ort-dll")), intra_threads: a.num("--infer-threads")?.unwrap_or(threads), spinning: a.get("--spinning") == Some("1") };
    let eval = if a.has("--uniform") {
        EvalKind::Uniform
    } else {
        EvalKind::Onnx { model: PathBuf::from(a.get("--model").ok_or("--model <onnx> or --uniform")?), ort: ort.clone(), sha256: a.get("--sha256").map(String::from), reference: a.get("--reference").map(PathBuf::from) }
    };
    let net = match (a.num::<u32>("--net")?, &eval) {
        (Some(n), _) => n,
        (None, EvalKind::Onnx { model, ort, sha256, .. }) => checkpoint_iter_checked(model, ort, sha256.as_deref())?,
        (None, EvalKind::Uniform) => 0,
    };
    let mut meta = Map::new();
    meta.insert("iter".into(), json!(net + 1));
    meta.insert("source".into(), json!("farm"));
    meta.insert("build_id".into(), json!(build_id()));
    let record = match (a.num::<usize>("--record-batches")?, a.get("--record-file")) {
        (Some(n), Some(f)) => Some((PathBuf::from(f), n)),
        (None, None) => None,
        _ => return Err("--record-batches N and --record-file F go together".into()),
    };
    let job = Job {
        lease_id: a.get("--lease").unwrap_or("cli").to_string(),
        settings,
        eval,
        games: a.num("--games")?.ok_or("--games N")?,
        gid_base: a.num("--gid-base")?.unwrap_or(0),
        seed: a.num("--seed")?.unwrap_or(0),
        net,
        search_threads: a.num("--search-threads")?.or(Some(threads)),
        conc: a.num("--conc")?,
        groups: a.num("--groups")?,
        out_dir: PathBuf::from(a.get("--out").unwrap_or("out")),
        prefix: a.get("--prefix").unwrap_or("f").to_string(),
        meta,
        record_batches: record,
        status_s: a.num("--status-s")?.unwrap_or(10.0),
        output: crate::records::Output::parse(a.get("--output").unwrap_or("shards"))?,
    };
    let sink = Arc::new(Stdout { mu: Mutex::new(()) });
    let lines = a.has("--stdin-control").then(|| stdin_lines(sink.clone()));
    let o = run_with_control(game, job, sink.clone(), lines, std::env::current_dir().unwrap_or_default())?;
    sink.event(end_event(&o));
    Ok(0)
}

fn eval_cli(w: &'static Worker, a: &Args) -> Result<i32, String> {
    use crate::evaljob::{EvalJob, ModelSpec};
    let settings = read_settings(a.get("--settings").ok_or("--settings <config.json>")?)?;
    let game = game_of(w.games, &settings)?;
    let threads: usize = a.num("--threads")?.unwrap_or(1);
    let seats = seats_of_value(&serde_json::from_str::<Value>(a.get("--seats").ok_or("--seats '[{\"net\": 1}, {\"net\": 2}]'")?).map_err(|e| format!("--seats: {e}"))?)?;
    let refs: std::collections::BTreeMap<u32, String> = gen_pairs(a, "--reference")?.into_iter().collect();
    let shas: std::collections::BTreeMap<u32, String> = gen_pairs(a, "--sha256")?.into_iter().collect();
    let mut models = std::collections::BTreeMap::new();
    for (g, m) in gen_pairs(a, "--model")? {
        models.insert(g, ModelSpec { model: PathBuf::from(m), sha256: shas.get(&g).cloned(), reference: refs.get(&g).map(PathBuf::from) });
    }
    let uniform = a.has("--uniform");
    if uniform {
        for s in &seats {
            if let crate::settings::EvalSeat::Net(g) = s {
                models.entry(*g).or_insert(ModelSpec { model: PathBuf::from("uniform"), sha256: None, reference: None });
            }
        }
    }
    let ort = (!uniform).then(|| OrtCfg { dylib: ort_dylib(a.get("--ort-dll")), intra_threads: a.num("--infer-threads").ok().flatten().unwrap_or(threads), spinning: a.get("--spinning") == Some("1") });
    let record = match (a.num::<usize>("--record-batches")?, a.get("--record-file")) {
        (Some(n), Some(f)) => Some((PathBuf::from(f), n)),
        (None, None) => None,
        _ => return Err("--record-batches N and --record-file F go together".into()),
    };
    let job = EvalJob {
        lease_id: a.get("--lease").unwrap_or("cli").to_string(),
        settings,
        seats,
        models,
        ort,
        games: a.num("--games")?.ok_or("--games N")?,
        gid_base: a.num("--gid-base")?.unwrap_or(0),
        seed: a.num("--seed")?.unwrap_or(0),
        paired: a.has("--paired"),
        search_threads: a.num("--search-threads")?.or(Some(threads)),
        conc: a.num("--conc")?,
        groups: a.num("--groups")?,
        out_dir: PathBuf::from(a.get("--out").unwrap_or("out")),
        prefix: a.get("--prefix").unwrap_or("e").to_string(),
        flush_games: a.num("--flush-games")?.unwrap_or(16),
        flush_s: a.num("--flush-s")?.unwrap_or(120.0),
        record_batches: record,
        status_s: a.num("--status-s")?.unwrap_or(10.0),
    };
    let bad = eval_settings_failures(w, game, &job.settings, &job.seats)?;
    if !bad.is_empty() {
        return Err(format!("the eval job's settings are refused (check-config --eval): {}", bad.join("; ")));
    }
    let sink = Arc::new(Stdout { mu: Mutex::new(()) });
    let lines = a.has("--stdin-control").then(|| stdin_lines(sink.clone()));
    let o = run_eval_with_control(game, job, sink.clone(), lines, std::env::current_dir().unwrap_or_default())?;
    sink.event(end_event(&o));
    Ok(0)
}

fn check_config(w: &'static Worker, a: &Args) -> Result<i32, String> {
    let games = w.games;
    let path = a.positional().ok_or("check-config <config.json>")?;
    let s = read_settings(path)?;
    let game = game_of(games, &s)?;
    let info = (game.info)();
    let others: Vec<_> = games.iter().filter(|g| g.name != game.name).map(|g| (g.info)()).collect();
    if a.has("--eval") {
        let seats = match a.get("--seats") {
            Some(t) => seats_of_value(&serde_json::from_str::<Value>(t).map_err(|e| format!("--seats: {e}"))?)?,
            None => vec![crate::settings::EvalSeat::Net(1), crate::settings::EvalSeat::Net(2)],
        };
        let v = classify::classify_eval(&s, &info, &others, w.package_settings)?;
        let bad = classify::failures(&v);
        let built = crate::settings::eval_driver_cfg(&s, &info, &seats, 2, 0, true, None, None, None).err();
        if a.has("--json") {
            let rows: Vec<Value> = v.iter().map(|x| json!({"key": x.key, "value": x.value, "status": x.status, "why": x.why})).collect();
            println!("{}", json!({"game": info.name, "job": "eval", "settings": rows, "failures": bad.iter().map(|x| &x.key).collect::<Vec<_>>(), "driver_error": built, "eval_reads": crate::settings::EVAL_READS}));
        } else {
            for st in ["play", "trainer", "unknown", "unsupported"] {
                let ks: Vec<&str> = v.iter().filter(|x| x.status == st).map(|x| x.key.as_str()).collect();
                println!("{st} ({}): {}", ks.len(), ks.join(", "));
            }
            for x in &bad {
                println!("REFUSED {} = {}: {} ({})", x.key, x.value, x.status, x.why);
            }
            if let Some(e) = &built {
                println!("REFUSED: the evaluation match: {e}");
            }
        }
        return Ok(if bad.is_empty() && built.is_none() { 0 } else { 1 });
    }
    let v = classify::classify_with(&s, &info, &others, w.package_settings)?;
    let bad = classify::failures(&v);
    // the settings must also build the driver (the Python remote worker's checks)
    let built = crate::settings::selfplay_driver_cfg(&s, &info, 0, 0).err();
    if a.has("--json") {
        let rows: Vec<Value> = v.iter().map(|x| json!({"key": x.key, "value": x.value, "status": x.status, "why": x.why})).collect();
        println!("{}", json!({"game": info.name, "settings": rows, "failures": bad.iter().map(|x| &x.key).collect::<Vec<_>>(), "driver_error": built}));
    } else {
        for st in ["play", "trainer", "off", "unsupported", "unknown"] {
            let ks: Vec<&str> = v.iter().filter(|x| x.status == st).map(|x| x.key.as_str()).collect();
            println!("{st} ({}): {}", ks.len(), ks.join(", "));
        }
        for x in &bad {
            println!("REFUSED {} = {}: {} ({})", x.key, x.value, x.status, x.why);
        }
        if let Some(e) = &built {
            println!("REFUSED: the self-play driver: {e}");
        }
    }
    Ok(if bad.is_empty() && built.is_none() { 0 } else { 1 })
}

fn driver_cfg_cli(games: &'static [GameEntry], a: &Args) -> Result<i32, String> {
    let s = read_settings(a.positional().ok_or("driver-cfg <config.json>")?)?;
    let game = game_of(games, &s)?;
    if a.has("--eval") {
        let seats = seats_of_value(&serde_json::from_str::<Value>(a.get("--seats").ok_or("--seats JSON")?).map_err(|e| format!("--seats: {e}"))?)?;
        let m = crate::settings::eval_driver_cfg(&s, &(game.info)(), &seats, a.num("--games")?.ok_or("--games N")?, a.num("--seed")?.unwrap_or(0), a.has("--paired"), a.num("--conc")?, a.num("--threads")?, a.num("--groups")?)?;
        let cfg = if a.has("--lease") { crate::settings::eval_lease_cfg(m) } else { m };
        println!("{cfg:#?}");
        return Ok(0);
    }
    let cfg = crate::settings::selfplay_driver_cfg(&s, &(game.info)(), a.num("--seed")?.unwrap_or(0), a.num("--net")?.unwrap_or(0))?;
    println!("{cfg:#?}");
    Ok(0)
}

fn eval_batches(a: &Args) -> Result<i32, String> {
    use crate::eval::Evaluator;
    use crate::npz::read_npz;
    use crate::shard::{write_npz, Npy};
    let model = PathBuf::from(a.get("--model").ok_or("--model")?);
    let threads: usize = a.num("--threads")?.unwrap_or(4);
    let ort = OrtCfg { dylib: ort_dylib(a.get("--ort-dll")), intra_threads: threads, spinning: a.get("--spinning") == Some("1") };
    let z = read_npz(Path::new(a.get("--batches").ok_or("--batches")?))?;
    let g = |k: &str| z.get(k).ok_or_else(|| format!("batches: no {k}"));
    let meta: Value = serde_json::from_str(&g("meta")?.text()?).map_err(|e| format!("batches meta: {e}"))?;
    let (xs_off, xs_idx, xs_val) = (g("xs_off")?.i64s()?, g("xs_idx")?.i64s()?, g("xs_val")?.f32s()?);
    let (np_, extra, opt_off, opt_action, bo) = (g("n_players")?.i64s()?, g("extra")?.i64s()?, g("opt_off")?.i64s()?, g("opt_action")?.i64s()?, g("batch_off")?.i64s()?);
    let mut ev = {
        // the graph's own metadata as the expectation (the gate compares answers, not the game)
        let env = ort_env(&ort.dylib)?;
        let s = ort::session::Session::builder(env).map_err(|e| e.to_string())?.commit_from_file(&model).map_err(|e| e.to_string())?;
        let md = s.metadata().map_err(|e| e.to_string())?;
        let num = |k: &str| md.custom(k).and_then(|v| v.parse::<usize>().ok()).ok_or_else(|| format!("{}: metadata {k}", model.display()));
        let want = crate::eval::Expect { game: md.custom("game").unwrap_or_default(), features: num("features")?, actions: num("actions")?, aux_width: num("aux_width")?, belief: false };
        drop(md);
        OrtEvaluator::load(&model, ort, &want)?
    };
    let reps: usize = a.num("--reps")?.unwrap_or(1);
    let (mut pri, mut val, mut mar, mut aux) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut secs = Vec::new();
    for rep in 0..reps {
        for k in 0..bo.len() - 1 {
            let (r0, r1) = (bo[k] as usize, bo[k + 1] as usize);
            let mut b = zero_search::driver::Batch::default();
            b.id = k as u64;
            b.b = r1 - r0;
            b.xs_off = xs_off[r0..=r1].iter().map(|&x| x - xs_off[r0]).collect();
            b.xs_idx = xs_idx[xs_off[r0] as usize..xs_off[r1] as usize].iter().map(|&x| x as u32).collect();
            b.xs_val = xs_val[xs_off[r0] as usize..xs_off[r1] as usize].to_vec();
            b.n_players = np_[r0..r1].iter().map(|&x| x as i32).collect();
            b.opt_off = opt_off[r0..=r1].iter().map(|&x| (x - opt_off[r0]) as i32).collect();
            b.opt_action = opt_action[opt_off[r0] as usize..opt_off[r1] as usize].iter().map(|&x| x as i32).collect();
            b.extra = extra[r0..r1].iter().map(|&x| x as u8).collect();
            b.player = vec![0; r1 - r0];
            b.gid = vec![0; r1 - r0];
            let t0 = Instant::now();
            let ans = ev.evaluate(&b)?;
            secs.push(t0.elapsed().as_secs_f64());
            if rep == 0 {
                pri.extend(ans.priors);
                val.extend(ans.value);
                mar.extend(ans.margin);
                aux.extend(ans.aux);
            }
        }
    }
    let f32n = |v: &[f32], shape: Vec<usize>| Npy { descr: "<f4".into(), shape, data: v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let rows = *bo.last().unwrap() as usize;
    let w = if rows > 0 && !aux.is_empty() { aux.len() / extra.iter().filter(|&&e| e == 2).count().max(1) } else { 0 };
    let info = json!({"model": model.display().to_string(), "ort": ort_version(), "threads": threads, "batches": bo.len() - 1, "rows": rows, "reps": reps, "seconds": secs, "source": meta});
    write_npz(Path::new(a.get("--out").ok_or("--out")?), &[("meta", Npy::unicode(&info.to_string())), ("priors", f32n(&pri, vec![pri.len()])), ("value", f32n(&val, vec![rows, 8])), ("margin", f32n(&mar, vec![rows, 8])), ("aux", f32n(&aux, vec![aux.len() / w.max(1), w]))], 1)?;
    println!("{}", json!({"batches": bo.len() - 1, "rows": rows, "seconds": secs.iter().sum::<f64>(), "ort": ort_version()}));
    Ok(0)
}

fn info_cli(games: &'static [GameEntry], repo: &str, kinds: Vec<&'static str>, a: &Args) -> Result<i32, String> {
    let dylib = ort_dylib_in(a.get("--ort-dll"), a.get("--runtime-dir"));
    let ort = match ort_env(&dylib) {
        Ok(_) => {
            // where each library came from (the runtime pack's check on a clean PC)
            let mods: Map<String, Value> = ["onnxruntime.dll"].iter().chain(crate::eval::MSVC_RUNTIME).map(|n| (n.to_string(), json!(crate::eval::module_path(n)))).collect();
            json!({"dylib": dylib.display().to_string(), "version": ort_version(), "modules": mods})
        }
        Err(e) => json!({"dylib": dylib.display().to_string(), "error": e}),
    };
    let gs: Vec<Value> = games.iter().map(|g| {
        let i = (g.info)();
        json!({"name": i.name, "features": i.num_features, "actions": i.num_actions, "aux": i.aux.iter().map(|x| json!([x.0, x.1])).collect::<Vec<_>>()})
    }).collect();
    println!("{}", json!({"repo": repo, "aw": AW, "zero_onnx": ZERO_ONNX, "build_id": build_id(), "games": gs, "onnxruntime": ort, "kinds": kinds}));
    Ok(0)
}

pub fn cli(games: &'static [GameEntry], repo: &'static str, args: &[String]) -> i32 {
    cli_with(Box::leak(Box::new(Worker { games, repo, selfplay: true, fuzz: None, package_settings: &[] })), args)
}

pub fn cli_with(w: &'static Worker, args: &[String]) -> i32 {
    let (games, repo) = (w.games, w.repo);
    let (cmd, rest) = match args.split_first() {
        Some((c, r)) => (c.as_str(), r),
        None => ("", &[][..]),
    };
    let a = Args { v: rest };
    let no_fuzz = || Err::<i32, String>("this worker build has no fuzz loops".into());
    let r = match cmd {
        "serve" => return serve(w),
        "fuzz-chunk" => {
            return match (w.fuzz, a.positional()) {
                (Some(fg), Some(spec)) => crate::fuzz::chunk_main(fg, spec),
                _ => {
                    eprintln!("usage: <worker> fuzz-chunk <spec.json> (a fuzz build's)");
                    2
                }
            }
        }
        "fuzz" => w.fuzz.map_or_else(no_fuzz, |fg| fuzz_cli(fg, &a)),
        "fuzz-check" => w.fuzz.map_or_else(no_fuzz, |fg| fuzz_check(fg, &a)),
        "selfplay" | "eval" if !w.selfplay => Err("this worker build runs fuzz jobs only (a fuzz-profile build)".into()),
        "selfplay" => selfplay_cli(games, &a),
        "eval" => eval_cli(w, &a),
        "check-config" => check_config(w, &a),
        "driver-cfg" => driver_cfg_cli(games, &a),
        "eval-batches" => eval_batches(&a),
        "info" => info_cli(games, repo, w.kinds(), &a),
        _ => Err("usage: <worker> serve | selfplay ... | eval ... | fuzz ... | fuzz-check ... | check-config <config.json> | driver-cfg <config.json> | eval-batches ... | info (zero-core crates/worker/src/proto.rs)".into()),
    };
    match r {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            2
        }
    }
}
