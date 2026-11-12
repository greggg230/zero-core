//! The worker's command line and the agent<->worker protocol `aw/1` (zero-farm PROTOCOL.md 5).
//!
//! ```text
//! <worker> serve                         aw/1: JSON lines in (op) / out (ev); stdout JSON only, stderr the log
//! <worker> selfplay --settings F (--model M | --uniform) --games N [--gid-base G] [--seed S] [--net N]
//!          [--threads T] [--infer-threads I] [--conc C] [--groups G] [--out DIR] [--prefix P]
//!          [--ort-dll PATH] [--spinning 0|1] [--record-batches N --record-file F] [--sha256 H]
//!          [--status-s X] [--stdin-control]        one job from flags (the same runner; events on stdout)
//! <worker> check-config <config.json> [--json]     classify every setting; exit 1 on unknown / unsupported
//! <worker> driver-cfg <config.json> --seed S --net N   the Python remote worker's DriverCfg (Debug text)
//! <worker> eval-batches --model M --batches F --out F [--threads T] [--ort-dll P]
//!                                                  answers to recorded batches (the evaluator's gate)
//! <worker> info                                    games, protocol, ONNX Runtime
//! ```
//!
//! `serve`: `hello` first (aw, build_id, game, kinds, zero_onnx, pid), then one `job` line (`op` job:
//! lease_id, kind 'selfplay', settings, games, gid_base, seed, model, reference, threads, out_dir;
//! optional gen - the rows' network id, default the graph's `checkpoint_iter` -, sha256, conc,
//! groups, infer_threads, spinning, prefix); during it `yield` -> `yielded` (no batch started, the
//! session dropped), `resume` -> `resumed`, `finish`, `quit`, `checkpoint` {gen, model, sha256?},
//! `ping` -> `status`; events `fin`, `shard` (seq, path under out_dir, games, rows, bytes, gids:
//! half-open [a, b) ranges), `status` every <= 10 s (PROTOCOL.md 3 Stats fields), `end` (reason, stats,
//! unfinished {gids: [[a, b)]}), `error` (message, fatal). The process exits after `end` (or a fatal
//! error, status 1).
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};

use crate::eval::{ort_env, ort_version, OrtCfg, OrtEvaluator};
use crate::job::{do_resume, do_yield, run_job, Checkpoint, Control, EvalKind, EvalLink, Job, Outcome, Progress, Sink};
use crate::settings::Settings;
use crate::{classify, GameEntry, AW, ZERO_ONNX};

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
    fn positional(&self) -> Option<&str> {
        self.v.first().filter(|a| !a.starts_with("--")).map(|s| s.as_str())
    }
}

/// The ONNX Runtime library: `--ort-dll`, else env ZERO_ORT_DYLIB / ORT_DYLIB_PATH, else
/// `onnxruntime.dll` (`libonnxruntime.so` / `.dylib`) next to the executable.
pub fn ort_dylib(given: Option<&str>) -> PathBuf {
    if let Some(g) = given {
        return PathBuf::from(g);
    }
    for k in ["ZERO_ORT_DYLIB", "ORT_DYLIB_PATH"] {
        if let Ok(v) = std::env::var(k) {
            if !v.is_empty() {
                return PathBuf::from(v);
            }
        }
    }
    let name = if cfg!(windows) {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    };
    std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join(name))).unwrap_or_else(|| PathBuf::from(name))
}

/// The build id: the first 24 hex digits of sha256(manifest.json) next to the executable (PROTOCOL.md
/// 2), else env ZERO_FARM_BUILD_ID, else 'unsigned-dev'.
fn build_id() -> String {
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
    let ort = OrtCfg { dylib: ort_dylib(o.get("ort_dylib").and_then(|x| x.as_str())), intra_threads: u("infer_threads").map_or(threads, |t| t as usize), spinning: o.get("spinning").and_then(|x| x.as_bool()).unwrap_or(false) };
    let sha256 = o.get("sha256").and_then(|x| x.as_str()).map(String::from);
    let eval = match o.get("model").and_then(|m| m.as_str()) {
        Some(m) => EvalKind::Onnx { model: base.join(m), ort, sha256 },
        None if o.get("uniform").and_then(|x| x.as_bool()) == Some(true) => EvalKind::Uniform,
        None => return Err("job: no model (a self-play job needs its network)".into()),
    };
    let net = match (u("gen"), &eval) {
        (Some(g), _) => g as u32,
        (None, EvalKind::Onnx { model, ort, .. }) => checkpoint_iter(model, ort)?,
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
    })
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
                (Some(g), Some(m)) => ctl.checkpoint(Checkpoint { gen: g as u32, model: base.join(m), sha256: line.get("sha256").and_then(|x| x.as_str()).map(String::from) }),
                _ => sink.event(json!({"ev": "error", "message": "checkpoint: gen and model are required", "fatal": false})),
            }
        }
        "ping" => {
            let mut s = progress.lock().unwrap().stats();
            s["ev"] = json!("status");
            sink.event(s);
        }
        "job" => sink.event(json!({"ev": "error", "message": "a second job line: one job per worker process", "fatal": false})),
        other => sink.event(json!({"ev": "error", "message": format!("unknown op {other:?}"), "fatal": false})),
    }
}

/// Run `job` on a thread while this thread reads control lines from `lines` (None: no control) and
/// emits a status line every `status_s` -> the outcome.
fn run_with_control(game: &'static GameEntry, mut job: Job, sink: Arc<Stdout>, lines: Option<std::sync::mpsc::Receiver<Option<Value>>>, base: PathBuf) -> Result<Outcome, String> {
    let ctl = Arc::new(Control::default());
    let link: Arc<Mutex<Option<EvalLink>>> = Arc::new(Mutex::new(None));
    let progress = Arc::new(Mutex::new(Progress::default()));
    // (the status lines come from this thread, which also runs while the job waits on its search threads)
    let status_s = job.status_s;
    job.status_s = f64::INFINITY;
    let (c2, l2, p2, s2) = (ctl.clone(), link.clone(), progress.clone(), sink.clone());
    let worker = std::thread::Builder::new()
        .name("zero-worker-job".into())
        .spawn(move || {
            let lk = l2.clone();
            let c3 = c2.clone();
            run_job(game, &job, &c2, &*s2, p2, &move |l| {
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
            let mut s = progress.lock().unwrap().stats();
            s["ev"] = json!("status");
            sink.event(s);
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

fn serve(games: &'static [GameEntry], repo: &str) -> i32 {
    let sink = Arc::new(Stdout { mu: Mutex::new(()) });
    sink.event(json!({"ev": "hello", "aw": AW, "build_id": build_id(), "game": repo, "kinds": ["selfplay"], "zero_onnx": ZERO_ONNX, "pid": std::process::id(), "games": games.iter().map(|g| g.name).collect::<Vec<_>>()}));
    let rx = stdin_lines(sink.clone());
    let base = std::env::current_dir().unwrap_or_default();
    loop {
        let line = match rx.recv() {
            Ok(Some(v)) => v,
            _ => return 0,
        };
        match line.get("op").and_then(|x| x.as_str()) {
            Some("job") => {
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
            Some("ping") => sink.event(json!({"ev": "status", "games": 0})),
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

fn selfplay_cli(games: &'static [GameEntry], a: &Args) -> Result<i32, String> {
    let settings = read_settings(a.get("--settings").ok_or("--settings <config.json>")?)?;
    let game = game_of(games, &settings)?;
    let threads: usize = a.num("--threads")?.unwrap_or(1);
    let ort = OrtCfg { dylib: ort_dylib(a.get("--ort-dll")), intra_threads: a.num("--infer-threads")?.unwrap_or(threads), spinning: a.get("--spinning") == Some("1") };
    let eval = if a.has("--uniform") {
        EvalKind::Uniform
    } else {
        EvalKind::Onnx { model: PathBuf::from(a.get("--model").ok_or("--model <onnx> or --uniform")?), ort: ort.clone(), sha256: a.get("--sha256").map(String::from) }
    };
    let net = match (a.num::<u32>("--net")?, &eval) {
        (Some(n), _) => n,
        (None, EvalKind::Onnx { model, ort, .. }) => checkpoint_iter(model, ort)?,
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
    };
    let sink = Arc::new(Stdout { mu: Mutex::new(()) });
    let lines = a.has("--stdin-control").then(|| stdin_lines(sink.clone()));
    let o = run_with_control(game, job, sink.clone(), lines, std::env::current_dir().unwrap_or_default())?;
    sink.event(end_event(&o));
    Ok(0)
}

fn check_config(games: &'static [GameEntry], a: &Args) -> Result<i32, String> {
    let path = a.positional().ok_or("check-config <config.json>")?;
    let s = read_settings(path)?;
    let game = game_of(games, &s)?;
    let info = (game.info)();
    let others: Vec<_> = games.iter().filter(|g| g.name != game.name).map(|g| (g.info)()).collect();
    let v = classify::classify(&s, &info, &others)?;
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
        let want = crate::eval::Expect { game: md.custom("game").unwrap_or_default(), features: num("features")?, actions: num("actions")?, aux_width: num("aux_width")?, belief: false, sha256: None };
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

fn info_cli(games: &'static [GameEntry], repo: &str, a: &Args) -> Result<i32, String> {
    let dylib = ort_dylib(a.get("--ort-dll"));
    let ort = match ort_env(&dylib) {
        Ok(_) => json!({"dylib": dylib.display().to_string(), "version": ort_version()}),
        Err(e) => json!({"dylib": dylib.display().to_string(), "error": e}),
    };
    let gs: Vec<Value> = games.iter().map(|g| {
        let i = (g.info)();
        json!({"name": i.name, "features": i.num_features, "actions": i.num_actions, "aux": i.aux.iter().map(|x| json!([x.0, x.1])).collect::<Vec<_>>()})
    }).collect();
    println!("{}", json!({"repo": repo, "aw": AW, "zero_onnx": ZERO_ONNX, "build_id": build_id(), "games": gs, "onnxruntime": ort}));
    Ok(0)
}

pub fn cli(games: &'static [GameEntry], repo: &'static str, args: &[String]) -> i32 {
    let (cmd, rest) = match args.split_first() {
        Some((c, r)) => (c.as_str(), r),
        None => ("", &[][..]),
    };
    let a = Args { v: rest };
    let r = match cmd {
        "serve" => return serve(games, repo),
        "selfplay" => selfplay_cli(games, &a),
        "check-config" => check_config(games, &a),
        "driver-cfg" => driver_cfg_cli(games, &a),
        "eval-batches" => eval_batches(&a),
        "info" => info_cli(games, repo, &a),
        _ => Err("usage: <worker> serve | selfplay ... | check-config <config.json> | driver-cfg <config.json> | eval-batches ... | info (zero-core crates/worker/src/proto.rs)".into()),
    };
    match r {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            2
        }
    }
}
