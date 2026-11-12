//! Fuzz jobs (zero-farm PROTOCOL.md 5 `fuzz` and 5.1): a game's fuzz loop over a lease's seeds, run
//! in supervised chunk processes so that no seed can take the lease down.
//!
//! The game binary registers a [`FuzzGame`]: which loop a job's `mode` and `env` mean (refusing
//! settings the loop would ignore), the loop itself over a range of seeds (in the chunk process), and
//! the reproduction string of a failing seed. Everything else is here and game-agnostic:
//!
//! * the lease's seeds `[first, first + count)` are cut into chunks (at least the loop's
//!   `min_chunk` seeds, so its checks over a whole chunk still mean what they mean in the game's own
//!   fuzz farm), and up to `threads` chunk processes run at a time (`<worker> fuzz-chunk <spec>`,
//!   one thread each, at the worker's priority, no console window);
//! * a chunk process plays its seeds in order and writes one progress line per seed (start, pass,
//!   fail) to a file; stdout and stderr (the loop's report, the panic messages) go to its log;
//! * a seed that panics ends its chunk there (the game's loop stops at a failing seed, as its tests
//!   do): it is reported as a `failure` (seed, mode, panic, site = the panic's `file:line`, the
//!   reproduction, the log's tail) and the chunk goes on from the next seed in a new process;
//! * a chunk process that dies without unwinding (an abort, a stack overflow) or runs one seed past
//!   the per-seed limit of CPU time (`seed_timeout_s`, default 300 s: CPU time, so a PC that starves
//!   the worker of CPU is not a hang) is cut at the seed it was playing - the seeds before it
//!   passed - and that seed is run alone: if it crashes or hangs again it is the failure (site
//!   `crash exit <code>` / `HANG <mode>`, as `tools/fuzz_farm.py` names them), and either way the
//!   run goes on after it. A crashing seed never kills the lease and is never handed out again;
//! * `yield` suspends the chunk processes (and their clocks), `resume` resumes them, `finish` lets
//!   each finish the seed it is playing, `quit` kills them: the seeds that passed are kept, the rest
//!   is `unfinished`;
//! * at the end the 5.1 summary (`final: true`) is written to `out_dir/fuzz-final.json`; the agent
//!   (0.1.3) builds and uploads its own from the `failure` events and `end`, so the worker announces
//!   its file only when the job asks (`announce_result: true`).
use std::collections::VecDeque;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};

use crate::job::Sink;

/// A game binary's fuzz loops.
pub trait FuzzGame: Sync {
    /// What `mode` with the settings `env` (in the job's order) means: the loop and its chunk floor;
    /// an error for a mode the binary doesn't know, or a setting the loop would ignore or can't take.
    fn plan(&self, mode: &str, env: &[(String, String)]) -> Result<FuzzPlan, String>;
    /// In a chunk process: play seeds `first..first + count` of `plan`'s loop with the settings `env`
    /// (the loop's only settings in this process), seed by seed through `hooks`.
    fn run_chunk(&self, plan: &FuzzPlan, env: &[(String, String)], first: u64, count: u64, hooks: &mut dyn ChunkHooks) -> ChunkEnd;
    /// How to rerun seed `seed` alone at commit `commit`, built with cargo profile `profile`.
    fn repro(&self, plan: &FuzzPlan, env: &[(String, String)], seed: u64, commit: &str, profile: &str) -> String;
    /// The cargo profile this binary was built with ('release', 'fuzz' or 'dev').
    fn profile(&self) -> &'static str;
}

/// A fuzz job's loop, as the game decided it.
#[derive(Clone, Debug)]
pub struct FuzzPlan {
    pub mode: String,
    /// The loop (the game's name for it, e.g. 'games').
    pub kind: String,
    /// Seeds per chunk at least.
    pub min_chunk: u64,
    /// The loop's count setting, for a whole chunk's reproduction (`(the chunk: FUZZ_GAMES=n)`).
    pub count_var: String,
}

/// What a chunk's loop tells the chunk process, seed by seed.
pub trait ChunkHooks {
    /// Before seed `seed`: false stops the chunk here (the agent's `finish`).
    fn before(&mut self, seed: u64) -> bool;
    /// Seed `seed` passed; `outcome` is the game's description of it (e.g. its fingerprint).
    fn passed(&mut self, seed: u64, outcome: Value);
    /// A failure inside seed `seed` that does not stop it (e.g. a draft event's heuristic-played match
    /// panicking in the engine): `message` is the failure's text as the loop reports it, `repro_suffix`
    /// what follows the seed's reproduction, `fallback_site` the site when no recorded panic has the
    /// message.
    fn side_failure(&mut self, seed: u64, message: &str, repro_suffix: &str, fallback_site: &str);
}

/// How a chunk's loop ended.
pub enum ChunkEnd {
    /// Every seed passed (and the loop's checks over the chunk held).
    Done,
    /// Seed `seed` panicked; nothing after it was played.
    Failed { seed: u64, payload: String },
    /// `before` stopped it at seed `at`.
    Stopped { at: u64 },
}

/// A fuzz job (PROTOCOL.md 5: `{"kind": "fuzz", "mode", "env", "first", "count"}` + the agent's fields).
#[derive(Clone, Debug)]
pub struct FuzzJob {
    pub lease_id: String,
    pub campaign: Option<String>,
    pub mode: String,
    /// The settings in the job's order (worker-only keys removed: see `fault`).
    pub env: Vec<(String, String)>,
    pub first: u64,
    pub count: u64,
    pub threads: usize,
    /// Seeds per chunk (default: max(the loop's floor, count / (4 threads))).
    pub chunk: Option<u64>,
    /// CPU seconds one seed may take before it counts as a hang.
    pub seed_timeout_s: f64,
    pub out_dir: PathBuf,
    /// Where the chunk processes' files go.
    pub work_dir: PathBuf,
    pub announce_result: bool,
    /// Keep every chunk's files (spec, progress, log), not only failing ones' (checks, debugging).
    pub keep_chunks: bool,
    /// Test-only faults (`ZERO_FARM_TEST_FAULT`): see [`Fault`].
    pub fault: Vec<Fault>,
}

/// A test-only fault a chunk process injects at a seed, to test the supervision (`env`
/// `ZERO_FARM_TEST_FAULT=panic:S,abort:S,overflow:S,hang:S`). Taken only by builds with debug
/// assertions (the `fuzz` profile) whose process environment has `ZERO_FARM_ALLOW_TEST_FAULTS=1`;
/// anywhere else the key is refused like any unknown setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    Panic(u64),
    Abort(u64),
    Overflow(u64),
    Hang(u64),
}

pub const FAULT_KEY: &str = "ZERO_FARM_TEST_FAULT";

pub fn faults_allowed() -> bool {
    cfg!(debug_assertions) && std::env::var("ZERO_FARM_ALLOW_TEST_FAULTS").as_deref() == Ok("1")
}

pub fn parse_faults(s: &str) -> Result<Vec<Fault>, String> {
    s.split(',')
        .filter(|x| !x.trim().is_empty())
        .map(|x| {
            let (k, v) = x.trim().split_once(':').ok_or_else(|| format!("{FAULT_KEY}: {x:?} is not kind:seed"))?;
            let seed: u64 = v.parse().map_err(|_| format!("{FAULT_KEY}: {x:?}: the seed is not a number"))?;
            Ok(match k {
                "panic" => Fault::Panic(seed),
                "abort" => Fault::Abort(seed),
                "overflow" => Fault::Overflow(seed),
                "hang" => Fault::Hang(seed),
                _ => return Err(format!("{FAULT_KEY}: {k:?}: panic, abort, overflow or hang")),
            })
        })
        .collect()
}

/// Faults as `ZERO_FARM_TEST_FAULT` writes them.
pub fn faults_text(f: &[Fault]) -> String {
    f.iter()
        .map(|f| match f {
            Fault::Panic(s) => format!("panic:{s}"),
            Fault::Abort(s) => format!("abort:{s}"),
            Fault::Overflow(s) => format!("overflow:{s}"),
            Fault::Hang(s) => format!("hang:{s}"),
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// The job's settings, split: the loop's, and the faults (only where [`faults_allowed`]).
pub fn split_env(env: Vec<(String, String)>) -> Result<(Vec<(String, String)>, Vec<Fault>), String> {
    let mut loop_env = Vec::new();
    let mut faults = Vec::new();
    for (k, v) in env {
        if k == FAULT_KEY {
            if !faults_allowed() {
                return Err(format!("{FAULT_KEY}: test faults are taken only by a fuzz-profile build with ZERO_FARM_ALLOW_TEST_FAULTS=1"));
            }
            faults = parse_faults(&v)?;
        } else {
            loop_env.push((k, v));
        }
    }
    Ok((loop_env, faults))
}

// ---------------------------------------------------------------------------- id ranges

/// Sorted, disjoint, half-open ranges.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ranges(pub Vec<(u64, u64)>);

impl Ranges {
    pub fn add(&mut self, lo: u64, hi: u64) {
        if lo >= hi {
            return;
        }
        let v = &mut self.0;
        let i = v.partition_point(|r| r.1 < lo);
        let mut j = i;
        let (mut a, mut b) = (lo, hi);
        while j < v.len() && v[j].0 <= hi {
            a = a.min(v[j].0);
            b = b.max(v[j].1);
            j += 1;
        }
        v.splice(i..j, [(a, b)]);
    }
    pub fn total(&self) -> u64 {
        self.0.iter().map(|r| r.1 - r.0).sum()
    }
    /// `[lo, hi)` minus these.
    pub fn missing(&self, lo: u64, hi: u64) -> Ranges {
        let mut out = Vec::new();
        let mut at = lo;
        for &(a, b) in &self.0 {
            if b <= at || a >= hi {
                continue;
            }
            if a > at {
                out.push((at, a.min(hi)));
            }
            at = at.max(b);
        }
        if at < hi {
            out.push((at, hi));
        }
        Ranges(out)
    }
    pub fn json(&self) -> Value {
        json!(self.0.iter().map(|r| [r.0, r.1]).collect::<Vec<_>>())
    }
}

// ---------------------------------------------------------------------------- the chunk process

/// One panic seen by the hook (the chunk process records every panic, caught or not).
#[derive(Clone, Debug)]
struct PanicRec {
    thread: String,
    file: String,
    line: u32,
    col: u32,
    message: String,
}

impl PanicRec {
    /// The dedupe key `tools/fuzz_farm.py` uses for a panic: its `file:line` ('/'-separated).
    fn site(&self) -> String {
        format!("{}:{}", self.file.replace('\\', "/"), self.line)
    }
    /// The panic as Rust's hook prints it.
    fn text(&self) -> String {
        format!("thread '{}' panicked at {}:{}:{}:\n{}", self.thread, self.file, self.line, self.col, self.message)
    }
}

static PANICS: Mutex<Vec<PanicRec>> = Mutex::new(Vec::new());

fn panics() -> std::sync::MutexGuard<'static, Vec<PanicRec>> {
    PANICS.lock().unwrap_or_else(|e| e.into_inner())
}

/// The message of a panic payload.
pub fn payload_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "Box<dyn Any>".into())
}

fn install_panic_recorder() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let (file, line, col) = info.location().map(|l| (l.file().to_string(), l.line(), l.column())).unwrap_or_default();
        let thread = std::thread::current().name().unwrap_or("<unnamed>").to_string();
        panics().push(PanicRec { thread, file, line, col, message: payload_text(info.payload()) });
        default(info);
    }));
}

/// The chunk process's progress file (one JSON line per event, each written with one call).
struct Progress {
    f: std::fs::File,
}

impl Progress {
    fn put(&mut self, v: Value) {
        let mut line = v.to_string();
        line.push('\n');
        // (a progress line that can't be written: the supervisor sees the chunk die without an end)
        if self.f.write_all(line.as_bytes()).is_err() {
            std::process::exit(70);
        }
    }
}

fn panic_json(p: Option<&PanicRec>, payload: Option<&str>) -> Value {
    match p {
        Some(p) => json!({"site": p.site(), "panic": p.text()}),
        None => json!({"site": "unknown", "panic": payload.unwrap_or("")}),
    }
}

struct ChildHooks {
    out: Progress,
    stop: Arc<AtomicBool>,
    faults: Vec<Fault>,
    /// The seed being played.
    cur: Option<u64>,
    /// Panics of `cur` before this index were side failures' own.
    side_idx: usize,
}

/// A stack overflow (the `overflow` test fault).
#[inline(never)]
#[allow(unconditional_recursion)]
fn overflow(depth: u64) -> u64 {
    let pad = [depth; 64];
    std::hint::black_box(&pad);
    1 + overflow(std::hint::black_box(depth + 1)) + pad[(depth % 64) as usize]
}

impl ChunkHooks for ChildHooks {
    fn before(&mut self, seed: u64) -> bool {
        if self.stop.load(Ordering::SeqCst) {
            return false;
        }
        panics().clear();
        self.side_idx = 0;
        self.cur = Some(seed);
        self.out.put(json!({"ev": "start", "seed": seed}));
        for f in self.faults.clone() {
            match f {
                Fault::Panic(s) if s == seed => panic!("{FAULT_KEY}: an injected panic at seed {seed}"),
                Fault::Abort(s) if s == seed => std::process::abort(),
                Fault::Overflow(s) if s == seed => {
                    std::hint::black_box(overflow(0));
                }
                Fault::Hang(s) if s == seed => {
                    let mut x = 0u64;
                    loop {
                        x = std::hint::black_box(x.wrapping_mul(6364136223846793005).wrapping_add(1));
                    }
                }
                _ => {}
            }
        }
        true
    }
    fn passed(&mut self, seed: u64, outcome: Value) {
        self.out.put(json!({"ev": "pass", "seed": seed, "outcome": outcome}));
        self.cur = None;
        panics().clear();
        self.side_idx = 0;
    }
    fn side_failure(&mut self, seed: u64, message: &str, repro_suffix: &str, fallback_site: &str) {
        // `tools/fuzz_farm.py`'s rule for a listed match panic: the site of the first recorded panic
        // whose message starts like the listed one (first line, 60 characters).
        let head = |s: &str| -> String { s.lines().next().unwrap_or("").trim().chars().take(60).collect() };
        let want = head(message);
        let ps = panics();
        let found = ps.iter().find(|p| head(&p.message) == want).cloned();
        let n = ps.len();
        drop(ps);
        self.side_idx = n;
        let (site, panic) = match found {
            Some(p) => (p.site(), p.text()),
            None => (fallback_site.to_string(), message.to_string()),
        };
        self.out.put(json!({"ev": "side", "seed": seed, "site": site, "panic": panic, "repro_suffix": repro_suffix}));
    }
}

/// `<worker> fuzz-chunk <spec.json>`: one chunk (spec: mode, env, first, count, progress, faults).
pub fn chunk_main(game: &dyn FuzzGame, spec_path: &str) -> i32 {
    let spec: Value = match std::fs::read_to_string(spec_path).map_err(|e| e.to_string()).and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string())) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("fuzz-chunk: {spec_path}: {e}");
            return 2;
        }
    };
    let env: Vec<(String, String)> = spec["env"].as_array().map(|a| a.iter().filter_map(|p| Some((p[0].as_str()?.to_string(), p[1].as_str()?.to_string()))).collect()).unwrap_or_default();
    let faults = spec["faults"].as_str().map(parse_faults).transpose().unwrap_or_else(|e| {
        eprintln!("fuzz-chunk: {e:?}");
        std::process::exit(2)
    }).unwrap_or_default();
    let (first, count) = (spec["first"].as_u64().unwrap_or(0), spec["count"].as_u64().unwrap_or(0));
    let plan = match game.plan(spec["mode"].as_str().unwrap_or(""), &env) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("fuzz-chunk: {e}");
            return 2;
        }
    };
    let f = match std::fs::OpenOptions::new().append(true).create(true).open(spec["progress"].as_str().unwrap_or("progress.jsonl")) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("fuzz-chunk: progress file: {e}");
            return 2;
        }
    };
    // stdin: "stop" (or its end: the supervisor is gone) -> stop before the next seed
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        std::thread::spawn(move || {
            let mut line = String::new();
            loop {
                line.clear();
                match std::io::stdin().read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) if line.trim() == "stop" => break,
                    Ok(_) => {}
                }
            }
            stop.store(true, Ordering::SeqCst);
        });
    }
    // (a crash ends the process quietly: no error dialog on a friend's screen - the agent's job object
    // does the same for the worker)
    os::quiet_crashes();
    install_panic_recorder();
    let mut hooks = ChildHooks { out: Progress { f }, stop, faults, cur: None, side_idx: 0 };
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| game.run_chunk(&plan, &env, first, count, &mut hooks)));
    match r {
        Ok(ChunkEnd::Done) => hooks.out.put(json!({"ev": "end", "how": "done"})),
        Ok(ChunkEnd::Stopped { at }) => hooks.out.put(json!({"ev": "end", "how": "stopped", "at": at})),
        Ok(ChunkEnd::Failed { seed, payload }) => {
            let p = panics().get(hooks.side_idx).cloned();
            let mut v = panic_json(p.as_ref(), Some(&payload));
            v["ev"] = json!("fail");
            v["seed"] = json!(seed);
            hooks.out.put(v);
        }
        Err(payload) => {
            // a panic outside the loop's per-seed catch: the seed in play (an assertion of its own
            // outside the catch, or a fault), else the chunk's own checks after its last seed
            let text = payload_text(&*payload);
            let p = panics().get(hooks.side_idx).cloned();
            let mut v = panic_json(p.as_ref(), Some(&text));
            match hooks.cur {
                Some(s) => {
                    v["ev"] = json!("fail");
                    v["seed"] = json!(s);
                }
                None => v["ev"] = json!("chunk_fail"),
            }
            hooks.out.put(v);
        }
    }
    0
}

// ---------------------------------------------------------------------------- processes (OS)

#[cfg(windows)]
mod os {
    use std::os::windows::io::AsRawHandle;
    use std::process::Child;
    type H = *mut core::ffi::c_void;

    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct FileTime {
        lo: u32,
        hi: u32,
    }
    #[repr(C)]
    struct Pmc {
        cb: u32,
        page_faults: u32,
        peak_working_set: usize,
        working_set: usize,
        rest: [usize; 6],
    }
    #[repr(C)]
    #[derive(Default)]
    struct PowerThrottling {
        version: u32,
        control_mask: u32,
        state_mask: u32,
    }
    extern "system" {
        fn GetProcessTimes(h: H, c: *mut FileTime, e: *mut FileTime, k: *mut FileTime, u: *mut FileTime) -> i32;
        fn GetModuleHandleW(name: *const u16) -> H;
        fn GetProcAddress(m: H, name: *const u8) -> *const core::ffi::c_void;
        fn GetCurrentProcess() -> H;
        fn GetProcessInformation(h: H, class: i32, info: *mut core::ffi::c_void, size: u32) -> i32;
        fn SetProcessInformation(h: H, class: i32, info: *const core::ffi::c_void, size: u32) -> i32;
        fn SetErrorMode(mode: u32) -> u32;
    }
    #[link(name = "psapi")]
    extern "system" {
        fn GetProcessMemoryInfo(h: H, pmc: *mut Pmc, cb: u32) -> i32;
    }
    const PROCESS_POWER_THROTTLING: i32 = 4;

    fn h(c: &Child) -> H {
        c.as_raw_handle() as H
    }

    /// CPU seconds (user + kernel) the process has used.
    pub fn cpu_s(c: &Child) -> Option<f64> {
        let (mut a, mut b, mut k, mut u) = (FileTime::default(), FileTime::default(), FileTime::default(), FileTime::default());
        if unsafe { GetProcessTimes(h(c), &mut a, &mut b, &mut k, &mut u) } == 0 {
            return None;
        }
        let t = |f: FileTime| ((f.hi as u64) << 32 | f.lo as u64) as f64 * 1e-7;
        Some(t(k) + t(u))
    }

    fn ntdll(name: &[u8]) -> Option<unsafe extern "system" fn(H) -> i32> {
        let w: Vec<u16> = "ntdll.dll".encode_utf16().chain([0]).collect();
        unsafe {
            let m = GetModuleHandleW(w.as_ptr());
            if m.is_null() {
                return None;
            }
            let p = GetProcAddress(m, name.as_ptr());
            if p.is_null() {
                return None;
            }
            Some(std::mem::transmute::<*const core::ffi::c_void, unsafe extern "system" fn(H) -> i32>(p))
        }
    }

    pub fn suspend(c: &Child) -> bool {
        ntdll(b"NtSuspendProcess\0").is_some_and(|f| unsafe { f(h(c)) } >= 0)
    }

    pub fn resume(c: &Child) -> bool {
        ntdll(b"NtResumeProcess\0").is_some_and(|f| unsafe { f(h(c)) } >= 0)
    }

    pub fn working_set_mb(c: &Child) -> u64 {
        let mut p: Pmc = unsafe { std::mem::zeroed() };
        p.cb = std::mem::size_of::<Pmc>() as u32;
        if unsafe { GetProcessMemoryInfo(h(c), &mut p, p.cb) } != 0 {
            (p.working_set >> 20) as u64
        } else {
            0
        }
    }

    /// The worker's own EcoQoS state (the agent sets it for CPU work), given to a chunk process too
    /// (priority classes are inherited, power throttling is not). -> whether it was set.
    pub fn copy_power_throttling(c: &Child) -> bool {
        let mut s = PowerThrottling { version: 1, ..Default::default() };
        let size = std::mem::size_of::<PowerThrottling>() as u32;
        unsafe {
            if GetProcessInformation(GetCurrentProcess(), PROCESS_POWER_THROTTLING, &mut s as *mut _ as *mut core::ffi::c_void, size) == 0 {
                return false;
            }
            if s.control_mask == 0 {
                return false;
            }
            SetProcessInformation(h(c), PROCESS_POWER_THROTTLING, &s as *const _ as *const core::ffi::c_void, size) != 0
        }
    }

    pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    /// SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX: a crash ends the process
    /// without an error dialog (what JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION does under the agent).
    pub fn quiet_crashes() {
        unsafe {
            SetErrorMode(0x0001 | 0x0002 | 0x8000);
        }
    }
}

#[cfg(unix)]
mod os {
    use std::process::Child;
    extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
        fn sysconf(name: i32) -> i64;
    }
    #[cfg(target_os = "linux")]
    const STOP_CONT: (i32, i32) = (19, 18);
    #[cfg(not(target_os = "linux"))]
    const STOP_CONT: (i32, i32) = (17, 19);

    pub fn cpu_s(c: &Child) -> Option<f64> {
        // Linux: /proc/<pid>/stat utime + stime (clock ticks); elsewhere unknown (wall time is used)
        let s = std::fs::read_to_string(format!("/proc/{}/stat", c.id())).ok()?;
        let rest = &s[s.rfind(')')? + 2..];
        let f: Vec<&str> = rest.split_whitespace().collect();
        let ticks = f.get(11)?.parse::<f64>().ok()? + f.get(12)?.parse::<f64>().ok()?;
        let hz = unsafe { sysconf(2) } as f64;
        Some(ticks / if hz > 0.0 { hz } else { 100.0 })
    }
    pub fn suspend(c: &Child) -> bool {
        unsafe { kill(c.id() as i32, STOP_CONT.0) == 0 }
    }
    pub fn resume(c: &Child) -> bool {
        unsafe { kill(c.id() as i32, STOP_CONT.1) == 0 }
    }
    pub fn working_set_mb(c: &Child) -> u64 {
        std::fs::read_to_string(format!("/proc/{}/status", c.id())).ok().and_then(|s| s.lines().find(|l| l.starts_with("VmRSS:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())).map_or(0, |kb| kb / 1024)
    }
    pub fn copy_power_throttling(_c: &Child) -> bool {
        false
    }
    pub fn quiet_crashes() {}
}

/// How a process exit code reads in a failure's site (Windows: the DWORD, as Python prints it).
fn exit_code_text(st: &std::process::ExitStatus) -> String {
    #[cfg(windows)]
    {
        st.code().map_or("?".into(), |c| (c as u32).to_string())
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        st.code().map(|c| c.to_string()).or_else(|| st.signal().map(|s| format!("-{s}"))).unwrap_or_else(|| "?".into())
    }
}

// ---------------------------------------------------------------------------- the supervisor

/// The job's control state (the agent's lines), shared with the supervisor.
#[derive(Default)]
pub struct FuzzControl {
    s: Mutex<CtlState>,
    cv: Condvar,
}

#[derive(Default)]
struct CtlState {
    quit: bool,
    finish: bool,
}

impl FuzzControl {
    pub fn quit(&self) {
        self.s.lock().unwrap().quit = true;
        self.cv.notify_all();
    }
    pub fn finish(&self) {
        self.s.lock().unwrap().finish = true;
        self.cv.notify_all();
    }
    fn state(&self) -> (bool, bool) {
        let s = self.s.lock().unwrap();
        (s.quit, s.finish)
    }
    fn wait(&self, d: Duration) {
        let g = self.s.lock().unwrap();
        let _ = self.cv.wait_timeout(g, d);
    }
}

/// Counters for `status` (PROTOCOL.md 3 Stats: `games` = seeds done here).
#[derive(Clone, Debug, Default)]
pub struct FuzzProgress {
    pub seeds: u64,
    pub failures: u64,
    pub running: usize,
    pub threads: usize,
    pub chunks: u64,
    pub yields: u64,
    pub yield_ms_max: f64,
    pub wall_s: f64,
    pub children_rss_mb: u64,
    pub cpu_s: f64,
}

impl FuzzProgress {
    pub fn stats(&self) -> Value {
        json!({
            "games": self.seeds, "seeds": self.seeds, "failures": self.failures, "threads": self.threads, "concurrency": self.running,
            "chunks": self.chunks, "yields": self.yields, "yield_ms_max": (self.yield_ms_max * 10.0).round() / 10.0,
            "rss_mb": crate::job::rss_mb() + self.children_rss_mb, "wall_s": (self.wall_s * 10.0).round() / 10.0,
            "cpu_s": self.cpu_s.round(), "seeds_per_s": if self.wall_s > 0.0 { ((self.seeds as f64 / self.wall_s) * 10.0).round() / 10.0 } else { 0.0 },
            "provider": "none",
        })
    }
}

/// Why a single seed is run alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Confirm {
    Crash,
    Hang,
}

#[derive(Clone, Debug)]
struct Chunk {
    first: u64,
    count: u64,
    confirm: Option<Confirm>,
}

struct Running {
    chunk: Chunk,
    id: u64,
    child: Child,
    stdin: Option<ChildStdin>,
    progress: std::fs::File,
    buf: Vec<u8>,
    log: PathBuf,
    files: Vec<PathBuf>,
    /// The seed being played, the child's CPU seconds and the active wall time when it started.
    cur: Option<(u64, f64, f64)>,
    passed: u64,
    /// Seen in its progress: end (done | stopped at), a failing seed, a chunk failure.
    ended: Option<Value>,
    failed_seed: Option<u64>,
    killed_hang: Option<(u64, f64)>,
    stop_sent: bool,
}

/// Shared between the supervisor and the control thread (yield / resume reach the children).
pub struct FuzzShared {
    running: Mutex<Vec<Running>>,
    yielded: Mutex<Option<Instant>>,
    paused_s: Mutex<f64>,
    pub progress: Mutex<FuzzProgress>,
}

impl Default for FuzzShared {
    fn default() -> Self {
        FuzzShared { running: Mutex::new(Vec::new()), yielded: Mutex::new(None), paused_s: Mutex::new(0.0), progress: Mutex::new(FuzzProgress::default()) }
    }
}

impl FuzzShared {
    /// Suspend every chunk process (and the clocks). -> the milliseconds it took, None if already yielded.
    pub fn do_yield(&self) -> Option<f64> {
        let t0 = Instant::now();
        let r = self.running.lock().unwrap();
        let mut y = self.yielded.lock().unwrap();
        if y.is_some() {
            return None;
        }
        for c in r.iter() {
            os::suspend(&c.child);
        }
        *y = Some(Instant::now());
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let mut p = self.progress.lock().unwrap();
        p.yields += 1;
        p.yield_ms_max = p.yield_ms_max.max(ms);
        Some(ms)
    }
    /// Resume them. -> whether it was yielded.
    pub fn do_resume(&self) -> bool {
        let r = self.running.lock().unwrap();
        let mut y = self.yielded.lock().unwrap();
        let Some(t) = y.take() else { return false };
        for c in r.iter() {
            os::resume(&c.child);
        }
        *self.paused_s.lock().unwrap() += t.elapsed().as_secs_f64();
        true
    }
    fn is_yielded(&self) -> bool {
        self.yielded.lock().unwrap().is_some()
    }
}

/// How a fuzz job ended (`end`'s fields).
pub struct FuzzOutcome {
    pub reason: &'static str,
    pub stats: Value,
    pub unfinished: Ranges,
    pub covered: Ranges,
    pub failures: Vec<Value>,
}

/// The build's identity for failures and results: the manifest next to the executable (signed), else
/// the variables `ZERO_FARM_ENGINE_COMMIT` / `ZERO_FARM_PROFILE`, else 'unknown' and the compiled
/// profile. A manifest whose profile is not the compiled one is an error (a mislabelled build).
pub fn build_identity(game: &dyn FuzzGame) -> Result<(String, String), String> {
    let manifest = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join("manifest.json"))).and_then(|p| std::fs::read_to_string(p).ok()).and_then(|s| serde_json::from_str::<Value>(&s).ok());
    let commit = manifest.as_ref().and_then(|m| m["engine_commit"].as_str().map(String::from)).or_else(|| std::env::var("ZERO_FARM_ENGINE_COMMIT").ok()).unwrap_or_else(|| "unknown".into());
    let profile = manifest.as_ref().and_then(|m| m["profile"].as_str().map(String::from)).or_else(|| std::env::var("ZERO_FARM_PROFILE").ok()).unwrap_or_else(|| game.profile().into());
    if profile != game.profile() {
        return Err(format!("this binary was built with the cargo profile {:?}, its manifest says {profile:?}", game.profile()));
    }
    Ok((commit, profile))
}

fn tail_of(path: &Path, max_bytes: u64, max_lines: usize) -> String {
    let Ok(mut f) = std::fs::File::open(path) else { return String::new() };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let _ = f.seek(SeekFrom::Start(len.saturating_sub(max_bytes)));
    let mut b = Vec::new();
    let _ = f.read_to_end(&mut b);
    let s = String::from_utf8_lossy(&b);
    let lines: Vec<&str> = s.lines().collect();
    lines[lines.len().saturating_sub(max_lines)..].join("\n")
}

struct Sup<'a> {
    game: &'a dyn FuzzGame,
    plan: FuzzPlan,
    job: &'a FuzzJob,
    sink: &'a dyn Sink,
    shared: &'a FuzzShared,
    commit: String,
    profile: String,
    queue: VecDeque<Chunk>,
    done: Ranges,
    failures: Vec<Value>,
    next_id: u64,
    unreported: u64,
    startup_crashes: u32,
    t0: Instant,
}

impl Sup<'_> {
    /// Wall seconds since the job started, without the time it was yielded.
    fn active_s(&self) -> f64 {
        // (one lock at a time: `do_resume` takes `yielded` then `paused_s`)
        let paused = *self.shared.paused_s.lock().unwrap();
        let now = self.shared.yielded.lock().unwrap().map_or(0.0, |t| t.elapsed().as_secs_f64());
        self.t0.elapsed().as_secs_f64() - paused - now
    }

    fn repro(&self, seed: u64) -> String {
        self.game.repro(&self.plan, &self.job.env_with_faults(), seed, &self.commit, &self.profile)
    }

    fn failure(&mut self, seed: u64, site: String, panic: String, repro: String, log: &Path) {
        let cap = |s: String, n: usize| if s.len() > n { let mut i = n; while !s.is_char_boundary(i) { i -= 1; } s[..i].to_string() } else { s };
        let f = json!({"seed": seed, "panic": cap(panic, 64 * 1024), "site": site, "repro": cap(repro, 32 * 1024), "output_tail": tail_of(log, 32 * 1024, 200)});
        let mut ev = f.clone();
        ev["ev"] = json!("failure");
        ev["mode"] = json!(self.job.mode);
        self.sink.event(ev);
        self.failures.push(f);
        self.shared.progress.lock().unwrap().failures += 1;
    }

    fn pass(&mut self, seed: u64) {
        self.done.add(seed, seed + 1);
        self.unreported += 1;
    }

    fn spawn(&mut self, chunk: Chunk) -> Result<Running, String> {
        let id = self.next_id;
        self.next_id += 1;
        let d = &self.job.work_dir;
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        let (spec, progress, log) = (d.join(format!("chunk-{id}.json")), d.join(format!("chunk-{id}.progress")), d.join(format!("chunk-{id}.log")));
        let v = json!({"mode": self.job.mode, "env": self.job.env.iter().map(|(k, v)| json!([k, v])).collect::<Vec<_>>(),
                       "first": chunk.first, "count": chunk.count, "progress": progress.display().to_string(),
                       "faults": if self.job.fault.is_empty() { Value::Null } else { json!(faults_text(&self.job.fault)) }});
        std::fs::write(&spec, v.to_string()).map_err(|e| format!("{}: {e}", spec.display()))?;
        std::fs::write(&progress, b"").map_err(|e| format!("{}: {e}", progress.display()))?;
        let logf = std::fs::File::create(&log).map_err(|e| format!("{}: {e}", log.display()))?;
        let exe = std::env::current_exe().map_err(|e| format!("the worker's executable: {e}"))?;
        let mut cmd = Command::new(exe);
        cmd.arg("fuzz-chunk").arg(&spec).current_dir(d).stdin(Stdio::piped()).stdout(logf.try_clone().map_err(|e| e.to_string())?).stderr(logf).env("RUST_BACKTRACE", "0");
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(os::CREATE_NO_WINDOW);
        }
        let mut child = cmd.spawn().map_err(|e| format!("start a chunk process: {e}"))?;
        os::copy_power_throttling(&child);
        let stdin = child.stdin.take();
        let pf = std::fs::File::open(&progress).map_err(|e| format!("{}: {e}", progress.display()))?;
        self.shared.progress.lock().unwrap().chunks += 1;
        Ok(Running { chunk, id, child, stdin, progress: pf, buf: Vec::new(), log: log.clone(), files: vec![spec, progress, log], cur: None, passed: 0, ended: None, failed_seed: None, killed_hang: None, stop_sent: false })
    }

    /// New progress lines of `r`.
    fn read_progress(&mut self, r: &mut Running) {
        let mut b = Vec::new();
        let _ = r.progress.read_to_end(&mut b);
        r.buf.extend_from_slice(&b);
        while let Some(i) = r.buf.iter().position(|&c| c == b'\n') {
            let line: Vec<u8> = r.buf.drain(..=i).collect();
            let Ok(v) = serde_json::from_slice::<Value>(&line) else { continue };
            self.on_line(r, v);
        }
    }

    fn on_line(&mut self, r: &mut Running, v: Value) {
        let seed = v["seed"].as_u64();
        match v["ev"].as_str().unwrap_or("") {
            "start" => {
                if let Some(s) = seed {
                    let cpu = os::cpu_s(&r.child).unwrap_or(0.0);
                    r.cur = Some((s, cpu, self.active_s()));
                }
            }
            "pass" => {
                if let Some(s) = seed {
                    self.pass(s);
                    r.passed += 1;
                    r.cur = None;
                }
            }
            "side" => {
                if let Some(s) = seed {
                    let repro = self.repro(s) + v["repro_suffix"].as_str().unwrap_or("");
                    self.failure(s, v["site"].as_str().unwrap_or("unknown").into(), v["panic"].as_str().unwrap_or("").into(), repro, &r.log.clone());
                }
            }
            "fail" => {
                if let Some(s) = seed {
                    let repro = self.repro(s);
                    self.failure(s, v["site"].as_str().unwrap_or("unknown").into(), v["panic"].as_str().unwrap_or("").into(), repro, &r.log.clone());
                    self.pass(s);
                    r.failed_seed = Some(s);
                    r.cur = None;
                }
            }
            "chunk_fail" => {
                let c = &r.chunk;
                let repro = format!("{} (the chunk: {}={})", self.repro(c.first), self.plan.count_var, c.count);
                self.failure(c.first, v["site"].as_str().unwrap_or("unknown").into(), v["panic"].as_str().unwrap_or("").into(), repro, &r.log.clone());
                r.ended = Some(json!({"how": "chunk_fail"}));
            }
            "end" => r.ended = Some(v),
            _ => {}
        }
    }

    /// `r` exited: what its chunk leaves to do.
    fn on_exit(&mut self, mut r: Running, st: std::process::ExitStatus, finishing: bool) -> Result<(), String> {
        self.read_progress(&mut r);
        self.shared.progress.lock().unwrap().cpu_s += os::cpu_s(&r.child).unwrap_or(0.0);
        let end = r.chunk.first + r.chunk.count;
        let mut keep_files = false;
        let rest_from = |s: u64| Chunk { first: s, count: end - s, confirm: None };
        if let Some(s) = r.failed_seed {
            // the chunk stopped at its failing seed: the rest in a new process
            keep_files = true;
            if s + 1 < end && !finishing {
                self.queue.push_front(rest_from(s + 1));
            }
        } else if let Some(e) = r.ended.clone() {
            match e["how"].as_str() {
                Some("done") => {
                    if r.passed != r.chunk.count {
                        return Err(format!("chunk [{}, {}) ended 'done' after {} of its {} seeds: the loop left seeds out (a mode whose loop plays nothing?)", r.chunk.first, end, r.passed, r.chunk.count));
                    }
                }
                Some("stopped") => {
                    let at = e["at"].as_u64().unwrap_or(end);
                    if !finishing && at < end {
                        self.queue.push_front(rest_from(at));
                    }
                }
                _ => keep_files = true, // chunk_fail: reported
            }
            self.startup_crashes = 0;
        } else {
            // no end: it crashed or was killed for a hang
            keep_files = true;
            let code = exit_code_text(&st);
            match (r.killed_hang, r.cur) {
                (Some((s, cpu)), _) => {
                    if r.chunk.confirm.is_some() || r.chunk.count == 1 {
                        let limit = self.job.seed_timeout_s;
                        self.failure(s, format!("HANG {}", self.job.mode), format!("no exit after {cpu:.0} s of running time (CPU; the limit is {limit:.0} s a seed): seeds {s}..{s}"), self.repro(s), &r.log.clone());
                        self.pass(s);
                        if s + 1 < end && !finishing {
                            self.queue.push_front(rest_from(s + 1));
                        }
                    } else if !finishing {
                        if s + 1 < end {
                            self.queue.push_front(rest_from(s + 1));
                        }
                        self.queue.push_front(Chunk { first: s, count: 1, confirm: Some(Confirm::Hang) });
                    }
                }
                (None, Some((s, _, _))) => {
                    if r.chunk.confirm.is_some() || r.chunk.count == 1 {
                        let lines: Vec<String> = tail_of(&r.log, 16 * 1024, 400).lines().filter(|x| !x.trim().is_empty()).map(String::from).collect();
                        let tail = lines[lines.len().saturating_sub(6)..].join("\n");
                        self.failure(s, format!("crash exit {code}"), tail, self.repro(s), &r.log.clone());
                        self.pass(s);
                        if s + 1 < end && !finishing {
                            self.queue.push_front(rest_from(s + 1));
                        }
                    } else if !finishing {
                        // cut at the seed it was playing, which runs alone
                        if s + 1 < end {
                            self.queue.push_front(rest_from(s + 1));
                        }
                        self.queue.push_front(Chunk { first: s, count: 1, confirm: Some(Confirm::Crash) });
                    }
                    self.sink.log(&format!("chunk {} [{}, {}) died (exit {code}) at seed {s}", r.id, r.chunk.first, end));
                }
                (None, None) if r.passed == r.chunk.count && r.chunk.count > 0 => {
                    // every seed passed, then it died in the chunk's own checks
                    let lines: Vec<String> = tail_of(&r.log, 16 * 1024, 400).lines().filter(|x| !x.trim().is_empty()).map(String::from).collect();
                    let tail = lines[lines.len().saturating_sub(6)..].join("\n");
                    let repro = format!("{} (the chunk: {}={})", self.repro(r.chunk.first), self.plan.count_var, r.chunk.count);
                    self.failure(r.chunk.first, format!("crash exit {code}"), tail, repro, &r.log.clone());
                }
                (None, None) => {
                    // died before its first seed (or between seeds): try again, but not forever
                    self.startup_crashes += 1;
                    let at = r.chunk.first + r.passed;
                    if self.startup_crashes >= 3 {
                        return Err(format!("chunk processes die before playing a seed (exit {code}); the last one's log:\n{}", tail_of(&r.log, 8 * 1024, 60)));
                    }
                    if !finishing && at < end {
                        self.queue.push_front(Chunk { first: at, count: end - at, confirm: r.chunk.confirm });
                    }
                }
            }
        }
        if !keep_files && !self.job.keep_chunks {
            for f in &r.files {
                let _ = std::fs::remove_file(f);
            }
        }
        Ok(())
    }
}

impl FuzzJob {
    fn env_with_faults(&self) -> Vec<(String, String)> {
        let mut e = self.env.clone();
        if !self.fault.is_empty() {
            e.push((FAULT_KEY.into(), faults_text(&self.fault)));
        }
        e
    }
}

/// Run a fuzz job to its end (or `quit`).
pub fn run_fuzz(game: &dyn FuzzGame, job: &FuzzJob, ctl: &FuzzControl, sink: &dyn Sink, shared: &FuzzShared) -> Result<FuzzOutcome, String> {
    let plan = game.plan(&job.mode, &job.env)?;
    let (commit, profile) = build_identity(game)?;
    let threads = job.threads.max(1);
    let end = job.first.checked_add(job.count).ok_or("first + count overflows")?;
    let size = job.chunk.unwrap_or_else(|| plan.min_chunk.max(job.count.div_ceil(4 * threads as u64))).max(1);
    let mut queue = VecDeque::new();
    let mut at = job.first;
    while at < end {
        let n = size.min(end - at);
        queue.push_back(Chunk { first: at, count: n, confirm: None });
        at += n;
    }
    shared.progress.lock().unwrap().threads = threads;
    sink.log(&format!("fuzz {} [{}, {}): {} chunk(s) of {} seed(s), {} at a time; commit {}, profile {}", job.mode, job.first, end, queue.len(), size, threads, &commit, &profile));
    let mut s = Sup { game, plan, job, sink, shared, commit, profile, queue, done: Ranges::default(), failures: Vec::new(), next_id: 0, unreported: 0, startup_crashes: 0, t0: Instant::now() };
    let mut last_fin = Instant::now();
    let mut reason = "done";
    loop {
        let (quit, finish) = ctl.state();
        if quit {
            let mut r = shared.running.lock().unwrap();
            for c in r.iter_mut() {
                let _ = c.child.kill();
                let _ = c.child.wait();
            }
            // (the seeds that passed before the kill count; the rest is unfinished)
            let gone: Vec<Running> = r.drain(..).collect();
            drop(r);
            for mut c in gone {
                s.read_progress(&mut c);
            }
            reason = "quit";
            break;
        }
        if finish {
            reason = "finish";
        }
        if !shared.is_yielded() {
            let mut r = shared.running.lock().unwrap();
            let mut i = 0;
            let mut exited = Vec::new();
            while i < r.len() {
                let mut c = r.remove(i);
                s.read_progress(&mut c);
                if finish && !c.stop_sent {
                    if let Some(w) = c.stdin.as_mut() {
                        let _ = w.write_all(b"stop\n");
                        let _ = w.flush();
                    }
                    c.stop_sent = true;
                }
                // a seed past its CPU limit is a hang (CPU time: a starved PC is not hanging)
                if let (Some((seed, cpu0, wall0)), None) = (c.cur, c.killed_hang) {
                    let used = os::cpu_s(&c.child).map(|x| x - cpu0).unwrap_or_else(|| s.active_s() - wall0);
                    if used > job.seed_timeout_s {
                        sink.log(&format!("chunk {}: seed {seed} ran {used:.0} s (limit {:.0} s): killing it (a hang)", c.id, job.seed_timeout_s));
                        let _ = c.child.kill();
                        c.killed_hang = Some((seed, used));
                    }
                }
                match c.child.try_wait() {
                    Ok(Some(st)) => exited.push((c, st)),
                    _ => {
                        r.insert(i, c);
                        i += 1;
                    }
                }
            }
            drop(r);
            for (c, st) in exited {
                s.on_exit(c, st, finish)?;
            }
            let mut r = shared.running.lock().unwrap();
            while r.len() < threads && !finish {
                let Some(ch) = s.queue.pop_front() else { break };
                let c = s.spawn(ch)?;
                r.push(c);
            }
            let mut p = shared.progress.lock().unwrap();
            p.running = r.len();
            p.children_rss_mb = r.iter().map(|c| os::working_set_mb(&c.child)).sum();
            drop(p);
            let idle = r.is_empty();
            drop(r);
            if idle && (s.queue.is_empty() || finish) {
                break;
            }
        }
        {
            let mut p = shared.progress.lock().unwrap();
            p.seeds = s.done.total();
            p.wall_s = s.active_s();
        }
        if s.unreported > 0 && last_fin.elapsed() >= Duration::from_secs(1) {
            sink.event(json!({"ev": "fin", "n": s.unreported}));
            s.unreported = 0;
            last_fin = Instant::now();
        }
        ctl.wait(Duration::from_millis(50));
    }
    if s.unreported > 0 {
        sink.event(json!({"ev": "fin", "n": s.unreported}));
    }
    let unfinished = s.done.missing(job.first, end);
    {
        let mut p = shared.progress.lock().unwrap();
        p.seeds = s.done.total();
        p.wall_s = s.active_s();
        p.running = 0;
    }
    let mut stats = shared.progress.lock().unwrap().stats();
    stats["peak_rss_mb"] = json!(crate::job::peak_rss_mb());
    stats["seconds"] = json!((s.t0.elapsed().as_secs_f64() * 10.0).round() / 10.0);
    if reason == "done" && !unfinished.0.is_empty() {
        return Err(format!("the job ended with seeds unplayed: {:?}", unfinished.0));
    }
    // the 5.1 summary
    let covered = s.done.clone();
    let summary = json!({"campaign": job.campaign, "mode": job.mode, "first": job.first, "count": job.count, "done": covered.total(),
                         "failures": s.failures, "build_id": crate::proto::build_id(), "engine_commit": s.commit, "final": true,
                         "seeds": covered.json(), "profile": s.profile});
    std::fs::create_dir_all(&job.out_dir).map_err(|e| format!("{}: {e}", job.out_dir.display()))?;
    let path = job.out_dir.join("fuzz-final.json");
    let bytes = serde_json::to_vec(&summary).map_err(|e| e.to_string())?;
    std::fs::write(&path, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    if job.announce_result && !covered.0.is_empty() {
        sink.event(json!({"ev": "records", "seq": 0, "path": "fuzz-final.json", "games": covered.total(), "rows": 0, "bytes": bytes.len(), "gids": covered.json()}));
    }
    Ok(FuzzOutcome { reason, stats, unfinished, covered, failures: s.failures })
}

/// A fuzz job from its `job` line (PROTOCOL.md 5): the env object's order kept (from the raw line).
pub fn job_from_value(v: &Value, base: &Path) -> Result<FuzzJob, String> {
    let o = v.as_object().ok_or("job: a JSON object")?;
    let env = env_pairs(o, v)?;
    let (env, fault) = split_env(env)?;
    let u = |k: &str| o.get(k).and_then(|x| x.as_u64());
    let lease_id = o.get("lease_id").and_then(|x| x.as_str()).unwrap_or("job").to_string();
    Ok(FuzzJob {
        campaign: o.get("campaign").and_then(|x| x.as_str()).map(String::from),
        mode: o.get("mode").and_then(|x| x.as_str()).ok_or("job: mode")?.to_string(),
        env,
        first: u("first").ok_or("job: first")?,
        count: u("count").ok_or("job: count")?,
        threads: u("threads").unwrap_or(1).max(1) as usize,
        chunk: u("chunk").filter(|&c| c > 0),
        seed_timeout_s: o.get("seed_timeout_s").and_then(|x| x.as_f64()).filter(|&t| t > 0.0).unwrap_or(300.0),
        out_dir: base.join(o.get("out_dir").and_then(|x| x.as_str()).unwrap_or("out")),
        work_dir: base.join("fuzz"),
        announce_result: o.get("announce_result").and_then(|x| x.as_bool()).unwrap_or(false),
        keep_chunks: false,
        fault,
        lease_id,
    })
}

/// The job's `env` as (key, value) pairs in the line's order; values must be strings.
fn env_pairs(o: &Map<String, Value>, v: &Value) -> Result<Vec<(String, String)>, String> {
    let raw_env = match v.get("_raw").and_then(|r| r.as_str()) {
        Some(raw) => match crate::pyjson::PyJson::parse(raw)?.get("env") {
            Some(crate::pyjson::PyJson::Dict(d)) => Some(d.clone()),
            Some(crate::pyjson::PyJson::Null) | None => None,
            Some(other) => return Err(format!("job: env must be an object, not {}", other.dumps())),
        },
        None => None,
    };
    if let Some(d) = raw_env {
        return d.into_iter().map(|(k, x)| match x {
            crate::pyjson::PyJson::Str(s) => Ok((k, s)),
            other => Err(format!("job: env {k}: a string, not {}", other.dumps())),
        }).collect();
    }
    match o.get("env") {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Object(m)) => m.iter().map(|(k, x)| x.as_str().map(|s| (k.clone(), s.to_string())).ok_or_else(|| format!("job: env {k}: a string, not {x}"))).collect(),
        Some(other) => Err(format!("job: env must be an object, not {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_add_and_missing() {
        let mut r = Ranges::default();
        r.add(5, 6);
        r.add(1, 3);
        r.add(3, 5);
        r.add(10, 12);
        assert_eq!(r.0, vec![(1, 6), (10, 12)]);
        r.add(6, 10);
        assert_eq!(r.0, vec![(1, 12)]);
        assert_eq!(r.missing(0, 15).0, vec![(0, 1), (12, 15)]);
        assert_eq!(Ranges(vec![(2, 4), (6, 8)]).missing(3, 7).0, vec![(4, 6)]);
        assert_eq!(r.total(), 11);
    }

    #[test]
    fn faults_parse_and_are_refused_without_the_switch() {
        assert_eq!(parse_faults("panic:5, abort:7,hang:9,overflow:11").unwrap(), vec![Fault::Panic(5), Fault::Abort(7), Fault::Hang(9), Fault::Overflow(11)]);
        assert!(parse_faults("explode:3").is_err());
        if std::env::var("ZERO_FARM_ALLOW_TEST_FAULTS").is_err() {
            assert!(split_env(vec![(FAULT_KEY.into(), "panic:1".into())]).is_err());
        }
        assert_eq!(split_env(vec![("FUZZ_POOL".into(), "eoe".into())]).unwrap().0, vec![("FUZZ_POOL".to_string(), "eoe".to_string())]);
    }
}
