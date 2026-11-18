//! The platform a worker build runs on (zero-farm PROTOCOL.md 11) and what it asks of its OS there (docs/az.md 62).
//!
//! * [`PLATFORM`]: the build's platform name as manifests write it (`windows-x64`, `macos-arm64`; others are named
//!   but have no farm builds). `info` announces it.
//! * [`REFERENCE`]: the platform whose bits are the reference. What the engine alone decides (a game from its seed and
//!   its moves, a fuzz seed) is identical on every platform, except where it calls the platform's libm (`f64::exp`,
//!   `powf`, `ln`, ...: their last bits differ between the C runtimes). A game declares such settings
//!   (`SearchGame::libm_replay`, `fuzz::FuzzGame::libm_dependence`) and a worker off the reference platform refuses
//!   them (`check-config`, `fuzz-check`, every job): its records would be refused by the trainer's replay, its fuzz
//!   failures would not reproduce on the desktop.
//! * [`ORT_LIB`]: the file name of ONNX Runtime's library in a runtime pack on this platform.
//! * macOS: a process's CPU time, physical footprint and its peak (`proc_pid_rusage`: what the macOS agent's memory
//!   cap reads, PROTOCOL.md 11.5) and the path a loaded library came from (dyld's image list).

pub const WINDOWS_X64: &str = "windows-x64";
pub const MACOS_ARM64: &str = "macos-arm64";

/// The reference platform (module docs).
pub const REFERENCE: &str = WINDOWS_X64;

/// This build's platform (module docs).
pub const PLATFORM: &str = if cfg!(all(windows, target_arch = "x86_64")) {
    WINDOWS_X64
} else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
    MACOS_ARM64
} else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
    "macos-x64"
} else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
    "linux-x64"
} else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
    "linux-arm64"
} else {
    "other"
};

/// ONNX Runtime's library in a runtime pack (the name `proto::ort_dylib_in` loads from a `runtime_dir`).
pub const ORT_LIB: &str = if cfg!(windows) {
    "onnxruntime.dll"
} else if cfg!(target_os = "macos") {
    "libonnxruntime.dylib"
} else {
    "libonnxruntime.so"
};

/// The libraries `info` and `probe-providers` say the origin of: ONNX Runtime's and, on Windows, the ones a runtime
/// pack preloads (`session::PRELOAD`).
pub fn ort_modules() -> Vec<&'static str> {
    if cfg!(windows) {
        std::iter::once(ORT_LIB).chain(crate::session::PRELOAD.iter().copied()).collect()
    } else {
        vec![ORT_LIB]
    }
}

/// A known platform name of zero-farm (`--platform` arguments).
pub fn check_name(p: &str) -> Result<&str, String> {
    match p {
        WINDOWS_X64 | MACOS_ARM64 => Ok(p),
        _ => Err(format!("platform {p:?}: one of {WINDOWS_X64}, {MACOS_ARM64}")),
    }
}

/// Is a worker on platform `p` held to the reference's bits by refusing libm-dependent settings (module docs)?
pub fn off_reference(p: &str) -> bool {
    p != REFERENCE
}

#[cfg(target_os = "macos")]
pub mod darwin {
    //! libproc and dyld through libSystem (no bindings crate).
    use core::ffi::{c_char, c_int, c_void};

    #[repr(C)]
    struct Timebase {
        numer: u32,
        denom: u32,
    }
    extern "C" {
        fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *mut c_void) -> c_int;
        fn mach_timebase_info(info: *mut Timebase) -> c_int;
        fn _dyld_image_count() -> u32;
        fn _dyld_get_image_name(i: u32) -> *const c_char;
    }
    /// `RUSAGE_INFO_V4` (<sys/resource.h>): `struct rusage_info_v4`, 36 64-bit fields after a 16-byte uuid.
    const RUSAGE_INFO_V4: c_int = 4;
    // (indices in u64 units: the uuid is the first two)
    const USER_TIME: usize = 2;
    const SYSTEM_TIME: usize = 3;
    const PHYS_FOOTPRINT: usize = 9;
    const LIFETIME_MAX_PHYS_FOOTPRINT: usize = 30;

    /// A process's resource usage.
    #[derive(Clone, Copy, Debug)]
    pub struct Usage {
        /// CPU seconds, user + system.
        pub cpu_s: f64,
        /// The physical footprint (bytes): what Activity Monitor's "Memory" shows and the agent's cap reads.
        pub footprint: u64,
        /// The largest footprint the process had (bytes).
        pub max_footprint: u64,
    }

    /// `pid`'s usage (a live or not yet reaped process), None if it can't be read. The times of `rusage_info` are
    /// in mach absolute time units (nanoseconds on Intel, 125/3 ns per tick on Apple silicon): converted with
    /// `mach_timebase_info`.
    pub fn usage(pid: u32) -> Option<Usage> {
        let mut b = [0u64; 64];
        if unsafe { proc_pid_rusage(pid as c_int, RUSAGE_INFO_V4, b.as_mut_ptr().cast()) } != 0 {
            return None;
        }
        let mut tb = Timebase { numer: 0, denom: 0 };
        if unsafe { mach_timebase_info(&mut tb) } != 0 || tb.denom == 0 {
            return None;
        }
        let ticks = b[USER_TIME].saturating_add(b[SYSTEM_TIME]);
        let ns = ticks as f64 * tb.numer as f64 / tb.denom as f64;
        Some(Usage { cpu_s: ns * 1e-9, footprint: b[PHYS_FOOTPRINT], max_footprint: b[LIFETIME_MAX_PHYS_FOOTPRINT] })
    }

    /// The path of a loaded image whose file name is `name` (dyld's list, in load order), else None.
    pub fn image_path(name: &str) -> Option<String> {
        let n = unsafe { _dyld_image_count() };
        for i in 0..n {
            let p = unsafe { _dyld_get_image_name(i) };
            if p.is_null() {
                continue;
            }
            let s = unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().to_string();
            if std::path::Path::new(&s).file_name().is_some_and(|f| f == name) {
                return Some(s);
            }
        }
        None
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn own_usage_counts_cpu_time_and_memory() {
            let u0 = usage(std::process::id()).expect("our own usage");
            // a CPU-bound half second, and 64 MB touched
            let t = std::time::Instant::now();
            let mut x = 0u64;
            while t.elapsed().as_secs_f64() < 0.5 {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            }
            let v = vec![x as u8; 64 << 20];
            let u1 = usage(std::process::id()).unwrap();
            let used = u1.cpu_s - u0.cpu_s;
            // (seconds, not mach ticks: 0.5 s of work reads as 0.4..0.7 s, never 12 s)
            assert!((0.4..0.7).contains(&used), "cpu {used} s for 0.5 s of work");
            assert!(u1.footprint >= u0.footprint + (60 << 20), "footprint {} -> {}", u0.footprint, u1.footprint);
            assert!(u1.max_footprint >= u1.footprint, "peak {} < now {}", u1.max_footprint, u1.footprint);
            assert_eq!(v[1 << 20], x as u8);
        }

        #[test]
        fn a_loaded_image_has_its_path() {
            let p = image_path("libSystem.B.dylib").expect("libSystem is loaded");
            assert!(p.ends_with("/libSystem.B.dylib"), "{p}");
            assert!(image_path("no-such-library.dylib").is_none());
        }
    }
}
