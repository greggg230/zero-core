//! Windows ML's execution-provider catalog through its flat C API (`WinMLEpCatalog.h`, NuGet
//! `Microsoft.Windows.AI.MachineLearning` 2.x): the GPU vendors' ONNX Runtime providers that Windows 11 24H2+
//! downloads and shares system-wide (NVIDIA TensorRT-RTX, AMD GPU / MIGraphX, Intel OpenVINO), usable from an
//! unpackaged per-user process with no Windows App SDK runtime and no WinRT registration: the runtime pack holds
//! `Microsoft.Windows.AI.MachineLearning.dll` next to Windows ML's `onnxruntime.dll` (which it imports, so that
//! copy must already be loaded) and `DirectML.dll`. On Windows 10 and older Windows 11 builds the catalog lists
//! nothing (or fails): DirectML and the CPU, built into Windows ML's ONNX Runtime, remain.
//!
//! A provider is used only if Windows ML says it is certified; `ensure_ready` (the probe, never a job) lets
//! Windows ML download and install it (`WinMLEpEnsureReady`, synchronous).
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// The catalog DLL's file name in a runtime pack.
pub const DLL: &str = "Microsoft.Windows.AI.MachineLearning.dll";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ready {
    Ready,
    NotReady,
    NotPresent,
    Other(i32),
}

#[derive(Clone, Debug)]
pub struct CatalogEp {
    pub name: String,
    pub version: String,
    pub package_family: String,
    pub library_path: String,
    pub ready: Ready,
    /// WinMLEpCertification_Certified
    pub certified: bool,
    /// What `ensure_ready` did (None: not asked).
    pub ensure: Option<String>,
}

impl CatalogEp {
    pub fn to_json(&self) -> Value {
        // (library_path: a path under the system's package store; the agent redacts paths before upload)
        json!({"name": self.name, "version": self.version, "package_family": self.package_family,
               "ready": format!("{:?}", self.ready), "certified": self.certified, "ensure": self.ensure,
               "library": Path::new(&self.library_path).file_name().map(|f| f.to_string_lossy().to_string())})
    }
}

/// The catalog's providers, `ensure` naming the ones to make ready (download / install) first.
pub fn list(runtime_dir: &Path, ensure: &dyn Fn(&str) -> bool) -> Result<Vec<CatalogEp>, String> {
    #[cfg(windows)]
    {
        win::list(&runtime_dir.join(DLL), ensure)
    }
    #[cfg(not(windows))]
    {
        let _ = (runtime_dir, ensure);
        Err("Windows ML is Windows only".into())
    }
}

/// Whether a runtime pack carries the catalog.
pub fn present(runtime_dir: &Path) -> bool {
    runtime_dir.join(DLL).is_file()
}

pub fn library(ep: &CatalogEp) -> PathBuf {
    PathBuf::from(&ep.library_path)
}

#[cfg(windows)]
mod win {
    use super::{CatalogEp, Ready};
    use core::ffi::{c_char, c_void};
    use std::path::Path;

    type Hr = i32;
    type Catalog = *mut c_void;
    type Ep = *mut c_void;

    #[repr(C)]
    struct Info {
        name: *const c_char,
        version: *const c_char,
        package_family_name: *const c_char,
        library_path: *const c_char,
        package_root_path: *const c_char,
        ready_state: i32,
        certification: i32,
    }

    type EnumCb = unsafe extern "system" fn(Ep, *const Info, *mut c_void) -> i32;

    extern "system" {
        fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
        fn GetProcAddress(h: *mut c_void, name: *const u8) -> *const c_void;
        fn GetLastError() -> u32;
    }
    const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x8;

    struct Api {
        create: unsafe extern "system" fn(*mut Catalog) -> Hr,
        release: unsafe extern "system" fn(Catalog),
        enum_providers: unsafe extern "system" fn(Catalog, EnumCb, *mut c_void) -> Hr,
        ensure_ready: unsafe extern "system" fn(Ep) -> Hr,
        get_ready: unsafe extern "system" fn(Ep, *mut i32) -> Hr,
        lib_size: unsafe extern "system" fn(Ep, *mut usize) -> Hr,
        lib_path: unsafe extern "system" fn(Ep, usize, *mut c_char, *mut usize) -> Hr,
    }

    fn cstr(p: *const c_char) -> String {
        if p.is_null() {
            return String::new();
        }
        unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().to_string()
    }

    fn ready(s: i32) -> Ready {
        match s {
            0 => Ready::Ready,
            1 => Ready::NotReady,
            2 => Ready::NotPresent,
            x => Ready::Other(x),
        }
    }

    unsafe fn api(dll: &Path) -> Result<Api, String> {
        let w: Vec<u16> = dll.as_os_str().to_string_lossy().encode_utf16().chain([0]).collect();
        let h = LoadLibraryExW(w.as_ptr(), std::ptr::null_mut(), LOAD_WITH_ALTERED_SEARCH_PATH);
        if h.is_null() {
            return Err(format!("{}: LoadLibrary failed (error {})", dll.display(), GetLastError()));
        }
        let f = |n: &[u8]| -> Result<*mut c_void, String> {
            let p = GetProcAddress(h, n.as_ptr()) as *mut c_void;
            if p.is_null() {
                Err(format!("{}: no {}", dll.display(), String::from_utf8_lossy(&n[..n.len() - 1])))
            } else {
                Ok(p)
            }
        };
        Ok(Api {
            create: std::mem::transmute(f(b"WinMLEpCatalogCreate\0")?),
            release: std::mem::transmute(f(b"WinMLEpCatalogRelease\0")?),
            enum_providers: std::mem::transmute(f(b"WinMLEpCatalogEnumProviders\0")?),
            ensure_ready: std::mem::transmute(f(b"WinMLEpEnsureReady\0")?),
            get_ready: std::mem::transmute(f(b"WinMLEpGetReadyState\0")?),
            lib_size: std::mem::transmute(f(b"WinMLEpGetLibraryPathSize\0")?),
            lib_path: std::mem::transmute(f(b"WinMLEpGetLibraryPath\0")?),
        })
    }

    struct Ctx<'a> {
        api: &'a Api,
        ensure: &'a dyn Fn(&str) -> bool,
        out: Vec<CatalogEp>,
    }

    unsafe extern "system" fn on_ep(ep: Ep, info: *const Info, ctx: *mut c_void) -> i32 {
        let ctx = &mut *(ctx as *mut Ctx);
        let i = &*info;
        let mut e = CatalogEp {
            name: cstr(i.name),
            version: cstr(i.version),
            package_family: cstr(i.package_family_name),
            library_path: cstr(i.library_path),
            ready: ready(i.ready_state),
            certified: i.certification == 1,
            ensure: None,
        };
        if e.ready != Ready::Ready && (ctx.ensure)(&e.name) {
            let hr = (ctx.api.ensure_ready)(ep);
            e.ensure = Some(if hr >= 0 { "ok".into() } else { format!("hresult 0x{:08x}", hr as u32) });
            let mut s = -1;
            if (ctx.api.get_ready)(ep, &mut s) >= 0 {
                e.ready = ready(s);
            }
            let mut n = 0usize;
            if (ctx.api.lib_size)(ep, &mut n) >= 0 && n > 0 {
                let mut buf = vec![0u8; n + 1];
                let mut used = 0usize;
                if (ctx.api.lib_path)(ep, buf.len(), buf.as_mut_ptr() as *mut c_char, &mut used) >= 0 {
                    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
                    e.library_path = String::from_utf8_lossy(&buf[..end]).to_string();
                }
            }
        }
        ctx.out.push(e);
        1 // continue
    }

    pub fn list(dll: &Path, ensure: &dyn Fn(&str) -> bool) -> Result<Vec<CatalogEp>, String> {
        unsafe {
            let api = api(dll)?;
            let mut cat: Catalog = std::ptr::null_mut();
            let hr = (api.create)(&mut cat);
            if hr < 0 || cat.is_null() {
                return Err(format!("WinMLEpCatalogCreate: hresult 0x{:08x}", hr as u32));
            }
            let mut ctx = Ctx { api: &api, ensure, out: Vec::new() };
            let hr = (api.enum_providers)(cat, on_ep, &mut ctx as *mut Ctx as *mut c_void);
            (api.release)(cat);
            if hr < 0 {
                return Err(format!("WinMLEpCatalogEnumProviders: hresult 0x{:08x}", hr as u32));
            }
            Ok(ctx.out)
        }
    }
}
