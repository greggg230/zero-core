//! The PC's graphics adapters as DXGI lists them (Windows; elsewhere none): the adapter LUID, which the
//! agent's profile, the job line's `gpu` and ONNX Runtime's device metadata all name an adapter by
//! (zero-farm PROTOCOL.md 8), its vendor, name and memory, and its position in DXGI's order (DirectML's
//! `device_id`). Adapters DXGI lists twice (one card, one LUID) are listed once; software adapters and the
//! Microsoft Basic Render Driver are left out.
use serde_json::{json, Value};

/// One hardware graphics adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Adapter {
    /// `(HighPart << 32) | LowPart` of the adapter's LUID.
    pub luid: u64,
    pub vendor_id: u32,
    pub device_id: u32,
    pub name: String,
    pub vram_mb: u64,
    /// Its index in `IDXGIFactory1::EnumAdapters1` (the first index when DXGI lists it more than once):
    /// DirectML's `device_id`.
    pub dxgi_index: u32,
}

/// The vendor's short name (the agent profile's `vendor`).
pub fn vendor_name(vendor_id: u32) -> &'static str {
    match vendor_id {
        0x10DE => "nvidia",
        0x1002 | 0x1022 => "amd",
        0x8086 => "intel",
        0x1414 => "microsoft",
        _ => "other",
    }
}

/// A LUID as the protocol writes it: 16 lowercase hex digits.
pub fn luid_hex(luid: u64) -> String {
    format!("{luid:016x}")
}

/// A protocol LUID (hex, up to 16 digits, either case; an optional `0x`).
pub fn parse_luid(s: &str) -> Option<u64> {
    let t = s.trim().trim_start_matches("0x").trim_start_matches("0X");
    if t.is_empty() || t.len() > 16 {
        return None;
    }
    u64::from_str_radix(t, 16).ok()
}

impl Adapter {
    pub fn vendor(&self) -> &'static str {
        vendor_name(self.vendor_id)
    }
    pub fn to_json(&self) -> Value {
        json!({"luid": luid_hex(self.luid), "vendor": self.vendor(), "vendor_id": self.vendor_id, "device_id": self.device_id,
               "name": self.name, "vram_mb": self.vram_mb, "dxgi_index": self.dxgi_index})
    }
}

/// De-duplicate by LUID, keeping the first (DXGI's lowest index) of each.
pub fn dedup(list: Vec<Adapter>) -> Vec<Adapter> {
    let mut out: Vec<Adapter> = Vec::new();
    for a in list {
        if !out.iter().any(|b| b.luid == a.luid) {
            out.push(a);
        }
    }
    out
}

/// The hardware adapters (Windows: DXGI; elsewhere: none).
pub fn adapters() -> Vec<Adapter> {
    #[cfg(windows)]
    {
        dedup(win::enumerate())
    }
    #[cfg(not(windows))]
    {
        Vec::new()
    }
}

#[cfg(windows)]
mod win {
    //! DXGI through its COM vtables (no bindings crate): CreateDXGIFactory1, IDXGIFactory1::EnumAdapters1,
    //! IDXGIAdapter1::GetDesc1.
    use super::Adapter;
    use core::ffi::c_void;

    #[repr(C)]
    struct Guid(u32, u16, u16, [u8; 8]);
    const IID_IDXGI_FACTORY1: Guid = Guid(0x770aae78, 0xf26f, 0x4dba, [0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87]);

    #[repr(C)]
    struct Desc1 {
        description: [u16; 128],
        vendor_id: u32,
        device_id: u32,
        sub_sys_id: u32,
        revision: u32,
        dedicated_video_memory: usize,
        dedicated_system_memory: usize,
        shared_system_memory: usize,
        luid_low: u32,
        luid_high: i32,
        flags: u32,
    }
    const DXGI_ADAPTER_FLAG_SOFTWARE: u32 = 2;

    #[link(name = "dxgi")]
    extern "system" {
        fn CreateDXGIFactory1(riid: *const Guid, out: *mut *mut c_void) -> i32;
    }

    type Vtbl = *const usize;
    unsafe fn method(obj: *mut c_void, index: usize) -> usize {
        let vt = *(obj as *const Vtbl);
        *vt.add(index)
    }
    unsafe fn release(obj: *mut c_void) {
        let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(method(obj, 2));
        f(obj);
    }

    pub fn enumerate() -> Vec<Adapter> {
        let mut out = Vec::new();
        unsafe {
            let mut factory: *mut c_void = std::ptr::null_mut();
            if CreateDXGIFactory1(&IID_IDXGI_FACTORY1, &mut factory) < 0 || factory.is_null() {
                return out;
            }
            // IUnknown (3) + IDXGIObject (4) + IDXGIFactory (5): EnumAdapters1 is method 12
            let enum_adapters1: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> i32 = std::mem::transmute(method(factory, 12));
            let mut i = 0u32;
            loop {
                let mut adapter: *mut c_void = std::ptr::null_mut();
                if enum_adapters1(factory, i, &mut adapter) < 0 || adapter.is_null() {
                    break;
                }
                // IUnknown (3) + IDXGIObject (4) + IDXGIAdapter (3): GetDesc1 is method 10
                let get_desc1: unsafe extern "system" fn(*mut c_void, *mut Desc1) -> i32 = std::mem::transmute(method(adapter, 10));
                let mut d: Desc1 = std::mem::zeroed();
                let ok = get_desc1(adapter, &mut d) >= 0;
                release(adapter);
                if ok && d.flags & DXGI_ADAPTER_FLAG_SOFTWARE == 0 && d.vendor_id != 0x1414 {
                    let n = d.description.iter().position(|&c| c == 0).unwrap_or(d.description.len());
                    out.push(Adapter {
                        luid: ((d.luid_high as u32 as u64) << 32) | d.luid_low as u64,
                        vendor_id: d.vendor_id,
                        device_id: d.device_id,
                        name: String::from_utf16_lossy(&d.description[..n]).trim().to_string(),
                        vram_mb: (d.dedicated_video_memory as u64) >> 20,
                        dxgi_index: i,
                    });
                }
                i += 1;
            }
            release(factory);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a(luid: u64, idx: u32) -> Adapter {
        Adapter { luid, vendor_id: 0x10DE, device_id: 1, name: "x".into(), vram_mb: 1, dxgi_index: idx }
    }

    #[test]
    fn luids_round_trip_and_duplicates_go() {
        assert_eq!(luid_hex(0xd1f3), "000000000000d1f3");
        assert_eq!(parse_luid("000000000000D1F3"), Some(0xd1f3));
        assert_eq!(parse_luid("0x1_"), None);
        assert_eq!(parse_luid(""), None);
        assert_eq!(parse_luid("12345678901234567"), None);
        let d = dedup(vec![a(5, 0), a(7, 1), a(5, 2)]);
        assert_eq!(d.iter().map(|x| (x.luid, x.dxgi_index)).collect::<Vec<_>>(), vec![(5, 0), (7, 1)]);
        assert_eq!(vendor_name(0x1002), "amd");
    }

    #[test]
    fn adapters_enumerate_without_failing() {
        // (whatever the machine has; each LUID once)
        let list = adapters();
        let mut l: Vec<u64> = list.iter().map(|x| x.luid).collect();
        l.dedup();
        assert_eq!(l.len(), list.len());
    }
}
