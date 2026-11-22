//! The PC's graphics adapters as DXGI lists them (Windows; elsewhere none): the adapter LUID, which the
//! agent's profile, the job line's `gpu` and ONNX Runtime's device metadata all name an adapter by
//! (zero-farm PROTOCOL.md 8), its vendor, name and memory, and its position in DXGI's order (DirectML's
//! `device_id`). Adapters DXGI lists twice (one card, one LUID) are listed once; software adapters and the
//! Microsoft Basic Render Driver are left out.
//!
//! NVIDIA cards and CUDA (`cuda_filter`, `device_ordinal`): DXGI can also list ONE card under a SECOND LUID - a remote
//! desktop session's indirect display renders on the GPU and DXGI shows it again (seen 2026-11-21: an RTX 5090 at
//! LUIDs ..f58f and ..0b3978b9df) - and CUDA's device order need not be DXGI's (CUDA_DEVICE_ORDER, several cards). So
//! when DXGI lists two or more NVIDIA adapters, the CUDA driver (nvcuda.dll: `cuDeviceGetLuid` per CUDA ordinal) is
//! asked: an NVIDIA adapter no CUDA device has the LUID of is a second listing and is left out, and a CUDA-family
//! provider's device ordinal is the CUDA ordinal of the adapter's LUID. With one NVIDIA adapter (most PCs) the driver
//! is never asked and nothing changes: ordinal 0.
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

const NVIDIA: u32 = 0x10DE;

/// More than one NVIDIA adapter listed: the case where the CUDA driver is asked (module docs).
fn several_nvidia(list: &[Adapter]) -> bool {
    list.iter().filter(|a| a.vendor_id == NVIDIA).count() > 1
}

/// Leave out the NVIDIA adapters no CUDA device has the LUID of (`cuda`: the CUDA devices' LUIDs by ordinal) - DXGI's
/// second listing of a card (module docs). Nothing is left out when the driver gave no LUIDs, or when none of them is
/// one DXGI lists (no evidence which listing is the card's).
pub fn cuda_filter(list: Vec<Adapter>, cuda: Option<&[u64]>) -> Vec<Adapter> {
    let Some(c) = cuda else { return list };
    if !list.iter().any(|a| a.vendor_id == NVIDIA && c.contains(&a.luid)) {
        return list;
    }
    list.into_iter().filter(|a| a.vendor_id != NVIDIA || c.contains(&a.luid)).collect()
}

/// The device ordinal of a provider appended by name on adapter `a` (CUDA, TensorRT-RTX, MIGraphX): for an NVIDIA
/// adapter beside other NVIDIA adapters, the CUDA ordinal of its LUID (`cuda`); else its index among its vendor's
/// adapters (one NVIDIA card: 0).
pub fn device_ordinal(a: &Adapter, adapters: &[Adapter], cuda: Option<&[u64]>) -> u32 {
    if a.vendor_id == NVIDIA && several_nvidia(adapters) {
        if let Some(i) = cuda.and_then(|c| c.iter().position(|&l| l == a.luid)) {
            return i as u32;
        }
    }
    adapters.iter().filter(|x| x.vendor_id == a.vendor_id).position(|x| x.luid == a.luid).unwrap_or(0) as u32
}

/// The CUDA devices' LUIDs by CUDA ordinal (the NVIDIA driver's `cuDeviceGetLuid`), asked once per process and only
/// when `list` holds several NVIDIA adapters; None: not asked, no driver, or it failed.
pub fn cuda_luids(list: &[Adapter]) -> Option<&'static [u64]> {
    static L: std::sync::OnceLock<Option<Vec<u64>>> = std::sync::OnceLock::new();
    if !several_nvidia(list) {
        return None;
    }
    #[cfg(windows)]
    {
        L.get_or_init(win::cuda_luids).as_deref()
    }
    #[cfg(not(windows))]
    {
        L.get_or_init(|| None).as_deref()
    }
}

/// The hardware adapters (Windows: DXGI, an NVIDIA card's second listing left out; elsewhere: none).
pub fn adapters() -> Vec<Adapter> {
    #[cfg(windows)]
    {
        let list = dedup(win::enumerate());
        let cuda = cuda_luids(&list);
        cuda_filter(list, cuda)
    }
    #[cfg(not(windows))]
    {
        Vec::new()
    }
}

#[cfg(windows)]
mod win {
    //! DXGI through its COM vtables (no bindings crate): CreateDXGIFactory1, IDXGIFactory1::EnumAdapters1,
    //! IDXGIAdapter1::GetDesc1. The CUDA driver through nvcuda.dll's exports (loaded when asked; absent without an
    //! NVIDIA driver): cuInit, cuDeviceGetCount, cuDeviceGet, cuDeviceGetLuid.
    use super::Adapter;
    use core::ffi::c_void;

    extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *const c_void;
    }

    /// The CUDA devices' LUIDs by ordinal (`cuDeviceGetLuid`'s 8 bytes: the LUID struct, LowPart then HighPart, little
    /// endian - DXGI's `(HighPart << 32) | LowPart`), or None.
    pub(super) fn cuda_luids() -> Option<Vec<u64>> {
        unsafe {
            let name: Vec<u16> = "nvcuda.dll".encode_utf16().chain(std::iter::once(0)).collect();
            let m = LoadLibraryW(name.as_ptr());
            if m.is_null() {
                return None;
            }
            let f = |s: &[u8]| {
                let p = GetProcAddress(m, s.as_ptr());
                (!p.is_null()).then_some(p)
            };
            let (init, count, get, luid) = (f(b"cuInit\0")?, f(b"cuDeviceGetCount\0")?, f(b"cuDeviceGet\0")?, f(b"cuDeviceGetLuid\0")?);
            let init: unsafe extern "system" fn(u32) -> i32 = std::mem::transmute(init);
            let count: unsafe extern "system" fn(*mut i32) -> i32 = std::mem::transmute(count);
            let get: unsafe extern "system" fn(*mut i32, i32) -> i32 = std::mem::transmute(get);
            let luid: unsafe extern "system" fn(*mut u8, *mut u32, i32) -> i32 = std::mem::transmute(luid);
            if init(0) != 0 {
                return None;
            }
            let mut n = 0i32;
            if count(&mut n) != 0 || n < 0 {
                return None;
            }
            let mut out = Vec::with_capacity(n as usize);
            for i in 0..n {
                let (mut dev, mut bytes, mut mask) = (0i32, [0u8; 8], 0u32);
                if get(&mut dev, i) != 0 || luid(bytes.as_mut_ptr(), &mut mask, dev) != 0 {
                    return None;
                }
                out.push(u64::from_le_bytes(bytes));
            }
            Some(out)
        }
    }

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
    fn a_second_listing_of_an_nvidia_card_goes_and_cuda_names_the_ordinal() {
        // the 2026-11-21 desktop under a remote desktop session: the card at ..f58f (DXGI 0) and again at ..b9df (DXGI
        // 2), an Intel iGPU between; CUDA has one device, LUID ..f58f
        let intel = Adapter { vendor_id: 0x8086, ..a(0x107d8, 1) };
        let list = vec![a(0xf58f, 0), intel, a(0x0b3978b9df, 2)];
        let cuda = [0xf58fu64];
        let kept = cuda_filter(list.clone(), Some(&cuda));
        assert_eq!(kept.iter().map(|x| x.luid).collect::<Vec<_>>(), vec![0xf58f, 0x107d8]);
        assert_eq!(device_ordinal(&kept[0], &kept, Some(&cuda)), 0);
        // before the filter (both listings) the second one had ordinal 1 - the "Invalid device ID: 1" of the release
        assert_eq!(device_ordinal(&list[2], &list, None), 1);
        // without the driver's answer, or when it names no listed LUID: nothing left out (the behaviour of before)
        assert_eq!(cuda_filter(list.clone(), None), list);
        assert_eq!(cuda_filter(list.clone(), Some(&[0x99])), list);
        // two cards in CUDA's order, not DXGI's (CUDA_DEVICE_ORDER fastest first): the ordinal follows the LUID
        let two = vec![a(0xa, 0), a(0xb, 1)];
        let cuda = [0xbu64, 0xa];
        assert_eq!(cuda_filter(two.clone(), Some(&cuda)), two);
        assert_eq!((device_ordinal(&two[0], &two, Some(&cuda)), device_ordinal(&two[1], &two, Some(&cuda))), (1, 0));
        // one NVIDIA card: ordinal 0 (the driver is never asked); other vendors by their index among their own
        assert_eq!(device_ordinal(&two[1], &two[1..], Some(&cuda)), 0);
        assert!(cuda_luids(&two[..1]).is_none(), "one NVIDIA adapter: not asked");
        let amd = vec![Adapter { vendor_id: 0x1002, ..a(1, 0) }, Adapter { vendor_id: 0x1002, ..a(2, 1) }];
        assert_eq!(device_ordinal(&amd[1], &amd, Some(&[2])), 1);
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
