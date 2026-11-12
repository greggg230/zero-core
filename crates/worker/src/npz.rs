//! Reading `.npz` files (numpy's zip of `.npy` members, stored or deflated): recorded network batches
//! (zero-core `tools/onnx_record.py`'s format) for the worker's `eval-batches`, and the worker's own
//! files in tests.
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use flate2::read::DeflateDecoder;

/// A loaded array: dtype descr ('<i8', '<f4', '|u1', ...), shape, raw little-endian bytes.
#[derive(Clone, Debug)]
pub struct Arr {
    pub descr: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

fn u16le(b: &[u8], i: usize) -> usize {
    u16::from_le_bytes([b[i], b[i + 1]]) as usize
}
fn u32le(b: &[u8], i: usize) -> usize {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize
}

fn parse_npy(raw: &[u8]) -> Result<Arr, String> {
    if raw.len() < 10 || &raw[..6] != b"\x93NUMPY" {
        return Err("not a .npy member".into());
    }
    let (hl, start) = match raw[6] {
        1 => (u16le(raw, 8), 10),
        2 | 3 => (u32le(raw, 8), 12),
        v => return Err(format!(".npy version {v}")),
    };
    let h = std::str::from_utf8(&raw[start..start + hl]).map_err(|_| "npy header")?;
    let field = |k: &str| -> Result<&str, String> {
        let i = h.find(&format!("'{k}':")).ok_or_else(|| format!("npy header without {k}: {h}"))?;
        Ok(h[i + k.len() + 3..].trim_start())
    };
    let d = field("descr")?;
    let descr = d.trim_start_matches('\'').split('\'').next().unwrap_or("").to_string();
    if field("fortran_order")?.starts_with("True") {
        return Err("fortran-order arrays are not read".into());
    }
    let s = field("shape")?;
    let s = &s[1..s.find(')').ok_or("npy shape")?];
    let shape: Vec<usize> = s.split(',').map(str::trim).filter(|x| !x.is_empty()).map(|x| x.parse().map_err(|_| format!("npy shape {s}"))).collect::<Result<_, _>>()?;
    Ok(Arr { descr, shape, data: raw[start + hl..].to_vec() })
}

/// Every member of an npz (stored or deflated), by name without `.npy`.
pub fn read_npz(path: &Path) -> Result<BTreeMap<String, Arr>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let e = (0..b.len().saturating_sub(21)).rev().find(|&i| u32le(&b, i) == 0x0605_4b50).ok_or_else(|| format!("{}: not a zip", path.display()))?;
    let n = u16le(&b, e + 10);
    let mut cd = u32le(&b, e + 16);
    if cd == 0xffff_ffff {
        return Err(format!("{}: a zip64 archive (not read)", path.display()));
    }
    let mut out = BTreeMap::new();
    for _ in 0..n {
        if u32le(&b, cd) != 0x0201_4b50 {
            return Err(format!("{}: bad central directory", path.display()));
        }
        let method = u16le(&b, cd + 10);
        let mut csize = u32le(&b, cd + 20);
        let mut usize_ = u32le(&b, cd + 24);
        let nl = u16le(&b, cd + 28);
        let xl = u16le(&b, cd + 30);
        let cl = u16le(&b, cd + 32);
        let mut lho = u32le(&b, cd + 42);
        let name = String::from_utf8_lossy(&b[cd + 46..cd + 46 + nl]).to_string();
        // zip64 extra field in the central directory (Python's zipfile force_zip64)
        let mut x = cd + 46 + nl;
        let xe = x + xl;
        while x + 4 <= xe {
            let (id, sz) = (u16le(&b, x), u16le(&b, x + 2));
            if id == 1 {
                let mut p = x + 4;
                let rd = |p: &mut usize| {
                    let v = u64::from_le_bytes(b[*p..*p + 8].try_into().unwrap()) as usize;
                    *p += 8;
                    v
                };
                if usize_ == 0xffff_ffff {
                    usize_ = rd(&mut p);
                }
                if csize == 0xffff_ffff {
                    csize = rd(&mut p);
                }
                if lho == 0xffff_ffff {
                    lho = rd(&mut p);
                }
            }
            x += 4 + sz;
        }
        cd += 46 + nl + xl + cl;
        let data_at = lho + 30 + u16le(&b, lho + 26) + u16le(&b, lho + 28);
        let comp = &b[data_at..data_at + csize];
        let raw = match method {
            0 => comp.to_vec(),
            8 => {
                let mut v = Vec::with_capacity(usize_);
                DeflateDecoder::new(comp).read_to_end(&mut v).map_err(|e| format!("{}: {name}: {e}", path.display()))?;
                v
            }
            m => return Err(format!("{}: {name}: zip method {m}", path.display())),
        };
        let key = name.strip_suffix(".npy").unwrap_or(&name).to_string();
        out.insert(key, parse_npy(&raw).map_err(|e| format!("{}: {name}: {e}", path.display()))?);
    }
    Ok(out)
}

impl Arr {
    fn check(&self, want: &[&str]) -> Result<(), String> {
        if want.contains(&self.descr.as_str()) {
            Ok(())
        } else {
            Err(format!("dtype {} (want {want:?})", self.descr))
        }
    }
    pub fn i64s(&self) -> Result<Vec<i64>, String> {
        match self.descr.as_str() {
            "<i8" => Ok(self.data.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()),
            "<i4" => Ok(self.data.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64).collect()),
            "<i2" => Ok(self.data.chunks_exact(2).map(|c| i16::from_le_bytes(c.try_into().unwrap()) as i64).collect()),
            "<u2" => Ok(self.data.chunks_exact(2).map(|c| u16::from_le_bytes(c.try_into().unwrap()) as i64).collect()),
            "|i1" => Ok(self.data.iter().map(|&x| x as i8 as i64).collect()),
            "|u1" | "|b1" => Ok(self.data.iter().map(|&x| x as i64).collect()),
            d => Err(format!("dtype {d}: not an integer array")),
        }
    }
    pub fn f32s(&self) -> Result<Vec<f32>, String> {
        self.check(&["<f4"])?;
        Ok(self.data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }
    pub fn u16s(&self) -> Result<Vec<u16>, String> {
        self.check(&["<u2", "<f2"])?;
        Ok(self.data.chunks_exact(2).map(|c| u16::from_le_bytes(c.try_into().unwrap())).collect())
    }
    /// A 0-d unicode array's text (`np.array(str)`).
    pub fn text(&self) -> Result<String, String> {
        if !self.descr.starts_with("<U") {
            return Err(format!("dtype {}: not text", self.descr));
        }
        Ok(self.data.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).take_while(|&u| u != 0).filter_map(char::from_u32).collect())
    }
}
