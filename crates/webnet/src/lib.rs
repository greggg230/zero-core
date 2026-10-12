//! Zero Core's web network: the network of `zero/net.py` (residual MLP with LayerNorm) as a
//! plain Rust forward pass over float16 weights exported by `zero.webexport.export_model`
//! (`.bin`, format below). Moved from Caverna Zero's `wasm/src/net.rs` (its browser player,
//! which re-exports it as `cvz_wasm::net`); any game's web player can use it.
//!
//! Determinism: every sum runs in a fixed order with plain f32 `+` / `*` (no FMA contraction,
//! no reassociation) and `exp` is our own f64 routine, so the native build and the wasm build
//! compute bit-identical outputs (tested in Caverna Zero: `cvzweb` native vs the wasm module).
//!
//! File format (little endian): `b"CVZW"` (the magic kept from Caverna Zero, where the format
//! was born: every exported .bin carries it), u32 version (1), u32 header length, the header
//! (UTF-8 JSON: features, actions, hidden, blocks, head, seats, source, iter, tensors =
//! [{name, shape, offset}] with offsets in f16 elements), spaces to an even length, then the
//! f16 data. Tensors: `inp.*`, `blocks.k.{ln,fc1,fc2}.*`, `ln.*`, `pol.*`, `val.0.*`, `val.2.*`
//! (the margin head `mar.*` is not exported: the browser search does not use it).

use serde_json::Value;

pub const MAX_SEATS: usize = 8;

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as u32;
    let f = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if f == 0 {
            s << 31
        } else {
            // subnormal: normalise
            let mut e2 = 127 - 15 + 1;
            let mut f2 = f;
            while f2 & 0x400 == 0 {
                f2 <<= 1;
                e2 -= 1;
            }
            (s << 31) | ((e2 as u32) << 23) | ((f2 & 0x3ff) << 13)
        }
    } else if e == 31 {
        (s << 31) | (0xff << 23) | (f << 13)
    } else {
        (s << 31) | ((e + 127 - 15) << 23) | (f << 13)
    };
    f32::from_bits(bits)
}

/// e^x in f64 with plain arithmetic only (bit-identical on every target).
pub fn exp64(x: f64) -> f64 {
    if x < -745.0 {
        return 0.0;
    }
    if x > 709.0 {
        return f64::INFINITY;
    }
    const LN2: f64 = std::f64::consts::LN_2;
    let k = (x / LN2 + if x >= 0.0 { 0.5 } else { -0.5 }) as i64;
    let r = x - k as f64 * LN2;
    // Taylor series of e^r, |r| <= ln2/2: 14 terms are below 1e-17 relative
    let mut term = 1.0;
    let mut sum = 1.0;
    for n in 1..15 {
        term *= r / n as f64;
        sum += term;
    }
    // multiply by 2^k (k in -1075..1024)
    let mut k = k;
    let mut out = sum;
    while k > 1000 {
        out *= f64::from_bits(((1000 + 1023) as u64) << 52);
        k -= 1000;
    }
    while k < -1000 {
        out *= f64::from_bits(((-1000 + 1023) as u64) << 52);
        k += 1000;
    }
    out * f64::from_bits(((k + 1023) as u64) << 52)
}

struct Block {
    ln_g: Vec<f32>,
    ln_b: Vec<f32>,
    fc1_t: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2_t: Vec<f32>,
    fc2_b: Vec<f32>,
}

/// A loaded network (weights as f32; dense layers stored input-major for the sparse-input
/// axpy form).
pub struct Model {
    pub features: usize,
    pub actions: usize,
    pub hidden: usize,
    pub head: usize,
    pub seats: usize,
    pub header: Value,
    inp_t: Vec<f32>,
    inp_b: Vec<f32>,
    blocks: Vec<Block>,
    ln_g: Vec<f32>,
    ln_b: Vec<f32>,
    pol_w: Vec<f32>,
    pol_b: Vec<f32>,
    val0_t: Vec<f32>,
    val0_b: Vec<f32>,
    val2_w: Vec<f32>,
    val2_b: Vec<f32>,
}

/// Reusable buffers of a forward pass.
#[derive(Default, Clone)]
pub struct Scratch {
    h: Vec<f32>,
    a: Vec<f32>,
    u: Vec<f32>,
    d: Vec<f32>,
    v1: Vec<f32>,
    nz: Vec<(u32, f32)>,
}

/// y = b + W x with W stored input-major (`wt[i * nout + o]`); zero inputs are skipped.
/// Each output is summed over the non-zero inputs in increasing order (deterministic, the
/// same on every target); outputs are processed in register-sized blocks.
#[inline(never)]
fn affine_t(x: &[f32], wt: &[f32], b: &[f32], y: &mut [f32], nz: &mut Vec<(u32, f32)>) {
    const BLK: usize = 32;
    let nout = y.len();
    nz.clear();
    for (i, &xi) in x.iter().enumerate() {
        if xi != 0.0 {
            nz.push((i as u32, xi));
        }
    }
    let mut c = 0;
    while c + BLK <= nout {
        let mut acc = [0.0f32; BLK];
        acc.copy_from_slice(&b[c..c + BLK]);
        for &(i, xi) in nz.iter() {
            let o = i as usize * nout + c;
            let w: &[f32; BLK] = wt[o..o + BLK].try_into().unwrap();
            for k in 0..BLK {
                acc[k] += xi * w[k];
            }
        }
        y[c..c + BLK].copy_from_slice(&acc);
        c += BLK;
    }
    if c < nout {
        y[c..].copy_from_slice(&b[c..]);
        for &(i, xi) in nz.iter() {
            let o = i as usize * nout;
            for k in c..nout {
                y[k] += xi * wt[o + k];
            }
        }
    }
}

/// Dot product in increasing index order.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        s += x * y;
    }
    s
}

fn layer_norm(x: &[f32], g: &[f32], b: &[f32], y: &mut [f32]) {
    let n = x.len() as f32;
    let mut m = 0.0f32;
    for &v in x {
        m += v;
    }
    m /= n;
    let mut var = 0.0f32;
    for &v in x {
        let d = v - m;
        var += d * d;
    }
    var /= n;
    let r = 1.0 / (var + 1e-5).sqrt();
    for i in 0..x.len() {
        y[i] = (x[i] - m) * r * g[i] + b[i];
    }
}

fn relu(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

impl Model {
    /// Parse a `.bin` weight file.
    pub fn from_bytes(bytes: &[u8]) -> Result<Model, String> {
        if bytes.len() < 12 || &bytes[0..4] != b"CVZW" {
            return Err("not a CVZW weight file".into());
        }
        let u32_at = |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as usize;
        let version = u32_at(4);
        if version != 1 {
            return Err(format!("CVZW version {version} (expected 1)"));
        }
        let hlen = u32_at(8);
        let hend = 12 + hlen;
        if bytes.len() < hend {
            return Err("truncated header".into());
        }
        let header: Value = serde_json::from_slice(&bytes[12..hend]).map_err(|e| format!("header: {e}"))?;
        let data = &bytes[hend..];
        let geti = |k: &str| header.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| format!("header: missing {k}"));
        let features = geti("features")?;
        let actions = geti("actions")?;
        let hidden = geti("hidden")?;
        let nblocks = geti("blocks")?;
        let head = geti("head")?;
        let seats = geti("seats")?;
        let tensors = header.get("tensors").and_then(|v| v.as_array()).ok_or("header: no tensors")?;
        let find = |name: &str, shape: &[usize]| -> Result<Vec<f32>, String> {
            let t = tensors.iter().find(|t| t.get("name").and_then(|n| n.as_str()) == Some(name)).ok_or_else(|| format!("missing tensor {name}"))?;
            let sh: Vec<usize> = t.get("shape").and_then(|s| s.as_array()).ok_or("shape")?.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect();
            if sh != shape {
                return Err(format!("tensor {name}: shape {sh:?}, expected {shape:?}"));
            }
            let off = t.get("offset").and_then(|v| v.as_u64()).ok_or("offset")? as usize;
            let n: usize = shape.iter().product();
            if (off + n) * 2 > data.len() {
                return Err(format!("tensor {name}: data out of range"));
            }
            Ok((0..n).map(|k| f16_to_f32(u16::from_le_bytes([data[2 * (off + k)], data[2 * (off + k) + 1]]))).collect())
        };
        // [out, in] -> [in, out]
        let transpose = |w: Vec<f32>, nout: usize, nin: usize| -> Vec<f32> {
            let mut t = vec![0.0f32; w.len()];
            for o in 0..nout {
                for i in 0..nin {
                    t[i * nout + o] = w[o * nin + i];
                }
            }
            t
        };
        let h = hidden;
        let mut blocks = Vec::new();
        for k in 0..nblocks {
            let p = format!("blocks.{k}.");
            blocks.push(Block {
                ln_g: find(&format!("{p}ln.weight"), &[h])?,
                ln_b: find(&format!("{p}ln.bias"), &[h])?,
                fc1_t: transpose(find(&format!("{p}fc1.weight"), &[h, h])?, h, h),
                fc1_b: find(&format!("{p}fc1.bias"), &[h])?,
                fc2_t: transpose(find(&format!("{p}fc2.weight"), &[h, h])?, h, h),
                fc2_b: find(&format!("{p}fc2.bias"), &[h])?,
            });
        }
        Ok(Model {
            features,
            actions,
            hidden,
            head,
            seats,
            inp_t: transpose(find("inp.weight", &[h, features])?, h, features),
            inp_b: find("inp.bias", &[h])?,
            blocks,
            ln_g: find("ln.weight", &[h])?,
            ln_b: find("ln.bias", &[h])?,
            pol_w: find("pol.weight", &[actions, h])?,
            pol_b: find("pol.bias", &[actions])?,
            val0_t: transpose(find("val.0.weight", &[head, h])?, head, h),
            val0_b: find("val.0.bias", &[head])?,
            val2_w: find("val.2.weight", &[seats, head])?,
            val2_b: find("val.2.bias", &[seats])?,
            header,
        })
    }

    /// Forward pass: `x` = features (at least `self.features` long; a narrower network reads
    /// a prefix), `acts` = the legal actions whose logits go to `logits` (same order), `n` =
    /// players. Returns the value head's softmax over the `n` relative seats (P(win), seat j =
    /// (decider + j) % n), zero beyond n.
    pub fn forward(&self, x: &[f32], acts: &[u16], n: usize, sc: &mut Scratch, logits: &mut Vec<f32>) -> [f32; MAX_SEATS] {
        let h = self.hidden;
        sc.h.resize(h, 0.0);
        sc.a.resize(h, 0.0);
        sc.u.resize(h, 0.0);
        sc.d.resize(h, 0.0);
        sc.v1.resize(self.head, 0.0);
        affine_t(&x[..self.features], &self.inp_t, &self.inp_b, &mut sc.h, &mut sc.nz);
        relu(&mut sc.h);
        for b in &self.blocks {
            layer_norm(&sc.h, &b.ln_g, &b.ln_b, &mut sc.a);
            affine_t(&sc.a, &b.fc1_t, &b.fc1_b, &mut sc.u, &mut sc.nz);
            relu(&mut sc.u);
            affine_t(&sc.u, &b.fc2_t, &b.fc2_b, &mut sc.d, &mut sc.nz);
            for i in 0..h {
                sc.h[i] += sc.d[i];
            }
        }
        layer_norm(&sc.h, &self.ln_g, &self.ln_b, &mut sc.a);
        relu(&mut sc.a);
        let z = &sc.a;
        logits.clear();
        for &a in acts {
            let a = a as usize;
            logits.push(if a < self.actions { self.pol_b[a] + dot(&self.pol_w[a * h..(a + 1) * h], z) } else { 0.0 });
        }
        affine_t(z, &self.val0_t, &self.val0_b, &mut sc.v1, &mut sc.nz);
        relu(&mut sc.v1);
        let n = n.min(self.seats).min(MAX_SEATS);
        let mut vl = [0.0f32; MAX_SEATS];
        for (j, v) in vl.iter_mut().enumerate().take(n) {
            *v = self.val2_b[j] + dot(&self.val2_w[j * self.head..(j + 1) * self.head], &sc.v1);
        }
        softmax_n(&vl[..n])
    }
}

/// Softmax of up to MAX_SEATS values (f64 inside, deterministic), padded with zeros.
pub fn softmax_n(v: &[f32]) -> [f32; MAX_SEATS] {
    let mut out = [0.0f32; MAX_SEATS];
    let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut e = [0.0f64; MAX_SEATS];
    let mut tot = 0.0;
    for (j, &x) in v.iter().enumerate() {
        e[j] = exp64(x as f64 - mx);
        tot += e[j];
    }
    for j in 0..v.len() {
        out[j] = (e[j] / tot) as f32;
    }
    out
}

/// Softmax of logits into priors (f64 inside, deterministic).
pub fn softmax_into(logits: &[f32], out: &mut Vec<f32>) {
    out.clear();
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut tot = 0.0f64;
    for &l in logits {
        tot += exp64(l as f64 - mx);
    }
    for &l in logits {
        out.push((exp64(l as f64 - mx) / tot) as f32);
    }
}

/// f32 -> f16 bits (round to nearest even; overflow to inf, tiny values to zero / subnormal).
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 0xff {
        return s | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let e2 = e - 127 + 15;
    if e2 >= 31 {
        return s | 0x7c00;
    }
    if e2 <= 0 {
        if e2 < -10 {
            return s;
        }
        let m = m | 0x80_0000;
        let shift = (14 - e2) as u32;
        let half = 1u32 << (shift - 1);
        let mut r = m >> shift;
        let rem = m & ((1 << shift) - 1);
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return s | r as u16;
    }
    let mut r = ((e2 as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1;
    }
    s | r as u16
}

/// A random network as `.bin` bytes (tests and benchmarks without a checkpoint).
pub fn random_model_bytes(features: usize, actions: usize, hidden: usize, blocks: usize, head: usize, seats: usize, seed: u64) -> Vec<u8> {
    let mut st = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut rnd = move || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        ((st >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
    };
    let mut shapes: Vec<(String, Vec<usize>)> = vec![("inp.weight".into(), vec![hidden, features]), ("inp.bias".into(), vec![hidden])];
    for k in 0..blocks {
        for (n, sh) in [("ln.weight", vec![hidden]), ("ln.bias", vec![hidden]), ("fc1.weight", vec![hidden, hidden]), ("fc1.bias", vec![hidden]), ("fc2.weight", vec![hidden, hidden]), ("fc2.bias", vec![hidden])] {
            shapes.push((format!("blocks.{k}.{n}"), sh));
        }
    }
    shapes.push(("ln.weight".into(), vec![hidden]));
    shapes.push(("ln.bias".into(), vec![hidden]));
    shapes.push(("pol.weight".into(), vec![actions, hidden]));
    shapes.push(("pol.bias".into(), vec![actions]));
    shapes.push(("val.0.weight".into(), vec![head, hidden]));
    shapes.push(("val.0.bias".into(), vec![head]));
    shapes.push(("val.2.weight".into(), vec![seats, head]));
    shapes.push(("val.2.bias".into(), vec![seats]));
    let mut tensors = Vec::new();
    let mut data: Vec<u8> = Vec::new();
    let mut off = 0usize;
    for (name, sh) in &shapes {
        let n: usize = sh.iter().product();
        let scale = if name.ends_with("ln.weight") { 0.0 } else { 0.4 / (sh.last().copied().unwrap_or(1) as f32).sqrt() };
        for _ in 0..n {
            let v = if name.ends_with("ln.weight") { 1.0 + 0.2 * rnd() } else { rnd() * scale * 2.0 };
            data.extend_from_slice(&f32_to_f16(v).to_le_bytes());
        }
        tensors.push(serde_json::json!({ "name": name, "shape": sh, "offset": off }));
        off += n;
    }
    let header = serde_json::json!({ "features": features, "actions": actions, "hidden": hidden, "blocks": blocks, "head": head,
                                     "seats": seats, "source": "random", "iter": 0, "tensors": tensors });
    let mut hb = header.to_string().into_bytes();
    if (12 + hb.len()) % 2 == 1 {
        hb.push(b' ');
    }
    let mut out = b"CVZW".to_vec();
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(hb.len() as u32).to_le_bytes());
    out.extend_from_slice(&hb);
    out.extend_from_slice(&data);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_matches_std() {
        for k in -2000..2000 {
            let x = k as f64 * 0.0137;
            let a = exp64(x);
            let b = x.exp();
            assert!(((a - b) / b).abs() < 1e-14, "{x}: {a} vs {b}");
        }
        assert_eq!(exp64(0.0), 1.0);
    }

    #[test]
    fn f16_round_trip() {
        for k in -3000..3000 {
            let x = k as f32 * 0.00137;
            let h = f32_to_f16(x);
            let y = f16_to_f32(h);
            assert!((x - y).abs() <= x.abs() * 1e-3 + 1e-7, "{x} -> {y}");
            assert_eq!(f32_to_f16(y), h);
        }
    }

    #[test]
    fn f16_decoding() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x0200), 2f32.powi(-15));
        assert_eq!(f16_to_f32(0x3555), 0.33325195);
    }
}
