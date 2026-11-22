//! CUDA graphs for a static zero-onnx/2 graph (zero-core docs/az.md 72, WIP): one GPU launch per call.
//!
//! ONNX Runtime launches the fast graph's ~1,200 kernels one by one each call, with 27 device -> host syncs for its
//! data-dependent shapes: ~20 ms of a CPU thread per call whatever the batch, which leaves a 4-core GPU PC's crew
//! CPU-bound at 0.86x of the trainer's path (whose evaluator replays CUDA graphs). The static graph (docs/az.md 67) has
//! no data-dependent shape; with every symbolic dimension fixed (ONNX Runtime free-dimension overrides) all its nodes
//! run on the CUDA provider (its one int64 ArgMax is float32 from this change: `onnx_export.ZeroOnnx2Static`), so
//! ONNX Runtime can capture it as a CUDA graph and replay it.
//!
//! A *bucket* is one such fixed-shape session: capacities for the rows, non-zeros, options, tail tokens, tail floats,
//! overrides and tokens a row ([`Caps`]). A piece is padded to its bucket's capacities with inert padding rows after
//! its own ([`pad`]): every padding element sits in a padding row - non-zeros on global features with value 0,
//! options distinct actions, tail tokens valid ones whose widths sum to exactly the padding floats (the graph refuses
//! every row when the floats don't fit the widths), overrides the generic logit of a padding row's option - and the
//! real rows' answers are the unpadded call's (checked bit for bit on the CPU; on CUDA within float rounding: a row's
//! answer depends on its piece only by rounding, as for any piece). The bucket binds pinned host buffers as its inputs
//! and outputs (IoBinding at fixed addresses: the replay copies from and to them), so a call is: the padded piece
//! written into the input buffers, one replay, the real rows read from the output buffers.
//!
//! Opt-in while it is measured: `ZERO_WORKER_CUDA_GRAPHS=1` and a static graph on the CUDA provider.

use std::path::{Path, PathBuf};

use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{IoBinding, Session};
use ort::value::Tensor;

/// The environment variable that turns the buckets on (`1`).
pub const ENV: &str = "ZERO_WORKER_CUDA_GRAPHS";

/// Buckets a session keeps at most (each holds its own activations on the GPU); the least recently used goes.
pub const MAX_BUCKETS: usize = 4;

pub fn wanted() -> bool {
    std::env::var(ENV).as_deref() == Ok("1")
}

/// A piece's sizes, or a bucket's capacities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Caps {
    pub b: usize,
    pub nnz: usize,
    pub k: usize,
    pub nt: usize,
    pub l: usize,
    pub no: usize,
    pub t: usize,
}

fn up(n: usize, a: usize) -> usize {
    n.div_ceil(a.max(1)) * a.max(1)
}

impl Caps {
    /// A new bucket's capacities for a piece of `s`: the rows rounded up to 64 with at least one padding row, each
    /// per-row quantity a quarter over the piece's mean per row, the tail floats 120 a padding token (the padding
    /// tokens' mean width; any mean from 56 to 438 fits), T as the piece's.
    pub fn for_piece(s: &Caps) -> Caps {
        let b = up(s.b + 1, 64);
        let per = |x: usize, a: usize| up((x as f64 / s.b.max(1) as f64 * 1.25 * b as f64).ceil() as usize + a, a);
        let nt = per(s.nt, 64);
        Caps { b, nnz: per(s.nnz, 256), k: per(s.k, 64), nt, l: s.l + 120 * (nt - s.nt), no: per(s.no, 16).max(16), t: s.t }
    }

    /// The bucket holds piece `s` (its tail padding found too: [`tail_plan`]).
    pub fn holds(&self, s: &Caps, tails: &Tails) -> bool {
        self.b > s.b && self.nnz >= s.nnz && self.k >= s.k && self.nt >= s.nt && self.l >= s.l && self.no >= s.no && self.t == s.t && tail_plan(self.nt - s.nt, self.l - s.l, tails).is_some()
    }
}

/// The tail segments a padding token may use: (segment, width in floats, slots), from the graph's metadata `tails`
/// ([segment, name, width, slots]: a tail segment with slots and floats).
#[derive(Clone, Debug, Default)]
pub struct Tails(pub Vec<(i32, usize, usize)>);

impl Tails {
    pub fn of(meta: &[(String, String)]) -> Result<Tails, String> {
        let raw = meta.iter().find(|e| e.0 == "tails").map(|e| e.1.as_str()).ok_or("a static graph without metadata tails")?;
        let v: Vec<serde_json::Value> = serde_json::from_str(raw).map_err(|e| format!("metadata tails: {e}"))?;
        let mut out = Vec::new();
        for t in v {
            let (seg, w, n) = (t[0].as_i64(), t[2].as_u64(), t[3].as_u64());
            if let (Some(seg), Some(w), Some(n)) = (seg, w, n) {
                if w > 0 && n > 0 {
                    out.push((seg as i32, w as usize, n as usize));
                }
            }
        }
        if out.is_empty() {
            return Err("metadata tails: no tail segment with slots and floats".into());
        }
        Ok(Tails(out))
    }
}

/// `n` padding tokens' segments whose widths sum to exactly `l` floats: the two narrowest widths fill the bulk, `h`
/// of the widest and `g` of a middle width fix the total; None if there is no such plan.
pub fn tail_plan(n: usize, l: usize, tails: &Tails) -> Option<Vec<i32>> {
    if n == 0 {
        return (l == 0).then(Vec::new);
    }
    let mut ws: Vec<usize> = tails.0.iter().map(|t| t.1).collect();
    ws.sort_unstable();
    ws.dedup();
    let seg_of = |w: usize| tails.0.iter().filter(|t| t.1 == w).max_by_key(|t| t.2).map(|t| t.0).expect("a width's segment");
    let (a, b, hw) = (ws[0], *ws.get(1).unwrap_or(&ws[0]), *ws.last().unwrap());
    if l < a * n || l > hw * n {
        return None;
    }
    let mids: Vec<usize> = if ws.len() > 3 { ws[2..ws.len() - 1].to_vec() } else { vec![hw] };
    for &gw in &mids {
        for h in 0..=n {
            for g in 0..=(n - h).min(2 * b) {
                let m = n - h - g;
                let fixed = hw * h + gw * g;
                if fixed > l {
                    break;
                }
                let r = l - fixed;
                if r < a * m {
                    break;
                }
                if r > b * m {
                    continue;
                }
                let x = if a == b {
                    if r != a * m {
                        continue;
                    }
                    m
                } else {
                    let num = b * m - r;
                    if num % (b - a) != 0 {
                        continue;
                    }
                    num / (b - a)
                };
                let mut out = vec![seg_of(hw); h];
                out.extend(std::iter::repeat_n(seg_of(gw), g));
                out.extend(std::iter::repeat_n(seg_of(a), x));
                out.extend(std::iter::repeat_n(seg_of(b), m - x));
                return Some(out);
            }
        }
    }
    None
}

/// A piece's zero-onnx/2 inputs (the static graph's: `t_cap`'s length is `t`), as `session::run_piece` builds them.
#[derive(Clone, Debug, Default)]
pub struct Piece {
    pub x_ptr: Vec<i64>,
    pub x_idx: Vec<i32>,
    pub x_val: Vec<f32>,
    pub n_players: Vec<i64>,
    pub belief: Vec<bool>,
    pub opt_off: Vec<i64>,
    pub opt_action: Vec<i32>,
    pub t_off: Vec<i64>,
    pub t_seg: Vec<i32>,
    pub t_slot_hi: Vec<i32>,
    pub t_slot_lo: Vec<i32>,
    pub t_val: Vec<f32>,
    pub ov_off: Vec<i64>,
    pub ov_opt: Vec<i32>,
    pub ov_seg: Vec<i32>,
    pub ov_col: Vec<i32>,
    pub ov_slot_hi: Vec<i32>,
    pub ov_slot_lo: Vec<i32>,
    pub t: usize,
}

impl Piece {
    pub fn sizes(&self) -> Caps {
        Caps { b: self.n_players.len(), nnz: self.x_idx.len(), k: self.opt_action.len(), nt: self.t_seg.len(), l: self.t_val.len(), no: self.ov_opt.len(), t: self.t }
    }
}

/// `n` padding elements over `p` padding rows, the last rows first, each at most `max` -> per padding row.
fn spread(n: usize, p: usize, max: usize) -> Result<Vec<usize>, String> {
    let mut c = vec![0usize; p];
    let mut left = n;
    for r in (0..p).rev() {
        let k = left.min(max);
        c[r] = k;
        left -= k;
    }
    if left > 0 {
        return Err(format!("{n} padding elements do not fit {p} padding rows of {max}"));
    }
    Ok(c)
}

fn extend_off(off: &mut Vec<i64>, counts: &[usize]) {
    let mut last = *off.last().unwrap_or(&0);
    for &c in counts {
        last += c as i64;
        off.push(last);
    }
}

/// `p` padded to `caps` (module docs): `actions` the network's (a padding row's options are distinct actions).
/// `global_features`: non-zeros on features below it are no token's (the first head slot's present flag).
pub fn pad(p: &Piece, caps: &Caps, tails: &Tails, actions: usize, global_features: usize) -> Result<Piece, String> {
    let s = p.sizes();
    if caps.b <= s.b || caps.t != s.t {
        return Err(format!("bucket {caps:?} for a piece of {s:?}"));
    }
    let np = caps.b - s.b;
    let mut q = p.clone();
    // non-zeros: global features 0, 1, ... with value 0 in padding rows
    let c = spread(caps.nnz - s.nnz, np, global_features.max(1))?;
    extend_off(&mut q.x_ptr, &c);
    for &k in &c {
        q.x_idx.extend((0..k).map(|i| i as i32));
        q.x_val.extend(std::iter::repeat_n(0.0f32, k));
    }
    q.n_players.extend(std::iter::repeat_n(2i64, np));
    q.belief.extend(std::iter::repeat_n(false, np));
    // options: distinct actions
    let c = spread(caps.k - s.k, np, actions)?;
    extend_off(&mut q.opt_off, &c);
    for &k in &c {
        q.opt_action.extend((0..k).map(|i| i as i32));
    }
    // tail tokens: exact widths, valid (a tail segment, a slot below its count, sorted, no duplicate) per padding row
    let plan = tail_plan(caps.nt - s.nt, caps.l - s.l, tails).ok_or_else(|| format!("no tail padding of {} tokens, {} floats", caps.nt - s.nt, caps.l - s.l))?;
    let count = |seg: i32| tails.0.iter().find(|t| t.0 == seg).map(|t| t.2).unwrap_or(0);
    let width = |seg: i32| tails.0.iter().find(|t| t.0 == seg).map(|t| t.1).unwrap_or(0);
    let mut rows: Vec<std::collections::BTreeMap<i32, usize>> = vec![Default::default(); np];
    let mut r = 0usize;
    for seg in plan {
        let mut tries = 0;
        while rows[r].get(&seg).copied().unwrap_or(0) >= count(seg) {
            r = (r + 1) % np;
            tries += 1;
            if tries > np {
                return Err("padding tail tokens do not fit the padding rows".into());
            }
        }
        *rows[r].entry(seg).or_insert(0) += 1;
        r = (r + 1) % np;
    }
    let mut c = Vec::with_capacity(np);
    for row in &rows {
        let mut k = 0;
        for (&seg, &n) in row {
            for slot in 0..n {
                q.t_seg.push(seg);
                q.t_slot_hi.push((slot >> 16) as i32);
                q.t_slot_lo.push((slot & 0xffff) as i32);
                q.t_val.extend(std::iter::repeat_n(0.0f32, width(seg)));
                k += 1;
            }
        }
        c.push(k);
    }
    extend_off(&mut q.t_off, &c);
    // overrides: a padding row's option 0, the generic logit (-1, 0, 0, 0)
    let c = spread(caps.no - s.no, np, usize::MAX)?;
    extend_off(&mut q.ov_off, &c);
    let n = caps.no - s.no;
    q.ov_opt.extend(std::iter::repeat_n(0i32, n));
    q.ov_seg.extend(std::iter::repeat_n(-1i32, n));
    q.ov_col.extend(std::iter::repeat_n(0i32, n));
    q.ov_slot_hi.extend(std::iter::repeat_n(0i32, n));
    q.ov_slot_lo.extend(std::iter::repeat_n(0i32, n));
    debug_assert_eq!(q.sizes(), Caps { b: caps.b, ..*caps });
    Ok(q)
}

/// One fixed-shape session with its graph captured: pinned host buffers bound as inputs and outputs.
pub struct Bucket {
    pub caps: Caps,
    session: Session,
    binding: IoBinding,
    inputs: Vec<(&'static str, Buf)>,
    outputs: Vec<(&'static str, Buf)>,
    pub calls: u64,
    pub last: u64,
}

enum Buf {
    I64(Tensor<i64>),
    I32(Tensor<i32>),
    F32(Tensor<f32>),
    Bool(Tensor<bool>),
}

impl Buf {
    fn write_i64(&mut self, v: &[i64]) -> Result<(), String> {
        match self {
            Buf::I64(t) => {
                t.extract_tensor_mut().1[..v.len()].copy_from_slice(v);
                Ok(())
            }
            _ => Err("buffer type".into()),
        }
    }
    fn write_i32(&mut self, v: &[i32]) -> Result<(), String> {
        match self {
            Buf::I32(t) => {
                t.extract_tensor_mut().1[..v.len()].copy_from_slice(v);
                Ok(())
            }
            _ => Err("buffer type".into()),
        }
    }
    fn write_f32(&mut self, v: &[f32]) -> Result<(), String> {
        match self {
            Buf::F32(t) => {
                t.extract_tensor_mut().1[..v.len()].copy_from_slice(v);
                Ok(())
            }
            _ => Err("buffer type".into()),
        }
    }
    fn write_bool(&mut self, v: &[bool]) -> Result<(), String> {
        match self {
            Buf::Bool(t) => {
                t.extract_tensor_mut().1[..v.len()].copy_from_slice(v);
                Ok(())
            }
            _ => Err("buffer type".into()),
        }
    }
}

fn oerr(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// A static graph's buckets on one CUDA device (`ord`).
pub struct Cache {
    pub path: PathBuf,
    pub ord: i32,
    pub tails: Tails,
    pub actions: usize,
    pub aux_w: usize,
    pub global_features: usize,
    pub buckets: Vec<Bucket>,
    pub built: u64,
    pub tick: u64,
}

impl Cache {
    /// The bucket for piece `s` (made, or remade larger, when none holds it): its index.
    fn bucket_for(&mut self, s: &Caps, make: &dyn Fn(&Path, &Caps) -> Result<Session, String>) -> Result<usize, String> {
        self.tick += 1;
        if let Some(i) = self.buckets.iter().position(|b| b.caps.holds(s, &self.tails)) {
            self.buckets[i].last = self.tick;
            return Ok(i);
        }
        // a bucket of this T that holds the rows but not the rest grows (its successor holds both)
        let mut caps = Caps::for_piece(s);
        if let Some(i) = self.buckets.iter().position(|b| b.caps.t == s.t && b.caps.b >= caps.b) {
            let o = self.buckets.remove(i).caps;
            caps = Caps { b: o.b.max(caps.b), nnz: o.nnz.max(caps.nnz), k: o.k.max(caps.k), nt: o.nt.max(caps.nt), l: 0, no: o.no.max(caps.no), t: s.t };
            caps.l = s.l + 120 * (caps.nt - s.nt);
        }
        if !caps.holds(s, &self.tails) {
            return Err(format!("no bucket can hold a piece of {s:?} (made {caps:?})"));
        }
        if self.buckets.len() >= MAX_BUCKETS {
            let lru = self.buckets.iter().enumerate().min_by_key(|(_, b)| b.last).map(|(i, _)| i).expect("a bucket");
            self.buckets.remove(lru);
        }
        let session = make(&self.path, &caps)?;
        let b = self.bind(session, caps)?;
        self.buckets.push(b);
        self.built += 1;
        let n = self.buckets.len() - 1;
        self.buckets[n].last = self.tick;
        Ok(n)
    }

    fn bind(&self, session: Session, caps: Caps) -> Result<Bucket, String> {
        let pinned = Allocator::new(&session, MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, AllocatorType::Device, MemoryType::CPUOutput).map_err(oerr)?).map_err(oerr)?;
        let mut binding = session.create_binding().map_err(oerr)?;
        let i64t = |n: usize| Tensor::<i64>::new(&pinned, [n]).map(Buf::I64).map_err(oerr);
        let i32t = |n: usize| Tensor::<i32>::new(&pinned, [n]).map(Buf::I32).map_err(oerr);
        let f32t = |n: usize| Tensor::<f32>::new(&pinned, [n]).map(Buf::F32).map_err(oerr);
        let b = caps.b;
        let inputs: Vec<(&'static str, Buf)> = vec![
            ("x_ptr", i64t(b + 1)?),
            ("x_idx", i32t(caps.nnz)?),
            ("x_val", f32t(caps.nnz)?),
            ("n_players", i64t(b)?),
            ("belief", Tensor::<bool>::new(&pinned, [b]).map(Buf::Bool).map_err(oerr)?),
            ("opt_off", i64t(b + 1)?),
            ("opt_action", i32t(caps.k)?),
            ("t_off", i64t(b + 1)?),
            ("t_seg", i32t(caps.nt)?),
            ("t_slot_hi", i32t(caps.nt)?),
            ("t_slot_lo", i32t(caps.nt)?),
            ("t_val", f32t(caps.l)?),
            ("ov_off", i64t(b + 1)?),
            ("ov_opt", i32t(caps.no)?),
            ("ov_seg", i32t(caps.no)?),
            ("ov_col", i32t(caps.no)?),
            ("ov_slot_hi", i32t(caps.no)?),
            ("ov_slot_lo", i32t(caps.no)?),
            ("t_cap", i32t(caps.t)?),
        ];
        let f2 = |w: usize| Tensor::<f32>::new(&pinned, [b, w]).map(Buf::F32).map_err(oerr);
        let outputs: Vec<(&'static str, Buf)> = vec![
            ("policy", f2(self.actions)?),
            ("value", f2(zero_search::MAX_SEATS)?),
            ("margin", f2(zero_search::MAX_SEATS)?),
            ("aux", f2(self.aux_w.max(1))?),
            ("refused", i64t(b)?),
        ];
        for (name, buf) in &inputs {
            match buf {
                Buf::I64(t) => binding.bind_input(*name, t),
                Buf::I32(t) => binding.bind_input(*name, t),
                Buf::F32(t) => binding.bind_input(*name, t),
                Buf::Bool(t) => binding.bind_input(*name, t),
            }
            .map_err(|e| format!("bind {name}: {e}"))?;
        }
        let mut outputs = outputs;
        for (name, buf) in outputs.iter_mut() {
            match buf {
                Buf::I64(t) => binding.bind_output(*name, t),
                Buf::F32(t) => binding.bind_output(*name, t),
                _ => unreachable!(),
            }
            .map_err(|e| format!("bind {name}: {e}"))?;
        }
        Ok(Bucket { caps, session, binding, inputs, outputs, calls: 0, last: 0 })
    }

    /// The graph's outputs for piece `p` (its own rows only): (policy [n A], value [n 8], margin [n 8], aux [n W],
    /// refused [n]).
    #[allow(clippy::type_complexity)]
    pub fn run(&mut self, p: &Piece, make: &dyn Fn(&Path, &Caps) -> Result<Session, String>) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<i64>), String> {
        let s = p.sizes();
        let i = self.bucket_for(&s, make)?;
        let caps = self.buckets[i].caps;
        let q = pad(p, &caps, &self.tails, self.actions, self.global_features)?;
        let bk = &mut self.buckets[i];
        for (name, buf) in bk.inputs.iter_mut() {
            match *name {
                "x_ptr" => buf.write_i64(&q.x_ptr)?,
                "x_idx" => buf.write_i32(&q.x_idx)?,
                "x_val" => buf.write_f32(&q.x_val)?,
                "n_players" => buf.write_i64(&q.n_players)?,
                "belief" => buf.write_bool(&q.belief)?,
                "opt_off" => buf.write_i64(&q.opt_off)?,
                "opt_action" => buf.write_i32(&q.opt_action)?,
                "t_off" => buf.write_i64(&q.t_off)?,
                "t_seg" => buf.write_i32(&q.t_seg)?,
                "t_slot_hi" => buf.write_i32(&q.t_slot_hi)?,
                "t_slot_lo" => buf.write_i32(&q.t_slot_lo)?,
                "t_val" => buf.write_f32(&q.t_val)?,
                "ov_off" => buf.write_i64(&q.ov_off)?,
                "ov_opt" => buf.write_i32(&q.ov_opt)?,
                "ov_seg" => buf.write_i32(&q.ov_seg)?,
                "ov_col" => buf.write_i32(&q.ov_col)?,
                "ov_slot_hi" => buf.write_i32(&q.ov_slot_hi)?,
                "ov_slot_lo" => buf.write_i32(&q.ov_slot_lo)?,
                "t_cap" => {}
                other => return Err(format!("input {other}")),
            }
        }
        bk.session.run_binding(&bk.binding).map_err(oerr)?;
        bk.calls += 1;
        let n = s.b;
        let take = |buf: &Buf, w: usize| -> Vec<f32> {
            match buf {
                Buf::F32(t) => t.extract_tensor().1[..n * w].to_vec(),
                _ => Vec::new(),
            }
        };
        let (a, w) = (self.actions, self.aux_w.max(1));
        let pol = take(&bk.outputs[0].1, a);
        let val = take(&bk.outputs[1].1, zero_search::MAX_SEATS);
        let mar = take(&bk.outputs[2].1, zero_search::MAX_SEATS);
        let aux = if self.aux_w == 0 { Vec::new() } else { take(&bk.outputs[3].1, w) };
        let refused = match &bk.outputs[4].1 {
            Buf::I64(t) => t.extract_tensor().1[..n].to_vec(),
            _ => Vec::new(),
        };
        Ok((pol, val, mar, aux, refused))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tails() -> Tails {
        // the legacy network's: segments 2, 3 (281 floats, 32 slots), 4 (64, 16), 5 (438, 8), 6 (271, 4), 7 (64, 64), 8 (56, 4)
        Tails(vec![(2, 281, 32), (3, 281, 32), (4, 64, 16), (5, 438, 8), (6, 271, 4), (7, 64, 64), (8, 56, 4)])
    }

    #[test]
    fn tail_plans_sum_exactly() {
        let t = tails();
        for (n, l) in [(0usize, 0usize), (1, 56), (313, 37_560), (448 - 330, 120 * 118), (10, 4_380)] {
            let p = tail_plan(n, l, &t).unwrap_or_else(|| panic!("{n} {l}"));
            let w = |s: i32| t.0.iter().find(|x| x.0 == s).unwrap().1;
            assert_eq!(p.len(), n);
            assert_eq!(p.iter().map(|&s| w(s)).sum::<usize>(), l, "{n} {l}");
        }
        assert!(tail_plan(1, 55, &t).is_none());
        assert!(tail_plan(1, 439, &t).is_none());
        assert!(tail_plan(0, 1, &t).is_none());
    }

    #[test]
    fn a_padded_piece_has_the_buckets_sizes_and_keeps_its_rows() {
        let t = tails();
        let p = Piece {
            x_ptr: vec![0, 2, 3],
            x_idx: vec![5, 30_000, 7],
            x_val: vec![1.0, 1.0, 0.5],
            n_players: vec![2, 2],
            belief: vec![false, true],
            opt_off: vec![0, 2, 3],
            opt_action: vec![3, 9, 4],
            t_off: vec![0, 1, 1],
            t_seg: vec![4],
            t_slot_hi: vec![0],
            t_slot_lo: vec![3],
            t_val: vec![0.25; 64],
            ov_off: vec![0, 0, 0],
            t: 16,
            ..Default::default()
        };
        let s = p.sizes();
        let caps = Caps::for_piece(&s);
        assert!(caps.holds(&s, &t), "{caps:?}");
        let q = pad(&p, &caps, &t, 1024, 20_000).unwrap();
        assert_eq!(q.sizes(), caps);
        assert_eq!(&q.x_ptr[..3], &p.x_ptr[..]);
        assert_eq!(&q.x_idx[..3], &p.x_idx[..]);
        assert_eq!(&q.opt_off[..3], &p.opt_off[..]);
        assert_eq!(&q.t_off[..3], &p.t_off[..]);
        assert_eq!(*q.x_ptr.last().unwrap() as usize, caps.nnz);
        assert_eq!(*q.opt_off.last().unwrap() as usize, caps.k);
        assert_eq!(*q.t_off.last().unwrap() as usize, caps.nt);
        assert_eq!(*q.ov_off.last().unwrap() as usize, caps.no);
        // padding non-zeros on global features only, options distinct within a row
        assert!(q.x_idx[3..].iter().all(|&i| i < 20_000));
        for r in 2..caps.b {
            let (lo, hi) = (q.opt_off[r] as usize, q.opt_off[r + 1] as usize);
            let mut a: Vec<i32> = q.opt_action[lo..hi].to_vec();
            a.sort_unstable();
            a.dedup();
            assert_eq!(a.len(), hi - lo);
        }
    }
}
