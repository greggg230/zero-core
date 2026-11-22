//! CUDA graphs for a static zero-onnx/2 graph (zero-core docs/az.md 75, WIP): one GPU launch per call.
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
//! When: a fixed graph (below) on the CUDA provider always - the run opts in by sending it (its pool variant 'fixed-fp16',
//! zero-core docs/az.md 76); a static graph only with `ZERO_WORKER_CUDA_GRAPHS=1` (a measurement's); `ZERO_WORKER_CUDA_GRAPHS=0`
//! turns them off for either (`wanted`).
//!
//! The fixed graph (zero-core docs/az.md 76: the fast graph's arithmetic at fixed capacities) takes the same buckets
//! with two capacities more: the batch's present head slots (`s`) and a row's non-zeros in a run of features (`mr`).
//! Its padding rows' non-zeros go to features outside every run (value 0: inert anywhere), so they count in no run;
//! a padding row's always-present head slots count in `s`.

use std::path::{Path, PathBuf};

use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{IoBinding, Session};
use ort::value::Tensor;

/// The environment variable that turns the buckets on for a static graph (`1`) or off for any graph (`0`).
pub const ENV: &str = "ZERO_WORKER_CUDA_GRAPHS";

/// Buckets a session keeps at most (each holds its own activations on the GPU); the least recently used goes.
pub const MAX_BUCKETS: usize = 4;

/// The fixed graph's: a bucket of T tokens a row takes pieces of up to this many fewer (its rows' padding keys are
/// masked: the same answers up to float rounding), so that pieces whose longest row varies reuse buckets.
pub const T_SLACK: usize = 32;

/// The buckets for a graph on the CUDA provider: a fixed graph's (`fixed`) unless `ZERO_WORKER_CUDA_GRAPHS` is `0`; a
/// static graph's only when it is `1` (`env`: the variable's value, if set).
pub fn wanted(fixed: bool, env: Option<&str>) -> bool {
    match env {
        Some("0") => false,
        Some("1") => true,
        _ => fixed,
    }
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
    /// The fixed graph's present head slots (the batch's, the global tokens aside) and a row's most non-zeros in a
    /// run; 0 for the static graph.
    pub s: usize,
    pub mr: usize,
}

fn up(n: usize, a: usize) -> usize {
    n.div_ceil(a.max(1)) * a.max(1)
}

impl Caps {
    /// A new bucket's capacities for a piece of `s`: the rows rounded up to 64 with at least one padding row, each
    /// per-row quantity a quarter over the piece's mean per row, the tail floats 120 a padding token (the padding
    /// tokens' mean width; any mean from 56 to 438 fits), T as the piece's.
    /// The fixed graph's: `s` as the other per-row quantities plus `ax` (a padding row's always-present head slots)
    /// a padding row, `mr` 8 over the piece's rounded up to 16.
    pub fn for_piece(s: &Caps, ax: usize) -> Caps {
        // (at least 16 padding rows: the tail padding's tokens need slots in them)
        let b = up(s.b + 16, 64);
        let per = |x: usize, a: usize| up((x as f64 / s.b.max(1) as f64 * 1.25 * b as f64).ceil() as usize + a, a);
        let nt = per(s.nt, 64);
        let (sc, mr) = if s.mr == 0 { (0, 0) } else { (per(s.s, 64) + ax * (b - s.b), up(s.mr + 8, 16)) };
        Caps { b, nnz: per(s.nnz, 256), k: per(s.k, 64), nt, l: s.l + 120 * (nt - s.nt), no: per(s.no, 16).max(16), t: s.t, s: sc, mr }
    }

    /// The bucket holds piece `s` (its tail padding found too: [`tail_plan`]).
    pub fn holds(&self, s: &Caps, tails: &Tails, ax: usize) -> bool {
        let t_ok = if s.mr == 0 { self.t == s.t } else { self.t >= s.t && self.t <= s.t + T_SLACK };
        self.b >= s.b + 16 && self.nnz >= s.nnz && self.k >= s.k && self.nt >= s.nt && self.l >= s.l && self.no >= s.no && t_ok
            && self.mr >= s.mr && (s.mr == 0 || self.s >= s.s + ax * (self.b - s.b))
            && tail_plan(self.nt - s.nt, self.l - s.l, tails, self.b - s.b).is_some()
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

/// `n` padding tokens' segments whose widths sum to exactly `l` floats, within `np` padding rows (each segment holds
/// its slot count per row): two width classes take the bulk, at most two others fix the remainder (small counts) -
/// every pair and pair of fixers tried, the roomiest first; None if there is no such plan.
pub fn tail_plan(n: usize, l: usize, tails: &Tails, np: usize) -> Option<Vec<i32>> {
    if n == 0 {
        return (l == 0).then(Vec::new);
    }
    // width classes: (width, capacity in all, its segments roomiest first)
    let mut classes: Vec<(usize, usize, Vec<(i32, usize)>)> = Vec::new();
    for &(seg, w, count) in &tails.0 {
        match classes.iter_mut().find(|c| c.0 == w) {
            Some(c) => {
                c.1 += count * np;
                c.2.push((seg, count * np));
            }
            None => classes.push((w, count * np, vec![(seg, count * np)])),
        }
    }
    for c in classes.iter_mut() {
        c.2.sort_by_key(|x| std::cmp::Reverse(x.1));
    }
    let k = classes.len();
    let mut pairs: Vec<(usize, usize)> = (0..k).flat_map(|a| (0..k).map(move |b| (a, b))).filter(|(a, b)| classes[*a].0 < classes[*b].0).collect();
    pairs.sort_by_key(|&(a, b)| std::cmp::Reverse(classes[a].1.min(classes[b].1)));
    const FIX: usize = 48;
    for &(a, b) in &pairs {
        let (wa, ca, wb, cb) = (classes[a].0, classes[a].1, classes[b].0, classes[b].1);
        let others: Vec<usize> = (0..k).filter(|&x| x != a && x != b).collect();
        let fixers: Vec<(Option<usize>, Option<usize>)> = std::iter::once((None, None))
            .chain(others.iter().map(|&x| (Some(x), None)))
            .chain(others.iter().flat_map(|&x| others.iter().filter(move |&&y| y > x).map(move |&y| (Some(x), Some(y)))))
            .collect();
        for (f1, f2) in fixers {
            let lim = |f: Option<usize>| f.map_or(0, |x| classes[x].1.min(FIX));
            for c1 in 0..=lim(f1) {
                for c2 in 0..=lim(f2) {
                    let used = c1 + c2;
                    if used > n {
                        continue;
                    }
                    let fixed = c1 * f1.map_or(0, |x| classes[x].0) + c2 * f2.map_or(0, |x| classes[x].0);
                    if fixed > l {
                        continue;
                    }
                    let (m, r) = (n - used, l - fixed);
                    // x of wa, m - x of wb: wa x + wb (m - x) = r
                    if r > wb * m || r < wa * m {
                        continue;
                    }
                    let num = wb * m - r;
                    if num % (wb - wa) != 0 {
                        continue;
                    }
                    let x = num / (wb - wa);
                    if x > ca || m - x > cb {
                        continue;
                    }
                    let mut out = Vec::with_capacity(n);
                    let mut take = |cls: usize, cnt: usize| {
                        let mut left = cnt;
                        for &(seg, cap) in &classes[cls].2 {
                            let t = left.min(cap);
                            out.extend(std::iter::repeat_n(seg, t));
                            left -= t;
                        }
                    };
                    take(a, x);
                    take(b, m - x);
                    if let Some(f) = f1 {
                        take(f, c1);
                    }
                    if let Some(f) = f2 {
                        take(f, c2);
                    }
                    if out.len() == n {
                        return Some(out);
                    }
                }
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
    /// The fixed graph's present head slots and a row's most non-zeros in a run (`Caps`); 0 for the static graph.
    pub s: usize,
    pub mr: usize,
}

impl Piece {
    pub fn sizes(&self) -> Caps {
        Caps { b: self.n_players.len(), nnz: self.x_idx.len(), k: self.opt_action.len(), nt: self.t_seg.len(), l: self.t_val.len(), no: self.ov_opt.len(), t: self.t, s: self.s, mr: self.mr }
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
/// `pad_features`: the features a padding row's non-zeros (value 0) go to, ascending - the static graph's: those below
/// the first head slot's present flag (no token's); the fixed graph's: those in no run.
pub fn pad(p: &Piece, caps: &Caps, tails: &Tails, actions: usize, pad_features: &[i32]) -> Result<Piece, String> {
    let s = p.sizes();
    if caps.b <= s.b || caps.t < s.t || (s.mr == 0 && caps.t != s.t) {
        return Err(format!("bucket {caps:?} for a piece of {s:?}"));
    }
    let np = caps.b - s.b;
    let mut q = p.clone();
    // non-zeros: the padding features with value 0 in padding rows
    let c = spread(caps.nnz - s.nnz, np, pad_features.len())?;
    extend_off(&mut q.x_ptr, &c);
    for &k in &c {
        q.x_idx.extend_from_slice(&pad_features[..k]);
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
    let plan = tail_plan(caps.nt - s.nt, caps.l - s.l, tails, np).ok_or_else(|| format!("no tail padding of {} tokens, {} floats", caps.nt - s.nt, caps.l - s.l))?;
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
    q.s = caps.s;
    q.mr = caps.mr;
    q.t = caps.t;
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
    /// The device buffers bound when the pinned host buffers are only staging (default; `ZERO_WORKER_CUDA_GRAPHS_PINNED=1`
    /// binds the pinned buffers themselves): inputs copied in, outputs copied out each call.
    dev_in: Vec<Buf>,
    dev_out: Vec<Buf>,
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
    /// This buffer's contents copied into `dst` (another device's buffer of the same type and size).
    fn copy_to(&self, dst: &mut Buf) -> Result<(), String> {
        match (self, dst) {
            (Buf::I64(a), Buf::I64(b)) => a.copy_into(b).map_err(oerr),
            (Buf::I32(a), Buf::I32(b)) => a.copy_into(b).map_err(oerr),
            (Buf::F32(a), Buf::F32(b)) => a.copy_into(b).map_err(oerr),
            (Buf::Bool(a), Buf::Bool(b)) => a.copy_into(b).map_err(oerr),
            _ => Err("buffer types differ".into()),
        }
    }
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
    /// The features padding rows' non-zeros go to (`pad`).
    pub pad_features: Vec<i32>,
    /// A padding row's always-present head slots (the global token aside): the fixed graph's `s` counts them.
    pub always_extra: usize,
    pub buckets: Vec<Bucket>,
    pub built: u64,
    pub tick: u64,
}

impl Cache {
    /// The bucket for piece `s` (made, or remade larger, when none holds it): its index.
    fn bucket_for(&mut self, s: &Caps, make: &dyn Fn(&Path, &Caps) -> Result<Session, String>) -> Result<usize, String> {
        self.tick += 1;
        let ax = self.always_extra;
        // (a bucket fits when it holds the piece and its padding rows have room for the padding non-zeros and options)
        let (nf, na) = (self.pad_features.len(), self.actions);
        let fits = |c: &Caps| c.holds(s, &self.tails, ax) && c.nnz - s.nnz <= (c.b - s.b) * nf && c.k - s.k <= (c.b - s.b) * na;
        if let Some(i) = self.buckets.iter().position(|b| fits(&b.caps)) {
            self.buckets[i].last = self.tick;
            return Ok(i);
        }
        // a bucket of this T that holds the rows but not the rest grows (its successor holds both)
        let mut caps = Caps::for_piece(s, ax);
        let near = |t: usize| if s.mr == 0 { t == s.t } else { t >= s.t && t <= s.t + T_SLACK };
        if let Some(i) = self.buckets.iter().position(|b| near(b.caps.t) && b.caps.b >= caps.b) {
            let o = self.buckets.remove(i).caps;
            let b = o.b.max(caps.b).max(up(s.b + 16, 64));
            let sc = if s.mr == 0 { 0 } else { o.s.max(caps.s).max(s.s + ax * (b - s.b)) };
            caps = Caps { b, nnz: o.nnz.max(caps.nnz), k: o.k.max(caps.k), nt: o.nt.max(caps.nt), l: 0, no: o.no.max(caps.no), t: o.t.max(s.t), s: sc, mr: o.mr.max(caps.mr) };
            caps.l = s.l + 120 * (caps.nt - s.nt);
        }
        if !fits(&caps) {
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
        let mut inputs = inputs;
        if caps.mr > 0 {
            // (the fixed graph's other capacities: only their lengths are read)
            inputs.push(("s_cap", i32t(caps.s)?));
            inputs.push(("m_cap", i32t(caps.mr)?));
        }
        let f2 = |w: usize| Tensor::<f32>::new(&pinned, [b, w]).map(Buf::F32).map_err(oerr);
        let outputs: Vec<(&'static str, Buf)> = vec![
            ("policy", f2(self.actions)?),
            ("value", f2(zero_search::MAX_SEATS)?),
            ("margin", f2(zero_search::MAX_SEATS)?),
            ("aux", f2(self.aux_w.max(1))?),
            ("refused", i64t(b)?),
        ];
        let pinned_only = std::env::var("ZERO_WORKER_CUDA_GRAPHS_PINNED").as_deref() == Ok("1");
        let cuda = Allocator::new(&session, MemoryInfo::new(AllocationDevice::CUDA, self.ord, AllocatorType::Device, MemoryType::Default).map_err(oerr)?).map_err(oerr)?;
        let twin = |b: &Buf| -> Result<Buf, String> {
            Ok(match b {
                Buf::I64(t) => Buf::I64(Tensor::<i64>::new(&cuda, t.shape().clone()).map_err(oerr)?),
                Buf::I32(t) => Buf::I32(Tensor::<i32>::new(&cuda, t.shape().clone()).map_err(oerr)?),
                Buf::F32(t) => Buf::F32(Tensor::<f32>::new(&cuda, t.shape().clone()).map_err(oerr)?),
                Buf::Bool(t) => Buf::Bool(Tensor::<bool>::new(&cuda, t.shape().clone()).map_err(oerr)?),
            })
        };
        let (mut dev_in, mut dev_out) = (Vec::new(), Vec::new());
        if !pinned_only {
            for (_, b) in &inputs {
                dev_in.push(twin(b)?);
            }
            for (_, b) in &outputs {
                dev_out.push(twin(b)?);
            }
        }
        let bind_in: Vec<(&'static str, &Buf)> = if pinned_only { inputs.iter().map(|(n, b)| (*n, b)).collect() } else { inputs.iter().map(|(n, _)| *n).zip(dev_in.iter()).collect() };
        for (name, buf) in bind_in {
            match buf {
                Buf::I64(t) => binding.bind_input(name, t),
                Buf::I32(t) => binding.bind_input(name, t),
                Buf::F32(t) => binding.bind_input(name, t),
                Buf::Bool(t) => binding.bind_input(name, t),
            }
            .map_err(|e| format!("bind {name}: {e}"))?;
        }
        let mut outputs = outputs;
        let names: Vec<&'static str> = outputs.iter().map(|(n, _)| *n).collect();
        let targets: Vec<&mut Buf> = if pinned_only { outputs.iter_mut().map(|(_, b)| b).collect() } else { dev_out.iter_mut().collect() };
        for (name, buf) in names.into_iter().zip(targets) {
            match buf {
                Buf::I64(t) => binding.bind_output(name, t),
                Buf::F32(t) => binding.bind_output(name, t),
                _ => unreachable!(),
            }
            .map_err(|e| format!("bind {name}: {e}"))?;
        }
        Ok(Bucket { caps, session, binding, inputs, outputs, dev_in, dev_out, calls: 0, last: 0 })
    }

    /// The graph's outputs for piece `p` (its own rows only): (policy [n A], value [n 8], margin [n 8], aux [n W],
    /// refused [n]).
    #[allow(clippy::type_complexity)]
    pub fn run(&mut self, p: &Piece, make: &dyn Fn(&Path, &Caps) -> Result<Session, String>) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<i64>), String> {
        let s = p.sizes();
        let i = self.bucket_for(&s, make)?;
        let caps = self.buckets[i].caps;
        let q = pad(p, &caps, &self.tails, self.actions, &self.pad_features)?;
        if let Ok(d) = std::env::var("ZERO_WORKER_CUDA_GRAPHS_DUMP") {
            // (a diagnostic: the first padded piece's inputs as JSON, the unpadded piece beside it)
            let path = std::path::Path::new(&d);
            if !path.exists() {
                let j = |q: &Piece| serde_json::json!({"x_ptr": q.x_ptr, "x_idx": q.x_idx, "x_val": q.x_val, "n_players": q.n_players, "belief": q.belief,
                    "opt_off": q.opt_off, "opt_action": q.opt_action, "t_off": q.t_off, "t_seg": q.t_seg, "t_slot_hi": q.t_slot_hi, "t_slot_lo": q.t_slot_lo,
                    "t_val_len": q.t_val.len(), "ov_off": q.ov_off, "ov_opt": q.ov_opt, "ov_seg": q.ov_seg, "ov_col": q.ov_col, "ov_slot_hi": q.ov_slot_hi, "ov_slot_lo": q.ov_slot_lo, "t": q.t});
                let _ = std::fs::write(path, serde_json::json!({"padded": j(&q), "piece": j(p), "caps": format!("{caps:?}")}).to_string());
            }
        }
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
                "t_cap" | "s_cap" | "m_cap" => {}
                other => return Err(format!("input {other}")),
            }
        }
        for ((_, h), d) in bk.inputs.iter().zip(bk.dev_in.iter_mut()) {
            h.copy_to(d)?;
        }
        bk.session.run_binding(&bk.binding).map_err(oerr)?;
        for (d, (_, h)) in bk.dev_out.iter().zip(bk.outputs.iter_mut()) {
            d.copy_to(h)?;
        }
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
    fn a_fixed_graph_replays_cuda_graphs_unless_turned_off() {
        assert!(wanted(true, None) && wanted(true, Some("")) && wanted(true, Some("1")) && !wanted(true, Some("0")));
        assert!(!wanted(false, None) && wanted(false, Some("1")) && !wanted(false, Some("0")));
    }

    #[test]
    fn tail_plans_sum_exactly() {
        let t = tails();
        for (n, l) in [(0usize, 0usize), (1, 56), (313, 37_560), (448 - 330, 120 * 118), (10, 4_380)] {
            let p = tail_plan(n, l, &t, 64).unwrap_or_else(|| panic!("{n} {l}"));
            let w = |s: i32| t.0.iter().find(|x| x.0 == s).unwrap().1;
            assert_eq!(p.len(), n);
            assert_eq!(p.iter().map(|&s| w(s)).sum::<usize>(), l, "{n} {l}");
        }
        assert!(tail_plan(1, 55, &t, 64).is_none());
        assert!(tail_plan(1, 439, &t, 64).is_none());
        assert!(tail_plan(0, 1, &t, 64).is_none());
        // few padding rows: the roomy segments carry it (the self-test's small pieces)
        assert!(tail_plan(300, 36_000, &t, 24).is_some());
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
        let caps = Caps::for_piece(&s, 0);
        assert!(caps.holds(&s, &t, 0), "{caps:?}");
        let pf: Vec<i32> = (0..20_000).collect();
        let q = pad(&p, &caps, &t, 1024, &pf).unwrap();
        assert_eq!(q.sizes(), caps);
        assert_eq!(&q.x_ptr[..3], &p.x_ptr[..]);
        assert_eq!(&q.x_idx[..3], &p.x_idx[..]);
        assert_eq!(&q.opt_off[..3], &p.opt_off[..]);
        assert_eq!(&q.t_off[..3], &p.t_off[..]);
        assert_eq!(*q.x_ptr.last().unwrap() as usize, caps.nnz);
        assert_eq!(*q.opt_off.last().unwrap() as usize, caps.k);
        assert_eq!(*q.t_off.last().unwrap() as usize, caps.nt);
        assert_eq!(*q.ov_off.last().unwrap() as usize, caps.no);
        // the fixed graph's: room for the padding rows' always-present slots, a row's runs, a grown bucket keeps both
        let pf = Piece { s: 9, mr: 3, ..p.clone() };
        let fs = pf.sizes();
        let fc = Caps::for_piece(&fs, 2);
        assert!(fc.holds(&fs, &t, 2) && fc.s >= 9 + 2 * (fc.b - 2) && fc.mr >= 3, "{fc:?}");
        assert!(!Caps { s: 9 + 2 * (fc.b - 2) - 1, ..fc }.holds(&fs, &t, 2));
        assert!(!Caps { mr: 2, ..fc }.holds(&fs, &t, 2));
        // a fixed-graph bucket of a larger T (within T_SLACK) takes the piece; the static graph's needs its T
        assert!(Caps { t: 16 + T_SLACK, ..fc }.holds(&fs, &t, 2) && !Caps { t: 17 + T_SLACK, ..fc }.holds(&fs, &t, 2));
        assert!(!Caps { t: 32, ..caps }.holds(&s, &t, 0));
        let wide = Caps { t: 48, ..fc };
        assert_eq!(pad(&pf, &wide, &t, 1024, &(40_000..40_100).collect::<Vec<i32>>()).unwrap().sizes(), wide);
        let free: Vec<i32> = (40_000..40_100).collect();
        let fq = pad(&pf, &fc, &t, 1024, &free).unwrap();
        assert_eq!(fq.sizes(), fc);
        assert!(fq.x_idx[3..].iter().all(|i| free.contains(i)));
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
