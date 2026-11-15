//! Sample shards in zero-core's format (`python/zero/data.py`: `write_shard`, `ShardWriter`), so the
//! trainer's arrival check, non-finite guard and window take them as a remote worker's: a zip of `.npy`
//! members (deflate, zlib level 1 as remote shards), in data.py's order -
//! `meta` (0-d unicode: the JSON meta), then the features (`shard_sparse` 1: `x_ptr` int64 [N+1],
//! `x_idx` uint16 (int32 past 65,536 columns), `x_val` float16 - the non-zeros of each row's float16
//! cast, a -0.0 kept; 0: `x` float16 [N, F]), `player` int8, `n_players` int8, `opt_off` int32 [N+1],
//! `opt_action` in the narrowest of int8 / int16 / int32 / int64, `pi` float16, `z` / `q` / `sc`
//! float16 [N, P] (P = the most players in the shard), `net` int32, and `aux` float16 [N, W] when the
//! samples carry auxiliary targets. Meta: the writer's keys + `rows` (+ `x_version` 1, `x_cols`,
//! `x_nnz` for sparse). Float16 casts round to nearest even, as numpy's `astype(np.float16)`.
//!
//! A `ShardWriter` takes finished games' samples in order and writes a shard once it holds at least
//! `rows` samples (whole games), and the rest at `close` - data.py's cut points. A shard with a NaN /
//! inf after the casts is written as `<name>.bad.npz` and refused (`NonFinite`, as `write_shard`).
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use flate2::write::DeflateEncoder;
use flate2::Compression;
use half::f16;
use serde_json::{Map, Value};
use zero_search::driver::Record;
use zero_search::MAX_SEATS;

/// A numpy array to store: dtype descr, shape, little-endian bytes.
pub struct Npy {
    pub descr: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

fn shape_text(s: &[usize]) -> String {
    match s.len() {
        0 => "()".into(),
        1 => format!("({},)", s[0]),
        _ => format!("({})", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")),
    }
}

impl Npy {
    /// The `.npy` bytes (format 1.0, the header padded to 64 bytes, as numpy writes).
    pub fn bytes(&self) -> Vec<u8> {
        let mut h = format!("{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}", self.descr, shape_text(&self.shape));
        let total = 10 + h.len() + 1;
        h.push_str(&" ".repeat((64 - total % 64) % 64));
        h.push('\n');
        let mut out = Vec::with_capacity(10 + h.len() + self.data.len());
        out.extend_from_slice(b"\x93NUMPY\x01\x00");
        out.extend_from_slice(&(h.len() as u16).to_le_bytes());
        out.extend_from_slice(h.as_bytes());
        out.extend_from_slice(&self.data);
        out
    }

    pub fn f16(v: &[f16], shape: Vec<usize>) -> Npy {
        Npy { descr: "<f2".into(), shape, data: v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect() }
    }
    pub fn i8(v: &[i8], shape: Vec<usize>) -> Npy {
        Npy { descr: "|i1".into(), shape, data: v.iter().map(|x| *x as u8).collect() }
    }
    pub fn i16(v: &[i16], shape: Vec<usize>) -> Npy {
        Npy { descr: "<i2".into(), shape, data: v.iter().flat_map(|x| x.to_le_bytes()).collect() }
    }
    pub fn u16(v: &[u16], shape: Vec<usize>) -> Npy {
        Npy { descr: "<u2".into(), shape, data: v.iter().flat_map(|x| x.to_le_bytes()).collect() }
    }
    pub fn u32(v: &[u32], shape: Vec<usize>) -> Npy {
        Npy { descr: "<u4".into(), shape, data: v.iter().flat_map(|x| x.to_le_bytes()).collect() }
    }
    pub fn i32(v: &[i32], shape: Vec<usize>) -> Npy {
        Npy { descr: "<i4".into(), shape, data: v.iter().flat_map(|x| x.to_le_bytes()).collect() }
    }
    pub fn i64(v: &[i64], shape: Vec<usize>) -> Npy {
        Npy { descr: "<i8".into(), shape, data: v.iter().flat_map(|x| x.to_le_bytes()).collect() }
    }
    /// `np.array(s)`: a 0-d unicode array (UTF-32).
    pub fn unicode(s: &str) -> Npy {
        let n = s.chars().count();
        Npy { descr: format!("<U{}", n.max(1)), shape: vec![], data: if n == 0 { vec![0; 4] } else { s.chars().flat_map(|c| (c as u32).to_le_bytes()).collect() } }
    }
}

/// `_narrow_int`: the narrowest of int8 / int16 / int32 / int64 holding every value (empty: int8).
fn narrow(v: &[i64], shape: Vec<usize>) -> Npy {
    let (lo, hi) = (v.iter().copied().min().unwrap_or(0), v.iter().copied().max().unwrap_or(0));
    if lo >= i8::MIN as i64 && hi <= i8::MAX as i64 {
        Npy::i8(&v.iter().map(|&x| x as i8).collect::<Vec<_>>(), shape)
    } else if lo >= i16::MIN as i64 && hi <= i16::MAX as i64 {
        Npy::i16(&v.iter().map(|&x| x as i16).collect::<Vec<_>>(), shape)
    } else if lo >= i32::MIN as i64 && hi <= i32::MAX as i64 {
        Npy::i32(&v.iter().map(|&x| x as i32).collect::<Vec<_>>(), shape)
    } else {
        Npy::i64(v, shape)
    }
}

/// A zip of `members` (deflate at `level`; no zip64: members and the file stay under 4 GiB) at `path`.
pub fn write_npz(path: &Path, members: &[(&str, Npy)], level: u32) -> Result<u64, String> {
    let f = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut w = BufWriter::new(f);
    let mut central = Vec::new();
    let mut off: u64 = 0;
    let ioe = |e: std::io::Error| format!("{}: {e}", path.display());
    for (name, a) in members {
        let raw = a.bytes();
        let crc = crc32fast::hash(&raw);
        let mut enc = DeflateEncoder::new(Vec::new(), Compression::new(level));
        enc.write_all(&raw).map_err(ioe)?;
        let comp = enc.finish().map_err(ioe)?;
        let fname = format!("{name}.npy");
        if raw.len() as u64 >= u32::MAX as u64 || comp.len() as u64 >= u32::MAX as u64 || off >= u32::MAX as u64 {
            return Err(format!("{}: member {fname} over 4 GiB (zip64 not written)", path.display()));
        }
        let mut lh = Vec::new();
        lh.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        lh.extend_from_slice(&20u16.to_le_bytes()); // version needed
        lh.extend_from_slice(&0u16.to_le_bytes()); // flags
        lh.extend_from_slice(&8u16.to_le_bytes()); // deflate
        lh.extend_from_slice(&0u16.to_le_bytes()); // time
        lh.extend_from_slice(&0x21u16.to_le_bytes()); // date 1980-01-01
        lh.extend_from_slice(&crc.to_le_bytes());
        lh.extend_from_slice(&(comp.len() as u32).to_le_bytes());
        lh.extend_from_slice(&(raw.len() as u32).to_le_bytes());
        lh.extend_from_slice(&(fname.len() as u16).to_le_bytes());
        lh.extend_from_slice(&0u16.to_le_bytes());
        lh.extend_from_slice(fname.as_bytes());
        w.write_all(&lh).map_err(ioe)?;
        w.write_all(&comp).map_err(ioe)?;
        let mut ch = Vec::new();
        ch.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        ch.extend_from_slice(&20u16.to_le_bytes()); // made by
        ch.extend_from_slice(&20u16.to_le_bytes());
        ch.extend_from_slice(&0u16.to_le_bytes());
        ch.extend_from_slice(&8u16.to_le_bytes());
        ch.extend_from_slice(&0u16.to_le_bytes());
        ch.extend_from_slice(&0x21u16.to_le_bytes());
        ch.extend_from_slice(&crc.to_le_bytes());
        ch.extend_from_slice(&(comp.len() as u32).to_le_bytes());
        ch.extend_from_slice(&(raw.len() as u32).to_le_bytes());
        ch.extend_from_slice(&(fname.len() as u16).to_le_bytes());
        ch.extend_from_slice(&[0u8; 8]); // extra len, comment len, disk, internal attrs
        ch.extend_from_slice(&0u32.to_le_bytes()); // external attrs
        ch.extend_from_slice(&(off as u32).to_le_bytes());
        ch.extend_from_slice(fname.as_bytes());
        central.extend_from_slice(&ch);
        off += (lh.len() + comp.len()) as u64;
    }
    let mut end = Vec::new();
    end.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    end.extend_from_slice(&[0u8; 4]);
    end.extend_from_slice(&(members.len() as u16).to_le_bytes());
    end.extend_from_slice(&(members.len() as u16).to_le_bytes());
    end.extend_from_slice(&(central.len() as u32).to_le_bytes());
    end.extend_from_slice(&(off as u32).to_le_bytes());
    end.extend_from_slice(&0u16.to_le_bytes());
    w.write_all(&central).map_err(ioe)?;
    w.write_all(&end).map_err(ioe)?;
    w.flush().map_err(ioe)?;
    Ok(off + central.len() as u64 + end.len() as u64)
}

/// Shard format v2 of a game with variable-length rows (zero-core data.py `rows_meta`, docs/az.md 43.1): its tail
/// segments (segment, name, width, head slots), pointer overrides, row limit; the shard's meta keys `t_segs`,
/// `option_pointers`, `max_row_tokens` and its members `t_ptr` .. `ov_tok`.
#[derive(Clone, Debug)]
pub struct RowsFmt {
    pub tails: Vec<(u16, String, usize, usize)>,
    pub option_pointers: bool,
    pub max_row_tokens: Option<usize>,
}

impl RowsFmt {
    /// A game's format v2, None for a game without tail segments or pointer overrides (the format of before).
    pub fn of(info: &zero_search::dynapi::GameInfo) -> Option<RowsFmt> {
        if info.tails.is_empty() && !info.option_pointers {
            return None;
        }
        Some(RowsFmt { tails: info.tails.iter().map(|t| (t.seg, t.name.to_string(), t.width, t.head)).collect(), option_pointers: info.option_pointers, max_row_tokens: info.max_row_tokens })
    }
    /// The meta keys of `shard_format(cfg, info)`, in its order.
    pub fn meta(&self) -> Vec<(&'static str, Value)> {
        vec![
            ("t_segs", Value::from(self.tails.iter().map(|(s, n, w, h)| serde_json::json!([s, n, w, h])).collect::<Vec<_>>())),
            ("option_pointers", Value::from(self.option_pointers)),
            ("max_row_tokens", self.max_row_tokens.map_or(Value::Null, Value::from)),
        ]
    }
}

/// One finished game's samples.
pub struct GameSamples {
    pub gid: u32,
    pub records: Vec<Record>,
}

/// The shard arrays of `games`' samples (data.py's `compact` / `compact_sparse`; with `v2`, format v2's members
/// after them, `compact_rows`) -> (members without meta, meta additions, non-finite problems). Records with tail
/// tokens or overrides need `v2` (and sparse) - else Err: never dropped in silence (data.py R8).
#[allow(clippy::type_complexity)]
pub fn shard_arrays(games: &[GameSamples], f: usize, sparse: bool, v2: Option<&RowsFmt>) -> Result<(Vec<(&'static str, Npy)>, Map<String, Value>, Vec<String>), String> {
    let recs: Vec<&Record> = games.iter().flat_map(|g| g.records.iter()).collect();
    if v2.is_some() && !sparse {
        return Err("tail tokens / pointer overrides need a sparse shard (shard_sparse 1; zero-core docs/az.md 43.1, R8)".into());
    }
    if v2.is_none() && recs.iter().any(|r| !r.tail.seg.is_empty() || !r.ov.is_empty()) {
        return Err("samples with tail tokens / pointer overrides for a shard of format 1: they would be dropped (zero-core data.py R8)".into());
    }
    let n = recs.len();
    let mut out: Vec<(&'static str, Npy)> = Vec::new();
    let mut extra = Map::new();
    let mut probs = Vec::new();
    let bad = |v: &[f16]| v.iter().position(|x| !x.is_finite());
    if sparse {
        let mut ptr = Vec::with_capacity(n + 1);
        ptr.push(0i64);
        let mut idx: Vec<u32> = Vec::new();
        let mut val: Vec<f16> = Vec::new();
        for r in &recs {
            for (i, &x) in r.x.iter().enumerate() {
                let h = f16::from_f32(x);
                if h.to_bits() != 0 {
                    idx.push(i as u32);
                    val.push(h);
                }
            }
            ptr.push(idx.len() as i64);
        }
        if let Some(k) = bad(&val) {
            let row = ptr.partition_point(|&p| p <= k as i64) - 1;
            probs.push(format!("x: non-finite value at row {row}, column {}", idx[k]));
        }
        extra.insert("x_version".into(), Value::from(if v2.is_some() { 2 } else { 1 }));
        extra.insert("x_cols".into(), Value::from(f));
        extra.insert("x_nnz".into(), Value::from(val.len()));
        let nnz = val.len();
        out.push(("x_ptr", Npy::i64(&ptr, vec![n + 1])));
        if f <= 1 << 16 {
            out.push(("x_idx", Npy::u16(&idx.iter().map(|&i| i as u16).collect::<Vec<_>>(), vec![nnz])));
        } else {
            out.push(("x_idx", Npy::i32(&idx.iter().map(|&i| i as i32).collect::<Vec<_>>(), vec![nnz])));
        }
        out.push(("x_val", Npy::f16(&val, vec![nnz])));
    } else {
        let x: Vec<f16> = recs.iter().flat_map(|r| r.x.iter().map(|&v| f16::from_f32(v))).collect();
        if let Some(k) = bad(&x) {
            probs.push(format!("x: non-finite value at row {}, column {}", k / f.max(1), k % f.max(1)));
        }
        out.push(("x", Npy::f16(&x, vec![n, f])));
    }
    let p = recs.iter().map(|r| r.n_players as usize).max().unwrap_or(1);
    out.push(("player", Npy::i8(&recs.iter().map(|r| r.player as i8).collect::<Vec<_>>(), vec![n])));
    out.push(("n_players", Npy::i8(&recs.iter().map(|r| r.n_players as i8).collect::<Vec<_>>(), vec![n])));
    let mut off = Vec::with_capacity(n + 1);
    off.push(0i32);
    let mut acts = Vec::new();
    let mut pi = Vec::new();
    for r in &recs {
        acts.extend(r.opts.iter().map(|&a| a as i64));
        pi.extend(r.pi.iter().map(|&v| f16::from_f32(v)));
        off.push(acts.len() as i32);
    }
    out.push(("opt_off", Npy::i32(&off, vec![n + 1])));
    out.push(("opt_action", narrow(&acts, vec![acts.len()])));
    if let Some(k) = bad(&pi) {
        probs.push(format!("pi: non-finite value at option {k}"));
    }
    out.push(("pi", Npy::f16(&pi, vec![pi.len()])));
    for (name, get) in [("z", 0usize), ("q", 1), ("sc", 2)] {
        let v: Vec<f16> = recs
            .iter()
            .flat_map(|r| {
                let a: &[f32; MAX_SEATS] = match get {
                    0 => &r.z,
                    1 => &r.q,
                    _ => &r.sc,
                };
                a[..p].iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>()
            })
            .collect();
        if let Some(k) = bad(&v) {
            probs.push(format!("{name}: non-finite value at row {}", k / p));
        }
        out.push((name, Npy::f16(&v, vec![n, p])));
    }
    out.push(("net", Npy::i32(&recs.iter().map(|r| r.net as i32).collect::<Vec<_>>(), vec![n])));
    let aw = recs.first().map_or(0, |r| r.aux.len());
    if aw > 0 {
        let v: Vec<f16> = recs.iter().flat_map(|r| r.aux.iter().map(|&x| f16::from_f32(x))).collect();
        if let Some(k) = bad(&v) {
            probs.push(format!("aux: non-finite value at row {}", k / aw));
        }
        out.push(("aux", Npy::f16(&v, vec![n, aw])));
    }
    if let Some(fmt) = v2 {
        // format v2 (data.py `compact_rows`): t_ptr int64 [N+1], t_seg uint16, t_slot uint32, t_val float16 (each
        // token's floats), ov_ptr int64 [N+1], ov_opt int32, ov_tok int64 - empty CSRs for the kind the game lacks
        let (mut tp, mut ts, mut tl, mut tv) = (vec![0i64], Vec::new(), Vec::new(), Vec::new());
        let (mut op, mut oo, mut ot) = (vec![0i64], Vec::new(), Vec::new());
        for r in &recs {
            if !fmt.tails.is_empty() {
                ts.extend_from_slice(&r.tail.seg);
                tl.extend_from_slice(&r.tail.slot);
                tv.extend(r.tail.val.iter().map(|&x| f16::from_f32(x)));
            } else if !r.tail.seg.is_empty() {
                return Err("samples with tail tokens for a game without tail segments".into());
            }
            tp.push(ts.len() as i64);
            if fmt.option_pointers {
                oo.extend(r.ov.iter().map(|o| o.0 as i32));
                ot.extend(r.ov.iter().map(|o| o.1));
            } else if !r.ov.is_empty() {
                return Err("samples with pointer overrides for a game without them".into());
            }
            op.push(oo.len() as i64);
        }
        if let Some(k) = bad(&tv) {
            probs.push(format!("t_val: non-finite tail float {k}"));
        }
        out.push(("t_ptr", Npy::i64(&tp, vec![n + 1])));
        out.push(("t_seg", Npy::u16(&ts, vec![ts.len()])));
        out.push(("t_slot", Npy::u32(&tl, vec![tl.len()])));
        out.push(("t_val", Npy::f16(&tv, vec![tv.len()])));
        out.push(("ov_ptr", Npy::i64(&op, vec![n + 1])));
        out.push(("ov_opt", Npy::i32(&oo, vec![oo.len()])));
        out.push(("ov_tok", Npy::i64(&ot, vec![ot.len()])));
    }
    Ok((out, extra, probs))
}

/// A written shard.
#[derive(Clone, Debug)]
pub struct Written {
    pub seq: usize,
    pub path: PathBuf,
    pub rows: usize,
    pub bytes: u64,
    pub games: usize,
    pub gids: Vec<u32>,
    /// The oldest network id of its rows (PROTOCOL.md 5's `gen`: the oldest checkpoint generation its games
    /// were played with).
    pub gen: Option<u32>,
}

pub struct ShardWriter {
    dir: PathBuf,
    prefix: String,
    rows: usize,
    meta: Map<String, Value>,
    sparse: bool,
    features: usize,
    level: u32,
    /// Format v2 (a game with variable-length rows), None: the format of before.
    v2: Option<RowsFmt>,
    buf: Vec<GameSamples>,
    nbuf: usize,
    /// Games added since the last shard that had no samples (counted with the next shard).
    empty_games: Vec<u32>,
    pub seq: usize,
    pub written: Vec<Written>,
}

impl ShardWriter {
    /// `v2`: format v2 for a game with variable-length rows (`RowsFmt::of`; refused with dense shards, R8).
    #[allow(clippy::too_many_arguments)]
    pub fn new(dir: &Path, prefix: &str, rows: usize, meta: Map<String, Value>, sparse: bool, features: usize, level: u32, v2: Option<RowsFmt>) -> Result<ShardWriter, String> {
        if v2.is_some() && !sparse {
            return Err("a game with tail tokens / pointer overrides: its shards are sparse only (shard_sparse 1; zero-core docs/az.md 43.1, R8)".into());
        }
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        Ok(ShardWriter { dir: dir.to_path_buf(), prefix: prefix.to_string(), rows: rows.max(1), meta, sparse, features, level, v2, buf: Vec::new(), nbuf: 0, empty_games: Vec::new(), seq: 0, written: Vec::new() })
    }

    /// A finished game's samples (data.py `ShardWriter.add`: a shard at `rows` samples) -> the shard
    /// written, if any.
    pub fn add(&mut self, g: GameSamples) -> Result<Option<Written>, String> {
        if g.records.is_empty() {
            self.empty_games.push(g.gid);
            return Ok(None);
        }
        self.nbuf += g.records.len();
        self.buf.push(g);
        if self.nbuf >= self.rows {
            return self.flush();
        }
        Ok(None)
    }

    pub fn flush(&mut self) -> Result<Option<Written>, String> {
        if self.buf.is_empty() {
            return Ok(None);
        }
        let games = std::mem::take(&mut self.buf);
        self.nbuf = 0;
        let (arrays, extra, probs) = shard_arrays(&games, self.features, self.sparse, self.v2.as_ref())?;
        let rows: usize = games.iter().map(|g| g.records.len()).sum();
        let mut meta = self.meta.clone();
        if self.sparse {
            meta.insert("x_format".into(), Value::String("sparse".into()));
        }
        if let Some(fmt) = &self.v2 {
            // (`shard_format(cfg, info)`'s keys after x_format)
            for (k, v) in fmt.meta() {
                meta.insert(k.into(), v);
            }
        }
        meta.insert("rows".into(), Value::from(rows));
        meta.extend(extra);
        let name = format!("{}_{:03}", self.prefix, self.seq);
        let path = self.dir.join(format!("{name}.npz"));
        let mut gids: Vec<u32> = games.iter().map(|g| g.gid).collect();
        gids.extend(std::mem::take(&mut self.empty_games));
        let ngames = gids.len();
        let mut members = vec![("meta", Npy::unicode(&Value::Object(meta.clone()).to_string()))];
        if !probs.is_empty() {
            let bad = self.dir.join(format!("{name}.bad.npz"));
            meta.insert("nonfinite".into(), Value::from(probs.clone()));
            members[0] = ("meta", Npy::unicode(&Value::Object(meta).to_string()));
            members.extend(arrays);
            write_npz(&bad, &members, self.level)?;
            return Err(format!("NonFinite: non-finite sample values in shard {name}: {}. The samples were kept in {} (never loaded by the window)", probs.join("; "), bad.display()));
        }
        members.extend(arrays);
        let tmp = self.dir.join(format!("{name}.{}.tmp.npz", std::process::id()));
        let bytes = write_npz(&tmp, &members, self.level)?;
        std::fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
        let gen = games.iter().flat_map(|g| g.records.iter().map(|r| r.net)).min();
        let w = Written { seq: self.seq, path, rows, bytes, games: ngames, gids, gen };
        self.seq += 1;
        self.written.push(w.clone());
        Ok(Some(w))
    }

    /// Games added that are in no shard yet.
    pub fn pending_games(&self) -> usize {
        self.buf.len() + self.empty_games.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npy_header_is_aligned() {
        let a = Npy::i32(&[1, 2, 3], vec![3]).bytes();
        let hl = u16::from_le_bytes([a[8], a[9]]) as usize;
        assert_eq!((10 + hl) % 64, 0);
        assert!(std::str::from_utf8(&a[10..10 + hl]).unwrap().starts_with("{'descr': '<i4', 'fortran_order': False, 'shape': (3,), }"));
        assert_eq!(&a[10 + hl..], &[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]);
    }

    #[test]
    fn narrowest_int() {
        assert_eq!(narrow(&[], vec![0]).descr, "|i1");
        assert_eq!(narrow(&[-128, 127], vec![2]).descr, "|i1");
        assert_eq!(narrow(&[300], vec![1]).descr, "<i2");
        assert_eq!(narrow(&[70000], vec![1]).descr, "<i4");
    }

    #[test]
    fn f16_rounds_to_nearest_even() {
        // 2049 is halfway between 2048 and 2050 (f16 spacing 2 there): even mantissa 2048
        assert_eq!(f16::from_f32(2049.0).to_f32(), 2048.0);
        assert_eq!(f16::from_f32(2051.0).to_f32(), 2052.0);
        assert_eq!(f16::from_f32(-0.0).to_bits(), 0x8000);
        assert_eq!(f16::from_f32(1e-9).to_bits(), 0);
        assert!(f16::from_f32(70000.0).is_infinite());
    }
}
