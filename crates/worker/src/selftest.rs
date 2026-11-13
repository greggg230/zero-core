//! The per-machine self-test (zero-farm DESIGN.md 2.1 point 5; PROTOCOL.md 5 `selftest`): before a
//! worker plays with a network it answers the checkpoint's reference set - real network batch rows the
//! exporter stored with torch's answers (`zero.onnx_export --reference N`, `reference.npz`) - through the
//! very evaluator it will play with (this PC, this ONNX Runtime, this execution provider) and must
//! match within [`TOL`] (the parity gate's 2e-5) on every number: each row's priors over its legal
//! options, value, margin and the belief rows' auxiliary answers. A worker whose evaluator fails it
//! never plays: the job ends with a fatal error naming the difference.
//!
//! `reference.npz`: `meta` (JSON: zero_onnx 1, game, features, actions, aux_width, checkpoint,
//! checkpoint_sha256, rows, tol, the torch side), the rows as the driver sends them (`xs_off` i64 [R+1],
//! `xs_idx` i32, `xs_val` f32, `n_players` i64 [R], `extra` u8 [R] - ROW_BELIEF rows want the auxiliary
//! answer -, `opt_off` i64 [R+1], `opt_action` i32), `batch_off` i64 [K+1] (the rows' batches, as
//! recorded), and torch's answers: `priors` f32 [options], `value` f32 [R, 8], `margin` f32 [R, 8],
//! `aux` f32 [belief rows, W].
use std::path::Path;

use serde_json::Value;
use zero_search::driver::{Batch, ROW_BELIEF};

use crate::eval::{Answer, Evaluator, Expect};
use crate::npz::read_npz;

/// The largest difference allowed on any number (the parity gate's, DESIGN.md 8.3).
pub const TOL: f64 = 2e-5;

/// A checkpoint's reference set.
pub struct Reference {
    pub meta: Value,
    pub batches: Vec<Batch>,
    pub want: Answer,
    pub rows: usize,
}

/// Load `reference.npz` (every array checked against the others' shapes).
pub fn load(path: &Path) -> Result<Reference, String> {
    let z = read_npz(path).map_err(|e| format!("reference {}: {e}", path.display()))?;
    let g = |k: &str| z.get(k).ok_or_else(|| format!("reference {}: no {k}", path.display()));
    let meta: Value = serde_json::from_str(&g("meta")?.text()?).map_err(|e| format!("reference meta: {e}"))?;
    let (xs_off, xs_idx, xs_val) = (g("xs_off")?.i64s()?, g("xs_idx")?.i64s()?, g("xs_val")?.f32s()?);
    let (np_, extra, opt_off, opt_action, bo) = (g("n_players")?.i64s()?, g("extra")?.i64s()?, g("opt_off")?.i64s()?, g("opt_action")?.i64s()?, g("batch_off")?.i64s()?);
    let want = Answer { priors: g("priors")?.f32s()?, value: g("value")?.f32s()?, margin: g("margin")?.f32s()?, aux: g("aux")?.f32s()? };
    let rows = np_.len();
    let bad = |what: &str| Err(format!("reference {}: {what}", path.display()));
    if rows == 0 || xs_off.len() != rows + 1 || opt_off.len() != rows + 1 || extra.len() != rows {
        return bad("the rows' arrays disagree");
    }
    if *xs_off.last().unwrap() as usize != xs_idx.len() || xs_idx.len() != xs_val.len() || *opt_off.last().unwrap() as usize != opt_action.len() {
        return bad("the offsets disagree with the arrays");
    }
    if xs_off.windows(2).any(|w| w[0] > w[1]) || opt_off.windows(2).any(|w| w[0] > w[1]) || xs_off[0] != 0 || opt_off[0] != 0 {
        return bad("offsets must start at 0 and never decrease");
    }
    if bo.first() != Some(&0) || bo.last().copied() != Some(rows as i64) || bo.windows(2).any(|w| w[0] >= w[1]) {
        return bad("batch_off must cover the rows in non-empty batches");
    }
    let n_belief = extra.iter().filter(|&&e| e == ROW_BELIEF as i64).count();
    let ms = zero_search::MAX_SEATS;
    if want.priors.len() != opt_action.len() || want.value.len() != rows * ms || want.margin.len() != rows * ms {
        return bad("torch's answers disagree with the rows");
    }
    if (n_belief == 0 && !want.aux.is_empty()) || (n_belief > 0 && want.aux.len() % n_belief != 0) {
        return bad("the auxiliary answers disagree with the belief rows");
    }
    let mut batches = Vec::new();
    for k in 0..bo.len() - 1 {
        let (r0, r1) = (bo[k] as usize, bo[k + 1] as usize);
        let mut b = Batch::default();
        b.id = k as u64;
        b.b = r1 - r0;
        b.xs_off = xs_off[r0..=r1].iter().map(|&x| x - xs_off[r0]).collect();
        b.xs_idx = xs_idx[xs_off[r0] as usize..xs_off[r1] as usize].iter().map(|&x| x as u32).collect();
        b.xs_val = xs_val[xs_off[r0] as usize..xs_off[r1] as usize].to_vec();
        b.n_players = np_[r0..r1].iter().map(|&x| x as i32).collect();
        b.opt_off = opt_off[r0..=r1].iter().map(|&x| (x - opt_off[r0]) as i32).collect();
        b.opt_action = opt_action[opt_off[r0] as usize..opt_off[r1] as usize].iter().map(|&x| x as i32).collect();
        b.extra = extra[r0..r1].iter().map(|&x| x as u8).collect();
        b.player = vec![0; r1 - r0];
        b.gid = vec![0; r1 - r0];
        batches.push(b);
    }
    Ok(Reference { meta, batches, want, rows })
}

/// The reference set belongs to the network and the game the job plays: its game / features / actions
/// are the job's, and its checkpoint's sha256 is the graph's (`checkpoint_sha256` metadata), when both
/// say one.
pub fn check_meta(r: &Reference, want: &Expect, graph_meta: &[(String, String)]) -> Result<(), String> {
    let m = &r.meta;
    let s = |k: &str| m.get(k).map(|v| v.as_str().map(String::from).unwrap_or_else(|| v.to_string()));
    if let Some(g) = s("game") {
        if g != want.game {
            return Err(format!("reference: game {g:?}, the job plays {:?}", want.game));
        }
    }
    for (k, n) in [("features", want.features), ("actions", want.actions)] {
        if let Some(v) = s(k) {
            if v != n.to_string() {
                return Err(format!("reference: {k} {v}, the game has {n}"));
            }
        }
    }
    let graph_sha = graph_meta.iter().find(|(k, _)| k == "checkpoint_sha256").map(|(_, v)| v.clone());
    if let (Some(a), Some(b)) = (s("checkpoint_sha256"), graph_sha) {
        if a != b {
            return Err(format!("reference: made for checkpoint {a}, the network is {b}"));
        }
    }
    Ok(())
}

/// The self-test's result.
#[derive(Clone, Debug)]
pub struct SelfTest {
    /// The largest |evaluator - torch| over every number (inf if anything is not finite or a size differs).
    pub max_abs: f64,
    /// Per item: priors, value, margin, aux.
    pub items: [f64; 4],
    pub rows: usize,
    pub ok: bool,
}

fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| {
        let d = (*x as f64 - *y as f64).abs();
        if d.is_nan() || !x.is_finite() || !y.is_finite() {
            f64::INFINITY
        } else {
            m.max(d)
        }
    })
}

/// The largest |got - want| per item (priors, value, margin, aux; inf when not finite or a size differs).
pub fn compare(got: &Answer, want: &Answer) -> [f64; 4] {
    [max_abs(&got.priors, &want.priors), max_abs(&got.value, &want.value), max_abs(&got.margin, &want.margin), max_abs(&got.aux, &want.aux)]
}

/// Answer the reference set with `ev` (batch by batch, as the driver's batches are answered) and compare.
pub fn run(ev: &mut dyn Evaluator, r: &Reference) -> Result<SelfTest, String> {
    let mut got = Answer::default();
    for b in &r.batches {
        let a = ev.evaluate(b).map_err(|e| format!("selftest: {e}"))?;
        got.priors.extend(a.priors);
        got.value.extend(a.value);
        got.margin.extend(a.margin);
        got.aux.extend(a.aux);
    }
    let items = compare(&got, &r.want);
    let m = items.iter().cloned().fold(0.0f64, f64::max);
    Ok(SelfTest { max_abs: m, items, rows: r.rows, ok: m <= TOL })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shard::{write_npz, Npy};

    /// An evaluator that answers what it was told to (the reference's answers, maybe off).
    struct Fixed(Answer, f32);

    impl Evaluator for Fixed {
        fn evaluate(&mut self, b: &Batch) -> Result<Answer, String> {
            // (one batch: the whole answer, the first prior moved by the offset)
            assert_eq!(b.b, 3);
            let mut a = self.0.clone();
            a.priors[0] += self.1;
            Ok(a)
        }
        fn describe(&self) -> String {
            "fixed".into()
        }
    }

    fn arr<T: Copy>(descr: &str, shape: Vec<usize>, v: &[T], f: impl Fn(T) -> Vec<u8>) -> Npy {
        Npy { descr: descr.into(), shape, data: v.iter().flat_map(|&x| f(x)).collect() }
    }

    /// A 3-row reference set (a leaf row with 2 options, a value-only row, a belief row with aux width 2).
    fn write(path: &std::path::Path, xs_off: &[i64], priors: &[f32]) {
        let i8s = |v: &[i64]| arr("<i8", vec![v.len()], v, |x: i64| x.to_le_bytes().to_vec());
        let f4 = |v: &[f32], shape: Vec<usize>| arr("<f4", shape, v, |x: f32| x.to_le_bytes().to_vec());
        let value: Vec<f32> = (0..24).map(|i| i as f32 / 100.0).collect();
        write_npz(path, &[
            ("meta", Npy::unicode(r#"{"zero_onnx": "1", "game": "toy", "features": 5, "actions": 4, "checkpoint_sha256": "ab"}"#)),
            ("xs_off", i8s(xs_off)),
            ("xs_idx", arr("<i4", vec![3], &[0i32, 2, 4], |x: i32| x.to_le_bytes().to_vec())),
            ("xs_val", f4(&[1.0, 0.5, 2.0], vec![3])),
            ("n_players", i8s(&[2, 2, 2])),
            ("extra", arr("|u1", vec![3], &[0u8, 1, 2], |x: u8| vec![x])),
            ("opt_off", i8s(&[0, 2, 2, 2])),
            ("opt_action", arr("<i4", vec![2], &[1i32, 3], |x: i32| x.to_le_bytes().to_vec())),
            ("batch_off", i8s(&[0, 3])),
            ("priors", f4(priors, vec![priors.len()])),
            ("value", f4(&value, vec![3, 8])),
            ("margin", f4(&value, vec![3, 8])),
            ("aux", f4(&[0.25, 0.75], vec![1, 2])),
        ], 1)
        .unwrap();
    }

    #[test]
    fn a_reference_passes_within_the_tolerance_and_fails_past_it() {
        let d = std::env::temp_dir().join(format!("zw-selftest-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("reference.npz");
        write(&p, &[0, 1, 2, 3], &[0.25, 0.75]);
        let r = load(&p).unwrap();
        assert_eq!((r.rows, r.batches.len(), r.batches[0].xs_idx.clone()), (3, 1, vec![0, 2, 4]));
        let want = Expect { game: "toy".into(), features: 5, actions: 4, aux_width: 2, belief: true };
        check_meta(&r, &want, &[("checkpoint_sha256".into(), "ab".into())]).unwrap();
        assert!(check_meta(&r, &want, &[("checkpoint_sha256".into(), "cd".into())]).unwrap_err().contains("made for checkpoint ab"));
        assert!(check_meta(&r, &Expect { game: "mtg".into(), ..want.clone() }, &[]).is_err());
        let ok = run(&mut Fixed(r.want.clone(), 1e-6), &r).unwrap();
        assert!(ok.ok && (ok.max_abs - 1e-6).abs() < 1e-7, "{ok:?}");
        let bad = run(&mut Fixed(r.want.clone(), 1e-3), &r).unwrap();
        assert!(!bad.ok && bad.items[0] > TOL && bad.items[1] == 0.0, "{bad:?}");
        let nan = run(&mut Fixed(r.want.clone(), f32::NAN), &r).unwrap();
        assert!(!nan.ok && nan.max_abs.is_infinite());
        // a corrupted file: offsets that disagree with the arrays, answers of the wrong length
        write(&p, &[0, 1, 2, 4], &[0.25, 0.75]);
        assert!(load(&p).err().unwrap().contains("offsets"));
        write(&p, &[0, 1, 2, 3], &[0.25]);
        assert!(load(&p).err().unwrap().contains("answers"));
        let _ = std::fs::remove_dir_all(&d);
    }
}
