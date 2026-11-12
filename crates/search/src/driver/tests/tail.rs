//! Variable-length rows through the driver (docs/search.md 2.15): `toy_tail` (tail segments, a tail-only segment,
//! pointer overrides, pages) against a "row echo" evaluator whose answers depend on every float of the head, every
//! tail token and every override, so a key or a transport that dropped any of it would change the games.
use super::*;
use crate::tail::{pack_tok, unpack_tok};
use crate::toy::ToyGame;
use crate::toy_tail::{TailGame, HEAD, NUM_FEATURES as TF, W_ITEM};

fn seat(sims: u32, selfplay: bool) -> NnSeat {
    let mut s = NnSeat::eval(0, sims);
    s.selfplay = selfplay;
    if selfplay {
        s.temp_moves = 6;
    }
    s
}

fn tcfg(s: NnSeat, games: usize, spec: &str) -> DriverCfg {
    DriverCfg {
        spec: spec.into(),
        seats: vec![SeatCfg::Nn(s)],
        games: vec![vec![0, 0]; games],
        keys: vec![("m".into(), KeyKind::Nn)],
        concurrency: 8,
        seed: 11,
        record: true,
        threads: 2,
        groups: 2,
        max_steps: 10_000,
        rolling: false,
        net: 0,
        queue_only: false,
        move_log: 0,
        aux: false,
        paired: false,
        sparse: false,
        branch: None,
        starts: None,
        max_row_tokens: None,
    }
}

fn width(seg: u16) -> usize {
    TailGame::TAILS.iter().find(|s| s.seg == seg).unwrap().width
}

/// Row `r`'s dense head (from `x` or the sparse pairs).
fn head(b: &Batch, r: usize, f: usize) -> Vec<f32> {
    if b.xs_off.is_empty() {
        return b.x[r * f..(r + 1) * f].to_vec();
    }
    let mut x = vec![0.0; f];
    for k in b.xs_off[r] as usize..b.xs_off[r + 1] as usize {
        x[b.xs_idx[k] as usize] = b.xs_val[k];
    }
    x
}

/// Row `r`'s tail tokens: (seg, slot, floats).
fn tail_of(b: &Batch, r: usize) -> Vec<(u16, u32, Vec<f32>)> {
    let mut off: usize = b.t_seg[..b.t_off[r] as usize].iter().map(|&s| width(s)).sum();
    (b.t_off[r] as usize..b.t_off[r + 1] as usize)
        .map(|j| {
            let w = width(b.t_seg[j]);
            let v = b.t_val[off..off + w].to_vec();
            off += w;
            (b.t_seg[j], b.t_slot[j], v)
        })
        .collect()
}

fn ov_of(b: &Batch, r: usize) -> Vec<(u32, i64)> {
    (b.ov_off[r] as usize..b.ov_off[r + 1] as usize).map(|j| (b.ov_opt[j] as u32, b.ov_tok[j])).collect()
}

/// The row echo: a value from every head float, tail token and override; priors favouring overridden options.
fn echo(d: &mut Driver<TailGame>, b: &Batch) {
    let mut pri = Vec::new();
    let mut v = vec![0.0; b.b * MAX_SEATS];
    let m = vec![0.0; b.b * MAX_SEATS];
    for r in 0..b.b {
        let x = head(b, r, TF);
        let mut h: f32 = x.iter().enumerate().map(|(i, x)| x * ((i * 7 % 13) as f32 - 6.0)).sum();
        for (s, l, f) in tail_of(b, r) {
            h += (s as f32 * 1.7 + l as f32 * 0.3) * f.iter().enumerate().map(|(i, x)| x * ((i * 5 % 11) as f32 - 5.0)).sum::<f32>();
        }
        let ov = ov_of(b, r);
        for &(o, t) in &ov {
            h += o as f32 * 0.5 + (t & 0xff) as f32 * 0.25;
        }
        let k = (b.opt_off[r + 1] - b.opt_off[r]) as usize;
        let mut p: Vec<f32> = (0..k).map(|i| if ov.iter().any(|o| o.0 as usize == i) { 2.0 } else { 1.0 }).collect();
        let s: f32 = p.iter().sum();
        p.iter_mut().for_each(|x| *x /= s);
        pri.extend(p);
        let w = 1.0 / (1.0 + (-h * 0.1).exp());
        v[r * MAX_SEATS] = w;
        v[r * MAX_SEATS + 1] = 1.0 - w;
    }
    d.submit(b.id, &pri, &v, &m).unwrap();
}

/// Run to the end, checking every batch's tail and override arrays; -> results, samples, rows seen
/// (player, head, tail, overrides).
#[allow(clippy::type_complexity)]
fn run_tail(d: &mut Driver<TailGame>) -> (Vec<GameResult>, Vec<Record>, Vec<(i32, Vec<f32>, Vec<(u16, u32, Vec<f32>)>, Vec<(u32, i64)>)>) {
    let (mut res, mut smp, mut rows) = (Vec::new(), Vec::new(), Vec::new());
    while let Some(b) = d.next_batch(true).unwrap() {
        check_batch(&b);
        for r in 0..b.b {
            rows.push((b.player[r], head(&b, r, TF), tail_of(&b, r), ov_of(&b, r)));
        }
        echo(d, &b);
        res.extend(d.take_results());
        smp.extend(d.take_samples());
    }
    res.extend(d.take_results());
    smp.extend(d.take_samples());
    res.sort_by_key(|r| r.gid);
    (res, smp, rows)
}

/// The batch's CSRs are well formed and every override points at a token of its row.
fn check_batch(b: &Batch) {
    assert_eq!(b.t_off.len(), b.b + 1);
    assert_eq!(b.ov_off.len(), b.b + 1);
    assert_eq!(*b.t_off.last().unwrap() as usize, b.t_seg.len());
    assert_eq!(b.t_slot.len(), b.t_seg.len());
    assert_eq!(b.t_val.len(), b.t_seg.iter().map(|&s| width(s)).sum::<usize>());
    assert_eq!(*b.ov_off.last().unwrap() as usize, b.ov_opt.len());
    for r in 0..b.b {
        let t = tail_of(b, r);
        for w in t.windows(2) {
            assert!((w[0].0, w[0].1) < (w[1].0, w[1].1), "sorted, unique");
        }
        for (s, l, f) in &t {
            let h = TailGame::TAILS.iter().find(|x| x.seg == *s).unwrap().head;
            assert!(*l as usize >= h);
            assert_eq!(f[0], 1.0, "a tail token is present");
        }
        let nopt = (b.opt_off[r + 1] - b.opt_off[r]) as usize;
        let x = head(b, r, TF);
        let mut last = -1i64;
        for (o, tok) in ov_of(b, r) {
            assert!((o as i64) > last && (o as usize) < nopt);
            last = o as i64;
            let (s, c, l) = unpack_tok(tok);
            assert_eq!((s, c), (0, 0));
            if (l as usize) < HEAD {
                assert_eq!(x[8 + l as usize * W_ITEM], 1.0, "a present head slot");
            } else {
                assert!(t.iter().any(|z| z.0 == 0 && z.1 == l), "the target tail token is in the row");
            }
        }
        if nopt == 0 {
            assert!(ov_of(b, r).is_empty());
        }
    }
}

#[test]
fn toy_tail_rows_carry_tails_and_overrides() {
    let mut d = Driver::<TailGame>::new(tcfg(seat(24, true), 16, r#"{"items0": 12, "max_moves": 24}"#)).unwrap();
    let (res, smp, rows) = run_tail(&mut d);
    assert_eq!(res.len(), 16);
    assert!(res.iter().all(|r| r.over && !r.aborted && r.abort_reason.is_none()));
    let st = d.stats();
    assert!(st.tail_rows > 0 && st.tail_tokens >= st.tail_rows && st.tail_max >= 8, "{} {} {}", st.tail_rows, st.tail_tokens, st.tail_max);
    assert_eq!(st.aborted_row_tokens, 0);
    assert!(rows.iter().any(|r| !r.3.is_empty()), "some rows override");
    assert!(rows.iter().any(|r| r.2.iter().any(|t| t.0 == 1)), "some rows hold marks");
    // the records: the decider's row with its tail and overrides; the arrays carry them exactly
    assert!(!smp.is_empty());
    assert!(smp.iter().any(|s| !s.tail.is_empty()) && smp.iter().any(|s| !s.ov.is_empty()));
    for s in &smp {
        crate::driver::check_overrides(&s.ov, s.opts.len());
    }
    let a = sample_arrays_rows(&smp, TF, true, true);
    assert_eq!(a.t_off.len(), smp.len() + 1);
    assert_eq!(a.ov_off.len(), smp.len() + 1);
    assert_eq!(a.t_seg.len(), smp.iter().map(|s| s.tail.len()).sum::<usize>());
    assert_eq!(a.t_val, smp.iter().flat_map(|s| s.tail.val.iter().copied()).collect::<Vec<_>>());
    assert_eq!(a.ov_tok, smp.iter().flat_map(|s| s.ov.iter().map(|o| o.1)).collect::<Vec<_>>());
}

#[test]
#[should_panic(expected = "a record with tail tokens")]
fn head_only_sample_arrays_refuse_a_tail() {
    let r = Record { x: vec![0.0; 2], player: 0, n_players: 2, opts: vec![], pi: vec![], z: [0.0; MAX_SEATS], q: [0.0; MAX_SEATS], sc: [0.0; MAX_SEATS], net: 0, aux: vec![], tail: TailRow { seg: vec![0], slot: vec![4], val: vec![1.0; 16] }, ov: vec![] };
    sample_arrays(&[r], 2);
}

#[test]
fn dense_and_sparse_rows_are_the_same() {
    let mk = |sparse: bool| {
        let mut c = tcfg(seat(16, true), 12, r#"{"items0": 6}"#);
        c.sparse = sparse;
        let mut d = Driver::<TailGame>::new(c).unwrap();
        run_tail(&mut d)
    };
    let (r0, s0, w0) = mk(false);
    let (r1, s1, w1) = mk(true);
    let key = |r: &GameResult| (r.gid, r.steps, r.scores.clone(), r.samples);
    assert_eq!(r0.iter().map(key).collect::<Vec<_>>(), r1.iter().map(key).collect::<Vec<_>>());
    assert_eq!(w0, w1, "the same rows: heads, tails, overrides");
    assert_eq!(s0.len(), s1.len());
    for (a, b) in s0.iter().zip(&s1) {
        assert_eq!((&a.x, &a.tail, &a.ov), (&b.x, &b.tail, &b.ov));
    }
}

/// Results and samples of a run (`key` per game; the samples' rows as a multiset: the evaluation cache changes
/// when games finish, so the order their records come out in, never the records).
type Digest = (Vec<(u32, u64, Vec<f32>, u32)>, Vec<String>);

fn digest(res: &[GameResult], smp: &[Record]) -> Digest {
    let mut s: Vec<String> = smp.iter().map(|s| format!("{:?}", (&s.x, &s.tail, &s.ov, &s.pi, s.player, &s.z))).collect();
    s.sort();
    (res.iter().map(|r| (r.gid, r.steps, r.scores.clone(), r.samples)).collect(), s)
}

#[test]
fn the_evaluation_cache_keys_cover_tails_and_overrides() {
    // the echo answers depend on the tail: a key without it would hand one row's answer to another and change play
    let mk = |cache: u32, sparse: bool| {
        let mut s = seat(24, true);
        s.eval_cache = cache;
        let mut c = tcfg(s, 12, r#"{"items0": 9}"#);
        c.sparse = sparse;
        let mut d = Driver::<TailGame>::new(c).unwrap();
        let (r, s, _) = run_tail(&mut d);
        (digest(&r, &s), d.stats().cache_hits)
    };
    let (off, h0) = mk(0, false);
    let (on, h1) = mk(4096, false);
    let (on_sparse, h2) = mk(4096, true);
    assert_eq!(h0, 0);
    assert!(h1 > 0, "the cache answered rows");
    assert_eq!(off, on);
    assert_eq!(on, on_sparse);
    assert_eq!(h1, h2, "dense and sparse rows have equal keys");
}

#[test]
fn merged_trees_share_their_encoded_tails() {
    // information-set merging with the leaf rows taken from the merged tree's encoding (tail included) or encoded
    // again: the same rows, the same games
    let mk = |off: bool| {
        let mut s = seat(24, true);
        s.infoset_merge = true;
        s.leaf_value_view = LeafView::Root;
        let mut d = Driver::<TailGame>::new(tcfg(s, 8, r#"{"items0": 9}"#)).unwrap();
        d.sh.counters.share_off.store(off, Ordering::Relaxed);
        let (r, s, rows) = run_tail(&mut d);
        (digest(&r, &s), rows, d.stats().merge_shared)
    };
    let (a, ra, shared) = mk(false);
    let (b, rb, none) = mk(true);
    assert!(shared > 0, "rows were shared");
    assert_eq!(none, 0);
    assert_eq!(a, b);
    assert_eq!(ra, rb);
}

#[test]
fn a_row_over_the_limit_aborts_its_game() {
    // 70 items: 66 tail tokens, capacity 71 > 64 at the first row
    let mut d = Driver::<TailGame>::new(tcfg(seat(8, true), 6, r#"{"items0": 70}"#)).unwrap();
    let (res, smp, rows) = run_tail(&mut d);
    assert_eq!(res.len(), 6);
    assert!(res.iter().all(|r| r.aborted && !r.over && r.abort_reason == Some("max_row_tokens") && r.samples == 0));
    assert!(smp.is_empty() && rows.is_empty(), "no row over the limit is ever sent");
    assert_eq!(d.stats().aborted_row_tokens, 6);
    // a lower limit: 9 items = 5 tail tokens, capacity 10 > 9
    let mut c = tcfg(seat(8, true), 4, r#"{"items0": 9}"#);
    c.max_row_tokens = Some(9);
    let mut d = Driver::<TailGame>::new(c).unwrap();
    let (res, _, rows) = run_tail(&mut d);
    assert!(res.iter().all(|r| r.aborted));
    assert!(rows.iter().all(|r| 5 + r.2.len() <= 9));
    // a limit that lets the first rows through: the games that grow past it abort mid-game, the rows they sent
    // before are fine
    let mut c = tcfg(seat(8, true), 16, r#"{"items0": 8}"#);
    c.max_row_tokens = Some(12);
    let mut d = Driver::<TailGame>::new(c).unwrap();
    let (res, smp, rows) = run_tail(&mut d);
    assert!(rows.iter().all(|r| 5 + r.2.len() <= 12));
    let ab = res.iter().filter(|r| r.aborted).count();
    assert_eq!(d.stats().aborted_row_tokens as usize, ab);
    let kept: u32 = res.iter().filter(|r| !r.aborted).map(|r| r.samples).sum();
    assert_eq!(kept as usize, smp.len());
    // the limit may only be lowered, and never below the head
    let mut c = tcfg(seat(8, true), 1, "{}");
    c.max_row_tokens = Some(65);
    assert!(Driver::<TailGame>::new(c).err().unwrap().contains("above the game's 64"));
    let mut c = tcfg(seat(8, true), 1, "{}");
    c.max_row_tokens = Some(4);
    assert!(Driver::<TailGame>::new(c).err().unwrap().contains("below the game's head tokens 5"));
}

#[test]
fn the_lookahead_refuses_tail_rows() {
    let mut s = seat(8, false);
    s.look = Some(Box::new(crate::lookahead::LookCfg { plies: 1, worlds: 1, ..Default::default() }));
    let mut d = Driver::<TailGame>::new(tcfg(s, 2, r#"{"items0": 9}"#)).unwrap();
    let mut err = None;
    loop {
        match d.next_batch(true) {
            Ok(Some(b)) => echo(&mut d, &b),
            Ok(None) => break,
            Err(e) => {
                err = Some(e);
                break;
            }
        }
    }
    let e = err.expect("refused");
    assert!(e.contains("lookahead: tail rows not supported yet"), "{e}");
}

#[test]
fn a_game_without_tails_has_no_tail_arrays() {
    let mut c = tcfg(seat(8, true), 4, "{}");
    c.sparse = true;
    let mut d = Driver::<ToyGame>::new(c).unwrap();
    while let Some(b) = d.next_batch(true).unwrap() {
        assert!(b.t_off.is_empty() && b.t_seg.is_empty() && b.t_slot.is_empty() && b.t_val.is_empty());
        assert!(b.ov_off.is_empty() && b.ov_opt.is_empty() && b.ov_tok.is_empty());
        let k = b.opt_off[b.b] as usize;
        let pri: Vec<f32> = (0..b.b).flat_map(|r| {
            let n = (b.opt_off[r + 1] - b.opt_off[r]) as usize;
            std::iter::repeat_n(1.0 / n.max(1) as f32, n)
        }).collect();
        assert_eq!(pri.len(), k);
        d.submit(b.id, &pri, &vec![0.5; b.b * MAX_SEATS], &vec![0.0; b.b * MAX_SEATS]).unwrap();
    }
    let smp = d.take_samples();
    assert!(!smp.is_empty() && smp.iter().all(|s| s.tail.is_empty() && s.ov.is_empty()));
    let a = sample_arrays(&smp, crate::toy::NUM_FEATURES);
    assert!(!a.tails && !a.pointers && a.t_off.is_empty() && a.ov_off.is_empty());
    let st = d.stats();
    assert_eq!((st.tail_rows, st.tail_tokens, st.tail_max, st.aborted_row_tokens), (0, 0, 0, 0));
}

#[test]
fn tail_keys_extend_the_head_key() {
    use crate::lookahead::{obs_hash, obs_hash_row, obs_hash_sparse, obs_state, obs_state_sparse};
    let mut r = Rng::new(3);
    for _ in 0..200 {
        let f = 1 + r.below(90) as usize;
        let mut x = vec![0.0f32; f];
        for v in x.iter_mut() {
            if r.below(3) == 0 {
                *v = [1.0, -0.0, 0.5, f32::MIN_POSITIVE, -3.0][r.below(5) as usize];
            }
        }
        let (mut idx, mut val) = (Vec::new(), Vec::new());
        let mut y = x.clone();
        crate::game::sparse_scan(&mut y, &mut idx, &mut val);
        // K1: no tail, no overrides: today's key, both routes
        let k = obs_hash(&x);
        assert_eq!(k, obs_hash_sparse(&idx, &val, f));
        assert_eq!(k, obs_hash_row(obs_state(&x), f, &[], &[], &[], &[]));
        assert_eq!(k, obs_hash_row(obs_state_sparse(&idx, &val), f, &[], &[], &[], &[]));
        // K2: a tail or an override changes it, the same on both routes
        let seg = [0u16, 0, 1];
        let slot = [4u32, 9, 2];
        let tv: Vec<f32> = (0..2 * W_ITEM + 4).map(|i| if i % 3 == 0 { 0.0 } else { i as f32 }).collect();
        let ov = [(1u32, pack_tok(0, 0, 9)), (3, -1)];
        let kt = obs_hash_row(obs_state(&x), f, &seg, &slot, &tv, &ov);
        assert_eq!(kt, obs_hash_row(obs_state_sparse(&idx, &val), f, &seg, &slot, &tv, &ov));
        assert_ne!(kt, k);
        assert_ne!(kt, obs_hash_row(obs_state(&x), f, &seg, &slot, &tv, &ov[..1]));
        assert_ne!(kt, obs_hash_row(obs_state(&x), f, &seg, &slot, &tv, &[]));
        assert_ne!(kt, obs_hash_row(obs_state(&x), f, &seg, &slot, &tv, &[(1, pack_tok(0, 0, 9)), (3, pack_tok(0, 0, 0))]));
        assert_ne!(kt, obs_hash_row(obs_state(&x), f, &seg, &slot, &tv, &[(1, pack_tok(0, 0, 9)), (2, -1)]));
        assert_ne!(kt, obs_hash_row(obs_state(&x), f, &seg, &[4, 10, 2], &tv, &ov), "a slot");
        assert_ne!(kt, obs_hash_row(obs_state(&x), f, &[0, 1, 1], &slot, &tv, &ov), "a segment");
        assert_ne!(kt, obs_hash_row(obs_state(&x), f, &seg[..2], &slot[..2], &tv[..2 * W_ITEM], &ov), "a token fewer");
        for i in 0..tv.len() {
            for bit in [0u32, 22, 31] {
                let mut t2 = tv.clone();
                t2[i] = f32::from_bits(t2[i].to_bits() ^ (1 << bit));
                assert_ne!(kt, obs_hash_row(obs_state(&x), f, &seg, &slot, &t2, &ov), "float {i} bit {bit}");
            }
        }
    }
    // the coordinator's check: the same floats split differently across tokens of different segments (widths 1, 2)
    let a = obs_hash_row((1, 2), 8, &[0, 0, 0], &[1, 2, 3], &[1.0, 2.0, 3.0], &[]);
    let b = obs_hash_row((1, 2), 8, &[0, 2], &[1, 1], &[1.0, 2.0, 3.0], &[]);
    let c = obs_hash_row((1, 2), 8, &[0, 2], &[1, 2], &[1.0, 2.0, 3.0], &[]);
    assert!(a != b && b != c && a != c);
}
