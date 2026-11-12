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

/// A lookahead's rows of one round with their tails and overrides (`RowSink`, docs/search.md 2.15).
#[derive(Default)]
struct LookRowsT {
    x: Vec<f32>,
    opts: Vec<Vec<u16>>,
    kind: Vec<u8>,
    tail: Vec<Vec<(u16, u32, Vec<f32>)>>,
    ov: Vec<Vec<(u32, i64)>>,
}

impl crate::lookahead::RowSink<TailGame> for LookRowsT {
    fn push_row(&mut self, _viewer: usize, opts: &[u16], kind: u8) -> &mut [f32] {
        self.opts.push(opts.to_vec());
        self.kind.push(kind);
        self.tail.push(Vec::new());
        self.ov.push(Vec::new());
        let off = self.x.len();
        self.x.resize(off + TF, 0.0);
        &mut self.x[off..]
    }
    fn pop_row(&mut self) {
        self.opts.pop();
        self.kind.pop();
        self.tail.pop();
        self.ov.pop();
        self.x.truncate(self.x.len() - TF);
    }
    fn row_tail(&mut self, tail: &crate::tail::TailBuf, ov: &[(u32, i64)]) -> Result<(), String> {
        let mut off = 0;
        let t = tail.seg().iter().zip(tail.slot()).map(|(&s, &l)| {
            let w = width(s);
            off += w;
            (s, l, tail.val()[off - w..off].to_vec())
        });
        *self.tail.last_mut().unwrap() = t.collect();
        *self.ov.last_mut().unwrap() = ov.to_vec();
        Ok(())
    }
}

/// The row echo of one row (`echo`'s formula): its value for the row's player and its priors.
fn echo_row(x: &[f32], tail: &[(u16, u32, Vec<f32>)], ov: &[(u32, i64)], k: usize) -> (f32, Vec<f32>) {
    let mut h: f32 = x.iter().enumerate().map(|(i, x)| x * ((i * 7 % 13) as f32 - 6.0)).sum();
    for (s, l, f) in tail {
        h += (*s as f32 * 1.7 + *l as f32 * 0.3) * f.iter().enumerate().map(|(i, x)| x * ((i * 5 % 11) as f32 - 5.0)).sum::<f32>();
    }
    for &(o, t) in ov {
        h += o as f32 * 0.5 + (t & 0xff) as f32 * 0.25;
    }
    let mut p: Vec<f32> = (0..k).map(|i| if ov.iter().any(|o| o.0 as usize == i) { 2.0 } else { 1.0 }).collect();
    let s: f32 = p.iter().sum();
    p.iter_mut().for_each(|x| *x /= s);
    (1.0 / (1.0 + (-h * 0.1).exp()), p)
}

/// A lookahead of a toy_tail position run to its decision against the row echo -> (each root option's mean, the
/// stats, every round's rows).
fn look_tail(settings: &str, spec: &str, seed: u64) -> (Vec<(u16, f64)>, crate::lookahead::LookStats, Vec<LookRowsT>) {
    use crate::lookahead::{Answers, Look, LookCfg, Step};
    let g = TailGame::new_game(&TailGame::parse_spec(spec).unwrap(), 2, seed).unwrap();
    let p = g.to_move().unwrap();
    let mut l = Look::new(LookCfg::parse(settings).unwrap(), &g, p, None, Vec::new(), vec![1, 2, 3, 4], false).unwrap();
    let mut rng = Rng::new(0);
    let mut rounds = Vec::new();
    loop {
        let mut rows = LookRowsT::default();
        match l.step(&mut rows, &mut rng).unwrap() {
            Step::Done => break,
            Step::Rows(n) => {
                assert_eq!(n, rows.kind.len());
                let (mut pri, mut val, mut opt_off) = (Vec::new(), vec![0.0; n * MAX_SEATS], vec![0i32]);
                for i in 0..n {
                    let (v, pr) = echo_row(&rows.x[i * TF..(i + 1) * TF], &rows.tail[i], &rows.ov[i], rows.opts[i].len());
                    pri.extend(pr);
                    opt_off.push(pri.len() as i32);
                    val[i * MAX_SEATS] = v;
                    val[i * MAX_SEATS + 1] = 1.0 - v;
                }
                l.answer(&Answers { base: 0, opt_off: &opt_off, priors: &pri, value: &val, aux: &[], aux_row: &[], aux_width: 0 }).unwrap();
                rounds.push(rows);
            }
        }
    }
    let o = l.options().unwrap().into_iter().map(|o| (o.act, o.mean)).collect();
    (o, l.stats().clone(), rounds)
}

/// The bits of a row (head, tail) as a key for exact comparisons.
fn row_bits(x: &[f32], tail: &[(u16, u32, Vec<f32>)]) -> Vec<u32> {
    let mut k: Vec<u32> = x.iter().map(|v| v.to_bits()).collect();
    for (s, l, f) in tail {
        k.extend([u32::MAX, *s as u32, *l]);
        k.extend(f.iter().map(|v| v.to_bits()));
    }
    k
}

#[test]
fn the_lookahead_carries_tail_rows_and_keys_them_exactly() {
    // 9 items: every row has tail tokens; ADD's new items are tail tokens whose values differ by world, so leaves
    // whose heads are equal differ in their tails only - a key that dropped the tail would share their values
    for (settings, spec) in [(r#""plies": 2, "worlds": 4"#, r#"{"items0": 9}"#), (r#""plies": 3, "worlds": 4, "topk": 3"#, r#"{"items0": 6}"#), (r#""plies": 2, "worlds": 4, "mode": "world""#, r#"{"items0": 9}"#)] {
        let (on, st_on, rounds) = look_tail(&format!("{{{settings}, \"dedup\": true}}"), spec, 5);
        let (off, st_off, rounds_off) = look_tail(&format!("{{{settings}, \"dedup\": false}}"), spec, 5);
        assert_eq!(on, off, "{settings}: the shared rows' values are their own rows' values");
        assert!(st_on.value_dups > 0 && st_off.value_dups == 0, "{settings}");
        let mut tails = 0;
        for r in &rounds {
            for i in 0..r.kind.len() {
                let t = &r.tail[i];
                tails += !t.is_empty() as usize;
                for w in t.windows(2) {
                    assert!((w[0].0, w[0].1) < (w[1].0, w[1].1), "sorted, unique");
                }
                let x = &r.x[i * TF..(i + 1) * TF];
                assert!(r.ov[i].is_empty() || !r.opts[i].is_empty(), "overrides only on a row with options");
                for &(o, tok) in &r.ov[i] {
                    assert!((o as usize) < r.opts[i].len());
                    let (s, c, l) = unpack_tok(tok);
                    assert_eq!((s, c), (0, 0));
                    if (l as usize) < HEAD {
                        assert_eq!(x[8 + l as usize * W_ITEM], 1.0, "a present head slot");
                    } else {
                        assert!(t.iter().any(|z| z.0 == 0 && z.1 == l), "the target tail token is in the row");
                    }
                }
            }
        }
        assert!(tails > 0, "{settings}: tail rows");
        // exact dedup: the value rows sent with it are the distinct value rows (head and tail) of the run without it
        let distinct = |rs: &[LookRowsT]| {
            let mut v: Vec<Vec<u32>> = rs.iter().flat_map(|r| (0..r.kind.len()).filter(|&i| r.opts[i].is_empty()).map(move |i| row_bits(&r.x[i * TF..(i + 1) * TF], &r.tail[i]))).collect();
            let n = v.len();
            v.sort();
            v.dedup();
            (n, v.len())
        };
        let (sent, uniq) = distinct(&rounds);
        assert_eq!(sent, uniq, "{settings}: no two value rows of the same observation sent");
        let (_, uniq_off) = distinct(&rounds_off);
        assert_eq!(sent, uniq_off, "{settings}: every distinct observation sent once");
    }
}

#[test]
fn a_lookahead_seat_plays_tail_games() {
    let mut s = seat(8, false);
    // (topk: prior rows with options, so overrides too)
    s.look = Some(Box::new(crate::lookahead::LookCfg { plies: 2, worlds: 2, topk: 3, ..Default::default() }));
    let mut out = Vec::new();
    for sparse in [false, true] {
        let mut c = tcfg(s.clone(), 4, r#"{"items0": 9, "max_moves": 10}"#);
        c.sparse = sparse;
        let mut d = Driver::<TailGame>::new(c).unwrap();
        let (res, _, rows) = run_tail(&mut d);
        assert_eq!(res.len(), 4);
        assert!(res.iter().all(|r| r.over && !r.aborted), "the games end normally");
        assert!(rows.iter().any(|r| !r.2.is_empty()) && rows.iter().any(|r| !r.3.is_empty()), "tail rows and overrides");
        let st = d.stats();
        assert!(st.look.decisions > 0 && st.tail_rows > 0, "{} {}", st.look.decisions, st.tail_rows);
        out.push((res.iter().map(|r| (r.steps, r.scores.clone())).collect::<Vec<_>>(), rows));
    }
    assert_eq!(out[0], out[1], "dense and sparse batches: the same rows, the same games");
}

#[test]
fn a_lookahead_row_over_the_limit_aborts_its_game() {
    // 7 items: the root's row fits a limit of 8 (5 + 3 tail tokens), the lookahead's leaf after ADD 1 does not
    let mut s = seat(8, false);
    s.look = Some(Box::new(crate::lookahead::LookCfg { plies: 1, worlds: 1, ..Default::default() }));
    let mut c = tcfg(s, 3, r#"{"items0": 7}"#);
    c.max_row_tokens = Some(8);
    let mut d = Driver::<TailGame>::new(c).unwrap();
    let (res, _, rows) = run_tail(&mut d);
    assert_eq!(res.len(), 3);
    assert!(res.iter().all(|r| r.aborted && !r.over && r.abort_reason == Some("max_row_tokens")));
    assert!(rows.iter().all(|r| 5 + r.2.len() <= 8), "no row over the limit is sent");
    assert_eq!(d.stats().aborted_row_tokens, 3);
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

/// A record's tail tokens as `tail_of` gives a batch row's: (seg, slot, floats).
fn record_tail(s: &Record) -> Vec<(u16, u32, Vec<f32>)> {
    let mut off = 0;
    (0..s.tail.len())
        .map(|j| {
            let w = width(s.tail.seg[j]);
            let v = s.tail.val[off..off + w].to_vec();
            off += w;
            (s.tail.seg[j], s.tail.slot[j], v)
        })
        .collect()
}

/// A row as bits: player, head, tail tokens, overrides.
fn row_key(p: i32, x: &[f32], t: &[(u16, u32, Vec<f32>)], o: &[(u32, i64)]) -> Vec<u64> {
    let mut k = vec![p as u64];
    k.extend(x.iter().map(|v| v.to_bits() as u64));
    for (s, l, f) in t {
        k.push(u64::MAX);
        k.push(*s as u64);
        k.push(*l as u64);
        k.extend(f.iter().map(|v| v.to_bits() as u64));
    }
    for (i, tok) in o {
        k.push(u64::MAX - 1);
        k.push(*i as u64);
        k.push(*tok as u64);
    }
    k
}

#[test]
fn chained_contests_carry_every_episodes_tails_and_overrides() {
    // a contest of three toy_tail games (docs/search.md 2.14 with 2.15): every episode's records keep their own tail
    // and overrides - each record's row one the network was sent for its player, head, tail tokens and overrides bit
    // for bit -, the contest's one row carries them all behind it, the sample arrays' CSRs exactly theirs; dense and
    // sparse batches alike
    for sparse in [false, true] {
        let mut c = tcfg(seat(24, true), 8, r#"{"items0": 9, "max_moves": 16, "episodes": 3}"#);
        c.sparse = sparse;
        let mut d = Driver::<TailGame>::new(c).unwrap();
        let (res, smp, rows) = run_tail(&mut d);
        assert_eq!(res.len(), 8);
        assert!(res.iter().all(|r| r.over && !r.aborted && r.abort_reason.is_none() && r.episodes.len() == 3), "{res:?}");
        assert!(res.iter().all(|r| r.samples == r.episodes.iter().map(|e| e.samples).sum::<u32>() && r.episodes.iter().all(|e| e.over && e.samples > 0)));
        assert_eq!(smp.len(), res.iter().map(|r| r.samples as usize).sum::<usize>());
        let seen: std::collections::HashSet<Vec<u64>> = rows.iter().map(|(p, x, t, o)| row_key(*p, x, t, o)).collect();
        for s in &smp {
            assert!(seen.contains(&row_key(s.player as i32, &s.x, &record_tail(s), &s.ov)), "a record's row the network never saw");
            crate::driver::check_overrides(&s.ov, s.opts.len());
        }
        assert!(smp.iter().filter(|s| !s.tail.is_empty()).count() > smp.len() / 4 && smp.iter().any(|s| !s.ov.is_empty()));
        let a = sample_arrays_rows(&smp, TF, true, true);
        assert_eq!(a.t_off.len(), smp.len() + 1);
        assert_eq!(a.t_seg, smp.iter().flat_map(|s| s.tail.seg.iter().copied()).collect::<Vec<_>>());
        assert_eq!(a.t_slot, smp.iter().flat_map(|s| s.tail.slot.iter().copied()).collect::<Vec<_>>());
        assert_eq!(a.t_val, smp.iter().flat_map(|s| s.tail.val.iter().copied()).collect::<Vec<_>>());
        assert_eq!(a.ov_tok, smp.iter().flat_map(|s| s.ov.iter().map(|o| o.1)).collect::<Vec<_>>());
        for k in 0..smp.len() {
            assert_eq!((a.t_off[k + 1] - a.t_off[k]) as usize, smp[k].tail.len());
            assert_eq!((a.ov_off[k + 1] - a.ov_off[k]) as usize, smp[k].ov.len());
        }
    }
}

#[test]
fn a_later_episode_over_the_row_limit_ends_the_contest_aborted() {
    // game 2 starts from 6 + 70 items (`items_step`): its first row passes the limit - an aborted episode, no samples of
    // its own; the contest unfinished, aborted for the row limit, game 1's samples kept behind its row
    let mut d = Driver::<TailGame>::new(tcfg(seat(16, true), 4, r#"{"items0": 6, "max_moves": 10, "episodes": 3, "items_step": 70}"#)).unwrap();
    let (res, smp, _) = run_tail(&mut d);
    assert_eq!(res.len(), 4);
    for r in &res {
        assert!(!r.over && r.aborted && r.abort_reason == Some("max_row_tokens") && r.episodes.len() == 2, "{r:?}");
        assert!(r.episodes[0].over && !r.episodes[0].aborted && r.episodes[0].samples > 0);
        assert!(!r.episodes[1].over && r.episodes[1].aborted && r.episodes[1].samples == 0);
        assert_eq!(r.samples, r.episodes[0].samples);
    }
    assert_eq!(smp.len(), res.iter().map(|r| r.samples as usize).sum::<usize>());
    assert_eq!(d.stats().aborted_row_tokens, 4);
}
