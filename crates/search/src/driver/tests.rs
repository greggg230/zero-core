use super::*;
use crate::toy::{ToyGame, N_SPACES};

fn nn(key: u16, sims: u32, selfplay: bool) -> SeatCfg {
    let mut s = NnSeat::eval(key, sims);
    s.selfplay = selfplay;
    if selfplay {
        s.sims_fast = Some(sims / 4);
        s.p_full = 0.5;
        s.temp_moves = 10;
    }
    SeatCfg::Nn(s)
}

fn cfg(seats: Vec<SeatCfg>, games: Vec<Vec<u16>>, keys: &[(&str, KeyKind)]) -> DriverCfg {
    DriverCfg {
        spec: "{}".into(),
        seats,
        games,
        keys: keys.iter().map(|(k, t)| (k.to_string(), *t)).collect(),
        concurrency: 16,
        seed: 7,
        record: true,
        threads: 2,
        groups: 2,
        max_steps: 10_000,
        rolling: false,
        net: 0,
        x_dtype: XDtype::F32,
        aux: false,
    }
}

/// Answer every batch: uniform priors, value by a hash of the features (deterministic), the
/// external seat plays the first legal action of the exported state.
fn answer(d: &mut Driver<ToyGame>, b: &Batch) {
    if b.external {
        let pl: Vec<String> = b
            .states
            .iter()
            .map(|s| {
                let v: serde_json::Value = serde_json::from_str(s).unwrap();
                format!("[{}]", v["legal"][0])
            })
            .collect();
        d.submit_external(b.id, pl).unwrap();
        return;
    }
    let f = ToyGame::NUM_FEATURES;
    let x = b.x.to_f32();
    let mut pri = Vec::new();
    for r in 0..b.b {
        let k = (b.opt_off[r + 1] - b.opt_off[r]) as usize;
        pri.extend(std::iter::repeat_n(1.0 / k as f32, k));
    }
    let mut v = vec![0.0; b.b * MAX_SEATS];
    let m = vec![0.0; b.b * MAX_SEATS];
    for r in 0..b.b {
        let n = b.n_players[r] as usize;
        let h: f32 = x[r * f..(r + 1) * f].iter().enumerate().map(|(i, x)| x * ((i * 7 % 13) as f32 - 6.0)).sum();
        let w = 1.0 / (1.0 + (-h * 0.1).exp());
        v[r * MAX_SEATS] = w;
        for j in 1..n {
            v[r * MAX_SEATS + j] = (1.0 - w) / (n - 1) as f32;
        }
    }
    d.submit(b.id, &pri, &v, &m).unwrap();
}

fn run(d: &mut Driver<ToyGame>) -> (Vec<GameResult>, Vec<Record>) {
    let mut res = Vec::new();
    let mut smp = Vec::new();
    while let Some(b) = d.next_batch(true).unwrap() {
        answer(d, &b);
        res.extend(d.take_results());
        smp.extend(d.take_samples());
    }
    res.extend(d.take_results());
    smp.extend(d.take_samples());
    res.sort_by_key(|r| r.gid);
    (res, smp)
}

#[test]
fn bots_only_games_finish() {
    let c = cfg(vec![SeatCfg::Bot("random".into()), SeatCfg::Bot("heuristic".into())], (0..200).map(|g| if g % 2 == 0 { vec![0, 1] } else { vec![1, 0] }).collect(), &[]);
    let mut d = Driver::<ToyGame>::new(c).unwrap();
    let (res, smp) = run(&mut d);
    assert_eq!(res.len(), 200);
    assert!(smp.is_empty());
    assert!(d.finished());
    let hw: f32 = res.iter().map(|r| r.result[if r.gid % 2 == 0 { 1 } else { 0 }]).sum::<f32>() / 200.0;
    assert!(hw > 0.9, "heuristic vs random {hw}");
    assert!(res.iter().all(|r| r.over && r.rounds == 6));
}

#[test]
fn selfplay_records_and_is_reproducible_across_threads() {
    let mk = |threads: usize, groups: usize| {
        let mut c = cfg(vec![nn(0, 16, true)], vec![vec![0, 0]; 24], &[("m", KeyKind::Nn)]);
        c.threads = threads;
        c.groups = groups;
        let mut d = Driver::<ToyGame>::new(c).unwrap();
        run(&mut d)
    };
    let (r1, s1) = mk(1, 1);
    let (r4, s4) = mk(4, 3);
    assert_eq!(r1.len(), 24);
    let key = |r: &GameResult| (r.gid, r.steps, r.scores.clone(), r.samples);
    assert_eq!(r1.iter().map(key).collect::<Vec<_>>(), r4.iter().map(key).collect::<Vec<_>>());
    assert_eq!(s1.len(), s4.len());
    assert!(!s1.is_empty());
    let total: u32 = r1.iter().map(|r| r.samples).sum();
    assert_eq!(total as usize, s1.len());
    for s in &s1 {
        assert_eq!(s.x.len(), ToyGame::NUM_FEATURES);
        assert_eq!(s.pi.len(), s.opts.len());
        assert!((s.pi.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!((s.z[0] + s.z[1] - 1.0).abs() < 1e-6);
        assert!((s.q[0] + s.q[1] - 1.0).abs() < 1e-4);
        assert!((s.sc[0] + s.sc[1]).abs() < 1e-6, "2p margins are opposite");
        assert!(s.opts.len() >= 2);
    }
    let st = Driver::<ToyGame>::new(cfg(vec![nn(0, 8, false)], vec![vec![0, 0]; 4], &[("m", KeyKind::Nn)])).unwrap().stats();
    assert_eq!(st.games_started, 4);
}

#[test]
fn arena_two_keys_and_external_seat() {
    // game g: seats alternate between key m, key o and an external seat
    let seats = vec![nn(0, 8, false), nn(1, 8, false), SeatCfg::External(2)];
    let games: Vec<Vec<u16>> = (0..30).map(|g| match g % 3 {
        0 => vec![0, 1],
        1 => vec![2, 0],
        _ => vec![1, 2, 0],
    }).collect();
    let c = cfg(seats, games, &[("m", KeyKind::Nn), ("o", KeyKind::Nn), ("ext", KeyKind::External)]);
    let mut d = Driver::<ToyGame>::new(c).unwrap();
    let mut seen = [0usize; 3];
    let mut res = Vec::new();
    while let Some(b) = d.next_batch(true).unwrap() {
        seen[b.key as usize] += b.b;
        if b.external {
            for s in &b.states {
                let v: serde_json::Value = serde_json::from_str(s).unwrap();
                assert!(!v["legal"].as_array().unwrap().is_empty());
            }
        }
        answer(&mut d, &b);
        res.extend(d.take_results());
    }
    res.extend(d.take_results());
    assert_eq!(res.len(), 30);
    assert!(seen.iter().all(|&x| x > 0), "{seen:?}");
    for r in &res {
        assert!(r.over);
        if r.gid % 3 != 0 {
            assert!(r.ext_moves > 0);
        } else {
            assert_eq!(r.ext_moves, 0);
        }
    }
    // no samples without selfplay seats
    assert!(d.take_samples().is_empty());
}

#[test]
fn rolling_phases() {
    let mut c = cfg(vec![nn(0, 8, true)], vec![vec![0, 0]], &[("m", KeyKind::Nn)]);
    c.rolling = true;
    c.concurrency = 12;
    let mut d = Driver::<ToyGame>::new(c).unwrap();
    assert!(d.paused());
    assert!(d.next_batch(true).unwrap().is_none());
    let mut total = 0;
    for phase in 0..4u32 {
        d.resume(10, phase / 2).unwrap();
        let mut fin = 0;
        let mut nets = std::collections::HashSet::new();
        while let Some(b) = d.next_batch(true).unwrap() {
            answer(&mut d, &b);
            fin += d.take_results().len();
            for s in d.take_samples() {
                nets.insert(s.net);
            }
        }
        fin += d.take_results().len();
        assert!(d.paused());
        assert!(fin >= 10, "phase {phase}: {fin} games");
        total += fin;
        let inf = d.in_flight();
        assert!(inf.games > 0, "games in flight are kept across the pause");
        if phase > 0 {
            assert!(nets.iter().all(|&n| n <= phase / 2));
        }
    }
    let st = d.stats();
    assert_eq!(st.games_finished as usize, total);
    // net 0 -> 0 (same), 0 -> 1 (new epoch), 1 -> 1
    assert_eq!(st.epoch, 1);
    assert!(st.restarts > 0);
}

#[test]
fn bad_configs_are_rejected() {
    assert!(Driver::<ToyGame>::new(cfg(vec![SeatCfg::Bot("nope".into())], vec![vec![0, 0]], &[])).is_err());
    assert!(Driver::<ToyGame>::new(cfg(vec![SeatCfg::Bot("random".into())], vec![vec![0]], &[])).is_err());
    assert!(Driver::<ToyGame>::new(cfg(vec![nn(0, 8, false)], vec![vec![0, 0]], &[("e", KeyKind::External)])).is_err());
    let mut d = Driver::<ToyGame>::new(cfg(vec![nn(0, 8, false)], vec![vec![0, 0]], &[("m", KeyKind::Nn)])).unwrap();
    let b = d.next_batch(true).unwrap().unwrap();
    assert!(d.submit(b.id, &[0.5], &[0.0; 8], &[0.0; 8]).is_err(), "wrong shapes");
    assert!(d.submit_external(b.id, vec![]).is_err());
    answer(&mut d, &b);
    assert!(d.submit(b.id, &[], &[], &[]).is_err(), "already answered");
}

#[test]
fn external_errors_stop_the_driver() {
    let c = cfg(vec![SeatCfg::External(0), SeatCfg::Bot("random".into())], vec![vec![0, 1]], &[("ext", KeyKind::External)]);
    let mut d = Driver::<ToyGame>::new(c).unwrap();
    let b = d.next_batch(true).unwrap().unwrap();
    d.submit_external(b.id, vec!["[999]".into()]).unwrap();
    let e = d.next_batch(true).unwrap_err();
    assert!(e.contains("external move failed"), "{e}");
}

/// FNV-1a over the records' bytes (a stable digest: std's hasher may change between Rust
/// versions).
fn records_digest(recs: &[Record]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut eat = |b: &[u8]| {
        for &x in b {
            h ^= x as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    };
    for r in recs {
        for v in r.x.iter().chain(&r.pi).chain(&r.z).chain(&r.q).chain(&r.sc) {
            eat(&v.to_bits().to_le_bytes());
        }
        eat(&[r.player, r.n_players]);
        for a in &r.opts {
            eat(&a.to_le_bytes());
        }
        eat(&r.net.to_le_bytes());
    }
    h
}

/// Self-play records of a fixed configuration (the deterministic `answer` evaluator).
fn golden_selfplay(seat: SeatCfg) -> (Vec<GameResult>, Vec<Record>) {
    let mut c = cfg(vec![seat], vec![vec![0, 0]; 24], &[("m", KeyKind::Nn)]);
    c.seed = 2026;
    let mut d = Driver::<ToyGame>::new(c).unwrap();
    run(&mut d)
}

#[test]
fn visits_records_unchanged() {
    // digest of these records at Glass Road Zero 96baefd / Caverna Zero 9a2f451 (before policy
    // targets existed; unchanged by the move into zero-core): the default
    // policy target ('visits') must reproduce them bit for bit
    let (_, recs) = golden_selfplay(nn(0, 16, true));
    assert_eq!(recs.len(), GOLDEN_RECORDS.0);
    assert_eq!(records_digest(&recs), GOLDEN_RECORDS.1, "visits records changed");
}
const GOLDEN_RECORDS: (usize, u64) = (347, 8501300995519660688);

fn with_target(seat: SeatCfg, t: PolicyTarget) -> SeatCfg {
    match seat {
        SeatCfg::Nn(mut s) => {
            s.policy_target = t;
            SeatCfg::Nn(s)
        }
        s => s,
    }
}

#[test]
fn completed_q_changes_only_the_policy_targets() {
    // the search is unchanged, so the games, moves and every other field are identical; only
    // pi differs (sharper than the visit distribution on average)
    let cq = PolicyTarget::CompletedQ { c_visit: 50.0, c_scale: 1.0 };
    let (rv, sv) = golden_selfplay(with_target(nn(0, 16, true), PolicyTarget::Visits));
    let (rc, sc) = golden_selfplay(with_target(nn(0, 16, true), cq));
    let key = |r: &GameResult| (r.gid, r.steps, r.scores.clone(), r.samples, r.decisions);
    assert_eq!(rv.iter().map(key).collect::<Vec<_>>(), rc.iter().map(key).collect::<Vec<_>>());
    assert_eq!(records_digest(&sv), GOLDEN_RECORDS.1);
    assert_eq!(sv.len(), sc.len());
    let (mut differ, mut hv, mut hc) = (0, 0.0, 0.0);
    let ent = |p: &[f32]| -p.iter().filter(|&&x| x > 0.0).map(|&x| x as f64 * (x as f64).ln()).sum::<f64>() / (p.len() as f64).ln();
    for (a, b) in sv.iter().zip(&sc) {
        assert_eq!((&a.x, a.player, a.n_players, &a.opts, a.z, a.q, a.sc, a.net), (&b.x, b.player, b.n_players, &b.opts, b.z, b.q, b.sc, b.net));
        assert_eq!(b.pi.len(), b.opts.len());
        assert!((b.pi.iter().sum::<f32>() - 1.0).abs() < 1e-5 && b.pi.iter().all(|x| x.is_finite() && *x >= 0.0));
        differ += (a.pi != b.pi) as usize;
        hv += ent(&a.pi);
        hc += ent(&b.pi);
    }
    assert!(differ * 10 > sv.len() * 9, "{differ} of {} targets differ", sv.len());
    assert!(hc < hv, "normalised entropy: completed Q {hc} vs visits {hv}");
    // bad settings are rejected
    let bad = with_target(nn(0, 8, true), PolicyTarget::CompletedQ { c_visit: f64::NAN, c_scale: 1.0 });
    assert!(Driver::<ToyGame>::new(cfg(vec![bad], vec![vec![0, 0]], &[("m", KeyKind::Nn)])).is_err());
    assert!(PolicyTarget::parse("cq", 50.0, -1.0).is_err() && PolicyTarget::parse("argmax", 50.0, 1.0).is_err());
    assert_eq!(PolicyTarget::parse("visits", 0.0, 0.0), Ok(PolicyTarget::Visits));
}

/// Uniform priors and even values whatever the features (two drivers stay in lockstep).
fn answer_flat(d: &mut Driver<ToyGame>, b: &Batch) {
    let pri: Vec<f32> = (0..b.b).flat_map(|r| {
        let k = (b.opt_off[r + 1] - b.opt_off[r]) as usize;
        std::iter::repeat_n(1.0 / k as f32, k)
    }).collect();
    let mut v = vec![0.0; b.b * MAX_SEATS];
    for r in 0..b.b {
        let n = b.n_players[r] as usize;
        for j in 0..n {
            v[r * MAX_SEATS + j] = 1.0 / n as f32;
        }
    }
    let m = vec![0.0; b.b * MAX_SEATS];
    d.submit(b.id, &pri, &v, &m).unwrap();
}

#[test]
fn f16_features_are_the_rounded_f32_features() {
    let mk = |dt: XDtype| {
        let mut c = cfg(vec![nn(0, 8, true)], vec![vec![0, 0]; 12], &[("m", KeyKind::Nn)]);
        c.x_dtype = dt;
        c.groups = 3;
        Driver::<ToyGame>::new(c).unwrap()
    };
    let (mut a, mut h) = (mk(XDtype::F32), mk(XDtype::F16));
    let f = ToyGame::NUM_FEATURES;
    let mut nb = 0;
    loop {
        let (ba, bh) = (a.next_batch(true).unwrap(), h.next_batch(true).unwrap());
        let (Some(ba), Some(bh)) = (ba, bh) else {
            assert!(a.finished() && h.finished());
            break;
        };
        assert_eq!((ba.b, &ba.opt_off, &ba.opt_action, &ba.gid), (bh.b, &bh.opt_off, &bh.opt_action, &bh.gid));
        let (xa, xh) = (ba.x.as_f32().unwrap(), bh.x.as_f16_bits().unwrap());
        assert_eq!((xa.len(), xh.len()), (ba.b * f, ba.b * f));
        let mut want = vec![0u16; xa.len()];
        crate::xbuf::f32_to_f16_soft(xa, &mut want);
        assert_eq!(xh, &want[..]);
        answer_flat(&mut a, &ba);
        answer_flat(&mut h, &bh);
        nb += 1;
    }
    assert!(nb > 10);
    // the buffers came back to the pools (at most groups x keys + 2 are kept)
    for d in [&a, &h] {
        let s = d.sh.x_pool.spare();
        assert!((1..=5).contains(&s), "{s} spare buffers");
    }
}

#[test]
fn aux_targets_recorded_per_sample_at_game_end() {
    // the same self-play with and without aux: identical games and records but the aux
    // targets; with aux every sample carries its decider's view of the final state
    let mk = |aux: bool| {
        let mut c = cfg(vec![nn(0, 16, true)], vec![vec![0, 0]; 16], &[("m", KeyKind::Nn)]);
        c.aux = aux;
        c.threads = 3;
        let mut d = Driver::<ToyGame>::new(c).unwrap();
        // results and samples in emission order (a game's samples follow each other)
        let (mut res, mut smp) = (Vec::new(), Vec::new());
        while let Some(b) = d.next_batch(true).unwrap() {
            answer(&mut d, &b);
            res.extend(d.take_results());
            smp.extend(d.take_samples());
        }
        res.extend(d.take_results());
        smp.extend(d.take_samples());
        (res, smp)
    };
    let (r0, s0) = mk(false);
    let (r1, s1) = mk(true);
    let key = |r: &GameResult| (r.gid, r.steps, r.scores.clone(), r.samples);
    assert_eq!(r0.iter().map(key).collect::<Vec<_>>(), r1.iter().map(key).collect::<Vec<_>>());
    assert_eq!(records_digest(&s0), records_digest(&s1));
    assert!(s0.iter().all(|r| r.aux.is_empty()));
    let w = <ToyGame as SearchGame>::aux_width();
    assert!(!s1.is_empty() && s1.iter().all(|r| r.aux.len() == w));
    let mut k = 0;
    for r in &r1 {
        let recs = &s1[k..k + r.samples as usize];
        k += r.samples as usize;
        let mut seen = [None::<&Vec<f32>>; 2];
        for s in recs {
            let p = s.player as usize;
            // the score parts: the decider's first, each summing to that seat's final score
            for j in 0..2 {
                let tot: f32 = s.aux[N_SPACES + 6 * j..N_SPACES + 6 * j + 6].iter().sum();
                assert!((tot * 10.0 - r.scores[(p + j) % 2]).abs() < 1e-4);
            }
            // one final state: the same targets for every sample of a seat
            if let Some(a) = seen[p] {
                assert_eq!(a, &s.aux);
            }
            seen[p] = Some(&s.aux);
        }
        if let (Some(a), Some(b)) = (seen[0], seen[1]) {
            // 2 players: ownership flips sign, the score blocks swap
            assert!((0..N_SPACES).all(|c| a[c] == -b[c]));
            assert_eq!(a[N_SPACES..N_SPACES + 6], b[N_SPACES + 6..N_SPACES + 12]);
        }
    }
    assert_eq!(k, s1.len());
    let arr = sample_arrays(&s1, ToyGame::NUM_FEATURES);
    assert_eq!((arr.aw, arr.aux.len()), (w, s1.len() * w));
    assert_eq!(sample_arrays(&s0, ToyGame::NUM_FEATURES).aw, 0);
}
