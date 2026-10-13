use super::*;
use crate::toy::ToyGame;

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
        trace: false,
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
    let mut pri = Vec::new();
    for r in 0..b.b {
        let k = (b.opt_off[r + 1] - b.opt_off[r]) as usize;
        pri.extend(std::iter::repeat_n(1.0 / k as f32, k));
    }
    let mut v = vec![0.0; b.b * MAX_SEATS];
    let m = vec![0.0; b.b * MAX_SEATS];
    for r in 0..b.b {
        let n = b.n_players[r] as usize;
        let h: f32 = b.x[r * f..(r + 1) * f].iter().enumerate().map(|(i, x)| x * ((i * 7 % 13) as f32 - 6.0)).sum();
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

// ------------------------------------------------------------ analysis options, opponent models

fn with_seat(seat: SeatCfg, f: impl FnOnce(&mut NnSeat)) -> SeatCfg {
    match seat {
        SeatCfg::Nn(mut s) => {
            f(&mut s);
            SeatCfg::Nn(s)
        }
        s => s,
    }
}

#[test]
fn oracle_and_class_sims_settings() {
    let one = |seat: SeatCfg| Driver::<ToyGame>::new(cfg(vec![seat], vec![vec![0, 0]], &[("m", KeyKind::Nn)]));
    // the toy has no oracle and one decision class
    assert!(!ToyGame::HAS_ORACLE && crate::game::n_classes::<ToyGame>() == 1);
    assert!(one(with_seat(nn(0, 8, false), |s| s.oracle = 1)).is_err(), "no oracle determinization");
    assert!(one(with_seat(nn(0, 8, false), |s| s.class_sims = vec![8, 8])).is_err(), "two classes");
    for p in [0.0, -1.0, 1.5, f64::NAN] {
        assert!(one(with_seat(nn(0, 8, false), |s| s.oracle_p = p)).is_err(), "oracle_p {p}");
    }
    assert!(one(with_seat(nn(0, 8, false), |s| s.class_sims = vec![8])).is_ok());
    // class_sims equal to sims (or 0 = sims) and oracle 0 reproduce the golden records
    for cs in [vec![16], vec![0]] {
        let (_, recs) = golden_selfplay(with_seat(nn(0, 16, true), |s| s.class_sims = cs.clone()));
        assert_eq!((recs.len(), records_digest(&recs)), GOLDEN_RECORDS, "class_sims {cs:?}");
    }
    // mask_features: the masked inputs are zero in every record; out-of-range indices rejected
    let (_, recs) = golden_selfplay(with_seat(nn(0, 16, true), |s| s.mask_features = vec![0, 3]));
    assert!(!recs.is_empty() && recs.iter().all(|r| r.x[0] == 0.0 && r.x[3] == 0.0));
    let (_, gold) = golden_selfplay(nn(0, 16, true));
    assert!(gold.iter().any(|r| r.x[0] != 0.0 || r.x[3] != 0.0), "the toy's features 0 and 3 are not always zero");
    assert!(one(with_seat(nn(0, 8, false), |s| s.mask_features = vec![ToyGame::NUM_FEATURES as u32])).is_err());
    // class_sims applies to full searches: fewer simulations, other records
    let (_, recs) = golden_selfplay(with_seat(nn(0, 16, true), |s| s.class_sims = vec![12]));
    assert_ne!(records_digest(&recs), GOLDEN_RECORDS.1);
}

#[test]
fn trace_replays_every_game() {
    let mut c = cfg(vec![nn(0, 16, true), SeatCfg::Bot("random".into())], (0..12).map(|g| if g % 3 == 0 { vec![1, 0] } else { vec![0, 0] }).collect(), &[("m", KeyKind::Nn)]);
    c.trace = true;
    let mut d = Driver::<ToyGame>::new(c.clone()).unwrap();
    let (res, recs) = run(&mut d);
    assert_eq!(res.len(), 12);
    let spec = ToyGame::parse_spec(&c.spec).unwrap();
    for r in &res {
        assert!(!r.actions.is_empty());
        let mut g = ToyGame::new_game(&spec, r.n, r.seed).unwrap();
        let mut key = Vec::new();
        let mut legal = Vec::new();
        for &a in &r.actions {
            crate::tree::transition(&mut g, a, None, &mut key, &mut legal).unwrap();
        }
        assert!(g.to_move().is_none(), "game {} replayed to its end", r.gid);
        assert_eq!((0..r.n).map(|i| g.score(i)).collect::<Vec<_>>(), r.scores);
    }
    // tracing changes nothing else
    let mut c2 = c.clone();
    c2.trace = false;
    let (res2, recs2) = run(&mut Driver::<ToyGame>::new(c2).unwrap());
    assert_eq!(records_digest(&recs), records_digest(&recs2));
    assert!(res2.iter().all(|r| r.actions.is_empty() && r.seed == 0));
    assert_eq!(res.iter().map(|r| (&r.scores, r.steps)).collect::<Vec<_>>(), res2.iter().map(|r| (&r.scores, r.steps)).collect::<Vec<_>>());
    // games with external seats cannot be traced
    let mut c3 = cfg(vec![SeatCfg::External(0), nn(1, 8, false)], vec![vec![0, 1]], &[("e", KeyKind::External), ("m", KeyKind::Nn)]);
    c3.trace = true;
    assert!(Driver::<ToyGame>::new(c3).is_err());
}

use crate::secret::SecretGame;

/// Card preferences of the secret game's stand-in policy.
const SECRET_W: [f32; 5] = [0.2, 0.6, 1.0, 3.0, 0.5];

/// Any game: priors ~ SECRET_W[action % 5] (the secret game's card preferences), value by a hash
/// of the features. -> (results, records, rows evaluated).
fn run_any<G: SearchGame>(d: &mut Driver<G>) -> (Vec<GameResult>, Vec<Record>, u64) {
    let f = G::NUM_FEATURES;
    let (mut res, mut smp, mut rows) = (Vec::new(), Vec::new(), 0u64);
    while let Some(b) = d.next_batch(true).unwrap() {
        let mut pri = Vec::new();
        for r in 0..b.b {
            let acts = &b.opt_action[b.opt_off[r] as usize..b.opt_off[r + 1] as usize];
            let s: f32 = acts.iter().map(|&a| SECRET_W[a as usize % 5]).sum();
            pri.extend(acts.iter().map(|&a| SECRET_W[a as usize % 5] / s));
        }
        let mut v = vec![0.0; b.b * MAX_SEATS];
        for r in 0..b.b {
            let h: f32 = b.x[r * f..(r + 1) * f].iter().enumerate().map(|(i, x)| x * ((i * 7 % 13) as f32 - 6.0)).sum();
            let w = 1.0 / (1.0 + (-h * 0.1).exp());
            v[r * MAX_SEATS] = w;
            v[r * MAX_SEATS + 1] = 1.0 - w;
        }
        rows += b.b as u64;
        d.submit(b.id, &pri, &v, &vec![0.0; b.b * MAX_SEATS]).unwrap();
        res.extend(d.take_results());
        smp.extend(d.take_samples());
    }
    res.extend(d.take_results());
    smp.extend(d.take_samples());
    res.sort_by_key(|r| r.gid);
    (res, smp, rows)
}

fn secret_cfg(seats: Vec<SeatCfg>, games: Vec<Vec<u16>>) -> DriverCfg {
    let mut c = cfg(seats, games, &[("m", KeyKind::Nn)]);
    c.seed = 11;
    c
}

#[test]
fn opponent_model_settings_are_checked() {
    let toy = |seat: SeatCfg| Driver::<ToyGame>::new(cfg(vec![seat], vec![vec![0, 0]], &[("m", KeyKind::Nn)]));
    let sec = |seat: SeatCfg| Driver::<SecretGame>::new(secret_cfg(vec![seat], vec![vec![0, 0]]));
    const { assert!(!ToyGame::HAS_PSAMPLE && !ToyGame::HAS_BELIEF) };
    // a game without the hooks refuses the options (they would silently do nothing)
    assert!(toy(with_seat(nn(0, 8, false), |s| s.psample = true)).is_err());
    assert!(toy(with_seat(nn(0, 8, false), |s| s.belief_mix = 0.5)).is_err());
    assert!(sec(with_seat(nn(0, 8, false), |s| s.psample = true)).is_ok());
    for (mix, temp, k) in [(1.5, 1.0, 64), (-0.1, 1.0, 64), (f64::NAN, 1.0, 64), (0.5, 0.0, 64), (0.5, f64::INFINITY, 64), (0.5, 1.0, 0)] {
        assert!(sec(with_seat(nn(0, 8, false), |s| (s.belief_mix, s.belief_temp, s.belief_particles) = (mix, temp, k))).is_err(), "{mix} {temp} {k}");
    }
    assert!(sec(with_seat(nn(0, 8, false), |s| (s.belief_mix, s.belief_temp, s.belief_particles) = (1.0, 0.5, 1))).is_ok());
    // the defaults are the plain search
    assert_eq!(NnSeat::eval(0, 8).tree_opts(), crate::tree::TreeOpts::plain(true));
}

#[test]
fn psample_and_belief_in_secret_selfplay() {
    let base = nn(0, 24, true);
    let games = vec![vec![0, 0]; 40];
    let mk = |seat: SeatCfg, threads: usize, groups: usize| {
        let mut c = secret_cfg(vec![seat], games.clone());
        (c.threads, c.groups) = (threads, groups);
        let mut d = Driver::<SecretGame>::new(c).unwrap();
        let r = run_any(&mut d);
        (r, d.stats())
    };
    // plain search: none of the opponent-model counters move
    let ((r0, s0, rows0), st0) = mk(base.clone(), 2, 2);
    assert_eq!((st0.psampled, st0.belief_rows, st0.belief_jobs, st0.belief_worlds), (0, 0, 0, 0));
    assert_eq!(rows0, st0.leaves);
    // both options on: reproducible across threads / groups, every counter moves
    let both = with_seat(base.clone(), |s| (s.psample, s.belief_mix) = (true, 1.0));
    let ((r1, s1, rows1), st1) = mk(both.clone(), 1, 1);
    let ((r3, s3, _), _) = mk(both.clone(), 3, 2);
    let key = |r: &GameResult| (r.gid, r.steps, r.scores.clone(), r.samples);
    assert_eq!(r1.iter().map(key).collect::<Vec<_>>(), r3.iter().map(key).collect::<Vec<_>>());
    assert_eq!(records_digest(&s1), records_digest(&s3));
    assert_ne!(records_digest(&s0), records_digest(&s1));
    assert!(r1.iter().all(|r| r.over) && r0.len() == r1.len());
    assert!(st1.psampled > 0, "{st1:?}");
    // at most one belief per game and seat (the other seat's hand; sampled from its snapshot at
    // the first searched card play of the seat)
    assert!(st1.belief_jobs > 20 && st1.belief_jobs <= 80 && st1.belief_missing == 0, "{st1:?}");
    assert!(st1.belief_searches > 0 && st1.belief_worlds > 0 && st1.belief_targets >= st1.belief_searches, "{st1:?}");
    // belief rows ride in the network batches: at most 1 + 4 rows per belief (2 picks; 4
    // distinct first picks), counted apart from the search's leaves
    assert_eq!(rows1, st1.leaves + st1.belief_rows);
    assert!(st1.belief_rows >= 2 * st1.belief_jobs && st1.belief_rows <= 5 * st1.belief_jobs, "{st1:?}");
    // the options separately
    let ((_, sp, _), stp) = mk(with_seat(base.clone(), |s| s.psample = true), 2, 2);
    assert!(stp.psampled > 0 && stp.belief_rows == 0);
    let ((_, sb, _), stb) = mk(with_seat(base.clone(), |s| s.belief_mix = 0.5), 2, 2);
    assert!(stb.psampled == 0 && stb.belief_jobs > 20 && stb.belief_worlds > 0);
    assert!(stb.belief_worlds * 3 < st1.belief_worlds * 2, "belief_mix 0.5 uses about half as many particle worlds");
    assert!(records_digest(&sp) != records_digest(&s0) && records_digest(&sb) != records_digest(&s0));
}

#[test]
fn belief_of_a_bot_opponent_and_mask_features() {
    // seat 1 is a random bot: its hand's snapshot is taken all the same; the belief rows carry
    // the masked inputs too
    let seat = with_seat(nn(0, 16, false), |s| (s.belief_mix, s.mask_features) = (1.0, vec![3]));
    let games: Vec<Vec<u16>> = (0..20).map(|g| if g % 2 == 0 { vec![0, 1] } else { vec![1, 0] }).collect();
    let mut d = Driver::<SecretGame>::new(secret_cfg(vec![seat, SeatCfg::Bot("random".into())], games)).unwrap();
    let mut masked = true;
    let f = SecretGame::NUM_FEATURES;
    while let Some(b) = d.next_batch(true).unwrap() {
        masked &= (0..b.b).all(|r| b.x[r * f + 3] == 0.0);
        let k: Vec<f32> = (0..b.b)
            .flat_map(|r| {
                let n = (b.opt_off[r + 1] - b.opt_off[r]) as usize;
                std::iter::repeat_n(1.0 / n as f32, n)
            })
            .collect();
        let mut v = vec![0.0; b.b * MAX_SEATS];
        for r in 0..b.b {
            (v[r * MAX_SEATS], v[r * MAX_SEATS + 1]) = (0.5, 0.5);
        }
        d.submit(b.id, &k, &v, &vec![0.0; b.b * MAX_SEATS]).unwrap();
    }
    let st = d.stats();
    assert!(masked);
    assert!(st.belief_jobs > 10 && st.belief_jobs <= 20 && st.belief_missing == 0, "{st:?}");
    assert!(st.belief_searches > 0, "{st:?}");
    assert_eq!(d.take_results().len(), 20);
}

#[test]
fn belief_survives_rolling_epochs() {
    let seat = with_seat(nn(0, 16, true), |s| (s.psample, s.belief_mix) = (true, 1.0));
    let mut c = secret_cfg(vec![seat], vec![vec![0, 0]]);
    (c.rolling, c.concurrency) = (true, 12);
    let mut d = Driver::<SecretGame>::new(c).unwrap();
    for (k, net) in [(0u64, 0u32), (1, 1), (2, 1), (3, 2)] {
        d.resume(10 + k, net).unwrap();
        let _ = run_any(&mut d);
        assert!(d.paused());
    }
    let st = d.stats();
    assert!(st.games_finished >= 46 && st.belief_jobs > 0 && st.restarts > 0, "{st:?}");
}
