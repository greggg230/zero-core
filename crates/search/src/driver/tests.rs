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
        queue_only: false,
        move_log: 0,
        aux: false,
        paired: false,
        sparse: false,
        branch: None,
        starts: None,
    }
}

/// Answer every batch: uniform priors, value by a hash of the features (deterministic), the
/// external seat plays the first legal action of the exported state.
fn answer<G: SearchGame>(d: &mut Driver<G>, b: &Batch) {
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
    let f = G::NUM_FEATURES;
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

fn run<G: SearchGame>(d: &mut Driver<G>) -> (Vec<GameResult>, Vec<Record>) {
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
    let cq = PolicyTarget::CompletedQ { c_visit: 50.0, c_scale: 1.0, prior_w: 1.0 };
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
    let bad = with_target(nn(0, 8, true), PolicyTarget::CompletedQ { c_visit: f64::NAN, c_scale: 1.0, prior_w: 1.0 });
    assert!(Driver::<ToyGame>::new(cfg(vec![bad], vec![vec![0, 0]], &[("m", KeyKind::Nn)])).is_err());
    assert!(PolicyTarget::parse("cq", 50.0, -1.0).is_err() && PolicyTarget::parse("argmax", 50.0, 1.0).is_err());
    assert_eq!(PolicyTarget::parse("visits", 0.0, 0.0), Ok(PolicyTarget::Visits));
}

/// Self-play records of a fixed configuration under skewed priors (`answer_skewed`: under
/// `answer`'s uniform priors P0^beta is uniform too, and a prior exponent changes nothing).
fn skewed_selfplay(seat: SeatCfg) -> (Vec<GameResult>, Vec<Record>) {
    let mut c = cfg(vec![seat], vec![vec![0, 0]; 24], &[("m", KeyKind::Nn)]);
    c.seed = 2026;
    let mut d = Driver::<ToyGame>::new(c).unwrap();
    let (mut res, mut smp) = (Vec::new(), Vec::new());
    while let Some(b) = d.next_batch(true).unwrap() {
        answer_skewed(&mut d, &b);
        res.extend(d.take_results());
        smp.extend(d.take_samples());
    }
    res.extend(d.take_results());
    smp.extend(d.take_samples());
    res.sort_by_key(|r| r.gid);
    (res, smp)
}

/// The 'cq' target's records at main aae4ca2 (before `prior_w`): `golden_selfplay` at c_scale 1.0
/// and `skewed_selfplay` at c_scale 0.3 (the Delver line's) - prior_w 1.0 must reproduce both.
const GOLDEN_CQ_RECORDS: (usize, u64) = (347, 1109346877610972733);
const GOLDEN_CQ_SKEWED_RECORDS: (usize, u64) = (371, 6033783840401516154);

#[test]
fn cq_prior_w_changes_only_the_recorded_targets() {
    // docs/search.md 3.13: the prior exponent beta is read by the recorded 'cq' target alone - the
    // moves (PUCT, temperature, the 'cq' move rule, the Gumbel root), results, q, z and every other
    // field of a record are those of beta 1; pi is softer.
    let cq = |cs: f64, pw: f64| PolicyTarget::CompletedQ { c_visit: 50.0, c_scale: cs, prior_w: pw };
    // prior_w 1.0: the records of before, bit for bit
    let (_, s) = golden_selfplay(with_target(nn(0, 16, true), cq(1.0, 1.0)));
    assert_eq!((s.len(), records_digest(&s)), GOLDEN_CQ_RECORDS);
    let (r1, s1) = skewed_selfplay(with_target(nn(0, 16, true), cq(0.3, 1.0)));
    assert_eq!((s1.len(), records_digest(&s1)), GOLDEN_CQ_SKEWED_RECORDS);
    let key = |r: &GameResult| (r.gid, r.steps, r.scores.clone(), r.samples, r.decisions);
    let keys = |rs: &[GameResult]| rs.iter().map(key).collect::<Vec<_>>();
    let ent = |p: &[f32]| -p.iter().filter(|&&x| x > 0.0).map(|&x| x as f64 * (x as f64).ln()).sum::<f64>() / (p.len() as f64).ln();
    let maxs = |p: &[f32]| p.iter().copied().fold(0.0f32, f32::max) as f64;
    // c_scale 0: the target is the floored prior itself (ln P0 for the identity below)
    let (r0, s0) = skewed_selfplay(with_target(nn(0, 16, true), cq(0.0, 1.0)));
    assert_eq!(keys(&r0), keys(&r1));
    let mut last = (mean_of(&s1, &ent), mean_of(&s1, &maxs));
    for pw in [0.8, 0.5] {
        let (rb, sb) = skewed_selfplay(with_target(nn(0, 16, true), cq(0.3, pw)));
        assert_eq!(keys(&rb), keys(&r1), "prior_w {pw}: the games changed");
        assert_eq!(sb.len(), s1.len());
        let (mut differ, mut worst, mut multi) = (0, 0.0f64, 0);
        for ((a, b), e) in s1.iter().zip(&sb).zip(&s0) {
            assert_eq!((&a.x, a.player, a.n_players, &a.opts, a.z, a.q, a.sc, a.net), (&b.x, b.player, b.n_players, &b.opts, b.z, b.q, b.sc, b.net));
            assert!((b.pi.iter().sum::<f32>() - 1.0).abs() < 1e-5 && b.pi.iter().all(|x| x.is_finite() && *x >= 0.0));
            differ += (a.pi != b.pi) as usize;
            // ln pi_beta - ln pi_1 = (beta - 1) ln P0 + a constant per root (ln P0 = ln pi_E + a constant)
            let d: Vec<f64> = (0..a.pi.len())
                .filter(|&k| a.pi[k] > 1e-30 && b.pi[k] > 1e-30 && e.pi[k] > 1e-30)
                .map(|k| (b.pi[k] as f64).ln() - (a.pi[k] as f64).ln() - (pw - 1.0) * (e.pi[k] as f64).ln())
                .collect();
            if d.len() >= 2 {
                multi += 1;
                let (lo, hi) = d.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));
                worst = worst.max(hi - lo);
            }
        }
        assert!(differ * 10 > s1.len() * 9, "prior_w {pw}: {differ} of {} targets differ", s1.len());
        assert!(multi * 2 > s1.len() && worst < 1e-3, "prior_w {pw}: {multi} roots checked, worst spread {worst}");
        // softer as beta falls: higher normalised entropy, lower max-share
        let now = (mean_of(&sb, &ent), mean_of(&sb, &maxs));
        assert!(now.0 > last.0 && now.1 < last.1, "prior_w {pw}: entropy / max-share {now:?} after {last:?}");
        last = now;
    }
    // the moves never read beta: the 'cq' move rule (self-play's after temp_moves) and the Gumbel root
    for seat in [
        with_select(with_target(nn(0, 16, true), cq(0.3, 1.0)), Select::Cq { c_visit: 50.0, c_scale: 0.3, min_visits: 0 }),
        with_gumbel(nn(0, 16, true), 4, 1.0, 0.3),
    ] {
        let set_pw = |s: &SeatCfg, pw: f64| match s.clone() {
            SeatCfg::Nn(mut n) => {
                n.policy_target = n.policy_target.with_prior_w(pw).unwrap();
                SeatCfg::Nn(n)
            }
            o => o,
        };
        let (ra, sa) = skewed_selfplay(set_pw(&seat, 1.0));
        let (rb, sb) = skewed_selfplay(set_pw(&seat, 0.6));
        assert_eq!(keys(&ra), keys(&rb));
        assert!(sa.len() == sb.len() && !sa.is_empty());
        let mut differ = 0;
        for (a, b) in sa.iter().zip(&sb) {
            assert_eq!((&a.x, a.player, &a.opts, a.z, a.q, a.sc), (&b.x, b.player, &b.opts, b.z, b.q, b.sc));
            differ += (a.pi != b.pi) as usize;
        }
        assert!(differ * 2 > sa.len(), "{differ} of {}", sa.len());
    }
    // beta outside (0, 1] is refused, and 'visits' takes none
    let t = PolicyTarget::parse("cq", 50.0, 0.3).unwrap();
    assert_eq!(t, cq(0.3, 1.0));
    assert_eq!(t.with_prior_w(0.7), Ok(cq(0.3, 0.7)));
    assert_eq!(t.with_prior_w(1.0), Ok(t));
    for bad in [0.0, -0.5, 1.0001, 2.0, f64::NAN, f64::INFINITY] {
        assert!(t.with_prior_w(bad).is_err(), "prior_w {bad}");
        let seat = with_target(nn(0, 8, true), cq(0.3, bad));
        assert!(Driver::<ToyGame>::new(cfg(vec![seat], vec![vec![0, 0]], &[("m", KeyKind::Nn)])).is_err(), "prior_w {bad}");
    }
    assert_eq!(PolicyTarget::Visits.with_prior_w(1.0), Ok(PolicyTarget::Visits));
    assert!(PolicyTarget::Visits.with_prior_w(0.7).is_err());
}

fn mean_of(s: &[Record], f: &dyn Fn(&[f32]) -> f64) -> f64 {
    let v: Vec<f64> = s.iter().filter(|r| r.pi.len() >= 2).map(|r| f(&r.pi)).collect();
    v.iter().sum::<f64>() / v.len() as f64
}

/// The toy with its two-option decisions marked minor (`SearchGame::minor_decision`).
#[derive(Clone)]
struct MinorToy(ToyGame);

impl SearchGame for MinorToy {
    type Spec = <ToyGame as SearchGame>::Spec;
    const NAME: &'static str = "minortoy";
    const NUM_ACTIONS: usize = ToyGame::NUM_ACTIONS;
    const NUM_FEATURES: usize = ToyGame::NUM_FEATURES;
    const MIN_PLAYERS: usize = ToyGame::MIN_PLAYERS;
    const MAX_PLAYERS: usize = ToyGame::MAX_PLAYERS;
    fn parse_spec(json: &str) -> Result<Self::Spec, String> {
        ToyGame::parse_spec(json)
    }
    fn new_game(spec: &Self::Spec, players: usize, seed: u64) -> Result<Self, String> {
        ToyGame::new_game(spec, players, seed).map(MinorToy)
    }
    fn num_players(&self) -> usize {
        self.0.num_players()
    }
    fn to_move(&self) -> Option<usize> {
        self.0.to_move()
    }
    fn legal_actions(&self, out: &mut Vec<u16>) {
        self.0.legal_actions(out)
    }
    fn step(&mut self, a: u16, chance: &mut Vec<u64>) -> Result<(), String> {
        self.0.step(a, chance)
    }
    fn force_chance(&mut self, forced: &[u64]) {
        self.0.force_chance(forced)
    }
    fn result(&self) -> [f32; MAX_SEATS] {
        self.0.result()
    }
    fn margin(&self, seat: usize) -> f32 {
        self.0.margin(seat)
    }
    fn score(&self, seat: usize) -> f32 {
        self.0.score(seat)
    }
    fn determinize(&mut self, viewer: usize, rng: &mut crate::rng::Rng) {
        self.0.determinize(viewer, rng)
    }
    fn encode(&self, viewer: usize, out: &mut [f32]) {
        self.0.encode(viewer, out)
    }
    fn minor_decision(&self) -> bool {
        let mut v = Vec::new();
        self.0.legal_actions(&mut v);
        v.len() == 2
    }
}

fn with_minor(seat: SeatCfg, m: Option<u32>) -> SeatCfg {
    match seat {
        SeatCfg::Nn(mut s) => {
            s.sims_minor = m;
            SeatCfg::Nn(s)
        }
        s => s,
    }
}

#[test]
fn minor_decisions_get_their_own_budget_and_no_records() {
    let golden = |seat: SeatCfg| {
        let mut c = cfg(vec![seat], vec![vec![0, 0]; 24], &[("m", KeyKind::Nn)]);
        c.seed = 2026;
        let mut d = Driver::<MinorToy>::new(c).unwrap();
        let (r, s) = run(&mut d);
        (r, s, d.stats())
    };
    // without sims_minor the hook is never consulted: the golden records
    let (_, recs, st) = golden(nn(0, 16, true));
    assert_eq!((recs.len(), records_digest(&recs)), GOLDEN_RECORDS);
    assert_eq!(st.minor, 0);
    // a game without minor decisions is unchanged by the setting
    let (_, recs) = golden_selfplay(with_minor(nn(0, 16, true), Some(1)));
    assert_eq!((recs.len(), records_digest(&recs)), GOLDEN_RECORDS);
    // with it: two-option decisions take 1 simulation (the prior's choice) and are never recorded
    let (res, recs, st) = golden(with_minor(nn(0, 16, true), Some(1)));
    assert!(st.minor > 0 && !recs.is_empty());
    assert!(recs.iter().all(|r| r.opts.len() != 2), "a minor decision was recorded");
    assert_eq!(res.len(), 24);
    assert!(res.iter().all(|r| r.over));
    let (_, _, st0) = golden(nn(0, 16, true));
    assert!(st.leaves < st0.leaves, "{} leaves with sims_minor vs {}", st.leaves, st0.leaves);
    // evaluation seats ignore it; zero is rejected
    let (_, _, se) = golden(with_minor(nn(0, 16, false), Some(1)));
    assert_eq!(se.minor, 0);
    assert!(Driver::<MinorToy>::new(cfg(vec![with_minor(nn(0, 8, true), Some(0))], vec![vec![0, 0]], &[("m", KeyKind::Nn)])).is_err());
}

/// The toy whose two-option decisions are light (`SearchGame::sims_divisor` 4).
#[derive(Clone)]
struct DivToy(ToyGame, u32);

impl SearchGame for DivToy {
    type Spec = <ToyGame as SearchGame>::Spec;
    const NAME: &'static str = "divtoy";
    const NUM_ACTIONS: usize = ToyGame::NUM_ACTIONS;
    const NUM_FEATURES: usize = ToyGame::NUM_FEATURES;
    const MIN_PLAYERS: usize = ToyGame::MIN_PLAYERS;
    const MAX_PLAYERS: usize = ToyGame::MAX_PLAYERS;
    fn parse_spec(json: &str) -> Result<Self::Spec, String> {
        ToyGame::parse_spec(json)
    }
    fn new_game(spec: &Self::Spec, players: usize, seed: u64) -> Result<Self, String> {
        ToyGame::new_game(spec, players, seed).map(|g| DivToy(g, DIV.load(std::sync::atomic::Ordering::Relaxed)))
    }
    fn num_players(&self) -> usize {
        self.0.num_players()
    }
    fn to_move(&self) -> Option<usize> {
        self.0.to_move()
    }
    fn legal_actions(&self, out: &mut Vec<u16>) {
        self.0.legal_actions(out)
    }
    fn step(&mut self, a: u16, chance: &mut Vec<u64>) -> Result<(), String> {
        self.0.step(a, chance)
    }
    fn force_chance(&mut self, forced: &[u64]) {
        self.0.force_chance(forced)
    }
    fn result(&self) -> [f32; MAX_SEATS] {
        self.0.result()
    }
    fn margin(&self, seat: usize) -> f32 {
        self.0.margin(seat)
    }
    fn score(&self, seat: usize) -> f32 {
        self.0.score(seat)
    }
    fn determinize(&mut self, viewer: usize, rng: &mut crate::rng::Rng) {
        self.0.determinize(viewer, rng)
    }
    fn encode(&self, viewer: usize, out: &mut [f32]) {
        self.0.encode(viewer, out)
    }
    fn sims_divisor(&self) -> u32 {
        let mut v = Vec::new();
        self.0.legal_actions(&mut v);
        if v.len() == 2 {
            self.1
        } else {
            1
        }
    }
}

/// The divisor of `DivToy`'s two-option decisions, read by its new games (only the test below
/// plays `DivToy`, one driver at a time).
static DIV: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

#[test]
fn light_decisions_divide_the_simulations_and_are_recorded() {
    let go = |div: u32, seat: SeatCfg| {
        DIV.store(div, std::sync::atomic::Ordering::Relaxed);
        let mut c = cfg(vec![seat], vec![vec![0, 0]; 24], &[("m", KeyKind::Nn)]);
        c.seed = 2026;
        c.threads = 1;
        let mut d = Driver::<DivToy>::new(c).unwrap();
        let (r, s) = run(&mut d);
        (r, s, d.stats())
    };
    // divisor 1: the toy's golden self-play records
    let (_, recs, st1) = go(1, nn(0, 16, true));
    assert_eq!((recs.len(), records_digest(&recs)), GOLDEN_RECORDS);
    // divisor 4: two-option decisions searched with a quarter of the simulations, still recorded
    let (res, recs4, st4) = go(4, nn(0, 16, true));
    assert_eq!(res.len(), 24);
    assert!(res.iter().all(|r| r.over));
    assert!(recs4.iter().any(|r| r.opts.len() == 2), "light decisions are recorded");
    assert!(st4.leaves < st1.leaves, "{} leaves with divisor 4 vs {}", st4.leaves, st1.leaves);
    assert!(st4.sims < st1.sims);
    // evaluation seats too
    let (_, _, e1) = go(1, nn(0, 16, false));
    let (_, _, e4) = go(4, nn(0, 16, false));
    assert!(e4.sims < e1.sims);
}

/// The toy with an engine bug: game `PANIC_GID` of a driver seeded `PANIC_SEED` panics at its
/// `PANIC_STEP`-th transition (a real move or one inside a search simulation), after sleeping
/// the spec's milliseconds (the spec is that number: a delay makes the caller wait for the
/// panicking group).
#[derive(Clone)]
struct PanicToy {
    g: ToyGame,
    boom: bool,
    steps: u32,
    sleep_ms: u64,
}

const PANIC_SEED: u64 = 4242;
const PANIC_GID: u32 = 3;
const PANIC_STEP: u32 = 12;

impl SearchGame for PanicToy {
    type Spec = u64;
    const NAME: &'static str = "panictoy";
    const NUM_ACTIONS: usize = ToyGame::NUM_ACTIONS;
    const NUM_FEATURES: usize = ToyGame::NUM_FEATURES;
    const MIN_PLAYERS: usize = ToyGame::MIN_PLAYERS;
    const MAX_PLAYERS: usize = ToyGame::MAX_PLAYERS;
    fn parse_spec(json: &str) -> Result<u64, String> {
        json.trim().parse().map_err(|e| format!("panictoy spec: {e}"))
    }
    fn new_game(spec: &u64, players: usize, seed: u64) -> Result<Self, String> {
        // the game seed init_slot derives for (PANIC_SEED, PANIC_GID)
        let target = Rng::new(split64(PANIC_SEED ^ split64(PANIC_GID as u64 + 1))).next_u64();
        Ok(PanicToy { g: ToyGame::new_game(&Default::default(), players, seed)?, boom: seed == target, steps: 0, sleep_ms: *spec })
    }
    fn num_players(&self) -> usize {
        self.g.num_players()
    }
    fn to_move(&self) -> Option<usize> {
        self.g.to_move()
    }
    fn legal_actions(&self, out: &mut Vec<u16>) {
        self.g.legal_actions(out)
    }
    fn step(&mut self, a: u16, chance: &mut Vec<u64>) -> Result<(), String> {
        self.steps += 1;
        if self.boom && self.steps == PANIC_STEP {
            std::thread::sleep(std::time::Duration::from_millis(self.sleep_ms));
            panic!("toy engine bug at step {PANIC_STEP}");
        }
        self.g.step(a, chance)
    }
    fn force_chance(&mut self, forced: &[u64]) {
        self.g.force_chance(forced)
    }
    fn result(&self) -> [f32; MAX_SEATS] {
        self.g.result()
    }
    fn margin(&self, seat: usize) -> f32 {
        self.g.margin(seat)
    }
    fn score(&self, seat: usize) -> f32 {
        self.g.score(seat)
    }
    fn determinize(&mut self, viewer: usize, rng: &mut crate::rng::Rng) {
        self.g.determinize(viewer, rng)
    }
    fn encode(&self, viewer: usize, out: &mut [f32]) {
        self.g.encode(viewer, out)
    }
}

/// Drive a `PanicToy` self-play run (`groups` groups, the game sleeping `sleep_ms` before its
/// panic, the caller sleeping `answer_ms` before each answer) on a thread; its outcome must come
/// back within 30 s (a hang fails the test instead of hanging it).
fn panicking_run(groups: usize, sleep_ms: u64, answer_ms: u64) -> String {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut c = cfg(vec![nn(0, 8, true)], vec![vec![0, 0]; 8], &[("m", KeyKind::Nn)]);
        c.spec = sleep_ms.to_string();
        c.seed = PANIC_SEED;
        c.concurrency = 8;
        c.threads = 2;
        c.groups = groups;
        let mut d = Driver::<PanicToy>::new(c).unwrap();
        let r = loop {
            match d.next_batch(true) {
                Ok(Some(b)) => {
                    std::thread::sleep(std::time::Duration::from_millis(answer_ms));
                    answer(&mut d, &b);
                }
                Ok(None) => break Err("the run finished without the panic".to_string()),
                Err(e) => break Ok(e),
            }
        };
        drop(d);
        let _ = tx.send(r);
    });
    match rx.recv_timeout(std::time::Duration::from_secs(30)) {
        Ok(r) => r.unwrap(),
        Err(_) => panic!("next_batch hung after a game panicked (groups {groups}, game sleep {sleep_ms} ms, caller delay {answer_ms} ms)"),
    }
}

#[test]
fn a_panicking_game_fails_the_run_instead_of_hanging_it() {
    // the caller already waiting for the panicking group (the game sleeps first), the caller
    // busy when it panics (answers delayed), and both with two groups
    for (groups, sleep_ms, answer_ms) in [(1, 300, 0), (1, 0, 20), (2, 300, 0), (2, 0, 20), (1, 0, 0)] {
        let e = panicking_run(groups, sleep_ms, answer_ms);
        let want = format!("game {PANIC_GID} (driver seed {PANIC_SEED}): search worker panicked: toy engine bug at step {PANIC_STEP}");
        assert!(e.contains(&want), "groups {groups}, sleep {sleep_ms}, delay {answer_ms}: {e}");
    }
}

// ------------------------------------------------------------------ deferred results, queued setups

use crate::toy_team::ToyTeam;

#[test]
fn f16_round_trip_and_rounding() {
    // every finite half value survives f16 -> f32 -> f16; NaN stays NaN
    for h in 0..=u16::MAX {
        let x = f16_to_f32(h);
        if x.is_nan() {
            assert!(f32_to_f16(x) & 0x7c00 == 0x7c00 && f32_to_f16(x) & 0x3ff != 0);
            continue;
        }
        assert_eq!(f32_to_f16(x), h, "{h:#06x} -> {x}");
        // halfway to the next value (same sign) rounds to the even one, just below it down
        if h & 0x7fff < 0x7bff {
            let y = f16_to_f32(h + 1);
            let mid = (x as f64 + y as f64) / 2.0;
            let m = mid as f32;
            assert_eq!(m as f64, mid, "the midpoint is an f32");
            let even = if h & 1 == 0 { h } else { h + 1 };
            assert_eq!(f32_to_f16(m), even, "midpoint of {h:#06x}");
            let below = f32::from_bits(m.to_bits() - 1);
            if below.abs() < m.abs() && m != 0.0 {
                assert_eq!(f32_to_f16(below), h, "just below the midpoint of {h:#06x}");
            }
        }
    }
    assert_eq!(f32_to_f16(1e6), 0x7c00);
    assert_eq!(f32_to_f16(-1e6), 0xfc00);
    assert_eq!(f32_to_f16(1e-9), 0);
    assert_eq!(f32_to_f16(65504.0), 0x7bff);
    assert_eq!(f32_to_f16(65520.0), 0x7c00, "half-way to 65536 rounds to inf (even)");
    assert_eq!(f16_to_f32(f32_to_f16(0.1)), 0.099975586);
}

/// Play every deferred event's matches with the toy heuristic in a second (queue_only) driver,
/// fed by `enqueue` with tags "gid/k", and resolve the events.
fn play_and_resolve(d: &mut Driver<ToyTeam>, m: &mut Driver<ToyGame>, parked_seen: &mut usize) {
    let deferred = d.take_deferred();
    *parked_seen += deferred.len();
    let mut want = HashMap::new();
    for (gid, payload) in &deferred {
        assert!(d.parked_game(*gid).unwrap().deferred());
        let v: serde_json::Value = serde_json::from_str(payload).unwrap();
        let eps = v["episodes"].as_array().unwrap();
        want.insert(*gid, eps.len());
        for (k, e) in eps.iter().enumerate() {
            m.enqueue(&format!("{gid}/{k}"), &e.to_string()).unwrap();
        }
    }
    let mut rows: HashMap<u32, Vec<(usize, GameResult)>> = HashMap::new();
    while let Some(b) = m.next_batch(true).unwrap() {
        answer(m, &b);
    }
    for r in m.take_results() {
        let tag = r.tag.clone().expect("a queued game's row carries its tag");
        let (g, k) = tag.split_once('/').unwrap();
        rows.entry(g.parse().unwrap()).or_default().push((k.parse().unwrap(), r));
    }
    for (gid, mut rs) in rows {
        assert_eq!(rs.len(), want[&gid]);
        rs.sort_by_key(|x| x.0);
        let res: Vec<serde_json::Value> = rs.iter().map(|(_, r)| serde_json::json!({"result": r.result, "scores": r.scores})).collect();
        d.resolve(gid, &serde_json::Value::Array(res).to_string()).unwrap();
    }
}

fn match_driver() -> Driver<ToyGame> {
    let mut mc = cfg(vec![SeatCfg::Bot("heuristic".into())], vec![vec![0, 0]], &[]);
    mc.rolling = true;
    mc.queue_only = true;
    mc.record = false;
    mc.concurrency = 8;
    let mut m = Driver::<ToyGame>::new(mc).unwrap();
    m.resume(u64::MAX, 0).unwrap();
    m
}

#[test]
fn deferred_games_park_and_resolve_with_their_samples() {
    let mut c = cfg(vec![nn(0, 16, true)], vec![vec![0, 0, 0, 0]; 6], &[("m", KeyKind::Nn)]);
    c.concurrency = 6;
    let mut d = Driver::<ToyTeam>::new(c).unwrap();
    let mut m = match_driver();
    assert!(m.next_batch(true).unwrap().is_none(), "queue_only: idle while the queue is empty");
    // the events: every batch answered, nothing finishes before its matches
    let (res, smp) = run(&mut d);
    assert!(res.is_empty() && smp.is_empty());
    assert_eq!(d.pending(), 6);
    assert!(!d.finished());
    let st = d.stats();
    assert_eq!((st.parked, st.deferred, st.resolved), (6, 6, 0));
    assert!(st.parked_bytes > 0);
    // a malformed result is refused and the event stays parked; an unknown game is an error
    let first = d.take_deferred();
    assert_eq!(first.len(), 6);
    assert!(d.resolve(first[0].0, "[[1, 0]]").is_err());
    assert!(d.resolve(999, "[]").is_err());
    assert_eq!(d.pending(), 6);
    d.deferred_out = first;
    let mut seen = 0;
    play_and_resolve(&mut d, &mut m, &mut seen);
    assert_eq!((seen, d.pending()), (6, 0));
    assert!(d.finished());
    let rows = d.take_results();
    let smp = d.take_samples();
    assert_eq!(rows.len(), 6);
    assert_eq!(rows.iter().map(|r| r.samples as usize).sum::<usize>(), smp.len());
    assert!(!smp.is_empty());
    for r in &rows {
        assert!(r.over && r.tag.is_none());
        assert!((r.result.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // a team's seats share the outcome: 1/2, 0 or 1/4 each
        assert_eq!((r.result[0], r.result[1]), (r.result[2], r.result[3]));
        assert!([0.0, 0.25, 0.5].contains(&r.result[0]));
    }
    for s in &smp {
        // z = the event's outcome relative to the decider, q the root search's estimate (both
        // sum to 1), sc = the team's match difference / matches; the features float16 values
        assert!((s.z[..4].iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((s.q[..4].iter().sum::<f32>() - 1.0).abs() < 1e-3);
        assert_eq!((s.z[0], s.sc[0]), (s.z[2], s.sc[2]));
        assert!((s.sc[0] + s.sc[1]).abs() < 1e-6);
        assert!(s.x.iter().all(|&v| f16_to_f32(f32_to_f16(v)) == v));
    }
    let st = d.stats();
    assert_eq!((st.parked, st.parked_bytes, st.resolved, st.games_finished), (0, 0, 6, 6));
    let ms = m.stats();
    assert_eq!((ms.queued, ms.games_finished), (0, 24));
    assert!(m.next_batch(true).unwrap().is_none());
}

#[test]
fn rolling_quotas_count_resolved_games() {
    let mut c = cfg(vec![nn(0, 8, true)], vec![vec![0, 0]], &[("m", KeyKind::Nn)]);
    c.rolling = true;
    c.concurrency = 8;
    let mut d = Driver::<ToyTeam>::new(c).unwrap();
    let mut m = match_driver();
    let mut seen = 0;
    for phase in 0..3u32 {
        d.resume(5, phase).unwrap();
        let mut resolved = 0;
        while let Some(b) = d.next_batch(true).unwrap() {
            answer(&mut d, &b);
            play_and_resolve(&mut d, &mut m, &mut seen);
            resolved += d.take_results().len();
            let _ = d.take_samples();
        }
        resolved += d.take_results().len();
        assert!(d.paused());
        assert!(resolved >= 5, "phase {phase}: {resolved} resolved");
    }
    let st = d.stats();
    assert_eq!(st.games_finished, st.resolved);
    seen += d.take_deferred().len();
    assert_eq!(st.resolved as usize + d.pending(), seen);
}

#[test]
fn queued_setups_replace_the_spec_and_carry_their_tags() {
    let mut c = cfg(vec![SeatCfg::Bot("heuristic".into())], vec![vec![0, 0]; 6], &[]);
    c.record = false;
    let mut plain = Driver::<ToyGame>::new(c.clone()).unwrap();
    assert!(plain.enqueue("bad", r#"{"food": "x"}"#).is_err(), "a bad spec is refused at enqueue");
    assert!(plain.enqueue("bad", r#"{"nope": 1}"#).is_err());
    let (base, _) = run(&mut plain);
    let mut c2 = c.clone();
    c2.rolling = true;
    c2.queue_only = true;
    c2.concurrency = 3;
    let mut q = Driver::<ToyGame>::new(c2).unwrap();
    q.resume(u64::MAX, 0).unwrap();
    for k in 0..4 {
        q.enqueue(&format!("t{k}"), &format!(r#"{{"food": [{}, 0]}}"#, 20 * (k % 2))).unwrap();
    }
    let (rows, _) = run(&mut q);
    assert_eq!(q.queued(), 0);
    let tags: Vec<_> = rows.iter().map(|r| (r.gid, r.tag.clone().unwrap())).collect();
    assert_eq!(tags, (0..4).map(|k| (k, format!("t{k}"))).collect::<Vec<_>>());
    // game g with no extra food is the driver's game g (the same streams); 20 food changes it
    for r in rows.iter().filter(|r| r.gid % 2 == 0) {
        let b = &base[r.gid as usize];
        assert_eq!((r.steps, &r.scores), (b.steps, &b.scores));
    }
    assert!(rows.iter().filter(|r| r.gid % 2 == 1).any(|r| r.scores != base[r.gid as usize].scores));
    // queue_only needs a rolling driver
    let mut bad = c;
    bad.queue_only = true;
    assert!(Driver::<ToyGame>::new(bad).is_err());
}

#[test]
fn set_spec_plays_the_new_spec_in_the_games_started_after_it() {
    // docs/search.md 2.13: a rolling driver told a new spec between phases starts its next games with it, and the
    // games in flight keep theirs: game g started after the switch is game g of a driver made with the new spec
    // (the same streams), every earlier game is the old spec's
    let mut c = cfg(vec![SeatCfg::Bot("heuristic".into())], vec![vec![0, 0]], &[]);
    c.record = false;
    c.rolling = true;
    c.concurrency = 3;
    let new = r#"{"food": [20, 0]}"#;
    let phases = |switch: Option<&str>, initial: &str| {
        let mut cc = c.clone();
        cc.spec = initial.into();
        let mut d = Driver::<ToyGame>::new(cc).unwrap();
        d.resume(6, 0).unwrap();
        let (mut rows, _) = run(&mut d);
        let started = d.stats().games_finished + d.in_flight().games as u64;
        if let Some(s) = switch {
            d.set_spec(s).unwrap();
        }
        d.resume(6, 0).unwrap();
        rows.extend(run(&mut d).0);
        rows.sort_by_key(|r| r.gid);
        (rows, started)
    };
    let (switched, g0) = phases(Some(new), "{}");
    let (old, _) = phases(None, "{}");
    let (fresh, _) = phases(None, new);
    assert!(g0 > 6 && switched.iter().any(|r| r.gid as u64 >= g0), "games on both sides of the switch");
    let by = |v: &[GameResult], g: u32| v.iter().find(|r| r.gid == g).map(|r| (r.steps, r.scores.clone()));
    let mut changed = 0;
    for r in &switched {
        let want = if (r.gid as u64) < g0 { by(&old, r.gid) } else { by(&fresh, r.gid) };
        if let Some(w) = want {
            assert_eq!((r.steps, r.scores.clone()), w, "game {} (started {} the switch)", r.gid, if (r.gid as u64) < g0 { "before" } else { "after" });
        }
        changed += (r.gid as u64 >= g0 && by(&old, r.gid).is_some_and(|o| o != (r.steps, r.scores.clone()))) as u32;
    }
    assert!(changed > 0, "the new spec changed a game");
    // refused: a bad spec (the old one kept), a running phase, a driver that is not rolling
    let mut d = Driver::<ToyGame>::new(c.clone()).unwrap();
    assert!(d.set_spec(r#"{"food": "x"}"#).is_err());
    d.resume(4, 0).unwrap();
    assert!(d.set_spec(new).unwrap_err().contains("phase"));
    run(&mut d);
    d.set_spec(new).unwrap();
    let mut plain = c.clone();
    plain.rolling = false;
    plain.games = vec![vec![0, 0]; 2];
    assert!(Driver::<ToyGame>::new(plain).unwrap().set_spec(new).unwrap_err().contains("rolling"));
}

#[test]
fn queued_setups_are_shared_between_the_groups() {
    // a queue_only driver of 2 groups x 8 slots fed 8 setups (as an evaluation's matches arrive):
    // each group gets 4 of them, so the groups' batches alternate at full size (the first group
    // re-dispatched used to take every queued setup: 7 / 1); the same games as one at a time
    let mut c = cfg(vec![nn(0, 8, false)], vec![vec![0, 0]], &[("m", KeyKind::Nn)]);
    c.rolling = true;
    c.queue_only = true;
    c.record = false;
    c.concurrency = 16;
    let mut d = Driver::<ToyGame>::new(c.clone()).unwrap();
    d.resume(u64::MAX, 0).unwrap();
    for k in 0..8 {
        d.enqueue(&format!("t{k}"), "{}").unwrap();
    }
    let mut sizes = Vec::new();
    let mut rows = Vec::new();
    while let Some(b) = d.next_batch(true).unwrap() {
        sizes.push((b.group, b.b));
        answer(&mut d, &b);
        rows.extend(d.take_results());
    }
    rows.extend(d.take_results());
    assert_eq!(rows.len(), 8);
    // Each `next_batch` call reports one group's CURRENTLY active game count, so the same size
    // repeats across many calls while those games play out their turns (a batch per ply, not per
    // dispatch); `sizes`' overall interleaving between the two groups is thread-scheduling order
    // (which group's worker thread gets to report next) and is not deterministic under load - a
    // loaded machine can give group 1 several consecutive reports before group 0's first one, so
    // asserting on absolute positions in the combined `sizes` vector is flaky (seen empirically:
    // group 1 reporting (1, 4) three times running before group 0's first (0, 4)). What IS
    // deterministic (`queue_share`, not thread scheduling) is each GROUP's own sequence of
    // distinct sizes: one setup as it arrives, then a fair share of 4 once the rest of the queue
    // is split between the two groups (the regression this guards: the first group re-dispatched
    // used to take the whole queue, 7 / 1).
    let own_sequence = |g: usize| -> Vec<usize> {
        let mut v: Vec<usize> = sizes.iter().filter(|&&(gg, _)| gg == g).map(|&(_, b)| b).collect();
        v.dedup();
        v.truncate(2);
        v
    };
    assert_eq!(own_sequence(0), vec![1, 4], "group 0's own dispatch sizes: {sizes:?}");
    assert_eq!(own_sequence(1), vec![1, 4], "group 1's own dispatch sizes: {sizes:?}");
    assert!(sizes.iter().all(|&(_, b)| b <= 4), "{sizes:?}");
    // a game's play depends on its setup and gid only: the same rows with one group
    let mut c1 = c;
    c1.groups = 1;
    let mut d1 = Driver::<ToyGame>::new(c1).unwrap();
    d1.resume(u64::MAX, 0).unwrap();
    for k in 0..8 {
        d1.enqueue(&format!("t{k}"), "{}").unwrap();
    }
    let (rows1, _) = run(&mut d1);
    rows.sort_by_key(|r| r.gid);
    let key = |r: &GameResult| (r.gid, r.tag.clone(), r.steps, r.scores.clone());
    assert_eq!(rows.iter().map(key).collect::<Vec<_>>(), rows1.iter().map(key).collect::<Vec<_>>());
}

#[test]
fn queued_setups_fill_every_free_slot_when_the_queue_is_long() {
    // 24 setups for 16 slots: every slot plays (the groups' shares never leave a free slot idle
    // while setups wait), 8 + 8 games in the first full batches
    let mut c = cfg(vec![nn(0, 8, false)], vec![vec![0, 0]], &[("m", KeyKind::Nn)]);
    c.rolling = true;
    c.queue_only = true;
    c.record = false;
    c.concurrency = 16;
    let mut d = Driver::<ToyGame>::new(c).unwrap();
    d.resume(u64::MAX, 0).unwrap();
    for k in 0..24 {
        d.enqueue(&format!("t{k}"), "{}").unwrap();
    }
    let mut sizes = Vec::new();
    let mut n = 0;
    while let Some(b) = d.next_batch(true).unwrap() {
        sizes.push(b.b);
        answer(&mut d, &b);
        n += d.take_results().len();
    }
    n += d.take_results().len();
    assert_eq!(n, 24);
    assert_eq!(&sizes[2..4], &[8, 8], "{sizes:?}");
}

// ------------------------------------------------------------------ a slow step holds up its group only

/// The toy game whose step `at` sleeps `ms` once (the spec "<at> <ms>"; the clones a search
/// makes share the flag, so the real game's step or a simulation's sleeps, once): an engine's
/// slow path (MTG Zero: a castability check that ran for minutes, its ENGINE.md 51.3, 53).
#[derive(Clone)]
struct SlowToy {
    g: ToyGame,
    steps: u32,
    at: u32,
    ms: u64,
    slept: std::sync::Arc<AtomicBool>,
}

impl SearchGame for SlowToy {
    type Spec = (u32, u64);
    const NAME: &'static str = "slowtoy";
    const NUM_ACTIONS: usize = ToyGame::NUM_ACTIONS;
    const NUM_FEATURES: usize = ToyGame::NUM_FEATURES;
    const MIN_PLAYERS: usize = ToyGame::MIN_PLAYERS;
    const MAX_PLAYERS: usize = ToyGame::MAX_PLAYERS;
    fn parse_spec(json: &str) -> Result<(u32, u64), String> {
        let v: Vec<u64> = json.split_whitespace().map(|x| x.parse().map_err(|e| format!("slowtoy spec: {e}"))).collect::<Result<_, _>>()?;
        Ok(if v.len() == 2 { (v[0] as u32, v[1]) } else { (0, 0) })
    }
    fn new_game(spec: &(u32, u64), players: usize, seed: u64) -> Result<Self, String> {
        Ok(SlowToy { g: ToyGame::new_game(&Default::default(), players, seed)?, steps: 0, at: spec.0, ms: spec.1, slept: Default::default() })
    }
    fn num_players(&self) -> usize {
        self.g.num_players()
    }
    fn to_move(&self) -> Option<usize> {
        self.g.to_move()
    }
    fn legal_actions(&self, out: &mut Vec<u16>) {
        self.g.legal_actions(out)
    }
    fn step(&mut self, a: u16, chance: &mut Vec<u64>) -> Result<(), String> {
        self.steps += 1;
        if self.ms > 0 && self.steps == self.at && !self.slept.swap(true, Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_millis(self.ms));
        }
        self.g.step(a, chance)
    }
    fn force_chance(&mut self, forced: &[u64]) {
        self.g.force_chance(forced)
    }
    fn result(&self) -> [f32; MAX_SEATS] {
        self.g.result()
    }
    fn margin(&self, seat: usize) -> f32 {
        self.g.margin(seat)
    }
    fn score(&self, seat: usize) -> f32 {
        self.g.score(seat)
    }
    fn determinize(&mut self, viewer: usize, rng: &mut crate::rng::Rng) {
        self.g.determinize(viewer, rng)
    }
    fn encode(&self, viewer: usize, out: &mut [f32]) {
        self.g.encode(viewer, out)
    }
}

#[test]
fn a_slow_step_holds_up_its_own_group_only() {
    // co-training's episode player (a queue_only driver, 2 groups of 4) with 8 queued games, the
    // first of which sleeps 1.5 s in one step: the other group's 4 games are searched, evaluated
    // and finished meanwhile. (Served in strict round-robin order, every game of the driver waited
    // for the slow one - and in co-training the draft network behind them: MTG Zero's smoke idled
    // ~10 minutes, its ENGINE.md 53.) The slow step is counted.
    const MS: u64 = 1500;
    let mut c = cfg(vec![nn(0, 8, false)], vec![vec![0, 0]], &[("m", KeyKind::Nn)]);
    c.spec = String::new();
    c.rolling = true;
    c.queue_only = true;
    c.record = false;
    c.concurrency = 8;
    let mut d = Driver::<SlowToy>::new(c).unwrap();
    d.resume(u64::MAX, 0).unwrap();
    for k in 0..8 {
        d.enqueue(&format!("t{k}"), if k == 0 { "10 1500" } else { "" }).unwrap();
    }
    let t0 = Instant::now();
    let mut last = [t0; 2];
    let mut stalled = None;
    let mut done = Vec::new();
    let mut members = [std::collections::BTreeSet::new(), std::collections::BTreeSet::new()];
    let mut served = Vec::new();
    while let Some(b) = d.next_batch(true).unwrap() {
        let now = Instant::now();
        members[b.group].extend(b.gid.iter().copied());
        served.push((b.group, now));
        if stalled.is_none() && now.duration_since(last[b.group]).as_millis() as u64 >= MS * 2 / 3 {
            stalled = Some((b.group, last[b.group], now));
        }
        last[b.group] = now;
        answer(&mut d, &b);
        done.extend(d.take_results().into_iter().map(|r| (r.gid, Instant::now())));
    }
    done.extend(d.take_results().into_iter().map(|r| (r.gid, Instant::now())));
    assert_eq!(done.len(), 8);
    let (g, t_before, t_back) = stalled.expect("the slow step");
    assert!(members[g].contains(&0), "the slow game (gid 0) is in the group that stalled: {members:?}");
    assert!(members[1 - g].len() >= 2 && members[0].len() + members[1].len() == 8, "{members:?}");
    // while the slow game slept, the other group went on to its last batch (in strict
    // round-robin order it got one batch at most)
    let during = served.iter().filter(|&&(h, t)| h != g && t > t_before && t < t_back).count();
    let last_other = served.iter().filter(|x| x.0 != g).map(|x| x.1).max().unwrap();
    assert!(during >= 10 && last_other < t_back, "the other group: {during} batches while the slow game slept, its last {:?} before the slow group came back", t_back.saturating_duration_since(last_other));
    let st = d.stats();
    assert!(st.step_max_s >= MS as f64 * 1e-3 && st.step_max_gid == 0, "{} s, game {}", st.step_max_s, st.step_max_gid);
    assert_eq!(st.slow_steps, 0, "under SLOW_STEP_NS");
    // in_flight never waits for the sleeping search thread
    let mut c2 = cfg(vec![nn(0, 8, false)], vec![vec![0, 0]], &[("m", KeyKind::Nn)]);
    c2.spec = String::new();
    c2.rolling = true;
    c2.queue_only = true;
    c2.record = false;
    c2.concurrency = 2;
    c2.groups = 1;
    let mut d2 = Driver::<SlowToy>::new(c2).unwrap();
    d2.resume(u64::MAX, 0).unwrap();
    d2.enqueue("slow", "1 1500").unwrap();
    std::thread::sleep(std::time::Duration::from_millis(300));
    let t = Instant::now();
    let f = d2.in_flight();
    assert!(t.elapsed().as_millis() < 500, "in_flight waited {:?} for a search thread", t.elapsed());
    assert_eq!(f.games, 1);
    while let Some(b) = d2.next_batch(true).unwrap() {
        answer(&mut d2, &b);
    }
}

// ------------------------------------------------------------------ the move rule (Select) and the move log

fn with_select(seat: SeatCfg, sel: Select) -> SeatCfg {
    match seat {
        SeatCfg::Nn(mut s) => {
            s.select = sel;
            SeatCfg::Nn(s)
        }
        s => s,
    }
}

const CQ: Select = Select::Cq { c_visit: 50.0, c_scale: 1.0, min_visits: 0 };
const CQ2: Select = Select::Cq { c_visit: 50.0, c_scale: 1.0, min_visits: 2 };

/// FNV-1a over result rows (gid, steps, decisions, result, scores, over): a match's games.
fn results_digest(rows: &[GameResult]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut eat = |b: &[u8]| {
        for &x in b {
            h ^= x as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    };
    for r in rows {
        eat(&r.gid.to_le_bytes());
        eat(&r.steps.to_le_bytes());
        eat(&r.decisions.to_le_bytes());
        for v in r.result.iter().chain(&r.scores) {
            eat(&v.to_bits().to_le_bytes());
        }
        eat(&[r.over as u8]);
    }
    h
}

/// An evaluation match of the deterministic evaluator: key m (rule `a`) vs key o (rule `b`),
/// 40 toy games at 16 simulations, seats alternating.
fn golden_match(a: Select, b: Select) -> Vec<GameResult> {
    let seats = vec![with_select(nn(0, 16, false), a), with_select(nn(1, 16, false), b)];
    let games = (0..40).map(|g| if g % 2 == 0 { vec![0, 1] } else { vec![1, 0] }).collect();
    let mut c = cfg(seats, games, &[("m", KeyKind::Nn), ("o", KeyKind::Nn)]);
    c.seed = 2026;
    c.record = false;
    run(&mut Driver::<ToyGame>::new(c).unwrap()).0
}

#[test]
fn select_visits_and_the_move_log_change_nothing() {
    // the explicit default rule and a move log reproduce the golden records bit for bit
    let (r0, recs) = golden_selfplay(with_select(nn(0, 16, true), Select::Visits));
    assert_eq!(records_digest(&recs), GOLDEN_RECORDS.1);
    let mut c = cfg(vec![nn(0, 16, true)], vec![vec![0, 0]; 24], &[("m", KeyKind::Nn)]);
    c.seed = 2026;
    c.move_log = 1000;
    let (r1, recs1) = run(&mut Driver::<ToyGame>::new(c).unwrap());
    assert_eq!(records_digest(&recs1), GOLDEN_RECORDS.1);
    let key = |r: &GameResult| (r.gid, r.steps, r.scores.clone(), r.samples, r.decisions);
    assert_eq!(r0.iter().map(key).collect::<Vec<_>>(), r1.iter().map(key).collect::<Vec<_>>());
    assert!(r0.iter().all(|r| r.moves.is_empty()));
    // every real move is logged (no external seats here), each legal and from its decider's list
    for r in &r1 {
        assert_eq!(r.moves.len() as u64, r.steps.min(1000));
        assert!(r.moves.iter().all(|(p, a, legal)| (*p as usize) < r.n && legal.contains(a)));
    }
    // an evaluation match at the default rule: the games of zero-core before the rule existed
    assert_eq!(results_digest(&golden_match(Select::Visits, Select::Visits)), GOLDEN_MATCH_VISITS);
}
#[test]
fn paired_games_share_their_streams() {
    // DriverCfg::paired: games 2k and 2k + 1 draw every random stream from k. Two seats that search
    // alike (same settings, the same answers) swapped between the two games of a pair - the table
    // of a match - then play the same game twice: same moves, same scores, the result mirrored
    // onto the swapped seats' keys. Unpaired, the two games of a "pair" are unrelated.
    let games: Vec<Vec<u16>> = (0..40).map(|g| if g % 2 == 0 { vec![0, 1] } else { vec![1, 0] }).collect();
    let play = |paired: bool, games: Vec<Vec<u16>>| {
        let mut c = cfg(vec![nn(0, 16, false), nn(1, 16, false)], games, &[("m", KeyKind::Nn), ("o", KeyKind::Nn)]);
        c.seed = 2026;
        c.record = false;
        c.move_log = 1000;
        c.paired = paired;
        run(&mut Driver::<ToyGame>::new(c).unwrap()).0
    };
    let game = |r: &GameResult| (r.steps, r.decisions, r.scores.clone(), r.result.clone(), r.moves.clone());
    let p = play(true, games.clone());
    assert_eq!(p.len(), 40);
    for k in 0..20 {
        assert_eq!(game(&p[2 * k]), game(&p[2 * k + 1]), "pair {k}: the same deal from each seat");
    }
    let u = play(false, games.clone());
    assert!((0..20).any(|k| game(&u[2 * k]) != game(&u[2 * k + 1])), "unpaired games share nothing");
    // a pair's game is the game the pair's id would play unpaired... from stream k, not 2k: the
    // unpaired driver's game k (both seats alike) is the paired driver's games 2k and 2k + 1
    for k in 0..20 {
        assert_eq!(game(&p[2 * k]), game(&u[k]), "pair {k} plays the unpaired game {k}'s streams");
    }
    // an odd count: the last game has no partner and plays its pair's stream alone
    let odd = play(true, games[..7].to_vec());
    assert_eq!(odd.len(), 7);
    assert_eq!(game(&odd[6]), game(&p[6]));
    // off: today's match, bit for bit (the golden of before the setting)
    assert_eq!(results_digest(&golden_match(Select::Visits, Select::Visits)), GOLDEN_MATCH_VISITS);
}

// `golden_match(Visits, Visits)`: the same digest on zero-core 3c26d9f (the driver before
// `Select`, the same match without the select field)
const GOLDEN_MATCH_VISITS: u64 = 12492034373733022939;

#[test]
fn select_cq_golden_records_and_matches() {
    // the 'cq' rule's self-play records and evaluation matches, recorded when the rule was
    // productionized: a change of its moves (or of the cq logits) shows here
    let (_, recs) = golden_selfplay(with_select(nn(0, 16, true), CQ));
    assert_eq!((recs.len(), records_digest(&recs)), GOLDEN_SELECT_CQ_RECORDS);
    assert_eq!(results_digest(&golden_match(CQ, Select::Visits)), GOLDEN_MATCH_CQ);
    assert_eq!(results_digest(&golden_match(CQ2, CQ2)), GOLDEN_MATCH_CQ2);
}
const GOLDEN_SELECT_CQ_RECORDS: (usize, u64) = (329, 8716559848855127382);
const GOLDEN_MATCH_CQ: u64 = 10451580073337998459;
const GOLDEN_MATCH_CQ2: u64 = 2225760264326051811;

#[test]
fn select_rules_change_only_moves_after_temp_moves() {
    // the first temp_moves searched decisions are sampled whatever the rule, so the games agree
    // up to there; the other rules then play differently somewhere, and every game still ends
    let log = |sel: Select| {
        let mut c = cfg(vec![with_select(nn(0, 16, true), sel)], vec![vec![0, 0]; 24], &[("m", KeyKind::Nn)]);
        c.seed = 2026;
        c.move_log = 1000;
        run(&mut Driver::<ToyGame>::new(c).unwrap()).0
    };
    let base = log(Select::Visits);
    for sel in [CQ, CQ2] {
        let other = log(sel);
        assert_eq!(other.len(), base.len());
        assert!(other.iter().all(|r| r.over));
        let mut differ = 0;
        for (a, b) in base.iter().zip(&other) {
            // the games' first moves (setup + the sampled opening) agree
            let k = a.moves.iter().zip(&b.moves).take_while(|(x, y)| x == y).count();
            assert!(k >= 10.min(a.moves.len()), "{sel:?}: game {} differs at move {k}", a.gid);
            differ += (a.moves != b.moves) as usize;
        }
        assert!(differ > 0, "{sel:?} never changed a move");
    }
    // evaluation seats never sample: every move follows the rule, so 'cq' vs 'visits' plays other
    // games than 'visits' vs 'visits'
    assert_ne!(results_digest(&golden_match(CQ, Select::Visits)), results_digest(&golden_match(Select::Visits, Select::Visits)));
}

#[test]
fn select_choose_by_hand() {
    // sigma = (50 + 27) x Q dwarfs ln P0: the one-visit option with the best Q wins the plain cq
    // argmax; a guard of 2 visits takes option 1 (4 visits, the better logit of the two left), a
    // guard above every count the most visited option; the visits rule option 0
    let p0 = [0.9f32, 0.08, 0.02];
    let n = [27u32, 4, 1];
    let w = [27.0 * 0.40, 4.0 * 0.47, 1.0 * 0.60];
    let v = 0.45;
    let lg = crate::tree::completed_q_logits(&p0, &n, &w, v, 50.0, 1.0);
    let pi = crate::tree::completed_q_policy(&p0, &n, &w, v, 50.0, 1.0);
    let am = |x: &[f64]| (0..x.len()).fold(0, |b, k| if x[k] > x[b] { k } else { b });
    assert_eq!(am(&lg), am(&pi.iter().map(|&x| x as f64).collect::<Vec<_>>()));
    assert_eq!(Select::Visits.choose(&p0, &n, &w, v), 0);
    assert_eq!(CQ.choose(&p0, &n, &w, v), 2);
    assert_eq!(Select::Cq { c_visit: 50.0, c_scale: 1.0, min_visits: 1 }.choose(&p0, &n, &w, v), 2);
    assert_eq!(CQ2.choose(&p0, &n, &w, v), 1);
    assert_eq!(Select::Cq { c_visit: 50.0, c_scale: 1.0, min_visits: 5 }.choose(&p0, &n, &w, v), 0);
    assert_eq!(Select::Cq { c_visit: 50.0, c_scale: 1.0, min_visits: 1000 }.choose(&p0, &n, &w, v), 0);
    // c_scale 0: the logits are ln P0, so the prior's favourite
    assert_eq!(Select::Cq { c_visit: 50.0, c_scale: 0.0, min_visits: 0 }.choose(&p0, &n, &w, v), 0);
    // an unvisited option is completed at v_mix: with a high prior and no guard it can win
    let (p0u, nu, wu) = ([0.5f32, 0.5], [3u32, 0], [3.0 * 0.2, 0.0]);
    assert_eq!(CQ.choose(&p0u, &nu, &wu, 0.9), 1, "v_mix = (0.9 + 3 x 0.2) / 4 > 0.2");
    assert_eq!(Select::Cq { c_visit: 50.0, c_scale: 1.0, min_visits: 1 }.choose(&p0u, &nu, &wu, 0.9), 0);
    // nothing visited: both rules take the prior's argmax; exact ties the first
    assert_eq!(CQ.choose(&[0.2, 0.7, 0.1], &[0, 0, 0], &[0.0; 3], 0.5), 1);
    assert_eq!(CQ2.choose(&[0.2, 0.7, 0.1], &[0, 0, 0], &[0.0; 3], 0.5), 1);
    assert_eq!(CQ.choose(&[0.5, 0.5], &[2, 2], &[1.0, 1.0], 0.5), 0);
    assert_eq!(CQ.choose(&[], &[], &[], 0.5), 0);
    assert_eq!(Select::Visits.choose(&[], &[], &[], 0.5), 0);
}

#[test]
fn select_settings_are_parsed() {
    assert_eq!(Select::parse("visits", 0, 50.0, 1.0), Ok(Select::Visits));
    assert_eq!(Select::parse("cq", 0, 50.0, 1.0), Ok(CQ));
    assert_eq!(Select::parse("cq", 2, 25.0, 0.5), Ok(Select::Cq { c_visit: 25.0, c_scale: 0.5, min_visits: 2 }));
    assert!(Select::parse("visits", 2, 50.0, 1.0).is_err(), "the guard is cq's");
    assert!(Select::parse("q", 0, 50.0, 1.0).is_err());
    assert!(Select::parse("cq", 0, f64::NAN, 1.0).is_err());
    assert!(Select::parse("cq", 0, 50.0, -1.0).is_err());
}

/// `answer` with skewed priors (a hash of the features and the action; `answer`'s are uniform,
/// and under uniform priors the cq rule's argmax is the same at every c_scale > 0).
fn answer_skewed<G: SearchGame>(d: &mut Driver<G>, b: &Batch) {
    let f = G::NUM_FEATURES;
    let mut pri = Vec::new();
    for r in 0..b.b {
        let h: f32 = b.x[r * f..(r + 1) * f].iter().enumerate().map(|(i, x)| x * ((i * 5 % 11) as f32 - 5.0)).sum();
        let (lo, hi) = (b.opt_off[r] as usize, b.opt_off[r + 1] as usize);
        let e: Vec<f32> = b.opt_action[lo..hi].iter().map(|&a| (1.5 * (a as f32 * 1.7 + h * 0.3).sin()).exp()).collect();
        let t: f32 = e.iter().sum();
        pri.extend(e.iter().map(|x| x / t));
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

#[test]
fn select_transform_moves_only_the_moves_never_the_targets() {
    // A self-play seat's 'cq' rule with its own completed-Q transform (the bindings' seat keys
    // select_c_visit / select_c_scale, run settings *_select_c_scale) next to a 'cq' target at
    // another one: the moves follow the rule's transform alone, the targets the target's alone.
    // A: target c_scale 1.0, rule RS; B: target RS, rule RS (one shared knob, as before the
    // keys); E: target 0.0 (= the floored prior), rule RS; C: target 1.0, rule 1.0. Skewed priors
    // (`answer_skewed`), so that the rule's c_scale matters.
    const RS: f64 = 0.3;
    let seat = |target: f64, rule: f64| {
        let s = with_target(nn(0, 16, true), PolicyTarget::CompletedQ { c_visit: 50.0, c_scale: target, prior_w: 1.0 });
        with_select(s, Select::Cq { c_visit: 50.0, c_scale: rule, min_visits: 0 })
    };
    let play = |s: SeatCfg| {
        let mut c = cfg(vec![s], vec![vec![0, 0]; 24], &[("m", KeyKind::Nn)]);
        c.seed = 2026;
        let mut d = Driver::<ToyGame>::new(c).unwrap();
        let (mut res, mut smp) = (Vec::new(), Vec::new());
        while let Some(b) = d.next_batch(true).unwrap() {
            answer_skewed(&mut d, &b);
            res.extend(d.take_results());
            smp.extend(d.take_samples());
        }
        res.extend(d.take_results());
        smp.extend(d.take_samples());
        res.sort_by_key(|r| r.gid);
        (res, smp)
    };
    let (ra, sa) = play(seat(1.0, RS));
    let (rb, sb) = play(seat(RS, RS));
    let (re, se) = play(seat(0.0, RS));
    let (rc, _) = play(seat(1.0, 1.0));
    let key = |r: &GameResult| (r.gid, r.steps, r.scores.clone(), r.samples, r.decisions);
    let keys = |rs: &[GameResult]| rs.iter().map(key).collect::<Vec<_>>();
    // the same rule: the same games and the same records but for pi
    assert_eq!(keys(&ra), keys(&rb));
    assert_eq!(keys(&ra), keys(&re));
    assert!(sa.len() == sb.len() && sa.len() == se.len() && !sa.is_empty());
    // another rule transform: other games (the rule's transform reaches the moves)
    assert_ne!(keys(&ra), keys(&rc), "the rule's c_scale never changed a game");
    let (mut differ, mut checked, mut worst) = (0, 0, 0.0f64);
    for ((a, b), e) in sa.iter().zip(&sb).zip(&se) {
        for r in [b, e] {
            assert_eq!((&a.x, a.player, a.n_players, &a.opts, a.z, a.q, a.sc, a.net), (&r.x, r.player, r.n_players, &r.opts, r.z, r.q, r.sc, r.net));
        }
        differ += (a.pi != b.pi) as usize;
        // ln pi_s = ln P0 + s (c_visit + max N) completedQ - ln Z_s, so per option
        // (ln pi_A - ln pi_E) - (1.0 / RS) (ln pi_B - ln pi_E) is one constant: A's target is the
        // c_scale 1.0 one of the same root, whatever the rule's RS
        let d: Vec<f64> = (0..a.pi.len())
            .filter(|&k| a.pi[k] > 1e-30 && b.pi[k] > 1e-30 && e.pi[k] > 1e-30)
            .map(|k| {
                let l = |p: &[f32]| (p[k] as f64).ln();
                (l(&a.pi) - l(&e.pi)) - (1.0 / RS) * (l(&b.pi) - l(&e.pi))
            })
            .collect();
        if d.len() >= 2 {
            checked += 1;
            let (lo, hi) = d.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));
            worst = worst.max(hi - lo);
        }
    }
    assert!(differ * 10 > sa.len() * 9, "{differ} of {} targets differ between target c_scale 1.0 and {RS}", sa.len());
    assert!(checked * 2 > sa.len(), "{checked} of {} records checked", sa.len());
    assert!(worst < 1e-3, "the targets are not c_scale 1.0's of the same roots: spread {worst}");
}

#[test]
fn an_unrecorded_seat_plays_the_same_games_and_records_nothing() {
    // A league game (zero's run setting selfplay_league, docs/az.md 14): the current network's
    // self-play seat (key m) against a past checkpoint's (key L), seats alternating. With L's
    // `record` false the games are the same move for move (its search, noise, temperature and
    // playout-cap draws are a recorded seat's) and the records are exactly the m seat's of the
    // same games recorded on both sides: z, q and sc from the m seat's view.
    let unrecorded = |key: u16, record: bool| match nn(key, 16, true) {
        SeatCfg::Nn(mut s) => {
            s.record = record;
            SeatCfg::Nn(s)
        }
        _ => unreachable!(),
    };
    let play = |l_record: bool| {
        let l = unrecorded(1, l_record);
        let games: Vec<Vec<u16>> = (0..24).map(|g| if g % 2 == 0 { vec![0, 1] } else { vec![1, 0] }).collect();
        let c = cfg(vec![nn(0, 16, true), l], games, &[("m", KeyKind::Nn), ("L", KeyKind::Nn)]);
        let mut d = Driver::<ToyGame>::new(c).unwrap();
        let (mut res, mut smp) = (Vec::new(), Vec::new());
        while let Some(b) = d.next_batch(true).unwrap() {
            answer(&mut d, &b);
            res.extend(d.take_results());
            smp.extend(d.take_samples());
        }
        res.extend(d.take_results());
        smp.extend(d.take_samples());
        // each game's records, in the order the games were emitted
        let mut per: Vec<(u32, Vec<Record>)> = Vec::new();
        let mut it = smp.into_iter();
        for r in &res {
            per.push((r.gid, it.by_ref().take(r.samples as usize).collect()));
        }
        assert!(it.next().is_none());
        per.sort_by_key(|p| p.0);
        res.sort_by_key(|r| r.gid);
        (res, per)
    };
    let (ra, pa) = play(true);
    let (rb, pb) = play(false);
    let key = |r: &GameResult| (r.gid, r.steps, r.scores.clone(), r.decisions, r.result.clone(), r.over);
    assert_eq!(ra.iter().map(key).collect::<Vec<_>>(), rb.iter().map(key).collect::<Vec<_>>());
    let (mut kept, mut dropped) = (0, 0);
    for ((g, a), (_, b)) in pa.iter().zip(&pb) {
        let me = (*g % 2) as u8;                           // the m seat of game g
        let want: Vec<&Record> = a.iter().filter(|r| r.player == me).collect();
        assert_eq!(want.len(), b.len(), "game {g}");
        for (x, y) in want.iter().zip(b) {
            assert_eq!((&x.x, x.player, x.n_players, &x.opts, &x.pi, x.z, x.q, x.sc, x.net), (&y.x, y.player, y.n_players, &y.opts, &y.pi, y.z, y.q, y.sc, y.net));
        }
        kept += b.len();
        dropped += a.len() - b.len();
        let rb_g = &rb[*g as usize];
        assert_eq!(rb_g.samples as usize, b.len());
        // z from the m seat's view: its result in the game
        for y in b {
            assert_eq!(y.z[0], rb_g.result[me as usize]);
        }
    }
    assert!(kept > 0 && dropped > 0, "kept {kept}, dropped {dropped}");
    // an unrecorded seat alone: nothing at all
    let mut d = Driver::<ToyGame>::new(cfg(vec![unrecorded(0, false)], vec![vec![0, 0]; 8], &[("m", KeyKind::Nn)])).unwrap();
    let (res, smp) = run(&mut d);
    assert_eq!(res.len(), 8);
    assert!(smp.is_empty() && res.iter().all(|r| r.samples == 0));
}

// ------------------------------------------------------------------ root minimum visits

use crate::mull_game::{mull_eval, Mull, MULL};

fn with_floor(seat: SeatCfg, k: u32, max_actions: u32) -> SeatCfg {
    match seat {
        SeatCfg::Nn(mut s) => {
            s.root_min_visits = k;
            s.root_min_visits_max_actions = max_actions;
            SeatCfg::Nn(s)
        }
        s => s,
    }
}

#[test]
fn root_floor_takes_full_searches_of_few_options_with_the_budget() {
    assert_eq!(root_floor(4, 4, true, 2, 64), 4);
    assert_eq!(root_floor(4, 4, true, 4, 64), 4);
    assert_eq!(root_floor(4, 4, true, 2, 16), 4, "2 x 2 x 4 = 16: half the budget");
    assert_eq!(root_floor(4, 4, true, 2, 15), 0, "the floor would take more than half");
    assert_eq!(root_floor(4, 4, true, 5, 64), 0, "a wider root than max_actions");
    assert_eq!(root_floor(4, 6, true, 5, 64), 4);
    assert_eq!(root_floor(4, 4, false, 2, 64), 0, "a fast or minor search");
    assert_eq!(root_floor(0, 4, true, 2, 64), 0, "off");
    assert_eq!(root_floor(4, 4, true, 1, 64), 0);
    assert_eq!(root_floor(u32::MAX, u32::MAX, true, 2, u32::MAX), 0, "no overflow");
    // a seat with the floor on and no root it could apply to is refused
    let bad = cfg(vec![with_floor(nn(0, 16, true), 4, 1)], vec![vec![0, 0]], &[("m", KeyKind::Nn)]);
    assert!(Driver::<ToyGame>::new(bad).err().unwrap().contains("root_min_visits_max_actions"));
}

#[test]
fn root_min_visits_off_or_unpayable_change_nothing() {
    // the golden records and the golden match (zero-core edf4b89's, computed before the setting
    // existed) with the floor at 0 and with one no search can pay (2 x options x k > sims: off)
    for k in [0, 1000] {
        let (_, recs) = golden_selfplay(with_floor(nn(0, 16, true), k, 4));
        assert_eq!((recs.len(), records_digest(&recs)), GOLDEN_RECORDS, "k {k}");
        let seats = vec![with_floor(nn(0, 16, false), k, 4), with_floor(nn(1, 16, false), k, 4)];
        let games = (0..40).map(|g| if g % 2 == 0 { vec![0, 1] } else { vec![1, 0] }).collect();
        let mut c = cfg(seats, games, &[("m", KeyKind::Nn), ("o", KeyKind::Nn)]);
        c.seed = 2026;
        c.record = false;
        assert_eq!(results_digest(&run(&mut Driver::<ToyGame>::new(c).unwrap()).0), GOLDEN_MATCH_VISITS, "k {k}");
    }
    // a payable floor changes the toy's full searches (roots of <= 4 options at 16 simulations)
    let (_, recs) = golden_selfplay(with_floor(nn(0, 16, true), 2, 4));
    assert_ne!(records_digest(&recs), GOLDEN_RECORDS.1);
}

/// Answer a keep-or-mulligan batch with the stand-in network (`mull_eval`: KEEP 0.999).
fn answer_mull(d: &mut Driver<Mull>, b: &Batch) {
    let f = Mull::NUM_FEATURES;
    let mut pri = Vec::new();
    let mut v = vec![0.0; b.b * MAX_SEATS];
    for r in 0..b.b {
        let k = (b.opt_off[r + 1] - b.opt_off[r]) as usize;
        let (p, val) = mull_eval(&b.x[r * f..(r + 1) * f], k);
        pri.extend(p);
        v[r * MAX_SEATS..(r + 1) * MAX_SEATS].copy_from_slice(&val);
    }
    d.submit(b.id, &pri, &v, &vec![0.0; b.b * MAX_SEATS]).unwrap();
}

/// A keep-or-mulligan network seat: 64 simulations, every search full; self-play without root
/// noise (eps 0: what the floor does is then the only way MULL is ever visited) and sampled by
/// visit count at both decisions.
fn mull_seat(k: u32, selfplay: bool, target: PolicyTarget, select: Select) -> SeatCfg {
    let mut s = NnSeat::eval(0, 64);
    s.selfplay = selfplay;
    s.eps = 0.0;
    s.temp_moves = 10;
    s.policy_target = target;
    s.select = select;
    s.root_min_visits = k;
    SeatCfg::Nn(s)
}

/// `games` keep-or-mulligan games of `seat` on both sides (move log on): result rows, records,
/// and per game the dealt hands.
fn mull_run(seat: SeatCfg, games: usize, record: bool) -> (Vec<GameResult>, Vec<Record>, Vec<[u8; 2]>) {
    let mut c = cfg(vec![seat], vec![vec![0, 0]; games], &[("m", KeyKind::Nn)]);
    c.seed = 31;
    c.record = record;
    c.move_log = 4;
    let seed = c.seed;
    let mut d = Driver::<Mull>::new(c).unwrap();
    let (mut res, mut smp) = (Vec::new(), Vec::new());
    while let Some(b) = d.next_batch(true).unwrap() {
        answer_mull(&mut d, &b);
        res.extend(d.take_results());
        smp.extend(d.take_samples());
    }
    res.extend(d.take_results());
    smp.extend(d.take_samples());
    res.sort_by_key(|r| r.gid);
    // the deal of game gid (the driver's stream: worker::init_slot)
    let hands = (0..games as u64)
        .map(|gid| {
            let mut base = Rng::new(split64(seed ^ split64(gid + 1)));
            Mull::new_game(&(), 2, base.next_u64()).unwrap().hand
        })
        .collect();
    (res, smp, hands)
}

/// MULL moves per (seat's hand bad, good) and decisions per (bad, good).
fn mull_moves(res: &[GameResult], hands: &[[u8; 2]]) -> ([u32; 2], [u32; 2]) {
    let (mut mull, mut all) = ([0u32; 2], [0u32; 2]);
    for r in res {
        assert_eq!(r.moves.len(), 2);
        for &(seat, a, _) in &r.moves {
            let h = hands[r.gid as usize][seat as usize] as usize;
            all[h] += 1;
            mull[h] += (a == MULL) as u32;
        }
    }
    (mull, all)
}

#[test]
fn root_min_visits_selfplay_targets_and_moves() {
    // The Delver mulligan in miniature: KEEP's prior 0.999 at every hand, 64 simulations.
    let cq = PolicyTarget::CompletedQ { c_visit: 50.0, c_scale: 1.0, prior_w: 1.0 };
    let (r0, s0, hands) = mull_run(mull_seat(0, true, cq, Select::Visits), 160, true);
    let (r4, s4, hands4) = mull_run(mull_seat(4, true, cq, Select::Visits), 160, true);
    assert_eq!(hands, hands4);
    assert_eq!((s0.len(), s4.len()), (320, 320));
    let bad = |r: &Record| r.x[0] == 0.0 && r.x[3] == 0.0;
    let i_mull = MULL as usize;
    // without the floor MULL is never visited: its completed-Q target stays at its prior
    assert!(s0.iter().all(|r| r.pi[i_mull] < 0.01), "k 0: a MULL target above its prior");
    // with it the target follows Q: MULL with a bad hand, KEEP with a good one
    let (nb, ng) = (s4.iter().filter(|r| bad(r)).count(), s4.iter().filter(|r| !bad(r)).count());
    assert!(nb > 40 && ng > 40, "{nb} bad hands, {ng} good ones");
    // (a bad hand's MULL Q rests on 4 draws at least: 4 unlucky ones (a bad hand again, 1 in 4 each)
    // leave it no better than KEEP, so not every bad-hand target is high)
    let mut high = 0;
    for r in &s4 {
        if bad(r) {
            high += (r.pi[i_mull] > 0.9) as usize;
        } else {
            assert!(r.pi[i_mull] < 1e-3, "a good hand's cq target: {:?}", r.pi);
            // a good hand's root value q: KEEP's walks alone (deterministic here: the value of
            // keeping), the forced MULL walks (worth ~0.14 less) left out
            let want = 0.5 + 0.3 * (r.x[0] - r.x[1]);
            assert!((r.q[0] - want).abs() < 1e-5, "a good hand's q {} (KEEP's value {want})", r.q[0]);
        }
    }
    assert!(high * 10 >= nb * 8, "{high} of {nb} bad-hand targets on MULL");
    eprintln!("root_min_visits selfplay: cq targets > 0.9 on MULL for {high} of {nb} bad hands, < 1e-3 for all {ng} good ones");
    // The moves are sampled by visit count without the forced walks: never MULL with a good hand
    // (PUCT never chose it) - with or without the floor; with a bad hand MULL is sampled once PUCT
    // follows the Q the forced walks found (without the floor: never).
    let (m0, a0) = mull_moves(&r0, &hands);
    let (m4, a4) = mull_moves(&r4, &hands);
    assert_eq!(a0, a4);
    assert_eq!((m0, m4[1]), ([0, 0], 0), "MULL played: k 0 {m0:?}, k 4 good hands {}", m4[1]);
    assert!(m4[0] > 0, "the floor's Q never moved PUCT to MULL with a bad hand");
    eprintln!("root_min_visits selfplay: MULL sampled with {} of {} bad hands, {} of {} good ones (k 0: {m0:?})", m4[0], a4[0], m4[1], a4[1]);
    // 'visits' targets: the forced walks are not in the visit distribution - a good hand's MULL
    // target is exactly 0 (the raw counts would give it 4 / 63)
    let (_, sv, _) = mull_run(mull_seat(4, true, PolicyTarget::Visits, Select::Visits), 160, true);
    for r in &sv {
        assert!(r.pi.iter().all(|x| x.is_finite()) && (r.pi.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        if !bad(r) {
            assert_eq!(r.pi[i_mull], 0.0, "a good hand's visits target holds forced visits: {:?}", r.pi);
        }
    }
}

#[test]
fn root_min_visits_and_the_move_rules() {
    // Evaluation seats (no temperature): the 'visits' rule never plays a forced move - it needs
    // the most visits without the forced walks; the guarded 'cq' rule plays what Q says once the
    // floor gives every option its k visits, and never an unvisited one without it.
    let cq4 = Select::Cq { c_visit: 50.0, c_scale: 1.0, min_visits: 4 };
    let t = PolicyTarget::Visits;
    let games = 200;
    let (rv0, _, hands) = mull_run(mull_seat(0, false, t, Select::Visits), games, false);
    let (rv4, _, _) = mull_run(mull_seat(4, false, t, Select::Visits), games, false);
    let (rc0, _, _) = mull_run(mull_seat(0, false, t, cq4), games, false);
    let (rc4, _, _) = mull_run(mull_seat(4, false, t, cq4), games, false);
    let (mv0, all) = mull_moves(&rv0, &hands);
    let (mv4, _) = mull_moves(&rv4, &hands);
    let (mc0, _) = mull_moves(&rc0, &hands);
    let (mc4, _) = mull_moves(&rc4, &hands);
    assert!(all[0] > 100 && all[1] > 100, "{all:?}");
    assert_eq!((mv0, mc0), ([0, 0], [0, 0]), "without the floor MULL is never played");
    assert_eq!(mv4, [0, 0], "the 'visits' rule played a move the floor forced");
    assert_eq!(mc4[1], 0, "the 'cq' rule played MULL with a good hand");
    // (but where 4 unlucky draws hid MULL's worth: 1 bad hand in ~20)
    assert!(mc4[0] * 10 >= all[0] * 9, "the 'cq' rule took MULL with {} of {} bad hands", mc4[0], all[0]);
    eprintln!("root_min_visits rules: visits k0 {mv0:?} k4 {mv4:?}, cq-n4 k0 {mc0:?} k4 {mc4:?} MULL moves of [bad, good] {all:?}");
}

#[test]
fn root_min_visits_never_touch_fast_searches() {
    // a self-play seat whose every search is fast (p_full 0) plays the same games with and without
    // the floor (fast and minor searches never take it, like root noise)
    let fast = |k: u32| match mull_seat(k, true, PolicyTarget::Visits, Select::Visits) {
        SeatCfg::Nn(mut s) => {
            s.sims_fast = Some(64);
            s.p_full = 0.0;
            s.eps = 0.25;
            SeatCfg::Nn(s)
        }
        s => s,
    };
    let (ra, sa, _) = mull_run(fast(0), 80, true);
    let (rb, sb, _) = mull_run(fast(4), 80, true);
    assert!(sa.is_empty() && sb.is_empty());
    assert_eq!(results_digest(&ra), results_digest(&rb));
    assert_eq!(ra.iter().map(|r| r.moves.clone()).collect::<Vec<_>>(), rb.iter().map(|r| r.moves.clone()).collect::<Vec<_>>());
}

/// `DriverCfg::aux` (docs/search.md 1.2): the same games and records with the auxiliary targets
/// added to each record (the decider's `aux_target` at that decision: the toy's next round card,
/// one-hot or all zero in the last round), the golden records unchanged; off, every record's `aux`
/// is empty; a game without targets refuses `aux`.
#[test]
fn aux_records_the_decider_targets_and_changes_nothing_else() {
    let run_aux = |aux: bool| {
        let mut c = cfg(vec![nn(0, 16, true)], vec![vec![0, 0]; 24], &[("m", KeyKind::Nn)]);
        c.seed = 2026;
        c.aux = aux;
        let mut d = Driver::<ToyGame>::new(c).unwrap();
        run(&mut d)
    };
    let (r0, s0) = run_aux(false);
    let (r1, s1) = run_aux(true);
    assert_eq!(s0.len(), GOLDEN_RECORDS.0);
    assert_eq!(records_digest(&s0), GOLDEN_RECORDS.1, "aux off: the golden records");
    assert_eq!(records_digest(&s1), GOLDEN_RECORDS.1, "aux on: the same records besides aux");
    let key = |r: &GameResult| (r.gid, r.steps, r.scores.clone(), r.samples);
    assert_eq!(r0.iter().map(key).collect::<Vec<_>>(), r1.iter().map(key).collect::<Vec<_>>());
    assert!(s0.iter().all(|r| r.aux.is_empty()));
    let (mut hot, mut none) = (0, 0);
    for r in &s1 {
        assert_eq!(r.aux.len(), ToyGame::aux_width());
        let s: f32 = r.aux.iter().sum();
        assert!(r.aux.iter().all(|&v| v == 0.0 || v == 1.0));
        if s == 1.0 {
            hot += 1;
        } else {
            assert_eq!(s, 0.0);
            none += 1;
        }
    }
    assert!(hot > 200 && none > 0, "one-hot {hot}, last-round {none}");
    let a = sample_arrays(&s1, ToyGame::NUM_FEATURES);
    assert_eq!((a.aw, a.aux.len()), (ToyGame::aux_width(), s1.len() * ToyGame::aux_width()));
    assert_eq!(sample_arrays(&s0, ToyGame::NUM_FEATURES).aw, 0);
    // replay one recorded game move for move: each record's target is its own decision's
    let mut c = cfg(vec![nn(0, 16, true)], vec![vec![0, 0]; 1], &[("m", KeyKind::Nn)]);
    c.aux = true;
    c.move_log = 10_000;
    let seed = c.seed;
    let mut d = Driver::<ToyGame>::new(c).unwrap();
    let (res, smp) = run(&mut d);
    assert!(res[0].over && !smp.is_empty());
    let mut g = ToyGame::new_game(&ToyGame::parse_spec("{}").unwrap(), 2, Rng::new(split64(seed ^ split64(1))).next_u64()).unwrap();
    let mut at = std::collections::HashMap::new();
    let mut keys = Vec::new();
    for (p, a, _) in &res[0].moves {
        let mut x = vec![0.0f32; ToyGame::NUM_FEATURES];
        g.encode(*p as usize, &mut x);
        let mut t = vec![0.0f32; ToyGame::aux_width()];
        g.aux_target(*p as usize, &mut t);
        at.insert(x.iter().map(|v| v.to_bits()).collect::<Vec<u32>>(), t);
        g.step(*a, &mut keys).unwrap();
        // (the driver steps single-option decisions itself: not in the move log)
        let mut legal = Vec::new();
        while g.to_move().is_some() {
            legal.clear();
            g.legal_actions(&mut legal);
            if legal.len() != 1 {
                break;
            }
            g.step(legal[0], &mut keys).unwrap();
        }
    }
    for r in &smp {
        let t = at.get(&r.x.iter().map(|v| v.to_bits()).collect::<Vec<u32>>()).expect("a recorded decision of the game");
        assert_eq!(&r.aux, t, "a record's target is the decider's at that decision");
    }
    // a game without auxiliary targets refuses aux
    let mut c = cfg(vec![SeatCfg::Bot("random".into())], vec![vec![0, 0, 0, 0]; 1], &[]);
    c.aux = true;
    assert!(Driver::<crate::toy_team::ToyTeam>::new(c).is_err());
}

// ------------------------------------------------------------------ root-view leaf values

use crate::mull_game::{hmull_eval, HiddenMull};

fn with_view(seat: SeatCfg, view: LeafView, share: f64, target: ViewTarget) -> SeatCfg {
    match seat {
        SeatCfg::Nn(mut s) => {
            s.leaf_value_view = view;
            s.view_samples = share;
            s.view_target = target;
            SeatCfg::Nn(s)
        }
        s => s,
    }
}

#[test]
fn leaf_view_settings_and_view_pick() {
    assert_eq!(LeafView::parse("mover"), Ok(LeafView::Mover));
    assert_eq!(LeafView::parse("root"), Ok(LeafView::Root));
    assert!(LeafView::parse("decider").is_err());
    assert_eq!(ViewTarget::parse("next_q"), Ok(ViewTarget::NextQ));
    assert_eq!(ViewTarget::parse("z"), Ok(ViewTarget::Z));
    assert!(ViewTarget::parse("q").is_err());
    let n = NnSeat::eval(0, 8);
    assert_eq!((n.leaf_value_view, n.view_samples, n.view_target), (LeafView::Mover, 0.0, ViewTarget::NextQ));
    // the share: a hash of (seed, gid, decision), about the share of the decisions, none at 0
    assert!((0..1000).all(|d| view_pick(7, 3, d, 1.0) && !view_pick(7, 3, d, 0.0)));
    let picked = (0..20_000u32).filter(|&d| view_pick(7, d / 100, d % 100, 0.25)).count();
    assert!((4_600..5_400).contains(&picked), "{picked} of 20000 at 0.25");
    let a: Vec<bool> = (0..64).map(|d| view_pick(1, 2, d, 0.5)).collect();
    assert_eq!(a, (0..64).map(|d| view_pick(1, 2, d, 0.5)).collect::<Vec<_>>());
    assert_ne!(a, (0..64).map(|d| view_pick(2, 2, d, 0.5)).collect::<Vec<_>>());
    for bad in [1.5, -0.1, f64::NAN] {
        let c = cfg(vec![with_view(nn(0, 16, true), LeafView::Mover, bad, ViewTarget::NextQ)], vec![vec![0, 0]], &[("m", KeyKind::Nn)]);
        assert!(Driver::<ToyGame>::new(c).err().unwrap().contains("view_samples"), "{bad}");
    }
}

#[test]
fn root_view_off_and_value_samples_leave_the_searches_alone() {
    // the golden records and match (computed before the settings existed) with the settings given
    // at their defaults
    let (_, recs) = golden_selfplay(with_view(nn(0, 16, true), LeafView::Mover, 0.0, ViewTarget::Z));
    assert_eq!((recs.len(), records_digest(&recs)), GOLDEN_RECORDS);
    let seat = |k| with_view(nn(k, 16, false), LeafView::Mover, 0.0, ViewTarget::NextQ);
    let games = (0..40).map(|g| if g % 2 == 0 { vec![0, 1] } else { vec![1, 0] }).collect();
    let mut c = cfg(vec![seat(0), seat(1)], games, &[("m", KeyKind::Nn), ("o", KeyKind::Nn)]);
    c.seed = 2026;
    c.record = false;
    assert_eq!(results_digest(&run(&mut Driver::<ToyGame>::new(c).unwrap()).0), GOLDEN_MATCH_VISITS);
    // value-only samples (every recorded decision) add records and change nothing else: the same
    // games, the decisions' records the golden ones; one value-only record per decision record (two
    // players), right after it: the other seat, no options, no policy target
    let (res, recs) = golden_selfplay(with_view(nn(0, 16, true), LeafView::Mover, 1.0, ViewTarget::NextQ));
    let (res0, _) = golden_selfplay(nn(0, 16, true));
    assert_eq!(results_digest(&res), results_digest(&res0));
    let dec: Vec<Record> = recs.iter().filter(|r| !r.opts.is_empty()).cloned().collect();
    assert_eq!((dec.len(), records_digest(&dec)), GOLDEN_RECORDS);
    assert_eq!(recs.len(), 2 * dec.len());
    for k in (0..recs.len()).step_by(2) {
        let (d, v) = (&recs[k], &recs[k + 1]);
        assert!(!d.opts.is_empty() && v.opts.is_empty() && v.pi.is_empty());
        assert_eq!(v.player, 1 - d.player);
        assert!((v.z[0] + d.z[0] - 1.0).abs() < 1e-6, "z in the observer's perspective");
        assert_ne!(v.x, d.x, "the observer's view of the state");
    }
    assert_eq!(res.iter().map(|r| r.samples as usize).sum::<usize>(), recs.len());
}

/// Answer a hidden-hand mulligan batch with the stand-in network (`hmull_eval`), every row's value
/// relative to its own player; checks the layout of root-view batches: a value-only row follows
/// its leaf's row (same game, the other seat, no options). -> value-only rows.
fn answer_hmull(d: &mut Driver<HiddenMull>, b: &Batch) -> usize {
    let f = HiddenMull::NUM_FEATURES;
    let mut pri = Vec::new();
    let mut v = vec![0.0; b.b * MAX_SEATS];
    let mut vrows = 0;
    for r in 0..b.b {
        let k = (b.opt_off[r + 1] - b.opt_off[r]) as usize;
        // (no deferred terminals here: a row without options is a flagged value-only row)
        assert_eq!(b.extra[r], if k == 0 { ROW_VALUE } else { ROW_LEAF }, "row {r}'s kind");
        if k == 0 {
            vrows += 1;
            let leaf_before = r > 0 && b.gid[r] == b.gid[r - 1] && b.player[r] != b.player[r - 1] && b.opt_off[r] > b.opt_off[r - 1];
            assert!(leaf_before, "a value-only row without its leaf's row before it");
        }
        let (p, val) = hmull_eval(&b.x[r * f..(r + 1) * f], k);
        pri.extend(p);
        v[r * MAX_SEATS..(r + 1) * MAX_SEATS].copy_from_slice(&val);
    }
    d.submit(b.id, &pri, &v, &vec![0.0; b.b * MAX_SEATS]).unwrap();
    vrows
}

/// Hidden-hand mulligan self-play of `seat` on both sides: (records, value-only rows sent, stats).
fn hmull_run(seat: SeatCfg, games: usize, seed: u64) -> (Vec<Record>, usize, DriverStats) {
    let mut c = cfg(vec![seat], vec![vec![0, 0]; games], &[("m", KeyKind::Nn)]);
    c.seed = seed;
    let mut d = Driver::<HiddenMull>::new(c).unwrap();
    let (mut smp, mut vrows) = (Vec::new(), 0);
    while let Some(b) = d.next_batch(true).unwrap() {
        vrows += answer_hmull(&mut d, &b);
        smp.extend(d.take_samples());
    }
    smp.extend(d.take_samples());
    (smp, vrows, d.stats())
}

fn hmull_seat(sims: u32, view: LeafView, share: f64, target: ViewTarget) -> SeatCfg {
    let mut s = NnSeat::eval(0, sims);
    s.selfplay = true;
    s.eps = 0.0;
    s.temp_moves = 0;
    with_view(SeatCfg::Nn(s), view, share, target)
}

#[test]
fn root_view_values_read_the_root_deciders_hand() {
    // Two simulations: the first expands the root, the second takes KEEP (prior 0.6) to the other
    // seat's decision - a leaf, whose value is the root's value q. From the other seat's view (the
    // search of before) it reads that seat's hand in the sampled world and none of the keeper's:
    // 1 - (0.5 + 0.3 (q1 - 0.5)) with the sampled q1 0 or 1. From the root decider's view: 0.5 +
    // 0.3 (q0 - 0.5), its own hand, in every world.
    let first = |recs: &[Record]| -> Vec<(f32, f32)> { recs.iter().filter(|r| r.player == 0 && r.x[1] == 0.0).map(|r| (r.x[0], r.q[0])).collect() };
    let (rm, vm, sm) = hmull_run(hmull_seat(2, LeafView::Mover, 0.0, ViewTarget::NextQ), 200, 11);
    let (rr, vr, sr) = hmull_run(hmull_seat(2, LeafView::Root, 0.0, ViewTarget::NextQ), 200, 11);
    assert_eq!((vm, sm.view_rows), (0, 0));
    assert!(vr > 150 && vr as u64 == sr.view_rows, "{vr} value-only rows, stats {}", sr.view_rows);
    assert_eq!(sr.leaves, sm.leaves, "one evaluation per leaf either way (the value-only rows are extra rows)");
    let (fm, fr) = (first(&rm), first(&rr));
    assert_eq!((fm.len(), fr.len()), (200, 200));
    let want = |q0: f32| 0.5 + 0.3 * (q0 - 0.5);
    assert!(fr.iter().all(|&(q0, q)| (q - want(q0)).abs() < 1e-6), "root view: the keeper's own hand");
    let off = fm.iter().filter(|&&(q0, q)| (q - want(q0)).abs() > 0.1).count();
    assert!((60..140).contains(&off), "mover view: {off} of 200 roots valued by the other seat's sampled hand");
    let bad = |f: &[(f32, f32)]| {
        let v: Vec<f32> = f.iter().filter(|x| x.0 == 0.0).map(|x| x.1).collect();
        v.iter().sum::<f32>() / v.len().max(1) as f32
    };
    eprintln!("hidden mull, a bad hand's KEEP after 2 simulations: mover view {:.3}, root view {:.3} (its value 0.35)", bad(&fm), bad(&fr));
    assert!(bad(&fm) > 0.45 && (bad(&fr) - 0.35).abs() < 1e-6);
}

#[test]
fn value_only_samples_and_their_targets() {
    // every recorded decision (a hidden-hand game has two, both searched) records the other seat's
    // view of the real state: [seat 0's decision, seat 1's view, seat 1's decision, seat 0's view]
    // per game. Its z is the outcome for the observer; 'next_q': seat 1's view takes seat 1's next
    // search's root value (its own decision's q: the same search); seat 0 never searches again
    // after seat 1's decision, so its view takes z; 'z': q = z.
    for target in [ViewTarget::NextQ, ViewTarget::Z] {
        let (recs, _, st) = hmull_run(hmull_seat(16, LeafView::Root, 1.0, target), 60, 5);
        assert_eq!(recs.len(), 240);
        assert_eq!((st.recorded, st.view_recorded), (120, 120));
        let mut differ = 0;
        for g in recs.chunks(4) {
            let (d0, v1, d1, v0) = (&g[0], &g[1], &g[2], &g[3]);
            assert_eq!((d0.player, v1.player, d1.player, v0.player), (0, 1, 1, 0));
            assert!(v1.opts.is_empty() && v0.opts.is_empty() && v1.pi.is_empty() && !d0.opts.is_empty());
            // the views: seat 1 before seat 0's move (nobody decided), seat 0 before seat 1's
            assert_eq!((v1.x[1], v0.x[1], d1.x[1]), (0.0, 1.0, 1.0));
            assert_eq!(v1.z, d1.z);
            assert_eq!(v0.z, d0.z);
            assert_eq!(v0.q, v0.z);
            if target == ViewTarget::NextQ {
                assert_eq!(v1.q, d1.q, "seat 1's view: its next search's root value");
                differ += (v1.q != v1.z) as usize;
            } else {
                assert_eq!(v1.q, v1.z);
            }
        }
        assert!(target == ViewTarget::Z || differ > 30, "{differ}");
    }
    // a share: about that many, the decisions' records unchanged
    let (all, _, _) = hmull_run(hmull_seat(16, LeafView::Root, 0.0, ViewTarget::NextQ), 200, 9);
    let (half, _, st) = hmull_run(hmull_seat(16, LeafView::Root, 0.5, ViewTarget::NextQ), 200, 9);
    assert!((150..250).contains(&st.view_recorded), "{}", st.view_recorded);
    let dec: Vec<Record> = half.iter().filter(|r| !r.opts.is_empty()).cloned().collect();
    assert_eq!(records_digest(&dec), records_digest(&all));
}

#[test]
fn value_only_samples_carry_the_observers_auxiliary_targets() {
    // with auxiliary targets on (`DriverCfg::aux`) a value-only record gets its observer's targets
    // (`aux_target(observer)` of the real state: a label only), so every record has the same width
    let mut c = cfg(vec![with_view(nn(0, 16, true), LeafView::Root, 1.0, ViewTarget::NextQ)], vec![vec![0, 0]; 24], &[("m", KeyKind::Nn)]);
    c.seed = 2026;
    c.aux = true;
    let (_, recs) = run(&mut Driver::<ToyGame>::new(c).unwrap());
    let w = ToyGame::aux_width();
    assert!(recs.iter().any(|r| r.opts.is_empty()) && recs.iter().all(|r| r.aux.len() == w));
    let a = sample_arrays(&recs, ToyGame::NUM_FEATURES);
    assert_eq!((a.aw, a.aux.len()), (w, recs.len() * w));
}

// ---------------------------------------------------------------- the root's belief (docs/search.md 3.8)

fn with_belief(seat: SeatCfg, tilt: f64, clip: f64, mix: f64, fast: bool) -> SeatCfg {
    match seat {
        SeatCfg::Nn(mut s) => {
            s.belief_tilt = tilt;
            s.belief_clip = clip;
            s.belief_mix = mix;
            s.belief_fast = fast;
            SeatCfg::Nn(s)
        }
        s => s,
    }
}

/// The root belief rows of a batch, their layout checked: no options, and either right after its
/// game's root leaf row (a fresh root: the leaf of the same seat) or its game's only row (a reused
/// root). -> (rows after a leaf, rows alone).
fn belief_rows_of(b: &Batch) -> (usize, usize) {
    let (mut with, mut alone) = (0, 0);
    for r in 0..b.b {
        if b.extra[r] != ROW_BELIEF {
            continue;
        }
        assert_eq!(b.opt_off[r + 1], b.opt_off[r], "a belief row has no options");
        let lead = r > 0 && b.extra[r - 1] == ROW_LEAF && b.gid[r - 1] == b.gid[r] && b.player[r - 1] == b.player[r] && b.opt_off[r] > b.opt_off[r - 1];
        let only = (0..b.b).filter(|&k| b.gid[k] == b.gid[r]).count() == 1;
        assert!(lead || only, "a belief row neither after its root's leaf row nor alone (row {r}: {:?})", &b.extra);
        if lead {
            with += 1;
        } else {
            alone += 1;
        }
    }
    (with, alone)
}

/// `answer`, its root belief rows answered `pred` (a stand-in auxiliary head) -> (rows after a leaf,
/// rows alone).
fn answer_belief<G: SearchGame>(d: &mut Driver<G>, b: &Batch, pred: &[f32]) -> (usize, usize) {
    let (w, a) = belief_rows_of(b);
    let f = G::NUM_FEATURES;
    let mut pri = Vec::new();
    let mut v = vec![0.0; b.b * MAX_SEATS];
    for r in 0..b.b {
        let k = (b.opt_off[r + 1] - b.opt_off[r]) as usize;
        pri.extend(std::iter::repeat_n(1.0 / k as f32, k));
        let n = b.n_players[r] as usize;
        let h: f32 = b.x[r * f..(r + 1) * f].iter().enumerate().map(|(i, x)| x * ((i * 7 % 13) as f32 - 6.0)).sum();
        let p = 1.0 / (1.0 + (-h * 0.1).exp());
        v[r * MAX_SEATS] = p;
        for j in 1..n {
            v[r * MAX_SEATS + j] = (1.0 - p) / (n - 1) as f32;
        }
    }
    let aux: Vec<f32> = (0..w + a).flat_map(|_| pred.iter().copied()).collect();
    d.submit_aux(b.id, &pri, &v, &vec![0.0; b.b * MAX_SEATS], &aux).unwrap();
    (w, a)
}

fn run_belief<G: SearchGame>(d: &mut Driver<G>, pred: &[f32]) -> (Vec<GameResult>, Vec<Record>, (usize, usize)) {
    let (mut res, mut smp, mut rows) = (Vec::new(), Vec::new(), (0, 0));
    while let Some(b) = d.next_batch(true).unwrap() {
        let (w, a) = answer_belief(d, &b, pred);
        rows = (rows.0 + w, rows.1 + a);
        res.extend(d.take_results());
        smp.extend(d.take_samples());
    }
    res.extend(d.take_results());
    smp.extend(d.take_samples());
    res.sort_by_key(|r| r.gid);
    (res, smp, rows)
}

#[test]
fn belief_off_is_the_search_of_before() {
    // the golden records and match with the belief's settings given (tilt 0: off) and the others
    // set to anything valid
    let (_, recs) = golden_selfplay(with_belief(nn(0, 16, true), 0.0, 7.0, 0.3, true));
    assert_eq!((recs.len(), records_digest(&recs)), GOLDEN_RECORDS);
    let seat = |k| with_belief(nn(k, 16, false), 0.0, 3.0, 0.5, false);
    let games = (0..40).map(|g| if g % 2 == 0 { vec![0, 1] } else { vec![1, 0] }).collect();
    let mut c = cfg(vec![seat(0), seat(1)], games, &[("m", KeyKind::Nn), ("o", KeyKind::Nn)]);
    c.seed = 2026;
    c.record = false;
    let mut d = Driver::<ToyGame>::new(c).unwrap();
    assert_eq!(results_digest(&run(&mut d).0), GOLDEN_MATCH_VISITS);
    assert_eq!(d.stats().belief_rows, 0);
}

#[test]
fn belief_settings_are_checked() {
    for (tilt, clip, mix, what) in [(-1.0, 20.0, 0.0, "belief_tilt"), (f64::NAN, 20.0, 0.0, "belief_tilt"), (1.0, 0.5, 0.0, "belief_clip"), (1.0, 20.0, 1.0, "belief_mix"), (1.0, 20.0, -0.1, "belief_mix")] {
        let c = cfg(vec![with_belief(nn(0, 16, true), tilt, clip, mix, false)], vec![vec![0, 0]], &[("m", KeyKind::Nn)]);
        let e = Driver::<ToyGame>::new(c).err().unwrap();
        assert!(e.contains(what), "{tilt} {clip} {mix}: {e}");
    }
    // a game without auxiliary targets has nothing for a belief to predict
    let c = cfg(vec![with_belief(nn(0, 16, true), 1.0, 20.0, 0.0, false)], vec![vec![0, 0]], &[("m", KeyKind::Nn)]);
    assert!(Driver::<Mull>::new(c).err().unwrap().contains("no auxiliary targets"));
}

#[test]
fn one_belief_row_per_search_that_takes_it() {
    // self-play (p_full 0.5): the full searches take the belief - as many rows as recorded decisions -,
    // the fast ones not; with belief_fast every searched decision. Fresh roots send the row after the
    // root's leaf, reused roots alone
    let uni = vec![1.0 / 6.0; 6];
    let mut c = cfg(vec![with_belief(nn(0, 16, true), 1.0, 20.0, 0.0, false)], vec![vec![0, 0]; 24], &[("m", KeyKind::Nn)]);
    c.seed = 2026;
    let mut d = Driver::<ToyGame>::new(c.clone()).unwrap();
    let (res, recs, (with, alone)) = run_belief(&mut d, &uni);
    let st = d.stats();
    assert_eq!(st.belief_rows as usize, with + alone);
    assert_eq!(st.belief_rows, st.recorded, "one belief row per full search");
    assert_eq!(recs.len() as u64, st.recorded);
    assert!(with > 0 && alone > 0, "fresh roots {with}, reused roots {alone}");
    let decisions: u64 = res.iter().map(|r| r.decisions as u64).sum();
    assert!(st.belief_rows < decisions);
    c.seats = vec![with_belief(nn(0, 16, true), 1.0, 20.0, 0.0, true)];
    let mut d = Driver::<ToyGame>::new(c.clone()).unwrap();
    let (res, _, _) = run_belief(&mut d, &uni);
    let decisions: u64 = res.iter().map(|r| r.decisions as u64).sum();
    assert_eq!(d.stats().belief_rows, decisions, "belief_fast: every searched decision");
    // an evaluation seat: every search is full
    c.seats = vec![with_belief(nn(0, 16, false), 1.0, 20.0, 0.0, false)];
    c.record = false;
    let mut d = Driver::<ToyGame>::new(c).unwrap();
    let (res, _, _) = run_belief(&mut d, &uni);
    let decisions: u64 = res.iter().map(|r| r.decisions as u64).sum();
    assert_eq!(d.stats().belief_rows, decisions);
}

#[test]
fn an_evaluator_without_the_belief_stops_the_driver() {
    // a batch with root belief rows answered without their auxiliary outputs (a network without the
    // head) is an error, never a search dealing uniformly in silence
    let c = cfg(vec![with_belief(nn(0, 16, true), 1.0, 20.0, 0.0, false)], vec![vec![0, 0]; 4], &[("m", KeyKind::Nn)]);
    let mut d = Driver::<ToyGame>::new(c).unwrap();
    let mut err = None;
    while let Some(b) = d.next_batch(true).unwrap() {
        if b.extra.contains(&ROW_BELIEF) {
            let k = b.opt_off[b.b] as usize;
            let nb = b.extra.iter().filter(|&&x| x == ROW_BELIEF).count();
            let e = d.submit(b.id, &vec![0.5; k], &vec![0.5; b.b * MAX_SEATS], &vec![0.0; b.b * MAX_SEATS]).unwrap_err();
            // the wrong width too, and a non-finite prediction
            let e2 = d.submit_aux(b.id, &vec![0.5; k], &vec![0.5; b.b * MAX_SEATS], &vec![0.0; b.b * MAX_SEATS], &[0.5; 5]).unwrap_err();
            let e3 = d.submit_aux(b.id, &vec![0.5; k], &vec![0.5; b.b * MAX_SEATS], &vec![0.0; b.b * MAX_SEATS], &vec![f32::NAN; 6 * nb]).unwrap_err();
            err = Some((e, e2, e3));
            break;
        }
        answer(&mut d, &b);
    }
    let (e, e2, e3) = err.expect("a batch with a belief row");
    assert!(e.contains("root belief rows") && e.contains("auxiliary"), "{e}");
    assert!(e2.contains("root belief rows"), "{e2}");
    assert!(e3.contains("finite"), "{e3}");
    // and a batch without belief rows takes no auxiliary values
    let c = cfg(vec![nn(0, 16, true)], vec![vec![0, 0]; 2], &[("m", KeyKind::Nn)]);
    let mut d = Driver::<ToyGame>::new(c).unwrap();
    let b = d.next_batch(true).unwrap().unwrap();
    let k = b.opt_off[b.b] as usize;
    assert!(d.submit_aux(b.id, &vec![0.5; k], &vec![0.5; b.b * MAX_SEATS], &vec![0.0; b.b * MAX_SEATS], &[0.1; 6]).unwrap_err().contains("no root belief rows"));
}

/// Hidden-hand mulligan games of `seat` on both sides, the belief rows answered `pred`, every row's
/// value from `hmull_eval` -> (records, stats, value-only rows, belief rows).
fn hmull_belief_run(seat: SeatCfg, games: usize, seed: u64, pred: &[f32]) -> (Vec<Record>, DriverStats, usize, usize) {
    let mut c = cfg(vec![seat], vec![vec![0, 0]; games], &[("m", KeyKind::Nn)]);
    c.seed = seed;
    let mut d = Driver::<HiddenMull>::new(c).unwrap();
    let f = HiddenMull::NUM_FEATURES;
    let (mut smp, mut vrows, mut brows) = (Vec::new(), 0, 0);
    while let Some(b) = d.next_batch(true).unwrap() {
        let (w, a) = belief_rows_of(&b);
        brows += w + a;
        let mut pri = Vec::new();
        let mut v = vec![0.0; b.b * MAX_SEATS];
        for r in 0..b.b {
            let k = (b.opt_off[r + 1] - b.opt_off[r]) as usize;
            vrows += (b.extra[r] == ROW_VALUE) as usize;
            let (p, val) = hmull_eval(&b.x[r * f..(r + 1) * f], k);
            pri.extend(p);
            v[r * MAX_SEATS..(r + 1) * MAX_SEATS].copy_from_slice(&val);
        }
        let aux: Vec<f32> = (0..w + a).flat_map(|_| pred.iter().copied()).collect();
        d.submit_aux(b.id, &pri, &v, &vec![0.0; b.b * MAX_SEATS], &aux).unwrap();
        smp.extend(d.take_samples());
    }
    smp.extend(d.take_samples());
    (smp, d.stats(), vrows, brows)
}

fn hmull_belief_seat(sims: u32, tilt: f64, view: LeafView, redet: bool) -> SeatCfg {
    let mut s = NnSeat::eval(0, sims);
    s.selfplay = true;
    s.eps = 0.0;
    s.temp_moves = 0;
    s.redet = redet;
    with_belief(with_view(SeatCfg::Nn(s), view, 0.0, ViewTarget::NextQ), tilt, 1000.0, 0.0, false)
}

/// The mean root value q of seat 1's decisions after seat 0 kept (its records), for its own hand.
fn seat1_q(recs: &[Record]) -> f64 {
    let v: Vec<f64> = recs.iter().filter(|r| r.player == 1 && r.x[3] == 0.0).map(|r| r.q[0] as f64).collect();
    v.iter().sum::<f64>() / v.len() as f64
}

#[test]
fn the_belief_deals_the_searches_worlds() {
    // HiddenMull: seat 1 decides after seat 0; every walk of its search ends the game, valued exactly
    // by the hands of its world: 0.5 - 0.3 (q0 - q1) for seat 1, q0 seat 0's hand dealt by the search
    // (good with 1/2 uniformly). A belief "seat 0's hand is good with 0.9" (tilt 1: the deal's odds
    // exactly 0.9) lowers seat 1's root value by 0.3 x (0.9 - 0.5) x walks / (walks + 1); belief 0.5 is
    // the uniform deal
    let sims = 64;
    let (uni, st0, _, b0) = hmull_belief_run(hmull_belief_seat(sims, 0.0, LeafView::Mover, true), 300, 3, &[]);
    let (half, _, _, _) = hmull_belief_run(hmull_belief_seat(sims, 1.0, LeafView::Mover, true), 300, 3, &[0.5, 0.5]);
    let (good, st, _, b1) = hmull_belief_run(hmull_belief_seat(sims, 1.0, LeafView::Mover, true), 300, 3, &[0.1, 0.9]);
    assert_eq!((b0, st0.belief_rows), (0, 0));
    assert_eq!((b1 as u64, st.belief_rows), (st.recorded, st.recorded));
    let (qu, qh, qg) = (seat1_q(&uni), seat1_q(&half), seat1_q(&good));
    let want = -0.3 * 0.4 * (sims - 1) as f64 / sims as f64;
    eprintln!("hidden mull, seat 1's root value after seat 0 kept: uniform {qu:.4}, belief 0.5 {qh:.4}, belief 0.9 {qg:.4} (shift {:.4}, want {want:.4})", qg - qu);
    assert!((qh - qu).abs() < 0.01, "a belief equal to the uniform deal: {qh} vs {qu}");
    assert!((qg - qu - want).abs() < 0.015, "the belief's shift {} vs {want}", qg - qu);
    // without redet (one world per decision, dealt again with the belief once it came): the same mean
    let (one, _, _, _) = hmull_belief_run(hmull_belief_seat(sims, 1.0, LeafView::Mover, false), 300, 3, &[0.1, 0.9]);
    let (one_u, _, _, _) = hmull_belief_run(hmull_belief_seat(sims, 0.0, LeafView::Mover, false), 300, 3, &[]);
    assert!((seat1_q(&one) - seat1_q(&one_u) - want).abs() < 0.06, "redet off: {} vs {}", seat1_q(&one), seat1_q(&one_u));
}

#[test]
fn per_target_tilts_take_the_belief_and_deal_by_it() {
    // docs/search.md 3.8.1: a seat with belief_tilt 0 and a per-target tilt asks the root's belief (one
    // row per full search) and the game reads the target's tilt: HiddenMull's `hand` at 1 deals as
    // belief_tilt 1 does, the same shift of seat 1's root value
    let sims = 64;
    let mut s = hmull_belief_seat(sims, 0.0, LeafView::Mover, true);
    if let SeatCfg::Nn(n) = &mut s {
        n.belief_tilts = vec![("hand".to_string(), 1.0)];
    }
    let (good, st, _, b1) = hmull_belief_run(s, 300, 3, &[0.1, 0.9]);
    assert_eq!((b1 as u64, st.belief_rows), (st.recorded, st.recorded));
    let (uni, _, _, _) = hmull_belief_run(hmull_belief_seat(sims, 0.0, LeafView::Mover, true), 300, 3, &[]);
    let want = -0.3 * 0.4 * (sims - 1) as f64 / sims as f64;
    let (qu, qg) = (seat1_q(&uni), seat1_q(&good));
    assert!((qg - qu - want).abs() < 0.015, "the per-target tilt's shift {} vs {want}", qg - qu);
    // the mapping to AUX order, and a seat naming none: empty (every Belief of before)
    let mut n = NnSeat::eval(0, 8);
    assert!(n.target_tilts(ToyGame::AUX).is_empty() && !n.belief_on());
    n.belief_tilts = vec![("next_card".to_string(), 0.5)];
    assert_eq!(n.target_tilts(ToyGame::AUX), vec![0.5]);
    assert!(n.belief_on() && n.takes_belief(true) && !n.takes_belief(false));
    n.belief_tilts = vec![("next_card".to_string(), 0.0)];
    assert!(!n.belief_on(), "a per-target tilt of 0 is off");
}

#[test]
fn per_target_tilts_are_checked() {
    // an unknown target, a target twice, a tilt < 0 or not finite: refused by the driver; and a
    // lookahead seat (its belief is belief_tilt's)
    for (tilts, what) in [
        (vec![("hand".to_string(), 1.0)], "not an auxiliary target"),
        (vec![("next_card".to_string(), 1.0), ("next_card".to_string(), 0.5)], "twice"),
        (vec![("next_card".to_string(), -1.0)], "finite number >= 0"),
        (vec![("next_card".to_string(), f64::NAN)], "finite number >= 0"),
    ] {
        let mut s = with_belief(nn(0, 16, true), 0.0, 20.0, 0.0, false);
        if let SeatCfg::Nn(n) = &mut s {
            n.belief_tilts = tilts.clone();
        }
        let c = cfg(vec![s], vec![vec![0, 0]], &[("m", KeyKind::Nn)]);
        let e = Driver::<ToyGame>::new(c).err().unwrap();
        assert!(e.contains(what), "{tilts:?}: {e}");
    }
}

#[test]
fn root_view_values_and_the_belief_share_the_batches() {
    // both extra row kinds in one driver: value-only rows after other seats' leaves, belief rows
    // after fresh roots' leaves; the counts match the stats and the deal still follows the belief
    let (recs, st, vrows, brows) = hmull_belief_run(hmull_belief_seat(32, 1.0, LeafView::Root, true), 200, 5, &[0.1, 0.9]);
    assert!(vrows > 0 && vrows as u64 == st.view_rows, "{vrows} value-only rows, stats {}", st.view_rows);
    assert_eq!((brows as u64, st.belief_rows), (st.recorded, st.recorded));
    let (uni, _, _, _) = hmull_belief_run(hmull_belief_seat(32, 0.0, LeafView::Root, true), 200, 5, &[]);
    assert!(seat1_q(&recs) < seat1_q(&uni) - 0.08, "{} vs {}", seat1_q(&recs), seat1_q(&uni));
}

#[test]
fn rolling_phases_ask_the_belief_again_of_a_new_network() {
    // a new network restarts every search (rolling epochs): a belief asked of the old one is asked
    // again; every batch's belief rows answered, every full search took one
    let mut c = cfg(vec![with_belief(nn(0, 8, true), 1.0, 20.0, 0.25, false)], vec![vec![0, 0]], &[("m", KeyKind::Nn)]);
    c.rolling = true;
    c.concurrency = 12;
    let mut d = Driver::<ToyGame>::new(c).unwrap();
    let uni = vec![1.0 / 6.0; 6];
    for phase in 0..4u32 {
        d.resume(10, phase / 2).unwrap();
        while let Some(b) = d.next_batch(true).unwrap() {
            answer_belief(&mut d, &b, &uni);
            d.take_results();
            d.take_samples();
        }
    }
    let st = d.stats();
    assert_eq!(st.epoch, 1);
    assert!(st.restarts > 0);
    assert!(st.belief_rows >= st.recorded && st.recorded > 0, "belief rows {}, recorded {}", st.belief_rows, st.recorded);
}

// ------------------------------------------------------------------ the Gumbel root search (docs/search.md 3.7)

/// `seat` with the Gumbel root search (m, noise scale, sigma's scale; c_visit 50); a self-play seat
/// records the 'cq' target with the same sigma.
fn with_gumbel(seat: SeatCfg, m: u32, scale: f64, c_scale: f64) -> SeatCfg {
    match seat {
        SeatCfg::Nn(mut s) => {
            s.root_search = RootSearch::Gumbel { m, scale, c_visit: 50.0, c_scale };
            if s.selfplay {
                s.policy_target = PolicyTarget::CompletedQ { c_visit: 50.0, c_scale, prior_w: 1.0 };
            }
            SeatCfg::Nn(s)
        }
        s => s,
    }
}

#[test]
fn gumbel_settings_are_checked() {
    assert_eq!(RootSearch::parse("puct", 0, f64::NAN, 0.0, 0.0), Ok(RootSearch::Puct));
    assert_eq!(RootSearch::parse("gumbel", 16, 1.0, 50.0, 0.3), Ok(RootSearch::Gumbel { m: 16, scale: 1.0, c_visit: 50.0, c_scale: 0.3 }));
    for (m, scale, cv, cs) in [(0, 1.0, 50.0, 1.0), (16, -1.0, 50.0, 1.0), (16, f64::INFINITY, 50.0, 1.0), (16, 1.0, f64::NAN, 1.0), (16, 1.0, 50.0, -0.1)] {
        assert!(RootSearch::parse("gumbel", m, scale, cv, cs).is_err(), "{m} {scale} {cv} {cs}");
    }
    assert!(RootSearch::parse("mcts", 16, 1.0, 50.0, 1.0).is_err());
    let one = |seat: SeatCfg, record: bool| {
        let mut c = cfg(vec![seat], vec![vec![0, 0]], &[("m", KeyKind::Nn)]);
        c.record = record;
        Driver::<ToyGame>::new(c).err()
    };
    // exclusive with the root minimum visits
    let e = one(with_floor(with_gumbel(nn(0, 16, true), 16, 1.0, 1.0), 4, 4), true).unwrap();
    assert!(e.contains("exclusive"), "{e}");
    // a recording seat records the 'cq' target (sequential halving's counts are no policy) ...
    let visits = match with_gumbel(nn(0, 16, true), 16, 1.0, 1.0) {
        SeatCfg::Nn(mut s) => {
            s.policy_target = PolicyTarget::Visits;
            SeatCfg::Nn(s)
        }
        s => s,
    };
    assert!(one(visits.clone(), true).unwrap().contains("policy_target 'cq'"));
    // ... but a driver that records nothing, or an evaluation seat, may have any target
    assert!(one(visits, false).is_none());
    assert!(one(with_gumbel(nn(0, 16, false), 16, 0.0, 1.0), true).is_none());
}

#[test]
fn gumbel_selfplay_is_reproducible_and_its_records_valid() {
    // toy self-play with fast searches (every search of the seat is a Gumbel one): the same records
    // whatever the threads and groups; valid targets and root values; other records than PUCT's
    let mk = |threads: usize, groups: usize, m: u32| {
        let mut c = cfg(vec![with_gumbel(nn(0, 16, true), m, 1.0, 1.0)], vec![vec![0, 0]; 24], &[("m", KeyKind::Nn)]);
        c.threads = threads;
        c.groups = groups;
        c.seed = 2026;
        run(&mut Driver::<ToyGame>::new(c).unwrap())
    };
    let (r1, s1) = mk(1, 1, 16);
    let (r4, s4) = mk(4, 3, 16);
    let key = |r: &GameResult| (r.gid, r.steps, r.scores.clone(), r.samples, r.decisions);
    assert_eq!(r1.iter().map(key).collect::<Vec<_>>(), r4.iter().map(key).collect::<Vec<_>>());
    assert_eq!(records_digest(&s1), records_digest(&s4));
    assert!(!s1.is_empty());
    assert_ne!(records_digest(&s1), GOLDEN_RECORDS.1, "the Gumbel search's records are its own");
    for s in &s1 {
        assert_eq!(s.pi.len(), s.opts.len());
        assert!((s.pi.iter().sum::<f32>() - 1.0).abs() < 1e-4 && s.pi.iter().all(|x| x.is_finite() && *x >= 0.0));
        assert!((s.q[0] + s.q[1] - 1.0).abs() < 1e-4 && s.q.iter().all(|x| x.is_finite()));
    }
    // m 2 considers fewer options: other games
    let (_, s2) = mk(2, 2, 2);
    assert_ne!(records_digest(&s1), records_digest(&s2));
}

#[test]
fn gumbel_noise_only_within_temp_moves() {
    // the Gumbel noise is the temperature: a self-play seat past its temp_moves searches without it -
    // a self-play seat with temp_moves 0 plays the games of an evaluation seat with noise scale 0 (no
    // Dirichlet noise either: the Gumbel seat has none), move for move; with temp_moves it does not
    let games = |seat: SeatCfg| {
        let mut c = cfg(vec![seat], vec![vec![0, 0]; 30], &[("m", KeyKind::Nn)]);
        c.seed = 99;
        c.record = false;
        c.move_log = 1000;
        let (res, _) = run(&mut Driver::<ToyGame>::new(c).unwrap());
        res.iter().map(|r| r.moves.clone()).collect::<Vec<_>>()
    };
    let sp = |temp: u32| match with_gumbel(SeatCfg::Nn(NnSeat::eval(0, 24)), 16, 1.0, 1.0) {
        SeatCfg::Nn(mut s) => {
            s.selfplay = true;
            s.temp_moves = temp;
            SeatCfg::Nn(s)
        }
        s => s,
    };
    let eval0 = games(with_gumbel(SeatCfg::Nn(NnSeat::eval(0, 24)), 16, 0.0, 1.0));
    assert_eq!(games(sp(0)), eval0, "past temp_moves: no noise");
    assert_ne!(games(sp(1000)), eval0, "within temp_moves: noise");
    // an evaluation seat with noise samples its moves too
    assert_ne!(games(with_gumbel(SeatCfg::Nn(NnSeat::eval(0, 24)), 16, 1.0, 1.0)), eval0);
}

#[test]
fn gumbel_mulligan_selfplay_and_evaluation() {
    // The Delver mulligan in miniature (KEEP's prior 0.999): PUCT never visits MULL; the Gumbel root
    // walks both options of every search (2 <= m), 32 / 31 of 63. Self-play with sigma (50, 1.0): the
    // 'cq' target follows Q - MULL with a bad hand, KEEP with a good one -, the moves are sampled from
    // it (the Gumbel noise) and a good hand's root value q is KEEP's (the improved policy's value, not
    // the halving's 50/50 mix). An evaluation seat without noise plays MULL with the bad hands; at the
    // Delver line's sigma scale 0.3 the prior still wins.
    let gumbel_mull = |selfplay: bool, scale: f64, cs: f64| match mull_seat(0, selfplay, PolicyTarget::Visits, Select::Visits) {
        SeatCfg::Nn(s) => with_gumbel(SeatCfg::Nn(s), 16, scale, cs),
        s => s,
    };
    let (rs, ss, hands) = mull_run(gumbel_mull(true, 1.0, 1.0), 160, true);
    assert_eq!(ss.len(), 320);
    let bad = |r: &Record| r.x[0] == 0.0 && r.x[3] == 0.0;
    let i_mull = MULL as usize;
    let (mut nb, mut high) = (0, 0);
    for r in &ss {
        if bad(r) {
            nb += 1;
            high += (r.pi[i_mull] > 0.9) as usize;
        } else {
            assert!(r.pi[i_mull] < 1e-3, "a good hand's cq target: {:?}", r.pi);
            let want = 0.5 + 0.3 * (r.x[0] - r.x[1]);
            assert!((r.q[0] - want).abs() < 1e-5, "a good hand's q {} (KEEP's value {want})", r.q[0]);
        }
    }
    assert!(nb > 40 && high * 10 >= nb * 9, "{high} of {nb} bad-hand targets on MULL");
    let (m, all) = mull_moves(&rs, &hands);
    assert_eq!(m[1], 0, "MULL sampled with a good hand");
    assert!(m[0] * 10 >= all[0] * 8, "MULL sampled with {} of {} bad hands", m[0], all[0]);
    eprintln!("gumbel mull selfplay: cq targets > 0.9 on MULL for {high} of {nb} bad hands; MULL played with {} of {} bad hands, {} of {} good", m[0], all[0], m[1], all[1]);
    // evaluation seats, no noise
    let (re, _, he) = mull_run(gumbel_mull(false, 0.0, 1.0), 200, false);
    let (me, ae) = mull_moves(&re, &he);
    assert_eq!(me[1], 0);
    assert!(me[0] * 10 >= ae[0] * 9, "eval: MULL with {} of {} bad hands", me[0], ae[0]);
    let (r3, _, h3) = mull_run(gumbel_mull(false, 0.0, 0.3), 200, false);
    let (m3, a3) = mull_moves(&r3, &h3);
    assert_eq!(m3, [0, 0], "sigma scale 0.3 against a 0.999 prior: KEEP ({a3:?} decisions)");
    eprintln!("gumbel mull eval: MULL with {} of {} bad hands at sigma scale 1.0, {} at 0.3", me[0], ae[0], m3[0]);
}

#[test]
fn gumbel_with_root_view_leaves_and_rolling_restarts() {
    // root-view leaf values are orthogonal: the hidden-hand mulligan with both - the value-only rows go
    // out as before and a bad hand's KEEP is valued from the keeper's own hand
    let seat = match hmull_seat(64, LeafView::Root, 0.0, ViewTarget::NextQ) {
        SeatCfg::Nn(s) => with_gumbel(SeatCfg::Nn(s), 16, 1.0, 1.0),
        s => s,
    };
    // (each search of seat 0 meets seat 1's decision after KEEP and after each redraw: ~3 leaves of the
    // other seat, every later walk a terminal)
    let (recs, vrows, st) = hmull_run(seat, 100, 5);
    assert!(vrows >= 250 && vrows as u64 == st.view_rows, "{vrows} value-only rows");
    assert!(recs.iter().all(|r| (r.pi.iter().sum::<f32>() - 1.0).abs() < 1e-4));
    // a rolling driver whose network changes restarts its Gumbel searches (the decision's noise kept)
    let mut c = cfg(vec![with_gumbel(nn(0, 8, true), 16, 1.0, 1.0)], vec![vec![0, 0]], &[("m", KeyKind::Nn)]);
    c.rolling = true;
    c.concurrency = 12;
    let mut d = Driver::<ToyGame>::new(c).unwrap();
    for phase in 0..4u32 {
        d.resume(10, phase / 2).unwrap();
        while let Some(b) = d.next_batch(true).unwrap() {
            answer(&mut d, &b);
            for s in d.take_samples() {
                assert!((s.pi.iter().sum::<f32>() - 1.0).abs() < 1e-4 && (s.q[0] + s.q[1] - 1.0).abs() < 1e-4);
            }
        }
    }
    let st = d.stats();
    assert!(st.restarts > 0 && st.games_finished >= 40);
}

#[test]
fn gumbel_with_a_root_belief() {
    // the Gumbel root and the root's belief together (docs/search.md 3.7, 3.8): toy self-play with
    // fast searches, the belief on every search (belief_fast) - a fresh root's belief row rides with
    // its own leaf, a reused root's waits alone, inside a Gumbel search whose schedule is its
    // simulations' either way: games finish, records valid, reproducible across threads; the belief
    // changes the Gumbel searches' worlds (other records than the Gumbel search without it)
    let pred = [0.9f32, 0.02, 0.02, 0.02, 0.02, 0.02];
    let mk = |threads: usize, groups: usize, tilt: f64| {
        let seat = with_belief(with_gumbel(nn(0, 16, true), 16, 1.0, 1.0), tilt, 20.0, 0.0, true);
        let mut c = cfg(vec![seat], vec![vec![0, 0]; 24], &[("m", KeyKind::Nn)]);
        c.threads = threads;
        c.groups = groups;
        c.seed = 2026;
        let mut d = Driver::<ToyGame>::new(c).unwrap();
        let (res, smp, rows) = run_belief(&mut d, &pred);
        (res, smp, rows, d.stats())
    };
    let (r1, s1, rows1, st1) = mk(1, 1, 1.0);
    let (r4, s4, rows4, _) = mk(3, 2, 1.0);
    assert_eq!(r1.len(), 24);
    assert!(r1.iter().all(|r| r.over));
    let key = |r: &GameResult| (r.gid, r.steps, r.scores.clone(), r.samples, r.decisions);
    assert_eq!(r1.iter().map(key).collect::<Vec<_>>(), r4.iter().map(key).collect::<Vec<_>>());
    assert_eq!((records_digest(&s1), rows1), (records_digest(&s4), rows4));
    assert!(rows1.0 > 0 && rows1.1 > 0, "belief rows after a fresh root's leaf {} and alone {}", rows1.0, rows1.1);
    assert_eq!(st1.belief_rows as usize, rows1.0 + rows1.1);
    for s in &s1 {
        assert!((s.pi.iter().sum::<f32>() - 1.0).abs() < 1e-4 && s.pi.iter().all(|x| x.is_finite() && *x >= 0.0));
        assert!((s.q[0] + s.q[1] - 1.0).abs() < 1e-4);
    }
    let (_, s0, rows0, _) = mk(1, 1, 0.0);
    assert_eq!(rows0, (0, 0));
    assert_ne!(records_digest(&s0), records_digest(&s1), "the belief changed nothing in the Gumbel searches");
}

// ------------------------------------------------------------------- branched games, archive starts

/// A self-play seat whose every search is full (no playout cap): each searched decision that is not a
/// deviation is recorded.
fn nn_full(sims: u32) -> SeatCfg {
    let mut s = NnSeat::eval(0, sims);
    s.selfplay = true;
    s.temp_moves = 6;
    SeatCfg::Nn(s)
}

/// Every result row with its own samples (a game's samples follow its predecessors' in emission
/// order), sorted by (gid, branch or not).
fn run_split<G: SearchGame>(d: &mut Driver<G>, skewed: bool) -> Vec<(GameResult, Vec<Record>)> {
    fn take<G: SearchGame>(d: &mut Driver<G>, rows: &mut Vec<(GameResult, Vec<Record>)>) {
        let rs = d.take_results();
        let smp = d.take_samples();
        let mut a = 0;
        for r in rs {
            let k = r.samples as usize;
            rows.push((r, smp[a..a + k].to_vec()));
            a += k;
        }
        assert_eq!(a, smp.len(), "every sample belongs to a row taken with it");
    }
    let mut rows = Vec::new();
    while let Some(b) = d.next_batch(true).unwrap() {
        if skewed {
            answer_skewed(d, &b);
        } else {
            answer(d, &b);
        }
        take(d, &mut rows);
    }
    take(d, &mut rows);
    rows.sort_by_key(|(r, _)| (r.gid, r.branch.is_some()));
    rows
}

fn branch_cfg(games: usize, p: f64) -> DriverCfg {
    let mut c = cfg(vec![nn_full(12)], vec![vec![0, 0]; games], &[("m", KeyKind::Nn)]);
    c.seed = 31;
    c.branch = Some(BranchCfg { p, moves: 3, temp: 2.0, differ: false });
    c
}

#[test]
fn branched_games_leave_their_games_alone_and_record_their_own_outcome() {
    let on = run_split(&mut Driver::<ToyGame>::new(branch_cfg(40, 0.5)).unwrap(), true);
    let mut c = branch_cfg(40, 0.5);
    c.branch = None;
    let off = run_split(&mut Driver::<ToyGame>::new(c).unwrap(), true);
    assert_eq!(off.len(), 40);
    // the games themselves: move for move and record for record the games without branching
    let main: Vec<_> = on.iter().filter(|(r, _)| r.branch.is_none()).collect();
    assert_eq!(main.len(), 40);
    for ((r, s), (r0, s0)) in main.iter().zip(&off) {
        assert_eq!((r.gid, r.steps, &r.scores, r.decisions, r.samples), (r0.gid, r0.steps, &r0.scores, r0.decisions, r0.samples));
        assert_eq!(records_digest(s), records_digest(s0), "game {}: its records changed", r.gid);
    }
    let br: Vec<_> = on.iter().filter(|(r, _)| r.branch.is_some()).collect();
    assert!((10..=30).contains(&br.len()), "{} branches of 40 games at p 0.5", br.len());
    let mut gids: Vec<u32> = br.iter().map(|(r, _)| r.gid).collect();
    gids.dedup();
    assert_eq!(gids.len(), br.len(), "one branch a game at most");
    let mut moves = [0u32; 4];
    for (r, s) in &br {
        let b = r.branch.as_ref().unwrap();
        let parent = &off[r.gid as usize].0;
        assert!((1..=3).contains(&b.moves) && b.played <= b.moves && b.at < b.of, "{b:?}");
        assert!(b.decision < parent.decisions && b.step < parent.steps, "{b:?} in a game of {} decisions", parent.decisions);
        assert!(r.over && b.start.is_none());
        moves[b.moves as usize] += 1;
        // nothing recorded at or before the deviation moves: with every search full, the records are
        // exactly the branch's searched decisions after them
        assert_eq!(r.samples, r.decisions - b.decision - b.played, "{b:?}");
        assert_eq!(s.len(), r.samples as usize);
        for x in s.iter() {
            // the branch's own outcome, relative to each record's decider
            for j in 0..2 {
                assert_eq!(x.z[j], r.result[(x.player as usize + j) % 2]);
            }
        }
    }
    assert!(moves[1] > 0 && moves[2] > 0 && moves[3] > 0, "deviation moves 1..=3 all drawn: {moves:?}");
    // the branches play games of their own (another score or length than their game's somewhere)
    assert!(br.iter().any(|(r, _)| r.scores != off[r.gid as usize].0.scores || r.steps != off[r.gid as usize].0.steps));
    let mut d = Driver::<ToyGame>::new(branch_cfg(40, 0.5)).unwrap();
    let again = run_split(&mut d, true);
    let st = d.stats();
    assert_eq!(st.branches as usize, br.len());
    assert_eq!(st.dev_moves, br.iter().map(|(r, _)| r.branch.as_ref().unwrap().played as u64).sum::<u64>());
    // reproducible from the seed
    let key = |v: &Vec<(GameResult, Vec<Record>)>| v.iter().map(|(r, s)| (r.gid, r.steps, r.scores.clone(), r.branch.clone(), records_digest(s))).collect::<Vec<_>>();
    assert_eq!(key(&on), key(&again));
}

#[test]
fn branches_count_in_rolling_quotas_and_survive_the_phases() {
    let mut c = branch_cfg(1, 0.4);
    c.rolling = true;
    c.concurrency = 10;
    let mut d = Driver::<ToyGame>::new(c).unwrap();
    let mut rows = Vec::new();
    for phase in 0..5u32 {
        d.resume(12, phase).unwrap();
        let mut fin = 0;
        while let Some(b) = d.next_batch(true).unwrap() {
            answer(&mut d, &b);
            let r = d.take_results();
            fin += r.len();
            rows.extend(r);
            d.take_samples();
        }
        let r = d.take_results();
        fin += r.len();
        rows.extend(r);
        assert!(fin >= 12, "phase {phase}: {fin} games (branches count)");
    }
    let nb = rows.iter().filter(|r| r.branch.is_some()).count();
    assert!(nb > 0);
    // a branch's row comes after its game's
    for (i, r) in rows.iter().enumerate().filter(|(_, r)| r.branch.is_some()) {
        assert!(rows[..i].iter().any(|q| q.gid == r.gid && q.branch.is_none()), "branch of game {} before its game", r.gid);
    }
    let st = d.stats();
    assert_eq!(st.games_finished as usize, rows.len());
    assert!(st.branches as usize >= nb && st.branches as usize <= nb + 10, "branches started {} finished {nb}", st.branches);
}

#[test]
fn archive_starts_play_their_entries_specs() {
    let entries = vec![("rich".to_string(), r#"{"food": [30, 30]}"#.to_string()), ("poor".to_string(), r#"{"food": [0, 9]}"#.to_string())];
    let mk = |spec: &str, starts: Option<StartsCfg>| {
        let mut c = cfg(vec![nn_full(8)], vec![vec![0, 0]; 30], &[("m", KeyKind::Nn)]);
        c.spec = spec.into();
        c.starts = starts;
        let mut d = Driver::<ToyGame>::new(c).unwrap();
        let r = run_split(&mut d, false);
        (r, d.stats())
    };
    let (on, st) = mk("{}", Some(StartsCfg { p: 0.5, specs: entries.clone() }));
    let (plain, _) = mk("{}", None);
    let (rich, _) = mk(&entries[0].1, None);
    let (poor, _) = mk(&entries[1].1, None);
    let n = on.iter().filter(|(r, _)| r.start.is_some()).count();
    assert!((7..=23).contains(&n), "{n} archive starts of 30 at p 0.5");
    assert_eq!(st.starts as usize, n);
    assert!(on.iter().any(|(r, _)| r.start == Some(0)) && on.iter().any(|(r, _)| r.start == Some(1)));
    // game g started from entry j is game g of a driver whose spec is entry j's (the same streams);
    // the others are the driver's own games
    for (r, s) in &on {
        let refr = match r.start {
            None => &plain,
            Some(0) => &rich,
            Some(_) => &poor,
        };
        let (r0, s0) = &refr[r.gid as usize];
        assert_eq!((r.steps, &r.scores, r.samples), (r0.steps, &r0.scores, r0.samples), "game {} start {:?}", r.gid, r.start);
        assert_eq!(records_digest(s), records_digest(s0));
    }
}

#[test]
fn a_branch_of_an_archive_start_names_its_entry() {
    let mut c = branch_cfg(30, 1.0);
    c.starts = Some(StartsCfg { p: 1.0, specs: vec![("e".into(), r#"{"food": [5, 5]}"#.into())] });
    let rows = run_split(&mut Driver::<ToyGame>::new(c).unwrap(), false);
    assert!(rows.iter().filter(|(r, _)| r.branch.is_none()).all(|(r, _)| r.start == Some(0)));
    let br: Vec<_> = rows.iter().filter_map(|(r, _)| r.branch.as_ref().map(|b| (r, b))).collect();
    assert!(br.len() >= 25, "every game branches at p 1 ({} of 30)", br.len());
    assert!(br.iter().all(|(r, b)| r.start.is_none() && b.start == Some(0)));
}

#[test]
fn branch_and_starts_settings_are_checked() {
    let bad_branch = |b: BranchCfg| {
        let mut c = branch_cfg(2, 0.5);
        c.branch = Some(b);
        Driver::<ToyGame>::new(c).err().unwrap_or_default()
    };
    assert!(bad_branch(BranchCfg { p: 0.0, moves: 3, temp: 2.0, differ: false }).contains("branch p"));
    assert!(bad_branch(BranchCfg { p: 1.5, moves: 3, temp: 2.0, differ: false }).contains("branch p"));
    assert!(bad_branch(BranchCfg { p: 0.5, moves: 0, temp: 2.0, differ: false }).contains("moves 0"));
    assert!(bad_branch(BranchCfg { p: 0.5, moves: 2, temp: 0.0, differ: false }).contains("temp"));
    // only drivers that record a self-play seat branch or start from an archive
    let mut c = branch_cfg(2, 0.5);
    c.record = false;
    assert!(Driver::<ToyGame>::new(c).err().unwrap_or_default().contains("never branches"));
    let mut c = cfg(vec![nn(0, 8, false)], vec![vec![0, 0]], &[("m", KeyKind::Nn)]);
    c.branch = Some(BranchCfg { p: 0.5, moves: 3, temp: 2.0, differ: false });
    assert!(Driver::<ToyGame>::new(c).is_err());
    let bad_starts = |s: StartsCfg| {
        let mut c = cfg(vec![nn_full(8)], vec![vec![0, 0]; 2], &[("m", KeyKind::Nn)]);
        c.starts = Some(s);
        Driver::<ToyGame>::new(c).err().unwrap_or_default()
    };
    assert!(bad_starts(StartsCfg { p: 0.5, specs: vec![] }).contains("no entries"));
    assert!(bad_starts(StartsCfg { p: 0.0, specs: vec![("a".into(), "{}".into())] }).contains("starts p"));
    assert!(bad_starts(StartsCfg { p: 0.5, specs: vec![("a".into(), r#"{"nope": 1}"#.into())] }).contains("entry \"a\""));
    // an entry the game refuses to start is refused when the driver is made
    assert!(bad_starts(StartsCfg { p: 0.5, specs: vec![("three".into(), r#"{"food": [1, 1, 1]}"#.into())] }).contains("does not start"));
}

#[test]
fn differ_never_replays_the_games_own_move_first() {
    let run = |differ: bool| {
        let mut c = branch_cfg(60, 1.0);
        c.branch = Some(BranchCfg { p: 1.0, moves: 1, temp: 2.0, differ });
        run_split(&mut Driver::<ToyGame>::new(c).unwrap(), true)
    };
    let br = |v: &Vec<(GameResult, Vec<Record>)>| v.iter().filter_map(|(r, _)| r.branch.clone()).collect::<Vec<_>>();
    let plain = br(&run(false));
    let differ = br(&run(true));
    assert!(plain.iter().any(|b| b.same_first), "without differ some first deviations are the game's own move");
    assert!(differ.len() >= 50 && differ.iter().all(|b| !b.same_first && b.played == 1));
}

// ---- the evaluation cache (NnSeat::eval_cache, docs/search.md 3.12) ----

fn with_cache(seat: SeatCfg, n: u32) -> SeatCfg {
    match seat {
        SeatCfg::Nn(mut s) => {
            s.eval_cache = n;
            SeatCfg::Nn(s)
        }
        s => s,
    }
}

/// A batch's features dense (a sparse batch's non-zeros scattered).
fn dense_x(b: &Batch, f: usize) -> Vec<f32> {
    if b.xs_off.is_empty() {
        return b.x.clone();
    }
    let mut x = vec![0.0; b.b * f];
    for r in 0..b.b {
        for k in b.xs_off[r] as usize..b.xs_off[r + 1] as usize {
            x[r * f + b.xs_idx[k] as usize] = b.xs_val[k];
        }
    }
    x
}

/// `answer_skewed` (priors and value functions of the row alone) shifted by `salt` (a network id, an
/// evaluator key): what a cache would answer wrongly if it kept another network's answers.
fn answer_salted<G: SearchGame>(d: &mut Driver<G>, b: &Batch, salt: f32) {
    let f = G::NUM_FEATURES;
    let x = dense_x(b, f);
    let mut pri = Vec::new();
    let mut v = vec![0.0; b.b * MAX_SEATS];
    for r in 0..b.b {
        let row = &x[r * f..(r + 1) * f];
        let h: f32 = row.iter().enumerate().map(|(i, x)| x * ((i * 5 % 11) as f32 - 5.0)).sum();
        let (lo, hi) = (b.opt_off[r] as usize, b.opt_off[r + 1] as usize);
        let e: Vec<f32> = b.opt_action[lo..hi].iter().map(|&a| (1.5 * (a as f32 * 1.7 + h * 0.3 + salt).sin()).exp()).collect();
        let t: f32 = e.iter().sum();
        pri.extend(e.iter().map(|x| x / t));
        let n = b.n_players[r] as usize;
        let h: f32 = row.iter().enumerate().map(|(i, x)| x * ((i * 7 % 13) as f32 - 6.0)).sum();
        let w = 1.0 / (1.0 + (-h * 0.1 - salt).exp());
        v[r * MAX_SEATS] = w;
        for j in 1..n {
            v[r * MAX_SEATS + j] = (1.0 - w) / (n - 1) as f32;
        }
    }
    d.submit(b.id, &pri, &v, &vec![0.0; b.b * MAX_SEATS]).unwrap();
}

/// The records' digest as a multiset (each record's digest, sorted, digested): the evaluation cache changes
/// when games finish, so the order their records come out in, never the records.
fn records_set_digest(recs: &[Record]) -> u64 {
    let mut each: Vec<u64> = recs.iter().map(|r| records_digest(std::slice::from_ref(r))).collect();
    each.sort_unstable();
    let mut h: u64 = 0xcbf29ce484222325;
    for v in each {
        for x in v.to_le_bytes() {
            h ^= x as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Self-play / a match of `seats` (keys "a", "b" when two) on ToyGame answered by `answer_salted` (salt =
/// the batch's key) -> (results digest, records digest (a multiset), rows sent, value-only rows sent, stats).
fn cache_run(seats: Vec<SeatCfg>, games: usize, sparse: bool) -> (u64, u64, u64, u64, DriverStats) {
    let two = seats.len() == 2;
    let keys: &[(&str, KeyKind)] = if two { &[("a", KeyKind::Nn), ("b", KeyKind::Nn)] } else { &[("m", KeyKind::Nn)] };
    let table = (0..games).map(|g| if !two { vec![0, 0] } else if g % 2 == 0 { vec![0, 1] } else { vec![1, 0] }).collect();
    let mut c = cfg(seats, table, keys);
    c.sparse = sparse;
    let mut d = Driver::<ToyGame>::new(c).unwrap();
    let (mut res, mut smp, mut rows, mut vrows) = (Vec::new(), Vec::new(), 0u64, 0u64);
    while let Some(b) = d.next_batch(true).unwrap() {
        rows += b.b as u64;
        vrows += b.extra.iter().filter(|&&e| e == ROW_VALUE).count() as u64;
        answer_salted(&mut d, &b, b.key as f32);
        res.extend(d.take_results());
        smp.extend(d.take_samples());
    }
    res.extend(d.take_results());
    smp.extend(d.take_samples());
    res.sort_by_key(|r| r.gid);
    (results_digest(&res), records_set_digest(&smp), rows, vrows, d.stats())
}

#[test]
fn the_eval_cache_plays_the_same_games_with_fewer_rows() {
    // An evaluator whose answer is a function of the row alone: the cache changes which rows are sent,
    // never an answer -> the same games and training records bit for bit, every row either sent or
    // answered from the cache, as many simulations; dense and sparse batches; a tiny cache (emptied
    // when full) too
    let seat = |n: u32| with_cache(nn(0, 32, true), n);
    let (r0, s0, n0, _, st0) = cache_run(vec![seat(0)], 40, false);
    assert_eq!((st0.cache_hits, st0.cache_leaves, st0.cache_clears), (0, 0, 0));
    for (cap, sparse) in [(1 << 20, false), (1 << 20, true), (3, false)] {
        let (r1, s1, n1, _, st1) = cache_run(vec![seat(cap)], 40, sparse);
        eprintln!("eval_cache {cap} (sparse {sparse}): {n1} rows sent of {n0}, {} answered from the cache, {} leaves at once, {} clears", st1.cache_hits, st1.cache_leaves, st1.cache_clears);
        assert_eq!((r1, s1), (r0, s0), "eval_cache {cap}: other games");
        assert_eq!(st1.sims, st0.sims);
        assert_eq!(n1 + st1.cache_hits, n0, "every row sent or cached");
        assert!(st1.cache_hits > 0 && st1.cache_leaves > 0, "no hits: the test proves nothing");
        assert_eq!(st1.leaves + st1.cache_leaves, st0.leaves);
        assert_eq!(st1.cache_clears > 0, cap == 3);
        assert_eq!(st1.cache_partial, 0, "a leaf of one row answered in part");
    }
}

#[test]
fn the_eval_cache_keeps_each_networks_answers_apart() {
    // a match: each evaluator key its own answers (the salt); a cache shared by the keys would answer one
    // network's rows with the other's
    let s = |k: u16, n: u32| with_cache(nn(k, 24, false), n);
    let (r0, s0, n0, _, _) = cache_run(vec![s(0, 0), s(1, 0)], 24, false);
    let (r1, s1, n1, _, st1) = cache_run(vec![s(0, 4096), s(1, 4096)], 24, false);
    assert_eq!((r1, s1), (r0, s0));
    assert!(st1.cache_hits > 0 && n1 + st1.cache_hits == n0);
    // rolling phases with a new network every other phase (games in flight across the change: their
    // searches start again): the network id is in every key and the new search epoch empties the
    // caches, so after the last change every game's cache holds the new network's answers alone. (The
    // games themselves are not compared with a run without the cache: the cache changes when games
    // finish, so which games a phase's quota ends and which are in flight at a change.)
    let mut c = cfg(vec![with_cache(nn(0, 16, true), 4096)], vec![vec![0, 0]], &[("m", KeyKind::Nn)]);
    c.rolling = true;
    c.concurrency = 12;
    let mut d = Driver::<ToyGame>::new(c).unwrap();
    for phase in 0..4u32 {
        let net = phase / 2;
        d.resume(10, net).unwrap();
        while let Some(b) = d.next_batch(true).unwrap() {
            answer_salted(&mut d, &b, net as f32 * 0.7);
            d.take_results();
            d.take_samples();
        }
    }
    let st = d.stats();
    assert!(st.cache_hits > 0 && st.epoch == 1 && st.restarts > 0, "hits {}, epoch {}, restarts {}", st.cache_hits, st.epoch, st.restarts);
    let (mut held, mut stale) = (0, 0);
    for g in &d.sh.groups {
        for ch in &g.chunks {
            for slot in &lock(ch).slots {
                held += slot.cache.len();
                stale += slot.cache.keys().filter(|k| k.1 != 1).count();
            }
        }
    }
    assert!(held > 0 && stale == 0, "{held} answers held, {stale} of the old network's");
}

#[test]
fn the_eval_cache_with_root_view_value_rows() {
    // LeafView::Root: another seat's leaf sends its own row (priors) and a value-only row of the root
    // decider; each is looked up alone, so a leaf may send one of them (the other cached), both or none.
    // The same games and records bit for bit; value-only rows cached too
    let seat = |n: u32| with_cache(with_view(nn(0, 32, true), LeafView::Root, 0.0, ViewTarget::NextQ), n);
    let (r0, s0, n0, v0, _) = cache_run(vec![seat(0)], 40, false);
    let (r1, s1, n1, v1, st1) = cache_run(vec![seat(4096)], 40, false);
    eprintln!("root view: {n1} rows sent of {n0} ({v1} value-only of {v0}), {} cached, {} leaves at once", st1.cache_hits, st1.cache_leaves);
    assert_eq!((r1, s1), (r0, s0));
    assert_eq!(n1 + st1.cache_hits, n0);
    assert!(v0 > 0 && v1 < v0, "no value-only row cached");
    assert!(st1.cache_partial > 0, "no leaf answered in part");
}

#[test]
fn the_eval_cache_with_root_view_rows_and_the_belief() {
    // HiddenMull with root-view values and the root's belief: fresh roots whose leaf is cached send their
    // belief row alone (a reused root's way); the same records bit for bit, one belief row per search
    let run = |n: u32| {
        let mut c = cfg(vec![with_cache(hmull_belief_seat(64, 1.0, LeafView::Root, true), n)], vec![vec![0, 0]; 200], &[("m", KeyKind::Nn)]);
        c.seed = 5;
        let mut d = Driver::<HiddenMull>::new(c).unwrap();
        let f = HiddenMull::NUM_FEATURES;
        let (mut smp, mut rows, mut vrows, mut brows) = (Vec::new(), 0u64, 0u64, 0u64);
        while let Some(b) = d.next_batch(true).unwrap() {
            let (w, a) = belief_rows_of(&b);
            brows += (w + a) as u64;
            rows += b.b as u64;
            let mut pri = Vec::new();
            let mut v = vec![0.0; b.b * MAX_SEATS];
            for r in 0..b.b {
                let k = (b.opt_off[r + 1] - b.opt_off[r]) as usize;
                vrows += (b.extra[r] == ROW_VALUE) as u64;
                let (p, val) = hmull_eval(&b.x[r * f..(r + 1) * f], k);
                pri.extend(p);
                v[r * MAX_SEATS..(r + 1) * MAX_SEATS].copy_from_slice(&val);
            }
            let aux: Vec<f32> = (0..w + a).flat_map(|_| [0.1f32, 0.9]).collect();
            d.submit_aux(b.id, &pri, &v, &vec![0.0; b.b * MAX_SEATS], &aux).unwrap();
            smp.extend(d.take_samples());
        }
        smp.extend(d.take_samples());
        (records_set_digest(&smp), rows, vrows, brows, d.stats())
    };
    let (s0, n0, v0, b0, st0) = run(0);
    let (s1, n1, v1, b1, st1) = run(4096);
    eprintln!("hidden mull: {n1} rows sent of {n0} ({v1} value-only of {v0}), {} cached, {} leaves at once", st1.cache_hits, st1.cache_leaves);
    assert_eq!(s1, s0, "other records");
    assert_eq!((b1, st1.belief_rows), (b0, st0.belief_rows), "one belief row per search either way");
    assert_eq!(n1 + st1.cache_hits, n0);
    assert!(st1.cache_leaves > 0 && v0 > 0, "{} leaves at once, {v0} value-only rows", st1.cache_leaves);
}

// ---- self-play resignation (NnSeat::resign_t, docs/search.md 3.14) ----

fn with_resign(seat: SeatCfg, t: f64, k: u32, noresign: f64) -> SeatCfg {
    match seat {
        SeatCfg::Nn(mut s) => {
            s.resign_t = t;
            s.resign_k = k;
            s.resign_noresign = noresign;
            SeatCfg::Nn(s)
        }
        s => s,
    }
}

/// Uniform priors; the decider's value from its score lead and the round (ToyGame's features: a lead late in
/// the game is decisive) - a function of the row alone, so every game is the same whatever its batches.
fn answer_lead(d: &mut Driver<ToyGame>, b: &Batch) {
    let f = ToyGame::NUM_FEATURES;
    let base = 6 + 3 + 4 + 12 + 6 + crate::toy::N_SPACES * 7;
    let mut pri = Vec::new();
    for r in 0..b.b {
        let k = (b.opt_off[r + 1] - b.opt_off[r]) as usize;
        pri.extend(std::iter::repeat_n(1.0 / k as f32, k));
    }
    let mut v = vec![0.0; b.b * MAX_SEATS];
    let m = vec![0.0; b.b * MAX_SEATS];
    for r in 0..b.b {
        let x = &b.x[r * f..(r + 1) * f];
        let lead = (x[base + 14] - x[base + 16 + 14]) * 20.0;
        let round = (0..6).find(|&i| x[i] > 0.0).unwrap_or(0) as f32 + 1.0;
        let w = 1.0 / (1.0 + (-1.5 * lead * round / 6.0).exp());
        v[r * MAX_SEATS] = w;
        v[r * MAX_SEATS + 1] = 1.0 - w;
    }
    d.submit(b.id, &pri, &v, &m).unwrap();
}

/// `run_split` with `answer_lead`: every result row with its own samples, sorted by (gid, branch or not).
fn run_lead(d: &mut Driver<ToyGame>) -> Vec<(GameResult, Vec<Record>)> {
    fn take(d: &mut Driver<ToyGame>, rows: &mut Vec<(GameResult, Vec<Record>)>) {
        let rs = d.take_results();
        let smp = d.take_samples();
        let mut a = 0;
        for r in rs {
            let k = r.samples as usize;
            rows.push((r, smp[a..a + k].to_vec()));
            a += k;
        }
        assert_eq!(a, smp.len(), "every sample belongs to a row taken with it");
    }
    let mut rows = Vec::new();
    while let Some(b) = d.next_batch(true).unwrap() {
        answer_lead(d, &b);
        take(d, &mut rows);
    }
    take(d, &mut rows);
    rows.sort_by_key(|(r, _)| (r.gid, r.branch.is_some()));
    rows
}

fn resign_cfg(seat: SeatCfg, games: usize, t: f64, k: u32, noresign: f64) -> DriverCfg {
    let mut c = cfg(vec![with_resign(seat, t, k, noresign)], vec![vec![0, 0]; games], &[("m", KeyKind::Nn)]);
    c.seed = 11;
    c
}

#[test]
fn resignation_off_or_a_dry_run_plays_the_games_of_before() {
    // off (resign_t 0): the golden records; a dry run (every game a no-resign game): the same games, records and
    // streams, each row reporting where the rule would have fired
    let (r0, s0) = golden_selfplay(nn(0, 16, true));
    assert_eq!(records_digest(&s0), GOLDEN_RECORDS.1);
    assert!(r0.iter().all(|r| r.resign.is_none()));
    let (r1, s1) = golden_selfplay(with_resign(nn(0, 16, true), 0.3, 2, 1.0));
    assert_eq!((s1.len(), records_digest(&s1)), GOLDEN_RECORDS, "a dry run changed the records");
    let key = |r: &GameResult| (r.gid, r.over, r.result.clone(), r.steps, r.scores.clone(), r.samples, r.decisions);
    assert_eq!(r0.iter().map(key).collect::<Vec<_>>(), r1.iter().map(key).collect::<Vec<_>>());
    let fired = r1.iter().filter(|r| r.resign.as_ref().unwrap().winner.is_some()).count();
    assert!(r1.iter().all(|r| r.resign.as_ref().is_some_and(|x| x.noresign && !x.resigned)));
    assert!(fired > 0, "the dry run never fired at t 0.3, k 2");
}

#[test]
fn resigned_games_end_where_the_rule_fires_with_its_outcome() {
    // every game resigns where the rule fires (noresign 0) vs the dry run (noresign 1: the games of before, each
    // noting where it would have): a resigned game is its dry twin up to that point - the same records but z, the
    // resignation's outcome for the favoured seat - and nothing after it. With a playout cap (fast searches never
    // count), a Gumbel root (its root value counts) and every search full.
    let seats = [nn_full(16), nn(0, 16, true), with_gumbel(nn(0, 16, true), 4, 1.0, 1.0)];
    for (si, seat) in seats.into_iter().enumerate() {
        let on = run_lead(&mut Driver::<ToyGame>::new(resign_cfg(seat.clone(), 40, 0.05, 4, 0.0)).unwrap());
        let dry = run_lead(&mut Driver::<ToyGame>::new(resign_cfg(seat, 40, 0.05, 4, 1.0)).unwrap());
        assert_eq!((on.len(), dry.len()), (40, 40));
        let (mut resigned, mut wrong) = (0, 0);
        for ((r, s), (r0, s0)) in on.iter().zip(&dry) {
            assert_eq!(r.gid, r0.gid);
            let (x, x0) = (r.resign.as_ref().unwrap(), r0.resign.as_ref().unwrap());
            assert!(!x.noresign && x0.noresign && !x0.resigned);
            let Some(w) = x0.winner else {
                // the rule never fired: the game of before
                assert!(!x.resigned && x.winner.is_none());
                assert_eq!((r.over, &r.result, r.steps, r.samples, r.decisions), (r0.over, &r0.result, r0.steps, r0.samples, r0.decisions));
                assert_eq!(records_digest(s), records_digest(s0));
                continue;
            };
            resigned += 1;
            assert!(x.resigned, "seat {si} game {}: the rule fired in its twin but it did not resign", r.gid);
            assert_eq!((x.winner, x.decision, x.step, x.records), (x0.winner, x0.decision, x0.step, x0.records), "seat {si} game {}", r.gid);
            assert!(r.over && x.decision <= r0.decisions && x.step < r0.steps);
            assert_eq!((r.steps, r.decisions, r.samples), (x.step, x.decision, x.records));
            let mut z = vec![0.0; 2];
            z[w as usize] = 1.0;
            assert_eq!(r.result, z, "the favoured seat wins");
            wrong += (r0.result[w as usize] < 1.0) as usize;
            // its records: the twin's first ones, z the resignation's (relative to each record's decider)
            assert_eq!(s.len(), x.records as usize);
            for (a, b) in s.iter().zip(s0) {
                assert_eq!((&a.x, &a.pi, &a.opts, a.player, a.q, a.net), (&b.x, &b.pi, &b.opts, b.player, b.q, b.net));
                for j in 0..2 {
                    assert_eq!(a.z[j], z[(a.player as usize + j) % 2]);
                }
            }
            // the firing search (the last record): its root value calls the favoured seat's win (q: the decider's view)
            let last = s.last().unwrap();
            let qf = last.q[(w as usize + 2 - last.player as usize) % 2];
            assert!(qf as f64 >= 0.95 - 1e-6, "the firing search's favoured estimate {qf}");
        }
        eprintln!("seat {si}: {resigned} of 40 games resigned, {wrong} against their played-out result");
        assert!((8..=36).contains(&resigned) && wrong < resigned, "seat {si}: {resigned} of 40 games resigned, {wrong} wrongly");
    }
}

#[test]
fn the_no_resign_share_is_a_hash_of_seed_and_gid() {
    assert!(!noresign_pick(5, 3, 0.0) && noresign_pick(5, 3, 1.0));
    let n = (0..20_000).filter(|&g| noresign_pick(7, g, 0.1)).count();
    assert!((1_800..2_200).contains(&n), "{n} of 20,000 at 0.1");
    assert!((0..20_000).all(|g| noresign_pick(7, g, 0.1) == noresign_pick(7, g, 0.1)));
    assert!((0..200).any(|g| noresign_pick(7, g, 0.5) != noresign_pick(8, g, 0.5)), "the seed matters");
    // a run with a share: each game's draw is noresign_pick's; the no-resign games are the dry run's games, the
    // others the all-resign run's (no stream moves)
    let all = run_lead(&mut Driver::<ToyGame>::new(resign_cfg(nn(0, 16, true), 40, 0.1, 3, 0.0)).unwrap());
    let dry = run_lead(&mut Driver::<ToyGame>::new(resign_cfg(nn(0, 16, true), 40, 0.1, 3, 1.0)).unwrap());
    let mix = run_lead(&mut Driver::<ToyGame>::new(resign_cfg(nn(0, 16, true), 40, 0.1, 3, 0.4)).unwrap());
    let mut free = 0;
    for (k, (r, s)) in mix.iter().enumerate() {
        let x = r.resign.as_ref().unwrap();
        assert_eq!(x.noresign, noresign_pick(11, r.gid, 0.4));
        let (r1, s1) = if x.noresign { &dry[k] } else { &all[k] };
        let y = r1.resign.as_ref().unwrap();
        assert_eq!((r.gid, &r.result, r.steps, r.samples), (r1.gid, &r1.result, r1.steps, r1.samples));
        assert_eq!((x.resigned, x.winner, x.decision, x.step, x.records), (y.resigned, y.winner, y.decision, y.step, y.records));
        assert_eq!(records_digest(s), records_digest(s1));
        free += x.noresign as usize;
    }
    assert!((6..=26).contains(&free), "{free} no-resign games of 40 at 0.4");
}

#[test]
fn resigned_games_count_in_rolling_quotas() {
    let mut c = resign_cfg(nn(0, 16, true), 1, 0.1, 3, 0.2);
    c.rolling = true;
    c.concurrency = 10;
    let mut d = Driver::<ToyGame>::new(c).unwrap();
    let (mut rows, mut samples) = (Vec::new(), 0usize);
    for phase in 0..5u32 {
        d.resume(12, phase).unwrap();
        let mut fin = 0;
        loop {
            let b = d.next_batch(true).unwrap();
            if let Some(b) = &b {
                answer_lead(&mut d, b);
            }
            let r = d.take_results();
            fin += r.len();
            samples += d.take_samples().len();
            rows.extend(r);
            if b.is_none() {
                break;
            }
        }
        assert!(fin >= 12, "phase {phase}: {fin} games (resigned games count)");
    }
    let st = d.stats();
    assert_eq!(st.games_finished as usize, rows.len());
    assert_eq!(rows.iter().map(|r| r.samples as usize).sum::<usize>(), samples);
    let res = rows.iter().filter(|r| r.resign.as_ref().unwrap().resigned).count();
    assert!(res > 0 && res < rows.len(), "{res} of {} resigned", rows.len());
    assert!(rows.iter().all(|r| r.over));
}

#[test]
fn branches_of_resigning_games() {
    // branched games (docs/search.md 2.11) with resignation: a game that resigns spawns its branch like one that
    // plays out; a branch counts its own searches (after its deviation moves) and shares its game's draw
    let mut c = branch_cfg(40, 1.0);
    c.seats = vec![with_resign(nn_full(12), 0.1, 3, 0.3)];
    let rows = run_lead(&mut Driver::<ToyGame>::new(c.clone()).unwrap());
    let br: Vec<_> = rows.iter().filter(|(r, _)| r.branch.is_some()).collect();
    assert!(br.len() >= 30, "{} branches of 40 games at p 1", br.len());
    let mut of_resigned = 0;
    for (r, s) in &br {
        let b = r.branch.as_ref().unwrap();
        let x = r.resign.as_ref().unwrap();
        let (g, _) = rows.iter().find(|(q, _)| q.gid == r.gid && q.branch.is_none()).unwrap();
        let gx = g.resign.as_ref().unwrap();
        assert_eq!(x.noresign, gx.noresign, "a branch shares its game's draw");
        assert_eq!(x.noresign, noresign_pick(c.seed, r.gid, 0.3));
        of_resigned += gx.resigned as usize;
        assert!(b.decision < g.decisions, "the branch point is a decision its game played");
        assert_eq!(s.len(), r.samples as usize);
        if x.resigned {
            // its records: its searched decisions after the deviation moves, up to the firing one
            assert_eq!(r.samples, x.decision - b.decision - b.played);
            assert_eq!(r.samples, x.records);
        } else {
            assert_eq!(r.samples, r.decisions - b.decision - b.played);
        }
    }
    assert!(of_resigned > 0, "some branches come from resigned games");
    assert!(br.iter().any(|(r, _)| r.resign.as_ref().unwrap().resigned), "some branches resign");
    // reproducible
    let again = run_lead(&mut Driver::<ToyGame>::new(c).unwrap());
    let key = |v: &Vec<(GameResult, Vec<Record>)>| v.iter().map(|(r, s)| (r.gid, r.steps, r.branch.clone(), r.resign.clone(), records_digest(s))).collect::<Vec<_>>();
    assert_eq!(key(&rows), key(&again));
}

#[test]
fn resign_settings_are_checked() {
    let err = |c: DriverCfg| Driver::<ToyGame>::new(c).err().unwrap_or_default();
    assert!(err(resign_cfg(nn(0, 8, true), 2, 0.5, 5, 0.1)).contains("resign_t 0.5"));
    assert!(err(resign_cfg(nn(0, 8, true), 2, -0.1, 5, 0.1)).contains("resign_t"));
    assert!(err(resign_cfg(nn(0, 8, true), 2, f64::NAN, 5, 0.1)).contains("resign_t"));
    assert!(err(resign_cfg(nn(0, 8, true), 2, 0.05, 0, 0.1)).contains("resign_k 0"));
    assert!(err(resign_cfg(nn(0, 8, true), 2, 0.05, 5, 1.5)).contains("resign_noresign"));
    // a seat that records no self-play never resigns: an evaluation seat, a league opponent, a driver that records nothing
    assert!(err(resign_cfg(nn(0, 8, false), 2, 0.05, 5, 0.1)).contains("never resigns"));
    let mut c = resign_cfg(nn(0, 8, true), 2, 0.05, 5, 0.1);
    c.record = false;
    assert!(err(c).contains("never resigns"));
    let league = match with_resign(nn(0, 8, true), 0.05, 5, 0.1) {
        SeatCfg::Nn(mut s) => {
            s.record = false;
            SeatCfg::Nn(s)
        }
        s => s,
    };
    assert!(err(cfg(vec![league], vec![vec![0, 0]], &[("m", KeyKind::Nn)])).contains("never resigns"));
    // two-player games; one counter per game
    let mut c = resign_cfg(nn(0, 8, true), 2, 0.05, 5, 0.1);
    c.games = vec![vec![0, 0, 0]];
    assert!(err(c).contains("two-player"));
    let c = cfg(vec![with_resign(nn(0, 8, true), 0.05, 5, 0.1), with_resign(nn(1, 8, true), 0.05, 3, 0.1)], vec![vec![0, 1]], &[("m", KeyKind::Nn), ("o", KeyKind::Nn)]);
    assert!(err(c).contains("differ"));
    // a resigning seat against one that does not (another network, say): fine
    let c = cfg(vec![with_resign(nn(0, 8, true), 0.05, 5, 0.1), nn(1, 8, true)], vec![vec![0, 1]], &[("m", KeyKind::Nn), ("o", KeyKind::Nn)]);
    assert!(Driver::<ToyGame>::new(c).is_ok());
    // a game with deferred terminals cannot resign: refused at its first one (a rule that never fires here)
    let c = cfg(vec![with_resign(nn(0, 8, true), 0.0001, 50, 0.0)], vec![vec![0, 0]; 2], &[("m", KeyKind::Nn)]);
    let mut d = Driver::<ToyTeam>::new(c).unwrap();
    let e = loop {
        match d.next_batch(true) {
            Ok(Some(b)) => answer(&mut d, &b),
            Ok(None) => break String::new(),
            Err(e) => break e,
        }
    };
    assert!(e.contains("deferred terminals cannot resign"), "{e:?}");
}

#[test]
fn sparse_rows_from_encode_sparse_are_the_dense_rows() {
    // docs/search.md 2.12: a sparse driver reads every row with `encode_sparse` (the default: `encode` +
    // `sparse_scan`) and keys the evaluation cache by `obs_hash_sparse` of its non-zeros. Batch for batch the
    // dense driver's rows (densified), the same games and records, the same cache hits (the same keys): leaf
    // rows, root-view value rows and root belief rows, without and with the cache
    for cap in [0u32, 4096] {
        let seat = with_cache(with_view(nn(0, 32, true), LeafView::Root, 0.0, ViewTarget::NextQ), cap);
        let (r0, s0, n0, v0, st0) = cache_run(vec![seat.clone()], 40, false);
        let (r1, s1, n1, v1, st1) = cache_run(vec![seat], 40, true);
        assert_eq!((r1, s1, n1, v1), (r0, s0, n0, v0), "toy, eval_cache {cap}: sparse plays otherwise");
        assert_eq!((st1.cache_hits, st1.cache_leaves, st1.cache_partial, st1.sims), (st0.cache_hits, st0.cache_leaves, st0.cache_partial, st0.sims));
        assert!(v0 > 0 && (cap == 0 || st1.cache_hits > 0));
        let run = |sparse: bool| {
            let mut c = cfg(vec![with_cache(hmull_belief_seat(64, 1.0, LeafView::Root, true), cap)], vec![vec![0, 0]; 120], &[("m", KeyKind::Nn)]);
            c.seed = 11;
            c.sparse = sparse;
            let mut d = Driver::<HiddenMull>::new(c).unwrap();
            let f = HiddenMull::NUM_FEATURES;
            let (mut smp, mut xs, mut kinds) = (Vec::new(), Vec::new(), Vec::new());
            while let Some(b) = d.next_batch(true).unwrap() {
                assert_eq!(b.x.is_empty(), sparse);
                if sparse {
                    for r in 0..b.b {
                        let row = &b.xs_idx[b.xs_off[r] as usize..b.xs_off[r + 1] as usize];
                        assert!(row.windows(2).all(|w| w[0] < w[1]), "a row's indexes not ascending");
                    }
                    assert!(b.xs_val.iter().all(|v| v.to_bits() != 0), "a zero in the non-zeros");
                }
                let x = dense_x(&b, f);
                let (w, a) = belief_rows_of(&b);
                let mut pri = Vec::new();
                let mut v = vec![0.0; b.b * MAX_SEATS];
                for r in 0..b.b {
                    let k = (b.opt_off[r + 1] - b.opt_off[r]) as usize;
                    let (p, val) = hmull_eval(&x[r * f..(r + 1) * f], k);
                    pri.extend(p);
                    v[r * MAX_SEATS..(r + 1) * MAX_SEATS].copy_from_slice(&val);
                }
                let aux: Vec<f32> = (0..w + a).flat_map(|_| [0.1f32, 0.9]).collect();
                d.submit_aux(b.id, &pri, &v, &vec![0.0; b.b * MAX_SEATS], &aux).unwrap();
                xs.push(x);
                kinds.push(b.extra.clone());
                smp.extend(d.take_samples());
            }
            smp.extend(d.take_samples());
            (records_set_digest(&smp), xs, kinds, d.stats())
        };
        let (h0, x0, k0, t0) = run(false);
        let (h1, x1, k1, t1) = run(true);
        assert_eq!(x1.len(), x0.len(), "hidden mull, eval_cache {cap}: other batches");
        assert!(x1 == x0 && k1 == k0, "hidden mull, eval_cache {cap}: other rows");
        assert_eq!(h1, h0, "hidden mull, eval_cache {cap}: other records");
        assert_eq!((t1.belief_rows, t1.view_rows, t1.cache_hits, t1.leaves), (t0.belief_rows, t0.view_rows, t0.cache_hits, t0.leaves));
        assert!(t0.belief_rows > 0 && t0.view_rows > 0 && (cap == 0 || t0.cache_hits > 0));
    }
}

// ------------------------------------------------------------------ information-set merging (docs/search.md 3.16)

use crate::counter_game::{counter_eval, Counter};

fn with_merge(seat: SeatCfg, on: bool) -> SeatCfg {
    match seat {
        SeatCfg::Nn(mut s) => {
            s.infoset_merge = on;
            if on {
                s.leaf_value_view = LeafView::Root;
            }
            SeatCfg::Nn(s)
        }
        s => s,
    }
}

/// Answer a counter batch with its stand-in network (`counter_eval`: priors by option over the row's options,
/// the value relative to the row's player).
fn answer_counter(d: &mut Driver<Counter>, b: &Batch) {
    let f = Counter::NUM_FEATURES;
    let mut pri = Vec::new();
    let mut v = vec![0.0; b.b * MAX_SEATS];
    for r in 0..b.b {
        let opts: Vec<u16> = b.opt_action[b.opt_off[r] as usize..b.opt_off[r + 1] as usize].iter().map(|&a| a as u16).collect();
        let (p, val) = counter_eval(&b.x[r * f..(r + 1) * f], &opts);
        pri.extend(p);
        v[r * MAX_SEATS..(r + 1) * MAX_SEATS].copy_from_slice(&val);
    }
    d.submit(b.id, &pri, &v, &vec![0.0; b.b * MAX_SEATS]).unwrap();
}

/// A counter match (`spec`): key a (`sa`) vs key b (`sb`), seats alternating. -> (results, stats).
fn counter_match(sa: SeatCfg, sb: SeatCfg, games: usize, spec: &str, seed: u64, threads: usize) -> (Vec<GameResult>, DriverStats) {
    let gs = (0..games).map(|g| if g % 2 == 0 { vec![0, 1] } else { vec![1, 0] }).collect();
    let mut c = cfg(vec![sa, sb], gs, &[("a", KeyKind::Nn), ("b", KeyKind::Nn)]);
    c.spec = spec.into();
    c.seed = seed;
    c.record = false;
    c.threads = threads;
    let mut d = Driver::<Counter>::new(c).unwrap();
    let mut res = Vec::new();
    while let Some(b) = d.next_batch(true).unwrap() {
        answer_counter(&mut d, &b);
        res.extend(d.take_results());
    }
    res.extend(d.take_results());
    res.sort_by_key(|r| r.gid);
    (res, d.stats())
}

#[test]
fn infoset_merge_needs_root_view_values() {
    let mk = |view: LeafView, selfplay: bool| {
        let mut s = NnSeat::eval(0, 8);
        s.infoset_merge = true;
        s.leaf_value_view = view;
        s.selfplay = selfplay;
        cfg(vec![SeatCfg::Nn(s)], vec![vec![0, 0]], &[("m", KeyKind::Nn)])
    };
    for selfplay in [false, true] {
        let e = Driver::<ToyGame>::new(mk(LeafView::Mover, selfplay)).err().unwrap();
        assert!(e.contains("leaf_value_view 'root'"), "{e}");
        // (evaluation and self-play seats: docs/search.md 3.16)
        assert!(Driver::<ToyGame>::new(mk(LeafView::Root, selfplay)).is_ok());
    }
}

#[test]
fn infoset_merge_off_changes_nothing() {
    // the golden match and records with the setting given off (computed before it existed)
    let seats = vec![with_merge(nn(0, 16, false), false), with_merge(nn(1, 16, false), false)];
    let games = (0..40).map(|g| if g % 2 == 0 { vec![0, 1] } else { vec![1, 0] }).collect();
    let mut c = cfg(seats, games, &[("m", KeyKind::Nn), ("o", KeyKind::Nn)]);
    c.seed = 2026;
    c.record = false;
    let mut d = Driver::<ToyGame>::new(c).unwrap();
    assert_eq!(results_digest(&run(&mut d).0), GOLDEN_MATCH_VISITS);
    let st = d.stats();
    assert_eq!((st.merge_links, st.merge_merged, st.merge_growths), (0, 0, 0));
    let (_, recs) = golden_selfplay(with_merge(nn(0, 16, true), false));
    assert_eq!((recs.len(), records_digest(&recs)), GOLDEN_RECORDS);
}

#[test]
fn infoset_merge_plays_counter_matches_that_merge_and_reproduce() {
    // a merged seat against an unmerged one, both with root-view values: the merged trees merge (the other seat's
    // option lists, the next round across depths) and grow; the games are the same across thread counts, with the
    // evaluation cache (the stand-in network is a function of the row), and with tree reuse through advance_in
    let a = with_merge(nn(0, 24, false), true);
    let b = with_view(nn(1, 24, false), LeafView::Root, 0.0, ViewTarget::NextQ);
    let mut differ = 0;
    for spec in ["{}", r#"{"rounds": 4}"#, r#"{"rounds": 5, "forget": true}"#] {
        let (r0, s0) = counter_match(a.clone(), b.clone(), 200, spec, 11, 2);
        assert_eq!(r0.len(), 200);
        assert!(r0.iter().all(|r| r.over));
        assert!(s0.merge_merged > 0 && s0.merge_growths > 0 && s0.merge_links > s0.merge_merged, "{spec}: {s0:?}");
        if spec.contains("forget") {
            assert!(s0.merge_aliases > 0, "{spec}: {s0:?}");
        }
        let (r1, s1) = counter_match(a.clone(), b.clone(), 200, spec, 11, 1);
        assert_eq!(results_digest(&r1), results_digest(&r0), "{spec}: threads");
        assert_eq!((s1.merge_merged, s1.merge_growths), (s0.merge_merged, s0.merge_growths));
        let cached = |s: &SeatCfg| match s.clone() {
            SeatCfg::Nn(mut n) => {
                n.eval_cache = 64;
                SeatCfg::Nn(n)
            }
            s => s,
        };
        let (r2, s2) = counter_match(cached(&a), cached(&b), 200, spec, 11, 2);
        assert_eq!(results_digest(&r2), results_digest(&r0), "{spec}: the eval cache");
        assert!(s2.cache_hits > 0 && s2.merge_growths == s0.merge_growths);
        // the merged seat is not the unmerged one (in the longer games: the 2-round game's moves may coincide)
        let (r3, _) = counter_match(with_view(nn(0, 24, false), LeafView::Root, 0.0, ViewTarget::NextQ), b.clone(), 200, spec, 11, 2);
        differ += (results_digest(&r3) != results_digest(&r0)) as u32;
    }
    assert!(differ >= 1, "merging changed no game");
}

#[test]
fn infoset_merge_with_the_root_floor_gumbel_and_the_toy() {
    let floor = |s: SeatCfg| match s {
        SeatCfg::Nn(mut n) => {
            n.root_min_visits = 3;
            SeatCfg::Nn(n)
        }
        s => s,
    };
    let gumbel = |s: SeatCfg| match s {
        SeatCfg::Nn(mut n) => {
            n.root_search = RootSearch::Gumbel { m: 4, scale: 0.0, c_visit: 50.0, c_scale: 1.0 };
            SeatCfg::Nn(n)
        }
        s => s,
    };
    let b = with_view(nn(1, 16, false), LeafView::Root, 0.0, ViewTarget::NextQ);
    for a in [floor(with_merge(nn(0, 16, false), true)), gumbel(with_merge(nn(0, 16, false), true))] {
        let (r, s) = counter_match(a.clone(), b.clone(), 100, r#"{"rounds": 3}"#, 5, 2);
        assert_eq!(r.len(), 100);
        assert!(s.merge_growths > 0, "{s:?}");
        // the toy: chance keys, more seats, reuse
        let seats = vec![a, with_view(nn(1, 16, false), LeafView::Root, 0.0, ViewTarget::NextQ)];
        let games: Vec<Vec<u16>> = (0..24).map(|g| if g % 3 == 0 { vec![0, 1, 1] } else if g % 2 == 0 { vec![0, 1] } else { vec![1, 0] }).collect();
        let mut c = cfg(seats, games, &[("m", KeyKind::Nn), ("o", KeyKind::Nn)]);
        c.record = false;
        let mut d = Driver::<ToyGame>::new(c).unwrap();
        let (res, _) = run(&mut d);
        assert_eq!(res.len(), 24);
        assert!(d.stats().merge_links > 0);
    }
}

#[test]
fn infoset_merge_survives_rolling_network_changes() {
    // a rolling match driver whose network changes between phases: every merged search restarts from a fresh tree
    // (reset + set_merge) and the games go on
    let seats = vec![with_merge(nn(0, 16, false), true), with_view(nn(1, 16, false), LeafView::Root, 0.0, ViewTarget::NextQ)];
    let mut c = cfg(seats, vec![vec![0, 1], vec![1, 0]], &[("a", KeyKind::Nn), ("b", KeyKind::Nn)]);
    c.spec = r#"{"rounds": 6}"#.into();
    c.rolling = true;
    c.record = false;
    c.concurrency = 12;
    let mut d = Driver::<Counter>::new(c).unwrap();
    let mut total = 0;
    for phase in 0..6u32 {
        d.resume(10, phase).unwrap();
        while let Some(b) = d.next_batch(true).unwrap() {
            answer_counter(&mut d, &b);
            total += d.take_results().len();
        }
        total += d.take_results().len();
    }
    let st = d.stats();
    assert!(total >= 60 && st.restarts > 0 && st.epoch == 5, "{total} games, {st:?}");
    assert!(st.merge_merged > 0 && st.merge_growths > 0);
}

// ------------------------------------------------------- information-set merging in self-play (docs/search.md 3.16)

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &x in bytes {
        *h ^= x as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}

/// A batch's bytes as the evaluator receives them (dense or sparse rows, options, kinds, players, games, key).
fn batch_digest(h: &mut u64, b: &Batch) {
    for v in &b.x {
        fnv(h, &v.to_bits().to_le_bytes());
    }
    for v in &b.xs_off {
        fnv(h, &v.to_le_bytes());
    }
    for v in &b.xs_idx {
        fnv(h, &v.to_le_bytes());
    }
    for v in &b.xs_val {
        fnv(h, &v.to_bits().to_le_bytes());
    }
    for v in &b.opt_off {
        fnv(h, &v.to_le_bytes());
    }
    for v in &b.opt_action {
        fnv(h, &v.to_le_bytes());
    }
    fnv(h, &b.extra);
    for v in &b.player {
        fnv(h, &v.to_le_bytes());
    }
    for v in &b.gid {
        fnv(h, &v.to_le_bytes());
    }
    fnv(h, &b.key.to_le_bytes());
}

/// A stand-in network: (row's dense features, its options, players) -> (priors over the options, value per seat
/// relative to the row's player).
type RowEval = dyn Fn(&[f32], &[u16], usize) -> (Vec<f32>, [f32; MAX_SEATS]);

/// Answer a network batch with `ans`, row by row (a sparse batch densified).
fn answer_with<G: SearchGame>(d: &mut Driver<G>, b: &Batch, ans: &RowEval) {
    let f = G::NUM_FEATURES;
    let x = dense_x(b, f);
    let mut pri = Vec::new();
    let mut v = vec![0.0; b.b * MAX_SEATS];
    for r in 0..b.b {
        let opts: Vec<u16> = b.opt_action[b.opt_off[r] as usize..b.opt_off[r + 1] as usize].iter().map(|&a| a as u16).collect();
        let (p, val) = ans(&x[r * f..(r + 1) * f], &opts, b.n_players[r] as usize);
        pri.extend(p);
        v[r * MAX_SEATS..(r + 1) * MAX_SEATS].copy_from_slice(&val);
    }
    d.submit(b.id, &pri, &v, &vec![0.0; b.b * MAX_SEATS]).unwrap();
}

/// The finished games' rows, each with its own samples.
fn take_rows<G: SearchGame>(d: &mut Driver<G>, rows: &mut Vec<(GameResult, Vec<Record>)>) {
    let rs = d.take_results();
    let smp = d.take_samples();
    let mut a = 0;
    for r in rs {
        let k = r.samples as usize;
        rows.push((r, smp[a..a + k].to_vec()));
        a += k;
    }
    assert_eq!(a, smp.len(), "every sample belongs to a row taken with it");
}

/// Run a driver to the end, every batch answered by `ans` -> (rows with their own samples sorted by (gid, branch or
/// not), stats, the digest of every batch sent, in order).
fn run_rows<G: SearchGame>(d: &mut Driver<G>, ans: &RowEval) -> (Vec<(GameResult, Vec<Record>)>, DriverStats, u64) {
    let mut rows = Vec::new();
    let mut h: u64 = 0xcbf29ce484222325;
    while let Some(b) = d.next_batch(true).unwrap() {
        batch_digest(&mut h, &b);
        answer_with(d, &b, ans);
        take_rows(d, &mut rows);
    }
    take_rows(d, &mut rows);
    rows.sort_by_key(|(r, _)| (r.gid, r.branch.is_some()));
    (rows, d.stats(), h)
}

/// The rows' digest: each game's outcome and its own records, in gid order.
fn rows_digest(rows: &[(GameResult, Vec<Record>)]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for (r, s) in rows {
        fnv(&mut h, &r.gid.to_le_bytes());
        fnv(&mut h, &r.steps.to_le_bytes());
        fnv(&mut h, &r.decisions.to_le_bytes());
        fnv(&mut h, &[r.over as u8, r.branch.is_some() as u8, r.resign.as_ref().is_some_and(|x| x.resigned) as u8]);
        for v in &r.result {
            fnv(&mut h, &v.to_bits().to_le_bytes());
        }
        fnv(&mut h, &records_digest(s).to_le_bytes());
    }
    h
}

/// Valid self-play rows: every game over, its samples its own; a decision record's pi a distribution over its
/// options (two or more), a value-only record (no options) no pi; q a win vector in range; z the game's result
/// from the record's player's view. -> (decision records, value-only records).
fn check_rows(rows: &[(GameResult, Vec<Record>)], games: usize) -> (usize, usize) {
    assert!(rows.len() >= games, "{} rows of {games} games", rows.len());
    let (mut dec, mut val) = (0, 0);
    for (r, s) in rows {
        assert!(r.over, "game {} did not finish", r.gid);
        assert_eq!(s.len(), r.samples as usize);
        for x in s {
            let n = x.n_players as usize;
            assert_eq!(n, r.n);
            if x.opts.is_empty() {
                assert!(x.pi.is_empty());
                val += 1;
            } else {
                assert!(x.opts.len() >= 2 && x.pi.len() == x.opts.len(), "{:?} {:?}", x.opts, x.pi);
                assert!(x.pi.iter().all(|&p| (0.0..=1.0 + 1e-6).contains(&p)), "pi {:?}", x.pi);
                assert!((x.pi.iter().sum::<f32>() - 1.0).abs() < 1e-4, "pi {:?} sums to {}", x.pi, x.pi.iter().sum::<f32>());
                dec += 1;
            }
            for j in 0..n {
                assert!((-1e-6..=1.0 + 1e-6).contains(&x.q[j]), "q {:?}", x.q);
                assert_eq!(x.z[j], r.result[(x.player as usize + j) % n], "z from the record's player's view");
            }
            assert!((x.q[..n].iter().sum::<f32>() - 1.0).abs() < 1e-4, "q {:?}", x.q);
        }
    }
    (dec, val)
}

fn counter_sp_cfg(seat: SeatCfg, games: usize, spec: &str, seed: u64) -> DriverCfg {
    let mut c = cfg(vec![seat], vec![vec![0, 0]; games], &[("m", KeyKind::Nn)]);
    c.spec = spec.into();
    c.seed = seed;
    c
}

fn counter_ans(x: &[f32], opts: &[u16], _n: usize) -> (Vec<f32>, [f32; MAX_SEATS]) {
    counter_eval(x, opts)
}

/// Counter self-play of the configuration's seat on both sides, answered by `counter_eval`.
fn counter_sp(c: DriverCfg) -> (Vec<(GameResult, Vec<Record>)>, DriverStats, u64) {
    run_rows(&mut Driver::<Counter>::new(c).unwrap(), &counter_ans)
}

/// ToyGame: uniform priors, the value a hash of the features (`answer`'s, a function of the row).
fn toy_hash_ans(x: &[f32], opts: &[u16], n: usize) -> (Vec<f32>, [f32; MAX_SEATS]) {
    let k = opts.len();
    let pri = vec![1.0 / k.max(1) as f32; k];
    let h: f32 = x.iter().enumerate().map(|(i, x)| x * ((i * 7 % 13) as f32 - 6.0)).sum();
    let w = 1.0 / (1.0 + (-h * 0.1).exp());
    let mut v = [0.0; MAX_SEATS];
    v[0] = w;
    for j in 1..n {
        v[j] = (1.0 - w) / (n - 1) as f32;
    }
    (pri, v)
}

/// ToyGame: uniform priors, the value from the row player's score lead and the round (`answer_lead`'s: a lead late
/// in the game is decisive, so resignation fires).
fn toy_lead_ans(x: &[f32], opts: &[u16], _n: usize) -> (Vec<f32>, [f32; MAX_SEATS]) {
    let base = 6 + 3 + 4 + 12 + 6 + crate::toy::N_SPACES * 7;
    let k = opts.len();
    let pri = vec![1.0 / k.max(1) as f32; k];
    let lead = (x[base + 14] - x[base + 16 + 14]) * 20.0;
    let round = (0..6).find(|&i| x[i] > 0.0).unwrap_or(0) as f32 + 1.0;
    let w = 1.0 / (1.0 + (-1.5 * lead * round / 6.0).exp());
    let mut v = [0.0; MAX_SEATS];
    v[0] = w;
    v[1] = 1.0 - w;
    (pri, v)
}

fn sp_seat(sims: u32, merge: bool, share: f64) -> SeatCfg {
    let s = with_view(nn(0, sims, true), LeafView::Root, share, ViewTarget::NextQ);
    with_merge(s, merge)
}

#[test]
fn infoset_merge_selfplay_records_are_valid_and_reproduce() {
    // a merged self-play seat on both sides of the counter game, with its playout cap (fast searches; root noise on
    // the full ones), temperature and value-only samples: valid records; the same games and records across thread
    // counts and with the evaluation cache (the stand-in network is a function of the row); merges, growth leaves
    // and shared leaf encodings happen; the games or records are not unmerged self-play's
    for spec in [r#"{"rounds": 4}"#, r#"{"rounds": 5, "forget": true}"#] {
        for share in [0.0, 1.0] {
            let (r0, s0, _) = counter_sp(counter_sp_cfg(sp_seat(24, true, share), 120, spec, 13));
            let (dec, val) = check_rows(&r0, 120);
            assert_eq!(r0.len(), 120);
            assert!(dec > 200, "{spec}: {dec} decision records");
            if share > 0.0 {
                // (every decision record of a seat has the other seat's value-only record with it)
                assert_eq!(val, dec, "{spec}: value-only records");
            } else {
                assert_eq!(val, 0);
            }
            assert!(s0.merge_merged > 0 && s0.merge_growths > 0 && s0.merge_shared > 0, "{spec}: {s0:?}");
            if spec.contains("forget") {
                assert!(s0.merge_aliases > 0, "{spec}: {s0:?}");
            }
            for (threads, groups) in [(1, 1), (4, 3)] {
                let mut c = counter_sp_cfg(sp_seat(24, true, share), 120, spec, 13);
                c.threads = threads;
                c.groups = groups;
                let (r1, s1, _) = counter_sp(c);
                assert_eq!(rows_digest(&r1), rows_digest(&r0), "{spec}: {threads} threads, {groups} groups");
                assert_eq!((s1.merge_merged, s1.merge_growths, s1.merge_aliases, s1.merge_shared), (s0.merge_merged, s0.merge_growths, s0.merge_aliases, s0.merge_shared));
            }
            let (r2, s2, _) = counter_sp(counter_sp_cfg(with_cache(sp_seat(24, true, share), 256), 120, spec, 13));
            assert_eq!(rows_digest(&r2), rows_digest(&r0), "{spec}: the eval cache");
            assert!(s2.cache_hits > 0 && s2.merge_growths == s0.merge_growths);
            let (r3, s3, _) = counter_sp(counter_sp_cfg(sp_seat(24, false, share), 120, spec, 13));
            assert_ne!(rows_digest(&r3), rows_digest(&r0), "{spec}: merging changed nothing");
            assert_eq!((s3.merge_links, s3.merge_shared), (0, 0));
        }
    }
}

#[test]
fn infoset_merge_selfplay_with_gumbel_the_root_floor_and_the_belief() {
    let floor = |s: SeatCfg| match s {
        SeatCfg::Nn(mut n) => {
            n.root_min_visits = 3;
            SeatCfg::Nn(n)
        }
        s => s,
    };
    // the Gumbel root (its noise, the 'cq' target) and the root floor, merged, self-play: valid, reproducible
    for seat in [with_gumbel(sp_seat(24, true, 0.0), 4, 1.0, 1.0), floor(sp_seat(24, true, 0.0))] {
        let (r, st, _) = counter_sp(counter_sp_cfg(seat.clone(), 80, r#"{"rounds": 4}"#, 3));
        let (dec, _) = check_rows(&r, 80);
        assert!(dec > 100 && st.merge_merged > 0 && st.merge_growths > 0, "{dec} records, {st:?}");
        let mut c = counter_sp_cfg(seat, 80, r#"{"rounds": 4}"#, 3);
        c.threads = 1;
        c.groups = 1;
        assert_eq!(rows_digest(&counter_sp(c).0), rows_digest(&r));
    }
    // the root belief (the hidden-hand mulligan's auxiliary head): one belief row per full search, merged trees
    let seat = with_merge(hmull_belief_seat(32, 1.0, LeafView::Root, true), true);
    let (recs, st, vrows, brows) = hmull_belief_run(seat.clone(), 80, 5, &[0.1, 0.9]);
    assert!(!recs.is_empty() && brows > 0 && vrows > 0 && st.merge_links > 0, "{} records, {brows} belief rows, {st:?}", recs.len());
    for x in &recs {
        assert!((x.pi.iter().sum::<f32>() - 1.0).abs() < 1e-4 && (x.q[0] + x.q[1] - 1.0).abs() < 1e-4);
    }
    let (again, ..) = hmull_belief_run(seat, 80, 5, &[0.1, 0.9]);
    assert_eq!(records_set_digest(&again), records_set_digest(&recs));
    // minor decisions (`sims_minor`: their own budget, never recorded) and the playout cap of a merged self-play seat
    let mut c = cfg(vec![with_minor(sp_seat(16, true, 0.0), Some(2))], vec![vec![0, 0]; 24], &[("m", KeyKind::Nn)]);
    c.seed = 2026;
    let (rows, st, _) = run_rows(&mut Driver::<MinorToy>::new(c.clone()).unwrap(), &toy_hash_ans);
    let (dec, _) = check_rows(&rows, 24);
    assert!(dec > 0 && st.minor > 0 && st.merge_links > 0, "{dec} records, {st:?}");
    c.threads = 1;
    assert_eq!(rows_digest(&run_rows(&mut Driver::<MinorToy>::new(c).unwrap(), &toy_hash_ans).0), rows_digest(&rows));
}

#[test]
fn infoset_merge_shares_the_leaf_encoding_byte_for_byte() {
    // the driver takes the leaf's row of the viewer (a viewer's leaf's own row, another seat's leaf's value-only row)
    // from the merged tree's encoding of it (`Tree::leaf_obs`): every batch is the bytes it is when each row is
    // encoded again - dense and sparse, without and with the evaluation cache, on the counter game (two seats) and
    // the toy (chance keys, three seats); the same games and records
    for sparse in [false, true] {
        for cap in [0u32, 256] {
            let counter = |off: bool| {
                let mut c = counter_sp_cfg(with_cache(sp_seat(24, true, 0.5), cap), 60, r#"{"rounds": 4}"#, 17);
                c.sparse = sparse;
                let mut d = Driver::<Counter>::new(c).unwrap();
                d.sh.counters.share_off.store(off, Ordering::Relaxed);
                run_rows(&mut d, &counter_ans)
            };
            let toy = |off: bool| {
                let games: Vec<Vec<u16>> = (0..30).map(|g| if g % 3 == 0 { vec![0, 0, 0] } else { vec![0, 0] }).collect();
                let mut c = cfg(vec![with_cache(sp_seat(16, true, 0.5), cap)], games, &[("m", KeyKind::Nn)]);
                c.sparse = sparse;
                let mut d = Driver::<ToyGame>::new(c).unwrap();
                d.sh.counters.share_off.store(off, Ordering::Relaxed);
                run_rows(&mut d, &toy_hash_ans)
            };
            for (name, (r_on, s_on, h_on), (r_off, s_off, h_off)) in [("counter", counter(false), counter(true)), ("toy", toy(false), toy(true))] {
                let what = format!("{name}, sparse {sparse}, eval_cache {cap}");
                assert_eq!(h_on, h_off, "{what}: other batches");
                assert_eq!(rows_digest(&r_on), rows_digest(&r_off), "{what}: other games or records");
                assert_eq!((s_on.leaves, s_on.view_rows, s_on.cache_hits, s_on.merge_links), (s_off.leaves, s_off.view_rows, s_off.cache_hits, s_off.merge_links), "{what}");
                assert_eq!(s_off.merge_shared, 0, "{what}");
                // (nearly every leaf comes from a new signature, whose observation the tree encoded)
                let rows = s_on.leaves + s_on.cache_leaves;
                eprintln!("{what}: shared {} of {} leaves (+{} cached), {} merged, {} growths", s_on.merge_shared, s_on.leaves, s_on.cache_leaves, s_on.merge_merged, s_on.merge_growths);
                assert!(s_on.merge_shared > 0 && s_on.merge_shared <= rows, "{what}: {s_on:?}");
            }
        }
    }
}

#[test]
fn infoset_merge_selfplay_resigns_and_branches() {
    // self-play resignation (docs/search.md 3.14: it reads the merged search's root win vector) with every game
    // resigning where the rule fires, against the dry run (the same games played out): a resigned game is its dry
    // twin up to that point, z the resignation's
    let seat = with_merge(nn_full(16), true);
    let (on, s_on, _) = run_rows(&mut Driver::<ToyGame>::new(resign_cfg(seat.clone(), 40, 0.05, 4, 0.0)).unwrap(), &toy_lead_ans);
    let (dry, _, _) = run_rows(&mut Driver::<ToyGame>::new(resign_cfg(seat, 40, 0.05, 4, 1.0)).unwrap(), &toy_lead_ans);
    check_rows(&on, 40);
    check_rows(&dry, 40);
    assert!(s_on.merge_links > 0);
    let mut resigned = 0;
    for ((r, s), (r0, s0)) in on.iter().zip(&dry) {
        assert_eq!(r.gid, r0.gid);
        let (x, x0) = (r.resign.as_ref().unwrap(), r0.resign.as_ref().unwrap());
        let Some(w) = x0.winner else {
            assert!(!x.resigned);
            assert_eq!(records_digest(s), records_digest(s0));
            continue;
        };
        resigned += 1;
        assert!(x.resigned && x.winner == Some(w) && (x.decision, x.step, x.records) == (x0.decision, x0.step, x0.records));
        assert_eq!(s.len(), x.records as usize);
        for (a, b) in s.iter().zip(s0) {
            assert_eq!((&a.x, &a.pi, &a.opts, a.player, a.q), (&b.x, &b.pi, &b.opts, b.player, b.q));
        }
    }
    assert!((4..=38).contains(&resigned), "{resigned} of 40 games resigned");
    // branched games (docs/search.md 2.11) of resigning merged games: reproducible across thread counts
    let mut c = branch_cfg(40, 1.0);
    c.seats = vec![with_resign(with_merge(nn_full(12), true), 0.1, 3, 0.3)];
    let (rows, st, _) = run_rows(&mut Driver::<ToyGame>::new(c.clone()).unwrap(), &toy_lead_ans);
    check_rows(&rows, 40);
    let nb = rows.iter().filter(|(r, _)| r.branch.is_some()).count();
    assert!(nb >= 30 && st.merge_links > 0 && st.dev_moves > 0, "{nb} branches, {st:?}");
    c.threads = 1;
    c.groups = 1;
    let (again, _, _) = run_rows(&mut Driver::<ToyGame>::new(c).unwrap(), &toy_lead_ans);
    assert_eq!(rows_digest(&again), rows_digest(&rows));
}

#[test]
fn infoset_merge_selfplay_rolling_quotas_and_network_changes() {
    // a rolling self-play driver whose network changes every other phase (its searches restart from fresh merged
    // trees: reset + set_merge), with resignation: every phase finishes at least its quota, every game is counted
    // once, its samples its own and valid; merges and restarts happen
    for counter in [true, false] {
        let mut c = if counter {
            let mut c = counter_sp_cfg(sp_seat(16, true, 0.5), 1, r#"{"rounds": 6}"#, 21);
            c.seats = vec![with_resign(c.seats[0].clone(), 0.2, 2, 0.2)];
            c
        } else {
            resign_cfg(with_merge(nn(0, 16, true), true), 1, 0.1, 3, 0.2)
        };
        c.rolling = true;
        c.concurrency = 10;
        let ans: &RowEval = if counter { &counter_ans } else { &toy_lead_ans };
        let mut rows = Vec::new();
        let (st, games) = if counter {
            let mut d = Driver::<Counter>::new(c).unwrap();
            let games = rolling_phases_of(&mut d, ans, &mut rows);
            (d.stats(), games)
        } else {
            let mut d = Driver::<ToyGame>::new(c).unwrap();
            let games = rolling_phases_of(&mut d, ans, &mut rows);
            (d.stats(), games)
        };
        assert_eq!(st.games_finished as usize, games);
        assert_eq!(rows.len(), games);
        check_rows(&rows, games);
        let res = rows.iter().filter(|(r, _)| r.resign.as_ref().unwrap().resigned).count();
        assert!(res > 0 && res < rows.len(), "counter {counter}: {res} of {} resigned", rows.len());
        assert!(st.restarts > 0 && st.epoch == 2 && st.merge_links > 0, "counter {counter}: {st:?}");
        if counter {
            assert!(st.merge_merged > 0 && st.merge_growths > 0);
        }
    }
}

/// Six phases of a rolling driver (quota 12, the network changing every other phase) -> games finished.
fn rolling_phases_of<G: SearchGame>(d: &mut Driver<G>, ans: &RowEval, rows: &mut Vec<(GameResult, Vec<Record>)>) -> usize {
    let mut total = 0;
    for phase in 0..6u32 {
        d.resume(12, phase / 2).unwrap();
        let before = rows.len();
        loop {
            let b = d.next_batch(true).unwrap();
            if let Some(b) = &b {
                answer_with(d, b, ans);
            }
            take_rows(d, rows);
            if b.is_none() {
                break;
            }
        }
        let fin = rows.len() - before;
        assert!(fin >= 12, "phase {phase}: {fin} games");
        assert!(d.paused());
        total += fin;
    }
    total
}

/// `SearchGame::aborted`: the toy whose every end is aborted (no result). Nothing may call
/// `result()` or `margin()` on it (they panic): the driver ends each game as not over with no
/// samples, and the search evaluates an aborted terminal like a deferred one, by the network.
#[derive(Clone)]
struct AbortToy(ToyGame);

impl SearchGame for AbortToy {
    type Spec = <ToyGame as SearchGame>::Spec;
    const NAME: &'static str = "abort_toy";
    const NUM_ACTIONS: usize = ToyGame::NUM_ACTIONS;
    const NUM_FEATURES: usize = ToyGame::NUM_FEATURES;
    const MIN_PLAYERS: usize = ToyGame::MIN_PLAYERS;
    const MAX_PLAYERS: usize = ToyGame::MAX_PLAYERS;
    fn parse_spec(json: &str) -> Result<Self::Spec, String> {
        ToyGame::parse_spec(json)
    }
    fn new_game(spec: &Self::Spec, players: usize, seed: u64) -> Result<Self, String> {
        ToyGame::new_game(spec, players, seed).map(AbortToy)
    }
    fn num_players(&self) -> usize {
        self.0.num_players()
    }
    fn to_move(&self) -> Option<usize> {
        self.0.to_move()
    }
    fn legal_actions(&self, out: &mut Vec<u16>) {
        self.0.legal_actions(out)
    }
    fn step(&mut self, a: u16, chance: &mut Vec<u64>) -> Result<(), String> {
        self.0.step(a, chance)
    }
    fn result(&self) -> [f32; MAX_SEATS] {
        panic!("result() of an aborted game")
    }
    fn margin(&self, _seat: usize) -> f32 {
        panic!("margin() of an aborted game")
    }
    fn score(&self, seat: usize) -> f32 {
        self.0.score(seat)
    }
    fn determinize(&mut self, viewer: usize, rng: &mut crate::rng::Rng) {
        self.0.determinize(viewer, rng)
    }
    fn encode(&self, viewer: usize, out: &mut [f32]) {
        self.0.encode(viewer, out)
    }
    fn aborted(&self) -> bool {
        self.0.to_move().is_none()
    }
}

#[test]
fn aborted_games_end_unfinished_without_samples_and_search_never_reads_their_result() {
    let mut d = Driver::<AbortToy>::new(cfg(vec![nn(0, 16, true)], vec![vec![0, 0]; 12], &[("m", KeyKind::Nn)])).unwrap();
    let (res, smp) = run(&mut d);
    assert_eq!(res.len(), 12);
    assert!(res.iter().all(|r| r.aborted && !r.over && r.samples == 0), "{res:?}");
    assert!(smp.is_empty());
    assert!(d.finished());
    // (the default: a toy game that ends is over, not aborted)
    let mut d = Driver::<ToyGame>::new(cfg(vec![nn(0, 16, true)], vec![vec![0, 0]; 4], &[("m", KeyKind::Nn)])).unwrap();
    let (res, smp) = run(&mut d);
    assert!(res.iter().all(|r| r.over && !r.aborted) && !smp.is_empty());
}

/// The toy with its two-option decisions marked value choices (`SearchGame::value_choice`, docs/search.md 3.17).
#[derive(Clone)]
struct ChoiceToy(ToyGame);

impl SearchGame for ChoiceToy {
    type Spec = <ToyGame as SearchGame>::Spec;
    const NAME: &'static str = "choicetoy";
    const NUM_ACTIONS: usize = ToyGame::NUM_ACTIONS;
    const NUM_FEATURES: usize = ToyGame::NUM_FEATURES;
    const MIN_PLAYERS: usize = ToyGame::MIN_PLAYERS;
    const MAX_PLAYERS: usize = ToyGame::MAX_PLAYERS;
    fn parse_spec(json: &str) -> Result<Self::Spec, String> {
        ToyGame::parse_spec(json)
    }
    fn new_game(spec: &Self::Spec, players: usize, seed: u64) -> Result<Self, String> {
        ToyGame::new_game(spec, players, seed).map(ChoiceToy)
    }
    fn num_players(&self) -> usize {
        self.0.num_players()
    }
    fn to_move(&self) -> Option<usize> {
        self.0.to_move()
    }
    fn legal_actions(&self, out: &mut Vec<u16>) {
        self.0.legal_actions(out)
    }
    fn step(&mut self, a: u16, chance: &mut Vec<u64>) -> Result<(), String> {
        self.0.step(a, chance)
    }
    fn force_chance(&mut self, forced: &[u64]) {
        self.0.force_chance(forced)
    }
    fn result(&self) -> [f32; MAX_SEATS] {
        self.0.result()
    }
    fn margin(&self, seat: usize) -> f32 {
        self.0.margin(seat)
    }
    fn score(&self, seat: usize) -> f32 {
        self.0.score(seat)
    }
    fn determinize(&mut self, viewer: usize, rng: &mut crate::rng::Rng) {
        self.0.determinize(viewer, rng)
    }
    fn encode(&self, viewer: usize, out: &mut [f32]) {
        self.0.encode(viewer, out)
    }
    fn value_choice(&self) -> bool {
        let mut v = Vec::new();
        self.0.legal_actions(&mut v);
        v.len() == 2
    }
}

fn with_value_choice(seat: SeatCfg, worlds: u32, temp: f64) -> SeatCfg {
    match seat {
        SeatCfg::Nn(mut s) => {
            s.value_choice_worlds = worlds;
            s.value_choice_temp = temp;
            SeatCfg::Nn(s)
        }
        s => s,
    }
}

#[test]
fn value_choices_are_valued_by_a_one_ply_lookahead_and_never_recorded() {
    let go = |seat: SeatCfg, threads: usize| {
        let mut c = cfg(vec![seat], vec![vec![0, 0]; 24], &[("m", KeyKind::Nn)]);
        c.seed = 2026;
        c.threads = threads;
        let mut d = Driver::<ChoiceToy>::new(c).unwrap();
        let (r, s) = run(&mut d);
        (r, s, d.stats())
    };
    // off (the default): the hook is never consulted - the toy's golden self-play records
    let (_, recs, st) = go(nn(0, 16, true), 2);
    assert_eq!((recs.len(), records_digest(&recs)), GOLDEN_RECORDS);
    assert_eq!((st.value_choices, st.value_choice_rows), (0, 0));
    // and a game without value choices is unchanged by the setting
    let (_, recs) = golden_selfplay(with_value_choice(nn(0, 16, true), 4, 0.05));
    assert_eq!((recs.len(), records_digest(&recs)), GOLDEN_RECORDS);
    // on: the two-option decisions are valued (rows from the lookahead), chosen, never recorded
    let (res, recs, st) = go(with_value_choice(nn(0, 16, true), 4, 0.0), 2);
    assert_eq!(res.len(), 24);
    assert!(res.iter().all(|r| r.over));
    assert!(st.value_choices > 0 && st.value_choice_rows > 0, "{} choices, {} rows", st.value_choices, st.value_choice_rows);
    assert!(st.value_choice_rows <= st.value_choices * 2 * 4, "at most options x worlds rows a choice");
    assert!(!recs.is_empty() && recs.iter().all(|r| r.opts.len() != 2), "a value choice was recorded");
    // reproducible whatever the threads
    let (res1, recs1, _) = go(with_value_choice(nn(0, 16, true), 4, 0.0), 1);
    assert_eq!(results_digest(&res1), results_digest(&res));
    assert_eq!(records_digest(&recs1), records_digest(&recs));
    // a temperature changes the moves, not the machinery; evaluation seats too
    let (res_t, _, st_t) = go(with_value_choice(nn(0, 16, true), 4, 0.05), 2);
    assert!(res_t.iter().all(|r| r.over) && st_t.value_choices > 0);
    let (res_e, recs_e, st_e) = go(with_value_choice(nn(0, 16, false), 2, 0.0), 2);
    assert!(res_e.iter().all(|r| r.over) && recs_e.is_empty() && st_e.value_choices > 0);
    // the checks
    let bad = |seat: SeatCfg| Driver::<ChoiceToy>::new(cfg(vec![seat], vec![vec![0, 0]], &[("m", KeyKind::Nn)])).is_err();
    assert!(bad(with_value_choice(nn(0, 8, true), VALUE_CHOICE_MAX_WORLDS + 1, 0.0)));
    assert!(bad(with_value_choice(nn(0, 8, true), 4, -1.0)));
    assert!(bad(with_value_choice(nn(0, 8, true), 4, f64::NAN)));
    let mut look = NnSeat::eval(0, 8);
    look.look = Some(Box::new(LookCfg::default()));
    look.value_choice_worlds = 2;
    assert!(bad(SeatCfg::Nn(look)));
}

#[test]
fn value_choice_pick_takes_the_best_or_draws_by_the_softmax() {
    let opts = [(5u16, 0.5), (7, 0.6), (9, 0.6)];
    let mut rng = Rng::new(3);
    assert_eq!(value_choice_pick(&opts, 0.0, &mut rng).unwrap(), 7, "the first best");
    assert!(value_choice_pick(&[], 0.0, &mut rng).is_err());
    let n = 40_000;
    let mut c = [0u32; 3];
    for _ in 0..n {
        let a = value_choice_pick(&opts, 0.1, &mut rng).unwrap();
        c[opts.iter().position(|o| o.0 == a).unwrap()] += 1;
    }
    let w = [(-1.0f64).exp(), 1.0, 1.0];
    let s: f64 = w.iter().sum();
    for i in 0..3 {
        let p = w[i] / s;
        let f = c[i] as f64 / n as f64;
        assert!((f - p).abs() < 0.01, "option {i}: {f} against {p}");
    }
}

// ------------------------------------------------------------------ a rolling driver's first game id (2.14)

/// (gid, steps, result, records) of every game a rolling queue_only driver plays for `n` queued setups.
fn rolling_games(first: Option<u32>, n: usize) -> Vec<(u32, u64, Vec<f32>, Vec<(Vec<f32>, Vec<u16>, Vec<f32>, [f32; MAX_SEATS])>)> {
    let mut c = cfg(vec![nn(0, 8, true)], vec![vec![0, 0]], &[("m", KeyKind::Nn)]);
    c.rolling = true;
    c.queue_only = true;
    c.concurrency = 3;
    let mut d = Driver::<ToyGame>::new(c).unwrap();
    if let Some(g) = first {
        d.set_first_gid(g).unwrap();
    }
    d.resume(n as u64, 0).unwrap();
    for k in 0..n {
        d.enqueue(&format!("s{k}"), "{}").unwrap();
    }
    let mut out = Vec::new();
    let mut take = |d: &mut Driver<ToyGame>| {
        let res = d.take_results();
        let mut recs = d.take_samples().into_iter();
        for r in res {
            let rr: Vec<_> = recs.by_ref().take(r.samples as usize).map(|x| (x.x, x.opts, x.pi, x.q)).collect();
            out.push((r.gid, r.steps, r.result, rr));
        }
        assert!(recs.next().is_none());
    };
    while let Some(b) = d.next_batch(true).unwrap() {
        answer(&mut d, &b);
        take(&mut d);
    }
    take(&mut d);
    out.sort_by_key(|g| g.0);
    out
}

#[test]
fn first_gid_plays_the_same_games() {
    let all = rolling_games(None, 7);
    assert_eq!(all.iter().map(|g| g.0).collect::<Vec<_>>(), (0..7).collect::<Vec<u32>>());
    let tail = rolling_games(Some(4), 3);
    assert_eq!(tail.iter().map(|g| g.0).collect::<Vec<_>>(), vec![4, 5, 6]);
    assert!(all[4..].iter().zip(&tail).all(|(a, b)| a == b), "games 4..7 of a driver from 0 == a driver from 4");
    assert!(all[4..].iter().any(|g| !g.3.is_empty()), "the games recorded samples");
    // only a rolling driver, only before its first resume
    let mut fixed = Driver::<ToyGame>::new(cfg(vec![nn(0, 4, true)], vec![vec![0, 0]], &[("m", KeyKind::Nn)])).unwrap();
    assert!(fixed.set_first_gid(3).is_err());
    let mut c = cfg(vec![nn(0, 4, true)], vec![vec![0, 0]], &[("m", KeyKind::Nn)]);
    c.rolling = true;
    let mut d = Driver::<ToyGame>::new(c).unwrap();
    d.resume(1, 0).unwrap();
    assert!(d.set_first_gid(3).is_err());
    while let Some(b) = d.next_batch(true).unwrap() {
        answer(&mut d, &b);
    }
}
