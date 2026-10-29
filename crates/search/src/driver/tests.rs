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
    // (in either order: a queue_only driver serves whichever group is ready first - on a loaded
    // machine group 1's first search can finish before group 0's)
    let pair = |a: &[(usize, usize)]| {
        let mut v = a.to_vec();
        v.sort();
        v
    };
    assert_eq!(pair(&sizes[..2]), vec![(0, 1), (1, 1)], "one setup each as they arrive: {sizes:?}");
    assert_eq!(pair(&sizes[2..4]), vec![(0, 4), (1, 4)], "then 4 and 4: {sizes:?}");
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
