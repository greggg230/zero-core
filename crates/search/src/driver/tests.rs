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
        let s = with_target(nn(0, 16, true), PolicyTarget::CompletedQ { c_visit: 50.0, c_scale: target });
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
    let cq = PolicyTarget::CompletedQ { c_visit: 50.0, c_scale: 1.0 };
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
