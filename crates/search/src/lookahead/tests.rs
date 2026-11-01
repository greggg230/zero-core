use super::*;
use crate::driver::{Driver, DriverCfg, KeyKind, NnSeat, SeatCfg, ROW_BELIEF};
use crate::peek_game::*;

/// The rows of one round (`RowSink`): features, options, viewer, kind.
#[derive(Default)]
struct Rows {
    x: Vec<f32>,
    opts: Vec<Vec<u16>>,
    viewer: Vec<usize>,
    kind: Vec<u8>,
}

impl<G: SearchGame> RowSink<G> for Rows {
    fn push_row(&mut self, viewer: usize, opts: &[u16], kind: u8) -> &mut [f32] {
        self.opts.push(opts.to_vec());
        self.viewer.push(viewer);
        self.kind.push(kind);
        let off = self.x.len();
        self.x.resize(off + G::NUM_FEATURES, 0.0);
        &mut self.x[off..]
    }
    fn pop_row(&mut self) {
        self.opts.pop();
        self.viewer.pop();
        self.kind.pop();
        self.x.truncate(self.x.len() - G::NUM_FEATURES);
    }
}

/// The belief `peek_eval`'s stand-in answers for a belief row: coin 1 at 0.7.
const BELIEF: [f32; 2] = [0.3, 0.7];

/// Run a lookahead to its decision against `peek_eval` (-> the rows of every round, for checks).
fn solve(look: &mut Look<Peek>, rng: &mut Rng) -> Vec<Rows> {
    let mut rounds = Vec::new();
    loop {
        let mut rows = Rows::default();
        match look.step(&mut rows, rng).unwrap() {
            Step::Done => return rounds,
            Step::Rows(n) => {
                assert_eq!(n, rows.kind.len());
                let mut opt_off = vec![0i32];
                let (mut pri, mut val, mut aux, mut aux_row) = (Vec::new(), Vec::new(), Vec::new(), vec![u32::MAX; n]);
                for i in 0..n {
                    let (p, v) = peek_eval(&rows.x[i * 8..(i + 1) * 8], rows.opts[i].len());
                    pri.extend(p);
                    opt_off.push(pri.len() as i32);
                    val.extend(v);
                    if rows.kind[i] == ROW_BELIEF {
                        aux_row[i] = (aux.len() / 2) as u32;
                        aux.extend(BELIEF);
                    }
                }
                look.answer(&Answers { base: 0, opt_off: &opt_off, priors: &pri, value: &val, aux: &aux, aux_row: &aux_row, aux_width: 2 }).unwrap();
                rounds.push(rows);
            }
        }
    }
}

fn cfg(s: &str) -> LookCfg {
    LookCfg::parse(s).unwrap()
}

/// The two worlds of seat 0's root: coin 0 and coin 1.
fn coins() -> Vec<Peek> {
    vec![Peek::with_coin(0), Peek::with_coin(1)]
}

/// The lookahead of the root with both coins as its worlds -> (PEEK's mean, GO's mean, the decision, it).
fn look(settings: &str) -> (f64, f64, Outcome, Look<Peek>) {
    let mut l = Look::new(cfg(settings), &Peek::with_coin(0), 0, None, coins(), Vec::new(), true).unwrap();
    solve(&mut l, &mut Rng::new(0));
    let o = l.options().unwrap();
    assert_eq!(o.iter().map(|x| x.act).collect::<Vec<_>>(), vec![PEEK, GO], "the root's options in their order");
    (o[0].mean, o[1].mean, l.outcome().unwrap(), l)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

#[test]
fn looking_is_worth_something_per_information_set_and_nothing_per_world() {
    // per information set: after GO the guess is one choice for both coins (1/2 of the payoff;
    // seat 1 blocks: min); after PEEK the coin splits the groups
    let (peek, go, d, mut l) = look(r#"{"plies": 3, "opp": "min"}"#);
    assert!(close(peek, 0.72) && close(go, 0.4), "infoset: PEEK {peek} GO {go}");
    assert_eq!(d, Outcome::Act(PEEK));
    let o = l.options().unwrap();
    // GO per world: coin 0 guessed right (blocked: 0.8), coin 1 wrong (0)
    assert!(close(o[1].values[0].1, 0.8) && close(o[1].values[1].1, 0.0), "{:?}", o[1].values);
    assert!(close(o[1].sd, (0.32f64).sqrt()), "sd {}", o[1].sd);
    // per world (perfect information): the guess is right anyway, looking costs 0.1
    let (peek, go, d, _) = look(r#"{"plies": 3, "opp": "min", "mode": "world"}"#);
    assert!(close(peek, 0.72) && close(go, 0.8), "world: PEEK {peek} GO {go}");
    assert_eq!(d, Outcome::Act(GO));
}

#[test]
fn opponent_models() {
    // expect: BLOCK 0.25 / PASS 0.75; top: PASS only
    let (peek, go, _, _) = look(r#"{"plies": 3, "opp": "expect"}"#);
    assert!(close(peek, 0.25 * 0.72 + 0.75 * 0.9) && close(go, 0.5 * (0.25 * 0.8 + 0.75 * 1.0)), "expect: PEEK {peek} GO {go}");
    let (peek, go, _, _) = look(r#"{"plies": 3, "opp": "top"}"#);
    assert!(close(peek, 0.9) && close(go, 0.5), "top: PEEK {peek} GO {go}");
    // the aliases
    assert_eq!(cfg(r#"{"opp": "best"}"#).opp, Opp::Min);
    assert_eq!(cfg(r#"{"opp": "policy"}"#).opp, Opp::Expect);
    // top-1 below the root under min: the opponent's top-prior reply (PASS) only, and our guess its
    // top prior (uniform: GUESS0, right with coin 0 only)
    let (peek, go, _, _) = look(r#"{"plies": 3, "opp": "min", "topk": 1}"#);
    assert!(close(peek, 0.45) && close(go, 0.5), "min top-1: PEEK {peek} GO {go}");
}

#[test]
fn the_minor_payment_takes_the_top_prior_and_counts_no_ply() {
    // one ply: our move, then the payment (TAP_B, prior 0.6) stepped, then the leaf at seat 1's
    // decision, valued from seat 0's view (GO: half the payoff; PEEK: the payoff)
    let (peek, go, _, mut l) = look(r#"{"plies": 1}"#);
    assert!(close(peek, 0.9) && close(go, 0.5), "PEEK {peek} GO {go}");
    let s = l.stats().clone();
    // (the two worlds after GO look the same to seat 0: one value row for both, `dedup`)
    assert_eq!((s.nodes, s.leaves, s.minor_steps, s.value_rows, s.value_dups, s.prior_rows, s.groups), (4, 4, 4, 3, 1, 4, 1));
    let (peek2, go2, _, l2) = look(r#"{"plies": 1, "dedup": false}"#);
    assert_eq!((l2.stats().value_rows, l2.stats().value_dups), (4, 0));
    assert!(peek2 == peek && go2 == go, "the same values without dedup");
    let (line, end) = l.line(1, GO).unwrap().unwrap();
    assert_eq!(line, vec![(0, GO, false), (0, TAP_B, true)]);
    assert_eq!(end, End::Plies);
}

#[test]
fn horizons() {
    // to the end of the turn: the leaf at seat 1's reply (turn 2)
    let (peek, go, _, mut l) = look(r#"{"horizon": "turn"}"#);
    assert!(close(peek, 0.9) && close(go, 0.5), "turn: PEEK {peek} GO {go}");
    assert_eq!(l.line(0, PEEK).unwrap().unwrap().1, End::Horizon);
    // to the start of seat 0's next turn: the leaf at the guess (turn 3), seat 1's reply looked at
    let (peek, go, _, mut l) = look(r#"{"horizon": "next_turn", "opp": "min"}"#);
    assert!(close(peek, 0.72) && close(go, 0.4), "next_turn: PEEK {peek} GO {go}");
    let (line, end) = l.line(0, PEEK).unwrap().unwrap();
    assert_eq!(line, vec![(0, PEEK, false), (0, TAP_B, true), (1, BLOCK, false)]);
    assert_eq!(end, End::Horizon);
    // a cap below the horizon: cut (counted)
    let (_, _, _, l) = look(r#"{"horizon": "next_turn", "max_plies": 1}"#);
    assert_eq!(l.stats().cut, 4);
    // a game without turns refuses next_turn
    let m = crate::mull_game::Mull::with_hands(1, 0, 3);
    assert!(Look::new(cfg(r#"{"horizon": "next_turn"}"#), &m, 0, None, Vec::new(), vec![1; 8], false).is_err());
}

#[test]
fn the_budget_falls_back_by_stages_then_to_mcts_or_the_prior() {
    // 3 plies need 28 nodes: over 12 the configured stage and the top-k stage are abandoned (at 14
    // nodes each: the root's 4, the replies' 8, the first guess group's 2) and 2 plies decide
    let (peek, go, d, l) = look(r#"{"plies": 3, "budget": 12}"#);
    let s = l.stats();
    assert_eq!((s.stage, s.last, s.nodes, s.aborted_nodes), (2, false, 12, 28), "{s:?}");
    assert_eq!(l.cfg().plies, 2);
    assert!(close(peek, 0.72) && close(go, 0.4), "2 plies: PEEK {peek} GO {go}");
    assert_eq!(d, Outcome::Act(PEEK));
    assert_eq!(cfg(r#"{"plies": 3, "budget": 12}"#).stages().len(), 3);
    // below the root's 4 nodes every stage is abandoned: the seat's MCTS
    let mut l = Look::new(cfg(r#"{"plies": 3, "budget": 3}"#), &Peek::with_coin(0), 0, None, coins(), Vec::new(), false).unwrap();
    solve(&mut l, &mut Rng::new(0));
    assert_eq!(l.outcome(), Some(Outcome::Mcts));
    assert!(l.stats().last && l.stats().stage == 3 && l.stats().aborted_nodes == 12);
    // ... or the network's top prior (uniform: the first option)
    let mut l = Look::new(cfg(r#"{"plies": 3, "budget": 3, "fallback": "prior"}"#), &Peek::with_coin(0), 0, None, coins(), Vec::new(), false).unwrap();
    let rounds = solve(&mut l, &mut Rng::new(0));
    assert_eq!(l.outcome(), Some(Outcome::Act(PEEK)));
    assert_eq!(rounds.last().unwrap().opts, vec![vec![PEEK, GO]]);
    // an explicit stage list
    let st = cfg(r#"{"plies": 6, "budget": 100, "fallback_topk": 2, "fallback_plies": [4, 2]}"#).stages();
    assert_eq!(st.iter().map(|c| (c.plies, c.topk)).collect::<Vec<_>>(), vec![(6, 0), (6, 2), (4, 2), (2, 2)]);
}

#[test]
fn bad_settings_are_errors() {
    for s in [r#"{"plys": 3}"#, r#"{"plies": 0}"#, r#"{"opp": "max"}"#, r#"{"mode": "perfect"}"#, r#"{"topk_root": true}"#, r#"{"worlds": 0}"#, r#"{"rows": 0}"#, r#"{"plies": "6"}"#, r#"{"fallback_plies": [4]}"#, r#"[1]"#, r#"{"horizon": "game"}"#] {
        assert!(LookCfg::parse(s).is_err(), "{s} accepted");
    }
    let c = cfg(r#"{"plies": 6, "worlds": 4, "topk": 3, "topk_root": true, "opp": "top", "mode": "world", "budget": 1000, "fallback": "prior", "rows": 64}"#);
    assert_eq!(LookCfg::parse(&c.to_json()).unwrap(), c, "to_json reads back");
}

#[test]
fn the_root_can_be_capped_and_top_k_follows_the_prior() {
    // topk 1 with topk_root: the root's first option by prior (uniform: PEEK) only
    let mut l = Look::new(cfg(r#"{"plies": 3, "topk": 1, "topk_root": true}"#), &Peek::with_coin(0), 0, None, coins(), Vec::new(), false).unwrap();
    solve(&mut l, &mut Rng::new(0));
    assert_eq!(l.options().unwrap().iter().map(|o| o.act).collect::<Vec<_>>(), vec![PEEK]);
    // topk_from_turn: every option before that turn, top-k from it - from turn 2 seat 1's reply (PASS)
    // and our guess (GUESS0) are capped, from turn 3 only the guess (seat 1 blocks: min)
    let (peek, go, _, _) = look(r#"{"plies": 3, "topk": 1, "topk_from_turn": 2}"#);
    assert!(close(peek, 0.45) && close(go, 0.5), "from turn 2: PEEK {peek} GO {go}");
    let (peek, go, _, _) = look(r#"{"plies": 3, "topk": 1, "topk_from_turn": 3}"#);
    assert!(close(peek, 0.36) && close(go, 0.4), "from turn 3: PEEK {peek} GO {go}");
}

#[test]
fn worlds_are_dealt_from_seeds_reproducibly_with_the_belief() {
    let run = |seeds: Vec<u64>, belief: Option<(f64, f64, f64)>| {
        let mut l = Look::new(cfg(r#"{"plies": 3, "worlds": 6}"#), &Peek::with_coin(0), 0, belief, Vec::new(), seeds, false).unwrap();
        let rounds = solve(&mut l, &mut Rng::new(99));
        let coins: Vec<u8> = l.worlds().iter().map(|w| w.coin).collect();
        let o: Vec<(u16, Vec<(u16, f64)>)> = l.options().unwrap().into_iter().map(|o| (o.act, o.values)).collect();
        (coins, o, rounds.first().map(|r| r.kind.clone()), l.belief().map(|b| b.pred.clone()))
    };
    let a = run(vec![1, 2, 3, 4, 5, 6], None);
    assert_eq!(a, run(vec![1, 2, 3, 4, 5, 6], None), "the same seeds, the same lookahead");
    assert_ne!(a.0, run(vec![7, 8, 9, 10, 11, 12], None).0);
    // the belief: one root belief row first, the worlds dealt by it
    let b = run(vec![1, 2, 3, 4, 5, 6], Some((1.0, 20.0, 0.0)));
    assert_eq!(b.2, Some(vec![ROW_BELIEF]));
    assert_eq!(b.3, Some(BELIEF.to_vec()));
    assert_eq!(b, run(vec![1, 2, 3, 4, 5, 6], Some((1.0, 20.0, 0.0))));
    // the belief's weights: coin 1 dealt 7 in 10 (P = w1 / (w0 + w1) = 0.7)
    let seeds: Vec<u64> = (0..400).collect();
    let mut ones = 0;
    for chunk in seeds.chunks(8) {
        let mut l = Look::new(cfg(r#"{"plies": 1, "worlds": 8}"#), &Peek::with_coin(0), 0, Some((1.0, 20.0, 0.0)), Vec::new(), chunk.to_vec(), false).unwrap();
        solve(&mut l, &mut Rng::new(0));
        ones += l.worlds().iter().filter(|w| w.coin == 1).count();
    }
    assert!((240..=320).contains(&ones), "{ones} of 400 coins 1");
}

#[test]
fn obs_hash_tells_encodings_apart() {
    let mut x = vec![0.0f32; 37];
    let h0 = obs_hash(&x);
    x[36] = 1.0;
    let h1 = obs_hash(&x);
    x[36] = -0.0;
    let h2 = obs_hash(&x);
    assert!(h0 != h1 && h0 != h2 && h1 != h2, "the float bits, -0.0 apart from 0.0 (the lab hashes the bytes)");
    assert_eq!(obs_hash(&[0.5, 1.0, 2.0]), obs_hash(&[0.5, 1.0, 2.0]));
    assert_ne!(obs_hash(&[1.0, 2.0]), obs_hash(&[2.0, 1.0]));
}

// ------------------------------------------------------------------ the driver's lookahead seats

fn look_seat(settings: &str, sims: u32) -> SeatCfg {
    let mut s = NnSeat::eval(0, sims);
    s.look = Some(Box::new(cfg(settings)));
    SeatCfg::Nn(s)
}

fn driver_cfg(seats: Vec<SeatCfg>, games: usize, threads: usize) -> DriverCfg {
    DriverCfg {
        spec: "{}".into(),
        seats,
        games: (0..games).map(|g| if g % 2 == 0 { vec![0, 1] } else { vec![1, 0] }).collect(),
        keys: vec![("m".into(), KeyKind::Nn)],
        concurrency: 8,
        seed: 11,
        record: false,
        threads,
        groups: 2,
        max_steps: 1000,
        rolling: false,
        net: 0,
        queue_only: false,
        move_log: 20,
        aux: false,
        paired: false,
    }
}

/// Run a driver of the peek game to the end against `peek_eval` (belief rows: `BELIEF`).
fn run_peek(d: &mut Driver<Peek>) -> Vec<crate::driver::GameResult> {
    let mut res = Vec::new();
    while let Some(b) = d.next_batch(true).unwrap() {
        let (mut pri, mut val, mut aux) = (Vec::new(), vec![0.0; b.b * MAX_SEATS], Vec::new());
        for r in 0..b.b {
            let k = (b.opt_off[r + 1] - b.opt_off[r]) as usize;
            let (p, v) = peek_eval(&b.x[r * 8..(r + 1) * 8], k);
            pri.extend(p);
            val[r * MAX_SEATS..(r + 1) * MAX_SEATS].copy_from_slice(&v);
            if b.extra[r] == ROW_BELIEF {
                aux.extend(BELIEF);
            }
        }
        let m = vec![0.0; b.b * MAX_SEATS];
        d.submit_aux(b.id, &pri, &val, &m, &aux).unwrap();
        res.extend(d.take_results());
    }
    res.extend(d.take_results());
    res.sort_by_key(|r| r.gid);
    res
}

#[test]
fn a_lookahead_seat_plays_in_the_driver_and_peeks() {
    let run = |threads: usize| {
        let mut d = Driver::<Peek>::new(driver_cfg(vec![look_seat(r#"{"plies": 3, "worlds": 8}"#, 16), SeatCfg::Nn(NnSeat::eval(0, 16))], 12, threads)).unwrap();
        let res = run_peek(&mut d);
        (res, d.stats())
    };
    let (res, st) = run(2);
    assert_eq!(res.len(), 12);
    for r in &res {
        assert!(r.over);
        if r.gid % 2 == 0 {
            // the lookahead is seat 0: it peeks (its first move), then guesses the coin it saw
            assert_eq!(r.moves[0].1, PEEK, "game {}: {:?}", r.gid, r.moves);
            assert!(r.result[0] > 0.7, "game {}: {:?}", r.gid, r.result);
        }
    }
    // its decisions: the root (8 worlds of 2 options) and the guess (its worlds' coin seen) as seat 0,
    // the reply as seat 1; the payments are not its decisions (minor: the top prior, no lookahead)
    assert!(st.look.decisions >= 12 * 2 && st.look.minor_root == 6, "{:?}", st.look);
    assert_eq!(st.look.fallback_last, 0);
    assert!(st.look.rows > 0 && st.look.nodes > 0);
    // reproducible whatever the threads
    let (res4, st4) = run(4);
    assert_eq!(format!("{res:?}"), format!("{res4:?}"));
    assert_eq!((st.look.decisions, st.look.nodes, st.look.rows), (st4.look.decisions, st4.look.nodes, st4.look.rows));
}

#[test]
fn a_lookahead_seat_over_its_budget_searches_with_mcts_and_takes_the_belief() {
    let mut s = NnSeat::eval(0, 16);
    s.look = Some(Box::new(cfg(r#"{"plies": 3, "budget": 3}"#)));
    s.belief_tilt = 1.0;
    let mut d = Driver::<Peek>::new(driver_cfg(vec![SeatCfg::Nn(s), SeatCfg::Nn(NnSeat::eval(0, 16))], 6, 2)).unwrap();
    let res = run_peek(&mut d);
    assert_eq!(res.len(), 6);
    let st = d.stats();
    // seat 0's root: 2 options x 8 worlds = 16 nodes > 3 at every stage: MCTS
    assert!(st.look.fallback_mcts >= 3 && st.look.fallback_mcts == st.look.fallback_last, "{:?}", st.look);
    assert!(st.sims > 0, "the MCTS fallback searched");
    // a belief row per lookahead that dealt worlds (none for the payments)
    assert!(st.look.rows >= st.look.decisions - st.look.minor_root);
    // a self-play lookahead seat is refused
    let mut s = NnSeat::eval(0, 16);
    s.look = Some(Box::new(cfg(r#"{"plies": 2}"#)));
    s.selfplay = true;
    assert!(Driver::<Peek>::new(driver_cfg(vec![SeatCfg::Nn(s), SeatCfg::Nn(NnSeat::eval(0, 16))], 2, 1)).is_err());
}
