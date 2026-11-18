//! The playout cap's rule by the prior's uncertainty (`p_full_rule` 'prior_uncertainty', docs/search.md 3.21): a
//! floor equal to `p_full` is the coin of before bit for bit (the coin is drawn where the random rule draws it, the
//! fresh root's noise and belief come after its evaluation with the same values); the rule's share of full searches
//! matches `p_full` when `p_full_u_mean` is the mean of g(u); its full searches go to the unsure decisions; the games
//! are the same at any thread / group layout; the settings are checked.
use super::*;

fn with_rule(seat: SeatCfg, rule: FullRule) -> SeatCfg {
    match seat {
        SeatCfg::Nn(mut s) => {
            s.p_full_rule = rule;
            SeatCfg::Nn(s)
        }
        s => s,
    }
}

fn prior_rule(floor: f64, shape: UShape, u_mean: f64) -> FullRule {
    FullRule::PriorUncertainty { floor, shape, u_mean }
}

/// Self-play of 48 games under skewed priors (`answer_skewed`) at the given thread / group layout.
fn skewed_run(seat: SeatCfg, threads: usize, groups: usize) -> (Vec<GameResult>, Vec<Record>, DriverStats) {
    let mut c = cfg(vec![seat], vec![vec![0, 0]; 48], &[("m", KeyKind::Nn)]);
    c.seed = 2026;
    (c.threads, c.groups) = (threads, groups);
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
    let st = d.stats();
    (res, smp, st)
}

fn game_keys(res: &[GameResult]) -> Vec<String> {
    res.iter().map(|r| format!("{:?}", (r.gid, r.steps, &r.scores, r.samples, r.decisions))).collect()
}

#[test]
fn a_floor_at_p_full_is_the_coin_bit_for_bit() {
    // floor = p_full: f = p_full + 0 * g(u) / u_mean for every decision - the coin's threshold, compared with the coin
    // drawn at the same stream position; the fresh root's noise is added after its evaluation instead of with it
    // (the same noise stream, the same values), so the games, the records and the search's counters are the same
    let base = nn(0, 16, true);
    let (r0, s0, t0) = skewed_run(base.clone(), 2, 2);
    for shape in [UShape::Linear, UShape::Sqrt, UShape::Square] {
        let (r1, s1, t1) = skewed_run(with_rule(base.clone(), prior_rule(0.5, shape, 0.123)), 2, 2);
        assert_eq!(game_keys(&r0), game_keys(&r1));
        assert_eq!((s0.len(), records_digest(&s0)), (s1.len(), records_digest(&s1)), "{shape:?}");
        assert_eq!((t0.sims, t0.leaves, t0.recorded, t0.cap_decisions, t0.cap_full, t0.cap_sims), (t1.sims, t1.leaves, t1.recorded, t1.cap_decisions, t1.cap_full, t1.cap_sims));
        assert_eq!(t0.cap_f_e6, t1.cap_f_e6, "the chance is p_full for every decision");
    }
    // the golden records of the default seat are untouched (and the counters do not move the search)
    let (_, recs) = golden_selfplay(nn(0, 16, true));
    assert_eq!((recs.len(), records_digest(&recs)), GOLDEN_RECORDS);
}

#[test]
fn a_floor_at_p_full_is_the_coin_with_the_root_belief_too() {
    // a fresh root's belief row goes alone after its evaluation (the budget is not known before), with the same
    // answer: the same games and records as the coin's, the rows' layout (and so the order the games end in) aside
    let uni = vec![1.0 / 6.0; 6];
    let play = |seat: SeatCfg| {
        let mut c = cfg(vec![with_belief(seat, 1.0, 20.0, 0.0, false)], vec![vec![0, 0]; 24], &[("m", KeyKind::Nn)]);
        c.seed = 2026;
        let mut d = Driver::<ToyGame>::new(c).unwrap();
        let (res, recs, rows) = run_belief(&mut d, &uni);
        (res, recs, rows, d.stats())
    };
    let (r0, s0, (w0, a0), t0) = play(nn(0, 16, true));
    let (r1, s1, (w1, a1), t1) = play(with_rule(nn(0, 16, true), prior_rule(0.5, UShape::Linear, 0.4)));
    assert_eq!(game_keys(&r0), game_keys(&r1));
    // (the records as a multiset: the belief rows sent alone take more rounds, so the games end in another order)
    let sorted = |s: &[Record]| {
        let mut v: Vec<u64> = s.iter().map(|r| records_digest(std::slice::from_ref(r))).collect();
        v.sort_unstable();
        v
    };
    assert_eq!(sorted(&s0), sorted(&s1));
    assert_eq!((t0.belief_rows, t0.recorded), (t1.belief_rows, t1.recorded));
    assert_eq!(w0 + a0, w1 + a1);
    assert!(w0 > 0 && w1 == 0, "the rule's belief rows all go alone ({w1} after a leaf)");
}

#[test]
fn the_rule_matches_p_full_and_moves_the_full_searches_to_unsure_decisions() {
    // the random rule's counters give the mean of g(u) of these games (the network and the games are fixed); the
    // rule with that mean spends p_full in expectation - its mean chance is p_full but for the cap at 1 - and the
    // full searches it draws match their chance; the unsure decisions get them: the full searches' targets move off
    // the prior's argmax more often than the coin's
    let base = nn(0, 16, true);
    let (_, s0, t0) = skewed_run(base.clone(), 2, 2);
    let n0 = t0.cap_decisions as f64;
    assert!(t0.cap_decisions > 1000, "{} capped decisions", t0.cap_decisions);
    let share0 = t0.cap_full as f64 / n0;
    assert!((share0 - 0.5).abs() < 0.05, "the coin's share {share0}");
    assert_eq!(t0.cap_targets, s0.len() as u64);
    for shape in [UShape::Linear, UShape::Sqrt, UShape::Square] {
        // the shape's mean g(u) under the coin's games (a floor at p_full plays them; its counters read g)
        let (_, _, tg) = skewed_run(with_rule(base.clone(), prior_rule(0.5, shape, 1.0)), 2, 2);
        let g_mean = tg.cap_g_e6 as f64 * 1e-6 / tg.cap_decisions as f64;
        assert!(g_mean > 0.0 && g_mean < 1.0, "{shape:?} mean g {g_mean}");
        let floor = 0.1;
        let (res, recs, t1) = skewed_run(with_rule(base.clone(), prior_rule(floor, shape, g_mean)), 2, 2);
        assert_eq!(res.len(), 48);
        assert!(res.iter().all(|r| r.over));
        let n1 = t1.cap_decisions as f64;
        let (share1, f1) = (t1.cap_full as f64 / n1, t1.cap_f_e6 as f64 * 1e-6 / n1);
        // the drawn share matches the decisions' chances (a binomial: 4 sd), and their mean is p_full to the games'
        // drift (the rule plays other games than the coin: g's mean moves with them) and the cap at 1
        let sd = (0.25 / n1).sqrt();
        assert!((share1 - f1).abs() < 4.0 * sd, "{shape:?}: drawn {share1} vs chances {f1}");
        assert!((f1 - 0.5).abs() < 0.06, "{shape:?}: mean chance {f1}");
        assert!(f1 >= floor && t1.cap_g_e6 > 0);
        assert_eq!(t1.cap_sims, t1.cap_full * 16 + (t1.cap_decisions - t1.cap_full) * 4, "budgets: 16 full, 4 fast");
        assert_eq!(t1.cap_targets, recs.len() as u64);
        let (ch0, ch1) = (t0.cap_changed as f64 / t0.cap_targets as f64, t1.cap_changed as f64 / t1.cap_targets as f64);
        assert!(ch1 > ch0, "{shape:?}: targets off the prior's argmax {ch1} (rule) vs {ch0} (coin)");
    }
}

#[test]
fn the_rule_plays_the_same_games_at_any_thread_layout() {
    // the coin comes from the game's stream and the chance from its root's priors: a game's budgets do not depend on
    // the threads, groups or batch composition
    let seat = with_rule(nn(0, 16, true), prior_rule(0.05, UShape::Linear, 0.3));
    let (r1, s1, t1) = skewed_run(seat.clone(), 1, 1);
    let (r4, s4, t4) = skewed_run(seat, 4, 3);
    assert_eq!(game_keys(&r1), game_keys(&r4));
    assert_eq!((s1.len(), records_digest(&s1)), (s4.len(), records_digest(&s4)));
    assert_eq!((t1.cap_decisions, t1.cap_full, t1.cap_sims, t1.cap_g_e6, t1.cap_f_e6, t1.cap_changed), (t4.cap_decisions, t4.cap_full, t4.cap_sims, t4.cap_g_e6, t4.cap_f_e6, t4.cap_changed));
    // and they differ from the coin's
    let (_, s0, _) = skewed_run(nn(0, 16, true), 1, 1);
    assert_ne!(records_digest(&s0), records_digest(&s1));
}

#[test]
fn the_rule_keeps_minor_decisions_as_before() {
    // minor decisions take sims_minor (no coin is compared, none recorded) and the games play to the end
    let seat = with_rule(with_minor(nn(0, 16, true), Some(1)), prior_rule(0.05, UShape::Linear, 0.3));
    let mut c = cfg(vec![seat], vec![vec![0, 0]; 24], &[("m", KeyKind::Nn)]);
    c.seed = 2026;
    let mut d = Driver::<MinorToy>::new(c).unwrap();
    let (res, recs) = run(&mut d);
    let st = d.stats();
    assert!(st.minor > 0 && !recs.is_empty() && res.iter().all(|r| r.over));
    assert!(recs.iter().all(|r| r.opts.len() != 2), "a minor decision was recorded");
    assert_eq!(st.cap_targets, recs.len() as u64);
    // (the game's divisor: `light_decisions_divide_the_simulations_and_are_recorded`, which owns `DivToy`)
}

#[test]
fn the_rule_settings_are_checked() {
    let bad = |rule: FullRule, edit: &dyn Fn(&mut NnSeat)| {
        let SeatCfg::Nn(mut s) = with_rule(nn(0, 16, true), rule) else { unreachable!() };
        edit(&mut s);
        Driver::<ToyGame>::new(cfg(vec![SeatCfg::Nn(s)], vec![vec![0, 0]], &[("m", KeyKind::Nn)])).err().unwrap_or_default()
    };
    let none = |_: &mut NnSeat| {};
    assert!(bad(prior_rule(0.6, UShape::Linear, 0.3), &none).contains("p_full_floor"));
    assert!(bad(prior_rule(-0.1, UShape::Linear, 0.3), &none).contains("p_full_floor"));
    assert!(bad(prior_rule(0.05, UShape::Linear, 0.0), &none).contains("p_full_u_mean"));
    assert!(bad(prior_rule(0.05, UShape::Linear, f64::NAN), &none).contains("p_full_u_mean"));
    assert!(bad(prior_rule(0.0, UShape::Linear, 0.3), &|s| s.p_full = 0.0).contains("p_full 0"));
    assert!(bad(prior_rule(0.05, UShape::Linear, 0.3), &|s| s.sims_fast = None).contains("sims_fast"));
    assert!(bad(prior_rule(0.05, UShape::Linear, 0.3), &|s| s.root_search = RootSearch::Gumbel { m: 4, scale: 1.0, c_visit: 50.0, c_scale: 1.0 }).contains("gumbel"));
    assert_eq!(bad(prior_rule(0.0, UShape::Square, 0.3), &none), "", "a floor of 0 is valid");
    // the parser: the rule's keys belong to 'prior_uncertainty', which needs the mean
    assert_eq!(FullRule::parse("random", 0.05, "linear", None), Ok(FullRule::Random));
    assert!(FullRule::parse("prior_uncertainty", 0.05, "linear", None).unwrap_err().contains("p_full_u_mean"));
    assert!(FullRule::parse("prior_uncertainty", 0.05, "cubic", Some(0.3)).unwrap_err().contains("p_full_shape"));
    assert!(FullRule::parse("coin", 0.05, "linear", Some(0.3)).unwrap_err().contains("p_full_rule"));
    // the chance: the floor, linear in g(u) / u_mean, capped at 1; u from the priors' largest
    let r = prior_rule(0.05, UShape::Linear, 0.2);
    assert_eq!(r.full_share(0.25, &[1.0, 0.0]), 0.05);
    assert!((r.full_share(0.25, &[0.8, 0.2]) - 0.25).abs() < 1e-7);
    assert!((r.full_share(0.25, &[0.25, 0.25, 0.25, 0.25]) - 0.8).abs() < 1e-7);
    assert!((r.full_share(0.25, &[0.1; 10]) - 0.95).abs() < 1e-7);
    assert_eq!(prior_rule(0.05, UShape::Linear, 0.01).full_share(0.25, &[0.5, 0.5]), 1.0, "capped at 1");
    assert_eq!(FullRule::Random.full_share(0.25, &[0.5, 0.5]), 0.25);
    assert_eq!(prior_uncertainty(&[]), 0.0);
    assert_eq!((UShape::Sqrt.apply(0.25), UShape::Square.apply(0.5)), (0.5, 0.25));
}

#[test]
fn the_seat_parser_reads_the_rule() {
    use crate::settings::driver_cfg;
    use crate::settings::DriverArgs;
    let seat = |extra: serde_json::Value| {
        let mut d = serde_json::json!({"key": "m", "sims": 16, "selfplay": true, "sims_fast": 4, "p_full": 0.25});
        d.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        driver_cfg(&[vec![d.clone(), d]], None, None, DriverArgs::default()).map(|c| match &c.seats[0] {
            SeatCfg::Nn(s) => s.p_full_rule,
            _ => unreachable!(),
        })
    };
    assert_eq!(seat(serde_json::json!({})), Ok(FullRule::Random));
    assert_eq!(seat(serde_json::json!({"p_full_rule": "random"})), Ok(FullRule::Random));
    assert_eq!(seat(serde_json::json!({"p_full_rule": "prior_uncertainty", "p_full_u_mean": 0.2})), Ok(prior_rule(0.05, UShape::Linear, 0.2)));
    assert_eq!(seat(serde_json::json!({"p_full_rule": "prior_uncertainty", "p_full_floor": 0.1, "p_full_shape": "sqrt", "p_full_u_mean": 0.4})), Ok(prior_rule(0.1, UShape::Sqrt, 0.4)));
    assert!(seat(serde_json::json!({"p_full_floor": 0.1})).unwrap_err().contains("p_full_floor: a setting of p_full_rule"));
    assert!(seat(serde_json::json!({"p_full_rule": "prior_uncertainty"})).unwrap_err().contains("p_full_u_mean"));
}
