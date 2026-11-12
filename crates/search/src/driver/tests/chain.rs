//! Chained episodes (docs/search.md 1.3, 2.14; docs/az.md 42): the toy `toy_match` (a best of three toy games, each
//! game its own episode) through the driver - one row per contest with its episodes, every record labelled with its
//! own game's outcome, the match's rules, resignation as a concession, refusals.
use super::*;
use crate::toy;
use crate::toy_match::{ToyMatch, A_FIRST, A_SECOND};
use crate::tree::{leaf_values, Scratch, Sim, Tree, Walk};

/// Every row with its own samples (a row's samples follow the row taken with it: `workqueue.split_games`' contract).
fn run_rows<G: SearchGame>(d: &mut Driver<G>, ans: &dyn Fn(&mut Driver<G>, &Batch)) -> Vec<(GameResult, Vec<Record>)> {
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
        ans(d, &b);
        take(d, &mut rows);
    }
    take(d, &mut rows);
    rows.sort_by_key(|(r, _)| r.gid);
    rows
}

/// Uniform priors; the decider's value from the toy features' score lead and round (a lead late in a game is
/// decisive): the toy block of the row (at a start choice: the finished game's final state, a decided lead).
fn answer_lead_match(d: &mut Driver<ToyMatch>, b: &Batch) {
    let f = ToyMatch::NUM_FEATURES;
    let base = 6 + 3 + 4 + 12 + 6 + toy::N_SPACES * 7;
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

fn match_cfg(seat: SeatCfg, games: usize) -> DriverCfg {
    let mut c = cfg(vec![seat], vec![vec![0, 0]; games], &[("m", KeyKind::Nn)]);
    c.seed = 23;
    c
}

/// A record's features show a pending start choice (the match block's flag).
fn at_choice(r: &Record) -> bool {
    r.x[toy::NUM_FEATURES + 4] > 0.5
}

/// The checks every contest row must pass: the match's rules (36.1's analogue), the row's sums, and each record
/// labelled with its own episode's outcome.
fn check_contest(r: &GameResult, smp: &[Record]) {
    assert!(r.over, "game {}: unfinished", r.gid);
    let eps = &r.episodes;
    assert!((2..=3).contains(&eps.len()), "game {}: {} episodes", r.gid, eps.len());
    let mut wins = [0u8; 2];
    for (k, e) in eps.iter().enumerate() {
        assert!(e.over && !e.aborted);
        let w = (0..2).find(|&s| e.result[s] > 0.75);
        if let Some(w) = w {
            wins[w] += 1;
        }
        // the match ends at the first two wins or after three games
        let decided = wins.iter().any(|&x| x >= 2) || k + 1 == 3;
        assert_eq!(decided, k + 1 == eps.len(), "game {}: the match ended at episode {} of {}, wins {wins:?}", r.gid, k + 1, eps.len());
    }
    let want = match wins[0].cmp(&wins[1]) {
        std::cmp::Ordering::Greater => [1.0, 0.0],
        std::cmp::Ordering::Less => [0.0, 1.0],
        std::cmp::Ordering::Equal => [0.5, 0.5],
    };
    assert_eq!(r.result, want, "game {}: the contest's result is the match's", r.gid);
    assert_eq!(r.scores, vec![wins[0] as f32, wins[1] as f32]);
    assert_eq!(r.samples as usize, smp.len());
    assert_eq!(r.samples, eps.iter().map(|e| e.samples).sum::<u32>());
    assert_eq!(r.decisions, eps.iter().map(|e| e.decisions).sum::<u32>());
    assert_eq!(r.steps, eps.iter().map(|e| e.steps).sum::<u64>());
    assert!(r.resign.is_none());
    let mut a = 0;
    for (k, e) in eps.iter().enumerate() {
        let part = &smp[a..a + e.samples as usize];
        a += e.samples as usize;
        for s in part {
            for j in 0..2 {
                let seat = (s.player as usize + j) % 2;
                assert_eq!(s.z[j], e.result[seat], "game {} episode {}: z is the episode's result", r.gid, k + 1);
                assert_eq!(s.sc[j], e.margins[seat], "game {} episode {}: sc is the episode's margin", r.gid, k + 1);
            }
            // a start choice is the first decision of a later episode, never of game 1
            if at_choice(s) {
                assert!(k > 0, "game {}: a start choice recorded in game 1's episode", r.gid);
            }
        }
    }
}

#[test]
fn chained_contests_label_every_record_by_its_own_game() {
    let rows = run_rows(&mut Driver::<ToyMatch>::new(match_cfg(nn(0, 16, true), 40)).unwrap(), &|d, b| answer(d, b));
    assert_eq!(rows.len(), 40, "one row per contest");
    let mut three = 0;
    let mut choices = 0;
    for (r, smp) in &rows {
        check_contest(r, smp);
        three += (r.episodes.len() == 3) as usize;
        choices += smp.iter().filter(|s| at_choice(s)).count();
    }
    assert!(three > 0 && three < 40, "{three} matches of three games");
    assert!(choices > 0, "the start choices are recorded (with the next game's outcome)");
}

#[test]
fn a_match_lost_1_2_by_the_winner_of_game_2() {
    // the user's case: the player who won game 2 and lost the match 1-2 - its game-2 rows (its start choice
    // before game 2 among them, the loser of game 1 choosing) carry a win, its game-1 and game-3 rows a loss
    let rows = run_rows(&mut Driver::<ToyMatch>::new(match_cfg(nn(0, 16, true), 120)).unwrap(), &|d, b| answer(d, b));
    let mut found = 0;
    for (r, smp) in &rows {
        let eps = &r.episodes;
        if eps.len() != 3 {
            continue;
        }
        let winner = |e: &EpisodeRow| (0..2).find(|&s| e.result[s] > 0.75);
        let (Some(w1), Some(w2), Some(w3)) = (winner(&eps[0]), winner(&eps[1]), winner(&eps[2])) else { continue };
        if w2 == w1 || w3 != w1 {
            continue;
        }
        let p = w2; // won game 2, lost games 1 and 3: the match 1-2
        assert_eq!(r.result[p], 0.0, "the match is lost");
        let (a, b) = (eps[0].samples as usize, eps[1].samples as usize);
        let mine = |part: &[Record]| part.iter().filter(|s| s.player as usize == p).map(|s| s.z[0]).collect::<Vec<f32>>();
        let (g1, g2, g3) = (mine(&smp[..a]), mine(&smp[a..a + b]), mine(&smp[a + b..]));
        assert!(!g1.is_empty() && !g2.is_empty() && !g3.is_empty());
        assert!(g1.iter().all(|&z| z == 0.0), "game 1: a loss");
        assert!(g2.iter().all(|&z| z == 1.0), "game 2: a win");
        assert!(g3.iter().all(|&z| z == 0.0), "game 3: a loss");
        // p lost game 1, so it chose who starts game 2: that choice is a game-2 row of p's, labelled a win
        let choice: Vec<&Record> = smp[a..a + b].iter().filter(|s| at_choice(s) && s.opts == [A_FIRST, A_SECOND]).collect();
        if let Some(c) = choice.iter().find(|s| s.player as usize == p) {
            assert_eq!(c.z[0], 1.0);
            found += 1;
        }
    }
    assert!(found > 0, "no match lost 1-2 by game 2's winner with its start choice recorded");
}

#[test]
fn chained_contests_replay_across_threads_and_count_in_rolling_quotas() {
    let mk = |threads: usize, groups: usize| {
        let mut c = match_cfg(nn(0, 12, true), 16);
        c.threads = threads;
        c.groups = groups;
        run_rows(&mut Driver::<ToyMatch>::new(c).unwrap(), &|d, b| answer(d, b))
    };
    let key = |(r, s): &(GameResult, Vec<Record>)| (r.gid, r.steps, r.result.clone(), r.episodes.iter().map(|e| (e.steps, e.samples, e.result.clone())).collect::<Vec<_>>(), records_digest(s));
    let a: Vec<_> = mk(1, 1).iter().map(key).collect();
    let b: Vec<_> = mk(4, 3).iter().map(key).collect();
    assert_eq!(a, b);
    // a rolling driver's quota counts contests (one row each), never episodes
    let mut c = match_cfg(nn(0, 8, true), 1);
    c.rolling = true;
    c.concurrency = 6;
    let mut d = Driver::<ToyMatch>::new(c).unwrap();
    for phase in 0..2 {
        d.resume(5, phase).unwrap();
        let rows = run_rows(&mut d, &|d, b| answer(d, b));
        assert!(rows.len() >= 5, "phase {phase}: {} contests", rows.len());
        for (r, s) in &rows {
            check_contest(r, s);
        }
    }
}

#[test]
fn a_resignation_concedes_the_game_and_the_match_goes_on() {
    let seat = with_resign(nn(0, 16, true), 0.45, 1, 0.0);
    let rows = run_rows(&mut Driver::<ToyMatch>::new(match_cfg(seat, 40)).unwrap(), &answer_lead_match);
    assert_eq!(rows.len(), 40);
    let (mut resigned, mut later) = (0, 0);
    for (r, smp) in &rows {
        assert!(r.over);
        let eps = &r.episodes;
        let mut wins = [0u8; 2];
        let mut a = 0;
        for (k, e) in eps.iter().enumerate() {
            let rs = e.resign.as_ref().expect("a resigning seat's episodes report the rule");
            if rs.resigned {
                resigned += 1;
                let w = rs.winner.unwrap() as usize;
                assert_eq!(e.result[w], 1.0, "the resigned game is the favoured seat's");
                // a start choice is no game in progress: the rule never counts it (its toy block is the last game's
                // final, decided state, so without `can_concede` it would fire there in nearly every later episode)
                assert!(!(k > 0 && rs.decision == 1), "game {} episode {}: resigned at the start choice", r.gid, k + 1);
                later += (k + 1 < eps.len()) as usize;
            }
            if let Some(w) = (0..2).find(|&s| e.result[s] > 0.75) {
                wins[w] += 1;
            }
            for s in &smp[a..a + e.samples as usize] {
                assert_eq!(s.z[0], e.result[s.player as usize]);
            }
            a += e.samples as usize;
        }
        assert!(wins.iter().any(|&x| x >= 2) || eps.len() == 3);
    }
    assert!(resigned > 20, "{resigned} resigned games");
    assert!(later > 0, "a match goes on after a resigned game");
}

#[test]
fn branched_games_are_refused_with_chained_episodes() {
    let mut c = match_cfg(nn(0, 8, true), 2);
    c.branch = Some(BranchCfg { p: 1.0, moves: 1, temp: 1.0, differ: false });
    let mut d = Driver::<ToyMatch>::new(c).unwrap();
    let e = loop {
        match d.next_batch(true) {
            Ok(Some(b)) => answer(&mut d, &b),
            Ok(None) => panic!("the driver finished: no refusal"),
            Err(e) => break e,
        }
    };
    assert!(e.contains("chains its episodes"), "{e}");
}

#[test]
fn a_bots_contest_and_the_move_log_span_the_match() {
    let mut c = cfg(vec![SeatCfg::Bot("random".into()), SeatCfg::Bot("heuristic".into())], (0..40).map(|g| if g % 2 == 0 { vec![0, 1] } else { vec![1, 0] }).collect(), &[]);
    c.move_log = 100_000;
    let rows = run_rows(&mut Driver::<ToyMatch>::new(c).unwrap(), &|d, b| answer(d, b));
    assert_eq!(rows.len(), 40);
    let mut hw = 0.0;
    for (r, smp) in &rows {
        assert!(smp.is_empty());
        check_contest(r, smp);
        hw += r.result[if r.gid % 2 == 0 { 1 } else { 0 }];
        assert_eq!(r.moves.len() as u64, r.steps, "the move log covers every episode");
    }
    assert!(hw / 40.0 > 0.9, "heuristic vs random, matches: {hw}");
}

/// The last decision of game 1 of a match (every option ends the game), and the match itself.
fn last_decision(seed: u64) -> Option<ToyMatch> {
    let mut rng = Rng::new(seed);
    let mut m = ToyMatch::new(seed);
    let mut keys = Vec::new();
    let mut legal = Vec::new();
    loop {
        legal.clear();
        m.legal_actions(&mut legal);
        if m.to_move().is_none() {
            return None;
        }
        let ends = legal.iter().all(|&a| {
            let mut c = m.clone();
            c.step(a, &mut Vec::new()).unwrap();
            c.to_move().is_none()
        });
        if ends && legal.len() >= 2 {
            return Some(m);
        }
        m.step(legal[rng.below(legal.len() as u32) as usize], &mut keys).unwrap();
    }
}

#[test]
fn the_search_treats_a_games_end_as_terminal() {
    // at game 1's last decision every walk below the root reaches the game's end and backs up that game's result:
    // nothing past it (the next game's start choice, its deal) is ever expanded or evaluated
    let w = Walk { c_puct: 1.0, chance_k: 0, margin_w: 0.0 };
    let mut tested = 0;
    for seed in 0..200 {
        let Some(live) = last_decision(seed) else { continue };
        let p = live.to_move().unwrap();
        let mut legal = Vec::new();
        live.legal_actions(&mut legal);
        let mut rng = Rng::new(seed);
        let mut t = Tree::new(&live, p, &legal, true, &mut rng);
        let mut g = live.clone();
        let mut sc = Scratch::default();
        let (mut leaves, mut terminals) = (0, 0);
        for _ in 0..64 {
            match t.simulate(&live, &mut g, &mut rng, &w, &mut sc).unwrap() {
                Sim::Terminal => terminals += 1,
                Sim::Leaf { node, player } => {
                    leaves += 1;
                    assert_eq!(node, t.root, "only the root is evaluated");
                    let k = t.acts(node).len();
                    let mut v = [0.0; MAX_SEATS];
                    v[0] = 0.5;
                    v[1] = 0.5;
                    let (util, win) = leaf_values(&v, &[0.0; MAX_SEATS], player, 2, 0.0);
                    t.expand(node, &vec![1.0 / k as f32; k], &util);
                    t.backup(&sc.path, &util, &win);
                }
            }
        }
        assert_eq!((leaves, terminals), (1, 63));
        // each option's value is its game result for the decider (the toy's world is fully determined at its end)
        let wins: Vec<f64> = legal
            .iter()
            .map(|&a| {
                let mut c = live.clone();
                c.step(a, &mut Vec::new()).unwrap();
                c.result()[p] as f64
            })
            .collect();
        let n = t.visits(t.root);
        let q = t.values(t.root);
        for (i, &wi) in wins.iter().enumerate() {
            if n[i] > 0 {
                assert!((q[i] / n[i] as f64 - wi).abs() < 1e-9, "seed {seed}: option {i}: {} vs {wi}", q[i] / n[i] as f64);
            }
        }
        tested += 1;
    }
    assert!(tested >= 20, "{tested} positions");
}

#[test]
fn next_episode_continues_the_match_or_ends_it() {
    // by the rules: the loser chooses who starts the next game; the match ends at two wins or three games; a
    // concession scores the game in progress as the conceding seat's loss; the contest's result by game wins
    let mut m = ToyMatch::new(5);
    assert!(m.chained() && m.can_concede());
    assert!(m.next_episode(None).is_err(), "not at a game's end");
    let Next::Continue(m2) = m.next_episode(Some(0)).unwrap() else { panic!("one game of three") };
    assert_eq!((m2.wins, m2.game_no, m2.choosing), ([0, 1], 1, Some(0)), "seat 0 conceded game 1 and chooses");
    assert!(!m2.can_concede() && m2.to_move() == Some(0));
    let mut legal = Vec::new();
    m2.legal_actions(&mut legal);
    assert_eq!(legal, vec![A_FIRST, A_SECOND]);
    assert!(m2.next_episode(Some(0)).is_err(), "no game in progress to concede");
    let mut m3 = m2.clone();
    let mut keys = Vec::new();
    m3.step(A_SECOND, &mut keys).unwrap();
    assert_eq!((m3.game_no, m3.g.start, keys.len()), (2, 1, 1), "seat 1 starts game 2; its setup is keyed");
    let Next::Over(r) = m3.next_episode(Some(0)).unwrap() else { panic!("two wins end the match") };
    assert_eq!(&r[..2], &[0.0, 1.0]);
    // played out by random moves, every match ends by the rules
    for seed in 0..50 {
        let mut rng = Rng::new(seed);
        m = ToyMatch::new(seed);
        let mut games = 1;
        loop {
            legal.clear();
            m.legal_actions(&mut legal);
            if m.to_move().is_none() {
                match m.next_episode(None).unwrap() {
                    Next::Continue(n) => {
                        games += 1;
                        m = n;
                        continue;
                    }
                    Next::Over(r) => {
                        assert!(m.wins.iter().any(|&w| w >= 2) || m.game_no == 3);
                        assert_eq!(r, m.match_result());
                        assert_eq!(games, m.game_no as usize);
                        break;
                    }
                }
            }
            m.step(legal[rng.below(legal.len() as u32) as usize], &mut keys).unwrap();
        }
    }
}

#[test]
fn traced_contests_replay_to_their_rows_bit_for_bit() {
    // a traced rolling driver of toy matches (docs/search.md 2.17): every contest's record - its episode boundaries,
    // each episode's outcome and setup, resignations conceding a game while the match goes on - replays into the
    // rows the driver emitted, each labelled by its own game's outcome, and the match's result
    let seat = with_resign(with_view(nn(0, 16, true), LeafView::Mover, 0.5, ViewTarget::NextQ), 0.45, 1, 0.2);
    let mut c = match_cfg(seat, 1);
    (c.rolling, c.concurrency, c.groups, c.threads, c.aux) = (true, 8, 2, 3, true);
    let mut d = Driver::<ToyMatch>::new(c).unwrap();
    d.set_trace(true).unwrap();
    let mut rows: Vec<(GameResult, Vec<Record>)> = Vec::new();
    let take = |d: &mut Driver<ToyMatch>, rows: &mut Vec<(GameResult, Vec<Record>)>| {
        let rs = d.take_results();
        let smp = d.take_samples();
        let mut a = 0;
        for r in rs {
            let k = r.samples as usize;
            rows.push((r, smp[a..a + k].to_vec()));
            a += k;
        }
    };
    for net in [0u32, 0, 1] {
        d.resume(12, net).unwrap();
        while let Some(b) = d.next_batch(true).unwrap() {
            answer_lead_match(&mut d, &b);
            take(&mut d, &mut rows);
        }
        take(&mut d, &mut rows);
    }
    let spec = ToyMatch::parse_spec("{}").unwrap();
    let (mut multi, mut conceded, mut views) = (0, 0, 0);
    for (r, s) in &rows {
        let rec: serde_json::Value = serde_json::from_str(&crate::record::game_json(r, s).unwrap().to_string()).unwrap();
        let rp = crate::record::replay::<ToyMatch>(&spec, &rec, true).unwrap_or_else(|e| panic!("contest {}: {e}", r.gid));
        assert_eq!((rp.episodes, rp.over), (r.episodes.len(), r.over));
        assert_eq!(rp.result.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), r.result.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        assert_eq!(rp.records.len(), s.len());
        for (k, (a, b)) in rp.records.iter().zip(s).enumerate() {
            super::same_row(a, b).unwrap_or_else(|e| panic!("contest {} row {k}: {e}", r.gid));
        }
        multi += (r.episodes.len() > 1) as usize;
        conceded += r.episodes.iter().filter(|e| e.resign.as_ref().is_some_and(|x| x.resigned)).count();
        views += s.iter().filter(|x| x.opts.is_empty()).count();
        // a tampered episode outcome or boundary is refused
        let mut bad = rec.clone();
        bad["episodes"][0]["result"][0] = serde_json::json!(0.5f32.to_bits());
        assert!(crate::record::replay::<ToyMatch>(&spec, &bad, true).is_err());
        if let Some(i) = rec["steps"].as_array().unwrap().iter().position(|v| v.is_object()) {
            let mut bad = rec.clone();
            bad["steps"].as_array_mut().unwrap().remove(i);
            assert!(crate::record::replay::<ToyMatch>(&spec, &bad, true).is_err(), "an episode boundary dropped");
        }
    }
    assert!(rows.len() >= 36, "{} contests", rows.len());
    assert!(multi > 0 && conceded > 0 && views > 0, "contests of several games {multi}, conceded games {conceded}, value rows {views}");
}
