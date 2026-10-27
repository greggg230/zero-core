use super::*;

fn play_random(n: usize, rounds: u8, seed: u64, rng: &mut Rng) -> (ToyTeam, Vec<u64>) {
    let mut g = ToyTeam::new(n, rounds, seed);
    let mut keys = Vec::new();
    let mut legal = Vec::new();
    while let Some(p) = g.to_move() {
        legal.clear();
        g.legal_actions(&mut legal);
        assert_eq!(legal.len(), n + 1 - g.turn as usize, "seat {p}");
        assert!(legal.windows(2).all(|w| w[0] < w[1]), "sorted, no duplicates");
        let a = legal[rng.below(legal.len() as u32) as usize];
        g.step(a, &mut keys).unwrap();
    }
    (g, keys)
}

#[test]
fn events_end_at_a_deferred_terminal_and_resolve() {
    let mut rng = Rng::new(3);
    for n in [2usize, 4] {
        for rounds in 1..=3u8 {
            for s in 0..40 {
                let (mut g, keys) = play_random(n, rounds, s, &mut rng);
                assert_eq!(keys.len(), rounds as usize - 1, "one offer key per later round");
                assert!(g.deferred() && g.to_move().is_none());
                // every seat picked once per round, no card twice
                let mut seen = 0u32;
                for p in 0..n {
                    for r in 0..rounds as usize {
                        let c = g.picks[p][r];
                        assert!(c < 16 && seen & (1 << c) == 0);
                        seen |= 1 << c;
                    }
                }
                assert!(std::panic::catch_unwind(|| g.result()).is_err(), "result() is refused before resolve");
                assert!(std::panic::catch_unwind(|| g.margin(0)).is_err());
                assert_eq!(g.score(0), 0.0);
                let v: Value = serde_json::from_str(&g.deferred_payload().unwrap()).unwrap();
                let t = n / 2;
                let eps = v["episodes"].as_array().unwrap();
                assert_eq!(eps.len(), t * t);
                for (e, pr) in eps.iter().zip(v["pairs"].as_array().unwrap()) {
                    let (a, b) = (pr[0].as_u64().unwrap() as usize, pr[1].as_u64().unwrap() as usize);
                    assert!(a % 2 == 0 && b % 2 == 1);
                    assert_eq!(e["food"], json!([g.bonus(a), g.bonus(b)]));
                    // a toy spec
                    assert!(<crate::toy::ToyGame as SearchGame>::parse_spec(&e.to_string()).is_ok());
                }
                // malformed or impossible results are errors and leave the event deferred
                assert!(g.resolve("[]").is_err());
                assert!(g.resolve("not json").is_err());
                let bad: Vec<Value> = (0..t * t).map(|_| json!([0.7, 0.3])).collect();
                assert!(g.resolve(&Value::Array(bad).to_string()).is_err());
                assert!(g.deferred());
                // A wins the first match, the rest alternate
                let res: Vec<Value> = (0..t * t).map(|k| json!({"result": if k % 2 == 0 { [1.0, 0.0] } else { [0.0, 1.0] }, "scores": [3, 1]})).collect();
                g.resolve(&Value::Array(res).to_string()).unwrap();
                assert!(!g.deferred() && g.to_move().is_none());
                let r = g.result();
                assert!((r.iter().sum::<f32>() - 1.0).abs() < 1e-6);
                let o = g.outcome.clone().unwrap();
                let a_wins = (t * t).div_ceil(2) as u32;
                assert_eq!(o.wins, [a_wins, (t * t) as u32 - a_wins]);
                for (s, &rs) in r.iter().enumerate().take(n) {
                    assert_eq!(rs, o.team[s % 2] / t as f32);
                    assert_eq!(g.margin(s), (o.wins[s % 2] as f32 - o.wins[1 - s % 2] as f32) / (t * t) as f32);
                }
                assert!(g.resolve("[[1, 0]]").is_err(), "resolved once");
            }
        }
    }
}

#[test]
fn determinize_keeps_the_viewers_knowledge() {
    let mut rng = Rng::new(5);
    for s in 0..60 {
        let mut g = ToyTeam::new(4, 3, s);
        let mut keys = Vec::new();
        let steps = rng.below(10) as usize;
        for _ in 0..steps {
            let mut l = Vec::new();
            g.legal_actions(&mut l);
            g.step(l[rng.below(l.len() as u32) as usize], &mut keys).unwrap();
        }
        let p = g.to_move().unwrap();
        let mut x = vec![0.0; NUM_FEATURES];
        g.encode(p, &mut x);
        let mut l0 = Vec::new();
        g.legal_actions(&mut l0);
        let mut differ = false;
        for k in 0..20 {
            let mut d = g.clone();
            d.determinize(p, &mut Rng::new(100 + k));
            let mut y = vec![0.0; NUM_FEATURES];
            d.encode(p, &mut y);
            let mut l1 = Vec::new();
            d.legal_actions(&mut l1);
            assert_eq!((x.clone(), l0.clone()), (y, l1));
            assert_eq!(&d.deck[..d.drawn as usize], &g.deck[..g.drawn as usize]);
            differ |= d.deck != g.deck;
        }
        assert!(differ || g.drawn as usize >= CARDS - 1, "the undealt deck is reshuffled");
    }
}

#[test]
fn forced_offers_are_dealt_and_impossible_ones_fall_back() {
    // the last pick of round 0 deals round 1's offer
    for s in 0..30 {
        let mut g = ToyTeam::new(2, 3, s);
        let mut keys = Vec::new();
        let mut l = Vec::new();
        g.legal_actions(&mut l);
        g.step(l[0], &mut keys).unwrap();
        let mut real = g.clone();
        l.clear();
        real.legal_actions(&mut l);
        let a = l[0];
        let mut k0 = Vec::new();
        real.step(a, &mut k0).unwrap();
        assert_eq!(k0.len(), 1);
        for w in 0..10 {
            let mut d = g.clone();
            d.determinize(0, &mut Rng::new(w));
            d.force_chance(&k0);
            let mut k1 = Vec::new();
            d.step(a, &mut k1).unwrap();
            assert_eq!(k1, k0, "a forced offer of undealt cards is dealt");
            assert_eq!(d.offer, real.offer);
        }
        // an offer holding a dealt card cannot be dealt again: the world's own
        let dealt = g.deck[0] as u64;
        let bad = (1u64 << 32) | (1 << dealt) | 0b111;
        let mut d = g.clone();
        d.force_chance(&[bad]);
        let mut k2 = Vec::new();
        d.step(a, &mut k2).unwrap();
        assert_eq!(k2, k0);
    }
}

#[test]
fn greedy_takes_the_most_food() {
    let g = ToyTeam::new(2, 2, 1);
    let mut b = Greedy;
    assert_eq!(b.choose(&g, &[0, 5, 7, 11]), 7);
    assert_eq!(b.choose(&g, &[4, 8]), 4);
    assert!(ToyTeam::parse_spec(r#"{"rounds": 4}"#).is_err() && ToyTeam::parse_spec(r#"{"x": 1}"#).is_err());
    assert_eq!(ToyTeam::parse_spec("").unwrap().rounds, 2);
    assert!(ToyTeam::new_game(&ToyTeamSpec { rounds: 2 }, 3, 0).is_err());
    let g = ToyTeam::new_game(&ToyTeamSpec { rounds: 2 }, 4, 0).unwrap();
    assert_eq!((g.team_size(), g.num_players()), (2, 4));
    assert_eq!(crate::game::team_of(3, 4, 2), 1);
    assert_eq!(crate::game::team_of(2, 4, 2), 0);
    assert_eq!(crate::game::team_of(2, 3, 1), 2);
}
