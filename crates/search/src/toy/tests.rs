use super::*;

fn random_game(n: usize, seed: u64, rng: &mut Rng) -> (ToyGame, Vec<u64>) {
    let mut g = ToyGame::new(n, seed);
    let mut keys = Vec::new();
    let mut legal = Vec::new();
    let mut steps = 0;
    while !g.over {
        legal.clear();
        g.legal(&mut legal);
        assert!(!legal.is_empty());
        let a = legal[rng.below(legal.len() as u32) as usize];
        g.apply(a, &mut keys).unwrap();
        check_invariants(&g);
        steps += 1;
        assert!(steps < 1000, "game does not end");
    }
    (g, keys)
}

fn check_invariants(g: &ToyGame) {
    for p in &g.players[..g.n as usize] {
        assert!(p.food >= 0 && p.wood >= 0 && p.stone >= 0 && p.grain >= 0, "{p:?}");
        assert!(p.workers >= 2 && p.workers <= 4 && p.newborn <= p.workers, "{p:?}");
        assert!(p.rooms >= 2 && p.rooms <= 4 && p.plowed <= 4, "{p:?}");
        assert!(p.placed + p.newborn <= p.workers, "{p:?}");
    }
}

#[test]
fn random_games_terminate_with_sane_results() {
    let mut rng = Rng::new(1);
    let mut ties = 0;
    for n in 2..=4 {
        for s in 0..300 {
            let (g, keys) = random_game(n, s, &mut rng);
            // one reveal per round after the first
            assert_eq!(keys.len(), 5);
            for (i, k) in keys.iter().enumerate() {
                assert_eq!((k >> 16) as usize, i + 2);
            }
            let r = g.result();
            let tot: f32 = r.iter().sum();
            assert!((tot - 1.0).abs() < 1e-6);
            if r[..n].iter().any(|&x| x > 0.0 && x < 1.0) {
                ties += 1;
            }
            assert_eq!(g.to_move(), None);
            let m: f32 = (0..n).map(|i| g.margin(i)).fold(f32::MIN, f32::max);
            assert!(m >= 0.0);
        }
    }
    assert!(ties > 10, "ties should happen: {ties}");
}

#[test]
fn clone_reproduces_the_continuation() {
    let mut rng = Rng::new(9);
    for s in 0..50 {
        let mut g = ToyGame::new(2, s);
        let mut keys = Vec::new();
        let mut legal = Vec::new();
        for _ in 0..(s as usize % 17) {
            legal.clear();
            g.legal(&mut legal);
            g.apply(legal[rng.below(legal.len() as u32) as usize], &mut keys).unwrap();
        }
        let mut h = g.clone();
        while !g.over {
            legal.clear();
            g.legal(&mut legal);
            let a = legal[rng.below(legal.len() as u32) as usize];
            let (mut k1, mut k2) = (Vec::new(), Vec::new());
            g.apply(a, &mut k1).unwrap();
            h.apply(a, &mut k2).unwrap();
            assert_eq!(k1, k2);
            assert_eq!(g.to_json(), h.to_json());
        }
    }
}

#[test]
fn determinize_keeps_revealed_facts_and_legal_actions() {
    let mut rng = Rng::new(3);
    let mut legal = Vec::new();
    let mut changed = 0;
    for s in 0..200 {
        let mut g = ToyGame::new(2 + (s as usize % 3), s);
        let mut keys = Vec::new();
        let steps = rng.below(60);
        for _ in 0..steps {
            if g.over {
                break;
            }
            legal.clear();
            g.legal(&mut legal);
            g.apply(legal[rng.below(legal.len() as u32) as usize], &mut keys).unwrap();
        }
        if g.over {
            continue;
        }
        let mut l0 = Vec::new();
        g.legal(&mut l0);
        let x0 = enc(&g, g.cur as usize);
        for _ in 0..5 {
            let mut d = g.clone();
            SearchGame::determinize(&mut d, 0, &mut rng);
            let r = g.round as usize;
            assert_eq!(&d.order[..r], &g.order[..r]);
            assert_eq!(&d.markers[..r], &g.markers[..r]);
            assert_eq!(d.markers[5], M_HARVEST);
            let mut a = d.order[..3].to_vec();
            a.sort();
            assert_eq!(a, vec![0, 1, 2]);
            let mut b = d.order[3..].to_vec();
            b.sort();
            assert_eq!(b, vec![3, 4, 5]);
            let mut m = d.markers[1..5].to_vec();
            m.sort();
            assert_eq!(m, vec![1, 1, 2, 2]);
            let mut l1 = Vec::new();
            d.legal(&mut l1);
            assert_eq!(l0, l1);
            // features never see the hidden future
            assert_eq!(x0, enc(&d, g.cur as usize));
            changed += (d.order != g.order || d.markers != g.markers) as u32;
        }
    }
    assert!(changed > 100);
}

#[test]
fn forced_chance_outcomes() {
    // play to the end of round 1 and force round 2's card and marker
    let mut rng = Rng::new(5);
    let mut hits = 0;
    for s in 0..100 {
        let g0 = ToyGame::new(2, s);
        for c in 0..3u8 {
            for m in [M_HARVEST, M_FEED] {
                let mut g = g0.clone();
                SearchGame::force_chance(&mut g, &[chance_key(2, c, m)]);
                let mut keys = Vec::new();
                let mut legal = Vec::new();
                while g.round == 1 {
                    legal.clear();
                    g.legal(&mut legal);
                    g.apply(legal[rng.below(legal.len() as u32) as usize], &mut keys).unwrap();
                }
                let possible = g0.order[0] != c;
                assert_eq!(keys.len(), 1);
                if possible {
                    assert_eq!(keys[0], chance_key(2, c, m));
                    hits += 1;
                } else {
                    // the round-1 card cannot come again: falls back to the world's own card
                    assert_eq!((keys[0] >> 8) & 0xff, g0.order[1] as u64);
                    assert_eq!(keys[0] & 0xff, m as u64);
                }
            }
        }
    }
    assert!(hits > 300);
}

#[test]
fn heuristic_beats_random() {
    let mut rng = Rng::new(11);
    let mut wins = 0.0;
    let games = 400;
    for s in 0..games {
        let mut g = ToyGame::new(2, s);
        let hseat = (s % 2) as usize;
        let mut keys = Vec::new();
        let mut legal = Vec::new();
        while !g.over {
            legal.clear();
            g.legal(&mut legal);
            let a = if g.cur as usize == hseat { g.heuristic(&legal) } else { legal[rng.below(legal.len() as u32) as usize] };
            g.apply(a, &mut keys).unwrap();
        }
        wins += SearchGame::result(&g)[hseat];
    }
    let w = wins / games as f32;
    assert!(w > 0.9, "heuristic vs random {w}");
}

#[test]
fn heuristic_always_legal_and_json_parses() {
    let mut rng = Rng::new(13);
    for s in 0..100 {
        let mut g = ToyGame::new(2 + (s as usize % 3), s);
        let mut keys = Vec::new();
        let mut legal = Vec::new();
        while !g.over {
            legal.clear();
            g.legal(&mut legal);
            let a = if rng.below(2) == 0 { g.heuristic(&legal) } else { legal[rng.below(legal.len() as u32) as usize] };
            assert!(legal.contains(&a));
            let v: serde_json::Value = serde_json::from_str(&g.to_json()).unwrap();
            assert_eq!(v["legal"].as_array().unwrap().len(), legal.len());
            g.apply(a, &mut keys).unwrap();
        }
    }
}


fn enc(g: &ToyGame, viewer: usize) -> Vec<f32> {
    let mut v = vec![0.0; NUM_FEATURES];
    SearchGame::encode(g, viewer, &mut v);
    v
}
