use super::*;
use crate::toy::{ToyGame, A_DONE, A_INC0};

/// Deterministic stand-in evaluator: priors from a hash of the action, value from the score
/// lead of the decider.
fn eval(g: &ToyGame, player: usize, acts: &[u16]) -> (Vec<f32>, [f32; MAX_SEATS], [f32; MAX_SEATS]) {
    let raw: Vec<f32> = acts.iter().map(|&a| 1.0 + ((a as u32).wrapping_mul(2654435761) >> 28) as f32).collect();
    let s: f32 = raw.iter().sum();
    let pri = raw.iter().map(|x| x / s).collect();
    let n = g.num_players();
    let lead = g.score(player) - (0..n).filter(|&i| i != player).map(|i| g.score(i)).fold(f32::MIN, f32::max);
    let w = 1.0 / (1.0 + (-lead / 4.0).exp());
    let mut v = [0.0; MAX_SEATS];
    v[0] = w;
    for j in 1..n {
        v[j] = (1.0 - w) / (n - 1) as f32;
    }
    (pri, v, [0.0; MAX_SEATS])
}

fn search(tree: &mut Tree<ToyGame>, live: &ToyGame, sims: usize, rng: &mut Rng, w: &Walk) -> usize {
    let mut g = live.clone();
    let mut sc = Scratch::default();
    let mut terminals = 0;
    for _ in 0..sims {
        match tree.simulate(live, &mut g, rng, w, &mut sc).unwrap() {
            Sim::Terminal => terminals += 1,
            Sim::Leaf { node, player } => {
                let (pri, v, m) = eval(&g, player, tree.acts(node));
                let (util, win) = leaf_values(&v, &m, player, tree.n, w.margin_w);
                tree.expand(node, &pri, &util);
                tree.backup(&sc.path, &util, &win);
            }
        }
    }
    terminals
}

fn mid_game(seed: u64, steps: usize) -> ToyGame {
    let mut rng = Rng::new(seed);
    let mut g = ToyGame::new(2, seed);
    let mut keys = Vec::new();
    let mut legal = Vec::new();
    for _ in 0..steps {
        legal.clear();
        g.legal(&mut legal);
        g.apply(legal[rng.below(legal.len() as u32) as usize], &mut keys).unwrap();
        if g.over {
            break;
        }
    }
    g
}

fn legal_of(g: &ToyGame) -> Vec<u16> {
    let mut l = Vec::new();
    g.legal(&mut l);
    l
}

const W: Walk = Walk { c_puct: 1.0, chance_k: 0, margin_w: 0.0 };

#[test]
fn visits_add_up() {
    let mut rng = Rng::new(1);
    for s in 0..20 {
        let g = mid_game(s, 10 + s as usize);
        if g.over {
            continue;
        }
        let p = g.to_move().unwrap();
        let mut t = Tree::new(&g, p, &legal_of(&g), true, &mut rng);
        search(&mut t, &g, 200, &mut rng, &W);
        let v = t.visits(t.root);
        // the first simulation only expands the root
        assert_eq!(v.iter().sum::<u32>(), 199);
        assert_eq!(t.tot(t.root), 199);
        let win = t.win(t.root).unwrap();
        assert!((win[0] + win[1] - 1.0).abs() < 1e-6);
    }
}

#[test]
fn search_reaches_terminals_late() {
    let mut rng = Rng::new(2);
    // a game near its end: simulations hit terminal states
    let mut g = mid_game(5, 0);
    let mut keys = Vec::new();
    while g.round < 6 || g.players[g.cur as usize].placed < 1 {
        let l = legal_of(&g);
        g.apply(l[0], &mut keys).unwrap();
    }
    let p = g.to_move().unwrap();
    let mut t = Tree::new(&g, p, &legal_of(&g), true, &mut rng);
    let term = search(&mut t, &g, 300, &mut rng, &W);
    assert!(term > 0);
}

#[test]
fn chance_children_keyed_by_outcome_and_chance_k_caps_them() {
    let mut rng = Rng::new(3);
    // the last decision of round 1: its transition reveals round 2
    for s in 0..10 {
        let mut g = ToyGame::new(2, s);
        let mut keys = Vec::new();
        loop {
            let l = legal_of(&g);
            let mut h = g.clone();
            let mut k = Vec::new();
            h.apply(l[0], &mut k).unwrap();
            if !k.is_empty() {
                break;
            }
            g.apply(l[0], &mut keys).unwrap();
        }
        let p = g.to_move().unwrap();
        for (ck, cap) in [(0u32, 6usize), (2, 2)] {
            let mut t = Tree::new(&g, p, &legal_of(&g), true, &mut rng);
            search(&mut t, &g, 400, &mut rng, &Walk { c_puct: 1.0, chance_k: ck, margin_w: 0.0 });
            let mut any = false;
            for i in 0..t.acts(t.root).len() {
                let kids = t.kids(t.root, i);
                if kids.is_empty() {
                    continue;
                }
                any = true;
                let mut ks: Vec<_> = kids.iter().map(|(k, _)| k.clone()).collect();
                ks.sort();
                ks.dedup();
                assert!(ks.len() <= cap, "chance_k {ck}: {} outcomes", ks.len());
                assert!(ks.iter().all(|k| k.len() == 1 && (k[0] >> 16) == 2));
                if ck == 2 {
                    assert!(t.outs(t.root, i).len() <= 2);
                }
            }
            assert!(any);
        }
    }
}

#[test]
fn advance_keeps_the_subtree() {
    let mut rng = Rng::new(4);
    let mut done = 0;
    for s in 0..40 {
        // a farm placement leads to further decisions of the same player
        let mut g = mid_game(s, 0);
        if !legal_of(&g).contains(&(crate::toy::S_FARM as u16)) {
            continue;
        }
        g.players[g.cur as usize].wood = 2;
        g.players[g.cur as usize].grain = 2;
        let p = g.to_move().unwrap();
        let mut t = Tree::new(&g, p, &legal_of(&g), true, &mut rng);
        search(&mut t, &g, 300, &mut rng, &W);
        let a = crate::toy::S_FARM as u16;
        let i = t.acts(t.root).iter().position(|&x| x == a).unwrap();
        let kids = t.kids(t.root, i);
        if kids.is_empty() {
            continue;
        }
        let kid_visits: u32 = t.visits(kids[0].1).iter().sum();
        let mut keys = Vec::new();
        let mut legal = Vec::new();
        transition(&mut g, a, None, &mut keys, &mut legal).unwrap();
        assert_eq!(g.to_move(), Some(p));
        assert!(t.advance(a, &keys, p, &legal));
        assert_eq!(t.visits(t.root).iter().sum::<u32>(), kid_visits);
        // continue the search from the reused root
        t.set_base(&g, &mut rng);
        search(&mut t, &g, 50, &mut rng, &W);
        assert!(t.visits(t.root).iter().sum::<u32>() >= kid_visits + 50 - 1);
        let mut c = t.clone();
        c.compact();
        assert_eq!(c.visits(c.root), t.visits(t.root));
        assert_eq!(c.acts(c.root), t.acts(t.root));
        done += 1;
    }
    assert!(done > 5);
}

#[test]
fn counts_decisions_are_searched() {
    // a Build counts decision: INC0 / DONE
    let mut rng = Rng::new(6);
    let mut g = mid_game(7, 0);
    g.sub = crate::toy::Sub::Build(0);
    let c = g.cur as usize;
    g.players[c].wood = 4;
    g.players[c].stone = 2;
    let l = legal_of(&g);
    assert_eq!(l, vec![A_INC0, A_DONE]);
    let mut t = Tree::new(&g, c, &l, true, &mut rng);
    search(&mut t, &g, 100, &mut rng, &W);
    assert_eq!(t.visits(t.root).iter().sum::<u32>(), 99);
}

#[test]
fn redet_off_uses_one_world() {
    let mut rng = Rng::new(8);
    let g = mid_game(9, 5);
    let p = g.to_move().unwrap();
    let mut t = Tree::new(&g, p, &legal_of(&g), false, &mut rng);
    search(&mut t, &g, 200, &mut rng, &W);
    // with one world every edge sees at most one outcome per chance event
    for i in 0..t.acts(t.root).len() {
        let ks: std::collections::HashSet<_> = t.kids(t.root, i).into_iter().map(|(k, _)| k).collect();
        assert!(ks.len() <= 1);
    }
}

#[test]
fn terminal_values_split_ties() {
    let mut rng = Rng::new(10);
    let mut seen_tie = false;
    for s in 0..300 {
        let mut g = ToyGame::new(2, s);
        let mut keys = Vec::new();
        while !g.over {
            let l = legal_of(&g);
            g.apply(l[rng.below(l.len() as u32) as usize], &mut keys).unwrap();
        }
        let (u, w) = terminal_values(&g, 0.5);
        if w[0] == 0.5 {
            seen_tie = true;
            assert_eq!(u[0], 0.5);
        }
        assert!((w[0] + w[1] - 1.0).abs() < 1e-9);
        assert!(u[0] >= w[0] - 0.5 && u[0] <= w[0] + 0.5);
    }
    assert!(seen_tie);
}

fn close(a: &[f32], b: &[f64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| (x as f64 - y).abs() <= tol * y.abs().max(1e-30) + 1e-12)
}

#[test]
fn completed_q_hand_checked() {
    // 3 options, the last unvisited: Q = [0.6, 0.8, v_mix], v_mix = (0.5 + 9 * (0.5*0.6 + 0.3*0.8)
    // / 0.8) / 10 = 0.6575, scale (50 + 6) * c_scale; reference values computed with 40-digit
    // decimals
    let (p0, n, w) = ([0.5f32, 0.3, 0.2], [6u32, 3, 0], [3.6f64, 2.4, 0.0]);
    let pi = completed_q_policy(&p0, &n, &w, 0.5, 50.0, 1.0);
    assert!(close(&pi, &[2.278460898e-05, 9.997491130e-01, 2.281023727e-04], 1e-6), "{pi:?}");
    let pi = completed_q_policy(&p0, &n, &w, 0.5, 50.0, 0.1);
    assert!(close(&pi, &[2.949098763e-01, 5.423134643e-01, 1.627766594e-01], 1e-6), "{pi:?}");
    // ties: equal priors, equal Q -> equal targets; unvisited options share v_mix = 4.3 / 11
    let pi = completed_q_policy(&[0.25; 4], &[5, 5, 0, 0], &[2.0, 2.0, 0.0, 0.0], 0.3, 50.0, 1.0);
    assert!(close(&pi, &[3.112296656e-01, 3.112296656e-01, 1.887703344e-01, 1.887703344e-01], 1e-6), "{pi:?}");
    assert_eq!(pi[0], pi[1]);
    assert_eq!(pi[2], pi[3]);
}

#[test]
fn completed_q_edge_cases() {
    // a single option
    assert_eq!(completed_q_policy(&[1.0], &[7], &[3.0], 0.4, 50.0, 1.0), vec![1.0]);
    assert_eq!(completed_q_policy(&[1.0], &[0], &[0.0], 0.4, 50.0, 1.0), vec![1.0]);
    // no visits: every option completed with v -> the prior
    let p0 = [0.1f32, 0.6, 0.3];
    let pi = completed_q_policy(&p0, &[0, 0, 0], &[0.0; 3], 0.7, 50.0, 1.0);
    assert!(close(&pi, &[0.1, 0.6, 0.3], 1e-6), "{pi:?}");
    // c_scale 0: the prior whatever the values
    let pi = completed_q_policy(&p0, &[4, 0, 9], &[4.0, 0.0, 0.0], 0.7, 50.0, 0.0);
    assert!(close(&pi, &[0.1, 0.6, 0.3], 1e-6), "{pi:?}");
    // a prior that underflowed to 0 stays finite (floored), and an unvisited zero-prior option
    // gets (almost) nothing
    let pi = completed_q_policy(&[1.0, 0.0, 0.0], &[3, 1, 0], &[1.5, 0.9, 0.0], 0.5, 50.0, 1.0);
    assert!(pi.iter().all(|x| x.is_finite() && *x >= 0.0), "{pi:?}");
    assert!((pi.iter().sum::<f32>() - 1.0).abs() < 1e-6 && pi[0] > 0.999 && pi[2] < 1e-30);
    // all options unvisited but one: the unvisited ones share v_mix (pulled towards the visited
    // Q), so a good visited option gains mass over its prior and a bad one loses it
    let up = completed_q_policy(&[0.25; 4], &[10, 0, 0, 0], &[9.0, 0.0, 0.0, 0.0], 0.5, 50.0, 1.0);
    let down = completed_q_policy(&[0.25; 4], &[10, 0, 0, 0], &[1.0, 0.0, 0.0, 0.0], 0.5, 50.0, 1.0);
    assert!(up[0] > 0.25 && down[0] < 0.25 && up[1] == up[3] && down[1] == down[2]);
    // monotone in c_scale: the best option's share grows
    let mut last = 0.0;
    for cs in [0.0, 0.05, 0.1, 0.3, 1.0, 3.0] {
        let pi = completed_q_policy(&[0.4, 0.4, 0.2], &[10, 10, 2], &[5.0, 6.0, 0.4], 0.5, 50.0, cs);
        assert!(pi[1] >= last, "c_scale {cs}: {pi:?}");
        last = pi[1];
    }
}

#[test]
fn cq_target_of_a_searched_root() {
    let mut rng = Rng::new(11);
    for s in 0..10 {
        let g = mid_game(30 + s, 12 + s as usize);
        if g.over {
            continue;
        }
        let p = g.to_move().unwrap();
        let mut t = Tree::new(&g, p, &legal_of(&g), true, &mut rng);
        search(&mut t, &g, 120, &mut rng, &W);
        let r = t.root;
        // the root's network value for its decider is what `eval` returned at expansion
        let mut g2 = g.clone();
        g2.determinize(p, &mut Rng::new(0));
        let (_, v, _) = eval(&g2, p, t.acts(r));
        assert_eq!(t.value0(r), v[0] as f64);
        let w: Vec<f64> = t.values(r);
        let pi = t.cq_target(r, 50.0, 1.0);
        assert_eq!(pi, completed_q_policy(&t.priors0(r), &t.visits(r), &w, t.value0(r), 50.0, 1.0));
        assert!((pi.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        // the option with the best Q among visited ones (and enough visits) gets the most mass
        let nv = t.visits(r);
        let best_q = (0..nv.len()).filter(|&i| nv[i] > 0).max_by(|&a, &b| (w[a] / nv[a] as f64).total_cmp(&(w[b] / nv[b] as f64))).unwrap();
        let best_pi = (0..pi.len()).max_by(|&a, &b| pi[a].total_cmp(&pi[b])).unwrap();
        let q = |i: usize| w[i] / nv[i] as f64;
        assert!(best_pi == best_q || nv[best_pi] == 0 || (q(best_pi) - q(best_q)).abs() < 0.05, "{pi:?} {nv:?} {w:?}");
    }
    // not expanded: empty
    let g = mid_game(1, 3);
    let t = Tree::new(&g, g.to_move().unwrap(), &legal_of(&g), true, &mut rng);
    assert!(t.cq_target(t.root, 50.0, 1.0).is_empty());
}
