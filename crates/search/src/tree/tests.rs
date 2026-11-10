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
fn shape_counts_every_walk_and_edge() {
    // a fresh search: every walk but the terminal ones expanded one node, the root's own evaluation the first
    let mut rng = Rng::new(11);
    let mut checked = 0;
    for s in 0..20 {
        let g = mid_game(s, 10 + s as usize);
        if g.over {
            continue;
        }
        let p = g.to_move().unwrap();
        let mut t = Tree::new(&g, p, &legal_of(&g), true, &mut rng);
        let term = search(&mut t, &g, 120, &mut rng, &W);
        let sh = t.shape();
        assert_eq!(sh.nodes as usize, t.n_nodes(), "a fresh tree holds no garbage");
        assert_eq!(sh.expanded as usize + term, 120);
        assert_eq!(sh.depth_hist.iter().sum::<u32>(), sh.expanded);
        assert_eq!(sh.opp_hist.iter().sum::<u32>(), sh.expanded);
        assert_eq!(sh.switch_hist.iter().sum::<u32>(), sh.expanded);
        assert_eq!(sh.depth_hist[0], 1, "the root");
        assert!(sh.depth_hist.len() as u32 <= sh.max_depth + 1);
        // every node but the root is one edge's child: the edges' children add up to the nodes
        assert_eq!(sh.edges.iter().map(|e| e.kids).sum::<u32>() + 1, sh.nodes);
        for e in sh.edges.iter().chain(&sh.main_line) {
            assert!(e.opp_lists <= e.opp_kids && e.opp_keys <= e.opp_kids && (e.opp_kids == 0) == (e.opp_lists == 0));
            assert!(e.opp_modal_tot <= e.opp_tot && e.opp_tot <= e.n);
        }
        let root_n: u32 = sh.edges.iter().filter(|e| e.depth == 0).map(|e| e.n).sum();
        assert_eq!(root_n, t.tot(t.root));
        let ml = sh.main_line.first().unwrap();
        assert_eq!(ml.n, *t.visits(t.root).iter().max().unwrap());
        assert!(sh.main_line.windows(2).all(|w| w[1].n <= w[0].n));
        checked += 1;
    }
    assert!(checked > 10);
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

#[test]
fn completed_q_prior_w_by_hand() {
    // docs/search.md 3.13: pi ~ P0^beta exp(sigma completedQ). References computed with 50-digit
    // mpmath (E:/mtg-zero-scratch/ideas-1/ref_numbers.py; the Python test uses the same numbers).
    let (p0, n, w) = ([0.5f32, 0.3, 0.2], [6u32, 3, 0], [3.6f64, 2.4, 0.0]);
    let pi = completed_q_policy_w(&p0, &n, &w, 0.5, 50.0, 0.1, 0.5);
    assert!(close(&pi, &[0.2354746509, 0.5590229101, 0.205502439], 1e-6), "{pi:?}");
    let pi = completed_q_policy_w(&p0, &n, &w, 0.5, 50.0, 1.0, 0.7);
    assert!(close(&pi, &[1.954679451e-5, 0.9997228533, 0.0002575999492], 1e-6), "{pi:?}");
    let (p1, n1, w1) = ([0.9f32, 0.08, 0.02], [27u32, 4, 1], [27.0 * 0.40, 4.0 * 0.47, 0.60]);
    let pi = completed_q_policy_w(&p1, &n1, &w1, 0.45, 50.0, 0.3, 0.6);
    assert!(close(&pi, &[0.07987888371, 0.09418779954, 0.8259333168], 1e-6), "{pi:?}");
    // a favourite at 0.97 with a hair more Q than v_mix: beta 0.8 gives the unvisited option back
    // a share of its own (the ratchet of section 3.13)
    let (p2, n2, w2) = ([0.97f32, 0.02, 0.01], [60u32, 3, 0], [60.0 * 0.55, 3.0 * 0.52, 0.0]);
    let pi = completed_q_policy_w(&p2, &n2, &w2, 0.55, 50.0, 0.3, 0.8);
    assert!(close(&pi, &[0.9597958246, 0.01598210311, 0.02422207231], 1e-6), "{pi:?}");
    assert!(pi[0] < completed_q_policy(&p2, &n2, &w2, 0.55, 50.0, 0.3)[0]);
    // nothing visited (every option at v): P0^beta normalised, 0.8 / 0.2 at beta 0.5 -> 2 / 1
    let pi = completed_q_policy_w(&[0.8, 0.2], &[0, 0], &[0.0; 2], 0.5, 50.0, 1.0, 0.5);
    assert!(close(&pi, &[2.0 / 3.0, 1.0 / 3.0], 1e-6), "{pi:?}");
    // 1.0 is completed_q_policy bit for bit, everywhere
    for (p, nn, ww, v, cs) in [
        (&p0[..], &n[..], &w[..], 0.5, 1.0),
        (&p1[..], &n1[..], &w1[..], 0.45, 0.3),
        (&p2[..], &n2[..], &w2[..], 0.55, 0.3),
        (&[1.0f32, 0.0, 0.0][..], &[3u32, 1, 0][..], &[1.5f64, 0.9, 0.0][..], 0.5, 1.0),
    ] {
        assert_eq!(completed_q_policy_w(p, nn, ww, v, 50.0, cs, 1.0), completed_q_policy(p, nn, ww, v, 50.0, cs));
    }
    // the log-ratio of two targets of one node is (beta - 1) ln P0 + a constant: only the prior's
    // term moves, v_mix and sigma do not
    let (a, b) = (completed_q_policy_w(&p1, &n1, &w1, 0.45, 50.0, 0.3, 0.6), completed_q_policy(&p1, &n1, &w1, 0.45, 50.0, 0.3));
    let d: Vec<f64> = (0..3).map(|k| (a[k] as f64).ln() - (b[k] as f64).ln() - (0.6 - 1.0) * (p1[k] as f64).ln()).collect();
    assert!(d.iter().all(|x| (x - d[0]).abs() < 1e-5), "{d:?}");
    // a zero prior stays finite (floored); softer than beta 1 but still ~0 next to a real option
    let pi = completed_q_policy_w(&[1.0, 0.0, 0.0], &[3, 1, 0], &[1.5, 0.9, 0.0], 0.5, 50.0, 1.0, 0.5);
    assert!(pi.iter().all(|x| x.is_finite() && *x >= 0.0) && (pi.iter().sum::<f32>() - 1.0).abs() < 1e-6 && pi[0] > 0.999, "{pi:?}");
    // a single option: 1
    assert_eq!(completed_q_policy_w(&[1.0], &[7], &[3.0], 0.4, 50.0, 1.0, 0.3), vec![1.0]);
    // the tree's target: the formula over the root's statistics; 1.0 = cq_target
    let mut rng = Rng::new(12);
    let g = mid_game(31, 13);
    let p = g.to_move().unwrap();
    let mut t = Tree::new(&g, p, &legal_of(&g), true, &mut rng);
    search(&mut t, &g, 120, &mut rng, &W);
    let r = t.root;
    assert_eq!(t.cq_target_w(r, 50.0, 0.3, 1.0), t.cq_target(r, 50.0, 0.3));
    let want = completed_q_policy_w(&t.priors0(r), &t.visits(r), &t.values(r), t.value0(r), 50.0, 0.3, 0.7);
    assert_eq!(t.cq_target_w(r, 50.0, 0.3, 0.7), want);
}

// ------------------------------------------------------------------ deferred terminals, teams

use crate::toy_team::ToyTeam;

/// A toy_team position `left` picks before the deferred terminal (4 players, 2 rounds).
fn team_position(seed: u64, left: usize) -> ToyTeam {
    let mut rng = Rng::new(seed);
    let mut g = ToyTeam::new(4, 2, seed);
    let mut keys = Vec::new();
    let mut legal = Vec::new();
    for _ in 0..8 - left {
        legal.clear();
        g.legal_actions(&mut legal);
        g.step(legal[rng.below(legal.len() as u32) as usize], &mut keys).unwrap();
    }
    g
}

/// Stand-in evaluation of a toy_team leaf: uniform priors, the decider's value from its team's
/// drafted food lead (relative seats; it sums to 1 over the 4 seats).
fn team_eval(g: &ToyTeam, player: usize, nacts: usize) -> (Vec<f32>, [f32; MAX_SEATS]) {
    let lead: i16 = (0..4).map(|s| if s % 2 == player % 2 { g.bonus(s) } else { -g.bonus(s) }).sum();
    let w = 0.5 / (1.0 + (-(lead as f32) / 3.0).exp());
    let mut v = [0.0; MAX_SEATS];
    for (j, x) in v.iter_mut().enumerate().take(4) {
        *x = if j % 2 == 0 { w } else { 0.5 - w };
    }
    (vec![1.0 / nacts.max(1) as f32; nacts], v)
}

fn legal_of_team(g: &ToyTeam) -> Vec<u16> {
    let mut l = Vec::new();
    g.legal_actions(&mut l);
    l
}

#[test]
fn deferred_terminals_are_leaves_evaluated_once() {
    let mut rng = Rng::new(7);
    for seed in 0..12 {
        for left in [2usize, 3] {
            let g = team_position(seed, left);
            let p = g.to_move().unwrap();
            let mut t = Tree::new(&g, p, &legal_of_team(&g), true, &mut rng);
            assert_eq!(t.team, 2);
            let mut sg = g.clone();
            let mut sc = Scratch::default();
            let (mut deferred_leaves, mut revisits) = (0, 0);
            let mut seen = std::collections::HashSet::new();
            for _ in 0..300 {
                // ToyTeam::result() panics at the deferred terminal: the search never calls it
                match t.simulate(&g, &mut sg, &mut rng, &W, &mut sc).unwrap() {
                    Sim::Leaf { node, player } => {
                        let nacts = t.acts(node).len();
                        if sg.to_move().is_none() {
                            assert!(sg.deferred() && nacts == 0 && t.is_deferred_leaf(node));
                            assert_eq!(player, p, "a deferred leaf is evaluated from the viewer's view");
                            assert!(seen.insert(node), "a deferred node is evaluated once");
                            deferred_leaves += 1;
                        }
                        let (pri, v) = team_eval(&sg, player, nacts);
                        let (util, win) = leaf_values_team(&v, &[0.0; MAX_SEATS], player, 4, 0.0, t.team);
                        // utilities are team x win: Q in [0, 1]
                        assert!((util[p] - 2.0 * win[p]).abs() < 1e-12 && util[p] <= 1.0);
                        t.expand_leaf(node, &pri, &util, &win);
                        t.backup(&sc.path, &util, &win);
                        if nacts == 0 {
                            assert_eq!(t.win(node).unwrap()[..4], win[..4]);
                        }
                    }
                    Sim::Terminal => revisits += 1,
                }
            }
            assert!(deferred_leaves > 0 && revisits > 0, "seed {seed} left {left}: {deferred_leaves} deferred leaves, {revisits} revisits");
            let tot: u32 = t.visits(t.root).iter().sum();
            assert_eq!(tot, 299);
            let w = t.win(t.root).unwrap();
            assert!((w[..4].iter().sum::<f64>() - 1.0).abs() < 1e-6);
            for i in 0..t.acts(t.root).len() {
                let n = t.n_of(t.root, i);
                if n > 0 {
                    let q = t.w_of(t.root, i) / n as f64;
                    assert!((0.0..=1.0).contains(&q), "Q {q}");
                }
            }
        }
    }
}

#[test]
fn team_utilities_and_terminal_values() {
    // leaf values: team 1 is leaf_values bit for bit; team T scales the utility, not the win
    let v = [0.3f32, 0.2, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0];
    let m = [0.5f32, -0.5, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0];
    for mw in [0.0, 0.2] {
        assert_eq!(leaf_values(&v, &m, 1, 4, mw), leaf_values_team(&v, &m, 1, 4, mw, 1));
        let (u2, w2) = leaf_values_team(&v, &m, 1, 4, mw, 2);
        let (u1, w1) = leaf_values(&v, &m, 1, 4, mw);
        assert_eq!(w1, w2);
        for s in 0..4 {
            assert!((u2[s] - (u1[s] + w1[s])).abs() < 1e-12);
        }
    }
    // a resolved toy_team terminal: util = 2 x result
    let mut g = team_position(1, 0);
    assert!(g.deferred());
    g.resolve("[[1, 0], [1, 0], [0.5, 0.5], [0, 1]]").unwrap();
    let (u, w) = terminal_values(&g, 0.0);
    assert_eq!(&w[..4], &[0.5, 0.0, 0.5, 0.0]);
    assert_eq!(&u[..4], &[1.0, 0.0, 1.0, 0.0]);
    let (u, _) = terminal_values(&g, 0.5);
    assert!((u[0] - (1.0 + 0.5 * (0.25f64).tanh())).abs() < 1e-6);
}

// ------------------------------------------------------------------ root minimum visits

use crate::mull_game::{mull_eval, Mull, KEEP, MULL, P_MULL};

/// `search` on the keep-or-mulligan game with its stand-in network (`mull_eval`).
fn mull_search(t: &mut Tree<Mull>, live: &Mull, sims: usize, rng: &mut Rng) {
    let mut g = live.clone();
    let mut sc = Scratch::default();
    for _ in 0..sims {
        if let Sim::Leaf { node, player } = t.simulate(live, &mut g, rng, &W, &mut sc).unwrap() {
            let mut x = [0.0f32; 4];
            g.encode(player, &mut x);
            let (pri, v) = mull_eval(&x, t.acts(node).len());
            let (util, win) = leaf_values(&v, &[0.0; MAX_SEATS], player, 2, 0.0);
            t.expand(node, &pri, &util);
            t.backup(&sc.path, &util, &win);
        }
    }
}

fn mull_tree(h0: u8, h1: u8, k: u32, seed: u64) -> Tree<Mull> {
    let g = Mull::with_hands(h0, h1, seed);
    let mut rng = Rng::new(seed);
    let mut t = Tree::new(&g, 0, &[KEEP, MULL], true, &mut rng);
    t.set_root_min(k);
    mull_search(&mut t, &g, 64, &mut rng);
    t
}

#[test]
fn root_min_visits_reach_a_starved_option() {
    // A 2-option root with priors 0.999 / 0.001 (the Delver mirror's mulligan at 64 simulations):
    // without the floor the 0.001 option gets no visit, its Q is never estimated and the 'cq'
    // target keeps it at its prior; with k = 4 both options get >= 4 visits, the 4 forced walks
    // are counted, and the target moves to what Q says - toward MULL with a bad hand (worth ~0.16
    // more), away from it with a good one.
    let (i_keep, i_mull) = (KEEP as usize, MULL as usize);
    for seed in 0..6u64 {
        for (h0, h1) in [(0u8, 0u8), (0, 1), (1, 0), (1, 1)] {
            let t0 = mull_tree(h0, h1, 0, seed);
            let r = t0.root;
            assert_eq!(t0.priors0(r), vec![1.0 - P_MULL, P_MULL]);
            assert_eq!(t0.visits(r), vec![63, 0], "hands {h0} {h1}: the floor off starves MULL");
            assert!(t0.root_forced().is_empty() && t0.root_visits_pruned() == t0.visits(r));
            assert_eq!(t0.root_win_pruned(), t0.win(r));
            let cq0 = t0.cq_target(r, 50.0, 1.0);
            assert!((cq0[i_mull] - P_MULL).abs() < 1e-4, "an unvisited option's target stays at its prior: {cq0:?}");

            let t = mull_tree(h0, h1, 4, seed);
            let r = t.root;
            let nv = t.visits(r);
            assert_eq!(nv.iter().sum::<u32>(), 63);
            assert!(nv.iter().all(|&n| n >= 4), "hands {h0} {h1}: {nv:?}");
            // forced walks only ever go to the option below the floor, at most k of them
            let f = t.root_forced().to_vec();
            assert_eq!(f[i_keep], 0);
            assert_eq!(f[i_mull], 4, "PUCT never picks MULL before its Q is known: all 4 forced");
            assert_eq!(t.forced_n, 4);
            assert_eq!(t.root_visits_pruned(), vec![nv[0], nv[1] - 4]);
            let q = |i: usize| t.w_of(r, i) / t.n_of(r, i) as f64;
            let cq = t.cq_target(r, 50.0, 1.0);
            assert!((cq.iter().sum::<f32>() - 1.0).abs() < 1e-5 && cq.iter().all(|x| x.is_finite()));
            // every option visited: the target is the softmax of ln P0 + sigma Q (no v_mix)
            let sigma = 50.0 + *nv.iter().max().unwrap() as f64;
            let want = completed_q_policy(&t.priors0(r), &nv, &t.values(r), t.value0(r), 50.0, 1.0);
            assert_eq!(cq, want);
            let l = |i: usize| (t.p0_of(r, i) as f64).ln() + sigma * q(i);
            let e = (l(i_mull) - l(i_keep)).exp();
            assert!((cq[i_mull] as f64 - e / (1.0 + e)).abs() < 1e-5);
            if h0 == 0 {
                assert!(q(i_mull) > q(i_keep) + 0.08, "a bad hand: MULL is better ({} vs {})", q(i_mull), q(i_keep));
                assert!(cq[i_mull] > 0.9, "a bad hand: the target follows Q to MULL: {cq:?}");
            } else {
                assert!(q(i_mull) < q(i_keep) - 0.08, "a good hand: MULL is worse ({} vs {})", q(i_mull), q(i_keep));
                assert!(cq[i_mull] < 1e-3, "a good hand: the target stays on KEEP: {cq:?}");
                // PUCT never chose the worse MULL itself: no visit is left once the forced are
                // pruned, and the root value without the forced walks is KEEP's mean (util = win)
                assert_eq!(t.root_visits_pruned(), vec![nv[0], 0]);
                let w = t.root_win_pruned().unwrap();
                assert!((w[0] - q(i_keep)).abs() < 1e-12, "{} vs {}", w[0], q(i_keep));
                assert!((w[0] + w[1] - 1.0).abs() < 1e-12);
                // ... while the raw root value mixes the 4 forced MULL walks in
                assert!(t.win(r).unwrap()[0] < w[0]);
            }
        }
    }
}

#[test]
fn root_min_visits_on_wider_roots_and_their_accounting() {
    // toy roots of up to 6 options, the hashed priors of `eval`: with k = 3 every option gets
    // >= 3 of 96 visits; the forced walks are at most k per option and add up; advance and reset
    // turn the floor off (the next search sets its own).
    let mut rng = Rng::new(21);
    let mut checked = 0;
    let mut advanced = 0;
    for s in 0..80 {
        let g = mid_game(100 + s, 6 + (s as usize % 30));
        if g.over {
            continue;
        }
        let l = legal_of(&g);
        if !(2..=6).contains(&l.len()) {
            continue;
        }
        let p = g.to_move().unwrap();
        let mut t = Tree::new(&g, p, &l, true, &mut rng);
        t.set_root_min(3);
        search(&mut t, &g, 96, &mut rng, &W);
        let nv = t.visits(t.root);
        assert!(nv.iter().all(|&n| n >= 3), "{nv:?}");
        let f = t.root_forced().to_vec();
        assert_eq!(f.len(), l.len());
        assert!(f.iter().all(|&x| x <= 3));
        assert_eq!(f.iter().sum::<u32>(), t.forced_n);
        let pr = t.root_visits_pruned();
        assert_eq!(pr.iter().sum::<u32>() + t.forced_n, t.tot(t.root));
        assert!(t.walk_forced == NIL, "no walk in flight");
        let i = (0..nv.len()).max_by_key(|&i| nv[i]).unwrap();
        if let Some((key, kid)) = t.kids(t.root, i).into_iter().next() {
            let (pl, acts) = (t.player(kid), t.acts(kid).to_vec());
            if t.advance(l[i], &key, pl, &acts) {
                assert_eq!((t.root_min(), t.root_forced().len(), t.forced_n), (0, 0, 0));
                assert_eq!(t.root_visits_pruned(), t.visits(t.root));
                advanced += 1;
            }
        }
        checked += 1;
    }
    assert!(checked >= 10 && advanced >= 3, "{checked} positions, {advanced} advanced");
    let g = mid_game(3, 10);
    let mut t = Tree::new(&g, g.to_move().unwrap(), &legal_of(&g), true, &mut rng);
    t.set_root_min(2);
    search(&mut t, &g, 30, &mut rng, &W);
    t.reset(&g, g.to_move().unwrap(), &legal_of(&g), true, &mut rng);
    assert_eq!((t.root_min(), t.root_forced().len(), t.forced_n), (0, 0, 0));
}

#[test]
fn root_min_visits_off_changes_no_walk() {
    // set_root_min(0) (what the driver calls for every search the floor does not apply to) is the
    // tree of before: the same visits, values and root win as a tree that never heard of it
    let mut checked = 0;
    for s in 0..12u64 {
        let g = mid_game(200 + s, 8 + s as usize);
        if g.over {
            continue;
        }
        let p = g.to_move().unwrap();
        let (mut ra, mut rb) = (Rng::new(s), Rng::new(s));
        let mut a = Tree::new(&g, p, &legal_of(&g), true, &mut ra);
        let mut b = Tree::new(&g, p, &legal_of(&g), true, &mut rb);
        b.set_root_min(0);
        search(&mut a, &g, 80, &mut ra, &W);
        search(&mut b, &g, 80, &mut rb, &W);
        assert_eq!(a.visits(a.root), b.visits(b.root));
        assert_eq!(a.values(a.root), b.values(b.root));
        assert_eq!(a.win(a.root), b.root_win_pruned());
        checked += 1;
    }
    assert!(checked > 5);
}

// ---------------------------------------------------------------- the root's belief (search.md 3.8)

/// A toy position whose next round's card is still hidden among `n` unrevealed cards of its stage.
fn toy_with_next_card(seed: u64) -> ToyGame {
    let mut s = seed;
    for k in 0.. {
        assert!(k < 10_000, "no toy position with a hidden next card");
        let g = mid_game(s, 3 + (k * 7 + seed as usize) % 40);
        let r = g.round as usize;
        if !g.over && r < 6 && legal_of(&g).len() >= 2 && (if r < 3 { 3 } else { 6 }) - r >= 2 {
            return g;
        }
        s += 1000;
    }
    unreachable!()
}

/// `n` searches of `sims` simulations from `live` with `belief` (None: uniform worlds) and `mix`;
/// -> per card, the share of the leaves whose world deals it as the next round's card.
fn next_card_shares(live: &ToyGame, belief: Option<Belief>, mix: f64, searches: u64, sims: usize) -> [f64; 6] {
    let r = live.round as usize;
    let mut cnt = [0u64; 6];
    let mut tot = 0u64;
    let w = Walk { c_puct: 1.0, chance_k: 0, margin_w: 0.0 };
    for s in 0..searches {
        let mut rng = Rng::new(900 + s);
        let mut t = Tree::new(live, live.to_move().unwrap(), &legal_of(live), true, &mut rng);
        t.set_belief(belief.clone(), mix, live, &mut rng);
        let mut g = live.clone();
        let mut sc = Scratch::default();
        for _ in 0..sims {
            if let Sim::Leaf { node, player } = t.simulate(live, &mut g, &mut rng, &w, &mut sc).unwrap() {
                // (the world of this walk: its next round's card, unless the walk already revealed it)
                cnt[g.order[r] as usize] += 1;
                tot += 1;
                let (pri, v, m) = eval(&g, player, t.acts(node));
                let (util, win) = leaf_values(&v, &m, player, t.n, 0.0);
                t.expand(node, &pri, &util);
                t.backup(&sc.path, &util, &win);
            }
        }
    }
    let mut out = [0.0; 6];
    for c in 0..6 {
        out[c] = cnt[c] as f64 / tot.max(1) as f64;
    }
    out
}

#[test]
fn belief_weight_formula() {
    use crate::game::belief_weight;
    // the ratio to the uniform deal to the tilt, clipped to [1/R, R]
    assert!((belief_weight(0.6, 0.3, 1.0, 20.0) - 2.0).abs() < 1e-12);
    assert!((belief_weight(0.6, 0.3, 0.5, 20.0) - 2f64.sqrt()).abs() < 1e-12);
    assert_eq!(belief_weight(0.0, 0.3, 1.0, 20.0), 0.05, "a card the belief rules out: 1/R, never 0");
    assert_eq!(belief_weight(-0.2, 0.3, 1.0, 20.0), 0.05);
    assert_eq!(belief_weight(9.0, 0.3, 1.0, 20.0), 20.0, "clipped at R");
    assert_eq!(belief_weight(0.3, 0.0, 1.0, 4.0), 4.0, "no uniform mass: eps, clipped");
    assert_eq!(belief_weight(0.3, 0.3, 3.0, 1.0), 1.0, "R 1: every weight 1");
}

#[test]
fn a_belief_deals_the_worlds_by_its_weights() {
    // the toy's next round's card (its aux target) among n unrevealed cards of its stage: uniform 1/n
    // each without a belief; with a belief on one card c at tilt 1 and a large clip, c's weight is
    // (p_c / (1/n)) against the others' 1/R floors: the deal follows the belief
    let live = toy_with_next_card(3);
    let r = live.round as usize;
    let hi = if r < 3 { 3 } else { 6 };
    let pool: Vec<u8> = live.order[r..hi].to_vec();
    assert!(pool.len() >= 2, "{pool:?}");
    let c = pool[1] as usize;
    let mut pred = vec![0.0f32; 6];
    pred[c] = 1.0;
    let uni = next_card_shares(&live, None, 0.0, 20, 40);
    for &k in &pool {
        assert!((uni[k as usize] - 1.0 / pool.len() as f64).abs() < 0.08, "uniform deal {uni:?} over {pool:?}");
    }
    let sharp = next_card_shares(&live, Some(Belief { pred: pred.clone(), tilt: 1.0, clip: 1000.0, tilts: Vec::new() }), 0.0, 20, 40);
    assert!(sharp[c] > 0.98, "a sharp belief: {sharp:?} (card {c})");
    // the clip bounds it: R = 2 at most 2 / (2 + (n - 1) / 2) of the worlds
    let n = pool.len() as f64;
    let clipped = next_card_shares(&live, Some(Belief { pred: pred.clone(), tilt: 1.0, clip: 2.0, tilts: Vec::new() }), 0.0, 20, 40);
    let want = 2.0 / (2.0 + (n - 1.0) * 0.5);
    assert!((clipped[c] - want).abs() < 0.06, "clip 2: {} vs {want}", clipped[c]);
    // the uniform share: mix 0.5 deals half the worlds uniformly
    let mixed = next_card_shares(&live, Some(Belief { pred, tilt: 1.0, clip: 1000.0, tilts: Vec::new() }), 0.5, 20, 40);
    let want = 0.5 + 0.5 / n;
    assert!((mixed[c] - want).abs() < 0.06, "mix 0.5: {} vs {want}", mixed[c]);
}

#[test]
fn no_belief_is_the_tree_of_before_and_advance_clears_it() {
    // set_belief(None) changes no draw: the same visits as a tree never given one
    let live = toy_with_next_card(5);
    let w = Walk { c_puct: 1.0, chance_k: 0, margin_w: 0.0 };
    let mut r1 = Rng::new(42);
    let mut t1 = Tree::new(&live, live.to_move().unwrap(), &legal_of(&live), true, &mut r1);
    search(&mut t1, &live, 64, &mut r1, &w);
    let mut r2 = Rng::new(42);
    let mut t2 = Tree::new(&live, live.to_move().unwrap(), &legal_of(&live), true, &mut r2);
    t2.set_belief(None, 0.3, &live, &mut r2);
    assert!(t2.belief().is_none() && t2.belief_mix() == 0.0);
    search(&mut t2, &live, 64, &mut r2, &w);
    assert_eq!(t1.visits(t1.root), t2.visits(t2.root));
    assert_eq!(t1.values(t1.root), t2.values(t2.root));
    // a belief belongs to one decision: advance and reset clear it
    let mut rng = Rng::new(1);
    let mut t = Tree::new(&live, live.to_move().unwrap(), &legal_of(&live), true, &mut rng);
    t.set_belief(Some(Belief { pred: vec![1.0 / 6.0; 6], tilt: 1.0, clip: 20.0, tilts: Vec::new() }), 0.0, &live, &mut rng);
    search(&mut t, &live, 64, &mut rng, &w);
    assert!(t.belief().is_some());
    let a = t.acts(t.root)[0];
    let mut g = live.clone();
    let (mut key, mut legal) = (Vec::new(), Vec::new());
    transition(&mut g, a, None, &mut key, &mut legal).unwrap();
    if let Some(p) = g.to_move() {
        if t.advance(a, &key, p, &legal) {
            assert!(t.belief().is_none(), "advance keeps no belief");
        }
    }
    t.set_belief(Some(Belief { pred: vec![1.0 / 6.0; 6], tilt: 1.0, clip: 20.0, tilts: Vec::new() }), 0.0, &live, &mut rng);
    t.reset(&live, live.to_move().unwrap(), &legal_of(&live), true, &mut rng);
    assert!(t.belief().is_none(), "reset keeps no belief");
}

#[test]
fn without_redet_the_belief_deals_the_decisions_one_world_again() {
    let live = toy_with_next_card(7);
    let r = live.round as usize;
    let hi = if r < 3 { 3 } else { 6 };
    let c = live.order[hi - 1] as usize;
    let mut pred = vec![0.0f32; 6];
    pred[c] = 1.0;
    let mut hits = 0;
    for s in 0..200u64 {
        let mut rng = Rng::new(s);
        let mut t = Tree::new(&live, live.to_move().unwrap(), &legal_of(&live), false, &mut rng);
        t.set_belief(Some(Belief { pred: pred.clone(), tilt: 1.0, clip: 1000.0, tilts: Vec::new() }), 0.0, &live, &mut rng);
        let mut g = live.clone();
        let mut sc = Scratch::default();
        let w = Walk { c_puct: 1.0, chance_k: 0, margin_w: 0.0 };
        // (the one world: every walk clones it)
        if let Sim::Leaf { .. } = t.simulate(&live, &mut g, &mut rng, &w, &mut sc).unwrap() {
            hits += (g.order[r] as usize == c) as u32;
        }
    }
    assert!(hits > 195, "{hits} of 200 worlds dealt the believed card");
}

// ------------------------------------------------------------------ the Gumbel root search

/// The visit counts sequential halving leaves on `m` considered options after `n` walks of its
/// schedule (walk j raises one option holding `seq[j]` walks), sorted, largest first.
fn halving_counts(m: u32, n: u32) -> Vec<u32> {
    let mut c = vec![0u32; m as usize];
    for v in gumbel_sequence(m, n) {
        let i = c.iter().position(|&x| x == v).expect("the schedule asks for a count no option holds");
        c[i] += 1;
    }
    c.sort_unstable_by(|a, b| b.cmp(a));
    c
}

#[test]
fn gumbel_sequence_is_mctxs() {
    // mctx get_sequence_of_considered_visits (seq_halving.py), worked by hand
    assert_eq!(gumbel_sequence(4, 8), vec![0, 0, 0, 0, 1, 1, 2, 2]);
    assert_eq!(gumbel_sequence(2, 8), vec![0, 0, 1, 1, 2, 2, 3, 3]);
    assert_eq!(gumbel_sequence(3, 7), vec![0, 0, 0, 1, 1, 2, 2]);
    assert_eq!(gumbel_sequence(1, 4), vec![0, 1, 2, 3]);
    assert_eq!(gumbel_sequence(0, 3), vec![0, 1, 2]);
    assert_eq!(gumbel_sequence(16, 4), vec![0; 4]);
    assert!(gumbel_sequence(5, 0).is_empty());
    for (m, n) in [(16, 64), (16, 63), (4, 63), (2, 63), (5, 16), (16, 15), (7, 200)] {
        assert_eq!(gumbel_sequence(m, n).len(), n as usize);
    }
    // 16 options, 64 walks: rounds of 1, 2, 4, 8 walks per option on 16, 8, 4, 2 options
    let mut want = vec![15, 15, 7, 7, 3, 3, 3, 3];
    want.extend([1; 8]);
    assert_eq!(halving_counts(16, 64), want);
    // 2 options (the mulligan): 63 walks split 32 / 31
    assert_eq!(halving_counts(2, 63), vec![32, 31]);
}

/// A Gumbel search of `sims` simulations (a fresh root: `sims - 1` root walks).
fn gumbel_tree(g: &ToyGame, sims: usize, m: u32, noise: &[f64], seed: u64) -> Tree<ToyGame> {
    let mut rng = Rng::new(seed);
    let p = g.to_move().unwrap();
    let mut t = Tree::new(g, p, &legal_of(g), true, &mut rng);
    t.set_gumbel(m, sims as u32 - 1, noise, 50.0, 0.3);
    search(&mut t, g, sims, &mut rng, &W);
    t
}

#[test]
fn gumbel_root_walks_follow_sequential_halving() {
    // toy roots of 3-12 options, 64 simulations, m 4 and 16: the root's walks are exactly the
    // schedule's counts (the first round on the top options by g + ln P0: the Gumbel-top-k sample),
    // the move is the best score among the most walked options, and nothing but the root changed
    // (the tree's own walks below it are PUCT's)
    let mut rng = Rng::new(5);
    let mut checked = 0;
    for s in 0..60u64 {
        let g = mid_game(300 + s, 4 + (s as usize % 40));
        if g.over || legal_of(&g).len() < 3 {
            continue;
        }
        let k = legal_of(&g).len();
        for m in [4u32, 16] {
            let noise: Vec<f64> = (0..k).map(|_| crate::noise::gumbel(&mut rng)).collect();
            let t = gumbel_tree(&g, 64, m, &noise, s);
            let r = t.root;
            let nv = t.visits(r);
            assert_eq!(t.gumbel_visits(), nv.as_slice(), "a fresh root: every visit is this search's");
            assert_eq!(nv.iter().sum::<u32>(), 63);
            let c = m.min(k as u32);
            let mut sorted = nv.clone();
            sorted.sort_unstable_by(|a, b| b.cmp(a));
            assert_eq!(sorted[..c as usize], halving_counts(c, 63)[..], "m {m}, {k} options: {nv:?}");
            assert!(sorted[c as usize..].iter().all(|&x| x == 0));
            // the considered options: the top c by g + ln P0
            let p0 = t.priors0(r);
            let mut by: Vec<usize> = (0..k).collect();
            by.sort_by(|&a, &b| (noise[b] + (p0[b] as f64).ln()).partial_cmp(&(noise[a] + (p0[a] as f64).ln())).unwrap());
            let mut top: Vec<usize> = by[..c as usize].to_vec();
            top.sort_unstable();
            let visited: Vec<usize> = (0..k).filter(|&i| nv[i] > 0).collect();
            assert_eq!(visited, top, "m {m}: the first round visits the Gumbel-top-{c}");
            // the move: the best g + completed-Q logit among the most walked
            let lg = completed_q_logits(&p0, &nv, &t.values(r), t.value0(r), 50.0, 0.3);
            let mx = *nv.iter().max().unwrap();
            let a = t.gumbel_choose().unwrap();
            assert_eq!(nv[a], mx);
            assert!((0..k).filter(|&i| nv[i] == mx).all(|i| lg[i] + noise[i] <= lg[a] + noise[a]));
            // the root value: the improved policy's over the walked options
            let pi = completed_q_policy(&p0, &nv, &t.values(r), t.value0(r), 50.0, 0.3);
            let q = t.gumbel_q().unwrap();
            let gw = t.gumbel_win();
            let tot: f64 = (0..k).filter(|&i| nv[i] > 0).map(|i| pi[i] as f64).sum();
            for j in 0..2 {
                let want: f64 = (0..k).filter(|&i| nv[i] > 0).map(|i| pi[i] as f64 * gw[i][j] / nv[i] as f64).sum::<f64>() / tot;
                assert!((q[j] - want).abs() < 1e-6, "q {q:?} want {want}");
            }
            assert!((q[0] + q[1] - 1.0).abs() < 1e-6, "{q:?}");
            checked += 1;
        }
    }
    assert!(checked >= 30, "{checked} roots");
}

#[test]
fn gumbel_off_changes_no_walk_and_reuse_budgets_this_search() {
    // clear_gumbel (and never setting it) is the tree of before; advance and reset turn it off
    let mut checked = 0;
    for s in 0..12u64 {
        let g = mid_game(500 + s, 8 + s as usize);
        if g.over {
            continue;
        }
        let p = g.to_move().unwrap();
        let (mut ra, mut rb) = (Rng::new(s), Rng::new(s));
        let mut a = Tree::new(&g, p, &legal_of(&g), true, &mut ra);
        let mut b = Tree::new(&g, p, &legal_of(&g), true, &mut rb);
        b.set_gumbel(16, 79, &[], 50.0, 1.0);
        b.clear_gumbel();
        search(&mut a, &g, 80, &mut ra, &W);
        search(&mut b, &g, 80, &mut rb, &W);
        assert_eq!((a.visits(a.root), a.values(a.root), a.win(a.root)), (b.visits(b.root), b.values(b.root), b.win(b.root)));
        assert!(!b.gumbel_on() && b.gumbel_choose().is_none() && b.gumbel_q().is_none());
        checked += 1;
    }
    assert!(checked > 5);
    // a reused root: its earlier visits stay (Q, sigma) and the schedule counts this search's walks
    let mut rng = Rng::new(77);
    let mut reused = 0;
    for s in 0..80u64 {
        let g = mid_game(700 + s, 3 + (s as usize % 30));
        if g.over || legal_of(&g).len() < 2 {
            continue;
        }
        let mut t = gumbel_tree(&g, 64, 16, &[], s);
        let (r, nv) = (t.root, t.visits(t.root));
        let i = t.gumbel_choose().unwrap();
        let a = legal_of(&g)[i];
        // the same seat deciding again after the move, with the child's chance key
        for (key, kid) in t.kids(r, i) {
            if t.player(kid) != t.viewer || !t.expanded(kid) || t.acts(kid).len() < 2 {
                continue;
            }
            let (pl, acts) = (t.player(kid), t.acts(kid).to_vec());
            assert!(t.advance(a, &key, pl, &acts));
            assert!(!t.gumbel_on(), "advance turns the Gumbel search off");
            let before = t.visits(t.root);
            assert!(before.iter().sum::<u32>() > 0);
            // replay the game to the child: the toy's step with the key forced
            let mut h = g.clone();
            let mut keys = Vec::new();
            let mut legal = Vec::new();
            crate::tree::transition(&mut h, a, Some(&key), &mut keys, &mut legal).unwrap();
            if keys != key || h.to_move() != Some(pl) || legal != acts {
                break;
            }
            t.set_gumbel(16, 32, &[], 50.0, 0.3);
            search(&mut t, &h, 32, &mut rng, &W);
            let after = t.visits(t.root);
            let gv = t.gumbel_visits().to_vec();
            assert_eq!(gv.iter().sum::<u32>(), 32, "an expanded root: every simulation is a root walk");
            for j in 0..after.len() {
                assert_eq!(after[j], before[j] + gv[j]);
            }
            let c = 16.min(acts.len() as u32);
            let mut sorted = gv.clone();
            sorted.sort_unstable_by(|x, y| y.cmp(x));
            assert_eq!(sorted[..c as usize], halving_counts(c, 32)[..]);
            reused += 1;
            break;
        }
        let _ = nv;
    }
    assert!(reused >= 3, "{reused} reused roots");
    let g = mid_game(3, 10);
    let mut t = gumbel_tree(&g, 20, 4, &[], 1);
    assert!(t.gumbel_on());
    t.reset(&g, g.to_move().unwrap(), &legal_of(&g), true, &mut rng);
    assert!(!t.gumbel_on());
}

#[test]
fn gumbel_measures_the_starved_mulligan() {
    // the keep-or-mulligan root at 64 simulations, KEEP's prior 0.999: PUCT never visits MULL; the
    // Gumbel root (2 options <= m) walks both, 32 / 31, so Q decides: with sigma (50 + 32) x 1.0, a
    // bad hand's +0.16 outweighs the log-prior gap ln 999 = 6.9 - the move and the 'cq' target go to
    // MULL - and a good hand's -0.14 keeps; at c_scale 0.3 (24.6 logits per unit of Q) neither moves:
    // the Delver line's scale needs a Q gap of 0.28 against this prior
    let (i_keep, i_mull) = (KEEP as usize, MULL as usize);
    for seed in 0..6u64 {
        for (h0, h1) in [(0u8, 0u8), (0, 1), (1, 0), (1, 1)] {
            for cs in [1.0, 0.3] {
                let g = Mull::with_hands(h0, h1, seed);
                let mut rng = Rng::new(seed);
                let mut t = Tree::new(&g, 0, &[KEEP, MULL], true, &mut rng);
                t.set_gumbel(16, 63, &[], 50.0, cs);
                mull_search(&mut t, &g, 64, &mut rng);
                let r = t.root;
                let mut nv = t.visits(r);
                nv.sort_unstable();
                assert_eq!(nv, vec![31, 32]);
                let q = |i: usize| t.w_of(r, i) / t.n_of(r, i) as f64;
                let cq = t.cq_target(r, 50.0, cs);
                let mv = t.gumbel_choose().unwrap();
                if h0 == 0 && cs == 1.0 {
                    assert!(q(i_mull) > q(i_keep) + 0.08);
                    assert!(cq[i_mull] > 0.9 && mv == i_mull, "a bad hand: {cq:?}, move {mv}");
                } else {
                    assert!(mv == i_keep, "hands {h0} {h1}, c_scale {cs}: move {mv}, cq {cq:?}");
                }
                if h0 == 1 {
                    // a good hand: pi' is all KEEP, so the root value is KEEP's (the raw root mean mixes
                    // 31 MULL walks in)
                    let w = t.gumbel_q().unwrap();
                    assert!((w[0] - q(i_keep)).abs() < 1e-4, "{} vs {}", w[0], q(i_keep));
                    assert!(t.win(r).unwrap()[0] < w[0] - 0.04);
                }
            }
        }
    }
}

// ------------------------------------------------------------------ information-set merging (docs/search.md 3.16)

use crate::counter_game::{counter_eval, Counter, ALLOW, CAST, DAZE, FORCE, HOLD};

/// A backed-up walk: its path, utility and win vectors.
type Backup = (Vec<(u32, u32)>, [f64; MAX_SEATS], [f64; MAX_SEATS]);

/// A row's stand-in network for a game: (priors over `opts`, value relative to `viewer`).
type Net<'a, G> = &'a dyn Fn(&G, usize, &[u16]) -> (Vec<f32>, [f32; MAX_SEATS]);

fn counter_net(g: &Counter, viewer: usize, opts: &[u16]) -> (Vec<f32>, [f32; MAX_SEATS]) {
    let mut x = [0.0f32; 8];
    g.encode(viewer, &mut x);
    counter_eval(&x, opts)
}

fn toy_net(g: &ToyGame, viewer: usize, opts: &[u16]) -> (Vec<f32>, [f32; MAX_SEATS]) {
    let (pri, v, _) = eval(g, viewer, opts);
    let pri = if opts.is_empty() { Vec::new() } else { pri };
    (pri, v)
}

/// `sims` simulations with root-view leaf values (the driver's `LeafView::Root`): a leaf's priors from its
/// decider's row over `leaf_acts`, its value from the viewer's row; every backup appended to `log`. -> terminal walks.
fn rv_search<G: SearchGame>(t: &mut Tree<G>, live: &G, sims: usize, rng: &mut Rng, w: &Walk, net: Net<G>, log: &mut Vec<Backup>) -> usize {
    let mut g = live.clone();
    let mut sc = Scratch::default();
    let mut terminals = 0;
    for _ in 0..sims {
        match t.simulate(live, &mut g, rng, w, &mut sc).unwrap() {
            Sim::Terminal => {
                terminals += 1;
                let (util, win) = terminal_values(&g, w.margin_w);
                log.push((sc.path.clone(), util, win));
            }
            Sim::Leaf { node, player } => {
                let (pri, _) = net(&g, player, t.leaf_acts(node));
                assert_eq!(pri.len(), t.leaf_acts(node).len());
                let (_, v) = net(&g, t.viewer, &[]);
                let (util, win) = leaf_values(&v, &[0.0; MAX_SEATS], t.viewer, t.n, w.margin_w);
                t.expand_leaf(node, &pri, &util, &win);
                t.backup(&sc.path, &util, &win);
                log.push((sc.path.clone(), util, win));
            }
        }
    }
    terminals
}

/// The statistics of a fresh tree recounted from its backed-up walks: N and W per edge, tot and wsum per node,
/// bit for bit (the same additions in the same order); no walk holds a node twice; every expanded node's N add up
/// to its tot and its W to its decider's backed-up wins (margin 0, teams of 1); an other seat's merged node never
/// counts an option's visits above the walks it was available in, nor those above the node's.
fn recount<G: SearchGame>(t: &Tree<G>, log: &[Backup]) {
    use std::collections::HashMap;
    let mut n: HashMap<(u32, u32), u32> = HashMap::new();
    let mut w: HashMap<(u32, u32), f64> = HashMap::new();
    let mut tot: HashMap<u32, u32> = HashMap::new();
    let mut wsum: HashMap<u32, [f64; MAX_SEATS]> = HashMap::new();
    for (path, util, win) in log {
        let mut seen = std::collections::HashSet::new();
        for &(nd, i) in path {
            assert!(seen.insert(nd), "a walk held node {nd} twice: {path:?}");
            *n.entry((nd, i)).or_insert(0) += 1;
            *w.entry((nd, i)).or_insert(0.0) += util[t.player(nd)];
            *tot.entry(nd).or_insert(0) += 1;
            let s = wsum.entry(nd).or_insert([0.0; MAX_SEATS]);
            for j in 0..MAX_SEATS {
                s[j] += win[j];
            }
        }
    }
    for nd in 0..t.n_nodes() as u32 {
        if !t.expanded(nd) || t.is_deferred_leaf(nd) {
            continue;
        }
        let k = t.acts(nd).len();
        let mut sn = 0;
        let mut sw = 0.0;
        for i in 0..k {
            let key = (nd, i as u32);
            assert_eq!(t.n_of(nd, i), n.get(&key).copied().unwrap_or(0), "N of node {nd} option {i}");
            assert_eq!(t.w_of(nd, i).to_bits(), w.get(&key).copied().unwrap_or(0.0).to_bits(), "W of node {nd} option {i}");
            sn += t.n_of(nd, i);
            sw += t.w_of(nd, i);
        }
        let nt = tot.get(&nd).copied().unwrap_or(0);
        assert_eq!(t.tot(nd), nt, "tot of node {nd}");
        assert_eq!(sn, nt, "node {nd}: N add up to tot");
        if nt > 0 {
            let ws = wsum[&nd];
            let mine = t.win(nd).unwrap();
            for j in 0..t.n {
                assert!((mine[j] - ws[j] / nt as f64).abs() < 1e-12, "wsum of node {nd}");
            }
            assert!((sw - ws[t.player(nd)]).abs() < 1e-9, "node {nd}: W add up to its decider's wins");
        }
        if t.merge_on() && t.player(nd) != t.viewer {
            let av = t.avail(nd);
            assert_eq!(av.len(), k);
            for i in 0..k {
                assert!(av[i] >= t.n_of(nd, i) && av[i] <= t.tot(nd), "node {nd} option {i}: avail {} N {} tot {}", av[i], t.n_of(nd, i), t.tot(nd));
            }
        }
    }
}

/// A canonical dump of the tree under its root (breadth first through `kids`): per node its decider, options,
/// backups, win vector and network vector, per option N, W, P, P0, availability, sampled keys and links (key, the
/// child's dump index) - equal for two trees iff they hold the same search, node ids aside.
fn dump<G: SearchGame>(t: &Tree<G>) -> Vec<String> {
    use std::collections::HashMap;
    let mut idx: HashMap<u32, usize> = HashMap::new();
    let mut order = vec![t.root];
    idx.insert(t.root, 0);
    let mut out = Vec::new();
    let mut k = 0;
    while k < order.len() {
        let nd = order[k];
        k += 1;
        let mut s = format!("p{} acts{:?} tot{} win{:?} vec{:?} av{:?}", t.player(nd), t.acts(nd), t.tot(nd), t.win(nd), t.vec_of(nd), t.avail(nd));
        if t.expanded(nd) {
            let pr = t.priors(nd);
            for i in 0..t.acts(nd).len() {
                s += &format!(" |{i}: n{} w{} p{} p0{} outs{:?}", t.n_of(nd, i), t.w_of(nd, i).to_bits(), pr[i].to_bits(), t.p0_of(nd, i).to_bits(), t.outs(nd, i));
                for (key, c) in t.kids(nd, i) {
                    let next = order.len();
                    let j = *idx.entry(c).or_insert_with(|| {
                        order.push(c);
                        next
                    });
                    s += &format!(" {key:?}->{j}");
                }
            }
        }
        out.push(s);
    }
    out
}

/// Seat 0's node of the next round under (node, option i) of a counter tree (a child decided by seat 0).
fn seat0_kid(t: &Tree<Counter>, node: u32, i: usize) -> Option<u32> {
    t.kids(node, i).into_iter().map(|(_, c)| c).find(|&c| t.player(c) == 0)
}

#[test]
fn set_merge_takes_a_fresh_tree_and_reset_turns_it_off() {
    let g = Counter::with_hand(true, false);
    let mut rng = Rng::new(1);
    let mut t = Tree::new(&g, 0, &[CAST, HOLD], true, &mut rng);
    assert!(!t.merge_on() && t.merge_stats().is_none());
    t.set_merge(true).unwrap();
    assert!(t.merge_on());
    let mut log = Vec::new();
    rv_search(&mut t, &g, 10, &mut rng, &W, &counter_net, &mut log);
    assert!(t.set_merge(false).is_err(), "a tree with a search in it");
    assert!(t.set_merge(true).is_ok(), "already on: nothing to do");
    t.reset(&g, 0, &[CAST, HOLD], true, &mut rng);
    assert!(!t.merge_on(), "reset: the tree of before");
}

#[test]
fn the_counter_game_merges_the_other_seats_options_and_the_next_round() {
    // Seat 0 casts or holds; seat 1 answers a cast with ALLOW plus the counters its sampled hand holds (auto-stepped
    // when it holds none). The tree of before keys seat 1's answer by its option list - three nodes under CAST - and
    // seat 0's next round under each answer and under the auto-stepped world. Merged: one seat-1 node over the union
    // {ALLOW, DAZE, FORCE}, its options available in their worlds only, and one seat-0 node of "cast resolved, round
    // 2" reached from the root's CAST (depth 1) and from seat 1's ALLOW (depth 2).
    for seed in 0..8u64 {
        let g = Counter::with_hand(seed % 2 == 0, seed % 3 == 0);
        let mut rng = Rng::new(seed);
        let mut t = Tree::new(&g, 0, &[CAST, HOLD], true, &mut rng);
        t.set_merge(true).unwrap();
        let mut log = Vec::new();
        rv_search(&mut t, &g, 300, &mut rng, &W, &counter_net, &mut log);
        recount(&t, &log);
        let ms = t.merge_stats().unwrap();
        assert!(ms.merged > 0 && ms.growths > 0, "{ms:?}");
        let r = t.root;
        let ci = t.acts(r).iter().position(|&a| a == CAST).unwrap();
        let kids = t.kids(r, ci);
        let opp: Vec<u32> = kids.iter().map(|k| k.1).filter(|&c| t.player(c) == 1).collect();
        assert!(opp.len() >= 3 && opp.windows(2).all(|w| w[0] == w[1]), "every seat-1 link under CAST leads to one node: {kids:?}");
        let o = opp[0];
        let mut union = t.acts(o).to_vec();
        union.sort_unstable();
        assert_eq!(union, vec![ALLOW, DAZE, FORCE]);
        // availability: ALLOW in every world, each counter in about 2/3 of the worlds reaching the node (1/2 of the
        // deals hold it, 3/4 reach the node)
        let av = t.avail(o);
        let ai = t.acts(o).iter().position(|&a| a == ALLOW).unwrap();
        assert_eq!(av[ai], t.tot(o));
        for c in [DAZE, FORCE] {
            let j = t.acts(o).iter().position(|&a| a == c).unwrap();
            let share = av[j] as f64 / t.tot(o) as f64;
            assert!((0.45..0.9).contains(&share), "{c}: available in {share} of the walks at the node");
        }
        // seat 0's next round after a resolved cast: the auto-stepped world's link under CAST and seat 1's ALLOW
        let x0 = seat0_kid(&t, r, ci).expect("the auto-stepped world");
        let x1 = seat0_kid(&t, o, ai).expect("seat 1 allowed");
        assert_eq!(x0, x1, "one information set of seat 0, two parents at two depths");
        assert!(t.in_links()[&x0] >= 2);
        // the unmerged tree: seat 1's answers split by option list, seat 0's next round split with them
        let mut rng = Rng::new(seed);
        let mut u = Tree::new(&g, 0, &[CAST, HOLD], true, &mut rng);
        let mut log = Vec::new();
        rv_search(&mut u, &g, 300, &mut rng, &W, &counter_net, &mut log);
        recount(&u, &log);
        let ukids = u.kids(u.root, ci);
        let uopp: Vec<u32> = ukids.iter().map(|k| k.1).filter(|&c| u.player(c) == 1).collect();
        assert!(uopp.len() >= 3);
        let mut xs: Vec<u32> = uopp
            .iter()
            .filter_map(|&o| {
                let ai = u.acts(o).iter().position(|&a| a == ALLOW)?;
                seat0_kid(&u, o, ai)
            })
            .collect();
        xs.extend(seat0_kid(&u, u.root, ci));
        xs.sort_unstable();
        xs.dedup();
        assert!(xs.len() >= 3, "the tree of before splits seat 0's next round: {xs:?}");
        assert!(t.n_nodes() < u.n_nodes(), "merged {} nodes, unmerged {}", t.n_nodes(), u.n_nodes());
        // the merged node of seat 0's next round has the visits of every world that reached it
        let ux: u32 = xs.iter().map(|&x| u.tot(x)).sum();
        assert!(t.tot(x0) as f64 > 0.6 * ux as f64);
    }
}

#[test]
fn merged_trees_recount_exactly_with_chance_noise_floor_and_gumbel() {
    // fresh merged searches on the toy (chance keys, revelations) and the counter game (growth leaves), PUCT,
    // root Dirichlet noise, the root floor and the Gumbel root: every statistic recounted from the backups
    let mut checked = 0;
    for s in 0..12u64 {
        for mode in 0..4 {
            let mut rng = Rng::new(100 + s * 4 + mode);
            let mut log = Vec::new();
            if s % 2 == 0 {
                let g = mid_game(s, 6 + s as usize);
                if g.over {
                    continue;
                }
                let p = g.to_move().unwrap();
                let mut t = Tree::new(&g, p, &legal_of(&g), true, &mut rng);
                t.set_merge(true).unwrap();
                rv_search(&mut t, &g, 1, &mut rng, &W, &toy_net, &mut log);
                log.clear();
                let k = t.acts(t.root).len();
                match mode {
                    1 => t.add_noise(t.root, 0.3, 0.25, &mut rng),
                    2 => t.set_root_min(3),
                    3 => t.set_gumbel(4, 150, &(0..k).map(|i| i as f64 * 0.1).collect::<Vec<_>>(), 50.0, 1.0),
                    _ => {}
                }
                let w = Walk { c_puct: 1.0, chance_k: (s % 3) as u32, margin_w: 0.0 };
                rv_search(&mut t, &g, 149, &mut rng, &w, &toy_net, &mut log);
                recount(&t, &log);
                if mode == 3 {
                    assert_eq!(t.gumbel_visits().iter().sum::<u32>(), 149);
                }
                if mode == 2 && k <= 49 {
                    assert!(t.visits(t.root).iter().all(|&n| n >= 3));
                }
            } else {
                let g = Counter::with_hand(s % 4 == 1, s % 3 == 1);
                let mut t = Tree::new(&g, 0, &[CAST, HOLD], true, &mut rng);
                t.set_merge(true).unwrap();
                rv_search(&mut t, &g, 1, &mut rng, &W, &counter_net, &mut log);
                log.clear();
                match mode {
                    1 => t.add_noise(t.root, 0.3, 0.25, &mut rng),
                    2 => t.set_root_min(40),
                    3 => t.set_gumbel(2, 200, &[0.3, -0.2], 50.0, 1.0),
                    _ => {}
                }
                rv_search(&mut t, &g, 199, &mut rng, &W, &counter_net, &mut log);
                recount(&t, &log);
                assert!(t.merge_stats().unwrap().growths > 0);
            }
            checked += 1;
        }
    }
    assert!(checked >= 40, "{checked}");
}

#[test]
fn merged_compact_copies_each_node_once_and_searches_on_identically() {
    for s in 0..6u64 {
        let g = Counter::with_hand(s % 4 == 0, s % 3 != 1);
        let mut a = Tree::new(&g, 0, &[CAST, HOLD], true, &mut Rng::new(s));
        a.set_merge(true).unwrap();
        let mut rng = Rng::new(7 + s);
        let mut log = Vec::new();
        rv_search(&mut a, &g, 120, &mut rng, &W, &counter_net, &mut log);
        let mut b = a.clone();
        b.compact();
        assert_eq!(dump(&a), dump(&b));
        assert!(b.n_nodes() <= a.n_nodes());
        assert_eq!(a.in_links().values().sum::<u32>(), b.in_links().values().sum::<u32>());
        // the same walks afterwards (the map was rebuilt: the same merges)
        let (mut ra, mut rb) = (rng.clone(), rng.clone());
        rv_search(&mut a, &g, 80, &mut ra, &W, &counter_net, &mut Vec::new());
        rv_search(&mut b, &g, 80, &mut rb, &W, &counter_net, &mut Vec::new());
        assert_eq!(dump(&a), dump(&b));
        assert_eq!(a.merge_stats().unwrap().merged, b.merge_stats().unwrap().merged);
    }
    // the toy (chance keys and outs)
    for s in 0..6u64 {
        let g = mid_game(s, 5);
        if g.over {
            continue;
        }
        let p = g.to_move().unwrap();
        let mut rng = Rng::new(s);
        let mut a = Tree::new(&g, p, &legal_of(&g), true, &mut rng);
        a.set_merge(true).unwrap();
        let w = Walk { c_puct: 1.0, chance_k: 2, margin_w: 0.0 };
        rv_search(&mut a, &g, 150, &mut rng, &w, &toy_net, &mut Vec::new());
        let mut b = a.clone();
        b.compact();
        assert_eq!(dump(&a), dump(&b));
        let (mut ra, mut rb) = (rng.clone(), rng.clone());
        rv_search(&mut a, &g, 60, &mut ra, &w, &toy_net, &mut Vec::new());
        rv_search(&mut b, &g, 60, &mut rb, &w, &toy_net, &mut Vec::new());
        assert_eq!(dump(&a), dump(&b));
    }
}

#[test]
fn merged_advance_finds_the_root_by_signature_or_by_observation() {
    let mut checked = [0; 2];
    for s in 0..10u64 {
        let g = Counter::with_hand(s % 2 == 0, s % 3 == 0);
        let mut rng = Rng::new(s);
        let mut t = Tree::new(&g, 0, &[CAST, HOLD], true, &mut rng);
        t.set_merge(true).unwrap();
        rv_search(&mut t, &g, 200, &mut rng, &W, &counter_net, &mut Vec::new());
        let r = t.root;
        let ci = t.acts(r).iter().position(|&a| a == CAST).unwrap();
        let x = seat0_kid(&t, r, ci).unwrap();
        let before = {
            let mut c = t.clone();
            c.root = x;
            dump(&c)
        };
        // the real game: seat 0 casts; seat 1 (holding Daze) allows - the same observation for seat 0 as the
        // auto-stepped world's. By signature: the real transition as the auto-stepped world keyed it ([]); by
        // observation: an unknown real key (as a revelation the search never sampled), found by encode(viewer)
        let mut live = Counter::with_hand(true, false);
        let mut keys = Vec::new();
        live.step(CAST, &mut keys).unwrap();
        live.step(ALLOW, &mut keys).unwrap();
        let legal = vec![CAST, HOLD];
        let by_obs = s % 2 == 1;
        let key: Vec<u64> = if by_obs { vec![99] } else { vec![] };
        let mut u = t.clone();
        let enc = u.merge_stats().unwrap().encodes;
        assert!(!u.advance_in(CAST, &key, 1, &legal, &live), "another viewer: no reuse");
        assert!(u.advance_in(CAST, &key, 0, &legal, &live));
        assert_eq!(u.merge_stats().unwrap().encodes, enc + by_obs as u64);
        assert_eq!(u.root, x);
        assert_eq!(dump(&u), before, "the reused subtree as it was");
        checked[by_obs as usize] += 1;
        // the search goes on from the new root
        let tot0 = u.tot(u.root);
        rv_search(&mut u, &live, 100, &mut rng, &W, &counter_net, &mut Vec::new());
        assert_eq!(u.tot(u.root), tot0 + 100);
        // and a compaction of the reused tree keeps the search
        let mut c = u.clone();
        c.compact();
        assert_eq!(dump(&c), dump(&u));
        // an observation the tree never reached: no reuse
        let mut other = Counter::with_hand(false, false);
        other.step(HOLD, &mut keys).unwrap();
        other.step(HOLD, &mut keys).unwrap();
        assert!(other.to_move().is_none());
    }
    assert!(checked[0] > 0 && checked[1] > 0);
}

#[test]
fn a_repeated_observation_on_the_walks_path_gets_a_fresh_node() {
    // a forgetful counter game (features without the round and the score) of 5 rounds: seat 0's next round after a
    // hold looks like this one, so a walk would reach a node already on its path - it gets an alias instead, and
    // every walk's backup stays exact
    use crate::counter_game::CounterSpec;
    let spec = CounterSpec { rounds: 5, forget: true };
    let mut aliases = 0;
    for s in 0..6u64 {
        let g = Counter::with_spec(spec, s % 2 == 0, s % 3 == 0);
        let mut rng = Rng::new(s);
        let mut t = Tree::new(&g, 0, &[CAST, HOLD], true, &mut rng);
        t.set_merge(true).unwrap();
        let mut log = Vec::new();
        rv_search(&mut t, &g, 400, &mut rng, &W, &counter_net, &mut log);
        recount(&t, &log);
        let ms = t.merge_stats().unwrap();
        assert!(ms.merged > 0, "{ms:?}");
        aliases += ms.aliases;
        let mut b = t.clone();
        b.compact();
        assert_eq!(dump(&t), dump(&b));
        let (mut ra, mut rb) = (rng.clone(), rng.clone());
        let (mut la, mut lb) = (log.clone(), log.clone());
        rv_search(&mut t, &g, 100, &mut ra, &W, &counter_net, &mut la);
        rv_search(&mut b, &g, 100, &mut rb, &W, &counter_net, &mut lb);
        assert_eq!(dump(&t), dump(&b));
        recount(&t, &la);
    }
    assert!(aliases > 0, "no walk met its own observation");
}
