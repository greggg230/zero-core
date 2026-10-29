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
