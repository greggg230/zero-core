//! Near-tie branches (`BranchRule::NearTie`, docs/search.md 2.11.1): the gate's boundaries (`NearTie::alt`), a branch
//! plays its runner-up first and records nothing before it, the games themselves untouched, merges on a game with
//! transpositions (`PickGame`: the branch ends there, its records and row carry the game's outcome, the tallies), no
//! merge without `merge_key`, nothing spawned without an eligible decision, the settings refused, reproducibility.
use super::*;
use crate::game::MAX_SEATS;
use crate::rng::Rng;

// ---------------------------------------------------------------- the gate

fn gate() -> NearTie {
    NearTie { eps: 0.25, n_min: 4, p_min: 0.25, q_lo: 0.25, q_hi: 0.75 }
}

#[test]
fn the_gate_at_its_boundaries() {
    let acts = [10u16, 11, 12];
    // Q = W / N: the played move (index 0) 0.5, option 11 0.25, option 12 0.0 (exact in binary)
    let n = [8u32, 4, 4];
    let w = [4.0, 1.0, 0.0];
    let p0 = [0.5f32, 0.25, 0.25];
    let all = |_: u16| true;
    // the gap exactly eps is eligible, just over it is not
    assert_eq!(gate().alt(&acts, &n, &w, &p0, 0, all), Some((11, 0.25)));
    assert_eq!(NearTie { eps: 0.2499, ..gate() }.alt(&acts, &n, &w, &p0, 0, all), None);
    // visits below n_min: not a candidate (the next best is too far)
    assert_eq!(gate().alt(&acts, &[8, 3, 4], &w, &p0, 0, all), None);
    assert_eq!(NearTie { eps: 0.5, ..gate() }.alt(&acts, &[8, 3, 4], &w, &p0, 0, all), Some((12, 0.5)));
    // a prior below p_min: not a candidate; exactly p_min is
    assert_eq!(gate().alt(&acts, &n, &w, &[0.5, 0.2, 0.3], 0, all), None);
    assert_eq!(gate().alt(&acts, &n, &w, &[0.5, 0.25, 0.25], 0, all), Some((11, 0.25)));
    // the played move's Q outside [q_lo, q_hi]: never eligible; the bounds themselves are inside
    assert_eq!(NearTie { q_lo: 0.5, ..gate() }.alt(&acts, &n, &w, &p0, 0, all), Some((11, 0.25)));
    assert_eq!(NearTie { q_lo: 0.5001, ..gate() }.alt(&acts, &n, &w, &p0, 0, all), None);
    assert_eq!(NearTie { q_hi: 0.5, ..gate() }.alt(&acts, &n, &w, &p0, 0, all), Some((11, 0.25)));
    assert_eq!(NearTie { q_hi: 0.4999, ..gate() }.alt(&acts, &n, &w, &p0, 0, all), None);
    // branch_alt_ok false: the best candidate is skipped for the next one
    let mut asked = Vec::new();
    assert_eq!(gate().alt(&acts, &n, &w, &p0, 0, |b| { asked.push(b); b != 11 }), None);
    assert_eq!(asked, vec![11, 12], "best Q first, until one is ok");
    assert_eq!(NearTie { eps: 0.5, ..gate() }.alt(&acts, &n, &w, &p0, 0, |b| b != 11), Some((12, 0.5)));
    // a runner-up better than the played move counts by the absolute gap; the first option on equal Qs
    assert_eq!(gate().alt(&acts, &[4, 8, 4], &[1.0, 4.0, 2.0], &p0, 0, all), Some((11, 0.25)));
    assert_eq!(gate().alt(&acts, &[4, 8, 4], &[1.0, 3.0, 2.0], &p0, 0, all), Some((12, 0.25)));
    assert_eq!(gate().alt(&acts, &[8, 4, 4], &[4.0, 2.0, 2.0], &p0, 0, all), Some((11, 0.0)));
    // the played move without a visit, a single option, mismatched lengths: nothing
    assert_eq!(gate().alt(&acts, &[0, 4, 4], &[0.0, 2.0, 2.0], &p0, 0, all), None);
    assert_eq!(gate().alt(&acts[..1], &n[..1], &w[..1], &p0[..1], 0, all), None);
    assert_eq!(gate().alt(&acts, &n[..2], &w, &p0, 0, all), None);
    // another chosen index: the gate is about that move
    assert_eq!(gate().alt(&acts, &[4, 4, 4], &[2.0, 2.0, 1.0], &p0, 1, all), Some((10, 0.0)));
}

// ---------------------------------------------------------------- a game with transpositions

/// "Pick": two players alternate `turns` turns (seat 0 first); the mover adds 0, 1 or 2 to its own sum; the higher sum
/// wins (a tie is a draw). Picks commute, so lines transpose: seat 0 picking 1 then 2 reaches the state of 2 then 1
/// when seat 1 picked the same between. `keys`: `merge_key` = (turn, sums) - the whole state; without it None.
#[derive(Clone, Debug)]
struct PickGame {
    turns: u8,
    keys: bool,
    turn: u8,
    sum: [u8; 2],
}

impl SearchGame for PickGame {
    type Spec = (u8, bool);
    const NAME: &'static str = "pick";
    const NUM_ACTIONS: usize = 3;
    const NUM_FEATURES: usize = 4;
    const MIN_PLAYERS: usize = 2;
    const MAX_PLAYERS: usize = 2;

    fn parse_spec(json: &str) -> Result<(u8, bool), String> {
        let v: serde_json::Value = serde_json::from_str(if json.trim().is_empty() { "{}" } else { json }).map_err(|e| e.to_string())?;
        Ok((v.get("turns").and_then(|x| x.as_u64()).unwrap_or(8) as u8, v.get("keys").and_then(|x| x.as_bool()).unwrap_or(true)))
    }
    fn new_game(spec: &(u8, bool), players: usize, _seed: u64) -> Result<PickGame, String> {
        if players != 2 {
            return Err("pick: 2 players".into());
        }
        Ok(PickGame { turns: spec.0, keys: spec.1, turn: 0, sum: [0, 0] })
    }
    fn num_players(&self) -> usize {
        2
    }
    fn to_move(&self) -> Option<usize> {
        (self.turn < self.turns).then_some(self.turn as usize % 2)
    }
    fn legal_actions(&self, out: &mut Vec<u16>) {
        if self.turn < self.turns {
            out.extend_from_slice(&[0, 1, 2]);
        }
    }
    fn step(&mut self, a: u16, _chance: &mut Vec<u64>) -> Result<(), String> {
        if a > 2 || self.turn >= self.turns {
            return Err(format!("pick: {a} is not legal"));
        }
        self.sum[self.turn as usize % 2] += a as u8;
        self.turn += 1;
        Ok(())
    }
    fn result(&self) -> [f32; MAX_SEATS] {
        let mut r = [0.0; MAX_SEATS];
        match self.sum[0].cmp(&self.sum[1]) {
            std::cmp::Ordering::Greater => r[0] = 1.0,
            std::cmp::Ordering::Less => r[1] = 1.0,
            std::cmp::Ordering::Equal => r[..2].fill(0.5),
        }
        r
    }
    fn margin(&self, seat: usize) -> f32 {
        (self.sum[seat] as f32 - self.sum[1 - seat] as f32) / 16.0
    }
    fn score(&self, seat: usize) -> f32 {
        self.sum[seat] as f32
    }
    fn determinize(&mut self, _viewer: usize, _rng: &mut Rng) {}
    fn encode(&self, viewer: usize, out: &mut [f32]) {
        out[0] = self.turn as f32 / 8.0;
        out[1] = self.sum[viewer] as f32 / 8.0;
        out[2] = self.sum[1 - viewer] as f32 / 8.0;
        out[3] = (self.turn as usize % 2 == viewer) as u8 as f32;
    }
    fn merge_key(&self) -> Option<u128> {
        self.keys.then(|| (self.turn as u128) << 16 | (self.sum[0] as u128) << 8 | self.sum[1] as u128)
    }
}

/// A near-tie rule that admits every decision with a visited runner-up.
fn loose() -> NearTie {
    NearTie { eps: 1.0, n_min: 1, p_min: 0.0, q_lo: 0.0, q_hi: 1.0 }
}

fn nt_cfg(games: usize, p: f64, nt: NearTie, spec: &str) -> DriverCfg {
    let mut c = cfg(vec![nn_full(12)], vec![vec![0, 0]; games], &[("m", KeyKind::Nn)]);
    c.seed = 41;
    c.spec = spec.into();
    c.move_log = 1000;
    c.branch = Some(BranchCfg { p, moves: 1, temp: 2.0, differ: false, rule: BranchRule::NearTie(nt) });
    c
}

type Rows = Vec<(GameResult, Vec<Record>)>;

fn key(v: &Rows) -> Vec<(u32, u64, Vec<f32>, Option<BranchInfo>, Option<u32>, u64)> {
    v.iter().map(|(r, s)| (r.gid, r.steps, r.result.clone(), r.branch.clone(), r.neartie, records_digest(s))).collect()
}

#[test]
fn a_neartie_branch_plays_its_runner_up_first_and_leaves_its_game_alone() {
    let mut d = Driver::<ToyGame>::new(nt_cfg(40, 0.5, loose(), "{}")).unwrap();
    let on = run_split(&mut d, true);
    let st = d.stats();
    let mut c = nt_cfg(40, 0.5, loose(), "{}");
    c.branch = None;
    let off = run_split(&mut Driver::<ToyGame>::new(c).unwrap(), true);
    // the games: move for move and record for record the games without branching; each counts its eligible decisions
    let main: Vec<_> = on.iter().filter(|(r, _)| r.branch.is_none()).collect();
    assert_eq!(main.len(), 40);
    for ((r, s), (r0, s0)) in main.iter().zip(&off) {
        assert_eq!((r.gid, r.steps, &r.scores, r.decisions, r.samples, &r.moves), (r0.gid, r0.steps, &r0.scores, r0.decisions, r0.samples, &r0.moves));
        assert_eq!(records_digest(s), records_digest(s0), "game {}: its records changed", r.gid);
        assert!(r0.neartie.is_none());
        assert!(r.neartie.unwrap() > 0 && r.neartie.unwrap() <= r.samples, "game {}: {:?} eligible of {} full searches", r.gid, r.neartie, r.samples);
    }
    assert_eq!(st.neartie_seen, main.iter().map(|(r, _)| r.neartie.unwrap() as u64).sum::<u64>());
    let br: Vec<_> = on.iter().filter(|(r, _)| r.branch.is_some()).collect();
    assert!((10..=30).contains(&br.len()), "{} branches of 40 games at p 0.5", br.len());
    assert_eq!(st.branches as usize, br.len());
    assert_eq!(st.dev_moves as usize, br.len(), "exactly one deviation a branch");
    assert_eq!((st.merges, st.merge_decisions), (0, 0), "the toy has no merge key: nothing merges");
    for (r, s) in &br {
        let b = r.branch.as_ref().unwrap();
        let (g, _) = &off[r.gid as usize];
        let alt = b.alt.expect("a near-tie branch names its runner-up");
        assert!(alt != b.action && b.moves == 1 && b.played == 1 && !b.same_first && b.merged.is_none(), "{b:?}");
        assert!(b.gap <= 1.0 && b.at < b.of && b.of == main[r.gid as usize].0.neartie.unwrap(), "{b:?}");
        assert!(r.neartie.is_none());
        // the branch's moves: its game's up to the branch point, then the runner-up where the game played its move
        let k = b.step as usize;
        assert_eq!(r.moves[..k], g.moves[..k]);
        assert_eq!((g.moves[k].1, r.moves[k].1), (b.action, alt), "game {}: the branch's first move is its runner-up", r.gid);
        assert!(g.moves[k].2.contains(&alt));
        // nothing recorded at the deviation; every full search after it is (every search full)
        assert_eq!(r.samples, r.decisions - b.decision - 1, "{b:?}");
        assert!(r.over);
        for x in s.iter() {
            for j in 0..2 {
                assert_eq!(x.z[j], r.result[(x.player as usize + j) % 2], "a branch that did not merge: its own outcome");
            }
        }
    }
    // reproducible from the seed, and the same at another thread / group layout
    let mut c = nt_cfg(40, 0.5, loose(), "{}");
    c.threads = 1;
    c.groups = 1;
    let again = run_split(&mut Driver::<ToyGame>::new(c).unwrap(), true);
    assert_eq!(key(&on), key(&again));
}

#[test]
fn a_game_without_an_eligible_decision_spawns_nothing() {
    // no runner-up ever has 10,000 visits: no decision is eligible
    let nt = NearTie { n_min: 10_000, ..loose() };
    let mut d = Driver::<ToyGame>::new(nt_cfg(20, 1.0, nt, "{}")).unwrap();
    let rows = run_split(&mut d, true);
    assert_eq!(rows.len(), 20);
    assert!(rows.iter().all(|(r, _)| r.branch.is_none() && r.neartie == Some(0)));
    let st = d.stats();
    assert_eq!((st.branches, st.neartie_seen, st.dev_moves), (0, 0, 0));
    // a strict gap: fewer eligible decisions than the loose gate's, never more
    let strict = run_split(&mut Driver::<ToyGame>::new(nt_cfg(20, 1.0, NearTie { eps: 0.002, ..loose() }, "{}")).unwrap(), true);
    let lax = run_split(&mut Driver::<ToyGame>::new(nt_cfg(20, 1.0, loose(), "{}")).unwrap(), true);
    let seen = |v: &Rows| v.iter().filter_map(|(r, _)| r.neartie).sum::<u32>();
    assert!(seen(&strict) < seen(&lax), "eps 0.002: {} eligible, eps 1: {}", seen(&strict), seen(&lax));
    for (r, _) in &strict {
        if let Some(b) = &r.branch {
            assert!(b.gap <= 0.002, "{b:?}");
        }
    }
}

#[test]
fn a_branch_that_reaches_its_games_state_merges_with_the_games_outcome() {
    let mut d = Driver::<PickGame>::new(nt_cfg(80, 1.0, loose(), r#"{"turns": 8, "keys": true}"#)).unwrap();
    let rows = run_split(&mut d, true);
    let st = d.stats();
    let main = |gid: u32| rows.iter().find(|(r, _)| r.gid == gid && r.branch.is_none()).unwrap();
    let br: Vec<_> = rows.iter().filter(|(r, _)| r.branch.is_some()).collect();
    let merged: Vec<_> = br.iter().filter(|(r, _)| r.branch.as_ref().unwrap().merged.is_some()).collect();
    assert!(br.len() >= 60, "{} branches of 80 games at p 1", br.len());
    assert!(!merged.is_empty() && merged.len() < br.len(), "{} of {} branches merged", merged.len(), br.len());
    assert_eq!(st.merges as usize, merged.len());
    assert_eq!(st.merge_decisions, merged.iter().map(|(r, _)| r.branch.as_ref().unwrap().merged.unwrap() as u64).sum::<u64>());
    for (r, s) in &merged {
        let b = r.branch.as_ref().unwrap();
        let (g, gs) = main(r.gid);
        let k = b.merged.unwrap();
        // it ended at the merge: a state of its game's after the branch point (picks commute: a later turn, the
        // same sums), after k searched decisions past its deviation
        assert!(r.over && !r.aborted);
        assert_eq!(r.decisions, b.decision + 1 + k, "{b:?}");
        assert!(r.steps > b.step + 1 && r.steps < g.steps, "merged at move {} of a game of {} (branch point {})", r.steps, g.steps, b.step);
        // its row and records carry the game's outcome - the z and sc the game's records got
        assert_eq!((&r.result, &r.scores), (&g.result, &g.scores));
        assert_eq!(s.len(), r.samples as usize);
        assert_eq!(r.samples, k, "every full search after the deviation recorded, up to the merge");
        let gz = gs.first().map(|x| (x.player, x.z, x.sc)).unwrap();
        for x in s.iter() {
            for j in 0..2 {
                assert_eq!(x.z[j], g.result[(x.player as usize + j) % 2]);
            }
            if x.player == gz.0 {
                assert_eq!((x.z, x.sc), (gz.1, gz.2));
            }
        }
    }
    // the branches that did not merge played to their own end with their own outcome
    for (r, s) in br.iter().filter(|(r, _)| r.branch.as_ref().unwrap().merged.is_none()) {
        assert_eq!(r.steps, 8, "an unmerged branch plays all 8 turns");
        for x in s.iter() {
            assert_eq!(x.z[0], r.result[x.player as usize]);
        }
    }
    // reproducible
    let mut d2 = Driver::<PickGame>::new(nt_cfg(80, 1.0, loose(), r#"{"turns": 8, "keys": true}"#)).unwrap();
    assert_eq!(key(&rows), key(&run_split(&mut d2, true)));
    // without merge keys (merge_key None) nothing merges: the same branch points, played to the end
    let mut d3 = Driver::<PickGame>::new(nt_cfg(80, 1.0, loose(), r#"{"turns": 8, "keys": false}"#)).unwrap();
    let nokeys = run_split(&mut d3, true);
    assert_eq!(d3.stats().merges, 0);
    let pts = |v: &Rows| v.iter().filter_map(|(r, _)| r.branch.as_ref().map(|b| (r.gid, b.at, b.of, b.action, b.alt))).collect::<Vec<_>>();
    assert_eq!(pts(&rows), pts(&nokeys));
    assert!(nokeys.iter().all(|(r, _)| r.branch.as_ref().and_then(|b| b.merged).is_none() && r.steps == 8));
}

#[test]
fn neartie_with_gumbel_roots_and_root_floors() {
    // the gate reads every visit of a root option: a Gumbel root search and a root floor are supported
    for variant in 0..2 {
        let mut c = nt_cfg(20, 1.0, NearTie { eps: 0.2, ..loose() }, "{}");
        if let SeatCfg::Nn(s) = &mut c.seats[0] {
            if variant == 0 {
                s.root_search = RootSearch::Gumbel { m: 4, scale: 1.0, c_visit: 50.0, c_scale: 1.0 };
                s.policy_target = PolicyTarget::CompletedQ { c_visit: 50.0, c_scale: 1.0, prior_w: 1.0 };
            } else {
                s.root_min_visits = 2;
            }
        }
        let mut d = Driver::<ToyGame>::new(c).unwrap();
        let rows = run_split(&mut d, true);
        let br: Vec<_> = rows.iter().filter_map(|(r, _)| r.branch.as_ref()).collect();
        assert!(!br.is_empty() && br.iter().all(|b| b.gap <= 0.2 && b.alt.is_some()), "variant {variant}: {} branches", br.len());
        assert_eq!(d.stats().neartie_seen, rows.iter().filter_map(|(r, _)| r.neartie.map(u64::from)).sum::<u64>());
    }
}

#[test]
fn neartie_settings_are_checked() {
    let bad = |f: &dyn Fn(&mut DriverCfg)| {
        let mut c = nt_cfg(2, 0.5, NearTie::default(), "{}");
        f(&mut c);
        Driver::<ToyGame>::new(c).err().unwrap_or_default()
    };
    let set = |b: BranchCfg| move |c: &mut DriverCfg| c.branch = Some(b.clone());
    let base = BranchCfg { p: 0.5, moves: 1, temp: 2.0, differ: false, rule: BranchRule::NearTie(NearTie::default()) };
    assert!(bad(&set(BranchCfg { moves: 2, ..base.clone() })).contains("one deviation move"));
    assert!(bad(&set(BranchCfg { temp: 1.0, ..base.clone() })).contains("one deviation move"));
    assert!(bad(&set(BranchCfg { differ: true, ..base.clone() })).contains("one deviation move"));
    assert!(bad(&set(BranchCfg { rule: BranchRule::NearTie(NearTie { n_min: 0, ..NearTie::default() }), ..base.clone() })).contains("n_min"));
    assert!(bad(&set(BranchCfg { rule: BranchRule::NearTie(NearTie { q_lo: 0.9, q_hi: 0.1, ..NearTie::default() }), ..base.clone() })).contains("q_lo"));
    assert!(bad(&set(BranchCfg { rule: BranchRule::NearTie(NearTie { eps: f64::NAN, ..NearTie::default() }), ..base.clone() })).contains("eps"));
    // a recording self-play seat with a margin term: its Q is no win estimate
    assert!(bad(&|c: &mut DriverCfg| {
        if let SeatCfg::Nn(s) = &mut c.seats[0] {
            s.margin_w = 0.1;
        }
    })
    .contains("margin_w"));
    assert!(bad(&|_: &mut DriverCfg| {}).is_empty(), "the defaults are accepted");
}
