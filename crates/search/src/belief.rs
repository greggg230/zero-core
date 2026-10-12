//! Belief particles (docs/search.md 10.3): another seat's hidden choice sampled K times with the
//! network's own policy, replayed from the state where that choice began
//! (`SearchGame::belief_start`) as the chooser saw it. A `BeliefJob` advances one level (one
//! atomic decision of the chooser) per network evaluation: `level()` lists the distinct states
//! awaiting an evaluation with the number of particles that went through each (identical
//! prefixes are evaluated once), `apply` draws every particle's next option from the
//! (tempered) priors. The driver runs jobs inside its ordinary network batches (belief rows);
//! `sample` runs one synchronously (tests, analysis).
use crate::game::SearchGame;
use crate::rng::Rng;
use crate::tree::transition;

/// One belief being sampled.
pub struct BeliefJob<G: SearchGame> {
    pub chooser: usize,
    pub key: u64,
    level: Vec<(G, u32)>,
    done: Vec<u64>,
}

impl<G: SearchGame> BeliefJob<G> {
    /// `particles` particles of `chooser`'s hidden choice from `root` (a state where the chooser
    /// begins it: `belief_start` was Some(key)).
    pub fn new(root: G, chooser: usize, key: u64, particles: u32) -> Result<BeliefJob<G>, String> {
        let mut j = BeliefJob { chooser, key, level: Vec::new(), done: Vec::new() };
        j.push(root, particles)?;
        Ok(j)
    }

    /// The states awaiting an evaluation of the chooser's decision (each with >= 2 options) and
    /// their particle counts. Empty = done.
    pub fn level(&self) -> &[(G, u32)] {
        &self.level
    }

    pub fn is_done(&self) -> bool {
        self.level.is_empty()
    }

    /// The completed particles (repeats kept: frequencies follow the policy).
    pub fn particles(&self) -> &[u64] {
        &self.done
    }

    pub fn into_particles(self) -> Vec<u64> {
        self.done
    }

    /// Add `count` particles at state `st`: complete if the chooser's choice is (then
    /// `belief_particle`), otherwise single-option decisions are auto-stepped and the state joins
    /// the next level. The chooser must be the decider until its choice completes.
    fn push(&mut self, mut st: G, count: u32) -> Result<(), String> {
        let q = self.chooser;
        let mut key = Vec::new();
        let mut legal = Vec::new();
        loop {
            if let Some(x) = st.belief_particle(q) {
                self.done.extend(std::iter::repeat_n(x, count as usize));
                return Ok(());
            }
            if st.to_move() != Some(q) {
                return Err(format!("belief sampling: seat {q}'s hidden choice did not complete (decider {:?})", st.to_move()));
            }
            legal.clear();
            st.legal_actions(&mut legal);
            if legal.len() >= 2 {
                break;
            }
            let a = *legal.first().ok_or("belief sampling: a decision without legal actions")?;
            transition(&mut st, a, None, &mut key, &mut legal)?;
        }
        // distinct option paths give distinct states (no merging: states are not comparable)
        self.level.push((st, count));
        Ok(())
    }

    /// The evaluation of the current level: `priors(r)` = the network's priors over the legal
    /// actions of `level()[r]` (in their order). Each state's particles draw their next option
    /// from p^(1 / temp) (renormalised; all-zero priors: uniform); the distinct results form the
    /// next level.
    pub fn apply<'a>(&mut self, mut priors: impl FnMut(usize) -> &'a [f32], temp: f64, rng: &mut Rng) -> Result<(), String> {
        let level = std::mem::take(&mut self.level);
        let inv_t = 1.0 / temp;
        let mut legal = Vec::new();
        let mut key = Vec::new();
        let mut tl = Vec::new();
        for (r, (st, m)) in level.into_iter().enumerate() {
            let pri = priors(r);
            legal.clear();
            st.legal_actions(&mut legal);
            if legal.len() != pri.len() {
                return Err(format!("belief sampling: {} priors for {} options", pri.len(), legal.len()));
            }
            let mut w: Vec<f64> = pri.iter().map(|&x| if inv_t == 1.0 { (x as f64).max(0.0) } else { (x as f64).max(0.0).powf(inv_t) }).collect();
            let mut tot: f64 = w.iter().sum();
            if !(tot > 0.0 && tot.is_finite()) {
                w.iter_mut().for_each(|x| *x = 1.0);
                tot = w.len() as f64;
            }
            let last = w.iter().rposition(|&x| x > 0.0).unwrap_or(0);
            let mut counts = vec![0u32; w.len()];
            for _ in 0..m {
                let mut u = rng.unit() * tot;
                let mut pick = last;
                for (i, &x) in w.iter().enumerate() {
                    u -= x;
                    if u < 0.0 {
                        pick = i;
                        break;
                    }
                }
                counts[pick] += 1;
            }
            for (i, &c) in counts.iter().enumerate() {
                if c == 0 {
                    continue;
                }
                let mut child = st.clone();
                transition(&mut child, legal[i], None, &mut key, &mut tl).map_err(|e| format!("belief sampling: {e}"))?;
                self.push(child, c)?;
            }
        }
        Ok(())
    }
}

/// Sample `particles` particles of `chooser`'s hidden choice from `root` synchronously:
/// `eval(state, chooser, legal)` -> priors over `legal` (tests, analysis).
pub fn sample<G: SearchGame>(root: G, chooser: usize, particles: u32, temp: f64, rng: &mut Rng, mut eval: impl FnMut(&G, usize, &[u16]) -> Vec<f32>) -> Result<Vec<u64>, String> {
    let mut job = BeliefJob::new(root, chooser, 0, particles)?;
    let mut legal = Vec::new();
    while !job.is_done() {
        let pri: Vec<Vec<f32>> = job
            .level()
            .iter()
            .map(|(st, _)| {
                legal.clear();
                st.legal_actions(&mut legal);
                eval(st, chooser, &legal)
            })
            .collect();
        job.apply(|r| &pri[r], temp, rng)?;
    }
    Ok(job.into_particles())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secret::SecretGame;
    use std::collections::HashMap;

    /// Priors proportional to w[card] over the legal cards.
    fn weights(w: [f32; 5]) -> impl FnMut(&SecretGame, usize, &[u16]) -> Vec<f32> {
        move |_, _, legal| {
            let s: f32 = legal.iter().map(|&a| w[a as usize]).sum();
            legal.iter().map(|&a| w[a as usize] / s).collect()
        }
    }

    /// The exact distribution of two ascending picks with priors ~ w^(1/temp) per pick.
    fn exact(w: [f32; 5], temp: f64) -> HashMap<u64, f64> {
        let wt: Vec<f64> = w.iter().map(|&x| (x as f64).powf(1.0 / temp)).collect();
        let mut out = HashMap::new();
        let first: Vec<usize> = (0..4).collect();
        let z1: f64 = first.iter().map(|&c| wt[c]).sum();
        for &a in &first {
            let z2: f64 = (a + 1..5).map(|c| wt[c]).sum();
            for b in a + 1..5 {
                *out.entry((1u64 << a) | (1u64 << b)).or_insert(0.0) += wt[a] / z1 * wt[b] / z2;
            }
        }
        out
    }

    #[test]
    fn particles_follow_the_tempered_policy() {
        let w = [0.1, 0.5, 1.0, 2.0, 0.4];
        for (temp, seat) in [(1.0, 0usize), (2.0, 0), (1.0, 1)] {
            let mut g = SecretGame::new();
            let mut k = Vec::new();
            if seat == 1 {
                g.step(0, &mut k).unwrap();
                g.step(4, &mut k).unwrap();
            }
            assert_eq!(g.belief_start(), Some(0));
            let n = 40_000;
            let ps = sample(g, seat, n, temp, &mut Rng::new(3), weights(w)).unwrap();
            assert_eq!(ps.len(), n as usize);
            let ex = exact(w, temp);
            let mut cnt: HashMap<u64, f64> = HashMap::new();
            for p in ps {
                *cnt.entry(p).or_insert(0.0) += 1.0 / n as f64;
            }
            for (h, p) in &ex {
                let q = cnt.get(h).copied().unwrap_or(0.0);
                let se = (p * (1.0 - p) / n as f64).sqrt();
                assert!((q - p).abs() < 5.0 * se + 1e-4, "temp {temp} seat {seat} hand {h:#b}: {q:.4} vs {p:.4}");
            }
            assert!(cnt.keys().all(|h| ex.contains_key(h)));
        }
    }

    #[test]
    fn identical_prefixes_share_one_row() {
        let mut job = BeliefJob::new(SecretGame::new(), 0, 0, 64).unwrap();
        assert_eq!(job.level().len(), 1);
        let one_hot = [0.0f32, 1.0, 0.0, 0.0];
        job.apply(|_| &one_hot[..], 1.0, &mut Rng::new(1)).unwrap();
        // every particle picked card 1: one state at level 2
        assert_eq!(job.level().len(), 1);
        assert_eq!(job.level()[0].1, 64);
        let pri = [0.25f32; 3];
        job.apply(|_| &pri[..], 1.0, &mut Rng::new(2)).unwrap();
        assert!(job.is_done() && job.particles().len() == 64);
        assert!(job.particles().iter().all(|&p| p & 0b10 != 0 && p.count_ones() == 2));
    }
}
