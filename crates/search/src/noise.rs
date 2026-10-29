//! Gamma / Dirichlet sampling for root noise (no external crates; port of domz), and the standard
//! Gumbel noise of the Gumbel root search (docs/search.md 3.7).
use crate::rng::Rng;

/// Standard normal (Marsaglia polar method).
fn normal(rng: &mut Rng) -> f64 {
    loop {
        let u = 2.0 * rng.unit() - 1.0;
        let v = 2.0 * rng.unit() - 1.0;
        let s = u * u + v * v;
        if s > 0.0 && s < 1.0 {
            return u * (-2.0 * s.ln() / s).sqrt();
        }
    }
}

/// Gamma(shape, 1) (Marsaglia & Tsang; shape < 1 via the U^(1/shape) boost).
pub fn gamma(shape: f64, rng: &mut Rng) -> f64 {
    if shape < 1.0 {
        let u = loop {
            let u = rng.unit();
            if u > 0.0 {
                break u;
            }
        };
        return gamma(shape + 1.0, rng) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let mut x;
        let mut v;
        loop {
            x = normal(rng);
            v = 1.0 + c * x;
            if v > 0.0 {
                break;
            }
        }
        v = v * v * v;
        let u = rng.unit();
        if u < 1.0 - 0.0331 * x * x * x * x || (u > 0.0 && u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln())) {
            return d * v;
        }
    }
}

/// Symmetric Dirichlet(alpha) sample of dimension n.
pub fn dirichlet(alpha: f64, n: usize, rng: &mut Rng) -> Vec<f64> {
    let mut x: Vec<f64> = (0..n).map(|_| gamma(alpha, rng)).collect();
    let s: f64 = x.iter().sum();
    if s > 0.0 {
        for v in x.iter_mut() {
            *v /= s;
        }
    } else {
        let k = rng.below(n as u32) as usize;
        for (i, v) in x.iter_mut().enumerate() {
            *v = if i == k { 1.0 } else { 0.0 };
        }
    }
    x
}

/// A standard Gumbel(0, 1) sample, `-ln(-ln U)` with U uniform in the open interval (0, 1) (the
/// Gumbel root search's noise, docs/search.md 3.7: argmax of `g + logits` over options is a sample
/// of `softmax(logits)`, the Gumbel-max trick).
pub fn gumbel(rng: &mut Rng) -> f64 {
    let u = ((rng.next_u64() >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64);
    -(-u.ln()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_moments() {
        let mut r = Rng::new(3);
        for &a in &[0.3, 1.0, 2.5] {
            let n = 200_000;
            let xs: Vec<f64> = (0..n).map(|_| gamma(a, &mut r)).collect();
            let m = xs.iter().sum::<f64>() / n as f64;
            let v = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n as f64;
            assert!((m - a).abs() < 0.02 * a.max(1.0), "shape {a}: mean {m}");
            assert!((v - a).abs() < 0.05 * a.max(1.0), "shape {a}: var {v}");
        }
    }

    #[test]
    fn dirichlet_sums_to_one_with_symmetric_mean() {
        let mut r = Rng::new(5);
        let mut acc = [0.0; 4];
        for _ in 0..20_000 {
            let x = dirichlet(0.3, 4, &mut r);
            assert!((x.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            for i in 0..4 {
                acc[i] += x[i];
            }
        }
        for a in acc {
            assert!((a / 20_000.0 - 0.25).abs() < 0.01, "{acc:?}");
        }
    }

    #[test]
    fn gumbel_moments_and_the_max_trick() {
        // mean = Euler's gamma 0.5772, variance pi^2 / 6; argmax(g + ln p) samples p
        let mut r = Rng::new(11);
        let n = 200_000;
        let xs: Vec<f64> = (0..n).map(|_| gumbel(&mut r)).collect();
        assert!(xs.iter().all(|x| x.is_finite()));
        let m = xs.iter().sum::<f64>() / n as f64;
        let v = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n as f64;
        assert!((m - 0.5772).abs() < 0.01, "mean {m}");
        assert!((v - std::f64::consts::PI.powi(2) / 6.0).abs() < 0.03, "var {v}");
        let p = [0.6f64, 0.3, 0.1];
        let mut c = [0u32; 3];
        for _ in 0..60_000 {
            let k = (0..3).map(|i| p[i].ln() + gumbel(&mut r)).enumerate().fold((0, f64::MIN), |b, (i, x)| if x > b.1 { (i, x) } else { b }).0;
            c[k] += 1;
        }
        for i in 0..3 {
            assert!((c[i] as f64 / 60_000.0 - p[i]).abs() < 0.01, "{c:?}");
        }
    }
}
