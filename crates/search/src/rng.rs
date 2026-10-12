//! Small, fast, reproducible RNG for the search (xoshiro256++ seeded through splitmix64).
//! Games use it in `determinize`; the driver derives every game's streams from
//! (driver seed, game index) with `split64`, so a driver run is reproducible.

/// splitmix64 finaliser: a good 64-bit mix (seed derivation).
pub fn split64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Default for Rng {
    fn default() -> Self {
        Rng::new(0)
    }
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut x = seed;
        let mut s = [0u64; 4];
        for v in s.iter_mut() {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            *v = split64(x);
        }
        if s.iter().all(|&v| v == 0) {
            s[0] = 1;
        }
        Rng { s }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let r = (self.s[0].wrapping_add(self.s[3])).rotate_left(23).wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }

    /// Uniform in [0, 1).
    #[inline]
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in [0, n), n >= 1 (Lemire's multiply-shift; bias < 2^-32 for n < 2^32).
    #[inline]
    pub fn below(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0);
        (((self.next_u64() >> 32) * n as u64) >> 32) as u32
    }

    /// Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i as u32 + 1) as usize;
            v.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn below_is_uniform() {
        let mut r = Rng::new(7);
        let mut c = [0u32; 5];
        for _ in 0..100_000 {
            c[r.below(5) as usize] += 1;
        }
        for x in c {
            assert!((x as f64 - 20_000.0).abs() < 600.0, "{c:?}");
        }
    }

    #[test]
    fn reproducible() {
        let a: Vec<u64> = {
            let mut r = Rng::new(42);
            (0..8).map(|_| r.next_u64()).collect()
        };
        let b: Vec<u64> = {
            let mut r = Rng::new(42);
            (0..8).map(|_| r.next_u64()).collect()
        };
        assert_eq!(a, b);
        assert_ne!(Rng::new(1).next_u64(), Rng::new(2).next_u64());
    }
}
