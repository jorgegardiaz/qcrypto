//! Global thread-local random number generation.
//!
//! By default, this module uses a fast `ChaCha8Rng` initialized from OS entropy.
//! This ensures high-quality randomness for simulations without the overhead of the OS pool on every call.
//!
//! For reproducible simulations, users can call `set_global_seed(seed)` to lock the RNG
//! to a specific deterministic sequence for the current thread.
//!
//! # Example
//! ```rust
//! use qcrypto::rng::set_global_seed;
//!
//! // Lock the RNG to a specific sequence for this thread
//! set_global_seed(42);
//! ```

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::cell::RefCell;

thread_local! {
    // We use ChaCha8Rng as the default deterministic RNG for speed and quality.
    // By default, it initializes from OS entropy.
    static QCRYPTO_RNG: RefCell<ChaCha8Rng> = RefCell::new(ChaCha8Rng::from_os_rng());
}

/// Sets a deterministic seed for all `qcrypto` operations on the current thread.
///
/// This is highly recommended for users who need identically reproducible simulation data
/// (e.g., error rates, generated keys) across multiple runs.
///
/// # Arguments
///
/// * `seed` - The 64-bit seed used to initialize the pseudo-random generator.
///
/// # Example
/// ```rust
/// use qcrypto::rng::set_global_seed;
///
/// set_global_seed(42);
/// ```
pub fn set_global_seed(seed: u64) {
    QCRYPTO_RNG.with(|rng| {
        *rng.borrow_mut() = ChaCha8Rng::seed_from_u64(seed);
    });
}

/// Generates a random boolean with a specified probability of being `true`.
///
/// This uses the thread-local RNG sequence, making it deterministic if `set_global_seed`
/// was called on the current thread.
///
/// # Arguments
///
/// * `p` - The probability (`0.0` to `1.0`) that the function returns `true`.
///
/// # Returns
///
/// A boolean value: `true` with probability `p`, and `false` with probability `1.0 - p`.
///
/// # Example
/// ```rust
/// use qcrypto::rng::{set_global_seed, random_bool};
///
/// set_global_seed(42);
/// let is_heads = random_bool(0.5); // 50% chance of being true
/// ```
pub fn random_bool(p: f64) -> bool {
    QCRYPTO_RNG.with(|rng| rng.borrow_mut().random_bool(p))
}

/// Generates a random floating-point number in the half-open range `[0.0, 1.0)`.
///
/// This is used internally for probabilistic state collapse and sampling, but is
/// exposed for users building custom protocols or noise models.
///
/// # Returns
///
/// A uniformly distributed `f64` between `0.0` (inclusive) and `1.0` (exclusive).
///
/// # Example
/// ```rust
/// use qcrypto::rng::{set_global_seed, random_f64};
///
/// set_global_seed(42);
/// let random_val = random_f64();
/// assert!(random_val >= 0.0 && random_val < 1.0);
/// ```
pub fn random_f64() -> f64 {
    QCRYPTO_RNG.with(|rng| rng.borrow_mut().random())
}

/// Generates a random floating-point number in the half-open range `[min, max)`.
///
/// This uses the thread-local RNG sequence, making it deterministic if `set_global_seed`
/// was called on the current thread.
///
/// # Arguments
///
/// * `min` - The minimum value (inclusive).
/// * `max` - The maximum value (exclusive).
///
/// # Returns
///
/// A uniformly distributed `f64` between `min` and `max`.
///
/// # Example
/// ```rust
/// use qcrypto::rng::{set_global_seed, random_f64_range};
///
/// set_global_seed(42);
/// let random_val = random_f64_range(1.0, 5.0);
/// assert!(random_val >= 1.0 && random_val < 5.0);
/// ```
pub fn random_f64_range(min: f64, max: f64) -> f64 {
    QCRYPTO_RNG.with(|rng| rng.borrow_mut().random_range(min..max))
}

/// Generates a random integer in the half-open range `[min, max)`.
///
/// This uses the thread-local RNG sequence, making it deterministic if `set_global_seed`
/// was called on the current thread.
///
/// # Arguments
///
/// * `min` - The minimum value (inclusive).
/// * `max` - The maximum value (exclusive).
///
/// # Returns
///
/// A uniformly distributed `usize` between `min` and `max`.
///
/// # Example
/// ```rust
/// use qcrypto::rng::{set_global_seed, random_usize_range};
///
/// set_global_seed(42);
/// let random_val = random_usize_range(0, 3);
/// assert!(random_val < 3);
/// ```
pub fn random_usize_range(min: usize, max: usize) -> usize {
    QCRYPTO_RNG.with(|rng| rng.borrow_mut().random_range(min..max))
}

/// Shuffles a mutable slice randomly using the thread-local RNG.
///
/// This sequence is deterministic if the thread was seeded using `set_global_seed`.
///
/// # Arguments
///
/// * `slice` - A mutable reference to a slice of items to be shuffled in-place.
///
/// # Example
/// ```rust
/// use qcrypto::rng::{set_global_seed, shuffle_slice};
///
/// set_global_seed(42);
/// let mut bases = vec![0, 1, 2, 3, 4];
/// shuffle_slice(&mut bases);
/// // bases is now deterministically shuffled
/// ```
pub fn shuffle_slice<T>(slice: &mut [T]) {
    QCRYPTO_RNG.with(|rng| {
        rand::seq::SliceRandom::shuffle(slice, &mut *rng.borrow_mut());
    });
}

/// Draws a fresh 64-bit master seed from the current thread's RNG.
///
/// Used internally by parallel protocol variants to derive per-iteration child RNGs
/// while preserving determinism: after `set_global_seed(s)`, this returns a
/// reproducible value, which is then mixed with each iteration index to seed
/// independent `LocalRng` instances.
pub(crate) fn draw_master_seed() -> u64 {
    QCRYPTO_RNG.with(|rng| rng.borrow_mut().random())
}

/// An explicit, owned random number generator for use inside parallel sections.
///
/// Unlike the module-level functions which borrow a thread-local RNG, `LocalRng`
/// is a stack value that can be passed across rayon iterations. Construct one
/// per parallel work unit via [`LocalRng::child`] so each iteration produces
/// independent, deterministic random streams.
///
/// # Example
/// ```rust
/// use qcrypto::rng::LocalRng;
///
/// let mut rng = LocalRng::child(42, 7);
/// let bit = rng.random_bool(0.5);
/// let val = rng.random_f64();
/// assert!((0.0..1.0).contains(&val));
/// ```
#[derive(Debug, Clone)]
pub struct LocalRng(ChaCha8Rng);

impl LocalRng {
    /// Builds a deterministic child RNG from a master seed and a stream id.
    ///
    /// Two child RNGs with the same `(master_seed, stream_id)` produce identical
    /// sequences; differing stream ids produce statistically independent ones.
    /// The mixing constant is the golden-ratio multiplier `0x9E3779B97F4A7C15`,
    /// which avoids collisions for small seeds.
    pub fn child(master_seed: u64, stream_id: u64) -> Self {
        let mixed = master_seed
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(stream_id);
        Self(ChaCha8Rng::seed_from_u64(mixed))
    }

    /// Builds a `LocalRng` from a raw 64-bit seed.
    pub fn from_seed(seed: u64) -> Self {
        Self(ChaCha8Rng::seed_from_u64(seed))
    }

    /// Generates a random boolean with probability `p` of being `true`.
    pub fn random_bool(&mut self, p: f64) -> bool {
        self.0.random_bool(p)
    }

    /// Generates a uniform `f64` in `[0.0, 1.0)`.
    pub fn random_f64(&mut self) -> f64 {
        self.0.random()
    }

    /// Generates a uniform `f64` in `[min, max)`.
    pub fn random_f64_range(&mut self, min: f64, max: f64) -> f64 {
        self.0.random_range(min..max)
    }

    /// Generates a uniform `usize` in `[min, max)`.
    pub fn random_usize_range(&mut self, min: usize, max: usize) -> usize {
        self.0.random_range(min..max)
    }

    /// Shuffles a mutable slice in-place.
    pub fn shuffle_slice<T>(&mut self, slice: &mut [T]) {
        rand::seq::SliceRandom::shuffle(slice, &mut self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_global_seed_deterministic() {
        set_global_seed(42);
        let val1 = random_f64();

        set_global_seed(42);
        let val2 = random_f64();

        assert_eq!(val1, val2);
    }

    #[test]
    fn test_random_bool() {
        set_global_seed(123);
        // p=1.0 should always be true
        assert!(random_bool(1.0));
        // p=0.0 should always be false
        assert!(!random_bool(0.0));
    }

    #[test]
    fn test_random_f64() {
        let val = random_f64();
        assert!((0.0..1.0).contains(&val));
    }

    #[test]
    fn test_random_f64_range() {
        let min = 1.5;
        let max = 3.5;

        // Check that bounds are strictly respected over many iterations
        for _ in 0..1000 {
            let val = random_f64_range(min, max);
            assert!(val >= min && val < max);
        }

        // Check determinism
        set_global_seed(999);
        let val1 = random_f64_range(min, max);

        set_global_seed(999);
        let val2 = random_f64_range(min, max);

        assert_eq!(val1, val2);
    }

    #[test]
    fn test_random_usize_range() {
        let min = 1;
        let max = 10;

        for _ in 0..1000 {
            let val = random_usize_range(min, max);
            assert!(val >= min && val < max);
        }

        let max2 = 100;
        // Check determinism
        set_global_seed(42);
        let val1 = random_usize_range(min, max2);

        set_global_seed(42);
        let val2 = random_usize_range(min, max2);

        assert_eq!(val1, val2);
    }

    #[test]
    fn test_shuffle_slice() {
        set_global_seed(42);
        let mut data1 = vec![1, 2, 3, 4, 5];
        shuffle_slice(&mut data1);

        set_global_seed(42);
        let mut data2 = vec![1, 2, 3, 4, 5];
        shuffle_slice(&mut data2);

        // Should shuffle the same way
        assert_eq!(data1, data2);
        // And it should actually shuffle (highly unlikely to be 1,2,3,4,5 for seed 42)
        assert_ne!(data1, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_local_rng_child_reproducible() {
        let mut a = LocalRng::child(42, 7);
        let mut b = LocalRng::child(42, 7);
        for _ in 0..16 {
            assert_eq!(a.random_f64(), b.random_f64());
        }
    }

    #[test]
    fn test_local_rng_child_independent_streams() {
        let mut a = LocalRng::child(42, 0);
        let mut b = LocalRng::child(42, 1);
        // Probability of collision on 16 floats is astronomically small.
        let seq_a: Vec<f64> = (0..16).map(|_| a.random_f64()).collect();
        let seq_b: Vec<f64> = (0..16).map(|_| b.random_f64()).collect();
        assert_ne!(seq_a, seq_b);
    }

    #[test]
    fn test_local_rng_from_seed_reproducible() {
        let mut a = LocalRng::from_seed(123);
        let mut b = LocalRng::from_seed(123);
        assert_eq!(a.random_bool(0.5), b.random_bool(0.5));
        assert_eq!(a.random_f64(), b.random_f64());
        assert_eq!(a.random_f64_range(0.0, 10.0), b.random_f64_range(0.0, 10.0));
        assert_eq!(a.random_usize_range(0, 100), b.random_usize_range(0, 100));

        let mut data1: Vec<u32> = (0..16).collect();
        let mut data2 = data1.clone();
        a.shuffle_slice(&mut data1);
        b.shuffle_slice(&mut data2);
        assert_eq!(data1, data2);
    }

    #[test]
    fn test_draw_master_seed_deterministic() {
        set_global_seed(42);
        let s1 = draw_master_seed();
        set_global_seed(42);
        let s2 = draw_master_seed();
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_local_rng_extreme_probabilities() {
        let mut rng = LocalRng::from_seed(7);
        assert!(rng.random_bool(1.0));
        assert!(!rng.random_bool(0.0));
    }

    #[test]
    fn test_measurement_determinism_with_seed() {
        use crate::{Gate, Measurement, QuantumState};

        let create_superposition = || {
            let mut state = QuantumState::new(3); // 3 qubits
            // Apply Hadamard to all 3 to put them in uniform superposition
            state.apply(&Gate::h(), &[0]).unwrap();
            state.apply(&Gate::h(), &[1]).unwrap();
            state.apply(&Gate::h(), &[2]).unwrap();
            state
        };

        // Run 1: Set seed and measure all
        set_global_seed(42);
        let mut state1 = create_superposition();
        let m1_q0 = state1.measure(&Measurement::z_basis(), &[0]).unwrap();
        let m1_q1 = state1.measure(&Measurement::z_basis(), &[1]).unwrap();
        let m1_q2 = state1.measure(&Measurement::z_basis(), &[2]).unwrap();

        // Run 2: Reset to same seed and measure all
        set_global_seed(42);
        let mut state2 = create_superposition();
        let m2_q0 = state2.measure(&Measurement::z_basis(), &[0]).unwrap();
        let m2_q1 = state2.measure(&Measurement::z_basis(), &[1]).unwrap();
        let m2_q2 = state2.measure(&Measurement::z_basis(), &[2]).unwrap();

        // They must be absolutely identical due to determinism
        assert_eq!(m1_q0.value, m2_q0.value);
        assert_eq!(m1_q1.value, m2_q1.value);
        assert_eq!(m1_q2.value, m2_q2.value);
    }
}
