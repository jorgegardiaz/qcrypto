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
        assert!(val >= 0.0 && val < 1.0);
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
