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
//! set_global_seed(12345);
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
/// * `seed` - The 64-bit seed used to initialize the pseudo-random generator.
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
/// * `p` - The probability (`0.0` to `1.0`) that the function returns `true`.
///
/// # Returns
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

/// Shuffles a mutable slice randomly using the thread-local RNG.
///
/// This sequence is deterministic if the thread was seeded using `set_global_seed`.
///
/// # Arguments
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
