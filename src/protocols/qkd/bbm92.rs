//! BBM92 Quantum Key Distribution Protocol.
//!
//! BBM92 is an entanglement-based QKD protocol proposed by Bennett, Brassard, and Mermin in 1992.
//! It is logically equivalent to BB84 but uses entangled photon pairs (EPR source) instead of single
//! photon pulses prepared by Alice.

use crate::rng::LocalRng;
use crate::{Gate, Measurement, QuantumChannel, QuantumState, errors::StateError};
use rayon::prelude::*;

/// The result of the BBM92 protocol execution.
pub struct Bbm92Result {
    /// The total length of the raw key (number of entangled pairs).
    pub raw_length: usize,
    /// The number of bits where bases matched (before sacrificing).
    pub total_sifted: usize,
    /// The number of errors found in the check bits.
    pub check_errors: usize,
    /// The Quantum Bit Error Rate (QBER) on check bits.
    pub qber: f64,
    /// The number of times Eve intercepted a qubit (simulated).
    pub eve_intercept_count: usize,
    /// Alice's chosen bases (0: Z, 1: X).
    pub alice_bases: Vec<bool>,
    /// Bob's chosen bases (0: Z, 1: X).
    pub bob_bases: Vec<bool>,
    /// Alice's measurement results.
    pub alice_bits: Vec<bool>,
    /// Bob's measurement results.
    pub bob_results: Vec<bool>,
    /// Alice's final key (sifted key minus check bits).
    pub alice_key: Vec<bool>,
    /// Bob's final key (sifted key minus check bits). May differ from Alice's due to channel noise.
    pub bob_key: Vec<bool>,
}

/// Executes the BBM92 QKD protocol.
///
/// BBM92 is an entanglement-based version of BB84.
/// Instead of Alice sending states, a source distributes entangled pairs (EPR pairs) to Alice and Bob.
/// They measure their respective qubits in random bases.
///
/// # Arguments
///
/// * `num_pairs` - Number of entangled pairs to distribute.
/// * `channel` - The quantum channel (noise model) affecting the transmission.
/// * `eve_ratio` - Probability of Eve intercepting (and measuring) a qubit.
/// * `check_ratio` - Fraction of sifted bits to sacrifice for QBER estimation.
///
/// # Returns
///
/// A `Result` containing `Bbm92Result` with the simulation statistics and keys.
///
/// # Errors
///
/// Returns a `StateError` if quantum operations fail.
///
/// # Example
/// ```rust
/// use qcrypto::protocols::bbm92;
/// use qcrypto::QuantumChannel;
///
/// let channel = QuantumChannel::bit_flip(0.0);
/// let result = bbm92::run(100, &channel, &channel, 0.0, 0.5).unwrap();
///
/// assert_eq!(result.raw_length, 100);
/// ```
pub fn run(
    num_pairs: usize,
    channel_alice: &QuantumChannel,
    channel_bob: &QuantumChannel,
    eve_ratio: f64,
    check_ratio: f64,
) -> Result<Bbm92Result, StateError> {
    let mut alice_bits = Vec::with_capacity(num_pairs);
    let mut alice_bases = Vec::with_capacity(num_pairs);
    let mut bob_bases = Vec::with_capacity(num_pairs);
    let mut bob_results = Vec::with_capacity(num_pairs);
    let mut eve_intercept_count = 0;

    for _ in 0..num_pairs {
        // Create EPR pair
        let mut state = QuantumState::new(2);
        state
            .apply(&Gate::h(), &[0])?
            .apply(&Gate::cnot(), &[0, 1])?;

        // Alice and Bob each receive their qubit through independent noisy channels
        state
            .apply_channel(channel_alice, &[0])?
            .apply_channel(channel_bob, &[1])?;

        // Eavesdropper intercepts Bob's channel (intercept-and-resend attack)
        if eve_ratio > 0.0 && crate::rng::random_bool(eve_ratio) {
            eve_intercept_count += 1;
            let e_basis = crate::rng::random_bool(0.5);
            let measurement = if e_basis {
                Measurement::x_basis()
            } else {
                Measurement::z_basis()
            };

            let _ = state.measure(&measurement, &[1])?;

            // Eve send qubit to Bob through channel
            state.apply_channel(channel_bob, &[1])?;
        }

        // Alice measures
        let a_basis = crate::rng::random_bool(0.5);
        let a_measurement = if a_basis {
            Measurement::x_basis()
        } else {
            Measurement::z_basis()
        };

        let res_a = state.measure(&a_measurement, &[0])?;

        let a_bit = res_a.index == 1;

        //Bob measures his qubit
        let b_basis = crate::rng::random_bool(0.5);
        let b_measurement = if b_basis {
            Measurement::x_basis()
        } else {
            Measurement::z_basis()
        };

        let res_b = state.measure(&b_measurement, &[1])?;
        let b_bit = res_b.index == 1;

        alice_bits.push(a_bit);
        alice_bases.push(a_basis);
        bob_bases.push(b_basis);
        bob_results.push(b_bit);
    }

    // Sifting stage
    // 1. Identify indices where bases match
    let input_indices: Vec<usize> = (0..num_pairs).collect();
    let mut match_indices: Vec<usize> = input_indices
        .into_iter()
        .filter(|&i| alice_bases[i] == bob_bases[i])
        .collect();

    let total_sifted = match_indices.len();

    // 2. Shuffle indices
    crate::rng::shuffle_slice(&mut match_indices);

    // 3. Split into check and key indices
    let num_check = (total_sifted as f64 * check_ratio).round() as usize;
    let (check_indices, key_indices) = match_indices.split_at(num_check);

    // 4. Calculate QBER on check bits
    let mut check_errors = 0;
    for &i in check_indices {
        if alice_bits[i] != bob_results[i] {
            check_errors += 1;
        }
    }

    let qber = if num_check > 0 {
        check_errors as f64 / num_check as f64
    } else {
        0.0
    };

    // 5. Build keys for Alice and Bob separately
    let mut alice_key = Vec::with_capacity(key_indices.len());
    let mut bob_key = Vec::with_capacity(key_indices.len());
    for &i in key_indices {
        alice_key.push(alice_bits[i]);
        bob_key.push(bob_results[i]);
    }

    Ok(Bbm92Result {
        raw_length: num_pairs,
        total_sifted,
        check_errors,
        qber,
        eve_intercept_count,
        alice_bases,
        bob_bases,
        alice_bits,
        bob_results,
        alice_key,
        bob_key,
    })
}

/// Parallelized variant of [`run`] using rayon. See [`run`] for protocol semantics.
///
/// # Arguments
///
/// * `num_pairs` - Number of entangled pairs to distribute.
/// * `channel` - The quantum channel (noise model) affecting the transmission.
/// * `eve_ratio` - Probability of Eve intercepting (and measuring) a qubit.
/// * `check_ratio` - Fraction of sifted bits to sacrifice for QBER estimation.
///
/// # Returns
///
/// A `Result` containing `Bbm92Result` with the simulation statistics and keys.
///
/// # Errors
///
/// Returns a `StateError` if quantum operations fail.
///
/// # Example
/// ```rust
/// use qcrypto::protocols::bbm92;
/// use qcrypto::QuantumChannel;
///
/// let channel_alice = QuantumChannel::bit_flip(0.1);
/// let channel_bob = QuantumChannel::bit_flip(0.05);
///
/// qcrypto::rng::set_global_seed(42);
/// let r1 = bbm92::run_par(300, &channel_alice, &channel_bob, 0.1, 0.2).unwrap();
///
/// qcrypto::rng::set_global_seed(42);
/// let r2 = bbm92::run_par(300, &channel_alice, &channel_bob, 0.1, 0.2).unwrap();
///
/// assert_eq!(r1.alice_key, r2.alice_key);
/// ```
pub fn run_par(
    num_pairs: usize,
    channel_alice: &QuantumChannel,
    channel_bob: &QuantumChannel,
    eve_ratio: f64,
    check_ratio: f64,
) -> Result<Bbm92Result, StateError> {
    let master = crate::rng::draw_master_seed();
    let process_seed = crate::rng::draw_master_seed();

    type Step = (bool, bool, bool, bool, bool);

    let steps: Vec<Step> = (0..num_pairs)
        .into_par_iter()
        .map(|i| -> Result<Step, StateError> {
            let mut rng = LocalRng::child(master, i as u64);

            let mut state = QuantumState::new(2);
            state
                .apply(&Gate::h(), &[0])?
                .apply(&Gate::cnot(), &[0, 1])?;

            state
                .apply_channel(channel_alice, &[0])?
                .apply_channel(channel_bob, &[1])?;

            let eve_intercepted = eve_ratio > 0.0 && rng.random_bool(eve_ratio);
            if eve_intercepted {
                let e_basis = rng.random_bool(0.5);
                let measurement = if e_basis {
                    Measurement::x_basis()
                } else {
                    Measurement::z_basis()
                };
                let _ = state.measure_with_rng(&measurement, &[1], &mut rng)?;

                state.apply_channel(channel_bob, &[1])?;
            }

            let a_basis = rng.random_bool(0.5);
            let a_measurement = if a_basis {
                Measurement::x_basis()
            } else {
                Measurement::z_basis()
            };
            let res_a = state.measure_with_rng(&a_measurement, &[0], &mut rng)?;
            let a_bit = res_a.index == 1;

            let b_basis = rng.random_bool(0.5);
            let b_measurement = if b_basis {
                Measurement::x_basis()
            } else {
                Measurement::z_basis()
            };
            let res_b = state.measure_with_rng(&b_measurement, &[1], &mut rng)?;
            let b_bit = res_b.index == 1;

            Ok((a_bit, a_basis, b_basis, b_bit, eve_intercepted))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut alice_bits = Vec::with_capacity(num_pairs);
    let mut alice_bases = Vec::with_capacity(num_pairs);
    let mut bob_bases = Vec::with_capacity(num_pairs);
    let mut bob_results = Vec::with_capacity(num_pairs);
    let mut eve_intercept_count = 0usize;

    for (a_bit, a_basis, b_basis, b_bit, eve_intercepted) in steps {
        alice_bits.push(a_bit);
        alice_bases.push(a_basis);
        bob_bases.push(b_basis);
        bob_results.push(b_bit);
        if eve_intercepted {
            eve_intercept_count += 1;
        }
    }

    let mut match_indices: Vec<usize> = (0..num_pairs)
        .filter(|&i| alice_bases[i] == bob_bases[i])
        .collect();
    let total_sifted = match_indices.len();
    let mut rng = LocalRng::from_seed(process_seed);
    rng.shuffle_slice(&mut match_indices);

    let num_check = (total_sifted as f64 * check_ratio).round() as usize;
    let (check_indices, key_indices) = match_indices.split_at(num_check);

    let mut check_errors = 0;
    for &i in check_indices {
        if alice_bits[i] != bob_results[i] {
            check_errors += 1;
        }
    }

    let qber = if num_check > 0 {
        check_errors as f64 / num_check as f64
    } else {
        0.0
    };

    let mut alice_key = Vec::with_capacity(key_indices.len());
    let mut bob_key = Vec::with_capacity(key_indices.len());
    for &i in key_indices {
        alice_key.push(alice_bits[i]);
        bob_key.push(bob_results[i]);
    }

    Ok(Bbm92Result {
        raw_length: num_pairs,
        total_sifted,
        check_errors,
        qber,
        eve_intercept_count,
        alice_bases,
        bob_bases,
        alice_bits,
        bob_results,
        alice_key,
        bob_key,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bbm92_noiseless() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(100, &channel, &channel, 0.0, 0.5).unwrap();

        assert_eq!(result.raw_length, 100);
        assert_eq!(result.check_errors, 0);
        assert_eq!(result.qber, 0.0);
        assert_eq!(result.eve_intercept_count, 0);
        assert_eq!(result.alice_key.len(), result.bob_key.len());
        assert_eq!(result.alice_key, result.bob_key);
    }

    #[test]
    fn test_bbm92_noisy_keys_differ() {
        let channel = QuantumChannel::bit_flip(0.3);
        let result = run(1000, &channel, &channel, 0.0, 0.0).unwrap();

        assert_eq!(result.alice_key.len(), result.bob_key.len());
        let mismatches = result
            .alice_key
            .iter()
            .zip(&result.bob_key)
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            mismatches > 0,
            "noisy channel should produce key mismatches between Alice and Bob"
        );
    }

    #[test]
    fn test_bbm92_eve() {
        let channel = QuantumChannel::bit_flip(0.0);
        // Eve intercepts everything.
        // BBM92 (entanglement-based BB84) also has 25% theoretical QBER under intercept-and-resend.
        // 5000 pairs -> ~1250 check bits -> σ≈0.012, tolerance 0.06 covers ~5σ
        let result = run(5000, &channel, &channel, 1.0, 0.5).unwrap();

        assert!(result.eve_intercept_count > 0);
        assert!(
            (result.qber - 0.25).abs() < 0.06,
            "QBER {} should be around 0.25",
            result.qber
        );
    }

    #[test]
    fn test_bbm92_zero_check() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(100, &channel, &channel, 0.0, 0.0).unwrap();
        assert_eq!(result.qber, 0.0);
    }

    #[test]
    fn test_bbm92_par_zero_check() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run_par(100, &channel, &channel, 0.0, 0.0).unwrap();
        assert_eq!(result.qber, 0.0);
    }

    #[test]
    fn test_bbm92_par_noiseless() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run_par(200, &channel, &channel, 0.0, 0.5).unwrap();
        assert_eq!(result.raw_length, 200);
        assert_eq!(result.check_errors, 0);
        assert_eq!(result.qber, 0.0);
        assert_eq!(result.alice_key, result.bob_key);
    }

    #[test]
    fn test_bbm92_par_deterministic_with_seed() {
        let channel = QuantumChannel::bit_flip(0.05);

        crate::rng::set_global_seed(11);
        let r1 = run_par(200, &channel, &channel, 0.1, 0.2).unwrap();
        crate::rng::set_global_seed(11);
        let r2 = run_par(200, &channel, &channel, 0.1, 0.2).unwrap();

        assert_eq!(r1.alice_bits, r2.alice_bits);
        assert_eq!(r1.alice_bases, r2.alice_bases);
        assert_eq!(r1.bob_bases, r2.bob_bases);
        assert_eq!(r1.bob_results, r2.bob_results);
        assert_eq!(r1.alice_key, r2.alice_key);
        assert_eq!(r1.bob_key, r2.bob_key);
    }

    #[test]
    fn test_bbm92_par_eve() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run_par(2000, &channel, &channel, 1.0, 0.5).unwrap();

        assert!(result.eve_intercept_count > 0);
        assert!((result.qber - 0.25).abs() < 0.05);
    }
}
