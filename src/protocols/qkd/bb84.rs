//! BB84 Quantum Key Distribution Protocol.
//!
//! BB84 is the first quantum cryptography protocol, developed by Charles Bennett and Gilles Brassard in 1984.
//! It uses four quantum states from two mutually unbiased bases (e.g., rectilinear and diagonal)
//! to securely establish a shared secret key.

use crate::rng::LocalRng;
use crate::{Gate, Measurement, QuantumChannel, QuantumState, errors::StateError};
use rayon::prelude::*;

/// The result of the BB84 protocol execution.
pub struct BB84Result {
    /// The total length of the raw key (number of qubits sent).
    pub raw_length: usize,
    /// The number of bits where bases matched (before sacrificing).
    pub total_sifted: usize,
    /// The number of errors found in the check bits.
    pub check_errors: usize,
    /// The Quantum Bit Error Rate (QBER) in percentage (on check bits).
    pub qber: f64,
    /// The number of times Eve was detected (simulated).
    pub eve_intercepted_count: usize,
    /// Alice's final key (sifted key minus check bits).
    pub alice_key: Vec<bool>,
    /// Bob's final key (sifted key minus check bits).
    pub bob_key: Vec<bool>,
    /// Alice's original bits.
    pub alice_bits: Vec<bool>,
    /// Alice's chosen bases (0: Z, 1: X).
    pub alice_bases: Vec<bool>,
    /// Bob's chosen bases (0: Z, 1: X).
    pub bob_bases: Vec<bool>,
    /// Bob's measurement results.
    pub bob_results: Vec<bool>,
}

/// Executes the BB84 QKD protocol.
///
/// In BB84, Alice prepares qubits in one of four states ($|0\rangle, |1\rangle, |+\rangle, |-\rangle$)
/// chosen by a random bit and a random basis (Z or X).
/// Bob measures in a random basis (Z or X).
///
/// # Arguments
///
/// * `num_qubits` - Number of qubits to transmit.
/// * `channel` - The quantum channel (noise model).
/// * `eve_ratio` - Probability of Eve intercepting (and measuring) a qubit.
/// * `check_ratio` - Fraction of sifted bits to sacrifice for QBER estimation.
///
/// # Returns
///
/// A `Result` containing `BB84Result` with the simulation statistics and keys.
///
/// # Errors
///
/// Returns a `StateError` if quantum operations fail (e.g. invalid dimensions).
///
/// # Example
/// ```rust
/// use qcrypto::protocols::bb84;
/// use qcrypto::QuantumChannel;
///
/// let channel = QuantumChannel::bit_flip(0.0);
/// let result = bb84::run(100, &channel, 0.0, 0.5).unwrap();
///
/// assert_eq!(result.raw_length, 100);
/// ```
pub fn run(
    num_qubits: usize,
    channel: &QuantumChannel,
    eve_ratio: f64,
    check_ratio: f64,
) -> Result<BB84Result, StateError> {
    let mut alice_bits = Vec::with_capacity(num_qubits);
    let mut alice_bases = Vec::with_capacity(num_qubits);
    let mut bob_bases = Vec::with_capacity(num_qubits);
    let mut bob_results = Vec::with_capacity(num_qubits);

    let mut eve_intercepted_count = 0;

    for _ in 0..num_qubits {
        // Alice prepares qubits
        let a_bit = crate::rng::random_bool(0.5);
        let a_basis = crate::rng::random_bool(0.5);

        let mut state = QuantumState::new(1);

        if a_bit {
            state.apply(&Gate::x(), &[0])?;
        }
        if a_basis {
            state.apply(&Gate::h(), &[0])?;
        }

        // Alice sends qubit to Bob
        state.apply_channel(channel, &[0])?;

        // Eavesdropper Intercepts
        if eve_ratio > 0.0 && crate::rng::random_bool(eve_ratio) {
            eve_intercepted_count += 1;

            let e_basis = crate::rng::random_bool(0.5);
            let measurement = if e_basis {
                Measurement::x_basis()
            } else {
                Measurement::z_basis()
            };

            let _ = state.measure(&measurement, &[0])?;
        }

        // Eve send qubit to Bob through channel
        state.apply_channel(channel, &[0])?;

        // Bob measures
        let b_basis = crate::rng::random_bool(0.5);
        let measurement = if b_basis {
            Measurement::x_basis()
        } else {
            Measurement::z_basis()
        };

        let result = state.measure(&measurement, &[0])?;

        let b_val = result.value as usize == 1;

        alice_bits.push(a_bit);
        alice_bases.push(a_basis);
        bob_bases.push(b_basis);
        bob_results.push(b_val);
    }

    // Sifting stage
    // 1. Identify indices where bases match
    let input_indices: Vec<usize> = (0..num_qubits).collect();
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

    // 5. Build established keys
    let mut alice_key = Vec::with_capacity(key_indices.len());
    let mut bob_key = Vec::with_capacity(key_indices.len());
    for &i in key_indices {
        alice_key.push(alice_bits[i]);
        bob_key.push(bob_results[i]);
    }

    Ok(BB84Result {
        raw_length: num_qubits,
        total_sifted,
        check_errors,
        qber,
        eve_intercepted_count,
        alice_key,
        bob_key,
        alice_bits,
        alice_bases,
        bob_bases,
        bob_results,
    })
}

/// Parallelized variant of [`run`] using rayon. See [`run`] for protocol semantics.
///
/// # Arguments
///
/// * `num_qubits` - Number of qubits to transmit.
/// * `channel` - The quantum channel (noise model).
/// * `eve_ratio` - Probability of Eve intercepting (and measuring) a qubit.
/// * `check_ratio` - Fraction of sifted bits to sacrifice for QBER estimation.
///
/// # Returns
///
/// A `Result` containing `BB84Result` with the simulation statistics and keys.
///
/// # Errors
///
/// Returns a `StateError` if quantum operations fail (e.g. invalid dimensions).
///
/// # Example
/// ```rust
/// use qcrypto::protocols::bb84;
/// use qcrypto::QuantumChannel;
///
/// let channel = QuantumChannel::bit_flip(0.1);
///
/// qcrypto::rng::set_global_seed(42);
/// let r1 = bb84::run_par(300, &channel, 0.1, 0.2).unwrap();
///
/// qcrypto::rng::set_global_seed(42);
/// let r2 = bb84::run_par(300, &channel, 0.1, 0.2).unwrap();
///
/// assert_eq!(r1.alice_key, r2.alice_key);
/// ```
pub fn run_par(
    num_qubits: usize,
    channel: &QuantumChannel,
    eve_ratio: f64,
    check_ratio: f64,
) -> Result<BB84Result, StateError> {
    let master = crate::rng::draw_master_seed();
    let process_seed = crate::rng::draw_master_seed();

    type Step = (bool, bool, bool, bool, bool);

    let steps: Vec<Step> = (0..num_qubits)
        .into_par_iter()
        .map(|i| -> Result<Step, StateError> {
            let mut rng = LocalRng::child(master, i as u64);

            let a_bit = rng.random_bool(0.5);
            let a_basis = rng.random_bool(0.5);

            let mut state = QuantumState::new(1);
            if a_bit {
                state.apply(&Gate::x(), &[0])?;
            }
            if a_basis {
                state.apply(&Gate::h(), &[0])?;
            }

            state.apply_channel(channel, &[0])?;

            let eve_intercepted = eve_ratio > 0.0 && rng.random_bool(eve_ratio);
            if eve_intercepted {
                let e_basis = rng.random_bool(0.5);
                let measurement = if e_basis {
                    Measurement::x_basis()
                } else {
                    Measurement::z_basis()
                };
                let _ = state.measure_with_rng(&measurement, &[0], &mut rng)?;
            }

            state.apply_channel(channel, &[0])?;

            let b_basis = rng.random_bool(0.5);
            let measurement = if b_basis {
                Measurement::x_basis()
            } else {
                Measurement::z_basis()
            };
            let result = state.measure_with_rng(&measurement, &[0], &mut rng)?;
            let b_val = result.value as usize == 1;

            Ok((a_bit, a_basis, b_basis, b_val, eve_intercepted))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut alice_bits = Vec::with_capacity(num_qubits);
    let mut alice_bases = Vec::with_capacity(num_qubits);
    let mut bob_bases = Vec::with_capacity(num_qubits);
    let mut bob_results = Vec::with_capacity(num_qubits);
    let mut eve_intercepted_count = 0usize;

    for (a_bit, a_basis, b_basis, b_val, eve_intercepted) in steps {
        alice_bits.push(a_bit);
        alice_bases.push(a_basis);
        bob_bases.push(b_basis);
        bob_results.push(b_val);
        if eve_intercepted {
            eve_intercepted_count += 1;
        }
    }

    let mut match_indices: Vec<usize> = (0..num_qubits)
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

    Ok(BB84Result {
        raw_length: num_qubits,
        total_sifted,
        check_errors,
        qber,
        eve_intercepted_count,
        alice_key,
        bob_key,
        alice_bits,
        alice_bases,
        bob_bases,
        bob_results,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bb84_noiseless() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(100, &channel, 0.0, 0.5).unwrap();

        assert_eq!(result.raw_length, 100);
        assert_eq!(result.check_errors, 0);
        assert_eq!(result.qber, 0.0);
        assert_eq!(result.eve_intercepted_count, 0);
    }

    #[test]
    fn test_bb84_keys_equal_noiseless() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(200, &channel, 0.0, 0.0).unwrap();

        assert_eq!(result.alice_key.len(), result.bob_key.len());
        assert_eq!(result.alice_key, result.bob_key);
    }

    #[test]
    fn test_bb84_noisy_keys_differ() {
        let channel = QuantumChannel::bit_flip(0.3);
        let result = run(500, &channel, 0.0, 0.0).unwrap();

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
    fn test_bb84_eve() {
        let channel = QuantumChannel::bit_flip(0.0);
        // Eve intercepts everything.
        // Expected QBER for BB84: (1/2) * (1/2) = 1/4 = 0.25
        // 5000 qubits -> ~1250 check bits -> σ≈0.012, tolerance 0.06 covers ~5σ
        let result = run(5000, &channel, 1.0, 0.5).unwrap();

        assert!(result.eve_intercepted_count > 0);
        assert!(
            (result.qber - 0.25).abs() < 0.06,
            "QBER {} should be around 0.25",
            result.qber
        );
    }

    #[test]
    fn test_bb84_zero_check() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(100, &channel, 0.0, 0.0).unwrap();
        assert_eq!(result.qber, 0.0);
    }

    #[test]
    fn test_bb84_par_zero_check() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run_par(100, &channel, 0.0, 0.0).unwrap();
        assert_eq!(result.qber, 0.0);
    }

    #[test]
    fn test_bb84_par_noiseless() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run_par(200, &channel, 0.0, 0.5).unwrap();
        assert_eq!(result.raw_length, 200);
        assert_eq!(result.check_errors, 0);
        assert_eq!(result.qber, 0.0);
        assert_eq!(result.eve_intercepted_count, 0);
    }

    #[test]
    fn test_bb84_par_deterministic_with_seed() {
        let channel = QuantumChannel::bit_flip(0.05);

        crate::rng::set_global_seed(123);
        let r1 = run_par(300, &channel, 0.1, 0.2).unwrap();
        crate::rng::set_global_seed(123);
        let r2 = run_par(300, &channel, 0.1, 0.2).unwrap();

        assert_eq!(r1.alice_bits, r2.alice_bits);
        assert_eq!(r1.alice_bases, r2.alice_bases);
        assert_eq!(r1.bob_bases, r2.bob_bases);
        assert_eq!(r1.bob_results, r2.bob_results);
        assert_eq!(r1.alice_key, r2.alice_key);
        assert_eq!(r1.bob_key, r2.bob_key);
        assert_eq!(r1.eve_intercepted_count, r2.eve_intercepted_count);
    }

    #[test]
    fn test_bb84_par_eve() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run_par(5000, &channel, 1.0, 0.5).unwrap();
        assert!(result.eve_intercepted_count > 0);
        assert!(
            (result.qber - 0.25).abs() < 0.06,
            "QBER {} should be around 0.25",
            result.qber
        );
    }
}
