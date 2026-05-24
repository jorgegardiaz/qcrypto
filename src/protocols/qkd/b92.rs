//! B92 Quantum Key Distribution Protocol.
//!
//! B92 is a simplified version of BB84 proposed by Charles Bennett in 1992.
//! It uses only two non-orthogonal quantum states (e.g., |0> and |+>).

use crate::rng::LocalRng;
use crate::{
    Gate, Measurement, QuantumChannel, QuantumState,
    errors::{MeasurementError, StateError},
};
use ndarray::{Array2, arr2};
use num_complex::Complex64;
use rayon::prelude::*;

/// The result of the B92 protocol execution.
pub struct B92Result {
    /// The total length of the raw key (number of qubits sent).
    pub raw_length: usize,
    /// The length of the conclusive key (after sifting, where Bob gets a conclusive result).
    pub conclusive_count: usize,
    /// The number of errors found in the check bits.
    pub check_errors: usize,
    /// The Quantum Bit Error Rate (QBER) calculated on the check bits.
    pub qber: f64,
    /// The number of times Eve was detected (simulated).
    pub eve_intercepted_count: usize,
    /// Alice's final key (conclusive results excluding check bits).
    pub alice_key: Vec<bool>,
    /// Bob's final key (conclusive results excluding check bits).
    pub bob_key: Vec<bool>,
    /// Alice's original bits.
    pub alice_bits: Vec<bool>,
    /// Bob's measurement results (0: bit 1, 1: bit 0, -1: inconclusive).
    pub bob_results: Vec<i8>,
}

/// Executes the B92 QKD protocol.
///
/// In B92, Alice sends one of two non-orthogonal states:
/// - Bit 0 -> $|0\rangle$
/// - Bit 1 -> $|+\rangle$ (Hadamard state)
///
/// Bob measures using a POVM that can conclusively identify the bit or return an inconclusive result.
///
/// # Arguments
///
/// * `num_qubits` - Number of qubits to transmit.
/// * `channel` - The quantum channel (noise model).
/// * `measurement` - Bob's POVM measurement device.
/// * `eve_ratio` - Probability of Eve intercepting a qubit.
/// * `check_ratio` - Fraction of conclusive bits to sacrifice for QBER estimation (0.0 to 1.0).
///
/// # Returns
///
/// A `Result` containing `B92Result` with the simulation statistics and keys.
///
/// # Errors
///
/// Returns a `StateError` if quantum operations fail.
///
/// # Example
/// ```rust
/// use qcrypto::protocols::b92;
/// use qcrypto::QuantumChannel;
///
/// let channel = QuantumChannel::bit_flip(0.0);
/// let measurement = b92::build_optimal_povm_b92().unwrap();
/// let result = b92::run(100, &channel, &measurement, 0.0, 0.5).unwrap();
///
/// assert_eq!(result.raw_length, 100);
/// ```
pub fn run(
    num_qubits: usize,
    channel: &QuantumChannel,
    measurement: &Measurement,
    eve_ratio: f64,
    check_ratio: f64,
) -> Result<B92Result, StateError> {
    // Bob's POVM
    let bob_device = measurement;

    let mut alice_bits = Vec::with_capacity(num_qubits);
    let mut bob_results = Vec::with_capacity(num_qubits);
    let mut eve_intercepted_count = 0;

    for _ in 0..num_qubits {
        // Alice prepare qubit
        let a_bit = crate::rng::random_bool(0.5);
        let mut state = QuantumState::new(1);

        if a_bit {
            state.apply(&Gate::h(), &[0])?;
        }

        // Alice send qubit to Bob through channel
        state.apply_channel(channel, &[0])?;

        // Eavesdropper intercepts
        if eve_ratio > 0.0 && crate::rng::random_bool(eve_ratio) {
            eve_intercepted_count += 1;
            let e_basis = crate::rng::random_bool(0.5);
            let m = if e_basis {
                Measurement::x_basis()
            } else {
                Measurement::z_basis()
            };
            let _ = state.measure(&m, &[0])?;
        }

        // Eve send qubit to Bob through channel
        state.apply_channel(channel, &[0])?;

        // Bob measures using his POVM
        let res = state.measure(bob_device, &[0])?;

        let inferred_bit_opt = match res.index {
            0 => Some(true),  // Detected E1 -> Bit 1
            1 => Some(false), // Detected E2 -> Bit 0
            _ => None,        // Detected E3 -> Inconclusive
        };

        alice_bits.push(a_bit);

        let res_code = match inferred_bit_opt {
            Some(true) => 0,
            Some(false) => 1,
            None => -1,
        };
        bob_results.push(res_code);
    }

    // Sifting Stage
    // 1. Identify indices where Bob got a conclusive result
    let mut conclusive_indices: Vec<usize> = bob_results
        .iter()
        .enumerate()
        .filter_map(|(i, &res)| if res != -1 { Some(i) } else { None })
        .collect();

    let total_conclusive = conclusive_indices.len();

    // 2. Shuffle indices to randomly select bits for error checking
    crate::rng::shuffle_slice(&mut conclusive_indices);

    // 3. Split into check bits and key bits
    let num_check = (total_conclusive as f64 * check_ratio).round() as usize;
    let (check_indices, key_indices) = conclusive_indices.split_at(num_check);

    // 4. Calculate QBER on check bits
    let mut check_errors = 0;
    for &idx in check_indices {
        // bob_results[idx] is 0 or 1 here (conclusive)
        let b_val = bob_results[idx] == 0; // 0 -> true, 1 -> false (based on res_code logic above: 0->true, 1->false)
        if alice_bits[idx] != b_val {
            check_errors += 1;
        }
    }

    let qber = if num_check > 0 {
        check_errors as f64 / num_check as f64
    } else {
        0.0
    };

    // 5. Build established keys from key bits
    let mut alice_key = Vec::with_capacity(key_indices.len());
    let mut bob_key = Vec::with_capacity(key_indices.len());
    for &idx in key_indices {
        // bob_results[idx] is 0 or 1 here (conclusive)
        // From run logic: index 0 -> bit 1, index 1 -> bit 0
        let b_val = bob_results[idx] == 0;
        alice_key.push(alice_bits[idx]);
        bob_key.push(b_val);
    }

    Ok(B92Result {
        raw_length: num_qubits,
        conclusive_count: total_conclusive,
        check_errors,
        qber,
        eve_intercepted_count,
        alice_key,
        bob_key,
        alice_bits,
        bob_results,
    })
}

/// Parallelized variant of [`run`] using rayon. See [`run`] for protocol semantics.
///
/// # Arguments
///
/// * `num_qubits` - Number of qubits to transmit.
/// * `channel` - The quantum channel (noise model).
/// * `measurement` - Bob's POVM measurement device.
/// * `eve_ratio` - Probability of Eve intercepting a qubit.
/// * `check_ratio` - Fraction of conclusive bits to sacrifice for QBER estimation (0.0 to 1.0).
///
/// # Returns
///
/// A `Result` containing `B92Result` with the simulation statistics and keys.
///
/// # Errors
///
/// Returns a `StateError` if quantum operations fail.
///
/// # Example
/// ```rust
/// use qcrypto::protocols::b92;
/// use qcrypto::QuantumChannel;
///
/// let channel = QuantumChannel::bit_flip(0.1);
/// let measurement = b92::build_optimal_povm_b92().unwrap();
///
/// qcrypto::rng::set_global_seed(42);
/// let r1 = b92::run_par(300, &channel, &measurement, 0.1, 0.2).unwrap();
///
/// qcrypto::rng::set_global_seed(42);
/// let r2 = b92::run_par(300, &channel, &measurement, 0.1, 0.2).unwrap();
///
/// assert_eq!(r1.alice_key, r2.alice_key);
/// ```
pub fn run_par(
    num_qubits: usize,
    channel: &QuantumChannel,
    measurement: &Measurement,
    eve_ratio: f64,
    check_ratio: f64,
) -> Result<B92Result, StateError> {
    let master = crate::rng::draw_master_seed();
    let process_seed = crate::rng::draw_master_seed();

    type Step = (bool, i8, bool);

    let steps: Vec<Step> = (0..num_qubits)
        .into_par_iter()
        .map(|i| -> Result<Step, StateError> {
            let mut rng = LocalRng::child(master, i as u64);

            let a_bit = rng.random_bool(0.5);
            let mut state = QuantumState::new(1);
            if a_bit {
                state.apply(&Gate::h(), &[0])?;
            }

            state.apply_channel(channel, &[0])?;

            let eve_intercepted = eve_ratio > 0.0 && rng.random_bool(eve_ratio);
            if eve_intercepted {
                let e_basis = rng.random_bool(0.5);
                let m = if e_basis {
                    Measurement::x_basis()
                } else {
                    Measurement::z_basis()
                };
                let _ = state.measure_with_rng(&m, &[0], &mut rng)?;
            }

            state.apply_channel(channel, &[0])?;

            let res = state.measure_with_rng(measurement, &[0], &mut rng)?;
            let res_code = match res.index {
                0 => 0i8,
                1 => 1i8,
                _ => -1i8,
            };

            Ok((a_bit, res_code, eve_intercepted))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut alice_bits = Vec::with_capacity(num_qubits);
    let mut bob_results = Vec::with_capacity(num_qubits);
    let mut eve_intercepted_count = 0usize;

    for (a_bit, res_code, eve_intercepted) in steps {
        alice_bits.push(a_bit);
        bob_results.push(res_code);
        if eve_intercepted {
            eve_intercepted_count += 1;
        }
    }

    let mut conclusive_indices: Vec<usize> = bob_results
        .iter()
        .enumerate()
        .filter_map(|(i, &res)| if res != -1 { Some(i) } else { None })
        .collect();

    let total_conclusive = conclusive_indices.len();
    let mut rng = LocalRng::from_seed(process_seed);
    rng.shuffle_slice(&mut conclusive_indices);

    let num_check = (total_conclusive as f64 * check_ratio).round() as usize;
    let (check_indices, key_indices) = conclusive_indices.split_at(num_check);

    let mut check_errors = 0;
    for &idx in check_indices {
        let b_val = bob_results[idx] == 0;
        if alice_bits[idx] != b_val {
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
    for &idx in key_indices {
        let b_val = bob_results[idx] == 0;
        alice_key.push(alice_bits[idx]);
        bob_key.push(b_val);
    }

    Ok(B92Result {
        raw_length: num_qubits,
        conclusive_count: total_conclusive,
        check_errors,
        qber,
        eve_intercepted_count,
        alice_key,
        bob_key,
        alice_bits,
        bob_results,
    })
}

/// Constructs the optimal POVM for the B92 protocol.
///
/// The POVM consists of three elements:
/// - $E_1$: Detects state $|+\rangle$ (implies bit 1 sent).
/// - $E_2$: Detects state $|0\rangle$ (implies bit 0 sent).
/// - $E_3$: Inconclusive result.
///
/// # Returns
///
/// A `Result` containing the `Measurement` POVM.
///
/// # Errors
///
/// Returns a `MeasurementError` if the POVM elements are invalid.
///
/// # Example
/// ```rust
/// use qcrypto::protocols::b92;
///
/// let povm = b92::build_optimal_povm_b92().unwrap();
/// assert_eq!(povm.operators.len(), 3);
/// ```
pub fn build_optimal_povm_b92() -> Result<Measurement, MeasurementError> {
    let zero = Complex64::new(0.0, 0.0);
    let sqrt2 = 2.0_f64.sqrt();

    let a_val = sqrt2 / (1.0 + sqrt2);
    let a = Complex64::new(a_val, 0.0);

    // 1. E1 = a * |1><1|
    let e1 = arr2(&[[zero, zero], [zero, a]]);

    // 2. E2 = a * |-><-|
    // |-><-| = 0.5 * [[1, -1], [-1, 1]]
    let half_a = Complex64::new(a_val * 0.5, 0.0);
    let e2 = arr2(&[[half_a, -half_a], [-half_a, half_a]]);

    // 3. E3 = I - E1 - E2
    let identity = Array2::<Complex64>::eye(2);
    let e3 = identity - &e1 - &e2;

    Measurement::from_povm(
        vec![e1, e2, e3],
        vec![1.0, 0.0, -1.0],
        vec!["1".to_string(), "0".to_string(), "-1".to_string()],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_b92_noiseless() {
        let channel = QuantumChannel::bit_flip(0.0);
        let measurement = build_optimal_povm_b92().unwrap();
        let result = run(100, &channel, &measurement, 0.0, 0.5).unwrap();

        assert_eq!(result.raw_length, 100);
        assert_eq!(result.check_errors, 0);
        assert_eq!(result.qber, 0.0);
        assert_eq!(result.eve_intercepted_count, 0);
    }

    #[test]
    fn test_b92_keys_equal_noiseless() {
        let channel = QuantumChannel::bit_flip(0.0);
        let measurement = build_optimal_povm_b92().unwrap();
        let result = run(500, &channel, &measurement, 0.0, 0.0).unwrap();

        assert_eq!(result.alice_key.len(), result.bob_key.len());
        assert_eq!(result.alice_key, result.bob_key);
    }

    #[test]
    fn test_b92_noisy_keys_differ() {
        let channel = QuantumChannel::bit_flip(0.3);
        let measurement = build_optimal_povm_b92().unwrap();
        let result = run(1000, &channel, &measurement, 0.0, 0.0).unwrap();

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
    fn test_b92_eve() {
        let channel = QuantumChannel::bit_flip(0.0);
        let measurement = build_optimal_povm_b92().unwrap();
        // Eve intercepts everything.
        // B92 is very sensitive to intercept-and-resend.
        let result = run(1000, &channel, &measurement, 1.0, 0.5).unwrap();

        assert!(result.eve_intercepted_count > 0);
        // In B92, if Eve measures in the wrong basis (50%), she can resend a state
        // that Bob's POVM detects as the *wrong* conclusive result.
        assert!(
            result.qber > 0.15,
            "QBER {} should be significant",
            result.qber
        );
    }

    #[test]
    fn test_b92_zero_check() {
        let channel = QuantumChannel::bit_flip(0.0);
        let measurement = build_optimal_povm_b92().unwrap();
        let result = run(100, &channel, &measurement, 0.0, 0.0).unwrap();
        assert_eq!(result.qber, 0.0);
    }

    #[test]
    fn test_b92_par_zero_check() {
        let channel = QuantumChannel::bit_flip(0.0);
        let measurement = build_optimal_povm_b92().unwrap();
        let result = run_par(100, &channel, &measurement, 0.0, 0.0).unwrap();
        assert_eq!(result.qber, 0.0);
    }

    #[test]
    fn test_b92_par_noiseless() {
        let channel = QuantumChannel::bit_flip(0.0);
        let measurement = build_optimal_povm_b92().unwrap();
        let result = run_par(200, &channel, &measurement, 0.0, 0.5).unwrap();
        assert_eq!(result.raw_length, 200);
        assert_eq!(result.check_errors, 0);
        assert_eq!(result.qber, 0.0);
        assert_eq!(result.eve_intercepted_count, 0);
    }

    #[test]
    fn test_b92_par_deterministic_with_seed() {
        let channel = QuantumChannel::bit_flip(0.05);
        let measurement = build_optimal_povm_b92().unwrap();

        crate::rng::set_global_seed(7);
        let r1 = run_par(300, &channel, &measurement, 0.1, 0.2).unwrap();
        crate::rng::set_global_seed(7);
        let r2 = run_par(300, &channel, &measurement, 0.1, 0.2).unwrap();

        assert_eq!(r1.alice_bits, r2.alice_bits);
        assert_eq!(r1.bob_results, r2.bob_results);
        assert_eq!(r1.alice_key, r2.alice_key);
        assert_eq!(r1.bob_key, r2.bob_key);
    }

    #[test]
    fn test_b92_par_eve() {
        let channel = QuantumChannel::bit_flip(0.0);
        let measurement = build_optimal_povm_b92().unwrap();
        let result = run_par(1000, &channel, &measurement, 1.0, 0.5).unwrap();
        assert!(result.eve_intercepted_count > 0);
        assert!(result.qber > 0.15);
    }
}
