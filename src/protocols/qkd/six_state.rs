//! Six-State Quantum Key Distribution Protocol.
//!
//! Six-State is a QKD protocol proposed by Pasquinucci and Gisin in 1999.
//! It is an extension of BB84 that uses three mutually unbiased bases (Z, X, and Y)
//! instead of two, providing higher security against eavesdropping.

use crate::rng::LocalRng;
use crate::{Gate, Measurement, QuantumChannel, QuantumState, errors::StateError};
use rayon::prelude::*;

/// The result of the Six-State protocol execution.
pub struct SixStateResult {
    /// The total length of the raw key (number of qubits sent).
    pub raw_length: usize,
    /// The number of bits where bases matched (before sacrificing).
    pub total_sifted: usize,
    /// The number of errors found in the check bits.
    pub check_errors: usize,
    /// The Quantum Bit Error Rate (QBER) on check bits.
    pub qber: f64,
    /// The number of times Eve was detected (simulated).
    pub eve_intercepted_count: usize,
    /// Alice's final key (sifted key minus check bits).
    pub alice_key: Vec<bool>,
    /// Bob's final key (sifted key minus check bits).
    pub bob_key: Vec<bool>,
    /// Alice's original bits.
    pub alice_bits: Vec<bool>,
    /// Alice's chosen bases (0: Z, 1: X, 2: Y).
    pub alice_bases: Vec<usize>,
    /// Bob's chosen bases (0: Z, 1: X, 2: Y).
    pub bob_bases: Vec<usize>,
    /// Bob's measurement results.
    pub bob_results: Vec<bool>,
}

/// Executes the Six-State QKD protocol.
///
/// In Six-State, Alice prepares qubits in one of six states:
/// - Basis Z (0): |0>, |1>
/// - Basis X (1): |+>, |->
/// - Basis Y (2): |+i>, |-i>
///
/// Bob measures in a random basis (Z, X or Y).
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
/// A `Result` containing `SixStateResult` with the simulation statistics and keys.
///
/// # Errors
///
/// Returns a `StateError` if quantum operations fail.
///
/// # Example
/// ```rust
/// use qcrypto::protocols::six_state;
/// use qcrypto::QuantumChannel;
///
/// let channel = QuantumChannel::bit_flip(0.0);
/// let result = six_state::run(100, &channel, 0.0, 0.5).unwrap();
///
/// assert_eq!(result.raw_length, 100);
/// ```
pub fn run(
    num_qubits: usize,
    channel: &QuantumChannel,
    eve_ratio: f64,
    check_ratio: f64,
) -> Result<SixStateResult, StateError> {
    let mut alice_bits = Vec::with_capacity(num_qubits);
    let mut alice_bases = Vec::with_capacity(num_qubits);
    let mut bob_bases = Vec::with_capacity(num_qubits);
    let mut bob_results = Vec::with_capacity(num_qubits);

    let mut eve_intercepted_count = 0;

    for _ in 0..num_qubits {
        // Alice prepares qubits
        let a_bit = crate::rng::random_bool(0.5);
        let a_basis = crate::rng::random_usize_range(0, 3);

        let mut state = QuantumState::new(1);

        if a_bit {
            state.apply(&Gate::x(), &[0])?;
        }

        match a_basis {
            1 => {
                // X basis: H |0> = |+>, H |1> = |->
                state.apply(&Gate::h(), &[0])?;
            }
            2 => {
                // Y basis: S H |0> = |+i>, S H |1> = |-i>
                state.apply(&Gate::h(), &[0])?.apply(&Gate::s(), &[0])?;
            }
            _ => {} // Z basis: |0> or |1>
        }

        // Alice sends qubit to Bob
        state.apply_channel(channel, &[0])?;

        // Eavesdropper Intercepts
        if eve_ratio > 0.0 && crate::rng::random_bool(eve_ratio) {
            eve_intercepted_count += 1;

            let e_basis = crate::rng::random_usize_range(0, 3);
            let measurement = match e_basis {
                1 => Measurement::x_basis(),
                2 => Measurement::y_basis(),
                _ => Measurement::z_basis(),
            };

            state.measure(&measurement, &[0])?;

            // Eve send qubit to Bob through channel
            state.apply_channel(channel, &[0])?;
        }

        // Bob measures
        let b_basis = crate::rng::random_usize_range(0, 3);
        let measurement = match b_basis {
            1 => Measurement::x_basis(),
            2 => Measurement::y_basis(),
            _ => Measurement::z_basis(),
        };

        let res = state.measure(&measurement, &[0])?;
        let b_val = res.index == 1;

        alice_bits.push(a_bit);
        alice_bases.push(a_basis);
        bob_bases.push(b_basis);
        bob_results.push(b_val);
    }

    // Sifting stage
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

    Ok(SixStateResult {
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
/// A `Result` containing `SixStateResult` with the simulation statistics and keys.
///
/// # Errors
///
/// Returns a `StateError` if quantum operations fail.
///
/// # Example
/// ```rust
/// use qcrypto::protocols::six_state;
/// use qcrypto::QuantumChannel;
///
/// let channel = QuantumChannel::bit_flip(0.1);
///
/// qcrypto::rng::set_global_seed(42);
/// let r1 = six_state::run_par(300, &channel, 0.1, 0.2).unwrap();
///
/// qcrypto::rng::set_global_seed(42);
/// let r2 = six_state::run_par(300, &channel, 0.1, 0.2).unwrap();
///
/// assert_eq!(r1.alice_key, r2.alice_key);
/// ```
pub fn run_par(
    num_qubits: usize,
    channel: &QuantumChannel,
    eve_ratio: f64,
    check_ratio: f64,
) -> Result<SixStateResult, StateError> {
    let master = crate::rng::draw_master_seed();
    let process_seed = crate::rng::draw_master_seed();

    type Step = (bool, usize, usize, bool, bool);

    let steps: Vec<Step> = (0..num_qubits)
        .into_par_iter()
        .map(|i| -> Result<Step, StateError> {
            let mut rng = LocalRng::child(master, i as u64);

            let a_bit = rng.random_bool(0.5);
            let a_basis = rng.random_usize_range(0, 3);

            let mut state = QuantumState::new(1);
            if a_bit {
                state.apply(&Gate::x(), &[0])?;
            }
            match a_basis {
                1 => {
                    state.apply(&Gate::h(), &[0])?;
                }
                2 => {
                    state.apply(&Gate::h(), &[0])?.apply(&Gate::s(), &[0])?;
                }
                _ => {}
            }

            state.apply_channel(channel, &[0])?;

            let eve_intercepted = eve_ratio > 0.0 && rng.random_bool(eve_ratio);
            if eve_intercepted {
                let e_basis = rng.random_usize_range(0, 3);
                let measurement = match e_basis {
                    1 => Measurement::x_basis(),
                    2 => Measurement::y_basis(),
                    _ => Measurement::z_basis(),
                };
                state.measure_with_rng(&measurement, &[0], &mut rng)?;

                state.apply_channel(channel, &[0])?;
            }

            let b_basis = rng.random_usize_range(0, 3);
            let measurement = match b_basis {
                1 => Measurement::x_basis(),
                2 => Measurement::y_basis(),
                _ => Measurement::z_basis(),
            };
            let res = state.measure_with_rng(&measurement, &[0], &mut rng)?;
            let b_val = res.index == 1;

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

    Ok(SixStateResult {
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
    fn test_six_state_noiseless() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(300, &channel, 0.0, 0.5).unwrap();

        assert_eq!(result.raw_length, 300);
        assert_eq!(result.check_errors, 0);
        assert_eq!(result.qber, 0.0);
        assert_eq!(result.eve_intercepted_count, 0);
        // Sifted rate should be around 1/3
        assert!(result.total_sifted > 50 && result.total_sifted < 150);
    }

    #[test]
    fn test_six_state_keys_equal_noiseless() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(300, &channel, 0.0, 0.0).unwrap();

        assert_eq!(result.alice_key.len(), result.bob_key.len());
        assert_eq!(result.alice_key, result.bob_key);
    }

    #[test]
    fn test_six_state_noisy_keys_differ() {
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
    fn test_six_state_eve() {
        let channel = QuantumChannel::bit_flip(0.0);
        // Eve intercepts everything.
        // Expected QBER for 6-state: (2/3) * (1/2) = 1/3
        // 5000 qubits -> ~833 check bits (1/3 sifted * 0.5) -> σ≈0.016, tolerance 0.06 covers ~4σ
        let result = run(5000, &channel, 1.0, 0.5).unwrap();

        assert!(result.eve_intercepted_count > 0);
        assert!(
            (result.qber - 0.333).abs() < 0.06,
            "QBER {} should be around 0.333",
            result.qber
        );
    }

    #[test]
    fn test_six_state_zero_check() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(100, &channel, 0.0, 0.0).unwrap();
        assert_eq!(result.qber, 0.0);
    }

    #[test]
    fn test_six_state_par_zero_check() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run_par(100, &channel, 0.0, 0.0).unwrap();
        assert_eq!(result.qber, 0.0);
    }

    #[test]
    fn test_six_state_par_noiseless() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run_par(300, &channel, 0.0, 0.5).unwrap();
        assert_eq!(result.raw_length, 300);
        assert_eq!(result.check_errors, 0);
        assert_eq!(result.qber, 0.0);
    }

    #[test]
    fn test_six_state_par_deterministic_with_seed() {
        let channel = QuantumChannel::bit_flip(0.05);

        crate::rng::set_global_seed(55);
        let r1 = run_par(300, &channel, 0.1, 0.2).unwrap();
        crate::rng::set_global_seed(55);
        let r2 = run_par(300, &channel, 0.1, 0.2).unwrap();

        assert_eq!(r1.alice_bits, r2.alice_bits);
        assert_eq!(r1.alice_bases, r2.alice_bases);
        assert_eq!(r1.bob_bases, r2.bob_bases);
        assert_eq!(r1.bob_results, r2.bob_results);
        assert_eq!(r1.alice_key, r2.alice_key);
        assert_eq!(r1.bob_key, r2.bob_key);
    }

    #[test]
    fn test_six_state_par_eve() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run_par(5000, &channel, 1.0, 0.5).unwrap();

        assert!(result.eve_intercepted_count > 0);
        assert!((result.qber - 0.333).abs() < 0.06);
    }
}
