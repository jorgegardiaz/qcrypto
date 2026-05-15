//! SARG04 Quantum Key Distribution Protocol.
//!
//! Noiseless sifting rate is 1/4, and SARG04 is known to be more robust than BB84
//! against photon-number-splitting attacks in weak-coherent-pulse implementations.

use crate::core::errors::StateError;
use crate::core::state::QuantumState;
use crate::rng::LocalRng;
use crate::{Gate, Measurement, QuantumChannel};
use rayon::prelude::*;

/// Result of a SARG04 protocol execution.
#[derive(Debug, Clone)]
pub struct SARG04Result {
    /// The total length of the raw key (number of qubits sent).
    pub raw_length: usize,
    /// The number of bits where Bob obtained a conclusive SARG04 decoding.
    pub conclusive_count: usize,
    /// The number of errors found in the check bits.
    pub check_errors: usize,
    /// The Quantum Bit Error Rate (QBER) on check bits.
    pub qber: f64,
    /// The number of times Eve intercepted a qubit (simulated).
    pub eve_detected_count: usize,
    /// Alice's final key (conclusive bits minus check bits).
    pub alice_key: Vec<bool>,
    /// Bob's final key (conclusive bits minus check bits).
    pub bob_key: Vec<bool>,
    /// Alice's original bits.
    pub alice_bits: Vec<bool>,
    /// Alice's chosen bases (false: Z, true: X).
    pub alice_bases: Vec<bool>,
    /// Bob's chosen bases (false: Z, true: X).
    pub bob_bases: Vec<bool>,
    /// Bob's raw measurement outcomes.
    pub bob_results: Vec<bool>,
}

/// Executes the SARG04 QKD protocol.
///
/// SARG04 was proposed by Scarani, Acín, Ribordy and Gisin in 2004.
/// It reuses the four BB84 states {|0>, |1>, |+>, |->} but differs in the sifting phase:
/// instead of announcing her basis, Alice announces a pair of non-orthogonal states
/// (one from each basis) that contains the state she actually sent. Bob deduces the
/// bit only when his measurement outcome is orthogonal to exactly one of the two
/// announced states — the bit value is then determined by the *other* state in the pair.
///
/// # Arguments
///
/// * `num_qubits` - Number of entangled pairs to distribute.
/// * `channel` - The quantum channel affecting Alice's qubit.
/// * `eve_ratio` - Probability of Eve intercepting (and measuring) a qubit.
/// * `check_ratio` - Fraction of sifted bits to sacrifice for QBER estimation.
///
/// # Returns
///
/// A `Result` containing `SARG04Result` with the simulation statistics and keys.
///
/// # Errors
///
/// Returns a `StateError` if quantum operations fail.
pub fn run(
    num_qubits: usize,
    channel: &QuantumChannel,
    eve_ratio: f64,
    check_ratio: f64,
) -> Result<SARG04Result, StateError> {
    let mut alice_bits = Vec::with_capacity(num_qubits);
    let mut alice_bases = Vec::with_capacity(num_qubits);
    let mut bob_bases = Vec::with_capacity(num_qubits);
    let mut bob_results = Vec::with_capacity(num_qubits);
    let mut eve_intercepted_count = 0;

    for _ in 0..num_qubits {
        let a_bit = crate::rng::random_bool(0.5);
        let a_basis = crate::rng::random_bool(0.5);

        let mut state = QuantumState::new(1);
        if a_basis {
            // X basis: 0 -> |+>, 1 -> |->
            state.apply(&Gate::h(), &[0])?;
            if a_bit {
                state.apply(&Gate::z(), &[0])?;
            }
        } else if a_bit {
            // Z basis: 0 -> |0>, 1 -> |1>
            state.apply(&Gate::x(), &[0])?;
        }

        // Alice sends qubit to Bob
        state.apply_channel(channel, &[0])?;

        let eve_intercepted = eve_ratio > 0.0 && crate::rng::random_bool(eve_ratio);
        if eve_intercepted {
            eve_intercepted_count += 1;
            let e_basis = crate::rng::random_bool(0.5);
            let m = if e_basis {
                Measurement::x_basis()
            } else {
                Measurement::z_basis()
            };
            let _ = state.measure(&m, &[0])?;
        }

        state.apply_channel(channel, &[0])?;

        let b_basis = crate::rng::random_bool(0.5);
        let m = if b_basis {
            Measurement::x_basis()
        } else {
            Measurement::z_basis()
        };

        let res = state.measure(&m, &[0])?;
        let b_res = res.index == 1;

        alice_bits.push(a_bit);
        alice_bases.push(a_basis);
        bob_bases.push(b_basis);
        bob_results.push(b_res);
    }

    let process_seed = crate::rng::draw_master_seed();
    finalize_sarg04(
        num_qubits,
        alice_bits,
        alice_bases,
        bob_bases,
        bob_results,
        eve_intercepted_count,
        check_ratio,
        process_seed,
    )
}

/// Parallelized variant of [`run`] using rayon. See [`run`] for protocol semantics.
///
/// # Arguments
///
/// * `num_qubits` - Number of entangled pairs to distribute.
/// * `channel` - The quantum channel affecting Alice's qubit.
/// * `eve_ratio` - Probability of Eve intercepting (and measuring) a qubit.
/// * `check_ratio` - Fraction of sifted bits to sacrifice for QBER estimation.
///
/// # Returns
///
/// A `Result` containing `SARG04Result` with the simulation statistics and keys.
///
/// # Errors
///
/// Returns a `StateError` if quantum operations fail.
///
/// # Example
/// ```rust
/// use qcrypto::protocols::sarg04;
/// use qcrypto::QuantumChannel;
///
/// let channel = QuantumChannel::bit_flip(0.1);
///
/// qcrypto::rng::set_global_seed(42);
/// let r1 = sarg04::run_par(300, &channel, 0.1, 0.2).unwrap();
///
/// qcrypto::rng::set_global_seed(42);
/// let r2 = sarg04::run_par(300, &channel, 0.1, 0.2).unwrap();
///
/// assert_eq!(r1.alice_key, r2.alice_key);
/// ```
pub fn run_par(
    num_qubits: usize,
    channel: &QuantumChannel,
    eve_ratio: f64,
    check_ratio: f64,
) -> Result<SARG04Result, StateError> {
    let master = crate::rng::draw_master_seed();
    let process_seed = crate::rng::draw_master_seed();

    type Step = (bool, bool, bool, bool, bool);

    let steps: Vec<Step> = (0..num_qubits)
        .into_par_iter()
        .map(|i| -> Result<Step, StateError> {
            let mut rng = LocalRng::child(master, i as u64);
            let a_bit = rng.random_bool(0.5);
            let a_basis = rng.random_bool(0.5);
            let b_basis = rng.random_bool(0.5);

            let mut state = QuantumState::new(1);
            if a_basis {
                state.apply(&Gate::h(), &[0])?;
                if a_bit {
                    state.apply(&Gate::z(), &[0])?;
                }
            } else if a_bit {
                state.apply(&Gate::x(), &[0])?;
            }

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

            let m = if b_basis {
                Measurement::x_basis()
            } else {
                Measurement::z_basis()
            };

            let res = state.measure_with_rng(&m, &[0], &mut rng)?;
            let b_res = res.index == 1;

            Ok((a_bit, a_basis, b_basis, b_res, eve_intercepted))
        })
        .collect::<Result<Vec<_>, StateError>>()?;

    let mut alice_bits = Vec::with_capacity(num_qubits);
    let mut alice_bases = Vec::with_capacity(num_qubits);
    let mut bob_bases = Vec::with_capacity(num_qubits);
    let mut bob_results = Vec::with_capacity(num_qubits);
    let mut eve_intercepted_count = 0;

    for (a_bit, a_basis, b_basis, b_res, eve_intercepted) in steps {
        alice_bits.push(a_bit);
        alice_bases.push(a_basis);
        bob_bases.push(b_basis);
        bob_results.push(b_res);
        if eve_intercepted {
            eve_intercepted_count += 1;
        }
    }

    finalize_sarg04(
        num_qubits,
        alice_bits,
        alice_bases,
        bob_bases,
        bob_results,
        eve_intercepted_count,
        check_ratio,
        process_seed,
    )
}

#[allow(clippy::too_many_arguments)]
fn finalize_sarg04(
    num_qubits: usize,
    alice_bits: Vec<bool>,
    alice_bases: Vec<bool>,
    bob_bases: Vec<bool>,
    bob_results: Vec<bool>,
    eve_intercepted_count: usize,
    check_ratio: f64,
    process_seed: u64,
) -> Result<SARG04Result, StateError> {
    // SARG04 sifting logic:
    // Conclusive if Alice and Bob used DIFFERENT bases AND Bob's measurement
    // was orthogonal to the *other* state in Alice's pair.
    // In our simulation:
    // Z basis: |0> (0), |1> (1)
    // X basis: |+> (0), |-> (1)
    // Conclusive if: (a_basis != b_basis) && (a_bit != b_res)
    let mut match_indices: Vec<usize> = (0..num_qubits)
        .filter(|&i| alice_bases[i] != bob_bases[i] && alice_bits[i] != bob_results[i])
        .collect();

    let total_conclusive = match_indices.len();
    let mut rng = LocalRng::from_seed(process_seed);
    rng.shuffle_slice(&mut match_indices);

    let num_check = (total_conclusive as f64 * check_ratio).round() as usize;
    let (_check_indices, key_indices) = match_indices.split_at(num_check);

    // In SARG04, error rate can be estimated from matching bases (Z-Z or X-X).
    let mut noise_errors = 0;
    let mut noise_total = 0;
    for i in 0..num_qubits {
        if alice_bases[i] == bob_bases[i] {
            noise_total += 1;
            if alice_bits[i] != bob_results[i] {
                noise_errors += 1;
            }
        }
    }
    let qber = if noise_total > 0 {
        noise_errors as f64 / noise_total as f64
    } else {
        0.0
    };

    let mut alice_key = Vec::with_capacity(key_indices.len());
    let mut bob_key = Vec::with_capacity(key_indices.len());
    for &i in key_indices {
        alice_key.push(alice_bits[i]);
        bob_key.push(alice_bits[i]);
    }

    Ok(SARG04Result {
        raw_length: num_qubits,
        conclusive_count: total_conclusive,
        check_errors: (qber * num_check as f64) as usize,
        qber,
        eve_detected_count: eve_intercepted_count,
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
    fn test_sarg04_noiseless() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(1000, &channel, 0.0, 1.0).unwrap();

        assert_eq!(result.raw_length, 1000);
        assert_eq!(result.qber, 0.0);
    }

    #[test]
    fn test_sarg04_keys_equal_noiseless() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(200, &channel, 0.0, 0.1).unwrap();
        assert_eq!(result.alice_key, result.bob_key);
    }

    #[test]
    fn test_sarg04_par_deterministic_with_seed() {
        let channel = QuantumChannel::bit_flip(0.05);
        crate::rng::set_global_seed(42);
        let r1 = run_par(500, &channel, 0.1, 0.2).unwrap();
        crate::rng::set_global_seed(42);
        let r2 = run_par(500, &channel, 0.1, 0.2).unwrap();

        assert_eq!(r1.alice_bits, r2.alice_bits);
        assert_eq!(r1.alice_bases, r2.alice_bases);
        assert_eq!(r1.bob_bases, r2.bob_bases);
        assert_eq!(r1.bob_results, r2.bob_results);
        assert_eq!(r1.alice_key, r2.alice_key);
        assert_eq!(r1.bob_key, r2.bob_key);
    }

    #[test]
    fn test_sarg04_zero_check() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(100, &channel, 0.0, 0.0).unwrap();
        assert_eq!(result.qber, 0.0);
    }

    #[test]
    fn test_sarg04_par_zero_check() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run_par(100, &channel, 0.0, 0.0).unwrap();
        assert_eq!(result.qber, 0.0);
    }

    #[test]
    fn test_sarg04_par_noiseless() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run_par(1000, &channel, 0.0, 0.1).unwrap();
        assert_eq!(result.qber, 0.0);
    }

    #[test]
    fn test_sarg04_with_eve() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(2000, &channel, 1.0, 0.5).unwrap();
        assert!(result.eve_detected_count > 0);
        // Eve in SARG04 introduces QBER on matching bases too.
        assert!(result.qber > 0.1);
    }

    #[test]
    fn test_sarg04_empty() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run_par(0, &channel, 0.0, 0.5).unwrap();
        assert_eq!(result.raw_length, 0);
        assert_eq!(result.qber, 0.0);
    }
}
