//! SARG04 Quantum Key Distribution Protocol.
//!
//! SARG04 was proposed by Scarani, Acín, Ribordy and Gisin in 2004.
//! It reuses the four BB84 states {|0>, |1>, |+>, |->} but differs in the sifting phase:
//! instead of announcing her basis, Alice announces a pair of non-orthogonal states
//! (one from each basis) that contains the state she actually sent. Bob deduces the
//! bit only when his measurement outcome is orthogonal to exactly one of the two
//! announced states — the bit value is then determined by the *other* state in the pair.
//!
//! Noiseless sifting rate is 1/4, and SARG04 is known to be more robust than BB84
//! against photon-number-splitting attacks in weak-coherent-pulse implementations.

use crate::{Gate, Measurement, QuantumChannel, QuantumState, errors::StateError};

/// The result of the SARG04 protocol execution.
pub struct Sarg04Result {
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
    /// Bob's final key (conclusive bits minus check bits). May differ from Alice's due to channel noise.
    pub bob_key: Vec<bool>,
    /// Alice's original bits.
    pub alice_bits: Vec<bool>,
    /// Alice's chosen bases (false: Z, true: X).
    pub alice_bases: Vec<bool>,
    /// Bob's chosen bases (false: Z, true: X).
    pub bob_bases: Vec<bool>,
    /// Bob's raw measurement outcomes (before SARG04 decoding).
    pub bob_results: Vec<bool>,
}

/// Executes the SARG04 QKD protocol.
///
/// Alice prepares one of four BB84 states encoded by a random bit and a random basis.
/// Bob measures in a random basis (Z or X). During sifting, Alice announces a pair of
/// non-orthogonal states {(a_basis, a_bit), (!a_basis, partner_bit)} containing her
/// transmitted state. Bob's outcome is conclusive only when it is orthogonal to exactly
/// one of the two announced states; the decoded bit is then the bit of the *other* state.
///
/// # Arguments
///
/// * `num_qubits` - Number of qubits to transmit.
/// * `channel` - The quantum channel (noise model).
/// * `eve_ratio` - Probability of Eve intercepting (and measuring) a qubit.
/// * `check_ratio` - Fraction of conclusive bits to sacrifice for QBER estimation.
///
/// # Returns
///
/// A `Result` containing `Sarg04Result` with the simulation statistics and keys.
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
/// let channel = QuantumChannel::bit_flip(0.0);
/// let result = sarg04::run(200, &channel, 0.0, 0.5).unwrap();
///
/// assert_eq!(result.raw_length, 200);
/// ```
pub fn run(
    num_qubits: usize,
    channel: &QuantumChannel,
    eve_ratio: f64,
    check_ratio: f64,
) -> Result<Sarg04Result, StateError> {
    let mut alice_bits = Vec::with_capacity(num_qubits);
    let mut alice_bases = Vec::with_capacity(num_qubits);
    let mut bob_bases = Vec::with_capacity(num_qubits);
    let mut bob_results = Vec::with_capacity(num_qubits);
    let mut bob_inferred: Vec<Option<bool>> = Vec::with_capacity(num_qubits);
    let mut eve_intercepted_count = 0;

    for _ in 0..num_qubits {
        // Alice prepares one of {|0>, |1>, |+>, |->}
        let a_bit = crate::rng::random_bool(0.5);
        let a_basis = crate::rng::random_bool(0.5);

        let mut state = QuantumState::new(1);
        if a_bit {
            state.apply(&Gate::x(), &[0])?;
        }
        if a_basis {
            state.apply(&Gate::h(), &[0])?;
        }

        // Alice sends qubit to Bob through the channel
        state.apply_channel(channel, &[0])?;

        // Eavesdropper intercepts (intercept-and-resend)
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

        // Bob measures in a random basis
        let b_basis = crate::rng::random_bool(0.5);
        let measurement = if b_basis {
            Measurement::x_basis()
        } else {
            Measurement::z_basis()
        };
        let res = state.measure(&measurement, &[0])?;
        let b_bit = res.index == 1;

        // SARG04 sifting: Alice picks a random partner bit to build the announced pair
        // {(a_basis, a_bit), (!a_basis, partner_bit)}. Both states are non-orthogonal.
        let partner_bit = crate::rng::random_bool(0.5);

        // Two states (B1, x1) and (B2, x2) are orthogonal iff B1 == B2 && x1 != x2.
        // Bob checks orthogonality of his outcome against each announced state.
        let orth_to_ann1 = b_basis == a_basis && b_bit != a_bit;
        let orth_to_ann2 = b_basis != a_basis && b_bit != partner_bit;

        let inferred = match (orth_to_ann1, orth_to_ann2) {
            // Orthogonal only to the partner state -> Alice sent (a_basis, a_bit).
            (false, true) => Some(a_bit),
            // Orthogonal only to Alice's actual state -> Bob decodes the partner's bit.
            // Cannot happen noiselessly; under noise/Eve it typically yields the wrong bit.
            (true, false) => Some(partner_bit),
            // Inconclusive: outcome compatible with both announced states.
            (false, false) => None,
            // Impossible: the announced pair is non-orthogonal by construction.
            (true, true) => unreachable!("SARG04 announced pair is non-orthogonal"),
        };

        alice_bits.push(a_bit);
        alice_bases.push(a_basis);
        bob_bases.push(b_basis);
        bob_results.push(b_bit);
        bob_inferred.push(inferred);
    }

    // Sifting stage
    // 1. Identify indices where Bob got a conclusive decoding
    let mut conclusive_indices: Vec<usize> = bob_inferred
        .iter()
        .enumerate()
        .filter_map(|(i, b)| if b.is_some() { Some(i) } else { None })
        .collect();

    let conclusive_count = conclusive_indices.len();

    // 2. Shuffle indices to randomly select bits for error checking
    crate::rng::shuffle_slice(&mut conclusive_indices);

    // 3. Split into check and key indices
    let num_check =
        ((conclusive_count as f64 * check_ratio).round() as usize).min(conclusive_count);
    let (check_indices, key_indices) = conclusive_indices.split_at(num_check);

    // 4. Calculate QBER on check bits
    let mut check_errors = 0;
    for &i in check_indices {
        if alice_bits[i] != bob_inferred[i].expect("conclusive index must have a value") {
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
        bob_key.push(bob_inferred[i].expect("conclusive index must have a value"));
    }

    Ok(Sarg04Result {
        raw_length: num_qubits,
        conclusive_count,
        check_errors,
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
        let result = run(500, &channel, 0.0, 0.5).unwrap();

        assert_eq!(result.raw_length, 500);
        assert_eq!(result.check_errors, 0);
        assert_eq!(result.qber, 0.0);
        assert_eq!(result.eve_detected_count, 0);
        // Noiseless SARG04 sifting rate is 1/4 -> ~125 conclusive bits out of 500.
        assert!(
            result.conclusive_count > 50 && result.conclusive_count < 200,
            "conclusive_count {} should be near 125 (1/4 sifting rate)",
            result.conclusive_count
        );
    }

    #[test]
    fn test_sarg04_keys_equal_noiseless() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(1000, &channel, 0.0, 0.0).unwrap();

        assert_eq!(result.alice_key.len(), result.bob_key.len());
        assert_eq!(result.alice_key, result.bob_key);
    }

    #[test]
    fn test_sarg04_noisy_keys_differ() {
        let channel = QuantumChannel::bit_flip(0.3);
        let result = run(2000, &channel, 0.0, 0.0).unwrap();

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
    fn test_sarg04_eve() {
        let channel = QuantumChannel::bit_flip(0.0);
        // Eve intercepts everything. Theoretical SARG04 QBER under intercept-and-resend
        // is in the 10-15% range; we use a loose lower bound.
        let result = run(5000, &channel, 1.0, 0.5).unwrap();

        assert!(result.eve_detected_count > 0);
        assert!(
            result.qber > 0.05,
            "QBER {} should be significant under full Eve interception",
            result.qber
        );
    }

    #[test]
    fn test_sarg04_zero_check() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(200, &channel, 0.0, 0.0).unwrap();
        assert_eq!(result.qber, 0.0);
    }

    #[test]
    fn test_sarg04_zero_qubits() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(0, &channel, 0.0, 0.5).unwrap();
        assert_eq!(result.raw_length, 0);
        assert_eq!(result.conclusive_count, 0);
        assert_eq!(result.qber, 0.0);
        assert!(result.alice_key.is_empty());
        assert!(result.bob_key.is_empty());
    }

    #[test]
    fn test_sarg04_full_check() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(500, &channel, 0.0, 1.0).unwrap();
        assert_eq!(result.qber, 0.0);
        assert!(result.alice_key.is_empty());
        assert!(result.bob_key.is_empty());
    }

    #[test]
    fn test_sarg04_check_ratio_overflow() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(200, &channel, 0.0, 2.0).unwrap();
        assert!(result.alice_key.is_empty());
        assert!(result.bob_key.is_empty());
    }

    #[test]
    fn test_sarg04_noisy_qber_nonzero() {
        let channel = QuantumChannel::bit_flip(0.3);
        let result = run(2000, &channel, 0.0, 0.5).unwrap();
        assert!(
            result.qber > 0.0,
            "QBER {} should be non-zero under a noisy channel with check bits",
            result.qber
        );
    }
}
