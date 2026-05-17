//! GC01 Quantum Digital Signature Protocol.
//!
//! The Gottesman-Chuang (2001) protocol is a quantum digital signature scheme
//! that allows a signer (Alice) to sign a classical message bit so that multiple
//! recipients (Bob and Charlie) can each independently verify the signature, with
//! the guarantee that a valid signature cannot be forged or repudiated.
//!
//! ## Quantum One-Way Function
//!
//! Each private-key position is a pair `(basis, value)` mapped to one of the six
//! states spanning three mutually unbiased bases (MUBs):
//!
//! | basis | value=false | value=true |
//! |-------|-------------|------------|
//! | 0 (Z) | \|0⟩        | \|1⟩       |
//! | 1 (X) | \|+⟩        | \|-⟩       |
//! | 2 (Y) | \|+i⟩       | \|-i⟩      |
//!
//! Using 3 MUBs reduces the probability that an adversary prepares a valid forgery
//! to at most (2/3) per position (vs 3/4 with two MUBs), making the scheme harder
//! to forge for the same key length.
//!
//! ## Protocol flow
//!
//! 1. **Distribution**: Alice prepares quantum public-key states and sends independent
//!    copies to Bob and Charlie through a quantum channel.
//! 2. **Signing**: Alice broadcasts `(m, K_m)` — the message bit and its classical key.
//! 3. **Verification**: Each verifier assembles a 3-qubit SWAP-test circuit from their
//!    received qubit, a fresh ancilla, and a fresh reference state prepared according
//!    to `K_m`:
//!
//! P(fail) = (1 − |⟨received|reference⟩|²) / 2.
//!
//! **Eve model**: intercept-resend with a uniformly random basis from the 3 MUBs.

use crate::rng::{self, LocalRng};
use crate::{Gate, Measurement, QuantumChannel, QuantumState, errors::StateError};
use rayon::prelude::*;

/// The result of a GC01 protocol execution.
pub struct GC01Result {
    /// The total length of each public key (number of qubits per message value).
    pub num_qubits: usize,
    /// The message bit Alice signed.
    pub message: bool,
    /// Alice's revealed private key for the signed message: one `(basis, value)` per qubit.
    pub private_key: Vec<(usize, bool)>,
    /// Number of SWAP-test failures observed by Bob.
    pub bob_mismatches: usize,
    /// Number of SWAP-test failures observed by Charlie.
    pub charlie_mismatches: usize,
    /// Bob's mismatch rate (mismatches / num_qubits).
    pub bob_mismatch_rate: f64,
    /// Charlie's mismatch rate (mismatches / num_qubits).
    pub charlie_mismatch_rate: f64,
    /// Whether both Bob and Charlie accepted the signature (both rates <= threshold).
    pub signature_accepted: bool,
    /// Total number of qubits Eve intercepted across all transmissions.
    pub eve_intercepted_count: usize,
}

/// Prepares a single-qubit state from a `(basis, value)` private-key pair.
///
/// Encoding matches the Six-State protocol:
/// - basis 0 (Z): `value=false` -> |0>, `value=true` -> |1>
/// - basis 1 (X): `value=false` -> |+>, `value=true` -> |->
/// - basis 2 (Y): `value=false` -> |+i>, `value=true` -> |-i>
fn prepare_qowf_state(basis: usize, value: bool) -> Result<QuantumState, StateError> {
    let mut state = QuantumState::new(1);
    if value {
        state.apply(&Gate::x(), &[0])?;
    }
    match basis {
        1 => {
            state.apply(&Gate::h(), &[0])?;
        }
        2 => {
            state.apply(&Gate::h(), &[0])?.apply(&Gate::s(), &[0])?;
        }
        _ => {}
    }
    Ok(state)
}

/// Assembles the 3-qubit SWAP-test system and returns whether the test failed.
///
/// Layout: ancilla |+> (q0) x received (q1) x reference (q2).
/// Returns `true` if the ancilla measured 1 (states differ).
fn swap_test_fail(received: QuantumState, basis: usize, value: bool) -> Result<bool, StateError> {
    let mut ancilla = QuantumState::new(1);
    ancilla.apply(&Gate::h(), &[0])?;

    let reference = prepare_qowf_state(basis, value)?;

    let mut state = ancilla.compose(&received)?.compose(&reference)?;
    state.apply_controlled(&Gate::swap(), &[1, 2], &[0])?;
    state.apply(&Gate::h(), &[0])?;
    let result = state.measure(&Measurement::z_basis(), &[0])?;
    Ok(result.value as usize == 1)
}

/// Variant of [`swap_test_fail`] using a [`LocalRng`] for deterministic parallel execution.
fn swap_test_fail_with_rng(
    received: QuantumState,
    basis: usize,
    value: bool,
    rng: &mut LocalRng,
) -> Result<bool, StateError> {
    let mut ancilla = QuantumState::new(1);
    ancilla.apply(&Gate::h(), &[0])?;

    let reference = prepare_qowf_state(basis, value)?;

    let mut state = ancilla.compose(&received)?.compose(&reference)?;
    state.apply_controlled(&Gate::swap(), &[1, 2], &[0])?;
    state.apply(&Gate::h(), &[0])?;
    let result = state.measure_with_rng(&Measurement::z_basis(), &[0], rng)?;
    Ok(result.value as usize == 1)
}

/// Executes the GC01 Quantum Digital Signature protocol.
///
/// Alice distributes quantum public-key states to Bob and Charlie independently.
/// Each verifier stores their received qubit and, after Alice reveals `K_m`, assembles
/// a 3-qubit SWAP-test circuit to verify the signature. The signature is accepted
/// when both verifiers observe a SWAP-test failure rate <= `threshold`.
///
/// # Arguments
///
/// * `num_qubits` - Length of each public key (qubits per message value).
/// * `message` - The classical bit Alice signs (`false` = 0, `true` = 1).
/// * `channel` - The quantum channel applied during state distribution.
/// * `eve_ratio` - Probability that Eve intercepts each qubit (intercept-resend, random MUB basis).
/// * `threshold` - Maximum SWAP-test failure rate a verifier accepts.
///
/// # Returns
///
/// A `Result` containing `GC01Result` with verification outcomes and statistics.
///
/// # Errors
///
/// Returns a `StateError` if any quantum operation fails.
///
/// # Example
/// ```rust
/// use qcrypto::protocols::gc01;
/// use qcrypto::QuantumChannel;
///
/// let channel = QuantumChannel::bit_flip(0.0);
/// let result = gc01::run(100, &channel, &channel, 0.0, 0.1).unwrap();
///
/// assert!(result.signature_accepted);
/// assert_eq!(result.bob_mismatches, 0);
/// assert_eq!(result.charlie_mismatches, 0);
/// ```
pub fn run(
    num_qubits: usize,
    channel_bob: &QuantumChannel,
    channel_charlie: &QuantumChannel,
    eve_ratio: f64,
    threshold: f64,
) -> Result<GC01Result, StateError> {
    let private_key_0: Vec<(usize, bool)> = (0..num_qubits)
        .map(|_| {
            (
                crate::rng::random_usize_range(0, 3),
                crate::rng::random_bool(0.5),
            )
        })
        .collect();
    let private_key_1: Vec<(usize, bool)> = (0..num_qubits)
        .map(|_| {
            (
                crate::rng::random_usize_range(0, 3),
                crate::rng::random_bool(0.5),
            )
        })
        .collect();

    let message = rng::random_bool(0.5);

    let private_key = if message {
        &private_key_1
    } else {
        &private_key_0
    };

    let mut bob_mismatches = 0usize;
    let mut charlie_mismatches = 0usize;
    let mut eve_intercepted_count = 0usize;

    for &(basis, value) in private_key.iter() {
        // --- Bob's copy ---
        let mut bob_qubit = prepare_qowf_state(basis, value)?;
        bob_qubit.apply_channel(channel_bob, &[0])?;
        if eve_ratio > 0.0 && crate::rng::random_bool(eve_ratio) {
            eve_intercepted_count += 1;
            let eve_measurement = match crate::rng::random_usize_range(0, 3) {
                1 => Measurement::x_basis(),
                2 => Measurement::y_basis(),
                _ => Measurement::z_basis(),
            };
            let _ = bob_qubit.measure(&eve_measurement, &[0])?;
        }
        bob_qubit.apply_channel(channel_bob, &[0])?;
        if swap_test_fail(bob_qubit, basis, value)? {
            bob_mismatches += 1;
        }

        // --- Charlie's copy (independent transmission) ---
        let mut charlie_qubit = prepare_qowf_state(basis, value)?;
        charlie_qubit.apply_channel(channel_charlie, &[0])?;
        if eve_ratio > 0.0 && crate::rng::random_bool(eve_ratio) {
            eve_intercepted_count += 1;
            let eve_measurement = match crate::rng::random_usize_range(0, 3) {
                1 => Measurement::x_basis(),
                2 => Measurement::y_basis(),
                _ => Measurement::z_basis(),
            };
            let _ = charlie_qubit.measure(&eve_measurement, &[0])?;
        }
        charlie_qubit.apply_channel(channel_charlie, &[0])?;
        if swap_test_fail(charlie_qubit, basis, value)? {
            charlie_mismatches += 1;
        }
    }

    let bob_mismatch_rate = if num_qubits > 0 {
        bob_mismatches as f64 / num_qubits as f64
    } else {
        0.0
    };
    let charlie_mismatch_rate = if num_qubits > 0 {
        charlie_mismatches as f64 / num_qubits as f64
    } else {
        0.0
    };

    let signature_accepted = bob_mismatch_rate <= threshold && charlie_mismatch_rate <= threshold;

    Ok(GC01Result {
        num_qubits,
        message,
        private_key: private_key.clone(),
        bob_mismatches,
        charlie_mismatches,
        bob_mismatch_rate,
        charlie_mismatch_rate,
        signature_accepted,
        eve_intercepted_count,
    })
}

/// Parallelized variant of [`run`] using rayon. See [`run`] for protocol semantics.
///
/// # Arguments
///
/// * `num_qubits` - Length of each public key (qubits per message value).
/// * `message` - The classical bit Alice signs (`false` = 0, `true` = 1).
/// * `channel` - The quantum channel applied during state distribution.
/// * `eve_ratio` - Probability that Eve intercepts each qubit (intercept-resend, random MUB basis).
/// * `threshold` - Maximum SWAP-test failure rate a verifier accepts.
///
/// # Returns
///
/// A `Result` containing `GC01Result` with verification outcomes and statistics.
///
/// # Errors
///
/// Returns a `StateError` if any quantum operation fails.
///
/// # Example
/// ```rust
/// use qcrypto::protocols::gc01;
/// use qcrypto::QuantumChannel;
///
/// let channel = &QuantumChannel::bit_flip(0.05);
/// qcrypto::rng::set_global_seed(42);
/// let r1 = gc01::run_par(300, channel, channel, 0.0, 0.1).unwrap();
///
/// qcrypto::rng::set_global_seed(42);
/// let r2 = gc01::run_par(300, channel, channel, 0.0, 0.1).unwrap();
///
/// assert_eq!(r1.bob_mismatch_rate, r2.bob_mismatch_rate);
/// assert_eq!(r1.private_key, r2.private_key);
/// ```
pub fn run_par(
    num_qubits: usize,
    channel_bob: &QuantumChannel,
    channel_charlie: &QuantumChannel,
    eve_ratio: f64,
    threshold: f64,
) -> Result<GC01Result, StateError> {
    let master_key = crate::rng::draw_master_seed();
    let master_dist = crate::rng::draw_master_seed();
    let master_msg = crate::rng::draw_master_seed();

    let private_key_0: Vec<(usize, bool)> = (0..num_qubits)
        .map(|i| {
            let mut rng = LocalRng::child(master_key, i as u64);
            (rng.random_usize_range(0, 3), rng.random_bool(0.5))
        })
        .collect();
    let private_key_1: Vec<(usize, bool)> = (0..num_qubits)
        .map(|i| {
            let mut rng = LocalRng::child(master_key, num_qubits as u64 + i as u64);
            (rng.random_usize_range(0, 3), rng.random_bool(0.5))
        })
        .collect();

    let message = LocalRng::child(master_msg, 0).random_bool(0.5);

    let private_key = if message {
        &private_key_1
    } else {
        &private_key_0
    };

    type Step = (bool, bool, bool, bool);

    let steps: Vec<Step> = private_key
        .par_iter()
        .enumerate()
        .map(|(i, &(basis, value))| -> Result<Step, StateError> {
            let mut rng = LocalRng::child(master_dist, i as u64);

            let mut bob_qubit = prepare_qowf_state(basis, value)?;
            bob_qubit.apply_channel(channel_bob, &[0])?;
            let bob_eve = eve_ratio > 0.0 && rng.random_bool(eve_ratio);
            if bob_eve {
                let eve_measurement = match rng.random_usize_range(0, 3) {
                    1 => Measurement::x_basis(),
                    2 => Measurement::y_basis(),
                    _ => Measurement::z_basis(),
                };
                let _ = bob_qubit.measure_with_rng(&eve_measurement, &[0], &mut rng)?;
            }
            bob_qubit.apply_channel(channel_bob, &[0])?;
            let bob_mismatch = swap_test_fail_with_rng(bob_qubit, basis, value, &mut rng)?;

            let mut charlie_qubit = prepare_qowf_state(basis, value)?;
            charlie_qubit.apply_channel(channel_charlie, &[0])?;
            let charlie_eve = eve_ratio > 0.0 && rng.random_bool(eve_ratio);
            if charlie_eve {
                let eve_measurement = match rng.random_usize_range(0, 3) {
                    1 => Measurement::x_basis(),
                    2 => Measurement::y_basis(),
                    _ => Measurement::z_basis(),
                };
                let _ = charlie_qubit.measure_with_rng(&eve_measurement, &[0], &mut rng)?;
            }
            charlie_qubit.apply_channel(channel_charlie, &[0])?;
            let charlie_mismatch = swap_test_fail_with_rng(charlie_qubit, basis, value, &mut rng)?;

            Ok((bob_mismatch, charlie_mismatch, bob_eve, charlie_eve))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut bob_mismatches = 0usize;
    let mut charlie_mismatches = 0usize;
    let mut eve_intercepted_count = 0usize;

    for (bob_mismatch, charlie_mismatch, bob_eve, charlie_eve) in steps {
        if bob_mismatch {
            bob_mismatches += 1;
        }
        if charlie_mismatch {
            charlie_mismatches += 1;
        }
        if bob_eve {
            eve_intercepted_count += 1;
        }
        if charlie_eve {
            eve_intercepted_count += 1;
        }
    }

    let bob_mismatch_rate = if num_qubits > 0 {
        bob_mismatches as f64 / num_qubits as f64
    } else {
        0.0
    };
    let charlie_mismatch_rate = if num_qubits > 0 {
        charlie_mismatches as f64 / num_qubits as f64
    } else {
        0.0
    };

    let signature_accepted = bob_mismatch_rate <= threshold && charlie_mismatch_rate <= threshold;

    Ok(GC01Result {
        num_qubits,
        message,
        private_key: private_key.clone(),
        bob_mismatches,
        charlie_mismatches,
        bob_mismatch_rate,
        charlie_mismatch_rate,
        signature_accepted,
        eve_intercepted_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gc01_noiseless_accepted() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(100, &channel, &channel, 0.0, 0.1).unwrap();

        assert_eq!(result.num_qubits, 100);
        assert_eq!(result.bob_mismatches, 0);
        assert_eq!(result.charlie_mismatches, 0);
        assert_eq!(result.bob_mismatch_rate, 0.0);
        assert_eq!(result.charlie_mismatch_rate, 0.0);
        assert!(result.signature_accepted);
        assert_eq!(result.eve_intercepted_count, 0);
    }

    #[test]
    fn test_gc01_message_noiseless_accepted() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(100, &channel, &channel, 0.0, 0.1).unwrap();

        assert_eq!(result.bob_mismatches, 0);
        assert_eq!(result.charlie_mismatches, 0);
        assert!(result.signature_accepted);
    }

    #[test]
    fn test_gc01_private_key_contains_all_bases() {
        crate::rng::set_global_seed(1);
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(300, &channel, &channel, 0.0, 0.1).unwrap();

        let bases: std::collections::HashSet<usize> =
            result.private_key.iter().map(|&(b, _)| b).collect();
        assert!(bases.contains(&0), "Z basis missing from key");
        assert!(bases.contains(&1), "X basis missing from key");
        assert!(bases.contains(&2), "Y basis missing from key");
    }

    #[test]
    fn test_gc01_noisy_increases_mismatch() {
        let channel = QuantumChannel::bit_flip(0.3);
        let result = run(500, &channel, &channel, 0.0, 0.0).unwrap();

        assert!(result.bob_mismatches > 0 || result.charlie_mismatches > 0);
        assert!(!result.signature_accepted);
    }

    #[test]
    fn test_gc01_eve_detected() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(200, &channel, &channel, 1.0, 1.0).unwrap();

        assert!(result.eve_intercepted_count > 0);
    }

    #[test]
    fn test_gc01_eve_raises_mismatch() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(2000, &channel, &channel, 1.0, 0.0).unwrap();

        assert!(!result.signature_accepted);
        assert!(
            result.bob_mismatch_rate > 0.04,
            "Eve should cause detectable SWAP-test failures: bob={}",
            result.bob_mismatch_rate
        );
    }

    #[test]
    fn test_gc01_private_key_length() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(50, &channel, &channel, 0.0, 0.1).unwrap();

        assert_eq!(result.private_key.len(), 50);
    }

    #[test]
    fn test_gc01_zero_qubits() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(0, &channel, &channel, 0.0, 0.0).unwrap();

        assert_eq!(result.num_qubits, 0);
        assert_eq!(result.bob_mismatch_rate, 0.0);
        assert_eq!(result.charlie_mismatch_rate, 0.0);
        assert!(result.signature_accepted);
    }

    #[test]
    fn test_gc01_par_noiseless_accepted() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run_par(100, &channel, &channel, 0.0, 0.1).unwrap();

        assert_eq!(result.num_qubits, 100);
        assert_eq!(result.bob_mismatches, 0);
        assert_eq!(result.charlie_mismatches, 0);
        assert!(result.signature_accepted);
        assert_eq!(result.eve_intercepted_count, 0);
    }

    #[test]
    fn test_gc01_par_message_noiseless_accepted() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run_par(100, &channel, &channel, 0.0, 0.1).unwrap();

        assert_eq!(result.bob_mismatches, 0);
        assert!(result.signature_accepted);
    }

    #[test]
    fn test_gc01_par_eve_detected() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run_par(200, &channel, &channel, 1.0, 1.0).unwrap();

        assert!(result.eve_intercepted_count > 0);
    }

    #[test]
    fn test_gc01_par_zero_qubits() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run_par(0, &channel, &channel, 0.0, 0.0).unwrap();

        assert_eq!(result.num_qubits, 0);
        assert_eq!(result.bob_mismatch_rate, 0.0);
        assert_eq!(result.charlie_mismatch_rate, 0.0);
        assert!(result.signature_accepted);
    }

    #[test]
    fn test_gc01_par_deterministic_with_seed() {
        let channel = QuantumChannel::bit_flip(0.05);

        crate::rng::set_global_seed(42);
        let r1 = run_par(300, &channel, &channel, 0.1, 0.2).unwrap();

        crate::rng::set_global_seed(42);
        let r2 = run_par(300, &channel, &channel, 0.1, 0.2).unwrap();

        assert_eq!(r1.private_key, r2.private_key);
        assert_eq!(r1.bob_mismatches, r2.bob_mismatches);
        assert_eq!(r1.charlie_mismatches, r2.charlie_mismatches);
        assert_eq!(r1.bob_mismatch_rate, r2.bob_mismatch_rate);
        assert_eq!(r1.charlie_mismatch_rate, r2.charlie_mismatch_rate);
        assert_eq!(r1.eve_intercepted_count, r2.eve_intercepted_count);
        assert_eq!(r1.signature_accepted, r2.signature_accepted);
    }
}
