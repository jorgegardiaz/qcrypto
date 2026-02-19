//! BBM92 Quantum Key Distribution Protocol.
//!
//! BBM92 is an entanglement-based QKD protocol proposed by Bennett, Brassard, and Mermin in 1992.
//! It is logically equivalent to BB84 but uses entangled photon pairs (EPR source) instead of single
//! photon pulses prepared by Alice.

use crate::{Gate, Measurement, QuantumChannel, QuantumState, errors::StateError};
use rand::Rng;
use rand::seq::SliceRandom;

/// The result of the BBM92 protocol execution.
pub struct Bbm92Result {
    /// The total length of the raw key (number of entangled pairs).
    pub raw_length: usize,
    /// The number of bits where bases matched (before sacrificing).
    pub total_sifted: usize,
    /// The number of errors found in the check bits.
    pub check_errors: usize,
    /// The Quantum Bit Error Rate (QBER) in percentage (on check bits).
    pub qber: f64,
    /// The number of times Eve was detected (simulated).
    pub eve_detected_count: usize,
    /// Alice's chosen bases (0: Z, 1: X).
    pub alice_bases: Vec<bool>,
    /// Bob's chosen bases (0: Z, 1: X).
    pub bob_bases: Vec<bool>,
    /// Alice's measurement results.
    pub alice_bits: Vec<bool>,
    /// Bob's measurement results.
    pub bob_results: Vec<bool>,
    /// The final established key (sifted key minus check bits).
    pub established_key: Vec<bool>,
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
/// A `Bbm92Result` with the simulation statistics and keys.
pub fn run(
    num_pairs: usize,
    channel: &QuantumChannel,
    eve_ratio: f64,
    check_ratio: f64,
) -> Result<Bbm92Result, StateError> {
    let mut rng = rand::rng();

    let mut alice_bits = Vec::with_capacity(num_pairs);
    let mut alice_bases = Vec::with_capacity(num_pairs);
    let mut bob_bases = Vec::with_capacity(num_pairs);
    let mut bob_results = Vec::with_capacity(num_pairs);
    let mut eve_detected_count = 0;

    for _ in 0..num_pairs {
        // Create EPR pair
        let mut state = QuantumState::new(2);
        state.apply(&Gate::h(), &[0])?;
        state.apply(&Gate::cnot(), &[0, 1])?;

        // Alice sends one of the EPR's state qubit to Bob
        state.apply_channel(channel, &[1])?;

        // Eavesdropper intercepts
        if eve_ratio > 1e-9 && rng.random_bool(eve_ratio) {
            eve_detected_count += 1;
            let e_basis = rng.random_bool(0.5);
            let measurement = if e_basis {
                Measurement::x_basis()
            } else {
                Measurement::z_basis()
            };

            let _ = state.measure(&measurement, &[1])?;
        }

        // Alice measures
        let a_basis = rng.random_bool(0.5);
        let a_measurement = if a_basis {
            Measurement::x_basis()
        } else {
            Measurement::z_basis()
        };

        let res_a = state.measure(&a_measurement, &[0])?;

        let a_bit = res_a.index == 1;

        //Bob measures his qubit
        let b_basis = rng.random_bool(0.5);
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
    match_indices.shuffle(&mut rng);

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
        (check_errors as f64 / num_check as f64) * 100.0
    } else {
        0.0
    };

    // 5. Build established key
    let mut established_key = Vec::with_capacity(key_indices.len());
    for &i in key_indices {
        established_key.push(alice_bits[i]);
    }

    Ok(Bbm92Result {
        raw_length: num_pairs,
        total_sifted,
        check_errors,
        qber,
        eve_detected_count,
        alice_bases,
        bob_bases,
        alice_bits,
        bob_results,
        established_key,
    })
}
