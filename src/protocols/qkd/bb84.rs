//! BB84 Quantum Key Distribution Protocol.
//!
//! BB84 is the first quantum cryptography protocol, developed by Charles Bennett and Gilles Brassard in 1984.
//! It uses four quantum states from two mutually unbiased bases (e.g., rectilinear and diagonal)
//! to securely establish a shared secret key.

use crate::{Gate, Measurement, QuantumChannel, QuantumState, errors::StateError};
use rand::Rng;
use rand::seq::SliceRandom;

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
    pub eve_detected_count: usize,
    /// The final established key (sifted key minus check bits).
    pub established_key: Vec<bool>,
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
/// A `BB84Result` with the simulation statistics and keys.
pub fn run(
    num_qubits: usize,
    channel: &QuantumChannel,
    eve_ratio: f64,
    check_ratio: f64,
) -> Result<BB84Result, StateError> {
    let mut rng = rand::rng();

    let mut alice_bits = Vec::with_capacity(num_qubits);
    let mut alice_bases = Vec::with_capacity(num_qubits);
    let mut bob_bases = Vec::with_capacity(num_qubits);
    let mut bob_results = Vec::with_capacity(num_qubits);

    let mut eve_intercepted_count = 0;

    for _ in 0..num_qubits {
        // Alice prepares qubits
        let a_bit = rng.random_bool(0.5);
        let a_basis = rng.random_bool(0.5);

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
        if eve_ratio > 1e-12 && rng.random_bool(eve_ratio) {
            eve_intercepted_count += 1;

            let e_basis = rng.random_bool(0.5);
            let measurement = if e_basis {
                Measurement::x_basis()
            } else {
                Measurement::z_basis()
            };

            let _ = state.measure(&measurement, &[0])?;
        }

        // Bob measures
        let b_basis = rng.random_bool(0.5);
        let measurement = if b_basis {
            Measurement::x_basis()
        } else {
            Measurement::z_basis()
        };

        let res = state.measure(&measurement, &[0])?;

        let b_val = res.index == 1;

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

    Ok(BB84Result {
        raw_length: num_qubits,
        total_sifted,
        check_errors,
        qber,
        eve_detected_count: eve_intercepted_count,
        established_key,
        alice_bits,
        alice_bases,
        bob_bases,
        bob_results,
    })
}
