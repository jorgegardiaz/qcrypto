//! Quantum Identity Authentication - Quantum Zero Knowledge Proof (QIA-QZKP) protocol.
//!
//! This module implements a QIA-QZKP scheme based on conjugate coding, allowing
//! a prover to demonstrate his identity without revealing information about his secret.

use crate::{Gate, Measurement, QuantumChannel, QuantumState, errors::StateError};
use rand::Rng;

/// The result of the QIA-QZKP protocol execution.
pub struct QiaQZKPResult {
    /// Total number of qubits used.
    pub total_qubits: usize,
    /// Number of matching outcomes between Alice and Bob.
    pub matches: usize,
    /// Accuracy of the authentication (matches / total_qubits).
    pub accuracy: f64,
    /// Whether the authentication was successful based on the threshold.
    pub authenticated: bool,
    /// Alice's secret identity/key 'a'.
    pub alice_id_a: Vec<bool>,
    /// Alice's commitment 'b'.
    pub alice_commitment_b: Vec<bool>,
    /// Bob's challenge 'c'.
    pub bob_challenge_c: Vec<bool>,
    /// Bob's recovered challenge 'c_prime'.
    pub bob_recovered_c: Vec<bool>,
}

/// Executes a Quantum Zero Knowledge Proof for Identity Authentication (QIA-QZKP).
///
/// This protocol is based on Conjugate Coding and ping-pong like interactions.
/// It allows Alice to prove her identity to Bob without revealing her secret key.
///
/// # Arguments
///
/// * `num_qubits` - The number of rounds/qubits to run the protocol.
/// * `channel` - The quantum channel to use for communication (simulating noise).
/// * `acceptance_threshold` - The minimum accuracy required for successful authentication.
///
/// # Returns
///
/// A `QiaQZKPResult` containing the details of the protocol execution.
///
/// # Errors
///
/// Returns `StateError` if any quantum operation fails.
pub fn run(
    num_qubits: usize,
    channel: &QuantumChannel,
    acceptance_threshold: f64,
) -> Result<QiaQZKPResult, StateError> {
    let mut rng = rand::rng();

    let a: Vec<bool> = (0..num_qubits).map(|_| rng.random_bool(0.5)).collect();

    let mut b_vec = Vec::with_capacity(num_qubits);
    let mut c_vec = Vec::with_capacity(num_qubits);
    let mut c_recovered_vec = Vec::with_capacity(num_qubits);
    let mut matches = 0;

    for &a_bit in &a {
        // Alice's commitment
        let b_bit = rng.random_bool(0.5);
        b_vec.push(b_bit);

        // Sends (a XOR b) to Bob. Bob obtains 'b' using 'a'.

        //Bob generates the secret state |psi>
        let mut state = QuantumState::new(1);

        if a_bit {
            state.apply(&Gate::x(), &[0])?;
        }

        if b_bit {
            state.apply(&Gate::h(), &[0])?;
        }

        // Bob generats random challenge 'c'
        let c_bit = rng.random_bool(0.5);
        c_vec.push(c_bit);

        // Bob modifies |psi> to create the challenge state |psi'>
        if c_bit {
            if !b_bit {
                state.apply(&Gate::x(), &[0])?;
            } else {
                state.apply(&Gate::z(), &[0])?;
            }
        }

        // Bob's challenge
        // Bob sends |psi'> to Alice
        state.apply_channel(channel, &[0])?;

        // --- FASE 3: RESPUESTA (ALICE) ---
        // Alice applies gates in order: Z_b -> H_(a XOR b) -> Z_a

        if b_bit {
            state.apply(&Gate::z(), &[0])?;
        }

        if a_bit ^ b_bit {
            state.apply(&Gate::h(), &[0])?;
        }

        if a_bit {
            state.apply(&Gate::z(), &[0])?;
        }

        // Alice sends the proof state to Bob
        state.apply_channel(channel, &[0])?;

        // Bob's verification
        // Bob measures using a for basis
        let measurement = if a_bit {
            Measurement::x_basis()
        } else {
            Measurement::z_basis()
        };

        let res = state.measure(&measurement, &[0])?;

        // Bob recovers c'
        let measured_bit = res.index == 1;

        let c_prime = measured_bit ^ b_bit; //  "b XOR c' XOR b = c'"
        c_recovered_vec.push(c_prime);

        if c_prime == c_bit {
            matches += 1;
        }
    }

    // Accpetance criterion
    let accuracy = matches as f64 / num_qubits as f64;
    let authenticated = accuracy >= acceptance_threshold;

    Ok(QiaQZKPResult {
        total_qubits: num_qubits,
        matches,
        accuracy,
        authenticated,
        alice_id_a: a,
        alice_commitment_b: b_vec,
        bob_challenge_c: c_vec,
        bob_recovered_c: c_recovered_vec,
    })
}
