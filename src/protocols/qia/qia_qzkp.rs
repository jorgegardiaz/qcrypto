//! Quantum Identity Authentication - Quantum Zero Knowledge Proof (QIA-QZKP) protocol.
//!
//! This module implements a QIA-QZKP scheme based on conjugate coding, allowing
//! a prover to demonstrate his identity without revealing information about his secret.

use crate::rng::LocalRng;
use crate::{Gate, Measurement, QuantumChannel, QuantumState, errors::StateError};
use rayon::prelude::*;

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
/// A `Result` containing `QiaQZKPResult` with the details of the protocol execution.
///
/// # Errors
///
/// Returns `StateError` if any quantum operation fails.
///
/// # Example
/// ```rust
/// use qcrypto::protocols::qia_qzkp;
/// use qcrypto::QuantumChannel;
///
/// let channel = QuantumChannel::bit_flip(0.0);
/// let result = qia_qzkp::run(100, &channel, 0.9).unwrap();
///
/// assert!(result.authenticated);
/// ```
pub fn run(
    num_qubits: usize,
    channel: &QuantumChannel,
    acceptance_threshold: f64,
) -> Result<QiaQZKPResult, StateError> {
    let a: Vec<bool> = (0..num_qubits)
        .map(|_| crate::rng::random_bool(0.5))
        .collect();

    let mut b_vec = Vec::with_capacity(num_qubits);
    let mut c_vec = Vec::with_capacity(num_qubits);
    let mut c_recovered_vec = Vec::with_capacity(num_qubits);
    let mut matches = 0;

    for &a_bit in &a {
        // Alice's commitment
        let b_bit = crate::rng::random_bool(0.5);
        b_vec.push(b_bit);

        // Sends (a XOR b) to Bob. Bob obtains 'b' using 'a'.

        // Challenge Generation
        //Bob generates the secret state |psi>
        let mut state = QuantumState::new(1);

        if a_bit {
            state.apply(&Gate::x(), &[0])?;
        }

        if b_bit {
            state.apply(&Gate::h(), &[0])?;
        }

        // Bob generates random challenge 'c'
        let c_bit = crate::rng::random_bool(0.5);
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

        // Proof Generation
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
        // Bob measures using `a` bits as basis
        let measurement = if a_bit {
            Measurement::x_basis()
        } else {
            Measurement::z_basis()
        };

        let result = state.measure(&measurement, &[0])?;

        // Bob recovers c'
        let measured_bit = result.index == 1;

        let c_prime = measured_bit ^ b_bit; //  "b XOR c' XOR b = c'"
        c_recovered_vec.push(c_prime);

        if c_prime == c_bit {
            matches += 1;
        }
    }

    // Acceptance criterion
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

/// Parallel variant of [`run`] using rayon. See [`run`] for protocol semantics.
pub fn run_par(
    num_qubits: usize,
    channel: &QuantumChannel,
    acceptance_threshold: f64,
) -> Result<QiaQZKPResult, StateError> {
    let master = crate::rng::draw_master_seed();

    type Step = (bool, bool, bool, bool, bool);

    let steps: Vec<Step> = (0..num_qubits)
        .into_par_iter()
        .map(|i| -> Result<Step, StateError> {
            let mut rng = LocalRng::child(master, i as u64);

            let a_bit = rng.random_bool(0.5);
            let b_bit = rng.random_bool(0.5);

            let mut state = QuantumState::new(1);
            if a_bit {
                state.apply(&Gate::x(), &[0])?;
            }
            if b_bit {
                state.apply(&Gate::h(), &[0])?;
            }

            let c_bit = rng.random_bool(0.5);
            if c_bit {
                if !b_bit {
                    state.apply(&Gate::x(), &[0])?;
                } else {
                    state.apply(&Gate::z(), &[0])?;
                }
            }

            state.apply_channel(channel, &[0])?;

            if b_bit {
                state.apply(&Gate::z(), &[0])?;
            }
            if a_bit ^ b_bit {
                state.apply(&Gate::h(), &[0])?;
            }
            if a_bit {
                state.apply(&Gate::z(), &[0])?;
            }

            state.apply_channel(channel, &[0])?;

            let measurement = if a_bit {
                Measurement::x_basis()
            } else {
                Measurement::z_basis()
            };
            let result = state.measure_with_rng(&measurement, &[0], &mut rng)?;
            let measured_bit = result.index == 1;
            let c_prime = measured_bit ^ b_bit;
            let is_match = c_prime == c_bit;

            Ok((a_bit, b_bit, c_bit, c_prime, is_match))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut a_vec = Vec::with_capacity(num_qubits);
    let mut b_vec = Vec::with_capacity(num_qubits);
    let mut c_vec = Vec::with_capacity(num_qubits);
    let mut c_recovered_vec = Vec::with_capacity(num_qubits);
    let mut matches = 0usize;

    for (a_bit, b_bit, c_bit, c_prime, is_match) in steps {
        a_vec.push(a_bit);
        b_vec.push(b_bit);
        c_vec.push(c_bit);
        c_recovered_vec.push(c_prime);
        if is_match {
            matches += 1;
        }
    }

    let accuracy = matches as f64 / num_qubits as f64;
    let authenticated = accuracy >= acceptance_threshold;

    Ok(QiaQZKPResult {
        total_qubits: num_qubits,
        matches,
        accuracy,
        authenticated,
        alice_id_a: a_vec,
        alice_commitment_b: b_vec,
        bob_challenge_c: c_vec,
        bob_recovered_c: c_recovered_vec,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qia_qzkp_noiseless() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(100, &channel, 0.9).unwrap();

        assert_eq!(result.total_qubits, 100);
        assert_eq!(result.matches, 100);
        assert_eq!(result.accuracy, 1.0);
        assert!(result.authenticated);
    }

    #[test]
    fn test_qia_qzkp_noise_rejection() {
        let channel = QuantumChannel::bit_flip(1.0); // Extreme noise
        // This should cause enough errors to drop below the 0.9 threshold
        let result = run(100, &channel, 0.9).unwrap();

        assert!(!result.authenticated);
    }

    #[test]
    fn test_qia_qzkp_par_noiseless() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run_par(100, &channel, 0.9).unwrap();
        assert_eq!(result.total_qubits, 100);
        assert_eq!(result.matches, 100);
        assert_eq!(result.accuracy, 1.0);
        assert!(result.authenticated);
    }

    #[test]
    fn test_qia_qzkp_par_deterministic_with_seed() {
        let channel = QuantumChannel::bit_flip(0.05);

        crate::rng::set_global_seed(77);
        let r1 = run_par(200, &channel, 0.9).unwrap();
        crate::rng::set_global_seed(77);
        let r2 = run_par(200, &channel, 0.9).unwrap();

        assert_eq!(r1.alice_id_a, r2.alice_id_a);
        assert_eq!(r1.alice_commitment_b, r2.alice_commitment_b);
        assert_eq!(r1.bob_challenge_c, r2.bob_challenge_c);
        assert_eq!(r1.bob_recovered_c, r2.bob_recovered_c);
        assert_eq!(r1.matches, r2.matches);
    }

    #[test]
    fn test_qia_qzkp_par_noise_rejection() {
        let channel = QuantumChannel::bit_flip(1.0);
        let result = run_par(100, &channel, 0.9).unwrap();
        assert!(!result.authenticated);
    }
}
