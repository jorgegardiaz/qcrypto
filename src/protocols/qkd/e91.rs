//! E91 Quantum Key Distribution Protocol.
//!
//! E91 is an entanglement-based QKD protocol proposed by Ekert in 1991.
//! It uses entangled photon pairs and Bell's inequality to ensure security.

use crate::rng::LocalRng;
use crate::{
    Gate, Measurement, QuantumChannel, QuantumState, errors::StateError, utils::outer_product,
};
use ndarray::array;
use num_complex::Complex64;
use rayon::prelude::*;
use std::f64::consts::PI;

/// The result of the E91 protocol execution.
pub struct E91Result {
    /// The total length of the raw key (number of entangled pairs).
    pub raw_length: usize,
    /// The number of bits where bases matched (potential key bits).
    pub total_sifted: usize,
    /// The number of errors found in the check bits.
    pub check_errors: usize,
    /// The Quantum Bit Error Rate (QBER) on check bits.
    /// `None` if no check bits were sampled.
    pub qber: Option<f64>,
    /// The calculated CHSH S-value for Bell's inequality.
    pub chsh_value: f64,
    /// The number of times Eve intercepted a qubit (simulated).
    pub eve_intercept_count: usize,
    /// Alice's chosen bases (0, 1, 2).
    pub alice_bases: Vec<usize>,
    /// Bob's chosen bases (0, 1, 2).
    pub bob_bases: Vec<usize>,
    /// Alice's measurement results.
    pub alice_bits: Vec<bool>,
    /// Bob's measurement results (raw, not flipped).
    pub bob_results: Vec<bool>,
    /// Alice's final key (after sifting and removing check bits).
    pub alice_key: Vec<bool>,
    /// Bob's final key (already flipped to account for the singlet's anticorrelation,
    /// so in the noiseless case `bob_key == alice_key`).
    pub bob_key: Vec<bool>,
}

/// Helper function to create a measurement in a basis defined by an angle theta.
fn angle_measurement(theta: f64) -> Measurement {
    let cos = Complex64::new(theta.cos(), 0.0);
    let sin = Complex64::new(theta.sin(), 0.0);

    // Basis vectors: |v0> = cos|0> + sin|1>, |v1> = sin|0> - cos|1>
    let v0 = array![cos, sin];
    let v1 = array![sin, -cos];

    let p0 = outer_product(&v0, &v0);
    let p1 = outer_product(&v1, &v1);

    Measurement::new(vec![p0, p1], vec![0.0, 1.0]).expect("Invalid angle measurement")
}

/// Calculates the correlation E(a, b) = (N_same - N_diff) / N_total.
fn calculate_correlation(indices: &[usize], alice_bits: &[bool], bob_bits: &[bool]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let mut same = 0i64;
    let mut diff = 0i64;
    for &i in indices {
        if alice_bits[i] == bob_bits[i] {
            same += 1;
        } else {
            diff += 1;
        }
    }
    (same - diff) as f64 / (indices.len() as f64)
}

/// Executes the E91 QKD protocol.
///
/// E91 is an entanglement-based QKD protocol based on Bell's inequality.
/// Instead of Alice sending states, a source distributes entangled pairs (EPR pairs) to Alice and Bob.
/// They measure their respective qubits in random bases.
///
/// Alice's bases (physical angles): 0, pi/8, pi/4.
/// Bob's bases (physical angles): pi/8, pi/4, 3pi/8.
///
/// Key sifting uses pairs where Alice and Bob measured the same physical angle:
/// (a=1, b=0) and (a=2, b=1). Since the state is a singlet, matching angles give
/// anticorrelated outcomes, so Bob flips his bits to align with Alice's key.
///
/// CHSH is evaluated on the remaining "non-matching" pairs (0,0), (0,2), (2,0), (2,2):
///   S = E(a1,b1) - E(a1,b3) + E(a3,b1) + E(a3,b3)
/// For an ideal singlet this gives -2*sqrt(2).
///
/// # Arguments
///
/// * `num_pairs` - Number of entangled pairs to distribute.
/// * `channel_alice` - The quantum channel affecting Alice's qubit.
/// * `channel_bob` - The quantum channel affecting Bob's qubit.
/// * `eve_ratio` - Probability of Eve intercepting (and measuring) a qubit.
/// * `check_ratio` - Fraction of sifted bits to sacrifice for QBER estimation.
///
/// # Returns
///
/// A `Result` containing `E91Result` with the simulation statistics and keys.
///
/// # Errors
///
/// Returns a `StateError` if quantum operations fail.
pub fn run(
    num_pairs: usize,
    channel_alice: &QuantumChannel,
    channel_bob: &QuantumChannel,
    eve_ratio: f64,
    check_ratio: f64,
) -> Result<E91Result, StateError> {
    let mut alice_bits = Vec::with_capacity(num_pairs);
    let mut alice_bases = Vec::with_capacity(num_pairs);
    let mut bob_bases = Vec::with_capacity(num_pairs);
    let mut bob_results = Vec::with_capacity(num_pairs);
    let mut eve_intercept_count = 0;

    // Angles for E91 (physical angles; Bloch angle is 2x)
    let a_angles = [0.0, PI / 8.0, PI / 4.0];
    let b_angles = [PI / 8.0, PI / 4.0, 3.0 * PI / 8.0];

    // Pre-generate measurements for performance
    let a_measurements = [
        angle_measurement(a_angles[0]),
        angle_measurement(a_angles[1]),
        angle_measurement(a_angles[2]),
    ];
    let b_measurements = [
        angle_measurement(b_angles[0]),
        angle_measurement(b_angles[1]),
        angle_measurement(b_angles[2]),
    ];

    // Eve doesn't know Bob's basis choice at interception time. A reasonable
    // intercept-and-resend strategy is to pick from a fixed set of bases.
    // We use Alice's three angles (any fixed set works; this is symmetric).
    let eve_measurements = &a_measurements;

    for _ in 0..num_pairs {
        // Create singlet state |psi-> = (|01> - |10>)/sqrt(2)
        let mut state = QuantumState::new(2);
        state
            .apply(&Gate::x(), &[0])?
            .apply(&Gate::h(), &[0])?
            .apply(&Gate::cnot(), &[0, 1])?
            .apply(&Gate::x(), &[1])?;

        // Channels
        state
            .apply_channel(channel_alice, &[0])?
            .apply_channel(channel_bob, &[1])?;

        // Eve's intercept-and-resend attack on Bob's qubit.
        // Eve picks a basis blindly (she doesn't know Bob's choice yet).
        if eve_ratio > 0.0 && crate::rng::random_bool(eve_ratio) {
            eve_intercept_count += 1;
            let e_idx = crate::rng::random_usize_range(0, eve_measurements.len());
            let _ = state.measure(&eve_measurements[e_idx], &[1])?;
        }

        // Eve send qubit to Bob through channel
        state.apply_channel(channel_bob, &[1])?;

        // Alice and Bob choose random bases
        let a_idx = crate::rng::random_usize_range(0, a_measurements.len());
        let b_idx = crate::rng::random_usize_range(0, b_measurements.len());

        let res_a = state.measure(&a_measurements[a_idx], &[0])?;
        let res_b = state.measure(&b_measurements[b_idx], &[1])?;

        alice_bits.push(res_a.index == 1);
        bob_results.push(res_b.index == 1);
        alice_bases.push(a_idx);
        bob_bases.push(b_idx);
    }

    // Sifting Stage
    let mut key_indices = Vec::new();
    let mut bell_00 = Vec::new();
    let mut bell_02 = Vec::new();
    let mut bell_20 = Vec::new();
    let mut bell_22 = Vec::new();

    for i in 0..num_pairs {
        let (a_idx, b_idx) = (alice_bases[i], bob_bases[i]);
        if (a_idx == 1 && b_idx == 0) || (a_idx == 2 && b_idx == 1) {
            key_indices.push(i);
        } else if a_idx == 0 && b_idx == 0 {
            bell_00.push(i);
        } else if a_idx == 0 && b_idx == 2 {
            bell_02.push(i);
        } else if a_idx == 2 && b_idx == 0 {
            bell_20.push(i);
        } else if a_idx == 2 && b_idx == 2 {
            bell_22.push(i);
        }
    }

    let total_sifted = key_indices.len();

    // CHSH Calculation: S = E(a1,b1) - E(a1,b3) + E(a3,b1) + E(a3,b3)
    let e00 = calculate_correlation(&bell_00, &alice_bits, &bob_results);
    let e02 = calculate_correlation(&bell_02, &alice_bits, &bob_results);
    let e20 = calculate_correlation(&bell_20, &alice_bits, &bob_results);
    let e22 = calculate_correlation(&bell_22, &alice_bits, &bob_results);
    let chsh_value = e00 - e02 + e20 + e22;

    // Shuffle key indices for QBER check
    crate::rng::shuffle_slice(&mut key_indices);

    let num_check = (total_sifted as f64 * check_ratio).round() as usize;
    let num_check = num_check.min(total_sifted);
    let (check_indices, actual_key_indices) = key_indices.split_at(num_check);

    // Singlet => matching bases produce anticorrelated outcomes.
    // An error is when Alice and Bob got the SAME bit.
    let mut check_errors = 0;
    for &i in check_indices {
        if alice_bits[i] == bob_results[i] {
            check_errors += 1;
        }
    }

    let qber = if num_check > 0 {
        Some(check_errors as f64 / num_check as f64)
    } else {
        None
    };

    // Build both keys. Bob flips his bit because of the singlet's anticorrelation,
    // so in the absence of noise/Eve, alice_key == bob_key.
    let mut alice_key = Vec::with_capacity(actual_key_indices.len());
    let mut bob_key = Vec::with_capacity(actual_key_indices.len());
    for &i in actual_key_indices {
        alice_key.push(alice_bits[i]);
        bob_key.push(!bob_results[i]);
    }

    Ok(E91Result {
        raw_length: num_pairs,
        total_sifted,
        check_errors,
        qber,
        chsh_value,
        eve_intercept_count,
        alice_bases,
        bob_bases,
        alice_bits,
        bob_results,
        alice_key,
        bob_key,
    })
}

/// Parallelized variant of [`run`] using rayon. See [`run`] for protocol semantics.
///
/// # Arguments
///
/// * `num_pairs` - Number of entangled pairs to distribute.
/// * `channel_alice` - The quantum channel affecting Alice's qubit.
/// * `channel_bob` - The quantum channel affecting Bob's qubit.
/// * `eve_ratio` - Probability of Eve intercepting (and measuring) a qubit.
/// * `check_ratio` - Fraction of sifted bits to sacrifice for QBER estimation.
///
/// # Returns
///
/// A `Result` containing `E91Result` with the simulation statistics and keys.
///
/// # Errors
///
/// Returns a `StateError` if quantum operations fail.
///
/// # Example
/// ```rust
/// use qcrypto::protocols::e91;
/// use qcrypto::QuantumChannel;
///
/// let channel_alice = QuantumChannel::bit_flip(0.1);
/// let channel_bob = QuantumChannel::bit_flip(0.05);
///
/// qcrypto::rng::set_global_seed(42);
/// let r1 = e91::run_par(300, &channel_alice, &channel_bob, 0.1, 0.2).unwrap();
///
/// qcrypto::rng::set_global_seed(42);
/// let r2 = e91::run_par(300, &channel_alice, &channel_bob, 0.1, 0.2).unwrap();
///
/// assert_eq!(r1.alice_key, r2.alice_key);
/// ```
pub fn run_par(
    num_pairs: usize,
    channel_alice: &QuantumChannel,
    channel_bob: &QuantumChannel,
    eve_ratio: f64,
    check_ratio: f64,
) -> Result<E91Result, StateError> {
    let master = crate::rng::draw_master_seed();
    let process_seed = crate::rng::draw_master_seed();

    let a_angles = [0.0, PI / 8.0, PI / 4.0];
    let b_angles = [PI / 8.0, PI / 4.0, 3.0 * PI / 8.0];

    let a_measurements = [
        angle_measurement(a_angles[0]),
        angle_measurement(a_angles[1]),
        angle_measurement(a_angles[2]),
    ];
    let b_measurements = [
        angle_measurement(b_angles[0]),
        angle_measurement(b_angles[1]),
        angle_measurement(b_angles[2]),
    ];

    type Step = (usize, usize, bool, bool, bool);

    let steps: Vec<Step> = (0..num_pairs)
        .into_par_iter()
        .map(|i| -> Result<Step, StateError> {
            let mut rng = LocalRng::child(master, i as u64);

            let mut state = QuantumState::new(2);
            state
                .apply(&Gate::x(), &[0])?
                .apply(&Gate::h(), &[0])?
                .apply(&Gate::cnot(), &[0, 1])?
                .apply(&Gate::x(), &[1])?;

            state
                .apply_channel(channel_alice, &[0])?
                .apply_channel(channel_bob, &[1])?;

            let eve_intercepted = eve_ratio > 0.0 && rng.random_bool(eve_ratio);
            if eve_intercepted {
                let e_idx = rng.random_usize_range(0, a_measurements.len());
                let _ = state.measure_with_rng(&a_measurements[e_idx], &[1], &mut rng)?;
            }

            state.apply_channel(channel_bob, &[1])?;

            let a_idx = rng.random_usize_range(0, a_measurements.len());
            let b_idx = rng.random_usize_range(0, b_measurements.len());

            let res_a = state.measure_with_rng(&a_measurements[a_idx], &[0], &mut rng)?;
            let res_b = state.measure_with_rng(&b_measurements[b_idx], &[1], &mut rng)?;

            Ok((
                a_idx,
                b_idx,
                res_a.index == 1,
                res_b.index == 1,
                eve_intercepted,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut alice_bits = Vec::with_capacity(num_pairs);
    let mut alice_bases = Vec::with_capacity(num_pairs);
    let mut bob_bases = Vec::with_capacity(num_pairs);
    let mut bob_results = Vec::with_capacity(num_pairs);
    let mut eve_intercept_count = 0usize;

    for (a_idx, b_idx, a_bit, b_bit, eve_intercepted) in steps {
        alice_bits.push(a_bit);
        bob_results.push(b_bit);
        alice_bases.push(a_idx);
        bob_bases.push(b_idx);
        if eve_intercepted {
            eve_intercept_count += 1;
        }
    }

    let mut key_indices = Vec::new();
    let mut bell_00 = Vec::new();
    let mut bell_02 = Vec::new();
    let mut bell_20 = Vec::new();
    let mut bell_22 = Vec::new();

    for i in 0..num_pairs {
        let (a_idx, b_idx) = (alice_bases[i], bob_bases[i]);
        if (a_idx == 1 && b_idx == 0) || (a_idx == 2 && b_idx == 1) {
            key_indices.push(i);
        } else if a_idx == 0 && b_idx == 0 {
            bell_00.push(i);
        } else if a_idx == 0 && b_idx == 2 {
            bell_02.push(i);
        } else if a_idx == 2 && b_idx == 0 {
            bell_20.push(i);
        } else if a_idx == 2 && b_idx == 2 {
            bell_22.push(i);
        }
    }

    let total_sifted = key_indices.len();

    let e00 = calculate_correlation(&bell_00, &alice_bits, &bob_results);
    let e02 = calculate_correlation(&bell_02, &alice_bits, &bob_results);
    let e20 = calculate_correlation(&bell_20, &alice_bits, &bob_results);
    let e22 = calculate_correlation(&bell_22, &alice_bits, &bob_results);
    let chsh_value = e00 - e02 + e20 + e22;

    let mut rng = LocalRng::from_seed(process_seed);
    rng.shuffle_slice(&mut key_indices);

    let num_check = (total_sifted as f64 * check_ratio).round() as usize;
    let num_check = num_check.min(total_sifted);
    let (check_indices, actual_key_indices) = key_indices.split_at(num_check);

    let mut check_errors = 0;
    for &i in check_indices {
        if alice_bits[i] == bob_results[i] {
            check_errors += 1;
        }
    }

    let qber = if num_check > 0 {
        Some(check_errors as f64 / num_check as f64)
    } else {
        None
    };

    let mut alice_key = Vec::with_capacity(actual_key_indices.len());
    let mut bob_key = Vec::with_capacity(actual_key_indices.len());
    for &i in actual_key_indices {
        alice_key.push(alice_bits[i]);
        bob_key.push(!bob_results[i]);
    }

    Ok(E91Result {
        raw_length: num_pairs,
        total_sifted,
        check_errors,
        qber,
        chsh_value,
        eve_intercept_count,
        alice_bases,
        bob_bases,
        alice_bits,
        bob_results,
        alice_key,
        bob_key,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e91_noiseless() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(2000, &channel, &channel, 0.0, 0.5).unwrap();

        assert_eq!(result.raw_length, 2000);
        assert_eq!(result.check_errors, 0);
        assert_eq!(result.qber, Some(0.0));
        assert_eq!(result.eve_intercept_count, 0);

        // In the noiseless case Alice's and Bob's keys must agree exactly.
        assert_eq!(result.alice_key, result.bob_key);

        // CHSH value should be near -2*sqrt(2) approx -2.828
        assert!(
            result.chsh_value.abs() > 2.5,
            "CHSH value {} should violate Bell inequality (> 2.0)",
            result.chsh_value
        );
        assert!(
            (result.chsh_value + 2.0 * 2.0_f64.sqrt()).abs() < 0.3,
            "CHSH value {} too far from -2.828",
            result.chsh_value
        );
    }

    #[test]
    fn test_e91_eve() {
        let channel = QuantumChannel::bit_flip(0.0);
        // Eve intercepts everything
        let result = run(2000, &channel, &channel, 1.0, 0.5).unwrap();

        assert!(result.eve_intercept_count > 0);
        assert!(result.qber.unwrap() > 0.0);
        // Eve should destroy the Bell inequality violation
        assert!(
            result.chsh_value.abs() < 2.2,
            "CHSH value {} should be near classical limit with Eve",
            result.chsh_value
        );
    }

    #[test]
    fn test_e91_noisy_keys_differ() {
        let channel = QuantumChannel::bit_flip(0.3);
        let result = run(500, &channel, &channel, 0.0, 0.0).unwrap();

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
    fn test_e91_zero_check() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run(100, &channel, &channel, 0.0, 0.0).unwrap();
        assert_eq!(result.qber, None);
    }

    #[test]
    fn test_e91_par_zero_check() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run_par(100, &channel, &channel, 0.0, 0.0).unwrap();
        assert_eq!(result.qber, None);
    }

    #[test]
    fn test_e91_basis_measurement_convention() {
        // Sanity check: |0> measured in the theta=0 basis must yield index 0.
        let m = angle_measurement(0.0);
        let mut s = QuantumState::new(1);
        let r = s.measure(&m, &[0]).unwrap();
        assert_eq!(r.index, 0);
    }

    #[test]
    fn test_calculate_correlation_empty() {
        assert_eq!(calculate_correlation(&[], &[], &[]), 0.0);
    }

    #[test]
    fn test_e91_par_noiseless() {
        let channel = QuantumChannel::bit_flip(0.0);
        let result = run_par(5000, &channel, &channel, 0.0, 0.5).unwrap();
        assert_eq!(result.raw_length, 5000);
        assert_eq!(result.check_errors, 0);
        assert_eq!(result.qber, Some(0.0));
        assert_eq!(result.alice_key, result.bob_key);
        assert!(
            (result.chsh_value + 2.0 * 2.0_f64.sqrt()).abs() < 0.3,
            "CHSH value {} too far from -2.828",
            result.chsh_value
        );
    }

    #[test]
    fn test_e91_par_deterministic_with_seed() {
        let channel = QuantumChannel::bit_flip(0.05);

        crate::rng::set_global_seed(99);
        let r1 = run_par(300, &channel, &channel, 0.1, 0.2).unwrap();
        crate::rng::set_global_seed(99);
        let r2 = run_par(300, &channel, &channel, 0.1, 0.2).unwrap();

        assert_eq!(r1.alice_bits, r2.alice_bits);
        assert_eq!(r1.bob_results, r2.bob_results);
        assert_eq!(r1.alice_key, r2.alice_key);
        assert_eq!(r1.bob_key, r2.bob_key);
        assert_eq!(r1.chsh_value, r2.chsh_value);
    }

    #[test]
    fn test_e91_par_noisy() {
        let channel = QuantumChannel::bit_flip(0.2); // High noise
        let result = run_par(500, &channel, &channel, 0.0, 1.0).unwrap();
        assert!(result.check_errors > 0);
        assert!(result.qber.unwrap() > 0.0);
    }
}
