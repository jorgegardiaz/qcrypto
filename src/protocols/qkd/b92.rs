use crate::{
    Gate, Measurement, QuantumChannel, QuantumState,
    errors::{MeasurementError, StateError},
};
use ndarray::{Array2, arr2};
use num_complex::Complex64;
use rand::Rng;

pub struct B92Result {
    pub raw_length: usize,
    pub conclusive_count: usize,
    pub errors: usize,
    pub qber: f64,
    pub eve_detected_count: usize,
    pub established_key: Vec<bool>,
    pub alice_bits: Vec<bool>,
    pub bob_results: Vec<i8>,
}

pub fn run(
    num_qubits: usize,
    channel: &QuantumChannel,
    measurement: &Measurement,
    eve_ratio: f64,
) -> Result<B92Result, StateError> {
    let mut rng = rand::rng();

    // Bob's POVM
    let bob_device = measurement;

    let mut alice_bits = Vec::with_capacity(num_qubits);
    let mut bob_results = Vec::with_capacity(num_qubits);
    let mut established_key = Vec::new();
    let mut eve_intercepted_count = 0;
    let mut errors = 0;

    for _ in 0..num_qubits {
        // Alice prepare qubit
        let a_bit = rng.random_bool(0.5);
        let mut state = QuantumState::new(1);

        if a_bit {
            state.apply(&Gate::h(), &[0])?;
        }

        // Alice send qubit to Bob through channel
        state.apply_channel(channel, &[0])?;

        // Eavesdropper intercepts
        if eve_ratio > 1e-12 && rng.random_bool(eve_ratio) {
            eve_intercepted_count += 1;
            let e_basis = rng.random_bool(0.5);
            let m = if e_basis {
                Measurement::x_basis()
            } else {
                Measurement::z_basis()
            };
            let _ = state.measure(&m, &[0])?;
        }

        // Bob measures using his POVM
        let res = state.measure(&bob_device, &[0])?;

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

        // Sifting Stage
        if let Some(inferred_bit) = inferred_bit_opt {
            established_key.push(inferred_bit);
            if inferred_bit != a_bit {
                errors += 1;
            }
        }
    }

    let conclusive_len = established_key.len();
    let qber = if conclusive_len > 0 {
        (errors as f64 / conclusive_len as f64) * 100.0
    } else {
        0.0
    };

    Ok(B92Result {
        raw_length: num_qubits,
        conclusive_count: conclusive_len,
        errors,
        qber,
        eve_detected_count: eve_intercepted_count,
        established_key,
        alice_bits,
        bob_results,
    })
}

/// Contructs optimal POVM using Measurement::from_povm
pub fn build_optimal_povm() -> Result<Measurement, MeasurementError> {
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

    Measurement::from_povm(vec![e1, e2, e3], vec![1.0, 0.0, -1.0])
}
