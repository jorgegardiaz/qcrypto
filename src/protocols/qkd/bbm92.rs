use crate::{Gate, Measurement, QuantumChannel, QuantumState, errors::StateError};
use rand::Rng;

/// BBM92 results
pub struct Bbm92Result {
    pub raw_length: usize,
    pub sifted_length: usize,
    pub errors: usize,
    pub qber: f64,
    pub eve_detected_count: usize,
    pub alice_bases: Vec<bool>,
    pub bob_bases: Vec<bool>,
    pub alice_bits: Vec<bool>,
    pub bob_results: Vec<bool>,
}

/// Runs BBM92 protocol experiment (entanglement-based protocol)
pub fn run(
    num_pairs: usize,
    channel: &QuantumChannel,
    eve_ratio: f64,
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
    let mut sifted_len = 0;
    let mut errors = 0;

    for i in 0..num_pairs {
        if alice_bases[i] == bob_bases[i] {
            sifted_len += 1;
            if alice_bits[i] != bob_results[i] {
                errors += 1;
            }
        }
    }

    let qber = if sifted_len > 0 {
        (errors as f64 / sifted_len as f64) * 100.0
    } else {
        0.0
    };

    Ok(Bbm92Result {
        raw_length: num_pairs,
        sifted_length: sifted_len,
        errors,
        qber,
        eve_detected_count,
        alice_bases,
        bob_bases,
        alice_bits,
        bob_results,
    })
}
