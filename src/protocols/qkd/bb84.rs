use crate::{Gate, Measurement, QuantumChannel, QuantumState, errors::StateError};
use rand::Rng;

/// BB84 results
pub struct BB84Result {
    pub raw_length: usize,
    pub sifted_length: usize,
    pub errors: usize,
    pub qber: f64,
    pub eve_detected_count: usize,
    pub sifted_key: Vec<bool>,
    pub alice_bits: Vec<bool>,
    pub alice_bases: Vec<bool>,
    pub bob_bases: Vec<bool>,
    pub bob_results: Vec<bool>,
}

/// Runs BB84 protocol
pub fn run(
    num_qubits: usize,
    channel: &QuantumChannel,
    eve_ratio: f64,
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
    let mut sifted_key = Vec::new();
    let mut errors = 0;

    for i in 0..num_qubits {
        if alice_bases[i] == bob_bases[i] {
            sifted_key.push(alice_bits[i]);

            if alice_bits[i] != bob_results[i] {
                errors += 1;
            }
        }
    }

    let sifted_len = sifted_key.len();
    let qber = if sifted_len > 0 {
        (errors as f64 / sifted_len as f64) * 100.0
    } else {
        0.0
    };

    Ok(BB84Result {
        raw_length: num_qubits,
        sifted_length: sifted_len,
        errors,
        qber,
        eve_detected_count: eve_intercepted_count,
        sifted_key,
        alice_bits,
        alice_bases,
        bob_bases,
        bob_results,
    })
}
