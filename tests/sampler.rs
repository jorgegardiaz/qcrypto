//! Integration tests for the Sampler.
//!
//! Tests measurement sampling with deterministic states, superpositions,
//! noise channels, multi-qubit systems, and seeded reproducibility.

use qcrypto::{Gate, Measurement, QuantumChannel, QuantumState, Sampler};

// ─── Deterministic states ────────────────────────────────────────────────────

#[test]
fn test_sampler_ground_state_always_zero() {
    let state = QuantumState::new(1); // |0>
    let sampler = Sampler::new();
    let counts = sampler
        .run(&state, &Measurement::z_basis(), &[0], 500)
        .unwrap();

    assert_eq!(counts.len(), 1);
    assert_eq!(*counts.get("0").unwrap(), 500);
}

#[test]
fn test_sampler_excited_state_always_one() {
    let mut state = QuantumState::new(1);
    state.apply(&Gate::x(), &[0]).unwrap(); // |1>

    let sampler = Sampler::new();
    let counts = sampler
        .run(&state, &Measurement::z_basis(), &[0], 500)
        .unwrap();

    assert_eq!(counts.len(), 1);
    assert_eq!(*counts.get("1").unwrap(), 500);
}

// ─── Superposition statistics ────────────────────────────────────────────────

#[test]
fn test_sampler_hadamard_roughly_balanced() {
    qcrypto::set_global_seed(1000);
    let mut state = QuantumState::new(1);
    state.apply(&Gate::h(), &[0]).unwrap(); // |+>

    let sampler = Sampler::new();
    let num_shots = 5000;
    let counts = sampler
        .run(&state, &Measurement::z_basis(), &[0], num_shots)
        .unwrap();

    let c0 = *counts.get("0").unwrap_or(&0);
    let c1 = *counts.get("1").unwrap_or(&0);

    assert_eq!(c0 + c1, num_shots);
    // With 5000 shots, expect ~2500 each. Allow 15% deviation.
    assert!(c0 > 1750 && c0 < 3250, "Expected ~2500, got {}", c0);
    assert!(c1 > 1750 && c1 < 3250, "Expected ~2500, got {}", c1);
}

#[test]
fn test_sampler_x_basis_on_plus_state() {
    let mut state = QuantumState::new(1);
    state.apply(&Gate::h(), &[0]).unwrap(); // |+>

    let sampler = Sampler::new();
    let counts = sampler
        .run(&state, &Measurement::x_basis(), &[0], 500)
        .unwrap();

    // |+> measured in X basis should always give outcome 0 (eigenvalue +1)
    assert_eq!(*counts.get("0").unwrap_or(&0), 500);
}

// ─── Multi-qubit sampling ────────────────────────────────────────────────────

#[test]
fn test_sampler_bell_state_z_basis() {
    qcrypto::set_global_seed(2000);
    let mut state = QuantumState::new(2);
    state.apply(&Gate::h(), &[0]).unwrap();
    state.apply(&Gate::cnot(), &[0, 1]).unwrap(); // Bell state |Φ+>

    let sampler = Sampler::new();

    // Measure qubit 0 only — should be roughly 50/50
    let counts = sampler
        .run(&state, &Measurement::z_basis(), &[0], 2000)
        .unwrap();

    let c0 = *counts.get("0").unwrap_or(&0);
    let c1 = *counts.get("1").unwrap_or(&0);
    assert_eq!(c0 + c1, 2000);
    assert!(c0 > 700 && c0 < 1300);
}

#[test]
fn test_sampler_bell_basis_on_bell_state() {
    let mut state = QuantumState::new(2);
    state.apply(&Gate::h(), &[0]).unwrap();
    state.apply(&Gate::cnot(), &[0, 1]).unwrap(); // |Φ+>

    let sampler = Sampler::new();
    let counts = sampler
        .run(&state, &Measurement::bell_basis(), &[0, 1], 500)
        .unwrap();

    // Bell state measured in Bell basis should give 100% one outcome
    assert_eq!(
        counts.len(),
        1,
        "Bell state should give a single Bell outcome"
    );
    let total: usize = counts.values().sum();
    assert_eq!(total, 500);
}

// ─── Sampler with channel ────────────────────────────────────────────────────

#[test]
fn test_sampler_with_bit_flip_channel() {
    let state = QuantumState::new(1); // |0>
    let channel = QuantumChannel::bit_flip(1.0); // Always flip
    let sampler = Sampler::new().with_channel(channel);

    let counts = sampler
        .run(&state, &Measurement::z_basis(), &[0], 200)
        .unwrap();

    // Full bit flip: |0> -> |1>
    assert_eq!(*counts.get("1").unwrap(), 200);
}

#[test]
fn test_sampler_with_depolarizing_channel() {
    qcrypto::set_global_seed(3000);
    let state = QuantumState::new(1); // |0>
    let channel = QuantumChannel::depolarizing(1.0); // Fully depolarize
    let sampler = Sampler::new().with_channel(channel);

    let num_shots = 5000;
    let counts = sampler
        .run(&state, &Measurement::z_basis(), &[0], num_shots)
        .unwrap();

    let c0 = *counts.get("0").unwrap_or(&0);
    let c1 = *counts.get("1").unwrap_or(&0);

    // Maximally mixed state should give ~50/50
    assert!(c0 > 1750 && c0 < 3250, "Expected ~2500, got {}", c0);
    assert!(c1 > 1750 && c1 < 3250, "Expected ~2500, got {}", c1);
}

#[test]
fn test_sampler_identity_channel_no_effect() {
    let state = QuantumState::new(1); // |0>
    let channel = QuantumChannel::bit_flip(0.0); // Identity
    let sampler = Sampler::new().with_channel(channel);

    let counts = sampler
        .run(&state, &Measurement::z_basis(), &[0], 200)
        .unwrap();

    assert_eq!(*counts.get("0").unwrap(), 200);
}

// ─── Seeded determinism ──────────────────────────────────────────────────────

#[test]
fn test_sampler_deterministic_with_seed() {
    let run_once = |seed: u64| {
        qcrypto::set_global_seed(seed);
        let mut state = QuantumState::new(1);
        state.apply(&Gate::h(), &[0]).unwrap();
        let sampler = Sampler::new();
        sampler
            .run(&state, &Measurement::z_basis(), &[0], 100)
            .unwrap()
    };

    let counts1 = run_once(42);
    let counts2 = run_once(42);

    assert_eq!(
        counts1, counts2,
        "Same seed should produce identical counts"
    );
}

#[test]
fn test_sampler_different_seeds_differ() {
    // Not strictly guaranteed but extremely likely with enough shots
    let run_once = |seed: u64| {
        qcrypto::set_global_seed(seed);
        let mut state = QuantumState::new(1);
        state.apply(&Gate::h(), &[0]).unwrap();
        let sampler = Sampler::new();
        sampler
            .run(&state, &Measurement::z_basis(), &[0], 1000)
            .unwrap()
    };

    let counts1 = run_once(1);
    let counts2 = run_once(999);

    // Very unlikely to be identical with 1000 shots of a fair coin
    let c0_1 = *counts1.get("0").unwrap_or(&0);
    let c0_2 = *counts2.get("0").unwrap_or(&0);
    // This can technically fail, but with probability < 1e-6
    assert_ne!(
        c0_1, c0_2,
        "Different seeds should produce different results"
    );
}

// ─── Error propagation ───────────────────────────────────────────────────────

#[test]
fn test_sampler_out_of_bounds_target() {
    let state = QuantumState::new(1);
    let sampler = Sampler::new();

    let result = sampler.run(&state, &Measurement::z_basis(), &[5], 10);
    assert!(result.is_err());
}

#[test]
fn test_sampler_dimension_mismatch() {
    let state = QuantumState::new(1);
    let sampler = Sampler::new();

    // Bell basis requires 2 qubits but we pass only 1 target
    let result = sampler.run(&state, &Measurement::bell_basis(), &[0], 10);
    assert!(result.is_err());
}

// ─── Builder pattern ─────────────────────────────────────────────────────────

#[test]
fn test_sampler_builder_chain() {
    let channel = QuantumChannel::phase_flip(0.1);
    let sampler = Sampler::new().with_channel(channel);

    assert!(sampler.channel.is_some());
    assert_eq!(sampler.channel.as_ref().unwrap().num_qubits, 1);
}

#[test]
fn test_sampler_default_has_no_channel() {
    let sampler = Sampler::new();
    assert!(sampler.channel.is_none());
}

// ─── run_par: deterministic states ───────────────────────────────────────────

#[test]
fn test_sampler_par_ground_state_always_zero() {
    let state = QuantumState::new(1); // |0>
    let sampler = Sampler::new();
    let counts = sampler
        .run_par(&state, &Measurement::z_basis(), &[0], 500)
        .unwrap();

    assert_eq!(counts.len(), 1);
    assert_eq!(*counts.get("0").unwrap(), 500);
}

#[test]
fn test_sampler_par_excited_state_always_one() {
    let mut state = QuantumState::new(1);
    state.apply(&Gate::x(), &[0]).unwrap(); // |1>

    let sampler = Sampler::new();
    let counts = sampler
        .run_par(&state, &Measurement::z_basis(), &[0], 500)
        .unwrap();

    assert_eq!(counts.len(), 1);
    assert_eq!(*counts.get("1").unwrap(), 500);
}

// ─── run_par: superposition statistics ───────────────────────────────────────

#[test]
fn test_sampler_par_hadamard_roughly_balanced() {
    qcrypto::set_global_seed(5000);
    let mut state = QuantumState::new(1);
    state.apply(&Gate::h(), &[0]).unwrap(); // |+>

    let sampler = Sampler::new();
    let num_shots = 5000;
    let counts = sampler
        .run_par(&state, &Measurement::z_basis(), &[0], num_shots)
        .unwrap();

    let c0 = *counts.get("0").unwrap_or(&0);
    let c1 = *counts.get("1").unwrap_or(&0);

    assert_eq!(c0 + c1, num_shots);
    assert!(c0 > 1750 && c0 < 3250, "Expected ~2500, got {}", c0);
    assert!(c1 > 1750 && c1 < 3250, "Expected ~2500, got {}", c1);
}

// ─── run_par: multi-qubit ─────────────────────────────────────────────────────

#[test]
fn test_sampler_par_bell_state_z_basis() {
    qcrypto::set_global_seed(6000);
    let mut state = QuantumState::new(2);
    state.apply(&Gate::h(), &[0]).unwrap();
    state.apply(&Gate::cnot(), &[0, 1]).unwrap(); // Bell state |Φ+>

    let sampler = Sampler::new();
    let counts = sampler
        .run_par(&state, &Measurement::z_basis(), &[0], 2000)
        .unwrap();

    let c0 = *counts.get("0").unwrap_or(&0);
    let c1 = *counts.get("1").unwrap_or(&0);
    assert_eq!(c0 + c1, 2000);
    assert!(c0 > 700 && c0 < 1300);
}

#[test]
fn test_sampler_par_bell_basis_on_bell_state() {
    let mut state = QuantumState::new(2);
    state.apply(&Gate::h(), &[0]).unwrap();
    state.apply(&Gate::cnot(), &[0, 1]).unwrap(); // |Φ+>

    let sampler = Sampler::new();
    let counts = sampler
        .run_par(&state, &Measurement::bell_basis(), &[0, 1], 500)
        .unwrap();

    assert_eq!(counts.len(), 1, "Bell state measured in Bell basis should give one outcome");
    let total: usize = counts.values().sum();
    assert_eq!(total, 500);
}

// ─── run_par: channel ─────────────────────────────────────────────────────────

#[test]
fn test_sampler_par_with_bit_flip_channel() {
    let state = QuantumState::new(1); // |0>
    let channel = QuantumChannel::bit_flip(1.0); // always flip
    let sampler = Sampler::new().with_channel(channel);

    let counts = sampler
        .run_par(&state, &Measurement::z_basis(), &[0], 200)
        .unwrap();

    assert_eq!(*counts.get("1").unwrap(), 200);
}

#[test]
fn test_sampler_par_identity_channel_no_effect() {
    let state = QuantumState::new(1); // |0>
    let channel = QuantumChannel::bit_flip(0.0); // identity
    let sampler = Sampler::new().with_channel(channel);

    let counts = sampler
        .run_par(&state, &Measurement::z_basis(), &[0], 200)
        .unwrap();

    assert_eq!(*counts.get("0").unwrap(), 200);
}

// ─── run_par: seeded determinism ─────────────────────────────────────────────

#[test]
fn test_sampler_par_deterministic_with_seed() {
    let run_once = |seed: u64| {
        qcrypto::set_global_seed(seed);
        let mut state = QuantumState::new(1);
        state.apply(&Gate::h(), &[0]).unwrap();
        let sampler = Sampler::new();
        sampler
            .run_par(&state, &Measurement::z_basis(), &[0], 1000)
            .unwrap()
    };

    let c1 = run_once(77);
    let c2 = run_once(77);
    assert_eq!(c1, c2, "Same seed must produce identical counts");
}

#[test]
fn test_sampler_par_different_seeds_differ() {
    let run_once = |seed: u64| {
        qcrypto::set_global_seed(seed);
        let mut state = QuantumState::new(1);
        state.apply(&Gate::h(), &[0]).unwrap();
        let sampler = Sampler::new();
        sampler
            .run_par(&state, &Measurement::z_basis(), &[0], 1000)
            .unwrap()
    };

    let c1 = run_once(2);
    let c2 = run_once(998);
    let n0_1 = *c1.get("0").unwrap_or(&0);
    let n0_2 = *c2.get("0").unwrap_or(&0);
    assert_ne!(n0_1, n0_2, "Different seeds should produce different results");
}

// ─── run_par: error propagation ──────────────────────────────────────────────

#[test]
fn test_sampler_par_out_of_bounds_target() {
    let state = QuantumState::new(1);
    let sampler = Sampler::new();

    let result = sampler.run_par(&state, &Measurement::z_basis(), &[5], 10);
    assert!(result.is_err());
}

#[test]
fn test_sampler_par_dimension_mismatch() {
    let state = QuantumState::new(1);
    let sampler = Sampler::new();

    let result = sampler.run_par(&state, &Measurement::bell_basis(), &[0], 10);
    assert!(result.is_err());
}
