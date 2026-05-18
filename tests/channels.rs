//! Integration tests for quantum channels.
//!
//! Tests channel composition, mixing, operator expansion, CPTP preservation,
//! and the physical effects of channels on quantum states.

use ndarray::Array2;
use num_complex::Complex64;
use qcrypto::{Gate, Measurement, QuantumChannel, QuantumState};

/// Verify CPTP: sum(K†K) = I for a given channel.
fn assert_cptp(channel: &QuantumChannel, tol: f64) {
    let dim = channel.kraus_ops[0].dim().0;
    let eye = Array2::<Complex64>::eye(dim);
    let mut sum = Array2::<Complex64>::zeros((dim, dim));
    for op in &channel.kraus_ops {
        let dag = op.t().mapv(|c| c.conj());
        sum = sum + dag.dot(op);
    }
    for (a, b) in sum.iter().zip(eye.iter()) {
        assert!(
            (*a - *b).norm() < tol,
            "CPTP violation: element diff = {}",
            (*a - *b).norm()
        );
    }
}

// ─── CPTP preservation after composition and mixing ──────────────────────────

#[test]
fn test_compose_preserves_cptp() {
    let channels = [
        QuantumChannel::identity(),
        QuantumChannel::bit_flip(0.1),
        QuantumChannel::phase_flip(0.2),
        QuantumChannel::depolarizing(0.15),
        QuantumChannel::amplitude_damping(0.3),
    ];

    for (i, c1) in channels.iter().enumerate() {
        for c2 in channels.iter().skip(i) {
            let composed = c1.compose(c2).unwrap();
            assert_cptp(&composed, 1e-10);
        }
    }
}

#[test]
fn test_mix_preserves_cptp() {
    let c1 = QuantumChannel::bit_flip(0.3);
    let c2 = QuantumChannel::phase_flip(0.4);

    for i in 0..=10 {
        let p = i as f64 / 10.0;
        let mixed = c1.mix(&c2, p).unwrap();
        assert_cptp(&mixed, 1e-10);
    }
}

#[test]
fn test_triple_composition_cptp() {
    let c1 = QuantumChannel::bit_flip(0.1);
    let c2 = QuantumChannel::phase_flip(0.2);
    let c3 = QuantumChannel::amplitude_damping(0.15);

    let composed = c1.compose(&c2).unwrap().compose(&c3).unwrap();
    assert_cptp(&composed, 1e-10);
}

// ─── Channel effects on quantum states ───────────────────────────────────────

#[test]
fn test_bit_flip_p1_flips_state() {
    let mut state = QuantumState::new(1); // |0>
    let channel = QuantumChannel::bit_flip(1.0);
    state.apply_channel(&channel, &[0]).unwrap();

    // After full bit flip, should be |1><1|
    let probs = state
        .set_measurement(&Measurement::z_basis(), &[0])
        .unwrap();
    assert!((probs[0]).abs() < 1e-12, "P(|0>) should be 0");
    assert!((probs[1] - 1.0).abs() < 1e-12, "P(|1>) should be 1");
}

#[test]
fn test_depolarizing_p1_gives_maximally_mixed() {
    let mut state = QuantumState::new(1); // |0>
    let channel = QuantumChannel::depolarizing(1.0);
    state.apply_channel(&channel, &[0]).unwrap();

    // Should become I/2
    let probs = state
        .set_measurement(&Measurement::z_basis(), &[0])
        .unwrap();
    assert!((probs[0] - 0.5).abs() < 1e-10, "Should be maximally mixed");
    assert!((probs[1] - 0.5).abs() < 1e-10, "Should be maximally mixed");
}

#[test]
fn test_amplitude_damping_decays_excited_state() {
    let mut state = QuantumState::new(1);
    state.apply(&Gate::x(), &[0]).unwrap(); // |1>

    let channel = QuantumChannel::amplitude_damping(1.0);
    state.apply_channel(&channel, &[0]).unwrap();

    // Full amplitude damping: |1> -> |0>
    let probs = state
        .set_measurement(&Measurement::z_basis(), &[0])
        .unwrap();
    assert!((probs[0] - 1.0).abs() < 1e-12, "Should decay to |0>");
    assert!(probs[1].abs() < 1e-12);
}

#[test]
fn test_amplitude_damping_preserves_ground_state() {
    let mut state = QuantumState::new(1); // |0>
    let channel = QuantumChannel::amplitude_damping(0.5);
    state.apply_channel(&channel, &[0]).unwrap();

    // |0> should be unaffected by amplitude damping
    let probs = state
        .set_measurement(&Measurement::z_basis(), &[0])
        .unwrap();
    assert!((probs[0] - 1.0).abs() < 1e-12);
}

#[test]
fn test_phase_damping_preserves_populations() {
    let mut state = QuantumState::new(1);
    state.apply(&Gate::h(), &[0]).unwrap(); // |+>

    let probs_before = state
        .set_measurement(&Measurement::z_basis(), &[0])
        .unwrap();

    let channel = QuantumChannel::phase_damping(0.5);
    state.apply_channel(&channel, &[0]).unwrap();

    let probs_after = state
        .set_measurement(&Measurement::z_basis(), &[0])
        .unwrap();

    // Phase damping should NOT change Z-basis populations
    assert!((probs_before[0] - probs_after[0]).abs() < 1e-12);
    assert!((probs_before[1] - probs_after[1]).abs() < 1e-12);
}

#[test]
fn test_phase_damping_destroys_coherence() {
    let mut state = QuantumState::new(1);
    state.apply(&Gate::h(), &[0]).unwrap(); // |+>

    let channel = QuantumChannel::phase_damping(1.0);
    state.apply_channel(&channel, &[0]).unwrap();

    // Full phase damping on |+> should give maximally mixed X-basis measurement
    // but the purity should drop
    assert!(
        state.purity() < 1.0 - 1e-6,
        "Full phase damping should reduce purity"
    );
}

// ─── Purity effects ─────────────────────────────────────────────────────────

#[test]
fn test_identity_channel_preserves_purity() {
    let mut state = QuantumState::new(1);
    state.apply(&Gate::h(), &[0]).unwrap();

    let purity_before = state.purity();

    let channel = QuantumChannel::bit_flip(0.0); // Identity
    state.apply_channel(&channel, &[0]).unwrap();

    let purity_after = state.purity();
    assert!((purity_before - purity_after).abs() < 1e-12);
}

#[test]
fn test_depolarizing_reduces_purity() {
    let mut state = QuantumState::new(1);
    state.apply(&Gate::h(), &[0]).unwrap();

    let channel = QuantumChannel::depolarizing(0.5);
    state.apply_channel(&channel, &[0]).unwrap();

    assert!(
        state.purity() < 1.0 - 1e-6,
        "Depolarizing should reduce purity"
    );
    assert!(
        state.purity() > 0.5 - 1e-6,
        "Purity should remain above 0.5 for p=0.5"
    );
}

#[test]
fn test_purity_monotone_with_noise_strength() {
    let noise_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0];
    let mut prev_purity = 2.0; // Start above max so first comparison always passes

    for &p in &noise_levels {
        let mut state = QuantumState::new(1);
        state.apply(&Gate::h(), &[0]).unwrap();

        let channel = QuantumChannel::depolarizing(p);
        state.apply_channel(&channel, &[0]).unwrap();

        let purity = state.purity();
        assert!(
            purity <= prev_purity + 1e-10,
            "Purity should decrease with noise: p={}, purity={}, prev={}",
            p,
            purity,
            prev_purity
        );
        prev_purity = purity;
    }
}

// ─── Operator expansion ─────────────────────────────────────────────────────

#[test]
fn test_expanded_operators_dimensions() {
    let channel = QuantumChannel::bit_flip(0.3);

    for total_qubits in 2..=4 {
        let expanded = channel.get_expanded_operators(total_qubits, &[0]).unwrap();
        let expected_dim = 1 << total_qubits;
        for op in &expanded {
            assert_eq!(op.dim(), (expected_dim, expected_dim));
        }
    }
}

#[test]
fn test_expanded_operators_cptp() {
    let channel = QuantumChannel::bit_flip(0.3);
    let expanded = channel.get_expanded_operators(3, &[1]).unwrap();

    // Expanded operators should also satisfy sum(K†K) = I
    let dim = expanded[0].dim().0;
    let eye = Array2::<Complex64>::eye(dim);
    let mut sum = Array2::<Complex64>::zeros((dim, dim));
    for op in &expanded {
        let dag = op.t().mapv(|c| c.conj());
        sum = sum + dag.dot(op);
    }
    for (a, b) in sum.iter().zip(eye.iter()) {
        assert!((*a - *b).norm() < 1e-10);
    }
}

#[test]
fn test_channel_on_different_targets() {
    qcrypto::set_global_seed(5000);

    // Apply bit flip on qubit 0 vs qubit 1 of a 2-qubit system — different effects
    let channel = QuantumChannel::bit_flip(1.0);

    let mut state0 = QuantumState::new(2); // |00>
    state0.apply_channel(&channel, &[0]).unwrap();
    let probs0 = state0
        .set_measurement(&Measurement::z_basis(), &[0])
        .unwrap();
    // Qubit 0 should be flipped to |1>
    assert!((probs0[1] - 1.0).abs() < 1e-12);

    let mut state1 = QuantumState::new(2); // |00>
    state1.apply_channel(&channel, &[1]).unwrap();
    let probs1 = state1
        .set_measurement(&Measurement::z_basis(), &[1])
        .unwrap();
    // Qubit 1 should be flipped to |1>
    assert!((probs1[1] - 1.0).abs() < 1e-12);
}

// ─── Combined channel ────────────────────────────────────────────────────────

#[test]
fn test_combined_amplitude_phase_damping_cptp() {
    for i in 0..=5 {
        for j in 0..=5 {
            let gamma = i as f64 / 5.0;
            let lambda = j as f64 / 5.0;
            let ch = QuantumChannel::combined_amplitude_phase_damping(gamma, lambda);
            assert_cptp(&ch, 1e-10);
        }
    }
}

#[test]
fn test_combined_channel_with_zero_params_is_identity() {
    let ch = QuantumChannel::combined_amplitude_phase_damping(0.0, 0.0);

    let mut state = QuantumState::new(1);
    state.apply(&Gate::h(), &[0]).unwrap();
    let probs_before = state
        .set_measurement(&Measurement::z_basis(), &[0])
        .unwrap();

    state.apply_channel(&ch, &[0]).unwrap();
    let probs_after = state
        .set_measurement(&Measurement::z_basis(), &[0])
        .unwrap();

    assert!((probs_before[0] - probs_after[0]).abs() < 1e-12);
    assert!((probs_before[1] - probs_after[1]).abs() < 1e-12);
}

// ─── Error cases ─────────────────────────────────────────────────────────────

#[test]
fn test_compose_different_sizes_fails() {
    let c1 = QuantumChannel::bit_flip(0.1); // 1-qubit
    let eye4: Array2<Complex64> = Array2::eye(4);
    let c2 = QuantumChannel::new(vec![eye4]).unwrap(); // 2-qubit

    assert!(c1.compose(&c2).is_err());
}

#[test]
fn test_mix_invalid_probability_fails() {
    let c1 = QuantumChannel::bit_flip(0.1);
    let c2 = QuantumChannel::phase_flip(0.1);

    assert!(c1.mix(&c2, -0.1).is_err());
    assert!(c1.mix(&c2, 1.5).is_err());
}

#[test]
fn test_expanded_operators_wrong_target_count_fails() {
    let channel = QuantumChannel::bit_flip(0.5);
    // 1-qubit channel but 2 targets
    assert!(channel.get_expanded_operators(3, &[0, 1]).is_err());
}
