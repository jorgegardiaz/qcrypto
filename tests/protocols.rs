//! Integration tests for quantum cryptography protocols.
//!
//! End-to-end tests for BB84, B92, BBM92, and QIA-QZKP protocols
//! covering noiseless runs, noisy channels, Eve interception,
//! and deterministic seeded reproducibility.

use qcrypto::QuantumChannel;
use qcrypto::protocols::{b92, bb84, bbm92, qia_qzkp};

// ─── BB84 ────────────────────────────────────────────────────────────────────

#[test]
fn test_bb84_noiseless_zero_qber() {
    qcrypto::set_global_seed(100);
    let channel = QuantumChannel::bit_flip(0.0);
    let result = bb84::run(200, &channel, 0.0, 0.5).unwrap();

    assert_eq!(result.raw_length, 200);
    assert_eq!(result.check_errors, 0);
    assert_eq!(result.qber, 0.0);
    assert_eq!(result.eve_intercepted_count, 0);
    // Sifted key should be roughly half of raw (matching bases ~50%)
    assert!(result.total_sifted > 50);
    assert!(!result.alice_key.is_empty());
}

#[test]
fn test_bb84_eve_introduces_errors() {
    qcrypto::set_global_seed(200);
    let channel = QuantumChannel::bit_flip(0.0);
    // Eve intercepts every qubit
    let result = bb84::run(500, &channel, 1.0, 0.5).unwrap();

    assert!(result.eve_intercepted_count > 0);
    // Eve measuring in random bases should introduce ~25% QBER
    assert!(result.qber > 0.05, "Eve should introduce measurable errors");
}

#[test]
fn test_bb84_noisy_channel_introduces_qber() {
    qcrypto::set_global_seed(300);
    // Moderate bit-flip noise, no Eve
    let channel = QuantumChannel::bit_flip(0.3);
    let result = bb84::run(500, &channel, 0.0, 0.5).unwrap();

    assert_eq!(result.eve_intercepted_count, 0);
    assert!(result.qber > 0.0, "Noise should introduce QBER");
}

#[test]
fn test_bb84_full_check_ratio() {
    qcrypto::set_global_seed(400);
    let channel = QuantumChannel::bit_flip(0.0);
    // Sacrifice ALL sifted bits for checking — established key should be empty
    let result = bb84::run(100, &channel, 0.0, 1.0).unwrap();

    assert!(result.alice_key.is_empty());
    assert_eq!(result.check_errors, 0);
}

#[test]
fn test_bb84_deterministic_with_seed() {
    let run_once = |seed: u64| {
        qcrypto::set_global_seed(seed);
        let channel = QuantumChannel::bit_flip(0.0);
        let r = bb84::run(100, &channel, 0.0, 0.5).unwrap();
        (
            r.alice_bits.clone(),
            r.bob_results.clone(),
            r.alice_key.clone(),
        )
    };

    let (a1, b1, k1) = run_once(42);
    let (a2, b2, k2) = run_once(42);

    assert_eq!(a1, a2);
    assert_eq!(b1, b2);
    assert_eq!(k1, k2);
}

#[test]
fn test_bb84_depolarizing_channel() {
    qcrypto::set_global_seed(500);
    let channel = QuantumChannel::depolarizing(0.2);
    let result = bb84::run(500, &channel, 0.0, 0.5).unwrap();

    // Depolarizing noise should introduce errors
    assert!(result.qber > 0.0);
}

// ─── B92 ─────────────────────────────────────────────────────────────────────

#[test]
fn test_b92_noiseless_zero_qber() {
    qcrypto::set_global_seed(600);
    let channel = QuantumChannel::bit_flip(0.0);
    let measurement = b92::build_optimal_povm_b92().unwrap();
    let result = b92::run(200, &channel, &measurement, 0.0, 0.5).unwrap();

    assert_eq!(result.raw_length, 200);
    assert_eq!(result.check_errors, 0);
    assert_eq!(result.qber, 0.0);
    assert_eq!(result.eve_intercepted_count, 0);
    // B92 has lower yield than BB84 due to inconclusive results
    assert!(result.conclusive_count > 0);
    assert!(result.conclusive_count < result.raw_length);
}

#[test]
fn test_b92_povm_has_three_elements() {
    let povm = b92::build_optimal_povm_b92().unwrap();
    assert_eq!(povm.operators.len(), 3);
}

#[test]
fn test_b92_eve_introduces_errors() {
    qcrypto::set_global_seed(700);
    let channel = QuantumChannel::bit_flip(0.0);
    let measurement = b92::build_optimal_povm_b92().unwrap();
    let result = b92::run(500, &channel, &measurement, 1.0, 0.5).unwrap();

    assert!(result.eve_intercepted_count > 0);
    assert!(result.qber > 0.0);
}

#[test]
fn test_b92_noisy_channel() {
    qcrypto::set_global_seed(800);
    let channel = QuantumChannel::bit_flip(0.2);
    let measurement = b92::build_optimal_povm_b92().unwrap();
    let result = b92::run(500, &channel, &measurement, 0.0, 0.5).unwrap();

    assert!(
        result.qber > 0.0,
        "Channel noise should cause errors in B92"
    );
}

#[test]
fn test_b92_deterministic_with_seed() {
    let run_once = |seed: u64| {
        qcrypto::set_global_seed(seed);
        let channel = QuantumChannel::bit_flip(0.0);
        let m = b92::build_optimal_povm_b92().unwrap();
        let r = b92::run(100, &channel, &m, 0.0, 0.5).unwrap();
        (
            r.alice_bits.clone(),
            r.bob_results.clone(),
            r.alice_key.clone(),
        )
    };

    let (a1, b1, k1) = run_once(77);
    let (a2, b2, k2) = run_once(77);

    assert_eq!(a1, a2);
    assert_eq!(b1, b2);
    assert_eq!(k1, k2);
}

// ─── BBM92 ───────────────────────────────────────────────────────────────────

#[test]
fn test_bbm92_noiseless_zero_qber() {
    qcrypto::set_global_seed(900);
    let channel = QuantumChannel::bit_flip(0.0);
    let result = bbm92::run(200, &channel, &channel, 0.0, 0.5).unwrap();

    assert_eq!(result.raw_length, 200);
    assert_eq!(result.check_errors, 0);
    assert_eq!(result.qber, 0.0);
    assert_eq!(result.eve_intercept_count, 0);
    assert!(result.total_sifted > 50);
}

#[test]
fn test_bbm92_eve_introduces_errors() {
    qcrypto::set_global_seed(1000);
    let channel = QuantumChannel::bit_flip(0.0);
    let result = bbm92::run(500, &channel, &channel, 1.0, 0.5).unwrap();

    assert!(result.eve_intercept_count > 0);
    assert!(result.qber > 0.0);
}

#[test]
fn test_bbm92_noisy_channel() {
    qcrypto::set_global_seed(1100);
    let channel = QuantumChannel::depolarizing(0.15);
    let result = bbm92::run(500, &channel, &channel, 0.0, 0.5).unwrap();

    assert!(
        result.qber > 0.0,
        "Channel noise should introduce QBER in BBM92"
    );
}

#[test]
fn test_bbm92_deterministic_with_seed() {
    let run_once = |seed: u64| {
        qcrypto::set_global_seed(seed);
        let channel = QuantumChannel::bit_flip(0.0);
        let r = bbm92::run(100, &channel, &channel, 0.0, 0.5).unwrap();
        (
            r.alice_bits.clone(),
            r.bob_results.clone(),
            r.alice_key.clone(),
        )
    };

    let (a1, b1, k1) = run_once(55);
    let (a2, b2, k2) = run_once(55);

    assert_eq!(a1, a2);
    assert_eq!(b1, b2);
    assert_eq!(k1, k2);
}

#[test]
fn test_bbm92_full_check_ratio() {
    qcrypto::set_global_seed(1200);
    let channel = QuantumChannel::bit_flip(0.0);
    let result = bbm92::run(100, &channel, &channel, 0.0, 1.0).unwrap();

    assert!(result.alice_key.is_empty());
    assert_eq!(result.check_errors, 0);
}

// ─── QIA-QZKP ────────────────────────────────────────────────────────────────

#[test]
fn test_qia_qzkp_noiseless_authenticates() {
    qcrypto::set_global_seed(1300);
    let channel = QuantumChannel::bit_flip(0.0);
    let result = qia_qzkp::run(200, &channel, 0.9).unwrap();

    assert_eq!(result.total_qubits, 200);
    assert_eq!(result.matches, 200);
    assert_eq!(result.accuracy, 1.0);
    assert!(result.authenticated);
}

#[test]
fn test_qia_qzkp_extreme_noise_rejects() {
    qcrypto::set_global_seed(1400);
    // Full bit-flip should destroy the protocol
    let channel = QuantumChannel::bit_flip(1.0);
    let result = qia_qzkp::run(200, &channel, 0.9).unwrap();

    assert!(
        !result.authenticated,
        "Extreme noise should cause authentication failure"
    );
    assert!(result.accuracy < 0.9);
}

#[test]
fn test_qia_qzkp_moderate_noise() {
    qcrypto::set_global_seed(1500);
    let channel = QuantumChannel::bit_flip(0.1);
    let result = qia_qzkp::run(500, &channel, 0.95).unwrap();

    // With 10% bit flip the accuracy should drop below 95%
    assert!(result.accuracy < 1.0);
}

#[test]
fn test_qia_qzkp_deterministic_with_seed() {
    let run_once = |seed: u64| {
        qcrypto::set_global_seed(seed);
        let channel = QuantumChannel::bit_flip(0.0);
        let r = qia_qzkp::run(100, &channel, 0.9).unwrap();
        (
            r.alice_id_a.clone(),
            r.bob_challenge_c.clone(),
            r.bob_recovered_c.clone(),
        )
    };

    let (a1, c1, cr1) = run_once(99);
    let (a2, c2, cr2) = run_once(99);

    assert_eq!(a1, a2);
    assert_eq!(c1, c2);
    assert_eq!(cr1, cr2);
}

#[test]
fn test_qia_qzkp_low_threshold_accepts_noisy() {
    qcrypto::set_global_seed(1600);
    let channel = QuantumChannel::bit_flip(0.05);
    // Very low threshold — should still authenticate
    let result = qia_qzkp::run(500, &channel, 0.5).unwrap();

    assert!(result.authenticated);
}

// ─── Cross-protocol comparison ───────────────────────────────────────────────

#[test]
fn test_bb84_vs_bbm92_noiseless_both_zero_qber() {
    qcrypto::set_global_seed(2000);
    let channel = QuantumChannel::bit_flip(0.0);
    let bb84_result = bb84::run(300, &channel, 0.0, 0.5).unwrap();

    qcrypto::set_global_seed(2001);
    let bbm92_result = bbm92::run(300, &channel, &channel, 0.0, 0.5).unwrap();

    // Both should have zero QBER in noiseless scenario
    assert_eq!(bb84_result.qber, 0.0);
    assert_eq!(bbm92_result.qber, 0.0);
}

#[test]
fn test_all_protocols_handle_amplitude_damping() {
    qcrypto::set_global_seed(3000);
    let channel = QuantumChannel::amplitude_damping(0.2);

    let bb84_r = bb84::run(300, &channel, 0.0, 0.5).unwrap();
    assert_eq!(bb84_r.raw_length, 300);

    qcrypto::set_global_seed(3001);
    let measurement = b92::build_optimal_povm_b92().unwrap();
    let b92_r = b92::run(300, &channel, &measurement, 0.0, 0.5).unwrap();
    assert_eq!(b92_r.raw_length, 300);

    qcrypto::set_global_seed(3002);
    let bbm92_r = bbm92::run(300, &channel, &channel, 0.0, 0.5).unwrap();
    assert_eq!(bbm92_r.raw_length, 300);

    qcrypto::set_global_seed(3003);
    let qzkp_r = qia_qzkp::run(300, &channel, 0.5).unwrap();
    assert_eq!(qzkp_r.total_qubits, 300);
}
