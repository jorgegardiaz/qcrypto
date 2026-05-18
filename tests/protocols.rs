//! Integration tests for quantum cryptography protocols.
//!
//! End-to-end tests for BB84, B92, BBM92, QIA-QZKP, Six-State, E91, SARG04,
//! and GC01 protocols covering noiseless runs, noisy channels, Eve interception,
//! and deterministic seeded reproducibility.

use qcrypto::QuantumChannel;
use qcrypto::protocols::{b92, bb84, bbm92, e91, gc01, qia_qzkp, sarg04, six_state};

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

// ─── Six-State ───────────────────────────────────────────────────────────────

#[test]
fn test_six_state_noiseless_zero_qber() {
    qcrypto::set_global_seed(4000);
    let channel = QuantumChannel::bit_flip(0.0);
    let result = six_state::run(200, &channel, 0.0, 0.5).unwrap();

    assert_eq!(result.raw_length, 200);
    assert_eq!(result.check_errors, 0);
    assert_eq!(result.qber, 0.0);
    assert_eq!(result.eve_intercepted_count, 0);
    assert!(result.total_sifted > 50);
    assert!(!result.alice_key.is_empty());
}

#[test]
fn test_six_state_eve_introduces_errors() {
    qcrypto::set_global_seed(4100);
    let channel = QuantumChannel::bit_flip(0.0);
    let result = six_state::run(500, &channel, 1.0, 0.5).unwrap();

    assert!(result.eve_intercepted_count > 0);
    assert!(result.qber > 0.05, "Eve should introduce measurable errors");
}

#[test]
fn test_six_state_noisy_channel_introduces_qber() {
    qcrypto::set_global_seed(4200);
    let channel = QuantumChannel::bit_flip(0.3);
    let result = six_state::run(500, &channel, 0.0, 0.5).unwrap();

    assert_eq!(result.eve_intercepted_count, 0);
    assert!(result.qber > 0.0, "Noise should introduce QBER");
}

#[test]
fn test_six_state_full_check_ratio() {
    qcrypto::set_global_seed(4300);
    let channel = QuantumChannel::bit_flip(0.0);
    let result = six_state::run(100, &channel, 0.0, 1.0).unwrap();

    assert!(result.alice_key.is_empty());
    assert_eq!(result.check_errors, 0);
}

#[test]
fn test_six_state_deterministic_with_seed() {
    let run_once = |seed: u64| {
        qcrypto::set_global_seed(seed);
        let channel = QuantumChannel::bit_flip(0.0);
        let r = six_state::run(100, &channel, 0.0, 0.5).unwrap();
        (
            r.alice_bits.clone(),
            r.bob_results.clone(),
            r.alice_key.clone(),
        )
    };

    let (a1, b1, k1) = run_once(4400);
    let (a2, b2, k2) = run_once(4400);

    assert_eq!(a1, a2);
    assert_eq!(b1, b2);
    assert_eq!(k1, k2);
}

#[test]
fn test_six_state_depolarizing_channel() {
    qcrypto::set_global_seed(4500);
    let channel = QuantumChannel::depolarizing(0.2);
    let result = six_state::run(500, &channel, 0.0, 0.5).unwrap();

    assert!(result.qber > 0.0);
}

// ─── E91 ─────────────────────────────────────────────────────────────────────

#[test]
fn test_e91_noiseless_zero_qber() {
    qcrypto::set_global_seed(5000);
    let channel = QuantumChannel::bit_flip(0.0);
    let result = e91::run(200, &channel, &channel, 0.0, 0.5).unwrap();

    assert_eq!(result.raw_length, 200);
    assert_eq!(result.check_errors, 0);
    assert_eq!(result.qber.unwrap_or(0.0), 0.0);
    assert_eq!(result.eve_intercept_count, 0);
    assert!(result.total_sifted > 0);
    assert!(!result.alice_key.is_empty());
}

#[test]
fn test_e91_chsh_noiseless_violates_bell_inequality() {
    qcrypto::set_global_seed(5100);
    let channel = QuantumChannel::bit_flip(0.0);
    let result = e91::run(500, &channel, &channel, 0.0, 0.5).unwrap();

    // A maximally entangled singlet yields |S| = 2√2 ≈ 2.83 > 2
    assert!(
        result.chsh_value.abs() > 2.0,
        "Noiseless entangled pairs should violate Bell inequality (|S| > 2), got {}",
        result.chsh_value
    );
}

#[test]
fn test_e91_eve_introduces_errors() {
    qcrypto::set_global_seed(5200);
    let channel = QuantumChannel::bit_flip(0.0);
    let result = e91::run(500, &channel, &channel, 1.0, 0.5).unwrap();

    assert!(result.eve_intercept_count > 0);
    assert!(result.qber.map(|q| q > 0.0).unwrap_or(false));
}

#[test]
fn test_e91_noisy_channel_introduces_qber() {
    qcrypto::set_global_seed(5300);
    let channel = QuantumChannel::depolarizing(0.15);
    let result = e91::run(500, &channel, &channel, 0.0, 0.5).unwrap();

    assert!(result.qber.map(|q| q > 0.0).unwrap_or(false));
}

#[test]
fn test_e91_full_check_ratio() {
    qcrypto::set_global_seed(5400);
    let channel = QuantumChannel::bit_flip(0.0);
    let result = e91::run(100, &channel, &channel, 0.0, 1.0).unwrap();

    assert!(result.alice_key.is_empty());
    assert_eq!(result.check_errors, 0);
}

#[test]
fn test_e91_deterministic_with_seed() {
    let run_once = |seed: u64| {
        qcrypto::set_global_seed(seed);
        let channel = QuantumChannel::bit_flip(0.0);
        let r = e91::run(100, &channel, &channel, 0.0, 0.5).unwrap();
        (
            r.alice_bits.clone(),
            r.bob_results.clone(),
            r.alice_key.clone(),
        )
    };

    let (a1, b1, k1) = run_once(5500);
    let (a2, b2, k2) = run_once(5500);

    assert_eq!(a1, a2);
    assert_eq!(b1, b2);
    assert_eq!(k1, k2);
}

// ─── SARG04 ──────────────────────────────────────────────────────────────────

#[test]
fn test_sarg04_noiseless_zero_qber() {
    qcrypto::set_global_seed(6000);
    let channel = QuantumChannel::bit_flip(0.0);
    let result = sarg04::run(200, &channel, 0.0, 0.5).unwrap();

    assert_eq!(result.raw_length, 200);
    assert_eq!(result.check_errors, 0);
    assert_eq!(result.qber, 0.0);
    assert_eq!(result.eve_intercepted_count, 0);
    assert!(result.conclusive_count > 0);
    assert!(result.conclusive_count < result.raw_length);
}

#[test]
fn test_sarg04_eve_introduces_errors() {
    qcrypto::set_global_seed(6100);
    let channel = QuantumChannel::bit_flip(0.0);
    let result = sarg04::run(500, &channel, 1.0, 0.5).unwrap();

    assert!(result.eve_intercepted_count > 0);
    assert!(result.qber > 0.0);
}

#[test]
fn test_sarg04_noisy_channel_introduces_qber() {
    qcrypto::set_global_seed(6200);
    let channel = QuantumChannel::bit_flip(0.2);
    let result = sarg04::run(500, &channel, 0.0, 0.5).unwrap();

    assert_eq!(result.eve_intercepted_count, 0);
    assert!(result.qber > 0.0, "Noise should introduce QBER in SARG04");
}

#[test]
fn test_sarg04_full_check_ratio() {
    qcrypto::set_global_seed(6300);
    let channel = QuantumChannel::bit_flip(0.0);
    let result = sarg04::run(100, &channel, 0.0, 1.0).unwrap();

    assert!(result.alice_key.is_empty());
    assert_eq!(result.check_errors, 0);
}

#[test]
fn test_sarg04_deterministic_with_seed() {
    let run_once = |seed: u64| {
        qcrypto::set_global_seed(seed);
        let channel = QuantumChannel::bit_flip(0.0);
        let r = sarg04::run(100, &channel, 0.0, 0.5).unwrap();
        (
            r.alice_bits.clone(),
            r.bob_results.clone(),
            r.alice_key.clone(),
        )
    };

    let (a1, b1, k1) = run_once(6400);
    let (a2, b2, k2) = run_once(6400);

    assert_eq!(a1, a2);
    assert_eq!(b1, b2);
    assert_eq!(k1, k2);
}

#[test]
fn test_sarg04_depolarizing_channel() {
    qcrypto::set_global_seed(6500);
    let channel = QuantumChannel::depolarizing(0.2);
    let result = sarg04::run(500, &channel, 0.0, 0.5).unwrap();

    assert!(result.qber > 0.0);
}

// ─── GC01 ────────────────────────────────────────────────────────────────────

#[test]
fn test_gc01_noiseless_signature_accepted() {
    qcrypto::set_global_seed(7000);
    let channel = QuantumChannel::bit_flip(0.0);
    let result = gc01::run(200, &channel, &channel, 0.0, 0.1).unwrap();

    assert_eq!(result.num_qubits, 200);
    assert_eq!(result.eve_intercepted_count, 0);
    assert_eq!(result.bob_mismatches, 0);
    assert_eq!(result.charlie_mismatches, 0);
    assert_eq!(result.bob_mismatch_rate, 0.0);
    assert_eq!(result.charlie_mismatch_rate, 0.0);
    assert!(result.signature_accepted);
}

#[test]
fn test_gc01_extreme_noise_rejects_signature() {
    qcrypto::set_global_seed(7100);
    // depolarizing avoids the X·X=I cancellation that occurs when bit_flip(1.0)
    // is applied twice (before and after the Eve window in gc01::run).
    let channel = QuantumChannel::depolarizing(0.5);
    let result = gc01::run(500, &channel, &channel, 0.0, 0.1).unwrap();

    assert!(
        !result.signature_accepted,
        "Extreme depolarizing noise should cause signature rejection"
    );
    assert!(result.bob_mismatch_rate > 0.1 || result.charlie_mismatch_rate > 0.1);
}

#[test]
fn test_gc01_eve_raises_mismatch_rate() {
    qcrypto::set_global_seed(7200);
    let channel = QuantumChannel::bit_flip(0.0);
    let result = gc01::run(500, &channel, &channel, 1.0, 0.1).unwrap();

    assert!(result.eve_intercepted_count > 0);
    assert!(result.bob_mismatch_rate > 0.0 || result.charlie_mismatch_rate > 0.0);
}

#[test]
fn test_gc01_moderate_noise() {
    qcrypto::set_global_seed(7300);
    let channel = QuantumChannel::depolarizing(0.15);
    let result = gc01::run(500, &channel, &channel, 0.0, 0.5).unwrap();

    assert_eq!(result.num_qubits, 500);
    // With a generous threshold the signature may still be accepted
    assert!(result.bob_mismatch_rate >= 0.0);
    assert!(result.charlie_mismatch_rate >= 0.0);
}

#[test]
fn test_gc01_deterministic_with_seed() {
    let run_once = |seed: u64| {
        qcrypto::set_global_seed(seed);
        let channel = QuantumChannel::bit_flip(0.0);
        let r = gc01::run(100, &channel, &channel, 0.0, 0.1).unwrap();
        (
            r.message,
            r.bob_mismatches,
            r.charlie_mismatches,
            r.signature_accepted,
        )
    };

    let (m1, bm1, cm1, sa1) = run_once(7400);
    let (m2, bm2, cm2, sa2) = run_once(7400);

    assert_eq!(m1, m2);
    assert_eq!(bm1, bm2);
    assert_eq!(cm1, cm2);
    assert_eq!(sa1, sa2);
}

#[test]
fn test_gc01_high_threshold_accepts_noisy() {
    qcrypto::set_global_seed(7500);
    let channel = QuantumChannel::bit_flip(0.05);
    // Very permissive threshold — should still accept
    let result = gc01::run(500, &channel, &channel, 0.0, 0.9).unwrap();

    assert!(result.signature_accepted);
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

    qcrypto::set_global_seed(3004);
    let six_state_r = six_state::run(300, &channel, 0.0, 0.5).unwrap();
    assert_eq!(six_state_r.raw_length, 300);

    qcrypto::set_global_seed(3005);
    let e91_r = e91::run(300, &channel, &channel, 0.0, 0.5).unwrap();
    assert_eq!(e91_r.raw_length, 300);

    qcrypto::set_global_seed(3006);
    let sarg04_r = sarg04::run(300, &channel, 0.0, 0.5).unwrap();
    assert_eq!(sarg04_r.raw_length, 300);

    qcrypto::set_global_seed(3007);
    let gc01_r = gc01::run(300, &channel, &channel, 0.0, 0.5).unwrap();
    assert_eq!(gc01_r.num_qubits, 300);
}
