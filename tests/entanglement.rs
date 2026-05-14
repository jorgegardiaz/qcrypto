//! Integration tests for multi-qubit entanglement.
//!
//! Bell state creation, entanglement correlations, teleportation circuits,
//! GHZ states, and RNG determinism across complex operations.

use num_complex::Complex64;
use qcrypto::state::StateVector;
use qcrypto::{Gate, Measurement, QuantumState};

// ─── Bell state correlations ─────────────────────────────────────────────────

#[test]
fn test_bell_phi_plus_correlation() {
    // |Φ+> = (|00> + |11>) / √2 — both qubits always agree
    qcrypto::set_global_seed(100);

    for _ in 0..50 {
        let mut state = QuantumState::new(2);
        state.apply(&Gate::h(), &[0]).unwrap();
        state.apply_controlled(&Gate::x(), &[1], &[0]).unwrap();

        let r0 = state.measure(&Measurement::z_basis(), &[0]).unwrap();
        let r1 = state.measure(&Measurement::z_basis(), &[1]).unwrap();

        assert_eq!(r0.value, r1.value, "Bell Φ+ qubits must agree in Z basis");
    }
}

#[test]
fn test_bell_psi_plus_anticorrelation() {
    // |Ψ+> = (|01> + |10>) / √2 — qubits always disagree
    qcrypto::set_global_seed(200);

    for _ in 0..50 {
        let mut state = QuantumState::new(2);
        state.apply(&Gate::x(), &[1]).unwrap();
        state.apply(&Gate::h(), &[0]).unwrap();
        state.apply_controlled(&Gate::x(), &[1], &[0]).unwrap();

        let r0 = state.measure(&Measurement::z_basis(), &[0]).unwrap();
        let r1 = state.measure(&Measurement::z_basis(), &[1]).unwrap();

        assert_ne!(
            r0.value, r1.value,
            "Bell Ψ+ qubits must anti-correlate in Z basis"
        );
    }
}

#[test]
fn test_bell_state_x_basis_correlation() {
    // |Φ+> measured in X basis should also always agree
    qcrypto::set_global_seed(300);

    for _ in 0..50 {
        let mut state = QuantumState::new(2);
        state.apply(&Gate::h(), &[0]).unwrap();
        state.apply_controlled(&Gate::x(), &[1], &[0]).unwrap();

        let r0 = state.measure(&Measurement::x_basis(), &[0]).unwrap();
        let r1 = state.measure(&Measurement::x_basis(), &[1]).unwrap();

        assert_eq!(r0.value, r1.value, "Bell Φ+ qubits must agree in X basis");
    }
}

// ─── GHZ state ───────────────────────────────────────────────────────────────

#[test]
fn test_ghz_three_qubit_correlation() {
    // GHZ = (|000> + |111>) / √2 — all three qubits agree
    qcrypto::set_global_seed(400);

    for _ in 0..50 {
        let mut state = QuantumState::new(3);
        state.apply(&Gate::h(), &[0]).unwrap();
        state.apply_controlled(&Gate::x(), &[1], &[0]).unwrap();
        state.apply_controlled(&Gate::x(), &[2], &[0]).unwrap();

        let r0 = state.measure(&Measurement::z_basis(), &[0]).unwrap();
        let r1 = state.measure(&Measurement::z_basis(), &[1]).unwrap();
        let r2 = state.measure(&Measurement::z_basis(), &[2]).unwrap();

        assert_eq!(r0.value, r1.value, "GHZ qubits 0,1 must agree");
        assert_eq!(r1.value, r2.value, "GHZ qubits 1,2 must agree");
    }
}

#[test]
fn test_ghz_amplitudes() {
    let mut sv = StateVector::new(3);
    sv.apply(&Gate::h(), &[0]).unwrap();
    sv.apply_controlled(&Gate::x(), &[1], &[0]).unwrap();
    sv.apply_controlled(&Gate::x(), &[2], &[0]).unwrap();

    let s = 1.0 / 2.0_f64.sqrt();

    // Only |000> (index 0) and |111> (index 7) should be non-zero
    for (i, amp) in sv.amplitudes.iter().enumerate() {
        let expected = if i == 0 || i == 7 { s } else { 0.0 };
        assert!(
            (amp - Complex64::new(expected, 0.0)).norm() < 1e-12,
            "GHZ amplitude mismatch at |{:03b}>: got {}, expected {}",
            i,
            amp,
            expected,
        );
    }
}

// ─── Teleportation circuit ───────────────────────────────────────────────────

#[test]
fn test_quantum_teleportation() {
    // Teleport an arbitrary state from qubit 0 to qubit 2 using Bell pair on qubits 1,2
    qcrypto::set_global_seed(500);

    // We'll teleport |ψ> = H|0> = |+> and verify the outcome statistically
    // The teleportation circuit:
    //   1. Prepare |ψ> on qubit 0
    //   2. Create Bell pair on qubits 1,2
    //   3. CNOT(0,1), H(0)
    //   4. Measure qubits 0,1
    //   5. Apply corrections on qubit 2

    let num_trials = 100;
    let mut plus_count = 0;

    for _ in 0..num_trials {
        let mut state = QuantumState::new(3);

        // Prepare |ψ> = |+> on qubit 0
        state.apply(&Gate::h(), &[0]).unwrap();

        // Create Bell pair on qubits 1,2
        state.apply(&Gate::h(), &[1]).unwrap();
        state.apply_controlled(&Gate::x(), &[2], &[1]).unwrap();

        // Bell measurement on qubits 0,1
        state.apply_controlled(&Gate::x(), &[1], &[0]).unwrap();
        state.apply(&Gate::h(), &[0]).unwrap();

        let m0 = state.measure(&Measurement::z_basis(), &[0]).unwrap();
        let m1 = state.measure(&Measurement::z_basis(), &[1]).unwrap();

        // Apply corrections
        if m1.index == 1 {
            state.apply(&Gate::x(), &[2]).unwrap();
        }
        if m0.index == 1 {
            state.apply(&Gate::z(), &[2]).unwrap();
        }

        // Qubit 2 should now be in state |+>
        // Measure in X basis to verify
        let result = state.measure(&Measurement::x_basis(), &[2]).unwrap();
        if result.index == 0 {
            plus_count += 1;
        }
    }

    // If teleportation works, qubit 2 is |+>, so X-basis measurement always gives 0
    assert_eq!(
        plus_count, num_trials,
        "Teleported |+> should always give X-basis outcome 0, got {}/{}",
        plus_count, num_trials
    );
}

#[test]
fn test_teleportation_of_one_state() {
    // Teleport |1> and verify
    qcrypto::set_global_seed(600);

    let num_trials = 100;
    let mut one_count = 0;

    for _ in 0..num_trials {
        let mut state = QuantumState::new(3);

        // Prepare |1> on qubit 0
        state.apply(&Gate::x(), &[0]).unwrap();

        // Bell pair on 1,2
        state.apply(&Gate::h(), &[1]).unwrap();
        state.apply_controlled(&Gate::x(), &[2], &[1]).unwrap();

        // Bell measurement
        state.apply_controlled(&Gate::x(), &[1], &[0]).unwrap();
        state.apply(&Gate::h(), &[0]).unwrap();

        let m0 = state.measure(&Measurement::z_basis(), &[0]).unwrap();
        let m1 = state.measure(&Measurement::z_basis(), &[1]).unwrap();

        // Corrections
        if m1.index == 1 {
            state.apply(&Gate::x(), &[2]).unwrap();
        }
        if m0.index == 1 {
            state.apply(&Gate::z(), &[2]).unwrap();
        }

        // Measure in Z basis — should always be |1>
        let result = state.measure(&Measurement::z_basis(), &[2]).unwrap();
        if result.index == 1 {
            one_count += 1;
        }
    }

    assert_eq!(
        one_count, num_trials,
        "Teleported |1> should always give Z-basis outcome 1"
    );
}

// ─── SWAP test ───────────────────────────────────────────────────────────────

#[test]
fn test_swap_gate_exchanges_states() {
    // |10> with SWAP → |01>
    let mut state = QuantumState::new(2);
    state.apply(&Gate::x(), &[0]).unwrap(); // |10>
    state.apply(&Gate::swap(), &[0, 1]).unwrap();

    let sv = state
        .state
        .as_any()
        .downcast_ref::<StateVector>()
        .unwrap();
    // Should be |01> = index 1
    assert!((sv.amplitudes[1] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
}

#[test]
fn test_swap_is_self_inverse() {
    let mut state = QuantumState::new(2);
    state.apply(&Gate::x(), &[0]).unwrap(); // |10>
    state.apply(&Gate::swap(), &[0, 1]).unwrap(); // |01>
    state.apply(&Gate::swap(), &[0, 1]).unwrap(); // back to |10>

    let sv = state
        .state
        .as_any()
        .downcast_ref::<StateVector>()
        .unwrap();
    assert!((sv.amplitudes[2] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
}

// ─── Toffoli gate ────────────────────────────────────────────────────────────

#[test]
fn test_toffoli_only_flips_when_both_controls_set() {
    let inputs: [(bool, bool, bool); 4] = [
        (false, false, false), // |000> → |000>
        (true, false, false),  // |100> → |100>
        (false, true, false),  // |010> → |010>
        (true, true, true),    // |110> → |111>
    ];

    for (q0, q1, expected_q2) in inputs {
        let mut state = QuantumState::new(3);

        if q0 {
            state.apply(&Gate::x(), &[0]).unwrap();
        }
        if q1 {
            state.apply(&Gate::x(), &[1]).unwrap();
        }

        state.apply(&Gate::toffoli(), &[0, 1, 2]).unwrap();

        let probs = state.set_measurement(&Measurement::z_basis(), &[2]).unwrap();
        if expected_q2 {
            assert!(
                (probs[1] - 1.0).abs() < 1e-12,
                "Toffoli({},{}) should flip target",
                q0,
                q1
            );
        } else {
            assert!(
                (probs[0] - 1.0).abs() < 1e-12,
                "Toffoli({},{}) should NOT flip target",
                q0,
                q1
            );
        }
    }
}

// ─── Compose (tensor product) ────────────────────────────────────────────────

#[test]
fn test_compose_product_state() {
    let mut s1 = QuantumState::new(1);
    s1.apply(&Gate::x(), &[0]).unwrap(); // |1>

    let s2 = QuantumState::new(1); // |0>
    let combined = s1.compose(&s2).unwrap(); // |10>

    let sv = combined
        .state
        .as_any()
        .downcast_ref::<StateVector>()
        .unwrap();

    // |10> = index 2 in big-endian
    assert!((sv.amplitudes[2] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
}

#[test]
fn test_compose_then_entangle() {
    let s1 = QuantumState::new(1); // |0>
    let s2 = QuantumState::new(1); // |0>
    let mut combined = s1.compose(&s2).unwrap(); // |00>

    // Create Bell state on the composed system
    combined.apply(&Gate::h(), &[0]).unwrap();
    combined.apply_controlled(&Gate::x(), &[1], &[0]).unwrap();

    let probs = combined
        .set_measurement(&Measurement::bell_basis(), &[0, 1])
        .unwrap();
    // Should be 100% Φ+
    assert!((probs[0] - 1.0).abs() < 1e-12);
}

// ─── Seeded determinism across entangled measurements ────────────────────────

#[test]
fn test_seeded_bell_measurements_reproducible() {
    let run_once = |seed: u64| {
        qcrypto::set_global_seed(seed);
        let mut results = Vec::new();
        for _ in 0..20 {
            let mut state = QuantumState::new(2);
            state.apply(&Gate::h(), &[0]).unwrap();
            state.apply_controlled(&Gate::x(), &[1], &[0]).unwrap();
            let r = state.measure(&Measurement::z_basis(), &[0]).unwrap();
            results.push(r.index);
        }
        results
    };

    let r1 = run_once(42);
    let r2 = run_once(42);

    assert_eq!(r1, r2, "Seeded Bell measurements must be reproducible");
}

// ─── Purity of entangled subsystems ──────────────────────────────────────────

#[test]
fn test_entangled_state_has_unit_purity() {
    // The full 2-qubit Bell state is pure
    let mut state = QuantumState::new(2);
    state.apply(&Gate::h(), &[0]).unwrap();
    state.apply_controlled(&Gate::x(), &[1], &[0]).unwrap();

    assert!((state.purity() - 1.0).abs() < 1e-12, "Bell state should be pure");
}
