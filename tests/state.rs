//! Integration tests verifying that StateVector and StateDensityMatrix
//! produce equivalent results for the same quantum operations.

use num_complex::Complex64;
use qcrypto::state::{StateDensityMatrix, StateVector};
use qcrypto::{Gate, Measurement};

/// Helper: extract probabilities from a StateVector via set_measurement.
fn sv_probs(sv: &StateVector, m: &Measurement, targets: &[usize]) -> Vec<f64> {
    sv.set_measurement(m, targets).unwrap()
}

/// Helper: extract probabilities from a StateDensityMatrix via set_measurement.
fn dm_probs(dm: &StateDensityMatrix, m: &Measurement, targets: &[usize]) -> Vec<f64> {
    dm.set_measurement(m, targets).unwrap()
}

/// Helper: assert two probability vectors are approximately equal.
fn assert_probs_eq(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "Probability vectors differ in length");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < tol,
            "Probability mismatch at index {}: sv={}, dm={}",
            i,
            x,
            y
        );
    }
}

#[test]
fn test_single_qubit_gates_equivalence() {
    let gates = [
        Gate::x(),
        Gate::y(),
        Gate::z(),
        Gate::h(),
        Gate::s(),
        Gate::t_gate(),
    ];

    for gate in &gates {
        let mut sv = StateVector::new(1);
        let mut dm = StateDensityMatrix::new(1);

        sv.apply(gate, &[0]).unwrap();
        dm.apply(gate, &[0]).unwrap();

        let probs_sv = sv_probs(&sv, &Measurement::z_basis(), &[0]);
        let probs_dm = dm_probs(&dm, &Measurement::z_basis(), &[0]);
        assert_probs_eq(&probs_sv, &probs_dm, 1e-12);
    }
}

#[test]
fn test_bell_state_equivalence() {
    let mut sv = StateVector::new(2);
    let mut dm = StateDensityMatrix::new(2);

    sv.apply(&Gate::h(), &[0]).unwrap();
    sv.apply_controlled(&Gate::x(), &[1], &[0]).unwrap();

    dm.apply(&Gate::h(), &[0]).unwrap();
    dm.apply_controlled(&Gate::x(), &[1], &[0]).unwrap();

    let probs_sv = sv_probs(&sv, &Measurement::bell_basis(), &[0, 1]);
    let probs_dm = dm_probs(&dm, &Measurement::bell_basis(), &[0, 1]);
    assert_probs_eq(&probs_sv, &probs_dm, 1e-12);

    // Should be 100% Phi+
    assert!((probs_sv[0] - 1.0).abs() < 1e-12);
}

#[test]
fn test_multi_gate_sequence_equivalence() {
    // Apply a complex sequence of gates and verify both representations agree
    let mut sv = StateVector::new(2);
    let mut dm = StateDensityMatrix::new(2);

    let ops: Vec<(Gate, Vec<usize>)> = vec![
        (Gate::h(), vec![0]),
        (Gate::cnot(), vec![0, 1]),
        (Gate::z(), vec![1]),
        (Gate::h(), vec![1]),
        (Gate::swap(), vec![0, 1]),
    ];

    for (gate, targets) in &ops {
        sv.apply(gate, targets).unwrap();
        dm.apply(gate, targets).unwrap();
    }

    // Check Z-basis probabilities on both qubits
    for q in 0..2 {
        let probs_sv = sv_probs(&sv, &Measurement::z_basis(), &[q]);
        let probs_dm = dm_probs(&dm, &Measurement::z_basis(), &[q]);
        assert_probs_eq(&probs_sv, &probs_dm, 1e-12);
    }
}

#[test]
fn test_purity_equivalence() {
    let mut sv = StateVector::new(2);
    let mut dm = StateDensityMatrix::new(2);

    sv.apply(&Gate::h(), &[0]).unwrap();
    dm.apply(&Gate::h(), &[0]).unwrap();

    // Both should be pure
    assert!((sv.purity() - 1.0).abs() < 1e-12);
    assert!((dm.purity() - 1.0).abs() < 1e-12);
}

#[test]
fn test_compose_equivalence() {
    let mut sv1 = StateVector::new(1);
    let mut dm1 = StateDensityMatrix::new(1);

    sv1.apply(&Gate::h(), &[0]).unwrap();
    dm1.apply(&Gate::h(), &[0]).unwrap();

    let sv2 = StateVector::new(1);
    let dm2 = StateDensityMatrix::new(1);

    let sv_combined = sv1.compose(&sv2).unwrap();
    let dm_combined = dm1.compose(&dm2).unwrap();

    assert_eq!(sv_combined.num_qubits, 2);
    assert_eq!(dm_combined.num_qubits, 2);

    // Verify the density matrix from the composed vector matches the composed density matrix
    let dm_from_sv = StateDensityMatrix::from_state_vector(sv_combined.amplitudes.clone()).unwrap();

    for ((i, j), &val) in dm_from_sv.density_matrix.indexed_iter() {
        let diff = (val - dm_combined.density_matrix[[i, j]]).norm();
        assert!(diff < 1e-12, "Compose mismatch at [{}, {}]", i, j);
    }
}

#[test]
fn test_measurement_in_all_bases_equivalence() {
    // Prepare |+i> = S * H|0> and check all three bases
    let mut sv = StateVector::new(1);
    let mut dm = StateDensityMatrix::new(1);

    sv.apply(&Gate::h(), &[0]).unwrap();
    sv.apply(&Gate::s(), &[0]).unwrap();

    dm.apply(&Gate::h(), &[0]).unwrap();
    dm.apply(&Gate::s(), &[0]).unwrap();

    let bases = [
        Measurement::z_basis(),
        Measurement::x_basis(),
        Measurement::y_basis(),
    ];

    for basis in &bases {
        let probs_sv = sv_probs(&sv, basis, &[0]);
        let probs_dm = dm_probs(&dm, basis, &[0]);
        assert_probs_eq(&probs_sv, &probs_dm, 1e-12);
    }
}

#[test]
fn test_three_qubit_toffoli_equivalence() {
    let mut sv = StateVector::new(3);
    let mut dm = StateDensityMatrix::new(3);

    // Set |110> by applying X to qubits 0 and 1
    sv.apply(&Gate::x(), &[0]).unwrap();
    sv.apply(&Gate::x(), &[1]).unwrap();
    dm.apply(&Gate::x(), &[0]).unwrap();
    dm.apply(&Gate::x(), &[1]).unwrap();

    // Apply Toffoli: controls 0,1 target 2 -> should flip qubit 2
    sv.apply(&Gate::toffoli(), &[0, 1, 2]).unwrap();
    dm.apply(&Gate::toffoli(), &[0, 1, 2]).unwrap();

    // State should be |111>
    // Check qubit 2 in Z basis: should be 100% |1>
    let probs_sv = sv_probs(&sv, &Measurement::z_basis(), &[2]);
    let probs_dm = dm_probs(&dm, &Measurement::z_basis(), &[2]);
    assert_probs_eq(&probs_sv, &probs_dm, 1e-12);
    assert!((probs_sv[1] - 1.0).abs() < 1e-12, "Qubit 2 should be |1>");
}

#[test]
fn test_as_density_matrix_conversion() {
    // Verify that converting a StateVector to StateDensityMatrix preserves the state
    let mut sv = StateVector::new(2);
    sv.apply(&Gate::h(), &[0]).unwrap();
    sv.apply_controlled(&Gate::x(), &[1], &[0]).unwrap();

    let dm = StateDensityMatrix::from_state_vector(sv.amplitudes.clone()).unwrap();

    // The density matrix should be |Φ+><Φ+|
    let expected_entries = [((0, 0), 0.5), ((0, 3), 0.5), ((3, 0), 0.5), ((3, 3), 0.5)];

    for &((i, j), expected) in &expected_entries {
        assert!(
            (dm.density_matrix[[i, j]] - Complex64::new(expected, 0.0)).norm() < 1e-12,
            "Conversion mismatch at [{}, {}]",
            i,
            j,
        );
    }
}
