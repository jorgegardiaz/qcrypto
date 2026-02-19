use crate::core::errors::GateError;
use crate::core::utils;
use ndarray::{Array2, arr2};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Represents a quantum gate.
///
/// A gate is defined by its unitary matrix and the number of qubits it acts on.
pub struct Gate {
    /// The unitary matrix of the gate.
    pub matrix: Array2<Complex64>,
    /// The number of qubits the gate acts on.
    pub num_qubits: usize,
}

impl Gate {
    /// Creates a new `Gate` from a unitary matrix.
    ///
    /// # Arguments
    ///
    /// * `matrix` - A square, unitary `Array2<Complex64>`.
    ///
    /// # Errors
    ///
    /// Returns a `GateError` if:
    /// - The matrix is not square.
    /// - The matrix dimensions are not a power of 2.
    /// - The matrix is not unitary.
    pub fn new(matrix: Array2<Complex64>) -> Result<Self, GateError> {
        let (rows, cols) = matrix.dim();

        if rows != cols {
            return Err(GateError::NotSquareMatrix);
        }

        if !rows.is_power_of_two() {
            return Err(GateError::InvalidDimensions);
        }

        if !Self::check_unitary(&matrix) {
            return Err(GateError::NonUnitary);
        }

        let num_qubits = rows.trailing_zeros() as usize;

        Ok(Self { matrix, num_qubits })
    }

    /// Checks if a given matrix is unitary
    fn check_unitary(matrix: &Array2<Complex64>) -> bool {
        let (rows, _) = matrix.dim();
        let eye = Array2::<Complex64>::eye(rows);

        let u_dagger = matrix.t().mapv(|x| x.conj());
        let product = matrix.dot(&u_dagger);

        product
            .iter()
            .zip(eye.iter())
            .all(|(a, b)| (*a - *b).norm() < 1e-6)
    }

    /// Expands a gate to act on a larger system of qubits.
    ///
    /// This function creates a new gate that acts on `num_total_qubits` by applying the original `gate`
    /// to the specified `targets` and `controls` (if any), and Identity on the rest.
    ///
    /// # Arguments
    ///
    /// * `num_total_qubits` - The total number of qubits in the system.
    /// * `gate` - The base gate to expand.
    /// * `targets` - Indices of the target qubits.
    /// * `controls` - Indices of the control qubits.
    ///
    /// # Errors
    ///
    /// Returns `GateError` if:
    /// - Duplicate indices are found in `targets` or `controls`.
    /// - A qubit is used as both control and target.
    pub fn expand_gate(
        num_total_qubits: usize,
        gate: &Gate,
        targets: &[usize],
        controls: &[usize],
    ) -> Result<Gate, GateError> {
        if let Some(dup) = utils::find_duplicate(targets) {
            return Err(GateError::DuplicateQubit(dup));
        }

        if let Some(dup) = utils::find_duplicate(controls) {
            return Err(GateError::DuplicateQubit(dup));
        }

        for &c in controls {
            if targets.contains(&c) {
                return Err(GateError::ControlTargetOverlap(c));
            }
        }

        Ok(Gate {
            matrix: utils::expand_operator(num_total_qubits, &gate.matrix, targets, controls),
            num_qubits: num_total_qubits,
        })
    }

    // --- Standard Gates ---

    /// Creates an Identity gate.
    pub fn i() -> Gate {
        Gate::new(arr2(&[
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ]))
        .unwrap()
    }

    /// Creates a Pauli-X gate (NOT gate).
    pub fn x() -> Gate {
        Gate::new(arr2(&[
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ]))
        .unwrap()
    }

    /// Creates a Pauli-Y gate.
    pub fn y() -> Gate {
        Gate::new(arr2(&[
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
            [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
        ]))
        .unwrap()
    }

    /// Creates a Pauli-Z gate.
    pub fn z() -> Gate {
        Gate::new(arr2(&[
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
        ]))
        .unwrap()
    }

    /// Creates a Hadamard gate.
    pub fn h() -> Gate {
        let factor = 1.0 / 2.0_f64.sqrt();
        Gate::new(arr2(&[
            [Complex64::new(factor, 0.0), Complex64::new(factor, 0.0)],
            [Complex64::new(factor, 0.0), Complex64::new(-factor, 0.0)],
        ]))
        .unwrap()
    }

    /// Creates an S gate (Phase gate, Z^1/2).
    pub fn s() -> Gate {
        Gate::new(arr2(&[
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
        ]))
        .unwrap()
    }

    /// Creates a T gate (Z^1/4).
    pub fn t_gate() -> Gate {
        let angle = PI / 4.0;
        Gate::new(arr2(&[
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(angle.cos(), angle.sin()),
            ],
        ]))
        .unwrap()
    }

    /// Creates a CNOT (Controlled-NOT) gate.
    pub fn cnot() -> Gate {
        Gate::expand_gate(2, &Gate::x(), &[1], &[0]).unwrap()
    }

    /// Creates a SWAP gate.
    pub fn swap() -> Gate {
        Gate::new(arr2(&[
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        ]))
        .unwrap()
    }

    /// Creates a Toffoli gate.
    pub fn toffoli() -> Gate {
        Gate::expand_gate(3, &Gate::x(), &[2], &[0, 1]).unwrap()
    }
}
