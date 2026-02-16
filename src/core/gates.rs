use crate::core::utils::{find_duplicate, gen_operator};
use ndarray::{Array2, arr2};
use num_complex::Complex64;
use std::f64::consts::PI;
use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum GateError {
    #[error("Matrix is not Unitary (U†U != I)")]
    NonUnitary,

    #[error("Matrix must be square")]
    NotSquareMatrix,

    #[error("Invalid Dimensions")]
    InvalidDimensions,

    #[error("Qubit {0} cannot be both control and target")]
    ControlTargetOverlap(usize),

    #[error("Duplicate qubit index found: {0}")]
    DuplicateQubit(usize),
}

pub struct Gate {
    pub matrix: Array2<Complex64>,
    pub num_qubits: usize,
}

impl Gate {
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

    /// Generates full system gate
    pub fn expand_gate(
        num_total_qubits: usize,
        gate: &Gate,
        targets: &[usize],
        controls: &[usize],
    ) -> Result<Gate, GateError> {
        if let Some(dup) = find_duplicate(targets) {
            return Err(GateError::DuplicateQubit(dup));
        }

        if let Some(dup) = find_duplicate(controls) {
            return Err(GateError::DuplicateQubit(dup));
        }

        for &c in controls {
            if targets.contains(&c) {
                return Err(GateError::ControlTargetOverlap(c));
            }
        }

        Ok(Gate {
            matrix: gen_operator(num_total_qubits, &gate.matrix, targets, controls),
            num_qubits: num_total_qubits,
        })
    }
}

/// Identity Gate
pub fn i() -> Gate {
    Gate::new(arr2(&[
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ]))
    .unwrap()
}

/// Pauli-X Gate
pub fn x() -> Gate {
    Gate::new(arr2(&[
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ]))
    .unwrap()
}

/// Pauli-Y Gate
pub fn y() -> Gate {
    Gate::new(arr2(&[
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
        [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
    ]))
    .unwrap()
}

/// Pauli-Z Gate
pub fn z() -> Gate {
    Gate::new(arr2(&[
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
    ]))
    .unwrap()
}

/// Hadamard Gate
pub fn h() -> Gate {
    let factor = 1.0 / 2.0_f64.sqrt();
    Gate::new(arr2(&[
        [Complex64::new(factor, 0.0), Complex64::new(factor, 0.0)],
        [Complex64::new(factor, 0.0), Complex64::new(-factor, 0.0)],
    ]))
    .unwrap()
}

/// S Gate (Phase Gate, Z^1/2)
pub fn s() -> Gate {
    Gate::new(arr2(&[
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
    ]))
    .unwrap()
}

/// T Gate (Z^1/4)
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

/// CNOT Gate
pub fn cnot() -> Gate {
    Gate::new(arr2(&[
        [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
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
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    ]))
    .unwrap()
}

/// SWAP Gate
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
