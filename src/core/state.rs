use crate::core::Gate;
use crate::core::gates::GateError;
use crate::core::utils::trace;
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use thiserror::Error;

/// Error type for QuantumState operations
#[derive(Error, Debug, Clone)]
pub enum StateError {
    #[error("Trace is not unity: {0}")]
    InvalidTrace(Complex64),

    #[error("Vector is not normalized. Norm squared: {0}")]
    NotNormalized(f64),

    #[error("Invalid dimensions")]
    InvalidDimensions,

    #[error("Dimension mismatch")]
    DimensionMismatch {
        expected: usize,
        got_rows: usize,
        got_cols: usize,
    },

    #[error("Qubit index out of bounds")]
    IndexOutOfBounds { index: usize, num_qubits: usize },

    #[error("Gate error: {0}")]
    GateError(#[from] GateError),
}

#[derive(Clone, Debug)]
pub struct QuantumState {
    pub density_matrix: Array2<Complex64>,
    pub num_qubits: usize,
}

impl QuantumState {
    /// Creates a new quantum state initialized to |0...0>.
    pub fn new(num_qubits: usize) -> Self {
        let dim = 1 << num_qubits;
        let mut density_matrix = Array2::<Complex64>::zeros((dim, dim));
        density_matrix[[0, 0]] = Complex64::new(1.0, 0.0);

        Self {
            density_matrix,
            num_qubits,
        }
    }

    /// Validates that the input vector is a valid quantum state.
    fn check_vector_state(vector: &Array1<Complex64>) -> Result<(), StateError> {
        let dim = vector.len();

        // Dimension must be a power of 2
        if !dim.is_power_of_two() {
            return Err(StateError::InvalidDimensions);
        }

        // Sum of squared amplitudes must be 1.
        let norm_sqr: f64 = vector.iter().map(|c| c.norm_sqr()).sum();

        if (norm_sqr - 1.0).abs() > 1e-12 {
            return Err(StateError::NotNormalized(norm_sqr));
        }

        Ok(())
    }

    /// Checks the validity of a density matrix
    fn check_density_matrix(matrix: &Array2<Complex64>) -> Result<(), StateError> {
        let (rows, cols) = matrix.dim();

        if rows != cols {
            return Err(StateError::DimensionMismatch {
                expected: rows,
                got_rows: rows,
                got_cols: cols,
            });
        }
        if !rows.is_power_of_two() {
            return Err(StateError::InvalidDimensions);
        }

        let tr = trace(matrix);
        if (tr - Complex64::new(1.0, 0.0)).norm() > 1e-12 {
            return Err(StateError::InvalidTrace(tr));
        }

        Ok(())
    }

    /// Apply already extended operator ot whole system
    fn apply_operator(&mut self, u: &Array2<Complex64>) -> Result<(), StateError> {
        let (rows, cols) = u.dim();
        let dim = 1 << self.num_qubits;

        if rows != dim || cols != dim {
            return Err(StateError::DimensionMismatch {
                expected: dim,
                got_rows: rows,
                got_cols: cols,
            });
        }

        let temp = u.dot(&self.density_matrix);

        let u_dagger = u.t().mapv(|x| x.conj());
        self.density_matrix = temp.dot(&u_dagger);

        Ok(())
    }

    /// Checks if a given index is within the system QuantumState's range
    fn validate_qubit_index(&self, index: usize) -> Result<(), StateError> {
        if index >= self.num_qubits {
            return Err(StateError::IndexOutOfBounds {
                index,
                num_qubits: self.num_qubits,
            });
        }
        Ok(())
    }

    /// Creates a QuantumState from a generic vector state.
    pub fn from_state_vector(vector: Array1<Complex64>) -> Result<Self, StateError> {
        Self::check_vector_state(&vector)?;

        // Calculate number of qubits: dim = 2^n, so n = log2(dim)
        let dim = vector.len();
        let num_qubits = (dim as f64).log2() as usize;

        // Compute the density matrix of the pure state: rho = |psi><psi|
        let col_vector = vector.view().into_shape_with_order((dim, 1)).unwrap();
        let row_vector_owned = col_vector.mapv(|c| c.conj());
        let matrix = col_vector.dot(&row_vector_owned.t());

        Ok(Self {
            density_matrix: matrix,
            num_qubits,
        })
    }

    /// Creates a QuantumState from a generic density matrix.
    pub fn from_density_matrix(matrix: Array2<Complex64>) -> Result<Self, StateError> {
        Self::check_density_matrix(&matrix)?;
        let (rows, _) = matrix.dim();
        let num_qubits = rows.trailing_zeros() as usize;

        Ok(Self {
            density_matrix: matrix,
            num_qubits,
        })
    }

    /// Checks if a QuantumState is valid.
    pub fn is_valid(&self) -> Result<(), StateError> {
        Self::check_density_matrix(&self.density_matrix)?;
        Ok(())
    }

    /// Applies non controlled quantum gate
    pub fn apply(&mut self, gate: &Gate, target_qubits: &[usize]) -> Result<(), StateError> {
        self.apply_controlled(gate, target_qubits, None)
    }

    /// Applies generic quantum gate
    pub fn apply_controlled(
        &mut self,
        gate: &Gate,
        target_qubits: &[usize],
        control_qubits: Option<&[usize]>,
    ) -> Result<(), StateError> {
        if gate.num_qubits != target_qubits.len() {
            return Err(StateError::DimensionMismatch {
                expected: gate.num_qubits,
                got_rows: target_qubits.len(),
                got_cols: 0,
            });
        }

        for &q in target_qubits {
            self.validate_qubit_index(q)?;
        }

        let controls = control_qubits.unwrap_or(&[]);
        for &q in controls {
            self.validate_qubit_index(q)?;
        }

        let full_gate_operator = Gate::expand_gate(self.num_qubits, gate, target_qubits, controls)?;

        self.apply_operator(&full_gate_operator.matrix)
    }
}
