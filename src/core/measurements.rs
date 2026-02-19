use crate::core::errors::MeasurementError;
use crate::core::utils;
use ndarray::{Array1, Array2, array};
use num_complex::Complex64;

/// Represents a general quantum measurement.
///
/// A measurement is defined by a set of operators $\{M_k\}$ such that $\sum M_k^\dagger M_k = I$.
/// It also associates a real value with each measurement outcome.
#[derive(Clone, Debug)]
pub struct Measurement {
    /// List of measurement operators (Kraus operators).
    pub operators: Vec<Array2<Complex64>>,
    /// Associated measurement values for each outcome.
    pub values: Vec<f64>,
    /// Number of qubits the measurement acts on.
    pub num_qubits: usize,
}

impl Measurement {
    /// Creates a new `Measurement` from a set of operators and values.
    ///
    /// # Arguments
    ///
    /// * `operators` - A vector of `Array2<Complex64>` representing the measurement operators.
    /// * `values` - A vector of `f64` values corresponding to the output of each operator.
    ///
    /// # Errors
    ///
    /// Returns `MeasurementError` if:
    /// - The number of operators and values do not match.
    /// - The operators are not of correct dimensions.
    /// - The operators do not satisfy the completeness relation ($\sum M_k^\dagger M_k = I$).
    pub fn new(
        operators: Vec<Array2<Complex64>>,
        values: Vec<f64>,
    ) -> Result<Self, MeasurementError> {
        if operators.len() != values.len() {
            return Err(MeasurementError::CountMismatch {
                ops: operators.len(),
                vals: values.len(),
            });
        }

        if operators.is_empty() {
            return Err(MeasurementError::InvalidDimensions);
        }

        let (rows, cols) = operators[0].dim();
        if rows != cols || !rows.is_power_of_two() {
            return Err(MeasurementError::InvalidDimensions);
        }
        // log_2 as rows is power of two
        let num_qubits = rows.trailing_zeros() as usize;

        for op in &operators {
            if op.dim() != (rows, cols) {
                return Err(MeasurementError::InvalidDimensions);
            }
        }

        if !utils::check_kraus_completeness(&operators, rows) {
            return Err(MeasurementError::NotComplete);
        }

        Ok(Self {
            operators,
            values,
            num_qubits,
        })
    }

    /// Creates a valid Measurement from a given POVM (Positive Operator-Valued Measure).
    ///
    /// # Arguments
    ///
    /// * `povm_elements` - A vector of POVM elements $\{E_k\}$ where each $E_k$ is positive semi-definite and $\sum E_k = I$.
    /// * `values` - A vector of values associated with each POVM element.
    ///
    /// # Errors
    ///
    /// Returns `MeasurementError` if:
    /// - The elements dimensions are invalid or mismatched.
    /// - The elements do not sum to Identity.
    pub fn from_povm(
        povm_elements: Vec<Array2<Complex64>>,
        values: Vec<f64>,
    ) -> Result<Self, MeasurementError> {
        if povm_elements.len() != values.len() {
            return Err(MeasurementError::CountMismatch {
                ops: povm_elements.len(),
                vals: values.len(),
            });
        }

        if povm_elements.is_empty() {
            return Err(MeasurementError::InvalidDimensions);
        }

        let (rows, cols) = povm_elements[0].dim();

        if rows != cols || !rows.is_power_of_two() {
            return Err(MeasurementError::InvalidDimensions);
        }

        // log_2
        let num_qubits = rows.trailing_zeros() as usize;

        if !utils::check_povm_completeness(&povm_elements, rows) {
            return Err(MeasurementError::NotComplete);
        }

        let kraus_ops = povm_elements
            .iter()
            .map(utils::sqrt_positive_matrix)
            .collect();

        Ok(Measurement {
            operators: kraus_ops,
            values,
            num_qubits,
        })
    }

    /// Expands the measurement operators to act on a larger system.
    ///
    /// # Arguments
    ///
    /// * `num_total_qubits` - The size of the full system.
    /// * `targets` - The indices of the qubits this measurement applies to.
    pub fn get_expanded_operators(
        &self,
        num_total_qubits: usize,
        targets: &[usize],
    ) -> Result<Vec<Array2<Complex64>>, MeasurementError> {
        if targets.len() != self.num_qubits {
            return Err(MeasurementError::InvalidDimensions); // O crea un error TargetMismatch
        }

        let mut expanded_ops = Vec::with_capacity(self.operators.len());

        for op in &self.operators {
            let full_op = utils::expand_operator(num_total_qubits, op, targets, &[]);
            expanded_ops.push(full_op);
        }

        Ok(expanded_ops)
    }

    /// Creates a measurement in the Z basis (Computational basis) -> {|0>, |1>}.
    pub fn z_basis() -> Measurement {
        let v0: Array1<Complex64> = array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let v1: Array1<Complex64> = array![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];

        let p0 = utils::outer_product(&v0, &v0);
        let p1 = utils::outer_product(&v1, &v1);

        Measurement::new(vec![p0, p1], vec![0.0, 1.0]).expect("Error in basis Z")
    }

    /// Creates a measurement in the X basis (Hadamard basis) -> {|+>, |->}.
    pub fn x_basis() -> Measurement {
        let inv_sqrt2 = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);

        let v_plus: Array1<Complex64> = array![inv_sqrt2, inv_sqrt2];
        let v_minus: Array1<Complex64> = array![inv_sqrt2, -inv_sqrt2];

        let p_plus = utils::outer_product(&v_plus, &v_plus);
        let p_minus = utils::outer_product(&v_minus, &v_minus);

        Measurement::new(vec![p_plus, p_minus], vec![0.0, 1.0]).expect("Error in basis X")
    }

    /// Creates a measurement in the Y basis -> {|+i>, |-i>}.
    pub fn y_basis() -> Measurement {
        let inv_sqrt2 = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        let i_inv_sqrt2 = Complex64::new(0.0, 1.0 / 2.0_f64.sqrt());

        let v_plus_i: Array1<Complex64> = array![inv_sqrt2, i_inv_sqrt2];
        let v_minus_i: Array1<Complex64> = array![inv_sqrt2, -i_inv_sqrt2];

        let p_plus_i = utils::outer_product(&v_plus_i, &v_plus_i);
        let p_minus_i = utils::outer_product(&v_minus_i, &v_minus_i);

        Measurement::new(vec![p_plus_i, p_minus_i], vec![0.0, 1.0]).expect("Error in basis Y")
    }
}

/// The result of a quantum measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasurementResult {
    /// The index of the outcome (and operator) that occurred.
    pub index: usize,
    /// The value associated with the outcome.
    pub value: f64,
}
