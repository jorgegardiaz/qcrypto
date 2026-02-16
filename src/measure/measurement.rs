use crate::core::gates::GateError;
use crate::core::utils;
use ndarray::Array2;
use num_complex::Complex64;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasurementResult {
    /// Applied measurment operator index
    pub index: usize,
    /// Measurement value
    pub value: f64,
}

#[derive(Error, Debug, Clone)]
pub enum MeasurementError {
    #[error("Number of operators ({ops}) does not match number of values ({vals})")]
    CountMismatch { ops: usize, vals: usize },

    #[error("Measurement operators do not sum to Identity (Completeness relation failed)")]
    NotComplete,

    #[error("Invalid operator dimensions")]
    InvalidDimensions,

    #[error("Operator expansion failed: {0}")]
    ExpansionError(#[from] GateError),
}

#[derive(Clone, Debug)]
pub struct Measurement {
    /// List of measurement operators
    pub operators: Vec<Array2<Complex64>>,
    /// Associeted measurment value of the performed measurement
    pub values: Vec<f64>,
    /// Number of qubits which the measurment acts
    pub num_qubits: usize,
}

impl Measurement {
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
        let num_qubits = rows.trailing_zeros() as usize;

        for op in &operators {
            if op.dim() != (rows, cols) {
                return Err(MeasurementError::InvalidDimensions);
            }
        }

        if !Self::check_completeness(&operators, rows) {
            return Err(MeasurementError::NotComplete);
        }

        Ok(Self {
            operators,
            values,
            num_qubits,
        })
    }

    /// Checks completeness realtion for measurment operators
    fn check_completeness(ops: &[Array2<Complex64>], dim: usize) -> bool {
        let mut sum = Array2::<Complex64>::zeros((dim, dim));

        for op in ops {
            let dag = op.t().mapv(|c| c.conj());
            sum = sum + dag.dot(op);
        }

        let eye = Array2::<Complex64>::eye(dim);

        sum.iter()
            .zip(eye.iter())
            .all(|(a, b)| (a - b).norm() < 1e-12)
    }

    /// Expands measurements operator to a larger system
    pub fn get_expanded_operators(
        &self,
        num_total_qubits: usize,
        targets: &[usize],
    ) -> Result<Vec<Array2<Complex64>>, MeasurementError> {
        let mut expanded_ops = Vec::with_capacity(self.operators.len());

        for op in &self.operators {
            let full_op = utils::gen_operator(num_total_qubits, op, targets, &[]);
            expanded_ops.push(full_op);
        }

        Ok(expanded_ops)
    }
}
