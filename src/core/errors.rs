use num_complex::Complex64;
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

    #[error("Duplicate qubit index found: {0}")]
    DuplicateQubit(usize),
}

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

    #[error("Measurement error: {0}")]
    MeasurementError(#[from] MeasurementError),

    #[error("Gate error: {0}")]
    GateError(#[from] GateError),

    #[error("Channel error: {0}")]
    ChannelError(#[from] ChannelError),
}

#[derive(Error, Debug, Clone)]
pub enum ChannelError {
    #[error("Channel must have at least one Kraus operator")]
    Empty,

    #[error("Kraus operators do not sum to Identity (Trace preserving relation failed)")]
    NotComplete,

    #[error("Invalid operator dimensions: Matrices must be square and 2^n")]
    InvalidDimensions,

    #[error("Dimension mismatch: All Kraus operators must have the same size")]
    OperatorSizeMismatch,

    #[error("Invalid probability: {0}. Must be between 0.0 and 1.0")]
    InvalidProbability(f64),

    #[error("Duplicate qubit index found: {0}")]
    DuplicateQubit(usize),
}
