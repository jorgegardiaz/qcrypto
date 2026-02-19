//! Error types for the core module.
//!
//! This module defines custom error types for gates, measurements, states, and channels.
//! It uses `thiserror` to derive `Error` and `Display` traits.

use num_complex::Complex64;
use thiserror::Error;

/// Errors related to Quantum Gates.
#[derive(Error, Debug, Clone)]
pub enum GateError {
    /// The matrix provided is not unitary (U†U != I).
    #[error("Matrix is not Unitary (U†U != I)")]
    NonUnitary,

    /// The matrix provided is not square.
    #[error("Matrix must be square")]
    NotSquareMatrix,

    /// The matrix dimensions are invalid (e.g., not a power of 2).
    #[error("Invalid Dimensions")]
    InvalidDimensions,

    /// A qubit index is used as both control and target.
    #[error("Qubit {0} cannot be both control and target")]
    ControlTargetOverlap(usize),

    /// A duplicate qubit index was found in the arguments.
    #[error("Duplicate qubit index found: {0}")]
    DuplicateQubit(usize),
}

/// Errors related to Quantum Measurements.
#[derive(Error, Debug, Clone)]
pub enum MeasurementError {
    /// The number of operators does not match the number of outcome values.
    #[error("Number of operators ({ops}) does not match number of values ({vals})")]
    CountMismatch { ops: usize, vals: usize },

    /// The measurement operators do not sum to Identity (Completeness relation failed).
    #[error("Measurement operators do not sum to Identity (Completeness relation failed)")]
    NotComplete,

    /// The operator dimensions are invalid.
    #[error("Invalid operator dimensions")]
    InvalidDimensions,

    /// Failed to expand the operator to the full system size.
    #[error("Operator expansion failed: {0}")]
    ExpansionError(#[from] GateError),

    /// A duplicate qubit index was found.
    #[error("Duplicate qubit index found: {0}")]
    DuplicateQubit(usize),

    /// The operator is not Hermitian.
    #[error("Not an hermitian operator")]
    NotHermitian,
}

/// Errors related to Quantum States.
#[derive(Error, Debug, Clone)]
pub enum StateError {
    /// The trace of the density matrix is not unity.
    #[error("Trace is not unity: {0}")]
    InvalidTrace(Complex64),

    /// The state vector is not normalized.
    #[error("Vector is not normalized. Norm squared: {0}")]
    NotNormalized(f64),

    /// The dimensions of the state are invalid.
    #[error("Invalid dimensions")]
    InvalidDimensions,

    /// There is a mismatch between expected and actual dimensions.
    #[error("Dimension mismatch")]
    DimensionMismatch {
        expected: usize,
        got_rows: usize,
        got_cols: usize,
    },

    /// A qubit index is out of bounds for the current state.
    #[error("Qubit index out of bounds")]
    IndexOutOfBounds { index: usize, num_qubits: usize },

    /// An error occurred during measurement.
    #[error("Measurement error: {0}")]
    MeasurementError(#[from] MeasurementError),

    /// An error occurred while applying a gate.
    #[error("Gate error: {0}")]
    GateError(#[from] GateError),

    /// An error occurred while applying a channel.
    #[error("Channel error: {0}")]
    ChannelError(#[from] ChannelError),
}

/// Errors related to Quantum Channels.
#[derive(Error, Debug, Clone)]
pub enum ChannelError {
    /// The channel has no Kraus operators.
    #[error("Channel must have at least one Kraus operator")]
    Empty,

    /// The Kraus operators do not sum to Identity (Trace preserving relation failed).
    #[error("Kraus operators do not sum to Identity (Trace preserving relation failed)")]
    NotComplete,

    /// The operator dimensions are invalid.
    #[error("Invalid operator dimensions: Matrices must be square and 2^n")]
    InvalidDimensions,

    /// The Kraus operators have different sizes.
    #[error("Dimension mismatch: All Kraus operators must have the same size")]
    OperatorSizeMismatch,

    /// The probability is invalid (not between 0.0 and 1.0).
    #[error("Invalid probability: {0}. Must be between 0.0 and 1.0")]
    InvalidProbability(f64),

    /// A duplicate qubit index was found.
    #[error("Duplicate qubit index found: {0}")]
    DuplicateQubit(usize),
}
