pub mod density;
pub mod vector;

pub use density::StateDensityMatrix;
pub use vector::StateVector;

use crate::core::errors::StateError;
use crate::{Gate, Measurement, MeasurementResult, QuantumChannel};

/// Represents a general quantum state that can be pure (vector) or mixed (density matrix).
#[derive(Clone, Debug)]
pub enum QuantumState {
    /// A pure quantum state represented by a state vector.
    StateVector(StateVector),
    /// A mixed quantum state represented by a density matrix.
    StateDensityMatrix(StateDensityMatrix),
}

impl QuantumState {
    /// Creates a new quantum state initialized to the ground state |0...0> as a `StateVector`.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - The number of qubits in the quantum system.
    pub fn new(num_qubits: usize) -> Self {
        QuantumState::StateVector(StateVector::new(num_qubits))
    }

    /// Verifies that the internal state representation is mathematically valid.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if dimensions are mismatched or if the probabilities do not normalize properly.
    pub fn is_valid(&self) -> Result<(), StateError> {
        match self {
            QuantumState::StateVector(v) => v.is_valid(),
            QuantumState::StateDensityMatrix(dm) => dm.is_valid(),
        }
    }

    /// Applies a local quantum gate to the specified target qubits.
    ///
    /// # Arguments
    ///
    /// * `gate` - The quantum gate to apply.
    /// * `target_qubits` - The indices of the qubits the gate acts upon.
    pub fn apply(&mut self, gate: &Gate, target_qubits: &[usize]) -> Result<(), StateError> {
        self.apply_controlled(gate, target_qubits, None)
    }

    /// Applies a controlled quantum gate to the specified target qubits.
    ///
    /// # Arguments
    ///
    /// * `gate` - The quantum gate to apply.
    /// * `target_qubits` - The indices of the target qubits.
    /// * `control_qubits` - Optional slice containing the indices of the control qubits.
    pub fn apply_controlled(
        &mut self,
        gate: &Gate,
        target_qubits: &[usize],
        control_qubits: Option<&[usize]>,
    ) -> Result<(), StateError> {
        match self {
            QuantumState::StateVector(v) => v.apply_controlled(gate, target_qubits, control_qubits),
            QuantumState::StateDensityMatrix(dm) => {
                dm.apply_controlled(gate, target_qubits, control_qubits)
            }
        }
    }

    /// Calculates measurement outcome probabilities without collapsing the state.
    ///
    /// # Arguments
    ///
    /// * `measurement` - The `Measurement` operator to simulate.
    /// * `target_qubits` - The indices of the qubits to apply the measurement on.
    ///
    /// # Returns
    ///
    /// A vector of probabilities corresponding to each possible measurement outcome.
    pub fn set_measurement(
        &self,
        measurement: &Measurement,
        target_qubits: &[usize],
    ) -> Result<Vec<f64>, StateError> {
        match self {
            QuantumState::StateVector(v) => v.set_measurement(measurement, target_qubits),
            QuantumState::StateDensityMatrix(dm) => dm.set_measurement(measurement, target_qubits),
        }
    }

    /// Performs a physical measurement, collapsing the quantum state according to the outcome.
    ///
    /// # Arguments
    ///
    /// * `measurement` - The `Measurement` to perform.
    /// * `target_qubits` - The indices of the qubits being measured.
    ///
    /// # Returns
    ///
    /// A `MeasurementResult` containing the index of the triggered operator and its associated value.
    pub fn measure(
        &mut self,
        measurement: &Measurement,
        target_qubits: &[usize],
    ) -> Result<MeasurementResult, StateError> {
        match self {
            QuantumState::StateVector(v) => v.measure(measurement, target_qubits),
            QuantumState::StateDensityMatrix(dm) => dm.measure(measurement, target_qubits),
        }
    }

    /// Applies a quantum channel (noise model) to the specified qubits.
    ///
    /// Note: If the state is currently a `StateVector`, this operation will automatically
    /// convert it into a `StateDensityMatrix` to support mixed states.
    ///
    /// # Arguments
    ///
    /// * `channel` - The quantum channel to apply.
    /// * `target_qubits` - The indices of the qubits affected by the channel.
    pub fn apply_channel(
        &mut self,
        channel: &QuantumChannel,
        target_qubits: &[usize],
    ) -> Result<(), StateError> {
        match self {
            QuantumState::StateVector(v) => {
                let mut dm = StateDensityMatrix::from_state_vector(v.amplitudes.clone())?;
                dm.apply_channel(channel, target_qubits)?;
                *self = QuantumState::StateDensityMatrix(dm);
                Ok(())
            }
            QuantumState::StateDensityMatrix(dm) => dm.apply_channel(channel, target_qubits),
        }
    }

    /// Composes the current quantum state with an ancilla state via the tensor product.
    ///
    /// If both states are pure (`StateVector`), the resulting state is also pure.
    /// Otherwise, the resulting state is returned as a `StateDensityMatrix`.
    ///
    /// # Arguments
    ///
    /// * `ancilla_state` - The quantum state to append to the system.
    pub fn compose(&self, ancilla_state: &QuantumState) -> Result<QuantumState, StateError> {
        match (self, ancilla_state) {
            (QuantumState::StateVector(v1), QuantumState::StateVector(v2)) => {
                Ok(QuantumState::StateVector(v1.compose(v2)?))
            }
            (QuantumState::StateDensityMatrix(dm1), QuantumState::StateDensityMatrix(dm2)) => {
                Ok(QuantumState::StateDensityMatrix(dm1.compose(dm2)?))
            }
            (QuantumState::StateVector(v1), QuantumState::StateDensityMatrix(dm2)) => {
                let dm1 = StateDensityMatrix::from_state_vector(v1.amplitudes.clone())?;
                Ok(QuantumState::StateDensityMatrix(dm1.compose(dm2)?))
            }
            (QuantumState::StateDensityMatrix(dm1), QuantumState::StateVector(v2)) => {
                let dm2 = StateDensityMatrix::from_state_vector(v2.amplitudes.clone())?;
                Ok(QuantumState::StateDensityMatrix(dm1.compose(&dm2)?))
            }
        }
    }

    /// Calculates the purity of the quantum state $Tr(\rho^2)$.
    ///
    /// Returns 1.0 for completely pure states (`StateVector`), and $< 1.0$ for mixed states.
    pub fn purity(&self) -> f64 {
        match self {
            QuantumState::StateVector(v) => v.purity(),
            QuantumState::StateDensityMatrix(dm) => dm.purity(),
        }
    }
}
