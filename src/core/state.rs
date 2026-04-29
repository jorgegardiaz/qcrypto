pub mod density;
pub mod vector;

pub use density::StateDensityMatrix;
pub use vector::StateVector;

use crate::core::errors::StateError;
use crate::{Gate, Measurement, MeasurementResult, QuantumChannel};

pub trait Validatable {
    fn is_valid(&self) -> Result<(), StateError>;
}

pub trait GateApplicable {
    fn apply(&mut self, gate: &Gate, target_qubits: &[usize]) -> Result<(), StateError>;
    fn apply_controlled(
        &mut self,
        gate: &Gate,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<(), StateError>;
}

pub trait Measurable {
    fn set_measurement(
        &self,
        measurement: &Measurement,
        target_qubits: &[usize],
    ) -> Result<Vec<f64>, StateError>;
    fn measure(
        &mut self,
        measurement: &Measurement,
        target_qubits: &[usize],
    ) -> Result<MeasurementResult, StateError>;
}

pub trait PurityComputable {
    fn purity(&self) -> f64;
}

pub trait StateClone {
    fn clone_box(&self) -> Box<dyn QuantumStateImpl>;
}

impl<T> StateClone for T
where
    T: 'static + QuantumStateImpl + Clone,
{
    fn clone_box(&self) -> Box<dyn QuantumStateImpl> {
        Box::new(self.clone())
    }
}

pub trait QuantumStateImpl:
    Validatable
    + GateApplicable
    + Measurable
    + PurityComputable
    + StateClone
    + std::fmt::Debug
    + Send
    + Sync
{
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_density_matrix(&self) -> Result<StateDensityMatrix, StateError>;
    fn try_apply_channel(
        &mut self,
        channel: &QuantumChannel,
        target_qubits: &[usize],
    ) -> Result<bool, StateError>;
}

/// Represents a general quantum state that can be pure (vector) or mixed (density matrix).
#[derive(Debug)]
pub struct QuantumState {
    pub state: Box<dyn QuantumStateImpl>,
}

impl Clone for QuantumState {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone_box(),
        }
    }
}

impl QuantumState {
    /// Creates a new quantum state initialized to the ground state |0...0> as a `StateVector`.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - The number of qubits in the quantum system.
    ///
    /// # Returns
    ///
    /// A `QuantumState` instance wrapping a `StateVector`.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::state::QuantumState;
    ///
    /// let state = QuantumState::new(2); // |00>
    /// ```
    pub fn new(num_qubits: usize) -> Self {
        QuantumState {
            state: Box::new(StateVector::new(num_qubits)),
        }
    }

    /// Verifies that the internal state representation is mathematically valid.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success if valid, or a `StateError`.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if dimensions are mismatched or if the probabilities do not normalize properly.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::state::QuantumState;
    ///
    /// let state = QuantumState::new(1);
    /// assert!(state.is_valid().is_ok());
    /// ```
    pub fn is_valid(&self) -> Result<(), StateError> {
        self.state.is_valid()
    }

    /// Applies a local quantum gate to the specified target qubits.
    ///
    /// # Arguments
    ///
    /// * `gate` - The quantum gate to apply.
    /// * `target_qubits` - The indices of the qubits the gate acts upon.
    ///
    /// # Returns
    ///
    /// A `Result` containing a mutable reference to the state.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if target indices are invalid or dimensions mismatch.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{state::QuantumState, Gate};
    ///
    /// let mut state = QuantumState::new(1);
    /// state.apply(&Gate::x(), &[0]).unwrap();
    /// ```
    pub fn apply(&mut self, gate: &Gate, target_qubits: &[usize]) -> Result<&mut Self, StateError> {
        self.state.apply(gate, target_qubits)?;
        Ok(self)
    }

    /// Applies a controlled quantum gate to the specified target qubits.
    ///
    /// # Arguments
    ///
    /// * `gate` - The quantum gate to apply.
    /// * `target_qubits` - The indices of the target qubits.
    /// * `control_qubits` - Optional slice containing the indices of the control qubits.
    ///
    /// # Returns
    ///
    /// A `Result` containing a mutable reference to the state.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if target/control indices are invalid or dimensions mismatch.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{state::QuantumState, Gate};
    ///
    /// let mut state = QuantumState::new(2);
    /// state.apply(&Gate::h(), &[0]).unwrap();
    /// state.apply_controlled(&Gate::x(), &[1], &[0]).unwrap();
    /// ```
    pub fn apply_controlled(
        &mut self,
        gate: &Gate,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<&mut Self, StateError> {
        self.state
            .apply_controlled(gate, target_qubits, control_qubits)?;
        Ok(self)
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
    /// A `Result` containing a vector of probabilities corresponding to each possible measurement outcome.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if the indices are invalid or if there are duplicate targets.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{state::QuantumState, Measurement};
    ///
    /// let state = QuantumState::new(1);
    /// let probs = state.set_measurement(&Measurement::z_basis(), &[0]).unwrap();
    /// assert_eq!(probs, vec![1.0, 0.0]);
    /// ```
    pub fn set_measurement(
        &self,
        measurement: &Measurement,
        target_qubits: &[usize],
    ) -> Result<Vec<f64>, StateError> {
        self.state.set_measurement(measurement, target_qubits)
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
    /// A `Result` containing a `MeasurementResult` with the index of the triggered operator and its associated value.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if dimensions or indices are invalid, or if the resulting trace is zero.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{state::QuantumState, Measurement};
    ///
    /// let mut state = QuantumState::new(1);
    /// let result = state.measure(&Measurement::z_basis(), &[0]).unwrap();
    /// assert_eq!(result.value, 0.0);
    /// ```
    pub fn measure(
        &mut self,
        measurement: &Measurement,
        target_qubits: &[usize],
    ) -> Result<MeasurementResult, StateError> {
        self.state.measure(measurement, target_qubits)
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
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if dimensions or indices are invalid.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{QuantumState, QuantumChannel};
    ///
    /// let mut state = QuantumState::new(1);
    /// let channel = QuantumChannel::bit_flip(1.0);
    /// state.apply_channel(&channel, &[0]).unwrap();
    /// ```
    pub fn apply_channel(
        &mut self,
        channel: &QuantumChannel,
        target_qubits: &[usize],
    ) -> Result<(), StateError> {
        if !self.state.try_apply_channel(channel, target_qubits)? {
            let mut dm = self.state.as_density_matrix()?;
            dm.apply_channel(channel, target_qubits)?;
            self.state = Box::new(dm);
        }
        Ok(())
    }

    /// Composes the current quantum state with an ancilla state via the tensor product.
    ///
    /// If both states are `StateVector`, the resulting state is also `StateVector`.
    /// Otherwise, the resulting state is returned as a `StateDensityMatrix`.
    ///
    /// # Arguments
    ///
    /// * `ancilla_state` - The quantum state to append to the system.
    ///
    /// # Returns
    ///
    /// A `Result` containing the newly combined `QuantumState`.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if tensor operations fail.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{QuantumState, Gate};
    /// # use num_complex::Complex64;
    ///
    /// let state1 = QuantumState::new(1); // |0>
    /// let mut state2 = QuantumState::new(1); // |0>
    /// state2.apply(&Gate::x(), &[0]); // |1>
    /// let combined = state1.compose(&state2).unwrap(); // |01>
    ///
    /// // We must downcast the internal Box<dyn QuantumStateImpl> to access the raw StateVector
    /// let vector_state = combined.state
    ///                            .as_any()
    ///                            .downcast_ref::<qcrypto::state::StateVector>()
    ///                            .unwrap();
    ///
    /// for (i, &val) in vector_state.amplitudes.indexed_iter() {
    ///     let expected = if i == 1 { 1.0 } else { 0.0 };
    ///     assert_eq!(val, Complex64::new(expected, 0.0));
    /// }
    /// ```
    pub fn compose(&self, ancilla_state: &QuantumState) -> Result<QuantumState, StateError> {
        if let (Some(v1), Some(v2)) = (
            self.state.as_any().downcast_ref::<StateVector>(),
            ancilla_state.state.as_any().downcast_ref::<StateVector>(),
        ) {
            Ok(QuantumState {
                state: Box::new(v1.compose(v2)?),
            })
        } else {
            let dm1 = self.state.as_density_matrix()?;
            let dm2 = ancilla_state.state.as_density_matrix()?;
            Ok(QuantumState {
                state: Box::new(dm1.compose(&dm2)?),
            })
        }
    }

    /// Calculates the purity of the quantum state $Tr(\rho^2)$.
    ///
    /// Returns 1.0 for completely pure states, and $< 1.0$ for mixed states.
    ///
    /// # Returns
    ///
    /// The computed purity as an `f64`.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{QuantumState, QuantumChannel};
    ///
    /// let mut state = QuantumState::new(1);
    /// assert_eq!(state.purity(), 1.0);
    /// state.apply_channel(&QuantumChannel::depolarizing(0.5), &[0]).unwrap();
    /// assert!(state.purity() < 1.0);
    /// ```
    pub fn purity(&self) -> f64 {
        self.state.purity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::errors::MeasurementError;
    use num_complex::Complex64;

    #[test]
    fn test_apply_out_of_bounds() {
        let mut state = QuantumState::new(1);
        let result = state.apply(&Gate::x(), &[1]);
        assert!(matches!(result, Err(StateError::IndexOutOfBounds { .. })));
    }

    #[test]
    fn test_apply_dimension_mismatch() {
        let mut state = QuantumState::new(2);
        let result = state.apply(&Gate::cnot(), &[0]);
        assert!(matches!(result, Err(StateError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_measurement_duplicate_qubits() {
        let state = QuantumState::new(2);
        let result = state.set_measurement(&Measurement::bell_basis(), &[0, 0]);
        assert!(matches!(
            result,
            Err(StateError::MeasurementError(
                MeasurementError::DuplicateQubit(0)
            ))
        ));
    }

    #[test]
    fn test_apply_identity_sequence() {
        let mut state = QuantumState::new(1);

        state
            .apply(&Gate::x(), &[0])
            .unwrap()
            .apply(&Gate::h(), &[0])
            .unwrap()
            .apply(&Gate::z(), &[0])
            .unwrap()
            .apply(&Gate::h(), &[0])
            .unwrap();

        let vector = state.state.as_any().downcast_ref::<StateVector>().unwrap();

        assert!((vector.amplitudes[0].re - 1.0).abs() < 1e-12);
        assert!((vector.amplitudes[1].re).abs() < 1e-12);
    }

    #[test]
    fn test_apply_controlled_bell_state() {
        let mut state = QuantumState::new(2); // |00>

        state
            .apply(&Gate::h(), &[0])
            .unwrap()
            .apply_controlled(&Gate::x(), &[1], &[0])
            .unwrap();

        let vector = state.state.as_any().downcast_ref::<StateVector>().unwrap();
        let expected_val = 1.0 / std::f64::consts::SQRT_2;

        for (i, &val) in vector.amplitudes.indexed_iter() {
            let expected = if i == 0 || i == 3 { expected_val } else { 0.0 };
            assert!((val - Complex64::new(expected, 0.0)).norm() < 1e-12);
        }
    }

    #[test]
    fn test_apply_cnot_bell_state() {
        // Same Bell state but using Gate::cnot() through apply()
        let mut state = QuantumState::new(2);
        state
            .apply(&Gate::h(), &[0])
            .unwrap()
            .apply(&Gate::cnot(), &[0, 1])
            .unwrap();

        let vector = state.state.as_any().downcast_ref::<StateVector>().unwrap();
        let s = 1.0 / std::f64::consts::SQRT_2;

        // Φ+ = (|00⟩+|11⟩)/√2
        assert!((vector.amplitudes[0] - Complex64::new(s, 0.0)).norm() < 1e-12);
        assert!(vector.amplitudes[1].norm() < 1e-12);
        assert!(vector.amplitudes[2].norm() < 1e-12);
        assert!((vector.amplitudes[3] - Complex64::new(s, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_apply_cnot_equivalence() {
        // apply(&cnot, &[0,1]) must produce the same result as apply_controlled(&x, &[1], &[0])
        let mut state_a = QuantumState::new(2);
        state_a
            .apply(&Gate::h(), &[0])
            .unwrap()
            .apply(&Gate::cnot(), &[0, 1])
            .unwrap();

        let mut state_b = QuantumState::new(2);
        state_b
            .apply(&Gate::h(), &[0])
            .unwrap()
            .apply_controlled(&Gate::x(), &[1], &[0])
            .unwrap();

        let va = state_a
            .state
            .as_any()
            .downcast_ref::<StateVector>()
            .unwrap();
        let vb = state_b
            .state
            .as_any()
            .downcast_ref::<StateVector>()
            .unwrap();
        for (a, b) in va.amplitudes.iter().zip(vb.amplitudes.iter()) {
            assert!((a - b).norm() < 1e-12, "cnot via apply != apply_controlled");
        }
    }

    #[test]
    fn test_apply_cnot_control_does_not_fire() {
        // |01⟩: qubit 0 = 0, so CNOT(control=0) should NOT flip qubit 1.
        let mut state = QuantumState::new(2);
        state.apply(&Gate::x(), &[1]).unwrap(); // |01⟩
        state.apply(&Gate::cnot(), &[0, 1]).unwrap();

        let vector = state.state.as_any().downcast_ref::<StateVector>().unwrap();
        assert!(
            (vector.amplitudes[1] - Complex64::new(1.0, 0.0)).norm() < 1e-12,
            "Must stay |01⟩"
        );
    }

    #[test]
    fn test_apply_cnot_reversed_targets() {
        // targets=[1,0] → gate qubit 0 (control)=state qubit 1, gate qubit 1 (target)=state qubit 0
        // |01⟩: qubit 1 = 1 → control fires → flip qubit 0 → |11⟩
        let mut state = QuantumState::new(2);
        state.apply(&Gate::x(), &[1]).unwrap(); // |01⟩
        state.apply(&Gate::cnot(), &[1, 0]).unwrap();

        let vector = state.state.as_any().downcast_ref::<StateVector>().unwrap();
        assert!(
            (vector.amplitudes[3] - Complex64::new(1.0, 0.0)).norm() < 1e-12,
            "Expected |11⟩"
        );
    }

    #[test]
    fn test_apply_swap() {
        // |10⟩ with SWAP → |01⟩
        let mut state = QuantumState::new(2);
        state.apply(&Gate::x(), &[0]).unwrap(); // |10⟩
        state.apply(&Gate::swap(), &[0, 1]).unwrap();

        let vector = state.state.as_any().downcast_ref::<StateVector>().unwrap();
        assert!(
            (vector.amplitudes[1] - Complex64::new(1.0, 0.0)).norm() < 1e-12,
            "Expected |01⟩"
        );
        assert!(vector.amplitudes[2].norm() < 1e-12, "|10⟩ should be empty");
    }

    #[test]
    fn test_measure_collapse() {
        let mut state = QuantumState::new(1);

        let _result = state
            .apply(&Gate::h(), &[0])
            .unwrap()
            .measure(&Measurement::z_basis(), &[0])
            .unwrap();

        assert!((state.purity() - 1.0).abs() < 1e-12);

        let vector = state.state.as_any().downcast_ref::<StateVector>().unwrap();

        // Norm should be 1.0
        let norm_sqr: f64 = vector.amplitudes.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm_sqr - 1.0).abs() < 1e-12);

        let mut hit_0 = false;
        let mut hit_1 = false;
        for _ in 0..20 {
            let mut state = QuantumState::new(1);
            state.apply(&Gate::h(), &[0]).unwrap();
            let result = state.measure(&Measurement::z_basis(), &[0]).unwrap();
            let vector = state.state.as_any().downcast_ref::<StateVector>().unwrap();
            if result.value == 0.0 {
                assert!((vector.amplitudes[0].re - 1.0).abs() < 1e-12);
                assert!((vector.amplitudes[1].re).abs() < 1e-12);
                hit_0 = true;
            } else {
                assert!((vector.amplitudes[1].re - 1.0).abs() < 1e-12);
                assert!((vector.amplitudes[0].re).abs() < 1e-12);
                hit_1 = true;
            }
            if hit_0 && hit_1 {
                break;
            }
        }
        assert!(hit_0 && hit_1, "Both branches should be hit eventually");
    }

    #[test]
    fn test_apply_channel_conversion() {
        let mut state = QuantumState::new(1); // Starts as StateVector |0>

        // Assert it is currently a vector
        assert!(state.state.as_any().downcast_ref::<StateVector>().is_some());

        state.apply(&Gate::h(), &[0]).unwrap(); // |+>

        let channel = QuantumChannel::bit_flip(1.0);
        state.apply_channel(&channel, &[0]).unwrap(); // Applies noise, should convert to density matrix

        // Assert it is NOW a density matrix
        assert!(
            state
                .state
                .as_any()
                .downcast_ref::<StateDensityMatrix>()
                .is_some()
        );

        let dm = state
            .state
            .as_any()
            .downcast_ref::<StateDensityMatrix>()
            .unwrap();

        // Since X|+> = |+>, density matrix should be exactly |+><+|
        for &val in dm.density_matrix.iter() {
            assert!((val - Complex64::new(0.5, 0.0)).norm() < 1e-12);
        }
    }

    #[test]
    fn test_apply_channel_on_density_matrix() {
        let mut state = QuantumState::new(1);
        let channel = QuantumChannel::bit_flip(1.0);
        // First conversion
        state.apply_channel(&channel, &[0]).unwrap();
        // Second apply directly on density matrix
        state.apply_channel(&channel, &[0]).unwrap();
        let dm = state
            .state
            .as_any()
            .downcast_ref::<StateDensityMatrix>()
            .unwrap();
        assert!((dm.density_matrix[[0, 0]].re - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_quantum_state_clone() {
        let state = QuantumState::new(1);
        let cloned = state.clone();
        assert!(cloned.is_valid().is_ok());
    }

    #[test]
    fn test_quantum_state_is_valid() {
        let state = QuantumState::new(2);
        assert!(state.is_valid().is_ok());
    }

    #[test]
    fn test_quantum_state_compose_vector() {
        let state1 = QuantumState::new(1);
        let mut state2 = QuantumState::new(1);
        state2.apply(&Gate::x(), &[0]).unwrap();
        let combined = state1.compose(&state2).unwrap();
        let v = combined
            .state
            .as_any()
            .downcast_ref::<StateVector>()
            .unwrap();
        assert!((v.amplitudes[1].re - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_quantum_state_compose_density_matrix() {
        let mut state1 = QuantumState::new(1);
        state1
            .apply_channel(&QuantumChannel::bit_flip(1.0), &[0])
            .unwrap(); // Converts to DM

        let mut state2 = QuantumState::new(1);
        state2
            .apply_channel(&QuantumChannel::bit_flip(1.0), &[0])
            .unwrap(); // Converts to DM

        let combined = state1.compose(&state2).unwrap();
        let dm = combined
            .state
            .as_any()
            .downcast_ref::<StateDensityMatrix>()
            .unwrap();
        assert!((dm.density_matrix[[3, 3]].re - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_entanglement_correlation() {
        // Create Bell state |\Phi+> = (|00> + |11>) / √2
        let mut state = QuantumState::new(2);
        state
            .apply(&Gate::h(), &[0])
            .unwrap()
            .apply_controlled(&Gate::x(), &[1], &[0])
            .unwrap();

        // Measure qubit 0 in Z basis
        let result_q0 = state.measure(&Measurement::z_basis(), &[0]).unwrap();

        // Now measure qubit 1 — it MUST give the same outcome (entanglement correlation)
        let result_q1 = state.measure(&Measurement::z_basis(), &[1]).unwrap();

        assert_eq!(
            result_q0.value, result_q1.value,
            "Entangled qubits must collapse to the same value in Bell state Φ+"
        );
    }
}
