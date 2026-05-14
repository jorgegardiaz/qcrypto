use crate::core::errors::{GateError, MeasurementError, StateError};
use crate::core::state::density::StateDensityMatrix;
use crate::core::state::{
    GateApplicable, Measurable, PurityComputable, QuantumStateImpl, Validatable,
};
use crate::{Gate, Measurement, MeasurementResult, QuantumChannel, core::utils};
use ndarray::Array1;
use num_complex::Complex64;
use rayon::prelude::*;

/// Represents a quantum state using a State Vector (pure state).
#[derive(Clone, Debug)]
pub struct StateVector {
    /// The probability amplitudes of the state vector.
    pub amplitudes: Array1<Complex64>,
    /// The number of qubits within the system.
    pub num_qubits: usize,
}

impl StateVector {
    /// Creates a new pure quantum state initialized to the ground state |0...0>.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - The number of qubits in the system.
    ///
    /// # Returns
    ///
    /// A new `StateVector` instance.
    ///
    /// # Example
    ///
    /// ```rust
    /// use qcrypto::state::StateVector;
    /// # use num_complex::Complex64;
    ///
    /// // Initialize a 2-qubit state
    /// let state = StateVector::new(2);
    ///
    /// // Verify qubit count
    /// assert_eq!(state.num_qubits, 2);
    ///
    /// // 2 qubits means the state vector dimension should be 2^2 (4)
    /// assert_eq!(state.amplitudes.dim(), 4);
    ///
    /// // The state is |00>
    /// for (i, &val) in state.amplitudes.indexed_iter() {
    ///     let expected = if i == 0 { 1.0 } else { 0.0 };
    ///     assert_eq!(val, Complex64::new(expected, 0.0));
    /// }
    /// ```
    pub fn new(num_qubits: usize) -> Self {
        assert!(
            num_qubits > 0,
            "Number of qubits must be at least 1, got 0"
        );
        let dim = 1 << num_qubits;
        let mut amplitudes = Array1::<Complex64>::zeros(dim);
        amplitudes[0] = Complex64::new(1.0, 0.0);

        Self {
            amplitudes,
            num_qubits,
        }
    }

    /// Checks if a vector amplitude array represents a valid quantum state.
    fn check_vector_state(vector: &Array1<Complex64>) -> Result<(), StateError> {
        let dim = vector.len();

        if !dim.is_power_of_two() {
            return Err(StateError::InvalidDimensions);
        }

        let norm_sqr: f64 = vector.iter().map(|c| c.norm_sqr()).sum();

        if (norm_sqr - 1.0).abs() > 1e-12 {
            return Err(StateError::NotNormalized(norm_sqr));
        }

        Ok(())
    }

    /// Validates whether a given qubit index falls within the system's defined boundaries.
    fn validate_qubit_index(&self, index: usize) -> Result<(), StateError> {
        if index >= self.num_qubits {
            return Err(StateError::IndexOutOfBounds {
                index,
                num_qubits: self.num_qubits,
            });
        }
        Ok(())
    }

    /// Checks if the underlying vector state holds mathematical validity properties.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success `Ok(())` if the state is normalized and dimensions are valid.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if dimensions are not a power of 2 or the state vector is not normalized to 1.0.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::state::StateVector;
    ///
    /// let state = StateVector::new(2);
    /// assert!(state.is_valid().is_ok());
    /// ```
    pub fn is_valid(&self) -> Result<(), StateError> {
        Self::check_vector_state(&self.amplitudes)
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
    /// A `Result` containing a mutable reference to `Self` (`&mut Self`) to allow method chaining.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if target qubits are out of bounds or the gate dimension mismatches.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{state::StateVector, Gate};
    /// # use num_complex::Complex64;
    ///
    /// let mut state = StateVector::new(1); // |0>
    ///
    /// // Apply NOT gate
    /// state.apply(&Gate::x(), &[0]).unwrap();
    ///
    /// // Now it should be |1>
    /// assert_eq!(state.amplitudes[1], Complex64::new(1.0, 0.0));
    /// assert_eq!(state.amplitudes[0], Complex64::new(0.0, 0.0));
    /// ```
    pub fn apply(&mut self, gate: &Gate, target_qubits: &[usize]) -> Result<&mut Self, StateError> {
        self.apply_controlled(gate, target_qubits, &[])
    }

    /// Applies a controlled quantum gate to the specified target qubits.
    ///
    /// This performs local unitary evolution $U |\psi\rangle$ specifically restricted
    /// to the requested local subsystem to avoid global tensor memory allocation.
    ///
    /// # Arguments
    ///
    /// * `gate` - The quantum gate to apply.
    /// * `target_qubits` - The indices of the target qubits.
    /// * `control_qubits` - Slice with the indices of the control qubits (empty if none).
    ///
    /// # Returns
    ///
    /// A `Result` containing a mutable reference to `Self` (`&mut Self`) to allow method chaining.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if target/control qubits are out of bounds or the gate dimension mismatches.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{state::StateVector, Gate};
    /// # use num_complex::Complex64;
    ///
    /// let mut state = StateVector::new(2);
    ///
    /// // Apply X gate to qubit 0 and CNOT -> |11>
    /// state.apply(&Gate::x(), &[0]).unwrap()
    ///      .apply_controlled(&Gate::x(), &[1], &[0]).unwrap();
    ///
    /// // The state vector for |11> should have a 1.0 at index 3 and 0.0 elsewhere
    /// for (i, &val) in state.amplitudes.indexed_iter() {
    ///     let expected = if i == 3 { 1.0 } else { 0.0 };
    ///     assert_eq!(val, Complex64::new(expected, 0.0));
    /// }
    /// ```
    pub fn apply_controlled(
        &mut self,
        gate: &Gate,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<&mut Self, StateError> {
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

        for &q in control_qubits {
            self.validate_qubit_index(q)?;
        }

        if let Some(dup) = utils::find_duplicate(target_qubits) {
            return Err(GateError::DuplicateQubit(dup).into());
        }

        if let Some(dup) = utils::find_duplicate(control_qubits) {
            return Err(GateError::DuplicateQubit(dup).into());
        }

        for &c in control_qubits {
            if target_qubits.contains(&c) {
                return Err(GateError::ControlTargetOverlap(c).into());
            }
        }

        // Apply local operator over specific target qubits using the underlying utils engine
        self.amplitudes = utils::apply_local_vector(
            self.num_qubits,
            &self.amplitudes,
            &gate.matrix,
            target_qubits,
            control_qubits,
        );

        Ok(self)
    }

    /// Calculates measurement outcome probabilities without collapsing the state.
    ///
    /// The probability relies on calculating the norm squared of the state after applying
    /// the measurement Kraus operator: $p_k = \| M_k |\psi\rangle \|^2$.
    ///
    /// # Arguments
    ///
    /// * `measurement` - The measurement protocol containing the operator targets.
    /// * `target_qubits` - The subspace over which to measure.
    ///
    /// # Returns
    ///
    /// A `Result` containing a `Vec<f64>` mapping probability to each measurement operator.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if the measurement dimensions or target qubits are invalid.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{state::StateVector, Measurement};
    ///
    /// let state = StateVector::new(1); // |0>
    ///
    /// // Probabilities of measuring in Z basis
    /// let probs = state.set_measurement(&Measurement::z_basis(), &[0]).unwrap();
    /// assert_eq!(probs, vec![1.0, 0.0]); // 100% |0>, 0% |1>
    /// ```
    pub fn set_measurement(
        &self,
        measurement: &Measurement,
        target_qubits: &[usize],
    ) -> Result<Vec<f64>, StateError> {
        if measurement.num_qubits != target_qubits.len() {
            return Err(StateError::DimensionMismatch {
                expected: measurement.num_qubits,
                got_rows: target_qubits.len(),
                got_cols: 0,
            });
        }

        for &q in target_qubits {
            self.validate_qubit_index(q)?;
        }

        if let Some(dup) = utils::find_duplicate(target_qubits) {
            return Err(StateError::MeasurementError(
                MeasurementError::DuplicateQubit(dup),
            ));
        }

        // p_k = || M_k |\psi> ||^2
        let mut probs: Vec<f64> = measurement
            .operators
            .par_iter()
            .map(|m_k| {
                let temp = utils::apply_local_vector(
                    self.num_qubits,
                    &self.amplitudes,
                    m_k,
                    target_qubits,
                    &[],
                );

                // Compute norm squared
                temp.iter().map(|c| c.norm_sqr()).sum()
            })
            .collect();

        let sum_probs: f64 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum_probs;
        }

        Ok(probs)
    }

    /// Randomly selects an operator index weighted by the calculated probability distribution `probs`.
    fn pick_outcome(&self, probs: &[f64]) -> usize {
        let roll: f64 = crate::rng::random_f64();

        let mut cumulative = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if roll < cumulative {
                return i;
            }
        }
        probs.len().saturating_sub(1)
    }

    /// Performs a physical measurement, collapsing the quantum state.
    ///
    /// The collapse maps the quantum state according to the formula: $|\psi\rangle \to \frac{M_k |\psi\rangle}{\sqrt{p_k}}$.
    ///
    /// # Arguments
    ///
    /// * `measurement` - The `Measurement` operation to perform.
    /// * `target_qubits` - The indices of the qubits being measured.
    ///
    /// # Returns
    ///
    /// A `Result` containing a `MeasurementResult` tracking both the index of the outcome operator alongside its generic value.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if measurement dimensions or target qubits are invalid, or if the resulting trace is 0.0.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{state::StateVector, Measurement};
    ///
    /// let mut state = StateVector::new(1); // |0>
    ///
    /// // Measure in Z basis
    /// let result = state.measure(&Measurement::z_basis(), &[0]).unwrap();
    /// assert_eq!(result.value, 0.0); // Output should correspond to |0>
    /// ```
    pub fn measure(
        &mut self,
        measurement: &Measurement,
        target_qubits: &[usize],
    ) -> Result<MeasurementResult, StateError> {
        let probs = self.set_measurement(measurement, target_qubits)?;

        let outcome_idx = self.pick_outcome(&probs);
        let p_selected = probs[outcome_idx];

        if p_selected > 1e-12 {
            let m_k = &measurement.operators[outcome_idx];

            let temp = utils::apply_local_vector(
                self.num_qubits,
                &self.amplitudes,
                m_k,
                target_qubits,
                &[],
            );

            let norm = p_selected.sqrt();
            self.amplitudes = temp.mapv(|val| val / Complex64::new(norm, 0.0));
        } else {
            return Err(StateError::InvalidTrace(Complex64::new(0.0, 0.0)));
        }

        Ok(MeasurementResult {
            index: outcome_idx,
            value: measurement.values[outcome_idx],
        })
    }

    /// Composes the current state vector with an ancilla state vector (tensor product).
    ///
    /// # Arguments
    ///
    /// * `ancilla_state` - Another `StateVector` to append to the system via Kronecker product.
    ///
    /// # Returns
    ///
    /// A `Result` containing the combined new `StateVector`.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if tensor operations fail (though typical structural constraints prevent this).
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::state::StateVector;
    ///
    /// let state1 = StateVector::new(1);
    /// let state2 = StateVector::new(2);
    ///
    /// let combined = state1.compose(&state2).unwrap();
    /// assert_eq!(combined.num_qubits, 3);
    /// ```
    pub fn compose(&self, ancilla_state: &StateVector) -> Result<StateVector, StateError> {
        // Kronecker product for 1D arrays
        let n = self.amplitudes.len();
        let m = ancilla_state.amplitudes.len();
        let mut composite_amplitudes = Array1::<Complex64>::zeros(n * m);

        for i in 0..n {
            for j in 0..m {
                composite_amplitudes[i * m + j] = self.amplitudes[i] * ancilla_state.amplitudes[j];
            }
        }

        Ok(StateVector {
            amplitudes: composite_amplitudes,
            num_qubits: self.num_qubits + ancilla_state.num_qubits,
        })
    }

    /// Extrapolates the purity of a `StateVector`.
    ///
    /// Since state vectors only represent fully pure mathematical quantum states,
    /// this function will always return `1.0`.
    ///
    /// # Returns
    ///
    /// The purity as a `f64` (always 1.0).
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::state::StateVector;
    ///
    /// let state = StateVector::new(1);
    /// assert_eq!(state.purity(), 1.0);
    /// ```
    pub fn purity(&self) -> f64 {
        1.0
    }
}

impl Validatable for StateVector {
    fn is_valid(&self) -> Result<(), StateError> {
        self.is_valid()
    }
}

impl GateApplicable for StateVector {
    fn apply(&mut self, gate: &Gate, target_qubits: &[usize]) -> Result<(), StateError> {
        self.apply(gate, target_qubits)?;
        Ok(())
    }

    fn apply_controlled(
        &mut self,
        gate: &Gate,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<(), StateError> {
        self.apply_controlled(gate, target_qubits, control_qubits)?;
        Ok(())
    }
}

impl Measurable for StateVector {
    fn set_measurement(
        &self,
        measurement: &Measurement,
        target_qubits: &[usize],
    ) -> Result<Vec<f64>, StateError> {
        self.set_measurement(measurement, target_qubits)
    }

    fn measure(
        &mut self,
        measurement: &Measurement,
        target_qubits: &[usize],
    ) -> Result<MeasurementResult, StateError> {
        self.measure(measurement, target_qubits)
    }
}

impl PurityComputable for StateVector {
    fn purity(&self) -> f64 {
        self.purity()
    }
}

impl QuantumStateImpl for StateVector {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_density_matrix(&self) -> Result<StateDensityMatrix, StateError> {
        StateDensityMatrix::from_state_vector(self.amplitudes.clone())
    }

    fn try_apply_channel(
        &mut self,
        _channel: &QuantumChannel,
        _target_qubits: &[usize],
    ) -> Result<bool, StateError> {
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_vector_state_invalid_dimensions() {
        let vec = Array1::<Complex64>::zeros(3);
        assert!(matches!(
            StateVector::check_vector_state(&vec),
            Err(StateError::InvalidDimensions)
        ));
    }

    #[test]
    fn test_check_vector_state_not_normalized() {
        let vec = Array1::<Complex64>::zeros(2);
        assert!(matches!(
            StateVector::check_vector_state(&vec),
            Err(StateError::NotNormalized(_))
        ));
    }

    #[test]
    fn test_set_measurement_dimension_mismatch() {
        let state = StateVector::new(1);
        let m = Measurement::bell_basis(); // 2-qubit measurement
        let result = state.set_measurement(&m, &[0]);
        assert!(matches!(result, Err(StateError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_apply_out_of_bounds() {
        let mut state = StateVector::new(1); // 1 qubit (index 0)

        // Target an index that doesn't exist
        let result = state.apply(&Gate::x(), &[1]);

        assert!(matches!(
            result,
            Err(StateError::IndexOutOfBounds {
                index: 1,
                num_qubits: 1,
            })
        ));
    }

    #[test]
    fn test_apply_dimension_mismatch() {
        let mut state = StateVector::new(2);

        // CNOT is a 2-qubit gate, but we only give it 1 target qubit
        let result = state.apply(&Gate::cnot(), &[0]);

        assert!(matches!(result, Err(StateError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_measurement_duplicate_qubits() {
        let state = StateVector::new(2);

        // Use bell_basis (2 qubits) but target qubit 0 twice
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
        let mut state = StateVector::new(1); // |0>
        let initial_amplitudes = state.amplitudes.clone();

        // Apply X, then H, then Z, then H.
        // HZH = X. So HZHX = XX = I.
        state
            .apply(&Gate::x(), &[0])
            .unwrap()
            .apply(&Gate::h(), &[0])
            .unwrap()
            .apply(&Gate::z(), &[0])
            .unwrap()
            .apply(&Gate::h(), &[0])
            .unwrap();

        // Check it's identical
        for (i, &val) in state.amplitudes.indexed_iter() {
            let diff = (val - initial_amplitudes[i]).norm();
            assert!(
                diff < 1e-12,
                "State modified at index {:?}, diff: {}",
                i,
                diff
            );
        }
    }

    #[test]
    fn test_apply_controlled_bell_state() {
        let mut state = StateVector::new(2); // |00>

        // H on qubit 0, CNOT on 0 -> 1
        state
            .apply(&Gate::h(), &[0])
            .unwrap()
            .apply_controlled(&Gate::x(), &[1], &[0])
            .unwrap();

        // The state vector for (|00> + |11>)/sqrt(2)
        // Indices 0 and 3 should be 1/sqrt(2) ~ 0.70710678118
        let expected_val = 1.0 / std::f64::consts::SQRT_2;

        for (i, &val) in state.amplitudes.indexed_iter() {
            let expected = if i == 0 || i == 3 { expected_val } else { 0.0 };
            let diff = (val - Complex64::new(expected, 0.0)).norm();
            assert!(diff < 1e-12, "Unexpected value at {}", i);
        }
    }

    #[test]
    fn test_apply_cnot_bell_state() {
        let mut state = StateVector::new(2);
        state
            .apply(&Gate::h(), &[0])
            .unwrap()
            .apply(&Gate::cnot(), &[0, 1])
            .unwrap();

        let s = 1.0 / std::f64::consts::SQRT_2;
        assert!((state.amplitudes[0] - Complex64::new(s, 0.0)).norm() < 1e-12);
        assert!(state.amplitudes[1].norm() < 1e-12);
        assert!(state.amplitudes[2].norm() < 1e-12);
        assert!((state.amplitudes[3] - Complex64::new(s, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_apply_cnot_equivalence() {
        let mut state_a = StateVector::new(2);
        state_a
            .apply(&Gate::h(), &[0])
            .unwrap()
            .apply(&Gate::cnot(), &[0, 1])
            .unwrap();

        let mut state_b = StateVector::new(2);
        state_b
            .apply(&Gate::h(), &[0])
            .unwrap()
            .apply_controlled(&Gate::x(), &[1], &[0])
            .unwrap();

        for (a, b) in state_a.amplitudes.iter().zip(state_b.amplitudes.iter()) {
            assert!((a - b).norm() < 1e-12, "cnot via apply != apply_controlled");
        }
    }

    #[test]
    fn test_apply_cnot_reversed_targets() {
        // targets=[1,0]: control=qubit 1, target=qubit 0.
        // |01⟩: qubit 1=1 fires → flip qubit 0 → |11⟩
        let mut state = StateVector::new(2);
        state.apply(&Gate::x(), &[1]).unwrap(); // |01⟩
        state.apply(&Gate::cnot(), &[1, 0]).unwrap();

        assert!((state.amplitudes[3] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        assert!(state.amplitudes[1].norm() < 1e-12);
    }

    #[test]
    fn test_apply_swap() {
        let mut state = StateVector::new(2);
        state.apply(&Gate::x(), &[0]).unwrap(); // |10⟩
        state.apply(&Gate::swap(), &[0, 1]).unwrap();

        assert!(
            (state.amplitudes[1] - Complex64::new(1.0, 0.0)).norm() < 1e-12,
            "Expected |01⟩"
        );
        assert!(state.amplitudes[2].norm() < 1e-12);
    }

    #[test]
    fn test_measure_collapse() {
        let mut state = StateVector::new(1);

        // Put in |+> and measure in Z basis
        let _result = state
            .apply(&Gate::h(), &[0])
            .unwrap()
            .measure(&Measurement::z_basis(), &[0])
            .unwrap();

        // Purity should be 1.0 after measurement
        assert!((state.purity() - 1.0).abs() < 1e-12);

        // Norm should be 1.0
        let norm_sqr: f64 = state.amplitudes.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm_sqr - 1.0).abs() < 1e-12);

        let mut hit_0 = false;
        let mut hit_1 = false;
        for _ in 0..20 {
            let mut state = StateVector::new(1);
            let result = state
                .apply(&Gate::h(), &[0])
                .unwrap()
                .measure(&Measurement::z_basis(), &[0])
                .unwrap();

            if result.value == 0.0 {
                assert!((state.amplitudes[0].re - 1.0).abs() < 1e-12);
                assert!((state.amplitudes[1].re).abs() < 1e-12);
                hit_0 = true;
            } else {
                assert!((state.amplitudes[1].re - 1.0).abs() < 1e-12);
                assert!((state.amplitudes[0].re).abs() < 1e-12);
                hit_1 = true;
            }
            if hit_0 && hit_1 {
                break;
            }
        }
        assert!(hit_0 && hit_1, "Both branches should be hit eventually");
    }

    #[test]
    fn test_pick_outcome_fallback() {
        let state = StateVector::new(1);
        let probs = vec![0.0, 0.0];
        let idx = state.pick_outcome(&probs);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_measure_zero_prob() {
        let mut state = StateVector::new(1);
        state.amplitudes = ndarray::Array1::zeros(2);
        let result = state.measure(&Measurement::z_basis(), &[0]);
        assert!(matches!(result, Err(StateError::InvalidTrace(_))));
    }

    #[test]
    fn test_apply_controlled_duplicate_targets() {
        let mut state = StateVector::new(2);
        let result = state.apply_controlled(&Gate::swap(), &[0, 0], &[]);
        assert!(matches!(
            result,
            Err(StateError::GateError(GateError::DuplicateQubit(0)))
        ));
    }

    #[test]
    fn test_apply_controlled_duplicate_controls() {
        let mut state = StateVector::new(3);
        let result = state.apply_controlled(&Gate::x(), &[2], &[0, 0]);
        assert!(matches!(
            result,
            Err(StateError::GateError(GateError::DuplicateQubit(0)))
        ));
    }

    #[test]
    fn test_apply_controlled_target_control_overlap() {
        let mut state = StateVector::new(2);
        let result = state.apply_controlled(&Gate::x(), &[0], &[0]);
        assert!(matches!(
            result,
            Err(StateError::GateError(GateError::ControlTargetOverlap(0)))
        ));
    }

    #[test]
    #[should_panic(expected = "Number of qubits must be at least 1")]
    fn test_new_zero_qubits() {
        StateVector::new(0);
    }
}
