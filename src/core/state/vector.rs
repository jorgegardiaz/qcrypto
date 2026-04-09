use crate::core::errors::{MeasurementError, StateError};
use crate::{Gate, Measurement, MeasurementResult, core::utils};
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
    pub fn new(num_qubits: usize) -> Self {
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

    /// Checks if the current state vector is mathematically valid (normalized to 1.0).
    pub fn is_valid(&self) -> Result<(), StateError> {
        Self::check_vector_state(&self.amplitudes)
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
    /// This performs local unitary evolution $U |\psi\rangle$ specifically restricted
    /// to the requested local subsystem to avoid global tensor memory allocation.
    ///
    /// # Arguments
    ///
    /// * `gate` - The quantum gate to apply.
    /// * `target_qubits` - The indices of the target qubits.
    /// * `control_qubits` - Optional slice with the indices of the control qubits.
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

        // Apply local operator over specific target qubits using the underlying utils engine
        self.amplitudes = utils::apply_local_vector(
            self.num_qubits,
            &self.amplitudes,
            &gate.matrix,
            target_qubits,
            controls,
        );

        Ok(())
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
    /// A vector containing the floating point probability mapping to each measurement operator.
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
    /// A `MeasurementResult` tracking both the index of the outcome operator alongside its generic value.
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
    pub fn purity(&self) -> f64 {
        1.0
    }
}
