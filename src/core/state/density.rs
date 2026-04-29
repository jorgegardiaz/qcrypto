use crate::core::errors::{ChannelError, MeasurementError, StateError};
use crate::core::state::{
    GateApplicable, Measurable, PurityComputable, QuantumStateImpl, Validatable,
};
use crate::{Gate, Measurement, MeasurementResult, QuantumChannel, core::utils};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use rayon::prelude::*;

/// Represents a quantum state using density matrices.
///
/// The physical state is mathematically represented by a $2^N \times 2^N$ density matrix $\rho$,
/// satisfying $\rho^\dagger = \rho$, $\text{Tr}(\rho) = 1$, and $\rho \ge 0$.
#[derive(Clone, Debug)]
pub struct StateDensityMatrix {
    /// The mathematical $2^N \times 2^N$ density matrix corresponding to the mixed state.
    pub density_matrix: Array2<Complex64>,
    /// The number of qubits composing the system.
    pub num_qubits: usize,
}

impl StateDensityMatrix {
    /// Creates a new pure quantum state initialized to the ground state $|0\dots 0\rangle$.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - The number of qubits in the quantum system.
    ///
    /// # Returns
    ///
    /// A new `StateDensityMatrix` instance.
    ///
    /// # Example
    ///
    /// ```rust
    /// use qcrypto::state::StateDensityMatrix;
    /// # use num_complex::Complex64;
    ///
    /// // Initialize a 2-qubit state
    /// let state = StateDensityMatrix::new(2);
    ///
    /// // Verify qubit count
    /// assert_eq!(state.num_qubits, 2);
    ///
    /// // 2 qubits means the density matrix dimension should be 2^2 x 2^2 (4x4)
    /// assert_eq!(state.density_matrix.dim(), (4, 4));
    ///
    /// // The state is |00><00|
    /// for ((i, j), &val) in state.density_matrix.indexed_iter() {
    ///     let expected = if i == 0 && j == 0 { 1.0 } else { 0.0 };
    ///     assert_eq!(val, Complex64::new(expected, 0.0));
    /// }
    /// ```
    pub fn new(num_qubits: usize) -> Self {
        let dim = 1 << num_qubits;
        let mut density_matrix = Array2::<Complex64>::zeros((dim, dim));
        density_matrix[[0, 0]] = Complex64::new(1.0, 0.0);

        Self {
            density_matrix,
            num_qubits,
        }
    }

    /// Validates mathematically that a given pure state vector fulfills probability bounds.
    fn check_vector_state(vector: &Array1<Complex64>) -> Result<(), StateError> {
        let dim = vector.len();

        // Dimension must be a power of 2
        if !dim.is_power_of_two() {
            return Err(StateError::InvalidDimensions);
        }

        // Sum of squared amplitudes must closely approximate 1.0.
        let norm_sqr: f64 = vector.iter().map(|c| c.norm_sqr()).sum();

        if (norm_sqr - 1.0).abs() > 1e-12 {
            return Err(StateError::NotNormalized(norm_sqr));
        }

        Ok(())
    }
    /// Extrapolates that a density matrix representation accurately mirrors a quantum state.
    ///
    /// Tests for square dimension constraints and exact Trace value $= 1.0$.
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

        let tr = utils::trace(matrix);
        if (tr - Complex64::new(1.0, 0.0)).norm() > 1e-12 {
            return Err(StateError::InvalidTrace(tr));
        }

        Ok(())
    }

    /// Verifies if a user-supplied target query index matches available hardware constraints.
    fn validate_qubit_index(&self, index: usize) -> Result<(), StateError> {
        if index >= self.num_qubits {
            return Err(StateError::IndexOutOfBounds {
                index,
                num_qubits: self.num_qubits,
            });
        }
        Ok(())
    }

    /// Creates a pure `StateDensityMatrix` instantiated directly from a state vector representation.
    ///
    /// The algorithm implicitly projects the state vector |psi> into a corresponding density trace rho = |psi><psi|.
    ///
    /// # Arguments
    ///
    /// * `vector` - The one-dimensional state vector acting as structural input `Array1<Complex64>`.
    ///
    /// # Returns
    ///
    /// A `Result` containing the new `StateDensityMatrix` if successful.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if validation rules fail.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::state::StateDensityMatrix;
    /// # use ndarray::array;
    /// # use num_complex::Complex64;
    ///
    /// // |0> state vector
    /// let vec = array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    /// let state = StateDensityMatrix::from_state_vector(vec).unwrap();
    ///
    /// // Transforms to a 1 qubit density matrix |0><0|
    /// assert_eq!(state.num_qubits, 1);
    /// for ((i, j), &val) in state.density_matrix.indexed_iter() {
    ///     let expected = if i == 0 && j == 0 { 1.0 } else { 0.0 };
    ///     assert_eq!(val, Complex64::new(expected, 0.0));
    /// }
    /// ```
    pub fn from_state_vector(vector: Array1<Complex64>) -> Result<Self, StateError> {
        Self::check_vector_state(&vector)?;

        // Calculate number of qubits: dim = 2^n, so n = log2(dim)
        let dim = vector.len();
        let num_qubits = (dim as f64).log2() as usize;

        // Compute the density matrix of the pure state: rho = |psi><psi|
        let col_vector = vector
            .view()
            .into_shape_with_order((dim, 1))
            .expect("Error shaping vector into column");
        let row_vector_owned = col_vector.mapv(|c| c.conj());
        let matrix = col_vector.dot(&row_vector_owned.t());

        Ok(Self {
            density_matrix: matrix,
            num_qubits,
        })
    }

    /// Instantiates a pure algorithm representation directly originating from a custom unverified trace.
    ///
    /// # Arguments
    ///
    /// * `matrix` - The mathematically equivalent rho.
    ///
    /// # Returns
    ///
    /// A `Result` containing the new `StateDensityMatrix` if successful.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if the underlying trace does not equal 1.0 or array representations fragment.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::state::StateDensityMatrix;
    /// # use ndarray::array;
    /// # use num_complex::Complex64;
    ///
    /// // Maximally mixed state for 1 qubit: I/2
    /// let matrix = array![
    ///     [Complex64::new(0.5, 0.0), Complex64::new(0.0, 0.0)],
    ///     [Complex64::new(0.0, 0.0), Complex64::new(0.5, 0.0)]
    /// ];
    ///
    /// let state = StateDensityMatrix::from_density_matrix(matrix).unwrap();
    /// assert_eq!(state.num_qubits, 1);
    /// ```
    pub fn from_density_matrix(matrix: Array2<Complex64>) -> Result<Self, StateError> {
        Self::check_density_matrix(&matrix)?;
        let (rows, _) = matrix.dim();
        // Extract logical power of 2 using trailing zeros.
        let num_qubits = rows.trailing_zeros() as usize;

        Ok(Self {
            density_matrix: matrix,
            num_qubits,
        })
    }

    /// Checks if the underlying loaded density matrix holds mathematical validity properties.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success `Ok(())` if the state is mathematically valid.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if the density matrix dimensions are invalid or the trace is not 1.0.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::state::StateDensityMatrix;
    ///
    /// let state = StateDensityMatrix::new(2);
    /// assert!(state.is_valid().is_ok());
    /// ```
    pub fn is_valid(&self) -> Result<(), StateError> {
        Self::check_density_matrix(&self.density_matrix)?;
        Ok(())
    }

    /// Applies a quantum gate matrix structurally representing operations acting locally over specifically targeted matrices.
    ///
    /// # Arguments
    ///
    /// * `gate` - The matrix describing logical `Gate` instructions to map.
    /// * `target_qubits` - Pointers bounding execution targets on hardware.
    ///
    /// # Returns
    ///
    /// A `Result` containing a mutable reference to `Self` (`&mut Self`) to allow method chaining.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if the target qubits are out of bounds or the gate dimension mismatches.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{state::StateDensityMatrix, Gate};
    /// # use num_complex::Complex64;
    ///
    /// let mut state = StateDensityMatrix::new(1); // |0><0|
    ///
    /// // Apply NOT gate and CNOT gate
    /// let mut state = state.apply(&Gate::x(), &[0]).unwrap();
    ///
    ///
    /// // Now it should be |1><1|
    /// for ((i, j), &val) in state.density_matrix.indexed_iter() {
    ///     let expected = if i == 1 && j == 1 { 1.0 } else { 0.0 };
    ///     assert_eq!(val, Complex64::new(expected, 0.0));
    /// }
    /// ```
    pub fn apply(&mut self, gate: &Gate, target_qubits: &[usize]) -> Result<&mut Self, StateError> {
        self.apply_controlled(gate, target_qubits, &[])
    }

    /// Applies local tensor matrices using highly-performant unitary execution boundaries.
    ///
    /// This specific block executes mathematical logical evolutions equivalent to structurally simulating $\rho' = U \rho U^\dagger$.
    ///
    /// # Arguments
    ///
    /// * `gate` - The base local operation template structure (i.e Hadamard constraint vectors).
    /// * `target_qubits` - Slice targeting array pointers for action application.
    /// * `control_qubits` - Opt-in indices structurally limiting constraints according to a sequence boundary.
    ///
    /// # Returns
    ///
    /// A `Result` containing a mutable reference to `Self` (`&mut Self`) to allow method chaining.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if the target/control qubits are out of bounds or the gate dimension mismatches.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{state::StateDensityMatrix, Gate};
    /// # use num_complex::Complex64;
    ///
    /// let mut state = StateDensityMatrix::new(2);
    ///
    /// // Apply X gate to qubit 0 and CNOT -> |11>
    /// state.apply(&Gate::x(), &[0]).unwrap()
    ///      .apply_controlled(&Gate::x(), &[1], &[0]).unwrap();
    ///
    ///
    /// // The density matrix for |11><11| should have a 1.0 at index [3, 3] and 0.0 elsewhere
    /// for ((i, j), &val) in state.density_matrix.indexed_iter() {
    ///     let expected = if i == 3 && j == 3 { 1.0 } else { 0.0 };
    ///     assert_eq!(val, Complex64::new(expected, 0.0));
    /// }
    /// ```
    pub fn apply_controlled(
        &mut self,
        gate: &Gate,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<&mut Self, StateError> {
        // 1. Validate boundary dimensions
        if gate.num_qubits != target_qubits.len() {
            return Err(StateError::DimensionMismatch {
                expected: gate.num_qubits,
                got_rows: target_qubits.len(),
                got_cols: 0,
            });
        }

        // 2. Map logical indices back to hardware limitations
        for &q in target_qubits {
            self.validate_qubit_index(q)?;
        }

        for &q in control_qubits {
            self.validate_qubit_index(q)?;
        }

        // Left multiplication projection mapping rho_temp = U * rho.
        // Operation runs logically directly limiting dense sparse arrays to save explicit storage cost.
        let temp_rho = utils::apply_local_left(
            self.num_qubits,
            &self.density_matrix,
            &gate.matrix,
            target_qubits,
            control_qubits,
        );

        // Map logical conjugate transpose structural translation bounding values U_dagger.
        let u_dagger = gate.matrix.t().mapv(|c| c.conj());

        // Run structural algorithm backwards projecting density representation rho_new = rho_temp * U_dagger.
        let final_rho = utils::apply_local_right(
            self.num_qubits,
            &temp_rho,
            &u_dagger,
            target_qubits,
            control_qubits,
        );

        self.density_matrix = final_rho;

        Ok(self)
    }

    /// Calculates measurement outcome probabilities without collapsing the state.
    ///
    /// Evaluates the probability of each outcome $k$ using $p_k = \text{Tr}(M_k \rho M_k^\dagger)$.
    ///
    /// # Arguments
    ///
    /// * `measurement` - The `Measurement` protocol to evaluate (e.g. Z basis, Bell basis).
    /// * `target_qubits` - The indices of the qubits to be measured.
    ///
    /// # Returns
    ///
    /// A `Result` containing a `Vec<f64>` with the calculated probabilities for each measurement outcome.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if the number of target qubits doesn't match the measurement dimensions,
    /// if the target qubits are out of bounds, or if there are duplicate target qubits.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{state::StateDensityMatrix, Measurement};
    ///
    /// let state = StateDensityMatrix::new(1);
    ///
    /// // Calculate probabilities of measuring in Z basis
    /// let probs = state.set_measurement(&Measurement::z_basis(), &[0]).unwrap();
    ///
    /// // Since it's |0>, the probability of |0> is 1.0, and |1> is 0.0
    /// assert_eq!(probs, vec![1.0, 0.0]);
    /// ```
    pub fn set_measurement(
        &self,
        measurement: &Measurement,
        target_qubits: &[usize],
    ) -> Result<Vec<f64>, StateError> {
        // Validate measurement mapping bounds locally before execution paths block underlying architecture bounds
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

        let mut probs: Vec<f64> = measurement
            .operators
            .par_iter()
            .map(|m_k| {
                // rho_temp = M_k * rho
                let temp = utils::apply_local_left(
                    self.num_qubits,
                    &self.density_matrix,
                    m_k,
                    target_qubits,
                    &[],
                );

                // M_k_dagger
                let m_k_dagger = m_k.t().mapv(|c| c.conj());

                // unnormalized_rho = rho_temp * M_k_dagger
                let unnormalized_rho = utils::apply_local_right(
                    self.num_qubits,
                    &temp,
                    &m_k_dagger,
                    target_qubits,
                    &[],
                );

                // Calculate the true outcome boundaries mapping Trace(rho).
                let tr = utils::trace(&unnormalized_rho);
                tr.re.max(0.0)
            })
            .collect();

        let sum_probs: f64 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum_probs;
        }

        Ok(probs)
    }

    /// Randomly selects an operator index weighted according to a given generic probability sequence `probs`.
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

    /// Measures the state and collapses the density matrix according to the chosen outcome.
    ///
    /// Evaluates probabilities, randomly selects an outcome based on those probabilities,
    /// and collapses the state using $\rho \to \frac{M_k \rho M_k^\dagger}{\text{Tr}(M_k \rho M_k^\dagger)}$.
    ///
    /// # Arguments
    ///
    /// * `measurement` - The `Measurement` protocol to apply.
    /// * `target_qubits` - The indices of the qubits being measured.
    ///
    /// # Returns
    ///
    /// A `Result` containing a `MeasurementResult` (with the chosen outcome index and its logical value).
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if the measurement dimensions or target qubits are invalid,
    /// or if the resulting trace is theoretically 0.0 (an impossible measurement outcome).
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{state::StateDensityMatrix, Measurement};
    ///
    /// let mut state = StateDensityMatrix::new(1);
    ///
    /// // Measure in Z basis
    /// let result = state.measure(&Measurement::z_basis(), &[0]).unwrap();
    ///
    /// // State |0> collapes to itself giving value 0.0
    /// assert_eq!(result.value, 0.0);
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
            let m_k_dagger = m_k.t().mapv(|c| c.conj());

            let temp = utils::apply_local_left(
                self.num_qubits,
                &self.density_matrix,
                m_k,
                target_qubits,
                &[],
            );

            let numerator =
                utils::apply_local_right(self.num_qubits, &temp, &m_k_dagger, target_qubits, &[]);

            self.density_matrix = numerator.mapv(|val| val / Complex64::new(p_selected, 0.0));
        } else {
            return Err(StateError::InvalidTrace(Complex64::new(0.0, 0.0)));
        }

        Ok(MeasurementResult {
            index: outcome_idx,
            value: measurement.values[outcome_idx],
        })
    }

    /// Applies a noisy quantum channel modeled by Kraus operators to the density matrix.
    ///
    /// Computes the evolution $\rho \to \sum K_i \rho K_i^\dagger$ in parallel to simulate decoherence or noise.
    ///
    /// # Arguments
    ///
    /// * `channel` - The `QuantumChannel` containing the Kraus operators.
    /// * `target_qubits` - The indices of the qubits the channel acts upon.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success `Ok(())`.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if target qubits are out of bounds, duplicated, or mismatch the channel dimension.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{state::StateDensityMatrix, QuantumChannel};
    ///
    /// let mut state = StateDensityMatrix::new(1);
    ///
    /// // Apply depolarizing channel
    /// state.apply_channel(&QuantumChannel::depolarizing(0.5), &[0]).unwrap();
    ///
    /// // A depolarizing channel strictly limits the pure representation, lowering its purity < 1.0.
    /// assert!(state.purity() < 1.0);
    /// ```
    pub fn apply_channel(
        &mut self,
        channel: &QuantumChannel,
        target_qubits: &[usize],
    ) -> Result<(), StateError> {
        // Prevent structurally duplicating indexes across arrays
        if let Some(dup) = utils::find_duplicate(target_qubits) {
            return Err(StateError::ChannelError(ChannelError::DuplicateQubit(dup)));
        }

        if channel.num_qubits != target_qubits.len() {
            return Err(StateError::DimensionMismatch {
                expected: channel.num_qubits,
                got_rows: target_qubits.len(),
                got_cols: 0,
            });
        }

        for &q in target_qubits {
            self.validate_qubit_index(q)?;
        }

        let dim = self.density_matrix.nrows();
        let num_total_qubits = self.num_qubits;

        // Iteration path processing threads targeting isolated hardware nodes mapping explicitly array loops
        let new_rho = channel
            .kraus_ops
            .par_iter()
            .map(|k| {
                // Left mathematical tracing limiting local iterations mapping loops explicitly: rho_temp = K_i * rho
                let rho_temp = utils::apply_local_left(
                    num_total_qubits,
                    &self.density_matrix,
                    k,
                    target_qubits,
                    &[], // Noise boundaries strictly apply unstructured
                );

                // Right mapping iteration structurally bounds memory according to representations mapping values
                let k_dagger = k.t().mapv(|c| c.conj());

                // Tracing backwards processing loops implicitly mapping limits explicitly tracking structurally
                utils::apply_local_right(num_total_qubits, &rho_temp, &k_dagger, target_qubits, &[])
            })
            .reduce(
                || Array2::<Complex64>::zeros((dim, dim)), // Trace map iteration zero bounding initialization
                |acc, term| acc + term, // Binds memory tracking iterations looping boundaries maps explicitly
            );

        self.density_matrix = new_rho;

        Ok(())
    }

    /// Composes this state with an ancilla state using the Kronecker tensor product.
    ///
    /// Yields a combined larger system $\rho_{total} = \rho_{self} \otimes \rho_{ancilla}$.
    ///
    /// # Arguments
    ///
    /// * `ancilla_state` - The `StateDensityMatrix` to append to the current system.
    ///
    /// # Returns
    ///
    /// A `Result` containing the combined new `StateDensityMatrix` representing the full composite system.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if tensor operations fail (though typical structural constraints prevent this).
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::state::StateDensityMatrix;
    ///
    /// let state1 = StateDensityMatrix::new(1);
    /// let state2 = StateDensityMatrix::new(2);
    ///
    /// // Tensor product of a 1 qubit and a 2 qubit state results in a 3 qubit state
    /// let combined = state1.compose(&state2).unwrap();
    /// assert_eq!(combined.num_qubits, 3);
    /// ```
    pub fn compose(
        &self,
        ancilla_state: &StateDensityMatrix,
    ) -> Result<StateDensityMatrix, StateError> {
        let composite_matrix =
            utils::kronecker_product_matrix(&self.density_matrix, &ancilla_state.density_matrix);
        let composite_num_qubits = self.num_qubits + ancilla_state.num_qubits;

        // Output new mathematical representations explicitly mapping iterations structurally extending hardware limitations
        Ok(StateDensityMatrix {
            density_matrix: composite_matrix,
            num_qubits: composite_num_qubits,
        })
    }

    /// Computes the purity of the quantum state.
    ///
    /// Purity is defined as $\text{Tr}(\rho^2)$. A completely pure state returns `1.0`,
    /// while a mixed state returns a value `< 1.0` (down to $1/d$ for a maximally mixed state).
    ///
    /// # Returns
    ///
    /// The purity as a `f64`.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::state::StateDensityMatrix;
    ///
    /// let state = StateDensityMatrix::new(1);
    ///
    /// // A newly initialized state is 100% pure (purity == 1.0)
    /// assert_eq!(state.purity(), 1.0);
    /// ```
    pub fn purity(&self) -> f64 {
        self.density_matrix.iter().map(|c| c.norm_sqr()).sum()
    }
}

impl Validatable for StateDensityMatrix {
    fn is_valid(&self) -> Result<(), StateError> {
        self.is_valid()
    }
}

impl GateApplicable for StateDensityMatrix {
    fn apply(&mut self, gate: &Gate, target_qubits: &[usize]) -> Result<(), StateError> {
        StateDensityMatrix::apply(self, gate, target_qubits)?;
        Ok(())
    }

    fn apply_controlled(
        &mut self,
        gate: &Gate,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<(), StateError> {
        StateDensityMatrix::apply_controlled(self, gate, target_qubits, control_qubits)?;
        Ok(())
    }
}

impl Measurable for StateDensityMatrix {
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

impl PurityComputable for StateDensityMatrix {
    fn purity(&self) -> f64 {
        self.purity()
    }
}

impl QuantumStateImpl for StateDensityMatrix {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_density_matrix(&self) -> Result<StateDensityMatrix, StateError> {
        Ok(self.clone())
    }

    fn try_apply_channel(
        &mut self,
        channel: &QuantumChannel,
        target_qubits: &[usize],
    ) -> Result<bool, StateError> {
        self.apply_channel(channel, target_qubits)?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_density_matrix_errors() {
        let mat_nonsquare = Array2::<Complex64>::zeros((2, 3));
        assert!(matches!(
            StateDensityMatrix::check_density_matrix(&mat_nonsquare),
            Err(StateError::DimensionMismatch { .. })
        ));

        let mat_nonpow2 = Array2::<Complex64>::zeros((3, 3));
        assert!(matches!(
            StateDensityMatrix::check_density_matrix(&mat_nonpow2),
            Err(StateError::InvalidDimensions)
        ));

        let mat_invalid_trace = Array2::<Complex64>::zeros((2, 2));
        assert!(matches!(
            StateDensityMatrix::check_density_matrix(&mat_invalid_trace),
            Err(StateError::InvalidTrace(_))
        ));
    }

    #[test]
    fn test_from_density_matrix_success() {
        let mat = Array2::<Complex64>::zeros((2, 2));
        // Matrix needs trace 1.0
        let mut valid_mat = mat.clone();
        valid_mat[[0, 0]] = Complex64::new(1.0, 0.0);
        let result = StateDensityMatrix::from_density_matrix(valid_mat);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().num_qubits, 1);
    }

    #[test]
    fn test_set_measurement_dimension_mismatch() {
        let state = StateDensityMatrix::new(1);
        let m = Measurement::bell_basis(); // 2 qubits
        assert!(matches!(
            state.set_measurement(&m, &[0]),
            Err(StateError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_apply_channel_dimension_mismatch() {
        let mut state = StateDensityMatrix::new(1);
        let channel = QuantumChannel::new(vec![Array2::<Complex64>::eye(4)]).unwrap(); // 2 qubits
        assert!(matches!(
            state.apply_channel(&channel, &[0]),
            Err(StateError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_density_matrix_check_vector_state_errors() {
        let vec_bad_dim = Array1::<Complex64>::zeros(3);
        assert!(matches!(
            StateDensityMatrix::from_state_vector(vec_bad_dim),
            Err(StateError::InvalidDimensions)
        ));

        let vec_bad_norm = Array1::<Complex64>::zeros(2);
        assert!(matches!(
            StateDensityMatrix::from_state_vector(vec_bad_norm),
            Err(StateError::NotNormalized(_))
        ));
    }

    #[test]
    fn test_is_valid_success() {
        let state = StateDensityMatrix::new(2);
        assert!(state.is_valid().is_ok());
    }

    #[test]
    fn test_apply_channel_duplicate_qubits() {
        let mut state = StateDensityMatrix::new(1);
        let channel = QuantumChannel::bit_flip(1.0);
        assert!(matches!(
            state.apply_channel(&channel, &[0, 0]),
            Err(StateError::ChannelError(_))
        ));
    }

    #[test]
    fn test_trait_implementations() {
        let mut state = StateDensityMatrix::new(1);

        // Validatable
        assert!(Validatable::is_valid(&state).is_ok());

        // GateApplicable
        GateApplicable::apply(&mut state, &Gate::x(), &[0]).unwrap();
        GateApplicable::apply_controlled(&mut state, &Gate::x(), &[0], &[]).unwrap();

        // PurityComputable
        assert!((PurityComputable::purity(&state) - 1.0).abs() < 1e-12);

        // Measurable
        let probs = Measurable::set_measurement(&state, &Measurement::z_basis(), &[0]).unwrap();
        assert_eq!(probs.len(), 2);
        let _res = Measurable::measure(&mut state, &Measurement::z_basis(), &[0]).unwrap();
    }

    #[test]
    fn test_apply_out_of_bounds() {
        let mut state = StateDensityMatrix::new(1); // 1 qubit (index 0)

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
        let mut state = StateDensityMatrix::new(2);

        // CNOT is a 2-qubit gate, but we only give it 1 target qubit
        let result = state.apply(&Gate::cnot(), &[0]);

        assert!(matches!(result, Err(StateError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_invalid_density_matrix_trace() {
        // Create a fake matrix with trace != 1.0
        let invalid_matrix = array![
            [Complex64::new(0.5, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)] // Trace is 0.5
        ];

        let result = StateDensityMatrix::from_density_matrix(invalid_matrix);

        assert!(matches!(result, Err(StateError::InvalidTrace(_))));
    }

    #[test]
    fn test_invalid_density_matrix_dimensions() {
        // Create a 3x3 matrix (not a power of 2)
        let invalid_matrix = Array2::<Complex64>::zeros((3, 3));

        let result = StateDensityMatrix::from_density_matrix(invalid_matrix);

        assert!(matches!(result, Err(StateError::InvalidDimensions)));
    }

    #[test]
    fn test_measurement_duplicate_qubits() {
        let state = StateDensityMatrix::new(2);

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
        let mut state = StateDensityMatrix::new(1); // |0><0|
        let initial_matrix = state.density_matrix.clone();

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
        for (i, &val) in state.density_matrix.indexed_iter() {
            let diff = (val - initial_matrix[i]).norm();
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
        let mut state = StateDensityMatrix::new(2); // |00><00|

        // H on qubit 0, CNOT on 0 -> 1
        state
            .apply(&Gate::h(), &[0])
            .unwrap()
            .apply_controlled(&Gate::x(), &[1], &[0])
            .unwrap();

        // The density matrix for (|00> + |11>)/sqrt(2)
        // Indices [0,0], [0,3], [3,0], [3,3] should be 0.5
        for ((i, j), &val) in state.density_matrix.indexed_iter() {
            let expected = if (i == 0 && j == 0)
                || (i == 0 && j == 3)
                || (i == 3 && j == 0)
                || (i == 3 && j == 3)
            {
                0.5
            } else {
                0.0
            };
            let diff = (val - Complex64::new(expected, 0.0)).norm();
            assert!(diff < 1e-12, "Unexpected value at [{}, {}]", i, j);
        }
    }

    #[test]
    fn test_apply_cnot_bell_state() {
        // Same Bell state but using Gate::cnot() through apply()
        let mut state = StateDensityMatrix::new(2);
        state
            .apply(&Gate::h(), &[0])
            .unwrap()
            .apply(&Gate::cnot(), &[0, 1])
            .unwrap();

        for ((i, j), &val) in state.density_matrix.indexed_iter() {
            let expected = if (i == 0 && j == 0)
                || (i == 0 && j == 3)
                || (i == 3 && j == 0)
                || (i == 3 && j == 3)
            {
                0.5
            } else {
                0.0
            };
            let diff = (val - Complex64::new(expected, 0.0)).norm();
            assert!(diff < 1e-12, "Unexpected at [{}, {}]", i, j);
        }
    }

    #[test]
    fn test_apply_cnot_equivalence() {
        let mut state_a = StateDensityMatrix::new(2);
        state_a
            .apply(&Gate::h(), &[0])
            .unwrap()
            .apply(&Gate::cnot(), &[0, 1])
            .unwrap();

        let mut state_b = StateDensityMatrix::new(2);
        state_b
            .apply(&Gate::h(), &[0])
            .unwrap()
            .apply_controlled(&Gate::x(), &[1], &[0])
            .unwrap();

        for ((i, j), &a) in state_a.density_matrix.indexed_iter() {
            let b = state_b.density_matrix[[i, j]];
            assert!((a - b).norm() < 1e-12, "Mismatch at [{}, {}]", i, j);
        }
    }

    #[test]
    fn test_apply_swap() {
        // |10⟩⟨10| with SWAP → |01⟩⟨01|
        let mut state = StateDensityMatrix::new(2);
        state.apply(&Gate::x(), &[0]).unwrap(); // |10⟩⟨10|
        state.apply(&Gate::swap(), &[0, 1]).unwrap();

        // |01⟩⟨01| → only [1,1] = 1.0
        assert!((state.density_matrix[[1, 1]] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        assert!(state.density_matrix[[2, 2]].norm() < 1e-12);
    }

    #[test]
    fn test_measure_collapse() {
        let mut state = StateDensityMatrix::new(1);

        // Put in |+> and measure in Z basis
        let _result = state
            .apply(&Gate::h(), &[0])
            .unwrap()
            .measure(&Measurement::z_basis(), &[0])
            .unwrap();

        // Purity should be 1.0 after measurement
        assert!((state.purity() - 1.0).abs() < 1e-12);

        // Trace should be 1.0
        let trace: Complex64 = state.density_matrix.diag().sum();
        assert!((trace.re - 1.0).abs() < 1e-12);

        let mut hit_0 = false;
        let mut hit_1 = false;
        for _ in 0..20 {
            let mut state = StateDensityMatrix::new(1);

            let result = state
                .apply(&Gate::h(), &[0])
                .unwrap()
                .measure(&Measurement::z_basis(), &[0])
                .unwrap();

            if result.value == 0.0 {
                assert!((state.density_matrix[[0, 0]].re - 1.0).abs() < 1e-12);
                assert!(state.density_matrix[[1, 1]].re.abs() < 1e-12);
                hit_0 = true;
            } else {
                assert!((state.density_matrix[[1, 1]].re - 1.0).abs() < 1e-12);
                assert!(state.density_matrix[[0, 0]].re.abs() < 1e-12);
                hit_1 = true;
            }
            if hit_0 && hit_1 {
                break;
            }
        }
        assert!(hit_0 && hit_1, "Both branches should be hit eventually");

        let channel = QuantumChannel::bit_flip(1.0);
        let mut state = StateDensityMatrix::new(1);
        let mut comparison_state = StateDensityMatrix::new(1);
        state
            .apply(&Gate::h(), &[0])
            .unwrap()
            .apply_channel(&channel, &[0])
            .unwrap();

        // Store matrix before channel
        let initial_matrix = comparison_state
            .apply(&Gate::h(), &[0])
            .unwrap()
            .density_matrix
            .clone();

        // Since X|+> = |+>, the bit flip channel should leave the |+><+| density matrix completely unchanged.
        for (i, &val) in state.density_matrix.indexed_iter() {
            let diff = (val - initial_matrix[i]).norm();
            assert!(
                diff < 1e-12,
                "State modified at index {:?}, diff: {}",
                i,
                diff
            );
        }
    }

    #[test]
    fn test_pick_outcome_fallback() {
        let state = StateDensityMatrix::new(1);
        let probs = vec![0.0, 0.0];
        let idx = state.pick_outcome(&probs);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_measure_zero_prob() {
        let mut state = StateDensityMatrix::new(1);
        state.density_matrix = Array2::zeros((2, 2));
        let result = state.measure(&Measurement::z_basis(), &[0]);
        assert!(matches!(result, Err(StateError::InvalidTrace(_))));
    }
}
