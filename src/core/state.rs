use crate::QuantumChannel;
use crate::core::Gate;
use crate::core::errors::{ChannelError, MeasurementError, StateError};
use crate::core::utils;
use crate::{Measurement, MeasurementResult};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use rand::Rng;
use rayon::prelude::*;

/// Represents a quantum state using density matrices.
///
/// The state is represented by a 2^N x 2^N density matrix $\rho$,
/// satisfying $\rho^\dagger = \rho$, $\text{Tr}(\rho) = 1$, and $\rho \ge 0$.
#[derive(Clone, Debug)]
pub struct QuantumState {
    /// The density matrix representing the state.
    pub density_matrix: Array2<Complex64>,
    /// The number of qubits in the state.
    pub num_qubits: usize,
}

impl QuantumState {
    /// Creates a new quantum state initialized to the ground state |0...0>.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - The number of qubits in the system.
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

        let tr = utils::trace(matrix);
        if (tr - Complex64::new(1.0, 0.0)).norm() > 1e-12 {
            return Err(StateError::InvalidTrace(tr));
        }

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

    /// Creates a `QuantumState` from a state vector (pure state).
    ///
    /// Converts a state vector $|\psi\rangle$ into a density matrix $\rho = |\psi\rangle\langle\psi|$.
    ///
    /// # Arguments
    ///
    /// * `vector` - The state vector as an `Array1<Complex64>`.
    ///
    /// # Errors
    ///
    /// Returns `StateError` if the vector is not normalized or has invalid dimensions.
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

    /// Creates a `QuantumState` directly from a density matrix.
    ///
    /// # Arguments
    ///
    /// * `matrix` - The density matrix $\rho$.
    ///
    /// # Errors
    ///
    /// Returns `StateError` if the matrix is invalid (utils::trace != 1, invalid dims).
    pub fn from_density_matrix(matrix: Array2<Complex64>) -> Result<Self, StateError> {
        Self::check_density_matrix(&matrix)?;
        let (rows, _) = matrix.dim();
        // log_2 as rows is power of two
        let num_qubits = rows.trailing_zeros() as usize;

        Ok(Self {
            density_matrix: matrix,
            num_qubits,
        })
    }

    /// Checks if the current state is valid.
    pub fn is_valid(&self) -> Result<(), StateError> {
        Self::check_density_matrix(&self.density_matrix)?;
        Ok(())
    }

    /// Applies a quantum gate to the specified target qubits.
    ///
    /// # Arguments
    ///
    /// * `gate` - The gate to apply.
    /// * `target_qubits` - The qubits the gate acts on.
    pub fn apply(&mut self, gate: &Gate, target_qubits: &[usize]) -> Result<(), StateError> {
        self.apply_controlled(gate, target_qubits, None)
    }

    /// Applies a quantum gate with optional control qubits using local tensor updates.
    ///
    /// This method updates the density matrix according to the unitary evolution:
    /// $\rho' = U \rho U^\dagger$.
    /// By using local operations, it avoids O(N^3) global matrix multiplications.
    ///
    /// # Arguments
    ///
    /// * `gate` - The local gate to apply (e.g., Pauli-X, Hadamard).
    /// * `target_qubits` - The target qubits.
    /// * `control_qubits` - Optional control qubits.
    pub fn apply_controlled(
        &mut self,
        gate: &Gate,
        target_qubits: &[usize],
        control_qubits: Option<&[usize]>,
    ) -> Result<(), StateError> {
        // 1. Validate dimensions
        if gate.num_qubits != target_qubits.len() {
            return Err(StateError::DimensionMismatch {
                expected: gate.num_qubits,
                got_rows: target_qubits.len(),
                got_cols: 0,
            });
        }

        // 2. Validate qubit indices
        for &q in target_qubits {
            self.validate_qubit_index(q)?;
        }

        let controls = control_qubits.unwrap_or(&[]);
        for &q in controls {
            self.validate_qubit_index(q)?;
        }

        // --- TENSOR UPDATE ENGINE ---

        // Step 1: Left multiplication -> rho_temp = U * rho
        // We use the local matrix of the gate directly.
        let temp_rho = utils::apply_local_left(
            self.num_qubits,
            &self.density_matrix,
            &gate.matrix,
            target_qubits,
            controls,
        );

        // Step 2: Calculate U_dagger (Conjugate Transpose)
        let u_dagger = gate.matrix.t().mapv(|c| c.conj());

        // Step 3: Right multiplication -> rho_new = rho_temp * U_dagger
        let final_rho = utils::apply_local_right(
            self.num_qubits,
            &temp_rho,
            &u_dagger,
            target_qubits,
            controls,
        );

        // Step 4: Update the system's density matrix
        self.density_matrix = final_rho;

        Ok(())
    }

    /// Calculates measurement probabilities without collapsing the state.
    ///
    /// Uses parallel local tensor updates to compute $p_k = \text{Tr}(M_k \rho M_k^\dagger)$
    /// efficiently without expanding the measurement operators.
    pub fn set_measurement(
        &self,
        measurement: &Measurement,
        target_qubits: &[usize],
    ) -> Result<Vec<f64>, StateError> {
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

                // p_k = Tr(unnormalized_rho)
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

    /// Randomly selects operator index ponderating using `probs`
    fn pick_outcome(&self, probs: &[f64]) -> usize {
        let mut rng = rand::rng();
        let roll: f64 = rng.random();

        let mut cumulative = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if roll < cumulative {
                return i;
            }
        }
        probs.len().saturating_sub(1)
    }

    /// Performs a physical measurement, collapsing the state.
    ///
    /// The state is updated according to the measurement outcome: $\rho \to \frac{M_k \rho M_k^\dagger}{\text{Tr}(\dots)}$.
    /// Uses O(N^2) local tensor updates instead of dense matrix multiplications.
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

    /// Applies a quantum channel (noise model) using local tensor updates and parallel processing.
    ///
    /// The evolution follows the Kraus representation: $\rho \to \sum K_i \rho K_i^\dagger$.
    /// This implementation leverages Rayon to parallelize the application of each Kraus operator
    /// and uses bitwise local updates to maintain an O(N^2) complexity per operator.
    ///
    /// # Arguments
    ///
    /// * `channel` - The quantum channel containing the Kraus operators.
    /// * `target_qubits` - The indices of the qubits affected by the noise.
    pub fn apply_channel(
        &mut self,
        channel: &QuantumChannel,
        target_qubits: &[usize],
    ) -> Result<(), StateError> {
        // Validate that there are no duplicate qubit indices in the targets
        if let Some(dup) = utils::find_duplicate(target_qubits) {
            return Err(StateError::ChannelError(ChannelError::DuplicateQubit(dup)));
        }

        let dim = self.density_matrix.nrows();
        let num_total_qubits = self.num_qubits;

        // Use Rayon to parallelize the summation of Kraus terms
        let new_rho = channel
            .kraus_ops
            .par_iter() // Parallel iteration over the small local Kraus matrices
            .map(|k| {
                // 1. Left multiplication: rho_temp = K_i * rho
                let rho_temp = utils::apply_local_left(
                    num_total_qubits,
                    &self.density_matrix,
                    k,
                    target_qubits,
                    &[], // Noise channels typically don't have control qubits
                );

                // 2. Compute the adjoint (conjugate transpose) of the Kraus operator
                let k_dagger = k.t().mapv(|c| c.conj());

                // 3. Right multiplication: result = rho_temp * K_i_dagger
                utils::apply_local_right(num_total_qubits, &rho_temp, &k_dagger, target_qubits, &[])
            })
            .reduce(
                || Array2::<Complex64>::zeros((dim, dim)), // Identity for the summation (zero matrix)
                |acc, term| acc + term,                    // Sum the density matrix terms
            );

        // Update the state's density matrix with the result of the channel evolution
        self.density_matrix = new_rho;

        Ok(())
    }

    /// Composes the `QuantumState` with another state (tensor product).
    ///
    /// Creates a composite system $\rho_{total} = \rho_{self} \otimes \rho_{ancilla}$.
    pub fn compose(&self, ancilla_state: &QuantumState) -> Result<QuantumState, StateError> {
        let composite_matrix =
            utils::kronecker_product(&self.density_matrix, &ancilla_state.density_matrix);
        let composite_num_qubits = self.num_qubits + ancilla_state.num_qubits;
        // Returns a different QuantumState
        Ok(QuantumState {
            density_matrix: composite_matrix,
            num_qubits: composite_num_qubits,
        })
    }

    /// Gives the purity of a QuantumState $Tr(\rho^2)$
    ///
    /// The state is pure if and only if $Tr(\rho^2)$
    pub fn purity(&self) -> f64 {
        self.density_matrix.iter().map(|c| c.norm_sqr()).sum()
    }
}
