use crate::channels::QuantumChannel;
use crate::core::Gate;
use crate::core::errors::{ChannelError, MeasurementError, StateError};
use crate::core::utils::{find_duplicate, trace};
use crate::measure::{Measurement, MeasurementResult};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use rand::Rng;

#[derive(Clone, Debug)]
pub struct QuantumState {
    pub density_matrix: Array2<Complex64>,
    pub num_qubits: usize,
}

impl QuantumState {
    /// Creates a new quantum state initialized to |0...0>.
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

        let tr = trace(matrix);
        if (tr - Complex64::new(1.0, 0.0)).norm() > 1e-12 {
            return Err(StateError::InvalidTrace(tr));
        }

        Ok(())
    }

    /// Apply already extended operator ot whole system
    fn apply_operator(&mut self, u: &Array2<Complex64>) -> Result<(), StateError> {
        let (rows, cols) = u.dim();
        let dim = 1 << self.num_qubits;

        if rows != dim || cols != dim {
            return Err(StateError::DimensionMismatch {
                expected: dim,
                got_rows: rows,
                got_cols: cols,
            });
        }

        let temp = u.dot(&self.density_matrix);

        let u_dagger = u.t().mapv(|x| x.conj());
        self.density_matrix = temp.dot(&u_dagger);

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

    /// Creates a QuantumState from a generic vector state.
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

    /// Creates a QuantumState from a generic density matrix.
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

    /// Checks if a QuantumState is valid.
    pub fn is_valid(&self) -> Result<(), StateError> {
        Self::check_density_matrix(&self.density_matrix)?;
        Ok(())
    }

    /// Applies non controlled quantum gate
    pub fn apply(&mut self, gate: &Gate, target_qubits: &[usize]) -> Result<(), StateError> {
        self.apply_controlled(gate, target_qubits, None)
    }

    /// Applies generic quantum gate
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

        let full_gate_operator = Gate::expand_gate(self.num_qubits, gate, target_qubits, controls)?;

        self.apply_operator(&full_gate_operator.matrix)
    }

    /// Returns the probability of each operator expanded to the whole system
    pub fn set_measurement(
        &self,
        measurement: &Measurement,
        target_qubits: &[usize],
    ) -> Result<(Vec<f64>, Vec<Array2<Complex64>>), StateError> {
        for &q in target_qubits {
            self.validate_qubit_index(q)?;
        }

        if let Some(dup) = find_duplicate(target_qubits) {
            return Err(StateError::MeasurementError(
                MeasurementError::DuplicateQubit(dup),
            ));
        }

        let expanded_ops = measurement.get_expanded_operators(self.num_qubits, target_qubits)?;

        let mut probs = Vec::with_capacity(expanded_ops.len());
        let mut sum_probs = 0.0;

        for op in &expanded_ops {
            let op_dagger = op.t().mapv(|c| c.conj());

            let temp = op.dot(&self.density_matrix);
            let unnormalized_rho_prime = temp.dot(&op_dagger);
            let tr = trace(&unnormalized_rho_prime);

            let p_k = tr.re.max(0.0);

            probs.push(p_k);
            sum_probs += p_k;
        }

        // Due to float, renormalazation of probabilities to ensure completeness
        for p in &mut probs {
            *p /= sum_probs;
        }

        Ok((probs, expanded_ops))
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

    /// Phisical measurment which changes the state irretrievably
    pub fn measure(
        &mut self,
        measurement: &Measurement,
        target_qubits: &[usize],
    ) -> Result<MeasurementResult, StateError> {
        let (probs, ops) = self.set_measurement(measurement, target_qubits)?;

        let outcome_idx = self.pick_outcome(&probs);
        let p_selected = probs[outcome_idx];

        // rho' = (M_k * rho * M_k†) / p_k
        if p_selected > 1e-12 {
            let m_k = &ops[outcome_idx];
            let m_k_dagger = m_k.t().mapv(|c| c.conj());

            // M * rho * M†
            let numerator = m_k.dot(&self.density_matrix).dot(&m_k_dagger);

            self.density_matrix = numerator.mapv(|val| val / Complex64::new(p_selected, 0.0));
        } else {
            return Err(StateError::InvalidTrace(Complex64::new(0.0, 0.0)));
        }

        Ok(MeasurementResult {
            index: outcome_idx,
            value: measurement.values[outcome_idx],
        })
    }

    /// Apply QuantumChannel to QuantumState
    pub fn apply_channel(
        &mut self,
        channel: &QuantumChannel,
        target_qubits: &[usize],
    ) -> Result<(), StateError> {
        if let Some(dup) = find_duplicate(target_qubits) {
            return Err(StateError::ChannelError(ChannelError::DuplicateQubit(dup)));
        }

        let ops = channel.get_expanded_operators(self.num_qubits, target_qubits)?;

        let dim = self.density_matrix.nrows();
        let mut new_rho = Array2::<Complex64>::zeros((dim, dim));

        // Apply Kraus operators and sum
        for k in ops {
            let k_dagger = k.t().mapv(|c| c.conj());

            // K * rho
            let temp = k.dot(&self.density_matrix);

            // (K * rho) * K†
            let term = temp.dot(&k_dagger);

            // Sum
            new_rho = new_rho + term;
        }

        self.density_matrix = new_rho;

        Ok(())
    }
}
