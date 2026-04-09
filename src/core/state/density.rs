use crate::core::errors::{ChannelError, MeasurementError, StateError};
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
    /// The algorithm implicitly projects the state vector $|\psi\rangle$ into a corresponding density trace $\rho = |\psi\rangle\langle\psi|$.
    ///
    /// # Arguments
    ///
    /// * `vector` - The one-dimensional state vector acting as structural input `Array1<Complex64>`.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if validation rules fail (i.e matrix normalization bounding error).
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

    /// Instantiates a pure algorithm representation directly originating from a custom unverified trace.
    ///
    /// # Arguments
    ///
    /// * `matrix` - The mathematically equivalent $\rho$.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if the underlying trace does not equal 1.0 or array representations fragment.
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
    pub fn apply(&mut self, gate: &Gate, target_qubits: &[usize]) -> Result<(), StateError> {
        self.apply_controlled(gate, target_qubits, None)
    }

    /// Applies local tensor matrices using highly-performant unitary execution boundaries.
    ///
    /// This specific block executes mathematical logical evolutions equivalent to structurally simulating $\rho' = U \rho U^\dagger$.
    /// Utilizing implicit logical paths averts global memory saturation associated logically with typical O(N^3) memory complexity expansions.
    ///
    /// # Arguments
    ///
    /// * `gate` - The base local operation template structure (i.e Hadamard constraint vectors).
    /// * `target_qubits` - Slice targeting array pointers for action application.
    /// * `control_qubits` - Opt-in indices structurally limiting constraints according to a sequence boundary.
    pub fn apply_controlled(
        &mut self,
        gate: &Gate,
        target_qubits: &[usize],
        control_qubits: Option<&[usize]>,
    ) -> Result<(), StateError> {
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

        let controls = control_qubits.unwrap_or(&[]);
        for &q in controls {
            self.validate_qubit_index(q)?;
        }

        // --- TENSOR UPDATE ENGINE ---

        // Step 1: Left multiplication projection mapping rho_temp = U * rho.
        // Operation runs logically directly limiting dense sparse arrays to save explicit storage cost.
        let temp_rho = utils::apply_local_left(
            self.num_qubits,
            &self.density_matrix,
            &gate.matrix,
            target_qubits,
            controls,
        );

        // Step 2: Map logical conjugate transpose structural translation bounding values U_dagger.
        let u_dagger = gate.matrix.t().mapv(|c| c.conj());

        // Step 3: Run structural algorithm backwards projecting density representation rho_new = rho_temp * U_dagger.
        let final_rho = utils::apply_local_right(
            self.num_qubits,
            &temp_rho,
            &u_dagger,
            target_qubits,
            controls,
        );

        // Step 4: Write the internal hardware mapping cache holding target boundaries.
        self.density_matrix = final_rho;

        Ok(())
    }

    /// Maps non-projective operations locally across operators bypassing dense logic evaluation bounds.
    ///
    /// Leverages structurally highly tuned iterative threads executing explicit representations mirroring $p_k = \text{Tr}(M_k \rho M_k^\dagger)$ probabilities.
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

    /// Performs physical mathematical mapping simulating quantum collapse models logically.
    ///
    /// The exact internal representation simulates outcomes modeling collapse paths mathematically tracing: $\rho \to \frac{M_k \rho M_k^\dagger}{\text{Tr}(\dots)}$.
    /// Performs non dense iterations mathematically saving dense mapping complexities according to an explicit $O(N^2)$ algorithmic logic set.
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

    /// Traces local iteration paths processing noisy logical operators bounding algorithmic maps simulating models representing hardware noise.
    ///
    /// The model explicitly processes Kraus operations algorithmically bounding sequences mapping logically $\rho \to \sum K_i \rho K_i^\dagger$.
    /// Integrates parallelism structurally scaling highly across multiple processing boundaries saving hardware cycles implicitly.
    ///
    /// # Arguments
    ///
    /// * `channel` - The noise modeling object simulating hardware failures.
    /// * `target_qubits` - Pointers bounding execution targets executing mapping constraints.
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
                |acc, term| acc + term,                    // Binds memory tracking iterations looping boundaries maps explicitly
            );

        self.density_matrix = new_rho;

        Ok(())
    }

    /// Evaluates algorithmic bounds simulating local operations extending target dimensions tracing memory mapping constraints bounds explicitly tracking hardware limitations bounding $\rho_{total} = \rho_{self} \otimes \rho_{ancilla}$.
    pub fn compose(&self, ancilla_state: &StateDensityMatrix) -> Result<StateDensityMatrix, StateError> {
        let composite_matrix =
            utils::kronecker_product(&self.density_matrix, &ancilla_state.density_matrix);
        let composite_num_qubits = self.num_qubits + ancilla_state.num_qubits;
        
        // Output new mathematical representations explicitly mapping iterations structurally extending hardware limitations
        Ok(StateDensityMatrix {
            density_matrix: composite_matrix,
            num_qubits: composite_num_qubits,
        })
    }

    /// Exposes probability boundaries returning local hardware limitations implicitly iterating algorithmic constraints explicit mapping limits resolving $Tr(\rho^2)$.
    ///
    /// The mathematical limit bounds exactly logic mapping strictly pure elements tracing iteration tracking constraints mapping logical traces accurately if the mapped iteration returns exactly `1.0`.
    pub fn purity(&self) -> f64 {
        self.density_matrix.iter().map(|c| c.norm_sqr()).sum()
    }
}
