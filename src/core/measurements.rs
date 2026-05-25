use crate::core::errors::MeasurementError;
use crate::core::utils;
use ndarray::{Array1, Array2, array};
use num_complex::Complex64;

/// Represents a general quantum measurement.
///
/// A measurement is defined by a set of operators $\{M_k\}$ such that $\sum M_k^\dagger M_k = I$.
/// Each outcome has an associated eigenvalue (physical value used in protocols) and a string label
/// (used as the key in [`Sampler`](crate::Sampler) result maps).
#[derive(Clone, Debug)]
pub struct Measurement {
    /// List of measurement operators (Kraus operators).
    pub operators: Vec<Array2<Complex64>>,
    /// Physical eigenvalue associated with each outcome.
    pub eigenvalues: Vec<f64>,
    /// String label for each outcome, used as HashMap keys by the Sampler.
    pub labels: Vec<String>,
    /// Number of qubits the measurement acts on.
    pub num_qubits: usize,
}

impl Measurement {
    /// Creates a new `Measurement` from a set of operators, eigenvalues, and labels.
    ///
    /// # Arguments
    ///
    /// * `operators` - A vector of `Array2<Complex64>` representing the measurement operators.
    /// * `eigenvalues` - Physical values associated with each outcome (used in protocols).
    /// * `labels` - String labels for each outcome (used as keys in Sampler result maps).
    ///
    /// # Returns
    ///
    /// A `Result` containing the constructed `Measurement`.
    ///
    /// # Errors
    ///
    /// Returns `MeasurementError` if:
    /// - The counts of operators, eigenvalues, and labels do not all match.
    /// - The operators are not of correct dimensions.
    /// - The operators do not satisfy the completeness relation ($\sum M_k^\dagger M_k = I$).
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::Measurement;
    /// use ndarray::Array2;
    /// use num_complex::Complex64;
    ///
    /// // Identity as a single trivial measurement
    /// let eye: Array2<Complex64> = Array2::eye(2);
    /// let m = Measurement::new(vec![eye], vec![0.0], vec!["0".to_string()]).unwrap();
    /// assert_eq!(m.num_qubits, 1);
    /// ```
    pub fn new(
        operators: Vec<Array2<Complex64>>,
        eigenvalues: Vec<f64>,
        labels: Vec<String>,
    ) -> Result<Self, MeasurementError> {
        if operators.len() != eigenvalues.len() {
            return Err(MeasurementError::CountMismatch {
                ops: operators.len(),
                vals: eigenvalues.len(),
            });
        }

        if labels.len() != eigenvalues.len() {
            return Err(MeasurementError::CountMismatch {
                ops: labels.len(),
                vals: eigenvalues.len(),
            });
        }

        if operators.is_empty() {
            return Err(MeasurementError::InvalidDimensions);
        }

        let (rows, cols) = operators[0].dim();
        if rows != cols || !rows.is_power_of_two() {
            return Err(MeasurementError::InvalidDimensions);
        }
        // log_2 as rows is power of two
        let num_qubits = rows.trailing_zeros() as usize;

        for op in &operators {
            if op.dim() != (rows, cols) {
                return Err(MeasurementError::InvalidDimensions);
            }
        }

        if !utils::check_kraus_completeness(&operators, rows) {
            return Err(MeasurementError::NotComplete);
        }

        Ok(Self {
            operators,
            eigenvalues,
            labels,
            num_qubits,
        })
    }

    /// Creates a valid Measurement from a given POVM (Positive Operator-Valued Measure).
    ///
    /// # Arguments
    ///
    /// * `povm_elements` - A vector of POVM elements $\{E_k\}$ where each $E_k$ is positive semi-definite and $\sum E_k = I$.
    /// * `eigenvalues` - Physical values associated with each POVM element (used in protocols).
    /// * `labels` - String labels for each outcome (used as keys in Sampler result maps).
    ///
    /// # Returns
    ///
    /// A `Result` containing the constructed `Measurement`.
    ///
    /// # Errors
    ///
    /// Returns `MeasurementError` if:
    /// - The elements dimensions are invalid or mismatched.
    /// - The elements do not sum to Identity.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::Measurement;
    /// use ndarray::Array2;
    /// use num_complex::Complex64;
    ///
    /// // Z-basis POVM: |0><0| and |1><1|
    /// let p0 = Array2::from_diag(&ndarray::array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
    /// let p1 = Array2::from_diag(&ndarray::array![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]);
    /// let m = Measurement::from_povm(vec![p0, p1], vec![0.0, 1.0], vec!["0".to_string(), "1".to_string()]).unwrap();
    /// assert_eq!(m.num_qubits, 1);
    /// ```
    pub fn from_povm(
        povm_elements: Vec<Array2<Complex64>>,
        eigenvalues: Vec<f64>,
        labels: Vec<String>,
    ) -> Result<Self, MeasurementError> {
        if povm_elements.len() != eigenvalues.len() {
            return Err(MeasurementError::CountMismatch {
                ops: povm_elements.len(),
                vals: eigenvalues.len(),
            });
        }

        if labels.len() != eigenvalues.len() {
            return Err(MeasurementError::CountMismatch {
                ops: labels.len(),
                vals: eigenvalues.len(),
            });
        }

        if povm_elements.is_empty() {
            return Err(MeasurementError::InvalidDimensions);
        }

        let (rows, cols) = povm_elements[0].dim();

        if rows != cols || !rows.is_power_of_two() {
            return Err(MeasurementError::InvalidDimensions);
        }

        // log_2
        let num_qubits = rows.trailing_zeros() as usize;

        if !utils::check_povm_completeness(&povm_elements, rows) {
            return Err(MeasurementError::NotComplete);
        }

        let kraus_ops = povm_elements
            .iter()
            .map(utils::sqrt_positive_matrix)
            .collect();

        Ok(Measurement {
            operators: kraus_ops,
            eigenvalues,
            labels,
            num_qubits,
        })
    }

    /// Expands the measurement operators to act on a larger system.
    ///
    /// # Arguments
    ///
    /// * `num_total_qubits` - The size of the full system.
    /// * `targets` - The indices of the qubits this measurement applies to.
    ///
    /// # Returns
    ///
    /// A `Result` containing the expanded operators.
    ///
    /// # Errors
    ///
    /// Returns `MeasurementError` if the number of targets does not match `num_qubits`.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::Measurement;
    ///
    /// let m = Measurement::z_basis();
    /// let expanded = m.get_expanded_operators(2, &[0]).unwrap();
    /// assert_eq!(expanded[0].dim(), (4, 4));
    /// ```
    pub fn get_expanded_operators(
        &self,
        num_total_qubits: usize,
        targets: &[usize],
    ) -> Result<Vec<Array2<Complex64>>, MeasurementError> {
        if targets.len() != self.num_qubits {
            return Err(MeasurementError::InvalidDimensions); // Or create a TargetMismatch error
        }

        let mut expanded_ops = Vec::with_capacity(self.operators.len());

        for op in &self.operators {
            let full_op = utils::expand_operator(op, num_total_qubits, targets, &[]);
            expanded_ops.push(full_op);
        }

        Ok(expanded_ops)
    }

    /// Creates a measurement in the Z basis (Computational basis) -> {|0>, |1>}.
    ///
    /// # Returns
    ///
    /// A single-qubit Z-basis measurement.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{QuantumState, Measurement};
    ///
    /// let state = QuantumState::new(1); // |0>
    /// let probs = state.set_measurement(&Measurement::z_basis(), &[0]).unwrap();
    /// assert_eq!(probs, vec![1.0, 0.0]);
    /// ```
    pub fn z_basis() -> Measurement {
        let v0: Array1<Complex64> = array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let v1: Array1<Complex64> = array![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];

        let p0 = utils::outer_product(&v0, &v0);
        let p1 = utils::outer_product(&v1, &v1);

        Measurement::new(
            vec![p0, p1],
            vec![1.0, -1.0],
            vec!["0".to_string(), "1".to_string()],
        )
        .expect("Error in basis Z")
    }

    /// Creates a measurement in the X basis (Hadamard basis) -> {|+>, |->}.
    ///
    /// # Returns
    ///
    /// A single-qubit X-basis measurement.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{QuantumState, Gate, Measurement};
    ///
    /// let mut state = QuantumState::new(1);
    /// state.apply(&Gate::h(), &[0]).unwrap(); // |+>
    /// let probs = state.set_measurement(&Measurement::x_basis(), &[0]).unwrap();
    /// assert!((probs[0] - 1.0).abs() < 1e-12);
    /// ```
    pub fn x_basis() -> Measurement {
        let inv_sqrt2 = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);

        let v_plus: Array1<Complex64> = array![inv_sqrt2, inv_sqrt2];
        let v_minus: Array1<Complex64> = array![inv_sqrt2, -inv_sqrt2];

        let p_plus = utils::outer_product(&v_plus, &v_plus);
        let p_minus = utils::outer_product(&v_minus, &v_minus);

        Measurement::new(
            vec![p_plus, p_minus],
            vec![1.0, -1.0],
            vec!["0".to_string(), "1".to_string()],
        )
        .expect("Error in basis X")
    }

    /// Creates a measurement in the Y basis -> {|+i>, |-i>}.
    ///
    /// # Returns
    ///
    /// A single-qubit Y-basis measurement.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::Measurement;
    ///
    /// let m = Measurement::y_basis();
    /// assert_eq!(m.num_qubits, 1);
    /// assert_eq!(m.operators.len(), 2);
    /// ```
    pub fn y_basis() -> Measurement {
        let inv_sqrt2 = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        let i_inv_sqrt2 = Complex64::new(0.0, 1.0 / 2.0_f64.sqrt());

        let v_plus_i: Array1<Complex64> = array![inv_sqrt2, i_inv_sqrt2];
        let v_minus_i: Array1<Complex64> = array![inv_sqrt2, -i_inv_sqrt2];

        let p_plus_i = utils::outer_product(&v_plus_i, &v_plus_i);
        let p_minus_i = utils::outer_product(&v_minus_i, &v_minus_i);

        Measurement::new(
            vec![p_plus_i, p_minus_i],
            vec![1.0, -1.0],
            vec!["0".to_string(), "1".to_string()],
        )
        .expect("Error in basis Y")
    }

    /// Creates a joint 2-qubit measurement in the mathematically inseparable Bell Basis.
    ///
    /// Outcomes map directly to the 4 maximally entangled Bell states:
    /// - 0: $\Phi^+$ (Phi Plus)
    /// - 1: $\Psi^+$ (Psi Plus)
    /// - 2: $\Phi^-$ (Phi Minus)
    /// - 3: $\Psi^-$ (Psi Minus)
    ///
    /// # Returns
    ///
    /// A two-qubit Bell-basis measurement.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::Measurement;
    ///
    /// let m = Measurement::bell_basis();
    /// assert_eq!(m.num_qubits, 2);
    /// assert_eq!(m.operators.len(), 4);
    /// ```
    pub fn bell_basis() -> Measurement {
        let inv_sqrt2 = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        let zero = Complex64::new(0.0, 0.0);

        // |Phi+> = (|00> + |11>) / sqrt(2)
        let v_phi_plus: Array1<Complex64> = array![inv_sqrt2, zero, zero, inv_sqrt2];
        // |Psi+> = (|01> + |10>) / sqrt(2)
        let v_psi_plus: Array1<Complex64> = array![zero, inv_sqrt2, inv_sqrt2, zero];
        // |Phi-> = (|00> - |11>) / sqrt(2)
        let v_phi_minus: Array1<Complex64> = array![inv_sqrt2, zero, zero, -inv_sqrt2];
        // |Psi-> = (|01> - |10>) / sqrt(2)
        let v_psi_minus: Array1<Complex64> = array![zero, inv_sqrt2, -inv_sqrt2, zero];

        let p_phi_plus = utils::outer_product(&v_phi_plus, &v_phi_plus);
        let p_psi_plus = utils::outer_product(&v_psi_plus, &v_psi_plus);
        let p_phi_minus = utils::outer_product(&v_phi_minus, &v_phi_minus);
        let p_psi_minus = utils::outer_product(&v_psi_minus, &v_psi_minus);

        Measurement::new(
            vec![p_phi_plus, p_psi_plus, p_phi_minus, p_psi_minus],
            vec![0.0, 1.0, 2.0, 3.0],
            vec![
                "Φ+".to_string(),
                "Ψ+".to_string(),
                "Φ-".to_string(),
                "Ψ-".to_string(),
            ],
        )
        .expect("Error in Bell basis")
    }

    /// Returns a new measurement that acts on the joint system `self ⊗ other`.
    ///
    /// The resulting measurement has:
    /// - `operators[i*n + j] = self.operators[i] ⊗ other.operators[j]`
    /// - `eigenvalues[i*n + j] = self.eigenvalues[i] * other.eigenvalues[j]`
    /// - `labels[i*n + j] = self.labels[i] + &other.labels[j]`
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::Measurement;
    ///
    /// let zz = Measurement::z_basis().compose(&Measurement::z_basis());
    /// assert_eq!(zz.num_qubits, 2);
    /// assert_eq!(zz.labels, vec!["00", "01", "10", "11"]);
    /// ```
    pub fn compose(&self, other: &Measurement) -> Measurement {
        let n = self.operators.len() * other.operators.len();
        let mut operators = Vec::with_capacity(n);
        let mut eigenvalues = Vec::with_capacity(n);
        let mut labels = Vec::with_capacity(n);

        for (i, op_a) in self.operators.iter().enumerate() {
            for (j, op_b) in other.operators.iter().enumerate() {
                operators.push(utils::kronecker_product_matrix(op_a, op_b));
                eigenvalues.push(self.eigenvalues[i] * other.eigenvalues[j]);
                labels.push(format!("{}{}", self.labels[i], other.labels[j]));
            }
        }

        Measurement {
            operators,
            eigenvalues,
            labels,
            num_qubits: self.num_qubits + other.num_qubits,
        }
    }
}

/// The result of a quantum measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementResult {
    /// The index of the outcome (and operator) that occurred.
    pub index: usize,
    /// The physical eigenvalue associated with the outcome.
    pub value: f64,
    /// The string label associated with the outcome.
    pub label: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Gate, QuantumState};
    use ndarray::Array2;

    // --- Measurement::new boundary tests ---

    fn s(v: &str) -> String {
        v.to_string()
    }

    #[test]
    fn test_new_count_mismatch() {
        let eye: Array2<Complex64> = Array2::eye(2);
        let result = Measurement::new(vec![eye], vec![0.0, 1.0], vec![s("0"), s("1")]);
        assert!(matches!(
            result,
            Err(MeasurementError::CountMismatch { .. })
        ));
    }

    #[test]
    fn test_new_labels_count_mismatch() {
        let eye: Array2<Complex64> = Array2::eye(2);
        let result = Measurement::new(vec![eye], vec![0.0], vec![s("0"), s("1")]);
        assert!(matches!(
            result,
            Err(MeasurementError::CountMismatch { .. })
        ));
    }

    #[test]
    fn test_new_empty_operators() {
        let result = Measurement::new(vec![], vec![], vec![]);
        assert!(matches!(result, Err(MeasurementError::InvalidDimensions)));
    }

    #[test]
    fn test_new_non_square_operator() {
        let matrix = Array2::from_shape_vec((2, 3), vec![Complex64::new(1.0, 0.0); 6]).unwrap();
        let result = Measurement::new(vec![matrix], vec![0.0], vec![s("0")]);
        assert!(matches!(result, Err(MeasurementError::InvalidDimensions)));
    }

    #[test]
    fn test_new_non_power_of_two() {
        let matrix: Array2<Complex64> = Array2::eye(3);
        let result = Measurement::new(vec![matrix], vec![0.0], vec![s("0")]);
        assert!(matches!(result, Err(MeasurementError::InvalidDimensions)));
    }

    #[test]
    fn test_new_not_complete() {
        let p0 = Array2::from_diag(&array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
        let result = Measurement::new(vec![p0], vec![0.0], vec![s("0")]);
        assert!(matches!(result, Err(MeasurementError::NotComplete)));
    }

    #[test]
    fn test_new_mismatched_operator_sizes() {
        let k0: Array2<Complex64> = Array2::eye(2);
        let k1: Array2<Complex64> = Array2::eye(4);
        let result = Measurement::new(vec![k0, k1], vec![0.0, 1.0], vec![s("0"), s("1")]);
        assert!(matches!(result, Err(MeasurementError::InvalidDimensions)));
    }

    // --- from_povm boundary tests ---

    #[test]
    fn test_from_povm_count_mismatch() {
        let eye: Array2<Complex64> = Array2::eye(2);
        let result = Measurement::from_povm(vec![eye], vec![0.0, 1.0], vec![s("0"), s("1")]);
        assert!(matches!(
            result,
            Err(MeasurementError::CountMismatch { .. })
        ));
    }

    #[test]
    fn test_from_povm_labels_count_mismatch() {
        let p0 = Array2::from_diag(&array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
        let p1 = Array2::from_diag(&array![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]);
        let result = Measurement::from_povm(vec![p0, p1], vec![0.0, 1.0], vec![s("0")]);
        assert!(matches!(
            result,
            Err(MeasurementError::CountMismatch { .. })
        ));
    }

    #[test]
    fn test_from_povm_empty() {
        let result = Measurement::from_povm(vec![], vec![], vec![]);
        assert!(matches!(result, Err(MeasurementError::InvalidDimensions)));
    }

    #[test]
    fn test_from_povm_non_square() {
        let matrix = Array2::from_shape_vec((2, 3), vec![Complex64::new(1.0, 0.0); 6]).unwrap();
        let result = Measurement::from_povm(vec![matrix], vec![0.0], vec![s("0")]);
        assert!(matches!(result, Err(MeasurementError::InvalidDimensions)));
    }

    #[test]
    fn test_from_povm_non_power_of_two() {
        let matrix: Array2<Complex64> = Array2::eye(3);
        let result = Measurement::from_povm(vec![matrix], vec![0.0], vec![s("0")]);
        assert!(matches!(result, Err(MeasurementError::InvalidDimensions)));
    }

    #[test]
    fn test_from_povm_not_complete() {
        let p0 = Array2::from_diag(&array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
        let result = Measurement::from_povm(vec![p0], vec![0.0], vec![s("0")]);
        assert!(matches!(result, Err(MeasurementError::NotComplete)));
    }

    #[test]
    fn test_from_povm_success() {
        let p0 = Array2::from_diag(&array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
        let p1 = Array2::from_diag(&array![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]);
        let m = Measurement::from_povm(vec![p0, p1], vec![0.0, 1.0], vec![s("0"), s("1")]).unwrap();
        assert_eq!(m.num_qubits, 1);
        assert_eq!(m.operators.len(), 2);
    }

    // --- get_expanded_operators boundary test ---

    #[test]
    fn test_expanded_operators_target_mismatch() {
        let m = Measurement::z_basis();
        let result = m.get_expanded_operators(2, &[0, 1]);
        assert!(matches!(result, Err(MeasurementError::InvalidDimensions)));
    }

    #[test]
    fn test_expanded_operators_success() {
        let m = Measurement::z_basis();
        let expanded = m.get_expanded_operators(2, &[0]).unwrap();
        assert_eq!(expanded.len(), 2);
        assert_eq!(expanded[0].dim(), (4, 4));
        assert_eq!(expanded[1].dim(), (4, 4));
    }

    // --- Standard bases: construction + completeness ---

    fn assert_complete(m: &Measurement) {
        let dim = m.operators[0].dim().0;
        let eye = Array2::<Complex64>::eye(dim);
        let mut sum = Array2::<Complex64>::zeros((dim, dim));
        for op in &m.operators {
            let dag = op.t().mapv(|x| x.conj());
            sum = sum + dag.dot(op);
        }
        for (a, b) in sum.iter().zip(eye.iter()) {
            assert!((*a - *b).norm() < 1e-10);
        }
    }

    #[test]
    fn test_z_basis_properties() {
        let m = Measurement::z_basis();
        assert_eq!(m.num_qubits, 1);
        assert_eq!(m.operators.len(), 2);
        assert_complete(&m);
    }

    #[test]
    fn test_x_basis_properties() {
        let m = Measurement::x_basis();
        assert_eq!(m.num_qubits, 1);
        assert_eq!(m.operators.len(), 2);
        assert_complete(&m);
    }

    #[test]
    fn test_y_basis_properties() {
        let m = Measurement::y_basis();
        assert_eq!(m.num_qubits, 1);
        assert_eq!(m.operators.len(), 2);
        assert_complete(&m);
    }

    #[test]
    fn test_bell_basis_properties() {
        let m = Measurement::bell_basis();
        assert_eq!(m.num_qubits, 2);
        assert_eq!(m.operators.len(), 4);
        assert_complete(&m);
    }

    // --- Orthogonality ---

    #[test]
    fn test_z_basis_operators_are_orthogonal() {
        let m = Measurement::z_basis();
        let product = m.operators[0].dot(&m.operators[1]);
        for &val in product.iter() {
            assert!(val.norm() < 1e-12);
        }
    }

    #[test]
    fn test_bell_basis_operators_are_orthogonal() {
        let m = Measurement::bell_basis();
        for i in 0..4 {
            for j in (i + 1)..4 {
                let product = m.operators[i].dot(&m.operators[j]);
                for &val in product.iter() {
                    assert!(
                        val.norm() < 1e-12,
                        "Bell operators {} and {} not orthogonal",
                        i,
                        j
                    );
                }
            }
        }
    }

    // --- Deterministic measurements: eigenstate in its own basis ---

    #[test]
    fn test_z_basis_deterministic_on_eigenstates() {
        // |0> measured in Z → outcome 0 with probability 1
        let mut state = QuantumState::new(1);
        let probs = state
            .set_measurement(&Measurement::z_basis(), &[0])
            .unwrap();
        assert!((probs[0] - 1.0).abs() < 1e-12);
        assert!(probs[1].abs() < 1e-12);

        // |1> measured in Z → outcome 1 with probability 1
        state.apply(&Gate::x(), &[0]).unwrap();
        let probs = state
            .set_measurement(&Measurement::z_basis(), &[0])
            .unwrap();
        assert!(probs[0].abs() < 1e-12);
        assert!((probs[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_x_basis_deterministic_on_eigenstates() {
        // |+> measured in X → outcome 0 with probability 1
        let mut state = QuantumState::new(1);
        state.apply(&Gate::h(), &[0]).unwrap(); // |+>
        let probs = state
            .set_measurement(&Measurement::x_basis(), &[0])
            .unwrap();
        assert!((probs[0] - 1.0).abs() < 1e-12);
        assert!(probs[1].abs() < 1e-12);

        // |-> measured in X → outcome 1 with probability 1
        let mut state2 = QuantumState::new(1);
        state2.apply(&Gate::x(), &[0]).unwrap();
        state2.apply(&Gate::h(), &[0]).unwrap(); // X then H = |->
        let probs = state2
            .set_measurement(&Measurement::x_basis(), &[0])
            .unwrap();
        assert!(probs[0].abs() < 1e-12);
        assert!((probs[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_y_basis_deterministic_on_eigenstates() {
        // |+i> = S·H|0> measured in Y → outcome 0 with probability 1
        let mut state = QuantumState::new(1);
        state
            .apply(&Gate::h(), &[0])
            .unwrap()
            .apply(&Gate::s(), &[0])
            .unwrap(); // |+i>
        let probs = state
            .set_measurement(&Measurement::y_basis(), &[0])
            .unwrap();
        assert!((probs[0] - 1.0).abs() < 1e-12);
        assert!(probs[1].abs() < 1e-12);
    }

    #[test]
    fn test_bell_basis_deterministic_on_phi_plus() {
        // |\Phi+> = (|00> + |11>) / \sqrt{2} measured in Bell basis → outcome 0
        let mut state = QuantumState::new(2);
        state
            .apply(&Gate::h(), &[0])
            .unwrap()
            .apply_controlled(&Gate::x(), &[1], &[0])
            .unwrap();

        let probs = state
            .set_measurement(&Measurement::bell_basis(), &[0, 1])
            .unwrap();
        assert!((probs[0] - 1.0).abs() < 1e-12); // \Phi+
        assert!(probs[1].abs() < 1e-12);
        assert!(probs[2].abs() < 1e-12);
        assert!(probs[3].abs() < 1e-12);
    }

    // --- compose ---

    #[test]
    fn test_compose_z_z() {
        let zz = Measurement::z_basis().compose(&Measurement::z_basis());
        assert_eq!(zz.num_qubits, 2);
        assert_eq!(zz.operators.len(), 4);
        assert_eq!(zz.labels, vec!["00", "01", "10", "11"]);
        assert_eq!(zz.eigenvalues, vec![1.0, -1.0, -1.0, 1.0]);
    }

    #[test]
    fn test_compose_operator_dimensions() {
        let zz = Measurement::z_basis().compose(&Measurement::z_basis());
        for op in &zz.operators {
            assert_eq!(op.dim(), (4, 4));
        }
    }

    #[test]
    fn test_compose_bell_x() {
        let bx = Measurement::bell_basis().compose(&Measurement::x_basis());
        assert_eq!(bx.num_qubits, 3);
        assert_eq!(bx.operators.len(), 8);
        assert_eq!(bx.labels[0], "Φ+0");
        assert_eq!(bx.labels[1], "Φ+1");
    }
}
