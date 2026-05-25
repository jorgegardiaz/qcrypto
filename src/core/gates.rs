use crate::core::errors::GateError;
use crate::core::utils;
use ndarray::{Array2, arr2};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Represents a quantum gate.
///
/// A gate is defined by its unitary matrix and the number of qubits it acts on.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Gate {
    /// The unitary matrix of the gate.
    pub matrix: Array2<Complex64>,
    /// The number of qubits the gate acts on.
    pub num_qubits: usize,
}

impl Gate {
    /// Creates a new `Gate` from a unitary matrix.
    ///
    /// # Arguments
    ///
    /// * `matrix` - A square, unitary `Array2<Complex64>`.
    ///
    /// # Returns
    ///
    /// A `Result` containing the constructed `Gate`.
    ///
    /// # Errors
    ///
    /// Returns a `GateError` if:
    /// - The matrix is not square.
    /// - The matrix dimensions are not a power of 2.
    /// - The matrix is not unitary.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::Gate;
    /// use ndarray::arr2;
    /// use num_complex::Complex64;
    ///
    /// let matrix = arr2(&[
    ///     [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ///     [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    /// ]);
    /// let gate = Gate::new(matrix).unwrap();
    /// assert_eq!(gate.num_qubits, 1);
    /// ```
    pub fn new(matrix: Array2<Complex64>) -> Result<Self, GateError> {
        let (rows, cols) = matrix.dim();

        if rows != cols {
            return Err(GateError::NotSquareMatrix);
        }

        if !rows.is_power_of_two() {
            return Err(GateError::InvalidDimensions);
        }

        if !Self::check_unitary(&matrix) {
            return Err(GateError::NonUnitary);
        }

        let num_qubits = rows.trailing_zeros() as usize;

        Ok(Self { matrix, num_qubits })
    }

    /// Checks if a given matrix is unitary
    fn check_unitary(matrix: &Array2<Complex64>) -> bool {
        let (rows, _) = matrix.dim();
        let eye = Array2::<Complex64>::eye(rows);

        let u_dagger = matrix.t().mapv(|x| x.conj());
        let product = matrix.dot(&u_dagger);

        product
            .iter()
            .zip(eye.iter())
            .all(|(a, b)| (*a - *b).norm() < 1e-12)
    }

    /// Expands a gate to act on a larger system of qubits.
    ///
    /// This function creates a new gate that acts on `num_total_qubits` by applying the original `gate`
    /// to the specified `targets` and `controls` (if any), and Identity on the rest.
    ///
    /// # Arguments
    ///
    /// * `num_total_qubits` - The total number of qubits in the system.
    /// * `gate` - The base gate to expand.
    /// * `targets` - Indices of the target qubits.
    /// * `controls` - Indices of the control qubits.
    ///
    /// # Returns
    ///
    /// A `Result` containing the expanded `Gate`.
    ///
    /// # Errors
    ///
    /// Returns `GateError` if:
    /// - Duplicate indices are found in `targets` or `controls`.
    /// - A qubit is used as both control and target.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::Gate;
    ///
    /// // Expand X gate into a CNOT (control=0, target=1) on a 2-qubit system
    /// let cnot = Gate::expand_gate(&Gate::x(), 2, &[1], &[0]).unwrap();
    /// assert_eq!(cnot.num_qubits, 2);
    /// assert_eq!(cnot.matrix.dim(), (4, 4));
    /// ```
    pub fn expand_gate(
        &self,
        num_total_qubits: usize,
        targets: &[usize],
        controls: &[usize],
    ) -> Result<Gate, GateError> {
        if let Some(dup) = utils::find_duplicate(targets) {
            return Err(GateError::DuplicateQubit(dup));
        }

        if let Some(dup) = utils::find_duplicate(controls) {
            return Err(GateError::DuplicateQubit(dup));
        }

        for &c in controls {
            if targets.contains(&c) {
                return Err(GateError::ControlTargetOverlap(c));
            }
        }

        Ok(Gate {
            matrix: utils::expand_operator(&self.matrix, num_total_qubits, targets, controls),
            num_qubits: num_total_qubits,
        })
    }

    // --- Standard Gates ---

    /// Creates an Identity gate.
    ///
    /// # Returns
    ///
    /// A single-qubit Identity gate.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::Gate;
    ///
    /// let gate = Gate::i();
    /// assert_eq!(gate.num_qubits, 1);
    /// assert_eq!(gate.matrix.dim(), (2, 2));
    /// ```
    pub fn i() -> Gate {
        Gate::new(arr2(&[
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ]))
        .expect("Error in I gate")
    }

    /// Creates a Pauli-X gate (NOT gate).
    ///
    /// Flips the state of a qubit: $|0> \to |1>$ and $|1> \to |0>$.
    ///
    /// # Returns
    ///
    /// A single-qubit Pauli-X gate.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{QuantumState, Gate};
    ///
    /// let mut state = QuantumState::new(1); // |0>
    /// state.apply(&Gate::x(), &[0]).unwrap(); // |1>
    /// ```
    pub fn x() -> Gate {
        Gate::new(arr2(&[
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ]))
        .expect("Error in X gate")
    }

    /// Creates a Pauli-Y gate.
    ///
    /// Applies a rotation around the Y axis: $Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$.
    ///
    /// # Returns
    ///
    /// A single-qubit Pauli-Y gate.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::Gate;
    ///
    /// let gate = Gate::y();
    /// assert_eq!(gate.num_qubits, 1);
    /// ```
    pub fn y() -> Gate {
        Gate::new(arr2(&[
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
            [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
        ]))
        .expect("Error in Y gate")
    }

    /// Creates a Pauli-Z gate.
    ///
    /// Applies a phase flip: $|0> \to |0>$, $|1> \to -|1>$.
    ///
    /// # Returns
    ///
    /// A single-qubit Pauli-Z gate.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::Gate;
    ///
    /// let gate = Gate::z();
    /// assert_eq!(gate.num_qubits, 1);
    /// ```
    pub fn z() -> Gate {
        Gate::new(arr2(&[
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
        ]))
        .expect("Error in Z gate")
    }

    /// Creates a Hadamard gate.
    ///
    /// Creates a superposition: $|0> \to |+>$ and $|1> \to |->$.
    ///
    /// # Returns
    ///
    /// A single-qubit Hadamard gate.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{QuantumState, Gate, Measurement};
    ///
    /// let mut state = QuantumState::new(1); // |0>
    /// state.apply(&Gate::h(), &[0]).unwrap(); // |+>
    /// let probs = state.set_measurement(&Measurement::z_basis(), &[0]).unwrap();
    /// assert!((probs[0] - 0.5).abs() < 1e-12);
    /// assert!((probs[1] - 0.5).abs() < 1e-12);
    /// ```
    pub fn h() -> Gate {
        let factor = 1.0 / 2.0_f64.sqrt();
        Gate::new(arr2(&[
            [Complex64::new(factor, 0.0), Complex64::new(factor, 0.0)],
            [Complex64::new(factor, 0.0), Complex64::new(-factor, 0.0)],
        ]))
        .expect("Error in H gate")
    }

    /// Creates an S gate (Phase gate, $Z^{1/2}$).
    ///
    /// Applies a $\pi/2$ phase: $|1> \to i|1>$.
    ///
    /// # Returns
    ///
    /// A single-qubit S gate.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::Gate;
    ///
    /// let gate = Gate::s();
    /// assert_eq!(gate.num_qubits, 1);
    /// ```
    pub fn s() -> Gate {
        Gate::new(arr2(&[
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
        ]))
        .expect("Error in S gate")
    }

    /// Creates a T gate ($Z^{1/4}$).
    ///
    /// Applies a $\pi/4$ phase: $|1> \to e^{i\pi/4}|1>$.
    ///
    /// # Returns
    ///
    /// A single-qubit T gate.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::Gate;
    ///
    /// let gate = Gate::t_gate();
    /// assert_eq!(gate.num_qubits, 1);
    /// ```
    pub fn t_gate() -> Gate {
        let angle = PI / 4.0;
        Gate::new(arr2(&[
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(angle.cos(), angle.sin()),
            ],
        ]))
        .expect("Error in T gate")
    }

    /// Creates a CNOT (Controlled-NOT) gate.
    ///
    /// Flips the target qubit (qubit 1) if the control qubit (qubit 0) is $|1>$.
    ///
    /// # Returns
    ///
    /// A two-qubit CNOT gate.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::Gate;
    ///
    /// let gate = Gate::cnot();
    /// assert_eq!(gate.num_qubits, 2);
    /// assert_eq!(gate.matrix.dim(), (4, 4));
    /// ```
    pub fn cnot() -> Gate {
        Gate::x()
            .expand_gate(2, &[1], &[0])
            .expect("Error in CNOT gate")
    }

    /// Creates a SWAP gate.
    ///
    /// Swaps the states of two qubits: $|01> \to |10>$.
    ///
    /// # Returns
    ///
    /// A two-qubit SWAP gate.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::Gate;
    ///
    /// let gate = Gate::swap();
    /// assert_eq!(gate.num_qubits, 2);
    /// assert_eq!(gate.matrix.dim(), (4, 4));
    /// ```
    pub fn swap() -> Gate {
        Gate::new(arr2(&[
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        ]))
        .expect("Error in SWAP gate")
    }

    /// Creates a Toffoli gate (CCNOT).
    ///
    /// Flips the target qubit (qubit 2) only if both control qubits (qubits 0 and 1) are $|1>$.
    ///
    /// # Returns
    ///
    /// A three-qubit Toffoli gate.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::Gate;
    ///
    /// let gate = Gate::toffoli();
    /// assert_eq!(gate.num_qubits, 3);
    /// assert_eq!(gate.matrix.dim(), (8, 8));
    /// ```
    pub fn toffoli() -> Gate {
        Gate::x()
            .expand_gate(3, &[2], &[0, 1])
            .expect("Error in Toffoli gate")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // --- Gate::new boundary tests ---

    #[test]
    fn test_new_non_square_matrix() {
        let matrix = Array2::from_shape_vec((2, 3), vec![Complex64::new(1.0, 0.0); 6]).unwrap();

        assert!(matches!(Gate::new(matrix), Err(GateError::NotSquareMatrix)));
    }

    #[test]
    fn test_new_non_power_of_two() {
        // 3x3 is square but not a power of 2
        let matrix = Array2::eye(3);
        assert!(matches!(
            Gate::new(matrix),
            Err(GateError::InvalidDimensions)
        ));
    }

    #[test]
    fn test_new_non_unitary() {
        // A 2x2 matrix that is NOT unitary
        let matrix = arr2(&[
            [Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ]);
        assert!(matches!(Gate::new(matrix), Err(GateError::NonUnitary)));
    }

    #[test]
    fn test_new_valid_identity() {
        let matrix: Array2<Complex64> = Array2::eye(2);
        let gate = Gate::new(matrix).unwrap();
        assert_eq!(gate.num_qubits, 1);
    }

    #[test]
    fn test_new_valid_4x4() {
        // 4x4 identity is a valid 2-qubit gate
        let matrix: Array2<Complex64> = Array2::eye(4);
        let gate = Gate::new(matrix).unwrap();
        assert_eq!(gate.num_qubits, 2);
    }

    // --- Gate::expand_gate boundary tests ---

    #[test]
    fn test_expand_gate_duplicate_targets() {
        let result = Gate::expand_gate(&Gate::x(), 2, &[0, 0], &[]);
        assert!(matches!(result, Err(GateError::DuplicateQubit(0))));
    }

    #[test]
    fn test_expand_gate_duplicate_controls() {
        let result = Gate::expand_gate(&Gate::x(), 3, &[2], &[0, 0]);
        assert!(matches!(result, Err(GateError::DuplicateQubit(0))));
    }

    #[test]
    fn test_expand_gate_control_target_overlap() {
        let result = Gate::expand_gate(&Gate::x(), 2, &[0], &[0]);
        assert!(matches!(result, Err(GateError::ControlTargetOverlap(0))));
    }

    // --- Standard gate mathematical properties ---

    #[test]
    fn test_all_standard_gates_are_unitary() {
        assert!(Gate::check_unitary(&Gate::i().matrix));
        assert!(Gate::check_unitary(&Gate::x().matrix));
        assert!(Gate::check_unitary(&Gate::y().matrix));
        assert!(Gate::check_unitary(&Gate::z().matrix));
        assert!(Gate::check_unitary(&Gate::h().matrix));
        assert!(Gate::check_unitary(&Gate::s().matrix));
        assert!(Gate::check_unitary(&Gate::t_gate().matrix));
        assert!(Gate::check_unitary(&Gate::cnot().matrix));
        assert!(Gate::check_unitary(&Gate::swap().matrix));
        assert!(Gate::check_unitary(&Gate::toffoli().matrix));
    }

    #[test]
    fn test_pauli_x_is_involution() {
        // X² = I
        let x = &Gate::x().matrix;
        let x_squared = x.dot(x);
        let eye = Array2::<Complex64>::eye(2);

        for (a, b) in x_squared.iter().zip(eye.iter()) {
            assert!((*a - *b).norm() < 1e-12);
        }
    }

    #[test]
    fn test_pauli_y_is_involution() {
        // Y² = I
        let y = &Gate::y().matrix;
        let y_squared = y.dot(y);
        let eye = Array2::<Complex64>::eye(2);

        for (a, b) in y_squared.iter().zip(eye.iter()) {
            assert!((*a - *b).norm() < 1e-12);
        }
    }

    #[test]
    fn test_pauli_z_is_involution() {
        // Z² = I
        let z = &Gate::z().matrix;
        let z_squared = z.dot(z);
        let eye = Array2::<Complex64>::eye(2);

        for (a, b) in z_squared.iter().zip(eye.iter()) {
            assert!((*a - *b).norm() < 1e-12);
        }
    }

    #[test]
    fn test_hadamard_is_involution() {
        // H² = I
        let h = &Gate::h().matrix;
        let h_squared = h.dot(h);
        let eye = Array2::<Complex64>::eye(2);

        for (a, b) in h_squared.iter().zip(eye.iter()) {
            assert!((*a - *b).norm() < 1e-12);
        }
    }

    #[test]
    fn test_s_squared_is_z() {
        // S² = Z
        let s = &Gate::s().matrix;
        let z = &Gate::z().matrix;
        let s_squared = s.dot(s);

        for (a, b) in s_squared.iter().zip(z.iter()) {
            assert!((*a - *b).norm() < 1e-12);
        }
    }

    #[test]
    fn test_swap_is_involution() {
        // SWAP² = I
        let sw = &Gate::swap().matrix;
        let sw_squared = sw.dot(sw);
        let eye = Array2::<Complex64>::eye(4);

        for (a, b) in sw_squared.iter().zip(eye.iter()) {
            assert!((*a - *b).norm() < 1e-12);
        }
    }

    #[test]
    fn test_cnot_is_involution() {
        // CNOT² = I
        let cnot = &Gate::cnot().matrix;
        let cnot_squared = cnot.dot(cnot);
        let eye = Array2::<Complex64>::eye(4);

        for (a, b) in cnot_squared.iter().zip(eye.iter()) {
            assert!((*a - *b).norm() < 1e-12);
        }
    }

    #[test]
    fn test_xhz_equals_zhx_equals_h() {
        let x = &Gate::x().matrix;
        let h = &Gate::h().matrix;
        let z = &Gate::z().matrix;

        // XHZ = H
        let xhz = x.dot(h).dot(z);
        for (a, b) in xhz.iter().zip(h.iter()) {
            assert!((*a - *b).norm() < 1e-12);
        }

        // ZHX = H
        let zhx = z.dot(h).dot(x);
        for (a, b) in zhx.iter().zip(h.iter()) {
            assert!((*a - *b).norm() < 1e-12);
        }
    }

    #[test]
    fn test_standard_gates_dimensions() {
        assert_eq!(Gate::i().num_qubits, 1);
        assert_eq!(Gate::x().num_qubits, 1);
        assert_eq!(Gate::y().num_qubits, 1);
        assert_eq!(Gate::z().num_qubits, 1);
        assert_eq!(Gate::h().num_qubits, 1);
        assert_eq!(Gate::s().num_qubits, 1);
        assert_eq!(Gate::t_gate().num_qubits, 1);
        assert_eq!(Gate::cnot().num_qubits, 2);
        assert_eq!(Gate::swap().num_qubits, 2);
        assert_eq!(Gate::toffoli().num_qubits, 3);
    }
}
