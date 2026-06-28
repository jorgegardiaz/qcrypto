//! Utility functions for quantum operations.
//!
//! This module contains helper functions for:
//! - Matrix operations (Kronecker product, trace, outer product, square root).
//! - Operator expansion to larger systems.
//! - Completeness checks for measurements and channels.
//! - Bit manipulation for state indices.

use nalgebra::DMatrix;
use ndarray::{Array1, Array2, Axis};
use num_complex::Complex64;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Computes the Kronecker (Tensor) product of two matrices.
///
/// If `m1` is an $m \times n$ matrix and `m2` is a $p \times q$ matrix,
/// the result is an $mp \times nq$ matrix.
///
/// # Arguments
///
/// * `m1` - First matrix.
/// * `m2` - Second matrix.
///
/// # Returns
///
/// A `Array2<Complex64>` matrix with the Kronecker product of the input matrices.
///
/// # Example
/// ```rust
/// use qcrypto::{utils::kronecker_product_matrix, Gate};
///
/// let m1 = Gate::i().matrix;
/// let m2 = Gate::x().matrix;
/// let m3 = kronecker_product_matrix(&m1, &m2);
/// assert_eq!(m3, Gate::x().expand_gate(2, &[1], &[]).unwrap().matrix);
/// ```
pub fn kronecker_product_matrix(
    m1: &Array2<Complex64>,
    m2: &Array2<Complex64>,
) -> Array2<Complex64> {
    let (m, n) = m1.dim();
    let (p, q) = m2.dim();

    // Tensor product implementation using broadcasting
    // A is (m, n), B is (p, q)
    // We want output (m*p, n*q)

    // 1. Reshape A to (m, 1, n, 1)
    let m1_expanded = m1.view().insert_axis(Axis(1)).insert_axis(Axis(3));

    // 2. Reshape B to (1, p, 1, q)
    let m2_expanded = m2.view().insert_axis(Axis(0)).insert_axis(Axis(2));

    // 3. Broadcast multiply -> (m, p, n, q)
    let tensor_product = &m1_expanded * &m2_expanded;

    // 4. Reshape to (m*p, n*q)
    tensor_product
        .into_shape_with_order((m * p, n * q))
        .expect("Error in Kronecker product")
}

/// Computes the trace of a matrix (sum of diagonal elements).
///
/// # Arguments
///
/// * `matrix` - The square matrix to compute the trace of.
///
/// # Returns
///
/// The complex sum of the diagonal elements.
///
/// # Example
/// ```rust
/// use qcrypto::utils::trace;
/// use ndarray::Array2;
/// use num_complex::Complex64;
///
/// let eye = Array2::<Complex64>::eye(3);
/// assert_eq!(trace(&eye), Complex64::new(3.0, 0.0));
/// ```
pub fn trace(matrix: &Array2<Complex64>) -> Complex64 {
    matrix.diag().sum()
}

/// Generates the full operator matrix ($2^N \times 2^N$) for the whole system.
///
/// It expands a local operator acting on `targets` (and controlled by `controls`)
/// to an operator on the full system of `num_total_qubits`.
///
/// # Arguments
///
/// * `num_total_qubits` - Total number of qubits in the system.
/// * `matrix` - The matrix representation of the local gate.
/// * `targets` - Indices of the target qubits.
/// * `controls` - Indices of the control qubits.
///
/// # Returns
///
/// A `Array2<Complex64>` representing the expanded operator for the specified system.
///
/// # Example
/// ```rust
/// use qcrypto::{utils::{kronecker_product_matrix, expand_operator}, Gate};
///
/// let expanded = expand_operator(&Gate::x().matrix, 2, &[1], &[]);
/// let m1 = Gate::i().matrix;
/// let m2 = Gate::x().matrix;
/// let m3 = kronecker_product_matrix(&m1, &m2);
///
/// assert_eq!(expanded, m3);
/// ```
pub fn expand_operator(
    matrix: &Array2<Complex64>,
    num_total_qubits: usize,
    targets: &[usize],
    controls: &[usize],
) -> Array2<Complex64> {
    let dim = 1 << num_total_qubits;
    let mut full_matrix = Array2::<Complex64>::zeros((dim, dim));

    let mapped_controls: Vec<usize> = controls.iter().map(|&c| num_total_qubits - 1 - c).collect();
    let mut mapped_targets: Vec<usize> =
        targets.iter().map(|&t| num_total_qubits - 1 - t).collect();
    mapped_targets.reverse(); // Preserve Big-Endian: gate qubit i ↔ targets[i]

    // 1 in position c if qubit c is a control qubit
    let mut control_mask = 0usize;
    for &c in &mapped_controls {
        control_mask |= 1 << c;
    }
    // 1 in position t if qubit t is a target qubit
    let mut target_mask = 0usize;
    for &t in &mapped_targets {
        target_mask |= 1 << t;
    }
    // Bits that are not target (do not change)
    let passive_mask = !target_mask;
    // Iterate over every column, each column corresponding to a basic state
    for col_idx in 0..dim {
        // If NOT ALL control qubits are 1 in the associated state applied to the column
        // This checks if the state associated to the column is affected by the matrix
        if (col_idx & control_mask) != control_mask {
            // This basic state is not affected by the matrix -> 1 in diagonal
            full_matrix[[col_idx, col_idx]] = Complex64::new(1.0, 0.0);
            continue;
        }
        // If ALL control qubits are 1 in the associated sate to the column
        // This means the basic state associated to the column is affected by the matrix
        // We extract the bits of col_idx in the positions of the targets
        let small_col = extract_bits(col_idx, &mapped_targets);
        // Iterate over the rows of the matrix applied to the subsystem of the target qubits
        for small_row in 0..matrix.nrows() {
            // Get the value of the matrix in the position associated to the iteration
            let val = matrix[[small_row, small_col]];
            // Check if val is 0
            if val.norm_sqr() < 1e-12 {
                // Due to floating-point representation, really small entries are skipped
                continue;
            }
            // Construct global row index: preserve passive bits, update target bits
            // Scatter local 'small_row' bits to their physical target positions
            let new_target_bits = deposit_bits(small_row, &mapped_targets);
            // Combine preserved passive bits with the new target bits
            let row_idx = (col_idx & passive_mask) | new_target_bits;
            // Populate full matrix entry
            full_matrix[[row_idx, col_idx]] = val;
        }
    }
    full_matrix
}

/// Extracts the bits in the positions specified by `indices` from the sequence `value`
fn extract_bits(value: usize, indices: &[usize]) -> usize {
    let mut result = 0;
    for (i, &pos) in indices.iter().enumerate() {
        if (value >> pos) & 1 == 1 {
            result |= 1 << i;
        }
    }
    result
}

/// Scatters bits from `compact_value` into the positions specified by `indices`.
fn deposit_bits(compact_value: usize, indices: &[usize]) -> usize {
    // Maps the i-th bit of `compact_value` to bit position `indices[i]` in the result.
    let mut result = 0;
    for (i, &pos) in indices.iter().enumerate() {
        if (compact_value >> i) & 1 == 1 {
            result |= 1 << pos;
        }
    }
    result
}

/// Finds the first duplicate value in a slice of indices.
///
/// # Arguments
///
/// * `indices` - A slice of qubit indices to check.
///
/// # Returns
///
/// `Some(index)` if a duplicate is found, `None` otherwise.
///
/// # Example
/// ```rust
/// use qcrypto::utils::find_duplicate;
///
/// assert_eq!(find_duplicate(&[0, 1, 2, 1]), Some(1));
/// assert_eq!(find_duplicate(&[0, 1, 2]), None);
/// ```
pub fn find_duplicate(indices: &[usize]) -> Option<usize> {
    let mut seen = std::collections::HashSet::new();
    indices.iter().find(|&&idx| !seen.insert(idx)).copied()
}

/// Checks the completeness relation for Kraus operators.
///
/// Verifies if $\sum M_k^\dagger M_k = I$.
///
/// # Arguments
///
/// * `ops` - A slice of Kraus operator matrices.
/// * `dim` - The dimension of the Hilbert space.
///
/// # Returns
///
/// `true` if the operators satisfy the trace-preserving condition.
///
/// # Example
/// ```rust
/// use qcrypto::utils::check_kraus_completeness;
/// use ndarray::Array2;
/// use num_complex::Complex64;
///
/// // Identity is a valid single Kraus operator
/// let eye = Array2::<Complex64>::eye(2);
/// assert!(check_kraus_completeness(&[eye], 2));
/// ```
pub fn check_kraus_completeness(ops: &[Array2<Complex64>], dim: usize) -> bool {
    let eye = Array2::<Complex64>::eye(dim);
    let sum = ops
        .iter()
        .fold(Array2::<Complex64>::zeros((dim, dim)), |acc, op| {
            let dag = op.t().mapv(|c| c.conj());
            acc + dag.dot(op)
        });
    sum.iter()
        .zip(eye.iter())
        .all(|(a, b)| (a - b).norm() < 1e-12)
}

/// Checks the POVM completeness relation.
///
/// Verifies if $\sum E_k = I$.
///
/// # Arguments
///
/// * `ops` - A slice of POVM element matrices.
/// * `dim` - The dimension of the Hilbert space.
///
/// # Returns
///
/// `true` if the elements sum to the Identity matrix.
///
/// # Example
/// ```rust
/// use qcrypto::utils::check_povm_completeness;
/// use ndarray::Array2;
/// use num_complex::Complex64;
///
/// // |0><0| + |1><1| = I
/// let p0 = Array2::from_diag(&ndarray::array![
///     Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)
/// ]);
/// let p1 = Array2::from_diag(&ndarray::array![
///     Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)
/// ]);
/// assert!(check_povm_completeness(&[p0, p1], 2));
/// ```
pub fn check_povm_completeness(ops: &[Array2<Complex64>], dim: usize) -> bool {
    let mut sum = Array2::<Complex64>::zeros((dim, dim));
    for op in ops {
        sum += op;
    }
    let identity = Array2::<Complex64>::eye(dim);
    (sum - identity).iter().all(|x| x.norm() < 1e-9)
}

/// Computes the outer product of two vectors $|a\rangle\langle b|$.
///
/// # Arguments
///
/// * `a` - The ket vector $|a\rangle$.
/// * `b` - The bra vector (conjugated automatically to form $\langle b|$).
///
/// # Returns
///
/// The matrix $|a\rangle\langle b|$.
///
/// # Example
/// ```rust
/// use qcrypto::utils::outer_product;
/// use ndarray::array;
/// use num_complex::Complex64;
///
/// let v = array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
/// let proj = outer_product(&v, &v);
/// // |0><0| = diag(1, 0)
/// assert_eq!(proj[[0, 0]], Complex64::new(1.0, 0.0));
/// assert_eq!(proj[[1, 1]], Complex64::new(0.0, 0.0));
/// ```
pub fn outer_product(a: &Array1<Complex64>, b: &Array1<Complex64>) -> Array2<Complex64> {
    let n = a.len();
    let m = b.len();
    let mut res = Array2::zeros((n, m));

    for i in 0..n {
        for j in 0..m {
            res[[i, j]] = a[i] * b[j].conj();
        }
    }
    res
}

/// Finds the square root of a positive semi-definite matrix.
///
/// Uses spectral decomposition $M = V D V^\dagger$ to compute $\sqrt{M} = V \sqrt{D} V^\dagger$.
/// An analytical shortcut is used for 2x2 matrices.
///
/// # Arguments
///
/// * `mat` - A positive semi-definite matrix.
///
/// # Returns
///
/// The unique positive semi-definite matrix $\sqrt{M}$.
///
/// # Example
/// ```rust
/// use qcrypto::utils::sqrt_positive_matrix;
/// use ndarray::Array2;
/// use num_complex::Complex64;
///
/// // sqrt(I) = I
/// let eye = Array2::<Complex64>::eye(2);
/// let result = sqrt_positive_matrix(&eye);
/// for (a, b) in result.iter().zip(eye.iter()) {
///     assert!((*a - *b).norm() < 1e-12);
/// }
/// ```
pub fn sqrt_positive_matrix(mat: &Array2<Complex64>) -> Array2<Complex64> {
    let (rows, cols) = mat.dim();

    // Optimized case for 2x2 matrix
    if rows == 2 && cols == 2 {
        return sqrt_2x2_analytical(mat);
    }

    // General case using nalgebra
    sqrt_nxn_nalgebra(mat)
}

/// Directo formula for square root of a 2x2 matrix
/// sqrt(A) = (A + sqrt(det)I) / sqrt(tr + 2sqrt(det))
fn sqrt_2x2_analytical(mat: &Array2<Complex64>) -> Array2<Complex64> {
    let tr = mat[[0, 0]] + mat[[1, 1]];
    let det = mat[[0, 0]] * mat[[1, 1]] - mat[[0, 1]] * mat[[1, 0]];

    let clean_det = if det.norm() < 1e-12 {
        Complex64::new(0.0, 0.0)
    } else {
        det
    };
    let sqrt_det = clean_det.sqrt();

    let s_sq = tr + Complex64::new(2.0, 0.0) * sqrt_det;
    let s = s_sq.sqrt();

    if s.norm() < 1e-12 {
        return Array2::zeros((2, 2));
    }

    let factor = Complex64::new(1.0, 0.0) / s;
    let identity = Array2::<Complex64>::eye(2);

    let numerator = mat + &identity.mapv(|x| x * sqrt_det);
    numerator.mapv(|x| x * factor)
}

/// General implementation using nalgebra
fn sqrt_nxn_nalgebra(mat: &Array2<Complex64>) -> Array2<Complex64> {
    let (rows, cols) = mat.dim();

    // Convert ndarray -> nalgebra
    let na_mat = DMatrix::from_fn(rows, cols, |r, c| mat[[r, c]]);

    // Schur/Eigen decomposition
    let eigen = na_mat.symmetric_eigen();

    // sqrt(D)
    let mut sqrt_eigenvals = DMatrix::zeros(rows, rows);
    for i in 0..rows {
        let val = eigen.eigenvalues[i];
        let clean_val = if val < 0.0 { 0.0 } else { val };
        sqrt_eigenvals[(i, i)] = Complex64::new(clean_val.sqrt(), 0.0);
    }

    // V * sqrt(D) * V†
    // eigen.eigenvectors are the vectors V
    let v = &eigen.eigenvectors;
    let v_adjoint = v.adjoint();

    let result_na = v * sqrt_eigenvals * v_adjoint;

    // Reconvert nalgebra -> ndarray
    let mut result_nd = Array2::<Complex64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            result_nd[[r, c]] = result_na[(r, c)];
        }
    }

    result_nd
}

/// Checks if a matrix is Hermitian ($M = M^\dagger$).
///
/// # Arguments
///
/// * `mat` - The matrix to check.
/// * `tol` - Tolerance for floating-point comparisons.
///
/// # Returns
///
/// `true` if the matrix is Hermitian within the given tolerance.
///
/// # Example
/// ```rust
/// use qcrypto::utils::is_hermitian;
/// use ndarray::Array2;
/// use num_complex::Complex64;
///
/// let eye = Array2::<Complex64>::eye(2);
/// assert!(is_hermitian(&eye, 1e-12));
/// ```
pub fn is_hermitian(mat: &Array2<Complex64>, tol: f64) -> bool {
    mat.iter()
        .zip(mat.t().iter())
        .all(|(a, b)| (a - b.conj()).norm() < tol)
}

/// Applies a local quantum operator to the left side of a density matrix: $\rho_{new} = U \cdot \rho$.
///
/// Uses bitwise operations to perform the local tensor update efficiently,
/// avoiding the computationally expensive full-space matrix expansion.
///
/// # Arguments
///
/// * `num_total_qubits` - Total number of qubits in the system.
/// * `rho` - The current density matrix.
/// * `local_matrix` - The operator matrix (e.g., 2x2 for a single-qubit gate).
/// * `targets` - Indices of the target qubits.
/// * `controls` - Indices of the control qubits.
///
/// # Returns
///
/// The updated density matrix after left-multiplication.
pub fn apply_local_left(
    num_total_qubits: usize,
    rho: &Array2<Complex64>,
    local_matrix: &Array2<Complex64>,
    targets: &[usize],
    controls: &[usize],
) -> Array2<Complex64> {
    let dim = 1 << num_total_qubits;
    let mut new_rho = Array2::<Complex64>::zeros((dim, dim));

    let mapped_controls: Vec<usize> = controls.iter().map(|&c| num_total_qubits - 1 - c).collect();
    let mut mapped_targets: Vec<usize> =
        targets.iter().map(|&t| num_total_qubits - 1 - t).collect();
    mapped_targets.reverse(); // Preserve Big-Endian: gate qubit i ↔ targets[i]

    // Build the control mask: bits are 1 at control qubit positions.
    let mut control_mask = 0usize;
    for &c in &mapped_controls {
        control_mask |= 1 << c;
    }

    // Build the target mask: bits are 1 at target qubit positions.
    let mut target_mask = 0usize;
    for &t in &mapped_targets {
        target_mask |= 1 << t;
    }

    // Passive mask: bits are 1 where qubits are neither targets nor controls.
    // Since the operator only acts on targets, passive bits remain unchanged.
    let passive_mask = !target_mask;

    // Dimension of the local operator (e.g., 2 for 1 target qubit, 4 for 2 target qubits).
    let k_dim = 1 << targets.len();

    // Iterate over all columns. Left multiplication only mixes row elements.
    #[cfg(feature = "parallel")]
    let iter = new_rho.axis_iter_mut(Axis(1)).into_par_iter();
    #[cfg(not(feature = "parallel"))]
    let iter = new_rho.axis_iter_mut(Axis(1)).into_iter();
    iter.enumerate().for_each(|(col, mut col_view)| {
        for row in 0..dim {
            if (row & control_mask) != control_mask {
                col_view[row] = rho[[row, col]];
                continue;
            }

            let small_row = extract_bits(row, &mapped_targets);
            let mut sum = Complex64::new(0.0, 0.0);

            for small_col in 0..k_dim {
                let val = local_matrix[[small_row, small_col]];

                if val.norm_sqr() < 1e-12 {
                    continue;
                }

                let m = (row & passive_mask) | deposit_bits(small_col, &mapped_targets);
                sum += val * rho[[m, col]];
            }

            col_view[row] = sum;
        }
    });

    new_rho
}

/// Applies a local quantum operator to the right side of a density matrix: $\rho_{new} = \rho \cdot U$.
///
/// Similar to `apply_local_left`, but acts on columns instead of rows.
///
/// # Arguments
///
/// * `num_total_qubits` - Total number of qubits in the system.
/// * `rho` - The current density matrix.
/// * `local_matrix` - The operator matrix.
/// * `targets` - Indices of the target qubits.
/// * `controls` - Indices of the control qubits.
///
/// # Returns
///
/// The updated density matrix after right-multiplication.
pub fn apply_local_right(
    num_total_qubits: usize,
    rho: &Array2<Complex64>,
    local_matrix: &Array2<Complex64>,
    targets: &[usize],
    controls: &[usize],
) -> Array2<Complex64> {
    let dim = 1 << num_total_qubits;
    let mut new_rho = Array2::<Complex64>::zeros((dim, dim));

    let mapped_controls: Vec<usize> = controls.iter().map(|&c| num_total_qubits - 1 - c).collect();
    let mut mapped_targets: Vec<usize> =
        targets.iter().map(|&t| num_total_qubits - 1 - t).collect();
    mapped_targets.reverse(); // Preserve Big-Endian: gate qubit i ↔ targets[i]

    let mut control_mask = 0usize;
    for &c in &mapped_controls {
        control_mask |= 1 << c;
    }

    let mut target_mask = 0usize;
    for &t in &mapped_targets {
        target_mask |= 1 << t;
    }

    let passive_mask = !target_mask;
    let k_dim = 1 << targets.len();

    // Iterate over all rows. Right multiplication only mixes column elements.
    #[cfg(feature = "parallel")]
    let iter = new_rho.axis_iter_mut(Axis(0)).into_par_iter();
    #[cfg(not(feature = "parallel"))]
    let iter = new_rho.axis_iter_mut(Axis(0)).into_iter();
    iter.enumerate().for_each(|(row, mut row_view)| {
        for col in 0..dim {
            if (col & control_mask) != control_mask {
                row_view[col] = rho[[row, col]];
                continue;
            }

            let small_col_idx = extract_bits(col, &mapped_targets);
            let mut sum = Complex64::new(0.0, 0.0);

            for small_row_idx in 0..k_dim {
                let val = local_matrix[[small_row_idx, small_col_idx]];

                if val.norm_sqr() < 1e-12 {
                    continue;
                }

                let m = (col & passive_mask) | deposit_bits(small_row_idx, &mapped_targets);
                sum += rho[[row, m]] * val;
            }

            row_view[col] = sum;
        }
    });

    new_rho
}

/// Applies a local quantum operator to a state vector: $|\psi_{new}\rangle = U |\psi\rangle$.
///
/// # Arguments
///
/// * `num_total_qubits` - Total number of qubits in the system.
/// * `psi` - The current state vector.
/// * `local_matrix` - The operator matrix.
/// * `targets` - Indices of the target qubits.
/// * `controls` - Indices of the control qubits.
///
/// # Returns
///
/// The updated state vector after applying the operator.
pub fn apply_local_vector(
    num_total_qubits: usize,
    psi: &Array1<Complex64>,
    local_matrix: &Array2<Complex64>,
    targets: &[usize],
    controls: &[usize],
) -> Array1<Complex64> {
    let dim = 1 << num_total_qubits;
    let mut new_psi = Array1::<Complex64>::zeros(dim);

    let mapped_controls: Vec<usize> = controls.iter().map(|&c| num_total_qubits - 1 - c).collect();
    let mut mapped_targets: Vec<usize> =
        targets.iter().map(|&t| num_total_qubits - 1 - t).collect();
    mapped_targets.reverse(); // Preserve Big-Endian: gate qubit i ↔ targets[i]

    let mut control_mask = 0usize;
    for &c in &mapped_controls {
        control_mask |= 1 << c;
    }

    let mut target_mask = 0usize;
    for &t in &mapped_targets {
        target_mask |= 1 << t;
    }

    let passive_mask = !target_mask;
    let k_dim = 1 << targets.len();

    // In a state vector representation, left multiplication by an operator M is:
    // (M |psi>)_i = sum_j M_{ij} psi_j
    // where 'row' maps to 'i', and 'j' is constructed by iterating over small_col.
    let slice = new_psi.as_slice_mut().unwrap();
    #[cfg(feature = "parallel")]
    let iter = slice.par_iter_mut();
    #[cfg(not(feature = "parallel"))]
    let iter = slice.iter_mut();
    iter.enumerate().for_each(|(row, el)| {
        if (row & control_mask) != control_mask {
            *el = psi[row];
            return;
        }

        let small_row = extract_bits(row, &mapped_targets);
        let mut sum = Complex64::new(0.0, 0.0);

        for small_col in 0..k_dim {
            let val = local_matrix[[small_row, small_col]];

            if val.norm_sqr() < 1e-12 {
                continue;
            }

            let m = (row & passive_mask) | deposit_bits(small_col, &mapped_targets);
            sum += val * psi[m];
        }

        *el = sum;
    });

    new_psi
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    // --- extract_bits / deposit_bits ---

    #[test]
    fn test_extract_bits() {
        // value = 0b1101 (13), indices = [0, 2] → bits at pos 0 and 2 → 1,1 → 0b11 = 3
        assert_eq!(extract_bits(0b1101, &[0, 2]), 0b11);
        // value = 0b1010, indices = [1, 3] → bits at pos 1 and 3 → 1,1 → 0b11 = 3
        assert_eq!(extract_bits(0b1010, &[1, 3]), 0b11);
        // value = 0b1010, indices = [0, 2] → bits at pos 0 and 2 → 0,0 → 0b00 = 0
        assert_eq!(extract_bits(0b1010, &[0, 2]), 0b00);
    }

    #[test]
    fn test_deposit_bits() {
        // compact = 0b11 (3), indices = [0, 2] → bit 0 goes to pos 0, bit 1 goes to pos 2 → 0b101 = 5
        assert_eq!(deposit_bits(0b11, &[0, 2]), 0b101);
        // compact = 0b01 (1), indices = [1, 3] → bit 0 goes to pos 1 → 0b0010 = 2
        assert_eq!(deposit_bits(0b01, &[1, 3]), 0b0010);
    }

    #[test]
    fn test_extract_deposit_roundtrip() {
        let indices = vec![1, 3];
        let original = 0b1010usize; // bits at [1,3] = 1,1 = 0b11
        let extracted = extract_bits(original, &indices);
        let deposited = deposit_bits(extracted, &indices);
        // The deposited value should match the original bits at the target positions
        assert_eq!(deposited, 0b1010); // mask relevant bits
        assert_eq!(extracted, 0b11);
    }

    // --- kronecker_product ---

    #[test]
    fn test_kronecker_product_identity() {
        let i2: Array2<Complex64> = Array2::eye(2);
        let result = kronecker_product_matrix(&i2, &i2);
        let i4: Array2<Complex64> = Array2::eye(4);
        for (a, b) in result.iter().zip(i4.iter()) {
            assert!((*a - *b).norm() < 1e-12);
        }
    }

    #[test]
    fn test_kronecker_product_dimensions() {
        let a = Array2::<Complex64>::eye(2);
        let b = Array2::<Complex64>::eye(4);
        let result = kronecker_product_matrix(&a, &b);
        assert_eq!(result.dim(), (8, 8));
    }

    // --- trace ---

    #[test]
    fn test_trace_identity() {
        let eye: Array2<Complex64> = Array2::eye(4);
        let tr = trace(&eye);
        assert!((tr - Complex64::new(4.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn test_trace_zero() {
        let zero = Array2::<Complex64>::zeros((3, 3));
        let tr = trace(&zero);
        assert!(tr.norm() < 1e-12);
    }

    // --- outer_product ---

    #[test]
    fn test_outer_product_projector() {
        // |0><0| should be diag(1, 0)
        let v0: Array1<Complex64> = array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let proj = outer_product(&v0, &v0);
        assert!((proj[[0, 0]] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        assert!(proj[[0, 1]].norm() < 1e-12);
        assert!(proj[[1, 0]].norm() < 1e-12);
        assert!(proj[[1, 1]].norm() < 1e-12);
    }

    // --- find_duplicate ---

    #[test]
    fn test_find_duplicate_none() {
        assert_eq!(find_duplicate(&[0, 1, 2, 3]), None);
    }

    #[test]
    fn test_find_duplicate_found() {
        assert_eq!(find_duplicate(&[0, 1, 2, 1]), Some(1));
    }

    #[test]
    fn test_find_duplicate_empty() {
        let empty: &[usize] = &[];
        assert_eq!(find_duplicate(empty), None);
    }

    // --- check_kraus_completeness ---

    #[test]
    fn test_kraus_completeness_identity() {
        let eye: Array2<Complex64> = Array2::eye(2);
        assert!(check_kraus_completeness(&[eye], 2));
    }

    #[test]
    fn test_kraus_completeness_fails() {
        let zero = Array2::<Complex64>::zeros((2, 2));
        assert!(!check_kraus_completeness(&[zero], 2));
    }

    // --- check_povm_completeness ---

    #[test]
    fn test_povm_completeness_z_basis() {
        let p0 = Array2::from_diag(&array![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
        let p1 = Array2::from_diag(&array![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]);
        assert!(check_povm_completeness(&[p0, p1], 2));
    }

    // --- is_hermitian ---

    #[test]
    fn test_is_hermitian_identity() {
        let eye: Array2<Complex64> = Array2::eye(2);
        assert!(is_hermitian(&eye, 1e-12));
    }

    #[test]
    fn test_is_hermitian_fails() {
        let mat = array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
            [Complex64::new(0.0, 1.0), Complex64::new(1.0, 0.0)]
        ];
        assert!(!is_hermitian(&mat, 1e-12));
    }

    // --- expand_operator ---

    #[test]
    fn test_expand_identity_is_identity() {
        let i2: Array2<Complex64> = Array2::eye(2);
        let expanded = expand_operator(&i2, 2, &[0], &[]);
        let i4: Array2<Complex64> = Array2::eye(4);
        for (a, b) in expanded.iter().zip(i4.iter()) {
            assert!((*a - *b).norm() < 1e-12);
        }
    }
    // --- sqrt_positive_matrix ---

    #[test]
    fn test_sqrt_positive_matrix_2x2_zero() {
        let zero = Array2::<Complex64>::zeros((2, 2));
        let result = sqrt_positive_matrix(&zero);
        for &val in result.iter() {
            assert!(val.norm() < 1e-12);
        }
    }

    #[test]
    fn test_sqrt_positive_matrix_4x4() {
        let eye = Array2::<Complex64>::eye(4);
        let result = sqrt_positive_matrix(&eye);
        for (a, b) in result.iter().zip(eye.iter()) {
            assert!((*a - *b).norm() < 1e-12);
        }
    }

    // --- apply_local_* ---

    #[test]
    fn test_apply_local_left_identity() {
        let i2 = Array2::<Complex64>::eye(2);
        let rho = Array2::<Complex64>::eye(4);
        let new_rho = apply_local_left(2, &rho, &i2, &[0], &[]);
        for (a, b) in new_rho.iter().zip(rho.iter()) {
            assert!((*a - *b).norm() < 1e-12);
        }
    }

    #[test]
    fn test_apply_local_right_identity() {
        let i2 = Array2::<Complex64>::eye(2);
        let rho = Array2::<Complex64>::eye(4);
        let new_rho = apply_local_right(2, &rho, &i2, &[0], &[]);
        for (a, b) in new_rho.iter().zip(rho.iter()) {
            assert!((*a - *b).norm() < 1e-12);
        }
    }

    #[test]
    fn test_apply_local_vector_identity() {
        let i2 = Array2::<Complex64>::eye(2);
        let psi = Array1::<Complex64>::zeros(4);
        let new_psi = apply_local_vector(2, &psi, &i2, &[0], &[]);
        for (a, b) in new_psi.iter().zip(psi.iter()) {
            assert!((*a - *b).norm() < 1e-12);
        }
    }

    // --- Multi-qubit Big-Endian ordering tests ---

    #[test]
    fn test_expand_cnot_matrix_structure() {
        // Gate::cnot() expands X with control=0, target=1.
        // The resulting 4x4 matrix must swap |10⟩↔|11⟩ (indices 2↔3)
        // and leave |00⟩,|01⟩ unchanged.
        let cnot = super::super::gates::Gate::cnot();
        let m = &cnot.matrix;

        // |00⟩ → |00⟩
        assert!((m[[0, 0]] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        // |01⟩ → |01⟩
        assert!((m[[1, 1]] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        // |10⟩ → |11⟩
        assert!((m[[3, 2]] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        assert!(m[[2, 2]].norm() < 1e-12);
        // |11⟩ → |10⟩
        assert!((m[[2, 3]] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        assert!(m[[3, 3]].norm() < 1e-12);
    }

    #[test]
    fn test_expand_operator_qubit_ordering_3q() {
        // Expand X gate on target=qubit 1 in a 3-qubit system (no controls).
        // This should flip only the middle qubit.
        let x = super::super::gates::Gate::x();
        let expanded = expand_operator(&x.matrix, 3, &[1], &[]);

        // |000⟩ (0) → |010⟩ (2): flip qubit 1
        assert!((expanded[[2, 0]] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        assert!(expanded[[0, 0]].norm() < 1e-12);
        // |010⟩ (2) → |000⟩ (0)
        assert!((expanded[[0, 2]] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        // |101⟩ (5) → |111⟩ (7)
        assert!((expanded[[7, 5]] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        assert!(expanded[[5, 5]].norm() < 1e-12);
    }

    #[test]
    fn test_apply_local_vector_cnot_ordering() {
        // Apply the 4×4 CNOT matrix via apply_local_vector with targets=[0,1].
        // On |10⟩ (qubit 0=1, qubit 1=0), control fires → should flip to |11⟩.
        let cnot = super::super::gates::Gate::cnot();
        let mut psi = Array1::<Complex64>::zeros(4);
        psi[2] = Complex64::new(1.0, 0.0); // |10⟩

        let result = apply_local_vector(2, &psi, &cnot.matrix, &[0, 1], &[]);

        // Expected: |11⟩ (index 3)
        assert!(result[2].norm() < 1e-12, "|10⟩ should be empty");
        assert!(
            (result[3] - Complex64::new(1.0, 0.0)).norm() < 1e-12,
            "Expected |11⟩"
        );
    }

    #[test]
    fn test_apply_local_vector_cnot_no_fire() {
        // On |01⟩ (qubit 0=0, qubit 1=1), control is off → should stay |01⟩.
        let cnot = super::super::gates::Gate::cnot();
        let mut psi = Array1::<Complex64>::zeros(4);
        psi[1] = Complex64::new(1.0, 0.0); // |01⟩

        let result = apply_local_vector(2, &psi, &cnot.matrix, &[0, 1], &[]);

        assert!(
            (result[1] - Complex64::new(1.0, 0.0)).norm() < 1e-12,
            "|01⟩ must stay"
        );
    }

    #[test]
    fn test_apply_local_vector_cnot_reversed_targets() {
        // targets=[1,0] means gate qubit 0 (control) → state qubit 1,
        //                      gate qubit 1 (target) → state qubit 0.
        // On |01⟩ (qubit 0=0, qubit 1=1): control=qubit1=1 fires → flip qubit0 → |11⟩
        let cnot = super::super::gates::Gate::cnot();
        let mut psi = Array1::<Complex64>::zeros(4);
        psi[1] = Complex64::new(1.0, 0.0); // |01⟩

        let result = apply_local_vector(2, &psi, &cnot.matrix, &[1, 0], &[]);

        assert!(
            (result[3] - Complex64::new(1.0, 0.0)).norm() < 1e-12,
            "Expected |11⟩"
        );
        assert!(result[1].norm() < 1e-12, "|01⟩ should be empty");
    }

    #[test]
    fn test_apply_local_vector_swap_ordering() {
        // SWAP gate applied with targets=[0,1]: swap qubit 0 and qubit 1.
        // |10⟩ → |01⟩
        let swap = super::super::gates::Gate::swap();
        let mut psi = Array1::<Complex64>::zeros(4);
        psi[2] = Complex64::new(1.0, 0.0); // |10⟩

        let result = apply_local_vector(2, &psi, &swap.matrix, &[0, 1], &[]);

        assert!(
            (result[1] - Complex64::new(1.0, 0.0)).norm() < 1e-12,
            "Expected |01⟩"
        );
        assert!(result[2].norm() < 1e-12, "|10⟩ should be empty");
    }

    #[test]
    fn test_apply_local_vector_bell_state() {
        // Create Bell state: H|0⟩ on qubit 0, then CNOT[0,1].
        // Start from |00⟩, apply H to qubit 0 → (|00⟩+|10⟩)/√2
        let h = super::super::gates::Gate::h();
        let mut psi = Array1::<Complex64>::zeros(4);
        psi[0] = Complex64::new(1.0, 0.0); // |00⟩

        let psi = apply_local_vector(2, &psi, &h.matrix, &[0], &[]);

        // Now apply CNOT with targets=[0,1]
        let cnot = super::super::gates::Gate::cnot();
        let result = apply_local_vector(2, &psi, &cnot.matrix, &[0, 1], &[]);

        let s = 1.0 / 2.0_f64.sqrt();
        // Bell state Φ+: (|00⟩+|11⟩)/√2
        assert!(
            (result[0] - Complex64::new(s, 0.0)).norm() < 1e-12,
            "Expected |00⟩ amp"
        );
        assert!(result[1].norm() < 1e-12, "|01⟩ must be zero");
        assert!(result[2].norm() < 1e-12, "|10⟩ must be zero");
        assert!(
            (result[3] - Complex64::new(s, 0.0)).norm() < 1e-12,
            "Expected |11⟩ amp"
        );
    }

    #[test]
    fn test_apply_local_left_cnot_ordering() {
        // Apply CNOT to density matrix |10⟩⟨10| → should become |11⟩⟨11|.
        let cnot = super::super::gates::Gate::cnot();
        let mut rho = Array2::<Complex64>::zeros((4, 4));
        rho[[2, 2]] = Complex64::new(1.0, 0.0); // |10⟩⟨10|

        let result = apply_local_left(2, &rho, &cnot.matrix, &[0, 1], &[]);
        // Left multiplication: CNOT * |10⟩⟨10| → |11⟩⟨10|
        // Only column 2 is nonzero, row should shift from 2 to 3.
        assert!((result[[3, 2]] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        assert!(result[[2, 2]].norm() < 1e-12);
    }

    #[test]
    fn test_apply_local_right_cnot_ordering() {
        // Right multiplication: |10⟩⟨10| * CNOT† = |10⟩⟨11|
        // CNOT is self-adjoint, so CNOT† = CNOT.
        let cnot = super::super::gates::Gate::cnot();
        let mut rho = Array2::<Complex64>::zeros((4, 4));
        rho[[2, 2]] = Complex64::new(1.0, 0.0); // |10⟩⟨10|

        let result = apply_local_right(2, &rho, &cnot.matrix, &[0, 1], &[]);
        // Row 2 stays, column should shift from 2 to 3.
        assert!((result[[2, 3]] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        assert!(result[[2, 2]].norm() < 1e-12);
    }

    #[test]
    fn test_apply_local_left_right_full_unitary() {
        // Full unitary channel: ρ' = U ρ U†. For |10⟩⟨10| with CNOT → |11⟩⟨11|.
        let cnot = super::super::gates::Gate::cnot();
        let mut rho = Array2::<Complex64>::zeros((4, 4));
        rho[[2, 2]] = Complex64::new(1.0, 0.0);

        let tmp = apply_local_left(2, &rho, &cnot.matrix, &[0, 1], &[]);
        // CNOT is Hermitian (self-adjoint), so U† = U.
        let result = apply_local_right(2, &tmp, &cnot.matrix, &[0, 1], &[]);

        // |11⟩⟨11| → index [3,3] = 1.0
        assert!((result[[3, 3]] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        assert!(result[[2, 2]].norm() < 1e-12);
        assert!(result[[2, 3]].norm() < 1e-12);
        assert!(result[[3, 2]].norm() < 1e-12);
    }

    #[test]
    fn test_apply_local_vector_cnot_in_3q_system() {
        // 3-qubit system: apply CNOT(control=0, target=2), should skip qubit 1.
        // |100⟩ (index 4): control=qubit0=1 fires → flip qubit2 → |101⟩ (index 5)
        let cnot = super::super::gates::Gate::cnot();
        let mut psi = Array1::<Complex64>::zeros(8);
        psi[4] = Complex64::new(1.0, 0.0); // |100⟩

        let result = apply_local_vector(3, &psi, &cnot.matrix, &[0, 2], &[]);

        assert!(
            (result[5] - Complex64::new(1.0, 0.0)).norm() < 1e-12,
            "Expected |101⟩"
        );
        assert!(result[4].norm() < 1e-12, "|100⟩ should be empty");
    }

    #[test]
    fn test_apply_local_vector_cnot_3q_no_fire() {
        // 3-qubit: CNOT(control=0, target=2) on |010⟩ (index 2).
        // Control qubit 0 = 0 → no flip → stays |010⟩.
        let cnot = super::super::gates::Gate::cnot();
        let mut psi = Array1::<Complex64>::zeros(8);
        psi[2] = Complex64::new(1.0, 0.0); // |010⟩

        let result = apply_local_vector(3, &psi, &cnot.matrix, &[0, 2], &[]);

        assert!(
            (result[2] - Complex64::new(1.0, 0.0)).norm() < 1e-12,
            "|010⟩ must stay"
        );
    }

    #[test]
    fn test_sqrt_2x2_analytical_zero_trace() {
        let mat = ndarray::array![
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)]
        ];
        let sqrt = sqrt_positive_matrix(&mat);
        for &val in sqrt.iter() {
            assert!(val.norm() < 1e-12);
        }
    }

    #[test]
    fn test_sqrt_2x2_analytical_nonzero_det() {
        let mat = ndarray::array![
            [Complex64::new(4.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(9.0, 0.0)]
        ];
        let sqrt = sqrt_positive_matrix(&mat);
        assert!((sqrt[[0, 0]] - Complex64::new(2.0, 0.0)).norm() < 1e-12);
        assert!((sqrt[[1, 1]] - Complex64::new(3.0, 0.0)).norm() < 1e-12);
    }
}
