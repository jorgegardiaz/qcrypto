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

/// Computes the Kronecker (Tensor) product of two matrices.
///
/// If `A` is an $m \times n$ matrix and `B` is a $p \times q$ matrix,
/// the result is an $mp \times nq$ matrix.
pub fn kronecker_product(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let (m, n) = a.dim();
    let (p, q) = b.dim();

    // Tensor product implementation using broadcasting
    // A is (m, n), B is (p, q)
    // We want output (m*p, n*q)

    // 1. Reshape A to (m, 1, n, 1)
    let a_expanded = a.view().insert_axis(Axis(1)).insert_axis(Axis(3));

    // 2. Reshape B to (1, p, 1, q)
    let b_expanded = b.view().insert_axis(Axis(0)).insert_axis(Axis(2));

    // 3. Broadcast multiply -> (m, p, n, q)
    let tensor_product = &a_expanded * &b_expanded;

    // 4. Reshape to (m*p, n*q)
    tensor_product
        .into_shape_with_order((m * p, n * q))
        .unwrap()
}

/// Computes the trace of a matrix (sum of diagonal elements).
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
pub fn expand_operator(
    num_total_qubits: usize,
    matrix: &Array2<Complex64>,
    targets: &[usize],
    controls: &[usize],
) -> Array2<Complex64> {
    let dim = 1 << num_total_qubits;
    let mut full_matrix = Array2::<Complex64>::zeros((dim, dim));
    // 1 in position c if qubit c is a control qubit
    let mut control_mask = 0usize;
    for &c in controls {
        control_mask |= 1 << c;
    }
    // 1 in position t if qubit t is a target qubit
    let mut target_mask = 0usize;
    for &t in targets {
        target_mask |= 1 << t;
    }
    // Bits that are not target (do not change)
    let passive_mask = !target_mask;
    // Itereation over every column, each column corrrespond to a basic state
    for col_idx in 0..dim {
        // If NOT ALL control qubits are 1 in the associated sate to the column
        // This checks if the state associeted to the column is affected by the matrix
        if (col_idx & control_mask) != control_mask {
            // This basic state is not affected by the matrix -> 1 in diagonal
            full_matrix[[col_idx, col_idx]] = Complex64::new(1.0, 0.0);
            continue;
        }
        // If ALL control qubits are 1 in the associated sate to the column
        // This means the basic state associated to the column is affected by the matrix
        // We extract the bits of col_idx in the positions of targets
        let small_col = extract_bits(col_idx, targets);
        // Itereation over the rows of the matrix applied to the subsystem of the target qubits
        for small_row in 0..matrix.nrows() {
            // Get the value of the matrix in the position associated to the iteration
            let val = matrix[[small_row, small_col]];
            // Check if val is 0
            if val.norm_sqr() < f64::EPSILON {
                // Due to floating-point representation machine epsilon is used
                continue;
            }
            // Construct global row index: preserve passive bits, update target bits
            // Scatter local 'small_row' bits to their physical target positions
            let new_target_bits = deposit_bits(small_row, targets);
            // Combine preserved passive bits with the new target bits
            let row_idx = (col_idx & passive_mask) | new_target_bits;
            // Populate full matrix entry
            full_matrix[[row_idx, col_idx]] = val;
        }
    }
    full_matrix
}

/// Extracs the bits in positions `indices` of the sequence `value`
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

/// Find duplicate in a slice of usize
pub fn find_duplicate(indices: &[usize]) -> Option<usize> {
    let mut seen = std::collections::HashSet::new();
    indices.iter().find(|&&idx| !seen.insert(idx)).copied()
}

/// Checks completeness relation for Kraus operators.
///
/// Verifies if $\sum M_k^\dagger M_k = I$.
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
        .all(|(a, b)| (a - b).norm() < 1e-9)
}

/// Checks POVM completeness relation.
///
/// Verifies if $\sum E_k = I$.
pub fn check_povm_completeness(ops: &[Array2<Complex64>], dim: usize) -> bool {
    let mut sum = Array2::<Complex64>::zeros((dim, dim));
    for op in ops {
        sum += op;
    }
    let identity = Array2::<Complex64>::eye(dim);
    (sum - identity).iter().all(|x| x.norm() < 1e-9)
}

/// Computes the outer product of two vectors $|a\rangle\langle b|$.
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

/// Checks if a matrix is Hermitian
pub fn is_hermitian(mat: &Array2<Complex64>, tol: f64) -> bool {
    mat.iter()
        .zip(mat.t().iter())
        .all(|(a, b)| (a - b.conj()).norm() < tol)
}

/// Applies a local quantum operator to the left side of a density matrix: `rho_new = U * rho`.
///
/// This function uses bitwise operations to perform a local tensor update,
/// avoiding the computationally expensive O(N^3) expansion of the local matrix
/// to the full 2^N x 2^N Hilbert space.
///
/// # Arguments
///
/// * `num_total_qubits` - Total number of qubits in the system (N).
/// * `rho` - The current density matrix of the full system.
/// * `local_matrix` - The small operator matrix (e.g., 2x2 for a single-qubit gate).
/// * `targets` - Indices of the target qubits the operator acts upon.
/// * `controls` - Indices of the control qubits.
pub fn apply_local_left(
    num_total_qubits: usize,
    rho: &Array2<Complex64>,
    local_matrix: &Array2<Complex64>,
    targets: &[usize],
    controls: &[usize],
) -> Array2<Complex64> {
    let dim = 1 << num_total_qubits;
    let mut new_rho = Array2::<Complex64>::zeros((dim, dim));

    // Build the control mask: bits are 1 at control qubit positions.
    let mut control_mask = 0usize;
    for &c in controls {
        control_mask |= 1 << c;
    }

    // Build the target mask: bits are 1 at target qubit positions.
    let mut target_mask = 0usize;
    for &t in targets {
        target_mask |= 1 << t;
    }

    // Passive mask: bits are 1 where qubits are neither targets nor controls.
    // Since the operator only acts on targets, passive bits remain unchanged.
    let passive_mask = !target_mask;

    // Dimension of the local operator (e.g., 2 for 1 target qubit, 4 for 2 target qubits).
    let k_dim = 1 << targets.len();

    // Iterate over all columns. Left multiplication only mixes row elements.
    for col in 0..dim {
        // Iterate over all rows.
        for row in 0..dim {
            // If the current basis state does not satisfy the control conditions,
            // the operator acts as the Identity matrix for this row.
            if (row & control_mask) != control_mask {
                new_rho[[row, col]] = rho[[row, col]];
                continue;
            }

            // Extract the bit values of the target qubits from the current physical row index.
            let small_row = extract_bits(row, targets);

            let mut sum = Complex64::new(0.0, 0.0);

            // Perform the local matrix-vector multiplication for the target subspace.
            for small_col in 0..k_dim {
                let val = local_matrix[[small_row, small_col]];

                // Skip negligible values to optimize performance (sparse matrix behavior).
                if val.norm_sqr() < f64::EPSILON {
                    continue;
                }

                // Reconstruct the physical row index 'm':
                // Combine the preserved passive bits of 'row' with the new target bits.
                let m = (row & passive_mask) | deposit_bits(small_col, targets);

                sum += val * rho[[m, col]];
            }

            new_rho[[row, col]] = sum;
        }
    }

    new_rho
}

/// Applies a local quantum operator to the right side of a density matrix: `rho_new = rho * U`.
///
/// Similar to `apply_local_left`, this function uses bitwise operations to update
/// the state locally, but it acts on the columns of the density matrix instead of the rows.
pub fn apply_local_right(
    num_total_qubits: usize,
    rho: &Array2<Complex64>,
    local_matrix: &Array2<Complex64>,
    targets: &[usize],
    controls: &[usize],
) -> Array2<Complex64> {
    let dim = 1 << num_total_qubits;
    let mut new_rho = Array2::<Complex64>::zeros((dim, dim));

    let mut control_mask = 0usize;
    for &c in controls {
        control_mask |= 1 << c;
    }

    let mut target_mask = 0usize;
    for &t in targets {
        target_mask |= 1 << t;
    }

    let passive_mask = !target_mask;
    let k_dim = 1 << targets.len();

    // Iterate over all rows. Right multiplication only mixes column elements.
    for row in 0..dim {
        for col in 0..dim {
            // If the current basis state does not satisfy the control conditions,
            // the operator acts as the Identity matrix for this column.
            if (col & control_mask) != control_mask {
                new_rho[[row, col]] = rho[[row, col]];
                continue;
            }

            // Extract the bit values of the target qubits from the current physical column index.
            let small_col_idx = extract_bits(col, targets);

            let mut sum = Complex64::new(0.0, 0.0);

            // Perform the local vector-matrix multiplication for the target subspace.
            for small_row_idx in 0..k_dim {
                // Note the index order for the local matrix: it's multiplied from the right.
                let val = local_matrix[[small_row_idx, small_col_idx]];

                if val.norm_sqr() < f64::EPSILON {
                    continue;
                }

                // Reconstruct the physical column index 'm'.
                let m = (col & passive_mask) | deposit_bits(small_row_idx, targets);

                sum += rho[[row, m]] * val;
            }

            new_rho[[row, col]] = sum;
        }
    }

    new_rho
}
