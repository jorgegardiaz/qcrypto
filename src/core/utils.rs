use ndarray::{Array2, Axis};
use num_complex::Complex64;

/// Computes the Kronecker (Tensor) product of two matrices.
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

/// Generates the full unitary matrix (2^N x 2^N) for the whole system
pub fn gen_operator(
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

/// Extracs the bits in positions `indices` of the sequence `value``
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

pub fn find_duplicate(indices: &[usize]) -> Option<usize> {
    let mut seen = std::collections::HashSet::new();
    indices.iter().find(|&&idx| !seen.insert(idx)).copied()
}
