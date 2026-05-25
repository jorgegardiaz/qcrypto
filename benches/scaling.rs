//! Benchmarks for state construction scaling: wall time and memory vs qubit count.
//!
//! Empirically verifies the core memory claim of `qcrypto`: a `StateVector`
//! scales as O(2^N) and a `DensityMatrix` as O(4^N).
//!
//! The following workload is measured:
//!
//! 1. **GHZ construction** (`bench_scaling`): H followed by N-1 CNOTs.  This
//!    traverses the full state and is a standard representative workload.
//!
//! A CSV with *theoretical* memory per qubit count and formalism is also written.
//! Theoretical values are exact and allocator-independent; runtime RSS is
//! too noisy and OS-specific to be useful as a primary measurement.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use qcrypto::{Gate, QuantumChannel, QuantumState};
use std::hint::black_box;

#[path = "common.rs"]
mod common;

/// Qubit counts for `StateVector` benchmarks.
const SV_QUBITS: &[usize] = &[2, 3, 4, 5, 6, 8, 10, 12, 14];

/// Qubit counts for `DensityMatrix` benchmarks.
const DM_QUBITS: &[usize] = &[2, 3, 4, 5, 6, 8, 10, 12];

/// Theoretical byte size of an n-qubit `StateVector`: 2^n `Complex64` amplitudes.
fn sv_bytes(n: usize) -> u64 {
    (1u64 << n) * std::mem::size_of::<num_complex_shim::Complex64>() as u64
}

/// Theoretical byte size of an n-qubit `DensityMatrix`: (2^n)^2 = 4^n entries.
fn dm_bytes(n: usize) -> u64 {
    let dim = 1u64 << n;
    dim * dim * std::mem::size_of::<num_complex_shim::Complex64>() as u64
}

/// Local size shim so that `Complex64 = 2 × f64 = 16 bytes` is explicit and
/// does not introduce a hard dependency on `num-complex` in the bench binary.
mod num_complex_shim {
    #[allow(dead_code)]
    pub struct Complex64 {
        _re: f64,
        _im: f64,
    }
}

// ---------------------------------------------------------------------------
// State builders
// ---------------------------------------------------------------------------

fn build_ghz_sv(n: usize) -> QuantumState {
    let mut s = QuantumState::new(n);
    s.apply(&Gate::h(), &[0]).unwrap();
    for q in 0..(n - 1) {
        s.apply(&Gate::cnot(), &[q, q + 1]).unwrap();
    }
    s
}

fn build_ghz_dm(n: usize) -> QuantumState {
    let mut s = QuantumState::new(n);
    let id = QuantumChannel::identity();
    s.apply_channel(&id, &[0]).unwrap(); // promote to DensityMatrix
    s.apply(&Gate::h(), &[0]).unwrap();
    for q in 0..(n - 1) {
        s.apply(&Gate::cnot(), &[q, q + 1]).unwrap();
    }
    s
}

// ---------------------------------------------------------------------------
// Benchmark functions
// ---------------------------------------------------------------------------

fn bench_scaling(c: &mut Criterion) {
    common::write_environment();

    // Write theoretical memory reference values (exact, allocator-independent).
    let mut mem_csv =
        common::RawCsv::create("scaling_memory_theoretical.csv", "qubits,state_type,bytes");
    for &n in SV_QUBITS {
        mem_csv.row(format_args!("{n},StateVector,{}", sv_bytes(n)));
    }
    for &n in DM_QUBITS {
        mem_csv.row(format_args!("{n},DensityMatrix,{}", dm_bytes(n)));
    }
    mem_csv.flush();

    let mut group = c.benchmark_group("scaling/build_ghz");
    // 50 samples balances statistical resolution at low qubit counts
    // (microsecond range) against total runtime at 12 qubits.
    group.sample_size(50);

    for &n in SV_QUBITS {
        group.bench_with_input(BenchmarkId::new("StateVector", n), &n, |b, &n| {
            common::seed_thread();
            b.iter(|| black_box(build_ghz_sv(black_box(n))));
        });
    }
    for &n in DM_QUBITS {
        group.bench_with_input(BenchmarkId::new("DensityMatrix", n), &n, |b, &n| {
            common::seed_thread();
            b.iter(|| black_box(build_ghz_dm(black_box(n))));
        });
    }
    group.finish();
}

#[cfg(feature = "parallel")]
criterion_group!(benches, bench_scaling);
#[cfg(feature = "parallel")]
criterion_main!(benches);
