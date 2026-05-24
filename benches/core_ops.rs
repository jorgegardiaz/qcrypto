//! Benchmarks for core quantum operations: gates and measurement.
//!
//! Compares the cost of applying gates on a `StateVector` (O(2^N) memory) vs
//! a `DensityMatrix` (O(4^N) memory), demonstrating the trade-off at the heart
//! of the Dual-State Architecture.
//!
//! criterion is particularly valuable here because gates on small qubit counts
//! complete in microseconds or less — a regime where wall-clock measurement
//! (`Instant::now`) mostly captures its own overhead.  criterion runs each
//! operation in batches, subtracts overhead, and reports estimates with
//! confidence intervals.
//!
//! Each timed iteration starts from a freshly prepared state via `iter_batched`,
//! preventing the cost of one gate application from contaminating the next.
//! The mixed state is forced by applying an identity channel, which promotes
//! the state to a density matrix without changing its physics.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use qcrypto::{Gate, Measurement, QuantumChannel, QuantumState, Sampler};
use std::hint::black_box;

#[path = "common.rs"]
mod common;

/// Qubit range for `StateVector` benchmarks.
const SV_QUBITS: &[usize] = &[2, 3, 4, 5, 6, 8, 10, 12, 14];

/// Qubit range for `DensityMatrix` benchmarks.
///
/// The upper bound is kept lower than for `StateVector` because O(4^N) grows
/// rapidly: 10 qubits requires a 1024×1024 matrix of `Complex64` (~16 MB), and
/// operations are correspondingly more expensive.
const DM_QUBITS: &[usize] = &[2, 3, 4, 5, 6, 8, 10];

/// Returns a fresh n-qubit `StateVector` in |0…0⟩.
fn fresh_sv(n: usize) -> QuantumState {
    QuantumState::new(n)
}

/// Returns a fresh n-qubit state already promoted to `DensityMatrix`.
///
/// Applies an identity channel to force the `SV → DM` conversion without
/// altering the quantum state.
fn fresh_dm(n: usize) -> QuantumState {
    let mut s = QuantumState::new(n);
    let id = QuantumChannel::identity();
    s.apply_channel(&id, &[0]).expect("identity channel");
    s
}

fn bench_gates(c: &mut Criterion) {
    common::write_environment();

    // --- X gate (single qubit) ---
    let mut group = c.benchmark_group("gates/single_qubit_X");
    for &n in SV_QUBITS {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::new("StateVector", n), &n, |b, &n| {
            common::seed_thread();
            b.iter_batched(
                || fresh_sv(n),
                |mut s| {
                    s.apply(black_box(&Gate::x()), black_box(&[0])).unwrap();
                    s
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    for &n in DM_QUBITS {
        group.bench_with_input(BenchmarkId::new("DensityMatrix", n), &n, |b, &n| {
            common::seed_thread();
            b.iter_batched(
                || fresh_dm(n),
                |mut s| {
                    s.apply(black_box(&Gate::x()), black_box(&[0])).unwrap();
                    s
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();

    // --- CNOT gate (two qubits) ---
    let mut group = c.benchmark_group("gates/two_qubit_CNOT");
    for &n in SV_QUBITS {
        group.bench_with_input(BenchmarkId::new("StateVector", n), &n, |b, &n| {
            common::seed_thread();
            b.iter_batched(
                || fresh_sv(n),
                |mut s| {
                    s.apply(black_box(&Gate::cnot()), black_box(&[0, 1]))
                        .unwrap();
                    s
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    for &n in DM_QUBITS {
        group.bench_with_input(BenchmarkId::new("DensityMatrix", n), &n, |b, &n| {
            common::seed_thread();
            b.iter_batched(
                || fresh_dm(n),
                |mut s| {
                    s.apply(black_box(&Gate::cnot()), black_box(&[0, 1]))
                        .unwrap();
                    s
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();

    // --- Toffoli gate (three qubits) — requires n >= 3 ---
    let mut group = c.benchmark_group("gates/three_qubit_Toffoli");
    for &n in SV_QUBITS.iter().filter(|&&n| n >= 3) {
        group.bench_with_input(BenchmarkId::new("StateVector", n), &n, |b, &n| {
            common::seed_thread();
            b.iter_batched(
                || fresh_sv(n),
                |mut s| {
                    s.apply(black_box(&Gate::toffoli()), black_box(&[0, 1, 2]))
                        .unwrap();
                    s
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    for &n in DM_QUBITS.iter().filter(|&&n| n >= 3) {
        group.bench_with_input(BenchmarkId::new("DensityMatrix", n), &n, |b, &n| {
            common::seed_thread();
            b.iter_batched(
                || fresh_dm(n),
                |mut s| {
                    s.apply(black_box(&Gate::toffoli()), black_box(&[0, 1, 2]))
                        .unwrap();
                    s
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Benchmarks the one-time `StateVector → DensityMatrix` conversion cost.
///
/// This isolates the overhead incurred by the Dual-State Architecture the
/// first time a noise channel is applied to a pure state.
fn bench_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("conversion/sv_to_dm");
    let id = QuantumChannel::identity();
    for &n in DM_QUBITS {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            common::seed_thread();
            b.iter_batched(
                || fresh_sv(n),
                |mut s| {
                    // First channel application promotes the state to DensityMatrix.
                    s.apply_channel(black_box(&id), black_box(&[0])).unwrap();
                    s
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Benchmarks measurement and state collapse as a function of qubit count.
fn bench_measurement(c: &mut Criterion) {
    let mut group = c.benchmark_group("measurement/z_basis");
    let z = Measurement::z_basis();

    // Use a GHZ state as a representative entangled input.
    let make_ghz = |n: usize| {
        let mut s = QuantumState::new(n);
        s.apply(&Gate::h(), &[0]).unwrap();
        for q in 0..(n - 1) {
            s.apply(&Gate::cnot(), &[q, q + 1]).unwrap();
        }
        s
    };

    for &n in &[4usize, 6, 8, 10] {
        // Measure a single qubit.
        group.bench_with_input(BenchmarkId::new("measure_1_qubit", n), &n, |b, &n| {
            common::seed_thread();
            b.iter_batched(
                || make_ghz(n),
                |mut s| {
                    let _ = s.measure(black_box(&z), black_box(&[0])).unwrap();
                    s
                },
                criterion::BatchSize::SmallInput,
            );
        });
        // Measure all qubits sequentially.
        group.bench_with_input(BenchmarkId::new("measure_all_qubits", n), &n, |b, &n| {
            common::seed_thread();
            b.iter_batched(
                || make_ghz(n),
                |mut s| {
                    for q in 0..n {
                        let _ = s.measure(black_box(&z), black_box(&[q])).unwrap();
                    }
                    s
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Benchmarks `state.purity()` = Tr(ρ²) on density matrices.
///
/// Matches QuTiP's `metric_purity` task in `compare_qutip.py`, enabling
/// a direct timing comparison between qcrypto (Rust) and QuTiP (Python).
/// The input state is a genuinely mixed density matrix obtained by applying
/// amplitude_damping(0.3) to every qubit of the |0⟩ state.
fn bench_purity(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics/purity");

    let make_mixed_dm = |n: usize| -> QuantumState {
        let mut s = QuantumState::new(n);
        let ch = QuantumChannel::amplitude_damping(0.3);
        for q in 0..n {
            s.apply_channel(&ch, &[q]).unwrap();
        }
        s
    };

    for &n in DM_QUBITS {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            common::seed_thread();
            b.iter_batched(
                || make_mixed_dm(n),
                |s| black_box(s.purity()),
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Benchmarks multi-shot sampling via `Sampler`, matching Qiskit Aer's `sampling` bench.
///
/// `Sampler` is qcrypto's shot-based interface: it computes the probability distribution
/// from the amplitude vector *once* (O(2^N)), then draws each shot with a binary-search
/// CDF lookup (O(log 2^N) per shot) without ever collapsing or cloning the state.
/// This is architecturally equivalent to Qiskit Aer's shot-based simulation.
///
/// Contrast with `QuantumState::measure`, which collapses the state in place and must
/// be called on a freshly prepared copy for every shot — making it unsuitable for
/// high-shot benchmarks and not comparable with Qiskit's sampling cost.
///
fn bench_sampling(c: &mut Criterion) {
    let sampler = Sampler::new();

    let make_state = |n: usize| {
        let mut s = QuantumState::new(n);
        for q in 0..n {
            s.apply(&Gate::h(), &[q]).unwrap();
        }
        s
    };

    for (shots, label) in [(100_000usize, "100k"), (1_000_000, "1M")] {
        let mut group = c.benchmark_group(format!("sampling/{label}"));
        for &n in SV_QUBITS {
            let state = make_state(n);
            group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
                common::seed_thread();
                b.iter(|| {
                    black_box(
                        sampler
                            .run_par_computational_basis(black_box(&state), black_box(shots))
                            .unwrap(),
                    )
                });
            });
        }
        group.finish();
    }
}

criterion_group!(
    benches,
    bench_gates,
    bench_conversion,
    bench_measurement,
    bench_purity,
    bench_sampling
);
criterion_main!(benches);
