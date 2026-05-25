//! Benchmarks for noise channel application.
//!
//! Measures the cost of applying each channel supported by `qcrypto`.
//!
//! Applying a channel to a `StateVector` triggers a one-time conversion to a
//! density matrix on the first call.  To separate conversion cost from steady-
//! state channel cost, each channel is benchmarked in two scenarios:
//!
//! - **cold**: state is still a `StateVector` — the first application includes
//!   the `SV → DM` promotion.
//! - **warm**: state is already a `DensityMatrix` — measures channel application
//!   only.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use qcrypto::{QuantumChannel, QuantumState};
use std::hint::black_box;

#[path = "common.rs"]
mod common;

const QUBITS: &[usize] = &[2, 3, 4, 5, 6, 8, 10];

/// Returns the set of channels to benchmark, paired with a display name.
fn channels() -> Vec<(&'static str, QuantumChannel)> {
    vec![
        ("bit_flip", QuantumChannel::bit_flip(0.1)),
        ("phase_flip", QuantumChannel::phase_flip(0.1)),
        ("depolarizing", QuantumChannel::depolarizing(0.1)),
        ("amplitude_damping", QuantumChannel::amplitude_damping(0.1)),
        ("phase_damping", QuantumChannel::phase_damping(0.1)),
        (
            "amp_phase_damping",
            QuantumChannel::combined_amplitude_phase_damping(0.1, 0.1),
        ),
    ]
}

/// Returns an n-qubit state in the `StateVector` representation (cold path).
fn cold_state(n: usize) -> QuantumState {
    QuantumState::new(n)
}

/// Returns an n-qubit state already promoted to `DensityMatrix` (warm path).
fn warm_state(n: usize) -> QuantumState {
    let mut s = QuantumState::new(n);
    s.apply_channel(&QuantumChannel::identity(), &[0]).unwrap();
    s
}

fn bench_channels(c: &mut Criterion) {
    common::write_environment();

    let mut group = c.benchmark_group("channels/apply_single_qubit");
    group.sample_size(50);

    for (name, ch) in channels() {
        for &n in QUBITS {
            // Cold path: includes the SV → DM conversion on the first application.
            let id_cold = BenchmarkId::new(format!("{name}/cold"), n);
            group.bench_with_input(id_cold, &n, |b, &n| {
                common::seed_thread();
                b.iter_batched(
                    || cold_state(n),
                    |mut s| {
                        s.apply_channel(black_box(&ch), black_box(&[0])).unwrap();
                        s
                    },
                    criterion::BatchSize::SmallInput,
                );
            });

            // Warm path: state is already a density matrix; measures channel only.
            let id_warm = BenchmarkId::new(format!("{name}/warm"), n);
            group.bench_with_input(id_warm, &n, |b, &n| {
                common::seed_thread();
                b.iter_batched(
                    || warm_state(n),
                    |mut s| {
                        s.apply_channel(black_box(&ch), black_box(&[0])).unwrap();
                        s
                    },
                    criterion::BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

#[cfg(feature = "parallel")]
criterion_group!(benches, bench_channels);
#[cfg(feature = "parallel")]
criterion_main!(benches);
