//! Benchmarks for complete protocol runs across all three qcrypto protocol families.
//!
//! Three families are measured, each with distinct output metrics:
//!
//! - **QKD** (BB84, B92, BBM92, E91, SixState, SARG04): key-length scaling
//!   under light noise (`bit_flip(0.01)`) → QBER + established key length.
//!   BB84 is also swept across all seven noise channels at a fixed key length.
//!
//! - **QIA** (QIA-QZKP): authentication-qubit scaling → accuracy + authenticated flag.
//!
//! - **QDS** (GC01): signature-qubit scaling → Bob/Charlie mismatch rates + accepted flag.
//!   GC01 applies a separate channel to each verifier, so the same channel is passed
//!   to both Bob and Charlie to measure symmetric noise.
//!
//! Each family writes its own correctness CSV because the output fields differ across
//! families.  All timing is captured by criterion and stored in `target/criterion/`.
//!
//! Reproducibility: `run_par` derives its randomness from the thread-local RNG via
//! `draw_master_seed`.  The RNG is reseeded with `common::seed_thread()` before each
//! benchmark group so results are bit-for-bit reproducible with `common::SEED`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use qcrypto::protocols::qds::gc01;
use qcrypto::protocols::qia::qia_qzkp;
use qcrypto::protocols::qkd::{b92, bb84, bbm92, e91, sarg04, six_state};
use qcrypto::QuantumChannel;
use std::hint::black_box;
use std::time::Duration;

#[path = "common.rs"]
mod common;

/// Key lengths for the QKD key-length scaling sweep.
const KEY_LENGTHS: &[usize] = &[500, 1_000, 2_000, 5_000, 10_000, 20_000];

/// Qubit counts for QIA-QZKP authentication scaling.
const QIA_SIZES: &[usize] = &[500, 1_000, 2_000, 5_000, 10_000, 20_000];

/// Qubit counts for GC01 digital-signature scaling.
///
/// Smaller upper bound than QKD: GC01 runs SWAP tests on two independent
/// channels per qubit, making each iteration roughly twice as expensive.
const QDS_SIZES: &[usize] = &[50, 100, 200, 500, 1_000, 2_000];

const EVE_RATIO: f64 = 0.0;
const CHECK_RATIO: f64 = 0.2;

/// Minimum accuracy for QIA-QZKP authentication to succeed.
const QIA_THRESHOLD: f64 = 0.9;

/// Maximum mismatch rate for GC01 signature to be accepted.
const GC01_THRESHOLD: f64 = 0.1;

/// All noise channels used in the channel sweep.
fn channel_sweep() -> Vec<(&'static str, QuantumChannel)> {
    vec![
        ("noiseless", QuantumChannel::bit_flip(0.0)),
        ("bit_flip_0.1", QuantumChannel::bit_flip(0.1)),
        ("phase_flip_0.1", QuantumChannel::phase_flip(0.1)),
        ("depolarizing_0.1", QuantumChannel::depolarizing(0.1)),
        ("amplitude_damping_0.1", QuantumChannel::amplitude_damping(0.1)),
        ("phase_damping_0.1", QuantumChannel::phase_damping(0.1)),
        (
            "amp_phase_damping_0.1",
            QuantumChannel::combined_amplitude_phase_damping(0.1, 0.1),
        ),
    ]
}

// ---------------------------------------------------------------------------
// QKD key-length scaling: BB84 / B92 / BBM92 / E91 / SixState / SARG04
// ---------------------------------------------------------------------------

fn bench_qkd_scaling(c: &mut Criterion) {
    common::write_environment();

    let mut csv = common::RawCsv::create(
        "protocols_qkd_correctness.csv",
        "protocol,channel,key_length,qber,alice_key_len",
    );

    // Light baseline channel (1 % bit-flip) for the key-length sweep.
    let channel = QuantumChannel::bit_flip(0.01);
    let b92_meas = b92::build_optimal_povm_b92().expect("B92 POVM construction");

    let mut group = c.benchmark_group("protocols/qkd_key_length_scaling");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(8));
    group.warm_up_time(Duration::from_secs(2));

    for &len in KEY_LENGTHS {
        // --- BB84 ---
        group.bench_with_input(BenchmarkId::new("BB84", len), &len, |b, &len| {
            common::seed_thread();
            b.iter(|| {
                black_box(bb84::run_par(black_box(len), &channel, EVE_RATIO, CHECK_RATIO).unwrap())
            });
        });
        {
            common::seed_thread();
            let r = bb84::run_par(len, &channel, EVE_RATIO, CHECK_RATIO).unwrap();
            csv.row(format_args!(
                "BB84,bit_flip_0.01,{len},{:.6},{}",
                r.qber,
                r.alice_key.len()
            ));
        }

        // --- B92 ---
        group.bench_with_input(BenchmarkId::new("B92", len), &len, |b, &len| {
            common::seed_thread();
            b.iter(|| {
                black_box(
                    b92::run_par(black_box(len), &channel, &b92_meas, EVE_RATIO, CHECK_RATIO)
                        .unwrap(),
                )
            });
        });
        {
            common::seed_thread();
            let r = b92::run_par(len, &channel, &b92_meas, EVE_RATIO, CHECK_RATIO).unwrap();
            csv.row(format_args!(
                "B92,bit_flip_0.01,{len},{:.6},{}",
                r.qber,
                r.alice_key.len()
            ));
        }

        // --- BBM92 (entanglement-based; same channel for Alice and Bob) ---
        group.bench_with_input(BenchmarkId::new("BBM92", len), &len, |b, &len| {
            common::seed_thread();
            b.iter(|| {
                black_box(
                    bbm92::run_par(black_box(len), &channel, &channel, EVE_RATIO, CHECK_RATIO)
                        .unwrap(),
                )
            });
        });
        {
            common::seed_thread();
            let r = bbm92::run_par(len, &channel, &channel, EVE_RATIO, CHECK_RATIO).unwrap();
            csv.row(format_args!(
                "BBM92,bit_flip_0.01,{len},{:.6},{}",
                r.qber,
                r.alice_key.len()
            ));
        }

        // --- E91 (entanglement-based; qber is Option<f64> — always Some when check_ratio > 0) ---
        group.bench_with_input(BenchmarkId::new("E91", len), &len, |b, &len| {
            common::seed_thread();
            b.iter(|| {
                black_box(
                    e91::run_par(black_box(len), &channel, &channel, EVE_RATIO, CHECK_RATIO)
                        .unwrap(),
                )
            });
        });
        {
            common::seed_thread();
            let r = e91::run_par(len, &channel, &channel, EVE_RATIO, CHECK_RATIO).unwrap();
            csv.row(format_args!(
                "E91,bit_flip_0.01,{len},{:.6},{}",
                r.qber.unwrap_or(f64::NAN),
                r.alice_key.len()
            ));
        }

        // --- SixState ---
        group.bench_with_input(BenchmarkId::new("SixState", len), &len, |b, &len| {
            common::seed_thread();
            b.iter(|| {
                black_box(
                    six_state::run_par(black_box(len), &channel, EVE_RATIO, CHECK_RATIO).unwrap(),
                )
            });
        });
        {
            common::seed_thread();
            let r = six_state::run_par(len, &channel, EVE_RATIO, CHECK_RATIO).unwrap();
            csv.row(format_args!(
                "SixState,bit_flip_0.01,{len},{:.6},{}",
                r.qber,
                r.alice_key.len()
            ));
        }

        // --- SARG04 ---
        group.bench_with_input(BenchmarkId::new("SARG04", len), &len, |b, &len| {
            common::seed_thread();
            b.iter(|| {
                black_box(
                    sarg04::run_par(black_box(len), &channel, EVE_RATIO, CHECK_RATIO).unwrap(),
                )
            });
        });
        {
            common::seed_thread();
            let r = sarg04::run_par(len, &channel, EVE_RATIO, CHECK_RATIO).unwrap();
            csv.row(format_args!(
                "SARG04,bit_flip_0.01,{len},{:.6},{}",
                r.qber,
                r.alice_key.len()
            ));
        }
    }
    group.finish();
    csv.flush();
}

// ---------------------------------------------------------------------------
// BB84 channel sweep: fixed key length, all seven noise channels
// ---------------------------------------------------------------------------

fn bench_qkd_channel_sweep(c: &mut Criterion) {
    let mut csv = common::RawCsv::append("protocols_qkd_correctness.csv");

    let len = 5_000usize;
    let channels = channel_sweep();

    let mut group = c.benchmark_group("protocols/bb84_by_channel");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(8));

    for (name, ch) in &channels {
        group.bench_with_input(BenchmarkId::from_parameter(name), ch, |b, ch| {
            common::seed_thread();
            b.iter(|| {
                black_box(bb84::run_par(black_box(len), ch, EVE_RATIO, CHECK_RATIO).unwrap())
            });
        });
        {
            common::seed_thread();
            let r = bb84::run_par(len, ch, EVE_RATIO, CHECK_RATIO).unwrap();
            csv.row(format_args!(
                "BB84,{name},{len},{:.6},{}",
                r.qber,
                r.alice_key.len()
            ));
        }
    }
    group.finish();
    csv.flush();
}

// ---------------------------------------------------------------------------
// QIA-QZKP scaling
// ---------------------------------------------------------------------------

fn bench_qia_scaling(c: &mut Criterion) {
    let mut csv = common::RawCsv::create(
        "protocols_qia_correctness.csv",
        "protocol,channel,num_qubits,accuracy,authenticated",
    );

    let channel = QuantumChannel::bit_flip(0.01);

    let mut group = c.benchmark_group("protocols/qia_scaling");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(8));
    group.warm_up_time(Duration::from_secs(2));

    for &n in QIA_SIZES {
        group.bench_with_input(BenchmarkId::new("QIA-QZKP", n), &n, |b, &n| {
            common::seed_thread();
            b.iter(|| {
                black_box(qia_qzkp::run_par(black_box(n), &channel, QIA_THRESHOLD).unwrap())
            });
        });
        {
            common::seed_thread();
            let r = qia_qzkp::run_par(n, &channel, QIA_THRESHOLD).unwrap();
            csv.row(format_args!(
                "QIA-QZKP,bit_flip_0.01,{n},{:.6},{}",
                r.accuracy, r.authenticated
            ));
        }
    }
    group.finish();
    csv.flush();
}

// ---------------------------------------------------------------------------
// GC01 digital-signature scaling
// ---------------------------------------------------------------------------

fn bench_qds_scaling(c: &mut Criterion) {
    let mut csv = common::RawCsv::create(
        "protocols_qds_correctness.csv",
        "protocol,channel,num_qubits,bob_mismatch_rate,charlie_mismatch_rate,signature_accepted",
    );

    // Symmetric noise: same channel applied to both verifier legs.
    let channel = QuantumChannel::bit_flip(0.01);

    let mut group = c.benchmark_group("protocols/qds_scaling");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(8));
    group.warm_up_time(Duration::from_secs(2));

    for &n in QDS_SIZES {
        group.bench_with_input(BenchmarkId::new("GC01", n), &n, |b, &n| {
            common::seed_thread();
            b.iter(|| {
                black_box(
                    gc01::run_par(black_box(n), &channel, &channel, EVE_RATIO, GC01_THRESHOLD)
                        .unwrap(),
                )
            });
        });
        {
            common::seed_thread();
            let r = gc01::run_par(n, &channel, &channel, EVE_RATIO, GC01_THRESHOLD).unwrap();
            csv.row(format_args!(
                "GC01,bit_flip_0.01,{n},{:.6},{:.6},{}",
                r.bob_mismatch_rate, r.charlie_mismatch_rate, r.signature_accepted
            ));
        }
    }
    group.finish();
    csv.flush();
}

criterion_group!(
    benches,
    bench_qkd_scaling,
    bench_qkd_channel_sweep,
    bench_qia_scaling,
    bench_qds_scaling
);
criterion_main!(benches);
