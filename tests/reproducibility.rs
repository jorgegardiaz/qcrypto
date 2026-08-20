//! Reproducibility of parallel execution.
//!
//! These tests verify the property that the seeded-substream design of
//! [`qcrypto::rng::LocalRng`] exists to provide: that a seeded run produces
//! **bit-for-bit identical** output regardless of how many worker threads
//! Rayon uses to execute it.
//!
//! This is distinct from the determinism already covered elsewhere in the
//! suite, which only checks that the same seed run twice under the *same*
//! thread configuration yields the same result. That weaker property holds
//! trivially for any seeded RNG; it is invariance across scheduling that the
//! `draw_master_seed` + `LocalRng::child(master, stream_id)` design is
//! actually claimed to deliver.

#![cfg(feature = "parallel")]

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use qcrypto::protocols::{b92, bb84, bbm92, e91, gc01, qia_qzkp, sarg04, six_state};
use qcrypto::{Gate, Measurement, QuantumChannel, QuantumState, Sampler};

/// Thread counts swept by every invariance test.
const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8];

/// Master seed used across the whole file.
const SEED: u64 = 20260819;

/// Number of qubits / rounds per protocol run. Small enough to keep the
/// suite fast, large enough that a scheduling-dependent bug would show up.
const N: usize = 256;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Canonical fingerprint of a run.
///
/// Floats are hashed through `to_bits()` rather than compared with `==`, so
/// this is a genuine bit-for-bit check: it distinguishes `+0.0` from `-0.0`
/// and does not silently pass on `NaN`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Fingerprint {
    bits: Vec<u8>,
    floats: Vec<u64>,
    counts: Vec<usize>,
}

impl Fingerprint {
    fn digest(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }
}

/// Runs `f` on a freshly built Rayon pool with exactly `threads` workers.
fn on_pool<T, F>(threads: usize, f: F) -> T
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build Rayon thread pool");

    pool.install(|| {
        qcrypto::set_global_seed(SEED);
        f()
    })
}

/// Asserts that `run` produces the same fingerprint under every thread count.
///
/// On failure the message reports which thread count diverged and both
/// digests, so the report is directly usable as evidence.
fn assert_thread_invariant<F>(label: &str, run: F)
where
    F: Fn() -> Fingerprint + Send + Sync + Copy,
{
    let mut reference: Option<(usize, Fingerprint)> = None;

    for &threads in THREAD_COUNTS {
        let fp = on_pool(threads, run);

        match &reference {
            None => reference = Some((threads, fp)),
            Some((ref_threads, ref_fp)) => {
                assert_eq!(
                    *ref_fp,
                    fp,
                    "{label}: output differs between {ref_threads} and {threads} threads \
                     (digest {:#018x} vs {:#018x}). The seeded-substream design is \
                     supposed to make execution independent of the work-stealing \
                     scheduler; this failure means some code path draws randomness \
                     from the thread-local RNG inside the parallel region instead of \
                     from a LocalRng derived via LocalRng::child.",
                    ref_fp.digest(),
                    fp.digest(),
                );
            }
        }
    }
}

/// Canonicaliza una clave de bits a bytes para el fingerprint.
fn bits_to_bytes(bits: &[bool]) -> Vec<u8> {
    bits.iter().map(|&b| b as u8).collect()
}

// ─── QKD protocols ───────────────────────────────────────────────────────────

#[test]
fn bb84_par_is_thread_count_invariant() {
    assert_thread_invariant("BB84", || {
        let channel = QuantumChannel::bit_flip(0.05);
        let r = bb84::run_par(N, &channel, 0.1, 0.5).unwrap();
        Fingerprint {
            bits: bits_to_bytes(&r.alice_key),
            floats: vec![r.qber.to_bits()],
            counts: vec![
                r.raw_length,
                r.total_sifted,
                r.check_errors,
                r.eve_intercepted_count,
            ],
        }
    });
}

#[test]
fn b92_par_is_thread_count_invariant() {
    assert_thread_invariant("B92", || {
        let channel = QuantumChannel::bit_flip(0.05);
        let m = b92::build_optimal_povm_b92().unwrap();
        let r = b92::run_par(N, &channel, &m, 0.1, 0.5).unwrap();
        Fingerprint {
            bits: bits_to_bytes(&r.alice_key),
            floats: vec![r.qber.to_bits()],
            counts: vec![r.raw_length, r.conclusive_count, r.check_errors],
        }
    });
}

#[test]
fn bbm92_par_is_thread_count_invariant() {
    assert_thread_invariant("BBM92", || {
        let channel = QuantumChannel::depolarizing(0.05);
        let r = bbm92::run_par(N, &channel, &channel, 0.1, 0.5).unwrap();
        Fingerprint {
            bits: bits_to_bytes(&r.alice_key),
            floats: vec![r.qber.to_bits()],
            counts: vec![r.raw_length, r.total_sifted, r.check_errors],
        }
    });
}

#[test]
fn e91_par_is_thread_count_invariant() {
    assert_thread_invariant("E91", || {
        let channel = QuantumChannel::depolarizing(0.05);
        let r = e91::run_par(N, &channel, &channel, 0.1, 0.5).unwrap();
        Fingerprint {
            bits: bits_to_bytes(&r.alice_key),
            // El CHSH es el observable más sensible del paquete: cualquier
            // reordenación de la aleatoriedad lo mueve en los últimos bits.
            floats: vec![r.qber.unwrap_or(0.0).to_bits(), r.chsh_value.to_bits()],
            counts: vec![r.raw_length, r.total_sifted, r.check_errors],
        }
    });
}

#[test]
fn sarg04_par_is_thread_count_invariant() {
    assert_thread_invariant("SARG04", || {
        let channel = QuantumChannel::bit_flip(0.05);
        let r = sarg04::run_par(N, &channel, 0.1, 0.5).unwrap();
        Fingerprint {
            bits: bits_to_bytes(&r.alice_key),
            floats: vec![r.qber.to_bits()],
            counts: vec![r.raw_length, r.conclusive_count, r.check_errors],
        }
    });
}

#[test]
fn six_state_par_is_thread_count_invariant() {
    assert_thread_invariant("Six-State", || {
        let channel = QuantumChannel::bit_flip(0.05);
        let r = six_state::run_par(N, &channel, 0.1, 0.5).unwrap();
        Fingerprint {
            bits: bits_to_bytes(&r.alice_key),
            floats: vec![r.qber.to_bits()],
            counts: vec![r.raw_length, r.total_sifted, r.check_errors],
        }
    });
}

// ─── Non-QKD protocols ───────────────────────────────────────────────────────

#[test]
fn qia_qzkp_par_is_thread_count_invariant() {
    assert_thread_invariant("QIA-QZKP", || {
        let channel = QuantumChannel::bit_flip(0.05);
        let r = qia_qzkp::run_par(N, &channel, 0.9).unwrap();
        Fingerprint {
            bits: bits_to_bytes(&r.bob_recovered_c),
            floats: vec![r.accuracy.to_bits()],
            counts: vec![r.total_qubits, r.matches, r.authenticated as usize],
        }
    });
}

#[test]
fn gc01_par_is_thread_count_invariant() {
    assert_thread_invariant("GC01", || {
        let channel = QuantumChannel::bit_flip(0.05);
        let r = gc01::run_par(N, &channel, &channel, 0.1, 0.1).unwrap();
        Fingerprint {
            bits: Vec::new(),
            floats: vec![
                r.bob_mismatch_rate.to_bits(),
                r.charlie_mismatch_rate.to_bits(),
            ],
            counts: vec![
                r.num_qubits,
                r.bob_mismatches,
                r.charlie_mismatches,
                r.signature_accepted as usize,
            ],
        }
    });
}

// ─── Sampler ─────────────────────────────────────────────────────────────────

#[test]
fn sampler_par_is_thread_count_invariant() {
    assert_thread_invariant("Sampler::run_par", || {
        let mut state = QuantumState::new(2);
        state.apply(&Gate::h(), &[0]).unwrap();
        state.apply(&Gate::cnot(), &[0, 1]).unwrap();

        let sampler = Sampler::new().with_channel(QuantumChannel::depolarizing(0.1));
        let counts = sampler
            .run_par(&state, &Measurement::z_basis(), &[0], 20_000)
            .unwrap();

        // Orden canónico: HashMap no tiene orden estable entre ejecuciones.
        let mut entries: Vec<(String, usize)> = counts.into_iter().collect();
        entries.sort();

        Fingerprint {
            bits: entries
                .iter()
                .flat_map(|(k, _)| k.as_bytes().to_vec())
                .collect(),
            floats: Vec::new(),
            counts: entries.iter().map(|(_, v)| *v).collect(),
        }
    });
}

// ─── The mechanism itself ────────────────────────────────────────────────────

/// `LocalRng::child` must be a pure function of `(master_seed, stream_id)`:
/// the thread that evaluates it is irrelevant.
///
/// This is the primitive on which every test above depends. Checking it
/// directly means that when a protocol-level test fails, the fault can be
/// localised to the protocol driver rather than to the RNG.
#[test]
fn local_rng_child_is_thread_independent() {
    use qcrypto::rng::LocalRng;

    const STREAMS: u64 = 64;
    const DRAWS: usize = 32;

    let sequential: Vec<Vec<u64>> = (0..STREAMS)
        .map(|id| {
            let mut rng = LocalRng::child(SEED, id);
            (0..DRAWS).map(|_| rng.random_f64().to_bits()).collect()
        })
        .collect();

    for &threads in THREAD_COUNTS {
        let parallel: Vec<Vec<u64>> = on_pool(threads, || {
            use rayon::prelude::*;
            (0..STREAMS)
                .into_par_iter()
                .map(|id| {
                    let mut rng = LocalRng::child(SEED, id);
                    (0..DRAWS).map(|_| rng.random_f64().to_bits()).collect()
                })
                .collect()
        });

        assert_eq!(
            sequential, parallel,
            "LocalRng::child produced different streams on {threads} threads; \
             it must depend only on (master_seed, stream_id)"
        );
    }
}

/// Distinct stream ids must yield distinct sequences.
#[test]
fn local_rng_child_streams_are_pairwise_distinct() {
    use qcrypto::rng::LocalRng;
    use std::collections::HashSet;

    const STREAMS: u64 = 256;

    let mut seen = HashSet::new();
    for id in 0..STREAMS {
        let mut rng = LocalRng::child(SEED, id);
        let seq: Vec<u64> = (0..32).map(|_| rng.random_f64().to_bits()).collect();
        assert!(
            seen.insert(seq),
            "stream {id} collided with an earlier stream for master seed {SEED}"
        );
    }
}
