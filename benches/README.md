# qcrypto benchmark suite

Criterion-based benchmarks for `qcrypto` covering gate operations, noise
channels, state scaling, and complete QKD protocol runs.  All benchmarks are
deterministically seeded and write raw CSV output alongside criterion's
standard reports.

## Structure

```
benches/
├── common.rs          # shared seed, environment capture, CSV writer
├── core_ops.rs        # gates (X, CNOT, Toffoli), SV↔DM conversion, measurement
├── scaling.rs         # wall time and memory vs qubit count (GHZ)
├── channels.rs        # noise channel cost: cold (SV→DM) vs warm (DM only)
├── protocols.rs       # BB84 / B92 / BBM92 scaling and channel sweep + QBER CSV
├── data/              # generated: raw CSVs and environment.txt
├── figures/           # generated: PDF/PNG/EPS figures produced by analyze.py
└── scripts/
    ├── graphs/
    │   └── analyze.py         # reads criterion + CSVs → figures
    └── qiskit/
        └── compare_qiskit.py  # qcrypto vs Qiskit Aer comparison
    └── qutip/
        └── compare_qutip.py  # qcrypto vs QuTiP comparison
```

## What is benchmarked

### `channels.rs` — noise channel application

Qubit counts: **2, 3, 4, 5, 6, 8, 10**

Each of the six supported channels is measured at `p = 0.1` in two regimes:

| Channel | Cold (SV → DM + apply) | Warm (DM only) |
|---|:---:|:---:|
| `bit_flip` | ✓ | ✓ |
| `phase_flip` | ✓ | ✓ |
| `depolarizing` | ✓ | ✓ |
| `amplitude_damping` | ✓ | ✓ |
| `phase_damping` | ✓ | ✓ |
| `amplitude_phase_damping` | ✓ | ✓ |

The **cold** path includes the one-time `StateVector → DensityMatrix` promotion.
The **warm** path measures channel application on an already-promoted state.

### `core_ops.rs` — gates and measurement

Each gate is benchmarked on both `StateVector` and `DensityMatrix`:

| Gate | SV qubit counts | DM qubit counts |
|---|---|---|
| X (1 qubit) | 2–3–4–5–6–8–10–12–14 | 2–3–4–5–6–8–10 |
| CNOT (2 qubits) | 2–3–4–5–6–8–10–12–14 | 2–3–4–5–6–8–10 |
| Toffoli (3 qubits, n ≥ 3) | 3–4–5–6–8–10–12–14 | 3–4–5–6–8–10 |

Also measured:

- **SV → DM conversion** (identity channel, qubit counts 2–10)
- **Z-basis measurement** on a GHZ input at qubit counts 4, 6, 8, 10:
  single-qubit measurement and full all-qubit sequential measurement.

### `scaling.rs` — state construction scaling

Qubit counts: **2, 3, 4, 5, 6, 8, 10, 12** (both `StateVector` and `DensityMatrix`)

| Workload | StateVector | DensityMatrix |
|---|:---:|:---:|
| GHZ construction (H + N−1 CNOTs) | ✓ | ✓ |

Also writes `data/scaling_memory_theoretical.csv` with exact theoretical byte
counts (16 bytes per `Complex64` amplitude) for both formalisms at every qubit
count.

### `protocols.rs` — full QKD protocol runs

**Key-length scaling** at `bit_flip(0.01)`:

| Protocol | Key lengths |
|---|---|
| BB84 | 500 / 1 000 / 2 000 / 5 000 / 10 000 / 20 000 |
| B92 (optimal POVM) | same |
| BBM92 | same |

**Channel sweep** — BB84 at fixed **5 000 qubits**:

| Channel | p |
|---|---|
| noiseless (bit_flip) | 0.0 |
| bit_flip | 0.1 |
| phase_flip | 0.1 |
| depolarizing | 0.1 |
| amplitude_damping | 0.1 |

Each protocol run also records **QBER** and **established key length** to
`data/protocols_correctness.csv` for correctness verification.

## Running the benchmarks

### Rust benchmarks (criterion)

```bash
# Run all benchmark groups.
# Writes target/criterion/.../new/estimates.json and CSVs to benches/data/.
cargo bench

# Run a single file:
cargo bench --bench scaling
cargo bench --bench core_ops
cargo bench --bench channels
cargo bench --bench protocols

# Filter to a specific group within a file:
cargo bench --bench channels -- bit_flip
```

criterion HTML reports are written to `target/criterion/report/index.html`.

### Qiskit Aer comparison

```bash
cd benches/scripts/qiskit
uv run compare_qiskit.py \
    --qubits 4 6 8 10 12 --repeats 20 \
    --out ../../data/aer_results.csv \
    --env-out ../../data/aer_environment.txt
```

### Figures

```bash
cd benches/scripts/graphs
uv run analyze.py \
    --criterion-root ../../../target/criterion \
    --data ../../data \
    --aer ../../data/aer_results.csv \
    --out ../../figures
```

## Methodological notes

- Each timed iteration starts from a freshly prepared state via
  `iter_batched`, preventing accumulated state history from biasing results.
- Channels are measured in two regimes (`cold` / `warm`) to separate the
  one-time `SV → DM` conversion cost from the steady-state channel cost.
- Memory is reported as an exact theoretical bound (16 bytes per `Complex64`),
  not as measured RSS, which is noisy and allocator-dependent.
- The global RNG seed (`common::SEED`) is fixed across all benchmarks.
  Protocol runs are bit-for-bit reproducible with this seed.
- qcrypto timings come from criterion (median + confidence interval, outliers
  removed).  Aer timings come from `timeit` (median over N repetitions).
  These are different measurement methods and should not be compared directly
  without acknowledging the difference.
