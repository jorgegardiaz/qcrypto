# qcrypto Benchmark Suite

This directory contains the complete benchmark suite for `qcrypto`. It covers four areas: state-vector and density-matrix scaling, core gate throughput, noise channel application cost, and end-to-end quantum protocol simulation. Two Python comparison scripts measure equivalent workloads against Qiskit Aer and QuTiP, and a shell script orchestrates the full pipeline.

## Table of Contents

- [Running the Full Pipeline](#running-the-full-pipeline)
- [Rust Benchmarks](#rust-benchmarks)
  - [1. State Construction and Scaling](#1-state-construction-and-scaling-scalingrs)
  - [2. Core Gate Operations](#2-core-gate-operations-core_opsrs)
  - [3. Noise Channel Application](#3-noise-channel-application-channelsrs)
  - [4. Protocol Simulation](#4-protocol-simulation-protocolsrs)
    - [QKD Key-Length Scaling](#qkd-key-length-scaling-protocolsqkd_key_length_scaling)
    - [BB84 Channel Sweep](#bb84-channel-sweep-protocolsbb84_by_channel)
    - [QIA-QZKP Scaling](#qia-qzkp-scaling-protocolsqia_scaling)
    - [GC01 Digital Signature Scaling](#gc01-digital-signature-scaling-protocolsqds_scaling)
- [Comparison Scripts](#comparison-scripts)
  - [Qiskit Aer Comparison](#qiskit-aer-comparison-benchesscriptsqiskitcompare_qiskitpy)
  - [QuTiP Comparison](#qutip-comparison-benchesscriptsqutipcompare_qutippy)
- [Statistical Methodology](#statistical-methodology)
- [Plot Color Conventions](#plot-color-conventions)

---

## Running the Full Pipeline

The `reproduce.sh` script runs all four steps in order: Rust benchmarks, Qiskit Aer comparison, QuTiP comparison, and figure generation.

```bash
./benches/scripts/reproduce.sh [options]
```

| Option | Description |
| :--- | :--- |
| `--skip-rust` | Skip the Rust benchmarks; reuse existing data from `target/criterion/`. |
| `--skip-qiskit` | Skip the Qiskit Aer comparison step. |
| `--skip-qutip` | Skip the QuTiP comparison step. |
| `--filter <pattern>` | Pass a filter to `cargo bench` (e.g. `--filter scaling`). |

Each step writes its output to a fixed location before the next step begins:

| Step | Output |
| :--- | :--- |
| Rust criterion | `target/criterion/` + `benches/data/*.csv` |
| Qiskit Aer | `benches/data/qiskit/aer_results.csv` |
| QuTiP | `benches/data/qutip/qutip_results.csv` |
| Figures | `benches/figures/` |

---

## Rust Benchmarks

All Rust benchmarks use [Criterion](https://github.com/bheisler/criterion.rs). Each measurement is taken with `iter_batched`, so the state is freshly prepared before each timed iteration and setup cost is excluded. The thread-local RNG is reseeded with a fixed constant (`common::SEED`) before every group, making results bit-for-bit reproducible across runs.

### 1. State Construction and Scaling (`scaling.rs`)

Measures wall time and theoretical memory footprint as a function of qubit count, for both the `StateVector` and `DensityMatrix` representations.

**GHZ construction** (`scaling/build_ghz`): The workload is a Hadamard gate on qubit 0 followed by a chain of N−1 CNOTs, producing the maximally entangled state (|00...0⟩ + |11...1⟩) / √2. This traverses the full state amplitude array and is a representative construction task.

- `StateVector`: N ∈ {2, 3, 4, 5, 6, 8, 10, 12, 14}
- `DensityMatrix`: N ∈ {2, 3, 4, 5, 6, 8, 10, 12}

**Theoretical memory** (`scaling_memory_theoretical.csv`): A CSV is written alongside the timing data with exact byte counts: 16 bytes × 2^N for `StateVector` and 16 bytes × 4^N for `DensityMatrix`. These values are derived analytically (one `Complex64` = two `f64` = 16 bytes) and are allocator-independent; runtime RSS is too noisy and OS-specific to serve as a primary measurement.

---

### 2. Core Gate Operations (`core_ops.rs`)

Measures the cost of individual gate and measurement operations on both representations. Because gates on small qubit counts complete in microseconds or less, Criterion's batched measurement is essential here: it runs each operation in large batches, subtracts per-iteration overhead, and reports an estimate with confidence intervals.

**Gate throughput** (`gates/single_qubit_X`, `gates/two_qubit_CNOT`, `gates/three_qubit_Toffoli`): Each gate is applied to qubit 0 (or qubits 0, 1, 2 for the Toffoli) of a freshly allocated state. The state starts in |0...0⟩ for `StateVector` or is promoted to `DensityMatrix` via an identity channel application before the timed loop.

- X gate (Pauli-X, single qubit): SV N ∈ {2..14}, DM N ∈ {2..10}
- CNOT gate (two qubits): SV N ∈ {2..14}, DM N ∈ {2..10}
- Toffoli gate (three qubits): SV N ∈ {3..14}, DM N ∈ {3..10}

**StateVector → DensityMatrix conversion** (`conversion/sv_to_dm`): Isolates the one-time overhead of the first channel application on a pure state. `qcrypto` uses a lazy dual-state architecture: a state starts as a `StateVector` and is promoted to a `DensityMatrix` only when a noise channel is first applied. This benchmark measures that promotion cost separately from steady-state channel cost.

- Range: DM N ∈ {2, 3, 4, 5, 6, 8, 10}

**Measurement and collapse** (`measurement/z_basis`): Applied to an N-qubit GHZ state. Two sub-benchmarks are measured at each qubit count:

- `measure_1_qubit`: A single qubit-0 measurement; collapses the state once.
- `measure_all_qubits`: Sequential measurement of all N qubits; collapses the state N times, with each measurement operating on the post-collapse state of the previous step.
- Range: N ∈ {4, 6, 8, 10}

**Purity Tr(ρ²)** (`metrics/purity`): Measures the cost of computing the purity of a genuinely mixed density matrix. The mixed state is prepared by applying `amplitude_damping(0.3)` to every qubit of |0...0⟩ before the timed loop. This benchmark is designed to be directly comparable to QuTiP's `metric_purity` task in `compare_qutip.py`.

- Range: DM N ∈ {2, 3, 4, 5, 6, 8, 10}

**Multi-shot sampling** (`sampling/100k`, `sampling/1M`): Benchmarks the `Sampler` interface for high-shot experiments. `Sampler` computes the full probability distribution from the amplitude vector once (O(2^N)), then performs M binary-search CDF lookups (O(log 2^N) per shot) without collapsing or cloning the state. This is the correct architectural equivalent to Qiskit Aer's shot-based simulation. Note that `QuantumState::measure` collapses the state in place and cannot be used for multi-shot experiments without re-preparing the state for every shot.

- Shot counts: 100,000 and 1,000,000
- Input state: H⊗N|0...0⟩ (uniform superposition over all basis states)
- Range: SV N ∈ {2, 3, 4, 5, 6, 8, 10, 12, 14}

---

### 3. Noise Channel Application (`channels.rs`)

Quantifies the cost of applying each of the six noise channels supported by `qcrypto` to a single target qubit of an N-qubit state.

**Channels** (all evaluated at p = 0.1):

| Name | Description |
| :--- | :--- |
| `bit_flip` | Flips the qubit with probability p: E(ρ) = (1−p)ρ + p·X·ρ·X |
| `phase_flip` | Applies a phase flip with probability p: E(ρ) = (1−p)ρ + p·Z·ρ·Z |
| `depolarizing` | Uniform mixture with maximally mixed state: E(ρ) = (1−p)ρ + p·I/2 |
| `amplitude_damping` | Models energy loss (T1 decay): K0=[[1,0],[0,√(1−p)]], K1=[[0,√p],[0,0]] |
| `phase_damping` | Models pure dephasing (T2 decay) without energy loss |
| `amp_phase_damping` | Combined amplitude and phase damping |

**Cold vs. warm paths**: Each channel is benchmarked in two regimes to separate two distinct costs:

- **Cold** (`<channel>/cold`): The input is still a `StateVector`. The first channel application triggers the SV → DM promotion, so this measures the total cost: conversion plus channel application.
- **Warm** (`<channel>/warm`): The input is already a `DensityMatrix` (promoted beforehand). This measures the steady-state cost of channel application only.

- Range: N ∈ {2, 3, 4, 5, 6, 8, 10}

---

### 4. Protocol Simulation (`protocols.rs`)

End-to-end timing for all three protocol families implemented in `qcrypto`: quantum key distribution (QKD), quantum identity authentication (QIA), and quantum digital signatures (QDS). Each family writes a correctness CSV alongside the timing data so that protocol behavior under noise can be verified independently.

All protocols use the parallelized `run_par` variant. The thread-local RNG is reseeded with `common::SEED` before each group, ensuring reproducible correctness values.

#### QKD Key-Length Scaling (`protocols/qkd_key_length_scaling`)

Measures execution time as a function of requested key length for six QKD protocols, under a light baseline channel of `bit_flip(0.01)`. The correctness CSV (`protocols_qkd_correctness.csv`) records the QBER and final Alice key length for each run.

| Protocol | Description |
| :--- | :--- |
| BB84 | Standard four-state BB84 with rectilinear and diagonal bases |
| B92 | Two-state protocol using an optimal POVM measurement |
| BBM92 | Entanglement-based variant of BB84 using Bell pairs |
| E91 | Ekert 1991 entanglement-based protocol with Bell inequality testing |
| SixState | Six-state protocol with higher information-theoretic security bound |
| SARG04 | Sarg 2004 protocol, resistant to photon-number-splitting attacks |

- Key lengths: 500, 1 000, 2 000, 5 000, 10 000, 20 000 qubits
- Eve interception ratio: 0.0 (no eavesdropping)
- Check ratio: 0.2 (20% of raw bits used for error estimation)

#### BB84 Channel Sweep (`protocols/bb84_by_channel`)

Runs BB84 at a fixed key length of 5 000 qubits across all seven noise channels (including the noiseless baseline), measuring both execution time and the resulting QBER for each channel type.

Channels: noiseless, bit_flip(0.1), phase_flip(0.1), depolarizing(0.1), amplitude_damping(0.1), phase_damping(0.1), amp_phase_damping(0.1, 0.1).

#### QIA-QZKP Scaling (`protocols/qia_scaling`)

Measures execution time for the Quantum Identity Authentication with Zero-Knowledge Proof (QIA-QZKP) protocol as a function of authentication qubit count. The correctness CSV (`protocols_qia_correctness.csv`) records the authentication accuracy and the boolean `authenticated` result for each run. A minimum accuracy threshold of 0.9 is required for the protocol to return an authenticated result.

- Qubit counts: 500, 1 000, 2 000, 5 000, 10 000, 20 000
- Baseline channel: `bit_flip(0.01)`

#### GC01 Digital Signature Scaling (`protocols/qds_scaling`)

Measures execution time for the GC01 quantum digital signature scheme as a function of signature qubit count. GC01 runs SWAP tests across two independent verifier legs (Bob and Charlie), making each iteration roughly twice as expensive as a QKD run of the same size; the upper bound is therefore smaller than for QKD. The correctness CSV (`protocols_qds_correctness.csv`) records Bob and Charlie mismatch rates and the boolean `signature_accepted` result.

- Qubit counts: 50, 100, 200, 500, 1 000, 2 000
- Symmetric noise: the same `bit_flip(0.01)` channel is applied to both verifier legs
- Acceptance threshold: maximum mismatch rate of 0.1

---

## Comparison Scripts

### Qiskit Aer Comparison (`benches/scripts/qiskit/compare_qiskit.py`)

Compares `qcrypto` (Rust, Criterion) against Qiskit Aer (Python/C++) for the workloads where a fair, equivalent comparison is possible.

```bash
cd benches/scripts/qiskit
uv run compare_qiskit.py [options]
```

| Option | Default | Description |
| :--- | :--- | :--- |
| `--qubits` | `2 3 4 5 6 8 10 12 14` | Qubit counts for GHZ and gate benchmarks. |
| `--shots` | `100000 1000000` | Shot counts for the sampling benchmark. |
| `--repeats` | `50` | Number of independent repetitions per data point (used for bootstrapping). |
| `--skip` | (none) | Skip one or more tasks: `ghz`, `sampling`, `gates`. |
| `--out` | `../data/qiskit/aer_results.csv` | Output CSV path. |
| `--env-out` | `../data/qiskit/aer_environment.txt` | Output path for version and platform metadata. |

**Benchmarked tasks:**

- **GHZ state vector** (`statevector_ghz`): Build an N-qubit GHZ state using Hadamard + chain of CNOTs, then extract the state vector. Timed end-to-end including circuit execution.
- **Sampling** (`sampling`): Full N-qubit bitstring sampling from an H⊗N|0⟩ uniform superposition. Runs with the specified shot counts. Timed end-to-end including circuit compilation, execution, and result extraction.
- **Gate throughput** (`gate_X`, `gate_CX`, `gate_CCX`): Apply a single gate (X, CX, or CCX) to a pre-loaded state vector. The initial state (H on all qubits) is precomputed and injected via `set_statevector` before each timed run, so only the gate application cost is measured — mirroring `iter_batched` in Criterion.

**Methodological decisions:**

- `optimization_level=0` is used for all Aer benchmarks. This disables transpilation passes such as gate fusion and pattern simplification. Without it, recognizable circuits like GHZ can be simplified before simulation, artificially favouring Aer.
- Noisy density-matrix simulation is intentionally excluded. qcrypto and Aer model noise at different abstraction levels (Aer injects errors inline during circuit execution; qcrypto applies channels as explicit post-gate operations), so no single equivalent task exists.
- QKD protocols are excluded because Aer does not implement them.
- Cases where qcrypto is slower are not hidden or filtered.

---

### QuTiP Comparison (`benches/scripts/qutip/compare_qutip.py`)

Compares `qcrypto` against QuTiP (Python) for open-system density-matrix operations: noise channel application, purity computation, and Lindblad evolution. Gate simulation and sampling are handled by the Qiskit comparison and are excluded here.

```bash
cd benches/scripts/qutip
uv run compare_qutip.py [options]
```

| Option | Default | Description |
| :--- | :--- | :--- |
| `--dm-qubits` | `2 3 4 5 6 8 10` | Qubit counts for density-matrix benchmarks. |
| `--repeats` | `50` | Number of independent repetitions per data point. |
| `--skip` | (none) | Skip one or more tasks: `channels`, `metrics`, `lindblad`. |
| `--out` | `../../data/qutip/qutip_results.csv` | Output CSV path. |
| `--env-out` | `../../data/qutip/qutip_environment.txt` | Output path for version and platform metadata. |

**Benchmarked tasks:**

- **Noise channel application** (`channel_depolarizing`, `channel_amplitude_damping`, `channel_phase_damping`): Apply a single-qubit channel to qubit 0 of an N-qubit uniform density matrix using a direct Kraus sum Σ K·ρ·K†. The superoperator representation (which would require a (4^N)² matrix — approximately 68 GB at N = 8) is not used; the Kraus-sum path keeps memory at O(4^N), matching qcrypto's internal cost structure. Expanded Kraus operators are precomputed outside the timed loop, mirroring `iter_batched` in Criterion.

- **Purity** (`metric_purity`): Compute Tr(ρ²) = 1 − entropy_linear(ρ) on a genuinely mixed density matrix. The input state is prepared by applying `amplitude_damping(0.3)` to every qubit of |+...+⟩, matching the setup in `core_ops.rs` to enable a direct timing comparison.

- **Lindblad evolution** (`lindblad_evolution`): Evolve the system under amplitude damping using QuTiP's `mesolve` ODE integrator (H = 0, collapse operator L = √γ·σ₋ on qubit 0, t ∈ [0, 1]). This is mathematically equivalent to a single discrete Kraus step, but `mesolve` incurs solver setup and ODE step-control overhead that is not present in qcrypto's single matrix multiplication. These timings characterize QuTiP's ODE overhead rather than channel throughput and should not be compared directly with qcrypto's channel application times.

**Channel conventions and equivalence assertions**: QuTiP has no built-in convenience functions for depolarizing, amplitude damping, or phase damping. All channels are constructed from Kraus operators that exactly match qcrypto's internal definitions. Before any timing loop, the script verifies at one qubit that the QuTiP superoperator path (kraus_to_super → operator_to_vector → vector_to_operator) produces the same density matrix as a direct NumPy Kraus sum, which mirrors qcrypto's internal computation. The script aborts if any equivalence check fails with a tolerance of 1e-10.

---

## Statistical Methodology

**Confidence intervals**: Every data point includes a 95% confidence interval.

- Rust (Criterion): Bootstrap-based CI over the sample distribution reported by Criterion.
- Python (Qiskit and QuTiP scripts): Custom NumPy bootstrap over `--repeats` independent timeit measurements, with 1 000 bootstrap resamples, estimating the CI for the median.

**Warm-up**: One untimed execution precedes each measurement in all scripts to eliminate JIT compilation, OS page-fault, and library initialization artifacts.

**Timing method**: Rust uses Criterion's batch measurement with overhead subtraction. Python uses `timeit.timeit(fn, number=1)` repeated `--repeats` times; the median is reported as the primary statistic.

---

## Plot Color Conventions

The figure generation script (`benches/scripts/graphs/plot_graphs.py`) uses two distinct color schemes depending on what is being compared.

**qcrypto vs Qiskit Aer or qcrypto vs QuTiP**: qcrypto is always represented as a single orange series regardless of whether the underlying data comes from `StateVector` or `DensityMatrix` operations.

| Series | Color | Marker |
| :--- | :--- | :--- |
| qcrypto | Orange | Circle (`-o`) |
| Qiskit Aer | Blue | Square (`--s`) |
| QuTiP | Green | Square (`--s`) |

**StateVector vs DensityMatrix (qcrypto-only plots)**: when both representations appear on the same figure, they are distinguished by color.

| Series | Color | Marker |
| :--- | :--- | :--- |
| qcrypto StateVector | Orange | Circle (`-o`) |
| qcrypto DensityMatrix | SaddleBrown | Square (`--s`) |

Shaded areas in all plots represent the 95% confidence interval around each median.
