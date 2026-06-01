<div align="center">

  <h1>qcrypto</h1>
  
  <p>
    <strong>A Pure Rust Framework for Quantum Cryptography Simulation</strong>
  </p>

  <img src="./assets/qcrypto_logo.png" alt="qcrypto logo" width="150">
    
  [![Pure Rust](https://img.shields.io/badge/Pure-Rust-orange)](https://www.rust-lang.org)
  [![Crates.io](https://img.shields.io/crates/v/qcrypto.svg)](https://crates.io/crates/qcrypto)
  [![Docs](https://docs.rs/qcrypto/badge.svg)](https://docs.rs/qcrypto)
  [![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
  [![Rust CI](https://github.com/jorgegardiaz/qcrypto/actions/workflows/test.yml/badge.svg)](https://github.com/jorgegardiaz/qcrypto/actions/workflows/test.yml)
  ![Coverage](https://raw.githubusercontent.com/jorgegardiaz/qcrypto/master/.github/badges/coverage.svg)


</div>

<br />

`qcrypto` is a cross-platform Rust library tailored for the design, simulation, and validation of quantum cryptographic protocols.

Unlike general-purpose quantum simulators that strictly focus on state-vector evolution for logical circuits, `qcrypto` implements an intelligent **Dual-State Architecture**. It dynamically switches between highly-efficient **State Vectors** for pure states and robust **Density Matrices** when open quantum systems, decoherence, or noisy channels are introduced. This design choice enables both high-performance execution of unitary logical circuits and the precise simulation of generalized measurements and quantum channels.

The library is implemented in **100% Safe Rust**, minimizing external dependencies and with a pure-Rust dependency tree.

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
  - [Cargo Features](#cargo-features)
- [Library Architecture](#library-architecture)
  - [Core Structures](#core-structures)
  - [Qubit Ordering Convention](#qubit-ordering-convention)
- [Implemented Protocols](#implemented-protocols)
  - [1. BB84](#1-bb84-bennett--brassard-1984)
  - [2. B92 with Optimal POVM](#2-b92-bennett-1992-with-optimal-povm)
  - [3. BBM92](#3-bbm92-bennett-brassard--mermin-1992)
  - [4. E91](#4-e91-ekert-1991)
  - [5. SARG04](#5-sarg04-scarani-acín-ribordy--gisin-2004)
  - [6. Six-State](#6-six-state-pasquinucci--gisin-1999)
  - [7. QIA-QZKP](#7-qia-qzkp-garcia-diaz-et-al-2025)
  - [8. GC01](#8-gc01-gottesman--chuang-2001)
- [Usage Examples](#usage-example)
  - [Simulating a Noisy Channel with Density Matrices](#simulating-a-noisy-channel-with-density-matrices)
  - [Running the QIA-QZKP Protocol](#running-the-qia-qzkp-protocol)
  - [Reproducible Simulations (Deterministic RNG)](#reproducible-simulations-deterministic-rng)
  - [Running Multiple Executions and Saving to CSV](#running-multiple-executions-and-saving-to-csv)
- [CLI Tool: qcryptool](#cli-tool-qcryptool)
- [Benchmarks](#benchmarks)
- [References](#references)
- [License](#license)
- [Contributing](#contributing)

## Key Features

* **Dual-State Formalism:** Automatic transparent conversion from `StateVector` ($O(2^N)$ memory) to `StateDensityMatrix` ($O(4^N)$ memory) exactly only when a noisy channel is applied.
* **Open Quantum Systems:** Implementation of quantum channels (Bit Flip, Phase Damping, Amplitude Damping, Depolarizing) satisfying the Trace-Preserving condition.
* **Generalized Measurements:** Support for Positive Operator-Valued Measures (POVM), essential for protocols like B92 and unambiguous state discrimination.
* **Efficient Operator Expansion:** Native implementation of optimized algorithms avoiding global matrix expansion to perform local tensor updates mathematically.
* **Reproducible Simulations:** A RNG system (`qcrypto::rng`) allows researchers to lock simulations to deterministic entropy sequences to exactly replicate experimental protocol runs.

## Installation

To use `qcrypto` in your Rust project, you can easily add it via Cargo.

Run the following command in your project directory:

```bash
cargo add qcrypto
```

### Cargo features

- `parallel` (default): enables Rayon-based parallelism and the `run_par` variants of all protocols.
- `serde`: enables `serde::Serialize` / `Deserialize` on the main types.

Disable Rayon for WASM or embedded targets:

```toml
[dependencies]
qcrypto = { version = "x.x", default-features = false }
```

---

## Library Architecture

`qcrypto` is built upon a mathematically rigorous foundation, avoiding common simplifications found in other simulators. The core components are designed to efficiently handle both pure unitary logic and mixed states natively.

### Core Structures

* **`QuantumState`**: A dynamic wrapper containing a pointer (`Box<dyn QuantumStateImpl>`) to the `QuantumStateImpl` trait, which is implemented by both **`StateVector`** and **`StateDensityMatrix`**. This trait-based architecture enables transparent dynamic dispatch. For optimal efficiency, simulations initialize using pure state vectors ($O(2^N)$ memory) and intelligently cast themselves into density matrices ($O(4^N)$ memory) ONLY when the state becomes mixed due to interaction with an environment via a quantum noise channel.
* **`QuantumChannel`**: Models physical noise and decoherence (e.g., Bit Flip, Phase Damping, Amplitude Damping) using **Kraus Operators**. It ensures the evolution is Trace-Preserving.
* **`Measurement`**: A generalized measurement framework supporting both standard Projective Measurements and **Positive Operator-Valued Measures (POVM)**. This is crucial for implementing optimal discrimination strategies and ambiguous state detection.
* **`Gate`**: Provides standard unitary operations and allows for the definition of custom single and multi-qubit unitaries.
* **`Sampler`**: Permits to run multiple shots of measurements using a `Measurement` and `QuantumChannel`.

### Qubit Ordering Convention

`qcrypto` uses **big-endian** ordering: qubit 0 is the most significant bit. The state $|q_0\, q_1\, \cdots\, q_{N-1}\rangle$ maps to the amplitude at index $q_0 \cdot 2^{N-1} + q_1 \cdot 2^{N-2} + \cdots + q_{N-1} \cdot 2^0$. In the notation $|01\rangle$, qubit 0 is in state $|0\rangle$ and qubit 1 is in state $|1\rangle$. Measurement labels produced by `Measurement::compose` follow the same order: the leftmost character corresponds to the first element of the `targets` slice. This is the **opposite** of [Qiskit's convention](https://quantum.cloud.ibm.com/docs/en/guides/bit-ordering), which places qubit 0 at the least significant bit.

## Implemented Protocols

The library includes reference implementations for standard and novel quantum cryptographic schemes. All protocols include `run` and  `run_par`, the second one using `rayon` to parallelize sampling instances. For low nomber of qubits (num_qubits < 500) `run` is faster due to the overhead of thread managing on `run_par`.

`qcypto` offers some implementations of these types of cryptographic protocols: **QKD**, **QIA** and **QDS**.

### 1. BB84 (Bennett & Brassard, 1984)

The standard protocol for Quantum Key Distribution. The implementation supports:

* Intercept-Resend attacks.
* Real-time QBER (Quantum Bit Error Rate) estimation.
* Sifting and error reconciliation simulation.

### 2. B92 (Bennett, 1992) with Optimal POVM

An implementation of B92 utilizing generalized measurements for **Unambiguous State Discrimination**.

* **Mechanism:** Constructs the optimal POVM such that inconclusive results are explicitly handled.
* **Yield:** Achieves the optimal theoretical sifting rate (approx. 29.3% for standard non-orthogonal states), strictly outperforming standard projective measurements in a noiseless channel.

### 3. BBM92 (Bennett, Brassard & Mermin, 1992)

An entanglement-based QKD protocol that adapts BB84 to work with a source of entangled pairs instead of prepared single-qubit states.

* **Mechanism:** A source distributes EPR pairs in the Bell state $|\Phi^+\rangle$. Alice and Bob each measure their qubit independently in a randomly chosen basis (Z or X). Sifting keeps only rounds where both chose the same basis, producing perfectly correlated key bits.
* **Security:** Eavesdropping disturbs the entangled state and raises the QBER above the noise floor, detectable through sacrificed check bits.
* **Sifting rate:** ~50% of raw pairs survive sifting, matching BB84's rate.

### 4. E91 (Ekert, 1991)

An entanglement-based QKD protocol whose security is grounded in Bell's inequality rather than information-theoretic arguments alone.

* **Mechanism:** A source distributes singlet pairs $|\Psi^-\rangle$. Alice measures in one of three angles ($0, \pi/8, \pi/4$) and Bob in one of three angles ($\pi/8, \pi/4, 3\pi/8$). Pairs where Alice and Bob shared a physical angle are sifted into the key (anticorrelated outcomes, so Bob flips his bits). The remaining pairs are used to evaluate the **CHSH inequality** ($|S| \leq 2$ classically; $|S| = 2\sqrt{2}$ for a perfect singlet).
* **Security:** Any intercept-and-resend attack collapses the entanglement, reducing the CHSH value toward the classical limit and raising the QBER.
* **Sifting rate:** ~33% of raw pairs contribute to the key.

### 5. SARG04 (Scarani, Acín, Ribordy & Gisin, 2004)

A QKD protocol that reuses BB84's four states but replaces basis announcement with a pair-announcement sifting scheme, providing stronger resistance against photon-number-splitting (PNS) attacks.

* **Mechanism:** Alice prepares one of $\{|0\rangle, |1\rangle, |+\rangle, |-\rangle\}$. Instead of announcing her basis, she announces a pair of non-orthogonal states — one from each basis — that contains her transmitted state. Bob's outcome is conclusive only when it is orthogonal to exactly one of the two announced states; the decoded bit is then determined by the *other* state in the pair.
* **Security advantage:** The PNS attack requires Eve to distinguish between non-orthogonal states, which is harder than the basis-guessing attack against BB84.
* **Sifting rate:** ~25% of qubits yield conclusive decoding in the noiseless case.

### 6. Six-State (Pasquinucci & Gisin, 1999)

An extension of BB84 that uses **three mutually unbiased bases** (Z, X, and Y) instead of two, increasing the information cost of any eavesdropping strategy.

* **Mechanism:** Alice prepares qubits in one of six states: $|0\rangle, |1\rangle$ (Z basis), $|+\rangle, |-\rangle$ (X basis), $|{+i}\rangle, |{-i}\rangle$ (Y basis). Bob measures in a randomly chosen basis. Only rounds where both chose the same basis are kept.
* **Security advantage:** The optimal intercept-and-resend QBER is $1/3$ (vs. $1/4$ for BB84), making eavesdropping easier to detect.
* **Sifting rate:** ~33% of qubits survive sifting.

### 7. QIA-QZKP (Garcia-Diaz et al., 2025)

A reference implementation of the protocol described in *"Conjugate Coding Based Designated Verifier Quantum Zero Knowledge Proof for User Authentication"*.

This protocol establishes a Quantum Zero-Knowledge Proof (QZKP) for identity authentication without revealing the prover's secret key.

* **Security Model:** Relies on the uncertainty principle of conjugate coding (Computational vs. Hadamard bases).
* **Properties:**
* *Completeness:* Honest provers are accepted with probability approaching 1 (adjusted for channel noise models).
* *Soundness:* The probability of a dishonest prover successfully impersonating an identity follows a binomial distribution , decaying exponentially with key length .
* *Zero-Knowledge:* The designated verifier gains no information about the long-term secret  due to the ephemeral masking .

### 8. GC01 (Gottesman & Chuang, 2001)
This protocol establishes a Quantum Digital Signature (QDS) scheme that allows a sender to sign classical messages such that any recipient can verify authenticity and transferability is guaranteed.

* **Security Model**: Relies on the quantum impossibility of distinguishing non-orthogonal quantum states (analogous to one-way functions in the classical setting). Security is information-theoretic, not computational.
* **Properties**:
* Unforgeability: The probability of a dishonest party forging a valid signature is exponentially suppressed in the number of qubits $n$ used per signature element, following from the indistinguishability of the quantum public keys.
* Transferability: A signature accepted by one honest recipient is guaranteed to be accepted by any other honest recipient, preventing repudiation across parties.
* Non-repudiation: The signer cannot later deny having signed a message, as the distributed quantum public keys bind the signature to a unique originator.



---

## Usage Example

### Simulating a Noisy Channel with Density Matrices

```rust
use qcrypto::{Gate, Measurement, QuantumChannel, QuantumState};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize a pure qubit state |0><0|
    let mut rho = QuantumState::new(1);

    // 2. Apply Hadamard Gate -> |+><+|
    rho.apply(&Gate::h(), &[0])?;

    // 3. Evolve through an Amplitude Damping Channel (gamma = 0.3)
    // This transforms the pure state into a mixed state.
    let channel = QuantumChannel::amplitude_damping(0.3);
    rho.apply_channel(&channel, &[0])?;
    println!("State Purity (Tr(rho^2)): {:.4}", rho.purity());
    // Purity will be < 1.0 due to the non-unitary channel evolution.

    // 4. Measure in the Z basis
    let measurement = Measurement::z_basis();
    let outcome = rho.measure(&measurement, &[0])?;

    println!("Measurement Outcome: {}", outcome.index);
    println!("State Purity (Tr(rho^2)): {:.4}", rho.purity());
    // Purity will be 1.0 because it has been projected to a pure state

    Ok(())
}
```

### Running the QIA-QZKP Protocol

```rust
use qcrypto::QuantumChannel;
use qcrypto::protocols::qia_qzkp;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_qubits = 1024;
    let threshold = 0.8; // Acceptance threshold based on expected QBER

    // Simulate a realistic channel with 5% noise
    let noisy_channel = QuantumChannel::bit_flip(0.1);

    let result = qia_qzkp::run(n_qubits, &noisy_channel, threshold)?;

    println!("Protocol Accuracy: {:.2}%", result.accuracy * 100.0);
    println!("Authenticated: {}", result.authenticated);
    Ok(())
}
```

### Reproducible Simulations (Deterministic RNG)

For testing and research, it is often critical to perfectly reproduce a specific run of a protocol (yielding identical keys and error distributions). `qcrypto` provides a high-performance **Thread-Local Deterministic RNG** that does not require you to pass RNG instances to every function.

```rust
use qcrypto::protocols::qkd::bbm92;
use qcrypto::{QuantumChannel, rng::set_global_seed};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let channel_alice = QuantumChannel::depolarizing(0.1);
    let channel_bob = QuantumChannel::depolarizing(0.05);

    // Lock the RNG to a specific seed
    set_global_seed(42);

    // Every call to bbm92::run or QuantumState::measure
    // will now be 100% deterministic and reproducible.
    let result = bbm92::run_par(1000, &channel_alice, &channel_bob, 1.0, 0.2)?;

    // Running this program tomorrow will yield the exact same QBER and key.
    println!("Deterministically reproducible QBER: {:.2}%", result.qber);
    Ok(())
}
```

### Running Multiple Executions and Saving to CSV

For statistical analysis, it is often necessary to run a protocol multiple times across different parameters (e.g., varying noise levels) and store the results.

```rust
use qcrypto::{QuantumChannel, protocols::bb84};
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let num_qubits = 1000;
    let eve_ratio = 0.0;
    let check_ratio = 0.2;
    let num_executions = 50;

    let noise_levels = [0.0, 0.05, 0.10, 0.15];

    // Create a CSV file for output
    let mut file = File::create("bb84_results.csv")?;
    writeln!(file, "Noise,Execution,QBER,KeyLength")?;

    // Iterate over different noise configurations
    for &noise in &noise_levels {
        let channel = QuantumChannel::depolarizing(noise);

        // Run the protocol multiple times for each configuration
        for execution in 1..=num_executions {
            let result = bb84::run_par(num_qubits, &channel, eve_ratio, check_ratio)?;

            // Write the extracted data to the CSV
            writeln!(
                file,
                "{:.2},{},{:.2},{}",
                noise, execution, result.qber, result.alice_key.len(),
            )?;
        }
    }

    println!("Simulation complete. Data saved to bb84_results.csv");
    Ok(())
}
```

## CLI Tool: qcryptool

<div align="center">
  <img src="./assets/qcryptool_logo.png" alt="qcryptool logo" width="130">

  [![Pure Rust](https://img.shields.io/badge/Pure-Rust-orange)](https://www.rust-lang.org)
  [![Crates.io](https://img.shields.io/crates/v/qcryptool.svg)](https://crates.io/crates/qcryptool)
</div>

[`qcryptool`](https://github.com/jorgegardiaz/qcryptool) is a command-line simulator for quantum cryptography protocols built directly on top of `qcrypto`. It is designed for researchers, educators, and engineers who want to run and analyse quantum protocol simulations without writing any Rust code.

**What you can do with it:**

* Run any of the eight protocols implemented in `qcrypto` (BB84, B92, BBM92, E91, Six-State, SARG04, QIA-QZKP, GC01) from the terminal with a single command.
* Execute **multi-shot** experiments and collect aggregate statistics (mean QBER, key length, Eve detection rate, …) automatically.
* Configure **noise models** per channel (bit-flip, depolarizing, amplitude damping, …) via flags or a JSON channel-mix file that samples parameters stochastically across shots.
* Export results to **CSV, JSON, or plain text** for downstream analysis in Python, R, or any data tool.
* Guarantee **reproducibility** with a `--seed` flag: the same seed always produces identical keys, QBER values, and measurement outcomes.
* Drive full experiments from a single **JSON config file** (`--experiment-config`) without any CLI flags.

```bash
cargo install qcryptool

# Single shot — noiseless BB84
qcryptool bb84 -n 1000

# 100 shots with depolarizing noise, saved to CSV
qcryptool bb84 -n 1024 -s 100 --channel1 depolarizing --p1 0.03 -o results.csv

# Entanglement-based E91 with asymmetric channels
qcryptool e91 -n 1000 --channel1 depolarizing --p1 0.01 --channel2 depolarizing --p2 0.04

# Fully reproducible run
qcryptool bb84 -n 1024 -s 50 --seed 42 -o run_a.csv
```

See the [qcryptool repository](https://github.com/jorgegardiaz/qcryptool) for the full documentation.

---

## Benchmarks

`qcrypto` ships a full benchmark suite under `benches/` that covers gate throughput, noise channel application cost, state-vector vs. density-matrix scaling, multi-shot sampling, and end-to-end execution of all protocol families (QKD, QIA, QDS). Two Python comparison scripts measure equivalent workloads against [Qiskit Aer](https://github.com/Qiskit/qiskit-aer) and [QuTiP](https://qutip.org), and a shell script orchestrates the complete pipeline.

To run the Rust benchmarks:

```bash
cargo bench
```

To run the full pipeline including comparisons and figure generation:

```bash
./benches/scripts/reproduce.sh
```

See [`benches/README.md`](benches/README.md) for a detailed description of every benchmark, the comparison methodology, and CLI options for each script.

---

## References

If you use this software in your research or project, please cite it using the information in [CITATION](CITATION.cff). Additionally, if you use the QIA-QZKP protocol in your research, please cite the original paper:

> Garcia-Diaz, J., Escanez-Exposito, D., Caballero-Gil, P. et al. Conjugate coding based designated verifier quantum zero knowledge proof for user authentication. Cryptogr. Commun. (2026). https://doi.org/10.1007/s12095-026-00878-y

## License

This project is dual-licensed under either:

- [MIT License](LICENSE)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/jorgegardiaz/qcrypto).
