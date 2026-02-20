<div align="center">

  <h1>qcrypto</h1>

  <p>
    <strong>A Pure Rust Framework for Quantum Cryptography Simulation</strong>
  </p>
  
  [![Pure Rust](https://img.shields.io/badge/Pure-Rust-orange)](https://www.rust-lang.org)
  [![Crates.io](https://img.shields.io/crates/v/qcrypto.svg)](https://crates.io/crates/qcrypto)
  [![Docs](https://docs.rs/qcrypto/badge.svg)](https://docs.rs/qcrypto)
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

<br />

`qcrypto` cross-platform Rust library designed for the design, simulation, and validation of Quantum Cryptography protocols.

Unlike general-purpose quantum simulators that focus on state-vector evolution for logical circuits, `qcrypto` is architected around **Density Matrices ($\rho$)** and **Kraus Operators**. This design choice enables the precise simulation of open quantum systems, decoherence, noisy channels, and generalized measurements (POVMs), which are critical for validating the physical security of cryptographic protocols.

The library is implemented in **100% Safe Rust**, eliminating external dependencies.

## Key Features

* **Density Matrix Formalism:** Native support for mixed states, enabling the simulation of statistical ensembles and entanglement degradation.
* **Open Quantum Systems:** Implementation of quantum channels (Bit Flip, Phase Damping, Amplitude Damping, Depolarizing) satisfying the Trace-Preserving condition ($\sum K_i^\dagger K_i = I$).
* **Generalized Measurements:** Support for Positive Operator-Valued Measures (POVM), essential for protocols like B92 and unambiguous state discrimination.
* **Efficient Operator Expansion:** Native implementation of optimized algorithms to extend single-qubit operator matrices to multi-qubit composite systems.

## Installation

To use `qcrypto` in your Rust project, you can easily add it via Cargo.

Run the following command in your project directory:

```bash
cargo add qcrypto
```
---

## Library Architecture

`qcrypto` is built upon a mathematically rigorous foundation, avoiding common simplifications found in other simulators. The core components are designed to handle open quantum systems and mixed states natively.

### Core Structures

* **`QuantumState`**: Represents the state of the system using **Density Matrices** ($\rho$). Unlike state-vector simulators, this allows for the accurate representation of mixed states, statistical ensembles, and decoherence effects.
* **`QuantumChannel`**: Models physical noise and decoherence (e.g., Bit Flip, Phase Damping, Amplitude Damping) using **Kraus Operators**. It ensures the evolution is Trace-Preserving by verifying $\sum K_i^\dagger K_i = I$.
* **`Measurement`**: A generalized measurement framework supporting both standard Projective Measurements (Von Neumann) and **Positive Operator-Valued Measures (POVM)**. This is crucial for implementing optimal discrimination strategies and ambiguous state detection.
* **`Gate`**: Provides standard unitary operations ($X, Z, H, CNOT, \dots$) and allows for the definition of custom single and multi-qubit unitaries.

## Implemented Protocols

The library includes reference implementations for standard and novel quantum cryptographic schemes.

### 1. BB84 (Bennett & Brassard, 1984)

The standard protocol for Quantum Key Distribution. The implementation supports:

* Intercept-Resend attacks.
* Real-time QBER (Quantum Bit Error Rate) estimation.
* Sifting and error reconciliation simulation.

### 2. B92 (Bennett, 1992) with Optimal POVM

An implementation of B92 utilizing generalized measurements for **Unambiguous State Discrimination (USD)**.

* **Mechanism:** Constructs the optimal POVM such that inconclusive results are explicitly handled.
* **Yield:** Achieves the optimal theoretical sifting rate (approx. 29.3% for standard non-orthogonal states), strictly outperforming standard projective measurements in a noiseless channel.

### 3. QIA-QZKP (Garcia-Diaz et al., 2025)

A reference implementation of the protocol described in *"Conjugate Coding Based Designated Verifier Quantum Zero Knowledge Proof for User Authentication"*.

This protocol establishes a Quantum Zero-Knowledge Proof (QZKP) for identity authentication without revealing the prover's secret key.

* **Security Model:** Relies on the uncertainty principle of conjugate coding (Computational vs. Hadamard bases).
* **Properties:**
* *Completeness:* Honest provers are accepted with probability approaching 1 (adjusted for channel noise models).
* *Soundness:* The probability of a dishonest prover successfully impersonating an identity follows a binomial distribution , decaying exponentially with key length .
* *Zero-Knowledge:* The designated verifier gains no information about the long-term secret  due to the ephemeral masking .

---

## Usage Example

### Simulating a Noisy Channel with Density Matrices

```rust
use qcrypto::{QuantumState, Gate, Measurement, QuantumChannel, errors::StateError};

fn main() -> Result<(), StateError> {
    // 1. Initialize a pure qubit state |0><0|
    let mut rho = QuantumState::new(1);

    // 2. Apply Hadamard Gate -> |+><+|
    rho.apply(&Gate::h(), &[0])?;

    // 3. Evolve through an Amplitude Damping Channel (gamma = 0.3)
    // This transforms the pure state into a mixed state.
    let channel = QuantumChannel::amplitude_damping(0.3)?;
    rho.apply_channel(&channel, &[0])?;
    println!("State Purity (Tr(rho^2)): {:.4}", rho.purity()); 
    // Purity will be < 1.0 due to the non-unitary channel evolution.

    // 4. Measure in the Z basis
    let measurement = Measurement::z_basis();
    let outcome = rho.measure(&measurement, &[0])?;

    println!("Measurement Outcome: {}", outcome.index);
    println!("State Purity (Tr(rho^2)): {:.4}", rho.purity()); 
    // Purity will be 1 because it has been proyected

    Ok(())
}
```

### Running the QIA-QZKP Protocol

```rust
use qcrypto::protocols::qia_qzkp;
use qcrypto::{QuantumChannel, errors::StateError};

fn main() -> Result<(), StateError> {
    let n_qubits = 1024;
    let threshold = 0.85; // Acceptance threshold based on expected QBER
    
    // Simulate a realistic channel with 5% noise
    let noisy_channel = QuantumChannel::bit_flip(0.05)?;

    let result = qia_qzkp::run(n_qubits, &noisy_channel, threshold)?;

    println!("Protocol Accuracy: {:.2}%", result.accuracy * 100.0);
    println!("Authenticated: {}", result.authenticated);
    
    Ok(())
}
```

## References

If you use this software in your research or project, please cite it using the information in [CITATION](CITATION.cff). Additionally, if you use the QIA-QZKP module in your research, please cite the original paper:

> **Garcia-Diaz, J.**, Escanez-Exposito, D., Caballero-Gil, P., & Molina-Gil, J. (2025). *Conjugate Coding Based Designated Verifier Quantum Zero Knowledge Proof for User Authentication*.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/jorgegardiaz/qcrypto).
