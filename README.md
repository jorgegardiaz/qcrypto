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

Unlike general-purpose quantum simulators that strictly focus on state-vector evolution for logical circuits, `qcrypto` implements an intelligent **Dual-State Architecture**. It dynamically switches between highly-efficient **State Vectors** for pure states and robust **Density Matrices** when open quantum systems, decoherence, or noisy channels are introduced. This design choice enables both high-performance execution of unitary logical circuits and the precise simulation of generalized measurements (POVMs) and hardware vulnerabilities.

The library is implemented in **100% Safe Rust**, eliminating external dependencies.

## Key Features

* **Dual-State Formalism:** Automatic transparent conversion from `StateVector` ($O(2^N)$ memory) to `StateDensityMatrix` ($O(4^N)$ memory) exactly only when a non-unitary noisy channel is applied.
* **Open Quantum Systems:** Implementation of quantum channels (Bit Flip, Phase Damping, Amplitude Damping, Depolarizing) satisfying the Trace-Preserving condition.
* **Generalized Measurements:** Support for Positive Operator-Valued Measures (POVM), essential for protocols like B92 and unambiguous state discrimination.
* **Efficient Operator Expansion:** Native implementation of optimized algorithms avoiding global matrix expansion to perform local tensor updates mathematically.
* **Reproducible Simulations:** A Thread-Local RNG system (`qcrypto::rng`) allows researchers to lock simulations to deterministic entropy sequences to exactly replicate experimental protocol runs.

## Installation

To use `qcrypto` in your Rust project, you can easily add it via Cargo.

Run the following command in your project directory:

```bash
cargo add qcrypto
```
---

## Library Architecture

`qcrypto` is built upon a mathematically rigorous foundation, avoiding common simplifications found in other simulators. The core components are designed to efficiently handle both pure unitary logic and mixed states natively.

### Core Structures

* **`QuantumState`**: An intelligent `enum` that transparently encapsulates either a **`StateVector`** or a **`StateDensityMatrix`**. For efficiency, fresh algorithms execute utilizing pure vectors and intelligently cast themselves into density traces ONLY when entangled with an environment via a quantum noise channel.
* **`QuantumChannel`**: Models physical noise and decoherence (e.g., Bit Flip, Phase Damping, Amplitude Damping) using **Kraus Operators**. It ensures the evolution is Trace-Preserving.
* **`Measurement`**: A generalized measurement framework supporting both standard Projective Measurements and **Positive Operator-Valued Measures (POVM)**. This is crucial for implementing optimal discrimination strategies and ambiguous state detection.
* **`Gate`**: Provides standard unitary operations and allows for the definition of custom single and multi-qubit unitaries.
* **`Sampler`**: Permits to run multiple shots of measurements using a `Measurement` and `QuantumChannel`.

## Implemented Protocols

The library includes reference implementations for standard and novel quantum cryptographic schemes.

### 1. BB84 (Bennett & Brassard, 1984)

The standard protocol for Quantum Key Distribution. The implementation supports:

* Intercept-Resend attacks.
* Real-time QBER (Quantum Bit Error Rate) estimation.
* Sifting and error reconciliation simulation.

### 2. B92 (Bennett, 1992) with Optimal POVM

An implementation of B92 utilizing generalized measurements for **Unambiguous State Discrimination**.

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
use qcrypto::{QuantumState, Gate, Measurement, QuantumChannel, errors::*};

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    // Purity will be 1.0 because it has been projected to a pure state

    Ok(())
}
```

### Running the QIA-QZKP Protocol

```rust
use qcrypto::protocols::qia_qzkp;
use qcrypto::{QuantumChannel, errors::*};

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

### Reproducible Simulations (Deterministic RNG)

For testing and research, it is often critical to perfectly reproduce a specific run of a protocol (yielding identical keys and error distributions). `qcrypto` provides a high-performance **Thread-Local Deterministic RNG** that does not require you to pass RNG instances to every function.

```rust
use qcrypto::protocols::qkd::bbm92;
use qcrypto::{QuantumChannel, rng::set_global_seed};

fn main() {
    let channel = QuantumChannel::depolarizing(0.1)?;
    
    // Lock the RNG for this thread to a specific seed
    set_global_seed(42);
    
    // Every call to bbm92::run or QuantumState::measure on this thread 
    // will now be 100% deterministic and reproducible.
    let result = bbm92::run(1000, &channel, 0.1, 0.2)?;
    
    // Running this program tomorrow will yield the exact same QBER and key.
    println!("Deterministically reproducible QBER: {:.2}%", result.qber);
}
```

### Running Multiple Executions and Saving to CSV

For statistical analysis, it is often necessary to run a protocol multiple times across different parameters (e.g., varying noise levels) and store the results.

```rust
use qcrypto::protocols::qkd::bb84;
use qcrypto::{QuantumChannel, errors::*};
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
    writeln!(file, "Noise,Execution,QBER,EstablishedKeyLength")?;

    // Iterate over different noise configurations
    for &noise in &noise_levels {
        let channel = QuantumChannel::depolarizing(noise)?;

        // Run the protocol multiple times for each configuration
        for execution in 1..=num_executions {
            let result = bb84::run(num_qubits, &channel, eve_ratio, check_ratio)?;
            
            // Write the extracted data to the CSV
            writeln!(
                file, 
                "{:.2},{},{:.2},{}", 
                noise, execution, result.qber, result.established_key.len()
            )?;
        }
    }

    println!("Simulation complete. Data saved to bb84_results.csv");
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
