Project Specification: qcrypto Library

1. Project Overview
   Name: qcrypto Language: Rust (2021/2024 Edition) Description:

qcrypto is a novel, open-source software library tailored for the scientific exploration, implementation, and rigorous testing of Quantum Cryptographic (QC) primitives. The library leverages Rust for its memory safety and high performance, which are critical for simulating computationally intensive quantum protocols.

Primary Goals:

Scientific Accuracy: Facilitate advanced research into quantum-secure communication.

Real-world Simulation: accurately model real-world performance by simulating quantum channel conditions, specifically noise and decoherence.

Extensibility: Allow researchers to easily integrate new protocols and modify existing parameters.

Scope of Protocols: The library must support:

QKD: BB84, Six-State, E91, B92, BBM92, SARG04.

Advanced Primitives: Quantum Zero-Knowledge Proofs , Quantum Bit Commitment , Quantum Oblivious Transfer , Quantum Coin Flipping, and Digital Signatures.

2. Technical Architecture
   To handle the requirement for simulating noise and decoherence, the architecture decouples the physical simulation (algebra) from the protocol logic (algorithms).

Layer 1: The Physics Core (qcrypto::core)

Math Backend: Uses ndarray and num-complex for linear algebra.

State Representation: MUST support Density Matrices (ρ). While pure states (∣ψ⟩) are simpler, they cannot represent the mixed states resulting from environmental noise.

Layer 2: The Infrastructure (qcrypto::channels)

Defines the environment.

Implements noise models (e.g., Depolarizing, Amplitude Damping) that act upon Density Matrices.

Layer 3: The Application (qcrypto::protocols)

Implements the logic for Alice and Bob.

Abstracts the specific QKD or cryptographic flow.

3. Directory Structure
   ```plaintext
   qcrypto/
   ├── Cargo.toml # Dependencies
   ├── README.md
   └── src/
   ├── lib.rs # Crate root and exports
   ├── core/ # Physics engine
   │ ├── mod.rs
   │ ├── state.rs # DensityMatrix struct
   │ ├── gates.rs # Quantum gates (Pauli-X, H, etc.)
   │ └── utils.rs # Math helpers (tensor products)
   ├── channels/ # Noise simulation
   │ ├── mod.rs
   │ ├── channel_trait.rs # Trait definition
   │ └── noise_models.rs # Specific noise implementations
   ├── measurement/
   │ ├── mod.rs
   │ └── basis.rs # Basis definitions (Rectilinear, Diagonal)
   └── protocols/ # Cryptographic primitives
   ├── mod.rs
   ├── qkd/ # BB84, E91 implementation
   ├── ot.rs # Oblivious Transfer
   └── commitment.rs # Bit Commitment
   ```
