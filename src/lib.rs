//! # Quantum Cryptography Simulation Library
//!
//! A library for simulating quantum cryptography protocols and quantum information primitives.
//!
//! This crate provides tools for:
//! - Simulating quantum states and gates.
//! - Measuring quantum states.
//! - Simulating quantum channels and noise.
//! - Implementing quantum cryptography protocols.
//! - Reproducible simulations using deterministic RNGs using seed.
//!
//! ## Cargo features
//!
//! - `parallel` (default): enables Rayon-based parallelism and the `run_par` variants of all protocols.
//! - `serde`: enables `serde::Serialize` / `Deserialize` on the main types.
//!
//! # Examples
//!
//! ```rust
//! // See examples/ directory for more usage examples.
//! ```

mod core;
pub mod protocols;
pub mod rng;
mod sampler;

pub use crate::core::{
    Gate, Measurement, MeasurementResult, QuantumChannel, QuantumState, errors, utils,
};

pub use crate::core::state;

pub use crate::sampler::Sampler;

#[doc(hidden)]
pub use crate::rng::{
    LocalRng, random_bool, random_f64, random_f64_range, random_usize_range, set_global_seed,
    shuffle_slice,
};
