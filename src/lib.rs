//! # Quantum Cryptography Simulation Library
//!
//! A library for simulating quantum cryptography protocols and quantum information primitives.
//!
//! This crate provides tools for:
//! - Simulating quantum states and gates.
//! - Measuring quantum states.
//! - Simulating quantum channels and noise.
//! - Implementing quantum cryptography protocols.
//! - Reproducible simulations using deterministic thread-local RNGs.

mod core;
pub mod protocols;
pub mod rng;
mod sampler;

pub use crate::core::{
    Gate, Measurement, MeasurementResult, QuantumChannel, QuantumState, errors, utils,
};

pub use crate::core::state;
pub use crate::rng::{
    random_bool, random_f64, random_f64_range, random_usize_range, set_global_seed, shuffle_slice,
};
pub use crate::sampler::Sampler;
