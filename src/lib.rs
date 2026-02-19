//! # Quantum Cryptography Simulation Library
//!
//! A library for simulating quantum cryptography protocols and quantum information primitives.
//!
//! This crate provides tools for:
//! - Simulating quantum states and gates.
//! - Measuring quantum states.
//! - Simulating quantum channels and noise.
//! - Implementing quantum cryptography protocols.

mod core;
pub mod protocols;
mod sampler;

pub use crate::core::{
    Gate, Measurement, MeasurementResult, QuantumChannel, QuantumState, errors, utils,
};
pub use crate::sampler::Sampler;
