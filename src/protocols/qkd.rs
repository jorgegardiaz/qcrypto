//! Quantum Key Distribution (QKD) Protocols.
//!
//! This module contains implementations of standard QKD protocols:
//! - **BB84**: The first quantum key distribution protocol.
//! - **B92**: A simplified version of BB84 using only two non-orthogonal states.
//! - **BBM92**: An entanglement-based version of BB84.

pub mod b92;
pub mod bb84;
pub mod bbm92;
