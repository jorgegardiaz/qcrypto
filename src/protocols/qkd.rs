//! Quantum Key Distribution (QKD) Protocols.
//!
//! This module contains implementations of standard QKD protocols:
//! - **BB84**: The first quantum key distribution protocol.
//! - **Six-State**: Three bases version of BB84.
//! - **B92**: A simplified version of BB84 using only two non-orthogonal states.
//! - **BBM92**: An entanglement-based version of BB84.
//! - **E91**: An entanglement-based protocol using Bell's inequality.
//! - **SARG04**: A BB84 variant with pair-of-states sifting, more robust against PNS attacks.

pub mod b92;
pub mod bb84;
pub mod bbm92;
pub mod e91;
pub mod sarg04;
pub mod six_state;
