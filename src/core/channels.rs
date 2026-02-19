use crate::core::errors::ChannelError;
use crate::core::utils;
use ndarray::{Array2, array};
use num_complex::Complex64;

/// Represents a quantum channel.
///
/// A quantum channel is a completely positive trace-preserving (CPTP) map,
/// represented by a set of Kraus operators $\{K_i\}$ satisfying $\sum K_i^\dagger K_i = I$.
#[derive(Clone, Debug)]
pub struct QuantumChannel {
    /// The Kraus operators defining the channel.
    pub kraus_ops: Vec<Array2<Complex64>>,
    /// The number of qubits the channel acts on.
    pub num_qubits: usize,
}

impl QuantumChannel {
    /// Creates a new `QuantumChannel` from a set of Kraus operators.
    ///
    /// # Arguments
    ///
    /// * `kraus_ops` - A vector of `Array2<Complex64>` representing the Kraus operators.
    ///
    /// # Errors
    ///
    /// Returns `ChannelError` if:
    /// - No operators are provided.
    /// - The operators are not square or have invalid dimensions (not power of 2).
    /// - The operators do not all have the same size.
    /// - The operators do not sum to Identity (trace-preserving condition failed).
    pub fn new(kraus_ops: Vec<Array2<Complex64>>) -> Result<Self, ChannelError> {
        if kraus_ops.is_empty() {
            return Err(ChannelError::Empty);
        }

        let (rows, cols) = kraus_ops[0].dim();

        if rows != cols || !rows.is_power_of_two() {
            return Err(ChannelError::InvalidDimensions);
        }

        // log_2
        let num_qubits = rows.trailing_zeros() as usize;

        for op in &kraus_ops {
            if op.dim() != (rows, cols) {
                return Err(ChannelError::OperatorSizeMismatch);
            }
        }

        if !utils::check_completeness(&kraus_ops, rows) {
            return Err(ChannelError::NotComplete);
        }

        Ok(Self {
            kraus_ops,
            num_qubits,
        })
    }

    /// Composes the current `QuantumChannel` with another one.
    ///
    /// This creates a new channel that represents the sequential application of this channel followed by the `other` channel.
    /// The resulting Kraus operators are the pairwise products of the operators from both channels.
    ///
    /// # Arguments
    ///
    /// * `other` - The channel to apply after this one.
    ///
    /// # Errors
    ///
    /// Returns `ChannelError` if the channels act on different numbers of qubits.
    pub fn compose(&self, other: &QuantumChannel) -> Result<QuantumChannel, ChannelError> {
        if self.num_qubits != other.num_qubits {
            return Err(ChannelError::OperatorSizeMismatch);
        }

        let new_ops: Vec<_> = other
            .kraus_ops
            .iter()
            .flat_map(|op_b| self.kraus_ops.iter().map(move |op_a| op_b.dot(op_a)))
            .collect();

        // Returns a diferent QuantumChannel
        Ok(QuantumChannel {
            kraus_ops: new_ops,
            num_qubits: self.num_qubits,
        })
    }

    /// Mixes the current `QuantumChannel` with another one with probability `p`.
    ///
    /// This creates a convex combination of the two channels: $\mathcal{E}_{new} = (1-p)\mathcal{E}_{self} + p\mathcal{E}_{other}$.
    ///
    /// # Arguments
    ///
    /// * `other` - The channel to mix with.
    /// * `p` - The mixing probability (weight of the `other` channel).
    ///
    /// # Errors
    ///
    /// Returns `ChannelError` if:
    /// - The channels act on different numbers of qubits.
    /// - `p` is not between 0.0 and 1.0.
    pub fn mix(&self, other: &QuantumChannel, p: f64) -> Result<QuantumChannel, ChannelError> {
        if self.num_qubits != other.num_qubits {
            return Err(ChannelError::OperatorSizeMismatch);
        }

        validate_prob(p)?;

        let scale_self = Complex64::new((1.0 - p).sqrt(), 0.0);
        let scale_other = Complex64::new(p.sqrt(), 0.0);

        let mut new_kraus_ops = Vec::with_capacity(self.kraus_ops.len() + other.kraus_ops.len());

        for op in &self.kraus_ops {
            new_kraus_ops.push(op * scale_self);
        }

        for op in &other.kraus_ops {
            new_kraus_ops.push(op * scale_other);
        }

        // Returns a different QuantumChannel
        Ok(QuantumChannel {
            kraus_ops: new_kraus_ops,
            num_qubits: self.num_qubits,
        })
    }

    /// Expands the channel's operators to act on a larger system.
    ///
    /// # Arguments
    ///
    /// * `num_total_qubits` - The size of the full system.
    /// * `targets` - The indices of the qubits this channel acts on.
    pub fn get_expanded_operators(
        &self,
        num_total_qubits: usize,
        targets: &[usize],
    ) -> Result<Vec<Array2<Complex64>>, ChannelError> {
        if targets.len() != self.num_qubits {
            return Err(ChannelError::InvalidDimensions);
        }

        let mut expanded_ops = Vec::with_capacity(self.kraus_ops.len());

        for op in &self.kraus_ops {
            let full_op = utils::expand_operator(num_total_qubits, op, targets, &[]);
            expanded_ops.push(full_op);
        }

        Ok(expanded_ops)
    }

    /// Creates a Bit Flip channel.
    ///
    /// With probability `p`, applies X; otherwise Identity.
    pub fn bit_flip(p: f64) -> Result<QuantumChannel, ChannelError> {
        validate_prob(p)?;

        let p_stay = (1.0 - p).sqrt();
        let p_flip = p.sqrt();

        let k0 = array![
            [Complex64::new(p_stay, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(p_stay, 0.0)]
        ];

        let k1 = array![
            [Complex64::new(0.0, 0.0), Complex64::new(p_flip, 0.0)],
            [Complex64::new(p_flip, 0.0), Complex64::new(0.0, 0.0)]
        ];

        QuantumChannel::new(vec![k0, k1])
    }

    /// Creates a Phase Flip channel.
    ///
    /// With probability `p`, applies Z; otherwise Identity.
    pub fn phase_flip(p: f64) -> Result<QuantumChannel, ChannelError> {
        validate_prob(p)?;

        let p_stay = (1.0 - p).sqrt();
        let p_flip = p.sqrt();

        let k0 = array![
            [Complex64::new(p_stay, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(p_stay, 0.0)]
        ];

        let k1 = array![
            [Complex64::new(p_flip, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(-p_flip, 0.0)]
        ];

        QuantumChannel::new(vec![k0, k1])
    }

    /// Creates a Bit-Phase Flip channel.
    ///
    /// With probability `p`, applies Y; otherwise Identity.
    pub fn bit_phase_flip(p: f64) -> Result<QuantumChannel, ChannelError> {
        validate_prob(p)?;

        let p_stay = (1.0 - p).sqrt();
        let p_flip = p.sqrt();

        let k0 = array![
            [Complex64::new(p_stay, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(p_stay, 0.0)]
        ];

        let k1 = array![
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, -p_flip)],
            [Complex64::new(0.0, p_flip), Complex64::new(0.0, 0.0)]
        ];

        QuantumChannel::new(vec![k0, k1])
    }

    /// Creates a Depolarizing channel.
    ///
    /// The state is replaced by the maximally mixed state $I/2$ with probability $p$,
    /// and left unchanged with probability $1-p$.
    ///
    /// Note: This implementation treats `p` as the probability of error.
    /// The channel is $\rho \to (1-p)\rho + p \frac{I}{d}$.
    /// For a single qubit, this is equivalent to applying I, X, Y, Z with appropriate weights.
    pub fn depolarizing(p: f64) -> Result<QuantumChannel, ChannelError> {
        validate_prob(p)?;

        // p is the total error probability

        let weight_i = (1.0 - 0.75 * p).sqrt();
        let weight_xyz = (p / 4.0).sqrt();

        let k0 = array![
            // ~ I
            [Complex64::new(weight_i, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(weight_i, 0.0)]
        ];

        let k1 = array![
            // ~ X
            [Complex64::new(0.0, 0.0), Complex64::new(weight_xyz, 0.0)],
            [Complex64::new(weight_xyz, 0.0), Complex64::new(0.0, 0.0)]
        ];

        let k2 = array![
            // ~ Y
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, -weight_xyz)],
            [Complex64::new(0.0, weight_xyz), Complex64::new(0.0, 0.0)]
        ];

        let k3 = array![
            // ~ Z
            [Complex64::new(weight_xyz, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(-weight_xyz, 0.0)]
        ];

        QuantumChannel::new(vec![k0, k1, k2, k3])
    }

    /// Creates an Amplitude Damping channel (T1 relaxation).
    ///
    /// Models energy loss from a quantum system ($|1\rangle \to |0\rangle$).
    pub fn amplitude_damping(gamma: f64) -> Result<QuantumChannel, ChannelError> {
        validate_prob(gamma)?;

        let g_sqrt = gamma.sqrt();
        let one_minus_g_sqrt = (1.0 - gamma).sqrt();

        let k0 = array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(one_minus_g_sqrt, 0.0)
            ]
        ];

        let k1 = array![
            [Complex64::new(0.0, 0.0), Complex64::new(g_sqrt, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)]
        ];

        QuantumChannel::new(vec![k0, k1])
    }

    /// Creates a Phase Damping channel (T2 relaxation).
    ///
    /// Models loss of quantum information (coherence) without loss of energy.
    pub fn phase_damping(lambda: f64) -> Result<QuantumChannel, ChannelError> {
        validate_prob(lambda)?;

        let sqrt_one_minus_lambda = (1.0 - lambda).sqrt();
        let sqrt_lambda = lambda.sqrt();

        let k0 = array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(sqrt_one_minus_lambda, 0.0)
            ]
        ];

        let k1 = array![
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(sqrt_lambda, 0.0)]
        ];

        QuantumChannel::new(vec![k0, k1])
    }

    /// Creates a combined Amplitude and Phase Damping channel.
    ///
    /// Models simultaneous T1 and T2 relaxation processes.
    pub fn combined_amplitude_phase_damping(
        gamma: f64,
        lambda: f64,
    ) -> Result<QuantumChannel, ChannelError> {
        validate_prob(gamma)?;
        validate_prob(lambda)?;

        let amp_channel = Self::amplitude_damping(gamma)?;

        let phase_channel = Self::phase_damping(lambda)?;

        let mut combined_ops = Vec::with_capacity(4);

        for p_op in &phase_channel.kraus_ops {
            for a_op in &amp_channel.kraus_ops {
                let combined = p_op.dot(a_op);
                combined_ops.push(combined);
            }
        }

        QuantumChannel::new(combined_ops)
    }
}

/// Validate probability parameter
fn validate_prob(p: f64) -> Result<(), ChannelError> {
    if !(0.0..=1.0).contains(&p) {
        return Err(ChannelError::InvalidProbability(p));
    }
    Ok(())
}
