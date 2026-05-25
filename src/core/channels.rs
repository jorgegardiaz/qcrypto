use crate::core::errors::ChannelError;
use crate::core::utils;
use ndarray::{Array2, array};
use num_complex::Complex64;

/// Represents a quantum channel.
///
/// A quantum channel is a completely positive trace-preserving (CPTP) map,
/// represented by a set of Kraus operators $\{K_i\}$ satisfying $\sum K_i^\dagger K_i = I$.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// # Returns
    ///
    /// A `Result` containing the constructed `QuantumChannel`.
    ///
    /// # Errors
    ///
    /// Returns `ChannelError` if:
    /// - No operators are provided.
    /// - The operators are not square or have invalid dimensions (not power of 2).
    /// - The operators do not all have the same size.
    /// - The operators do not sum to Identity (trace-preserving condition failed).
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::QuantumChannel;
    /// use ndarray::Array2;
    /// use num_complex::Complex64;
    ///
    /// // Identity channel (single Kraus operator = I)
    /// let eye: Array2<Complex64> = Array2::eye(2);
    /// let channel = QuantumChannel::new(vec![eye]).unwrap();
    /// assert_eq!(channel.num_qubits, 1);
    /// ```
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

        if !utils::check_kraus_completeness(&kraus_ops, rows) {
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
    /// # Returns
    ///
    /// A `Result` containing the composed `QuantumChannel`.
    ///
    /// # Errors
    ///
    /// Returns `ChannelError` if the channels act on different numbers of qubits.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::QuantumChannel;
    ///
    /// let bf = QuantumChannel::bit_flip(0.1);
    /// let composed = bf.compose(&bf).unwrap();
    /// assert_eq!(composed.num_qubits, 1);
    /// ```
    pub fn compose(&self, other: &QuantumChannel) -> Result<QuantumChannel, ChannelError> {
        if self.num_qubits != other.num_qubits {
            return Err(ChannelError::OperatorSizeMismatch);
        }

        let new_ops: Vec<_> = other
            .kraus_ops
            .iter()
            .flat_map(|op_b| self.kraus_ops.iter().map(move |op_a| op_b.dot(op_a)))
            .collect();

        QuantumChannel::new(new_ops)
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
    /// # Returns
    ///
    /// A `Result` containing the mixed `QuantumChannel`.
    ///
    /// # Errors
    ///
    /// Returns `ChannelError` if:
    /// - The channels act on different numbers of qubits.
    /// - `p` is not between 0.0 and 1.0.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::QuantumChannel;
    ///
    /// let bf = QuantumChannel::bit_flip(0.5);
    /// let pf = QuantumChannel::phase_flip(0.5);
    /// let mixed = bf.mix(&pf, 0.3).unwrap();
    /// assert_eq!(mixed.num_qubits, 1);
    /// ```
    pub fn mix(&self, other: &QuantumChannel, p: f64) -> Result<QuantumChannel, ChannelError> {
        if self.num_qubits != other.num_qubits {
            return Err(ChannelError::OperatorSizeMismatch);
        }

        validate_prob(p)?;

        let scale_self = Complex64::new((1.0 - p).sqrt(), 0.0);
        let scale_other = Complex64::new(p.sqrt(), 0.0);

        let new_kraus_ops: Vec<_> = self
            .kraus_ops
            .iter()
            .map(|op| op * scale_self)
            .chain(other.kraus_ops.iter().map(|op| op * scale_other))
            .collect();

        QuantumChannel::new(new_kraus_ops)
    }

    /// Expands the channel's operators to act on a larger system.
    ///
    /// # Arguments
    ///
    /// * `num_total_qubits` - The size of the full system.
    /// * `targets` - The indices of the qubits this channel acts on.
    ///
    /// # Returns
    ///
    /// A `Result` containing the expanded Kraus operators.
    ///
    /// # Errors
    ///
    /// Returns `ChannelError` if the number of targets does not match `num_qubits`.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::QuantumChannel;
    ///
    /// let channel = QuantumChannel::bit_flip(0.5);
    /// let expanded = channel.get_expanded_operators(2, &[0]).unwrap();
    /// assert_eq!(expanded[0].dim(), (4, 4));
    /// ```
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
            let full_op = utils::expand_operator(op, num_total_qubits, targets, &[]);
            expanded_ops.push(full_op);
        }

        Ok(expanded_ops)
    }

    /// Creates a Identity channel.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::QuantumChannel;
    ///
    /// let channel = QuantumChannel::identity();
    /// assert_eq!(channel.num_qubits, 1);
    /// ```
    pub fn identity() -> QuantumChannel {
        let k0 = Array2::<Complex64>::eye(2);

        QuantumChannel::new(vec![k0]).expect("Error in identity channel")
    }

    /// Creates a Bit Flip channel.
    ///
    /// With probability `p`, applies X; otherwise Identity.
    /// Panics if `p` is not between 0.0 and 1.0.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::QuantumChannel;
    ///
    /// let channel = QuantumChannel::bit_flip(0.5);
    /// assert_eq!(channel.num_qubits, 1);
    /// ```
    pub fn bit_flip(p: f64) -> QuantumChannel {
        validate_prob(p).expect("Invalid probability for bit_flip channel");

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

        QuantumChannel::new(vec![k0, k1]).expect("Error in bit_flip channel")
    }

    /// Creates a Phase Flip channel.
    ///
    /// With probability `p`, applies Z; otherwise Identity.
    /// Panics if `p` is not between 0.0 and 1.0.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::QuantumChannel;
    ///
    /// let channel = QuantumChannel::phase_flip(0.3);
    /// assert_eq!(channel.num_qubits, 1);
    /// ```
    pub fn phase_flip(p: f64) -> QuantumChannel {
        validate_prob(p).expect("Invalid probability for phase_flip channel");

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

        QuantumChannel::new(vec![k0, k1]).expect("Error in phase_flip channel")
    }

    /// Creates a Bit-Phase Flip channel.
    ///
    /// With probability `p`, applies Y; otherwise Identity.
    /// Panics if `p` is not between 0.0 and 1.0.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::QuantumChannel;
    ///
    /// let channel = QuantumChannel::bit_phase_flip(0.2);
    /// assert_eq!(channel.num_qubits, 1);
    /// ```
    pub fn bit_phase_flip(p: f64) -> QuantumChannel {
        validate_prob(p).expect("Invalid probability for bit_phase_flip channel");

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

        QuantumChannel::new(vec![k0, k1]).expect("Error in bit_phase_flip channel")
    }

    /// Creates a Depolarizing channel.
    ///
    /// The state is replaced by the maximally mixed state $I/2$ with probability $p$,
    /// and left unchanged with probability $1-p$.
    /// Panics if `p` is not between 0.0 and 1.0.
    ///
    /// Note: This implementation treats `p` as the probability of error.
    /// The channel is $\rho \to (1-p)\rho + p \frac{I}{d}$.
    /// For a single qubit, this is equivalent to applying I, X, Y, Z with appropriate weights.
    pub fn depolarizing(p: f64) -> QuantumChannel {
        validate_prob(p).expect("Invalid probability for depolarizing channel");

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

        QuantumChannel::new(vec![k0, k1, k2, k3]).expect("Error in depolarizing channel")
    }

    /// Creates an Amplitude Damping channel (T1 relaxation).
    ///
    /// Models energy loss from a quantum system ($|1\rangle \to |0\rangle$).
    /// Panics if `gamma` is not between 0.0 and 1.0.
    pub fn amplitude_damping(gamma: f64) -> QuantumChannel {
        validate_prob(gamma).expect("Invalid probability for amplitude_damping channel");

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

        QuantumChannel::new(vec![k0, k1]).expect("Error in amplitude_damping channel")
    }

    /// Creates a Phase Damping channel (T2 relaxation).
    ///
    /// Models loss of quantum information (coherence) without loss of energy.
    /// Panics if `lambda` is not between 0.0 and 1.0.
    pub fn phase_damping(lambda: f64) -> QuantumChannel {
        validate_prob(lambda).expect("Invalid probability for phase_damping channel");

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

        QuantumChannel::new(vec![k0, k1]).expect("Error in phase_damping channel")
    }

    /// Creates a combined Amplitude and Phase Damping channel.
    ///
    /// Models simultaneous T1 and T2 relaxation processes.
    /// Panics if `gamma` or `lambda` are not between 0.0 and 1.0.
    pub fn combined_amplitude_phase_damping(gamma: f64, lambda: f64) -> QuantumChannel {
        validate_prob(gamma).expect("Invalid gamma probability");
        validate_prob(lambda).expect("Invalid lambda probability");

        let amp_channel = Self::amplitude_damping(gamma);
        let phase_channel = Self::phase_damping(lambda);

        let mut combined_ops = Vec::with_capacity(4);

        for p_op in &phase_channel.kraus_ops {
            for a_op in &amp_channel.kraus_ops {
                let combined = p_op.dot(a_op);
                combined_ops.push(combined);
            }
        }

        QuantumChannel::new(combined_ops).expect("Error in combined_amplitude_phase_damping")
    }
}

/// Validate probability parameter
fn validate_prob(p: f64) -> Result<(), ChannelError> {
    if !(0.0..=1.0).contains(&p) {
        return Err(ChannelError::InvalidProbability(p));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // --- QuantumChannel::new boundary tests ---

    #[test]
    fn test_new_empty() {
        let result = QuantumChannel::new(vec![]);
        assert!(matches!(result, Err(ChannelError::Empty)));
    }

    #[test]
    fn test_new_non_square() {
        let matrix = Array2::from_shape_vec((2, 3), vec![Complex64::new(1.0, 0.0); 6]).unwrap();
        let result = QuantumChannel::new(vec![matrix]);
        assert!(matches!(result, Err(ChannelError::InvalidDimensions)));
    }

    #[test]
    fn test_new_non_power_of_two() {
        let matrix: Array2<Complex64> = Array2::eye(3);
        let result = QuantumChannel::new(vec![matrix]);
        assert!(matches!(result, Err(ChannelError::InvalidDimensions)));
    }

    #[test]
    fn test_new_size_mismatch() {
        let k0: Array2<Complex64> = Array2::eye(2);
        let k1: Array2<Complex64> = Array2::eye(4);
        let result = QuantumChannel::new(vec![k0, k1]);
        assert!(matches!(result, Err(ChannelError::OperatorSizeMismatch)));
    }

    #[test]
    fn test_new_not_complete() {
        // A single zero matrix does not satisfy sum(K†K) = I
        let zero = Array2::<Complex64>::zeros((2, 2));
        let result = QuantumChannel::new(vec![zero]);
        assert!(matches!(result, Err(ChannelError::NotComplete)));
    }

    // --- compose/mix boundary tests ---

    #[test]
    fn test_compose_size_mismatch() {
        let c1 = QuantumChannel::bit_flip(0.1);
        // Create a 2-qubit identity channel
        let eye4: Array2<Complex64> = Array2::eye(4);
        let c2 = QuantumChannel::new(vec![eye4]).unwrap();
        let result = c1.compose(&c2);
        assert!(matches!(result, Err(ChannelError::OperatorSizeMismatch)));
    }

    #[test]
    fn test_compose_success() {
        let c1 = QuantumChannel::bit_flip(0.1);
        let c2 = QuantumChannel::phase_flip(0.2);
        let composed = c1.compose(&c2).unwrap();
        // A composed channel of two 1-qubit channels with 2 kraus ops each should have 4 kraus ops
        assert_eq!(composed.kraus_ops.len(), 4);
        assert_eq!(composed.num_qubits, 1);
    }

    #[test]
    fn test_mix_invalid_probability() {
        let c1 = QuantumChannel::bit_flip(0.1);
        let c2 = QuantumChannel::phase_flip(0.1);
        let result = c1.mix(&c2, 1.5);
        assert!(matches!(result, Err(ChannelError::InvalidProbability(_))));
    }

    #[test]
    fn test_mix_size_mismatch() {
        let c1 = QuantumChannel::bit_flip(0.1);
        let eye4: Array2<Complex64> = Array2::eye(4);
        let c2 = QuantumChannel::new(vec![eye4]).unwrap();
        let result = c1.mix(&c2, 0.5);
        assert!(matches!(result, Err(ChannelError::OperatorSizeMismatch)));
    }

    #[test]
    fn test_mix_success() {
        let c1 = QuantumChannel::bit_flip(0.1);
        let c2 = QuantumChannel::phase_flip(0.1);
        let mixed = c1.mix(&c2, 0.3).unwrap();
        // Mixing two channels concatenates their Kraus operators and scales them
        assert_eq!(
            mixed.kraus_ops.len(),
            c1.kraus_ops.len() + c2.kraus_ops.len()
        );
        assert_eq!(mixed.num_qubits, 1);
    }

    // --- get_expanded_operators boundary test ---

    #[test]
    fn test_expanded_operators_target_mismatch() {
        let channel = QuantumChannel::bit_flip(0.5);
        let result = channel.get_expanded_operators(2, &[0, 1]);
        assert!(matches!(result, Err(ChannelError::InvalidDimensions)));
    }

    #[test]
    fn test_expanded_operators_success() {
        let channel = QuantumChannel::bit_flip(0.5);
        let expanded = channel.get_expanded_operators(2, &[1]).unwrap();
        assert_eq!(expanded.len(), 2);
        assert_eq!(expanded[0].dim(), (4, 4));
        assert_eq!(expanded[1].dim(), (4, 4));
    }

    // --- Predefined channels: panic on invalid probability ---

    #[test]
    #[should_panic(expected = "Invalid probability")]
    fn test_bit_flip_invalid_prob() {
        QuantumChannel::bit_flip(-0.1);
    }

    #[test]
    #[should_panic(expected = "Invalid probability")]
    fn test_depolarizing_invalid_prob() {
        QuantumChannel::depolarizing(1.5);
    }

    // --- CPTP property: sum(K†K) = I for all predefined channels ---

    fn assert_trace_preserving(channel: &QuantumChannel) {
        let dim = channel.kraus_ops[0].dim().0;
        let eye = Array2::<Complex64>::eye(dim);
        let mut sum = Array2::<Complex64>::zeros((dim, dim));
        for op in &channel.kraus_ops {
            let dag = op.t().mapv(|x| x.conj());
            sum = sum + dag.dot(op);
        }
        for (a, b) in sum.iter().zip(eye.iter()) {
            assert!((*a - *b).norm() < 1e-10);
        }
    }

    #[test]
    fn test_all_predefined_channels_are_cptp() {
        for i in 0..=10 {
            let p = i as f64 / 10.0;
            assert_trace_preserving(&QuantumChannel::identity());
            assert_trace_preserving(&QuantumChannel::bit_flip(p));
            assert_trace_preserving(&QuantumChannel::phase_flip(p));
            assert_trace_preserving(&QuantumChannel::bit_phase_flip(p));
            assert_trace_preserving(&QuantumChannel::depolarizing(p));
            assert_trace_preserving(&QuantumChannel::amplitude_damping(p));
            assert_trace_preserving(&QuantumChannel::phase_damping(p));
            for j in 0..=10 {
                let p2 = j as f64 / 10.0;
                assert_trace_preserving(&QuantumChannel::combined_amplitude_phase_damping(p, p2));
            }
        }
    }

    // --- Identity channel at p=0 ---

    #[test]
    fn test_bit_flip_p0_is_identity() {
        let channel = QuantumChannel::bit_flip(0.0);
        // K0 should be I, K1 should be zero
        let identity = QuantumChannel::identity();
        for (a, b) in channel.kraus_ops[0].iter().zip(&identity.kraus_ops[0]) {
            assert!((*a - *b).norm() < 1e-12);
        }
        for &val in channel.kraus_ops[1].iter() {
            assert!(val.norm() < 1e-12);
        }
    }
}
