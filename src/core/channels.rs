use crate::core::errors::ChannelError;
use crate::core::utils;
use ndarray::{Array2, array};
use num_complex::Complex64;

#[derive(Clone, Debug)]
pub struct QuantumChannel {
    pub kraus_ops: Vec<Array2<Complex64>>,
    pub num_qubits: usize,
}

impl QuantumChannel {
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

    /// Composes the current QuantumChannel with another one
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

    /// Mixes the current QuantumChannel with another one with ponderation p (convex combination)
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

    /// Expands measurements operator to a larger system
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

    /// Bit Flip Channel -> X
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

    /// Phase Flip Channel -> Z
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

    /// Bit-Phase Flip Channel -> Phase Flip + Bit FLip
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

    /// Depolarizing Channel
    /// The QuantumState totally randomices with porbability p
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

    /// Amplitude Damping -> T1 relaxation
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

    /// Phase Damping -> T2 relaxation
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

    /// Combined Amplitude & Phase Damping -> T1 relaxation + T2 relaxation
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
