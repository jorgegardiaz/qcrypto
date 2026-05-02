use crate::{Measurement, QuantumChannel, QuantumState, errors::StateError};
use std::collections::HashMap;

/// A simulator for sampling quantum states.
///
/// The `Sampler` allows running multiple shots of a quantum measurement on a quantum state,
/// optionally applying a quantum channel before measurement.
///
/// # Example
/// ```rust
/// use qcrypto::{Sampler, QuantumState, Measurement};
///
/// let state = QuantumState::new(1); // |0>
/// let sampler = Sampler::new();
/// let results = sampler.run(&state, &Measurement::z_basis(), &[0], 10).unwrap();
/// ```
#[derive(Debug, Clone, Default)]
pub struct Sampler {
    /// Optional quantum channel to apply to the state before measurement.
    pub channel: Option<QuantumChannel>,
}

impl Sampler {
    /// Creates a new `Sampler` instance with no channel (noise-free).
    ///
    /// # Returns
    ///
    /// A new `Sampler` instance initialized with `None` for its `channel` field.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::Sampler;
    ///
    /// let sampler = Sampler::new();
    /// assert!(sampler.channel.is_none());
    /// ```
    pub fn new() -> Self {
        Self { channel: None }
    }

    /// Sets the quantum channel for the sampler.
    ///
    /// This method allows for a fluent, builder-style API to configure the sampler
    /// with a noise model or decoherence channel before taking measurements.
    ///
    /// # Arguments
    ///
    /// * `channel` - The `QuantumChannel` to apply.
    ///
    /// # Returns
    ///
    /// The modified `Sampler` instance containing the provided channel.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{Sampler, QuantumChannel};
    ///
    /// let channel = QuantumChannel::bit_flip(0.1);
    /// let sampler = Sampler::new().with_channel(channel);
    /// assert!(sampler.channel.is_some());
    /// ```
    pub fn with_channel(mut self, channel: QuantumChannel) -> Self {
        self.channel = Some(channel);
        self
    }

    /// Samples a `QuantumState` multiple times using a specified `Measurement`.
    ///
    /// This method simulates the process of preparing a state, optionally passing it through a
    /// channel, and then measuring it. It returns a distribution of measurement outcomes.
    ///
    /// # Arguments
    ///
    /// * `state` - The quantum state to measure.
    /// * `measurement` - The measurement operator (POVM) to apply.
    /// * `targets` - The indices of the qubits to measure.
    /// * `num_shots` - The number of times to repeat the measurement.
    ///
    /// # Returns
    ///
    /// A `Result` containing a `HashMap` mapping outcome labels (strings) to their counts.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if there is a dimension mismatch, invalid target indices,
    /// or if applying the channel or measurement fails.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{Sampler, QuantumState, Measurement};
    ///
    /// let state = QuantumState::new(1); // |0>
    /// let sampler = Sampler::new();
    ///
    /// // Sample 100 times in Z basis
    /// let counts = sampler.run(&state, &Measurement::z_basis(), &[0], 100).unwrap();
    ///
    /// // Since it's exactly |0>, we should get 100 shots of "0"
    /// assert_eq!(*counts.get("0").unwrap_or(&0), 100);
    /// assert!(counts.get("1").is_none());
    /// ```
    pub fn run(
        &self,
        state: &QuantumState,
        measurement: &Measurement,
        targets: &[usize],
        num_shots: usize,
    ) -> Result<HashMap<String, usize>, StateError> {
        let mut state_copy = state.clone();

        if let Some(chan) = &self.channel {
            state_copy.apply_channel(chan, targets)?;
        }

        // Get probabilities for each possible outcome
        let probs = state_copy.set_measurement(measurement, targets)?;

        // Pre-calculate Cumulative Distribution Function (CDF) once.
        let mut cdf = Vec::with_capacity(probs.len());
        let mut current_sum = 0.0;
        for &p in &probs {
            current_sum += p;
            cdf.push(current_sum);
        }

        // Run Simulation
        let mut raw_counts = vec![0usize; probs.len()];

        for _ in 0..num_shots {
            let r: f64 = crate::rng::random_f64(); // Generates [0.0, 1.0)

            // Determine outcome index based on CDF using binary search (O(log N))
            let mut outcome_idx = cdf.partition_point(|&cumulative_prob| cumulative_prob <= r);

            // Safety check for floating point rounding errors
            if outcome_idx >= probs.len() {
                outcome_idx = probs.len() - 1;
            }

            // Increment the counter for this index directly
            raw_counts[outcome_idx] += 1;
        }

        // Convert indices to the final Result HashMap.
        let mut counts = HashMap::new();
        for (idx, &count) in raw_counts.iter().enumerate() {
            if count > 0 {
                let val_string = measurement.values[idx].to_string();
                counts.insert(val_string, count);
            }
        }

        Ok(counts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gate;

    #[test]
    fn test_sampler_deterministic_zero() {
        let state = QuantumState::new(1); // |0>
        let sampler = Sampler::new();
        let counts = sampler
            .run(&state, &Measurement::z_basis(), &[0], 100)
            .unwrap();

        assert_eq!(counts.len(), 1);
        assert_eq!(*counts.get("0").unwrap(), 100);
    }

    #[test]
    fn test_sampler_deterministic_one() {
        let mut state = QuantumState::new(1);
        state.apply(&Gate::x(), &[0]).unwrap(); // |1>

        let sampler = Sampler::new();
        let counts = sampler
            .run(&state, &Measurement::z_basis(), &[0], 100)
            .unwrap();

        assert_eq!(counts.len(), 1);
        assert_eq!(*counts.get("1").unwrap(), 100);
    }

    #[test]
    fn test_sampler_superposition() {
        let mut state = QuantumState::new(1);
        state.apply(&Gate::h(), &[0]).unwrap(); // |+>

        let sampler = Sampler::new();
        let num_shots = 1000;
        let counts = sampler
            .run(&state, &Measurement::z_basis(), &[0], num_shots)
            .unwrap();

        // We expect roughly a 50/50 split.
        // With 1000 shots, getting less than 350 of one is statistically extremely unlikely.
        let count_0 = *counts.get("0").unwrap_or(&0);
        let count_1 = *counts.get("1").unwrap_or(&0);

        assert!(
            count_0 > 350 && count_0 < 650,
            "Expected roughly 500, got {}",
            count_0
        );
        assert!(
            count_1 > 350 && count_1 < 650,
            "Expected roughly 500, got {}",
            count_1
        );
        assert_eq!(count_0 + count_1, num_shots);
    }

    #[test]
    fn test_sampler_with_bit_flip_channel() {
        let state = QuantumState::new(1); // |0>
        // Apply a channel that flips the bit with 100% probability
        let channel = QuantumChannel::bit_flip(1.0);
        let sampler = Sampler::new().with_channel(channel);

        let counts = sampler
            .run(&state, &Measurement::z_basis(), &[0], 100)
            .unwrap();

        // The noise should have flipped the state to |1> before measurement
        assert_eq!(counts.len(), 1);
        assert_eq!(*counts.get("1").unwrap(), 100);
    }

    #[test]
    fn test_sampler_errors_propagated() {
        let state = QuantumState::new(1);
        let sampler = Sampler::new();

        // Out of bounds target
        let result = sampler.run(&state, &Measurement::z_basis(), &[5], 10);
        assert!(result.is_err());

        // Dimension mismatch (2-qubit measurement on 1-qubit target list)
        let result2 = sampler.run(&state, &Measurement::bell_basis(), &[0], 10);
        assert!(result2.is_err());
    }
}
