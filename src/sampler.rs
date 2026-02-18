use crate::{Measurement, QuantumChannel, QuantumState, errors::StateError};
use rand::Rng;
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct Sampler {
    pub channel: Option<QuantumChannel>,
}

impl Sampler {
    pub fn new() -> Self {
        Self { channel: None }
    }

    /// Add QuantumChannel to QuantumSampler
    pub fn with_channel(mut self, channel: QuantumChannel) -> Self {
        self.channel = Some(channel);
        self
    }

    /// Samples a QuantumState num_shots times using a designed QuantumMeasurement.
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
        let (probs, _) = state_copy.set_measurement(measurement, targets)?;

        // Pre-calculate Cumulative Distribution Function (CDF) once.
        let mut cdf = Vec::with_capacity(probs.len());
        let mut current_sum = 0.0;
        for &p in &probs {
            current_sum += p;
            cdf.push(current_sum);
        }

        // Run Simulation
        let mut raw_counts = vec![0usize; probs.len()];
        let mut rng = rand::rng();

        for _ in 0..num_shots {
            let r: f64 = rng.random(); // Generates [0.0, 1.0)

            // Determine outcome index based on CDF
            let mut outcome_idx = 0;
            for (i, &cumulative_prob) in cdf.iter().enumerate() {
                if r < cumulative_prob {
                    outcome_idx = i;
                    break;
                }
            }

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
