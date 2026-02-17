use crate::channels::QuantumChannel;
use crate::core::errors::StateError;
use crate::core::state::QuantumState;
use crate::measure::Measurement;
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct QuantumSimulator {
    pub channel: Option<QuantumChannel>,
}

impl QuantumSimulator {
    pub fn new() -> Self {
        Self { channel: None }
    }

    /// Add QuantumChannel to QuantumSimulator
    pub fn with_channel(mut self, channel: QuantumChannel) -> Self {
        self.channel = Some(channel);
        self
    }

    /// Samples a QuantumState num_shots times using a designed QuantumMeasurement
    pub fn sample(
        &self,
        state: &QuantumState,
        measurement: &Measurement,
        targets: &[usize],
        num_shots: usize,
    ) -> Result<HashMap<String, usize>, StateError> {
        let mut counts = HashMap::new();

        for _ in 0..num_shots {
            // Clones QuantumState to not modify it
            let mut state_copy = state.clone();

            // Applies channel if exists
            if let Some(chan) = &self.channel {
                state_copy.apply_channel(chan, targets)?;
            }

            let result_value = state_copy.measure(measurement, targets)?;

            let key = result_value.value.to_string();
            *counts.entry(key).or_insert(0) += 1;
        }

        Ok(counts)
    }
}
