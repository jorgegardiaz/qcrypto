#[cfg(feature = "parallel")]
use crate::rng::LocalRng;
use crate::{Measurement, QuantumChannel, QuantumState, errors::StateError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
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
            let outcome_idx = cdf.partition_point(|&cumulative_prob| cumulative_prob <= r);

            // Increment the counter for this index directly
            raw_counts[outcome_idx] += 1;
        }

        // Convert indices to the final Result HashMap.
        let mut counts = HashMap::new();
        for (idx, &count) in raw_counts.iter().enumerate() {
            if count > 0 {
                counts.insert(measurement.labels[idx].clone(), count);
            }
        }

        Ok(counts)
    }

    /// Parallelised variant of [`run`] using rayon.
    ///
    /// The state is cloned and the channel (if any) is applied once on the calling thread.
    /// The probability distribution and CDF are then computed once, and all shots are drawn
    /// in parallel — each shot gets an independent child RNG derived from a master seed
    /// drawn from the global thread-local RNG before the parallel section begins.
    ///
    /// Calling `qcrypto::set_global_seed` before this method makes the full output
    /// reproducible, because the master seed is derived deterministically from the seeded RNG.
    ///
    /// Requires the `parallel` feature (enabled by default).
    ///
    /// # Arguments
    ///
    /// * `state` - The quantum state to sample from.
    /// * `measurement` - The measurement operator (POVM) to apply.
    /// * `targets` - The indices of the qubits to measure.
    /// * `num_shots` - The number of parallel shots to draw.
    ///
    /// # Returns
    ///
    /// A `Result` containing a `HashMap` mapping outcome labels to their counts.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if there is a dimension mismatch, invalid target indices,
    /// or if applying the channel or computing the probability distribution fails.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{Gate, Measurement, QuantumState, Sampler};
    ///
    /// let mut state = QuantumState::new(1);
    /// state.apply(&Gate::h(), &[0]).unwrap(); // |+>
    /// let sampler = Sampler::new();
    ///
    /// qcrypto::set_global_seed(42);
    /// let r1 = sampler.run_par(&state, &Measurement::z_basis(), &[0], 1000).unwrap();
    ///
    /// qcrypto::set_global_seed(42);
    /// let r2 = sampler.run_par(&state, &Measurement::z_basis(), &[0], 1000).unwrap();
    ///
    /// assert_eq!(r1, r2);
    /// ```
    #[cfg(feature = "parallel")]
    pub fn run_par(
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

        let probs = state_copy.set_measurement(measurement, targets)?;

        let mut cdf = Vec::with_capacity(probs.len());
        let mut current_sum = 0.0;
        for &p in &probs {
            current_sum += p;
            cdf.push(current_sum);
        }

        // Draw master seed from the global RNG before entering the parallel section.
        // This preserves determinism: after set_global_seed(s), the master is always
        // the same value, which makes every child RNG, and thus every shot outcome,
        // reproducible regardless of thread scheduling.
        let master = crate::rng::draw_master_seed();

        let outcomes: Vec<usize> = (0..num_shots)
            .into_par_iter()
            .map(|i| {
                let mut rng = LocalRng::child(master, i as u64);
                let r = rng.random_f64();
                cdf.partition_point(|&cp| cp <= r)
            })
            .collect();

        let mut raw_counts = vec![0usize; probs.len()];
        for outcome_idx in outcomes {
            raw_counts[outcome_idx] += 1;
        }

        let mut counts = HashMap::new();
        for (idx, &count) in raw_counts.iter().enumerate() {
            if count > 0 {
                counts.insert(measurement.labels[idx].clone(), count);
            }
        }

        Ok(counts)
    }

    /// Samples all qubits in the computational (Z) basis using the fast O(2^N) path.
    ///
    /// Unlike [`run`], this method reads probabilities directly from the state's amplitudes
    /// or density-matrix diagonal rather than applying measurement operators. It is
    /// equivalent to `run` with a fully-composed Z-basis measurement over all qubits but
    /// avoids the O(4^N) operator overhead of the general path.
    ///
    /// If the sampler has a channel set, it is applied to a clone of the state first (which
    /// converts it to a density matrix), after which the diagonal is read.
    ///
    /// Outcome labels are zero-padded binary strings, e.g. `"001"` for a 3-qubit state.
    ///
    /// # Arguments
    ///
    /// * `state` - The quantum state to sample from.
    /// * `num_shots` - Number of shots to draw.
    ///
    /// # Returns
    ///
    /// A `HashMap` mapping binary outcome strings to their counts.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if applying the channel fails.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{Gate, QuantumState, Sampler};
    ///
    /// let mut state = QuantumState::new(2);
    /// state.apply(&Gate::h(), &[0]).unwrap();
    /// state.apply(&Gate::h(), &[1]).unwrap(); // fully superposed
    ///
    /// qcrypto::set_global_seed(42);
    /// let counts = Sampler::new().run_computational_basis(&state, 1000).unwrap();
    /// let total: usize = counts.values().sum();
    /// assert_eq!(total, 1000);
    /// ```
    pub fn run_computational_basis(
        &self,
        state: &QuantumState,
        num_shots: usize,
    ) -> Result<HashMap<String, usize>, StateError> {
        let mut state_copy = state.clone();
        if let Some(chan) = &self.channel {
            // apply_channel needs target qubits — apply to all
            let n = state_copy.probabilities().len().trailing_zeros() as usize;
            let targets: Vec<usize> = (0..n).collect();
            state_copy.apply_channel(chan, &targets)?;
        }

        let probs = state_copy.probabilities();
        let n_qubits = probs.len().trailing_zeros() as usize;

        let mut cdf = Vec::with_capacity(probs.len());
        let mut sum = 0.0;
        for &p in &probs {
            sum += p;
            cdf.push(sum);
        }

        let mut raw_counts = vec![0usize; probs.len()];
        for _ in 0..num_shots {
            let r = crate::rng::random_f64();
            let idx = cdf.partition_point(|&cp| cp <= r);
            raw_counts[idx] += 1;
        }

        let mut counts = HashMap::new();
        for (idx, &count) in raw_counts.iter().enumerate() {
            if count > 0 {
                counts.insert(format!("{:0>width$b}", idx, width = n_qubits), count);
            }
        }

        Ok(counts)
    }

    /// Parallel variant of [`run_computational_basis`] using rayon.
    ///
    /// Follows the same deterministic seeding pattern as [`run_par`]: the master seed is
    /// drawn from the global RNG before entering the parallel section, so calling
    /// `qcrypto::set_global_seed` before this method makes the output fully reproducible.
    ///
    /// Requires the `parallel` feature (enabled by default).
    ///
    /// # Arguments
    ///
    /// * `state` - The quantum state to sample from.
    /// * `num_shots` - Number of parallel shots to draw.
    ///
    /// # Returns
    ///
    /// A `HashMap` mapping binary outcome strings to their counts.
    ///
    /// # Errors
    ///
    /// Returns a `StateError` if applying the channel fails.
    ///
    /// # Example
    /// ```rust
    /// use qcrypto::{Gate, QuantumState, Sampler};
    ///
    /// let mut state = QuantumState::new(2);
    /// state.apply(&Gate::h(), &[0]).unwrap();
    /// state.apply(&Gate::h(), &[1]).unwrap();
    ///
    /// qcrypto::set_global_seed(42);
    /// let r1 = Sampler::new().run_par_computational_basis(&state, 1000).unwrap();
    /// qcrypto::set_global_seed(42);
    /// let r2 = Sampler::new().run_par_computational_basis(&state, 1000).unwrap();
    /// assert_eq!(r1, r2);
    /// ```
    #[cfg(feature = "parallel")]
    pub fn run_par_computational_basis(
        &self,
        state: &QuantumState,
        num_shots: usize,
    ) -> Result<HashMap<String, usize>, StateError> {
        let mut state_copy = state.clone();
        if let Some(chan) = &self.channel {
            let n = state_copy.probabilities().len().trailing_zeros() as usize;
            let targets: Vec<usize> = (0..n).collect();
            state_copy.apply_channel(chan, &targets)?;
        }

        let probs = state_copy.probabilities();
        let n_qubits = probs.len().trailing_zeros() as usize;

        let mut cdf = Vec::with_capacity(probs.len());
        let mut sum = 0.0;
        for &p in &probs {
            sum += p;
            cdf.push(sum);
        }

        let master = crate::rng::draw_master_seed();

        let outcomes: Vec<usize> = (0..num_shots)
            .into_par_iter()
            .map(|i| {
                let mut rng = LocalRng::child(master, i as u64);
                let r = rng.random_f64();
                cdf.partition_point(|&cp| cp <= r)
            })
            .collect();

        let mut raw_counts = vec![0usize; probs.len()];
        for idx in outcomes {
            raw_counts[idx] += 1;
        }

        let mut counts = HashMap::new();
        for (idx, &count) in raw_counts.iter().enumerate() {
            if count > 0 {
                counts.insert(format!("{:0>width$b}", idx, width = n_qubits), count);
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

    // ── run_par unit tests ────────────────────────────────────────────────────

    #[cfg(feature = "parallel")]
    #[test]
    fn test_run_par_deterministic_state() {
        let state = QuantumState::new(1); // |0>
        let sampler = Sampler::new();
        let counts = sampler
            .run_par(&state, &Measurement::z_basis(), &[0], 200)
            .unwrap();
        assert_eq!(counts.len(), 1);
        assert_eq!(*counts.get("0").unwrap(), 200);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_run_par_excited_state() {
        let mut state = QuantumState::new(1);
        state.apply(&Gate::x(), &[0]).unwrap(); // |1>
        let sampler = Sampler::new();
        let counts = sampler
            .run_par(&state, &Measurement::z_basis(), &[0], 200)
            .unwrap();
        assert_eq!(counts.len(), 1);
        assert_eq!(*counts.get("1").unwrap(), 200);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_run_par_deterministic_with_seed() {
        let run_once = |seed: u64| {
            crate::rng::set_global_seed(seed);
            let mut state = QuantumState::new(1);
            state.apply(&Gate::h(), &[0]).unwrap();
            let sampler = Sampler::new();
            sampler
                .run_par(&state, &Measurement::z_basis(), &[0], 200)
                .unwrap()
        };

        let c1 = run_once(99);
        let c2 = run_once(99);
        assert_eq!(c1, c2, "same seed must yield identical counts");
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_run_par_with_channel() {
        let state = QuantumState::new(1); // |0>
        let channel = QuantumChannel::bit_flip(1.0); // always flip
        let sampler = Sampler::new().with_channel(channel);
        let counts = sampler
            .run_par(&state, &Measurement::z_basis(), &[0], 100)
            .unwrap();
        assert_eq!(*counts.get("1").unwrap(), 100);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_run_par_errors_propagated() {
        let state = QuantumState::new(1);
        let sampler = Sampler::new();

        let result = sampler.run_par(&state, &Measurement::z_basis(), &[5], 10);
        assert!(result.is_err());

        let result2 = sampler.run_par(&state, &Measurement::bell_basis(), &[0], 10);
        assert!(result2.is_err());
    }

    // ── computational basis unit tests ────────────────────────────────────────

    #[test]
    fn test_run_computational_basis_deterministic() {
        let mut state = QuantumState::new(2);
        state.apply(&Gate::x(), &[0]).unwrap(); // |10>
        let sampler = Sampler::new();
        let counts = sampler.run_computational_basis(&state, 100).unwrap();
        assert_eq!(counts.len(), 1);
        assert_eq!(*counts.get("10").unwrap(), 100);
    }

    #[test]
    fn test_run_computational_basis_superposition() {
        let mut state = QuantumState::new(1);
        state.apply(&Gate::h(), &[0]).unwrap(); // |+>
        let sampler = Sampler::new();
        let counts = sampler.run_computational_basis(&state, 1000).unwrap();
        let c0 = *counts.get("0").unwrap_or(&0);
        let c1 = *counts.get("1").unwrap_or(&0);
        assert!(c0 > 350 && c0 < 650);
        assert!(c1 > 350 && c1 < 650);
    }

    #[test]
    fn test_run_computational_basis_with_channel() {
        let state = QuantumState::new(1); // |0>
        let channel = QuantumChannel::bit_flip(1.0);
        let sampler = Sampler::new().with_channel(channel);
        let counts = sampler.run_computational_basis(&state, 100).unwrap();
        assert_eq!(*counts.get("1").unwrap(), 100);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_run_par_computational_basis_deterministic() {
        let mut state = QuantumState::new(2);
        state.apply(&Gate::x(), &[1]).unwrap(); // |01>
        let sampler = Sampler::new();
        let counts = sampler.run_par_computational_basis(&state, 100).unwrap();
        assert_eq!(counts.len(), 1);
        assert_eq!(*counts.get("01").unwrap(), 100);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_run_par_computational_basis_with_channel() {
        let state = QuantumState::new(1); // |0>
        let channel = QuantumChannel::bit_flip(1.0);
        let sampler = Sampler::new().with_channel(channel);
        let counts = sampler.run_par_computational_basis(&state, 100).unwrap();
        assert_eq!(*counts.get("1").unwrap(), 100);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_run_par_computational_basis_with_seed() {
        let run_once = |seed: u64| {
            crate::rng::set_global_seed(seed);
            let mut state = QuantumState::new(1);
            state.apply(&Gate::h(), &[0]).unwrap();
            let sampler = Sampler::new();
            sampler.run_par_computational_basis(&state, 1000).unwrap()
        };

        let c1 = run_once(42);
        let c2 = run_once(42);
        assert_eq!(c1, c2);
    }
}
