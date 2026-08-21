//! Categorical (discrete) emission distribution.
//!
//! Ported from pomegranate 0.10.0 `distributions/DiscreteDistribution.pyx`.
//! Symbols are represented by integer indices `0..n` (pomegranate maps
//! symbols to indices through the model's keymap); the distribution holds the
//! log probability of each index.

use crate::stats::normal::{log_pos, Emission, NEG_INF};

/// A discrete distribution over `n` symbols (indexed `0..n`).
///
/// The M step re-estimates probabilities from weighted counts, adding
/// `emission_pseudocount` to every symbol (Laplace-style smoothing) and
/// blending with the previous parameters by `inertia`.
#[derive(Debug, Clone)]
pub struct DiscreteDistribution {
    log_probs: Vec<f64>,
    summaries: Vec<f64>,
}

impl DiscreteDistribution {
    /// Build from linear probabilities in index order.
    pub fn new(probs: &[f64]) -> Self {
        DiscreteDistribution {
            log_probs: probs.iter().map(|&p| log_pos(p)).collect(),
            summaries: vec![0.0; probs.len()],
        }
    }

    /// Number of symbols.
    pub fn n(&self) -> usize {
        self.log_probs.len()
    }
}

impl Emission for DiscreteDistribution {
    fn log_probability(&self, x: f64) -> f64 {
        let i = x as usize;
        if x < 0.0 || i >= self.log_probs.len() {
            NEG_INF
        } else {
            self.log_probs[i]
        }
    }

    fn summarize(&mut self, xs: &[f64], weights: &[f64]) {
        for (w, &x) in weights.iter().zip(xs) {
            self.summaries[x as usize] += w;
        }
    }

    fn from_summaries(&mut self, inertia: f64, pseudocount: f64) {
        let n = self.log_probs.len();
        let total: f64 = self.summaries.iter().sum();
        if total == 0.0 {
            return;
        }
        let denom = total + pseudocount * n as f64;
        for i in 0..n {
            let value = (self.summaries[i] + pseudocount) / denom;
            self.log_probs[i] =
                log_pos(self.log_probs[i].exp() * inertia + value * (1.0 - inertia));
            self.summaries[i] = 0.0;
        }
    }

    fn clear_summaries(&mut self) {
        self.summaries.iter_mut().for_each(|s| *s = 0.0);
    }
}
