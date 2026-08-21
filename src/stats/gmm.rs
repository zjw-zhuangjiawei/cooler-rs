//! Gaussian mixture model over `NormalDistribution` components.
//!
//! Ported from pomegranate 0.10.0 (`gmm.pyx` `GeneralMixtureModel` plus the
//! `BayesModel` base class in `base.pyx`): `log_probability` is a
//! log-sum-exp over components, `fit` is EM via `summarize`/`from_summaries`.

use crate::stats::normal::{log_pos, pair_lse, Emission, NEG_INF};
use crate::stats::NormalDistribution;

/// A mixture of `NormalDistribution` components with learned weights.
///
/// Component weights are kept in log space, matching pomegranate.
#[derive(Debug, Clone)]
pub struct GeneralMixtureModel {
    pub components: Vec<NormalDistribution>,
    /// Log weights (length = number of components).
    pub log_weights: Vec<f64>,
    /// Linear component-marginal counts accumulated during `summarize`.
    summaries: Vec<f64>,
}

impl GeneralMixtureModel {
    /// Build a mixture over the given components. `weights` (if given) need
    /// not sum to 1; otherwise components are uniform.
    pub fn new(components: Vec<NormalDistribution>, weights: Option<Vec<f64>>) -> Self {
        let n = components.len();
        assert!(n >= 2, "must have at least two components");
        let w = match weights {
            Some(w) => {
                let total: f64 = w.iter().sum();
                w.into_iter().map(|x| x / total).collect()
            }
            None => vec![1.0 / n as f64; n],
        };
        GeneralMixtureModel {
            log_weights: w.iter().map(|&x| log_pos(x)).collect(),
            components,
            summaries: vec![0.0; n],
        }
    }

    /// Fit via EM until the log-probability improvement falls below
    /// `stop_threshold` or `max_iterations` is reached. Returns the total
    /// improvement, matching pomegranate's `fit`.
    pub fn fit(&mut self, xs: &[f64], stop_threshold: f64, max_iterations: usize) -> f64 {
        let weights = vec![1.0; xs.len()];
        let mut improvement = f64::INFINITY;
        let mut iteration = 0usize;
        let mut last = 0.0;
        let mut total_improvement = 0.0;
        while improvement > stop_threshold && iteration < max_iterations + 1 {
            self.from_summaries(0.0, 0.0);
            let logp_sum = self.summarize(xs, &weights);
            if iteration > 0 {
                improvement = logp_sum - last;
                total_improvement += improvement;
            }
            iteration += 1;
            last = logp_sum;
        }
        total_improvement
    }

    /// Log probability of a single symbol.
    pub fn log_probability(&self, x: f64) -> f64 {
        let mut logp = NEG_INF;
        for (c, &w) in self.components.iter().zip(&self.log_weights) {
            logp = pair_lse(logp, c.log_probability(x) + w);
        }
        logp
    }

    /// E step: accumulate responsibilities into each component and into the
    /// mixture-marginal counts. Returns the log probability of the data.
    pub fn summarize(&mut self, xs: &[f64], weights: &[f64]) -> f64 {
        let n = xs.len();
        let m = self.components.len();
        // r[j][i] = P(component j | x_i) * weights[i]
        let mut r = vec![vec![0.0; n]; m];
        let mut logp_sum = 0.0;
        for i in 0..n {
            let mut total = NEG_INF;
            for (j, c) in self.components.iter().enumerate() {
                r[j][i] = c.log_probability(xs[i]) + self.log_weights[j];
                total = pair_lse(total, r[j][i]);
            }
            for (j, rj) in r.iter_mut().enumerate() {
                rj[i] = (rj[i] - total).exp() * weights[i];
                self.summaries[j] += rj[i];
            }
            logp_sum += total * weights[i];
        }
        for (j, c) in self.components.iter_mut().enumerate() {
            c.summarize(xs, &r[j]);
        }
        logp_sum
    }

    /// M step: re-estimate mixture weights and each component's parameters.
    pub fn from_summaries(&mut self, inertia: f64, emission_pseudocount: f64) {
        let total: f64 = self.summaries.iter().sum();
        if total == 0.0 {
            return;
        }
        for (i, c) in self.components.iter_mut().enumerate() {
            c.from_summaries(inertia, emission_pseudocount);
            self.log_weights[i] = log_pos(self.summaries[i] / total);
            self.summaries[i] = 0.0;
        }
    }

    pub fn clear_summaries(&mut self) {
        self.summaries.iter_mut().for_each(|s| *s = 0.0);
        for c in &mut self.components {
            c.clear_summaries();
        }
    }
}

impl Emission for GeneralMixtureModel {
    fn log_probability(&self, x: f64) -> f64 {
        GeneralMixtureModel::log_probability(self, x)
    }

    fn summarize(&mut self, xs: &[f64], weights: &[f64]) {
        GeneralMixtureModel::summarize(self, xs, weights);
    }

    fn from_summaries(&mut self, inertia: f64, emission_pseudocount: f64) {
        GeneralMixtureModel::from_summaries(self, inertia, emission_pseudocount);
    }

    fn clear_summaries(&mut self) {
        GeneralMixtureModel::clear_summaries(self);
    }
}
