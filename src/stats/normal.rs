//! Emission distributions shared by the HMM port.
//!
//! `NormalDistribution` is a faithful port of pomegranate 0.10.0's
//! `distributions/NormalDistribution.pyx`; `Emission` mirrors the `Model`
//! interface pomegranate's HMM uses to talk to a state's distribution
//! (`_log_probability` / `_summarize` / `from_summaries`).

/// Natural log, but `-inf` for `0` (pomegranate's `_log`).
pub fn log_pos(x: f64) -> f64 {
    if x > 0.0 {
        x.ln()
    } else {
        NEG_INF
    }
}

/// `log(e^x + e^y)` computed stably (pomegranate's `pair_lse`).
pub fn pair_lse(x: f64, y: f64) -> f64 {
    if x == f64::INFINITY || y == f64::INFINITY {
        return f64::INFINITY;
    }
    if x == NEG_INF {
        return y;
    }
    if y == NEG_INF {
        return x;
    }
    if x > y {
        x + (y - x).exp().ln_1p()
    } else {
        y + (x - y).exp().ln_1p()
    }
}

/// Negative infinity in log space.
pub const NEG_INF: f64 = f64::NEG_INFINITY;
/// pomegranate's `SQRT_2_PI` constant (truncated to 11 decimals in the source).
pub const SQRT_2_PI: f64 = 2.50662827463;

/// The interface pomegranate's HMM needs from a state's emission model:
/// score a symbol, collect weighted sufficient statistics (E step) and
/// re-estimate parameters from them (M step).
pub trait Emission {
    /// Log probability of a single symbol.
    fn log_probability(&self, x: f64) -> f64;
    /// Accumulate sufficient statistics over `xs`, each weighted by `weights`.
    fn summarize(&mut self, xs: &[f64], weights: &[f64]);
    /// Re-estimate parameters from accumulated statistics (pomegranate's
    /// `from_summaries`). `emission_pseudocount` only affects discrete
    /// distributions.
    #[allow(clippy::wrong_self_convention)]
    fn from_summaries(&mut self, inertia: f64, emission_pseudocount: f64);
    /// Zero out accumulated statistics.
    fn clear_summaries(&mut self);
}

/// 1-D normal distribution `N(mu, sigma)`.
///
/// Ported from pomegranate 0.10.0 `NormalDistribution.pyx`. The summary
/// statistics are `[sum w, sum w·x, sum w·x²]`; the M-step recomputes
/// `mu`, `sigma` from them, clamping `sigma` to `min_std`.
#[derive(Debug, Clone)]
pub struct NormalDistribution {
    pub mu: f64,
    pub sigma: f64,
    summaries: [f64; 3],
    min_std: f64,
    log_sigma_sqrt_2_pi: f64,
    two_sigma_squared: f64,
}

impl NormalDistribution {
    pub fn new(mu: f64, sigma: f64) -> Self {
        Self::with_min_std(mu, sigma, 0.0)
    }

    pub fn with_min_std(mu: f64, sigma: f64, min_std: f64) -> Self {
        NormalDistribution {
            mu,
            sigma,
            summaries: [0.0; 3],
            min_std,
            log_sigma_sqrt_2_pi: -(sigma * SQRT_2_PI).ln(),
            two_sigma_squared: 1.0 / (2.0 * sigma * sigma),
        }
    }
}

impl Emission for NormalDistribution {
    fn log_probability(&self, x: f64) -> f64 {
        // pomegranate skips NaN symbols (logp 0); DI values never go NaN.
        self.log_sigma_sqrt_2_pi - (x - self.mu).powi(2) * self.two_sigma_squared
    }

    fn summarize(&mut self, xs: &[f64], weights: &[f64]) {
        for (w, &x) in weights.iter().zip(xs) {
            self.summaries[0] += w;
            self.summaries[1] += w * x;
            self.summaries[2] += w * x * x;
        }
    }

    fn from_summaries(&mut self, inertia: f64, _emission_pseudocount: f64) {
        if self.summaries[0] < 1e-8 {
            return;
        }

        let mu = self.summaries[1] / self.summaries[0];
        let var = self.summaries[2] / self.summaries[0]
            - self.summaries[1].powi(2) / self.summaries[0].powi(2);

        let mut sigma = var.sqrt();
        if sigma < self.min_std {
            sigma = self.min_std;
        }

        self.mu = self.mu * inertia + mu * (1.0 - inertia);
        self.sigma = self.sigma * inertia + sigma * (1.0 - inertia);
        self.summaries = [0.0; 3];
        // pomegranate recomputes the constants from the *new* sigma.
        self.log_sigma_sqrt_2_pi = -(sigma * SQRT_2_PI).ln();
        self.two_sigma_squared = 1.0 / (2.0 * sigma * sigma);
    }

    fn clear_summaries(&mut self) {
        self.summaries = [0.0; 3];
    }
}
