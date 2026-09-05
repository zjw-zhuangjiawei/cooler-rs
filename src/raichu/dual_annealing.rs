//! Faithful port of scipy `scipy.optimize.dual_annealing`.
//!
//! Generalized simulated annealing (Tsallis/Stariolo visiting distribution,
//! Metropolis acceptance) coupled to a local search, exactly as scipy
//! implements it in `_dual_annealing.py`. Defaults match scipy: `visit=2.62`,
//! `accept=-5.0`, `initial_temp=5230.0`, `restart_temp_ratio=2e-5`,
//! `maxfun=1e7`.
//!
//! The local search is [`lbfgsb`], run at scipy's L-BFGS-B defaults (unbounded,
//! `maxiter=15000`); out-of-bounds or non-improving results are rejected, as
//! scipy's `LocalSearchWrapper.local_search` does.
//!
//! RNG is an MT19937-style seeded generator (via `rand`); the seed is fixed so
//! runs are reproducible, but the exact variate sequence does not match
//! numpy's `RandomState` bit-for-bit (numpy uses its own Ziggurat normals and
//! 53-bit uniforms), so results agree with scipy only to numerical tolerance,
//! not exactly.

use crate::raichu::lbfgsb::lbfgsb;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const INITIAL_TEMP: f64 = 5230.0;
const RESTART_TEMP_RATIO: f64 = 2e-5;
const VISIT: f64 = 2.62;
const ACCEPT: f64 = -5.0;
const TAIL_LIMIT: f64 = 1e8;
const MIN_VISIT_BOUND: f64 = 1e-10;
const MAX_REINIT_COUNT: usize = 1000;
const LBFGSB_MAXITER: usize = 15000;
const LBFGSB_GTOL: f64 = 1e-5;
const LBFGSB_FTOL: f64 = 2.22e-9;

/// C `fmod` (truncating remainder), matching `np.fmod`.
fn fmod(a: f64, b: f64) -> f64 {
    a - (a / b).trunc() * b
}

/// `scipy.special.gammaln` = `log(|gamma(x)|)` = C `lgamma`.
fn gammaln(x: f64) -> f64 {
    libm::lgamma(x)
}

/// Standard normal via Box–Muller (distribution-correct, not numpy's Ziggurat).
fn standard_normal(rng: &mut StdRng) -> f64 {
    let u1 = rng.random::<f64>().max(1e-300);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Distorted Cauchy-Lorentz visiting distribution (`VisitingDistribution`).
struct VisitingDistribution {
    qv: f64,
    lower: Vec<f64>,
    bound_range: Vec<f64>,
    factor4_p: f64,
    factor6: f64,
}

impl VisitingDistribution {
    fn new(lower: &[f64], upper: &[f64], qv: f64) -> Self {
        let bound_range: Vec<f64> = upper.iter().zip(lower).map(|(u, l)| u - l).collect();
        let factor2 = ((4.0 - qv) * (qv - 1.0).ln()).exp();
        let factor3 = ((2.0 - qv) * 2.0_f64.ln() / (qv - 1.0)).exp();
        let factor4_p = std::f64::consts::PI.sqrt() * factor2 / (factor3 * (3.0 - qv));
        let factor5 = 1.0 / (qv - 1.0) - 0.5;
        let d1 = 2.0 - factor5;
        let factor6 = std::f64::consts::PI * (1.0 - factor5)
            / (std::f64::consts::PI * (1.0 - factor5)).sin()
            / gammaln(d1).exp();
        VisitingDistribution {
            qv,
            lower: lower.to_vec(),
            bound_range,
            factor4_p,
            factor6,
        }
    }

    /// `visit_fn`: one distorted-Cauchy sample per coordinate. The `x`/`y`
    /// normals interleave exactly as numpy's `normal(size=(dim, 2)).T`.
    fn visit_fn(&self, rng: &mut StdRng, temperature: f64, dim: usize) -> Vec<f64> {
        let factor1 = (temperature.ln() / (self.qv - 1.0)).exp();
        let factor4 = self.factor4_p * factor1;
        let mut out = Vec::with_capacity(dim);
        for _ in 0..dim {
            let x = standard_normal(rng);
            let y = standard_normal(rng);
            let xx = x * (-(self.qv - 1.0) * (self.factor6 / factor4).ln() / (3.0 - self.qv)).exp();
            let den = ((self.qv - 1.0) * y.abs().ln() / (3.0 - self.qv)).exp();
            out.push(xx / den);
        }
        out
    }

    /// `visiting`: generate a new point, clipped and wrapped into bounds.
    fn visiting(&self, rng: &mut StdRng, x: &[f64], step: usize, temperature: f64) -> Vec<f64> {
        let dim = x.len();
        if step < dim {
            let mut visits = self.visit_fn(rng, temperature, dim);
            let upper_sample = rng.random::<f64>();
            let lower_sample = rng.random::<f64>();
            for v in visits.iter_mut() {
                if *v > TAIL_LIMIT {
                    *v = TAIL_LIMIT * upper_sample;
                } else if *v < -TAIL_LIMIT {
                    *v = -TAIL_LIMIT * lower_sample;
                }
            }
            let mut out = vec![0.0; dim];
            for i in 0..dim {
                out[i] = visits[i] + x[i];
                let a = out[i] - self.lower[i];
                let b = fmod(a, self.bound_range[i]) + self.bound_range[i];
                out[i] = fmod(b, self.bound_range[i]) + self.lower[i];
                if (out[i] - self.lower[i]).abs() < MIN_VISIT_BOUND {
                    out[i] += 1e-10;
                }
            }
            out
        } else {
            let mut out = x.to_vec();
            let index = step - dim;
            let mut visit = self.visit_fn(rng, temperature, 1)[0];
            if visit > TAIL_LIMIT {
                visit = TAIL_LIMIT * rng.random::<f64>();
            } else if visit < -TAIL_LIMIT {
                visit = -TAIL_LIMIT * rng.random::<f64>();
            }
            out[index] = visit + x[index];
            let a = out[index] - self.lower[index];
            let b = fmod(a, self.bound_range[index]) + self.bound_range[index];
            out[index] = fmod(b, self.bound_range[index]) + self.lower[index];
            if (out[index] - self.lower[index]).abs() < MIN_VISIT_BOUND {
                out[index] += MIN_VISIT_BOUND;
            }
            out
        }
    }
}

/// Result of a [`dual_annealing`] run.
pub struct DualAnnealingResult {
    pub x: Vec<f64>,
    pub fun: f64,
    pub nfev: u64,
    pub nit: usize,
}

/// The full dual-annealing state machine (energy state + strategy chain).
struct DualAnnealer<'f> {
    visit_dist: VisitingDistribution,
    lower: Vec<f64>,
    upper: Vec<f64>,
    rng: StdRng,
    f: &'f mut dyn FnMut(&[f64]) -> f64,
    nfev: u64,
    maxfun: u64,
    // energy state
    ebest: Option<f64>,
    xbest: Vec<f64>,
    current_energy: f64,
    current_location: Vec<f64>,
    // strategy chain
    emin: f64,
    xmin: Vec<f64>,
    not_improved_idx: usize,
    not_improved_max_idx: usize,
    k: usize,
    temperature_step: f64,
    energy_state_improved: bool,
}

impl<'f> DualAnnealer<'f> {
    fn call_f(&mut self, x: &[f64]) -> f64 {
        self.nfev += 1;
        (self.f)(x)
    }

    fn reset(&mut self, x0: Option<&[f64]>) {
        match x0 {
            None => {
                self.current_location = self
                    .lower
                    .iter()
                    .zip(&self.upper)
                    .map(|(l, u)| self.rng.random_range(*l..*u))
                    .collect();
            }
            Some(x0) => self.current_location = x0.to_vec(),
        }
        let mut reinit = 0usize;
        loop {
            let loc = self.current_location.clone();
            self.current_energy = self.call_f(&loc);
            if self.current_energy.is_nan() || !self.current_energy.is_finite() {
                reinit += 1;
                if reinit >= MAX_REINIT_COUNT {
                    panic!("dual annealing: objective NaN/inf after 1000 random starts");
                }
                self.current_location = self
                    .lower
                    .iter()
                    .zip(&self.upper)
                    .map(|(l, u)| self.rng.random_range(*l..*u))
                    .collect();
                continue;
            }
            break;
        }
        if self.ebest.is_none() {
            self.ebest = Some(self.current_energy);
            self.xbest = self.current_location.clone();
        }
    }

    fn accept_reject(&mut self, j: usize, e: f64, x_visit: &[f64]) {
        let r = self.rng.random::<f64>();
        let pqv_temp = 1.0 - (1.0 - ACCEPT) * (e - self.current_energy) / self.temperature_step;
        let pqv = if pqv_temp <= 0.0 {
            0.0
        } else {
            (pqv_temp.ln() / (1.0 - ACCEPT)).exp()
        };
        if r <= pqv {
            self.current_energy = e;
            self.current_location = x_visit.to_vec();
            self.xmin = self.current_location.clone();
        }
        if self.not_improved_idx >= self.not_improved_max_idx
            && (j == 0 || self.current_energy < self.emin)
        {
            self.emin = self.current_energy;
            self.xmin = self.current_location.clone();
        }
    }

    fn run(&mut self, step: usize, temperature: f64) -> Option<String> {
        self.temperature_step = temperature / (step as f64 + 1.0);
        self.not_improved_idx += 1;
        let dim = self.current_location.len();
        for j in 0..(2 * dim) {
            if j == 0 {
                self.energy_state_improved = step == 0;
            }
            let x_visit =
                self.visit_dist
                    .visiting(&mut self.rng, &self.current_location, j, temperature);
            let e = self.call_f(&x_visit);
            if e < self.current_energy {
                self.current_energy = e;
                self.current_location = x_visit.clone();
                if let Some(eb) = self.ebest {
                    if e < eb {
                        self.ebest = Some(e);
                        self.xbest = x_visit.clone();
                        self.energy_state_improved = true;
                        self.not_improved_idx = 0;
                    }
                }
            } else {
                self.accept_reject(j, e, &x_visit);
            }
            if self.nfev >= self.maxfun {
                return Some("Maximum number of function call reached during annealing".into());
            }
        }
        None
    }

    /// Local search with scipy's post-hoc validity/improvement gate: returns
    /// the minimizer result only if finite, in-bounds and better than `e0`.
    fn local_minimize(&mut self, x0: &[f64], e0: f64) -> (Vec<f64>, f64) {
        let (x, e) = lbfgsb(
            &mut |x: &[f64]| self.call_f(x),
            x0,
            LBFGSB_MAXITER,
            LBFGSB_GTOL,
            LBFGSB_FTOL,
        );
        let finite = e.is_finite() && x.iter().all(|v| v.is_finite());
        let in_bounds = x.iter().zip(&self.lower).all(|(v, l)| v >= l)
            && x.iter().zip(&self.upper).all(|(v, u)| v <= u);
        if finite && in_bounds && e < e0 {
            (x, e)
        } else {
            (x0.to_vec(), e0)
        }
    }

    fn local_search(&mut self) -> Option<String> {
        if self.energy_state_improved {
            if let Some(eb) = self.ebest {
                let (x, e) = self.local_minimize(&self.xbest.clone(), eb);
                if e < eb {
                    self.not_improved_idx = 0;
                    self.ebest = Some(e);
                    self.xbest = x.clone();
                    self.current_energy = e;
                    self.current_location = x;
                }
            }
            if self.nfev >= self.maxfun {
                return Some("Maximum number of function call reached during local search".into());
            }
        }
        let mut do_ls = false;
        let dim = self.current_location.len();
        if self.k < 90 * dim {
            if let Some(eb) = self.ebest {
                let pls =
                    (self.k as f64 * (eb - self.current_energy) / self.temperature_step).exp();
                if pls >= self.rng.random::<f64>() {
                    do_ls = true;
                }
            }
        }
        if self.not_improved_idx >= self.not_improved_max_idx {
            do_ls = true;
        }
        if do_ls {
            let (x, e) = self.local_minimize(&self.xmin.clone(), self.emin);
            self.xmin = x.clone();
            self.emin = e;
            self.not_improved_idx = 0;
            self.not_improved_max_idx = dim;
            if let Some(eb) = self.ebest {
                if e < eb {
                    self.ebest = Some(e);
                    self.xbest = self.xmin.clone();
                    self.current_energy = e;
                    self.current_location = x;
                }
            }
            if self.nfev >= self.maxfun {
                return Some(
                    "Maximum number of function call reached during dual annealing".into(),
                );
            }
        }
        None
    }
}

/// Minimize `f` over `[lower, upper]` via dual annealing, starting from `x0`.
pub fn dual_annealing(
    f: &mut dyn FnMut(&[f64]) -> f64,
    lower: &[f64],
    upper: &[f64],
    x0: &[f64],
    maxiter: usize,
    seed: u64,
) -> DualAnnealingResult {
    let mut da = DualAnnealer {
        visit_dist: VisitingDistribution::new(lower, upper, VISIT),
        lower: lower.to_vec(),
        upper: upper.to_vec(),
        rng: StdRng::seed_from_u64(seed),
        f,
        nfev: 0,
        maxfun: 10_000_000,
        ebest: None,
        xbest: Vec::new(),
        current_energy: 0.0,
        current_location: Vec::new(),
        emin: 0.0,
        xmin: Vec::new(),
        not_improved_idx: 0,
        not_improved_max_idx: 1000,
        k: 100 * lower.len(),
        temperature_step: 0.0,
        energy_state_improved: false,
    };
    da.reset(Some(x0));
    da.emin = da.current_energy;
    da.xmin = da.current_location.clone();

    let t1 = ((VISIT - 1.0) * 2.0_f64.ln()).exp() - 1.0;
    let temperature_restart = INITIAL_TEMP * RESTART_TEMP_RATIO;

    let mut iteration = 0usize;
    let mut need_to_stop = false;
    while !need_to_stop {
        for i in 0..maxiter {
            let s = i as f64 + 2.0;
            let t2 = ((VISIT - 1.0) * s.ln()).exp() - 1.0;
            let temperature = INITIAL_TEMP * t1 / t2;
            if iteration >= maxiter {
                need_to_stop = true;
                break;
            }
            if temperature < temperature_restart {
                da.reset(None);
                break;
            }
            if da.run(i, temperature).is_some() {
                need_to_stop = true;
                break;
            }
            if da.local_search().is_some() {
                need_to_stop = true;
                break;
            }
            iteration += 1;
        }
    }

    DualAnnealingResult {
        x: da.xbest.clone(),
        fun: da.ebest.unwrap_or(da.current_energy),
        nfev: da.nfev,
        nit: iteration,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_quadratic_minimum() {
        // f(x) = (x0-1)^2 + (x1+2)^2, bounds [-10,10]^2, min at (1,-2).
        let mut f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] + 2.0).powi(2);
        let r = dual_annealing(&mut f, &[-10.0, -10.0], &[10.0, 10.0], &[0.0, 0.0], 50, 42);
        assert!((r.x[0] - 1.0).abs() < 0.1, "x0 = {}", r.x[0]);
        assert!((r.x[1] + 2.0).abs() < 0.1, "x1 = {}", r.x[1]);
        assert!(r.fun < 0.1, "fun = {}", r.fun);
    }
}
