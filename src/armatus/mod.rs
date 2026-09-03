//! Rust port of Armatus 2.3 (Filippova, Patro, Duggal & Kingsford, 2014):
//! multiresolution domain calling via a dynamic program over a dense Hi-C
//! matrix, plus consensus extraction by weighted interval scheduling.
//!
//! The pipeline mirrors `multiscaleDomains` / `consensusDomains` /
//! `IntervalScheduler` in `ArmatusUtil.cpp`, `ArmatusDAG.cpp` (the DP with
//! top-K near-optimal solutions) and `ArmatusParams.cpp` (prefix sums and
//! per-size means). Input is an already-transformed dense `n x n` matrix —
//! the caller decides whether to log-transform the counts (Armatus's sparse /
//! Rao parsers apply `log(count)`; its Dixon dense parser does not).

use std::collections::BTreeMap;

use ndarray::Array2;

pub mod dag;
pub mod sums;

pub use dag::ArmatusDag;
pub use sums::Sums;

/// Parameters controlling the multiresolution sweep (defaults match Armatus).
#[derive(Debug, Clone, Copy)]
pub struct Params {
    /// Highest gamma to generate domains at (`-g`).
    pub gamma_max: f64,
    /// Step size between resolutions (`-s`).
    pub step_size: f64,
    /// Number of near-optimal solutions per resolution (`-k`).
    pub top_k: usize,
    /// Minimum samples required to compute a per-size mean (`-n`).
    pub min_mean_samples: usize,
    /// Only generate domains at `gamma_max` (`-j`).
    pub just_gamma_max: bool,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            gamma_max: 0.5,
            step_size: 0.05,
            top_k: 1,
            min_mean_samples: 100,
            just_gamma_max: false,
        }
    }
}

/// A domain, `[start, end]` inclusive in 0-based bin indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Domain {
    pub start: usize,
    pub end: usize,
}

/// The ensemble of domain sets (one per resolution/rank) and their weights.
#[derive(Debug, Default)]
pub struct WeightedDomainEnsemble {
    pub domain_sets: Vec<Vec<Domain>>,
    pub weights: Vec<f64>,
    /// The `gamma` each domain set was generated at (one per domain set).
    pub resolutions: Vec<f64>,
}

// --- Interval scheduling (consensus) ---

#[derive(Debug, Clone, Copy)]
struct WeightedInterval {
    start: usize,
    end: usize,
    score: f64,
}

impl WeightedInterval {
    fn new(s: usize, e: usize, sc: f64) -> Self {
        let (start, end) = if s < e { (s, e) } else { (e, s) };
        WeightedInterval { start, end, score: sc }
    }
}

struct IntervalScheduler {
    ivals: Vec<WeightedInterval>,
    best: Vec<f64>,
    p: Vec<usize>,
}

impl IntervalScheduler {
    fn new(input: Vec<WeightedInterval>) -> Self {
        let n = input.len();
        let mut sorted = input;
        sorted.sort_by_key(|iv| iv.end);

        let mut ivals = Vec::with_capacity(n + 1);
        ivals.push(WeightedInterval::new(0, 0, -1.0)); // sentinel at index 0
        ivals.extend(sorted);

        let mut p = vec![0usize; n + 1];
        for (j, slot) in p.iter_mut().enumerate().skip(1) {
            *slot = Self::previous_disjoint(&ivals, j);
        }

        IntervalScheduler {
            ivals,
            best: vec![0.0; n + 1],
            p,
        }
    }

    /// Largest index `i` such that `ivals[i].end < ivals[j].start`.
    fn previous_disjoint(ivals: &[WeightedInterval], j: usize) -> usize {
        let mut i = j;
        while i > 0 {
            if ivals[i].end < ivals[j].start {
                return i;
            }
            i -= 1;
        }
        0
    }

    fn compute_schedule(&mut self) {
        let n = self.ivals.len() - 1;
        for j in 1..=n {
            let chosen = self.ivals[j].score + self.best[self.p[j]];
            let ignored = self.best[j - 1];
            self.best[j] = chosen.max(ignored);
        }
    }

    fn extract(&self) -> Vec<WeightedInterval> {
        let mut out = Vec::new();
        let mut j = self.best.len() - 1;
        while j > 0 {
            if self.best[j] != self.best[j - 1] {
                out.push(self.ivals[j]);
                j = self.p[j];
            } else {
                j -= 1;
            }
        }
        out
    }
}

// --- Consensus ---

/// Weight each domain by its persistence across the ensemble, then extract the
/// maximum-weight set of non-overlapping domains (`consensusDomains`).
pub fn consensus_domains(ensemble: &WeightedDomainEnsemble) -> Vec<Domain> {
    let mut pmap: BTreeMap<Domain, f64> = BTreeMap::new();
    for (domain_set, weight) in ensemble.domain_sets.iter().zip(&ensemble.weights) {
        for domain in domain_set {
            *pmap.entry(*domain).or_insert(0.0) += weight;
        }
    }

    let mut scheduler = IntervalScheduler::new(
        pmap.into_iter()
            .map(|(d, persistence)| WeightedInterval::new(d.start, d.end, persistence))
            .collect(),
    );
    scheduler.compute_schedule();

    let mut dset: Vec<Domain> = scheduler
        .extract()
        .into_iter()
        .map(|iv| Domain {
            start: iv.start,
            end: iv.end,
        })
        .collect();
    dset.sort();
    dset
}

// --- Multiresolution sweep ---

/// Run the DP at each resolution from 0 to `gamma_max` (or just `gamma_max`),
/// collecting the top-K domain sets per resolution (`multiscaleDomains`).
pub fn multiscale_domains(matrix: &Array2<f64>, params: &Params) -> WeightedDomainEnsemble {
    let mut ensemble = WeightedDomainEnsemble::default();
    let mut gamma = if params.just_gamma_max {
        params.gamma_max
    } else {
        0.0
    };

    loop {
        if gamma > params.gamma_max + 1e-5 {
            break;
        }
        log::debug!("gamma={gamma}");

        let sums = Sums::compute(matrix, gamma, params.min_mean_samples);
        let mut dag = ArmatusDag::new(&sums, params.top_k);
        dag.build();
        dag.compute_top_k();

        let top_k = dag.extract_top_k();
        ensemble.domain_sets.extend(top_k.domain_sets);
        ensemble.weights.extend(top_k.weights);
        ensemble
            .resolutions
            .extend(std::iter::repeat_n(gamma, params.top_k));

        gamma += params.step_size;
    }

    ensemble
}

/// Call consensus domains from a dense (already-transformed) matrix.
pub fn call_domains(matrix: &Array2<f64>, params: &Params) -> Vec<Domain> {
    consensus_domains(&multiscale_domains(matrix, params))
}
