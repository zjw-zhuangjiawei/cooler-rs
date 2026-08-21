//! Probabilistic models ported from `pomegranate` 0.10.0 — the statistical
//! core that `domaincaller`'s TAD calling is built on: a 1-D Gaussian, a
//! discrete (categorical) distribution, a Gaussian mixture model, and a
//! Hidden Markov Model with mixture emissions trained by multi-sequence
//! Baum-Welch.
//!
//! Ported faithfully so the hard-coded numeric oracles in
//! `pomegranate/tests/{test_hmm,test_gmm,test_distributions}.py` hold
//! verbatim (see `tests/stats.rs`).

pub mod discrete;
pub mod gmm;
pub mod hmm;
pub mod normal;

pub use discrete::DiscreteDistribution;
pub use gmm::GeneralMixtureModel;
pub use hmm::{HiddenMarkovModel, State, END, START};
pub use normal::{Emission, NormalDistribution, NEG_INF, SQRT_2_PI};
