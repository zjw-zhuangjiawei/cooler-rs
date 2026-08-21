//! Chromosome-level TAD calling, ported from TADLib's `domaincaller`
//! (Dixon et al., Nature 2012): adaptive DI windows, gap-free region
//! splitting, and a 4-state Gaussian-mixture HMM decoded by Viterbi.
//!
//! - [`chrom`] holds the pipeline itself (`tadlib/domaincaller/chromLev.py`
//!   plus the HMM setup in `tadlib/hitad/genomeLev.py`).
//! - [`aligner`] is the hierarchical domain aligner
//!   (`tadlib/hitad/aligner.py`), used only to compute the mismatch ratio
//!   that lets the window-refinement loop exit early.
//!
//! The probabilistic models the pipeline trains on live in [`crate::stats`]
//! (a `pomegranate` 0.10.0 port); both are faithful ports, verified against
//! TADLib's exact output in `tests/domaincaller.rs`.

pub mod aligner;
pub mod chrom;

pub use chrom::Chrom;
