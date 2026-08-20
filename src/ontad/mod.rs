//! Rust port of the OnTAD algorithm (An et al., Genome Biology 2019),
//! translated from the original C++ implementation (<https://github.com/anlin00007/OnTAD>,
//! `step1`–`step4`), keeping the same numeric behavior.
//!
//! OnTAD calls hierarchical TADs from a Hi-C contact matrix in four steps:
//! 1. load the matrix (only a band around the diagonal is used),
//! 2. compute a "corner score" and find local minima (candidate boundaries),
//! 3. remove the distance effect by standardizing each diagonal,
//! 4. assemble boundaries into nested TADs with a dynamic program.

mod algorithm;
mod output;

pub use algorithm::{cal_mins, call_tads, cumsum, get_score, hicnorm, matrix_from_cooler, set_pair};
pub use output::{write_bed, write_tad};

/// Parameters controlling TAD calling (defaults match OnTAD v1.4).
#[derive(Debug, Clone, Copy)]
pub struct Params {
    /// Maximum TAD size in bins (`-maxsz`, default 200).
    pub maxsz: usize,
    /// Minimum TAD size in bins (`-minsz`, default 3).
    pub minsz: usize,
    /// Penalty for adding a TAD (`-penalty`, default 0.1).
    pub penalty: f64,
    /// Window half-size for local-minimum detection (`-lsize`, default 5).
    pub lsize: usize,
    /// Threshold in standard deviations for local minima (`-ldiff`, default 1.96).
    pub ldiff: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            maxsz: 200,
            minsz: 3,
            penalty: 0.1,
            lsize: 5,
            ldiff: 1.96,
        }
    }
}

/// A hierarchical TAD call set, mirroring the C++ `TAD` struct.
#[derive(Debug, Clone, Default)]
pub struct Tad {
    /// 0-based inclusive bin boundaries `[start, end]`.
    pub bound: Vec<[usize; 2]>,
    /// Nesting level (0 = whole chromosome, 1 = top-level TAD, ...).
    pub level: Vec<usize>,
    /// Mean contact frequency inside each TAD.
    pub mean: Vec<f64>,
    /// DP score of each TAD.
    pub score: Vec<f64>,
}

impl Tad {
    /// Number of called TADs.
    pub fn len(&self) -> usize {
        self.bound.len()
    }

    /// Whether no TADs were called.
    pub fn is_empty(&self) -> bool {
        self.bound.is_empty()
    }
}
