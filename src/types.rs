//! Core data types shared by readers and writers.

use crate::error::Result;

/// A chromosome (or contig) entry from the `/chroms` table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chrom {
    /// Chromosome name, e.g. `"chr1"`.
    pub name: String,
    /// Chromosome length in base pairs.
    pub length: i32,
}

/// A genomic bin from the `/bins` table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bin {
    /// Index of the chromosome this bin belongs to (into the chroms table).
    pub chrom_id: i32,
    /// Start coordinate (0-based, inclusive).
    pub start: i32,
    /// End coordinate (0-based, exclusive).
    pub end: i32,
}

/// A non-zero matrix entry from the `/pixels` table.
///
/// Pixels are stored in "symmetric-upper" mode: `bin1_id <= bin2_id`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pixel {
    /// Row bin index.
    pub bin1_id: i64,
    /// Column bin index.
    pub bin2_id: i64,
    /// Contact count (integer-valued for observed data, float for balanced).
    pub count: f64,
}

/// Metadata about a chromosome extracted from a cooler file.
#[derive(Debug, Clone)]
pub struct ChromMeta {
    /// Chromosome name.
    pub name: String,
    /// Chromosome length in base pairs.
    pub length: u64,
    /// Bin size in base pairs.
    pub resolution: u64,
}

/// Per-bin weights for normalizing a contact matrix.
///
/// `divisive` selects how [`Weights::apply`] scales a pixel's count:
/// `count / (w_i * w_j)` when true, `count * w_i * w_j` when false.
/// `.hic` normalization vectors are divisive; cooler `bins/weight` columns
/// (IC/ICE) are multiplicative.
#[derive(Debug, Clone, PartialEq)]
pub struct Weights {
    /// Weight per global bin id.
    pub values: Vec<f64>,
    /// True: divide by the weight product; false: multiply.
    pub divisive: bool,
}

impl Weights {
    /// Number of bins covered.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// True when there are no weights.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Scale the weights so the finite entries sum to `target_sum`.
    pub fn rescale(&self, target_sum: f64) -> Weights {
        let s: f64 = self.values.iter().filter(|v| v.is_finite()).sum();
        let factor = target_sum / s;
        Weights {
            values: self.values.iter().map(|v| v * factor).collect(),
            divisive: self.divisive,
        }
    }

    /// Apply the weights to one pixel's count, by the `divisive` flag.
    pub fn apply(&self, p: &Pixel) -> f64 {
        let w = self.values[p.bin1_id as usize] * self.values[p.bin2_id as usize];
        if self.divisive {
            p.count / w
        } else {
            p.count * w
        }
    }
}

/// Common pixel + index read surface consumed by the iterative-correction
/// core. Implemented by [`crate::Cooler`] and [`crate::File`] so balancing
/// works on both `.cool` and `.hic` inputs.
pub(crate) trait MatrixSource {
    fn n_pixels(&self) -> Result<u64>;
    fn n_bins(&self) -> Result<usize>;
    fn bin_chrom(&self) -> Result<Vec<i32>>;
    fn chrom_offset(&self) -> Result<Vec<i64>>;
    fn bin1_offset(&self) -> Result<Vec<i64>>;
    /// Read pixel rows `[lo, hi)` in stored order (a chunk of a full pass).
    fn pixels_range(&self, lo: i64, hi: i64) -> Result<Vec<Pixel>>;
}
