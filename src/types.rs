//! Core data types shared by readers and writers.

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
