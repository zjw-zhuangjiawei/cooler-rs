//! Rust port of HiCExplorer's `hicFindTADs`.
//!
//! `hicFindTADs` scores every bin with the mean z-score of the interactions
//! that cross it (the "TAD-separation score"), computed over a range of
//! window sizes, and calls a TAD boundary at each local minimum of the mean
//! score that clears a depth and a significance filter.
//!
//! The pipeline is the one in `hicexplorer/hicFindTADs.py`:
//!
//! 1. drop bins with no usable weight and close the gaps ([`matrix`]),
//! 2. z-score the matrix per chromosome ([`matrix::Band::zscore`]),
//! 3. score every bin at every window ([`matrix::tad_separation_scores`]),
//! 4. find the minima and filter them ([`call`]),
//! 5. write the tables ([`write`]).
//!
//! Upstream reads a `.h5` written by `hicCorrectMatrix`; here the input is a
//! `.cool`/`.mcool` and the correction comes from a `bins` weight column
//! (`--norm`), applied the way `File::fetch` applies it.

mod call;
mod matrix;
mod write;

pub use call::{find_consensus_minima, peakdetect, ranksums, ScoreRow, ScoreTable};
pub use matrix::{bin_size, enlarge_bins, Band, ChromBins};
pub use write::{write_outputs, Outputs};

use std::collections::HashMap;

use rayon::prelude::*;

use crate::cooler::Cooler;
use crate::error::{Error, Result};
use crate::file::File;

use matrix::tad_separation_scores;

/// Pixels read per chunk while the diagonal bands are filled. A pixel is 24
/// bytes, so this caps that pass at roughly 120 MB of pixel buffer regardless
/// of how large the matrix is.
const PIXEL_CHUNK: i64 = 5_000_000;

/// Which multiple-testing correction to apply to the boundary p-values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultipleTesting {
    /// Benjamini-Hochberg false discovery rate (`--correctForMultipleTesting fdr`).
    Fdr,
    /// Bonferroni family-wise error rate.
    Bonferroni,
    /// Raw p-values, no correction.
    None,
}

/// Parameters for one `find-tads` run. Depths are in bp; `None` means "use the
/// bin-size-dependent default", as upstream does.
#[derive(Debug, Clone)]
pub struct Params {
    /// Minimum window length to each side of a bin (bp).
    pub min_depth: Option<i64>,
    /// Maximum window length to each side of a bin (bp).
    pub max_depth: Option<i64>,
    /// First step between window lengths (bp); later steps grow as
    /// `step * x**1.5`.
    pub step: Option<i64>,
    /// Minimum drop from the surrounding mean score for a minimum to count.
    pub delta: f64,
    /// Minimum distance between boundaries (bp). Defaults to four bins.
    pub min_boundary_distance: Option<i64>,
    /// Multiple-testing correction.
    pub correction: MultipleTesting,
    /// p-value (Bonferroni) or q-value (FDR) threshold.
    pub threshold_comparisons: f64,
    /// `bins` column holding the correction weights, if any.
    pub norm: Option<String>,
    /// Chromosomes to analyse, in this order. `None` means all of them.
    pub chromosomes: Option<Vec<String>>,
    /// Prefix for the output files.
    pub out_prefix: String,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            min_depth: None,
            max_depth: None,
            step: None,
            delta: 0.01,
            min_boundary_distance: None,
            correction: MultipleTesting::Fdr,
            threshold_comparisons: 0.01,
            norm: None,
            chromosomes: None,
            out_prefix: "TADs".into(),
        }
    }
}

/// Depths and window sizes resolved against the matrix bin size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Depths {
    /// Bin size in bp.
    pub binsize: i64,
    /// Minimum window (bp).
    pub min_depth: i64,
    /// Maximum window (bp).
    pub max_depth: i64,
    /// First step (bp).
    pub step: i64,
    /// Window lengths in bp, in increasing order.
    pub windows: Vec<i64>,
    /// Diagonals the z-score matrix is built from.
    pub zscore_depth: usize,
    /// Diagonals kept for the TAD-separation score.
    pub band_limit: usize,
}

/// `get_incremental_step_size`: `min + step * x**1.5` for `x = 0, 1, ...`,
/// skipping repeats and stopping once `max` is passed.
pub fn incremental_step_size(min: i64, max: i64, step: i64) -> Vec<i64> {
    let mut out: Vec<i64> = Vec::new();
    if step <= 0 {
        return out;
    }
    let mut x: i64 = -1;
    loop {
        x += 1;
        let inc = min + (step as f64 * (x as f64).powf(1.5)) as i64;
        if x > 1 && out.last() == Some(&inc) {
            continue;
        }
        if inc > max {
            break;
        }
        out.push(inc);
    }
    out
}

/// `HicFindTads.set_variables`: resolve the defaults and apply the range
/// checks upstream turns into `exit(1)`.
pub fn resolve_depths(
    binsize: i64,
    min_depth: Option<i64>,
    max_depth: Option<i64>,
    step: Option<i64>,
) -> Result<Depths> {
    if binsize <= 0 {
        return Err(Error::InvalidInput(
            "could not determine the matrix bin size".into(),
        ));
    }

    let max_depth = match max_depth {
        None if binsize < 1000 => binsize * 60,
        None if binsize < 20000 => binsize * 40,
        None => binsize * 10,
        Some(value) if value < binsize * 5 => {
            return Err(Error::InvalidInput(
                "maxDepth must be at least 5 times the matrix bin size".into(),
            ))
        }
        Some(value) => value,
    };

    let min_depth = match min_depth {
        None if binsize < 1000 => binsize * 30,
        None if binsize < 20000 => binsize * 10,
        None => binsize * 5,
        Some(value) if value < binsize * 3 => {
            return Err(Error::InvalidInput(
                "minDepth must be at least 3 times the matrix bin size".into(),
            ))
        }
        Some(value) => value,
    };

    let step = match step {
        None if binsize < 1000 => binsize * 4,
        None => binsize * 2,
        Some(value) if value < binsize => {
            return Err(Error::InvalidInput(
                "step must be at least the matrix bin size".into(),
            ))
        }
        Some(value) => value,
    };

    if max_depth <= min_depth {
        return Err(Error::InvalidInput(
            "maxDepth must be larger than minDepth".into(),
        ));
    }

    let min_depth_in_bins = min_depth / binsize;
    let max_depth_in_bins = max_depth / binsize;
    if min_depth_in_bins <= 1 {
        return Err(Error::InvalidInput(format!(
            "minDepth is too small; use at least twice the bin size ({binsize})"
        )));
    }
    if max_depth_in_bins <= 1 {
        return Err(Error::InvalidInput(format!(
            "maxDepth is too small; use more than the bin size ({binsize})"
        )));
    }

    Ok(Depths {
        binsize,
        min_depth,
        max_depth,
        step,
        windows: incremental_step_size(min_depth, max_depth, step),
        zscore_depth: (max_depth as f64 * 2.5 * 1.5 / binsize as f64) as usize,
        band_limit: 2 * max_depth_in_bins as usize,
    })
}

/// A cooler prepared for scoring: the bins that survived masking, grouped by
/// chromosome, and one z-scored band each.
pub struct Prepared {
    /// Bins per chromosome, in the order the matrix lists them.
    pub chrom_bins: Vec<ChromBins>,
    /// Chromosome lengths, one per entry of `chrom_bins`.
    pub chrom_lengths: Vec<i32>,
    /// Z-scored band per chromosome.
    pub bands: Vec<Band>,
    /// Resolved depths.
    pub depths: Depths,
}

/// Read the cooler, drop unusable bins and build the z-scored bands.
pub fn prepare(cooler: &Cooler, params: &Params) -> Result<Prepared> {
    let weights = load_weights(cooler, params.norm.as_deref())?;

    // `getBinSize` runs before masking and caches its answer.
    let unmasked = group_bins(cooler, params.chromosomes.as_deref(), None)?.0;
    let binsize = bin_size(&unmasked);
    let depths = resolve_depths(binsize, params.min_depth, params.max_depth, params.step)?;

    let (mut chrom_bins, chrom_lengths, slot_of) =
        group_bins(cooler, params.chromosomes.as_deref(), weights.as_deref())?;

    let mut bands: Vec<Band> = chrom_bins
        .iter()
        .map(|c| Band::new(c.len(), depths.zscore_depth))
        .collect();

    // Only the band survives, and that is a small fraction of the pixel
    // table, so read it in chunks rather than materializing every pixel of
    // the matrix at once.
    for chunk in cooler.pixels_chunked(PIXEL_CHUNK)? {
        for pixel in chunk?.iter() {
            let (Some((slot_a, a)), Some((slot_b, b))) = (
                slot_of.get(pixel.bin1_id as usize).copied().flatten(),
                slot_of.get(pixel.bin2_id as usize).copied().flatten(),
            ) else {
                continue;
            };
            if slot_a != slot_b {
                continue;
            }
            let (a, b) = (a.min(b), a.max(b));
            let d = b - a;
            if d == 0 {
                continue; // `diagflat(value=0)` zeroes the diagonal
            }
            if d < depths.zscore_depth {
                let count = match weights.as_deref() {
                    Some(w) => {
                        let (wa, wb) = (w[pixel.bin1_id as usize], w[pixel.bin2_id as usize]);
                        pixel.count * wa * wb
                    }
                    None => pixel.count,
                };
                bands[slot_a].set(a, d, count);
            }
        }
    }

    let band_limit = depths.band_limit;
    bands.par_iter_mut().for_each(|band| {
        band.zscore();
        band.truncate(band_limit);
    });

    enlarge_bins(&mut chrom_bins);

    Ok(Prepared {
        chrom_bins,
        chrom_lengths,
        bands,
        depths,
    })
}

/// The TAD-separation score table of a prepared matrix. Chromosomes are
/// scored in parallel; the rows come back in matrix order.
pub fn score_table(prepared: &Prepared) -> ScoreTable {
    let windows = &prepared.depths.windows;
    let chunks: Vec<Vec<ScoreRow>> = prepared
        .chrom_bins
        .par_iter()
        .zip(prepared.bands.par_iter())
        .map(|(chrom, band)| {
            tad_separation_scores(band, chrom, windows)
                .into_iter()
                .map(|(cut, values)| ScoreRow {
                    chrom: chrom.name.clone(),
                    bin: cut,
                    start: chrom.start[cut],
                    end: chrom.end[cut],
                    values,
                })
                .collect()
        })
        .collect();
    ScoreTable {
        rows: chunks.into_iter().flatten().collect(),
    }
}

/// Run the whole `hicFindTADs` pipeline on an open `.cool`/`.mcool` file.
pub fn run(input: &File, params: &Params) -> Result<Outputs> {
    let File::Cooler(cooler) = input else {
        return Err(Error::InvalidInput(
            "find-tads needs a .cool or .mcool input".into(),
        ));
    };
    run_cooler(cooler, params)
}

/// Run the whole `hicFindTADs` pipeline on a single-resolution cooler.
pub fn run_cooler(cooler: &Cooler, params: &Params) -> Result<Outputs> {
    let prepared = prepare(cooler, params)?;
    let table = score_table(&prepared);
    if table.is_empty() {
        return Err(Error::InvalidInput(
            "no bin produced a TAD-separation score".into(),
        ));
    }
    let boundaries = find_boundaries(&table, &prepared, params)?;
    write_outputs(&table, &boundaries, &prepared, params)
}

/// Weights as a multiplicative factor per bin. A `divisive_weights` column is
/// inverted once here, so the pixel loop always multiplies.
fn load_weights(cooler: &Cooler, norm: Option<&str>) -> Result<Option<Vec<f64>>> {
    let Some(name) = norm else { return Ok(None) };
    let values = cooler
        .bins_column_f64(name)?
        .ok_or_else(|| Error::InvalidInput(format!("no 'bins/{name}' column")))?;
    let divisor = cooler
        .bins_column_weight_type(name)?
        .is_some_and(|t| t.is_divisive());
    Ok(Some(if divisor {
        values.iter().map(|w| 1.0 / w).collect()
    } else {
        values
    }))
}

/// Group the kept bins by chromosome and record where each cooler bin landed.
///
/// Returns the per-chromosome bins, the chromosome lengths, and a
/// `bin_id -> (chromosome slot, position)` map for the pixels.
#[allow(clippy::type_complexity)]
fn group_bins(
    cooler: &Cooler,
    chromosomes: Option<&[String]>,
    weights: Option<&[f64]>,
) -> Result<(Vec<ChromBins>, Vec<i32>, Vec<Option<(usize, usize)>>)> {
    let all_bins = cooler.bins()?;
    let chroms = cooler.chroms()?;

    let keep: Vec<usize> = match chromosomes {
        None => (0..chroms.len()).collect(),
        Some(names) => {
            let mut keep = Vec::new();
            for name in names {
                match chroms.iter().position(|c| &c.name == name) {
                    Some(id) => keep.push(id),
                    None => log::warn!("chromosome '{name}' is not in the matrix"),
                }
            }
            keep
        }
    };

    let mut chrom_bins: Vec<ChromBins> = keep
        .iter()
        .map(|&id| ChromBins {
            name: chroms[id].name.clone(),
            start: Vec::new(),
            end: Vec::new(),
        })
        .collect();
    let chrom_lengths: Vec<i32> = keep.iter().map(|&id| chroms[id].length).collect();

    // A bin with no usable weight is one upstream drops as a `nan_bin`.
    let mut slot_of: Vec<Option<(usize, usize)>> = vec![None; all_bins.len()];
    for (bin_id, bin) in all_bins.iter().enumerate() {
        let Some(slot) = keep.iter().position(|&id| id as i32 == bin.chrom_id) else {
            continue;
        };
        if weights.is_some_and(|w| !w[bin_id].is_finite()) {
            continue;
        }
        slot_of[bin_id] = Some((slot, chrom_bins[slot].len()));
        chrom_bins[slot].start.push(bin.start as i64);
        chrom_bins[slot].end.push(bin.end as i64);
    }

    Ok((chrom_bins, chrom_lengths, slot_of))
}

/// The boundary set produced by one run.
#[derive(Debug, Clone, Default)]
pub struct Boundaries {
    /// Table row indices of the boundaries that passed every filter, in
    /// ascending order.
    pub min_idx: Vec<usize>,
    /// Row indices that a boundary may never occupy: the last bin before each
    /// chromosome start.
    pub chrom_end_idx: Vec<usize>,
    /// Local delta of each consensus minimum.
    pub delta: HashMap<usize, f64>,
    /// p-value (or q-value) of each consensus minimum.
    pub pvalues: HashMap<usize, f64>,
    /// FDR cutoff the filter used.
    pub fdr_cutoff: f64,
}

/// `HicFindTads.find_boundaries`.
pub fn find_boundaries(
    table: &ScoreTable,
    prepared: &Prepared,
    params: &Params,
) -> Result<Boundaries> {
    let mut sizes: Vec<f64> = table
        .rows
        .iter()
        .map(|r| (r.end - r.start) as f64)
        .collect();
    sizes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg_bin_size = median(&sizes);
    if avg_bin_size <= 0.0 {
        return Err(Error::InvalidInput(
            "the returned bins have no width".into(),
        ));
    }

    let min_boundary_distance = params
        .min_boundary_distance
        .map(|v| v as f64)
        .unwrap_or(avg_bin_size * 4.0);
    let lookahead = (min_boundary_distance / avg_bin_size) as usize;
    if lookahead < 1 {
        return Err(Error::InvalidInput(
            "minBoundaryDistance must be at least one bin".into(),
        ));
    }

    let (minima, delta_of_min) = find_consensus_minima(table, lookahead);
    if minima.is_empty() {
        return Err(Error::InvalidInput(format!(
            "no boundaries were found; check delta ({}) and minBoundaryDistance ({min_boundary_distance})",
            params.delta
        )));
    }
    if minima.len() <= 10 {
        log::info!(
            "only {} boundaries found; check delta ({}) and minBoundaryDistance ({min_boundary_distance})",
            minima.len(),
            params.delta
        );
    }

    let (_, pvalues, fdr_cutoff) = call::min_pvalues(
        table,
        prepared,
        &minima,
        prepared.depths.min_depth,
        params.correction,
        params.threshold_comparisons,
    );

    let (min_idx, chrom_end_idx) = call::filter_boundaries(
        table,
        &minima,
        &delta_of_min,
        &pvalues,
        call::Thresholds {
            correction: params.correction,
            threshold: params.threshold_comparisons,
            fdr_cutoff,
            delta: params.delta,
        },
    );

    Ok(Boundaries {
        min_idx,
        chrom_end_idx,
        delta: delta_of_min,
        pvalues,
        fdr_cutoff,
    })
}

/// The median of a sorted slice, as `np.median` computes it.
pub(crate) fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}
