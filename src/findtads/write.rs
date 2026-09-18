//! The output files of `hicFindTADs`, with the same names and formats.
//!
//! `<prefix>_tad_score.bm` (the TAD-separation score matrix, one column per
//! window), `<prefix>_zscore_matrix.cool`, `<prefix>_boundaries.bed`,
//! `<prefix>_boundaries.gff`, `<prefix>_domains.bed` and
//! `<prefix>_score.bedgraph`.

use std::fmt::Write as _;
use std::path::PathBuf;

use crate::cooler::CoolerWriter;
use crate::error::Result;
use crate::types::{Bin, Chrom, Pixel};

use super::{Boundaries, Params, Prepared, ScoreTable};

/// Paths written by one run.
#[derive(Debug, Clone)]
pub struct Outputs {
    /// `<prefix>_tad_score.bm`.
    pub tad_score: PathBuf,
    /// `<prefix>_zscore_matrix.cool`.
    pub zscore_matrix: PathBuf,
    /// `<prefix>_boundaries.bed`.
    pub boundaries: PathBuf,
    /// `<prefix>_boundaries.gff`.
    pub boundaries_gff: PathBuf,
    /// `<prefix>_domains.bed`.
    pub domains: PathBuf,
    /// `<prefix>_score.bedgraph`.
    pub score_bedgraph: PathBuf,
    /// Number of boundaries written.
    pub n_boundaries: usize,
    /// Number of domains written.
    pub n_domains: usize,
}

/// Write every output file for a run.
pub fn write_outputs(
    table: &ScoreTable,
    boundaries: &Boundaries,
    prepared: &Prepared,
    params: &Params,
) -> Result<Outputs> {
    let prefix = &params.out_prefix;
    let tad_score = PathBuf::from(format!("{prefix}_tad_score.bm"));
    let zscore_matrix = PathBuf::from(format!("{prefix}_zscore_matrix.cool"));
    let boundaries_path = PathBuf::from(format!("{prefix}_boundaries.bed"));
    let boundaries_gff = PathBuf::from(format!("{prefix}_boundaries.gff"));
    let domains = PathBuf::from(format!("{prefix}_domains.bed"));
    let score_bedgraph = PathBuf::from(format!("{prefix}_score.bedgraph"));

    write_tad_score(&tad_score, table, prepared, params)?;
    write_zscore_matrix(&zscore_matrix, prepared)?;
    let (n_boundaries, n_domains) = write_boundaries(
        &boundaries_path,
        &boundaries_gff,
        &domains,
        table,
        boundaries,
        params,
    )?;
    write_score_bedgraph(&score_bedgraph, table)?;

    Ok(Outputs {
        tad_score,
        zscore_matrix,
        boundaries: boundaries_path,
        boundaries_gff,
        domains,
        score_bedgraph,
        n_boundaries,
        n_domains,
    })
}

/// `<prefix>_tad_score.bm`: a `#`-prefixed JSON header with the parameters,
/// then one line per scored bin.
fn write_tad_score(
    path: &PathBuf,
    table: &ScoreTable,
    prepared: &Prepared,
    params: &Params,
) -> Result<()> {
    let depths = &prepared.depths;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "#{{\"step\":{},\"minDepth\":{},\"maxDepth\":{},\"binsize\":{}}}",
        depths.step, depths.min_depth, depths.max_depth, depths.binsize
    );
    let _ = params;
    for row in &table.rows {
        let _ = write!(out, "{}\t{}\t{}", row.chrom, row.start, row.end);
        for value in &row.values {
            let _ = write!(out, "\t{value:.6}");
        }
        out.push('\n');
    }
    std::fs::write(path, out)?;
    Ok(())
}

/// `<prefix>_score.bedgraph`: the mean score, spanning from the previous bin's
/// midpoint to this one's.
fn write_score_bedgraph(path: &PathBuf, table: &ScoreTable) -> Result<()> {
    let means = table.row_means();
    let mut out = String::new();
    for (idx, pair) in table.rows.windows(2).enumerate() {
        let (previous, current) = (&pair[0], &pair[1]);
        let right_center = current.start + (current.end - current.start) / 2;
        let left_center = previous.start + (previous.end - previous.start) / 2;
        if right_center <= left_center {
            continue; // happens at chromosome borders
        }
        let _ = writeln!(
            out,
            "{}\t{}\t{}\t{:.12}",
            current.chrom,
            left_center,
            right_center,
            means[idx + 1]
        );
    }
    std::fs::write(path, out)?;
    Ok(())
}

/// `<prefix>_boundaries.bed`, `<prefix>_boundaries.gff` and
/// `<prefix>_domains.bed`.
fn write_boundaries(
    bed_path: &PathBuf,
    gff_path: &PathBuf,
    domains_path: &PathBuf,
    table: &ScoreTable,
    boundaries: &Boundaries,
    params: &Params,
) -> Result<(usize, usize)> {
    let means = table.row_means();
    let chrom: Vec<&str> = table.chroms();
    let n = chrom.len();
    let start: Vec<i64> = table.rows.iter().map(|r| r.start).collect();
    let end: Vec<i64> = table.rows.iter().map(|r| r.end).collect();

    let mut bed = String::new();
    let mut gff = String::new();
    let mut domains = String::new();
    let mut count = 1usize;
    let mut written = 0usize;

    for (idx, &min_bin_id) in boundaries.min_idx.iter().enumerate() {
        if boundaries.chrom_end_idx.contains(&min_bin_id) {
            continue;
        }
        // A boundary never sits on the first bin of a chromosome; the
        // previous bin would belong to another chromosome. Python reaches for
        // `chrom[-1]` there, which is the last bin of the table.
        let previous = if min_bin_id == 0 {
            n - 1
        } else {
            min_bin_id - 1
        };
        let right_center = start[min_bin_id] + (end[min_bin_id] - start[min_bin_id]) / 2;
        let left_center = start[previous] + (end[previous] - start[previous]) / 2;
        if chrom[min_bin_id] != chrom[previous] {
            continue;
        }

        let delta = boundaries
            .delta
            .get(&min_bin_id)
            .copied()
            .unwrap_or(f64::NAN);
        let pvalue = boundaries
            .pvalues
            .get(&min_bin_id)
            .copied()
            .unwrap_or(f64::NAN);
        let score = means[min_bin_id];

        let _ = writeln!(
            bed,
            "{}\t{}\t{}\tB{:05}\t{:.12}\t.",
            chrom[min_bin_id], left_center, right_center, min_bin_id, score
        );
        let _ = writeln!(
            gff,
            "{chrom}\tHiCExplorer\tboundary\t{left_center}\t{right_center}\t{score:.12}\t.\t.\t\
             ID=B{min_bin_id:05};delta={delta:.12};pvalue={pvalue:.12};tad_sep={score:.12}",
            chrom = chrom[min_bin_id]
        );
        written += 1;

        if idx + 1 == boundaries.min_idx.len()
            || chrom[min_bin_id] != chrom[boundaries.min_idx[idx + 1]]
        {
            continue;
        }
        let domain_start = start[min_bin_id];
        let domain_end = start[boundaries.min_idx[idx + 1]];
        let rgb = if count.is_multiple_of(2) {
            "51,160,44"
        } else {
            "31,120,180"
        };
        let _ = writeln!(
            domains,
            "{}\t{}\t{}\tID_{}_{}\t{:.12}\t.\t{}\t{}\t{}",
            chrom[min_bin_id],
            domain_start,
            domain_end,
            params.delta,
            count,
            score,
            domain_start,
            domain_end,
            rgb
        );
        count += 1;
    }

    std::fs::write(bed_path, bed)?;
    std::fs::write(gff_path, gff)?;
    std::fs::write(domains_path, domains)?;
    Ok((written, count - 1))
}

/// `<prefix>_zscore_matrix.cool`: the z-scored matrix, block-diagonal and
/// banded.
///
/// Upstream saves a `HiCMatrix` `.h5`; the values here are the same ones, in
/// the cooler schema. One difference: the source keeps `NaN` cells that lie
/// between the band it scores and the band it saves, because it subtracts one
/// triangular matrix from another and `NaN - NaN` survives. Those cells
/// carry no information, so this writer stops at the scored band.
fn write_zscore_matrix(path: &PathBuf, prepared: &Prepared) -> Result<()> {
    let chroms: Vec<Chrom> = prepared
        .chrom_bins
        .iter()
        .zip(prepared.chrom_lengths.iter())
        .map(|(bins, &length)| Chrom {
            name: bins.name.clone(),
            length,
        })
        .collect();

    let mut bins: Vec<Bin> = Vec::new();
    let mut offsets: Vec<i64> = Vec::with_capacity(chroms.len());
    for (slot, chrom) in prepared.chrom_bins.iter().enumerate() {
        offsets.push(bins.len() as i64);
        for i in 0..chrom.len() {
            bins.push(Bin {
                chrom_id: slot as i32,
                start: chrom.start[i] as i32,
                end: chrom.end[i] as i32,
            });
        }
    }

    let mut pixels: Vec<Pixel> = Vec::new();
    for (slot, band) in prepared.bands.iter().enumerate() {
        let n = band.size();
        for d in 0..band.depth() {
            for i in 0..n.saturating_sub(d) {
                let value = band.get(i, d);
                if value == 0.0 {
                    continue; // `eliminate_zeros`
                }
                pixels.push(Pixel {
                    bin1_id: offsets[slot] + i as i64,
                    bin2_id: offsets[slot] + (i + d) as i64,
                    count: value,
                });
            }
        }
    }

    let writer = CoolerWriter::create_with_bins(path, &chroms, &bins)?;
    writer.write_pixels(&pixels)?;
    Ok(())
}
