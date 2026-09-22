//! Conversion between cooler format and other *Hi-C container* formats.
//!
//! Plain-text matrix formats are deliberately not here: the dense N×N OnTAD
//! `.mat` reader/writer lives with the `dump`/`load` commands.

use std::path::Path;

use crate::cooler::Cooler;
use crate::error::{Error, Result};
use crate::hic::HicWriter;
use crate::mcool::Mcool;
use crate::types::Chrom;

/// Convert a `.cool` or `.mcool` file into a `.hic` file.
///
/// Every resolution present in the input becomes one resolution of the output
/// `.hic` (`.mcool` multi-resolution in, multi-resolution out; `.cool` yields
/// a single-resolution `.hic`). Input is auto-detected: an `.mcool` first
/// (recognized by its root `format = "HDF5::MCOOL"` attribute), then a `.cool`.
///
/// `genome_id` is stored in the `.hic` header — `.cool`/`.mcool` carry no
/// genome identifier, so the caller supplies one. When `weight_col` names a
/// column of the input `bins` table (e.g. `"weight"`), its per-bin values are
/// written to the `.hic` footer as normalization vectors named `weight_name`
/// (or the column name when `weight_name` is `None`); resolutions lacking the
/// column are skipped with a log line.
///
/// The values are **inverted** on the way out. A `.hic` consumer divides by
/// the vector (`count / (v1 * v2)`, juicer's convention) while a cooler
/// `bins/weight` column is a multiplicative bias (`count * w1 * w2`, cooler's
/// convention), so a verbatim copy balances the matrix by the reciprocal of
/// the intended factor — off by `w^4`, and absurd in magnitude. hictk inverts
/// too (`convert/cool_to_hic.cpp`).
// ponytail: unconditionally inverts, i.e. assumes the column is multiplicative
// — true for cooler's `weight` and for this crate's own balance/Raichu output.
// A cooler file can mark a column divisive with a `divisive_weights` attribute;
// honour that here if such a file ever shows up.
///
/// Fails before writing when the input is not a fixed-bin-size cooler, when
/// the bins are not a uniform `div_ceil(chrom.length, resolution)` tiling
/// (the only binning `.hic` can represent), or when a chromosome is named
/// `All` (reserved for the genome-wide pseudo-chromosome).
// ponytail: Bounded-RAM contract — at most one chromosome pair's classify
// data is held in RAM at a time (≈ chunk_size × ~32 B/entry while streaming;
// one pair's full pixel set while finalizing); per-pair classified pixels
// spill to scratch files under the writer's tempdir. One `Cooler` HDF5 handle
// is open at a time. Upgrade path: if disk
// I/O dominates at petabyte scale, swap scratch tempfiles for a streaming
// block-level protocol (e.g. write classified blocks directly into the .hic
// matrix body as they're produced, hold only the per-res block index).
pub fn cooler_to_hic<P: AsRef<Path>, Q: AsRef<Path>>(
    input: P,
    output: Q,
    genome_id: &str,
    weight_cols: &[String],
    weight_name: Option<&str>,
    resolutions: &[u32],
) -> Result<()> {
    let input = input.as_ref();

    // One name cannot label several vectors, and defaulting it to the first
    // column would silently mislabel the rest.
    if weight_name.is_some() && weight_cols.len() > 1 {
        return Err(Error::InvalidInput(
            "--weight-name names a single vector; with several weight columns each keeps its own column name"
                .into(),
        ));
    }

    // Phase 1: discover resolutions + chroms without holding every Cooler.
    let (resolutions_u64, chroms) = if let Ok(mcool) = Mcool::open(input) {
        let resolutions = mcool.resolutions()?;
        if resolutions.is_empty() {
            return Err(Error::InvalidInput(
                "input .mcool has no resolutions".into(),
            ));
        }
        let chroms = mcool.cooler(resolutions[0])?.chroms()?;
        (resolutions, chroms)
    } else {
        let cool = Cooler::open_any(input).map_err(|e| {
            Error::InvalidInput(format!(
                "cannot read '{}' as a .cool/.mcool file: {e}",
                input.display()
            ))
        })?;
        let res = match cool.bin_size()? {
            Some(b) if b > 0 => resolution_u32(b)?,
            _ => {
                return Err(Error::InvalidInput(
                    "conversion to .hic requires a fixed-bin-size .cool/.mcool input".into(),
                ))
            }
        };
        (vec![res as u64], cool.chroms()?)
    };

    // An empty `resolutions` is "all of them". A requested resolution the
    // input does not have is an error rather than a silent omission — the
    // caller asked for a specific output and getting fewer resolutions than
    // named would only show up later, in the .hic's resolution list.
    let resolutions_u64 = if resolutions.is_empty() {
        resolutions_u64
    } else {
        let mut want: Vec<u64> = resolutions.iter().map(|&r| r as u64).collect();
        want.sort_unstable();
        want.dedup();
        for r in &want {
            if !resolutions_u64.contains(r) {
                return Err(Error::InvalidInput(format!(
                    "resolution {r} is not present in '{}' (has {resolutions_u64:?})",
                    input.display()
                )));
            }
        }
        if want.len() < resolutions_u64.len() {
            log::info!(
                "converting {} of {} resolutions: {want:?}",
                want.len(),
                resolutions_u64.len()
            );
        }
        want
    };

    // The `All` pseudo-chromosome is reserved by the writer (and filtered by
    // the reader); a real `All` chromosome would be silently dropped.
    for c in &chroms {
        if c.name.eq_ignore_ascii_case("All") {
            return Err(Error::InvalidInput(
                "chromosome named 'All' is reserved by .hic and cannot be converted".into(),
            ));
        }
    }

    // Phase 2: validate per-res by reopening one Cooler at a time.
    let mut res_u32 = Vec::with_capacity(resolutions_u64.len());
    let mut weight_seen = vec![false; weight_cols.len()];
    for &r in &resolutions_u64 {
        let cool = open_res(input, r)?;
        if cool.chroms()? != chroms {
            return Err(Error::InvalidInput(
                "chromosome sets differ across resolutions".into(),
            ));
        }
        let res32 = resolution_u32(r)?;
        check_uniform_bins(&cool, &chroms, res32)?;
        for (i, col) in weight_cols.iter().enumerate() {
            weight_seen[i] |= cool.bins_has_column(col)?;
        }
        res_u32.push(res32);
        drop(cool);
    }
    for (col, seen) in weight_cols.iter().zip(&weight_seen) {
        if !*seen {
            return Err(Error::InvalidInput(format!(
                "bins column '{col}' not found in the input"
            )));
        }
    }

    // Phase 3: drive the writer, one resolution at a time.
    let mut writer = HicWriter::create(output, genome_id, &chroms, &res_u32)?;
    // ponytail: 1M-row chunk ≈ 24 MB raw; halve if peak classify-map RAM matters more than throughput.
    let chunk_size: i64 = 1_000_000;
    for &res in &res_u32 {
        let cool = open_res(input, res as u64)?;
        let mut n_pix: u64 = 0;
        for chunk in cool.pixels_chunked(chunk_size)? {
            let chunk = chunk?;
            n_pix += chunk.len() as u64;
            writer.add_pixel_chunk(res, &chunk)?;
        }
        writer.finish_resolution(res)?;
        for col in weight_cols {
            if cool.bins_has_column(col)? {
                let name = if weight_cols.len() == 1 {
                    weight_name.unwrap_or(col.as_str())
                } else {
                    col.as_str()
                };
                let mut vectors = split_bins_column(&cool, &chroms, col)?;
                // cooler weights multiply, `.hic` normalization vectors divide.
                for (_, values) in &mut vectors {
                    for w in values.iter_mut() {
                        *w = 1.0 / *w;
                    }
                }
                writer.add_normalization_vectors(res, name, &vectors)?;
            } else {
                log::info!(
                    "resolution {res} has no bins/{col} column; writing no normalization vectors"
                );
            }
        }
        log::info!("wrote resolution {res} ({n_pix} pixels)");
        drop(cool);
    }
    writer.finalize()?;
    Ok(())
}

/// Open the input as a `Cooler` for the given resolution: `.mcool` group if
/// the input is multi-resolution, otherwise the single-resolution `.cool`.
fn open_res(input: &Path, res: u64) -> Result<Cooler> {
    if let Ok(mcool) = Mcool::open(input) {
        return mcool.cooler(res);
    }
    Cooler::open_any(input).map_err(|e| {
        Error::InvalidInput(format!(
            "cannot read '{}' as a .cool/.mcool file: {e}",
            input.display()
        ))
    })
}

/// Reject resolutions that cannot be represented in the `.hic` header.
fn resolution_u32(res: u64) -> Result<u32> {
    let r = u32::try_from(res)
        .map_err(|_| Error::InvalidInput(format!("resolution {res} too large for .hic")))?;
    if r == 0 || r as i64 > i32::MAX as i64 {
        return Err(Error::InvalidInput(format!(
            "resolution {res} out of range for .hic"
        )));
    }
    Ok(r)
}

/// Check that every chromosome's bins form a uniform `div_ceil(length, res)`
/// tiling — the only binning a `.hic` can express (`.hic` derives bin counts
/// from chromosome length + resolution).
fn check_uniform_bins(cool: &Cooler, chroms: &[Chrom], res: u32) -> Result<()> {
    let offset = cool.chrom_offset()?;
    if offset.len() != chroms.len() + 1 {
        return Err(Error::InvalidInput("chrom_offset/chroms mismatch".into()));
    }
    for (i, c) in chroms.iter().enumerate() {
        let n_bins = (c.length as u64).div_ceil(res as u64) as i64;
        let have = offset[i + 1] - offset[i];
        if have != n_bins {
            return Err(Error::InvalidInput(format!(
                "resolution {res}: chromosome {} has {have} bins but a uniform {res} bp tiling needs {n_bins}; non-uniform bins cannot convert to .hic",
                c.name
            )));
        }
    }
    Ok(())
}

/// Split a flat `bins` column into one per-bin vector per chromosome, slicing
/// by `chrom_offset` (bins are grouped contiguously per chromosome).
fn split_bins_column(
    cool: &Cooler,
    chroms: &[Chrom],
    col: &str,
) -> Result<Vec<(String, Vec<f64>)>> {
    let offset = cool.chrom_offset()?;
    if offset.len() != chroms.len() + 1 {
        return Err(Error::InvalidInput("chrom_offset/chroms mismatch".into()));
    }
    let values = cool.bins_column_f64(col)?.ok_or_else(|| {
        Error::InvalidInput(format!("bins column '{col}' not found in the input"))
    })?;
    if values.len() as i64 != *offset.last().unwrap() {
        return Err(Error::InvalidInput(format!(
            "bins/{col} has {} values but the input has {} bins",
            values.len(),
            offset.last().unwrap()
        )));
    }
    Ok(chroms
        .iter()
        .enumerate()
        .map(|(i, c)| {
            (
                c.name.clone(),
                values[offset[i] as usize..offset[i + 1] as usize].to_vec(),
            )
        })
        .collect())
}
