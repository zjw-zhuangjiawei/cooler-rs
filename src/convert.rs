//! Conversion between cooler format and other matrix formats.

use std::path::Path;

use crate::cooler::Cooler;
use crate::error::{Error, Result};
use crate::hic::HicWriter;
use crate::mcool::Mcool;
use crate::types::{Chrom, Pixel};

/// Parse a dense N×N whitespace-separated text matrix into sparse pixels.
///
/// Only upper-triangle non-zero entries are kept (symmetric-upper sparse
/// storage); zeros are implicit and omitted. Returns the matrix dimension
/// `n` together with the pixels, in row-major order.
///
/// This is the original OnTAD `.mat` text format.
pub fn dense_txt_to_pixels(text: &str) -> Result<(usize, Vec<Pixel>)> {
    let mut pixels: Vec<Pixel> = Vec::new();
    let mut width: Option<usize> = None;

    // Blank lines are skipped; only non-blank lines count as matrix rows.
    let mut i = 0;
    for (lineno, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut cols = 0;
        for (j, field) in line.split_whitespace().enumerate() {
            let v: f64 = field.parse().map_err(|_| {
                Error::InvalidInput(format!(
                    "line {}, column {}: '{field}' is not a number",
                    lineno + 1,
                    j + 1
                ))
            })?;
            if j >= i && v > 0.0 {
                pixels.push(Pixel {
                    bin1_id: i as i64,
                    bin2_id: j as i64,
                    count: v,
                });
            }
            cols += 1;
        }
        match width {
            None => width = Some(cols),
            Some(w) if w != cols => {
                return Err(Error::InvalidInput(format!(
                    "input is not a square N×N matrix: line {} has {cols} columns, expected {w}",
                    lineno + 1
                )));
            }
            _ => {}
        }
        i += 1;
    }

    let n = width.unwrap_or(0);
    if n == 0 {
        return Err(Error::InvalidInput("input is empty".into()));
    }
    Ok((n, pixels))
}

/// Convert a `.cool` or `.mcool` file into a `.hic` (format v8) file.
///
/// Every resolution present in the input becomes one resolution of the output
/// `.hic` (`.mcool` multi-resolution in, multi-resolution out; `.cool` yields
/// a single-resolution `.hic`). Input is auto-detected: an `.mcool` first
/// (recognized by its root `format = "HDF5::MCOOL"` attribute), then a `.cool`.
///
/// `genome_id` is stored in the `.hic` header — `.cool`/`.mcool` carry no
/// genome identifier, so the caller supplies one. When `weight_col` names a
/// column of the input `bins` table (e.g. `"weight"`), its per-bin values are
/// copied verbatim into the `.hic` footer as divisive normalization vectors,
/// named `weight_name` (or the column name when `weight_name` is `None`);
/// resolutions lacking the column are skipped with a log line. Copying is
/// verbatim: `.hic` consumers *divide* by these vectors while cooler `bins`
/// weights are conventionally multiplicative biases, so choosing the column
/// and its `.hic` name (juicer looks up `KR`/`VC`) is the operator's call.
///
/// Fails before writing when the input is not a fixed-bin-size cooler, when
/// the bins are not a uniform `div_ceil(chrom.length, resolution)` tiling
/// (the only binning `.hic` can represent), or when a chromosome is named
/// `All` (reserved for the genome-wide pseudo-chromosome).
// ponytail: Bounded-RAM contract — at most one resolution's classify data
// (≈ chunk_size × ~32 B/entry) is held in RAM at a time during streaming;
// per-resolution classified pixels spill to scratch files under the writer's
// tempdir. One `Cooler` HDF5 handle is open at a time. Upgrade path: if disk
// I/O dominates at petabyte scale, swap scratch tempfiles for a streaming
// block-level protocol (e.g. write classified blocks directly into the .hic
// matrix body as they're produced, hold only the per-res block index).
pub fn cooler_to_hic<P: AsRef<Path>, Q: AsRef<Path>>(
    input: P,
    output: Q,
    genome_id: &str,
    weight_col: Option<&str>,
    weight_name: Option<&str>,
) -> Result<()> {
    let input = input.as_ref();

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
    let mut weight_seen = false;
    for &r in &resolutions_u64 {
        let cool = open_res(input, r)?;
        if cool.chroms()? != chroms {
            return Err(Error::InvalidInput(
                "chromosome sets differ across resolutions".into(),
            ));
        }
        let res32 = resolution_u32(r)?;
        check_uniform_bins(&cool, &chroms, res32)?;
        if let Some(col) = weight_col {
            weight_seen |= cool.bins_has_column(col)?;
        }
        res_u32.push(res32);
        drop(cool);
    }
    if let Some(col) = weight_col {
        if !weight_seen {
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
        if let Some(col) = weight_col {
            let name = weight_name.unwrap_or(col);
            if cool.bins_has_column(col)? {
                let vectors = split_bins_column(&cool, &chroms, col)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_only_upper_triangle_nonzero() {
        // Lower triangle and zeros are dropped; values are kept exactly.
        let (n, pixels) = dense_txt_to_pixels("1 0 2\n3 4 5\n6 0 7\n").unwrap();
        assert_eq!(n, 3);
        assert_eq!(
            pixels,
            vec![
                Pixel {
                    bin1_id: 0,
                    bin2_id: 0,
                    count: 1.0
                },
                Pixel {
                    bin1_id: 0,
                    bin2_id: 2,
                    count: 2.0
                },
                Pixel {
                    bin1_id: 1,
                    bin2_id: 1,
                    count: 4.0
                },
                Pixel {
                    bin1_id: 1,
                    bin2_id: 2,
                    count: 5.0
                },
                Pixel {
                    bin1_id: 2,
                    bin2_id: 2,
                    count: 7.0
                },
            ]
        );
    }

    #[test]
    fn skips_blank_lines() {
        let (n, pixels) = dense_txt_to_pixels("1 2\n\n3 4\n").unwrap();
        assert_eq!(n, 2);
        assert_eq!(
            pixels,
            vec![
                Pixel {
                    bin1_id: 0,
                    bin2_id: 0,
                    count: 1.0
                },
                Pixel {
                    bin1_id: 0,
                    bin2_id: 1,
                    count: 2.0
                },
                Pixel {
                    bin1_id: 1,
                    bin2_id: 1,
                    count: 4.0
                },
            ]
        );
    }

    #[test]
    fn rejects_non_square_matrix() {
        let err = dense_txt_to_pixels("1 2 3\n4 5\n").unwrap_err();
        assert!(err.to_string().contains("not a square"), "{err}");
    }

    #[test]
    fn rejects_non_numeric_entry() {
        let err = dense_txt_to_pixels("1 2\n3 x\n").unwrap_err();
        assert!(err.to_string().contains("'x' is not a number"), "{err}");
    }

    #[test]
    fn rejects_empty_input() {
        let err = dense_txt_to_pixels("  \n\n").unwrap_err();
        assert!(err.to_string().contains("empty"), "{err}");
    }
}
