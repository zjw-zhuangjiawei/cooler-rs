//! Zoomify: coarsen a single-resolution `.cool` into a multi-resolution
//! `.mcool`.
//!
//! Each target resolution is coarsened directly from the base resolution by
//! pooling `factor` × `factor` blocks of bins and summing counts (`factor =
//! target / base`), mirroring `cooler.zoomify_cooler` / `hictk zoomify`.
//! Bins are pooled per-chromosome so coarse bin ids never span chromosome
//! boundaries (a naive global floor-division corrupts ids when a chromosome's
//! bin count is not a multiple of `factor`).

use std::collections::HashMap;
use std::path::Path;

use crate::cooler::Cooler;
use crate::error::{Error, Result};
use crate::mcool::McoolWriter;
use crate::types::{Chrom, Pixel};

/// HiGlass tile dimension: the coarsest resolution keeps the genome within a
/// single 256×256 tile (cooler's `HIGLASS_TILE_DIM`).
const HIGLASS_TILE_DIM: u64 = 256;
/// Pixels buffered per streaming read (mirrors `BalanceParams::chunksize`).
const CHUNKSIZE: usize = 10_000_000;

/// Parameters for [`zoomify_cooler`].
#[derive(Debug, Clone)]
pub struct ZoomifyParams {
    /// Target resolutions (bin sizes). Each must be a multiple of the base
    /// resolution and at least as large. Empty => auto-generate.
    pub resolutions: Vec<u32>,
    /// Include the base resolution in the output `.mcool`. Default `true`.
    pub copy_base_resolution: bool,
    /// Use nice (1-2-5) steps when auto-generating. Default `true`; `false`
    /// uses power-of-two (×2) steps.
    pub nice_steps: bool,
    /// Overwrite an existing output file. Default `false`.
    pub force: bool,
}

impl Default for ZoomifyParams {
    fn default() -> Self {
        ZoomifyParams {
            resolutions: Vec::new(),
            copy_base_resolution: true,
            nice_steps: true,
            force: false,
        }
    }
}

/// Geometric ×2 progression from `base` up to `max` (inclusive).
pub fn pow2_resolutions(base: u32, max: u32) -> Vec<u32> {
    let mut out = vec![base];
    loop {
        let next = out.last().copied().unwrap().saturating_mul(2);
        if next > max {
            break;
        }
        out.push(next);
    }
    out
}

/// Nice (1-2-5) progression from `base` up to `max` (inclusive), mirroring
/// cooler's `niceprog`: emit `seed×2, seed×5, seed×10`, then ×10 the seed and
/// repeat.
pub fn nice_resolutions(base: u32, max: u32) -> Vec<u32> {
    let mut out = vec![base];
    let mut seed = base;
    loop {
        let mut over = false;
        for mul in [2u32, 5, 10] {
            let next = seed.saturating_mul(mul);
            if next > max {
                over = true;
                break;
            }
            out.push(next);
        }
        if over {
            break;
        }
        seed = seed.saturating_mul(10);
    }
    out
}

/// Coarsen a single-chromosome pixel set by pooling `factor`×`factor` bins and
/// summing counts. Bin ids are floor-divided by `factor`; `factor` must be >= 2.
pub fn coarsen_pixels(pixels: impl IntoIterator<Item = Pixel>, factor: u32) -> Vec<Pixel> {
    assert!(factor >= 2, "coarsening factor must be >= 2");
    let f = i64::from(factor);
    let mut acc: HashMap<(i64, i64), f64> = HashMap::new();
    for p in pixels {
        *acc.entry((p.bin1_id / f, p.bin2_id / f)).or_insert(0.0) += p.count;
    }
    let mut out: Vec<Pixel> = acc
        .into_iter()
        .map(|((bin1_id, bin2_id), count)| Pixel {
            bin1_id,
            bin2_id,
            count,
        })
        .collect();
    out.sort_by_key(|p| (p.bin1_id, p.bin2_id));
    out
}

/// Map each base bin to its coarser bin id, pooled per-chromosome. Bins are
/// laid out chromosome-by-chromosome, so `base_off` (the `chrom_offset` index)
/// partitions them; each chromosome contributes `ceil(n_local / factor)` coarse
/// bins.
fn coarse_bin_map(base_off: &[i64], factor: u32) -> Vec<i64> {
    let f = i64::from(factor);
    let n_bins = *base_off.last().unwrap_or(&0) as usize;
    let mut map = vec![0i64; n_bins];
    let mut coarse_off = 0i64;
    for w in base_off.windows(2) {
        let lo = w[0] as usize;
        let hi = w[1] as usize;
        let n_local = (hi - lo) as i64;
        let n_coarse = (n_local + f - 1) / f; // ceil(n_local / f), n_local >= 0
        for local in 0..n_local {
            map[lo + local as usize] = coarse_off + local / f;
        }
        coarse_off += n_coarse;
    }
    map
}

/// Coarsen `clr` by `factor` into a per-chromosome pooled pixel set, streaming
/// the input so memory is bounded by the (factor² smaller) coarse output.
fn coarsen_level(clr: &Cooler, factor: u32) -> Result<Vec<Pixel>> {
    let map = coarse_bin_map(&clr.chrom_offset()?, factor);
    let nnz = clr.n_pixels()? as usize;
    let mut acc: HashMap<(i64, i64), f64> = HashMap::new();
    let mut lo = 0usize;
    while lo < nnz {
        let hi = (lo + CHUNKSIZE).min(nnz);
        for p in clr.pixels_range(lo as i64, hi as i64)? {
            let b1 = map[p.bin1_id as usize];
            let b2 = map[p.bin2_id as usize];
            *acc.entry((b1, b2)).or_insert(0.0) += p.count;
        }
        lo = hi;
    }
    let mut out: Vec<Pixel> = acc
        .into_iter()
        .map(|((bin1_id, bin2_id), count)| Pixel {
            bin1_id,
            bin2_id,
            count,
        })
        .collect();
    out.sort_by_key(|p| (p.bin1_id, p.bin2_id));
    Ok(out)
}

/// Zoomify a single-resolution `.cool` into a multi-resolution `.mcool`.
pub fn zoomify_cooler<P: AsRef<Path>, Q: AsRef<Path>>(
    input: P,
    output: Q,
    params: &ZoomifyParams,
) -> Result<()> {
    let output = output.as_ref();
    if output.exists() && !params.force {
        return Err(Error::InvalidInput(format!(
            "refusing to overwrite '{}'; pass --force to overwrite",
            output.display()
        )));
    }

    let clr = Cooler::open(input)?;
    let base = match clr.bin_size()? {
        Some(b) if b > 0 => b as u32,
        _ => {
            return Err(Error::InvalidInput(
                "zoomify requires a fixed-bin-size .cool input".into(),
            ))
        }
    };
    let chroms = clr.chroms()?;
    let targets = resolve_targets(&chroms, base, params)?;

    let mw = McoolWriter::create(output)?;
    for res in targets {
        let factor = res / base; // >= 1; factor == 1 copies the base resolution
        let pixels = coarsen_level(&clr, factor)?;
        let cw = mw.create_cooler(&chroms, res)?;
        cw.write_pixels(&pixels)?;
        log::info!("wrote resolution {} ({} pixels)", res, pixels.len());
    }
    Ok(())
}

/// Resolve the list of resolutions to write, base first (when requested).
fn resolve_targets(chroms: &[Chrom], base: u32, params: &ZoomifyParams) -> Result<Vec<u32>> {
    let targets: Vec<u32> = if params.resolutions.is_empty() {
        let total: u64 = chroms.iter().map(|c| c.length as u64).sum();
        let max = total.div_ceil(HIGLASS_TILE_DIM) as u32;
        if params.nice_steps {
            nice_resolutions(base, max)
        } else {
            pow2_resolutions(base, max)
        }
    } else {
        let mut rs = params.resolutions.clone();
        rs.sort_unstable();
        for w in rs.windows(2) {
            if w[0] == w[1] {
                return Err(Error::InvalidInput(format!(
                    "duplicate resolution {}",
                    w[0]
                )));
            }
        }
        for r in &rs {
            if *r < base {
                return Err(Error::InvalidInput(format!(
                    "resolution {r} is smaller than the base resolution {base}"
                )));
            }
            if r % base != 0 {
                return Err(Error::InvalidInput(format!(
                    "resolution {r} is not a multiple of the base resolution {base}"
                )));
            }
        }
        rs
    };

    let mut out = Vec::new();
    if params.copy_base_resolution {
        out.push(base);
    }
    out.extend(targets.into_iter().filter(|&r| r > base));

    if out.len() == 1 && params.copy_base_resolution {
        log::warn!("no coarser resolution generated; wrote the base resolution only");
    }
    if out.is_empty() {
        return Err(Error::InvalidInput("no resolutions to write".into()));
    }
    Ok(out)
}
