//! Unified read entry over `.cool`/`.mcool` and `.hic`, mirroring hictk's
//! `File` variant: one type exposes a common pixel/index read surface plus
//! normalization-aware `fetch`, so callers (and balancing) work on either
//! format without branching.

use std::borrow::Cow;
use std::sync::OnceLock;

use crate::cooler::Cooler;
use crate::error::{Error, Result};
use crate::hic::HiCFile;
use crate::mcool::Mcool;
use crate::region::Region;
use crate::types::{Bin, Chrom, MatrixSource, Pixel};

/// A single-resolution contact matrix, backed by a `.cool`/`.mcool` collection
/// or a `.hic` file opened at one resolution.
pub enum File {
    Cooler(Cooler),
    Hic(Box<HicState>),
}

/// A `.hic` file plus a materialized, bin-sorted view of it at one resolution.
pub struct HicState {
    file: HiCFile,
    resolution: u32,
    cache: OnceLock<std::result::Result<HicData, String>>,
}

/// Materialized data for one `.hic` resolution: every pixel sorted by
/// `(bin1_id, bin2_id)`, plus the row offsets that slicing depends on.
///
/// Deliberately holds no bins or chromosome offsets — those come from
/// [`HicState::tiling`] and cost nothing, so a bin-table query never drags the
/// matrix in with it.
struct HicData {
    pixels: Vec<Pixel>,
    bin1_offset: Vec<i64>,
}

impl HicState {
    fn data(&self) -> Result<&HicData> {
        let r = self
            .cache
            .get_or_init(|| build_hic(self).map_err(|e| e.to_string()));
        r.as_ref().map_err(|s| Error::Format(s.clone()))
    }

    /// Uniform bins and per-chromosome offsets at this resolution.
    ///
    /// Derived from the header's chromosome lengths, so it never forces
    /// `data()` — the whole point, since `data()` materializes and sorts every
    /// pixel of the resolution. Callers that want a bin table, a bin count or
    /// a chromosome offset must not pay for the matrix.
    fn tiling(&self) -> (Vec<Bin>, Vec<i64>) {
        let mut bins = Vec::new();
        let mut chrom_offset = vec![0i64];
        for (k, c) in self.file.chromosomes().iter().enumerate() {
            let n = (c.length as u64).div_ceil(self.resolution as u64);
            for i in 0..n {
                bins.push(Bin {
                    chrom_id: k as i32,
                    start: (i * self.resolution as u64) as i32,
                    end: (((i + 1) * self.resolution as u64).min(c.length as u64)) as i32,
                });
            }
            chrom_offset.push(bins.len() as i64);
        }
        (bins, chrom_offset)
    }
}

fn build_hic(state: &HicState) -> Result<HicData> {
    let (bins, _) = state.tiling();

    let mut pixels = state.file.pixels(state.resolution)?;
    pixels.sort_unstable_by_key(|p| (p.bin1_id, p.bin2_id));

    let n_bins = bins.len();
    let mut bin1_offset = vec![0i64; n_bins + 1];
    for p in &pixels {
        bin1_offset[p.bin1_id as usize + 1] += 1;
    }
    for i in 0..n_bins {
        bin1_offset[i + 1] += bin1_offset[i];
    }

    Ok(HicData {
        pixels,
        bin1_offset,
    })
}

impl File {
    /// Open by path, detecting the format from the file magic, and select
    /// `resolution` (an `.mcool` bin size or a `.hic` base-pair resolution; a
    /// `.cool` file must already be at that resolution).
    pub fn open(path: &str, resolution: u32) -> Result<Self> {
        use std::io::Read;

        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;

        if &magic == b"HIC\0" {
            let file = HiCFile::open(path)?;
            return Ok(File::Hic(Box::new(HicState {
                file,
                resolution,
                cache: OnceLock::new(),
            })));
        }

        if let Ok(mcool) = Mcool::open(path) {
            return Ok(File::Cooler(mcool.cooler(resolution as u64)?));
        }

        let cooler = Cooler::open_any(path)?;
        if let Some(bs) = cooler.bin_size()? {
            if bs != resolution as u64 {
                return Err(Error::InvalidInput(format!(
                    "resolution {resolution} does not match cooler bin-size {bs}"
                )));
            }
        }
        Ok(File::Cooler(cooler))
    }

    pub fn chroms(&self) -> Result<Vec<Chrom>> {
        match self {
            File::Cooler(c) => c.chroms(),
            File::Hic(h) => Ok(h.file.chromosomes()),
        }
    }

    pub fn bins(&self) -> Result<Vec<Bin>> {
        match self {
            File::Cooler(c) => c.bins(),
            File::Hic(h) => Ok(h.tiling().0),
        }
    }

    pub fn resolution(&self) -> u32 {
        match self {
            File::Cooler(c) => c.bin_size().ok().flatten().unwrap_or(0) as u32,
            File::Hic(h) => h.resolution,
        }
    }

    pub fn n_bins(&self) -> Result<usize> {
        match self {
            File::Cooler(c) => Ok(c.bins()?.len()),
            File::Hic(h) => Ok(h.tiling().0.len()),
        }
    }

    /// Every pixel at this resolution, sorted by `(bin1_id, bin2_id)`.
    ///
    /// Borrowed for `.hic` (the cached view is reused, not copied) and owned
    /// for `.cool`, whose pixels are read out of HDF5 on demand. The earlier
    /// `.hic` arm cloned the cache, doubling peak RAM on a fine resolution.
    pub fn pixels(&self) -> Result<Cow<'_, [Pixel]>> {
        match self {
            File::Cooler(c) => Ok(Cow::Owned(c.pixels()?)),
            File::Hic(h) => Ok(Cow::Borrowed(&h.data()?.pixels)),
        }
    }

    pub fn chrom_offset(&self) -> Result<Vec<i64>> {
        match self {
            File::Cooler(c) => c.chrom_offset(),
            File::Hic(h) => Ok(h.tiling().1),
        }
    }

    pub fn bin1_offset(&self) -> Result<Vec<i64>> {
        match self {
            File::Cooler(c) => c.bin1_offset(),
            File::Hic(h) => Ok(h.data()?.bin1_offset.clone()),
        }
    }

    /// Fetch the pixels overlapping `region`, applying `norm` (a `bins` column
    /// name for `.cool`, a normalization-vector name for `.hic`; `None` means
    /// raw counts).
    pub fn fetch(&self, region: &Region, norm: Option<&str>) -> Result<Vec<Pixel>> {
        match self {
            File::Cooler(c) => {
                let mut pixels = c.pixels_in(region)?;
                if let Some(name) = norm {
                    let w = c
                        .bins_column_f64(name)?
                        .ok_or_else(|| Error::InvalidInput(format!("no 'bins/{name}' column")))?;
                    for p in &mut pixels {
                        p.count *= w[p.bin1_id as usize] * w[p.bin2_id as usize];
                    }
                }
                Ok(pixels)
            }
            File::Hic(h) => {
                let chroms = h.file.chromosomes();
                let Some((cid, clen)) = chroms
                    .iter()
                    .enumerate()
                    .find(|(_, c)| c.name == region.chrom)
                    .map(|(i, c)| (i, c.length as u64))
                else {
                    return Err(Error::InvalidInput(format!(
                        "unknown sequence label: {}",
                        region.chrom
                    )));
                };
                let start = region.start.unwrap_or(0);
                let end = region.end.unwrap_or(clen);
                if end < start || end > clen {
                    return Err(Error::InvalidInput(format!(
                        "region out of bounds on '{}' (length {clen}): [{start}, {end})",
                        region.chrom
                    )));
                }
                let res = h.resolution as u64;
                let lo = start / res;
                let hi = if end == start { lo } else { end.div_ceil(res) };
                let (_, chrom_offset) = h.tiling();
                let g0 = chrom_offset[cid];
                let g1 = chrom_offset[cid + 1];
                let data = h.data()?;
                let lo_g = g0 + lo as i64;
                let hi_g = g0 + hi as i64;
                let mut pixels: Vec<Pixel> = data
                    .pixels
                    .iter()
                    .filter(|p| {
                        p.bin1_id >= lo_g && p.bin1_id < hi_g && p.bin2_id >= g0 && p.bin2_id < g1
                    })
                    .copied()
                    .collect();
                if let Some(name) = norm {
                    let w = h
                        .file
                        .norm_vector(h.resolution, &region.chrom, name)?
                        .ok_or_else(|| {
                            Error::InvalidInput(format!(
                                "no '{name}' normalization for {} at {} bp",
                                region.chrom, h.resolution
                            ))
                        })?;
                    for p in &mut pixels {
                        let li = (p.bin1_id - g0) as usize;
                        let lj = (p.bin2_id - g0) as usize;
                        p.count /= w[li] * w[lj];
                    }
                }
                Ok(pixels)
            }
        }
    }

    /// Names of the available normalizations (bins columns for `.cool`,
    /// normalization-vector types for `.hic`).
    pub fn avail_normalizations(&self) -> Result<Vec<String>> {
        match self {
            File::Cooler(c) => c.bins_column_names(),
            File::Hic(h) => h.file.avail_normalizations(),
        }
    }

    pub fn has_normalization(&self, name: &str) -> bool {
        match self {
            File::Cooler(c) => c.bins_has_column(name).unwrap_or(false),
            File::Hic(h) => h
                .file
                .avail_normalizations()
                .map(|v| v.iter().any(|n| n == name))
                .unwrap_or(false),
        }
    }

    /// Per-distance expected values for a chromosome under `norm`, if the file
    /// stores them (`.hic` only). Always `None` for `.cool`.
    pub fn expected_values(&self, _chrom: &str, _norm: Option<&str>) -> Result<Option<Vec<f64>>> {
        Ok(None)
    }
}

impl MatrixSource for Box<HicState> {
    fn n_pixels(&self) -> Result<u64> {
        Ok(self.data()?.pixels.len() as u64)
    }

    fn n_bins(&self) -> Result<usize> {
        Ok(self.tiling().0.len())
    }

    fn bin_chrom(&self) -> Result<Vec<i32>> {
        Ok(self.tiling().0.iter().map(|b| b.chrom_id).collect())
    }

    fn chrom_offset(&self) -> Result<Vec<i64>> {
        Ok(self.tiling().1)
    }

    fn bin1_offset(&self) -> Result<Vec<i64>> {
        Ok(self.data()?.bin1_offset.clone())
    }

    fn pixels_range(&self, lo: i64, hi: i64) -> Result<Vec<Pixel>> {
        let d = self.data()?;
        Ok(d.pixels[lo as usize..hi as usize].to_vec())
    }
}

impl MatrixSource for File {
    fn n_pixels(&self) -> Result<u64> {
        match self {
            File::Cooler(c) => c.n_pixels(),
            File::Hic(h) => h.n_pixels(),
        }
    }

    fn n_bins(&self) -> Result<usize> {
        match self {
            File::Cooler(c) => c.n_bins(),
            File::Hic(h) => h.n_bins(),
        }
    }

    fn bin_chrom(&self) -> Result<Vec<i32>> {
        match self {
            File::Cooler(c) => c.bin_chrom(),
            File::Hic(h) => h.bin_chrom(),
        }
    }

    fn chrom_offset(&self) -> Result<Vec<i64>> {
        match self {
            File::Cooler(c) => c.chrom_offset(),
            File::Hic(h) => h.chrom_offset(),
        }
    }

    fn bin1_offset(&self) -> Result<Vec<i64>> {
        match self {
            File::Cooler(c) => c.bin1_offset(),
            File::Hic(h) => h.bin1_offset(),
        }
    }

    fn pixels_range(&self, lo: i64, hi: i64) -> Result<Vec<Pixel>> {
        match self {
            File::Cooler(c) => c.pixels_range(lo, hi),
            File::Hic(h) => h.pixels_range(lo, hi),
        }
    }
}
