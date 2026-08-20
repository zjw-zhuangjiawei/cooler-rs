//! Reading and writing single-resolution `.cool` files.
//!
//! A cooler collection is a group (the file root for `.cool`, or
//! `/resolutions/<binsize>` for `.mcool`) containing `chroms`, `bins`,
//! `pixels` and `indexes` tables, following schema version 3:
//! <https://cooler.readthedocs.io/en/latest/schema.html>

use std::path::Path;

use hdf5_metno::types::VarLenUnicode;
use hdf5_metno::{File, Group};

use crate::error::{Error, Result};
use crate::types::{Bin, Chrom, Pixel};

/// Value of the `format` attribute for single-resolution files.
pub const COOL_FORMAT: &str = "HDF5::Cooler";
/// Schema version written to the `format-version` attribute.
pub const COOL_FORMAT_VERSION: i64 = 3;

fn write_strings(group: &Group, name: &str, values: &[String]) -> Result<()> {
    // cooler-python writes fixed-length ASCII (|S<maxlen>), which h5py/cooler
    // decode to str on read. Vlen strings come back as bytes and break
    // cooler-python's region parsing, so match the fixed-length layout
    // (bucketed sizes, null-padded).
    let maxlen = values.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
    macro_rules! write_fixed {
        ($n:expr) => {{
            use hdf5_metno::types::FixedAscii;
            let mut vals = Vec::with_capacity(values.len());
            for s in values {
                vals.push(FixedAscii::<$n>::from_ascii(s).map_err(|e| {
                    Error::Format(format!("cannot write string '{s}' as ASCII: {e}"))
                })?);
            }
            let ds = group
                .new_dataset::<FixedAscii<$n>>()
                .shape(values.len())
                .create(name)?;
            ds.write(&vals)?;
            Ok(())
        }};
    }
    match maxlen {
        1 => write_fixed!(1),
        2 => write_fixed!(2),
        3 => write_fixed!(3),
        4 => write_fixed!(4),
        5 => write_fixed!(5),
        6 => write_fixed!(6),
        7 => write_fixed!(7),
        8..=10 => write_fixed!(10),
        11..=12 => write_fixed!(12),
        13..=16 => write_fixed!(16),
        17..=24 => write_fixed!(24),
        25..=32 => write_fixed!(32),
        33..=64 => write_fixed!(64),
        65..=128 => write_fixed!(128),
        129..=256 => write_fixed!(256),
        n => Err(Error::Format(format!(
            "string too long ({n} bytes) for fixed-string dataset '{name}'"
        ))),
    }
}

fn write_attr_str(group: &Group, name: &str, value: &str) -> Result<()> {
    group
        .new_attr::<VarLenUnicode>()
        .create(name)?
        .write_scalar(&value.parse::<VarLenUnicode>().expect("valid UTF-8"))?;
    Ok(())
}

fn write_attr_int(group: &Group, name: &str, value: i64) -> Result<()> {
    group.new_attr::<i64>().create(name)?.write_scalar(&value)?;
    Ok(())
}

/// Writer for a single-resolution cooler collection.
///
/// Use [`CoolerWriter::create`] for a standalone `.cool` file, or
/// [`crate::McoolWriter::create_cooler`] to add a resolution to an `.mcool`
/// file. Bins of fixed size are generated from the chromosome table;
/// pixels can then be appended with [`CoolerWriter::write_pixels`].
pub struct CoolerWriter {
    group: Group,
    n_bins: u64,
}

impl CoolerWriter {
    /// Create a new `.cool` file (overwriting any existing file) and write
    /// the chromosome and bin tables.
    ///
    /// `bin_size` must be positive; bins are generated to tile each
    /// chromosome: `start = i * bin_size`, `end = min(start + bin_size, length)`.
    pub fn create<P: AsRef<Path>>(path: P, chroms: &[Chrom], bin_size: u32) -> Result<Self> {
        let file = File::create(path)?;
        let group = file.group("/")?;
        Self::from_group(group, chroms, bin_size)
    }

    /// Write a cooler collection into an existing HDF5 group.
    pub fn from_group(group: Group, chroms: &[Chrom], bin_size: u32) -> Result<Self> {
        if bin_size == 0 {
            return Err(Error::InvalidInput("bin_size must be positive".into()));
        }
        for chrom in chroms {
            if chrom.length < 0 {
                return Err(Error::InvalidInput(format!(
                    "chromosome '{}' has negative length",
                    chrom.name
                )));
            }
        }

        // Required attributes (schema v3).
        write_attr_str(&group, "format", COOL_FORMAT)?;
        write_attr_int(&group, "format-version", COOL_FORMAT_VERSION)?;
        write_attr_str(&group, "bin-type", "fixed")?;
        write_attr_int(&group, "bin-size", i64::from(bin_size))?;
        write_attr_str(&group, "storage-mode", "symmetric-upper")?;

        // /chroms table.
        let chrom_group = group.create_group("chroms")?;
        write_strings(
            &chrom_group,
            "name",
            &chroms.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        )?;
        chrom_group
            .new_dataset::<i32>()
            .shape(chroms.len())
            .create("length")?
            .write(&chroms.iter().map(|c| c.length).collect::<Vec<_>>())?;

        // /bins table, generated for fixed-size bins.
        let mut bin_chrom: Vec<i32> = Vec::new();
        let mut bin_start: Vec<i32> = Vec::new();
        let mut bin_end: Vec<i32> = Vec::new();
        let mut chrom_offset: Vec<i64> = Vec::with_capacity(chroms.len() + 1);
        chrom_offset.push(0);
        for (chrom_id, chrom) in chroms.iter().enumerate() {
            let n = ((i64::from(chrom.length) + i64::from(bin_size) - 1)
                / i64::from(bin_size)) as i32;
            for i in 0..n {
                let start = i * bin_size as i32;
                bin_chrom.push(chrom_id as i32);
                bin_start.push(start);
                bin_end.push((start + bin_size as i32).min(chrom.length));
            }
            chrom_offset.push(chrom_offset.last().unwrap() + i64::from(n));
        }
        let n_bins = bin_chrom.len() as u64;

        let bin_group = group.create_group("bins")?;
        bin_group
            .new_dataset::<i32>()
            .shape(bin_chrom.len())
            .create("chrom")?
            .write(&bin_chrom)?;
        bin_group
            .new_dataset::<i32>()
            .shape(bin_start.len())
            .create("start")?
            .write(&bin_start)?;
        bin_group
            .new_dataset::<i32>()
            .shape(bin_end.len())
            .create("end")?
            .write(&bin_end)?;

        // /indexes/chrom_offset.
        let index_group = group.create_group("indexes")?;
        index_group
            .new_dataset::<i64>()
            .shape(chrom_offset.len())
            .create("chrom_offset")?
            .write(&chrom_offset)?;

        write_attr_int(&group, "nchroms", chroms.len() as i64)?;
        write_attr_int(&group, "nbins", n_bins as i64)?;

        Ok(CoolerWriter { group, n_bins })
    }

    /// Number of bins in this collection.
    pub fn n_bins(&self) -> u64 {
        self.n_bins
    }

    /// Write the pixel table and the `bin1_offset` index.
    ///
    /// Pixels do not need to be sorted or upper-triangular: they are
    /// normalized (`bin1_id <= bin2_id`) and sorted by `(bin1_id, bin2_id)`
    /// before writing, as required by the `symmetric-upper` storage mode.
    pub fn write_pixels(&self, pixels: &[Pixel]) -> Result<()> {
        let mut pixels = pixels.to_vec();
        for p in &mut pixels {
            if p.bin1_id > p.bin2_id {
                std::mem::swap(&mut p.bin1_id, &mut p.bin2_id);
            }
            if p.bin1_id < 0 || p.bin2_id >= self.n_bins as i64 {
                return Err(Error::InvalidInput(format!(
                    "pixel ({}, {}) out of range for {} bins",
                    p.bin1_id, p.bin2_id, self.n_bins
                )));
            }
        }
        pixels.sort_by_key(|p| (p.bin1_id, p.bin2_id));

        let pixel_group = self.group.create_group("pixels")?;
        pixel_group
            .new_dataset::<i64>()
            .shape(pixels.len())
            .create("bin1_id")?
            .write(&pixels.iter().map(|p| p.bin1_id).collect::<Vec<_>>())?;
        pixel_group
            .new_dataset::<i64>()
            .shape(pixels.len())
            .create("bin2_id")?
            .write(&pixels.iter().map(|p| p.bin2_id).collect::<Vec<_>>())?;
        pixel_group
            .new_dataset::<f64>()
            .shape(pixels.len())
            .create("count")?
            .write(&pixels.iter().map(|p| p.count).collect::<Vec<_>>())?;

        // /indexes/bin1_offset: cumulative counts over bin1_id.
        let mut bin1_offset = vec![0_i64; self.n_bins as usize + 1];
        for p in &pixels {
            bin1_offset[p.bin1_id as usize + 1] += 1;
        }
        for i in 1..bin1_offset.len() {
            bin1_offset[i] += bin1_offset[i - 1];
        }
        self.group
            .group("indexes")?
            .new_dataset::<i64>()
            .shape(bin1_offset.len())
            .create("bin1_offset")?
            .write(&bin1_offset)?;

        write_attr_int(&self.group, "nnz", pixels.len() as i64)?;
        Ok(())
    }
}

/// Reader for a single-resolution cooler collection.
pub struct Cooler {
    group: Group,
    // Keep the file alive when the group is a subgroup (e.g. from .mcool).
    _file: Option<File>,
}

impl Cooler {
    /// Open an existing `.cool` file, validating the `format` attribute.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let group = file.group("/")?;
        let cooler = Cooler {
            group,
            _file: Some(file),
        };
        cooler.check_format()?;
        Ok(cooler)
    }

    /// Open a `.cool` file, falling back to legacy formats.
    ///
    /// Older cooler files may store data under a numeric-named subgroup
    /// (e.g. `/10000/`) instead of at the root level. This method tries
    /// the standard layout first, then searches for numeric subgroups
    /// that contain a `chroms` table.
    pub fn open_any<P: AsRef<Path>>(path: P) -> Result<Self> {
        // Try standard format first.
        match Self::open(path.as_ref()) {
            Ok(cooler) => {
                // Verify the root actually has chroms data; if not, fall through.
                if cooler.group.link_exists("chroms") {
                    return Ok(cooler);
                }
            }
            Err(_) => {}
        }

        // Search for a numeric-named subgroup containing chroms.
        let file = File::open(path.as_ref())?;
        let root = file.group("/")?;
        let members = root.member_names()?;

        for name in &members {
            if name.parse::<u64>().is_ok() {
                if let Ok(grp) = root.group(name) {
                    if grp.link_exists("chroms") {
                        let group = root.group(name)?;
                        return Ok(Cooler {
                            group,
                            _file: Some(file),
                        });
                    }
                }
            }
        }

        Err(Error::Format(format!(
            "no valid cooler collection found in '{}'",
            path.as_ref().display()
        )))
    }

    /// Wrap an existing HDF5 group (e.g. `/resolutions/10000` in an `.mcool`
    /// file) as a cooler collection.
    pub fn from_group(group: Group) -> Result<Self> {
        let cooler = Cooler {
            group,
            _file: None,
        };
        cooler.check_format()?;
        Ok(cooler)
    }

    /// Wrap an HDF5 group while keeping the parent [`File`] alive.
    pub fn from_group_with_file(group: Group, file: File) -> Result<Self> {
        let cooler = Cooler {
            group,
            _file: Some(file),
        };
        cooler.check_format()?;
        Ok(cooler)
    }

    fn check_format(&self) -> Result<()> {
        match self.group.attr("format") {
            Ok(attr) => {
                let format = attr.read_scalar::<VarLenUnicode>()?.to_string();
                if format != COOL_FORMAT {
                    return Err(Error::Format(format!(
                        "expected format '{COOL_FORMAT}', found '{format}'"
                    )));
                }
                Ok(())
            }
            Err(_) => {
                // Older cooler files may lack the format attribute;
                // accept them and proceed.
                Ok(())
            }
        }
    }

    fn read_strings(&self, path: &str) -> Result<Vec<String>> {
        let ds = self.group.dataset(path)?;

        // Try VarLenUnicode first (modern coolers).
        if let Ok(values) = ds.read_1d::<VarLenUnicode>() {
            return Ok(values.iter().map(|v| v.to_string()).collect());
        }
        // Try VarLenAscii.
        if let Ok(values) = ds.read_1d::<hdf5_metno::types::VarLenAscii>() {
            return Ok(values.iter().map(|v| v.to_string()).collect());
        }

        // Handle fixed-length strings: match on common sizes.
        let desc = ds
            .dtype()
            .and_then(|dt| dt.to_descriptor())
            .map_err(|e| Error::Format(format!("cannot inspect type of '{path}': {e}")))?;

        let len = match &desc {
            hdf5_metno::types::TypeDescriptor::FixedAscii(n)
            | hdf5_metno::types::TypeDescriptor::FixedUnicode(n) => *n,
            _ => {
                return Err(Error::Format(format!(
                    "unsupported string type in dataset '{path}': {desc}"
                )));
            }
        };

        // Match on common fixed-string lengths (1-64 covers all chromosome names).
        macro_rules! read_fixed {
            ($n:expr) => {{
                use hdf5_metno::types::FixedAscii;
                let values: Vec<FixedAscii<$n>> = ds.read_1d()?.to_vec();
                values
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
            }};
        }

        Ok(match len {
            1 => read_fixed!(1),
            2 => read_fixed!(2),
            3 => read_fixed!(3),
            4 => read_fixed!(4),
            5 => read_fixed!(5),
            6 => read_fixed!(6),
            7 => read_fixed!(7),
            8 => read_fixed!(8),
            10 => read_fixed!(10),
            12 => read_fixed!(12),
            16 => read_fixed!(16),
            24 => read_fixed!(24),
            32 => read_fixed!(32),
            64 => read_fixed!(64),
            128 => read_fixed!(128),
            256 => read_fixed!(256),
            n => {
                return Err(Error::Format(format!(
                    "unsupported fixed-string length {n} in dataset '{path}'"
                )));
            }
        })
    }

    /// The fixed bin size, if declared in the attributes.
    pub fn bin_size(&self) -> Result<Option<u64>> {
        match self.group.attr("bin-size") {
            Ok(attr) => {
                let v: i64 = attr.read_scalar()?;
                Ok(Some(v as u64))
            }
            Err(_) => Ok(None),
        }
    }

    /// Read the `/chroms` table.
    pub fn chroms(&self) -> Result<Vec<Chrom>> {
        let names = self.read_strings("chroms/name")?;
        let lengths: Vec<i32> = self.group.dataset("chroms/length")?.read_1d()?.to_vec();
        Ok(names
            .into_iter()
            .zip(lengths)
            .map(|(name, length)| Chrom { name, length })
            .collect())
    }

    /// Read the `/bins` table.
    pub fn bins(&self) -> Result<Vec<Bin>> {
        let chrom_id: Vec<i32> = self.group.dataset("bins/chrom")?.read_1d()?.to_vec();
        let start: Vec<i32> = self.group.dataset("bins/start")?.read_1d()?.to_vec();
        let end: Vec<i32> = self.group.dataset("bins/end")?.read_1d()?.to_vec();
        Ok(chrom_id
            .into_iter()
            .zip(start)
            .zip(end)
            .map(|((chrom_id, start), end)| Bin {
                chrom_id,
                start,
                end,
            })
            .collect())
    }

    /// Read the `/pixels` table.
    pub fn pixels(&self) -> Result<Vec<Pixel>> {
        let bin1_id: Vec<i64> = self.group.dataset("pixels/bin1_id")?.read_1d()?.to_vec();
        let bin2_id: Vec<i64> = self.group.dataset("pixels/bin2_id")?.read_1d()?.to_vec();
        let count: Vec<f64> = self.group.dataset("pixels/count")?.read_1d()?.to_vec();
        Ok(bin1_id
            .into_iter()
            .zip(bin2_id)
            .zip(count)
            .map(|((bin1_id, bin2_id), count)| Pixel {
                bin1_id,
                bin2_id,
                count,
            })
            .collect())
    }

    /// Check if a column exists in the bins table.
    pub fn bins_has_column(&self, name: &str) -> Result<bool> {
        let path = format!("bins/{}", name);
        Ok(self.group.link_exists(&path))
    }

    /// Read a float64 attribute on a bins column dataset
    /// (e.g. `scale` on `weight`). Returns None if missing.
    pub fn bins_column_attr_f64(&self, column: &str, attr: &str) -> Result<Option<f64>> {
        let path = format!("bins/{}", column);
        if !self.group.link_exists(&path) {
            return Ok(None);
        }
        let ds = self.group.dataset(&path)?;
        match ds.attr(attr) {
            Ok(a) => Ok(Some(a.read_scalar::<f64>()?)),
            Err(_) => Ok(None),
        }
    }

    /// Read pixels whose `bin1_id` falls in `[first, last)` — one
    /// chromosome's rows. Uses `/indexes/bin1_offset` to slice.
    pub fn pixels_for_bins(&self, first: usize, last: usize) -> Result<Vec<Pixel>> {
        let offsets = self.bin1_offset()?;
        let lo = offsets[first] as usize;
        let hi = offsets[last] as usize;
        let bin1_id: Vec<i64> = self
            .group
            .dataset("pixels/bin1_id")?
            .read_slice_1d(lo..hi)?
            .to_vec();
        let bin2_id: Vec<i64> = self
            .group
            .dataset("pixels/bin2_id")?
            .read_slice_1d(lo..hi)?
            .to_vec();
        let count: Vec<f64> = self
            .group
            .dataset("pixels/count")?
            .read_slice_1d(lo..hi)?
            .to_vec();
        Ok(bin1_id
            .into_iter()
            .zip(bin2_id)
            .zip(count)
            .map(|((bin1_id, bin2_id), count)| Pixel {
                bin1_id,
                bin2_id,
                count,
            })
            .collect())
    }

    /// List all column names in the bins table.
    pub fn bins_column_names(&self) -> Result<Vec<String>> {
        let all = self.group.member_names()?;
        Ok(all
            .into_iter()
            .filter(|n| {
                n.starts_with("bins/")
                    && n != "bins/chrom"
                    && n != "bins/start"
                    && n != "bins/end"
            })
            .map(|n| n.strip_prefix("bins/").unwrap().to_string())
            .collect())
    }

    /// Read a float64 column from the bins table by name.
    /// Returns None if the column does not exist.
    /// The returned vector has length equal to the number of bins.
    pub fn bins_column_f64(&self, name: &str) -> Result<Option<Vec<f64>>> {
        let path = format!("bins/{}", name);
        if !self.group.link_exists(&path) {
            return Ok(None);
        }
        let ds = self.group.dataset(&path)?;
        let values: Vec<f64> = ds.read_1d()?.to_vec();
        Ok(Some(values))
    }

    /// Number of non-zero pixels (`nnz` attribute, or the table length).
    pub fn n_pixels(&self) -> Result<u64> {
        Ok(self.group.dataset("pixels/bin1_id")?.size() as u64)
    }

    /// Read the `/indexes/bin1_offset` table.
    pub fn bin1_offset(&self) -> Result<Vec<i64>> {
        Ok(self.group.dataset("indexes/bin1_offset")?.read_1d()?.to_vec())
    }

    /// Read the `/indexes/chrom_offset` table.
    pub fn chrom_offset(&self) -> Result<Vec<i64>> {
        Ok(self.group.dataset("indexes/chrom_offset")?.read_1d()?.to_vec())
    }
}
