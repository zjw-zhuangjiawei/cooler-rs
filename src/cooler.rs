//! Reading and writing single-resolution `.cool` files.
//!
//! A cooler collection is a group (the file root for `.cool`, or
//! `/resolutions/<binsize>` for `.mcool`) containing `chroms`, `bins`,
//! `pixels` and `indexes` tables, following schema version 3:
//! <https://cooler.readthedocs.io/en/latest/schema.html>

use std::ops::Range;
use std::path::Path;

use hdf5_metno::types::VarLenUnicode;
use hdf5_metno::{File, Group};
use ndarray::Array2;
use sprs::TriMat;

use crate::error::{Error, Result};
use crate::region::Region;
use crate::types::{Bin, Chrom, Pixel};

/// Value of the `format` attribute for single-resolution files.
pub const COOL_FORMAT: &str = "HDF5::Cooler";
/// Schema version written to the `format-version` attribute.
pub const COOL_FORMAT_VERSION: i64 = 3;

/// A value to store as a dataset attribute (e.g. balancing stats).
#[derive(Debug, Clone)]
pub enum AttrValue {
    /// 64-bit float.
    F64(f64),
    /// 64-bit integer (also used for booleans, since HDF5 has no bool type).
    I64(i64),
    /// 64-bit float array (per-chromosome balancing stats).
    F64Array(Vec<f64>),
}

/// Rectangular submatrix query on the contact heatmap.
///
/// `rows` and `cols` are global bin-id half-open ranges `[lo, hi)` (see
/// [`Cooler::extent`] to derive them from a genomic region). The returned
/// matrix has shape `(rows.len(), cols.len())`, index `0` = bin `rows.start`.
///
/// Symmetric-upper storage: when `fill_lower` is true (default) the logical
/// genome matrix — a reflection of every stored upper-triangle entry — is
/// returned, so `m[[i, j]] == m[[j, i]]` holds for the on-diagonal case. When
/// false, only the stored (upper) entries that fall inside `rows × cols` are
/// placed, mirroring cooler-python's `DirectRangeQuery`.
///
/// `balance` names a `bins` column (e.g. `"weight"`) to multiply each cell by
/// `w[i] * w[j]`. A missing column is an error, not silent.
#[derive(Debug, Clone)]
pub struct SubMatrix {
    /// Row (bin1) range.
    pub rows: Range<i64>,
    /// Column (bin2) range.
    pub cols: Range<i64>,
    /// Mirror symmetric-upper entries into the other triangle. Default true.
    pub fill_lower: bool,
    /// Name of a bins column to scale values by (multiplicative). Default raw.
    pub balance: Option<String>,
}

impl SubMatrix {
    /// Square query: rows == cols.
    pub fn square(bins: Range<i64>) -> Self {
        let cols = bins.clone();
        SubMatrix {
            rows: bins,
            cols,
            fill_lower: true,
            balance: None,
        }
    }

    /// Rectangular query, with lower-triangle filling enabled.
    pub fn rect(rows: Range<i64>, cols: Range<i64>) -> Self {
        SubMatrix {
            rows,
            cols,
            fill_lower: true,
            balance: None,
        }
    }
}

/// Integrity report from [`Cooler::validate`].
///
/// `issues` lists every problem found; an empty vector means the file is
/// internally consistent. Structural problems (a missing dataset, an
/// undecodable dtype) are reported as issues too, so `validate` only fails
/// when the file cannot be opened at all.
#[derive(Debug, Default)]
pub struct Validation {
    /// Human-readable problems, empty when the file is valid.
    pub issues: Vec<String>,
}

impl Validation {
    /// True when no problems were found.
    pub fn is_ok(&self) -> bool {
        self.issues.is_empty()
    }
}

/// Open an existing file read-write and resolve the collection group
/// (`/` for a `.cool`, `/resolutions/<bin_size>` for an `.mcool`).
fn open_group_rw<P: AsRef<Path>>(path: P, group_path: &str) -> Result<(File, Group)> {
    let file = File::open_rw(path)?;
    let group = if group_path == "/" {
        file.group("/")?
    } else {
        file.group(group_path)?
    };
    Ok((file, group))
}

/// Add a float64 column to the `bins` table of an existing cooler file,
/// overwriting any existing column of that name, and attach metadata attrs.
///
/// `group_path` is `/` for a `.cool`, or `/resolutions/<bin_size>` for an
/// `.mcool`.
pub fn write_bins_column<P: AsRef<Path>>(
    path: P,
    group_path: &str,
    name: &str,
    data: &[f64],
    attrs: &[(&str, AttrValue)],
) -> Result<()> {
    let (_file, group) = open_group_rw(path, group_path)?;
    let bins = group.group("bins")?;
    if bins.link_exists(name) {
        bins.unlink(name)?;
    }
    let ds = bins
        .new_dataset::<f64>()
        .shape(data.len())
        .deflate(6)
        .create(name)?;
    ds.write(data)?;
    for (attr_name, value) in attrs {
        match value {
            AttrValue::F64(v) => ds.new_attr::<f64>().create(*attr_name)?.write_scalar(v)?,
            AttrValue::I64(v) => ds.new_attr::<i64>().create(*attr_name)?.write_scalar(v)?,
            AttrValue::F64Array(v) => ds
                .new_attr::<f64>()
                .shape(v.len())
                .create(*attr_name)?
                .write(v)?,
        }
    }
    Ok(())
}

/// Rename chromosome/contig entries of an existing cooler collection in
/// place, mapping each `(old, new)` name. Names not listed are kept as is.
///
/// Bin ids and coordinates are untouched — this rewrites only the
/// `chroms/name` strings (the chrom-to-bin mapping is by row index, not by
/// name), so the pixel table and indexes stay valid. The bins/chrom column
/// must hold plain integer codes; cooler-python files that store an HDF5
/// enum mapping here are refused, because renaming without rebuilding that
/// mapping would corrupt bin decoding.
///
/// Errors on an unknown old name or on a rename that would leave duplicate
/// (or empty) names. `group_path` is `/` for a `.cool`, or
/// `/resolutions/<bin_size>` for an `.mcool`.
pub fn rename_chroms<P: AsRef<Path>>(
    path: P,
    group_path: &str,
    renames: &[(&str, &str)],
) -> Result<()> {
    if renames.is_empty() {
        return Ok(());
    }
    for (old, new) in renames {
        if new.is_empty() {
            return Err(Error::InvalidInput(format!(
                "rename of '{old}' to an empty name"
            )));
        }
    }
    let (_file, group) = open_group_rw(path, group_path)?;

    let cool = Cooler::from_group(group.clone())?;
    let current = cool.chroms()?;

    for (old, _) in renames {
        if !current.iter().any(|c| c.name == *old) {
            return Err(Error::InvalidInput(format!(
                "cannot rename unknown chromosome '{old}'"
            )));
        }
    }

    let out: Vec<String> = current
        .iter()
        .map(|c| {
            renames
                .iter()
                .find(|(old, _)| *old == c.name)
                .map(|(_, new)| new.to_string())
                .unwrap_or_else(|| c.name.clone())
        })
        .collect();
    let mut seen = std::collections::HashSet::new();
    for name in &out {
        if !seen.insert(name.clone()) {
            return Err(Error::InvalidInput(format!(
                "rename would produce duplicate chromosome name '{name}'"
            )));
        }
    }

    // Guard against py-cooler's enum-typed bins/chrom: codes would decode
    // through the embedded name mapping, which we are not rebuilding.
    let bins = group.group("bins")?;
    let chrom_ds = bins.dataset("chrom")?;
    if chrom_ds.size() > 0 {
        chrom_ds.read_slice_1d::<i32, _>(0..1).map_err(|_| {
            Error::Format(
                "'bins/chrom' is not a plain int-code column (e.g. py-cooler \
                 enum dtype); in-place rename would corrupt bin decoding"
                    .into(),
            )
        })?;
    }

    // Rewrite the name column (bucket size may change with new lengths).
    let chroms = group.group("chroms")?;
    if chroms.link_exists("name") {
        chroms.unlink("name")?;
    }
    write_strings(&chroms, "name", &out)?;
    Ok(())
}

/// Remove a column from the `bins` table of an existing cooler file.
///
/// The required `chrom`/`start`/`end` columns cannot be removed; deleting a
/// column that does not exist is an error. `group_path` is `/` for a
/// `.cool`, or `/resolutions/<bin_size>` for an `.mcool`.
pub fn delete_bins_column<P: AsRef<Path>>(path: P, group_path: &str, name: &str) -> Result<()> {
    if matches!(name, "chrom" | "start" | "end") {
        return Err(Error::InvalidInput(format!(
            "'{name}' is a required bins column; cannot delete"
        )));
    }
    let (_file, group) = open_group_rw(path, group_path)?;
    let path = format!("bins/{name}");
    if !group.link_exists(&path) {
        return Err(Error::InvalidInput(format!("no '{path}' column to delete")));
    }
    let bins = group.group("bins")?;
    bins.unlink(name)?;
    Ok(())
}

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
            let n =
                ((i64::from(chrom.length) + i64::from(bin_size) - 1) / i64::from(bin_size)) as i32;
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
        // Try standard format first; fall through if the root has no
        // chroms data.
        if let Ok(cooler) = Self::open(path.as_ref()) {
            if cooler.group.link_exists("chroms") {
                return Ok(cooler);
            }
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
        let cooler = Cooler { group, _file: None };
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
                values.iter().map(|v| v.to_string()).collect::<Vec<_>>()
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

    /// Read a contiguous slice of the pixel table by stored row position —
    /// one chunk of a streaming pass over all pixels. Rows come back in
    /// stored order.
    pub fn pixels_range(&self, lo: i64, hi: i64) -> Result<Vec<Pixel>> {
        let lo = lo as usize;
        let hi = hi as usize;
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

    /// Read the `bins/chrom` column (chromosome id per bin).
    pub fn bin_chrom(&self) -> Result<Vec<i32>> {
        Ok(self.group.dataset("bins/chrom")?.read_1d()?.to_vec())
    }

    /// Read pixels whose `bin1_id` falls in `[first, last)` — one
    /// chromosome's rows. Uses `/indexes/bin1_offset` to slice.
    pub fn pixels_for_bins(&self, first: i64, last: i64) -> Result<Vec<Pixel>> {
        let offsets = self.bin1_offset()?;
        let lo = offsets[first as usize] as usize;
        let hi = offsets[last as usize] as usize;
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
                n.starts_with("bins/") && n != "bins/chrom" && n != "bins/start" && n != "bins/end"
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
        Ok(self
            .group
            .dataset("indexes/bin1_offset")?
            .read_1d()?
            .to_vec())
    }

    /// Read the `/indexes/chrom_offset` table.
    pub fn chrom_offset(&self) -> Result<Vec<i64>> {
        Ok(self
            .group
            .dataset("indexes/chrom_offset")?
            .read_1d()?
            .to_vec())
    }

    /// Check the file for internal consistency (schema + index invariants)
    /// and return every problem found as a [`Validation`].
    ///
    /// Verifies in a single pass that: the required tables/columns exist and
    /// match the declared `nchroms`/`nbins`/`nnz` attributes; `chrom_offset`
    /// and `bin1_offset` are monotone and partition the bins and pixel rows
    /// exactly; every bin/chrom code matches its offset band and its
    /// coordinates stay inside the chromosome with no overlaps; and every
    /// pixel is symmetric-upper (`bin1 <= bin2`), in range, finite, and
    /// stored grouped by `bin1_id` exactly as the row offsets say.
    ///
    /// Assumes the storage the rest of the crate reads: plain integer
    /// `bins/chrom` codes (not py-cooler's enum dtype) and `symmetric-upper`
    /// storage. Larger tables are streamed in chunks, so memory stays
    /// bounded regardless of file size.
    pub fn validate(&self) -> Result<Validation> {
        let mut v = Validation::default();
        let issues = &mut v.issues;
        let group = &self.group;

        macro_rules! read_col {
            ($ty:ty, $path:expr) => {{
                let p = $path;
                match group.dataset(p) {
                    Ok(ds) => match ds.read_1d::<$ty>() {
                        Ok(x) => Some(x.to_vec()),
                        Err(e) => {
                            issues.push(format!("cannot decode '{p}' as {}: {e}", stringify!($ty)));
                            None
                        }
                    },
                    Err(e) => {
                        issues.push(format!("missing dataset '{p}': {e}"));
                        None
                    }
                }
            }};
        }
        let attr_i64 = |name: &str| -> Option<i64> {
            group
                .attr(name)
                .ok()
                .and_then(|a| a.read_scalar::<i64>().ok())
        };

        // --- chroms ------------------------------------------------------
        let names = match self.read_strings("chroms/name") {
            Ok(n) => Some(n),
            Err(e) => {
                issues.push(format!("cannot decode 'chroms/name': {e}"));
                None
            }
        };
        let lengths = read_col!(i32, "chroms/length");
        let clens: Option<&Vec<i32>> = match (&names, &lengths) {
            (Some(n), Some(l)) if n.len() == l.len() => Some(l),
            _ => None,
        };
        let nchroms: Option<i64> = names.as_ref().map(|n| n.len() as i64);
        if let (Some(n), Some(l)) = (&names, &lengths) {
            if n.len() != l.len() {
                issues.push(format!(
                    "'chroms/name' and 'chroms/length' have different lengths ({} vs {})",
                    n.len(),
                    l.len()
                ));
            }
            let mut seen = std::collections::HashSet::new();
            for (name, &len) in n.iter().zip(l.iter()) {
                if len < 0 {
                    issues.push(format!("chromosome '{name}' has negative length {len}"));
                }
                if !seen.insert(name) {
                    issues.push(format!("duplicate chromosome name '{name}'"));
                }
            }
            if n.is_empty() {
                issues.push("chroms table is empty".into());
            }
        }
        if let (Some(a), Some(n)) = (attr_i64("nchroms"), nchroms) {
            if a != n {
                issues.push(format!(
                    "'nchroms' attribute {a} != chroms table length {n}"
                ));
            }
        }

        // --- bins --------------------------------------------------------
        let bins_chrom = read_col!(i32, "bins/chrom");
        let bin_starts = read_col!(i32, "bins/start");
        let bin_ends = read_col!(i32, "bins/end");
        let nbins: Option<i64> = match (&bins_chrom, &bin_starts, &bin_ends) {
            (Some(a), Some(b), Some(c)) => {
                if b.len() == a.len() && c.len() == a.len() {
                    Some(a.len() as i64)
                } else {
                    issues.push(format!(
                        "bins column lengths differ: chrom {}, start {}, end {}",
                        a.len(),
                        b.len(),
                        c.len()
                    ));
                    None
                }
            }
            _ => None,
        };
        if let (Some(a), Some(n)) = (attr_i64("nbins"), nbins) {
            if a != n {
                issues.push(format!("'nbins' attribute {a} != bins table length {n}"));
            }
        }

        // Chrom codes in range, and exactly matching the chrom_offset bands.
        if let (Some(bc), Some(nch)) = (&bins_chrom, nchroms) {
            for (k, &code) in bc.iter().enumerate() {
                if code < 0 || i64::from(code) >= nch {
                    issues.push(format!(
                        "bin {k}: chrom code {code} out of range (nchroms={nch})"
                    ));
                }
            }
        }
        let chrom_offset = read_col!(i64, "indexes/chrom_offset");
        if let (Some(bc), Some(o), Some(nch)) = (&bins_chrom, &chrom_offset, nchroms) {
            let n = nch as usize;
            if o.len() != n + 1 {
                issues.push(format!(
                    "'chrom_offset' has {} entries, expected nchroms+1 = {}",
                    o.len(),
                    n + 1
                ));
            } else if o[0] != 0 || *o.last().unwrap() != bc.len() as i64 {
                issues.push(format!(
                    "'chrom_offset' spans {}..={}, expected 0..={}",
                    o[0],
                    o.last().unwrap(),
                    bc.len()
                ));
            } else if !o.windows(2).all(|w| w[1] >= w[0]) {
                issues.push("'chrom_offset' is not monotone".into());
            } else {
                let mut c = 0usize;
                for (k, &code) in bc.iter().enumerate() {
                    while c < n && (k as i64) >= o[c + 1] {
                        c += 1;
                    }
                    if i64::from(code) != c as i64 {
                        issues.push(format!(
                            "bin {k}: chrom code {code} does not match chrom_offset band {c}"
                        ));
                    }
                }
            }
        }

        // Bin coordinates: valid intervals, inside the chromosome, sorted.
        if let (Some(bc), Some(bs), Some(be)) = (&bins_chrom, &bin_starts, &bin_ends) {
            let mut prev: Option<(i32, i32)> = None; // (chrom, start)
            for (k, (&code, (&s, &e))) in bc.iter().zip(bs.iter().zip(be.iter())).enumerate() {
                if s < 0 || e <= s {
                    issues.push(format!("bin {k}: empty or invalid interval [{s}, {e})"));
                }
                if let Some(clens) = clens {
                    if (0..clens.len() as i32).contains(&code) && e > clens[code as usize] {
                        issues.push(format!(
                            "bin {k}: end {e} past chromosome {code} length {}",
                            clens[code as usize]
                        ));
                    }
                }
                if let Some((pc, ps)) = prev {
                    if pc == code && s <= ps {
                        issues.push(format!(
                            "bin {k}: interval [{s}, {e}) overlaps or duplicates the \
                             previous bin of chromosome {code}"
                        ));
                    }
                }
                prev = Some((code, s));
            }
        }

        // --- pixels ------------------------------------------------------
        let nnz = match group.dataset("pixels/bin1_id") {
            Ok(ds) => Some(ds.size() as u64),
            Err(e) => {
                issues.push(format!("missing dataset 'pixels/bin1_id': {e}"));
                None
            }
        };
        if let (Some(a), Some(n)) = (attr_i64("nnz"), nnz) {
            if a as u64 != n {
                issues.push(format!("'nnz' attribute {a} != pixel table length {n}"));
            }
        }
        if let Some(n) = nnz {
            for col in ["pixels/bin2_id", "pixels/count"] {
                match group.dataset(col) {
                    Ok(ds) if ds.size() as u64 == n => {}
                    Ok(ds) => issues.push(format!(
                        "'{col}' has {} rows, expected {n} pixels",
                        ds.size()
                    )),
                    Err(e) => issues.push(format!("missing dataset '{col}': {e}")),
                }
            }
        }

        let bin1_offset = read_col!(i64, "indexes/bin1_offset");
        if let (Some(nnz), Some(o1), Some(nb)) = (nnz, bin1_offset, nbins) {
            let n = nb as usize;
            if o1.len() != n + 1 {
                issues.push(format!(
                    "'bin1_offset' has {} entries, expected nbins+1 = {}",
                    o1.len(),
                    n + 1
                ));
            } else if o1[0] != 0 || *o1.last().unwrap() != nnz as i64 {
                issues.push(format!(
                    "'bin1_offset' spans {}..={}, expected 0..={nnz}",
                    o1[0],
                    o1.last().unwrap()
                ));
            } else if !o1.windows(2).all(|w| w[1] >= w[0]) {
                issues.push("'bin1_offset' is not monotone".into());
            } else {
                // Stream rows, checking each row's bin1_id against the offset
                // group its position falls into (implies grouped + sorted).
                let step = 1_000_000usize;
                let mut c = 0usize;
                'rows: for lo in (0..nnz as usize).step_by(step) {
                    let hi = (lo + step).min(nnz as usize);
                    let rows = match self.pixels_range(lo as i64, hi as i64) {
                        Ok(r) => r,
                        Err(e) => {
                            issues.push(format!("cannot read pixel rows [{lo}, {hi}): {e}"));
                            break 'rows;
                        }
                    };
                    for (i, row) in rows.iter().enumerate() {
                        let pos = lo as i64 + i as i64;
                        while c < n && pos >= o1[c + 1] {
                            c += 1;
                        }
                        if row.bin1_id != c as i64 {
                            issues.push(format!(
                                "pixel row {pos}: bin1_id {} does not match bin1_offset group {c}",
                                row.bin1_id
                            ));
                        }
                        if row.bin1_id > row.bin2_id {
                            issues.push(format!(
                                "pixel row {pos}: bin1_id {} > bin2_id {} (not symmetric-upper)",
                                row.bin1_id, row.bin2_id
                            ));
                        }
                        if row.bin1_id < 0
                            || row.bin1_id >= nb
                            || row.bin2_id < 0
                            || row.bin2_id >= nb
                        {
                            issues.push(format!(
                                "pixel row {pos}: bin ids ({}, {}) out of range for {n} bins",
                                row.bin1_id, row.bin2_id
                            ));
                        }
                        if !row.count.is_finite() {
                            issues.push(format!("pixel row {pos}: non-finite count {}", row.count));
                        }
                    }
                }
            }
        }

        Ok(v)
    }

    /// Global bin id containing the left end of a genomic region (the first
    /// bin of an open-ended or whole-chromosome region). Equals the first
    /// element of [`Cooler::extent`].
    pub fn offset(&self, region: &Region) -> Result<i64> {
        Ok(self.extent(region)?.0)
    }

    /// Bin-id range `[i0, i1)` covered by a genomic region, in global bin ids
    /// across the whole collection (`i1` exclusive).
    ///
    /// The ids slice the bin table (`bins[i0..i1]`) and the pixel row index
    /// (`bin1_offset[i0..i1]`), the latter giving the pixel rows whose
    /// `bin1_id` falls in the range. Bins that overlap the half-open interval
    /// are included; no boundary alignment is required.
    ///
    /// Regions are validated, not clamped: an unknown chromosome, an `end`
    /// past the chromosome length, or `end < start` returns
    /// [`Error::InvalidInput`]. A whole-chromosome region needs no bin size
    /// and works for any file (variable bins included); a partial interval
    /// requires a fixed `bin-size`.
    pub fn extent(&self, region: &Region) -> Result<(i64, i64)> {
        let chroms = self.chroms()?;
        // TODO(cache): chroms/chrom_offset/bin-size are re-read per query.
        // Cache them once region queries become a hot loop.
        let Some((cid, clen)) = chroms
            .iter()
            .enumerate()
            .find(|(_, c)| c.name == region.chrom)
            .map(|(cid, c)| (cid, c.length))
        else {
            return Err(Error::InvalidInput(format!(
                "unknown sequence label: {}",
                region.chrom
            )));
        };
        if clen < 0 {
            return Err(Error::Format(format!(
                "chromosome '{}' has negative length {clen}",
                region.chrom
            )));
        }
        let clen = clen as u64;
        let start = region.start.unwrap_or(0);
        let end = region.end.unwrap_or(clen);
        if end < start {
            return Err(Error::InvalidInput(format!(
                "region out of bounds on '{}' (length {clen}): end {end} < start {start}",
                region.chrom
            )));
        }
        if end > clen {
            return Err(Error::InvalidInput(format!(
                "region out of bounds on '{}' (length {clen}): [{start}, {end})",
                region.chrom
            )));
        }

        let chrom_offset = self.chrom_offset()?;
        let c0 = chrom_offset[cid];
        let c1 = chrom_offset[cid + 1];

        // Whole chromosome: offset table alone suffices, so this also works
        // for files without a bin size (variable bins).
        if start == 0 && end == clen {
            return Ok((c0, c1));
        }

        // Partial interval: fixed-bin arithmetic only.
        let Some(bin_size) = self.bin_size()? else {
            // TODO(variable-bins): coordinate → bin needs a searchsorted pass
            // over the chrom's bins/start slice (cf. cooler-python
            // `core/_rangequery.py`); add when a variable-bins cooler needs
            // region queries.
            return Err(Error::Format(format!(
                "region query on '{}' needs a fixed bin-size, but this file has none (variable bins)",
                region.chrom
            )));
        };
        if bin_size == 0 {
            return Err(Error::Format("invalid file: bin-size is 0".into()));
        }
        let bs = bin_size;
        let i0 = c0 + (start / bs) as i64;
        let i1 = if end == start {
            i0 // empty interval (e.g. a boundary point): no bins in range
        } else {
            c0 + end.div_ceil(bs) as i64
        };
        Ok((i0, i1))
    }

    /// Read bin-table rows `[lo, hi)` (stored row order).
    pub fn bins_slice(&self, lo: i64, hi: i64) -> Result<Vec<Bin>> {
        let (lo, hi) = (lo as usize, hi as usize);
        let chrom_id: Vec<i32> = self
            .group
            .dataset("bins/chrom")?
            .read_slice_1d(lo..hi)?
            .to_vec();
        let start: Vec<i32> = self
            .group
            .dataset("bins/start")?
            .read_slice_1d(lo..hi)?
            .to_vec();
        let end: Vec<i32> = self
            .group
            .dataset("bins/end")?
            .read_slice_1d(lo..hi)?
            .to_vec();
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

    /// Read the bins overlapping `region` (see [`Cooler::extent`]).
    pub fn bins_in(&self, region: &Region) -> Result<Vec<Bin>> {
        let (lo, hi) = self.extent(region)?;
        self.bins_slice(lo, hi)
    }

    /// Read the stored pixels whose row (`bin1_id`) falls in `region` — the
    /// lower-bound side of the region's band.
    ///
    /// Because pixels are stored symmetric-upper (`bin1_id <= bin2_id`), this
    /// returns every row whose first bin is inside the region, including
    /// cross-chromosome entries whose `bin2_id` lies beyond it. Dropping
    /// those (keeping `bin2_id < extent.end`) leaves the intra-region
    /// upper-triangle.
    pub fn pixels_in(&self, region: &Region) -> Result<Vec<Pixel>> {
        let (lo, hi) = self.extent(region)?;
        self.pixels_for_bins(lo, hi)
    }

    /// Fetch the stored, de-duplicated non-zero cells of the `rows × cols`
    /// submatrix as global `(row_bin, col_bin, value)` triples, optionally
    /// reflected into the other triangle.
    fn submatrix_cells(&self, q: &SubMatrix) -> Result<Vec<(i64, i64, f64)>> {
        let nbins = self.group.dataset("bins/chrom")?.size() as i64;
        for (name, r) in [("rows", &q.rows), ("cols", &q.cols)] {
            if r.start < 0 || r.end > nbins || r.start >= r.end {
                return Err(Error::InvalidInput(format!(
                    "matrix {name} range {:?} out of bounds for {nbins} bins",
                    r.start..r.end
                )));
            }
        }
        let r0 = q.rows.start;
        let r1 = q.rows.end;
        let c0 = q.cols.start;
        let c1 = q.cols.end;

        // Symmetric-upper: a stored entry (a, b) with a <= b contributes the
        // logical cell (a, b) whenever a ∈ rows & b ∈ cols, and its reflection
        // (b, a) whenever b ∈ rows & a ∈ cols. Reading rows-col and cols-row
        // bands covers both; overlapping ranges duplicate cells, so results
        // are sorted and de-duplicated below.
        let mut cells: Vec<(i64, i64, f64)> = Vec::new();
        let collect = |cells: &mut Vec<(i64, i64, f64)>, p: &Pixel| {
            let (a, b) = (p.bin1_id, p.bin2_id);
            if (r0..r1).contains(&a) && (c0..c1).contains(&b) {
                cells.push((a, b, p.count));
            }
            if q.fill_lower && (r0..r1).contains(&b) && (c0..c1).contains(&a) {
                cells.push((b, a, p.count));
            }
        };
        for p in self.pixels_for_bins(r0, r1)? {
            if (c0..c1).contains(&p.bin2_id) {
                collect(&mut cells, &p);
            }
        }
        if q.fill_lower {
            for p in self.pixels_for_bins(c0, c1)? {
                if (r0..r1).contains(&p.bin2_id) {
                    collect(&mut cells, &p);
                }
            }
        }

        cells.sort_unstable_by_key(|c| (c.0, c.1));
        cells.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

        if let Some(name) = &q.balance {
            let w = self.bins_column_f64(name)?.ok_or_else(|| {
                Error::InvalidInput(format!("no 'bins/{name}' column for balance"))
            })?;
            for c in &mut cells {
                c.2 *= w[c.0 as usize] * w[c.1 as usize];
            }
        }
        Ok(cells)
    }

    /// Fetch a dense 2D submatrix over `rows × cols` (see [`SubMatrix`]).
    /// Shape `(nr, nc)`; cells with no stored (or reflected) entry are 0.0.
    pub fn matrix_dense(&self, q: &SubMatrix) -> Result<Array2<f64>> {
        let (nr, nc) = (q.rows.end - q.rows.start, q.cols.end - q.cols.start);
        let mut out = Array2::zeros((nr as usize, nc as usize));
        for (i, j, v) in self.submatrix_cells(q)? {
            out[[(i - q.rows.start) as usize, (j - q.cols.start) as usize]] = v;
        }
        Ok(out)
    }

    /// Fetch the same submatrix in sparse coordinate form ([`sprs::TriMat`],
    /// the equivalent of cooler-python's COO). Local coordinates, shape
    /// `(nr, nc)`.
    pub fn matrix_sparse(&self, q: &SubMatrix) -> Result<TriMat<f64>> {
        let (nr, nc) = (q.rows.end - q.rows.start, q.cols.end - q.cols.start);
        let mut tri = TriMat::new((nr as usize, nc as usize));
        for (i, j, v) in self.submatrix_cells(q)? {
            tri.add_triplet((i - q.rows.start) as usize, (j - q.cols.start) as usize, v);
        }
        Ok(tri)
    }
}
