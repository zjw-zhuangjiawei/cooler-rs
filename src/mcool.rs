//! Reading and writing multi-resolution `.mcool` files.
//!
//! An `.mcool` file stores one full cooler collection per resolution under
//! `/resolutions/<binsize>`, with root attribute `format = "HDF5::MCOOL"`.
//!
//! Schema v1 (cooler < 0.8, or `cooler zoomify --legacy`) instead stores one
//! collection per *zoom level* as a root-level group named `"0"`, `"1"`, ...,
//! and carries no root `format` attribute. Both layouts are read here.

use std::path::Path;

use hdf5_metno::File;

use crate::cooler::{Cooler, CoolerWriter};
use crate::error::{Error, Result};
use crate::types::Chrom;

/// Value of the `format` attribute for multi-resolution files.
pub const MCOOL_FORMAT: &str = "HDF5::MCOOL";
/// Schema version written to the `format-version` attribute.
pub const MCOOL_FORMAT_VERSION: i64 = 2;

const RESOLUTIONS_GROUP: &str = "resolutions";

/// Writer for `.mcool` files.
pub struct McoolWriter {
    file: File,
}

impl McoolWriter {
    /// Create a new `.mcool` file (overwriting any existing file) with the
    /// root attributes and an empty `/resolutions` group.
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path)?;

        let root = file.group("/")?;
        root.new_attr::<hdf5_metno::types::VarLenUnicode>()
            .create("format")?
            .write_scalar(
                &MCOOL_FORMAT
                    .parse::<hdf5_metno::types::VarLenUnicode>()
                    .expect("valid UTF-8"),
            )?;
        root.new_attr::<i64>()
            .create("format-version")?
            .write_scalar(&MCOOL_FORMAT_VERSION)?;
        root.new_attr::<hdf5_metno::types::VarLenUnicode>()
            .create("bin-type")?
            .write_scalar(
                &"fixed"
                    .parse::<hdf5_metno::types::VarLenUnicode>()
                    .expect("valid UTF-8"),
            )?;

        file.create_group(RESOLUTIONS_GROUP)?;
        Ok(McoolWriter { file })
    }

    /// Add a new resolution (`/resolutions/<bin_size>`) and write its
    /// chromosome and bin tables. Returns a [`CoolerWriter`] that can be
    /// used to append pixels.
    pub fn create_cooler(&self, chroms: &[Chrom], bin_size: u32) -> Result<CoolerWriter> {
        let resolutions = self.file.group(RESOLUTIONS_GROUP)?;
        let name = bin_size.to_string();
        if resolutions.link_exists(&name) {
            return Err(Error::InvalidInput(format!(
                "resolution {bin_size} already exists"
            )));
        }
        let group = resolutions.create_group(&name)?;
        CoolerWriter::from_group(group, chroms, bin_size)
    }
}

/// Legacy (schema v1) layout: one cooler per zoom level as a root-level group
/// named `"0"`, `"1"`, ... The group name is a zoom *level*, not a bin size —
/// the resolution is each group's own `bin-size` attribute. Ordered coarsest
/// (level `0`) to base.
///
/// Returns `(bin_size, group name)` sorted by resolution.
fn legacy_layout(file: &File) -> Result<Vec<(u64, String)>> {
    let root = file.group("/")?;
    let mut out = Vec::new();
    for name in root.member_names()? {
        if name.parse::<u64>().is_err() {
            continue;
        }
        let Ok(group) = root.group(&name) else {
            continue;
        };
        if !(group.link_exists("bins") && group.link_exists("pixels")) {
            continue;
        }
        let Ok(attr) = group.attr("bin-size") else {
            continue;
        };
        out.push((attr.read_scalar::<i64>()? as u64, name));
    }
    out.sort_unstable();
    Ok(out)
}

/// Reader for `.mcool` files.
pub struct Mcool {
    file: File,
    /// Schema v1 layout (see [`legacy_layout`]) instead of
    /// `/resolutions/<binsize>`.
    legacy: bool,
}

impl Mcool {
    /// Open an existing `.mcool` file, validating the `format` attribute.
    ///
    /// Schema v1 files carry no `format` attribute; they are recognised by
    /// their root-level zoom-level groups.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let root = file.group("/")?;
        if let Ok(attr) = root.attr("format") {
            let format = attr
                .read_scalar::<hdf5_metno::types::VarLenUnicode>()?
                .to_string();
            if format != MCOOL_FORMAT {
                return Err(Error::Format(format!(
                    "expected format '{MCOOL_FORMAT}', found '{format}'"
                )));
            }
        } else if !legacy_layout(&file)?.is_empty() {
            return Ok(Mcool { file, legacy: true });
        } else {
            return Err(Error::Format("missing 'format' attribute".into()));
        }
        Ok(Mcool {
            file,
            legacy: false,
        })
    }

    /// List the available resolutions (bin sizes), sorted ascending.
    pub fn resolutions(&self) -> Result<Vec<u64>> {
        if self.legacy {
            return Ok(legacy_layout(&self.file)?
                .into_iter()
                .map(|(bin_size, _)| bin_size)
                .collect());
        }
        let group = self.file.group(RESOLUTIONS_GROUP)?;
        let mut resolutions = Vec::new();
        for name in group.member_names()? {
            let res: u64 = name
                .parse()
                .map_err(|_| Error::Format(format!("non-numeric resolution group '{name}'")))?;
            resolutions.push(res);
        }
        resolutions.sort_unstable();
        Ok(resolutions)
    }

    /// Open the cooler collection for a given resolution.
    pub fn cooler(&self, bin_size: u64) -> Result<Cooler> {
        let path = if self.legacy {
            let name = legacy_layout(&self.file)?
                .into_iter()
                .find(|(bs, _)| *bs == bin_size)
                .map(|(_, name)| name)
                .ok_or_else(|| Error::Format(format!("resolution {bin_size} not found")))?;
            name
        } else {
            format!("{RESOLUTIONS_GROUP}/{bin_size}")
        };
        let group = self
            .file
            .group(&path)
            .map_err(|_| Error::Format(format!("resolution {bin_size} not found")))?;
        Cooler::from_group(group)
    }
}
