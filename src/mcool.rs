//! Reading and writing multi-resolution `.mcool` files.
//!
//! An `.mcool` file stores one full cooler collection per resolution under
//! `/resolutions/<binsize>`, with root attribute `format = "HDF5::MCOOL"`.

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

/// Reader for `.mcool` files.
pub struct Mcool {
    file: File,
}

impl Mcool {
    /// Open an existing `.mcool` file, validating the `format` attribute.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let root = file.group("/")?;
        match root.attr("format") {
            Ok(attr) => {
                let format = attr
                    .read_scalar::<hdf5_metno::types::VarLenUnicode>()?
                    .to_string();
                if format != MCOOL_FORMAT {
                    return Err(Error::Format(format!(
                        "expected format '{MCOOL_FORMAT}', found '{format}'"
                    )));
                }
            }
            Err(_) => return Err(Error::Format("missing 'format' attribute".into())),
        }
        Ok(Mcool { file })
    }

    /// List the available resolutions (bin sizes), sorted ascending.
    pub fn resolutions(&self) -> Result<Vec<u64>> {
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
        let path = format!("{RESOLUTIONS_GROUP}/{bin_size}");
        let group = self
            .file
            .group(&path)
            .map_err(|_| Error::Format(format!("resolution {bin_size} not found")))?;
        Cooler::from_group(group)
    }
}
