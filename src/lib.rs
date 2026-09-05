//! cooler — read and write `.cool` and `.mcool` (Hi-C contact matrix) files.
//!
//! Implements the [cooler schema](https://cooler.readthedocs.io/en/latest/schema.html)
//! on top of HDF5: single-resolution `.cool` files store a sparse matrix at
//! the file root, while multi-resolution `.mcool` files store one collection
//! per bin size under `/resolutions/<binsize>`.
//!
//! # Writing a `.cool` file
//!
//! ```no_run
//! use cooler_rs::{Chrom, CoolerWriter, Pixel};
//!
//! # fn main() -> cooler_rs::Result<()> {
//! let chroms = vec![
//!     Chrom { name: "chr1".into(), length: 1_000_000 },
//!     Chrom { name: "chr2".into(), length: 500_000 },
//! ];
//! let writer = CoolerWriter::create("out.cool", &chroms, 100_000)?;
//! writer.write_pixels(&[Pixel { bin1_id: 0, bin2_id: 3, count: 42.0 }])?;
//! # Ok(())
//! # }
//! ```
//!
//! # Reading an `.mcool` file
//!
//! ```no_run
//! use cooler_rs::Mcool;
//!
//! # fn main() -> cooler_rs::Result<()> {
//! let mcool = Mcool::open("out.mcool")?;
//! for res in mcool.resolutions()? {
//!     let cool = mcool.cooler(res)?;
//!     println!("{res}: {} pixels", cool.n_pixels()?);
//! }
//! # Ok(())
//! # }
//! ```

pub mod armatus;
pub mod balance;
pub mod convert;
pub mod cooler;
pub mod domaincaller;
pub mod error;
pub mod mcool;
pub mod ontad;
pub mod raichu;
pub mod region;
pub mod stats;
pub mod types;
pub mod zoomify;

pub use armatus::{call_domains, Domain};
pub use balance::{balance_cooler, BalanceParams, BalanceStats};
pub use cooler::{
    delete_bins_column, rename_chroms, write_bins_column, AttrValue, Cooler, CoolerWriter,
    SubMatrix, Validation,
};
pub use error::{Error, Result};
pub use mcool::{Mcool, McoolWriter};
pub use ontad::{Params, Tad};
pub use raichu::{raichu_normalize, RaichuParams};
pub use region::Region;
pub use stats::{DiscreteDistribution, GeneralMixtureModel, HiddenMarkovModel, NormalDistribution};
pub use types::{Bin, Chrom, ChromMeta, Pixel};
pub use zoomify::{
    coarsen_pixels, nice_resolutions, pow2_resolutions, zoomify_cooler, ZoomifyParams,
};
