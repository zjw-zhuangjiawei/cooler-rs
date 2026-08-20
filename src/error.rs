//! Error types for the crate.

/// Errors that can occur in this crate.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// An error from the underlying HDF5 library.
    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5_metno::Error),
    /// An I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The file does not conform to the cooler schema.
    #[error("invalid cooler file: {0}")]
    Format(String),
    /// Invalid input data (e.g. out-of-range bin ids, zero bin size).
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

/// A convenient alias for results returned by this crate.
pub type Result<T> = std::result::Result<T, Error>;
