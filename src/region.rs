//! Genomic region parsing and typing for coordinate → bin queries
//! ([`Cooler::offset`], [`Cooler::extent`]).
//!
//! Mirrors cooler-python's region convention: a region is a 0-based,
//! half-open interval `[start, end)` on a single chromosome. Coordinates are
//! **strict** decimal integers — no thousand separators, no `k`/`M`/`G` unit
//! suffixes (unlike cooler-python's lenient parser).

use std::str::FromStr;

use crate::error::{Error, Result};

/// A genomic region on one chromosome: 0-based, half-open `[start, end)`.
///
/// Fields are optional so a whole-chromosome or open-ended region can be
/// expressed without knowing the chromosome length up front:
///
/// - `start = None` means `0`; `end = None` means the chromosome's full
///   length (resolved against the file's `/chroms` table at query time).
///
/// Construct with [`Region::chrom`] / [`Region::range`], parse a UCSC-style
/// string with [`Region::parse`] / [`FromStr`]:
///
/// - `"chr1"` — whole chromosome
/// - `"chr1:1000-2000"` — bounded interval
/// - `"chr1:1000-"` — open-ended (from 1000 to the end of the chromosome)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Region {
    /// Chromosome (or contig) name, e.g. `"chr1"`.
    pub chrom: String,
    /// Interval start; `None` means the start of the chromosome.
    pub start: Option<u64>,
    /// Interval end (exclusive); `None` means the end of the chromosome.
    pub end: Option<u64>,
}

impl Region {
    /// A whole-chromosome region.
    pub fn chrom(name: impl Into<String>) -> Self {
        Region {
            chrom: name.into(),
            start: None,
            end: None,
        }
    }

    /// A bounded 0-based, half-open interval `[start, end)`.
    pub fn range(chrom: impl Into<String>, start: u64, end: u64) -> Self {
        Region {
            chrom: chrom.into(),
            start: Some(start),
            end: Some(end),
        }
    }

    /// Parse a strict UCSC-style region string.
    ///
    /// Accepts `"chr1"`, `"chr1:1000-2000"` and `"chr1:1000-"` (open end).
    /// Coordinates are plain decimal integers; anything else (commas, unit
    /// suffixes, a missing hyphen, an empty name) is rejected.
    pub fn parse(s: &str) -> Result<Self> {
        Self::from_str(s)
    }
}

fn err(msg: impl Into<String>) -> Error {
    Error::InvalidInput(msg.into())
}

impl FromStr for Region {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let s = s.trim();
        if s.is_empty() {
            return Err(err("empty region string"));
        }
        match s.split_once(':') {
            // Whole chromosome: "chr1".
            None => Ok(Region {
                chrom: s.to_string(),
                start: None,
                end: None,
            }),
            // Bounded / open-ended: "chr1:1000-2000", "chr1:1000-".
            Some((chrom, coords)) => {
                let chrom = chrom.trim();
                if chrom.is_empty() {
                    return Err(err(format!("invalid region '{s}': empty chromosome name")));
                }
                let coords = coords.trim();
                let (start_str, end_str) = coords.split_once('-').ok_or_else(|| {
                    err(format!("invalid region '{s}': expected 'chr:start-end'"))
                })?;
                let start = start_str.trim();
                if start.is_empty() {
                    return Err(err(format!(
                        "invalid region '{s}': missing start coordinate"
                    )));
                }
                let start: u64 = start.parse().map_err(|_| {
                    err(format!(
                        "invalid region '{s}': bad start coordinate '{start}'"
                    ))
                })?;
                let end_str = end_str.trim();
                let end = if end_str.is_empty() {
                    None
                } else {
                    Some(end_str.parse::<u64>().map_err(|_| {
                        err(format!(
                            "invalid region '{s}': bad end coordinate '{end_str}'"
                        ))
                    })?)
                };
                Ok(Region {
                    chrom: chrom.to_string(),
                    start: Some(start),
                    end,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(s: &str) -> Region {
        s.parse().unwrap()
    }

    #[test]
    fn whole_chromosome() {
        assert_eq!(
            region("chr1"),
            Region {
                chrom: "chr1".into(),
                start: None,
                end: None
            }
        );
    }

    #[test]
    fn bounded_interval() {
        assert_eq!(region("chr1:1000-2000"), Region::range("chr1", 1000, 2000));
    }

    #[test]
    fn open_ended_interval() {
        assert_eq!(
            region("chr1:1000-"),
            Region {
                chrom: "chr1".into(),
                start: Some(1000),
                end: None
            }
        );
    }

    #[test]
    fn trims_whitespace() {
        assert_eq!(
            region("  chr1 : 1000 - 2000 "),
            Region::range("chr1", 1000, 2000)
        );
    }

    #[test]
    fn rejects_commas_and_units() {
        assert!(Region::parse("chr1:10,000-20,000").is_err());
        assert!(Region::parse("chr1:10kb-20kb").is_err());
    }

    #[test]
    fn rejects_malformed() {
        assert!(Region::parse("").is_err());
        assert!(Region::parse("chr1:").is_err());
        assert!(Region::parse(":1000-2000").is_err());
        assert!(Region::parse("chr1:1000").is_err()); // missing hyphen
        assert!(Region::parse("chr1:-2000").is_err()); // open start unsupported
        assert!(Region::parse("chr1:1000-2000-3000").is_err());
        assert!(Region::parse("chr1:abc-2000").is_err());
        assert!(Region::parse("chr1:1000-xyz").is_err());
    }
}
