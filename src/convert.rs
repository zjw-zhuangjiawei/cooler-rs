//! Conversion between cooler format and other matrix formats.

use crate::error::{Error, Result};
use crate::types::Pixel;

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
                Pixel { bin1_id: 0, bin2_id: 0, count: 1.0 },
                Pixel { bin1_id: 0, bin2_id: 2, count: 2.0 },
                Pixel { bin1_id: 1, bin2_id: 1, count: 4.0 },
                Pixel { bin1_id: 1, bin2_id: 2, count: 5.0 },
                Pixel { bin1_id: 2, bin2_id: 2, count: 7.0 },
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
                Pixel { bin1_id: 0, bin2_id: 0, count: 1.0 },
                Pixel { bin1_id: 0, bin2_id: 1, count: 2.0 },
                Pixel { bin1_id: 1, bin2_id: 1, count: 4.0 },
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
