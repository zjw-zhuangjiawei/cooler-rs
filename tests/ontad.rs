//! Regression test: with default parameters, the Rust OnTAD port must
//! reproduce the reference output of the original C++ OnTAD v1.4 on its
//! published test data.
//!
//! `mES_rep2.40Kb.raw.chr19.mat` is taken unmodified from
//! <https://github.com/anlin00007/OnTAD> (`test/`): a dense 1534×1534
//! contact matrix (mouse ES cells, chr19, 40 kb resolution).
//! `mES_rep2.40Kb.raw.chr19.mat.tad` is the reference output produced by
//! running the C++ executable with default parameters on that matrix.

use cooler_rs::ontad::{call_tads, Params, Tad};
use ndarray::Array2;

const MATRIX: &str = include_str!("data/mES_rep2.40Kb.raw.chr19.mat");
const EXPECTED_TAD: &str = include_str!("data/mES_rep2.40Kb.raw.chr19.mat.tad");

/// Parse a dense N×N whitespace-separated text matrix.
fn parse_dense_matrix(text: &str) -> Array2<f64> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let first = lines.next().expect("empty matrix");
    let n = first.split_whitespace().count();
    let mut x = Array2::zeros((n, n));
    for (j, field) in first.split_whitespace().enumerate() {
        x[[0, j]] = field.parse().expect("non-numeric matrix entry");
    }
    for (i, line) in lines.enumerate() {
        for (j, field) in line.split_whitespace().enumerate() {
            x[[i + 1, j]] = field.parse().expect("non-numeric matrix entry");
        }
    }
    x
}

/// One line of a `.tad` file: 1-based inclusive bounds, nesting level,
/// mean contact frequency, DP score (both rounded to 3 decimals).
struct TadRecord {
    start: usize,
    end: usize,
    level: usize,
    mean: f64,
    score: f64,
}

/// Parse the reference `.tad` output of the C++ executable.
fn parse_tad(text: &str) -> Vec<TadRecord> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let mut fields = line.split_whitespace();
            let mut next = || fields.next().expect("too few columns");
            TadRecord {
                start: next().parse().expect("bad start"),
                end: next().parse().expect("bad end"),
                level: next().parse().expect("bad level"),
                mean: next().parse().expect("bad mean"),
                score: next().parse().expect("bad score"),
            }
        })
        .collect()
}

#[test]
fn default_params_reproduce_reference_tad_output() {
    let params = Params::default();
    let mut x = parse_dense_matrix(MATRIX);

    // Like the C++ `loadMatrix` (and our `matrix_from_cooler`), keep only
    // the band |j - i| <= 2 * maxsz around the diagonal.
    let band = params.maxsz * 2;
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            if i.abs_diff(j) > band {
                x[[i, j]] = 0.0;
            }
        }
    }

    let tad: Tad = call_tads(&mut x, &params);
    let expected = parse_tad(EXPECTED_TAD);
    assert_eq!(tad.len(), expected.len(), "TAD count differs");

    for (j, e) in expected.iter().enumerate() {
        // .tad bounds are 1-based; Tad::bound is 0-based.
        assert_eq!(
            [tad.bound[j][0] + 1, tad.bound[j][1] + 1],
            [e.start, e.end],
            "TAD {j} bounds differ"
        );
        assert_eq!(tad.level[j], e.level, "TAD {j} level differs");
        // Reference values were rounded to 3 decimals on output.
        assert!(
            (tad.mean[j] - e.mean).abs() <= 5e-4,
            "TAD {j} mean differs: {} vs {}",
            tad.mean[j],
            e.mean
        );
        assert!(
            (tad.score[j] - e.score).abs() <= 5e-4,
            "TAD {j} score differs: {} vs {}",
            tad.score[j],
            e.score
        );
    }
}
