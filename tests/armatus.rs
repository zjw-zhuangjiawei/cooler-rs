//! Regression test: the Rust Armatus port must reproduce the reference output
//! of the original Armatus 2.3 binary on its published example data.
//!
//! `three-domains.150.txt` is `armatus/examples/three-domains.txt.gz`
//! (decompressed): a 150x150 matrix with three 50-bin blocks. The reference
//! consensus output was produced by running:
//!
//!     armatus -i three-domains.txt.gz -g 0.5 -n 5 -j -o ref
//!
//! giving `[0, 49]`, `[50, 99]`, `[100, 149]` and an optimal score 349.913 at
//! gamma=0.5. The Dixon dense parser does not log-transform, so the matrix is
//! fed in raw (no log) for both.

use cooler_rs::armatus::{self, ArmatusDag, Params, Sums};
use ndarray::Array2;

const MATRIX: &str = include_str!("data/three-domains.150.txt");

fn parse_dense_matrix(text: &str) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split_whitespace()
                .map(|f| f.parse().expect("non-numeric entry"))
                .collect()
        })
        .collect();
    let n = rows.len();
    let mut x = Array2::zeros((n, n));
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            x[[i, j]] = v;
        }
    }
    x
}

fn params(just_gamma_max: bool) -> Params {
    Params {
        gamma_max: 0.5,
        step_size: 0.05,
        top_k: 1,
        min_mean_samples: 5,
        just_gamma_max,
    }
}

#[test]
fn just_gamma_max_matches_binary_consensus() {
    let x = parse_dense_matrix(MATRIX);
    assert_eq!(x.nrows(), 150);

    let domains = armatus::call_domains(&x, &params(true));

    assert_eq!(domains.len(), 3, "domain count differs");
    let expected = [[0usize, 49], [50, 99], [100, 149]];
    for (d, e) in domains.iter().zip(expected) {
        assert_eq!([d.start, d.end], e, "domain bounds differ");
    }
}

#[test]
fn optimal_score_matches_binary() {
    let x = parse_dense_matrix(MATRIX);
    let sums = Sums::compute(&x, 0.5, 5);
    let mut dag = ArmatusDag::new(&sums, 1);
    dag.build();

    // Reference binary prints "OPTIMAL SCORE: 349.913" (6 sig figs).
    assert!(
        (dag.optimal_score() - 349.913).abs() < 1e-3,
        "optimal score {} != 349.913",
        dag.optimal_score()
    );
}

#[test]
fn multiscale_consensus_matches_binary() {
    let x = parse_dense_matrix(MATRIX);

    let ensemble = armatus::multiscale_domains(&x, &params(false));
    // 0.0..=0.5 at step 0.05 -> 11 resolutions, one domain set each (top_k=1).
    assert_eq!(ensemble.resolutions.len(), 11);

    let domains = armatus::consensus_domains(&ensemble);
    let expected = [[0usize, 49], [50, 99], [100, 149]];
    assert_eq!(domains.len(), 3);
    for (d, e) in domains.iter().zip(expected) {
        assert_eq!([d.start, d.end], e);
    }
}
