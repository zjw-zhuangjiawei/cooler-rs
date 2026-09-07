//! `compare` — similarity metrics between two coolers.
//!
//! Covers the whole pipeline (SCC, Pearson, Spearman) end-to-end against
//! hand-computed values, plus unit-level checks of the private helpers that
//! the integration tests can't reach (mean filter, ranks, weights).

use cooler_rs::{compare_pair, Chrom, CompareMetric, CompareParams, Cooler, CoolerWriter, Pixel};
use std::path::Path;

fn write_cool(dir: &Path, name: &str, pixels: &[Pixel]) -> Cooler {
    let path = dir.join(name);
    let writer = CoolerWriter::create(
        &path,
        &[Chrom {
            name: "chr1".into(),
            length: 400_000,
        }],
        100_000,
    )
    .unwrap();
    writer.write_pixels(pixels).unwrap();
    Cooler::open(path).unwrap()
}

fn px(bin1: i64, bin2: i64, count: f64) -> Pixel {
    Pixel {
        bin1_id: bin1,
        bin2_id: bin2,
        count,
    }
}

/// A: d=1 values [1,2,3]; B: d=1 values [3,1,2]. Off-diagonal Pearson is
/// -0.5 on the only populated diagonal, so every metric returns -0.5.
fn pair(dir: &Path) -> (Cooler, Cooler) {
    let a = write_cool(
        dir,
        "a.cool",
        &[px(0, 1, 1.0), px(1, 2, 2.0), px(2, 3, 3.0)],
    );
    let b = write_cool(
        dir,
        "b.cool",
        &[px(0, 1, 3.0), px(1, 2, 1.0), px(2, 3, 2.0)],
    );
    (a, b)
}

fn scc_default() -> CompareParams {
    CompareParams {
        h: 0,
        ..CompareParams::default()
    }
}

#[test]
fn scc_cross_is_minus_half() {
    let dir = tempfile::tempdir().unwrap();
    let (a, b) = pair(dir.path());
    let out = compare_pair(&a, &b, CompareMetric::Scc, &scc_default()).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].0, "chr1");
    assert!((out[0].1 - -0.5).abs() < 1e-12, "got {}", out[0].1);
}

#[test]
fn pearson_cross_is_minus_half() {
    let dir = tempfile::tempdir().unwrap();
    let (a, b) = pair(dir.path());
    let out = compare_pair(&a, &b, CompareMetric::Pearson, &scc_default()).unwrap();
    assert!((out[0].1 - -0.5).abs() < 1e-12, "got {}", out[0].1);
}

#[test]
fn spearman_cross_is_minus_half() {
    let dir = tempfile::tempdir().unwrap();
    let (a, b) = pair(dir.path());
    let out = compare_pair(&a, &b, CompareMetric::Spearman, &scc_default()).unwrap();
    assert!((out[0].1 - -0.5).abs() < 1e-12, "got {}", out[0].1);
}

#[test]
fn self_comparison_is_one() {
    let dir = tempfile::tempdir().unwrap();
    let (a, _) = pair(dir.path());
    for h in [0usize, 1] {
        let params = CompareParams {
            h,
            ..CompareParams::default()
        };
        let scc = compare_pair(&a, &a, CompareMetric::Scc, &params).unwrap();
        assert!((scc[0].1 - 1.0).abs() < 1e-9, "SCC h={h} got {}", scc[0].1);
    }
    let pearson = compare_pair(&a, &a, CompareMetric::Pearson, &scc_default()).unwrap();
    assert!((pearson[0].1 - 1.0).abs() < 1e-12);
    let spearman = compare_pair(&a, &a, CompareMetric::Spearman, &scc_default()).unwrap();
    assert!((spearman[0].1 - 1.0).abs() < 1e-12);
}

/// With h=2 the 5×5 window over a 4×4 matrix leaves diagonal d=2 constant
/// (both cells 0.5), which hicrep counts as rho=0. The weighted mean over
/// d=1 (rho=1, w=1/3) and d=2 (rho=0, w=1/4) is exactly 4/7.
#[test]
fn scc_h2_matches_hand_value() {
    let dir = tempfile::tempdir().unwrap();
    let (a, _) = pair(dir.path());
    let params = CompareParams {
        h: 2,
        ..CompareParams::default()
    };
    let scc = compare_pair(&a, &a, CompareMetric::Scc, &params).unwrap();
    assert!((scc[0].1 - 4.0 / 7.0).abs() < 1e-12, "got {}", scc[0].1);
}

#[test]
fn rejects_mismatched_bin_size() {
    let dir = tempfile::tempdir().unwrap();
    let (a, _) = pair(dir.path());
    // Rebuild b at a different bin size.
    let path = dir.path().join("b2.cool");
    let writer = CoolerWriter::create(
        &path,
        &[Chrom {
            name: "chr1".into(),
            length: 800_000,
        }],
        200_000,
    )
    .unwrap();
    writer
        .write_pixels(&[px(0, 1, 3.0), px(1, 2, 1.0)])
        .unwrap();
    let b2 = Cooler::open(path).unwrap();
    assert!(compare_pair(&a, &b2, CompareMetric::Scc, &scc_default()).is_err());
}
