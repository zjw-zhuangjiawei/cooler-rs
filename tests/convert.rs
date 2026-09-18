#![cfg(any())]
//! Integration tests for `cooler_rs::convert::cooler_to_hic` (.cool/.mcool
//! -> .hic v8 conversion).
//!
//! Nearly all fixtures are synthetic (no external data): a small
//! multi-resolution `.mcool` / single-resolution `.cool` is built in place with
//! `McoolWriter`/`CoolerWriter`, converted, and the output re-read with
//! `HiCFile` and compared against the source. The one exception is the
//! bounded-RAM test, which needs a real fine-resolution matrix and so takes
//! `dmel-root-13res` through the shared harness.
//!
//! Chromosomes use non-divisible lengths (`chr1` 250_000, `chr2` 100_000) to
//! exercise the ceil/partial-bin path at both resolutions.

mod common;

use cooler_rs::{
    convert::cooler_to_hic, write_bins_column, Chrom, Cooler, CoolerWriter, HiCFile, Mcool,
    McoolWriter, Pixel,
};

fn chroms() -> Vec<Chrom> {
    vec![
        Chrom {
            name: "chr1".into(),
            length: 250_000,
        },
        Chrom {
            name: "chr2".into(),
            length: 100_000,
        },
    ]
}

/// Coarse (100 kb) pixels over 4 bins (chr1: 3 = ceil(250k/100k), chr2: 1).
fn coarse_pixels() -> Vec<Pixel> {
    vec![
        Pixel {
            bin1_id: 0,
            bin2_id: 0,
            count: 1.0,
        },
        Pixel {
            bin1_id: 0,
            bin2_id: 2,
            count: 2.0,
        },
        Pixel {
            bin1_id: 1,
            bin2_id: 3,
            count: 3.0,
        },
        Pixel {
            bin1_id: 2,
            bin2_id: 2,
            count: 4.0,
        },
        Pixel {
            bin1_id: 3,
            bin2_id: 3,
            count: 5.0,
        },
    ]
}

/// Fine (50 kb) pixels over 7 bins (chr1: 5, chr2: 2).
fn fine_pixels() -> Vec<Pixel> {
    vec![Pixel {
        bin1_id: 0,
        bin2_id: 6,
        count: 7.0,
    }]
}

/// Sorted `(bin1_id, bin2_id, count)` tuples for multiset comparison.
fn keyed(pixels: &[Pixel]) -> Vec<(i64, i64, u64)> {
    let mut v: Vec<(i64, i64, u64)> = pixels
        .iter()
        .map(|p| (p.bin1_id, p.bin2_id, p.count.to_bits()))
        .collect();
    v.sort_unstable();
    v
}

#[test]
fn mcool_to_hic_converts_all_resolutions() {
    let dir = tempfile::tempdir().unwrap();
    let mcool_path = dir.path().join("in.mcool");
    let hic_path = dir.path().join("out.hic");

    let writer = McoolWriter::create(&mcool_path).unwrap();
    writer
        .create_cooler(&chroms(), 100_000)
        .unwrap()
        .write_pixels(&coarse_pixels())
        .unwrap();
    writer
        .create_cooler(&chroms(), 50_000)
        .unwrap()
        .write_pixels(&fine_pixels())
        .unwrap();
    drop(writer);

    cooler_to_hic(&mcool_path, &hic_path, "test", &[], None, &[]).unwrap();

    let hic = HiCFile::open(&hic_path).unwrap();
    assert_eq!(hic.genome_id(), "test");
    assert_eq!(hic.chromosomes(), chroms());
    let mut res = hic.resolutions().to_vec();
    res.sort_unstable();
    assert_eq!(res, vec![50_000, 100_000]);
    assert_eq!(hic.avail_normalizations().unwrap(), Vec::<String>::new());

    // Each output resolution matches the input pixel-for-pixel.
    for res in [100_000u32, 50_000] {
        let source = Mcool::open(&mcool_path)
            .unwrap()
            .cooler(res as u64)
            .unwrap();
        let src_pixels = source.pixels().unwrap();
        let out_pixels = hic.pixels(res).unwrap();
        assert_eq!(out_pixels.len(), src_pixels.len(), "res {res}");
        assert_eq!(keyed(&out_pixels), keyed(&src_pixels), "res {res}");
    }
}

/// The `bins/weight` column is a *multiplicative* bias; a `.hic` normalization
/// vector is *divisive*. The two are reciprocal, so the conversion must invert
/// — copying the column through verbatim would balance the output matrix by
/// the reciprocal of the intended factor.
#[test]
fn cool_to_hic_inverts_weight_column() {
    let dir = tempfile::tempdir().unwrap();
    let cool_path = dir.path().join("in.cool");
    let hic_path = dir.path().join("out.hic");

    CoolerWriter::create(&cool_path, &chroms(), 100_000)
        .unwrap()
        .write_pixels(&coarse_pixels())
        .unwrap();
    // Distinct per-bin values catch chromosome-split boundary bugs. Stored
    // under an overridden name ("KR") to exercise the renaming path.
    write_bins_column(&cool_path, "/", "weight", &[0.5, 1.0, 2.0, 4.0], &[]).unwrap();

    cooler_to_hic(
        &cool_path,
        &hic_path,
        "test",
        &["weight".to_string()],
        Some("KR"),
        &[],
    )
    .unwrap();

    let hic = HiCFile::open(&hic_path).unwrap();
    assert_eq!(hic.chromosomes(), chroms());
    assert_eq!(hic.avail_normalizations().unwrap(), vec!["KR".to_string()]);
    assert_eq!(
        hic.norm_vector(100_000, "chr1", "KR").unwrap().unwrap(),
        vec![2.0, 1.0, 0.5] // chr1 has 3 bins; 1/w of [0.5, 1.0, 2.0]
    );
    assert_eq!(
        hic.norm_vector(100_000, "chr2", "KR").unwrap().unwrap(),
        vec![0.25] // chr2 has 1 bin; 1/w of [4.0]
    );
    // Pixels unchanged by the norm copy.
    let src_pixels = Cooler::open(&cool_path).unwrap().pixels().unwrap();
    assert_eq!(keyed(&hic.pixels(100_000).unwrap()), keyed(&src_pixels));
}

#[test]
fn mcool_partial_weight_column_is_skipped_when_absent() {
    let dir = tempfile::tempdir().unwrap();
    let mcool_path = dir.path().join("in.mcool");
    let hic_path = dir.path().join("out.hic");

    let writer = McoolWriter::create(&mcool_path).unwrap();
    writer
        .create_cooler(&chroms(), 100_000)
        .unwrap()
        .write_pixels(&coarse_pixels())
        .unwrap();
    writer
        .create_cooler(&chroms(), 50_000)
        .unwrap()
        .write_pixels(&fine_pixels())
        .unwrap();
    drop(writer);
    // Add the weight column to the coarse resolution only.
    write_bins_column(
        &mcool_path,
        "/resolutions/100000",
        "weight",
        &[0.5, 1.0, 2.0, 4.0],
        &[],
    )
    .unwrap();

    cooler_to_hic(
        &mcool_path,
        &hic_path,
        "test",
        &["weight".to_string()],
        None,
        &[],
    )
    .unwrap();

    let hic = HiCFile::open(&hic_path).unwrap();
    assert_eq!(
        hic.avail_normalizations().unwrap(),
        vec!["weight".to_string()]
    );
    // Weight present (inverted) at the coarse resolution that carries the
    // column…
    assert_eq!(
        hic.norm_vector(100_000, "chr1", "weight").unwrap().unwrap(),
        vec![2.0, 1.0, 0.5]
    );
    // …and absent (None, not an error) where the column does not exist.
    assert_eq!(hic.norm_vector(50_000, "chr1", "weight").unwrap(), None);
}

/// Peak resident set size of this process, in MB, from `/proc/self/status`.
///
/// `VmHWM` is a high-water mark, so it catches a transient allocation that a
/// pair of spot samples misses — the shape of the 36 GB `finalize` blowup this
/// test exists to catch. It is monotonic for the life of the process, so the
/// assertion below is an absolute ceiling, not a before/after delta.
#[cfg(target_os = "linux")]
fn peak_rss_mb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kb: u64 = status
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    Some(kb / 1024)
}

#[test]
fn convert_large_mcool_stays_bounded() {
    let Some(input) = common::fixture_or_skip("dmel-root-13res") else {
        return;
    };
    if !cfg!(target_os = "linux") {
        eprintln!("SKIP convert_large_mcool_stays_bounded (no VmHWM outside Linux)");
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out.hic");

    // Two resolutions, not all thirteen: 1 kb is where the per-pair scratch and
    // the 1M-pixel chunk boundary are both exercised, and 100 kb keeps a coarse
    // level in the file. Measured on this machine: ~19 s and ~540 MB peak.
    // Converting every resolution takes minutes and covers no path this one
    // does not.
    cooler_to_hic(&input, &out, "test", &[], None, &[1000, 100_000]).unwrap();

    let peak = peak_rss_mb().expect("checked by cfg! above");
    assert!(
        peak < 1536,
        "peak RSS reached {peak} MB during conversion; budget is 1536 MB. \
         The failure this guards against was a 36 GB allocation, which a \
         before/after sample of /proc/self/statm could not see."
    );

    let hic = HiCFile::open(&out).unwrap();
    // `resolutions()` returns the stored order, which is finest-last, so sort
    // before comparing — `dump` is what presents them ascending.
    let mut got = hic.resolutions().to_vec();
    got.sort_unstable();
    assert_eq!(got, vec![1000, 100_000]);
    let pixels = hic.pixels(100_000).unwrap();
    assert!(!pixels.is_empty(), "no pixels at 100 kb");
}
