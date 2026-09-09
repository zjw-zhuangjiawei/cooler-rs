//! Integration tests for `cooler_rs::convert::cooler_to_hic` (.cool/.mcool
//! -> .hic v8 conversion).
//!
//! All fixtures are synthetic (no external data): a small multi-resolution
//! `.mcool` / single-resolution `.cool` is built in place with
//! `McoolWriter`/`CoolerWriter`, converted, and the output re-read with
//! `HiCFile` and compared against the source.
//!
//! Chromosomes use non-divisible lengths (`chr1` 250_000, `chr2` 100_000) to
//! exercise the ceil/partial-bin path at both resolutions.

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

    cooler_to_hic(&mcool_path, &hic_path, "test", None, None).unwrap();

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

#[test]
fn cool_to_hic_copies_weight_column() {
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

    cooler_to_hic(&cool_path, &hic_path, "test", Some("weight"), Some("KR")).unwrap();

    let hic = HiCFile::open(&hic_path).unwrap();
    assert_eq!(hic.chromosomes(), chroms());
    assert_eq!(hic.avail_normalizations().unwrap(), vec!["KR".to_string()]);
    assert_eq!(
        hic.norm_vector(100_000, "chr1", "KR").unwrap().unwrap(),
        vec![0.5, 1.0, 2.0] // chr1 has 3 bins
    );
    assert_eq!(
        hic.norm_vector(100_000, "chr2", "KR").unwrap().unwrap(),
        vec![4.0] // chr2 has 1 bin
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

    cooler_to_hic(&mcool_path, &hic_path, "test", Some("weight"), None).unwrap();

    let hic = HiCFile::open(&hic_path).unwrap();
    assert_eq!(
        hic.avail_normalizations().unwrap(),
        vec!["weight".to_string()]
    );
    // Weight present at the coarse resolution that carries the column…
    assert_eq!(
        hic.norm_vector(100_000, "chr1", "weight").unwrap().unwrap(),
        vec![0.5, 1.0, 2.0]
    );
    // …and absent (None, not an error) where the column does not exist.
    assert_eq!(hic.norm_vector(50_000, "chr1", "weight").unwrap(), None);
}

/// Resolve the optional large-fixture .mcool used for the bounded-RAM test.
/// Skips (no-op) when the fixture is absent, matching the discipline in
/// `tests/hic.rs` (`fixture()`).
fn fixture_mcool() -> Option<std::path::PathBuf> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/4DNFIZ1ZVXC8.mcool");
    p.exists().then_some(p)
}

#[cfg(unix)]
fn resident_bytes() -> u64 {
    // /proc/self/statm: size | resident | shared | text | data | dirty
    let s = std::fs::read_to_string("/proc/self/statm").unwrap();
    let mut it = s.split_whitespace();
    let _size: u64 = it.next().unwrap().parse().unwrap();
    let resident_pages: u64 = it.next().unwrap().parse().unwrap();
    resident_pages * 4096
}

#[cfg(not(unix))]
fn resident_bytes() -> u64 {
    0
}

#[test]
fn convert_large_mcool_stays_bounded() {
    let Some(input) = fixture_mcool() else {
        eprintln!("skipping: 4DNFIZ1ZVXC8.mcool not present");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out.hic");

    let peak_before = resident_bytes();
    cooler_to_hic(&input, &out, "test", None, None).unwrap();
    let peak_after = resident_bytes();
    let delta_mb = peak_after.saturating_sub(peak_before) / 1_000_000;

    assert!(
        delta_mb < 512,
        "peak RSS grew by {delta_mb} MB during conversion; budget is 512 MB"
    );

    let hic = HiCFile::open(&out).unwrap();
    assert!(!hic.resolutions().is_empty());
    let coarse = *hic.resolutions().iter().min().unwrap();
    let pixels = hic.pixels(coarse).unwrap();
    assert!(!pixels.is_empty(), "no pixels at coarsest resolution");
}
