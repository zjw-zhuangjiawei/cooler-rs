//! Integration tests for `Cooler::offset` / `Cooler::extent` region queries.

use cooler_rs::{write_bins_column, Chrom, Cooler, CoolerWriter, Pixel, Region, SubMatrix};

fn make_cool(path: &std::path::Path) {
    // chr1: 250 kb, chr2: 100 kb at 100 kb bins → bin ids chr1 = 0,1,2; chr2 = 3.
    let chroms = vec![
        Chrom {
            name: "chr1".into(),
            length: 250_000,
        },
        Chrom {
            name: "chr2".into(),
            length: 100_000,
        },
    ];
    let writer = CoolerWriter::create(path, &chroms, 100_000).unwrap();
    writer
        .write_pixels(&[Pixel {
            bin1_id: 0,
            bin2_id: 2,
            count: 1.0,
        }])
        .unwrap();
    drop(writer);
}

#[test]
fn whole_chromosome_extent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.cool");
    make_cool(&path);
    let cool = Cooler::open(&path).unwrap();

    assert_eq!(cool.offset(&Region::chrom("chr1")).unwrap(), 0);
    assert_eq!(cool.extent(&Region::chrom("chr1")).unwrap(), (0, 3));
    assert_eq!(cool.offset(&Region::chrom("chr2")).unwrap(), 3);
    assert_eq!(cool.extent(&Region::chrom("chr2")).unwrap(), (3, 4));
}

#[test]
fn partial_interval_extent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.cool");
    make_cool(&path);
    let cool = Cooler::open(&path).unwrap();

    // [150 kb, 250 kb): overlaps bins 1 ([100,200)) and 2 ([200,250)).
    assert_eq!(
        cool.offset(&Region::range("chr1", 150_000, 250_000))
            .unwrap(),
        1
    );
    assert_eq!(
        cool.extent(&Region::range("chr1", 150_000, 250_000))
            .unwrap(),
        (1, 3)
    );

    // Aligned at a bin boundary: exactly bin 0.
    assert_eq!(
        cool.extent(&Region::range("chr1", 0, 100_000)).unwrap(),
        (0, 1)
    );

    // Empty interval at a bin boundary → empty range.
    assert_eq!(
        cool.offset(&Region::range("chr1", 200_000, 200_000))
            .unwrap(),
        2
    );
    assert_eq!(
        cool.extent(&Region::range("chr1", 200_000, 200_000))
            .unwrap(),
        (2, 2)
    );

    // Open-ended interval runs to the chromosome end.
    assert_eq!(
        cool.extent(&Region::parse("chr1:100000-").unwrap())
            .unwrap(),
        (1, 3)
    );
}

#[test]
fn fetch_bins_and_pixels() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.cool");
    make_cool(&path);
    let cool = Cooler::open(&path).unwrap();

    // bins_in(chr1) → all three chr1 bins.
    let bins = cool.bins_in(&Region::chrom("chr1")).unwrap();
    assert_eq!(bins.len(), 3);
    assert_eq!((bins[0].start, bins[0].end), (0, 100_000));
    assert_eq!((bins[2].start, bins[2].end), (200_000, 250_000));

    // bins_slice is row-based.
    assert_eq!(cool.bins_slice(1, 3).unwrap().len(), 2);

    // pixels_in row band: only the pixel with bin1_id in [0,1) exists.
    let px = cool.pixels_in(&Region::range("chr1", 0, 100_000)).unwrap();
    assert_eq!(
        px,
        vec![Pixel {
            bin1_id: 0,
            bin2_id: 2,
            count: 1.0
        }]
    );

    // A band on chr2 (bin id 3) has no rows → empty.
    assert!(cool.pixels_in(&Region::chrom("chr2")).unwrap().is_empty());
}

#[test]
fn matrix_dense_mirrors_lower() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.cool");
    make_cool(&path);
    let cool = Cooler::open(&path).unwrap();

    // Default (fill_lower): the stored entry (0,2) is mirrored to (2,0).
    let m = cool.matrix_dense(&SubMatrix::square(0..3)).unwrap();
    assert_eq!(m.shape(), [3, 3]);
    assert_eq!(m[[0, 2]], 1.0);
    assert_eq!(m[[2, 0]], 1.0);
    assert_eq!(m[[0, 0]], 0.0);
    assert_eq!(m[[1, 2]], 0.0);

    // fill_lower = false: only the stored upper entry is placed.
    let q = SubMatrix {
        rows: 0..3,
        cols: 0..3,
        fill_lower: false,
        balance: None,
    };
    let m = cool.matrix_dense(&q).unwrap();
    assert_eq!(m[[0, 2]], 1.0);
    assert_eq!(m[[2, 0]], 0.0);
}

#[test]
fn matrix_sparse_deduplicates_mirror() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.cool");
    make_cool(&path);
    let cool = Cooler::open(&path).unwrap();

    let tri = cool.matrix_sparse(&SubMatrix::square(0..3)).unwrap();
    let csr: sprs::CsMat<f64> = tri.to_csr();
    assert_eq!(csr.nnz(), 2); // (0,2) and its mirror (2,0), no duplicate
    let mut found = Vec::new();
    for i in 0..3 {
        if let Some(row) = csr.outer_view(i) {
            for (c, &v) in row.iter() {
                found.push((i, c, v));
            }
        }
    }
    found.sort_unstable_by_key(|t| (t.0, t.1));
    assert_eq!(found, vec![(0, 2, 1.0), (2, 0, 1.0)]);
}

#[test]
fn matrix_balance_scales_by_weight() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.cool");
    make_cool(&path); // 4 bins; pixel (0,2) = 1.0
    write_bins_column(&path, "/", "weight", &[2.0, 1.0, 3.0, 5.0], &[]).unwrap();

    let cool = Cooler::open(&path).unwrap();
    let q = SubMatrix {
        rows: 0..3,
        cols: 0..3,
        fill_lower: false,
        balance: Some("weight".into()),
    };
    let m = cool.matrix_dense(&q).unwrap();
    // count * w[0] * w[2] = 1 * 2 * 3
    assert_eq!(m[[0, 2]], 6.0);

    // Missing balance column is an error, not silent.
    let bad = SubMatrix {
        rows: 0..3,
        cols: 0..3,
        fill_lower: false,
        balance: Some("nope".into()),
    };
    assert!(matches!(
        cool.matrix_dense(&bad).unwrap_err(),
        cooler_rs::Error::InvalidInput(_)
    ));
}

#[test]
fn invalid_regions_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.cool");
    make_cool(&path);
    let cool = Cooler::open(&path).unwrap();

    let err = cool.extent(&Region::chrom("chr3")).unwrap_err();
    assert!(
        matches!(err, cooler_rs::Error::InvalidInput(_)),
        "unknown chrom: {err}"
    );

    // end past chromosome length → error, never clamped.
    let err = cool.extent(&Region::range("chr1", 0, 300_000)).unwrap_err();
    assert!(
        matches!(err, cooler_rs::Error::InvalidInput(_)),
        "out of bounds: {err}"
    );

    // end before start.
    let err = cool
        .extent(&Region::range("chr2", 50_000, 10_000))
        .unwrap_err();
    assert!(
        matches!(err, cooler_rs::Error::InvalidInput(_)),
        "end < start: {err}"
    );
}
