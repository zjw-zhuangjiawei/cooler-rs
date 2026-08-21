//! Round-trip tests for `.cool` and `.mcool` files.

use cooler_rs::{Chrom, Cooler, CoolerWriter, Mcool, McoolWriter, Pixel};

fn test_chroms() -> Vec<Chrom> {
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

fn test_pixels() -> Vec<Pixel> {
    vec![
        Pixel {
            bin1_id: 0,
            bin2_id: 1,
            count: 5.0,
        },
        Pixel {
            bin1_id: 1,
            bin2_id: 3,
            count: 2.5,
        },
        Pixel {
            bin1_id: 3,
            bin2_id: 3,
            count: 7.0,
        },
    ]
}

#[test]
fn cool_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.cool");

    let chroms = test_chroms();
    let pixels = test_pixels();

    // chr1: 3 bins, chr2: 1 bin at 100kb resolution.
    let writer = CoolerWriter::create(&path, &chroms, 100_000).unwrap();
    assert_eq!(writer.n_bins(), 4);
    writer.write_pixels(&pixels).unwrap();
    drop(writer);

    let cool = Cooler::open(&path).unwrap();
    assert_eq!(cool.bin_size().unwrap(), Some(100_000));
    assert_eq!(cool.chroms().unwrap(), chroms);
    assert_eq!(cool.pixels().unwrap(), pixels);
    assert_eq!(cool.n_pixels().unwrap(), 3);

    let bins = cool.bins().unwrap();
    assert_eq!(bins.len(), 4);
    assert_eq!(bins[0].start, 0);
    assert_eq!(bins[0].end, 100_000);
    assert_eq!(bins[2].end, 250_000); // last bin truncated to chrom length
    assert_eq!(bins[3].chrom_id, 1);

    assert_eq!(cool.chrom_offset().unwrap(), vec![0, 3, 4]);
    assert_eq!(cool.bin1_offset().unwrap(), vec![0, 1, 2, 2, 3]);
}

#[test]
fn writer_normalizes_pixels() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("norm.cool");

    let writer = CoolerWriter::create(&path, &test_chroms(), 100_000).unwrap();
    // Lower-triangle and unsorted input.
    writer
        .write_pixels(&[
            Pixel {
                bin1_id: 3,
                bin2_id: 1,
                count: 1.0,
            },
            Pixel {
                bin1_id: 0,
                bin2_id: 0,
                count: 2.0,
            },
        ])
        .unwrap();
    drop(writer);

    let cool = Cooler::open(&path).unwrap();
    let pixels = cool.pixels().unwrap();
    assert_eq!(pixels[0].bin1_id, 0);
    assert_eq!(pixels[1].bin1_id, 1);
    assert_eq!(pixels[1].bin2_id, 3);
}

#[test]
fn writer_rejects_out_of_range_pixels() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.cool");

    let writer = CoolerWriter::create(&path, &test_chroms(), 100_000).unwrap();
    let err = writer
        .write_pixels(&[Pixel {
            bin1_id: 0,
            bin2_id: 99,
            count: 1.0,
        }])
        .unwrap_err();
    assert!(matches!(err, cooler_rs::Error::InvalidInput(_)));
}

#[test]
fn mcool_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.mcool");

    let chroms = test_chroms();

    let writer = McoolWriter::create(&path).unwrap();
    writer
        .create_cooler(&chroms, 100_000)
        .unwrap()
        .write_pixels(&test_pixels())
        .unwrap();
    writer
        .create_cooler(&chroms, 50_000)
        .unwrap()
        .write_pixels(&[Pixel {
            bin1_id: 0,
            bin2_id: 6,
            count: 3.0,
        }])
        .unwrap();
    // Duplicate resolution is rejected.
    assert!(writer.create_cooler(&chroms, 100_000).is_err());
    drop(writer);

    let mcool = Mcool::open(&path).unwrap();
    assert_eq!(mcool.resolutions().unwrap(), vec![50_000, 100_000]);

    let coarse = mcool.cooler(100_000).unwrap();
    assert_eq!(coarse.chroms().unwrap(), chroms);
    assert_eq!(coarse.pixels().unwrap(), test_pixels());

    let fine = mcool.cooler(50_000).unwrap();
    assert_eq!(fine.bins().unwrap().len(), 7); // ceil(250k/50k) + ceil(100k/50k)
    assert_eq!(fine.n_pixels().unwrap(), 1);

    assert!(mcool.cooler(10_000).is_err());
}

#[test]
fn opening_wrong_format_fails() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain.h5");
    // A plain HDF5 file without the cooler format attribute.
    let file = hdf5_metno::File::create(&path).unwrap();
    drop(file);

    // Mcool should still reject a plain file (missing format attribute).
    let mcool_err = Mcool::open(&path).err().expect("expected format error");
    assert!(matches!(mcool_err, cooler_rs::Error::Format(_)));

    // Cooler::open is now lenient (old files may lack the format attribute),
    // but reading data from a plain HDF5 file should fail.
    let cool = Cooler::open(&path).expect("open should succeed (lenient format check)");
    let chroms_err = cool.chroms().expect_err("expected chroms read error");
    assert!(
        matches!(chroms_err, cooler_rs::Error::Hdf5(_)),
        "expected Hdf5 error, got {chroms_err:?}"
    );
}
