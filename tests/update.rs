//! Tests for in-place update/delete helpers and the integrity checker:
//! `rename_chroms`, `delete_bins_column`, `Cooler::validate`.

use cooler_rs::{
    delete_bins_column, rename_chroms, write_bins_column, AttrValue, Chrom, Cooler, CoolerWriter,
    Mcool, McoolWriter, Pixel, Region,
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

/// 4 bins (chr1: 3, chr2: 1) at 100 kb, 5 stored pixels.
fn pixels() -> Vec<Pixel> {
    vec![
        Pixel {
            bin1_id: 0,
            bin2_id: 0,
            count: 4.0,
        },
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
            bin1_id: 2,
            bin2_id: 2,
            count: 7.0,
        },
        Pixel {
            bin1_id: 3,
            bin2_id: 3,
            count: 1.0,
        },
    ]
}

fn make_cool(dir: &tempfile::TempDir) -> std::path::PathBuf {
    let path = dir.path().join("t.cool");
    let writer = CoolerWriter::create(&path, &chroms(), 100_000).unwrap();
    writer.write_pixels(&pixels()).unwrap();
    path
}

/// Rewrite one dataset in place (test corruption helpers).
fn rewrite_i32(path: &std::path::Path, dset: &str, new: Vec<i32>) {
    let file = hdf5_metno::File::open_rw(path).unwrap();
    file.group("/")
        .unwrap()
        .dataset(dset)
        .unwrap()
        .write(&new)
        .unwrap();
}

#[test]
fn rename_chroms_rewrites_names_in_place() {
    let dir = tempfile::tempdir().unwrap();
    let path = make_cool(&dir);

    rename_chroms(&path, "/", &[("chr1", "chrX")]).unwrap();

    let cool = Cooler::open(&path).unwrap();
    assert_eq!(
        cool.chroms().unwrap(),
        vec![
            Chrom {
                name: "chrX".into(),
                length: 250_000
            },
            Chrom {
                name: "chr2".into(),
                length: 100_000
            },
        ]
    );
    // Bin ids, pixels, and indexes are untouched.
    assert_eq!(cool.pixels().unwrap(), pixels());
    assert_eq!(cool.bin_chrom().unwrap(), vec![0, 0, 0, 1]);
    assert_eq!(
        cool.offset(&Region::chrom("chrX")).unwrap(),
        0,
        "region lookup works by the new name"
    );
    assert!(cool.validate().unwrap().is_ok());
}

#[test]
fn rename_chroms_keeps_unnamed_and_renames_all() {
    let dir = tempfile::tempdir().unwrap();
    let path = make_cool(&dir);

    rename_chroms(&path, "/", &[("chr1", "chrA"), ("chr2", "chrB")]).unwrap();
    let names: Vec<String> = Cooler::open(&path)
        .unwrap()
        .chroms()
        .unwrap()
        .into_iter()
        .map(|c| c.name)
        .collect();
    assert_eq!(names, ["chrA", "chrB"]);

    // Partial rename leaves the rest alone.
    rename_chroms(&path, "/", &[("chrA", "chr1")]).unwrap();
    let names: Vec<String> = Cooler::open(&path)
        .unwrap()
        .chroms()
        .unwrap()
        .into_iter()
        .map(|c| c.name)
        .collect();
    assert_eq!(names, ["chr1", "chrB"]);
}

#[test]
fn rename_chroms_rejects_bad_input() {
    let dir = tempfile::tempdir().unwrap();
    let path = make_cool(&dir);

    // Unknown old name.
    assert!(rename_chroms(&path, "/", &[("chr9", "chrZ")]).is_err());
    // Would produce duplicate names.
    assert!(rename_chroms(&path, "/", &[("chr1", "x"), ("chr2", "x")]).is_err());
    // Empty new name.
    assert!(rename_chroms(&path, "/", &[("chr1", "")]).is_err());
    // No-op is fine.
    assert!(rename_chroms(&path, "/", &[]).is_ok());
}

#[test]
fn rename_chroms_works_on_mcool_group() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.mcool");
    let mw = McoolWriter::create(&path).unwrap();
    let c = mw.create_cooler(&chroms(), 100_000).unwrap();
    c.write_pixels(&pixels()).unwrap();
    drop(mw);

    rename_chroms(&path, "/resolutions/100000", &[("chr1", "chrX")]).unwrap();

    let mcool = Mcool::open(&path).unwrap();
    let cool = mcool.cooler(100_000).unwrap();
    assert_eq!(cool.chroms().unwrap()[0].name, "chrX");
}

#[test]
fn delete_bins_column_lifecycle() {
    let dir = tempfile::tempdir().unwrap();
    let path = make_cool(&dir);

    write_bins_column(
        &path,
        "/",
        "weight",
        &[1.0, 2.0, 3.0, 4.0],
        &[("scale", AttrValue::F64(1.0))],
    )
    .unwrap();
    {
        let cool = Cooler::open(&path).unwrap();
        assert!(cool.bins_has_column("weight").unwrap());
    } // drop the read handle before the read-write delete below.

    delete_bins_column(&path, "/", "weight").unwrap();
    assert!(!Cooler::open(&path)
        .unwrap()
        .bins_has_column("weight")
        .unwrap());

    // Deleting again, or deleting a required column, is an error.
    assert!(delete_bins_column(&path, "/", "weight").is_err());
    assert!(delete_bins_column(&path, "/", "chrom").is_err());
    assert!(Cooler::open(&path).unwrap().validate().unwrap().is_ok());
}

#[test]
fn validate_passes_a_wellformed_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = make_cool(&dir);

    let cool = Cooler::open(&path).unwrap();
    let report = cool.validate().unwrap();
    assert!(
        report.is_ok(),
        "expected clean file, got issues: {:?}",
        report.issues
    );
}

#[test]
fn validate_flags_a_bad_chrom_code() {
    let dir = tempfile::tempdir().unwrap();
    let path = make_cool(&dir);

    // 2 = out of range (only chr0/chr1 exist) and outside the chrom_offset band.
    rewrite_i32(&path, "bins/chrom", vec![2, 0, 0, 1]);

    let report = Cooler::open(&path).unwrap().validate().unwrap();
    assert!(!report.is_ok());
    assert!(
        report.issues.iter().any(|i| i.contains("chrom code 2")),
        "issues: {:?}",
        report.issues
    );
}

#[test]
fn validate_flags_a_broken_bin1_offset() {
    let dir = tempfile::tempdir().unwrap();
    let path = make_cool(&dir);

    // First entry must be 0 (rows start at offset 0).
    let file = hdf5_metno::File::open_rw(&path).unwrap();
    let ds = file
        .group("/")
        .unwrap()
        .dataset("indexes/bin1_offset")
        .unwrap();
    let mut o: Vec<i64> = ds.read_1d().unwrap().to_vec();
    o[0] = 1;
    ds.write(&o).unwrap();
    drop(file);

    let report = Cooler::open(&path).unwrap().validate().unwrap();
    assert!(!report.is_ok());
}

#[test]
fn validate_flags_a_pixel_out_of_range() {
    let dir = tempfile::tempdir().unwrap();
    let path = make_cool(&dir);

    let file = hdf5_metno::File::open_rw(&path).unwrap();
    let ds = file.group("/").unwrap().dataset("pixels/bin1_id").unwrap();
    let mut ids: Vec<i64> = ds.read_1d().unwrap().to_vec();
    ids[0] = 5; // only 4 bins exist
    ds.write(&ids).unwrap();
    drop(file);

    let report = Cooler::open(&path).unwrap().validate().unwrap();
    assert!(!report.is_ok());
    assert!(
        report.issues.iter().any(|i| i.contains("out of range")),
        "issues: {:?}",
        report.issues
    );
}

#[test]
fn rename_chroms_refuses_enum_bins_chrom() {
    // A bins/chrom dataset that is not plain i32 (simulated via a mismatch)
    // must be refused rather than corrupt bin decoding. We simulate the
    // py-cooler enum case by pointing rename at a file whose bins/chrom
    // cannot be read as i32 slices — here a vlen-string dataset.
    let dir = tempfile::tempdir().unwrap();
    let path = make_cool(&dir);

    let file = hdf5_metno::File::open_rw(&path).unwrap();
    let bins = file.group("/").unwrap().group("bins").unwrap();
    if bins.link_exists("chrom") {
        bins.unlink("chrom").unwrap();
    }
    use hdf5_metno::types::VarLenUnicode;
    let strs: Vec<VarLenUnicode> = ["x", "y", "z", "w"]
        .map(String::from)
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();
    bins.new_dataset::<VarLenUnicode>()
        .shape(4)
        .create("chrom")
        .unwrap()
        .write(&strs)
        .unwrap();
    drop(file);

    assert!(rename_chroms(&path, "/", &[("chr1", "chrX")]).is_err());
}
