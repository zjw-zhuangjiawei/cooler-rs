//! Schema-version read paths the current writer does not produce:
//!
//! * `.cool` v1 — [`bins/chrom`](Cooler::bin_chrom) holds chromosome *names*
//!   (v2 replaced it with an integer), and the coordinate/index columns are
//!   64-bit where v3 uses 32-bit.
//! * `.cool` v3 `storage-mode` other than `symmetric-upper` — must be refused,
//!   not silently read as if the upper triangle were the whole matrix.
//! * `.mcool` v1 — root-level zoom-level groups instead of
//!   `/resolutions/<binsize>`, and no root `format` attribute.
//!
//! Fixtures are built attribute by attribute because cooler-rs only writes the
//! current schema.

use hdf5_metno::types::{FixedAscii, VarLenUnicode};
use hdf5_metno::File;

use cooler_rs::{Chrom, Cooler, CoolerWriter, File as AnyFile, Mcool, Pixel};

fn vlen(s: &str) -> VarLenUnicode {
    s.parse::<VarLenUnicode>().unwrap()
}

fn attr_str(group: &hdf5_metno::Group, name: &str, value: &str) {
    group
        .new_attr::<VarLenUnicode>()
        .create(name)
        .unwrap()
        .write_scalar(&vlen(value))
        .unwrap();
}

fn attr_i64(group: &hdf5_metno::Group, name: &str, value: i64) {
    group
        .new_attr::<i64>()
        .create(name)
        .unwrap()
        .write_scalar(&value)
        .unwrap();
}

/// A schema-v1 `.cool`: 3 bins over 2 chromosomes, 2 pixels, `chrom` as names.
///
/// `fixed_width_chrom` picks the string dtype: `FixedAscii<32>` as the v1
/// schema documents it, or a variable-length string, which is what h5py
/// actually writes for a numpy `U` array.
fn write_v1_cool(path: &std::path::Path, fixed_width_chrom: bool) {
    let f = File::create(path).unwrap();
    let root = f.group("/").unwrap();
    attr_str(&root, "format", "HDF5::Cooler");
    attr_str(&root, "format-version", "1"); // a *string* before v3
    attr_str(&root, "bin-type", "fixed");
    attr_i64(&root, "bin-size", 100_000);
    attr_i64(&root, "nchroms", 2);
    attr_i64(&root, "nbins", 3);
    attr_i64(&root, "nnz", 2);

    let chroms = root.create_group("chroms").unwrap();
    let names: Vec<FixedAscii<32>> = ["chr1", "chr2"]
        .iter()
        .map(|s| FixedAscii::<32>::from_ascii(s).unwrap())
        .collect();
    chroms
        .new_dataset::<FixedAscii<32>>()
        .shape(2)
        .create("name")
        .unwrap()
        .write(&names)
        .unwrap();
    chroms
        .new_dataset::<i64>() // int32 from v3 on
        .shape(2)
        .create("length")
        .unwrap()
        .write(&[250_000i64, 100_000])
        .unwrap();

    let bins = root.create_group("bins").unwrap();
    if fixed_width_chrom {
        let chrom: Vec<FixedAscii<32>> = ["chr1", "chr1", "chr2"]
            .iter()
            .map(|s| FixedAscii::<32>::from_ascii(s).unwrap())
            .collect();
        bins.new_dataset::<FixedAscii<32>>()
            .shape(3)
            .create("chrom")
            .unwrap()
            .write(&chrom)
            .unwrap();
    } else {
        let chrom: Vec<VarLenUnicode> = ["chr1", "chr1", "chr2"].iter().map(|s| vlen(s)).collect();
        bins.new_dataset::<VarLenUnicode>()
            .shape(3)
            .create("chrom")
            .unwrap()
            .write(&chrom)
            .unwrap();
    }
    bins.new_dataset::<i64>()
        .shape(3)
        .create("start")
        .unwrap()
        .write(&[0i64, 100_000, 0])
        .unwrap();
    bins.new_dataset::<i64>()
        .shape(3)
        .create("end")
        .unwrap()
        .write(&[100_000i64, 200_000, 100_000])
        .unwrap();

    // v1 pixel ids and counts are int32; the reader widens them.
    let px = root.create_group("pixels").unwrap();
    px.new_dataset::<i32>()
        .shape(2)
        .create("bin1_id")
        .unwrap()
        .write(&[0i32, 1])
        .unwrap();
    px.new_dataset::<i32>()
        .shape(2)
        .create("bin2_id")
        .unwrap()
        .write(&[1i32, 2])
        .unwrap();
    px.new_dataset::<i32>()
        .shape(2)
        .create("count")
        .unwrap()
        .write(&[7i32, 3])
        .unwrap();

    let ix = root.create_group("indexes").unwrap();
    ix.new_dataset::<i32>()
        .shape(3)
        .create("chrom_offset")
        .unwrap()
        .write(&[0i32, 2, 3])
        .unwrap();
    ix.new_dataset::<i32>()
        .shape(4)
        .create("bin1_offset")
        .unwrap()
        .write(&[0i32, 1, 2, 2])
        .unwrap();
}

fn check_v1_cool(path: &std::path::Path) {
    let clr = Cooler::open(path).unwrap();
    assert_eq!(
        clr.chroms().unwrap(),
        vec![
            Chrom {
                name: "chr1".into(),
                length: 250_000
            },
            Chrom {
                name: "chr2".into(),
                length: 100_000
            },
        ]
    );
    assert_eq!(clr.bin_size().unwrap(), Some(100_000));

    // Names mapped through /chroms/name onto the same ids v2 stores directly.
    let bins = clr.bins().unwrap();
    assert_eq!(
        bins.iter().map(|b| b.chrom_id).collect::<Vec<_>>(),
        [0, 0, 1]
    );
    assert_eq!((bins[1].start, bins[1].end), (100_000, 200_000));

    let pixels = clr.pixels().unwrap();
    assert_eq!(
        pixels
            .iter()
            .map(|p| (p.bin1_id, p.bin2_id, p.count))
            .collect::<Vec<_>>(),
        [(0, 1, 7.0), (1, 2, 3.0)]
    );
    assert_eq!(clr.chrom_offset().unwrap(), [0, 2, 3]);
    assert_eq!(clr.bin1_offset().unwrap(), [0, 1, 2, 2]);
}

#[test]
fn reads_a_v1_cool_with_fixed_width_chrom_names() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v1.cool");
    write_v1_cool(&path, true);
    check_v1_cool(&path);
}

#[test]
fn reads_a_v1_cool_with_variable_length_chrom_names() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v1_vlen.cool");
    write_v1_cool(&path, false);
    check_v1_cool(&path);
}

#[test]
fn reads_a_v2_cool_without_storage_mode() {
    // v2 is v1 plus an integer `bins/chrom` and minus `storage-mode`, which
    // only v3 introduced. A missing `storage-mode` must not be required, and
    // the ids must be read directly rather than mapped through `chroms`.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v2.cool");
    write_v1_cool(&path, true);
    {
        let f = File::open_rw(&path).unwrap();
        let root = f.group("/").unwrap();
        let bins = root.group("bins").unwrap();
        bins.unlink("chrom").unwrap();
        bins.new_dataset::<i32>()
            .shape(3)
            .create("chrom")
            .unwrap()
            .write(&[0i32, 0, 1])
            .unwrap();
        root.delete_attr("format-version").unwrap();
        attr_str(&root, "format-version", "2");
    }
    check_v1_cool(&path);
}

#[test]
fn rejects_a_square_storage_mode() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("square.cool");
    {
        let w = CoolerWriter::create(
            &path,
            &[Chrom {
                name: "chr1".into(),
                length: 250_000,
            }],
            100_000,
        )
        .unwrap();
        w.write_pixels(&[Pixel {
            bin1_id: 0,
            bin2_id: 1,
            count: 1.0,
        }])
        .unwrap();
    }
    {
        let f = File::open_rw(&path).unwrap();
        let root = f.group("/").unwrap();
        root.delete_attr("storage-mode").unwrap();
        attr_str(&root, "storage-mode", "square");
    }

    // The reader must refuse, not treat the file as symmetric-upper.
    let err = match Cooler::open(&path) {
        Ok(_) => panic!("square storage-mode should not open"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("storage-mode 'square'"), "{err}");
}

/// Legacy `.mcool` v1: root-level `<zoom level>` groups, no `/resolutions`,
/// no root `format` attribute. Resolution is each group's own `bin-size`.
fn write_legacy_mcool(path: &std::path::Path) -> Vec<Chrom> {
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
    let f = File::create(path).unwrap();
    for (level, bin_size) in [("0", 200_000u32), ("1", 100_000)] {
        let group = f.create_group(level).unwrap();
        let w = CoolerWriter::from_group(group, &chroms, bin_size).unwrap();
        w.write_pixels(&[Pixel {
            bin1_id: 0,
            bin2_id: 1,
            count: 5.0,
        }])
        .unwrap();
    }
    chroms
}

#[test]
fn reads_a_legacy_v1_mcool() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("legacy.mcool");
    let chroms = write_legacy_mcool(&path);

    let mcool = Mcool::open(&path).unwrap();
    assert_eq!(mcool.resolutions().unwrap(), [100_000, 200_000]);
    // The group name is the zoom level, not the bin size.
    assert_eq!(mcool.group_path(100_000).unwrap(), "1");
    assert_eq!(mcool.group_path(200_000).unwrap(), "0");
    assert_eq!(mcool.cooler(100_000).unwrap().chroms().unwrap(), chroms);
    assert_eq!(
        mcool.cooler(200_000).unwrap().bin_size().unwrap(),
        Some(200_000)
    );
    assert!(mcool.cooler(50_000).is_err());

    assert_eq!(
        AnyFile::open(path.to_str().unwrap(), 200_000)
            .unwrap()
            .resolution(),
        200_000
    );
}

#[test]
fn rejects_a_plain_cool_as_mcool() {
    // A root-level `.cool` has no numeric groups; it must not be mistaken for
    // the legacy layout.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain.cool");
    CoolerWriter::create(
        &path,
        &[Chrom {
            name: "chr1".into(),
            length: 250_000,
        }],
        100_000,
    )
    .unwrap()
    .write_pixels(&[])
    .unwrap();
    {
        let f = File::open_rw(&path).unwrap();
        f.group("/").unwrap().delete_attr("format").unwrap();
    }
    let err = match Mcool::open(&path) {
        Ok(_) => panic!("plain .cool should not open as .mcool"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("missing 'format' attribute"), "{err}");
}
