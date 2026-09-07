//! Read-only `.hic` reader tests against `tests/data/4DNFIOTPSS3L.hic`
//! (Drosophila, v8). Skipped when the file is absent (it is gitignored).

use std::path::Path;

use cooler_rs::{Chrom, File, HiCFile, HicWriter, Pixel, Region};

fn fixture() -> Option<std::path::PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/4DNFIOTPSS3L.hic");
    p.exists().then_some(p)
}

#[test]
fn reads_header() {
    let Some(path) = fixture() else {
        eprintln!("skipping: 4DNFIOTPSS3L.hic not present");
        return;
    };
    let hic = HiCFile::open(&path).unwrap();
    assert_eq!(hic.version(), 8);
    assert!(!hic.genome_id().is_empty());

    let resolutions = hic.resolutions();
    assert!(resolutions.contains(&5000));
    assert!(resolutions.contains(&10000000));

    let chroms = hic.chromosomes();
    let names: Vec<&str> = chroms.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["2L", "2R", "3L", "3R", "4", "X", "Y"]);
    // Lengths must match the mcool twin (no `chr` prefix in .hic).
    assert_eq!(chroms[0].length, 23513712);
    assert_eq!(chroms[6].length, 3667352);
}

#[test]
fn reads_pixels() {
    let Some(path) = fixture() else {
        eprintln!("skipping: 4DNFIOTPSS3L.hic not present");
        return;
    };
    let hic = HiCFile::open(&path).unwrap();

    let pixels = hic.pixels(5000).unwrap();
    assert!(!pixels.is_empty());

    // symmetric-upper invariant
    for p in &pixels {
        assert!(p.bin1_id <= p.bin2_id);
        assert!(p.count >= 0.0);
    }

    // chr2L is the first non-All chromosome; its bins span [0, 4703) at 5 kb.
    let n_bins_2l = (23513712_u64).div_ceil(5000);
    let chr2l_sum: f64 = pixels
        .iter()
        .filter(|p| p.bin2_id < n_bins_2l as i64)
        .map(|p| p.count)
        .sum();
    let chr2l_nnz = pixels
        .iter()
        .filter(|p| p.bin2_id < n_bins_2l as i64)
        .count();

    // Identity against the reference straw reader (hic side, upper triangle).
    assert_eq!(chr2l_nnz, 2_676_608);
    assert_eq!(chr2l_sum, 19_009_928.0);
}

#[test]
fn roundtrips_pixels() {
    let Some(path) = fixture() else {
        eprintln!("skipping: 4DNFIOTPSS3L.hic not present");
        return;
    };
    let hic = HiCFile::open(&path).unwrap();
    let chroms = hic.chromosomes();
    let genome_id = hic.genome_id().to_string();
    let pixels = hic.pixels(5000).unwrap();

    let tmp = std::env::temp_dir().join(format!("cooler_rs_roundtrip_{}.hic", std::process::id()));
    {
        let mut w = HicWriter::create(&tmp, &genome_id, &chroms, &[5000]).unwrap();
        w.add_pixels(5000, &pixels).unwrap();
        w.finalize().unwrap();
    }

    let hic2 = HiCFile::open(&tmp).unwrap();
    assert_eq!(hic2.chromosomes(), chroms);
    assert!(hic2.resolutions().contains(&5000));
    let p2 = hic2.pixels(5000).unwrap();

    let mut a: Vec<_> = pixels
        .iter()
        .map(|p| (p.bin1_id, p.bin2_id, p.count.to_bits()))
        .collect();
    let mut b: Vec<_> = p2
        .iter()
        .map(|p| (p.bin1_id, p.bin2_id, p.count.to_bits()))
        .collect();
    a.sort_unstable();
    b.sort_unstable();
    assert_eq!(a.len(), b.len());
    assert_eq!(a, b);

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn reads_normalization_vectors() {
    let Some(path) = fixture() else {
        eprintln!("skipping: 4DNFIOTPSS3L.hic not present");
        return;
    };
    let hic = HiCFile::open(&path).unwrap();
    assert_eq!(hic.avail_normalizations().unwrap(), ["KR", "VC", "VC_SQRT"]);
    // chr2L = 23,513,712 bp -> 4703 bins at 5 kb.
    let kr = hic.norm_vector(5000, "2L", "KR").unwrap().unwrap();
    assert_eq!(kr.len(), 4703);
    assert!(hic.norm_vector(5000, "2L", "VC").unwrap().is_some());
    assert!(hic.norm_vector(5000, "2L", "SCALE").unwrap().is_none());
}

#[test]
fn file_fetch_normalizes() {
    let Some(path) = fixture() else {
        eprintln!("skipping: 4DNFIOTPSS3L.hic not present");
        return;
    };
    let f = File::open(path.to_str().unwrap(), 5000).unwrap();
    let region = Region::chrom("2L");
    let raw = f.fetch(&region, None).unwrap();
    let kr = f.fetch(&region, Some("KR")).unwrap();
    assert_eq!(raw.len(), 2_676_608);
    assert_eq!(raw.len(), kr.len());
    assert_ne!(raw[0].count, kr[0].count);
}

#[test]
fn writes_and_reads_normalization_vectors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("norms.hic");
    let chroms = vec![
        Chrom {
            name: "chr1".into(),
            length: 10_000,
        },
        Chrom {
            name: "chr2".into(),
            length: 10_000,
        },
    ];
    {
        let mut w = HicWriter::create(&path, "test", &chroms, &[5000]).unwrap();
        w.add_pixels(
            5000,
            &[Pixel {
                bin1_id: 0,
                bin2_id: 1,
                count: 10.0,
            }],
        )
        .unwrap();
        w.add_normalization_vectors(
            5000,
            "KR",
            &[
                ("chr1".to_string(), vec![2.0, 3.0]),
                ("chr2".to_string(), vec![4.0, 5.0]),
            ],
        )
        .unwrap();
        w.finalize().unwrap();
    }
    let hic = HiCFile::open(&path).unwrap();
    assert_eq!(hic.avail_normalizations().unwrap(), ["KR"]);
    assert_eq!(
        hic.norm_vector(5000, "chr1", "KR").unwrap().unwrap(),
        vec![2.0, 3.0]
    );
    assert_eq!(
        hic.norm_vector(5000, "chr2", "KR").unwrap().unwrap(),
        vec![4.0, 5.0]
    );
    assert!(hic.norm_vector(5000, "chr1", "VC").unwrap().is_none());
}
