//! Identity / self-consistency tests for `zoomify`.

use cooler_rs::{
    coarsen_pixels, nice_resolutions, pow2_resolutions, zoomify_cooler, Chrom, CoolerWriter, Mcool,
    Pixel, ZoomifyParams,
};

fn px(bin1_id: i64, bin2_id: i64, count: f64) -> Pixel {
    Pixel {
        bin1_id,
        bin2_id,
        count,
    }
}

#[test]
fn coarsen_sums_factor_blocks() {
    let pixels = vec![px(0, 0, 1.0), px(0, 1, 2.0), px(1, 1, 3.0), px(2, 3, 4.0)];
    let out = coarsen_pixels(pixels, 2);
    assert_eq!(out, vec![px(0, 0, 6.0), px(1, 1, 4.0)]);
}

#[test]
fn resolution_progressions() {
    assert_eq!(pow2_resolutions(1000, 8000), vec![1000, 2000, 4000, 8000]);
    assert_eq!(
        nice_resolutions(1000, 10_000),
        vec![1000, 2000, 5000, 10_000]
    );
    assert_eq!(
        nice_resolutions(1000, 100_000),
        vec![1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]
    );
}

#[test]
fn zoomify_pools_per_chromosome() {
    let dir = tempfile::tempdir().unwrap();
    let cool_path = dir.path().join("in.cool");
    let mcool_path = dir.path().join("out.mcool");

    // chrA: 3 bins (0,1,2); chrB: 2 bins (3,4). Base resolution 10 bp.
    let chroms = vec![
        Chrom {
            name: "chrA".into(),
            length: 30,
        },
        Chrom {
            name: "chrB".into(),
            length: 20,
        },
    ];
    let pixels = vec![
        px(0, 0, 1.0),
        px(0, 1, 2.0),
        px(1, 1, 3.0),
        px(2, 2, 4.0),
        px(3, 3, 10.0),
        px(3, 4, 20.0),
        px(4, 4, 30.0),
        px(2, 3, 100.0), // trans chrA-chrB
    ];
    let writer = CoolerWriter::create(&cool_path, &chroms, 10).unwrap();
    writer.write_pixels(&pixels).unwrap();

    // Coarsen to 20 bp (factor 2), dropping the base resolution.
    zoomify_cooler(
        &cool_path,
        &mcool_path,
        &ZoomifyParams {
            resolutions: vec![20],
            copy_base_resolution: false,
            ..ZoomifyParams::default()
        },
    )
    .unwrap();

    let mcool = Mcool::open(&mcool_path).unwrap();
    assert_eq!(mcool.resolutions().unwrap(), vec![20]);

    let coarse = mcool.cooler(20).unwrap();
    let out = coarse.pixels().unwrap();
    assert_eq!(
        out,
        vec![
            px(0, 0, 6.0),
            px(1, 1, 4.0),
            px(1, 2, 100.0),
            px(2, 2, 60.0)
        ]
    );

    // Count is conserved under coarsening.
    let base_sum: f64 = pixels.iter().map(|p| p.count).sum();
    let coarse_sum: f64 = out.iter().map(|p| p.count).sum();
    assert_eq!(base_sum, coarse_sum);
}

#[test]
fn zoomify_copies_base_by_default() {
    let dir = tempfile::tempdir().unwrap();
    let cool_path = dir.path().join("in.cool");
    let mcool_path = dir.path().join("out.mcool");

    let chroms = vec![Chrom {
        name: "chr1".into(),
        length: 40,
    }];
    let writer = CoolerWriter::create(&cool_path, &chroms, 10).unwrap();
    writer
        .write_pixels(&[px(0, 0, 1.0), px(1, 2, 2.0), px(2, 3, 3.0)])
        .unwrap();

    zoomify_cooler(
        &cool_path,
        &mcool_path,
        &ZoomifyParams {
            resolutions: vec![20],
            ..ZoomifyParams::default()
        },
    )
    .unwrap();

    let mcool = Mcool::open(&mcool_path).unwrap();
    assert_eq!(mcool.resolutions().unwrap(), vec![10, 20]);
}
