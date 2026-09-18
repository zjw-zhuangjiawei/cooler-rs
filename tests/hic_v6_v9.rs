#![cfg(any())]
//! Version-specific `.hic` read paths: the v6 block body, the v9 block header
//! (two extra per-axis width flags) and the v9 header-recorded
//! normalization-vector index.
//!
//! Fixtures are assembled byte by byte: neither hictk nor cooler-rs writes v6,
//! and hictk's `convert` does not copy norm vectors into a v9 `.hic`, so there
//! is no external writer to borrow.

use std::io::Write;

use byteorder::{LittleEndian, WriteBytesExt};
use cooler_rs::HiCFile;
use flate2::write::ZlibEncoder;
use flate2::Compression;

/// A minimal single-matrix `(0,0)` `.hic` file, one resolution, `blockColumnCount = 1`.
struct Fixture {
    version: i32,
    resolution: u32,
    /// Real (non-`All`) chromosomes: `(name, length)`.
    chroms: Vec<(&'static str, u32)>,
    /// Uncompressed block bodies.
    blocks: Vec<Vec<u8>>,
    /// `(name, header chrom index, values)`.
    norms: Vec<(&'static str, i32, Vec<f64>)>,
}

impl Fixture {
    fn build(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_all(b"HIC\0").unwrap();
        buf.write_i32::<LittleEndian>(self.version).unwrap();
        buf.write_i64::<LittleEndian>(0).unwrap(); // footer position: patched below
        buf.write_all(b"test\0").unwrap();
        if self.version > 8 {
            buf.write_i64::<LittleEndian>(0).unwrap(); // nviPosition: patched below
            buf.write_i64::<LittleEndian>(0).unwrap(); // nviLength
        }
        buf.write_i32::<LittleEndian>(0).unwrap(); // nAttributes
        buf.write_i32::<LittleEndian>(self.chroms.len() as i32)
            .unwrap();
        for (name, length) in &self.chroms {
            buf.write_all(name.as_bytes()).unwrap();
            buf.write_all(b"\0").unwrap();
            if self.version > 8 {
                buf.write_i64::<LittleEndian>(*length as i64).unwrap();
            } else {
                buf.write_i32::<LittleEndian>(*length as i32).unwrap();
            }
        }
        buf.write_i32::<LittleEndian>(1).unwrap(); // nBpRes
        buf.write_i32::<LittleEndian>(self.resolution as i32)
            .unwrap();
        buf.write_i32::<LittleEndian>(0).unwrap(); // nFragRes

        // Blocks, then the matrix record that indexes them.
        let mut block_meta = Vec::new();
        for body in &self.blocks {
            let compressed = zlib(body);
            block_meta.push((buf.len() as i64, compressed.len() as i32));
            buf.write_all(&compressed).unwrap();
        }

        let rec_pos = buf.len() as i64;
        let mut rec = Vec::new();
        rec.write_i32::<LittleEndian>(0).unwrap(); // chr1
        rec.write_i32::<LittleEndian>(0).unwrap(); // chr2
        rec.write_i32::<LittleEndian>(1).unwrap(); // nRes
        rec.write_all(b"BP\0").unwrap();
        rec.write_i32::<LittleEndian>(0).unwrap(); // resIdx
        for _ in 0..4 {
            rec.write_f32::<LittleEndian>(0.0).unwrap(); // sumCounts, occupied, stdDev, p95
        }
        rec.write_i32::<LittleEndian>(self.resolution as i32)
            .unwrap(); // binSize
        rec.write_i32::<LittleEndian>(500).unwrap(); // blockSize
        rec.write_i32::<LittleEndian>(1).unwrap(); // blockColumnCount
        rec.write_i32::<LittleEndian>(block_meta.len() as i32)
            .unwrap(); // blockCount
        for (number, (pos, size)) in block_meta.iter().enumerate() {
            rec.write_i32::<LittleEndian>(number as i32).unwrap();
            rec.write_i64::<LittleEndian>(*pos).unwrap();
            rec.write_i32::<LittleEndian>(*size).unwrap();
        }
        buf.write_all(&rec).unwrap();

        // Footer: master index only.
        let footer_pos = buf.len() as i64;
        if self.version > 8 {
            buf.write_i64::<LittleEndian>(0).unwrap(); // footer size (unread)
        } else {
            buf.write_i32::<LittleEndian>(0).unwrap();
        }
        buf.write_i32::<LittleEndian>(1).unwrap(); // nEntries
        buf.write_all(b"0_0\0").unwrap();
        buf.write_i64::<LittleEndian>(rec_pos).unwrap();
        buf.write_i32::<LittleEndian>(rec.len() as i32).unwrap();

        // v9: norm index + vector bodies, placed after the footer. Only the
        // header position reaches them, which is the point of the test.
        let mut nvi_pos = 0i64;
        if !self.norms.is_empty() {
            nvi_pos = buf.len() as i64;
            let bodies: Vec<Vec<u8>> = self
                .norms
                .iter()
                .map(|(_, _, values)| {
                    let mut b = Vec::new();
                    b.write_i64::<LittleEndian>(values.len() as i64).unwrap();
                    for v in values {
                        b.write_f32::<LittleEndian>(*v as f32).unwrap();
                    }
                    b
                })
                .collect();
            // name + nul, chrIdx, "BP" + nul, resolution, position, nBytes
            let index_size: i64 = 4 + self
                .norms
                .iter()
                .map(|(name, _, _)| name.len() as i64 + 1 + 4 + 3 + 4 + 8 + 8)
                .sum::<i64>();

            let mut index = Vec::new();
            index
                .write_i32::<LittleEndian>(self.norms.len() as i32)
                .unwrap();
            let mut pos = nvi_pos + index_size;
            for ((name, chrom_idx, _), body) in self.norms.iter().zip(&bodies) {
                index.write_all(name.as_bytes()).unwrap();
                index.write_all(b"\0").unwrap();
                index.write_i32::<LittleEndian>(*chrom_idx).unwrap();
                index.write_all(b"BP\0").unwrap();
                index
                    .write_i32::<LittleEndian>(self.resolution as i32)
                    .unwrap();
                index.write_i64::<LittleEndian>(pos).unwrap();
                index.write_i64::<LittleEndian>(body.len() as i64).unwrap();
                pos += body.len() as i64;
            }
            buf.write_all(&index).unwrap();
            for body in &bodies {
                buf.write_all(body).unwrap();
            }
        }

        // Patch the two positions recorded in the header.
        buf[8..16].copy_from_slice(&footer_pos.to_le_bytes());
        if self.version > 8 {
            buf[21..29].copy_from_slice(&nvi_pos.to_le_bytes());
        }
        buf
    }

    fn write_to(&self, path: &std::path::Path) -> std::path::PathBuf {
        std::fs::write(path, self.build()).unwrap();
        path.to_path_buf()
    }
}

/// v6 block body: `nRecords` then `nRecords` × (i32 bin_x, i32 bin_y, f32 count),
/// no per-block header, no offsets to add.
fn v6_block(records: &[(i32, i32, f64)]) -> Vec<u8> {
    let mut b = Vec::new();
    b.write_i32::<LittleEndian>(records.len() as i32).unwrap();
    for &(x, y, count) in records {
        b.write_i32::<LittleEndian>(x).unwrap();
        b.write_i32::<LittleEndian>(y).unwrap();
        b.write_f32::<LittleEndian>(count as f32).unwrap();
    }
    b
}

/// v7+ block body: header, then one type-1 ("list of rows") section. `rows` are
/// block-relative, keyed by block-relative `bin_y`.
fn row_list_block(
    version: i32,
    short_x: bool,
    short_y: bool,
    short_counts: bool,
    offsets: (i32, i32),
    rows: &[(i32, Vec<(i32, f64)>)],
) -> Vec<u8> {
    let mut b = Vec::new();
    let n_records: i32 = rows.iter().map(|(_, cols)| cols.len() as i32).sum();
    b.write_i32::<LittleEndian>(n_records).unwrap();
    b.write_i32::<LittleEndian>(offsets.0).unwrap();
    b.write_i32::<LittleEndian>(offsets.1).unwrap();
    b.write_u8(u8::from(!short_counts)).unwrap();
    if version > 8 {
        b.write_u8(u8::from(!short_x)).unwrap();
        b.write_u8(u8::from(!short_y)).unwrap();
    }
    b.write_u8(1).unwrap(); // representation: list of rows

    write_bin(&mut b, short_y, rows.len() as i32);
    for (bin_y, cols) in rows {
        write_bin(&mut b, short_y, *bin_y);
        write_bin(&mut b, short_x, cols.len() as i32);
        for (bin_x, count) in cols {
            write_bin(&mut b, short_x, *bin_x);
            if short_counts {
                b.write_i16::<LittleEndian>(*count as i16).unwrap();
            } else {
                b.write_f32::<LittleEndian>(*count as f32).unwrap();
            }
        }
    }
    b
}

fn write_bin<W: Write>(w: &mut W, short: bool, v: i32) {
    if short {
        w.write_i16::<LittleEndian>(v as i16).unwrap();
    } else {
        w.write_i32::<LittleEndian>(v).unwrap();
    }
}

fn zlib(data: &[u8]) -> Vec<u8> {
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

fn open(fixture: &Fixture, name: &str) -> HiCFile {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture.write_to(&dir.path().join(name));
    HiCFile::open(&path).unwrap()
}

#[test]
fn reads_a_v6_block_body() {
    let fixture = Fixture {
        version: 6,
        resolution: 100_000,
        chroms: vec![("chr1", 300_000), ("chr2", 200_000)],
        blocks: vec![v6_block(&[(0, 0, 4.0), (1, 2, 7.0)])],
        norms: Vec::new(),
    };
    let hic = open(&fixture, "v6.hic");

    let pixels = hic.pixels(100_000).unwrap();
    assert_eq!(pixels.len(), 2);
    assert_eq!((pixels[0].bin1_id, pixels[0].bin2_id), (0, 0));
    assert_eq!(pixels[0].count, 4.0);
    assert_eq!((pixels[1].bin1_id, pixels[1].bin2_id), (1, 2));
    assert_eq!(pixels[1].count, 7.0);
}

#[test]
fn reads_a_v9_block_with_32bit_bins() {
    // hictk writes every v9 block this way: i32 bins, f32 counts.
    let fixture = Fixture {
        version: 9,
        resolution: 100_000,
        chroms: vec![("chr1", 3_000_000), ("chr2", 2_000_000)],
        blocks: vec![row_list_block(
            9,
            false,
            false,
            false,
            (10, 20),
            &[(0, vec![(1, 3.5)]), (4, vec![(2, 8.0)])],
        )],
        norms: Vec::new(),
    };
    let hic = open(&fixture, "v9_long.hic");

    let pixels = hic.pixels(100_000).unwrap();
    assert_eq!(pixels.len(), 2);
    assert_eq!((pixels[0].bin1_id, pixels[0].bin2_id), (11, 20));
    assert_eq!(pixels[0].count, 3.5);
    assert_eq!((pixels[1].bin1_id, pixels[1].bin2_id), (12, 24));
    assert_eq!(pixels[1].count, 8.0);
}

#[test]
fn reads_a_v9_block_with_16bit_bins() {
    let fixture = Fixture {
        version: 9,
        resolution: 100_000,
        chroms: vec![("chr1", 300_000), ("chr2", 200_000)],
        blocks: vec![row_list_block(
            9,
            true,
            true,
            true,
            (0, 0),
            &[(2, vec![(0, 5.0), (1, 6.0)])],
        )],
        norms: Vec::new(),
    };
    let hic = open(&fixture, "v9_short.hic");

    let pixels = hic.pixels(100_000).unwrap();
    assert_eq!(pixels.len(), 2);
    assert_eq!((pixels[0].bin1_id, pixels[0].bin2_id), (0, 2));
    assert_eq!(pixels[0].count, 5.0);
    assert_eq!((pixels[1].bin1_id, pixels[1].bin2_id), (1, 2));
    assert_eq!(pixels[1].count, 6.0);
}

#[test]
fn reads_a_v8_block_body() {
    // Guards the pre-v9 header size: no per-axis width flags.
    let fixture = Fixture {
        version: 8,
        resolution: 100_000,
        chroms: vec![("chr1", 300_000), ("chr2", 200_000)],
        blocks: vec![row_list_block(
            8,
            true,
            true,
            false,
            (0, 0),
            &[(1, vec![(1, 2.5)])],
        )],
        norms: Vec::new(),
    };
    let hic = open(&fixture, "v8.hic");

    let pixels = hic.pixels(100_000).unwrap();
    assert_eq!(pixels.len(), 1);
    assert_eq!((pixels[0].bin1_id, pixels[0].bin2_id), (1, 1));
    assert_eq!(pixels[0].count, 2.5);
}

#[test]
fn reads_the_v9_norm_index_from_the_header_position() {
    let fixture = Fixture {
        version: 9,
        resolution: 100_000,
        chroms: vec![("chr1", 300_000), ("chr2", 200_000)],
        blocks: vec![v6_block(&[(0, 0, 1.0)])],
        norms: vec![("VC", 0, vec![1.0, 2.0, 3.0])],
    };
    let hic = open(&fixture, "v9_norm.hic");

    assert_eq!(hic.avail_normalizations().unwrap(), vec!["VC"]);
    let weights = hic.norm_vector(100_000, "chr1", "VC").unwrap().unwrap();
    assert_eq!(weights, vec![1.0, 2.0, 3.0]);
}

#[test]
fn rejects_unsupported_versions() {
    for version in [5, 10] {
        let fixture = Fixture {
            version,
            resolution: 100_000,
            chroms: vec![("chr1", 300_000)],
            blocks: vec![v6_block(&[(0, 0, 1.0)])],
            norms: Vec::new(),
        };
        let dir = tempfile::tempdir().unwrap();
        let path = fixture.write_to(&dir.path().join("bad.hic"));
        let err = match HiCFile::open(&path) {
            Ok(_) => panic!("v{version} should not open"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("not supported"), "v{version}: {err}");
    }
}
