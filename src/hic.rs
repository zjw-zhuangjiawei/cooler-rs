//! Read-only `.hic` (Hi-C) contact matrix reader.
//!
//! Implements format v6-v8 (the layout documented in `HiCFormatV8.md` and
//! followed by the reference `straw` reader). Blocks are zlib (deflate)
//! compressed. Only base-pair resolution matrices are supported; fragment
//! resolution levels, if present, are skipped.
//!
//! Data is returned in the same shape as [`Cooler`](crate::Cooler): a flat
//! vector of [`Pixel`]s in `symmetric-upper` form (`bin1_id <= bin2_id`), with
//! bin ids spanning the non-`All` chromosomes in header order.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Mutex;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;

use crate::error::{Error, Result};
use crate::types::{Chrom, Pixel};

/// Reader for a `.hic` file.
pub struct HiCFile {
    // `Mutex` (not `RefCell`) so a `HiCFile` is `Sync`: arrowhead reads one
    // `File` from many threads at once. Each method takes the lock for the
    // duration of a single call; nothing re-enters it, so there is no deadlock.
    file: Mutex<File>,
    version: i32,
    genome_id: String,
    /// Chromosomes in header order, including the `All` pseudo-chromosome.
    chroms: Vec<Chrom>,
    resolutions: Vec<u32>,
    index: Vec<IndexEntry>,
    /// File offset of the footer (master index, expected values, norm index).
    footer_pos: u64,
    /// Header attribute dictionary (key/value string pairs).
    attributes: BTreeMap<String, String>,
}

/// One `(chr1, chr2)` matrix record from the footer master index.
#[derive(Clone, Copy)]
struct IndexEntry {
    chrom1: usize,
    chrom2: usize,
    position: u64,
    #[allow(dead_code)]
    size: u32,
}

/// Per-resolution matrix metadata (block index).
struct MatrixMeta {
    blocks: Vec<Block>,
}

/// A single compressed block's location.
struct Block {
    position: u64,
    size: u32,
}

impl HiCFile {
    /// Open and parse a `.hic` file.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path)?;

        // Magic: "HIC\0".
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"HIC\0" {
            return Err(Error::Format(
                "not a .hic file (missing \"HIC\" magic)".to_string(),
            ));
        }

        let version = file.read_i32::<LittleEndian>()?;
        if version < 6 {
            return Err(Error::Format(format!(
                ".hic version {version} is not supported (need >= 6)"
            )));
        }

        let master_index_pos = file.read_i64::<LittleEndian>()? as u64;
        let genome_id = read_cstring(&mut file)?;

        // v9+ stores an extra (nviPosition, nviLength) pair; we target <= v8.
        if version > 8 {
            file.read_i64::<LittleEndian>()?;
            file.read_i64::<LittleEndian>()?;
        }

        // Attribute dictionary (key/value string pairs).
        let n_attrs = file.read_i32::<LittleEndian>()?;
        let mut attributes = BTreeMap::new();
        for _ in 0..n_attrs {
            let key = read_cstring(&mut file)?;
            let value = read_cstring(&mut file)?;
            attributes.insert(key, value);
        }

        // Chromosome list (length is i32 for v8, i64 for v9).
        let n_chroms = file.read_i32::<LittleEndian>()?;
        let mut chroms = Vec::with_capacity(n_chroms as usize);
        for _ in 0..n_chroms {
            let name = read_cstring(&mut file)?;
            let length = if version > 8 {
                file.read_i64::<LittleEndian>()?
            } else {
                file.read_i32::<LittleEndian>()? as i64
            };
            chroms.push(Chrom {
                name,
                length: length as i32,
            });
        }

        // Base-pair resolutions.
        let n_bp = file.read_i32::<LittleEndian>()?;
        let mut resolutions = Vec::with_capacity(n_bp as usize);
        for _ in 0..n_bp {
            resolutions.push(file.read_i32::<LittleEndian>()? as u32);
        }

        // Fragment resolutions (Hi-C files use none); skip their site tables.
        let n_frag = file.read_i32::<LittleEndian>()?;
        for _ in 0..n_frag {
            file.read_i32::<LittleEndian>()?;
        }
        if n_frag > 0 {
            for _ in 0..n_chroms {
                let n_sites = file.read_i32::<LittleEndian>()?;
                for _ in 0..n_sites {
                    file.read_i32::<LittleEndian>()?;
                }
            }
        }

        let index = read_master_index(&mut file, master_index_pos, version)?;

        Ok(HiCFile {
            file: Mutex::new(file),
            version,
            genome_id,
            chroms,
            resolutions,
            index,
            footer_pos: master_index_pos,
            attributes,
        })
    }

    /// The file format version.
    pub fn version(&self) -> i32 {
        self.version
    }

    /// The genome identifier string.
    pub fn genome_id(&self) -> &str {
        &self.genome_id
    }

    /// Available base-pair resolutions (bin sizes), in file order.
    pub fn resolutions(&self) -> &[u32] {
        &self.resolutions
    }

    /// Non-`All` chromosomes, in header order.
    pub fn chromosomes(&self) -> Vec<Chrom> {
        self.chroms
            .iter()
            .filter(|c| !is_all_chrom(&c.name))
            .cloned()
            .collect()
    }

    /// Read all pixels at `resolution` across every non-`All` chromosome pair.
    ///
    /// Returns `symmetric-upper` pixels (`bin1_id <= bin2_id`) over the
    /// non-`All` chromosomes.
    pub fn pixels(&self, resolution: u32) -> Result<Vec<Pixel>> {
        // Map header chromosome index -> non-All index (skip the "All" chrom).
        let real_indices: Vec<usize> = (0..self.chroms.len())
            .filter(|&i| !is_all_chrom(&self.chroms[i].name))
            .collect();
        let mut header_to_cooler = vec![usize::MAX; self.chroms.len()];
        for (cooler_id, &header_id) in real_indices.iter().enumerate() {
            header_to_cooler[header_id] = cooler_id;
        }

        // Cumulative bin count per non-All chromosome.
        let mut offsets = vec![0i64; real_indices.len() + 1];
        for (i, &header_id) in real_indices.iter().enumerate() {
            let n_bins = (self.chroms[header_id].length as u64).div_ceil(resolution as u64) as i64;
            offsets[i + 1] = offsets[i] + n_bins;
        }

        let entries = self.index.clone();
        let mut file = self.file.lock().expect("hic lock poisoned");
        let mut pixels = Vec::new();
        for entry in &entries {
            let c1 = header_to_cooler[entry.chrom1];
            let c2 = header_to_cooler[entry.chrom2];
            if c1 == usize::MAX || c2 == usize::MAX {
                continue; // involves the "All" pseudo-chromosome
            }
            let meta = read_matrix(&mut file, *entry, resolution)?;
            for block in &meta.blocks {
                for (bin_x, bin_y, count) in read_block(&mut file, block)? {
                    pixels.push(Pixel {
                        bin1_id: offsets[c1] + bin_x as i64,
                        bin2_id: offsets[c2] + bin_y as i64,
                        count,
                    });
                }
            }
        }
        Ok(pixels)
    }

    /// Header attribute dictionary.
    pub fn attributes(&self) -> &BTreeMap<String, String> {
        &self.attributes
    }

    /// Names of the normalization vectors stored in the footer (deduped,
    /// `BP` unit only), e.g. `["KR", "VC", "SCALE"]`.
    pub fn avail_normalizations(&self) -> Result<Vec<String>> {
        let mut names: Vec<String> = self
            .read_norm_entries()?
            .into_iter()
            .filter(|e| e.unit == "BP")
            .map(|e| e.name)
            .collect();
        names.sort();
        names.dedup();
        Ok(names)
    }

    /// Read one normalization vector (divisive) for a chromosome, if present.
    pub fn norm_vector(
        &self,
        resolution: u32,
        chrom: &str,
        name: &str,
    ) -> Result<Option<Vec<f64>>> {
        let chrom_idx = self
            .chroms
            .iter()
            .position(|c| c.name == chrom)
            .ok_or_else(|| Error::InvalidInput(format!("unknown sequence label: {chrom}")))?
            as i32;
        let entry = self.read_norm_entries()?.into_iter().find(|e| {
            e.name == name
                && e.resolution == resolution
                && e.chrom_idx == chrom_idx
                && e.unit == "BP"
        });
        let Some(entry) = entry else {
            return Ok(None);
        };
        let mut file = self.file.lock().expect("hic lock poisoned");
        let f: &mut File = &mut file;
        f.seek(SeekFrom::Start(entry.position))?;
        let n_values = read_n_values(f, self.version)? as usize;
        // The stored vector can carry a few trailing zeros past the real bin
        // count (a known .hic quirk); read only the expected values.
        let expected =
            (self.chroms[chrom_idx as usize].length as u64).div_ceil(resolution as u64) as usize;
        if n_values < expected {
            return Err(Error::Format(format!(
                "normalization vector truncated: {n_values} values, expected {expected}"
            )));
        }
        let mut values = Vec::with_capacity(expected);
        for _ in 0..expected {
            let v = if self.version > 8 {
                f.read_f32::<LittleEndian>()? as f64
            } else {
                f.read_f64::<LittleEndian>()?
            };
            values.push(v);
        }
        Ok(Some(values))
    }

    /// Walk the footer to the normalization-vector index and return its entries.
    fn read_norm_entries(&self) -> Result<Vec<NormEntry>> {
        let mut file = self.file.lock().expect("hic lock poisoned");
        let f: &mut File = &mut file;
        f.seek(SeekFrom::Start(self.footer_pos))?;
        // Footer size in bytes.
        if self.version > 8 {
            f.read_i64::<LittleEndian>()?;
        } else {
            f.read_i32::<LittleEndian>()?;
        }
        // Master index.
        let n = f.read_i32::<LittleEndian>()?;
        for _ in 0..n {
            read_cstring(f)?;
            f.read_i64::<LittleEndian>()?;
            f.read_i32::<LittleEndian>()?;
        }
        // A footer with no expected/norm sections ends here (older writers).
        let file_len = f.metadata()?.len();
        if f.stream_position()? == file_len {
            return Ok(Vec::new());
        }
        // Expected value vectors, then normalized expected value vectors.
        skip_expected_section(f, self.version, false)?;
        skip_expected_section(f, self.version, true)?;
        // Normalization vector index.
        let n = f.read_i32::<LittleEndian>()?;
        let mut out = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let name = read_cstring(f)?;
            let chrom_idx = f.read_i32::<LittleEndian>()?;
            let unit = read_cstring(f)?;
            let resolution = f.read_i32::<LittleEndian>()? as u32;
            let position = f.read_i64::<LittleEndian>()? as u64;
            let _size = if self.version > 8 {
                f.read_i64::<LittleEndian>()? as u64
            } else {
                f.read_i32::<LittleEndian>()? as u32 as u64
            };
            out.push(NormEntry {
                name,
                chrom_idx,
                unit,
                resolution,
                position,
            });
        }
        Ok(out)
    }
}

/// `"All"` (case-insensitive) is the genome-wide pseudo-chromosome.
fn is_all_chrom(name: &str) -> bool {
    name.eq_ignore_ascii_case("All")
}

/// Read the footer master index at `position`.
fn read_master_index(file: &mut File, position: u64, version: i32) -> Result<Vec<IndexEntry>> {
    file.seek(SeekFrom::Start(position))?;
    if version > 8 {
        file.read_i64::<LittleEndian>()?;
    } else {
        file.read_i32::<LittleEndian>()?;
    }

    let n_entries = file.read_i32::<LittleEndian>()?;
    let mut index = Vec::with_capacity(n_entries as usize);
    for _ in 0..n_entries {
        let key = read_cstring(file)?;
        let (chrom1, chrom2) = parse_index_key(&key)?;
        let position = file.read_i64::<LittleEndian>()? as u64;
        let size = file.read_i32::<LittleEndian>()? as u32;
        index.push(IndexEntry {
            chrom1,
            chrom2,
            position,
            size,
        });
    }
    Ok(index)
}

/// Master-index keys are `"<chr1Idx>_<chr2Idx>"`.
fn parse_index_key(key: &str) -> Result<(usize, usize)> {
    let (a, b) = key.split_once('_').ok_or_else(|| {
        Error::Format(format!(
            "invalid master index key '{key}' (expected \"c1_c2\")"
        ))
    })?;
    let c1 = a
        .parse::<usize>()
        .map_err(|e| Error::Format(format!("invalid chromosome index in '{key}': {e}")))?;
    let c2 = b
        .parse::<usize>()
        .map_err(|e| Error::Format(format!("invalid chromosome index in '{key}': {e}")))?;
    Ok((c1, c2))
}

/// One normalization-vector entry from the footer index.
struct NormEntry {
    name: String,
    chrom_idx: i32,
    unit: String,
    resolution: u32,
    position: u64,
}

/// Number-of-values prefix shared by expected and normalization vectors.
fn read_n_values(file: &mut File, version: i32) -> Result<i64> {
    Ok(if version > 8 {
        file.read_i64::<LittleEndian>()?
    } else {
        file.read_i32::<LittleEndian>()? as i64
    })
}

/// Skip one expected-value-vector section (`normalized = false` is the raw
/// expected vectors, `true` the normalized ones).
fn skip_expected_section(file: &mut File, version: i32, normalized: bool) -> Result<()> {
    let value_size: i64 = if version > 8 { 4 } else { 8 };
    let n = file.read_i32::<LittleEndian>()?;
    for _ in 0..n {
        if normalized {
            read_cstring(file)?; // normalization type
        }
        read_cstring(file)?; // unit
        file.read_i32::<LittleEndian>()?; // resolution
        let n_values = read_n_values(file, version)?;
        file.seek(SeekFrom::Current(n_values * value_size))?; // skip values
        let n_factors = file.read_i32::<LittleEndian>()?;
        for _ in 0..n_factors {
            file.read_i32::<LittleEndian>()?; // chromIdx
            file.seek(SeekFrom::Current(value_size))?; // skip factor
        }
    }
    Ok(())
}

/// Read the matrix metadata + block index for `entry` at `resolution`.
fn read_matrix(file: &mut File, entry: IndexEntry, resolution: u32) -> Result<MatrixMeta> {
    file.seek(SeekFrom::Start(entry.position))?;
    let _chr1 = file.read_i32::<LittleEndian>()?;
    let _chr2 = file.read_i32::<LittleEndian>()?;
    let n_res = file.read_i32::<LittleEndian>()?;

    for _ in 0..n_res {
        let unit = read_cstring(file)?;
        let _res_idx = file.read_i32::<LittleEndian>()?;
        let _sum_counts = file.read_f32::<LittleEndian>()?;
        let _occupied = file.read_f32::<LittleEndian>()?;
        let _std_dev = file.read_f32::<LittleEndian>()?;
        let _percent95 = file.read_f32::<LittleEndian>()?;
        let bin_size = file.read_i32::<LittleEndian>()? as u32;
        let _block_size = file.read_i32::<LittleEndian>()? as u32;
        let _block_column_count = file.read_i32::<LittleEndian>()? as u32;
        let block_count = file.read_i32::<LittleEndian>()?;

        let mut blocks = Vec::with_capacity(block_count as usize);
        for _ in 0..block_count {
            let _number = file.read_i32::<LittleEndian>()?;
            let position = file.read_i64::<LittleEndian>()? as u64;
            let size = file.read_i32::<LittleEndian>()? as u32;
            blocks.push(Block { position, size });
        }

        if unit == "BP" && bin_size == resolution {
            return Ok(MatrixMeta { blocks });
        }
    }

    Err(Error::Format(format!(
        "resolution {resolution} not found for chromosome pair ({}, {})",
        entry.chrom1, entry.chrom2
    )))
}

/// Decompress and decode one block into `(bin_x, bin_y, count)` records.
///
/// `bin_x`/`bin_y` are block-relative (already offset by the block's origin).
fn read_block(file: &mut File, block: &Block) -> Result<Vec<(i32, i32, f64)>> {
    if block.size == 0 {
        return Ok(Vec::new());
    }

    file.seek(SeekFrom::Start(block.position))?;
    let mut compressed = vec![0u8; block.size as usize];
    file.read_exact(&mut compressed)?;
    let mut decoder = ZlibDecoder::new(&compressed[..]);
    let mut raw = Vec::new();
    decoder
        .read_to_end(&mut raw)
        .map_err(|e| Error::Format(format!("zlib decompression failed: {e}")))?;

    let mut cur = Cursor::new(raw);
    let _n_records = cur.read_i32::<LittleEndian>()?;
    let bin_x_offset = cur.read_i32::<LittleEndian>()?;
    let bin_y_offset = cur.read_i32::<LittleEndian>()?;
    // 0 => short counts, non-zero => float counts (inverted from the spec's
    // `useFloat` flag, matching the reference straw reader).
    let use_short = cur.read_u8()? == 0;
    let representation = cur.read_u8()?;

    let mut out = Vec::new();
    match representation {
        1 => {
            // Sparse "list of rows".
            let row_count = cur.read_i16::<LittleEndian>()?;
            for _ in 0..row_count {
                let bin_y = bin_y_offset + cur.read_i16::<LittleEndian>()? as i32;
                let col_count = cur.read_i16::<LittleEndian>()?;
                for _ in 0..col_count {
                    let bin_x = bin_x_offset + cur.read_i16::<LittleEndian>()? as i32;
                    let count = if use_short {
                        cur.read_i16::<LittleEndian>()? as f64
                    } else {
                        cur.read_f32::<LittleEndian>()? as f64
                    };
                    out.push((bin_x, bin_y, count));
                }
            }
        }
        2 => {
            // Dense block, `w` wide, stored row-major.
            let n_pts = cur.read_i32::<LittleEndian>()? as usize;
            let w = cur.read_i16::<LittleEndian>()? as usize;
            for i in 0..n_pts {
                let row = i / w;
                let col = i - row * w;
                let bin_x = bin_x_offset + col as i32;
                let bin_y = bin_y_offset + row as i32;
                let count = if use_short {
                    let c = cur.read_i16::<LittleEndian>()?;
                    if c == -32768 {
                        continue; // sentinel for an empty dense cell
                    }
                    c as f64
                } else {
                    let c = cur.read_f32::<LittleEndian>()?;
                    if c.is_nan() {
                        continue;
                    }
                    c as f64
                };
                out.push((bin_x, bin_y, count));
            }
        }
        other => {
            return Err(Error::Format(format!(
                "unknown block representation {other} (expected 1 or 2)"
            )));
        }
    }

    Ok(out)
}

/// Read a null-terminated UTF-8 string (byteorder has no such helper).
fn read_cstring<R: Read>(reader: &mut R) -> Result<String> {
    let mut bytes = Vec::new();
    loop {
        let mut b = [0u8; 1];
        reader.read_exact(&mut b)?;
        if b[0] == 0 {
            break;
        }
        bytes.push(b[0]);
    }
    String::from_utf8(bytes).map_err(|e| Error::Format(format!("invalid UTF-8 in string: {e}")))
}

/// Write a null-terminated string (byteorder has no such helper).
fn write_cstring<W: Write>(w: &mut W, s: &str) -> Result<()> {
    w.write_all(s.as_bytes())?;
    w.write_all(&[0])?;
    Ok(())
}

/// Compress a buffer with zlib (deflate), as the `.hic` format expects.
fn zlib_compress(data: &[u8]) -> Result<Vec<u8>> {
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
    enc.write_all(data)?;
    enc.finish()
        .map_err(|e| Error::Format(format!("zlib compression failed: {e}")))
}

/// Default number of bins per block (`blockSize`).
const DEFAULT_BLOCK_SIZE: i64 = 500;
/// Target number of bins for the genome-wide `All` matrix.
const ALL_TARGET_BINS: u64 = 500;
/// Scale factor applied to the `All` chromosome length and bin size, keeping
/// both small enough to store in a v8 `i32` (matching juicer/hictk).
const ALL_SCALE_FACTOR: i64 = 1000;

/// Writer for `.hic` (format v8) files.
///
/// Accepts pixels per resolution (like [`CoolerWriter::write_pixels`]) and
/// writes a valid v8 file including the genome-wide `All` pseudo-chromosome.
/// Pixels are buffered in memory and written by [`HicWriter::finalize`].
pub struct HicWriter {
    file: File,
    genome_id: String,
    /// Real (non-`All`) chromosomes, in header order.
    chroms: Vec<Chrom>,
    /// Bin sizes, sorted coarse-to-fine.
    resolutions: Vec<u32>,
    /// Buffered pixels per resolution (symmetric-upper, global bin ids).
    buffers: BTreeMap<u32, Vec<Pixel>>,
    /// Header attributes (key/value string pairs).
    attributes: BTreeMap<String, String>,
    /// Normalization vectors to write into the footer.
    norms: Vec<NormVec>,
}

/// One normalization vector to write into the footer.
struct NormVec {
    resolution: u32,
    name: String,
    chrom: String,
    values: Vec<f64>,
}

/// Chromosome-relative `(bin_x, bin_y, count)` records.
type PairPixels = Vec<(i64, i64, f64)>;
/// Buffered pixels classified by resolution, then by `(c1, c2)` pair.
type Classified = BTreeMap<u32, BTreeMap<(usize, usize), PairPixels>>;

/// One resolution's worth of pixel data for a single chromosome pair.
struct ResSpec {
    bin_size: i64,
    /// Number of bins over the first chromosome (drives `blockColumnCount`).
    n_bins1: i64,
    /// Chromosome-relative `(bin_x, bin_y, count)` records.
    pixels: Vec<(i64, i64, f64)>,
}

/// Computed metadata for one resolution of a chromosome pair.
struct ResRecord {
    bin_size: i64,
    block_cols: i64,
    block_count: i32,
    sum_counts: f64,
    nnz: i64,
    metas: Vec<(i32, i64, i32)>,
}

impl HicWriter {
    /// Create a writer for `path`. `chroms` are the real (non-`All`)
    /// chromosomes; the `All` pseudo-chromosome is added automatically.
    pub fn create<P: AsRef<Path>>(
        path: P,
        genome_id: &str,
        chroms: &[Chrom],
        resolutions: &[u32],
    ) -> Result<Self> {
        if chroms.is_empty() {
            return Err(Error::InvalidInput("no chromosomes".to_string()));
        }
        let mut resolutions = resolutions.to_vec();
        resolutions.sort_unstable_by(|a, b| b.cmp(a));
        resolutions.dedup();
        if resolutions.is_empty() {
            return Err(Error::InvalidInput("no resolutions".to_string()));
        }
        let file = File::create(path)?;
        Ok(HicWriter {
            file,
            genome_id: genome_id.to_string(),
            chroms: chroms.to_vec(),
            resolutions,
            buffers: BTreeMap::new(),
            attributes: BTreeMap::new(),
            norms: Vec::new(),
        })
    }

    /// Set the header attribute dictionary (written at [`HicWriter::finalize`]).
    pub fn set_attributes(&mut self, attrs: &BTreeMap<String, String>) {
        self.attributes = attrs.clone();
    }

    /// Buffer divisive normalization vectors for one resolution. `vectors` maps
    /// each real chromosome name to its per-bin weights; the header index is
    /// derived from `chroms` order (the `All` pseudo-chromosome is index 0).
    pub fn add_normalization_vectors(
        &mut self,
        resolution: u32,
        name: &str,
        vectors: &[(String, Vec<f64>)],
    ) -> Result<()> {
        if !self.resolutions.contains(&resolution) {
            return Err(Error::InvalidInput(format!(
                "resolution {resolution} was not declared at creation"
            )));
        }
        for (chrom, values) in vectors {
            if !self.chroms.iter().any(|c| c.name == *chrom) {
                return Err(Error::InvalidInput(format!("unknown chromosome '{chrom}'")));
            }
            self.norms.push(NormVec {
                resolution,
                name: name.to_string(),
                chrom: chrom.clone(),
                values: values.clone(),
            });
        }
        Ok(())
    }

    /// Buffer pixels for one resolution.
    pub fn add_pixels(&mut self, resolution: u32, pixels: &[Pixel]) -> Result<()> {
        if !self.resolutions.contains(&resolution) {
            return Err(Error::InvalidInput(format!(
                "resolution {resolution} was not declared at creation"
            )));
        }
        self.buffers
            .entry(resolution)
            .or_default()
            .extend_from_slice(pixels);
        Ok(())
    }

    /// Write the file, consuming the writer.
    pub fn finalize(mut self) -> Result<()> {
        let n_real = self.chroms.len();
        let genome_size: i64 = self.chroms.iter().map(|c| c.length as i64).sum();
        let mut classified = self.classify_all()?;

        // `All` matrix parameters, derived from the finest resolution.
        let finest = *self.resolutions.last().unwrap() as u64;
        let factor =
            ((genome_size as f64 / ALL_TARGET_BINS as f64 / finest as f64).ceil() as u64).max(1);
        let bin_size = factor * finest;
        let mut offsets_f = vec![0i64; n_real + 1];
        for (i, c) in self.chroms.iter().enumerate() {
            offsets_f[i + 1] = offsets_f[i] + (c.length as u64).div_ceil(finest) as i64;
        }
        let total_bins = offsets_f[n_real];
        let all_length = total_bins * finest as i64 / ALL_SCALE_FACTOR;
        let bin_size_scaled = (bin_size as i64 / ALL_SCALE_FACTOR).max(1);

        // Coarsen the finest-resolution pixels into the genome-wide matrix.
        let mut all_map: BTreeMap<(i64, i64), f64> = BTreeMap::new();
        if let Some(pairs) = classified.get(&(finest as u32)) {
            for (&(c1, c2), px) in pairs {
                let (o1, o2) = (offsets_f[c1], offsets_f[c2]);
                for &(bx, by, count) in px {
                    let g1 = (o1 + bx) / factor as i64;
                    let g2 = (o2 + by) / factor as i64;
                    *all_map.entry((g1, g2)).or_insert(0.0) += count;
                }
            }
        }
        let all_pixels: Vec<(i64, i64, f64)> =
            all_map.into_iter().map(|((a, b), c)| (a, b, c)).collect();
        let all_n_bins = (total_bins as u64).div_ceil(factor) as i64;

        // Header. `All` is header chromosome 0; real chrom i is header chrom i+1.
        let mut header = Vec::new();
        header.write_all(b"HIC\0")?;
        header.write_i32::<LittleEndian>(8)?; // version
        header.write_i64::<LittleEndian>(0)?; // master index position (patched below)
        write_cstring(&mut header, &self.genome_id)?;
        header.write_i32::<LittleEndian>(self.attributes.len() as i32)?; // nAttributes
        for (k, v) in &self.attributes {
            write_cstring(&mut header, k)?;
            write_cstring(&mut header, v)?;
        }
        header.write_i32::<LittleEndian>((n_real + 1) as i32)?; // nChrs (incl All)
        write_cstring(&mut header, "All")?;
        header.write_i32::<LittleEndian>(all_length as i32)?;
        for c in &self.chroms {
            write_cstring(&mut header, &c.name)?;
            header.write_i32::<LittleEndian>(c.length)?;
        }
        header.write_i32::<LittleEndian>(self.resolutions.len() as i32)?;
        for &r in &self.resolutions {
            header.write_i32::<LittleEndian>(r as i32)?;
        }
        header.write_i32::<LittleEndian>(0)?; // nFragRes
        self.file.write_all(&header)?;

        // Matrices: each real pair, then All:All.
        let mut footers: Vec<(String, i64, i32)> = Vec::new();
        for c1 in 0..n_real {
            for c2 in c1..n_real {
                let mut specs = Vec::with_capacity(self.resolutions.len());
                for &res in &self.resolutions {
                    let n_bins1 = (self.chroms[c1].length as u64).div_ceil(res as u64) as i64;
                    let pixels = classified
                        .get_mut(&res)
                        .and_then(|m| m.remove(&(c1, c2)))
                        .unwrap_or_default();
                    specs.push(ResSpec {
                        bin_size: res as i64,
                        n_bins1,
                        pixels,
                    });
                }
                let (pos, size) =
                    self.write_matrix_body((c1 + 1) as i32, (c2 + 1) as i32, &specs)?;
                footers.push((format!("{}_{}", c1 + 1, c2 + 1), pos, size));
            }
        }
        let all_spec = ResSpec {
            bin_size: bin_size_scaled,
            n_bins1: all_n_bins,
            pixels: all_pixels,
        };
        let (pos, size) = self.write_matrix_body(0, 0, std::slice::from_ref(&all_spec))?;
        footers.push(("0_0".to_string(), pos, size));

        // Footer (master index + expected values + normalization vectors).
        let master_pos = self.file.stream_position()?;

        // Serialize normalization vector bodies; record their index entries.
        let mut bodies: Vec<Vec<u8>> = Vec::with_capacity(self.norms.len());
        let mut norm_entries: Vec<(i32, u32, String, i32)> = Vec::new(); // (chrIdx, res, name, size)
        for n in &self.norms {
            let mut b = Vec::new();
            b.write_i32::<LittleEndian>(n.values.len() as i32)?;
            for &v in &n.values {
                b.write_f64::<LittleEndian>(v)?;
            }
            let chr_idx = self
                .chroms
                .iter()
                .position(|c| c.name == n.chrom)
                .map(|i| (i + 1) as i32)
                .unwrap_or(-1);
            norm_entries.push((chr_idx, n.resolution, n.name.clone(), b.len() as i32));
            bodies.push(b);
        }

        // Byte layout of the footer, so each vector body gets its file offset.
        let mut master_index_size = 4i64; // nEntries
        for (k, _, _) in &footers {
            master_index_size += k.len() as i64 + 1 + 8 + 4;
        }
        let mut norm_index_size = 4i64; // nNormEntries
        for (_, _, name, _) in &norm_entries {
            // name(cstring) + chrIdx(4) + "BP"(cstring 3) + res(4) + pos(8) + size(4)
            norm_index_size += name.len() as i64 + 1 + 4 + 3 + 4 + 8 + 4;
        }
        let body_start_offset = 4 + master_index_size + 8 + norm_index_size;
        let mut body_offset = body_start_offset;
        let mut positions = Vec::with_capacity(norm_entries.len());
        for b in &bodies {
            positions.push(master_pos as i64 + body_offset);
            body_offset += b.len() as i64;
        }
        let total_footer = body_offset as i32;

        let mut footer = Vec::new();
        footer.write_i32::<LittleEndian>(total_footer)?; // nBytes
        footer.write_i32::<LittleEndian>(footers.len() as i32)?;
        for (k, pos, size) in &footers {
            write_cstring(&mut footer, k)?;
            footer.write_i64::<LittleEndian>(*pos)?;
            footer.write_i32::<LittleEndian>(*size)?;
        }
        footer.write_i32::<LittleEndian>(0)?; // nExpectedValues
        footer.write_i32::<LittleEndian>(0)?; // nExpectedValues (normalized)
        footer.write_i32::<LittleEndian>(norm_entries.len() as i32)?;
        for (i, (chr_idx, res, name, size)) in norm_entries.iter().enumerate() {
            write_cstring(&mut footer, name)?;
            footer.write_i32::<LittleEndian>(*chr_idx)?;
            write_cstring(&mut footer, "BP")?;
            footer.write_i32::<LittleEndian>(*res as i32)?;
            footer.write_i64::<LittleEndian>(positions[i])?;
            footer.write_i32::<LittleEndian>(*size)?;
        }
        self.file.write_all(&footer)?;
        for b in &bodies {
            self.file.write_all(b)?;
        }

        // Patch the master index position in the header.
        self.file.seek(SeekFrom::Start(8))?;
        self.file.write_i64::<LittleEndian>(master_pos as i64)?;
        self.file.flush()?;
        Ok(())
    }

    /// Classify buffered pixels per resolution into `(c1, c2)` pairs with
    /// chromosome-relative bin coordinates.
    fn classify_all(&self) -> Result<Classified> {
        let n_real = self.chroms.len();
        let mut out = BTreeMap::new();
        for &res in &self.resolutions {
            let mut offsets = vec![0i64; n_real + 1];
            for (i, c) in self.chroms.iter().enumerate() {
                offsets[i + 1] = offsets[i] + (c.length as u64).div_ceil(res as u64) as i64;
            }
            let total = offsets[n_real];
            let mut pairs: BTreeMap<(usize, usize), PairPixels> = BTreeMap::new();
            if let Some(px) = self.buffers.get(&res) {
                for p in px {
                    if p.bin1_id < 0 || p.bin2_id < 0 || p.bin1_id > p.bin2_id || p.bin2_id >= total
                    {
                        return Err(Error::InvalidInput(format!(
                            "pixel ({}, {}) out of range or not symmetric-upper",
                            p.bin1_id, p.bin2_id
                        )));
                    }
                    let c1 = chrom_of(&offsets, p.bin1_id)?;
                    let c2 = chrom_of(&offsets, p.bin2_id)?;
                    let bx = p.bin1_id - offsets[c1];
                    let by = p.bin2_id - offsets[c2];
                    pairs.entry((c1, c2)).or_default().push((bx, by, p.count));
                }
            }
            out.insert(res, pairs);
        }
        Ok(out)
    }

    /// Write one chromosome pair's matrix record (all resolutions) and its
    /// compressed blocks, returning the record's `(position, size)` for the footer.
    fn write_matrix_body(&mut self, chr1: i32, chr2: i32, specs: &[ResSpec]) -> Result<(i64, i32)> {
        let mut res_records = Vec::with_capacity(specs.len());
        for spec in specs {
            let block_cols = (spec.n_bins1 as u64).div_ceil(DEFAULT_BLOCK_SIZE as u64) as i64;
            let mut blocks: BTreeMap<i32, Vec<(i64, i64, f64)>> = BTreeMap::new();
            for &(bin_x, bin_y, count) in &spec.pixels {
                let block_col = bin_x / DEFAULT_BLOCK_SIZE;
                let block_row = bin_y / DEFAULT_BLOCK_SIZE;
                let number = (block_row * block_cols + block_col) as i32;
                blocks
                    .entry(number)
                    .or_default()
                    .push((bin_x, bin_y, count));
            }
            let mut sum_counts = 0.0f64;
            let mut nnz = 0i64;
            let mut metas = Vec::with_capacity(blocks.len());
            for (number, mut px) in blocks {
                px.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
                let bin_x_off = (number as i64 % block_cols) * DEFAULT_BLOCK_SIZE;
                let bin_y_off = (number as i64 / block_cols) * DEFAULT_BLOCK_SIZE;
                let (data, _) = serialize_block(&px, bin_x_off, bin_y_off)?;
                let compressed = zlib_compress(&data)?;
                let pos = self.file.stream_position()? as i64;
                self.file.write_all(&compressed)?;
                sum_counts += px.iter().map(|p| p.2).sum::<f64>();
                nnz += px.len() as i64;
                metas.push((number, pos, compressed.len() as i32));
            }
            res_records.push(ResRecord {
                bin_size: spec.bin_size,
                block_cols,
                block_count: metas.len() as i32,
                sum_counts,
                nnz,
                metas,
            });
        }

        // Matrix record: metadata + block index.
        let rec_pos = self.file.stream_position()? as i64;
        let mut buf = Vec::new();
        buf.write_i32::<LittleEndian>(chr1)?;
        buf.write_i32::<LittleEndian>(chr2)?;
        buf.write_i32::<LittleEndian>(res_records.len() as i32)?;
        for rr in &res_records {
            write_cstring(&mut buf, "BP")?;
            buf.write_i32::<LittleEndian>(0)?; // resIdx
            buf.write_f32::<LittleEndian>(rr.sum_counts as f32)?;
            buf.write_f32::<LittleEndian>(rr.nnz as f32)?; // occupiedCellCount
            buf.write_f32::<LittleEndian>(0.0)?; // stdDev
            buf.write_f32::<LittleEndian>(0.0)?; // percent95
            buf.write_i32::<LittleEndian>(rr.bin_size as i32)?;
            buf.write_i32::<LittleEndian>(DEFAULT_BLOCK_SIZE as i32)?;
            buf.write_i32::<LittleEndian>(rr.block_cols as i32)?;
            buf.write_i32::<LittleEndian>(rr.block_count)?;
            for (number, pos, size) in &rr.metas {
                buf.write_i32::<LittleEndian>(*number)?;
                buf.write_i64::<LittleEndian>(*pos)?;
                buf.write_i32::<LittleEndian>(*size)?;
            }
        }
        self.file.write_all(&buf)?;
        Ok((rec_pos, buf.len() as i32))
    }
}

/// Encode one sparse block (representation 1) into its uncompressed byte form.
fn serialize_block(
    pixels: &[(i64, i64, f64)],
    bin_x_off: i64,
    bin_y_off: i64,
) -> Result<(Vec<u8>, bool)> {
    let use_short = pixels
        .iter()
        .all(|&(_, _, c)| c.fract() == 0.0 && (0.0..=32767.0).contains(&c));
    let mut buf = Vec::new();
    buf.write_i32::<LittleEndian>(pixels.len() as i32)?; // nRecords
    buf.write_i32::<LittleEndian>(bin_x_off as i32)?;
    buf.write_i32::<LittleEndian>(bin_y_off as i32)?;
    buf.write_u8(if use_short { 0 } else { 1 })?;
    buf.write_u8(1)?; // list of rows

    // Group pixels (sorted by bin_y then bin_x) into rows.
    let mut rows: Vec<(i64, Vec<(i64, f64)>)> = Vec::new();
    for &(bx, by, count) in pixels {
        match rows.last_mut() {
            Some((y, cols)) if *y == by => cols.push((bx, count)),
            _ => rows.push((by, vec![(bx, count)])),
        }
    }
    buf.write_i16::<LittleEndian>(rows.len() as i16)?;
    for (by, cols) in &rows {
        buf.write_i16::<LittleEndian>((by - bin_y_off) as i16)?;
        buf.write_i16::<LittleEndian>(cols.len() as i16)?;
        for (bx, count) in cols {
            buf.write_i16::<LittleEndian>((bx - bin_x_off) as i16)?;
            if use_short {
                buf.write_i16::<LittleEndian>(*count as i16)?;
            } else {
                buf.write_f32::<LittleEndian>(*count as f32)?;
            }
        }
    }
    Ok((buf, use_short))
}

/// Index of the chromosome whose bin range contains `bin`.
fn chrom_of(offsets: &[i64], bin: i64) -> Result<usize> {
    let idx = offsets.partition_point(|&o| o <= bin);
    if idx == 0 || idx >= offsets.len() {
        return Err(Error::InvalidInput(format!(
            "bin id {bin} out of chromosome range"
        )));
    }
    Ok(idx - 1)
}
