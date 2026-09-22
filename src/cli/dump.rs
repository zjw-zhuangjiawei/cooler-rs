//! `dump` — write a `.hic`/`.cool`/`.mcool` out as text.
//!
//! `dump pixels` is a port of `hictk dump`'s interaction table, sized so the
//! two outputs can be `diff`ed. hictk's other tables (`chroms`, `bins`,
//! `normalizations`, `resolutions`, `weights`) are not implemented: the table
//! selector and its five renderers were dropped, so `--resolution` /
//! `-r` / `-b` / `--join` are all that is left of that surface. Adding one
//! back means a `Table`-like flag again, or a subcommand next to `pixels`.
//!
//! The formatting rules below are copied from
//! `hictk/src/hictk/dump/common.cpp`; the two that are easy to get wrong:
//!
//! - pixel counts print with printf `%.16g` — 16 significant digits, `%g`
//!   presentation ([`g16`]);
//! - `--balance` scales through **f32**: `pixel.count /= (float)(w1 * w2)`
//!   with `count` an `f32` (`hictk/src/libhictk/hic/include/hictk/hic/impl/
//!   pixel_selector_impl.hpp`, both `transform_pixel` overloads). A plain f64
//!   divide agrees to ~15 digits but not to the 16th, so the last printed
//!   digit differs ([`scale_divisive_f32`]).
//!
//! `dump matrix` is the one subcommand that is not hictk's: it writes the
//! whole dense N×N square for one chromosome in the original OnTAD `.mat` text
//! format. That format — both directions — lives here rather than in
//! `convert`, which is for Hi-C containers only; `load` reads it back.
//!
//! Pixels are always emitted in the ascending order our readers return them
//! in, which is what `hictk dump` does by default.

use std::io::{self, BufWriter, Write};

use clap::{Args, Subcommand};

use cooler_rs::cooler::Cooler;
use cooler_rs::error::{Error, Result};
use cooler_rs::file::File;
use cooler_rs::hic::HiCFile;
use cooler_rs::mcool::Mcool;
use cooler_rs::region::Region;
use cooler_rs::types::{Bin, Chrom, Pixel, WeightType};

/// Fields every `dump` mode shares. Split out so the `matrix` subcommand —
/// which is ours, not hictk's — can carry the same ones without `--join`.
#[derive(Args)]
struct CommonArgs {
    /// Path to a .hic, .cool or .mcool file
    #[arg(value_name = "URI")]
    uri: String,

    /// HiC matrix resolution (required for .hic/.mcool with >1 resolution)
    #[arg(long, value_name = "BP")]
    resolution: Option<u32>,

    /// UCSC-style coordinates of the region to dump (`chr1:0-1000`)
    #[arg(short = 'r', long, default_value = "all")]
    range: String,

    /// Balance interactions using the given method (`.hic` norm vector name,
    /// or a `bins` column for `.cool`/`.mcool`)
    #[arg(short = 'b', long, default_value = "NONE")]
    balance: String,
}

/// One subcommand per mode, both required: an optional subcommand sitting
/// alongside the positional `uri` is what a flat `dump F` would need, and clap
/// 4.6.4 does not parse that however the `subcommand_*` settings are combined
/// (`dump matrix F` always reports `uri` as missing).
#[derive(Args)]
pub struct DumpArgs {
    #[command(subcommand)]
    command: DumpCommand,
}

#[derive(Subcommand)]
enum DumpCommand {
    /// Interaction table: `bin1_id<TAB>bin2_id<TAB>count`, one row per pixel
    Pixels(PixelsArgs),
    /// Dense N×N matrix over one chromosome, whitespace-separated — the
    /// original OnTAD `.mat` text format, and the inverse of `load`.
    /// Requires `-r <chrom>`
    Matrix(MatrixArgs),
}

#[derive(Args)]
struct PixelsArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Output pixels in BG2 format (chrom/start/end pairs instead of bin ids)
    #[arg(long)]
    join: bool,
}

#[derive(Args)]
struct MatrixArgs {
    #[command(flatten)]
    common: CommonArgs,
}

pub fn run(args: DumpArgs) -> Result<()> {
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    match &args.command {
        DumpCommand::Pixels(p) => dump_pixels(p, &mut out),
        DumpCommand::Matrix(m) => dump_matrix(&m.common, &mut out),
    }
}

/// printf `%.16g`: 16 significant digits, `%g` presentation (scientific when
/// the decimal exponent is `< -4` or `>= 16`, plain otherwise, trailing zeros
/// stripped). `fmt` spells NaN/Infinity in lower case, so we do too.
fn g16(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0" } else { "0" }.to_string();
    }

    const P: i32 = 16;
    // Round to P significant digits first: this also carries the exponent
    // (9.99..9e15 -> 1e16), which the presentation choice depends on.
    let sci = format!("{:.*e}", (P - 1) as usize, x);
    let (mantissa, exp) = sci.split_once('e').expect("`{:e}` always has an exponent");
    let exp: i32 = exp.parse().expect("`{:e}` exponent is an integer");

    // `%g` switches to scientific outside this exponent range.
    if !(-4..P).contains(&exp) {
        let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
        format!(
            "{mantissa}e{}{:02}",
            if exp < 0 { '-' } else { '+' },
            exp.abs()
        )
    } else {
        let s = format!("{:.*}", (P - 1 - exp).max(0) as usize, x);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// hictk divides a `.hic` count by the f64 product of its two weights, cast to
/// `f32`, with the division itself in `f32`. Reproducing that exactly is what
/// makes the printed `%.16g` match.
fn scale_divisive_f32(count: f64, w1: f64, w2: f64) -> f64 {
    ((count as f32) / ((w1 * w2) as f32)) as f64
}

enum Input {
    Hic(HiCFile),
    Mcool(Mcool),
    Cool(Cooler),
}

fn open_input(args: &CommonArgs) -> Result<Input> {
    if let Ok(hic) = HiCFile::open(&args.uri) {
        return Ok(Input::Hic(hic));
    }
    if let Ok(mcool) = Mcool::open(&args.uri) {
        return Ok(Input::Mcool(mcool));
    }
    Ok(Input::Cool(Cooler::open_any(&args.uri)?))
}

fn resolutions_of(input: &Input) -> Result<Vec<u32>> {
    Ok(match input {
        Input::Hic(h) => h.resolutions().to_vec(),
        Input::Mcool(m) => m.resolutions()?.into_iter().map(|r| r as u32).collect(),
        Input::Cool(c) => c.bin_size()?.map(|r| vec![r as u32]).unwrap_or_default(),
    })
}

/// `--resolution` is mandatory for a multi-resolution `.hic`/`.mcool` (hictk
/// enforces the same, `cli_dump.cpp`:160); a single-resolution file or a plain
/// `.cool` supplies its own.
fn resolve_resolution(args: &CommonArgs, input: &Input) -> Result<u32> {
    if let Some(r) = args.resolution {
        return Ok(r);
    }
    let res = resolutions_of(input)?;
    match res.as_slice() {
        [only] => Ok(*only),
        _ => Err(Error::InvalidInput(
            "--resolution is mandatory when the file is in .hic or .mcool format".to_string(),
        )),
    }
}

/// `all` (the default) means "the whole file"; anything else is a UCSC region
/// on one chromosome, which is also what `--range2` defaults to in hictk.
fn parse_range(range: &str) -> Result<Option<Region>> {
    if range == "all" {
        return Ok(None);
    }
    Ok(Some(Region::parse(range)?))
}

/// Global bin ids covered by `region`, matching hictk's `find_overlap`: bins
/// that *overlap* the interval, not bins fully inside it.
fn bin_ids_for(
    chrom_offset: &[i64],
    chroms: &[Chrom],
    res: u32,
    region: &Region,
) -> Result<(i64, i64)> {
    let chrom_id = chroms
        .iter()
        .position(|c| c.name == region.chrom)
        .ok_or_else(|| Error::InvalidInput(format!("unknown sequence label: {}", region.chrom)))?;
    let len = chroms[chrom_id].length as u64;
    let start = region.start.unwrap_or(0);
    let end = region.end.unwrap_or(len);
    if end < start || end > len {
        return Err(Error::InvalidInput(format!(
            "region out of bounds on '{}' (length {len}): [{start}, {end})",
            region.chrom
        )));
    }
    let res = res.max(1) as u64;
    let g0 = chrom_offset[chrom_id];
    let g1 = chrom_offset[chrom_id + 1];
    let i0 = g0 + (start / res) as i64;
    let i1 = if end == start {
        i0
    } else {
        g0 + end.div_ceil(res) as i64
    };
    Ok((i0, i1.min(g1)))
}

// ---------------------------------------------------------------- tables ---

/// The chromosomes of `input`, seen at `resolution`.
fn input_chroms(input: &Input, resolution: u32) -> Result<Vec<Chrom>> {
    Ok(match input {
        Input::Hic(h) => h.chromosomes(),
        Input::Cool(c) => c.chroms()?,
        Input::Mcool(m) => m.cooler(resolution as u64)?.chroms()?,
    })
}

/// The uniform `div_ceil(length, resolution)` bins of `chroms`, plus their
/// per-chromosome offsets.
///
/// Derived here rather than read through `File`, whose `.hic` variant
/// materializes *and sorts* every pixel of the resolution on first access
/// (`file.rs`, `build_hic`). At 250 bp that is ~10 GB before a single row is
/// printed, so `dump` used to OOM on any fine-resolution input.
fn tiled_bins(chroms: &[Chrom], resolution: u32) -> (Vec<Bin>, Vec<i64>) {
    let mut bins = Vec::new();
    let mut chrom_offset = vec![0i64];
    for (k, c) in chroms.iter().enumerate() {
        let n = (c.length as u64).div_ceil(resolution as u64);
        for i in 0..n {
            bins.push(Bin {
                chrom_id: k as i32,
                start: (i * resolution as u64) as i32,
                end: (((i + 1) * resolution as u64).min(c.length as u64)) as i32,
            });
        }
        chrom_offset.push(bins.len() as i64);
    }
    (bins, chrom_offset)
}

/// Append `resolution`'s pixels that fall inside the cis block `[i0, i1)`.
/// Streams: only the retained region is held.
fn collect_cis(
    input: &Input,
    resolution: u32,
    i0: i64,
    i1: i64,
    sink: &mut Vec<Pixel>,
) -> Result<()> {
    let keep = |p: &Pixel| p.bin1_id >= i0 && p.bin1_id < i1 && p.bin2_id >= i0 && p.bin2_id < i1;
    let mut push = |p: Pixel| {
        if keep(&p) {
            sink.push(p);
        }
    };
    match input {
        Input::Hic(h) => h.for_each_pixel(resolution, |p| {
            push(p);
            Ok(())
        })?,
        // ponytail: 1M-pixel read chunks; bounds the read buffer only, the
        // retained set is still whatever `--range` selects.
        Input::Cool(c) => {
            for chunk in c.pixels_chunked(1 << 20)? {
                for p in chunk? {
                    push(p);
                }
            }
        }
        Input::Mcool(m) => {
            for chunk in m.cooler(resolution as u64)?.pixels_chunked(1 << 20)? {
                for p in chunk? {
                    push(p);
                }
            }
        }
    }
    Ok(())
}

/// Normalization vectors for every chromosome, concatenated into one global
/// per-bin vector. `.hic` norm vectors are stored per chromosome.
fn hic_global_weights(hic: &HiCFile, res: u32, name: &str) -> Result<Vec<f64>> {
    let mut out = Vec::new();
    for c in hic.chromosomes() {
        if c.name == "ALL" {
            continue;
        }
        let w = hic.norm_vector(res, &c.name, name)?.ok_or_else(|| {
            Error::InvalidInput(format!("unable to find {name} normalization vector"))
        })?;
        out.extend(w);
    }
    Ok(out)
}

fn file_weights(file: &File, name: &str, n_bins: usize) -> Result<Vec<f64>> {
    let w = match file {
        File::Cooler(c) => c.bins_column_f64(name)?,
        File::Hic(_) => None,
    };
    w.filter(|w| w.len() == n_bins)
        .ok_or_else(|| Error::InvalidInput(format!("no '{name}' bins column")))
}

/// Resolve a weight column's convention on an opened [`File`].
///
/// `.hic` normalization vectors are always divisive. A cooler column is
/// resolved attribute-first, then by name; a column neither decides defaults
/// to multiplicative, which is what every column did before the type was
/// resolved at all — so no existing caller changes behavior.
fn file_weight_type(file: &File, name: &str) -> WeightType {
    match file {
        File::Cooler(c) => c
            .bins_column_weight_type(name)
            .ok()
            .flatten()
            .unwrap_or(WeightType::Multiplicative),
        File::Hic(_) => WeightType::Divisive,
    }
}

fn dump_pixels(args: &PixelsArgs, out: &mut impl Write) -> Result<()> {
    let input = open_input(&args.common)?;
    let res = resolve_resolution(&args.common, &input)?;
    let chroms = input_chroms(&input, res)?;
    let (bins, chrom_offset) = tiled_bins(&chroms, res);

    // A region is a cis block: both ends must fall inside it. `File::fetch`
    // is a *row* query (bin2 spans the whole chromosome), so filter here.
    let (i0, i1) = match parse_range(&args.common.range)? {
        None => (0i64, bins.len() as i64),
        Some(r) => bin_ids_for(&chrom_offset, &chroms, res, &r)?,
    };

    // Read streamed (`File::pixels` used to materialize and sort the whole
    // resolution). The sort is still needed — `.hic` stores blocks in
    // (bin2-block, bin1-block) order, not the ascending order
    // `DumpConfig::sorted{true}` makes hictk's default — so this buffers the
    // *selected* pixels.
    // ponytail: a whole-file dump of a 250 bp matrix therefore still holds
    // ~10 GB (429M pixels). `--range` bounds it, and a region is the sane way
    // to inspect a fine resolution. Upgrade path: k-way merge over block
    // columns, as hictk's sorted pixel selector does, to stream in order.
    let mut pixels = Vec::new();
    collect_cis(&input, res, i0, i1, &mut pixels)?;
    pixels.sort_unstable_by_key(|p| (p.bin1_id, p.bin2_id));
    apply_balance(&mut pixels, &args.common, &input, res, &bins)?;

    for p in &pixels {
        if args.join {
            let b1 = &bins[p.bin1_id as usize];
            let b2 = &bins[p.bin2_id as usize];
            writeln!(
                out,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                chroms[b1.chrom_id as usize].name,
                b1.start,
                b1.end,
                chroms[b2.chrom_id as usize].name,
                b2.start,
                b2.end,
                g16(p.count)
            )?;
        } else {
            writeln!(out, "{}\t{}\t{}", p.bin1_id, p.bin2_id, g16(p.count))?;
        }
    }
    Ok(())
}

/// Apply `--balance` to `pixels` in place; a no-op for the default `NONE`.
///
/// The *precision* depends on the format — `pixel_selector_impl.hpp:141`
/// divides an f32 by `(float)(w1 * w2)` for `.hic`, while
/// `weights_impl.hpp:182` divides an f64 by the f64 product — but the
/// *operation* depends on the column's resolved type, not the container: a
/// cooler `KR` column is divisive and its `weight` column is not.
fn apply_balance(
    pixels: &mut [Pixel],
    args: &CommonArgs,
    input: &Input,
    res: u32,
    bins: &[Bin],
) -> Result<()> {
    if args.balance == "NONE" {
        return Ok(());
    }
    let is_hic = matches!(input, Input::Hic(_));
    let (w, wtype) = match input {
        Input::Hic(h) => (
            hic_global_weights(h, res, &args.balance)?,
            WeightType::Divisive,
        ),
        _ => {
            let file = File::open(&args.uri, res)?;
            let w = file_weights(&file, &args.balance, bins.len())?;
            let t = file_weight_type(&file, &args.balance);
            (w, t)
        }
    };
    for p in pixels.iter_mut() {
        let (a, b) = (p.bin1_id as usize, p.bin2_id as usize);
        p.count = if is_hic {
            scale_divisive_f32(p.count, w[a], w[b])
        } else if wtype.is_divisive() {
            p.count / (w[a] * w[b])
        } else {
            p.count * (w[a] * w[b])
        };
    }
    Ok(())
}

/// Parse a dense N×N whitespace-separated text matrix into sparse pixels.
///
/// Only upper-triangle non-zero entries are kept (symmetric-upper sparse
/// storage); zeros are implicit and omitted. Returns the matrix dimension
/// `n` together with the pixels, in row-major order.
///
/// This is the original OnTAD `.mat` text format, the inverse of
/// [`dump_matrix`]. `load` is what calls it.
pub(crate) fn dense_txt_to_pixels(text: &str) -> Result<(usize, Vec<Pixel>)> {
    let mut pixels: Vec<Pixel> = Vec::new();
    let mut width: Option<usize> = None;

    // Blank lines are skipped; only non-blank lines count as matrix rows.
    let mut i = 0;
    for (lineno, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut cols = 0;
        for (j, field) in line.split_whitespace().enumerate() {
            let v: f64 = field.parse().map_err(|_| {
                Error::InvalidInput(format!(
                    "line {}, column {}: '{field}' is not a number",
                    lineno + 1,
                    j + 1
                ))
            })?;
            if j >= i && v > 0.0 {
                pixels.push(Pixel {
                    bin1_id: i as i64,
                    bin2_id: j as i64,
                    count: v,
                });
            }
            cols += 1;
        }
        match width {
            None => width = Some(cols),
            Some(w) if w != cols => {
                return Err(Error::InvalidInput(format!(
                    "input is not a square N×N matrix: line {} has {cols} columns, expected {w}",
                    lineno + 1
                )));
            }
            _ => {}
        }
        i += 1;
    }

    let n = width.unwrap_or(0);
    if n == 0 {
        return Err(Error::InvalidInput("input is empty".into()));
    }
    Ok((n, pixels))
}

/// Dense N×N matrix over one chromosome, one whitespace-separated row per
/// line — the original OnTAD `.mat` text format, and the inverse of `load`.
///
/// Only the upper triangle is stored, so the output is mirrored to fill the
/// whole square: the format OnTAD reads is symmetric, not triangular.
///
/// ponytail: the square is held in RAM (8·N² bytes; 19 MB for a 1534-bin
/// chromosome, GBs at 1 kb on a large chromosome). Streaming it would mean
/// re-reading the pixels once per row. `-r` bounds it to a region.
fn dump_matrix(args: &CommonArgs, out: &mut impl Write) -> Result<()> {
    let input = open_input(args)?;
    let res = resolve_resolution(args, &input)?;
    let chroms = input_chroms(&input, res)?;
    let (bins, chrom_offset) = tiled_bins(&chroms, res);

    // A dense N×N square is per-chromosome by construction; "all" has no shape.
    let Some(region) = parse_range(&args.range)? else {
        return Err(Error::InvalidInput(
            "-t matrix covers one chromosome: pass -r <chrom>".into(),
        ));
    };
    let (i0, i1) = bin_ids_for(&chrom_offset, &chroms, res, &region)?;
    let n = (i1 - i0) as usize;

    let mut pixels = Vec::new();
    collect_cis(&input, res, i0, i1, &mut pixels)?;
    apply_balance(&mut pixels, args, &input, res, &bins)?;

    let mut m = vec![0.0f64; n * n];
    for p in &pixels {
        let (a, b) = ((p.bin1_id - i0) as usize, (p.bin2_id - i0) as usize);
        m[a * n + b] = p.count;
        m[b * n + a] = p.count;
    }

    let mut row = vec![String::new(); n];
    for r in 0..n {
        for (c, cell) in row.iter_mut().enumerate() {
            *cell = g16(m[r * n + c]);
        }
        writeln!(out, "{}", row.join("\t"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    // The sub-f64-precision literal below is the point of the case.
    #![allow(clippy::excessive_precision)]

    use super::*;

    #[test]
    fn g16_matches_printf_percent_dot_16g() {
        // Expected strings generated with Python's `"%.16g"`, which follows the
        // same C formatting rules `fmt` uses for `{:.16g}`.
        let cases: &[(f64, &str)] = &[
            (0.0, "0"),
            (-0.0, "-0"),
            (1.0, "1"),
            (0.1, "0.1"),
            (1329.0, "1329"),
            (3922.172607421875, "3922.172607421875"),
            (594.9529418945312, "594.9529418945312"),
            (3089.741943359375, "3089.741943359375"),
            (0.0001234567890123456789, "0.0001234567890123457"),
            (1e-5, "1e-05"),
            (1e-4, "0.0001"),
            (1e16, "1e+16"),
            (1234567890123456.0, "1234567890123456"),
            (12345678901234567.0, "1.234567890123457e+16"),
            (1.5e20, "1.5e+20"),
            (1e-300, "1e-300"),
            (5e-324, "4.940656458412465e-324"),
            (1.7976931348623157e308, "1.797693134862316e+308"),
            (0.5, "0.5"),
            (2.5, "2.5"),
            (12345678901234567890.0, "1.234567890123457e+19"),
        ];
        for (input, want) in cases {
            assert_eq!(g16(*input), *want, "g16({input})");
        }
        assert_eq!(g16(f64::NAN), "nan");
        assert_eq!(g16(f64::INFINITY), "inf");
        assert_eq!(g16(f64::NEG_INFINITY), "-inf");
    }

    /// The `.hic` scaling must round through f32 *and* take the product in f64
    /// before the cast; both details are visible in the 16th digit.
    #[test]
    fn divisive_scaling_rounds_through_f32() {
        let (w1, w2) = (0.5821020557379992, 0.5821020557379992);
        assert_eq!(g16(scale_divisive_f32(1329.0, w1, w2)), "3922.172607421875");
        // A plain f64 divide lands on a different 16-digit string.
        assert_ne!(g16(1329.0 / (w1 * w2)), "3922.172607421875");

        let (w1, w2) = (1.3004152366749544, 1.3004152366749544);
        assert_eq!(g16(scale_divisive_f32(5225.0, w1, w2)), "3089.741943359375");
    }

    // Moved here from `cooler_rs::convert` with `dense_txt_to_pixels`: the
    // OnTAD `.mat` text format belongs to `dump`/`load`, not to container
    // conversion.

    #[test]
    fn keeps_only_upper_triangle_nonzero() {
        // Lower triangle and zeros are dropped; values are kept exactly.
        let (n, pixels) = dense_txt_to_pixels("1 0 2\n3 4 5\n6 0 7\n").unwrap();
        assert_eq!(n, 3);
        assert_eq!(
            pixels,
            vec![
                Pixel {
                    bin1_id: 0,
                    bin2_id: 0,
                    count: 1.0
                },
                Pixel {
                    bin1_id: 0,
                    bin2_id: 2,
                    count: 2.0
                },
                Pixel {
                    bin1_id: 1,
                    bin2_id: 1,
                    count: 4.0
                },
                Pixel {
                    bin1_id: 1,
                    bin2_id: 2,
                    count: 5.0
                },
                Pixel {
                    bin1_id: 2,
                    bin2_id: 2,
                    count: 7.0
                },
            ]
        );
    }

    #[test]
    fn skips_blank_lines() {
        let (n, pixels) = dense_txt_to_pixels("1 2\n\n3 4\n").unwrap();
        assert_eq!(n, 2);
        assert_eq!(
            pixels,
            vec![
                Pixel {
                    bin1_id: 0,
                    bin2_id: 0,
                    count: 1.0
                },
                Pixel {
                    bin1_id: 0,
                    bin2_id: 1,
                    count: 2.0
                },
                Pixel {
                    bin1_id: 1,
                    bin2_id: 1,
                    count: 4.0
                },
            ]
        );
    }

    #[test]
    fn rejects_non_square_matrix() {
        let err = dense_txt_to_pixels("1 2 3\n4 5\n").unwrap_err();
        assert!(err.to_string().contains("not a square"), "{err}");
    }

    #[test]
    fn rejects_non_numeric_entry() {
        let err = dense_txt_to_pixels("1 2\n3 x\n").unwrap_err();
        assert!(err.to_string().contains("'x' is not a number"), "{err}");
    }

    #[test]
    fn rejects_empty_input() {
        let err = dense_txt_to_pixels("  \n\n").unwrap_err();
        assert!(err.to_string().contains("empty"), "{err}");
    }
}
