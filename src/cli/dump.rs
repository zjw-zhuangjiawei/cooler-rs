//! `dump` — write tables out of a `.hic`/`.cool`/`.mcool` file to stdout.
//!
//! This is a port of `hictk dump` (`src/hictk/dump/`), sized so the two
//! outputs can be `diff`ed. The formatting rules below are copied from
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
//! Not ported (yet): `--table cells` (`.scool` only), `--range2`,
//! `--query-file`, `--cis-only`/`--trans-only`, `--matrix-type`/`--matrix-unit`
//! (oe/expected/FRAG), `--sorted`/`--unsorted`. Pixels are always emitted in
//! the ascending order our readers return them in, which is what `hictk dump`
//! does by default.

use std::io::{self, BufWriter, Write};

use clap::{Args, ValueEnum};

use cooler_rs::cooler::Cooler;
use cooler_rs::error::{Error, Result};
use cooler_rs::file::File;
use cooler_rs::hic::HiCFile;
use cooler_rs::mcool::Mcool;
use cooler_rs::region::Region;
use cooler_rs::types::{Bin, Chrom, Pixel, WeightType};

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Table {
    /// Chromosome table: `name<TAB>length`
    Chroms,
    /// Bin table: `chrom<TAB>start<TAB>end`
    Bins,
    /// Interaction table (default): `bin1_id<TAB>bin2_id<TAB>count`
    Pixels,
    /// Normalization method names, one per line
    Normalizations,
    /// Available resolutions, one per line
    Resolutions,
    /// Normalization weights, one column per method
    Weights,
}

#[derive(Args)]
pub struct DumpArgs {
    /// Path to a .hic, .cool or .mcool file
    pub uri: String,

    /// HiC matrix resolution (required for .hic/.mcool with >1 resolution)
    #[arg(long, value_name = "BP")]
    pub resolution: Option<u32>,

    /// Name of the table to dump
    #[arg(short = 't', long, value_enum, default_value_t = Table::Pixels)]
    pub table: Table,

    /// UCSC-style coordinates of the region to dump (`chr1:0-1000`)
    #[arg(short = 'r', long, default_value = "all")]
    pub range: String,

    /// Balance interactions using the given method (`.hic` norm vector name,
    /// or a `bins` column for `.cool`/`.mcool`)
    #[arg(short = 'b', long, default_value = "NONE")]
    pub balance: String,

    /// Output pixels in BG2 format (chrom/start/end pairs instead of bin ids)
    #[arg(long)]
    pub join: bool,
}

pub fn run(args: DumpArgs) -> Result<()> {
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    match args.table {
        Table::Resolutions => dump_resolutions(&args, &mut out),
        Table::Normalizations => dump_normalizations(&args, &mut out),
        Table::Chroms => dump_chroms(&args, &mut out),
        Table::Bins => dump_bins(&args, &mut out),
        Table::Weights => dump_weights(&args, &mut out),
        Table::Pixels => dump_pixels(&args, &mut out),
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

fn open_input(args: &DumpArgs) -> Result<Input> {
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
fn resolve_resolution(args: &DumpArgs, input: &Input) -> Result<u32> {
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

fn dump_resolutions(args: &DumpArgs, out: &mut impl Write) -> Result<()> {
    let input = open_input(args)?;
    // `.hic` zoom levels come back finest-last; hictk lists them ascending.
    let mut res = resolutions_of(&input)?;
    res.sort_unstable();
    if let Some(want) = args.resolution {
        if !res.contains(&want) {
            return Err(Error::InvalidInput(format!(
                "file does not have interactions for {want} resolution"
            )));
        }
        res.retain(|r| *r == want);
    }
    for r in res {
        // hictk prints `variable` for the variable-bin-size resolution (0).
        if r == 0 {
            writeln!(out, "variable")?;
        } else {
            writeln!(out, "{r}")?;
        }
    }
    Ok(())
}

fn dump_normalizations(args: &DumpArgs, out: &mut impl Write) -> Result<()> {
    let input = open_input(args)?;
    let mut names = match &input {
        Input::Hic(h) => h.avail_normalizations()?,
        Input::Mcool(m) => {
            let res = resolve_resolution(args, &input)?;
            m.cooler(res as u64)?.bins_column_names()?
        }
        Input::Cool(c) => c.bins_column_names()?,
    };
    names.sort();
    names.dedup();
    for n in names {
        writeln!(out, "{n}")?;
    }
    Ok(())
}

fn dump_chroms(args: &DumpArgs, out: &mut impl Write) -> Result<()> {
    let input = open_input(args)?;
    let chroms = match &input {
        Input::Hic(h) => h.chromosomes(),
        _ => {
            let res = resolve_resolution(args, &input)?;
            File::open(&args.uri, res)?.chroms()?
        }
    };
    let Some(region) = parse_range(&args.range)? else {
        for c in &chroms {
            // hictk skips the `ALL` pseudo-chromosome that `.hic` files carry.
            if c.name != "ALL" {
                writeln!(out, "{}\t{}", c.name, c.length)?;
            }
        }
        return Ok(());
    };
    for c in &chroms {
        if c.name == region.chrom {
            writeln!(out, "{}\t{}", c.name, c.length)?;
            return Ok(());
        }
    }
    Err(Error::InvalidInput(format!(
        "unknown sequence label: {}",
        region.chrom
    )))
}

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

fn dump_bins(args: &DumpArgs, out: &mut impl Write) -> Result<()> {
    let input = open_input(args)?;
    let res = resolve_resolution(args, &input)?;
    let chroms = input_chroms(&input, res)?;
    let (bins, chrom_offset) = tiled_bins(&chroms, res);

    let (i0, i1) = match parse_range(&args.range)? {
        None => (0, bins.len() as i64),
        Some(r) => bin_ids_for(&chrom_offset, &chroms, res, &r)?,
    };
    for b in &bins[i0 as usize..i1 as usize] {
        let chrom = &chroms[b.chrom_id as usize].name;
        writeln!(out, "{chrom}\t{}\t{}", b.start, b.end)?;
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

fn dump_weights(args: &DumpArgs, out: &mut impl Write) -> Result<()> {
    let input = open_input(args)?;
    let res = resolve_resolution(args, &input)?;
    let chroms = input_chroms(&input, res)?;
    let (bins, chrom_offset) = tiled_bins(&chroms, res);

    let (i0, i1) = match parse_range(&args.range)? {
        None => (0, bins.len() as i64),
        Some(r) => bin_ids_for(&chrom_offset, &chroms, res, &r)?,
    };

    let (names, columns): (Vec<String>, Vec<Vec<f64>>) = match &input {
        // `.hic` vectors are divisive and are printed as stored.
        Input::Hic(h) => {
            let names = h.avail_normalizations()?;
            let mut columns = Vec::with_capacity(names.len());
            for n in &names {
                columns.push(hic_global_weights(h, res, n)?);
            }
            (names, columns)
        }
        // cooler `bins` columns have no inherent convention, so they are
        // resolved per column and this table is printed in the DIVISIVE one
        // whatever the column holds (`src/hictk/dump/common.cpp:88-92`): a
        // multiplicative `weight` comes out as `1/w`, a divisive `KR` as
        // stored. Only the `bins` table is read here — unlike the `.hic`
        // variant, `File` on a cooler never touches the pixels.
        _ => {
            let file = File::open(&args.uri, res)?;
            let mut names = file.avail_normalizations()?;
            names.sort();
            names.dedup();
            let mut columns = Vec::with_capacity(names.len());
            for n in &names {
                let w = file_weights(&file, n, bins.len())?;
                columns.push(if file_weight_type(&file, n).is_divisive() {
                    w
                } else {
                    w.into_iter().map(|v| 1.0 / v).collect()
                });
            }
            (names, columns)
        }
    };

    if names.is_empty() {
        return Ok(());
    }
    writeln!(out, "{}", names.join("\t"))?;
    // hictk joins the formatted row with `\t`, i.e. default (shortest
    // round-trip) float formatting, not `%.16g`.
    let mut row = vec![String::new(); names.len()];
    for i in i0.max(0) as usize..i1 as usize {
        for (j, col) in columns.iter().enumerate() {
            row[j] = fmt_short(col[i]);
        }
        writeln!(out, "{}", row.join("\t"))?;
    }
    Ok(())
}

/// Shortest round-trip float formatting, with `fmt`'s lower-case NaN/Infinity
/// spellings (Rust prints `NaN`/`inf`).
fn fmt_short(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    format!("{x}")
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

fn dump_pixels(args: &DumpArgs, out: &mut impl Write) -> Result<()> {
    let input = open_input(args)?;
    let res = resolve_resolution(args, &input)?;
    let chroms = input_chroms(&input, res)?;
    let (bins, chrom_offset) = tiled_bins(&chroms, res);

    // A region is a cis block: both ends must fall inside it. `File::fetch`
    // is a *row* query (bin2 spans the whole chromosome), so filter here.
    let (i0, i1) = match parse_range(&args.range)? {
        None => (0i64, bins.len() as i64),
        Some(r) => bin_ids_for(&chrom_offset, &chroms, res, &r)?,
    };

    // The *precision* depends on the format — `pixel_selector_impl.hpp:141`
    // divides an f32 by `(float)(w1 * w2)` for `.hic`, while
    // `weights_impl.hpp:182` divides an f64 by the f64 product — but the
    // *operation* depends on the column's resolved type, not the container:
    // a cooler `KR` column is divisive and its `weight` column is not.
    let is_hic = matches!(input, Input::Hic(_));
    let balance = match (args.balance != "NONE").then_some(args.balance.as_str()) {
        None => None,
        Some(name) => Some(match &input {
            Input::Hic(h) => (hic_global_weights(h, res, name)?, WeightType::Divisive),
            _ => {
                let file = File::open(&args.uri, res)?;
                let w = file_weights(&file, name, bins.len())?;
                let t = file_weight_type(&file, name);
                (w, t)
            }
        }),
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
    if let Some((w, wtype)) = &balance {
        for p in &mut pixels {
            let (a, b) = (p.bin1_id as usize, p.bin2_id as usize);
            p.count = if is_hic {
                scale_divisive_f32(p.count, w[a], w[b])
            } else if wtype.is_divisive() {
                p.count / (w[a] * w[b])
            } else {
                p.count * (w[a] * w[b])
            };
        }
    }

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
}
