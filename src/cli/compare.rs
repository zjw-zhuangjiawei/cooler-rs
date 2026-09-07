//! `cooler-rs compare` — pairwise similarity between N contact matrices,
//! rendered as a correlation heatmap.

use std::path::{Path, PathBuf};

use clap::Args;
use cooler_rs::{compare_pair, CompareMetric, CompareParams, Cooler, Error, Mcool, Result};
use ndarray::Array2;
use plotters::prelude::*;

#[derive(Clone, Copy, clap::ValueEnum)]
enum MetricArg {
    Scc,
    Pearson,
    Spearman,
}

impl MetricArg {
    fn metric(self) -> CompareMetric {
        match self {
            MetricArg::Scc => CompareMetric::Scc,
            MetricArg::Pearson => CompareMetric::Pearson,
            MetricArg::Spearman => CompareMetric::Spearman,
        }
    }
}

#[derive(Args)]
pub struct CompareArgs {
    /// Input files (.cool/.mcool); at least two
    #[arg(value_name = "COOL", num_args = 2..)]
    inputs: Vec<PathBuf>,

    /// Metrics to compute (repeatable); default: all three
    #[arg(long, value_enum, value_name = "METRIC")]
    metric: Vec<MetricArg>,

    /// Mean-filter half-size for SCC (0 = no smoothing)
    #[arg(long, default_value_t = 1)]
    h: usize,

    /// Maximum genomic distance in bp for SCC (-1 = whole chromosome)
    #[arg(long, value_name = "BP", default_value_t = -1)]
    max_dist: i64,

    /// Downsample the larger matrix to the smaller's contact count (SCC)
    #[arg(long)]
    downsample: bool,

    /// Resolution to compare (.mcool input)
    #[arg(long, value_name = "N")]
    res: Option<u64>,

    /// Chromosome names to compare (comma-separated; empty = all)
    #[arg(long, value_delimiter = ',', value_name = "NAME")]
    chroms: Vec<String>,

    /// Output heatmap stem. One metric writes '{stem}.png'; several write
    /// '{stem}.{metric}.png'.
    #[arg(short = 'o', long, value_name = "STEM", default_value = "heatmap")]
    output: String,

    /// Also print the pairwise matrices as TSV to stdout
    #[arg(long)]
    text: bool,
}

pub fn run(args: CompareArgs) -> Result<()> {
    let metrics: Vec<CompareMetric> = if args.metric.is_empty() {
        vec![
            CompareMetric::Scc,
            CompareMetric::Pearson,
            CompareMetric::Spearman,
        ]
    } else {
        args.metric.iter().map(|m| m.metric()).collect()
    };

    let coolers: Vec<Cooler> = args
        .inputs
        .iter()
        .map(|p| open_cooler(p, args.res))
        .collect::<Result<_>>()?;
    let labels: Vec<String> = args.inputs.iter().map(|p| basename(p)).collect();
    let n = coolers.len();

    let params = CompareParams {
        h: args.h,
        max_dist: args.max_dist,
        downsample: args.downsample,
        chroms: args.chroms.clone(),
    };

    for &metric in &metrics {
        let mut mat = Array2::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                let v = if i == j {
                    1.0
                } else {
                    mean(&compare_pair(&coolers[i], &coolers[j], metric, &params)?)
                };
                mat[[i, j]] = v;
                mat[[j, i]] = v;
            }
        }

        let path = heatmap_path(&args.output, metric, metrics.len());
        render_heatmap(&mat, &labels, metric, &path)?;
        log::info!("wrote {}", path.display());
        if args.text {
            print_tsv(&mat, &labels, metric);
        }
    }
    Ok(())
}

/// Open a `.cool`/`.mcool` path at the requested resolution (mirrors the
/// resolution selection used by `normalize`).
fn open_cooler(path: &Path, res: Option<u64>) -> Result<Cooler> {
    let fin = path.display().to_string();
    if fin.ends_with(".cool") {
        Cooler::open_any(&fin)
    } else if fin.ends_with(".mcool") {
        let mcool = Mcool::open(&fin)?;
        let resolutions = mcool.resolutions()?;
        let res = match (res, resolutions.as_slice()) {
            (Some(r), _) => r,
            (None, [only]) => *only,
            (None, _) => {
                return Err(Error::InvalidInput(format!(
                    ".mcool contains multiple resolutions; select one with --res ({resolutions:?})"
                )));
            }
        };
        mcool.cooler(res)
    } else {
        Err(Error::InvalidInput(
            "input must be a .cool or .mcool file".into(),
        ))
    }
}

/// Mean of the per-chromosome values, ignoring NaN.
fn mean(values: &[(String, f64)]) -> f64 {
    let finite: Vec<f64> = values
        .iter()
        .map(|&(_, v)| v)
        .filter(|v| v.is_finite())
        .collect();
    if finite.is_empty() {
        f64::NAN
    } else {
        finite.iter().sum::<f64>() / finite.len() as f64
    }
}

fn metric_name(m: CompareMetric) -> &'static str {
    match m {
        CompareMetric::Scc => "scc",
        CompareMetric::Pearson => "pearson",
        CompareMetric::Spearman => "spearman",
    }
}

fn heatmap_path(stem: &str, metric: CompareMetric, n_metrics: usize) -> PathBuf {
    if n_metrics == 1 {
        PathBuf::from(format!("{stem}.png"))
    } else {
        PathBuf::from(format!("{stem}.{}.png", metric_name(metric)))
    }
}

fn basename(p: &Path) -> String {
    p.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.display().to_string())
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(n).collect();
        out.push('…');
        out
    }
}

/// Diverging blue-white-red scale over [-1, 1].
fn corr_color(v: f64) -> RGBColor {
    let v = if v.is_finite() {
        v.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let (r, g, b) = if v >= 0.0 {
        // white -> red (220, 50, 50)
        (255.0 - 35.0 * v, 255.0 - 205.0 * v, 255.0 - 205.0 * v)
    } else {
        // white -> blue (50, 50, 220)
        let t = -v;
        (255.0 - 205.0 * t, 255.0 - 205.0 * t, 255.0 - 35.0 * t)
    };
    RGBColor(r.round() as u8, g.round() as u8, b.round() as u8)
}

/// Black on light cells, white on dark cells.
fn text_color(bg: RGBColor) -> RGBColor {
    let lum = 0.299 * f64::from(bg.0) + 0.587 * f64::from(bg.1) + 0.114 * f64::from(bg.2);
    if lum < 140.0 {
        WHITE
    } else {
        BLACK
    }
}

fn render_heatmap(
    mat: &Array2<f64>,
    labels: &[String],
    metric: CompareMetric,
    path: &Path,
) -> Result<()> {
    let n = mat.nrows();
    let cell = 80usize;
    let margin = 130usize;
    let title_h = 30usize;
    let w = margin + n * cell + 24;
    let h = margin + title_h + n * cell + 24;

    let root = BitMapBackend::new(path, (w as u32, h as u32)).into_drawing_area();
    root.fill(&WHITE).map_err(plot_err)?;

    root.draw_text(
        &format!("{} correlation (n={n})", metric_name(metric)),
        &("sans-serif", 20).into_font().color(&BLACK),
        (12, 10),
    )
    .map_err(plot_err)?;

    let x0 = margin;
    let y0 = margin + title_h;

    for i in 0..n {
        for j in 0..n {
            let v = mat[[i, j]];
            let bg = corr_color(v);
            let rect = Rectangle::new(
                [
                    ((x0 + j * cell) as i32, (y0 + i * cell) as i32),
                    ((x0 + (j + 1) * cell) as i32, (y0 + (i + 1) * cell) as i32),
                ],
                bg.filled(),
            );
            root.draw(&rect).map_err(plot_err)?;

            let label = if v.is_finite() {
                format!("{v:.2}")
            } else {
                "NaN".to_string()
            };
            let tc = text_color(bg);
            let style = ("sans-serif", 14).into_font().color(&tc);
            root.draw_text(
                &label,
                &style,
                (
                    (x0 + j * cell + cell / 2 - 12) as i32,
                    (y0 + i * cell + cell / 2 - 9) as i32,
                ),
            )
            .map_err(plot_err)?;
        }
    }

    // Row labels (left, horizontal) and column labels (top, rotated).
    for (i, lab) in labels.iter().enumerate() {
        let l = truncate(lab, 14);
        root.draw_text(
            &l,
            &("sans-serif", 13).into_font().color(&BLACK),
            (8, (y0 + i * cell + cell / 2 - 8) as i32),
        )
        .map_err(plot_err)?;
    }
    for (j, lab) in labels.iter().enumerate() {
        let l = truncate(lab, 18);
        let style = ("sans-serif", 13)
            .into_font()
            .color(&BLACK)
            .transform(FontTransform::Rotate90);
        root.draw_text(
            &l,
            &style,
            ((x0 + j * cell + cell / 2 - 8) as i32, (margin - 6) as i32),
        )
        .map_err(plot_err)?;
    }

    root.present().map_err(plot_err)
}

fn print_tsv(mat: &Array2<f64>, labels: &[String], metric: CompareMetric) {
    println!("# {}", metric_name(metric));
    let mut header = String::new();
    for l in labels {
        header.push('\t');
        header.push_str(l);
    }
    println!("{header}");
    for i in 0..mat.nrows() {
        let mut line = labels[i].clone();
        for j in 0..mat.ncols() {
            let v = mat[[i, j]];
            if v.is_finite() {
                line.push_str(&format!("\t{v:.6}"));
            } else {
                line.push_str("\tNaN");
            }
        }
        println!("{line}");
    }
}

fn plot_err<E: std::fmt::Display>(e: E) -> Error {
    Error::InvalidInput(format!("heatmap rendering failed: {e}"))
}
