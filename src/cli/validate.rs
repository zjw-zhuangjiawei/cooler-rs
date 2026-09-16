//! `cooler-rs validate` — check a `.cool`/`.mcool` file for internal
//! consistency (schema + index invariants) via [`Cooler::validate`].
//!
//! Every issue is printed; the command exits non-zero when at least one was
//! found, so it can gate a pipeline. `.hic` is not supported: the validator
//! checks the stored cooler schema, and `.hic` bins/offsets are synthesized
//! when the file is read rather than stored in the file.

use std::path::Path;

use clap::Args;
use cooler_rs::{Cooler, Error, Mcool};

#[derive(Args)]
pub struct ValidateArgs {
    /// Input file (.cool or .mcool)
    #[arg(value_name = "INPUT")]
    input: std::path::PathBuf,

    /// Check only this resolution (default: every resolution of a .mcool)
    #[arg(long, value_name = "N")]
    res: Option<u64>,

    /// Print at most this many issues per resolution
    #[arg(long, default_value_t = 20, value_name = "N")]
    max_issues: usize,
}

pub fn run(args: ValidateArgs) -> cooler_rs::Result<()> {
    let mut n_issues = 0usize;
    for (label, clr) in targets(&args.input, args.res)? {
        let report = clr.validate()?;
        if report.is_ok() {
            log::info!("{label}: OK");
            continue;
        }
        n_issues += report.issues.len();
        println!("{label}: {} issue(s) found", report.issues.len());
        for issue in report.issues.iter().take(args.max_issues) {
            println!("  {issue}");
        }
        if report.issues.len() > args.max_issues {
            println!("  ... ({} more)", report.issues.len() - args.max_issues);
        }
    }
    if n_issues > 0 {
        return Err(Error::Format(format!("{n_issues} issue(s) found")));
    }
    Ok(())
}

/// Resolve the input into the list of `(group path, collection)` pairs to
/// check: one per resolution for a `.mcool`, `/` for a `.cool`.
fn targets(path: &Path, res: Option<u64>) -> cooler_rs::Result<Vec<(String, Cooler)>> {
    let fin = path.display().to_string();
    if fin.ends_with(".hic") {
        return Err(Error::InvalidInput(
            "validate does not support .hic files".into(),
        ));
    }
    if !fin.ends_with(".mcool") {
        return Ok(vec![("/".into(), Cooler::open_any(&fin)?)]);
    }

    let mcool = Mcool::open(&fin)?;
    let resolutions = mcool.resolutions()?;
    let selected = match res {
        Some(r) => vec![r],
        None => resolutions,
    };
    selected
        .into_iter()
        .map(|r| Ok((format!("/resolutions/{r}"), mcool.cooler(r)?)))
        .collect()
}
