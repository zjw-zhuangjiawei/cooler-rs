//! `cooler-rs zoomify` — coarsen a single-resolution `.cool` into a
//! multi-resolution `.mcool`.

use std::path::PathBuf;

use clap::Args;
use cooler_rs::{zoomify_cooler, ZoomifyParams};

#[derive(Args)]
pub struct ZoomifyArgs {
    /// Input file (single-resolution .cool)
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output file (.mcool)
    #[arg(short = 'o', long, value_name = "FILE")]
    output: PathBuf,

    /// Target resolutions (bin sizes), comma-separated.
    /// Each must be a multiple of the base resolution.
    #[arg(long, value_name = "R", value_delimiter = ',', num_args = 1..)]
    resolutions: Vec<u32>,

    /// Do not include the base resolution in the output (copy is the default)
    #[arg(long)]
    no_copy_base_resolution: bool,

    /// Use power-of-two (×2) steps for auto-generated resolutions
    /// (nice 1-2-5 steps are the default)
    #[arg(long)]
    pow2_steps: bool,

    /// Overwrite the output file if it exists
    #[arg(short = 'f', long)]
    force: bool,
}

pub fn run(args: ZoomifyArgs) -> cooler_rs::Result<()> {
    let params = ZoomifyParams {
        resolutions: args.resolutions,
        copy_base_resolution: !args.no_copy_base_resolution,
        nice_steps: !args.pow2_steps,
        force: args.force,
    };
    zoomify_cooler(&args.input, &args.output, &params)
}
