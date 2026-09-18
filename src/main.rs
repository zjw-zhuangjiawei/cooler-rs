//! `cooler-rs` — unified command-line interface for the cooler-rs crate.
//!
//! Subcommands:
//!   `call-tad`  call TADs from a .cool/.mcool contact matrix, one
//!               subcommand per algorithm
//!   `compare`   compare multiple matrices and plot a correlation heatmap
//!   `convert`   convert other matrix formats to/from cooler format
//!   `normalize` normalize a contact matrix (ic or raichu)
//!   `validate`  check a .cool/.mcool file for internal consistency
//!   `zoomify`   coarsen a single-resolution .cool into a multi-resolution .mcool

mod cli;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "cooler-rs",
    about = "Read/write .cool/.mcool Hi-C contact matrices and run Hi-C analyses",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Call TADs from a .cool/.mcool contact matrix
    CallTad(Box<cli::call_tad::CallTadArgs>),
    /// Compare multiple contact matrices and plot a correlation heatmap
    Compare(cli::compare::CompareArgs),
    /// Convert other matrix formats to/from cooler format
    Convert(cli::convert::ConvertArgs),
    /// Write tables from a .hic/.cool/.mcool file to stdout
    Dump(cli::dump::DumpArgs),
    /// Normalize a contact matrix (iterative correction or Raichu)
    Normalize(cli::normalize::NormalizeArgs),
    /// Check a .cool/.mcool file for internal consistency
    Validate(cli::validate::ValidateArgs),
    /// Coarsen a single-resolution .cool into a multi-resolution .mcool
    Zoomify(cli::zoomify::ZoomifyArgs),
}

fn main() {
    // Default to `info` so per-chromosome progress is visible out of the box;
    // `RUST_LOG` still overrides (e.g. `RUST_LOG=debug` for window-level logs).
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();
    let result = match cli.command {
        Commands::CallTad(args) => cli::call_tad::run(*args),
        Commands::Compare(args) => cli::compare::run(args),
        Commands::Convert(args) => cli::convert::run(args),
        Commands::Dump(args) => cli::dump::run(args),
        Commands::Normalize(args) => cli::normalize::run(args),
        Commands::Validate(args) => cli::validate::run(args),
        Commands::Zoomify(args) => cli::zoomify::run(args),
    };
    if let Err(e) = result {
        log::error!("{e}");
        std::process::exit(1);
    }
}
