//! `cooler-rs` — unified command-line interface for the cooler-rs crate.
//!
//! Subcommands:
//!   `call-tad`  call hierarchical TADs from a .cool/.mcool contact matrix
//!   `convert`   convert other matrix formats to/from cooler format

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
    /// Call hierarchical TADs from a .cool/.mcool contact matrix
    CallTad(cli::call_tad::CallTadArgs),
    /// Convert other matrix formats to/from cooler format
    Convert(cli::convert::ConvertArgs),
}

fn main() {
    env_logger::init();

    let cli = Cli::parse();
    let result = match cli.command {
        Commands::CallTad(args) => cli::call_tad::run(args),
        Commands::Convert(args) => cli::convert::run(args),
    };
    if let Err(e) = result {
        log::error!("{e}");
        std::process::exit(1);
    }
}
