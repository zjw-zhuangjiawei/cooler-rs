# cooler-rs

A Rust implementation of the [cooler](https://cooler.readthedocs.io/en/latest/schema.html)
file format — read/write for `.cool` and `.mcool` Hi-C contact matrices (HDF5) —
plus the `cooler-rs` command-line tool for Hi-C analysis.

## Features

- **Library**: `cooler_rs::Cooler` / `cooler_rs::Mcool` read and write single-resolution
  `.cool` and multi-resolution `.mcool` files following the cooler schema
  (bin table, sparse pixel matrix, chromosome offsets).
- **CLI**: a single `cooler-rs` binary:
  - `cooler-rs call-tad` — TAD calling, one subcommand per algorithm:
    - `call-tad ontad` — a port of
      [OnTAD v1.4](https://github.com/anlin00007/OnTAD)
    - `call-tad domaincaller` — a TADLib port
    - `call-tad armatus` — an
      [Armatus 2.3](https://github.com/kingsfordgroup/armatus) port
    - `call-tad arrowhead` — a juicer Arrowhead port
    - `call-tad hicexplorer` — HiCExplorer's `hicFindTADs`: z-scores the
      matrix per chromosome, scores every bin with the mean z-score of the
      contacts crossing it over a range of window sizes (the TAD-separation
      score), and calls a boundary at each local minimum that clears the delta
      and significance filters. Writes `_tad_score.bm`, `_zscore_matrix.cool`,
      `_boundaries.bed`, `_boundaries.gff`, `_domains.bed` and
      `_score.bedgraph`; runs genome-wide, and takes `--chromosomes` for a
      subset.
  - `cooler-rs convert` — format conversion (e.g. `--from dense-txt`, a dense
    N×N text matrix to `.cool`).
  - `cooler-rs zoomify` — coarsen a single-resolution `.cool` into a
    multi-resolution `.mcool` (port of `cooler zoomify` / `hictk zoomify`):
    pools `factor`×`factor` bins per chromosome and sums counts, with nice
    (1-2-5) or power-of-two auto-generated resolutions.
  - `cooler-rs normalize` — contact matrix normalization:
    - `--method ic` (default) — out-of-core matrix balancing / iterative
      correction (port of `cooler balance`): genome-wide, cis-only and
      trans-only modes, MAD-max / min-nnz / min-count / blacklist bin filters,
      writes a `weight` column back to the `.cool`/`.mcool` file.
    - `--method raichu` — the Raichu sliding-window optimizer (port of
      [RaichuNorm](https://github.com/XiaoTaoWang/Raichu)), writes an
      `obj_weight` column.
  - `cooler-rs compare` — pairwise similarity between N contact matrices,
    rendered as a correlation heatmap: `--metric scc` (the HiCRep
    stratum-adjusted correlation coefficient, a port of
    [hicrep](https://github.com/cmdoret/hicrep)), `--metric pearson`, and
    `--metric spearman`.
  - `cooler-rs dump` — write tables out of a `.hic`/`.cool`/`.mcool` file to
    stdout (`chroms`, `bins`, `pixels`, `resolutions`, `normalizations`,
    `weights`), a port of `hictk dump` that reproduces its output byte-for-byte.
  - `cooler-rs validate` — check a `.cool`/`.mcool` file for internal
    consistency (schema + index invariants: offsets, bin/chrom codes, pixel
    ordering and ranges). Checks every resolution of a `.mcool`; prints each
    issue and exits non-zero if any was found.

## Usage

### Library

Write a `.cool`:

```rust
use cooler_rs::{Chrom, CoolerWriter, Pixel};

let chroms = vec![
    Chrom { name: "chr1".into(), length: 1_000_000 },
    Chrom { name: "chr2".into(), length: 500_000 },
];
let writer = CoolerWriter::create("out.cool", &chroms, 100_000)?;
writer.write_pixels(&[Pixel { bin1_id: 0, bin2_id: 3, count: 42.0 }])?;
```

Read an `.mcool`:

```rust
use cooler_rs::Mcool;

let mcool = Mcool::open("out.mcool")?;
for res in mcool.resolutions()? {
    let cool = mcool.cooler(res)?;
    println!("{res}: {} pixels", cool.n_pixels()?);
}
```

### Command-line

```sh
# Generate toy Hi-C data (for testing downstream tools)
cargo run --example generate /tmp/toy

# Hierarchical TAD calling (OnTAD, default)
cargo run --release -- call-tad ontad /tmp/toy.cool --chr chr1 -o out

# DomainCaller TAD calling (TADLib port; writes .domains + .DIs.bedGraph)
cargo run --release -- call-tad domaincaller /tmp/toy.cool --chr chr1 -o out

# Armatus TAD calling (multiresolution; writes .consensus.txt)
cargo run --release -- call-tad armatus /tmp/toy.cool --chr chr1 --gamma 0.5 -o out

# Dense matrix -> .cool
cargo run --release -- convert --from dense-txt matrix.txt -o out.cool -L 250000000 -r 100000

# Dump a table to stdout (hictk-compatible output)
cargo run --release -- dump ranks.hic -t resolutions
cargo run --release -- dump ranks.hic -t pixels --resolution 10000 -r 2L:0-30000 -b KR
cargo run --release -- dump ranks.hic -t pixels --resolution 10000 -r 2L:0-30000 --join

# Coarsen a single-resolution .cool into a multi-resolution .mcool
cargo run --release -- zoomify /tmp/toy.cool -o /tmp/toy.mcool

# Balance a contact matrix (iterative correction; writes a 'weight' column)
cargo run --release -- normalize /tmp/toy.cool

# Normalize with Raichu (writes an 'obj_weight' column)
cargo run --release -- normalize /tmp/toy.cool --method raichu

# Compare two matrices and write a correlation heatmap (default: all metrics)
cargo run --release -- compare /tmp/toy.cool /tmp/toy2.cool --metric scc --metric pearson -o heatmap

# hicFindTADs TAD boundaries, genome-wide (weights come from a bins column)
cargo run --release -- call-tad hicexplorer /tmp/toy.cool --norm weight \
    --min-depth 60000 --max-depth 180000 --window-step 20000 -o TADs
```

Run `cooler-rs <COMMAND> --help` for the full option list of each subcommand.

### Examples

- `examples/generate.rs` — generate a toy `.cool`/`.mcool` with TAD-like block structure.

## Build

```sh
cargo build --release
cargo test
```

Requires HDF5. Use the `hdf5-metno` `static` feature (see `Cargo.toml`) to
link statically without a system HDF5.

## Modules

| Module            | Purpose                             |
|-------------------|-------------------------------------|
| `cooler`          | `.cool` reading / writing           |
| `mcool`           | `.mcool` multi-resolution container |
| `ontad`           | OnTAD hierarchical TAD algorithm    |
| `domaincaller`    | TADLib DomainCaller port (Dixon et al., 2012) |
| `armatus`         | Armatus 2.3 multiresolution TAD port (Filippova et al., 2014) |
| `findtads`        | hicFindTADs TAD-separation score and boundary caller |
| `balance`         | iterative-correction matrix balancing (port of `cooler balance`) |
| `zoomify`         | coarsen a `.cool` into a multi-resolution `.mcool` (port of `cooler zoomify`) |
| `raichu`          | Raichu sliding-window normalization (port of RaichuNorm) |
| `compare`         | pairwise SCC / Pearson / Spearman similarity (port of hicrep) |
| `stats`           | pomegranate 0.10.0 port: GMM / HMM / normal / discrete |
| `error` / `types` | error type and shared structs       |

## License

MIT
