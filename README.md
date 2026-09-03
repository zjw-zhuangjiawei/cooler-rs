# cooler-rs

A Rust implementation of the [cooler](https://cooler.readthedocs.io/en/latest/schema.html)
file format — read/write for `.cool` and `.mcool` Hi-C contact matrices (HDF5) —
plus the `cooler-rs` command-line tool for Hi-C analysis.

## Features

- **Library**: `cooler_rs::Cooler` / `cooler_rs::Mcool` read and write single-resolution
  `.cool` and multi-resolution `.mcool` files following the cooler schema
  (bin table, sparse pixel matrix, chromosome offsets).
- **CLI**: a single `cooler-rs` binary:
  - `cooler-rs call-tad` — hierarchical TAD calling (`--method ontad`, a port of
    [OnTAD v1.4](https://github.com/anlin00007/OnTAD); `--method domaincaller`,
    a TADLib port; `--method armatus`, an
    [Armatus 2.3](https://github.com/kingsfordgroup/armatus) port).
  - `cooler-rs convert` — format conversion (e.g. `--from dense-txt`, a dense
    N×N text matrix to `.cool`).
  - `cooler-rs balance` — out-of-core matrix balancing / iterative correction
    (port of `cooler balance`): genome-wide, cis-only and trans-only modes,
    MAD-max / min-nnz / min-count / blacklist bin filters, writes a `weight`
    column back to the `.cool`/`.mcool` file.

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
cargo run --release -- call-tad /tmp/toy.cool --method ontad --chr chr1 -o out

# DomainCaller TAD calling (TADLib port; writes .domains + .DIs.bedGraph)
cargo run --release -- call-tad /tmp/toy.cool --method domaincaller --chr chr1 -o out

# Armatus TAD calling (multiresolution; writes .consensus.txt)
cargo run --release -- call-tad /tmp/toy.cool --method armatus --chr chr1 --gamma 0.5 -o out

# Dense matrix -> .cool
cargo run --release -- convert --from dense-txt matrix.txt -o out.cool -L 250000000 -r 100000

# Balance a contact matrix (writes a 'weight' column back to the file)
cargo run --release -- balance /tmp/toy.cool
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
| `stats`           | pomegranate 0.10.0 port: GMM / HMM / normal / discrete |
| `error` / `types` | error type and shared structs       |

## License

MIT
