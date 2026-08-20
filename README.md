# cooler-rs

A Rust implementation of the [cooler](https://cooler.readthedocs.io/en/latest/schema.html)
file format — read/write for `.cool` and `.mcool` Hi-C contact matrices (HDF5) —
plus Hi-C analysis command-line tools.

## Features

- **Library**: `cooler_rs::Cooler` / `cooler_rs::Mcool` read and write single-resolution
  `.cool` and multi-resolution `.mcool` files following the cooler schema
  (bin table, sparse pixel matrix, chromosome offsets).
- **OnTAD**: hierarchical TAD calling (port of [OnTAD v1.4](https://github.com/zhanglabtools/OnTAD)).
- **mat2cool**: convert a dense N×N text matrix to `.cool`.

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

# OnTAD hierarchical TAD calling
cargo run --release --bin ontad /tmp/toy.cool --chr chr1 -o out

# Dense matrix -> .cool
cargo run --release --bin mat2cool matrix.txt -o out.cool -L 250000000 -r 100000
```

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
| `error` / `types` | error type and shared structs       |

## License

MIT
