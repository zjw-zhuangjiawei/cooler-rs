#!/usr/bin/env python3
"""Dump an OnTAD `.mat` fixture from the 4DN mcool.

OnTAD's C++ `loadMatrix` reads a plain dense N x N text matrix
(whitespace-separated, no header, `atof` per token). The reference fixture is
raw (unbalanced) contact counts, matching our Rust `matrix_from_cooler`
semantics (`p.count`, no balance column). chr2L at 50 kb keeps the fixture
small (471 x 471).

Run inside the cooler-rs/tadlib-fixture image (has cooler + numpy):
    python dump_mat.py /data /out
"""
import sys

import cooler
import numpy as np

MCOOL = "4DNFIZ1ZVXC8.mcool"
RES = 50000
CHROM = "chr2L"


def main(data_dir, out_dir):
    uri = f"{data_dir}/{MCOOL}::resolutions/{RES}"
    mat = cooler.Cooler(uri).matrix(balance=None, sparse=False).fetch(CHROM)
    assert mat.shape[0] == mat.shape[1]
    assert not np.isnan(mat).any()
    base = f"{out_dir}/4DNFIZ1ZVXC8.{RES // 1000}kb.{CHROM}.raw.mat"
    np.savetxt(base, mat, fmt="%.6g", delimiter="\t")
    print(f"wrote {base} ({mat.shape[0]} x {mat.shape[1]})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
