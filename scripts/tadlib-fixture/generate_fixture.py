#!/usr/bin/env python3
"""Regenerate the cooler-rs DomainCaller regression fixtures.

Runs the TADLib `domaincaller` CLI pipeline end to end on the 4DN dataset
bind-mounted at /data: the whole genome is trained jointly (one 4-state
GMM-HMM for all chromosomes) and TADs are called per chromosome, exactly as
tadlib/scripts/domaincaller does (Genome.learning + callDomains +
outputDomain). Writes the DI bedGraph and the domain list next to the input.

Usage: generate_fixture.py /data [chromns? no — follows the CLI]
"""
import fractions
import math
import os
import sys
import types

# networkx 1.x uses fractions.gcd (removed on Python 3.11; harmless here).
fractions.gcd = math.gcd

# chromLev.py imports tadlib.calfea.analyze (sklearn/scipy.stats.itemfreq)
# but never calls it on this path; stub it so the import succeeds.
stub = types.ModuleType("tadlib.calfea.analyze")
stub.Core = stub.manipulation = None
sys.modules["tadlib.calfea.analyze"] = stub

from tadlib.domaincaller.genomeLev import Genome  # noqa: E402
from tadlib.hitad.genomeLev import Genome as HitadGenome  # noqa: E402


def run_domaincaller(data_dir, out_dir):
    mcool = f"{data_dir}/4DNFIZ1ZVXC8.mcool"
    res = 50000  # no 40 kb layer in this mcool; 50 kb is the closest
    uri = f"{mcool}::resolutions/{res}"
    base = f"{out_dir}/4DNFIZ1ZVXC8.{res // 1000}kb"
    dis_out = base + ".tadlib.dis"
    dom_out = base + ".tadlib.domains"

    # The CLI's default balance column is 'weight'; fall back to the raw
    # matrix when the file has no weight column. Cache defaults to the
    # container's tmpdir: genomeLev makedirs()es the unexpanded cache arg
    # (writes a literal '~' dir), so an explicit path must be absolute.
    # chr4 (27 bins) is excluded: TADLib's domaincaller window is 40 bins
    # (defaultwindow//res without a small-chromosome guard) and crashes on
    # anything shorter; chrY/M are the CLI defaults.
    exclude = ["chr4", "chrY", "chrM"]
    try:
        G = Genome(uri, balance_type="weight",
                   exclude=exclude, DIout=dis_out)
    except KeyError:
        G = Genome(uri, balance_type="RAW",
                   exclude=exclude, DIout=dis_out)

    G.learning()
    G.callDomains()
    G.outputDomain(dom_out)
    G.wipeDisk()

    print(f"wrote {dis_out}")
    print(f"wrote {dom_out} ({len(G.Results)} domains)")


def run_hitad(data_dir, out_dir):
    """Run the hitad pipeline on a writable copy of the mcool.

    The CLI's DI write-back (`bins/DIs` column) needs a writable cool, so
    the mcool is copied to the output mount first; the DIs column is then
    dumped as a bedGraph fixture (the Rust port keeps DIs in memory and
    exposes them through Genome::di_track, so no write-back there).
    exclude is ['chr4', 'chrY', 'chrM'] — chr4 is excluded as well because
    TADLib 0.4.4 + pomegranate 0.10 viterbi (probability space, not log
    space) underflows with the HMM trained on chr4's DI segments and then
    returns None for EVERY chromosome (and the multiprocessing task queue
    then deadlocks); without chr4 the shared HMM matches the domaincaller
    one and the reference runs cleanly.
    """
    import shutil

    import cooler

    mcool = f"{data_dir}/4DNFIZ1ZVXC8.mcool"
    res = 50000
    copy = f"{out_dir}/4DNFIZ1ZVXC8.{res // 1000}kb.hitad.mcool"
    shutil.copy(mcool, copy)
    datasets = {res: {"rep1": f"{copy}::/resolutions/{res}"}}
    base = f"{out_dir}/4DNFIZ1ZVXC8.{res // 1000}kb"
    dom_out = base + ".hitad.domains"
    dis_out = base + ".hitad.dis"

    G = HitadGenome(datasets, balance_type="weight", maxsize=4000000,
                    cache=f"{out_dir}/hitad-cache",
                    exclude=["chr4", "chrY", "chrM"],
                    DIcol="DIs", min_chrom_size=1000000)
    G.learning()
    G.callHierDomain()
    G.outputDomain(dom_out)
    # remove the container-owned pickle cache in-container (the host cannot)
    G.wipeDisk()

    lib = cooler.Cooler(datasets[res]["rep1"])
    with open(dis_out, "w") as out:
        for chrom in lib.chromnames:
            size = lib.chromsizes[chrom]
            dis = lib.bins().fetch(chrom)["DIs"].values
            for i, v in enumerate(dis):
                end = min((i + 1) * res, size)
                out.write("{0}\t{1}\t{2}\t{3:.4g}\n".format(chrom, i * res, end, v))

    print(f"wrote {dom_out} ({len(G.Results)} domains)")
    print(f"wrote {dis_out}")


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/out"
    run_domaincaller(data_dir, out_dir)
    run_hitad(data_dir, out_dir)


if __name__ == "__main__":
    main()
