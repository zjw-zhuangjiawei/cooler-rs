#!/usr/bin/env bash
# Regenerate the find-tads regression fixture from the committed 4DN mcool:
# the findtads-fixture image (HiCExplorer 3.7.6, the version the Rust port is
# aligned to, installed the way its docs recommend: conda via micromamba) runs
# `hicFindTADs` on the 5 kb layer for four parameter sets, and the reference
# outputs land in tests/data/ as
# 4DNFIZ1ZVXC8.5kb.findtads.<case>_{boundaries.bed,boundaries.gff,domains.bed,
# score.bedgraph,tad_score.bm}.
#
# Notes:
# - hicFindTADs reads a .mcool through cooler's '::' group-path syntax.
# - hicmatrix's cooler reader applies the bins/weight column (multiplicative)
#   by default and masks NaN-weight bins; the Rust port passes --norm weight
#   for the same behaviour.
# - conda.anaconda.org is reachable directly, so no proxy handling is needed.
set -euo pipefail
repo_root=$(cd "$(dirname "$0")/.." && pwd)
source "$repo_root/.env" 2>/dev/null || true

base_image=mambaorg/micromamba:latest
[ -n "${TADLIB_REGISTRY_PREFIX:-}" ] && base_image=$TADLIB_REGISTRY_PREFIX/$base_image

docker build -q -t cooler-rs/findtads-fixture:3.7.6 \
  --build-arg BASE_IMAGE=$base_image \
  "$repo_root/scripts/findtads-fixture" >/dev/null

outdir=$(mktemp -d /tmp/findtads-fixture.XXXXXX)
chmod 777 "$outdir"

run() { # <case> <correction> <threshold> [extra hicFindTADs args...]
  local case=$1 correction=$2 threshold=$3
  shift 3
  docker run --rm -v "$repo_root/tests/data:/data:ro" -v "$outdir:/out" \
    cooler-rs/findtads-fixture:3.7.6 \
    --matrix '/data/4DNFIZ1ZVXC8.mcool::/resolutions/5000' \
    --outPrefix "/out/$case" \
    --minDepth 60000 --maxDepth 180000 --step 20000 --minBoundaryDistance 20000 \
    --correctForMultipleTesting "$correction" --thresholdComparisons "$threshold" \
    "$@" >/dev/null
}

run none None 1.0
run fdr fdr 0.1
run bonferroni bonferroni 0.1
run fdrchr fdr 0.5 --chromosomes chr2L chr3R

for case in none fdr bonferroni fdrchr; do
  for suffix in boundaries.bed boundaries.gff domains.bed; do
    cp "$outdir/${case}_${suffix}" \
      "$repo_root/tests/data/4DNFIZ1ZVXC8.5kb.findtads.${case}_${suffix}"
  done
done

# The correction only enters after scoring, so the three full-genome cases
# share one tad_score / score pair (asserted rather than assumed); the
# subset case has its own. The big shared files are gzipped to stay under the
# repository's 500 KB fixture limit, the shared tad_score.bm split in two
gzip -9 -c "$outdir/none_score.bedgraph" \
  > "$repo_root/tests/data/4DNFIZ1ZVXC8.5kb.findtads.full_score.bedgraph.gz"
for case in fdr bonferroni; do
  cmp -s "$outdir/none_tad_score.bm" "$outdir/${case}_tad_score.bm"
  cmp -s "$outdir/none_score.bedgraph" "$outdir/${case}_score.bedgraph"
done
half=$(( ($(wc -l < "$outdir/none_tad_score.bm") + 1) / 2 ))
gzip -9 -c <(head -n $half "$outdir/none_tad_score.bm") \
  > "$repo_root/tests/data/4DNFIZ1ZVXC8.5kb.findtads.full_tad_score.bm.part0.gz"
gzip -9 -c <(tail -n +$((half + 1)) "$outdir/none_tad_score.bm") \
  > "$repo_root/tests/data/4DNFIZ1ZVXC8.5kb.findtads.full_tad_score.bm.part1.gz"
cp "$outdir/fdrchr_score.bedgraph" \
  "$repo_root/tests/data/4DNFIZ1ZVXC8.5kb.findtads.fdrchr_score.bedgraph"
gzip -9 -c "$outdir/fdrchr_tad_score.bm" \
  > "$repo_root/tests/data/4DNFIZ1ZVXC8.5kb.findtads.fdrchr_tad_score.bm.gz"
echo "wrote tests/data/4DNFIZ1ZVXC8.5kb.findtads.*"
