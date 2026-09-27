#!/usr/bin/env bash
# Generate the OnTAD regression fixtures from the 4DNFIZ1ZVXC8 mcool:
#   1. dump chr2L raw 50kb contact matrix as a dense `.mat` text file
#      (cooler, inside the existing tadlib-fixture image),
#   2. run the original C++ OnTAD v1.4 (compiled in the ontad-fixture
#      image) with default parameters on that `.mat`,
# writing 4DNFIZ1ZVXC8.50kb.chr2L.raw.{mat,tad} into tests/data/.
#
# Base image prefix comes from the repo .env (TADLIB_REGISTRY_PREFIX).
set -euo pipefail
cd "$(dirname "$0")/.."
repo_root="$(pwd)"

# docker build does not read .env; source it here so the build arg
# picks up TADLIB_REGISTRY_PREFIX.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

# Step 1: .mat dump (cooler lives in the tadlib-fixture image).
tadlib_image="cooler-rs/tadlib-fixture:0.4.4"
tadlib_base="${TADLIB_REGISTRY_PREFIX:+$TADLIB_REGISTRY_PREFIX/}mambaorg/micromamba:latest"
docker build --build-arg BASE_IMAGE="$tadlib_base" -t "$tadlib_image" \
  -f scripts/tadlib-fixture/Dockerfile scripts/tadlib-fixture

# Step 2: OnTAD compiler image (plain ubuntu + g++; no python needed).
ontad_image="cooler-rs/ontad-fixture:1.4"
ontad_base="${TADLIB_REGISTRY_PREFIX:+$TADLIB_REGISTRY_PREFIX/}ubuntu:22.04"
docker build --build-arg BASE_IMAGE="$ontad_base" -t "$ontad_image" \
  -f scripts/ontad-fixture/Dockerfile scripts/ontad-fixture

# Container users cannot create files in the zjw-owned repo, so
# generate into a tmp outdir mounted at /out, then copy back.
outdir=$(mktemp -d)
chmod 777 "$outdir"
docker run --rm --entrypoint /bin/bash \
  -v "$repo_root/tests/data:/data:ro" \
  -v "$outdir:/out:rw" \
  -v "$repo_root/scripts/ontad-fixture/dump_mat.py:/opt/dump_mat.py:ro" \
  "$tadlib_image" -lc 'micromamba run -n tadlib python /opt/dump_mat.py /data /out'

# Default parameters, matching Params::default() in the Rust port.
docker run --rm \
  -v "$outdir:/out:rw" \
  "$ontad_image" /out/4DNFIZ1ZVXC8.50kb.chr2L.raw.mat -o /out/4DNFIZ1ZVXC8.50kb.chr2L.raw

cp "$outdir"/4DNFIZ1ZVXC8.50kb.chr2L.raw.tad "$repo_root/tests/data/"
gzip -9 -c "$outdir"/4DNFIZ1ZVXC8.50kb.chr2L.raw.mat \
  > "$repo_root/tests/data/4DNFIZ1ZVXC8.50kb.chr2L.raw.mat.gz"
status=$?
rm -rf "$outdir"
exit $status
