#!/usr/bin/env bash
# Rebuild (if needed) and run the TADLib fixture generators (domaincaller
# + hitad) on the 4DNFIZ1ZVXC8 mcool, writing
# 4DNFIZ1ZVXC8.50kb.{tadlib,hitad}.* fixtures into tests/data/.
#
# Base image prefix comes from the repo .env (TADLIB_REGISTRY_PREFIX).
set -euo pipefail
cd "$(dirname "$0")/.."
repo_root="$(pwd)"

# docker build does not read .env; source it here so the build args
# and pick up TADLIB_REGISTRY_PREFIX.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

image="cooler-rs/tadlib-fixture:0.4.4"
# 完整 base 镜像名 = <前缀>/mambaorg/micromamba:latest;前缀留空则直连 Hub。
base="${TADLIB_REGISTRY_PREFIX:+$TADLIB_REGISTRY_PREFIX/}mambaorg/micromamba:latest"
# Always build (layer cache makes it cheap) so generator/Dockerfile edits stick.
docker build --build-arg BASE_IMAGE="$base" -t "$image" -f scripts/tadlib-fixture/Dockerfile scripts/tadlib-fixture

# Container user (uid 57439) cannot create files in the zjw-owned repo, so
# generate into a tmp outdir mounted at /out, then copy back.
outdir=$(mktemp -d)
chmod 777 "$outdir"  # container user (uid 57439) must create files in it
docker run --rm --entrypoint /bin/bash \
  -v "$repo_root/tests/data:/data:ro" \
  -v "$outdir:/out:rw" \
  "$image" -lc 'micromamba run -n tadlib python /opt/generate_fixture.py /data /out'
status=$?
cp "$outdir"/4DNFIZ1ZVXC8.50kb.tadlib.* "$repo_root/tests/data/" 2>/dev/null
cp "$outdir"/4DNFIZ1ZVXC8.50kb.hitad.domains "$outdir"/4DNFIZ1ZVXC8.50kb.hitad.dis "$repo_root/tests/data/" 2>/dev/null
rm -rf "$outdir"
exit $status
