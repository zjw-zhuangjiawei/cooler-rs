#!/usr/bin/env bash
# Regenerate the arrowhead regression fixture from the committed 4DN mcool:
#
#   1. the hictk image converts tests/data/4DNFIZ1ZVXC8.mcool to a
#      single-resolution .hic (5 kb; lossless — raw counts plus the
#      KR/VC/VC_SQRT/ICE normalization vectors),
#   2. the arrowhead-fixture image (juicer_tools 2.20.00) runs
#      `arrowhead -k KR -r 5000` on it,
#   3. both the .hic and the reference bedpe land in tests/data/.
#
# -k KR is explicit: juicer_tools 2.20.00 defaults to SCALE, which this
# dataset has no vector for (passing nothing silently yields zero domains).
set -euo pipefail
repo_root=$(cd "$(dirname "$0")/.." && pwd)
source "$repo_root/.env" 2>/dev/null || true

hictk_image=${HICTK_IMAGE:-ghcr.io/paulsengroup/hictk:2.2.0}
base_image=eclipse-temurin:21-jre-jammy
[ -n "${TADLIB_REGISTRY_PREFIX:-}" ] && base_image=$TADLIB_REGISTRY_PREFIX/$base_image

# The jar download inside the build needs the host's proxy route when one is
# set: a localhost proxy is unreachable from the bridge network, so build on
# the host network and hand the proxy to the download step only (apt stays
# direct; through this proxy it returns 502).
net_args=()
jar_proxy=
[ -n "${https_proxy:-}" ] && {
  net_args=(--network=host)
  jar_proxy=$https_proxy
}

docker build -q -t cooler-rs/arrowhead-fixture:2.20.00 \
  --build-arg BASE_IMAGE=$base_image --build-arg JAR_PROXY=$jar_proxy \
  "${net_args[@]}" "$repo_root/scripts/arrowhead-fixture" >/dev/null

outdir=$(mktemp -d /tmp/arrowhead-fixture.XXXXXX)
chmod 777 "$outdir"

docker run --rm -v "$repo_root/tests/data:/data:ro" -v "$outdir:/out" \
  $hictk_image convert -f /data/4DNFIZ1ZVXC8.mcool /out/5kb.hic -r 5000 >/dev/null

docker run --rm -v "$outdir:/data" cooler-rs/arrowhead-fixture:2.20.00 \
  arrowhead -k KR -r 5000 /data/5kb.hic /data/out

cp "$outdir/5kb.hic" "$repo_root/tests/data/4DNFIZ1ZVXC8.5kb.hic"
cp "$outdir/out/5000_blocks.bedpe" \
  "$repo_root/tests/data/4DNFIZ1ZVXC8.5kb.arrowhead.bedpe"
echo "wrote tests/data/4DNFIZ1ZVXC8.5kb.hic + 4DNFIZ1ZVXC8.5kb.arrowhead.bedpe"
