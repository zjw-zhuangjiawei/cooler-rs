# juicer_tools image for regenerating the arrowhead fixture: reads a .hic and
# writes the reference `arrowhead` output. juicer_tools 2.20.00 is the release
# the Rust port is aligned to (first-line reference in .gitignore's
# `/reference` note); the jar is downloaded at build time from the pinned
# GitHub release instead of vendoring a 37 MB binary.
#
# JAR_PROXY (set by the wrapper from the host's https_proxy) is applied to the
# download only — apt stays direct. The wrapper builds with --network=host
# when a proxy is set, because a localhost proxy is unreachable from the
# bridge network.
ARG BASE_IMAGE=eclipse-temurin:21-jre-jammy
FROM ${BASE_IMAGE}

ARG JUICEBOX_VERSION=2.20.00
ARG JAR_PROXY=
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && env https_proxy=${JAR_PROXY} curl -fsSL -o /usr/local/lib/juicer_tools.jar \
      "https://github.com/aidenlab/Juicebox/releases/download/v${JUICEBOX_VERSION}/juicer_tools.${JUICEBOX_VERSION}.jar"

WORKDIR /data
ENTRYPOINT ["java", "-Xmx8g", "-jar", "/usr/local/lib/juicer_tools.jar"]
