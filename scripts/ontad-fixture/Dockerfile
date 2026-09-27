# OnTAD fixture builder for the cooler-rs OnTAD regression tests.
#
# Compiles the original C++ OnTAD v1.4 so its `.tad` output can serve as the
# reference for our Rust port. Source is cloned from upstream at build time,
# pinned to commit dc36df9 (2021-06-30, same source as OnTAD's own
# docker/Dockerfile). The makefile's default target also runs a demo on its
# own test data, so this Dockerfile invokes g++ directly with the makefile's
# exact flags instead.
#
# BASE_IMAGE comes from the repo .env via scripts/generate_ontad_fixture.sh,
# which prefixes ubuntu:22.04 with TADLIB_REGISTRY_PREFIX when Docker Hub is
# unreachable. Only the base image goes through the registry prefix; the
# GitHub clone uses direct network access.
ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ARG ONTAD_COMMIT=dc36df93598bd2dce92fe6e1e97425a60c73178b

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        g++ git ca-certificates libcurl4-openssl-dev zlib1g-dev \
 && rm -rf /var/lib/apt/lists/* \
 && git clone https://github.com/anlin00007/OnTAD.git /opt/OnTAD \
 && cd /opt/OnTAD && git checkout --detach "$ONTAD_COMMIT" && rm -rf .git \
 && cd src \
 && g++ -std=c++11 main.cpp step1.cpp step2.cpp step3.cpp step4.cpp \
        common.cpp straw.cpp -lm -lcurl -lz -o /usr/local/bin/ontad \
 && rm -rf /opt/OnTAD

WORKDIR /data
ENTRYPOINT ["/usr/local/bin/ontad"]
