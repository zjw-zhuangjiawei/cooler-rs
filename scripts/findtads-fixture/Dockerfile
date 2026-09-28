# hicexplorer image for regenerating the findtads fixtures: runs `hicFindTADs`
# on the 4DNFIZ1ZVXC8 matrix. HiCExplorer's official installation route is
# conda (bioconda channel; pip is discontinued), so this image is built with
# micromamba. 3.7.6 is the version the Rust port is aligned to. conda.anaconda.org
# is reachable directly from here, so no proxy handling is needed.
ARG BASE_IMAGE=mambaorg/micromamba:latest
FROM ${BASE_IMAGE}

RUN micromamba create -y -n hicx -c bioconda -c conda-forge \
      'hicexplorer=3.7.6'

ENV ENV_NAME=hicx
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "hicFindTADs"]
