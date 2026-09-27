# TADLib fixture generator for the cooler-rs DomainCaller regression tests.
#
# Follows TADLib's official install recipe
# (https://xiaotaowang.github.io/TADLib/install.html): conda + pomegranate
# 0.10.0 + networkx 1.x, using our three standard channels (main,
# conda-forge, bioconda, same set as the official defaults/bioconda/
# conda-forge: pomegranate 0.10.0 py37 prebuild lives in main (its
# conda-forge builds were pruned), pairix/cooler py37 live in bioconda.
# No compiler/Cython needed. TADLib itself comes from PyPI.
#
# BASE_IMAGE comes from the repo .env via scripts/generate_tadlib_fixture.sh,
# which prefixes mambaorg/micromamba:latest with TADLIB_REGISTRY_PREFIX
# (e.g. docker.1ms.run/) when Docker Hub is unreachable.
ARG BASE_IMAGE=mambaorg/micromamba:latest
FROM ${BASE_IMAGE}

RUN micromamba create -y -n tadlib -c anaconda -c conda-forge -c bioconda \
        python=3.7 pomegranate=0.10.0 networkx=1.11 joblib=0.13.2 \
        numpy scipy matplotlib scikit-learn cooler \
 && micromamba run -n tadlib pip install --no-cache-dir TADLib==0.4.4

COPY generate_fixture.py /opt/generate_fixture.py
WORKDIR /data
