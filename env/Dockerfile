FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV MAMBA_EXE=/usr/local/bin/micromamba
ENV CONDA_DEFAULT_ENV=SpatialPPIv2
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/opt/conda/envs/SpatialPPIv2/bin:/opt/conda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bzip2 \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

COPY env/environment.yaml /tmp/environment.yaml

RUN mkdir -p "${MAMBA_ROOT_PREFIX}" \
    && micromamba create -y -n SpatialPPIv2 -f /tmp/environment.yaml \
    && micromamba clean -a -y \
    && rm -f /tmp/environment.yaml

RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -e' \
    'if [ "$#" -eq 0 ]; then' \
    '  exec /bin/bash' \
    'else' \
    '  exec micromamba run -n SpatialPPIv2 "$@"' \
    'fi' \
    > /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
