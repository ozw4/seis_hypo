# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:0.10.4 AS uvbin

FROM nvcr.io/nvidia/pytorch:24.09-py3 AS develop

ARG USERNAME=dcuser
ARG UID=1000
ARG GID=1000
ARG CODEX_VERSION=latest

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    DEBIAN_FRONTEND=noninteractive

RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections

RUN --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    git \
    git-lfs \
    fontconfig \
    ttf-mscorefonts-installer \
    build-essential \
    cmake \
    && git lfs install --system \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm i -g @openai/codex \
    && codex --version \
    && node --version \
    && npm --version \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN fc-cache -fv

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install torchaudio==2.7.0

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=bind,source=.devcontainer/requirements-dev.txt,target=/tmp/requirements-dev.txt \
    python -m pip install -r /tmp/requirements-dev.txt

WORKDIR /opt
COPY external_source/NonLinLoc /opt/NonLinLoc

WORKDIR /opt/NonLinLoc/src
RUN mkdir -p bin
RUN cmake .
RUN make
RUN install -m 0755 /opt/NonLinLoc/src/bin/* /usr/local/bin/

WORKDIR /opt
COPY external_source/loki /opt/loki
RUN pip install /opt/loki

RUN addgroup --gid ${GID} ${USERNAME} \
    && adduser --disabled-password --gecos "" --shell "/bin/bash" --uid ${UID} --gid ${GID} ${USERNAME} \
    && mkdir -p /home/${USERNAME}/.codex \
    && chown -R ${UID}:${GID} /home/${USERNAME}

ENV CODEX_HOME=/home/${USERNAME}/.codex
ENV PYTHONPATH="${PYTHONPATH}:/workspace/src"

USER ${USERNAME}

COPY --chown=${UID}:${GID} ruff.toml /home/${USERNAME}/ruff.toml

WORKDIR /workspace

CMD ["/bin/bash"]
