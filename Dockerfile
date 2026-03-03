# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:0.10.4 AS uvbin

FROM nvcr.io/nvidia/pytorch:24.09-py3 AS develop

ARG USERNAME=dcuser
ARG UID=1000
ARG GID=1000

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    DEBIAN_FRONTEND=noninteractive

# ---- OS packages (fonts + build tools) ----
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN fc-cache -fv

# Codex CLI
RUN npm i -g @openai/codex

RUN fc-cache -fv

# ---- Python deps for your dev env ----
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install torchaudio==2.7.0

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=bind,source=.devcontainer/requirements-dev.txt,target=/tmp/requirements-dev.txt \
    python -m pip install -r /tmp/requirements-dev.txt

# ---- Build NonLinLoc from local repo ----
# If your folder name differs, adjust the COPY source path.
WORKDIR /opt
COPY external_source/NonLinLoc /opt/NonLinLoc

WORKDIR /opt/NonLinLoc/src
RUN mkdir -p bin
RUN cmake .
RUN make
RUN install -m 0755 /opt/NonLinLoc/src/bin/* /usr/local/bin/

# ---- Install LOKI from local repo ----
WORKDIR /opt
COPY external_source/loki /opt/loki
RUN pip install /opt/loki

# ---- Create user for devcontainer ----
RUN addgroup --gid $GID $USERNAME && \
    adduser --disabled-password --gecos "" --shell "/bin/bash" --uid $UID --gid $GID $USERNAME

USER ${USERNAME}

# ---- Devcontainer niceties ----
COPY ruff.toml /home/${USERNAME}/ruff.toml

ENV PYTHONPATH="${PYTHONPATH}:/workspace/src"
WORKDIR /workspace

CMD ["/bin/bash"]

