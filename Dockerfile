# VideoAnnotator image: one Dockerfile for CPU and GPU machines, and the dev container.
#
# Model weights are not in the image; they download on first use into /app/models
# (mount a volume there). Pipelines missing from the image can be installed from the
# viewer at runtime.
#
# No nvidia/cuda base: torch's Linux wheels bring their own CUDA runtime and cuDNN, and
# the host's driver arrives with `--gpus all`. The same image runs on CPU-only machines.
# A CPU-wheel variant would be ~3.6 GB smaller, but TensorFlow (DeepFace, face_analysis)
# and open_clip segfault together on CPU torch wheels (2026-10-02), so it waits for
# face_analysis's Phase 5 replacement.
#
#   docker build -t videoannotator .                              # slim: core, no torch
#   docker build --build-arg EXTRAS=all -t videoannotator:all .   # every pipeline
#   docker build --build-arg EXTRAS=scene,person -t videoannotator:scene-person .
#   docker run --gpus all -p 18011:18011 -v videoannotator-models:/app/models videoannotator:all

FROM ubuntu:24.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

# ffmpeg: audio extraction and torchcodec's shared libraries (pyannote.audio 4).
# libgl1/libglib: OpenCV. git, git-lfs: the dev container.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git git-lfs ffmpeg \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 locales \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system

COPY --from=ghcr.io/astral-sh/uv:0.12.22 /uv /uvx /usr/local/bin/

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python

WORKDIR /app

# true for the dev container, whose postCreateCommand syncs into its own .venv volume.
ARG SKIP_IMAGE_UV_SYNC=false
# Extras group(s) to install, e.g. "scene", "scene,person" or "all". Empty: slim image.
ARG EXTRAS=""

COPY pyproject.toml uv.lock .python-version README.md ./
# Ubuntu 24.04 ships Python 3.12; this installs the .python-version one (3.13).
RUN uv python install

COPY src/ ./src/
COPY configs/ ./configs/

# Non-editable, without the dev group. The cache mount keeps uv's download cache
# (several GB with extras) out of the image.
# hadolint ignore=SC2086
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$SKIP_IMAGE_UV_SYNC" = "true" ]; then \
        echo "[BUILD] Skipping uv sync (SKIP_IMAGE_UV_SYNC=true)"; \
    else \
        FLAGS=""; \
        for e in ${EXTRAS//,/ }; do FLAGS="$FLAGS --extra $e"; done; \
        uv sync --frozen --no-dev --no-editable $FLAGS; \
    fi

# UV_NO_SYNC: the environment is built above; don't re-sync (and pull the dev group)
# at start. NVIDIA_*: what the nvidia/cuda base used to set; the NVIDIA runtime reads
# them with --gpus.
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    UV_NO_SYNC=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    VIDEOANNOTATOR_MODELS_DIR=/app/models \
    VIDEOANNOTATOR_LOG_DIR=/app/logs \
    VIDEOANNOTATOR_DB_PATH=/app/database/videoannotator.db \
    STORAGE_ROOT=/app/storage/jobs

RUN mkdir -p /app/data /app/output /app/logs /app/database /app/models

EXPOSE 18011

CMD ["videoannotator", "server", "--host", "0.0.0.0", "--port", "18011"]
