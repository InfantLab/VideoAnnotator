# VideoAnnotator Docker Deployment Guide

One `Dockerfile` builds VideoAnnotator for CPU and GPU machines. It is slim by default; pipelines are
added at build time with `--build-arg EXTRAS=...` or installed later from the viewer. Model weights are
never in the image: they download on first use into `/app/models`, so mount a volume there.

There is no `nvidia/cuda` base image: torch's wheels bring their own CUDA runtime and cuDNN, and the
host's driver arrives with `--gpus all`. Without `--gpus all` the same image runs on the CPU.

## Build

```bash
# Slim: core only (API, viewer, CLI), no torch
docker build -t videoannotator .

# One or more pipeline families (see the extras in pyproject.toml)
docker build --build-arg EXTRAS=scene,person -t videoannotator:scene-person .

# Every pipeline
docker build --build-arg EXTRAS=all -t videoannotator:all .
```

Sizes: see `docs/development/roadmap_v1.6.0.md` (Phase 1, "Docker image size") until they are
measured and recorded here.

## Run

```bash
docker run --rm -p 18011:18011 --gpus all \
  -v videoannotator-models:/app/models \
  -v videoannotator-database:/app/database \
  -v "${PWD}/data:/app/data" \
  -v "${PWD}/output:/app/output" \
  videoannotator:all
```

Leave out `--gpus all` on a machine without an NVIDIA GPU. The API is at http://localhost:18011, the
viewer at http://localhost:18011/viewer.

### Mounts

| Path | Holds | Mount |
|---|---|---|
| `/app/models` | Model weights (`VIDEOANNOTATOR_MODELS_DIR`) | A named volume, so weights download once |
| `/app/database` | Jobs, users and API keys (`VIDEOANNOTATOR_DB_PATH`) | A named volume, so they survive a new container |
| `/app/data` | Input videos | Your video folder |
| `/app/output` | Exported results | A host folder |
| `/app/logs` | Logs (`VIDEOANNOTATOR_LOG_DIR`) | Optional |

### Windows PowerShell

```powershell
docker run --rm -p 18011:18011 --gpus all -v videoannotator-models:/app/models -v "${PWD}\data:/app/data" videoannotator:all
```

## Docker Compose

```bash
docker compose --profile gpu up videoannotator-gpu    # GPU
docker compose --profile prod up videoannotator-prod  # CPU
```

Both build the slim image and keep models and the database in named volumes
(`videoannotator-models`, `videoannotator-database`).

## Prerequisites for a GPU

- **Linux**: the NVIDIA driver and the NVIDIA Container Toolkit:

  ```bash
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo systemctl restart docker
  ```

- **Windows**: Docker Desktop with the WSL 2 backend and a current NVIDIA driver.
- The driver must support CUDA 12.6: NVIDIA driver 560 or newer.

Check GPU access:

```bash
docker run --gpus all --rm videoannotator:all python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Troubleshooting

- **No GPU detected**: run with `--gpus all`; check the driver version (560+) and, on Linux, the
  NVIDIA Container Toolkit.
- **Models download on every run**: mount a volume at `/app/models`.
- **Jobs or API keys gone after recreating the container**: mount a volume at `/app/database`.
- **Permission denied on a mounted folder**: check the host folder's permissions.

## What's in the image

- Ubuntu 24.04, Python 3.13 (installed by uv), ffmpeg
- VideoAnnotator and the extras you chose, installed from `uv.lock` without the dev tools
- The server starts with `videoannotator server --host 0.0.0.0 --port 18011`

## The dev container

`.devcontainer/devcontainer.json` builds the same `Dockerfile` with `SKIP_IMAGE_UV_SYNC=true`; its
`postCreateCommand` installs the environment into a `.venv` volume instead. See
`docs/installation/INSTALLATION.md`, "Dev Container (VS Code)".
