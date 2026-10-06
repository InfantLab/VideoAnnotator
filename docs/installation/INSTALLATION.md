# VideoAnnotator Installation Guide

> 📖 **Navigation**: [Getting Started](../usage/GETTING_STARTED.md) | [Demo Commands](../usage/demo_commands.md) | [Pipeline Specs](../usage/pipeline_specs.md) | [Main Documentation](../README.md)

VideoAnnotator is a modern video analysis toolkit that uses AI models for comprehensive behavioral annotation. This guide covers installation using our modern **uv-based workflow** for fast, reliable dependency management.

## Prerequisites

- **Python 3.12 or 3.13** (required; 3.13 recommended. Python 3.14 isn't supported yet.)
- **Git** for cloning repositories
- **uv** package manager (fast, modern Python dependency management)
- **NVIDIA driver 560 or newer** for GPU acceleration (Linux ≥ 560.28.03, Windows ≥ 560.76: the CUDA 12.6
  level). Drivers from 525 usually work through CUDA's minor-version compatibility. No CUDA Toolkit
  install is needed: the PyTorch wheels bring their own CUDA libraries.
- **NVIDIA GPU** with CUDA support (GTX 1060 6GB+ or better recommended)

## System Requirements

### Minimum Requirements

- **CPU**: Intel Core i5 (8th gen) or AMD Ryzen 5 (2000 series) or better
- **RAM**: 16GB
- **Storage**: 20GB free space
- **GPU**: NVIDIA GTX 1060 6GB (for CUDA acceleration)
- **OS**: Windows 10/11, Ubuntu 20.04+, or macOS 11+

### Recommended Requirements

- **CPU**: Intel Core i7/i9 (11th gen+) or AMD Ryzen 7/9 (5000 series+)
- **RAM**: 32GB or more
- **Storage**: 50GB+ SSD
- **GPU**: NVIDIA RTX 3060 12GB or better
- **OS**: Windows 11, Ubuntu 22.04+, or macOS 12+

## macOS specifics (concise)

These tips address the most common macOS installation issues.

- Architecture: Apple Silicon (M1/M2/M3) is supported in CPU mode. CUDA GPU acceleration is not available on macOS.
- Prerequisites (Homebrew):
  ```bash
  # Install Homebrew if missing: https://brew.sh
  brew install libomp ffmpeg node
  ```
  - libomp: fixes OpenMP errors and segfaults in some audio models.
  - ffmpeg: required by whisper/librosa/audio pipelines.
  - node: only needed if you run Video Annotation Viewer standalone (outside this install) for development. The bundled copy at `/viewer` needs no Node/npm.
- uv in PATH:
  ```bash
  # If 'uv' is not found after install
  export PATH="$HOME/.local/bin:$PATH"
  ```
- Shell config permissions: If `~/.zshrc` or `~/.config` are owned by root (from earlier sudo usage), fix ownership before writing PATH updates:
  ```bash
  sudo chown -R "$USER" ~/.zshrc ~/.config
  chmod u+w ~/.zshrc
  # If .zshrc is missing, write PATH to ~/.zprofile instead for login shells
  ```
- Hugging Face token (for speaker diarization): set `HUGGINGFACE_TOKEN` in a `.env` file or your shell. See Environment Setup guide.
- Viewer: available at `/viewer` on the running VideoAnnotator server, no separate setup. For
  standalone development of the viewer repo itself, use `npm run dev` (not `npm start`).

## Quick Start (Recommended)

### 1. Install uv Package Manager

```bash
# Install uv (fast, modern Python package manager)
# Linux/Mac:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell):
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and Choose Your Install

```bash
# Clone the repository
git clone https://github.com/your-org/VideoAnnotator.git
cd VideoAnnotator
```

As of v1.5.0, `uv sync` / `pip install .` on its own installs only the **core**
package (API server, CLI, storage, exporters) — no pipeline, no torch, no
GPU downloads. Pipelines are opt-in via `[project.optional-dependencies]`
extras groups, so you only download what you actually need. Pick the
install that matches what you're doing:

| I want to...                          | Install command                                | What you get |
| -------------------------------------- | ----------------------------------------------- | ------------ |
| Run only scene labelling               | `uv sync --extra scene`                         | torch, open-clip-torch, PySceneDetect — no face/audio/person deps |
| Run only person tracking               | `uv sync --extra person`                        | torch, ultralytics (YOLO11), supervision |
| Run only face analysis (DeepFace)      | `uv sync --extra face`                          | deepface, imutils — **no torch** (the one torch-free pipeline family) |
| Run only speech/diarization            | `uv sync --extra audio`                         | torch, torchaudio, librosa, openai-whisper, pyannote.audio |
| Mix a few families                     | `uv sync --extra scene --extra person`          | union of the groups listed |
| Run a slim API server (no local pipelines) | `uv sync`                                   | core only — useful if pipelines run on separate workers/nodes |
| Reproduce the old "everything installed" behaviour | `uv sync --all-extras` (or `uv sync --extra all`) | every pipeline family |

Available extras groups: `face`, `face-openface3`, `audio`, `scene`, `person`,
`llm`, plus the meta-group `all`. `face-openface3` is **not** included by a plain
`--extra face`: it's a separate, deliberately opt-in group (see "OpenFace 3"
below). `videoannotator pipelines
--all` shows every pipeline the registry knows about, including ones your
current install doesn't have the extras for (each with an install hint).

```bash
# Example: a researcher who only needs scene detection
uv sync --extra scene

# Example: everything, matching pre-v1.5.0 behaviour
uv sync --all-extras

# Development tools (ruff, mypy, pytest, pre-commit) come with any `uv sync` by default
# (dependency group `dev`); `--no-dev` leaves them out.

# Initialize the local SQLite database (creates tables + admin API key)
uv run videoannotator setup-db --admin-email you@example.com --admin-username you
```

> The `setup-db` command is idempotent. Re-run it after pulling new schema changes or use `--force` when you want to drop and recreate tables. Pass `--skip-admin` if you prefer to manage API keys yourself later with `videoannotator generate-token`.

#### OpenFace 3 (separate opt-in extras group)

`face-openface3` (landmarks, action units, gaze) is separate from `face` because it
pulls in a different, heavier dependency set (`openface-test`, torch). It's included
in `--all-extras`/`--extra all`, but not in a plain `--extra face`. **OpenFace 3.0 is
licensed for academic or non-profit, non-commercial research use only.**

The LAION pipelines (`face_laion_clip`, `laion_voice`) and their extras groups
(`face-laion`, `audio-laion`) were removed in v1.6.0; a job naming one gets a message
saying so and what to use instead.

### Upgrading from v1.4.x

v1.4.x installed every pipeline by default. If your config references
`face_openface3_embedding` and you install anything less than `--all-extras`, job
submission returns a clear message rather than a crash:

```
Error: pipeline 'face_openface3_embedding' is not available in this install.
As of v1.5.0, pipelines requiring the 'face-openface3' extras group are no longer installed by default.
Install it with: pip install videoannotator[face-openface3]
```

Run `uv sync --all-extras` to restore full v1.4.x-equivalent behaviour, or
add just the extras group named in the message.

### 3. Install CUDA-enabled PyTorch (GPU acceleration)

```bash
# `uv sync` installs torch 2.11 from the CUDA 12.6 wheel index on Linux (see
# [tool.uv.sources] in pyproject.toml); macOS and Windows get PyPI's builds.
# torch stays at 2.11 because pyannote.audio 4 needs torchaudio, discontinued at 2.11.
uv sync --all-extras
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 4. Install Native Dependencies (if needed)

Some dependencies like `dlib` require system-level installation:

```bash
# If using conda for native dependencies:
conda install -n videoannotator cmake dlib -c conda-forge

# Otherwise, install cmake system-wide from cmake.org
```

### 2. Install OpenFace 3.0 (Manual Installation Required)

OpenFace 3.0 requires manual compilation. Follow these steps:

#### Windows Installation

```bash
# Install additional dependencies
conda install -c conda-forge boost
conda install -c conda-forge openblas

# Clone OpenFace 3.0
git clone https://github.com/CMU-MultiComp-Lab/OpenFace-3.0.git
cd OpenFace-3.0

# Build OpenFace
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release

# Install Python bindings
cd ../python
python setup.py install
```

#### Linux/Mac Installation

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libopenblas-dev liblapack-dev
sudo apt-get install libboost-all-dev

# Clone and build OpenFace 3.0
git clone https://github.com/CMU-MultiComp-Lab/OpenFace-3.0.git
cd OpenFace-3.0
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Release ..
make -j4

# Install Python bindings
cd ../python
python setup.py install
```

## Where model weights are stored

Pipelines download their models on first use (the viewer's pipeline cards say how much). Since
v1.6.0 every pipeline keeps them in **one directory**:

| Platform | Default |
| --- | --- |
| Linux | `~/.local/share/videoannotator/models` (or `$XDG_DATA_HOME/videoannotator/models`) |
| macOS | `~/Library/Application Support/videoannotator/models` |
| Windows | `%LOCALAPPDATA%\videoannotator\models` |
| Dev container, Docker images | `/app/models` or `<repo>/models` (a mounted directory or volume) |

Set `VIDEOANNOTATOR_MODELS_DIR` to put them somewhere else (a shared data disk, for example).
Inside it there's one folder per source: `huggingface/`, `pyannote/`, `torch/`, `deepface/`,
`whisper/`, `yolo/`. VideoAnnotator points the libraries' own variables (`HF_HUB_CACHE`,
`TORCH_HOME`, `PYANNOTE_CACHE`, `DEEPFACE_HOME`) there unless you've set them yourself; it never
changes `HF_HOME`, so a `huggingface-cli login` token stays where it is.

`uv run videoannotator diagnose models` shows the directory, its size per source, and any weights
left in the pre-v1.6.0 locations (`~/.cache/huggingface`, `~/.cache/whisper`, `~/.deepface`,
`./models`). Moving those into the new directory saves downloading them again.

## Choosing videos on your own computer

When the viewer and VideoAnnotator run on the same computer, the new-job wizard opens on
**My folders**: open a folder, tick the videos (or **Select all**, optionally with subfolders), and
the run starts at once. Videos are read where they are. Nothing is uploaded or copied, and deleting
a job never touches the original video.

My folders shows only the folders VideoAnnotator may read:

| Install | Default |
| --- | --- |
| Plain install | your home folder |
| Docker (below) | the folder you set as `VIDEOS_DIR` |

To allow others (an external drive, a shared data disk), set `VIDEOANNOTATOR_INGEST_ROOTS` to the
folders, separated by `:` (`;` on Windows), and restart the server. Only an administrator may read
videos in place; the first user is one by default.

When VideoAnnotator runs on another computer (a lab server), its folders aren't yours, so the
wizard offers **Upload videos** instead. Uploading copies each video into VideoAnnotator's own
storage.

## Where results go

Every run's results go to one visible folder, **`~/VideoAnnotator`** by default
(`C:\Users\<you>\VideoAnnotator` on Windows): one folder per run, named after the run and its
date, then one folder per video, named after the video.

```
~/VideoAnnotator/
  BabyJokes wave 2 (2026-10-06)/
    run.json                 what ran: pipelines, settings, versions, and each video's source
    child01/
      child01_face_detections.json
      child01_speech_recognition.vtt
      ...
```

Result files keep their usual names (`<video>_<output>`). Folders are never overwritten: a second
run with the same name and date gets `(2026-10-06 2)`. These folders hold results only, never a
copy of a video. `videoannotator server` prints the folder when it starts, and the viewer shows it
on each run's and job's page, with **Open folder**, **Copy location** and, for a run,
**Download results** (one zip, without videos).

To put results somewhere else (an encrypted or backed-up drive), set `VIDEOANNOTATOR_RESULTS_DIR`,
in the environment or in `.env`, and restart. Runs from before the change stay where they were
written. Deleting a job or run in the viewer deletes its results folder.

## Verify Installation

```bash
# Confirm the CLI/API come up and list which pipelines your install has
# extras for (only the ones matching what you installed above will show up)
uv run videoannotator pipelines
uv run videoannotator pipelines --all   # also show unavailable ones + install hints

# Test the API server
uv run videoannotator server --host 0.0.0.0 --port 18011
# Should start server on http://localhost:18011
```

If you installed a torch-backed extras group (`scene`, `person`, `audio`,
`face-openface3`), you can additionally confirm the GPU/CPU
build:

```bash
uv run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
"
```

This will fail with `ModuleNotFoundError` if you installed core-only or a
non-torch extras group (e.g. `face`) — that's expected, not a bug.

## Development Commands

Once installed, use these commands for development:

```bash
# Start API server
uv run videoannotator server --host 0.0.0.0 --port 18011

# Run linting and formatting
uv run ruff check .
uv run ruff format .

# Run type checking
uv run mypy src

# Run tests
uv run pytest

# CLI help
uv run videoannotator --help

# API server (compatibility wrapper)
uv run python api_server.py
```

## Docker Installation (Alternative)

One `Dockerfile` builds the image for CPU and GPU machines. It builds **slim** by default (no
pipeline extras, no torch), matching the core-only install above; install pipelines from the viewer,
or build them in with `--build-arg EXTRAS=...`. torch's wheels bring their own CUDA, so the same image
uses an NVIDIA GPU when run with `--gpus all` (needs the NVIDIA Container Toolkit, or Docker Desktop
with WSL 2 on Windows) and runs on the CPU otherwise.

```bash
# Slim (no extras, no torch)
docker build -t videoannotator .

# One or more pipeline families
docker build --build-arg EXTRAS=scene,person -t videoannotator:scene-person .

# Every pipeline
docker build --build-arg EXTRAS=all -t videoannotator:all .

# Run on this computer: your videos read in place, results in ~/VideoAnnotator
docker run --rm -p 127.0.0.1:18011:18011 --gpus all \
  -v videoannotator-models:/app/models \
  -v "$HOME/Studies":/videos:ro -v "$HOME/VideoAnnotator":/results \
  -e VIDEOANNOTATOR_INGEST_ROOTS=/videos -e VIDEOANNOTATOR_RESULTS_DIR=/results \
  -e VIDEOANNOTATOR_PUBLISHED_LOCALLY=1 \
  -e VIDEOANNOTATOR_HOST_PATHS="/videos=$HOME/Studies;/results=$HOME/VideoAnnotator" \
  videoannotator:all
```

### Your videos and results under Docker

The compose services `videoannotator-prod` and `videoannotator-gpu` need two folders from you:

| Variable | Default | What it is |
| --- | --- | --- |
| `VIDEOS_DIR` | `./videos` | Your video folder. Mounted read-only: VideoAnnotator never changes it. |
| `RESULTS_DIR` | `~/VideoAnnotator` | Where results go, on your own computer. |

```bash
VIDEOS_DIR=~/Studies RESULTS_DIR=~/VideoAnnotator docker compose --profile prod up videoannotator-prod
```

Then open `http://127.0.0.1:18011/viewer/`. My folders shows your video folder, and run and job
pages show the results' location as your computer sees it, with **Copy location** (Docker can't
open folders on your computer, so there is no **Open folder**). Started without `VIDEOS_DIR`, the
wizard says how to set it, and uploading still works.

**This machine only.** The port is published on `127.0.0.1`, so nothing else on your network can
reach the server, and `VIDEOANNOTATOR_PUBLISHED_LOCALLY=1` tells the server that every caller is
therefore on this computer. If you publish the port more widely (for example `18011:18011`, to share
the server with a colleague), you **must** remove `VIDEOANNOTATOR_PUBLISHED_LOCALLY`: otherwise
anyone who can reach the port can read the files in your video folder. The server warns at startup
whenever the variable is set.

**File ownership (Linux).** The container runs as root, so result files in `RESULTS_DIR` belong to
root on the host. Adding `--user "$(id -u):$(id -g)"` to `docker run` makes them yours instead.

### Dev Container (VS Code)

The dev container is a complete development environment: every pipeline, the GPU if you have one,
the viewer's toolchain (Bun) and the pre-commit hooks.

**Opening it**

- **Linux and macOS**: clone the repository, open it in VS Code and run **Dev Containers: Reopen in
  Container**.
- **Windows**: run **Dev Containers: Clone Repository in Container Volume…** and give it
  `https://github.com/InfantLab/VideoAnnotator`. The code then lives on the Linux side. If you
  clone to `C:\` and reopen it in the container instead, every file read crosses the Windows–WSL
  file bridge and Defender scans each one: in our measurements collecting the tests took 3–4 minutes
  that way against 10–30 seconds from a volume. The container prints a warning at start if it finds
  itself on a Windows drive.

The first time, creating the container installs every pipeline's dependencies
(`uv sync --inexact --all-extras`): allow 5–10 minutes and about 9 GB of disk.

**Memory**

The container caps itself at 12 GB (`--memory=12g` in `.devcontainer/devcontainer.json`), so it
can't take over the machine. What it actually uses, measured on 2026-10-02:

| Doing | Peak memory |
|---|---|
| VS Code and its extensions, idle | about 2 GB |
| The full test suite | 5.4 GB |
| A job running six pipelines on the demo video | 6.5 GB |

Installing the extras fills the cap with page cache (cached files, not memory any program holds);
Linux reclaims it when something needs the room, so a full cap during an install is normal.

*On a 16 GB machine*, Docker Desktop's VM gets 8 GB by default (half your RAM), so that is the real
limit, not the 12 GB cap. Everything above fits. If a long video or several jobs at once need more,
the sign is a process ending with `Killed` (exit code 137). Raise the VM's memory in Docker Desktop
(Settings → Resources, or `memory=` in `%USERPROFILE%\.wslconfig` on Windows), and the `--memory`
value in `devcontainer.json` if the container's own cap is what was hit.

**What survives a rebuild**

The Python environment (`.venv`) and the model weights live in Docker named volumes
(`videoannotator-venv`, `videoannotator-models`, the latter shared with `docker-compose.yml`). They
survive container rebuilds and `docker system prune --volumes`, which removes only anonymous
volumes (Docker 23 and later); only `docker volume rm`, `docker volume prune --all` or resetting
Docker Desktop deletes them. A rebuild that moves Python (as the move to one `Dockerfile` did)
leaves the old `.venv` pointing at an interpreter that's gone; `uv` recreates it and the
post-create step reinstalls everything, which takes the same 5–10 minutes as the first time.
See [troubleshooting](troubleshooting.md#the-dev-container-is-slow-freezes-or-runs-out-of-memory) to
back up or wipe the volumes.

## Troubleshooting

### uv Installation Issues

1. **uv command not found**:

   ```bash
   # Restart your terminal or run:
   export PATH="$HOME/.local/bin:$PATH"
   ```

2. **Permission errors**:

   ```bash
   # On Windows, run PowerShell as Administrator
   # On Linux/Mac, ensure you have write permissions to ~/.local/
   ```

3. **Slow dependency resolution**: uv is typically very fast, but large ML dependencies can take time on first install.

### CUDA/GPU Issues

1. **CUDA not found**: Run `nvidia-smi` to check GPU detection. Update drivers if needed.

2. **PyTorch doesn't detect CUDA**:

   ```bash
   # Verify CUDA-enabled PyTorch installation
   uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

3. **Driver too old**: torch's CUDA 12.6 wheels need NVIDIA driver 560+ (525+ with CUDA's
   minor-version compatibility). `nvidia-smi` shows the driver version; update it if
   `torch.cuda.is_available()` is False on a machine with a GPU. The pipelines fall back to CPU.

### Native Dependencies

1. **dlib/cmake errors**: Install cmake system-wide or use conda for native dependencies
2. **OpenFace compilation**: Requires manual setup - see OpenFace section above
3. **Build tool errors**: Ensure you have Visual Studio Build Tools (Windows) or GCC (Linux/Mac)

## Modern Architecture

VideoAnnotator uses:

- **uv** - Fast, reliable Python package management
- **Ruff** - Unified linting and formatting (replaces Black, isort, flake8)
- **FastAPI** - Modern API framework
- **Hatchling/setuptools** - Modern build backend
- **Docker** - CPU and GPU containerization
- **Python 3.12 or 3.13**

## Dependencies Overview

| Extras group      | Tools                          | Purpose                             | Needs torch |
| ------------------ | ------------------------------ | ------------------------------------ | ----------- |
| `person`           | YOLO11, ByteTrack              | Person detection & tracking          | ✅          |
| `scene`            | PySceneDetect, OpenCLIP        | Scene segmentation & classification  | ✅          |
| `face`             | DeepFace                       | Face detection, emotion, age/gender  | ❌          |
| `face-openface3`   | OpenFace 3.0                   | Landmarks, action units, gaze        | ✅          |
| `audio`            | Whisper, pyannote.audio        | Speech transcription & diarization   | ✅          |
| *(core, always on)* | FastAPI, uvicorn, SQLAlchemy   | REST API server, job/storage state   | N/A         |

`face` is the only pipeline family that doesn't need torch at all — see the
install-matrix table above for exact `uv sync --extra ...` commands per
group. `--all-extras` installs every row.

## Next Steps

After installation:

- See `docs/usage/GETTING_STARTED.md` for usage examples
- Check `docs/development/` for development workflows
- Review `configs/` for configuration options
- Use `uv run videoannotator --help` to test the CLI
- Use `uv run videoannotator server --host 0.0.0.0 --port 18011` to start the server

## Performance Tips

- **GPU acceleration**: Install CUDA-enabled PyTorch for 10x speedup
- **Batch processing**: Process multiple videos for efficiency
- **Memory management**: Use appropriate model sizes for your GPU
- **Container deployment**: Use Docker for consistent environments
