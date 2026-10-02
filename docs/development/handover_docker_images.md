# Handover: build, check and measure the Docker image

**For**: an agent with Docker on the host machine (outside the dev container).
**From**: the agent that rewrote the Docker setup on `1.6-dev`, 2026-10-02. It had no Docker, so
nothing below has been built yet.
**Ask Caspar before**: pushing anything, deleting images or volumes you didn't create, or changing
host settings (`.wslconfig`, Docker Desktop).

## Why

Three Dockerfiles (`Dockerfile.cpu`, `Dockerfile.gpu`, `Dockerfile.dev`) were replaced by one
`Dockerfile` on `ubuntu:24.04`. The changes:

- No `nvidia/cuda` base. It was a 2024-08 snapshot never rebuilt. torch's wheels bring their own
  CUDA 12.6 runtime and cuDNN, and `--gpus all` brings the driver.
- The CPU image used to replace torch with 2.6.0 CPU wheels after installing the extras. That was a
  leftover from before the torch 2.11 upgrade, and it broke pyannote.audio 4. That step is gone.
- The GPU image used to skip installing VideoAnnotator itself (`--no-install-project`). It now
  installs it.
- The dev tools (`dev` group) and uv's download cache used to ship in the images. Both are now left
  out.
- The image starts with `videoannotator server --host 0.0.0.0 --port 18011`.
- One image serves CPU and GPU machines. A CPU-wheel variant would be about 3.6 GB smaller, but it
  is blocked: TensorFlow (DeepFace) followed by open_clip segfaults on CPU torch wheels.

The roadmap item this closes is v1.6.0 Phase 1, "Docker image size". v1.5.0 set a target of an 80%
smaller default image than v1.4.3 (about 30 GB) but never measured it.

## Before you start

1. **Machine**: Docker Desktop with the WSL 2 backend (Windows), or Docker Engine plus the NVIDIA
   Container Toolkit (Linux). The NVIDIA driver must be 560 or newer (`nvidia-smi`).
2. **Where**: a terminal in a WSL distro or on Linux. Clone onto the Linux filesystem (`~/...`), not
   under `/mnt/c`: the build context is slow over the Windows file bridge.
3. **Disk**: at least 60 GB free for Docker (`docker system df`). Add 30 GB if you do step 6.
4. **Memory**: this host froze once from memory exhaustion (see
   `handover_windows_devcontainer_freeze.md`). Image builds aren't covered by the dev container's
   12 GB cap: installing the extras alone filled 10 GB with page cache. Ask Caspar to close heavy
   apps (browsers, other containers). Run one build or container at a time. Stop the dev container
   if it is running.
5. **Hugging Face token**: speaker diarization downloads a gated pyannote model. You need a token
   whose account has accepted the model's licence. Ask Caspar for it and set
   `export HUGGINGFACE_TOKEN=...`. Never write it into a file or a log you report.

## Steps

```bash
git clone -b 1.6-dev https://github.com/InfantLab/VideoAnnotator ~/va-images
cd ~/va-images && git log --oneline -1    # report this commit
```

### 1. Build

Record the wall time of each build.

```bash
time docker build -t va-check:slim .
time docker build --build-arg EXTRAS=all -t va-check:all .
time docker build --build-arg SKIP_IMAGE_UV_SYNC=true -t va-check:devcontainer .
docker images va-check --format '{{.Tag}}\t{{.Size}}'
```

If a build fails, stop and report the last 40 lines of its output.

### 2. Slim image: the server starts and reports what's missing

```bash
docker run -d --name va-slim -p 18011:18011 -e AUTH_REQUIRED=false va-check:slim
sleep 20; curl -s localhost:18011/health; echo
curl -s localhost:18011/api/v1/pipelines/ | python3 -c "import json,sys; [print(p['name'], p.get('available'), p.get('install_hint','')) for p in json.load(sys.stdin)['pipelines']]"
docker logs va-slim 2>&1 | grep -E "Logs:|Database:|ERROR" | head
docker rm -f va-slim
```

Expected:
- `/health` says healthy.
- Every pipeline shows `False`, with an install hint.
- The log shows `Logs: /app/logs` and `Database: /app/database/videoannotator.db`.

### 3. Full image on the GPU: libraries and imports

```bash
docker run --rm --gpus all va-check:all python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
import tensorflow as tf
print("tensorflow GPUs:", tf.config.list_physical_devices("GPU"))
with tf.device("/GPU:0"):
    tf.nn.conv2d(tf.ones((1, 8, 8, 3)), tf.ones((3, 3, 3, 4)), 1, "SAME")
libs = sorted({l.split()[-1] for l in open("/proc/self/maps") if ".so" in l and any(k in l for k in ("cuda", "cudnn", "cublas", "cusolver", "cusparse", "nvJitLink"))})
print("\n".join(libs))
import open_clip, pyannote.audio, whisper, ultralytics
print("imports ok (tensorflow then open_clip in one process)")
EOF
```

Expected:
- torch reports `cuda True`.
- TensorFlow lists one GPU and the conv2d runs.
- Every CUDA library path is under `/app/.venv/.../nvidia/`. No library path comes from
  `/usr/local/cuda`, because that folder no longer exists.
- The script reaches "imports ok".

If TensorFlow finds no GPU, report it. DeepFace would then silently run on the CPU. That isn't fatal,
but it decides whether we need a fallback.

### 4. Full image: a real job with every local pipeline

```bash
docker volume create va-check-models
docker run -d --name va-all --gpus all -p 18011:18011 -e AUTH_REQUIRED=false \
  -e HUGGINGFACE_TOKEN -v va-check-models:/app/models va-check:all
sleep 30
VIDEO=viewer/demo-assets/2UWdXP.joke1.rep3.take1.Peekaboo_h265.mp4
JOB=$(curl -s -X POST localhost:18011/api/v1/jobs/ -F "video=@$VIDEO" \
  -F "selected_pipelines=face_analysis,face_openface3_embedding,person_tracking,scene_detection,speaker_diarization,speech_recognition" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
while :; do S=$(curl -s localhost:18011/api/v1/jobs/$JOB | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])"); echo "$S"; case $S in completed|failed|cancelled) break;; esac; sleep 15; done
curl -s localhost:18011/api/v1/jobs/$JOB/results | python3 -c "import json,sys; [print(k, v['status'], v.get('annotation_count'), v.get('error_message')) for k, v in json.load(sys.stdin)['pipeline_results'].items()]"
docker stats --no-stream va-all
docker exec va-all ls /app/database /app/logs
docker rm -f va-all
```

The first run downloads the weights (a few GB), so allow 10–20 minutes. Expected annotation
counts, from the same video in the dev container on 2026-10-02:

| Pipeline | Count |
|---|---|
| face_analysis | 0 (DeepFace finds no faces in this clip; not an error) |
| face_openface3_embedding | 1 |
| person_tracking | 22 |
| scene_detection | 1 |
| speaker_diarization | 4 |
| speech_recognition | 1 |

Every pipeline should say `completed`. Report the `docker stats` memory figure. The dev-container
peak for this job was about 6.5 GB, including VS Code.

### 5. Full image without a GPU

Repeat step 4 without `--gpus all`, keeping the same models volume so nothing downloads again.
Everything should still complete, more slowly. Report the wall time.

### 6. Optional: the v1.4.3 baseline

This measures the "~30 GB" image that v1.5.0's target was set against. It's long, and needs about
30 GB of disk.

```bash
git -C ~/va-images worktree add ~/va-v143 v1.4.3 && cd ~/va-v143
time docker build -f Dockerfile.gpu --build-arg SKIP_IMAGE_UV_SYNC=false --build-arg SKIP_TORCH_INSTALL=false -t va-check:v143 .
docker images va-check:v143 --format '{{.Size}}'
cd ~/va-images && git worktree remove --force ~/va-v143
```

### 7. Compose file

```bash
docker compose config -q && echo "compose ok"
```

### 8. Clean up

```bash
docker rmi va-check:slim va-check:all va-check:devcontainer va-check:v143 2>/dev/null
docker volume rm va-check-models
docker builder prune -f    # the build cache, several GB
```

## Report back to Caspar

Fill this in and send it, with any failure output (no tokens):

| | Result |
|---|---|
| Commit | |
| `slim` size / build time | |
| `all` size / build time | |
| `devcontainer` size / build time | |
| v1.4.3 size (optional) | |
| Step 2: health, pipelines unavailable with hints, log/database paths | |
| Step 3: torch CUDA, TensorFlow GPU, library paths, imports | |
| Step 4: per-pipeline status and counts, wall time, peak memory | |
| Step 5: CPU-only job status, wall time | |
| Step 7: compose | |
| Anything else that looked wrong | |

The agent in the dev container will record the sizes in the roadmap and CHANGELOG and fix anything
that failed.
