# Implementation Plan · Headless Engine & API (`VideoAnnotator`)

**Goal:** Keep `VideoAnnotator` as the headless engine. Add a minimal, robust HTTP API + job queue so a web UI can create batches, monitor progress, and open finished artifacts in the Viewer.

---

## 1) High-Level Deliverables

* **FastAPI service** exposing dataset, pipeline, job, and artifact endpoints.
* **Worker(s)** executing long-running annotation jobs (GPU).
* **Job queue** (Redis + RQ, or Celery) with persistence and logs.
* **Artifact Manifest** per job for the Viewer to consume.
* **Presets & Manifests** (versioned YAML/JSON) for reproducibility.
* **Docker & Compose** for `api`, `worker`, `redis`, volumes.
* **Tests** (unit + integration), **logging**, and **basic auth**.

---

## 2) Repo Structure (proposed)

```
VideoAnnotator/
├─ annotator_api/
│  ├─ main.py                 # FastAPI app (CORS, auth, routes)
│  ├─ deps.py                 # settings, auth, redis clients
│  ├─ models.py               # Pydantic schemas
│  ├─ routes/
│  │  ├─ datasets.py
│  │  ├─ pipelines.py
│  │  ├─ jobs.py
│  │  └─ artifacts.py
│  ├─ services/
│  │  ├─ registry.py          # discover pipelines + metadata
│  │  ├─ queue.py             # enqueue, status, logs
│  │  ├─ manifests.py         # dataset/job manifests
│  │  └─ artifacts.py         # artifact manifest builder
│  ├─ events.py               # SSE/WebSocket stream
│  └─ __init__.py
├─ workers/
│  ├─ rq_worker.py            # RQ worker entrypoint
│  ├─ tasks.py                # run_job(job_spec), emit progress
│  └─ __init__.py
├─ pipelines/                 # existing engines/tools
│  ├─ __init__.py
│  └─ registry_config.yaml    # declares available pipelines & params
├─ configs/
│  ├─ presets/                # *.yaml (named recipes)
│  └─ datasets/               # *.yaml (optional saved dataset defs)
├─ docker/
│  ├─ Dockerfile.api
│  ├─ Dockerfile.worker
│  └─ docker-compose.yaml
├─ tests/
│  ├─ api/
│  ├─ worker/
│  └─ e2e/
└─ README.md
```

---

## 3) Data Contracts (Pydantic)

```python
# annotator_api/models.py
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Optional
from datetime import datetime

JobState = Literal["queued","running","succeeded","failed","canceled"]

class DatasetCreate(BaseModel):
    name: str
    base_path: str  # server path or mounted volume

class Dataset(BaseModel):
    id: str
    name: str
    base_path: str
    created_at: datetime

class VideoRef(BaseModel):
    path: str                # absolute or dataset-relative
    duration_sec: Optional[float] = None

class PipelineParam(BaseModel):
    name: str
    type: Literal["int","float","bool","string","enum"]
    default: Optional[str|int|float|bool]
    choices: Optional[List[str]] = None

class Pipeline(BaseModel):
    key: str                 # e.g., "faces_openface3"
    label: str
    version: str
    params: List[PipelineParam]

class Preset(BaseModel):
    name: str
    pipelines: Dict[str, Dict[str, object]]  # {pipeline_key: {param: value}}

class JobCreate(BaseModel):
    dataset_id: str
    videos: List[VideoRef]
    preset: Optional[str] = None
    overrides: Dict[str, Dict[str, object]] = {}  # pipeline->params
    tags: List[str] = []

class Job(BaseModel):
    id: str
    state: JobState
    queued_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    progress_pct: float
    current_stage: Optional[str]
    logs_tail: List[str] = []

class Artifact(BaseModel):
    type: Literal["video","vtt","rttm","json","csv","png","wav"]
    role: str               # e.g., "transcript","speakers","tracks","scenes"
    path: str               # absolute or dataset-relative
    meta: Dict[str, object] = {}

class ArtifactManifest(BaseModel):
    job_id: str
    dataset_id: str
    outputs: Dict[str, List[Artifact]]      # keyed by input video path
    viewer_hint: Dict[str, object] = {}     # how Viewer should open overlays
```

**Artifact Manifest (example)**

```json
{
  "job_id": "job_abc123",
  "dataset_id": "ds_movies",
  "outputs": {
    "clips/001.mp4": [
      {"type":"vtt","role":"transcript","path":"outputs/001/whisper.vtt"},
      {"type":"rttm","role":"speakers","path":"outputs/001/diarize.rttm"},
      {"type":"json","role":"scenes","path":"outputs/001/scenes.json"},
      {"type":"json","role":"tracks","path":"outputs/001/tracks.json"}
    ]
  },
  "viewer_hint": {"overlayOrder":["tracks","speakers","scenes","transcript"]}
}
```

---

## 4) HTTP API (v0)

* `GET  /healthz`
* `GET  /pipelines` → `[Pipeline]`
* `GET  /presets` → `[Preset]`
* `POST /datasets` → `Dataset`
* `GET  /datasets` → `[Dataset]`
* `POST /datasets/{id}/videos/scan` (optional) → auto-discover videos under base path
* `POST /jobs` (JobCreate) → `Job`
* `GET  /jobs` → list with filters `state`, `tag`
* `GET  /jobs/{id}` → `Job`
* `POST /jobs/{id}/cancel` → `Job`
* `GET  /jobs/{id}/artifacts` → `ArtifactManifest`
* `GET  /events/stream` (SSE) → `event: job.update` JSON payloads

**SSE example response (newline-delimited):**

```
event: job.update
data: {"id":"job_abc123","state":"running","progress_pct":37.5,"current_stage":"whisper"}

event: job.log
data: {"id":"job_abc123","line":"[whisper] chunk 07/16 ..."}
```

---

## 5) Pipeline Registry

**Config-first discovery**, decoupled from code. Each entry declares:

* `key`, `label`, `version`
* entrypoint: `python -m videoannotator.pipelines.faces_openface3`
* input types & required models
* parameters (with defaults and ranges)

`pipelines/registry_config.yaml`:

```yaml
- key: faces_openface3
  label: "Faces (OpenFace 3)"
  version: "3.0.0"
  params:
    confidence_threshold: { type: float, default: 0.6, min: 0.0, max: 1.0 }
- key: speech_whisper
  label: "Speech (Whisper)"
  version: "large-v3"
  params:
    language: { type: enum, default: "auto", choices: ["auto","en","es","fr","pt"] }
    diarization: { type: bool, default: true }
```

---

## 6) Worker + Queue

**RQ + Redis (simple, reliable)**

* `enqueue(job_spec)` → returns `job_id`
* Worker executes `tasks.run_job(job_spec)`:

  * Stage loop: for each selected pipeline → process → write outputs under `outputs/{video_basename}/...`
  * Emit progress via Redis pubsub → forwarded to SSE.

```python
# workers/tasks.py
def run_job(job_spec: dict):
    # resolve dataset path, preset, and overrides
    for i, video in enumerate(job_spec["videos"]):
        stage_ctx = {...}
        emit_progress(job_id, pct(i/len(videos)*100), "preflight")
        for stage in build_stages(job_spec):
            emit_progress(job_id, current_pct, stage.name)
            stage.run(video, stage_ctx)  # your existing pipeline calls
    build_artifact_manifest(job_id)
```

---

## 7) Filesystem Layout (volumes)

* **Inputs:** `/data/videos/...`
* **Outputs:** `/data/outputs/{job_id}/{video_basename}/...`
* **Scratch/cache:** `/data/cache/...`
* **Presets/manifests:** `configs/presets/*.yaml`

Keep **relative paths** inside manifests so the Viewer can map them.

---

## 8) Docker & Compose (GPU-aware)

`docker/docker-compose.yaml` (sketch):

```yaml
version: "3.9"
services:
  redis:
    image: redis:7-alpine
    restart: unless-stopped

  api:
    build: { context: .., dockerfile: docker/Dockerfile.api }
    environment:
      - REDIS_URL=redis://redis:6379/0
      - AUTH_TOKEN=${AUTH_TOKEN:-dev-token}
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    ports: ["8080:8080"]
    volumes:
      - ./../:/app
      - video_data:/data
    depends_on: [redis]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  worker:
    build: { context: .., dockerfile: docker/Dockerfile.worker }
    environment:
      - REDIS_URL=redis://redis:6379/0
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    volumes:
      - ./../:/app
      - video_data:/data
    depends_on: [redis]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
volumes:
  video_data:
```

> Note: On some setups the `deploy` block is ignored outside Swarm. If so, start with `docker compose up` and ensure NVIDIA toolkit is installed; or use `docker run --gpus all` equivalent. Alternatively, use a Compose version supporting `gpus: all`.

---

## 9) Security & Settings

* **Auth:** Single bearer token header (e.g., `X-API-Key`) for now.
* **CORS:** Allow `video-annotation-viewer` origin.
* **Path safety:** Only allow datasets in **whitelisted roots**; block `..` and symlinks escaping roots.
* **Rate limits:** Minimal (optional).

---

## 10) Testing Strategy

* **Unit:** pipeline registry, parameter parsing, manifest builder.
* **Integration:** spin `redis`, start `worker` & `api`, enqueue a tiny test pipeline (CPU fallback) on a sample clip, assert artifact manifest shape.
* **E2E (headless):** POST dataset → POST job → poll SSE → GET artifacts → verify files exist.

---

## 11) Acceptance Criteria

* Can create dataset, enqueue a batch job, observe live progress, and fetch an artifact manifest that the Viewer can open without manual path edits.
* Reproducible runs via saved **preset** files.
* Jobs survive API restarts (Redis persists state + filesystem outputs).
* Logs stream over SSE; failures expose error messages and partial artifacts.
* All API responses are documented in OpenAPI and pass basic contract tests.
