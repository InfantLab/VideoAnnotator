
# Implementation Plan · Web UI (`video-annotation-viewer`)

**Goal:** Extend the existing Viewer with a *Create/Batch* UI that orchestrates annotation jobs via the new API, then opens completed outputs in the existing playback/overlay views.

---

## 1) High-Level Deliverables

* New **Create** route with wizard-style flow.
* **Job list** & **Job details** (progress, logs).
* **Dataset manager** (register paths; optional scanning).
* **Preset manager** (create/save/edit).
* **Open in Viewer** deep links using the artifact manifest.

---

## 2) Tech & Libraries

* React + TypeScript (existing)
* Router (existing)
* State: **Zustand** or **React Query** (choose one; examples below use React Query for data + SSE)
* UI: Tailwind + your components; (optionally) shadcn/ui
* API client: Generate from OpenAPI (`openapi-typescript` or `orval`)
* Tests: Vitest + React Testing Library; Playwright for smoke E2E

---

## 3) Routes & Screens

```
/create
  ├─ datasets          # register/list datasets
  ├─ new               # wizard: dataset -> videos -> pipelines/preset -> confirm
  ├─ jobs              # job table
  └─ jobs/:id          # job detail (progress, logs, artifacts, "Open in Viewer")
```

---

## 4) UI Flow (MVP)

1. **Datasets**

   * List known datasets (`GET /datasets`)
   * “Register dataset” (name, base\_path) → `POST /datasets`
   * Optional: “Scan for videos” → `POST /datasets/{id}/videos/scan` → table of found files

2. **New Batch**

   * Step 1: Pick dataset + select videos (checkbox list; display duration if known)
   * Step 2: Choose **Preset** or **custom**:

     * Load presets (`GET /presets`)
     * Show pipelines (`GET /pipelines`) with param editors
   * Step 3: Confirm:

     * Show **estimated runtime/storage** (coarse)
     * Submit → `POST /jobs`

3. **Jobs**

   * Table with filters (`state`, `tag`)
   * Columns: ID, Created, State, Progress, Tags, Actions
   * Live progress via **SSE** (subscribe to `/events/stream`)

4. **Job Detail**

   * Progress bar; current stage
   * Logs tail (autoscroll)
   * When completed, fetch `GET /jobs/{id}/artifacts`
   * **Open in Viewer** button → deep link into existing playback with overlays from `viewer_hint` or artifact roles

---

## 5) Component Sketch

* `DatasetManager.tsx`
* `VideoPicker.tsx`
* `PipelinePicker.tsx`
* `PresetEditor.tsx`
* `JobTable.tsx`
* `JobDetail.tsx` (includes `LogsPane.tsx`, `ArtifactsList.tsx`)
* `SSEProvider.tsx` (context to multiplex a single EventSource)

---

## 6) OpenAPI Client

**Generate types & hooks (example with `openapi-typescript`):**

```bash
npx openapi-typescript http://localhost:8080/openapi.json -o src/api/schema.d.ts
```

Create a thin `api.ts` wrapper with `fetch` and auth header injection.

---

## 7) SSE Handling

```ts
// src/events/sse.ts
export function connectSSE(token: string, onEvent: (ev: MessageEvent) => void) {
  const es = new EventSource(`/events/stream?token=${token}`, { withCredentials: false });
  es.onmessage = onEvent; // default event
  es.addEventListener("job.update", onEvent as any);
  es.addEventListener("job.log", onEvent as any);
  return () => es.close();
}
```

Use React context so multiple components can subscribe without multiple sockets.

---

## 8) “Open in Viewer” Deep Link

Consume `ArtifactManifest.viewer_hint` plus `outputs[video]` roles. Example:

```
/viewer?dataset=ds_movies&video=clips/001.mp4
  &overlay=tracks:outputs/001/tracks.json
  &overlay=scenes:outputs/001/scenes.json
  &overlay=transcript:outputs/001/whisper.vtt
```

(Adapt parameterization to your current Viewer query scheme.)

---

## 9) Error Handling & UX

* Toasts on API errors (network, 4xx/5xx).
* Guardrails: if API is unreachable, show reconnect banner.
* Disable “Start Batch” until required selections made.
* Show job failure reason and surface last 50 log lines.

---

## 10) Configuration

* `VITE_API_BASE_URL`
* `VITE_API_TOKEN`
* CORS origin must match `api` config.

---

## 11) Tests

* **Unit/Component:** Pipeline param editors, JobTable sorting, SSE reducer.
* **Integration:** Mock API + SSE; assert job row updates to 100% and “Open in Viewer” becomes enabled.
* **E2E:** Playwright script: register dataset → new job on a tiny MP4 → wait until done → click “Open in Viewer” and assert overlays present.

---

## 12) Acceptance Criteria

* Can register a dataset and list videos.
* Can start a batch using a preset or custom params.
* Live progress is visible without manual refresh.
* On completion, the job provides an artifact list and a working deep link into the playback view with overlays.
* Basic error states are surfaced (invalid path, failed pipeline, API down).

---

## 13) Developer Tasks (checklist)

**Engine/API**

* [ ] Implement Pydantic models & FastAPI routes
* [ ] Registry loader for `pipelines/registry_config.yaml`
* [ ] RQ queue + worker baseline; pubsub → SSE bridge
* [ ] Artifact manifest builder (per video)
* [ ] Preset loader & save
* [ ] Auth token + CORS
* [ ] Dockerfiles + Compose
* [ ] Unit + integration tests
* [ ] OpenAPI doc + TS client generation

**Viewer**

* [ ] Add `/create` section and navigation
* [ ] Dataset manager page
* [ ] New batch wizard (dataset → videos → preset/pipelines → confirm)
* [ ] Jobs table + detail views
* [ ] SSE connection + reducers
* [ ] Artifact view + “Open in Viewer” deep link
* [ ] API client (OpenAPI) + error toasts
* [ ] Tests (Vitest + Playwright)

---

## 14) Example Endpoint Stubs (FastAPI)

```python
# annotator_api/main.py
from fastapi import FastAPI, Depends, Header, HTTPException
from . import models
from .services import registry, queue, manifests, artifacts

app = FastAPI(title="VideoAnnotator API", version="0.1.0")

def auth(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != "dev-token":  # replace with config
        raise HTTPException(status_code=401)
    return True

@app.get("/pipelines", response_model=list[models.Pipeline])
def get_pipelines(_: bool = Depends(auth)):
    return registry.list_pipelines()

@app.post("/datasets", response_model=models.Dataset)
def create_dataset(payload: models.DatasetCreate, _: bool = Depends(auth)):
    return manifests.create_dataset(payload)

@app.post("/jobs", response_model=models.Job)
def create_job(payload: models.JobCreate, _: bool = Depends(auth)):
    return queue.enqueue(payload)

@app.get("/jobs/{job_id}", response_model=models.Job)
def get_job(job_id: str, _: bool = Depends(auth)):
    return queue.status(job_id)

@app.get("/jobs/{job_id}/artifacts", response_model=models.ArtifactManifest)
def get_artifacts(job_id: str, _: bool = Depends(auth)):
    return artifacts.for_job(job_id)
```

---

## 15) Example Preset

```yaml
# configs/presets/lightweight_realtime.yaml
name: lightweight_realtime
pipelines:
  speech_whisper:
    model: "small"
    language: "auto"
    diarization: true
  scenes_adaptive:
    min_scene_len_sec: 8
  tracks_person:
    conf_thresh: 0.5
```

---

## 16) Observability (optional but recommended)

* Structured logs (JSON) per stage and per video.
* `/metrics` (Prometheus) with counters: jobs\_started, jobs\_failed, stage\_duration\_seconds.
* `X-Job-Id` correlation id across logs.

