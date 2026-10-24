# VLM Frame Annotation Pipeline

## What this is

`vlm_annotation` is a VideoAnnotator pipeline that samples frames from a video
on a fixed cadence and classifies/describes each one using a vision-language
model (VLM) on a model server, [Ollama](https://ollama.com) by default or any
OpenAI-compatible server (llama.cpp, vLLM, LM Studio, SGLang; spec 033),
driven by a user-supplied text prompt, optionally with an answer shape. Every sample point is one independent,
stateless model call — no conversation history carries over between frames,
so one prediction can never be biased by a previous one.

It implements the "Local LLM/VLM Backend" item from
[`roadmap_v1.6.0.md`](roadmap_v1.6.0.md) Phase 2, brought forward and driven
by a real user: Irene's PhD research project on mother-infant touch detection
(`mother-infant-touch-detection` repo — a sibling project, not part of this
one). That project's own ad hoc research scripts
(`core/inference.py`, `core/context_inference.py`) are what this pipeline is
ported from — see "Architecture" below for the mapping.

## Status

| Phase | What | Status |
|---|---|---|
| 1 | Pipeline itself (this doc) | **Done** — merged into `v1.5.0` |
| 2 | End-to-end validation | **Done** — see "Step-by-step: test it yourself" |
| 3 | Job-creation UX (prompt textarea, sampling-mode picker) | **Done** — see below |
| 4 | Full research-review UI (ELAN ground truth, cross-prompt comparison, failure-mode buckets) | **Foundation done**, rest planned — lives in the `video-annotation-viewer` repo |

Phase 3 landed in two repos:

- **This repo**: a new `GET /api/v1/pipelines/{name}/schema` endpoint, plus
  two additive `PipelineConfigField` YAML hints (`widget: textarea`,
  `enum: [...]`) that `vlm_annotation.yaml` now uses for its `prompt` and
  `sampling_mode` fields. Turned out to be a broader fix than just this
  pipeline: the viewer's job-creation form was already coded to call this
  exact endpoint and only fell back to a blank parameter list on 404 —
  since the endpoint never existed, **every** pipeline's Configure step
  showed "No configurable parameters," not just this one. Every pipeline
  now gets a real form for free.
- **`video-annotation-viewer` repo**: a `'text'` parameter type +
  `Textarea` case (`DynamicPipelineParameters.tsx`), and wiring
  `usePipelineSchema`/`pipelineSchemaQueryOptions` (already written,
  previously unused) into `NewJob.tsx`'s Configure step so it actually
  fetches and merges each selected pipeline's real schema. Multi-video
  batch submission with one shared config — the other thing this phase
  needed — already existed (`NewJob.tsx`'s file input already takes
  multiple files and loops `submitJob` per file with the same config).

Verified live in a browser against a real server: `vlm_annotation`'s
Configure step renders `prompt` as a pre-filled multi-line textarea and
`sampling_mode` as a dropdown, with no console errors and no regression for
pipelines without either hint.

Phase 4 is tracked in this document's "Full plan" section below for
continuity, but the actual work happens in the `video-annotation-viewer`
repo, not here.

## Architecture

```
src/videoannotator/pipelines/vlm_annotation/
├── __init__.py
├── vlm_pipeline.py      VLMAnnotationPipeline(BasePipeline) — sampling + orchestration
├── clients.py           VLMClient protocol, make_client(backend), check_server() (spec 033)
├── ollama_client.py     OllamaVLMClient — stateless client.chat() wrapper, retry/backoff
├── openai_client.py     OpenAICompatibleVLMClient — /models, /chat/completions on httpx (spec 033)
├── answer_shape.py      answer shapes: validate, JSON Schema, check replies, suggest (spec 033)
├── sampling.py, frames.py   where samples fall; the frames (or clip) each one sends
└── gate.py              tries go ahead of runs on the same model

src/videoannotator/registry/metadata/vlm_annotation.yaml   config_schema, requires_extras: [llm]
```

- **Model clients** (`clients.py`): `backend: ollama` (the default) wraps the
  `ollama` Python package's `Client.chat()` — the library the research repo
  uses, and the one that reports model digests for provenance.
  `backend: openai_compatible` (spec 033) is `openai_client.py`, plain httpx
  against `GET {base}/models` and `POST {base}/chat/completions`, one
  `image_url` data URL per frame: llama.cpp's `llama-server`, vLLM, LM Studio,
  SGLang, Ollama's own `/v1` and hosted APIs. This resolves the deviation from
  `roadmap_v1.6.0.md`'s "one connector against the OpenAI-compatible contract"
  that v1.5.0 took to land the pipeline quickly. Retries, backoff, the gate,
  stopping and progressive writes are the same for both; a 401/403 raises
  `KeyRefusedError` and stops the run at once. Load time and tokens/s are
  `null` where the server doesn't report them. The model ref records the
  server's `system_fingerprint` (`revision_kind: server-fingerprint`) or
  `unknown` with a reason.
- **Answer shapes** (`answer_shape.py`, spec 033 R8): `answer_shape` is a list
  of fields (`choice`, `yes_no`, `number`, `text`). It is sent as Ollama's
  `format` or as `response_format: json_schema`; a server that rejects that
  gets the shape described at the end of the prompt instead (remembered per
  listed server; provenance `structured_output: prompt_only`). Every reply is
  checked here: `fields`, `fit`, `fit_problem` on each annotation, `label` the
  first field or `DIDN'T FIT`, and the COCO top level gains
  `vlm_answer_shape`; provenance gains `answer_shape_sha256`. A misfit counts
  as answered. No shape: nothing changes.
- **Keys and remote servers** (spec 033 R4, R5): a config never carries a key
  (`KEY_IN_CONFIG`); `initialize()` takes it from Settings → Model servers
  (`model_servers.key_for_url`). `initialize()` also refuses a remote address
  unless remote servers are switched on and it is confirmed
  (`endpoint_policy.require_sendable`), so a run from the CLI or the API can't
  skip the check.
- **Sampling modes** (`vlm_pipeline.py::process`), both supported from day
  one because the research repo's own roadmap treats the choice between them
  as a still-open research question (see `docs/research_roadmap.md` §Phase 4
  Decision D in the touch-detection repo):
  - `single_frame` — one image per sample point. Ported from
    `core/inference.py`.
  - `frame_burst` — a window of frames (`burst_offsets`, default
    `[-2,-1,0,1,2]`, in units of `burst_spacing_sec`, which defaults to
    `frame_interval_sec`) attached to *one* chat
    call as multiple `images` list entries. Ported from
    `core/context_inference.py`. **This is not true video/temporal input** —
    it's several still JPEGs in one message, exactly like the research repo's
    own "3-image grid" mode.
  - `clip` (spec 033 R12) — genuine video input, for servers and models that
    take it (vLLM with Qwen-VL): the frames from `t − clip_length_sec/2` to
    `t + clip_length_sec/2` (shortened at the video's ends) at `clip_fps`, sent
    as one `video_url` frame-sequence part, with the fps in `media_io_kwargs`
    where the server accepts it. `initialize()` first probes the model with a
    synthetic two-frame clip. Annotations record `clip_start_sec`,
    `clip_end_sec`, `clip_fps`. Ollama with `clip` is refused.
- **Output**: COCO-shaped annotation dicts (`create_coco_annotation`), one
  per sample point, with `label` (parsed from the model's response),
  `reasoning` (thinking text if `think: true`, else the raw response),
  `raw_response`, `timestamp_sec`, `frame_number`, `sampling_mode`,
  `context_frame_offsets`/`context_frame_numbers` (burst mode only), and full
  provenance per record: `model`, `backend`, `base_url`, `prompt`, plus
  timing/token stats (`total_time`, `load_time`, `prompt_tokens`,
  `resp_tokens`, `tokens_per_sec`). Written to
  `{output_dir}/{video_id}_vlm_annotation.json` and validated with
  `pycocotools` on write, same as `scene_detection`.
- **Answers as they come** (spec 027): the file is rewritten (to a temporary
  name, then moved into place) after each answer, at most once a second, and
  at the end. Each annotation also has `frame_times` and `status`
  (`answered` | `error`); the top level has `vlm_progress`
  (`{planned, done, finished, stopped_early}`) and `vlm_skipped` (sample
  points whose frames couldn't be read, `{t, reason}`: listed, never
  answered). Readers that don't know these keys see the same annotations.
- **Stopping keeps the answers**: the job runner gives the pipeline a stop
  check (`set_stop_check`), asked before each sample. Cancelling the job
  (`POST /api/v1/jobs/{id}/cancel`) makes the pipeline write what it has and
  return normally: the job is `cancelled`, its `vlm_annotation` result
  `completed`, and the provenance's `vlm` block says `stopped_early`,
  `samples_planned` and `samples_answered`, with `endpoint_class`
  (`local` | `remote`) and `endpoint_host`.
- **Tries go first**: every model call holds a per-model gate in the server
  process (`gate.py`); a run's next sample waits while an Ask AI try is
  waiting. A job run by a separate `videoannotator worker` process doesn't
  share the gate.
- **Where samples fall** is one set of functions (`sampling.py`), mirrored in
  the viewer (`viewer/src/lib/sampling.ts`) and pinned by
  `tests/fixtures/ask_ai/sampling_cases.json`.

## Configuration reference

All fields have defaults — nothing is required. Set via job `config` keyed
by pipeline name: `{"vlm_annotation": {"prompt": "...", ...}}`.

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | Irene's `maternal_perspective` touch-detection prompt | Applied identically and independently to every sample point |
| `backend` | string | `ollama` | `ollama` or `openai_compatible` (spec 033) |
| `base_url` | string | `""` (server's `OLLAMA_BASE_URL`, else `http://127.0.0.1:11434`) | The model server. Empty only with Ollama. The devcontainer and `docker-compose.yml` set `OLLAMA_BASE_URL=http://host.docker.internal:11434` so a server in a container reaches Ollama on the host |
| `model` | string | `qwen3.5:9b` | As the server names it; with Ollama, already pulled (`ollama pull <name>`) — nothing is auto-downloaded |
| `answer_shape` | list \| null | `null` | Named fields, e.g. `[{name: touch, kind: yes_no}]` (spec 033) |
| `sampling_mode` | string | `single_frame` | `single_frame`, `frame_burst` or `clip` |
| `clip_length_sec` | float | `2.0` | `clip` only: seconds of video around each sample point |
| `clip_fps` | float | `4.0` | `clip` only: frames per second read for each clip |
| `frame_interval_sec` | float | `5.0` | Seconds between sample points |
| `start_sec` | float | `0.0` | First sample point |
| `end_sec` | float \| null | `null` | Last possible sample point; `null` is the end of the video |
| `moments_sec` | list[float] \| null | `null` | Sample only at these seconds; replaces the interval grid (and `start_sec`/`end_sec`). Moments outside the video are left out |
| `burst_offsets` | list[int] | `[-2,-1,0,1,2]` | `frame_burst` only, in multiples of the frame spacing |
| `burst_spacing_sec` | float \| null | `null` | `frame_burst` only: seconds between a sample's frames; `null` is `frame_interval_sec`, as before this setting existed. A frame outside the video is dropped, and the annotation lists only the frames sent |
| `think` | boolean | `false` | Ollama "thinking" mode — ~50x slower, diagnostic use only |
| `request_timeout_sec` | integer | `240` | Per-request timeout |
| `keep_alive` | string | `"2h"` | How long Ollama keeps the model loaded between calls; not sent to OpenAI-compatible servers |

## Known gaps / sharp edges

- **Two independent job-execution engines exist in VideoAnnotator today**
  (pre-existing, not introduced by this pipeline — `roadmap_v1.6.0.md`
  already commits to consolidating them):
  1. `videoannotator server` **automatically** starts an in-process polling
     loop on startup (`api/background_tasks.py`) that processes pending jobs
     via `api/job_processor.py`.
  2. The separate `videoannotator worker` CLI command starts a *second*,
     independent polling loop (`worker/job_processor.py`) via
     `batch/batch_orchestrator.py`.

  **Don't run both against the same job database at once** — they'd both try
  to claim the same pending jobs. For everything below, `videoannotator
  server` alone is enough; no separate `worker` process is needed.
- `api/job_processor.py` previously didn't pass job config into the pipeline
  constructor at all (fixed as part of this work — see the commit that added
  this pipeline). If you see a pipeline ignoring its submitted config on a
  future VideoAnnotator checkout, check that fix is still in place.
- The CLI's `job submit` / `job download-annotations` commands don't support
  passing an API key — they'll 401 against a server with auth enabled. For
  local testing, run the server with `AUTH_REQUIRED=false` (see below). This
  is a pre-existing CLI gap, not specific to this pipeline.
- `config_schema`'s `prompt` field has no "long text" widget hint yet, so a
  UI building a form from it today would render a one-line input, not a
  textarea. That's Phase 3 work (see "Full plan").
- `videoannotator diagnose` doesn't yet detect a reachable Ollama server the
  way it detects GPU availability — a `roadmap_v1.6.0.md` item that's still
  open.

## Step-by-step: test it yourself

### Prerequisites

1. Ollama installed and a vision model pulled: `ollama pull qwen3.5:9b` (or
   edit the `model` config field to whatever you have — `ollama list` shows
   what's already pulled).
2. VideoAnnotator installed with the `llm` extra:
   ```bash
   pip install -e ".[llm]"
   # or: uv pip install -e ".[llm]"
   ```

### Option A — fastest: direct Python smoke test (no server, no auth)

```python
from videoannotator.pipelines.vlm_annotation import VLMAnnotationPipeline

pipeline = VLMAnnotationPipeline(
    {
        "model": "qwen3.5:9b",
        "sampling_mode": "single_frame",  # or "frame_burst"
        "frame_interval_sec": 5.0,
    }
)
annotations = pipeline.process(
    video_path="path/to/your/video.mp4",
    output_dir="path/to/output_dir",
)
for ann in annotations:
    print(ann["timestamp_sec"], ann["label"], ann["raw_response"])
pipeline.cleanup()
```

This is the quickest way to check a prompt or model change — it talks
straight to Ollama, no API server, no database, no auth. Output also lands as
COCO JSON at `{output_dir}/{video_id}_vlm_annotation.json`.

### Option B — the realistic path: CLI + API server

This is closer to what the eventual viewer UI will do.

```bash
# 1. Start the server. AUTH_REQUIRED=false avoids the CLI's lack of
#    API-key support (see "Known gaps" above) — local testing only.
AUTH_REQUIRED=false videoannotator server --port 18011

# 2. In another terminal: write a config file (keyed by pipeline name)
cat > vlm_config.json <<'EOF'
{
  "vlm_annotation": {
    "model": "qwen3.5:9b",
    "sampling_mode": "single_frame",
    "frame_interval_sec": 5.0
  }
}
EOF

# 3. Submit a job — the server's background poller (started automatically
#    with the server, see "Known gaps") picks it up within a few seconds.
videoannotator job submit path/to/video.mp4 \
    --pipelines vlm_annotation \
    --config vlm_config.json \
    --server http://127.0.0.1:18011

# 4. Poll status (or just wait ~5-10s for a short video)
videoannotator job status <job-id> --server http://127.0.0.1:18011

# 5. See a summary
videoannotator job results <job-id> --server http://127.0.0.1:18011
```

Output JSON lands at `storage/jobs/<job-id>/<video-stem>_output/<video-stem>_vlm_annotation.json`
under the server's working directory.

Both paths were run against a live `ollama serve` + `qwen3.5:9b` as part of
landing this pipeline, in both `single_frame` and `frame_burst` modes —
Option B additionally confirmed the job-config-passing fix (see "Known gaps")
actually takes effect end-to-end through the API.

## Phase 4 progress

The foundation landed: a `VLMFrameAnnotation` type, a parser
(`src/lib/parsers/vlm.ts`), detection wired into `merger.ts` (placed ahead
of the generic COCO-format checks, since this pipeline's export uses the
same COCO info/annotations envelope as person_tracking/scene_detection — a
`"reasoning"`+`"sampling_mode"` signature check claims it first), a
point-marker Timeline track, and a `VlmAnnotationPanel` showing the current
label + expandable reasoning synced to playback time. Verified in a real
browser against real pipeline output — labels render correctly, timeline
markers are clickable, and the panel updates correctly on scrub.

Building this surfaced a real, separate bug: the viewer has **four**
independent places that each re-derive a file's pipeline type from its
JSON content (`merger.ts`, `fileUtils.ts`, and two hardcoded arrays inside
`FileUploader.tsx`), not one. All four needed a `vlm_annotation` case
added; `fileUtils.ts`'s version turned out not to even handle the
COCO-wrapped `scene_detection` export correctly either (pre-existing, left
alone — out of scope here, but worth knowing about before trusting any one
of those four in isolation).

ELAN ground-truth comparison also now landed (`video-annotation-viewer`
commit `09991c1`, branch `feature/vlm-annotation-support`): `.eaf` parsing
(ported from `00_preprocessing/parse_eaf_to_timeline.py`), a timeline
segment track colored by tier, and a live agree/disagree indicator
comparing the VLM prediction against the ELAN four-way category at the
current time. Verified with a synthetic `.eaf` file covering both the
agree and disagree cases against real pipeline output.

Still not built: lap-state stratification, a cross-job "compare prompts"
view, client-side failure-mode bucketing, and the human-in-the-loop
agree/disagree review flow — see "Full plan" below for what each of those
involves.

## Full plan

The full four-phase plan this pipeline is part of:

1. **Backend pipeline** (this doc, `VideoAnnotator` repo) — done.
2. **End-to-end validation** — done (see above).
3. **Job-creation UX** (`video-annotation-viewer` repo) — a prompt textarea
   (needs a `multiline`/`text` widget hint added to `PipelineConfigField` and
   threaded through to the frontend's `DynamicPipelineParameters.tsx`), a
   sampling-mode picker (works today via the existing `enum` → `Select`
   pattern once the schema exposes it that way), and — if it doesn't already
   exist — a way to submit one config across a whole folder of videos at
   once, since Irene's real workflow is "one prompt, whole corpus."
4. **Full research-review UI** (`video-annotation-viewer` repo) — the
   largest remaining phase: a new `VLMFrameAnnotation` annotation type +
   parser, a timeline track + expandable-reasoning overlay component, an ELAN
   ground-truth `.eaf` parser (ported from
   `00_preprocessing/parse_eaf_to_timeline.py` in the touch-detection repo),
   lap-state stratification, a cross-job "compare prompts" view (since a
   VideoAnnotator job is one-video-one-config, comparing prompts means
   diffing two jobs' results), client-side failure-mode bucketing, and a
   human-in-the-loop agree/disagree + notes review flow. This is intended to
   reach parity with the touch-detection repo's existing standalone
   `tools/viewer/viewer.html`, generalized to work for any prompt/job rather
   than a hardcoded set of experiment runs.

Phases 3–4 are scoped in detail (file:line references into
`video-annotation-viewer`) in the planning conversation that produced this
pipeline — ask for that plan to be re-surfaced if picking this up fresh.

## References

- `roadmap_v1.6.0.md` — the release roadmap this pipeline's Phase 2 item
  belongs to.
- `mother-infant-touch-detection` repo (sibling project) — the research
  codebase this pipeline is ported from: `core/inference.py`,
  `core/context_inference.py`, `docs/research_roadmap.md`,
  `tools/viewer/viewer.html`.
- `.specify/memory/constitution.md` — Principle I (Local-First, explicitly
  names "a self-hosted Ollama instance" as an allowed backend) and Principle
  III (Provenance & Reproducibility, why every annotation carries
  model/backend/base_url/prompt) are the two most relevant here.
