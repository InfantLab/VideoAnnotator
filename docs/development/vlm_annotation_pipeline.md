# VLM Frame Annotation Pipeline

## What this is

`vlm_annotation` is a VideoAnnotator pipeline that samples frames from a video
on a fixed cadence and classifies/describes each one using a locally-hosted
vision-language model (VLM), served by [Ollama](https://ollama.com), driven
by a user-supplied text prompt. Every sample point is one independent,
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
| 3 | Job-creation UX (prompt textarea, sampling-mode picker) | Planned — lives in the `video-annotation-viewer` repo |
| 4 | Full research-review UI (ELAN ground truth, cross-prompt comparison, failure-mode buckets) | Planned — lives in the `video-annotation-viewer` repo |

Phases 3 and 4 are tracked in this document's "Full plan" section below for
continuity, but the actual work happens in the `video-annotation-viewer`
repo, not here.

## Architecture

```
src/videoannotator/pipelines/vlm_annotation/
├── __init__.py
├── vlm_pipeline.py      VLMAnnotationPipeline(BasePipeline) — sampling + orchestration
└── ollama_client.py     OllamaVLMClient — stateless client.chat() wrapper, retry/backoff

src/videoannotator/registry/metadata/vlm_annotation.yaml   config_schema, requires_extras: [llm]
```

- **Model client** (`ollama_client.py`): wraps the `ollama` Python package's
  `Client.chat()` directly — the same library the research repo already uses
  in production — rather than a hand-rolled OpenAI-compatible HTTP client.
  `roadmap_v1.6.0.md` originally proposed "one connector against the
  OpenAI-compatible `/v1/chat/completions` contract" so the same code could
  also drive a `llama.cpp` server; that's a deliberate deviation for now,
  not an oversight — reusing the proven `ollama` client de-risked getting a
  working pipeline landed quickly. `backends: [ollama]` in the pipeline
  metadata leaves room for an `openai_compatible` backend alongside it later
  without redesigning the pipeline.
- **Sampling modes** (`vlm_pipeline.py::process`), both supported from day
  one because the research repo's own roadmap treats the choice between them
  as a still-open research question (see `docs/research_roadmap.md` §Phase 4
  Decision D in the touch-detection repo):
  - `single_frame` — one image per sample point. Ported from
    `core/inference.py`.
  - `frame_burst` — a window of frames (`burst_offsets`, default
    `[-2,-1,0,1,2]`, in units of `frame_interval_sec`) attached to *one* chat
    call as multiple `images` list entries. Ported from
    `core/context_inference.py`. **This is not true video/temporal input** —
    it's several still JPEGs in one message, exactly like the research repo's
    own "3-image grid" mode. Genuine video-native input is out of scope here,
    same as it is (still unimplemented, explicitly conditional) in the
    research repo.
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

## Configuration reference

All fields have defaults — nothing is required. Set via job `config` keyed
by pipeline name: `{"vlm_annotation": {"prompt": "...", ...}}`.

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | Irene's `maternal_perspective` touch-detection prompt | Applied identically and independently to every sample point |
| `base_url` | string | `http://127.0.0.1:11434` | Local Ollama server |
| `model` | string | `qwen3.5:9b` | Must already be pulled (`ollama pull <name>`) — nothing is auto-downloaded |
| `sampling_mode` | string | `single_frame` | `single_frame` or `frame_burst` |
| `frame_interval_sec` | float | `5.0` | Seconds between sample points |
| `burst_offsets` | list[int] | `[-2,-1,0,1,2]` | `frame_burst` only, in units of `frame_interval_sec` |
| `think` | boolean | `false` | Ollama "thinking" mode — ~50x slower, diagnostic use only |
| `request_timeout_sec` | integer | `240` | Per-request timeout |
| `keep_alive` | string | `"2h"` | How long Ollama keeps the model loaded between calls |

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

pipeline = VLMAnnotationPipeline({
    "model": "qwen3.5:9b",
    "sampling_mode": "single_frame",   # or "frame_burst"
    "frame_interval_sec": 5.0,
})
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
