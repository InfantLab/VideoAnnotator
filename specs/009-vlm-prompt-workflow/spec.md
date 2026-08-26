# Feature Specification: VLM Prompt Workflow UX — Preview, Model Discovery & Reachability

**Feature Branch**: `009-vlm-prompt-workflow`
**Created**: 2026-08-26
**Status**: Draft
**Input**: User description: "Make the vlm_annotation pipeline's prompt-based workflow as clear and low-friction as possible for a researcher iterating on prompt wording: let them test a prompt against a single frame before committing to a full multi-video run, pick a model from what's actually installed rather than typing a name, and know whether the configured Ollama server is reachable at all."

## Relationship to Existing Specs

Independent of [`006`](../006-job-execution-consolidation/spec.md),
[`007`](../007-datasets-and-presets/spec.md), and
[`008`](../008-batch-group-workflow/spec.md) — no functional dependency either direction. Can be
built and shipped in parallel with any of them. Builds on the already-shipped `vlm_annotation`
pipeline (`src/videoannotator/pipelines/vlm_annotation/`) and its `OllamaVLMClient`
(`ollama_client.py`), reusing both rather than introducing a second implementation of "call a local
VLM."

Backend-only. The consumer — a model picker, a "test this prompt" panel in the job-creation wizard,
and a two-job comparison view — is `video-annotation-viewer`'s own spec. See
[`viewer-handoff.md`](viewer-handoff.md).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Test a prompt against one frame before running it on a corpus (Priority: P1)

A researcher is refining prompt wording for a touch-classification task. Today the only way to see
how a prompt performs is submitting a full job against real video — for a 40-video corpus, that's a
slow, expensive way to discover a prompt needs a tweak. They instead test the current prompt wording
against a single frame and see the result immediately.

**Why this priority**: This is the actual, daily workflow for anyone iterating on a VLM prompt —
without it, every wording change costs a full job submission to evaluate.

**Independent Test**: Submit one frame (or a video path + timestamp) with a prompt, model, and
sampling mode; receive a label and reasoning back synchronously; confirm no job or annotation record
was created as a side effect.

**Acceptance Scenarios**:

1. **Given** an image and a prompt, **When** a preview is requested, **Then** the system returns a
   label and the model's reasoning/raw response without requiring a video upload or job submission.
2. **Given** a video already on the server and a timestamp within its duration, **When** a preview is
   requested by video path + timestamp instead of an uploaded image, **Then** the system extracts
   that frame itself and returns the same shape of result.
3. **Given** a preview request using `frame_burst` sampling mode, **When** it is processed, **Then**
   the system samples the same burst-offset window the real pipeline would use for that timestamp and
   sends all frames in one call, exactly as a real job would.

---

### User Story 2 - Pick a model from what's actually installed (Priority: P1)

A researcher configuring `vlm_annotation` today types a model name into a free-text field with no
feedback about whether it's actually pulled on the configured Ollama server — a typo or an unpulled
model silently fails only once a job is submitted and run. They instead pick from a live list of
models the server can actually reach.

**Why this priority**: Equal to User Story 1 — both remove a slow, expensive way to discover a
mistake (a failed multi-video job) in favor of an immediate one.

**Independent Test**: With a known Ollama server and a known set of pulled models, request the model
list; confirm it matches what `ollama list` reports for that same server.

**Acceptance Scenarios**:

1. **Given** a reachable Ollama server with one or more models pulled, **When** the model list is
   requested, **Then** the system returns exactly those model names.
2. **Given** an unreachable or misconfigured Ollama server, **When** the model list is requested,
   **Then** the system returns a clear error distinguishing "server unreachable" from "no models
   pulled" — not a generic failure.

---

### User Story 3 - Know whether Ollama is reachable at all (Priority: P2)

Before configuring anything, a researcher wants to know whether a local VLM is even usable right now
— the same way the existing diagnostic command already reports GPU availability.

**Why this priority**: Lower than US1/US2 because it's a smaller, standalone diagnostic addition
rather than part of the core iteration loop, but it directly closes an already-open item from
`roadmap_v1.6.0.md`.

**Independent Test**: Run the diagnostic command against a machine with Ollama running and again with
it stopped; confirm the reported reachability differs correctly in each case.

**Acceptance Scenarios**:

1. **Given** a running, reachable Ollama server at the configured address, **When** diagnostics are
   run, **Then** it reports Ollama as reachable.
2. **Given** no Ollama server running at the configured address, **When** diagnostics are run,
   **Then** it reports Ollama as unreachable, without the diagnostic command itself failing or
   hanging.

---

### Edge Cases

- A preview is requested for a `video_path` + timestamp beyond that video's actual duration: MUST
  return a clear error, not an out-of-range/garbage frame or a crash.
- A preview is requested with `frame_burst` sampling near the very start or end of a video (some
  burst offsets would fall outside `[0, duration]`): MUST clamp exactly as the real pipeline's burst
  sampling already does — reuse that logic, don't reimplement it with different edge behavior.
- The configured model is cold (not currently loaded in Ollama) when a preview is requested: MUST
  apply the same generous per-request timeout the real pipeline uses for a cold-start call, not a
  shorter "this is just a preview" timeout that spuriously fails.
- The model list is requested while the Ollama server is reachable but returns zero pulled models:
  MUST be reported distinctly from "server unreachable" (empty list vs. connection error).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a way to test a prompt against a single frame or a small burst
  of frames (matching `vlm_annotation`'s existing sampling modes) and receive the resulting label and
  reasoning synchronously, without creating a job or any persisted annotation record.
- **FR-002**: The preview action MUST reuse the same model-calling logic (`OllamaVLMClient`) the
  `vlm_annotation` pipeline itself uses, so a preview result is representative of what a full job
  submission would actually produce — not a separate, potentially-divergent implementation.
- **FR-003**: A preview request MUST accept either a directly-uploaded image or a reference to an
  already-uploaded video plus a timestamp, extracting the frame(s) itself in the latter case.
- **FR-004**: The system MUST provide a way to list the vision-language models currently available on
  the configured Ollama server.
- **FR-005**: The model-list action MUST distinguish, in its response, "server unreachable" from
  "server reachable but no models pulled" — both are valid states, not one generic failure.
- **FR-006**: The system MUST extend its existing diagnostic capability (`videoannotator diagnose`)
  to report whether the configured Ollama server is reachable.
- **FR-007**: This feature MUST NOT change the `vlm_annotation` pipeline's job-execution behavior,
  stored output format, or config schema — preview and model-listing are separate, additive
  capabilities layered on top of the existing pipeline's own client code.

### Key Entities

- **VLM preview request/result**: an ephemeral, non-persisted request/response pair — an image source
  (uploaded file, or a video reference plus timestamp), a prompt, a model, a sampling mode, and
  (optionally) burst offsets in; a label, reasoning text, raw model response, and timing information
  out. Never written to storage or associated with any job.

## API Contract for Downstream Consumers

- **Model list**: `GET /api/v1/vlm/models` → `{ base_url, models: string[] }` on success; a clear
  error body distinguishing unreachable-server from empty-model-list on failure.
- **Preview**: `POST /api/v1/vlm/preview` → accepts either a multipart image upload or a JSON body
  `{ video_path, timestamp_sec }`, plus `prompt`, `model`, `sampling_mode`, `burst_offsets` (optional,
  defaults matching the pipeline's own), `think` (optional). Returns `{ label, reasoning,
  raw_response, total_time, load_time, prompt_tokens, resp_tokens, tokens_per_sec }` — the same
  per-annotation fields a real job's output already carries, minus job/frame-identity fields that
  don't apply to an ephemeral preview.
- **Diagnostics**: `videoannotator diagnose` output gains an `ollama_reachable: bool` field (and,
  where reachable, the model list alongside it) — same presentation pattern as its existing GPU
  reporting.
- **Stability expectation**: matches prior specs' precedent once published.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A prompt can be tested against one frame and a result returned within the pipeline's own
  per-request timeout, with zero job or annotation records created as a side effect, verified by
  test.
- **SC-002**: Preview's label-parsing and model-calling behavior is verified, by a shared-code-path
  test (not merely "similar output"), to be the same code the real pipeline uses.
- **SC-003**: The model-list endpoint returns exactly the models a direct `ollama list` against the
  same server would show, verified by test against a real Ollama instance.
- **SC-004**: `videoannotator diagnose` correctly reports Ollama reachability in both the
  reachable and unreachable case, without hanging or crashing in either, verified by test.

## Assumptions

- **No functional dependency on 006/007/008** — this can be built and shipped independently, in
  parallel with any of them.
- **Preview results are never persisted.** No DB record, no annotation file — this is deliberately a
  lightweight iteration tool, not a parallel job-tracking system.
- **Cross-job comparison of two full jobs' stored annotations on the same video is out of scope
  here** — it needs no new server endpoint (each job's output is already stored and retrievable); see
  [`viewer-handoff.md`](viewer-handoff.md) for that piece, scoped to the viewer alone.
- **Comparison across an entire dataset (many videos) is explicitly out of scope** — that's
  [`010-corpus-analysis-foundations`](../010-corpus-analysis-foundations/spec.md)'s job, not this
  spec's.
