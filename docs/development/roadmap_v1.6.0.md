# 🚀 VideoAnnotator v1.6.0 Development Roadmap

## Release Overview

VideoAnnotator v1.6.0 is the **Plugin Ecosystem & Local LLM Backends Release**. It picks up the parts
of [`specs/003-modular-pipeline-architecture/spec.md`](../../specs/003-modular-pipeline-architecture/spec.md)
that [`roadmap_v1.5.0.md`](roadmap_v1.5.0.md) deliberately deferred to keep the JOSS resubmission
scope small, and adds a new capability: local LLM/VLM pipelines running against a user's own
`ollama serve` or `llama.cpp` server instead of an in-process, bundled model.

**Target Release**: After v1.5.0 ships and the JOSS resubmission is under review
**Current Status**: Planning Phase
**Main Goal**: Complete the plugin architecture seam; make current-generation local models usable
without adding them as bundled, versioned dependencies

**Prerequisites**: v1.5.0's extras-based install and metadata-driven registry loading (this release
builds directly on both — a plugin is just an externally-installed extras group, and a local-LLM
pipeline is just another `backends` entry in pipeline metadata).

---

## 📋 v1.6.0 Deliverables

### Phase 1: Plugin Ecosystem (User Story 3 of spec 003)

**Problem**: v1.5.0 makes the *bundled* pipelines optional and slim, but a third party still cannot
ship a new pipeline without modifying VideoAnnotator's source.

**Solution**:
- [ ] `Dispatcher` ABC with `LocalThreadDispatcher` preserving current behaviour — the seam for a
      future `HTTPDispatcher`/`SlurmDispatcher`, not their implementation.
- [ ] Consolidate the duplicate job-execution paths (`api/job_processor.py` and
      `batch/batch_orchestrator._process_single_job`) into one function that both the CLI and API
      call, per FR-007.
- [ ] Pipeline discovery via `importlib.metadata.entry_points`, unioned with the bundled YAML
      metadata from v1.5.0. Reserved extras-group names enforced at the registry level; a
      third-party plugin declaring a colliding name is a deterministic error (not silent override).
- [ ] Migrate cross-cutting utilities (`person_identity`, `automatic_labeling`, `model_loader`,
      `size_based_person_analysis`) to a sibling `videoannotator-utils` package in the monorepo, so
      plugin authors have a stable, independently-versioned surface to depend on.
- [ ] Reference plugin: a ≤ 50-line third-party pipeline package, installed alongside core, appearing
      in `GET /api/v1/pipelines` and runnable through the standard CLI/API with zero core changes.

**Reference**: spec 003 FR-006, FR-007, FR-008, FR-010; SC-004.

**Still deferred to v2.0** (per spec 003 assumptions, unchanged): isolating Ultralytics (AGPL-3.0)
person-tracking into its own plugin package.

---

### Phase 2: Local LLM/VLM Backend (Ollama / llama.cpp)

**Status**: mostly done — the `vlm_annotation` pipeline landed on `v1.5.0` ahead of the rest of this
release, driven by a real user (a PhD researcher's touch-detection study). See
[`vlm_annotation_pipeline.md`](vlm_annotation_pipeline.md) for the full writeup, config reference, and
a step-by-step testing guide. Checklist below updated to reflect what actually shipped vs. what's
still open.

**Problem**: VideoAnnotator's current model-loading pattern is in-process — pipelines call
`transformers.AutoModel.from_pretrained(...)` directly (e.g.
[`laion_face_pipeline.py`](../../src/videoannotator/pipelines/face_analysis/laion_face_pipeline.py)),
which means every capability upgrade means a new bundled, versioned model dependency. Meanwhile
capable open local models (e.g. the Qwen3 family) are now practically runnable on a researcher's own
machine via `ollama serve` or a `llama.cpp` server, without VideoAnnotator vendoring any weights.

**Solution**:
- [x] New pipeline family — landed as `vlm_annotation` (generic per-frame VLM classification/
      captioning driven by a user-supplied prompt, not scoped to a single task), not
      `scene_description` as originally sketched here.
- [ ] One connector, not two: **deviation, not done as originally scoped.** `vlm_annotation` calls the
      `ollama` Python package's `Client.chat()` directly rather than a hand-rolled OpenAI-compatible
      `/v1/chat/completions` client — reusing the already-proven client de-risked landing a working
      pipeline quickly. `backends: [ollama]` in the metadata leaves room for an `openai_compatible`
      backend (covering both Ollama's and `llama.cpp`'s compatible endpoint) alongside it later
      without a pipeline redesign, but that second connector is still unbuilt.
- [x] Gated behind a new `llm` extras group (`ollama` client + `cv2`; `httpx` was already core, so
      needed no new declaration).
- [x] Uses `backends: list[str]` (`backends: [ollama]`) in the pipeline's metadata YAML.
- [x] Configurable base URL and model name (defaults `http://127.0.0.1:11434` / `qwen3.5:9b`); no
      model weights are bundled or auto-pulled — `model` is just a default config value the user can
      override, same as any other field.
- [ ] `videoannotator diagnose` does **not** yet detect a reachable Ollama server — still open.
- [x] Provenance: every annotation record carries `model`, `backend`, `base_url`, and `prompt`
      directly (per-record, not just job-level metadata) — see Principle III in the constitution.

**Also fixed along the way (not originally scoped here)**: `api/job_processor.py` was silently
dropping per-pipeline job config (`pipeline_class()` with no config, vs.
`batch/batch_orchestrator.py`'s correct `pipeline_class(pipeline_config)`) — jobs submitted via the
API would have ignored `vlm_annotation`'s prompt/model/sampling-mode config entirely. Fixed as part of
landing this pipeline; see `vlm_annotation_pipeline.md`'s "Known gaps" section for the broader
dual-job-execution-engine issue this points at, which Phase 1's job-execution-path consolidation
(above) should resolve properly.

**Still not built**: the job-creation UX and review UI in `video-annotation-viewer` (prompt textarea,
sampling-mode picker, ELAN ground-truth comparison, cross-prompt disagreement view) — tracked as
Phases 3–4 in `vlm_annotation_pipeline.md`, not in this repo.

**Why this matters for JOSS**: it's a genuine, defensible "why does this exist now" story — this
capability wasn't practical for most researchers' hardware a year ago — and it demonstrates the
plugin/backend seam from Phase 1 is real, not aspirational.

---

### Phase 3: Server-Side Timeseries Post-Processing (pandas)

**Problem**: Per-frame annotations (person bounding boxes/keypoints) ship as static per-frame COCO
records; anything temporal — e.g. how far a tracked person moved between frames — is currently left
to Video Annotation Viewer to compute client-side, where compute is more constrained than the
VideoAnnotator server.

**Solution**:
- [ ] A `person_tracking` post-processing step that computes frame-to-frame movement deltas
      (bbox-center or keypoint-centroid displacement) per `person_id`/`track_id`, using `pandas` —
      already a core, no-extras dependency (confirmed unused in any active pipeline code as of
      v1.5.0, so this adds zero new dependency weight, just a new use of an existing one).
- [ ] Additive-only export fields (e.g. `movement_delta_x`, `movement_delta_y`, `movement_speed`)
      alongside the existing COCO annotation, not replacing anything — v1.5.x consumers unaware of
      the new fields are unaffected (Principle V, Backward Compatibility by Default).
- [ ] Scope to person-tracking first (clearest "movement" semantics); revisit whether
      face/gaze/other keypoint-bearing pipelines want the same treatment once the pattern is
      proven.

**Reference**: raised in v1.5.0's merge follow-up discussion, not part of spec 003/004 — no
existing FR/SC coverage. Write a proper spec before implementing if this grows beyond a small,
additive post-processing step.

---

## ✅ Success Criteria

- [ ] A third-party plugin package (≤ 50 lines) is installable and runnable through the standard
      CLI/API with zero VideoAnnotator core changes.
- [ ] The CLI batch path and the API job-submission path share one job-execution function.
- [ ] A local-LLM pipeline runs end-to-end against a user-provided `ollama serve` instance with no
      bundled model weights and no torch/transformers pulled in by the `llm` extra.
- [ ] Provenance metadata for local-LLM pipeline runs records backend, base URL, and model identity.
- [ ] `person_tracking` output includes frame-to-frame movement-delta fields via a pandas-based
      post-processing step, additive-only (no v1.5.x output field ever removed or renamed).

---

**Last Updated**: 2026-07-08
**Target Release**: After v1.5.0
**Status**: Planning Phase — Plugin Ecosystem & Local LLM Focus
