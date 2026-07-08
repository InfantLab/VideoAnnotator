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

**Problem**: VideoAnnotator's current model-loading pattern is in-process — pipelines call
`transformers.AutoModel.from_pretrained(...)` directly (e.g.
[`laion_face_pipeline.py`](../../src/videoannotator/pipelines/face_analysis/laion_face_pipeline.py)),
which means every capability upgrade means a new bundled, versioned model dependency. Meanwhile
capable open local models (e.g. the Qwen3 family) are now practically runnable on a researcher's own
machine via `ollama serve` or a `llama.cpp` server, without VideoAnnotator vendoring any weights.

**Solution**:
- [ ] New pipeline family (e.g. `scene_description` / VLM captioning, exact scope TBD) that talks to
      a local model server over HTTP instead of loading a model in-process.
- [ ] One connector, not two: both Ollama and `llama.cpp`'s server mode expose an
      OpenAI-compatible `/v1/chat/completions` endpoint — implement a single client against that
      contract rather than bespoke integrations per backend.
- [ ] Gate this pipeline behind a new lightweight `llm` extras group that pulls in only an HTTP
      client (e.g. `httpx`), not torch/transformers — consistent with v1.5.0's slim-install goal.
- [ ] Use the `backends: list[str]` field already present in
      [`PipelineMetadata`](../../src/videoannotator/registry/pipeline_registry.py) to mark
      backend type (`ollama`, `llamacpp`) and let the registry surface it.
- [ ] Configurable base URL and model name (default `http://localhost:11434` for Ollama); no default
      model is bundled or auto-pulled.
- [ ] Extend `videoannotator diagnose` to detect a reachable local model server, the way it currently
      detects GPU availability.
- [ ] Document the tradeoff plainly: local-LLM pipelines depend on a server the user runs themselves
      and are not deterministic/reproducible in the same sense as the bundled, versioned pipelines —
      output provenance metadata should record the backend, base URL, and reported model name/version
      so runs stay traceable.

**Why this matters for JOSS**: it's a genuine, defensible "why does this exist now" story — this
capability wasn't practical for most researchers' hardware a year ago — and it demonstrates the
plugin/backend seam from Phase 1 is real, not aspirational.

---

## ✅ Success Criteria

- [ ] A third-party plugin package (≤ 50 lines) is installable and runnable through the standard
      CLI/API with zero VideoAnnotator core changes.
- [ ] The CLI batch path and the API job-submission path share one job-execution function.
- [ ] A local-LLM pipeline runs end-to-end against a user-provided `ollama serve` instance with no
      bundled model weights and no torch/transformers pulled in by the `llm` extra.
- [ ] Provenance metadata for local-LLM pipeline runs records backend, base URL, and model identity.

---

**Last Updated**: 2026-07-08
**Target Release**: After v1.5.0
**Status**: Planning Phase — Plugin Ecosystem & Local LLM Focus
