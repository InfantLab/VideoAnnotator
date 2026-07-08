# 🚀 VideoAnnotator v1.5.0 Development Roadmap

## Release Overview

VideoAnnotator v1.5.0 is the **Modularity & Integration Release**. It is a direct response to the JOSS
pre-review of VideoAnnotator ([openjournals/joss-reviews#10182](https://github.com/openjournals/joss-reviews/issues/10182))
and Video Annotation Viewer ([#10183](https://github.com/openjournals/joss-reviews/issues/10183)),
where the editor flagged the monolithic install footprint (~30 GB Docker image, every pipeline's
dependencies mandatory) and asked for a clearer, demonstrable relationship between the two projects.

This release scopes down the full modular-pipeline-architecture design in
[`specs/003-modular-pipeline-architecture/spec.md`](../../specs/003-modular-pipeline-architecture/spec.md)
to the subset that matters for the resubmission: a slim, extras-based install, and a runtime
integration between VideoAnnotator and Video Annotation Viewer. The plugin-discovery and
remote-dispatch seams from that spec are real and worth building, but they are not blocking a
resubmission — they move to [`roadmap_v1.6.0.md`](roadmap_v1.6.0.md).

**Target Release**: Alongside/ahead of the JOSS resubmission (v1.4.4 is the current release; the
Phase 2 work below — /viewer integration — shipped in v1.4.4, ahead of the Phase 1 extras-based
install work still to come)
**Current Status**: Planning Phase
**Main Goal**: Slim, per-pipeline install; LAION demoted from the default install; VideoAnnotator can
optionally serve Video Annotation Viewer directly
**Constitution Principle in play**: Principle V (Backward Compatibility by Default) — v1.4.x
configs and CLI invocations must keep working unchanged

**Prerequisites**: v1.4.3 fixed the reviewer-blocking installability bugs (missing `storage/manager.py`
source files, CUDA-only torch pin, stale `openai-whisper` pin) — see commits `f3a9676`..`d55bc56`.

---

## 🎯 Core Principles

- ✅ **Slim by default** — `pip install videoannotator` with no extras must not pull in torch,
  transformers, pyannote, ultralytics, or whisper.
- ✅ **Additive, not breaking** — `pip install videoannotator[all]` reproduces v1.4.3 behaviour
  exactly; existing config files and CLI invocations are untouched.
- ✅ **One integrated onboarding path** — a user can go from `pip install` to reviewing annotated
  output in a browser without standing up a second project by hand.
- ✅ **LAION is opt-in, not default** — LAION Empathic-Insight (face and voice emotion) is an
  actively-changing external project with a narrow user base; it should not tax every install.

**Out of scope for v1.5.0** (moved to v1.6.0 — see below):
- ❌ Third-party plugin discovery via `importlib.metadata.entry_points`
- ❌ Dispatcher ABC / HTTPDispatcher / SlurmDispatcher seam
- ❌ Cross-cutting utilities split into a sibling `videoannotator-utils` package
- ❌ Ollama / llama.cpp local LLM pipeline backend

**Deferred, unscheduled** (the original v1.5.0 UX wishlist — setup wizard, progress indicators,
FiftyOne/Label Studio export, quality-assessment tooling, etc.) has been archived to
[`docs/archive/development/roadmap_v1.5.0_ux_wishlist.md`](../archive/development/roadmap_v1.5.0_ux_wishlist.md).
None of it shipped; it has no committed release slot and should be re-triaged into v1.7.0+ once this
release lands.

---

## 📋 v1.5.0 Deliverables

### Phase 1: Extras-Based Install (User Stories 1 & 2 of spec 003)

**Problem**: Every pipeline's dependencies (torch, ultralytics, pyannote, whisper, transformers,
open-clip, LAION's stack) are hard dependencies in `pyproject.toml`. A user who only wants scene
labelling still downloads everything.

**Solution**:
- [ ] Move heavy ML deps from `[project.dependencies]` into named `[project.optional-dependencies]`
      groups: `face`, `audio`, `scene`, `person`, `embedding`.
- [ ] Give LAION its own extras (e.g. `face-laion`, `voice-laion`) separate from the base `face`/`audio`
      groups, so standard face/audio pipelines don't pull in LAION's dependencies. LAION stays fully
      supported, just not mandatory.
- [ ] Add an `all` meta-extra that reproduces v1.4.3's install footprint exactly.
- [ ] Replace the hardcoded `LEGACY_MAPPINGS` path in
      [`registry/pipeline_loader.py`](../../src/videoannotator/registry/pipeline_loader.py) with
      fully metadata-driven loading — per-pipeline YAML already carries `module_path`; add
      `requires_extras`.
- [ ] Registry omits pipelines whose extras aren't installed rather than crashing; CLI/API error
      messages include the exact `pip install videoannotator[...]` command needed.
- [ ] Migration message for v1.4.x users whose config references a pipeline (e.g. LAION) that is no
      longer in the default install.
- [ ] Drop the `numpy<2.0` pin (`pyproject.toml:43`) once the pipeline suite is verified against
      NumPy 2.x.
- [ ] Slim `Dockerfile.cpu`/`Dockerfile.gpu` base image (no extras); document the `[all]` image
      separately.

**Acceptance**: v1.4.3 acceptance-test fixtures pass byte-identically (or within documented tolerance)
against a v1.5 `[all]` install with zero config-file changes. See spec 003 User Story 2 for the full
backward-compatibility test plan.

**Reference**: [`specs/003-modular-pipeline-architecture/spec.md`](../../specs/003-modular-pipeline-architecture/spec.md)
FR-001 through FR-005, FR-009, FR-011, FR-014, FR-015; SC-001, SC-002, SC-003, SC-005, SC-008, SC-009.

---

### Phase 2: FastAPI Serves Video Annotation Viewer

**Problem**: The two JOSS submissions are architecturally separate by design (server vs. review UI),
but that separation currently reads as two disconnected projects. A reviewer or new user has to stand
up both independently to see the full workflow, and the pre-review discussion showed the editor
struggling to understand how they relate.

**Solution**:
- [x] Video Annotation Viewer's static production build is vendored into VideoAnnotator's package
      data (`src/videoannotator/viewer_static/`). VAV now builds with a configurable `--base=/viewer/`
      and reads `basename` from `import.meta.env.BASE_URL`, so vendoring is a straight copy — no
      path rewriting needed on the VideoAnnotator side. A JS bundle is a few MB — nowhere near the
      multi-GB ML-dependency problem this release is otherwise about — so it ships in the core wheel;
      no pip extra needed to gate it.
- [x] FastAPI app mounts the viewer's static assets at `/viewer` by default, pre-configured to talk
      to the local API (same-origin, so no CORS/base-URL configuration needed at runtime); the
      `VIDEOANNOTATOR_ENABLE_VIEWER` env var disables it for anyone who doesn't want it.
- [x] `videoannotator server` prints the viewer URL when it's mounted.
- [x] Video Annotation Viewer remains fully independent: standalone use (drag in a video + any
      COCO/WebVTT/RTTM/scene-JSON file, no server required) is unaffected and stays the primary
      documented use case. Demo content is now fetched at runtime from VAV's own repo rather than
      bundled, so this holds for the embedded copy too.
- [x] Python-side half: `tests/contract/test_viewer_contract.py` generates real fixtures via
      VideoAnnotator's own exporters and validates them against VAV's actual Zod schemas (modeled in
      Python, sourced from `video-annotation-viewer`'s `src/lib/validation.ts`). Runs automatically in
      the existing CI test job — no workflow changes needed. This already found two real contract
      gaps (see below).
- [ ] TypeScript-side half (deferred): a CI job that checks out VAV and runs its actual parsers
      against these fixtures. Needs Node/bun tooling to build and verify — out of scope for this
      environment; do this as a follow-up once the Python-side gaps below are resolved.
- [x] Single quickstart doc covering install → process a sample video → open the viewer
      ([`docs/usage/GETTING_STARTED.md`](../usage/GETTING_STARTED.md)).

**Why this matters for JOSS**: this is a runnable artifact that answers "how do these relate" more
convincingly than paper prose alone, and it's the lowest-effort, highest-signal change available
before resubmission.

---

## ✅ Success Criteria

- [ ] `pip install videoannotator` (no extras) installs without torch/transformers/pyannote/ultralytics/whisper.
- [ ] `pip install videoannotator[all]` reproduces v1.4.3 behaviour exactly; no config/CLI changes needed.
- [ ] Default Docker image size reduced by ≥ 80% vs. the v1.4.3 baseline.
- [ ] A user with no extras installed gets an actionable install command, not a traceback, when
      requesting an unavailable pipeline.
- [ ] `pip install videoannotator[all]` + `videoannotator serve` gets a user from install to a
      reviewable annotated video in one command, without cloning a second repository.
- [ ] Full test suite passes on NumPy 2.x, Linux/macOS/Windows.
- [ ] Cross-repo contract test between VideoAnnotator and Video Annotation Viewer is running in CI.

---

**Last Updated**: 2026-07-08
**Target Release**: Ahead of / alongside JOSS resubmission
**Status**: Planning Phase — Modularity & Integration Focus
