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

**Target Release**: Alongside/ahead of the JOSS resubmission (Phase 2 — /viewer integration —
shipped in v1.4.4; Phase 1 — extras-based install — shipped in v1.5.0, 2026-07-19)
**Current Status**: v1.5.0 released
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
- [x] Move heavy ML deps from `[project.dependencies]` into named `[project.optional-dependencies]`
      groups: `face`, `audio`, `scene`, `person` (research.md §1 supersedes the originally-sketched
      `embedding` group — no pipeline uses `pipeline_family: embedding`, so it was never created;
      `face-openface3`/`face-laion`/`audio-laion` cover the embedding-shaped pipelines instead).
- [x] Give LAION its own extras (`face-laion`, `audio-laion`) separate from the base `face`/`audio`
      groups, so standard face/audio pipelines don't pull in LAION's dependencies. LAION stays fully
      supported, just not mandatory.
- [x] Add an `all` meta-extra that reproduces v1.4.4's install footprint exactly.
- [x] Replace the hardcoded `LEGACY_MAPPINGS` path in
      [`registry/pipeline_loader.py`](../../src/videoannotator/registry/pipeline_loader.py) with
      fully metadata-driven loading — per-pipeline YAML already carries `module_path`; add
      `requires_extras`.
- [x] Registry omits pipelines whose extras aren't installed rather than crashing; CLI/API error
      messages include the exact `pip install videoannotator[...]` command needed.
- [x] Migration message for v1.4.x users whose config references a pipeline (e.g. LAION) that is no
      longer in the default install.
- [x] Drop the `numpy<2.0` pin (`pyproject.toml`) — numpy is now unpinned at the lower bound and
      resolves per `numba`'s own declared ceiling (numpy 2.2.x as of this writing); full suite
      (1065 passed, 33 skipped) verified green under numpy 2.2.6 with `[all]` extras installed.
- [x] Slim `Dockerfile.cpu`/`Dockerfile.gpu` base image (no extras); document the `[all]` image
      separately.

**Acceptance**: v1.4.3 acceptance-test fixtures pass byte-identically (or within documented tolerance)
against a v1.5 `[all]` install with zero config-file changes. See spec 003 User Story 2 for the full
backward-compatibility test plan.

**Reference**: [`specs/003-modular-pipeline-architecture/spec.md`](../../specs/003-modular-pipeline-architecture/spec.md)
FR-001 through FR-005, FR-009, FR-011, FR-014, FR-015; SC-001, SC-002, SC-003, SC-005, SC-008, SC-009.
Implementation-ready slice: [`specs/004-extras-based-install/spec.md`](../../specs/004-extras-based-install/spec.md).

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

- [x] `pip install videoannotator` (no extras) installs without torch/transformers/pyannote/ultralytics/whisper.
      Verified: `[project.dependencies]` contains none of them; confirmed via real `pip install
      videoannotator[scene]`/`[face]` runs during 004's manual quickstart pass (§1/§2) that only the
      requested family's deps land, nothing else.
- [ ] `pip install videoannotator[all]` reproduces v1.4.3 behaviour exactly; no config/CLI changes needed.
      Mechanism exists (`tests/integration/test_v144_parity.py`, run and passing — 4 passed, 6
      skipped) but the skips are real: no v1.4.4 golden fixtures have been captured yet (needs
      checking out the v1.4.4 tag and running real model inference to generate them — substantial,
      not done in this pass). See the test file's own module docstring for the capture procedure.
- [ ] Default Docker image size reduced by ≥ 80% vs. the v1.4.3 baseline. Not measurable in this
      sandbox (no `docker` binary available) — needs an environment with Docker to build
      `Dockerfile.cpu`/`Dockerfile.gpu` both slim and `--build-arg EXTRAS=all`, and compare.
- [x] A user with no extras installed gets an actionable install command, not a traceback, when
      requesting an unavailable pipeline. Verified: `tests/contract/test_unavailable_pipeline_error.py`
      + `test_pipeline_availability_contract.py` pass (7 tests), and confirmed live via a real running
      server returning `422` + `install_hint` for an unavailable pipeline (004's manual quickstart §1).
- [~] `pip install videoannotator[all]` + `videoannotator server --dev` gets a user from install to a
      reviewable annotated video, without cloning a second repository. (Roadmap text said `serve`;
      the actual command is `server`.) Substantially verified: a real install + running server +
      `job submit` produced valid COCO output for `scene_detection` and `face_analysis`, and the
      viewer mounts at `/viewer` same-origin. Not yet confirmed: actually opening `/viewer` in a
      browser and seeing that output rendered — no browser available in this sandbox to check that
      last step.
- [x] Full test suite passes on NumPy 2.x — **on Linux**. Verified repeatedly this session (most
      recently 1066 passed, 32 skipped, 0 failed under numpy 2.2.6 with `[all]` extras). **Not**
      verified on macOS/Windows — this sandbox is Linux-only; needs the real CI matrix
      (`.github/workflows/ci-cd.yml`'s `test` job already covers all three OSes, just needs a run
      against the numpy-unpinned state to confirm).
- [~] Cross-repo contract test between VideoAnnotator and Video Annotation Viewer is running in CI.
      Python-side half: yes — `tests/contract/test_viewer_contract.py` passes (4 tests) and runs
      automatically in the existing CI `test` job, no workflow changes needed. TypeScript-side half
      (VAV's own parsers against these fixtures): still deferred, needs Node/bun tooling not
      available in this sandbox — see Phase 2's own checklist above.

---

**Last Updated**: 2026-07-19
**Target Release**: Ahead of / alongside JOSS resubmission
**Status**: v1.5.0 released — Phase 1 (extras-based install) complete
