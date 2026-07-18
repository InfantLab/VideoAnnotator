---
description: "Task list for Extras-Based Modular Install & Registry Refactor"
---

# Tasks: Extras-Based Modular Install & Registry Refactor

**Input**: Design documents from `/specs/004-extras-based-install/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Included. Not explicitly requested in spec.md's own text, but the ratified
constitution's Engineering Standards ("New pipelines and new public CLI/API surface MUST ship
with tests", coverage ≥80%) makes this a standing repo-wide requirement, not an opt-in.

**Organization**: Tasks are grouped by user story (US1, US2, US3 — spec.md priorities P1, P1, P2
respectively) to enable independent implementation and testing of each.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)

## Path Conventions

Single project (`src/videoannotator/`, `tests/`) per plan.md's Structure Decision.

---

## Phase 1: Setup

**Purpose**: Declare the extras groups so later phases have somewhere to put dependencies.

- [ ] T001 [P] Define `[project.optional-dependencies]` groups in `pyproject.toml` — `face`
      (deepface, opencv-python, imutils), `face-laion` (torch, torchvision, transformers,
      huggingface-hub), `face-openface3` (openface-test, scipy), `audio` (torch, torchaudio,
      librosa, openai-whisper, pyannote.audio+core+database+metrics+pipeline), `audio-laion`
      (torch, transformers, huggingface-hub, librosa), `scene` (torch, open-clip-torch,
      scenedetect[opencv]), `person` (torch, torchvision, ultralytics, supervision), `all`
      (unions all of the above). Remove the corresponding entries from `[project.dependencies]`.
      Per research.md §1 — `face` (DeepFace variant) needs no torch; every other group does.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Metadata-driven registry loading. MUST complete before any user story — every story
depends on the registry being able to read `requires_extras`/`module_path` from YAML instead of
`LEGACY_MAPPINGS`.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T002 Add `requires_extras: list[str]` field (default `field(default_factory=list)`) to the
      `PipelineMetadata` dataclass and its YAML-parsing logic in
      `src/videoannotator/registry/pipeline_registry.py`, per
      `specs/004-extras-based-install/contracts/pipeline-metadata-schema.md`
- [ ] T003 [P] Add `module_path: videoannotator.pipelines.face_analysis:FaceAnalysisPipeline` and
      `requires_extras: [face]` to `src/videoannotator/registry/metadata/face_analysis.yaml`
- [ ] T004 [P] Add `module_path:
      videoannotator.pipelines.face_analysis.laion_face_pipeline:LAIONFacePipeline` and
      `requires_extras: [face-laion]` to
      `src/videoannotator/registry/metadata/face_laion_clip.yaml`
- [ ] T005 [P] Add `module_path:
      videoannotator.pipelines.face_analysis.openface3_pipeline:OpenFace3Pipeline` and
      `requires_extras: [face-openface3]` to
      `src/videoannotator/registry/metadata/face_openface3_embedding.yaml`
- [ ] T006 [P] Add `module_path:
      videoannotator.pipelines.audio_processing:AudioPipeline` and `requires_extras: [audio]` to
      `src/videoannotator/registry/metadata/audio_processing.yaml`
- [ ] T007 [P] Add `module_path:
      videoannotator.pipelines.audio_processing:SpeechPipeline` and `requires_extras: [audio]` to
      `src/videoannotator/registry/metadata/speech_recognition.yaml`
- [ ] T008 [P] Add `module_path:
      videoannotator.pipelines.audio_processing:DiarizationPipeline` and
      `requires_extras: [audio]` to
      `src/videoannotator/registry/metadata/speaker_diarization.yaml`
- [ ] T009 [P] Add `module_path:
      videoannotator.pipelines.audio_processing.laion_voice_pipeline:LAIONVoicePipeline` and
      `requires_extras: [audio-laion]` to `src/videoannotator/registry/metadata/laion_voice.yaml`
- [ ] T010 [P] Add `module_path:
      videoannotator.pipelines.scene_detection:SceneDetectionPipeline` and
      `requires_extras: [scene]` to `src/videoannotator/registry/metadata/scene_detection.yaml`
- [ ] T011 [P] Add `module_path:
      videoannotator.pipelines.person_tracking:PersonTrackingPipeline` and
      `requires_extras: [person]` to `src/videoannotator/registry/metadata/person_tracking.yaml`
- [ ] T012 Implement an extras-availability check helper (`importlib.util.find_spec`-based, per
      research.md §3) in `src/videoannotator/registry/pipeline_loader.py` — depends on T002
- [ ] T013 Remove `LEGACY_MAPPINGS` and `_infer_module_path` from
      `src/videoannotator/registry/pipeline_loader.py`; resolve `module_path` purely from
      metadata; warn-and-skip (not crash) a pipeline whose YAML lacks `module_path` — depends on
      T003–T011 (every YAML must carry `module_path` before the fallback can be safely deleted)
- [ ] T014 [P] Unit tests for `requires_extras` parsing, `module_path`-required validation, and
      the extras-availability helper in `tests/unit/registry/test_pipeline_registry.py` — depends
      on T002, T012

**Checkpoint**: Registry is fully metadata-driven; `LEGACY_MAPPINGS` is gone. User story
implementation can now begin.

---

## Phase 3: User Story 1 - Slim, targeted install for a single research need (Priority: P1) 🎯 MVP

**Goal**: `pip install videoannotator[scene]` (or any single family) pulls only that family's
deps; unavailable pipelines are omitted from listings and produce actionable errors, not
tracebacks.

**Independent Test**: quickstart.md §1–2 — install `[scene]` in a clean venv, confirm
torch/open-clip present and pyannote/ultralytics/deepface absent, run scene labelling
successfully, confirm requesting `face_analysis` gives an actionable error.

### Tests for User Story 1 ⚠️

- [ ] T015 [P] [US1] Contract test: unavailable pipelines omitted from `GET /api/v1/pipelines`
      default listing in `tests/contract/test_pipeline_availability_contract.py`
- [ ] T016 [P] [US1] Contract test: CLI and API unavailable-pipeline error shape (API: `422` +
      `install_hint` field; CLI: non-zero exit, no traceback) per
      `specs/004-extras-based-install/contracts/unavailable-pipeline-error.md` in
      `tests/contract/test_unavailable_pipeline_error.py`
- [ ] T017 [P] [US1] Integration test: subprocess-driven extras isolation — install
      `videoannotator[scene]` into a clean venv, assert `torch`/`open-clip-torch` importable and
      `pyannote.audio`/`ultralytics`/`deepface` are not, in
      `tests/integration/test_extras_isolation.py`

### Implementation for User Story 1

- [ ] T018 [US1] Wire the extras-availability check (T012) into
      `PipelineLoader.load_all_pipelines()` so it's consulted before attempting each pipeline's
      import — depends on T012
- [ ] T019 [US1] `PipelineRegistry`/`PipelineLoader`: omit pipelines whose extras aren't installed
      from default `list()`/`load_all_pipelines()` output, per FR-005 — depends on T018
- [ ] T020 [US1] `GET /api/v1/pipelines` excludes unavailable pipelines by default; add
      `?include_unavailable=true` returning `available: false` + `install_hint` per pipeline in
      `src/videoannotator/api/` — depends on T019
- [ ] T021 [US1] `videoannotator pipelines list` (CLI) reflects the same availability filtering in
      `src/videoannotator/cli.py` — depends on T019
- [ ] T022 [US1] Job submission (CLI `job submit` and API `POST /api/v1/jobs`) returns the
      actionable error contract (exact `pip install videoannotator[...]` command, `422` not
      `500`, no Python traceback) when a requested pipeline is unavailable — depends on T019
- [ ] T023 [US1] Slim `Dockerfile.cpu` and `Dockerfile.gpu` to build with no extras by default;
      document the `[all]`-equivalent build variant (research.md §6)

**Checkpoint**: User Story 1 fully functional and independently testable — this is the MVP slice.

---

## Phase 4: User Story 2 - Backward-compatible upgrade for existing studies (Priority: P1)

**Goal**: `pip install videoannotator[all]` reproduces v1.4.4 behaviour exactly; a v1.4.x config
referencing a now-demoted pipeline (LAION, openface3) gets a specific migration message; full
suite passes on NumPy 2.x and the `<2.0` pin is dropped.

**Independent Test**: quickstart.md §3–5 — `[all]` install passes v1.4.4 acceptance fixtures
byte-identical (or within documented tolerance); an old LAION-referencing config produces the
migration message, not a raw ImportError; suite passes with `numpy>=2.0` installed.

### Tests for User Story 2 ⚠️

- [ ] T024 [P] [US2] Contract test: migration-message variant (names the extras group, states
      "no longer installed by default as of v1.5.0", distinct from the generic unavailable-error
      text) in `tests/contract/test_migration_message_contract.py`
- [ ] T025 [P] [US2] v1.4.4 acceptance-fixture parity test under a `[all]` install —
      byte-identical output for deterministic pipelines, documented tolerance for
      non-deterministic ones (FR-009) — in `tests/integration/test_v144_parity.py`
- [ ] T026 [P] [US2] Model-cache reuse test: a pre-existing HF/torch cache directory is reused,
      not re-downloaded, under the new extras-aware loader in
      `tests/integration/test_model_cache_reuse.py`

### Implementation for User Story 2

- [ ] T027 [US2] Wire the migration-message lookup (data-model.md's migration table:
      `face_laion_clip`→`face-laion`, `laion_voice`→`audio-laion`,
      `face_openface3_embedding`→`face-openface3`) into the same error path as T022, so demoted
      pipelines get the distinct message — depends on T022
- [ ] T028 [US2] Add a NumPy 2.x CI job variant alongside the existing job in
      `.github/workflows/ci-cd.yml` (research.md §5)
- [ ] T029 [US2] Fix NumPy 2.x incompatibilities surfaced by T028 (watch specifically for
      `pyannote`/`scipy`/`matplotlib` binary-wheel issues flagged in the current
      `pyproject.toml:43` pin comment); iterate until the NumPy-2.x job is green — depends on T028
- [ ] T030 [US2] Remove the `numpy<2.0` pin from `pyproject.toml`; retire the NumPy-1.x CI job
      once T029's job has been green for a full run — depends on T029
- [ ] T031 [US2] Write the per-use-case install matrix ("I want only scene labelling", "I want
      everything", "I want a slim API server") in `docs/installation/INSTALLATION.md` (FR-013,
      SC-006)

**Checkpoint**: User Stories 1 AND 2 both independently functional.

---

## Phase 5: User Story 3 - Metadata schema has room for non-local execution (Priority: P2)

**Goal**: Prove the schema changes from Phase 2 don't need to be redone for v1.6.0's Ollama
backend or v1.7+'s remote/HPC dispatch — the actual reason this phase was prioritized now.

**Independent Test**: quickstart.md §6 — a throwaway pipeline YAML with `requires_extras: []`
and a stub `module_path` loads successfully through the registry with zero loader code changes.

### Tests for User Story 3 ⚠️

- [ ] T032 [P] [US3] Unit test: a stub pipeline metadata entry with `requires_extras: []` and a
      `module_path` pointing at a `NotImplementedError` stub class loads via the registry without
      any loader-code changes, in `tests/unit/registry/test_forward_compat_stub.py`

### Implementation for User Story 3

- [ ] T033 [US3] Author the throwaway stub pipeline fixture (`tests/fixtures/stub_pipeline.yaml`
      + a stub module) used by T032, demonstrating `requires_extras: []` plus a non-ML `backends`
      value — depends on T013
- [ ] T034 [US3] Write the SC-007 design-review note — confirm in writing (appended to
      research.md or a new short doc) that adding v1.6.0's `llm` extras group and its pipeline
      entries requires zero changes to `PipelineMetadata`, the registry loader, or the
      extras-naming scheme from Phase 1/2 — only additive YAML + a new
      `[project.optional-dependencies]` entry

**Checkpoint**: All three user stories independently functional.

---

## Final Phase: Polish & Cross-Cutting Concerns

- [ ] T035 [P] Check off the completed items in `docs/development/roadmap_v1.5.0.md`'s Phase 1
      deliverables list to match what shipped
- [ ] T036 Run `pytest tests/`, `mypy src/videoannotator`, `ruff check .`; confirm the ≥80%
      coverage gate (constitution Engineering Standards) still holds
- [ ] T037 Run every section of `specs/004-extras-based-install/quickstart.md` end-to-end
- [ ] T038 [P] Update `CHANGELOG.md` with the v1.5.0 Phase 1 entry

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately.
- **Foundational (Phase 2)**: Depends on Setup (extras groups must exist for `requires_extras`
  values to mean anything) — BLOCKS all user stories.
- **User Stories (Phase 3+)**: All depend on Foundational completion. US1 and US2 (both P1) can
  proceed in parallel if staffed; US3 (P2) can also start in parallel — it only depends on T013
  (LEGACY_MAPPINGS removal), not on US1/US2's own work.
- **Polish (Final Phase)**: Depends on all three user stories being complete.

### User Story Dependencies

- **US1 (P1)**: Starts after Foundational. No dependency on US2/US3.
- **US2 (P1)**: Starts after Foundational. T027 depends on US1's T022 (same error-path code), so
  in practice US2's implementation tasks trail US1's by that one dependency; US2's tests (T024–26)
  and CI/NumPy work (T028–30) have no US1 dependency and can start immediately after Foundational.
- **US3 (P2)**: Starts after Foundational (specifically T013). No dependency on US1/US2.

### Within Each User Story

- Tests written before implementation tasks that make them pass.
- Registry/loader wiring before API/CLI surface changes.
- Story complete and checkpointed before considering the next priority tier done.

### Parallel Opportunities

- T001 (Setup) has nothing to parallelize against within its own phase.
- T003–T011 (9 YAML edits) are fully parallel — independent files.
- T014 can run parallel to T015–T017 once its own dependencies (T002, T012) are met.
- Within US1: T015, T016, T017 (tests) in parallel; T023 (Docker) in parallel with T020/T021/T022
  once T019 lands.
- Within US2: T024, T025, T026 (tests) in parallel; T028 can start immediately after Foundational,
  in parallel with all of US1.
- US1 and US2 can be staffed in parallel after Foundational, with the single T022→T027 handoff
  point noted above.
- US3 can be staffed in parallel with both US1 and US2 after Foundational.

---

## Parallel Example: Foundational Phase

```bash
# After T002 lands, launch all 9 metadata-file edits together:
Task: "Add module_path + requires_extras: [face] to face_analysis.yaml"
Task: "Add module_path + requires_extras: [face-laion] to face_laion_clip.yaml"
Task: "Add module_path + requires_extras: [face-openface3] to face_openface3_embedding.yaml"
Task: "Add module_path + requires_extras: [audio] to audio_processing.yaml"
Task: "Add module_path + requires_extras: [audio] to speech_recognition.yaml"
Task: "Add module_path + requires_extras: [audio] to speaker_diarization.yaml"
Task: "Add module_path + requires_extras: [audio-laion] to laion_voice.yaml"
Task: "Add module_path + requires_extras: [scene] to scene_detection.yaml"
Task: "Add module_path + requires_extras: [person] to person_tracking.yaml"
```

## Parallel Example: User Story 1

```bash
# Launch all US1 tests together:
Task: "Contract test: unavailable pipelines omitted from GET /api/v1/pipelines"
Task: "Contract test: CLI/API unavailable-pipeline error shape"
Task: "Integration test: subprocess-driven extras isolation for [scene]"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001).
2. Complete Phase 2: Foundational (T002–T014) — CRITICAL, blocks everything.
3. Complete Phase 3: User Story 1 (T015–T023).
4. **STOP and VALIDATE**: run quickstart.md §1–2 for real, not just the automated tests.
5. This is a legitimately shippable increment: a user can install a single pipeline family slim
   and get actionable errors for the rest, even before US2/US3 land.

### Incremental Delivery

1. Setup + Foundational → registry is metadata-driven, `LEGACY_MAPPINGS` gone.
2. Add US1 → validate independently → this is the MVP.
3. Add US2 → validate independently → v1.4.4 upgraders are unblocked, NumPy 2.x lands.
4. Add US3 → validate independently → the forward-compatibility guarantee (the actual point of
   doing this work now, per spec.md) is proven, not just asserted.
5. Polish → roadmap doc, changelog, full-suite confirmation.

### Parallel Team Strategy

With more than one developer: complete Setup + Foundational together first (it's genuinely
blocking and mostly sequential/parallel-file-edit work, not parallel-workstream work). Once
Foundational is checkpointed, one developer can take US1, another US2, another US3 — the
dependency notes above show only one real cross-story coupling (US2's T027 needs US1's T022).

---

## Notes

- [P] tasks touch different files with no dependency on incomplete same-phase work.
- Every metadata YAML edit (T003–T011) is independently verifiable: load the registry and confirm
  that one pipeline resolves via metadata, not `LEGACY_MAPPINGS`.
- Commit after each task or logical group, per constitution Development Workflow (atomic commits,
  conventional `<area>: <imperative summary>` messages).
- Avoid combining T013 (LEGACY_MAPPINGS removal) with any of T003–T011 in one commit — keep the
  "every YAML has module_path now" fact independently verifiable before deleting the fallback that
  depends on it.
