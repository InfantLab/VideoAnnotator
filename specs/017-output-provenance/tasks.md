# Tasks: Output Provenance and Overlay Attribution

**Input**: Design documents from `specs/017-output-provenance/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/provenance-in-files.md

**Tests**: included. The constitution requires tests for every new API, CLI and file-format
surface, and FR-006/FR-007/FR-014 are test-defined.

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [x] T001 Copy the current viewer contract fixtures (no provenance) to `tests/fixtures/viewer_contract/legacy/` with a README line saying they are pre-provenance outputs kept for backward-compatibility tests

## Phase 2: Foundational (blocks every story)

- [x] T002 Create `src/videoannotator/provenance.py`: `ModelRef` and `build_record(pipeline_name, *, models, settings, determinism, job_id, input_name, input_sha256, vlm=None, sub_pipeline=None) -> dict` (schema_version 1, `created_at` UTC with `+00:00`, `videoannotator_version`), per data-model.md
- [x] T003 In `src/videoannotator/provenance.py` add `redact(settings)` (keys matching token|secret|password|api_key|apikey|auth at any depth → `"<redacted>"`) and `file_sha256(path)` with a per-process cache keyed by (resolved path, size, mtime)
- [x] T004 In `src/videoannotator/provenance.py` add `stamp_file(path, record)`: JSON → top-level `provenance` key (rewrite preserving the rest, `indent=2` as written today); `.vtt` → insert `NOTE videoannotator-provenance <json>` block after the header with `-->` escaped as `-->`, replacing an existing one; `.rttm` → write `<name>.provenance.json` beside it; and `read_record(path)` for all three
- [x] T005 [P] Unit tests for T002–T004 in `tests/unit/test_provenance.py` (record fields, redaction at depth, hash cache hit, each format round-trips, re-stamping replaces rather than duplicates, `-->` escaping)
- [x] T006 Add `provenance_models(self) -> list[ModelRef]` returning `[]` to `src/videoannotator/pipelines/base_pipeline.py`

## Phase 3: User Story 1 - A result file says what made it (P1) 🎯 MVP

**Goal**: every output file of a job carries a complete record; standard readers still work.
**Independent Test**: run the demo clip through all pipelines; each file has a record with every FR-002 field; pycocotools, a WebVTT parser and pyannote's RTTM loader read the files as before.

- [x] T007 [P] [US1] `provenance_models()` for YOLO in `src/videoannotator/pipelines/person_tracking/person_pipeline.py` (sha256 of the loaded `.pt`)
- [x] T008 [P] [US1] `provenance_models()` for OpenFace 3's three weights in `src/videoannotator/pipelines/face_analysis/openface3_pipeline.py`
- [x] T009 [P] [US1] `provenance_models()` for DeepFace attribute models and detector backend in `src/videoannotator/pipelines/face_analysis/face_pipeline.py`
- [x] T010 [P] [US1] `provenance_models()` for open_clip (model, pretrained tag, checkpoint sha256) in `src/videoannotator/pipelines/scene_detection/scene_pipeline.py`
- [x] T011 [P] [US1] `provenance_models()` for Whisper (size, checkpoint sha256) in `src/videoannotator/pipelines/audio_processing/speech_pipeline.py`
- [x] T012 [P] [US1] `provenance_models()` for pyannote (repo, Hub commit from the cached snapshot) in `src/videoannotator/pipelines/audio_processing/diarization_pipeline.py`
- [x] T013 [P] [US1] `provenance_models()` for `audio_processing` (union of its sub-pipelines) in `src/videoannotator/pipelines/audio_processing/audio_pipeline_modular.py`
- [x] T014 [US1] VLM: add `model_details(model)` (digest, quantisation) to `src/videoannotator/pipelines/vlm_annotation/ollama_client.py`; `provenance_models()` and a `provenance_vlm()` (prompt sha256, digest, quantisation, base_url) in `vlm_pipeline.py`
- [x] T015 [US1] In `src/videoannotator/batch/job_execution.py`: hash the input video once per job; after `initialize()` build the record (models, `redact(pipeline.config)`, determinism from `apply_torch_settings`, job id); after `process()` stamp every file from the registry `outputs[].file` list that exists (for `audio_processing`, set `sub_pipeline` per file)
- [x] T016 [US1] Real `date_created`/`year` in `src/videoannotator/exporters/native_formats.py`
- [x] T017 [US1] Include companion `.provenance.json` files in `pipeline_result_files()` in `src/videoannotator/batch/result_files.py`
- [x] T018 [P] [US1] Contract tests in `tests/contract/test_output_readers.py`: stamped COCO loads with `pycocotools.COCO`; stamped VTT parses (cue count/text unchanged) with the `webvtt` reader used by the exporters' tests, or a strict line parser if none is installed; RTTM loads with pyannote's `load_rttm` when installed (skip otherwise)
- [x] T019 [US1] Update `tests/integration/test_output_baseline.py` to drop `provenance`, `info.date_created`, `info.year`, the VTT provenance NOTE and companion files before comparing; check every compared file has a record
- [x] T020 [P] [US1] Job-path test in `tests/unit/batch/test_job_execution.py`: a fake pipeline writing a JSON, a VTT and an RTTM file per a fake registry entry gets all three stamped, with deterministic flag and redacted secret

## Phase 4: User Story 2 - Every overlay in the viewer names its source (P1)

**Goal**: each overlay shows "<pipeline> · VideoAnnotator <version>", details on demand.
**Independent Test**: open the regenerated fixtures in the viewer: every overlay labelled; details show the record verbatim.

- [x] T021 [US2] Types `ProvenanceRecord`, `ProvenanceInfo` in `viewer/src/types/annotations.ts`; `StandardAnnotationData.provenance?`
- [x] T022 [US2] `viewer/src/lib/provenance.ts`: `fromJSON(data)` (top-level `provenance`, else COCO `info.version` → partial, else none), `fromWebVTT(text)`, `fromCompanion(text)`, `attributionLabel(info, pipeline)`
- [x] T023 [US2] In `viewer/src/lib/parsers/merger.ts` collect provenance per track: JSON and VTT from the file being parsed; for RTTM, a file named `<rttm name>.provenance.json` in the same set (detectFileType classifies `*.provenance.json` as `unknown` with confidence 0.9 so it's never parsed as annotations)
- [x] T024 [US2] Attribution line and info popover per overlay in `viewer/src/components/UnifiedControls.tsx`; ELAN shows "Ground truth · <file>"
- [x] T025 [P] [US2] Vitest: `viewer/src/test/lib/provenance.test.ts` (each source, partial, none, `-->` in VTT JSON)
- [x] T026 [P] [US2] Vitest: `viewer/src/test/components/UnifiedControls.provenance.test.tsx` (labels for recorded, partial, none; popover values verbatim)
- [x] T027 [US2] Regenerate `tests/fixtures/viewer_contract/` from a real job (README procedure) so fixtures carry provenance; update the README provenance note

## Phase 5: User Story 3 - Older files still open, honestly labelled (P2)

**Goal**: pre-provenance files open unchanged and say "version not recorded".
**Independent Test**: the contract test passes on both fixture sets.

- [x] T028 [US3] Extend `viewer/src/test/contract/videoannotator-outputs.test.ts` to run over `legacy/` too, asserting identical annotation parsing and `partial`/`none` provenance
- [x] T029 [P] [US3] Python contract test `tests/contract/test_viewer_contract.py`: legacy fixtures still pass every existing check

## Phase 6: User Story 4 - The job keeps each pipeline's provenance (P2)

**Goal**: results API and CLI return the record per pipeline.
**Independent Test**: after a job, `GET /jobs/{id}/results` and `videoannotator job results` show records equal to the files'.

- [x] T030 [US4] `PipelineResult.provenance` in `src/videoannotator/batch/types.py`; set in `job_execution.py` on success and on failure after initialise
- [x] T031 [US4] `provenance` JSON column on `pipeline_results` in `src/videoannotator/storage/models.py`; additive migration in `sqlite_backend.py` (like `_ensure_batch_columns`); read/write in both backends (`sqlite_backend.py`, `file_backend.py`)
- [x] T032 [US4] `PipelineResultResponse.provenance` in `src/videoannotator/api/v1/jobs.py`; regenerate viewer API types (`scripts/gen_viewer_api_types.sh`)
- [x] T033 [US4] Show model and VideoAnnotator version per pipeline in `videoannotator job results` (`src/videoannotator/cli.py`)
- [x] T034 [P] [US4] Tests: migration adds the column to an old DB (`tests/unit/storage/`); API returns provenance and `null` for an old job (`tests/api/test_result_files.py`)

## Phase 7: Polish

- [x] T035 CHANGELOG `[Unreleased]` entries; roadmap tick (overlay labels) with notes
- [x] T036 Run quickstart.md end to end; full pytest, viewer lint/typecheck/tests, bundle rebuild; push

## Dependencies & Execution Order

- Phase 2 blocks all stories. US1 (Phase 3) before US2 (US2 needs stamped files for fixtures,
  T027) and US4 (records come from T015). US3 needs T001 and US2's parser changes.
- Within US1: T007–T014 parallel; T015 after T006+T002–T004; T019 after T015–T016.

## Parallel Example: User Story 1

```
T007 T008 T009 T010 T011 T012 T013   # one pipeline file each
T018 T020                            # tests in separate files
```

## Implementation Strategy

MVP = Phases 1–3 (files carry provenance, readers unaffected, baseline green). Then US4 (job
record, small), US2 + US3 together (viewer, fixtures), then polish. Commit and push after each
phase with the full suite green.
