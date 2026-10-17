---
description: "Tasks for spec 026: Pipeline Guide"
---

# Tasks: Pipeline Guide

**Input**: `specs/026-pipeline-guide/` (plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md)
**Tests**: included. The constitution requires tests for new API/CLI surface, and SC-001/002/004/005
are automated checks.
**Organization**: by user story (US1 know what it does, US2 defaults and the settings that matter,
US3 judge and report).

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup

- [ ] T001 Add a stub metadata fixture without any guide keys to `tests/fixtures/registry/no_guide.yaml` (copy of an existing built-in's required fields only) and one with every guide key, including an invalid `group`, a benchmark missing `source`, and an `applies_when` naming a missing setting, to `tests/fixtures/registry/full_guide.yaml`

---

## Phase 2: Foundational (blocks every story)

- [ ] T002 In `src/videoannotator/registry/pipeline_registry.py`: add `Benchmark` dataclass (`name`, `measure`, `figure`, `source`, `note`); add `title`, `summary`, `homepage`, `benchmarks: list[Benchmark] | None` (None = not stated, [] = none published), `quality_note` to `PipelineMetadata`; add `group` (default `"technical"`), `label`, `unit`, `min`, `max`, `applies_when: dict[str, list[str]] | None` to `PipelineConfigField`; parse leniently per contracts/metadata.md (warnings, never a skipped file); `figure` kept as a string
- [ ] T003 Add `title_or_display_name` and `summary_or_description` helpers (or properties) on `PipelineMetadata` in `src/videoannotator/registry/pipeline_registry.py` for the fallbacks in data-model.md
- [ ] T004 [P] Parsing tests over both fixtures in `tests/unit/registry/test_pipeline_guide.py`: absent keys → fallbacks and `benchmarks is None`; invalid group → technical + warning; bad benchmark dropped; bad `applies_when` dropped; existing fields unchanged
- [ ] T005 In `src/videoannotator/api/v1/pipelines.py`: add `title`, `summary`, `homepage`, `benchmarks`, `quality_note`, `citations` (key + bibtex) to `PipelineInfo` and fill them in both the list and the single-pipeline routes; add `group`, `label`, `unit`, `min`, `max`, `applies_when` to each `config_schema` entry; in `_to_parameter_schema` use the metadata `label` (fallback: today's title-cased key) and add `group`, `unit`, `min`, `max`, `applies_when` to `PipelineParameterSchema`; add `title`, `summary` to `PipelineSchemaDescriptor`
- [ ] T006 [P] API tests in `tests/api/test_pipeline_guide_api.py` per contracts/api.md: new fields for a built-in on both routes and the schema route; nulls for the no-guide stub; a snapshot of today's existing keys still present
- [ ] T007 In `viewer/src/types/pipelines.ts` add `title`, `summary`, `homepage`, `benchmarks`, `qualityNote`, `citations` to `PipelineDescriptor` and `appliesWhen` to `PipelineParameterSchema`; map the snake_case fields in `viewer/src/api/client.ts` where the catalog and schema are parsed

**Checkpoint**: registry, API and viewer types carry the guide; nothing displays it yet.

---

## Phase 3: User Story 1 - Know what a pipeline does before running it (P1) 🎯 MVP

**Goal**: every pipeline shows a plain title and a behavioural summary on Pipelines, in the job
wizard, and in the CLI.
**Independent test**: Pipelines page and `videoannotator pipelines describe <name>` show a title and
a ≤ 2-sentence summary for every listed pipeline.

- [ ] T008 [P] [US1] Write `title` and `summary` for all 8 pipelines in `src/videoannotator/registry/metadata/*.yaml` (behavioural, ≤ 2 sentences, no model names unless the model is the point; spec 023 Decision 1), e.g. face_analysis "Faces: expression, age, gender"
- [ ] T009 [US1] Implement `videoannotator pipelines describe <name> [--json]` in `src/videoannotator/cli.py` reading `get_registry()` locally (no server): header, summary, deprecation notice, then (until US2/US3 land) the settings as one list; unknown name → exit 1 with the message in contracts/cli.md; `--json` matches the API's guide fields
- [ ] T010 [US1] Add a "Title" column after "Name" to `pipelines list` table and markdown output in `src/videoannotator/cli.py` when the server returns `title`
- [ ] T011 [P] [US1] CLI tests in `tests/unit/cli/test_cli_pipelines_describe.py`: text and JSON for face_analysis, unknown name exit, no-guide stub falls back to display name and description
- [ ] T012 [US1] Create `viewer/src/components/PipelineGuide.tsx` with the header (title fallback display name, readiness badge as in `PipelineSelectionStep`, summary fallback description, deprecation notice) and a slot for settings; use it for each pipeline in `PipelineSelectionStep` (`viewer/src/pages/NewJob.tsx`), which `viewer/src/pages/Pipelines.tsx` already renders
- [ ] T013 [P] [US1] `viewer/src/test/components/PipelineGuide.test.tsx`: title/summary and their fallbacks; deprecation notice first

**Checkpoint**: US1 complete; the guide's headline is everywhere a pipeline is chosen.

---

## Phase 4: User Story 2 - Defaults, and only the settings that matter (P1)

**Goal**: behavioural settings shown with plain labels and units; technical collapsed with counts;
settings hidden when they don't apply; unread settings gone.
**Independent test**: for every pipeline, behavioural group visible, technical collapsed with
"n · m changed", every setting labelled, `scale_factor` hidden unless the detector is OpenCV.

- [ ] T014 [P] [US2] Classify and label every setting in `src/videoannotator/registry/metadata/*.yaml` per data-model.md's table (group, label, unit, min, max); add `applies_when` to `face_analysis.scale_factor`, `min_neighbors` (`detection_backend: [opencv]`) and `vlm_annotation.burst_offsets` (`sampling_mode: [frame_burst]`)
- [ ] T015 [P] [US2] Remove `enable_clip_labels` from `src/videoannotator/registry/metadata/scene_detection.yaml` and `enable_diarization` from `src/videoannotator/registry/metadata/audio_processing.yaml` (R6); confirm with a repo search that no fixture, example or config uses them
- [ ] T016 [US2] Metadata check in `tests/unit/registry/test_pipeline_guide.py` over every built-in (`get_registry()` with the default directory): explicit `group` and `label` on every setting; ≤ 5 behavioural settings (SC-002); `applies_when` keys exist and values valid; every `config_schema` key appears as a word in the pipeline's package (R6)
- [ ] T017 [US2] In `viewer/src/components/DynamicPipelineParameters.tsx`: split parameters into "What to measure" (behavioural, open) and a `Collapsible` "Technical settings (n · m changed / all at defaults)"; labels from `label`; implement `appliesWhen` (hide when conditions fail, unless the value differs from its default, then show with "Not used with these settings"); the produced config object is unchanged
- [ ] T018 [US2] In `src/videoannotator/cli.py` `describe`: print "What to measure" and "Technical (n)" groups with label, default and unit, and "only when <setting> is <values>" for `applies_when`
- [ ] T019 [P] [US2] `viewer/src/test/components/DynamicPipelineParameters.test.tsx`: grouping; counts and changed count updating on edit; `appliesWhen` hide/show; the changed-but-not-applicable case; config output identical to before for the same inputs
- [ ] T020 [P] [US2] Extend `tests/unit/cli/test_cli_pipelines_describe.py` with the groups and the `applies_when` note

**Checkpoint**: US2 complete; settings read as behaviour first, method second.

---

## Phase 5: User Story 3 - Judge and report the method (P2)

**Goal**: About shows homepage, published benchmarks (or "No published benchmark"), quality note,
and citations with Copy BibTeX.
**Independent test**: every pipeline's About has a homepage, a benchmark with its source or the
explicit "none", and its citations; Copy BibTeX copies them all.

- [ ] T021 [P] [US3] Add `homepage` to all 8 metadata files in `src/videoannotator/registry/metadata/*.yaml` (the method's or model's own page or repository)
- [ ] T022 [US3] Source benchmarks, one pipeline at a time, into `src/videoannotator/registry/metadata/*.yaml` (`benchmarks:`, figures quoted exactly with source URL; `[]` where none is published); start from `specs/030-model-defaults/research.md` for detector, diarization, speech and pose figures; add `quality_note` to `vlm_annotation.yaml` ("Depends on the model you choose…" with a link to the model's page). Record each figure's source and the date checked in `specs/026-pipeline-guide/research.md` under a new "Sourced figures" section
- [ ] T023 [US3] Extend the metadata check in `tests/unit/registry/test_pipeline_guide.py`: `title`, `summary` (≤ 2 sentences), `homepage` and the `benchmarks` key present for every built-in; every benchmark has all four required fields and an `http(s)` source (SC-001)
- [ ] T024 [US3] Add the About block to `viewer/src/components/PipelineGuide.tsx` per contracts/viewer.md (open on Pipelines, collapsed in the wizard): homepage link, benchmarks linking to source with note, "No published benchmark" for `[]`, quality note, citation keys with Copy BibTeX (all entries joined as the methods paragraph joins them)
- [ ] T025 [US3] Add the About section to `describe` text output in `src/videoannotator/cli.py` per contracts/cli.md
- [ ] T026 [P] [US3] Extend `viewer/src/test/components/PipelineGuide.test.tsx` (benchmarks, empty, quality note, Copy BibTeX) and `tests/unit/cli/test_cli_pipelines_describe.py` (About, "No published benchmark", quality note)

**Checkpoint**: all three stories complete.

---

## Phase 6: Polish & Cross-Cutting

- [ ] T027 Consistency test `test_api_cli_agree` in `tests/unit/registry/test_pipeline_guide.py`: for every built-in, the API's guide fields (TestClient) and `describe --json` equal the metadata (SC-004)
- [ ] T028 [P] Add per-pipeline guide sections (title, summary, behavioural settings, homepage, benchmarks, citation keys) to `scripts/generate_pipeline_specs.py`; regenerate `docs/pipelines_spec.md`; `tests/unit/docs/test_pipeline_spec_doc.py` passes
- [ ] T029 [P] Add a viewer contract fixture of `GET /pipelines/face_analysis/schema` with the new fields under `viewer/src/test/contract/` and assert it parses into the viewer types
- [ ] T030 Rebuild the viewer bundle with `bash scripts/build_viewer.sh` (commit `src/videoannotator/viewer_static/`)
- [ ] T031 [P] CHANGELOG `[Unreleased]` entries (Added: the pipeline guide, `pipelines describe`; Removed: the two unread settings, with the "will be ignored" note) in `CHANGELOG.md`; tick roadmap v1.6.0 Phase 3b's spec 026 item in `docs/development/roadmap_v1.6.0.md`
- [ ] T032 Run the full gates: `pytest tests/`, `ruff check .`, `mypy src/videoannotator`, `cd viewer && bun run lint && bunx tsc --noEmit -p tsconfig.app.json && bun run test:run`, `bash scripts/build_viewer.sh --check`, `pre-commit run --all-files`; fix anything failing
- [ ] T033 Walk `specs/026-pipeline-guide/quickstart.md` sections 2–4 against a running server (`bash scripts/dev.sh`) and note any deviation in the quickstart

---

## Dependencies

- Phase 1 → Phase 2 → stories. US1, US2, US3 each depend only on Phase 2 and can proceed in parallel
  by file, except that they share `cli.py`'s `describe` (T009 before T018 and T025) and
  `PipelineGuide.tsx` (T012 before T024) and the test file `test_pipeline_guide.py` (T004 → T016 →
  T023 → T027, sequential edits).
- T022 (benchmarks) is the long pole; it can start as soon as T002 lands.
- Polish after the stories it covers.

## Parallel examples

- After Phase 2: T008 (titles), T014 (classification), T021 (homepages) touch the same YAML files in
  different keys; do them as one pass per file if editing concurrently.
- Tests T011, T013, T019 are in separate files.

## Implementation strategy

MVP is US1 (titles and summaries everywhere, plus `describe`), shippable alone. US2 then removes the
long list. US3 adds About; its benchmark sourcing (T022) can trail and land per pipeline, since an
explicit `benchmarks: []` is never written as a placeholder: a pipeline without sourced figures
keeps `benchmarks` absent (shown as not stated) until its figures are checked, and the metadata
check (T023) lands last.
