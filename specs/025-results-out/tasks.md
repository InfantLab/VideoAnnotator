---
description: "Tasks for spec 025: Results Out"
---

# Tasks: Results Out

**Input**: `specs/025-results-out/` (plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md)
**Tests**: included. The constitution requires tests for new API/CLI surface, and SC-002/004/005/006/007
are automated checks.
**Organization**: by user story (US1 table, US2 methods, US3 ELAN, US4 notebooks).

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup

- [X] T001 Add optional extras group `parquet = ["pyarrow>=17"]` and include it in `all` in `pyproject.toml`; add `xmlschema` to the `dev` dependency group; run `uv lock`
- [X] T002 [P] Commit the ELAN schema downloaded 2026-10-16 to `tests/fixtures/elan/EAFv3.0.xsd` with a `tests/fixtures/elan/README.md` naming its source URL and date
- [X] T003 [P] Create test fixture builders in `tests/unit/exporters/conftest.py`: a run layout (`<tmp>/Demo run (2026-10-16)/run.json` + `demo_clip/` holding copies of `tests/fixtures/viewer_contract/*`), a legacy layout (copies of `tests/fixtures/viewer_contract/legacy/*`), and a flat job folder

---

## Phase 2: Foundational (blocks every story)

- [X] T004 Implement `Row` (19 columns, contracts/table.md order), `COLUMNS`, `TABLE_FORMAT_VERSION = 1`, `NOT_RECORDED`, and the reader table (suffix → kind, default pipeline; known-but-unread suffixes; silently ignored names) in `src/videoannotator/exporters/readers.py` (no readers yet)
- [X] T005 Implement `ResultFile`, `SkippedFile`, `MissingVideo`, `VideoResults`, `ResultsSet` and `from_folder(path)` (run / video-of-run / flat folder / recursive, per data-model.md; provenance via `provenance.read_record`; media path from run.json `source.path`; missing reasons from run.json status/error) in `src/videoannotator/exporters/results_set.py`
- [X] T006 Implement `from_jobs(jobs)` in `src/videoannotator/exporters/results_set.py` using `batch.result_files.job_folder` and `results_folder.run_folder_of`; jobs without files → `missing` with status / `error_message`
- [X] T007 [P] Add `list_jobs_by_dataset(dataset_id)` to `src/videoannotator/storage/base.py`, `sqlite_backend.py` and `file_backend.py` (and any other backend implementing `list_jobs_by_batch`)
- [X] T008 [P] Tests for results sets (run, video-of-run, flat, legacy, nested folder of runs, unknown file → skipped, `run.json` failed video → missing, `from_jobs` with a fake job) in `tests/unit/exporters/test_results_set.py`; storage query test in `tests/unit/storage/test_list_jobs_by_dataset.py`

**Checkpoint**: a folder or a list of jobs resolves to videos and their recognised files.

---

## Phase 3: User Story 1 - One table of everything (P1) 🎯 MVP

**Goal**: one CSV/Parquet table of every pipeline's events, from CLI (any folder, offline), API (job, run, dataset) and viewer.
**Independent test**: export the demo clip fixtures; every native event appears with matching times, labels and values; CSV reads back in pandas and R-compatible `csv`.

### Tests (write first)

- [X] T009 [P] [US1] Reader tests, one per format, against the demo fixtures: row counts per measure equal the native counts; times/labels/values/boxes equal the native fields (SC-002); `audio_processing` provenance gives `pipeline = audio_processing`; legacy files give `not recorded` (SC-007); VLM without provenance uses the annotation's `model` — in `tests/unit/exporters/test_readers.py`
- [X] T010 [P] [US1] Table tests: column order, row order (FR-015), identical output on re-export, CSV round trip of a cue containing a comma, quote, newline and non-Latin text via `csv` and pandas, empty-field missing values, summary contents, Parquet equals CSV (skipped without pyarrow), `ParquetNotInstalled` when pyarrow is missing (monkeypatched import) — in `tests/unit/exporters/test_table.py`

### Implementation

- [X] T011 [US1] Implement readers in `src/videoannotator/exporters/readers.py`: COCO person, DeepFace face (face, emotion, emotion_score, age, gender), OpenFace 3 (face, action_unit, action_unit_presence, head_pose, gaze, emotion, emotion_score, valence, arousal), scene (scene, scene_score), VLM (vlm_label, vlm_reasoning), RTTM (speaker_turn), WebVTT (speech, skipping the provenance NOTE) — per contracts/table.md; malformed file raises a `ReaderError` with a reason
- [X] T012 [US1] Implement `rows(results_set)` (per-video ordering, per-(run, video, pipeline) event numbering, model/version from provenance) and `ExportSummary` in `src/videoannotator/exporters/table.py`
- [X] T013 [US1] Implement `write_csv(results_set, out)` and `write_parquet(results_set, path)` (pyarrow imported lazily; typed schema; metadata `videoannotator.table_format_version`, `videoannotator.export_summary` with `sources`) and `ParquetNotInstalled` in `src/videoannotator/exporters/table.py`
- [X] T014 [US1] Add the `export` Typer group with `export table FOLDER [--format] [--output] [--json]` per contracts/cli.md (default output, exit codes 0/1/2/3, summary to stderr, refusing to overwrite a non-export file) in `src/videoannotator/cli.py`
- [X] T015 [P] [US1] CLI tests (default output path, `--output -`, `--json`, exit codes, offline on the run fixture, Parquet missing → exit 3) in `tests/unit/exporters/test_cli_export.py`
- [X] T016 [US1] Create `src/videoannotator/api/v1/exports.py` with table routes for jobs, batches and datasets per contracts/api.md (threadpool, temp file + `FileResponse` with cleanup, summary header ≤ 8 KB with `truncated`, `Access-Control-Expose-Headers`, errors `NOTHING_TO_EXPORT`, `PARQUET_NOT_INSTALLED`, 404s, invalid format); include the router in `src/videoannotator/api/v1/__init__.py` and expose the header in the CORS setup
- [X] T017 [P] [US1] API tests: job/batch/dataset table downloads equal the CLI output for the same files (FR-010); summary header; 404s; nothing-to-export; Parquet error — in `tests/api/test_export_routes.py`
- [X] T018 [US1] Viewer client: `exportTable(scope, id, format)` returning `{blob, filename, summary}` (parse `X-VideoAnnotator-Export-Summary`) in `viewer/src/api/client.ts`; regenerate `viewer/src/api/schema.d.ts` if the project's generator is used for new routes
- [X] T019 [US1] Viewer `ExportMenu` component (Table (CSV), Table (Parquet) with install hint / admin install via `lib/pipelineExtras.ts` on `PARQUET_NOT_INSTALLED`, busy state, summary toast when anything was skipped or missing) in `viewer/src/components/ExportMenu.tsx`; mount it beside "Download results" on `viewer/src/pages/JobDetail.tsx` and `viewer/src/pages/BatchDetail.tsx`, and add "Table (CSV)" to the saved dataset card in `viewer/src/pages/Datasets.tsx`
- [X] T020 [P] [US1] Viewer tests for `ExportMenu` (calls client with the right scope, shows summary, Parquet hint) in `viewer/src/test/components/ExportMenu.test.tsx`
- [X] T021 [US1] Write `docs/usage/exporting_results.md`: the table's columns and measures (from contracts/table.md), reading it in R and Python, pivoting wide, CLI and viewer usage; link it from `docs/usage/README.md` and `docs/usage/accessing_results.md`

**Checkpoint**: US1 complete and usable alone (MVP).

---

## Phase 4: User Story 2 - Methods paragraph (P2)

**Goal**: prose + BibTeX for a job or run, from API, CLI and viewer.
**Independent test**: methods for the demo clip name every pipeline with its recorded revisions; every `[@key]` resolves in the BibTeX.

- [X] T022 [US2] Add optional `citations: list[Citation]` (`key`, `bibtex`) to `PipelineMetadata` and its YAML loading in `src/videoannotator/registry/pipeline_registry.py` (and the metadata schema/validator if one exists)
- [X] T023 [P] [US2] Add `citations:` to all eight files in `src/videoannotator/registry/metadata/` with the entries chosen in research.md §10
- [X] T024 [P] [US2] VideoAnnotator's own BibTeX entry builder (`videoannotator_bibtex(version)`) in `src/videoannotator/exporters/citation.py`
- [X] T025 [P] [US2] Tests: every shipped pipeline has ≥ 1 citation, each parses as one entry with a matching key, keys unique; VideoAnnotator's entry matches `CITATION.cff` title, authors and repository — in `tests/unit/registry/test_citations.py`
- [X] T026 [US2] Implement `methods.describe(results_set) -> MethodsDescription` (per-video signatures, grouping, default-diffing against `config_schema` with the non-result key denylist, VLM digest/prompt fingerprint and prompt texts, "not recorded"/"no citation recorded", model revisions list, no names/paths/ids) and `to_markdown()` in `src/videoannotator/exporters/methods.py`
- [X] T027 [P] [US2] Tests: demo clip prose names each pipeline and its revisions exactly (SC-005); every key resolves; one group vs two groups (changed setting on one video); legacy → "not recorded"; VLM prompt listed; no video name, path, job id or base URL in the prose (FR-029) — in `tests/unit/exporters/test_methods.py`
- [X] T028 [US2] CLI `export methods FOLDER [--output] [--json]` in `src/videoannotator/cli.py`, with tests added to `tests/unit/test_cli_export.py`
- [X] T029 [US2] API `GET /jobs/{id}/methods` and `GET /batches/{id}/methods` (JSON, `?format=markdown`) in `src/videoannotator/api/v1/exports.py`, with tests in `tests/api/test_export_routes.py`
- [X] T030 [US2] Viewer: `getMethods(scope, id)` in `viewer/src/api/client.ts`; lazy `MethodsDialog` (prose, BibTeX, prompts, Copy buttons, Markdown download) in `viewer/src/components/MethodsDialog.tsx`; "Methods paragraph" item in `ExportMenu.tsx`; test in `viewer/src/test/components/MethodsDialog.test.tsx`

**Checkpoint**: US2 works alone on top of Phase 2.

---

## Phase 5: User Story 3 - ELAN (P3)

**Goal**: one valid `.eaf` per video with one tier per pipeline track, opening in ELAN and the viewer.
**Independent test**: the demo clip's `.eaf` validates against EAF 3.0, its annotations match the table, and the viewer's parser reads its tiers.

- [X] T031 [US3] Implement `elan.build(video_results) -> ElementTree` and `write_eaf` (tiers and values per contracts/elan-and-methods.md, frame-detection merging by track with the 2× median-spacing gap rule, VLM hold-until-next-sample with equal-label merging, ms rounding, unique tier ids, media descriptor from the recorded video path, `left_out` in the summary) in `src/videoannotator/exporters/elan.py`
- [X] T032 [P] [US3] Tests: schema validation with `xmlschema` against `tests/fixtures/elan/EAFv3.0.xsd`; interval annotations equal table rows; merging rule on synthetic detections (gap splits, single frame widened); VLM holds; continuous measures absent; legacy fixtures export — in `tests/unit/exporters/test_elan.py`
- [X] T033 [US3] CLI `export elan FOLDER [--output] [--json]` in `src/videoannotator/cli.py`, tests in `tests/unit/exporters/test_cli_export.py`
- [X] T034 [US3] API `GET /jobs/{id}/export/elan` (one `.eaf`) and `GET /batches/{id}/export/elan` (zip) in `src/videoannotator/api/v1/exports.py`, tests in `tests/api/test_export_routes.py`
- [X] T035 [US3] Viewer: `exportElan(scope, id)` in `viewer/src/api/client.ts`; "ELAN files (.eaf)" item in `viewer/src/components/ExportMenu.tsx`
- [X] T036 [US3] Add `tests/fixtures/viewer_contract/exported/demo_clip.eaf` (generated) and `demo_clip_edited_in_elan.eaf` (same plus a hand-coded tier with its own linguistic type, as ELAN saves it), and extend `viewer/src/test/contract/videoannotator-outputs.test.ts` to parse both and find the expected tiers (FR-022)

**Checkpoint**: US3 works alone on top of Phase 2 and US1's rows.

---

## Phase 6: User Story 4 - Worked notebooks (P4)

**Goal**: R and Python notebooks over the demo clip's table.
**Independent test**: each notebook runs top to bottom on the shipped CSV.

- [X] T037 [US4] Write `scripts/make_demo_table.py`: builds the run-layout copy of the viewer contract fixtures and writes `examples/notebooks/demo_clip_table.csv` and `tests/fixtures/viewer_contract/exported/demo_clip.eaf`; run it and commit the outputs
- [X] T038 [P] [US4] Python notebook `examples/notebooks/results_table_python.ipynb` (pandas, matplotlib): load, overview by pipeline/measure, speaking time per speaker, speech timeline, people in view per track, emotion over time, infant-in-view × speech overlap, pivot wide; one `TABLE = "demo_clip_table.csv"` line to change
- [X] T039 [P] [US4] R notebook `examples/notebooks/results_table_r.Rmd` (readr, dplyr, tidyr, ggplot2) with the same analyses
- [X] T040 [P] [US4] `examples/notebooks/README.md`: what each notebook shows, packages, how to point it at your own export
- [X] T041 [US4] Tests in `tests/unit/test_notebooks.py`: the shipped CSV equals a fresh export of the fixtures (stale check); the Python notebook's code cells run in order with `MPLBACKEND=Agg` (skipped without pandas/matplotlib)
- [X] T042 [US4] Manual check for the R notebook in `tests/manual/results_out_r_notebook.md` (and run it here if R can be installed; record the result in quickstart.md's walkthrough log)

---

## Phase 7: Polish & Cross-Cutting

- [X] T043 Run `ruff check .`, `ruff format`, `mypy src/videoannotator`, the full `pytest tests/` (non-real-model) and `cd viewer && bun run lint && bun run test:run && npx tsc --noEmit -p tsconfig.app.json`; fix everything
- [X] T044 Check the slim install path: exporters import and CSV export works with no extras (pyarrow absent), matching CI's core-only job
- [X] T045 Rebuild the viewer bundle with `bash scripts/build_viewer.sh` and commit `src/videoannotator/viewer_static/`
- [X] T046 [P] CHANGELOG `[Unreleased]` entry for Results Out; tick Phase 3 items in `docs/development/roadmap_v1.6.0.md` with dates and what shipped; README "Get your results out" line pointing to `docs/usage/exporting_results.md`
- [X] T047 Walk `specs/025-results-out/quickstart.md` steps 1, 2 and 4 (the viewer step with `scripts/dev.sh` where possible) and record results in its walkthrough log
- [X] T048 Run `pre-commit run --all-files`

---

## Dependencies & Execution Order

- Phase 1 → Phase 2 → stories. US1 first (MVP).
- US2 (methods) needs only Phase 2 (results sets, provenance); can run beside US1.
- US3 (ELAN) uses US1's rows (`table.rows`) — after T012.
- US4 needs US1 (the CSV) and T036's generator shares `make_demo_table.py` with T037.
- API tasks (T016, T029, T034) touch the same file `exports.py`: sequential. CLI tasks (T014, T028, T033) likewise in `cli.py`. Viewer `ExportMenu.tsx` edits (T019, T030, T035) likewise.

## Parallel Examples

- Phase 2: T007 beside T005/T006.
- US1: T009 and T010 together; then T015, T017, T020 alongside their implementations.
- US2: T023, T024, T025 together.
- US4: T038, T039, T040 together.

## Implementation Strategy

MVP = Phases 1–3 (the table): this alone meets the release's main goal. Then US2, US3, US4 in
order, each committed when its checkpoint passes. Commit per phase on `1.6-dev`
(`exporters: …`, `api: …`, `viewer: …`, `docs: …`).
