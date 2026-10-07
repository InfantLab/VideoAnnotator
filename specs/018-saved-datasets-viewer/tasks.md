# Tasks: Saved Datasets in the Viewer

**Input**: `specs/018-saved-datasets-viewer/` (spec, plan, research, data-model, contracts)
**Tests**: included (new API, CLI and UI surface).

## Phase 1: Foundational (backend)

- [x] T001 Add `server_folder` (String, nullable) and `server_folder_recursive` (Boolean, default false) to `SavedDataset` in `src/videoannotator/database/models.py`; add both to `ensure_columns("saved_datasets", …)` in `src/videoannotator/database/migrations.py`
- [x] T002 In `src/videoannotator/api/v1/datasets.py`: `VideoManifestEntry.relative_path`; request/response `server_folder`, `server_folder_recursive`; response `owner_name`; 409 `DATASET_NAME_CONFLICT` (exists since spec 007) on duplicate name (create, update)
- [x] T003 `GET /api/v1/ingest/scan` in `src/videoannotator/api/v1/ingest.py` (reuse `find_videos`, same guards as ingest)
- [x] T004 [P] Tests: new fields round-trip, spec-007 export imports unchanged, owner_name, 409, migration on an old database (`tests/api/test_dataset_preset_endpoints.py`); scan guards and listing (`tests/api/test_ingest_endpoints.py`)
- [x] T005 Regenerate viewer API types (`scripts/gen_viewer_api_types.sh`); dataset types and client methods (`list/get/create/update/deleteDataset`, `scanServerFolder`) in `viewer/src/api/client.ts`, `viewer/src/types/datasets.ts`

## Phase 2: US1 - Save the videos I chose, and reuse them (P1)

- [x] T006 [US1] `viewer/src/lib/datasetMatch.ts`: `manifestFromFiles(files)`, `matchDataset(manifest, files)` per research R3
- [x] T007 [P] [US1] `viewer/src/test/lib/datasetMatch.test.ts`: matched/missing/added/changed/ambiguous, relative paths vs names, 5,000-file timing
- [x] T008 [US1] `viewer/src/lib/datasetHandles.ts`: store/get a folder handle per dataset; re-request read permission; walk to `File`s with relative paths
- [x] T009 [US1] `viewer/src/components/DatasetDriftDialog.tsx`: lists differences; Continue with matching / Update dataset / Cancel
- [x] T010 [US1] `viewer/src/components/DatasetPicker.tsx`: list datasets; on choose → server folder (scan, drift, set wizard server folder) or files (handle or re-pick, match, drift, set selected files)
- [x] T011 [US1] `viewer/src/pages/NewJob.tsx`: "Use a saved dataset" source; "Save as dataset" (name dialog, manifest from files or server scan, store handle); submit with `dataset_id`
- [x] T012 [P] [US1] Component tests for picker + drift dialog with mocked client and files (`viewer/src/test/components/DatasetPicker.test.tsx`)

## Phase 3: US2 - See and manage my datasets (P1)

- [x] T013 [US2] `viewer/src/pages/Datasets.tsx` + `components/datasets/DatasetList.tsx`, `DatasetDetail.tsx`: list (name, count, owner, created, last used), detail (manifest), rename/description/remove videos/delete (owner or admin only, using `/auth/me`), Start a job (→ wizard with dataset), empty state
- [x] T014 [US2] Route `/datasets` (replace the Library redirect) and nav link in `viewer/src/App.tsx` / `AppLayout`
- [x] T015 [P] [US2] `viewer/src/test/components/Datasets.test.tsx`: list, owner-only actions, delete confirmation text, empty state

## Phase 4: US3 - Share a dataset or preset (P2)

- [x] T016 [US3] Export/Import for datasets (`components/datasets/ImportExport.tsx`) and presets (`PresetBar.tsx`): download JSON named after it; import with clash renaming (R4) and invalid-file message
- [x] T017 [P] [US3] Tests for import/export (clash, invalid file) in `viewer/src/test/components/ImportExport.test.tsx`
- [x] T018 [US3] CLI `videoannotator dataset list|show|export|import|delete` in `src/videoannotator/cli.py`; tests `tests/unit/cli/test_cli_dataset.py`

## Phase 5: Polish

- [x] T019 Playwright check of the Datasets page and wizard flow against the production build (scratch script; uploaded-files path with re-pick)
- [x] T020 CHANGELOG, roadmap tick, bundle rebuild, full test suites, push

## Dependencies

Phase 1 → Phases 2–4. US1 and US2 share client types (T005); US3 after US2 (buttons live on the page).
