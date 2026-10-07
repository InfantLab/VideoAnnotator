# Implementation Plan: Saved Datasets in the Viewer

**Branch**: `1.6-dev` (spec `018-saved-datasets-viewer`) | **Date**: 2026-10-04 | **Spec**: [spec.md](spec.md)

## Summary

Give the server's saved datasets (spec 007) a home in the viewer: a Datasets page to list, view,
edit, delete, export, import and start jobs from them; "Use a saved dataset" and "Save as
dataset" in the job wizard, covering both uploaded files (re-found by folder handle or re-pick,
matched by path/name/size, differences shown before running) and server folders. Presets gain
export/import. Backend: two optional fields (server folder, manifest relative path), owner names
in responses, a folder-scan endpoint; CLI `dataset` commands.

## Technical Context

**Language/Version**: Python 3.12/3.13; TypeScript/React (Bun)
**Primary Dependencies**: existing: FastAPI, SQLAlchemy (users-database layer), React Query,
File System Access API (with `<input webkitdirectory>` fallback)
**Storage**: `saved_datasets` gains two nullable columns (additive migration); browser IndexedDB
for folder handles
**Testing**: pytest (API, migration, CLI); Vitest (matching, components); Playwright smoke of
the page against a production build
**Target Platform**: server on Linux/macOS/Windows; evergreen browsers (folder handles in
Chromium-based ones, re-pick elsewhere)
**Project Type**: web application
**Performance Goals**: list and detail stay responsive with 5,000-video manifests; matching a
re-picked 5,000-file folder under 2 s
**Constraints**: no video bytes uploaded to save a dataset (FR-013); backward-compatible API
**Scale/Scope**: 1 page, 3 wizard touch points, 1 backend endpoint, 5 CLI commands

## Constitution Check

| Principle | Assessment |
|---|---|
| I. Local-First | Pass. Manifests only; folder handles stay in the browser. |
| II. Stable Pipeline Contract | Not touched. |
| III. Provenance | Pass. Jobs keep recording `dataset_id`. |
| IV. Modular | Pass. No pipeline coupling. |
| V. Backward Compatibility | Pass. Optional fields, additive columns; spec 007 exports import unchanged. |
| VI. Faithful Display | N/A (no annotations drawn). |
| Engineering Standards | Tests for new API/CLI/UI; typecheck/lint/bundle; no placeholder UI. |

No violations.

## Project Structure

```
src/videoannotator/
├── database/models.py, migrations.py   # server_folder, server_folder_recursive (+ ensure_columns)
├── api/v1/datasets.py                  # new fields, owner_name, 409 on name clash
├── api/v1/ingest.py                    # GET /ingest/scan
└── cli.py                              # dataset list|show|export|import|delete

viewer/src/
├── pages/Datasets.tsx                  # real page (route + nav)
├── components/datasets/                # DatasetList, DatasetDetail, ImportExport
├── components/DatasetPicker.tsx        # wizard: Use a saved dataset
├── components/DatasetDriftDialog.tsx   # differences before running
├── components/PresetBar.tsx            # Export / Import
├── lib/datasetMatch.ts, lib/datasetHandles.ts
├── api/client.ts, types/datasets.ts
└── pages/NewJob.tsx                    # Save as dataset; dataset source; dataset_id on submit

tests/api/test_dataset_preset_endpoints.py (extend), tests/api/test_ingest_endpoints.py (scan),
tests/unit/cli/test_cli_dataset.py, viewer/src/test/lib/datasetMatch.test.ts,
viewer/src/test/components/Datasets*.test.tsx
```

## Complexity Tracking

None.
