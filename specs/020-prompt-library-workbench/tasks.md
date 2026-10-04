# Tasks: Prompt Library and Prompt Workbench

**Tests**: included.

## Phase 1: Backend

- [x] T001 `Prompt`, `PromptUse` models (`database/models.py`)
- [x] T002 `prompt_library.py`: record_use (upsert by sha256), search (text/name/tag/model, starred first, last used), get with uses (models, jobs), update (name/tags/starred/hidden, attributed), delete (409 if used by a job)
- [x] T003 Record job prompts in `batch/job_execution.py` and previews in `api/v1/vlm.py` (best effort); preview response `frames`
- [x] T004 `/api/v1/prompts` router (list, get, put, delete) registered in `api/v1/__init__.py`
- [x] T005 [P] Tests: service, API, job hook, preview hook and frames (`tests/unit/test_prompt_library.py`, `tests/api/test_prompts.py`)
- [x] T006 CLI `prompts list|show|diff`, `vlm preview` + tests

## Phase 2: US1 Library page (P1)

- [ ] T007 [US1] types, client methods; `lib/wordDiff.ts` + tests
- [ ] T008 [US1] `pages/Prompts.tsx`: search, starred first, detail (text, models, jobs, first/last, who), name/star/tag/hide, diff two, Use in a new job, Open in workbench; route + nav

## Phase 3: US2 Workbench (P1)

- [ ] T009 [US2] `pages/Workbench.tsx`: video (past job or server folder), moments, single/burst, prompts (edit, load from library), models; run grid prompt × model × moment; rounds kept; per-cell errors; frames shown; Send to job / Save as preset; Local-First notice
- [ ] T010 [US2] Wizard test panel links to the workbench with its prompt and model
- [ ] T011 [P] [US2] Vitest for the workbench grid with a mocked client

## Phase 4: Polish

- [ ] T012 Real end-to-end with Ollama: a VLM job and workbench previews recorded once each; provenance VLM fields present; Chromium check; CHANGELOG, roadmap, bundle, suites, push
