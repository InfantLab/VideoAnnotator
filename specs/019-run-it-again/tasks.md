# Tasks: Run It Again

**Tests**: included.

## Phase 1: Backend (blocks all stories)

- [ ] T001 `rerun_of` on `BatchJob` (`batch/types.py`, to_dict/from_dict), `jobs.rerun_of` column + migration + read/write (`storage/models.py`, `sqlite_backend.py`), `list_reruns` on base, sqlite and file backends
- [ ] T002 `create_rerun(original, storage, overrides, batch)` and `POST /api/v1/jobs/{id}/rerun` in `api/v1/jobs.py`; `JobResponse.rerun_of`, `reruns`
- [ ] T003 `POST /api/v1/batches/{id}/rerun` in `api/v1/batches.py`
- [ ] T004 [P] Tests `tests/api/test_rerun.py`: new job same video/pipelines/config; overrides; original byte-for-byte unchanged (metadata, files); upload video linked into new folder and survives deleting the original; ingest video referenced in place; missing video 409; running job 409; reruns listed; batch rerun with a skipped job; migration on old DB
- [ ] T005 CLI `job rerun` (`cli.py`) + `tests/unit/cli/test_cli_rerun.py`; regenerate viewer API types; client methods `rerunJob`, `rerunBatch`

## Phase 2: US1 Rerun (P1)

- [ ] T006 [US1] `viewer/src/lib/wizardStart.ts`: typed wizard navigation state (`fromJob` with mode `edit`|`settings`, `fromBatch`, `startFromDataset`)
- [ ] T007 [US1] `viewer/src/components/RunAgainActions.tsx`: Run again (confirm → rerun → navigate to new job), Edit and run again, Use these settings, Save as preset; video-missing message
- [ ] T008 [US1] JobDetail: actions near results; "Rerun of" / "Reruns" links; Retry labelled "Retry in place"
- [ ] T009 [US1] NewJob: `edit` mode with read-only "Videos from job …" source, prefilled settings, submit via rerun endpoint (job or batch)

## Phase 3: US2 Reuse settings (P1)

- [ ] T010 [US2] NewJob: `settings` mode prefills pipelines/config at Choose videos, names the source; unavailable pipelines reported (existing partitionSelection)
- [ ] T011 [US2] BatchDetail: the same actions (batch rerun, Use these settings from the batch's first job)

## Phase 4: US3 Cues (P2)

- [ ] T012 [US3] Wizard first step: recent finished jobs ("Use settings") and recent presets ("Apply")
- [ ] T013 [US3] Failed/partial job alert: "Fix settings and run again"; completed batch: "Run on more videos"
- [ ] T014 [P] [US3] Vitest for RunAgainActions, wizard modes and recent strip

## Phase 5: Polish

- [ ] T015 Playwright check against a real server (rerun a job; edit and run again; reuse settings); CHANGELOG, roadmap, bundle, full suites, push
