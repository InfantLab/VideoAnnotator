---
description: "Task list for 022 — Videos and Results Where You Expect Them"
---

# Tasks: Videos and Results Where You Expect Them

**Input**: Design documents from `specs/022-videos-read-in-place/`
**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md),
[data-model.md](data-model.md), [contracts/api.md](contracts/api.md),
[contracts/docker.md](contracts/docker.md), [quickstart.md](quickstart.md)

**Tests**: Included. The plan's Constitution Check requires tests for every new endpoint and rule,
and names the new test files.

**Organization**: by user story. Phases follow the plan's Delivery Order, which puts the server-side
results folder first because every job-creation path in later stories sets `output_dir` through it.
Each phase is committed and pushed to `1.6-dev` when green (`pre-commit run --all-files`).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files, no dependency on an incomplete task)
- **[Story]**: US1–US8, from spec.md
- Paths are relative to the repository root. Server code is under `src/videoannotator/`, the viewer
  under `viewer/src/`.

---

## Phase 1: Setup

**Purpose**: a green baseline and shared test fixtures.

- [X] T001 Run `pytest tests/api tests/unit -q`, `cd viewer && bun run lint && bun run test:run`, `cd viewer && bunx tsc --noEmit -p tsconfig.app.json` (record the error count; it is non-zero today, see the constitution's follow-up TODOs) and `bash scripts/build_viewer.sh --check` (record the gzipped initial bundle size); record any pre-existing failures in the commit message of T003 so later phases are not blamed for them (SC-006)
- [X] T002 [P] Add a `results_root` pytest fixture in `tests/conftest.py` that points `VIDEOANNOTATOR_RESULTS_DIR` at a `tmp_path` subfolder (monkeypatch env and reload/patch `config_env.RESULTS_DIR`), and an `ingest_root` fixture doing the same for `VIDEOANNOTATOR_INGEST_ROOTS`, so no test ever writes into the real `~/VideoAnnotator`
- [X] T003 Make the existing API test suites use the `results_root` fixture by default (autouse in `tests/api/conftest.py`, creating that file if absent), so jobs created by existing tests don't write to the developer's home folder once `output_dir` is set at creation

---

## Phase 2: Foundational (blocking prerequisites)

**Purpose**: settings, the `results_folder.py` module holding every naming and layout rule, and the
single "same machine" rule. Every story depends on these.

**⚠️ No user-story work starts until this phase is complete.**

- [X] T004 Add `RESULTS_DIR` (env `VIDEOANNOTATOR_RESULTS_DIR`, default `str(Path.home() / "VideoAnnotator")`; `.env` is already loaded by `load_env_file`, there is no config-file key), `PUBLISHED_LOCALLY` (`get_bool_env("VIDEOANNOTATOR_PUBLISHED_LOCALLY", False)`) and `HOST_PATHS` (env `VIDEOANNOTATOR_HOST_PATHS`, `container=host` pairs separated by `;`) next to `INGEST_ROOTS` in `src/videoannotator/config_env.py`; include all three in `print_config()`
- [X] T005 Create `src/videoannotator/results_folder.py` with `results_root() -> Path` (expanduser + resolve of `config_env.RESULTS_DIR`), `sanitize_component(name: str) -> str` (drop `<>:"/\|?*` and control chars, strip trailing dots/spaces, prefix `_` on Windows reserved names `CON PRN AUX NUL COM1-9 LPT1-9` case-insensitively, cap at 120 chars, fall back to `"Run"`/`"video"` when empty) per research.md R2
- [X] T006 In `src/videoannotator/results_folder.py` add `run_name(batch_name, dataset_name, source_folder, video_path, now)` implementing R2's fallback order (batch name → dataset name → source folder name → single video's stem → `"Run <YYYY-MM-DD HH-MM>"`) and `create_run_folder(name: str, today: date) -> Path` that creates `<root>/<name> (<YYYY-MM-DD>)` with `mkdir(parents=False, exist_ok=False)` after ensuring the root exists, retrying with ` 2`, ` 3`, … inserted before the closing bracket on `FileExistsError`
- [X] T007 In `src/videoannotator/results_folder.py` add `video_folder_names(videos: list[Path], source_root: Path | None) -> dict[Path, str]`: stem by default; for stems that collide within the run use the path relative to `source_root` without suffix, separators replaced by `__`; if still colliding (or no source root), append ` 2`, ` 3`; every name passed through `sanitize_component`. Add `create_video_folder(run_folder, name) -> Path` (exclusive mkdir)
- [X] T008 In `src/videoannotator/results_folder.py` add `display_path(path: Path) -> str` (rewrite the longest matching container prefix from `config_env.HOST_PATHS` to its host path; unchanged otherwise), `folder_ref(path) -> dict` returning `{"path", "display_path"}`, and `check_writable() -> None` that creates the root if needed, writes and deletes a probe file, and raises `ResultsDirUnwritable(folder, reason)` on `OSError` (FR-026)
- [X] T009 In `src/videoannotator/results_folder.py` add `assert_no_overlap(allowed_roots: list[Path]) -> None` raising when the results root is inside an allowed root or vice versa (data-model.md Job rules, FR-021), and `assert_output_dir_safe(output_dir, storage_path, allowed_roots)` used at job creation
- [X] T010 [P] Write `tests/unit/test_results_folder_names.py`: sanitising (each Windows-invalid char, trailing dots, `CON`/`nul.txt`, 300-char names, emoji/unicode kept), run-name fallback order, run-folder clash numbering `(2026-10-06 2)`, video-folder stem collisions → `site_a__child01`, `display_path` prefix mapping with nested prefixes, `assert_no_overlap` both directions, `check_writable` on a read-only dir (skip on Windows/root)
- [X] T011 Refactor `require_local_caller` in `src/videoannotator/api/v1/ingest.py` into `is_same_machine(request: Request) -> bool` (loopback client address, or `config_env.PUBLISHED_LOCALLY`) plus the existing raising wrapper, so the access endpoint, ingest and `results/open` share one definition (R5, R6)
- [X] T012 In the server startup path (`src/videoannotator/cli.py` `server` command, and the API lifespan in `src/videoannotator/api/` where startup logging already happens) log a warning when `PUBLISHED_LOCALLY` is set, naming the assumption and saying to unset it if the port is published beyond loopback (R6); call `results_folder.assert_no_overlap(allowed_roots())` and log a clear error (in-place reading disabled) rather than crash on overlap

**Checkpoint**: `pytest tests/unit/test_results_folder_names.py` green; nothing user-visible yet.

---

## Phase 3: User Story 6 — Find a run's results in a folder I recognise (P1) 🎯 MVP part 1

**Goal**: every new job writes to `<results root>/<run> (<date>)/<video>/`, with `run.json`, no
videos, never overwriting.

**Independent test**: run 3 videos as a named run through the API; without the viewer, find
`~/VideoAnnotator/<name> (<date>)/<video>/<video>_*.json` and `run.json`; no `.mp4` there.

### Tests for User Story 6

- [X] T013 [P] [US6] Write `tests/api/test_results_folder.py`: (a) folder ingest of 3 videos with `batch_name="BabyJokes wave 2"` sets each job's `output_dir` to `<root>/BabyJokes wave 2 (<today>)/<stem>`; (b) a second identical ingest gets `(<today> 2)`; (c) upload (`POST /api/v1/jobs/`) sets `output_dir` and the uploaded copy stays under `storage_path`, not under the results root; (d) dataset run and job/batch rerun each create a new run folder; (e) unwritable root → 422 `RESULTS_DIR_UNWRITABLE` with the folder in the message and no jobs created; (f) `run.json` exists after creation with the data-model.md shape; (g) responses carry `results_folder`
- [X] T014 [P] [US6] Add to `tests/unit/test_results_folder_names.py` (or a new `tests/unit/test_run_record.py`) tests for `run.json` writing: atomic replace (temp file in same folder, then `os.replace`), concurrent `update_video_entry` calls from threads leave valid JSON with all entries, uploaded sources record `original_filename` and never a server path
- [X] T015 [P] [US6] Add a test in `tests/unit/` (extend the existing job-execution tests, or `tests/unit/test_job_execution_output_dir.py`) that a job with `output_dir` set writes pipeline outputs there and nothing to `storage_path`, and that a job with `output_dir=None` still falls back to `storage_path` (pre-feature jobs, FR-018)

### Implementation for User Story 6

- [X] T016 [US6] In `src/videoannotator/results_folder.py` add the run record: `write_run_record(run_folder, *, batch_id, name, created_at, pipelines, config, videos)` and `update_video_entry(run_folder, job_id, **fields)`; both write via a temp file + `os.replace` under a module-level `threading.Lock` keyed by run folder; include `videoannotator_version` from `src/videoannotator/version.py`; `source` is `{"kind": "in_place", "path", "size_bytes"}` or `{"kind": "uploaded", "original_filename"}` (data-model.md)
- [X] T017 [US6] In `src/videoannotator/results_folder.py` add one helper used by every creation path: `plan_run(*, name_hint: dict, videos: list[tuple[Path, dict]], source_root, pipelines, config, batch_id) -> RunPlan` that calls `check_writable`, creates the run folder and video folders, writes `run.json`, and returns the mapping video → `output_dir` plus the run folder `folder_ref`. Add `run_folder_for_batch(batch_id, create: Callable[[], Path]) -> Path`: under a module-level lock, return the run folder already used by that `batch_id` (process-level dict, falling back to scanning the batch's jobs' `output_dir.parent` after a restart), creating it only once, so concurrent first uploads of one batch can't make two run folders; video folders are created up front (FR-023 "clearly incomplete" = empty folder until results land)
- [X] T018 [US6] Use `plan_run` in folder ingest (`ingest_folder` in `src/videoannotator/api/v1/ingest.py`): check writability before creating any job (422 `RESULTS_DIR_UNWRITABLE`), set each `BatchJob.output_dir`, call `assert_output_dir_safe`, and add `results_folder: dict | None` to `IngestResponse`
- [X] T019 [US6] Use `plan_run` in upload submission (`submit_job` in `src/videoannotator/api/v1/jobs.py`): jobs sharing a `batch_id` must share one run folder — get the run folder from `results_folder.run_folder_for_batch` (T017), and append the new video to its `run.json`; the uploaded copy keeps going to `storage_path`
- [X] T020 [US6] Use `plan_run` in `create_rerun` / `job_from_stored_video` (`src/videoannotator/api/v1/jobs.py`) and in `rerun_batch` / `retry_batch` (`src/videoannotator/api/v1/batches.py`): rerun creates a new run folder named `<original run name> (rerun)`; retry of the same job reuses that job's existing `output_dir` (it's the same run), after emptying that video folder — its files are the failed attempt's partial outputs, never another run's results, so FR-024 holds
- [X] T021 [US6] Use `plan_run` in `run_dataset` (`src/videoannotator/api/v1/datasets.py`), with the dataset name as the run-name hint; stored-copy datasets record `source.kind = "uploaded"`
- [X] T022 [US6] In `src/videoannotator/batch/job_execution.py`, after a job reaches a terminal status, call `results_folder.update_video_entry` with status, `finished_at`, the result file names in `output_dir`, and the per-pipeline model references from spec 017 provenance (`src/videoannotator/provenance.py`); skip silently for jobs whose `output_dir` has no `run.json` above it (pre-feature jobs); never let a run-record failure fail the job (log it)
- [X] T023 [US6] Add `results_folder` to `JobResponse` (`src/videoannotator/api/v1/jobs.py`, from `output_dir`, `null` when unset) and to `BatchSummaryResponse` (`src/videoannotator/api/v1/batches.py`, the run folder from any job's `output_dir.parent` when it contains `run.json`), both via `results_folder.folder_ref`
- [X] T024 [US6] In `delete_job` in `src/videoannotator/storage/sqlite_backend.py` also remove the job's `output_dir` when it lies inside the results root (never otherwise — guard with `is_relative_to(results_root())`), drop its entry from `run.json`, and remove the run folder when only `run.json` remains; never touch `video_path`
- [X] T025 [US6] In `src/videoannotator/cli.py` `server` command, print the results folder (display path) at startup next to the storage root it already prints (R4)
- [X] T026 [US6] CLI `process` in place (R14): in `src/videoannotator/batch/local_job.py` drop `_place_video` (no hard link/copy), set `video_path` to the original and `output_dir` via `plan_run` (run named after the video stem); keep `--output` copying result files there too; update `tests/` for `local_job` accordingly (find with `grep -rl local_job tests/`)
- [X] T027 [US6] Regenerate viewer API types with `bash scripts/gen_viewer_api_types.sh` so `viewer/src/api/schema.d.ts` has `results_folder` on job, batch and ingest responses

**Checkpoint**: `pytest tests/api/test_results_folder.py tests/unit -q` green; quickstart "Story 6" passes from the CLI. Commit and push.

---

## Phase 4: User Story 1 — Choose videos on my own laptop without copying them (P1) 🎯 MVP part 2

**Goal**: on a same-machine connection the wizard opens on "My folders"; upload is a secondary link.

**Independent test**: plain install, open New job: "My folders" is shown, submit a folder, and
`find <storage root>/jobs ~/VideoAnnotator -name '*.mp4'` prints nothing.

### Tests for User Story 1

- [X] T028 [P] [US1] Write `tests/api/test_ingest_access.py`: `GET /api/v1/ingest/access` from a loopback client as admin → `same_machine`/`can_read_in_place` true, `allowed_folders` and `results_root` with `display_path`; non-loopback client → `same_machine` false with the "another computer" `reason`; non-admin on loopback → `can_read_in_place` false with the administrator reason; non-loopback with `PUBLISHED_LOCALLY` → `same_machine` true; no existing allowed folder → `can_read_in_place` false with the `VIDEOS_DIR` reason; `can_open_folders` false when not same machine
- [X] T029 [P] [US1] Write `viewer/src/test/components/MyFolders.test.tsx` (part 1): with access `same_machine: true` the New job step 1 shows tabs "My folders" (selected) and "Saved datasets", plus the "Videos on another computer? Upload them" link that switches to the uploader; with `same_machine: false` it shows upload and Saved datasets, no My folders; the "used where they are, never copied" line is rendered

- [X] T030 [P] [US1] Write `tests/integration/test_in_place_equivalence.py` (marked like `tests/integration/test_output_baseline.py`; needs real models): run the clip in `viewer/demo-assets/` with `scene_detection` once by folder ingest and once by upload, and assert the output JSONs are equal after removing timestamps and paths from provenance (reuse that test's normalisation) (FR-007)

### Implementation for User Story 1

- [X] T031 [US1] Add `GET /api/v1/ingest/access` to `src/videoannotator/api/v1/ingest.py` with an `IngestAccessResponse` model matching contracts/api.md (`same_machine`, `can_read_in_place`, `reason`, `allowed_folders`, `results_root`, `can_open_folders`); `can_open_folders` = same machine and a desktop opener is available (`sys.platform` win32/darwin, or `shutil.which("xdg-open")` and `DISPLAY`/`WAYLAND_DISPLAY` set, and not running in a container — check `/.dockerenv`); any authenticated caller may call it
- [X] T032 [US1] Regenerate `viewer/src/api/schema.d.ts` (`bash scripts/gen_viewer_api_types.sh`) and add `getIngestAccess()` to `viewer/src/api/client.ts`
- [X] T033 [P] [US1] Create `viewer/src/hooks/useIngestAccess.ts` (React Query, keyed on the API base URL and token, so switching server re-fetches; treat errors as `same_machine: false`)
- [X] T034 [US1] In `viewer/src/pages/NewJob.tsx` choose the step-1 source from `useIngestAccess`: same machine → tabs "My folders" (default) and "Saved datasets", with upload behind the "Videos on another computer? Upload them" link (and a "Back to My folders" link from upload); otherwise upload + Saved datasets as today; while access is loading, render neither default to avoid a flash of the upload tab. Respect existing wizard-start handling in `viewer/src/lib/wizardStart.ts`
- [X] T035 [US1] Rename the `ServerFolderPicker` tab label to "My folders" in `viewer/src/components/ServerFolderPicker.tsx`, start it at the first allowed folder (`display_path` shown), and add the line "N videos selected – used where they are, never copied" plus "Deleting a job never deletes the original video" (FR-004); when `can_read_in_place` is false but same machine, show `reason` and the upload link. Below the list, always show a muted hint: "Videos somewhere else, like an external drive? Add that folder to `VIDEOANNOTATOR_INGEST_ROOTS` (Docker: `VIDEOS_DIR`) and restart", linking to the INSTALLATION.md section T071 extends (edge case: video outside allowed folders, FR-010)
- [X] T036 [US1] After a My folders submission, navigate to the run (batch) page as ingest already does, and confirm no upload progress UI appears; update any existing tests in `viewer/src/test/` that assumed upload is the default tab

**Checkpoint**: quickstart "Story 1" passes. Commit and push.

---

## Phase 5: User Story 2 — Pick a few videos out of a large folder (P1)

**Goal**: tick individual videos (or Select all, optionally with subfolders); only those run.

**Independent test**: in a 20-video folder, select 3 and submit; exactly 3 jobs, each reading in place.

### Tests for User Story 2

- [X] T037 [P] [US2] Extend `tests/api/test_ingest_endpoints.py`: `files` limits jobs to exactly those (incl. a `site_a/child02.mp4` subpath); `recursive` is ignored when `files` is given; a `files` entry escaping the root (`../x.mp4`, symlink out) lands in `skipped` with a reason; duplicate entries (and `a/../child01.mp4`) produce one job; absent `files` behaves exactly as before; timing check: a folder of 100 small dummy `.mp4` files with all 100 in `files` returns 2xx in under 5 s, creating 100 video folders and one `run.json` (SC-001)
- [X] T038 [P] [US2] Extend `viewer/src/test/components/MyFolders.test.tsx` (part 2): checkboxes per video, "Select all" toggles all visible, "Include subfolders" lists subfolder videos with their relative path, the count line updates, submission sends `path` + `files` (relative paths), durations blank until known without blocking render

### Implementation for User Story 2

- [X] T039 [US2] Add optional `files: list[str] | None` to `IngestRequest` in `src/videoannotator/api/v1/ingest.py`; when set, resolve each against `path` with `resolve_within_roots`, require a video extension (same check as `find_videos`), dedupe by resolved path, report rejects in `skipped`, and pass the selected list to the run creation (Phase 3's `plan_run`, with `source_root=path` for video-folder naming)
- [X] T040 [US2] Ensure the `scan` endpoint in `src/videoannotator/api/v1/ingest.py` returns quickly for 1,000 files without probing durations (return `duration: null` unless cheaply known), and supports `recursive=true` listing with relative paths; add a timing assertion test (<1 s for 1,000 empty `.mp4` files) to `tests/api/test_ingest_endpoints.py`
- [X] T041 [US2] Regenerate `viewer/src/api/schema.d.ts` and update the ingest call in `viewer/src/api/client.ts` to accept `files`
- [X] T042 [US2] In `viewer/src/components/ServerFolderPicker.tsx` list the current folder's videos (name, duration where known, size) with checkboxes, "Select all", and an "Include subfolders" toggle that re-scans recursively; keep folder navigation (breadcrumb + Up); selection survives toggling subfolders where paths still appear; submit `path` + `files` (whole-folder selection with nothing unticked may send no `files`, matching the old request)
- [X] T043 [US2] Keep the picker usable for thousands of files in `viewer/src/components/ServerFolderPicker.tsx`: render the list in a scroll container with a fixed max height, and avoid per-row re-renders on selection (memoised rows or a `Set` in a ref plus a counter)

**Checkpoint**: quickstart "Story 2" passes. Commit and push.

---

## Phase 6: User Story 7 — Get to the results from the viewer (P1)

**Goal**: run and job pages show the results location, open it on this computer, copy it, and
download a whole run as one zip without videos; job zips exclude the video by default.

**Independent test**: from a finished run's page, open its folder, then download the run zip and
confirm every video's results and no videos.

### Tests for User Story 7

- [ ] T044 [P] [US7] Write `tests/api/test_results_open.py`: non-same-machine → 403 `NOT_SAME_MACHINE`; path outside the results root (incl. `..` and a symlink pointing out) → 422 `PATH_OUTSIDE_RESULTS`; opener patched to succeed → 204 and called with the resolved folder; opener unavailable or raising → 409 `OPEN_FOLDER_UNSUPPORTED`
- [ ] T045 [P] [US7] Add to `tests/api/test_batch_endpoints.py`: `GET /api/v1/batches/{id}/results.zip` contains `run.json` and `<video>/<files>` matching the on-disk layout and no video extensions; unknown batch → 404 `BATCH_NOT_FOUND`; a pre-feature batch (jobs with `output_dir=None`) zips the same layout built from each job's folder, excluding videos
- [ ] T046 [P] [US7] Add to `tests/api/test_batch_endpoints.py`: `DELETE /api/v1/batches/{id}` removes all the run's jobs, video folders and the run folder; the source videos are untouched; a running job is cancelled first; unknown batch → 404 `BATCH_NOT_FOUND`
- [ ] T047 [P] [US7] Add an artifacts test (in `tests/api/test_result_files.py` or a new `tests/api/test_artifacts_include_video.py`): `GET /api/v1/jobs/{id}/artifacts` omits the video by default (with the `X-VideoAnnotator-Notice` header) and includes it with `include_video=true`
- [ ] T048 [P] [US7] Write `viewer/src/test/components/ResultsLocation.test.tsx`: shows `display_path`; "Open folder" only when `can_open_folders`; 409 falls back to a "copy the location" message; Copy writes to the clipboard; "Download results" calls the run zip URL; renders nothing for `results_folder: null`

### Implementation for User Story 7

- [ ] T049 [US7] Create `src/videoannotator/api/v1/results.py` with `POST /results/open` per contracts/api.md (uses `is_same_machine`, resolves the path and requires it inside `results_root()`, opens with `os.startfile` / `open` / `xdg-open` via `subprocess.Popen` without a shell, maps failures to 409 `OPEN_FOLDER_UNSUPPORTED`), and register it in `src/videoannotator/api/v1/__init__.py` with `prefix="/results"`
- [ ] T050 [US7] Add `GET /batches/{batch_id}/results.zip` to `src/videoannotator/api/v1/batches.py`, streaming with `zipfile` into a generator-backed `StreamingResponse` (write to a spooled temp file or chunked pipe — never build the zip in memory); skip any file with a video extension; for pre-feature jobs use `batch/result_files.py: job_folder` and name folders by video stem with R2's collision rule
- [ ] T051 [US7] Add `include_video: bool = False` to the artifacts endpoint in `src/videoannotator/api/v1/endpoints/artifacts.py` and exclude the video file unless set; when excluded, add the response header `X-VideoAnnotator-Notice: video excluded; use include_video=true` (contracts/api.md)
- [ ] T052 [US7] Add `DELETE /batches/{batch_id}` to `src/videoannotator/api/v1/batches.py`: cancel non-terminal jobs (reuse `cancel_job_by_id` from `src/videoannotator/api/v1/jobs.py`), `storage.delete_job` each (T024 removes their video folders), then remove the run folder if still present, only when inside `results_root()` (FR-030)
- [ ] T053 [US7] Regenerate `viewer/src/api/schema.d.ts`; add `openResultsFolder(path)`, `batchResultsZipUrl(batchId)` and the `include_video` parameter on the job download to `viewer/src/api/client.ts`
- [ ] T054 [US7] Create `viewer/src/components/ResultsLocation.tsx` (shadcn `Card`/`Button`): location text, Copy button, "Open folder" when `useIngestAccess().can_open_folders`, and an optional "Download results" action; download through the existing authenticated download path (`viewer/src/hooks/useZipDownloader.ts` or the client's blob download) since the API needs the token
- [ ] T055 [US7] Add `ResultsLocation` to `viewer/src/pages/BatchDetail.tsx` (run folder + run zip download) and `viewer/src/pages/JobDetail.tsx` (video folder); on JobDetail make "Download Results" exclude the video, with an "Include video" checkbox next to it
- [ ] T056 [US7] Update the delete confirmation in `viewer/src/components/JobDeleteButton.tsx` to name the results folder that will be deleted (from `results_folder.display_path`) and say the original video is not touched (FR-030); add a "Delete run" button to `viewer/src/pages/BatchDetail.tsx` (calling `DELETE /batches/{id}` through a new `deleteBatch` in `viewer/src/api/client.ts`) whose confirmation names the run folder and the number of videos, and says the originals are not touched

**Checkpoint**: quickstart "Story 7" passes. Commit and push. **MVP complete (US6 + US1 + US2 + US7).**

---

## Phase 7: User Story 4 — A video moved or deleted after its run (P2)

**Goal**: missing videos produce messages naming the video and its expected location; results stay
viewable; reruns list missing videos before starting.

**Independent test**: run 3 videos, move one; its job's results still show with a "Video not found
at …" player message; "Run again" lists the missing video first and can run the rest.

### Tests for User Story 4

- [ ] T057 [P] [US4] Add to `tests/api/test_rerun.py`: `POST /api/v1/batches/{id}/rerun?check=true` returns `created: []` and lists the moved video in `skipped` with `RERUN_VIDEO_MISSING`, creating no jobs or run folder; without `check` the rest still run
- [ ] T058 [P] [US4] Add to `tests/api/test_rerun.py`: with one video moved into `site_b/`, `rerun?check=true&relocate_folder=<site_b>` lists it in `relocated` and creates nothing; without `check` it reruns from the new path; a same-named file of a different size is not matched; a `relocate_folder` outside the allowed folders → 422
- [ ] T059 [P] [US4] Add a unit test (extend the job-execution tests from T015) that a job whose `video_path` is missing at start fails with `Video not found: <display path> (moved or deleted since the job was created)` and the next job in the batch still runs (FR-016)
- [ ] T060 [P] [US4] Add to `tests/api/` (e.g. `tests/api/test_result_files.py`): `JobResponse.video_available` is false after the video is removed and `GET /jobs/{id}/results` still returns all outputs (FR-014)

### Implementation for User Story 4

- [ ] T061 [US4] In `src/videoannotator/batch/job_execution.py`, check `Path(job.video_path).exists()` before running pipelines; on failure mark the job failed with the R10 message (via `results_folder.display_path`) and update its `run.json` entry
- [ ] T062 [US4] Add `video_available: bool` to `JobResponse` in `src/videoannotator/api/v1/jobs.py` (computed from `video_path` existence at response time)
- [ ] T063 [US4] Add `check: bool = False` to `rerun_batch` in `src/videoannotator/api/v1/batches.py`: compute `skipped` with the existing missing-video logic and return without creating jobs or a run folder. Also accept `relocate_folder` / `recursive`: check the folder with `resolve_within_roots`, match each missing video by name and size (`source.size_bytes` from `run.json`, else the job's recorded size), rerun matches from the new path, and return `relocated: [{job_id, from, to}]` (contracts/api.md)
- [ ] T064 [US4] Regenerate `viewer/src/api/schema.d.ts`; in `viewer/src/components/VideoPlayer.tsx` (or where `JobResultsViewer.tsx`/`JobDetail.tsx` mounts it) show "Video not found at <location>" in place of playback when `video_available` is false, keeping all annotation panels
- [ ] T065 [US4] In `viewer/src/components/RunAgainActions.tsx` call rerun with `check=true` first; when anything is skipped, list each missing video and its expected location in a dialog with "Run the rest", "Locate…" and "Cancel". "Locate…" opens the My folders picker in folder-only mode, re-runs the check with `relocate_folder`, and shows which videos matched before submitting (FR-015). Datasets keep using `DatasetDriftDialog`

**Checkpoint**: quickstart "Story 4" passes. Commit and push.

---

## Phase 8: User Story 3 — Same experience under Docker (P2)

**Goal**: the documented Docker setup gives My folders, in-place reading and host-side results, on
loopback only.

**Independent test**: `VIDEOS_DIR=~/Studies RESULTS_DIR=~/VideoAnnotator docker compose up videoannotator-prod`, then repeat Story 1 from the host browser.

### Tests for User Story 3

- [ ] T066 [P] [US3] Add to `tests/api/test_ingest_access.py`: with `PUBLISHED_LOCALLY` and `HOST_PATHS="/videos=/home/ada/Studies;/results=/home/ada/VideoAnnotator"`, `allowed_folders[0].display_path` and `results_root.display_path` are host paths, and `can_open_folders` is false inside a container (patch the container check); with `/videos` empty, `reason` mentions `VIDEOS_DIR`
- [ ] T067 [P] [US3] Add a static test (e.g. `tests/unit/test_docker_compose_contract.py`, parsing `docker-compose.yml` with `yaml.safe_load`) asserting `videoannotator-prod` and `videoannotator-gpu` publish `127.0.0.1:18011:18011`, mount `/videos` read-only and `/results`, and set the four env vars from contracts/docker.md

### Implementation for User Story 3

- [ ] T068 [US3] Update `docker-compose.yml` services `videoannotator-prod` and `videoannotator-gpu` per contracts/docker.md (loopback port, `${VIDEOS_DIR:-./videos}:/videos:ro`, `${RESULTS_DIR:-~/VideoAnnotator}:/results`, the four env vars); keep existing volumes, including `./data:/app/data:ro`; check whether `~` expands in compose interpolation and use `${HOME}/VideoAnnotator` if it doesn't
- [ ] T069 [US3] Treat an allowed folder that is missing or empty as unusable for `can_read_in_place` in `src/videoannotator/api/v1/ingest.py`, with the `VIDEOS_DIR` reason when running in a container (FR-012)
- [ ] T070 [US3] In `viewer/src/pages/NewJob.tsx` / `viewer/src/components/ServerFolderPicker.tsx`, when `same_machine` but not `can_read_in_place`, show `reason` with a link to the Docker docs section, and keep upload working
- [ ] T071 [US3] Update `docs/installation/INSTALLATION.md` Docker section: `VIDEOS_DIR` / `RESULTS_DIR`, the `docker run` equivalent, the loopback guarantee and the "remove `VIDEOANNOTATOR_PUBLISHED_LOCALLY` if you publish more widely" warning, no Open folder under Docker, and root-owned results with the `--user "$(id -u):$(id -g)"` option
- [ ] T072 [US3] Run quickstart "Story 3" against a locally built image (`docker compose build videoannotator-prod`), record the outcome in the commit message; if Docker is unavailable in the dev container, say so explicitly and leave this task unchecked

**Checkpoint**: quickstart "Story 3" passes. Commit and push.

---

## Phase 9: User Story 5 — A saved dataset of videos from My folders reruns with no prompts (P3)

**Goal**: a dataset saved from a selection runs exactly those videos in place, with no prompts.

**Independent test**: save a 3-video selection as a dataset, use it: straight to pipelines, 3 jobs,
no copies.

### Tests for User Story 5

- [ ] T073 [P] [US5] Add to `tests/api/test_dataset_preset_endpoints.py`: create/update/get round-trip `server_selection`; existing datasets default to false; migration adds the column to an old database (copy the pattern of the existing `server_folder_recursive` migration test, if any)
- [ ] T074 [P] [US5] Add to `viewer/src/test/components/DatasetPicker.test.tsx` (and `viewer/src/test/lib/datasetMatch.test.ts` if matching lives there): a selection dataset whose files are all present goes straight to pipelines with exactly its files; a new file in the folder is not a difference; a missing file is reported via the drift dialog

### Implementation for User Story 5

- [ ] T075 [US5] Add `server_selection` (`Boolean, nullable=False, default=False`) to the saved dataset model in `src/videoannotator/database/models.py`, the `ALTER TABLE saved_datasets ADD COLUMN server_selection BOOLEAN NOT NULL DEFAULT 0` entry next to `server_folder_recursive` in `src/videoannotator/database/migrations.py`, and pass-through in `src/videoannotator/database/crud.py`
- [ ] T076 [US5] Add `server_selection: bool = False` to `DatasetCreateRequest`, `DatasetUpdateRequest` and `DatasetResponse` in `src/videoannotator/api/v1/datasets.py`, and to `_to_response`
- [ ] T077 [US5] Regenerate `viewer/src/api/schema.d.ts`; in `viewer/src/components/SaveDatasetDialog.tsx` save a My folders selection as `server_folder` + manifest of the selected relative paths + `server_selection: true` (whole folder with nothing unticked stays a folder dataset)
- [ ] T078 [US5] In `viewer/src/components/DatasetPicker.tsx` and `viewer/src/lib/datasetMatch.ts`, for `server_selection` datasets compare only manifest files (ignore additions), and when all are present submit ingest with `path=server_folder` and `files=<manifest>` without prompting; missing files go through `viewer/src/components/DatasetDriftDialog.tsx` (Story 4 scenario 2)

**Checkpoint**: quickstart "Story 5" passes. Commit and push.

---

## Phase 10: User Story 8 — Choose where results go (P3)

**Goal**: the researcher sets the results folder once; new runs go there, old runs stay viewable.

**Independent test**: restart with `VIDEOANNOTATOR_RESULTS_DIR=~/elsewhere`, run a video, confirm its
results there and that earlier runs still open.

- [ ] T079 [P] [US8] Add to `tests/api/test_results_folder.py`: create a run, change `RESULTS_DIR` (fixture), create another; the second lands in the new root; the first job's results and `results_folder` still resolve to the old location; a `VIDEOANNOTATOR_RESULTS_DIR` line in `.env` is honoured when the variable isn't set in the environment
- [ ] T080 [US8] Show the results folder (`results_root.display_path` from `useIngestAccess`) on `viewer/src/pages/Settings.tsx`, with a short note on how to change it (`VIDEOANNOTATOR_RESULTS_DIR`, or `RESULTS_DIR` under Docker) and that earlier runs stay where they were written
- [ ] T081 [US8] Make sure result lookup (`src/videoannotator/batch/result_files.py: job_folder`) and the run zip use each job's stored `output_dir`, never `results_root()`, so a root change doesn't hide old runs (FR-025); show the R10-style "Results not found at <folder>" message on JobDetail when `output_dir` no longer exists (edge case: run folder renamed outside VideoAnnotator)

**Checkpoint**: quickstart "Story 8" passes. Commit and push.

---

## Phase 11: Polish & cross-cutting

- [ ] T082 [P] Add a CHANGELOG entry in `CHANGELOG.md`: My folders default and in-place reading, results folder (`~/VideoAnnotator`, `VIDEOANNOTATOR_RESULTS_DIR`), `run.json`, run zip, **changed default**: job artifacts zip omits the video (`include_video=true` restores it), Docker now publishes on loopback only with `VIDEOS_DIR`/`RESULTS_DIR`
- [ ] T083 [P] Update the getting-started docs (`docs/` getting-started page and `viewer/src/pages/GettingStarted.tsx` if it describes uploading) to describe My folders and where results go
- [ ] T084 [P] Update `docs/development/handover_walkthrough_rc1.md` with quickstart.md's input/output section (the file already has uncommitted edits — read and merge, don't overwrite)
- [ ] T085 Rebuild the viewer with `bash scripts/build_viewer.sh` and commit `src/videoannotator/viewer_static/`
- [ ] T086 Run the full gate: `pytest tests/`, `pytest tests/integration/test_output_baseline.py` (outputs must match baseline — FR-007, results only moved), `ruff check .`, `mypy src/videoannotator`, `cd viewer && bun run lint && bun run test:run`, `cd viewer && bunx tsc --noEmit -p tsconfig.app.json` (no more errors than T001's count), `bash scripts/build_viewer.sh --check` (bundle not larger than T001's size beyond the new components), `pre-commit run --all-files`
- [ ] T087 Walk through the whole of [quickstart.md](quickstart.md) on a plain install, including "Upload route" (uploaded copy in internal storage, results in `~/VideoAnnotator`), and the SC-002 check `find ~/.local/share/videoannotator/jobs ~/VideoAnnotator -name '*.mp4'`; note results in the commit message
- [ ] T088 Tick the 022 items in `docs/development/roadmap_v1.6.0.md` (if listed) and update the "Current Plan" line in `CLAUDE.md` to point at the next spec

---

## Dependencies & execution order

### Phase dependencies

- **Setup (1)** → **Foundational (2)** → all story phases.
- **US6 (3)** must precede every other story: it adds `plan_run`, which US1/US2 ingest, US5 dataset
  runs and US7's run zip build on.
- **US1 (4)** precedes **US2 (5)** (same picker component and ingest call) and **US7 (6)**'s
  "Open folder" visibility (uses `useIngestAccess`).
- **US4 (7)**, **US3 (8)**, **US8 (10)** depend only on US6 + US1; they can run in any order.
- **US5 (9)** depends on US2 (`files` on ingest and the selection UI).
- **Polish (11)** after all desired stories.

### Story dependency graph

```
Setup → Foundational → US6 ─┬─ US1 ─┬─ US2 ── US5
                            │       ├─ US7
                            │       ├─ US3
                            │       └─ US4
                            └─ US8
```

### Within each story

Tests first (they should fail), then server code, then `schema.d.ts` regeneration, then viewer code.
`schema.d.ts` regeneration tasks touch one shared file, so never run two in parallel.

---

## Parallel opportunities

- **Phase 2**: T010 (tests) in parallel with T011–T012 once T005–T009 exist.
- **US6**: T013, T014, T015 together; T018–T021 touch different files (`ingest.py`, `jobs.py`,
  `batches.py`, `datasets.py`) but T019/T020 both edit `jobs.py` — do those two sequentially.
- **US1**: T028 and T029 together; T033 alongside T031.
- **US7**: T044–T048 together; T049 (`results.py`), T050 (`batches.py`; T052 follows it, same file) and T051 (`artifacts.py`)
  in parallel.
- **After US1**: US7, US3, US4 and US8 can be worked by different people at once, coordinating only
  on `schema.d.ts` regeneration and `NewJob.tsx` (US1, US3).

### Parallel example: User Story 7

```
Task: "T044 tests/api/test_results_open.py"
Task: "T045 + T046 run zip and run delete tests in tests/api/test_batch_endpoints.py"
Task: "T047 artifacts include_video test"
Task: "T048 viewer/src/test/components/ResultsLocation.test.tsx"
then
Task: "T049 POST /results/open in src/videoannotator/api/v1/results.py"
Task: "T050 results.zip in src/videoannotator/api/v1/batches.py"
Task: "T051 include_video in src/videoannotator/api/v1/endpoints/artifacts.py"
```

---

## Implementation strategy

### MVP first

1. Setup + Foundational.
2. **US6** (results folder, server side) — already valuable from the CLI and API alone. Stop and validate with quickstart Story 6.
3. **US1** + **US2** (My folders, selection) — the inputs half of the P1 value.
4. **US7** (results from the viewer) — completes the P1 set. This is the MVP for rc1.

### Incremental delivery

Each phase ends at a checkpoint that is committed and pushed to `1.6-dev` when green. After the
MVP: US4 (missing videos, needed because in-place reading makes them normal), US3 (Docker), then the
P3 stories US5 and US8, then Polish.

---

## Notes

- Never write into allowed (video) folders; never overwrite results — both enforced in
  `results_folder.py`, not in callers.
- Jobs created before this feature keep `output_dir=None` and must keep working everywhere
  (FR-018); every new code path that reads `output_dir` handles `None`.
- No new dependencies (plan Technical Context).
