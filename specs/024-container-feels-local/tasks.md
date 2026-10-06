---
description: "Task list for 024: a container that feels local"
---

# Tasks: A Container That Feels Local

**Input**: Design documents from `specs/024-container-feels-local/` (plan.md, spec.md, research.md
R1–R16, data-model.md, contracts/launcher.md, contracts/api.md, contracts/image.md, quickstart.md)

**Tests**: Requested by plan.md (Testing) and research.md R12: pytest for server changes, Vitest
for the viewer, `bats` and Pester over one shared case table for the launcher, shellcheck and
PSScriptAnalyzer, and a Linux CI end-to-end job with Docker and with Podman. Test tasks come before
the implementation they cover and must fail first.

**Organization**: By user story. Phases follow plan.md's Delivery Order, which puts US4 (server
only, smallest, the screen that started this spec) ahead of US1. US1, US2 and US4 are all P1.
Work lands on `1.6-dev`; commit and push each green step.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files, no dependency on an unfinished task)
- **[Story]**: US1–US6, from spec.md
- Paths are from the repository root

## Conventions every task follows

- **Launcher (sh)**: `launcher/videoannotator-start` is POSIX `sh` (no bashisms; shellcheck
  clean, `# shellcheck shell=sh`). Logic lives in small named functions (`va_*`) so `bats` can
  source the script with `VA_SOURCE_ONLY=1` and call them without running `main`. Functions that
  run the engine call it through one `va_engine` wrapper so tests can stub it.
- **Launcher messages**: exactly the wording of contracts/launcher.md § Messages; one line plus
  the next step; no unexplained container vocabulary (FR-005). Exit codes: 0 started or already
  running, 1 problem, 2 cancelled.
- **Shared cases**: `tests/launcher/cases.json` is the single source of expected behaviour for
  both scripts (R12). Every pure-function behaviour added below adds rows there first.
- **Server**: ruff (line-length 88), mypy clean; comments only for non-obvious *why*.
- **Viewer**: after viewer changes, `cd viewer && bun run lint && bunx tsc --noEmit -p tsconfig.app.json && bun run test:run`, then
  `bash scripts/build_viewer.sh` and commit `src/videoannotator/viewer_static/` (CI checks it).

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: The launcher folder, its test harness and its lint gates.

- [X] T001 Create `launcher/` with an executable `launcher/videoannotator-start` skeleton: `#!/bin/sh`, `set -eu`, a `VA_VERSION` constant read at release time (default to the package version in `pyproject.toml` via a placeholder `@VERSION@` that T076 resolves; when it is still unreplaced (a source checkout), use `:latest` unless `--image` is given and print "development launcher", per research R16), an empty `main "$@"` guarded by `[ "${VA_SOURCE_ONLY:-}" = 1 ] || main "$@"`, and `launcher/README.md` for maintainers (structure, the `VA_SOURCE_ONLY` convention, how to run `bats`/Pester, the shared-cases rule)
- [X] T002 [P] Create `tests/launcher/cases.json` with its schema documented at the top of `launcher/README.md`: an object of named groups (`paths`, `classify`, `settings`, `run`, `engine`, `messages`), each a list of `{ "name", "os", "input": {...}, "expect": {...} }` rows; seed one row per group from contracts/launcher.md's Linux/Podman example so later tasks extend rather than invent the format
- [X] T003 [P] Create `tests/launcher/helpers.bash` (loads `cases.json` with `jq`, sources the launcher with `VA_SOURCE_ONLY=1`, stubs `va_engine` to record its argv into `$BATS_TEST_TMPDIR/engine.log`, sets `HOME`/`XDG_CONFIG_HOME` to temp dirs) and an empty smoke test `tests/launcher/smoke.bats` that sources the script
- [X] T004 [P] Add launcher checks to `.pre-commit-config.yaml`: extend the existing shellcheck hook's `files:` to cover `launcher/videoannotator-start` and `launcher/install.sh` (no `.sh` extension on the former), and add a local `bats tests/launcher` hook (`language: system`, `pass_filenames: false`, `files: ^(launcher/|tests/launcher/)`)
- [X] T005 [P] Add a `launcher` job to `.github/workflows/ci-cd.yml`: on `ubuntu-latest`, install `bats` and `jq`, run `shellcheck launcher/videoannotator-start launcher/install.sh` and `bats tests/launcher`, as a matrix over `ubuntu-latest` and `macos-latest` (no engine needed; catches BSD-userland differences, research R12); and a `launcher-windows` job on `windows-latest` that runs `Invoke-ScriptAnalyzer -Path launcher -Recurse -Severity Warning -EnableExit` and `Invoke-Pester tests/launcher` (both pass trivially until Phase 9 adds `.ps1` files)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Server settings, image layout and the published image that every story's launcher run
depends on.

**⚠️ CRITICAL**: Launcher stories (US1–US3, US5, US6) need T006–T012. US4 needs T006, and T010 for its compose half.

- [X] T006 Add to `src/videoannotator/config_env.py`: `missing_shares() -> list[str]` from `VIDEOANNOTATOR_MISSING_SHARES` (`;`-separated host paths), `LAUNCHER_ENV = "VIDEOANNOTATOR_LAUNCHER"` with `managed_by_launcher() -> bool` (via `get_bool_env`), `RESULTS_OWNER_ENV = "VIDEOANNOTATOR_RESULTS_OWNER"` with `results_owner() -> tuple[int, int] | None` (parses `uid:gid`, returns `None` when unset or malformed and logs a warning for malformed), and `LAUNCHER_REQUESTS_DIR = Path("/app/launcher/requests")`; document both variables in `docs/usage/environment_variables.md`
- [X] T007 [P] Add tests for T006 to `tests/unit/test_config_env.py`: `managed_by_launcher()` true/false; `results_owner()` for `"1000:1000"`, unset, `"abc"`, `"1000"` (malformed → `None`); `missing_shares()` empty and with two paths
- [X] T008 Update `Dockerfile`: create `/app/cache` and `/app/launcher/requests` alongside the existing `mkdir -p /app/data ...` line (around line 78), set `ENV UV_CACHE_DIR=/app/cache/uv`, and declare the cache in the runtime stage so a named volume at `/app/cache` keeps uv's download cache (research R7; contracts/image.md)
- [X] T009 [P] Update `docker-compose.yml`: mount a new `videoannotator-cache` named volume at `/app/cache` in `videoannotator-dev`, `videoannotator-prod` and `videoannotator-gpu`, and declare it under top-level `volumes:` with `name: videoannotator-cache` (matching how `videoannotator-models`/`-database`/`-storage` are declared), so compose shares volumes with the launcher (FR-024)
- [X] T010 Update `docker-compose.yml` (research R8, Compose): in `videoannotator-prod` and `videoannotator-gpu`, set `VIDEOANNOTATOR_INGEST_ROOTS=${VIDEOS_DIR:+/videos}` so `/videos` is shared only when `VIDEOS_DIR` is set (the default `./videos` mount stays but is no longer read); change the `VIDEOS_DIR` row of the table in `docs/installation/INSTALLATION.md` (line ~384) from default `./videos` to "(none: nothing shared)"; T079 records the change
- [X] T011 Add an `image` job to `.github/workflows/ci-cd.yml` (contracts/image.md): build the slim image (`EXTRAS=""`) for `linux/amd64`, log in to `ghcr.io` with `GITHUB_TOKEN` (`permissions: packages: write`), push `ghcr.io/infantlab/videoannotator:<version>` on release tags and `:latest` from the default branch; keep the existing Docker Hub push as a mirror, gated on its secrets existing; add OCI labels (`org.opencontainers.image.source`, `.version`)
- [X] T012 Create `tests/launcher/Dockerfile.e2e` (`FROM localhost/videoannotator:ci`; copies `tests/fixtures/stub_pipeline.yaml` into the registry metadata directory and `tests/fixtures/stub_pipeline_module.py` onto the image's Python path, mirroring how the test suite registers the fixture) and a reusable CI step in `.github/workflows/ci-cd.yml` that builds the slim image as `localhost/videoannotator:ci`, then the e2e image as `localhost/videoannotator:e2e`, and saves both as workflow artifacts so T038/T067 can load them into Docker and Podman without GHCR. The fixture is never in the published image (`tests/unit/registry/test_no_test_fixtures_shipped.py` still passes; research R12)

**Checkpoint**: settings and image ready; stories can begin.

---

## Phase 3: User Story 4 - Never shown the inside of the container (Priority: P1) 🎯 first delivery

**Goal**: In a container, My folders lists only shared folders, never `/root`; with none shared it
says so plainly and offers upload; a video in a folder no longer shared says "isn't shared any
more", not "moved or deleted". Works however the container was started (compose included).

**Independent Test**: Start the container with no shared folder (compose without `VIDEOS_DIR`),
open the new-job wizard: no container folder anywhere, the message says how to share a folder,
Upload works. Then an old job whose folder is outside the current shares explains "isn't shared".

### Tests for User Story 4 ⚠️ write first, must fail

- [X] T013 [P] [US4] In `tests/api/test_ingest_access.py`, add tests with `in_container()` patched true and `VIDEOANNOTATOR_INGEST_ROOTS` empty: `allowed_roots()` is `[]`; `GET /api/v1/ingest/access` returns `can_read_in_place: false`, empty `allowed_folders` and `places`, `in_container: true`, and `reason` equal to the compose wording from contracts/api.md; with `VIDEOANNOTATOR_LAUNCHER=1`, `managed_by_launcher: true` and the launcher wording; `VIDEOANNOTATOR_INGEST_ROOTS=""` (compose without `VIDEOS_DIR`, T010) behaves as unset; `browse`/`scan` of `/root` return 403. Also assert that outside a container the home default is unchanged (Principle V)
- [X] T014 [P] [US4] Create `tests/unit/test_video_unavailable_reason.py` for the new `video_unavailable_reason(path)` in `src/videoannotator/results_folder.py`: a path outside every allowed root returns "`<display path of its top folder>` isn't shared with VideoAnnotator any more" (top folder = the path's first component under its old root, shown via `display_path`); a path inside an allowed root that doesn't exist returns today's "moved or deleted since the job was created" text; a host-path mapping (`VIDEOANNOTATOR_HOST_PATHS`) is applied to the folder named
- [X] T015 [P] [US4] Add to `tests/api/test_rerun.py`, `tests/api/test_batch_endpoints.py` and `tests/api/test_dataset_stored_run.py` one case each: a job (or stored run) whose video lies outside the current `INGEST_ROOTS` reports the "isn't shared" reason in rerun `skipped` reasons, in `video_available`'s explanation, and in the dataset differences

### Implementation for User Story 4

- [X] T016 [US4] In `src/videoannotator/api/v1/ingest.py` `allowed_roots()` (line ~85): when no root is configured and `in_container()` is true, return `[]` instead of `[Path.home().resolve()]`; update its docstring with the why (FR-016: the container's home is the inside of the box). Keep the home default outside a container
- [X] T017 [US4] In `src/videoannotator/api/v1/ingest.py`, replace `NO_VIDEO_FOLDER_DOCKER` (line ~263) with two constants, `NO_SHARE_LAUNCHER` ("VideoAnnotator can only see folders you share with it. To share one, run: videoannotator-start share") and `NO_SHARE_COMPOSE` ("VideoAnnotator can only see folders you share with it. Set VIDEOS_DIR when starting it (see the installation guide).") per contracts/api.md, chosen in `access()` by `managed_by_launcher()`; update every other reference to the old constant (grep for it)
- [X] T018 [US4] Add `in_container: bool` and `managed_by_launcher: bool` to `IngestAccessResponse` in `src/videoannotator/api/v1/ingest.py` and set them in `access()`; add the same optional fields to the `IngestAccess` type in `viewer/src/types/ingest.ts`
- [X] T019 [US4] Add `video_unavailable_reason(path: Path | str) -> str` to `src/videoannotator/results_folder.py` (research R9), importing `allowed_roots` lazily from `api/v1/ingest.py` to avoid an import cycle, or moving `allowed_roots` to a shared module if that is cleaner; make T014 pass
- [X] T020 [US4] Use `video_unavailable_reason` everywhere a missing video is explained: `src/videoannotator/batch/job_execution.py` (the "moved or deleted since the job was created" message at line ~92), `video_available` and rerun checks/relocation in `src/videoannotator/api/v1/jobs.py` and `src/videoannotator/api/v1/batches.py`, and dataset differences in `src/videoannotator/api/v1/datasets.py`; `find_moved_video` relocation must not search outside allowed roots; make T015 pass
- [X] T021 [P] [US4] Add a Vitest test in `viewer/src/test/ServerFolderPicker.test.tsx` (the repo's Vitest location): with access `{ can_read_in_place: false, in_container: true, reason: <launcher wording> }` the picker shows the reason verbatim, shows an Upload action, and lists no folders
- [X] T022 [US4] Update `viewer/src/components/ServerFolderPicker.tsx` so the no-share card shows the server's `reason` as the headline ("VideoAnnotator can only see folders you share with it") with the how-to on its own line and an Upload button that switches the wizard to upload; remove any client-side Docker wording now owned by the server; make T021 pass, then rebuild `viewer_static`

**Checkpoint**: US4 done and shippable on its own; compose installs already stop showing `/root`.

---

## Phase 4: User Story 1 - Start VideoAnnotator and share my video folder, once (Priority: P1) 🎯 MVP

**Goal**: On Linux/macOS, `videoannotator-start` finds a running engine, asks for the video folder
(picker), confirms in plain words, shares it read-only at its real path, starts the image on
127.0.0.1, gets an admin key, and opens the viewer connected. Results belong to the researcher.

**Independent Test**: On a machine with only Docker or Podman, run `videoannotator-start`, choose
a folder, and run one video from My folders, typing no container command, path or setting
(quickstart Story 1). CI: T038 does this non-interactively.

### Tests for User Story 1 ⚠️ write first, must fail

- [X] T023 [P] [US1] Add `paths` rows to `tests/launcher/cases.json` and `tests/launcher/paths.bats` for `va_container_path HOSTPATH OS`: Linux/macOS real path kept (`/home/ada/Studies`); first component in the image's own top-level dirs (`app bin boot dev etc lib lib32 lib64 libx32 opt proc root run sbin srv sys tmp usr var`) goes under `/host` (`/opt/data` → `/host/opt/data`); Windows `C:\Users\ada\Studies` → `/c/Users/ada/Studies`; spaces, accents and non-Latin characters preserved; trailing separators removed; a path containing `:`, `;`, `=` or `,` (`/home/ada/Study: 2024`, `/data/a,b`, `/data/x=y`) goes under `/host/<n>`, numbered in share order (research R3)
- [X] T024 [P] [US1] Add `classify` rows and `tests/launcher/classify.bats` for `va_classify PATH`: `ok`; `broad` with its explanation for home, `/`, `/Volumes`, `/media/<user>`, `/mnt`, `/Users`, `/etc`, `/usr`; `refused` for the results folder itself and for a folder inside the results folder (message "That folder is inside your results folder, which VideoAnnotator can already read."); `ok` with `nested_results` for a share containing the results folder; `duplicate` for a path equal to or inside an existing share; `replaces` for a path containing existing shares (research R4)
- [X] T025 [P] [US1] Add `settings` rows and `tests/launcher/settings.bats` for `va_settings_path`, `va_settings_read`, `va_settings_write`: location per OS (`$XDG_CONFIG_HOME/videoannotator/start.conf`, default `~/.config/...`; macOS `~/Library/Application Support/VideoAnnotator/start.conf`); `key=value` lines with repeated `share=`; round-trip preserves order and paths with spaces; written file is mode 600; `requests/` created beside it (data-model.md, research R13)
- [X] T026 [P] [US1] Add `run` rows and `tests/launcher/run.bats` for `va_build_run`: produces exactly the argument list of contracts/launcher.md § "The `run` it builds" (shares, results and requests as `--mount type=bind,…`, never `-v`; the results mount after every read-only share; `VIDEOANNOTATOR_MISSING_SHARES` with the missing shares' host paths; name, `-p 127.0.0.1:PORT:18011`, four named volumes including `videoannotator-cache`, each share `:ro` at its container path, results writable, the requests folder at `/app/launcher/requests`, the `-e` settings with `HOST_PATHS` pairs `container=host` joined by `;`, the fully qualified image); Docker on Linux adds `VIDEOANNOTATOR_RESULTS_OWNER=<uid>:<gid>`, Podman does not and never passes `--userns=keep-id`; no `--restart` flag (research R2, R6)
- [X] T027 [P] [US1] Add `tests/unit/test_results_owner.py`: with `VIDEOANNOTATOR_RESULTS_OWNER` set and `os.chown` patched, every folder `RunFolder` creates under the results root, every file in a job's folder, and `run.json` are chowned to that owner when `record_job_finished` runs; with it unset, `os.chown` is never called; a chown `PermissionError` is logged, not raised
- [X] T028 [P] [US1] Add `tests/unit/test_display_paths_windows.py`: with `VIDEOANNOTATOR_HOST_PATHS='/c/Users/ada/Studies=C:\Users\ada\Studies'`, `display_path('/c/Users/ada/Studies/Day 1/a.mp4')` is `C:\Users\ada\Studies\Day 1\a.mp4` (remaining separators become backslashes when the host prefix is a Windows path); `host_paths()` keeps a host value with backslashes intact and strips a trailing `\`; Linux pairs unchanged

### Implementation for User Story 1 — server

- [X] T029 [P] [US1] In `src/videoannotator/results_folder.py`, give results to `results_owner()` (research R6): chown each directory created under the results root at creation (in `RunFolder` and `check_writable`'s mkdir), and in `record_job_finished` chown every file under the job's folder plus `run.json`; swallow and log `OSError`; make T027 pass
- [X] T030 [US1] In `src/videoannotator/config_env.py` `host_paths()` and `src/videoannotator/results_folder.py` `display_path()`: treat a host prefix matching `^[A-Za-z]:\\` as Windows — strip a trailing `\` rather than `/`, and render the remainder of a mapped path with `\` separators; make T028 pass

### Implementation for User Story 1 — launcher (sh)

- [X] T031 [US1] In `launcher/videoannotator-start`, implement `va_os` (linux/macos via `uname`), `va_settings_path`, `va_settings_read` (into shell variables plus a newline-separated `VA_SHARES`), and `va_settings_write` (atomic write via temp file + `mv`, `umask 077`, creating `requests/`); make T025 pass
- [X] T032 [US1] Implement `va_container_path` and `va_classify` (with the broad-share explanation strings from contracts/launcher.md § Messages, "Broad share"); make T023 and T024 pass
- [X] T033 [US1] Implement `va_engine_detect` for the simple case: whichever of `docker`/`podman` responds to `info` (Docker first when both respond and no `engine=` is saved); set `VA_ENGINE`; implement the `va_engine` wrapper that runs `"$VA_ENGINE" "$@"`
- [X] T034 [US1] Implement `va_build_run` and `va_start` (`va_engine run -d ...` from `va_build_run`; detect "already exists" by removing a stopped `videoannotator` container first); make T026 pass
- [X] T035 [US1] Implement `va_pick_folder` (research R5): macOS `osascript -e 'POSIX path of (choose folder ...)'`; Linux `zenity --file-selection --directory`, else `kdialog --getexistingdirectory`; each opening in `~/Videos` (macOS `~/Movies`) if present, else `~/Documents`; fallback text prompt listing likely folders under home that hold videos (Videos, Movies, Desktop, Documents; ≤2 levels; ≤2,000 entries looked at); returns status 2 on cancel
- [X] T036 [US1] Implement the first-run flow in `main` (contracts/launcher.md § "What it prints"): "Starting VideoAnnotator with <Engine>."; ask "Which folder are your videos in?" via `va_pick_folder`; classify; for `broad`, ask the broad-share question defaulting to No and loop back to the picker on No; confirm "VideoAnnotator will be able to read, but never change: … Results go to: … Share this folder? [Y/n]"; create the results folder (`~/VideoAnnotator`) if missing; save settings; options `--share PATH` (repeatable), `--results PATH`, `--yes`, `--no-browser`, `--port N`, `--image REF`, `--engine` parsed up front so CI runs non-interactively
- [X] T037 [US1] Implement readiness and connection (research R14): `va_wait_ready` polls `http://127.0.0.1:PORT/api/v1/system/health` (curl, else wget) with "Starting... ready."; `va_ensure_key` runs `va_engine exec videoannotator videoannotator generate-token --user researcher@localhost --key-name "start-up program" --admin --output /tmp/key.json`, reads the key with `va_engine exec ... cat`, deletes the file, stores `key=` in settings, and regenerates when the saved key is rejected (401 from `GET /api/v1/auth/me`); `va_open_browser` opens `http://127.0.0.1:PORT/viewer-connect?token=<key>` with `xdg-open`/`open` unless `--no-browser`; print the FR-014 line "VideoAnnotator can read: …. Results: …." using host paths
- [X] T038 [US1] Add `tests/launcher/test_launcher_e2e.sh` and a `launcher-e2e` job in `.github/workflows/ci-cd.yml` (matrix `engine: [docker, podman]`, loads T012's image): run `launcher/videoannotator-start --engine $engine --share $TMP/videos --results $TMP/results --yes --no-browser --image localhost/videoannotator:e2e`; submit a job with `stub_pipeline` on a video in the share via the API using the stored key; assert the job completes, its files under `$TMP/results` are owned by the runner's uid, `GET /api/v1/ingest/access` lists only the share (never `/root`), and `browse` of `/root` is 403 (SC-004)
- [X] T039 [US1] Implement `stop` (`va_engine stop -t 30 videoannotator` then `rm`) and `logs` (`va_engine logs --tail 200 videoannotator`) commands in `launcher/videoannotator-start` so a first-time researcher can stop it

**Checkpoint**: MVP — Linux/macOS researchers can start, share one folder and run a video.

---

## Phase 5: User Story 2 - Start again with no questions (Priority: P1)

**Goal**: Later starts reuse the saved shares and results without asking, skip missing shares
with a note, open the running copy instead of starting a second, survive updates — including
installed pipelines, which are restored after every container recreation.

**Independent Test**: After US1, stop and start twice, once after `videoannotator-start update`:
no question; same folders; earlier results, datasets and installed pipelines still work
(quickstart Story 2).

### Tests for User Story 2 ⚠️ write first, must fail

- [X] T040 [P] [US2] Add `tests/launcher/restart.bats`: with a saved `start.conf`, `main` asks nothing (stdin closed) and builds the same run; a share whose folder is missing is left in the file, omitted from the run, and produces "Couldn't find <path> (an unplugged drive?), so it isn't shared this time."; when `va_engine ps` shows `videoannotator` running, `main` opens the browser and exits 0 without `run` (FR-006); a saved `image=` older than the launcher's own version is replaced by the launcher's pinned image
- [X] T041 [P] [US2] Add `tests/launcher/migrate.bats`: no `start.conf` but `VIDEOS_DIR`/`RESULTS_DIR` set (or `va_engine volume inspect videoannotator-database` succeeds) → offers "Use your existing VideoAnnotator folders?" and, on yes (or `--yes`), seeds `share=`/`results=` from them; when only the volumes exist, prints "Your jobs and models will be kept." and continues with the normal first-run picker (FR-024, research R13)
- [X] T042 [P] [US2] Create `tests/api/test_extras_restore.py`: given a completed `extras_install_jobs` row for group `scene` whose packages are not importable (patch the importability check), server start schedules exactly one reinstall through the existing installer and readiness reports state `restoring` for that group's pipelines; an importable group is not reinstalled; a failed or pending row is not "remembered"; while restoring, a pending job requesting a pipeline of that group is not picked up by `BackgroundJobManager._process_cycle` and is picked up once restore completes; a restore failure leaves the card showing the install error, and the waiting job then fails with that reason rather than waiting forever

### Implementation for User Story 2 — pipelines that survive (research R7)

- [X] T043 [US2] In `src/videoannotator/api/extras_install.py`, add `remembered_groups() -> list[str]` (distinct `extra_name` of `ExtrasInstallJob` rows with completed status, from `src/videoannotator/database/models.py`), `group_importable(extra) -> bool` (reusing the readiness/registry check for the group's modules), and `restore_missing_groups()` that, for each remembered non-importable group, creates a new install job row marked as a restore and calls `start_install`; track restoring groups in a module-level set exposed as `restoring(extra) -> bool`
- [X] T044 [US2] Call `restore_missing_groups()` from the lifespan in `src/videoannotator/api/main.py` after database initialisation and before background processing starts; log one line per group restored; never block startup
- [X] T045 [US2] In `src/videoannotator/api/readiness.py` `pipeline_readiness` (around the existing `state = "installing"` at line ~437), report `state = "restoring"` when the pipeline's extras group is restoring
- [X] T046 [US2] In `src/videoannotator/api/background_tasks.py` `_process_cycle`, skip (leave pending) jobs whose requested pipelines belong to a restoring group, and fail them with the install error if that group's restore failed; make T042 pass
- [X] T047 [P] [US2] Show "Restoring…" for the `restoring` state with the same progress display as installing in `viewer/src/components/LockedPipelineCard.tsx` and wherever `NewJob.tsx`/`Settings.tsx` render install state; add the state to the readiness type; add a Vitest case alongside the card's existing tests; rebuild `viewer_static`

### Implementation for User Story 2 — launcher (sh)

- [X] T048 [US2] In `launcher/videoannotator-start`, make `main` reuse saved settings with no questions: skip missing shares with the note (keep them in `start.conf`), detect an already running `videoannotator` container and just open the browser (FR-006, also covers "two copies started at once"), print the FR-014 line once; make T040 pass
- [X] T049 [US2] Implement first-run migration (FR-024) offering to reuse `VIDEOS_DIR`/`RESULTS_DIR` or detected existing volumes; make T041 pass
- [X] T050 [US2] Implement `update` (research R16, Updating): look up the latest release tag; if newer than `VA_VERSION`, download that release's `install.sh`, run it with `--no-shortcut --quiet`, and `exec` the new launcher as `videoannotator-start update --continue`; otherwise (or with `--continue`) pull `ghcr.io/infantlab/videoannotator:<VA_VERSION>` (the launcher's pinned image), save `image=`, then restart via a new `va_restart` (stop -t 30, rm, `va_start`), which T058 extends with the running-jobs check; print "Updated. Your folders, results, models and installed pipelines are kept."; already latest: "VideoAnnotator is up to date." and no restart; add `tests/launcher/update.bats` rows with the release lookup and installer stubbed

**Checkpoint**: US1 + US2 + US4 — the P1 experience is complete on Linux/macOS.

---

## Phase 6: User Story 3 - Share another folder, or stop sharing one (Priority: P2)

**Goal**: `share`/`unshare`/`list` in the launcher with an announced restart that asks about
running jobs; Settings lists shared folders and the results folder and offers Stop sharing
through the one-way requests folder.

**Independent Test**: Share a second folder, run a video from it, stop sharing the first from
Settings, restart: My folders no longer lists it (quickstart Story 3).

### Tests for User Story 3 ⚠️ write first, must fail

- [X] T051 [P] [US3] Create `tests/api/test_shares.py`: `GET /api/v1/ingest/access` includes `shares` (`path`, `display_path`, `present`, `stop_requested`), with shares from `VIDEOANNOTATOR_MISSING_SHARES` listed as `present: false`; `POST /api/v1/ingest/shares/stop` (with `LAUNCHER_REQUESTS_DIR` patched to a temp dir) appends the host path to `stop-sharing.txt` and returns the share with `stop_requested: true`, and later access reads report it; 404 `SHARE_NOT_FOUND` for a non-share; 409 `NOT_MANAGED_BY_LAUNCHER` without `VIDEOANNOTATOR_LAUNCHER`; 403 for a non-admin or non-local caller; a repeated request doesn't duplicate the line; with `VIDEOANNOTATOR_RESULTS_OWNER` set the file is chowned to it (`os.chown` patched) (contracts/api.md)
- [X] T052 [P] [US3] Add `tests/launcher/share.bats`: `share PATH` adds a normalised share (duplicates and nested paths ignored; a containing path replaces the nested ones after confirmation if broad) and restarts; `unshare PATH` removes it; `list` prints shares, results, engine and image; at start, each line of `requests/stop-sharing.txt` that is a current share is removed and announced, other lines are ignored (never added), and the file is deleted, not truncated (it may be root-owned; research R10); with `va_engine`/curl stubbed to report 2 running jobs, the restart prompt is "2 videos are being processed. [W]ait for them, or [r]estart now (they'll be marked failed and can be retried)?"
- [X] T053 [P] [US3] Add Vitest tests for `VideosAndResultsCard` in `viewer/src/test/VideosAndResultsCard.test.tsx`: lists each share's `display_path` marked read-only (a `present: false` share marked "not found at the last start") and the results folder; with `managed_by_launcher` true shows Stop sharing, which calls `stopSharing(path)` and then shows "Stops when VideoAnnotator next starts"; with it false shows "To change shared folders, change the compose settings (see the installation guide)" and no button

### Implementation for User Story 3

- [X] T054 [US3] In `src/videoannotator/api/v1/ingest.py`, add a `Share` model and `shares` to `IngestAccessResponse` (present ones from the configured roots, missing ones from `VIDEOANNOTATOR_MISSING_SHARES` as `present: false`, each with `display_path` and `stop_requested` read from `LAUNCHER_REQUESTS_DIR / "stop-sharing.txt"`); add `POST /api/v1/ingest/shares/stop` guarded by `require_local_caller` and the admin check, matching on host (display) path or container path, appending the host path once with a plain append (`open(..., "a")`) and chowning the file to `results_owner()` when set; register error codes `SHARE_NOT_FOUND` and `NOT_MANAGED_BY_LAUNCHER` in `src/videoannotator/api/v1/errors.py`; make T051 pass
- [X] T055 [P] [US3] Add the `shares` field to `viewer/src/types/ingest.ts` and `stopSharing(path: string)` to `viewer/src/api/client.ts` (POST `/api/v1/ingest/shares/stop`)
- [X] T056 [US3] Add a "Shared folders" section to `viewer/src/components/VideosAndResultsCard.tsx` (rendered in `viewer/src/pages/Settings.tsx`): each share's path, "read-only", "not found at the last start" for missing ones, Stop sharing per share when `managed_by_launcher`, the pending note after a request, and the results folder; compose wording otherwise; make T053 pass; rebuild `viewer_static`
- [X] T057 [US3] In `launcher/videoannotator-start`, implement `va_apply_stop_requests` (called at every start before building the run), `share [PATH]` (picker when no PATH, classify, confirm, save), `unshare [PATH]` (numbered list when no PATH), and `list`; make the share-related parts of T052 pass
- [X] T058 [US3] Implement `va_restart` with the running-jobs check (research R11): `GET /api/v1/jobs?status_filter=running` with the saved key; if any, ask Wait/Restart (`--yes` means restart); Wait polls every 10 s; then `stop -t 30`, `rm`, `va_start`; announce "Restarting VideoAnnotator to share <folder>…"; `share`/`unshare`/`update` all use it; make T052 pass

**Checkpoint**: US3 works on Linux/macOS.

---

## Phase 7: User Story 5 - Plain words when something goes wrong (Priority: P2)

**Goal**: Every failure in contracts/launcher.md § Messages produces its one plain line; GPU used
when possible, CPU with a note otherwise.

**Independent Test**: Trigger each situation (engine missing, Docker stopped, Podman machine
stopped, port in use, download failure, out of memory, GPU unusable, broad share) and check the
message (quickstart Story 5).

### Tests for User Story 5 ⚠️ write first, must fail

- [X] T059 [P] [US5] Add `messages` rows to `tests/launcher/cases.json` and `tests/launcher/messages.bats` for `va_explain_error ENGINE STDERR EXITCODE`: maps engine stderr to the contract message for no engine, Docker not running ("Cannot connect to the Docker daemon", "error during connect"), port in use ("address already in use", "port is already allocated"), pull failure ("manifest unknown", "dial tcp", "TLS handshake"), out of memory (exit 137 / "OOMKilled"); Linux Docker service stopped ("Is the docker daemon running?"); no permission ("permission denied while trying to connect"); macOS Docker Desktop file sharing ("is not shared from the host", "Mounts denied"); name already in use ("already in use by container") → "VideoAnnotator is already running." and open the browser, exit 0; unknown errors print "VideoAnnotator couldn't start. Run: videoannotator-start logs" plus the raw line indented
- [X] T060 [P] [US5] Add `engine` rows and `tests/launcher/gpu.bats` for `va_gpu_flags`: Docker with `nvidia-smi` and `docker info` listing the `nvidia` runtime → `--gpus all`; Podman with `nvidia-ctk cdi list` showing devices → `--device nvidia.com/gpu=all`; otherwise none plus the "Running without the GPU…" note; a start that fails with the GPU flag is retried once without it and says so (research R15)

### Implementation for User Story 5

- [X] T061 [US5] In `launcher/videoannotator-start`, implement `va_explain_error` and route every `va_engine` failure in `va_engine_detect`, pull, `va_start` and `va_wait_ready` (including the container exiting during start, checked via `inspect`) through it, exiting 1; "no engine" when neither `docker` nor `podman` is on `PATH`; make T059 pass
- [X] T062 [US5] Start Podman's machine when needed (macOS; also Windows in Phase 9): if `podman` is installed but `podman info` fails and `podman machine list` shows a stopped machine, print "Starting Podman's virtual machine (first time takes a minute)…" and run `podman machine start`; if no machine exists, `podman machine init` first
- [X] T063 [US5] Implement `va_gpu_flags` and the one retry without GPU in `va_start`; make T060 pass
- [X] T064 [US5] Check the port before starting: if `127.0.0.1:PORT` answers and isn't VideoAnnotator, print the port message with the next port as the suggestion; print "Downloading VideoAnnotator (first time only, about 1 GB)..." before a first pull

**Checkpoint**: US5 works on Linux/macOS.

---

## Phase 8: User Story 6 - Docker or Podman, whichever I have (Priority: P2)

**Goal**: Engine choice per Story 6, and identical results with either engine.

**Independent Test**: Run US1–US4 with Podman and with Docker; with both running, the last-used one
is chosen and named (quickstart Story 6).

- [X] T065 [P] [US6] Add `engine` rows and `tests/launcher/engine.bats` for `va_engine_detect`: only Podman → podman with no extra step; both running and `engine=podman` saved → podman, announced; both running, nothing saved → docker; only one running → that one, regardless of saved; `--engine` overrides
- [X] T066 [US6] Extend `va_engine_detect` in `launcher/videoannotator-start` to the full rule (research R2), save `engine=` after a successful start, and name the engine in the first line; make T065 pass
- [X] T067 [US6] Extend the `launcher-e2e` job (T038) to also run US2's restart, US3's share/unshare (with a stop request written through the API) and US4's no-share case, on both Docker and Podman; assert results ownership for Podman rootless (SC-006 on Linux); queue a job before `share` and assert it is still pending, then completes, after the restart (FR-013); log the restart time and warn above 60 s (plan Performance Goals)
- [X] T068 [US6] Add a weekly `extras-restore-e2e` job in `.github/workflows/ci-cd.yml` (schedule plus `workflow_dispatch`): start the slim image via the launcher, install the `scene` group through `POST /api/v1/pipelines/extras/scene/install`, `videoannotator-start share` a second folder (recreating the container), and assert readiness passes through `restoring` to ready without re-downloading (uv cache hits in the log), and a scene job submitted meanwhile waits and then completes (research R7, R12)

**Checkpoint**: Every story works on Linux with both engines; macOS shares the script.

---

## Phase 9: Windows Launcher (Stories 1–6 on Windows)

**Purpose**: The PowerShell twin with the same behaviour, tested against the same cases (plan
Delivery step 7). Cross-cutting, so tasks name the stories they cover rather than carry labels.

- [X] T069 Create `launcher/videoannotator-start.ps1` (PowerShell 5.1+, `Set-StrictMode -Version Latest`) with the same function set as the sh script under `Verb-Noun` names (`Get-VaSettingsPath`, `ConvertTo-VaContainerPath`, `Get-VaClassification`, `New-VaRunArgs`, `Get-VaEngine`, `Get-VaGpuFlags`, `Get-VaErrorMessage`, …), dot-sourceable without running `Main` when `$env:VA_SOURCE_ONLY -eq '1'`; settings at `%APPDATA%\VideoAnnotator\start.conf`
- [X] T070 [P] Create `launcher/videoannotator-start.cmd`: `@powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0videoannotator-start.ps1" %*` (research R1)
- [X] T071 [P] Create `tests/launcher/Launcher.Tests.ps1` (Pester 5): load `cases.json` and run every `paths`, `classify`, `settings`, `run`, `engine` and `messages` row whose `os` is `windows` or `any` against the PowerShell functions; add Windows rows to `cases.json` (drive mapping `C:\…` → `/c/…`, `HOST_PATHS` keeping backslashes, broad shares `C:\`, `C:\Users`, `C:\Windows`, `C:\Program Files`, `%USERPROFILE%`, Docker Desktop GPU detection by `nvidia-smi` alone)
- [X] T072 Implement US1 on Windows in `launcher/videoannotator-start.ps1`: `System.Windows.Forms.FolderBrowserDialog` picker opening in Videos else Documents, first-run confirmation, run, readiness, key, `Start-Process` the viewer-connect URL, `stop`, `logs`; no `RESULTS_OWNER` (Docker Desktop and Podman own files correctly)
- [X] T073 Implement US2, US3, US5 and US6 on Windows in `launcher/videoannotator-start.ps1`: saved-settings start, missing-share note (drive letters), already-running, migration, `update`, `share`/`unshare`/`list`, stop requests, restart with running-jobs check, error mapping, Podman machine start, engine choice; make `tests/launcher/Launcher.Tests.ps1` pass in the `launcher-windows` CI job

---

## Phase 10: Polish & Cross-Cutting Concerns

- [X] T074 [P] Create `launcher/install.sh` (research R16): copies `videoannotator-start` from the matching GitHub release to `~/.local/bin`, makes it executable, warns if `~/.local/bin` isn't on `PATH`, and with `--shortcut` (default on; `--no-shortcut` to skip) adds "Start VideoAnnotator" as a `.desktop` file on Linux (`~/.local/share/applications` and the desktop) or a `.command` file on macOS
- [X] T075 [P] Create `launcher/install.ps1`: copies the `.ps1` and `.cmd` to `%LOCALAPPDATA%\VideoAnnotator`, adds it to the user `PATH`, and creates a "Start VideoAnnotator" `.lnk` on the desktop pointing at the `.cmd`
- [X] T076 Release wiring in `.github/workflows/ci-cd.yml`: on release tags, replace `@VERSION@` in the launcher scripts with the tag version (T001) and attach `videoannotator-start`, `videoannotator-start.ps1`, `videoannotator-start.cmd`, `install.sh` and `install.ps1` as release assets
- [X] T077 [P] Rewrite `docs/installation/INSTALLATION.md` launcher-first: install Docker Desktop or Podman Desktop, the one-line installer, first start; the one-sentence access statement for ethics applications (SC-007); GPU enablement per engine; macOS as community-tested and CPU-only; compose kept as the lab/server route (FR-023), with `VIDEOS_DIR` and the new `videoannotator-cache` volume
- [X] T078 [P] Update `docs/usage/GETTING_STARTED.md` to start with `videoannotator-start`, and add each contracts/launcher.md message with its fix to `docs/installation/troubleshooting.md` (the `<guide>` links in launcher messages point at these anchors)
- [X] T079 [P] Add a CHANGELOG entry (`CHANGELOG.md`): the launcher, GHCR slim image, pipelines restored after updates, the no-home-default change in containers (Complexity Tracking), compose sharing `/videos` only when `VIDEOS_DIR` is set (T010), Windows display paths, "isn't shared any more" messages
- [X] T080 [P] Update `README.md`'s install section launcher-first (Docker Desktop or Podman Desktop, the one-line installer, "Start VideoAnnotator"; compose for labs), and add or tick 024 in `docs/development/roadmap_v1.6.0.md` (constitution, Engineering Standards: Documentation)
- [X] T081 Run the full gates: `pytest tests/`, `ruff check .`, `mypy src/videoannotator`, `cd viewer && bun run lint && bunx tsc --noEmit -p tsconfig.app.json && bun run test:run`, `bats tests/launcher`, `pre-commit run --all-files`; fix anything failing
- [ ] T082 Walk `specs/024-container-feels-local/quickstart.md` on Linux with Docker and with Podman; record results (and any deviations) in a "Walkthrough log" section at the end of that file; confirm SC-002 (no container path on any screen)
- [ ] T083 Ask a pilot researcher (non-technical) to run quickstart Story 5 unaided on their own machine; record in the walkthrough log whether each message let them act without help (SC-005)
- [ ] T084 Walk the quickstart on Windows with Docker Desktop and with Podman Desktop (maintainer's machine, by hand); record results in the same log; ask the pilot lab to run it on macOS and note it as pending until they report

---

## Dependencies & Execution Order

### Phase dependencies

- **Setup (Phase 1)**: none.
- **Foundational (Phase 2)**: after Setup. T006 and T010 block US4; T006–T012 block the launcher stories.
- **US4 (Phase 3)**: needs only T006 and T010. Independent of every launcher task.
- **US1 (Phase 4)**: needs Phase 2. The MVP.
- **US2 (Phase 5)**: needs US1's launcher functions (T031–T037). Its server half (T042–T047)
  needs only Phase 2 and can run in parallel with US1.
- **US3 (Phase 6)**: needs US1 (start/run) and T050 (`va_restart`); server and
  viewer halves (T051, T053–T056) need only T018 and can run in parallel with US1/US2.
- **US5 (Phase 7)**, **US6 (Phase 8)**: need US1's launcher; independent of each other and of US3.
- **Windows (Phase 9)**: needs the sh behaviour settled (US1–US6) and its cases in `cases.json`.
- **Polish (Phase 10)**: T074–T080 can start once US1 is done; T081–T084 last.

### Within each story

Tests (and `cases.json` rows) first and failing → pure functions → flows that use them → e2e.
Server model/config before endpoints; endpoints before viewer.

## Parallel Opportunities

- Phase 1: T002, T003, T004, T005 together after T001.
- Phase 2: T007, T009 and T010 alongside T006/T008; T011 and T012 together.
- US4: T013, T014, T015, T021 together; then T016–T020 (same file for T016–T018, so sequential).
- US1: all six test tasks T023–T028 together; server tasks T029 and T030 in parallel with the
  launcher tasks T031–T037 (different files).
- Across stories, once Phase 2 is done, three streams can run at once:
  - **Server/viewer**: US4 → US2 server half (T042–T047) → US3 server/viewer (T051, T053–T056).
  - **Launcher sh**: US1 → US2 → US3 → US5/US6.
  - **CI/docs**: T038 harness, T074–T080.

### Example: US1 kick-off

```text
Task: "T023 paths cases + tests/launcher/paths.bats"
Task: "T024 classify cases + tests/launcher/classify.bats"
Task: "T025 settings cases + tests/launcher/settings.bats"
Task: "T026 run cases + tests/launcher/run.bats"
Task: "T027 tests/unit/test_results_owner.py"
Task: "T028 tests/unit/test_display_paths_windows.py"
```

### Example: US3 kick-off

```text
Task: "T051 tests/api/test_shares.py"
Task: "T052 tests/launcher/share.bats"
Task: "T053 VideosAndResultsCard Vitest"
Task: "T055 types/ingest.ts + api/client.ts stopSharing()"
```

## Implementation Strategy

### First delivery (US4)

Phases 1–3 up to T022: the server never shows the container's filesystem and says plainly why
My folders is empty. Small, fixes the screen that started the spec, and helps compose users at
once. Commit and push.

### MVP (US1)

Phase 4: a Linux/macOS researcher starts with one command, shares a folder, runs a video, and owns
the results. Validate with T038 on both engines.

### Incremental delivery

1. US4 → push.
2. US1 → push (MVP).
3. US2 (restored pipelines are useful under compose straight away) → push.
4. US3 → US5 → US6, each pushed when green.
5. Windows twin (Phase 9), installers, docs, then the walkthroughs (T082–T084).

## Notes

- `[P]` tasks touch different files and depend on nothing unfinished.
- Every task names its file; launcher behaviour is defined by `cases.json` rows, not by either
  script.
- No new runtime dependency anywhere (plan Technical Context); `bats`, `jq` and Pester are
  test-only.
