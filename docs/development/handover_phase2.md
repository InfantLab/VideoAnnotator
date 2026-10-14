# Handover: v1.6.0 Phase 2, "Clean First Contact"

**For**: the agent taking over v1.6.0 Phase 2, starting cold.
**From**: the agent that finished Phase 1, 2026-10-02.
**Branch**: `1.6-dev`. The state at handover is the commit that adds this file.
**Maintainer**: Caspar Addyman.

## Read first

1. `docs/development/roadmap_v1.6.0.md`: Phase 2 is the plan, and its items are your task list. Also
   read the Release Overview (why v1.6.0 exists: the JOSS scope check needs outside users) and the
   Core Principles.
2. `CLAUDE.md` (commands, code style) and `AGENTS.md` (§23 for the viewer).
3. `.specify/memory/constitution.md` (v1.1.0). It is binding: Local-First, Stable Pipeline Contract,
   Provenance, Modular, Backward Compatibility, and the viewer's Engineering Standards.
4. `CHANGELOG.md`, `[Unreleased]`: everything v1.6.0 has changed so far.

## Where things stand

- **Phase 0** (one repository, the viewer merged in): done.
- **Phase 1** (pipelines and dependencies): done.
  - Python 3.13 support.
  - torch 2.11 and pyannote.audio 4.
  - The LAION pipelines dropped.
  - One per-user folder each for models, logs and the database, used by both database layers.
  - The output baseline re-captured against v1.5.0, checked by
    `tests/integration/test_output_baseline.py` with real models.
  - One `Dockerfile` for CPU and GPU, measured: slim 1.35 GB, every pipeline 14.9 GB, v1.4.3 26.1 GB.
  - The Windows dev-container freeze diagnosed (memory exhaustion) and capped (`--memory=12g`).

  Still in Phase 1: "Download weights ahead of time". It's a feature, not housekeeping; move it to
  Phase 2 or 3 with Caspar.
- **Phase 2**: not started.
- **Decided, not for now**: diarization alternatives and the torch 2.11 cap
  (`dependency_audit_v1.6.0.md` §8). Phase 5 benchmarks Sortformer against pyannote. Caspar is to
  ask LAAC-LSCP about VTC 2's licence.

## Suggested order

The README and docs describe behaviour, so fix the behaviour first.

**A. Correctness (small, testable; one commit each)**
1. **A pipeline that fails must say so.** `speech_recognition` swallows transcription errors and
   reports "completed" with no transcript. Make it raise, then audit the other pipelines for the
   same pattern.
2. **`GET /api/v1/jobs/{id}/results/files/{pipeline}`** returns `OUTPUT_FILE_MISSING` for every
   result stored in the database, and it's what every job advertises as `download_url`. The files
   are in the job's storage folder (`<stem>_<pipeline suffix>.json/.rttm/.vtt`). Serve them from
   there, or stop advertising the URL.
3. **`videoannotator process <video>`** is a stub (`cli.py`, "not yet implemented"). Implement it on
   the shared job path, or remove it. `tests/integration/test_output_baseline.py` shows how to drive
   a whole job through a server.
4. **Queue position** for pending jobs.
5. **OpenFace 3 deterministic mode**, recorded in provenance.

**B. Viewer**
- Zero `tsc` errors, with typechecking in CI. Run `bunx tsc --noEmit -p tsconfig.app.json`; the root
  `tsconfig.json` compiles nothing.
- One function decides which pipeline produced a file (start from `merger.ts`'s
  `detectJSONStructure`).
- The Datasets page: wire it up or hide it.
- Constitution 1.1.0's open standards: a 300 KB gzipped initial bundle (304 KB now), and overlays
  naming the pipeline and version that drew them.
- After viewer changes run `bash scripts/build_viewer.sh` and commit `viewer_static/`; CI fails if
  the bundle doesn't match its source.

**C. "Run it again" and the VLM work**: rerun, reuse settings, prompt library, prompt workbench,
compare two VLM jobs. These are features, so write a spec for each coherent one (`/speckit-specify`,
then `/speckit-plan` and `/speckit-tasks`).

**D. Docs, last**
- README rewritten for researchers; internal material out of `docs/`; a docs site; one entry page and
  a link check in CI.
- **The dev-container docs need a rewrite** (`docs/installation/INSTALLATION.md`, "Dev Container"; and
  troubleshooting). They still lead with a WSL `.wslconfig` cap, which Caspar asked not to lead with.
  The repo now caps the container itself. Use the measured numbers
  (`handover_windows_devcontainer_freeze.md`, Update 6, and `handover_docker_images.md`):
  - Editor baseline about 2 GB.
  - Full test suite peak 5.4 GB.
  - Six-pipeline job peak 6.5 GB.
  - Installing the extras fills the cap with page cache, which is reclaimed.
  - Add a "16 GB machine" paragraph and the out-of-memory symptom (`Killed`, exit 137): raise
    `--memory` in `devcontainer.json`.
  - Say that a rebuild which changes the Python location (as the move to one Dockerfile did)
    recreates `.venv`; post-create reinstalls everything.
- `docs/deployment/Docker.md` is current; put the measured sizes in it.

**E. The structural pass and the Playwright first-time-user run** along install → add videos → run
→ review → export, once A–D have settled.

## Smaller leftovers

- `ruff check .` reports 7 errors in `viewer/scripts/`: helper scripts imported with the viewer. CI
  lints only `src/` and `tests/`. Fix, exclude, or delete them if dead.
- `.devcontainer/devcontainer.json.bak-20260108-122537` and `.devcontainer/noop.txt` are tracked
  strays. Ask before deleting.
- CLAUDE.md's "Current Plan" still names spec 012. The speckit script rewrites it on the next
  `/speckit-plan`.
- `FileStorageBackend.cleanup_old_files(0)` now means "everything". Before, a Windows clock-tick race
  kept just-written files.

## How to work here

- **Commits**: in the last session Caspar asked for each finished, tested piece to be committed and
  pushed to `1.6-dev`. Confirm that still holds when you start. End commit messages with the
  attribution line your session gives you.
- **pre-commit** runs on every commit. `ruff format` may rewrite files, including Python code blocks
  inside Markdown, and abort the commit: re-add the files and commit again.
- **Record as you go**: user-visible changes go in `CHANGELOG.md` `[Unreleased]`. Tick roadmap items
  with a dated note of what was done, and add newly found problems to the roadmap rather than
  leaving them in chat.
- **Tests**:
  - `pytest tests/` takes about 2 minutes and includes the real-model baseline test (about 40 s,
    needs the GPU and model weights).
  - CI runs Ubuntu, macOS and Windows on Python 3.12 and 3.13, with `--all-extras`. It also has a
    `core-install` job (no dev group, no extras; starts the server on a fresh database).
- **Dependencies**: the dev container sets `UV_NO_SYNC=1`. To change the environment, run
  `env -u UV_NO_SYNC uv sync --inexact --all-extras`, and run `uv lock` after editing
  `pyproject.toml`.
- **Memory**: the container is capped at 12 GB. Don't run an extras install, the full suite and a
  server job at the same time.

## Pitfalls that cost time in Phase 1

- **Windows and macOS CI.**
  - Path tests that assume Linux (XDG, `/` roots) fail elsewhere. Compare with `samefile`/`resolve`,
    and point every platform's base (`XDG_*`, `LOCALAPPDATA`, `HOME`, `USERPROFILE`) at `tmp_path`.
  - On GitHub's Windows runners `bash` can be WSL's launcher with no distro installed.
  - Windows can't delete an open SQLite file: close engines (`reset_storage_backend()`) before the
    temp folder goes.
- **Test isolation.**
  - Use `monkeypatch.setenv`, never `os.environ[...] =`; leaked database settings broke later tests.
  - `tests/conftest.py` points the database at a temp file at import time, because modules bind
    `SessionLocal` when imported. Keep it that way.
- **Two database layers.** `api/database.py` (storage backend) and `database/database.py` (SQLAlchemy:
  users, keys, datasets, presets) both resolve through `database_location.py`. `DATABASE_URL`
  overrides `VIDEOANNOTATOR_DB_PATH`.
- **Core dependencies.** Every test environment has the dev group, so a missing core dependency only
  shows in `pip install videoannotator` (`httpx` was one). The `core-install` CI job now catches it.
- **`.gitignore` surprises.** `.dockerignore` was ignored. Check `git check-ignore -v` when a new file
  won't stage.
- **The viewer and the server.** The viewer forces `127.0.0.1`, while server messages say
  `localhost`. Mismatches cause "can't connect" or 401s.

## Useful references

- `handover_windows_devcontainer_freeze.md`: the freeze and the memory measurements.
- `handover_docker_images.md`: the Docker checks and their results.
- `pipeline_review_v1.6.0.md`, `pipeline_landscape_v1.6.0.md`, `dependency_audit_v1.6.0.md`: why each
  pipeline and pin is what it is.
- `tests/fixtures/viewer_contract/README.md`: the real outputs that are both the viewer's contract and
  the pipelines' baseline, and how to regenerate them.
