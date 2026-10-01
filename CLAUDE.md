# VideoAnnotator Development Guidelines

Auto-generated from feature plans by `.specify/scripts/bash/update-agent-context.sh`. Last updated: 2026-08-26

## Active Technologies
- Python 3.12 and 3.13 (`requires-python = ">=3.12,<3.14"`; `.python-version` = 3.13, the dev/Docker default), FastAPI, SQLAlchemy, Pydantic, Typer/Click (core — stays required with no extras installed)
- Per-pipeline extras (torch, ultralytics, pyannote.audio, transformers, deepface, open-clip-torch, openai-whisper, etc.) — being moved from required dependencies to `[project.optional-dependencies]` groups (`face`, `face-laion`, `face-openface3`, `audio`, `audio-laion`, `scene`, `person`, `all`) as of 004-extras-based-install
- SQLite/SQLAlchemy for job/pipeline state; local filesystem model cache (HF/torch cache dirs)
- New `extras_install_jobs` table (005-pipeline-extras-install) tracks admin-triggered, in-app
  installs of a named extras group (`pending`/`running`/`completed`/`failed`); no new dependency —
  reuses FastAPI/SQLAlchemy already listed above

## Project Structure
```
src/videoannotator/
├── api/            # FastAPI app, job submission/status endpoints
├── batch/          # batch_orchestrator — CLI batch job execution path
├── cli.py          # videoannotator CLI entry point
├── pipelines/      # face_analysis/, audio_processing/, scene_detection/, person_tracking/
├── registry/       # pipeline_registry.py, pipeline_loader.py, metadata/*.yaml
├── storage/        # job/annotation storage backends
└── exporters/      # COCO/RTTM/WebVTT/native-format writers
viewer/             # Video Annotation Viewer (React/Vite, Bun); built into src/videoannotator/viewer_static/
tests/
├── unit/ integration/ pipelines/ api/ contract/
specs/<NNN>-<slug>/  # spec-kit feature specs (spec.md, plan.md, tasks.md, ...)
docs/development/roadmap_v1.{5,6,7}.0.md, roadmap_v1.7_to_v2.0.md  # release roadmap (v1.6.0 = public release)
```

## Commands
```bash
pytest tests/                 # full suite
pytest tests/ -k acceptance   # v1.4.x behaviour-parity fixtures
ruff check .                  # lint (see pyproject.toml [tool.ruff])
mypy src/videoannotator        # type check
pre-commit run --all-files    # full pre-commit gate (used on every commit)
videoannotator pipelines list # CLI: list available pipelines
videoannotator job submit <video> --pipelines <name>
bash scripts/build_viewer.sh  # rebuild viewer/ into viewer_static/ (commit the result; CI checks it)
cd viewer && bun run lint && bun run test:run  # viewer lint + unit tests
```

## Code Style
Python 3.12+ syntax (ruff/mypy target 3.12, the oldest supported), ruff-enforced (line-length 88, see `[tool.ruff]` in `pyproject.toml` for the
per-file-ignore exceptions). Follow standard conventions; no comments explaining *what* code does,
only non-obvious *why*.

## Constitution
`.specify/memory/constitution.md` (v1.0.0) is binding — five core principles (Local-First
Execution, Stable Pipeline Contract, Provenance & Reproducibility, Modular by Construction,
Backward Compatibility by Default). `/speckit-plan`'s Constitution Check gate evaluates every plan
against it.

## Current Plan
<!-- SPECKIT START -->
`specs/012-python-313-support/plan.md`: support Python 3.13 alongside 3.12 (no library upgrades).
<!-- SPECKIT END -->

## Recent Changes
- 012-python-313-support: `requires-python` widens to `>=3.12,<3.14`; CI tests both;
  `.python-version` makes 3.13 the dev/Docker default; ruff/mypy targets stay at 3.12 (oldest
  supported).
- Post-005 follow-up: added `GET /api/v1/auth/me` (any authenticated caller can check its own
  `is_admin` — closes the gap where a 403 from an admin-only endpoint was undiagnosable from the
  frontend) and made `generate-token` grant admin by default to a brand-new user while the
  deployment is still single-user (0-1 existing users), with an explicit `--admin`/`--no-admin`
  override. Root cause: `generate-token` had no admin concept at all, so re-issuing a viewer key
  under a slightly different email silently produced a non-admin identity with no way to tell.
- 005-pipeline-extras-install: self-service, admin-only install of an extras group triggered via the
  API (`POST /api/v1/pipelines/extras/{extra}/install`), tracked as a background job, plus a
  `restart_required` signal — the write-side counterpart to 004's read-only `available`/
  `install_hint` fields. Backend-only; viewer UI is a separate spec in `video-annotation-viewer`.
- 004-extras-based-install: extras-based modular install + metadata-driven registry loading
  (removes `LEGACY_MAPPINGS`, adds `requires_extras` to `PipelineMetadata`), scoped to leave room
  for the Ollama backend (shipped in v1.5.0) and v1.8+'s remote/HPC dispatch without another schema migration.

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
