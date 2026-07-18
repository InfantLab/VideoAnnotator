# VideoAnnotator Development Guidelines

Auto-generated from feature plans by `.specify/scripts/bash/update-agent-context.sh`. Last updated: 2026-07-18

## Active Technologies
- Python 3.12 (`requires-python = ">=3.12,<3.13"`), FastAPI, SQLAlchemy, Pydantic, Typer/Click (core — stays required with no extras installed)
- Per-pipeline extras (torch, ultralytics, pyannote.audio, transformers, deepface, open-clip-torch, openai-whisper, etc.) — being moved from required dependencies to `[project.optional-dependencies]` groups (`face`, `face-laion`, `face-openface3`, `audio`, `audio-laion`, `scene`, `person`, `all`) as of 004-extras-based-install
- SQLite/SQLAlchemy for job/pipeline state; local filesystem model cache (HF/torch cache dirs)

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
tests/
├── unit/ integration/ pipelines/ api/ contract/
specs/<NNN>-<slug>/  # spec-kit feature specs (spec.md, plan.md, tasks.md, ...)
docs/development/roadmap_v1.{5,6}.0.md, roadmap_v1.7_to_v2.0.md  # release roadmap
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
```

## Code Style
Python 3.12, ruff-enforced (line-length 88, see `[tool.ruff]` in `pyproject.toml` for the
per-file-ignore exceptions). Follow standard conventions; no comments explaining *what* code does,
only non-obvious *why*.

## Constitution
`.specify/memory/constitution.md` (v1.0.0) is binding — five core principles (Local-First
Execution, Stable Pipeline Contract, Provenance & Reproducibility, Modular by Construction,
Backward Compatibility by Default). `/speckit-plan`'s Constitution Check gate evaluates every plan
against it.

## Recent Changes
- 004-extras-based-install: extras-based modular install + metadata-driven registry loading
  (removes `LEGACY_MAPPINGS`, adds `requires_extras` to `PipelineMetadata`), scoped to leave room
  for v1.6.0's Ollama backend and v1.7+'s remote/HPC dispatch without another schema migration.

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
