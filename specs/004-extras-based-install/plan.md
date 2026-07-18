# Implementation Plan: Extras-Based Modular Install & Registry Refactor

**Branch**: `004-extras-based-install` | **Date**: 2026-07-18 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/004-extras-based-install/spec.md`

## Summary

Move VideoAnnotator's heavy ML dependencies out of `[project.dependencies]` and into
per-pipeline-family `[project.optional-dependencies]` extras groups (`face`, `audio`, `scene`,
`person`, plus separate LAION groups), replace the registry's hardcoded `LEGACY_MAPPINGS` dict
with a `requires_extras` field read straight from each pipeline's YAML metadata, make the registry
degrade gracefully (omit, don't crash) when a pipeline's extras aren't installed, and drop the
`numpy<2.0` pin once the suite is green on NumPy 2.x. The extras-group naming and
`PipelineMetadata` schema changes are chosen so v1.6.0 (Ollama `llm` extra) and v1.7.0+
(HTTPDispatcher/SlurmDispatcher remote/HPC targets) can build on them without another schema
migration (spec FR-010–FR-012, SC-007).

## Technical Context

**Language/Version**: Python 3.12 (`requires-python = ">=3.12,<3.13"`, `pyproject.toml:26`)
**Primary Dependencies**: FastAPI, SQLAlchemy, Pydantic, Typer/Click (core, stays required);
torch, torchvision, torchaudio, ultralytics, pyannote.audio (+core/database/metrics/pipeline),
transformers, sentence-transformers, open-clip-torch, openai-whisper, timm, deepface (via
`openface-test`/opencv, no torch), huggingface-hub (moving to extras)
**Storage**: SQLite/SQLAlchemy for job/pipeline state (unaffected); local filesystem model cache
under the user's HF/torch cache dirs (must survive this refactor unchanged — FR-015 equivalent)
**Testing**: pytest (`tests/unit`, `tests/integration`, `tests/pipelines`, `tests/api`,
`tests/contract`), pytest-cov (≥80% gate per constitution Engineering Standards), mypy, ruff
**Target Platform**: Linux/macOS/Windows, CI matrix already runs all three (`.github/workflows/ci-cd.yml`)
**Project Type**: Single Python package (`src/videoannotator/`), no separate frontend/backend split
**Performance Goals**: Single-family install + first run < 5 min on broadband laptop (SC-001);
default Docker image ≥80% smaller than v1.4.4 baseline (SC-002)
**Constraints**: v1.5.0 `[all]` output byte-identical to v1.4.4 for deterministic pipelines,
documented tolerance for non-deterministic ones (FR-009); NumPy 2.x must pass full suite (FR-007);
zero config/CLI changes for existing users (User Story 2)
**Scale/Scope**: 9 existing pipeline metadata YAML files across 4 families (`face` ×3 variants,
`audio` ×4 variants, `scene` ×1, `person` ×1); none currently declare `module_path` or
`requires_extras` — all 9 resolve today purely through `pipeline_loader.py`'s `LEGACY_MAPPINGS`
fallback, so every file needs both fields added; `pyproject.toml` dependency block (~95 lines);
`registry/pipeline_loader.py` (LEGACY_MAPPINGS removal); 2 Dockerfiles (`Dockerfile.cpu`,
`Dockerfile.gpu`) plus `Dockerfile.dev`

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Evaluated against `.specify/memory/constitution.md` v1.0.0's Plan Gating checklist:

- **Principle I (Local-First Execution)**: PASS. This phase touches install/registry only; no
  pipeline gains a required network dependency. Extras that are HTTP-thin (future `llm` group) are
  explicitly out of scope for this phase (spec Assumptions).
- **Principles II & V (Stable Pipeline Contract, Backward Compatibility)**: PASS, with the
  refactor's entire purpose being to preserve these — `requires_extras` is an additive
  `PipelineMetadata` field, `BasePipeline`'s interface is untouched, no output keys change. User
  Story 2 (byte-identical `[all]` output vs v1.4.4) is the concrete test.
- **Principle III (Provenance & Reproducibility)**: PASS / not applicable — this phase adds no new
  annotation type, so no new provenance fields are required. Noted for later: v1.7.0's
  remote/HPC/VLM work (out of scope here) will need to extend provenance metadata per that
  principle; flagged already in `roadmap_v1.7_to_v2.0.md`.
- **Principle IV (Modular by Construction)**: PASS — this phase is a direct, literal
  implementation of this principle (it is currently violated: every heavy ML dep is unconditional
  today). This plan is what brings the codebase into compliance.
- **Engineering Standards**: Testing, CI (3-OS matrix), type safety, and docs updates are planned
  below (Phase 1 quickstart + install-matrix docs, CI job additions for per-extras install
  smoke-tests). No proprietary weights involved.

**Result**: No violations. No entries needed in Complexity Tracking.

## Project Structure

### Documentation (this feature)

```
specs/004-extras-based-install/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md         # Phase 1 output
├── quickstart.md         # Phase 1 output
├── contracts/             # Phase 1 output
│   ├── pipeline-metadata-schema.md
│   └── unavailable-pipeline-error.md
└── tasks.md              # Phase 2 output (/speckit-tasks, not this command)
```

### Source Code (repository root)

```
pyproject.toml                                   # [project.dependencies] slimmed;
                                                   # [project.optional-dependencies] gains
                                                   # face/audio/scene/person/face-laion/
                                                   # audio-laion/all groups
src/videoannotator/registry/
├── pipeline_registry.py                          # PipelineMetadata gains requires_extras: list[str]
├── pipeline_loader.py                            # LEGACY_MAPPINGS removed; module_path resolved
│                                                  # from metadata only; extras-aware availability
│                                                  # check; actionable ImportError → install-hint
│                                                  # translation
└── metadata/*.yaml                                # each file gains requires_extras: [...]
src/videoannotator/cli.py                         # pipeline-unavailable error path -> install hint
src/videoannotator/api/                           # GET /api/v1/pipelines excludes unavailable
                                                   # pipelines from listings; job submission returns
                                                   # actionable 4xx, not 500, for unavailable pipeline
Dockerfile.cpu, Dockerfile.gpu                    # slim by default (no extras); [all] variant documented
docs/installation/INSTALLATION.md                 # per-use-case install matrix (SC-006)
tests/unit/registry/                              # new: requires_extras resolution, graceful
                                                   # degradation, migration-message tests
tests/integration/                                # extras-isolation smoke tests (subprocess-based,
                                                   # one clean venv per extras group, CI-gated)
```

**Structure Decision**: Single-project layout (existing `src/videoannotator/` package); no new
top-level projects or services introduced. This phase is deliberately confined to the registry,
`pyproject.toml`, and Docker/docs — it does not touch `api/job_processor.py` or
`batch/batch_orchestrator.py` beyond what's needed for the actionable-error path (their
consolidation is FR-012's forward-compat guarantee for v1.6.0, not this phase's job).

## Complexity Tracking

*No Constitution Check violations — this section is empty by design.*
