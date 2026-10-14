# Implementation Plan: Python 3.13 Support

**Branch**: `1.6-dev` | **Date**: 2026-10-01 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/012-python-313-support/spec.md`

## Summary

Support Python 3.13 alongside 3.12 with no library change: widen `requires-python` to
`>=3.12,<3.14`, regenerate the lock (adds only 3.13-only backports), test both versions in CI on
every runner, and make 3.13 the default for the dev container and Docker images through a
`.python-version` file that uv reads. A startup warning covers the one install path that skips the
version check (`uv pip install` from a checkout). A comparison script verifies pipeline outputs
are unchanged. Docs, CHANGELOG and the constitution's CI line are updated. Evidence for every
decision: [research.md](research.md), from the trial in the dependency audit §7.

## Technical Context

**Language/Version**: Python 3.12 and 3.13 (adds 3.13; 3.12 unchanged)
**Primary Dependencies**: unchanged (FastAPI, SQLAlchemy, Pydantic, Typer; pipeline extras as
locked). New in the lock, 3.13 only: `audioop-lts`, `standard-aifc`, `standard-chunk`,
`standard-sunau`, `pyyaml-ft`
**Storage**: unchanged (SQLite)
**Testing**: pytest (existing suite, run on 3.12 and 3.13), plus a new pipeline-output
comparison script for manual verification
**Target Platform**: Linux (incl. CUDA), macOS, Windows; Docker images on Ubuntu 24.04 with uv's
managed CPython 3.13
**Project Type**: single Python package (plus the bundled viewer, unaffected)
**Performance Goals**: none new; CI wall time stays within one runner's duration (jobs run in
parallel)
**Constraints**: no library upgrades (FR-002); 3.12 behaviour and outputs unchanged (FR-004)
**Scale/Scope**: configuration, CI, three Dockerfiles, one small runtime check, one script, docs

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle / standard | Assessment |
|---|---|
| I. Local-first execution | Unaffected: same pipelines, same local defaults. **Pass** |
| II. Stable pipeline contract | No schema or output change; outputs verified identical (SC-003). **Pass** |
| III. Provenance & reproducibility | Provenance records already include the Python version where recorded; outputs identical across versions in the trial. **Pass** |
| IV. Modular by construction | Core stays slim; backports are 3.13-only markers inside existing extras' dependency trees. **Pass** |
| V. Backward compatibility | 3.12 remains supported; nothing renamed or removed. **Pass** |
| Engineering: CI on supported versions | Required by this spec: the constitution's own text anticipates it ("3.13 added when upstream deps allow"); amended to "3.12 and 3.13" (PATCH). **Pass** |
| Engineering: ruff, mypy, Trivy pass | Unchanged targets (oldest supported version); mypy runs once. **Pass** |
| Engineering: docs in sync | README, install docs, CLAUDE.md updated (FR-009). **Pass** |
| Engineering: coverage ≥ 80% | Pre-existing gap (CI enforces 45%); not changed by this spec. Noted, not a violation introduced here |

Re-checked after Phase 1 design: no change.

## Project Structure

### Documentation (this feature)

```
specs/012-python-313-support/
├── spec.md
├── plan.md              # this file
├── research.md          # Phase 0
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1: manual verification
├── contracts/
│   ├── supported-python.md   # install-time and runtime contract
│   └── ci-matrix.md          # what CI runs on every change
├── checklists/requirements.md
└── tasks.md             # /speckit-tasks (not created here)
```

### Source Code (repository root)

```
pyproject.toml                    # requires-python, classifiers (ruff/mypy targets stay 3.12)
uv.lock                           # regenerated: 3.13-only backports
.python-version                   # new: 3.13 (uv's default interpreter)
.github/workflows/ci-cd.yml       # test matrix: 3.12 + 3.13, UV_PYTHON per job
Dockerfile.cpu / .gpu / .dev      # `uv python install` before `uv sync`
.devcontainer/devcontainer.json   # unchanged unless the rebuild shows a need
src/videoannotator/version.py     # SUPPORTED_PYTHON range + warn_if_unsupported_python()
src/videoannotator/cli.py         # call the warning at CLI start
src/videoannotator/api/main.py    # call the warning at server start
scripts/compare_pipeline_outputs.py   # new: dump and compare pipeline outputs (SC-003)
tests/unit/test_supported_python.py   # new: range matches pyproject; warning behaviour
README.md, docs/installation/INSTALLATION.md, docs/usage/GETTING_STARTED.md, CLAUDE.md
CHANGELOG.md, .specify/memory/constitution.md
```

**Structure Decision**: single-package layout as today; one new module-level constant and function
in the existing `version.py`, one new script, one new unit test file.

## Complexity Tracking

No constitution violations to justify.
