# Implementation Plan: Core Dependency Clean-up and Tooling

**Branch**: `1.6-dev` | **Date**: 2026-10-01 | **Spec**: [spec.md](spec.md)

## Summary

Shrink the core install (73 packages / 746 MB today) by removing twelve unused core
dependencies, two dead modules (`visualization/`, `main.py`) and moving numba to `audio`; drop
unused extras dependencies and the empty `annotation` extra; upgrade core and dev tools within
constraints; make the pre-commit type check identical to CI's; drop dead hooks; one place for dev
tools; current GitHub Actions. Evidence: [research.md](research.md).

## Technical Context

**Language/Version**: Python 3.12 and 3.13 (spec 012)
**Primary Dependencies**: core after this spec: fastapi, uvicorn[standard], python-multipart,
pydantic, sqlalchemy, typer, numpy, Pillow, pyyaml, requests, packaging, psutil, python-dotenv,
pycocotools, webvtt-py, praatio, pyjwt
**Testing**: pytest on 3.12 + 3.13 (CI), `scripts/compare_pipeline_outputs.py` on the demo video
**Constraints**: no pipeline output change (FR-006); torch/pyannote/transformers/opencv untouched
**Scale/Scope**: pyproject, lock, two dead modules, version.py report, pre-commit config, CI
workflow, devcontainer/compose/docs commands

## Constitution Check

| Principle / standard | Assessment |
|---|---|
| IV. Modular by construction ("slim core") | Directly served: core shrinks. **Pass** |
| II/III. Contract, reproducibility | Output comparison before/after (FR-006). **Pass** |
| V. Backward compatibility | Removed modules weren't importable; removed transitive packages listed in CHANGELOG. **Pass** |
| Engineering: CI, mypy, docs | mypy hook = CI; docs updated for `--extra dev` → dependency group. **Pass** |

## Project Structure

```
pyproject.toml, uv.lock
src/videoannotator/main.py                 # removed
src/videoannotator/visualization/          # removed
src/videoannotator/version.py              # dependency report: drop removed packages
src/videoannotator/exporters/*.py          # docstring examples: videoannotator.exporters
.pre-commit-config.yaml
.github/workflows/ci-cd.yml
.devcontainer/devcontainer.json, docker-compose.yml
CONTRIBUTING.md, docs/installation/INSTALLATION.md, docs/usage/GETTING_STARTED.md
CHANGELOG.md
```

## Complexity Tracking

None.
