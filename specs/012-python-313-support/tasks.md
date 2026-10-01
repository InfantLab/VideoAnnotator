---
description: "Tasks for 012 Python 3.13 support"
---

# Tasks: Python 3.13 Support

**Input**: `specs/012-python-313-support/` (plan.md, spec.md, research.md, data-model.md,
contracts/, quickstart.md)

**Tests**: one new unit test file for the new runtime surface (constitution: new public behaviour
ships with tests); otherwise the existing suite, run on both versions, is the test.

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [X] T001 Set `requires-python = ">=3.12,<3.14"` and add the `Programming Language :: Python :: 3.13` classifier in `pyproject.toml`; leave ruff `target-version = "py312"` and mypy `python_version = "3.12"` unchanged (research R5)
- [X] T002 Regenerate `uv.lock` with `uv lock` and confirm the only additions are `audioop-lts`, `standard-aifc`, `standard-chunk`, `standard-sunau`, `pyyaml-ft`, each reachable only under a `python_full_version >= '3.13'` marker (research R1, FR-002)
- [X] T003 Add `.python-version` containing `3.13` at the repository root (research R3)

## Phase 2: Foundational

- [X] T004 Add `SUPPORTED_PYTHON = ((3, 12), (3, 13))` and `warn_if_unsupported_python()` to `src/videoannotator/version.py`: logs the warning text from `contracts/supported-python.md` once per process when `sys.version_info[:2]` isn't in the range; never raises
- [X] T005 [P] Add `tests/unit/test_supported_python.py`: `SUPPORTED_PYTHON` matches `requires-python` and the classifiers in `pyproject.toml`; the warning fires once for an unsupported version (monkeypatched `sys.version_info`) and never for 3.12/3.13

## Phase 3: User Story 1 - Install and run on Python 3.13 (P1) 🎯 MVP

**Goal**: install with every extra and run every pipeline on 3.13.
**Independent Test**: quickstart §1 and §2 on 3.13.

- [X] T006 [US1] Call `warn_if_unsupported_python()` at CLI start in `src/videoannotator/cli.py` (the Typer app callback, before any pipeline import)
- [X] T007 [US1] Call `warn_if_unsupported_python()` at server start in `src/videoannotator/api/main.py` (lifespan startup, before background processing)
- [X] T008 [P] [US1] Add `scripts/compare_pipeline_outputs.py` with `dump <video> --pipelines a,b -o out.json` (runs the job through `api.job_processor.JobProcessor` with a throwaway SQLite storage, loads `.env`, writes each pipeline's status and annotations) and `compare a.json b.json` (reports per-pipeline differences, ignoring run ids and timestamps; exit 1 on differences) (research R7)
- [X] T009 [US1] `uv sync --all-extras` on 3.13, run the full suite (`pytest -m "not real_models"`) and every pipeline on the demo video with `scripts/compare_pipeline_outputs.py dump` (quickstart §1–§2); record results in `specs/012-python-313-support/quickstart.md` under a "Results" heading

## Phase 4: User Story 2 - Existing 3.12 installs keep working (P1)

**Goal**: 3.12 unchanged in install and outputs.
**Independent Test**: dump on 3.12 before and after this change; compare.

- [X] T010 [US2] On 3.12 (`UV_PROJECT_ENVIRONMENT=.venv-312`), dump every pipeline's output on the demo video with the pre-change lock (`git stash` or the previous commit) and with this change, then `compare`; also dump 3.12 vs 3.13 (SC-003). Record results in `specs/012-python-313-support/quickstart.md` "Results"

## Phase 5: User Story 3 - Both versions checked on every change (P2)

**Goal**: CI runs the suite on 3.12 and 3.13.
**Independent Test**: the PR's checks list `test` for each OS × version.

- [X] T011 [US3] In `.github/workflows/ci-cd.yml`: `python-version: ["3.12", "3.13"]` in the `test` matrix; add `UV_PYTHON: ${{ matrix.python-version }}` to the job's `env`; run ruff, the format check and mypy only on `ubuntu-latest` + `3.13` (contracts/ci-matrix.md)
- [ ] T012 [US3] Push and confirm on PR #8 that all six `test` jobs run, ubuntu/macOS pass on both versions, and Windows keeps `continue-on-error`

## Phase 6: User Story 4 - Dev container and images on 3.13 (P3)

**Goal**: the dev container and Docker images run Python 3.13.
**Independent Test**: quickstart §4.

- [X] T013 [P] [US4] In `Dockerfile.cpu`, `Dockerfile.gpu` and `Dockerfile.dev`, `COPY .python-version` with `pyproject.toml`/`uv.lock` and run `uv python install` before `uv sync`, so 3.13 is baked into the image (research R6)
- [X] T014 [US4] Check whether `.devcontainer/devcontainer.json` needs any change (its post-create `uv sync` follows `.python-version`; `.venv` gets recreated for 3.13); document in `quickstart.md` that the first rebuild recreates `.venv`
- [X] T015 [US4] Build `Dockerfile.cpu` if Docker is available and check `python --version` inside; otherwise record that the image build is left to CI's `docker-build` job / the maintainer *(no Docker in the dev container: left to the maintainer / CI; hadolint shows no new warnings)*

## Phase 7: Polish & cross-cutting

- [X] T016 [P] Update "Python 3.12+" to "Python 3.12 or 3.13" in `README.md` (badge and text), `docs/installation/INSTALLATION.md` and `docs/usage/GETTING_STARTED.md` (FR-009)
- [X] T017 [P] Update `CLAUDE.md` Active Technologies line to `requires-python = ">=3.12,<3.14"` and the Recent Changes entry from "planned" to done
- [X] T018 [P] Amend `.specify/memory/constitution.md` Engineering Standards CI line to "3.12 and 3.13" with a PATCH version bump and a sync-impact note (research R8)
- [X] T019 [P] Add a CHANGELOG entry under `[Unreleased]` in `CHANGELOG.md`: Python 3.13 supported; 3.12 still supported, planned to end in v1.7.0; the startup warning (FR-011)
- [ ] T020 Run `ruff check`, `ruff format --check`, `mypy src/videoannotator` and `pre-commit run` on changed files; tick the spec's tasks; update `specs/012-python-313-support/spec.md` status

## Dependencies

- T001 → T002 → everything else. T003 before T013/T014.
- T004 → T005, T006, T007.
- US1 (T006–T009) and US3 (T011) are independent of each other; US2 (T010) needs T008.
- Polish (T016–T019) can run any time after T001.

## Parallel examples

- After T004: T005, T006/T007, T008 in parallel (different files).
- Polish: T016, T017, T018, T019 in parallel.

## Implementation strategy

MVP is US1 + US2 (both P1): 3.13 works and 3.12 is unchanged. US3 (CI) lands in the same push so
the PR proves both. US4 (images) can follow in the same branch.
