# Tasks: Core Dependency Clean-up and Tooling

**Input**: `specs/013-core-deps-cleanup/` (spec.md, plan.md, research.md)

## Phase 1: Setup

- [X] T001 Record the before state: core-only install size/count (research R1), and dump every pipeline's output on the demo video with `scripts/compare_pipeline_outputs.py` into the scratchpad

## Phase 2: User Story 1 - Smaller core (P1)

- [X] T002 [US1] Remove `src/videoannotator/main.py` and `src/videoannotator/visualization/`; fix the `src.exporters` docstring examples in `src/videoannotator/exporters/__init__.py` and `native_formats.py`
- [X] T003 [US1] In `pyproject.toml`: remove moviepy, matplotlib, tqdm, openpyxl, pandas, imageio, imageio-ffmpeg, av, alembic, rich, click, scikit-image, cryptography from core; move numba to `audio`; remove imutils (`face`) and supervision (`person`); remove the `annotation` extra
- [X] T004 [US1] In `src/videoannotator/version.py`, drop removed packages from the dependency report and make sure it tolerates absent ones
- [X] T005 [US1] `uv lock`; measure the core-only install again (SC-001)

## Phase 3: User Story 2 - Current core libraries (P1)

- [X] T006 [US2] `uv lock --upgrade-package` each core dependency and dev tool; run the suite; cap any failing major with its reason
- [X] T007 [US2] Re-run the output comparison on the demo video and compare with T001 (SC-003)

## Phase 4: User Story 3 - Hooks match CI (P2)

- [X] T008 [US3] `.pre-commit-config.yaml`: local `mypy` hook running `uv run mypy src/videoannotator`; drop pydocstyle and mirrors-prettier; upgrade the rest
- [X] T009 [US3] Dev tools in one place: remove the `dev` extra (and from `all`); add jupyter/jupyterlab to a `notebooks` dependency group; update `.devcontainer/devcontainer.json`, `docker-compose.yml`, `CONTRIBUTING.md`, `docs/installation/INSTALLATION.md`, `docs/usage/GETTING_STARTED.md`
- [X] T010 [US3] Check SC-004: a deliberate type error in a pipeline module is caught by the hook

## Phase 5: User Story 4 - Current Actions (P3)

- [X] T011 [US4] `.github/workflows/ci-cd.yml`: upgrade action versions; pin the JOSS draft action to a commit

## Phase 6: Polish

- [X] T012 CHANGELOG (removed core dependencies, dev-tools change, smaller install); audit doc §3/§6 marked done
- [ ] T013 ruff, mypy, pre-commit, full suite; commit; push; confirm CI
