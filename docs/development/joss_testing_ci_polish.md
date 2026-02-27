# JOSS Submission — Testing & CI Polish Task List

Tracked tasks for improving test coverage and CI quality before JOSS submission.
Created 2026-02-27 based on coverage analysis of v1.4.1.

**Baseline:** 774 tests, 750 passed, 24 skipped, 47.75% line coverage.

---

## 1. High-ROI Test Additions

### 1.1 Pure utility functions (0% coverage, easiest wins)

- [x] **`utils/math_utils.py`** — 45 lines, 0% -> covered. Tests in `tests/unit/utils/test_math_utils.py`.
- [x] **`utils/transform_utils.py`** — 17 lines, 0% -> covered. Tests in `tests/unit/utils/test_transform_utils.py`.

### 1.2 Validation and export logic (0–16% coverage, high JOSS value)

- [x] **`validation/coco_validator.py`** — 218 lines, 0% -> covered. Tests in `tests/unit/validation/test_coco_validator.py`.
- [x] **`exporters/native_formats.py`** — 215 lines, 16% -> covered. Tests in `tests/unit/exporters/test_native_formats.py`. Also fixed 3 bugs (WebVTT append, praatio import, TextGrid save).

### 1.3 Auth and data layer gaps (44–56% coverage)

- [ ] **`auth/token_manager.py`** — 141 lines, 56% coverage. JWT create/verify/refresh logic. Estimated +40–60 lines covered.
- [ ] **`database/crud.py`** — 166 lines, 44% coverage. CRUD operations, testable with in-memory SQLite. Estimated +50–80 lines covered.

### 1.4 Other notable gaps

- [ ] **`utils/automatic_labeling.py`** — 156 lines, 6% coverage.
- [ ] **`utils/person_identity.py`** — 163 lines, 18% coverage.
- [ ] **`pipelines/face_analysis/openface_compatibility.py`** — 69 lines, 0% coverage.
- [ ] **`schemas/industry_standards.py`** + **`schemas/storage.py`** — 21 lines total, 0% coverage.

### 1.5 Lower priority (large modules, harder to unit-test)

- [ ] **`cli.py`** — 637 lines, 8% coverage. CLI commands are integration-heavy; consider a few smoke tests via `typer.testing.CliRunner`.
- [ ] **`main.py`** — 84 lines, 0% coverage. App factory / startup glue.

**Target:** reach ≥55% line coverage (from 47.75%) by completing sections 1.1–1.2.

---

## 2. CI/CD Fixes

### 2.1 Linting alignment

- [x] **Match ruff rules in CI to pyproject.toml** — Removed `--select` override. CI now uses project config. All 82 pre-existing lint errors fixed.
- [x] **Add `ruff format --check`** to CI for consistent formatting.

### 2.2 Un-gate quality checks

- [x] **Run mypy on all PRs** — Now runs on ubuntu-latest for every push/PR.
- [x] **Run integration tests on PRs** — Now runs on push and workflow_dispatch.

### 2.3 Coverage enforcement

- [x] **Add `--cov-fail-under=45`** to the pytest step to prevent coverage regressions. Raise threshold as coverage improves.

### 2.4 Bump outdated GitHub Actions — DONE

All actions bumped in ci-cd.yml:

| Action | Old | New |
|---|---|---|
| `actions/setup-python` | `@v4` | `@v5` |
| `codecov/codecov-action` | `@v3` | `@v4` |
| `aquasecurity/trivy-action` | `@master` | `@0.28.0` |
| `github/codeql-action/upload-sarif` | `@v2` | `@v3` |
| `docker/setup-buildx-action` | `@v2` | `@v3` |
| `docker/login-action` | `@v2` | `@v3` |
| `docker/metadata-action` | `@v4` | `@v5` |
| `docker/build-push-action` | `@v4` | `@v6` |

### 2.5 Dependency hygiene

- [ ] **Move `pytest` and `pytest-asyncio` from `[project.dependencies]` to dev-only** — they are runtime test deps shipped to end users today.

---

## 3. Metadata Fixes

- [x] **Fix `project.urls`** in pyproject.toml — updated to `InfantLab/VideoAnnotator`.
- [ ] **Add Codecov badge** to README.md.

---

## 4. JOSS Checklist Remaining Items

- [ ] Push tag to remote: `git push origin v1.4.1`
- [ ] Create GitHub Release for v1.4.1
- [ ] Improve git contributor attribution (only 1 contributor visible in history)

---

## Progress Log

| Date | Change | Coverage Impact |
|---|---|---|
| 2026-02-27 | Baseline measured | 47.75% line, ~25% branch |
| 2026-02-27 | Added tests for math_utils, transform_utils, coco_validator, native_formats (4 new test files, 86 new tests) | 47.75% -> 51.97% (+4.22%) |
| 2026-02-27 | Fixed 3 bugs in native_formats.py (WebVTT append, praatio import, TextGrid save) | — |
| 2026-02-27 | Fixed all 82 ruff lint errors (31 autofix + 51 manual) | — |
| 2026-02-27 | CI: broadened ruff rules, un-gated mypy, un-gated integration tests, added coverage threshold, bumped all action versions | — |
| 2026-02-27 | Fixed project.urls to point to InfantLab/VideoAnnotator | — |
