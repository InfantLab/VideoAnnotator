# Implementation Plan: Pipeline Extras Discoverability & Self-Service Install

**Branch**: `v1.5.0` | **Date**: 2026-08-26 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/005-pipeline-extras-install/spec.md`

## Summary

Add an admin-only, API-triggered action that installs one named `pyproject.toml` extras group
(`face`, `audio`, `scene`, ...) as a trackable background job, plus a restart-required signal so a
client can tell "installed, needs a server restart" apart from "still unavailable." This is the
write-side counterpart to spec 004's existing read-side `available`/`install_hint` fields — nothing
about the extras/registry schema changes. Technical approach: a small new `ExtrasInstallJob`
SQLAlchemy model (auto-created via the existing `Base.metadata.create_all` path, no new migration
needed) tracked through `pending → running → {completed, failed}`; execution is a background thread
spawned at request time (not a polled queue — install requests are rare and human-triggered, unlike
the many-video-jobs case `BackgroundJobManager` was built for); the subprocess itself is `pip
install videoannotator[<extra>]==<running-version>` for a normal install, or `uv sync --extra
<name>` when running from this repo's own editable/source checkout; and a new `require_admin` auth
dependency that surfaces the `User.is_admin` column (already in the schema, never previously
plumbed through the API auth layer).

## Technical Context

**Language/Version**: Python 3.12 (`requires-python = ">=3.12,<3.13"`, unchanged)
**Primary Dependencies**: FastAPI, SQLAlchemy, Pydantic (all already core deps — no new dependency
added, per Constitution Principle IV)
**Storage**: SQLite/SQLAlchemy, same database the existing `users`/`api_keys`/`jobs` tables live in
(`database/models.py`, `Base.metadata.create_all`) — one new table, `extras_install_jobs`
**Testing**: pytest (`tests/unit`, `tests/api`), matching existing API-layer test conventions
**Target Platform**: Same as the rest of the API server — Linux/macOS/Windows, wherever
`videoannotator server` already runs
**Project Type**: Single project — this is additive surface on the existing FastAPI app, no new
top-level component
**Performance Goals**: Install-trigger endpoint responds in the sub-second range regardless of how
long the underlying install takes (FR-004 — the point of the job/polling shape is to never block the
HTTP request on a multi-minute `pip install`)
**Constraints**: MUST NOT add any dependency to the core (no-extras) install; MUST NOT change spec
004's `PipelineMetadata` schema, extras-group names, or existing `available`/`install_hint` fields
**Scale/Scope**: Single-admin, low-frequency action (installs are infrequent, deliberate operator
actions, not a high-throughput path) — no new concurrency infrastructure needed beyond de-duplicating
concurrent requests for the *same* extras group (FR-010)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Principle I (Local-First Execution)**: PASS. Installing is a local `pip`/`uv` action against
  PyPI or the local checkout; no annotation processing is sent anywhere. The feature doesn't touch
  where/how video is processed at all.
- **Principle II (Stable Pipeline Contract and Open Formats)**: PASS / not applicable. No change to
  `BasePipeline`, the registry YAML schema, or any output format. `ExtrasInstallJob` is a new,
  unrelated entity; it doesn't touch pipeline metadata.
- **Principle III (Provenance and Reproducibility)**: Not applicable — no new annotation type or
  pipeline output is introduced. (Pinning the install to the currently-running `videoannotator`
  version, decided in research.md §1, is itself a small reproducibility-flavored choice, though not
  one the constitution's provenance-metadata requirement covers.)
- **Principle IV (Modular by Construction)**: PASS. This feature adds zero dependencies to the core
  install — it only ever triggers an install of an *existing*, already-declared extras group, on
  explicit admin request. The core stays exactly as slim as spec 004 left it.
- **Principle V (Backward Compatibility by Default)**: PASS. Purely additive: one new table, two new
  endpoints, one new field on an existing endpoint's response (`restart_required`, additive per
  spec.md FR-012). No existing field, endpoint, config file, or CLI invocation changes shape or
  meaning.
- **Engineering Standards**: New public API surface (two endpoints, one extended endpoint, one new
  DB model, one new auth dependency) — MUST ship with unit + API-level tests (auth rejection paths,
  extras-name validation, job state transitions, restart-signal semantics, the crash/orphaned-job
  edge case) and full type annotations, consistent with existing `api/v1/*.py` and
  `database/models.py` conventions. Documentation: README's admin-workflow section and this
  project's API docs (FastAPI auto-docs already cover new endpoints; no separate doc system to
  update).

No violations — Complexity Tracking table intentionally omitted (nothing to justify).

## Project Structure

### Documentation (this feature)

```
specs/005-pipeline-extras-install/
├── plan.md              # This file
├── research.md           # Phase 0 output
├── data-model.md          # Phase 1 output
├── quickstart.md          # Phase 1 output
├── contracts/              # Phase 1 output
│   ├── extras-install-endpoints.md
│   └── restart-required-signal.md
└── tasks.md              # Phase 2 output (/speckit-tasks — not created by this command)
```

### Source Code (repository root)

Single project — extending the existing FastAPI app in place, no new top-level directory:

```
src/videoannotator/
├── api/
│   ├── middleware/
│   │   └── auth.py                # + require_admin dependency
│   ├── dependencies.py            # + is_admin surfaced in the returned user dict
│   ├── v1/
│   │   └── pipelines.py           # + POST .../extras/{extra}/install, GET .../install-jobs/{id};
│   │                               #   extend GET /api/v1/pipelines with restart_required
│   └── extras_install.py          # new: background-thread install runner, in-process
│                                   #   restart-required flag, pip/uv command selection
├── database/
│   └── models.py                  # + ExtrasInstallJob model
└── registry/
    └── pipeline_loader.py         # unchanged — extras_available()/install_hint() already exist
                                    #   and are reused, not modified

tests/
├── unit/
│   └── test_extras_install.py      # command selection (pip vs uv), state machine, dedup logic
└── api/
    └── test_pipeline_extras_endpoints.py  # auth gating, validation, contract shapes, restart signal
```

**Structure Decision**: No new project/package — this is additive surface inside the existing
`videoannotator-core` FastAPI app, following the same `api/v1/*.py` + `database/models.py` layering
every other endpoint family already uses. A new `api/extras_install.py` module (rather than inlining
subprocess/threading logic into `api/v1/pipelines.py`) keeps the HTTP-handler file thin and the
install-execution logic independently unit-testable without spinning up the FastAPI app, mirroring
how `job_processor.py` is kept separate from `api/v1/jobs.py`.

## Complexity Tracking

*No entries — Constitution Check found no violations requiring justification.*
