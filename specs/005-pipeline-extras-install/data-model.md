# Phase 1 Data Model: Pipeline Extras Discoverability & Self-Service Install

## ExtrasInstallJob

New persisted entity, sibling to `User`/`APIKey`/`Job` in `database/models.py`. Tracks one attempt
to install a single named extras group.

| Field | Type | Notes |
|---|---|---|
| `id` | UUID (primary key) | Same `GUID()` type as existing models, for SQLite/PostgreSQL portability. |
| `extra_name` | string, indexed | The target extras group (e.g. `face`, `audio`, `all`). Validated against `pyproject.toml`'s declared groups *before* a row is ever created (FR-002) — this column never holds an unvalidated value. |
| `requested_by_user_id` | UUID, FK → `users.id` | Which admin triggered it; for audit, not access control (access control already happened before job creation). |
| `status` | enum: `pending`, `running`, `completed`, `failed` | Mirrors `JobStatus` naming already used for annotation jobs (`batch/types.py`) for consistency, deliberately a smaller set — no `retrying`/`cancelled`, since installs aren't retried automatically and cancellation is out of scope. |
| `command_output` | text, nullable | Captured stdout/stderr from the install subprocess. Populated incrementally or on completion; MUST be present on `failed` (FR-006). |
| `created_at` | datetime | Set on creation. |
| `started_at` | datetime, nullable | Set when status transitions to `running`. |
| `finished_at` | datetime, nullable | Set when status reaches `completed` or `failed`. |

### State transitions

```
pending → running → completed
                   → failed
```

A row that is still `pending` or `running` when the server process starts up (i.e. it was orphaned
by a crash or unclean shutdown of a previous process) is transitioned to `failed` with
`command_output` noting the interruption, during application startup — satisfying the spec's crash
edge case ("MUST NOT be reported as silently running forever").

### Validation rules

- `extra_name` MUST be one of the extras groups the running `pyproject.toml`/installed package
  metadata currently declares (same source `registry/pipeline_loader.py` already reads for
  `extras_available()`/`install_hint()`) — checked at request time, not just at row-creation time,
  since the declared set could differ across code versions.
- Only one `ExtrasInstallJob` may be `pending` or `running` for a given `extra_name` at a time
  (FR-010); a duplicate request while one is in flight returns the existing job's identifier rather
  than creating a second row.
- A request naming an `extra_name` with no currently-missing dependencies (already fully installed)
  MUST still produce a valid job record that resolves to `completed` quickly with output noting
  nothing needed to change (FR-011) — not a rejected request and not a different response shape than
  a "real" install.

## Restart-required signal (not persisted)

Deliberately **not** a database column — see research.md §4. It is in-process runtime state: a
boolean, true from the moment any `ExtrasInstallJob` reaches `completed` in the current process's
lifetime, until that process exits. Exposed through the API (see contracts/), never written to or
read from the database.

## Relationship to existing entities

- `User`: `ExtrasInstallJob.requested_by_user_id` is a straightforward FK, following the same
  pattern as `Job.user_id`. No changes to the `User` model's columns — `is_admin` already exists;
  this feature is the first consumer to check it (see research.md §3).
- `PipelineMetadata` / the registry (spec 004): unchanged. `ExtrasInstallJob` does not reference a
  specific pipeline, only an extras-group name — the same decoupling spec 004 already established
  between pipelines and extras groups (one group can back multiple pipelines, e.g. `face` backs
  `face_analysis` and others).
