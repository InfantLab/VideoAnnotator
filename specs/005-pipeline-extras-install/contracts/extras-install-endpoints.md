# Contract: Extras Install Endpoints

This is the interface the `video-annotation-viewer` project's own spec should build against
(spec.md's "API Contract for Downstream Consumers"). Shapes are illustrative — field names/types are
the stable part; exact HTTP status codes follow this project's existing conventions
(422 for validation errors, 401/403 for auth, matching spec 004's `unavailable-pipeline-error.md`).

## Trigger install: `POST /api/v1/pipelines/extras/{extra}/install`

Admin-only (see [restart-required-signal.md](restart-required-signal.md) for the auth contract
shared across this feature's endpoints).

**Path parameter**: `extra` — one of the extras-group names currently declared by the running
install (same set `GET /api/v1/pipelines?include_unavailable=true` already surfaces via
`install_hint`). An unrecognized value is rejected **before** any job is created:

```json
// 422, unrecognized extra
{
  "detail": "Unknown extras group 'gpu-magic'.",
  "known_extras": ["face", "face-laion", "face-openface3", "audio", "audio-laion", "scene", "person", "llm", "all"]
}
```

**Success response** — returns immediately, does not wait for the install to finish (FR-004):

```json
// 202 Accepted
{
  "job_id": "6f1e2c3a-...",
  "extra_name": "face",
  "status": "pending"
}
```

If a job for the same `extra_name` is already `pending`/`running`, this endpoint returns that
existing job (same shape, same status) rather than creating a duplicate (FR-010) — the response is
indistinguishable in shape from a freshly created job, only `job_id` reveals it's the same one.

## Check install job status: `GET /api/v1/pipelines/extras/install-jobs/{job_id}`

```json
// 200, in progress
{
  "job_id": "6f1e2c3a-...",
  "extra_name": "face",
  "status": "running",
  "created_at": "2026-08-26T10:00:00Z",
  "started_at": "2026-08-26T10:00:01Z",
  "finished_at": null,
  "command_output": null,
  "restart_required": false
}
```

```json
// 200, completed successfully
{
  "job_id": "6f1e2c3a-...",
  "extra_name": "face",
  "status": "completed",
  "created_at": "2026-08-26T10:00:00Z",
  "started_at": "2026-08-26T10:00:01Z",
  "finished_at": "2026-08-26T10:04:22Z",
  "command_output": "Successfully installed deepface-0.0.91 ...",
  "restart_required": true
}
```

```json
// 200, failed
{
  "job_id": "6f1e2c3a-...",
  "extra_name": "face",
  "status": "failed",
  "created_at": "2026-08-26T10:00:00Z",
  "started_at": "2026-08-26T10:00:01Z",
  "finished_at": "2026-08-26T10:02:10Z",
  "command_output": "ERROR: Could not find a version that satisfies the requirement ...",
  "restart_required": false
}
```

`404` for an unknown `job_id`, same envelope shape as other not-found errors in this API.

## Extending `GET /api/v1/pipelines`

Adds one top-level field alongside the existing per-pipeline `available`/`install_hint` list (spec
004) — does not change any existing field's meaning or shape:

```json
{
  "pipelines": [ /* unchanged shape from spec 004 */ ],
  "restart_required": true
}
```

`restart_required` is true if any `ExtrasInstallJob` has reached `completed` since the current
server process started, regardless of which extras group. See
[restart-required-signal.md](restart-required-signal.md) for exact semantics.

## Non-goals for this contract

Does not cover the viewer's rendering of any of this (locked cards, progress bars, banners — that's
the viewer project's own spec); does not cover cancelling an in-progress install (not in scope, see
spec.md Assumptions); does not cover auto-restart (deferred, see spec.md Assumptions and
`docs/development/roadmap_v1.7_to_v2.0.md`).
