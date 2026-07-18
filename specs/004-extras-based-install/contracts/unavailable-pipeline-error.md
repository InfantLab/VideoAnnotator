# Contract: Unavailable-Pipeline Error Surface

Applies to every user-facing surface that can name a pipeline: CLI (`videoannotator job submit
... --pipelines <name>`), REST API (`POST /api/v1/jobs`, `GET /api/v1/pipelines`).

## Registry listing (`GET /api/v1/pipelines`, `videoannotator pipelines list`)

Unavailable pipelines (extras not installed) are **omitted** from the default listing — not shown
with an error, not shown at all — per FR-005. A `--all` / `?include_unavailable=true` variant MAY
list them with an `available: false` + `install_hint` field for discoverability (implementation
detail for `/speckit-tasks`; not required by any FR but consistent with SC-006's discoverability
goal).

## Job submission naming an unavailable pipeline

**CLI**: non-zero exit, stderr message, no Python traceback:

```
Error: pipeline 'face_laion_clip' is not available in this install.
Install it with: pip install videoannotator[face-laion]
```

**API**: `422 Unprocessable Entity` (not `500`), JSON body:

```json
{
  "detail": "Pipeline 'face_laion_clip' is not available in this install.",
  "install_hint": "pip install videoannotator[face-laion]",
  "pipeline": "face_laion_clip"
}
```

## Migration case (pipeline demoted from v1.4.4 default install)

Same shape as above, with `detail` distinguishing "no longer in the default install" from "not a
recognized pipeline" (research.md §4), e.g.:

```
Error: pipeline 'face_laion_clip' is not available in this install.
As of v1.5.0, LAION pipelines are no longer installed by default.
Install it with: pip install videoannotator[face-laion]
```

## Non-goals for this contract

Does not cover authentication/authorization errors, malformed config errors, or any error not
related to pipeline availability — those are unchanged by this phase.
