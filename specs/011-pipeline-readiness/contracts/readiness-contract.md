# Contract: Pipeline Readiness (spec 011)

Single source of truth for the shapes both `VideoAnnotator` (server) and `video-annotation-viewer`
(client) build against. Everything here is **additive** to specs 004/005: no existing field,
endpoint, status code or default changes meaning. Clients MUST treat a missing `readiness` object as
"server predates 011" and fall back to 005 behaviour (`available` / `install_hint` /
`restart_required`).

Auth: all endpoints need an API key (`X-API-Key` or `Authorization: Bearer`) unless marked
*public*. "Admin" means `GET /api/v1/auth/me` returns `is_admin: true`; otherwise `403`.

---

## 1. Pipeline readiness (extends `GET /api/v1/pipelines` and `GET /api/v1/pipelines/{name}`)

Each pipeline entry gains:

```jsonc
{
  "name": "speaker_diarization",
  "available": true,                 // unchanged (004): extras packages present
  "install_hint": null,              // unchanged (004)
  "readiness": {
    "state": "needs_setup",          // see state table
    "next_action": "setup",          // none | install | wait | restart | setup
    "extras_group": "audio",         // null when requires_extras is empty
    "install_job_id": null,          // set while state == "installing"
    "blockers": [
      {
        "kind": "secret",            // secret | service | import_error
        "name": "HF_AUTH_TOKEN",
        "message": "A Hugging Face access token is required to download the pyannote models.",
        "help_url": "https://huggingface.co/settings/tokens"
      }
    ],
    "notes": [
      {
        "kind": "licence",           // licence | weights_not_cached
        "name": "pyannote/speaker-diarization-3.1",
        "message": "Accept the model licence on Hugging Face with the same account as the token.",
        "help_url": "https://huggingface.co/pyannote/speaker-diarization-3.1",
        "approx_mb": null
      }
    ]
  }
}
```

Top level of `GET /api/v1/pipelines` is unchanged: `{ pipelines, total, restart_required }`.

### States (evaluated in this order; first match wins)

| `state`            | Meaning                                                                    | `next_action` |
|--------------------|----------------------------------------------------------------------------|---------------|
| `installing`       | An install job for this pipeline's extras group is `pending`/`running`     | `wait`        |
| `not_installed`    | Extras group not installed                                                 | `install`     |
| `restart_required` | Installed, but its install finished with `activation: "restart_required"`  | `restart`     |
| `needs_setup`      | Installed and active, but one or more `blockers` remain                    | `setup`       |
| `ready`            | Usable now. `notes` may still be non-empty (never blocking)                | `none`        |

Invariants:
- `blockers` is non-empty iff `state == "needs_setup"`.
- `state == "ready"` ⇒ `available == true`.
- `notes` never change `state`.
- Blocker `kind: "import_error"` carries the captured import message in `message`; it has no
  `help_url` and resolving it is an operator task (the viewer shows it, doesn't offer an action).
- Blocker `kind: "service"` with `name: "ollama"` — the viewer links to its existing Ollama
  diagnostics/model UI (spec 009); no new endpoint.

Clients MUST tolerate unknown `state`, `next_action` or `kind` values (render as a generic "not
ready" with the `message`), so later specs can add states without a viewer release.

---

## 2. Extras groups — `GET /api/v1/pipelines/extras`  *(new)*

```jsonc
{
  "extras": [
    {
      "name": "audio",
      "pipelines": ["audio_processing", "speech_recognition", "speaker_diarization"],
      "installed": false,
      "approx_download_mb": 3200,
      "includes_gpu_torch": true,
      "install_job_id": null        // id of a pending/running install, else null
    }
  ]
}
```

Order: as declared in `pyproject.toml`. Meta-groups (`all`, `dev`) are excluded.

---

## 3. Install job (extends 005's `GET /api/v1/pipelines/extras/install-jobs/{job_id}`)

Existing fields unchanged (`job_id, extra_name, status, created_at, started_at, finished_at,
command_output, restart_required`). Added, populated only when `status == "completed"`:

```jsonc
{
  "activation": "live",                  // live | restart_required
  "conflicting_distributions": [         // non-empty iff activation == "restart_required"
    { "name": "numpy", "old_version": "1.26.4", "new_version": "2.1.0" }
  ]
}
```

`restart_required` on the job equals `activation == "restart_required"`. The top-level
`restart_required` on `GET /api/v1/pipelines` is true iff any completed install since boot has
`activation == "restart_required"`.

Trigger endpoint is unchanged: `POST /api/v1/pipelines/extras/{extra}/install` → `202 { job_id,
extra_name, status }`.

---

## 4. Restart — `POST /api/v1/system/restart?force=false`  *(new, admin)*

- `202 { "restarting": true, "boot_id": "<current boot id>" }` — the process re-executes after the
  response is sent. Clients poll health (§5) until `boot_id` differs.
- `409` error body (standard error envelope) with `code`:
  - `RESTART_UNSUPPORTED` — `hint` holds the manual restart instruction for this deployment.
  - `JOBS_RUNNING` — `details.job_ids` lists running annotation jobs; retry with `force=true` to
    restart anyway (those jobs are marked failed/interrupted by existing recovery).
  - `INSTALL_IN_PROGRESS` — `details.install_job_ids`; `force` does not override this.

---

## 5. Boot identity (extends `GET /health`, `GET /api/v1/health`, `GET /api/v1/system/health`)

Added fields (*public* on `/health` and `/api/v1/health`, matching their current auth):

```jsonc
{
  "boot_id": "5f0c1c8e9b7a4d2e",   // random per process start
  "started_at": "2026-09-23T12:05:14Z",
  "restart_mode": "execv"          // execv | exit_for_supervisor | unsupported
}
```

Viewer restart wait: poll every 2 s; treat connection errors as "still restarting"; succeed when
`boot_id` changes; give up after 120 s with the `RESTART_UNSUPPORTED`-style manual hint.

---

## 6. Secrets: none

Dropped (2026-09-24): secrets such as `HF_AUTH_TOKEN` are set in the server's environment (the
container env). A missing one shows up only as a `secret` blocker in §1, whose `message` says where
to set it. There is no endpoint that reads or writes secrets.

---

## 7. Weight prefetch  *(new, admin, P3 — optional in first release)*

`POST /api/v1/pipelines/{name}/prefetch` → `202 { job_id, status }`. Polled with the same install-job
status endpoint and status vocabulary (`pending|running|completed|failed`, `command_output`).
`404` if the pipeline declares no `weights`.

---

## 8. Metadata schema addition (server-internal, listed for plugin authors)

```yaml
requires_setup:            # optional, default []
  - kind: secret           # secret | service | licence
    name: HF_AUTH_TOKEN
    description: Hugging Face access token
    help_url: https://huggingface.co/settings/tokens
  - kind: licence
    name: pyannote/speaker-diarization-3.1
    help_url: https://huggingface.co/pyannote/speaker-diarization-3.1
weights:                   # optional, default []
  - id: pyannote/speaker-diarization-3.1
    approx_mb: 50
```
