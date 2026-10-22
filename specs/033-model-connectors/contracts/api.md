# Contract: API and CLI

All new endpoints need a signed-in user (or the local anonymous user, spec 029), except where
marked admin. No response contains a key; servers say `has_key: true|false` only.

## Model servers — `/api/v1/model-servers`

### `GET /api/v1/model-servers`

```json
{
  "servers": [
    {"id": "default", "name": "This computer (Ollama)", "kind": "ollama",
     "base_url": "http://127.0.0.1:11434", "has_key": false, "api_key_env": null,
     "endpoint": {"host": "127.0.0.1", "class": "local", "confirmed": true,
                  "sendable": true, "before_switch": false},
     "structured_output": null, "video_models": {}, "removable": false},
    {"id": "6f…", "name": "lab-gpu", "kind": "openai_compatible",
     "base_url": "http://10.0.4.12:8000/v1", "has_key": false, "...": "..."}
  ],
  "remote": {"allowed": false, "fixed_by_environment": false},
  "keys_file": "/data/model_server_keys.json"
}
```

### `POST /api/v1/model-servers` — add

Body `{name, kind, base_url, api_key?, api_key_env?}`. 201 with the server. Errors:
409 `SERVER_EXISTS` (same normalised URL), 403 `REMOTE_NOT_ALLOWED` ("Remote model servers are
switched off on this server." hint: "An administrator can switch them on in Settings → Model
servers, unless the server's configuration fixes it."), 422 `INVALID_ENV_NAME`.
`api_key` is written to the keys file, never to the database.

### `PATCH /api/v1/model-servers/{id}` — rename, change key or env name

Body any of `{name, api_key, api_key_env, clear_key: true}`. `base_url` and `kind` are fixed
(remove and add instead, so confirmations and provenance stay meaningful).

### `DELETE /api/v1/model-servers/{id}` — 204; removes its stored key.

### `POST /api/v1/model-servers/check`

Body `{kind, base_url, api_key?, id?, model?}` (an unsaved server can be checked; `id` uses a
saved server's key). Response:

```json
{
  "outcome": "models",
  "message": "Reachable. 2 models.",
  "hint": null,
  "kind_matches": true,
  "suggest_kind": null,
  "suggest_base_url": null,
  "models": ["qwen3-vl-8b", "gemma-3-12b"],
  "endpoint": {"host": "10.0.4.12", "class": "local", "confirmed": true, "sendable": true},
  "video": null
}
```

`outcome`: `unreachable` | `wrong_kind` | `no_models` | `models` | `key_refused` (FR-005).
`suggest_base_url` when `/v1` would work (R1); `suggest_kind: "ollama"` when the host also
answers Ollama's API (R6). `video` is `true`/`false` only when `model` was given (the probe,
R12), else `null`.

### `GET /api/v1/model-servers/settings` / `PUT …/settings` (admin)

`{"remote_model_servers": true}`. PUT 409 `FIXED_BY_ENVIRONMENT` when the environment variable
is set. Response includes the warning text the viewer shows (one source for the words).

## Ask AI — `/api/v1/ask` (changed)

- `TryRequest`, `RunRequest`, `DraftRequest`: new optional `backend` (default `ollama`) and
  `answer_shape`; `sampling.frames` unchanged; new `sampling.clip: {length, fps} | null`.
- `Entry`: new `backend`, `answer_shape`; `Answer`: new `fields`, `fit`, `fit_problem`,
  `clip_start`, `clip_end`, `clip_fps`.
- `require_confirmed` becomes `require_sendable`: 403 `REMOTE_NOT_ALLOWED` before 409
  `REMOTE_NOT_CONFIRMED`.
- `require_model` uses the backend's client. Ollama keeps its codes (503 `OLLAMA_UNREACHABLE`,
  422 `MODEL_NOT_PULLED`, Principle V); OpenAI-compatible servers give 503
  `MODEL_SERVER_UNREACHABLE`, 422 `MODEL_NOT_FOUND` ("The server lists: …"), 401 `KEY_REFUSED`.
- `GET /ask/endpoint?base_url=` unchanged, plus `sendable` and `before_switch`.

### `POST /api/v1/ask/suggest-shape`

Body `{prompt, backend?, base_url?, model?}` → `{"shape": [...] | null, "made_by": "rule" |
"model" | null, "message": "…"}`. The model is asked only when the rule finds nothing and a model
is given; text only, never frames; at try priority on the gate.

### `GET /api/v1/ask/entries/{id}/export/wide` — the wide answers CSV (shaped entries; 422
`NOT_SHAPED` otherwise). For a run, the same is `GET /api/v1/jobs/{id}/export/wide`.

## Unchanged, extended

- `GET /api/v1/vlm/models?server=<id>`: models of a listed server (default: `default`).
- `POST /api/v1/vlm/preview`: accepts `backend`, `answer_shape`.
- Provenance, methods and table exports: as data-model.md.

## CLI

```
videoannotator model-servers list [--json]
videoannotator model-servers add NAME URL --kind {ollama,openai_compatible} [--key-env VAR]
videoannotator model-servers check URL [--kind K] [--model M] [--json]
videoannotator model-servers remove NAME
```

A key is never a flag (FR-017): `add` reads it from `--key-env VAR` (stored as the env name) or,
when run interactively without `--key-env`, from a hidden prompt (`--no-key` skips). Run
configs set `backend`, `base_url`, `answer_shape`, `sampling_mode: clip` like any other key;
`job submit --config` takes them as today. Exit codes: 0 reachable with models, 2 unreachable,
3 wrong kind, 4 no models, 5 key refused.
