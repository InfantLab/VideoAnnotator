# Data Model: Model Connectors

**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Research**: [research.md](research.md)

Everything here is additive: new tables, new nullable columns, new optional keys. Nothing
existing is renamed or removed (Principles II, V).

## Database

### `model_servers` (new)

A model server someone added. The default Ollama server (`OLLAMA_BASE_URL`) is not a row; the
API lists it first as `id: "default"`.

| Column | Type | Notes |
|---|---|---|
| `id` | GUID, PK | |
| `name` | String(100), not null | what the user calls it ("lab-gpu") |
| `kind` | String(20), not null | `ollama` \| `openai_compatible` |
| `base_url` | String(500), not null, unique | normalised (`endpoints.normalise_base_url`) |
| `api_key_env` | String(100), null | name of an environment variable holding the key; `[A-Z_][A-Z0-9_]*` |
| `structured_output` | Boolean, null | learnt from the first shaped call; null = not known yet (R8) |
| `video_models` | JSON, not null, default `{}` | `{model: true\|false}` from the video probe (R12) |
| `created_by` | String(64), null | user id |
| `created_at` | DateTime(tz), not null | |

Rules: a `base_url` that classifies as remote can only be added while remote servers are
allowed (R5). Removing a row deletes its stored key; outputs that used it are untouched.

### `server_settings` (new)

| Column | Type | Notes |
|---|---|---|
| `name` | String(64), PK | `remote_model_servers` |
| `value` | JSON, not null | `true` / `false` |
| `updated_by` | String(64), null | |
| `updated_at` | DateTime(tz), not null | |

Effective value: `VIDEOANNOTATOR_REMOTE_MODEL_SERVERS` (`on`/`off`) if set, else the row, else
`false`.

### `endpoint_confirmations` (changed)

| Column | Change |
|---|---|
| `before_switch` | **new** Boolean, not null, default `false`. The migration sets `true` on every row present at upgrade. |

A remote base URL is **sendable** when (remote switch on **or** `before_switch`) **and** a
confirmation row exists. Local base URLs are always sendable.

### `ask_entries` (changed)

| Column | Change |
|---|---|
| `backend` | **new** String(20), not null, default `ollama` |
| `answer_shape` | **new** JSON, null |

Each element of `answers` may carry `fields` (object), `fit` (bool), `fit_problem` (string), and
for clips `clip_start`, `clip_end`, `clip_fps`.

### `prompts` (changed)

| Column | Change |
|---|---|
| `answer_shape` | **new** JSON, null: the last shape used with this text |

## Files

### `model_server_keys.json` (new, beside the database)

`{"<normalised base_url>": "<key>"}`. Mode 0600, written atomically, read per call (so a key
changed in Settings applies to the next call). Never returned by the API; `has_key` is derived
from it. Same folder as `huggingface_token` (spec 031).

## Answer shape

```json
[
  {"name": "touch", "kind": "yes_no"},
  {"name": "initiator", "kind": "choice", "values": ["mother", "infant", "both", "none"]},
  {"name": "confidence", "kind": "number", "min": 1, "max": 5}
]
```

| Field | Rule |
|---|---|
| `name` | `^[a-z][a-z0-9_]{0,39}$`, unique, not `label`, `reasoning`, `fit`, `fit_problem`, `t` |
| `kind` | `choice` \| `yes_no` \| `number` \| `text` |
| `values` | `choice` only: 2–30 distinct strings, each 1–60 characters |
| `min`, `max` | `number` only, optional, `min ≤ max` |
| list | 1–12 fields |

`answer_shape_sha256`: SHA-256 of the shape serialised as compact JSON with sorted keys.

JSON Schema sent to the server: `{"type": "object", "properties": {...}, "required": [all
names in order], "additionalProperties": false}`; `choice` → `{"type": "string", "enum":
values}`; `yes_no` → `{"type": "string", "enum": ["yes", "no"]}`; `number` → `{"type":
"number", "minimum"?, "maximum"?}`; `text` → `{"type": "string", "maxLength": 200}`.

## Answer (one sample, in a try or a run's COCO annotation)

Existing fields unchanged. Added, only when they apply:

| Field | When | Value |
|---|---|---|
| `backend` | always (already present) | `ollama` \| `openai_compatible` |
| `fields` | shaped | object of the fields that parsed |
| `fit` | shaped | `true` / `false` |
| `fit_problem` | shaped, `fit: false` | e.g. "initiator: 'mom' is not one of mother, infant, both, none" |
| `label` | shaped | the first field's value as text if fit; `DIDN'T FIT` if not |
| `clip_start_sec`, `clip_end_sec`, `clip_fps` | `sampling_mode: clip` | as sent |
| `load_time`, `tokens_per_sec` | OpenAI-compatible | `null` when the server doesn't report them (FR-007) |

COCO top level gains `vlm_answer_shape` (the shape as run) for shaped runs.

## Provenance (`vlm` block, spec 017)

| Key | Change |
|---|---|
| `backend` | **new** |
| `answer_shape_sha256` | **new**, shaped runs only |
| `structured_output` | **new**: `enforced` \| `prompt_only` |
| `endpoint_class`, `endpoint_host`, `base_url`, `prompt_sha256`, `model_digest`, `quantization` | unchanged |

`models[]` entry: `source: openai_compatible`, `revision_kind: server-fingerprint | unknown`.

## Tidy table rows (spec 025, format version unchanged)

Per shaped answer, in addition to `vlm_label` and `vlm_reasoning`:

| `measure` | `label` | `value` |
|---|---|---|
| `vlm.<name>` (choice, yes_no, text) | the value | |
| `vlm.<name>` (number) | | the number |
| `vlm_fit` | `yes` / `no` | |
| `vlm_fit_problem` (only if not fit) | the problem | |

## Wide answers file (new export)

`<video>_answers_wide.csv`: `video, t, frame, <field 1> … <field n>, fit, fit_problem, model,
backend, raw_response`. One row per sample with an answer; errors and skipped samples are rows
with empty fields and `fit` empty.

## State: a model server's sendability

```
            add (local)                     add (remote; switch on)
  (none) ───────────────▶ listed, local ──▶ sendable
  (none) ─────────────────────────────────▶ listed, remote, unconfirmed
                                              │ confirm (027)
                                              ▼
                                            listed, remote, confirmed ──▶ sendable
  switch off ──▶ remote rows not sendable unless before_switch
```
