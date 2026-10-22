# Research: Model Connectors

**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Date**: 2026-10-22

What the code does today, what the servers accept, and the decisions this plan takes. Each
decision: what was chosen, why, and what else was considered.

## What exists (read 2026-10-22)

- **Client** (`pipelines/vlm_annotation/ollama_client.py`): `OllamaVLMClient` wraps
  `ollama.Client` (imported per call, so a live install of the `llm` extra is picked up).
  `list_models`, `model_details` (digest, quantisation), `preflight`, and `chat(...)` returning a
  `VLMCallResult` (raw text, thinking, total/load time, token counts, tokens/s, error). Retries
  with exponential backoff; a read timeout is not retried.
- **Pipeline** (`vlm_pipeline.py`): builds the client in `initialize()`, records a `ModelRef`
  with source `"ollama"` and the digest; writes `backend="ollama"` on every annotation; calls
  go through `gate_for(base_url, model)` at run priority. Labels come from `_parse_label`: a
  fixed pattern (`TOUCH`, `NO_TOUCH`, `YES`, `NO`, …), else the first 200 characters.
- **Ask AI** (`api/v1/ask.py`): tries are answered in a worker thread with its own
  `OllamaVLMClient`; `require_confirmed` (409 `REMOTE_NOT_CONFIRMED`) and `require_model`
  (503 `OLLAMA_UNREACHABLE`, 422 `MODEL_NOT_PULLED`) guard tries and runs. Runs are ordinary
  `vlm_annotation` jobs whose config carries `base_url`. The viewer never sets `base_url`: it
  uses whatever `/vlm/models` reports, the server's `OLLAMA_BASE_URL`. There is no list of
  servers and no way to choose one in the viewer.
- **Locality** (`endpoints.py`): `classify_endpoint` by resolved address; `normalise_base_url`
  keeps the path, so `http://h:8000/v1` and `http://h:8000` are different keys.
  `endpoint_confirmations` holds one row per confirmed normalised URL.
- **Elsewhere Ollama is assumed**: `/vlm/models` and `/vlm/preview` (`api/v1/vlm.py`),
  readiness (`api/readiness.py` via `diagnostics/ollama.py`), the CLI's diagnostics, the
  viewer's `VlmModelPicker` ("Checking Ollama…", "No models pulled") and Ask AI's no-model
  help ("ollama pull …").
- **Secrets**: `secrets_store.py` keeps the Hugging Face token in a file beside the database,
  mode 0600, written atomically, masked when shown (spec 031). `provenance.redact` replaces any
  value under a key matching `token|secret|password|api_key|auth` in recorded settings.
- **Tidy table** (spec 025, `exporters/readers.py`): long format, fixed `COLUMNS`
  (`measure`, `label`, `value`, …; `TABLE_FORMAT_VERSION` 2). A VLM answer gives a `vlm_label`
  row and, if different, a `vlm_reasoning` row.
- **Methods** (`exporters/methods.py`): says "on this computer", "on the local network" or "on a
  remote server", and deliberately doesn't name the server (spec 017 FR-029).
- **Prompt library**: `prompts` keyed by the SHA-256 of the exact text, the hash provenance
  records; tries and runs point at it from `ask_entries`.

## What the servers accept (checked 2026-10-22)

| | Images | Structured output | Reasoning | Models list | Video |
|---|---|---|---|---|---|
| vLLM | `image_url` (data URL) | `response_format: {type: json_schema, json_schema: {name, schema}}`; also `structured_outputs` in the body (`guided_json` removed in 0.12) | `reasoning_content` (newer: `reasoning`) | `/v1/models` | `video_url`; a frame sequence as `data:video/jpeg;base64,<f1>,<f2>,…`; per-request fps via `media_io_kwargs` only with `--trust-request-mm-kwargs` |
| llama.cpp `llama-server` | `image_url` with `--mmproj` | `response_format` `json_schema` (also `json_object` + `schema`) | `reasoning_content` (`--reasoning-format`) | `/v1/models` (often one model) | no |
| LM Studio | `image_url` | `response_format` `json_schema` | varies by model | `/v1/models` | no |
| SGLang | `image_url` | `response_format` `json_schema` | `reasoning_content` | `/v1/models` | Qwen-VL video, form to check |
| Ollama `/v1` | `image_url` | `response_format` `json_schema` (self-hosted ≥ 0.5) | `reasoning` / `reasoning_content` | `/v1/models` | no |
| Ollama native | `images` | `format: <JSON schema>` | `thinking` | `/api/tags`, with digest | no |
| Hosted (OpenAI and others) | `image_url` | `response_format` `json_schema` (`strict`) | not returned, or separate | `/v1/models` | no, or not through chat completions |

Sources: vLLM docs (multimodal inputs; structured outputs), llama.cpp server docs, Ollama's
OpenAI-compatibility page. Facts marked "varies" or "to check" are confirmed in the fake-server
tests' fixtures from real servers' replies at implementation (T-tasks), not assumed.

## Decisions

### R1 — One client for the OpenAI-compatible contract, on httpx

**Decision**: a new `openai_client.py` with `OpenAICompatibleVLMClient`, the same interface as
`OllamaVLMClient` (`list_models`, `model_details`, `preflight`, `chat` → `VLMCallResult`), plus
`check()` for the server check. Plain `httpx` (already a core dependency): `GET {base}/models`,
`POST {base}/chat/completions` with a `user` message of a text part and one `image_url` part
per frame (JPEG data URL). Both clients satisfy a small `VLMClient` protocol, and
`make_client(backend, base_url, timeout, key)` in a new `clients.py` chooses. Ask AI's try
worker, `require_model`, `/vlm/models` and `/vlm/preview` call `make_client` instead of
constructing `OllamaVLMClient`.

**Base URL**: the user gives the address as the server documents it. If it doesn't end in `/v1`
and `{base}/v1/models` answers where `{base}/models` doesn't, the check says so and offers the
`/v1` form. The stored address is what the user confirmed; no silent rewriting, so the address
in provenance and confirmations is the one requests go to.

**Why**: the `openai` package would be a new dependency for two endpoints (FR-009). The
existing client's retry and timeout behaviour is copied, not shared through a base class: the
two differ in where errors come from (exceptions vs. HTTP status) and the retry loop is short.

**Alternatives**: the `openai` SDK (new dependency, its own retries to tame); LiteLLM (large,
telemetry by default, Principle I); routing Ollama through `/v1` too (loses digests, R6).

### R2 — Calls, retries and errors

- Request: `model`, `messages`, `temperature` and `top_p` from `options`, `seed` from
  `options.seed` when set, `max_tokens` from `options.num_predict` when set, `stream: false`.
  Ollama-only keys (`keep_alive`, `think`) aren't sent; `think: false` is sent as
  `chat_template_kwargs: {enable_thinking: false}` only to servers whose check showed they
  accept it (vLLM, llama.cpp, SGLang); others ignore thinking settings.
- Retries: connection errors, 429, 5xx → backoff as today; read timeout → not retried, as today.
- **401/403 → `KeyRefusedError`**, never retried: the pipeline stops the run at once (FR-006,
  Story 3 scenario 5) with "the server refused the key"; Ask AI shows the same.
- 400 naming images or video ("does not support images", "multimodal", "video") → a plain
  "this model doesn't accept images/video" error, not retried.
- Result: `choices[0].message.content`; reasoning from `reasoning_content` or `reasoning`;
  `prompt_tokens`/`completion_tokens` from `usage`; tokens/s from completion tokens over the call
  time only when the server reports neither (marked as measured by us in the annotation, R10);
  `load_time` is `None`, not 0 (FR-007).

### R3 — Choosing the backend: a config key, Ollama by default

**Decision**: new optional config key `backend: ollama | openai_compatible`, default `ollama`.
A config without it behaves exactly as today (FR-003). `base_url` keeps its meaning; empty
still means the server's `OLLAMA_BASE_URL` and is only valid with `backend: ollama`.

**Why**: a config key is declarative (Principle III), shows in provenance and the settings form,
and keeps every existing config valid (Principle V).

**Alternatives**: inferring the backend from the URL (`:11434` → Ollama, `/v1` → OpenAI): wrong
for Ollama's own `/v1`, and invisible in provenance; a separate pipeline (`vlm_openai`): two of
everything in Ask AI and the guide.

### R4 — Model servers: a table the viewer can list, keys apart

**Decision**: a new table `model_servers` (data-model.md): id, name, kind, base URL
(normalised, unique), optional `api_key_env` (the name of an environment variable), whether a
stored key exists (derived, not a column), created by/at. The default Ollama server isn't a
row: it is always listed first, from `OLLAMA_BASE_URL`, and can't be deleted. Pipeline configs
carry `backend` and `base_url`, never a server id, so a config copied to another machine still
says where it went (Principle III); the server row supplies only the key and the name.

**Keys (Q2: C)**: at call time, for the normalised base URL: the row's `api_key_env`, if set and
present in the environment, wins; else the stored key; else none. Stored keys live in
`model_server_keys.json` beside the database (the `secrets_store.py` pattern: 0600, atomic,
never returned by any API, shown as "set" only). Keeping them out of SQLite means a copied
database file doesn't carry them; Settings says where the file is (FR-015 asked for "travels
with the server's data": the file sits in the same folder, so a copy of the folder does).

**Never in a config (FR-017)**: a `vlm_annotation` config with `api_key` (or any key matching
the secret pattern) is refused at submission with "keys go in Settings → Model servers or an
environment variable", so a key can't reach a job's stored config, provenance or a workflow.

**Never in logs (FR-016)**: the client sends the key only in the `Authorization` header and
never logs headers; httpx's own logging is at WARNING by default and logs no headers. A logging
filter on the `videoannotator` and `httpx` loggers replaces any known key's value with
`<redacted>` as a belt-and-braces measure, and SC-006's test searches every output for the key.

**Alternatives**: keys in the database (copied with it; visible to anyone reading the SQLite
file, the same as the file but easier to leak by sharing the DB for support); env var only (a
restart and a shell, Q2 rejected it as the only way); the OS keyring (not in containers).

### R5 — Remote servers switched off by default (Q1: A)

**Decision**: one server-wide setting, `remote_model_servers`, off on a fresh install.

- Stored in a new tiny `server_settings` key/value table; changed by an admin through
  `PUT /api/v1/model-servers/settings` (server administration, the one admin-only action here).
- The environment variable `VIDEOANNOTATOR_REMOTE_MODEL_SERVERS` (`on`/`off`), when set,
  overrides the stored value and the viewer shows the switch as fixed by the server's
  configuration. A lab can thus lock it off in its container config for an ethics board.
- Enforced on the server in one place, `endpoint_policy.require_sendable(db, base_url)`,
  called wherever `require_confirmed` is today (tries, runs, preview) and by the pipeline's
  `initialize()` for runs submitted any other way (CLI, API): a remote address is refused unless
  the switch is on **or** the address was confirmed before the upgrade. Then the spec 027
  confirmation still applies.
- **Existing confirmations keep working** (Principle V): the migration marks every
  `endpoint_confirmations` row that exists at upgrade as `before_switch = true`. Those addresses
  pass without the switch, and Settings lists them as "confirmed before the switch existed".
- Adding a remote server to the list while the switch is off is refused with the reason and how
  to switch it on (Story 3 scenario 1).

**Why**: a single, visible, lockable control is what an ethics application can cite; enforcing it
on the server (not the viewer) covers the CLI and the API. Grandfathering keeps v1.6 setups
with a remote Ollama working.

**Alternatives**: environment variable only (no way for a lab's admin without a shell); a
per-user switch (not what an approval covers); refusing grandfathered addresses (breaks
existing setups, Principle V).

### R6 — Ollama stays on its own client

**Decision**: `backend: ollama` keeps `OllamaVLMClient` unchanged. Ollama's `/v1` works through
`openai_compatible` (Story 1 scenario 6) but the server check, seeing Ollama's `/api/tags`
answer at the same host, suggests the Ollama kind "for model digests in your records".

**Why**: `/v1/models` has no digest; provenance would get worse for the default setup.

### R7 — Provenance for an OpenAI-compatible server

**Decision**: `ModelRef(name, source="openai_compatible", revision, revision_kind, note)`:

- revision = the response's `system_fingerprint` if any (`revision_kind: "server-fingerprint"`);
  else vLLM's `/v1/models` `root` (the weights' path or hub id) as a note; else `unknown` with
  "the server reported no digest".
- `provenance_vlm()` gains `backend` and, for shaped questions, `answer_shape_sha256` (R8). The
  existing `endpoint_class`/`endpoint_host` are unchanged. No key, ever (R4).
- Per annotation: `backend` is already a field, now `"openai_compatible"` where it applies.

Methods (`methods.py`) keeps its rule of not naming the server (spec 017 FR-029): "on a remote
server". The spec's FR-014 "name the host" is changed to match: the host is in provenance, the
paragraph says where in words. *(Spec amended alongside this plan.)*

### R8 — Answer shapes: a config key, sent the server's way, checked our way

**Decision**: new optional config key `answer_shape`: an ordered list of fields
`{name, kind, values?, min?, max?}`, kind one of `choice`, `yes_no`, `number`, `text`
(contracts/pipeline-config.md). Validation at submission: names are lower-case identifiers
(`[a-z][a-z0-9_]{0,39}`), unique, not reserved (`label`, `reasoning`, `fit`, `t`); 1–12 fields;
choices 2–30 values, each ≤ 60 characters.

- **To JSON Schema**: an object with every field required, `additionalProperties: false`;
  `choice` → `enum`, `yes_no` → `enum: ["yes", "no"]`, `number` → `number` with bounds,
  `text` → `string`, `maxLength` 200. Deterministic (sorted keys) so its SHA-256 is stable:
  provenance records `answer_shape_sha256` of the shape as stored.
- **Sent**: Ollama native → `format: <schema>`; OpenAI-compatible → `response_format: {type:
  json_schema, json_schema: {name: "answer", schema, strict: true}}`. If the server rejects
  `response_format` (400 naming it), the client retries once without it and the shape is
  described at the end of the prompt ("Answer with JSON only: {…}"); the server check records
  `structured_output: false` so later calls skip the attempt and Ask AI shows "this server can't
  enforce the answer shape" (Story 2 scenario 8).
- **The prompt itself is unchanged** for servers that enforce the shape. The prompt hash
  (`prompt_sha256`) stays the hash of the question's text; the shape has its own hash.
- **Checked**: our own small validator (no `jsonschema` dependency: four kinds, flat): take the
  first JSON object in the reply (tolerating a code fence or prose around it); check each field.
  Fits → `fields: {...}`, `fit: true`, and `label` = the first field's value, so lanes, tallies
  and every existing reader work unchanged. Doesn't fit → `fit: false`, `fit_problem: "…"`,
  `fields` with whatever did parse, `label` = `"DIDN'T FIT"`; `status` stays `answered`, no retry
  (FR-020).
- **No shape**: nothing changes, `_parse_label` as today (FR-021).

**Alternatives**: free JSON Schema from the user (too much for the audience, hard to make into
columns); the `jsonschema` package (a dependency for four field kinds); retrying on misfit (hides
how often a model misfits, which is a result in itself).

### R9 — Where the shape travels

- **Ask AI**: `TryRequest`, `RunRequest`, `Entry` gain `answer_shape`; `ask_entries` gains an
  `answer_shape` JSON column; answers gain `fields`, `fit`, `fit_problem`.
- **Library**: `prompts` gains `answer_shape` (the last shape used with this text), so opening a
  library question brings its shape; the history entry keeps the shape it ran with.
- **Workflows and project settings** (spec 029) store pipeline configs; `answer_shape` is a
  config key, so they carry it with no change.
- **Outputs**: the COCO file's top level gains `vlm_answer_shape` (the shape as run); each
  annotation gains `fields`, `fit`, `fit_problem` (additive, Principle II).

### R10 — The tidy table: one measure per field (FR-023 refined)

**Decision**: the spec 025 table is long format with fixed columns, and Principle II keeps it so.
A shaped answer becomes one row per field,
`measure = "vlm." + <field name>`, `label` = the value for choices/yes-no/text, `value` = the
number for `number`; plus one `vlm_fit` row (`label` = `yes`/`no`, and the problem as
`vlm_fit_problem` when it didn't fit). `vlm_label` and `vlm_reasoning` rows stay. One column per
field is then one `pivot_wider(names_from = measure, values_from = c(label, value))` in R or
`pivot` in pandas, which the export how-to shows. Ask AI's own "Download answers" offers, in
addition, a **wide CSV** for a shaped entry: one row per moment, one column per field, `fit`,
`fit_problem`, `t`, `frame`, `model` — the "straight into R" file SC-004 measures.

**Spec amendment**: FR-023 now says "one measure per field in the tidy table, and a wide file
with one column per field from Ask AI and the run's export menu". The table format version
doesn't change (no new columns; new measure names are additive, as 028's `note` was).

**Alternatives**: new columns in the tidy table per field (different columns per run; breaks the
fixed-column contract and concatenating runs); JSON in `label` (needs parsing, which SC-004
forbids).

### R11 — Suggest a shape (Q3: C)

**Decision**: `POST /api/v1/ask/suggest-shape {prompt, model?, base_url?, backend?}`:

1. **Rule** (`answer_shape.suggest_by_rule`): finds listed options in the question's text:
   "Return ONLY: A or B", "Answer with one of: A, B, C", "(A/B/C)", "yes or no" → a `choice`
   field per such list (named from a few words before it, else `answer`, `answer_2`), `yes_no`
   for yes/no, "on a scale of 1 to 5" → `number` 1–5. Deterministic, instant, tested on the
   example questions (`Examples.tsx`) and the default prompt.
2. **Model**, only when the rule finds nothing: one text-only call to the chosen model asking for
   fields in the shape's own JSON form, itself sent with a fixed meta-schema so the reply parses;
   validated like a user's shape; never with frames. Gated at try priority.
3. The reply says `made_by: rule | model`; the viewer shows it for editing and never applies it
   on its own (FR-024).

### R12 — Video input: a frame sequence, to servers that take it

**Decision**: `sampling_mode: clip`, with `clip_length_sec` (default 2.0) and `clip_fps`
(default 4). For each sample point, frames from `t − L/2` to `t + L/2` (shortened at the
video's ends, not shifted) at `clip_fps` are read with the existing `frames.read_sample` and
sent as one `video_url` part, `data:video/jpeg;base64,<f1>,<f2>,…`, vLLM's documented frame-
sequence form. When the server allows per-request media kwargs the call adds
`media_io_kwargs: {video: {fps: clip_fps}}` so the model sees real timing; the check records
whether it does.

- **Frames come only from the chosen video, cut on our server**: no file URL or upload.
- **Capability**: the server check sends one tiny synthetic clip (two 32×32 grey frames,
  `max_tokens: 1`) to the chosen model; a 2xx → `video: true`. No participant frames. The result
  is cached per (server, model) with the check; Ask AI offers the clip mode only when true
  (FR-027) and otherwise says "this model or server doesn't take video".
- `backend: ollama` with `clip` is refused at submission (Ollama has no video input).
- **Recorded**: `sampling_mode: "clip"`, `clip_start_sec`, `clip_end_sec`, `clip_fps`, plus
  `context_frame_numbers` and `frame_times` as bursts already record (FR-028). The viewer's
  sampling twin draws the span, as for bursts.
- **SGLang and hosted APIs**: not targeted; the probe decides. Documented as "vLLM with Qwen-VL
  models; other servers if the check says video".

**Alternatives**: an MP4 data URL (vLLM's docs show URLs and frame sequences; encoding MP4 needs
a codec on every platform); a file path with `--allowed-local-media-path` (only when vLLM shares
the disk; the frames-only-from-our-server rule is easier to hold with data we send).

### R13 — The server check

`POST /api/v1/model-servers/check {kind, base_url, api_key?, model?}` (also for unsaved
servers, so a user can check before adding) → `{reachable, kind_matches, models, endpoint:
{class, host, confirmed, sendable}, structured_output, video, hint}`; four outcomes as FR-005.
Structured-output support is learnt from the first shaped call (R8) rather than probed, because a
probe would need a model call per server; video is probed only when a model is named (R12).
`videoannotator model-servers check <url> [--kind] [--model] [--json]` prints the same (FR-030).
Readiness for `vlm_annotation` keeps checking the default Ollama server as today.

### R14 — The viewer

- **Settings → Model servers** (new card): the list (name, kind, address, badge, key set or
  not, remove), Add (name, kind, address, key or env-var name, Check), and for admins the remote
  switch with the warning text; "confirmed before the switch existed" rows.
- **Ask AI**: a server picker beside the model picker (default "This computer (Ollama)"); the
  badge follows the chosen server; `VlmModelPicker` takes the server and stops saying Ollama
  when it isn't one. The Answer shape editor under the question (Free text | Fields; field rows;
  Suggest); answers show fields; the tally and lane use `label` (the first field) unchanged.
  Frames per sample gains "A short clip" when the check says video.
- **Settings form** (New run, spec 026): `backend`, `answer_shape` and the clip keys come from
  the registry metadata; `answer_shape` uses the same editor component.
- **Words**: "Model server", "OpenAI-compatible (llama.cpp, vLLM, LM Studio…)", "Answer shape",
  "Fields"; never job/batch/dataset/preset (vocabulary test covers the new screens).

### R15 — Tests without real servers

A fake OpenAI-compatible server (`tests/fixtures/fake_openai_server.py`, an `httpx.MockTransport`
handler) answers `/models` and `/chat/completions` from recorded replies of vLLM, llama.cpp,
LM Studio and Ollama `/v1` (fixtures captured once from real servers at implementation and
committed, keys stripped). It can refuse keys, reject `response_format`, reject video, and
return reasoning in either field. One opt-in integration test (`-m real_vlm_server`, env
`VA_TEST_OPENAI_BASE_URL`) runs the demo clip against a real server.
