---
description: "Tasks for spec 033: Model Connectors"
---

# Tasks: Model Connectors

**Input**: `specs/033-model-connectors/` (plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md)
**Tests**: included. The constitution requires tests for new API surface; the shape validator and
clip spans have shared cases with the viewer (contracts/pipeline-config.md); SC-005 and SC-006
are checked by tests.
**Organization**: by user story in the spec's priority order (US1, US2 P1; US3 P2; US4, US5 P3).
Within a story, server before viewer, tests beside the code they test.

**Conventions for every task**: Python passes `ruff check`, `ruff format`, `mypy src/videoannotator`;
viewer passes `bun run lint`, `tsc --noEmit -p tsconfig.app.json`, `bun run test:run`. Model replies
are pipeline-supplied strings: never `dangerouslySetInnerHTML`. No comments saying *what* code
does. User-visible words: "model server", "OpenAI-compatible (llama.cpp, vLLM, LM Studio…)",
"answer shape", "fields", "a short clip"; never job, batch, dataset, preset. A key is never
logged, returned, or written outside `model_server_keys.json`. Commit per task or small group,
`<area>: <summary>`, with the Co-Authored-By trailer; the container has no git identity
(`git -c user.name=… -c user.email=…`).

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup

- [X] T001 Record real replies for the fake server in `tests/fixtures/openai_compatible/`: one JSON file per server (`vllm.json`, `llama_cpp.json`, `lm_studio.json`, `ollama_v1.json`), each with `models` (the `/v1/models` body), `chat` (a chat-completions body for one image), `chat_reasoning` (with the server's reasoning field, if any), `chat_schema` (a reply to a `response_format: json_schema` request) and `errors` (`schema_rejected`, `images_rejected`, `video_rejected`, `unauthorized` bodies and status codes). Capture from real servers where available (llama.cpp and Ollama in the dev container; vLLM from its docs' examples if no GPU), strip keys and hostnames, and note the source and date in a `_source` field. Where a reply couldn't be captured, mark `_source: "from docs, unverified"`
- [X] T002 [P] Create `tests/fixtures/answer_shape/cases.json` per contracts/pipeline-config.md: `shapes` (valid: one choice, yes/no + choice + number 1–5, text; invalid with the expected problem: bad name, duplicate, reserved `label`, 0 fields, 13 fields, choice with 1 value, `min > max`), `replies` per shape (fits; in a ```json fence; prose around the object; a value outside the list; a missing field; an extra field; a number out of range; not JSON; empty) each with expected `fields`, `fit`, `fit_problem`, `label`, and `suggest` (question texts → expected rule suggestion or `null`: "Return ONLY: TOUCH or NO_TOUCH", "Answer with one of: smile, frown, neutral", "(yes/no)", "on a scale of 1 to 5", the default prompt from `vlm_pipeline.py`, each `Examples.tsx` question, and a free question with none)
- [X] T003 [P] Add clip cases to `tests/fixtures/ask_ai/sampling_cases.json`: `clip: {length, fps}` with expected `clip_times_at_first_point` and span `[start, end]`: 2 s at 4 fps mid-video; at t=0 (shortened to [0, 1]); at the end; length longer than the video

---

## Phase 2: Foundational (blocks every story)

- [X] T004 Create `src/videoannotator/pipelines/vlm_annotation/clients.py`: `VLMClient` Protocol (`base_url`, `list_models() -> list[str]`, `model_details(model) -> dict[str, str | None]`, `preflight(model)`, `chat(*, model, prompt, images, think, keep_alive, options, max_retries, retry_backoff_sec, schema=None, video=None) -> VLMCallResult`), `Backend = Literal["ollama", "openai_compatible"]`, `KeyRefusedError(RuntimeError)`, `ModelServerUnavailableError` (subclassing `OllamaUnavailableError` so existing `except` clauses keep working), and `make_client(backend, base_url, timeout, api_key=None) -> VLMClient`. Make `VLMCallResult.load_time` and `tokens_per_sec` accept `None` (FR-007); check every reader of them (`vlm_pipeline.py` `round(...)`, `ask.py`) handles `None`
- [X] T005 Create `tests/fixtures/fake_openai_server.py`: `FakeOpenAIServer(flavour="vllm")` loading T001's file, exposing `transport` (an `httpx.MockTransport`) and switches `refuse_key`, `reject_schema`, `reject_images`, `reject_video`, `down`, plus a `requests` log (method, path, JSON body, whether an `Authorization` header was present — never its value). Usable by `make_client` through an injectable transport (T006)
- [X] T006 Create `src/videoannotator/pipelines/vlm_annotation/openai_client.py`: `OpenAICompatibleVLMClient(base_url, timeout, api_key=None, transport=None)` per research R1/R2: `list_models` (`GET {base}/models`, `data[].id`; connection error → `ModelServerUnavailableError`; 401/403 → `KeyRefusedError`); `model_details` (`root` and `owned_by` from the models entry, digest `None`); `preflight`; `chat` building `messages=[{role: user, content: [{type: text, text: prompt}, *({type: image_url, image_url: {url: "data:image/jpeg;base64,…"}})]}]`, `temperature`/`top_p`/`seed`/`max_tokens` from `options`, `stream: false`; retries on connection errors, 429 and 5xx with today's backoff; read timeout not retried; 401/403 raise `KeyRefusedError` (not retried); 400 mentioning image/multimodal → error result "this model doesn't accept images"; parse content, reasoning from `reasoning_content` or `reasoning`, `usage` tokens, `system_fingerprint` kept on the result for provenance. The key only in the `Authorization: Bearer` header; never in an exception message or log
- [X] T007 [P] `tests/unit/test_openai_client.py`: against each T001 flavour through T005: models listed; one-image chat parsed (content, reasoning in either field, tokens, `load_time is None`); retry on 503 then success; no retry on read timeout or 401; `KeyRefusedError` on 401/403; images rejected → plain error; the request body has one `image_url` per frame and no Ollama-only keys; the key appears in no exception text (`str(e)`), no log record (caplog at DEBUG), and only in the header
- [X] T008 Add the `backend` config key to `src/videoannotator/pipelines/vlm_annotation/vlm_pipeline.py` (default `"ollama"`) and use `make_client` in `initialize()`; record `backend` on every annotation (already a field, now from config) and in `provenance_vlm()`; `ModelRef` source = backend, `revision_kind` `server-fingerprint` from the first call's `system_fingerprint` for OpenAI-compatible, else `unknown` with "the server reported no digest" (research R7). Empty `base_url` with `openai_compatible` → `ValueError("Say which model server to use.")`
- [X] T009 [P] `tests/unit/test_vlm_pipeline_backends.py`: a config without `backend` builds `OllamaVLMClient` and writes output identical to today's (compare with an existing fake-VLM test's expectations); `backend: openai_compatible` with the fake writes COCO output with the same structure, `backend: openai_compatible`, provenance `vlm.backend`, the model ref; `null` timings stay `null`
- [X] T010 Add the additive migration in `src/videoannotator/database/migrations.py` and models in `src/videoannotator/database/models.py`: `ModelServer` (`model_servers`), `ServerSetting` (`server_settings`), `EndpointConfirmation.before_switch` (default false; the migration sets true on every row present when the column is added), `AskEntry.backend` (default `ollama`), `AskEntry.answer_shape`, `Prompt.answer_shape` (data-model.md). Test in `tests/unit/test_migrations_033.py`: an old database (create with the pre-033 schema) upgrades, keeps its rows, existing confirmations get `before_switch = true`, new ones default false

**Checkpoint**: the pipeline can call either backend from config; nothing user-facing has changed.

---

## Phase 3: User Story 1 — Use the model server the lab already runs (P1) 🎯 MVP

**Goal**: add an OpenAI-compatible server in Settings, check it, choose it in Ask AI, try and run.
**Independent test**: quickstart Story 1 with llama.cpp or vLLM and Ollama stopped.

- [X] T011 [US1] Create `src/videoannotator/model_servers.py`: `list_servers(db)` (the default Ollama server first as `id="default"`, from `default_ollama_base_url()`, then rows by name), `get(db, id)`, `add(db, name, kind, base_url, api_key=None, api_key_env=None, user_id=None)` (normalises the URL; 409 on duplicate; remote check left to T038 via a hook `endpoint_policy.require_addable` that allows everything until US3), `update`, `remove` (also removes the stored key), `server_for_url(db, base_url)`, and `key_for(db, base_url) -> str | None` (env var named by the row wins when set and non-empty, else the stored key, else None; research R4)
- [X] T012 [US1] Add keys-file helpers to `src/videoannotator/secrets_store.py`: `model_server_keys_path()` beside the database, `read_model_server_key(base_url)`, `save_model_server_key(base_url, key)` (0600, atomic, as the HF token), `remove_model_server_key(base_url)`, `known_model_server_keys()`; and `KeyRedactingFilter`, a `logging.Filter` that replaces any known key (from the file and from env vars named by servers) in a record's message and args with `<redacted>`; install it on the `videoannotator` and `httpx` loggers at app startup in `src/videoannotator/api/main.py` and in the CLI entry
- [X] T013 [P] [US1] `tests/unit/test_model_servers.py`: default listed first and not removable; add/duplicate/remove; key resolution order (env var set → env; env var named but unset → stored; neither → None); keys file mode 0600; removing a server removes its key; the filter redacts a key logged by accident
- [X] T014 [US1] Refuse keys in configs: in `src/videoannotator/pipelines/vlm_annotation/vlm_pipeline.py` `initialize()` and in job submission validation (where `api/v1/jobs` validates pipeline configs; find with `grep -rn "def validate_config\|config_validator" src/videoannotator/api`), any `vlm_annotation` key matching `provenance._SECRET_KEY` with a non-empty string value → 422 `KEY_IN_CONFIG` with contracts/pipeline-config.md's message. The pipeline gets its key from `model_servers.key_for(base_url)` at `initialize()` via a small accessor that opens its own DB session (the pipeline runs outside requests)
- [X] T015 [US1] Add `check()` to `OpenAICompatibleVLMClient` and an Ollama equivalent in `src/videoannotator/pipelines/vlm_annotation/clients.py` (`check_server(kind, base_url, api_key, model=None) -> dict`): outcomes `unreachable`, `wrong_kind`, `no_models`, `models`, `key_refused` (FR-005); `suggest_base_url` when `{base}/v1/models` answers and `{base}/models` doesn't; `suggest_kind: ollama` when `{host}/api/tags` answers (research R1, R6); each outcome with a plain message and hint
- [X] T016 [US1] Create `src/videoannotator/api/v1/model_servers.py` and register it in `src/videoannotator/api/v1/__init__.py`: `GET /model-servers`, `POST /model-servers`, `PATCH /model-servers/{id}`, `DELETE /model-servers/{id}`, `POST /model-servers/check` per contracts/api.md (no key in any response; `has_key` derived; `endpoint` from `ask.endpoint_state`; `keys_file` path). The settings endpoints come in T031
- [X] T017 [P] [US1] `tests/api/test_model_servers_api.py`: list/add/patch/delete/check with the fake; duplicate → 409; check outcomes for each switch of the fake; `suggest_base_url`; a key sent on add never appears in any response body (search every response in the test)
- [X] T018 [US1] Route Ask AI and `/vlm` through the backend in `src/videoannotator/api/v1/ask.py` and `src/videoannotator/api/v1/vlm.py`: `TryRequest`, `RunRequest`, `DraftRequest` gain `backend: Literal["ollama", "openai_compatible"] = "ollama"`; `Entry` gains `backend`; `_answer_try`, `require_model` and preview use `make_client(backend, base_url, timeout, key_for(db, base_url))`; Ollama keeps `OLLAMA_UNREACHABLE`/`MODEL_NOT_PULLED`, OpenAI-compatible gives `MODEL_SERVER_UNREACHABLE`/`MODEL_NOT_FOUND`/`KEY_REFUSED` (contracts/api.md); runs put `backend` in the job config; `ask_history.create_entry` stores `backend` (`src/videoannotator/ask_history.py`); `GET /vlm/models?server=<id>` lists that server's models (default `default`, today's behaviour)
- [X] T019 [P] [US1] `tests/api/test_ask_backends_api.py`: a try and a run through the fake OpenAI-compatible server (patch `make_client`'s transport) give answers like the Ollama fake's; the entry and job config record `backend`; the try worker respects the gate (reuse the gate test pattern from `tests/api/test_ask_api.py`); `/vlm/models?server=` lists the fake's models; existing `tests/api/test_ask_api.py` passes untouched
- [X] T020 [US1] Update `src/videoannotator/registry/metadata/vlm_annotation.yaml`: `backends: [ollama, openai_compatible]` (add `backends` to `PipelineMetadata` in `src/videoannotator/registry/` as an optional list if it isn't there), the `backend` key per contracts/pipeline-config.md, `base_url`/`model` descriptions that don't say Ollama only, `description`/`homepage` wording; check the pipeline guide (spec 026) still renders (`tests/registry` and the guide tests)
- [X] T021 [P] [US1] Viewer API: add model-server methods and the new `backend` fields to `viewer/src/api/client.ts`; regenerate `viewer/src/api/schema.d.ts` (the repo's OpenAPI script; see how 029 did it in `git log -p --stat -- viewer/src/api/schema.d.ts | head`); types in `viewer/src/types/ask.ts`
- [X] T022 [US1] Create `viewer/src/components/settings/ModelServers.tsx` and add it to `viewer/src/pages/Settings.tsx` per contracts/viewer.md (list with `EndpointBadge`, Add with Check, suggested address/kind buttons, Remove, keys-file line; no admin switch yet). The key input is a password field, cleared after save, never pre-filled
- [X] T023 [US1] Create `viewer/src/components/ask/ServerPicker.tsx`; wire it into `viewer/src/pages/AskAI.tsx` and `viewer/src/components/ask/QuestionPanel.tsx` before the model picker: default "This computer (Ollama)", choice remembered with the other Ask AI settings (spec 029 `user_preferences`), the badge (`EndpointBadge`) follows the chosen server, requests send `backend` and `base_url`. Make `viewer/src/components/VlmModelPicker.tsx` and `viewer/src/hooks/useVlmModels.ts` take a server id; words per contracts/viewer.md; the no-model help in `AskAI.tsx` says what to do for the chosen kind
- [X] T024 [US1] New run settings form: render `backend` from metadata and show `base_url` as a server picker when servers are known (`viewer/src/components/DynamicPipelineParameters.tsx` or the spec 026 form component it uses; find with `grep -rn "widget" viewer/src/components | head`)
- [X] T025 [P] [US1] Viewer tests: `viewer/src/components/settings/ModelServers.test.tsx` (list, add/check flow, suggestions, no key ever rendered), `viewer/src/pages/AskAI.servers.test.tsx` (picker, badge follows server, requests carry backend/base_url); extend `viewer/src/test/vocabulary.test.tsx` to render Settings with servers; rebuild with `bash scripts/build_viewer.sh` and commit `viewer_static`

**Checkpoint**: Story 1 works end to end; Ollama setups unchanged.

---

## Phase 4: User Story 2 — Answers in a fixed shape (P1)

**Goal**: a question carries fields; answers are requested and checked in that shape; files have a column per field.
**Independent test**: quickstart Story 2 with the default Ollama model.

- [X] T026 [US2] Create `src/videoannotator/pipelines/vlm_annotation/answer_shape.py`: `Field` and `validate_shape(raw) -> list[Field]` (data-model.md rules; `ValueError` naming the field and rule), `to_json_schema(shape)`, `shape_sha256(shape)` (compact JSON, sorted keys), `describe_for_prompt(shape)` (the fallback text), `check_reply(shape, text) -> ShapeResult(fields, fit, fit_problem, label)` (first JSON object in the text, tolerant of fences and prose; per-field checks; `label` = first field as text or `DIDN'T FIT`), and `suggest_by_rule(prompt) -> list[Field] | None` (research R11)
- [X] T027 [P] [US2] `tests/unit/test_answer_shape.py` reading `tests/fixtures/answer_shape/cases.json`: every shape, reply and suggestion case; `to_json_schema` output for each valid shape; the hash is stable across key order
- [X] T028 [US2] Send shapes: `OllamaVLMClient.chat(..., schema=...)` passes `format=schema` (`src/videoannotator/pipelines/vlm_annotation/ollama_client.py`); `OpenAICompatibleVLMClient.chat(..., schema=...)` sends `response_format: {type: json_schema, json_schema: {name: "answer", schema, strict: true}}`, and on a 400 naming `response_format`/`json_schema` retries once with the shape described at the end of the prompt and marks the result `structured_output="prompt_only"` (else `"enforced"`); `model_servers` records `structured_output` for the server on the first shaped call (research R8)
- [X] T029 [US2] Pipeline and tries: `answer_shape` config key in `vlm_pipeline.py` (validated in `initialize()`; `INVALID_ANSWER_SHAPE` at submission through the same validator as T014); shaped calls pass the schema; replies go through `check_reply`; annotations gain `fields`, `fit`, `fit_problem`, `label` from the result; misfits count as answered; the COCO top level gains `vlm_answer_shape`; provenance gains `answer_shape_sha256` and `structured_output`. In `ask.py`: `answer_shape` on `TryRequest`/`RunRequest`/`DraftRequest`/`Entry`, `Answer` gains `fields`/`fit`/`fit_problem`, `_answer_from_annotation` reads them; `ask_history` stores the entry's shape and sets `prompts.answer_shape` (library) on each use
- [X] T030 [P] [US2] Tests: `tests/unit/test_vlm_pipeline_shapes.py` (both backends via fakes: schema sent the right way; fit/misfit recorded; label; provenance hash; fallback to prompt-only on a rejecting server; unshaped output byte-identical to before), `tests/api/test_ask_shapes_api.py` (try/run with a shape; history and library carry it; reopening a library prompt returns its shape)
- [X] T031 [US2] Tidy table and wide file: in `src/videoannotator/exporters/readers.py` `read_vlm` adds `vlm.<name>` rows (label or value by kind), `vlm_fit`, `vlm_fit_problem` (data-model.md); in `src/videoannotator/exporters/table.py` add `write_vlm_wide_csv(annotations, shape, video, out)`; add `GET /api/v1/ask/entries/{id}/export/wide` (`ask.py`) and `GET /api/v1/jobs/{id}/export/wide` (where the job table export lives; `grep -rn "export/table" src/videoannotator/api`). Update `specs/025-results-out/contracts/table.md` with the new measure names (additive). Tests in `tests/unit/test_table_vlm_fields.py` and `tests/api/test_export_wide.py`; existing table tests pass untouched
- [X] T032 [US2] `POST /api/v1/ask/suggest-shape` in `ask.py` per contracts/api.md: rule first; else, if a model is given, one text-only call (gate at try priority) asking for fields, sent with a fixed meta-schema for the shape list, validated with `validate_shape`; `made_by`. Test in `tests/api/test_ask_suggest_shape.py` (rule hit makes no model call; rule miss calls the fake once with no images; invalid model proposal → `shape: null` with a message)
- [X] T033 [P] [US2] Create `viewer/src/lib/answerShape.ts` (validation twin) and `viewer/src/lib/answerShape.test.ts` reading the shared cases (shapes and the rule suggestions only; reply checking stays on the server)
- [X] T034 [US2] Create `viewer/src/components/ask/AnswerShapeEditor.tsx` (Free text | Fields; rows of name, kind, values as chips or range; add, remove, reorder; validation messages; Suggest with "suggested from your question" / "suggested by the model"; the "can't enforce" note when the server's `structured_output` is false) and wire it into `AskAI.tsx` under the question; the shape is saved and restored with the question (history, library, `restoreFields`), and sent with tries and runs
- [X] T035 [US2] Show fields: `viewer/src/components/ask/AnswersTable.tsx` (fields as key–value; "Didn't fit: …" with the raw reply under a disclosure), `Tally.tsx` and the lanes in `viewer/src/components/lanes/` (misfit as its own colour, never merged with a value; Principle VI); the Video page's data table shows a column per field for a shaped lane (`viewer/src/components/video/`); the download menu adds the wide CSV for shaped entries and runs
- [X] T036 [US2] Settings form widget: `widget: answer_shape` in `vlm_annotation.yaml` and the form renders `AnswerShapeEditor` for it (same component as T034; no Suggest without a model)
- [X] T037 [P] [US2] Viewer tests: `viewer/src/components/ask/AnswerShapeEditor.test.tsx` (edit, validate, Suggest never applies without the user, reorder), answers/tally/lane misfit rendering, contract fixture with a shaped output in `viewer/src/test/` (the existing contract test's fixtures folder); vocabulary test; rebuild `viewer_static`

**Checkpoint**: Stories 1 and 2 work independently; unshaped questions behave as before.

---

## Phase 5: User Story 3 — A hosted model, knowingly (P2)

**Goal**: remote servers off by default; an admin switches them on; keys never leak; a refused key stops a run.
**Independent test**: quickstart Story 3.

- [ ] T038 [US3] Create `src/videoannotator/endpoint_policy.py`: `remote_allowed(db) -> (allowed, fixed_by_environment)` (env `VIDEOANNOTATOR_REMOTE_MODEL_SERVERS` `on`/`off` overrides the `server_settings` row; default off), `set_remote_allowed(db, value, user_id)`, `require_addable(db, base_url)` (403 `REMOTE_NOT_ALLOWED` for a remote address while off), `require_sendable(db, base_url)` (local → ok; remote → allowed or `before_switch`, else 403 `REMOTE_NOT_ALLOWED`; then spec 027's confirmation, else 409 `REMOTE_NOT_CONFIRMED`), and the warning text (FR-012) as a constant served by the API
- [ ] T039 [US3] Enforce everywhere: replace `require_confirmed` with `require_sendable` in `ask.py` (tries, runs, drafts, suggest-shape with a model) and `vlm.py` (preview); call `require_addable` from `model_servers.add`; in `vlm_pipeline.py` `initialize()` call `require_sendable` through its own DB session so a job submitted by API or CLI with a remote `base_url` fails before any frame is read, with the same message; `endpoint_state` gains `sendable` and `before_switch`
- [ ] T040 [US3] `GET`/`PUT /api/v1/model-servers/settings` in `api/v1/model_servers.py` (PUT behind `require_admin`; 409 `FIXED_BY_ENVIRONMENT`); the list response's `remote` block
- [ ] T041 [US3] `KeyRefusedError` stops a run: in `vlm_pipeline.py` catch it around the call, write what it has, and raise `RuntimeError("The model server refused the key.")` without further samples; Ask AI tries stop at the first refusal with the same words
- [ ] T042 [P] [US3] `tests/api/test_ask_remote_switch.py` (SC-005): fresh database → adding, trying, previewing, running (API and `JobProcessor` path) against a remote address are all refused and the fake records zero chat requests; switch on → confirmation still required; env var `off` fixes it and PUT gets 409; a confirmation with `before_switch` passes with the switch off; non-admin PUT → 403
- [ ] T043 [P] [US3] `tests/unit/test_no_key_leaks.py` (SC-006): with a stored key and with an env-var key, run a shaped try and a run through the fake, export the tidy table, the wide file and the methods paragraph, list servers, read entries, check the server, trigger a refused-key error; then search every response body, every file under the results folder, the database file, and all captured log records for the key: none found. Also a config with `api_key:` is refused (`KEY_IN_CONFIG`) and nothing is stored
- [ ] T044 [US3] Viewer: in `ModelServers.tsx` the remote line and, for admins (`useCurrentUser`), the switch with the warning dialog (text from the API, a second click to confirm), disabled with "set by this server's configuration" when fixed; "Confirmed before remote servers needed switching on" on grandfathered rows; Add shows the refusal message and how to switch on. Ask AI's confirmation dialog (spec 027) uses the API's warning text. Tests in `ModelServers.test.tsx`; rebuild `viewer_static`

**Checkpoint**: no frame can reach a remote address on a fresh install; Stories 1–2 unaffected for local servers.

---

## Phase 6: User Story 4 — A short clip per sample (P3)

**Goal**: servers and models that take video get "A short clip" sampling.
**Independent test**: quickstart Story 4 with vLLM and a Qwen-VL model.

- [ ] T045 [US4] Clip spans: `clip_span(t, length, duration)` and `clip_times(t, length, fps, duration)` in `src/videoannotator/pipelines/vlm_annotation/sampling.py`; `read_span(cap, fps_video, duration, start, end, clip_fps)` in `frames.py` reusing `read_sample`'s frame reading; the clip cases from T003 pass in `tests/unit/test_vlm_sampling.py`
- [ ] T046 [US4] Video in the OpenAI client: `chat(..., video=list[bytes], video_fps=float)` sends one `video_url` part `data:video/jpeg;base64,<f1>,<f2>,…` (research R12), with `media_io_kwargs: {video: {fps}}` only when the server accepted it in the probe; 400 naming video → plain error. `probe_video(model)` sends two 32×32 grey JPEG frames with `max_tokens: 1` and returns whether the server accepted it (and whether `media_io_kwargs` was accepted); `check_server(..., model=)` runs the probe and `model_servers` stores the result in `video_models`. Tests in `test_openai_client.py` (part format, probe both ways, no participant frames: the probe's frames are the synthetic ones)
- [ ] T047 [US4] `sampling_mode: clip` in `vlm_pipeline.py` with `clip_length_sec`, `clip_fps`; submission refuses `clip` with `backend: ollama` (`CLIP_NOT_SUPPORTED`); annotations record `clip_start_sec`, `clip_end_sec`, `clip_fps`, `context_frame_numbers`, `frame_times`. Ask AI: `Sampling.clip`, tries read spans, `Answer` gains clip fields; registry metadata gains the keys and the `clip` enum label. Tests in `tests/unit/test_vlm_pipeline_clips.py` and `tests/api/test_ask_clips_api.py`
- [ ] T048 [US4] Viewer: `viewer/src/lib/sampling.ts` clip twin (shared cases in `sampling.test.ts`); `SamplingControls.tsx` "A short clip" with length, offered only when the chosen server's `video_models[model]` is true, else disabled with the reason; ticks show spans (`components/lanes/`); contract fixture with a clip output; rebuild `viewer_static`

---

## Phase 7: User Story 5 — Every setting from the command line (P3)

**Goal**: servers checked and added from the CLI with `--json`; runs configured by YAML.
**Independent test**: quickstart Story 5 from a shell only.

- [ ] T049 [US5] `videoannotator model-servers list|add|check|remove` in `src/videoannotator/cli.py` per contracts/api.md: `--json` on list and check, exit codes 0/2/3/4/5, `--key-env VAR`, hidden prompt only when interactive and neither `--key-env` nor `--no-key` is given; never a key flag
- [ ] T050 [P] [US5] `tests/unit/test_cli_model_servers.py` (Typer runner; the fake server): JSON output shape, exit codes per outcome, `add --key-env` stores the env name only, no prompt with `--no-key`, `job submit --config` with `backend`/`answer_shape` produces the job config, a config with `api_key` is refused

---

## Phase 8: Polish & cross-cutting

- [ ] T051 [P] `tests/integration/test_openai_real_server.py`, marked `real_vlm_server` (add the marker to `pyproject.toml`, excluded by default): with `VA_TEST_OPENAI_BASE_URL` and `VA_TEST_OPENAI_MODEL`, run the demo clip through the server with and without a shape; assert structure, fit rate recorded, provenance
- [ ] T052 [P] Docs: `docs/development/vlm_annotation_pipeline.md` (backends, the deviation note now resolved, shapes, clips), the Ask AI how-to under `docs/usage/` (servers, shapes, Suggest, clips), the export how-to (pivot to one column per field; the wide file), the CLI reference, and a data-protection section (remote switch; confirmations; a proxy on localhost looks local; where keys are kept)
- [ ] T053 [P] `CHANGELOG.md` Unreleased (Added: OpenAI-compatible servers, answer shapes, clips, remote switch; Changed: remote addresses need the switch unless confirmed before); `docs/development/roadmap_v1.7.0.md` Phase 2 ticks
- [ ] T054 Run the full gate: `pytest tests/`, `pytest tests/integration/test_output_baseline.py`, `ruff check .`, `mypy src/videoannotator`, `pre-commit run --all-files`, `cd viewer && bun run lint && bun run test:run && tsc --noEmit -p tsconfig.app.json`, `bash scripts/build_viewer.sh --check`
- [ ] T055 Walk through `quickstart.md` with the servers available, measure SC-001 and SC-003, and write `specs/033-model-connectors/checks.md` (what passed, the measured fit rate, what couldn't be checked without a GPU or a hosted key)

---

## Dependencies

- Phase 1 → Phase 2 → stories. T004 → T006 → T008; T010 before T011.
- **US1** (T011–T025) needs Phase 2. **US2** (T026–T037) needs Phase 2; its viewer tasks need
  T021–T023 (the server picker). **US3** (T038–T044) needs T011, T016. **US4** (T045–T048) needs
  T006, T015. **US5** (T049–T050) needs T015, T016.
- US2, US3 and US4 are independent of each other once US1's server side (T011–T019) is in.

## Parallel examples

- Phase 1: T001, T002, T003 together.
- US1: T013, T017, T019 (tests) while T021 (viewer API) proceeds; T022 and T024 after T021.
- US2: T027 and T033 (shared-case tests, server and viewer) together; T031 and T032 together after T029.
- US3: T042 and T043 together after T039–T041.

## Implementation strategy

1. **MVP = Phases 1–3 (US1)**: a lab's local server works in Ask AI and runs. Ship and try it on
   the walk-through laptop.
2. **US2** next (P1): the biggest gain for every user, including Ollama ones.
3. **US3** before announcing hosted support anywhere; until then remote addresses behave as in
   v1.6 (confirmation only) — note that **between US1 and US3 a remote OpenAI-compatible server
   can be added and confirmed like a remote Ollama one**, so don't tag a release candidate
   without US3.
4. **US4, US5** as time allows before the release candidates; both are P3.
