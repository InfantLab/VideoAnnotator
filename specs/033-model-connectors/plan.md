# Implementation Plan: Model Connectors

**Branch**: `1.7-dev` (spec `033-model-connectors`) | **Date**: 2026-10-22 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/033-model-connectors/spec.md` (clarified 2026-10-22)

## Summary

Ask AI and the `vlm_annotation` pipeline gain a second model backend, `openai_compatible`, so a
lab can use the llama.cpp, LM Studio, vLLM or SGLang server it already runs, or, if the
server's administrator switches remote servers on, a hosted API. Questions can carry an answer
shape, a flat list of fields, which is sent as the server's structured-output request and
checked on our side, so answers become columns. Servers that take video (vLLM with Qwen-VL) get
a third sampling mode, a short clip per sample.

Approach (from [research.md](research.md)): a second client on `httpx` behind a small client
protocol (R1, R2); a `backend` config key defaulting to Ollama (R3); a `model_servers` table,
keys in a 0600 file beside the database or from a named environment variable (R4); one
server-wide remote switch, off by default, with confirmations from before the upgrade kept
(R5); an `answer_shape` config key turned into JSON Schema, sent the server's way and checked by
our own validator (R8–R10); Suggest by rule, then by the model (R11); clips as vLLM's
frame-sequence video part, offered after a synthetic probe (R12).

## Technical Context

**Language/Version**: Python 3.12/3.13 (server); TypeScript, React 18 (viewer, Vite, Bun)
**Primary Dependencies**: FastAPI, SQLAlchemy, Pydantic, httpx, OpenCV (existing); `ollama`
under the `llm` extra (existing, unchanged). No new dependency (FR-009).
**Storage**: server SQLite: new tables `model_servers`, `server_settings`; new columns
`endpoint_confirmations.before_switch`, `ask_entries.backend`, `ask_entries.answer_shape`,
`prompts.answer_shape`; new file `model_server_keys.json` beside the database.
**Testing**: pytest with a fake OpenAI-compatible server (`httpx.MockTransport`, beside the
existing `tests/fixtures/fake_vlm.py`) fed by replies
recorded from real servers (R15); Vitest; one opt-in real-server integration test; quickstart
walk-through.
**Target Platform**: the API server on Linux, macOS, Windows; current Chrome, Firefox, Safari.
**Project Type**: web application (Python API + React viewer in one repo).
**Performance Goals**: no slower than Ollama per call apart from the server itself; Suggest by
rule < 50 ms; listing servers never waits on a server (checks are explicit, cached like
readiness).
**Constraints**: local-first (Principle I): remote off by default, enforced on the server for
every route; keys never in outputs, logs, responses or configs (FR-016, FR-017); existing
configs, outputs, error codes and the tidy table's columns unchanged (Principles II, V); viewer
bundle within 300 KB gzipped (the shape editor joins the lazy Ask AI chunk).
**Scale/Scope**: a handful of servers per lab; shapes of up to 12 fields; runs of a few thousand
samples.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design: still passes.*

| Gate | Status | How |
|---|---|---|
| **I. Local-first, no telemetry** | ✅ | Default stays local Ollama. Remote servers are off on a fresh install, switchable only by an admin and lockable off by environment (R5); every route (Ask AI, preview, jobs from the API or CLI) goes through one server-side check; spec 027's confirmation still applies after the switch. The video probe uses synthetic frames. No new outbound traffic of our own; no SDK with telemetry (R1). |
| **II. Stable pipeline contract, open formats** | ✅ | `BasePipeline` unchanged. Output stays COCO JSON; new keys are additive (`fields`, `fit`, `fit_problem`, clip fields, `vlm_answer_shape`). The tidy table keeps its columns and format version; fields are new measure names (R10). The viewer contract test gains shaped and clip fixtures. |
| **III. Provenance** | ✅ | `backend`, `answer_shape_sha256` and `structured_output` added to the `vlm` block; model refs record what the server reports or `unknown` with a reason (R7). Shapes and clips are declarative config. No key in any record (R4). |
| **IV. Modular** | ✅ | No new dependency: `httpx` is core. The `llm` extra is still what the pipeline requires; the OpenAI client doesn't import `ollama`. Viewer: the shape editor and server picker live in the lazy Ask AI and Settings chunks. |
| **V. Backward compatibility** | ✅ | No `backend` = Ollama, as before. Ollama error codes unchanged. Remote addresses confirmed before the upgrade keep working without the switch (R5). Migrations only add. `/vlm/models` without `server` behaves as today. |
| **VI. Faithful display** | ✅ | A misfit is shown as "Didn't fit" with the raw reply, its own colour in lanes and tallies, never mapped to a value (R8). Clip spans drawn from the same sampling cases as the pipeline. Lanes stay labelled with question, model and server. |
| **Engineering standards** | ✅ | Tests per area (below), SC-006 as a test; typed API; docs: VLM pipeline page, Ask AI how-to (servers, shapes, clips), a data-protection note on remote servers and local proxies, export how-to (pivot), CHANGELOG; viewer `tsc`, ESLint, Vitest, bundle check. |

No violations; Complexity Tracking is empty.

## Project Structure

### Documentation (this feature)

```
specs/033-model-connectors/
├── spec.md              # clarified and amended 2026-10-22
├── plan.md              # this file
├── research.md          # R1–R15
├── data-model.md        # tables, files, shape, answer, provenance, table rows
├── quickstart.md        # walk-through per story
├── contracts/
│   ├── api.md           # /model-servers, /ask changes, suggest-shape, wide export, CLI
│   ├── pipeline-config.md  # backend, answer_shape, clip keys, checks, shared cases
│   └── viewer.md        # Settings card, Ask AI, settings form, Video page, tests
├── checklists/requirements.md
└── tasks.md             # /speckit-tasks (not created here)
```

### Source Code

```
src/videoannotator/
├── pipelines/vlm_annotation/
│   ├── clients.py           # NEW VLMClient protocol, make_client(), KeyRefusedError (R1)
│   ├── openai_client.py     # NEW OpenAICompatibleVLMClient: models, chat, check, video probe (R1, R2, R12, R13)
│   ├── ollama_client.py     # chat(format=schema) for shapes (R8); otherwise unchanged
│   ├── answer_shape.py      # NEW validate, to_json_schema, sha256, check_reply, suggest_by_rule (R8, R11)
│   ├── sampling.py          # clip spans (R12)
│   ├── frames.py            # read frames over a span at an fps (R12)
│   └── vlm_pipeline.py      # backend, answer_shape, clip; submission checks; key refused stops (R3)
├── registry/metadata/vlm_annotation.yaml  # backends, new keys, wording
├── model_servers.py         # NEW list (with default), add/remove, key resolution, keys file (R4)
├── endpoint_policy.py       # NEW remote switch, require_sendable (R5)
├── secrets_store.py         # keys file helpers beside the HF token; log redaction filter (R4)
├── database/models.py       # ModelServer, ServerSetting; new columns
├── database/migrations.py   # additive columns; before_switch = true for existing rows
├── ask_history.py           # backend, answer_shape; library shape
├── api/v1/
│   ├── model_servers.py     # NEW /model-servers (R13, R5)
│   ├── ask.py               # backend/shape/clip; require_sendable; suggest-shape; wide export
│   ├── vlm.py               # models?server=, preview through make_client
│   └── __init__.py          # router
├── exporters/readers.py     # read_vlm: vlm.<field>, vlm_fit rows (R10)
├── exporters/table.py       # wide answers CSV (R10)
├── exporters/methods.py     # unchanged wording; reads backend for the model line
├── provenance.py            # (no change: the vlm block is passed in)
└── cli.py                   # model-servers list/add/check/remove (--json, exit codes)

viewer/src/
├── components/settings/ModelServers.tsx   # NEW card
├── components/ask/ServerPicker.tsx        # NEW
├── components/ask/AnswerShapeEditor.tsx   # NEW (also the settings-form widget)
├── components/ask/AnswersTable.tsx, Tally.tsx, SamplingControls.tsx  # fields, misfits, clip
├── components/VlmModelPicker.tsx          # server-aware words
├── components/lanes/                      # clip spans; misfit colour
├── lib/answerShape.ts                     # NEW validation twin
├── lib/sampling.ts                        # clip spans twin
├── pages/AskAI.tsx, Settings.tsx          # wire in
└── api/client.ts, api/schema.d.ts         # regenerated

tests/
├── fixtures/fake_openai_server.py         # NEW MockTransport fake + recorded replies (R15)
├── fixtures/openai_compatible/*.json      # recorded replies: vLLM, llama.cpp, LM Studio, Ollama /v1
├── fixtures/answer_shape/cases.json       # shared with the viewer
├── unit/test_openai_client.py, test_answer_shape.py, test_model_servers.py,
│   test_endpoint_policy.py, test_vlm_pipeline_backends.py, test_vlm_sampling.py (clip cases)
├── api/test_model_servers_api.py, test_ask_shapes_api.py, test_ask_remote_switch.py
├── unit/test_no_key_leaks.py              # SC-006: run, export, log; search for the key
└── integration/test_openai_real_server.py # -m real_vlm_server, opt-in
```

**Structure Decision**: the existing web-application layout: server code in
`src/videoannotator/`, viewer in `viewer/` built into `viewer_static/` (rebuilt and committed,
CI checks it).

## Order of work

Each stage ships on its own and keeps the suite green. Stages 1–3 are Story 1 (MVP).

1. **Client** (R1, R2, R15): `clients.py`, `openai_client.py`, the fake server and recorded
   replies; `make_client` used by the pipeline, Ask AI tries, `require_model`, `/vlm/models`,
   `/vlm/preview`. `backend` config key and metadata. Provenance `backend` and model ref (R7).
2. **Servers and keys** (R4, R13): `model_servers` table, keys file, key resolution, the
   `KEY_IN_CONFIG` check, the log filter; `/model-servers` list/add/patch/delete/check; CLI.
3. **Viewer, Story 1** (R14): Settings → Model servers; server picker in Ask AI; model picker
   words; settings form `backend` and server picker. Vocabulary test. Rebuild `viewer_static`.
4. **Remote switch** (R5, Story 3): `server_settings`, `before_switch` migration,
   `endpoint_policy.require_sendable` everywhere `require_confirmed` is and in `initialize()`;
   admin switch in Settings; `KeyRefusedError` stops runs; `test_no_key_leaks.py`.
5. **Answer shapes** (R8–R11, Story 2): `answer_shape.py` with shared cases; both clients send
   it; pipeline and tries check replies; entries and library carry it; readers' rows; wide CSV;
   suggest-shape; the editor, answers, lanes and tally in the viewer; settings-form widget.
6. **Clips** (R12, Story 4): sampling and frames over a span; the probe; `sampling_mode: clip`;
   viewer option and spans.
7. **Docs and checks**: VLM pipeline page (backends, shapes, clips), Ask AI how-to, Settings
   help, data protection note (remote switch; a localhost proxy looks local), export how-to
   (pivot), CLI reference, CHANGELOG, roadmap Phase 2 ticks; `checks.md` from the quickstart.

## Risks

- **Server differences are wider than the table in research** (reasoning field names, schema
  form, video form). Mitigation: recorded replies from each real server in the fixtures before
  the client is finished; the fallback to describing the shape in the prompt.
- **Small local models misfit shapes.** That is a result (SC-003 measures it on the default
  model); misfits are recorded, not hidden. If the default model is below 95%, the SC is
  revisited with the measurement rather than the validator loosened.
- **vLLM's per-request fps needs a server flag.** Without `--trust-request-mm-kwargs` the model
  gets no real timing; the check records it and the quickstart documents the flag.

## Complexity Tracking

None.
