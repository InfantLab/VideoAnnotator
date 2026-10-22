# Feature Specification: Model Connectors

**Feature Branch**: `033-model-connectors` (work lands on `1.7-dev`)
**Created**: 2026-10-22
**Status**: Planned 2026-10-22 ([plan](plan.md))
**Roadmap**: [v1.7.0 Phase 2](../../docs/development/roadmap_v1.7.0.md) (connectors)
**Builds on**: [spec 027](../027-ask-ai/spec.md) (Ask AI, the local/remote badge, confirming
before a remote send), [spec 009](../009-vlm-prompt-workflow/spec.md) (the `vlm_annotation` pipeline),
[spec 017](../017-output-provenance/spec.md) (provenance), [spec 025](../025-results-out/spec.md) (the tidy
export)
**Input**: User description: "OpenAI-compatible model connector for Ask AI and the vlm_annotation
pipeline. Add a second backend, `openai_compatible`, speaking the /v1/chat/completions contract,
so one client covers llama.cpp server, vLLM, LM Studio, SGLang, Ollama's own /v1 endpoint, and
hosted APIs. The local/remote badge and the confirmation before a first remote send apply
unchanged, decided by address. Hosted endpoints are off by default, with a plain warning that
frames leave the machine. API keys never appear in outputs, provenance or logs. Structured
outputs: a question can carry a JSON schema, so answers come back parseable; Ask AI helps build
it; answers that fail it are marked, not dropped; the tidy export gets one column per field. Video
input through vLLM (Qwen-VL) as a third sampling mode. Provenance records backend, endpoint class,
model, what the server reports about it, and the schema's hash. Backward compatible."

## Context

Ask AI and the `vlm_annotation` pipeline talk to one kind of model server: Ollama, through its
own Python package. That covers the default setup, a model on this computer, but not what labs
already run or ask about:

- a **llama.cpp** or **LM Studio** server on a lab workstation,
- **vLLM** or **SGLang** on a GPU server, the fast way to run hundreds of videos,
- a **hosted API** (OpenAI, Google, Anthropic or others) for a lab whose ethics approval allows
  it, or for videos that aren't of participants.

All of these speak one contract, OpenAI-style chat completions, and Ollama speaks it too. One
connector covers them all.

Two things become possible with it, and researchers have asked for both:

- **Answers in a fixed shape.** Today the answer is free text and the label is found by a pattern
  (`TOUCH`, `NO_TOUCH`, `YES`, `NO`), and otherwise it's the first 200 characters. A question that
  says what the answer must look like ("touch: yes or no; who touches: mother, infant, both,
  none; confidence: 1–5") gets answers that go straight into a table, one column each. Ollama
  supports this too, so it isn't tied to the new connector.
- **Video, not stills.** Some servers (vLLM with Qwen-VL models) take a short clip, so a model sees
  movement rather than a few frames side by side. Ollama has no video input as of September 2026.

What doesn't change: the badge says where the model runs, decided by address on the server
(spec 027 R9); a first send to a remote address asks first; every sample is one fresh call with no
history; Ollama setups, saved settings and outputs keep working as they are (Principle V).

### What the researcher sees

```
┌ Ask AI · p01_t1.mp4                     Local · qwen3-vl-8b on lab-gpu ┐
│ Model server: [lab-gpu (vLLM) ▾]   Model: [qwen3-vl-8b ▾]              │
│ Question                                                               │
│ ┌────────────────────────────────────────────────────────────────────┐ │
│ │ Is the mother touching the infant? Who initiates?                  │ │
│ └────────────────────────────────────────────────────────────────────┘ │
│ Answer shape   ○ Free text  ● Fields                     [Suggest ✦]   │
│   touch        one of  yes · no                                        │
│   initiator    one of  mother · infant · both · none                   │
│   + field                                                              │
│ Frames per sample: ○ One  ○ Several  ● A short clip (2 s)              │
└────────────────────────────────────────────────────────────────────────┘
```

Settings → Model servers lists the servers the viewer knows: the default Ollama one, and any
added, each with its kind, address, badge and whether it needs a key.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Use the model server the lab already runs (Priority: P1)

A lab runs vLLM on a GPU workstation on its network. A researcher opens Settings → Model servers,
adds it by address, picks "OpenAI-compatible", and checks it. The viewer lists the server's
models. Back in Ask AI they choose it, ask a question, and get answers as they did with Ollama:
the badge says Local, the history keeps every try, and a run on a folder uses the same server.

**Why this priority**: The connector itself. Everything else in this spec depends on it, and on
its own it opens llama.cpp, LM Studio, vLLM and SGLang to every lab.

**Independent Test**: With a llama.cpp or vLLM server on the local network and Ollama stopped, add
the server, get a first Ask AI answer, then run `vlm_annotation` on two videos through it. The
outputs read like an Ollama run's, and the provenance names the backend.

**Acceptance Scenarios**:

1. **Given** an OpenAI-compatible server at a local-network address, **When** the researcher adds
   it and checks it, **Then** the viewer shows "reachable", lists its models, and the badge says
   Local with the host's name.
2. **Given** that server is chosen in Ask AI, **When** the researcher tries a question, **Then**
   the answer, the model's reasoning (if it returns any), timings and token counts appear as they
   do with Ollama.
3. **Given** a run started from Ask AI on a folder, **When** it finishes, **Then** every video's
   output file has the same structure as an Ollama run's and records `openai_compatible` as the
   backend.
4. **Given** an address that doesn't answer, or answers but isn't an OpenAI-compatible server,
   **When** the researcher checks it, **Then** they see which of the two it is and what to try,
   in plain words.
5. **Given** an existing Ollama setup with saved questions, workflows and project settings,
   **When** the server is upgraded, **Then** all of them work unchanged and still use Ollama.
6. **Given** Ollama itself, **When** the researcher adds it as an OpenAI-compatible server (its
   `/v1` address), **Then** it works too; the Ollama backend stays the default for Ollama because
   it reports model digests.

---

### User Story 2 - Answers in a fixed shape (Priority: P1)

A researcher asks "Is the mother touching the infant? Who initiates?" and, under Answer shape,
chooses Fields: `touch` (one of yes, no) and `initiator` (one of mother, infant, both, none). Or
they press Suggest, and the viewer proposes those fields from the question for them to edit. Every
answer now comes back as those fields. The answers file has a `touch` column and an `initiator`
column. An answer that doesn't fit the shape is marked as not fitting, with the model's raw reply
kept, and is never silently dropped or guessed at.

**Why this priority**: Journey 1's lesson is that the question defines the answer. Free-text
answers that need a pattern to read are the main reason Ask AI output is hard to analyse. This
works with Ollama too, so it helps every user, not only those with a new server.

**Independent Test**: With the default Ollama model, define two fields, try the question on three
moments, run it on one video, and download the tidy table. It has one column per field, and every
row is either filled or marked as not fitting.

**Acceptance Scenarios**:

1. **Given** a question with fields, **When** a try or run sends it, **Then** the model server is
   asked for answers in that shape (by the server's own structured-output mechanism where it has
   one), and the answer is checked against the shape on our side as well.
2. **Given** an answer that fits, **When** it is shown, **Then** Ask AI shows the fields as a
   small table, and the timeline lane colours by the first field.
3. **Given** an answer that doesn't fit (not JSON, a value outside the list, a field missing),
   **When** it is recorded, **Then** it is marked "didn't fit the answer shape", with what was
   wrong, and the raw reply is kept. It counts as answered, not as an error, and isn't retried.
4. **Given** a run with fields, **When** the answers file is downloaded, **Then** it has one
   column per field, in the order defined, with empty cells and a `fit` column for answers that
   didn't fit; the tidy table has one measure per field.
5. **Given** a question without fields, **When** it runs, **Then** behaviour and output are exactly
   as today (the label pattern, the raw reply).
6. **Given** a saved question with fields, **When** it is reopened from the history or library, in
   a workflow, or in a project's remembered settings, **Then** the fields come with it.
7. **Given** Suggest, **When** the researcher presses it, **Then** a proposal appears for them to
   edit or discard. It is never applied without them seeing it, and it makes no model call on
   video frames.
8. **Given** a server that has no structured-output support, **When** a question with fields is
   sent, **Then** the shape is described in the prompt instead, the check on our side still
   applies, and Ask AI says that the server can't enforce the shape.

---

### User Story 3 - A hosted model, knowingly (Priority: P2)

A lab whose ethics approval covers a hosted API, or a researcher working on videos that aren't of
participants, wants to use one. Hosted models are off until the server's administrator switches
them on. When they are on, adding one takes an address and a key. Before the first frames go to
it, the viewer says plainly that frames from the video will be sent to that company's servers,
and that for participants' video this is usually a question for the ethics approval. The badge
then says Remote, in the warning colour, wherever the model is used.

**Why this priority**: Some labs need it and all of them need to be protected from it by default.
It comes after Stories 1 and 2 because most labs will use local servers, and because 027's
confirmation already stops an accidental remote send.

**Independent Test**: On a fresh install, try to add a hosted endpoint and find it refused with the
reason and how to enable it. Enable it, add one with a key, try a question, and find the key in no
output, provenance record, log line, export or API response.

**Acceptance Scenarios**:

1. **Given** a fresh install, **When** a user adds a server whose address is remote, **Then** it is
   refused with "Remote model servers are switched off on this server" and how to switch them on.
2. **Given** remote servers are switched on, **When** a user first sends frames to one, **Then**
   the spec 027 confirmation appears with the plain warning about where frames go, and nothing is
   sent until it is confirmed.
3. **Given** a server with a key, **When** any output, provenance record, methods paragraph,
   export, log, error message or API response is produced, **Then** the key appears in none of
   them; the server is identified by its address and host.
4. **Given** a remote server in a run, **When** the methods paragraph is generated, **Then** it
   says the model ran on a remote server; the provenance record names the host.
5. **Given** a key that is wrong or expired, **When** a try or run uses it, **Then** the user sees
   "the server refused the key" and the run stops at once rather than failing every sample.
6. **Given** a remote Ollama address confirmed before the upgrade, **When** the server is upgraded
   with remote servers still switched off, **Then** that address keeps working, and Settings shows
   it as confirmed before the switch existed.

---

### User Story 4 - A short clip per sample (Priority: P3)

A researcher studying movement chooses "A short clip" under Frames per sample, with a length of
2 seconds. Each sample point now sends the clip around that moment, and the model's answer is
about what happens in it. The timeline shows the clip's span around each tick, as it shows a
burst's frames today.

**Why this priority**: Valuable for movement and interaction questions, but only some servers
(vLLM with Qwen-VL models) take video, so it serves fewer labs than Stories 1–3. It is
independent of them apart from needing the connector.

**Independent Test**: With vLLM serving a Qwen-VL model, run one video with clip sampling. Each
answer records the clip's start and end, and the clip mode isn't offered for a server or model
that can't take video.

**Acceptance Scenarios**:

1. **Given** a server and model that take video, **When** the researcher chooses a clip, **Then**
   each sample sends the clip from (t − length/2) to (t + length/2), clipped to the video's ends.
2. **Given** a server or model that doesn't take video, **When** Frames per sample is shown,
   **Then** the clip option is not offered, with a one-line reason.
3. **Given** a clip run, **When** an answer is recorded, **Then** it records the clip's start, end
   and the frame rate it was sent at.
4. **Given** a clip run, **When** the sampling is shown on the timeline before any model call,
   **Then** each tick shows the clip's span, as bursts do today.

---

### User Story 5 - Every setting from the command line (Priority: P3)

A researcher who scripts their analysis, or works with an agent, sets the backend, server, model
and answer shape in a run's config or on the command line, and checks a server with one command,
with output as JSON.

**Why this priority**: The roadmap's success criterion that everything the viewer does is
available from the CLI. It costs little once Stories 1–2 exist.

**Independent Test**: From a shell only, check a vLLM server, submit a run with an answer shape on
one video, and read the fields from the output.

**Acceptance Scenarios**:

1. **Given** a config with `backend: openai_compatible`, a base URL and fields, **When** it is
   submitted, **Then** it runs as the same settings would from the viewer.
2. **Given** a server-check command with `--json`, **When** it runs, **Then** it prints
   reachability, the kind of server, its models, local or remote, and whether it accepts
   structured outputs and video where that can be told.
3. **Given** a key, **When** it is given to the CLI, **Then** it comes from an environment variable
   or the server's stored setting, never a flag that would land in shell history or a saved config.

### Edge Cases

- A server that lists no models (some llama.cpp builds serve one model and report it oddly): the
  user can type the model name; the check says what the server reported.
- A server that answers `/v1/models` but not chat with images (a text-only model): the first try
  says the model doesn't accept images, and names the model.
- Two names for one server (`localhost` and `127.0.0.1`): treated as one for the badge and the
  confirmation, as spec 027 already normalises them. The gate that puts tries ahead of runs
  (spec 027 FR-017a) is per normalised address and model, whatever the backend.
- A schema field whose name isn't a valid column name, or clashes with an existing column
  (`label`, `t`): the shape editor refuses it with the reason.
- A very long or deeply nested shape: limited to a flat list of fields (Assumptions), each a
  choice from a list, yes/no, a number with optional range, or short text.
- A model that wraps JSON in prose or a code fence: the check takes the first JSON object in the
  reply; it still counts as fitting if that object fits, and the raw reply is kept.
- A thinking model whose reasoning comes in a separate field (`reasoning_content`): recorded as
  reasoning, as Ollama's `thinking` is today.
- A remote address switched off after runs used it: earlier outputs stay readable; new tries and
  runs to it are refused with the reason.
- A server that is local by address but forwards to a hosted API (a proxy on localhost): the badge
  can't know this and says Local. The docs say so plainly (Assumptions).
- A key stored on the server and the server's database copied to another machine: the key travels
  with it. Settings says where keys are stored.
- A clip sample near the start or end of the video: shortened, not shifted, and the recorded span
  says so.
- A run whose server goes away mid-run: as today, consecutive failures stop it with what it has,
  and the outputs say why.

## Requirements *(mandatory)*

### Functional Requirements

**Connector**

- **FR-001**: The `vlm_annotation` pipeline and Ask AI MUST support a second backend,
  `openai_compatible`, beside `ollama`, sending each sample as one stateless chat-completions call
  with the question and the sample's images (or clip), and no history.
- **FR-002**: The backend MUST work against llama.cpp server, vLLM, LM Studio, SGLang and Ollama's
  `/v1` endpoint without per-server code paths in the pipeline. Differences between servers are
  handled by the connector reading what each server reports, not by the user choosing a server
  brand.
- **FR-003**: `ollama` MUST remain the default backend. A config without a `backend` setting MUST
  behave exactly as today, and existing outputs, questions, workflows and project settings MUST
  keep working unchanged.
- **FR-004**: The pipeline metadata MUST list both backends (`backends: [ollama,
  openai_compatible]`), and the settings form MUST show the backend as a setting with Ollama as
  the default.
- **FR-005**: Checking a server MUST report, separately: not reachable; reachable but not the
  chosen kind of server; reachable with no models; reachable with models (listed). Each gets a
  plain message and a hint.
- **FR-006**: Retries, backoff, timeouts, stopping a run, answers as they come, and stopping after
  consecutive failures MUST behave as they do for Ollama. A refused key (FR-016) stops the run at
  once.
- **FR-007**: The model's separate reasoning, timings and token counts MUST be recorded where the
  server returns them, in the same fields Ollama's are, and left empty (not zero-filled as if
  measured) where it doesn't.
- **FR-008**: Settings → Model servers MUST list the known servers (the default Ollama one and any
  added), each with its kind, address, badge, and whether it has a key. Any signed-in user may add
  a local server; remote ones follow FR-013.
- **FR-009**: The connector MUST add no new required dependency.

**Where frames go**

- **FR-010**: The badge and the confirmation before a first remote send (spec 027) MUST apply to
  every backend unchanged, decided on the server by the address frames are sent to.
- **FR-011**: Remote model servers MUST be switched off on a fresh install, by one server-wide
  setting that the server's administrator switches on (server administration, not a per-user
  step). A remote address already confirmed before the upgrade (spec 027) MUST keep working
  without the switch, so no existing setup breaks (Principle V).
- **FR-012**: Switching remote servers on and confirming a remote server MUST each show a plain
  warning: frames from the video will be sent to that host, and for participants' video this is
  usually a question for the ethics approval.
- **FR-013**: A remote server MUST NOT receive any frame until remote servers are switched on and
  that server has been confirmed (spec 027).
- **FR-014**: The methods paragraph MUST say whether the model ran on this computer, on the local
  network or on a remote server, without naming the server (spec 017 FR-029); the host is in the
  provenance record.

**Keys**

- **FR-015**: A server MAY have an API key, from either of two places: an environment variable
  on the server, named in that server's settings, which wins when set; or a key entered in
  Settings → Model servers, stored on the server and never shown again (only "set" or "not
  set"). Settings says where a stored key is kept and that it travels with the server's
  database.
- **FR-016**: A key MUST NOT appear in any output file, provenance record, methods paragraph,
  export, log line, error message, API response, saved workflow, project setting or Ask AI
  history. A response that refers to a key says only whether one is set.
- **FR-017**: Keys MUST NOT be accepted as a command-line flag or in a run's config file.

**Answer shape (structured outputs)**

- **FR-018**: A question MAY carry an answer shape: a flat, ordered list of named fields, each one
  of: a choice from a list, yes/no, a number (optionally with a range), or short text.
- **FR-019**: When a question has a shape, every call MUST ask the server for answers in that
  shape by the server's own structured-output mechanism where it has one (both Ollama and the
  OpenAI-compatible servers named in FR-002 do), and otherwise describe the shape in the prompt.
- **FR-020**: Every answer to a shaped question MUST be checked against the shape on our side. An
  answer that fits is recorded field by field; one that doesn't is recorded as not fitting, with
  what was wrong and the raw reply, counts as answered, and is not retried.
- **FR-021**: A question without a shape MUST behave and be recorded exactly as today.
- **FR-022**: The shape MUST travel with the question everywhere a question goes: tries, the
  history, the prompt library, workflows, project settings, runs, and their outputs.
- **FR-023**: The tidy export MUST give a shaped answer one measure per field (the fixed-column
  table of spec 025, so one pivot gives a column per field) plus whether it fit; and a shaped run
  or Ask AI entry MUST also offer a wide file with one row per moment and one column per field,
  in the shape's order, plus `fit`. Exports of unshaped runs are unchanged.
- **FR-024**: Ask AI MUST offer to suggest a shape from the question's text: first by rule (the
  options a question already lists, as in "Return ONLY: TOUCH or NO_TOUCH" or "one of …"), and
  when the rule finds nothing, by asking the chosen model with the question's text only, never
  frames. A suggestion is always shown for editing before use, and says which way it was made.
- **FR-025**: The Video page lane for a shaped question MUST colour by the first field, and the
  answer's fields MUST be shown wherever the answer is (Ask AI, the data table).

**Video input**

- **FR-026**: For servers and models that take video, Frames per sample MUST offer a short clip
  around each sample point, with a length setting, as a third mode beside one frame and several.
- **FR-027**: The clip mode MUST NOT be offered where the server or model can't take video, with a
  one-line reason; where this can't be told in advance, the first try says so plainly if the
  server refuses.
- **FR-028**: Each clip answer MUST record the clip's start, end and frame rate as sent, and the
  timeline MUST show the clip's span per tick before any call, as for bursts.

**Provenance and CLI**

- **FR-029**: Provenance MUST record, per run: the backend, the address and its class (local or
  remote) and host, the model as named and as the server reports it, a digest or fingerprint where
  the server gives one (marked "unknown" with the reason otherwise), the sampling mode, and the
  hash of the answer shape when there is one.
- **FR-030**: Every backend, server, model, shape and sampling setting MUST be settable in a run's
  config and on the command line, and a server check MUST be available from the CLI with `--json`.
- **FR-031**: The viewer MUST NOT show the words job, batch, dataset or preset on any new screen
  (spec 029 SC-002), and server kinds are named as users know them ("Ollama",
  "OpenAI-compatible (llama.cpp, vLLM, LM Studio…)").

### Key Entities

- **Model server**: an address, a kind (Ollama or OpenAI-compatible), an optional key (from an
  environment variable or stored, FR-015), whether it is local or remote (decided by address), and
  whether it has been confirmed (spec 027).
- **Answer shape**: an ordered list of fields (name, kind, allowed values or range), attached to a
  question; identified in provenance by its hash.
- **Answer**: as today (raw reply, reasoning, timings, label), plus, for a shaped question, its
  fields and whether it fit, with what didn't.
- **Clip sample**: a sample point's clip start, end and frame rate as sent.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A researcher with a llama.cpp, LM Studio or vLLM server on their network gets a first
  Ask AI answer through it within 3 minutes of opening Settings, without documentation.
- **SC-002**: Every existing Ollama setup, saved question, workflow and project setting works
  unchanged after the upgrade (the existing test suite and the demo-clip baseline pass untouched).
- **SC-003**: On the demo clip with a shaped yes/no question and the default local model, at least
  95% of answers fit the shape, and every answer is either filled in or marked as not fitting.
- **SC-004**: A shaped run's answers file opens in R or Python with one column per field, and the
  tidy table pivots to the same, with no parsing of the model's text.
- **SC-005**: On a fresh install, no frame can reach a remote address: 100% of attempts are refused
  until remote servers are switched on and the server is confirmed.
- **SC-006**: A key set on a server appears in none of the run's files, the logs, the exports or
  any API response (checked by searching all of them for the key).
- **SC-007**: Shown the Model servers list and the badge, users correctly say for each server
  whether their video leaves the computer (target: all test users, as spec 027 SC-005).

## Assumptions

- **One contract**: "OpenAI-compatible" means the chat-completions request with image content
  parts and the models list. Servers differ in details (how they report reasoning, which
  structured-output form they accept, whether they take video); the connector handles those by
  what the server reports and falls back gracefully (FR-019, FR-027).
- **Ollama stays as it is**: the `ollama` backend keeps using the Ollama package because it reports
  model digests, which OpenAI-compatible servers don't. Ollama's `/v1` endpoint working through
  the new backend is a bonus, not a migration.
- **Local by address**: as in spec 027, the badge can't see past the address. A local proxy to a
  hosted API looks Local. The data-protection docs (roadmap Phase 3) say so.
- **Flat shapes**: fields are flat (no nesting, no lists of objects) and kept to a handful, since
  each becomes a column and small local models follow small schemas best. Nested shapes can come
  later without changing stored ones.
- **Video input**: vLLM's Qwen-VL support is the target. Whether and how SGLang, llama.cpp or
  hosted APIs take video is checked at planning (research) and the clip mode is offered only where
  it works (FR-027). The clip is cut from the video on our server, so frames never come from
  anywhere but the video the user chose.
- **Determinism**: temperature 0 and, where the server takes one, a fixed seed, as close to
  Ollama's settings as each server allows; provenance records what was sent.
- **No admin friction** (2026-10-19): any signed-in user may add a local server and use any
  server. Switching remote servers on is server administration (FR-011).
- **Out of scope**: choosing a default hosted provider or recommending one; paying for or
  metering hosted calls; running models in-process (no transformers backend); remote job
  dispatch (v1.8+); nested answer shapes; audio input.

## Clarifications

### Session 2026-10-22 (Caspar)

- Q: Remote servers off by default: what does "off" cover, and who switches it on? → A: one
  server-wide switch, off on a fresh install, switched on by the administrator; remote addresses
  confirmed before the upgrade keep working (FR-011).
- Q: Where do keys live? → A: both: an environment variable on the server wins when set,
  otherwise a key entered in Settings and stored on the server (FR-015).
- Q: How does Suggest work? → A: by rule first, and by asking the chosen model (text only) when
  the rule finds nothing (FR-024).

### Amended at planning, 2026-10-22

- FR-014: the methods paragraph keeps spec 017's rule of not naming the server (research R7).
- FR-023, SC-004: the tidy table stays fixed-column, one measure per field; a wide answers file
  gives one column per field (research R10).
