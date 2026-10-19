# Research: Ask AI

**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Date**: 2026-10-18

What the code does today, and the decisions this plan takes. Each decision: what was chosen, why,
and what else was considered.

## What exists (read 2026-10-18)

- **Pipeline** (`pipelines/vlm_annotation/vlm_pipeline.py`): sample points every
  `frame_interval_sec` from `start_time` to `end_time` (the job runner always passes the whole
  video); a burst frame sits at `anchor + offset × frame_interval_sec`; answers are collected in
  memory and written once, as COCO JSON, at the end. A run that stops early raises, so its answers
  are kept on disk for diagnosis but the run is FAILED.
- **Preview** (`api/v1/vlm.py`, `POST /api/v1/vlm/preview`, spec 009/020): one frame or burst
  from a server-side `video_path`, answered synchronously, recorded in the prompt library as a
  `preview` use. The handler is `async def` but calls cv2 and Ollama synchronously, so **every
  preview blocks the server's event loop** for as long as the model takes (seconds to minutes):
  no other request, including the viewer's polling, is served meanwhile.
- **Prompt library** (`prompt_library.py`, tables `prompts`, `prompt_uses`, spec 020): one row per
  exact prompt text (SHA-256), name, tags, star, hidden; one use per job or preview, with model
  and user. Shared by everyone on the server. No answers, no sampling, no video.
- **Workbench** (`viewer/src/pages/Workbench.tsx`): prompts × models × moments, rounds kept in
  React state only (lost on reload). **Prompts** page lists the library. **VlmPromptTestPanel** is
  the wizard's test panel. All three call `/vlm/preview`.
- **Jobs** run in the API server's own thread pool (`api/background_tasks.py`,
  `run_in_executor`), up to `MAX_CONCURRENT_JOBS` (2) at once; `batch_orchestrator` runs them
  in a CLI worker process instead. Cancellation is a persisted status, checked between
  pipelines only (`batch/job_execution.py`).
- **Live updates**: `GET /api/v1/events/stream` (SSE) polls storage every 2 s and emits job
  status/progress changes. No per-answer events.
- **Video**: `GET /jobs/{id}/video` serves a job's video with range requests (spec 021) but needs
  the `Authorization` header, which a `<video>` element can't send; the viewer fetches the whole
  file as a blob. Uploaded videos are stored per job by the storage provider. Server folders are
  reached through `ingest.resolve_within_roots` and `GET /ingest/browse` (spec 022).
- **Table export** (spec 025, `exporters/table.py`): rows from result files through
  `readers.read_vlm`; per job, batch or dataset.
- **Endpoint locality**: the Workbench guesses "remote" from a hard-coded host list
  (`localhost`, `127.0.0.1`, `host.docker.internal`, …) in the browser.

## Decisions

### R1. Sampling settings are additive config keys on `vlm_annotation`

**Decision**: four new optional keys, all defaulting to today's behaviour (FR-010, FR-013):

| Key | Default | Meaning |
|---|---|---|
| `start_sec` | `0.0` | first sample point |
| `end_sec` | `null` (end of video) | last possible sample point |
| `moments_sec` | `null` | listed moments; when set, replaces the interval grid |
| `burst_spacing_sec` | `null` (= `frame_interval_sec`) | seconds between burst frames |

"X frames, Y s apart" in the window maps to `sampling_mode: frame_burst`, `burst_offsets` of X
consecutive integers centred on 0 (an even X has one more before the anchor than after:
4 → `[-2, -1, 0, 1]`), and `burst_spacing_sec: Y`. One frame maps to `single_frame`.

**Rationale**: Principle V. Every existing config means what it did: with `burst_spacing_sec`
unset, a burst frame is still at `anchor + offset × frame_interval_sec`. Settings stay declarative
and are recorded with the run (Principle III). Range lives in config, not in the job runner's
`process()` arguments, so `job_execution.py` needs no special case and the range is in the
provenance record's `settings`.

**Alternatives**: passing start/end through `process()` from the job (needs a per-pipeline
special case in the runner and isn't recorded as a setting); replacing `burst_offsets` with
"frames per sample" (breaks existing configs).

### R2. One sample-point function, mirrored in the viewer and pinned by shared cases

**Decision**: `sample_points(duration, interval, start, end, moments)` and
`burst_times(anchor, offsets, spacing, duration)` become module-level functions in the pipeline
package, used by runs, tries and the frames endpoint. The viewer has a TypeScript twin for the
ticks and count (FR-011, SC-003: no server call). Both are tested against one JSON file of cases
(`tests/fixtures/ask_ai/sampling_cases.json`), read by pytest and Vitest.

**Rationale**: the ticks must fall exactly where the run will sample (Principle VI: the window
must not show sample points the pipeline won't use). A round trip per keystroke would miss
SC-003.

**Alternatives**: an API call per change (latency, load); trusting two implementations without a
shared fixture (they drift: today's `_sample_points` and the Workbench's `evenlySpaced` already
differ).

### R3. "Try on 10 moments" picks every k-th sample point

**Decision**: from the clarification: of the S sample points of the current settings, take
indices `round(i × (S − 1) / (N − 1))` for i in 0…N−1 (all of them when S ≤ N; the middle one
when N = 1). Part of the shared sampling function and cases.

### R4. Tries are server-side records, answered in a worker thread

**Decision**: `POST /api/v1/ask/tries` creates a history entry (`kind: try`, `status: running`),
returns at once, and answers its moments one by one in a server thread, saving each answer to the
entry as it lands. The viewer polls `GET /api/v1/ask/entries/{id}` every second while the entry is
running. "Try on this frame" is a try with one moment at the playhead (not snapped to the grid).

**Rationale**: SC-002 (no try is ever lost) holds even if the browser closes mid-try, because the
record is written before the first model call. Running off the event loop fixes the blocking
found above. Answers appear one by one for a ten-moment try instead of after the last.

**Alternatives**: keep calling `/vlm/preview` per moment from the browser and post the results to
a history endpoint afterwards (a closed tab loses the try; two requests per answer); SSE per try
(a second live channel for what one-second polling does well enough at this scale).

`/vlm/preview` stays (Principle V; the CLI and scripts may use it), is moved off the event loop
(`run_in_threadpool`), and accepts `burst_spacing_sec`. It no longer has a viewer caller.

### R5. A try goes ahead of a run: an in-process priority gate per model

**Decision**: `pipelines/vlm_annotation/gate.py` holds one gate per `(base_url, model)`. Every
model call takes the gate: runs at low priority, tries at high. When a try is waiting, a run's
next sample waits until no try is waiting. The gate never interrupts a call in progress, so a try
waits at most one sample's answer (FR-017a).

**Rationale**: jobs started from the API run in the same process as the tries (the default and
the only Ask AI path), so a threading condition is enough; no new service. A model server that
answers several requests at once (Ollama with `OLLAMA_NUM_PARALLEL` > 1) is still respected: the
gate only orders, it doesn't cap concurrency beyond one waiting run sample.

**Limitation (documented in quickstart)**: a job run by a separate CLI worker process doesn't
share the gate; a try then queues behind it at the model server, as today.

**Alternatives**: a server-wide queue with priorities (a scheduler for one feature); pausing the
job through its persisted status (too slow, polled between pipelines).

### R6. Answers as they come: the results file is rewritten after each answer

**Decision**: the pipeline rewrites its COCO file atomically (write to a temporary name, then
replace) after each answer, at most once a second, plus once at the end. A new endpoint
`GET /api/v1/ask/runs/{job_id}/answers?after=N` reads it and returns tidy answers and progress;
the viewer polls it every 2 s while the run is going (SC-004: within 5 s).

**Rationale**: no new file format (Principle II): the file a finished run leaves is the one
readers, the table export and spec 028 already read. A stopped or crashed run keeps every answer
so far, on disk, in the normal place. A few hundred annotations rewritten once a second is cheap.

**Alternatives**: an append-only JSONL beside the COCO file (a second format for the same data);
answers in the job's database record (large JSON blobs in the jobs table; the results folder is
the record researchers keep, spec 022); SSE events from the executor thread (no push channel
exists from there; spec 008 chose poll-and-diff for the same reason).

### R7. Stop keeps the answers: a stop check between samples

**Decision**: a pipeline may expose `set_stop_check(callable)`. `job_execution` passes a check that
re-reads the job's persisted status (the existing `_cancellation_requested`). The VLM pipeline
calls it before each sample; when it returns true, the pipeline writes the answers so far, records
`stopped_after` in its run summary, and returns normally instead of raising. The job settles as
CANCELLED (an existing status the viewer already shows as not failed) with a COMPLETED
`vlm_annotation` result, so export, provenance and the results folder treat it as a normal run
(FR-018, FR-019). Ask AI's Stop calls the existing `POST /jobs/{id}/cancel`.

**Rationale**: no change to `BasePipeline` (Principle II): the hook is optional and duck-typed,
like `provenance_vlm`. Cancellation stays a persisted status, visible across processes.

**Check during planning tasks**: the table export and results listing must include a CANCELLED
job's completed results (they may filter on COMPLETED today).

**Alternatives**: a new STOPPED job status (touches every status consumer, the viewer's badges,
the CLI and the database for a distinction "Stopped after 23 of 40" makes in words).

### R8. Provenance records sampling, locality and how far the run got

**Decision**: additive fields in the record's `vlm` block (spec 017): `endpoint_class`
(`local` | `remote`), `endpoint_host`, and after the run `samples_planned`, `samples_answered`,
`stopped_early`. Sampling settings are already in `settings`. The methods paragraph (spec 025)
says "on a model running on this computer / on <host>" (FR-016, Principle III).

### R9. Local or remote is decided on the server, by address

**Decision**: `classify_endpoint(base_url)` resolves the host and returns `local` when every
address is loopback, RFC 1918 (`10/8`, `172.16/12`, `192.168/16`), IPv6 unique-local (`fc00::/7`)
or link-local; otherwise `remote`. A host that doesn't resolve is `remote` (warn, don't guess).
Explicit network list, not `ipaddress.is_private`, whose meaning changed in Python 3.13 for some
ranges. `GET /vlm/models` returns `endpoint: {host, class, confirmed}`; the badge shows it.

**Rationale**: the browser can't resolve names and today's host list misses lab servers; the
server is where the frames are sent from. Holds unchanged for the OpenAI-compatible connector
(spec Assumptions).

**Open, not blocking**: VPN ranges such as Tailscale's `100.64.0.0/10` are remote under this rule.
Decide with the connector (roadmap v1.7.0 Phase 2) whether a setting may mark a host as local.

### R10. Confirming a remote endpoint is remembered on the server

**Decision**: table `endpoint_confirmations` (base URL, who, when). A try or run to a remote
endpoint without a confirmation is refused with `409 REMOTE_NOT_CONFIRMED`;
`POST /api/v1/ask/endpoints/confirm` records it (FR-015). Shared by everyone on the server,
like the history.

**Rationale**: enforced where the frames leave, not only in the window; a second browser or a
colleague can't skip it unknowingly. Today every endpoint is local Ollama, so nobody sees it until
a remote one is configured.

### R11. The history is a new table that points at the prompt library

**Decision**: table `ask_entries` (see [data-model.md](data-model.md)): one row per try or run,
with the prompt's SHA-256 (the prompt text stays in `prompts`, once), model, endpoint, sampling,
video, moments, answers (tries only), who, when, name, star. A run's entry holds its `job_id`;
its answers are read from the run's results. `GET /api/v1/ask/entries` lists entries and, below
them, library prompts with no entry yet ("from the prompt library", restorable as question and
model), so FR-006's "the existing prompt library's records appear" holds without migrating them.

**Rationale**: the library is keyed by text and has no place for sampling, video or answers;
adding them there would make one prompt row stand for many tries. Uses are still recorded in
`prompt_uses` (kind `try`), so the library and the CLI's `prompts` commands keep working.

**Alternatives**: answers in a separate `ask_answers` table (normalised, but a try has at most a
few dozen answers and is read whole); browser storage (lost across browsers and machines, not
shared, SC-002).

### R12. Videos: picked on the server, or uploaded once and kept by content

**Decision**: Ask AI works on a video the server can read. Three ways in (FR-001):

1. **A server folder**: through the existing browse endpoint and `resolve_within_roots`.
2. **A video from an earlier run**: its `video_path`.
3. **Drop or upload**: `POST /api/v1/videos` stores it under the storage root at
   `videos/<first 16 hex of SHA-256>/<file name>` and returns its path. The same file uploaded
   twice is stored once; a later run reads it in place (spec 022).

The browser plays it through `GET /api/v1/videos/stream?path=…&token=…` (range requests) with a
short-lived token from `POST /api/v1/videos/stream-token`: an HMAC of path and expiry (1 hour)
with a per-server secret, so a `<video>` element can send it. A path is usable when it is under
the allowed roots, the uploads folder, or is an earlier run's video, for **any signed-in user**:
no admin or same-machine check (decided 2026-10-19: admin roles add friction for small labs,
which is most first users; everyone on a server sees everything, as with the history). Browsing
server folders keeps its existing rules. Spec 028 FR-013 reuses this.

**Alternatives**: a cookie session (a second auth mechanism beside API keys); blob download (no
playback until the whole file is fetched: today's problem).

### R13. Frames for the window come from the server's own extraction

**Decision**: `GET /api/v1/ask/frames?path=…&t=…&<sampling>` returns the frames a sample would
send (time, frame number, 320 px thumbnail), extracted by the pipeline's own functions. Used for
the burst thumbnails under the playhead (FR-011), hover on a marker, and showing a history entry's
frames (they are never stored: clarification 3). No model call.

**Rationale**: shows exactly the frames the model gets (same decoder, same frame rounding),
not a browser approximation. Three 320 px JPEGs locally take well under SC-003's half second;
the viewer debounces as the playhead moves.

### R14. The estimate

**Decision**: seconds per answer = the median `total_time` of this model's last 20 answers in
the history, scaled by frames per sample relative to theirs; with no answers yet, a rough 6 s
per frame, labelled "rough guess until you try it" (FR-012, US3 scenario 3). Warn above an hour
(edge case).

### R15. The window and its timeline

**Decision**: new route `/ask` (and `/ask?video=…`), code-split with `React.lazy` (constitution:
bundle budget). `/workbench` and `/prompts` redirect to `/ask`, keeping `prompt` and `model`
query parameters (FR-004, Principle V: old addresses keep working). The wizard's test panel is
replaced by an "Open in Ask AI" link that carries the prompt and settings. Compare (spec 021)
stays; it compares runs, not tries.

The timeline is new code under `viewer/src/components/lanes/`: a time axis, a sample-tick lane,
one lane per shown entry (answer markers, coloured per distinct answer; error and skipped markers
distinct), playhead, click to seek, hover. SVG, no new dependency. Spec 028 builds zoom, people
rows and numeric tracks on it; the existing `Timeline.tsx` is left for 028 to replace.

**Rationale**: spec Assumption (whichever spec lands first builds the timeline). Keeping it
small and dependency-free leaves 028 free to extend it.

### R16. Export

**Decision**: tries: `GET /api/v1/ask/entries/{id}/export/table` writes the spec 025 table from
the entry's answers through the same row builder (`readers.read_vlm`'s measures, same columns);
runs: the existing `/jobs/{id}/export/table`. "Export → Table" in the window downloads the shown
entries, one file per entry, or a zip when more than one (FR-021).

### R17. Example questions

**Decision**: a short static list in the viewer, each one stating the answer's form (FR-008):
touch (touching / not_touching), gaze (at_caregiver / at_object / elsewhere), number of people,
infant posture (lying / sitting / standing / held), activity (one of a list), describe in one
sentence. Inserting one replaces the question only after asking if the box isn't empty.
