---
description: "Tasks for spec 027: Ask AI"
---

# Tasks: Ask AI

**Input**: `specs/027-ask-ai/` (plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md)
**Tests**: included. The constitution requires tests for new API surface; the sampling twin needs
shared cases (research R2); SC-002/SC-003 are checked by tests.
**Organization**: by user story, in the spec's priority order (US1–US4 are P1, US5–US6 P2).
Within a story, server before viewer, tests beside the code they test.

**Conventions for every task**: Python passes `ruff check`, `ruff format`, `mypy src/videoannotator`;
viewer passes `bun run lint`, `tsc --noEmit -p tsconfig.app.json`, `bun run test:run`. Model answers
and reasoning are pipeline-supplied strings: never `dangerouslySetInnerHTML`. No comments
saying *what* code does. User-visible words follow spec 023 Decision 1: "question", "try", "run",
"answers"; never "job", "preview", "VLM", "prompt library" (contracts/viewer.md). Commit per
task or small group, `<area>: <summary>`, with the Co-Authored-By trailer; the container has no
git identity (`git -c user.name=… -c user.email=…`).

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup

- [X] T001 Create `tests/fixtures/ask_ai/sampling_cases.json`: a list of cases, each `{name, duration, interval, start, end, moments, frames, spacing, count, expect: {points, burst_times_at_first_point, chosen}}`; cover: defaults (5 s over 32 s), range 10–20 s, end past duration, listed moments (unsorted, duplicate, one past the end → dropped), 1 frame, 3 frames 0.5 s apart, 4 frames (offsets [-2,-1,0,1]), burst clipped at 0 and at the end, spacing unset (= interval, today's behaviour), count 10 of 40 points, count 10 of 6 points (all), count 1 (middle). Expected values per contracts/pipeline-config.md and research R3 (`round(i × (S − 1) / (N − 1))`)

---

## Phase 2: Foundational (blocks every story)

### Sampling and pipeline config

- [X] T002 Create `src/videoannotator/pipelines/vlm_annotation/sampling.py` with pure functions `sample_points(duration, interval, start=0.0, end=None, moments=None) -> list[float]`, `burst_offsets_for(frames: int) -> list[int]`, `burst_times(anchor, offsets, spacing, duration) -> list[tuple[int, float]]` (offset, time; drops times outside 0…duration), `choose_moments(points, count) -> list[float]`; times rounded to 3 decimals as `_sample_points` does today
- [X] T003 [P] Unit tests in `tests/unit/test_vlm_sampling.py` driven by `tests/fixtures/ask_ai/sampling_cases.json` (one parametrised test per function)
- [X] T004 In `src/videoannotator/pipelines/vlm_annotation/vlm_pipeline.py`: add config keys `start_sec` (0.0), `end_sec` (None), `moments_sec` (None), `burst_spacing_sec` (None) to the defaults; `process()` uses `sampling.sample_points` with config start/end/moments (the `start_time`/`end_time` arguments still narrow it when given, for callers that pass them); `_read_burst` takes the spacing (`burst_spacing_sec or frame_interval_sec`) and uses `sampling.burst_times`; remove `_sample_points` (update `tests/` callers, `grep -rn _sample_points`). Existing tests must pass unchanged in meaning
- [X] T005 [P] In `src/videoannotator/registry/metadata/vlm_annotation.yaml`: add the four keys with spec 026 fields per contracts/pipeline-config.md (group, label, unit, `applies_when` for `burst_spacing_sec`); reword `burst_offsets` description; run `pytest tests/ -k "registry or pipeline_guide or metadata"`
- [X] T006 [P] Create `viewer/src/lib/sampling.ts` mirroring T002 (`samplePoints`, `burstOffsetsFor`, `burstTimes`, `chooseMoments`) and `viewer/src/test/lib/sampling.test.ts` reading `../../../../tests/fixtures/ask_ai/sampling_cases.json` (check the relative import works under Vitest; otherwise copy the file at test time via `import.meta.glob` or a Vite `server.fs.allow` entry)

### Data and endpoints

- [X] T007 In `src/videoannotator/database/models.py`: add `AskEntry` (table `ask_entries`) and `EndpointConfirmation` (table `endpoint_confirmations`) per data-model.md; indexes on `ask_entries.created_at`, `video_path`, `user_id`, `job_id`; tables are created by the existing `create_all` (no ALTER needed); add a test in `tests/unit/test_database_models.py` (or the existing models test file) that both tables are created on a fresh SQLite database
- [X] T008 [P] Create `src/videoannotator/endpoints.py`: `normalise_base_url(url) -> str` (scheme, lower-case host, port, no trailing slash) and `classify_endpoint(base_url) -> EndpointInfo(host, cls: Literal["local","remote"])` per research R9 (explicit networks: 127/8, ::1, 10/8, 172.16/12, 192.168/16, fc00::/7, 169.254/16, fe80::/10; resolve with `socket.getaddrinfo`; any non-local address or a resolution failure → remote); tests in `tests/unit/test_endpoint_class.py` (patch `getaddrinfo`: localhost, a 192.168 address, `host.docker.internal` resolving to 192.168.65.254, a public address, a host resolving to one private and one public address → remote, unresolvable → remote, 100.64.1.1 → remote)
- [X] T009 Create `src/videoannotator/ask_history.py`: `create_entry(db, *, kind, prompt, model, base_url, endpoint_class, sampling, think, video_path, moments, job_id, user_id) -> AskEntry` (records the prompt via `prompt_library.record_use` with kind `try` or `job`, so `prompts` holds the text), `add_answer(db, entry_id, answer: dict)`, `finish(db, entry_id, status)`, `to_dict(entry) -> dict` (includes `prompt` text from the `prompts` row), `stop_running_tries(db) -> int`; tests in `tests/unit/test_ask_history.py` (create, add answers in order, finish, prompt use recorded with kind `try`, no field holds image data)
- [X] T010 In `src/videoannotator/api/main.py` (startup, where `create_all` runs): call `ask_history.stop_running_tries` so tries left `running` by a restart become `stopped` with their answers kept; test in `tests/unit/test_ask_history.py`

### Videos (shared with spec 028)

- [X] T011 Create `src/videoannotator/api/v1/videos.py` per contracts/api.md §Videos: `POST /videos` (multipart; stream to a temp file while hashing; store at `<storage root>/videos/<sha256[:16]>/<name>` via the storage config used for job uploads, reuse if present; return path, sha256, name, duration from `cv2`), `POST /videos/stream-token` (HMAC-SHA256 of `path|expires` with a per-server secret kept in the database directory as `stream_secret` (created 0600 on first use); 1-hour expiry), `GET /videos/stream` (no API key; verify token, then `FileResponse` with range support). A readable path is one under `ingest.allowed_roots()`, the videos folder, or a job's `video_path`, for any signed-in user (no admin or same-machine check: research R12); otherwise `403 PATH_NOT_ALLOWED`. Register the router in `src/videoannotator/api/v1/__init__.py`
- [X] T012 [P] Tests in `tests/api/test_videos_api.py`: upload stores once for identical content; token for one path refused for another; expired token refused; range request returns 206 with the right bytes; path outside the allowed places refused; stream without token refused

### Ask router and viewer plumbing

- [X] T013 Create `src/videoannotator/api/v1/ask.py` with the router (prefix `/ask`), Pydantic models `Sampling`, `Answer`, `Entry` matching data-model.md, a `sampling_to_config(sampling) -> dict` helper per contracts/pipeline-config.md §Mapping (tested in `tests/unit/test_vlm_sampling.py`), and the shared checks: `llm` extra present (`PipelineUnavailableException`), video readable (same rule as T011; factor it into one function in `videos.py` and import it), sampling valid (`422 INVALID_SAMPLING` with the field). Register in `src/videoannotator/api/v1/__init__.py`
- [X] T014 In `src/videoannotator/api/v1/vlm.py`: make `preview_vlm_prompt`'s blocking work (frame extraction and `client.chat`) run through `starlette.concurrency.run_in_threadpool`; accept `burst_spacing_sec: float | None = Form(None)`; test in `tests/api/test_vlm_preview_nonblocking.py` that a second request (`GET /api/v1/system/health`) completes while a preview's fake `chat` is sleeping
- [X] T015 [P] Regenerate `viewer/src/api/schema.d.ts` from the server's OpenAPI (the script or command used before; see `viewer/package.json` scripts) and add typed methods to `viewer/src/api/client.ts`: `uploadVideo`, `streamToken`, `createTry`, `getEntry`, `listEntries`, `updateEntry`, `deleteEntry`, `createRun`, `getRunAnswers`, `getFrames`, `confirmEndpoint`; `streamUrl(path, token)` builds the `<video>` src
- [X] T016 [P] Create the lane timeline in `viewer/src/components/lanes/`: `TimeAxis.tsx` (ticks and labels for a duration and width), `SampleTicks.tsx` (marks at given times), `AnswerLane.tsx` (markers at times, colour per distinct label via `lib/categoryColors.ts`, distinct shapes for `error` and `skipped`, highlighted marker, `onSeek(t)`, `onHover(answer)`), `Playhead.tsx`, `LaneLabel.tsx`; SVG, no new dependency, props only (no shared state between lanes: constitution IV); tests in `viewer/src/test/components/lanes.test.tsx` (marker positions, click seeks, error/skipped look different)
- [X] T017 Create `viewer/src/pages/AskAI.tsx` (lazy-loaded in `viewer/src/App.tsx` with `React.lazy`) with the layout of contracts/viewer.md: left panel, video right, timeline and table below; reads `video`, `entry`, `prompt`, `model` from the query string; `VideoPane` plays via `streamUrl` (token fetched with T015; refreshed before expiry) and warns with `lib/videoCodec.ts` when the browser can't play the codec (the demo clip is H.265); add the `/ask` route

**Checkpoint**: sampling is one definition in two languages; the ask router, videos API and page shell exist; nothing user-visible beyond an empty Ask AI page.

---

## Phase 3: User Story 1 — First answer in under a minute (P1) 🎯 MVP

**Goal**: open Ask AI from Home, give it a video, pick an example, press Try on this frame, see the answer.
**Independent test**: quickstart §1, on a fresh database with a local model.

- [X] T018 [US1] In `src/videoannotator/api/v1/ask.py`: `POST /ask/tries` with `at` (one moment) per contracts/api.md: classify the endpoint (T008); create the entry before any model call (T009); return `201` with the entry; answer in a background thread (`threading.Thread`, daemon) that extracts frames with the pipeline's own functions (shared with `_extract_video_frames` in `vlm.py`; move it to `sampling.py` or a `frames.py` beside it so both use one), calls `OllamaVLMClient.chat` with the pipeline defaults, parses with `_parse_label`, appends the Answer (status `answered`/`error`/`skipped`, `frame_times`, `total_time`) and finishes the entry. `503 OLLAMA_UNREACHABLE` up front (one `list_models` call) with `unreachable_hint`; a model not pulled → `422 MODEL_NOT_PULLED` with the `ollama pull` hint
- [X] T019 [US1] `GET /ask/entries/{id}` in `src/videoannotator/api/v1/ask.py`
- [X] T020 [P] [US1] Tests in `tests/api/test_ask_api.py` with a fake `OllamaVLMClient` (monkeypatch the class in `ask.py`): try returns 201 with `status: running` before the fake answers; polling reaches `done` with one answer whose `frame_numbers` match the frame at `at`; model error → answer `status: error`, entry `done`; unreadable frame → `skipped`; Ollama unreachable → 503 with hint; missing `llm` extra → 422; a `try` use is in `prompt_uses`
- [X] T021 [P] [US1] Create `viewer/src/components/ask/VideoPicker.tsx`: drop or choose a file (uploads with `uploadVideo`, shows progress), pick a server folder (reuse `ServerFolderPicker.tsx`; shown only when the browse endpoint allows it), or a recent video (from `listEntries` and recent jobs' `video_path`); on choice, navigate to `/ask?video=<path>`
- [X] T022 [P] [US1] Create `viewer/src/components/ask/Examples.tsx` with the questions of research R17 (each states the answer's form); inserting into a non-empty question box asks first
- [X] T023 [US1] Create `viewer/src/components/ask/QuestionPanel.tsx` (question box, Examples, model picker reusing `VlmModelPicker.tsx`, "Try on this frame") and `viewer/src/components/ask/AnswersTable.tsx` (time, frames, answer, reasoning; one answer column per shown entry); wire into `AskAI.tsx`: Try creates a try at the playhead, polls `getEntry` every second while `running`, shows the answer, reasoning and frames sent (from the entry's answer + `getFrames`; "2 of 3 frames" when a burst was clipped at either end) beside the video, and a marker on an `AnswerLane`
- [X] T024 [US1] In `AskAI.tsx`: when `/vlm/models` fails or lists no models, show what's missing and how to get it (install and start the model server; `ollama pull <model>`; link to the Pipelines page's readiness card), in words, not codes; in standalone mode (no server configured) say Ask AI needs a server and a model
- [X] T025 [US1] In `viewer/src/pages/Home.tsx`: add "Ask AI about a video" leading to `/ask`; in `viewer/src/components/AppLayout.tsx` add "Ask AI" to the nav; update `viewer/src/test/components/Home.test.tsx`
- [X] T026 [P] [US1] Tests in `viewer/src/test/components/AskAI.test.tsx` (mock the client): video from query string plays via a stream URL; Try shows the answer after the entry turns `done`; no-model state shows the explanation; example inserts

**Checkpoint**: US1 works end to end (quickstart §1).

---

## Phase 4: User Story 2 — Never lose a prompt (P1)

**Goal**: every try in a shared history, restorable, starrable, comparable.
**Independent test**: quickstart §2.

- [X] T027 [US2] In `src/videoannotator/api/v1/ask.py`: `GET /ask/entries` (filters `video_path`, `mine` = caller's user id, `starred`, paging by `before`; `library` = prompts with no entry and not hidden, newest first, empty when filtered by video), `PATCH /ask/entries/{id}` (`name`, `starred`), `DELETE /ask/entries/{id}` (author or admin, else 403)
- [X] T028 [P] [US2] Tests in `tests/api/test_ask_api.py`: five tries listed newest first with `user_id`; `mine` filters by caller (two API keys); a library prompt with no entry appears in `library` and disappears once a try uses it; patch name/star visible to another user; delete by non-author refused
- [X] T029 [US2] Create `viewer/src/components/ask/HistoryList.tsx`: entries (name or first line of the question, model, sampling summary, video name, who, when, star), library prompts below under "Earlier questions", filters (this video / all videos, mine), star and rename inline; choosing an entry restores question, model and sampling (`/ask?…&entry=<id>` pushes history) without touching the entry; if the question box holds text matching no entry, it is first saved as a `draft` entry (no answers), listed with the rest (US2 scenario 2). Library prompts under "Earlier questions" can be starred, renamed and hidden through the existing `PUT /prompts/{sha}`, and deleted through `DELETE /prompts/{sha}` (refused for prompts runs used, as today); tags stay available through `videoannotator prompts`
- [X] T030 [US2] In `AskAI.tsx` and `AnswersTable.tsx`: "Show" on up to four entries of this video → one `AnswerLane` and one table column each, aligned by sample time (a time present in one entry and not another shows an empty cell, not a guess: Principle VI); each lane labelled with the entry's name or question and model (FR-007); a moment tried more than once whose answers differ is marked in the table (icon, the answers on hover); identical repeats are not marked (edge case)
- [X] T031 [US2] In `viewer/src/App.tsx`: `/workbench` → `/ask` keeping `prompt` and `model`; `/prompts` → `/ask?history=open`; in `viewer/src/components/DynamicPipelineParameters.tsx` replace `VlmPromptTestPanel` with an "Open in Ask AI" link carrying prompt, model and sampling (opens in a new tab as the test panel's Workbench link did)
- [X] T032 [P] [US2] Tests in `viewer/src/test/components/AskAI.test.tsx`: restore fills the form; two shown entries give two lanes and columns; `/workbench?prompt=x` lands on Ask AI with `x`; replace `viewer/src/test/components/Workbench.test.tsx` with these redirect tests

**Checkpoint**: quickstart §2 passes; reloading the browser loses nothing.

---

## Phase 5: User Story 3 — Understand sampling by seeing it (P1)

**Goal**: sampling controls draw where samples fall and show the frames, with a count and estimate, before anything is sent.
**Independent test**: quickstart §3.

- [X] T033 [US3] In `src/videoannotator/api/v1/ask.py`: `GET /ask/frames` per contracts/api.md (frames a sample at `t` would send, 320 px thumbnails via `vlm._thumbnail` moved beside the frame helpers, `requested` count); run in the threadpool
- [X] T034 [US3] In `src/videoannotator/api/v1/ask.py`: `POST /ask/tries` with `count` → moments from `sampling.choose_moments(sample_points(...), count)` (research R3); answers in moment order
- [X] T035 [P] [US3] Tests in `tests/api/test_ask_api.py`: frames endpoint returns 3 frames 0.5 s apart; near the end returns fewer with `requested: 3`; `count: 10` on a 40-point grid tries the expected moments from the shared cases; listed moments are tried exactly
- [X] T036 [P] [US3] Create `viewer/src/lib/estimate.ts` (research R14: median `total_time` of the model's last 20 answers from the history, scaled by frames; else 6 s per frame and `rough: true`; `overHour` flag) with `viewer/src/test/lib/estimate.test.ts`
- [X] T037 [US3] Create `viewer/src/components/ask/SamplingControls.tsx`: every N s; from/to (mm:ss inputs, "end"); "only at these moments" (comma-separated mm:ss, parsed with the existing `lib/workbench.ts` `parseMoments`, then move that function to `lib/sampling.ts` and delete `lib/workbench.ts`); frames per sample (1–9) and seconds apart (shown when frames > 1); one-line explanation with a "More" link to the docs page (T064); count and estimate line ("40 samples · about 3 min", "rough guess until you try it", over-an-hour warning)
- [X] T038 [US3] In `AskAI.tsx`: `SampleTicks` from `samplePoints` on every change (no request); when frames > 1, thumbnails of the sample nearest the playhead from `getFrames` (debounced 150 ms) with their times; "Try on 10 moments" button; the nearest sample highlighted in lane and table (FR-003)
- [X] T039 [P] [US3] Tests in `viewer/src/test/components/AskAI.test.tsx`: changing interval updates tick count with no client call (assert the mocked client wasn't called); estimate shows "rough guess" before any answer and measured after; listed moments give exactly those ticks

**Checkpoint**: quickstart §3 passes.

---

## Phase 6: User Story 4 — Know where the video goes (P1)

**Goal**: a badge that always says where the model runs, a confirmation before the first remote send, and locality in provenance.
**Independent test**: quickstart §4.

- [X] T040 [US4] In `src/videoannotator/api/v1/vlm.py` `GET /vlm/models`: add `endpoint: {host, class, confirmed}` (T008; `confirmed` from `endpoint_confirmations`, always true for local) to `VlmModelsResponse`; include it when the server is unreachable too (in the 503 hint body is not enough: add a `GET /ask/endpoint` returning the same object, so the badge shows even with no models)
- [X] T041 [US4] In `src/videoannotator/api/v1/ask.py`: `POST /ask/endpoints/confirm`; tries and runs to a remote, unconfirmed endpoint → `409 REMOTE_NOT_CONFIRMED` with the host in the message
- [X] T042 [US4] In `src/videoannotator/pipelines/vlm_annotation/vlm_pipeline.py` `provenance_vlm()`: add `endpoint_class` and `endpoint_host` from `classify_endpoint`; in `src/videoannotator/exporters/methods.py` say "running on this computer" (local, loopback) / "running on the local network" (local, private) / "running on a remote server" (remote); update the methods tests. *Implemented without the host name: spec 025 FR-029 keeps server addresses out of the methods text; the host is in the provenance record.*
- [X] T043 [P] [US4] Tests: `tests/api/test_ask_api.py` (remote unconfirmed → 409; confirm → 204 then the try is accepted; confirmation seen by a second user), `tests/unit/test_provenance*.py` or the VLM provenance test (fields present), methods paragraph wording
- [X] T044 [US4] Create `viewer/src/components/ask/EndpointBadge.tsx`: "Local · <model> on this computer" / "Local · <model> on <host>" / "Remote · frames are sent to <host>" (warning style); first view of a local badge shows "Frames from your video are sent only to this computer" once (remember dismissal in `localStorage`, wrapped in try/catch); put it in the Ask AI header
- [X] T045 [US4] In `AskAI.tsx`: before the first try or run to an unconfirmed remote endpoint, a dialog naming the host and saying frames will leave this computer; confirm calls `confirmEndpoint` then proceeds; cancel sends nothing
- [X] T046 [P] [US4] Tests in `viewer/src/test/components/EndpointBadge.test.tsx` and `AskAI.test.tsx`: local and remote wording and style; dialog shown once and not after confirmation; cancel sends no request
- [X] T047 [US4] Remove the browser-side `LOCAL_HOSTS` guess: grep `LOCAL_HOSTS` and `remoteServer` in `viewer/src`; any remaining use reads `endpoint.class` from `/vlm/models`

**Checkpoint**: quickstart §4 passes.

---

## Phase 7: User Story 5 — Run on the whole video, answers as they come (P2)

**Goal**: a background run whose answers fill the window, that a try can jump ahead of, and that can be stopped without losing anything.
**Independent test**: quickstart §5.

### Pipeline

- [X] T048 [US5] In `src/videoannotator/pipelines/vlm_annotation/vlm_pipeline.py`: write the COCO file progressively (research R6): after each sample, at most once a second, and at the end, through a write-to-temp-then-`os.replace` helper; top-level `vlm_progress {planned, done, finished, stopped_early}` and `vlm_skipped [{t, reason}]`; per annotation `frame_times` and `status` (`answered`/`error`). `_save_coco_annotations` / `export_coco_json` gain an optional `extra` dict for the top-level keys (check `exporters/native_formats.py` signature; additive). Readers: confirm `exporters/readers.py` `read_vlm` and the viewer's VLM parser ignore the new keys (add a case to each test)
- [X] T049 [US5] Same file: `set_stop_check(check: Callable[[], bool])`; check before each sample; on stop: write what exists with `stopped_early: true` and return normally (no raise); the abort-after-consecutive-failures and no-answer-at-all failures stay as they are. `run_summary() -> dict` returns `samples_planned`, `samples_answered`, `stopped_early`
- [X] T050 [US5] Create `src/videoannotator/pipelines/vlm_annotation/gate.py`: `gate_for(base_url, model) -> ModelGate` (module-level dict, lock-protected); `ModelGate.acquire(priority: Literal["try","run"])` as a context manager: a `run` acquire waits while any `try` is waiting or a call holds the gate; a `try` waits only for the call in progress. Use it around `client.chat` in `vlm_pipeline.py` (run) and in the try worker (T018, try). Tests in `tests/unit/test_vlm_gate.py` with threads and events: a try queued during a run call is served before the run's next call; two runs alternate; no deadlock when a call raises
- [X] T051 [US5] In `src/videoannotator/batch/job_execution.py` `_run_one_pipeline`: if the pipeline has `set_stop_check`, pass `lambda: _cancellation_requested(job.job_id, storage)`; after `process()`, if it has `run_summary`, merge it into `record["vlm"]`. Check `exporters/results_set.py`, `api/v1/exports.py` and the results listing include a CANCELLED job's COMPLETED pipeline results in the table export and methods (fix the filter if not; test it)
- [X] T052 [P] [US5] Tests in `tests/unit/test_vlm_pipeline_progressive.py` (fake client, synthetic video from `tests/fixtures/synthetic_video.py`): the file exists with `done: 3` after three samples while the fourth call is blocked; stop check true after five samples → normal return, five annotations, `stopped_early: true`; unreadable frame listed in `vlm_skipped`, not in annotations; range and moments from config are honoured; old config (no new keys) gives today's sample points and burst frames
- [X] T053 [US5] Integration test `tests/integration/test_ask_run_stop.py`: through the API with a fake client, `POST /ask/runs`, answers appear in `GET /ask/runs/{id}/answers` before the run ends, `POST /jobs/{id}/cancel` → job CANCELLED, `vlm_annotation` result COMPLETED, results file has the answers so far, provenance `stopped_early: true`, `/jobs/{id}/export/table` has those rows

### Server

- [X] T054 [US5] In `src/videoannotator/api/v1/ask.py`: `POST /ask/runs` (same checks as tries; creates a job the way `POST /jobs` does for a server-side video (reuse the job-creation helper used by `ingest`/`datasets` run, not the upload path) with pipeline `vlm_annotation` and `sampling_to_config` + question, model, base URL, think; creates the run entry with `job_id`); `GET /ask/runs/{job_id}/answers?after=N` reading the results file (tolerate a missing or mid-replace file: return what was last readable, `done: 0`); entries for runs report the job's status and progress in `to_dict`
- [X] T055 [P] [US5] Tests in `tests/api/test_ask_api.py`: run creates a job with the mapped config (spacing, range, moments); answers endpoint pages with `after`; a try during a run on the same fake model is answered before the run's next sample (gate, using events in the fake client)

### Viewer

- [X] T056 [US5] Create `viewer/src/components/ask/RunControls.tsx` and wire into `AskAI.tsx`: "Run on whole video" (range respected) → `createRun`; while running, poll `getRunAnswers` every 2 s and append to the run's lane and table column; progress "23 of 40"; "Stop" → existing cancel; a stopped run reads "Stopped after 23 of 40", not failed; trying stays enabled during a run
- [X] T057 [US5] "Run on more videos" opens the New job wizard (`/jobs/new`) with `vlm_annotation` selected and question and sampling filled in (extend `lib/wizardStart.ts`'s start state; test in `viewer/src/test/lib/wizardStart.test.ts`); "Save as workflow" saves a pipeline preset (`/presets` API; spec 029 renames presets to workflows) named from the entry
- [X] T058 [P] [US5] Tests in `viewer/src/test/components/AskAI.test.tsx`: answers appear as the mocked answers endpoint grows; Stop shows the stopped wording and keeps answers; "Run on more videos" navigates with the state

**Checkpoint**: quickstart §5 passes.

---

## Phase 8: User Story 6 — Answers out (P2)

**Goal**: the answers as a table file, a tally with a hint, and run answers readable on the video's page.
**Independent test**: quickstart §6.

- [X] T059 [US6] In `src/videoannotator/api/v1/ask.py`: `GET /ask/entries/{id}/export/table`: for a try, the spec 025 CSV built from the entry's answers with the same columns and measures as `readers.read_vlm` (factor a `rows_from_vlm_answers(answers, video, model, …)` in `exporters/table.py` or `readers.py` used by both); for a run, `307` to `/jobs/{job_id}/export/table`; test in `tests/api/test_ask_api.py` that a try's CSV and a run's CSV have identical header rows
- [X] T060 [P] [US6] Create `viewer/src/components/ask/Tally.tsx`: distinct answers with counts for the shown entries (errors and skipped counted separately, never as answers); more than 10 distinct → the hint "Say in the question which answers are allowed, e.g. 'Answer with one word: touching or not_touching.'"; and `Export ▾ → Table` downloading one CSV per shown entry, zipped when more than one (reuse `lib/exports.ts` patterns)
- [X] T061 [P] [US6] Tests: `viewer/src/test/components/Tally.test.tsx` (counts; hint at 11; errors separate)
- [X] T062 [US6] Run answers on the video's page: spec 028 (FR-022) owns the lane and caption there. Here: confirm the current results viewer (`components/VlmAnnotationPanel.tsx`, `/view/:jobId`) still shows a run made from Ask AI, including a stopped one, and add a viewer contract fixture with the new fields in `tests/fixtures/viewer_contract/` (Principle II guard) checked by `viewer/src/test/contract/videoannotator-outputs.test.ts`

**Checkpoint**: quickstart §6 passes.

---

## Phase 9: Polish and cross-cutting

- [X] T063 Remove `viewer/src/pages/Workbench.tsx`, `viewer/src/pages/Prompts.tsx`, `viewer/src/components/VlmPromptTestPanel.tsx`, their imports and nav entries, and tests that only covered them; keep anything still used elsewhere (`lib/vlmCompare.ts` is used by Compare); `bun run lint && bun run test:run`
- [X] T064 [P] Docs: new `docs/usage/ask_ai.md` (what a try and a run are, sampling explained with a picture of ticks and a burst, local vs remote, history shared on the server, exporting, the CLI-worker gate limitation), linked from `mkdocs.yml` nav and `docs/usage/README.md`; update the VLM pipeline docs for the four new keys (`docs/usage/pipeline_specs.md` or wherever `vlm_annotation` config is documented: `grep -rn burst_offsets docs`); `uv run --no-sync mkdocs build --strict`
- [X] T065 [P] `CHANGELOG.md` (Unreleased): Ask AI replaces Workbench and Prompts; new sampling keys; runs show answers as they come and can be stopped keeping answers; preview no longer blocks the server; `/videos` streaming
- [X] T066 Bundle and build: `bash scripts/build_viewer.sh`; confirm the initial bundle stays within 300 KB gzipped (Ask AI in its own chunk); commit `src/videoannotator/viewer_static/`
- [ ] T067 Full gates: `pytest tests/`, `ruff check .`, `mypy src/videoannotator`, `pre-commit run --all-files` (known: `bats` missing locally), viewer lint/tsc/tests
- [ ] T068 Walk through `quickstart.md` §1–§6 on the demo clip with a real Ollama model (GPU machine); record results, timings (SC-001 first answer, SC-004 lag), whether the badge was read correctly by anyone watching (SC-005; the full check is the researcher try-out and the pilot) and anything that didn't match in a new `specs/027-ask-ai/checks.md`
- [ ] T069 Update `docs/development/roadmap_v1.7.0.md` Phase 1 (027 done; link checks.md) and `CLAUDE.md` Current Plan; mark spec 027 Status "Built"

---

## Dependencies and execution order

- **Phase 1 → Phase 2**: T001 before T003, T006.
- **Phase 2** blocks all stories. Inside it: T002 → T004 → (T005); T002 → T006; T007 → T009 → T010; T011 before T013 (shared path check); T013 before every `/ask` task; T015, T016 before T017.
- **US1** (T018–T026) needs Phase 2. It is the MVP.
- **US2** needs US1's entries (T018, T019).
- **US3** needs US1 (tries, page). T034 extends T018.
- **US4** needs Phase 2 (T008) and US1 (tries to guard); independent of US2/US3.
- **US5** pipeline tasks (T048–T053) need only Phase 2 and can start alongside US1; its server and viewer tasks (T054–T058) need US1 and US3's sampling controls. T050's gate is also used by T018's worker: wire it there when T050 lands.
- **US6** needs US1 (try export) and US5 (run export, stopped runs).
- **Polish** last; T063 only after T031 (redirects) and T029 (history replaces Prompts).

## Parallel opportunities

- Phase 2: T003, T005, T006, T008, T012, T015, T016 touch separate files.
- Server and viewer halves of a story run in parallel once the contract (contracts/api.md) is fixed: e.g. T018–T020 alongside T021–T022.
- The US5 pipeline work (T048–T053) is independent of all viewer work.

Example (US1):

```
T020 tests/api/test_ask_api.py           ║ T021 components/ask/VideoPicker.tsx
                                         ║ T022 components/ask/Examples.tsx
```

## Implementation strategy

1. **MVP**: Phases 1–3 (US1). A researcher can try a question on a frame of their video, in one
   window, with nothing lost on the server. Stop and walk through quickstart §1.
2. **The P1 set**: US2 (history), US3 (sampling), US4 (badge). After these, Workbench and Prompts
   can be retired (T063) and the sketches tried on researchers against the real thing.
3. **P2**: US5 (runs as they come, stop), US6 (out).
4. Each story ends at its checkpoint with the full suite green and a commit.
