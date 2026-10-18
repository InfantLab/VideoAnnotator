# Implementation Plan: Ask AI

**Branch**: `1.7-dev` (spec `027-ask-ai`) | **Date**: 2026-10-18 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/027-ask-ai/spec.md` (clarified 2026-10-18)

## Summary

One window, **Ask AI**, replaces the Workbench, the Prompts page and the wizard's prompt test
panel: the question and sampling beside the video, answers on a timeline and in a table below,
every try kept in a server-side history shared on the server, a badge saying where the model
runs, and a whole-video run whose answers appear as they come and can be stopped without losing
them.

Approach (from [research.md](research.md)): the VLM pipeline gains additive sampling keys
(range, listed moments, burst spacing), rewrites its results file as answers arrive, honours an
optional stop check and orders its model calls through a per-model gate that lets tries go first.
A new `/api/v1/ask` API records tries and runs as history entries (new `ask_entries` table
pointing at the prompt library), answers tries in a worker thread, classifies the model endpoint
as local or remote on the server, and serves sample frames. A new `/api/v1/videos` API uploads
videos once and streams them to the browser with a short-lived token (shared with spec 028). The
viewer gets the `/ask` page and a small lane timeline that spec 028 extends.

## Technical Context

**Language/Version**: Python 3.12/3.13 (server); TypeScript, React 18 (viewer, Vite, Bun)
**Primary Dependencies**: FastAPI, SQLAlchemy, Pydantic, OpenCV (existing); `ollama` client under
the `llm` extra (existing); viewer: TanStack Query, React Router, shadcn/ui (existing). No new
dependency.
**Storage**: server SQLite: new tables `ask_entries`, `endpoint_confirmations`; results folder
(spec 022) unchanged in layout; uploaded videos under the storage root's `videos/`.
**Testing**: pytest (unit, api, integration with a fake Ollama client); Vitest (sampling twin,
page behaviour, contract cases); manual walk-through [quickstart.md](quickstart.md) on the demo clip.
**Target Platform**: the API server on Linux, macOS, Windows; current Chrome, Firefox, Safari.
**Project Type**: web application (Python API + React viewer in one repo).
**Performance Goals**: sampling changes redraw in < 0.5 s with no request (SC-003); run answers
in the window ≤ 5 s after the model answers (SC-004, 2 s polling); a try waits at most one run
sample (FR-017a); the server stays responsive during tries (no blocking of the event loop).
**Constraints**: local-first (Principle I): nothing leaves the machine unless a remote endpoint
is configured and confirmed; no frames stored in the history; initial viewer bundle within
300 KB gzipped (Ask AI code-split); existing VLM configs and outputs unchanged in meaning.
**Scale/Scope**: one lab server, a handful of users; histories of thousands of entries; runs of
up to a few thousand samples.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design: still passes.*

| Gate | Status | How |
|---|---|---|
| **I. Local-first, no telemetry** | ✅ | Default endpoint is local Ollama. Locality decided on the server by address (R9); a remote endpoint needs a recorded confirmation, enforced by the API (R10). Video streams only from the user's own server. No new outbound traffic. |
| **II. Stable pipeline contract, open formats** | ✅ | `BasePipeline` unchanged: the stop check is an optional duck-typed hook (R7). Output stays COCO JSON with the same annotations; new keys are additive (`frame_times`, `status`, `vlm_skipped`, `vlm_progress`). The viewer contract test gains a fixture with the new fields. |
| **III. Provenance** | ✅ | Sampling is declarative config, recorded in `settings`. `vlm` provenance adds endpoint class and host, samples planned/answered and whether stopped (R8). Tries record question, model, endpoint, sampling and moments in their entry. |
| **IV. Modular** | ✅ | No new dependency. Model calls stay behind the `llm` extra; `/ask` endpoints report `PIPELINE_UNAVAILABLE` without it. The core doesn't import the pipeline: the API does, as `/vlm` already does. Viewer: lanes are a self-contained component; Ask AI is code-split. |
| **V. Backward compatibility** | ✅ | New config keys default to today's behaviour (burst spacing = interval). `/vlm/preview` kept. `/workbench` and `/prompts` redirect with their parameters. A stopped run is CANCELLED, an existing status, with a COMPLETED pipeline result. |
| **VI. Faithful display** | ✅ | Ticks come from the same sampling cases as the pipeline (R2). Errors and skipped samples are drawn distinctly, never as answers. Thumbnails are the frames the model gets (R13). Each lane is labelled with its entry (question name, model) and a run lane with the pipeline and version. |
| **Engineering standards** | ✅ | Tests per area (below); API surface typed; docs: the VLM pipeline page, the pipeline guide labels (spec 026), the Ask AI how-to, CHANGELOG; viewer `tsc`, ESLint, Vitest, bundle check. |

No violations; Complexity Tracking is empty.

## Project Structure

### Documentation (this feature)

```
specs/027-ask-ai/
├── spec.md              # clarified 2026-10-18
├── plan.md              # this file
├── research.md          # R1–R17: what exists, decisions
├── data-model.md        # ask_entries, endpoint_confirmations, sampling, output and provenance fields
├── quickstart.md        # walk-through per user story
├── contracts/
│   ├── api.md           # /ask, /videos, changes to /vlm
│   ├── pipeline-config.md  # vlm_annotation keys, mapping, behaviour, shared test cases
│   └── viewer.md        # routes, layout, behaviour the tests check
├── checklists/requirements.md
└── tasks.md             # /speckit-tasks (not created here)
```

### Source Code

```
src/videoannotator/
├── pipelines/vlm_annotation/
│   ├── vlm_pipeline.py      # new keys; progressive write; stop check; gate; run summary
│   ├── sampling.py          # NEW sample_points, burst_times, choose_moments (R2, R3)
│   └── gate.py              # NEW per-model priority gate (R5)
├── registry/metadata/vlm_annotation.yaml   # new keys, spec 026 labels
├── batch/job_execution.py   # set_stop_check hook; run summary into provenance
├── endpoints.py             # NEW classify_endpoint, normalise base URL (R9)
├── ask_history.py           # NEW entries: create, answer, list (with library), update (R11)
├── database/models.py       # AskEntry, EndpointConfirmation
├── api/v1/
│   ├── ask.py               # NEW /ask: tries, entries, runs, run answers, frames, confirm
│   ├── videos.py            # NEW /videos: upload, stream token, stream (R12)
│   ├── vlm.py               # preview off the event loop, burst_spacing_sec; models + endpoint
│   └── __init__.py          # routers
├── exporters/readers.py     # read_vlm: tolerate new fields; table rows from an entry (R16)
└── exporters/methods.py     # "on a model running on this computer / on <host>"

viewer/src/
├── pages/AskAI.tsx          # NEW (lazy)
├── components/ask/          # NEW QuestionPanel, Examples, HistoryList, SamplingControls,
│                            #     EndpointBadge, AnswersTable, Tally, RunControls
├── components/lanes/        # NEW TimeAxis, SampleTicks, AnswerLane, Playhead (shared with 028)
├── lib/sampling.ts          # NEW twin of sampling.py
├── lib/estimate.ts          # NEW (R14)
├── api/client.ts            # ask/videos methods; schema.d.ts regenerated
├── App.tsx                  # /ask; /workbench, /prompts redirects
└── pages/Workbench.tsx, Prompts.tsx, components/VlmPromptTestPanel.tsx   # removed

tests/
├── fixtures/ask_ai/sampling_cases.json
├── unit/test_vlm_sampling.py, test_vlm_gate.py, test_endpoint_class.py, test_ask_history.py
├── unit/test_vlm_pipeline_progressive.py   # progressive write, stop, skipped, run summary
├── api/test_ask_api.py, test_videos_api.py, test_vlm_preview_nonblocking.py
└── integration/test_ask_run_stop.py        # run → answers as they come → stop → export
```

**Structure Decision**: the existing web-application layout: server code in
`src/videoannotator/`, viewer in `viewer/` built into `viewer_static/` (rebuilt and committed,
CI checks it).

## Order of work

Each stage ships on its own and keeps the suite green.

1. **Pipeline** (R1–R3, R5–R8): `sampling.py` with the shared cases; new config keys and registry
   metadata; progressive write with `vlm_progress` and `vlm_skipped`; stop check through
   `job_execution`; the gate; provenance fields; readers tolerate the new fields; check that
   export and results listings include a CANCELLED job's completed results.
2. **Server** (R4, R9–R13, R16): `classify_endpoint` and `/vlm/models.endpoint`; confirmations;
   `ask_entries` and `ask_history`; `/ask/tries` with the worker thread and gate; `/ask/entries`
   (with library prompts), patch, delete, export; `/ask/runs` and run answers; `/ask/frames`;
   `/videos` upload, token and stream; preview off the event loop. Mark running tries stopped at
   startup.
3. **Viewer** (R14, R15, R17): sampling twin and estimate; lanes; the page in user-story order
   (first answer, history, sampling, badge, run, out); redirects and nav; remove Workbench,
   Prompts and the test panel; contract fixture; rebuild `viewer_static`.
4. **Docs**: Ask AI how-to (replacing the Workbench and Prompts pages in the docs), VLM pipeline
   page (new keys), CHANGELOG, roadmap v1.7.0 Phase 1 tick; quickstart walk-through on the demo
   clip, recorded in `specs/027-ask-ai/checks.md`.

## Risks

- **Range requests**: `FileResponse` serves ranges (Starlette 1.9.0, checked 2026-10-18; spec 021
  relies on it); the stream endpoint reuses it. To verify with Safari, which is strictest. The
  demo clip is H.265, which Firefox and some Chrome builds can't play: the page must say so
  rather than show a blank player.
- **Gate fairness under two concurrent runs**: both runs wait while a try waits, then take turns
  as today. Acceptable; tested.
- **Results-file rewrite on slow disks** (network shares, spec 022): throttled to once a second;
  the write is atomic, so readers never see half a file.
- **Spec 023 Process step 5** (sketches tried on researchers) hasn't happened; findings may
  change the layout but not the server work, which is why the server stages come first.

## Complexity Tracking

None.
