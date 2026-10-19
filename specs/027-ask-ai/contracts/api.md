# Contract: Ask AI HTTP API

All endpoints are under `/api/v1`, take the usual API key (`Authorization: Bearer`) unless
noted, and return errors in the standard `APIError` shape (`code`, `message`, `hint`). All
require the `llm` extra where a model is called (`422 PIPELINE_UNAVAILABLE` otherwise, as
`/vlm/*` today).

Shared shapes: **Sampling**, **Answer** and **Entry** are defined in
[../data-model.md](../data-model.md).

## Endpoint and models

### `GET /vlm/models` (changed, additive)

Adds `endpoint`:

```json
{ "base_url": "http://localhost:11434", "models": ["qwen3.5:9b"],
  "endpoint": { "host": "localhost", "class": "local", "confirmed": true } }
```

`class` is `local` or `remote` (research R9). `confirmed` is always true for `local`.

### `GET /ask/endpoint`

`{ "base_url": "…", "host": "…", "class": "local", "confirmed": true }`: works even when the
model server is unreachable, so the badge always shows.

### `POST /ask/endpoints/confirm`

Body `{ "base_url": "https://api.example.com" }` → `204`. Records the confirmation for everyone
on the server (FR-015).

## Tries

### `POST /ask/tries` → `201 Entry`

```json
{ "video_path": "/data/p01_t1.mp4", "prompt": "…", "model": "qwen3.5:9b",
  "base_url": null, "think": false,
  "sampling": { "interval": 5, "start": 0, "end": null, "moments": null,
                "frames": 3, "spacing": 0.5 },
  "at": 12.4 }
```

Exactly one of `at` (one moment: "Try on this frame") or `count` (N moments chosen per research
R3: "Try on N moments", default 10). Returns the entry at once with `status: running` and
`moments` filled; answers arrive in the background (research R4) and jump ahead of any run on
the same model (R5). Records a `try` use in the prompt library.

Errors: `404 VIDEO_NOT_FOUND`, `403 PATH_NOT_ALLOWED` (not under the allowed roots or uploads,
and not an earlier run's video; any signed-in user may use any of those),
`422 INVALID_SAMPLING`, `409 REMOTE_NOT_CONFIRMED`, `503 OLLAMA_UNREACHABLE` (with the
"what's missing" hint of US1 scenario 4).

## History

### `GET /ask/entries?video_path=&mine=&starred=&limit=&before=`

```json
{ "entries": [Entry, …],
  "library": [{ "sha256": "…", "text": "…", "name": null, "models": ["…"],
                "last_used_at": "…" }],
  "next_before": "2026-10-18T10:00:00Z" }
```

Entries newest first (starred first when `starred` isn't set). `library` lists prompts from the
prompt library that have no entry, so their records still appear (FR-006); empty when filtered by
video. An Entry includes `prompt` (the text), `user_id`, and for runs `job_id` and the job's
status and progress.

### `POST /ask/drafts` → `201 Entry`

Body as `POST /ask/tries` without `at`/`count`. Keeps a question that was never tried
(`kind: draft`, no answers) before another entry is restored over it (US2 scenario 2).

### `GET /ask/entries/{id}` → `Entry`

Polled every second while a try is `running`.

### `PATCH /ask/entries/{id}`

Body: any of `name`, `starred` → `Entry`.

### `DELETE /ask/entries/{id}` → `204`

Author or admin (`403` otherwise). A run's entry only; the run stays.

### `GET /ask/entries/{id}/export/table`

The spec 025 table (CSV) of a try's answers, same columns as a run's table (research R16). For a
run's entry, redirects (`307`) to `/jobs/{job_id}/export/table`.

## Runs

### `POST /ask/runs` → `201 Entry`

Body as `POST /ask/tries` without `at`/`count`. Creates an ordinary job (pipeline
`vlm_annotation`, config from the sampling per [pipeline-config.md](pipeline-config.md)) and its
history entry. Same errors as tries.

### `GET /ask/runs/{job_id}/answers?after=N`

```json
{ "status": "running", "planned": 40, "done": 23, "stopped_early": false,
  "answers": [Answer, …],  "skipped": [{ "t": 35.0, "reason": "…" }] }
```

Answers with index ≥ `after`, read from the run's results file (research R6). Polled every 2 s
while running. Stop is the existing `POST /jobs/{job_id}/cancel` (research R7).

## Frames

### `GET /ask/frames?path=&t=&interval=&frames=&spacing=`

The frames one sample at `t` would send, without calling a model (FR-011, R13):

```json
{ "t": 12.0, "frames": [{ "frame_number": 288, "timestamp_sec": 11.5,
                           "jpeg_base64": "…" }, …], "requested": 3 }
```

`requested` > `len(frames)` when the burst reaches past either end.

## Videos (shared with spec 028)

### `POST /videos` (multipart `video`) → `201`

```json
{ "path": "/…/storage/videos/3fa2c1d0e9b84f7a/p01_t1.mp4", "sha256": "…",
  "name": "p01_t1.mp4", "duration_sec": 132.4 }
```

Stored once per content (research R12).

### `POST /videos/stream-token`

Body `{ "path": "…" }` → `{ "token": "…", "expires_at": "…", "path": "…", "name": "…",
"duration_sec": 132.4 }` (the duration from the server's decoder, so the timeline draws even when
the browser can't play the codec). Path must be readable under the
rules above.

### `GET /videos/stream?path=&token=` (no API key; the token is the credential)

The file with range requests. `401 TOKEN_INVALID` when the token is wrong, expired or for
another path.

Picking a server folder uses the existing `GET /ingest/browse` (admin, same machine), unchanged.

## `POST /vlm/preview` (changed, additive)

Accepts `burst_spacing_sec`; now runs off the event loop. Otherwise unchanged.
