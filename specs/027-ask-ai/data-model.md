# Data Model: Ask AI

**Spec**: [spec.md](spec.md) | **Research**: [research.md](research.md)

Two new tables in the server database (`database/models.py`, created by `create_all`; nothing
existing is altered), additive pipeline config keys, and additive provenance fields.

## `ask_entries` (History entry: a try or a run)

| Column | Type | Notes |
|---|---|---|
| `id` | GUID, PK | |
| `kind` | str(8) | `try` \| `run` |
| `status` | str(12) | `running` \| `done` \| `stopped` \| `failed` (tries); runs mirror their job |
| `prompt_sha256` | str(64), FK `prompts.sha256` | the question; text lives once in `prompts` |
| `model` | str(200) | |
| `base_url` | str(500) | as resolved by the server |
| `endpoint_class` | str(8) | `local` \| `remote`, at the time of the try or run |
| `sampling` | JSON | see Sampling below |
| `think` | bool | |
| `video_path` | str(1000) | server path the video was read from |
| `video_name` | str(255) | file name, for listing and filtering |
| `video_sha256` | str(64), nullable | when known (uploads; runs' provenance) |
| `moments` | JSON list[float] | sample times tried (tries); null for runs |
| `answers` | JSON list[Answer] | tries only; filled as each lands; null for runs |
| `job_id` | str(64), nullable, indexed | runs only |
| `user_id` | str(64), nullable | who made it |
| `name` | str(200), nullable | |
| `starred` | bool, default false | |
| `created_at`, `finished_at` | datetime | |

Rules:
- Every try and run creates its entry **before** the first model call (SC-002).
- A try's `answers` has one item per moment once it finishes; while running, the answered ones.
- Nothing in the entry is an image (clarification 3).
- Shared by everyone on the server; filtered to the caller's own with `mine=true`
  (clarification 4). Name and star are shared.
- Restoring an entry copies question, model and sampling into the window; it doesn't change the
  entry.
- An entry is deleted only by its author or an admin; deleting a run's entry doesn't delete the
  run.

State: `running → done | stopped | failed`. A server restart marks tries still `running` as
`stopped`, keeping their answers.

### Answer (item in `answers`; also the shape the run-answers endpoint returns)

| Field | Type | Notes |
|---|---|---|
| `t` | float | sample time, s |
| `frame_numbers` | list[int] | frames sent; empty when skipped |
| `frame_times` | list[float] | their times |
| `status` | str | `answered` \| `error` \| `skipped` (Principle VI: never folded into an answer) |
| `label` | str \| null | parsed answer (`_parse_label`) |
| `reasoning` | str \| null | |
| `raw_response` | str \| null | |
| `error` | str \| null | |
| `total_time` | float \| null | seconds; feeds the estimate |

## `endpoint_confirmations`

| Column | Type | Notes |
|---|---|---|
| `base_url` | str(500), PK | normalised (scheme, host, port; no trailing slash) |
| `confirmed_by` | str(64) | user id |
| `confirmed_at` | datetime | |

## Sampling (in `ask_entries.sampling`, and as pipeline config)

| Field | Pipeline key | Default | Validation |
|---|---|---|---|
| interval | `frame_interval_sec` | 5.0 | > 0 |
| start | `start_sec` | 0.0 | 0 ≤ start < duration |
| end | `end_sec` | null (end) | start < end ≤ duration |
| moments | `moments_sec` | null | each 0 ≤ t ≤ duration; sorted, de-duplicated |
| frames per sample | `sampling_mode` + `burst_offsets` | 1 (`single_frame`) | 1–9 |
| frame spacing | `burst_spacing_sec` | null (= interval) | > 0 |

Sample points: `moments` when set; otherwise `start, start + interval, …` up to `end`. Burst
frames: `anchor + offset × spacing`, dropping any outside the video (edge case: the answer says
how many were sent).

## Pipeline output (unchanged format, additive fields)

`<stem>_vlm_annotation.json` stays COCO, with the same annotations as today (one per sample
that reached the model; a model error is still an annotation labelled `ERROR: …`). Additive:

- per annotation: `frame_times` (list[float]) and `status` (`answered` | `error`);
- top level: `vlm_skipped`, a list of `{t, reason}` for sample points whose frames couldn't be
  read (Principle VI: listed, not answered), and `vlm_progress`
  `{planned, done, finished, stopped_early}` while and after the run.

Existing readers see the same annotations as before (Principle V). The file is rewritten as
answers arrive (research R6).

## Provenance (spec 017 record, `vlm` block, additive)

`endpoint_class`, `endpoint_host`, `samples_planned`, `samples_answered`, `stopped_early`.

## Prompt library (`prompt_uses`)

New `kind` value `try` beside `job` and `preview`. No schema change.
