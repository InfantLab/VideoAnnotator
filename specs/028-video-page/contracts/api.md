# Contract: API changes for the Video page

All endpoints need the API key (`Authorization: Bearer`), except `GET /videos/stream`, whose
credential is its token (spec 027). Every signed-in user may use them; none is admin-only.
Errors use the existing `APIError` shape (`code`, `message`, `hint`).

## `GET /api/v1/videos/runs?path=<server path>`

Everything the Video page needs to draw its lanes for one video (research R2).

**200**

```json
{
  "path": "/data/pk/p01_t1.mp4",
  "name": "p01_t1.mp4",
  "available": true,
  "unavailable_reason": null,
  "duration_sec": 200.4,
  "sha256": "9f2c…",
  "runs": [
    {
      "job_id": "0b7e…",
      "status": "completed",
      "created_at": "2026-10-12T09:14:00Z",
      "completed_at": "2026-10-12T09:31:00Z",
      "batch_id": "a1…",
      "dataset_id": "5c…",
      "run_name": "Peekaboo study (2026-10-12)",
      "pipelines": [
        {
          "name": "face_analysis",
          "status": "completed",
          "error": null,
          "files": [
            {"name": "p01_t1_face_detections.json", "kind": "result",
             "download_url": "/api/v1/jobs/0b7e…/results/files/face_analysis?name=p01_t1_face_detections.json"},
            {"name": "p01_t1_face_detections.json.provenance.json", "kind": "provenance",
             "download_url": "…"}
          ]
        }
      ]
    }
  ],
  "projects": [{"id": "5c…", "name": "Peekaboo study"}]
}
```

- `runs` newest first; jobs in any status (a running job shows with its pipelines so far).
- `available: false` with `unavailable_reason` when the file is gone; `runs` still listed.
- `sha256`: from a run's provenance (`input.sha256`) or an upload; null when unknown. Never
  computed by this request.
- **403 `PATH_NOT_ALLOWED`**: the path is not one `readable_video` allows *and* no job read it.
  **422 `VIDEO_PATH_REQUIRED`**: empty path. A path that no job read but is readable returns 200
  with `runs: []` (a video with no results yet).

## Notes: `/api/v1/videos/notes`

### `GET /api/v1/videos/notes?path=<server path>`

**200** `{"notes": [Note, …]}`, ordered by `start_sec`, then `created_at`.

```json
{
  "id": "e3…",
  "video_path": "/data/pk/p01_t1.mp4",
  "start_sec": 12.4,
  "end_sec": null,
  "text": "infant looks away",
  "author_id": "u1",
  "author_name": "Caspar",
  "created_at": "2026-10-19T10:02:00Z",
  "updated_at": null,
  "updated_by": null
}
```

### `POST /api/v1/videos/notes`

Body `{"path": "...", "start_sec": 12.4, "end_sec": null, "text": "infant looks away"}`.
**201** the Note. The server resolves `path` (same rule as `/videos/runs`), fills
`video_name`, `video_sha256` (when known) and the author.

- **422 `NOTE_TEXT_REQUIRED`**: text empty after trimming. **422 `NOTE_TEXT_TOO_LONG`**: over
  2,000 characters. **422 `NOTE_TIME_INVALID`**: `start_sec` < 0, or `end_sec` ≤ `start_sec`
  after ordering, or beyond the duration when known.

### `PATCH /api/v1/videos/notes/{id}`

Any of `text`, `start_sec`, `end_sec` (null makes it a moment). **200** the Note, with
`updated_at`/`updated_by`. Same 422s. **404 `NOTE_NOT_FOUND`**.

### `DELETE /api/v1/videos/notes/{id}`

**204**. **404 `NOTE_NOT_FOUND`**.

## Exports (spec 025), additive

- `GET /jobs/{id}/export/table`, `/batches/{id}/export/table`, `/datasets/{id}/export/table`:
  table format version 2 ([table-v2.md](table-v2.md)); each video's notes are included as
  `note` rows.
- `GET /jobs/{id}/export/elan`, `/batches/{id}/export/elan`: a `Notes (researcher)` tier per
  video with notes.
- The export summary (`X-Export-Summary` / Parquet metadata) gains `notes: <count>` and, when
  the database can't be read, the line "notes not included: no database".

## Unchanged, relied on

- `POST /videos/stream-token`, `GET /videos/stream` (spec 027).
- `GET /jobs/{id}/results/files/{pipeline}?name=` (per-file download).
- `GET /jobs/{id}` (for the `/view/:jobId` redirect), `GET /batches/{id}`,
  `GET /datasets/{id}`, `GET /ingest/browse` (for previous/next lists).
