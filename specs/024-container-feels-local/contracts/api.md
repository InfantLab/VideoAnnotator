# API Contract: A Container That Feels Local

Additive (constitution, Principle V).

## Changed (additive): `GET /api/v1/ingest/access`

New fields: `in_container`, `managed_by_launcher`, `shares` (see data-model.md).

```json
{
  "same_machine": true,
  "can_read_in_place": true,
  "reason": null,
  "in_container": true,
  "managed_by_launcher": true,
  "shares": [{"path": "/c/Users/ada/Studies", "display_path": "C:\\Users\\ada\\Studies", "present": true, "stop_requested": false}],
  "allowed_folders": [{"path": "/c/Users/ada/Studies", "display_path": "C:\\Users\\ada\\Studies"}],
  "places": [{"label": "Studies", "path": "/c/Users/ada/Studies", "display_path": "C:\\Users\\ada\\Studies", "has_videos": true}],
  "results_root": {"path": "/c/Users/ada/VideoAnnotator", "display_path": "C:\\Users\\ada\\VideoAnnotator"},
  "can_open_folders": false
}
```

**Changed behaviour**: in a container with no shared folder, `allowed_folders`, `places` and
`shares` are empty (never the container's home), and `reason` is:

- With the launcher: "VideoAnnotator can only see folders you share with it. To share one, run:
  videoannotator-start share".
- Without it (compose): "VideoAnnotator can only see folders you share with it. Set VIDEOS_DIR
  when starting it (see the installation guide)." Compose reaches this state when `VIDEOS_DIR`
  is unset, because it then sets `VIDEOANNOTATOR_INGEST_ROOTS` empty (research.md R8).

## New: `POST /api/v1/ingest/shares/stop`

```json
{"path": "/c/Users/ada/Studies"}
```

- Same machine and administrator only (as ingest): 403 otherwise.
- 404 `SHARE_NOT_FOUND` if the path isn't a current share.
- 409 `NOT_MANAGED_BY_LAUNCHER` without the launcher (`VIDEOANNOTATOR_LAUNCHER` unset).
- 200 with the share, now `"stop_requested": true`. Takes effect when VideoAnnotator next starts.
- Appends the host path to `stop-sharing.txt` once (a repeated request doesn't duplicate it), and
  chowns the file to `VIDEOANNOTATOR_RESULTS_OWNER` when that is set.

## Changed messages

Wherever a video is unavailable (job start failure, `video_available` explanation, rerun
`skipped` reasons, dataset differences), a video outside every current shared folder says
"<folder> isn't shared with VideoAnnotator any more" instead of "moved or deleted".
