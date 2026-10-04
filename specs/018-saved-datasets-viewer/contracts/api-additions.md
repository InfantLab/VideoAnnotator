# Contract: API additions for saved datasets

All additive; existing clients and exports keep working.

## Datasets (`/api/v1/datasets`)

Request and response bodies gain optional fields:

```json
{
  "name": "Peekaboo study",
  "description": "Session 1",
  "server_folder": "/data/peekaboo/session1",
  "server_folder_recursive": true,
  "video_manifest": [
    { "filename": "p01.mp4", "size_bytes": 1048576, "relative_path": "p01/p01.mp4" }
  ]
}
```

Responses also carry `owner_name` (string or null). A duplicate `(owner, name)` on create or
rename returns 409 `DATASET_NAME_CONFLICT` (exists since spec 007).

## Scan a server folder

`GET /api/v1/ingest/scan?path=<folder>&recursive=<bool>`, with the same rules as
`POST /api/v1/ingest`: admin, local caller, and a path inside the allowed roots.

```json
{ "path": "/data/peekaboo/session1", "recursive": true,
  "videos": [ { "relative_path": "p01/p01.mp4", "name": "p01.mp4", "size_bytes": 1048576 } ] }
```

Errors: as `browse` (403 not local/admin, 400 outside roots, 404 not a folder).

## CLI

```
videoannotator dataset list [--server URL]
videoannotator dataset show <id>
videoannotator dataset export <id> [-o file.json]
videoannotator dataset import <file.json>
videoannotator dataset delete <id> [--yes]
```
