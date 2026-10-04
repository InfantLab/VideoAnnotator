# Research: Saved Datasets in the Viewer

## R1. What the server already has (spec 007)

`/api/v1/datasets` CRUD: `name`, `description`, `owner_user_id`, `video_manifest`
(`filename`, `size_bytes`, `last_seen_at`), timestamps. Shared read; owner (or admin) edits and
deletes; `(owner, name)` unique. Import = `POST` of an export; export = `GET`. Jobs carry
`dataset_id` (spec 008) and `touch_last_used` runs on ingest.

## R2. Backend additions (both optional; existing data and exports stay valid)

- `SavedDataset.server_folder: str | None` and `server_folder_recursive: bool` (default false).
- `VideoManifestEntry.relative_path: str | None` (path within the chosen folder).
- `DatasetResponse.owner_name` (username, else email) so "who saved it" needs no extra call.
- Columns added with the users-database layer's existing `ensure_columns` additive migration
  (`database/migrations.py`).
- `GET /api/v1/ingest/scan?path=&recursive=` lists a server folder's videos (relative path,
  name, size) via ingest's existing `find_videos`, with the same admin + local-caller + allowed
  roots checks as ingest. Used to save a server-folder dataset and to detect drift before a run.

**Alternatives**: computing a server folder's manifest in the browser via repeated `browse`
calls (one level each): slow and racy for deep trees.

## R3. Re-finding uploaded files

- On save from the wizard, if the files came from a folder pick, the viewer stores that
  `FileSystemDirectoryHandle` in IndexedDB under `datasets.handle.<dataset id>` (same mechanism
  as the library root handle, `lib/persistence/idbKv`).
- On use: get the handle → `queryPermission`/`requestPermission({mode:'read'})` (needs the user
  click that chose the dataset) → walk it → match. No handle, unsupported browser, or permission
  refused → ask the user to pick the folder (`<input webkitdirectory>` fallback) → match.
- Matching (pure function, `lib/datasetMatch.ts`): key = relative path when both sides have one,
  else file name; equal size = match; same key different size = changed; manifest-only = missing;
  folder-only = added; several candidates for one name and no path = ambiguous.

## R4. Name clashes on import

The server returns 409 on a duplicate `(owner, name)`. The viewer retries with
`"<name> (imported)"`, then `"(imported 2)"`… up to 20 tries, and says which name it used.

## R5. Where things go in the viewer

- `pages/Datasets.tsx` (new, real), route `/datasets` (replaces the Library redirect), nav link.
- `components/datasets/` list, detail/edit, import/export buttons; `components/DatasetPicker.tsx`
  for the wizard's "Use a saved dataset"; `components/DatasetDriftDialog.tsx`.
- `lib/datasetMatch.ts`, `lib/datasetHandles.ts`.
- `api/client.ts`: dataset methods, `scanServerFolder`.
- `PresetBar.tsx`: Export / Import.

## R6. CLI

`videoannotator dataset list|show|export|import|delete` call the API like `videoannotator job …`
(server URL option, API key from the same place `job` commands take it).
