# Data Model: Saved Datasets in the Viewer

## SavedDataset (exists; extended)

| Field | Type | New | Notes |
|---|---|---|---|
| id, name, description, owner_user_id, created_at, updated_at, last_used_at | | | spec 007 |
| video_manifest | VideoManifestEntry[] | | |
| server_folder | str \| null | yes | a folder the server can read; null for uploaded-file datasets |
| server_folder_recursive | bool | yes | default false |

Response adds `owner_name` (username, else email; null if the user is gone).

## VideoManifestEntry (exists; extended)

| Field | Type | New |
|---|---|---|
| filename | str | |
| size_bytes | int \| null | |
| last_seen_at | datetime \| null | |
| relative_path | str \| null | yes |

## DatasetMatch (viewer only)

```ts
interface DatasetMatch {
  matched: Array<{ entry: ManifestEntry; file: File }>;
  missing: ManifestEntry[];       // in the dataset, not found
  added: File[];                  // found, not in the dataset
  changed: Array<{ entry: ManifestEntry; file: File }>;  // same path/name, other size
  ambiguous: ManifestEntry[];     // several files fit and no path to choose
}
```

## Folder handle (viewer only)

IndexedDB `datasets.handle.<dataset id>` → `FileSystemDirectoryHandle`. Per browser; never sent to
the server.
