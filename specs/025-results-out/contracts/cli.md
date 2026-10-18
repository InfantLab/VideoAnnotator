# Contract: `videoannotator export`

Works on any folder of results, with no server and no network (FR-009). `FOLDER` may be a video's
folder, a run's folder, a folder of runs, an extracted results download, or a pre-022 job folder.

```
videoannotator export table FOLDER [--format csv|parquet] [--output PATH] [--json]
videoannotator export elan FOLDER [--output DIR] [--json]
videoannotator export methods FOLDER [--output PATH] [--json]
```

| Command | Default output |
|---|---|
| `table` | `FOLDER/<FOLDER name>.csv` (or `.parquet`) |
| `elan` | one `<video>.eaf` in each video's folder (inside a run), else beside the results |
| `methods` | printed to stdout as Markdown; `--output` writes it to a file |

- `--output -` writes the table/methods to stdout.
- After writing, prints one summary line (`1,284 rows, 12 videos, 5 pipelines → <path>`), then any
  skipped files and videos without results, one per line, to stderr.
- `--json`: prints the export summary (data-model.md) as JSON on stdout instead, plus `"output":
  "<path>"` (for agents, roadmap Phase 4).
- Exit 0 on success, even with skipped files; 1 when there is nothing to export or the folder doesn't
  exist; 2 for usage errors; 3 when Parquet is asked for but not installed (message gives
  `pip install videoannotator[parquet]`).
- Never overwrites an existing export file silently: an existing default output is replaced only
  when it is itself an export (same name), which is the normal "export again" case; `--output` to an
  existing non-export file fails with exit 1.
- Read-only folder: fails with a message suggesting `--output`.
