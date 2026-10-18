# Contract: Export API

All routes need the same API key as the job/batch/dataset routes they sit beside
(`validate_api_key`); no admin requirement. Work runs in a worker thread.

## Table

```
GET /api/v1/jobs/{job_id}/export/table?format=csv|parquet
GET /api/v1/batches/{batch_id}/export/table?format=csv|parquet
GET /api/v1/datasets/{dataset_id}/export/table?format=csv|parquet
```

- `format` defaults to `csv`.
- 200: the file. `Content-Type: text/csv; charset=utf-8` or `application/vnd.apache.parquet`.
  `Content-Disposition: attachment` with filename `<run or video name>.csv|.parquet` (dataset:
  `<dataset name>.csv`), RFC 5987 `filename*` for non-ASCII.
- Header `X-VideoAnnotator-Export-Summary`: the export summary (data-model.md), ASCII JSON, at most
  8 KB (lists cut and `"truncated": true` past that). Listed in `Access-Control-Expose-Headers`.
- 404 `JOB_NOT_FOUND` / `BATCH_NOT_FOUND` / `DATASET_NOT_FOUND`.
- 409 `NOTHING_TO_EXPORT`: no result files at all; `message` says why (still running, failed with
  `<error>`, no runs of this dataset).
- 422 `PARQUET_NOT_INSTALLED`: `hint` = `pip install videoannotator[parquet]`, `details.extra` =
  `parquet` (the viewer offers spec 005's install with it).
- 422 for an unknown `format`.

## ELAN

```
GET /api/v1/jobs/{job_id}/export/elan       → one .eaf (application/xml), filename <video>.eaf
GET /api/v1/batches/{batch_id}/export/elan  → zip of <run>/<video>.eaf, filename <run> ELAN.zip
```

Same summary header (with `left_out`) and errors (no Parquet error).

## Methods

```
GET /api/v1/jobs/{job_id}/methods
GET /api/v1/batches/{batch_id}/methods
```

200 `application/json`:

```json
{
  "prose": "Videos were annotated with VideoAnnotator 1.6.0 [@addyman2026videoannotator]. …",
  "bibtex": "@software{addyman2026videoannotator, …}\n\n@inproceedings{radford2023whisper, …}",
  "prompts": [{"sha256": "9f2c…", "text": "Classify as TOUCH only if: …"}],
  "summary": {"videos": 40, "groups": 1, "not_recorded": []}
}
```

`?format=markdown` returns `text/markdown` (the CLI's output) as an attachment `methods.md`.
Errors as for the table.
