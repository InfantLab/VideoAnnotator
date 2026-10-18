# Research: Results Out

Decisions made while planning spec 025, with what was considered. Inputs: the committed demo-clip
outputs (`tests/fixtures/viewer_contract/`), spec 017's provenance contract, spec 022's results
folder (`results_folder.py`), the job/batch/dataset API, and the viewer's ELAN reader.

## 1. One reader over files, for every entry point

**Decision**: every export (table, ELAN, methods) is built from the **files on disk** and the
provenance records in them, through one module (`exporters/results_set.py`). The API resolves a
job, run or dataset to a list of folders and hands them to the same code the CLI uses on a folder.

**Rationale**: FR-010 (same rows from CLI, API and viewer) holds by construction. Offline export of
a copied results folder (FR-009) is the same code path, not a second implementation. The job record
in the database adds nothing the files lack: since spec 017 every file names its pipeline, models,
settings and input.

**Alternatives**: building from `job.pipeline_results` in the database. Rejected: the CLI on a
copied folder has no database, and two sources would drift.

## 2. Which file is which

**Decision**: a file is recognised by the suffix after the video stem (`<stem>_<suffix>`), from a
fixed reader table:

| Suffix | Reader | Default pipeline (no provenance) |
|---|---|---|
| `person_tracking.json` | COCO person | `person_tracking` |
| `face_detections.json` | COCO face (DeepFace) | `face_analysis` |
| `openface3_analysis.json` | COCO face (OpenFace 3) | `face_openface3_embedding` |
| `scene_detection.json` | COCO scene | `scene_detection` |
| `vlm_annotation.json` | COCO VLM | `vlm_annotation` |
| `speaker_diarization.rttm` | RTTM | `speaker_diarization` |
| `speech_recognition.vtt` | WebVTT | `speech_recognition` |

Known but not read, because they restate facts another file holds (spec Assumptions):
`person_tracks.json` (labels already on each detection), `openface3_detailed.json` (same features
as the analysis file), `*.provenance.json` companions (read as provenance, not as results).

Ignored silently (they are not results): videos, `run.json`, `job_metadata.json`, a bare
`<pipeline>.json` (the file storage backend's copy in pre-022 job folders), and this spec's own
outputs (`*.csv`, `*.parquet`, `*.eaf`, `methods*.md`/`.bib`). Anything else is listed as skipped
(FR-013).

The `pipeline` column comes from the provenance record's `pipeline.name` when present (so
`audio_processing`'s two files say `audio_processing`), otherwise from the table above.

**Alternatives**: reading `registry/metadata/*.yaml` `outputs[].file` to find files. Kept for the
API's per-job listing (`pipeline_result_files`), but a reader still has to know each format's
fields, so the reader table is the real contract; a plugin's file is "skipped, no reader" until it
ships one (out of scope here).

## 3. Table layout: long, one row per measure

**Decision**: long layout. Columns, in order (contract: `contracts/table.md`):
`run, video, pipeline, event, track, start, end, frame, measure, label, value, confidence, x, y,
width, height, model, videoannotator_version, file`. Box columns are filled on the row that states
what an event is (`person`, `face`); attribute rows leave them empty.

Multi-component measures use `label` for the component and `value` for the number: `measure =
action_unit, label = AU12_Lip_Corner_Puller, value = 2.1`; `measure = emotion_score, label =
happy, value = 0.82`. One pattern for OpenFace AUs, head pose, gaze, emotion scores and scene
scores, so a new component never adds a column.

**Rationale**: a wide table would need one column per AU, emotion, scene class and so on, mostly
empty, and change shape whenever a pipeline or model changes. Long tables are what `tidyr` and
`pandas` expect for grouping and plotting; `pivot_wider` / `pivot` produce the wide view in one call
(the notebooks show it).

**Alternatives**: one file per pipeline (rejected: back to many parsers, if simpler ones); JSON
lines (rejected: R users read CSV first).

## 4. `event` and `track`

**Decision**: `event` is an integer numbering events within one `(run, video, pipeline)`, in reading
order (file order, then the file's own order). `track` is filled only when the pipeline really
follows something over time: `person_id` for person tracking, the speaker label for diarization,
`track_id` for OpenFace when present. DeepFace's `face_id` is the face's index within its frame, not
a track, so it is not used as one.

**Rationale**: annotation ids collide across `audio_processing`'s two files; a per-pipeline counter
can't. Putting a per-frame index in `track` would invite analyses that treat it as identity.

## 5. Time

**Decision**: seconds from the start of the video, as the file states them, unrounded: COCO
`timestamp` (OpenFace: `openface3.timestamp`; VLM: `timestamp_sec`), scene `start_time`/`end_time`,
RTTM start and start + duration, WebVTT cue times. `end` is empty for observations at one moment
(FR-003). `frame` from `frame_number` where recorded.

## 6. `model` and `videoannotator_version`

**Decision**: from the provenance record: `model` joins the record's model names with ` + `
(`RetinaFace + STAR 98_point + MTL backbone`); revisions are in the methods text and the Parquet
metadata, not repeated on every row. With no record, both read `not recorded` (FR-005), with one
exception: VLM annotations carry their own `model` field per annotation (written by the pipeline,
not inferred), which is used when the file has no record.

## 7. CSV and Parquet

**Decision**: CSV with the standard library's `csv` writer (UTF-8, no BOM, header row, `\n` line
ends, minimal quoting, empty string for missing, numbers in Python's shortest round-trip form).
Parquet through `pyarrow`, in a new optional extras group `parquet = ["pyarrow>=17"]`, included in
`all`. Without it, Parquet requests fail with `PARQUET_NOT_INSTALLED` and the hint
`pip install videoannotator[parquet]`; because spec 005 installs any declared extras group, an admin
can install it from the viewer with the existing flow. The Parquet file's key-value metadata holds
`videoannotator.table_format_version` and the export summary, including every source file's
provenance record.

**Rationale**: pyarrow is tens of MB; Principle IV keeps the core slim, and CI's core-only job
proves the exporters import without it. CSV needs nothing and is what R's `read.csv` and pandas'
`read_csv` read without options (tested with a transcript containing a comma, quote and newline).

**Alternatives**: `polars` (also heavy; less familiar to the target users); writing Parquet by hand
(no).

## 8. Large runs

**Decision**: rows are generated lazily, file by file, and written straight to a temporary file;
the API sends it with `FileResponse` and deletes it afterwards. The work runs in a worker thread
(`run_in_threadpool`), so the event loop keeps answering status polls. Sort order (FR-015: video,
start, pipeline, event, measure) is applied per video, which is all that's in memory at once.

**Rationale**: OpenFace at 1 fps on 100 ten-minute videos is about a million rows; holding them all
is unnecessary when the order key starts with `video`. SC-003 (100 videos in under a minute) is
dominated by JSON parsing, which this doesn't change.

## 9. ELAN

**Decision**: EAF 3.0 written with `xml.etree.ElementTree` from the table's rows, so ELAN and the
table can't disagree (SC: every annotation matches a row).

- One `.eaf` per video; one `LINGUISTIC_TYPE` (`videoannotator`, time-alignable, no constraints).
- Tiers: `<pipeline> <track or measure>`: `speaker_diarization SPEAKER_00`, `speech_recognition`,
  `scene_detection`, `person_tracking person_demo_clip_001 (parent)`, `face_analysis faces`,
  `vlm_annotation`. Values: the row's `label` (speech text, scene type, VLM label, person label) or
  `in view`.
- Frame detections → "in view" intervals per track: consecutive detections merge while the gap is at
  most twice the median spacing of that track's detections (≈ twice the sampling interval); each
  interval runs from its first detection to the last one plus one spacing. Untracked faces (DeepFace)
  form one `faces` tier: "a face is in view".
- VLM labels: each sample lasts until the next sample's time, the last one to the video's end when
  known (from the run's `duration` if recorded, else last sample + interval). Consecutive equal labels
  merge.
- Continuous measures (AUs, gaze, pose, scores, age) are not written (FR-020); the summary says so.
- `HEADER/MEDIA_DESCRIPTOR`: `MEDIA_URL` = `file://` URI of the video path recorded in `run.json`
  (`source.path`) or the provenance `input.name`, `RELATIVE_MEDIA_URL` to the same when it is
  relative-able; `MIME_TYPE` from the extension. ELAN prompts for a moved file itself.
- `TIME_UNITS="milliseconds"`, integer `TIME_VALUE`s (rounded to the nearest ms).
- Validation: the EAF 3.0 schema (`https://www.mpi.nl/tools/elan/EAFv3.0.xsd`, downloaded
  2026-10-16, 34 KB) is committed under `tests/fixtures/elan/` and checked with `xmlschema` (pure
  Python, added to the dev group).
- Viewer: its parser already reads `TIME_ORDER`/`TIER`/`ALIGNABLE_ANNOTATION` and draws raw tier
  segments on the timeline (`Timeline.tsx`, "ELAN Ground Truth Track"). A committed exported `.eaf`
  for the demo clip, plus a copy with an extra hand-coded tier (as ELAN saves it), go into the
  viewer contract test.

**Alternatives**: `pympi-ling` (an extra dependency for ~150 lines of XML); one tier per pipeline
with track in the value (rejected: ELAN users code per tier).

## 10. Methods paragraph

**Decision**: built from provenance records plus registry metadata:

- **Citations** live in each pipeline's metadata YAML under a new optional key `citations:` (list of
  `{key, bibtex}`; additive, allowed within a minor version by Principle II). VideoAnnotator's own
  entry is a package constant (`exporters/citation.py`), kept in step with `CITATION.cff` by a unit
  test (title, authors, repository). The version comes from the records, not the running code.
- **Settings**: a record's `settings` compared with the pipeline's `config_schema` defaults; only
  differing keys are named, minus keys that don't change results (`device`, `cache_dir`,
  `model_path`, `base_url`, `use_auth_token`, `*_dir`, `*_path`, timeouts, `batch_size`). Keys not in
  the schema are skipped (internal).
- **Grouping** (FR-025): each video's per-pipeline signature is (version, models with revisions,
  differing settings, deterministic). The largest group is described in full ("Of 40 videos, …"); every
  other group is described by what differs and its count.
- **VLM**: model name, `model_digest`, quantisation, `prompt_sha256` (first 12 hex in prose, full in
  the prompt list); the full prompt texts are returned beside the prose (from `settings.prompt`, or
  the annotations' `prompt` field).
- **Output**: `{prose, bibtex, prompts[], summary}` as JSON from the API; Markdown (`prose` then a
  fenced BibTeX block, then prompts) from the CLI and the viewer's download.
- No video names, paths, job ids, base URLs or secrets in the prose (FR-029); `settings` are already
  redacted by spec 017.

Citations chosen (checked 2026-10-16 against the projects' own pages):

| Pipeline | Cite |
|---|---|
| person_tracking | Ultralytics YOLO11 (Jocher & Qiu 2024, software, AGPL-3.0); ByteTrack (Zhang et al., ECCV 2022) |
| face_analysis | DeepFace as its README asks (Serengil & Ozpinar, Gazi Univ. J. Sci. 39(1), 2026, doi:10.35378/gujs.1794891); RetinaFace (Deng et al., CVPR 2020) |
| face_openface3_embedding | OpenFace 3.0 (Hu, Mathur, Liang & Morency, IEEE FG 2025, arXiv:2506.02891) |
| scene_detection | PySceneDetect (Castellano, software); CLIP (Radford et al., ICML 2021); OpenCLIP (Ilharco et al. 2021, doi:10.5281/zenodo.5143773); LAION-5B (Schuhmann et al., NeurIPS 2022) |
| speaker_diarization | Plaquet & Bredin, Interspeech 2023; Bredin, Interspeech 2023 (the model card's two entries) |
| speech_recognition | Whisper (Radford et al., ICML 2023) |
| audio_processing | the union of the two above |
| vlm_annotation | Ollama (software); the model itself has no fixed citation, so the prose asks the author to cite the model named |

## 11. Interfaces

**Decision**:

- API (`contracts/api.md`): `GET /api/v1/{jobs|batches|datasets}/{id}/export/table?format=csv|parquet`,
  `GET /api/v1/{jobs|batches}/{id}/export/elan` (job: one `.eaf`; run: a zip),
  `GET /api/v1/{jobs|batches}/{id}/methods`. The export summary travels in an
  `X-VideoAnnotator-Export-Summary` header (ASCII JSON, at most 8 KB, `truncated: true` past that) and
  is also exposed through `Access-Control-Expose-Headers` for the viewer.
- CLI (`contracts/cli.md`): `videoannotator export table|elan|methods <folder>`. Works on any folder
  with no server (FR-009, FR-023). The viewer already shows each run's and job's folder path with a
  Copy button (spec 022), which is the bridge from a job to a folder.
- Dataset export: jobs carry `dataset_id` already; storage backends gain
  `list_jobs_by_dataset(dataset_id)`.

## 12. Notebooks

**Decision**: `examples/notebooks/`:

- `results_table_python.ipynb` (pandas, matplotlib), executed in CI by a pytest that runs its code
  cells in order with `MPLBACKEND=Agg` (no Jupyter dependency needed; pandas and matplotlib are
  present with `--all-extras`; skipped when missing).
- `results_table_r.Rmd` (R Markdown with readr, dplyr, tidyr, ggplot2): the format R users open in
  RStudio. Run before each release (`tests/manual/`), as CI has no R.
- `demo_clip_table.csv`: the table export of `tests/fixtures/viewer_contract/` (renamed into a
  run/video layout), regenerated by a script; a test fails when it is stale.
- Analyses: speaking time per speaker; speech segments; who is in view when (person tracks);
  emotion over time (face rows); overlap of "infant in view" with speech; pivot to wide.

## 13. Viewer

**Decision**: one `ExportMenu` component (dropdown: Table (CSV), Table (Parquet), ELAN files,
Methods paragraph) beside `ResultsLocation`'s "Download results" on `JobDetail` and `BatchDetail`;
the saved dataset card gets a "Table (CSV)" action. `MethodsDialog` shows prose and BibTeX with Copy
buttons and a Markdown download. Parquet unavailable → the item shows the hint (and the existing
install flow for admins, `lib/pipelineExtras.ts`). The summary header's skipped list is shown as a
toast with details when non-empty. All code-split behind the dialog; no new dependency.
