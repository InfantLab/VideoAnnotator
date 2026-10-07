# Research: Output Provenance and Overlay Attribution

## R1. Where provenance is written: per writer, or stamped centrally

**Decision**: stamp centrally. After a pipeline's `process()` returns, `batch/job_execution.py`
adds the provenance record to each file the pipeline wrote, found by its registry
`outputs[].file` suffixes (the lookup `batch/result_files.py` already does for downloads).

**Rationale**: one place to get right; covers every pipeline (and future plugins, Principle IV)
without touching eight writers; `videoannotator process` gets it for free because it runs the same
job path.

**Alternatives**: each writer embeds `self.provenance` (eight code paths, easy to miss one, and
plugins would each have to do it); a single job-level sidecar (separates provenance from files
that travel alone, failing US1).

**Consequence**: a pipeline used directly as a library (not through a job) writes files without
provenance. Documented; the viewer labels those "version not recorded".

## R2. Embedding per format

| Format | Files | Where the record goes | Why safe |
|---|---|---|---|
| COCO JSON | person_tracking, scene_detection, face_detections, vlm_annotation, openface3_analysis | top-level `"provenance"` key | `pycocotools.COCO` reads only `images`/`annotations`/`categories`; extra top-level keys are ignored (checked against the installed version in tests) |
| Other JSON | openface3_detailed, person_tracks | top-level `"provenance"` key | our own formats; viewer detection keys on other fields |
| WebVTT | speech_recognition | a `NOTE videoannotator-provenance` block right after the `WEBVTT` header, holding the record as one line of JSON | WebVTT spec: NOTE blocks are comments; the viewer's parser already skips them. `-->` can't appear in a NOTE, so it is escaped as `-->` inside the JSON (still valid JSON) |
| RTTM | speaker_diarization | companion `<file>.provenance.json` | RTTM has no comment syntax every loader skips (pyannote's loader splits every line on whitespace) |

**Decision**: one top-level key name, `provenance`, everywhere JSON is used, so readers have one
rule. Companion files use the suffix `.provenance.json` appended to the full file name.

## R3. Weight revisions per model

| Pipeline | Model(s) | Revision recorded |
|---|---|---|
| person_tracking | YOLO weights (`.pt` in the models dir) | sha256 of the file loaded |
| face_openface3_embedding | RetinaFace, STAR landmarks, MTL backbone | sha256 of each weights file |
| face_analysis | DeepFace attribute models (+ detector backend) | sha256 of each weights file in DeepFace's home |
| scene_detection | open_clip model + pretrained tag | tag + sha256 of the cached checkpoint |
| speech_recognition | openai-whisper `<size>.pt` | sha256 of the checkpoint (whisper itself verifies it against the hash in its download URL) |
| speaker_diarization | pyannote pipeline from Hugging Face | Hub commit hash of the cached snapshot (the snapshot folder name), plus sub-model repos |
| vlm_annotation | Ollama model | model digest and quantisation reported by the Ollama server, server URL, prompt sha256 |

**Decision**: each pipeline implements `provenance_models() -> list[ModelRef]`; a shared helper
hashes weight files with a per-process cache keyed by (path, size, mtime), so a 100-MB file is
hashed once per server process, not per job. When a revision can't be determined, the record says
`"revision": null` with `"revision_note"` giving the reason (spec edge case).

## R4. Job record storage

**Decision**: add a nullable JSON `provenance` column to the `pipeline_results` table via the
existing additive-migration pattern (`_ensure_batch_columns` style `ALTER TABLE ... ADD COLUMN`);
`PipelineResult` dataclass gains `provenance: dict | None`; the file backend serialises it with the
rest. API `PipelineResultResponse` gains `provenance`.

**Rationale**: additive, nullable: old databases and old jobs keep working (Principle V).

## R5. Determinism and settings

- Determinism fields come from `apply_torch_settings()`'s return value (2026-10-04), plus the job's
  `deterministic` flag.
- Effective settings = the pipeline instance's `config` after construction (defaults merged with
  the job's config). Keys matching `token|secret|password|api_key|apikey|auth` (case-insensitive,
  at any depth) are replaced by `"<redacted>"`.

## R6. Times and input identity

- `created_at`: UTC ISO-8601 with `+00:00`, taken when the file is stamped.
- COCO `info.date_created` (hard-coded `2025-01-01T00:00:00Z` in
  `exporters/native_formats.py`) becomes the real UTC time; `info.year` the real year.
- Input video: name and sha256, computed once per job and cached on the job for its pipelines.

## R7. Baseline and fixtures

- `tests/integration/test_output_baseline.py` compares exactly for scene/person tracks/RTTM/VTT.
  Provenance, `info.date_created` and `info.year` differ per run, so the test strips them (JSON),
  drops the provenance NOTE block (VTT) and ignores companion files. Annotation values must still
  match (FR-006).
- Viewer contract fixtures are regenerated with provenance; the old fixtures without provenance
  are kept under `tests/fixtures/viewer_contract/legacy/` so the contract test covers both (FR-014).

## R8. Viewer attribution

- Each parser returns the file's provenance (JSON key, VTT NOTE, or matched companion) or a partial
  record (COCO `info.version` only) or none.
- `StandardAnnotationData` gains `provenance: Partial<Record<TrackKey, ProvenanceInfo>>`.
- `UnifiedControls` shows under each overlay's label: "<pipeline> · VideoAnnotator <version>" or
  "version not recorded"; an info button opens the full record, values verbatim (Principle VI).
- ELAN tracks: "Ground truth · <file name>".
