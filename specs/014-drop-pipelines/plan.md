# Implementation Plan: Drop Obsolete Pipelines

**Branch**: `1.6-dev` | **Date**: 2026-10-01 | **Spec**: [spec.md](spec.md)

## Summary

Remove `laion_voice` and `face_laion_clip` (code, metadata, extras, tests, docs); requests for them
get a removal message at every entry point. Deprecate `audio_processing`: hidden from listings,
still runs, warns, removed in v1.7.0. Make family short names resolve to a declared default.

## Design

- **Registry** (`registry/pipeline_registry.py`): `PipelineMetadata` gains `family_default: bool`
  and `deprecated: DeprecationInfo | None` (`removal_version`, `replacement`), parsed from YAML.
  `list(include_deprecated=False)`: every listing hides deprecated pipelines; the loader asks for
  them.
- **Loader** (`registry/pipeline_loader.py`): `REMOVED_PIPELINES` (name → version, reason) with
  `removed_pipeline_message()`; `deprecation_message()`; a declared family default outranks the
  stability ranking.
- **Entry points**: `validate_pipeline_selection` (job submission and folder ingest) raises
  `PipelineRemovedException` (422, `PIPELINE_REMOVED`); the batch path's `_unavailable_reason`
  returns the removal message; `JobResponse.warnings` carries deprecation messages;
  `job_execution` logs them.
- **Metadata**: `audio_processing.yaml` gets `deprecated` + `family_default`; `face_analysis.yaml`
  gets `family_default`; the two LAION YAMLs are deleted.
- **Configs**: the `audio_processing:` sections were dead (the pipeline reads
  `config["pipelines"][...]`; only `sample_rate` was read, and every config set it to the
  default), so they're replaced by a note; `configs/laion_pipelines.yaml` deleted.

## Constitution Check

| Principle | Assessment |
|---|---|
| II. Stable pipeline contract | Remaining pipelines' outputs unchanged (comparison). **Pass** |
| V. Backward compatibility | `audio_processing` deprecated, not removed; removed pipelines fail with an explanation, not a crash. Removal of two pipelines within v1.x is the review's explicit decision, recorded in the CHANGELOG. **Pass with justification** |
| IV. Modular | Two extras gone; `transformers` leaves the dependency tree. **Pass** |
