# Implementation Plan: Compare Two VLM Jobs

**Branch**: `1.6-dev` (spec `021-compare-vlm-jobs`) | **Date**: 2026-10-04 | **Spec**: [spec.md](spec.md)

## Summary

A viewer page, `/compare?a=<job>&b=<job>`, loads both jobs' `vlm_annotation` output (the per-file
download route fixed in A2), checks they are the same video (spec 017's input sha256; older jobs:
file name and size), pairs moments by time, and shows two timeline rows (plus ELAN ground truth
when an `.eaf` is loaded), the disagreements, a summary with a label-pair table, and CSV export.
Selecting a moment seeks the video and shows both runs' label, reasoning and raw response. Job
pages offer "Compare with…" and, for reruns, "Compare with original".

## Technical Context

**Language/Version**: TypeScript/React (viewer); Python for one small endpoint
**Dependencies**: existing only
**Storage**: none (a comparison is its link)
**Testing**: Vitest (pairing, summary, CSV, page), pytest (video endpoint), real check with two
Ollama VLM jobs (`gemma4:e4b` vs `qwen3.5:9b`) in Chromium

## Design decisions

- **Video for seeking**: `GET /api/v1/jobs/{id}/video` serves a job's stored video (Starlette's
  `FileResponse`, which answers range requests, so the browser can seek).
- **Pairing**: exact timestamps pair directly; otherwise nearest within half the larger sampling
  interval (from provenance `settings.frame_interval_sec`, else the median gap between samples).
  Each sample is used once; unpaired and error moments are counted, never dropped (SC-004).
- **Labels compared exactly**; an explicit "ignore case" toggle normalises for the figures only.
- **Ground truth** reuses the VLM-vs-ELAN panel's rule: ELAN four-way category vs the label's
  touch/no-touch reading (`isPositiveLabel`, moved to `lib/vlmLabels.ts` and shared).
- **Same video**: both provenance records have `input.sha256` → must match; otherwise both jobs'
  file name and size must match; otherwise the page refuses with the reason.

## Constitution Check

I–V: pass (viewer-only plus a read-only endpoint behind the same auth). VI Faithful Display: labels
shown and compared as recorded; normalisation is a visible option; row labels carry spec 017
attribution. No violations.
