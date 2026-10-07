# Implementation Plan: Prompt Library and Prompt Workbench

**Branch**: `1.6-dev` (spec `020-prompt-library-workbench`) | **Date**: 2026-10-04 | **Spec**: [spec.md](spec.md)

## Summary

Every VLM prompt that runs (a job, a preview from the wizard, the workbench or the CLI) is
recorded once, keyed by the SHA-256 of its exact text (the same hash spec 017's provenance
records), with each use (model, job, kind, user, time). A Prompts page searches, names, stars,
tags, hides, diffs and reuses them. A Workbench page runs prompts × models × moments of a
server-side video side by side, keeps earlier rounds, shows the frames the model saw, and sends
a winner to a job or preset. CLI: `videoannotator prompts list|show|diff`, `videoannotator vlm
preview`.

## Technical Context

**Language/Version**: Python 3.12/3.13; TypeScript/React (Bun)
**Primary Dependencies**: existing (FastAPI, SQLAlchemy users-db layer, ollama client)
**Storage**: two new tables in the users/datasets database (`prompts`, `prompt_uses`), created by
`create_all` (new tables need no migration)
**Testing**: pytest (service, API, CLI, job hook), Vitest (diff, pages), real Ollama end to end
(`gemma4:e4b`, `qwen3.5:9b` on this machine)
**Constraints**: recording a prompt never fails a job or preview; Local-First notice when the
model server isn't local

## Design decisions

- **Recording points.** `batch/job_execution.py` after a pipeline whose provenance has a `vlm`
  block (the record has the prompt hash; the text comes from the pipeline's settings);
  `POST /api/v1/vlm/preview` after a successful call. Both best-effort (logged on failure).
- **Identity.** `sha256(text.encode("utf-8"))`, exact text, no normalisation (spec edge case).
- **Frames used.** The preview response gains `frames: [{frame_number, timestamp_sec, jpeg_base64}]`
  (downscaled to 320 px wide) so the workbench shows what the model saw without a video stream.
  Additive; the wizard panel ignores it.
- **Workbench videos.** Videos of past jobs (their stored paths, which the server can read) and,
  for admins on the same machine, server folders via the existing ingest browse.
- **Deletion.** `DELETE /prompts/{sha}` refused (409) once any job used it; hiding is the
  alternative.
- **Diff.** Word-level LCS in the viewer (`lib/wordDiff.ts`); whitespace-only differences named
  as such.

## Constitution Check

I Local-First: pass (model server is the configured one; the workbench names it and warns when it
isn't on this machine). II: not touched. III Provenance: strengthens it (prompt history keyed by
the provenance hash). IV/V: additive tables and response field. VI: N/A. No violations.

## Project Structure

```
src/videoannotator/database/models.py      # Prompt, PromptUse
src/videoannotator/prompt_library.py       # record_use, search, get, update, delete
src/videoannotator/api/v1/prompts.py       # /api/v1/prompts
src/videoannotator/api/v1/vlm.py           # record previews; frames in response
src/videoannotator/batch/job_execution.py  # record job prompts
src/videoannotator/cli.py                  # prompts list|show|diff, vlm preview
viewer/src/pages/Prompts.tsx, pages/Workbench.tsx, lib/wordDiff.ts, types/prompts.ts
```
