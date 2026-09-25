# Handoff to video-annotation-viewer: fixes from the spec 011 end-to-end run

**From**: VideoAnnotator, v1.5.0 branch, 2026-09-25
**Context**: the SC-001 manual run ([tests/manual/pipeline_readiness_e2e.md](../../tests/manual/pipeline_readiness_e2e.md))
against the bundled viewer (v0.7.0): a 5-video batch with `face_analysis` + `person_tracking`
installed live from the viewer, while Ollama was down.
**Scope**: four viewer fixes. None needs a new spec; each builds on the
[011 handoff](viewer-handoff.md) and its [readiness contract](contracts/readiness-contract.md).
After v1.5.0 is tagged the viewer moves into this repo (`viewer/`, see
[roadmap v1.6.0](../../docs/development/roadmap_v1.6.0.md)), so this should be the last
cross-repo handoff.

## What already changed on the backend (no viewer work needed)

- **`person_tracking` silently didn't run after a live install.** A health poll's GPU probe imported
  torch while the install was still writing it, leaving a broken module cached in the server. Fixed:
  health checks no longer import torch during an install, and live activation clears any such
  half-installed module. If an import does still fail, the pipeline now shows up as `needs_setup` with
  an `import_error` blocker instead of disappearing.
- **Jobs now go to `running` as soon as a worker picks them up.** Before, they stayed `pending`
  through the first job's cold start, which imports the pipeline libraries and took ~5 minutes in the
  run. Only one job does that import now; the others wait for it rather than repeating it.
- **DeepFace weights (~1.1 GB) download once, not once per concurrent job**, and `face_analysis` now
  declares them, so its readiness carries a `weights_not_cached` note until they're on disk.
- **A selected pipeline the server can't run is now reported, not skipped silently.** The job ends
  `completed` with `error_message: "Completed with errors. Failed pipelines: <name>"`, and
  `GET /api/v1/jobs/{id}/results` has `pipeline_results[<name>]` with `status: "failed"` and an
  `error_message` giving the reason.

## 1. A pipeline that isn't ready can't be removed from the selection (bug)

**Seen**: with Ollama down, `vlm_annotation` read as not ready (`needs_setup`, Ollama blocker) but was
already in the selection, with no checkbox to take it out. All 5 jobs were submitted with it.

**Expected**: the 011 rule "not ready means not selectable" also covers pipelines that are already
selected, whether they came from a preset, the last-used selection, a dataset, or anywhere else.

- A selected pipeline whose `readiness.state` isn't `ready` can always be deselected, even though it
  can't be selected again.
- When a selection is built from a preset or previous choices, leave out pipelines that aren't ready
  and say so: "VLM Annotation left out: can't reach the Ollama server."
- If a pipeline stops being ready after it was selected (readiness is re-fetched), show that on its
  card and block submission with a message saying which pipeline is the problem.

**Done when**: with Ollama stopped, you can't submit a job that includes `vlm_annotation` from any
starting point in the wizard, and an existing selection containing it can always be cleared.

## 2. Say when the first run is slow because of setup, not because it's stuck

**Seen**: all 5 videos read `PENDING 0%` with "Time remaining: Estimating…" for about 10 minutes.
Most of that was the one-off cold start, followed by DeepFace downloading ~1.1 GB of weights. Nothing
in the UI said so.

**Expected**, using what the API already returns:

- **Before submitting** (Review step): for each selected pipeline, show its `weights_not_cached`
  readiness notes, plus one summary line, e.g. "First run downloads about 1.1 GB of model weights.
  The first video will take several minutes longer."
- **On the run page**, while a job is `running` at 0% and none of its pipelines have finished:
  "Preparing (first run loads models and may download weights)" instead of a bare 0%. Show it
  specifically when the run was submitted with `weights_not_cached` notes outstanding, and keep that
  list client-side from the submit step.
- Don't base "Time remaining" on the first video when it included setup. That finish time overstates
  every later video (the ETA already waits for the first video, which is right; just don't let that
  outlier dominate).

A true per-job stage field (`stage: "downloading_weights"` with bytes done) is a future backend spec.
Don't wait for it; the notes above cover the confusing part.

## 3. Show "completed with errors" and the reason

**Seen**: the run finished "ok" while `person_tracking` had produced nothing. The backend now reports
this (see above), but the run page only counts ok/failed.

**Expected**:

- Outcome card: count jobs that are `completed` with a non-empty `error_message` separately, e.g.
  "3 ok · 2 with errors · 0 failed".
- Job row and job page: show `error_message`. On the job page, list each failed pipeline with its
  `pipeline_results[name].error_message` from `GET /api/v1/jobs/{id}/results`.
- In the annotation viewer, a component whose pipeline failed shows "Failed: <reason>" rather than
  "(No data)".

## 4. H.265/HEVC videos play as a black frame with no explanation

**Seen**: `2UWdXP.joke1.rep2.take1.Peekaboo_h265.mp4` opened in the viewer with annotations loaded
(face/emotion tracks present), but the video area stayed black. Most browsers on Linux, and some
elsewhere, can't decode HEVC.

**Expected**:

- Detect it: a `<video>` `error` with `MEDIA_ERR_SRC_NOT_SUPPORTED`, or `loadedmetadata` with
  `videoWidth === 0`. Where a codec hint helps, `canPlayType('video/mp4; codecs="hvc1"')` returning
  `""` is a good early warning.
- Show it over the player: "This browser can't play this video's codec (H.265/HEVC). Annotations are
  still available below. To see the video, re-encode to H.264, e.g.
  `ffmpeg -i in.mp4 -c:v libx264 -crf 18 -c:a copy out.mp4`, or open it in a browser with HEVC
  support."
- Keep the timeline and annotation panels working, as they do now.

Converting HEVC on upload is a possible future backend feature; it isn't planned yet.

## Suggested order

1 (it lets through jobs that can't succeed), then 3, 2 and 4. Rebuild and hand the bundle back for
`viewer_static/`, then re-run SC-001 from a clean `~/.deepface` and a core-only install.
