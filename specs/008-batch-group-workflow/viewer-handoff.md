# Handoff to video-annotation-viewer: Batch/Group Progress, Controls & Review Navigation

**From**: VideoAnnotator core, `specs/008-batch-group-workflow/` (v1.5.0 branch)
**Purpose**: Written to be pasted as the input to that repo's own `/speckit-specify`. Describes UI
behavior, not implementation — component structure and state management are that repo's own call.

## Why this exists

Submitting several videos with one config today (`NewJob.tsx`'s multi-file picker + one shared
config) results in N fully independent `POST /api/v1/jobs` calls with no shared identity anywhere.
`Jobs.tsx` shows N unrelated rows; progress is a crude 4-value stepped bar per row
(`JobDetail.tsx:59-69`, `pending→0, cancelling→25, running→50, completed→100`); the wizard's "time
estimate" (`NewJob.tsx:1009`, `~{Math.ceil(selectedFiles.length * 7)} minutes`) is a static per-file
guess unrelated to actual video length or pipeline mix; cancel and retry are single-job-only
(`useJobCancellation.ts`, `useJobDeletion.ts`); and reviewing results means opening one job at a
time from the Jobs list with no way to step to "the next video in this batch."

## What the backend now provides

Full contract in [`spec.md`](spec.md)'s API section. Summary:

- `POST /api/v1/jobs` now accepts an optional `batch_id`, `batch_name` and `dataset_id` form
  field. Generate one UUID client-side per wizard submission and send it with every per-video call
  in that submission — this is the entire mechanism; there's no separate "create a batch" call.
  Send the same `batch_name` on every call too, so the batch has something human to be called.
- `GET /api/v1/batches` → `{ batches: [...], total, page, per_page }`, newest submission first.
  **This is new since this handoff was first written, and it changes the recommended design** — see
  "Things worth knowing" below. The viewer no longer has to remember client-side which jobs it
  submitted together.
- `GET /api/v1/batches/{batch_id}` → `{ batch_id, batch_name, dataset_id, created_at, total,
  by_status: {...}, completion_percentage, estimated_seconds_remaining }` — a real, computed-on-read
  aggregate. `estimated_seconds_remaining` is `null` until the batch has at least one completed job
  (don't render a time estimate before then).
- `GET /api/v1/jobs?batch_id=...` → the member jobs of one batch, composable with `status_filter`
  and pagination.
- `POST /api/v1/batches/{batch_id}/retry` → retries every currently-retryable (failed/cancelled) job
  in the batch, reports how many were retried vs. skipped and why.
- `POST /api/v1/batches/{batch_id}/cancel` → cancels every still-cancellable job in the batch,
  reporting jobs already finished in `skipped` rather than failing.
- The existing `/api/v1/events/stream` now emits real `job_status_changed` events (`{ job_id,
  batch_id, status, progress_percentage }`) alongside its existing heartbeat — `src/hooks/useSSE.ts`
  (`useJobSSE`/`useGlobalSSE`) is already fully built in this repo and currently unused anywhere;
  this is what finally gives it something real to consume.

### Things worth knowing before designing the UI

- **Polling still works and remains correct** — SSE is additive. It's fine to ship the group-progress
  UI on polling first (mirroring `useExtrasInstall.ts`'s proven pattern: `useQueries` over N tracked
  items, `localStorage` persistence so state survives a reload) and adopt SSE as an optimization
  pass, rather than blocking on wiring `useSSE.ts` in before shipping anything.
- **Batch membership is now server-queryable** — this supersedes the original advice here to track
  submitted job ids client-side. `GET /batches` enumerates batches and `GET /jobs?batch_id=...`
  returns one batch's member jobs, so batch identity survives a reload, a different browser, and a
  different machine. Client-side tracking of submitted ids is no longer necessary and shouldn't be
  the source of truth. `GET /batches/{id}` still returns the aggregate only, not the member list —
  use the filtered jobs endpoint for per-video detail.
- **Cancel-all is one call now** — the original advice here was to loop the single-job cancel hook
  over the batch's job ids. `POST /batches/{id}/cancel` does it server-side and reports per-job skip
  reasons, matching batch-retry's shape.
- **`estimated_seconds_remaining` can be `null`.** Don't render "0:00 remaining" or similar before the
  batch has at least one completed job — show an honest "estimating..." or nothing at all instead.
- **Retry-in-a-batch reuses spec 006's single-job retry semantics per job** — no new file upload,
  same config. If you've already built single-job retry UI against spec 006, the batch version is
  the same underlying capability at group scale, not a new interaction pattern.

## What the viewer needs to build

1. **A group progress card replacing N independent rows** in the Jobs list when jobs share a batch
   identifier — overall completion percentage, time estimate (once available), and per-video
   mini-status, collapsible to the individual rows if wanted.
2. **Real in-process messaging** — surface which pipeline/stage is currently running for a job
   (`progress_percentage` combined with `selected_pipelines` ordering) instead of the current
   4-value stepped bar.
3. **Bulk actions on a batch**: cancel-all and retry-all, each a single call to the corresponding
   batch endpoint.
4. **Batch review navigation**: when opening a job's results from within a batch context, carry the
   batch's ordered job-id list and current index into the results viewer
   (`VideoAnnotationViewer.tsx`/`JobResultsViewer.tsx`, currently strictly one-job-at-a-time with no
   sibling awareness at all) so prev/next controls can step between videos in that batch without
   navigating back out to the Jobs list each time.
5. **Tag every wizard submission with a batch id**, generated once per submission pass, sent on every
   per-video `submitJob` call already being made in the existing loop
   (`NewJob.tsx:243-247`) — no change to the upload mechanism itself, just one new field per call.

## Explicit non-goals for this piece of work

- **A true multi-file-upload-in-one-request endpoint.** Uploads stay one request per video; only the
  shared identifier is new (see spec's Assumptions).
- **Making SSE a requirement.** Everything must work correctly on polling alone; SSE is an
  optimization, not a dependency.
- **Corpus-wide/dataset-wide analysis views.** That's `specs/010-corpus-analysis-foundations/` — a
  separate handoff. This one is scoped to "jobs submitted together," not "everything in a dataset
  ever."

## Suggested next step

Paste "What the viewer needs to build" (plus the endpoint summary) into that repo's own
`/speckit-specify`.
