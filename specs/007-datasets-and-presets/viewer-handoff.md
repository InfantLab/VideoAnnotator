# Handoff to video-annotation-viewer: Datasets & Presets UI

**From**: VideoAnnotator core, `specs/007-datasets-and-presets/` (v1.5.0 branch)
**Purpose**: Written to be pasted as the input to that repo's own `/speckit-specify`. Describes a UI
feature, not an implementation — component structure, state management, and styling are that repo's
own call.

## Why this exists

Every job submission today starts from zero: re-pick every video file from a browser dialog
(`src/pages/NewJob.tsx`'s file input has no memory of a previous selection), re-select pipelines,
re-type or re-paste configuration (including, for `vlm_annotation`, a long prompt that ideally
should be reproduced verbatim run to run for research validity). `src/pages/Datasets.tsx` already
exists as a page titled "Dataset Management" — it is a fully disabled "Coming Soon" mockup today.
This handoff is what makes it real, plus adds preset save/load to the job wizard.

## What the backend now provides

Full contract in [`spec.md`](spec.md)'s "API Contract for Downstream Consumers" section. Summary:

- `GET/POST /api/v1/datasets`, `GET/PUT/DELETE /api/v1/datasets/{id}` — each dataset record has
  `name`, `description`, a `video_manifest` (list of `{filename, size_bytes, last_seen_at}`),
  `created_at`, `updated_at`, `last_used_at`.
- `GET/POST /api/v1/presets`, `GET/PUT/DELETE /api/v1/presets/{id}` — each preset has `name`,
  `description`, `selected_pipelines: string[]`, `config: object` (identical shape to what a job
  submission's own `config` field already looks like — no translation needed to apply one), `tags`,
  usage timestamps, and an `unavailable_pipelines: string[]` field flagging any referenced pipeline
  currently unavailable on this server.
- Export is just `GET` on either resource (already a self-contained definition); import is `POST`
  with a previously-exported body. No separate endpoints.
- **Visibility is shared**: any authenticated user on this server can see and use any saved
  dataset/preset (not just their own) — only editing/deleting is owner-restricted. Design the UI
  accordingly (e.g. show who saved it, since it isn't necessarily "yours").

### Things worth knowing before designing the UI

- **A dataset is a manifest, not the files.** The server never receives or stores the actual video
  bytes when a dataset is saved — only filenames/sizes. Re-selecting a saved dataset for a new job
  submission still requires the browser to have access to the actual files again. The File System
  Access API's `showDirectoryPicker()` (already used in this repo for *output* storage —
  `getRootDirHandle`/`setRootDirHandle` in `src/lib/localLibrary/libraryStore.ts`) is the natural
  mechanism: remember a directory handle alongside the saved dataset id, and re-request it (browsers
  that support the API will re-grant permission for a previously-used handle without a fresh picker
  dialog in many cases). Where the API isn't available/permission has lapsed, fall back to asking the
  user to re-pick the folder and match files against the manifest by filename+size, surfacing any
  drift (missing/added/renamed files) rather than silently proceeding.
- **Presets are a direct drop-in for the wizard's existing config state.** A preset's
  `selected_pipelines`/`config` are the exact same shape `NewJob.tsx` already manages internally
  (see how `RetryJobState` already pre-fills `config`/`selectedPipelines` from a past job — this is
  the same pattern, just sourced from a saved preset instead of a past job).
- **`unavailable_pipelines` needs a visible-but-not-blocking treatment.** Applying a preset that
  references a currently-locked pipeline (see `specs/002-pipeline-extras-install/` — the
  locked-pipeline-card UI that's already shipped) shouldn't fail; it should apply what it can and
  flag the unavailable pipeline(s), ideally reusing that existing locked-pipeline presentation rather
  than inventing a new one.

## What the viewer needs to build

1. **A real Datasets page**, replacing `src/pages/Datasets.tsx`'s current disabled mockup: list
   saved datasets (name, video count, created/last-used), create one from a locally-picked folder,
   view/edit its manifest, delete it.
2. **"Load from saved dataset" in the job wizard's Upload step**, alongside the existing multi-file
   picker — selecting a dataset re-requests its folder (or prompts re-selection) and populates the
   file list from it, same as picking files manually would.
3. **"Save as dataset" from a manually-picked file set**, so a dataset doesn't require pre-existing
   Datasets-page setup — a researcher who just picked 40 files in the wizard should be able to save
   that exact set as a named dataset in the same flow, not have to start over from the Datasets page.
4. **"Save as preset" / "Load preset" in the job wizard's Configure step**, applying a preset's
   `selected_pipelines`/`config` directly into the wizard's existing state.
5. **Export/import affordances** for both datasets and presets (download the definition as JSON /
   upload one to import) — this is the "share a configuration with a colleague" path; no new backend
   work needed beyond what's already in the contract.
6. **Attribution in shared listings**: since any user can see any saved dataset/preset, show who
   saved it (or at least don't imply exclusive ownership) in the list/detail views.

## Explicit non-goals for this piece of work

- **Private/per-team visibility scoping.** Everything is shared-read by design (see spec's
  Assumptions) — don't build a privacy toggle; it isn't backed by anything server-side yet.
- **Validating a preset's saved config against a pipeline's current schema field-by-field.** The
  backend only flags whole-pipeline unavailability (`unavailable_pipelines`), not per-field drift —
  don't build UI implying deeper validation than that.
- **Batch/group submission tagging, group progress, or corpus-wide analysis.** Those are
  `specs/008-batch-group-workflow/` and `specs/010-corpus-analysis-foundations/` — separate handoffs,
  separate specs, even though a saved dataset is what makes them possible.

## Suggested next step

Paste "What the viewer needs to build" (plus the endpoint summary) into that repo's own
`/speckit-specify` to produce a spec scoped to this codebase's actual components and conventions.
