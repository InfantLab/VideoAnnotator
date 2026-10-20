# Viewer contract: Videos and Projects

**Spec**: [../spec.md](../spec.md) | **API**: [api.md](api.md)

What the viewer's tests check. Words in **bold** are the labels on screen.

## Navigation (FR-018, SC-004)

**Home · Videos · Ask AI · Pipelines**, then the activity indicator, **Help**, **Settings**.
No other top-level entries. New run has no entry: it opens from **Run…** on Videos, a project,
a selection, a video, or Home.

## Routes

| Route | Page | Notes |
|---|---|---|
| `/` | Videos when the server has a video with a finished run, else Home (R18) | |
| `/home` | Home | Start points and setup checklist (US7) |
| `/videos` | Videos | `?project=`, `?folder=` (+`recursive=1`), `?not_in_project=1`, `?run=`, `?q=` |
| `/video` | Video page (028) | `from=` gains `all` and `unassigned` |
| `/run` | New run (R9) | Router state: `RunTarget` |
| `/ask`, `/pipelines`, `/settings`, `/compare`, `/viewer`, `/view/:jobId` | as now | `/viewer?demo=<key>` opens a demo (R14); Pipelines gains **Workflows** |
| `/help` | Help (FR-027) | Replaces Getting started |

Old addresses (R13, SC-006), each a redirect that keeps working:

| Old | Goes to |
|---|---|
| `/jobs` | `/videos` |
| `/jobs/new` | `/run` (router state passed on; old `WizardStart` shapes read) |
| `/jobs/:id` | `/video?path=<its video>&run=:id` (job not found: `/videos` with a toast) |
| `/batches` | `/videos` |
| `/batches/:id` | `/videos?run=:id` |
| `/datasets` | `/videos` |
| `/results`, `/library` | `/videos` |
| `/getting-started` | `/help` |
| `/prompts`, `/workbench` | `/ask` (as now) |

## Videos page

- **Left**: **All videos (n)**, **Not in a project (n)**, **Projects** (each with its count;
  owner in brackets when two share a name; **+ New project**), **Folders** (each with its count;
  uploads as **Uploaded videos**).
- **Header** for the scope: its name and count; **Run…** and **Export ▾** (**Results table**,
  **ELAN**, **Methods**) for what's shown. A project's header adds its **Settings** line
  (pipelines, and Ask AI's question if any) with **Edit**, its runs (`● running 31/48 · ✓ 12 Oct`),
  **What changed?**, **Save as workflow**, **Apply workflow**, **Rename**, **Remove project**;
  and, when the folder has new videos, "**3 new videos in /data/pk** · **Add them**".
- **Run filter** (`?run=`): the run's name, progress, project; **Cancel**, **Retry failed
  videos**, **Run again…**, **Delete run**, **What changed?**; when finished, the headline
  "**48 videos → results table** [Download]" with **ELAN** and **Methods** beside it; VLM runs
  add **Compare answers**.
- **Table**: a checkbox, **Video**, **Folder**, **Projects** (chips), then one column per
  pipeline present in the scope, showing ✓ done, ● running, ◌ waiting, ✕ failed (hover: the
  reason), – none. Rows of videos whose file is gone are marked **not on the server** and keep
  their results. Clicking a row opens the Video page with `from=` this scope.
- **Selection bar** (when rows are selected): "**n selected**", **Run…**, **Export ▾**, **Add to
  project ▾** (each project, **New project…**), and in a project **Remove from project**.
- Windowed past 200 rows; filter by name.

## New run (R9)

- Opened with a target: two steps, **What to find** (pipelines with their guide, settings, a
  **Workflow** picker, "Settings from: <source>") and **Start** (videos, pipelines, estimate,
  run name, project). **Start run on 48 videos** is the only confirmation when nothing was
  changed (FR-017).
- Without a target: a first step to choose videos (**A shared folder**, **This computer**,
  **A project**, **A video you ran before**).
- **Save as workflow** on the settings step.

## Video page (028) additions

- **Runs** tab: per run, status, progress (and queue position while waiting), each pipeline's
  state and error, its settings, results folder, **Download all files**, **Cancel**, **Delete**,
  **Run again…** (R12).
- The header's back link goes to the list it came from (`/videos?…`).

## Pipelines page

**Pipelines** (as now) and **Workflows**: each workflow's name, pipelines, where it came from,
**Apply to project…**, **Use in a new run**, **Export**, **Rename**, **Delete**; **Import
workflow**.

## Words (FR-019, SC-002)

No rendered text, `title`, `aria-label` or placeholder on any route contains `job`, `jobs`,
`batch`, `batches`, `dataset`, `datasets`, `preset` or `presets` (whole words, any case), in the
loaded, empty and error states. Checked by `src/test/vocabulary.test.tsx`. Server messages are
shown through `userMessage(error)` (R15).

## Help (FR-023–FR-027)

- Every routed page renders `PageHelp`: one sentence and **More in the docs**, linking to
  `https://infantlab.github.io/VideoAnnotator/<docs path>/#<anchor>` from `lib/docsLinks.ts`.
  `src/test/docsLinks.test.ts` checks every linked file exists under `docs/` and the anchor is a
  heading's MkDocs slug.
- `Term` tooltips for **video**, **folder**, **run**, **project**, **workflow**, **track** from
  `lib/glossary.ts`; every setting's label has a tooltip with its description and default.
- Toasts (`notify()`): run started ("Run started on 48 videos. You can close this tab; results
  will be here." + **Show run**), finished (+ **Results table**), failed (+ **Show run**),
  stopped; project created, videos added or removed, project removed; workflow saved, applied,
  imported.
- Without a server, Videos, New run and Pipelines' install actions show `NeedsServer`: what needs
  a server, **Connect** (Settings), and what works without one (**Open files from this
  computer**, **Demos**).
- No link to DeepWiki or any other wiki (already true; the test keeps it so).

## Keys

Unchanged from 028 on the Video page. On Videos: `/` focuses the filter; `Enter` opens the
focused row; `Space` selects it.
