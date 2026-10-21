# Videos and projects

The viewer is organised around your videos. **Videos** lists every video the server knows, with
the results each one has; you pick some (a folder, a project, a few rows) and run pipelines on
them, export their results, or put them in a project. This page explains how.

The words, once:

- A **video** is one recording. If the server keeps two copies of the same file (an upload and
  the copy a later run made of it), it is still one video.
- A **folder** is where videos are on the server: a folder shared with VideoAnnotator, or
  **Uploaded videos** for videos sent from a browser.
- A **run** is one go of pipelines over some videos, started together.
- A **project** is a set of videos you group for a study. A video can be in several projects,
  and a project can hold videos from several folders.
- A **workflow** is a saved set of pipelines and settings you can reuse and share.
- A **track** is one row of results on the Video page (speech, faces, a question's answers, …).

## The Videos page

On the left: **All videos**, **Not in a project**, your **Projects** and your **Folders**, each
with how many videos it holds. Pick one and the table shows its videos.

The table has a row per video and a column per pipeline that ever ran on them:

| Mark | Means |
|---|---|
| ✓ | done: open the video to see the results |
| ● | running |
| ◌ | waiting its turn |
| ✕ | failed: hover for why |
| – | not run |

A video whose file has gone from the server stays in the list, marked **not on the server**, with
the results it had. Click a row to open its Video page; ‹ › on that page move through the same
list. Type in the filter box (or press `/`) to find videos by name; `Enter` opens the row you're
on and `Space` selects it.

Past a few hundred videos the table only draws the rows on screen, so it stays quick.

## Folders

**Folders** lists the folders shared with VideoAnnotator that have videos, and **Uploaded videos**.
A folder shows every video in it, run or not, so it is where you start with new recordings: pick
the folder, **Run…**, and its rows fill in as the videos finish.

A folder that holds other folders lists them under its name (**Folders inside**): open one to see
just its videos.

You can only see folders the server has been given (see [Installation](../installation/INSTALLATION.md)
for sharing folders with a Docker or remote server). Videos from another computer are uploaded in
New run: choose **This computer**.

## Projects

A project is a set of videos you name, made whenever you like: before running anything, or long
after.

- **Make one**: select rows, **Add to project ▾ → New project…**, and name it. Or **+ New
  project** on the left for an empty one.
- **Add videos**: select rows anywhere (another folder, uploaded videos) and **Add to project ▾**.
  A project can hold videos from several folders.
- **Take videos out**: inside the project, select rows and **Remove from project**. The videos and
  their results stay where they are.
- **Remove a project**: **Remove project** in its header. Only the project goes; every video and
  every result stays. Only the person who made it, or an administrator, can remove it; anyone
  signed in can add to it, take videos out, rename it and change its settings.

A project made from a whole folder **follows the folder**: videos copied into the folder later are
in it too, and its header says so. A project made from some of a folder's videos holds just those;
when the folder gains new videos, its header offers them: **3 new videos in /data/pk · Add them**.
Adding videos from another folder to a project that follows its folder makes it a selection of the
folder's current videos plus the new ones.

A video's projects show as chips on its row and on its Video page. Two projects with the same name
(made by different people) show their owner's name.

Datasets saved in earlier versions are projects: they're listed with their name, videos and runs.

## Settings a project remembers

A project remembers the settings its next run starts from. They are set whenever a run starts on
the project, and you can change them without running: **Edit** beside **Settings** in the project's
header. A project from an earlier version starts from the settings of its newest run.

**New run** says where its settings came from, at the top:

1. the **workflow** you chose, if you chose one;
2. **this project's settings**, when you're running a project, with when they last changed;
3. **your last run**, otherwise;
4. **the defaults**, the first time.

**What changed?** in a project's or run's header compares two runs and lists the settings that
differ, such as "Faces · smallest face: 30 → 40".

## Workflows

A workflow is a set of pipelines and settings with a name, used only when you ask for it.

- **Save as workflow** in a project's or run's header saves its settings. You can also choose how
  its results come out: the table as CSV or Parquet, and whether ELAN files and a methods
  paragraph are offered.
- **Apply workflow** in a project's header copies a workflow's settings into the project. Changing
  the workflow later doesn't change the project.
- In New run, choose a workflow at the top of **What to find**.
- **Pipelines → Workflows** lists them all, with where each came from: **Apply to project…**,
  **Use in a new run**, **Export** (a file to share), **Rename**, **Delete**, and **Import
  workflow** for a file someone sent you.

Presets saved in earlier versions are workflows.

## New run

**Run…** on a folder, a project, a selection or a video opens New run with those videos chosen. It
has two steps:

1. **What to find**: the pipelines (with a guide to each) and their settings, starting from the
   settings described above. Hover a setting's name for what it does and its default.
2. **Start**: how many videos, which pipelines, roughly how long it will take, the run's name and
   project. **Start run on N videos** starts it: keeping the settings as they came, that is the
   only thing to confirm.

Opened from Home with nothing chosen, New run first asks for the videos: **A shared folder**,
**This computer** (upload), **A project**, or **A video you ran before**.

Once it starts you can close the tab: the run carries on on the server, and its results will be in
the list when you come back.

## Runs

The indicator beside **Help** in the top bar shows the runs going on (**2 runs · 31/48**); click it
for each run's progress. A message says when a run starts, finishes, fails or is stopped, with a
button to the next step.

A run is shown as a filter of the Videos page (from the indicator, a message, or a project's
runs): its videos, its progress, and **Cancel**, **Retry failed videos**, **Run again…** and
**Delete run**. When it has finished, its header leads with its results:
**48 videos → results table**, with **ELAN** and **Methods** beside it. A run of Ask AI's question
adds **Compare answers**.

Each video's Video page lists every run on it under **Runs**: its status (and place in the queue
while waiting), each pipeline's outcome and error, its settings, where its results are, **Download
all files**, **Cancel**, **Delete** and **Run again…**.

## Exporting results

**Export ▾** in the header exports what's shown (a folder, a project, a run), and in the selection
bar what's selected:

- **Results table**: one table of every pipeline's results (CSV, or Parquet if installed).
- **ELAN**: one `.eaf` per video (a zip for several).
- **Methods**: a paragraph for your paper, with the models, their versions and the settings used.

For a folder, project or selection you choose **Latest results** (for each video and pipeline, its
newest finished result: what the Video page shows) or **Every run** (all of them; the table's
`run` column tells them apart). See [Results out](exporting_results.md) for the table's columns.

## Without a server

Videos, New run and installing pipelines need a VideoAnnotator server. Without one, these pages say
so and how to connect (**Settings**), and you can still **Open files from this computer** (results
you have as files) or open a **Demo**.

## API

The API keeps its own words: a project is a saved *dataset*, a workflow a *preset*, a run a
*batch* (or one *job*). Every signed-in user may use these (`Authorization: Bearer <API key>`):

| Request | What it does |
|---|---|
| `GET /api/v1/videos` | The videos the server knows, for one scope: `project=<dataset id>`, `folder=<path>` (+`recursive=true`; `uploads` for uploaded videos), `not_in_project=true`, `run=<batch or job id>`, or none for all; `q` filters names; `offset`, `limit` (≤ 2,000). Each video has its `path` (newest existing), `paths`, `folder`, `available`, `projects` and, per pipeline that ran on it, `results` (`state`: `done`, `running`, `waiting`, `failed`, with `error`). Also `counts` for the left side, `runs` for a project or run, and `new_in_folder` for a project that is a selection of a folder. 403 `PATH_NOT_ALLOWED` for a folder outside the shared ones; 404 `DATASET_NOT_FOUND` / `RUN_NOT_FOUND`. |
| `POST /api/v1/videos/run` | `{"paths", "selected_pipelines", "config", "name", "project_id"}`: one run (one batch) on any videos the server can read; uploads are hard-linked from their stored copy, folder videos read in place. Unusable paths are listed in `skipped`; 422 `NOTHING_TO_RUN` when none can run. With `project_id` the videos join the project and its settings become these. |
| `GET /api/v1/videos/run-settings` | What a new run starts from (`project=`, `workflow=` optional): `selected_pipelines`, `config`, and `source.kind` (`workflow`, `project`, `last_used`, `defaults`). |
| `GET /api/v1/videos/activity` | Runs with videos waiting or running, and up to five finished in the last day. |
| `POST /api/v1/videos/export/{table,elan,methods}` | Any set of videos: see [Results out](exporting_results.md#over-the-api). |
| `POST` / `DELETE /api/v1/datasets/{id}/videos` | `{"paths": [...]}`: add videos to a project from any folder, or take them out (files and results are never touched). |
| `PUT /api/v1/datasets/{id}` | Also takes `current_settings` (`selected_pipelines`, `config`, `source`). Any signed-in user may change a project; only its owner or an administrator may delete it. |
| `/api/v1/presets…` | Workflows: `export_choices` (`{"table": "csv" \| "parquet", "elan", "methods"}`) and `source` (the project or run saved from), in requests, responses and exported files. |
