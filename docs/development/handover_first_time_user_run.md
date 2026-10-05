# First-time user run, 2026-10-05

Phase 2, item E: follow the README as a new researcher would, on the one path
install → add videos → run → review → export, and note every point of friction.

## How it was run

- A fresh clone of `1.6-dev` at `6a86ba1` (the README rewrite), in a scratch folder.
- A fresh `HOME`, so the database, API key, model weights and logs all started empty. None of
  the dev container's VideoAnnotator variables were set. uv's package cache was shared, which
  only made `uv sync` faster.
- Exactly the README's Linux steps: `uv sync`, then `uv run videoannotator server`. The port
  was `--port 18111`, because 18011 was taken by another server.
- The viewer was driven with Playwright (headless Chromium), from the link the server printed.
- The video was the 10-second demo clip as VP9 WebM, since headless Chromium can't play H.264.
- The pipeline was Scene Detection, installed from the viewer as the README says.

## What happened, in order

| Step | Result |
|---|---|
| `uv sync` | 4 s with a warm cache; 246 MB `.venv` |
| Server start | Database, first API key and one-click link printed. **The link named port 18011, not 18111.** Fixed |
| Open the link | Logs in and lands on Home: "0 of 7 pipelines ready" and the setup checklist. Good |
| New job → upload | Works. The native file input sits on its own line under "or pick files:" |
| Choose pipelines | **Every pipeline said "requires an administrator API key"** although the server's own first key asks for admin. Fixed |
| Install Scene Detection | **Failed in 1 s on Linux** (uv: no solution; torch 2.11 cu126 vs PyPI's cu13). Fixed. Then completed and activated without a restart; it took 25 min here, almost all downloading about 3 GB from the PyTorch index |
| Submit | Lands on the run (batch) page, which says the first run loads models and may download weights. Good |
| While the job ran | 344 polls of `GET /api/v1/jobs/{id}` over about 3 min: median 25 ms, slowest 349 ms. No stall |
| Job page | Clear: Run it again, Open in Viewer, Download Results |
| Review | "Choose a folder for your results" first, with "View without saving". Then the video plays with "Scene: clinic" over it |
| Export (zip) | **Held only the JSON: the WebM video was left out**, and the page promised a job log that is never written. Fixed |

## Fixed in this pass (commit `084ba61` and the one after it)

- The first-start key's user is now an admin (the key always asked for the `admin` scope).
- Extras installs add the project's CUDA index when the lock needs it (Linux torch).
- `server --port` reaches the first-start viewer link.
- Results zip: every accepted video format; no promise of a job log.
- The pipeline list no longer shows the variant as a version ("vpyscenedetect-clip").
- No `pyannote.core` warning at start without the audio extra.
- The "lost this key" hint also gives the `uv run` form for a clone.

## Not fixed: for a decision or a later pass

1. **Job files live in the folder the server was started from** (`./storage/jobs/...`, inside
   the clone here). The database, models and logs live in the per-user data folder (spec 016).
   Start the server from another folder and earlier jobs lose their videos and results. Suggest:
   the same per-user default as the database, with `STORAGE_ROOT` still overriding it. That needs
   a migration note for existing `./storage` folders.
2. **A new database logs four "[MIGRATION] Adding jobs... column" lines.** The users and
   datasets layer (`database/`) creates `jobs` from its own model first, then the job storage
   layer adds its columns. It's harmless, but it's the two-database-layers problem showing.
3. **Installing a pipeline can take a long time** (25 min for `scene` here, download-bound).
   The viewer shows the size but no time hint, and nothing tells the user it's fine to leave the
   page.
4. **Wizard wording and layout**: the subtitle "Process videos through the VideoAnnotator
   pipeline (supports batch processing)"; the file input on its own line.
5. **Review asks for a results folder before showing anything.** "View without saving" is the
   secondary button. A first-time user probably wants to see the result first and decide about
   keeping it later.
6. **The viewer lists pipelines the job didn't run** (Person, Face, Emotion, Speech, Speaker, each
   "(No data)") above the one it did.
7. **Scene label quality**: a home kitchen was labelled "clinic". This is the CLIP label set, a
   Phase 5 benchmark question.
8. **The job page's Configuration shows `{}`** when a pipeline ran with its defaults; it could
   show the defaults it used (provenance has them).
9. **The structural pass** ("anything off the path moves or goes") needs Caspar: candidates are
   the Settings → Getting Help tab, the Getting Started page (overlaps Home's checklist), and
   the jobs/batches double naming ("runs", "batch", "jobs").
10. **Not covered by this run**: Windows and macOS installs, the GPU path, a long video (the
    API-responsiveness item wants one), speaker diarization's Hugging Face token step, the VLM
    pipeline with Ollama, and the datasets path.
