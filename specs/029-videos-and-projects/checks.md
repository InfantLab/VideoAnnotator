# Checks: Videos and Projects (spec 029)

## Walk-through (T064), 2026-10-21

A browser (Playwright's Chromium in the dev container) against a real server started fresh
(`AUTH_REQUIRED=false`, no key, background processing on, one shared folder holding `qs/` with
the demo clip copied as `clip_a`, `clip_b`, `clip_c`), following [quickstart.md](quickstart.md).
Scripts and screenshots were kept in the session's scratch folder, not the repository.

| Story | What happened |
|---|---|
| US1 | The fresh server opened on **Home**. **Videos → Folders → …/shared**, then **Folders inside → qs**: 3 rows, no results. **Run…**: "Settings from: the defaults"; only Scenes ticked; **Start run on 3 videos**, the one press. Landed on the run (`/videos?run=…`), the indicator read "1 run · 0/3", a message said it started; after 16 s "**3 videos → results table**" with Download, ELAN, Methods, and a "finished" message with **Results table**. The download `qs (2026-10-21).csv` had 27 rows, 9 per video. **28 s** from opening the viewer to the downloaded table (scripted, not a first-time user: SC-001 still needs a person). |
| US2 | Two rows → **Add to project → New project… "QS"**: message "Project QS made · 2 videos in it", QS (2) on the left. A video from "this computer" uploaded through New run, run, then added to QS from **Uploaded videos**: QS holds 3 videos from two places. A second project QS2 with `clip_a`: its row shows both chips. **Remove project** on QS2: "Only the project goes. Its videos and all their results stay…"; afterwards All videos still 3. |
| US3 | A project made from a selection has no settings of its own: its header says "none of its own yet: your last run", and New run says "Settings from: your last run (qs)". **Edit** → **Save settings**: "Settings saved for QS · The next run starts from them"; the next New run says "Settings from: this project's settings, last changed Oct 21". |
| US4 | **Save as workflow** on QS, "Scenes only": message with **Workflows**; Pipelines → Workflows lists it "from the project QS" with Apply to project, Use in a new run, Export, Rename, Delete. |
| US5 | A row opens the Video page with `from=all`; its **Runs** tab lists the run. `/jobs`, `/batches`, `/datasets`, `/results`, `/library` → `/videos`; `/jobs/new` → `/run`; `/getting-started` → `/help`. After the first finished run, `/` opens Videos. |
| US6 | Help has the words, the start points, the demos and 10 links into the docs. No page visited (Home, Videos in every scope, New run, a run, the Video page, Pipelines, Ask AI, Settings, Help) showed job, batch, dataset or preset. |
| US7 | Fresh server: Home with the three start points and the setup checklist; after one run, the root opens Videos. |

### Found and fixed during the walk-through

1. **Three identical clips showed as one video.** The demo clip copied under three names has one
   content hash, and R1 joined paths by recorded `sha256`; the run said "3 videos" but the table
   showed one row. Now a hash joins paths only through an upload; copies in shared folders stay
   separate videos (research R1 amended; `test_identical_copies_in_a_shared_folder_stay_separate_videos`).
2. **A new folder inside a shared folder couldn't be reached.** On a fresh server the left side
   lists only the shared folder itself; `GET /videos?folder=` now returns `subfolders` and the
   header offers them (**Folders inside**).
3. **Projects couldn't be made without sign-in.** With `AUTH_REQUIRED=false` and no key, saving a
   dataset needed an identity ("Sign in with your API key"); projects and workflows are now owned
   by a local anonymous user there (`api/owner.py`; not counted by `generate-token`). With sign-in
   required, an identity is still needed.
4. A project with no settings of its own said "the defaults until it runs" while New run started
   from the last run; it now says which.

### Still for a person

- **SC-001**: time a first-time user from opening the viewer to a downloaded table (target under
  5 minutes, no docs). The scripted path took 28 s of server and browser time.
- Spec 023's journeys with pilot users (layout may change then).
- Ask AI's runs and **Compare answers** with a vision model (Ollama was not reachable here).
- Hover checks of tooltips (`Term`, settings) by eye; they are covered by tests.

## Measures

- **SC-002** (vocabulary): `src/test/vocabulary.test.tsx` renders 16 addresses loaded, empty and
  failing (48 cases); none shows job, batch, dataset or preset.
- **SC-004** (nav): four entries, Home, Videos, Ask AI, Pipelines, then the runs, Help, Settings
  (`AppRender.test.tsx`).
- **SC-006** (old addresses): `src/test/routes.test.tsx`, 20 cases, and by hand above.
- **SC-007** (docs links): `src/test/docsLinks.test.ts`; the built docs (`mkdocs build --strict`)
  have every anchor the viewer links to.
- **R11** (speed): `tests/unit/test_video_index_performance.py`, 5,000 jobs over 2,000 videos
  indexed and listed in 0.26 s (target 1 s).
- **Bundle** (T063): initial JavaScript 158.7 KB + 71.3 KB gzipped, CSS 13.6 KB: within 300 KB.
  Videos (16.4 KB) and New run are their own chunks.

## Gates

2026-10-21, at the end of the build:

- `pytest tests/`: 2,143 passed, 7 skipped. `ruff check .`, `ruff format`, `mypy
  src/videoannotator`: clean.
- Viewer: `bun run lint`, `bun run typecheck` clean; `bun run test:run` 997 passed, 1 skipped;
  `scripts/build_viewer.sh --check`: bundle up to date.
