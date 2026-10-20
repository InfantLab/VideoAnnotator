# Quickstart: checking Videos and Projects

**Spec**: [spec.md](spec.md) | **Contracts**: [contracts/](contracts/)

A walk-through per user story, in a browser against a real server. Record what you saw in
`checks.md`. Screenshots can be taken from the dev container (Chromium for Playwright is
installed on create).

## Setup

```bash
bash scripts/dev.sh                         # API + viewer at http://127.0.0.1:19011
mkdir -p ~/Videos/qs   # three short clips: the demo clip under three names
for n in a b c; do cp viewer/demo-assets/*Peekaboo_h265.mp4 ~/Videos/qs/clip_$n.mp4; done
```

Use a server with at least one saved dataset and one preset from before this spec (SC-003), or
make them first with v1.7.0.dev0's New job before upgrading.

## US1: a folder to a table

1. Open the viewer. With earlier results it opens on **Videos**; on a fresh server, **Home**.
2. Pick `~/Videos/qs` under **Folders**: three videos, no results.
3. **Run…** → **What to find**: tick Faces and Speech, keep the defaults → **Start** says
   "Start run on 3 videos" → press it. One confirmation, no other step.
4. A toast says the run started; the activity indicator shows `1 run · 0/3`; the folder's header
   shows its progress; rows fill in ✓ as videos finish.
5. When done: a toast with **Results table**; the run's header says "3 videos → results table".
   Download it; ELAN and Methods are beside it.
6. Nothing on the way said job, batch or dataset.

## US2: projects after the fact

1. Select two of the three rows → **Add to project ▾ → New project…**, name it "QS". It appears
   on the left with 2 videos; a toast says so.
2. Upload a fourth video through New run's **This computer**; add it to "QS" from **Uploaded
   videos**. "QS" now has 3 videos from two places.
3. Make a second project "QS2" with one video already in "QS": the row shows both chips.
4. Copy a new clip into `~/Videos/qs`: if "QS" is a selection of the folder, its header says
   "1 new video in ~/Videos/qs · Add them".
5. **Remove project** on "QS2": the confirmation says videos and results stay; afterwards they do.
6. A dataset saved before the upgrade is listed as a project with its name, videos and runs.

## US3: settings remembered

1. On "QS", **Run…**, change Faces' smallest face to 40, start.
2. **Run…** on "QS" again: "Settings from: this project's settings, last changed today"; the 40
   is there.
3. **Edit** in the header, set it to 50, save; **Run…** shows 50 without having run.
4. **Run…** on a video in no project: "Settings from: your last run" (or "the defaults").
5. **What changed?** between the two runs of "QS": "Faces · smallest face: 30 → 40".

## US4: workflows

1. On "QS", **Save as workflow** "Faces 40".
2. On Pipelines → **Workflows**: it is listed with where it came from. **Export** it to a file.
3. **Apply to project…** "QS2" (recreate it): QS2's settings say "from workflow Faces 40".
4. Rename the workflow and change it: QS2's settings stay as they were.
5. **Import workflow** from the file on another server (or after deleting it): the same.
6. A preset saved before the upgrade is listed as a workflow.

## US5: one place per video

1. Click a row: the Video page opens; ‹ › move through the same list.
2. Its **Runs** tab lists every run on the video, including runs of earlier uploads of the same
   file (R1), with status, settings, results folder, Cancel, Delete, Run again….
3. Open each old address: `/jobs`, `/jobs/<id>`, `/jobs/new`, `/batches`, `/batches/<id>`,
   `/datasets`, `/results`, `/library`, `/getting-started`: each lands somewhere meaningful.
4. A run on videos in no project: `/videos?run=<id>` shows its videos and progress.

## US6: help where the question arises

1. Every page has a line of help and **More in the docs**; follow three of them: each lands on
   the right section.
2. Hover **project**, **run**, **workflow**, and a setting's label: each has a tooltip.
3. Start, finish and cancel a run: three toasts, each with a next step.
4. Stop the server: Videos says it needs a server, how to connect, and offers Open files and
   Demos; a demo opens in the Video page.

## US7: first run

1. On a fresh server (empty database), the viewer opens on Home with three start points and the
   checklist.
2. After one finished run, opening the viewer's root lands on Videos; Home is in the nav.

## Measures

- SC-001: time a first-time user from opening the viewer to a downloaded table for a folder
  (target under 5 minutes, no docs).
- SC-002: `bun run test:run src/test/vocabulary.test.tsx` passes.
- SC-004: count the nav entries (four besides Help and Settings).
- SC-006: the route test passes, and step US5.3 by hand.
- SC-007: `bun run test:run src/test/docsLinks.test.ts` passes.
- R11: `pytest tests/unit/test_video_index_performance.py` (5,000 jobs, under 1 s).
