# Feature Specification: Viewer Overhaul (design)

**Feature Branch**: `023-viewer-overhaul` (work lands on `1.6-dev`)
**Created**: 2026-10-06
**Updated**: 2026-10-06, scope widened from the results player to the whole viewer experience;
2026-10-16, themes 7–13, research, decisions, draft proposal, journeys, and the split into
specs 026–029
**Status**: Design. Research, decisions, a draft proposal and journeys are here; it is built by
specs 026–029 (see [Split into specs](#split-into-specs)).
**Sequence**: the spec after [022](../022-videos-read-in-place/spec.md) (videos and results where
you expect them).

## Intent

Redesign the Video Annotation Viewer around what a researcher is trying to do, not around the
API's objects.

The roadmap had a visual redesign as "decided after the pilot"
([roadmap_v1.6.0.md](../../docs/development/roadmap_v1.6.0.md), Out of Scope). This placeholder
first recorded the decision to make it the next spec after 022, scoped to the results player. While
checking 022 (2026-10-06), it became clear the bigger problem is the experience as a whole.

## The problem

The viewer's structure mirrors the API, and is bewildering to a psychologist using the tool:

- **Too many nouns for one idea.** To code some videos and look at what came out, a researcher
  meets jobs, runs (batches in the API), datasets, results (the browser's local library), presets
  and prompts. Jobs, Results and Datasets each have a top-level page; the same videos appear on all
  three under different words, and the pages' subheadings (talking about "runs") add to it.
- **Pages are named for objects, not tasks.** None of them answers "where is my set of videos?" or
  "what did it find in this video?".
- **Spec 022 sharpened the clash.** Results now live in one folder per run on disk, while the
  viewer's "Results" page means something else (its own downloaded copies).
- **Use ranges from light to heavy.** Some people will just play with VLM prompts on a clip; others
  will code hundreds of videos for a paper. Today both get the same structure.

The API and spec 022's on-disk layout can stay as they are. This spec is about what the viewer
shows and how a researcher moves through it.

## Decisions (2026-10-16)

1. **Call things what they are.** A video is a "video", a folder is a "folder". A name earns its
   place only when no plain word fits, and then says what the thing is, not what it's for.
   Today's API words (job, batch) and coding-tool words (observation, task, tier) don't appear
   in the viewer unless they're the plainest word for the thing. Nor do discipline words:
   clinical psychologists say observations, experimental psychologists say participants, and a
   video may hold several of either. What the tool handles is videos and their annotations, so
   those are the words.
2. **Projects are optional, and the top level.** A project can group things, but nothing requires
   one. Nobody should have to create a project to look at a video, least of all to try a VLM
   prompt on one.
3. **Not modelled on Label Studio.** Its projects, tasks and labelling configs are confusing to
   use; nothing in this design should look or work like it.
4. **ELAN is familiar but not the model.** Researchers know it, and it stays an export target
   (spec 025). Its interface isn't practical as a template for ours; individual ideas from it
   (labelled rows, views kept in sync with the video) are taken only on their own merits.
5. **Two levels: video and project.** A project *is* the set of videos (what the strawman below
   called a dataset, and the server's saved datasets today), so there is no separate dataset
   level and nothing above the project. It holds its videos, from one folder or picked from
   several, and the runs made on them. A run is not a level: it's a record of settings applied
   to videos once, kept with the video and project it touched.
6. **A video can be in more than one project.** So a video's results belong to the video and
   the run that made them; a project shows the results of its videos rather than owning them.
7. **A saved dataset is a project.** The server's datasets become the viewer's projects. Where
   datasets carry more than a project needs, simplify them to get there; what a project needs
   that a dataset lacks (default settings, notes) is added to the API additively (Principle V).
8. **"Run" is the word** for settings applied to videos once. Alternatives clash with words
   researchers already use: "session" is a recording session, "analysis" is the statistics,
   "pass" is coding-tool jargon. Revisit only if pilot users stumble on it.

## Draft proposal (2026-10-16, for review)

Agreed in discussion so far, not yet tried on researchers. Journeys (Process step 5) will be
checked against it.

### Settings: light touch, three layers

1. **A run records what it used** (already true, spec 017). A run's settings never change, so the
   runs are the version history: no explicit versioning, and "what changed?" is "compare with an
   earlier run" ("threshold 0.7 → 0.5 since 12 Oct").
2. **A project remembers its current settings**, unasked: whatever was last used or edited
   there. A new run on the project starts from them. A video outside any project starts from the
   last settings used, or the defaults.
3. **Workflows, only when asked.** "Save as workflow" from a run or a project: pipelines, their
   settings, Ask AI prompts and sampling, export choices. Applying one to another project
   *copies* it into that project's current settings; editing the workflow later never changes a
   project silently. Workflows replace presets (the API's presets stay, Principle V) and keep
   presets' export-as-file for sharing.

The settings form (theme 9): one section per pipeline, the few settings that matter first with
defaults and units, the rest under "More", changes from the default marked, JSON behind "Edit as
JSON".

### Ask AI (the VLM window)

Named for what the researcher does: writes a question about the video. Not "Prompts" (indirect),
not "VLM" (jargon), not "AI analysis" (every pipeline is that).

**Where the model runs is a live badge, not part of the name.** "Local · qwen3.5:9b on this
computer" (or "on lab-server") when the endpoint is this machine or the lab's network; "Remote ·
frames are sent to api.openai.com", prominent and in a warning colour, if a cloud endpoint is
ever configured (v1.7.0's OpenAI-compatible connector makes that possible). A "(Local)" in the
name would turn false the first time someone did, which Principle VI rules out; the badge stays
true and, with today's Ollama-only setup, always says Local. The same badge belongs on Pipelines
and, if not there already, in the methods paragraph. This matters to researchers: participants'
video not leaving their machines is often an ethics-approval condition (Principle I).

Replaces Workbench, Prompts and the
wizard's prompt test panel, and is where a researcher may first meet rich, systematic prompting,
so results come back in the same window, with no going back and forth.

```
┌ Ask AI · p01_t1.mp4                   Local · qwen3.5:9b on this computer ┐
│ Prompt                     │                                          │
│ ┌────────────────────────┐ │                                          │
│ │ Is the infant touching │ │              video                       │
│ │ the parent? Answer...  │ │        (no overlays: just the video)     │
│ └────────────────────────┘ │                                          │
│ Model  [qwen3.5:9b ▾]      │                                          │
│ Sample every [5] s         │                                          │
│   from [0:00] to [end]     │                                          │
│   [3] frames, [0.5] s apart│                                          │
│ ≈ 120 samples, ~8 min      │                                          │
│ [Try on this frame]        │                                          │
│ [Try on 10 moments]        │                                          │
│ [Run on whole video]       │                                          │
├────────────────────────────┴──────────────────────────────────────────┤
│ touch v3  ●  ●●   ●     ●●●     ●   ●            (click: seek)        │
│ touch v2  ● ●  ●   ●   ●  ●●    ●●  ●                                 │
├───────────────────────────────────────────────────────────────────────┤
│ Table: time · frames · answer · reasoning      [Export ▾]             │
└───────────────────────────────────────────────────────────────────────┘
```

- **Prompt and settings left of the video**, to show they are applied to it.
- **No overlays.** The video stays clean; the link is the timeline: click a marker to seek, the
  sample nearest the playhead is highlighted, hover shows the frames sent and the answer.
- **Try** (current frame, or N moments) uses the existing preview endpoint and answers in
  seconds; each try is kept in a history beside the prompt and recorded in the prompt library.
  **Run on whole video** is a background run whose answers fill the lane and table as they come.
- **Prompts compared as lanes**, one above the other; the table can show them side by side.
- **Sampling**: every N seconds, over the whole video or a range, or at listed moments; per sample
  point, X frames spaced Y apart; an estimate of samples and time before starting.
- **Outputs**: timeline lane and table (spec 025 CSV) here; on the video's own page, the
  answers are also an overlay on the video. No video file is rendered (clarified 2026-10-17).

Gaps this exposes (to plan):
- Burst spacing is tied to the interval today (`burst_offsets` are multiples of
  `frame_interval_sec`), so "X frames Y apart" needs a new, additive `burst_spacing_sec`
  (default: the interval). No start/end range exists either.
- Answers filling in during a run need the VLM pipeline to write as it goes (to check: it may
  only write at the end).

### Pages

Nav: **Home · Videos · Ask AI · Pipelines**, with an activity indicator (what's running),
Help and Settings. Once set up, the viewer opens on Videos; Home keeps the first-run and no-server
start points.

**Videos and projects are one page**, like a mail client's folders and messages:

```
┌ Videos ───────────────────────────────────────────────────────────────┐
│ All videos      (212) │ Peekaboo study · 48 videos     [Run…] [Export▾]│
│ Not in a project (37) │ Settings: faces, speech, Ask AI "touch"  [Edit] │
│ ───────────────────── │ Runs: ● running 31/48 · ✓ 12 Oct · ✓ 3 Oct     │
│ Projects              │ ───────────────────────────────────────────────│
│  Peekaboo study  (48) │ ☐ Video         Folder      Faces Speech AI    │
│  Pilot lab B     (90) │ ☐ p01_t1.mp4    /data/pk    ✓     ✓      ●     │
│  + New project        │ ☐ p01_t2.mp4    /data/pk    ✓     ✓      –     │
│ ───────────────────── │ ...                                            │
│ Folders               │                                                │
│  /data/pk        (48) │ 2 selected: [Run…] [Export▾] [Add to project▾] │
└───────────────────────┴────────────────────────────────────────────────┘
```

- Left: All videos, Not in a project, each project, each folder.
- A project selected: its current settings, runs and progress, Run, Export (table, ELAN,
  methods), Compare runs. A folder or All videos: Run and Export for what's shown or selected.
- A project is made after the fact: select videos → "Add to project → New project…". A video in
  two projects shows under both.
- A multi-video run outside a project is a filter here ("videos in this run"), not its own page
  (its log aside).

| Page | What it is | Replaces |
|---|---|---|
| Home | First-run and no-server start points: ask AI about a video, run pipelines, open files from this computer | Home |
| Videos | Above | Jobs, Results, Datasets, Batch detail, Compare (runs) |
| Video | One video: player, timeline, data table beside it (theme 13), every run's results (pick the run per pipeline), notes, export, run more, its runs; previous/next within the list it was opened from | Results player, Job detail |
| Ask AI | Above | Workbench, Prompts, wizard's prompt test |
| New run (no nav entry) | Simplified wizard from a video, a project, a selection or Home; prefilled from the project's settings or a workflow | New job |
| Pipelines | What each does, installing them, saved workflows | Pipelines, presets |
| Help | Getting started rewritten, links into this repo's `docs/` | Getting started, DeepWiki links |
| Settings | Connection, token, diagnostics | Settings |

The browser's local copies of results become an invisible cache: no folder question before
playing. Standalone mode: "Open files from this computer" on Home; server pages stay visible but
say they need a server and what works without one.

### The Video page

One video: everything found in it, from any run, on one time axis, with the values behind it.

```
┌ ← Peekaboo study   p01_t1.mp4   3 of 48  ‹ ›          [Ask AI] [Run…] [Export ▾] [⋯] ┐
│                                                  │ Data │ Notes │ Runs │ About        │
│                                                  │ ─────────────────────────────────│
│              video (with overlays)               │ Lane: [all ▾]   Search [      ]  │
│                                                  │ 0:11.9  Speakers  SPEAKER_00     │
│                                                  │▶0:12.4  Speech    "Peekaboo!"    │
│                                                  │ 0:12.5  Ask AI    touching       │
│ ▶  0:12.4 / 3:20  ⏮ ⏭ event   1×   frame 372     │ 0:13.0  People    person 2 in    │
├──────────────────────────────────────────────────┴──────────────────────────────────┤
│ [Layers ▾]  [− zoom +]  ▕██████▏────────────────────────── (visible range)          │
│ 👁 Speech                     ▬▬▬▬  ▬▬▬▬▬     ▬▬                 run 12 Oct ▾       │
│ 👁 Speakers   SPEAKER_00      ▬▬      ▬▬▬                                           │
│               SPEAKER_01          ▬▬▬     ▬                                         │
│ 👁 People     person 1     ▔▔▔▔▔▔▔▔▔▔▔▔▔                                            │
│               person 2           ▔▔▔▔▔▔▔                                            │
│ 👁 Faces      count        ╱╲__╱╲_                                                  │
│ 👁 Ask AI     touch        ●  ●●   ●     ●                                          │
│    Notes      (yours)      ▲      ▬▬▬                     N: note · ⇧N: span        │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

- **Lanes are the controls.** Each lane's label shows what it is, an eye to show or hide its
  overlay on the video, and a click to collapse it; "Layers" lists them all. The separate
  controls card goes. Lanes come from what the results contain, not a fixed list, so a pipeline
  that didn't run takes no space.
- **One row per person or speaker** inside a lane, labelled, with stable colours (spec 023 fix,
  `41a38d2`). Numeric values (face count, scores, a future movement measure) are small charts.
- **Zoom and pan**, a visible-range bar, previous/next event in the selected lane (keys), hover
  for the values behind an event, frame number in the readout.
- **Data beside the video** (theme 13): the spec 025 table for this video, filterable by lane,
  searchable; the playing row is highlighted and a click seeks. Replaces "View data".
- **Which run.** Each lane says which run it shows; a video run more than once offers the others
  in the lane's menu, and two runs can be shown as two lanes to compare.
- **Notes** (theme 6): N marks a moment, Shift+N a span; notes are their own lane, visibly the
  researcher's, kept with the video and exported to ELAN as their own tier.
- **About** shows provenance (spec 017) for each lane: pipeline, model, version, settings.
- Debug, "Load different files" and the like move to the ⋯ menu.
- Opened from a project or folder, ‹ › move through it (theme 10c).

### Technical findings (2026-10-16)

- **The VLM pipeline writes only at the end** (`vlm_pipeline.py`: annotations are collected,
  then saved once). Answers filling in during a run needs it to write as it goes, or report
  answers through job progress.
- **Burst spacing is the sampling interval**: a burst frame is at `anchor + offset ×
  frame_interval_sec`. "X frames Y apart" needs a separate spacing setting.
- **No range from jobs**: `process()` takes start and end, but the job runner always passes the
  whole video (`batch/job_execution.py`).
- **Video streaming exists but isn't used**: `GET /jobs/{id}/video` serves the video with range
  requests (spec 021), and result files can be fetched one at a time. The player instead
  downloads a zip with the video inside and asks for a local folder. Streaming needs auth that a
  `<video>` element can send (a short-lived token in the URL, or a cookie).
- **Jobs record their dataset** (`dataset_id`), so a project's runs can be found without new
  links. There is no list of videos in the API: the Videos page needs one, built from jobs and
  datasets.
- **Datasets ≈ projects already**: name, description, videos, an optional server folder, and
  whether the videos are a selection from it. **Presets ≈ workflows already**: pipelines and
  their config, which includes VLM prompts and sampling.

### Split into specs

023 is the design; these specs build it. Each ships on its own, in this order:

| Spec | What | Depends on |
|---|---|---|
| [026 pipeline guide](../026-pipeline-guide/spec.md) | Pipeline metadata for researchers: what each does, behavioural vs technical settings, plain labels, homepage, benchmarks, settings hidden when they don't apply; shown on Pipelines and in New run | nothing |
| [027 Ask AI](../027-ask-ai/spec.md) | The VLM window: try, history, sampling taught on the timeline, local/remote badge; pipeline gains burst spacing, ranges and answers as they come | 026 (labels for VLM settings), shares the new timeline with 028 |
| [028 Video page](../028-video-page/spec.md) | Player, lanes as controls, zoom, data beside the video, run per lane, notes, streaming instead of the zip and folder question | none strictly; reuses 027's timeline work |
| [029 Videos and projects](../029-videos-and-projects/spec.md) | One Videos page with projects and folders, projects from datasets, settings that projects remember, workflows from presets, simplified New run, new nav, Home for first run, activity indicator, help in context | 028 (opening a video), 026 (New run) |

Not in any of them yet: a real movement measure (theme 2; roadmap v1.8.0 Phase 3 computes it on
the server), and human review of pipeline output (shelved).

## Starting point for the model (discussion of 2026-10-06, not settled; see Decisions above)

**Words.** "Study" is wrong: too high-level and too intentional. Annotating videos is one task
within a study. "Project" gives a sense of a cohesive goal, but may be too large a grouping.
"Dataset" is truthful: it claims nothing about purpose. A hierarchy may be needed.

**A strawman hierarchy, every level optional except the video** (superseded by Decision 5:
Dataset and Project merge into Project, and Run is a record rather than a level):

| Level | What it is | What might live there |
|---|---|---|
| Video | One recording | Playback; every result for it, from any run; the researcher's notes |
| Dataset | A set of videos: a folder, or a selection from one | Its videos; which videos have which results |
| Run | Settings applied to a dataset or one video, once | Progress while working; its results folder (022); the exact settings used |
| Project (optional) | A goal grouping datasets and runs | Its datasets and runs, default settings, export of everything |

Today's "jobs" would become runs of one video: visible while working, then just where a result
came from.

**Settings at more than one level.** Pipeline settings should apply widely (a project's default),
but researchers also vary them as they experiment. One idea: named, versioned settings
("recipes") that fold in today's presets and VLM prompts. Tweaking makes a new version ("Faces v2:
threshold 0.7 → 0.5") rather than a silent edit. Every run records which version it used, and
comparing two versions on the same dataset is a first-class view.

**Light-touch use first.** "Try it": one video (picked or dropped in), one pipeline or prompt,
results on the timeline, nothing to name. The VLM workbench belongs here. Structure is offered after
the fact ("run this on more videos?", "keep these together?"), never demanded up front.

**Open questions:**
1. ~~Which words, at which levels?~~ Settled by Decisions 1, 2, 5 and 8: video, folder, run,
   and an optional project as the only grouping.
2. ~~Can a video be in more than one project? Is a saved dataset a project?~~ Yes to both
   (Decisions 6 and 7). Still open: what simplifying datasets into projects means in practice.
3. Where do settings live, and how do they vary: project defaults, versions, per-run overrides?
4. What does a light-touch user see, and when does the structure appear?
5. Which pages survive, merge or go: Home, Jobs, Results, Datasets, Prompts, Workbench, Compare?
6. Which two or three journeys must be excellent for rc1 and the pilot?

## Process

This needs research and thought before a spec. Proposed steps, each written up in this folder:

1. **Inventory**: every current page, its purpose, and the terms it uses. Mark where the same thing
   has two names, or one name means two things. Done: [inventory.md](inventory.md) (pages,
   naming clashes, the results player, standalone mode, empty states).
2. **Users and tasks**: who uses it (a tinkerer with VLMs, a lab coding a corpus, a student
   reviewing someone else's results) and their main tasks, in their own words. Ask pilot labs and
   psychologist colleagues, not only us. Cold start: people can't say how they'll use a tool
   they've never had, so asking what they want of it gives little. Approach still to decide.
3. **Comparable tools**: how researchers already organise video coding work (ELAN, BORIS, Datavyu,
   Label Studio, DeepLabCut's GUI, FiftyOne, Noldus Observer). Which of their words and structures
   researchers already know. Desk research done: [comparable-tools.md](comparable-tools.md);
   hands-on tries still to do.
4. **Model and vocabulary**: settle the hierarchy, the words, and what lives at each level,
   including settings. Hierarchy and words settled (Decisions 1–8); settings (open question 3)
   still to do.
5. **Journeys and wireframes**: the chosen journeys step by step, and the navigation, as sketches.
   Try them on one or two researchers before specifying. Begun: [journeys.md](journeys.md)
   (New to AI analysis; experienced in manual coding), each checked against the draft proposal.
6. **Specify**: `/speckit-specify` from the above.

## Themes to cover (from the original placeholder; now one part of the whole)

1. **A better timeline for the existing temporal events**: speech, speakers, scenes, faces, VLM
   moments and the rest, made easier to read, compare and navigate.
2. **New metrics on the timeline**, starting with **movement per person** over time. Related:
   [roadmap_v1.8.0.md](../../docs/development/roadmap_v1.8.0.md) Phase 3 plans to compute movement
   on the server, per tracked person. The spec should decide whether to pull that forward or
   compute in the viewer first.
3. **The underlying data, easier to reach**: from any overlay or timeline event to the values
   behind it, and out to a file. Related: roadmap v1.6.0 Phase 3 (tidy export).
4. **A more modern interface** across the viewer.
5. **Better handling of multiple videos**: moving between a run's or dataset's videos, and seeing
   them together. Related: previous/next between a batch's videos (done), and roadmap v1.6.0
   Phase 6 (corpus overview).
6. **Researchers' own notes and annotations, light touch**: add a note or a simple label at a
   moment or over a span, kept with the results. Related: ELAN export (roadmap v1.6.0 Phase 3),
   so researcher annotations can travel to the tools labs already code in.

## Further themes (2026-10-16, after spec 025)

Each with what the viewer does today (checked in `viewer/src/` at `a4edb1a`).

7. **VLMs get a clear front door.** VLM prompting is probably the most popular light-touch use,
   but the Prompt workbench has no nav entry: it's reached only from a sentence on Home and a
   button on Prompts. Prompts, Workbench and Compare are three pages for one activity (try
   prompts, run one, compare runs), and only Compare is linked from a job. This belongs with
   "Try it" above.
8. **Help where the question arises, not on an external wiki.** The footer and Getting started
   link to DeepWiki for `InfantLab/video-annotation-viewer`, the repo archived when the viewer
   moved here (roadmap Phase 0), so it is both stale and out of context. Getting started still
   says "VAV". Only 5 files use a tooltip. Wanted: tooltips on terms and settings, toasts that say
   what just happened and what to do next, short inline help on each page, and docs links that
   point into this repo's `docs/` at the right section.
9. **Pipeline settings that are readable.** The job wizard's configuration step
   (`pages/NewJob.tsx`, 1,946 lines in all) shows registry-driven parameters for every selected
   pipeline at once, followed by a raw JSON editor; OpenFace 3 and the VLM have their own
   one-off controls. Wanted: settings grouped per pipeline, the few that matter shown first with
   their defaults and units, the rest behind "more", and the JSON kept for experts. This ties in
   with recipes (above) and with spec 017's provenance, which already records what was used.
10. **The results player.**
    a. *Controls take a lot of room.* `UnifiedControls` is a card with an overlay switch and a
       timeline switch per pipeline, plus lock and all-on/all-off; it hard-codes six pipelines
       (pose, faces, emotions, transcript, speakers, scenes), so new ones such as the VLM are
       handled elsewhere. Candidates: toggle lanes from the timeline itself (click a lane's
       label), a compact layers popover, keyboard shortcuts, and lanes derived from what the
       results contain rather than a fixed list.
    b. *The timeline is minimal.* Fixed lanes, the full video always (no zoom or pan), no
       hover details, no lane for faces over time or per person. The "motion" lane plots the
       sum of person-box areas per 0.1 s, which is not movement, so under Principle VI it needs
       a truthful label or a real measure (theme 2). Candidates: zoom and pan, hover for the
       values behind an event, per-person and per-speaker rows, numeric tracks (movement,
       face count, VLM scores) as small charts, jump to next/previous event, and researcher
       notes as their own lane (theme 6).
    c. *Aware of runs and datasets.* Opened from a batch, the player has previous/next
       (`BatchNavigation`); opened any other way it shows one video with no sense of where it
       belongs. Wanted: which run and dataset this video is from, its siblings, and other runs'
       results on the same video (theme 5).
11. **Exports in plain sight, methods included.** Spec 025's Export menu (table, ELAN, methods)
    appears only on Job detail and Batch detail. It is missing from the results player, the
    Datasets page (the API exports datasets) and Results. Export, and "how do I cite this?",
    should be reachable wherever results are.
12. **Reusable workflows.** What can be saved today is scattered: presets (pipelines and
    settings, on the server, exportable as JSON), datasets (lists of videos), the prompt library,
    and "Run again" from a past job. None of them joins videos, settings, prompts and export
    choices into one thing to rerun on new videos or share with a lab. Recipes (above) are the
    starting idea; the question is whether a recipe is that thing or one part of it.
13. **Data as a proper view.** "View data" (`FileViewer`) is a modal of pretty-printed JSON.
    Wanted: a table of the results (the spec 025 long-format table is the natural source),
    sortable and filterable, linked to the video (click a row to seek, playing highlights the
    row), next to the player rather than over it. This is theme 3 made concrete.

## Constraints already known

- The constitution's faithful-display principle applies: never create or hide values. Each overlay
  says which pipeline it came from (spec 017), and researcher notes stay visibly distinct from
  pipeline output.
- Standalone mode (dropping in files, no server) keeps working.
- Results are in the run folders defined by spec 022. Researcher annotations will be stored there
  too unless the spec decides otherwise.
- The API can grow to support the new model, but additively (constitution, Principle V).
