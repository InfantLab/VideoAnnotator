# Feature Specification: Viewer Overhaul (placeholder)

**Feature Branch**: `023-viewer-overhaul` (work lands on `1.6-dev`)
**Created**: 2026-10-06
**Updated**: 2026-10-06, scope widened from the results player to the whole viewer experience
**Status**: Placeholder. Not yet specified. Needs research and design thinking first (see
[Process](#process)); only then run `/speckit-specify` with
`SPECIFY_FEATURE_DIRECTORY=specs/023-viewer-overhaul`.
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

## Starting point for the model (discussion of 2026-10-06, not settled)

**Words.** "Study" is wrong: too high-level and too intentional. Annotating videos is one task
within a study. "Project" gives a sense of a cohesive goal, but may be too large a grouping.
"Dataset" is truthful: it claims nothing about purpose. A hierarchy may be needed.

**A strawman hierarchy, every level optional except the video:**

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
1. Which words, at which levels: project, dataset, run, video? Is a project optional?
2. Can a dataset belong to more than one project?
3. Where do settings live, and how do they vary: project defaults, versions, per-run overrides?
4. What does a light-touch user see, and when does the structure appear?
5. Which pages survive, merge or go: Home, Jobs, Results, Datasets, Prompts, Workbench, Compare?
6. Which two or three journeys must be excellent for rc1 and the pilot?

## Process

This needs research and thought before a spec. Proposed steps, each written up in this folder:

1. **Inventory**: every current page, its purpose, and the terms it uses. Mark where the same thing
   has two names, or one name means two things.
2. **Users and tasks**: who uses it (a tinkerer with VLMs, a lab coding a corpus, a student
   reviewing someone else's results) and their main tasks, in their own words. Ask pilot labs and
   psychologist colleagues, not only us.
3. **Comparable tools**: how researchers already organise video coding work (ELAN, BORIS, Datavyu,
   Label Studio, DeepLabCut's GUI, FiftyOne, Noldus Observer). Which of their words and structures
   researchers already know.
4. **Model and vocabulary**: settle the hierarchy, the words, and what lives at each level,
   including settings.
5. **Journeys and wireframes**: the chosen journeys step by step, and the navigation, as sketches.
   Try them on one or two researchers before specifying.
6. **Specify**: `/speckit-specify` from the above.

## Themes to cover (from the original placeholder; now one part of the whole)

1. **A better timeline for the existing temporal events**: speech, speakers, scenes, faces, VLM
   moments and the rest, made easier to read, compare and navigate.
2. **New metrics on the timeline**, starting with **movement per person** over time. Related:
   [roadmap_v1.7.0.md](../../docs/development/roadmap_v1.7.0.md) Phase 3 plans to compute movement
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

## Constraints already known

- The constitution's faithful-display principle applies: never create or hide values. Each overlay
  says which pipeline it came from (spec 017), and researcher notes stay visibly distinct from
  pipeline output.
- Standalone mode (dropping in files, no server) keeps working.
- Results are in the run folders defined by spec 022. Researcher annotations will be stored there
  too unless the spec decides otherwise.
- The API can grow to support the new model, but additively (constitution, Principle V).
