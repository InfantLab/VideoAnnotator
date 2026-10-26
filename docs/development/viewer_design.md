# Viewer Design

How the Video Annotation Viewer should look, read and behave. It outlives any one spec: specs say
what to build, this says how it should feel and the rules every page follows. Check a viewer change
against it; cite its principles (P1–P9) in reviews and UAT findings.

**Status**: Draft, 2026-10-26, from v1.7.0 UAT ([`uat_v1.7.0.md`](uat_v1.7.0.md)). Builds on
spec 023's decisions ([`specs/023-viewer-overhaul`](../../specs/023-viewer-overhaul/spec.md)) and
revises some of them; see [Decision log](#decision-log). Conventions for the code itself (where
files go, pitfalls) stay in `AGENTS.md` §23.

## Who it is for

Developmental researchers, most working alone on their own machine (AGENTS.md: 90% individual
researchers). Many are new to AI video analysis, but they know their field, they cite the tools
they use, and they will judge every number. They want to get results for a corpus, check them
against the video, and take a table into R or Python. They should never need to know how the
server, the API or the models work inside to do that.

## Principles

**P1. Show what is true.** The viewer never claims a state it hasn't checked: not connected until
a request with the key has succeeded, not "ready" until it is. When something fails, say what
failed, why (in the user's terms) and what to do next; never pass raw server text through. For
Ask AI this means **what you type is what the model receives**: no hidden instructions, schemas
or rewrites between the user and the model. Anything the viewer adds is shown and editable.

**P2. Name things by what runs, then by what it finds.** The tools are the names researchers cite
and the names they search for: pyannote, Whisper, DeepFace, OpenFace 3, YOLO26. The plain task is
the subtitle ("who spoke when"). Two pipelines can't share a name. Concepts keep plain words
(video, folder, project, run: spec 023 Decision 1 still holds for those).

**P3. Work you start stays findable.** A run, once started, is one click away from anywhere until
you've seen its result: in the nav's activity indicator (labelled, not a bare icon), on Home and on
the Runs page. A run's state is shown where its videos are shown, not only in the run's own header.

**P4. Show, don't tell.** Icons, values and layout carry meaning before words do. At most one
sentence of explanation per section, said once; no text that repeats a heading, a label or another
paragraph. A cell shows a value ("3 speakers", "12 faces"), not a tick that something ran. An icon
means what the control does (no shuffle icon on something that isn't random).

**P5. One control per job.** One seek control per video, one Run button whose scope is always
written on it ("Run on 2 selected videos"), one place to change a setting. Two controls that look
alike never do different things. Anything a keyboard shortcut does also has a visible control; shortcuts
speed things up, they never hide a feature.

**P6. Defaults are visible and safe.** Nothing is inherited silently: if settings come from a
project, a workflow or the last run, the page says so where the settings are shown. Nothing is
preselected beyond what the user asked for. Large or slow actions (many videos, many pipelines,
anything that leaves the machine) state their size and cost and ask first.

**P7. The researcher's words, not the API's.** No ids, snake_case pipeline names, endpoint or
protocol words (SSE, catalog, capabilities, schema), file paths or environment variables on
ordinary pages. They belong under an explicit "Technical details" disclosure or in Settings for the
person running the server, and even there in sentences.

**P8. One way to say each thing.** One name per thing, one date format, one time format, one
colour per kind of result, one icon per kind, used everywhere it appears: Videos, the Video page,
New run, Pipelines, Runs, exports.

**P9. Works on a laptop.** Designed at 1280 px wide, usable down to 768 px, with no sideways page
scroll at any width. Text truncates with the full value on hover; nothing spills out of its box.
Dark and light themes both work; the system setting is followed unless the user picks one.

## Information architecture

Nav: **Home · Videos · Runs · Ask AI · Pipelines**, then the activity indicator, Help and Settings.
The viewer opens on Home.

| Page | What it answers | Holds |
|---|---|---|
| Home | "What's going on, and where do I start?" | Runs going on and just finished; recent videos and projects; the three start points (Ask AI about a video, Run pipelines, Open files from this computer); setup checks only when something needs fixing |
| Videos | "What do I have, and what's been found in it?" | Videos by folder and project, a value per result kind, selection and its actions |
| Video | "What happens in this video, and when?" | Player, timeline, data beside it, notes, its runs |
| Runs | "What did I run, and how did it go?" | Every run, running first, as one row each; a run's page: its videos, results table, export, run again |
| Ask AI | "What does a vision model say about these moments?" | Question, model and where it runs, moments, answers |
| New run (no nav entry) | "What should be found in these videos?" | What to find, a summary of settings that differ from defaults, size and time, Start |
| Pipelines | "What can this server find, with what, how well?" | One card per pipeline: model, task, readiness, benchmarks, citation |
| Settings | "How is this viewer connected?" | Connection, keys, model servers for Ask AI, server facts |
| Help | "What do these words mean, how do I…?" | Getting started, glossary, links into the docs |

## Design language

### Icons

lucide-react (already a dependency). One icon per kind of result, used with its colour wherever
that kind appears: column header, lane, overlay label, New run card, Pipelines card, legend.

| Kind | Icon | Kind | Icon |
|---|---|---|---|
| Speech (Whisper) | `Captions` | Ask AI | `Sparkles` |
| Speakers (pyannote) | `AudioLines` | Notes | `StickyNote` |
| People and pose (YOLO26) | `PersonStanding` | Video | `Video` |
| Faces and expression (DeepFace) | `Smile` | Folder | `Folder` |
| Face measures (OpenFace 3) | `ScanFace` | Project | `FolderKanban` |
| Scenes | `Clapperboard` | Run | `ListVideo` |

States: running `Loader2` (spinning), waiting `Clock`, done `CheckCircle2`, failed `XCircle`,
stopped `Square`. Where it runs: on this computer `HardDrive`, elsewhere `Cloud`. An icon alone is
never the only carrier of meaning: it has a label or a tooltip, and an `aria-label`.

### Colour

- **One hue per kind of result**, the same in a column, a lane, an overlay and a legend. Tracks
  within a kind (speaker 1, speaker 2; person 1, person 2) are shades or a categorical palette
  within the kind, never another kind's hue.
- **State colours mean state only**: green done, red failed, blue running, grey waiting or
  stopped. Red is never used for an ordinary answer or a first series.
- Both themes from the same tokens; no light-only boxes in the dark theme.

### Words, numbers and time

- Dates: `26 Oct 2026, 17:50`; today shows `17:50`; this year drops the year. Relative times
  ("2 hours ago") only in lists, with the absolute time on hover.
- Positions in a video: `m:ss.s` (`0:12.4`), never `12.40 s`. Durations: `24 s`, `4 min`,
  `1 h 12 min`.
- Counts in words a researcher would say: "1 video", "3 speakers", "2 failed".
- Numbers keep only meaningful precision: confidence `0.91`, not `0.9865731`.

### Names

- **Pipelines**: display name is the model or tool (with the version where it changes results:
  "pyannote community-1", "Whisper large-v3-turbo"); the subtitle is the task ("who spoke when").
  Comes from the registry metadata, one field, used everywhere (P8). Where the model is chosen per
  run, results are labelled with the model that ran (from provenance), not the pipeline default.
- **Runs**: the name the user gave; otherwise "<video or folder> · <what was found>"
  ("expCP2 · Scenes", "peekaboo · pyannote, Whisper"). The date is shown beside the name, never
  folded into it; no "(2026-10-26 3)" suffixes.
- **Videos**: the file name, truncated in the middle so both ends stay visible
  ("2UWdXP.joke1.rep2…Peekaboo_h265.mp4", not "2UWdXP.joke1.rep2.take1.Peekaboo…"). Copies of the
  same file are told apart by folder or upload date.

## Patterns

**Run row** (Runs, Home, a project, the activity menu): state icon · name · videos (count, first
names) · what was found (icons or names) · outcome ("48 done", "46 done, 2 failed: why?") · when ·
how long. A progress bar only while running. One action menu, not a row of buttons.

**Status in place.** A video's row and a video's lanes show running, waiting and failed for the run
in progress, so following a run never requires going back to the run.

**Selection.** While anything is selected, the selection bar replaces the page's own actions;
every action says its scope ("Run on 2 videos").

**Settings.** Show a short summary of what differs from the defaults and where the settings came
from ("from the project peekaboo"; "from your last run, expCP2 · Scenes"). Full settings one click
away, grouped per pipeline, each setting named once with its unit and range.

**Empty, loading and error states.** Every list and panel has all three. Errors say what
happened, why in the user's terms, and the next step as a link or button ("Connect with your
key", "Start Ollama", "Run again"). Server and environment details go under "Technical details".

**Disclosure.** Ordinary pages show what a researcher needs; technical detail (paths, ids,
settings hashes, raw files) sits behind one consistent "Technical details" disclosure.

### Ask AI

- Answers are **free text by default**, because the model answers in free text. The page says so
  plainly, next to the question: answers come back as the model writes them, and to get
  structure you ask for it in the question ("Answer yes or no", "Reply as JSON with keys …").
  Examples show how.
- **What you type is what is sent** (P1). Any helper that adds to the request (a structured-answer
  template, an instruction to reply as JSON, extra frames for context) writes visibly into the
  question or sits beside it as a labelled, editable part of what is sent; nothing is added out
  of sight.
- **Fields is a helper, not a constraint** (D5). Picking fields writes the instruction into the
  question box ("Answer as JSON: {"touching": "yes" or "no"}"), where it can be read and edited;
  nothing else is sent. It stays small: a few plain kinds (choice, yes/no, number, text), names
  as the user types them, no rules to satisfy before asking. A reply that doesn't follow the
  instruction is shown as written and marked as not fitting, which is the truth about the model.
- **Forcing the format is a deliberate choice.** Where the model server can restrict replies to a
  format (Ollama's `format`, OpenAI-compatible `response_format`), one tick box offers it, off by
  default, saying what it costs: "Only let the model answer in this format: reliable columns, but
  the model can't say when it's unsure." Where the server can't, the box isn't shown. The choice
  is recorded with the answers.
- When an answer is valid JSON, the viewer shows it in columns and says it parsed the model's
  JSON; when it isn't, it shows the text as written. Answers are never silently rewritten.
- Model and where it runs (this computer, the lab's server, a hosted service) sit together, next
  to the question.

### The Video page

- **One seek control.** The timeline's playhead is the way to move through the video; frame and
  second step buttons sit beside the time readout. The timeline's visible range shows only when
  zoomed and looks like a scrollbar, not a second progress bar.
- Zoom and fit controls sit at the timeline's right end, next to the range readout.
- The data beside the video is summarised by default (one row per event: speech turn, speaker
  turn, scene, answer), with the full measures (per-frame scores, all emotion probabilities) one
  switch away.

## Decision log

| # | Date | Decision | Revises |
|---|---|---|---|
| D1 | 2026-10-26 | Pipelines are named by model first, task second (P2) | Spec 023 Decision 1 for pipelines; spec 026 guide titles as headings |
| D2 | 2026-10-26 | Runs get their own page in the nav; a run also stays a filter of Videos | Spec 029 R12 ("a run is a filter of Videos, not its own page") |
| D3 | 2026-10-26 | Home is a dashboard and the viewer opens on it | Spec 023/029: Home only for first run and no server; set-up viewer opens on Videos |
| D4 | 2026-10-26 | Ask AI answers are free text by default; structure comes from the prompt, visibly; nothing hidden is added to what the model receives (P1) | Spec 033 answer shapes as a separate, hidden structured layer |
| D5 | 2026-10-26 | Fields writes its instruction visibly into the question; forcing the format (server-side structured output) only by an explicit tick box, off by default, offered where the server supports it. Fields helps, never restricts: if it grows rules or steps, cut it back | Spec 033 R8 (schema sent by default; a hidden sentence appended for servers that can't enforce) and its field-name rules (UAT #29) |

## Open

- **Icon set**: the table above is a first pass; check it reads at 16 px in both themes.
- **Data table default**: summarised view as proposed, or the raw rows with better filters.
- **H.265**: make an H.264 copy for viewing on the server, or keep telling users to re-encode.
