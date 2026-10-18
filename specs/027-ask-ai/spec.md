# Feature Specification: Ask AI

**Feature Branch**: `027-ask-ai` (work lands on `1.7-dev`)
**Created**: 2026-10-16
**Status**: Draft
**Design**: [spec 023](../023-viewer-overhaul/spec.md) (Draft proposal: Ask AI; Decisions 1–8),
journey 1 in [journeys.md](../023-viewer-overhaul/journeys.md)
**Input**: User description: "A researcher new to AI video analysis gives it a video, asks
questions about it, and gets systematic answers back, quickly, in the same window. They play with
it a few times before running it on their full videos. They must be taught what sampling means
(every few seconds; sometimes a burst of frames), whether the AI is local or remote, and that the
prompt has to say what form the answer takes. Prompts they've tried must never be lost. A full
sampling interface (every N seconds, X frames Y apart), and outputs to a timeline, a table and a
video." (Clarified 2026-10-17: "to a video" means shown over the video in the main viewer, not a
rendered video file.)

## Context

Asking a vision-language model about a video is likely the most popular first use of
VideoAnnotator, and for many researchers their first contact with systematic prompting. Today it
is spread over four places: a test panel inside the job wizard, the Workbench (not in the nav),
the Prompts library, and Compare. Results come back on a different page from the one where the
prompt was written.

This spec makes it one window, **Ask AI** (spec 023 names it for what the researcher does), with
the prompt and settings to the left of the video they apply to, and the answers on a timeline
and in a table below. It also closes three gaps in the pipeline the design depends on (spec 023
technical findings): burst frames are spaced by the sampling interval, jobs can't be limited to a
range, and answers only appear when a run finishes.

### What the researcher sees

```
┌ Ask AI · p01_t1.mp4                         Local · qwen3.5:9b on this computer ┐
│ Question                    │                                                   │
│ ┌─────────────────────────┐ │                                                   │
│ │ Is the infant touching  │ │                 video                             │
│ │ the parent? Answer with │ │          (no overlays: just the video)            │
│ │ one word: touching or   │ │                                                   │
│ │ not_touching.           │ │                                                   │
│ └─────────────────────────┘ │                                                   │
│ Examples ▾   History (14) ▾ │                                                   │
│ Model   [qwen3.5:9b ▾]      │                                                   │
│ Sample every [5] s          │                                                   │
│   from [0:00] to [end]      │                                                   │
│   [3] frames, [0.5] s apart │                                                   │
│ 40 samples · about 3 min    │                                                   │
│ [Try on this frame]         │                                                   │
│ [Try on 10 moments]         │                                                   │
│ [Run on whole video]        │                                                   │
├─────────────────────────────┴───────────────────────────────────────────────────┤
│ samples   |    |    |    |    |    |    |    |    |    |    |    |    |    |    │
│ touch v3  ●         ●    ●              ●                        (click: seek)  │
│ touch v2  ●    ●              ●    ●                                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│ time   frames  touch v3      touch v2      reasoning       Answers: touching 9, │
│ 0:05   3       touching      touching      The infant's…   not_touching 31      │
│ 0:10   3       not_touching  touching      …               [Export ▾]           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Clarifications

### Session 2026-10-18

- Q: How does "Try on 10 moments" choose its moments? → A: Spread evenly over the current
  sampling's sample points (every k-th sample in the start–end range), so they land where a run
  would sample.
- Q: What happens to a try while a run is using the same model? → A: The try goes ahead of the
  run's remaining samples; the run pauses between samples and carries on afterwards.
- Q: Does the history keep copies of the frames sent? → A: No. It keeps times and settings;
  frames are extracted again from the video when shown.
- Q: Whose history is it when several people share one server? → A: Shared on the server, like
  the prompt library; each entry records who made it, filterable to "mine"; stars and names are
  shared.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - First answer in under a minute (Priority: P1)

A researcher who has never done this opens Ask AI from Home, drops in a video (or picks one from a
folder), and starts from an example question. They press "Try on this frame" and get an answer
and the model's reasoning beside the video within seconds. Nothing asked them to make a project
or name anything.

**Why this priority**: Journey 1's first need is "very quickly pass in a video and ask questions
about it". Everything else builds on this.

**Independent Test**: From a fresh install with a local model, a first-time user gets an answer
on their own video without reading documentation.

**Acceptance Scenarios**:

1. **Given** Home, **When** the user chooses "Ask AI about a video", **Then** they can drop a
   video, upload one, or pick one from a server folder, and Ask AI opens on it.
2. **Given** an empty question box, **When** Ask AI opens, **Then** it offers example questions,
   each of which says what form the answer must take, and inserts one on click.
3. **Given** a question and a frame, **When** "Try on this frame" is pressed, **Then** the answer,
   the reasoning and the frames sent appear in the window, and the frame's marker appears on the
   timeline.
4. **Given** no model is available, **When** Ask AI opens, **Then** it says what's missing and how
   to get it (install the local model server; pull a model), not an error code.

---

### User Story 2 - Never lose a prompt (Priority: P1)

The researcher tries a dozen wordings. Every try is kept, with its question, model, sampling and
answers, in a history beside the question. Coming back the next day, the history is there. One
click restores an earlier question and its settings; a star or a name marks the ones that work.

**Why this priority**: Journey 1: "they don't want to lose a sequence of prompts… finding a formula
that works… doesn't want to be lost if we move on".

**Independent Test**: Make five tries, close the browser, reopen Ask AI: all five are listed and
any one can be restored exactly.

**Acceptance Scenarios**:

1. **Given** any try or run, **When** it completes, **Then** it is in the history without the user
   doing anything, with question, model, sampling, the moments tried, and the answers.
2. **Given** the history, **When** an entry is chosen, **Then** its question and settings replace
   the current ones, and the replaced ones stay in the history.
3. **Given** two entries, **When** both are shown, **Then** their answers appear as two lanes and
   two table columns on the same moments.
4. **Given** a try on another video, **When** the history is opened, **Then** tries are listed for
   all videos, filterable to this one.
5. **Given** a server shared by several people, **When** the history is opened, **Then** everyone's
   entries are listed with who made each, filterable to the user's own.

---

### User Story 3 - Understand sampling by seeing it (Priority: P1)

The researcher doesn't know the video is sampled. As they set "every 5 s", tick marks appear on
the timeline where samples will fall; choosing "3 frames, 0.5 s apart" shows the three frames for
the current sample as thumbnails. A one-line explanation sits beside the controls, with a link to
more. Before a run, the window says how many samples and roughly how long.

**Why this priority**: Journey 1 names sampling as the first thing they won't understand, and the
cost (time) of a run depends on it.

**Independent Test**: Change each sampling control and check the ticks, thumbnails and estimate
update immediately, before anything is sent to the model.

**Acceptance Scenarios**:

1. **Given** sampling settings, **When** any changes, **Then** the timeline's sample ticks and the
   sample count update without contacting the model.
2. **Given** burst sampling, **When** the playhead is near a sample, **Then** the frames that
   sample would send are shown as thumbnails, with their times.
3. **Given** at least one try, **When** the estimate is shown, **Then** it uses the measured time
   per answer on this model; before any try, it says it's a rough guess.
4. **Given** listed moments instead of an interval (e.g. "0:12, 0:47, 1:30"), **When** set,
   **Then** only those are sampled.

---

### User Story 4 - Know where the video goes (Priority: P1)

A badge at the top says where the model runs: "Local · qwen3.5:9b on this computer" (or "on
lab-server"). The first time, it explains: "Frames from your video are sent only to this
computer." If a remote endpoint is ever configured, the badge says "Remote · frames are sent to
api.example.com" in a warning style, and the first send asks for confirmation.

**Why this priority**: Journey 1 requires local vs remote to be "very clearly signposted";
participants' video staying local is often an ethics condition (Principle I).

**Independent Test**: With a local endpoint the badge says Local; point it at a non-local host
and the badge changes and the first send asks first.

**Acceptance Scenarios**:

1. **Given** an endpoint on this machine or a private network address, **When** Ask AI is open,
   **Then** the badge says Local and names the model and host.
2. **Given** any other endpoint, **When** Ask AI is open, **Then** the badge says Remote, names the
   host, and is styled as a warning.
3. **Given** a remote endpoint not yet confirmed for this server, **When** the first try or run
   starts, **Then** the user must confirm, and the confirmation is remembered per endpoint.
4. **Given** a run, **When** its provenance is recorded, **Then** it says whether the endpoint was
   local or remote.

---

### User Story 5 - Run on the whole video, answers as they come (Priority: P2)

Happy with a question, the researcher presses "Run on whole video". The run goes in the
background; answers fill the lane and table as each sample is answered; they can watch, keep
trying other questions, or stop it. When it finishes, the results are a normal run (spec 022
folder, spec 017 provenance, spec 025 table).

**Why this priority**: The goal is systematic answers; trying is the means.

**Independent Test**: Start a run on a 2-minute clip; answers appear in the window before the run
ends; stopping keeps the answers so far.

**Acceptance Scenarios**:

1. **Given** a run in progress, **When** a sample is answered, **Then** it appears in the lane and
   table within a few seconds, without reloading.
2. **Given** a run in progress, **When** "Stop" is pressed, **Then** the answers so far are kept
   and the run is marked stopped, not failed.
3. **Given** a finished run, **When** "Run on more videos" is chosen, **Then** run setup opens on
   a folder or selection with this question and sampling filled in.

---

### User Story 6 - Answers out (Priority: P2)

From the window, the researcher exports the answers as a table (one row per sample: time,
frames, question, answer, reasoning, model). A run's answers also show on the video's own
page (spec 028) as a lane and as a caption on the video at each moment. A tally of distinct answers sits by
the table; when answers don't fall into a few values, a hint says to state the allowed answers in
the question.

**Why this priority**: Journey 1's systematic answers, and the prompt-defines-output lesson.

**Independent Test**: Export the table of a run and open it in R; distinct answers match the tally.

**Acceptance Scenarios**:

1. **Given** answers, **When** "Export → Table" is used, **Then** the spec 025 table for these
   answers is downloaded.
2. **Given** more than 10 distinct answers in a run or try set, **When** the tally is shown,
   **Then** the hint appears.
3. **Given** a finished run, **When** the video's page (spec 028) is opened, **Then** its answers
   are a lane there, and the answer for the current moment is shown on the video, toggled like
   any other overlay.

---

### Edge Cases

- Repeating a try on the same frame can give a different answer; the history keeps both, and the
  table marks moments tried more than once with differing answers.
- A sample whose frames can't be read is listed as skipped, not answered (Principle VI).
- A model error on one sample shows as an error marker in the lane, distinct from an answer.
- A burst that would reach before the start or past the end of the video uses the frames that
  exist and says how many were sent.
- A video already in a project opens Ask AI with that project's current Ask AI settings (spec 029).
- Standalone mode (no server): Ask AI says it needs a server and a model, and what to do.
- A try started during a run on the same model is answered before the run's next sample; the
  run's answers so far and its progress are unaffected.
- A history entry whose video has been moved or deleted keeps its question, settings and answers,
  and says its frames are unavailable; it can still be restored onto another video.
- Very long videos: the estimate warns when a run would take more than an hour.

## Requirements *(mandatory)*

### Functional Requirements

**The window**
- **FR-001**: The viewer MUST offer Ask AI as a top-level destination, opening on a video chosen
  by drop, upload, or server folder, with no project or name required.
- **FR-002**: Ask AI MUST place the question, model and sampling to the left of the video, and the
  timeline and table below, in one window.
- **FR-003**: The video MUST play without overlays; the link between video and answers is the
  timeline: clicking a marker or table row seeks; the sample nearest the playhead is highlighted
  in both; hover shows the frames sent and the answer.
- **FR-004**: Ask AI MUST replace the Workbench page, the Prompts page and the job wizard's prompt
  test panel; their old addresses MUST lead to Ask AI.

**Trying and history**
- **FR-005**: "Try on this frame" and "Try on N moments" MUST return answers in the window without
  creating a run. The N moments (10) MUST be sample points of the current sampling, spread evenly
  over them (every k-th sample within start and end; all of them when there are N or fewer), so
  a try's answers fall on the same moments as a run's and a repeated try on the same settings
  uses the same moments.
- **FR-006**: Every try and run MUST be kept in a persistent history (question, model, sampling,
  moments, answers, time), restorable in one action, with star and name; the existing prompt
  library's records MUST appear in it. It MUST NOT store frames or images from the video: frames
  are extracted again from the video when shown. It MUST be shared by everyone on the server,
  like the prompt library: each entry records who made it, the history can be filtered to the
  current user's entries, and stars and names are shared.
- **FR-007**: Any two history entries on the same video MUST be comparable as two lanes and two
  table columns.
- **FR-008**: Ask AI MUST offer example questions, each stating the form of the answer.
- **FR-009**: Ask AI MUST show a tally of distinct answers and the hint of User Story 6 when more
  than 10 distinct answers occur.

**Sampling**
- **FR-010**: Sampling MUST support: every N seconds; a start and end within the video; a list of
  moments; and per sample point, a number of frames and their spacing in seconds, independent of
  N.
- **FR-011**: The timeline MUST show where samples fall, and the frames of the current sample as
  thumbnails, updated as settings change, before anything is sent.
- **FR-012**: Ask AI MUST show the sample count and an estimated duration before a run, based on
  measured time per answer when available.
- **FR-013**: The VLM pipeline MUST accept the new sampling settings (frame spacing, start and
  end, listed moments) for runs as well as tries; existing configs MUST keep their meaning
  (spacing defaults to the interval, so `burst_offsets` behave as today).

**Local or remote**
- **FR-014**: Ask AI MUST show a badge naming the model and where it runs: Local when the
  endpoint is this machine or a private network address, Remote otherwise.
- **FR-015**: A Remote endpoint MUST be styled as a warning and require confirmation before its
  first use, remembered per endpoint.
- **FR-016**: Run provenance MUST record whether the endpoint was local or remote.

**Runs**
- **FR-017**: "Run on whole video" MUST start a background run whose answers appear in the window
  as each sample is answered.
- **FR-017a**: A try MUST NOT wait behind a run on the same model: the run's next sample waits
  while the try is answered, and the run then carries on. The run's estimate and progress MUST
  account for the pause.
- **FR-018**: A run MUST be stoppable, keeping the answers so far and recording that it was
  stopped.
- **FR-019**: A finished or stopped run MUST be a normal run: its results folder (spec 022),
  provenance (spec 017), and table and methods (spec 025).
- **FR-020**: Ask AI MUST offer "Run on more videos", opening run setup with the question and
  sampling filled in, and "Save as workflow" (spec 029).

**Out**
- **FR-021**: Ask AI MUST export the answers shown as the spec 025 table. No video file is
  rendered: a run's answers are shown over the video on its Video page (spec 028 FR-022).

### Key Entities

- **Try**: a question, model and sampling applied to some moments of a video, not a run; kept in
  the history.
- **History entry**: a try or a run, with everything needed to restore or compare it; star and
  name optional, shared by everyone on the server. Records who made it; refers to its video and
  moments by time; holds no frames.
- **Sampling**: interval, start, end, listed moments, frames per sample, frame spacing.
- **Endpoint**: the model server's address, its classification (local or remote), and whether
  remote use was confirmed.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A first-time user with a local model gets a first answer on their own video in under
  a minute from opening Ask AI, without documentation.
- **SC-002**: No try or run is ever lost: 100% of tries made are in the history after closing and
  reopening the browser.
- **SC-003**: Changing a sampling setting updates the ticks, thumbnails and count in under half a
  second, with no model call.
- **SC-004**: During a run, each answer appears in the window within 5 seconds of the model
  returning it.
- **SC-005**: Shown the badge, users correctly say whether their video leaves the computer
  (target: all test users).
- **SC-006**: Researchers can get from a working question to a run on a folder of videos in two
  actions.

## Assumptions

- The model server is Ollama today (spec 009); the local/remote rule is about the endpoint's
  address, so it holds for the OpenAI-compatible connector when it arrives.
- Tries use the existing preview endpoint (spec 020), extended for the new sampling settings.
- The history extends the prompt library (spec 020) rather than adding a separate store.
- The timeline here is the one spec 028 builds for the Video page; whichever spec lands first
  builds it, the other reuses it.
- "Private network address" means loopback, the RFC 1918 ranges and link-local; a host name is
  classified by what it resolves to.

## Out of Scope

- Changing what models are supported, or adding cloud providers.
- Rendering answers into a video file (clarified 2026-10-17: not wanted).
- Human review or correction of answers (shelved, spec 023).
- Running across many videos from this window beyond handing off to run setup (spec 029).
- Prompt optimisation or automatic prompt suggestions.
