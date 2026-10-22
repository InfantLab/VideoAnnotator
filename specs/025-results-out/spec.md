# Feature Specification: Results Out

**Feature Branch**: `025-results-out` (work lands on `1.6-dev`)
**Created**: 2026-10-16
**Status**: Shipped in v1.6.0 (2026-10-18); built 2026-10-16. Opening the ELAN files in ELAN itself by a person is still to do ([quickstart](quickstart.md))
**Input**: User description: "Results Out (roadmap v1.6.0 Phase 3). An outside lab must get
VideoAnnotator's results into R or Python without writing a parser per pipeline. Four parts: (1)
Tidy export: one row per event (video, pipeline, track or person, start, end, label, value,
confidence, model) as CSV and Parquet, per job and per run (batch) or dataset, from the API, the
CLI and a viewer button; built from the output files and their provenance records (spec 017) so it
also works offline on any results folder (spec 022 layout). (2) ELAN export: .eaf with one tier per
pipeline track. (3) Methods paragraph: GET /api/v1/jobs/{id}/methods, a CLI command and a viewer
button; models, versions, weight revisions, parameters and citations as prose plus BibTeX. (4) Two
worked analysis notebooks, one R and one Python, reading the tidy export of the demo video. Older
results without provenance still export, marked 'not recorded'."

## Context

Each pipeline writes its own native format: COCO JSON for people, faces, scenes and VLM labels,
OpenFace's own JSON, WebVTT for speech, RTTM for speaker turns. That is right for provenance and for
tools that already read those formats. It is wrong for analysis. To get one table into R, a
researcher today has to write a parser per pipeline, learn that COCO keeps time in a non-standard
field, that RTTM gives a start and a duration, and that a WebVTT file's first block is now a
provenance note. A lab we have never met will not do that; they will stop.

v1.6.0's main goal is that **an outside lab can go from install to results in R or Python, on its
own videos, without help** ([roadmap](../../docs/development/roadmap_v1.6.0.md), Phase 3). Spec 022
put results where researchers expect them (one folder per run, one per video). Spec 017 made every
output file record what made it. This spec turns those files into things a researcher uses
directly:

1. **One table** of everything every pipeline found, the same shape for every pipeline.
2. **An ELAN file** per video, for the many labs that already hand-code in ELAN.
3. **A methods paragraph**, with citations, for the paper.
4. **Two worked notebooks** (R and Python) that start from that table.

Nothing about how pipelines run, or about their native files, changes. Everything here reads what
is already on disk.

### What the researcher sees

On a finished job's page and a run's page, beside today's "Download results", an **Export** menu:

```
[ Download results ]  [ Export v ]
                        Table (CSV)
                        Table (Parquet)
                        ELAN files (.eaf)
                        Methods paragraph
```

"Methods paragraph" opens a panel with the prose and the BibTeX, each with a Copy button. A saved
dataset's page offers the table for every run made from it.

From a terminal, with or without a server running:

```
videoannotator export table "~/VideoAnnotator/BabyJokes wave 2 (2026-10-06)"
  → BabyJokes wave 2 (2026-10-06)/BabyJokes wave 2 (2026-10-06).csv  (1,284 rows, 12 videos)
videoannotator export elan  "~/VideoAnnotator/BabyJokes wave 2 (2026-10-06)"
videoannotator export methods "~/VideoAnnotator/BabyJokes wave 2 (2026-10-06)"
```

A few rows of the table for the demo clip:

| video | pipeline | event | track | start | end | measure | label | value | confidence | model |
|---|---|---|---|---|---|---|---|---|---|---|
| demo_clip | speech_recognition | 1 | | 0.000 | 1.320 | speech | Ready, baby girl? | | | whisper base |
| demo_clip | speaker_diarization | 1 | SPEAKER_00 | 0.638 | 1.431 | speaker_turn | SPEAKER_00 | | | pyannote/speaker-diarization-3.1 |
| demo_clip | person_tracking | 7 | person_demo_001 | 2.000 | | person | infant | | 0.91 | yolo11n-pose |
| demo_clip | face_analysis | 3 | 0 | 1.500 | | emotion | happy | | 0.82 | not recorded |
| demo_clip | face_analysis | 3 | 0 | 1.500 | | age | | 31 | | not recorded |

## User Scenarios & Testing *(mandatory)*

### User Story 1 - One table of everything, in R or Python (Priority: P1)

A developmental psychologist has run five pipelines on 40 videos. They want one table in R: who is
speaking when, when the infant is in view, what the faces' emotions are, all on one time axis, one
row per thing found. They click "Export → Table (CSV)" on the run's page (or run one command on the
run's folder), and read the file with a single `read.csv`. Every pipeline's events have the same
columns; nothing needs a per-pipeline parser.

**Why this priority**: it is the release's main goal. The other three parts are conveniences on top
of results a researcher can already analyse; without this they cannot analyse them at all.

**Independent Test**: export the demo clip's results (the committed fixtures) to CSV; load the file
in R and in Python with their standard CSV readers; every pipeline present in the folder appears,
and each row's times, labels and values match the native file it came from.

**Acceptance Scenarios**:

1. **Given** a finished run of several videos and pipelines, **When** the researcher exports the
   run's table, **Then** they get one file with one set of columns, one row per event per measure,
   covering every video and every pipeline in the run, with a `video` column telling them apart.
2. **Given** a single finished job, **When** they export its table, **Then** it holds that video's
   rows only, in the same columns as a run export.
3. **Given** a results folder copied from another machine, with no server running, **When** they
   run the export command on that folder (a video folder, a run folder, or a folder of runs),
   **Then** they get the same table the server would give for the same files.
4. **Given** results written before provenance was recorded (v1.5.0 or earlier), **When** they are
   exported, **Then** their rows are included, and `model` and `videoannotator_version` say
   `not recorded`; nothing is guessed.
5. **Given** a run in which one video failed, **When** the run is exported, **Then** the table has
   the other videos' rows, and the export says which video has no results and why.
6. **Given** a saved dataset that has been run three times, **When** the researcher exports the
   dataset's table, **Then** it holds every run's rows with a `run` column telling them apart.
7. **Given** the optional Parquet component is installed, **When** they choose Parquet, **Then** the
   file has the same rows and columns with numeric columns typed as numbers; **given** it is not
   installed, the option says how to add it rather than failing silently.

---

### User Story 2 - A methods paragraph for the paper (Priority: P2)

The researcher is writing up. They open "Export → Methods paragraph" on the run and get a
paragraph naming each pipeline, the model behind it with its exact weights, the settings that
change results, VideoAnnotator's version, and the papers to cite, plus a BibTeX block for those
papers. They paste both into their manuscript and reference manager.

**Why this priority**: reviewers ask for this, and getting it right by hand means reading every
provenance record. It is what spec 017's records were built for.

**Independent Test**: request the methods for the demo clip's job; the paragraph names every
pipeline in the job, each model and revision exactly as its provenance records them, and every
citation key used in the prose has an entry in the BibTeX.

**Acceptance Scenarios**:

1. **Given** a finished job, **When** the researcher asks for its methods (viewer, CLI or API),
   **Then** they get prose and BibTeX naming each pipeline that ran, its models with weight
   revisions, the settings that differ from defaults, VideoAnnotator's version and citation, and
   the deterministic-mode setting.
2. **Given** a run whose videos all used the same pipelines, models and settings, **When** they ask
   for the run's methods, **Then** the paragraph describes the run once ("40 videos were
   processed…"), not once per video.
3. **Given** a run whose videos differ (a rerun with a newer version, a changed setting), **When**
   they ask for its methods, **Then** the paragraph says what differed and for how many videos.
4. **Given** results with no provenance record, **When** they ask for methods, **Then** the
   paragraph still names the pipelines and says the model versions were not recorded, rather than
   naming today's defaults.
5. **Given** a VLM pipeline result, **When** they ask for methods, **Then** the paragraph names the
   model, its digest, and identifies the prompt by its recorded fingerprint, and the full prompt
   text is available alongside.

---

### User Story 3 - Open the results in ELAN (Priority: P3)

A lab hand-codes infant behaviour in ELAN. They want VideoAnnotator's speech, speaker turns,
scenes, people in view and VLM labels as tiers next to their own coding, to check them, correct
them, or use them as a first pass. They export ELAN files for the run, get one `.eaf` per video,
and open each in ELAN with its video already linked.

**Why this priority**: it closes the loop with the tool many target labs already use, but labs can
analyse without it.

**Independent Test**: export the demo clip to `.eaf`; the file validates against ELAN's published
schema, opens in the viewer's own ELAN reader with one tier per pipeline track, and every
annotation's start and end match the table export's rows.

**Acceptance Scenarios**:

1. **Given** a finished job, **When** the researcher exports ELAN, **Then** they get one `.eaf` file
   whose tiers are named by pipeline and track (for example `speaker_diarization SPEAKER_00`,
   `speech_recognition`, `person_tracking infant`), linked to the video file.
2. **Given** a run, **When** they export ELAN, **Then** they get one `.eaf` per video, together.
3. **Given** frame-by-frame detections (people, faces), **When** they become ELAN tiers, **Then**
   consecutive detections of the same track become one "in view" annotation, so a track seen for
   ten seconds is one annotation, not 300.
4. **Given** continuous measures (action-unit intensities, gaze angles, emotion scores), **When**
   exporting ELAN, **Then** they are left out of ELAN (they are in the table), and the export says
   so.
5. **Given** the exported `.eaf` opened in ELAN by a researcher, **When** they add their own tiers
   and save, **Then** the file still opens in the viewer, which shows both their tiers and
   VideoAnnotator's.

---

### User Story 4 - Worked notebooks that start from the table (Priority: P4)

A researcher who has never used VideoAnnotator opens the R notebook (or the Python one) from the
repository. It reads the demo clip's table export, which ships beside it, and walks through a small
analysis a developmental researcher would recognise: how long each speaker talks, when the infant
is in view, emotion over time, and whether the infant is in view while the caregiver speaks. They
change the file name to their own export and the same code runs.

**Why this priority**: it shows the table is enough, and gives labs a starting point; it depends on
User Story 1 and adds no new capability.

**Independent Test**: run each notebook from start to finish on a clean machine with only its
listed packages installed; it finishes without errors and produces its figures and summary tables.

**Acceptance Scenarios**:

1. **Given** a clean R installation with the packages the notebook lists, **When** the researcher
   runs the R notebook, **Then** it completes and shows each analysis step's output.
2. **Given** the same for Python, **When** they run the Python notebook, **Then** it completes.
3. **Given** the table format changes in a way that breaks a notebook, **When** the project's checks
   run, **Then** the Python notebook's failure is caught before release.

### Edge Cases

- **A video folder with an unrecognised file** (a researcher's own notes, a future pipeline's
  output): skipped, and listed in the export summary as skipped, never an error.
- **A malformed or truncated output file**: that file is skipped with its reason; the rest exports.
- **No result files at all** (job failed, still running, or an empty folder): the export says there
  is nothing to export and why, rather than producing an empty file silently.
- **A job still running**: its export covers the pipelines that have finished, and says which have
  not.
- **Two runs of the same video in a dataset export**: both appear, told apart by `run`; nothing is
  deduplicated.
- **Text with commas, quotes, line breaks or non-Latin scripts** (transcripts, VLM reasoning):
  quoted so that standard CSV readers in R and Python read each row back intact; files are UTF-8.
- **Pipelines that record time differently** (a timestamp, a frame number, a start and a duration):
  all become seconds from the start of the video.
- **Vectors** (pose keypoints, facial landmarks, face embeddings): not in the table; they stay in the
  native files, which the table's `file` column points to.
- **A video folder name differing from the video's own name** (`site_a__child01`, `clip 2`): the
  `video` column uses the folder's name, which is unique within a run.
- **A pipeline output from an older version with different field names**: read if the reader knows
  that version; otherwise skipped and listed.
- **A large run** (hundreds of videos, millions of face rows): the export completes without the
  server becoming unresponsive, and the viewer shows it is being prepared.
- **A video moved after the run**: the ELAN file still links to the path recorded in the run; ELAN
  asks for the new location, as it does for any moved media.
- **A citation missing from a pipeline's metadata**: the methods paragraph still names the model and
  says it has no citation recorded; it never invents one.

## Requirements *(mandatory)*

### Functional Requirements

**Table export**

- **FR-001**: The system MUST turn a set of VideoAnnotator output files into one table with a fixed,
  documented set of columns, the same for every pipeline: `run`, `video`, `pipeline`, `event`,
  `track`, `start`, `end`, `frame`, `measure`, `label`, `value`, `confidence`, `x`, `y`, `width`,
  `height`, `model`, `videoannotator_version`, `file`.
- **FR-002**: Each row MUST be one measure of one event: an event (a detection, a speech segment, a
  speaker turn, a scene, a VLM label) has one row for what it is and one more row per attribute the
  pipeline measured (emotion, age, an action-unit intensity, gaze). Rows of one event share `video`,
  `pipeline` and `event`, so they can be pivoted into one row per event with standard tools.
- **FR-003**: `start` and `end` MUST be in seconds from the start of the video. Events that last
  (speech, speaker turns, scenes) MUST have both; observations at one moment (a frame's detection)
  MUST have `start` and an empty `end`, and `frame` when the pipeline recorded it.
- **FR-004**: `label` MUST hold categorical or text results and `value` numeric ones; `confidence`
  MUST hold the pipeline's own score for that row where it gives one, and be empty otherwise.
- **FR-005**: `model` and `videoannotator_version` MUST come from the file's provenance record; when
  a file has none, both MUST read `not recorded`.
- **FR-006**: The table MUST cover every pipeline that ships with VideoAnnotator: person tracking,
  face analysis, OpenFace 3, scene detection, speaker diarization, speech recognition and VLM
  annotation. Each pipeline's mapping into the columns MUST be documented.
- **FR-007**: Vector outputs (keypoints, landmarks, embeddings) MUST NOT be expanded into the table;
  `file` MUST name the native file each row came from, so they can be found.
- **FR-008**: The table MUST be exportable for one job, for one run, and for a saved dataset (every
  run made from it), from the API and the viewer.
- **FR-009**: The table MUST be exportable from the command line from any folder holding results (a
  video's folder, a run's folder, a folder of runs, or an extracted results download), without a
  server running and without network access.
- **FR-010**: For the same files, the command line, the API and the viewer MUST produce the same
  rows.
- **FR-011**: The table MUST be available as CSV (UTF-8, with a header row, readable by R's and
  Python's standard CSV readers without options) in every install.
- **FR-012**: The table MUST be available as Parquet when the optional component for it is
  installed. When it is not, every place that offers Parquet MUST say how to add it.
- **FR-013**: Every export MUST report what it covered: videos and pipelines included, files
  skipped and why, and videos with no results and why. The command line prints it; the API returns
  it alongside the download; the viewer shows it.
- **FR-014**: The table's format MUST carry a version. Later changes MUST only add columns or
  measures, so code written against this version keeps working; anything else needs a new version.
- **FR-015**: Row order MUST be stable for the same input (by video, then start time, then pipeline,
  event and measure), so two exports of unchanged results are identical apart from the export time.

**ELAN export**

- **FR-016**: The system MUST produce one ELAN `.eaf` file per video, valid against ELAN's published
  EAF schema, for one job or for every video in a run.
- **FR-017**: Each file MUST have one tier per pipeline track: one per speaker for diarization, one
  for speech text, one for scenes, one per person or face track, and one per VLM label set. Tier
  names MUST start with the pipeline's name.
- **FR-018**: Frame-level detections MUST become "in view" annotations covering runs of consecutive
  detections of the same track; a gap longer than a documented tolerance starts a new annotation.
- **FR-019**: Point-in-time labels (VLM labels at sampled frames) MUST become annotations lasting
  until the next sample, or until the end of the video for the last one.
- **FR-020**: Continuous measures MUST NOT be exported to ELAN; the export summary MUST say they
  were left out and where they are.
- **FR-021**: Each file MUST link to its video at the path recorded for the run, so ELAN opens the
  video with it.
- **FR-022**: The viewer's own ELAN reader MUST open every exported file and show its tiers, and
  MUST still open it after a researcher has added tiers to it in ELAN.

**Methods paragraph**

- **FR-023**: The system MUST produce a methods description for one job and for one run, from the
  API, the command line (on a results folder, with or without a server running) and the viewer,
  as prose plus BibTeX.
- **FR-024**: The prose MUST name, for each pipeline that ran: what it does in plain terms, the
  models with their weight revisions as recorded, the settings that change results where they
  differ from the pipeline's defaults, and its citations; plus VideoAnnotator's version and
  citation, and whether deterministic mode was on.
- **FR-025**: For a run, results that share pipelines, models and settings MUST be described once,
  with the number of videos; differences between videos MUST be stated with how many videos each
  applies to.
- **FR-026**: For VLM results, the prose MUST name the model and its recorded digest and identify
  the prompt by its recorded fingerprint; the full prompt text MUST be provided alongside.
- **FR-027**: Citations MUST come from each pipeline's own metadata, so a pipeline's author keeps
  them with the pipeline; every citation key used in the prose MUST have an entry in the BibTeX.
- **FR-028**: Where provenance or a citation is missing, the text MUST say so ("model version not
  recorded", "no citation recorded") and MUST NOT fill in current defaults.
- **FR-029**: The text MUST contain no secrets, no machine-specific paths and no video names (which
  often carry participant codes); it counts videos instead.

**Viewer**

- **FR-030**: A finished job's page and a run's page MUST offer an Export menu with Table (CSV),
  Table (Parquet), ELAN files and Methods paragraph; a saved dataset's page MUST offer the table.
- **FR-031**: The Methods panel MUST show the prose and the BibTeX, each with a one-click copy, and
  a download of both.
- **FR-032**: While an export is prepared the viewer MUST show that it is working; when it finishes
  it MUST show the export summary (FR-013) where anything was skipped.

**Notebooks**

- **FR-033**: The repository MUST include an R notebook and a Python notebook that read the demo
  clip's table export, shipped beside them, and show at least: speaking time per speaker, when each
  person is in view, emotion over time, and the overlap of the infant being in view with speech.
- **FR-034**: Each notebook MUST list the packages it needs and run unchanged on a researcher's own
  export by changing one file path.
- **FR-035**: The Python notebook MUST be run by the project's automated checks; the R notebook
  MUST be run before each release.

**Compatibility**

- **FR-036**: Native output files, their names and their locations MUST NOT change.
- **FR-037**: Results from before spec 022's results folders (job folders) and from before spec 017's
  provenance MUST export, with the gaps marked as above.

### Key Entities

- **Results set**: what an export reads; one video's results, a run's, a dataset's runs, or any
  folder of them. Identified by run and video.
- **Table row**: one measure of one event, in the fixed columns of FR-001.
- **Event**: one thing a pipeline found at one time or over one interval, with a track when the
  pipeline follows the same person, face or speaker over time.
- **Export summary**: what an export covered and what it skipped, and why.
- **ELAN tier**: one pipeline track's events over one video, as annotations with start and end.
- **Methods description**: prose and BibTeX describing how a job's or run's results were made,
  built from provenance records and pipeline metadata.
- **Citation**: a reference a pipeline's metadata asks users to cite, as a BibTeX entry with a key.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A researcher can get a finished run's results into R or Python as one table with one
  click (or one command) and one standard read call, with no per-pipeline code.
- **SC-002**: For the demo clip, 100% of the events in the native files appear in the table export
  with matching times (to the millisecond), labels and values; a check compares them on every
  change.
- **SC-003**: Exporting a run of 100 videos with every pipeline completes in under one minute on a
  laptop, and the viewer stays responsive meanwhile.
- **SC-004**: Every exported `.eaf` validates against ELAN's schema and opens in the viewer; a
  researcher opening it in ELAN sees the video and every tier with no error.
- **SC-005**: Every model revision in the methods text matches its provenance record exactly, and
  every citation key in the prose resolves in the BibTeX.
- **SC-006**: Both notebooks run start to finish on the shipped demo export without errors.
- **SC-007**: Results produced by v1.5.0 export to the table and to ELAN without errors.

## Assumptions

- The table uses a long layout (one row per measure) rather than one wide row per event, because
  pipelines measure different things: a wide table would have dozens of mostly empty columns and
  change shape whenever a pipeline is added. Both R (`tidyr::pivot_wider`) and Python
  (`pandas.pivot`) turn it wide in one call, and the notebooks show how.
- Parquet needs a library that would add tens of megabytes to the slim install, so it is an optional
  component, consistent with the project's extras-based install; CSV is the format the notebooks and
  most labs use.
- The person-tracking summary file (`person_tracks.json`) restates labels already on each
  detection, and OpenFace's detailed file restates its analysis file; the table reads one source per
  fact, so nothing is counted twice.
- Pipeline metadata gains citations for each shipped pipeline (the method and the model papers);
  VideoAnnotator's own citation comes from the repository's citation file.
- ELAN's gap tolerance for merging frame detections defaults to twice the pipeline's sampling
  interval, so one missed frame does not split an "in view" annotation.
- Exports are produced on request and are not written into run folders automatically; doing so is
  left for later, once researchers say whether they want it.
- Access follows the existing API rules: anyone who can see a job can export it.

## Out of Scope

- Changing any pipeline's native output format.
- A corpus overview across videos (roadmap Phase 6), which will read this table.
- Importing edited ELAN files back into VideoAnnotator as corrections.
- Exporting vectors (keypoints, landmarks, embeddings) to the table or to ELAN.
- Other annotation tools' formats (Datavyu, BORIS, Praat TextGrid); the table is their common route.
