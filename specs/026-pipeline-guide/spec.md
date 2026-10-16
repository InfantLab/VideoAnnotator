# Feature Specification: Pipeline Guide

**Feature Branch**: `026-pipeline-guide` (work lands on `1.6-dev`)
**Created**: 2026-10-16
**Status**: Draft
**Design**: [spec 023](../023-viewer-overhaul/spec.md) (viewer overhaul), journey 2 in
[journeys.md](../023-viewer-overhaul/journeys.md)
**Input**: User description: "Explain each pipeline to a researcher: what it does in behavioural
terms, where it comes from, what it's capable of (its external benchmarks), its homepage and
citation. Sensible defaults always; the researcher shouldn't dig into settings, especially the
technical ones. Each pipeline shows behavioural settings, then technical settings, then the
defaults, not one long list of every parameter."

## Context

A researcher who codes video by hand (journey 2) meets a list of pipelines and, for each, a list
of every parameter it takes. `face_analysis` shows twelve settings in one list:
`detect_emotions` (a choice about what is measured) sits next to `scale_factor` and
`min_neighbors` (which matter only to an OpenCV detector the pipeline no longer uses by default).
Nothing says what the pipeline is for in a researcher's terms, how well it does it, or where to
read more. Citations exist since spec 025, but only the methods paragraph shows them.

This spec gives every pipeline a short guide, written once in its metadata so the viewer, the
API and the CLI all say the same thing:

1. **What it does**, in one or two behavioural sentences ("Finds faces in each sampled frame and
   estimates each face's expression, age and gender").
2. **Behavioural settings**: the few choices about *what* is measured, with defaults.
3. **Technical settings**: how it's computed, collapsed, left at their defaults.
4. **About**: where it comes from (homepage), how well it does (published benchmarks, if any),
   and what to cite.

### What the researcher sees

On the Pipelines page, and wherever pipelines are chosen for a run:

```
┌ Faces: expression, age, gender ─────────────────────────────── [Installed] ┐
│ Finds faces in each sampled frame and estimates each face's expression,    │
│ age and gender.                                                            │
│                                                                            │
│ What to measure                                                            │
│   ☑ Expression   ☑ Age   ☑ Gender                                          │
│   Most faces per frame  [10]                                               │
│                                                                            │
│ ▸ Technical settings (5, all at defaults)                                  │
│                                                                            │
│ About                                                                      │
│   DeepFace with RetinaFace detection · homepage ↗                          │
│   Benchmark: RetinaFace, WIDER FACE hard set, AP 91.4% ↗                   │
│   Cite: Serengil & Ozpinar 2026; Deng et al. 2020   [Copy BibTeX]          │
└────────────────────────────────────────────────────────────────────────────┘
```

(Figures illustrative; the real ones come from the published sources, FR-007.)

```
$ videoannotator pipelines describe face_analysis
Faces: expression, age, gender
Finds faces in each sampled frame and estimates each face's expression, age and gender.

What to measure
  detect_emotions   Expression           true
  detect_age        Age                  true
  ...
Technical (5)
  ...
About
  ...
```

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Know what a pipeline does before running it (Priority: P1)

A linguist wants speech utterances labelled. On the Pipelines page they see, for each pipeline, a
plain title and one or two sentences saying what it finds, in terms of behaviour, not models.
They pick speech recognition without having read any documentation.

**Why this priority**: Choosing the right pipeline is the first decision in journey 2; without
this, the researcher guesses from model names.

**Independent Test**: Show the Pipelines page to someone who hasn't used VideoAnnotator; they can
say what each pipeline will produce.

**Acceptance Scenarios**:

1. **Given** the Pipelines page, **When** it loads, **Then** every listed pipeline shows a plain
   title and a behavioural summary of at most two sentences.
2. **Given** the run setup's pipeline choice, **When** pipelines are listed, **Then** the same
   title and summary appear there.
3. **Given** the CLI, **When** `videoannotator pipelines describe <name>` runs, **Then** it
   prints the same title, summary and settings groups.

---

### User Story 2 - Defaults, and only the settings that matter (Priority: P1)

The researcher runs with defaults and never opens a settings panel. When they do want a change
(no age estimates; at most two faces), the choices are short, labelled in plain words and say
what they change. Technical settings are folded away with a count and "all at defaults".

**Why this priority**: The design's rule is sensible defaults and no digging; one long list of
parameters is what journey 2 called out.

**Independent Test**: For each pipeline, the behavioural group is visible without expanding
anything, the technical group is collapsed, and every setting has a plain label.

**Acceptance Scenarios**:

1. **Given** any pipeline, **When** its settings are shown, **Then** behavioural settings appear
   first with plain labels and defaults; technical settings are collapsed under a heading that
   says how many there are and whether any differ from defaults.
2. **Given** a technical setting changed from its default, **When** the group is collapsed,
   **Then** the heading says so ("5, 1 changed").
3. **Given** a setting that only applies when another setting has a given value (an
   OpenCV-only threshold when the detector isn't OpenCV), **When** settings are shown, **Then**
   it is hidden, and shown when it applies.

---

### User Story 3 - Judge and report the method (Priority: P2)

Before trusting a pipeline in a paper, the researcher wants to know where it comes from and how
well it does. The About section links to its homepage, gives any published benchmark with a link
to the source, and offers the citation to copy.

**Why this priority**: Researchers must justify and cite their methods; spec 025's methods
paragraph covers the paper, this covers the choice.

**Independent Test**: For each pipeline, About shows a homepage link, either at least one
benchmark with its source or a plain "no published benchmark", and its citations.

**Acceptance Scenarios**:

1. **Given** a pipeline with a published benchmark, **When** About is shown, **Then** it gives
   the benchmark's name, the measure and figure, and a link to the source.
2. **Given** a pipeline whose quality depends on a model the user chooses (the VLM), **When**
   About is shown, **Then** it says so and links to the chosen model's page rather than giving a
   figure.
3. **Given** a pipeline with no published benchmark, **When** About is shown, **Then** it says
   "No published benchmark" rather than leaving a gap.
4. **Given** any pipeline, **When** "Copy BibTeX" is used, **Then** the same entries as the
   methods paragraph are copied.

---

### Edge Cases

- A metadata file without the new fields (a third-party pipeline, an older file) still loads; its
  settings all show as technical, its summary falls back to its existing description, and About
  shows only what it has.
- A setting classified behavioural in one pipeline and technical in another (a confidence
  threshold) is allowed; the classification is per pipeline.
- A pipeline that isn't installed still shows its guide, so researchers can decide whether to
  install it.
- A deprecated pipeline (spec 014) keeps its deprecation notice above the guide.
- A setting that depends on another (FR-005) and has been changed from its default stays visible,
  marked as not applying, so a non-default value is never hidden.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Each pipeline's metadata MUST carry a plain title and a behavioural summary of at
  most two sentences, written for researchers, not naming models unless the model is the point.
- **FR-002**: Each setting in a pipeline's metadata MUST carry a group (behavioural or
  technical) and a plain label; the existing description stays as the longer explanation.
- **FR-003**: Each behavioural setting MUST state, where it applies, its unit (seconds, faces,
  pixels) and allowed range or values.
- **FR-004**: Every built-in pipeline's settings MUST be classified; a setting about what is
  measured or reported (which labels, which attributes, how often, how many people) is
  behavioural, and one about how it is computed (backends, model internals, retries, timeouts,
  cache) is technical.
- **FR-005**: A setting MUST be able to declare that it applies only when another setting has
  given values; it is then hidden when it doesn't apply, unless it differs from its default.
- **FR-006**: Each pipeline's metadata MUST carry a homepage link (the method's or model's own
  page or repository).
- **FR-007**: Each pipeline's metadata MUST carry zero or more benchmarks, each with a name, the
  measure and figure as published, and a link to the source; figures MUST be quoted from the
  source, not computed or rounded beyond it. When quality depends on a user-chosen model, the
  metadata says so instead.
- **FR-008**: The API's pipeline information MUST include the title, summary, groups, labels,
  units, dependencies, homepage and benchmarks; existing fields MUST NOT change.
- **FR-009**: The viewer's Pipelines page and the run setup's pipeline choice MUST show the title
  and summary for every pipeline, and the settings as behavioural (open), technical (collapsed,
  with count and changed count) and About.
- **FR-010**: The CLI MUST offer `videoannotator pipelines describe <name>` printing the same
  guide; `pipelines list` MUST show the plain title beside the name.
- **FR-011**: Metadata missing any of the new fields MUST still load (Edge Cases); a check MUST
  fail for any built-in pipeline missing a summary, homepage, classification of every setting, or
  benchmark entry (an explicit empty list counts as present).
- **FR-012**: Settings that the pipeline's code no longer reads MUST be found and either removed
  from the metadata (when no saved config uses them) or marked technical and dependent (FR-005).

### Key Entities

- **Pipeline guide**: title, behavioural summary, homepage, benchmarks, citations (existing).
- **Setting**: existing type, default, description, enum; new group, label, unit, range, and
  "applies when" condition.
- **Benchmark**: name, measure, figure as published, source link, optional note (dataset split,
  model size).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every built-in pipeline has a summary, homepage, benchmark entry (possibly empty)
  and every setting classified and labelled; the metadata check passes.
- **SC-002**: For every built-in pipeline, at most five behavioural settings are shown before
  anything is expanded.
- **SC-003**: A researcher new to the tool, shown the Pipelines page, correctly says what each
  pipeline produces for at least 6 of the 7 listed pipelines.
- **SC-004**: The viewer, the API and the CLI give the same title, summary and grouping for every
  pipeline (checked by a test against the metadata).
- **SC-005**: No setting is shown that the pipeline ignores under the current configuration,
  except one changed from its default.

## Assumptions

- The guide lives in the existing per-pipeline metadata files (spec 004's registry), added to,
  not restructured (Principle V).
- "Behavioural" and "technical" are the only two groups; a third ("advanced") would recreate the
  long list.
- Benchmarks are the published ones from each method's own paper or a standard leaderboard;
  VideoAnnotator does not run its own evaluations in this spec.
- The VLM's quality depends on the model chosen; its guide points to the model's page. Its
  settings (prompt, sampling) are classified here and presented by spec 027.
- Plain titles follow spec 023 Decision 1: what the pipeline finds ("Faces: expression, age,
  gender"), not the model ("DeepFace").

## Out of Scope

- Changing any pipeline's defaults or outputs.
- Running benchmarks on VideoAnnotator's own data.
- The redesigned run setup and Pipelines page layout beyond showing the guide (spec 029).
- Help elsewhere in the viewer (specs 027–029 each carry their own).
