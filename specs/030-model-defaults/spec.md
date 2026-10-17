# Feature Specification: Refresh Model Defaults

**Feature Branch**: `030-model-defaults` (work lands on `1.6-dev`)
**Created**: 2026-10-17
**Status**: Draft
**Input**: User description: "Refresh Model Defaults (roadmap v1.6.0 Phase 5c). Each change on the
upstream model's published evaluation plus a demo-clip check against the baseline: no model change
without a reason on record. Face: replace DeepFace's default detector if still needed, age and
gender off by default, the emotion model chosen with Uwerikowe et al.'s comparison on
caregiver–child video. Speech: openai-whisper → faster-whisper with large-v3-turbo. Diarization:
pyannote speaker-diarization-3.1 → speaker-diarization-community-1, with a Sortformer spike
alongside. Person: yolo11n-pose → yolo26n-pose. OpenFace 3: confirm the maintained distribution.
VTC 2: licence question. Done before the public release so its outputs are stable."

Research behind every figure here: [research.md](research.md).

## Context

v1.6.0 Phase 1 upgraded the *libraries* (torch 2.11, pyannote.audio 4, Python 3.13) and left every
*default model* as it was, on purpose: outputs on the demo clip stayed the same, and the CHANGELOG
says so. Several of those defaults are now a generation behind what their own upstreams publish
and recommend. v1.7.0 is the public release; once outside labs have results, changing a default
breaks their comparisons between videos coded before and after. So the defaults change now, once,
before anyone outside depends on them.

The roadmap's rule for this is one of its core principles: **a default changes only on the
upstream model's published evaluation, after its output on the demo clip has been checked against
the baseline, and the CHANGELOG and the pipeline's own description say why.** The constitution adds
two constraints: the output formats don't change (Principle II), and the model that produced each
file is recorded in it (Principle III). Principle V's "pin the previous model by configuration" is
**waived for this change** (maintainer, 2026-10-17): no outside lab depends on v1.5/v1.6 outputs
yet, and models will be updated again soon, so effort goes to good defaults rather than to keeping
old ones runnable. See [Constitution note](#constitution-note).

What changes, per pipeline (details and sources in research.md):

| Pipeline | Today | Proposed | Upstream evidence |
|---|---|---|---|
| Speech | openai-whisper, `base` | faster-whisper, `large-v3-turbo` | runtime 2.3–2.7x faster on its own benchmark (large-v2 GPU 2m23s → 1m03s); turbo: 809M params, "minor quality degradation" vs large-v3, Open ASR mean WER 7.83 |
| Speaker turns | pyannote `speaker-diarization-3.1` | pyannote `speaker-diarization-community-1` | lower DER on 10 of 12 published sets (e.g. AMI IHM 18.8 → 17.0) |
| People and pose | `yolo11n-pose` | `yolo26n-pose` | COCO-pose mAP50-95 50.0 → 57.2 |
| Faces | DeepFace, RetinaFace detector; emotion, age, gender on | age and gender off; DeepFace's own expression model kept (Q1) | age/gender say nothing about infants (pipeline review) |
| OpenFace 3 | `openface-test==0.1.13` | unchanged; its status recorded | the official package, dormant since 2025-06 |

The face detector the roadmap asked to replace (the Haar cascade) was already replaced by
RetinaFace earlier in v1.6.0; this spec records the reason in the metadata and adds nothing else.

### What the researcher sees

Nothing new on screen. What they notice:

- Transcripts are better on a first run, which downloads a larger speech model (about 1.6 GB
  instead of 140 MB) and says so before it starts.
- The speaker-turn pipeline needs one model licence accepted on Hugging Face instead of three.
- Face results no longer carry age and gender unless they switch them on, and the first face run
  downloads about 1 GB less.
- The pipeline guide (spec 026) and the methods paragraph (spec 025) name the new models, with
  the published figure each was chosen on.
- The release notes have a short section per pipeline:

```
speech_recognition: default model whisper base → large-v3-turbo, runtime faster-whisper.
  Why: ... (source). On the demo clip: <n> of <m> cues changed wording; timings within <t> s.
  To keep the old behaviour: {"speech_recognition": {"whisper_model": "base", ...}}
```

(The placeholders are filled from the demo-clip check.)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Better defaults, with the reason on record (Priority: P1)

A researcher installs v1.6.0 and runs speech, speaker turns and person tracking on their videos
with no settings. They get the current recommended models for each. When they write up, the
methods paragraph names each model and its exact weights, and the pipeline guide shows the
published figure each default was chosen on, with a link. A reviewer who asks "why this model?"
gets an answer from the docs, not from the maintainer.

**Why this priority**: it is what Phase 5c is for, and the public release's results should come
from the defaults it will keep.

**Independent Test**: run each changed pipeline on the demo clip with no settings; its output's
provenance names the new model and revision; the pipeline's metadata and the CHANGELOG each give a
reason with a source link for the change; the guide's benchmark entry shows the quoted figure.

**Acceptance Scenarios**:

1. **Given** a fresh install with the `audio` extra, **When** the researcher runs speech
   recognition with no settings, **Then** it uses faster-whisper with `large-v3-turbo`, and the
   output's provenance names that model's repository and revision.
2. **Given** the same, **When** they run speaker diarization with no settings, **Then** it uses
   `speaker-diarization-community-1`, and the setup check asks them to accept only that model's
   licence.
3. **Given** the `person` extra, **When** they run person tracking with no settings, **Then** it
   uses `yolo26n-pose`, and the output stays COCO with 17 keypoints per person.
4. **Given** any changed pipeline, **When** they open its guide, **Then** they see the benchmark,
   measure and figure the default was chosen on, quoted from its source, with a link.
5. **Given** the release notes, **When** they read a changed pipeline's entry, **Then** it gives the
   old and new model, the reason with its source, what changed on the demo clip, and the setting
   that restores the old model.

---

### User Story 2 - Old model where it is only a setting (Priority: P3)

A lab that wants the previous diarization or pose model for a comparison sets it by name in the
pipeline's settings and gets it, recorded as such in provenance. Nothing more is promised: the old
models are not tested against a baseline, and the old speech runtime is removed (Q3).

**Why this priority**: backward compatibility was waived for this change (Context); this story
records only what stays possible for free.

**Independent Test**: a config naming `pyannote/speaker-diarization-3.1` or `yolo11n-pose` runs and
records that model in provenance.

**Acceptance Scenarios**:

1. **Given** a config naming `pyannote/speaker-diarization-3.1`, **When** the job runs, **Then** it
   uses 3.1 and records it.
2. **Given** a config naming `yolo11n-pose`, **When** the job runs, **Then** it uses that model and
   records it.
3. **Given** a config written for v1.5.0 that names none of these settings, **When** it runs on
   v1.6.0, **Then** it runs without error with the new defaults.

---

### User Story 3 - Faces: only what says something about infants (Priority: P2)

A developmental researcher runs face analysis on parent–infant play. By default they get faces and
expressions, not an estimated age and gender that mean nothing for a nine-month-old. The expression
model stays DeepFace's own (Q1), and the guide says plainly it was trained on adult faces.

**Why this priority**: it removes misleading defaults and 1 GB of download, but face analysis is
one pipeline of several and is itself due for replacement.

**Independent Test**: run face analysis on the demo clip with no settings; no age or gender
appears in the output and the age and gender models are neither downloaded nor run; with them
switched on, both appear as before.

**Acceptance Scenarios**:

1. **Given** default settings, **When** face analysis runs, **Then** its output has detections and
   expressions and no age or gender, and the age and gender weights are not downloaded.
2. **Given** `detect_age: true`, **When** it runs, **Then** age is written and only the age model is
   added.
3. **Given** the demo clip, **When** face analysis runs, **Then** its output is checked against a
   baseline too (face analysis has been left out of the baseline check because the old detector
   found no faces in the demo clip).

---

### User Story 4 - Diarization choice made on evidence, not inertia (Priority: P3)

Before the diarization default is settled, the maintainer has a half-day spike's answers on NVIDIA
Streaming Sortformer: does it install on a newer torch without torchaudio, what does NeMo send over
the network and can it be switched off, how big is the install, and what does it produce on the demo
clip. The answers are written down beside the pyannote decision, so whether Sortformer replaces
pyannote later is a decision with a record.

**Why this priority**: it informs a future change (pyannote is what holds torch at 2.11), not this
release's defaults.

**Independent Test**: the spike's write-up answers each of the four questions with what was
observed, and states a recommendation.

**Acceptance Scenarios**:

1. **Given** the spike is done, **When** the maintainer reads the write-up, **Then** each of the
   four questions has an observed answer or "could not be determined" with the reason.
2. **Given** the write-up, **When** the diarization default is decided, **Then** the decision and
   its reason are recorded in the dependency audit's §8.

### Edge Cases

- **CPU-only machine**: the larger speech model is much slower without a GPU. The first run says
  which model it is using and that a smaller one can be chosen; the guide gives the setting.
- **`task: translate` with the new speech default**: `large-v3-turbo` is not trained to translate
  and returns the original language. A translate request must not silently return untranslated
  text: it uses a model that translates, or fails with a message naming one.
- **No GPU libraries the new speech runtime can find**: it falls back to CPU and says so, rather
  than failing the job.
- **Offline machine with the old models already downloaded**: a job with no settings needs the new
  weights and fails with the existing "model not downloaded" message naming them; a job naming the
  old diarization or pose model runs offline as before; old speech settings get the new runtime.
- **A user who accepted only the 3.1 licences**: the new default fails setup with the existing
  readiness message naming community-1's licence page.
- **A saved preset or rerun from before the change** (spec 019 "run it again"): it reruns with the
  settings it recorded; where it recorded none for the model, it gets the new default, and the rerun
  summary says the default changed.
- **Old results beside new ones in one run or export**: the table export (spec 025) and methods
  paragraph show the different models per video, as they already do for any difference.
- **A changed default whose demo-clip check shows a regression** (e.g. a missed speaker or a lost
  person): the switch does not ship until the regression is explained in the CHANGELOG or the
  default is kept.
- **Model weights from a third party** (the faster-whisper turbo conversion is published by neither
  OpenAI nor SYSTRAN): provenance names the exact repository and revision used.

## Requirements *(mandatory)*

### Functional Requirements

**Rules for every default change**

- **FR-001**: A pipeline's default model MUST change only when (a) the upstream model's published
  evaluation supports it, quoted with benchmark name, measure, figures and source link, and (b) the
  new default has been run on the demo clip through the server and its output compared with the
  current baseline (`tests/integration/test_output_baseline.py`), with the differences described.
- **FR-002**: Library upgrades that a model change needs (ultralytics for YOLO26, the faster-whisper
  runtime) MUST be checked against the baseline on their own, with the old model, before the model
  changes, so each difference is attributed to its cause.
- **FR-003**: Each change's reason MUST be recorded in the CHANGELOG (old model, new model, the
  evaluation with source, the demo-clip differences, and the setting that restores the old model)
  and in the pipeline's metadata (a reason and source link beside the default).
- **FR-004**: Each changed pipeline's benchmark entries for the pipeline guide (spec 026) MUST name
  the new default's published evaluation, with figures quoted exactly from the source. If spec 026
  has not landed when this spec is implemented, the entries are written in the form spec 026 defines
  and land with it.
- **FR-005**: After each intended change, the demo-clip baseline and viewer contract fixtures MUST
  be regenerated on purpose (as `tests/fixtures/viewer_contract/README.md` describes), the previous
  set kept where the contract test needs it, and both the baseline test and the viewer contract test
  MUST pass.
- **FR-006**: Output formats MUST NOT change: WebVTT for speech, RTTM for speaker turns, COCO for
  people and faces, with no key renamed or removed from the schema.
- **FR-007**: Every output from a new default MUST record the model's identifier and exact revision
  in its provenance (spec 017), so the methods paragraph (spec 025) names it without change to the
  exporter.
- **FR-008**: Where the previous model is only a setting value (diarization, pose), the release
  notes MUST name it; no test or baseline is kept for it. The previous speech runtime is not kept
  (Q3; Constitution note).
- **FR-009**: Each changed pipeline's metadata MUST list the new default's weights and download size,
  and the first run MUST state the download before it starts (as spec 011's readiness already does).
- **FR-010**: Each new default's licence MUST be recorded in the pipeline metadata, and no new
  default may be research-only or non-commercial (constitution, Engineering Standards).
- **FR-011**: No new default may send data off the machine; any telemetry a new dependency sends by
  default MUST be off (Principle I), with the existing no-telemetry test covering it.
- **FR-012**: Every changed pipeline's citations (spec 025) MUST cite the new model; a licence that
  asks for attribution (CC-BY-4.0) MUST have its citation present.

**Speech**

- **FR-013**: Speech recognition MUST run on the faster-whisper runtime by default, with
  `large-v3-turbo` as the default model.
- **FR-014**: Every existing speech setting (model size, language, task, beam size, best-of,
  temperature, patience, length penalty, word timestamps, punctuation, suppressed tokens, minimum
  segment duration) MUST keep its meaning under the new runtime, or be documented as having no
  effect.
- **FR-015**: A translate request MUST produce a translation or a clear error, never untranslated
  text labelled as a translation.
- **FR-016**: The `audio` extra MUST install on every supported platform and Python version without
  building from source.

**Speaker turns**

- **FR-017**: Speaker diarization MUST default to `pyannote/speaker-diarization-community-1` in
  v1.6.0 (Q2).
- **FR-018**: The setup check MUST ask only for community-1's licence for the default.
- **FR-019**: The existing speaker-count settings MUST keep working with the new default.

**People and pose**

- **FR-020**: Person tracking MUST default to `yolo26n-pose`, keeping COCO-17 keypoints and the
  current tracking behaviour; settings the new model ignores MUST be documented as such.
- **FR-021**: The pipeline's name, description and variant MUST stop naming YOLO11 as its model.

**Faces**

- **FR-022**: Age and gender MUST be off by default; when off, their models MUST NOT be downloaded or
  run.
- **FR-023**: The default expression model MUST stay DeepFace's own (Q1), and the pipeline guide
  MUST say it was trained on adult faces.
- **FR-024**: Face analysis MUST be included in the demo-clip baseline check if it detects faces in
  the demo clip with the RetinaFace detector.
- **FR-025**: The RetinaFace detector's reason (already in the CHANGELOG) MUST be recorded in the
  metadata and the guide, with its published evaluation.

**OpenFace 3 and VTC 2**

- **FR-026**: OpenFace 3's metadata MUST record that `openface-test` is the distribution CMU names,
  its last upstream change (2025-06), and its non-commercial licence; its version pin stays.
- **FR-027**: The VTC 2 licence question MUST be recorded as open in the roadmap, with the date it
  was asked, until LAAC-LSCP answers; this spec adds no VTC pipeline.

**Sortformer spike**

- **FR-028**: Not a v1.6.0 gate (Q2 decided without it). When done (v1.7.0), a time-boxed
  (half-day) spike records:
  whether Sortformer installs and runs on a torch newer than 2.11 without torchaudio; what NeMo
  sends over the network by default and whether it can be switched off; the install size; and its
  output on the demo clip next to community-1's. The findings go into the dependency audit §8.

### Key Entities

- **Default model**: the model a pipeline uses when its settings don't name one; identifier,
  revision, licence, download size.
- **Reason on record**: why a default changed: old and new model, upstream evaluation (benchmark,
  measure, figures, source), demo-clip differences, restore setting. Kept in the CHANGELOG and the
  pipeline metadata.
- **Benchmark entry** (spec 026): name, measure, figure as published, source link, note.
- **Demo-clip baseline**: the committed outputs of the demo clip that the baseline test compares
  against; regenerated on purpose after each change.
- **Provenance record** (spec 017): which model and revision made a file.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every default model change in v1.6.0 has a CHANGELOG entry with an upstream source
  link and the demo-clip differences (roadmap success criterion "Every change of default model has
  its reason in the CHANGELOG").
- **SC-002**: After the change, the demo-clip baseline test passes against the regenerated
  baseline, and the viewer contract test passes.
- **SC-003**: Withdrawn 2026-10-17 (backward compatibility waived; Constitution note). A config
  naming the previous diarization or pose model still runs and records that model (User Story 2).
- **SC-004**: 100% of outputs written by the new defaults name their model and revision in their
  provenance; the methods paragraph for a demo-clip job names every new model.
- **SC-005**: Every changed pipeline's guide shows at least one published benchmark figure that
  matches its source exactly.
- **SC-006**: A first face-analysis run with default settings downloads at least 1 GB less than in
  v1.6.0rc1.
- **SC-007**: A new user setting up speaker turns accepts one model licence instead of three.
- **SC-008**: The `audio` extra installs on Linux, macOS and Windows, Python 3.12 and 3.13, with no
  compiler on the machine.

## Assumptions

- The demo clip is the only check for each switch. The roadmap dropped a VideoAnnotator benchmark
  (2026-10-14): no human-coded infant video of our own, and the open sets can't be redistributed.
  The upstream evaluations are on adult speech and adult faces; the guide says so.
- Old models stay reachable only where they are a setting value (diarization, pose, faces'
  attribute flags); the old speech runtime goes (Q3).
- The faster-whisper runtime is introduced in two steps (runtime with `base`, then the model), each
  checked on the demo clip, so the CHANGELOG can say which caused what.
- The new speech default is the same on GPU and CPU; CPU users get a smaller model by one setting,
  documented in the guide and the first-run message, not by automatic switching (an automatic
  switch would make outputs depend on hardware, against Principle III).
- Ultralytics stays for v1.6.0 (AGPL notice as today); replacing it with RTMPose is later work.
- Sortformer and VTC 2 are not v1.6.0 defaults whatever the spike or the licence answer says: each
  would add a new library stack or pipeline, which is v1.7.0+ work.
- Speaker diarization's newer "exclusive" output (one speaker at a time) is not adopted; the RTTM
  keeps overlapping turns as today.
- Spec 026 defines the benchmark field; if this spec lands first, its entries wait in the form 026
  defines.

## Out of Scope

- New pipelines (VTC 2, RTMPose, iCatcher+, Infant AFAR) and replacing `face_analysis` with a
  non-DeepFace pipeline (Q1: later work).
- Replacing pyannote with Sortformer; lifting the torch cap.
- Keeping openai-whisper, or any old default, installed or tested (Q3; Constitution note).
- Running VideoAnnotator's own benchmark on infant data.
- Changing scene detection's or the VLM's default model.
- Changing output formats or the table export's columns.

## Clarifications (answered 2026-10-17)

- **Q1 (Faces)**: keep DeepFace's own expression model, stated as adult-trained. Replacing
  `face_analysis` is later work.
- **Q2 (Diarization)**: switch to community-1 in v1.6.0; the Sortformer spike moves to v1.7.0 and
  doesn't gate this.
- **Q3 (Speech)**: faster-whisper only; openai-whisper is removed. No backward compatibility is
  needed: "we're updating models soon, so it needn't be perfect now" (maintainer).

## Constitution note

Principle V says users MUST be able to pin a previous checkpoint via config. Removing openai-whisper
(Q3) means speech can't reproduce v1.5.0 transcripts exactly. Justification for the exception: no
outside lab has results yet (v1.6.0 is unannounced, the public release is v1.7.0), and models will
change again before the public release. The plan's Complexity Tracking records it, and the release
notes say plainly that v1.6.0 speech output differs from v1.5.0's and that v1.6.0rc1 is the last
version with openai-whisper.
