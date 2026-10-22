# Feature Specification: Test Strategy for v1.7.0: Journeys and Release Criteria

**Feature Branch**: `032-test-strategy` (work lands on `1.7-dev`)
**Created**: 2026-10-19
**Status**: Draft, not yet planned
**Input**: User description: "v1.7.0 test strategy: journeys and release criteria. Replaces the
remaining boxes of roadmap v1.7.0 Phase 7 (test audit)." Wanted: journey tests in a real browser
against the real server with fakes in place of models, the v1.7.0 success criteria as tests, the
viewer/server seam kept honest, fuzzing of the API from its schema, periodic mutation testing, a
coverage gate raised towards the constitution's 80%, and the audit's remaining chores.

## Context

The test audit ([`docs/development/test_audit_v1.7.0.md`](../../docs/development/test_audit_v1.7.0.md),
2026-10-19) found a suite that is broad but shallow, and organised by code module rather than by
how v1.7.0 is used:

- **Broad**: 99 of the server's 104 routes are requested by some test, and line coverage is 76%.
- **Shallow**: changing the code in small ways (flipping a comparison, emptying a return) went
  unnoticed by every test about 40% of the time, even in modules with 89% and 96% line coverage.
  Two of the changes that went unnoticed matter: a model's weights recorded with no hash (the
  constitution's provenance principle), and Ask AI's table counting skipped samples and mislabelling
  model errors. Three real bugs were hidden by tests that couldn't fail.
- **Never together**: every viewer test replaces the server with a hand-written stand-in; most of the
  viewer's description of the server's answers is hand-written; nothing checks that the generated
  description is current. The only browser test checks the page title and doesn't run.
- **The release's own promises are untested**: none of v1.7.0's success criteria (nothing leaves the
  machine unless a hosted backend is on; the viewer never says job, batch, dataset or preset; a
  first Ask AI answer in under a minute; everything the viewer does is also in the CLI) has a test.

v1.7.0 is the public release. Its users work through the viewer, in journeys that cross the viewer,
the server and the files on disk. This spec adds that layer and makes the existing one prove what it
claims. The pattern exists in one place: `tests/integration/test_ask_run_stop.py` runs a real
server with a fake model server and a synthetic video.

### Who this is for

The maintainer and contributors, who need to know before a merge that the researcher's path still
works; and, through them, the researchers and pilot labs who will run the release on their own
videos.

### Order of work

Journeys need the pages they walk through. Spec 027 (Ask AI) is being built now; 028 (video page)
and 029 (videos and projects) come after. A journey for a page that doesn't exist yet is written
when that spec lands, as part of it. Everything else here can start at once.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - The tutorial's path works, checked on every change (Priority: P1)

A contributor changes the viewer or the server and opens a pull request. Before it can merge, an
automated run opens the viewer in a real browser against a real server and walks the researcher's
path from the tutorial: add a video, ask a question of it, run that question on the whole video,
review the answers on the video, export them, and run again with a change. Models are replaced by
fakes that answer instantly and predictably, so nothing is downloaded and the run is quick. If any
step breaks, the run fails and says which step, with a screenshot.

**Why this priority**: it is the layer that's missing, and the one that protects what a
researcher does. Every other story strengthens a layer that already exists.

**Independent Test**: break one step on purpose (for example, rename the button that starts a run)
and confirm the journey run fails at that step and names it; restore it and confirm it passes.

**Acceptance Scenarios**:

1. **Given** a fresh server with no videos, **When** the journey adds the synthetic test video,
   **Then** the video appears where the viewer lists videos and can be played.
2. **Given** a video, **When** the journey asks a question in Ask AI, **Then** an answer from the
   fake model appears for each sampled moment.
3. **Given** a question that worked, **When** the journey runs it on the whole video, **Then**
   answers appear as they arrive, and stopping keeps the answers so far.
4. **Given** a finished run, **When** the journey opens the video, **Then** the answers are shown on
   its timeline at the moments they describe.
5. **Given** a finished run, **When** the journey exports it, **Then** the downloaded file opens and
   holds one row per answer, with the run's provenance.
6. **Given** a finished run, **When** the journey runs it again with one setting changed, **Then**
   both runs are kept and can be told apart.
7. **Given** any journey failure, **When** a contributor reads the result, **Then** it names the
   journey and step and includes a screenshot and the server's log.

---

### User Story 2 - The release's promises are checked, not hoped for (Priority: P1)

The maintainer preparing a release candidate wants evidence that v1.7.0 keeps its promises. Each
success criterion that can be checked by a machine has a check that runs with the journeys:

- **Local only**: while the journeys run with only local backends configured, nothing tries to
  reach any address other than the local machine. A journey that switches on a hosted backend sees
  the warning and confirmation first, and only then does a request leave (to the fake hosted
  server).
- **The researcher's words**: no page the journeys visit shows the words job, batch, dataset or
  preset to the user.
- **First answer quickly**: from adding a video to the first Ask AI answer takes under a minute
  against the fake model, so the time measured is VideoAnnotator's own.
- **Everything from the CLI too**: each thing the viewer does has a CLI command with JSON output, or
  is listed as a known gap.

**Why this priority**: these are the release's definition of done. Without checks, each is
re-verified by hand before every release candidate, or not at all.

**Independent Test**: add a request to an outside address in a local-only path, or the word "job"
to a page, and confirm the matching check fails and names the address or the page.

**Acceptance Scenarios**:

1. **Given** only local backends, **When** the journeys run, **Then** no connection to a
   non-local address is attempted, and any attempt fails the run with the address and the step.
2. **Given** a hosted backend, **When** a journey uses it, **Then** the request is made only after
   the confirmation, and the check records that it was confirmed.
3. **Given** every page the journeys visit, **When** the text shown to the user is scanned, **Then**
   none of the four words appears, except in places listed as allowed (for example, a file name the
   user chose).
4. **Given** the fake model, **When** the first-answer journey runs, **Then** it reports the time
   taken and fails if it is a minute or more.
5. **Given** the list of things the viewer does, **When** the parity check runs, **Then** each one
   maps to a CLI command that produces JSON, or appears in a reviewed list of known gaps.

---

### User Story 3 - The viewer and the server agree on every answer (Priority: P2)

A contributor changes what a server endpoint returns. Today the viewer's tests keep passing, because
they use hand-written stand-ins for the server, and the mismatch shows up only when a researcher
opens the page. After this story, the change fails a check at once: either the viewer's description
of the server is stale, or a viewer test's stand-in no longer matches what the server says it
returns.

**Why this priority**: the seam is where v1.7.0's large viewer rewrite meets a moving server.
The journeys catch some of this, but late and slowly; this catches it early and names the field.

**Independent Test**: rename a field in one endpoint's answer and confirm that a check fails and
names the field, before any journey runs.

**Acceptance Scenarios**:

1. **Given** a server change that alters an endpoint's answer, **When** checks run, **Then** they
   fail until the viewer's generated description of the server is regenerated and committed.
2. **Given** the viewer's description of the server's answers, **When** this story is done,
   **Then** it comes from the server's own published description, not from hand-written copies.
3. **Given** a viewer test that stands in for the server, **When** its stand-in answer doesn't
   match the server's published description, **Then** the viewer's type check fails.

---

### User Story 4 - The server never crashes on what it's sent (Priority: P2)

Researchers, scripts and (from v1.7.0 Phase 5) agents will send the server things the viewer never
does. A check generates many requests from the server's own description of its endpoints, odd and
ordinary, and fails if any request makes the server crash or answer that something went wrong
inside it, rather than explaining what was wrong with the request.

**Why this priority**: cheap to add, covers all routes at once, and protects the agent and CLI
users who don't have the viewer's guard rails.

**Independent Test**: make one endpoint raise an unexpected error on an empty field and confirm the
check finds it and prints a request that reproduces it.

**Acceptance Scenarios**:

1. **Given** every endpoint in the server's description, **When** the check runs, **Then** no
   generated request produces an internal server error.
2. **Given** a failure, **When** a contributor reads the result, **Then** it includes a request
   they can replay to reproduce it.
3. **Given** endpoints that start real work (installing extras, running pipelines), **When** the
   check runs, **Then** they are exercised without starting that work, or are listed as excluded
   with a reason.

---

### User Story 5 - Know whether the tests would notice a wrong change (Priority: P3)

The maintainer wants to know, for the parts that matter most (provenance, the exporters, and the
path a job takes from submission to results), whether the tests would catch the code being wrong,
not just whether they ran it. A scheduled run makes small changes to that code, one at a time,
and reports each change no test noticed. The meaningful ones get a test; the harmless ones are
recorded as harmless, so the next report shows only what's new.

**Why this priority**: it turns "76% covered" into a measure of what the tests actually protect,
but it is a report to act on, not a gate, so it follows the stories that gate merges.

**Independent Test**: run the report on provenance today and confirm it lists the weights-hash
change found in the audit; add the missing test and confirm that change disappears from the next
report.

**Acceptance Scenarios**:

1. **Given** the chosen modules, **When** the scheduled run finishes, **Then** it publishes the
   share of changes caught per module and lists each change that went unnoticed, with its line.
2. **Given** a change marked harmless, **When** the next report runs, **Then** it is not listed
   again unless the line changes.
3. **Given** the two survivors named in the audit, **When** this story is done, **Then** each has a
   test that catches it.
4. **Given** the measured coverage, **When** this story is done, **Then** the merge gate fails below
   that figure (not 45%), and the plan to reach 80% names the modules it depends on.

---

### User Story 6 - The testing docs describe the suite as it is (Priority: P3)

A new contributor reads how to test VideoAnnotator. Today they find a README describing a suite of
"6 focused files" with a 94% pass rate, and an overview telling them to set a variable nothing
reads. After this story, both describe the layers that exist (unit, server, journeys, release
checks, reports), how to run each, and which run before a merge.

**Why this priority**: necessary for the pilot and for outside contributions (a v1.7.0 success
criterion), but it documents the other stories, so it comes last.

**Independent Test**: a contributor who hasn't seen the suite follows the docs from a fresh clone
and runs each layer without asking anyone.

**Acceptance Scenarios**:

1. **Given** the testing docs, **When** a contributor follows them, **Then** each command works as
   written and its stated run time is within a factor of two of the real one.
2. **Given** the duplicate retry-logic tests found in the audit, **When** this story is done,
   **Then** one file covers that behaviour, keeping the stronger assertions of each.

---

### Edge Cases

- A journey for a page that hasn't been built yet: it is not written, and the journey list shows it
  as waiting for its spec, so the gap is visible.
- A journey that fails only sometimes: it is fixed or quarantined with an issue number within a
  week; a quarantined journey is reported on every run, never silently skipped.
- The browser or its dependencies can't be installed (for example, on a contributor's Windows
  machine): the journeys can still run in CI and in the dev container, and the docs say so.
- The fake model is too fast to show answers arriving one by one: the fake can be told to wait
  between answers, so "as they come" is tested rather than assumed.
- An allowed use of a banned word: a user-chosen name such as a video file called "job.mp4" is
  shown as given; the scan's allow list covers user content, not the viewer's own text.
- A route the viewer uses only in development or for diagnostics: the parity check lists it as
  excluded with a reason, rather than demanding a CLI command.
- An endpoint whose request schema is wrong (so generated requests are rejected before reaching our
  code): reported as a finding about the schema, not counted as a pass.
- A mutation report that takes too long: it runs on a schedule and per module, so a slow module
  doesn't hold up others, and it never blocks a merge.
- The server and viewer versions differ (a contributor's stale build): the journeys build the viewer
  from the same commit as the server, so they never test a mismatched pair.

## Requirements *(mandatory)*

### Functional Requirements

**Journeys**

- **FR-001**: The journeys MUST drive the viewer in a real browser against a real VideoAnnotator
  server started for the run, with its own empty data folder.
- **FR-002**: The journeys MUST NOT download models or need network access: model-backed steps use
  fakes (a fake vision-language model server and a stub pipeline), and the video is generated.
- **FR-003**: The journey set MUST cover, once their pages exist: add a video; ask a question (Ask
  AI try); run it on the whole video, with Stop; review answers on the video; export; run again
  with a change.
- **FR-004**: Each journey MUST be independent: it sets up what it needs and doesn't depend on
  another journey having run.
- **FR-005**: A failure MUST report the journey, the step, a screenshot, and the server's log for
  that run.
- **FR-006**: The journeys MUST run on every pull request to `1.7-dev` and `master`, and a failure
  MUST block the merge.
- **FR-007**: The journeys MUST finish in under 10 minutes in CI.
- **FR-008**: Each new viewer spec (028, 029 and later) MUST add or update the journeys for the
  pages it builds, as part of its own tasks.

**Release checks**

- **FR-009**: While journeys run with only local backends, any connection to a non-local address
  MUST fail the run, naming the address and the step.
- **FR-010**: A journey MUST switch on a (fake) hosted backend and confirm that frames are sent
  only after the warning and confirmation that spec 027 requires.
- **FR-011**: The text shown on every page a journey visits MUST be checked for the words job,
  batch, dataset and preset (whole words, any case), with an allow list for user content only.
- **FR-012**: The first-answer journey MUST measure the time from adding a video to the first Ask AI
  answer, report it on every run, and fail at one minute or more.
- **FR-013**: A parity check MUST list each user action the viewer offers, map it to a CLI command
  with JSON output, and fail on any action that has neither a command nor an entry in a reviewed
  list of known gaps.

**Viewer and server agreement**

- **FR-014**: Checks MUST fail when the viewer's generated description of the server's endpoints
  differs from what the current server publishes.
- **FR-015**: The viewer's types for server answers MUST come from the generated description;
  hand-written copies MUST be replaced (types that only the viewer uses may stay).
- **FR-016**: Stand-in server answers in viewer tests MUST be checked against the generated
  description, so a stand-in that the server could not return fails the type check.

**Robustness**

- **FR-017**: A check MUST generate requests for every endpoint from the server's published
  description and fail on any internal server error, printing a request that reproduces it.
- **FR-018**: Endpoints that would start real work (installing extras, running real pipelines,
  contacting outside services) MUST run against fakes or be excluded with a written reason.
- **FR-019**: The robustness check MUST run on every pull request, within the journeys' time budget
  or alongside them.

**Measures**

- **FR-020**: A scheduled run (at least weekly) MUST mutation-test provenance, the exporters, and
  the job path (submission, processing, storage of results), and publish per-module results.
- **FR-021**: Changes judged harmless MUST be recordable so they are not re-reported while their
  line is unchanged.
- **FR-022**: The two meaningful survivors named in the audit MUST have tests that catch them.
- **FR-023**: The coverage gate MUST be raised from 45% to the coverage measured when this spec
  lands (rounded down to a whole number), and MUST NOT be lowered without a recorded reason.
- **FR-024**: The plan to reach the constitution's 80% MUST name the modules with the largest gaps
  (the job path and the CLI, per the audit) and the order they will be covered in.

**Docs and chores**

- **FR-025**: `tests/README.md` and `docs/testing/testing_overview.md` MUST describe the layers that
  exist, how to run each, their run times, and which ones gate a merge.
- **FR-026**: The duplicate retry-logic tests MUST be merged into one file, keeping the stronger
  assertions of each.
- **FR-027**: The test audit document MUST record the outcome of each story, and the roadmap's
  Phase 7 MUST point to this spec for its remaining work.

### Key Entities

- **Journey**: a named path through the viewer taken from the tutorial; its steps, the pages it
  visits, the fakes it uses, and the spec that owns its pages.
- **Release check**: a machine check of one v1.7.0 success criterion; runs with the journeys; its
  result is pass, fail (with the evidence), or not yet applicable (page not built).
- **Fake backend**: a stand-in for a model server or pipeline that answers instantly and
  predictably, can be told to wait or fail, and records what it was sent.
- **Parity entry**: one viewer action, the CLI command that does the same with JSON output, or the
  reason it is a known gap.
- **Mutation report**: per module, the share of small code changes the tests caught, the changes
  they missed, and the ones marked harmless.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every tutorial step whose page exists is covered by a journey that runs before every
  merge; at v1.7.0's first release candidate, that is all six.
- **SC-002**: A deliberately broken step in any journey is caught before merge in 10 out of 10
  trials, and the failure names the step.
- **SC-003**: The journeys and release checks together finish in under 10 minutes per run, and in
  the month before the first release candidate fewer than 1 run in 50 fails for reasons unrelated to
  the change being tested.
- **SC-004**: Each of v1.7.0's four machine-checkable success criteria (local only, the
  researcher's words, first answer in under a minute, CLI parity) has a check that runs before every
  merge, and none is verified by hand at release.
- **SC-005**: Renaming a field in any endpoint's answer fails a check before any journey runs.
- **SC-006**: No generated request produces an internal server error on any endpoint.
- **SC-007**: In the modules tested for it, the share of small code changes the tests catch rises
  from about 60% to at least 80%, with every remaining miss either covered or recorded as harmless.
- **SC-008**: The coverage gate equals the measured figure (not 45%), and coverage at v1.7.0's
  release is no lower than when this spec lands.
- **SC-009**: A contributor new to the project runs every test layer from a fresh clone by following
  the docs alone.

## Assumptions

- The journeys run in one browser engine (Chromium) in CI; other engines are run before a release
  candidate, not on every merge.
- The fakes already in the repo (the fake VLM server, the stub pipeline, the synthetic video) are
  the starting point; they are extended rather than replaced.
- Real-model checks stay as they are (`real_models`, run by hand before a release); this spec
  doesn't change them.
- "Local" means the machine running the server, including a container's host as spec 027 defines
  it; the egress check uses the same definition.
- The CI runner has enough time and memory to run a server and a browser; the journeys share one
  server per run but not data, so journeys don't affect each other.
- The mutation report runs on the existing Monday schedule in CI.

## Out of Scope

- Load and performance testing beyond the first-answer time.
- Visual regression testing (comparing screenshots pixel by pixel).
- Accessibility auditing (worth its own spec before release; not part of the test audit).
- Testing real models' accuracy: that is the pipeline benchmark and model cards' job.
- Testing on the pilot labs' own machines and videos: that is Phase 4, which these checks prepare
  for.
