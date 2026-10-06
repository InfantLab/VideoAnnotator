# Feature Specification: Videos Read Where They Are

**Feature Branch**: `022-videos-read-in-place` (work lands on `1.6-dev`)
**Created**: 2026-10-06
**Status**: Draft
**Input**: User description: "Videos are read where they are, by default, for a single researcher
running VideoAnnotator on their own laptop (the primary first-user case for the v1.6.0 public
release)." Full description in the conversation that created this spec; summarised in Context.

## Context

The first users of the v1.6.0 public release will mostly be single researchers running everything
on their own laptop: the server and the viewer on the same machine, the videos on its disk.

Today the job wizard's default way to choose videos is "Upload from this computer". Every video is
sent through the browser and copied into the server's job folder, even though the browser and the
server are on the same machine. The copy is pointless and expensive:

- **100 videos is 100 uploads**, one after another, with the tab kept open throughout.
- **Disk use doubles**, and grows again with every new run.
- **Every copy is another copy of sensitive infant and family video**, which is a data-governance
  liability for the researcher's ethics approval, not just a disk-space problem.

A no-copy route already exists. "Folder on the server" (spec 008) starts one job per video in a
folder, in one step, and reads each video where it is. Saved datasets of such folders run again
with no copies (spec 018). But researchers rarely use it:

- **It's the second tab**, and upload is the default.
- **It's restricted** to an administrator on the server's own machine, and to allowed folders (the
  server user's home folder by default).
- **It only takes whole folders**, not a handful of chosen videos.
- **Under Docker it can't work at all.** The container's allowed folder is not where the videos are
  mounted, and the browser's requests reach the container from Docker's network, not from "this
  machine", so they are refused.

This spec makes reading videos in place the default for that local researcher. Upload remains for
servers on another machine, and for videos the server isn't allowed to read.

## What the researcher sees

The new-job wizard's first step, "Choose Videos", today has three tabs: "Upload from this
computer", "Folder on the server" and "Saved dataset". On a laptop, "this computer" and "the
server" are the same machine, so the first two read as the same thing. Their real difference, that
upload copies every video and the folder tab doesn't, is not shown anywhere.

After this spec, on a laptop install, "Choose Videos" has two tabs:

- **My folders** (the default): the researcher's own folders. They open a folder, see its videos
  (name, length, size), and tick the ones to run, or "Select all", optionally including subfolders.
  A line under the list says the selected videos are used where they are and never copied.
- **Saved datasets**: as today.

Below the list, a small link reads "Videos on another computer? Upload them". It is for the case
where VideoAnnotator runs on a different machine from the videos. It is the only route that copies
videos, and it says so.

When VideoAnnotator runs on a different machine from the researcher's browser (a lab server),
"My folders" is not offered, because that machine's folders are not the researcher's. Upload is
the main route there.

```
Choose Videos
[ My folders ]  [ Saved datasets ]

Home > Studies > BabyJokes                 [Up]
  [ ] Include subfolders
  [x] 4JDccE.joke5.rep3.take1...mp4   0:03  0.5 MB
  [ ] 6c6MZQ.joke1.rep1.take1...mp4   0:17  2.3 MB
  [x] 6c6MZQ.joke1.rep2.take1...mp4   0:09  1.3 MB
  [Select all]
2 videos selected - used where they are, never copied

Videos on another computer? Upload them >
```

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Choose videos on my own laptop without copying them (Priority: P1)

A researcher has VideoAnnotator running on their laptop and a folder of study videos on the same
disk. They start a new job. Step 1, "Choose Videos", opens on "My folders" and shows their folders.
They pick the study folder, choose pipelines, and submit. The run starts at once: nothing is
uploaded, no copy of any video is made, and they can close the tab while it runs.

**Why this priority**: This is the first thing every first user does, and today's default is the
worst option for them. It is the whole point of the feature, and it delivers value on its own.

**Independent Test**: On a local install with a folder of videos in the user's home folder, start a
new job with no settings changed. Confirm "My folders" is the tab shown, submit, and check
that no video file was created anywhere under the server's storage folder.

**Acceptance Scenarios**:

1. **Given** the viewer is connected to a server on the same machine, **When** the researcher opens
   the new-job wizard, **Then** "Choose Videos" opens on "My folders", and upload is offered only as
   the "Videos on another computer? Upload them" link.
2. **Given** the researcher picked a folder of 100 videos in My folders, **When** they submit,
   **Then** all 100 jobs exist within seconds, as one run, without any upload.
3. **Given** a run started from videos in My folders, **When** the jobs run and finish, **Then**
   no copy of any video exists in the server's storage, and the results are the same as for an
   uploaded copy of the same video.
4. **Given** the run has started, **When** the researcher closes the browser tab, **Then** the run
   continues to completion.

---

### User Story 2 - Pick a few videos out of a large folder (Priority: P1)

A researcher's folder holds 200 sessions; today they want to run 5 of them. They open the folder in
the wizard, tick the 5 videos, and submit. Only those 5 run, read in place.

**Why this priority**: Whole-folder-only selection forces researchers back to upload (with its
copies) whenever they want a subset, which is the common case while piloting a pipeline. Without
this, Story 1 only covers part of real use.

**Independent Test**: In a folder of 20 videos, select 3 and submit. Confirm exactly 3 jobs are
created, each reading its video in place.

**Acceptance Scenarios**:

1. **Given** a folder of videos in My folders, **When** the researcher opens it, **Then** they see
   its videos (name, size, and duration where known) and can select any of them, or all.
2. **Given** a selection of videos from one folder, **When** they submit, **Then** exactly those
   videos run, as one run, with no copies.
3. **Given** a folder with subfolders, **When** the researcher chooses to include subfolders, **Then**
   videos from the subfolders are listed and selectable too.

---

### User Story 3 - Same experience under Docker (Priority: P2)

A researcher runs VideoAnnotator with the documented Docker setup on their laptop. They point the
setup at their video folder once, as the instructions say. From then on, the wizard behaves exactly
as in Story 1: their videos are listed and read in place, with no copies.

**Why this priority**: Docker is a documented install route. Today in-place reading is broken
there, so Docker users are pushed into copying everything. It ranks below Stories 1 and 2 because
the plain install is the primary route for first users.

**Independent Test**: Start the documented Docker setup with a video folder configured, open the
viewer from the host's browser, and run Story 1's test.

**Acceptance Scenarios**:

1. **Given** the documented Docker setup with the researcher's video folder configured, **When** they
   open the new-job wizard from a browser on the same machine, **Then** "My folders" shows that folder, and Story 1's
   scenarios hold.
2. **Given** the Docker setup, **When** the server is reached from another machine on the network,
   **Then** in-place reading is not offered to that caller (see FR-008).
3. **Given** the Docker setup with no video folder configured, **When** the researcher opens the
   wizard, **Then** they are told how to configure one, and upload works meanwhile.

---

### User Story 4 - A video moved or deleted after its run (Priority: P2)

A researcher ran a folder last month, then reorganised their disk. They rerun the dataset, or open a
job's results. Results already produced are intact. Anything that needs a video that has moved says
which video is missing and what to do, instead of failing with an obscure error.

**Why this priority**: Reading in place means VideoAnnotator no longer controls the videos, so
missing videos become normal. They must be handled clearly, consistently with the stored-copy
dataset runs added in commit `6c19c06`.

**Independent Test**: Run a folder of 3 videos, move one, then (a) open each job's results and (b)
rerun the run and the dataset.

**Acceptance Scenarios**:

1. **Given** a finished job whose video was moved, **When** the researcher opens its results,
   **Then** every annotation and file it produced is shown. Only video playback says the video is no
   longer at its original location, naming that location.
2. **Given** a run or dataset where some videos have moved, **When** the researcher runs it again,
   **Then** they are told which videos are missing, before anything starts. They can run the rest,
   or choose where the videos are now.
3. **Given** a queued job whose video is deleted before it starts, **When** it comes to run, **Then**
   it fails with a message naming the missing video, and the rest of the run continues.

---

### User Story 5 - A saved dataset of videos from My folders reruns with no prompts (Priority: P3)

A researcher saved the videos they chose (a folder, or a selection) as a dataset. Next week they
choose "Use" on that dataset. It goes straight to choosing pipelines: no folder to find, no upload.

**Why this priority**: This extends spec 018 to the in-place selections of Story 2. Datasets of
whole server folders already behave this way.

**Independent Test**: Save a 3-video selection as a dataset, start a new job from it, and confirm
no prompt appears and no copies are made.

**Acceptance Scenarios**:

1. **Given** a dataset saved from a selection of videos from My folders, **When** the researcher
   uses it, **Then** the wizard proceeds with exactly those videos, read in place, without asking
   for a location.
2. **Given** such a dataset where a video has since moved, **When** it is used, **Then** Story 4's
   scenario 2 applies.

---

### Edge Cases

- **Video outside the allowed folders** (e.g. on an external drive, when only the home folder is
  allowed): the wizard can't list it. It says how to allow that location, and offers upload for it
  meanwhile.
- **Server on another machine**: in-place reading is not offered. The wizard defaults to upload, and
  explains that in-place reading needs the videos on the server's machine or on storage it can see.
- **Same video picked twice** (e.g. a folder plus a file inside it): it runs once.
- **Non-video files in a folder**: ignored, as today's folder ingest does. Empty or unreadable files
  are reported, not run.
- **A file that changes while its job runs**: out of scope to detect. Results reflect what was read.
- **Removable drive unplugged between submission and the job starting**: handled as a missing video
  (Story 4, scenario 3).
- **The researcher deletes a job**: the job's results are deleted; the original video is never
  touched. This is existing behaviour for in-place jobs, and it must be stated in the interface.
- **Very large folders** (thousands of files): listing stays usable. The folder opens without
  waiting for every video's duration to be read.

## Requirements *(mandatory)*

### Functional Requirements

**Default and source choice**

- **FR-001**: When the viewer is connected to a server on the same machine, the new-job wizard MUST
  open "Choose Videos" on "My folders": the researcher's own folders, from which videos are read in
  place.
- **FR-002**: Upload MUST remain available on every install. On a same-machine connection it MUST be
  offered as a secondary link ("Videos on another computer? Upload them"), not a tab. When the
  server is not on the same machine, "My folders" MUST NOT be offered, and upload MUST be the main
  route.
- **FR-003**: The viewer MUST be able to learn from the server whether the current connection counts
  as the same machine (FR-008), so it can choose the default and avoid offering what would be
  refused.
- **FR-004**: The interface MUST say, where the choice is made, that videos from My folders are
  read where they are and never copied, and that deleting a job never deletes the original video.

**Choosing videos in place**

- **FR-005**: Researchers MUST be able to browse the folders the server may read, and select a
  whole folder (optionally including subfolders), or any subset of the videos in it.
- **FR-006**: A run started from an in-place selection MUST create exactly one job per distinct
  selected video, as a single run, in one step, with no copy of any video made by VideoAnnotator.
- **FR-007**: Jobs reading videos in place MUST produce the same results as jobs on an uploaded
  copy of the same video, with the same pipelines and settings.

**Who may read in place**

- **FR-008**: In-place reading MUST be allowed only for callers on the same machine as the server.
  The Docker setup MUST establish "same machine" for the researcher's own browser, and MUST NOT
  extend it to other machines on the network.
- **FR-009**: On a single-user install, the researcher MUST be able to read in place with no extra
  setup beyond the standard install. The first user, who is an administrator by default, needs no
  further permission.
- **FR-010**: In-place reading MUST stay limited to allowed folders. The default MUST cover where a
  local researcher keeps their videos (the user's home folder on a plain install; the configured
  video folder under Docker). The allowed folders MUST be configurable, and the interface MUST
  point to how.

**Docker**

- **FR-011**: The documented Docker setup MUST include a single, clearly named place to configure
  the researcher's video folder, mounted read-only. That folder MUST be in-place-readable by default.
- **FR-012**: With no video folder configured under Docker, the wizard MUST explain how to configure
  one, and keep upload working.

**Missing videos**

- **FR-013**: When a job's video is missing at the moment it is needed, every message MUST name the
  video and the location it was expected at. This covers starting the job, playback, a rerun, and
  using a dataset.
- **FR-014**: A job's existing results MUST remain fully viewable when its video is missing; only
  playback is unavailable.
- **FR-015**: Before a rerun or dataset use starts any job, the researcher MUST be told which videos
  are missing. They MUST be able to proceed with the rest, or point to the videos' new location.
- **FR-016**: A queued job whose video disappears MUST fail with that message without affecting
  the other jobs in its run.

**Datasets**

- **FR-017**: A dataset saved from an in-place selection (a folder or a subset) MUST, when used,
  proceed with exactly those videos in place, with no location prompt while they are where they
  were.

**Compatibility**

- **FR-018**: Upload submission, folder ingest, and saved datasets of every existing kind MUST keep
  working as before for existing clients and saved data.

### Key Entities

- **In-place video reference**: a job's link to a video at its original location on the server's
  machine, as opposed to a stored copy. Records the location, the name, and the size when chosen,
  so a moved video can be recognised as missing.
- **Allowed folder**: a location the server may read videos from in place. It has a default, and
  can be configured.
- **In-place selection**: what the researcher chose. Either a folder (with or without subfolders) or
  a set of videos within the allowed folders. It can be saved as a dataset.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a local install, a researcher starts a run on 100 videos in under 1 minute from
  opening the wizard, with no step that waits on transferring video data.
- **SC-002**: On a local install with default settings, starting runs through the wizard creates
  zero additional copies of researchers' videos. The server's storage holds only results.
- **SC-003**: A first-time user on a plain local install or the documented Docker setup completes
  their first run from videos on their own disk without changing any setting other than, under
  Docker, the documented video folder.
- **SC-004**: Picking 5 videos out of a 200-video folder takes no more interactions than picking
  those 5 files in an ordinary file dialog would.
- **SC-005**: Every missing-video situation in User Story 4 produces a message that names the
  video, as confirmed in walkthrough testing. No missing-video situation produces a generic or
  internal error.
- **SC-006**: Existing upload-based workflows, server-folder runs and saved datasets pass their
  existing tests unchanged.

## Assumptions

- **Most first users run server and viewer on one machine.** Remote and lab servers are served by
  upload until a later spec adds a lab mode.
- **Picking in place means browsing inside the viewer, not the operating system's file dialog.** A
  browser can't tell a web page where a picked file is on disk, so an ordinary file dialog can only
  ever upload. The in-viewer browser shows only allowed folders.
- **"Same machine" is decided by the server**, from how the request reaches it, plus explicit setup
  under Docker. It is not something the browser can claim on its own.
- **Read-only access is enough.** VideoAnnotator never writes to, moves, or deletes a researcher's
  videos. Results always go to the server's own storage.
- **Duration and other video details are best effort** when listing. A folder must open quickly
  even when there are many videos.

## Out of Scope

- Storing uploads once by their content (deduplication), resumable or chunked uploads, and a
  retention policy for stored copies.
- A multi-user lab mode where authenticated remote users may choose server-side videos.
- Remote and HPC dispatch (v1.8+).
- Detecting that a video's content changed (rather than moved) since a run.
