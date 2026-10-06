# Feature Specification: A Container That Feels Local

**Feature Branch**: `024-container-feels-local` (work lands on `1.6-dev`)
**Created**: 2026-10-06
**Status**: Draft
**Sequence**: after [022](../022-videos-read-in-place/spec.md) (videos and results where you expect
them). Replaces 022's untried Docker walkthrough (its task T072). Independent of
[023](../023-viewer-overhaul/spec.md) (viewer overhaul).
**Input**: User description: "A container that feels local." Summarised in Context; the design
discussion of 2026-10-06 is recorded in Background.

## Context

Researchers run VideoAnnotator in a container, with Docker or Podman. The container is what keeps
torch and the pipelines working the same way on any machine. Any other route leads to endless
compatibility and support problems on researchers' own computers.

But containers bewilder non-technical users, and the worst of it is files. A container is a sealed
box. It sees only the folders someone has shared with it, and sharing them today means container
vocabulary: mounts, environment variables, container paths. Checking spec 022 showed exactly this.
The new-job wizard opened on "My folders", listing `/root` with `.bun`, `.cache` and `.config`:
the inside of the box, meaning nothing to a researcher, and none of their videos.

### The crux

**Access to a researcher's files can only be granted on their own computer, by them.** Only their
computer can see its disk, and only they can consent. The container can't grant itself access,
and the viewer runs on the container's side of that line.

Every bad experience in this area comes from asking people to make that grant in the container's
terms:

- Label Studio's file-serving environment variables, which even technical users find
  incomprehensible;
- CVAT's shared volume;
- this project's own `VIDEOS_DIR` and `/root` listing.

This spec makes the grant a moment on the researcher's own computer, in their terms: a folder
picker, and "VideoAnnotator will be able to read, but never change, this folder". It remembers the
answer, and teaches the viewer where the boundary is so it can point there.

### What the researcher experiences

The first time:

```
$ videoannotator-start          (or double-click "Start VideoAnnotator")

Starting VideoAnnotator.

Which folder are your videos in?      [a folder picker opens]

  VideoAnnotator will be able to read, but never change:
    /home/ada/Studies   (and everything inside it)
  Results go to:
    /home/ada/VideoAnnotator

  Share this folder?  [Yes]  [Choose another]

Starting... ready. Opening http://127.0.0.1:18011/viewer in your browser.
```

Every time after that, VideoAnnotator just starts, with the same shared folders, and says once
which folders it can read. In the viewer, "My folders" lists exactly those folders under the names
and paths the researcher knows, never the inside of the container.

## Background: the 2026-10-06 discussion

Recorded so the plan doesn't re-open settled points.

- **Containers are the route** for researchers' own computers. Native installs were considered
  and rejected: on researchers' machines they would mean endless compatibility and support issues.
- **Prior art borrowed:**
  - **fMRIPrep's `fmriprep-docker` wrapper** (neuroimaging): researchers give ordinary paths, and
    the wrapper builds the container command and shares the folders.
  - **Apptainer/Singularity** (HPC): folders appear at their real paths inside the container, and
    files written belong to the user.
  - **Phone and desktop app permissions** ("Allow access to Photos?"): a sandbox nobody sees, with
    access as a plain, revocable grant.
  - **Media servers' "library folders"** (Plex, Jellyfin): the app shows only folders you added,
    never the machine.
- **Cautionary tales:** Label Studio and CVAT, as above.
- **Hiding the container**: hide the mechanics (commands, mounts, container paths) and name the
  boundary honestly, as a permission. Hiding everything fails because the container shows through
  sooner or later, and a researcher never told about it is then more lost.
- **Least access**: share folders, not the computer; read-only; easy to see and revoke. Persistence
  and ease of access come from remembering grants, not from granting more.
- **Docker and Podman are both first class.** Podman is free for everyone, avoiding Docker
  Desktop's licensing at larger institutions. On Linux it needs no background service, and files
  it writes belong to the user.
- **Adding a folder from inside the viewer** needs something running on the researcher's own
  computer, because the container can't do it. That host-side helper is a later phase. This spec
  starts with the start-up program, built so it can grow into one.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Start VideoAnnotator and share my video folder, once (Priority: P1)

A researcher has Docker Desktop or Podman Desktop installed. They run VideoAnnotator's start-up
program for the first time. It finds the container engine, asks which folder their videos are in,
says plainly what VideoAnnotator will be able to read and where results will go, starts, and opens
the viewer in their browser. In the new-job wizard, My folders shows their video folder by its real
name and path, and they can run their videos.

**Why this priority**: This is the first thing every researcher does. Without it, the container
route fails at the first screen, as it did when 022 was checked.

**Independent Test**: On a computer with only a container engine installed, run the start-up
program, choose a folder of videos, and run one video from My folders without typing any container
command, path or setting.

**Acceptance Scenarios**:

1. **Given** a container engine is installed and running, **When** the researcher runs the start-up
   program for the first time, **Then** it asks which folder their videos are in, using a folder
   picker where the computer has one.
2. **Given** they chose a folder, **When** they are asked to confirm, **Then** the confirmation names
   that folder, says VideoAnnotator can read but never change it and everything inside it, and names
   where results will go.
3. **Given** they confirmed, **When** VideoAnnotator has started, **Then** the viewer opens in their
   browser, and My folders lists the shared folder with the path their own file browser shows.
4. **Given** VideoAnnotator is running, **When** a run finishes, **Then** its results are in the
   results folder on their computer, and the files belong to them (they can move or delete them
   without special permission).

---

### User Story 2 - Start again with no questions (Priority: P1)

The next day, the researcher starts VideoAnnotator again. It starts with the same shared folders
and results folder, asks nothing, and says once which folders it can read.

**Why this priority**: Ease comes from persistence. Asking every time would push people to share
their whole computer once to make the questions stop.

**Independent Test**: After Story 1, stop and start VideoAnnotator twice, including once after
updating it. No question is asked; the same folders are shared; earlier results and saved datasets
still open.

**Acceptance Scenarios**:

1. **Given** a folder was shared before, **When** VideoAnnotator starts again, **Then** it shares the
   same folders without asking, and says which folders it can read.
2. **Given** VideoAnnotator was updated to a new version, **When** it starts, **Then** the shared
   folders, results folder, jobs, results and saved datasets are all as before.
3. **Given** a shared folder is missing at start (moved, renamed, or on an unplugged drive),
   **When** VideoAnnotator starts, **Then** it starts without that folder and says which folder it
   couldn't find, rather than failing to start.

---

### User Story 3 - Share another folder, or stop sharing one (Priority: P2)

The researcher's new study is on an external drive. They share that folder too. Later they stop
sharing an old study folder.

**Why this priority**: Videos live in more than one place, and researchers must be able to see and
narrow what VideoAnnotator can read. It's P2 because one folder covers a first run.

**Independent Test**: Share a second folder through the start-up program, run a video from it, then
stop sharing the first folder and confirm My folders no longer lists it.

**Acceptance Scenarios**:

1. **Given** VideoAnnotator is running, **When** the researcher chooses to share another folder in
   the start-up program, **Then** after a restart that the program carries out and announces, My
   folders lists both folders.
2. **Given** jobs are running when a restart is needed, **When** the researcher asks to share or
   stop sharing a folder, **Then** they are told jobs are running and can wait for them to finish or
   restart anyway. Queued jobs are not lost either way.
3. **Given** a folder is shared, **When** the researcher stops sharing it (from the start-up program,
   or from Settings in the viewer), **Then** from the next start VideoAnnotator can no longer read
   it, and Settings says when the change takes effect.
4. **Given** the viewer's Settings, **When** the researcher opens it, **Then** it lists every shared
   folder with its path, says each is read-only, and names the results folder.

---

### User Story 4 - Never shown the inside of the container (Priority: P1)

Whatever the setup, a researcher using VideoAnnotator in a container never sees the container's own
folders in My folders. When no folder is shared, they are told plainly why My folders is empty and
how to share one, and can upload meanwhile.

**Why this priority**: This is the failure that started this spec. It must not recur for someone who
starts the container another way, such as the compose file or an old command.

**Independent Test**: Start the container with no shared folder, open the new-job wizard, and
confirm no container folder is listed anywhere, and the message says how to share a folder.

**Acceptance Scenarios**:

1. **Given** VideoAnnotator runs in a container, **When** the researcher opens My folders, **Then**
   only shared folders are listed, never the container's own filesystem.
2. **Given** no folder is shared, **When** the researcher opens the new-job wizard, **Then** they see:
   "VideoAnnotator can only see folders you share with it", with how to share one, and upload
   offered.
3. **Given** a job's video was in a folder that is no longer shared, **When** it is needed (a rerun,
   playback, using a dataset), **Then** the message says the folder isn't shared any more, not that
   the video was moved or deleted.

---

### User Story 5 - Plain words when something goes wrong (Priority: P2)

The researcher runs the start-up program while Docker Desktop isn't running, or while another
program uses the port, or on a machine where the container runs out of memory. Each time they get
one plain sentence saying what is wrong and what to do.

**Why this priority**: When the container shows through, it shows through as these errors. Raw
container-engine messages ("Cannot connect to the Docker daemon") are where non-technical users get
stuck.

**Independent Test**: Trigger each listed situation and check the message names the problem and
the next step without container vocabulary.

**Acceptance Scenarios**:

1. **Given** no container engine is installed, **When** the start-up program runs, **Then** it says
   neither Docker nor Podman was found and points to how to install one.
2. **Given** the engine is installed but not running, **When** the start-up program runs, **Then** it
   starts the engine where it can (Podman's virtual machine), or says to start Docker Desktop and
   run the program again.
3. **Given** the port is in use, an image download fails, or the container runs out of memory,
   **When** the start-up program runs, **Then** each gives its own plain message and next step.

---

### User Story 6 - Docker or Podman, whichever I have (Priority: P2)

A researcher's institution doesn't allow Docker Desktop, so they install Podman Desktop. Everything
in Stories 1–5 works the same.

**Why this priority**: Podman is free for every institution. Both must be first class, not one
supported and one tolerated.

**Independent Test**: Run Stories 1–4 with Podman on Linux and Windows and get the same results as
with Docker.

**Acceptance Scenarios**:

1. **Given** only Podman is installed, **When** the start-up program runs, **Then** it uses Podman
   with no extra step.
2. **Given** both are installed, **When** the start-up program runs, **Then** it uses the one that is
   running. If both are, it uses the one used last time, and says which.
3. **Given** Podman on Linux, **When** results are written, **Then** they belong to the researcher.

---

### Edge Cases

- **Sharing too much.** The researcher picks their home folder, the top of a drive, or a system
  folder. They are told exactly what would become readable ("everything in your home folder,
  including documents unrelated to your research"), and it is shared only if they explicitly
  confirm. The easy path is choosing a narrower folder.
- **The results folder inside a shared folder.** It is shared writable, as results, and the warning
  from spec 022 applies.
- **The same folder chosen twice, or a folder inside one already shared**: shared once; nothing
  changes.
- **A shared folder renamed while VideoAnnotator runs**: handled at the next start, as a missing
  folder (Story 2, scenario 3). Jobs needing it say the folder isn't found.
- **External drives**: shared like any folder. When the drive is unplugged at start, it is skipped
  with a note.
- **Paths with spaces, accents or non-Latin characters**: shared and shown correctly.
- **Windows paths**: shown as the researcher's own Windows paths (`C:\Users\ada\Studies`), with
  backslashes, wherever the viewer shows a location.
- **No folder picker available** (a Linux machine without a desktop, a remote terminal): the
  program asks for the path as text, suggesting likely folders.
- **A GPU is present but unusable by the container** (no NVIDIA container support installed): it
  starts on the CPU and says how to enable the GPU, rather than failing.
- **Two copies started at once**: the second says VideoAnnotator is already running and opens the
  browser on it.
- **The compose route** (labs, servers): unchanged and documented. Folders under compose also never
  show the container's filesystem (Story 4).
- **Not on loopback**: the container is reachable only from this computer. Anyone who publishes it
  more widely is in the lab-server setup, which keeps spec 022's rules.

## Requirements *(mandatory)*

### Functional Requirements

**Starting VideoAnnotator**

- **FR-001**: There MUST be one start-up program per platform family (Linux and macOS, and
  Windows). It starts VideoAnnotator in a container with no container command, path or setting typed
  by the researcher.
- **FR-002**: The start-up program MUST work with Docker and with Podman, through one shared
  behaviour, choosing between them as in Story 6.
- **FR-003**: The start-up program MUST make VideoAnnotator reachable from this computer only, and
  open the viewer in the researcher's browser when it is ready.
- **FR-004**: The start-up program MUST use an NVIDIA GPU when the computer and container engine
  support it, and otherwise start on the CPU with a note on enabling the GPU.
- **FR-005**: Each failure in Story 5 MUST produce one plain message naming the problem and the next
  step, with no container vocabulary left unexplained.
- **FR-006**: When VideoAnnotator is already running, starting it again MUST open the browser on the
  running one rather than start a second.

**Sharing folders**

- **FR-007**: VideoAnnotator in a container MUST be able to read only folders the researcher has
  shared, and MUST only be able to write to the results folder.
- **FR-008**: Shared folders MUST be read-only to VideoAnnotator.
- **FR-009**: On first start with no shared folder, the start-up program MUST ask which folder the
  videos are in, with a folder picker where the computer has one, and MUST confirm in plain words
  what becomes readable and where results go before sharing.
- **FR-010**: Choosing a home folder, a drive's top level or a system folder MUST require an explicit
  confirmation that names what becomes readable.
- **FR-011**: Shared folders and the results folder MUST be remembered on the researcher's computer,
  and reused without asking on every start, including after updates.
- **FR-012**: A shared folder missing at start MUST be skipped with a note naming it, and VideoAnnotator
  MUST still start.
- **FR-013**: Researchers MUST be able to share another folder and stop sharing one through the
  start-up program. A needed restart MUST be announced, and MUST wait for or ask about running jobs.
  Queued jobs MUST survive it.
- **FR-014**: Each start MUST state which folders VideoAnnotator can read and where results go.

**Where things appear**

- **FR-015**: Locations of shared folders, videos and results MUST be shown as the researcher's own
  computer shows them, including Windows paths, everywhere the viewer shows a location.
- **FR-016**: Inside a container, My folders MUST list only shared folders, never the container's
  own filesystem, however the container was started.
- **FR-017**: With no folder shared, the new-job wizard MUST say that VideoAnnotator can only see
  folders the researcher shares with it, say how to share one, and offer upload.
- **FR-018**: Settings MUST list the shared folders (read-only) and the results folder, and MUST let
  the researcher stop sharing a folder, saying when that takes effect.
- **FR-019**: Messages about an unavailable video MUST distinguish a folder that is no longer shared
  from a video that was moved or deleted.

**Files and continuity**

- **FR-020**: Results written from the container MUST belong to the researcher on their computer,
  with both Docker and Podman.
- **FR-021**: Jobs, results, saved datasets, settings and installed pipelines MUST survive restarts
  and updates of the container.
- **FR-022**: Jobs and datasets referring to videos in shared folders MUST remain valid across
  restarts, because each shared folder keeps the same path every time.

**Compatibility**

- **FR-023**: The documented compose route for labs and servers MUST keep working, and MUST apply
  FR-016 and FR-017.
- **FR-024**: Existing installs (spec 022's `VIDEOS_DIR` and `RESULTS_DIR`) MUST keep working. The
  first start of the start-up program on such an install MUST offer to reuse them.

### Later phase (not in this spec)

- Sharing a folder from inside the viewer ("Share a folder…") through a small helper running on the
  researcher's computer. The start-up program MUST be designed so it can grow into that helper,
  without the researcher installing anything else.

### Key Entities

- **Shared folder**: a folder on the researcher's computer that VideoAnnotator may read, never
  change. It has its path as the researcher's computer shows it, when it was shared, and whether it
  was present at the last start.
- **Results folder**: the one folder VideoAnnotator writes to (spec 022), on the researcher's
  computer.
- **Start-up settings**: on the researcher's computer, what the start-up program remembers: the
  shared folders, the results folder, which container engine was used, and the VideoAnnotator
  version.
- **Container engine**: Docker or Podman, as installed by the researcher.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A researcher with only a container engine installed goes from running the start-up
  program to a first finished run on their own video in under 10 minutes, excluding downloads,
  without typing a command, path or setting beyond choosing a folder.
- **SC-002**: In walkthrough testing, no researcher-facing screen shows a container's own folder, or
  a container path such as `/root`, `/videos` or `/results`.
- **SC-003**: Every second and later start asks no question, unless a shared folder is missing or
  the researcher asks to change something.
- **SC-004**: VideoAnnotator can read nothing outside the folders the researcher shared, verified by
  trying to reach an unshared folder through the viewer and the API.
- **SC-005**: Each failure in Story 5 produces a message that a non-technical tester can act on
  without help, in walkthrough testing.
- **SC-006**: Stories 1–4 pass with both Docker and Podman, on Linux and on Windows.
- **SC-007**: The explanation of what VideoAnnotator can access fits in one sentence a researcher can
  put in an ethics application: the folders they shared, read-only, plus the results folder.

## Assumptions

- **Researchers install a container engine themselves**, Docker Desktop or Podman Desktop, following
  the documentation. Installing it for them is out of scope.
- **Linux and Windows are tested; macOS is supported but community-tested.** No Mac is available to
  the project. The Linux and macOS start-up program is shared apart from the folder picker, and a
  pilot lab with Macs is asked to run the walkthrough.
- **Containers on macOS can't use the Mac's GPU**, so Mac users run on the CPU. This is documented,
  not solved here.
- **The container is reachable from this computer only**, as in spec 022. Lab servers shared over
  a network remain the compose route, with upload for remote users.
- **A restart takes seconds**, not minutes, once the image is downloaded, so restarting to change
  shared folders is acceptable until the later phase.
- **The start-up program needs no software beyond what the platform has** (a shell on Linux and
  macOS, PowerShell on Windows) and the container engine.
- **Further changes come from pilot users.** This spec settles the model; wording and details are
  expected to change with pilot feedback.

## Out of Scope

- Sharing folders from inside the viewer (later phase, above).
- A graphical installer, desktop app or tray icon (the start-up program may get a desktop shortcut).
- Native, container-free installs for researchers.
- GPU acceleration on macOS.
- Multi-user lab servers and remote access (compose route, spec 022 rules).
- The viewer's wider redesign and vocabulary (spec 023). This spec keeps "My folders" as the name.
