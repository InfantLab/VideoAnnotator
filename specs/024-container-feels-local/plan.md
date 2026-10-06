# Implementation Plan: A Container That Feels Local

**Branch**: `024-container-feels-local` (work lands on `1.6-dev`) | **Date**: 2026-10-06 |
**Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/024-container-feels-local/spec.md`

## Summary

Researchers start VideoAnnotator with a launcher, never a container command:
- **The launcher.** `videoannotator-start`: a POSIX shell script, plus a PowerShell twin on
  Windows. It finds Docker or Podman, asks once which folder the videos are in, shares it
  read-only at its real path, remembers it, starts the every-pipeline image on 127.0.0.1, and
  opens the viewer already connected.
- **The server** never lists the container's own filesystem. It says plainly when nothing is
  shared, distinguishes "not shared any more" from "moved or deleted", gives results to the
  researcher, and accepts "stop sharing" requests through a one-way folder that can only make
  sharing narrower.
- **The image** is published to GHCR, pullable by both engines.

## Technical Context

- **Language/Version**:
  - Launcher: POSIX `sh` (Linux, macOS) and Windows PowerShell 5.1+ (Windows).
  - Server: Python 3.12/3.13. Viewer: TypeScript and React 18.
- **Primary Dependencies**: none new.
  - Launcher: only the platform's shell and the container engine's CLI. Folder pickers use
    platform tools (Windows Forms, `osascript`, `zenity`/`kdialog`).
  - Server and viewer: existing stack.
- **Storage**:
  - Host: the launcher's `start.conf` and `requests/` folder (data-model.md).
  - Container: unchanged named volumes (`videoannotator-models`, `-database`, `-storage`), shared
    with `docker-compose.yml`. No database change.
- **Testing**:
  - pytest for the server changes.
  - Vitest for Settings and messages.
  - The launcher:
    - `bats` (sh) and Pester (PowerShell), over one shared table of cases
      (`tests/launcher/cases.json`);
    - shellcheck and PSScriptAnalyzer;
    - a Linux CI end-to-end job with Docker and with Podman;
    - the quickstart on Windows by hand.
- **Target Platform**:
  - Linux (Docker Engine, Podman) and Windows (Docker Desktop, Podman Desktop): tested.
  - macOS (Docker Desktop, Podman Desktop): supported, community-tested.
  - `linux/amd64` image.
- **Project Type**: a web application (server and viewer) plus two host-side launcher scripts and
  a CI image job.
- **Performance Goals**:
  - Second and later starts: ready within 60 s once the image is downloaded. Restart for a share
    change: under 60 s.
  - Launcher logic itself: under 2 s, excluding the engine.
- **Constraints**:
  - Read-only shares; only the results folder writable.
  - Published on 127.0.0.1 only.
  - The container can never widen its own access.
  - No container vocabulary in any message without a plain explanation.
  - No software installed on the host beyond the launcher.
- **Scale/Scope**:
  - One researcher per machine; 1–10 shared folders.
  - Launcher ~600 lines per script.
  - Server: ~6 files touched. Viewer: Settings plus two messages. CI: one new image job and one
    new end-to-end job.

## Constitution Check

*GATE: checked before research and again after design (below).*

| Principle | Assessment |
|---|---|
| **I. Local-first, no telemetry** | **Pass, strengthened.** Everything runs on the researcher's machine; the server is reachable from it only. The launcher contacts only the image registry (download, update), as Docker would. No telemetry. |
| **II. Stable pipeline contract, open formats** | **Pass.** No pipeline or output change. |
| **III. Provenance** | **Pass.** `run.json` sources record the host path the researcher knows (display path), a small clarity gain; output provenance unchanged. |
| **IV. Modular by construction** | **Pass, noted.** The launcher runs the every-pipeline image (research R7) because runtime-installed extras don't survive container recreation. The slim core install and the extras groups are unchanged; labs can still run the slim image and install extras. |
| **V. Backward compatibility** | **Pass, with one changed default** (Complexity Tracking). Compose, `VIDEOS_DIR`/`RESULTS_DIR`, named volumes, API requests and plain installs keep working; the launcher offers to reuse an existing compose install's folders and volumes. |
| **VI. Faithful display** | **Pass.** Display of annotations is unchanged. |
| **Engineering standards** | New launcher code is tested from shared cases on two shells, with shellcheck/PSScriptAnalyzer and a CI end-to-end job; server and viewer changes have tests. Docs: install guide (launcher first), getting started, CHANGELOG, troubleshooting. |

**Post-design re-check**: unchanged. Design adds no runtime dependency; the only new published
artifact is the GHCR image.

## Project Structure

### Documentation (this feature)

```
specs/024-container-feels-local/
├── spec.md
├── plan.md              # this file
├── research.md          # R1–R16
├── data-model.md
├── quickstart.md        # walkthrough: Windows (Docker, Podman), Linux; macOS by a pilot lab
├── contracts/
│   ├── launcher.md      # commands, output, the run it builds, messages
│   ├── api.md           # access fields, POST /ingest/shares/stop, messages
│   └── image.md         # GHCR tags
├── checklists/requirements.md
└── tasks.md             # /speckit-tasks
```

### Source Code (repository root)

```
launcher/                                # NEW
├── videoannotator-start                 # POSIX sh: Linux, macOS
├── videoannotator-start.ps1             # PowerShell: Windows
├── videoannotator-start.cmd             # double-click wrapper (-ExecutionPolicy Bypass)
├── install.sh, install.ps1              # one-line installers + desktop shortcut (R16)
└── README.md                            # for maintainers: structure, how to test

src/videoannotator/
├── api/v1/ingest.py                     # no home default in a container (R8); shares + stop (R10)
├── results_folder.py                    # results owner (R6); Windows display paths (R3);
│                                        #   video_unavailable_reason (R9)
├── config_env.py                        # LAUNCHER, RESULTS_OWNER; HOST_PATHS backslashes
├── batch/job_execution.py               # missing-video message via R9
└── api/v1/batches.py, jobs.py           # rerun / video_available messages via R9

viewer/src/
├── components/VideosAndResultsCard.tsx  # Shared folders, Stop sharing, "next start"
├── components/ServerFolderPicker.tsx    # no-share card wording (launcher vs compose)
└── types/ingest.ts, api/client.ts       # new access fields, stopSharing()

Dockerfile                               # /app/launcher/requests
.github/workflows/ci-cd.yml              # GHCR image job (slim + all); launcher tests; e2e job
tests/launcher/cases.json, *.bats, *.Tests.ps1, test_launcher_e2e.sh   # NEW
tests/api/test_shares.py, tests/unit/test_results_owner.py             # NEW
docs/installation/INSTALLATION.md, docs/usage/GETTING_STARTED.md, docs/troubleshooting
```

**Structure decision**: the launcher is its own top-level folder because it runs on the host, not
in the package. It is versioned with the package and attached to each release.

## Delivery Order

Each step is testable on its own and committed when green.

1. **Server: never the container's filesystem.** No home default in a container, plus the
   no-share message (R8). Small, and fixes the screen that started this. (Story 4.)
2. **Server: messages and ownership.** R9 "not shared any more", R6 results owner, R3 Windows
   display paths. (Stories 1 and 4.)
3. **Image on GHCR.** The CI job for slim and `-all`; `/app/launcher/requests`. (R7.)
4. **Launcher (sh).** Engine detection, settings, picker, broad-share guard, the run command,
   key, browser, messages; `bats` over shared cases; a Linux end-to-end job with Docker and
   Podman. (Stories 1, 2, 5, 6.)
5. **Share and unshare.** Restart with the running-jobs check; the stop-sharing channel; Settings'
   Shared folders. (Story 3.)
6. **Launcher (PowerShell).** Same behaviour, plus Pester over the same cases. Windows walkthrough
   by hand. (Stories 1–6 on Windows.)
7. **Installers and shortcuts**, then docs: install guide (launcher first, compose for labs),
   getting started, troubleshooting, CHANGELOG. Then the quickstart on Windows and Linux; a
   pilot lab on macOS.

## Complexity Tracking

| Deviation | Why needed | Simpler alternative rejected because |
|---|---|---|
| In a container with no shared folder, ingest's allowed folders become none instead of the server user's home. | FR-016: the home folder inside a container is the container's own filesystem (`/root`). | Keeping the default shows researchers the inside of the box, which is the failure this spec exists to fix. Plain installs keep the home default; compose installs with `VIDEOS_DIR` set are unaffected. |
| Two launcher implementations (sh, PowerShell). | Researchers' machines have only their platform's shell (R1). | A compiled launcher needs a new toolchain and signed binaries for three OSes; Python isn't on Windows by default. Drift is contained by shared test cases. |
| The launcher runs the large every-pipeline image. | Runtime-installed extras are lost when the container is recreated for a share change (R7). | A slim image plus an extras volume breaks silently on update; revisit with v1.7's plugin packaging. |
