# Data Model: A Container That Feels Local

No new tables. The new state lives on the researcher's computer, in the launcher's settings; the
server learns it from environment variables at each start.

## What persists where (research.md R7)

| Kind | Store | Survives app update | Survives share change | Reset by |
|---|---|---|---|---|
| App | image `ghcr.io/infantlab/videoannotator:<version>` | replaced | kept | `update` |
| Add-ons (installed pipeline groups) | completed `extras_install_jobs` rows + `videoannotator-cache` volume | restored | restored | removing the database or cache volume |
| Model weights | `videoannotator-models` volume | yes | yes | removing that volume |
| Activity and settings | `videoannotator-database`, `videoannotator-storage` volumes | yes | yes | removing those volumes |
| Shared folders, results | host disk; launcher's `start.conf` | yes | as changed | the researcher |

**Installed groups**: a group is installed when it has an `extras_install_jobs` row with status
`completed` (spec 005). At server start, each such group whose packages aren't importable is
reinstalled in the background (state `restoring` on the readiness card). Jobs needing it stay
pending until it is ready.

## Start-up settings (host file, `start.conf`; research.md R13)

Plain `key=value` lines, one setting per line; `share` repeats. Mode 600 on Linux and macOS.

| Key | Example | Meaning |
|---|---|---|
| `engine` | `podman` | Engine used last; chosen first when both are running |
| `image` | `ghcr.io/infantlab/videoannotator:1.6.0` | Image this launcher runs (pinned to its version) |
| `results` | `/home/ada/VideoAnnotator` | Results folder on the host (writable) |
| `share` | `/home/ada/Studies` | One per shared folder, as the host path; read-only |
| `key` | `va_…` | Admin API key the launcher created (R14) |
| `port` | `18011` | Port on 127.0.0.1 |

**Rules**:
- `share` paths are absolute, normalised (no trailing separator), unique, and none is inside another.
- A share missing at start stays in the file (it may be an unplugged drive) and is skipped for that
  start with a note.
- Only the launcher writes this file. It is never mounted into the container.

## Stop-sharing requests (host folder `requests/`, mounted writable at `/app/launcher/requests`; R10)

`stop-sharing.txt`: one host path per line, appended by the server. At the next start the launcher
removes each listed path that is currently a `share`, ignores anything else, then empties the
file. A request can never add a share.

## Server settings (environment, set by the launcher; additions to spec 022's table)

| Setting | Set to | Purpose |
|---|---|---|
| `VIDEOANNOTATOR_INGEST_ROOTS` | the container paths of the present shares | Allowed folders (022) |
| `VIDEOANNOTATOR_RESULTS_DIR` | container path of the results folder | Results root (022) |
| `VIDEOANNOTATOR_HOST_PATHS` | `container=host` per share and for results | Display as the host shows it (022); Windows host paths keep backslashes |
| `VIDEOANNOTATOR_PUBLISHED_LOCALLY` | `1` | Every caller is this computer (022) |
| `VIDEOANNOTATOR_LAUNCHER` | `1` | Started by the launcher: Settings offers Stop sharing (new) |
| `VIDEOANNOTATOR_RESULTS_OWNER` | `uid:gid` | Docker on Linux only: owner for results (new, R6) |

## Caller access (computed; additions to `GET /api/v1/ingest/access`)

| Field | Meaning |
|---|---|
| `in_container` | The server runs in a container |
| `managed_by_launcher` | `VIDEOANNOTATOR_LAUNCHER` is set |
| `shares` | Each shared folder: `path`, `display_path`, `stop_requested` |

`allowed_folders` stays as it is (the shares that are present). In a container with no share, both
are empty and `reason` explains how to share a folder (R8).

## Shared folder (concept)

A folder on the host that VideoAnnotator may read. Identified by its host path; mounted read-only at
its real path or under `/host` (R3); present or missing at each start; may have a stop request
pending.
