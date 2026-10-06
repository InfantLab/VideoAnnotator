# Research: A Container That Feels Local

Decisions behind [plan.md](plan.md). Each records what was chosen, why, and what else was
considered. Code references are as of commit `30daa60`.

## R1. One launcher, written twice, kept thin

**Decision**: Two scripts with the same behaviour:
- **`videoannotator-start`**: POSIX `sh`, for Linux and macOS.
- **`videoannotator-start.ps1`**: Windows PowerShell 5.1 and later. A `videoannotator-start.cmd`
  wrapper runs it with `-ExecutionPolicy Bypass`, so double-clicking works on Windows's default
  execution policy.

Both keep logic to the minimum: find the engine, read and write their settings, ask for folders,
build one `run` command, and translate errors. Everything that can live in the server lives
there (R8–R11).

**Rationale**: Researchers' machines have a shell or PowerShell and nothing else we can rely on,
and the container image can't build its own run command. Keeping the scripts thin limits the
drift between two implementations. Drift is caught by one shared table of test cases that both
test suites read (R12).

**Alternatives considered**:
- *A compiled launcher (Go, Rust)*: one implementation, but a new toolchain and signed binaries
  for three OSes. Worth revisiting if the later in-viewer helper (spec's later phase) grows.
- *Python on the host*: not reliably installed on researchers' Windows machines.
- *Compose*: `podman compose` varies between versions, and compose files can't ask questions or
  take a folder list.

## R2. Docker and Podman through one `run` command

**Decision**:
- **Detection**: use whichever engine is running. If both are, use the one recorded in the
  settings file, otherwise Docker. If neither is running but Podman is installed, run
  `podman machine start` on macOS and Windows; on Linux, Podman needs no service.
- **One command for both**, using only flags the two share: `run -d --name videoannotator -p
  127.0.0.1:18011:18011 -v ... -e ...`.
- **Fully qualified image names.** Podman's short-name resolution can stop and ask which registry
  to use.
- **Restart behaviour**: `--restart unless-stopped` is not used, because Podman needs systemd for
  it. The launcher starts VideoAnnotator when asked.

**Rationale**: FR-002 and Story 6. One command keeps the two engines on one code path.

**Alternatives considered**: separate code paths per engine, rejected because twice the
surface means twice the drift.

## R3. Sharing a folder: real paths where safe, mapped otherwise

**Decision**: each shared folder is mounted read-only (`:ro`):
- **Linux and macOS**: at its **real path** (`/home/ada/Studies` stays `/home/ada/Studies`).
  The exception is when its first component is one of the image's own top-level directories
  (`app`, `bin`, `boot`, `dev`, `etc`, `lib*`, `opt`, `proc`, `root`, `run`, `sbin`, `srv`,
  `sys`, `tmp`, `usr`, `var`). Those go under `/host` (`/opt/data` → `/host/opt/data`).
- **Windows**: at `/<drive>/<path>` (`C:\Users\ada\Studies` → `/c/Users/ada/Studies`).
- **Display**: every share is passed to the server as a `VIDEOANNOTATOR_HOST_PATHS` pair
  (container path = host path). Everything the viewer shows is therefore the researcher's own
  path, whatever the mount point.
- **Allowed folders**: the shares are also the server's `VIDEOANNOTATOR_INGEST_ROOTS`. Container
  paths never contain `:`, since we build them.
- **Results folder**: the only writable mount (`~/VideoAnnotator`), at its real path by the same
  rule, and set as `VIDEOANNOTATOR_RESULTS_DIR`.

**Rationale**:
- Real paths mean job records, saved datasets and `run.json` sources are the paths the
  researcher knows, and they stay the same on every start (FR-022), the Apptainer model.
- Mapping the exceptions avoids shadowing the container's own system folders.
- Reusing spec 022's two environment variables needs no new server configuration format.

**Alternatives considered**:
- *Everything under `/host/<path>`*: uniform, but container paths would leak into records.
- *Always `/videos`*: today's 022 setup, the source of the alien paths.

## R4. What may be shared, and the broad-share confirmation

**Decision**: the launcher classifies a chosen folder before sharing it:
- **Refused outright**: anything inside the container's own folders after R3 mapping (none
  remain), and the results folder itself (it's already shared, writable).
- **Broad, needs explicit confirmation naming what becomes readable**:
  - the user's home folder;
  - a filesystem or drive root (`/`, `C:\`, `/Volumes`, `/media/<user>`, `/mnt`);
  - `/Users` or `C:\Users`;
  - system folders (`/etc`, `/usr`, `C:\Windows`, `C:\Program Files`).

  The confirmation names the folder and gives an example of what it contains ("everything in
  your home folder, including documents unrelated to your research"). It defaults to "Choose a
  narrower folder".
- **Duplicates**: a folder already shared, or inside a shared folder, is not shared again. A
  folder containing shared folders replaces them, after the same confirmation if it is broad.

**Rationale**: FR-010 and the edge cases. Least access is the easy path, and broad access is
possible but deliberate.

## R5. Folder picker

**Decision**:
- **Windows**: `System.Windows.Forms.FolderBrowserDialog`, from PowerShell.
- **macOS**: `osascript -e 'POSIX path of (choose folder ...)'`.
- **Linux**: `zenity --file-selection --directory`, else `kdialog --getexistingdirectory`.
- **Otherwise** (no display, SSH): ask for the path as text, listing likely folders under the
  home folder that hold videos (Videos, Movies, Desktop, Documents; at most 2 levels deep, at most
  2,000 entries looked at).

Every picker opens in the Videos folder if it exists, else Documents, not Home.

**Rationale**: the spec's "folder picker where the computer has one". Opening outside Home nudges
towards a narrower share.

## R6. Who owns the results

**Decision**:
- **Podman (rootless)**: run with the default user namespace. Root in the container is the
  researcher on the host, so results belong to them. `--userns=keep-id` is **not** used: it
  makes the container's user non-root, and the image's `/app` (the database, model and storage
  volumes, the Python environment) is root-owned.
- **Docker Desktop (macOS, Windows)**: its file sharing already makes bind-mounted files the
  user's.
- **Docker Engine on Linux (rootful)**: the launcher passes
  `VIDEOANNOTATOR_RESULTS_OWNER=<uid>:<gid>`. The server, which runs as root there, gives every
  folder it creates under the results root to that owner at once. It does the same for every
  file in a job's folder, and `run.json`, when the job settles (`record_job_finished`).

**Rationale**: FR-020, with the fewest moving parts. Results are the only files on the host that
VideoAnnotator writes.

**Alternatives considered**:
- *The linuxserver.io `PUID/PGID` entrypoint* (start as root, chown `/app`, drop privileges):
  the standard pattern, but it means chowning a multi-gigabyte Python environment on first start
  and on every update, and runtime extras installs (R7) would need the same treatment.
- *`--userns=keep-id`*: see above.

**Known gap**: on Linux Docker, files of a job still running are root-owned until the job ends.
They are readable meanwhile; deleting them needs the job to finish first.

## R7. Five things, each persistent and updatable on its own

**Decision**: VideoAnnotator is treated like any installed app. Five kinds of thing each live in
their own place, so each can be updated, reset or kept independently:

| Like any app | In VideoAnnotator | Where it lives | Changes when |
|---|---|---|---|
| The app | the **slim** image | `ghcr.io/infantlab/videoannotator:<version>` | the researcher updates (`videoannotator-start update`) |
| Add-ons | pipeline groups they installed (spec 005) | remembered in the database (completed `extras_install_jobs`); downloads cached in the `videoannotator-cache` volume | they install one |
| Big downloads | model weights | `videoannotator-models` volume (unchanged) | a pipeline first needs them |
| Activity and settings | jobs, keys, datasets, presets, prompts | `videoannotator-database` and `videoannotator-storage` volumes (unchanged) | they use it |
| Their files | shared folders (read-only), results folder | their own disk | they share a folder, or a run finishes |

**Add-ons survive the app being replaced.**
- **The problem**: pipelines installed from the viewer go into the container's own Python
  environment (`api/extras_install.py`), which is lost whenever the container is recreated: on
  every update, and on every share change, since mounts are fixed at creation. Compose has the
  same latent problem whenever it recreates its container.
- **Remembering**: a group counts as installed when it has a completed install record in the
  database. No new table.
- **Restoring**: at server start, each remembered group whose packages aren't importable is
  reinstalled in the background through the existing installer. The pipeline card shows
  "Restoring…", the same progress display as installing.
- **Caching**: `UV_CACHE_DIR=/app/cache/uv` points at the `videoannotator-cache` volume. A
  restore copies from the cache rather than downloading, about a minute for the largest group;
  after an update, only changed packages download.
- **Waiting jobs**: jobs that need a group being restored stay queued until it is ready, not
  failed.
- **Versions**: because restore reinstalls against the *current* app, an update never leaves add-ons
  built for the old one.

**The app image**: published to **GHCR**, slim, tags `<version>` and `latest`, built in CI with
`GITHUB_TOKEN`; OCI-compliant, `linux/amd64` (arm64 later). The launcher pins the image to its own
version. Docker Hub stays a mirror if its secrets exist. An every-pipeline image (`EXTRAS=all`)
remains buildable for labs that want one, but is not what researchers get.

**Rationale**: researchers already think of software this way: install the app once, update it,
download big model files, keep an internal record of activity and settings, and do each
independently with as much persistence as possible. Mapping each to its own place delivers that.
It keeps VideoAnnotator modular (constitution IV): only what a researcher uses is downloaded.

**Alternatives considered**:
- *An every-pipeline image* (the first draft of this decision): survives recreation by having
  nothing to install, but downloads every pipeline whether used or not, against the project's move
  away from a monolithic image.
- *The whole Python environment on a volume*: survives recreation, but after an update the volume
  still holds the old app's environment, and add-ons built for the old core break silently.
- *Never recreating the container*: impossible, since adding a share needs new mounts.

## R8. Inside a container, no shared folder means no folders

**Decision**: when `in_container()` is true and `VIDEOANNOTATOR_INGEST_ROOTS` is empty,
`allowed_roots()` returns **no** folders, instead of the server user's home (`/root`). The
access endpoint then reports `can_read_in_place: false`, with the reason "VideoAnnotator can only
see folders you share with it" plus how to share one (the launcher's command). Places are then
empty too. This applies however the container was started (compose included, FR-023).

**Rationale**: FR-016 and FR-017, and the screen that started this spec.

## R9. "Not shared any more" vs "moved or deleted"

**Decision**: one helper, `video_unavailable_reason(path)`, used everywhere a missing video is
explained (job start, `video_available`, rerun check and relocation, dataset use):
- **Outside every current allowed folder**: "`<folder>` isn't shared with VideoAnnotator any
  more". It names the nearest parent the researcher would recognise, the display path of the
  missing path's top folder.
- **Otherwise**: today's "moved or deleted since the job was created".

**Rationale**: FR-019. Being outside the allowed folders is exactly "not shared", with no history
to keep.

## R10. Stopping a share from Settings

**Decision**:
- **The channel**: the launcher mounts one small host folder writable into the container at
  `/app/launcher/requests`, and sets `VIDEOANNOTATOR_LAUNCHER=1`.
- **The request**: Settings' **Stop sharing** calls `POST /api/v1/ingest/shares/stop`, which
  appends the folder's host path to `stop-sharing.txt` there. The response, and later access
  reads, mark that share "stops when VideoAnnotator next starts".
- **Applying it**: at the next start, the launcher removes **only** listed paths that are
  currently shared, then clears the file.
- **What a request can't do**: add a share, or change anything else. The launcher's own
  settings file lives outside the mounted folder (R13).
- **Without the launcher** (compose), there is no Stop sharing button. Settings says to change
  the compose settings.

**Rationale**:
- FR-018 without a process on the host.
- Safe by construction: a compromised container can at most ask to share *less*.
- This is also the seed of the later in-viewer helper.

**Alternatives considered**: mounting the launcher's settings file writable. Rejected, because the
container could then add shares.

## R11. Restarting for a change, and running jobs

**Decision**:
- **Before stopping**, the launcher asks the server for running jobs (`GET /api/v1/jobs?
  status_filter=running`, with the launcher's key). If there are any: "2 videos are being
  processed. [Wait for them] [Restart now: they'll be marked failed and can be retried]".
- **Waiting** polls every 10 s.
- **Restart**: `stop -t 30`, then `rm`, then `run` with the new mounts.

**Rationale**: FR-013. Queued jobs are in the database volume and survive; a job that was running
is marked failed on the next start (`background_tasks._resolve_orphaned_running_jobs`) and can be
retried. That's existing, tested behaviour, so the launcher only has to say it.

## R12. Testing two scripts and three OSes without a Mac

**Decision**:
- **Shared cases**: `tests/launcher/cases.json` lists inputs (OS, engine, shares, results folder,
  GPU, uid) and the expected `run` arguments, classifications (R4) and messages.
  - **`bats`** runs the `sh` script's pure functions against it (Linux CI, and macOS via the same
    script).
  - **Pester** runs the PowerShell functions against it (Windows CI).
- **End to end, Linux CI**: on the Ubuntu runner, which has Docker and Podman, start a
  locally built image with a shared temp folder through the launcher in non-interactive mode
  (`--share <path> --yes`). Run a job with `stub_pipeline`, and check that the results belong
  to the runner's user and that `/root` is never listed. Do it once with each engine.
- **Windows**: GitHub's Windows runners can't run Linux containers, so the Windows end to end is a
  manual walkthrough (quickstart) on the maintainer's machine, with Docker Desktop and Podman
  Desktop.
- **macOS**: shellcheck plus the shared cases; a pilot lab runs the quickstart (community-tested).

**Rationale**: SC-006 within the project's real constraints.

## R13. Where the launcher keeps things on the host

**Decision**:
- **Settings file** (plain `key=value` lines, readable by both scripts without a JSON parser):
  - Linux: `$XDG_CONFIG_HOME/videoannotator/start.conf` (default `~/.config/...`);
  - macOS: `~/Library/Application Support/VideoAnnotator/start.conf`;
  - Windows: `%APPDATA%\VideoAnnotator\start.conf`.
- **Contents**: `engine=`, `image=`, `results=`, one `share=` line per folder, and `key=` for the
  admin API key. Mode 600 on Linux and macOS.
- **The requests folder** (R10) sits beside it, in `requests/`.
- **Migration** (FR-024): if no settings file exists but spec 022's `VIDEOS_DIR`/`RESULTS_DIR`
  are set, or the old named volumes exist, offer to reuse them.

## R14. Getting the viewer connected without a key prompt

**Decision**:
- **First start**: once the server is healthy, the launcher runs `exec videoannotator
  generate-token --user researcher@localhost --key-name "start-up program" --admin --output
  /tmp/key.json`, reads the key, deletes the file, and stores the key in the settings file.
- **Every start**: it opens `http://127.0.0.1:18011/viewer-connect?token=<key>`, so the viewer
  is connected with no Settings step.
- **Lost key**: if the key is missing or rejected (database volume reset), the launcher makes a
  new one the same way.

**Rationale**: the viewer must be connected as an administrator for My folders (in-place reading
is admin-only). The one-click link already exists (`/viewer-connect`), and `generate-token`
already grants admin to the first user.

## R15. GPU

**Decision**:
- **Docker**: add `--gpus all` when `nvidia-smi` exists on the host and `docker info` lists the
  `nvidia` runtime, or (Docker Desktop on Windows) when `nvidia-smi` works.
- **Podman**: add `--device nvidia.com/gpu=all` when the NVIDIA CDI spec exists
  (`nvidia-ctk cdi list` shows devices).
- **Otherwise**: run on the CPU and print one line on how to enable the GPU, with a link to the
  install guide.

If the container then fails to start because of the GPU flag, retry once without it and say so.

## R16. Getting the launcher onto the researcher's machine

**Decision**:
- **Release assets**: `videoannotator-start`, `videoannotator-start.ps1` and
  `videoannotator-start.cmd`.
- **A one-line install, as uv does**:
  - Linux and macOS: `curl -LsSf https://.../install.sh | sh`;
  - Windows: `powershell -c "irm https://.../install.ps1 | iex"`.

  These copy the launcher to `~/.local/bin` (Windows: `%LOCALAPPDATA%\VideoAnnotator`). With
  `--shortcut` (default on), they also add a desktop shortcut "Start VideoAnnotator": a
  `.desktop` file on Linux, a `.command` file on macOS, a `.lnk` to the `.cmd` on Windows.

**Rationale**: the spec's double-click start, matching uv's own installers that researchers may
already have seen.
