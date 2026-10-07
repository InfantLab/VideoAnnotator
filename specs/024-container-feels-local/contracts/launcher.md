# Launcher Contract: `videoannotator-start`

The same behaviour from `videoannotator-start` (POSIX `sh`; Linux, macOS) and
`videoannotator-start.ps1` (PowerShell; Windows, run via `videoannotator-start.cmd`).

## Commands

| Command | What it does |
|---|---|
| `videoannotator-start` | Start (first run: ask for a folder and confirm), or open the browser if already running |
| `videoannotator-start share [PATH]` | Share another folder (picker if no PATH), then restart (asks about running jobs) |
| `videoannotator-start unshare [PATH]` | Stop sharing (list to choose from if no PATH), then restart |
| `videoannotator-start list` | Print shared folders, results folder, engine, image |
| `videoannotator-start stop` | Stop VideoAnnotator |
| `videoannotator-start update` | Update the launcher from the latest release, then the app: the new launcher pulls its own pinned image and restarts (research.md R16). Installed pipelines are restored; weights, jobs and settings are untouched. Already up to date: says so, changes nothing |
| `videoannotator-start logs` | Show the server's recent log |

Options for scripted and CI use: `--share PATH` (repeatable), `--results PATH`, `--engine
docker|podman`, `--yes` (accept confirmations, including broad shares), `--no-browser`,
`--port N`, `--image REF`.

## What it prints (first run, Linux)

```
Starting VideoAnnotator with Podman.

Which folder are your videos in?            (a folder picker opens)

VideoAnnotator will be able to read, but never change:
  /home/ada/Studies   (and everything inside it)
Results go to:
  /home/ada/VideoAnnotator
Share this folder? [Y/n]

Downloading VideoAnnotator (first time only, about 1 GB)...
Starting... ready.
VideoAnnotator can read: /home/ada/Studies. Results: /home/ada/VideoAnnotator.
Opening http://127.0.0.1:18011/viewer in your browser.
```

Later starts print only the last three lines, plus a note for each missing share or applied stop
request.

## The `run` it builds (illustrative; Linux, Podman, one share)

```
podman run -d --name videoannotator \
  -p 127.0.0.1:18011:18011 \
  -v videoannotator-models:/app/models \
  -v videoannotator-database:/app/database \
  -v videoannotator-storage:/app/storage \
  -v videoannotator-cache:/app/cache \
  --mount type=bind,source=/home/ada/Studies,target=/home/ada/Studies,readonly \
  --mount type=bind,source=/home/ada/VideoAnnotator,target=/home/ada/VideoAnnotator \
  --mount type=bind,source=/home/ada/.config/videoannotator/requests,target=/app/launcher/requests \
  -e VIDEOANNOTATOR_INGEST_ROOTS=/home/ada/Studies \
  -e VIDEOANNOTATOR_MISSING_SHARES= \
  -e VIDEOANNOTATOR_RESULTS_DIR=/home/ada/VideoAnnotator \
  -e VIDEOANNOTATOR_HOST_PATHS='/home/ada/Studies=/home/ada/Studies;/home/ada/VideoAnnotator=/home/ada/VideoAnnotator' \
  -e VIDEOANNOTATOR_PUBLISHED_LOCALLY=1 \
  -e VIDEOANNOTATOR_LAUNCHER=1 \
  ghcr.io/infantlab/videoannotator:1.6.0
```

Docker on Linux adds `-e VIDEOANNOTATOR_RESULTS_OWNER=<uid>:<gid>`; GPU flags per research.md R15.
Shares use `--mount`, never `-v`. A host path containing `:`, `;`, `=` or `,` is mounted at
`/host/<n>` instead of its real path (research.md R3). The results mount always comes after the
read-only shares, so a results folder inside a share stays writable (R4).
`VIDEOANNOTATOR_MISSING_SHARES` lists, `;`-separated, the host paths of shares missing at this
start, so Settings can still list them.
Windows maps `C:\Users\ada\Studies` to `/c/Users/ada/Studies` with the host path in `HOST_PATHS`.
Named volumes are the same as `docker-compose.yml`'s (plus the new cache volume, which compose
gains too), so a compose install's jobs, models and installed pipelines carry over (FR-024).

## Messages (each one line, plus the next step)

| Situation | Message |
|---|---|
| No engine | "VideoAnnotator needs Docker Desktop or Podman Desktop. Install one (see <guide>), then run this again." |
| Docker not running | "Docker Desktop isn't running. Start it, wait until it says it's running, then run this again." |
| Docker service stopped (Linux) | "Docker isn't running. Start it with: sudo systemctl start docker, then run this again." |
| No permission (Linux) | "You don't have permission to use Docker yet. Run: sudo usermod -aG docker $USER, log out and back in, then run this again. (Or use Podman, which needs no permission.)" |
| Folder not shared with Docker Desktop (macOS) | "Docker Desktop can't see /opt/data yet. Add it in Docker Desktop's Settings, Resources, File sharing (see <guide>), then run this again." |
| Already running (started twice at once) | (opens the browser on the running one) "VideoAnnotator is already running." |
| Podman machine stopped | (started automatically) "Starting Podman's virtual machine (first time takes a minute)…" |
| Port in use | "Something else is using port 18011. Close it, or run: videoannotator-start --port 18012" |
| Download failed | "Couldn't download VideoAnnotator. Check your internet connection and run this again." |
| Out of memory | "VideoAnnotator ran out of memory. Give Docker/Podman more (see <guide>), then run this again." |
| GPU unusable | "Running without the GPU: <engine> can't use it yet. To enable it, see <guide>." |
| Share missing | "Couldn't find E:\Data (an unplugged drive?), so it isn't shared this time." |
| Share inside results | "That folder is inside your results folder, which VideoAnnotator can already read." |
| Broad share | "This shares everything in your home folder, including documents unrelated to your research. Share it anyway? [y/N]" |
| Running jobs | "2 videos are being processed. [W]ait for them, or [r]estart now (they'll be marked failed and can be retried)?" |

## Exit codes

0 started or already running; 1 a problem with a message above; 2 the researcher cancelled.
