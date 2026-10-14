# Quickstart: Checking a Container That Feels Local

A walkthrough for each user story. Run it once with Docker Desktop and once with Podman Desktop on
Windows (the maintainer's machine), and on Linux. macOS: a pilot lab runs it (community-tested).
Start with no `start.conf` (delete it, and the `videoannotator` container, to repeat).

Setup: a folder of a few videos (e.g. `Studies\Demo`, from `viewer/demo-assets`), and on another
drive or folder a second one (`D:\Pilot`).

## Story 1: first start

1. Install the launcher (one line from the install guide). A "Start VideoAnnotator" shortcut appears.
2. Double-click it. It finds your engine and asks "Which folder are your videos in?" (a folder
   picker opening in Videos or Documents).
3. Choose `Studies`. The confirmation names `C:\Users\<you>\Studies`, says read but never change, and
   names `C:\Users\<you>\VideoAnnotator` for results. Confirm.
4. First time: the download. Then the browser opens on the viewer, already connected (no key prompt).
5. New job: My folders shows **Studies** with your own path. Nothing from inside the container
   (no `/root`, `/app`, `.cache`).
6. Run one video with Scene Detection. Its results are in `C:\Users\<you>\VideoAnnotator\<run>\...`,
   and you can delete them in Explorer (Linux: `ls -l` shows you as owner).

## Story 2: again, with no questions

1. Close everything, start VideoAnnotator again: no question; it says which folder it can read.
2. Install a pipeline group from the viewer's Pipelines page (e.g. Person Tracking). Then `videoannotator-start share`
   a second folder (this recreates the container): the card shows "Restoring…" briefly, then Person
   Tracking is ready again, without downloading it again. A job submitted meanwhile waits, then runs.
3. Run `videoannotator-start update` (or install a newer launcher): same folders, jobs, results,
   model weights and installed pipelines (restored for the new version).
4. Rename `D:\Pilot` after sharing it (Story 3), start again: it says it couldn't find it and starts.

## Story 3: share another folder, stop sharing one

1. `videoannotator-start share` → pick `D:\Pilot`. It restarts and says so. My folders lists both.
2. Start a long job, then `videoannotator-start share` again: it says a video is being processed and
   offers to wait.
3. Settings → Shared folders: both listed, read-only, and the results folder. **Stop sharing**
   Studies: it says "stops when VideoAnnotator next starts". Restart: Studies is gone from My folders.
4. Open an old job from Studies: "Studies isn't shared with VideoAnnotator any more" (not "moved or
   deleted").

## Story 4: never the inside of the container

1. `videoannotator-start unshare` every folder, start: My folders says "VideoAnnotator can only see
   folders you share with it" and how; Upload works.
2. Compose route: `docker compose --profile prod up videoannotator-prod` without `VIDEOS_DIR`: the
   same message (compose wording), no container folders.

## Story 5: plain words

1. Quit Docker Desktop, start: "Docker Desktop isn't running…".
2. With Podman: stop the machine (`podman machine stop`), start: it starts the machine itself.
3. Occupy port 18011 (another server), start: the port message.
4. Choose your home folder when sharing: the broad-share warning, defaulting to No.

## Story 6: Docker or Podman

1. Both installed and running: it uses the one used last, and says which.
2. Only Podman: it uses Podman with no extra step. Results belong to you on Linux.

## Walkthrough log

### 2026-10-08 — Windows, Docker Desktop (T084, in progress)

- First launch after a clean reset reported "Something else is using port 18011". It didn't recur:
  the port was free moments later and the image bound to it. Likely the port was still held just
  after `docker rm -f`, or a native `videoannotator server` (also 18011) was running. The hint
  `videoannotator-start --port 18012` can't be run as printed from a repo checkout, since the
  launcher is only on PATH once installed.
- **Decision (supersedes spec 022's "upload is a link"):** Upload videos is now a tab beside My
  folders and Saved datasets, not the footnote "Videos on another computer? Upload them". With the
  launcher, the server is on the same computer, so upload's real job is videos in a folder that
  hasn't been shared, which is one of the main ways to use the app, not an edge case.

### 2026-10-14 — Windows, Docker Desktop (T084, continued)

Story 1 largely works. Fixed from this walk:

- "Share this folder? [Y/n]" came straight after "Results go to:", so it read as asking about the
  results folder. The question now follows the videos folder and reads "Share this folder with
  VideoAnnotator?"; the results folder is in the summary line after start.
- "Your jobs and models will be kept." showed on what felt like a first start (Docker keeps named
  volumes when containers and images are removed). Now: "Found your earlier VideoAnnotator jobs
  and models. They'll be kept."
- After installing a pipeline, "Restart the server to activate it" gave no way to do so from the
  card, and the launcher had already exited. The card now has a "Restart now" button (sharing the
  banner's restart), the wording says VideoAnnotator rather than "server", and the launcher gains
  `videoannotator-start restart`, which the fallback hint names when the launcher runs things.
- Ollama on the host was unreachable: the launcher didn't set `OLLAMA_BASE_URL`, and the message
  explained container networking. The launcher now sets it (Docker: `host.docker.internal`, with
  `host-gateway` on Linux Docker Engine; Podman: `host.containers.internal`) and the message says
  to start the Ollama app. Checked under Docker Desktop on Windows: the container reaches a host
  service bound to 127.0.0.1 only (Ollama's default), so no `OLLAMA_HOST` change is needed there.
  Linux needs `OLLAMA_HOST=0.0.0.0` (in INSTALLATION.md, "Using Ollama"). Still to check: Podman
  Desktop, and macOS.
