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
2. Install a pipeline group from the viewer (e.g. Person Tracking). Then `videoannotator-start share`
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
