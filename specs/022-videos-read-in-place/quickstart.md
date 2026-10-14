# Quickstart: Checking Videos and Results Where You Expect Them

A walkthrough that exercises each user story end to end. Use it after implementation, and as the
rc1 walkthrough's input and output section. It assumes a plain local install, with the first user's
key (admin by default) saved in the viewer.

## Setup

```bash
mkdir -p ~/Studies/Demo/site_a
cp viewer/demo-assets/*.mp4 ~/Studies/Demo/                    # 1 clip
cp storage/jobs/*/*.mp4 ~/Studies/Demo/site_a/ 2>/dev/null     # a few more, if present
videoannotator server                                          # prints the results folder
```

Open `http://127.0.0.1:18011/viewer/`.

## Story 1: no copies

1. New job. Step 1 opens on **My folders**. There is no upload tab, only the "Videos on another
   computer? Upload them" link.
2. Open `Studies/Demo`, tick "Include subfolders", Select all, choose `scene_detection`, and submit.
   The run page opens within seconds.
3. Check that no video was copied. This should print nothing:
   `find ~/.local/share/videoannotator/jobs ~/VideoAnnotator -name '*.mp4'`
4. Close the tab mid-run, reopen the run later, and confirm it finished.

## Story 2: a subset

1. New job, My folders, `Studies/Demo/site_a`. Tick 2 videos, then submit.
2. The run has exactly 2 videos.

## Story 6: findable results

1. `ls ~/VideoAnnotator`: one folder per run, `<name> (<date>)`.
2. Inside each run folder: `run.json`, plus one folder per video named after the video, holding
   `<video>_scene_detection.json` and its provenance. No `.mp4` files.
3. Run a second run with the same name. It gets `… (<date> 2)`, and nothing is overwritten.

## Story 7: from the viewer

1. The run page shows the results folder location, an "Open folder" button that opens the file
   manager, and a copy button.
2. "Download results" gives one zip in the same layout, with no videos.
3. A job page's download has no video unless "include video" is ticked.

## Story 4: missing videos

1. Move `~/Studies/Demo/site_a` to `~/Studies/Demo/site_b`.
2. Open an old job from site_a. Its results show, and the player says "Video not found at …/site_a/…".
3. "Run again" on the run lists the missing videos before starting, and offers to run the rest.

## Story 5: datasets

1. In step 1, save the 2-video selection from Story 2 as a dataset, "Pair".
2. New job, Saved datasets, Pair, Use. It goes straight to pipelines, with exactly those 2 videos.

## Story 3: Docker

```bash
VIDEOS_DIR=~/Studies RESULTS_DIR=~/VideoAnnotator docker compose up videoannotator-prod
```

1. From the host's browser, repeat Story 1. My folders shows the Studies folder (displayed as
   `~/Studies`).
2. Results appear in `~/VideoAnnotator` on the host.
3. The run page shows the host path with a copy button, and no "Open folder" button.
4. From another machine, the server is unreachable: the port is published on loopback only.
5. Start without `VIDEOS_DIR`. Step 1 explains how to set it, and upload works.

## Story 8: changing the results folder

1. Restart with `VIDEOANNOTATOR_RESULTS_DIR=~/elsewhere`, and run one video. Its results are in
   `~/elsewhere`.
2. Earlier runs still open from `~/VideoAnnotator`.

## Upload route (unchanged, results only)

On the "Upload them" link, upload one file and run it. Its results are in `~/VideoAnnotator/<run>/`.
The uploaded copy is in the internal storage folder, not in the results folder.
