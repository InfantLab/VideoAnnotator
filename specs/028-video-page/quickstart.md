# Quickstart: checking the Video page

A walk-through per user story on the demo clip, run against `bash scripts/dev.sh`
(http://127.0.0.1:19011). Record what you saw in `specs/028-video-page/checks.md`.

## Setup

1. Run the demo clip (`tests/fixtures/` demo, or `videoannotator job submit demo_clip.mp4
   --pipelines person_tracking,face_analysis,audio_processing,scene_detection`) so it has
   results from every installed pipeline.
2. Run `face_analysis` on it a second time with a different detection threshold (for US5).
3. Note the time to first frame and measure the video area at 1440×900 on the old page
   (`/view/<job id>` before this spec) for SC-001 and SC-003.

## US1: Read the timeline

1. Open the job from Jobs (or `/view/<job id>`): the address becomes `/video?path=…&run=…`.
2. Each lane is labelled and names its run; Speakers has a row per speaker in its overlay colour.
3. Zoom to about ten seconds (`+`, or Ctrl+wheel); the range bar shows where you are; play: the
   view follows.
4. Select Speech; `]` jumps to the next utterance; hover shows its text and times.
5. Faces has a count chart with its scale labelled.

## US2: Toggle from the lanes

1. Click "Scenes": it collapses. Click the eye on Faces: the boxes go.
2. Layers → none, then all.
3. Reload, then open another video: the same lanes are collapsed and overlays off.
4. There is no controls card.

## US3: The values beside the video

1. Data tab: play; the highlighted row follows. Scroll the table: following stops; "Follow
   playback" resumes it.
2. Click a row in the second half: the video seeks there.
3. Filter to Speakers: "n of N rows". Search "peekaboo".
4. Export the table: the Data tab's rows for this run match the CSV's (order and values).

## US4: Plays at once

1. With the browser's network panel open, open the page: no `/artifacts` zip, no folder dialog;
   the video request is a range request to `/api/v1/videos/stream`.
2. Open a 2 GB video's page (any large file under a shared folder, run with one fast pipeline):
   playback starts within 3 s (SC-001).
3. Move the video away on the server and reload: lanes and table remain; the missing-video
   message shows.
4. `/viewer`: drop the demo video and its result files: the page works without a server; Notes
   says it needs one.

## US5: Which run

1. Faces names the newest run. From its menu choose the earlier run: lane, boxes and table
   change.
2. "Compare with…" the other: two Faces lanes, one under the other; About lists the threshold
   that differs.

## US6: Notes

1. Play; press N, type "infant looks away", Enter. Shift+N, wait, Shift+N, type a span note.
2. Reload: both there, in the Notes lane (last, styled as yours) and the Notes tab.
3. Type "n" in the search box: no note is created.
4. Export the table: two `note` rows with your name in `author`. Export ELAN and open it in ELAN:
   a "Notes (researcher)" tier.

## US7: Where this video belongs

1. Open a video from a batch (Batch detail): "3 of 48", ‹ › move through the batch with lane
   settings kept.
2. Open the same video from its address with no `from`: the header lists its projects.

## Performance (SC-004)

Open the hour-long synthetic fixture (`tests/fixtures/video_page/long_run/`, 50,000 events):
zoom and pan at full and narrow ranges, and play: no visible stutter (Chrome performance panel:
no long frames over 50 ms while panning).
