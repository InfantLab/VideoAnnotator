# Long-run fixture (spec 028)

A synthetic hour of results for the Video page's performance tests (SC-004: smooth with an
hour-long video and 50,000 events). Written by `scripts/make_long_run_fixture.py`, seeded, so
every run writes the same bytes:

- `long_run_person_tracking.json`: COCO, 4 tracks sampled at 5 fps, seen in stretches with gaps
  (41,161 detections over 3,600 s);
- `long_run_speech_recognition.vtt`: 2,000 cues;
- `long_run_speaker_diarization.rttm`: 3 speakers, 6,000 turns.

The files (about 15 MB) are not committed: `viewer/src/test/lib/lanePerformance.test.ts` runs the
script when they are missing. To write them by hand: `python3 scripts/make_long_run_fixture.py`.
Nothing in them comes from a real recording.
