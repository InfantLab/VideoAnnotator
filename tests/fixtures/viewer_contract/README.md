# Viewer contract fixtures

Real VideoAnnotator output files, run through the viewer's own file detection and parsers by
`viewer/src/test/contract/videoannotator-outputs.test.ts` (in CI's `viewer` job). If a change to
either side breaks how the viewer reads VideoAnnotator's output, that test fails.

Provenance (2026-10-01, `1.6-dev`):

- Video: the viewer's public demo clip (`viewer/demo-assets/…Peekaboo_h265.mp4`), run through the API
  with face_analysis, face_openface3_embedding, person_tracking, scene_detection,
  speaker_diarization and speech_recognition; files taken from the job's artifacts zip.
- `demo_clip_face_detections.json` and `demo_clip_vlm_annotation.json` come from another take of the
  same session (the demo clip has no detectable faces, and no VLM server was running); they contain
  boxes, emotion scores and `NO_TOUCH` labels, nothing identifying.
- Participant codes in names and contents are replaced by `demo_clip` / `demo`.

These files are also the pipelines' output baseline: `tests/integration/test_output_baseline.py`
runs the demo clip through a real server and compares every output with them (exact for scene
detection, person tracks, `.rttm` and `.vtt`; within GPU noise for person-tracking scores and
OpenFace 3). Rechecked 2026-10-02 against a fresh run on Python 3.13 / torch 2.11 and against
v1.5.0: unchanged up to that noise.

To regenerate after an intended output change: run the same job, take the files from the job's
storage folder, replace these keeping the `demo_clip_` names (and the participant-code
replacement above), then run `pytest tests/integration/test_output_baseline.py` and
`cd viewer && bun run test:run`. Record what changed in the CHANGELOG.
