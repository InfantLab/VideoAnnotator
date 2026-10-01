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

To regenerate after an output-format change: run the same job, download its artifacts, and replace
these files keeping the `demo_clip_` names.
