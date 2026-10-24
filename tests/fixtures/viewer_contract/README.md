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

Re-baselined 2026-10-04: `scene_detection.json` and both OpenFace 3 files, after each pipeline
started running with the same torch settings (`utils/torch_settings.py`). OpenFace 3 used to turn
on `cudnn.benchmark` for the rest of the process, so its results, and scene detection's after it,
varied between runs and with pipeline order. They now reproduce exactly on the same machine:
scene's top score moved from 0.6340 to 0.6352; OpenFace values moved by a median of 0.001, and by
up to 2.8 for the one face whose box shifted by a pixel. Every other file is unchanged.

Re-captured 2026-10-04 again with provenance (spec 017): every file from the demo clip run now
records what made it: a top-level `provenance` key in JSON, a `NOTE videoannotator-provenance`
block in the `.vtt`, and `demo_clip_speaker_diarization.rttm.provenance.json` beside the `.rttm`.
`demo_clip_face_detections.json` and `demo_clip_vlm_annotation.json` (another take, see above)
predate it and carry only COCO's `info.version`. Person tracking's coordinates moved by about
0.005 px from the previous capture: that one was made with OpenFace's leaked cuDNN benchmark
mode still on when YOLO ran (within the baseline test's tolerance, so it went unnoticed).

Re-baselined 2026-10-18 (spec 030, new default models) on a GTX 1060 6GB, from the demo clip run
kept as step 5b in `specs/030-model-defaults/checks.md`: `face_detections.json` is now the demo
clip's own (RetinaFace finds 19 faces in 11 frames; age and gender are off by default) and joins the
baseline; `person_*` come from yolo26n-pose; `speech_recognition.vtt` from Whisper large-v3-turbo on
faster-whisper. Scene and OpenFace 3 files changed only by GPU (scene's top score 0.6340 here,
0.6352 on the RTX 4060) and `info.version`. `demo_clip_speaker_diarization.rttm` comes from
community-1 (same day, once a token was saved): 3 turns from one speaker, where 3.1 split the
mother's last turn between two speakers (checked by listening). The tolerances were kept: two runs on
the GTX 1060 gave identical files for every pipeline but person tracking (within 0.00004 %).

`legacy/` keeps the previous set, without provenance, so both contract tests prove that older
outputs still open, parse the same, and are labelled "version not recorded".

To regenerate after an intended output change: run the same job, take the files from the job's
storage folder, replace these keeping the `demo_clip_` names (and the participant-code
replacement above), then run `pytest tests/integration/test_output_baseline.py` and
`cd viewer && bun run test:run`. Record what changed in the CHANGELOG.

`ask_ai/demo_clip_vlm_annotation_stopped.json` (spec 027): the first three annotations of
`demo_clip_vlm_annotation.json` as a run stopped after 4 of 7 samples writes them, with the
additive fields: per annotation `frame_times` and `status` (one `error`), top-level `vlm_progress`
and `vlm_skipped`. Readers must see the same annotations as without those fields.

Added 2026-10-24 (spec 033): `ask_ai/demo_clip_vlm_annotation_shaped.json` (a question with an
answer shape: `fields`, `fit`, `fit_problem`, `vlm_answer_shape`) and
`ask_ai/demo_clip_vlm_annotation_clip.json` (`sampling_mode: clip` with `clip_start_sec`,
`clip_end_sec`, `clip_fps`), made by the real pipeline on the demo clip with fake model servers
(Ollama replies scripted; the OpenAI-compatible one from `tests/fixtures/openai_compatible/vllm.json`):
the structure is the pipeline's, the answers are not a model's.
