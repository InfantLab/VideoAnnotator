# Demo-clip checks (spec 030)

One section per plan step. Every run: the demo clip through a real server
(`VIDEOANNOTATOR_BASELINE_KEEP=… pytest tests/integration/test_output_baseline.py -m real_models`),
compared with `scripts/compare_demo_outputs.py` against the previous step's run on the same machine
(research B1).

**Machine**: dev container on WSL2, NVIDIA GeForce GTX 1060 6GB (compute capability 6.1), about
1.6–1.8 GB of it held by the host. Not the RTX 4060 the committed fixtures came from.

## Step 0: before (2026-10-18, `794ccf8` + check tooling)

Kept in `~/va-030/step0`. Run time 2 min 41 s for six pipelines.

Against the committed fixtures (RTX 4060):

- `speech_recognition.vtt`, `person_tracks.json`: identical.
- `person_tracking.json`, `scene_detection.json`, `openface3_*.json`: differ in `info.version`
  (fixtures say 1.5.0, today 1.6.0rc1) and by GPU noise: largest person-box area difference
  12.5 px² of 120,479 (0.01 %); scene's top score 0.6340 (fixture 0.6352, the same two values the
  roadmap recorded as the run-order effect); one OpenFace 3 face's box 37 px² larger, flipping
  two action units' presence. These are 4 of handover item 4's 5 baseline failures: version
  string plus GPU, not a code change.
- `speaker_diarization.rttm`: **not written**. No Hugging Face token on this machine (the
  variables are set but empty, none saved) and no pyannote model cached, so the pipeline failed
  in `initialize()`. This is the 5th failure. Diarization checks wait for a token (steps 3, 6).
- `face_detections.json` (not in the baseline until now): RetinaFace finds **19 faces in 11
  frames** of the demo clip, every one with emotion, age and gender. The committed fixture is from
  another take (Haar cascade era), so FR-024 applies: face analysis joins the baseline.

## Step 2: faces, age and gender off (2026-10-18)

Kept in `~/va-030/step2`, compared with step 0:

- `face_detections.json`: the same 19 faces in 11 frames; with `age`, `gender`,
  `gender_confidence` and `gender_scores` set aside, identical to step 0 (boxes, scores,
  emotions). Provenance models: `DeepFace Emotion`, `DeepFace detector retinaface` (were also Age
  and Gender).
- Every other file identical to step 0.
- Empty `DEEPFACE_HOME`, default settings, 2 s of the demo clip: only `retinaface.h5` and
  `facial_expression_model_weights.h5` downloaded (125 MB; was 1,205 MB with age and gender).
  SC-006 met.
- Two runs gave identical face output, so `face_analysis` joins the baseline with person
  tracking's tolerance (FR-024).

## Step 3: speaker turns, community-1 (2026-10-18) — check not yet run

Code and metadata switched. The demo-clip run waits for a Hugging Face token whose account has
accepted community-1's licence (none on this machine; see step 0). The benchmark figures were
checked against pyannote-audio's public README (the same table as the gated model card,
"last updated in 2025-09"). Model repository: 10 files, 33.7 MB (Hub API).
