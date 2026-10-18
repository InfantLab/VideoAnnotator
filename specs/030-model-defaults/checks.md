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

## Step 4a: ultralytics 8.3.183 → 8.4.182, still yolo11n-pose (2026-10-18)

Kept in `~/va-030/step4a`, compared with step 2:

- `person_tracking.json`: within tolerance; the largest difference is a box area of 108,749.994
  vs 108,750.035 px² (0.00004 %). `person_tracks.json` identical.
- Every other file identical.
- 8.4 moved Ultralytics' analytics object from `ultralytics.hub.utils` (gone) to
  `ultralytics.utils.events`. The old import would have made person tracking report itself as
  not installed; the switch-off now targets the new module and the no-telemetry test follows it.
  New dependencies: `ultralytics-platform` (imported only if `ultralytics.Platform` is used; we
  don't), `polars`, `nvidia-ml-py`; `py-cpuinfo` dropped.

## Step 4b: yolo26n-pose (2026-10-18)

Kept in `~/va-030/step4b`, compared with step 4a. First run downloaded `yolo26n-pose.pt`
(7,878,574 bytes) to `<models>/yolo/`.

- `person_tracking.json`: the same 22 detections in 11 frames, the same 2 tracks with the same
  person labels, 17 keypoints each. Boxes: IoU with yolo11n's median 0.923, minimum 0.776.
  Keypoints moved a median 19.3 px (largest 99 px). Mean detection score 0.648 (was 0.716).
  Visible keypoints 112 (was 116). `person_tracks.json` identical.
- Every other file identical.
- `iou_threshold` (research P2): 0.7, 0.3 and 0.05 give identical output on the demo clip for
  both yolo26n-pose and yolo11n-pose, which says nothing (two people who never overlap). The
  source does: in Ultralytics 8.4 an end-to-end model's post-processing
  (`utils/nms.py`, the `end2end` branch) filters by confidence only and never reads the IoU
  threshold. Its description now says it has no effect with YOLO26.

## Step 5a: openai-whisper → faster-whisper 1.2.1 (CTranslate2 4.8.3), still `base` (2026-10-18)

Kept in `~/va-030/step5a`, compared with step 4b:

- `speech_recognition.vtt`: **identical** (every cue's text and timing); only the provenance block
  differs: `Systran/faster-whisper-base`, revision `ebe41f70d5b6`, quantization `float32`.
- Every other file identical.
- GPU libraries (research S3): CTranslate2 found cuBLAS and cuDNN from torch's NVIDIA wheels with
  no help; no preload needed. This GPU (compute capability 6.1) supports `float32`, `int8` and
  `int8_float32`, not `float16`, so `compute_type: default` ran as `float32` (CTranslate2 says so
  in a warning, and provenance records it).
- Pipeline time in the job: 3.45 s (openai-whisper 2.08 s), including the model load and the
  one-second warm-up transcription; transcribing the clip itself took 0.3 s.
- openai-whisper and tiktoken uninstalled; nothing imports `whisper` any more.

## Step 5b: Whisper `large-v3-turbo` (2026-10-18)

Kept in `~/va-030/step5b`, compared with step 5a. First load downloaded
`mobiuslabsgmbh/faster-whisper-large-v3-turbo` (Hugging Face serves it from
`dropbox-dash/faster-whisper-large-v3-turbo`; the cache folder keeps the requested name, so the
readiness check's weights entry uses it), revision `0a363e9161cb`, 1,622 MB, in 110 s including
the download.

| | step 5a (`base`) | step 5b (`large-v3-turbo`) |
|---|---|---|
| 1 | 0.000–1.320 Ready, baby girl? | 0.000–1.340 Ready, baby girl? |
| 2 | 2.000–3.160 Good morning, girl. | 1.860–3.240 Good mommy girl. |
| 3 | 4.440–6.040 Take a move! | 4.360–5.660 Peek-a-boo! |
| 4 | 8.960–10.260 Go to smile, huh? | 8.760–10.260 Gotta smile, huh? |

- 4 cues either way, at the same moments (starts within 0.2 s); 3 of 4 reworded. The clip is a
  peekaboo game ("…Peekaboo_h265.mp4"): turbo hears "Peek-a-boo!" where base heard "Take a
  move!". Which of the other two is right needs a listener.
- Every other file identical.
- GPU memory: about 3 GB (float32 on this GPU; float16 elsewhere would be about half). Pipeline
  time in the job 7.7 s (5a 3.5 s), including load and warm-up; transcribing the clip 2.4 s.
- Translate rule: `{"task": "translate"}` with the default fails at initialize naming large-v3 or
  medium (unit tests); with `whisper_model: large-v3` it loads.
