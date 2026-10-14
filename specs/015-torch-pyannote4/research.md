# Research: torch 2.11, pyannote.audio 4 and CUDA 12.6

Trial worktree `torch214-trial`, 2026-10-01, CPython 3.13.15, RTX 4060 Laptop GPU.

## R1. Which torch

- torch 2.14 resolves, but **torchaudio's last release is 2.11** (PyPI and PyTorch's indexes; the
  project is discontinued) and its compiled extension fails to load on torch 2.14
  (`OSError: Could not load this library: .../_torchaudio.abi3.so`). torchaudio 2.11 declares no
  torch requirement, so the resolver doesn't catch it.
- pyannote.audio 4.0.7 requires `torchaudio>=2.8` and imports it in `core/io.py`, the embedding and
  segmentation models: a hard dependency.
- **Decision**: torch `>=2.11,<2.12`, torchvision `>=0.26,<0.27`, torchaudio `>=2.11,<2.12`, all from
  the CUDA 12.6 index on Linux. Reason recorded beside the constraint.

## R2. torchcodec

- pyannote.audio 4 decodes files with torchcodec. torchcodec is compiled per torch release; the
  resolver picked 0.17 (built for torch 2.14), which fails to load on 2.11. 0.11.x was released
  with torch 2.11 (2026-03-24 vs 2026-03-23). **Decision**: `torchcodec>=0.11,<0.12`.
- Independently, our diarization pipeline passes the waveform in memory
  (`{"waveform", "sample_rate"}`) after decoding with the `ffmpeg` CLI, so it never needs torchcodec's
  decoder or FFmpeg's shared libraries (absent from static builds such as Chocolatey's on Windows).

## R3. pyannote.audio 4 API

- `from_pretrained(..., token=...)` replaces `use_auth_token`.
- A pipeline call returns `DiarizeOutput` (`.speaker_diarization`, `.exclusive_speaker_diarization`)
  instead of an `Annotation`; we unwrap it when the result has no `itertracks`.
- **Gating**: `SpeakerDiarization.__init__` defaults `plda` to
  `pyannote/speaker-diarization-community-1/plda`, and the 3.1 config doesn't override it, so loading
  3.1 under pyannote 4 downloads from the community-1 repo: **its licence must be accepted** (403
  otherwise). Added to the pipeline's `requires_setup`.
- pyannote.audio 4 downloads into the Hugging Face hub cache (pyannote 3 used `PYANNOTE_CACHE`);
  readiness already checks both.

## R4. Telemetry

pyannote.audio 4.0.7 ships `telemetry/config.yaml` with `metrics_enabled: true` and
`otlp_endpoint: https://otel.pyannote.ai/v1/traces`, sending pipeline name, file duration and
speaker counts. It reads `PYANNOTE_METRICS_ENABLED` and only sets it if absent. **Decision**:
`videoannotator/__init__.py` sets it to `0` unless the user set it (principle I); tested in a
subprocess both ways.

## R5. Results (demo video)

| Pipeline | vs torch 2.6 / pyannote 3 |
|---|---|
| speech_recognition | identical text; log-probs and word confidences differ in the 4th decimal |
| speaker_diarization | identical (4 turns: start, end, speaker) |
| scene_detection | same label; scores within 0.0024 |
| person_tracking | 22 detections; boxes within 0.08 px, keypoints within 0.39 px |
| face_analysis | identical (no faces in this video) |
| face_openface3_embedding | within its known sensitivity (a 1-px detector box shift moves a face's AU/gaze by up to ~2.4; spec 013 R6) |

Test suite in the trial: 1295 passed after updating three tests (two expected `use_auth_token`, one
read the machine's real hub cache).
