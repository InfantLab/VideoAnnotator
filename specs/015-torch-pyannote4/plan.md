# Implementation Plan: torch 2.11, pyannote.audio 4 and CUDA 12.6

**Branch**: `1.6-dev` | **Date**: 2026-10-01 | **Spec**: [spec.md](spec.md) | **Research**: [research.md](research.md)

## Summary

One torch range (2.11) across every torch-using extra, with matching torchvision, torchaudio and
torchcodec from the CUDA 12.6 index; pyannote.audio 4 (single `>=4.0.7,<5` constraint replacing the
five pyannote 3 caps); diarization migrated (`token`, `DiarizeOutput`, in-memory waveform, missing
input handled); telemetry off by default; community-1 licence in setup requirements; docs state the
driver floor.

## Constitution Check

| Principle | Assessment |
|---|---|
| I. Local-first | pyannote telemetry, on upstream by default, is turned off. **Pass** |
| II/III. Contract, reproducibility | Outputs compared on the demo video (R5); differences are numerical, recorded in the CHANGELOG. **Pass** |
| V. Backward compatibility | Same default model and output format; one extra licence to accept, surfaced by the setup checklist. **Pass** |

## Changes

`pyproject.toml`, `uv.lock`; `pipelines/audio_processing/diarization_pipeline.py`;
`videoannotator/__init__.py`; `registry/metadata/{speaker_diarization,audio_processing}.yaml`;
tests (`test_audio_speech_pipeline.py`, `test_audio_individual_components.py`, `test_readiness.py`,
`test_supported_python.py`); `docs/installation/INSTALLATION.md`; CHANGELOG; audit.
