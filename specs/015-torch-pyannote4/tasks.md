# Tasks: torch 2.11, pyannote.audio 4 and CUDA 12.6

- [X] T001 Trial worktree: lift torch/pyannote constraints, cu126 index; resolve; install all extras
- [X] T002 Find the torch ceiling (torchaudio 2.11) and the matching torchcodec (0.11); set constraints with reasons in `pyproject.toml`
- [X] T003 [US2] Migrate `diarization_pipeline.py`: `token=`, unwrap `DiarizeOutput`, in-memory waveform, missing input returns []
- [X] T004 [US3] Telemetry off by default in `videoannotator/__init__.py`; tests in `tests/unit/test_supported_python.py`
- [X] T005 [US2] community-1 licence in `requires_setup` of `speaker_diarization.yaml` and `audio_processing.yaml`
- [X] T006 Update tests for the new API and hub cache
- [X] T007 [US1][US2] Demo-video comparison (research R5), diarization with community-1 accepted: identical
- [X] T008 Docs: driver floor and torch cap in `docs/installation/INSTALLATION.md`; CHANGELOG; audit note
- [X] T009 Main environment on the new lock; full suite; ruff, mypy, pre-commit; commit; CI on Ubuntu/macOS × 3.12/3.13
