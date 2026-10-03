# Implementation Plan: One Models Directory

**Branch**: `1.6-dev` | **Date**: 2026-10-01 | **Spec**: [spec.md](spec.md)

## Design

- `src/videoannotator/models_dir.py` (stdlib only): `models_dir()` (env or per-user data dir,
  resolved), `source_dir()`, `configure_model_caches()` (sets `HF_HUB_CACHE`, `TORCH_HOME`,
  `PYANNOTE_CACHE`, `DEEPFACE_HOME` unless set; skips `HF_HUB_CACHE` when the user set `HF_HOME`;
  never sets `HF_HOME`), `resolve_yolo_model()`, `directory_size()`, `legacy_locations()`.
- Called from `videoannotator/__init__.py` before anything imports a model library.
- Whisper's `cache_dir` default and YOLO's model path come from it; provenance keeps the
  configured YOLO name.
- Readiness looks for Whisper weights in the models directory (others follow the env variables).
- `diagnostics/models.py` + `videoannotator diagnose models`; server start logs legacy locations
  once when the models directory is empty.
- Dev container: one variable, same layout. Dockerfiles: `ENV VIDEOANNOTATOR_MODELS_DIR=/app/models`;
  docker-compose: named volume `models`.
- Found and fixed: OpenFace's RetinaFace loaded `./weights/mobilenetV1X0.25_pretrain.tar` (cwd);
  `cfg_mnet["pretrain"] = False` (the full checkpoint overwrites it; outputs verified).

## Constitution Check

| Principle | Assessment |
|---|---|
| I. Local-first | Unchanged; weights are local, just in one place. **Pass** |
| III. Reproducibility | Same weights, outputs verified identical / within noise. **Pass** |
| V. Backward compatibility | User-set library variables and explicit paths respected; old configs' `models/yolo/...` resolves; upgraded installs re-download once, announced. **Pass** |
