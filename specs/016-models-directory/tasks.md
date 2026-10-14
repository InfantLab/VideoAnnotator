# Tasks: One Models Directory

- [X] T001 [US1][US2] `models_dir.py` resolver and `configure_model_caches()`, called first in `__init__.py`
- [X] T002 [US1] Whisper `cache_dir` default and YOLO path through the resolver
- [X] T003 [US3] Readiness: Whisper weights in the models directory
- [X] T004 [US1] `diagnose models` (`diagnostics/models.py`, CLI)
- [X] T005 [US4] Server start: one-time legacy-location notice
- [X] T006 [US2] Dev container one variable; Dockerfiles ENV; docker-compose volume
- [X] T007 Tests: `tests/unit/test_models_dir.py`; readiness test updated
- [X] T008 SC-001/SC-004: every pipeline from another working directory, models dir unchanged in size, outputs identical (OpenFace fix found here)
- [X] T009 Docs (INSTALLATION "Where model weights are stored"), CHANGELOG, roadmap
- [X] T010 Full suite, lint, mypy, pre-commit; commit; CI
