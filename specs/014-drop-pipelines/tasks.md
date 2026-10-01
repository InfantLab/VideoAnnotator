# Tasks: Drop Obsolete Pipelines

- [X] T001 [US3] `family_default` and `deprecated` in `PipelineMetadata`; `list(include_deprecated=...)` in `registry/pipeline_registry.py`
- [X] T002 [US2] `REMOVED_PIPELINES`, `removed_pipeline_message`, `deprecation_message`; loader loads deprecated pipelines; family default outranks stability in `registry/pipeline_loader.py`
- [X] T003 [US2] `PipelineRemovedException` (`api/v1/exceptions.py`); raised in `validate_pipeline_selection` and re-raised by `submit_job` (`api/v1/jobs.py`); `JobResponse.warnings`
- [X] T004 [US2] Batch path: removal message in `_unavailable_reason`, deprecation warning logged per pipeline (`batch/job_execution.py`)
- [X] T005 [US1] Metadata: delete `laion_voice.yaml`, `face_laion_clip.yaml`; `audio_processing.yaml` deprecated + family default; `face_analysis.yaml` family default
- [X] T006 [US1] Delete `laion_voice_pipeline.py`, `laion_face_pipeline.py` and their exports; drop LAION entries from `api/v1/debug.py`, `api/readiness.py`
- [X] T007 [US1] `pyproject.toml`: remove `face-laion`, `audio-laion` (and from `all`); re-lock
- [X] T008 [US1] Tests: delete LAION tests; move contract tests' example to `face_openface3_embedding`; new `tests/contract/test_removed_and_deprecated_pipelines.py`
- [X] T009 [US2] Configs: replace the dead `audio_processing:` sections; delete `configs/laion_pipelines.yaml`
- [X] T010 Docs: README, INSTALLATION, GETTING_STARTED, regenerated `pipelines_spec.md`, dev docs; CHANGELOG
- [X] T011 Full suite; demo-video comparison incl. `audio_processing` before/after (SC-003); commit; CI
