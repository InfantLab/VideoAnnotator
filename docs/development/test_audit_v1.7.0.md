# Test audit for v1.7.0

**Status**: in progress, started 2026-10-19. v1.7.0 Phase 7 ([`roadmap_v1.7.0.md`](roadmap_v1.7.0.md)).
Steps 1 and 2 are mostly done; steps 3 to 5 have findings below but no changes yet.

All numbers are for the run CI uses: `pytest -m "not real_models and not performance"`.

| | Before (2026-10-19) | After this pass |
|---|---:|---:|
| Passed | 2,015 | 2,017 |
| Skipped | 30 | 7 |
| Run time | 100 s | 88 s |
| Line + branch coverage | 75.2% | 76.2% |

The "after" count includes tests that spec 027's implementation added in parallel, so pass counts
can't be compared directly. Skips and coverage can.

## What the audit found

**Three real bugs, hidden by tests that couldn't fail.** Each was in code that a test "covered"
while it caught the error, skipped on it, or never reached the code:

- `storage-cleanup` (CLI) could never run. `find_old_jobs` imported
  `videoannotator.database.session`, which doesn't exist. `test_accepts_override` caught every
  exception, so `storage/cleanup.py` showed 89% coverage while its one query was unreachable.
- `storage-cleanup` could never report success. Its `raise typer.Exit(0)` sat inside
  `except Exception`, which caught it (Click's `Exit` is an exception) and printed
  "Cleanup failed: 0" with exit code 1. No test ran the command.
- Automatic person labeling crashed (`KeyError: 'position_based'`) when a job's config named only
  some labeling sections, because `AutomaticPersonLabeler` used a given config *instead of* its
  defaults. `test_person_tracking_with_size_analysis` should have caught it, but a broken mock
  (fps read as 0, so no frames) made it skip every run.

**Tests that couldn't fail** (59 flagged by an AST scan for tests with no unconditional check, or
that swallow broad exceptions; each was then read by hand):

- 9 "integration" tests gated on `TEST_INTEGRATION`, which nothing sets. Each turned any exception
  into a skip, so even when enabled they could only pass or skip. Real-model coverage of every
  pipeline is `tests/integration/test_output_baseline.py` (marked `real_models`).
- 7 placeholders that only called `pytest.skip` (4 in scene detection, 3 in the error envelope),
  and 1 that was skipped as "not implemented".
- Tests that check something the code doesn't have: `test_embedding_shape_validation` asserted
  only `if hasattr(...)` a method that doesn't exist, and `test_pipeline_module_exists` and
  `test_numpy_availability` checked imports.
- Tests that pass on either outcome: `if status == "ok": assert 200`, `if should_retry: ...`,
  `try: ... except (TypeError, ValueError): pass`, and a cleanup test with the `assert` inside
  `except`.
- `test_person_identity.py`: two tests that print and `return` on every path, including failure.
- `test_api_live_server.py`: two tests needing a `create_test_key.py` that no longer exists and a
  server on port 18011. Both always skipped.
- A stale `importorskip("whisper")` that skipped the VTT-writing test after the move to
  faster-whisper.
- An empty test body (`test_openface3_availability_check`).

Most of the remaining flags are deliberate "does not raise" tests (for example
`test_loopback_addresses_are_accepted`) or tests whose helpers do the asserting, and they stay.

**Duplicates.** `tests/pipelines/test_audio_individual_components.py` repeats
`test_audio_speech_pipeline.py` with weaker assertions (17 identical test bodies).
`tests/integration/test_storage_real.py` is byte-identical to
`tests/unit/storage/test_file_backend.py`. `tests/integration/test_recovery_real.py` overlaps
`tests/unit/batch/test_recovery.py` (4 shared names); it is the more thorough of the two, but not
a copy, so the two should be merged rather than one deleted.

## Done in this pass

- Fixed the three bugs above, each with a test that fails without the fix:
  `tests/unit/cli/test_cli_storage_cleanup.py`, `TestFindOldJobs` against a real temporary
  database, `test_partial_config_keeps_the_other_defaults`, and the repaired size-analysis test.
- Removed the `TEST_INTEGRATION` tests, the placeholders, the live-server tests, the check-less
  logging tests and the empty test.
- Made the conditional tests unconditional (health, retry workflow, storage round trip, live-server
  system health, Whisper device on CPU).
- Replaced the 7.7 s real-DeepFace error test, the slowest in the run, with a mocked one that
  states the behaviour: analysis and fallback both failing gives no faces, not an error.
- Deleted `tests/pipelines/test_audio_individual_components.py` and
  `tests/integration/test_storage_real.py` (duplicates), and
  `tests/pipelines/test_whisper_base_pipeline_stage1.py` and
  `tests/integration/test_person_identity.py` (couldn't fail).
- Re-enabled the DeepFace mock test (it was gated on an unset `TEST_DEEPFACE`, though it needs no
  real model) and the VTT test.

## Waiting

- Merge `test_recovery_real.py` into `tests/unit/batch/test_recovery.py`.
- `tests/README.md` describes a suite of "6 focused files" with a 94% pass rate, and
  `docs/testing/testing_overview.md` still says to set `TEST_INTEGRATION`. Rewrite both from this
  audit.

## Step 3: mocks at our boundary

Little is left to change. Spec 030 already deleted the openai-whisper internals. What remains
patches our own module's name for a library (`face_pipeline.DeepFace`, `person_pipeline.YOLO`),
which is the boundary the roadmap asks for. Exceptions to look at:
`test_size_analysis.py` builds YOLO's `Results.boxes` structure by hand, and 4 tests patch
`cv2.VideoCapture` directly.

## Step 4: markers and default run time

The default run is already fast: 100 s before, 88 s after, with no file over 8.1 s. Nothing slow
is unmarked now that the real-DeepFace test is mocked. `real_models` (22 tests, all in
`test_output_baseline.py` and `test_extras_isolation.py`) stays out of CI.

## Step 5: coverage

The constitution asks for 80%, but CI's gate is `--cov-fail-under=45`. After this pass the suite
is at 76.2%. The coverage number alone hides problems: `storage/cleanup.py` read 89% while broken.

The gaps that matter, by missed lines:

| Area | Module | Covered | Missed lines |
|---|---|---:|---:|
| Job path | `api/background_tasks.py` | 57% | 99 |
| Job path | `worker/job_processor.py` | 68% | 50 |
| Job path | `storage/sqlite_backend.py` | 66% | 106 |
| Job path | `batch/batch_orchestrator.py` | 66% | 72 |
| CLI | `cli.py` | 41% | 699 |
| Pipelines | `person_tracking/person_pipeline.py` | 33% | 145 |
| Pipelines | `scene_detection/scene_pipeline.py` | 35% | 115 |
| Pipelines | `face_analysis/openface3_pipeline.py` | 45% | 233 |
| VLM | `vlm_annotation/ollama_client.py` | 42% | 40 |
| Provenance | `provenance.py` | 89% | 13 |
| Exporters | `exporters/*` | 85 to 100% | 67 in all |

Provenance and the exporters are in good shape. The job path and the CLI are where an untested
change would reach users. Raise CI's gate to the measured figure now, and to 80% when the job path
reaches it.

## Appendix: per-file inventory (before this pass)

Time is the sum of each file's test durations in the baseline run. "Flagged" counts the scan's
hits, before reading each by hand; most flags in files not named above are deliberate
"does not raise" tests.

| File | Tests | Time (s) | Skipped | Flagged | What it covers (its docstring) |
|---|---:|---:|---:|---:|---|
| `api/test_api_server.py` | 24 | 3.7 |  |  | Tests for VideoAnnotator API Server v1.2.0 with Database Integration |
| `api/test_ask_api.py` | 33 | 4.1 |  |  | Ask AI's API (spec 027) with a fake model server |
| `api/test_batch_endpoints.py` | 39 | 2.9 |  |  | API tests for spec 008: submission batch tagging, aggregate batch status, |
| `api/test_dataset_preset_endpoints.py` | 21 | 8.1 |  |  | API tests for the saved-dataset and saved-preset endpoints (spec 007) |
| `api/test_dataset_stored_run.py` | 6 | 1.1 |  |  | Running a saved dataset from the copies its uploaded jobs keep: no folder to |
| `api/test_export_routes.py` | 17 | 1.0 |  |  | Results out through the API: the same exports as the CLI (spec 025) |
| `api/test_extras_restore.py` | 8 | 0.9 |  |  | Installed pipelines survive the container being recreated (spec 024, R7) |
| `api/test_health.py` | 22 | 0.2 |  | 1 | Tests for enhanced health check endpoint |
| `api/test_huggingface_token_api.py` | 11 | 1.0 |  |  | `/api/v1/system/huggingface-token` (spec 031, contracts/api.md) |
| `api/test_ingest_access.py` | 15 | 0.1 |  |  | `GET /api/v1/ingest/access`: what this caller may do with local videos (spec 022) |
| `api/test_ingest_endpoints.py` | 39 | 1.8 |  | 1 | API tests for server-side folder ingest |
| `api/test_pipeline_extras_endpoints.py` | 17 | 5.0 |  |  | API tests for the pipeline extras install endpoints (specs/005-pipeline-extras-install) |
| `api/test_pipeline_guide_api.py` | 6 | 0.3 |  |  | The pipeline guide on the API (spec 026, contracts/api.md) |
| `api/test_pipeline_readiness_endpoints.py` | 4 | 0.1 |  |  | API tests for pipeline readiness and the extras-group listing |
| `api/test_prompts.py` | 5 | 0.3 |  |  | The prompt library (spec 020): every prompt that runs, kept once, findable |
| `api/test_rerun.py` | 18 | 1.4 |  |  | Run it again (spec 019): a new, linked job; the original untouched |
| `api/test_result_files.py` | 7 | 0.4 |  |  | GET /jobs/{id}/results advertises download URLs that work |
| `api/test_results_folder.py` | 18 | 2.1 |  |  | Every run's results in one visible folder, by run then video (spec 022, US6/US8) |
| `api/test_results_open.py` | 9 | 0.1 |  |  | `POST /api/v1/results/open`: show a results folder on this computer (spec 022) |
| `api/test_run_download_and_delete.py` | 8 | 0.4 |  |  | A run's results as one download, and deleting a whole run (spec 022, US7) |
| `api/test_shares.py` | 11 | 0.1 |  |  | Shared folders in Settings, and Stop sharing (spec 024, research R10) |
| `api/test_system_restart.py` | 12 | 0.1 |  |  | API tests for server boot identity and self-restart (specs/011-pipeline-readiness) |
| `api/test_video_storage_fix.py` | 4 | 0.9 |  |  | Test that video files are properly stored in persistent storage |
| `api/test_videos_api.py` | 8 | 0.2 |  |  | Videos for Ask AI: upload once, play with a short-lived token (spec 027 R12) |
| `api/test_viewer_origin.py` | 6 | 0.0 |  |  | The viewer only works from 127.0.0.1: it pins its API URL there and the |
| `api/test_viewer_static.py` | 4 | 0.0 |  |  | The bundled viewer is a single-page app: unknown routes get the app shell, |
| `api/test_vlm_endpoints.py` | 13 | 0.3 |  |  | API tests for spec 009: VLM prompt preview and model discovery |
| `api/test_vlm_preview_nonblocking.py` | 1 | 0.1 |  | 1 | A preview waiting on the model doesn't stop the server answering others |
| `contract/test_migration_message_contract.py` | 10 | 0.0 |  |  | Contract test: the migration-message variant for pipelines demoted out of |
| `contract/test_no_hardcoded_tokens.py` | 9 | 1.0 |  |  | No hard-coded tokens; artifacts and debug endpoints follow AUTH_REQUIRED |
| `contract/test_output_readers.py` | 6 | 0.0 |  |  | Stamped outputs still open in the standard readers for their formats (spec 017 FR-007) |
| `contract/test_pipeline_availability_contract.py` | 4 | 0.2 |  |  | Contract test: unavailable pipelines are omitted from GET /api/v1/pipelines |
| `contract/test_removed_and_deprecated_pipelines.py` | 10 | 0.4 |  |  | Removed and deprecated pipelines (spec 014) |
| `contract/test_unavailable_pipeline_error.py` | 3 | 0.1 |  |  | Contract test: unavailable-pipeline error shape on every user-facing |
| `contract/test_viewer_contract.py` | 4 | 0.0 |  |  | Contract test: VideoAnnotator's real output vs. Video Annotation Viewer's parsers |
| `integration/test_api_integration.py` | 11 | 4.2 |  |  | Integration tests for VideoAnnotator API Server v1.2.0 |
| `integration/test_api_live_server.py` | 3 | 0.1 | 2 | 1 | Live API server integration tests |
| `integration/test_ask_run_stop.py` | 1 | 0.2 |  |  | A run from Ask AI: answers as they come, Stop keeps them (spec 027 US5) |
| `integration/test_background_processing.py` | 2 | 0.3 |  | 1 | Integration tests for background job processing system |
| `integration/test_batch_orchestration.py` | 8 | 0.7 |  | 1 | Integration tests for batch processing components |
| `integration/test_batch_orchestrator_real.py` | 21 | 0.1 |  |  | Unit tests for BatchOrchestrator - testing actual implementation |
| `integration/test_batch_storage.py` | 21 | 0.1 | 1 | 1 | Unit tests for batch processing storage backend functionality |
| `integration/test_concurrency.py` | 9 | 0.0 |  |  | Tests for concurrent job processing limits |
| `integration/test_config_validation.py` | 14 | 0.1 |  |  | Integration tests for configuration validation API endpoints |
| `integration/test_enhanced_logging.py` | 11 | 0.0 |  | 2 | Integration tests for enhanced logging functionality in v1.2.0 |
| `integration/test_error_envelope.py` | 9 | 0.3 | 3 | 3 | Integration tests for ErrorEnvelope format across all API endpoints |
| `integration/test_extras_isolation.py` | 2 | 0.0 | 2 |  | Integration test: extras groups actually isolate their dependencies |
| `integration/test_face_openface3_registration.py` | 1 | 0.0 |  |  | Test that face_openface3_embedding pipeline is available in both API and execution engines |
| `integration/test_job_cancellation.py` | 9 | 0.4 |  |  | Integration tests for job cancellation API endpoint |
| `integration/test_job_submission_validation.py` | 14 | 0.6 |  |  | Integration tests for configuration validation in job submission |
| `integration/test_model_cache_reuse.py` | 2 | 0.1 |  |  | Integration test: the extras-aware registry/loader never touches the |
| `integration/test_output_baseline.py` | 6 | 0.0 |  | 3 | Integration test: pipeline outputs on the demo clip match the committed baseline |
| `integration/test_person_identity.py` | 2 | 0.0 |  | 2 | Test Phase 2: PersonID Integration with Face Analysis Pipelines |
| `integration/test_progress_tracker_real.py` | 14 | 0.0 |  |  | Unit tests for ProgressTracker - testing actual implementation |
| `integration/test_recovery_real.py` | 20 | 0.0 |  | 1 | Unit tests for FailureRecovery covering real implementation paths |
| `integration/test_simple_workflows.py` | 6 | 0.0 |  |  | Simple integration tests for batch processing components |
| `integration/test_storage_paths.py` | 10 | 0.6 |  |  | Integration tests for storage path functionality (v1.3.0) |
| `integration/test_storage_provider_integration.py` | 2 | 0.2 |  |  | Integration tests for StorageProvider with API and Worker |
| `integration/test_storage_real.py` | 11 | 0.0 |  |  | Unit tests for FileStorageBackend - testing actual implementation |
| `pipelines/test_audio_individual_components.py` | 18 | 0.1 | 3 | 3 | Unit tests for Diarization and Speech Recognition Pipelines |
| `pipelines/test_audio_pipeline.py` | 9 | 1.9 | 1 | 1 | Unit tests for Modular Audio Processing Pipeline |
| `pipelines/test_audio_speech_pipeline.py` | 18 | 0.1 | 3 | 3 | Unit tests for diarization and speech recognition pipelines |
| `pipelines/test_face_analysis.py` | 28 | 7.8 | 3 | 3 | Modern Face Analysis Pipeline Tests |
| `pipelines/test_openface3.py` | 21 | 0.1 | 2 | 3 | OpenFace 3.0 Pipeline Tests |
| `pipelines/test_person_tracking.py` | 9 | 0.0 | 1 | 1 | Modern Person Tracking Pipeline Tests |
| `pipelines/test_scene_detection.py` | 9 | 0.0 | 5 | 5 | Modern Scene Detection Pipeline Tests |
| `pipelines/test_whisper_base_pipeline.py` | 6 | 0.7 |  | 4 | Speech pipeline device choice: CUDA, CPU, and the fall back without a driver |
| `pipelines/test_whisper_base_pipeline_stage1.py` | 5 | 0.0 |  | 1 | Stage 1: Basic Whisper Pipeline Test - Simple Structure Test |
| `unit/api/test_auth_me.py` | 4 | 0.1 |  |  | Tests for GET /api/v1/auth/me |
| `unit/api/test_error_envelope.py` | 1 | 0.0 |  |  |  |
| `unit/api/test_errors.py` | 26 | 0.1 |  |  | Unit tests for v1.3.0 error envelope system |
| `unit/api/test_extras_install.py` | 40 | 4.9 |  | 3 | Unit tests for api/extras_install.py: command selection, dedup tracking, |
| `unit/api/test_job_processor_partial_failure.py` | 2 | 0.0 |  |  |  |
| `unit/api/test_pipelines_api_taxonomy.py` | 2 | 0.1 |  |  |  |
| `unit/api/test_readiness.py` | 39 | 0.2 |  |  | Unit tests for api/readiness.py (specs/011-pipeline-readiness, contract §1-2) |
| `unit/auth/test_require_admin.py` | 4 | 0.0 |  |  | Unit tests for the require_admin auth dependency (api/middleware/auth.py) |
| `unit/auth/test_token_manager.py` | 30 | 0.1 |  | 1 | Unit tests for auth/token_manager.py |
| `unit/batch/test_batch_summary.py` | 7 | 0.0 |  |  | Unit tests for the stateless batch-summary aggregation (spec 008) |
| `unit/batch/test_batch_validation.py` | 24 | 0.1 |  |  | Test file to validate the actual batch processing APIs |
| `unit/batch/test_job_execution.py` | 7 | 0.1 |  |  | Unit tests for the spec-006 shared job-execution path |
| `unit/batch/test_job_execution_output_dir.py` | 5 | 0.0 |  |  | Where the runner writes (spec 022): the job's results folder, else its own folder |
| `unit/batch/test_job_provenance.py` | 3 | 0.0 |  |  | The job runner records provenance in every file a pipeline writes (spec 017) |
| `unit/batch/test_orchestrator.py` | 21 | 0.1 |  |  | Unit tests for the BatchOrchestrator class |
| `unit/batch/test_progress_tracker.py` | 9 | 0.0 |  | 1 | Unit tests for batch progress tracking functionality |
| `unit/batch/test_recovery.py` | 14 | 0.0 |  |  | Unit tests for batch failure recovery functionality |
| `unit/batch/test_types.py` | 24 | 0.1 |  |  | Unit tests for batch processing types and data structures |
| `unit/cli/test_cli_dataset.py` | 5 | 0.1 |  |  | `videoannotator dataset ...`: the Datasets page's actions from a terminal (spec 018) |
| `unit/cli/test_cli_pipelines_describe.py` | 7 | 0.1 |  |  | `videoannotator pipelines describe` (spec 026, contracts/cli.md) |
| `unit/cli/test_cli_process.py` | 4 | 0.2 |  |  | `videoannotator process`: one job, run here, through the shared job path |
| `unit/cli/test_cli_prompts.py` | 3 | 0.0 |  |  | `videoannotator prompts ...` and `vlm preview` (spec 020) |
| `unit/cli/test_cli_rerun.py` | 3 | 0.0 |  |  | `videoannotator job rerun` (spec 019), and the job commands' API key |
| `unit/cli/test_cli_settings_token.py` | 7 | 0.1 |  |  | `videoannotator settings huggingface-token` (spec 031) |
| `unit/cli/test_cli_validate_emotion.py` | 2 | 0.0 |  |  |  |
| `unit/cli/test_generate_token_admin_default.py` | 8 | 2.1 |  |  | Tests for generate-token's admin-grant defaults and --admin/--no-admin flag |
| `unit/database/test_ask_tables.py` | 1 | 0.0 |  |  | Spec 027's tables are created on a fresh database by create_all |
| `unit/database/test_crud.py` | 53 | 0.7 |  |  | Unit tests for database/crud.py using in-memory SQLite |
| `unit/database/test_migrations.py` | 5 | 0.2 |  |  | Unit tests for v1.3.0 database migrations |
| `unit/docs/test_docs_version.py` | 2 | 0.0 |  |  | Every docs page names the version it documents (the site build's hook) |
| `unit/docs/test_pipeline_spec_doc.py` | 1 | 0.1 |  |  | Test that generated docs/pipelines_spec.md covers all registry pipelines |
| `unit/exporters/test_cli_export.py` | 16 | 0.4 |  |  | `videoannotator export ...` on a results folder, no server (spec 025) |
| `unit/exporters/test_elan.py` | 11 | 0.5 |  | 1 | ELAN files: valid EAF 3.0, one tier per track, matching the table (spec 025, US3) |
| `unit/exporters/test_methods.py` | 19 | 0.2 |  | 1 | The methods paragraph says exactly what the provenance records (spec 025, US2) |
| `unit/exporters/test_native_formats.py` | 35 | 0.1 |  |  | Unit tests for native format exporters (COCO, WebVTT, RTTM, TextGrid) |
| `unit/exporters/test_readers.py` | 26 | 0.1 |  |  | Each native format reads into events that match the file exactly (SC-002) |
| `unit/exporters/test_results_set.py` | 14 | 0.1 |  |  | Which files an export reads, by run and video (spec 025) |
| `unit/exporters/test_table.py` | 14 | 0.3 |  |  | The results table: columns, order, CSV/Parquet, summary (spec 025, US1) |
| `unit/pipelines/test_openface3_star_setup.py` | 3 | 0.0 |  |  | OpenFace's STAR config adds root-logger handlers (and a TensorBoard dir |
| `unit/pipelines/test_openface_compatibility.py` | 25 | 0.1 |  | 1 | Unit tests for pipelines/face_analysis/openface_compatibility.py |
| `unit/pipelines/test_pipeline_failures.py` | 8 | 0.1 |  | 1 | A pipeline that fails must raise, not return an empty result |
| `unit/pipelines/test_previous_models.py` | 4 | 0.0 |  |  | Spec 030 User Story 2: a model the defaults moved away from is still a setting |
| `unit/pipelines/test_speech_faster_whisper.py` | 18 | 0.1 |  |  | Speech recognition on the faster-whisper runtime (spec 030) |
| `unit/pipelines/test_vlm_provenance.py` | 2 | 0.0 |  |  | The VLM provenance block says where the model ran (spec 027 R8) |
| `unit/registry/test_citations.py` | 10 | 0.1 |  |  | Every shipped pipeline says what to cite (spec 025) |
| `unit/registry/test_forward_compat_stub.py` | 2 | 0.0 |  |  | Forward-compatibility gate: a stub pipeline with `requires_extras: []` |
| `unit/registry/test_no_test_fixtures_shipped.py` | 1 | 0.1 |  |  | Guard: nothing in the shipped metadata directory points at test code |
| `unit/registry/test_pipeline_guide.py` | 57 | 0.9 |  |  | The pipeline guide (spec 026): parsing, and the built-ins' completeness |
| `unit/registry/test_pipeline_registry.py` | 9 | 0.0 |  |  | Unit tests for extras-based metadata parsing and availability checking |
| `unit/registry/test_registry_extended_fields.py` | 2 | 0.2 |  |  |  |
| `unit/schemas/test_schemas.py` | 12 | 0.0 |  |  | Unit tests for schema modules (industry_standards + storage) |
| `unit/services/test_artifacts.py` | 2 | 0.0 |  |  |  |
| `unit/storage/test_cleanup.py` | 32 | 0.1 |  | 1 | Tests for storage cleanup functionality |
| `unit/storage/test_config.py` | 15 | 0.1 |  |  | Unit tests for storage configuration and path management |
| `unit/storage/test_file_backend.py` | 11 | 0.0 |  |  | Unit tests for FileStorageBackend - testing actual implementation |
| `unit/storage/test_legacy_job_folders.py` | 3 | 0.1 |  |  | Jobs made before v1.6.0 live under ./storage/jobs; they must still open |
| `unit/storage/test_list_jobs_by_dataset.py` | 2 | 0.1 |  |  | Jobs can be listed by the saved dataset they were run from (spec 025) |
| `unit/storage/test_provenance_storage.py` | 4 | 0.2 |  |  | Jobs keep each pipeline's provenance record (spec 017 US4) |
| `unit/storage/test_rerun_storage.py` | 4 | 0.2 |  |  | Jobs remember which job they run again (spec 019) |
| `unit/test_api_startup.py` | 9 | 2.9 |  |  | Tests for API startup and auto-generation functionality |
| `unit/test_ask_history.py` | 7 | 0.1 |  |  | Ask AI history entries (spec 027 R11) |
| `unit/test_audio_native_files.py` | 3 | 0.0 | 1 |  | Speech and diarization write the files the viewer loads (spec 014 follow-up) |
| `unit/test_check_workspace_mount.py` | 6 | 0.0 |  |  | scripts/check_workspace_mount.sh warns only for a Windows-drive workspace |
| `unit/test_compare_demo_outputs.py` | 6 | 0.0 |  |  | scripts/compare_demo_outputs.py tells identical, noisy and changed runs apart |
| `unit/test_config_env.py` | 30 | 0.2 |  | 1 | Tests for configuration environment variable management |
| `unit/test_cors_config.py` | 8 | 0.0 |  |  | Tests for CORS configuration and security defaults |
| `unit/test_database_location.py` | 6 | 1.0 |  |  | One database location for both database layers, not ./videoannotator.db |
| `unit/test_diagnostics.py` | 20 | 0.3 |  | 1 | Tests for diagnostic modules |
| `unit/test_display_paths_windows.py` | 4 | 0.0 |  |  | Paths shown as a Windows host shows them (spec 024, R3) |
| `unit/test_docker_compose_contract.py` | 10 | 0.0 |  |  | The documented Docker setup (spec 022, contracts/docker.md) |
| `unit/test_endpoint_class.py` | 12 | 0.0 |  |  | Local or remote, by the addresses a model server's host resolves to |
| `unit/test_logs_dir.py` | 7 | 0.1 |  |  | Logs go to one per-user directory, not ./logs (v1.6.0 Phase 1) |
| `unit/test_models_dir.py` | 9 | 0.2 |  |  | One models directory (spec 016) |
| `unit/test_namespace.py` | 20 | 0.1 |  |  | Tests for videoannotator namespace and module imports |
| `unit/test_no_telemetry.py` | 3 | 0.0 |  |  | No telemetry from VideoAnnotator or the libraries it drives (constitution I) |
| `unit/test_notebooks.py` | 5 | 0.2 | 1 |  | The worked notebooks run, on a results table that is up to date (spec 025, US4) |
| `unit/test_provenance.py` | 7 | 0.0 |  |  | Provenance records and how they are stamped into output files (spec 017) |
| `unit/test_results_folder_names.py` | 41 | 0.2 | 1 | 1 | Naming and layout rules of the results folder (spec 022, research.md R2/R3) |
| `unit/test_results_owner.py` | 4 | 0.0 |  |  | Results belong to the researcher under rootful Docker (spec 024, R6) |
| `unit/test_secrets_store.py` | 10 | 0.0 |  |  | The saved Hugging Face token and where the server gets its token (spec 031) |
| `unit/test_supported_python.py` | 9 | 0.3 |  |  | The supported Python range is declared in four places; keep them in step |
| `unit/test_validate_docs_links.py` | 2 | 0.0 |  |  | scripts/validate_docs_links.py finds dead relative links and nothing else |
| `unit/test_versions_match.py` | 2 | 0.0 |  |  | One product, one version: pyproject, version.py and the bundled viewer agree |
| `unit/test_video_unavailable_reason.py` | 5 | 0.0 |  |  | Why a job's video can't be found: not shared any more, or moved (spec 024, R9) |
| `unit/test_vlm_gate.py` | 4 | 0.1 |  |  | The per-model gate: a try is served before a run's next call (spec 027 R5) |
| `unit/test_vlm_pipeline_progressive.py` | 7 | 0.2 |  |  | A VLM run writes its answers as they come and can stop keeping them (spec 027) |
| `unit/test_vlm_sampling.py` | 51 | 0.2 |  |  | The VLM sampling functions against the cases the viewer's twin also reads |
| `unit/utils/test_automatic_labeling.py` | 31 | 0.1 |  |  | Unit tests for utils/automatic_labeling.py |
| `unit/utils/test_console_output.py` | 14 | 0.1 |  |  | What the server's terminal shows, and that API keys never reach the logs |
| `unit/utils/test_math_utils.py` | 16 | 0.1 |  |  | Unit tests for mathematical utility functions |
| `unit/utils/test_model_loader.py` | 19 | 0.2 |  | 1 | Tests for model loading utilities with enhanced progress logging |
| `unit/utils/test_person_identity.py` | 41 | 0.1 |  |  | Unit tests for utils/person_identity.py |
| `unit/utils/test_size_analysis.py` | 7 | 0.0 | 1 | 1 | Test integration of size-based analysis with person tracking pipeline |
| `unit/utils/test_size_based_person_analysis.py` | 15 | 0.0 |  |  | Tests for Size-Based Person Analysis |
| `unit/utils/test_torch_settings.py` | 4 | 0.0 |  |  | Each pipeline starts from the same torch settings and can't leak its own |
| `unit/utils/test_transform_utils.py` | 9 | 0.0 |  |  | Unit tests for coordinate transform utility functions |
| `unit/validation/test_coco_validator.py` | 26 | 0.1 |  |  | Unit tests for COCO format validation |
| `unit/validation/test_emotion_validator.py` | 4 | 0.0 |  |  |  |
| `unit/validation/test_validator.py` | 33 | 0.1 |  |  | Unit tests for configuration validator |
| `unit/worker/test_cancellation.py` | 15 | 0.1 |  |  | Unit tests for CancellationManager |
| `unit/worker/test_job_processor.py` | 16 | 5.2 |  |  | Tests for JobProcessor worker (v1.3.0) |
