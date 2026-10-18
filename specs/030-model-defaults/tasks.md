---
description: "Tasks for spec 030: Refresh Model Defaults"
---

# Tasks: Refresh Model Defaults

**Input**: `specs/030-model-defaults/` (plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md)
**Tests**: included. The constitution requires tests for new API/CLI surface; FR-001/FR-005 make the
demo-clip check and the re-baseline part of every switch.
**Organization**: by user story (US1 better defaults with the reason on record, US2 old model where
it is only a setting, US3 faces). US4 (Sortformer spike) is not a v1.6.0 task (FR-028; roadmap
v1.7.0 Phase 6).

**Order**: phases follow the plan's Sequence, not story priority. Faces (US3) go first because
they need no library change; US1 is then done one pipeline at a time, library before model
(FR-002). Every "demo-clip check" task means quickstart §1 then §2 against the previous check's
folder, on the same machine (research B1), with the comparison summary in the commit message.

**GPU tasks** are marked **[GPU]**: they need the machine the maintainer chooses for every
comparison, with the `all` extra and the Hugging Face token.

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup (check tooling)

- [X] T001 In `tests/integration/test_output_baseline.py`: when `VIDEOANNOTATOR_BASELINE_KEEP` is set, add `face_analysis` to the job's pipelines and copy every output (after `RENAMES`, provenance included) plus `server.log` into that folder; behaviour without the variable unchanged
- [X] T002 [P] Create `scripts/compare_demo_outputs.py <before> <after>`: per file identical / within the baseline test's `TOLERANCE` / different (reuse `_without_run_details` logic by importing or copying it); for different files a summary per quickstart §2 (`.vtt` cues added/removed/reworded and largest timing shift; `.rttm` turns and speaker count; `person_*` people per frame, track count, keypoints per person; `face_detections` faces per frame and attribute keys present); ruff/mypy clean
- [X] T003 [P] Unit test for the comparison summary on small hand-made inputs in `tests/unit/test_compare_demo_outputs.py`
- [X] T004 [GPU] Step 0: run quickstart §1 into `~/va-030/step0` with today's code; record the GPU name and the causes of the current baseline failures (handover item 4: scene, diarization, OpenFace 3 ×2, person) in `specs/030-model-defaults/checks.md` (new file: one section per step, filled from T004 on)

**Checkpoint**: a "before" run exists on the comparison machine.

---

## Phase 2: Foundational (the reason beside a default)

- [X] T005 In `src/videoannotator/registry/pipeline_registry.py`: add `reason: str | None` and `reason_source: str | None` to `PipelineConfigField`; parse them; drop `reason_source` with a warning if `reason` is missing or the link isn't `https://` (data-model §1)
- [X] T006 In `src/videoannotator/api/v1/pipelines.py`: add nullable `reason`, `reason_source` to each config-field object in both pipeline routes (additive, next to spec 026's `group`/`label`)
- [X] T007 In `src/videoannotator/cli.py` (`pipelines describe`, near the `applies_when` printing): print `Why this default: <reason> (<reason_source>)` under a setting that has one
- [X] T008 [P] Tests: parsing and the warning in `tests/unit/registry/test_pipeline_guide.py`; API fields in `tests/api/test_pipeline_guide_api.py`; describe output in the existing `pipelines describe` CLI test
- [X] T009 [P] Regenerate the viewer's API types (`scripts/gen_viewer_api_types.sh`) if the OpenAPI schema changed, and run `cd viewer && bun run typecheck`; no viewer UI change
- [X] T010 Record unchanged defaults' reasons: RetinaFace detector in `src/videoannotator/registry/metadata/face_analysis.yaml` (`reason` on the detector setting if it is in `config_schema`, else in `quality_note`; plus a benchmark entry WIDER FACE hard AP 91.4 %, arXiv:1905.00641) (FR-025); OpenFace 3 status in `src/videoannotator/registry/metadata/face_openface3_embedding.yaml` (`openface-test` is the distribution CMU names, unchanged since 2025-06, non-commercial licence; pin stays) (FR-026)

**Checkpoint**: `pytest tests/unit tests/api -q` green; commit "registry: a setting can say why its default".

---

## Phase 3: User Story 3 — Faces: only what says something about infants (P2, plan step 2)

**Goal**: age and gender off by default, and not downloaded or run when off.
**Independent Test**: default face run has detections and expressions, no age/gender, and no
age/gender weights downloaded; with `detect_age: true` only the age model is added.

- [X] T011 [US3] Unit tests first in `tests/unit/pipelines/test_face_actions.py`: `_deepface_actions()` for each combination of the three flags (only enabled actions; none enabled → `["emotion"]`); `_prepare_deepface_models()` asks for age/gender weights only when enabled; default config has both off
- [X] T012 [US3] In `src/videoannotator/pipelines/face_analysis/face_pipeline.py`: `detect_age`/`detect_gender` default `False` (default config and every `self.config.get(..., True)` for those two keys); `_deepface_actions()` returns only the enabled actions (research F1)
- [X] T013 [US3] In `src/videoannotator/registry/metadata/face_analysis.yaml`: `detect_age`/`detect_gender` `default: false` with `reason` and `reason_source` (contracts/metadata.md); remove the age/gender entries from `weights:`; update the example API body; `quality_note` says the expression model was trained on adult faces (FR-023)
- [X] T014 [US3] [GPU] Demo-clip check into `~/va-030/step2`: detections and emotions identical to step 0, no age/gender keys, no age/gender files under `<models>/deepface/.deepface/weights` after a run with an empty DeepFace home; record in `checks.md`
- [X] T015 [US3] If T014 shows faces in the demo clip: add `face_analysis` to `PIPELINES` and `face_detections.json` (or the name the job writes) to `TOLERANCE` in `tests/integration/test_output_baseline.py`, with a tolerance measured from two runs (FR-024); otherwise update the comment there to say RetinaFace also finds none

**Checkpoint**: commit "face: age and gender off by default, and not run when off".

---

## Phase 4: User Story 1 — Better defaults, with the reason on record (P1, plan steps 3–5)

**Goal**: speech, speaker turns and people use the current recommended models, each with its
reason, benchmark and provenance.
**Independent Test**: each changed pipeline on the demo clip with no settings: provenance names
the new model and revision; metadata and guide give the reason and figure with a source link.

### Speaker turns (plan step 3)

- [ ] T016 [P] [US1] In `src/videoannotator/pipelines/audio_processing/diarization_pipeline.py`: default `model` `pyannote/speaker-diarization-community-1`; update docstrings naming 3.1
- [ ] T017 [P] [US1] In `src/videoannotator/registry/metadata/speaker_diarization.yaml` per contracts/metadata.md: `homepage`, `model.default` + `reason`/`reason_source`, `model.description` (other models need their own licences), `benchmarks` from the community-1 card (quote exactly; re-check the card), `requires_setup` (secret + community-1 licence only), `weights` (community-1, size from the Hub API), community-1 citation added (FR-012)
- [ ] T018 [US1] Update tests that assert the 3.1 default or three licences: `grep -rn "speaker-diarization-3.1\|segmentation-3.0" tests/ src/` and fix each (readiness tests, registry tests, viewer readiness copy if it names them: `grep -rn "segmentation-3.0" viewer/src`)
- [ ] T019 [US1] [GPU] Demo-clip check into `~/va-030/step3`: only `.rttm` differs; turns and speaker count recorded in `checks.md`; Speaker turns card ready with a token whose account accepted community-1 (quickstart §3)

**Checkpoint**: commit "diarization: default to pyannote community-1".

### People (plan step 4)

- [ ] T020 [US1] `pyproject.toml` `person` extra: `ultralytics>=8.4,<9`; `uv lock --upgrade-package ultralytics`; `uv sync --inexact --extra all`; note new transitive packages (`ultralytics-platform`, `polars`, `nvidia-ml-py`) in `docs/development/dependency_audit_v1.6.0.md`
- [ ] T021 [US1] Run `tests/unit/test_no_telemetry.py` and the person-tracking unit tests under 8.4; if `ultralytics-platform` or 8.4 adds a network call or a setting, switch it off where analytics are switched off today (`grep -rn "sync" src/videoannotator/pipelines/person_tracking/`) and extend the test
- [ ] T022 [US1] [GPU] Demo-clip check (still `yolo11n-pose`) into `~/va-030/step4a`: `person_*` within tolerance of step 3's; if not, stop and investigate before T023; record; commit "person: ultralytics 8.4 (same model)"
- [ ] T023 [US1] In `src/videoannotator/pipelines/person_tracking/person_pipeline.py`: default `model` `models/yolo/yolo26n-pose.pt`; docstrings and log lines stop naming YOLO11 (FR-021); `grep -rn "yolo11" src/ tests/ docs/ README.md` and update each that describes the default (leave history in CHANGELOG/roadmaps)
- [ ] T024 [US1] In `src/videoannotator/registry/metadata/person_tracking.yaml` per contracts/metadata.md: `description`, `variant`, `model.default` + `reason`/`reason_source`, `benchmarks` (YOLO26n-pose 57.2 e2e; re-check the source), `requirements.packages`, YOLO26 citation
- [ ] T025 [US1] [GPU] Demo-clip check into `~/va-030/step4b`; 17 keypoints per person; track count; run again with `{"person_tracking": {"iou_threshold": 0.3}}` and, if identical, write "No effect with YOLO26 (NMS-free)" into `iou_threshold`'s description in the metadata (research P2); record

**Checkpoint**: commit "person: default to yolo26n-pose".

### Speech (plan step 5)

- [ ] T026 [US1] `pyproject.toml` `audio` extra: remove `openai-whisper` and its `[tool.uv.extra-build-dependencies]` entry; add `faster-whisper>=1.2.1,<2`; keep `librosa`/`numba` only if still imported after T027 (check); `uv lock`; `uv sync --inexact --extra all`; remove `whisper` from mypy's ignore list in `pyproject.toml` if nothing imports it, add `faster_whisper`/`ctranslate2` if needed
- [ ] T027 [US1] In `src/videoannotator/pipelines/audio_processing/whisper_base_pipeline.py` (research S1–S3, S7): load `faster_whisper.WhisperModel(whisper_model, device=…, compute_type=…)` with `compute_type` from config (`use_fp16: false` → `float32`); after loading, transcribe one second of silence so CUDA library errors surface in `initialize()` and reach the existing CPU fallback; if T030 shows CTranslate2 can't find cuBLAS/cuDNN, preload them from the `nvidia.cublas`/`nvidia.cudnn` packages first; remove the transformers branch, `_get_standard_whisper_module`, `get_whisper_embedding`, `segment_audio`, `_pad_or_trim_embedding`; keep device selection and ffmpeg audio extraction; first-run line names model and download size, plus the smaller-model hint on CPU
- [ ] T028 [US1] In `src/videoannotator/pipelines/audio_processing/speech_pipeline.py`: transcribe with faster-whisper per contracts/speech-settings.md (one options builder for both path and array input; `suppress_tokens` comma string → `list[int]`; `vad_filter=False`); build the result dict from `(segments, info)` per data-model §2, consuming the generator inside the `try`; remove the `whisper` `SimpleNamespace` shim and `WHISPER_AVAILABLE` (replace with a `faster_whisper` availability check); `get_pipeline_info` reports the runtime
- [ ] T029 [US1] In `src/videoannotator/provenance.py`: remove `whisper_ref`; speech records `hub_ref(<repository faster-whisper resolved>)` and the effective compute type as quantisation (data-model §3; find where the pipeline's determinism/settings block is built for provenance and add it there); update `tests/unit` provenance tests
- [ ] T030 [US1] Rewrite the speech unit tests against a mocked `faster_whisper.WhisperModel`: `tests/unit/pipelines/test_speech_whisper_loader.py`, `tests/pipelines/test_whisper_base_pipeline.py`, `tests/pipelines/test_whisper_base_pipeline_stage1.py`, `tests/pipelines/test_audio_speech_pipeline.py`, `tests/pipelines/test_audio_individual_components.py`, `tests/unit/pipelines/test_pipeline_failures.py` (speech cases), `tests/conftest.py` (whisper fixtures); delete tests of removed helpers; add tests for the settings mapping, `suppress_tokens` parsing, the result adapter, `use_fp16: false`, and CUDA→CPU fallback when the warm-up raises a cuDNN error
- [ ] T031 [US1] Telemetry: extend `tests/unit/test_no_telemetry.py` to faster-whisper (no network beyond the Hugging Face download, which already has `HF_HUB_DISABLE_TELEMETRY`)
- [ ] T032 [US1] Keep `whisper_model: base` in `speech_recognition.yaml` for this step; set `requirements.packages`, `backends`, `homepage`, `variant`, `description` (runtime only) and the `weights` entry for base's CTranslate2 repository per contracts/metadata.md; update `src/videoannotator/api/readiness.py` docs/comments so `cache: whisper` is described as legacy
- [ ] T033 [US1] [GPU] Demo-clip check (faster-whisper, `base`) into `~/va-030/step5a`: on Linux with CUDA, confirm it ran on the GPU (log line) or record the cuDNN fix needed (then T027's preload); cues reworded and largest timing shift recorded; commit "speech: faster-whisper runtime (same model)"
- [ ] T034 [US1] Translate rule (research S5) in `src/videoannotator/pipelines/audio_processing/speech_pipeline.py` `initialize()` with a unit test; `task` description in metadata says translation needs a non-turbo model
- [ ] T035 [US1] Default `whisper_model: large-v3-turbo` in `speech_pipeline.py`, `whisper_base_pipeline.py` and `speech_recognition.yaml` (+ `reason`/`reason_source`, `benchmarks` replaced with the turbo card's figure, `weights` ~1,618 MB under the repository id the cache actually uses, `compute_type` setting, CPU note in the `whisper_model` description, examples updated)
- [ ] T036 [US1] [GPU] Demo-clip check into `~/va-030/step5b`: cue count and wording; GPU memory and run time vs step 5a from `server.log`; first-run message named the download; the translate rule (quickstart §3); record; commit "speech: default to large-v3-turbo"
- [ ] T037 [US1] Docs: `grep -rn "openai-whisper\|extra-build-dependencies\|whisper base\|yolo11\|speaker-diarization-3.1" docs/ README.md CONTRIBUTING.md docker* Dockerfile* scripts/` and update install pages, Docker files and the models docs to the new defaults and sizes; `uv run --no-sync python scripts/validate_docs_links.py`; `uv run --no-sync mkdocs build --strict`

**Checkpoint**: all three pipelines switched; unit suite green.

---

## Phase 5: User Story 2 — Old model where it is only a setting (P3)

**Goal**: configs naming the previous diarization or pose model still run and record it.
**Independent Test**: a config naming `pyannote/speaker-diarization-3.1` or `yolo11n-pose` runs and
records that model.

- [ ] T038 [P] [US2] Unit tests: `DiarizationPipeline({"model": "pyannote/speaker-diarization-3.1"})` loads that id (mocked `from_pretrained`) and records it via `hub_ref`; `PersonTrackingPipeline({"model": "yolo11n-pose.pt"})` resolves and records that file; a v1.5.0-style config with none of these keys initializes with the new defaults — in `tests/unit/pipelines/test_previous_models.py`
- [ ] T039 [US2] [GPU] One real job each with the two old-model configs on the demo clip; record in `checks.md` that they run and what provenance says (no baseline kept, FR-008)

---

## Phase 6: Polish & release record (plan step 6)

- [ ] T040 CHANGELOG: under the unreleased v1.6.0 section, one entry per changed pipeline in the spec's format ("What the researcher sees"): old → new, why with source, demo-clip differences from `checks.md`, the setting that restores the old model (none for the speech runtime: say v1.6.0rc1 is the last with openai-whisper; Principle V exception); faces: age/gender off and ~1 GB less download (SC-001, SC-006)
- [ ] T041 [GPU] Re-baseline per quickstart §4: replace `tests/fixtures/viewer_contract/demo_clip_*` from the final kept run (step 5b, rerun if anything changed since), keep `legacy/` as is; update the README there (date, what changed, GPU); baseline test green twice in a row on the same machine (re-measure `TOLERANCE` if the GPU isn't the RTX 4060); `cd viewer && bun run test:run` green
- [ ] T042 [P] Check every changed benchmark entry against its source and the guide output (`videoannotator pipelines describe <name>` for each of the four), SC-005
- [ ] T043 [P] Methods paragraph for a demo-clip job names every new model and revision (`videoannotator export methods <results folder>`), SC-004
- [ ] T044 [P] `audio` extra installs with no compiler: `uv pip install --dry-run` against a fresh venv on Linux; CI's install matrix covers macOS/Windows on push (SC-008); record
- [ ] T045 Roadmap: tick Phase 5c items in `docs/development/roadmap_v1.6.0.md` and the success criterion "Every change of default model has its reason in the CHANGELOG"; mark tasks done here
- [ ] T046 Full gate: `pytest tests/ -q`, `ruff check .`, `mypy src/videoannotator`, `pre-commit run --all-files`

---

## Dependencies

- Phase 1 → everything (T004 is the reference run for T014).
- Phase 2 → the metadata tasks in Phases 3–4 (they use `reason`).
- Within Phase 4: speaker turns, people, speech are independent of each other in code, but their
  GPU checks chain (each compares with the previous step's folder). Inside people and speech, the
  library task's check (T022, T033) must pass before the model task (T023, T035).
- Phase 5 after Phase 4. Phase 6 last; T041 after every GPU check.

## Parallel opportunities

- T002/T003 alongside T001; T008/T009 after T005–T007.
- T016 and T017 together; T038 any time after T016/T023.
- T042, T043, T044 together after T041.

## Implementation strategy

MVP: Phases 1–3 (tooling, reasons, faces) ship value alone: 1 GB less to download and no
misleading age/gender, with no library change. Then each Phase 4 sub-step is a releasable
increment of its own. If a check shows a regression, stop at that step (plan Sequence) and ask the
maintainer before going on.
