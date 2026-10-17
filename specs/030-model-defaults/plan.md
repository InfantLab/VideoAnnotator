# Implementation Plan: Refresh Model Defaults

**Branch**: `1.6-dev` (spec `030-model-defaults`) | **Date**: 2026-10-17 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/030-model-defaults/spec.md`

## Summary

Change four pipelines' defaults before the public release, each on its upstream's published
evaluation and a demo-clip check: speech recognition moves from openai-whisper `base` to
faster-whisper `large-v3-turbo` (openai-whisper removed, Q3); speaker diarization to pyannote
`speaker-diarization-community-1` (Q2); person tracking to `yolo26n-pose` (needs ultralytics 8.4);
face analysis turns age and gender off and stops running their models when off. OpenFace 3 and
the face detector only gain a recorded reason. Every switch is its own step: library upgrade with
the old model first (FR-002), then the model, each run on the demo clip through the server and
compared with a "before" run on the same machine, then the metadata, guide figures, citations and
CHANGELOG entry for that step. The fixtures are regenerated once at the end, on purpose.

Research behind every figure: [research.md](research.md) §0–§7. Decisions this plan adds:
[research.md § Plan decisions](research.md#plan-decisions-2026-10-17).

## Technical Context

**Language/Version**: Python 3.12 and 3.13 (ruff/mypy target 3.12)
**Primary Dependencies**: faster-whisper ≥1.2.1 (new; CTranslate2 4.x, PyAV, onnxruntime; no
torch), ultralytics ≥8.4 (from ≥8.3), pyannote.audio 4.0.7 (unchanged), deepface (unchanged).
Removed: openai-whisper and its source build (`tool.uv.extra-build-dependencies`).
**Storage**: model weights in the models directory (spec 016); faster-whisper downloads through
the Hugging Face hub cache (`HF_HUB_CACHE`, already pointed into the models directory)
**Testing**: pytest (unit, mocked models); `tests/integration/test_output_baseline.py`
(`real_models`, GPU, through a real server); viewer contract test (`cd viewer && bun run test:run`)
**Target Platform**: Linux, macOS, Windows; CUDA GPU optional
**Project Type**: single project (Python package with bundled viewer; viewer unchanged)
**Performance Goals**: none new. Recorded, not gated: speech runtime and VRAM on the demo clip
before and after, on the machine that runs the checks
**Constraints**: torch stays `>=2.11,<2.12`; output formats unchanged (FR-006); no telemetry
(FR-011); the `audio` extra installs with no compiler (FR-016)
**Scale/Scope**: 4 pipelines' code and metadata, `pyproject.toml`/`uv.lock`, CHANGELOG, docs
mentioning openai-whisper, fixtures. About 6 demo-clip runs on a GPU.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Check | Result |
|---|---|---|
| I. Local-First, No Telemetry | New dependencies (faster-whisper, ultralytics 8.4's `ultralytics-platform`) checked for network calls; `tests/unit/test_no_telemetry.py` extended to the new speech runtime and re-run after the ultralytics upgrade (FR-011). Weights download once from Hugging Face / GitHub as today; no inference leaves the machine. | Pass |
| II. Stable Contract, Open Formats | WebVTT, RTTM, COCO unchanged (FR-006). The speech result dict keeps its keys (adapter in [data-model.md](data-model.md)). Metadata schema gains optional keys only (`reason`, `reason_source` on a setting): allowed within a minor version. Viewer contract test re-run on the regenerated fixtures (FR-005). | Pass |
| III. Provenance | Every new default records repository and revision (`hub_ref`) or sha256 (`weights_ref`); speech also records CTranslate2's effective compute type as the quantisation (FR-007). Old behaviour reachable by declarative config where kept. | Pass |
| IV. Modular | faster-whisper goes into the `audio` extra only; core unchanged. The `audio` extra gets lighter (no source build). | Pass |
| V. Backward Compatibility | Defaults change within a minor version with schema preserved and release notes: allowed. **But** openai-whisper's removal means speech can't pin the previous checkpoint+runtime for bit-identical output. | **Exception** (Complexity Tracking) |
| VI. Faithful Display | Viewer unchanged; overlays keep naming pipeline and version; the new model names come through provenance. | Pass |
| Engineering: licensing | community-1 CC-BY-4.0 (citation added, FR-012); faster-whisper and turbo weights MIT; YOLO26 AGPL-3.0 as YOLO11 today, still isolated in `person`; no research-only default (OpenFace 3 stays opt-in). | Pass |

Re-check after Phase 1 design: unchanged. The design adds no output key, no endpoint and no
dependency outside the `audio` and `person` extras.

## Project Structure

### Documentation (this feature)

```
specs/030-model-defaults/
├── spec.md
├── research.md          # upstream evidence (§0–§7) + plan decisions (Phase 0)
├── plan.md              # this file
├── data-model.md        # metadata additions, speech result adapter, provenance per model
├── contracts/
│   ├── metadata.md      # per-pipeline metadata changes and the `reason` keys
│   └── speech-settings.md  # each speech setting under faster-whisper (FR-014)
├── quickstart.md        # the demo-clip check, step by step
└── tasks.md             # /speckit-tasks
```

### Source Code (repository root)

```
pyproject.toml, uv.lock                      # audio: -openai-whisper +faster-whisper; person: ultralytics>=8.4
src/videoannotator/
├── pipelines/audio_processing/
│   ├── whisper_base_pipeline.py             # model loading → faster-whisper; openai-whisper/HF branches removed
│   ├── speech_pipeline.py                   # transcribe via faster-whisper; result adapter; translate check
│   └── diarization_pipeline.py              # default model id
├── pipelines/person_tracking/person_pipeline.py   # default model, docstrings
├── pipelines/face_analysis/face_pipeline.py       # defaults off; _deepface_actions() fix
├── provenance.py                            # whisper_ref() → hub_ref for CTranslate2 repos; compute type
├── api/readiness.py                         # `whisper` cache kind no longer used by a default
├── registry/pipeline_registry.py            # PipelineConfigField.reason / reason_source
├── api/v1/pipelines.py, cli.py              # expose `reason` (additive), `pipelines describe` prints it
└── registry/metadata/{speech_recognition,speaker_diarization,person_tracking,face_analysis,
                      face_openface3_embedding}.yaml
tests/
├── unit/pipelines/ (speech loader, face actions), tests/pipelines/test_*whisper*  # rewritten for faster-whisper
├── unit/test_no_telemetry.py
├── integration/test_output_baseline.py      # + face_analysis (FR-024); VIDEOANNOTATOR_BASELINE_KEEP copies outputs
└── fixtures/viewer_contract/                # regenerated at the end (FR-005)
scripts/compare_demo_outputs.py              # step-to-step comparison summary (quickstart §2)
CHANGELOG.md, docs/ (install pages mentioning openai-whisper or its build)
```

**Structure Decision**: single project; changes stay inside the four pipelines, their metadata,
the registry's config-field model and the two extras. No viewer code changes ("Nothing new on
screen"), so no bundle rebuild unless the regenerated fixtures require a viewer fix.

## Sequence (FR-002: one cause per diff)

Each step ends with a demo-clip run through the server (quickstart §2), a written comparison
against the previous step's run on the same machine, and a commit. Steps 1–3 need no library
change; 4 and 5 are two commits each.

0. **Before**: on the machine that will run every later step, run the five baseline pipelines
   plus `face_analysis` with today's code and keep the outputs (quickstart §1). Today's committed
   fixtures were made on an RTX 4060; comparisons across GPUs would mix GPU noise into every
   diff. Also note the 5 current baseline failures' causes (handover item 4) so they aren't
   blamed on this spec.
1. **Metadata groundwork**: `reason`/`reason_source` on config fields (registry, API, CLI);
   reasons for the unchanged defaults that need one (RetinaFace, FR-025; OpenFace 3 status,
   FR-026). Unit tests only.
2. **Faces**: `detect_age`/`detect_gender` default false (code and metadata);
   `_deepface_actions()` returns only the enabled actions; metadata `weights` drop age and gender.
   Demo clip: detections and emotions identical to step 0, no age/gender. If RetinaFace finds
   faces in the demo clip, `face_analysis` joins the baseline (FR-024).
3. **Speaker turns**: default `pyannote/speaker-diarization-community-1`; `requires_setup` asks
   for its licence only; weights, homepage, benchmarks (community-1 card), citation.
4. **People**: (a) ultralytics ≥8.4 with `yolo11n-pose`; demo clip should match step 0 within
   tolerance, and the no-telemetry test passes. (b) `yolo26n-pose`; settings it ignores
   documented (research decision P2); name, description, variant, citation updated (FR-021).
5. **Speech**: (a) faster-whisper with `base`, openai-whisper removed from the extra and the
   lock. (b) `large-v3-turbo`; translate rule; first-run download message; CPU note in the guide.
6. **Release record**: CHANGELOG entry per pipeline in the spec's format, filled from steps 2–5;
   regenerate fixtures (`tests/fixtures/viewer_contract/README.md`), baseline test and viewer
   contract test green; spec 026 benchmark entries checked against sources (SC-005); roadmap
   Phase 5c ticked.

A step whose demo-clip check shows a regression (a speaker or person lost, a face missed) stops
there: the maintainer decides between explaining it in the CHANGELOG and keeping the old default.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Principle V: speech can no longer pin the previous runtime and checkpoint for bit-identical v1.5 transcripts (openai-whisper removed) | Q3 (maintainer, 2026-10-17). No outside lab has v1.5/v1.6 results (v1.6.0 is unannounced, v1.7.0 is public); models will change again before v1.7.0. Removing it also drops the only source build in the install. | Keeping openai-whisper as an optional runtime means two speech code paths, two test sets and the sdist build for a reproducibility need no user has yet. The release notes say v1.6.0rc1 is the last version with it. The old model *size* stays selectable (`whisper_model: base`), so only bit-identity is lost. |
