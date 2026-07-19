# Phase 1 Data Model: Extras-Based Modular Install

## `PipelineMetadata` (extended)

Location: `src/videoannotator/registry/pipeline_registry.py`. Additive change only — no existing
field renamed, removed, or retyped (constitution Principle II).

| Field | Type | Status | Notes |
|---|---|---|---|
| `name` | `str` | existing | unique pipeline id, e.g. `face_analysis` |
| `display_name`, `description`, `outputs`, `config_schema`, `version` | — | existing | unchanged |
| `pipeline_family` | `str \| None` | existing | `face` \| `audio` \| `scene` \| `person` |
| `variant` | `str \| None` | existing | e.g. `deepface`, `laion-clip-face` |
| `tasks`, `modalities`, `capabilities` | `list[str]` | existing | unchanged |
| `backends` | `list[str]` | existing | runtime backend descriptors (`tensorflow`, `opencv`, future `ollama`, `http`) — orthogonal to `requires_extras`, see research.md §2 |
| `stability` | `str \| None` | existing | `stable` \| `beta` \| `experimental` |
| **`requires_extras`** | `list[str]` | **new** | pip extras-group name(s) needed for this pipeline to import successfully; `[]` is valid (research.md §2) |
| `module_path` | `str` | **now required, no fallback** | `"module.path:ClassName"`; `LEGACY_MAPPINGS` removed, every YAML file must carry this explicitly |

Validation rule: `PipelineRegistry.load()` MUST treat a metadata file missing `module_path` as an
error (warn + skip that entry, matching existing graceful-degradation posture for malformed
files), not fall back to a hardcoded table.

## Extras group (declarative, `pyproject.toml`)

Not a Python class — a named key under `[project.optional-dependencies]`. Reserved names after
this phase (research.md §1):

`face`, `face-laion`, `face-openface3`, `audio`, `audio-laion`, `scene`, `person`, `all`, `dev`
(existing), `annotation` (existing, currently empty/commented).

Reserved-but-not-yet-created (namespace held open per FR-010, not created by this phase):
`llm` (v1.6.0).

## Migration message record

Not a persisted entity — a lookup table (implementation detail for `/speckit-tasks`) mapping a
pipeline `name` that existed in v1.4.4's default install to the extras group it now requires,
consulted only when that pipeline is requested and unavailable:

| Pipeline `name` | v1.4.4 status | v1.5.0 extras group |
|---|---|---|
| `face_laion_clip` | default | `face-laion` |
| `laion_voice` | default | `audio-laion` |
| `face_openface3_embedding` | default | `face-openface3` |
| (all other pipelines) | default | their `pipeline_family` group |

## State / lifecycle

No new stateful entities. `PipelineRegistry`'s existing load-once-cache lifecycle
(`_loaded` flag, `force` reload) is unchanged; `requires_extras` is read at the same load time as
every other metadata field — no new I/O pass.
