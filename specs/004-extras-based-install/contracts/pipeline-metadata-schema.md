# Contract: Pipeline Metadata YAML Schema (delta)

Applies to every file under `src/videoannotator/registry/metadata/*.yaml`. This is the contract
third-party plugin authors (v1.6.0+) and this repo's own pipeline authors must follow.

## Before (current)

```yaml
name: face_analysis
display_name: Face Analysis (DeepFace)
pipeline_family: face
variant: deepface
# ... tasks, modalities, capabilities, outputs, config_schema, version, stability, backends ...
requirements:
  packages:
    - deepface
    - opencv-python
```

`requirements.packages` is informational only — nothing in the loader reads it. Module resolution
falls back to a hardcoded `LEGACY_MAPPINGS` dict in `pipeline_loader.py` keyed by `name`.

## After (this phase)

```yaml
name: face_analysis
display_name: Face Analysis (DeepFace)
pipeline_family: face
variant: deepface
module_path: videoannotator.pipelines.face_analysis:FaceAnalysisPipeline   # now required
requires_extras: [face]                                                    # new, required (may be [])
# ... tasks, modalities, capabilities, outputs, config_schema, version, stability, backends ...
```

`requirements.packages` may remain as human-readable documentation but is no longer
load-bearing; `module_path` and `requires_extras` are the only fields the loader/registry consult
for import resolution and availability checks.

## Compatibility rule

- `module_path` absent → registry logs a warning and omits the pipeline (same posture as any other
  malformed metadata file today); this is a breaking change only for hand-authored YAML that never
  had `module_path`, which is none of the bundled files once this phase lands.
- `requires_extras` absent → treated as `[]` (no extras required) for backward compatibility with
  any third-party metadata file authored before this field existed, rather than erroring.
