# Contract: API additions

All additive; no field renamed or removed (Principle V). Absent metadata → field `null` (or `[]`).

## `GET /api/v1/pipelines` and `GET /api/v1/pipelines/{name}` (`PipelineInfo`)

New fields:

```json
{
  "title": "Faces: expression, age, gender",
  "summary": "Finds faces in each sampled frame and estimates each face's expression, age and gender.",
  "homepage": "https://github.com/serengil/deepface",
  "benchmarks": [
    {"name": "WIDER FACE (hard)", "measure": "AP", "figure": "91.4%",
     "source": "https://arxiv.org/abs/1905.00641", "note": "RetinaFace detector, ResNet-50 backbone"}
  ],
  "quality_note": null,
  "citations": [{"key": "deng2020retinaface", "bibtex": "@inproceedings{…}"}]
}
```

Each `config_schema` entry gains `group`, `label`, `unit`, `min`, `max`, `applies_when` (and keeps
`type`, `default`, `description`):

```json
"scale_factor": {"type": "float", "default": 1.1, "description": "…",
                 "group": "technical", "label": "Haar cascade scale factor",
                 "unit": null, "min": null, "max": null,
                 "applies_when": {"detection_backend": ["opencv"]}}
```

`benchmarks` is `null` when the metadata doesn't state it and `[]` when it states there is none,
so clients can tell "not stated" from "no published benchmark".

## `GET /api/v1/pipelines/{name}/schema` (`PipelineSchemaResponse`)

- `pipeline` gains `title`, `summary`.
- Each parameter gains `group` (`"behavioural"` | `"technical"`), `unit`, `min`, `max`,
  `applies_when`; `label` is the metadata label when present, else today's title-cased key.

The viewer's `PipelineParameterSchema` (`viewer/src/types/pipelines.ts`) already declares `group`,
`unit`, `min`, `max`; it gains `appliesWhen` (mapped from `applies_when` in the client, as other
snake_case fields are).

## Tests

`tests/api/test_pipeline_guide_api.py`: both routes return the new fields for a built-in; a stub
metadata file without them returns nulls; existing fields unchanged (snapshot of today's keys).
