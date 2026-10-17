# Contract: pipeline metadata additions

All keys optional for loading (FR-011); required for built-ins by the metadata check. Example,
`face_analysis.yaml` (figures illustrative until sourced, R5):

```yaml
name: face_analysis
display_name: Face Analysis (DeepFace)        # unchanged: names the method (R2)
title: "Faces: expression, age, gender"       # new: what it finds
summary: >-                                   # new: at most two sentences
  Finds faces in each sampled frame and estimates each face's expression, age and gender.
homepage: https://github.com/serengil/deepface
benchmarks:
  - name: WIDER FACE (hard)
    measure: AP
    figure: "91.4%"                           # quoted from the source, never computed
    source: https://arxiv.org/abs/1905.00641
    note: RetinaFace detector, ResNet-50 backbone
quality_note: null                            # VLM only: "Depends on the model you choose…"
citations: [...]                              # unchanged (spec 025)

config_schema:
  detect_emotions:
    type: boolean
    default: true
    description: Enable emotion recognition
    group: behavioural                        # new
    label: Expression                         # new
  max_faces:
    type: integer
    default: 10
    description: Maximum number of faces to detect per frame
    group: behavioural
    label: Most faces per frame
    unit: faces                               # new
    min: 1                                    # new
  scale_factor:
    type: float
    default: 1.1
    description: Scale factor for OpenCV Haar cascades
    group: technical
    label: Haar cascade scale factor
    applies_when:                             # new: shown only when this holds
      detection_backend: [opencv]
```

Parsing rules (`registry/pipeline_registry.py`):
- Wrong types are ignored with a warning, as for every optional field today.
- `group` other than `behavioural`/`technical` → `technical`, warning.
- A benchmark missing `name`, `measure`, `figure` or `source` is dropped, warning.
- `applies_when` naming a setting that doesn't exist is dropped, warning.
