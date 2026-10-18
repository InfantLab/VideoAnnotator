# Contract: CLI

## `videoannotator pipelines describe <name> [--json]`

Reads the local registry; needs no server (R8). Exit 1 with "Unknown pipeline '<name>'. Run
`videoannotator pipelines list`." for an unknown name.

Text output:

```
Faces: expression, age, gender  (face_analysis)
Finds faces in each sampled frame and estimates each face's expression, age and gender.

What to measure
  detect_emotions   Expression               true
  detect_age        Age                      true
  detect_gender     Gender                   true
  max_faces         Most faces per frame     10 faces (min 1)
  min_face_size     Smallest face            30 px

Technical (7)
  detection_backend Face detector            deepface
  ...
  scale_factor      Haar cascade scale       1.1   only when detection_backend is opencv

About
  Homepage   https://github.com/serengil/deepface
  Benchmark  WIDER FACE (hard), AP 91.4% (RetinaFace detector, ResNet-50 backbone)
             https://arxiv.org/abs/1905.00641
  Cite       serengil2026deepface, deng2020retinaface
             (--json includes the BibTeX)
```

A pipeline with `benchmarks: []` prints "Benchmark  No published benchmark"; with `quality_note`,
prints the note instead. Deprecated pipelines print their deprecation notice first.

`--json`: the same object as `GET /api/v1/pipelines/{name}` for the guide fields (title, summary,
homepage, benchmarks, quality_note, citations, config_schema with the new keys).

## `videoannotator pipelines list`

Table and markdown output gain a "Title" column after "Name" when the server returns `title`;
`--json` passes the new fields through unchanged.

## Tests

`tests/unit/cli/test_cli_pipelines_describe.py` (Typer runner): text and JSON for a built-in, the
unknown-name exit, a stub without guide fields (falls back to display name and description), the
`applies_when` note.
