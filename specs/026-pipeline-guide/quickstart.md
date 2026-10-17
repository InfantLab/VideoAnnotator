# Quickstart: checking the pipeline guide

## 1. The metadata check

```bash
pytest tests/unit/registry/test_pipeline_guide.py -q
```

Passes when every built-in pipeline has a title, a summary of at most two sentences, a homepage, a
`benchmarks` key, every setting grouped and labelled, at most five behavioural settings, valid
`applies_when` references, and no setting the code doesn't read (SC-001, SC-002).

## 2. The CLI, no server

```bash
videoannotator pipelines describe face_analysis
videoannotator pipelines describe vlm_annotation          # quality note instead of a figure
videoannotator pipelines describe face_analysis --json | jq '.benchmarks'
videoannotator pipelines describe nonexistent             # exit 1, points to `pipelines list`
```

## 3. The API

```bash
curl -s http://127.0.0.1:18011/api/v1/pipelines/face_analysis | jq '{title, summary, homepage, benchmarks}'
curl -s http://127.0.0.1:18011/api/v1/pipelines/face_analysis/schema | jq '.parameters[] | {name, group, label, applies_when}'
```

## 4. The viewer

`bash scripts/dev.sh`, then:

1. **Pipelines page**: every pipeline shows its title and summary; "What to measure" is open,
   "Technical settings (n · all at defaults)" is collapsed; About shows homepage, benchmarks with
   links (or "No published benchmark"), and Copy BibTeX.
2. **New job → pipelines step**: the same title and summary for each pipeline.
3. **Face analysis settings**: `scale_factor` and `min_neighbors` are not shown; set the detector
   to `opencv` and they appear; set the scale factor to 1.2, switch the detector back: it stays,
   marked "Not used with these settings".
4. Change `max_faces`: nothing in Technical changes; change `confidence_threshold`: the heading says
   "1 changed".

## 5. Consistency (SC-004)

```bash
pytest tests/unit/registry/test_pipeline_guide.py::test_api_cli_agree -q
```

The API's guide fields and `describe --json` match the metadata for every built-in pipeline.

## 6. Generated doc

```bash
uv run python scripts/generate_pipeline_specs.py && git diff --stat docs/pipelines_spec.md
pytest tests/unit/docs/test_pipeline_spec_doc.py -q
```
