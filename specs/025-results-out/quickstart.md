# Quickstart: Results Out

How to check spec 025 end to end. Automated checks first, then a manual walk.

## Automated

```bash
pytest tests/unit/exporters/ tests/api/test_export_routes.py tests/unit/test_notebooks.py -q
pytest tests/unit/test_cli_export.py -q
cd viewer && bun run test:run   # contract test reads the exported .eaf fixtures
```

They cover: every fixture event appears in the table with matching times/labels/values (SC-002);
CSV round-trips through `csv` and pandas with awkward text; legacy (no provenance) files export with
`not recorded` (SC-007); Parquet equals CSV when pyarrow is present and errors helpfully when not;
`.eaf` validates against EAF 3.0 and matches the table (SC-004); methods revisions match provenance
and every citation key resolves (SC-005); the Python notebook runs (SC-006); API, CLI agree (FR-010).

## Manual walk

1. Offline, on the fixtures:

   ```bash
   cp -r tests/fixtures/viewer_contract /tmp/va_demo/demo_clip   # one video folder
   videoannotator export table /tmp/va_demo/demo_clip
   videoannotator export elan /tmp/va_demo/demo_clip
   videoannotator export methods /tmp/va_demo/demo_clip
   ```

   Expect a CSV with rows from all seven readers, one `.eaf`, and a methods paragraph naming whisper,
   pyannote, YOLO11, OpenCLIP and OpenFace with revisions, and DeepFace/VLM as "not recorded".

2. Open the CSV in R: `read.csv("/tmp/va_demo/demo_clip/demo_clip.csv")`; and in Python:
   `pandas.read_csv(...)`. Same row count; the transcript with a comma is one field.

3. Open the `.eaf` in ELAN 6.x: the tiers listed in `contracts/elan-and-methods.md` appear, the video
   loads if the path exists (else ELAN asks for it). Add a tier, save, drag the file into the viewer:
   both sets of tiers show on the timeline.

4. Server + viewer (`bash scripts/dev.sh`): run Scene Detection and Speech Recognition on two demo
   videos as one run. On the run's page: Export → Table (CSV) downloads one file covering both
   videos; Export → Methods paragraph shows prose and BibTeX with working Copy buttons; Export → ELAN
   downloads a zip of two `.eaf`. On a job page the same for one video. Parquet without the extra
   shows the install hint; after `pip install videoannotator[parquet]` it downloads.

5. Run `examples/notebooks/results_table_r.Rmd` in RStudio (or `Rscript -e
   'rmarkdown::render("examples/notebooks/results_table_r.Rmd")'`) and
   `results_table_python.ipynb` in Jupyter: both finish with figures.

## Walkthrough log

**2026-10-16, Linux (dev container), Python 3.13, all extras installed.**

- Step 1: on a copy of the fixtures, `export table` wrote 633 rows (1 video, 7 pipelines),
  `export elan` one `.eaf` (14 annotations, the per-detection measures listed as left out),
  `export methods` the paragraph with every recorded revision and DeepFace/VLM "not recorded".
- Step 2: `read.csv` in R 4.3.3 and `pandas.read_csv` both read 633 rows; the transcript cue with
  a comma is one field.
- Step 3 (ELAN itself): **not done**; no ELAN install here. The `.eaf` validates against
  EAF 3.0 and the viewer opens it and an ELAN-edited copy (contract test). Still to do by a person.
- Step 4: a real server (auth off, temporary database and results folder) ran the demo video twice
  as one run with scene detection, speech recognition and person tracking (both completed). Every
  route answered: run table (70 rows, 2 videos, `run` = the run folder's name, the second video
  `... 2`), Parquet, ELAN zip with one `.eaf` per video, methods ("All 2 videos ..."). In headless
  Chromium the run page's Export → Table (CSV) downloaded a file byte-identical to the API's, and
  the Methods dialog showed prose and BibTeX. Fixed on the way: the dialog clipped long lines
  (BibTeX widened its grid); person tracking recorded "weights file not found" on a first run
  because it hashed the weights before ultralytics downloaded them (now after loading, with a
  test). Not checked here: the Parquet install hint against a server without pyarrow (covered by
  API and viewer tests).
- Step 5: the R notebook rendered start to finish on R 4.3.3 (Ubuntu packages); the Python one runs
  in `tests/unit/test_notebooks.py`.
