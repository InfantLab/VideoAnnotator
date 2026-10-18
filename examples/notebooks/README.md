# Worked analysis notebooks

Two notebooks that start from VideoAnnotator's **results table**: every pipeline's results in one
CSV, one row per event and measure, the same columns for every pipeline. Both do the same small
analysis of the demo clip, whose table ships here (`demo_clip_table.csv`):

- who speaks, and for how long (speaker diarization)
- what was said (speech recognition)
- who is in view, when (person tracking)
- facial expression over time (face analysis, OpenFace action units)
- turning the long table wide, one row per face
- who is in view while someone speaks (joining two pipelines on time)
- what made the results (models and versions per row)

| Notebook | Needs |
|---|---|
| [`results_table_r.Rmd`](results_table_r.Rmd) | R ≥ 4.1 with `readr`, `dplyr`, `tidyr`, `ggplot2` (and `rmarkdown` to knit it). Open it in RStudio. |
| [`results_table_python.ipynb`](results_table_python.ipynb) | Python with `pandas` and `matplotlib`. Open it in Jupyter or VS Code. |

## Your own results

1. Export the table: in the viewer, a run's or job's page → **Export → Table (CSV)**; or from a
   terminal, `videoannotator export table "<your results folder>"` (no server needed).
2. Change the `TABLE` line at the top (`TABLE <- "demo_clip_table.csv"` / `TABLE = "demo_clip_table.csv"`) to
   your file.
3. Run it. Sections for pipelines you didn't run say so and move on.

The columns and every pipeline's measures are documented in
[`docs/usage/exporting_results.md`](../../docs/usage/exporting_results.md).

## About the demo table

It is the table export of the committed demo-clip outputs (`tests/fixtures/viewer_contract/`): a
10-second clip of a parent and infant playing peekaboo. The face-analysis and VLM files there come
from another take of the same session (see that folder's README), and were made before
VideoAnnotator recorded provenance, so their rows say `not recorded` for the model.
Regenerate it after the fixtures change with `python scripts/make_demo_table.py`; a test fails
while it is out of date.
