# Manual check: the R notebook runs (spec 025, FR-035)

CI has no R, so before each release run the R notebook once on a clean R install.

1. Install R ≥ 4.1 and the packages: `install.packages(c("readr", "dplyr", "tidyr", "ggplot2",
   "rmarkdown"))` (on Ubuntu, `apt install r-base-core r-cran-readr r-cran-dplyr r-cran-tidyr
   r-cran-ggplot2 r-cran-rmarkdown pandoc` is quicker).
2. Render it in a scratch copy, so no output lands in the repository:

   ```bash
   tmp=$(mktemp -d)
   cp examples/notebooks/results_table_r.Rmd examples/notebooks/demo_clip_table.csv "$tmp"
   (cd "$tmp" && Rscript -e 'rmarkdown::render("results_table_r.Rmd", quiet = TRUE)')
   ```

3. Open `$tmp/results_table_r.html`: every section shows output, with two figures (detections per
   tracked person, OpenFace action units) and no errors.

Or with R installed, `CI_HAS_R=1 pytest tests/unit/test_notebooks.py -k r_notebook_runs`.

## Log

| Date | R | Result |
|---|---|---|
| 2026-10-16 | 4.3.3 (Ubuntu 24.04 packages: dplyr 1.1.4, readr 2.1.5, tidyr 1.3.1, ggplot2 3.4.4, rmarkdown 2.25) | Ran start to finish; both figures drawn |
