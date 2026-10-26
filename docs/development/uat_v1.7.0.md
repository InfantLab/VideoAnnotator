# v1.7.0 UAT and Refinement

Started 2026-10-26, on `1.7-dev` after specs 027–029 and 033 were built
([roadmap](roadmap_v1.7.0.md), Phase 1). Caspar uses the built viewer end to end; what he finds is
fixed here as refinement, not as new specs. One line per finding: what was seen, then what was done.

Walk-throughs still open, done as part of this: 027 T068 (Ask AI with a real vision model), 028
T062 (Video page on the demo clip and a large video), 033 T055 (real model servers).

## Findings

| # | Date | Where | Finding | Status |
|---|---|---|---|---|
| 1 | 2026-10-26 | Runs (old Jobs page) | Run cards said nothing useful ("3 videos — Sep 26": no pipelines, videos or outcome); progress bars on finished runs; a separate, different "Individual jobs" table | Page seen on a viewer from before 029. 029 retired it (`/jobs` → Videos), old jobs show as one-video runs. Still open: run labels everywhere (project "Runs:" line, activity indicator, run header) should say pipelines and outcome, not just a date; no list of past runs outside a project |
