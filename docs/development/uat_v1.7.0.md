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
| 2 | 2026-10-26 | Videos, filtered to a run (run header) | "started 26/10/2026, 17:50:44 · finished 26 Oct": finish has a date but no time, and the two use different formats | Open |
| 3 | 2026-10-26 | Videos, filtered to a run | Not clear how to view the results: the header offers Download, ELAN, Methods, Compare answers, but nothing says "open the video to see them" (the video name is the way in, and doesn't look like it) | Open |
| 4 | 2026-10-26 | Videos, filtered to a run | Not clear how the run box (top) and the videos table (bottom) relate: nothing says the table is "the videos in this run", and the page intro above still describes all videos | Open |
| 5 | 2026-10-26 | Videos table (seen in the #2 screenshot) | Two columns both headed "Faces" (likely face_analysis and the OpenFace 3 embedding), and an "Audio processing" column beside "Speaker turns" and "Speech"; empty "Projects" column takes space | Open (noticed, not reported) |
