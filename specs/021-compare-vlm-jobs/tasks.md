# Tasks: Compare Two VLM Jobs

- [x] T001 `GET /api/v1/jobs/{id}/video` (`api/v1/jobs.py`) + test; client `jobVideoUrl` with auth (blob URL)
- [x] T002 `viewer/src/lib/vlmLabels.ts` (`isPositiveLabel`, shared with VlmAnnotationPanel)
- [x] T003 `viewer/src/lib/vlmCompare.ts`: `pairRuns`, `summarize` (counts, rate, label pairs, vs ground truth), `comparisonCsv`; tests
- [x] T004 `viewer/src/pages/Compare.tsx`: load jobs + outputs + provenance; same-video check; timeline rows; disagreement list (filter: exactly one matches GT); moment detail + video seek; summary + label pairs; ignore-case toggle; ELAN file input; CSV download; route
- [x] T005 JobDetail: "Compare with…" (other completed VLM jobs on the same video; reruns and original first) and "Compare with original"
- [x] T006 Vitest for the page with mocked client; real check with two Ollama jobs; CHANGELOG, roadmap, bundle, suites, push
