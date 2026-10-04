# Tasks: Compare Two VLM Jobs

- [ ] T001 `GET /api/v1/jobs/{id}/video` (`api/v1/jobs.py`) + test; client `jobVideoUrl` with auth (blob URL)
- [ ] T002 `viewer/src/lib/vlmLabels.ts` (`isPositiveLabel`, shared with VlmAnnotationPanel)
- [ ] T003 `viewer/src/lib/vlmCompare.ts`: `pairRuns`, `summarize` (counts, rate, label pairs, vs ground truth), `comparisonCsv`; tests
- [ ] T004 `viewer/src/pages/Compare.tsx`: load jobs + outputs + provenance; same-video check; timeline rows; disagreement list (filter: exactly one matches GT); moment detail + video seek; summary + label pairs; ignore-case toggle; ELAN file input; CSV download; route
- [ ] T005 JobDetail: "Compare with…" (other completed VLM jobs on the same video; reruns and original first) and "Compare with original"
- [ ] T006 Vitest for the page with mocked client; real check with two Ollama jobs; CHANGELOG, roadmap, bundle, suites, push
