# JOSS Submission Checklist — Video Annotation Viewer

Based on a full review of the [submission guidelines](https://joss.readthedocs.io/en/latest/submitting.html),
[review checklist](https://joss.readthedocs.io/en/latest/review_checklist.html),
and [review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html).

Last updated: 2026-02-27

---

## Pending — Requires user action

- [ ] **New Figure 1** — Capture a new screenshot of the current interface to replace `paper/figure1.png`
- [x] **Co-author ORCIDs** — All 6 authors now have ORCIDs in `paper/paper.md` and `CITATION.cff`
- [ ] **Tagged release + Zenodo archive** — Tag `v0.6.1`, push, confirm Zenodo archive is current

---

## Done — Paper content

- [x] AI Usage Disclosure section added to paper
- [x] State of the Field split into its own section with build-vs-contribute justification
- [x] Research Impact Statement added (GPI programme, Stellenbosch University usage)
- [x] Figure added (`paper/figure1.png` with caption and label)
- [x] Self-citation `@vav` removed (was circular)
- [x] RTTM reference added to bib (`@rttm2009`, NIST RT-09)
- [x] Bibliography improved (author/year added to `@videoannotator`)
- [x] Paper date updated to 27 February 2026

## Done — Repository & metadata

- [x] CODE_OF_CONDUCT.md — Contributor Covenant v2.1
- [x] CONTRIBUTING.md — Dev setup, workflow, standards, testing, support
- [x] CITATION.cff — All 6 authors listed, bumped to CFF v1.2.0
- [x] Version bumped to 0.6.1 (`package.json`, `CITATION.cff`, README)
- [x] README Vite badge corrected (5 → 7)

## Done — Already compliant at review time

- [x] OSI-approved license — MIT, plaintext `LICENSE`
- [x] Public repo, browsable, clonable — GitHub
- [x] Open issue tracker — GitHub Issues
- [x] Installation instructions — README with clone/install/dev
- [x] Automated test suite — 20+ test files, Vitest + Playwright
- [x] CI/CD — GitHub Actions (lint, unit tests, e2e, Lighthouse)
- [x] Code coverage — Codecov integration
- [x] Documentation — README + `docs/` with 8+ guides
- [x] Usage examples — Demo mode, drag-and-drop, format samples
- [x] Dependency management — `package.json` with bun/npm
- [x] Summary section in paper
- [x] Statement of Need in paper
- [x] Software Design section ("Design and architecture")
- [x] Acknowledgements — Paper and README (LEGO Foundation)
- [x] Semantic versioning + CHANGELOG
- [x] Sustained development history — v0.1.0 through v0.6.1
- [x] Contact/support info — Email + GitHub Issues
- [x] Paper length — ~1200 words, within JOSS norms
