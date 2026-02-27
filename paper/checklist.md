# JOSS Submission Checklist for VideoAnnotator

Based on https://joss.readthedocs.io/en/latest/submitting.html and
https://joss.readthedocs.io/en/latest/review_checklist.html (checked 2026-02-27).

---

## Paper (paper.md)

- [x] Word count within 750–1750 range (~1255 words)
- [x] Summary section — describes all 10 pipelines across 4 modalities
- [x] Statement of need section
- [x] State of the field section — covers ELAN, Datavyu, DeepLabCut, YOLO, Py-Feat, OpenFace, openSMILE, PySceneDetect, pyannote
- [x] Software design section — four architectural layers with trade-offs
- [x] Research impact statement section — GPI/Stellenbosch/Oxford context, pilot corpus
- [x] AI usage disclosure section — Copilot and Claude; code/test/docs scope; human-review assertion
- [x] Quality control section
- [x] Statement of limitations section
- [x] Acknowledgements with funding
- [x] YAML frontmatter (title, tags, authors, affiliations, date, bibliography)
- [x] Corresponding author marked (`corresponding: true`)
- [x] All 6 authors have ORCIDs

## Bibliography (paper.bib)

- [x] Correct entry types (`@inproceedings`, `@article`, `@software`, `@book`)
- [x] DOIs present where available
- [x] Full venue names (not abbreviated)
- [x] 17 references covering all cited tools and upstream models
- [x] Companion project (Video Annotation Viewer) cited

## Repository & Metadata

- [x] OSI-approved license (MIT) with plain-text LICENSE file
- [x] Open repository on GitHub (InfantLab/VideoAnnotator)
- [x] Public issue tracker (GitHub Issues)
- [x] 6+ months public history (Sep 2023 – present, 2.5 years, 236 commits)
- [x] Multiple releases (v0.5alpha through v1.4.1, 8 tags)
- [x] CITATION.cff present, matches paper metadata, all ORCIDs included
- [x] Dependency management (pyproject.toml + uv.lock)

## Documentation

- [x] Installation docs (uv, Docker, DevContainer)
- [x] Usage examples (examples/ directory, 7+ scripts)
- [x] API and usage documentation (docs/ directory)
- [x] README with badges, quick start, architecture, pipeline tables
- [x] Docker support (CPU/GPU/Dev Dockerfiles + compose)

## Testing & CI

- [x] Automated tests (pytest, 74 test files, ~94% pass rate)
- [x] CI/CD pipeline (GitHub Actions: test matrix, lint, type check, security scan, build)

## Community Guidelines

- [x] CONTRIBUTING.md (476 lines — setup, style, PR process, release process)
- [x] CODE_OF_CONDUCT.md (Contributor Covenant v2.1 + research ethics)
- [x] SECURITY.md (disclosure process, response timeline, compliance)

---

## Before submission

- [x] v1.4.1 git tag created locally
- [ ] Push tag to remote: `git push origin v1.4.1`
- [ ] Create GitHub Release for v1.4.1
- [x] Improve git contributor attribution (only 1 contributor visible in history)

## After acceptance

- [ ] Create a Zenodo archive and obtain a DOI for the archived version
- [ ] Update the JOSS review issue with the version number and archive DOI
- [ ] Ensure the GitHub Release matches the tagged version
