## v1.4.2 — JOSS Review Version

This release accompanies the submission of VideoAnnotator to the [Journal of Open Source Software](https://joss.theoj.org/). A parallel submission is being made for the companion project [Video Annotation Viewer](https://github.com/InfantLab/video-annotation-viewer).

Below is a summary of all changes across the v1.4.x release series (v1.4.0 -- v1.4.2).

---

### Added (v1.4.0)

- **Flexible storage**: New artifact download capabilities, including source video retrieval.
- **Database-backed authentication**: Migrated from file-based to database-backed auth for improved security and scalability.
- **Artifacts API**: New endpoint `GET /api/v1/jobs/{id}/artifacts` to download job results as a ZIP archive.
- **Container tooling**: Baked `hadolint` into Docker images and devcontainer; added `git-lfs` to CPU/GPU Dockerfiles.

### Changed

- **CLIP migration** (v1.4.2): Migrated scene-classification pipeline from `clip` to `open_clip`, using the LAION-2B pretrained `ViT-B-32` model for improved availability and reproducibility.
- **HuggingFace auth** (v1.4.2): Updated diarization and Whisper pipelines to use the current `token` parameter instead of the deprecated `use_auth_token`.
- **JOSS manuscript** (v1.4.1): Consolidated the paper into `paper/paper.md` and replaced `docs/joss.md` with a pointer to avoid divergence.
- **Repository hygiene** (v1.4.1): Moved top-level helper scripts into organized subfolders under `scripts/`; updated imports to the `videoannotator.*` namespace.
- **Entrypoints** (v1.4.1): `api_server.py` now acts as a compatibility wrapper; documentation recommends the `videoannotator` CLI.
- **README** (v1.4.1): Rationalized setup/install instructions, fixed broken links, replaced hard-coded test/coverage claims with CI status badges.
- **Devcontainer** (v1.4.2): Simplified forwarded-port list to the single default API port (18011).

### Fixed

- **Artifact downloads** (v1.4.0): Ensured source video files are included in the downloaded artifact ZIP.
- **Database GUID handling** (v1.4.2): Added defensive error handling in the `GUID` type decorator for malformed UUID values.
- **Diarization init** (v1.4.2): Wrapped model loading in explicit error handling with a clear log message on failure.
- **Docs** (v1.4.1): Standardized examples on canonical API port `18011` and corrected Docker run port mappings.
- **Docs** (v1.4.1): Replaced placeholder `docs/usage/accessing_results.md` with a real results retrieval guide.

### Removed (v1.4.2)

- **Voice emotion baseline**: Removed `voice_emotion_baseline` pipeline metadata and associated tests (superseded by LAION EmoNet voice pipeline).

### Documentation (v1.4.2)

- Added JOSS cover letter (`paper/cover_letter.md`).
- Updated paper bibliography and CITATION.cff to v1.4.2.

---

**Full Changelog**: https://github.com/InfantLab/VideoAnnotator/compare/v1.3.0...v1.4.2
