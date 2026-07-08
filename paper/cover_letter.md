**Cover letter — revised combined JOSS submission**
**VideoAnnotator and Video Annotation Viewer: an open toolkit for automated multi-modal video annotation and cross-modal audit in behavioral research**

Dear Editor,

Following the pre-review discussion in issues #10182 and #10183, we are resubmitting VideoAnnotator and Video Annotation Viewer as a single combined paper, as agreed with the handling editor (@sneakers-the-rat). The two repositories remain separate — [InfantLab/VideoAnnotator](https://github.com/InfantLab/VideoAnnotator) and [InfantLab/video-annotation-viewer](https://github.com/InfantLab/video-annotation-viewer) — and the paper is hosted in the VideoAnnotator repository (issue #10182).

The revised paper addresses the specific concerns raised during pre-review:

- **Combined scope**: The paper now covers the full research workflow — pipeline execution (VideoAnnotator) through to cross-modal inspection of outputs (Video Annotation Viewer) — which is the contribution that neither tool adequately represents alone.
- **State of the field expanded**: We now compare directly to napari and FIJI/ImageJ (general scientific visualization tools), BORIS (behavioral coding), ELAN and Datavyu (manual annotation), Py-Feat, OpenFace, and openSMILE (single-modality detectors), and FiftyOne and Label Studio (dataset curation platforms), with a clear explanation of what each handles well and where our tools address a remaining gap.
- **Core contribution reframed**: The central contribution of Video Annotation Viewer is the synchronized multi-track timeline for cross-modal audit — seeing, for example, where a diarization speaker label disagrees with who is on screen, or where turn-taking pauses align with gaze changes. This is the use case that motivated the tool and that existing platforms do not address well for behavioral scientists working with speech and diarization formats (WebVTT, RTTM).
- **Windows CI bug fixed**: The SQLite engine disposal bug flagged during pre-review (WinError 32 on teardown) has been corrected as an implementation fix in the storage backend, not merely a test workaround. `StorageBackend` now has a `close()` lifecycle method; `SQLiteStorageBackend` calls `engine.dispose()` on close; and `reset_storage_backend()` calls `close()` before clearing the cached instance.

We believe the submission now addresses the JOSS review criteria as follows:

- **Open license**: Both repositories are released under the MIT license.
- **Repositories**: VideoAnnotator at [InfantLab/VideoAnnotator](https://github.com/InfantLab/VideoAnnotator); Video Annotation Viewer at [InfantLab/video-annotation-viewer](https://github.com/InfantLab/video-annotation-viewer). Zenodo DOIs will be generated upon acceptance.
- **Community guidelines**: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue templates, and `CITATION.cff` are present in both repositories.
- **Automated tests and CI**: VideoAnnotator has a pytest suite (74 test files, unit/integration/performance) running on Ubuntu, Windows, and macOS via GitHub Actions, with ruff, mypy, and Trivy. Video Annotation Viewer has Vitest/React Testing Library tests with a typed TypeScript build step.
- **Documentation**: API documentation, usage guides, and a reviewer getting-started guide are provided in both repositories.
- **Research application**: Both tools are in active use at Stellenbosch University and the University of Oxford for caregiver–child interaction studies under the Global Parenting Initiative, with pilot analyses on corpora of over 500 sessions.
- **AI disclosure**: Included per JOSS policy.

Thank you for the careful pre-review feedback — it has substantially improved the submission.

Sincerely,

Caspar Addyman (corresponding author), Jeremiah Ishaya, Irene Uwerikowe, Daniel Stamate, Jamie Lachman, and Mark Tomlinson
