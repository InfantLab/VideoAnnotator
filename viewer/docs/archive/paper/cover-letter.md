# Cover letter — JOSS submission

**Video Annotation Viewer: an interactive browser-based tool for reviewing multi-modal video annotations in behavioral research**

Dear Editor,

We are pleased to submit *Video Annotation Viewer* for consideration by the Journal of Open Source Software. This is an open-source browser-based application (React/TypeScript) that provides an interactive interface for visualizing, reviewing, and validating multi-modal video annotation data — including person tracking overlays, facial analysis, speech recognition subtitles, speaker diarization, and scene detection — aimed at behavioral, social, and health researchers.

We believe the submission addresses the JOSS review criteria as follows:

- **Open license**: The software is released under the MIT license.
- **Repository and archival**: The source is hosted on GitHub at [InfantLab/video-annotation-viewer](https://github.com/InfantLab/video-annotation-viewer) and we will generate a versioned Zenodo DOI upon acceptance.
- **Contribution and community guidelines**: The repository includes `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue templates, and `CITATION.cff`.
- **Automated tests and CI**: A Vitest suite of 374 tests (unit and integration) runs via GitHub Actions, alongside TypeScript type-checking and ESLint linting.
- **Functionality documentation**: Usage guides, a developer guide, and a Getting Started page are provided in the repository and within the application itself.
- **Statement of need and state of the field**: The paper includes both sections, positioning Video Annotation Viewer relative to existing annotation review tools (ELAN, Datavyu, ANVIL, VIA, VoTT) and explaining the gap it fills as a modern, zero-install viewer purpose-built for multi-modal machine-generated annotations.
- **References**: All key upstream tools and comparable software are cited in the bibliography.
- **Research application**: The paper includes a research-impact statement describing current use at Stellenbosch University and the University of Oxford for caregiver–child interaction studies under the Global Parenting Initiative.
- **AI disclosure**: Included per JOSS policy.

We would like to note that this submission is being made in parallel with the companion project, **VideoAnnotator** ([InfantLab/VideoAnnotator](https://github.com/InfantLab/VideoAnnotator)), which provides the automated annotation processing pipeline whose outputs this viewer is designed to display. The two packages are designed to work together but are independently installable, have distinct codebases, and serve different roles: VideoAnnotator generates annotations; Video Annotation Viewer enables researchers to interactively review and validate them. We believe each meets the JOSS threshold for a distinct scholarly contribution.

This is our first submission to JOSS, so we very much appreciate any guidance you can offer throughout the review process. We are happy to address any feedback promptly.

Thank you for your time and consideration.

Sincerely,

Caspar Addyman (corresponding author), Jeremiah Ishaya, Irene Uwerikowe, Daniel Stamate, Jamie Lachman, and Mark Tomlinson
