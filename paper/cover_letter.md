**Cover letter — JOSS submission**
**VideoAnnotator: an extensible, reproducible toolkit for automated video annotation in behavioral research**

Dear Editor,

We are pleased to submit *VideoAnnotator* for consideration by the Journal of Open Source Software. This is an open-source Python toolkit that provides a unified, locally deployed framework for automated multi-modal video annotation — covering person tracking, facial analysis, scene detection, and audio processing — aimed at behavioral, social, and health researchers.

We believe the submission addresses the JOSS review criteria as follows:

- **Open license**: The software is released under the MIT license.
- **Repository and archival**: The source is hosted on GitHub at [InfantLab/VideoAnnotator](https://github.com/InfantLab/VideoAnnotator) and we will generate a versioned Zenodo DOI upon acceptance.
- **Contribution and community guidelines**: The repository includes `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue templates, and `CITATION.cff`.
- **Automated tests and CI**: A pytest suite of 74 test files (unit, integration, and performance) runs via GitHub Actions on Ubuntu, Windows, and macOS with Python 3.12, alongside ruff linting, mypy type-checking, Trivy security scanning, and Codecov reporting.
- **Functionality documentation**: Full API documentation and usage guides are provided in the repository and rendered online.
- **Statement of need and state of the field**: The paper includes both sections, positioning VideoAnnotator relative to existing tools (ELAN, Datavyu, DeepLabCut, Py-Feat, OpenFace, openSMILE, PySceneDetect, pyannote) and explaining the gap it fills.
- **References**: All key upstream models and comparable tools are cited in the bibliography.
- **Research application**: The paper includes a research-impact statement describing current use at Stellenbosch University and the University of Oxford for caregiver–child interaction studies under the Global Parenting Initiative.
- **AI disclosure**: Included per JOSS policy.

We would also like to note that a parallel JOSS submission is being prepared for the companion project, **Video Annotation Viewer** ([InfantLab/video-annotation-viewer](https://github.com/InfantLab/video-annotation-viewer)), which provides the interactive browser-based interface for reviewing and validating VideoAnnotator outputs. The two packages are designed to work together but are independently installable and have distinct codebases.

This is our first submission to JOSS, so we very much appreciate any guidance you can offer throughout the review process. We are happy to address any feedback promptly.

Thank you for your time and consideration.

Sincerely,

Caspar Addyman (corresponding author), Jeremiah Ishaya, Irene Uwerikowe, Daniel Stamate, Jamie Lachman, and Mark Tomlinson
