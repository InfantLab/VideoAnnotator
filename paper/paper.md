---
title: "VideoAnnotator and Video Annotation Viewer: an open toolkit for automated multi-modal video annotation and cross-modal audit in behavioral research"
tags:
  - Python
  - TypeScript
  - video analysis
  - behavioral science
  - reproducibility
  - machine learning
authors:
  - name: Caspar Addyman
    orcid: 0000-0003-0001-9548
    corresponding: true
    affiliation: 1
  - name: Jeremiah Ishaya
    orcid: 0000-0002-9014-9372
    affiliation: 1
  - name: Irene Uwerikowe
    orcid: 0000-0002-1293-7349
    affiliation: 1
  - name: Daniel Stamate
    orcid: 0000-0001-8565-6890
    affiliation: 2
  - name: Jamie Lachman
    orcid: 0000-0001-9475-9218
    affiliation: 3
  - name: Mark Tomlinson
    orcid: 0000-0001-5846-3444
    affiliation: 1
affiliations:
  - name: Institute for Life Course Health Research (ILCHR), Stellenbosch University, South Africa
    index: 1
  - name: Department of Computing, Goldsmiths, University of London, United Kingdom
    index: 2
  - name: Department of Social Policy and Intervention (DISP), University of Oxford, United Kingdom
    index: 3
date: 2 July 2026
bibliography: paper.bib
---

# Summary

We present two companion open-source tools for reproducible video-based behavioral research: **VideoAnnotator**, a Python pipeline toolkit, and **Video Annotation Viewer** (VAV), a browser-based audit interface. In this integrated workflow, VideoAnnotator is the processing engine and VAV is the review surface. Together they cover the full analysis loop, from raw video to cross-modal inspection of automated outputs, while keeping all processing local to meet data-governance requirements common in research involving children or other vulnerable populations.

**VideoAnnotator** provides ten declaratively configured pipeline specifications spanning four modalities: person tracking via YOLOv11 with ByteTrack [@yolo11; @bytetrack]; facial analysis using DeepFace [@deepface], LAION EmoNet face emotion [@emonet_face], and OpenFace 3 [@openface3]; scene detection with PySceneDetect and CLIP-based labelling [@pyscenedetect; @clip]; and audio processing comprising Whisper speech recognition [@whisper], pyannote speaker diarization [@pyannote], and LAION EmoNet voice emotion [@emonet_voice]. Pipelines share a uniform interface behind a local-first FastAPI service [@fastapi] with Docker images for reproducible CPU and GPU execution. Outputs are standardized to established formats (COCO JSON, RTTM, WebVTT) and wrapped with provenance metadata recording the pipeline version and configuration.

**Video Annotation Viewer** is a React/TypeScript single-page application that renders those outputs both as overlays on the source video and as a synchronized, multi-track timeline beneath the source video. Each annotation stream (pose skeletons, speech captions, speaker diarization, scene boundaries) appears as a parallel, time-aligned lane, allowing researchers to audit cross-modal relationships at a glance: where a speaker label disagrees with who is on screen, where pose tracking drops out, or where turn-taking patterns emerge in caregiver–child interaction. The viewer reads standard files directly (COCO JSON, WebVTT, RTTM), and can also connect to a running VideoAnnotator service for job submission and results retrieval. Standalone use requires no server.

# Statement of need

Behavioral and interaction research depends on observational video, yet human annotation is costly, slow, and difficult to reproduce across sites [@observer]. Automated pipelines can scale, but they introduce a new problem: researchers must be able to *inspect and trust* what each detector produced before drawing scientific conclusions, especially when constructs are subjective, context-dependent, or concern vulnerable populations.

The current landscape makes this harder than it should be. Each upstream model — face detectors, speech recognizers, pose estimators — has its own installation procedure, input expectations, and output format, so building a multi-modal analysis workflow requires substantial per-lab engineering. Once outputs exist, there is no lightweight tool for reviewing heterogeneous annotation streams together in time: seeing whether the transcribed speech matches the on-screen speaker, or whether turn-taking pauses align with detected gaze changes, requires stitching together tools that were not designed to interoperate.

VideoAnnotator addresses the pipeline integration gap; Video Annotation Viewer addresses the cross-modal inspection gap. Critically, both run entirely locally, supporting the data-privacy requirements of research involving children, patients, and recordings collected in low- and middle-income settings.

# State of the field

**Manual annotation platforms** such as ELAN [@elan], Datavyu [@datavyu], and BORIS [@boris] provide rich time-aligned coding environments but require trained human coders and do not scale to large video corpora. They are built around the assumption that a human is the annotation source, not a consumer of machine output.

**Specialized computer-vision libraries** such as DeepLabCut [@deeplabcut] and YOLO [@yolo11] offer state-of-the-art pose estimation and detection but address a single modality and leave output standardization and batch orchestration to the user. For facial affect, Py-Feat [@pyfeat] and OpenFace [@openface3] extract action units and emotion labels; for audio, openSMILE [@opensmile] provides acoustic feature extraction. These are powerful tools, but each produces its own schema and none orchestrates across modalities.

**General scientific visualization environments** such as napari [@napari] and FIJI/ImageJ [@imagej] support extensible, multi-channel display of time-series data and are widely used in bio-imaging. They are capable tools, but their plugin ecosystems are built for image and microscopy data: they do not natively parse audio or speech-specific formats (WebVTT, RTTM). Their design assumes an interactive editing workflow rather than a read-only audit of machine-generated output.

**Dataset curation platforms** such as FiftyOne [@fiftyone] and Label Studio [@labelstudio] are powerful for managing datasets and producing annotations at scale, but they are optimized for labeling pipelines rather than lightweight time-synchronized review of heterogeneous outputs.

As researchers studying parent–child interaction we needed person tracking, facial expression analysis, speech segmentation, and speaker diarization on the same set of videos. Using existing tools required ad-hoc use of multiple libraries that had no shared output format, no batch orchestration, and no unified review surface. VideoAnnotator and Video Annotation Viewer were built to close that gap, and are shared to benefit other behavioral researchers with similar requirements.

# Software design

## VideoAnnotator

VideoAnnotator is organized around four layers.

**Pipeline registry.** Pipelines are registered via declarative YAML metadata files. Adding a new detector requires only a metadata file and a Python class inheriting from `BasePipeline`, which enforces a uniform interface (`initialize()`, `process()`, `cleanup()`, `get_schema()`). This metadata-driven approach allows plug-and-play composition without modifying existing pipelines.

**Standardized output and provenance.** All outputs map to established formats: COCO JSON for spatial annotations, RTTM for diarization, WebVTT for timed text. Each result is wrapped with provenance metadata — pipeline name, version, configuration parameters, processing timestamps — supporting reproducible re-analysis.

**Batch orchestration.** A thread-pool orchestrator manages concurrent job execution with configurable retry strategies (fixed, linear, or exponential backoff) and transient-versus-permanent error classification. Jobs are checkpointed to the local filesystem after each stage, providing recovery without an external message queue.

**Service and CLI.** A FastAPI service exposes endpoints for job submission, status polling, pipeline discovery, and results retrieval; a Typer-based CLI provides equivalent local-execution commands. Both share the same orchestrator and storage backend.

## Video Annotation Viewer

VAV is a stateless React/TypeScript single-page application organized around three concerns. **Format parsers** validate and normalize COCO JSON, WebVTT, and RTTM files into internal typed representations. **Viewer components** render a video player with overlay layer and a coordinated multi-track timeline beneath it, with each annotation stream displayed as a parallel lane synchronized to playback. **An optional API client** connects to a VideoAnnotator service for job management and result retrieval, but is not required for standalone use. Because VAV is stateless, a given video and set of annotation files deterministically produce the same display, supporting reproducible review (\autoref{fig:viewer}).

![Video Annotation Viewer displaying a caregiver–child interaction clip with synchronized person tracking, face detection, emotion recognition, speech recognition, speaker diarization, and scene detection overlays, alongside the multi-track timeline that enables cross-modal audit.\label{fig:viewer}](figure1.png)

# Research impact statement

Both tools were developed at Stellenbosch University to support analysis of caregiver–child interaction videos collected across multiple sites in sub-Saharan Africa as part of the Global Parenting Initiative. Researchers at Stellenbosch University and Goldsmiths, University of London use the full pipeline — VideoAnnotator for processing and VAV for review — in developmental psychology and parenting-intervention studies.  The local-first design was motivated by the ethical and governance requirements of working with video recordings of children in low- and middle-income settings. Pilot analyses have been conducted on corpora of over 500 sessions, and the software is being prepared for use in upcoming multi-site trials. A practical motivating example: VAV's synchronized multi-track timeline makes turn-taking patterns in caregiver–child interaction visible at a glance, a cue to reciprocity that is important for developmental assessment but difficult to judge from tabular outputs alone.

# Quality control

VideoAnnotator maintains a pytest-based test suite covering unit, integration, and performance tests across 74 test files. Continuous integration via GitHub Actions runs tests on Ubuntu, Windows, and macOS with Python 3.12, alongside ruff linting, mypy type checking, and Trivy security scanning. Docker images provide consistent execution environments. Video Annotation Viewer is tested with Vitest and React Testing Library and built via a typed TypeScript compilation step that catches interface errors at build time.

# Statement of limitations

Annotation quality is bounded by the accuracy of upstream detectors and their generalization to a given domain, population, and recording context. VideoAnnotator is a monolithic container application designed for single-machine execution and does not currently support distributed scheduling; a GPU is recommended for long videos. A more modular plugin-based design is planned for a future release. Ethical deployment — consent, data governance, and redaction — remains the responsibility of adopters. Video Annotation Viewer is a review tool and does not support editing or export of corrected annotations.

# AI usage disclosure

Generative AI tools (GitHub Copilot and Claude, Anthropic) were used during development for code scaffolding, test generation, and documentation drafting. All AI-assisted outputs were reviewed, edited, and validated by the human authors, who made all core design decisions. Architectural decisions — including the pipeline registry design, the provenance-metadata format, and the modularity plan described in this paper's Software design section — were specified by the human authors as versioned, dated design documents *before* implementation (e.g. `specs/003-modular-pipeline-architecture/spec.md` and the `docs/development/roadmap_*.md` series in the repository), giving reviewers a reviewable record of design intent that is separate from, and predates, any AI-assisted code generation. No AI tools were used to generate the scientific content or analysis reported in this manuscript. All authors take full responsibility for the software and this paper.

# Acknowledgements

We acknowledge the open-source communities behind the upstream models integrated by VideoAnnotator, in particular OpenFace, Whisper, pyannote, Ultralytics YOLO, LAION, and PySceneDetect.

This project was supported by the Global Parenting Initiative and funded by The LEGO Foundation.

# References
