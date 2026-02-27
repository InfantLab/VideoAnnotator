---
title: "VideoAnnotator: an extensible, reproducible toolkit for automated video annotation in behavioral research"
tags:
  - Python
  - video analysis
  - behavioral science
  - reproducibility
  - machine learning
authors:
  - name: Caspar Addyman
    orcid: https://orcid.org/0000-0003-0001-9548
    corresponding: true
    affiliation: 1
  - name: Jeremiah Ishaya
    orcid: https://orcid.org/0000-0002-9014-9372
    affiliation: 1
  - name: Irene Uwerikowe
    orcid: https://orcid.org/0000-0002-1293-7349
    affiliation: 1
  - name: Daniel Stamate
    orcid: https://orcid.org/0000-0001-8565-6890
    affiliation: 2
  - name: Jamie Lachman
    orcid: https://orcid.org/0000-0001-9475-9218
    affiliation: 3
  - name: Mark Tomlinson
    orcid: https://orcid.org/0000-0001-5846-3444
    affiliation: 1
affiliations:
  - name: Institute for Life Course Health Research (ILCHR), Stellenbosch University, South Africa
    index: 1
  - name: Department of Computing, Goldsmiths, University of London, United Kingdom
    index: 2
  - name: Department of Social Policy and Intervention (DISP), University of Oxford, United Kingdom
    index: 3
date: 18 December 2025
bibliography: paper.bib
---

# Summary

VideoAnnotator is an open-source Python toolkit for automated video annotation, designed for behavioral, social, and health research at scale. It ships ten declaratively configured pipelines spanning four modalities: person tracking via YOLOv11 with ByteTrack [@yolo11]; facial analysis using DeepFace, LAION CLIP face embeddings, and OpenFace 3 [@openface2]; scene detection with PySceneDetect and CLIP-based labelling [@pyscenedetect]; and audio processing comprising Whisper speech recognition [@whisper], pyannote speaker diarization [@pyannote], and LAION empathic voice emotion analysis. All pipelines share a uniform interface behind a local-first FastAPI service [@fastapi], with Docker images for consistent CPU and GPU execution. Outputs are standardized to established formats (COCO JSON, RTTM, WebVTT) and accompanied by provenance metadata suitable for downstream modeling and review.

A companion web application, Video Annotation Viewer [@viewer], provides an interactive interface for overlaying annotations on source video — rendering pose skeletons, speaker timelines, subtitle tracks, and scene boundaries — so that researchers can visually inspect and validate pipeline outputs before downstream analysis.

The toolkit targets researchers who need auditable, inspectable feature timelines (e.g., facial action units, gaze-related signals, diarized speech activity) while remaining domain-agnostic for use in psychology, HCI, education research, clinical observation, and related settings.

# Statement of need

Behavioral and interaction research depends heavily on observational video coding, yet human annotation is costly to train, slow to scale, and difficult to reproduce across sites and studies [@observer]. Automated tools can assist, but researchers face a fragmented landscape: each upstream model has its own installation procedure, input expectations, and output format. Integrating several models into a single analysis workflow requires substantial engineering effort that is typically duplicated across labs.

VideoAnnotator addresses this gap by providing a maintainable software layer that standardizes access to widely used open models for face, pose, audio, and scene analysis. It produces inspectable, timestamped events and tracks with per-stage provenance, allowing researchers to audit exactly which detector version produced each annotation. Critically, all processing runs locally, supporting the data-privacy requirements common in research involving children, patients, or other vulnerable populations.

# State of the field

Existing tools for behavioral video analysis fall into two broad categories. Manual annotation platforms such as ELAN [@elan] and Datavyu [@datavyu] provide flexible coding interfaces but require trained human coders and do not scale to large corpora. At the other end, specialized computer-vision libraries such as DeepLabCut [@deeplabcut] and YOLO [@yolo11] offer powerful pose estimation and object detection but target a single modality and leave integration, output standardization, and batch orchestration to the user.

For facial affect, toolkits such as Py-Feat [@pyfeat] and OpenFace [@openface2] extract action units, landmarks, and emotion labels from video frames. On the audio side, openSMILE [@opensmile] remains widely cited for acoustic feature extraction but has seen limited maintenance, and no current open-source toolkit offers end-to-end speech emotion analysis integrated with video. These tools all run locally but each addresses a single modality and produces its own output schema. Scene-detection libraries such as PySceneDetect [@pyscenedetect] and speaker-diarization toolkits like pyannote [@pyannote] similarly solve one piece of the puzzle. A researcher studying parent–child interaction, for example, may need person tracking, facial expression analysis, speech segmentation, and scene detection applied to the same set of videos — currently requiring ad-hoc glue code across four or more libraries with no shared output format or batch orchestration.

VideoAnnotator was built rather than contributed to an existing project because no single package offered the combination of multi-modal pipeline composition, declarative configuration, standardized output formats (COCO, RTTM, WebVTT), and local-first batch orchestration that our research workflows required. The closest comparable systems are either commercial, cloud-dependent, or tightly coupled to a single detector.

# Software design

VideoAnnotator is organized around four architectural layers chosen to balance extensibility with operational simplicity.

**Pipeline registry and detector layer.** Pipelines are registered via declarative YAML metadata files rather than code, so new detectors can be added by providing a metadata file and a Python class that inherits from `BasePipeline`. The base class enforces a uniform interface — `initialize()`, `process()`, `cleanup()`, and `get_schema()` — enabling plug-and-play composition without modifying application code. This metadata-driven approach trades a small amount of configuration overhead for the ability to extend the system without touching existing pipelines.

**Standardized output and provenance.** All pipeline outputs map to established data standards: COCO JSON for spatial annotations, RTTM for speaker diarization segments, and WebVTT for timed text. Each result is wrapped with provenance metadata recording the pipeline name, version, configuration parameters, and processing timestamps. This design sacrifices some pipeline-specific richness in favour of cross-tool interoperability and auditable reproducibility.

**Batch orchestration.** A thread-pool-based orchestrator manages concurrent job execution with configurable retry strategies (fixed, linear, or exponential backoff) and transient-versus-permanent error classification. Jobs are persisted to the local filesystem after each stage, providing checkpoint recovery without requiring an external message queue. This single-machine design keeps deployment simple for lab settings, at the cost of not supporting distributed multi-node processing.

**Service and CLI layer.** A FastAPI service exposes endpoints for job submission, status polling, pipeline discovery, and results retrieval, while a Typer-based CLI provides equivalent local-execution commands. Both interfaces share the same orchestrator and storage backend, ensuring consistent behavior. Token-based authentication is enabled by default and can be relaxed for development use. The companion Video Annotation Viewer [@viewer] connects to the same API, allowing researchers to submit jobs, monitor progress, and review results with synchronized video playback and annotation overlays in a single browser-based workflow.

# Research impact statement

VideoAnnotator was developed at Stellenbosch University to support large-scale analysis of caregiver–child interaction videos collected across multiple sites in sub-Saharan Africa as part of the Global Parenting Initiative. It is currently used by research teams at Stellenbosch University and the University of Oxford to process observational video data in developmental psychology and parenting-intervention studies. The toolkit's local-first design was specifically motivated by the ethical and governance requirements of working with video recordings of children in low- and middle-income settings. Pilot analyses have been conducted on corpora of over 500 sessions, and the software is being prepared for use in upcoming multi-site trials.

# Quality control

The project maintains a pytest-based test suite covering unit, integration, and performance tests across 64 test files. Continuous integration via GitHub Actions runs tests on Ubuntu, Windows, and macOS with Python 3.12, alongside ruff linting, mypy type checking, and Trivy security scanning. Reproducibility is supported by recording pipeline configuration and model versions in output metadata, and Docker images provide consistent execution environments.

# Statement of limitations

VideoAnnotator depends on the accuracy of upstream detectors, and annotation quality is bounded by their performance on a given domain. Processing speed depends on hardware; a GPU is recommended for long videos. The batch orchestrator is designed for single-machine execution and does not currently support distributed scheduling. Ethical deployment — including consent, data governance, and redaction — remains the responsibility of adopters; the toolkit provides hooks and documentation to support these steps but does not enforce them.

# AI usage disclosure

Generative AI tools (GitHub Copilot and Claude, Anthropic) were used during development for code scaffolding, test generation, and documentation drafting. All AI-assisted outputs were reviewed, edited, and validated by the human authors, who made all core design decisions. No AI tools were used to generate the scientific content or analysis reported in this manuscript. All authors take full responsibility for the software and this paper.

# Acknowledgements

We acknowledge the open-source communities behind the upstream models and libraries integrated by VideoAnnotator, in particular OpenFace, Whisper, pyannote, Ultralytics YOLO, LAION and PySceneDetect.

This project was supported by the Global Parenting Initiative and funded by The LEGO Foundation.

# References
