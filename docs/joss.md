---
title: "VideoAnnotator: an extensible, reproducible toolkit for automated and manual video annotation in behavioral research"
tags:
  - Python
  - video analysis
  - behavioral science
  - reproducibility
  - machine learning
authors:
  - name: Caspar Addyman
    orcid: https://orcid.org/0000-0003-0001-9548
    affiliation: 1
  - name: Jeremiah Ishaya
    affiliation: 1
  - name: Irene Uwerikowe
    affiliation: 1
  - name: Daniel Stamate
    affiliation: 2
  - name: Jamie Lachman
    affiliation: 3
  - name: Mark Tomlinson
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

**VideoAnnotator** is an open-source Python toolkit for _automated and manual annotation of video_, designed for behavioral, social, and health research at scale. It provides:

- a pluggable pipeline that wraps commonly used detectors (e.g., **OpenFace 3**, **DeepFace**) for face, action‐unit, affect, gaze, speech and motion features;
- a **FastAPI** service for local or server deployment;
- **Docker** images for fully reproducible execution with GPU support where available;
- a clear data contract for inputs/outputs (JSON/CSV/Parquet), timestamped tracks, and provenance metadata suitable for downstream modeling and review.

The toolkit targets researchers who need _auditable, explainable feature timelines_ (e.g., smiles, gaze‐on/off, vocal activity, proximity), while remaining domain‐agnostic for use in psychology, HCI, education research, clinical observation, sports science, or any scenario where video behaviors must be measured consistently.

# Statement of need

Across behavioral sciences, observational methods remain the gold standard for assessing rich interpersonal phenomena, but manual coding is costly, subjective, and difficult to scale. Prior work on parenting–child interaction assessment, for example, highlights both the value of holistic constructs and the practical limits of human macro-coding (training burden, reliability drift, cultural variance) when datasets grow beyond small lab cohorts. These concerns generalize to many video-based fields (therapy sessions, classroom interactions, telehealth triage), where the measurement gap—lack of scalable, standardized, and transparent coding—constrains progress.

**VideoAnnotator** addresses this need by (i) standardizing access to modern open models for faces, pose, and voice; (ii) emitting **timestamped micro-events** that are inspectable and auditable; and (iii) packaging the whole stack for reproducible, resource-constrained deployment (laptops, on-prem servers, or cloud GPUs). The library does _not_ prescribe a single theory of behavior; rather, it provides the _feature scaffolding_ upon which diverse constructs or downstream models can be built (e.g., sensitivity, synchrony, rapport), with outputs suitable for both qualitative review and quantitative ML.

# Functionality

- **Pipelines & plugins.** Modular wrappers for detectors (face/affect, keypoints, diarization/transcripts) chained into pipelines via declarative YAML/JSON configs. New detectors can be added with small adapter classes.
- **Annotations as first-class data.** Event schemas for segments, point events, and tracks (with confidences), plus per-stage provenance and hashes for audit.
- **Batch & service modes.** Run from CLI for batch processing, or as a **FastAPI** service to integrate into lab workflows, notebooks, or web apps.
- **Reproducible runs.** Dockerfiles/compose recipes and pinned environments for CPU/GPU, designed to minimize “works on my machine” bugs.
- **Privacy-aware processing.** Intended to run locally/on-prem; supports redaction steps (e.g., face blurring tracks) as optional pipeline stages.
- **Interoperability.** Outputs align with common tabular formats and can be visualized in the companion _Video Annotation Viewer_ (separate submission).

# Illustrative use cases

- **Education/HCI:** quantifying joint attention and participation in classroom videos.
- **Clinical & therapy:** triaging sessions by indicators such as engagement or agitation.
- **Team interaction / sports:** timing of gaze, gestures, or proximity changes in drills.
- **Developmental science:** producing objective micro-codes that later map to global constructs in a transparent, two-stage analysis.

# Design & architecture

VideoAnnotator exposes:

1. **`annotate()` API & CLI** to run configured pipelines over folders or manifests.
2. **Detectors layer** (e.g., wrappers for OpenFace 3, DeepFace, pose/landmarks, ASR/diarization) with consistent batching and GPU utilization.
3. **Event store** building timestamped tracks with confidences and per-stage provenance.
4. **FastAPI service** exposing health, queue, and processing endpoints for integration.
5. **Dockerized runtimes** (CPU/GPU) with pinned models and test fixtures.

# Quality control

We provide minimal smoke tests for pipeline execution, schema validation for outputs, and example configs with short clips to verify end-to-end operation. Reproducibility is validated by hashing the pipeline configuration, model versions, and container image used for each run (included in the metadata footer of outputs).

# Statement of limitations

- The toolkit depends on upstream detectors; accuracy/cultural generalizability reflect those models and recording conditions (lighting, angle, mic).
- Inference on long videos may require GPU resources for real-time/near real-time performance.
- Ethical deployment (consent, data governance, redaction) remains the responsibility of adopters; the library offers hooks to implement these steps.

# Acknowledgements

We thank colleagues in developmental science and global health for formative discussions on scalable, interpretable video measurement, including prior analyses of macro-coding, measurement scalability, and const
