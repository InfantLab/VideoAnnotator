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

VideoAnnotator is an open-source Python toolkit for automated video annotation, designed for behavioral, social, and health research at scale. It provides:

- a modular pipeline framework that wraps commonly used open detectors (e.g., OpenFace-based face and behavior features [@openface2], speech processing built on Whisper-style ASR [@whisper]);
- a local-first FastAPI service for reproducible, scriptable processing [@fastapi];
- Docker images for CPU/GPU execution;
- standardized, timestamped outputs with provenance metadata suitable for downstream modeling and review.

The toolkit targets researchers who need auditable, inspectable feature timelines (e.g., facial action units, gaze-related signals, diarized speech activity), while remaining domain-agnostic for use in psychology, HCI, education research, clinical observation, and related observational settings.

# Statement of need

Behavioral and interaction research often depends on observational video coding, but human annotation is costly to train, slow to scale, and difficult to reproduce across sites and studies. Automated tools may assist but must run locally to maintain data privacy. VideoAnnotator addresses this gap by providing a maintainable software system that standardizes access to widely used open models for face, pose, and audio analysis, and produces inspectable, timestamped events and tracks suitable for downstream analysis.

# Functionality

VideoAnnotator provides:

- Pipelines configured via declarative YAML/JSON, enabling detector composition without code changes.
- Batch and service execution (CLI and FastAPI) for integration into lab workflows, notebooks, and automation.
- Timestamped events and tracks with confidences plus per-stage provenance to support reproducibility and audit.
- Local/on-prem execution patterns to support privacy-sensitive workflows.

# Illustrative use cases

- Education/HCI: quantifying participation and attention-related behavior in classroom or interaction videos.
- Clinical and therapy: triaging sessions by indicators such as engagement or agitation.
- Team interaction and sports: timing changes in gaze and proximity during drills.
- Developmental science: producing objective micro-codes that can later be mapped to higher-level constructs in a transparent, two-stage analysis.

# Design and architecture

VideoAnnotator exposes:

1. An API and CLI to run configured pipelines over files, folders, or job queues.
2. A detectors layer with consistent batching and optional GPU utilization.
3. A structured event store emitting timestamped tracks and segments with provenance.
4. A FastAPI service providing endpoints for health checks, job submission, and results retrieval.
5. Dockerized runtimes (CPU/GPU) and documented installation paths for reproducible execution.

# Quality control

The project includes automated tests and a reviewer-friendly smoke-test path, alongside documented installation and troubleshooting guidance. Reproducibility is supported by recording pipeline configuration and versions in output metadata.

# Statement of limitations

VideoAnnotator depends on the behavior and accuracy of upstream detectors and models, and performance depends on hardware (GPU recommended for long videos). Ethical deployment (consent, governance, redaction) remains the responsibility of adopters; the library provides hooks to implement these steps.

# Acknowledgements

We acknowledge the open-source communities behind the upstream models and libraries integrated by VideoAnnotator.

This project was supported by the Global Parenting Initiative and funded by The LEGO Foundation.

Portions of documentation and development tooling were AI-assisted. All authors reviewed the repository and take responsibility for the software and this manuscript.
