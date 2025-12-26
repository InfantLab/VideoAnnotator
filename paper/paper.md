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

VideoAnnotator is an open-source Python toolkit for automated and manual annotation of video, designed for behavioral, social, and health research. It provides a modular pipeline framework, a CLI, and a local-first FastAPI service, with reproducible Docker images and standardized outputs.

# Statement of need

Behavioral and interaction research often depends on observational video coding, but manual annotation is costly to train, slow to scale, and difficult to reproduce across sites and studies. VideoAnnotator addresses this gap by providing a maintainable software system that standardizes access to widely used open models for face, pose, and audio analysis, and produces inspectable, timestamped events and tracks suitable for downstream analysis.

# Functionality

VideoAnnotator provides:

- Modular pipelines configured via YAML.
- Batch and service execution (CLI or FastAPI).
- Standardized outputs and provenance metadata to support reproducibility.
- Local/on-prem execution patterns to support privacy-sensitive workflows.

# Quality control

The project includes automated tests and a reviewer-friendly smoke-test path, alongside documented installation and troubleshooting guidance.

# Statement of limitations

VideoAnnotator depends on the behavior and accuracy of upstream detectors and models, and performance depends on hardware (GPU recommended for long videos). Ethical deployment (consent, governance, redaction) remains the responsibility of adopters; the library provides hooks to implement these steps.

# Acknowledgements

We acknowledge the open-source communities behind the upstream models and libraries integrated by VideoAnnotator.

Portions of documentation and development tooling were assisted by AI. All authors reviewed the repository and take responsibility for the software and this manuscript.
