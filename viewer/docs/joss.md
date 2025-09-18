---
title: "Video Annotation Viewer: an interactive web application for exploring, comparing, and auditing behavioral video annotations"
tags:
  - visualization
  - web application
  - scientific tooling
  - behavioral science
  - reproducibility
authors:
  - name: Caspar Addyman
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Irene Uwerikowe
    affiliation: 1
  - name: Jeremiah (Jerry) Ishaya
    affiliation: 1
  - name: Daniel Stamate
    affiliation: 2
  - name: Mark Tomlinson
    affiliation: 1
affiliations:
  - name: Institute for Life Course Health Research, Department of Global Health, Stellenbosch University, South Africa
    index: 1
  - name: Department of Computing, Goldsmiths, University of London, United Kingdom
    index: 2
date: 27 August 2025
bibliography: paper.bib
---

# Summary

**Video Annotation Viewer** is a lightweight web application for *visualizing, filtering, and comparing* annotations overlaid on video. It targets lab workflows where researchers need to audit automated outputs, align them with human judgments, and communicate findings. The Viewer:

- plays videos with **overlayed tracks** (e.g., landmarks, gaze on/off, bounding boxes, utterance spans);
- provides **timelines, filters, and side-by-side comparisons** across multiple runs or conditions;
- reads standardized JSON/CSV outputs (including those from **VideoAnnotator**) and exports selected segments, screenshots, or summary tables.

# Statement of need

When automated pipelines scale up, researchers must still *inspect and trust* what detectors produced—especially where constructs are subjective, culturally variable, or ethically sensitive. Visual inspection and efficient comparison improve reliability, uncover edge cases, and make downstream validation more efficient. Prior discussions of the “measurement gap” in large-scale observational research emphasize that scalable automation must be paired with *interpretable review* tools to be credible and useful. 

The Viewer addresses this by providing a focused, domain-agnostic interface: load a folder or manifest of videos + annotations; scrub, filter, and compare; export the evidence. It is not a monolithic analysis suite; it is the *inspection lens* researchers can adopt irrespective of their chosen detectors/models.

# Functionality

- **Overlayed playback.** Draw tracks and events (e.g., facial AUs, poses, segments) on the video; toggle layers; adjust thresholds.
- **Timeline & filters.** Zoomable timelines, text search over events, and filters by detector, label, confidence, or time window.
- **Comparison.** Side-by-side view of (a) different models on the same video or (b) the same model on different videos/conditions.
- **Review & export.** Bookmark moments, export stills or CSV summaries, and generate minimal “review packets” for human raters.
- **Interoperability.** Reads the standardized outputs from *VideoAnnotator* but accepts any tool that emits timestamped events/tracks.

# Design & architecture

- **Backend tasks (optional).** If configured, the Viewer can submit jobs to a running VideoAnnotator service to process new videos and display results when ready.
- **Decoupled front end.** Modern web stack with stateless reading of manifest + annotation files; no database required for local use.
- **Reproducibility.** View-state can be serialized (e.g., selected layers, thresholds, time range) for sharing exact audit contexts with collaborators.

# Validation and usability

We include demo manifests with short clips and annotations to exercise overlays, comparisons, and exports, supporting lab onboarding and smoke testing. The tool is designed to support expert review workflows where automated outputs are checked against human judgments before broader use in research or practice. :contentReference[oaicite:6]{index=6}

# Acknowledgements

We thank colleagues across behavioral research communities who emphasized the need for transparent *review tools* to complement automated pipelines and bridge the gap between scalable coding and expert interpretation. 

# References

*References are provided in `paper.bib` (visual analytics for behavioral data, auditability in ML, and prior work on observational methods that motivate inspection tools).*
