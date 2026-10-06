# Feature Specification: Viewer Overhaul (placeholder)

**Feature Branch**: `023-viewer-overhaul` (work lands on `1.6-dev`)
**Created**: 2026-10-06
**Status**: Placeholder. Not yet specified; run `/speckit-specify` with
`SPECIFY_FEATURE_DIRECTORY=specs/023-viewer-overhaul` to write the full spec.
**Sequence**: the spec after [022](../022-videos-read-in-place/spec.md) (videos and results where
you expect them).

## Intent

An overhaul of the Video Annotation Viewer: what a researcher sees when they open a job's results.
The roadmap had a visual redesign as "decided after the pilot"
([roadmap_v1.6.0.md](../../docs/development/roadmap_v1.6.0.md), Out of Scope). This placeholder
records the decision to make it the next spec after 022. Pilot feedback, if it arrives first,
should shape it.

## Themes to cover

1. **A better timeline for the existing temporal events**: speech, speakers, scenes, faces, VLM
   moments and the rest, made easier to read, compare and navigate.
2. **New metrics on the timeline**, starting with **movement per person** over time. Related:
   [roadmap_v1.7.0.md](../../docs/development/roadmap_v1.7.0.md) Phase 3 plans to compute movement
   on the server, per tracked person. The spec should decide whether to pull that forward or
   compute in the viewer first.
3. **The underlying data, easier to reach**: from any overlay or timeline event to the values
   behind it, and out to a file. Related: roadmap v1.6.0 Phase 3 (tidy export).
4. **A more modern interface** across the viewer.
5. **Better handling of multiple videos**: moving between a run's or dataset's videos, and seeing
   them together. Related: previous/next between a batch's videos (done), and roadmap v1.6.0
   Phase 6 (corpus overview).
6. **Researchers' own notes and annotations, light touch**: add a note or a simple label at a
   moment or over a span, kept with the job's results. Related: ELAN export (roadmap v1.6.0
   Phase 3), so researcher annotations can travel to the tools labs already code in.

## Constraints already known

- The constitution's faithful-display principle applies: never create or hide values. Each overlay
  says which pipeline it came from (spec 017), and researcher notes stay visibly distinct from
  pipeline output.
- Standalone mode (dropping in files, no server) keeps working.
- Results are in the run folders defined by spec 022. Researcher annotations will be stored there
  too unless the spec decides otherwise.
