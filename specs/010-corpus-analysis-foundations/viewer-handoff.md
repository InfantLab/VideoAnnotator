# Handoff to video-annotation-viewer: Dataset Overview & Corpus Summary

**From**: VideoAnnotator core, `specs/010-corpus-analysis-foundations/` (v1.5.0 branch)
**Purpose**: Written to be pasted as the input to that repo's own `/speckit-specify`. Describes UI
behavior, not implementation.

## Why this exists

Once datasets (`specs/007`) and batch-tagged submissions (`specs/008`) exist, there's still no place
to see a dataset as a whole — its videos, which have results, and what those results look like in
aggregate. This is also the natural home for "easily switch between videos in a batch" (item 7 from
the original workflow-upgrades request): a dataset overview is both the corpus-analysis entry point
and the batch-review entry point at once.

## What the backend now provides

Full contract in [`spec.md`](spec.md)'s API section:

- `GET /api/v1/datasets/{id}/summary` → per-video coverage (`filename`, `job_id` or `null`,
  `status`), a corpus-wide `label_distribution` (counts per label, aggregated across each video's
  most recent `vlm_annotation` run), and `elan_agreement` (`{covered_videos, agreement_rate}` or
  `null` if no video in the dataset has ground truth).

### Things worth knowing before designing the UI

- **This aggregates the *most recent* run per video, not every historical run.** If a researcher
  re-ran a video with a revised prompt, only the latest counts. If the UI needs to show "this video
  was run 3 times," that history has to come from a separate jobs-list query filtered by video —
  the summary endpoint itself doesn't expose per-video run history.
- **`elan_agreement` can be `null`.** Don't render a percentage or a chart before checking for this —
  show something honest like "no ground truth coded yet" instead of a misleading 0%.
- **`job_id: null` for a video means no completed run yet** — could be pending, running, failed, or
  simply never submitted. If the UI wants to distinguish those states specifically, cross-reference
  against the batch/jobs list (spec 008) rather than assuming the summary endpoint itself
  disambiguates them.

## What the viewer needs to build

1. **A Dataset Overview page**: thumbnail grid (or list) of every video in a dataset, each showing
   its current result status (has results / pending / failed / never run) at a glance.
2. **Quick-jump from any video's thumbnail directly into its results** in the annotation viewer —
   this is where batch review navigation (spec 008's prev/next) and this page meet: entering the
   viewer from here should carry the dataset's video order so prev/next work the same way they would
   from a batch context.
3. **A corpus summary card**: the label distribution as a simple chart or breakdown (e.g. "62% TOUCH,
   38% NO_TOUCH across 34 of 40 videos"), and the ELAN agreement rate where available, with its
   coverage stated plainly ("agreement measured on 12 of 40 videos with ground truth coded so far").
4. **An honest "nothing yet" state** for a freshly-created dataset with no completed jobs — this is
   an expected, common state (a dataset is often created before any job is submitted against it), not
   an error state.

## Explicit non-goals for this piece of work

- **Any statistic beyond basic counts/rates** — no kappa, no confidence intervals, no significance
  testing. If a researcher wants that, the answer is "export and use the Python research repo," not
  a chart in this UI.
- **Per-video multi-run history/comparison within the dataset view.** That's spec 009's two-job
  comparison, invoked from an individual video's context, not something this page itself needs to
  render inline for every video.
- **Editing the dataset's video manifest from this page.** That's spec 007's Datasets page; this one
  is read/review-focused.

## Suggested next step

Paste "What the viewer needs to build" (plus the endpoint summary) into that repo's own
`/speckit-specify`.
