# JOSS Review Coordination Handover (No Assumptions)

This handover coordinates VideoAnnotator and Video Annotation Viewer against the two active pre-review threads:

- https://github.com/openjournals/joss-reviews/issues/10182
- https://github.com/openjournals/joss-reviews/issues/10183

Last refreshed: 2026-07-14

## 1. What is confirmed in editor comments

The points below are direct quotes or paraphrases tied to specific comments.

1. Reviewer/editor confusion about repo relationship and scope was explicitly raised in 10183:
- Relationship clarification request: https://github.com/openjournals/joss-reviews/issues/10183#issuecomment-4405442786
- Standalone-use question: https://github.com/openjournals/joss-reviews/issues/10183#issuecomment-4409665662
- Detailed confusion with state-of-field framing: https://github.com/openjournals/joss-reviews/issues/10183#issuecomment-4433307991
- Scope-policy concern: https://github.com/openjournals/joss-reviews/issues/10183#issuecomment-4433461348

2. Author response in 10183 said joint evaluation is acceptable and clarified standalone-plus-optional-integration:
- Author clarification (scope, standalone, optional integration): https://github.com/openjournals/joss-reviews/issues/10183#issuecomment-4558881867
- Editor response that joint evaluation can make sense, with mechanics left open: https://github.com/openjournals/joss-reviews/issues/10183#issuecomment-4559454824

3. Installability and stale-doc risks were explicitly raised in 10182:
- Preflight installability blockers and stale reviewer docs: https://github.com/openjournals/joss-reviews/issues/10182#issuecomment-4435493794
- Author fix summary (v1.4.3 era): https://github.com/openjournals/joss-reviews/issues/10182#issuecomment-4559006208
- Editor asked for self-review to avoid stale information and clarified quality expectations: https://github.com/openjournals/joss-reviews/issues/10182#issuecomment-4559306724

## 2. Decision point (not pre-assumed here)

This document does not assume a single merged submission or two separate submissions.

Current decision status from comments:
- Joint evaluation was discussed positively in 10183, but mechanics were left to editors.

Therefore run one of these tracks:

1. Track A: Continue as two linked pre-review threads.
2. Track B: Move to a single merged manuscript path if editors request it.

Everything in Sections 3 to 6 works for either track.

## 3. Common actions required in both repos

1. Relationship statement consistency
- Ensure README and reviewer docs in both repos explain the same role split.
- Minimum required meaning:
  - VideoAnnotator handles processing and standardized outputs.
  - Viewer handles synchronized audit/review.
  - Viewer works standalone and can optionally connect to VideoAnnotator.

2. Standards consistency
- Ensure the same core formats are listed wherever relevant:
  - COCO
  - WebVTT
  - RTTM

3. Stale-doc prevention
- Re-run reviewer onboarding docs from a clean clone.
- Remove or fix any command/script references that no longer exist.
- This directly addresses the 10182 stale-doc concern:
  - https://github.com/openjournals/joss-reviews/issues/10182#issuecomment-4435493794
  - https://github.com/openjournals/joss-reviews/issues/10182#issuecomment-4559306724

4. Evidence discipline in thread replies
- When posting updates, link specific commits/paths and avoid high-level claims without concrete evidence.

## 4. Viewer repo handover checklist

Owner: Viewer maintainer

1. README framing
- Add a short section near top explaining standalone use first, optional integration second.
- Verify wording directly addresses concerns in:
  - https://github.com/openjournals/joss-reviews/issues/10183#issuecomment-4405442786
  - https://github.com/openjournals/joss-reviews/issues/10183#issuecomment-4409665662

2. Reviewer onboarding
- Provide two explicit quickstarts:
  - Standalone quickstart
  - Integrated quickstart with VideoAnnotator endpoint
- Confirm both paths actually run from clean clone.

3. State-of-field language
- Remove ambiguous contrasts that triggered confusion and keep comparisons concrete.
- Target comments to satisfy:
  - https://github.com/openjournals/joss-reviews/issues/10183#issuecomment-4433307991
  - https://github.com/openjournals/joss-reviews/issues/10183#issuecomment-4433461348

4. Pre-review update comment in 10183
- Post a concise update with links to:
  - the exact docs changed
  - exact commit(s)
  - explicit statement of current submission track (A or B)

## 5. Annotator repo handover checklist

Owner: Annotator maintainer

1. Keep reviewer path installable and current
- Re-validate reviewer quickstart instructions on clean clone.
- Ensure no stale script references remain.

2. Manuscript/docs sync
- Keep wording aligned with current repo behavior and version metadata.

3. Pre-review update comment in 10182
- Reply with concrete updates linked to files/commits, explicitly calling out stale-doc prevention and installability validation.

## 6. Status message templates (track-aware)

Use one pair only, based on current editor direction.

### Template A (if staying as two linked threads)

For 10183:

Thanks for the detailed guidance. We updated Viewer docs to reduce ambiguity around scope and usage. Specifically, we now state standalone use first and optional VideoAnnotator integration second, and we clarified the role split with concrete examples and format support.

Changes:
- <viewer commit link 1>
- <viewer doc path link 1>
- <viewer doc path link 2>

We are currently proceeding as two linked threads while keeping wording synchronized with 10182.

For 10182:

Quick coordination update: we completed a stale-doc and installability self-review and aligned reviewer-facing instructions with current behavior.

Changes:
- <annotator commit link 1>
- <annotator doc path link 1>
- <annotator doc path link 2>

We are keeping 10182 and 10183 synchronized as linked threads.

### Template B (if editors request merged path)

For 10183:

Thanks for the guidance. We have aligned Viewer docs and framing with the merged-submission path requested by editors, while preserving explicit standalone usage and optional integration wording.

Changes:
- <viewer commit link 1>
- <viewer doc path link 1>

We will treat 10183 as coordinated with the merged manuscript workflow and keep updates mirrored in 10182.

For 10182:

Coordination update: manuscript/docs wording and reviewer instructions have been synchronized for the merged-submission path, with concrete fixes for stale info and installability clarity.

Changes:
- <annotator commit link 1>
- <annotator doc path link 1>

We will mirror coordination updates across 10182 and 10183.

## 7. Definition of done

Done means all are true:

1. Both repos have clear, non-contradictory role statements.
2. Viewer standalone and optional integration paths are both documented and runnable.
3. Reviewer onboarding docs in both repos are clean-clone validated.
4. Both pre-review threads include concrete update comments with commit/doc links.
5. Thread updates explicitly state current track (A linked threads, or B merged path).

## 8. Reference URLs (quick index)

Primary threads:
- https://github.com/openjournals/joss-reviews/issues/10182
- https://github.com/openjournals/joss-reviews/issues/10183

Key relationship/scope comments in 10183:
- https://github.com/openjournals/joss-reviews/issues/10183#issuecomment-4405442786
- https://github.com/openjournals/joss-reviews/issues/10183#issuecomment-4409665662
- https://github.com/openjournals/joss-reviews/issues/10183#issuecomment-4433307991
- https://github.com/openjournals/joss-reviews/issues/10183#issuecomment-4433461348
- https://github.com/openjournals/joss-reviews/issues/10183#issuecomment-4558881867
- https://github.com/openjournals/joss-reviews/issues/10183#issuecomment-4559454824

Key installability/self-review comments in 10182:
- https://github.com/openjournals/joss-reviews/issues/10182#issuecomment-4435493794
- https://github.com/openjournals/joss-reviews/issues/10182#issuecomment-4559006208
- https://github.com/openjournals/joss-reviews/issues/10182#issuecomment-4559306724
