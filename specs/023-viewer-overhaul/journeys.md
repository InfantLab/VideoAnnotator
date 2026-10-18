# Journeys (Process step 5)

**Recorded**: 2026-10-16, dictated by the project lead and written up here. Each journey is
checked against the [draft proposal](spec.md#draft-proposal-2026-10-16-for-review); "Needs" lists
what the proposal must add or make sure of.

Out of scope for both: human review or correction of pipeline output (shelved for now). These
journeys are about getting videos in, asking questions of them, and getting data out.

## Journey 1: New to AI analysis

**Who.** A researcher who has heard about analysing video with AI and wants to try it in a
research context. Unfamiliar with it.

**What they want.**
- Very quickly give it a video, ask questions about it, and get systematic answers back.
- Play with it a few times, on something small, before committing to a run on their full-length,
  high-resolution videos.
- Not lose the prompts they've tried. Finding a formula that works takes many tries, and earlier
  attempts must not vanish when they move on to try something else.

**What they won't know yet, so the interface has to teach it.**
1. **Sampling.** The video isn't watched continuously: it is sampled every few seconds, and a
   sample may need a short burst of frames rather than one, for anything that moves.
2. **Local vs remote AI.** Whether the video stays on their machine or goes to a service. Must be
   signposted very clearly.
3. **The prompt defines the output.** For now, the answer's format (the labels or values they
   want back) has to be written into the prompt itself. A good example prompt shows how.

**Against the proposal (Ask AI).**

| Need | Proposal today | Needs |
|---|---|---|
| Video in fast | Home's "ask AI about a video"; Ask AI opens on a video | Drop a video straight into Ask AI, no project, nothing to name (Decision 2). Upload or pick from a folder. |
| Try before committing | "Try on this frame", "Try on 10 moments" | Make trying the obvious first action and the whole-video run the deliberate second; show the estimate (samples, time) before a run starts. |
| Understand sampling | Sampling controls with an estimate | Show what sampling *means* on the timeline: tick marks where samples fall, and the frames of a burst as thumbnails, updating as settings change. One-line explanation beside the controls, with a link to more. |
| Local vs remote | Live badge "Local · model on this computer" | Badge also explained on first use ("your video never leaves this computer"); remote, if ever configured, needs a confirmation before the first send, not just a colour. |
| Prompt defines output | Prompt box | Start from an example prompt that shows the answer format ("Answer with one of: touching, not touching"). A hint when answers don't match the requested labels. |
| Don't lose prompts | Each try kept in a history; prompt library | History is automatic and survives leaving the page; any earlier prompt is one click to restore; star or name the ones that work. Comparing two prompts on the same moments side by side. |
| Systematic answers back | Lane, table, export | Same window (as proposed). |

## Journey 2: Experienced in manual coding

**Who.** A researcher who already codes video by hand or live: for example a developmental
researcher coding eye gaze or social behaviour, or a linguist labelling speech utterances. They
know the idea, from ELAN and tools like it, of a shared timeline per video on which annotations
are entered by hand. What we offer is an automated process that produces something like that.

**What they want.**
- Ask simple questions of their videos, usually several videos at once rather than one.
- Above all, the videos turned into data in a standard table (spec 025).
- To see the results on a timeline for each video, as they would in ELAN, and take them into
  ELAN (spec 025's export).
- To understand where each pipeline comes from and what it can do: what it's for, its published
  evaluation or benchmarks, its homepage and citation.
- Sensible defaults, so they never need to dig into settings, and especially not the technical
  ones.

**The pipeline as they should see it.** At the top, what it does, in clear behavioural terms.
Then:

1. **Behavioural settings**: the few choices about *what* is measured (which labels, which
   people, how often), with sensible defaults.
2. **Technical settings**: how it's computed (backends, model sizes, thresholds that only matter
   to the method), collapsed and left at their defaults.
3. **About**: a short description, a link to its homepage, its citation, and its benchmark
   (external evaluation), so a researcher can judge and report it.

Today every pipeline's settings are one long list. For example, `face_analysis` shows
`scale_factor` and `min_neighbors`, which apply only to the OpenCV detector it no longer uses by
default, next to `detect_emotions`.

**Against the proposal (Videos, Video page, Pipelines).**

| Need | Proposal today | Needs |
|---|---|---|
| Several videos at once | Videos page: pick a folder or select videos, Run… | Start from a folder in one step; a project offered, not required. |
| Data in a table | Export on Videos and Video pages | Make the table the headline result of a run ("48 videos → results table"), not one item in a menu. |
| Timeline per video | Video page (not yet sketched) | Lanes that read like ELAN tiers: labelled, one row per person or speaker. The Video page sketch is the next piece of design. |
| Know the pipeline | Pipelines page | Each pipeline: what it does, behavioural settings, technical settings, About (homepage, citation, benchmark). Shown on Pipelines *and* where pipelines are chosen in New run. |
| Defaults, no digging | Settings form: "few that matter first, rest under More" | The split is behavioural vs technical, not "important vs rest", and it lives in the pipeline metadata so the CLI, API and viewer agree. |

**What this asks of the pipeline metadata** (`registry/metadata/*.yaml`; additive, Principle V):
- Per setting: a `group` (behavioural or technical), and a plain label (`config_schema` has only
  a description today).
- Per pipeline: a one-line behavioural summary, a `homepage`, and `benchmarks` (name, link,
  headline figure). `citations` already exist.
- Settings that apply only under another setting (OpenCV-only thresholds) marked as such, so they
  are hidden when they don't apply.

## Checks to make with the user

- Journey 2's example researchers were dictated as "an intimacy researcher… capturing eye gaze or
  sexual movements". Written above as a developmental researcher coding eye gaze or social
  behaviour, which may be a mis-hearing; correct if not.
- "We also agree that on the timeline with the BDA, we can pull that forward": read as "they see
  results on a timeline, and can take them into ELAN". Confirm.
