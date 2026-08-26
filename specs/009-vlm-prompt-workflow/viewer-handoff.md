# Handoff to video-annotation-viewer: VLM Prompt Preview, Model Picker & Two-Job Comparison

**From**: VideoAnnotator core, `specs/009-vlm-prompt-workflow/` (v1.5.0 branch)
**Purpose**: Written to be pasted as the input to that repo's own `/speckit-specify`. Describes UI
behavior, not implementation.

## Why this exists

`vlm_annotation`'s Configure-step UI today is entirely generic: `prompt` renders as a plain
`Textarea`, `model` as a plain single-line `Input` with no validation or suggestions
(`DynamicPipelineParameters.tsx`'s `default:` case, since `model` has no `enum` in its schema). A
researcher iterating on prompt wording has no way to know if a prompt works well, or if the model
name they typed is even installed, without submitting a full job against real video. Separately,
`VlmAnnotationPanel.tsx` already does one useful comparison — a single job's VLM predictions against
ELAN ground truth, at one instant in time — but there's no way to compare two *VLM* runs (e.g. two
different prompts) against each other on the same video.

## What the backend now provides

Full contract in [`spec.md`](spec.md)'s API section. Summary:

- `GET /api/v1/vlm/models` → `{ base_url, models: string[] }`, or a clear error if the configured
  Ollama server is unreachable — distinguish that from "reachable, zero models" in the UI (different
  messages: "can't reach the model server" vs. "no models pulled yet, run `ollama pull ...`").
- `POST /api/v1/vlm/preview` → send a prompt + model + sampling mode + (an uploaded frame image, OR
  a `{video_path, timestamp_sec}` reference into an already-uploaded video) and get back
  `{ label, reasoning, raw_response, total_time, ... }` synchronously — no job created.

### Things worth knowing before designing the UI

- **This is genuinely the first pipeline-specific UI exception.** Every other pipeline's Configure
  step is rendered by the fully generic `DynamicPipelineParameters.tsx` from its schema alone — this
  is deliberately scoped as a special case for `vlm_annotation` specifically (checked by pipeline id),
  not a new general mechanism for every pipeline to eventually need one. Keep the blast radius small:
  add a slot/wrapper around the generic form for this one pipeline, don't restructure the generic
  renderer itself.
- **Preview can be slow on a cold model** (the pipeline's own docs note first-call load time can
  dominate) — the UI should show a clear "running..." state, not assume sub-second response, and
  should use a generous timeout matching what the backend itself uses.
- **The frame source matters for UX**: if the user hasn't uploaded a video yet (e.g. they're setting
  up config before Step 1's upload), preview needs *some* image — either let them upload a single
  still image just for previewing, or only enable the "test prompt" panel once a video is already
  selected in the wizard, seeking to a chosen timestamp within it. Either is a reasonable design call
  for that repo to make; the backend accepts both input shapes.
- **Two-job comparison needs no new backend endpoint** — each job's `vlm_annotation` output is
  already a normal stored annotation file, fetchable the same way `VlmAnnotationPanel.tsx` already
  fetches one job's. Loading two jobs' worth of `VLMFrameAnnotation[]` client-side and diffing them at
  matching timestamps (reusing `getVlmAnnotationAtTime`'s nearest-within-tolerance lookup pattern
  from `src/lib/parsers/vlm.ts`, the same approach already used for the VLM-vs-ELAN comparison) is
  sufficient — this is a viewer-only capability.

## What the viewer needs to build

1. **Model picker**: a dropdown populated from `GET /api/v1/vlm/models`, falling back to the existing
   free-text input when the endpoint reports the server unreachable (don't hard-block configuration
   just because Ollama isn't running yet).
2. **"Test prompt" panel** in the Configure step, scoped to `vlm_annotation` specifically: run the
   current in-progress prompt/model/sampling-mode values against one frame via the preview endpoint,
   show the returned label + reasoning inline, without leaving the wizard or submitting a job.
3. **Ollama reachability indicator** near the prompt/model fields, sourced from the same models
   endpoint's success/failure (no separate call needed).
4. **Two-job VLM comparison view**: pick two completed jobs that both ran `vlm_annotation` against
   the same video, show their predictions side-by-side on a shared timeline with per-timestamp
   agree/disagree — a direct generalization of the existing `VlmAnnotationPanel` VLM-vs-ELAN pattern
   to VLM-vs-VLM. Reasonable to build as an extension of that existing component (accept a second
   optional `VLMFrameAnnotation[]` prop) rather than a wholly new one.

## Explicit non-goals for this piece of work

- **Comparison across more than two jobs, or across an entire dataset.** That's
  `specs/010-corpus-analysis-foundations/` — a separate handoff.
- **Persisting preview results anywhere** (history, saved-preview library). The backend doesn't store
  them; don't build client-side persistence implying they're saved server-side. A per-session
  in-memory preview history is fine if useful, just don't call it "saved."
- **Generalizing the pipeline-specific-UI-slot mechanism to other pipelines.** This handoff is scoped
  to `vlm_annotation` alone.

## Suggested next step

Paste "What the viewer needs to build" (plus the endpoint summary) into that repo's own
`/speckit-specify`.
