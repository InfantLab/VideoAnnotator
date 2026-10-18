# Contract: `vlm_annotation` config and output

Additive within v1.x (Principles II and V). Every config written before this spec means what it
meant.

## New config keys

| Key | Type | Default | Registry metadata (spec 026) |
|---|---|---|---|
| `start_sec` | float | `0.0` | behavioural, label "From", unit s |
| `end_sec` | float \| null | `null` | behavioural, label "To", unit s, null = end of video |
| `moments_sec` | list[float] \| null | `null` | behavioural, label "Only at these moments" |
| `burst_spacing_sec` | float \| null | `null` | behavioural, label "Frames apart", unit s, `applies_when: sampling_mode: [frame_burst]` |

`burst_offsets` keeps its meaning; its description changes to "multiples of the frame spacing
(the sample interval unless Frames apart is set)".

## Mapping from the window

| Window | Config |
|---|---|
| Sample every N s | `frame_interval_sec: N` |
| from A to B | `start_sec: A`, `end_sec: B` (or null for the end) |
| at listed moments | `moments_sec: [...]` |
| 1 frame | `sampling_mode: single_frame` |
| X frames (X ≥ 2), Y s apart | `sampling_mode: frame_burst`, `burst_offsets: [-⌊X/2⌋ … X−⌊X/2⌋−1]`, `burst_spacing_sec: Y` |

## Behaviour

- Sample points: `moments_sec` if set, else `start_sec + k × frame_interval_sec ≤ end_sec`.
- A burst frame outside the video is dropped; the annotation's `context_frame_numbers` and
  `frame_times` list only the frames sent.
- The results file is rewritten (atomically) after each answer, at most once a second, and at the
  end, with `vlm_progress` at the top level.
- With a stop check set (`set_stop_check`), the pipeline checks before each sample; on stop it
  writes what it has, sets `vlm_progress.stopped_early`, and returns normally.
- Every model call goes through the per-model gate at run priority.

## Test cases

`tests/fixtures/ask_ai/sampling_cases.json`: inputs (duration, interval, start, end, moments,
frames, spacing, N) and expected sample points, burst frame times and the N chosen moments.
Read by `tests/unit/test_vlm_sampling.py` and `viewer/src/lib/sampling.test.ts`.
