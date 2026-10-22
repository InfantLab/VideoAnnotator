# Contract: `vlm_annotation` config and output

Additive within v1.x (Principles II and V). Every config written before this spec means what it
meant: no `backend` key is `backend: ollama`.

## New config keys

| Key | Type | Default | Registry metadata (spec 026) |
|---|---|---|---|
| `backend` | `ollama` \| `openai_compatible` | `ollama` | technical, label "Model server kind", enum labels "Ollama", "OpenAI-compatible (llama.cpp, vLLM, LM Studio…)" |
| `answer_shape` | list of fields \| null | `null` | behavioural, label "Answer shape", widget `answer_shape` |
| `clip_length_sec` | float | `2.0` | behavioural, label "Clip length", unit s, `applies_when: sampling_mode: [clip]` |
| `clip_fps` | float | `4.0` | technical, label "Clip frames per second", `applies_when: sampling_mode: [clip]` |

`sampling_mode` gains `clip` (label "A short clip"). The metadata gains `backends: [ollama,
openai_compatible]` (FR-004). `description` and `homepage` stop saying Ollama only.

## Submission checks (422, before a job is created)

| Condition | Code | Message |
|---|---|---|
| any key matching the secret pattern (`api_key`, `token`, …) | `KEY_IN_CONFIG` | "Keys don't go in a run's settings. Add the key in Settings → Model servers, or name an environment variable there." |
| `backend: openai_compatible` and empty `base_url` | `MISSING_BASE_URL` | "Say which model server to use." |
| `backend: ollama` and `sampling_mode: clip` | `CLIP_NOT_SUPPORTED` | "Ollama can't take video. Choose one frame or several, or a server that takes video." |
| `answer_shape` invalid (data-model.md rules) | `INVALID_ANSWER_SHAPE` | names the field and the rule |
| remote `base_url` not sendable | `REMOTE_NOT_ALLOWED` / `REMOTE_NOT_CONFIRMED` | as api.md |

The same checks run in `initialize()` for configs that arrive by another route (a YAML file
through the CLI), so no route skips them.

## Behaviour

- `backend` chooses the client (research R1, R6); calls, retries, the gate, stop and progressive
  writes are as today.
- `KeyRefusedError` (401/403) stops the run at once: the results so far are written and the job
  fails with "the model server refused the key".
- `answer_shape` set: the call asks for the shape (R8); the reply is checked; `label` is the
  first field (or `DIDN'T FIT`); `fields`, `fit`, `fit_problem` recorded; a misfit counts as
  answered for `abort_after_consecutive_failures`.
- `sampling_mode: clip`: frames from `t − L/2` to `t + L/2` at `clip_fps`, clipped to the video,
  sent as one video part; `clip_start_sec`, `clip_end_sec`, `clip_fps`, `context_frame_numbers`
  and `frame_times` recorded.

## Shared test cases

- `tests/fixtures/ask_ai/sampling_cases.json` gains clip cases (spans at the ends of the video),
  read by `test_vlm_sampling.py` and `viewer/src/lib/sampling.test.ts`.
- `tests/fixtures/answer_shape/cases.json`: shapes (valid and invalid with the expected
  problem), replies (fits, code fence, prose around JSON, wrong value, missing field, not JSON)
  with the expected `fields`/`fit`/`fit_problem`/`label`, and question texts with the shape the
  rule suggests. Read by `test_answer_shape.py` and `viewer/src/lib/answerShape.test.ts` (the
  editor's validation twin).
