# Contract: viewer routes and behaviour

## Routes (Principle V: old addresses keep working)

| Route | Behaviour |
|---|---|
| `/ask` | Ask AI; with no video, the picker (drop, upload, server folder, recent videos) |
| `/ask?video=<server path>` | Ask AI on that video |
| `/ask?video=…&entry=<id>` | …with that history entry restored |
| `/ask?prompt=…&model=…` | …with question and model filled in |
| `/workbench` | redirect to `/ask`, keeping `prompt` and `model` |
| `/prompts` | redirect to `/ask` with the history open |

Home gains "Ask AI about a video". The nav gains "Ask AI"; Workbench and Prompts leave it.
The New job wizard's prompt test panel becomes "Open in Ask AI" (question, model and sampling
carried over).

## Layout (FR-002, FR-003)

Question, examples, history, model and sampling on the left; the video on the right with no
overlays; below, the timeline (time axis, sample ticks, one lane per shown entry) and the table
(time, frames, one answer column per shown entry, reasoning) with the tally and Export. The
badge in the header (FR-014).

## Behaviour the tests check

- Sampling controls change the ticks and the count with no network request (SC-003).
- The burst thumbnails under the playhead come from `GET /ask/frames` (debounced 150 ms).
- Clicking a marker or a table row seeks; the sample nearest the playhead is highlighted in both;
  hovering a marker shows its frames and answer.
- Answer, error and skipped markers look different (Principle VI); hover says which.
- More than 10 distinct answers shows the hint (FR-009).
- A remote badge is in the warning style; the first try or run to an unconfirmed remote endpoint
  asks first (FR-015).
- No server: the page says it needs a server and a model, and how to get them.
- Every user-visible word follows spec 023 Decision 1: "question", "try", "run", "answers";
  never "job", "preview", "prompt library", "VLM".
