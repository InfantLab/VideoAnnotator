# Contract: Viewer

No new route. New screens obey spec 029's vocabulary rule (the vocabulary test renders them).

## Settings → Model servers (new card on `/settings`)

- List: name, kind ("Ollama" / "OpenAI-compatible"), address, the endpoint badge (spec 027
  `EndpointBadge`), "key set" / "no key" / "key from $VAR", Remove (not on the default).
- Add: Name, Kind, Address, Key (password field) or "Use an environment variable" (name), Check.
  Check shows the outcome message and hint; offers the suggested address or kind as a button.
  Add is disabled until a check has passed or the user chooses "Add anyway".
- Remote: a line saying whether remote model servers are on. Admins see a switch; turning it on
  shows the warning dialog (text from the API) and needs a second click; if fixed by the
  environment, the switch is disabled with "set by this server's configuration".
- Rows confirmed before the switch: "Confirmed before remote servers needed switching on".
- Where keys are kept: one line naming the keys file.

## Ask AI (`/ask`)

- **Server picker** before the model picker; default "This computer (Ollama)". Remembered per
  user with the other Ask AI settings (spec 029 `user_preferences`). The badge follows it.
- `VlmModelPicker` takes a server id; messages say "Checking the model server…", "Can't reach
  lab-gpu", "No models on lab-gpu" (Ollama: "No models pulled", as now).
- **Answer shape** (under the question): Free text | Fields. Fields: rows of name, kind, values
  (chips) or range; add, remove, reorder; validation as the shared cases. **Suggest** fills the
  editor with a proposal marked "suggested from your question" or "suggested by the model";
  never applied without the user seeing it. When the chosen server can't enforce shapes: a note
  "lab-gpu can't enforce the answer shape; answers are checked after".
- **Answers**: shaped answers show fields as a small key–value list; a misfit shows "Didn't fit:
  <problem>" with the raw reply under a disclosure. The tally and lane use `label` (first field);
  misfits are their own colour, never merged with a value (Principle VI).
- **Frames per sample**: "One · Several · A short clip"; the clip option shows only when the
  check says the model takes video, otherwise a disabled option with "This model or server
  doesn't take video". Clip length in seconds. Ticks show the clip span.
- **Download**: shaped entries add "Answers, one column per field (CSV)".

## New run settings form (spec 026)

`backend`, `answer_shape`, `clip_length_sec`, `clip_fps` render from metadata; `answer_shape`
uses the Ask AI editor (`widget: answer_shape`); `base_url` becomes a server picker when the
viewer knows the servers (free text remains under "Edit as JSON").

## Video page (spec 028)

A shaped run's lane colours by the first field; the data table shows each field as a column for
that lane's rows, plus fit.

## Tests (Vitest)

`answerShape.test.ts` (shared cases), `ModelServers.test.tsx` (list, add/check flow, admin
switch, fixed-by-environment, no key ever rendered), `AnswerShapeEditor.test.tsx`,
`AskAI.servers.test.tsx` (picker, badge follows server, clip option gating), vocabulary test
extended, contract fixture with a shaped and a clip output.
