# Checks: Model Connectors (spec 033)

## What could and couldn't be checked (2026-10-24)

No model server was reachable from the dev container: no Ollama (the configured
`host.docker.internal:11434` wasn't running), no llama.cpp, no vLLM (no GPU), no hosted key. So:

- The OpenAI-compatible replies the fake server gives (`tests/fixtures/openai_compatible/*.json`,
  T001) come **from each server's documentation, not captured**; every file says so in `_source`.
  Error wording in particular (schema, images, video refused) may differ on real servers.
- **Not yet measured:** SC-001 (a first answer from a lab's server in 5 minutes, a person), SC-003
  (≥ 95% fit with the default local model on the demo clip), SC-007 (users read the badges right).
  Quickstart Story 3 with a hosted API and Story 4 with real vLLM + Qwen-VL are still to do.
- To do these: start a server, then `VA_TEST_OPENAI_BASE_URL=http://127.0.0.1:8080/v1
  VA_TEST_OPENAI_MODEL=… pytest -m real_vlm_server tests/integration/test_openai_real_server.py`
  (prints the fit rate), and walk through [quickstart.md](quickstart.md) in a browser.

## Walk-through with recorded replies over HTTP

A browser (Playwright's Chromium) against a fresh server (`--dev`, a fresh database, one shared
folder with the demo clip), with the llama.cpp replies served on `127.0.0.1:8080` and the vLLM
replies on `127.0.0.1:8000` by a small HTTP wrapper around the test fake (scripts and screenshots in
the session's scratch folder, not the repository). The default Ollama was unreachable, as in
quickstart Story 1 ("Ollama stopped"). 35 s from opening Settings to the clip try, scripted.

| Story | What happened |
|---|---|
| 1 | Settings → Model servers → Add: `http://127.0.0.1:8080` → Check offered **Use http://127.0.0.1:8080/v1**; with it, "Reachable. 1 model."; Add. The row reads "workstation · OpenAI-compatible · Local · workstation on this computer · no key". lab-gpu (vLLM) added the same way. Ask AI → Model server **workstation**: the badge followed ("Local · gemma-3-4b-it-Q4_K_M.gguf on this computer"); **Try on this frame** answered in 5.3 s from opening Settings. |
| 2 | **Suggest** on "…Return ONLY: TOUCH or NO_TOUCH" showed "Suggested from your question: answer · One of a list: TOUCH, NO_TOUCH" with Use/Discard (nothing applied). Fields `touch` (yes/no) and `initiator` (adult, child, both, none): tries showed fields per answer; a reply that wasn't JSON showed **Didn't fit: The reply isn't JSON.** with the reply under a disclosure, a dashed amber mark on the lane, and **didn't fit 1** apart in the tally. Run on whole video; **Answers, one column per field (CSV)** downloaded both shaped entries: `video, t, frame, touch, initiator, fit, fit_problem, model, backend, raw_response`, read by `pandas.read_csv` with no parsing (fit yes 2, no 1). Methods: "…with gemma-3-4b-it-Q4_K_M.gguf (revision b6500-1a2b3c), running on this computer…". |
| 3 | Adding `https://api.example.invalid/v1` (after **Add anyway**): "Remote model servers are switched off on this server. An administrator can switch them on in Settings → Model servers, unless the server's configuration fixes it." The remote line said they're off. |
| 4 | Ask AI → lab-gpu (vLLM replies): Frames per sample offered **A short clip** after the synthetic probe; a try sent one clip and the frame strip showed its 4 frames. On workstation (llama.cpp replies, which refuse video) and Ollama it stayed disabled with the reason. |
| 5 | Covered by `tests/unit/cli/test_cli_model_servers.py` (check's JSON and exit codes 0/2/3/4/5, `--key-env`, `--no-key`, YAML `job submit --config`, a config with `api_key` refused before sending). |

### Found and fixed during the walk-through

1. **The methods paragraph cited Ollama for a run on llama.cpp.** The pipeline's citation comes
   from its metadata; the Ollama citation is now left out when the recorded backend isn't Ollama
   (`test_ollama_is_cited_only_when_it_ran`).

Found earlier, by the contract fixtures: the viewer's VLM parser rejected `null` `load_time` and
`tokens_per_sec`, which OpenAI-compatible runs write (FR-007); fixed in `lib/validation.ts`.

## Deviations from the plan

- `clip_length_sec` is a **technical** setting, not behavioural as contracts/pipeline-config.md
  said: spec 026's guide allows at most five behavioural settings, and it only shows in clip mode.
- `answer_shape.describe_for_prompt` takes the JSON Schema, not the shape, so the OpenAI client
  can build the fallback prompt itself.
- `think: false` isn't sent to OpenAI-compatible servers (research R2 would send
  `chat_template_kwargs` to servers whose check showed they accept it; the check doesn't test that
  yet). A thinking model on vLLM thinks by default; its reasoning is recorded.
- Tokens per second is recorded only when the server reports it (llama.cpp's `timings`); it is
  not computed from the call time.
- The Video page's data table stays long-format: a shaped run's fields appear as `vlm.<field>`
  rows in the Ask AI track, not as one column per field (contracts/viewer.md); the lane colours by
  the first field, with `DIDN'T FIT` its own category. The wide file is downloaded from Ask AI or
  `GET /jobs/{id}/export/wide`, not yet from the Video page's menu.
- Clip spans aren't drawn on the timeline ticks yet; the frame strip shows a clip's frames.
- Ask AI remembers the chosen server through a new `GET/PUT /api/v1/ask/settings` (spec 029's
  `user_preferences`, name `ask_settings`).
- On a server without sign-in (`--dev`, `AUTH_REQUIRED=false`) nobody is an administrator, so the
  remote switch can't be changed from the viewer (it says only an administrator can); set
  `VIDEOANNOTATOR_REMOTE_MODEL_SERVERS` instead.
- `tests/api/test_ask_api.py::test_remote_unconfirmed_is_refused_until_confirmed` (spec 027) now
  switches remote servers on first and expects the two new endpoint fields: the intended change.

## Gate (T054)

- `pytest tests/`: 2385 passed, 9 skipped.
- `pytest tests/integration/test_output_baseline.py`: 14 passed (real models, 8 min).
- `ruff check .`, `mypy src/videoannotator`: clean.
- `pre-commit run --all-files`: passes except `bats (launcher)` (bats isn't installed in the
  container); its ruff-format also rewrites code blocks in 19 older docs untouched by this spec,
  left as they were.
- Viewer: `bun run lint`, `bun run typecheck`, `bun run test:run` (1067 passed),
  `scripts/build_viewer.sh --check`: bundle up to date.
