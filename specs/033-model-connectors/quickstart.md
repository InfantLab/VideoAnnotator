# Quickstart: Model Connectors

A walk-through per user story, for the checks after building (`checks.md`). Uses the demo clip
(`viewer/demo-assets/…Peekaboo_h265.mp4`) and `bash scripts/dev.sh` at http://127.0.0.1:19011.

## Servers to have

- Ollama with the default model (`qwen3.5:9b`), as today.
- One OpenAI-compatible server on this machine or the network. The cheapest:
  `llama-server -hf ggml-org/gemma-3-4b-it-GGUF --port 8080` (llama.cpp, with its projector) or
  LM Studio's server. For Story 4: vLLM with a Qwen-VL model, e.g.
  `vllm serve Qwen/Qwen3-VL-8B-Instruct --limit-mm-per-prompt '{"image": 8, "video": 1}'
  --trust-request-mm-kwargs`.
- For Story 3 only: a hosted API key, on videos that aren't of participants (the demo clip).

## Story 1 — the lab's server

1. Settings → Model servers → Add: "workstation", OpenAI-compatible, `http://127.0.0.1:8080/v1`,
   Check → "Reachable. 1 model." Add. Try `http://127.0.0.1:8080` too: the check offers `/v1`.
2. Stop the server, Check again → "Can't reach…" with a hint. Point it at Ollama's port with the
   OpenAI kind and no `/v1` → wrong kind, suggests the address or the Ollama kind.
3. Ask AI → server "workstation", the model → Try on this frame. Answer, reasoning if any, time.
   Badge: Local · on this computer.
4. Run on the whole video. Open the results: `backend: openai_compatible`; provenance `vlm.backend`
   and the model with `revision_kind: unknown` and a reason; methods paragraph "on this
   computer".
5. With Ollama: everything from before the upgrade (history, a workflow, a project's settings)
   opens and runs as it did. `pytest tests/integration/test_output_baseline.py` passes.

## Story 2 — answers in a fixed shape

1. Ask AI, default Ollama model, question: "Is the adult touching the child? Who started it?
   Return ONLY: TOUCH or NO_TOUCH". Suggest → `answer` (choice TOUCH, NO_TOUCH), "suggested from
   your question".
2. Clear the "Return ONLY" line, Suggest → "suggested by the model"; edit to `touch` (yes/no) and
   `initiator` (choice adult, child, both, none). Try on 5 moments: fields shown per answer.
3. Run on the video; download "Answers, one column per field": columns `touch`, `initiator`,
   `fit`. Read it in R (`readr::read_csv`) and pandas with no text parsing (SC-004). Download the
   tidy table; `pivot_wider` on `measure` gives the same columns.
4. Count misfits: at least 95% fit with the default model (SC-003). Force a misfit (a choice
   list the model won't use, or `--`) to see "Didn't fit: …" and the raw reply.
5. With llama.cpp's server, repeat 2 once. If it rejects the schema, the note "can't enforce the
   answer shape" appears and answers are still checked.

## Story 3 — a hosted model, knowingly

1. Fresh database: Add `https://api.openai.com/v1` → refused, "switched off on this server".
2. As admin, Settings → switch on; read the warning; confirm. Add the server with the key.
3. Ask AI → that server → Try: the spec 027 confirmation names the host; confirm; the badge says
   Remote in the warning colour.
4. Search for the key (SC-006): `grep -r "<key>"` over the results folder, the server log, the
   database file, exported tables and the methods text, and the network responses saved from the
   browser's dev tools: nothing.
5. Change the key to a wrong one → Try → "the server refused the key"; a run stops at once.
6. Set `VIDEOANNOTATOR_REMOTE_MODEL_SERVERS=off`, restart: the switch is fixed off; a remote
   Ollama address confirmed before the upgrade (copy a v1.6 database) still works and is listed
   as confirmed before the switch.

## Story 4 — a short clip per sample

1. Add the vLLM server; choose the Qwen-VL model: "A short clip" appears (probe passed). With the
   llama.cpp server it is disabled with the reason.
2. Clip 2 s, sample every 5 s; ticks show 2 s spans; the first and last spans are shortened.
3. Run; each answer records `clip_start_sec`, `clip_end_sec`, `clip_fps`.

## Story 5 — the command line

```
videoannotator model-servers check http://127.0.0.1:8080/v1 --kind openai_compatible --json
videoannotator model-servers add workstation http://127.0.0.1:8080/v1 --kind openai_compatible --no-key
videoannotator job submit viewer/demo-assets/2UWdXP.joke1.rep3.take1.Peekaboo_h265.mp4 --pipelines vlm_annotation --config shaped.yaml
```

`shaped.yaml` sets `backend`, `base_url`, `model`, `answer_shape`. A config with `api_key:` is
refused with the reason.
