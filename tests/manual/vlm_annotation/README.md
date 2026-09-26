# vlm_annotation manual tests

Manual (non-pytest) scripts for exercising the `vlm_annotation` pipeline
against a real, locally-running Ollama server. See
`docs/development/vlm_annotation_pipeline.md` for the full write-up — these
scripts are that doc's "Step-by-step: test it yourself" section, kept
runnable instead of copy-pasted from markdown.

Not collected by pytest; run directly. Both require Ollama running with the
target model pulled (`ollama pull <model>`) and VideoAnnotator installed with
the `llm` extra (`uv pip install -e ".[llm]"`).

## Option A — direct Python smoke test (no server, no auth)

```bash
python tests/manual/vlm_annotation/smoke_test.py
python tests/manual/vlm_annotation/smoke_test.py --model qwen3.5:9b --sampling-mode frame_burst
```

Talks straight to Ollama. Output lands in `tests/manual/vlm_annotation/output/`.

## Option B — CLI + API server

```bash
tests/manual/vlm_annotation/server_test.sh
MODEL=qwen3.5:9b SAMPLING_MODE=frame_burst tests/manual/vlm_annotation/server_test.sh
```

Starts `videoannotator server` with `AUTH_REQUIRED=false`, submits a job,
polls for completion, and prints results. Don't run a separate
`videoannotator worker` process against the same job database at the same
time (see the doc's "Known gaps" section).

Both default to `temp/lw002.mp4` — override with `--video` (script A) or
`VIDEO=` (script B).

## Running from a devcontainer with Ollama on the host

`127.0.0.1:11434` inside a container is the container's own loopback, not the
host's — it won't reach Ollama running on your laptop. Two changes needed:

1. On the host, make sure Ollama listens on all interfaces, not just
   loopback: set `OLLAMA_HOST=0.0.0.0:11434` before starting it (or in
   whatever env-config mechanism the Ollama app uses on your OS), then
   restart it.
2. From the container, point `base_url` at `http://host.docker.internal:11434`
   instead of `127.0.0.1`:
   ```bash
   python tests/manual/vlm_annotation/smoke_test.py --base-url http://host.docker.internal:11434
   OLLAMA_BASE_URL=http://host.docker.internal:11434 tests/manual/vlm_annotation/server_test.sh
   ```

This devcontainer already resolves `host.docker.internal` (Docker Desktop's
WSL2 backend provides it). For the API server itself, `devcontainer.json` and
`docker-compose.yml` set `OLLAMA_BASE_URL=http://host.docker.internal:11434`,
which the server uses whenever a request or job doesn't name a `base_url`.
In a container created before that setting existed, rebuild it or export the
variable in the shell that starts the server.
