#!/usr/bin/env bash
# Manual end-to-end test for vlm_annotation — Option B from
# docs/development/vlm_annotation_pipeline.md ("CLI + API server").
#
# Starts `videoannotator server` (AUTH_REQUIRED=false, local testing only),
# submits a job against temp/lw002.mp4, polls until it finishes, and prints
# results. Requires Ollama running with the target model pulled and
# VideoAnnotator installed with the `llm` extra.
#
# Usage:
#   tests/manual/vlm_annotation/server_test.sh
#   MODEL=qwen3.5:9b SAMPLING_MODE=frame_burst tests/manual/vlm_annotation/server_test.sh
#
# From inside a devcontainer/Docker container with Ollama on the host machine:
#   OLLAMA_BASE_URL=http://host.docker.internal:11434 tests/manual/vlm_annotation/server_test.sh
# and make sure Ollama is listening on all interfaces, not just loopback
# (OLLAMA_HOST=0.0.0.0 before starting it on the host) — 127.0.0.1 on the host
# is not reachable from a container.
#
# Do not run this alongside a separate `videoannotator worker` process against
# the same job database — see "Known gaps" in the doc above.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VIDEO="${VIDEO:-$REPO_ROOT/temp/lw002.mp4}"
MODEL="${MODEL:-qwen3.5:9b}"
SAMPLING_MODE="${SAMPLING_MODE:-single_frame}"
FRAME_INTERVAL_SEC="${FRAME_INTERVAL_SEC:-5.0}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
PORT="${PORT:-18011}"
SERVER="http://127.0.0.1:${PORT}"
CONFIG_FILE="$(dirname "${BASH_SOURCE[0]}")/vlm_config.json"

if [[ ! -f "$VIDEO" ]]; then
    echo "Video not found: $VIDEO" >&2
    exit 1
fi

cat > "$CONFIG_FILE" <<EOF
{
  "vlm_annotation": {
    "model": "${MODEL}",
    "sampling_mode": "${SAMPLING_MODE}",
    "frame_interval_sec": ${FRAME_INTERVAL_SEC},
    "base_url": "${OLLAMA_BASE_URL}"
  }
}
EOF

echo "Starting server on port ${PORT} (AUTH_REQUIRED=false)..."
# --host 0.0.0.0: required to be reachable from outside the container (e.g.
# a browser on the host via a forwarded port) — the CLI's own default of
# 127.0.0.1 is loopback-only within the container.
AUTH_REQUIRED=false videoannotator server --host 0.0.0.0 --port "$PORT" &
SERVER_PID=$!
trap 'echo "Stopping server (pid $SERVER_PID)..."; kill "$SERVER_PID" 2>/dev/null || true' EXIT

echo "Waiting for server to come up..."
for _ in $(seq 1 30); do
    if curl -sf "${SERVER}/health" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

echo "Submitting job for ${VIDEO}..."
JOB_ID=$(videoannotator job submit "$VIDEO" \
    --pipelines vlm_annotation \
    --config "$CONFIG_FILE" \
    --server "$SERVER" | tee /dev/stderr | grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' | head -1)

if [[ -z "$JOB_ID" ]]; then
    echo "Could not parse job ID from submit output" >&2
    exit 1
fi
echo "Job ID: $JOB_ID"

echo "Polling status..."
for _ in $(seq 1 60); do
    STATUS_OUTPUT=$(videoannotator job status "$JOB_ID" --server "$SERVER")
    echo "$STATUS_OUTPUT"
    if echo "$STATUS_OUTPUT" | grep -qiE 'completed|failed'; then
        break
    fi
    sleep 5
done

echo "Results:"
videoannotator job results "$JOB_ID" --server "$SERVER"
