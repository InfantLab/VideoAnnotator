#!/usr/bin/env bash
# Run the API server and the viewer's Vite dev server together, both reloading
# on change. Open http://127.0.0.1:19011 : Vite proxies /api to the server.
#
#   bash scripts/dev.sh          # auth off (the server's --dev mode)
#   bash scripts/dev.sh --auth   # auth on: paste a key in the viewer's Settings
#
# Ctrl+C stops both.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_PORT="${API_PORT:-18011}"

server_args=(--host 127.0.0.1 --port "$API_PORT" --reload)
case "${1:-}" in
  "") server_args+=(--dev) ;;
  --auth) ;;
  *) echo "usage: $0 [--auth]" >&2; exit 2 ;;
esac

if ! command -v bun >/dev/null 2>&1; then
  echo "bun not found: install it from https://bun.sh (the dev container has it)" >&2
  exit 1
fi

cleanup() {
  trap - EXIT INT TERM
  kill 0 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$ROOT_DIR"
uv run videoannotator server "${server_args[@]}" &
(cd viewer && bun install --frozen-lockfile >/dev/null && bun run dev) &

echo "API:    http://127.0.0.1:${API_PORT}/docs"
echo "Viewer: http://127.0.0.1:19011  (hot reload; /api proxied to the API)"
wait
