#!/usr/bin/env bash
# One-command VideoAnnotator startup: syncs dependencies, ensures the
# database + an admin API key exist, starts the API server, and — since
# Video Annotation Viewer is bundled and served by this same process at
# /viewer, not a separate frontend to start — prints the ready-to-click
# viewer login link once the server is actually up. Wraps the same
# `videoannotator setup-db`/`server` CLI commands from the README's "Get
# Started in 60 Seconds" quick start; this script just sequences and
# idempotency-checks them so a first-time user (or someone restarting after
# a container/VS Code restart) doesn't have to remember the order or which
# flags matter.
#
# Usage:
#   scripts/start_server.sh                         # interactive, foreground
#   scripts/start_server.sh --background             # detached, survives this terminal closing
#   scripts/start_server.sh --port 18012 --admin-email me@example.com
#   scripts/start_server.sh --non-interactive         # never prompts; uses defaults/flags/env only
#
# Env var equivalents (flags take precedence): VIDEOANNOTATOR_HOST, PORT,
# ADMIN_EMAIL, ADMIN_USERNAME, SYNC_EXTRAS (e.g. "llm" or "all").

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$HERE/.." && pwd)"
cd "$ROOT_DIR"

HOST="${VIDEOANNOTATOR_HOST:-0.0.0.0}"
PORT="${PORT:-18011}"
ADMIN_EMAIL="${ADMIN_EMAIL:-}"
ADMIN_USERNAME="${ADMIN_USERNAME:-}"
SYNC_EXTRAS="${SYNC_EXTRAS:-}"
BACKGROUND=false
SKIP_SYNC=false
NON_INTERACTIVE=false
[ -t 0 ] || NON_INTERACTIVE=true

usage() {
    sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --admin-email) ADMIN_EMAIL="$2"; shift 2 ;;
        --admin-username) ADMIN_USERNAME="$2"; shift 2 ;;
        --extra) SYNC_EXTRAS="$2"; shift 2 ;;
        --background|-d) BACKGROUND=true; shift ;;
        --skip-sync) SKIP_SYNC=true; shift ;;
        --non-interactive) NON_INTERACTIVE=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv is not installed. Install it first:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

# --- 1. Dependencies -------------------------------------------------------
# --inexact: a bare `uv sync` defaults to an *exact* sync -- it removes any
# package not required by this invocation's extras. Without it, restarting
# via this script would silently uninstall any extras group installed
# out-of-band (manually, or via the in-app extras-install API from
# specs/005-pipeline-extras-install) since the last time SYNC_EXTRAS/--extra
# named that group here. --inexact makes every sync additive-only, matching
# what a restart-to-activate workflow actually needs.
if [[ "$SKIP_SYNC" == true ]]; then
    echo "[SKIP] Dependency sync (--skip-sync)"
else
    echo "[START] Syncing dependencies..."
    if [[ -n "$SYNC_EXTRAS" ]]; then
        uv sync --extra "$SYNC_EXTRAS" --inexact
    else
        uv sync --inexact
    fi
    echo "[OK] Dependencies synced"
fi

# --- 2. Admin identity (only asked on a genuine first-time setup) ----------
# setup-db is idempotent and won't touch an existing admin, so once a
# database is already on disk these values are dead input — skip the
# prompts entirely rather than asking on every restart. `setup-db` reads its
# DB location from DATABASE_URL (default "sqlite:///./videoannotator.db"),
# not VIDEOANNOTATOR_DB_PATH (that one only affects the API's own storage
# backend), so mirror that same lookup here.
DB_URL="${DATABASE_URL:-sqlite:///./videoannotator.db}"
DB_EXISTS=false
if [[ "$DB_URL" == sqlite:///* ]]; then
    DB_PATH="${DB_URL#sqlite:///}"
    [[ "$DB_PATH" != /* ]] && DB_PATH="$ROOT_DIR/$DB_PATH"
    [[ -f "$DB_PATH" ]] && DB_EXISTS=true
else
    # Non-SQLite (e.g. Postgres) URL: assume it's a pre-existing, managed DB.
    DB_EXISTS=true
fi

if [[ -z "$ADMIN_EMAIL" ]]; then
    if [[ "$NON_INTERACTIVE" == true || "$DB_EXISTS" == true ]]; then
        ADMIN_EMAIL="admin@videoannotator.local"
    else
        read -r -p "Admin email [admin@videoannotator.local]: " ADMIN_EMAIL
        ADMIN_EMAIL="${ADMIN_EMAIL:-admin@videoannotator.local}"
    fi
fi
if [[ -z "$ADMIN_USERNAME" ]]; then
    default_username="${ADMIN_EMAIL%%@*}"
    if [[ "$NON_INTERACTIVE" == true || "$DB_EXISTS" == true ]]; then
        ADMIN_USERNAME="$default_username"
    else
        read -r -p "Admin username [$default_username]: " ADMIN_USERNAME
        ADMIN_USERNAME="${ADMIN_USERNAME:-$default_username}"
    fi
fi

# --- 3. Database + admin API key -------------------------------------------
# `setup-db` is idempotent: creates the schema if missing, and only prints a
# fresh admin key/viewer-connect link the first time the admin user is
# created. Safe to call on every startup. --port only affects the printed
# link/example, not the database itself.
echo "[START] Ensuring database and admin API key exist..."
SETUP_OUTPUT="$(uv run videoannotator setup-db --admin-email "$ADMIN_EMAIL" --admin-username "$ADMIN_USERNAME" --port "$PORT")"
echo "$SETUP_OUTPUT"
CONNECT_LINK="$(echo "$SETUP_OUTPUT" | grep -oE 'http://[^[:space:]]+/viewer-connect\?token=[^[:space:]]+' || true)"

DISPLAY_HOST="$HOST"
[[ "$HOST" == "0.0.0.0" ]] && DISPLAY_HOST="localhost"

print_viewer_info() {
    if [[ "${VIDEOANNOTATOR_ENABLE_VIEWER:-true}" == "false" ]]; then
        echo "[INFO] Viewer disabled (VIDEOANNOTATOR_ENABLE_VIEWER=false)"
        return
    fi
    # Video Annotation Viewer isn't a separate process — this same server
    # already serves it at /viewer, mounted whenever ENABLE_VIEWER is true.
    if [[ -n "$CONNECT_LINK" ]]; then
        echo "[VIEWER] First time here — open this to log the viewer in:"
        echo "  $CONNECT_LINK"
    else
        echo "[VIEWER] http://${DISPLAY_HOST}:${PORT}/viewer"
        echo "  (no fresh API key this run — admin already existed; if the viewer isn't"
        echo "  logged in, run: uv run videoannotator generate-token --port $PORT)"
    fi
}

# --- 4. Start the server -----------------------------------------------
# Started via `&` (not `exec`) in both modes so this script can poll for
# health and print the viewer link once it's actually reachable — `exec`
# would hand off the process image and end the script right there.
echo "[START] Starting server..."
if [[ "$BACKGROUND" == true ]]; then
    LOG_FILE="$(mktemp -t videoannotator-server-XXXXXX.log)"
    uv run videoannotator server --host "$HOST" --port "$PORT" > "$LOG_FILE" 2>&1 &
else
    # No redirection here: output streams straight to this terminal, same as
    # running the command directly — that's the point of foreground mode.
    uv run videoannotator server --host "$HOST" --port "$PORT" &
fi
SERVER_PID=$!

for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[OK] Server is up: http://${DISPLAY_HOST}:${PORT}"
        print_viewer_info
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[ERROR] Server process exited before becoming healthy" >&2
        [[ -n "${LOG_FILE:-}" ]] && cat "$LOG_FILE" >&2
        exit 1
    fi
    sleep 1
done

if [[ "$BACKGROUND" == true ]]; then
    disown
    echo "  Logs:  tail -f $LOG_FILE"
    echo "  Stop:  kill $SERVER_PID"
else
    echo "  (Ctrl+C stops the server)"
    trap 'kill "$SERVER_PID" 2>/dev/null || true' INT TERM
    wait "$SERVER_PID"
fi
