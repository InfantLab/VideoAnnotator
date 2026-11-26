#!/usr/bin/env bash
# One-command VideoAnnotator startup: syncs dependencies, ensures the
# database + an admin API key exist, starts the API server, and prints the
# viewer link once the server is actually up (the viewer is served by this
# same process at /viewer). If VideoAnnotator is already running on the port,
# it offers to use that one instead of failing.
#
# Usage:
#   scripts/start_server.sh                  # start (or offer the running one)
#   scripts/start_server.sh --background     # detached, survives this terminal closing
#   scripts/start_server.sh --port 18012 --admin-email me@example.com
#   scripts/start_server.sh --status         # is it running? where?
#   scripts/start_server.sh --stop           # stop the server on the port
#   scripts/start_server.sh --use-existing   # if one is running, just use it
#   scripts/start_server.sh --restart        # if one is running, stop it and start fresh
#   scripts/start_server.sh --non-interactive  # never prompts (uses a running server if any)
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
# What to do when a server already holds the port: ask | use | restart.
IF_RUNNING=ask
MODE=start

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
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
        --use-existing) IF_RUNNING=use; shift ;;
        --restart) IF_RUNNING=restart; shift ;;
        --status) MODE=status; shift ;;
        --stop) MODE=stop; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv is not installed. Install it first:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

DISPLAY_HOST="$HOST"
# 127.0.0.1, not localhost: the viewer only works from that origin.
[[ "$HOST" == "0.0.0.0" || "$HOST" == "localhost" ]] && DISPLAY_HOST="127.0.0.1"

# --- 0. Is something already on the port? ---------------------------------
# Checked before anything else: someone re-running va-start with a server
# already up almost always wants that server, and finding out shouldn't wait
# on a dependency sync.

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

# Generous timeout: a server busy analysing video can be slow to answer.
health_json() { curl -sf -m 5 "http://127.0.0.1:$1/health" 2>/dev/null; }

# PID listening on a port. lsof (macOS, most Linux), then ss, then /proc
# (minimal containers, such as the dev container, have neither tool).
listener_pid() {
    local port="$1" pid="" hex inode fd
    if command -v lsof >/dev/null 2>&1; then
        pid="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -1 || true)"
    elif command -v ss >/dev/null 2>&1; then
        pid="$(ss -ltnpH "sport = :$port" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2 || true)"
    elif [[ -r /proc/net/tcp ]]; then
        hex="$(printf '%04X' "$port")"
        inode="$(awk -v p=":$hex" '$2 ~ p"$" && $4 == "0A" {print $10; exit}' \
            /proc/net/tcp /proc/net/tcp6 2>/dev/null || true)"
        if [[ -n "$inode" ]]; then
            for fd in /proc/[0-9]*/fd/*; do
                if [[ "$(readlink "$fd" 2>/dev/null)" == "socket:[$inode]" ]]; then
                    pid="${fd#/proc/}"; pid="${pid%%/*}"; break
                fi
            done
        fi
    fi
    echo "$pid"
}

next_free_port() {
    local p
    for p in $(seq $(( $1 + 1 )) $(( $1 + 20 ))); do
        port_open "$p" || { echo "$p"; return; }
    done
}

# Sets RUNNING (none | va | busy | other), RUNNING_PID and RUNNING_DESC.
probe_port() {
    local json pid args started version
    RUNNING=none RUNNING_PID="" RUNNING_DESC=""
    port_open "$PORT" || return 0
    pid="$(listener_pid "$PORT")"
    args="" started=""
    if [[ -n "$pid" ]]; then
        args="$(ps -o args= -p "$pid" 2>/dev/null || true)"
        started="$(ps -o lstart= -p "$pid" 2>/dev/null | sed 's/  */ /g; s/^ //' || true)"
    fi
    RUNNING_PID="$pid"
    json="$(health_json "$PORT" || true)"
    if [[ "$json" == *"VideoAnnotator API is running"* ]]; then
        RUNNING=va
        version="$(sed -n 's/.*"videoannotator_version":"\([^"]*\)".*/\1/p' <<<"$json")"
        RUNNING_DESC="version ${version:-unknown}"
        # The server's own clock beats ps's lstart, which drifts in containers.
        started="$(sed -n 's/.*"started_at":"\([0-9-]*\)T\([0-9]*:[0-9]*\).*/\1 \2 UTC/p' <<<"$json")"
    elif [[ -z "$json" && ( -z "$args" || "$args" == *videoannotator* ) ]]; then
        RUNNING=busy
    else
        RUNNING=other
        RUNNING_DESC="${args:-an unknown program}"
    fi
    [[ -n "$started" ]] && RUNNING_DESC="${RUNNING_DESC:+$RUNNING_DESC, }started $started"
    [[ -n "$pid" ]] && RUNNING_DESC="${RUNNING_DESC:+$RUNNING_DESC, }process $pid"
    return 0
}

describe_running() {
    case "$RUNNING" in
        va)
            echo "VideoAnnotator is already running on port $PORT."
            echo "  Viewer:  http://${DISPLAY_HOST}:${PORT}/viewer"
            echo "  ($RUNNING_DESC)"
            ;;
        busy)
            echo "A VideoAnnotator server is on port $PORT but isn't answering."
            echo "  It may be busy with a heavy run, or stuck.${RUNNING_DESC:+ ($RUNNING_DESC)}"
            ;;
        other)
            echo "Port $PORT is in use by another program, not VideoAnnotator:"
            echo "  $RUNNING_DESC"
            ;;
    esac
}

stop_running() {
    local pid="$RUNNING_PID" _
    if [[ -z "$pid" ]]; then
        echo "[ERROR] Can't tell which process holds port $PORT, so can't stop it." >&2
        echo "  Look for it with: ps aux | grep videoannotator" >&2
        return 1
    fi
    echo "[STOP] Stopping the server on port $PORT (process $pid)..."
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 15); do
        port_open "$PORT" || { echo "[OK] Stopped"; return 0; }
        sleep 1
    done
    echo "[STOP] Still running after 15s; forcing it to stop..."
    kill -9 "$pid" 2>/dev/null || true
    for _ in $(seq 1 5); do
        port_open "$PORT" || { echo "[OK] Stopped"; return 0; }
        sleep 1
    done
    echo "[ERROR] Port $PORT is still in use." >&2
    return 1
}

probe_port

if [[ "$MODE" == status ]]; then
    if [[ "$RUNNING" == none ]]; then
        echo "VideoAnnotator is not running on port $PORT."
        echo "  Start it with: va-start  (or scripts/start_server.sh)"
        exit 3
    fi
    describe_running
    [[ "$RUNNING" == va ]] && exit 0 || exit 1
fi

if [[ "$MODE" == stop ]]; then
    if [[ "$RUNNING" == none ]]; then
        echo "Nothing is running on port $PORT."
        exit 0
    fi
    if [[ "$RUNNING" == other ]]; then
        describe_running
        echo "Not stopping it: it isn't VideoAnnotator." >&2
        exit 1
    fi
    stop_running
    exit $?
fi

if [[ "$RUNNING" != none ]]; then
    echo
    describe_running
    echo
    choice=""
    case "$IF_RUNNING" in
        use) [[ "$RUNNING" == va ]] && choice=use ;;
        restart) [[ "$RUNNING" != other ]] && choice=restart ;;
    esac
    ALT_PORT="$(next_free_port "$PORT")"
    if [[ -z "$choice" && "$NON_INTERACTIVE" == true ]]; then
        if [[ "$RUNNING" == va ]]; then
            choice=use
        else
            echo "[ERROR] Can't start on port $PORT." >&2
            [[ "$RUNNING" == busy ]] && echo "  To stop it and start fresh: --restart" >&2
            [[ -n "$ALT_PORT" ]] && echo "  To start on another port: --port $ALT_PORT" >&2
            exit 1
        fi
    fi
    if [[ -z "$choice" ]]; then
        options=() labels=()
        if [[ "$RUNNING" == va ]]; then
            options+=(use);     labels+=("Use it (nothing else to do; open the link above)")
        fi
        if [[ "$RUNNING" != other ]]; then
            options+=(restart); labels+=("Restart it (stops it, then starts a fresh one here; any run in progress stops)")
        fi
        if [[ -n "$ALT_PORT" ]]; then
            options+=(alt);     labels+=("Start another one on port $ALT_PORT")
        fi
        options+=(cancel);      labels+=("Cancel")
        echo "What would you like to do?"
        for i in "${!options[@]}"; do
            echo "  $((i + 1))) ${labels[$i]}"
        done
        while [[ -z "$choice" ]]; do
            read -r -p "Choose 1-${#options[@]} [1]: " answer
            answer="${answer:-1}"
            if [[ "$answer" =~ ^[0-9]+$ ]] && (( answer >= 1 && answer <= ${#options[@]} )); then
                choice="${options[$((answer - 1))]}"
            fi
        done
        echo
    fi
    case "$choice" in
        use)
            echo "[OK] Using the server that's already running."
            echo "  Viewer: http://${DISPLAY_HOST}:${PORT}/viewer"
            echo "  If the viewer asks you to sign in: uv run videoannotator generate-token --port $PORT"
            echo "  To stop it: va-stop  (or scripts/start_server.sh --stop)"
            exit 0
            ;;
        restart) stop_running || exit 1 ;;
        alt) PORT="$ALT_PORT"; echo "[INFO] Starting on port $PORT instead." ;;
        cancel) exit 0 ;;
    esac
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

print_viewer_info() {
    if [[ "${VIDEOANNOTATOR_ENABLE_VIEWER:-true}" == "false" ]]; then
        echo "  Viewer:  disabled (VIDEOANNOTATOR_ENABLE_VIEWER=false)"
        return
    fi
    # Video Annotation Viewer isn't a separate process — this same server
    # already serves it at /viewer, mounted whenever ENABLE_VIEWER is true.
    if [[ -n "$CONNECT_LINK" ]]; then
        echo "  Viewer:  open this link to sign the viewer in (first time only):"
        echo "           $CONNECT_LINK"
    else
        echo "  Viewer:  http://${DISPLAY_HOST}:${PORT}/viewer"
        echo "           If it asks you to sign in: uv run videoannotator generate-token --port $PORT"
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

# Our process must still be alive before /health counts: otherwise a server
# that grabbed the port in the meantime would answer for it.
UP=false
for _ in $(seq 1 60); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo >&2
        echo "[ERROR] The server stopped before it was ready (see the messages above)." >&2
        [[ -n "${LOG_FILE:-}" ]] && cat "$LOG_FILE" >&2
        exit 1
    fi
    if health_json "$PORT" >/dev/null; then
        UP=true
        break
    fi
    sleep 1
done

echo
if [[ "$UP" == true ]]; then
    echo "[OK] VideoAnnotator is running on port $PORT."
    print_viewer_info
else
    echo "[WARN] The server hasn't answered after 60s; it may still be starting."
    echo "  Check with: va-status  (or scripts/start_server.sh --status)"
fi

if [[ "$BACKGROUND" == true ]]; then
    disown
    echo "  Logs:    tail -f $LOG_FILE"
    echo "  Stop:    va-stop  (or scripts/start_server.sh --stop)"
else
    echo "  Stop:    Ctrl+C here, or va-stop from another terminal"
    trap 'kill "$SERVER_PID" 2>/dev/null || true' INT TERM
    wait "$SERVER_PID"
fi
