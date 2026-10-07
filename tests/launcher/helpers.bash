# Shared by every launcher test: sources the launcher's functions without
# running it, and runs rows of cases.json against them.

LAUNCHER="$BATS_TEST_DIRNAME/../../launcher/videoannotator-start"
CASES="$BATS_TEST_DIRNAME/cases.json"

setup_launcher() {
    export HOME="$BATS_TEST_TMPDIR/home"
    export XDG_CONFIG_HOME="$HOME/.config"
    mkdir -p "$HOME"
    unset VIDEOS_DIR RESULTS_DIR DISPLAY WAYLAND_DISPLAY
    # shellcheck source=../../launcher/videoannotator-start
    VA_SOURCE_ONLY=1 . "$LAUNCHER"
    # The launcher sets -eu; bats needs -e kept on to see failures.
    set +u
    VA_OS=linux
    VA_DOCKER_DESKTOP=no
    ENGINE_LOG="$BATS_TEST_TMPDIR/engine.log"
    : > "$ENGINE_LOG"
    # Every engine call is recorded, never run.
    va_engine() { printf '%s\n' "$*" >> "$ENGINE_LOG"; engine_stub "$@"; }
}

# Tests override this to answer engine calls; by default nothing exists.
engine_stub() {
    case "$1 $2" in
        "container inspect") return 1 ;;
        "image inspect") return 0 ;;
    esac
    return 0
}

# The rows of a group that apply to the sh launcher.
rows() {
    jq -c --arg g "$1" '.[$g][] | select(.os != "windows")' "$CASES"
}

field() { printf '%s' "$1" | jq -r "$2"; }

# run_cases GROUP CHECK: CHECK each row; report every failing row by name.
run_cases() {
    failures=0
    count=0
    while IFS= read -r row; do
        count=$((count + 1))
        if ! output=$("$2" "$row" 2>&1); then
            failures=$((failures + 1))
            printf '%s: %s\n%s\n' "$1" "$(field "$row" .name)" "$output" >&2
        fi
    done <<ROWS
$(rows "$1")
ROWS
    [ "$count" -gt 0 ] || { echo "no $1 rows" >&2; return 1; }
    [ "$failures" -eq 0 ]
}

expect_equal() {
    if [ "$1" != "$2" ]; then
        printf 'expected: %s\n     got: %s\n' "$2" "$1"
        return 1
    fi
}

# Makes the engines answer as a test wants: "running", "stopped" or "absent".
engines() {
    DOCKER_STATE=$1 PODMAN_STATE=$2
    STUB_BIN="$BATS_TEST_TMPDIR/bin"
    mkdir -p "$STUB_BIN"
    rm -f "$STUB_BIN/docker" "$STUB_BIN/podman"
    [ "$DOCKER_STATE" = absent ] || printf '#!/bin/sh\nexit 0\n' > "$STUB_BIN/docker"
    [ "$PODMAN_STATE" = absent ] || printf '#!/bin/sh\nexit 0\n' > "$STUB_BIN/podman"
    chmod +x "$STUB_BIN"/* 2>/dev/null || true
    PATH="$STUB_BIN:/usr/bin:/bin"
    # Not from PATH: CI runners have real docker and podman in /usr/bin.
    va_installed() {
        case "$1" in
            docker) [ "$DOCKER_STATE" != absent ] ;;
            podman) [ "$PODMAN_STATE" != absent ] ;;
            *) command -v "$1" >/dev/null 2>&1 ;;
        esac
    }
    va_engine_responds() {
        case "$1" in
            docker) [ "$DOCKER_STATE" = running ] ;;
            podman) [ "$PODMAN_STATE" = running ] ;;
        esac
    }
    va_engine_info_error() { printf '%s' "${ENGINE_STDERR:-}"; }
    va_podman_machine_start() { return 1; }
}

# A pretend engine and server for whole-command tests. State lives in files,
# because the launcher calls the engine inside command substitutions.
flow_setup() {
    setup_launcher
    engines absent running
    STATE="$BATS_TEST_TMPDIR/state"
    mkdir -p "$STATE"
    echo none > "$STATE/container"
    echo 0 > "$STATE/running_jobs"
    VA_NO_BROWSER=1
    VA_UID=1000 VA_GID=1000
    engine_stub() {
        case "$1" in
            container)
                case "$(cat "$STATE/container")" in
                    none) return 1 ;;
                    running) case "$*" in *Running*) echo true ;; esac ;;
                    stopped) case "$*" in *Running*) echo false ;; esac ;;
                esac ;;
            run) echo running > "$STATE/container" ;;
            stop) [ "$(cat "$STATE/container")" = none ] || echo stopped > "$STATE/container" ;;
            rm) echo none > "$STATE/container" ;;
            exec) case "$*" in *"cat /tmp/key.json"*) echo '{"token": "va_new_key", "user": "researcher"}' ;; esac ;;
            image) [ -z "${IMAGE_MISSING:-}" ] || return 1 ;;
            volume) [ -n "${OLD_VOLUMES:-}" ] || return 1 ;;
        esac
        return 0
    }
    va_http() {
        case "$1" in
            */api/v1/system/health) [ "$(cat "$STATE/container")" = running ] ;;
            */api/v1/auth/me) [ "${2:-}" = va_new_key ] || [ "${2:-}" = "${VALID_KEY:-}" ] ;;
            */api/v1/jobs/*)
                n=$(cat "$STATE/running_jobs")
                # Each look at the running jobs finds one fewer.
                [ "$n" -gt 0 ] && echo $((n - 1)) > "$STATE/running_jobs"
                printf '{"jobs": [], "total": %s, "page": 1}' "$n" ;;
            *) return 1 ;;
        esac
    }
    va_port_answers() { return 1; }
    sleep() { :; }
}

# Writes start.conf directly.
saved() {
    mkdir -p "$XDG_CONFIG_HOME/videoannotator/requests"
    printf '%s\n' "$@" > "$XDG_CONFIG_HOME/videoannotator/start.conf"
}

engine_runs() { grep -c '^run ' "$ENGINE_LOG" || true; }
