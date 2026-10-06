#!/usr/bin/env bats
# Docker or Podman, whichever is running (research R2, Story 6).

load helpers

setup() { setup_launcher; }

check_engine() {
    VA_OS=$(field "$1" .os); [ "$VA_OS" = any ] && VA_OS=linux
    VA_DOCKER_DESKTOP=yes
    engines "$(field "$1" .input.docker)" "$(field "$1" .input.podman)"
    ENGINE_STDERR=$(field "$1" '.input.stderr // ""')
    VA_S_ENGINE=$(field "$1" .input.saved)
    VA_OPT_ENGINE=$(field "$1" .input.option)
    VA_ENGINE=""
    out=$(va_engine_detect) && rc=0 || rc=$?
    va_engine_detect > /dev/null || true
    expected=$(field "$1" '.expect.engine // ""')
    if [ -n "$expected" ]; then
        [ "$rc" -eq 0 ] || { echo "failed: $out"; return 1; }
        expect_equal "$VA_ENGINE" "$expected" || return 1
        if [ "$(field "$1" '.expect.announced // false')" = true ]; then
            expect_equal "$VA_BOTH_RUNNING" 1
        fi
    else
        [ "$rc" -ne 0 ] || { echo "expected a failure"; return 1; }
        expect_equal "$out" "$(field "$1" .expect.message)"
    fi
}

@test "engine: cases.json" { run_cases engine check_engine; }

@test "Podman's machine is started on macOS when it is stopped" {
    engines absent stopped
    VA_OS=macos
    va_podman_machine_start() { echo "Starting Podman's virtual machine (first time takes a minute)..."; PODMAN_STATE=running; }
    run va_engine_detect
    [ "$status" -eq 0 ]
    [ "$output" = "Starting Podman's virtual machine (first time takes a minute)..." ]
}

@test "the engine used is saved after a start" {
    engines running running
    VA_S_ENGINE=podman
    va_engine_detect
    [ "$VA_ENGINE" = podman ]
}
