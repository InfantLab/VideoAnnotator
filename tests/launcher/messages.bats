#!/usr/bin/env bats
# One plain line for each thing that can go wrong (contracts/launcher.md, Messages).

load helpers

setup() { setup_launcher; }

check_message() {
    VA_OS=$(field "$1" .os); [ "$VA_OS" = any ] && VA_OS=linux
    VA_PORT=18011
    out=$(va_explain_error "$(field "$1" .input.engine)" "$(field "$1" .input.stderr)" "$(field "$1" .input.code)") && rc=0 || rc=$?
    expect_equal "$(printf '%s\n' "$out" | head -n 1)" "$(field "$1" .expect.message)" || return 1
    expect_equal "$rc" "$(field "$1" .expect.status)" || return 1
    detail=$(field "$1" '.expect.detail // ""')
    if [ -n "$detail" ]; then expect_equal "$(printf '%s\n' "$out" | sed -n 2p)" "$detail"; fi
}

@test "messages: cases.json" { run_cases messages check_message; }

@test "Docker Desktop on Linux is told to start the app, not the service" {
    VA_OS=linux VA_DOCKER_DESKTOP=yes
    run va_explain_error docker "Cannot connect to the Docker daemon. Is the docker daemon running?" 1
    [ "$output" = "Docker Desktop isn't running. Start it, wait until it says it's running, then run this again." ]
}

@test "no message uses container words without explaining them" {
    for word in container volume mount image; do
        ! jq -r '.messages[].expect.message' "$CASES" | grep -qiw "$word"
    done
}
