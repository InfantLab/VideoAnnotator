#!/usr/bin/env bats
# Later starts ask nothing (Story 2).

load helpers

setup() {
    flow_setup
    mkdir -p "$HOME/Studies" "$HOME/VideoAnnotator"
    saved engine=podman "image=ghcr.io/infantlab/videoannotator:latest" \
        "results=$HOME/VideoAnnotator" key=va_new_key "share=$HOME/Studies"
}

@test "a later start asks nothing and runs the saved folders" {
    run main < /dev/null
    [ "$status" -eq 0 ]
    [[ "$output" != *"?"* ]]
    [ "$(engine_runs)" -eq 1 ]
    grep -q "source=$HOME/Studies,target=$HOME/Studies,readonly" "$ENGINE_LOG"
    [[ "$output" == *"Starting... ready."* ]]
    [[ "$output" == *"VideoAnnotator can read: $HOME/Studies. Results: $HOME/VideoAnnotator."* ]]
    [[ "$output" == *"Opening http://127.0.0.1:18011/viewer in your browser."* ]]
    [[ "$output" != *"Starting VideoAnnotator with"* ]]
}

@test "the same start twice builds the same run" {
    run main < /dev/null
    first=$(grep '^run ' "$ENGINE_LOG")
    echo none > "$STATE/container"
    : > "$ENGINE_LOG"
    run main < /dev/null
    [ "$(grep '^run ' "$ENGINE_LOG")" = "$first" ]
}

@test "a missing share is skipped with a note and kept" {
    saved engine=podman "results=$HOME/VideoAnnotator" key=va_new_key \
        "share=$HOME/Studies" "share=/media/ada/Unplugged"
    run main < /dev/null
    [ "$status" -eq 0 ]
    [[ "$output" == *"Couldn't find /media/ada/Unplugged (an unplugged drive?), so it isn't shared this time."* ]]
    ! grep -q "Unplugged,target" "$ENGINE_LOG"
    grep -q "VIDEOANNOTATOR_MISSING_SHARES=/media/ada/Unplugged" "$ENGINE_LOG"
    grep -qx "share=/media/ada/Unplugged" "$(va_settings_path)"
}

@test "already running: opens the browser, starts nothing" {
    echo running > "$STATE/container"
    run main < /dev/null
    [ "$status" -eq 0 ]
    [[ "$output" == *"VideoAnnotator is already running."* ]]
    [[ "$output" == *"Opening http://127.0.0.1:18011/viewer"* ]]
    [ "$(engine_runs)" -eq 0 ]
}

@test "started twice at once: the second connects to the first" {
    engine_stub() {
        case "$1" in
            container) return 1 ;;
            run) echo 'Error: the container name "videoannotator" is already in use by 4f2' >&2; return 125 ;;
        esac
        return 0
    }
    run main < /dev/null
    [ "$status" -eq 0 ]
    [[ "$output" == *"VideoAnnotator is already running."* ]]
}

@test "a saved older image is replaced by the launcher's own version" {
    saved engine=podman "image=ghcr.io/infantlab/videoannotator:1.5.0" \
        "results=$HOME/VideoAnnotator" key=va_new_key "share=$HOME/Studies"
    VA_VERSION=1.6.0
    run main < /dev/null
    [ "$status" -eq 0 ]
    [ "$(grep '^run ' "$ENGINE_LOG" | awk '{print $NF}')" = ghcr.io/infantlab/videoannotator:1.6.0 ]
    grep -qx "image=ghcr.io/infantlab/videoannotator:1.6.0" "$(va_settings_path)"
}

@test "a rejected key is made again" {
    saved engine=podman "results=$HOME/VideoAnnotator" key=va_stale "share=$HOME/Studies"
    run main < /dev/null
    [ "$status" -eq 0 ]
    grep -q "exec videoannotator videoannotator generate-token" "$ENGINE_LOG"
    grep -qx "key=va_new_key" "$(va_settings_path)"
}

@test "the first download says so" {
    IMAGE_MISSING=1
    run main < /dev/null
    [[ "$output" == *"Downloading VideoAnnotator (first time only, about 1 GB)..."* ]]
}

@test "a port in use says which, and the next one to try" {
    va_port_answers() { return 0; }
    run main < /dev/null
    [ "$status" -eq 1 ]
    [[ "$output" == *"Something else is using port 18011. Close it, or run: videoannotator-start --port 18012"* ]]
}

@test "stop" {
    echo running > "$STATE/container"
    run main stop
    [ "$status" -eq 0 ]
    grep -q "^stop -t 30 videoannotator" "$ENGINE_LOG"
    [ "$(cat "$STATE/container")" = none ]
}
