#!/usr/bin/env bats
# update: the launcher first, then its own image (research R16).

load helpers

setup() {
    flow_setup
    mkdir -p "$HOME/Studies"
    saved engine=podman "image=ghcr.io/infantlab/videoannotator:1.6.0" \
        "results=$HOME/VideoAnnotator" key=va_new_key "share=$HOME/Studies"
    VA_VERSION=1.6.0
    va_install_release() { echo "installed $1" > "$STATE/installed"; }
}

@test "a newer release updates the launcher, then runs it to continue" {
    va_latest_release() { echo v1.7.0; }
    mkdir -p "$HOME/.local/bin"
    printf '#!/bin/sh\necho "new launcher: $*"\n' > "$HOME/.local/bin/videoannotator-start"
    chmod +x "$HOME/.local/bin/videoannotator-start"
    run main update
    [ "$status" -eq 0 ]
    [ "$(cat "$STATE/installed")" = "installed v1.7.0" ]
    [[ "$output" == *"new launcher: update --continue"* ]]
}

@test "already the latest, with nothing new to download: up to date, no restart" {
    va_latest_release() { echo v1.6.0; }
    engine_stub() {
        case "$1 $2" in
            "image inspect") echo sha256:same ;;
            "container inspect") case "$*" in *Image*) echo sha256:same ;; *Running*) echo true ;; esac ;;
        esac
        return 0
    }
    run main update
    [ "$status" -eq 0 ]
    [ "$output" = "VideoAnnotator is up to date." ]
    [ "$(engine_runs)" -eq 0 ]
}

@test "--continue pulls the pinned image, saves it and restarts" {
    saved engine=podman "image=ghcr.io/infantlab/videoannotator:1.5.0" \
        "results=$HOME/VideoAnnotator" key=va_new_key "share=$HOME/Studies"
    echo running > "$STATE/container"
    run main update --continue
    [ "$status" -eq 0 ]
    grep -q "^pull ghcr.io/infantlab/videoannotator:1.6.0" "$ENGINE_LOG"
    grep -qx "image=ghcr.io/infantlab/videoannotator:1.6.0" "$(va_settings_path)"
    [[ "$output" == *"Updated. Your folders, results, models and installed pipelines are kept."* ]]
    [ "$(engine_runs)" -eq 1 ]
}

@test "versions compare by number" {
    va_newer v1.10.0 1.9.2
    va_newer 1.6.1 1.6.0
    ! va_newer v1.6.0 1.6.0
    ! va_newer 1.5.9 1.6.0
}
