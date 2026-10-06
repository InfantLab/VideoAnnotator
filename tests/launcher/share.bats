#!/usr/bin/env bats
# Share another folder, or stop sharing one (Story 3, research R10, R11).

load helpers

setup() {
    flow_setup
    mkdir -p "$HOME/Studies" "$HOME/Second" "$HOME/VideoAnnotator"
    saved engine=podman "results=$HOME/VideoAnnotator" key=va_new_key "share=$HOME/Studies"
}

@test "share PATH adds it and restarts" {
    echo running > "$STATE/container"
    run main share "$HOME/Second/"
    [ "$status" -eq 0 ]
    [[ "$output" == *"Restarting VideoAnnotator to share $HOME/Second..."* ]]
    grep -qx "share=$HOME/Second" "$(va_settings_path)"
    grep -q "^stop -t 30 videoannotator" "$ENGINE_LOG"
    grep -q "source=$HOME/Second,target=$HOME/Second,readonly" "$ENGINE_LOG"
}

@test "share of a folder inside a share changes nothing" {
    mkdir -p "$HOME/Studies/Day 1"
    run main share "$HOME/Studies/Day 1"
    [ "$status" -eq 1 ]
    [ "$(engine_runs)" -eq 0 ]
}

@test "share with no path uses the picker and confirms" {
    va_pick_folder() { printf '%s\n' "$HOME/Second"; }
    run main share <<< "y"
    [ "$status" -eq 0 ]
    [[ "$output" == *"Share this folder? [Y/n]"* ]]
    grep -qx "share=$HOME/Second" "$(va_settings_path)"
}

@test "unshare PATH removes it and restarts" {
    saved engine=podman "results=$HOME/VideoAnnotator" key=va_new_key \
        "share=$HOME/Studies" "share=$HOME/Second"
    run main unshare "$HOME/Studies"
    [ "$status" -eq 0 ]
    ! grep -qx "share=$HOME/Studies" "$(va_settings_path)"
    grep -qx "share=$HOME/Second" "$(va_settings_path)"
    ! grep -q "source=$HOME/Studies," "$ENGINE_LOG"
}

@test "unshare with no path offers a numbered list" {
    saved engine=podman "results=$HOME/VideoAnnotator" key=va_new_key \
        "share=$HOME/Studies" "share=$HOME/Second"
    run main unshare <<< "2"
    [ "$status" -eq 0 ]
    [[ "$output" == *"  1. $HOME/Studies"*"  2. $HOME/Second"* ]]
    ! grep -qx "share=$HOME/Second" "$(va_settings_path)"
}

@test "stop requests from Settings are applied at start, then deleted" {
    saved engine=podman "results=$HOME/VideoAnnotator" key=va_new_key \
        "share=$HOME/Studies" "share=$HOME/Second"
    requests="$XDG_CONFIG_HOME/videoannotator/requests/stop-sharing.txt"
    printf '%s\n' "$HOME/Studies" "/etc" "$HOME/Studies" > "$requests"
    run main < /dev/null
    [ "$status" -eq 0 ]
    [[ "$output" == *"Stopped sharing $HOME/Studies, as asked in Settings."* ]]
    ! grep -qx "share=$HOME/Studies" "$(va_settings_path)"
    ! grep -qx "share=/etc" "$(va_settings_path)"
    grep -qx "share=$HOME/Second" "$(va_settings_path)"
    [ ! -e "$requests" ]
}

@test "a request can never add a share" {
    requests="$XDG_CONFIG_HOME/videoannotator/requests/stop-sharing.txt"
    printf '%s\n' "/home" > "$requests"
    run main < /dev/null
    [ "$(grep -c '^share=' "$(va_settings_path)")" -eq 1 ]
}

@test "running jobs: asks, and waits by default" {
    echo running > "$STATE/container"
    echo 2 > "$STATE/running_jobs"
    run main share "$HOME/Second" <<< ""
    [ "$status" -eq 0 ]
    [[ "$output" == *"2 videos are being processed. [W]ait for them, or [r]estart now (they'll be marked failed and can be retried)?"* ]]
    [[ "$output" == *"Waiting for them to finish..."* ]]
    [ "$(cat "$STATE/running_jobs")" -eq 0 ]
}

@test "running jobs: restart now" {
    echo running > "$STATE/container"
    echo 1 > "$STATE/running_jobs"
    run main share "$HOME/Second" <<< "r"
    [ "$status" -eq 0 ]
    [[ "$output" == *"1 video is being processed."* ]]
    [[ "$output" != *"Waiting"* ]]
}

@test "running jobs with --yes restart without asking" {
    echo running > "$STATE/container"
    echo 3 > "$STATE/running_jobs"
    run main share "$HOME/Second" --yes < /dev/null
    [ "$status" -eq 0 ]
    [[ "$output" != *"being processed"* ]]
}
