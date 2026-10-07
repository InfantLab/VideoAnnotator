#!/usr/bin/env bats
# An earlier Docker setup carries over (FR-024, research R13).

load helpers

setup() {
    flow_setup
    mkdir -p "$HOME/Studies" "$HOME/Results"
}

@test "VIDEOS_DIR and RESULTS_DIR are offered and reused" {
    export VIDEOS_DIR="$HOME/Studies" RESULTS_DIR="$HOME/Results"
    run main <<< "y"
    [ "$status" -eq 0 ]
    [[ "$output" == *"Use your existing VideoAnnotator folders? [Y/n]"* ]]
    grep -qx "share=$HOME/Studies" "$(va_settings_path)"
    grep -qx "results=$HOME/Results" "$(va_settings_path)"
}

@test "with --yes they are reused without asking" {
    export VIDEOS_DIR="$HOME/Studies"
    run main --yes
    [ "$status" -eq 0 ]
    grep -qx "share=$HOME/Studies" "$(va_settings_path)"
}

@test "only the old volumes: they are kept, and the picker asks" {
    OLD_VOLUMES=1
    va_pick_folder() { printf '%s\n' "$HOME/Studies"; }
    run main <<< "y"
    [ "$status" -eq 0 ]
    [[ "$output" == *"Your jobs and models will be kept."*"Which folder are your videos in?"* ]]
}

@test "nothing is offered once start.conf exists" {
    saved "results=$HOME/VideoAnnotator" "share=$HOME/Studies"
    export VIDEOS_DIR="$HOME/Results"
    run main < /dev/null
    [[ "$output" != *"existing VideoAnnotator folders"* ]]
}
