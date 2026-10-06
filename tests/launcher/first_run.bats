#!/usr/bin/env bats
# The first start: which folder, confirm, share, start (Story 1).

load helpers

setup() {
    flow_setup
    mkdir -p "$HOME/Studies"
}

@test "scripted first start: --share --results --yes" {
    run main --share "$HOME/Studies" --results "$HOME/Out" --yes --no-browser
    [ "$status" -eq 0 ]
    [[ "$output" == "Starting VideoAnnotator with Podman."* ]]
    [ -d "$HOME/Out" ]
    grep -qx "share=$HOME/Studies" "$(va_settings_path)"
    grep -qx "results=$HOME/Out" "$(va_settings_path)"
    grep -qx "engine=podman" "$(va_settings_path)"
    grep -qx "key=va_new_key" "$(va_settings_path)"
    [ "$(engine_runs)" -eq 1 ]
}

@test "first start asks for the folder, confirms in plain words, results in ~/VideoAnnotator" {
    va_pick_folder() { printf '%s\n' "$HOME/Studies"; }
    run main <<< "y"
    [ "$status" -eq 0 ]
    [[ "$output" == *"Which folder are your videos in?"* ]]
    [[ "$output" == *"VideoAnnotator will be able to read, but never change:"* ]]
    [[ "$output" == *"  $HOME/Studies   (and everything inside it)"* ]]
    [[ "$output" == *"Results go to:"*"  $HOME/VideoAnnotator"* ]]
    [[ "$output" == *"Share this folder? [Y/n]"* ]]
    [ -d "$HOME/VideoAnnotator" ]
}

@test "a broad choice, declined, goes back to the picker" {
    picks="$BATS_TEST_TMPDIR/picks"
    printf '%s\n' "$HOME" "$HOME/Studies" > "$picks"
    va_pick_folder() {
        first=$(head -n 1 "$picks"); sed -i.bak 1d "$picks"; printf '%s\n' "$first"
    }
    run main <<< $'n\ny'
    [ "$status" -eq 0 ]
    [[ "$output" == *"This shares everything in your home folder"* ]]
    grep -qx "share=$HOME/Studies" "$(va_settings_path)"
    ! grep -qx "share=$HOME" "$(va_settings_path)"
}

@test "cancelling the picker shares nothing and starts nothing" {
    va_pick_folder() { return 2; }
    run main < /dev/null
    [ "$status" -eq 2 ]
    [ "$(engine_runs)" -eq 0 ]
}

@test "the text prompt lists folders with videos" {
    mkdir -p "$HOME/Videos/Study A"
    : > "$HOME/Videos/Study A/child01.mp4"
    run va_pick_folder "Which folder are your videos in?" <<< "$HOME/Videos/Study A/"
    [ "$status" -eq 0 ]
    [[ "$output" == *"  $HOME/Videos/Study A"* ]]
    [[ "$output" == *"Type the folder path (or press Enter to cancel): $HOME/Videos/Study A" ]]
}

@test "the text prompt, left empty, cancels" {
    run va_pick_folder "Which folder are your videos in?" <<< ""
    [ "$status" -eq 2 ]
}

@test "logs" {
    run main logs
    grep -q "^logs --tail 200 videoannotator" "$ENGINE_LOG"
}

@test "list shows folders, results, engine and image" {
    saved engine=podman "image=ghcr.io/infantlab/videoannotator:1.6.0" \
        "results=$HOME/VideoAnnotator" "share=$HOME/Studies"
    run main list
    [ "$status" -eq 0 ]
    [[ "$output" == *"  $HOME/Studies"*"  $HOME/VideoAnnotator"*"Engine: Podman"*"Image:  ghcr.io/infantlab/videoannotator:1.6.0"* ]]
}
