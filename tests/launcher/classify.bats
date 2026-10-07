#!/usr/bin/env bats
# What may be shared, and what needs confirming (research R4).

load helpers

setup() { setup_launcher; }

check_classify() {
    HOME=$(field "$1" .input.home)
    VA_S_RESULTS=$(field "$1" .input.results)
    VA_SHARES=$(printf '%s' "$1" | jq -r '.input.shares | join("\n")')
    expect_equal "$(va_classify "$(field "$1" .input.path)")" "$(field "$1" .expect.result)"
}

@test "classify: cases.json" { run_cases classify check_classify; }

@test "a broad share names what becomes readable, and defaults to no" {
    mkdir -p "$HOME/Studies"
    VA_S_RESULTS="$HOME/VideoAnnotator" VA_SHARES=""
    run va_share_add "$HOME" < /dev/null
    [ "$status" -eq 2 ]
    [[ "$output" == *"This shares everything in your home folder, including documents unrelated to your research."* ]]
    [[ "$output" == *"Share it anyway? [y/N]"* ]]
}

@test "a broad share can be confirmed" {
    VA_S_RESULTS="/elsewhere" VA_SHARES=""
    va_share_add "$HOME" <<< "y" > /dev/null
    [ "$VA_SHARES" = "$HOME" ]
}

@test "a folder inside the results folder is refused, saying why" {
    mkdir -p "$HOME/VideoAnnotator/Run 1"
    VA_S_RESULTS="$HOME/VideoAnnotator" VA_SHARES=""
    run va_share_add "$HOME/VideoAnnotator/Run 1"
    [ "$status" -eq 1 ]
    [ "$output" = "That folder is inside your results folder, which VideoAnnotator can already read." ]
}

@test "a containing folder replaces the shares inside it" {
    mkdir -p "$HOME/Lab/A" "$HOME/Lab/B" "$HOME/Other"
    VA_S_RESULTS="$HOME/VideoAnnotator"
    VA_SHARES="$HOME/Lab/A
$HOME/Other
$HOME/Lab/B"
    va_share_add "$HOME/Lab"
    [ "$VA_SHARES" = "$HOME/Other
$HOME/Lab" ]
}

@test "a duplicate is not added twice" {
    mkdir -p "$HOME/Studies/Day 1"
    VA_S_RESULTS="$HOME/VideoAnnotator" VA_SHARES="$HOME/Studies"
    run va_share_add "$HOME/Studies/Day 1"
    [ "$status" -eq 1 ]
    [ "$VA_SHARES" = "$HOME/Studies" ]
}
