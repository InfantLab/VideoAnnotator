#!/usr/bin/env bats
# The launcher can be sourced for testing without running.

load helpers

setup() { setup_launcher; }

@test "sourcing defines the functions and runs nothing" {
    type va_build_run >/dev/null
    type main >/dev/null
    [ ! -s "$ENGINE_LOG" ]
}

@test "the script is plain sh" {
    run sh -n "$LAUNCHER"
    [ "$status" -eq 0 ]
}

@test "help" {
    run sh "$LAUNCHER" --help
    [ "$status" -eq 0 ]
    [[ "$output" == *"share [PATH]"* ]]
}
