#!/usr/bin/env bats
# Where a host folder is mounted (research R3).

load helpers

setup() { setup_launcher; }

check_path() {
    expect_equal "$(va_container_path "$(field "$1" .input.path)" "$(field "$1" .input.n)")" \
        "$(field "$1" .expect.container)"
}

check_normalise() {
    HOME=$(field "$1" .input.home)
    expect_equal "$(va_normalise "$(field "$1" .input.path)")" "$(field "$1" .expect.path)"
}

@test "paths: cases.json" { run_cases paths check_path; }

@test "normalise: cases.json" { run_cases normalise check_normalise; }

@test "a relative path is made absolute" {
    cd "$BATS_TEST_TMPDIR"
    [ "$(va_normalise Studies)" = "$BATS_TEST_TMPDIR/Studies" ]
}
