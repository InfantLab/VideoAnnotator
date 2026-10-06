#!/usr/bin/env bats
# start.conf: where it lives and what it holds (data-model.md, research R13).

load helpers

setup() { setup_launcher; }

check_settings_path() {
    VA_OS=$(field "$1" .os)
    HOME=$(field "$1" .input.home)
    XDG_CONFIG_HOME=$(field "$1" .input.xdg)
    expect_equal "$(va_settings_path)" "$(field "$1" .expect.path)"
}

@test "settings: cases.json" { run_cases settings check_settings_path; }

@test "settings round-trip, keeping order and spaces" {
    VA_S_ENGINE=podman VA_S_IMAGE=ghcr.io/infantlab/videoannotator:1.6.0
    VA_S_RESULTS="$HOME/My Results" VA_S_KEY=va_secret VA_S_PORT=18012
    VA_SHARES="$HOME/Studies/Day 1
/media/ada/Drive 2
$HOME/Études"
    va_settings_write
    expected_shares=$VA_SHARES
    VA_S_ENGINE="" VA_S_IMAGE="" VA_S_RESULTS="" VA_S_KEY="" VA_S_PORT="" VA_SHARES=""
    va_settings_read
    [ "$VA_S_ENGINE" = podman ]
    [ "$VA_S_IMAGE" = ghcr.io/infantlab/videoannotator:1.6.0 ]
    [ "$VA_S_RESULTS" = "$HOME/My Results" ]
    [ "$VA_S_KEY" = va_secret ]
    [ "$VA_S_PORT" = 18012 ]
    [ "$VA_SHARES" = "$expected_shares" ]
    [ "$(grep -c '^share=' "$(va_settings_path)")" -eq 3 ]
}

@test "the settings file is private, with a requests folder beside it" {
    VA_S_RESULTS="$HOME/VideoAnnotator" VA_SHARES=""
    va_settings_write
    file=$(va_settings_path)
    mode=$(stat -c %a "$file" 2>/dev/null || stat -f %Lp "$file")
    [ "$mode" = 600 ]
    [ -d "$(dirname "$file")/requests" ]
}

@test "no settings file reads as nothing set" {
    va_settings_read
    [ -z "$VA_SHARES" ] && [ -z "$VA_S_RESULTS" ]
}

@test "lines that aren't settings are ignored" {
    mkdir -p "$(dirname "$(va_settings_path)")"
    printf 'just a line\nshare=/a\nunknown=1\nshare=/b\n' > "$(va_settings_path)"
    va_settings_read
    [ "$VA_SHARES" = "/a
/b" ]
}
