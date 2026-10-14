#!/usr/bin/env bats
# The run command (contracts/launcher.md, "The run it builds").

load helpers

setup() { setup_launcher; }

check_run() {
    VA_OS=$(field "$1" .os)
    VA_ENGINE=$(field "$1" .input.engine)
    HOME=$(field "$1" .input.home)
    unset XDG_CONFIG_HOME
    VA_PRESENT=$(printf '%s' "$1" | jq -r '.input.shares | join("\n")')
    VA_MISSING=$(printf '%s' "$1" | jq -r '.input.missing | join("\n")')
    VA_S_RESULTS=$(field "$1" .input.results)
    VA_PORT=$(field "$1" .input.port)
    VA_IMAGE=$(field "$1" .input.image)
    VA_UID=$(field "$1" .input.uid) VA_GID=$(field "$1" .input.gid)
    VA_GPU_FLAGS=$(field "$1" .input.gpu)
    VA_DOCKER_DESKTOP=$(field "$1" .input.desktop)
    expect_equal "$(va_build_run)" "$(printf '%s' "$1" | jq -r '.expect.args | join("\n")')"
}

@test "run: cases.json" { run_cases run check_run; }

@test "shares use --mount, never -v, and no restart policy or user namespace" {
    VA_ENGINE=podman VA_PRESENT="$HOME/Studies" VA_MISSING="" VA_S_RESULTS="$HOME/VA"
    VA_PORT=18011 VA_IMAGE=img VA_GPU_FLAGS=""
    args=$(va_build_run)
    ! printf '%s\n' "$args" | grep -q -- '--restart'
    ! printf '%s\n' "$args" | grep -q -- '--userns'
    [ "$(printf '%s\n' "$args" | grep -c -- '^-v$')" -eq 4 ]
}

@test "the run reaches the engine as separate arguments" {
    mkdir -p "$HOME/My Studies"
    VA_ENGINE=podman VA_PRESENT="$HOME/My Studies" VA_MISSING="" VA_S_RESULTS="$HOME/VA"
    VA_PORT=18011 VA_IMAGE=img VA_GPU_FLAGS=""
    va_engine() { printf '<%s>\n' "$@" > "$ENGINE_LOG"; }
    VA_GPU_FLAGS="--gpus all"
    va_run_engine_with_args
    target=$(va_container_path "$HOME/My Studies" 1)
    grep -qx "<type=bind,source=$HOME/My Studies,target=$target,readonly>" "$ENGINE_LOG"
    grep -qx "<videoannotator-models:/app/models>" "$ENGINE_LOG"
    grep -qx "<videoannotator-cache:/app/cache>" "$ENGINE_LOG"
    grep -qx "<--gpus>" "$ENGINE_LOG"
    [ "$(printf '%s\n' "$(va_build_run)" | wc -l)" -eq "$(wc -l < "$ENGINE_LOG")" ]
    [ "$(head -n 1 "$ENGINE_LOG")" = "<run>" ]
    [ "$(tail -n 1 "$ENGINE_LOG")" = "<img>" ]
}

@test "a missing share is named and left out" {
    mkdir -p "$HOME/Studies"
    VA_SHARES="$HOME/Studies
/media/ada/Unplugged"
    run va_split_shares
    [ "$output" = "Couldn't find /media/ada/Unplugged (an unplugged drive?), so it isn't shared this time." ]
    va_split_shares > /dev/null
    [ "$VA_PRESENT" = "$HOME/Studies" ]
    [ "$VA_MISSING" = /media/ada/Unplugged ]
}
