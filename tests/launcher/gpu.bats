#!/usr/bin/env bats
# The GPU when the engine can use it, CPU with a note otherwise (research R15).

load helpers

setup() { setup_launcher; }

check_gpu() {
    VA_ENGINE=$(field "$1" .input.engine)
    NVIDIA=$(field "$1" .input.nvidia)
    RUNTIMES=$(field "$1" .input.runtimes)
    CDI=$(field "$1" .input.cdi)
    va_has_nvidia() { [ "$NVIDIA" = true ]; }
    va_docker_runtimes() { printf '%s' "$RUNTIMES"; }
    va_cdi_devices() { printf '%s\n' "$CDI"; }
    note=$(va_gpu_flags)
    va_gpu_flags > /dev/null
    expect_equal "$VA_GPU_FLAGS" "$(field "$1" .expect.flags)" || return 1
    if [ "$(field "$1" .expect.note)" = true ]; then
        [[ "$note" == "Running without the GPU: "*" can't use it yet. To enable it, see "* ]] || { echo "note: $note"; return 1; }
    else
        expect_equal "$note" ""
    fi
}

@test "gpu: cases.json" { run_cases gpu check_gpu; }

@test "a start that fails with the GPU is retried once without it, and says so" {
    VA_ENGINE=docker VA_GPU_FLAGS="--gpus all"
    VA_PRESENT="" VA_MISSING="" VA_S_RESULTS="$HOME/VA" VA_PORT=18011 VA_IMAGE=img
    engine_stub() {
        if [ "$1" = run ]; then
            case "$*" in *--gpus*) echo "could not select device driver \"\" with capabilities: [[gpu]]" >&2; return 125 ;; esac
        fi
        [ "$1 $2" = "container inspect" ] && return 1
        return 0
    }
    run va_start
    [ "$status" -eq 0 ]
    [[ "$output" == "Couldn't use the GPU, so VideoAnnotator is running without it."* ]]
    [ "$(grep -c '^run ' "$ENGINE_LOG")" -eq 2 ]
}
