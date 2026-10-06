#!/usr/bin/env bash
# Installed pipelines survive the container being recreated (spec 024, R7).
# Weekly in CI (extras-restore-e2e): installs the scene group (torch) for real.
#
#   tests/launcher/test_extras_restore_e2e.sh docker|podman
set -euo pipefail

ENGINE=${1:?usage: test_extras_restore_e2e.sh docker|podman}
IMAGE=${E2E_IMAGE:-localhost/videoannotator:e2e}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
LAUNCHER="$REPO/launcher/videoannotator-start"
ROOT=$(mktemp -d)
export XDG_CONFIG_HOME="$ROOT/config"
CONF="$XDG_CONFIG_HOME/videoannotator/start.conf"
URL=http://127.0.0.1:18011

step() { printf '\n=== %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; "$LAUNCHER" logs --engine "$ENGINE" 2>&1 | tail -n 60 >&2 || true; exit 1; }
cleanup() { "$LAUNCHER" stop --engine "$ENGINE" >/dev/null 2>&1 || true; }
trap cleanup EXIT

start() { "$LAUNCHER" --engine "$ENGINE" --image "$IMAGE" --no-browser --yes "$@"; }
key() { sed -n 's/^key=//p' "$CONF"; }
api() { curl -fsS -H "Authorization: Bearer $(key)" "$URL$1" "${@:2}"; }
py() { python3 -c "import json, sys; d = json.load(sys.stdin); $1" "${@:2}"; }
scene_state() {
    api /api/v1/pipelines | py 'print(next(p["readiness"]["state"] + " " + str(p["readiness"].get("install_job_id")) for p in d["pipelines"] if p["name"] == "scene_detection"))'
}

step "A short real video, made with the image's own ffmpeg"
mkdir -p "$ROOT/videos" "$ROOT/second"
"$ENGINE" run --rm --entrypoint ffmpeg -v "$ROOT/videos:/out" "$IMAGE" -loglevel error \
    -f lavfi -i testsrc=duration=3:size=160x120:rate=10 -pix_fmt yuv420p /out/clip.mp4

step "Start and install the scene group"
start --share "$ROOT/videos" --results "$ROOT/results"
job=$(api /api/v1/pipelines/extras/scene/install -X POST | py 'print(d["job_id"])')
for _ in $(seq 1 360); do
    s=$(api "/api/v1/pipelines/extras/install-jobs/$job" | py 'print(d["status"])')
    [ "$s" = completed ] && break
    [ "$s" = failed ] && fail "install failed"
    sleep 5
done
[ "$(scene_state | cut -d' ' -f1)" = ready ] || fail "scene not ready after install"

step "Share another folder: the container is recreated, scene is restored"
"$LAUNCHER" share "$ROOT/second" --engine "$ENGINE" --image "$IMAGE" --no-browser --yes
SHARE=$(api /api/v1/ingest/access | py 'print([f["path"] for f in d["allowed_folders"] if f["display_path"] == sys.argv[1]][0])' "$ROOT/videos")
queued=$(api /api/v1/ingest -X POST -H 'Content-Type: application/json' \
    -d "{\"path\": \"$SHARE\", \"selected_pipelines\": [\"scene_detection\"]}" | py 'print(d["created"][0])')
saw_restoring="" restore_job=""
for _ in $(seq 1 360); do
    read -r state install_job <<< "$(scene_state)"
    if [ "$state" = restoring ]; then saw_restoring=1; restore_job=$install_job; fi
    [ "$state" = ready ] && break
    sleep 5
done
[ -n "$saw_restoring" ] || fail "never reported restoring"
[ "$state" = ready ] || fail "scene not restored"
if [ -n "$restore_job" ] && [ "$restore_job" != None ]; then
    output=$(api "/api/v1/pipelines/extras/install-jobs/$restore_job" | py 'print(d.get("command_output") or "")')
    printf '%s\n' "$output" | tail -n 5
    if printf '%s\n' "$output" | grep -q "Downloading torch"; then
        fail "the restore downloaded torch again instead of using the cache"
    fi
fi

step "The job queued during the restore waited, then ran"
for _ in $(seq 1 120); do
    s=$(api "/api/v1/jobs/$queued" | py 'print(d["status"])')
    case "$s" in completed) break ;; failed) fail "queued scene job failed" ;; esac
    sleep 5
done
[ "$s" = completed ] || fail "queued scene job never completed"

step "Restore passed with $ENGINE"
