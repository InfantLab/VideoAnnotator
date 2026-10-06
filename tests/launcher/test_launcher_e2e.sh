#!/usr/bin/env bash
# The launcher end to end, with a real engine (CI: the launcher-e2e job).
#
#   tests/launcher/test_launcher_e2e.sh docker|podman
#
# Uses E2E_IMAGE (default localhost/videoannotator:e2e, built from
# tests/launcher/Dockerfile.e2e). Covers Stories 1-4 and 6: first start, results
# owned by the researcher, nothing but shares listed, a restart that asks
# nothing, sharing another folder with jobs queued, Stop sharing from Settings,
# and nothing shared at all.
set -euo pipefail

ENGINE=${1:?usage: test_launcher_e2e.sh docker|podman}
IMAGE=${E2E_IMAGE:-localhost/videoannotator:e2e}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
LAUNCHER="$REPO/launcher/videoannotator-start"
ROOT=$(mktemp -d)  # under /tmp: exercises mounting under /host
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
status_of() { curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $(key)" "$URL$1"; }
py() { python3 -c "import json, sys; d = json.load(sys.stdin); $1"; }

videos() {
    mkdir -p "$1"
    for i in $(seq -w 1 "$2"); do printf 'not really a video %s' "$i" > "$1/child$i.mp4"; done
}

# ingest FOLDER_CONTAINER_PATH SECONDS: prints the batch's job ids.
ingest() {
    api /api/v1/ingest -X POST -H 'Content-Type: application/json' \
        -d "{\"path\": \"$1\", \"recursive\": true, \"selected_pipelines\": [\"launcher_e2e\"], \"config\": {\"launcher_e2e\": {\"seconds\": $2}}}" \
        | py 'print("\n".join(d["created"]))'
}

job_status() { api "/api/v1/jobs/$1" | py 'print(d["status"])'; }

wait_terminal() {
    for _ in $(seq 1 120); do
        s=$(job_status "$1")
        case "$s" in completed | failed | cancelled) echo "$s"; return ;; esac
        sleep 2
    done
    echo timeout
}

access() { api /api/v1/ingest/access; }

step "First start ($ENGINE), sharing one folder"
videos "$ROOT/videos/Study A" 2
start --share "$ROOT/videos" --results "$ROOT/results" | tee "$ROOT/first.log"
grep -q "Starting... ready." "$ROOT/first.log" || fail "not ready"
[ -n "$(key)" ] || fail "no key saved"

step "Only the shared folder is listed, as the host shows it"
access | tee "$ROOT/access.json"
echo
py 'f = d["allowed_folders"]; assert len(f) == 1, f; print(f[0]["path"])' < "$ROOT/access.json" > "$ROOT/share_path"
py "assert d['allowed_folders'][0]['display_path'] == '$ROOT/videos', d" < "$ROOT/access.json" || fail "display path"
py 'assert d["managed_by_launcher"] and d["in_container"], d' < "$ROOT/access.json" || fail "launcher flags"
grep -q '"/root' "$ROOT/access.json" && fail "/root listed"
[ "$(status_of '/api/v1/ingest/browse?path=/root')" = 403 ] || fail "/root browsable"

step "A job runs, and its results belong to the researcher"
SHARE=$(cat "$ROOT/share_path")
for job in $(ingest "$SHARE" 0); do
    [ "$(wait_terminal "$job")" = completed ] || fail "job $job did not complete"
done
find "$ROOT/results" -name '*_launcher_e2e.json' | grep -q . || fail "no results written"
not_mine=$(find "$ROOT/results" ! -uid "$(id -u)")
[ -z "$not_mine" ] || fail "results not owned by $(id -u): $not_mine"

step "A later start asks nothing"
"$LAUNCHER" stop --engine "$ENGINE"
began=$(date +%s)
start < /dev/null | tee "$ROOT/second.log"
took=$(( $(date +%s) - began ))
echo "Second start took ${took}s"
[ "$took" -le 60 ] || echo "::warning::second start took ${took}s (goal: 60s)"
grep -q '?' "$ROOT/second.log" && fail "a later start asked a question"

step "Sharing another folder restarts; queued jobs survive it"
videos "$ROOT/second" 1
queued=$(ingest "$SHARE" 3)
began=$(date +%s)
"$LAUNCHER" share "$ROOT/second" --engine "$ENGINE" --image "$IMAGE" --no-browser --yes
took=$(( $(date +%s) - began ))
echo "Restart for a share took ${took}s"
[ "$took" -le 60 ] || echo "::warning::restart took ${took}s (goal: 60s)"
access | py 'assert len(d["allowed_folders"]) == 2, d' || fail "second share missing"
completed=0
for job in $queued; do
    s=$(wait_terminal "$job")
    [ "$s" = timeout ] && fail "job $job never finished after the restart"
    [ "$s" = completed ] && completed=$((completed + 1))
done
# Two run at a time; the rest were still queued at the restart and must complete.
[ "$completed" -ge 1 ] || fail "no queued job completed after the restart"

step "Stop sharing from Settings takes effect at the next start"
api /api/v1/ingest/shares/stop -X POST -H 'Content-Type: application/json' \
    -d "{\"path\": \"$ROOT/videos\"}" | py 'assert d["stop_requested"], d' || fail "stop request"
access | py 'assert [s for s in d["shares"] if s["stop_requested"]], d' || fail "stop not reported"
"$LAUNCHER" stop --engine "$ENGINE"
start < /dev/null | tee "$ROOT/third.log"
grep -q "Stopped sharing $ROOT/videos" "$ROOT/third.log" || fail "stop request not applied"
access | py "assert [f['display_path'] for f in d['allowed_folders']] == ['$ROOT/second'], d" || fail "still shared"

step "Nothing shared: no folders, and how to share one"
"$LAUNCHER" unshare "$ROOT/second" --engine "$ENGINE" --image "$IMAGE" --no-browser --yes
access | tee "$ROOT/none.json"
echo
py 'assert not d["can_read_in_place"] and d["allowed_folders"] == [] and d["places"] == [], d' < "$ROOT/none.json" || fail "folders listed"
py 'assert "videoannotator-start share" in d["reason"], d' < "$ROOT/none.json" || fail "reason"
grep -q '"/root' "$ROOT/none.json" && fail "/root listed"

step "All passed with $ENGINE"
