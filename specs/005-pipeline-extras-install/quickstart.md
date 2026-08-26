# Quickstart: Verifying Extras Discoverability & Self-Service Install

Manual/CI verification steps mapped to spec.md's acceptance scenarios. Assumes a running server
(`./scripts/start_server.sh` or `uv run videoannotator server --dev`) with a core-only install (no
extras) and an admin API key from `setup-db`/`generate-token`.

Set once:

```bash
export VA_URL=http://localhost:18011
export VA_ADMIN_KEY=va_...   # from `uv run videoannotator generate-token`
```

## §1 — Confirm the read side still works (spec 004, unchanged)

```bash
curl -s "$VA_URL/api/v1/pipelines?include_unavailable=true" -H "Authorization: Bearer $VA_ADMIN_KEY" | jq '.pipelines[] | select(.name=="face_analysis")'
```

Expect `available: false` and an `install_hint` naming `face`. Also confirm the new top-level field
is present and false on a clean start:

```bash
curl -s "$VA_URL/api/v1/pipelines" -H "Authorization: Bearer $VA_ADMIN_KEY" | jq '.restart_required'
# => false
```

## §2 — User Story 3: installs are safe by construction

```bash
# (a) unauthenticated
curl -s -o /dev/null -w '%{http_code}\n' -X POST "$VA_URL/api/v1/pipelines/extras/face/install"
# => 401

# (b) invalid extras name, even with valid admin auth
curl -s -X POST "$VA_URL/api/v1/pipelines/extras/gpu-magic/install" -H "Authorization: Bearer $VA_ADMIN_KEY"
# => 422, body names the unknown group, no subprocess is spawned (nothing to observe externally —
#    covered by a unit test asserting the install function is never called, not by this manual step)
```

## §3 — User Story 1: trigger and track a real install

```bash
JOB=$(curl -s -X POST "$VA_URL/api/v1/pipelines/extras/scene/install" -H "Authorization: Bearer $VA_ADMIN_KEY")
echo "$JOB" | jq .
JOB_ID=$(echo "$JOB" | jq -r .job_id)

# poll until status is completed or failed
watch -n2 "curl -s $VA_URL/api/v1/pipelines/extras/install-jobs/$JOB_ID -H 'Authorization: Bearer $VA_ADMIN_KEY' | jq ."
```

`scene` pulls in torch (CUDA build) like most extras groups except `face` — per pyproject.toml's
own comment, `face` (the DeepFace variant) is the *only* face/audio/scene/person group with no
torch dependency. Budget several minutes either way: a real run of `scene` in this repo's dev
container measured ~8 minutes end-to-end (torch's CUDA wheels alone are 700MB+), confirmed via a
real (non-mocked) run of this exact flow during implementation — this is expected, not a hung
job. The trigger/poll/restart mechanics themselves are fast (sub-second); only the underlying
`pip`/`uv` download is slow.

Expect: `status` moves `pending → running → completed`, `command_output` is non-empty on completion.

## §4 — User Story 2: restart-required signal

```bash
# immediately after §3 completes, before restarting the server:
curl -s "$VA_URL/api/v1/pipelines" -H "Authorization: Bearer $VA_ADMIN_KEY" | jq '.restart_required'
# => true

curl -s "$VA_URL/api/v1/pipelines/extras/install-jobs/$JOB_ID" -H "Authorization: Bearer $VA_ADMIN_KEY" | jq '.restart_required'
# => true

curl -s "$VA_URL/api/v1/pipelines?include_unavailable=true" -H "Authorization: Bearer $VA_ADMIN_KEY" | jq '.pipelines[] | select(.name=="scene_detection")'
# => still available: false — the *running* process hasn't picked up the new import yet, that's the point
```

Now restart the server (Ctrl+C, re-run `start_server.sh`), then repeat:

```bash
curl -s "$VA_URL/api/v1/pipelines" -H "Authorization: Bearer $VA_ADMIN_KEY" | jq '.restart_required'
# => false

curl -s "$VA_URL/api/v1/pipelines" -H "Authorization: Bearer $VA_ADMIN_KEY" | jq '.pipelines[] | select(.name=="scene_detection") | .available'
# => true
```

## §5 — Edge cases worth a manual pass

- Re-run §3 for the same extras group (`scene`) once it's already installed — expect a fast
  `completed` job whose output notes nothing needed to change, not an error (FR-011).
- Fire two install requests for the same group back-to-back — expect the same `job_id` back both
  times while the first is still in flight (FR-010).
- Kill the server process mid-install (`kill -9` the PID during `running`), restart it, then check
  that job's status — expect `failed`, not stuck `running` forever (crash edge case).
