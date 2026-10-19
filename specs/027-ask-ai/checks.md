# Checks: Ask AI

**Date**: 2026-10-19 | **Code**: `1.7-dev` at the commits for spec 027 | **Tasks**: T067, T068

## Automated (T067)

- `pytest tests/` (excluding `test_output_baseline.py`): 2023 passed, 30 skipped, on a clean
  checkout of the committed code. `test_output_baseline.py` (real models) did not finish within
  10 minutes on this machine (GTX 1060) and was stopped: not run.
- `ruff check .`, `mypy src/videoannotator`: clean. `pre-commit` ran on every commit's files;
  not run with `--all-files`, because the working tree held someone else's uncommitted edits
  that its fixers would have rewritten.
- Viewer: `bun run lint`, `tsc --noEmit -p tsconfig.app.json`, `vitest run`: clean, 818 passed.
- Bundle: initial chunk 157 KB gzipped (budget 300 KB); Ask AI is its own chunk (14 KB gzipped).

## Walk-through (T068): partly done

The machine's Ollama had only text models (`qwen2.5:7b`), and its GPU runner crashed on every
call (`CUDA error: the provided PTX was compiled with an unsupported toolchain`). So no quickstart
step that needs a real answer could be checked, nor SC-001 (first answer in under a minute),
SC-004 (lag) or SC-005 (reading the badge). Still to do on a machine with a vision model:
quickstart §1–§6 in the viewer.

What was checked, against a live server (isolated database and storage), the real Ollama and the
demo clip (`…Peekaboo_h265.mp4`):

| Check | Result |
|---|---|
| `GET /vlm/models`, `GET /ask/endpoint` | models listed; `host.docker.internal` classed `local`, confirmed |
| Stream token, range request | token returned with `duration_sec: 10.516`; `Range: bytes=0-999` → 206, 1000 bytes |
| `GET /ask/frames`, 3 frames 0.5 s apart at 5 s | frames 64, 71, 78 at 4.487, 4.978, 5.468 s |
| Try at 5 s | entry returned `running` at once; settled `done` with one answer, `status: error`, carrying Ollama's CUDA error, frame 71: an error, not an answer (Principle VI) |
| Run every 1 s, Stop after 2 answers | answers came in while running (2 of 11); after Stop the call in progress finished, then the file had 3 annotations with `stopped_early: true`; job `cancelled`, `vlm_annotation` `completed`, provenance `stopped_early: true`; `/jobs/{id}/export/table` → 200 |

Found and fixed: after Stop, `GET /ask/runs/{id}/answers` said `stopped` as soon as the job was
cancelled, while the sample in progress was still being answered, so the window would have
stopped polling and missed the last answer. It now says `running` until the results file is
finished (test in `tests/api/test_ask_api.py`).

## Deviations from the plan

- Range and moments keys are `technical`, not `behavioural`, in the registry: spec 026 SC-002
  allows at most five behavioural settings (contracts/pipeline-config.md updated).
- The methods paragraph says "running on this computer / on the local network / on a remote
  server" without the host: spec 025 FR-029 keeps server addresses out of it. The host is in the
  provenance record.
- Added `POST /ask/drafts` (US2 scenario 2 needed it) and `name`, `duration_sec` on the stream
  token response (the timeline needs the duration when the browser can't play the codec);
  contracts/api.md updated.
- The try worker and runs' answers use their own SQLite connection: the app's engine shares one
  connection between threads (StaticPool), and two threads on it at once crashed the interpreter.
