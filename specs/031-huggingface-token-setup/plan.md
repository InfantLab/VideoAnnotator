# Implementation Plan: Hugging Face Token Setup

**Branch**: `1.6-dev` (spec `031-huggingface-token-setup`) | **Date**: 2026-10-17 | **Spec**: [spec.md](spec.md)

## Summary

A token saved by an admin (viewer card, Settings, or CLI) goes into a 0600 file next to the
database; `config_env.huggingface_token()` reads it after the environment variables, so pipelines
and readiness use it at the next job and check, with no restart. Three endpoints under
`/api/v1/system/huggingface-token` (GET any signed-in user; PUT and DELETE admin) reuse readiness's
existing Hugging Face check (`_check_hf_access`). The viewer turns the token blocker into a
three-step panel; Settings gets a status row; the CLI gets `settings huggingface-token`.

## Technical Context

**Language/Version**: Python 3.12/3.13; TypeScript/React (viewer)
**Primary Dependencies**: existing only (FastAPI, Typer; huggingface.co REST calls readiness already makes)
**Storage**: one file, `<database dir>/huggingface_token`, mode 0600
**Testing**: pytest (store, resolution, API with admin overrides, CLI, no-leak check); Vitest (panel, Settings row)
**Constraints**: token never returned or logged (FR-011); environment wins (SC-003)

## Constitution Check

| Principle | Check | Status |
|---|---|---|
| I. Local-First, No Telemetry | Token stored on the machine; the only network call is to Hugging Face, which the pipeline needs anyway and readiness already makes. | Pass |
| II. Stable Contract | New endpoints only; no output change. | Pass |
| III. Provenance | The token is never recorded (FR-011); provenance unchanged. | Pass |
| IV. Modular | Core module (`secrets_store.py`) with no pipeline imports; pipelines keep calling `huggingface_token()`. | Pass |
| V. Backward Compatibility | Environment variables keep priority and meaning; nothing removed. | Pass |
| VI. Faithful Display | Licence ✓/✗ come from the server's checks, never assumed. | Pass |

## Structure

```
src/videoannotator/
├── secrets_store.py            # new: path, read, save (0600, atomic), remove
├── config_env.py               # huggingface_token(): env names, then saved; huggingface_token_source()
├── api/readiness.py            # secret check via huggingface_token(); friendlier messages; check for one token
├── api/v1/system.py            # GET/PUT/DELETE /system/huggingface-token
└── cli.py                      # `settings huggingface-token [--remove|--status]`
viewer/src/
├── api/client.ts               # getHuggingFaceToken / saveHuggingFaceToken / removeHuggingFaceToken
├── components/HuggingFaceTokenSetup.tsx   # the three-step panel
├── components/LockedPipelineCard.tsx      # ReadinessDetails shows the panel for the token blocker
└── pages/Settings.tsx          # status row
tests/
├── unit/test_secrets_store.py
├── api/test_huggingface_token_api.py
└── unit/cli/test_cli_settings_token.py
```

## Complexity Tracking

None.
