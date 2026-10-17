---
description: "Tasks for spec 031: Hugging Face Token Setup"
---

# Tasks: Hugging Face Token Setup

**Tests**: included (new API and CLI surface; SC-002 is an automated check).

## Phase 1: Foundational

- [X] T001 `src/videoannotator/secrets_store.py`: `huggingface_token_path()` (database dir / `huggingface_token`), `read_saved_huggingface_token()`, `save_huggingface_token(token)` (trim, atomic write, chmod 0600), `remove_saved_huggingface_token()`
- [X] T002 `src/videoannotator/config_env.py`: `huggingface_token()` falls back to the saved token after the environment variables; add `huggingface_token_source()` → (`"environment"`, var) | (`"saved"`, None) | (None, None)
- [X] T003 [P] `tests/unit/test_secrets_store.py`: save/read/remove, 0600 mode, trimming, empty/unreadable file → None, environment wins

## Phase 2: User Story 1 - Add the token from the pipeline's card (P1)

- [X] T004 [US1] `src/videoannotator/api/readiness.py`: secret check for `HUGGINGFACE_TOKEN` uses `huggingface_token()`; messages for a missing/rejected token point to the card and Settings, then the environment variable; add `check_huggingface_token(token)` (whoami via `_check_hf_access`) returning `(check, account)`
- [X] T005 [US1] `src/videoannotator/api/v1/system.py`: GET/PUT/DELETE `/huggingface-token` per contracts/api.md (GET authenticated; PUT/DELETE `require_admin`)
- [X] T006 [P] [US1] `tests/api/test_huggingface_token_api.py`: GET shapes for none/env/saved; PUT ok/rejected/unreachable/empty; DELETE; non-admin 403; the token string appears in no response body and no captured log (SC-002); readiness flips after PUT without restart
- [X] T007 [US1] `viewer/src/api/client.ts` + `viewer/src/types/pipelines.ts`: `HuggingFaceTokenStatus`, `getHuggingFaceToken`, `saveHuggingFaceToken`, `removeHuggingFaceToken`
- [X] T008 [US1] `viewer/src/components/HuggingFaceTokenSetup.tsx`: the three steps; licence items with ✓/✗/unchecked from readiness; password field + "Save and check" for admins, "ask whoever runs this VideoAnnotator" otherwise; results (account, rejected, couldn't check); calls `onSaved` (re-check readiness)
- [X] T009 [US1] `viewer/src/components/LockedPipelineCard.tsx` (`ReadinessDetails`) and `viewer/src/pages/NewJob.tsx`: show the panel in place of the token blocker line, passing licence items, admin status and `onCheckAgain`
- [X] T010 [P] [US1] `viewer/src/test/components/HuggingFaceTokenSetup.test.tsx`: admin/non-admin, save ok/rejected, licence states, no token value rendered after save

## Phase 3: User Story 2 - Settings (P2)

- [X] T011 [US2] `viewer/src/pages/Settings.tsx`: "Hugging Face token" row: status, source, account; admin Change (inline field) and Remove; environment note
- [X] T012 [P] [US2] Settings row test in `viewer/src/test/components/HuggingFaceTokenSetup.test.tsx` (or a Settings test file)

## Phase 4: User Story 3 - CLI (P3)

- [X] T013 [US3] `src/videoannotator/cli.py`: `settings` group with `huggingface-token [--remove] [--status]`; prompt with `hide_input`; check via `check_huggingface_token`; save/remove via `secrets_store`
- [X] T014 [P] [US3] `tests/unit/cli/test_cli_settings_token.py`: save (check mocked), rejected, status, remove; token not echoed

## Phase 5: Polish

- [X] T015 Docs: `docs/installation/INSTALLATION.md` (or the speaker-diarization usage page) describes the card/Settings/CLI routes first and the environment variable second; CHANGELOG
- [X] T016 Rebuild viewer bundle; full gates (ruff, mypy, pytest, viewer lint/types/tests, bundle check)
