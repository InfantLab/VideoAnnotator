# Handover: finishing v1.6.0 (updated 2026-10-18)

For the next session. Branch `1.6-dev`, all work committed locally, nothing pushed. v1.6.0 is the
**quiet consolidation release**; v1.7.0 is the public one
([`roadmap_v1.6.0.md`](roadmap_v1.6.0.md), [`roadmap_v1.7.0.md`](roadmap_v1.7.0.md)).

## Done 2026-10-18

- The size-analysis test failure (a mocked YOLO with no `ckpt_path`), fixed. Roadmaps updated.
- **Spec 030 (model defaults, Phase 5c)**: planned, tasked and built, one change per commit, each
  checked on the demo clip against the previous step on the same machine (a GTX 1060 6GB in this
  container). Record: [`specs/030-model-defaults/checks.md`](../../specs/030-model-defaults/checks.md).
  Faces: age/gender off and not run (the `_deepface_actions()` bug); speaker turns: community-1;
  people: yolo26n-pose on ultralytics 8.4; speech: large-v3-turbo on faster-whisper (openai-whisper
  removed). Settings can carry `reason`/`reason_source` ("Why this default"). Fixtures
  re-baselined; CHANGELOG written.

## Next, in order

1. **A run with a Hugging Face token** (accepted licence: `pyannote/speaker-diarization-community-1`).
   This machine has none, so diarization never ran here. Save it with
   `videoannotator settings huggingface-token` (the baseline test passes a saved token to its
   server), then:
   - `VIDEOANNOTATOR_BASELINE_KEEP=~/va-030/step3b uv run --no-sync pytest tests/integration/test_output_baseline.py -m real_models -q`;
     compare with `scripts/compare_demo_outputs.py ~/va-030/step5b ~/va-030/step3b` (only the
     `.rttm` should differ); record turns and speaker count in checks.md step 3 and the
     CHANGELOG's speaker-turns line (it says "not yet run").
   - Copy the new `.rttm` and its `.provenance.json` over the fixtures, rerun the baseline test
     (expect 8 of 8) and `cd viewer && bun run test:run` (the contract test compares the `.rttm`
     with the legacy one exactly; it will need the same shape-only change as speech).
   - One job naming `pyannote/speaker-diarization-3.1` (spec 030 T039).
2. **Re-run the R notebook** (`tests/manual/results_out_r_notebook.md`): its table was
   regenerated from the new fixtures.
3. **Release** (ask before each outward step): optional `v1.6.0rc2`; PR `1.6-dev` → `master`
   (spec 024's CI end-to-end runs start there; CI's install matrix confirms the `audio` extra on
   macOS/Windows); tag `v1.6.0`; GitHub release whose notes say the viewer is being redesigned for
   v1.7.0 and that v1.6.0rc1 was the last with openai-whisper; Zenodo DOI; `CITATION.cff`.

## Waiting on the maintainer

- Look in a browser at: the pipeline guide (Pipelines page; New job's settings step) and the
  Hugging Face token flow with a real token (Speaker turns card; Settings → API Configuration).
- Delete the old `logs/` files and make a new API key: before `c2edfbd` every request's
  `Authorization` header (full keys) went into `logs/api_server.log`.
- Decide: should a server with authentication off (`scripts/dev.sh`, `--dev`) treat callers as
  admin? Today admin endpoints refuse every keyless request, so admin actions fail under `dev.sh`.
- API responsiveness during a long, heavy job (roadmap Phase 2): check during spec 030's GPU runs.

## Done 2026-10-16/17 (for the roadmap and release notes)

- Spec 023 (viewer overhaul) designed: research, decisions 1–8, journeys, split into 026–029.
  027–029 are v1.7.0.
- Spec 026, pipeline guide: built (`c018523`, `27510c4`, `2da16c6`).
- Spec 031, Hugging Face token from the viewer/Settings/CLI, no restart (`0ffd059`, `ab400a7`,
  `fdc5346`, `9e8b530`).
- Docs: version on every page; `videoannotator --version` (`79582bd`).
- Viewer: timeline colours and labels that don't mislead (`41a38d2`); DeepWiki links gone
  (`4842843`).
- Server: terminal shows only what matters; API keys no longer logged (`c2edfbd`).

## Environment notes

- The container has no git identity: commit with
  `git -c "user.name=Caspar Addyman" -c user.email=infantologist@gmail.com commit …`.
- Test state 2026-10-18: full suite green (1881 passed); the real-model baseline passes 7 of 8 (the
  `.rttm`, item 1). `bats` isn't installed here, so `pre-commit run --all-files` reports the
  launcher hook failing; it also reformats about 20 old Markdown files (pre-existing drift).
- `mkdocs` needs `uv sync --inexact --group docs`; build with `uv run --no-sync mkdocs build --strict`.
