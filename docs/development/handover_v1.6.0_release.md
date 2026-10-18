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

1. ~~A run with a Hugging Face token~~: done 2026-10-18. Community-1 gives one speaker where 3.1
   gave two; listening showed 3.1 had split the mother's sentence. Fixtures complete; the
   baseline passes 8 of 8. **Revoke the token pasted into the chat on 2026-10-18** if it isn't
   already (Hugging Face rejected it by then; the saved one is a different token).
2. **Re-run the R notebook** (`tests/manual/results_out_r_notebook.md`): its table was
   regenerated from the new fixtures.
3. ~~Release~~: **v1.6.0 released 2026-10-18** (PR #12 → `31b26d9`, tag `v1.6.0`). Still to do:
   - **PyPI**: add a PyPI API token as the `PYPI_API_TOKEN` repository secret (the project
     name `videoannotator` was free on 2026-10-18), then re-run the release run's
     `build-and-publish` job (run 39155338590). The package built and passed `twine check`;
     only the upload failed. Alternative: register this repo as a PyPI trusted publisher and give
     the job `permissions: id-token: write` (no secret needed). Two docs lines say `pip install videoannotator[...]`.
   - Zenodo minted 10.5281/zenodo.23457429 (in `CITATION.cff`); GHCR has `1.6.0`; the launcher
     is attached to the release.
   - Merge `master` back into `1.6-dev` (or start `1.7-dev` from `master`).

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
- Test state 2026-10-18: full suite green; the real-model baseline passes 8 of 8 on the GTX 1060. `bats` isn't installed here, so `pre-commit run --all-files` reports the
  launcher hook failing; it also reformats about 20 old Markdown files (pre-existing drift).
- `mkdocs` needs `uv sync --inexact --group docs`; build with `uv run --no-sync mkdocs build --strict`.
