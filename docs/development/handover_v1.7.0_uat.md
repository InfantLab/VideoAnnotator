# Handover: v1.7.0 UAT fixes (updated 2026-10-27)

For the agent that carries on with the fixes. Branch `1.7-dev`. Specs 027–029 and 033 are built;
v1.7.0 is in **UAT and refinement** ([`roadmap_v1.7.0.md`](roadmap_v1.7.0.md), Phase 1): fixes
land on `1.7-dev` as refinements, not new specs.

## Where it stands

All batches (0–8) are done (2026-10-26 to 28; batch 6 `1fe6a5f`, batch 8 `82b7494`, batch 7
the commit after the mock-up). Every fixed finding in [`uat_v1.7.0.md`](uat_v1.7.0.md) says what
was done and in which commit. Left: the Open items in `viewer_design.md`, the walk-throughs
(027 T068, 028 T062, 033 T055), and whatever Caspar's next pass finds.

Not done in batch 8, on purpose: filtering the Videos table by column (sorting covers "none
found" and "failed"); result values are cached in memory only, so the first Videos page after a
server start reads the files again (about 1.5 s for this dev database).

## Read first

1. [`viewer_design.md`](viewer_design.md): principles P1–P9, page structure, icons, colour,
   names, patterns, decisions D1–D7. Every fix follows it.
2. [`uat_v1.7.0.md`](uat_v1.7.0.md): the findings and their status; batches at the top.
3. [`uat_v1.7.0_mockups.md`](uat_v1.7.0_mockups.md): the mock-ups Caspar approved for batches 3
   and 5, with his decisions; add batch 7's there.
4. `AGENTS.md` §23 for viewer code conventions.

## Decisions made (don't reopen without Caspar)

- **D1** Pipelines named model first, task second. **D2** Runs page in the nav (`/runs`, a run's
  page at `/runs/<id>`). **D3** Home is a dashboard and the viewer opens on it. **D4** Ask AI
  answers are free text by default. **D5** Fields writes its sentence into the question; forcing
  the format is an explicit tick box (`enforce_shape`), off by default. **D6** The Video page's
  Data tab opens on a summary. **D7** H.265: the server makes an H.264 copy on request
  (`playable.py`, `/api/v1/videos/playable-copy`).
- Caspar, 2026-10-27: the Videos sidebar is collapsible and stays on a run's page; the demo stays
  at the bottom of Home (to be made as rich as possible later); Ask AI answers are always
  coloured, consistently per question (`lib/answerColours.ts`); an unsupported force tick is
  shown disabled.
- **No backward compatibility work for data from before v1.7.0** (old run names, old addresses
  beyond the redirects that exist): it's the first public release.

## How to work

- Commit and push each tested piece without asking; update the finding's status in the UAT log
  (`Fixed <date>, \`<commit subject>\``) in the same commit.
- **Run the tests for what you changed, not the full suites** (Caspar: full runs per batch are
  excessive); lint and type checks are quick, so run those. Pre-commit and CI cover the rest.
  - Viewer: `cd viewer && bun run lint && bunx tsc --noEmit -p tsconfig.app.json`, then
    `bunx vitest run <files>`. Rebuild with `bash scripts/build_viewer.sh` and commit
    `viewer_static/` (CI checks it). After an API change: `bash scripts/gen_viewer_api_types.sh`
    (needs `.venv/bin` on PATH).
  - Server: `.venv/bin/python -m pytest <files>`, `.venv/bin/ruff check .`, `ruff format`.
- Known failures that aren't yours: `viewer/src/test/lib/lanePerformance.test.ts` (environment);
  `tests/api/test_ask_suggest_shape.py` run alone (missing table; passes in a wider run);
  `routes.test.tsx > /videos` sometimes times out under load.
- To see a change in the real app: start `.venv/bin/videoannotator server --port 18998` in the
  background, make a throwaway key with `videoannotator generate-token --user
  uat-claude@example.com --username uat-claude --key-name UAT-Claude --no-admin --expires-days 1
  --port 18998 -o <scratch>/realkey.json </dev/null`, open `/viewer-connect?token=…`, drive it
  with Playwright (installed under `viewer/node_modules`; require it via `createRequire` from
  `viewer/package.json`). Afterwards revoke the key
  (`get_token_manager().revoke_token(token)` from `videoannotator.auth.token_manager`) and stop
  the server (find its PID with `ps -eo pid,args`; `pkill -f` kills your own shell).
- Ollama with gemma4:12b is reachable from the dev container, so Ask AI can be tried for real.
- Headless Chromium can't play H.265; the demo files are H.265, so use "Make a copy this browser
  can play" (D7) in walk-throughs.

## Open questions for Caspar (small)

Icon set at 16 px in both themes (`viewer_design.md` Open); whether the uat-claude user left in
the database should be removed (no command for it); old playable copies are never cleaned up
(`<storage root>/playable/`), which may want a rule later.
