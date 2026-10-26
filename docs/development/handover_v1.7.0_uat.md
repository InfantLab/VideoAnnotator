# Handover: v1.7.0 UAT fixes (2026-10-26)

For the agent that builds the fixes. Branch `1.7-dev`. Specs 027–029 and 033 are built; v1.7.0
is now in **UAT and refinement** ([`roadmap_v1.7.0.md`](roadmap_v1.7.0.md), Phase 1): fixes land
on `1.7-dev` as refinements, not new specs.

## Read first

1. [`viewer_design.md`](viewer_design.md): principles P1–P9, page structure, icons, colour,
   names, patterns, and decisions D1–D5. Every fix follows it; it overrides specs 023/029/033
   where its decision log says so.
2. [`uat_v1.7.0.md`](uat_v1.7.0.md): 65 findings (1–37 Caspar's, 38–65 from two Playwright
   walk-throughs), each tagged with the principles it breaks, grouped into batches 0–8 at the top.
3. `AGENTS.md` §23 for viewer code conventions.

## Decisions already made (don't reopen without Caspar)

- **D1** Pipelines named model first, task second ("pyannote community-1 · who spoke when").
- **D2** A Runs page in the nav; a run is also still a filter of Videos.
- **D3** Home is a dashboard (runs going on and finished, recent videos and projects, start
  points) and the viewer opens on it.
- **D4** Ask AI answers are free text by default; structure comes from the prompt, visibly.
- **D5** Fields writes its instruction visibly into the question; forcing the format
  (Ollama `format`, OpenAI-compatible `response_format`) only by an explicit tick box, off by
  default. Fields helps, never restricts. This changes spec 033's server code
  (`pipelines/vlm_annotation/answer_shape.py`, `ollama_client.py`, `openai_client.py`), not just
  the viewer.

## Suggested order

0. **Server stability (#55)** first: SQLite errors under ordinary use ("database is locked",
   "another row available", a 500 in `GET /api/v1/ask/entries` at `api/v1/ask.py:734`). Look for
   sessions or connections shared across threads. Evidence: `logs/errors.log`, 2026-10-26.
1. **Bugs and hazards (batch 1)**: no design needed. Highest value: #38/#39 (claims connected
   without a key; errors that say "set AUTH_REQUIRED=false"), #42 (two clicks to run 21 videos ×
   8 pipelines), #12/#52 (overflow, 768 px header), #31 (Layers panel), #47 (overlay labels).
2. **Names, icons, colour (batch 2)**: one registry field for the display name (D1), the icon
   table in `viewer_design.md`, one hue per result kind. Most later batches depend on it.
3. **Runs page and Home dashboard (batch 3)**, then batches 4–8 in any order.

Write a short ASCII mock-up for Caspar before building batches 3 and 5 and the New run rework
(batch 7); the rest can go straight to code.

## How to work

- Commit and push each tested piece without asking; update the finding's status in the UAT log
  (`Fixed <date>, <commit>`) in the same commit.
- Viewer: `cd viewer && bun run lint && bun run test:run`; rebuild with
  `bash scripts/build_viewer.sh` and commit `viewer_static/` (CI checks it).
- Server: `pytest tests/`, `ruff check .`, `mypy src/videoannotator`.
- To see a change in the real app, the walk-through harness from this session is easy to rebuild:
  Playwright and Chromium are installed under `viewer/node_modules`; the server runs at
  `http://127.0.0.1:18011` (viewer at `/viewer/`, connect link `/viewer-connect?token=…`). Make
  a throwaway key with `videoannotator generate-token --user uat-claude@example.com --no-admin
  --expires-days 7`, label anything created "UAT-Claude", delete it afterwards, and revoke the
  key (no CLI command for that: `get_token_manager().revoke_token(token)` from
  `videoannotator.auth.token_manager`).
- Headless Chromium can't play H.265 (all demo files); that's #54, not a regression.

## Open questions for Caspar (small)

Icon set at 16 px, the Video page data table's default view, an H.264 copy for H.265 videos,
and whether the uat-claude user left in the database should be removed (there's no command for
it).
