# Checks: Video Page (spec 028)

## SC-003: video area at 1440×900 (T026)

Budget as built (`pages/VideoPage.tsx`): header and player controls in one 48 px row, no
footer, a 6 px resize handle, the timeline at its default 200 px with lanes scrolling inside it,
the side panel 360 px wide (Data tab open).

| Page | Video box | 16:9 picture |
|---|---|---|
| Video page (new) | (1440 − 360) × (900 − 48 − 6 − 200) = 1080 × 646 | 1080 × 607.5 (width-limited) = 656,100 px² |
| Results player (old, `/view/:jobId`) | at most (1440 − 384) wide; its header, controls card, 160 px timeline and footer take more of the height | at most 1056 × 594 = 627,264 px², whatever those heights are |

So the new picture is at least 4.6 % larger than the old one could be, with every lane of the demo
video on the timeline (11 lanes, scrolling inside the 200 px). Derived from the CSS: jsdom has no
layout. **Still to do in a browser (T062):** measure both pages at 1440×900 and record the
measured areas here.

## Gates (T060, T061), 2026-10-20

- `pytest tests/`: 2,055 passed, 7 skipped. `ruff check .`, `mypy src/videoannotator`: clean.
- Viewer: `bun run lint`, `bun run typecheck` clean; `bun run test:run` 909 passed, 1 skipped;
  `scripts/build_viewer.sh --check`: bundle up to date.
- Initial bundle 156,770 bytes gzipped (budget 300 KB); the Video page is its own chunk.
- `pre-commit run --all-files`: `bats` isn't installed in the dev container (launcher hook);
  `ruff format` rewrites Python snippets in 19 older Markdown files untouched by 028 (left as they
  were). Everything else passes.
- Performance proxy (`lanePerformance.test.ts`, synthetic hour, 49,161 events): lanes built in
  68 ms; 200 pan steps slicing and binning every row in 64 ms.

## Still to do

- **T062, the walk-through in a browser** (quickstart.md, demo clip and a large video): SC-001
  time to first frame, SC-003 measured areas (above), SC-004 frames while panning, and anything
  that doesn't match. Every behaviour so far is checked in Vitest against the real page with a
  mocked API; nothing has been looked at in a browser.
- **T056, the rest**: `VideoAnnotationViewer` and its controls still serve demo results from the
  local library and `/viewer`'s upload screen; they go with the local library in spec 029.
