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
