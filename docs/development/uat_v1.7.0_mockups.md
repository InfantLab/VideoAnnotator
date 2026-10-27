# UAT v1.7.0: mock-ups

Short ASCII mock-ups for Caspar before building the larger UAT batches (handover). Each section
says which findings it answers; the design rules it follows are in `viewer_design.md`.

## Batch 3: Runs page, a run's page, Home, status in place

Answers #1, #3, #4, #6, #11, #43, #44 (D2, D3, P3, P4, P6).

### Nav

```
 VideoAnnotator   Home  Videos  Runs  Ask AI  Pipelines            ⟳ 1 run · 3/8   ?  ⚙
```

Runs is new (D2). The viewer opens on Home (D3), including for a set-up viewer.

### Runs (`/runs`)

Every run, one row each: running first, then newest first. A whole row opens the run's page.

```
 Runs
 ┌─────────────────────────────────────────────────────────────────────────────────────────┐
 │ ⟳  expCP2 · Whisper, pyannote    8 videos: 2UWdXP.joke1…, 3KdpQ.joke2… +6   🗨 🔊      │
 │    3 of 8 done · 1 running · 4 waiting                     today 17:50 · 6 min so far ⋯ │
 │    ▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░                                                               │
 ├─────────────────────────────────────────────────────────────────────────────────────────┤
 │ ✓  expCP2 · PySceneDetect + CLIP  3 videos: 2UWdXP.joke1…, 9QmX…, Peek… 🎬              │
 │    3 done                                                  today 16:02 · 4 min        ⋯ │
 ├─────────────────────────────────────────────────────────────────────────────────────────┤
 │ ✕  Pilot faces                    48 videos in peekaboo                 😊 🧍           │
 │    46 done, 2 failed (why?)                                25 Oct, 11:14 · 1 h 12 min ⋯ │
 └─────────────────────────────────────────────────────────────────────────────────────────┘
                                       Show older runs
```

- State icon: ⟳ running (spinning), ◷ waiting, ✓ done, ✕ some failed, ■ stopped.
- Name: as #41 named it. Videos: count and the first names, or the project when it ran on one.
- What was found: the kinds' icons in their colours (batch 2), names on hover.
- Outcome in words; "why?" shows the first failure's reason in plain words.
- When and how long, in the one date format. Progress bar only while running.
- ⋯ menu: Open · Run again… · Retry failed videos · Stop · Download table · Delete.
- Empty: "No runs yet." and a **Run pipelines on videos** button.

### A run's page (`/runs/<id>`)

Today's run view on Videos, made into the run's own page (#3, #4). The Videos sidebar and the
all-videos intro go; the header says what to do with the results, and the table is labelled as this
run's videos.

```
 Runs › expCP2 · PySceneDetect + CLIP
 ┌─────────────────────────────────────────────────────────────────────────────────────────┐
 │ ✓ expCP2 · PySceneDetect + CLIP                     Run again…  Retry failed   ⋯        │
 │   3 videos · 🎬 Scenes · today 16:02 · 4 min · 3 done                                   │
 │                                                                                         │
 │   Results:  Open a video to see them on its timeline,                                   │
 │             or take all of them:  [Download table]  [ELAN]  [Methods]                   │
 └─────────────────────────────────────────────────────────────────────────────────────────┘

 Videos in this run (3)
 ┌──────────────────────────────────────┬──────────────┬───────────┐
 │ Video                                │ 🎬 Scenes    │           │
 ├──────────────────────────────────────┼──────────────┼───────────┤
 │ 2UWdXP.joke1.rep2…Peekaboo_h265.mp4  │ 14 scenes    │  Open  →  │
 │ 9QmXk.joke3.rep1…Peekaboo.mp4        │ 9 scenes     │  Open  →  │
 │ Peekaboo_demo.mp4                    │ ✕ failed     │  Open  →  │
 └──────────────────────────────────────┴──────────────┴───────────┘
```

- Only the run's pipelines as columns. Each row has an explicit **Open →** as well as the name link.
- While running: the progress bar under the header; rows show ⟳ / ◷ in their cells.
- `/videos?run=<id>` (today's links, the activity menu) redirects here.

### Home (`/home`, where the viewer opens)

```
 VideoAnnotator                                                   v1.7.0 · Release notes

 ┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
 │ ✨ Ask AI about a video  │ │ ▶ Run pipelines on      │ │ 📂 Open files from this │
 │                         │ │   videos                │ │   computer              │
 └─────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘

 Runs                                                                    All runs →
   ⟳ expCP2 · Whisper, pyannote   8 videos   3 of 8 done   ▓▓▓▓░░░░   today 17:50   ⋯
   ✓ expCP2 · PySceneDetect + CLIP 3 videos   3 done                   today 16:02   ⋯

 Recent videos                                                         All videos →
   2UWdXP.joke1.rep2…Peekaboo_h265.mp4    🗨 🔊 🎬 😊           ran today 17:50
   9QmXk.joke3.rep1…Peekaboo.mp4          🎬                    ran today 16:02
   …5 in all

 Projects                                                              All projects →
   peekaboo · 48 videos · last run 25 Oct        pilot-2 · 12 videos · no runs yet

 Try it on a demo clip — real results, opened on the Video page, no server needed
   [Peekaboo: a parent and baby; speech, speakers, faces, people]
```

- Runs: those going on and those finished in the last 24 hours (the activity list), same rows as
  Runs. Nothing going on: "Nothing running. Your last run: …" with one row.
- Recent videos: the 5 most recently run, with what's been found as kind icons.
- Projects: only when there are some.
- The status strip (connected, pipelines ready, videos) and the set-up checklist show **only when
  something needs doing** (no server, no key, no pipeline ready, no videos yet); a first-time
  user sees the checklist where Runs would be.

### Status in place on Videos (#6, #43)

A pipeline that's running or waiting on a video shows that in the video's cell, even when an older
result exists; when it finishes, the new value replaces it.

```
 │ 2UWdXP.joke1.rep2…Peekaboo_h265.mp4  │ ⟳ running    │ 14 scenes │ 2 speakers │
 │ 9QmXk.joke3.rep1…Peekaboo.mp4        │ ◷ waiting    │ 9 scenes  │ —          │
```

Hovering the cell names the run and links to it. (Values in cells instead of ticks are batch 8.)

### A new project's settings (#44)

A new project has no settings of its own, and New run on it starts from the defaults (nothing
ticked), not silently from your last run. The last run is offered visibly instead:

```
 What to find                                  Start from:  ( ) nothing   ( ) your last run, expCP2 · Scenes
```

## Decided (Caspar, 2026-10-27)

1. A run's page keeps the Videos sidebar, made collapsible.
2. The demo stays at the bottom of Home for everyone; the demo is to be made as rich as possible, as
   a reference for what the pipelines can find.
