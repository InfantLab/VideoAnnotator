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

## Batch 5: Ask AI

Answers #21–27, #29, #30, #50, #56–58, #65 (D4, D5; #28 is decided by them). The page keeps its
layout: question on the left, video on the right, then the timeline, the answers and the tally.

### Question and answer shape (#27, #29, #30, #57, #58; D4, D5)

```
 Question                                                         [Examples ▾]
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ How many adults are in the room, and is the child smiling?               │
 │                                                                          │
 │ Answer as JSON with "Adults": a number, "Child smiling": yes or no.      │ ← written by Fields,
 └──────────────────────────────────────────────────────────────────────────┘   in the text, editable
 Answer   ┃ Free text ┃▐ Fields ▌┃                              ← clearly selected (#30)
          Adults           [number      ▾]   ✕
          Child smiling    [yes or no   ▾]   ✕
          [+ Add a field]   [Suggest from the question]
          ☐ Make the model answer in exactly this form   (this server can enforce it)
```

- Free text by default (D4). Fields writes its sentence into the question, where you can see and
  edit it; changing a field rewrites that sentence (D5). Nothing else is added to what's sent.
- Field names as typed ("Child smiling", "Child's mood"); the column and JSON key are made from
  them behind the scenes (#29).
- The tick box forces the format through the server's structured output: off by default, shown only
  where the server supports it (D5).
- Suggest proposes one field per part of the question ("adults: number", "child smiling: yes or
  no"), or says it found none; it never replaces fields you made without asking (#57).
- Examples use plain words: "touching or not touching", "at the caregiver" (#58).

### Model, and where it asks (#21, #22, #25, #26)

```
 Model    [ gemma4:12b                ▾ ]   🖴 On this computer
                                            (☁ On ai.example.org: frames leave this computer)

 Ask at   (•) This frame: 0:12.4, frame 186                       [ Try ]
          ( ) The moments below: every [5] s from [0:00] to [end]  =  4 moments    [ Try on 4 ]
          Whole video: 43 moments, about 2 min                    [ Run on whole video ]

 Frames per moment  [ 1 ▾ ]  [0.2] s apart, centred on the moment
          Sends 0:12.2 · 0:12.4 · 0:12.6, so the model can see movement around the moment.
```

- Where it runs sits beside the picker, in words and an icon (#25); the top-right badge goes.
- "This frame" names the time and frame number from the player bar (#21, built in batch 4).
- The moments' grid is spelled out on the same line as its button, with no shuffle icon (#22).
- Frames per moment: 1, 3 or 5 frames, apart by a fraction of a second (0.04–2 s), always centred
  on the moment, with the times it will send written out (#26).

### Answers: one key, one selection (#23, #24, #27, #50)

```
 Answers                                                              [ Export ▾ ]
                0:00        0:05        0:10        0:15        0:20
 Try 14:02 · gemma4:12b · Fields    ●                                         ← only the frame tried
 Run 14:10 · gemma4:12b · Fields    ●     ●     ◉     ●     ●     ●            ◉ = selected
 Run 14:31 · qwen2.5vl  · Fields    ●     ●     ●     ●     ●     ●
          Child smiling:  ● yes  ● no  ● no answer               ← one colour key, no red (#23, #50)

 ┌ Time ┬ Run 14:10 · gemma4:12b ───────┬ Run 14:31 · qwen2.5vl ───────┐
 │      │ Adults │ Child smiling        │ Adults │ Child smiling       │
 │ 0:00 │ 2      │ yes                  │ 2      │ yes                 │
 │▶0:10 │ 1      │ no                   │ 1      │ yes                 │ ← selected row = ◉ dot
 └──────┴────────┴──────────────────────┴────────┴─────────────────────┘
 Tally (Run 14:10):  Child smiling  yes 4 · no 2      Adults  1 ×2 · 2 ×4
```

- Each try or run is labelled by when, model and shape, the same on its track, its table columns
  and its tally, so two tries of one question can be told apart (#24).
- Dots are coloured by the answer, from one key under the timeline: the first field's values for
  Fields, the answer itself for short free-text answers. Neutral categorical colours, never red;
  failed or missing answers are hollow grey (#23, #50).
- Selecting a row, a dot or a tally line selects the same moment everywhere (#24).
- The tally counts one try or run at a time, chosen above it, never mixing a frame try into a run's
  counts (#50).
- A free-text reply that is valid JSON (in ```json fences or not) is shown in columns, marked
  "read from the reply"; what's stored stays the reply as given (#27, D4).
- Times as `0:30` everywhere, including frame thumbnails (#50).

### The video, history and saved questions (#50, #56, #65)

```
 2UWdXP.joke1.rep2…Peekaboo_h265.mp4   [ Choose another video ▾ ]
     ┌ Videos on the server ───────────── Search… ┐
     │ expCP2.mov                       ran 26 Oct │
     │ 2UWdXP.joke1.rep2…Peekaboo.mp4   ran 26 Oct │
     │ …                                           │
     ├ From this computer: [ Upload a video… ]     ┤
     └─────────────────────────────────────────────┘

 History           ★ How many adults…  · Run 14:10 · gemma4:12b · Caspar     ⋯
 Saved questions     Is the child touching the tablet?                        ⋯  Use · Delete
```

- Choose another video lists the server's videos first, with upload second (#56).
- History names people, not user ids (#50).
- Deleting a history entry asks whether to remove its question from Saved questions too, and saved
  questions can be deleted on their own (#65).

### Decided (Caspar, 2026-10-27)

1. Dots are always coloured by the answer, free text included, and consistently: an answer keeps its
   colour across every try and run of the same question. 16 distinct colours; beyond 16 answers, the
   15 most common keep theirs and every other answer takes a shade of the colour of the answer it
   reads most like (similar answers, similar colours). Similarity is by shared letter sequences, no
   model needed; embeddings could replace it later.
2. Where the server can't enforce the format, the tick box shows, disabled, saying so.

## Batch 7: New run

Answers #35 and #59 (#42 and #44 are done) (P2, P4, P6, P8). One page, three short parts; today's
long Settings section moves behind "All settings".

### Which videos (#59)

Chosen already (from Videos, a project, a run): one line, not a section.

```
 New run
 Videos   6 videos in peekaboo: 2UWdXP.joke1.rep2…mp4, 4JDccE.joke3…mp4 +4      [ Change ]
```

Nothing chosen yet: the tabs a user can use first, in words that say whose computer it is.

```
 Which videos
 [ ⤒ Upload from your computer ]  [ 🗀 A project ]  [ ◷ A video you ran before ]  [ 🖴 A folder on the server ]
```

- "Your computer" is always the researcher's own machine; "the server" is where VideoAnnotator
  runs (also Home's "Open files from your computer", Help and Videos' empty state) (P8).
- Opens on the first tab the user can use: a folder on the server for an administrator, upload
  for everyone else. For a non-administrator the server tab stays, last, saying why it's locked
  and what to do (P1, P6).

### What to find, each with its settings (#35)

One row per pipeline, model first, task second (D1), grouped by kind's icon instead of headings;
no group names, ids or counts of "capabilities". Each pipeline's settings sit under it as an
accordion, collapsed by default (Caspar, 2026-10-27): little on the page, settings one click from
the pipeline they belong to.

```
 What to find                                         Settings from this project's, 26 Oct ▾
  [✓] 🔊 pyannote community-1      who spoke when                         defaults  ▸
  [✓] 🗨 Whisper large-v3-turbo    what was said                 Language: English  ▸
  [✓] 😊 DeepFace                  where faces are and their expression
                                                              Age on · Gender on   ▾
      ┌──────────────────────────────────────────────────────────────────────────┐
      │ Smallest face        [ 30 ] px      at least 1                           │
      │ Most faces per frame [ 10 ] faces                                        │
      │ Expression           [on]                                                │
      │ Age                  [on]   ● changed, from the project   Reset          │
      │ Gender               [on]   ● changed, from the project   Reset          │
      │ ▸ Technical settings (5, all at defaults)                                │
      └──────────────────────────────────────────────────────────────────────────┘
  [ ] 🙂 OpenFace 3                action units, gaze, head pose
  [✓] 🧍 YOLO26 pose               where people are and how they move    defaults  ▸
  [ ] 🎬 PySceneDetect + CLIP      cuts and settings
  [✓] ✨ Ask AI                    "what is the expression on the child's face?" · every 5 s  ▸
```

- The cue on a collapsed row is the summary: "defaults", or what differs ("Age on · Gender on"),
  so changed settings are visible without opening anything (P6). Unticked pipelines show no
  settings cue; ticking one doesn't open it.
- Open, each setting is one line: name, field, unit and range; the description on hover, not
  repeated under the name. A changed setting is marked, says where it came from, and has Reset.
  Technical settings stay folded inside.
- Age shown on with nothing saying why (#35) can't happen: the collapsed row says "Age on" and
  the open one says it came from the project.
- The source menu (▾, top right): this project's settings · your last run, <name> · a workflow ·
  the defaults. Says where the ticks and settings came from, as now (#44), in one place.
- Ask AI has no defaults to fall back on, so its row always shows its question and spacing.
- No separate Settings section and no "All settings" page: today's long section goes.

### Start

```
 Start   6 videos × 5 pipelines · about 25 min      Name [ peekaboo · pyannote, Whisper, +3 ]
         [ ▶ Start run on 6 videos ]      You can close this tab; results appear on Videos and Runs.
```

- Size and time on the button's line; above 10 video × pipeline pairs it still asks first (#42).

### Decided (Caspar, 2026-10-27/28)

1. Settings sit under their pipeline as an accordion, collapsed by default (above).
2. Non-administrators keep the server-folder tab, locked and last, saying why.
3. No time estimate before the first video finishes: keep "estimated once the first video
   finishes"; Start shows the size ("6 videos × 5 pipelines").
