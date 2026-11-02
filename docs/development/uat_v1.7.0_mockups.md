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

## Batch 9: reporting a problem, the footer, tips on the Video page

Answers #68, #69 and #70. Nothing new is sent anywhere by the viewer itself: a report goes where
the researcher sends it, with what they can read first (P1).

### Report a problem (#68)

One dialog, opened from the footer on every page, the Video page's `?` menu, and every error
message ("Something went wrong", a failed run's "why?", a connection that won't work).

```
 ┌ Report a problem ─────────────────────────────────────────────────────────┐
 │ What happened?                                                            │
 │ ┌───────────────────────────────────────────────────────────────────────┐ │
 │ │ The Speech lane is empty but the Data tab shows words                 │ │
 │ └───────────────────────────────────────────────────────────────────────┘ │
 │ What did you expect? (optional)                                           │
 │ ┌───────────────────────────────────────────────────────────────────────┐ │
 │ └───────────────────────────────────────────────────────────────────────┘ │
 │                                                                           │
 │ [✓] Add details that help us find it                                      │
 │     ▸ What's added                                                        │
 │       Viewer 1.7.0 · server 1.7.0 at 127.0.0.1:18011 · Firefox 132,      │
 │       Linux · 1440 × 900 · page: the Video page                           │
 │       [✓] the video's name: 2UWdXP.joke1.rep3…Peekaboo_h265.mp4          │
 │       Last 3 errors: "Can't read …speech_recognition.vtt: …"  (14:02)     │
 │       Never added: your key, folder paths, results.                       │
 │                                                                           │
 │ A screenshot helps: add one to the issue or email after it opens.         │
 │                                                                           │
 │           [ Copy ]   [ Email it ]   [ Open a GitHub issue ↗ ]             │
 └───────────────────────────────────────────────────────────────────────────┘
```

- **GitHub issue**: a new issue in InfantLab/VideoAnnotator, title and body filled in, for people
  with a GitHub account. A `bug_report.yml` issue template with the same headings, so reports that
  start on GitHub look the same.
- **Email**: the user's own mail program with the report filled in, to the project's address. For
  researchers without GitHub, which is most of them.
- **Copy**: for anything else (a lab's Teams channel, a support thread).
- The details are shown before anything leaves, and each can be unticked. The video's name is
  ticked by default only for demo clips; a study's file names can identify people, so for those
  it starts unticked (P1).
- Errors come from a small list the viewer keeps in memory of its own failures (a file it
  couldn't read, a request the server refused), not from the browser console.

### The footer (#69)

On every page, the Video page included. One line, each link once:

```
 VideoAnnotator 1.7.0 · server 1.7.0      Help · Docs ↗ · Source ↗ · How to cite ↗ · Report a problem
```

- "Video Annotation Viewer … Powered by VideoAnnotator ↗" goes: the viewer is part of
  VideoAnnotator. The server's version shows only when connected, in amber when it differs
  in amber ("server 1.6.0: older than this viewer").
- "How to cite" is the Zenodo record (the concept DOI, so it always reaches the latest version).
- On the Video page the footer is the same line in the space under the timeline, which today is
  empty below the last lane; it scrolls with the lanes rather than taking fixed height.

### Tips on the Video page (#70)

The first time someone opens a video, a few tips, one at a time, each pointing at the thing it's
about. Only tips for what is on screen: no list tip for a video opened on its own, no notes tip
without a server.

```
                           ┌──────────────────────────────────────────┐
  peekaboo · 3 of 48 ‹ ›   │ 1 of 4                                    │
  ▲                        │ You opened this from a list: ‹ › go to    │
  └────────────────────────│ the previous or next video in it, with    │
                           │ the same lanes shown.                     │
                           │                     [ Skip tips ] [ Next ]│
                           └──────────────────────────────────────────┘
```

1. **List** (when opened from a run, project or folder): "‹ › go to the previous or next video in
   it, with the same lanes shown."
2. **Layers**: "Choose which results show as lanes, and which are drawn on the video."
3. **A lane**: "Click a lane to select it; [ and ] jump to its previous or next event."
4. **Notes** (with a server): "Press N for a note at this moment, Shift+N twice for a span. They're
   kept with the video on the server and listed in the Notes tab."

- Skip ends them; they don't come back by themselves. The `?` menu becomes Help: Keys, "Show the
  tips again", Report a problem.
- Remembered in this browser only, like the panel's open or closed state.
- Built from the popover the viewer already has; no tour library.

### Decided (Caspar, 2026-11-02)

1. Emailed reports go to infantologist@gmail.com.
2. The same footer on every page, the Video page included.
3. The four tips as a start; Caspar tries them and says what to add.

Built 2026-11-02. Found while building: every page already had a footer (on long pages it is only
at the very bottom), so #69 became what the footer says, not where it is.

## Batch 10: the Pipelines page

Answers #71. Today the page is for installing; this makes it the place to learn what each pipeline
does and how it's set, day to day, with installing as one part of that (P1, P6, P7, P8).

### The top: what a pipeline is

```
 Pipelines                                                         [ ▶ New run ]
 A pipeline is a published research tool (a model and the code around it) that
 VideoAnnotator runs on your videos for you. Each finds one kind of thing, such as who
 spoke when or where faces are, and writes it as results you can view, compare and export.
 They run on the server; your videos aren't sent anywhere else.

 [ Pipelines ]   [ Workflows · 2 ]
```

- Two tabs: Workflows moves out from the bottom into its own tab (below). The tab is in the URL
  (`/pipelines#workflows`), so New run and Help can link to it.

### Each pipeline: what it finds, its settings, where it comes from

One row each, grouped under Sound, Faces, People, Scenes, collapsed by default; a row says what it
needs, if anything, without opening.

```
 Sound
  🔊 pyannote community-1   who spoke when     Needs setup: a Hugging Face key ▸
  🗨 Whisper large-v3-turbo what was said                                 Ready    ▾
     ┌───────────────────────────────────────────────────────────────────────────┐
     │ Writes each stretch of speech with its words and times: a Speech lane on  │
     │ the Video page; exports as WebVTT and CSV.                                │
     │                                                                           │
     │ Settings                     default        your last run, 30 Oct         │
     │ Language                     detected       English ●                     │
     │   The language spoken; left to detection, each video is guessed alone.    │
     │ Retry looping text           on                                           │
     │   Re-reads a stretch that repeats itself, as Whisper does by default.     │
     │ Word times                   off                                          │
     │ ▸ Technical settings (9)                                                  │
     │                                                                           │
     │ ▸ About: homepage · 13 min of audio in 1m03s on a GPU · Cite (BibTeX)     │
     │                                     [ ▶ Run on videos… ]  v1.7.0          │
     └───────────────────────────────────────────────────────────────────────────┘
 Faces
  😊 DeepFace               where faces are and their expression          Ready    ▸
  🙂 OpenFace 3             action units, gaze, head pose     Not installed [ Install ]
 People
  🧍 YOLO26 pose            where people are and how they move           Ready    ▸
 …
```

- **Settings** read as a reference: each in one line with its default, and under it, in a
  sentence, what it changes (the description that is only on hover in New run). The second
  column shows where your last run differed (●), so "why is Age on?" has an answer here too.
  Settings are changed for a run in New run, or saved under a name as a workflow; this page
  doesn't keep a third set of its own (see the decision below).
- **Run on videos…** opens New run with this pipeline ticked.
- **What it finds** says the lane and the files, in the viewer's words, so a researcher knows
  what to expect before running it.
- Install, setup and restart stay as they are, on the row they're about. The deprecated
  "Whisper + pyannote" goes under "Older pipelines" at the end, folded.

### Ask AI: set apart, saying how it differs

Not a row among the others but its own panel after them, in Ask AI's colour.

```
 ┌ ✨ Ask AI ──────────────────────────────────────────────────────────────────┐
 │ Different from the pipelines above. They're each one published model,     │
 │ tested on a benchmark, finding one fixed thing. With Ask AI you write the │
 │ question, and you choose the model that answers it:                       │
 │                                                                           │
 │  · Your question decides what's found: "Is the child looking at the       │
 │    caregiver?" Answers are as good as the question and the model.         │
 │  · No benchmark applies. Try a question on a few moments first, check the │
 │    answers yourself, then run it on your videos.                          │
 │  · The model runs on a model server you set up (Ollama, LM Studio,        │
 │    llama.cpp…), on this computer or another. Now: Ollama · qwen3.5 ●      │
 │  · Slower: each moment is one question to the model.                      │
 │                                                                           │
 │             [ Try a question ]   [ Model servers ]   ▸ Settings (13)       │
 └───────────────────────────────────────────────────────────────────────────┘
```

- **Try a question** opens Ask AI; **Model servers** opens Settings → Ask AI.
- The model line is live (P1): "No model server yet" with Model servers as the one thing to do.

### Workflows tab: what one is, and making one here

```
 [ Pipelines ]   [ Workflows · 2 ]

 A workflow is a set of pipelines and their settings saved under a name: to run the same
 way again, to give a project the same settings, or to send to a colleague as a file.
 Nothing uses one until you choose it, in New run or on a project.

 [ + New workflow ]   [ Import a file ]

 Peekaboo standard            from the project peekaboo · 26 Oct
 pyannote, Whisper, DeepFace (Age on)
 [ Use in a new run ] [ Apply to a project ▾ ] [ Edit ] [ Export ] [ ⋯ ]
```

- **New workflow**: a name and New run's "What to find" (ticks, settings under each); no videos.
  Also still saved from a project or a run, as now.
- **Edit** opens the same; today a workflow can only be renamed.
- Each card says what differs from the defaults ("DeepFace (Age on)"), as New run's rows do.

### Adding another tool

Last on the Pipelines tab, short:

```
 Another tool you'd like?
 VideoAnnotator comes with the pipelines above. To suggest one, Report a problem and say
 which tool and what it finds, or open a GitHub issue ↗. Many questions need no new
 pipeline: Ask AI can look for what you describe.
 Developers: from VideoAnnotator 1.8 a pipeline can be added as a plugin, a Python package
 installed beside it, which then shows on this page. How a pipeline is built ↗
```

- "Plugin" is used only here, for that way of adding one; the thing itself stays a pipeline
  everywhere (P8).

### To decide

1. Settings on this page: a reference (above), or also editable as "your defaults" that New run
   starts from? Editable means a third place settings come from, beside a project's and a
   workflow's; recommended: the reference, with New run and workflows for changing them.
2. Workflows: a tab here (above), or a page of their own?
3. The name: keep "Pipelines" (Help's glossary and every page say it), with "plugin" only for
   how one is added?

### Decided (Caspar, 2026-11-02)

1. Settings are editable here as your own defaults, which New run starts from.
2. Workflows are a tab on Pipelines.
3. The name stays "Pipelines"; "plugin" only for how one is added.

Built 2026-11-02. How your defaults fit with the rest (P6): they fill only settings that the starting
point (a project, a workflow, your last run, the defaults) leaves unset, never what one sets, and
New run says so ("with your defaults for Whisper", and "● your default" on the setting). A run
records them like any other setting. A run being repeated, or settings brought from Ask AI, are
kept exactly. Kept per user on the server (`user_preferences`, `pipeline_defaults`;
`GET/PUT /api/v1/videos/run-settings/defaults`). Found while building: deprecated pipelines aren't
in the catalog, so there is no "Older pipelines" fold; there is no developer guide to building a
pipeline yet, so the developers' line links to the pipelines' source. Edit replaces Rename on a
workflow card.
