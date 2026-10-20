# The Video page

One video, everything found in it by any run, on one time axis, with the values beside it and
your own notes. Open it from a run's videos, a project, the Jobs list or Home (**View**); its
address is `/video?path=<the video's path on the server>`.

```
┌ ←  p01_t1.mp4  Peekaboo study · 3 of 48 ‹ ›                 Ask AI  Note  Run…  Export  ⋯ ┐
│                                                  │ Data │ Notes │ Runs │ About        │
│              video (with overlays)               │ the results table, following     │
│                                                  │ playback                         │
│ ▶ 0:12.4 / 3:20  frame 372  ━━━━●──────  ⏮ ⏭ 1×  │                                  │
├──────────────────────────────────────────────────┴──────────────────────────────────┤
│ [Layers]  − + ⤢  ▕██████▏──────────────── (visible range)                           │
│ 👁 Speech                  ▬▬▬▬  ▬▬▬▬▬     ▬▬                         run 12 Oct ▾  │
│ 👁 Speakers  SPEAKER_00    ▬▬      ▬▬▬                                              │
│ 👁 People    person 1 (parent) ▔▔▔▔▔▔▔▔▔▔▔                                          │
│    Notes (yours)           ▲      ▭▭▭                                               │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

The video plays straight away, streamed from the server: nothing is downloaded first and nothing
asks for a folder. Only the result files the tracks show are fetched. The controls are under
the video: play, the time, a seek bar, previous and next event, and speed.

## Tracks

There is one track per kind of result the video has, and none for results it hasn't: Speech,
Speakers (a row per speaker), People (a row per tracked person, with the label the pipeline gave),
Faces (when faces were seen, and a **count** row: faces per sampled frame), Emotion, Scenes, Ask
AI, Action units, Head pose, Gaze, and **People · box area (not movement)**. A result the viewer
has no track for gets a track named after its pipeline and measure, never dropped. Hover a track's or
a row's name to see what it shows and which pipeline made it.
A pipeline that failed keeps its track, saying why. One that finished but whose result files are
no longer in the run's results folder (moved or deleted) says that instead: run it again to get
them back.

- **Spans** (speech, turns, scenes) are bars; **moments** (Ask AI answers) are ticks; **numbers**
  (scores, head pose, the face count) are small charts with their scale at the right. A person or
  face is drawn as present only between samples no further apart than the pipeline's sampling
  interval: gaps are left as gaps.
- When a track has more events than there are pixels, it is drawn as **density** and says so.
- **Zoom** with `+` / `−`, the buttons, or Ctrl/⌘ + wheel around the pointer; **move** along the
  timeline by dragging, Shift + wheel, or the visible-range bar; `0` shows the whole video. While
  playing, the view moves on when the playhead leaves it.
- **Hover** an event for its text, times and confidence; **click** it to go there.
- **Select** a track (click a row's name or the track) and step through its events with `[` and `]`,
  or ⏮ ⏭.
- **Click a track's name** to fold it to one line; its **eye** shows or hides what it draws on the
  video (tracks that draw nothing have no eye). OpenFace 3's Faces track has a menu for its
  landmarks, gaze, head pose and action units. **Layers** lists every track with both switches, and
  all / none. **Drag the timeline's top edge** to make it taller.

These choices are remembered for you, on this computer, for every video.

## Which run

Each track says which run it shows: by default the newest run whose result for that pipeline
completed (a link from a run starts on that run). Its menu lists the video's other runs (a failed
one says why) and **Compare with…**, which adds the other run's track directly below. **About**
then lists the settings that differ between the two, as their provenance records them.

## Data, About, Runs

- **Data** is the [results table](exporting_results.md#the-table) for this video and the runs
  shown, with your notes as `note` rows. The row at the playhead is highlighted and kept in view
  until you scroll (**Follow playback** resumes). Filter by track, search, and click a row to go
  there; clicking an event on the timeline picks out its row.
- **About** gives, for each pipeline and run, the model, VideoAnnotator's version and the settings
  ("not recorded" for results made before provenance), and the result files to download.
- **Runs** lists the runs on this video.

## Notes

Press **N** for a note at this moment (or **Note** at the top), type, and press Enter; Esc
cancels. **Shift+N** marks the start of a span and Shift+N again its end, in either order.
Notes are kept on the server with the video, not a run, so they survive new runs; everyone using
the server sees them, and each records who wrote and who last changed it. They have their own
track, last and outlined, and are never mixed into a pipeline's track. Click one to change or
delete it, or use the **Notes** tab. They are in the table export (`note` rows with an `author`)
and the ELAN export (a `Notes (researcher)` tier): see [Results out](exporting_results.md).

Keys never fire while you are typing in a note or the search box. **?** lists them all.

## Where the video belongs

Opened from a run, project or folder, the header names it with "3 of 48" and **‹ ›** to the
previous and next video there, with your track settings kept. Opened directly, it lists the
projects the video is in. **Ask AI** opens Ask AI on this video, **Run…** sets up a new run
starting from this video's newest, **Export** gives the table, ELAN files or a methods paragraph
for each run shown, and **⋯** has the debug panel.

## Files from your computer

`/viewer` opens a video and result files from your computer on the same page, without a server:
there are no runs to choose, and notes need a server.

## Older addresses

`/view/<job id>` still works: it opens the Video page for that job's video, starting on that run.

## API

Every signed-in user may use these (`Authorization: Bearer <API key>`):

| Request | What it does |
|---|---|
| `GET /api/v1/videos/runs?path=` | The video (name, whether it is still there, duration, content hash when a run recorded one), every run on it newest first with each pipeline's status, error and files (`kind`: `result`, `provenance` or `other`, with download addresses), and the projects holding it. 403 `PATH_NOT_ALLOWED` for a video no run read outside the shared folders. |
| `GET /api/v1/videos/notes?path=` | The video's notes, by time. |
| `POST /api/v1/videos/notes` | `{"path", "start_sec", "end_sec" (null for a moment), "text"}`. 422 `NOTE_TEXT_REQUIRED`, `NOTE_TEXT_TOO_LONG` (2,000 characters), `NOTE_TIME_INVALID`. |
| `PATCH /api/v1/videos/notes/{id}` | Any of `text`, `start_sec`, `end_sec`; records who changed it. 404 `NOTE_NOT_FOUND`. |
| `DELETE /api/v1/videos/notes/{id}` | Deletes it (204). |

The video itself plays through `POST /api/v1/videos/stream-token` and
`GET /api/v1/videos/stream?path=&token=` (see [Ask AI](ask_ai.md)).
