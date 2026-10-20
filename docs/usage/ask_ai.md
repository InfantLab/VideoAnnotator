# Ask AI

Ask AI puts a question to a vision model about moments of your video, and
shows its answers along the video. You can try a question on one frame in
seconds, change it until the answers look right, and then run it on the whole
video. Everything you try is kept.

Open it from Home (**Ask AI about a video**) or from **Ask AI** in the menu.

## What you need

- A VideoAnnotator server (the viewer on its own can't ask a model).
- A vision model in [Ollama](https://ollama.com/download), running beside the
  server, for example:

  ```bash
  ollama pull qwen3.5:9b
  ```

- The `llm` component on the server: `pip install "videoannotator[llm]"`
  (or `uv sync --extra llm`).

If something is missing, Ask AI says what and how to get it. The viewer's
Pipelines page shows what's ready.

## Choosing a video

- **Drop a video** on the page, or choose one. It is uploaded to the server
  once: the same file dropped again isn't stored twice, and a later run reads
  it where it is.
- **Pick a video in a folder shared with the server**, when the server can
  read your folders (see
  [choosing videos on your own computer](../installation/INSTALLATION.md#choosing-videos-on-your-own-computer)).
- **A recent video**: one asked about or run before.

Some browsers can't play every video format (H.265 often fails in Firefox). Ask
AI says so; questions and runs still work, because the server reads the video
itself.

## Trying a question

Write the question on the left, or start from an **Example**. Say what form the
answer should take ("Answer with one word: touching or not_touching."): the
model answers more consistently, and the answers are easier to count.

- **Try on this frame** asks about the frame under the playhead.
- **Try on 10 moments** asks about ten of the sample points (below), spread
  evenly over the video.

The answer, the model's reasoning and the frames it was sent appear beside the
video, and a marker appears on the timeline below it. A try is saved before the
model is asked, so closing the browser loses nothing: it carries on on the
server.

## Sampling

Sampling says which moments a run asks about. The timeline draws them as ticks
as you change the settings, before anything is sent:

```
every 5 s:   |    |    |    |    |    |    |
             0:00 0:05 0:10 0:15 0:20 0:25 0:30

3 frames, 0.5 s apart, at 0:10:
                  . | .
               9.5 10 10.5     one sample = these three frames, sent together
```

- **Every N s**: one sample every N seconds, **from** a start **to** an end
  (empty for the end of the video).
- **Only at these moments**: a list such as `0:12, 0:47, 1:30`, instead of the
  grid.
- **Frames per sample**: 1, or several frames **seconds apart** around each
  sample point, sent together in one question. Near the start or end of the
  video some of them fall outside it; the answer says how many were sent
  ("2 of 3 frames").

Each sample is a separate question: the model doesn't see the other samples or
its earlier answers.

Under the settings, Ask AI shows how many samples there are and how long a run
would take. Until this model has answered something here, that is a rough
guess, and says so; after that, it comes from the model's own recent answers.

## Where the video goes

The badge at the top says where the model runs:

- **Local · qwen3.5:9b on this computer**, or **on <server>** for a model
  server on your local network: frames from your video stay on this computer
  or network.
- **Remote · frames are sent to <server>**: the model server is elsewhere, and
  the frames you ask about are sent there. Ask AI asks before the first time,
  and the answer is remembered for everyone on this VideoAnnotator server.

The server decides which it is, from the address the model server's name
resolves to. Runs record it in their provenance, and the methods paragraph says
whether the model ran on this computer, on the local network or on a remote
server.

## History

Every try and run is in **History**, newest first, starred ones on top, with who
made it and when. History is kept on the server and shared by everyone using
it; **Mine** shows only yours, **This video only** only this video's.

- Choosing an entry puts its question, model and sampling back. If you had
  written a question you hadn't tried, it is kept first, as "not tried".
- Name and star entries to find them again.
- **Show** up to four entries of this video side by side: one track and one
  table column each, aligned by time. A moment tried more than once with
  different answers is marked in the table.
- **Earlier questions** are prompts from before Ask AI, or from the command
  line (`videoannotator prompts`); choose one to try it.

## Running on the whole video

**Run on whole video** starts a run with the current question and sampling.
Its answers fill in the track and table as the model gives them, and you can
keep trying questions meanwhile: a try goes ahead of the run's next sample.

**Stop** keeps the answers so far: the run reads "Stopped after 23 of 40", its
results folder has those answers, and its provenance says it stopped early. A
stopped run is a normal run otherwise: you can export it and open it on the
Video page.

**Run on more videos** opens run setup with the question and sampling filled
in; **Save as workflow** keeps them for later runs.

## Answers out

The panel beside the table counts the distinct answers. Errors and skipped
samples are counted apart: they are not answers. More than ten different
answers usually means the question doesn't say which answers are allowed; Ask
AI suggests saying so.

**Export table** downloads the shown entries' answers as the same table as
[Results out](exporting_results.md): one CSV per entry, zipped when there are
several.

## Limits

- A run started by a separate worker process (`videoannotator worker`), rather
  than by the server itself, doesn't let tries go first: a try then waits at
  the model server behind that run's current and queued questions.
- Frames are never stored in the history: showing an earlier try's frames reads
  them from the video again.
