# Ask AI

Ask AI puts a question to a vision model about moments of your video, and
shows its answers along the video. You can try a question on one frame in
seconds, change it until the answers look right, and then run it on the whole
video. Everything you try is kept.

Open it from Home (**Ask AI about a video**) or from **Ask AI** in the menu.

## What you need

- A VideoAnnotator server (the viewer on its own can't ask a model).
- A vision model on a model server: [Ollama](https://ollama.com/download)
  running beside the server, for example

  ```bash
  ollama pull qwen3.5:9b
  ```

  or a server your lab already runs: llama.cpp's `llama-server`, vLLM, LM
  Studio or SGLang (see [Model servers](#model-servers)).

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

## Model servers

Ask AI uses this server's own Ollama unless you choose another. To add one,
open **Settings → Model servers → Add a model server**:

- **Name**: what you call it ("lab-gpu").
- **Kind**: **Ollama**, or **OpenAI-compatible (llama.cpp, vLLM, LM Studio…)**.
  Ollama's own kind is better for Ollama, because it records each model's
  digest in your results.
- **Address**: as the server documents it. An OpenAI-compatible server's
  address usually ends in `/v1`: `http://127.0.0.1:8080/v1` for `llama-server`,
  `http://10.0.4.12:8000/v1` for vLLM on a lab machine.
- **Key**, if the server needs one: typed here, or the name of an environment
  variable on the server that holds it (which wins when set).
- **Check** says whether it is reachable, the right kind, has models and takes
  the key, and offers a better address or kind when one would work. **Add**
  is available once a check passes (or with **Add anyway**).

Then choose it under **Model server** in Ask AI, above the model. Ask AI
remembers your choice. New run has the same choice for the VLM pipeline's
"Model server address".

Starting a server for a quick test:

```bash
llama-server -hf ggml-org/gemma-3-4b-it-GGUF --port 8080
vllm serve Qwen/Qwen3-VL-8B-Instruct --limit-mm-per-prompt '{"image": 8, "video": 1}' --trust-request-mm-kwargs
```

## Trying a question

Write the question on the left, or start from an **Example**. Say what form the
answer should take ("Answer with one word: touching or not_touching."): the
model answers more consistently, and the answers are easier to count.

- **Try on this frame** asks about the frame under the playhead.
- **Try on 10 moments** asks about ten of the sample points (below), spread
  evenly over the video.

The answer, the model's reasoning and the frames it was sent appear beside the
video, and a marker appears on the timeline below it.

### Answer shape

Under the question, **Answer shape** says what the answer is made of:

- **Free text**: the model answers in words, as before.
- **Fields**: named parts, each **one of a list**, **yes or no**, **a number**
  (with an optional range) or **short text**. For example `touch` (yes or no)
  and `initiator` (one of adult, child, both, none).

With fields, the model server is asked to answer in exactly that shape, and
every answer is checked here too. An answer that doesn't fit is shown as
**Didn't fit**, with the reason and the model's reply; it is counted apart and
drawn in its own mark, never as a value. Each answer's fields become columns
when you download it.

**Suggest** proposes fields: from your question when it lists its answers
("Return ONLY: TOUCH or NO_TOUCH", "(yes/no)", "on a scale of 1 to 5"), else by
asking the model once, with your question's text only and no frames. A
suggestion is shown for you to **use** or **discard**; it never replaces your
fields on its own. A server that can't enforce a shape is noted; its answers are
still checked. A try is saved before the
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
- **A short clip**: a few seconds of video around each sample point (half
  before, half after; shorter at the start and end), sent as video. Offered only
  when the chosen model takes video: today that is vLLM with a Qwen-VL model.
  Ask AI checks once by sending the model a tiny made-up clip, never your video.
  Each answer records the clip's start, end and frame rate.

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

### Remote model servers and data protection

- **Off on a fresh install.** Until an administrator switches them on in
  **Settings → Model servers**, no remote server can be added and no frame is
  sent to one, from Ask AI, New run, the API or the command line. Switching
  them on shows what it means for participants' video.
- **Lockable.** `VIDEOANNOTATOR_REMOTE_MODEL_SERVERS=off` (or `on`) in the
  server's environment fixes the setting; Settings then shows it as set by the
  server's configuration. A lab can cite this in an ethics application.
- **Each remote server is still confirmed** before its first frames, as above.
  Addresses confirmed before this switch existed keep working without it, and
  Settings lists them as confirmed before remote servers needed switching on.
- **A proxy on this computer looks local.** The badge goes by the address frames
  are sent to: a tunnel or proxy on `localhost` that forwards to a hosted API is
  "Local" here, though frames leave the computer. Don't set one up for
  participants' video without the approval a remote server would need.
- **Keys** typed in Settings are kept on the server in
  `model_server_keys.json`, beside its database (readable only by the server's
  user), and never shown again; a key from an environment variable isn't
  stored at all. No key is written to results, provenance, exports, logs or
  the history, and a run's settings can't contain one: a config with
  `api_key` is refused. A key the server refuses stops a run at once with
  "The model server refused the key."

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

**Run on more videos** opens New run with the question and sampling filled
in; **Save as workflow** keeps them for later runs.

## Answers out

The panel beside the table counts the distinct answers. Errors and skipped
samples are counted apart: they are not answers. More than ten different
answers usually means the question doesn't say which answers are allowed; Ask
AI suggests saying so.

**Export table** downloads the shown entries' answers as the same table as
[Results out](exporting_results.md): one CSV per entry, zipped when there are
several. For questions with fields, **Answers, one column per field (CSV)**
downloads one row per moment with a column per field, then `fit` and
`fit_problem`: it opens in R or pandas with no text to parse.

## From a terminal

```bash
videoannotator model-servers check http://127.0.0.1:8080/v1 --kind openai_compatible --json
videoannotator model-servers add workstation http://127.0.0.1:8080/v1 --kind openai_compatible --no-key
videoannotator model-servers add lab-gpu http://10.0.4.12:8000/v1 --key-env LAB_GPU_KEY
videoannotator model-servers list
videoannotator model-servers remove workstation
```

`check` exits 0 when reachable with models, 2 unreachable, 3 the wrong kind, 4
no models, 5 key refused. A key is never a flag: `add` takes `--key-env`, or
asks for it without showing it when run in a terminal.

A run's settings file sets the same things as Ask AI:

```yaml
vlm_annotation:
  backend: openai_compatible
  base_url: http://127.0.0.1:8080/v1
  model: gemma-3-4b-it-Q4_K_M.gguf
  prompt: Is the adult touching the child?
  answer_shape:
    - {name: touch, kind: yes_no}
    - {name: initiator, kind: choice, values: [adult, child, both, none]}
  # sampling_mode: clip, clip_length_sec: 2, clip_fps: 4  (servers that take video)
```

```bash
videoannotator job submit video.mp4 --pipelines vlm_annotation --config shaped.yaml
```

## Limits

- A run started by a separate worker process (`videoannotator worker`), rather
  than by the server itself, doesn't let tries go first: a try then waits at
  the model server behind that run's current and queued questions.
- Frames are never stored in the history: showing an earlier try's frames reads
  them from the video again.
