# Quickstart: checking Ask AI

How to see each user story working, on the demo clip and a local Ollama. Automated tests cover
the same ground; this is the walk-through for a person (and for the researchers who try the
sketches).

## Setup

```bash
uv sync --inexact --extra llm
ollama pull qwen3.5:9b            # or any vision model
bash scripts/dev.sh               # API + viewer at http://127.0.0.1:19011
```

## 1. First answer (US1, SC-001)

1. Home → **Ask AI about a video** → drop the demo clip
   (`viewer/demo-assets/2UWdXP.joke1.rep3.take1.Peekaboo_h265.mp4`).
2. Pick the first example question. Press **Try on this frame**.
3. Expect: answer and reasoning beside the video within seconds; a marker at the playhead on the
   timeline; the badge says **Local · qwen3.5:9b on this computer**.
4. Stop Ollama, reload: the window says what's missing and how to start it, not an error code.

Time it from opening Ask AI: under a minute.

## 2. History (US2, SC-002)

1. Make five tries with different wordings. Close the browser. Reopen `/ask`.
2. Expect: all five in History, newest first, with who made them. Restore the third: question,
   model and sampling come back; the current ones are still in the history.
3. Show two entries: two lanes, two table columns, same moments.
4. `/workbench?prompt=hello` lands on Ask AI with "hello" in the question.

## 3. Sampling (US3, SC-003)

1. Set every 5 s, then 2 s: ticks and count change at once (browser dev tools: no request).
2. Set 3 frames, 0.5 s apart; move the playhead near a tick: three thumbnails with their times.
3. Before any try the estimate says it's a rough guess; after one, it uses the measured time.
4. Listed moments `0:12, 0:47, 1:30`: three ticks only.

## 4. Local or remote (US4)

1. Settings → model server → `http://<a public host>:11434` (any host that resolves outside the
   private ranges; it needn't answer).
2. Expect: badge **Remote · frames are sent to <host>** in the warning style; the first try asks
   to confirm; after confirming, a second browser doesn't ask again.
3. `curl -X POST /api/v1/ask/tries` to an unconfirmed remote endpoint → `409
   REMOTE_NOT_CONFIRMED`.

## 5. Run (US5)

1. **Run on whole video** on a 2-minute clip at every 5 s.
2. Expect: answers fill the lane and table while it runs (each within 5 s of the model answering).
3. While it runs, press **Try on this frame**: the try's answer comes after at most one of the
   run's samples (FR-017a).
4. **Stop**: answers so far stay; the run says "Stopped after N of M", not failed; its results
   folder has the COCO file with those answers and provenance with `stopped_early: true`.
5. **Run on more videos**: run setup opens with question and sampling filled in.

## 6. Out (US6)

1. **Export → Table** on a try and on the run; open both in R (`readr::read_csv`): same columns
   as any spec 025 table; distinct answers match the tally.
2. Ask a free-text question ("describe the scene") on 10 moments: the hint to state the allowed
   answers appears.

## Known limitation

A run started by a separate CLI worker process (`videoannotator worker`) doesn't share the
try-first gate (research R5); a try then waits at the model server behind that run's current
and queued calls.
