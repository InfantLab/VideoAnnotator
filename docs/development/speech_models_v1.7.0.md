# Speech recognition on infant-directed speech: what we tried (v1.7.0)

2026-11-23 to 25, on the five BabyJokes clips chosen for the demo (family FVM73L; see
[handover_babyjokes_dataset.md](handover_babyjokes_dataset.md)), RTX 4060 laptop GPU. UAT #76.

## The problem

Running the six default pipelines on the demo clips, speech recognition wrote "Oh, no" and then
"no" about a hundred times over 17 s of Nom Nom Nom. UAT #66 (2026-10-30) had already met this and
switched on Whisper's temperature fallback, but the loop was still there, and the fallback had
made it worse in another way: **the same clip gave a different transcript on every run.**

## It isn't the audio encoding

The clips are 48 kHz mono AAC. The pipeline extracts 16 kHz mono with ffmpeg and normalises the
peak, which is what Whisper expects. Normalised and raw audio gave the same text for every model.
The two clips with clear speech (Cat, Hat) come out word for word under every model and setting.

## Whisper sizes and settings

Each model on the three hard clips (Peekaboo rep 2, Nom Nom Nom rep 1, Tearing Paper rep 3), once at
temperature 0 and twice with the fallback (0.0 to 1.0 in steps of 0.2), on faster-whisper:

| Model | Peekaboo | Nom Nom Nom | Tearing Paper | With the fallback |
|---|---|---|---|---|
| base (default before v1.6) | clean | "Okay. / Ah." ×7 | "What?" ×8 | the words change ("Bigabo") |
| small | clean | "Thank you very much for watching this video…" (invented) | clean | the same invention |
| medium | clean | "no" ×~340 | "Oh" ×10 | inventions ("RIP / LMAO / Thanks for watching!") |
| large-v3 | clean, every run | "no" ×~180 | "What is this?" ×~12 | Nom Nom Nom empty |
| large-v3-turbo (default) | clean | "no" ×~240 | "Oh!" ×7 | once still looping, once "Thank you." |

- Every size loops on these clips; the bigger ones loop longer, the smaller ones invent YouTube
  phrases. This is Whisper's known behaviour on repetitive or non-speech audio: large-v3 writes
  text for about 40% of non-speech inputs ([Barański et al.](https://arxiv.org/pdf/2609.32560)).
  The old default (base) looked better mostly because its loops are short.
- Whisper large-v3 through `transformers` (greedy) loops the same way, so it isn't the
  faster-whisper runtime.
- Other faster-whisper settings didn't fix it: `hallucination_silence_threshold`, `vad_filter`,
  `repetition_penalty`, `condition_on_previous_text: false`.
- **The fallback samples at random.** One clip came out as "Oh, man" ×4, "Oh, no, no, no!", and
  900 characters of "nom" in three runs. `ctranslate2.set_random_seed` didn't make it repeat on
  the GPU. At temperature 0 every clip gave the same transcript every time.

## Other models

Same five clips, same audio, two runs each, in a separate environment (none of these are
dependencies):

| Clip | Qwen3-ASR 1.7B | Parakeet CTC 1.1B | Whisper large-v3 |
|---|---|---|---|
| Peekaboo | Mickey, peekaboo ×5, ah, peekaboo, ah | "nickyababble" | Mickey! Peek-a-boo! ×5 |
| Cat | right | right | right |
| Hat | right | right | right |
| Nom Nom Nom | **Ah! Ah!** | (nothing) | "Oh, no, no, no…" looping |
| Tearing Paper | What is this? What? **Hahaha.** | "is" | What is this? ×2 |

- **[Qwen3-ASR 1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)** (Apache-2.0; at or near the top of
  the open English leaderboard in late 2026) was the only model with no loop and no invention on
  any clip, and the same text on both runs. "Hahaha" falls on the clip where the parent said the
  baby laughed. Without `language="English"` it wrote Nom Nom Nom in Chinese characters. Against
  it for now:
  - Its timestamps come from a separate model (Qwen3-ForcedAligner 0.6B), and they were wrong
    here: Peekaboo's five "peekaboo"s all between 7 and 8 s (Whisper: 1.7 to 15.8 s), Cat's first
    words at 11.3 s (about 1 s).
  - The `qwen-asr` package pins `transformers==4.57.6` and `accelerate==1.12.0` and pulls in
    gradio, flask and Japanese and Korean tokenisers; we would load the model through
    `transformers` ourselves.
  - 5.9 GB of GPU memory with the aligner; 18 s on the 21 s Nom Nom Nom clip (under 1 s on the
    others); 4.4 GB to download, plus the aligner.
- **Parakeet CTC 1.1B** (NVIDIA) can't loop (it labels frames rather than writing text word by
  word), but drops whatever isn't clear adult speech: no use for sing-song games.
- Not tried: Canary-Qwen 2.5B and Parakeet TDT need NeMo, which the roadmap already flags for what
  it sends over the network; Voxtral Small (24B) doesn't fit an 8 GB GPU.

Background: transcribing child-centred home recordings with Whisper has not given usable results in
several labs ([Kocharov & Räsänen](https://arxiv.org/pdf/2506.11747v1)); models trained on adult
speech don't carry over to child speech ([2026 evaluation](https://arxiv.org/pdf/2605.28833)).

## What changed (2026-11-25)

- `temperature_fallback` is **off by default** again, so a transcript is reproducible.
- New setting `shorten_repeats` (default 8): a word or phrase of up to six words repeated more than
  8 times in a row keeps 8, then `[…]`; so do runs of identical lines, whose last kept line then
  spans the run. The segment is marked `repeats_shortened`. 8, not 4: in Peekaboo the parent
  really says "peekaboo" six times.
- On the demo clips: identical on every run; Nom Nom Nom reads "Oh, no, no, no, no, no, no, no, no,
  […]" (and a "Thank you." Whisper adds at the end); the baseline clip's transcript is unchanged.

## Next (v1.7.x)

- **A second speech pipeline on Qwen3-ASR**: its text, with times from Whisper's word timings or a
  better aligner. First a spike: can vLLM serve it as an OpenAI-compatible
  `/v1/audio/transcriptions` server, so it runs through the model servers (spec 033) with no new
  dependency?
- The model card says plainly that Whisper loops on repetitive child-directed speech and laughter.
- In the tutorial (chapter 9): the same clip through three models, and listening to the clips
  where they disagree.
