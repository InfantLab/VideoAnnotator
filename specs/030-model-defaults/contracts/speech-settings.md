# Contract: speech settings under faster-whisper (FR-014)

Every `speech_recognition` setting keeps its name and meaning. Each maps to an argument of
`faster_whisper.WhisperModel(...)` or `WhisperModel.transcribe(...)` (faster-whisper 1.2.x).

| Setting (metadata) | Default | faster-whisper | Notes |
|---|---|---|---|
| `whisper_model` | `large-v3-turbo` (was `base`) | `WhisperModel(model_size_or_path=…)` | Size names as faster-whisper knows them (`tiny` … `large-v3`, `large-v3-turbo`/`turbo`, `distil-*`), or a CTranslate2 repository id |
| `compute_type` *(new)* | `default` | `WhisperModel(compute_type=…)` | `default`, `float32`, `float16`, `int8`, `int8_float16`, … ; effective type recorded in provenance |
| `use_fp16` | `true` | — | `false` → `compute_type: float32` |
| `device` | `auto` | `WhisperModel(device=…)` | `auto` picks CUDA when torch sees it, as today; CUDA library failure at initialize → CPU with a warning |
| `language` | `null` | `language` | `null` → detect |
| `task` | `transcribe` | `task` | `translate` with a turbo model → error at initialize (research S5) |
| `beam_size` | 5 | `beam_size` | |
| `best_of` | 5 | `best_of` | only used when sampling (temperature > 0), as before |
| `temperature` | 0.0 | `temperature` | passed as a single float: no temperature fallback, as today |
| `patience` | 1.0 | `patience` | |
| `length_penalty` | 1.0 | `length_penalty` | |
| `word_timestamps` | true | `word_timestamps` | |
| `prepend_punctuations` | as today | `prepend_punctuations` | |
| `append_punctuations` | as today | `append_punctuations` | |
| `suppress_tokens` | `null` | `suppress_tokens` | comma string → `list[int]`; `null` → library default `[-1]` |
| `min_segment_duration` | 1.0 | — | applied by our code after transcription, unchanged |
| `sample_rate`, `normalize_audio`, `cache_dir` | | — | audio extraction, unchanged; `cache_dir` no longer sets the speech model's location (models go to the Hugging Face cache inside the models directory) and is documented as such |

Not exposed (faster-whisper defaults, the same as openai-whisper's): `condition_on_previous_text`
true, `compression_ratio_threshold` 2.4, `log_prob_threshold` −1.0, `no_speech_threshold` 0.6,
`vad_filter` false.

Errors: an unknown model name or repository fails `initialize()` with faster-whisper's message;
a model not downloaded on an offline machine fails with the existing "model not downloaded"
readiness message naming the repository.
