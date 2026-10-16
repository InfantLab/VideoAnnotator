# Contract: ELAN files and the methods paragraph

## ELAN (`.eaf`)

- EAF 3.0, valid against `EAFv3.0.xsd`; `AUTHOR="VideoAnnotator <version>"`, `DATE` = export time.
- `HEADER`: `MEDIA_FILE=""`, `TIME_UNITS="milliseconds"`; one `MEDIA_DESCRIPTOR` when the video's
  path is known (`MEDIA_URL` = `file://` URI, `MIME_TYPE` from the extension); a `PROPERTY`
  `videoannotator.table_format_version` = `1`.
- One `LINGUISTIC_TYPE` `videoannotator` (`TIME_ALIGNABLE="true"`, `GRAPHIC_REFERENCES="false"`).
- Tiers, in this order, each `LINGUISTIC_TYPE_REF="videoannotator"`:

| Tier id | From rows | Annotation value |
|---|---|---|
| `<pipeline> scenes` | `scene` | `label` |
| `<pipeline> speech` | `speech` | `label` (cue text) |
| `<pipeline> <speaker>` | `speaker_turn`, one per `track` | `label` |
| `<pipeline> <track> (<label>)` | `person`, one per `track` | `in view` |
| `<pipeline> faces` | `face` without `track` | `in view` |
| `<pipeline> <track>` | `face` with `track` | `in view` |
| `<pipeline> labels` | `vlm_label` | `label` |

- Merging of moment rows into intervals (FR-018, FR-019) as research.md §9. Annotation times are
  rounded to the nearest millisecond; an interval of zero length is widened to 1 ms.
- Tier ids are unique; a collision gets ` 2`, ` 3`.
- Every annotation's start/end equals a table row's `start`/`end` (interval rows) or the merged
  extent of moment rows (by the rule above). A test checks both on the demo clip.

## Methods paragraph

Prose shape for the demo clip (the real text is built from its records):

> Videos were annotated with VideoAnnotator 1.5.0 [@addyman2026videoannotator], running locally.
> Speech was transcribed with Whisper (base; weights sha256 ed3a0b6b1c0e) [@radford2023whisper].
> Speaker turns were identified with pyannote.audio's speaker-diarization-3.1 pipeline (revision
> …) [@plaquet2023powerset; @bredin2023pyannote], with between 1 and 10 speakers. People were
> detected and tracked with Ultralytics YOLO11 pose (yolo11n-pose; sha256 869e83fcdffd) and ByteTrack
> [@jocher2024yolo11; @zhang2022bytetrack], …. Settings not listed were the defaults. GPU
> deterministic mode was off.

Rules:

- One sentence per pipeline: what it does (from metadata `display_name`/a fixed phrase per
  pipeline family), model names with revisions shortened to 12 characters, non-default settings,
  citations.
- After the prose, a list "Model revisions" with each model's full revision and kind.
- Groups (FR-025): first sentence counts ("All 40 videos…" / "Of 40 videos, 31 …; for the other 9,
  …").
- Missing provenance: "<Pipeline> (model version not recorded)". Missing citation: "(no citation
  recorded)". Nothing filled from current defaults.
- BibTeX: VideoAnnotator's entry first, then in order of first citation, de-duplicated.
