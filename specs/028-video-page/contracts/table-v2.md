# Contract: Results Table, format version 2

Extends [spec 025's contract](../../025-results-out/contracts/table.md) (version 1). Everything
in version 1 holds; version 2 only adds.

## Added column (last)

| Column | Type | Meaning |
|---|---|---|
| `author` | text | who wrote a researcher note; empty on every pipeline row |

## Added measure

| Pipeline | `measure` | `label` | `value` | other |
|---|---|---|---|---|
| `notes` (not a pipeline: researcher notes, spec 028) | `note` | the note's text | | `start`; `end` for a span, empty for a moment; `author`; `track`, `frame`, `model`, `file` empty |

- Notes belong to the video, not a run, so `run` is **empty** on note rows and a video's notes
  appear **once per export**, however many runs of it the export covers; `video` is the
  video's file stem. They sort with the other rows by the version 1 order (empty `run` first).
- `event` numbers notes within (`video`, `notes`) in row order.
- `videoannotator_version` on note rows is the version that wrote the export.

## Readers

- Selecting columns by name: unaffected.
- Selecting by position: `author` is new at the end; columns 1–19 are unchanged.
- Code that filters by `pipeline` sees a new value, `notes`; code that keeps only named pipelines
  is unaffected.

## The viewer's twin

`viewer/src/lib/table/rows.ts` produces the same rows (all columns but `run`, which the viewer
fills from the run's name) for the same inputs; checked against
`tests/fixtures/viewer_contract/expected_table.csv`, which a Python test regenerates from
`exporters/table.py` and fails on when it differs from the committed copy.
