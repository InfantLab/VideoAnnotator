# Quickstart: the demo-clip check for each default change

The check every step of the plan ends with (FR-001, FR-002). It needs a GPU machine with every
extra installed (`uv sync --extra all`), the Hugging Face token saved (spec 031) with the
pyannote licences accepted, and about 2 minutes per run once weights are downloaded.

## 1. Keep a run's outputs

The baseline test already runs the demo clip through a real server. With
`VIDEOANNOTATOR_BASELINE_KEEP` set, it also copies each output (renamed as the fixtures are) into
that folder:

```bash
VIDEOANNOTATOR_BASELINE_KEEP=~/va-030/step0 \
  uv run --no-sync pytest tests/integration/test_output_baseline.py -m real_models -q
```

Failures against the committed fixtures are expected from step 2 on; the kept outputs are what
the check uses. Keep `server.log` too (it is copied alongside): it has the per-pipeline run times.

Note the GPU (`nvidia-smi --query-gpu=name --format=csv`) in each step's notes. Every comparison
is between two runs on the same machine (research B1).

## 2. Compare with the previous step

```bash
uv run --no-sync python scripts/compare_demo_outputs.py ~/va-030/step0 ~/va-030/step2
```

Prints, per file: identical / within the baseline tolerance / different, and for different files
a short summary: speech cues added, removed or reworded, and the largest timing shift; speaker
turns and speaker count; people per frame and track count; faces per frame and which attribute
keys appear. Provenance and creation times are ignored, as in the baseline test.

What each step should show:

| Step | Expected |
|---|---|
| 2 faces | detections and `emotion` identical to step 0; no `age`/`gender`; first run downloads no age/gender weights (check `<models>/deepface/.deepface/weights`) |
| 3 speaker turns | only `.rttm` differs; speaker count stated |
| 4a ultralytics 8.4 | `person_*` within tolerance of step 0 (anything more: stop and investigate) |
| 4b yolo26 | `person_*` differ; 17 keypoints per person; track count stated; repeat with `"iou_threshold": 0.3` (research P2) |
| 5a faster-whisper, `base` | `.vtt` close to step 0; cues reworded and the largest timing shift stated |
| 5b large-v3-turbo | `.vtt` differs; cue count and wording stated; first-run message named the ~1.6 GB download |

Write each comparison's summary into the step's commit message; step 6 collects them into the
CHANGELOG entries (spec "What the researcher sees").

## 3. Checks that need no GPU

```bash
uv run --no-sync pytest tests/unit tests/pipelines tests/api -q   # mocked models
uv run --no-sync pytest tests/unit/test_no_telemetry.py -q        # after 4a and 5a
uv run --no-sync videoannotator pipelines describe speech_recognition   # reason + benchmark shown
uv run --no-sync python scripts/validate_docs_links.py
```

Translate rule (5b): a job with `{"speech_recognition": {"task": "translate"}}` fails with the
message naming `large-v3`; with `"whisper_model": "large-v3"` added, it runs.

Licence (step 3): with a token whose account accepted only community-1, the Speaker turns card is
ready; with one that accepted only 3.1's licences, it names community-1's licence page.

## 4. Re-baseline (step 6)

As `tests/fixtures/viewer_contract/README.md` says: move the current set to `legacy/` only if the
contract test needs it (it already keeps one pre-provenance set), copy the final kept outputs
over the fixtures, record the re-baseline there and in the CHANGELOG, then:

```bash
uv run --no-sync pytest tests/integration/test_output_baseline.py -m real_models -q
cd viewer && bun run test:run
```

Both green, twice in a row on the same machine, before the step 6 commit. If the machine isn't
the RTX 4060 the current fixtures came from, re-measure the tolerances from the two runs.
