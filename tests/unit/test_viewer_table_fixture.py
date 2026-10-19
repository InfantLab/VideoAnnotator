"""The rows the viewer's twin of the table builder must produce (spec 028 R5).

`viewer/src/lib/table/rows.ts` turns the same demo-clip files into rows in the
browser; its Vitest test compares them with `expected_table.csv`, which this
test writes from the real exporter. A change on this side fails here until the
fixture is regenerated (`UPDATE_FIXTURES=1`), and then fails the viewer's test
until the twin follows.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

from videoannotator.exporters import table
from videoannotator.exporters.results_set import ResultsSet, _collect

FIXTURES = Path(__file__).parent.parent / "fixtures" / "viewer_contract"
EXPECTED = FIXTURES / "expected_table.csv"


def _table() -> str:
    video = _collect(FIXTURES, "", "demo_clip")
    video.skipped.clear()  # README.md and the fixture itself aren't results
    out = io.StringIO(newline="")
    table.write_csv(ResultsSet(videos=[video]), out)
    return out.getvalue()


def test_expected_table_matches_the_exporter():
    written = _table()
    if os.environ.get("UPDATE_FIXTURES"):
        EXPECTED.write_text(written, encoding="utf-8", newline="")
    assert EXPECTED.read_text(encoding="utf-8") == written, (
        "The exporter's rows for tests/fixtures/viewer_contract changed: rerun with "
        "UPDATE_FIXTURES=1, then make viewer/src/lib/table/rows.ts match"
    )


def test_fixture_covers_every_reader():
    pipelines = {line.split(",")[2] for line in _table().splitlines()[1:]}
    assert pipelines == {
        "face_analysis",
        "face_openface3_embedding",
        "person_tracking",
        "scene_detection",
        "speaker_diarization",
        "speech_recognition",
        "vlm_annotation",
    }
