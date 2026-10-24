"""A shaped VLM answer in the tidy table and the wide file (spec 033 R10)."""

import io
import json

from videoannotator.exporters.readers import read_vlm
from videoannotator.exporters.table import write_vlm_wide_csv

SHAPE = [
    {"name": "touch", "kind": "yes_no"},
    {"name": "confidence", "kind": "number", "min": 1, "max": 5},
]
FIT = {
    "id": 1, "image_id": "v_frame_000000", "timestamp_sec": 0.0, "frame_number": 0,
    "status": "answered", "label": "yes", "reasoning": '{"touch":"yes","confidence":4}',
    "raw_response": '{"touch":"yes","confidence":4}', "model": "m", "backend": "ollama",
    "fields": {"touch": "yes", "confidence": 4}, "fit": True, "fit_problem": None,
}  # fmt: skip
MISFIT = {
    **FIT, "id": 2, "image_id": "v_frame_000050", "timestamp_sec": 2.0, "frame_number": 50,
    "label": "DIDN'T FIT", "raw_response": "TOUCH", "reasoning": "TOUCH",
    "fields": {}, "fit": False, "fit_problem": "The reply isn't JSON.",
}  # fmt: skip


def _write(tmp_path, annotations):
    path = tmp_path / "v_vlm_annotation.json"
    path.write_text(json.dumps({"images": [], "annotations": annotations}))
    return path


def test_a_measure_per_field_then_fit(tmp_path):
    contents = read_vlm(_write(tmp_path, [FIT, MISFIT]))
    first, second = contents.events
    measures = {m.measure: m for m in first.measures}
    assert measures["vlm_label"].label == "yes"
    assert measures["vlm.touch"].label == "yes"
    assert (measures["vlm.confidence"].value, measures["vlm.confidence"].label) == (
        4,
        None,
    )
    assert measures["vlm_fit"].label == "yes"
    assert "vlm_fit_problem" not in measures
    names = [m.measure for m in second.measures]
    assert "vlm.touch" not in names
    problem = next(m for m in second.measures if m.measure == "vlm_fit_problem")
    assert problem.label == "The reply isn't JSON."


def test_unshaped_answers_have_no_new_rows(tmp_path):
    plain = {k: v for k, v in FIT.items() if k not in ("fields", "fit", "fit_problem")}
    (event,) = read_vlm(_write(tmp_path, [plain])).events
    assert [m.measure for m in event.measures] == ["vlm_label", "vlm_reasoning"]


def test_the_wide_file(tmp_path):
    out = io.StringIO()
    skipped = {**FIT, "status": "skipped"}
    error = {**FIT, "status": "error", "fields": None, "fit": None, "label": "ERROR: x"}
    assert (
        write_vlm_wide_csv([FIT, MISFIT, skipped, error], SHAPE, video="v.mp4", out=out)
        == 3
    )
    lines = out.getvalue().splitlines()
    assert lines[0] == (
        "video,t,frame,touch,confidence,fit,fit_problem,model,backend,raw_response"
    )
    assert lines[1].startswith("v.mp4,0.0,0,yes,4,yes,,m,ollama,")
    assert lines[2].startswith("v.mp4,2.0,50,,,no,The reply isn't JSON.,m,ollama,TOUCH")
    assert lines[3].startswith("v.mp4,0.0,0,,,,,m,ollama,")
