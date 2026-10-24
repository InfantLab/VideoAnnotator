"""The VLM sampling functions against the cases the viewer's twin also reads."""

import json
from pathlib import Path

import pytest

from videoannotator.pipelines.vlm_annotation import sampling

CASES = json.loads(
    (
        Path(__file__).parent.parent / "fixtures" / "ask_ai" / "sampling_cases.json"
    ).read_text()
)
IDS = [c["name"] for c in CASES]


def _points(case):
    return sampling.sample_points(
        case["duration"], case["interval"], case["start"], case["end"], case["moments"]
    )


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_sample_points(case):
    assert _points(case) == case["expect"]["points"]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_burst_times(case):
    offsets = sampling.burst_offsets_for(case["frames"])
    spacing = case["spacing"] or case["interval"]
    got = sampling.burst_times(_points(case)[0], offsets, spacing, case["duration"])
    assert [list(x) for x in got] == case["expect"]["burst_times_at_first_point"]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_choose_moments(case):
    assert (
        sampling.choose_moments(_points(case), case["count"])
        == case["expect"]["chosen"]
    )


@pytest.mark.parametrize(
    ("frames", "offsets"),
    [
        (1, [0]),
        (2, [-1, 0]),
        (3, [-1, 0, 1]),
        (4, [-2, -1, 0, 1]),
        (5, [-2, -1, 0, 1, 2]),
    ],
)
def test_burst_offsets_for(frames, offsets):
    assert sampling.burst_offsets_for(frames) == offsets


def test_interval_must_be_positive():
    with pytest.raises(ValueError):
        sampling.sample_points(10.0, 0.0)


CLIP_CASES = [c for c in CASES if "clip" in c]


def test_there_are_clip_cases():
    assert len(CLIP_CASES) >= 4


@pytest.mark.parametrize("case", CLIP_CASES, ids=lambda c: c["name"])
def test_clip_span_and_times(case):
    t = _points(case)[0]
    clip = case["clip"]
    span = sampling.clip_span(t, clip["length"], case["duration"])
    assert list(span) == case["expect"]["clip_span"]
    times = sampling.clip_times(t, clip["length"], clip["fps"], case["duration"])
    assert times == case["expect"]["clip_times_at_first_point"]
