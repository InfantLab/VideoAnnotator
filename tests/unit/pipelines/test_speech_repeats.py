"""Whisper's repetition loops are cut, marked, and nothing else changes (UAT #76)."""

import pytest

from videoannotator.pipelines.audio_processing.repeats import MARK, shorten_repeats


def _seg(text, start=0.0, end=1.0):
    return {"start": start, "end": end, "text": text}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Oh," + " no," * 30, "Oh," + " no," * 8 + f" {MARK}"),
        ("Om nom nom nom" + " nom" * 200 + ".", "Om" + " nom" * 8 + f" {MARK}"),
        ("Ha" + " ha" * 40 + " look!", "Ha" + " ha" * 7 + f" {MARK} look!"),
        ("Peek-a-boo! " * 20, "Peek-a-boo! " * 7 + f"Peek-a-boo! {MARK}"),
        (
            "this is my new hat " * 12,
            "this is my new hat " * 7 + f"this is my new hat {MARK}",
        ),
    ],
)
def test_a_loop_keeps_eight_repeats_then_the_mark(text, expected):
    assert shorten_repeats([_seg(text)], 8)[0]["text"] == expected.strip()


@pytest.mark.parametrize(
    "text",
    [
        "Mickey, peek-a-boo, peek-a-boo, peek-a-boo, peek-a-boo, peek-a-boo, peek-a-boo.",
        "What is this? What is this? Oops! Ruff ruff! Ruff ruff! That's not a doggy!",
        "no no no no no no no no",
        "",
    ],
)
def test_real_repetition_is_left_alone(text):
    segment = _seg(text)
    assert shorten_repeats([segment], 8) == [segment]


def test_words_follow_the_text():
    words = [
        {"raw": " no", "word": "no", "start": i * 0.1, "end": i * 0.1 + 0.1}
        for i in range(20)
    ] + [{"raw": " stop", "word": "stop", "start": 2.0, "end": 2.2}]
    out = shorten_repeats(
        [dict(_seg("".join(w["raw"] for w in words)), words=words)], 8
    )
    assert out[0]["text"] == "no" + " no" * 7 + f" {MARK} stop"
    assert [w["word"] for w in out[0]["words"]] == ["no"] * 8 + ["stop"]
    assert out[0]["repeats_shortened"] is True


def test_runs_of_identical_lines_keep_eight_and_the_span():
    segments = [_seg("What is this?", 0.0, 1.0)] + [
        _seg("What?", 1.0 + i, 2.0 + i) for i in range(15)
    ]
    out = shorten_repeats(segments, 8)
    assert [s["text"] for s in out] == ["What is this?"] + ["What?"] * 7 + [
        f"What? {MARK}"
    ]
    assert out[-1]["end"] == 16.0
    assert out[-1]["repeats_shortened"] is True
    assert "repeats_shortened" not in out[1]


def test_lines_differing_only_in_case_and_punctuation_are_one_run():
    segments = [_seg(t) for t in ["Oh!", "oh", "Oh.", "OH"]]
    assert [s["text"] for s in shorten_repeats(segments, 2)] == ["Oh!", f"oh {MARK}"]


def test_zero_keeps_everything():
    segments = [_seg("no " * 50)]
    assert shorten_repeats(segments, 0) is segments
