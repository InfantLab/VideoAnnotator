"""Researchers' notes in ELAN (spec 028 R10): a tier of their own."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from videoannotator.exporters import elan
from videoannotator.exporters.results_set import ResultsSet, _collect

FIXTURES = Path(__file__).parent.parent.parent / "fixtures" / "viewer_contract"


def _note(start, text, end=None, author="Caspar"):
    return SimpleNamespace(
        start_sec=start,
        end_sec=end,
        text=text,
        author_name=author,
        video_path="/data/demo_clip.mp4",
    )


def _tiers(tmp_path, notes):
    video = _collect(FIXTURES, "", "demo_clip")
    video.skipped.clear()
    video.media_path = "/data/demo_clip.mp4"
    path = tmp_path / "out.eaf"
    summary = elan.write_videos(
        ResultsSet(videos=[video]), lambda _v: path, lambda _v: notes
    )
    root = elan.ET.parse(path).getroot()
    slots = {
        s.get("TIME_SLOT_ID"): int(s.get("TIME_VALUE")) for s in root.iter("TIME_SLOT")
    }
    tiers = {}
    for tier in root.iter("TIER"):
        tiers[tier.get("TIER_ID")] = (
            tier.get("ANNOTATOR"),
            [
                (
                    slots[a.get("TIME_SLOT_REF1")],
                    slots[a.get("TIME_SLOT_REF2")],
                    a.findtext("ANNOTATION_VALUE"),
                )
                for a in tier.iter("ALIGNABLE_ANNOTATION")
            ],
        )
    return tiers, summary


def test_notes_are_one_tier_with_their_author(tmp_path):
    tiers, summary = _tiers(
        tmp_path, [_note(1.5, "looks away"), _note(4.0, "peekaboo", end=6.0)]
    )
    annotator, annotations = tiers["Notes (researcher)"]
    assert annotator == "Caspar"
    assert annotations == [(1500, 1501, "looks away"), (4000, 6000, "peekaboo")]
    assert list(tiers)[-1] == "Notes (researcher)"
    assert summary.researcher_notes == 2


def test_two_authors_get_a_tier_each(tmp_path):
    tiers, _ = _tiers(
        tmp_path, [_note(1.0, "a", author="Caspar"), _note(2.0, "b", author="Kim")]
    )
    assert tiers["Notes (researcher: Caspar)"][0] == "Caspar"
    assert tiers["Notes (researcher: Kim)"][1] == [(2000, 2001, "b")]
    assert "Notes (researcher)" not in tiers


def test_no_notes_no_tier(tmp_path):
    tiers, _ = _tiers(tmp_path, [])
    assert not [t for t in tiers if t.startswith("Notes")]
