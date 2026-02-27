"""Unit tests for utils/person_identity.py."""

import json

import pytest

from videoannotator.utils.person_identity import (
    PERSON_LABELS,
    PersonIdentityManager,
    PersonLabel,
    get_available_labels,
    get_label_aliases,
    normalize_person_label,
)

# ---------------------------------------------------------------------------
# PersonLabel dataclass
# ---------------------------------------------------------------------------


class TestPersonLabel:
    def test_defaults(self):
        pl = PersonLabel(label="infant", confidence=0.9)
        assert pl.method == "manual"
        assert pl.timestamp is None

    def test_custom_fields(self):
        pl = PersonLabel(
            label="parent", confidence=0.8, method="automatic_size_based", timestamp=1.5
        )
        assert pl.method == "automatic_size_based"
        assert pl.timestamp == 1.5


# ---------------------------------------------------------------------------
# PersonIdentityManager — construction
# ---------------------------------------------------------------------------


class TestPersonIdentityManagerInit:
    def test_default_values(self):
        mgr = PersonIdentityManager(video_id="v1")
        assert mgr.video_id == "v1"
        assert mgr.id_format == "semantic"
        assert mgr.next_person_id == 1

    def test_config_overrides_video_id(self):
        mgr = PersonIdentityManager(
            config={"video_id": "from_cfg", "id_format": "integer"}
        )
        assert mgr.video_id == "from_cfg"
        assert mgr.id_format == "integer"

    def test_none_video_id_becomes_default(self):
        mgr = PersonIdentityManager()
        assert mgr.video_id == "default_video"


# ---------------------------------------------------------------------------
# register_track / get_person_id
# ---------------------------------------------------------------------------


class TestTrackRegistration:
    def test_register_semantic(self):
        mgr = PersonIdentityManager(video_id="v1")
        pid = mgr.register_track(1)
        assert pid == "person_v1_001"

    def test_register_integer(self):
        mgr = PersonIdentityManager(video_id="v1", id_format="integer")
        pid = mgr.register_track(1)
        assert pid == "1"

    def test_register_same_track_idempotent(self):
        mgr = PersonIdentityManager(video_id="v1")
        pid1 = mgr.register_track(42)
        pid2 = mgr.register_track(42)
        assert pid1 == pid2

    def test_increments_id(self):
        mgr = PersonIdentityManager(video_id="v1")
        mgr.register_track(1)
        pid2 = mgr.register_track(2)
        assert pid2 == "person_v1_002"

    def test_get_person_id_registered(self):
        mgr = PersonIdentityManager(video_id="v1")
        mgr.register_track(10)
        assert mgr.get_person_id(10) is not None

    def test_get_person_id_unregistered(self):
        mgr = PersonIdentityManager(video_id="v1")
        assert mgr.get_person_id(999) is None


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


class TestLabeling:
    def test_set_and_get_label(self):
        mgr = PersonIdentityManager(video_id="v1")
        mgr.set_person_label("p1", "infant", confidence=0.9, method="manual")
        info = mgr.get_person_label("p1")
        assert info["label"] == "infant"
        assert info["confidence"] == 0.9
        assert info["method"] == "manual"

    def test_get_label_not_set(self):
        mgr = PersonIdentityManager(video_id="v1")
        assert mgr.get_person_label("nonexistent") is None

    def test_overwrite_label(self):
        mgr = PersonIdentityManager(video_id="v1")
        mgr.set_person_label("p1", "infant")
        mgr.set_person_label("p1", "parent")
        assert mgr.get_person_label("p1")["label"] == "parent"


# ---------------------------------------------------------------------------
# get_all_person_ids / get_person_summary
# ---------------------------------------------------------------------------


class TestPersonQueries:
    def test_get_all_ids(self):
        mgr = PersonIdentityManager(video_id="v1")
        mgr.register_track(1)
        mgr.register_track(2)
        ids = mgr.get_all_person_ids()
        assert len(ids) == 2

    def test_summary_without_label(self):
        mgr = PersonIdentityManager(video_id="v1")
        pid = mgr.register_track(1)
        summary = mgr.get_person_summary(pid)
        assert summary["person_id"] == pid
        assert summary["track_id"] == 1
        assert summary["video_id"] == "v1"

    def test_summary_with_label(self):
        mgr = PersonIdentityManager(video_id="v1")
        pid = mgr.register_track(1)
        mgr.set_person_label(pid, "infant", confidence=0.85)
        summary = mgr.get_person_summary(pid)
        assert summary["label"] == "infant"

    def test_summary_unknown_person(self):
        mgr = PersonIdentityManager(video_id="v1")
        summary = mgr.get_person_summary("ghost")
        assert summary["person_id"] == "ghost"
        assert summary["track_id"] is None


# ---------------------------------------------------------------------------
# _calculate_iou
# ---------------------------------------------------------------------------


class TestIoU:
    def test_identical_boxes(self):
        mgr = PersonIdentityManager()
        assert mgr._calculate_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)

    def test_no_overlap(self):
        mgr = PersonIdentityManager()
        assert mgr._calculate_iou([0, 0, 5, 5], [10, 10, 5, 5]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        mgr = PersonIdentityManager()
        iou = mgr._calculate_iou([0, 0, 10, 10], [5, 5, 10, 10])
        assert 0.0 < iou < 1.0

    def test_one_inside_other(self):
        mgr = PersonIdentityManager()
        iou = mgr._calculate_iou([0, 0, 10, 10], [2, 2, 3, 3])
        assert iou == pytest.approx(9.0 / 100.0)  # intersection=9, union=100


# ---------------------------------------------------------------------------
# link_face_to_person
# ---------------------------------------------------------------------------


class TestLinkFaceToPerson:
    def test_link_matching_face(self):
        mgr = PersonIdentityManager(video_id="v1")
        annotations = [
            {"category_id": 1, "bbox": [0, 0, 100, 200], "person_id": "p1"},
        ]
        # Face bbox inside person bbox
        result = mgr.link_face_to_person(
            [10, 10, 50, 60], annotations, iou_threshold=0.1
        )
        assert result == "p1"

    def test_no_match_below_threshold(self):
        mgr = PersonIdentityManager(video_id="v1")
        annotations = [
            {"category_id": 1, "bbox": [0, 0, 10, 10], "person_id": "p1"},
        ]
        # Face far away
        result = mgr.link_face_to_person([500, 500, 10, 10], annotations)
        assert result is None

    def test_skips_non_person_category(self):
        mgr = PersonIdentityManager(video_id="v1")
        annotations = [
            {"category_id": 2, "bbox": [0, 0, 100, 200], "person_id": "p1"},
        ]
        result = mgr.link_face_to_person([10, 10, 50, 60], annotations)
        assert result is None

    def test_empty_annotations(self):
        mgr = PersonIdentityManager(video_id="v1")
        result = mgr.link_face_to_person([0, 0, 10, 10], [])
        assert result is None


# ---------------------------------------------------------------------------
# from_person_tracks
# ---------------------------------------------------------------------------


class TestFromPersonTracks:
    def test_basic_reconstruction(self):
        tracks = [
            {"track_id": 1, "person_id": "person_v1_001", "image_id": "v1_frame_0"},
            {"track_id": 2, "person_id": "person_v1_002", "image_id": "v1_frame_0"},
        ]
        mgr = PersonIdentityManager.from_person_tracks(tracks)
        assert mgr.video_id == "v1"
        assert len(mgr.track_to_person_map) == 2
        assert mgr.next_person_id == 3  # 2 + 1

    def test_with_labels(self):
        tracks = [
            {
                "track_id": 1,
                "person_id": "person_v1_001",
                "person_label": "infant",
                "label_confidence": 0.9,
                "labeling_method": "manual",
            },
        ]
        mgr = PersonIdentityManager.from_person_tracks(tracks, video_id="v1")
        label = mgr.get_person_label("person_v1_001")
        assert label["label"] == "infant"

    def test_no_tracks(self):
        mgr = PersonIdentityManager.from_person_tracks([])
        assert mgr.video_id == "unknown"
        assert len(mgr.track_to_person_map) == 0

    def test_extracts_video_id_from_image_id(self):
        tracks = [{"track_id": 1, "person_id": "p1", "image_id": "myvid_frame_42"}]
        mgr = PersonIdentityManager.from_person_tracks(tracks)
        assert mgr.video_id == "myvid"

    def test_fallback_video_id(self):
        tracks = [{"track_id": 1, "person_id": "p1", "image_id": "no_pattern"}]
        mgr = PersonIdentityManager.from_person_tracks(tracks)
        assert mgr.video_id == "unknown"


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "tracks.json")
        mgr = PersonIdentityManager(video_id="v1")
        mgr.register_track(1)
        mgr.register_track(2)
        mgr.set_person_label("person_v1_001", "infant", confidence=0.9, method="manual")
        mgr.save_person_tracks(path)

        loaded = PersonIdentityManager.load_person_tracks(path)
        assert loaded.video_id == "v1"
        assert len(loaded.track_to_person_map) == 2
        label = loaded.get_person_label("person_v1_001")
        assert label["label"] == "infant"
        assert label["confidence"] == 0.9

    def test_save_with_detections_summary(self, tmp_path):
        path = str(tmp_path / "tracks.json")
        mgr = PersonIdentityManager(video_id="v1")
        mgr.register_track(1)
        mgr.save_person_tracks(path, detections_summary={"total": 100})

        with open(path) as f:
            data = json.load(f)
        assert data["labeling_metadata"]["detections_summary"]["total"] == 100


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------


class TestNormalizePersonLabel:
    def test_canonical_label(self):
        assert normalize_person_label("parent") == "parent"
        assert normalize_person_label("infant") == "infant"

    def test_alias_resolves(self):
        assert normalize_person_label("mother") == "parent"
        assert normalize_person_label("baby") == "infant"
        assert normalize_person_label("doctor") == "clinician"

    def test_case_insensitive(self):
        assert normalize_person_label("PARENT") == "parent"
        assert normalize_person_label("Baby") == "infant"

    def test_whitespace_stripped(self):
        assert normalize_person_label("  parent  ") == "parent"

    def test_unknown_returns_none(self):
        assert normalize_person_label("alien") is None


class TestGetAvailableLabels:
    def test_returns_all_canonical(self):
        labels = get_available_labels()
        assert "parent" in labels
        assert "infant" in labels
        assert "clinician" in labels
        assert len(labels) == len(PERSON_LABELS)


class TestGetLabelAliases:
    def test_known_label(self):
        aliases = get_label_aliases("parent")
        assert "mother" in aliases
        assert "father" in aliases

    def test_unknown_label(self):
        assert get_label_aliases("alien") == []
