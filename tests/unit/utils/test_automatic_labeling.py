"""Unit tests for utils/automatic_labeling.py."""

import pytest

from videoannotator.utils.automatic_labeling import (
    AutomaticPersonLabeler,
    calculate_labeling_confidence,
    infer_person_labels_from_tracks,
)

# ---------------------------------------------------------------------------
# Helpers — reusable test data builders
# ---------------------------------------------------------------------------


def _make_detection(person_id, bbox, frame_number=0, timestamp=None):
    """Build a single detection dict."""
    d = {"person_id": person_id, "bbox": bbox, "frame_number": frame_number}
    if timestamp is not None:
        d["timestamp"] = timestamp
    return d


def _make_detections(person_id, bbox, count, start_frame=0):
    """Repeat a detection *count* times with ascending frame numbers."""
    return [
        _make_detection(person_id, bbox, frame_number=start_frame + i)
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# AutomaticPersonLabeler — construction
# ---------------------------------------------------------------------------


class TestLabelerInit:
    def test_default_config(self):
        labeler = AutomaticPersonLabeler()
        cfg = labeler.config
        assert cfg["size_based"]["enabled"] is True
        assert cfg["position_based"]["enabled"] is True
        assert cfg["temporal_consistency"]["enabled"] is True

    def test_custom_config_overrides_defaults(self):
        cfg = {
            "size_based": {
                "enabled": False,
                "height_threshold": 0.5,
                "confidence": 0.9,
                "adult_label": "adult",
                "child_label": "child",
            },
            "position_based": {
                "enabled": False,
                "center_bias_threshold": 0.5,
                "confidence": 0.5,
                "primary_label": "a",
                "secondary_label": "b",
            },
            "temporal_consistency": {
                "enabled": False,
                "min_detections": 5,
                "consistency_threshold": 0.5,
            },
        }
        labeler = AutomaticPersonLabeler(config=cfg)
        assert labeler.config["size_based"]["enabled"] is False


# ---------------------------------------------------------------------------
# _group_detections_by_person
# ---------------------------------------------------------------------------


class TestGroupDetections:
    def test_groups_correctly(self):
        labeler = AutomaticPersonLabeler()
        annotations = [
            _make_detection("p1", [0, 0, 0.5, 0.8]),
            _make_detection("p2", [0, 0, 0.3, 0.4]),
            _make_detection("p1", [0, 0, 0.5, 0.8], frame_number=1),
        ]
        groups = labeler._group_detections_by_person(annotations)
        assert len(groups) == 2
        assert len(groups["p1"]) == 2
        assert len(groups["p2"]) == 1

    def test_skips_missing_person_id(self):
        labeler = AutomaticPersonLabeler()
        annotations = [{"bbox": [0, 0, 1, 1]}]  # no person_id
        groups = labeler._group_detections_by_person(annotations)
        assert groups == {}


# ---------------------------------------------------------------------------
# _apply_size_based_labeling
# ---------------------------------------------------------------------------


class TestSizeBasedLabeling:
    def test_adult_and_child_by_height(self):
        labeler = AutomaticPersonLabeler()
        detections = {
            "adult": [{"bbox": [0, 0, 0.5, 0.9]}],  # tall
            "child": [{"bbox": [0, 0, 0.3, 0.3]}],  # short
        }
        labels = labeler._apply_size_based_labeling(detections)
        assert labels["adult"]["label"] == "parent"
        assert labels["child"]["label"] == "infant"

    def test_single_person_labeled_as_adult(self):
        """One person is always max height -> normalized 1.0 -> adult."""
        labeler = AutomaticPersonLabeler()
        detections = {"solo": [{"bbox": [0, 0, 0.5, 0.5]}]}
        labels = labeler._apply_size_based_labeling(detections)
        assert labels["solo"]["label"] == "parent"

    def test_empty_detections(self):
        labeler = AutomaticPersonLabeler()
        assert labeler._apply_size_based_labeling({}) == {}

    def test_no_bbox(self):
        labeler = AutomaticPersonLabeler()
        detections = {"p1": [{"bbox": []}]}
        labels = labeler._apply_size_based_labeling(detections)
        assert labels == {}

    def test_metadata_contains_height_info(self):
        labeler = AutomaticPersonLabeler()
        detections = {"p1": [{"bbox": [0, 0, 0.5, 0.8]}]}
        labels = labeler._apply_size_based_labeling(detections)
        meta = labels["p1"]["metadata"]
        assert "normalized_height" in meta
        assert "raw_height" in meta
        assert "height_threshold" in meta


# ---------------------------------------------------------------------------
# _apply_position_based_labeling
# ---------------------------------------------------------------------------


class TestPositionBasedLabeling:
    def test_center_person_labeled_primary(self):
        labeler = AutomaticPersonLabeler()
        # Person at center of frame (normalized coords)
        detections = {
            "center": [{"bbox": [0.35, 0.35, 0.3, 0.3]}],  # centre ~0.5, 0.5
            "edge": [{"bbox": [0.0, 0.0, 0.1, 0.1]}],  # top-left edge
        }
        labels = labeler._apply_position_based_labeling(detections)
        assert labels["center"]["label"] == "infant"  # primary
        assert labels["edge"]["label"] == "parent"  # secondary

    def test_absolute_coords_skipped(self):
        labeler = AutomaticPersonLabeler()
        detections = {"p1": [{"bbox": [100, 100, 200, 300]}]}  # absolute
        labels = labeler._apply_position_based_labeling(detections)
        assert labels == {}

    def test_empty_detections(self):
        labeler = AutomaticPersonLabeler()
        assert labeler._apply_position_based_labeling({}) == {}


# ---------------------------------------------------------------------------
# _apply_temporal_consistency
# ---------------------------------------------------------------------------


class TestTemporalConsistency:
    def test_keeps_persons_with_enough_detections(self):
        labeler = AutomaticPersonLabeler()
        # confidence=0.9, 20 detections -> dc=min(1.0, 20/20)=1.0
        # adjusted=0.9*1.0=0.9 >= 0.8 threshold -> kept
        labels = {
            "p1": {
                "label": "parent",
                "confidence": 0.9,
                "method": "size",
                "metadata": {},
            },
        }
        detections = {"p1": _make_detections("p1", [0, 0, 0.5, 0.8], 20)}
        result = labeler._apply_temporal_consistency(labels, detections)
        assert "p1" in result

    def test_filters_persons_with_few_detections(self):
        labeler = AutomaticPersonLabeler()
        labels = {
            "p1": {
                "label": "parent",
                "confidence": 0.7,
                "method": "size",
                "metadata": {},
            },
        }
        detections = {"p1": _make_detections("p1", [0, 0, 0.5, 0.8], 3)}
        result = labeler._apply_temporal_consistency(labels, detections)
        assert "p1" not in result

    def test_filters_low_adjusted_confidence(self):
        """Person has enough detections but adjusted confidence < threshold."""
        labeler = AutomaticPersonLabeler()
        # confidence=0.7, 20 detections -> dc=1.0, adjusted=0.7 < 0.8 -> filtered
        labels = {
            "p1": {
                "label": "parent",
                "confidence": 0.7,
                "method": "size",
                "metadata": {},
            },
        }
        detections = {"p1": _make_detections("p1", [0, 0, 0.5, 0.8], 20)}
        result = labeler._apply_temporal_consistency(labels, detections)
        assert "p1" not in result

    def test_adjusts_confidence(self):
        labeler = AutomaticPersonLabeler()
        # confidence=0.95, 18 detections -> dc=min(1.0, 18/20)=0.9
        # adjusted=0.95*0.9=0.855 >= 0.8 -> kept and confidence changed
        labels = {
            "p1": {
                "label": "parent",
                "confidence": 0.95,
                "method": "size",
                "metadata": {},
            },
        }
        detections = {"p1": _make_detections("p1", [0, 0, 0.5, 0.8], 18)}
        result = labeler._apply_temporal_consistency(labels, detections)
        assert "p1" in result
        assert result["p1"]["confidence"] != 0.95  # was adjusted
        assert result["p1"]["metadata"]["temporal_consistency_applied"] is True


# ---------------------------------------------------------------------------
# label_persons_automatic (full pipeline)
# ---------------------------------------------------------------------------


class TestLabelPersonsAutomatic:
    def test_end_to_end_two_persons(self):
        # Lower consistency_threshold so default size_based confidence (0.7) survives
        cfg = AutomaticPersonLabeler()._default_config()
        cfg["temporal_consistency"]["consistency_threshold"] = 0.5
        labeler = AutomaticPersonLabeler(config=cfg)
        annotations = _make_detections(
            "adult", [0.0, 0.0, 0.5, 0.9], 20
        ) + _make_detections("child", [0.3, 0.3, 0.2, 0.3], 20)
        labels = labeler.label_persons_automatic([], annotations)
        assert len(labels) >= 1

    def test_empty_annotations(self):
        labeler = AutomaticPersonLabeler()
        labels = labeler.label_persons_automatic([], [])
        assert labels == {}

    def test_disabled_methods(self):
        cfg = {
            "size_based": {
                "enabled": False,
                "height_threshold": 0.4,
                "confidence": 0.7,
                "adult_label": "parent",
                "child_label": "infant",
            },
            "position_based": {
                "enabled": False,
                "center_bias_threshold": 0.7,
                "confidence": 0.6,
                "primary_label": "infant",
                "secondary_label": "parent",
            },
            "temporal_consistency": {
                "enabled": False,
                "min_detections": 10,
                "consistency_threshold": 0.8,
            },
        }
        labeler = AutomaticPersonLabeler(config=cfg)
        annotations = _make_detections("p1", [0, 0, 0.5, 0.8], 20)
        labels = labeler.label_persons_automatic([], annotations)
        assert labels == {}


# ---------------------------------------------------------------------------
# analyze_spatial_relationships
# ---------------------------------------------------------------------------


class TestSpatialRelationships:
    def test_single_person_returns_empty_patterns(self):
        labeler = AutomaticPersonLabeler()
        detections = {"p1": _make_detections("p1", [0, 0, 0.5, 0.5], 5)}
        result = labeler.analyze_spatial_relationships(detections)
        assert result["person_count"] == 1
        assert result["spatial_patterns"] == {}

    def test_two_persons_computes_distance(self):
        labeler = AutomaticPersonLabeler()
        detections = {
            "p1": [_make_detection("p1", [0.0, 0.0, 0.1, 0.1], timestamp=0)],
            "p2": [_make_detection("p2", [0.5, 0.5, 0.1, 0.1], timestamp=0)],
        }
        result = labeler.analyze_spatial_relationships(detections)
        assert result["person_count"] == 2
        assert len(result["spatial_patterns"]) == 1

    def test_close_persons_high_interaction(self):
        labeler = AutomaticPersonLabeler()
        detections = {
            "p1": [_make_detection("p1", [0.1, 0.1, 0.1, 0.1], timestamp=0)],
            "p2": [_make_detection("p2", [0.15, 0.15, 0.1, 0.1], timestamp=0)],
        }
        result = labeler.analyze_spatial_relationships(detections)
        pair_key = "p1_p2"
        assert result["spatial_patterns"][pair_key]["interaction_likelihood"] == "high"

    def test_far_persons_low_interaction(self):
        labeler = AutomaticPersonLabeler()
        detections = {
            "p1": [_make_detection("p1", [0.0, 0.0, 0.1, 0.1], timestamp=0)],
            "p2": [_make_detection("p2", [0.8, 0.8, 0.1, 0.1], timestamp=0)],
        }
        result = labeler.analyze_spatial_relationships(detections)
        pair_key = "p1_p2"
        assert result["spatial_patterns"][pair_key]["interaction_likelihood"] == "low"

    def test_temporally_distant_detections_ignored(self):
        labeler = AutomaticPersonLabeler()
        detections = {
            "p1": [_make_detection("p1", [0.0, 0.0, 0.1, 0.1], timestamp=0)],
            "p2": [_make_detection("p2", [0.1, 0.1, 0.1, 0.1], timestamp=10)],
        }
        result = labeler.analyze_spatial_relationships(detections)
        assert result["spatial_patterns"] == {}


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------


class TestInferPersonLabels:
    def test_delegates_to_labeler(self):
        annotations = _make_detections("p1", [0, 0, 0.5, 0.8], 20)
        labels = infer_person_labels_from_tracks([], annotations)
        # Should return without error; actual labeling tested above
        assert isinstance(labels, dict)


class TestCalculateLabelingConfidence:
    def test_high_detection_count(self):
        detections = {"p1": [{"frame_number": i} for i in range(60)]}
        scores = calculate_labeling_confidence(detections)
        assert scores["p1"] >= 0.9

    def test_medium_detection_count(self):
        detections = {"p1": [{"frame_number": i} for i in range(25)]}
        scores = calculate_labeling_confidence(detections)
        assert 0.7 < scores["p1"] <= 1.0

    def test_low_detection_count(self):
        detections = {"p1": [{"frame_number": 0}, {"frame_number": 1}]}
        scores = calculate_labeling_confidence(detections)
        assert scores["p1"] <= 0.6

    def test_good_temporal_coverage_boosts(self):
        detections = {
            "p1": [{"timestamp": 0}, {"timestamp": 40}],  # 40s span > 30
        }
        scores = calculate_labeling_confidence(detections)
        # Low count (2) base = 0.5, but temporal boost * 1.1 = 0.55
        assert scores["p1"] == pytest.approx(0.55)

    def test_limited_temporal_coverage_penalizes(self):
        detections = {
            "p1": [{"timestamp": 0}, {"timestamp": 1}],  # 1s span < 5
        }
        scores = calculate_labeling_confidence(detections)
        assert scores["p1"] == pytest.approx(0.45)

    def test_empty_detections(self):
        assert calculate_labeling_confidence({}) == {}
