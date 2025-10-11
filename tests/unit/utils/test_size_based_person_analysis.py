"""Tests for Size-Based Person Analysis.

Test suite for the simplified size-based person analysis functionality.
"""

import pytest

from src.utils.size_based_person_analysis import (
    SizeBasedPersonAnalyzer,
    print_analysis_results,
    run_size_based_analysis,
)


class TestSizeBasedPersonAnalyzer:
    """Test cases for SizeBasedPersonAnalyzer class."""

    def test_analyzer_initialization(self):
        """Test analyzer initialization with default and custom parameters."""
        # Default parameters
        analyzer = SizeBasedPersonAnalyzer()
        assert analyzer.height_threshold == 0.4
        assert analyzer.confidence == 0.7

        # Custom parameters
        analyzer = SizeBasedPersonAnalyzer(height_threshold=0.5, confidence=0.8)
        assert analyzer.height_threshold == 0.5
        assert analyzer.confidence == 0.8

    def test_group_by_person_id(self):
        """Test grouping annotations by person_id."""
        analyzer = SizeBasedPersonAnalyzer()

        annotations = [
            {"person_id": "person_001", "bbox": [100, 100, 80, 160]},
            {"person_id": "person_001", "bbox": [105, 105, 75, 155]},
            {"person_id": "person_002", "bbox": [200, 150, 50, 80]},
            {"person_id": "person_002", "bbox": [205, 145, 48, 85]},
        ]

        grouped = analyzer._group_by_person_id(annotations)

        assert len(grouped) == 2
        assert "person_001" in grouped
        assert "person_002" in grouped
        assert len(grouped["person_001"]) == 2
        assert len(grouped["person_002"]) == 2

    def test_group_by_person_id_missing_ids(self):
        """Test grouping handles annotations without person_id."""
        analyzer = SizeBasedPersonAnalyzer()

        annotations = [
            {"person_id": "person_001", "bbox": [100, 100, 80, 160]},
            {"bbox": [200, 150, 50, 80]},  # Missing person_id
            {"person_id": None, "bbox": [300, 200, 60, 120]},  # None person_id
        ]

        grouped = analyzer._group_by_person_id(annotations)

        assert len(grouped) == 1
        assert "person_001" in grouped
        assert len(grouped["person_001"]) == 1

    def test_calculate_average_heights(self):
        """Test calculation of average heights."""
        analyzer = SizeBasedPersonAnalyzer()

        person_detections = {
            "person_001": [
                {"bbox": [100, 100, 80, 160]},  # height = 160
                {"bbox": [105, 105, 75, 155]},  # height = 155
            ],
            "person_002": [
                {"bbox": [200, 150, 50, 80]},  # height = 80
                {"bbox": [205, 145, 48, 85]},  # height = 85
            ],
        }

        heights = analyzer._calculate_average_heights(person_detections)

        assert "person_001" in heights
        assert "person_002" in heights
        assert heights["person_001"] == 157.5  # (160 + 155) / 2
        assert heights["person_002"] == 82.5  # (80 + 85) / 2

    def test_calculate_average_heights_invalid_bbox(self):
        """Test handling of invalid bounding boxes."""
        analyzer = SizeBasedPersonAnalyzer()

        person_detections = {
            "person_001": [
                {"bbox": [100, 100, 80, 160]},  # Valid
                {"bbox": [100, 100]},  # Invalid - too short
                {"bbox": []},  # Invalid - empty
            ],
            "person_002": [
                {"bbox": []},  # All invalid
            ],
        }

        heights = analyzer._calculate_average_heights(person_detections)

        assert "person_001" in heights
        assert heights["person_001"] == 160.0  # Only valid bbox used
        assert "person_002" not in heights  # No valid bboxes

    def test_classify_by_size_adult_child(self):
        """Test classification of adult and child based on size."""
        analyzer = SizeBasedPersonAnalyzer(height_threshold=0.4)

        person_heights = {
            "person_001": 160.0,  # Tall - should be adult
            "person_002": 80.0,  # Short - should be child
        }

        labels = analyzer._classify_by_size(person_heights)

        assert len(labels) == 2

        # Check adult classification
        assert labels["person_001"]["label"] == "parent"
        assert labels["person_001"]["confidence"] == 0.7
        assert labels["person_001"]["method"] == "size_based_inference"
        assert "normalized_height" in labels["person_001"]["metadata"]
        assert labels["person_001"]["metadata"]["normalized_height"] == 1.0  # 160/160

        # Check child classification
        assert labels["person_002"]["label"] == "infant"
        assert labels["person_002"]["confidence"] == 0.7
        assert labels["person_002"]["method"] == "size_based_inference"
        assert labels["person_002"]["metadata"]["normalized_height"] == 0.5  # 80/160

    def test_classify_by_size_threshold_boundary(self):
        """Test classification at threshold boundary."""
        analyzer = SizeBasedPersonAnalyzer(height_threshold=0.5)

        person_heights = {
            "person_001": 100.0,  # Will be normalized to 1.0 (>= 0.5 -> adult)
            "person_002": 50.0,  # Will be normalized to 0.5 (>= 0.5 -> adult)
            "person_003": 49.0,  # Will be normalized to 0.49 (< 0.5 -> child)
        }

        labels = analyzer._classify_by_size(person_heights)

        assert labels["person_001"]["label"] == "parent"  # 1.0 >= 0.5
        assert labels["person_002"]["label"] == "parent"  # 0.5 >= 0.5
        assert labels["person_003"]["label"] == "infant"  # 0.49 < 0.5

    def test_classify_by_size_empty_input(self):
        """Test classification with empty input."""
        analyzer = SizeBasedPersonAnalyzer()

        labels = analyzer._classify_by_size({})

        assert len(labels) == 0

    def test_analyze_persons_end_to_end(self):
        """Test complete person analysis workflow."""
        analyzer = SizeBasedPersonAnalyzer(height_threshold=0.4, confidence=0.8)

        annotations = [
            {"person_id": "person_001", "bbox": [100, 100, 80, 160]},  # Adult
            {"person_id": "person_001", "bbox": [105, 105, 75, 155]},
            {"person_id": "person_002", "bbox": [200, 150, 50, 80]},  # Child
            {"person_id": "person_002", "bbox": [205, 145, 48, 85]},
        ]

        labels = analyzer.analyze_persons(annotations)

        assert len(labels) == 2
        assert labels["person_001"]["label"] == "parent"
        assert labels["person_002"]["label"] == "infant"
        assert all(info["confidence"] == 0.8 for info in labels.values())
        assert all(info["method"] == "size_based_inference" for info in labels.values())


class TestConvenienceFunctions:
    """Test convenience functions and utilities."""

    def test_run_size_based_analysis(self):
        """Test the convenience function for running analysis."""
        annotations = [
            {"person_id": "person_001", "bbox": [100, 100, 80, 160]},
            {"person_id": "person_002", "bbox": [200, 150, 50, 80]},
        ]

        labels = run_size_based_analysis(
            annotations, height_threshold=0.5, confidence=0.9
        )

        assert len(labels) == 2
        assert all(info["confidence"] == 0.9 for info in labels.values())
        assert all(
            info["metadata"]["threshold_used"] == 0.5 for info in labels.values()
        )

    def test_print_analysis_results(self, capsys):
        """Test printing of analysis results."""
        labels = {
            "person_001": {
                "label": "parent",
                "confidence": 0.8,
                "reasoning": "Large relative height (0.80 >= 0.4)",
            },
            "person_002": {
                "label": "infant",
                "confidence": 0.7,
                "reasoning": "Small relative height (0.30 < 0.4)",
            },
        }

        print_analysis_results(labels)

        captured = capsys.readouterr()
        assert "Size-Based Person Analysis Results" in captured.out
        assert "Total persons analyzed: 2" in captured.out
        assert "person_001" in captured.out
        assert "person_002" in captured.out
        assert "Label: parent" in captured.out
        assert "Label: infant" in captured.out


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_all_same_height(self):
        """Test when all persons have the same height."""
        analyzer = SizeBasedPersonAnalyzer(height_threshold=0.4)

        person_heights = {
            "person_001": 100.0,
            "person_002": 100.0,
            "person_003": 100.0,
        }

        labels = analyzer._classify_by_size(person_heights)

        # All should be classified as adults since normalized height = 1.0
        assert all(info["label"] == "parent" for info in labels.values())
        assert all(
            info["metadata"]["normalized_height"] == 1.0 for info in labels.values()
        )

    def test_zero_height(self):
        """Test handling of zero heights."""
        analyzer = SizeBasedPersonAnalyzer()

        person_heights = {
            "person_001": 0.0,
            "person_002": 100.0,
        }

        labels = analyzer._classify_by_size(person_heights)

        # Should handle gracefully
        assert len(labels) == 2
        assert labels["person_001"]["metadata"]["normalized_height"] == 0.0
        assert labels["person_002"]["metadata"]["normalized_height"] == 1.0

    def test_single_person(self):
        """Test analysis with only one person."""
        analyzer = SizeBasedPersonAnalyzer(height_threshold=0.4)

        annotations = [
            {"person_id": "person_001", "bbox": [100, 100, 80, 160]},
            {"person_id": "person_001", "bbox": [105, 105, 75, 155]},
        ]

        labels = analyzer.analyze_persons(annotations)

        # Single person should be classified as adult (normalized height = 1.0)
        assert len(labels) == 1
        assert labels["person_001"]["label"] == "parent"
        assert labels["person_001"]["metadata"]["normalized_height"] == 1.0


@pytest.fixture
def sample_data():
    """Fixture providing sample annotation data for tests."""
    return [
        {"person_id": "person_001", "bbox": [100, 100, 80, 160], "frame_number": 1},
        {"person_id": "person_001", "bbox": [105, 105, 75, 155], "frame_number": 2},
        {"person_id": "person_002", "bbox": [200, 150, 50, 80], "frame_number": 1},
        {"person_id": "person_002", "bbox": [205, 145, 48, 85], "frame_number": 2},
        {"person_id": "person_003", "bbox": [300, 120, 85, 170], "frame_number": 1},
    ]


def test_integration_with_sample_data(sample_data):
    """Integration test using sample data fixture."""
    labels = run_size_based_analysis(sample_data)

    assert len(labels) == 3
    assert "person_001" in labels
    assert "person_002" in labels
    assert "person_003" in labels

    # person_002 should be classified as infant (smallest height)
    assert labels["person_002"]["label"] == "infant"

    # person_001 and person_003 should be adults
    assert labels["person_001"]["label"] == "parent"
    assert labels["person_003"]["label"] == "parent"
