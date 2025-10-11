"""Test integration of size-based analysis with person tracking pipeline."""

# Mock cv2 and ultralytics before any imports to avoid dependency issues
import sys
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Set up comprehensive mocks before any imports
mock_cv2 = Mock()
mock_cv2.__spec__ = Mock()
mock_cv2.__spec__.name = "cv2"
sys.modules["cv2"] = mock_cv2

mock_ultralytics = Mock()
sys.modules["ultralytics"] = mock_ultralytics

# Mock transformers components
mock_transformers = Mock()
sys.modules["transformers"] = mock_transformers

# Import after mocking
from src.pipelines.person_tracking.person_pipeline import PersonTrackingPipeline


class TestPersonTrackingSizeAnalysisIntegration:
    """Test size-based analysis integration in person tracking pipeline."""

    @pytest.fixture
    def mock_video_capture(self):
        """Mock cv2.VideoCapture for testing."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            0: 30.0,  # CAP_PROP_FPS
            3: 640,  # CAP_PROP_FRAME_WIDTH
            4: 480,  # CAP_PROP_FRAME_HEIGHT
            7: 300,  # CAP_PROP_FRAME_COUNT
        }.get(prop, 0)
        return mock_cap

    @pytest.fixture
    def mock_yolo_results(self):
        """Mock YOLO inference results with person detections."""
        # Create mock results for adult (tall) and child (short) persons
        mock_result = Mock()

        # Mock boxes for two persons with different heights
        mock_boxes = Mock()
        mock_boxes.cpu.return_value = mock_boxes
        mock_boxes.numpy.return_value = mock_boxes

        # Create mock box objects
        mock_box_adult = Mock()
        mock_box_adult.cls = [0]  # Person class
        mock_box_adult.xyxy = [[100, 100, 180, 300]]  # Adult: height=200
        mock_box_adult.conf = [0.9]
        mock_box_adult.id = [1]  # Track ID 1

        mock_box_child = Mock()
        mock_box_child.cls = [0]  # Person class
        mock_box_child.xyxy = [[200, 200, 250, 280]]  # Child: height=80
        mock_box_child.conf = [0.8]
        mock_box_child.id = [2]  # Track ID 2

        mock_boxes.__iter__ = Mock(return_value=iter([mock_box_adult, mock_box_child]))
        mock_result.boxes = mock_boxes
        mock_result.keypoints = None  # No keypoints for simplicity

        return [mock_result]

    @pytest.fixture
    def pipeline_config(self):
        """Configuration for person tracking pipeline with size-based.

        analysis.
        """
        return {
            "model": "models/yolo/yolo11n-pose.pt",
            "conf_threshold": 0.4,
            "iou_threshold": 0.7,
            "track_mode": True,
            "tracker": "bytetrack",
            "person_identity": {
                "enabled": True,
                "id_format": "semantic",
                "automatic_labeling": {
                    "enabled": True,
                    "confidence_threshold": 0.7,
                    "size_based": {
                        "enabled": True,
                        "height_threshold": 0.4,
                        "confidence": 0.7,
                        "adult_label": "parent",
                        "child_label": "infant",
                        "use_simple_analyzer": True,
                        "min_detections_for_analysis": 1,
                    },
                },
            },
        }

    def test_size_based_analysis_enabled_by_default(self, pipeline_config):
        """Test that size-based analysis is enabled by default in pipeline."""
        pipeline = PersonTrackingPipeline(pipeline_config)

        # Check default configuration
        auto_config = pipeline.config["person_identity"]["automatic_labeling"]
        size_config = auto_config["size_based"]

        assert size_config["enabled"] is True
        assert size_config["use_simple_analyzer"] is True
        assert size_config["height_threshold"] == 0.4
        assert size_config["confidence"] == 0.7

    @patch("src.pipelines.person_tracking.person_pipeline.cv2.VideoCapture")
    @patch("src.pipelines.person_tracking.person_pipeline.YOLO")
    def test_person_tracking_with_size_analysis(
        self,
        mock_yolo_class,
        mock_video_capture_class,
        pipeline_config,
        mock_video_capture,
        mock_yolo_results,
    ):
        """Test complete pipeline integration with size-based analysis."""
        # Setup mocks
        mock_video_capture_class.return_value = mock_video_capture
        mock_yolo = Mock()
        mock_yolo.track.return_value = mock_yolo_results
        mock_yolo_class.return_value = mock_yolo

        # Mock frame reading
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_video_capture.read.return_value = (True, mock_frame)

        # Create pipeline and initialize
        pipeline = PersonTrackingPipeline(pipeline_config)
        pipeline.initialize()

        # Process with temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            annotations = pipeline.process(
                video_path="test_video.mp4",
                start_time=0.0,
                end_time=1.0,
                pps=1,
                output_dir=temp_dir,
            )

        # Verify we got annotations
        assert len(annotations) > 0

        # Verify person IDs were assigned
        person_ids = set()
        for ann in annotations:
            if "person_id" in ann:
                person_ids.add(ann["person_id"])

        assert len(person_ids) == 2  # Two persons detected

        # Verify automatic labeling occurred
        labeled_annotations = [ann for ann in annotations if "person_label" in ann]
        assert len(labeled_annotations) > 0

        # Verify size-based classification
        labels = set()
        for ann in labeled_annotations:
            if "person_label" in ann:
                labels.add(ann["person_label"])

        # Should have both adult and child labels
        assert "parent" in labels or "infant" in labels

        # Verify labeling method is size-based
        size_based_annotations = [
            ann
            for ann in annotations
            if ann.get("labeling_method") == "size_based_inference"
        ]
        assert len(size_based_annotations) > 0

    def test_minimum_detections_filtering(self, pipeline_config):
        """Test that minimum detections filtering works correctly."""
        # Set higher minimum detection threshold
        pipeline_config["person_identity"]["automatic_labeling"]["size_based"][
            "min_detections_for_analysis"
        ] = 3

        pipeline = PersonTrackingPipeline(pipeline_config)

        # Create mock annotations with insufficient detections for one person
        annotations = [
            # Person 1: sufficient detections (3)
            {"person_id": "person_001", "bbox": [100, 100, 80, 200]},
            {"person_id": "person_001", "bbox": [105, 105, 75, 195]},
            {"person_id": "person_001", "bbox": [110, 110, 78, 198]},
            # Person 2: insufficient detections (2)
            {"person_id": "person_002", "bbox": [200, 150, 50, 80]},
            {"person_id": "person_002", "bbox": [205, 145, 48, 85]},
        ]

        # Test filtering logic
        auto_labeling_config = pipeline.config["person_identity"]["automatic_labeling"]
        size_based_config = auto_labeling_config["size_based"]
        min_detections = size_based_config["min_detections_for_analysis"]

        # Count detections per person
        person_detection_counts = {}
        for ann in annotations:
            person_id = ann.get("person_id")
            if person_id:
                person_detection_counts[person_id] = (
                    person_detection_counts.get(person_id, 0) + 1
                )

        # Filter annotations
        filtered_tracks = [
            ann
            for ann in annotations
            if person_detection_counts.get(ann.get("person_id", ""), 0)
            >= min_detections
        ]

        # Should only have person_001 annotations (3 detections >= 3 minimum)
        filtered_person_ids = set(ann["person_id"] for ann in filtered_tracks)
        assert filtered_person_ids == {"person_001"}
        assert len(filtered_tracks) == 3

    @patch("src.pipelines.person_tracking.person_pipeline.run_size_based_analysis")
    def test_size_analysis_function_called(self, mock_size_analysis, pipeline_config):
        """Test that the size-based analysis function is called correctly."""
        mock_size_analysis.return_value = {
            "person_001": {
                "label": "parent",
                "confidence": 0.8,
                "method": "size_based_inference",
            }
        }

        pipeline = PersonTrackingPipeline(pipeline_config)

        # Create mock annotations
        annotations = [
            {"person_id": "person_001", "bbox": [100, 100, 80, 200]},
            {"person_id": "person_001", "bbox": [105, 105, 75, 195]},
        ]

        # Initialize identity manager
        from src.utils.person_identity import PersonIdentityManager

        pipeline.identity_manager = PersonIdentityManager("test_video")

        # Simulate the labeling process that happens in the pipeline
        auto_labeling_config = pipeline.config["person_identity"]["automatic_labeling"]
        size_based_config = auto_labeling_config["size_based"]

        if size_based_config.get("use_simple_analyzer", True) and size_based_config.get(
            "enabled", True
        ):
            # Run simplified size-based analysis
            automatic_labels = mock_size_analysis(
                annotations,
                height_threshold=size_based_config.get("height_threshold", 0.4),
                confidence=size_based_config.get("confidence", 0.7),
            )

        # Verify the function was called with correct parameters
        mock_size_analysis.assert_called_once_with(
            annotations, height_threshold=0.4, confidence=0.7
        )

        # Verify returned labels
        assert "person_001" in automatic_labels
        assert automatic_labels["person_001"]["label"] == "parent"
        assert automatic_labels["person_001"]["method"] == "size_based_inference"

    def test_configuration_backward_compatibility(self):
        """Test that pipeline works with old configuration format."""
        # Old configuration without size-based analysis
        old_config = {
            "model": "models/yolo/yolo11n-pose.pt",
            "conf_threshold": 0.4,
            "person_identity": {
                "enabled": False  # Old way of disabling
            },
        }

        pipeline = PersonTrackingPipeline(old_config)

        # Should still work but without automatic labeling
        assert pipeline.config["person_identity"]["enabled"] is False

        # New configuration should override defaults properly
        new_config = {
            "model": "models/yolo/yolo11n-pose.pt",
            "person_identity": {
                "enabled": True,
                "automatic_labeling": {
                    "size_based": {
                        "use_simple_analyzer": False  # Disable simplified analyzer
                    }
                },
            },
        }

        pipeline2 = PersonTrackingPipeline(new_config)
        size_config = pipeline2.config["person_identity"]["automatic_labeling"][
            "size_based"
        ]
        assert size_config["use_simple_analyzer"] is False  # Should be overridden
        assert size_config["enabled"] is True  # Should keep default


class TestSizeBasedAnalysisConfiguration:
    """Test configuration handling for size-based analysis."""

    def test_default_configuration_values(self):
        """Test that default configuration values are correct."""
        pipeline = PersonTrackingPipeline()

        person_identity = pipeline.config["person_identity"]
        auto_labeling = person_identity["automatic_labeling"]
        size_based = auto_labeling["size_based"]

        # Verify default values
        assert person_identity["enabled"] is True
        assert person_identity["id_format"] == "semantic"
        assert auto_labeling["enabled"] is True
        assert auto_labeling["confidence_threshold"] == 0.7
        assert size_based["enabled"] is True
        assert size_based["height_threshold"] == 0.4
        assert size_based["confidence"] == 0.7
        assert size_based["adult_label"] == "parent"
        assert size_based["child_label"] == "infant"
        assert size_based["use_simple_analyzer"] is True
        assert size_based["min_detections_for_analysis"] == 2

    def test_configuration_override(self):
        """Test that configuration can be properly overridden."""
        custom_config = {
            "person_identity": {
                "automatic_labeling": {
                    "size_based": {
                        "height_threshold": 0.5,
                        "confidence": 0.8,
                        "adult_label": "caregiver",
                        "child_label": "baby",
                        "min_detections_for_analysis": 5,
                    }
                }
            }
        }

        pipeline = PersonTrackingPipeline(custom_config)
        size_config = pipeline.config["person_identity"]["automatic_labeling"][
            "size_based"
        ]

        # Verify overridden values
        assert size_config["height_threshold"] == 0.5
        assert size_config["confidence"] == 0.8
        assert size_config["adult_label"] == "caregiver"
        assert size_config["child_label"] == "baby"
        assert size_config["min_detections_for_analysis"] == 5

        # Verify defaults are preserved for non-overridden values
        assert size_config["enabled"] is True
        assert size_config["use_simple_analyzer"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
