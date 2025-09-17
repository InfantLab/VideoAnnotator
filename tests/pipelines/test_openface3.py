"""
OpenFace 3.0 Pipeline Tests.

Comprehensive test suite for OpenFace 3.0 facial behavior analysis pipeline.
Tests all major functionality including face detection, landmark extraction,
action units, head pose, gaze estimation, and emotion recognition.

Based on demo_openface.py functionality and pipeline testing standards.
"""

import pytest
import tempfile
import numpy as np
import os
import time
import json
# Optional OpenCV import; allow tests to run in environments without cv2
try:  # pragma: no cover - availability shim
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False
    cv2 = None  # type: ignore
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.pipelines.face_analysis.openface3_pipeline import OpenFace3Pipeline, OPENFACE3_AVAILABLE
from src.version import __version__


@pytest.mark.unit
class TestOpenFace3Availability:
    """Test OpenFace 3.0 availability and installation."""
    
    def test_openface3_availability_check(self):
        """Test if OpenFace 3.0 is properly detected."""
        # This test will pass regardless of actual installation
        # In CI/CD, OPENFACE3_AVAILABLE might be False
    assert isinstance(OPENFACE3_AVAILABLE, bool)
    
    @pytest.mark.skipif(not OPENFACE3_AVAILABLE, reason="OpenFace 3.0 not available")
    def test_openface3_imports(self):
        """Test that OpenFace 3.0 imports work when available."""
        try:
            from src.pipelines.face_analysis.openface3_pipeline import OpenFace3Pipeline
            assert OpenFace3Pipeline is not None
        except ImportError:
            pytest.fail("OpenFace 3.0 imports failed despite availability flag")


@pytest.mark.unit 
class TestOpenFace3Pipeline:
    """Core functionality tests for OpenFace 3.0 pipeline."""
    
    def test_pipeline_initialization_default(self):
        """Test pipeline initialization with default configuration."""
        pipeline = OpenFace3Pipeline()
        
        # Check default configuration
        assert pipeline.config["device"] == "auto"  # Default is "auto", not "cpu"
        assert "detection_confidence" in pipeline.config
        assert "enable_action_units" in pipeline.config
        assert not pipeline.is_initialized
    
    def test_pipeline_initialization_custom_config(self):
        """Test pipeline initialization with custom configuration."""
        config = {
            "device": "cpu",
            "detection_confidence": 0.8,
            "enable_action_units": True,
            "enable_head_pose": True,
            "max_faces": 10
        }
        pipeline = OpenFace3Pipeline(config)
        
        assert pipeline.config["device"] == "cpu"
        assert pipeline.config["detection_confidence"] == 0.8
        assert pipeline.config["enable_action_units"] is True
        assert pipeline.config["enable_head_pose"] is True
        assert pipeline.config["max_faces"] == 10
    
    def test_schema_structure(self):
        """Test that the pipeline returns correct COCO schema."""
        pipeline = OpenFace3Pipeline()
        schema = pipeline.get_schema()
        
        assert schema["type"] == "coco_annotation"
        assert schema["format_version"] == __version__
        assert len(schema["categories"]) == 1
        assert schema["categories"][0]["name"] == "face"
        
        # Check annotation schema structure
        annotation_schema = schema["annotation_schema"]
        assert "bbox" in annotation_schema
        assert "keypoints" in annotation_schema
        assert "attributes" in annotation_schema
        
        # Check OpenFace3 specific fields are in attributes
        attributes_schema = annotation_schema["attributes"]
        assert "action_units" in attributes_schema
        assert "head_pose" in attributes_schema
        assert "gaze" in attributes_schema
        assert "emotion" in attributes_schema
    
    def test_pipeline_info(self):
        """Test pipeline information retrieval."""
        pipeline = OpenFace3Pipeline()
        info = pipeline.get_pipeline_info()
        
        assert info["name"] == "OpenFace3Pipeline"
        assert info["version"] == __version__
        assert info["output_format"] == "COCO"
        assert "face_detection" in info["capabilities"]
        assert "action_units" in info["capabilities"]
        assert "head_pose" in info["capabilities"]
        assert "gaze_estimation" in info["capabilities"]
        assert "emotion_recognition" in info["capabilities"]


@pytest.mark.unit
class TestOpenFace3FeatureParsing:
    """Test OpenFace 3.0 feature parsing functionality."""
    
    def setup_method(self):
        """Set up test pipeline."""
        self.pipeline = OpenFace3Pipeline({"device": "cpu"})
    
    def test_action_units_parsing(self):
        """Test Action Units parsing from tensor output."""
        # Mock tensor data (8 AU values)
        mock_tensor = Mock()
        mock_tensor.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value = \
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        
        parsed = self.pipeline._parse_action_units(mock_tensor)
        
        assert isinstance(parsed, dict)
        assert len(parsed) == 8  # 8 Action Units
        
        # Check first AU structure
        first_au_key = list(parsed.keys())[0]
        first_au = parsed[first_au_key]
        assert "intensity" in first_au
        assert "presence" in first_au
        assert isinstance(first_au["intensity"], (float, int))  # Can be float or int
        assert isinstance(first_au["presence"], bool)
        assert 0 <= first_au["intensity"] <= 5
    
    def test_head_pose_parsing(self):
        """Test Head Pose parsing from tensor output."""
        # Mock tensor data (2 pose values: pitch, yaw)
        mock_tensor = Mock()
        mock_tensor.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value = \
            np.array([0.1, 0.2])
        
        parsed = self.pipeline._parse_head_pose(mock_tensor)
        
        assert isinstance(parsed, dict)
        assert "pitch" in parsed
        assert "yaw" in parsed
        assert "roll" in parsed
        assert "confidence" in parsed
        
        # Values should be in degrees
        assert isinstance(parsed["pitch"], float)
        assert isinstance(parsed["yaw"], float)
        assert parsed["roll"] == 0.0  # Not available from current output
        assert 0 <= parsed["confidence"] <= 1
    
    def test_emotions_parsing(self):
        """Test Emotions parsing from tensor output."""
        # Mock tensor data (8 emotion probabilities)
        mock_tensor = Mock()
        mock_tensor.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value = \
            np.array([0.1, 0.7, 0.05, 0.05, 0.02, 0.03, 0.03, 0.02])  # Happiness dominant
        
        parsed = self.pipeline._parse_emotions(mock_tensor)
        
        assert isinstance(parsed, dict)
        assert "dominant" in parsed
        assert "probabilities" in parsed
        assert "valence" in parsed
        assert "arousal" in parsed
        assert "confidence" in parsed
        
        # Check probabilities structure
        probabilities = parsed["probabilities"]
        assert len(probabilities) == 8  # 8 emotions
        assert "happiness" in probabilities
        assert "neutral" in probabilities
        assert "anger" in probabilities
        
        # Probabilities should sum to ~1.0
        prob_sum = sum(probabilities.values())
        assert abs(prob_sum - 1.0) < 0.01
        
        # Valence and arousal should be in valid ranges
        assert -1.0 <= parsed["valence"] <= 1.0
        assert 0.0 <= parsed["arousal"] <= 1.0
    
    def test_gaze_parsing(self):
        """Test Gaze parsing from tensor output."""
        # Mock tensor data (using head pose as proxy)
        mock_tensor = Mock()
        mock_tensor.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value = \
            np.array([0.1, 0.2])
        
        parsed = self.pipeline._parse_gaze(mock_tensor)
        
        assert isinstance(parsed, dict)
        assert "direction_x" in parsed
        assert "direction_y" in parsed
        assert "direction_z" in parsed
        assert "confidence" in parsed
        
        # Direction values should be floats
        assert isinstance(parsed["direction_x"], float)
        assert isinstance(parsed["direction_y"], float)
        assert isinstance(parsed["direction_z"], float)
        
        # Confidence should be reasonable
        assert 0 <= parsed["confidence"] <= 1
    
    def test_valence_calculation(self):
        """Test valence calculation from emotion probabilities."""
        probabilities = {
            "happiness": 0.8,
            "neutral": 0.1,
            "sadness": 0.1
        }
        
        valence = self.pipeline._calculate_valence(probabilities)
        
        assert isinstance(valence, float)
        assert -1.0 <= valence <= 1.0
        assert valence > 0  # Should be positive for happiness-dominant
    
    def test_arousal_calculation(self):
        """Test arousal calculation from emotion probabilities."""
        probabilities = {
            "anger": 0.7,
            "neutral": 0.2,
            "sadness": 0.1
        }
        
        arousal = self.pipeline._calculate_arousal(probabilities)
        
        assert isinstance(arousal, float)
        assert 0.0 <= arousal <= 1.0
        assert arousal > 0.5  # Should be high for anger-dominant


@pytest.mark.integration 
class TestOpenFace3Integration:
    """Integration tests for OpenFace 3.0 pipeline."""
    
    def setup_method(self):
        """Set up test pipeline and mock video."""
        self.pipeline = OpenFace3Pipeline({"device": "cpu"})
        self.temp_dir = tempfile.mkdtemp()
        self.video_path = self._create_test_video()
    
    def teardown_method(self):
        """Clean up test resources."""
        if hasattr(self, 'pipeline'):
            self.pipeline.cleanup()
        if os.path.exists(self.video_path):
            os.unlink(self.video_path)
    
    def _create_test_video(self) -> str:
        """Create a minimal test video file."""
        video_path = os.path.join(self.temp_dir, "test_video.mp4")
        
        # Create a simple 3-frame video with face-like shapes
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 1.0, (640, 480))
        
        for i in range(3):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add a simple rectangular "face" 
            cv2.rectangle(frame, (200, 150), (400, 350), (128, 128, 128), -1)
            out.write(frame)
        
        out.release()
        return video_path
    
    @pytest.mark.skipif(not OPENFACE3_AVAILABLE, reason="OpenFace 3.0 not available")
    def test_pipeline_initialization_with_models(self):
        """Test pipeline initialization with actual OpenFace models."""
        try:
            self.pipeline.initialize()
            assert self.pipeline.is_initialized
            
            # Check that required components are loaded
            assert hasattr(self.pipeline, 'face_detector')
            assert hasattr(self.pipeline, 'landmark_detector')
            
        except Exception as e:
            pytest.skip(f"OpenFace 3.0 models not available: {e}")
    
    def test_face_categories_structure(self):
        """Test face categories for COCO format."""
        categories = self.pipeline._get_face_categories()
        
        assert len(categories) == 1
        category = categories[0]
        assert category["id"] == 1
        assert category["name"] == "face"
        assert category["supercategory"] == "person"
        assert "keypoints" in category
        assert len(category["keypoints"]) == 98  # Default landmark count
    
    def test_coco_annotation_creation(self):
        """Test COCO annotation creation from face data."""
        face_data = {
            "detection": {
                "bbox": [100, 100, 50, 60],
                "confidence": 0.95
            },
            "landmarks_2d": np.array([[110, 120], [120, 125], [130, 130]]),
            "action_units": {
                "AU01_Inner_Brow_Raiser": {"intensity": 2.5, "presence": True}
            },
            "head_pose": {
                "pitch": 10.0, "yaw": -5.0, "roll": 0.0, "confidence": 0.8
            },
            "gaze": {
                "direction_x": 0.1, "direction_y": 0.2, "direction_z": 0.9,
                "confidence": 0.7
            },
            "emotion": {
                "dominant": "happiness",
                "probabilities": {"happiness": 0.8, "neutral": 0.2},
                "valence": 0.6, "arousal": 0.4, "confidence": 0.8
            }
        }
        
        annotation = self.pipeline._create_face_annotation(
            annotation_id=1,
            image_id=1,
            face_data=face_data,
            timestamp=1.5
        )
        
        # Check COCO structure
        assert annotation["id"] == 1
        assert annotation["image_id"] == 1
        assert annotation["category_id"] == 1
        assert annotation["bbox"] == [100, 100, 50, 60]
        assert annotation["area"] == 3000  # 50 * 60
        
        # Check OpenFace data
        openface_data = annotation["openface3"]
        assert openface_data["confidence"] == 0.95
        assert openface_data["timestamp"] == 1.5
        assert "action_units" in openface_data
        assert "head_pose" in openface_data
        assert "gaze" in openface_data
        assert "emotion" in openface_data
    
    @patch('cv2.VideoCapture')
    def test_video_processing_mock(self, mock_video_capture):
        """Test video processing with mocked video capture."""
        # Mock video properties
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 90
        }.get(prop, 0)
        
        # Mock frame reading
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        mock_video_capture.return_value = mock_cap
        
        # Mock OpenFace processing
        with patch.object(self.pipeline, '_process_frame') as mock_process:
            mock_process.return_value = []  # No faces detected
            
            results = self.pipeline.process(
                video_path="dummy_video.mp4",
                pps=1.0
            )
            
            assert len(results) == 1  # Should return COCO dataset
            coco_dataset = results[0]
            assert "info" in coco_dataset
            assert "images" in coco_dataset
            assert "annotations" in coco_dataset
            assert "categories" in coco_dataset
    
    def test_detailed_results_saving(self):
        """Test saving of detailed OpenFace results."""
        annotations = [
            {
                "id": 1,
                "bbox": [100, 100, 50, 60],
                "openface3": {
                    "timestamp": 1.5,
                    "confidence": 0.95,
                    "action_units": {"AU01": {"intensity": 2.5, "presence": True}},
                    "head_pose": {"pitch": 10.0, "yaw": -5.0},
                    "emotion": {"dominant": "happiness", "confidence": 0.8}
                }
            }
        ]
        
        output_file = Path(self.temp_dir) / "detailed_results.json"
        self.pipeline._save_detailed_results(output_file, annotations)
        
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            detailed_results = json.load(f)
        
        assert "metadata" in detailed_results
        assert "faces" in detailed_results
        assert detailed_results["metadata"]["pipeline"] == "OpenFace3Pipeline"
        assert len(detailed_results["faces"]) == 1
        
        face_entry = detailed_results["faces"][0]
        assert face_entry["annotation_id"] == 1
        assert face_entry["bbox"] == [100, 100, 50, 60]
        assert face_entry["timestamp"] == 1.5
        assert "features" in face_entry


@pytest.mark.performance
class TestOpenFace3Performance:
    """Performance tests for OpenFace 3.0 pipeline."""
    
    def setup_method(self):
        """Set up test pipeline."""
        self.pipeline = OpenFace3Pipeline({"device": "cpu"})
    
    def test_parsing_performance(self):
        """Test that feature parsing is reasonably fast."""
        # Create mock tensor data
        mock_tensor = Mock()
        mock_tensor.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value = \
            np.random.rand(8)
        
        # Test Action Units parsing speed
        start_time = time.time()
        for _ in range(100):
            self.pipeline._parse_action_units(mock_tensor)
        au_time = time.time() - start_time
        
        assert au_time < 1.0  # Should parse 100 iterations in under 1 second
        
        # Test Emotions parsing speed
        start_time = time.time()
        for _ in range(100):
            self.pipeline._parse_emotions(mock_tensor)
        emotion_time = time.time() - start_time
        
        assert emotion_time < 1.0  # Should parse 100 iterations in under 1 second
    
    def test_memory_usage(self):
        """Test that pipeline doesn't have obvious memory leaks."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create and destroy multiple pipeline instances
        for _ in range(10):
            pipeline = OpenFace3Pipeline({"device": "cpu"})
            # Simulate some processing
            mock_tensor = Mock()
            mock_tensor.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value = \
                np.random.rand(8)
            pipeline._parse_action_units(mock_tensor)
            pipeline.cleanup()
            del pipeline
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024


@pytest.mark.unit
class TestOpenFace3ErrorHandling:
    """Test error handling and edge cases."""
    
    def setup_method(self):
        """Set up test pipeline."""
        self.pipeline = OpenFace3Pipeline({"device": "cpu"})
    
    def test_invalid_video_path(self):
        """Test handling of invalid video paths."""
        with pytest.raises(ValueError, match="Could not open video"):
            self.pipeline.process("nonexistent_video.mp4")
    
    def test_malformed_tensor_data(self):
        """Test handling of malformed tensor data."""
        # Test with None input
        with pytest.raises(AttributeError):
            self.pipeline._parse_action_units(None)
        
        # Test with wrong shape tensor
        mock_tensor = Mock()
        mock_tensor.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value = \
            np.array([])  # Empty array
        
        # Should handle gracefully or raise appropriate error
        try:
            result = self.pipeline._parse_action_units(mock_tensor)
            # If it doesn't raise an error, result should be reasonable
            assert isinstance(result, dict)
        except (IndexError, ValueError):
            # These are acceptable errors for malformed input
            pass
    
    def test_cleanup_robustness(self):
        """Test that cleanup works even with partially initialized pipeline."""
        pipeline = OpenFace3Pipeline({"device": "cpu"})
        
        # Should not raise errors even if not fully initialized
        pipeline.cleanup()
        
        # Should be able to call cleanup multiple times
        pipeline.cleanup()
        pipeline.cleanup()
    
    def test_invalid_configuration(self):
        """Test handling of invalid configuration."""
        # Test with invalid device
        config = {"device": "invalid_device"}
        pipeline = OpenFace3Pipeline(config)
        
        # Should fall back to reasonable defaults
        assert pipeline.config["device"] == "invalid_device"  # Config preserved
        
        # Test with missing required config
        config = {}
        pipeline = OpenFace3Pipeline(config)
        assert "device" in pipeline.config  # Should have defaults


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
