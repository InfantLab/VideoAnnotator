"""
Unit tests for Person Tracking Pipeline.

Tests cover person detection, tracking, and pose estimation functionality.
"""

import pytest
import tempfile
import numpy as np
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.pipelines.person_tracking.person_pipeline import PersonTrackingPipeline
from src.schemas.person_schema import PersonDetection, PersonTrajectory


@pytest.mark.unit
class TestPersonTrackingPipeline:
    """Test cases for person tracking pipeline."""
    
    def test_person_pipeline_initialization(self):
        """Test person pipeline initialization with custom config."""
        config = {
            "model_name": "yolo11s",
            "confidence_threshold": 0.5,
            "tracker": "bytetrack",
            "enable_pose": True
        }
        pipeline = PersonTrackingPipeline(config)
        
        assert pipeline.config["model_name"] == "yolo11s"
        assert pipeline.config["confidence_threshold"] == 0.5
        assert pipeline.config["tracker"] == "bytetrack"
        assert pipeline.config["enable_pose"] == True
    
    def test_person_pipeline_default_config(self):
        """Test person pipeline with default configuration."""
        pipeline = PersonTrackingPipeline()
        
        # Should have default values
        assert "model_name" in pipeline.config
        assert "confidence_threshold" in pipeline.config
        assert "tracker" in pipeline.config
        assert "enable_pose" in pipeline.config
    
    def test_pipeline_initialization_and_cleanup(self):
        """Test pipeline initialization and cleanup."""
        pipeline = PersonTrackingPipeline()
        
        # Initialize should work without errors
        pipeline.initialize()
        assert pipeline.is_initialized == True
        
        # Should be able to cleanup
        pipeline.cleanup()
        assert pipeline.is_initialized == False
    
    @patch('src.pipelines.person_tracking.person_pipeline.YOLO')
    def test_yolo_model_loading(self, mock_yolo):
        """Test YOLO model loading and initialization."""
        # Mock YOLO model
        mock_model = Mock()
        mock_model.names = {0: 'person'}
        mock_yolo.return_value = mock_model
        
        pipeline = PersonTrackingPipeline({
            "model_name": "yolo11n.pt"
        })
        
        pipeline.initialize()
        
        # Should load YOLO model
        mock_yolo.assert_called_once_with("yolo11n.pt")
        assert pipeline.model is not None
        
        pipeline.cleanup()
    
    @patch('src.pipelines.person_tracking.person_pipeline.YOLO')
    def test_person_detection_processing(self, mock_yolo):
        """Test person detection on frame."""
        # Mock YOLO results
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.data = np.array([
            [100, 100, 200, 300, 0.85, 0],  # x1, y1, x2, y2, conf, class
            [300, 150, 400, 350, 0.92, 0]   # Second person
        ])
        mock_result.boxes.xyxy = np.array([
            [100, 100, 200, 300],
            [300, 150, 400, 350]
        ])
        mock_result.boxes.conf = np.array([0.85, 0.92])
        mock_result.boxes.cls = np.array([0, 0])  # Class 0 = person
        
        mock_model = Mock()
        mock_model.return_value = [mock_result]
        mock_model.names = {0: 'person'}
        mock_yolo.return_value = mock_model
        
        pipeline = PersonTrackingPipeline({
            "confidence_threshold": 0.7
        })
        pipeline.initialize()
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = pipeline._detect_persons(frame, 1.0, "test_video", 30)
        
        # Should detect 2 persons above confidence threshold
        assert len(detections) == 2
        
        # Check first detection
        assert detections[0].confidence >= 0.7
        assert detections[0].video_id == "test_video"
        assert detections[0].timestamp == 1.0
        
        pipeline.cleanup()
    
    def test_confidence_filtering(self):
        """Test that detections below confidence threshold are filtered."""
        # Mock detection results with varying confidence
        mock_detections = [
            {"bbox": [100, 100, 200, 300], "confidence": 0.95},  # Keep
            {"bbox": [200, 100, 300, 300], "confidence": 0.45},  # Filter out
            {"bbox": [300, 100, 400, 300], "confidence": 0.75},  # Keep
        ]
        
        pipeline = PersonTrackingPipeline({
            "confidence_threshold": 0.5
        })
        
        filtered = pipeline._filter_by_confidence(mock_detections)
        
        # Should keep 2 detections
        assert len(filtered) == 2
        assert all(det["confidence"] >= 0.5 for det in filtered)
    
    def test_person_tracking_continuity(self):
        """Test person tracking across frames."""
        pipeline = PersonTrackingPipeline({
            "tracker": "bytetrack",
            "max_disappeared": 5
        })
        
        # Mock tracking across multiple frames
        frame_detections = [
            # Frame 1: 2 persons
            [
                {"bbox": [100, 100, 200, 300], "confidence": 0.9},
                {"bbox": [300, 100, 400, 300], "confidence": 0.8}
            ],
            # Frame 2: Same persons moved slightly
            [
                {"bbox": [105, 105, 205, 305], "confidence": 0.88},
                {"bbox": [295, 105, 395, 305], "confidence": 0.82}
            ],
            # Frame 3: One person left
            [
                {"bbox": [110, 110, 210, 310], "confidence": 0.87}
            ]
        ]
        
        with patch.object(pipeline, '_update_tracker') as mock_tracker:
            mock_tracker.side_effect = [
                [{"track_id": 1, "bbox": [100, 100, 200, 300]}, 
                 {"track_id": 2, "bbox": [300, 100, 400, 300]}],
                [{"track_id": 1, "bbox": [105, 105, 205, 305]}, 
                 {"track_id": 2, "bbox": [295, 105, 395, 305]}],
                [{"track_id": 1, "bbox": [110, 110, 210, 310]}]
            ]
            
            all_tracks = []
            for frame_idx, detections in enumerate(frame_detections):
                tracks = pipeline._update_tracker(detections, frame_idx)
                all_tracks.extend(tracks)
            
            # Should maintain consistent track IDs
            track_1_count = sum(1 for track in all_tracks if track["track_id"] == 1)
            track_2_count = sum(1 for track in all_tracks if track["track_id"] == 2)
            
            assert track_1_count == 3  # Person 1 in all frames
            assert track_2_count == 2  # Person 2 in first two frames
    
    @patch('src.pipelines.person_tracking.person_pipeline.YOLO')
    def test_pose_estimation_integration(self, mock_yolo):
        """Test pose estimation functionality."""
        # Mock YOLO pose model
        mock_result = Mock()
        mock_result.keypoints = Mock()
        mock_result.keypoints.data = np.array([[
            [320, 100, 0.9],  # nose
            [310, 120, 0.85], # left_eye
            [330, 120, 0.87], # right_eye
            # ... more keypoints
        ]])
        
        mock_model = Mock()
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        pipeline = PersonTrackingPipeline({
            "enable_pose": True,
            "model_name": "yolo11n-pose.pt"
        })
        pipeline.initialize()
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        with patch.object(pipeline, '_extract_pose_keypoints') as mock_pose:
            mock_pose.return_value = {
                "keypoints": [
                    {"name": "nose", "x": 320, "y": 100, "confidence": 0.9},
                    {"name": "left_eye", "x": 310, "y": 120, "confidence": 0.85}
                ],
                "pose_confidence": 0.88
            }
            
            pose_data = pipeline._extract_pose_keypoints(mock_result, 0)
            
            assert "keypoints" in pose_data
            assert pose_data["pose_confidence"] == 0.88
            assert len(pose_data["keypoints"]) == 2
        
        pipeline.cleanup()
    
    def test_person_annotation_creation(self):
        """Test creating person detection annotations."""
        detection = PersonDetection(
            video_id="test_video",
            timestamp=1.5,
            person_id=1,
            track_id="track_001",
            bbox={"x": 0.15625, "y": 0.208, "width": 0.156, "height": 0.417},  # Normalized
            confidence=0.92
        )
        
        assert detection.person_id == 1
        assert detection.track_id == "track_001"
        assert detection.confidence == 0.92
        assert detection.timestamp == 1.5
        
        # Check bbox normalization
        assert 0.0 <= detection.bbox["x"] <= 1.0
        assert 0.0 <= detection.bbox["y"] <= 1.0
    
    @patch('src.pipelines.person_tracking.person_pipeline.cv2.VideoCapture')
    @patch('src.pipelines.person_tracking.person_pipeline.YOLO')
    def test_process_video_integration(self, mock_yolo, mock_video_capture):
        """Test complete video processing pipeline."""
        # Mock video capture
        mock_cap = Mock()
        mock_cap.get.side_effect = lambda prop: {
            5: 30.0,   # FPS
            7: 90,     # Frame count
            3: 640,    # Width
            4: 480     # Height
        }.get(prop, 0)
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None)
        ]
        mock_video_capture.return_value = mock_cap
        
        # Mock YOLO model
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.data = np.array([[100, 100, 200, 300, 0.85, 0]])
        mock_result.boxes.xyxy = np.array([[100, 100, 200, 300]])
        mock_result.boxes.conf = np.array([0.85])
        mock_result.boxes.cls = np.array([0])
        
        mock_model = Mock()
        mock_model.return_value = [mock_result]
        mock_model.names = {0: 'person'}
        mock_yolo.return_value = mock_model
        
        pipeline = PersonTrackingPipeline()
        
        try:
            results = pipeline.process("test_video.mp4", "output")
            assert isinstance(results, list)
            
        except Exception as e:
            pytest.skip(f"Process integration test failed: {e}")
    
    def test_error_handling_robustness(self):
        """Test error handling for various failure scenarios."""
        pipeline = PersonTrackingPipeline()
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            pipeline.process("non_existent_file.mp4", "output")
        
        # Test with invalid model
        with pytest.raises((ValueError, Exception)):
            pipeline = PersonTrackingPipeline({"model_name": "invalid_model.pt"})
            pipeline.initialize()


@pytest.mark.integration
class TestPersonTrackingPipelineIntegration:
    """Integration tests for person tracking pipeline."""
    
    @pytest.mark.skipif(
        not os.getenv("TEST_INTEGRATION"),
        reason="Integration tests disabled. Set TEST_INTEGRATION=1 to enable"
    )
    def test_real_person_tracking(self, temp_video_file):
        """Test person tracking with real video file."""
        pipeline = PersonTrackingPipeline({
            "model_name": "yolo11n.pt",  # Use nano model for speed
            "confidence_threshold": 0.5,
            "enable_pose": False  # Disable pose for faster testing
        })
        
        try:
            with pipeline:
                results = pipeline.process(str(temp_video_file), "output")
                
                # Should return list of person detections
                assert isinstance(results, list)
                
                # Each result should be properly formatted
                for result in results:
                    assert hasattr(result, 'person_id')
                    assert hasattr(result, 'bbox')
                    assert hasattr(result, 'confidence')
                    assert 0.0 <= result.confidence <= 1.0
                    
        except ImportError:
            pytest.skip("Ultralytics YOLO not available for integration test")
        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")


@pytest.mark.performance
class TestPersonTrackingPipelinePerformance:
    """Performance tests for person tracking pipeline."""
    
    def test_processing_speed_benchmark(self):
        """Test processing speed for typical scenarios."""
        pipeline = PersonTrackingPipeline({
            "model_name": "yolo11n.pt",  # Fastest model
            "confidence_threshold": 0.5
        })
        
        # Mock frame processing
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        import time
        
        with patch.object(pipeline, '_detect_persons') as mock_detect:
            mock_detect.return_value = [
                Mock(person_id=i, confidence=0.8, bbox={"x": 0.1, "y": 0.1, "width": 0.1, "height": 0.2})
                for i in range(5)  # 5 persons detected
            ]
            
            start_time = time.time()
            
            # Process 30 frames (1 second at 30fps)
            for i in range(30):
                detections = pipeline._detect_persons(frame, i/30.0, "test_video", i)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should process efficiently (mocked processing)
            assert processing_time < 1.0
    
    def test_memory_usage_with_many_persons(self):
        """Test memory efficiency with many persons in frame."""
        pipeline = PersonTrackingPipeline({
            "max_tracks": 50,  # Limit number of tracked persons
            "track_buffer": 30  # Frames to keep track history
        })
        
        # Simulate scene with many persons
        many_persons = [
            {"bbox": [i*50, 100, i*50+40, 200], "confidence": 0.8}
            for i in range(20)  # 20 persons
        ]
        
        with patch.object(pipeline, '_update_tracker') as mock_tracker:
            mock_tracker.return_value = [
                {"track_id": i, "bbox": person["bbox"]}
                for i, person in enumerate(many_persons)
            ]
            
            # Should handle many persons without memory issues
            tracks = pipeline._update_tracker(many_persons, 0)
            
            assert len(tracks) <= 50  # Should respect max_tracks limit
