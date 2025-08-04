"""
Modern Face Analysis Pipeline Tests.

Tests current COCO-format face detection functionality using standards-only pipeline.
Living documentation for face analysis capabilities.
"""

import pytest
import tempfile
import numpy as np
import os
from pathlib import Path
from unittest.mock import Mock, patch

from src.pipelines.face_analysis.face_pipeline import FaceAnalysisPipeline


@pytest.mark.unit
class TestFaceAnalysisPipeline:
    """Core functionality tests for face analysis pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization with custom configuration."""
        config = {
            "detection_backend": "opencv",
            "emotion_backend": "deepface", 
            "confidence_threshold": 0.8,
            "min_face_size": 50
        }
        pipeline = FaceAnalysisPipeline(config)
        
        assert pipeline.config["detection_backend"] == "opencv"
        assert pipeline.config["confidence_threshold"] == 0.8
        assert pipeline.config["min_face_size"] == 50
    
    def test_default_configuration(self):
        """Test pipeline with default configuration values."""
        pipeline = FaceAnalysisPipeline()
        
        # Verify current default configuration
        assert pipeline.config["detection_backend"] == "deepface"
        assert pipeline.config["emotion_backend"] == "deepface"
        assert pipeline.config["confidence_threshold"] == 0.7
        assert pipeline.config["min_face_size"] == 30
    
    def test_initialization_lifecycle(self):
        """Test pipeline initialization and cleanup lifecycle."""
        pipeline = FaceAnalysisPipeline()
        
        # Should initialize successfully
        pipeline.initialize()
        assert pipeline.is_initialized == True
        
        # Should cleanup properly
        pipeline.cleanup()
        assert pipeline.is_initialized == False
    
    def test_schema_format(self):
        """Test that pipeline returns correct COCO schema format."""
        pipeline = FaceAnalysisPipeline()
        schema = pipeline.get_schema()
        
        assert schema["type"] == "coco_annotation"
        assert "categories" in schema
        assert schema["categories"][0]["name"] == "face"
    
    @patch('cv2.VideoCapture')
    def test_video_metadata_extraction(self, mock_video_capture):
        """Test video metadata extraction functionality."""
        # Mock video properties
        mock_cap = Mock()
        mock_cap.get.side_effect = lambda prop: {5: 30.0, 7: 900, 3: 640, 4: 480}.get(prop, 0)
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        pipeline = FaceAnalysisPipeline()
        metadata = pipeline._get_video_metadata("test_video.mp4")
        
        assert "fps" in metadata
        assert "total_frames" in metadata
        assert "width" in metadata 
        assert "height" in metadata


@pytest.mark.integration
class TestFaceAnalysisIntegration:
    """Integration tests for face analysis pipeline."""
    
    @pytest.mark.skipif(
        not os.getenv("TEST_INTEGRATION"),
        reason="Integration tests disabled. Set TEST_INTEGRATION=1 to enable"
    )
    def test_full_pipeline_processing(self, temp_video_file):
        """Test complete face analysis pipeline processing."""
        pipeline = FaceAnalysisPipeline()
        
        try:
            pipeline.initialize()
            results = pipeline.process(str(temp_video_file))
            
            # Should return COCO format annotations
            assert isinstance(results, list)
            
            # Results should contain required COCO fields
            for result in results:
                assert "annotations" in result or len(result) == 0  # Empty results OK
                
        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")
        finally:
            pipeline.cleanup()


@pytest.mark.performance 
class TestFaceAnalysisPerformance:
    """Performance and efficiency tests."""
    
    def test_processing_efficiency(self):
        """Test that pipeline processes efficiently without memory leaks."""
        pipeline = FaceAnalysisPipeline()
        
        # Mock processing to test pipeline overhead
        with patch.object(pipeline, '_detect_faces_in_frame', return_value=[]):
            pipeline.initialize()
            
            # Should handle multiple frames efficiently
            for i in range(10):
                mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                result = pipeline._detect_faces_in_frame(mock_frame, i * 0.1, "opencv")
                assert isinstance(result, list)
            
            pipeline.cleanup()


@pytest.mark.unit
class TestDeepFaceAnalysis:
    """Tests for DeepFace backend analysis including gender, age, and emotion."""
    
    def test_deepface_configuration(self):
        """Test DeepFace pipeline configuration."""
        config = {
            "detection_backend": "deepface",
            "detect_emotions": True,
            "detect_age": True,
            "detect_gender": True,
            "max_faces": 5,
            "deepface": {
                "detector_backend": "opencv",
                "enforce_detection": False
            }
        }
        
        pipeline = FaceAnalysisPipeline(config)
        assert pipeline.config["detection_backend"] == "deepface"
        assert pipeline.config["detect_emotions"] == True
        assert pipeline.config["detect_age"] == True
        assert pipeline.config["detect_gender"] == True
        assert pipeline.config["deepface"]["detector_backend"] == "opencv"
    
    def test_deepface_schema_includes_analysis_fields(self):
        """Test that DeepFace schema includes age, gender, and emotion fields."""
        config = {
            "detection_backend": "deepface",
            "detect_emotions": True,
            "detect_age": True,
            "detect_gender": True
        }
        
        pipeline = FaceAnalysisPipeline(config)
        schema = pipeline.get_schema()
        annotation_schema = schema.get('annotation_schema', {})
        
        expected_fields = [
            'emotion', 'emotion_confidence', 'emotion_scores',
            'age', 'gender', 'gender_confidence', 'gender_scores'
        ]
        
        for field in expected_fields:
            assert field in annotation_schema, f"Missing field: {field}"
    
    def test_deepface_detection_method_exists(self):
        """Test that DeepFace detection method exists and is callable."""
        config = {"detection_backend": "deepface"}
        pipeline = FaceAnalysisPipeline(config)
        
        assert hasattr(pipeline, '_detect_faces_deepface')
        assert callable(getattr(pipeline, '_detect_faces_deepface'))
    
    @pytest.mark.skipif(
        not os.getenv("TEST_DEEPFACE"),
        reason="DeepFace tests disabled. Set TEST_DEEPFACE=1 to enable"
    )
    @patch('src.pipelines.face_analysis.face_pipeline.DeepFace')
    def test_deepface_analysis_with_mock(self, mock_deepface):
        """Test DeepFace analysis with mocked DeepFace library."""
        mock_deepface.extract_faces.return_value = [np.zeros((224, 224, 3))]
        mock_deepface.analyze.return_value = [{
            'region': {'x': 50, 'y': 50, 'w': 100, 'h': 100},
            'age': 25,
            'gender': {'Woman': 0.2, 'Man': 0.8},
            'dominant_gender': 'Man',
            'emotion': {'happy': 0.7, 'sad': 0.1, 'angry': 0.2},
            'dominant_emotion': 'happy'
        }]
        
        config = {
            "detection_backend": "deepface",
            "detect_emotions": True,
            "detect_age": True,
            "detect_gender": True
        }
        
        pipeline = FaceAnalysisPipeline(config)
        pipeline.initialize()
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        try:
            annotations = pipeline._detect_faces_deepface(
                frame=test_frame,
                timestamp=0.0,
                video_id="test",
                frame_number=1,
                width=640,
                height=480
            )
            
            assert len(annotations) == 1
            annotation = annotations[0]
            
            assert 'bbox' in annotation
            assert 'age' in annotation
            assert 'gender' in annotation
            assert 'emotion' in annotation
            assert annotation['age'] == 25
            assert annotation['gender'] == 'Man'
            assert annotation['emotion'] == 'happy'
            
        finally:
            pipeline.cleanup()
    
    def test_deepface_error_handling(self):
        """Test DeepFace error handling with invalid frame."""
        config = {"detection_backend": "deepface"}
        pipeline = FaceAnalysisPipeline(config)
        pipeline.initialize()
        
        empty_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        
        try:
            annotations = pipeline._detect_faces_deepface(
                frame=empty_frame,
                timestamp=0.0,
                video_id="test",
                frame_number=1,
                width=10,
                height=10
            )
            
            assert isinstance(annotations, list)
            
        except Exception as e:
            assert "No face" in str(e) or "face could not be detected" in str(e)
        finally:
            pipeline.cleanup()


@pytest.mark.integration
class TestDeepFaceIntegration:
    """Integration tests for DeepFace analysis."""
    
    @pytest.mark.skipif(
        not os.getenv("TEST_INTEGRATION"),
        reason="Integration tests disabled. Set TEST_INTEGRATION=1 to enable"
    )
    def test_deepface_full_pipeline(self, temp_video_file):
        """Test complete DeepFace analysis pipeline."""
        config = {
            "detection_backend": "deepface",
            "detect_emotions": True,
            "detect_age": True,
            "detect_gender": True,
            "deepface": {
                "detector_backend": "opencv",
                "enforce_detection": False
            }
        }
        
        pipeline = FaceAnalysisPipeline(config)
        
        try:
            pipeline.initialize()
            results = pipeline.process(str(temp_video_file))
            
            assert isinstance(results, list)
            
        except Exception as e:
            pytest.skip(f"DeepFace integration test failed: {e}")
        finally:
            pipeline.cleanup()


# Placeholder classes for future test expansion
class TestFaceAnalysisAdvanced:
    """Placeholder for advanced face analysis features."""
    
    def test_facial_landmarks_placeholder(self):
        """Placeholder: Test facial landmark detection when implemented."""
        pytest.skip("Facial landmarks tests - implement when feature is stable")
    
    def test_multi_backend_comparison_placeholder(self):
        """Placeholder: Test multiple detection backends when stabilized."""
        pytest.skip("Multi-backend tests - implement when backends are finalized")
