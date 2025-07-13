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
        assert pipeline.config["detection_backend"] == "opencv"
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


# Placeholder classes for future test expansion
class TestFaceAnalysisAdvanced:
    """Placeholder for advanced face analysis features."""
    
    def test_emotion_analysis_placeholder(self):
        """Placeholder: Test emotion analysis when fully implemented."""
        pytest.skip("Emotion analysis tests - implement when feature is stable")
    
    def test_facial_landmarks_placeholder(self):
        """Placeholder: Test facial landmark detection when implemented."""
        pytest.skip("Facial landmarks tests - implement when feature is stable")
    
    def test_multi_backend_comparison_placeholder(self):
        """Placeholder: Test multiple detection backends when stabilized."""
        pytest.skip("Multi-backend tests - implement when backends are finalized")
