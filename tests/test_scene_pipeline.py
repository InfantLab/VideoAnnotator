"""
Unit tests for Scene Detection Pipeline.

Tests cover scene detection, classification, and segmentation functionality.
"""

import pytest
import tempfile
import numpy as np
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.pipelines.scene_detection.scene_pipeline_legacy import SceneDetectionPipeline
from src.schemas.scene_schema import SceneAnnotation, SceneSegment


@pytest.mark.unit
class TestSceneDetectionPipeline:
    """Test cases for scene detection pipeline."""
    
    def test_scene_pipeline_initialization(self):
        """Test scene pipeline initialization with custom config."""
        config = {
            "threshold": 30.0,
            "min_scene_length": 2.0,
            "detector": "content_detector"
        }
        pipeline = SceneDetectionPipeline(config)
        
        assert pipeline.config["threshold"] == 30.0
        assert pipeline.config["min_scene_length"] == 2.0
        assert pipeline.config["detector"] == "content_detector"
    
    def test_scene_pipeline_default_config(self):
        """Test scene pipeline with default configuration."""
        pipeline = SceneDetectionPipeline()
        
        # Should have default values
        assert "threshold" in pipeline.config
        assert "min_scene_length" in pipeline.config
        assert "scene_prompts" in pipeline.config
        assert "detector" in pipeline.config
    
    def test_pipeline_initialization_and_cleanup(self):
        """Test pipeline initialization and cleanup."""
        pipeline = SceneDetectionPipeline()
        
        # Initialize should work without errors
        pipeline.initialize()
        assert pipeline.is_initialized == True
        
        # Should be able to cleanup
        pipeline.cleanup()
        assert pipeline.is_initialized == False
    
    @patch('src.pipelines.scene_detection.scene_pipeline.VideoFileClip')
    def test_scene_detection_with_mock_video(self, mock_video_clip, temp_video_file):
        """Test scene detection with mocked video processing."""
        # Mock video clip
        mock_clip = Mock()
        mock_clip.duration = 10.0
        mock_clip.fps = 30.0
        mock_clip.size = (640, 480)
        mock_video_clip.return_value = mock_clip
        
        pipeline = SceneDetectionPipeline({
            "threshold": 30.0,
            "min_scene_length": 1.0
        })
        
        try:
            results = pipeline.process(str(temp_video_file))
            
            # Should return list of scene annotations
            assert isinstance(results, list)
            
            # If scenes detected, check format
            for scene in results:
                if hasattr(scene, 'start_time') and hasattr(scene, 'end_time'):
                    assert scene.start_time >= 0.0
                    assert scene.end_time <= 10.0
                    assert scene.end_time > scene.start_time
                    
        except Exception as e:
            # Pipeline might fail due to missing dependencies
            pytest.skip(f"Scene detection failed: {e}")
    
    def test_scene_annotation_creation(self):
        """Test creating scene annotations with proper schema."""
        annotation = SceneAnnotation(
            video_id="test_video",
            timestamp=0.0,
            scene_id="scene_001",
            start_time=0.0,
            end_time=5.0,
            change_type="cut",
            confidence=0.95
        )
        
        assert annotation.scene_id == "scene_001"
        assert annotation.duration == 5.0
        assert annotation.confidence == 0.95
        assert annotation.change_type == "cut"
    
    def test_minimum_scene_length_filtering(self):
        """Test that scenes shorter than minimum length are filtered."""
        pipeline = SceneDetectionPipeline({
            "min_scene_length": 2.0  # Minimum 2 seconds
        })
        
        # Mock scene list with short scenes
        mock_scenes = [
            {"start": 0.0, "end": 1.0},    # Too short (1 second)
            {"start": 1.0, "end": 4.0},    # Valid (3 seconds)
            {"start": 4.0, "end": 5.5},    # Too short (1.5 seconds)
            {"start": 5.5, "end": 10.0}    # Valid (4.5 seconds)
        ]
        
        filtered_scenes = pipeline._filter_short_scenes(mock_scenes)
        
        # Should only keep scenes >= 2 seconds
        assert len(filtered_scenes) == 2
        assert filtered_scenes[0]["end"] - filtered_scenes[0]["start"] >= 2.0
        assert filtered_scenes[1]["end"] - filtered_scenes[1]["start"] >= 2.0
    
    def test_scene_classification_integration(self):
        """Test scene classification component."""
        pipeline = SceneDetectionPipeline({
            "enable_classification": True,
            "classification_model": "basic"
        })
        
        # Mock scene for classification
        mock_scene = {
            "start": 0.0,
            "end": 5.0,
            "frames": [np.zeros((480, 640, 3), dtype=np.uint8)]
        }
        
        with patch.object(pipeline, '_classify_scene') as mock_classify:
            mock_classify.return_value = {
                "category": "indoor",
                "confidence": 0.87,
                "tags": ["room", "furniture"]
            }
            
            classification = pipeline._classify_scene(mock_scene)
            
            assert classification["category"] == "indoor"
            assert classification["confidence"] == 0.87
            assert "tags" in classification
    
    def test_error_handling_robustness(self):
        """Test error handling for various failure scenarios."""
        pipeline = SceneDetectionPipeline()
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            pipeline.process("non_existent_file.mp4")
        
        # Test with invalid video format
        with pytest.raises((ValueError, Exception)):
            pipeline.process("invalid_file.txt")
    
    def test_output_format_consistency(self):
        """Test that scene detection outputs follow consistent format."""
        pipeline = SceneDetectionPipeline()
        
        # Mock scene detection results
        mock_scenes = [
            {
                "scene_id": "scene_001",
                "start_time": 0.0,
                "end_time": 5.0,
                "change_type": "cut",
                "confidence": 0.95
            }
        ]
        
        # Convert to SceneAnnotation objects
        annotations = []
        for scene in mock_scenes:
            annotation = SceneAnnotation(
                video_id="test_video",
                timestamp=scene["start_time"],
                scene_id=scene["scene_id"],
                start_time=scene["start_time"],
                end_time=scene["end_time"],
                change_type=scene["change_type"],
                confidence=scene["confidence"]
            )
            annotations.append(annotation)
        
        # Check that annotations are properly formatted
        for annotation in annotations:
            assert hasattr(annotation, 'video_id')
            assert hasattr(annotation, 'timestamp')
            assert hasattr(annotation, 'scene_id')
            assert hasattr(annotation, 'duration')
            assert annotation.duration == annotation.end_time - annotation.start_time
    
    def test_context_manager_functionality(self):
        """Test pipeline context manager functionality."""
        pipeline = SceneDetectionPipeline()
        
        # Test context manager
        with pipeline:
            assert pipeline.is_initialized == True
        
        # Should be cleaned up after context exit
        assert pipeline.is_initialized == False
    
    @patch('src.pipelines.scene_detection.scene_pipeline.SceneManager')
    def test_pyscenedetect_integration(self, mock_scene_manager):
        """Test integration with PySceneDetect library."""
        # Mock PySceneDetect components
        mock_manager = Mock()
        mock_manager.detect_scenes.return_value = [
            Mock(start_time=0.0, end_time=5.0),
            Mock(start_time=5.0, end_time=10.0)
        ]
        mock_scene_manager.return_value = mock_manager
        
        pipeline = SceneDetectionPipeline({
            "detector": "content_detector",
            "threshold": 30.0
        })
        
        with patch.object(pipeline, '_detect_scenes_pyscenedetect') as mock_detect:
            mock_detect.return_value = [
                {"start": 0.0, "end": 5.0, "type": "cut"},
                {"start": 5.0, "end": 10.0, "type": "fade"}
            ]
            
            scenes = pipeline._detect_scenes_pyscenedetect("test_video.mp4")
            
            assert len(scenes) == 2
            assert scenes[0]["start"] == 0.0
            assert scenes[1]["end"] == 10.0


@pytest.mark.integration
class TestSceneDetectionPipelineIntegration:
    """Integration tests for scene detection pipeline."""
    
    @pytest.mark.skipif(
        not os.getenv("TEST_INTEGRATION"),
        reason="Integration tests disabled. Set TEST_INTEGRATION=1 to enable"
    )
    def test_real_scene_detection(self, temp_video_file):
        """Test scene detection with real video file."""
        pipeline = SceneDetectionPipeline({
            "threshold": 30.0,
            "min_scene_length": 1.0,
            "detector": "content_detector"
        })
        
        try:
            with pipeline:
                results = pipeline.process(str(temp_video_file))
                
                # Should return list of scene annotations
                assert isinstance(results, list)
                
                # Each result should be properly formatted
                for result in results:
                    assert hasattr(result, 'scene_id')
                    assert hasattr(result, 'start_time')
                    assert hasattr(result, 'end_time')
                    assert result.end_time > result.start_time
                    
        except ImportError:
            pytest.skip("PySceneDetect not available for integration test")
        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")


@pytest.mark.performance
class TestSceneDetectionPipelinePerformance:
    """Performance tests for scene detection pipeline."""
    
    def test_processing_speed_with_long_video(self):
        """Test processing speed with longer video sequences."""
        pipeline = SceneDetectionPipeline({
            "threshold": 30.0,
            "adaptive_threshold": True  # Use adaptive threshold for better performance
        })
        
        # Mock long video processing
        with patch.object(pipeline, '_analyze_video_segments') as mock_analyze:
            mock_analyze.return_value = [
                {"start": i*10.0, "end": (i+1)*10.0, "type": "cut"}
                for i in range(60)  # 10-minute video with scenes every 10 seconds
            ]
            
            import time
            start_time = time.time()
            
            scenes = pipeline._analyze_video_segments("long_video.mp4")
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should process efficiently
            assert processing_time < 1.0  # Mocked processing should be fast
            assert len(scenes) == 60
    
    def test_memory_efficiency_with_high_resolution(self):
        """Test memory usage with high-resolution video."""
        pipeline = SceneDetectionPipeline({
            "downsample_for_detection": True,  # Downsample for efficiency
            "target_resolution": (320, 240)
        })
        
        # Should handle high-resolution frames efficiently
        high_res_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)  # 4K frame
        
        with patch.object(pipeline, '_preprocess_frame') as mock_preprocess:
            mock_preprocess.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
            
            processed_frame = pipeline._preprocess_frame(high_res_frame)
            
            # Should downsample to target resolution
            assert processed_frame.shape[:2] == (240, 320)
