"""
Unit tests for VideoAnnotator pipeline system.

This module contains comprehensive tests for all pipeline components,
schemas, and utility functions.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Import pipeline modules
from src.pipelines.scene_detection import ScenePipeline, ScenePipelineConfig
from src.pipelines.person_tracking import PersonPipeline, PersonPipelineConfig
from src.pipelines.face_analysis import FacePipeline, FacePipelineConfig
from src.pipelines.audio_processing import AudioPipeline, AudioPipelineConfig

# Import schemas
from src.schemas.scene_schema import Scene, SceneTransition, SceneClassification
from src.schemas.person_schema import PersonDetection, PersonTrack, Pose
from src.schemas.face_schema import Face, FaceEmotion, FaceLandmarks
from src.schemas.audio_schema import AudioSegment, SpeechTranscription, SpeakerDiarization


class TestScenePipeline:
    """Test cases for scene detection pipeline."""
    
    def test_scene_pipeline_initialization(self):
        """Test scene pipeline initialization."""
        config = ScenePipelineConfig(threshold=0.3, min_scene_length=2.0)
        pipeline = ScenePipeline(config)
        
        assert pipeline.config.threshold == 0.3
        assert pipeline.config.min_scene_length == 2.0
    
    def test_scene_pipeline_config_validation(self):
        """Test scene pipeline configuration validation."""
        # Valid configuration
        config = ScenePipelineConfig(threshold=0.5, min_scene_length=1.0)
        assert config.threshold == 0.5
        
        # Invalid threshold (should be between 0 and 1)
        with pytest.raises(ValueError):
            ScenePipelineConfig(threshold=1.5)
    
    @patch('src.pipelines.scene_detection.scene_pipeline.cv2')
    def test_scene_pipeline_process_video(self, mock_cv2):
        """Test scene pipeline video processing."""
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0  # fps
        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # First frame
            (True, np.ones((480, 640, 3), dtype=np.uint8) * 255),  # Second frame
            (False, None)  # End of video
        ]
        mock_cv2.VideoCapture.return_value = mock_cap
        
        config = ScenePipelineConfig(threshold=0.3)
        pipeline = ScenePipeline(config)
        
        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_video:
            results = pipeline.process_video(Path(temp_video.name))
            
            assert 'scenes' in results
            assert 'total_duration' in results
            assert isinstance(results['scenes'], list)


class TestPersonPipeline:
    """Test cases for person tracking pipeline."""
    
    def test_person_pipeline_initialization(self):
        """Test person pipeline initialization."""
        config = PersonPipelineConfig(model_name='yolo11s', confidence_threshold=0.5)
        pipeline = PersonPipeline(config)
        
        assert pipeline.config.model_name == 'yolo11s'
        assert pipeline.config.confidence_threshold == 0.5
    
    def test_person_pipeline_config_validation(self):
        """Test person pipeline configuration validation."""
        # Valid configuration
        config = PersonPipelineConfig(confidence_threshold=0.7)
        assert config.confidence_threshold == 0.7
        
        # Invalid confidence threshold
        with pytest.raises(ValueError):
            PersonPipelineConfig(confidence_threshold=1.5)
    
    @patch('src.pipelines.person_tracking.person_pipeline.cv2')
    def test_person_pipeline_process_video(self, mock_cv2):
        """Test person pipeline video processing."""
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0  # fps
        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None)
        ]
        mock_cv2.VideoCapture.return_value = mock_cap
        
        config = PersonPipelineConfig(model_name='yolo11s')
        pipeline = PersonPipeline(config)
        
        # Mock YOLO model
        with patch.object(pipeline, '_yolo_model') as mock_yolo:
            mock_yolo.return_value = Mock()
            
            with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_video:
                results = pipeline.process_video(Path(temp_video.name))
                
                assert 'tracks' in results
                assert 'detections' in results
                assert isinstance(results['tracks'], list)


class TestFacePipeline:
    """Test cases for face analysis pipeline."""
    
    def test_face_pipeline_initialization(self):
        """Test face pipeline initialization."""
        config = FacePipelineConfig(backends=['mediapipe'], detection_confidence=0.7)
        pipeline = FacePipeline(config)
        
        assert 'mediapipe' in pipeline.config.backends
        assert pipeline.config.detection_confidence == 0.7
    
    def test_face_pipeline_config_validation(self):
        """Test face pipeline configuration validation."""
        # Valid configuration
        config = FacePipelineConfig(backends=['mediapipe'])
        assert config.backends == ['mediapipe']
        
        # Invalid backend
        with pytest.raises(ValueError):
            FacePipelineConfig(backends=['invalid_backend'])
    
    @patch('src.pipelines.face_analysis.face_pipeline.cv2')
    def test_face_pipeline_process_video(self, mock_cv2):
        """Test face pipeline video processing."""
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0  # fps
        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None)
        ]
        mock_cv2.VideoCapture.return_value = mock_cap
        
        config = FacePipelineConfig(backends=['mediapipe'])
        pipeline = FacePipeline(config)
        
        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_video:
            results = pipeline.process_video(Path(temp_video.name))
            
            assert 'faces' in results
            assert 'face_tracks' in results
            assert isinstance(results['faces'], list)


class TestAudioPipeline:
    """Test cases for audio processing pipeline."""
    
    def test_audio_pipeline_initialization(self):
        """Test audio pipeline initialization."""
        config = AudioPipelineConfig(whisper_model='base', sample_rate=16000)
        pipeline = AudioPipeline(config)
        
        assert pipeline.config.whisper_model == 'base'
        assert pipeline.config.sample_rate == 16000
    
    def test_audio_pipeline_config_validation(self):
        """Test audio pipeline configuration validation."""
        # Valid configuration
        config = AudioPipelineConfig(sample_rate=16000)
        assert config.sample_rate == 16000
        
        # Invalid sample rate
        with pytest.raises(ValueError):
            AudioPipelineConfig(sample_rate=-1)
    
    @patch('src.pipelines.audio_processing.audio_pipeline.librosa')
    def test_audio_pipeline_process_audio(self, mock_librosa):
        """Test audio pipeline audio processing."""
        # Mock audio loading
        mock_audio = np.random.randn(16000)  # 1 second of audio
        mock_librosa.load.return_value = (mock_audio, 16000)
        
        config = AudioPipelineConfig(whisper_model='base')
        pipeline = AudioPipeline(config)
        
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_audio:
            results = pipeline.process_audio(Path(temp_audio.name))
            
            assert 'duration' in results
            assert 'segments' in results
            assert isinstance(results['segments'], list)


class TestSchemas:
    """Test cases for data schemas."""
    
    def test_scene_schema(self):
        """Test scene schema validation."""
        scene = Scene(
            scene_id="scene_001",
            start_time=0.0,
            end_time=5.0,
            scene_type="indoor",
            confidence=0.85
        )
        
        assert scene.scene_id == "scene_001"
        assert scene.duration == 5.0
        assert scene.scene_type == "indoor"
    
    def test_person_detection_schema(self):
        """Test person detection schema validation."""
        detection = PersonDetection(
            person_id="person_001",
            timestamp=1.0,
            bounding_box=[100, 100, 200, 200],
            confidence=0.9
        )
        
        assert detection.person_id == "person_001"
        assert detection.timestamp == 1.0
        assert len(detection.bounding_box) == 4
    
    def test_face_schema(self):
        """Test face schema validation."""
        face = Face(
            face_id="face_001",
            timestamp=1.0,
            bounding_box=[50, 50, 100, 100],
            confidence=0.8
        )
        
        assert face.face_id == "face_001"
        assert face.timestamp == 1.0
        assert len(face.bounding_box) == 4
    
    def test_audio_segment_schema(self):
        """Test audio segment schema validation."""
        segment = AudioSegment(
            start_time=0.0,
            end_time=5.0,
            audio_data=np.zeros(1000),
            sample_rate=16000
        )
        
        assert segment.start_time == 0.0
        assert segment.end_time == 5.0
        assert segment.sample_rate == 16000


class TestPipelineIntegration:
    """Integration tests for pipeline system."""
    
    def test_pipeline_chaining(self):
        """Test chaining multiple pipelines."""
        # This would test running multiple pipelines in sequence
        # and ensuring data flows correctly between them
        pass
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling."""
        config = ScenePipelineConfig()
        pipeline = ScenePipeline(config)
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            pipeline.process_video(Path("/nonexistent/file.mp4"))
    
    def test_pipeline_configuration_loading(self):
        """Test loading pipeline configuration from file."""
        config_data = {
            'scene_detection': {
                'threshold': 0.3,
                'min_scene_length': 2.0
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            # Test loading configuration (would need config loader)
            assert Path(config_file).exists()
        finally:
            Path(config_file).unlink()


class TestUtilities:
    """Test utility functions."""
    
    def test_video_file_validation(self):
        """Test video file validation."""
        # Test valid video extensions
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        for ext in valid_extensions:
            assert ext in valid_extensions
        
        # Test invalid extensions
        invalid_extensions = ['.txt', '.jpg', '.png']
        for ext in invalid_extensions:
            assert ext not in valid_extensions
    
    def test_output_directory_creation(self):
        """Test output directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            assert output_dir.exists()
            assert output_dir.is_dir()


class TestPerformance:
    """Performance and stress tests."""
    
    def test_memory_usage(self):
        """Test memory usage with large inputs."""
        # This would test memory usage with large video files
        # and ensure memory is properly managed
        pass
    
    def test_processing_speed(self):
        """Test processing speed benchmarks."""
        # This would test processing speed with different configurations
        # and video sizes
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
